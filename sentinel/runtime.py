"""NemoSentinel challenge flows.

Text flow (per challenge)
-------------------------
 1. fetch challenge, anchor the Deadline, broadcast a first thought at once
 2. read every clue concurrently
 3. explore challenge tools by schema in bounded rounds (+ rediscovery)
 4. optional LLM tool-planner round when exploration left gaps
 5. deterministic decoding of encoded data found anywhere
 6. solve with the routed model under a hard timeout; fallback model on failure
 7. normalise + validate the answer format; cheap repair call if invalid
 8. guarantee every ``required_tools`` entry was called
 9. save draft, submit (exact-same-answer retry on transport failure)

A watchdog wraps steps 2-8: when the usable time is gone, whatever the best
answer is at that moment gets submitted. A late answer is a loss; an
imperfect one still scores.

Image flow mirrors this with analyse -> edit/generate -> self-check ->
(one regeneration if the self-check fails and time allows) -> submit.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import router
from .answer import AnswerFormat, detect_format, extract_answer, finalize, is_invalid, normalize, validate
from .deadline import Deadline
from .decoders import decode_hints
from .llm import LLM, LLMResult
from .toolbox import (
    CORE_TEXT_TOOLS,
    IMAGE_OUTPUT_TOOLS,
    Explorer,
    Knowledge,
    ToolSpec,
    build_arguments,
    parse_planned_calls,
    render_result,
    result_error,
    unwrap,
)

BLANK_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7/"
    "S7sAAAAASUVORK5CYII="
)


# ------------------------------------------------------------------ plumbing
@dataclass
class Ports:
    """Everything the flows talk to. Real clients in production, fakes in tests."""

    http: Any                 # HttpArenaClient-like (sync)
    mcp: Any                  # McpArenaClient-like (async)
    llm: LLM
    agent_id: str
    strategy: Any             # MyStrategy
    models: list[str]
    model_rows: list[dict] = field(default_factory=list)
    log: Callable[[str], None] = print
    sleep: Callable[[float], Any] = asyncio.sleep


@dataclass
class RunReport:
    modality: str
    agent_id: str
    challenge: dict = field(default_factory=dict)
    level: str = ""
    answer: str = ""
    answer_source: str = ""
    format: str = ""
    format_problems: list[str] = field(default_factory=list)
    model: str = ""
    tools_called: list[str] = field(default_factory=list)
    challenge_tools_ok: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    missing_required_tools: list[str] = field(default_factory=list)
    decoded_hints: list[str] = field(default_factory=list)
    thoughts: int = 0
    timings: dict = field(default_factory=dict)
    llm_usage: dict = field(default_factory=dict)
    submit: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class _Thoughts:
    """Best-effort broadcasting; counts successes for the penalty check."""

    def __init__(self, ports: Ports, report: RunReport, *, image: bool = False) -> None:
        self.p, self.r, self.image = ports, report, image

    async def say(self, text: str) -> None:
        text = str(text).strip()[:300]
        if not text:
            return
        self.p.log(f"   💭 {text}")
        ok = False
        try:
            ok = bool(await asyncio.wait_for(asyncio.to_thread(self.p.http.broadcast_thought, self.p.agent_id, text), 4.0))
        except Exception:
            ok = False
        if self.image and hasattr(self.p.mcp, "broadcast_image_thought"):
            try:
                await asyncio.wait_for(self.p.mcp.broadcast_image_thought(text, self.p.agent_id), 4.0)
                ok = True
            except Exception:
                pass
        if ok:
            self.r.thoughts += 1


async def _to_thread(fn, *args, timeout: float = 8.0, **kwargs):
    return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout)


def _required_from(raw: dict) -> list[str]:
    req = raw.get("required_tools") if isinstance(raw, dict) else None
    if isinstance(req, list):
        return [str(t) for t in req if str(t).strip()]
    if isinstance(req, str) and req.strip():
        return [t.strip() for t in req.split(",") if t.strip()]
    return []


# ---------------------------------------------------------------- text flow
class TextRun:
    def __init__(self, ports: Ports, tool_defs: list[Any]) -> None:
        self.p = ports
        self.s = ports.strategy
        self.tool_defs = tool_defs
        self.report = RunReport(modality="text", agent_id=ports.agent_id)
        self.thoughts = _Thoughts(ports, self.report)
        self.best: str = ""
        self.best_source: str = ""
        self.fmt: AnswerFormat = AnswerFormat()
        self.core_called: set[str] = set()
        self.explorer: Explorer | None = None
        self.deadline: Deadline | None = None
        self.challenge = None
        self.clues: list[str] = []
        self.model_used = ""

    # ---- candidate tracking -------------------------------------------
    def offer(self, answer: str, source: str, *, force: bool = False) -> bool:
        ans = normalize(answer, self.fmt)
        if is_invalid(ans):
            return False
        problems = validate(ans, self.fmt)
        best_problems = validate(self.best, self.fmt) if self.best else ["none"]
        if force or not self.best or (not problems and best_problems):
            self.best, self.best_source = ans, source
            return True
        return False

    # ---- phases --------------------------------------------------------
    async def fetch_challenge(self) -> None:
        while True:
            try:
                ch = await self.p.mcp.get_challenge(self.p.agent_id)
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "locked" in msg or "waiting for organizer" in msg or "lobby" in msg:
                    self.p.log("   Lobby open; waiting for GO...")
                    await self.p.sleep(2.0)
                    continue
                raise
        self.challenge = ch
        self.core_called.add("arena.get_challenge")
        raw = dict(getattr(ch, "raw_data", {}) or {})
        self.deadline = Deadline.from_challenge(getattr(ch, "max_time_s", 0), getattr(ch, "time_remaining_s", 0))
        self.report.challenge = {
            "type": ch.challenge_type,
            "puzzle_id": getattr(ch, "puzzle_id", ""),
            "max_time_s": ch.max_time_s,
            "description": (ch.description or "")[:500],
        }
        self.report.required_tools = _required_from(raw)
        self.fmt = detect_format(ch.rules or "", ch.description or "")
        self.report.format = self.fmt.describe()

    async def read_clues(self) -> None:
        ch = self.challenge
        texts: list[str] = []
        try:
            ids = await asyncio.wait_for(self.p.mcp.list_clues(self.p.agent_id), 8.0)
            self.core_called.add("arena.clues.list")
        except Exception as exc:
            ids = []
            self.report.notes.append(f"clues.list failed: {exc}")
        if ids:
            sem = asyncio.Semaphore(4)

            async def one(cid):
                async with sem:
                    try:
                        c = await asyncio.wait_for(self.p.mcp.get_clue(cid, self.p.agent_id), 8.0)
                        self.core_called.add("arena.clues.get")
                        return c.text
                    except Exception:
                        return ""

            texts = [t for t in await asyncio.gather(*(one(i) for i in ids)) if t and t.strip()]
        if not texts:
            texts = [str(c) for c in (ch.clues or []) if str(c).strip()]
        self.clues = texts

    def _context(self, **overrides):
        from base_strategy import ChallengeContext

        ch = self.challenge
        ctx = ChallengeContext(
            challenge_type=ch.challenge_type or "text",
            difficulty=self.report.level or "unknown",
            challenge_text=ch.description or "",
            description=ch.description or "",
            rules=ch.rules or "",
            clues=list(self.clues),
            time_remaining_s=self.deadline.remaining() if self.deadline else 0.0,
            max_time_s=int(ch.max_time_s or 0),
            available_models=list(self.p.models),
            tools_used=sorted(self.all_called()),
            tokens_used=self.p.llm.usage.total_tokens,
            required_tools=list(self.report.required_tools),
        )
        for k, v in overrides.items():
            setattr(ctx, k, v)
        return ctx

    def all_called(self) -> set[str]:
        called = set(self.core_called)
        if self.explorer:
            called |= self.explorer.called_names()
        return called

    async def explore(self) -> None:
        ch = self.challenge
        question = f"{ch.description}\n{ch.rules}".strip()
        kb = Knowledge(self.p.agent_id, question)
        known = {ToolSpec.from_def(d).name for d in self.tool_defs}
        for t in [ch.description, ch.rules, getattr(ch, "extra_text", ""), *self.clues]:
            kb.add_text(str(t or ""), known)
        for k in ("challenge_id", "puzzle_id"):
            v = getattr(ch, k, "")
            if v:
                kb.set_fact(k, v)
        self.explorer = Explorer(
            self.p.mcp.call_tool,
            getattr(self.p.mcp, "list_tool_defs", None),
            self.tool_defs,
            kb,
            modality="text",
            call_timeout_s=min(10.0, max(3.0, self.deadline.total_s * 0.1)),
            log=self.p.log,
        )
        targets = self.explorer.challenge_tools()
        required = set(self.report.required_tools)
        if not targets:
            return
        await self.thoughts.say(f"Exploring {len(targets)} challenge tool(s) by schema")
        budget = self.deadline.budget(getattr(self.s, "explore_fraction", 0.30), cap_s=30.0)
        await self.explorer.explore(required=required, budget_s=budget)
        await self.llm_tool_planner()

    def _planner_needed(self) -> bool:
        ex = self.explorer
        if not ex or not getattr(self.s, "use_llm_tool_planner", True):
            return False
        if self.deadline.usable() < 25:
            return False
        tools = ex.challenge_tools()
        if not tools:
            return False
        called = ex.called_names(ok_only=True)
        uncalled = [t for t in tools if t.name not in called]
        errored = any(not c.ok for c in ex.calls)
        mentioned_uncalled = ex.kb.mentioned_tools - called - CORE_TEXT_TOOLS
        return bool(uncalled or errored or mentioned_uncalled)

    async def llm_tool_planner(self) -> None:
        ex = self.explorer
        rounds = int(getattr(self.s, "planner_rounds", 2))
        for _ in range(max(0, rounds)):
            if not self._planner_needed():
                return
            tools = ex.challenge_tools()
            catalog = "\n".join(f"- {t.compact()}" for t in tools[:25])
            evidence = "\n".join(f"- {line}" for line in ex.evidence(limit_chars=5000)) or "- (none yet)"
            hints = "\n".join(f"- {h}" for h in decode_hints(ex.kb.texts[-30:], limit=6))
            ch = self.challenge
            prompt = (
                "You are exploring an unknown MCP server to solve a challenge.\n"
                f"Challenge: {ch.description}\nRules: {ch.rules}\n"
                f"Clues:\n" + "\n".join(f"- {c}" for c in self.clues[:12]) + "\n\n"
                f"Available tools (name(param*:type) - description; * = required):\n{catalog}\n\n"
                f"Calls made so far and their results:\n{evidence}\n"
                + (f"\nDecoded data:\n{hints}\n" if hints else "")
                + "\nPropose at most 3 NEW tool calls that would reveal information still needed to "
                "answer. Use exact parameter values found in the results/clues (ids, keys, codes). "
                "Do not repeat a call with identical arguments. If nothing more is needed, return [].\n"
                'Reply with JSON only: [{"tool": "<name>", "arguments": {...}}]'
            )
            planner = router.fast_model(self.p.models, fallback=self.model_for_solve())
            params = router.llm_params(planner, level="easy", usable_s=0)
            res = await self.p.llm.complete(
                planner,
                [{"role": "system", "content": "Return only JSON. No prose."}, {"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
                extra_body=params.get("extra_body"),
                timeout_s=min(15.0, self.deadline.budget(0.15)),
            )
            calls = parse_planned_calls(res.content or res.reasoning, ex.specs, self.p.agent_id)
            attempted = {(c.name, json.dumps(c.arguments, sort_keys=True, default=str)) for c in ex.calls}
            calls = [(n, a) for n, a in calls if (n, json.dumps(a, sort_keys=True, default=str)) not in attempted]
            if not calls:
                return
            await self.thoughts.say("Planner follow-up: " + ", ".join(n for n, _ in calls))
            await asyncio.gather(*(ex.call(n, a, round_no=90) for n, a in calls))
            await ex.rediscover()

    def model_for_solve(self) -> str:
        ranked = self.ranked_models()
        return ranked[0] if ranked else ""

    def ranked_models(self) -> list[str]:
        ctx = self._context()
        try:
            hook_ranked = list(self.s.rank_models(ctx, list(self.p.models)))
        except Exception:
            hook_ranked = []
        return hook_ranked or router.rank_text_models(self.p.models, level=self.report.level)

    def build_prompt(self, decoded: list[str]) -> tuple[str, str]:
        ctx = self._context()
        system = self.s.build_system_prompt(ctx)
        user = self.s.build_solver_prompt(ctx)
        extras = []
        extras.append(f"Required answer format: {self.fmt.describe()}.")
        if self.fmt.raw_hint:
            extras.append(f"Format rule from the challenge: {self.fmt.raw_hint}")
        if self.explorer and self.explorer.calls:
            ev = "\n".join(f"- {line}" for line in self.explorer.evidence(limit_chars=7000))
            extras.append(f"Tool evidence (results of MCP tool calls):\n{ev}")
        if decoded:
            extras.append("Deterministically decoded data (exact, computed by code):\n" + "\n".join(f"- {d}" for d in decoded))
        extras.append(
            "Use the tool evidence and decoded data as ground truth. Return the line\n"
            "ANSWER: <final answer>\nfirst, and nothing that breaks the required format."
        )
        return system, user + "\n\n" + "\n\n".join(extras)

    async def solve(self) -> None:
        texts = [self.challenge.description or "", self.challenge.rules or "", *self.clues]
        if self.explorer:
            texts += [render_result(c.result) for c in self.explorer.calls if c.ok]
        decoded = decode_hints(texts, limit=8)
        self.report.decoded_hints = decoded
        if decoded:
            await self.thoughts.say(f"Decoded {len(decoded)} encoded value(s) deterministically")

        system, user = self.build_prompt(decoded)
        ranked = self.ranked_models()
        if not ranked:
            self.report.notes.append("no chat models on roster")
            return
        try:
            hook_pick = self.s.pick_model("solve", ranked, self._context())
        except Exception:
            hook_pick = ""
        primary = hook_pick if hook_pick in ranked else ranked[0]
        order = [primary] + [m for m in ranked if m != primary]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        await self.thoughts.say(f"Solving ({self.report.level}) with {primary}")
        attempt = 0
        for model in order[: int(getattr(self.s, "max_solve_attempts", 3))]:
            # Keep enough back for validation + submit.
            usable = self.deadline.usable()
            if usable < 4:
                break
            first = attempt == 0
            level = self.report.level if first else "easy"  # fallbacks: reasoning off, fast
            params = router.llm_params(
                model,
                level=level,
                usable_s=usable,
                base_max_tokens=int(getattr(self.s, "text_max_tokens", 8192)),
                reasoning_on_easy=first and bool(getattr(self.s, "reasoning_on_easy", False)),
            )
            try:
                hook_params = self.s.get_llm_params(self._context())
                params["temperature"] = float(hook_params.get("temperature", params["temperature"]))
            except Exception:
                pass
            # First attempt gets 75% of the usable time so a fallback model
            # still fits; fallbacks may use nearly all that is left.
            timeout = max(3.0, usable * 0.75) if first else max(3.0, usable - 2.0)
            res = await self.p.llm.complete(
                model, messages, max_tokens=params["max_tokens"], temperature=params["temperature"],
                extra_body=params.get("extra_body"), timeout_s=timeout,
            )
            attempt += 1
            ans = extract_answer(res.content, self.fmt) or (extract_answer(res.reasoning, self.fmt) if not res.content.strip() else "")
            if ans and self.offer(ans, f"llm:{model}", force=not self.best):
                self.model_used = model
                await self._draft(self.best, f"candidate from {model}")
                if not validate(self.best, self.fmt):
                    return
            reason = res.error or (f"finish={res.finish_reason}" if not ans else "format problems")
            self.p.log(f"   ⚠️  {model}: {reason}")
            if self.best:
                return  # have a candidate; repair handles format issues

    async def repair_format(self) -> None:
        if not self.best:
            return
        problems = validate(self.best, self.fmt)
        self.report.format_problems = problems
        if not problems or self.deadline.usable() < 6:
            return
        model = router.fast_model(self.p.models, fallback=self.model_used or self.model_for_solve())
        params = router.llm_params(model, level="easy", usable_s=0)
        res = await self.p.llm.complete(
            model,
            [
                {"role": "system", "content": "You fix answer formatting. Output exactly one line: ANSWER: <answer>"},
                {
                    "role": "user",
                    "content": (
                        f"Challenge rules: {self.challenge.rules}\nRequired format: {self.fmt.describe()}\n"
                        f"Candidate answer: {self.best}\nProblems: {', '.join(problems)}\n"
                        "Rewrite the candidate so it satisfies the format without changing its meaning."
                    ),
                },
            ],
            max_tokens=300,
            extra_body=params.get("extra_body"),
            timeout_s=min(10.0, self.deadline.usable() - 3),
        )
        fixed, fixed_problems = finalize(res.content, self.fmt)
        if fixed and len(fixed_problems) < len(problems):
            self.best, self.best_source = fixed, self.best_source + "+repair"
        self.report.format_problems = validate(self.best, self.fmt)

    async def ensure_required_tools(self) -> None:
        required = [t for t in self.report.required_tools if t not in self.all_called()]
        if not required:
            return
        ex = self.explorer
        for name in required:
            if self.deadline.remaining() < 2:
                break
            if name == "arena.time_remaining":
                await self._time_remaining()
                continue
            if name == "arena.clues.list":
                try:
                    await self.p.mcp.list_clues(self.p.agent_id)
                    self.core_called.add(name)
                except Exception:
                    pass
                continue
            spec = ex.specs.get(name) if ex else None
            if ex and spec:
                args, _missing = build_arguments(spec, ex.kb, force=True)
                await ex.call(name, args, round_no=99)
            else:
                try:
                    await asyncio.wait_for(self.p.mcp.call_tool(name, {"agent_id": self.p.agent_id}), 6.0)
                    self.core_called.add(name)
                except Exception as exc:
                    self.report.notes.append(f"required tool {name} failed: {exc}")
        self.report.missing_required_tools = [t for t in self.report.required_tools if t not in self.all_called()]

    async def _time_remaining(self) -> None:
        try:
            info = await asyncio.wait_for(self.p.mcp.time_remaining(self.p.agent_id), 4.0)
            self.core_called.add("arena.time_remaining")
            self.deadline.resync(info.get("time_remaining_s"))
        except Exception:
            pass

    async def _draft(self, answer: str, reasoning: str) -> None:
        try:
            await _to_thread(self.p.http.save_draft, self.p.agent_id, answer, reasoning, timeout=4.0)
        except Exception:
            pass

    def _fallback_answer(self) -> str:
        """Something better than nothing when no model produced an answer."""
        kb = self.explorer.kb if self.explorer else None
        if kb:
            for key in ("answer", "solution", "flag", "result", "final_answer", "secret", "password", "code"):
                v = kb.first(key)
                if v not in (None, ""):
                    return str(v)
        if self.report.decoded_hints:
            h = self.report.decoded_hints[0]
            return h.split("-> ", 1)[-1].strip().strip("'\"")
        return {"integer": "0", "number": "0", "yesno": "no", "json": "{}"}.get(
            self.fmt.kind, self.fmt.choices[0] if self.fmt.choices else "no answer"
        )

    async def pipeline(self) -> None:
        t = time.monotonic()
        await self.read_clues()
        n_tools = 0
        self.report.level = router.assess(
            difficulty=str((getattr(self.challenge, "raw_data", {}) or {}).get("difficulty", "")),
            puzzle_id=str(getattr(self.challenge, "puzzle_id", "")),
            max_time_s=float(self.challenge.max_time_s or 0),
            text=f"{self.challenge.description}\n{self.challenge.rules}\n" + "\n".join(self.clues),
            n_clues=len(self.clues),
            n_tools=n_tools,
        ).level
        self.report.timings["clues_s"] = round(time.monotonic() - t, 2)
        t = time.monotonic()
        await self.explore()
        self.report.timings["explore_s"] = round(time.monotonic() - t, 2)
        t = time.monotonic()
        await self.solve()
        await self.repair_format()
        self.report.timings["solve_s"] = round(time.monotonic() - t, 2)

    async def run(self) -> RunReport:
        await self.fetch_challenge()
        await self.thoughts.say(f"NemoSentinel engaged: {self.challenge.challenge_type} challenge")
        try:
            await _to_thread(self.p.http.update_status, self.p.agent_id, "running", {"model_name": ""}, timeout=4.0)
        except Exception:
            pass

        # Watchdog: the pipeline gets the usable time; then we submit regardless.
        try:
            await asyncio.wait_for(self.pipeline(), timeout=max(1.0, self.deadline.usable()))
        except asyncio.TimeoutError:
            self.report.notes.append("watchdog fired: submitting best answer so far")
            self.p.log("   ⏰ Watchdog: submitting best answer so far")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never die without submitting
            self.report.notes.append(f"pipeline error: {type(exc).__name__}: {exc}")
            self.p.log(f"   ❌ pipeline error: {exc}")

        try:
            await asyncio.wait_for(self.ensure_required_tools(), timeout=max(1.0, self.deadline.remaining() - 3))
        except Exception:
            pass

        if not self.best:
            self.offer(self._fallback_answer(), "fallback", force=True)
            if not self.best:
                self.best, self.best_source = "no answer", "fallback"

        # Strategy hooks (kept for compatibility with the starter kit).
        ctx = self._context()
        try:
            self.s.should_submit_early(self.best, ctx)
            revised = self.s.on_time_warning(self.deadline.remaining(), self.best, ctx)
            if isinstance(revised, str) and revised.strip() and revised.strip() != self.best:
                self.offer(revised.strip(), "on_time_warning", force=True)
        except Exception:
            pass

        if self.report.thoughts < 2:
            await self.thoughts.say(f"Answer ready: {self.best[:120]}")
        await self._draft(self.best, f"final ({self.best_source})")
        await self.submit()
        return self.finish_report()

    async def submit(self) -> None:
        metrics = {
            "model_name": self.model_used or (self.p.llm.usage.models[0] if self.p.llm.usage.models else ""),
            "total_tokens": str(self.p.llm.usage.total_tokens),
            "prompt_tokens": str(self.p.llm.usage.prompt_tokens),
            "completion_tokens": str(self.p.llm.usage.completion_tokens),
            "total_time_ms": int(self.deadline.elapsed() * 1000),
        }
        self.report.submit = await submit_with_retry(
            lambda: self.p.http.submit(agent_id=self.p.agent_id, answer=self.best, client_metrics=metrics),
            deadline=self.deadline,
            log=self.p.log,
        )

    def finish_report(self) -> RunReport:
        r = self.report
        r.answer, r.answer_source, r.model = self.best, self.best_source, self.model_used
        r.tools_called = sorted(self.all_called())
        if self.explorer:
            r.challenge_tools_ok = sorted(
                {c.name for c in self.explorer.calls if c.ok} - CORE_TEXT_TOOLS
            )
        r.timings["total_s"] = round(self.deadline.elapsed(), 2) if self.deadline else 0
        u = self.p.llm.usage
        r.llm_usage = {"calls": u.calls, "total_tokens": u.total_tokens, "models": u.models}
        return r


async def submit_with_retry(do_submit: Callable[[], Any], *, deadline: Deadline, log=print, attempts: int = 3) -> dict:
    """Submit; on a transport failure retry the *exact same* answer.

    The server returns the original receipt for an identical retry, and 409 for
    a different answer, so retrying the same payload is always safe.
    """
    last_err = ""
    for i in range(attempts):
        try:
            res = await asyncio.wait_for(asyncio.to_thread(do_submit), timeout=max(5.0, min(20.0, deadline.remaining() + 5)))
            return _submit_dict(res)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            last_err = f"{type(exc).__name__}: {exc}"
            log(f"   ⚠️  submit attempt {i + 1} failed: {last_err}")
            if status in (400, 401, 403, 409, 422):
                break  # definitive answer from the server; retrying won't help
            await asyncio.sleep(0.5 * (i + 1))
    return {"error": last_err}


def _submit_dict(res: Any) -> dict:
    if isinstance(res, dict):
        return res
    out = {}
    for k in ("accepted", "status", "score", "answer"):
        if hasattr(res, k):
            out[k] = getattr(res, k)
    return out


# --------------------------------------------------------------- image flow
class ImageRun:
    def __init__(self, ports: Ports, tool_defs: list[Any]) -> None:
        self.p = ports
        self.s = ports.strategy
        self.tool_defs = tool_defs
        self.specs = {ToolSpec.from_def(d).name: ToolSpec.from_def(d) for d in tool_defs}
        self.report = RunReport(modality="image", agent_id=ports.agent_id)
        self.thoughts = _Thoughts(ports, self.report, image=True)
        self.called: set[str] = set()
        self.raw: dict = {}
        self.deadline: Deadline | None = None

    async def fetch(self) -> None:
        while True:
            data = unwrap(await self.p.mcp.call_tool("arena.image.get_challenge", {"agent_id": self.p.agent_id}))
            data = data if isinstance(data, dict) else {}
            err = result_error(data)
            if not err:
                break
            low = err.lower()
            if "locked" in low or "waiting for organizer" in low or "lobby" in low:
                await self.p.sleep(2.0)
                continue
            raise RuntimeError(f"image challenge unavailable: {err}")
        self.called.add("arena.image.get_challenge")
        prompt = data.get("prompt") or data.get("edit_prompt") or ""
        self.raw = dict(data, prompt=prompt)
        self.deadline = Deadline.from_challenge(data.get("max_time_s"), data.get("time_remaining_s"))
        self.report.required_tools = _required_from(data)
        self.report.challenge = {
            "type": data.get("challenge_type", ""),
            "puzzle_id": data.get("puzzle_id", ""),
            "max_time_s": data.get("max_time_s"),
            "prompt": str(prompt)[:400],
        }

    def _ctx(self):
        from base_strategy import ChallengeContext

        d = self.raw
        rules = "\n".join(str(x) for x in (d.get("prompt"), d.get("reference_notes")) if x)
        return ChallengeContext(
            challenge_type=str(d.get("challenge_type") or "image"),
            difficulty=str(d.get("difficulty") or "unknown"),
            challenge_text=str(d.get("description") or ""),
            description=str(d.get("description") or ""),
            rules=rules,
            time_remaining_s=self.deadline.remaining(),
            max_time_s=int(d.get("max_time_s") or 0),
            available_models=list(self.p.models),
            required_tools=list(self.report.required_tools),
            image_url=d.get("input_image_uri") or None,
        )

    def _tool_args(self, name: str, base: dict, model: str) -> dict:
        args = dict(base)
        spec = self.specs.get(name)
        if spec and "model" in spec.properties and model:
            args["model"] = model
        if spec and "agent_id" not in spec.properties and spec.properties:
            args.pop("agent_id", None)
        return args

    async def _call(self, name: str, args: dict, timeout: float) -> dict:
        try:
            res = unwrap(await asyncio.wait_for(self.p.mcp.call_tool(name, args), timeout=max(2.0, timeout)))
            self.called.add(name)
            return res if isinstance(res, dict) else {}
        except Exception as exc:
            self.called.add(name)
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def run(self) -> RunReport:
        await self.fetch()
        await self.thoughts.say("NemoSentinel engaged: image challenge")
        ctx = self._ctx()
        input_uri = ctx.image_url or ""
        tools = [t for t in ("image_edit", "image_generate", "image_analyze") if t in self.specs]
        try:
            choice = self.s.plan_image_tool(ctx, tools)
        except Exception:
            choice = ""
        if choice == "image_analyze" or choice not in tools:
            choice = "image_edit" if input_uri and "image_edit" in tools else ("image_generate" if "image_generate" in tools else "")
        prompt = (self.s.build_image_prompt(ctx) or ctx.rules or ctx.description).strip()
        models = router.rank_image_models(self.p.models, self.p.model_rows, getattr(self.s, "preferred_image_models", None))
        try:
            hook_ranked = [m for m in self.s.rank_models(ctx, list(self.p.models)) if m in models]
            if hook_ranked:
                models = hook_ranked + [m for m in models if m not in hook_ranked]
        except Exception:
            pass
        models = models or [""]
        self.report.model = models[0]
        analyze_ok = "image_analyze" in tools and getattr(self.s, "image_self_check", True)

        # 1) Analyse the source before editing (preservation), when time allows.
        if choice == "image_edit" and input_uri and analyze_ok and self.deadline.usable() > max(40.0, self.deadline.total_s * 0.5):
            a = await self._call(
                "image_analyze",
                self._tool_args("image_analyze", {"image_uri": input_uri, "question": "Describe the key subjects, layout, colours and style precisely.", "agent_id": self.p.agent_id}, models[0]),
                timeout=self.deadline.budget(0.2, cap_s=25),
            )
            desc = str(a.get("text") or "").strip()
            if desc:
                prompt += f"\n\nPreserve everything else from the source image: {desc[:800]}"
                await self.thoughts.say(f"Source analysed: {desc[:140]}")

        # 2) Produce, trying models in order while time allows.
        out_uri, used_model = "", ""
        for model in models[:3]:
            if not choice or self.deadline.usable() < 8:
                break
            base = {"prompt": prompt, "agent_id": self.p.agent_id}
            if choice == "image_edit":
                base["image_uri"] = input_uri
            await self.thoughts.say(f"{choice} with {model or 'default model'}")
            res = await self._call(choice, self._tool_args(choice, base, model), timeout=self.deadline.usable() - 4)
            uri = str(res.get("image_uri") or "").strip()
            if uri and not res.get("error"):
                out_uri, used_model = uri, model
                break
            self.report.notes.append(f"{choice}/{model}: {res.get('error') or 'no image'}")

        # 3) Self-check against the constraints; regenerate once if it fails.
        if out_uri and analyze_ok and self.deadline.usable() > 35 and getattr(self.s, "image_self_check", True):
            check = await self._call(
                "image_analyze",
                self._tool_args(
                    "image_analyze",
                    {
                        "image_uri": out_uri,
                        "question": (
                            "Does this image satisfy ALL of these requirements? "
                            f"Requirements: {ctx.rules[:900]}\n"
                            "Reply 'PASS' or 'FAIL: <violations>'."
                        ),
                        "agent_id": self.p.agent_id,
                    },
                    used_model,
                ),
                timeout=self.deadline.budget(0.25, cap_s=25),
            )
            verdict = str(check.get("text") or "").strip()
            if verdict.upper().startswith("FAIL") and self.deadline.usable() > 25:
                await self.thoughts.say(f"Self-check failed, regenerating: {verdict[:120]}")
                base = {"prompt": f"{prompt}\n\nFix these problems: {verdict[5:400]}", "agent_id": self.p.agent_id}
                if choice == "image_edit":
                    base["image_uri"] = input_uri
                res = await self._call(choice, self._tool_args(choice, base, used_model), timeout=self.deadline.usable() - 4)
                uri = str(res.get("image_uri") or "").strip()
                if uri and not res.get("error"):
                    out_uri = uri
                    self.report.notes.append("regenerated after self-check")

        # 4) Required tools.
        for name in self.report.required_tools:
            if name in self.called or name in {"arena.image.submit_edit"}:
                continue
            if self.deadline.remaining() < 3:
                break
            spec = self.specs.get(name)
            if spec and not spec.denied:
                kb = Knowledge(self.p.agent_id, ctx.rules)
                args, _ = build_arguments(spec, kb, force=True)
                if "image_uri" in spec.properties:
                    args["image_uri"] = out_uri or input_uri
                await self._call(name, args, timeout=6)

        if not out_uri:
            out_uri = input_uri or BLANK_PNG_DATA_URI
            self.report.notes.append("no generated image; submitted fallback")
        self.report.answer = f"<image {len(out_uri)} chars>"
        self.report.model = used_model or self.report.model
        metrics = {"model_name": self.report.model, "planner_tool": choice, "total_time_ms": int(self.deadline.elapsed() * 1000)}

        async def mcp_submit():
            return await self.p.mcp.submit_image(agent_id=self.p.agent_id, image_uri=out_uri, client_metrics=metrics, rationale="NemoSentinel image flow")

        try:
            res = await asyncio.wait_for(mcp_submit(), timeout=max(5.0, self.deadline.remaining() + 5))
            self.called.add("arena.image.submit_edit")
            self.report.submit = res if isinstance(res, dict) else {"result": str(res)}
        except Exception as exc:
            self.p.log(f"   MCP image submit failed ({exc}); trying REST")
            self.report.submit = await submit_with_retry(
                lambda: self.p.http.submit(self.p.agent_id, answer=out_uri, client_metrics=metrics, challenge_type="image"),
                deadline=self.deadline,
                log=self.p.log,
            )
        self.report.tools_called = sorted(self.called)
        self.report.missing_required_tools = [t for t in self.report.required_tools if t not in self.called]
        self.report.timings["total_s"] = round(self.deadline.elapsed(), 2)
        self.report.output_image_uri = out_uri  # type: ignore[attr-defined]
        return self.report


# ---------------------------------------------------------------- reporting
def save_report(report: RunReport, repo_root: Path) -> Path:
    """Append-only run history: outputs/<agent>/<timestamp>-<modality>.json."""
    out_dir = repo_root / "outputs" / report.agent_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}-{report.modality}.json"
    data = report.to_dict()
    data.pop("output_image_uri", None)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
