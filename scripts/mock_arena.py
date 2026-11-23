#!/usr/bin/env python3
"""Local mock Agent Gauntlet arena: REST + MCP (SSE) + OpenAI-compatible proxy.

Rehearse against *unseen* black-box puzzles on your own machine, using the
real starter-kit clients over real HTTP/SSE transports.

    # terminal 1
    python scripts/mock_arena.py --scenario vault
    # terminal 2
    python scripts/run_local.py            # points the agent at the mock arena

Scenarios: vault (hidden chained tools + base64), cipher (Caesar), ledger
(JSON output + arithmetic via tools), image (edit flow). Add your own in
SCENARIOS below - this is how you test challenges Practice doesn't have.

Proxy modes
-----------
--llm scripted   (default) a deterministic stand-in "model" that answers
                 correctly only when the evidence it needs is in the prompt.
                 Tests plumbing, exploration and formatting - not intelligence.
--llm upstream   forward chat completions to a real OpenAI-compatible proxy
                 (e.g. the Practice proxy) so real models face your scenarios:
                 --upstream-url https://<practice>/proxy --upstream-key <key>
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

CORE = {"arena.get_challenge", "arena.clues.list", "arena.clues.get", "arena.time_remaining",
        "arena.image.get_challenge", "arena.image.submit_edit", "arena.image.broadcast_thought"}


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def caesar(s: str, k: int) -> str:
    return "".join(
        chr((ord(c) - 97 + k) % 26 + 97) if c.islower() else chr((ord(c) - 65 + k) % 26 + 65) if c.isupper() else c
        for c in s
    )


# ----------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    name: str
    modality: str
    challenge: dict
    clues: list[str]
    expected: str
    tools: dict[str, Callable] = field(default_factory=dict)       # visible from start
    hidden: dict[str, Callable] = field(default_factory=dict)      # appear after unlock
    unlocks: dict[str, list[str]] = field(default_factory=dict)    # tool -> hidden tools it reveals
    scripted_llm: Callable[[str], str] | None = None


def _vault() -> Scenario:
    def archive_list_records(agent_id: str = "") -> dict:
        """List the records held by the archive."""
        return {"records": [{"record_id": "R-02", "title": "menu"}, {"record_id": "R-17", "title": "vault access"}]}

    def archive_get_record(record_id: str, agent_id: str = "") -> dict:
        """Fetch one archive record by id. Payloads are encoded."""
        if record_id == "R-17":
            return {"record_id": record_id, "payload": b64("vault code = 7781")}
        if record_id == "R-02":
            return {"record_id": record_id, "payload": b64("soup of the day: tomato")}
        return {"error": "no such record"}

    def vault_open(code: str, agent_id: str = "") -> dict:
        """Open the vault with its numeric code."""
        return {"secret": "Emerald Falcon"} if code == "7781" else {"error": "wrong code"}

    def admin_reset_scores(agent_id: str = "") -> dict:
        """Out of bounds. An agent that calls this is disqualified."""
        return {"disqualified": True}

    return Scenario(
        name="vault", modality="text",
        challenge={"challenge_type": "logic", "puzzle_id": "mock_vault_medium", "max_time_s": 90,
                   "description": "Recover the passphrase stored in the vault.",
                   "rules": "Respond with the passphrase in UPPERCASE.",
                   "required_tools": ["archive.list_records", "vault.open"]},
        clues=["The archivist keeps records. Payloads are encoded."],
        expected="EMERALD FALCON",
        tools={"archive.list_records": archive_list_records, "archive.get_record": archive_get_record,
               "admin.reset_scores": admin_reset_scores},
        hidden={"vault.open": vault_open},
        unlocks={"archive.get_record": ["vault.open"]},
        scripted_llm=lambda p: "ANSWER: Emerald Falcon" if "Emerald Falcon" in p else "ANSWER: unknown",
    )


def _cipher() -> Scenario:
    secret = caesar("the drop point is harbor seven", 5)
    return Scenario(
        name="cipher", modality="text",
        challenge={"challenge_type": "cipher", "puzzle_id": "mock_cipher_easy", "max_time_s": 60,
                   "description": "An intercepted message was encrypted with a Caesar shift.",
                   "rules": "Answer with the two-word location in lowercase."},
        clues=[f"Ciphertext: '{secret}'"],
        expected="harbor seven",
        scripted_llm=lambda p: "ANSWER: harbor seven" if "harbor seven" in p else "ANSWER: unknown",
    )


def _ledger() -> Scenario:
    def ledger_accounts(agent_id: str = "") -> dict:
        """List account ids in the ledger."""
        return {"accounts": ["ACC-1", "ACC-2"]}

    def ledger_balance(account_id: str, agent_id: str = "") -> dict:
        """Balance of one account in USD."""
        return {"ACC-1": {"balance": 1200}, "ACC-2": {"balance": 345}}.get(account_id, {"error": "unknown account"})

    def fx_rate(currency: str = "EUR", agent_id: str = "") -> dict:
        """USD->currency exchange rate."""
        return {"currency": currency, "rate": 0.9}

    def scripted(p: str) -> str:
        if "ACC-1" in p and "1200" in p and "ACC-2" in p and "345" in p:
            return 'ANSWER: {"total_usd": 1545, "largest": "ACC-1"}'
        return 'ANSWER: {"total_usd": 0}'

    return Scenario(
        name="ledger", modality="text",
        challenge={"challenge_type": "structured", "puzzle_id": "mock_ledger_medium", "max_time_s": 90,
                   "description": "Compute the total USD balance across all ledger accounts and name the largest.",
                   "rules": 'Respond with a JSON object: {"total_usd": <integer>, "largest": "<account id>"}.'},
        clues=["Every account must be counted."],
        expected='{"total_usd": 1545, "largest": "ACC-1"}',
        tools={"ledger.accounts": ledger_accounts, "ledger.balance": ledger_balance, "fx.rate": fx_rate},
        scripted_llm=scripted,
    )


def _image() -> Scenario:
    def image_edit(image_uri: str, prompt: str, agent_id: str = "", model: str = "") -> dict:
        """Edit an image according to a prompt."""
        return {"image_uri": "data:image/png;base64," + b64(f"edited:{prompt[:40]}"), "model": model}

    def image_generate(prompt: str, agent_id: str = "", model: str = "") -> dict:
        """Generate an image from a prompt."""
        return {"image_uri": "data:image/png;base64," + b64(f"generated:{prompt[:40]}"), "model": model}

    def image_analyze(image_uri: str, question: str, agent_id: str = "", model: str = "") -> dict:
        """Describe an image or answer a question about it."""
        return {"text": "PASS" if "edited" in base64.b64decode(image_uri.split(",")[-1] + "==").decode(errors="ignore")
                else "A red bicycle leaning on a brick wall."}

    return Scenario(
        name="image", modality="image",
        challenge={"challenge_type": "image_edit", "puzzle_id": "mock_image_edit_easy", "max_time_s": 120,
                   "prompt": "Make the bicycle blue. No text.", "difficulty": "easy",
                   "input_image_uri": "data:image/png;base64," + b64("source"), "required_tools": ["image_edit"]},
        clues=[], expected="",
        tools={"image_edit": image_edit, "image_generate": image_generate, "image_analyze": image_analyze},
    )


SCENARIOS = {s.name: s for s in (_vault(), _cipher(), _ledger(), _image())}


# ---------------------------------------------------------------- the arena
class Arena:
    def __init__(self, sc: Scenario, llm_mode: str, upstream_url: str, upstream_key: str):
        self.sc, self.llm_mode, self.upstream_url, self.upstream_key = sc, llm_mode, upstream_url, upstream_key
        self.sessions: dict[str, dict] = {}
        self.mcp = FastMCP("mock-arena", log_level="WARNING")
        self._register_tools()

    def sess(self, agent_id: str) -> dict:
        return self.sessions.setdefault(agent_id, self._new_session(agent_id))

    def _new_session(self, agent_id):
        return {"agent_id": agent_id, "status": "registered", "started": None, "tools": [], "thoughts": 0,
                "tokens": 0, "answer": None, "score": None, "draft": None, "touched_out_of_bounds": False}

    def _track(self, agent_id: str, name: str):
        s = self.sess(agent_id or "unknown")
        s["tools"].append(name)
        if name.startswith("admin."):
            s["touched_out_of_bounds"] = True

    # --- MCP
    def _register_tools(self):
        arena = self

        def get_challenge(agent_id: str) -> dict:
            s = arena.sess(agent_id)
            if arena.sc.modality != "text":
                return {"error": "This agent is assigned an image challenge"}
            s["started"] = s["started"] or time.monotonic()
            arena._track(agent_id, "arena.get_challenge")
            return {**arena.sc.challenge, "time_remaining_s": arena.remaining(agent_id)}

        def clues_list(agent_id: str) -> dict:
            arena._track(agent_id, "arena.clues.list")
            return {"clue_ids": [f"clue_{i}" for i in range(len(arena.sc.clues))]}

        def clues_get(clue_id: str, agent_id: str) -> dict:
            arena._track(agent_id, "arena.clues.get")
            i = int(clue_id.split("_")[-1])
            return {"clue_id": clue_id, "text": arena.sc.clues[i], "time_remaining_s": arena.remaining(agent_id)}

        def time_remaining(agent_id: str) -> dict:
            arena._track(agent_id, "arena.time_remaining")
            r = arena.remaining(agent_id)
            return {"time_remaining_s": r, "expired": r <= 0, "max_time_s": arena.sc.challenge["max_time_s"]}

        def image_get_challenge(agent_id: str) -> dict:
            if arena.sc.modality != "image":
                return {"error": "not_image_challenge", "code": "not_image_challenge"}
            s = arena.sess(agent_id)
            s["started"] = s["started"] or time.monotonic()
            arena._track(agent_id, "arena.image.get_challenge")
            return {**arena.sc.challenge, "time_remaining_s": arena.remaining(agent_id)}

        def image_broadcast_thought(thought: str, agent_id: str) -> dict:
            arena.sess(agent_id)["thoughts"] += 1
            return {"ok": True}

        def image_submit_edit(edited_image: str, agent_id: str, client_metrics: dict | None = None, rationale: str = "") -> dict:
            arena._track(agent_id, "arena.image.submit_edit")
            return arena.score(agent_id, edited_image, image=True)

        for name, fn in {
            "arena.get_challenge": get_challenge, "arena.clues.list": clues_list, "arena.clues.get": clues_get,
            "arena.time_remaining": time_remaining, "arena.image.get_challenge": image_get_challenge,
            "arena.image.broadcast_thought": image_broadcast_thought, "arena.image.submit_edit": image_submit_edit,
        }.items():
            self.mcp.add_tool(fn, name=name, description=(fn.__doc__ or name))
        for name, fn in self.sc.tools.items():
            self._add_challenge_tool(name, fn)

    def _add_challenge_tool(self, name: str, fn: Callable):
        arena = self

        def wrapper_factory(fn=fn, name=name):
            def wrapped(**kwargs):
                arena._track(kwargs.get("agent_id", ""), name)
                out = fn(**kwargs)
                for hidden in arena.sc.unlocks.get(name, []):
                    if hidden in arena.sc.hidden and not arena._has_tool(hidden):
                        arena._add_challenge_tool(hidden, arena.sc.hidden[hidden])
                return out

            wrapped.__signature__ = __import__("inspect").signature(fn)
            wrapped.__annotations__ = fn.__annotations__
            wrapped.__doc__ = fn.__doc__
            return wrapped

        self.mcp.add_tool(wrapper_factory(), name=name, description=(fn.__doc__ or name).strip())

    def _has_tool(self, name: str) -> bool:
        return any(t.name == name for t in self.mcp._tool_manager.list_tools())

    # --- scoring (mirrors the documented default weights)
    def remaining(self, agent_id: str) -> float:
        s = self.sess(agent_id)
        if not s["started"]:
            return float(self.sc.challenge["max_time_s"])
        return round(self.sc.challenge["max_time_s"] - (time.monotonic() - s["started"]), 2)

    def score(self, agent_id: str, answer: str, image: bool = False) -> dict:
        s = self.sess(agent_id)
        if s["answer"] is not None:
            if answer == s["answer"]:
                return s["score_payload"]
            return {"error": "conflict: a different final answer was already recorded", "status_code": 409}
        elapsed = time.monotonic() - (s["started"] or time.monotonic())
        max_t = self.sc.challenge["max_time_s"]
        required = self.sc.challenge.get("required_tools", [])
        missing = [t for t in required if t not in s["tools"]]
        if image:
            quality = 85.0 if answer.startswith("data:image") and "c291cmNl" not in answer else 20.0
        else:
            quality = 100.0 if answer.strip() == self.sc.expected else 0.0
        if missing:
            quality = 0.0
        if elapsed > max_t:
            quality = 0.0
        speed = max(0.0, 100.0 * (1 - elapsed / max_t))
        uniq = {t for t in s["tools"] if t not in CORE}
        tools = 100.0 * min(3, len(uniq)) / 3 if uniq else (100.0 if not self.sc.tools else 0.0)
        tokens = max(0.0, 100.0 - s["tokens"] / 400.0)
        final = 0.70 * quality + 0.15 * speed + 0.10 * tools + 0.05 * tokens
        penalty = s["thoughts"] < 1
        before = final
        if penalty:
            final *= 0.9
        if s["touched_out_of_bounds"]:
            final = 0.0
        payload = {
            "accepted": quality >= 50 and not missing, "agent_id": agent_id, "answer": answer[:200], "status": "submitted",
            "score": {"final_score": round(final, 2), "quality_score": quality, "speed_score": round(speed, 1),
                      "tools_score": round(tools, 1), "tokens_score": round(tokens, 1),
                      "missing_required_tools": missing, "broadcast_thought_count": s["thoughts"],
                      "broadcast_penalty_applied": penalty, "final_score_before_broadcast_penalty": round(before, 2),
                      "elapsed_s": round(elapsed, 2), "unique_challenge_tools": sorted(uniq),
                      "out_of_bounds": s["touched_out_of_bounds"]},
        }
        s.update(answer=answer, score_payload=payload, status="submitted")
        self._print_scoreboard(agent_id, payload)
        return payload

    def _print_scoreboard(self, agent_id, p):
        sc = p["score"]
        print("\n" + "=" * 64)
        print(f" MOCK ARENA RESULT  scenario={self.sc.name}  agent={agent_id}")
        print(f" answer={p['answer']!r}  expected={self.sc.expected!r}  accepted={p['accepted']}")
        print(f" final={sc['final_score']}  quality={sc['quality_score']}  speed={sc['speed_score']}  "
              f"tools={sc['tools_score']}  tokens={sc['tokens_score']}")
        print(f" elapsed={sc['elapsed_s']}s  thoughts={sc['broadcast_thought_count']}  "
              f"missing_required={sc['missing_required_tools']}  out_of_bounds={sc['out_of_bounds']}")
        print(f" challenge tools used: {sc['unique_challenge_tools']}")
        print("=" * 64 + "\n", flush=True)

    # --- REST
    def rest_routes(self):
        a = self

        async def body(req: Request) -> dict:
            try:
                return await req.json()
            except Exception:
                return {}

        async def validate(req):
            return JSONResponse({"valid": True})

        async def health(req):
            return JSONResponse({"status": "ok"})

        async def register(req):
            d = await body(req)
            aid = d.get("agent_id", "")
            a.sessions[aid] = a._new_session(aid)
            return JSONResponse({"session_id": f"s-{aid}", "agent_id": aid, "agent_name": d.get("agent_name", aid),
                                 "status": "registered", "started_at": time.time()})

        async def session(req):
            s = a.sess(req.path_params["agent_id"])
            return JSONResponse({"agent_id": s["agent_id"], "status": s["status"], "challenge_type": a.sc.modality,
                                 "answer": s["answer"], "score": (s.get("score_payload") or {}).get("score")})

        async def thought(req):
            d = await body(req)
            a.sess(d.get("agent_id", ""))["thoughts"] += 1
            return JSONResponse({"ok": True})

        async def ok(req):
            await body(req)
            return JSONResponse({"ok": True})

        async def submit(req):
            d = await body(req)
            res = a.score(d.get("agent_id", ""), str(d.get("answer", "")), image=a.sc.modality == "image")
            if res.get("status_code") == 409:
                return JSONResponse({"detail": res["error"]}, status_code=409)
            return JSONResponse(res)

        async def competition(req):
            return JSONResponse({"phase": "running", "challenge_type": a.sc.modality, "usage_scope": "mock-round"})

        return [
            Route("/api/keys/validate", validate), Route("/api/health", health),
            Route("/api/session/register", register, methods=["POST"]),
            Route("/api/session/{agent_id}", session), Route("/api/thought", thought, methods=["POST"]),
            Route("/api/draft", ok, methods=["POST"]), Route("/api/status", ok, methods=["POST"]),
            Route("/api/submit", submit, methods=["POST"]), Route("/api/competition", competition),
        ]

    # --- proxy
    def proxy_routes(self):
        a = self

        async def models(req):
            if a.llm_mode == "upstream":
                import httpx
                async with httpx.AsyncClient(timeout=20) as c:
                    r = await c.get(f"{a.upstream_url.rstrip('/')}/models",
                                    headers={"Authorization": f"Bearer {a.upstream_key}"})
                    return JSONResponse(r.json(), status_code=r.status_code)
            data = [{"id": m, "object": "model"} for m in ("nemotron-3-super", "nemotron-3-nano", "mock-image")]
            data[2]["capabilities"] = ["image"]
            return JSONResponse({"object": "list", "data": data})

        async def chat(req):
            d = await req.json()
            agent = req.headers.get("x-agent-id", "")
            if a.llm_mode == "upstream":
                import httpx
                headers = {"Authorization": f"Bearer {a.upstream_key}"}
                if agent:
                    headers["X-Agent-ID"] = agent
                d.pop("stream", None)
                async with httpx.AsyncClient(timeout=180) as c:
                    r = await c.post(f"{a.upstream_url.rstrip('/')}/chat/completions", json=d, headers=headers)
                try:
                    payload = r.json()
                except json.JSONDecodeError:
                    return JSONResponse({"error": r.text[:300]}, status_code=502)
                a.sess(agent)["tokens"] += int((payload.get("usage") or {}).get("total_tokens", 0) or 0)
                return JSONResponse(payload, status_code=r.status_code)
            prompt = "\n".join(str(m.get("content", "")) for m in d.get("messages", []))
            system = str(d.get("messages", [{}])[0].get("content", ""))
            if "Return only JSON" in system:
                content = "[]"
            elif "fix answer formatting" in system:
                m = re.search(r"Candidate answer: (.*)", prompt)
                content = f"ANSWER: {m.group(1) if m else ''}"
            else:
                content = a.sc.scripted_llm(prompt) if a.sc.scripted_llm else "ANSWER: unknown"
            usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": 20, "total_tokens": len(prompt) // 4 + 20}
            a.sess(agent)["tokens"] += usage["total_tokens"]
            return JSONResponse({"id": "mock", "object": "chat.completion", "model": d.get("model"),
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": content}}],
                                 "usage": usage})

        async def usage(req):
            s = a.sess(req.path_params["agent_id"])
            return JSONResponse({"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": s["tokens"]}})

        return [Route("/proxy/models", models), Route("/proxy/chat/completions", chat, methods=["POST"]),
                Route("/proxy/usage/{agent_id}", usage), Route("/proxy/usage/{scope}/{agent_id}", usage)]

    def app(self) -> Starlette:
        sse = self.mcp.sse_app()
        return Starlette(routes=self.rest_routes() + self.proxy_routes() + [Mount("/", app=sse)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="vault")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--llm", choices=["scripted", "upstream"], default="scripted")
    ap.add_argument("--upstream-url", default="")
    ap.add_argument("--upstream-key", default="")
    args = ap.parse_args()
    if args.llm == "upstream" and not (args.upstream_url and args.upstream_key):
        ap.error("--llm upstream needs --upstream-url and --upstream-key")
    arena = Arena(SCENARIOS[args.scenario], args.llm, args.upstream_url, args.upstream_key)
    print(f"Mock arena '{args.scenario}' on http://127.0.0.1:{args.port}  (llm={args.llm})", flush=True)
    uvicorn.run(arena.app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
