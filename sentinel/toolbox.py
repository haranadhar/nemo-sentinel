"""Schema-aware exploration of an unknown MCP server.

The baseline starter kit only calls a challenge tool when its *name* appears in
the challenge text, and then guesses arguments by trying a dozen payload shapes.
In a Black Box with hidden tools that leaves most of the server unexplored.

NemoSentinel instead:

1. reads every tool's ``inputSchema`` and builds arguments from it, filling
   values from what it already knows (agent id, ids/keys/codes found in clues
   and earlier tool results, the challenge question, enum/default values);
2. explores in bounded *rounds*: each round runs the callable tools
   concurrently, harvests new facts from the results, then re-plans;
3. re-lists tools after every round, so tools that appear only after another
   tool was called ("hidden functions") are discovered;
4. never calls tools that look state-changing or out of bounds (submit,
   register, reset, delete, admin, leaderboard...), and never executes code
   returned by a tool;
5. keeps a complete call log so required-tool coverage can be checked before
   the answer is submitted.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .decoders import decode_candidates

# Tools the runtime drives explicitly; exploration leaves them alone.
CORE_TEXT_TOOLS = {
    "arena.get_challenge",
    "arena.clues.list",
    "arena.clues.get",
    "arena.time_remaining",
}
CORE_IMAGE_TOOLS = {
    "arena.image.get_challenge",
    "arena.image.broadcast_thought",
    "arena.image.submit_edit",
}
IMAGE_OUTPUT_TOOLS = {"image_edit", "image_generate", "image_analyze"}

# Anything that could end, reset, or tamper with a run, or that the rules put
# out of bounds. Matched against name segments, case-insensitive.
_DENY_RE = re.compile(
    r"(?:^|[._\-])(submit|register|reset|restart|delete|remove|drop|destroy|kill|"
    r"shutdown|disconnect|admin|leaderboard|revoke|purge|wipe|terminate|"
    r"set_score|score_override|exec|eval|shell|broadcast)(?:$|[._\-])",
    re.IGNORECASE,
)

# Names that usually describe the server itself; worth calling first.
_DISCOVERY_HINT_RE = re.compile(
    r"(list|discover|describe|help|info|manifest|catalog|index|schema|about|"
    r"hint|readme|guide|explore|status|inventory|map|overview)",
    re.IGNORECASE,
)

_QUERY_PARAM_RE = re.compile(
    r"^(q|query|search|search_query|question|text|prompt|input|message|content|topic|term|keywords?)$",
    re.IGNORECASE,
)
_SECRET_PARAM_RE = re.compile(
    r"(key|token|code|password|passphrase|secret|pin|cipher|nonce|otp|auth)",
    re.IGNORECASE,
)


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    @classmethod
    def from_def(cls, tool_def: Any) -> "ToolSpec":
        if isinstance(tool_def, dict):
            name = str(tool_def.get("name") or "")
            desc = str(tool_def.get("description") or "")
            schema = tool_def.get("inputSchema") or tool_def.get("input_schema") or {}
        else:
            name = str(getattr(tool_def, "name", "") or "")
            desc = str(getattr(tool_def, "description", "") or "")
            schema = getattr(tool_def, "inputSchema", None) or {}
        if not isinstance(schema, dict):
            schema = {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        req = schema.get("required") if isinstance(schema.get("required"), list) else []
        return cls(name=name, description=desc, properties=dict(props), required=[str(r) for r in req])

    @property
    def denied(self) -> bool:
        return bool(_DENY_RE.search(self.name))

    @property
    def is_core(self) -> bool:
        return self.name in CORE_TEXT_TOOLS or self.name in CORE_IMAGE_TOOLS

    def compact(self, max_desc: int = 220) -> str:
        """One-line description for LLM planning prompts."""
        params = []
        for pname, pschema in self.properties.items():
            ptype = pschema.get("type", "any") if isinstance(pschema, dict) else "any"
            star = "*" if pname in self.required else ""
            enum = pschema.get("enum") if isinstance(pschema, dict) else None
            extra = f" one of {enum[:6]}" if isinstance(enum, list) and enum else ""
            params.append(f"{pname}{star}:{ptype}{extra}")
        desc = re.sub(r"\s+", " ", self.description).strip()[:max_desc]
        return f"{self.name}({', '.join(params)}) - {desc}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None
    error: str = ""
    elapsed_s: float = 0.0
    round: int = 0
    reached: bool = True  # False when the server rejected the call before running the tool

    def text(self, limit: int = 1800) -> str:
        body = render_result(self.result) if self.ok else f"ERROR: {self.error}"
        if len(body) > limit:
            body = body[:limit] + "...[truncated]"
        args = {k: v for k, v in self.arguments.items() if k != "agent_id"}
        return f"{self.name}({json.dumps(args, ensure_ascii=False)[:300]}) -> {body}"


# --------------------------------------------------------------------- facts
class Knowledge:
    """Everything learned so far.

    ``facts`` maps a normalised key to an *ordered list* of candidate values
    (most relevant first). Results often list several records with the same
    keys (``record_id`` R-17 and R-02); keeping only one would make the agent
    open the wrong record. Candidates from list items whose other fields share
    words with the question are ranked first, and newer tool results are
    ranked ahead of older ones.
    """

    def __init__(self, agent_id: str, question: str = "") -> None:
        self.agent_id = agent_id
        self.question = question
        self._qwords = _words(question)
        self.facts: dict[str, list[Any]] = {"agent_id": [agent_id]}
        self.texts: list[str] = []
        self.mentioned_tools: set[str] = set()
        self.urls: list[str] = []
        self.numbers: list[float] = []

    # ---------------------------------------------------------------- write
    def _put(self, key: str, value: Any, *, front: bool) -> None:
        bucket = self.facts.setdefault(key, [])
        if value in bucket:
            if front:
                bucket.remove(value)
            else:
                return
        if front:
            bucket.insert(0, value)
        else:
            bucket.append(value)

    def set_fact(self, key: str, value: Any) -> None:
        self._put(_norm(key), value, front=False)

    def add_text(self, text: str, known_tools: set[str] | None = None) -> None:
        if not text:
            return
        self.texts.append(text)
        for k, v in extract_text_facts(text).items():
            self._put(k, v, front=False)
        for url in re.findall(r"https?://[^\s)>\"']+", text):
            u = url.rstrip(".,);")
            if u not in self.urls:
                self.urls.append(u)
        for num in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)[:20]:
            try:
                self.numbers.append(float(num))
            except ValueError:
                pass
        if known_tools:
            lowered = text.lower()
            for t in known_tools:
                if t.lower() in lowered:
                    self.mentioned_tools.add(t)

    def add_result(self, result: Any, known_tools: set[str] | None = None) -> None:
        ranked = ranked_leaves(result, self._qwords)
        # Insert in reverse so the most relevant ends up first, ahead of
        # anything learned from older results.
        for key, value in reversed(ranked):
            self._put(key, value, front=True)
        self.add_text(render_result(result), known_tools)

    # ----------------------------------------------------------------- read
    def candidates(self, param: str, limit: int = 6) -> list[Any]:
        """Candidate values for a parameter: exact key first, then fuzzy."""
        p = _norm(param)
        out: list[Any] = []
        for v in self.facts.get(p, []):
            if v not in out:
                out.append(v)
        fuzzy: list[tuple[int, str]] = []
        for k in self.facts:
            if k == p or k == "agent_id":
                continue
            leaf = k.rsplit(".", 1)[-1]
            if leaf == p or (len(leaf) > 2 and (leaf.endswith(p) or p.endswith(leaf) or leaf in p)):
                fuzzy.append((len(leaf), k))
        for _len, k in sorted(fuzzy, key=lambda t: -t[0]):
            for v in self.facts[k]:
                if v not in out:
                    out.append(v)
        return out[:limit]

    def lookup(self, param: str) -> Any:
        c = self.candidates(param, limit=1)
        return c[0] if c else None

    def first(self, key: str) -> Any:
        vals = self.facts.get(_norm(key)) or []
        return vals[0] if vals else None


def _words(text: str) -> set[str]:
    stop = {"the", "and", "with", "that", "this", "from", "your", "what", "which", "into", "only", "answer", "respond"}
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in stop}


def ranked_leaves(value: Any, qwords: set[str]) -> list[tuple[str, Any]]:
    """(key, value) leaves; items of lists ordered by relevance to the question."""
    out: list[tuple[str, Any, float, int]] = []

    def walk(v: Any, prefix: str, relevance: float, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(v, dict):
            for k, child in v.items():
                key = _norm(k)
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(child, (dict, list)):
                    walk(child, path, relevance, depth + 1)
                elif isinstance(child, (str, int, float, bool)) and child != "" and not (
                    isinstance(child, str) and len(child) > 500
                ):
                    out.append((key, child, relevance, len(out)))
                    if prefix:
                        out.append((path, child, relevance, len(out)))
        elif isinstance(v, list):
            leaf = prefix.rsplit(".", 1)[-1].split("[", 1)[0] if prefix else ""
            if leaf and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v[:50]):
                singular = leaf[:-1] if leaf.endswith("s") and len(leaf) > 3 else leaf
                for x in v[:50]:
                    if x == "" or (isinstance(x, str) and len(x) > 200):
                        continue
                    out.append((singular, x, relevance, len(out)))
                    if singular != leaf:
                        out.append((leaf, x, relevance, len(out)))
                return
            for i, item in enumerate(v[:50]):
                rel = relevance
                if isinstance(item, dict) and qwords:
                    rel += len(_words(render_result(item)) & qwords)
                walk(item, f"{prefix}[{i}]" if prefix else f"[{i}]", rel, depth + 1)

    walk(value, "", 0.0, 0)
    out.sort(key=lambda t: (-t[2], t[3]))
    return [(k, v) for k, v, _r, _o in out]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def flatten_facts(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Scalar leaves of a JSON-like result, keyed by both leaf and dotted path."""
    out: dict[str, Any] = {}
    if depth > 6:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            key = _norm(k)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(v, (dict, list)):
                out.update(flatten_facts(v, path, depth + 1))
            elif isinstance(v, (str, int, float, bool)) and v != "" and not (
                isinstance(v, str) and len(v) > 500
            ):
                out[key] = v
                out[path] = v
    elif isinstance(value, list):
        for i, v in enumerate(value[:50]):
            out.update(flatten_facts(v, f"{prefix}[{i}]" if prefix else f"[{i}]", depth + 1))
    return out


_FACT_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9 _\-]{1,30}?)\s*(?:is|=|:|->)\s*[\"'`]?([A-Za-z0-9][A-Za-z0-9_\-./+=]{2,80})[\"'`]?"
)


def extract_text_facts(text: str) -> dict[str, str]:
    """'vault key: AX-77' -> {'vault_key': 'AX-77', 'key': 'AX-77'} (best effort)."""
    out: dict[str, str] = {}
    for m in _FACT_RE.finditer(text or ""):
        raw_key, val = m.group(1).strip(), m.group(2).strip().rstrip(".,;")
        key = _norm(raw_key)
        if not key or key in {"the", "it", "this", "that", "answer"} or len(val) < 3:
            continue
        # Only keep identifier-like keys; prose produces too much noise otherwise.
        words = key.split("_")
        if len(words) > 3:
            key = "_".join(words[-3:])
        out.setdefault(key, val)
        if len(words) > 1:
            out.setdefault(words[-1], val)
    return out


def render_result(result: Any) -> str:
    """Readable text for a tool result; large binary-ish fields are elided."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (int, float, bool)):
        return str(result)
    if isinstance(result, dict):
        cleaned = {}
        for k, v in result.items():
            if isinstance(v, str) and len(v) > 400 and ("image" in str(k).lower() or v.startswith("data:")):
                cleaned[k] = f"<{len(v)} chars elided>"
            else:
                cleaned[k] = v
        if set(cleaned) == {"raw"}:
            return str(cleaned["raw"]).strip()
        try:
            return json.dumps(cleaned, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(cleaned)
    if isinstance(result, list):
        return "\n".join(render_result(r) for r in result)
    return str(result)


def unwrap(result: Any) -> Any:
    """Undo common MCP envelopes: {"result": {...}}, {"structured": {...}}, JSON in "raw"."""
    for _ in range(3):
        if isinstance(result, dict) and len(result) == 1:
            (key, inner), = result.items()
            if key in {"result", "structured", "data", "content"} and isinstance(inner, (dict, list)):
                result = inner
                continue
            if key == "raw" and isinstance(inner, str) and inner.strip()[:1] in "{[":
                try:
                    result = json.loads(inner)
                    continue
                except json.JSONDecodeError:
                    pass
        break
    return result


def result_error(result: Any) -> str:
    """Return an error string if a tool result represents a failure."""
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            return str(err)
        raw = result.get("raw")
        if isinstance(raw, str) and raw.strip().lower().startswith("error executing tool"):
            return raw.strip()
    return ""


_REJECTED_RE = re.compile(
    r"(validation error|missing required|required argument|field required|unknown tool|tool not found|"
    r"no such tool|invalid arguments?|unexpected keyword)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------- arguments
def build_arguments(
    spec: ToolSpec, knowledge: Knowledge, *, force: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Fill a tool's arguments from its schema. Returns (args, unfilled_required).

    ``force`` fills still-unknown required strings with a neutral value so the
    call passes schema validation (a rejected call does not count as a use of
    the tool, which matters for ``required_tools``).
    """
    args: dict[str, Any] = {}
    missing: list[str] = []
    for pname, pschema in spec.properties.items():
        pschema = pschema if isinstance(pschema, dict) else {}
        value = _value_for(pname, pschema, knowledge, required=pname in spec.required)
        if value is None and force and pname in spec.required:
            value = _neutral(pschema, knowledge)
        if value is not None:
            args[pname] = value
        elif pname in spec.required:
            missing.append(pname)
    # Arena tools are scoped per agent; pass agent_id when no schema is given.
    if not spec.properties and spec.name.startswith("arena."):
        args["agent_id"] = knowledge.agent_id
    return args, missing


def argument_variants(spec: ToolSpec, knowledge: Knowledge, limit: int = 3) -> list[dict[str, Any]]:
    """Base arguments plus variants over alternative candidate values.

    Only the first required parameter that has several known candidates is
    varied, which keeps fan-out small (MCP calls cost time, not tokens).
    """
    base, missing = build_arguments(spec, knowledge)
    if missing:
        return []
    variants = [base]
    for pname in spec.required or list(spec.properties):
        norm = _norm(pname)
        if norm in {"agent_id", "agentid", "agent"} or _QUERY_PARAM_RE.match(norm):
            continue
        pschema = spec.properties.get(pname) if isinstance(spec.properties.get(pname), dict) else {}
        if isinstance(pschema.get("enum"), list):
            continue
        cands = knowledge.candidates(pname)
        if len(cands) > 1:
            for c in cands[1:]:
                v = dict(base)
                v[pname] = _coerce(c, pschema.get("type"))
                if v not in variants:
                    variants.append(v)
                if len(variants) >= limit:
                    break
            break
    return variants[:limit]


def _neutral(pschema: dict[str, Any], kb: Knowledge) -> Any:
    ptype = pschema.get("type")
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if t != "null"), None)
    return {"integer": 0, "number": 0.0, "boolean": False, "array": [], "object": {}}.get(
        ptype, (kb.question or "")[:200]
    )


def _value_for(pname: str, pschema: dict[str, Any], kb: Knowledge, *, required: bool) -> Any:
    ptype = pschema.get("type")
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if t != "null"), None)
    norm = _norm(pname)
    enum = pschema.get("enum")

    if norm in {"agent_id", "agentid", "agent"}:
        return kb.agent_id

    known = kb.lookup(pname)
    if isinstance(enum, list) and enum:
        if known in enum:
            return known
        lowered_q = (kb.question or "").lower()
        for option in enum:
            if str(option).lower() in lowered_q:
                return option
        return pschema.get("default", enum[0])

    if known is not None and norm not in {"query", "q", "text", "question"}:
        return _coerce(known, ptype)

    if _QUERY_PARAM_RE.match(norm):
        return kb.question[:500] if kb.question else ("" if required else None)
    if "url" in norm or "uri" in norm or "link" in norm:
        return kb.urls[0] if kb.urls else pschema.get("default")
    if "default" in pschema:
        return pschema["default"]
    if not required:
        return None
    # Required but unknown: give a type-correct neutral value so discovery
    # tools still answer (many return usage hints on bad input).
    if ptype in {"integer", "number"}:
        minimum = pschema.get("minimum")
        if isinstance(minimum, (int, float)):
            return int(minimum) if ptype == "integer" else minimum
        return 0 if ptype == "integer" else 0.0
    if ptype == "boolean":
        return False
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    if _SECRET_PARAM_RE.search(norm):
        return None  # don't guess secrets; wait until a fact supplies one
    if "id" in norm.split("_") or norm.endswith("id"):
        return None
    return None


def _coerce(value: Any, ptype: Any) -> Any:
    try:
        if ptype == "integer":
            return int(float(value))
        if ptype == "number":
            return float(value)
        if ptype == "boolean":
            return str(value).lower() in {"1", "true", "yes", "on"}
        if ptype == "string":
            return str(value)
        if ptype == "array" and not isinstance(value, list):
            return [value]
    except (TypeError, ValueError):
        return value
    return value


# --------------------------------------------------------------- exploration
CallFn = Callable[[str, dict[str, Any]], Awaitable[Any]]
ListFn = Callable[[], Awaitable[list[Any]]]


class Explorer:
    """Bounded, round-based exploration of challenge tools."""

    def __init__(
        self,
        call: CallFn,
        list_defs: ListFn | None,
        tool_defs: list[Any],
        knowledge: Knowledge,
        *,
        modality: str = "text",
        call_timeout_s: float = 8.0,
        concurrency: int = 4,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._call = call
        self._list_defs = list_defs
        self.kb = knowledge
        self.modality = modality
        self.call_timeout_s = call_timeout_s
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.log = log or (lambda _m: None)
        self.specs: dict[str, ToolSpec] = {}
        self.calls: list[ToolCall] = []
        self.max_variants = 3
        self._attempted: set[tuple[str, str]] = set()
        self.update_specs(tool_defs)

    # ----------------------------------------------------------- inventory
    def update_specs(self, tool_defs: list[Any]) -> list[str]:
        new = []
        for d in tool_defs or []:
            spec = ToolSpec.from_def(d)
            if spec.name and spec.name not in self.specs:
                self.specs[spec.name] = spec
                new.append(spec.name)
        return new

    @property
    def tool_names(self) -> set[str]:
        return set(self.specs)

    def challenge_tools(self) -> list[ToolSpec]:
        """Explorable tools: not core, not denied, not image-output in text mode."""
        out = []
        for s in self.specs.values():
            if s.is_core or s.denied:
                continue
            if self.modality == "text" and s.name in IMAGE_OUTPUT_TOOLS:
                continue
            if self.modality == "image" and s.name in IMAGE_OUTPUT_TOOLS:
                continue  # the image flow drives these itself
            out.append(s)
        return out

    def called_names(self, *, ok_only: bool = False) -> set[str]:
        """Tools actually executed by the server (rejected calls don't count)."""
        return {c.name for c in self.calls if (c.ok if ok_only else c.reached)}

    # ---------------------------------------------------------------- calls
    async def call(self, name: str, arguments: dict[str, Any], *, round_no: int = 0) -> ToolCall:
        loop = asyncio.get_running_loop()
        start = loop.time()
        async with self.sem:
            try:
                result = unwrap(await asyncio.wait_for(self._call(name, arguments), timeout=self.call_timeout_s))
                err = result_error(result)
                rec = ToolCall(name, arguments, ok=not err, result=result, error=err, round=round_no)
            except asyncio.TimeoutError:
                rec = ToolCall(name, arguments, ok=False, error=f"timeout after {self.call_timeout_s:.0f}s", round=round_no)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # tool failures must never crash the agent
                rec = ToolCall(name, arguments, ok=False, error=f"{type(exc).__name__}: {exc}"[:300], round=round_no)
        if not rec.ok and _REJECTED_RE.search(rec.error or ""):
            rec.reached = False
        rec.elapsed_s = round(loop.time() - start, 3)
        self.calls.append(rec)
        if rec.ok:
            self.kb.add_result(rec.result, self.tool_names)
            # Encrypted payloads often carry the *next* tool's arguments
            # (ids, codes, keys). Decode them now so the next round can use them.
            for d in decode_candidates(render_result(rec.result), limit=4):
                self.kb.add_text(d.text, self.tool_names)
        else:
            self.kb.add_text(rec.error, self.tool_names)
        return rec

    def _priority(self, spec: ToolSpec, required: set[str]) -> tuple:
        mentioned = spec.name in self.kb.mentioned_tools
        q = (self.kb.question or "").lower()
        desc_overlap = len(set(re.findall(r"[a-z]{4,}", spec.description.lower())) & set(re.findall(r"[a-z]{4,}", q)))
        return (
            0 if spec.name in required else 1,
            0 if mentioned else 1,
            0 if _DISCOVERY_HINT_RE.search(spec.name) else 1,
            -desc_overlap,
            len(spec.required),
            spec.name,
        )

    def plan_round(self, required: set[str], max_calls: int) -> list[tuple[ToolSpec, dict[str, Any]]]:
        plan = []
        for spec in sorted(self.challenge_tools(), key=lambda s: self._priority(s, required)):
            # Tools whose required args are still unknown wait for a later
            # round (or the planner / final required-tool pass).
            for args in argument_variants(spec, self.kb, limit=self.max_variants):
                sig = (spec.name, json.dumps(args, sort_keys=True, default=str))
                if sig in self._attempted:
                    continue
                self._attempted.add(sig)
                plan.append((spec, args))
                if len(plan) >= max_calls:
                    return plan
        return plan

    async def rediscover(self) -> list[str]:
        if not self._list_defs:
            return []
        try:
            defs = await asyncio.wait_for(self._list_defs(), timeout=min(5.0, self.call_timeout_s))
        except Exception:
            return []
        new = self.update_specs(defs)
        if new:
            self.log(f"New tools appeared: {', '.join(new)}")
        return new

    async def explore(
        self,
        *,
        required: set[str],
        budget_s: float,
        max_rounds: int = 4,
        max_calls_per_round: int = 6,
        max_total_calls: int = 18,
    ) -> list[ToolCall]:
        """Run exploration rounds until nothing new, budget, or call cap."""
        loop = asyncio.get_running_loop()
        stop_at = loop.time() + max(0.0, budget_s)
        for rnd in range(1, max_rounds + 1):
            left = stop_at - loop.time()
            if left <= 0.5 or len(self.calls) >= max_total_calls:
                break
            cap = min(max_calls_per_round, max_total_calls - len(self.calls))
            plan = self.plan_round(required, cap)
            if not plan:
                if not await self.rediscover():
                    break
                plan = self.plan_round(required, cap)
                if not plan:
                    break
            self.log(f"Exploring round {rnd}: {', '.join(s.name for s, _ in plan)}")
            tasks = [self.call(s.name, a, round_no=rnd) for s, a in plan]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=max(0.5, left))
            except asyncio.TimeoutError:
                self.log("Exploration budget reached")
                break
            await self.rediscover()
        return self.calls

    def evidence(self, limit_chars: int = 9000) -> list[str]:
        """Tool-call evidence lines for the solver, successful calls first."""
        ordered = sorted(self.calls, key=lambda c: (not c.ok, c.round))
        lines, used = [], 0
        for c in ordered:
            t = c.text()
            if used + len(t) > limit_chars:
                break
            lines.append(t)
            used += len(t)
        return lines


# -------------------------------------------------------- LLM-proposed calls
def parse_planned_calls(raw: str, specs: dict[str, ToolSpec], agent_id: str, limit: int = 3) -> list[tuple[str, dict]]:
    """Validate an LLM's JSON tool plan against the real schemas.

    Accepts ``[{"tool": name, "arguments": {...}}, ...]`` (optionally inside a
    fenced block or an object with a ``calls`` key). Unknown or denied tools and
    unknown parameters are dropped; agent_id is injected where the schema has it.
    """
    text = re.sub(r"<think>.*?(?:</think>|$)", "", raw or "", flags=re.DOTALL)
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("calls") or data.get("tool_calls") or [data]
    out: list[tuple[str, dict]] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or item.get("name") or "").strip()
        spec = specs.get(name)
        if not spec or spec.denied or spec.is_core:
            continue
        raw_args = item.get("arguments") or item.get("args") or {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        if spec.properties:
            args = {k: v for k, v in raw_args.items() if k in spec.properties}
        else:
            args = dict(raw_args)
        for p in spec.properties:
            if _norm(p) in {"agent_id", "agentid"}:
                args[p] = agent_id
        if any(r not in args for r in spec.required):
            continue
        out.append((name, args))
        if len(out) >= limit:
            break
    return out
