"""Challenge classification and model routing.

The roster differs between Practice and the live event, so nothing here is a
hardcoded model list. Models are scored from their alias (family, size,
tier words) and from an optional preference list you fill in after
benchmarking (``MyStrategy.preferred_text_models``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NON_CHAT = re.compile(r"(embed|rerank|guard|safety|moderat|tts|whisper|asr|speech|clip|-image$|image-|vision-only)", re.I)
_IMAGE = re.compile(r"(image|flux|sdxl|stable-diffusion|dall|imagen|gpt-image|seedream|recraft)", re.I)
_FAST_WORDS = re.compile(r"(nano|mini|flash|small|lite|tiny|haiku|fast|turbo|instant|8b|7b|4b|3b|1b)", re.I)
_STRONG_WORDS = {
    "ultra": 30, "super": 22, "pro": 14, "max": 14, "large": 10, "opus": 18, "sonnet": 12,
    "r1": 18, "reason": 12, "thinking": 12, "oss-120b": 18, "235b": 18, "405b": 20,
    "70b": 10, "72b": 10, "120b": 16, "480b": 18, "671b": 20, "kimi": 14, "deepseek": 12,
    "qwen3": 10, "gpt-5": 18, "gemini-2.5-pro": 18, "gemini-3": 18, "claude": 12, "glm": 8,
}
_HARD_WORDS = re.compile(
    r"\b(prove|constraint|constraints|each|exactly|grid|sequence|deduce|derive|optimi[sz]e|"
    r"permutation|combination|schedule|logic|riddle|cipher|decrypt|multi-?step|nested|chain)\b",
    re.I,
)


@dataclass(frozen=True)
class Assessment:
    level: str          # easy | medium | hard
    reasons: tuple[str, ...]


def assess(
    *,
    difficulty: str = "",
    puzzle_id: str = "",
    max_time_s: float = 0,
    text: str = "",
    n_clues: int = 0,
    n_tools: int = 0,
) -> Assessment:
    reasons = []
    d = f"{difficulty} {puzzle_id}".lower()
    for level in ("hard", "medium", "easy"):
        if level in d:
            return Assessment(level, (f"declared {level}",))
    score = 0
    if max_time_s >= 120:
        score += 2
        reasons.append("long time limit")
    elif max_time_s >= 90:
        score += 1
        reasons.append("medium time limit")
    hard_hits = len(_HARD_WORDS.findall(text or ""))
    if hard_hits >= 3:
        score += 2
        reasons.append(f"{hard_hits} reasoning keywords")
    elif hard_hits:
        score += 1
    if len(text or "") > 1200 or n_clues >= 5:
        score += 1
        reasons.append("long context")
    if n_tools >= 4:
        score += 1
        reasons.append("many challenge tools")
    level = "hard" if score >= 4 else "medium" if score >= 2 else "easy"
    return Assessment(level, tuple(reasons))


def is_image_model(alias: str) -> bool:
    return bool(_IMAGE.search(alias or ""))


def is_chat_model(alias: str) -> bool:
    return bool(alias) and not _NON_CHAT.search(alias) and not is_image_model(alias)


def strength(alias: str) -> int:
    a = (alias or "").lower()
    s = 0
    for word, pts in _STRONG_WORDS.items():
        if word in a:
            s += pts
    m = re.search(r"(\d{2,4})b\b", a)
    if m:
        s += min(25, int(m.group(1)) // 10)
    if _FAST_WORDS.search(a):
        s -= 15
    if "nemotron" in a:
        s += 6  # organiser's recommended family; reasoning on by default
    return s


def rank_text_models(available: list[str], preferred: list[str] | None = None, level: str = "medium") -> list[str]:
    """Order chat-capable models for solving. ``preferred`` (benchmarked) wins."""
    chat = [m for m in available if is_chat_model(m)] or list(available)
    pref = [p for p in (preferred or []) if p in chat]
    rest = [m for m in chat if m not in pref]
    rest.sort(key=lambda m: (-strength(m), chat.index(m)))
    # Even on easy puzzles the strongest model stays first: quality is 70% of
    # the score. Speed is won by switching reasoning off (see llm_params).
    _ = level
    return pref + rest


def fast_model(available: list[str], fallback: str = "") -> str:
    """A quick chat model for planning and format repair."""
    chat = [m for m in available if is_chat_model(m)]
    fast = [m for m in chat if _FAST_WORDS.search(m)]
    if fast:
        fast.sort(key=lambda m: -strength(m))
        return fast[0]
    return fallback or (chat[0] if chat else "")


def rank_image_models(available: list[str], model_rows: list[dict] | None = None, preferred: list[str] | None = None) -> list[str]:
    capable: list[str] = []
    rows = {str(r.get("id") or r.get("name") or ""): r for r in (model_rows or []) if isinstance(r, dict)}
    for m in available:
        caps = rows.get(m, {}).get("capabilities")
        if (isinstance(caps, list) and "image" in caps) or m.endswith("-image") or is_image_model(m):
            capable.append(m)
    # Strongest first (pro/ultra above flash above lite): image quality is
    # scored by a judge and outweighs the few seconds a stronger model costs.
    capable.sort(key=lambda m: (-strength(m), len(_FAST_WORDS.findall(m))))
    pref = [p for p in (preferred or []) if p in capable]
    return pref + [m for m in capable if m not in pref]


def is_nemotron(alias: str) -> bool:
    return "nemotron" in (alias or "").lower()


def llm_params(
    alias: str,
    *,
    level: str,
    usable_s: float,
    base_max_tokens: int = 8192,
    reasoning_on_easy: bool = False,
) -> dict:
    """Token budget and reasoning switch for one solve call.

    Nemotron reasons by default through the proxy and forces max_tokens>=8192.
    Reasoning is worth its latency on medium/hard puzzles; on easy ones, or
    when time is short, we switch it off and cap output. ``reasoning_on_easy``
    keeps it on for easy puzzles too (MyStrategy default: quality first).
    """
    params: dict = {"temperature": 0.0, "max_tokens": base_max_tokens}
    want_reasoning = (level != "easy" or reasoning_on_easy) and usable_s >= 25
    if is_nemotron(alias):
        if want_reasoning:
            params["max_tokens"] = max(8192, base_max_tokens if level == "hard" else 8192)
            if level == "hard" and usable_s >= 60:
                params["max_tokens"] = max(params["max_tokens"], 12288)
        else:
            params["max_tokens"] = 2048
            params["extra_body"] = {"reasoning": {"enabled": False}}
    else:
        params["max_tokens"] = base_max_tokens if want_reasoning else 2048
    return params
