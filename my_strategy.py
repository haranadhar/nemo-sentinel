"""NemoSentinel competitor strategy.

This file is shared by every runtime in the kit: the bundled Python Simple
example *and* the NemoSentinel runtime (examples/sentinel/agent.py) both
import it, so prompt and model-routing changes here apply to both.

Tune the values in the "BENCHMARK-DRIVEN SETTINGS" block after running
``python scripts/practice_bench.py`` against Practice.
"""

from __future__ import annotations

from base_strategy import _IMAGE_SIZE_HINT, BaseStrategy, ChallengeContext

try:  # optional, so the stock examples still package without sentinel/
    from sentinel import router
except ImportError:  # pragma: no cover - exercised only in a stock package
    router = None


class MyStrategy(BaseStrategy):
    """Adaptive, evidence-first strategy."""

    # Team identity. AGENT_ID in .env overrides agent_id.
    agent_id = "nemo-sentinel"
    agent_name = "NemoSentinel"

    # ------------------------------------------------------------------
    # BENCHMARK-DRIVEN SETTINGS  (edit after practice_bench.py runs)
    # ------------------------------------------------------------------
    # Text models to try first, best first. Only aliases present on the live
    # roster are used; anything else is ignored, so this is safe to leave
    # populated with Practice names.
    preferred_text_models: list[str] = ["nemotron-3-super"]
    # Image models to try first; empty = strongest available first (router).
    # Practice: pro-image beat flash-image (edit 87.8 vs 86.0, generate 89.8 vs 85.2).
    preferred_image_models: list[str] = []
    # Fraction of usable time spent exploring challenge tools.
    explore_fraction = 0.30
    # LLM-proposed follow-up tool calls after schema exploration (0 disables).
    use_llm_tool_planner = True
    planner_rounds = 2
    # Models tried for the main solve before falling back to deterministic data.
    max_solve_attempts = 3
    # Nemotron reasoning on easy puzzles. Off: Practice easy puzzles were 100%
    # correct but took 21-43 s with reasoning on (speed score 26-63).
    reasoning_on_easy = False
    # Image: verify output against the constraints, regenerate once on FAIL.
    image_self_check = True
    # Describe the source image before editing and append it to the prompt.
    # Off: on Practice it lowered edit quality and added latency.
    image_analyze_source = False

    # Text solver
    text_temperature = 0.0
    text_max_tokens = 8192
    default_text_model = "nemotron-3-super"

    text_system_prompt = (
        "You are NemoSentinel, a precise solver for timed text challenges inside an "
        "unknown tool environment.\n"
        "Rules:\n"
        "1. Your FIRST line must be exactly: ANSWER: <final answer>\n"
        "2. The answer must satisfy the challenge's required output format exactly "
        "(case, punctuation, units, JSON shape, choice letters). No extra words.\n"
        "3. Treat 'Tool evidence' and 'Deterministically decoded data' as ground truth; "
        "they were produced by real tool calls and exact code. Prefer them over guesses.\n"
        "4. For logic or arithmetic, check every constraint against your answer before "
        "writing it.\n"
        "5. Never answer 'unknown'. If unsure, give the most likely answer.\n"
        "6. After the ANSWER line you may add at most two short justification lines."
    )
    text_strategy_notes = "Evidence first. Exact format. One ANSWER line."

    image_strategy_notes = (
        "Prefer image_edit when an input image is available; preserve everything the "
        "prompt does not ask to change. Follow every explicit constraint (e.g. 'no text', "
        "aspect ratio, colour, count). Standard resolution only."
    )

    # ------------------------------------------------------------------ hooks
    def rank_models(self, ctx: ChallengeContext, available_models: list[str]) -> list[str]:
        if router is None:
            return super().rank_models(ctx, available_models)
        if self.is_image_challenge(ctx):
            ranked = router.rank_image_models(available_models, preferred=self.preferred_image_models)
            if ranked:
                return ranked
            return super().rank_models(ctx, available_models)
        level = ctx.difficulty if ctx.difficulty in {"easy", "medium", "hard"} else "medium"
        preferred = list(self.preferred_text_models)
        if self.default_text_model and self.default_text_model not in preferred:
            preferred.append(self.default_text_model)
        return router.rank_text_models(available_models, preferred=preferred, level=level)

    def build_solver_prompt(self, ctx: ChallengeContext) -> str:
        clues = "\n".join(f"- {c}" for c in ctx.clues if str(c).strip()) or "- (none)"
        return (
            f"Challenge type: {ctx.challenge_type or 'text'}\n"
            f"Difficulty (estimated): {ctx.difficulty}\n\n"
            f"Description:\n{ctx.description.strip() or '(none)'}\n\n"
            f"Rules and output format:\n{ctx.rules.strip() or '(none)'}\n\n"
            f"Clues:\n{clues}\n\n"
            "Work through the constraints carefully, then output:\n"
            "ANSWER: <final answer>"
        )

    def get_llm_params(self, ctx: ChallengeContext) -> dict:
        # The sentinel runtime adds reasoning on/off per difficulty and time;
        # these are the defaults the Python Simple example uses.
        return {"temperature": float(self.text_temperature), "max_tokens": int(self.text_max_tokens)}

    def on_time_warning(self, remaining_s: float, current_answer: str, ctx: ChallengeContext) -> str | None:
        answer = (current_answer or "").strip()
        if not answer or answer.lower() == "unknown":
            return None
        return answer

    def plan_image_tool(self, ctx: ChallengeContext, available_tools: list[str]) -> str:
        if ctx.image_url and "image_edit" in available_tools:
            return "image_edit"
        if "image_generate" in available_tools:
            return "image_generate"
        if ctx.image_url and "image_analyze" in available_tools:
            return "image_analyze"
        return available_tools[0] if available_tools else ""

    def build_image_prompt(self, ctx: ChallengeContext) -> str:
        instruction = ctx.rules.strip()
        context = ctx.challenge_text.strip() or ctx.description.strip()
        parts = [instruction or context or "Produce the requested image."]
        if context and context not in parts[0]:
            parts.append(f"Context: {context}")
        if ctx.image_url:
            parts.append(
                "Edit the provided image: change only what is requested, keep composition, "
                "subjects, lighting and style otherwise identical."
            )
        parts.append(
            "Follow every explicit constraint exactly. Do not add text, watermarks, borders "
            "or logos unless asked."
        )
        parts.append(f"Output constraints: {_IMAGE_SIZE_HINT}")
        return "\n\n".join(parts)
