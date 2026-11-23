"""NemoSentinel: an adaptive, deadline-aware Agent Gauntlet competitor.

Modules
-------
deadline   Monotonic time budget, phase budgets, submission reserve.
decoders   Deterministic, safe decoding of encoded data (no code execution).
toolbox    Schema-aware MCP tool classification, argument building, exploration.
answer     Answer extraction, format detection, validation and normalisation.
router     Challenge complexity classification and model ranking.
llm        Async, timeout-bounded LLM calls through the organizer proxy.
runtime    The text and image challenge flows that tie it all together.

Only the standard library plus the starter kit's locked dependencies are used,
so the submission lockfile does not change.
"""

__version__ = "1.0.0"
