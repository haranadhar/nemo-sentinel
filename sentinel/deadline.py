"""Deadline management.

The arena reports ``max_time_s`` and ``time_remaining_s`` when the challenge
is fetched. We anchor a monotonic clock to that and never trust wall-clock
time again, so a slow tool or LLM call cannot make us miss the submission.

A late answer is effectively a loss, so a fixed ``reserve_s`` is always kept
back for the final submit (plus its retry).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Deadline:
    total_s: float
    remaining_at_start_s: float
    reserve_s: float = 6.0
    _t0: float = field(default_factory=time.monotonic)

    @classmethod
    def from_challenge(
        cls,
        max_time_s: float | int | None,
        time_remaining_s: float | int | None,
        *,
        reserve_s: float | None = None,
        default_s: float = 60.0,
    ) -> "Deadline":
        total = float(max_time_s or 0) or default_s
        remaining = float(time_remaining_s or 0) or total
        remaining = min(remaining, total) if total else remaining
        if reserve_s is None:
            # ~8% of the budget, clamped to [4s, 12s]: enough for a submit,
            # one retry and network jitter, without wasting short puzzles.
            reserve_s = max(4.0, min(12.0, total * 0.08))
        return cls(total_s=total, remaining_at_start_s=remaining, reserve_s=reserve_s)

    # ------------------------------------------------------------------ clock
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def remaining(self) -> float:
        """Seconds until the server deadline (can go negative)."""
        return self.remaining_at_start_s - self.elapsed()

    def usable(self) -> float:
        """Seconds we may still spend on work before the submit reserve."""
        return max(0.0, self.remaining() - self.reserve_s)

    def resync(self, server_remaining_s: float | int | None) -> None:
        """Re-anchor to a fresh server reading (only ever tightens)."""
        try:
            server = float(server_remaining_s)
        except (TypeError, ValueError):
            return
        if server <= 0:
            return
        if server < self.remaining():
            self.remaining_at_start_s = server + self.elapsed()

    # ---------------------------------------------------------------- budgets
    def budget(self, fraction: float, *, floor_s: float = 1.0, cap_s: float | None = None) -> float:
        """A slice of the *usable* time, for one phase."""
        value = self.usable() * max(0.0, min(1.0, fraction))
        if cap_s is not None:
            value = min(value, cap_s)
        return max(min(floor_s, self.usable()), value)

    def must_submit_now(self) -> bool:
        return self.usable() <= 0.5

    def tight(self, threshold_s: float = 20.0) -> bool:
        return self.usable() < threshold_s

    def snapshot(self) -> dict[str, float]:
        return {
            "elapsed_s": round(self.elapsed(), 2),
            "remaining_s": round(self.remaining(), 2),
            "usable_s": round(self.usable(), 2),
        }
