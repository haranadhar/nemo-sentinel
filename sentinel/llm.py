"""Timeout-bounded LLM calls through the organizer's OpenAI-compatible proxy.

The starter kit calls the synchronous OpenAI client directly from async code,
which blocks the event loop and has no deadline. Here every call runs in a
worker thread, carries a per-request timeout, and is wrapped in
``asyncio.wait_for`` so a slow model can never make us miss the deadline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    model: str
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.content.strip() or self.reasoning.strip())


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    models: list[str] = field(default_factory=list)

    def add(self, r: LLMResult) -> None:
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.total_tokens += r.total_tokens
        self.calls += 1
        if r.model and r.model not in self.models:
            self.models.append(r.model)


class LLM:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.usage = LLMUsage()

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        extra_body: dict | None = None,
        timeout_s: float = 30.0,
    ) -> LLMResult:
        timeout_s = max(1.0, float(timeout_s))
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "timeout": timeout_s,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.chat.completions.create, **kwargs),
                timeout=timeout_s + 1.0,
            )
        except asyncio.TimeoutError:
            r = LLMResult(model=model, error=f"timeout after {timeout_s:.0f}s", elapsed_s=time.monotonic() - start)
            self.usage.add(r)
            return r
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            r = LLMResult(model=model, error=f"{type(exc).__name__}: {exc}"[:400], elapsed_s=time.monotonic() - start)
            self.usage.add(r)
            return r

        r = LLMResult(model=model, elapsed_s=round(time.monotonic() - start, 3))
        try:
            choice = response.choices[0]
            msg = choice.message
            r.content = str(getattr(msg, "content", "") or "")
            r.reasoning = str(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "")
            r.finish_reason = str(getattr(choice, "finish_reason", "") or "")
        except (AttributeError, IndexError, TypeError) as exc:
            r.error = f"malformed response: {exc}"
        usage = getattr(response, "usage", None)
        if usage is not None:
            r.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            r.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            r.total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or r.prompt_tokens + r.completion_tokens
        self.usage.add(r)
        return r
