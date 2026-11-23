"""Offline fakes of the arena (MCP + REST) and the LLM proxy.

They let the full NemoSentinel runtime run end to end without network access,
against 'black box' scenarios with hidden tools and encoded data.
"""

from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients.mcp_client import ChallengeInfo, ClueInfo  # noqa: E402


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


class FakeTool:
    def __init__(self, name: str, handler: Callable[[dict], Any], *, props: dict | None = None,
                 required: list[str] | None = None, description: str = "", hidden: bool = False):
        self.name = name
        self.handler = handler
        self.description = description
        self.inputSchema = {"type": "object", "properties": props or {}, "required": required or []}
        self.hidden = hidden


class FakeMcp:
    def __init__(self, *, challenge: dict, clues: list[str], tools: list[FakeTool], image: dict | None = None,
                 unlocks: dict[str, list[str]] | None = None):
        self.challenge = challenge
        self.clues = clues
        self.tools = {t.name: t for t in tools}
        self.image = image
        self.unlocks = unlocks or {}
        self.calls: list[tuple[str, dict]] = []
        self.started = time.monotonic()
        self.submitted_image: str | None = None
        self.thoughts: list[str] = []

    # discovery
    async def list_tool_defs(self):
        return [t for t in self.tools.values() if not t.hidden]

    async def list_tools(self):
        return [t.name for t in await self.list_tool_defs()]

    async def call_tool(self, name, arguments=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments))
        if name == "arena.image.get_challenge":
            return dict(self.image or {"error": "not_image_challenge"})
        t = self.tools.get(name)
        if t is None or t.hidden:
            return {"error": f"unknown tool {name}"}
        missing = [r for r in t.inputSchema["required"] if r not in arguments]
        if missing:
            return {"error": f"missing required argument(s): {missing}"}
        out = t.handler(arguments)
        if asyncio.iscoroutine(out):
            out = await out
        for unlocked in self.unlocks.get(name, []):
            if unlocked in self.tools:
                self.tools[unlocked].hidden = False
        return out

    # typed helpers used by the runtime
    async def get_challenge(self, agent_id):
        self.calls.append(("arena.get_challenge", {"agent_id": agent_id}))
        c = self.challenge
        return ChallengeInfo(
            challenge_type=c.get("challenge_type", "logic"),
            challenge_id=c.get("challenge_id", "c1"),
            puzzle_id=c.get("puzzle_id", "p1"),
            description=c.get("description", ""),
            rules=c.get("rules", ""),
            max_time_s=c.get("max_time_s", 60),
            clues=[],
            time_remaining_s=c.get("max_time_s", 60),
            raw_data=dict(c),
        )

    async def list_clues(self, agent_id):
        self.calls.append(("arena.clues.list", {"agent_id": agent_id}))
        return [f"clue_{i}" for i in range(len(self.clues))]

    async def get_clue(self, clue_id, agent_id):
        self.calls.append(("arena.clues.get", {"clue_id": clue_id}))
        return ClueInfo(clue_id=clue_id, text=self.clues[int(clue_id.split("_")[1])], time_remaining_s=0)

    async def time_remaining(self, agent_id):
        self.calls.append(("arena.time_remaining", {"agent_id": agent_id}))
        left = self.challenge.get("max_time_s", 60) - (time.monotonic() - self.started)
        return {"time_remaining_s": left, "expired": left <= 0}

    async def broadcast_image_thought(self, thought, agent_id):
        self.thoughts.append(thought)
        return {"ok": True}

    async def submit_image(self, agent_id, image_uri, client_metrics=None, rationale=""):
        self.calls.append(("arena.image.submit_edit", {"edited_image": image_uri[:40]}))
        self.submitted_image = image_uri
        return {"accepted": True, "status": "submitted", "score": {"final_score": 90.0}}

    def called(self, name: str) -> list[dict]:
        return [a for n, a in self.calls if n == name]


class FakeHttp:
    def __init__(self, expected_answer: str | None = None, fail_submits: int = 0, status_code: int | None = None):
        self.thoughts: list[str] = []
        self.drafts: list[str] = []
        self.submits: list[str] = []
        self.expected = expected_answer
        self.fail_submits = fail_submits
        self.status_code = status_code

    def broadcast_thought(self, agent_id, thought):
        self.thoughts.append(thought)
        return True

    def save_draft(self, agent_id, answer, reasoning=""):
        self.drafts.append(answer)
        return True

    def update_status(self, agent_id, status, metrics=None):
        return True

    def submit(self, agent_id, answer, client_metrics=None, challenge_type=None):
        self.submits.append(answer)
        if self.fail_submits > 0:
            self.fail_submits -= 1
            err = ConnectionError("connection reset")
            if self.status_code:
                err.status_code = self.status_code  # type: ignore[attr-defined]
            raise err
        accepted = self.expected is None or answer == self.expected
        return SimpleNamespace(accepted=accepted, status="submitted", answer=answer,
                               score={"final_score": 95.0 if accepted else 10.0})


class FakeCompletions:
    def __init__(self, responder: Callable[[str, list[dict], dict], Any], delay_s: float = 0.0):
        self.responder = responder
        self.delay_s = delay_s
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.delay_s:
            limit = float(kwargs.get("timeout", self.delay_s))
            time.sleep(min(self.delay_s, limit))
            if self.delay_s > limit:  # like openai.APITimeoutError
                raise TimeoutError(f"Request timed out after {limit}s")
        out = self.responder(kwargs["model"], kwargs["messages"], kwargs)
        if isinstance(out, Exception):
            raise out
        content, finish = (out, "stop") if isinstance(out, str) else out
        msg = SimpleNamespace(content=content, reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


class FakeOpenAI:
    def __init__(self, responder, delay_s: float = 0.0):
        self.completions = FakeCompletions(responder, delay_s)
        self.chat = SimpleNamespace(completions=self.completions)


def user_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")
