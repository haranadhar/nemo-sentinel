"""End-to-end NemoSentinel runs against simulated black-box arenas.

Each scenario is a fake MCP server with hidden tools / encoded data, a fake
REST API and a scripted LLM. The assertions check *behaviour that scores*:
the right tools were called with the right arguments, required tools were
covered, the answer is in the required format, thoughts were broadcast,
nothing out-of-bounds was touched, and the deadline was respected.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from sentinel_fakes import FakeHttp, FakeMcp, FakeOpenAI, FakeTool, b64, user_text

from my_strategy import MyStrategy
from sentinel.llm import LLM
from sentinel.runtime import ImageRun, Ports, TextRun

MODELS = ["nemotron-3-nano", "nemotron-3-super", "gpt-oss-120b", "flux-image"]


def run(coro):
    return asyncio.run(coro)


def make_ports(mcp, http, responder, delay_s=0.0, models=None):
    client = FakeOpenAI(responder, delay_s)
    ports = Ports(http=http, mcp=mcp, llm=LLM(client), agent_id="team-test", strategy=MyStrategy(),
                  models=models or MODELS, log=lambda _m: None, sleep=asyncio.sleep)
    return ports, client


def vault_arena(max_time_s=90):
    """list_records -> get_record(record_id) -> base64 'vault code = 7781' ->
    hidden vault.open(code) appears -> returns the secret."""
    return FakeMcp(
        challenge={
            "challenge_type": "logic",
            "puzzle_id": "blackbox_vault_medium",
            "description": "Recover the passphrase stored in the vault.",
            "rules": "Respond with the passphrase in UPPERCASE.",
            "max_time_s": max_time_s,
            "required_tools": ["archive.list_records", "vault.open"],
        },
        clues=["The archivist keeps records. Payloads are encoded."],
        tools=[
            FakeTool("arena.get_challenge", lambda a: {}),
            FakeTool("arena.clues.list", lambda a: {}),
            FakeTool("arena.clues.get", lambda a: {}),
            FakeTool("arena.time_remaining", lambda a: {}),
            FakeTool("archive.list_records", lambda a: {"records": [{"record_id": "R-02", "title": "menu"},
                                                                 {"record_id": "R-17", "title": "vault access"}]},
                     props={"agent_id": {"type": "string"}}, description="List archive records"),
            FakeTool("archive.get_record", lambda a: {"payload": b64("vault code = 7781")} if a.get("record_id") == "R-17"
                     else {"payload": b64("soup of the day: tomato")},
                     props={"record_id": {"type": "string"}, "agent_id": {"type": "string"}}, required=["record_id"],
                     description="Fetch one archive record"),
            FakeTool("vault.open", lambda a: {"secret": "Emerald Falcon"} if a.get("code") == "7781"
                     else {"error": "wrong code"},
                     props={"code": {"type": "string"}, "agent_id": {"type": "string"}}, required=["code"],
                     description="Open the vault", hidden=True),
            FakeTool("admin.reset_scores", lambda a: {"ok": True}, description="DANGER"),
            FakeTool("arena.submit_answer", lambda a: {"ok": True}),
        ],
        unlocks={"archive.get_record": ["vault.open"]},
    )


def vault_responder(model, messages, kwargs):
    text = user_text(messages)
    if "Return only JSON" in messages[0]["content"]:
        return "[]"
    if "Emerald Falcon" in text:
        return "ANSWER: emerald falcon\nFrom vault.open result."
    return "ANSWER: unknown"


class TextScenarioTests(unittest.TestCase):
    def test_chained_hidden_tool_with_encoded_key(self):
        mcp, http = vault_arena(), FakeHttp(expected_answer="EMERALD FALCON")
        ports, _ = make_ports(mcp, http, vault_responder)
        report = run(TextRun(ports, run(mcp.list_tool_defs())).run())

        # Relevance ranking: the record whose title matches the question first.
        self.assertEqual(mcp.called("archive.get_record")[0]["record_id"], "R-17")
        self.assertEqual(mcp.called("vault.open")[0]["code"], "7781")
        self.assertEqual(http.submits, ["EMERALD FALCON"])
        self.assertTrue(report.submit["accepted"])
        self.assertEqual(report.missing_required_tools, [])
        self.assertFalse(mcp.called("admin.reset_scores"))
        self.assertFalse(mcp.called("arena.submit_answer"))
        self.assertGreaterEqual(report.thoughts, 2)
        self.assertGreaterEqual(len(report.challenge_tools_ok), 3)  # tool-coverage axis

    def test_llm_planner_fills_argument_exploration_cannot(self):
        """The id is only described in prose; the planner must supply it."""
        mcp = FakeMcp(
            challenge={"challenge_type": "logic", "description": "What colour is the box labelled 'north'?",
                       "rules": "Answer with one word, lowercase.", "max_time_s": 90},
            clues=["Box ids follow the pattern box-<direction>."],
            tools=[FakeTool("boxes.inspect", lambda a: {"colour": "teal"} if a.get("box_id") == "box-north"
                            else {"error": "unknown box"},
                            props={"box_id": {"type": "string"}}, required=["box_id"], description="Inspect a box")],
        )
        http = FakeHttp(expected_answer="teal")

        def responder(model, messages, kwargs):
            if "Return only JSON" in messages[0]["content"]:
                return '[{"tool": "boxes.inspect", "arguments": {"box_id": "box-north"}}]'
            return "ANSWER: Teal" if "teal" in user_text(messages) else "ANSWER: unknown"

        ports, _ = make_ports(mcp, http, responder)
        report = run(TextRun(ports, run(mcp.list_tool_defs())).run())
        self.assertIn({"box_id": "box-north"}, mcp.called("boxes.inspect"))
        self.assertEqual(http.submits, ["teal"])
        self.assertEqual(report.format_problems, [])

    def test_deterministic_decoding_reaches_solver(self):
        mcp = FakeMcp(
            challenge={"challenge_type": "cipher", "description": "Decode the intercepted message.",
                       "rules": "Answer with the decoded city name only.", "max_time_s": 60},
            clues=[f"Intercepted: {b64('the meeting is in lisbon')}"],
            tools=[],
        )
        http = FakeHttp(expected_answer="lisbon")
        seen = {}

        def responder(model, messages, kwargs):
            seen["prompt"] = user_text(messages)
            return "ANSWER: lisbon" if "the meeting is in lisbon" in seen["prompt"] else "ANSWER: paris"

        ports, client = make_ports(mcp, http, responder)
        run(TextRun(ports, []).run())
        self.assertIn("[decoded base64]", seen["prompt"])
        self.assertEqual(http.submits, ["lisbon"])
        # MyStrategy.reasoning_on_easy=True: Nemotron keeps reasoning on.
        self.assertNotIn("extra_body", client.completions.requests[0])

    def test_reasoning_off_on_easy_when_strategy_says_so(self):
        mcp = FakeMcp(challenge={"description": "2+2?", "rules": "Integer only.", "max_time_s": 60}, clues=[], tools=[])
        ports, client = make_ports(mcp, FakeHttp(), lambda *a: "ANSWER: 4")
        ports.strategy.reasoning_on_easy = False
        run(TextRun(ports, []).run())
        self.assertEqual(client.completions.requests[0].get("extra_body"), {"reasoning": {"enabled": False}})

    def test_format_repair_json(self):
        mcp = FakeMcp(challenge={"description": "Give the two primes below 6 greater than 2.",
                                 "rules": "Respond with a JSON array of integers.", "max_time_s": 90},
                      clues=[], tools=[])
        http = FakeHttp()
        calls = {"n": 0}

        def responder(model, messages, kwargs):
            calls["n"] += 1
            if "fix answer formatting" in messages[0]["content"]:
                return "ANSWER: [3, 5]"
            return "ANSWER: 3 and 5"

        ports, _ = make_ports(mcp, http, responder)
        report = run(TextRun(ports, []).run())
        self.assertEqual(http.submits, ["[3, 5]"])
        self.assertIn("repair", report.answer_source)

    def test_fallback_model_after_token_exhaustion(self):
        mcp = FakeMcp(challenge={"description": "2+3?", "rules": "Answer with an integer only.", "max_time_s": 90},
                      clues=[], tools=[])
        http = FakeHttp(expected_answer="5")

        def responder(model, messages, kwargs):
            if model == "nemotron-3-super":
                return ("", "length")
            return "ANSWER: 5"

        ports, client = make_ports(mcp, http, responder)
        report = run(TextRun(ports, []).run())
        self.assertEqual(http.submits, ["5"])
        self.assertNotEqual(report.model, "nemotron-3-super")

    def test_watchdog_submits_before_deadline_when_llm_hangs(self):
        mcp = FakeMcp(challenge={"description": "Hard puzzle", "rules": "Answer with an integer only.",
                                 "max_time_s": 10}, clues=[], tools=[])
        http = FakeHttp()
        ports, _ = make_ports(mcp, http, lambda *a: "ANSWER: 7", delay_s=60)
        start = time.monotonic()
        report = run(TextRun(ports, []).run())
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 10.0)
        self.assertEqual(len(http.submits), 1)
        self.assertEqual(http.submits[0], "0")  # format-valid fallback, never 'unknown'
        self.assertTrue(report.submit.get("accepted") is not None)

    def test_required_tool_called_even_if_not_useful(self):
        mcp = FakeMcp(challenge={"description": "Say hello.", "rules": "Answer with one word.",
                                 "max_time_s": 60, "required_tools": ["arena.time_remaining", "notes.read"]},
                      clues=[], tools=[FakeTool("notes.read", lambda a: {"note": "irrelevant"},
                                                props={"page": {"type": "integer"}}, required=["page"])])
        http = FakeHttp()
        ports, _ = make_ports(mcp, http, lambda *a: "ANSWER: hello")
        report = run(TextRun(ports, run(mcp.list_tool_defs())).run())
        self.assertTrue(mcp.called("arena.time_remaining"))
        self.assertTrue(mcp.called("notes.read"))
        self.assertEqual(report.missing_required_tools, [])

    def test_submit_retries_same_answer_on_transport_error(self):
        mcp = FakeMcp(challenge={"description": "1+1?", "rules": "Answer with an integer only.", "max_time_s": 60},
                      clues=[], tools=[])
        http = FakeHttp(expected_answer="2", fail_submits=1)
        ports, _ = make_ports(mcp, http, lambda *a: "ANSWER: 2")
        report = run(TextRun(ports, []).run())
        self.assertEqual(http.submits, ["2", "2"])
        self.assertTrue(report.submit["accepted"])

    def test_no_retry_on_409(self):
        mcp = FakeMcp(challenge={"description": "1+1?", "rules": "Answer with an integer only.", "max_time_s": 60},
                      clues=[], tools=[])
        http = FakeHttp(fail_submits=3, status_code=409)
        ports, _ = make_ports(mcp, http, lambda *a: "ANSWER: 2")
        run(TextRun(ports, []).run())
        self.assertEqual(len(http.submits), 1)

    def test_tool_errors_and_exceptions_do_not_crash(self):
        def boom(a):
            raise RuntimeError("server exploded")

        mcp = FakeMcp(challenge={"description": "Name a colour.", "rules": "One word.", "max_time_s": 60},
                      clues=[], tools=[FakeTool("weird.tool", boom), FakeTool("bad.tool", lambda a: {"error": "nope"})])
        http = FakeHttp()
        ports, _ = make_ports(mcp, http, lambda *a: "ANSWER: red")
        run(TextRun(ports, run(mcp.list_tool_defs())).run())
        self.assertEqual(http.submits, ["red"])


class ImageScenarioTests(unittest.TestCase):
    def image_arena(self, verdicts):
        verdicts = list(verdicts)
        counter = {"edits": 0}

        def edit(a):
            counter["edits"] += 1
            return {"image_uri": f"data:image/png;base64,EDIT{counter['edits']}", "model": a.get("model")}

        def analyze(a):
            if a["image_uri"].endswith("SOURCE"):
                return {"text": "A red bicycle against a brick wall."}
            return {"text": verdicts.pop(0) if verdicts else "PASS"}

        tools = [
            FakeTool("arena.image.get_challenge", lambda a: {}),
            FakeTool("arena.image.submit_edit", lambda a: {}),
            FakeTool("arena.image.broadcast_thought", lambda a: {}),
            FakeTool("image_edit", edit, props={"image_uri": {}, "prompt": {}, "agent_id": {}, "model": {}},
                     required=["image_uri", "prompt"]),
            FakeTool("image_analyze", analyze, props={"image_uri": {}, "question": {}, "agent_id": {}},
                     required=["image_uri", "question"]),
        ]
        mcp = FakeMcp(challenge={}, clues=[], tools=tools, image={
            "challenge_type": "image_edit", "puzzle_id": "practice_image_edit_easy_001",
            "prompt": "Make the bicycle blue. No text.", "max_time_s": 120, "time_remaining_s": 120,
            "input_image_uri": "data:image/png;base64,SOURCE", "required_tools": ["image_edit"],
        })
        return mcp, counter

    def test_edit_with_analysis_and_self_check_regeneration(self):
        mcp, counter = self.image_arena(["FAIL: bicycle still red"])
        http = FakeHttp()
        ports, _ = make_ports(mcp, http, lambda *a: "")
        report = run(ImageRun(ports, run(mcp.list_tool_defs())).run())
        self.assertEqual(counter["edits"], 2)  # regenerated once after FAIL
        self.assertEqual(mcp.submitted_image, "data:image/png;base64,EDIT2")
        self.assertEqual(mcp.called("image_edit")[0]["model"], "flux-image")
        self.assertIn("Preserve everything", mcp.called("image_edit")[0]["prompt"])
        self.assertEqual(report.missing_required_tools, [])
        self.assertTrue(mcp.thoughts)

    def test_edit_pass_no_regeneration(self):
        mcp, counter = self.image_arena(["PASS"])
        ports, _ = make_ports(mcp, FakeHttp(), lambda *a: "")
        run(ImageRun(ports, run(mcp.list_tool_defs())).run())
        self.assertEqual(counter["edits"], 1)


if __name__ == "__main__":
    unittest.main()
