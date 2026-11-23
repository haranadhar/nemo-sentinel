"""Unit tests for the NemoSentinel building blocks."""

from __future__ import annotations

import asyncio
import json
import time
import unittest

import sentinel_fakes  # noqa: F401  (puts repo root on sys.path)
from sentinel_fakes import b64

from sentinel import router
from sentinel.answer import detect_format, extract_answer, finalize, normalize, validate
from sentinel.deadline import Deadline
from sentinel.decoders import caesar, decode_candidates, decode_hints
from sentinel.toolbox import Knowledge, ToolSpec, build_arguments, flatten_facts, parse_planned_calls


class DecoderTests(unittest.TestCase):
    def methods(self, text):
        return {d.method: d.text for d in decode_candidates(text)}

    def test_base64(self):
        self.assertEqual(self.methods(f"payload: {b64('the secret code is 4471')}")["base64"], "the secret code is 4471")

    def test_nested_base64(self):
        inner = b64("use tool vault.open with key AX9")
        self.assertIn("use tool vault.open with key AX9", self.methods(b64(inner)).values())

    def test_hex(self):
        hx = "the answer is blue".encode().hex()
        self.assertEqual(self.methods(f"data={hx}")["hex"], "the answer is blue")

    def test_binary(self):
        bits = " ".join(f"{b:08b}" for b in b"key is 42")
        self.assertIn("key is 42", self.methods(bits).values())

    def test_caesar_hinted(self):
        enc = caesar("the hidden word is falcon", 3)
        found = self.methods(f"Caesar cipher: '{enc}'")
        self.assertIn("the hidden word is falcon", found.values())

    def test_rot13(self):
        found = self.methods("rot13 message: 'gur nafjre vf rzrenyq'")
        self.assertEqual(found.get("rot13"), "the answer is emerald")

    def test_morse(self):
        self.assertIn("SOS HELP", self.methods("... --- ... / .... . .-.. .--.").values())

    def test_plain_prose_produces_no_noise(self):
        self.assertEqual(decode_hints(["Find the largest number among the clues and report it."]), [])


class AnswerTests(unittest.TestCase):
    def test_extract_last_valid_answer_and_strip_think(self):
        raw = "<think>ANSWER: wrong</think>\nANSWER: 42\nbecause..."
        self.assertEqual(extract_answer(raw), "42")

    def test_placeholder_rejected(self):
        self.assertEqual(extract_answer("ANSWER: unknown"), "")

    def test_json_multiline(self):
        fmt = detect_format("Respond with a JSON object with keys a and b.")
        raw = 'ANSWER:\n```json\n{"a": 1,\n "b": [2, 3]}\n```'
        ans, problems = finalize(raw, fmt)
        self.assertEqual(json.loads(ans), {"a": 1, "b": [2, 3]})
        self.assertEqual(problems, [])

    def test_integer_normalisation(self):
        fmt = detect_format("Answer with an integer only.")
        self.assertEqual(normalize("The total is 1,250.", fmt), "1250")
        self.assertEqual(validate("1250", fmt), [])

    def test_decimals(self):
        fmt = detect_format("Give the number rounded to 2 decimal places.")
        self.assertEqual(normalize("3.14159", fmt), "3.14")

    def test_choice(self):
        fmt = detect_format("Which is correct?\nA) red\nB) green\nC) blue\nAnswer with the letter.")
        self.assertEqual(fmt.kind, "choice")
        self.assertEqual(normalize("(B) green", fmt), "B")

    def test_uppercase(self):
        fmt = detect_format("Respond with the passphrase in UPPERCASE.")
        self.assertEqual(normalize("emerald falcon.", fmt), "EMERALD FALCON")

    def test_yes_no(self):
        fmt = detect_format("Answer yes or no.")
        self.assertEqual(normalize("Yes, because...", fmt), "yes")


class DeadlineTests(unittest.TestCase):
    def test_reserve_and_budget(self):
        d = Deadline.from_challenge(60, 58)
        self.assertAlmostEqual(d.reserve_s, 4.8, places=1)
        self.assertLessEqual(d.usable(), 58 - 4.8 + 0.01)
        self.assertLessEqual(d.budget(0.3), d.usable() * 0.3 + 0.01)

    def test_resync_only_tightens(self):
        d = Deadline.from_challenge(60, 60)
        d.resync(30)
        self.assertLess(d.remaining(), 31)
        d.resync(50)
        self.assertLess(d.remaining(), 31)

    def test_must_submit_now(self):
        d = Deadline.from_challenge(5, 5, reserve_s=4.6)
        time.sleep(0.05)
        self.assertTrue(d.must_submit_now())


class ToolboxTests(unittest.TestCase):
    def spec(self, name, props=None, required=None):
        return ToolSpec.from_def({"name": name, "description": "", "inputSchema": {"properties": props or {}, "required": required or []}})

    def test_deny_list(self):
        for bad in ("arena.submit", "admin.reset_scores", "leaderboard.get", "system.shutdown", "db_delete", "py.exec"):
            self.assertTrue(self.spec(bad).denied, bad)
        for ok in ("archive.list_records", "vault.open", "execute_query_readonly", "search"):
            self.assertFalse(self.spec(ok).denied, ok)

    def test_build_arguments_from_facts_and_schema(self):
        kb = Knowledge("team-x", "What is in record R-17?")
        kb.add_result({"records": [{"record_id": "R-17", "title": "vault"}]})
        spec = self.spec("archive.get_record", {"record_id": {"type": "string"}, "agent_id": {"type": "string"},
                                                "verbose": {"type": "boolean", "default": True}}, ["record_id"])
        args, missing = build_arguments(spec, kb)
        self.assertEqual(missing, [])
        self.assertEqual(args, {"record_id": "R-17", "agent_id": "team-x", "verbose": True})

    def test_secret_not_guessed(self):
        kb = Knowledge("team-x", "")
        spec = self.spec("vault.open", {"passcode": {"type": "string"}}, ["passcode"])
        _args, missing = build_arguments(spec, kb)
        self.assertEqual(missing, ["passcode"])

    def test_text_fact_extraction(self):
        kb = Knowledge("a", "")
        kb.add_text("Note: the vault code = 7781.")
        self.assertEqual(kb.lookup("code"), "7781")

    def test_enum_and_query(self):
        kb = Knowledge("a", "Search the archive for falcon sightings in europe")
        spec = self.spec("archive.search", {"query": {"type": "string"},
                                            "region": {"type": "string", "enum": ["asia", "europe"]}}, ["query"])
        args, _ = build_arguments(spec, kb)
        self.assertEqual(args["region"], "europe")
        self.assertIn("falcon", args["query"])

    def test_scalar_list_candidates_fan_out(self):
        from sentinel.toolbox import argument_variants
        kb = Knowledge("a", "total balance across accounts")
        kb.add_result({"accounts": ["ACC-1", "ACC-2"]})
        spec = self.spec("ledger.balance", {"account_id": {"type": "string"}}, ["account_id"])
        self.assertEqual([v["account_id"] for v in argument_variants(spec, kb)], ["ACC-1", "ACC-2"])

    def test_relevance_ranking_of_list_items(self):
        kb = Knowledge("a", "Recover the passphrase stored in the vault")
        kb.add_result({"records": [{"record_id": "R-02", "title": "menu"}, {"record_id": "R-17", "title": "vault access"}]})
        self.assertEqual(kb.candidates("record_id")[:2], ["R-17", "R-02"])

    def test_unwrap_envelopes(self):
        from sentinel.toolbox import unwrap
        self.assertEqual(unwrap({"result": {"image_uri": "x"}}), {"image_uri": "x"})
        self.assertEqual(unwrap({"raw": '{"a": 1}'}), {"a": 1})
        self.assertEqual(unwrap({"raw": "plain"}), {"raw": "plain"})

    def test_flatten_facts(self):
        f = flatten_facts({"a": {"b": {"key": "K1"}}, "list": [{"id": 5}]})
        self.assertEqual(f["key"], "K1")
        self.assertEqual(f["a.b.key"], "K1")

    def test_planned_calls_validated(self):
        specs = {s.name: s for s in [
            self.spec("vault.open", {"code": {"type": "string"}, "agent_id": {"type": "string"}}, ["code"]),
            self.spec("admin.reset"),
        ]}
        raw = '```json\n[{"tool":"vault.open","arguments":{"code":"7781","evil":1}},' \
              '{"tool":"admin.reset","arguments":{}},{"tool":"nope","arguments":{}},' \
              '{"tool":"vault.open","arguments":{}}]\n```'
        calls = parse_planned_calls(raw, specs, "team-x")
        self.assertEqual(calls, [("vault.open", {"code": "7781", "agent_id": "team-x"})])


class RouterTests(unittest.TestCase):
    ROSTER = ["llama-3.1-8b-instruct", "nemotron-3-nano", "nemotron-3-super", "flux-image", "text-embed-3", "gpt-oss-120b"]

    def test_chat_models_only_and_preferred_first(self):
        ranked = router.rank_text_models(self.ROSTER, preferred=["nemotron-3-super"])
        self.assertEqual(ranked[0], "nemotron-3-super")
        self.assertNotIn("flux-image", ranked)
        self.assertNotIn("text-embed-3", ranked)

    def test_strength_order_without_preference(self):
        ranked = router.rank_text_models(self.ROSTER)
        self.assertLess(ranked.index("gpt-oss-120b"), ranked.index("llama-3.1-8b-instruct"))

    def test_fast_model(self):
        self.assertIn(router.fast_model(self.ROSTER), {"nemotron-3-nano", "llama-3.1-8b-instruct"})

    def test_reasoning_policy(self):
        easy = router.llm_params("nemotron-3-super", level="easy", usable_s=50)
        self.assertEqual(easy["extra_body"], {"reasoning": {"enabled": False}})
        hard = router.llm_params("nemotron-3-super", level="hard", usable_s=100)
        self.assertNotIn("extra_body", hard)
        self.assertGreaterEqual(hard["max_tokens"], 8192)

    def test_assess(self):
        self.assertEqual(router.assess(puzzle_id="practice_logic_hard_001").level, "hard")
        self.assertEqual(router.assess(max_time_s=60, text="What is 2+2?").level, "easy")


if __name__ == "__main__":
    unittest.main()
