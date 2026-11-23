# NemoSentinel: an adaptive Agent Gauntlet competitor

An autonomous MCP agent for the **NVIDIA Developer Live Agent Gauntlet** (GTC DC 2026), built
on the official starter kit. The organizer's `arena_clients/` code is left **unchanged**, so
registration, telemetry and submission contracts are not touched. All the solving logic lives in `sentinel/`.

> The organizer's original starter-kit README is kept at [docs/STARTER_KIT_README.md](docs/STARTER_KIT_README.md).

## Capabilities compared with the stock starter kit

| Stock `python_simple` | NemoSentinel |
|---|---|
| Calls a challenge tool only if its *name* appears in the text; guesses args from 12 fixed payload shapes | Reads every tool's **JSON schema** and fills arguments from facts it has learned (ids, keys, codes) |
| One pass, no follow-up | **Round-based exploration**: results feed the next round; **re-lists tools** so hidden ones that appear later are found |
| Keeps one value per key | Keeps **ranked candidates** (list items relevant to the question first) and **fans out** across them |
| — | **LLM tool-planner** (bounded, schema-validated JSON) when exploration leaves gaps |
| LLM decodes base64/ciphers (often wrongly) | **Deterministic decoders**: base64 (nested), hex, binary, ROT13/Caesar/Atbash, Morse, URL/HTML |
| Ignores `required_tools` from the challenge | **Guarantees** every required tool is actually executed (a missing one zeroes quality on Practice) |
| Blocking sync LLM call, no deadline | Every LLM/tool call **timeout-bounded**; a **watchdog** submits the best answer before time runs out |
| Nemotron reasoning always on | Reasoning **on for medium/hard, off for easy** or when time is short |
| Answer = first `ANSWER:` line, else `"unknown"` | Format **detection + normalisation + validation** (JSON, integer, decimals, choice, yes/no, case) + cheap repair call |
| Submit once | Exact-same-answer **retry** on transport failure; never retries a 409 |
| Image: edit/generate once | Analyse source → edit → **self-check** vs. constraints → regenerate once if it fails |
| — | Never calls state-changing / out-of-bounds tools (`submit`, `reset`, `admin`, `leaderboard`, `exec`…) |

## Quick start (macOS / Linux)

```bash
git clone https://github.com/haranadhar/nemo-sentinel.git
cd nemo-sentinel
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env        # then edit: ARENA_API_KEY and a unique AGENT_ID
code .                      # open in VS Code
```

Windows (PowerShell): `py -3.12 -m venv .venv` then `.venv\Scripts\Activate.ps1`, the rest is the same.
In VS Code pick the interpreter **.venv** (Ctrl/Cmd+Shift+P → *Python: Select Interpreter*).

## Verify in this order

| Step | Command | VS Code |
|---|---|---|
| 1. Unit + scenario tests (offline) | `python -m unittest discover -s tests` | Testing panel, or Task *Run all tests* |
| 2. Offline black-box rehearsal | `python scripts/mock_arena.py --scenario vault` then in a 2nd terminal `python scripts/run_local.py` | Run ▸ **★ Rehearse offline** |
| 3. Practice readiness | `python -m arena_clients.doctor` → *All checks passed* | Task *Doctor* |
| 4. One real Practice run | `python examples/sentinel/agent.py` | Run ▸ **1. NemoSentinel vs Practice** |
| 5. Benchmark vs baseline | `python scripts/practice_bench.py --runs 10` and `--runs 10 --baseline`, then `--summary` | Run ▸ **4. Practice benchmark** |
| 6. Package + self-check | `python scripts/package_sentinel.py --force` | Task *Package submission* |

Scenarios for step 2: `vault` (hidden chained tools + base64 key), `cipher` (Caesar),
`ledger` (JSON output, fan-out over accounts), `image` (edit + self-check). Use
`--llm upstream --upstream-url <practice>/proxy --upstream-key <key>` to put **real models**
behind your own scenarios. Add new scenarios in `scripts/mock_arena.py` (`SCENARIOS`).

Every run writes a report to `outputs/<AGENT_ID>/<timestamp>-<modality>.json` (append-only),
so each run's tools, timings, decoded hints and score are kept.

## Configuration

Prompts, model preferences and runtime tunables live in `my_strategy.py`
(for example `preferred_text_models`, `explore_fraction`, `use_llm_tool_planner`,
`planner_rounds`, `image_self_check`).

## Packaging

```bash
python scripts/package_sentinel.py --force   # builds and self-checks the archive in dist/
```

## Layout

```
sentinel/            the agent's brain (stdlib only, no new dependencies)
  deadline.py        monotonic budget, submit reserve
  toolbox.py         schema-aware exploration, facts, safety deny-list, planner validation
  decoders.py        deterministic decoding
  answer.py          extraction, format detection/validation/normalisation
  router.py          difficulty assessment, model ranking, reasoning policy
  llm.py             async, timeout-bounded proxy calls
  runtime.py         text + image flows, watchdog, required-tool guarantee, submit retry
examples/sentinel/agent.py   entrypoint (packaged)
my_strategy.py       prompts + tunables (shared with the stock examples)
scripts/             mock_arena, run_local, practice_bench, package_sentinel
tests/               organizer tests + test_sentinel_units.py + test_sentinel_e2e.py
```

License: Apache-2.0 (inherits the starter kit's license).
