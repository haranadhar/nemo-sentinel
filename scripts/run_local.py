#!/usr/bin/env python3
"""Run the NemoSentinel agent against the local mock arena.

    python scripts/run_local.py [--port 8765] [--agent-id local-sentinel]

Overrides the service URLs for this process only; your .env is not touched.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--agent-id", default="local-sentinel")
args = ap.parse_args()

base = f"http://127.0.0.1:{args.port}"
os.environ.update({
    "ARENA_SERVER": base,
    "ARENA_API_BASE": base,
    "ARENA_MCP_URL": base,
    "LLM_PROXY_HOST": f"{base}/proxy",
    "ARENA_API_KEY": os.environ.get("MOCK_ARENA_KEY", "local-mock-key"),
    "AGENT_ID": args.agent_id,
    "ARENA_REGISTRATION_TIMEOUT_S": "10",
})
# python-dotenv never overrides variables that are already set, so your
# Practice values in .env are ignored for this run.
sys.argv = [str(ROOT / "examples/sentinel/agent.py")]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "examples/sentinel/agent.py"), run_name="__main__")
