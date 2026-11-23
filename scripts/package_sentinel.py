#!/usr/bin/env python3
"""Package the NemoSentinel agent with the organizer's packager, then self-check.

    python scripts/package_sentinel.py --agent-name "NemoSentinel" [--force]

Reads AGENT_ID from the environment or .env. Wraps
``python -m arena_clients.package`` with the right --entrypoint and an
--include for every file in sentinel/, so nothing is forgotten.
Works on macOS, Linux and Windows.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena_clients import package  # noqa: E402

ENTRYPOINT = "examples/sentinel/agent.py"


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", default=os.getenv("AGENT_ID", ""))
    ap.add_argument("--agent-name", default=os.getenv("AGENT_NAME", "NemoSentinel"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.agent_id:
        print("Set AGENT_ID in .env (or pass --agent-id) - it must match the id you practised with.")
        return 2

    includes = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "sentinel").glob("*.py"))
    argv = ["--agent-id", args.agent_id, "--agent-name", args.agent_name, "--entrypoint", ENTRYPOINT]
    for inc in includes:
        argv += ["--include", inc]
    if args.force:
        argv.append("--force")
    os.chdir(ROOT)
    print("Packaging with:", " ".join(argv))
    rc = package.main(argv)
    if rc:
        return rc
    print("\nSelf-check:")
    rc = package.main(["--check", "dist/gauntlet-submission"])
    if rc == 0:
        print(
            f"\nDone. Push exactly these two files to the root of a NEW PUBLIC repo:\n"
            f"  dist/{args.agent_id}-submission.tar.gz\n"
            f"  dist/{args.agent_id}-submission.tar.gz.sha256"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
