#!/usr/bin/env python3
"""Run the agent repeatedly against Practice and log every scored run.

    python scripts/practice_bench.py --runs 10                 # NemoSentinel
    python scripts/practice_bench.py --runs 10 --baseline      # stock Python Simple, for comparison

Each run registers afresh (= next puzzle in Practice's shared rotation),
plays, and its score is appended to bench/results.csv. Nothing is overwritten,
so you can compare versions over time:

    python scripts/practice_bench.py --summary

Be a good neighbour: Practice is shared, keep --pause >= 15s.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "bench" / "results.csv"
FIELDS = ["ts", "agent", "version", "puzzle_id", "modality", "accepted", "final", "quality", "speed", "tools",
          "tokens", "elapsed_s", "model", "missing_required", "answer", "notes"]


def _env_agent_id() -> str:
    try:
        from dotenv import dotenv_values

        return str(dotenv_values(ROOT / ".env").get("AGENT_ID") or os.getenv("AGENT_ID") or "")
    except ImportError:
        return os.getenv("AGENT_ID", "")


def _latest(pattern: str, after: float) -> Path | None:
    files = [Path(p) for p in glob.glob(pattern) if os.path.getmtime(p) >= after]
    return max(files, key=os.path.getmtime) if files else None


def _row_from_sentinel(path: Path) -> dict:
    d = json.loads(path.read_text())
    sub = d.get("submit") or {}
    sc = sub.get("score") if isinstance(sub.get("score"), dict) else {}
    return {
        "agent": "sentinel", "puzzle_id": d.get("challenge", {}).get("puzzle_id", ""), "modality": d.get("modality"),
        "accepted": sub.get("accepted"), "final": sc.get("final_score"), "quality": sc.get("quality_score"),
        "speed": sc.get("speed_score"), "tools": sc.get("tools_score", sc.get("tool_score")),
        "tokens": sc.get("tokens_score", sc.get("token_score")), "elapsed_s": d.get("timings", {}).get("total_s"),
        "model": d.get("model"), "missing_required": ";".join(sc.get("missing_required_tools") or d.get("missing_required_tools") or []),
        "answer": str(d.get("answer", ""))[:120], "notes": "; ".join(d.get("notes", []))[:200],
    }


def _row_from_baseline(path: Path) -> dict:
    d = json.loads(path.read_text())
    res = d.get("result") or {}
    sc = res.get("score") if isinstance(res.get("score"), dict) else {}
    ch = d.get("challenge") or {}
    return {
        "agent": "python_simple", "puzzle_id": ch.get("puzzle_id", ""), "modality": "image" if d.get("image_tool_sequence") else "text",
        "accepted": res.get("accepted"), "final": sc.get("final_score"), "quality": sc.get("quality_score"),
        "speed": sc.get("speed_score"), "tools": sc.get("tools_score", sc.get("tool_score")),
        "tokens": sc.get("tokens_score", sc.get("token_score")), "elapsed_s": (d.get("total_time_ms") or 0) / 1000,
        "model": d.get("model"), "missing_required": ";".join(sc.get("missing_required_tools") or []),
        "answer": str(d.get("answer", ""))[:120], "notes": "",
    }


def _append(row: dict) -> None:
    CSV.parent.mkdir(exist_ok=True)
    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def summary() -> None:
    if not CSV.exists():
        print("No results yet.")
        return
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups[(r["agent"], r["version"], r["puzzle_id"])].append(r)
    print(f"{'agent':14} {'version':8} {'puzzle':40} {'n':>3} {'acc%':>5} {'final':>6} {'qual':>6} {'speed':>6}")
    for (agent, ver, pid), rows in sorted(groups.items()):
        def avg(k):
            vals = [float(r[k]) for r in rows if r[k] not in ("", "None")]
            return f"{statistics.mean(vals):6.1f}" if vals else "     -"
        acc = 100 * sum(r["accepted"] == "True" for r in rows) / len(rows)
        print(f"{agent:14} {ver:8} {pid[:40]:40} {len(rows):3} {acc:5.0f} {avg('final')} {avg('quality')} {avg('speed')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pause", type=float, default=20.0)
    ap.add_argument("--baseline", action="store_true", help="run the stock examples/python_simple agent")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return 0
    agent_id = _env_agent_id()
    if not agent_id:
        print("Set AGENT_ID in .env first.")
        return 2
    sys.path.insert(0, str(ROOT))
    from sentinel import __version__

    script = "examples/python_simple/agent.py" if args.baseline else "examples/sentinel/agent.py"
    for i in range(1, args.runs + 1):
        print(f"\n===== run {i}/{args.runs}: {script} =====", flush=True)
        started = time.time() - 1
        proc = subprocess.run([sys.executable, script], cwd=ROOT, text=True, capture_output=True, timeout=600)
        print(proc.stdout[-1500:])
        if proc.returncode:
            print(proc.stderr[-1500:])
        if args.baseline:
            path = _latest(str(ROOT / "outputs" / f"{agent_id}-run.json"), started)
            row = _row_from_baseline(path) if path else {"agent": "python_simple", "notes": "no run record"}
        else:
            path = _latest(str(ROOT / "outputs" / agent_id / "*.json"), started)
            row = _row_from_sentinel(path) if path else {"agent": "sentinel", "notes": "no run record"}
        row.update(ts=time.strftime("%Y-%m-%d %H:%M:%S"), version="base" if args.baseline else __version__)
        _append(row)
        print(f"-> {row.get('puzzle_id')} accepted={row.get('accepted')} final={row.get('final')}")
        if i < args.runs:
            time.sleep(args.pause)
    print()
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
