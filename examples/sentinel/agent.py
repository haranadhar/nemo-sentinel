#!/usr/bin/env python3
"""NemoSentinel competitor agent entrypoint.

Usage (from the repository root or this directory):
    python examples/sentinel/agent.py

It reuses the organizer's shared clients in ``arena_clients/`` unchanged
(registration, MCP SSE, proxy attribution headers, telemetry) and replaces the
solving logic with the ``sentinel`` package.

Environment: ARENA_SERVER, ARENA_API_KEY, AGENT_ID (from .env locally; set by
the organizer's runner in the sandbox, where no .env exists).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # the sandbox sets the environment directly
    pass

from arena_clients import (
    HttpArenaClient,
    McpArenaClient,
    ensure_connected,
    get_api_base,
    get_arena_api_key,
    get_mcp_url,
    get_proxy_host,
    monitor_session,
)
from arena_clients.config import agent_id_problem
from arena_clients.proxy_headers import build_proxy_headers, resolve_usage_scope
from model_selector import cached_model_rows, fetch_available_models
from my_strategy import MyStrategy
from openai import OpenAI
from sentinel.llm import LLM
from sentinel.runtime import ImageRun, Ports, TextRun, save_report


def _log(msg: str) -> None:
    print(msg, flush=True)


async def main() -> int:
    strategy = MyStrategy()
    agent_id = (os.getenv("AGENT_ID") or strategy.agent_id).strip()
    agent_name = (os.getenv("AGENT_NAME") or strategy.agent_name).strip()
    os.environ["AGENT_ID"] = agent_id
    # Practice is shared: a default id lets another team's run replace yours.
    id_problem = agent_id_problem(os.environ["AGENT_ID"])
    if id_problem:
        raise SystemExit(f"❌ {id_problem}")
    _log(f"🛡️  {agent_name} ({agent_id}) starting")

    ensure_connected()
    api_base, mcp_url, llm_host, api_key = get_api_base(), get_mcp_url(), get_proxy_host(), get_arena_api_key()
    http = HttpArenaClient(api_base=api_base, api_key=api_key)

    # Register exactly once per process. After GO, never register again
    # (registration_too_late is terminal); reconnect transports instead.
    session = http.register(agent_id, agent_name)
    _log(f"   Registered: session={session.session_id} status={session.status}")
    monitor = monitor_session(http, agent_id).start()

    report = None
    try:
        report = await _play(http, mcp_url, llm_host, api_key, agent_id, strategy)
    finally:
        await monitor.stop()
    if report is not None:
        path = save_report(report, REPO_ROOT)
        sub = report.submit or {}
        score = sub.get("score") if isinstance(sub.get("score"), dict) else {}
        _log("\n📊 Result")
        _log(f"   Answer:    {report.answer}")
        _log(f"   Accepted:  {sub.get('accepted')}")
        _log(f"   Status:    {sub.get('status')}")
        if score:
            for k in ("final_score", "quality_score", "speed_score", "tools_score", "tokens_score"):
                if k in score:
                    _log(f"   {k}: {score[k]}")
            if score.get("missing_required_tools"):
                _log(f"   ⚠️ missing_required_tools: {score['missing_required_tools']}")
            if score.get("broadcast_penalty_applied"):
                _log("   ⚠️ broadcast penalty applied")
        if sub.get("error"):
            _log(f"   ❌ submit error: {sub['error']}")
        _log(f"   Report:    {path}")
    return 0


async def _play(http, mcp_url, llm_host, api_key, agent_id, strategy):
    # One reconnect attempt for the MCP transport (same identity, no re-register).
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with McpArenaClient(mcp_url) as mcp:
                tool_defs = await mcp.list_tool_defs()
                tools = [t.name for t in tool_defs]
                _log(f"   Discovered {len(tools)} MCP tools")
                modality = McpArenaClient.detect_modality(tools, agent_id=agent_id)
                _log(f"   Modality: {modality}")

                usage_scope = http.fetch_usage_scope() or resolve_usage_scope()
                if usage_scope:
                    os.environ["ARENA_USAGE_SCOPE"] = usage_scope
                client = OpenAI(
                    base_url=llm_host,
                    api_key=api_key,
                    default_headers=build_proxy_headers(agent_id, usage_scope),
                    max_retries=1,
                )
                models = fetch_available_models(llm_host, api_key)
                _log(f"   Models: {', '.join(models[:10])}{' ...' if len(models) > 10 else ''}")
                ports = Ports(
                    http=http,
                    mcp=mcp,
                    llm=LLM(client),
                    agent_id=agent_id,
                    strategy=strategy,
                    models=models,
                    model_rows=cached_model_rows(),
                    log=_log,
                )
                run = ImageRun(ports, tool_defs) if modality == "image" else TextRun(ports, tool_defs)
                return await run.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # transport failure: reconnect once
            last_exc = exc
            _log(f"   ⚠️  MCP session error ({type(exc).__name__}: {exc}); reconnecting" if attempt == 0 else f"   ❌ {exc}")
            await asyncio.sleep(1.0)
    if last_exc:
        raise last_exc
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except asyncio.CancelledError:
        raise SystemExit(0)
