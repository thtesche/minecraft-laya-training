from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from . import perception
from .actions import candidates, resolve
from .config import ROOT, load_env, load_settings
from .graph import Agent
from .mcp import MCPExecutor


async def _health(settings) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(settings.mcp.health_url)
        r.raise_for_status()
        return r.json()


async def cmd_probe(args) -> int:
    settings = load_settings()
    key = settings.api_key()
    env_file = "yes" if (ROOT / ".env").exists() else "no"
    print(f"env  .env={env_file}  {settings.llm.api_key_env}="
          f"{'set' if key else 'MISSING (System-2 escalation disabled)'}")
    health = await _health(settings)
    mc = health.get("minecraft", {})
    print(f"health ok  transport={health.get('transport')}  bot={mc.get('username')} "
          f"pos={mc.get('position')} food={mc.get('food')} busy={mc.get('busy')}")
    async with MCPExecutor(settings.mcp) as executor:
        tools = await executor.list_tools()
        print(f"tools: {len(tools)}")
        for t in tools:
            print("  -", t.name)
    return 0


async def cmd_decide(args) -> int:
    """Perceive and decide once, without executing a world action."""
    settings = load_settings()
    await _health(settings)
    from .brain import Brain

    async with MCPExecutor(settings.mcp) as executor:
        gs = await perception.build(executor)
        cands = candidates(gs)
        decision = Brain(settings.laya).decide(gs.narrative(), cands)
        action = resolve(decision.action_label, gs)
    print(json.dumps({
        "narrative": gs.narrative(),
        "state": gs.as_state(),
        "candidates": cands,
        "deferred": decision.deferred,
        "note": decision.note,
        "label": decision.action_label,
        "answers": decision.answers,
        "resolved_tool": action.tool if action else None,
        "resolved_args": action.args if action else None,
        "dry_run": "no action executed",
    }, indent=2, ensure_ascii=False, default=str))
    return 0


async def cmd_run(args) -> int:
    settings = load_settings()
    iterations = getattr(args, "iterations", 0) or settings.loop.max_iterations
    settings.loop.max_iterations = iterations
    await _health(settings)
    async with MCPExecutor(settings.mcp) as executor:
        agent = Agent(settings, executor)
        final = await agent.run()
    print(json.dumps({
        "iterations": final.get("iteration"),
        "note": final.get("note"),
        "log": str(settings.log_path()),
    }, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser("mlaya")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("probe", help="health check + tools/list")
    sub.add_parser("decide", help="perceive + decide once (dry run, no action)")
    run = sub.add_parser("run", help="run the decision loop")
    run.add_argument("-n", "--iterations", type=int, default=0)
    args = parser.parse_args(argv)
    cmd = args.cmd or "run"
    if cmd == "probe":
        return asyncio.run(cmd_probe(args))
    if cmd == "decide":
        return asyncio.run(cmd_decide(args))
    return asyncio.run(cmd_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
