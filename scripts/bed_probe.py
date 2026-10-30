"""Live probe: what exactly does the server say when a night sleep at the base bed fails?

run121 refused 32 `[BED_SLEEP_REJECTED] bot is not sleeping` ticks at the base. Some stances sit
outside vanilla's two-block bed-click box; others, e.g. (-10.44, 71, -49.46), are 0.89 squared from
the bed's top centre - well inside it - and still fail. mineflayer activates whichever bed block it
finds, and vanilla resolves a bed click on the head half, or on the foot only when the player faces
the head. This probe keeps the body where the run already stands, turns it through the four
cardinals, and prints the server's whole reply for each attempt so the refusal can be read rather
than guessed at.

Run it while a run is live: `python scripts/bed_probe.py`. It moves the head only; it never digs or
places.
"""
from __future__ import annotations

import asyncio
import json
import uuid

from mlaya.config import load_settings
from mlaya.mcp import MCPExecutor

BED = [(-10, 70, -51), (-10, 70, -50)]
YAWS = (0.0, 90.0, 180.0, 270.0)  # south, west, north, east


def payload(res):
    sc = getattr(res.raw, "structuredContent", None) or getattr(res.raw, "structured_content", None)
    return sc or {}


def data(res):
    return (payload(res).get("response", {}) or {}).get("data", {}).get("result", {})


async def situation(ex):
    res = await ex.read("view_status", {"rationale": "read the clock, the body and who owns it"},
                        "check whether the body is free and it is night")
    return data(res).get("situation", {})


async def idle(ex, tries: int = 8) -> bool:
    for _ in range(tries):
        if not (situation(ex).get("activity") or {}).get("activeAction"):
            return True
        await asyncio.sleep(4)
    return False


def line(res) -> str:
    out = data(res)
    keys = {k: v for k, v in out.items() if k not in ("observation", "situation")}
    return f"{res.status} | {res.text[:60]} | {json.dumps(keys, default=str)[:220]}"


async def main() -> None:
    s = load_settings()
    async with MCPExecutor(s.mcp) as ex:
        while True:
            clock = (await situation(ex)).get("clock") or {}
            if clock.get("phase") == "night":
                break
            print(f"phase {clock.get('phase')}, night in {clock.get('minutesUntilChange')} min", flush=True)
            await asyncio.sleep(30)

        for _round in range(2):
            if not await idle(ex):
                print("body busy, skipping round", flush=True)
                continue
            pos = (await situation(ex)).get("position") or {}
            gap = min((pos["x"] - (bx + 0.5)) ** 2 + (pos["y"] - (by + 1)) ** 2
                      + (pos["z"] - (bz + 0.5)) ** 2 for bx, by, bz in BED)
            print(f"round {_round}: body at {pos['x']},{pos['y']},{pos['z']} gap^2 {round(gap, 2)}",
                  flush=True)
            print(f"  heading {pos.get('headingDegrees')} -> {line(await ex.call('sleep', {}, 'sleep as the walk left the body facing'))}", flush=True)
            for yaw in YAWS:
                if not await idle(ex):
                    continue
                await ex.call("raw_action", {"operation": "look", "yaw": yaw, "pitch": 0.0,
                                            "submission_id": f"bedprobe-{uuid.uuid4()}",
                                            "rationale": "turn the body to a cardinal before the bed click"},
                              "test whether the facing direction decides the bed click")
                head = (await situation(ex)).get("position", {}).get("headingDegrees")
                print(f"  yaw {yaw} heading {head} -> {line(await ex.call('sleep', {}, 'sleep facing this way'))}", flush=True)
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
