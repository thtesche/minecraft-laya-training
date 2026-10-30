"""Live probe: the raw shape of a `sleep` call at the base bed.

run122 opens every night at the gate bed and answers `[BED_SLEEP_REJECTED] bot is not sleeping`,
while 334 sleeps in the archived runs succeeded. The refusal text alone does not say which of
mineflayer's checks the click ran into, so this probe prints the whole server envelope - the
payload, the clock's `sleeping` flag and the body's stance - for one attempt at the bed and one
turn of the four cardinals.

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
    return sc if sc is not None else {"raw": str(getattr(res.raw, "content", ""))[:400]}


async def status(ex):
    res = await ex.read("view_status", {"rationale": "read the clock, the body and who owns it"},
                        "check whether the body is free and it is night")
    return payload(res)


def sit(p):
    return ((p.get("response") or {}).get("data") or {}).get("result", {}).get("situation", {})


async def idle(ex, tries: int = 8) -> bool:
    for _ in range(tries):
        if not (sit(await status(ex)).get("activity") or {}).get("activeAction"):
            return True
        await asyncio.sleep(4)
    return False


async def main() -> None:
    s = load_settings()
    async with MCPExecutor(s.mcp) as ex:
        while True:
            clock = sit(await status(ex)).get("clock") or {}
            if clock.get("phase") == "night":
                break
            print(f"phase {clock.get('phase')}, night in {clock.get('minutesUntilChange')} min", flush=True)
            await asyncio.sleep(30)

        for _round in range(2):
            if not await idle(ex):
                print("body busy, skipping round", flush=True)
                continue
            st = sit(await status(ex))
            pos = st.get("position") or {}
            gap = min((pos["x"] - (bx + 0.5)) ** 2 + (pos["y"] - (by + 1)) ** 2
                      + (pos["z"] - (bz + 0.5)) ** 2 for bx, by, bz in BED)
            print(f"round {_round}: body {pos.get('x')},{pos.get('y')},{pos.get('z')} "
                  f"gap^2 {round(gap, 2)} heading {pos.get('headingDegrees')} "
                  f"clock={json.dumps(st.get('clock') or {}, default=str)[:120]}", flush=True)
            res = await ex.call("sleep", {}, "sleep as the walk left the body facing")
            print(f"  sleep -> {res.status} text={res.text[:80]!r} payload={json.dumps(payload(res), default=str)[:400]}", flush=True)
            for yaw in YAWS:
                if not await idle(ex):
                    continue
                await ex.call("raw_action", {"operation": "look", "yaw": yaw, "pitch": 0.0,
                                            "submission_id": f"bedprobe-{uuid.uuid4()}",
                                            "rationale": "turn the body to a cardinal before the bed click"},
                              "test whether the facing direction decides the bed click")
                head = (sit(await status(ex)).get("position") or {}).get("headingDegrees")
                res = await ex.call("sleep", {}, "sleep facing this way")
                print(f"  yaw {yaw} heading {head} -> {res.status} text={res.text[:80]!r} "
                      f"payload={json.dumps(payload(res), default=str)[:300]}", flush=True)
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
