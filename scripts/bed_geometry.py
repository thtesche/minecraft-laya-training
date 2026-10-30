"""Live read of the gate bed's geometry: which cells it occupies and what stands on it.

run122 opens every night at the gate bed and answers `[BED_SLEEP_REJECTED] bot is not sleeping`
while the archived runs slept 334 times. Before guessing at the click box this reads the ground
truth: the bed cells `view_blocks find` reports, the block in every cell around and above them, and
where the body stands.

Run it while a run is live: `python scripts/bed_geometry.py`. Read-only.
"""
from __future__ import annotations

import asyncio

from mlaya.config import load_settings
from mlaya.mcp import MCPExecutor
from mlaya.perception import parse_box, parse_resources

BASE = (-10, 69, -52)
HALF = 4


def result(res):
    sc = getattr(res.raw, "structuredContent", None) or getattr(res.raw, "structured_content", None)
    return ((sc or {}).get("response", {}) or {}).get("data", {}).get("result", {})


async def main() -> None:
    s = load_settings()
    async with MCPExecutor(s.mcp) as ex:
        res = await ex.read("view_blocks", {"find": {"block_names": ["white_bed", "bed"], "limit": 8}},
                            "find the gate bed")
        for hit in parse_resources(result(res), 200):
            print(f"bed hit {hit.name} at ({hit.x},{hit.y},{hit.z}) d={hit.distance} "
                  f"above={hit.above} below={hit.below}")

        bx, by, bz = BASE
        res = await ex.read("view_blocks",
                            {"box": {"x": bx, "y": by + 2, "z": bz, "half_width": HALF, "half_height": 2}},
                            "read the base footprint around the bed")
        grid = parse_box(result(res))
        for (x, y, z) in sorted(grid, key=lambda c: (c[1], c[0], c[2])):
            name = grid[(x, y, z)]
            mark = "*" if name in ("white_bed", "bed") else " "
            print(f"  {mark} {x},{y},{z}: {name}")

        st = result(await ex.read("view_status", {"rationale": "where is the body"}, "read the stance"))
        sit = st.get("situation", {})
        print("body", sit.get("position"), "clock", sit.get("clock"))


if __name__ == "__main__":
    asyncio.run(main())
