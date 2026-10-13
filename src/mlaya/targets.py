"""M4 parameter layer: choose a concrete target from the `view_blocks` candidates.

A label says *what kind* of step to take ("explore"); it does not say *where*. The
parameter layer turns the grounded block hits the perception already collected into a
small, ordered candidate list and lets Laya pick one -- the same "code filters, Laya
chooses" split as the label layer, just one level down.

The strongest use is navigation: the server annotates every `view_blocks` hit with
`above`/`below`, and `above == "air"` is real standing ground, which the run skill calls
the reliable way to choose a `navigate` target. So a destination is never guessed from a
fixed offset; it is one of the actual standable cells in the loaded chunks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .perception import GROUND_NAMES, BlockHit, GameState

# Queries that give a walkable destination. Stone counts as ground; logs do not.
DESTINATION_QUERIES = frozenset(GROUND_NAMES) | {"stone"}
# `explore` means "go somewhere else", so the pool is filtered to ground that is not
# already underfoot, then sorted farthest-first. `view_blocks find` lists the nearest
# hits of a query, so this is a hop toward open terrain, not a cross-map trek.
EXPLORE_MIN_M = 4.0
EXPLORE_POOL = 8


@dataclass(frozen=True)
class Target:
    """One selectable destination: the label/text Laya sees plus the grounded hit."""

    label: str
    text: str
    hit: BlockHit

    @property
    def pos(self) -> tuple[int, int, int]:
        return self.hit.pos

    @property
    def name(self) -> str:
        return self.hit.name


def _compass(dx: float, dz: float) -> str:
    """8-point compass for an offset. North is -z (Minecraft's convention)."""
    if dx == 0 and dz == 0:
        return "here"
    angle = math.degrees(math.atan2(dx, -dz)) % 360.0
    points = ("north", "north-east", "east", "south-east",
              "south", "south-west", "west", "north-west")
    return points[int(round(angle / 45.0)) % 8]


def ground_targets(gs: GameState, *, limit: int = EXPLORE_POOL,
                   min_distance: float = EXPLORE_MIN_M) -> list[Target]:
    """Standable destinations from the perceived block hits, farthest first.

    Each candidate is a distinct (x, z) column whose block has open air above it, drawn
    from the ground queries plus stone. Ground already underfoot (< `min_distance`) is
    dropped when farther ground exists, so an `explore` pick is a real step outward; when
    the bot is boxed in, the nearest ground is offered instead of nothing. Descriptions
    stay short because Laya shares one `head_max_len` token budget across every option and
    the brain shortlists a larger pool.
    """
    if not gs.resources:
        return []
    x0 = gs.pos[0] if gs.pos else 0.0
    z0 = gs.pos[2] if gs.pos else 0.0
    seen: set[tuple[int, int]] = set()
    hits: list[BlockHit] = []
    for query, group in gs.resources.items():
        if query not in DESTINATION_QUERIES:
            continue
        for hit in group:
            if not hit.standable:
                continue
            column = (hit.x, hit.z)
            if column in seen:
                continue
            seen.add(column)
            hits.append(hit)
    hits.sort(key=lambda h: h.distance, reverse=True)
    outward = [h for h in hits if h.distance >= min_distance]
    chosen = outward or hits
    out: list[Target] = []
    for hit in chosen[:limit]:
        direction = _compass(hit.x - x0, hit.z - z0)
        text = f"{hit.name} {hit.distance:.0f}m {direction}, open above"
        out.append(Target(label=f"{hit.name}@{hit.x},{hit.z}", text=text, hit=hit))
    return out


def base_target(gs: GameState) -> Target | None:
    """The nearest standable *surface* ground for the home base.

    Unlike `ground_targets` (farthest first, stone included, meant for wandering) this picks
    the closest real surface: only the ground queries, and only cells at or above the bot's
    own level, so the pick is the open ground overhead rather than a cave floor underfoot.
    """
    hits = [h for query in GROUND_NAMES for h in gs.resources.get(query, []) if h.standable]
    if not hits:
        return None
    x0 = gs.pos[0] if gs.pos else 0.0
    z0 = gs.pos[2] if gs.pos else 0.0
    y0 = gs.pos[1] if gs.pos else 0.0
    above = [h for h in hits if h.y >= y0] or hits
    hit = min(above, key=lambda h: (h.y, h.distance))
    text = f"{hit.name} {hit.distance:.0f}m {_compass(hit.x - x0, hit.z - z0)}, open above"
    return Target(label=f"{hit.name}@{hit.x},{hit.z}", text=text, hit=hit)


# Cells that are neither solid footing nor room to place into: water and lava would spoil a
# frame cell, and a bottom row over them would not hold the frame.
_NOT_SOLID = frozenset({"air", "cave_air", "void_air", "water", "lava"})
# A cell a frame block may be placed into: air, or the thin plants placement replaces
# without a separate dig.
_REPLACEABLE = frozenset({"air", "cave_air", "void_air", "short_grass", "tall_grass",
                          "grass", "fern", "snow", "torch"})

# Frame geometry of `build_structure`'s `portal_frame`: it takes the *interior corner*, so a
# frame whose base anchor is the ground block (bx, by, bz) stands as a west-east plane at
# z = bz -- bottom row (bx+1..bx+4, by), interior (bx+2..bx+3, by+1..by+3), top row
# (bx+1..bx+4, by+4). The bot stands at (bx, by+1, bz), just west of the western face.
FRAME_WIDTH = 5   # the anchor cell plus the 4 cells of the bottom row
FRAME_HEIGHT = 5  # the 5 structural layers from the bottom row up to the top row


def flat_build_site(grid: dict[tuple[int, int, int], str],
                    feet: tuple[float, float, float]) -> tuple[int, int, int] | None:
    """A level, clear surface cell near the bot where the portal frame fits, or None.

    `grid` is one `view_blocks box` read (see `perception.parse_box`). The frame needs solid
    ground under all five cells of its bottom row and replaceable air above them up to the top
    row; a site on a slope leaves those cells holding stone, which the build server will not
    dig. Picking level ground up front is what keeps the frame build from stalling -- live, the
    auto-base landed in a dip and 15 of 20 frame cells stayed stone.

    Candidates are ranked by squared distance to the bot (cells and level together), so the
    walk is short and the base lands at the bot's own height rather than on a ledge or in a pit.
    """
    if not grid:
        return None
    fx, fy, fz = (int(math.floor(v)) for v in feet)
    sites: list[tuple[int, int, int]] = []
    for (bx, by, bz), name in grid.items():
        if name in _NOT_SOLID:  # the anchor itself must be real ground
            continue
        row = range(bx, bx + FRAME_WIDTH)
        if not all(_solid(grid.get((x, by, bz))) for x in row):
            continue
        clear = True
        for x in row:
            for dy in range(1, FRAME_HEIGHT):
                if not _clear(grid.get((x, by + dy, bz))):
                    clear = False
                    break
            if not clear:
                break
        if clear:
            sites.append((bx, by, bz))
    if not sites:
        return None
    return min(sites, key=lambda a: (a[0] + 2 - fx) ** 2 + (a[2] - fz) ** 2 + (a[1] - fy) ** 2)


def _solid(name: str | None) -> bool:
    return name is not None and name not in _NOT_SOLID


def _clear(name: str | None) -> bool:
    return name in _REPLACEABLE
