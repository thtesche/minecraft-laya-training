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
