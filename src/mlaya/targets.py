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

from .perception import (GROUND_NAMES, RESCUE_GROUND_NAMES, RESCUE_Y_BAND, BlockHit,
                         GameState)

# Queries that give a walkable destination. Stone counts as ground; logs do not.
DESTINATION_QUERIES = frozenset(GROUND_NAMES) | {"stone"}
# `explore` means "go somewhere else", so the pool is filtered to ground that is not
# already underfoot, then sorted farthest-first. `view_blocks find` lists the nearest
# hits of a query, so this is a hop toward open terrain, not a cross-map trek.
EXPLORE_MIN_M = 4.0
EXPLORE_POOL = 8
# Chunk columns one `explore_frontier` hop extends the map boundary by. Two is a real step
# outward without committing the whole tick to a long route.
EXPLORE_FRONTIER_CHUNKS = 2


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


def _ground_hits(gs: GameState) -> list[BlockHit]:
    """Distinct standable (x, z) columns from the destination queries, farthest first."""
    if not gs.resources:
        return []
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
    return hits


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
    x0 = gs.pos[0] if gs.pos else 0.0
    z0 = gs.pos[2] if gs.pos else 0.0
    hits = _ground_hits(gs)
    outward = [h for h in hits if h.distance >= min_distance]
    chosen = outward or hits
    out: list[Target] = []
    for hit in chosen[:limit]:
        direction = _compass(hit.x - x0, hit.z - z0)
        text = f"{hit.name} {hit.distance:.0f}m {direction}, open above"
        out.append(Target(label=f"{hit.name}@{hit.x},{hit.z}", text=text, hit=hit))
    return out


def boxed_in(gs: GameState, *, min_distance: float = EXPLORE_MIN_M) -> bool:
    """True when no standable ground lies at or beyond `min_distance`.

    `ground_targets` then offers only cells at the bot's own feet, and a `navigate` between two
    of those ping-pongs in place: live, the from-zero run spent 211 of 298 ticks stepping 0.8 m
    back and forth (`grass_block@-32,-68` <-> `@-33,-68`) while the portal chain waited on lava.
    Callers should extend the frontier (`explore_frontier`) rather than navigate a near cell.
    """
    return not any(h.distance >= min_distance for h in _ground_hits(gs))


def frontier_heading(gs: GameState) -> float:
    """Compass heading (0 = north, 90 = east) that pushes into ground not worked yet.

    Away from the home base, so repeated frontier pushes fan over fresh terrain instead of
    re-covering the walk home. North when no base is set or the bot stands on it.
    """
    if gs.base is None or gs.pos is None:
        return 0.0
    dx = gs.pos[0] - gs.base[0]
    dz = gs.pos[2] - gs.base[2]
    if dx == 0 and dz == 0:
        return 0.0
    return round(math.degrees(math.atan2(dx, -dz)) % 360.0, 1)


def base_target(gs: GameState) -> Target | None:
    """The nearest standable *surface* ground for the home base.

    Unlike `ground_targets` (farthest first, stone included, meant for wandering) this picks
    the closest real surface: only the ground queries, and only cells at or above the bot's
    own level, so the pick is the open ground overhead rather than a cave floor underfoot.
    With nothing at or above the bot it takes the highest ground below it -- the surface
    nearest its own level -- never the lowest.
    """
    hits = [h for query in GROUND_NAMES for h in gs.resources.get(query, []) if h.standable]
    if not hits:
        return None
    x0 = gs.pos[0] if gs.pos else 0.0
    z0 = gs.pos[2] if gs.pos else 0.0
    y0 = gs.pos[1] if gs.pos else 0.0
    above = [h for h in hits if h.y >= y0]
    if above:
        hit = min(above, key=lambda h: (h.y, h.distance))
    else:
        # The bot stands *above* every surface cell it has seen (it climbed a scaffold or a
        # hill slope). Falling back to `min` over each hit -- the lowest ground there is --
        # anchored the base to the bottom of a pit: live, run26, the bot stood on the surface
        # at y=64 beside a dug-out dip, the base landed at (229,61,223) two blocks under the
        # surrounding ground, and the portal frame, chest and furnace were then laid 1-2 blocks
        # into the hillside (the furnace alone dug a dirt block out to fit). Take the highest
        # ground instead, so the base stays at the surface.
        hit = max(hits, key=lambda h: (h.y, -h.distance))
    text = f"{hit.name} {hit.distance:.0f}m {_compass(hit.x - x0, hit.z - z0)}, open above"
    return Target(label=f"{hit.name}@{hit.x},{hit.z}", text=text, hit=hit)


def nearest_ground(gs: GameState) -> Target | None:
    """The closest dry foothold the water rescue can actually swim to.

    `base_target` looks only at or above the bot, so an underground base is not anchored to the
    cave floor underfoot; a bot treading water needs the opposite -- the dry cell it can actually
    reach may sit a block below the waterline -- so this may take a hit below the bot too.

    Three live lessons narrow "closest". A cell at the bot's **own waterline** is the one escape a
    swimmer completes, and it may be roofed: live run20, the cobblestone ledge the pour left beside
    the bot sat 0.3 m off with `cave_air` overhead, while the rescue aimed 34 blocks up at grass
    and every path stalled 20 s (see `perception.RESCUE_GROUND_NAMES`). A cell far above or below
    is no swim at all -- live run19b, the outright nearest cell was a cave floor 32 m **down**
    (`[NAVIGATION_STOPPED] ... closest node was -182,59,-85`). So a waterline cell wins outright,
    roofed (`BlockHit.rescue_standable`) or not; only open-sky cells fall back at any level, so a
    world with only a low dry ledge behaves as before. The pool is `gs.rescue_ground`, read to
    `RESCUE_RADIUS` (live run19b: the shore that would have worked sat 65.1 m -- a whisker past
    the 64 m read). Two further narrowings come from run115: a cell the pathfinder already timed
    out at (`gs.rescue_skip`, recorded by `Agent._note_rescue`) is passed over, and with nothing at
    the waterline the nearest dry cell **above** the body is taken before one below it.
    """
    pool = gs.rescue_ground or gs.resources
    hits = [h for query in RESCUE_GROUND_NAMES
            for h in pool.get(query, [])
            if h.rescue_standable and (h.x, h.y + 1, h.z) not in gs.rescue_skip]
    if not hits:
        return None
    y0 = gs.pos[1] if gs.pos else 0.0
    # A roofed cell counts only at the waterline; at any other level it is the run19b dive.
    candidates = [h for h in hits if abs(h.y - y0) <= RESCUE_Y_BAND]
    if not candidates:
        # Nothing at the waterline, so the choice is between climbing and diving. A body treading
        # water climbs: live run115, 281 rescue walks from (-0.5,62,281) aimed 29 blocks down at a
        # cave floor, every one `[NAVIGATION_STOPPED] ... closest node was -1,59,281` -- three
        # blocks under the feet, twenty-nine above the target. Take the nearest dry cell above the
        # waterline first; a cell below is only ever the last resort.
        candidates = [h for h in hits if h.y >= y0] or [h for h in hits if h.standable]
    if not candidates:
        return None
    hit = min(candidates, key=lambda h: (h.distance, h.y))
    x0 = gs.pos[0] if gs.pos else 0.0
    z0 = gs.pos[2] if gs.pos else 0.0
    cover = "open above" if hit.standable else "under rock"
    text = f"{hit.name} {hit.distance:.0f}m {_compass(hit.x - x0, hit.z - z0)}, {cover}"
    return Target(label=f"{hit.name}@{hit.x},{hit.z}", text=text, hit=hit)


# Cells that are neither solid footing nor room to place into: water and lava would spoil a
# frame cell, and a bottom row over them would not hold the frame.
_NOT_SOLID = frozenset({"air", "cave_air", "void_air", "water", "lava"})
# A cell a block may be placed into: any air variant, or the thin plants placement replaces
# without a separate dig. Used by `_clear` for the portal opening, where enclosed air is fine.
_REPLACEABLE = frozenset({"air", "cave_air", "void_air", "short_grass", "tall_grass",
                          "grass", "fern", "snow", "torch"})
# The same, minus the enclosed air variants: a *home base* has to stand under open sky, so
# `flat_build_site` uses this for the frame volume and rejects a cave floor. Admitting `cave_air`
# there is what let the site scan anchor the base to a cave (live run8: the base landed at y=-9
# while the bot was still 70 m underground, and every later base step -- the water rescue
# included -- was then aimed straight down). The base criterion is "open surface ground".
_OPEN_SKY = frozenset({"air", "short_grass", "tall_grass",
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
    ground under all five cells of its bottom row and replaceable **open-sky** air above them up
    to the top row (see `_OPEN_SKY`: `cave_air` is not accepted, so a cave floor never becomes
    the home base); a site on a slope leaves those cells holding stone, which the build server
    will not dig. Picking level ground up front is what keeps the frame build from stalling --
    live, the auto-base landed in a dip and 15 of 20 frame cells stayed stone.

    The ground layer *below* the bottom row must be solid too. The server digs the bottom row
    out and re-places it as obsidian, so a cell with a hole beneath it has "nothing to place
    against": that frame cell is left wrong and the opening never closes (live, Nether run: a
    one-block hole under the bottom row left a frame `light_portal` refused as incomplete).

    Candidates are ranked by squared distance to the bot (cells and level together), so the
    walk is short and the base lands at the bot's own height rather than on a ledge or in a pit.
    Ranking alone is too weak a promise, though: in ground dug over by earlier runs no row at
    the bot's own height fits any more, and the pick then dropped to the floor of the scan box.
    A candidate more than one block below the bot's feet is therefore no candidate at all: that
    is a pit floor, not the surface the bot stands on (its own footing is `fy - 1`, and
    `_at_surface` proved that is open ground -- live, run26: the bot stood at y=64, the pick was
    the floor of a two-deep dip at y=61, and the frame, chest and furnace went in 1-2 blocks
    under the surrounding ground).
    """
    if not grid:
        return None
    fx, fy, fz = (int(math.floor(v)) for v in feet)
    lowest = fy - 1  # the bot's own footing block: level with the surface it stands on
    sites: list[tuple[int, int, int]] = []
    for (bx, by, bz), name in grid.items():
        if by < lowest:  # below the surface underfoot: a pit floor, not the base site
            continue
        if name in _NOT_SOLID:  # the anchor itself must be real ground
            continue
        row = range(bx, bx + FRAME_WIDTH)
        if not all(_solid(grid.get((x, by, bz))) for x in row):
            continue
        if not all(_solid(grid.get((x, by - 1, bz))) for x in row):
            continue  # a hole below the bottom row gives the obsidian nothing to sit on
        clear = True
        for x in row:
            for dy in range(1, FRAME_HEIGHT):
                if not _open_sky(grid.get((x, by + dy, bz))):
                    clear = False
                    break
            if not clear:
                break
        if clear:
            sites.append((bx, by, bz))
    if not sites:
        return None
    return min(sites, key=lambda a: (a[0] + 2 - fx) ** 2 + (a[2] - fz) ** 2 + (a[1] - fy) ** 2)


def bed_frame_site(grid: dict[tuple[int, int, int], str], bed) -> bool:
    """Whether the ground a standing bed anchors can carry the portal frame, bed included.

    `_adopt_homebase` takes a bed the world already holds as the home base, and it has to be as
    choosy as `establish_base`: `flat_build_site` proves the frame fits, but it demands the
    *anchor* column be clear, and the bed stands exactly there. This applies the same criteria to
    the frame's own four columns -- solid bottom row and ground beneath it, open-sky air up to the
    top row -- and asks only that the bed's own column be real ground. A bed the run laid on a
    cave floor or a slope is refused (its roof or step leaves frame cells the build server will not
    dig); live, run17: the base was pinned nine blocks under the surface and the frame's top and
    western cells stayed stone, so `light_portal` answered `[PORTAL_NO_FRAME]` on 150 ticks. The
    run then falls back to `establish_base` and anchors a site the frame can stand on.

    `bed` is a `perception.BlockHit` for the bed block; its ground cell is the base anchor.
    """
    if not grid:
        return False
    bx, by, bz = int(bed.x), int(bed.y) - 1, int(bed.z)
    if not _solid(grid.get((bx, by, bz))) or not _solid(grid.get((bx, by - 1, bz))):
        return False  # the bed's own footing
    for x in range(bx + 1, bx + FRAME_WIDTH):
        if not _solid(grid.get((x, by, bz))):
            return False
        if not _solid(grid.get((x, by - 1, bz))):
            return False  # a hole below the bottom row gives the obsidian nothing to sit on
        for dy in range(1, FRAME_HEIGHT):
            if not _open_sky(grid.get((x, by + dy, bz))):
                return False
    return True


# How far the walk off an open drop goes. A toss lands up to two blocks out in whatever way the
# bot faced and the pickup reaches about one more, so a cell four blocks from where the bot stood
# is clear of the junk whichever way it flew.
STEP_AWAY_MIN = 4.0
# Footing that is solid but no place to stand on.
_HAZARD_FLOOR = frozenset({"magma_block", "cactus", "campfire", "soul_campfire", "fire",
                           "soul_fire", "powder_snow", "sweet_berry_bush"})


def step_away_cell(grid: dict[tuple[int, int, int], str],
                   drop: tuple[float, float, float],
                   feet: tuple[float, float, float] | None,
                   refused=frozenset(),
                   heading: tuple[float, float] | None = None) -> tuple[int, int, int] | None:
    """A feet cell the walk off an open drop can actually stand in, or None.

    `grid` is one `view_blocks box` read around the drop (`perception.parse_box`). A cell
    qualifies with solid footing under it and room for the body (feet and head clear), at least
    `STEP_AWAY_MIN` from the drop and within two levels of it. The old walk aimed three blocks
    "away" from a drop that lay under the bot's own feet -- a zero vector, so always due west --
    at the bot's own height, and run86 answered `[NAVIGATION_TARGET_UNSUPPORTED] No usable
    footing ... of -93, 70, -6` nine times for a cell nobody had looked at. Cells a walk was
    already refused at (`refused`) are passed over. The nearest such cell to the bot wins, on its
    own level first.

    `heading` is the way the bot last walked (x, z), which is the way it faced when it tossed, so
    the junk lies ahead of the drop. A cell behind it wins over any cell ahead: run87 dug west
    along a tunnel, tossed west, and the tie between the cells four blocks east and west went to
    the lower x - west, through the pile - so four rounds of drop and step-away picked the same
    junk straight back up (it171-176). Without a heading every cell counts alike.
    """
    if not grid:
        return None
    dx0, dy0, dz0 = drop
    fx, fy, fz = feet if feet is not None else drop
    cells: list[tuple[float, float, tuple[int, int, int]]] = []
    for (x, y, z), name in grid.items():
        if not _clear(name) or not _clear(grid.get((x, y + 1, z))):
            continue
        floor = grid.get((x, y - 1, z))
        if not _solid(floor) or floor in _HAZARD_FLOOR:
            continue
        if abs(y - math.floor(dy0)) > 2 or (x, y, z) in refused:
            continue
        away = math.hypot(x + 0.5 - dx0, z + 0.5 - dz0)
        if away < STEP_AWAY_MIN:
            continue
        walk = (x + 0.5 - fx) ** 2 + (z + 0.5 - fz) ** 2 + 2 * (y - math.floor(fy)) ** 2
        ahead = (heading is not None
                 and (x + 0.5 - dx0) * heading[0] + (z + 0.5 - dz0) * heading[1] > 0)
        cells.append((ahead, walk, -away, (x, y, z)))
    if not cells:
        return None
    return min(cells)[3]


def _solid(name: str | None) -> bool:
    return name is not None and name not in _NOT_SOLID


def _clear(name: str | None) -> bool:
    return name in _REPLACEABLE


def _open_sky(name: str | None) -> bool:
    """Whether a cell is replaceable *and* under open sky (no enclosed `cave_air` above the base)."""
    return name in _OPEN_SKY
