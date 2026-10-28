from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

# view_status -> structured_content shape:
#   response.data.result.situation.{vitals,clock,position,inventory,tools,nearby,lastDeath,...}
# read_recent_events -> response.data.result.events[] (type/observedAt/summary/...)
# view_blocks -> response.data.result.blocks.find[] = {name, found, listed:[{x,y,z,name,distance,...}]}

# A lit portal block: the one kind a previous run's finished portal leaves standing, and the
# anchor a restart re-adopts its home base from (`parse_portal_corner`). A ruined portal has no
# `nether_portal` block (only obsidian, unlit), so a hit here is our own lit gate.
PORTAL_BLOCK = "nether_portal"
# Resources probed each tick to keep the progression decisions grounded in the world.
# The last three are the overworld -> Nether gate: the obsidian frame and the lit portal
# block say how far that chain has come, and a placed bed beside it is the run's end state.
# Eight is the most `view_blocks find` accepts, so this group sits exactly at the limit
# (the standable-ground group is scanned in a second call).
RESOURCE_NAMES = ("logs", "stone", "iron_ore", "coal_ore", "diamond_ore",
                  "obsidian", PORTAL_BLOCK, "white_bed")
# Hits requested when the loop looks for a lit portal anywhere loaded (`scan_portal_corner`).
# The opening is six cells and the radius filter is bypassed, so a dozen is ample to catch a
# portal well past `RESOURCE_RADIUS` while still bounding the read.
PORTAL_SCAN_LIMIT = 12
# Ground blocks probed for the M4 parameter layer: a `view_blocks` hit whose `above` is
# `air` is real standing ground -- the reliable way to choose a `navigate` target.
GROUND_NAMES = ("grass_block", "sand", "gravel", "dirt")
# The water rescue reads its own, wider cast. A pour on a cave lava pool leaves the bot standing
# on cobblestone, and a `find` for the four soil names above never lists that ledge: live run20,
# the escape cell sat 0.6 m away while the rescue aimed at grass 37 m off / 34 blocks up, so every
# `navigate` stalled for 20 s and the loop burned ~15 ticks before it broke free by luck. Only the
# rescue pool widens -- `gs.resources` keeps the soil-only view `explore` and `base_target` were
# built on. `view_blocks find` takes eight names, so the cast is the soil names plus the four
# rocks a miner digs through.
RESCUE_ROCK_NAMES = ("stone", "cobblestone", "deepslate", "andesite")
RESCUE_GROUND_NAMES = GROUND_NAMES + RESCUE_ROCK_NAMES
# The cell above a rescue foothold may be enclosed: a bot in a cave stands under `cave_air`,
# not open sky. `BlockHit.standable` stays open-sky only.
RESCUE_ABOVE = ("air", "cave_air")
# The blocks the Nether / home-base chain has to find precisely rather than read off a count.
# `lava` is where the carried water bucket is poured: obsidian is *made*, not dug (tunnelling to
# the natural y=-55 deposit is the live trap that left the bot swimming in its own flooded
# tunnel). `chest` lets a run that did not place the chest itself re-adopt one already standing
# at the base. `furnace` lets a run re-adopt the furnace it placed at the base (so cooking
# reuses it instead of placing and tearing a temporary one down on every call). `gold_ore` is
# the Nether-prep dig (the gold helmet that keeps piglins neutral): like the diamonds, a deposit
# deep under the surface counts through `resource_found` so the `collect_block` tunnel is still
# offered. Kept as its own `view_blocks find` group: that call takes at most eight names.
# `end_portal_frame` is the stronghold phase's landmark: it loads into the scan once the locate
# phase has walked the bot to the stronghold, and that is what hands the tick to the activation.
CHAIN_NAMES = ("lava", "chest", "furnace", "smoker", "gold_ore", "end_portal_frame")
# The Nether's own scan group, read *only* in the Nether: the bricks a fortress is built from are
# the one landmark that tells the bot a blaze spawner is plausibly near, and `spawner` names the
# cage itself. Blazes are hostile, so they only reach the state through `view_status` once one is
# actually loaded -- a fortress can sit hundreds of blocks off, and these names are what gives
# `explore` a heading to walk toward instead of wandering at random.
NETHER_CHAIN_NAMES = ("nether_bricks", "nether_brick_fence", "nether_brick_stairs", "spawner")
# How far the Nether group reads. `view_blocks.find` has no radius of its own - it returns the
# nearest hits of the asked kinds *anywhere loaded* - so this bound is the loop's, and 64 m threw
# away exactly what the group exists for: a fortress standing 70 m off was found by the server,
# listed in the answer, and dropped here, so `explore` marched past it on open lava sea. Four
# Nether runs never saw a fortress and this is the most likely reason. The bound is now wider than
# any plausible view distance, which is the honest shape for a landmark group: read what the world
# has loaded and let distance decide whether it is worth walking to, rather than deciding first.
FORTRESS_RADIUS = 256.0
# The server already tracks which chunks it has seen and which of them still border unobserved
# ground (`frontier_chunks.is_frontier`), joined against the live bot to a compass heading in
# `frontier_navigation`. The nearest such chunk is where the unexplored starts, which is the whole
# point of a search: the fortress bricks say what to walk to once it is in sight, this says where
# to walk to make it come in sight. Asking for it beats betting on an axis - the stripes were
# described east-west in one source and north-south in the next, and the loop had no measurement
# either way, because it never found a fortress to measure on.
FRONTIER_HEADING_SQL = ("SELECT heading_degrees FROM frontier_navigation "
                        "WHERE is_frontier = 1 ORDER BY distance_blocks ASC LIMIT 1")
# Groups whose nearest hit is worth naming in the narrative (Laya reads that text).
NAMED_RESOURCES = RESOURCE_NAMES + ("lava", "spawner", "nether_bricks", "nether_brick_fence", "nether_brick_stairs",
                                    "end_portal_frame")
RESOURCE_RADIUS = 64.0
# How far the water rescue looks for dry land. The shore it needs to swim to is the one
# reach the *last* live stranding missed: the nearest dry ground sat 65.1 m off, a whisker
# past `RESOURCE_RADIUS`, so the scan never listed it and the rescue aimed at the only
# standable cell it did see -- a cave floor 32 m under the bot, a dive no swim completes
# ("closest node was -182,59,-85"; live, run19b). Ground only: every other group keeps its
# 64 m read.
RESCUE_RADIUS = 128.0
# How far off the bot's own feet a rescue target may sit. A shore is at the waterline; a
# standable cell in this band is reachable by swimming, one far below is a flooded cave
# floor. `nearest_ground` prefers inside over outside, whichever is nearer.
RESCUE_Y_BAND = 4.0
RESOURCE_LIMIT = 4
# Per-group scan depth. The ore and log scans only want the nearest few cells, but two groups
# need the tail of the list: the lava group, because a pour cell is only recognisable when the
# *source* beside it is listed too (and that source need not be among the nearest cells), and
# the ground group, because `explore` drops ground already underfoot -- with only the nearest
# four hits every candidate sat within 4 m and `explore` navigated onto the block the bot was
# standing on, which stood it still for the rest of the run.
GROUND_LIMIT = 16
CHAIN_LIMIT = 16
# Carried inventory slots the server fills (main 27 + hotbar 9). A gathering action is refused
# with `[INVENTORY_FULL]` once all 36 are taken, so the loop treats "nearly full" as a reason to
# shed low-value blocks before it runs into that wall.
INVENTORY_SLOTS = 36
INVENTORY_NEAR_FULL = 33

# Equipped-armour slots, keyed by the `location` an inventory stack reports. These names
# mirror the `equip` destinations. `situation.tools.armour` lists armour the bot *carries*
# per class whether or not it is worn, so the worn set has to come from the stack location:
# an equipped piece moves to head/torso/legs/feet and leaves location main/hotbar.
ARMOR_SLOTS = {"head": "helmet", "torso": "chestplate", "legs": "leggings", "feet": "boots"}
# Off-hand slot: a shield (or anything else) carried there reports one of these locations. The
# hyphen form mirrors the `equip` destination; the others cover a server that normalises it.
OFFHAND_SLOTS = ("off-hand", "off_hand", "offhand")
_ARMOR_TIERS = ("netherite", "diamond", "iron", "chainmail", "golden", "leather")

# Event backlog drain. `read_recent_events` returns the OLDEST unread page (max 100),
# so a deep backlog must be walked to the present before fresh events arrive.
EVENT_PAGE = 100
EVENT_DRAIN_PAGES = 12
# Only events this recent drive a reaction; older backlog events are drained, not acted on.
EVENT_REACT_WINDOW_S = 120.0


def get(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def coerce(data: Any, text: str | None) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(text, str) and text.lstrip().startswith("{"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# The containers the bot has opened are kept by the server in its own SQLite database, and a read
# of that database is not a read of the world: the entry survives a death, a dimension change and
# a restart, and it answers from 300 blocks away, which no block scan can do. `view_blocks` is
# bound to the loaded chunks, so the portal and the bed a previous run left are invisible from
# across a map, and after dying in the Nether the loop re-based on open ground and started pouring
# a second portal while the lit gate and the chest it had filled stood 300 m off (live, run47:
# `establish_base` at (19,73,476) with 4 fresh obsidian and the chest still at (229,62,224)).
# This is the one anchor that survives the trip, so it is where the run walks first.
REMEMBERED_HOME_SQL = (
    "SELECT block_x, block_y, block_z FROM observed_containers "
    "WHERE block_name = 'chest' AND dimension = 'overworld' "
    "ORDER BY observed_at DESC LIMIT 1")

# Hunger floors, owned here because the narrative and the candidates must quote the same ones.
# `EAT_HUNGER` is the survival floor the reflex has always used: the bot eats below it to keep
# hunger off empty. `REGEN_HUNGER` is Minecraft's natural-regeneration threshold: health only
# regenerates while food is at least 18, so a wounded bot whose food sits in the 14-17 gap eats
# nothing and heals nothing until it is hit again. Hunger is 0-20.
EAT_HUNGER = 14.0
REGEN_HUNGER = 18.0

# What counts as food worth banking, for the reader that has to decide it from a chest snapshot.
# The same list `actions.EDIBLE` uses, kept here because the agent reads the chest before the
# candidate list exists; `actions` still gates on its own copy, and a test holds the two together.
EDIBLE_NAMES = frozenset({
    "bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_chicken",
    "cooked_mutton", "cooked_rabbit", "cooked_cod", "cooked_salmon", "carrot",
    "potato", "baked_potato", "beetroot", "melon_slice", "sweet_berries",
    "rotten_flesh", "dried_kelp", "cookie", "mushroom_stew", "beef", "porkchop",
    "chicken", "mutton", "rabbit", "cod", "salmon",
})


REMEMBERED_STOCK_SQL = (
    "SELECT block_x, block_y, block_z, contents_json FROM observed_containers "
    "WHERE block_name = 'chest' AND dimension = 'overworld' "
    "ORDER BY observed_at DESC LIMIT 1")


def parse_remembered_stock(payload: Any) -> tuple[tuple[int, int, int], dict[str, int]] | None:
    """What the newest chest last held, as counts per item, or None if there is nothing to read.

    The same database read as `parse_remembered_home` with one column more: the server records a
    full `contents_json` for every container it has seen (`contents: [{slot, item, count}]`), and
    the home query left it unread. That unread column is why the loop hunted with a full chest
    standing at its own base - `hunt_food` counts `_edible_count(inventory)`, the pack only, so
    food banked at home was invisible to it and every hunt looked like a shortfall.

    A read of the database is not a read of the world: it answers from the Nether, from 300
    blocks away, and after a restart, and it costs no `use_container` inspection on the tick. It
    is also a *snapshot from the last observation*, not a live count, which is the right shape
    here: what the chest held when the bot last touched it is exactly what the bot put there, and
    the loop is the only writer.
    """
    rows = get(payload, "response", "data", "result", "query", "rows")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    cells = row if isinstance(row, list) else [row]
    if len(cells) < 4:
        return None
    x, y, z = (_num(cells[0]), _num(cells[1]), _num(cells[2]))
    if x is None or y is None or z is None:
        return None
    raw = cells[3]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    counts: dict[str, int] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("item")
            n = _num(entry.get("count"))
            if isinstance(name, str) and name and n is not None and n > 0:
                counts[name] = counts.get(name, 0) + int(n)
    return (int(x), int(y), int(z)), counts


# The base's storage *network*: one chest fills up, the run sets another beside it, and the relief
# goes on shedding. A newest-first read could only ever see the chest the bot touched last, so
# provisions standing in the second chest stayed invisible to `take_stored_food` and the total
# that stops hunts was one container short of what is banked. This one lists every Overworld
# chest the server has seen opened; which of the cells belong to the run's network is a base-side
# question the loop applies afterwards - the database cannot know it.
REMEMBERED_STOCKS_SQL = (
    "SELECT block_x, block_y, block_z, contents_json FROM observed_containers "
    "WHERE block_name = 'chest' AND dimension = 'overworld' "
    "ORDER BY observed_at DESC")


def parse_remembered_stocks(payload: Any) -> list[tuple[tuple[int, int, int], dict[str, int]]]:
    """What every Overworld chest last held, as `[(cell, counts), ...]`, newest observation first.

    The rows of `parse_remembered_stock` without the `LIMIT 1`: an overflow chest the bot walked
    to once after placing it still says what its last deposit left in it, and that is where the
    banked provisions might stand. A chest never opened is simply not in the database yet - the
    deposit that first opens it is also the observation that lists it.
    """
    rows = get(payload, "response", "data", "result", "query", "rows")
    if not isinstance(rows, list):
        return []
    out: list[tuple[tuple[int, int, int], dict[str, int]]] = []
    for row in rows:
        cells = row if isinstance(row, list) else [row]
        if len(cells) < 4:
            continue
        x, y, z = (_num(cells[0]), _num(cells[1]), _num(cells[2]))
        if x is None or y is None or z is None:
            continue
        raw = cells[3]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                continue
        counts: dict[str, int] = {}
        if isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("item")
                n = _num(entry.get("count"))
                if isinstance(name, str) and name and n is not None and n > 0:
                    counts[name] = counts.get(name, 0) + int(n)
        out.append(((int(x), int(y), int(z)), counts))
    return out


def parse_remembered_home(payload: Any) -> tuple[int, int, int] | None:
    """The cell of the newest chest this bot has ever opened, or None if it has opened none.

    Newest first: `observed_at` is the last time the bot saw the chest, and the chest it kept
    filling is the one it saw last. No entry means a fresh world -- nothing has been stored, and
    there is nothing to walk back to, which is the ordinary case and not a failure.
    """
    rows = get(payload, "response", "data", "result", "query", "rows")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    cells = row if isinstance(row, list) else [row]
    if len(cells) < 3:
        return None
    x, y, z = (_num(cells[0]), _num(cells[1]), _num(cells[2]))
    if x is None or y is None or z is None:
        return None
    return (int(x), int(y), int(z))


async def remembered_home(executor) -> tuple[int, int, int] | None:
    """The remembered chest cell, or None. The read is the bot's database, not the world."""
    res = await executor.read(
        "query_bot_data",
        {"sql": REMEMBERED_HOME_SQL},
        "recall the chest this bot last kept, so a lost base can be walked back to",
    )
    return parse_remembered_home(coerce(res.data, res.text))


# The Nether side of the portal the last real crossing went through, as the server itself
# remembers it. `nether_arrival` is per-run: a run that starts with the bot already down there
# saw no crossing, and until run74 the loop recorded whatever cell the bot happened to stand
# on as its "arrival" - (-469,57,421), bare basalt with no portal and no obsidian, 250 blocks
# from the real gate, walked to and waited at until the bot died. The server writes a
# `player_dimension_change` event with the position it delivered the bot at for every crossing,
# and that read outlives the death, the restart and the distance: the same kind of anchor as
# REMEMBERED_HOME_SQL, and the one way out. The position is the cell the bot stood on when the
# gate let it go, which is where the exit portal stands.
REMEMBERED_GATE_SQL = (
    "SELECT CAST(json_extract(payload_json, '$.position.x') AS REAL), "
    "CAST(json_extract(payload_json, '$.position.y') AS REAL), "
    "CAST(json_extract(payload_json, '$.position.z') AS REAL) "
    "FROM events WHERE event_type = 'player_dimension_change' "
    "AND json_extract(payload_json, '$.to') = 'the_nether' "
    "ORDER BY observed_at DESC LIMIT 1")


def parse_remembered_gate(payload: Any) -> tuple[int, int, int] | None:
    """The cell the bot last stood on as the Nether let it go, or None if it never crossed."""
    rows = get(payload, "response", "data", "result", "query", "rows")
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    cells = row if isinstance(row, list) else [row]
    if len(cells) < 3:
        return None
    x, y, z = (_num(cells[0]), _num(cells[1]), _num(cells[2]))
    if x is None or y is None or z is None:
        return None
    return (math.floor(x), math.floor(y), math.floor(z))


async def remembered_gate(executor) -> tuple[int, int, int] | None:
    """The Nether side of the last crossing, or None. A read of memory, not of the world."""
    res = await executor.read(
        "query_bot_data",
        {"sql": REMEMBERED_GATE_SQL},
        "recall the Nether side of the last crossing, the way out of the Nether",
    )
    return parse_remembered_gate(coerce(res.data, res.text))


def _find_container(payload: Any) -> Any:
    """The innermost dict that carries a `contents` list, however the answer nested it.

    A settled action buries it under `output.result.container`, but the wrapper has changed shape
    before and the loop cannot afford to stop draining over a path. One dict wins: the one with
    both a `contents` list and a `blockName`, so a stray `contents` elsewhere cannot be mistaken
    for the window.
    """
    stack = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if isinstance(cur.get("contents"), list) and "blockName" in cur:
                return cur
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def parse_furnace_contents(payload: Any) -> dict[str, int]:
    """What a `use_container inspect` on a furnace reported, as item -> count.

    The drain is two calls and the second one needs this: `collect_smelted` issues an `inspect`
    when it holds nothing, and the answer's `container.contents` is the only place the loop ever
    learns what is in the furnace. Throwing it away left the step inspecting forever -- `_furnace_
    occupied` was cleared as if the look had emptied the cell, so `cook_food` met
    `[FURNACE_NOT_EMPTY]` again on the next tick and nothing ever moved (live, run50: five
    successful `collect_smelted`, no cooked food in the pack, no `withdraw` ever sent).

    A furnace window has three slots -- input, fuel, output -- so a name may repeat; the counts
    add rather than overwrite. An absent or unreadable container is an empty dict, which reads as
    "nothing in it" and makes the loop cook rather than drain, the safe way round.
    """
    contents = None
    window = _find_container(payload)
    if isinstance(window, dict):
        contents = window.get("contents")
    if not isinstance(contents, list):
        return {}
    out: dict[str, int] = {}
    for slot in contents:
        if not isinstance(slot, dict):
            continue
        name, count = slot.get("item"), slot.get("count")
        if not name or not isinstance(count, (int, float)) or isinstance(count, bool):
            continue
        out[str(name)] = out.get(str(name), 0) + int(count)
    return out


def parse_frontier_heading(payload: Any) -> float | None:
    """Compass heading to the nearest unobserved chunk edge, or None when there is none.

    `frontier_navigation` is the server's own join of the remembered chunks against the live bot,
    so it is already filtered to the bot's dimension and already a heading rather than a position:
    `explore_frontier` takes exactly this and walks it. No rows means the world is either fully
    remembered around the bot or the read failed, and both are the caller's problem to answer with
    a fallback, so this is None and not a guess.
    """
    rows = get(payload, "response", "data", "result", "query", "rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        cells = row if isinstance(row, list) else [row]
        for cell in cells:
            heading = _num(cell)
            if heading is not None:
                return round(heading % 360.0, 1)
    return None


def _iter_items(container: Any):
    """Yield (item_name, count) from inventory stacks / dicts."""
    if isinstance(container, list):
        for e in container:
            if isinstance(e, dict):
                name = e.get("item_name") or e.get("item") or e.get("name")
                cnt = e.get("count") or e.get("quantity") or 1
                if name:
                    yield str(name), int(cnt)
    elif isinstance(container, dict):
        for key, value in container.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                yield str(key), int(value)


def _armor_tier(item: str) -> str | None:
    """`diamond_chestplate` -> `diamond` (the material prefix the `tools` rows use)."""
    name = str(item).lower()
    for tier in _ARMOR_TIERS:
        if name.startswith(tier + "_"):
            return tier
    return None


# The server's air variants: a cell holding one of these has no block in it. `cave_air` is the
# variant for air enclosed by blocks (a cave, or a spot under an overhang) and behaves like open
# `air` wherever a cell only has to be empty -- a lava source in a cave is as pourable as one
# under open sky.
AIR_NAMES = frozenset({"air", "cave_air", "void_air"})


@dataclass(frozen=True)
class BlockHit:
    """One `view_blocks find` hit: a concrete block, its position, and its neighbours.

    `above`/`below` are the annotation the server adds to every hit; `above == "air"`
    means the block has open space over it -- standable footing for `navigate`, or an
    exposed ore that is easier to reach.

    `level` is the server's fluid level, reported for lava (and absent elsewhere): `0` is a
    *source* block of standing lava, anything above is a flow that spilled from one. The
    distinction decides the obsidian step -- water meeting a source forms obsidian, water
    meeting a flow forms only cobblestone -- so the pour target has to look at the level.
    """

    query: str  # the group the hit came from ("logs", "stone", ...)
    name: str   # the concrete block ("acacia_log", "stone", ...)
    x: int
    y: int
    z: int
    distance: float
    above: str | None = None
    below: str | None = None
    level: int | None = None

    @property
    def pos(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @property
    def standable(self) -> bool:
        return self.above == "air"

    @property
    def rescue_standable(self) -> bool:
        """A foothold a swimmer can pull onto, roofed or not.

        `standable` demands open sky (`above == "air"`), which `explore` and `base_target` rely
        on. A bot that poured water on a cave lava pool stands under `cave_air`, so a rescue that
        accepted only `air` saw no ledge beside it at all (live, run20). `targets.nearest_ground`
        still takes such a roofed cell only at the bot's own waterline, never far above or below.
        """
        return self.above in RESCUE_ABOVE

    def as_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "z": self.z,
                "distance": self.distance, "above": self.above, "below": self.below,
                "level": self.level}


# One sentence naming what the run is working on, in front of every model-facing prompt:
# it heads the System 1 state (the head is trained on exactly this rendering) and the
# System 2 escalation. A generic "Minecraft bot" spends no attention - the framing has
# to say which phase the bot is really in, because that changes what the next step is.

PHASE_TECH = "You are a Minecraft player working toward a full diamond kit"
PHASE_PREP = "You are a Minecraft player preparing for the nether"
PHASE_NETHER = "You are a Minecraft player killing blazes for blaze rods"
PHASE_STRONGHOLD = "You are a Minecraft player finding the stronghold"
PHASE_END = "You are a Minecraft player fighting the ender dragon"


def phase_persona(source) -> str:
    """The phase sentence for both model interfaces, from one source of truth.

    `source` is a `GameState` or its `as_state()` dict - System 1 renders it from the
    object, System 2 escalates with the logged state dict, and neither may frame the run
    differently than the other. The phases follow the world: the dimension owns nether
    and end work; in the Overworld a finished diamond kit (worn, diamond pickaxe and
    sword to hand) says the climb is over and only the crossing is left.
    In the Overworld the stronghold follows the Nether haul: once the blaze powder is
    banked (`powder_stowed`) and until the end portal stands active (`end_portal_active`),
    the run is out for eyes of ender and the stronghold (docs/stronghold.md). Both are loop
    latches that only ever set, so the sentence cannot flicker with the world.
    """
    def field(name: str, default=None):
        return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)
    dim = str(field("dimension") or "").lower()
    if "end" in dim:
        return PHASE_END
    if "nether" in dim:
        return PHASE_NETHER
    if field("powder_stowed") and not field("end_portal_active"):
        return PHASE_STRONGHOLD
    tools = field("tools") or {}
    armor = field("armor") or {}

    def worn(slot: str) -> bool:
        meta = armor.get(slot) if isinstance(armor, dict) else None
        return isinstance(meta, dict) and meta.get("tier") == "diamond"

    kit = (all(worn(slot) for slot in ("helmet", "chestplate", "leggings", "boots"))
           and tools.get("pickaxe") == "diamond" and tools.get("sword") == "diamond")
    return PHASE_PREP if kit else PHASE_TECH


@dataclass
class GameState:
    health: float | None = None
    food: float | None = None
    saturation: float | None = None
    phase: str | None = None
    raining: bool | None = None
    pos: tuple[float, float, float] | None = None
    on_ground: bool | None = None
    in_water: bool | None = None
    in_lava: bool | None = None
    dimension: str | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    # Occupied carried inventory slots (main + hotbar stacks). The server refuses a gathering
    # action with `[INVENTORY_FULL]` once all 36 are taken, so the loop needs the count to see
    # the pack filling up before that happens (worn armour and the off-hand do not count).
    slots_used: int = 0
    tools: dict[str, str] = field(default_factory=dict)
    hostiles: list[tuple[str, int, float | None]] = field(default_factory=list)
    nearest_hostile: float | None = None
    players: list[str] = field(default_factory=list)
    last_death_cause: str | None = None
    last_death_age_s: float | None = None
    # The server's timestamp of that death: the same string for every look at one death, a new
    # one for the next, so it tells two deaths in quick succession apart where the age cannot.
    last_death_at: str | None = None
    death_pos: tuple[float, float, float] | None = None
    death_dim: str | None = None
    activity_owner: str | None = None
    active_action: str | None = None
    # Compass heading to the nearest unobserved chunk edge, read from the server's frontier in the
    # Nether only (see `FRONTIER_HEADING_SQL`). None means "no frontier read this tick": the
    # fortress search then falls back to its own axis march.
    frontier_heading: float | None = None
    events: list[str] = field(default_factory=list)
    unread: int = 0
    survivors: list[str] = field(default_factory=list)
    # name -> [BlockHit] nearest first, already radius-filtered
    resources: dict[str, list[BlockHit]] = field(default_factory=dict)
    # The ground queries alone, read to `RESCUE_RADIUS` instead of `RESOURCE_RADIUS`, off the
    # same scan: what a bot treading water has to aim at is not a nearby resource but dry land.
    # Kept apart so no other consumer starts treating a 100 m block as reach.
    rescue_ground: dict[str, list[BlockHit]] = field(default_factory=dict)
    sleeping: bool | None = None
    minutes_to_change: float | None = None
    # (name, kind, count, nearest distance) for every nearby mob species
    mobs: list[tuple[str, str, int, float | None]] = field(default_factory=list)
    # Adult piglins loaded nearby as (entityId, nearest distance), nearest first - the partners
    # `barter_pearls` trades gold ingots with for ender pearls. Only adults are listed: the server
    # refuses a baby piglin id, and the group rows are keyed by name:age so the nearest adult is
    # the one the barter call is given.
    piglins: list[tuple[int, float | None]] = field(default_factory=list)
    # Barter calls this run has already made (`graph._note_barter`). The gold reserve is only dug
    # while the cap (`BARTER_MAX_ATTEMPTS`) is not reached, so a run that keeps missing pearls does
    # not turn the lull into a permanent gold farm.
    barter_attempts: int = 0
    # latest unread-notification previews (from the server's `notifications` block)
    notifications: list[str] = field(default_factory=list)
    # active tool per class -> {tier, item, left, max}; `left`/`max` are durability
    tool_meta: dict[str, dict] = field(default_factory=dict)
    # worn armour per slot class ("helmet"/"chestplate"/"leggings"/"boots") -> {tier, item, left, max}
    armor: dict[str, dict] = field(default_factory=dict)
    # item currently in the off-hand slot (a shield, usually); None when the slot is empty
    offhand: str | None = None
    # fresh "target died" reflex outcomes: the mob's drops are lying nearby
    drops: int = 0
    # Home base: the surface anchor the run puts the bed, the portal frame and the chest at.
    # Injected by the loop (perception cannot know it) and carried into every decision so the
    # base-relative steps have one fixed point that survives the bot wandering off to mine.
    base: tuple[int, int, int] | None = None
    # The placed chest at the base, remembered from `place_chest` so `use_container` can name
    # its exact coordinates without a second block scan.
    chest_pos: tuple[int, int, int] | None = None
    # The placed furnace at the base, remembered from `place_furnace` (or re-adopted from a scan
    # after a restart) so `cook_food`/`smelt_gold` use the furnace that already stands there.
    # `smelt_item` is given its exact coordinates and never places a temporary workstation, so
    # the furnace is not set up and torn down on every cooking call.
    furnace_pos: tuple[int, int, int] | None = None
    # The food station at the base. A smoker processes food only - it cannot smelt the iron
    # the tech tree needs - so it stands *beside* the furnace rather than replacing it.
    smoker_pos: tuple[int, int, int] | None = None
    # Ground covered since the crossing, in metres. The search budget is spent in this unit rather
    # than in decisions, because a decision is not search: run70 spent nineteen of twenty-four
    # ticks pinned by the combat reflex, standing still, and the hunt ended with 108 m covered.
    nether_travel: float = 0.0
    # True while the base was re-adopted from the world (a lit portal a previous run left) and the
    # bot has not reached it yet. It is what justifies walking home once (`go_home`) so a restart
    # finishes at the portal it built instead of drifting off to explore.
    base_adopted: bool = False
    # Lava cells whose pour the server already refused (`[BUCKET_NO_FACE]`): no solid block
    # beside them faces the bot, so retrying there can only fail again. The loop records them
    # and the obsidian step passes them over, which is what moves the search from a stopped-up
    # surface pond to the next lava pool instead of looping on the nearest cell forever.
    pour_skip: frozenset = frozenset()
    # Stances a `mine_obsidian` walk was refused at (`[NAVIGATION_TARGET_UNSUPPORTED]`): the
    # cell beside that lava has no footing, a property of the cell and not of the attempt. The
    # obsidian step stands on another neighbour, and once every stance of a source is refused it
    # passes over the source itself -- the same move `pour_skip` makes for a refused pour.
    stance_skip: frozenset = frozenset()
    # A `drop_junk` the terrain refused to give a disposal pit for (`[DROP_HOLE_UNAVAILABLE]`):
    # from then on the drop goes onto the open ground instead of into a dug hole. Both free the
    # slots the relief is for, and the open drop cannot fail, so the loop cannot stall on it.
    drop_hole_blocked: bool = False
    drop_open_pending: bool = False
    drop_pos: tuple[float,float,float] | None = None
    # The way the bot last walked before that drop (x, z), so the walk off it goes the other way:
    # the toss flies the way the bot faces (`graph.Agent._note_heading`).
    drop_heading: tuple[float, float] | None = None
    # The standable cell the walk off that drop goes to, picked from a ground read around it
    # (`targets.step_away_cell`); None when the read found none.
    drop_away: tuple[int, int, int] | None = None
    # A `place_chest` the terrain refused for want of a supporting block (`[PLACE_NO_SUPPORT]`):
    # no solid block touches the cell beside the base, so the chest cannot stand there. That is
    # a property of the base's ground, so the loop gives the chest chain up and lets the
    # inventory relief drop surplus instead of retrying a placement that can never land (live,
    # Nether run: the answer came on every tick and the run never reached the portal).
    chest_blocked: bool = False
    # A `place_furnace` the terrain refused for want of a supporting block (`[PLACE_NO_SUPPORT]`):
    # like the chest, that is a property of the base's ground, so the loop gives the permanent
    # furnace up and falls back to the temporary workstation `smelt_item` places itself.
    furnace_blocked: bool = False
    # What the standing furnace at the base was last seen to hold, read from a `use_container
    # inspect` (`Agent._note_furnace`). An empty dict means "not looked at yet", which is why
    # `collect_smelted` inspects before it reaches in: a cook is only interrupted by a furnace
    # that really is occupied.
    furnace_contents: dict[str, int] = field(default_factory=dict)
    # A `smelt_item` was refused with `[FURNACE_NOT_EMPTY]`: the standing furnace is occupied and
    # has to be drained before the next cook can start (Agent `_note_furnace`).
    furnace_occupied: bool = False
    # The cell a placement was refused at because a block stood in it (`[PLACE_TARGET_OCCUPIED]`):
    # `make_room` digs exactly that cell, since the block in it is the whole cause and a pocket
    # around the bot need not touch it (the chest cell sits beside the base, not the bot). Live,
    # from-zero run: the cell held `grass_block` and `place_chest` answered the code every tick.
    blocked_cell: tuple[int, int, int] | None = None
    # The opaque full cube the server found sitting directly above the base chest, refusing to
    # open it (`[CHEST_BLOCKED]`, `perception.parse_block_above`). `stow_surplus` digs that exact
    # cell before opening again: the lid cannot swing under a full cube, so the deposit (and with
    # it the whole inventory relief) fails on every tick otherwise (live, run26: the buried base
    # put a cobblestone on the chest lid and the full pack could never be unloaded).
    chest_above: tuple[int, int, int] | None = None
    # Whether a block kind exists anywhere in the loaded world, even beyond RESOURCE_RADIUS.
    # `view_blocks find` reports a per-name `found` count across every loaded chunk, so a deep
    # deposit the 64 m scan cannot list still proves the ore is there to dig for. This is what
    # gates `mine_diamond`: standing on the surface, the nearest diamond ore (y<16) is always
    # out of scan range, so without the raw count the loop could never dig down to the diamond
    # armour piece it still needs once one is lost.
    resource_found: dict[str, int] = field(default_factory=dict)

    # How many of each edible item the Overworld chest last held, from the server's container
    # database. Carried and banked are deliberately two fields and not one: the chest is at the
    # base and is worth nothing in the Nether, so what stops a hunt is the total while what has to
    # be carried across is the pack. Folding them into one number deadlocks the run - stop hunting,
    # and then nothing can put food in the pack.
    banked: dict[str, int] = field(default_factory=dict)
    # The overflow chests standing beside the base as the storage network grows
    # (`Agent._extra_chests`): the cells a deposit can look for room in once the main chest is
    # full. Injected per run like `chest_pos` itself; a restart forgets them until the server's
    # container database or the next placement names them again.
    extra_chests: tuple[tuple[int, int, int], ...] = ()
    # The network cells the server has already called full: a deposit into one answered
    # `[CONTAINER_BATCH_INCOMPLETE]` (`Agent._note_container`), and a deposit that succeeds opens
    # it again. Not whether *a* chest is full but which one: `stow_surplus` has to aim at the one
    # with room and the network may only grow once every one of them is full (live, run79: the
    # base chest held everything, every deposit after it was refused, and 102 `wait` ticks with
    # a 36/36 pack followed).
    chests_full: frozenset = frozenset()
    # Overflow candidate cells whose ground refused the chest (`[PLACE_NO_SUPPORT]`,
    # `Agent._note_chest`): the next candidate is chosen instead, and the network stops growing
    # when the candidates run out.
    overflow_skip: frozenset = frozenset()
    # What each network chest last held, as `(cell, item, count)` triples off the server's
    # container database (`Agent._note_stocked_chest`). The aggregate `banked` stays the number
    # that stops hunts; the per-cell shape is what `take_stored_food` needs - a `withdraw` that
    # names provisions in a chest that does not hold them takes nothing out and answers exactly
    # like a working chest does.
    banked_stores: tuple = ()
    # A level, clear surface cell to raise the home base on, proposed by a one-off
    # `view_blocks box` scan while `base` is still unset. The loop owns the base, so this is
    # only the *site proposal*; `None` means no flat site was found and the base pick falls
    # back to the nearest standable ground. A frame raised on a slope leaves its cells holding
    # stone, which the build server will not dig (live: 15 of 20 frame cells stayed stone).
    build_site: tuple[int, int, int] | None = None
    # The surface cell `(x, feet_y, z)` the loop last learned for the bot's column, or `None`
    # while it is unknown. A height-less `navigate` answers it (`[NAVIGATION_HEIGHT_REQUIRED]`
    # names the highest block, so the feet cell is one above); the loop owns it per run and
    # injects it, exactly like `base`. It is what tells a bot standing on the ground from one
    # standing on a cave floor: a mined tunnel's `air` reads the same as open sky, so the depth
    # below this cell -- not the block above the head -- is the surface test `establish_base`
    # needs (live, run9: the base landed at y=-16 under a y=73 surface).
    surface: tuple[int, int, int] | None = None
    # Cells of the portal opening (`portal_interior`) that hold a stray block, read from one
    # `view_blocks cells` scan while a frame build is pending. The opening sits *above* the
    # support layer and must end up air: a block left there is built around and `activate_portal`
    # answers `[PORTAL_NO_FRAME]` on every tick (live, Nether run: a bed the generic sleep laid
    # "beside the bot" landed in the opening). `clear_portal` digs these cells before the build.
    portal_blocked: list[tuple[int, int, int]] = field(default_factory=list)
    # The loop learned from the server that the frame build has nothing left to place: a
    # `build_portal` answered `ok` while spending no obsidian or cobblestone (`Agent._note_build`).
    # That success resets `_track_stuck`, so the "resume a partial frame" clause would offer the
    # step forever; while this stands `candidates` drops it so the lull work owns the tick (live,
    # run25: 163 identical `build_portal` ticks on an already finished frame, the bot frozen).
    frame_complete: bool = False
    # Frame cells a `build_structure` refused to touch (`[BUILD_INCOMPLETE] ... (x, y, z: ...)`,
    # read by `parse_refused_cells`). The server digs a wrong frame cell in place, but the movement
    # policy refuses a break it was not asked for, and a *container* (a furnace, a chest) always
    # needs that explicit request -- so the cell never clears and the frame never closes, leaving
    # `light_portal` to answer `[PORTAL_NO_FRAME]` forever. `clear_frame` digs each with the
    # explicit `collect_block` the policy does allow (live, run27: run26's furnace stood on the new
    # base's frame corner and the build refused it on every attempt).
    refused_frame: tuple[tuple[int, int, int], ...] = ()
    # The block name of each cell of the frame plane (`frame_volume`), read by the frame scan while
    # a build is pending. `clear_frame` needs the name to hand `collect_block`, which matches the
    # block it is asked for; a cell the scan has not named yet waits for the next read.
    frame_names: dict[tuple[int, int, int], str] = field(default_factory=dict)
    # The loop is *intending* a portal crossing this tick (`enter_portal` / `leave_nether`). The
    # `_in_portal` rescue exists to step a body out of a gate that merely walked into one, so it
    # must stand down for the one crossing that is asked for - otherwise the rescue undoes the
    # crossing every tick and the trip never happens (Agent `_enter_intent`).
    enter_intent: bool = False
    # The Nether expedition has already happened at least once, so the loop does not offer the
    # Overworld gate again while it is busy down there or on the way back.
    nether_entered: bool = False
    # Decisions spent in the Nether (Agent `_note_dimension`), reset the moment the bot is home.
    # `_nether_hopeless` needs it: the bot lands in open Nether, where a fortress is hundreds of
    # blocks off and nothing but lava is loaded, and that must not read as "nothing left to find"
    # on the first tick.
    nether_ticks: int = 0
    # The cell the bot arrived in on the Nether side (Agent `_note_arrival`). The exit portal is
    # that spot or right beside it, and a portal block only reaches the state while it is loaded -
    # so when the scan comes up empty this is the heading `leave_nether` walks instead.
    nether_arrival: tuple[int, int, int] | None = None
    # The Nether side of the portal the last real crossing went through, the server's own memory
    # (`perception.REMEMBERED_GATE_SQL`) rather than a world read: it outlives the death and the
    # restart, so it is the way home `leave_nether` takes when the run *began* in the Nether and
    # saw no crossing to record an arrival from. Run74 had neither and walked its bot to a
    # non-portal: the cell it started on, not the gate at (33,55,34) it never knew about.
    nether_gate: tuple[int, int, int] | None = None
    # Fortress landmark cells the bot has already walked onto and got nothing out of (Agent
    # `_note_landmark`): a cage whose `navigate` answered with 0 metres moved is a searched cage,
    # and offering it again next tick is the same 0-metre step (live, run104: 100+ ticks on one
    # spawner at (94,39,86)).
    nether_landmarks: frozenset = frozenset()
    # The blaze powder reached the chest at the Overworld portal (Agent `_note_stow`). Reading the
    # chest back would cost a `use_container` inspection every tick, and the deposit is already
    # verified by the action's own result - the same latch shape as `frame_complete`.
    powder_stowed: bool = False
    # The stronghold phase's loop latches (docs/stronghold.md), each set on its action's own
    # result, the same shape as `powder_stowed`. `end_portal_active`: `activate_portal` confirmed
    # the open portal (Agent `_note_portal`), the phase goal. `stronghold_estimated`: the
    # `locate_stronghold` estimate saved its bearing, so the next call travels (`phase: locate`).
    # `powder_exhausted`: a withdraw from the gate chest came up without powder, so the eyes can
    # only be paid for by a second Nether trip; a fresh stow clears it again.
    end_portal_active: bool = False
    stronghold_estimated: bool = False
    powder_exhausted: bool = False
    # The cells a blaze-powder stow actually deposited into (Agent `_note_stow`): the storage
    # network's roomy chest, which is not always `chest_pos`. `take_blaze_powder` withdraws from
    # these remembered cells, and `powder_exhausted` only latches once every one of them has
    # come up empty - a top-up aimed at the main chest while the powder sits in an overflow
    # chest would send the bot back through the gate for nothing.
    powder_cells: tuple[tuple[int, int, int], ...] = ()
    # Death cells (`actions._cell`) a recovery walk already reached and found empty
    # (`[NO_MATCHING_ITEMS]` and the other final codes). `_can_recover` refuses to offer the walk
    # there again, so a bot whose gear is gone does not stand at the death site for the whole
    # window while the tech tree waits (live, run14: two such loops, 117 ticks in all).
    recover_written_off: frozenset = frozenset()
    # The pathfinder cannot leave the few cells around the bot (`Agent._note_boxed`): `dig_out`
    # owns the tick, and `dig_out_try` turns each further tunnel to the next side (live, run90).
    boxed_in: bool = False
    dig_out_try: int = 0
    # Decisions the bot has spent off its feet in water, counted by the loop (`Agent._inject`).
    # `escape_water` owns the tick only once this reaches `actions.WATER_RESCUE_AFTER`, so a goal
    # that crosses water the server can swim is not torn up the moment an action comes back
    # (live, run15).
    water_ticks: int = 0
    # The defers in a row that met no damage and no survival reflex (`Agent._note_defer`): once
    # they reach `brain.DEFER_QUIET_TICKS` a mob in contact range no longer yields the tick.
    defer_quiet: int = 0
    # The labels the loop is currently withholding from `candidates` because each kept answering
    # the identical refusal without progress (`Agent._track_stuck`, see `graph.STUCK_REPEATS`).
    # While one is listed, `candidates` drops it so the last fallback (`explore`/`wait`) owns the
    # tick and the bot moves somewhere new instead of burning the run on the same call.
    stuck_labels: frozenset = frozenset()

    def resource_nearest(self, name: str) -> BlockHit | None:
        hits = self.resources.get(name)
        return hits[0] if hits else None

    def resource_distance(self, name: str) -> float | None:
        hit = self.resource_nearest(name)
        return hit.distance if hit else None

    def mob_nearest(self, name: str) -> float | None:
        ds = [d for n, _, _, d in self.mobs if n == name and d is not None]
        return min(ds) if ds else None

    def piglin_nearest(self) -> tuple[int, float] | None:
        """The nearest adult piglin as (entityId, distance), or None if none is loaded.

        The barter partner: `barter` needs an exact loaded adult piglin id, and the loop only
        trades with one close enough to reach without a long walk (see `BARTER_PIGLIN_RANGE`).
        """
        near = [(eid, d) for eid, d in self.piglins if d is not None]
        return min(near, key=lambda p: p[1]) if near else None

    def as_state(self) -> dict:
        """Structured view for logging and System-2 prompts (never sent raw to Laya).

        The furnace's own facts are in here because they are loop-owned facts of the same kind as
        `base` and `chest_pos`, and their absence made a live question unanswerable: run53 carried a
        furnace for 341 of 377 records and `place_furnace` never ran, and the decision log could not
        say why, because it held no `furnace_pos`, no `furnace_blocked` and no `occupied` flag. A
        field that is not logged cannot be diagnosed after the fact, and the state cannot be
        rebuilt from the log to ask a predicate about it either - `from_state` is that inverse, so
        the question is asked on the recorded tick instead of on a reconstruction of it.

        Note the one name that is not the attribute's: the position is logged as `"position"` while
        the field is `pos`, which is exactly the sort of gap that made three attempts to rebuild a
        tick fail. `from_state` reads the log's vocabulary, so callers never hand-map it.
        """
        return {
            "health": self.health,
            "food": self.food,
            "phase": self.phase,
            "raining": self.raining,
            "position": self.pos,
            "on_ground": self.on_ground,
            "in_water": self.in_water,
            "dimension": self.dimension,
            "inventory": self.inventory,
            "slots_used": self.slots_used,
            "tools": self.tools,
            "tool_meta": self.tool_meta,
            "armor": self.armor,
            "offhand": self.offhand,
            "hostiles": self.hostiles,
            "players": self.players,
            "last_death": self.last_death_cause,
            "death_pos": self.death_pos,
            "drops": self.drops,
            "base": self.base,
            "chest_pos": self.chest_pos,
            "base_adopted": self.base_adopted,
            "furnace_pos": self.furnace_pos,
            "smoker_pos": self.smoker_pos,
            "furnace_occupied": self.furnace_occupied,
            "furnace_contents": dict(self.furnace_contents),
            "furnace_blocked": self.furnace_blocked,
            "events": self.events,
            "unread": self.unread,
            "notifications": self.notifications,
            "sleeping": self.sleeping,
            "resources": {n: [h.as_dict() for h in hits[:2]] for n, hits in self.resources.items() if hits},
            "resource_found": self.resource_found,
            "banked": self.banked,
            # The storage network belongs in the log for the same reason `chest_pos` does: run79's
            # deadlock was only diagnosable afterwards, and afterwards the cells that were full,
            # the chests beside them and where each provision stood were nowhere to be read.
            "extra_chests": [list(cell) for cell in self.extra_chests],
            "chests_full": [list(cell) for cell in sorted(self.chests_full)],
            "banked_stores": [[list(cell), name, count]
                              for cell, name, count in self.banked_stores],
            # The Nether crossing and the cell it arrived on. Loop-owned, and a field that is
            # not logged cannot be diagnosed after the fact - which is the whole of run60's
            # "leave_nether is offered and refused" reading: `nether_arrival` is what
            # `resolve` asks for when no portal block is in the scan, so a rebuilt tick could
            # never answer it and looked like a broken gate rather than a missing field.
            "nether_entered": self.nether_entered,
            "nether_arrival": self.nether_arrival,
            "nether_gate": self.nether_gate,
            "nether_ticks": self.nether_ticks,
            "nether_travel": self.nether_travel,
            "mobs": self.mobs,
            "piglins": self.piglins,
            "barter_attempts": self.barter_attempts,
            # The stronghold latches, logged for the same reason the Nether facts are.
            "powder_stowed": self.powder_stowed,
            "powder_exhausted": self.powder_exhausted,
            "stronghold_estimated": self.stronghold_estimated,
            "end_portal_active": self.end_portal_active,
            "powder_cells": [list(cell) for cell in self.powder_cells],
            "nether_landmarks": [list(cell) for cell in self.nether_landmarks],
        }

    @classmethod
    def from_state(cls, data: dict) -> "GameState":
        """Rebuild a state from a logged one, so a recorded tick can be questioned directly.

        The inverse of `as_state`, tolerant of what it does not find: an old log entry, or a
        hand-written fixture, still yields a usable state with the dataclass defaults. That is the
        point. "Why did the bot never place the furnace" was unanswerable not because the code was
        hard to read but because the log held none of the furnace's facts and the state had to be
        reconstructed by hand - three attempts, three different missing fields. Asking the
        predicate on the tick that was actually recorded is not the same thing as guessing at it.

        Resources come back as `BlockHit`s, so `actions` predicates (`_furnace_site`,
        `_portal_frame_started`, `_bed_bad_spot`) read the same way on a logged tick as on a live
        one. The one name translation - the logged `"position"` for the field `pos` - happens here.
        """
        gs = cls()
        for field in ("health", "food", "phase", "raining", "on_ground", "in_water",
                      "dimension", "slots_used", "tools", "tool_meta", "armor", "offhand",
                      "hostiles", "players", "unread", "death_pos", "drops", "base", "chest_pos",
                      "base_adopted", "furnace_pos", "smoker_pos", "furnace_occupied", "furnace_blocked",
                      "sleeping", "resource_found", "mobs", "piglins", "barter_attempts", "banked",
                      "nether_entered", "nether_arrival", "nether_gate", "nether_ticks",
                      "nether_travel", "powder_stowed", "powder_exhausted",
                      "stronghold_estimated", "end_portal_active", "powder_cells",
                      "nether_landmarks"):
            value = data.get(field)
            if value is not None:
                setattr(gs, field, value)
        gs.powder_cells = tuple(tuple(cell) for cell in gs.powder_cells)
        gs.nether_landmarks = frozenset(tuple(cell) for cell in gs.nether_landmarks)
        gs.pos = data.get("position")
        gs.inventory = dict(data.get("inventory") or {})
        gs.furnace_contents = dict(data.get("furnace_contents") or {})
        gs.events = list(data.get("events") or [])
        gs.notifications = list(data.get("notifications") or [])
        gs.resources = {}
        for name, hits in (data.get("resources") or {}).items():
            rebuilt = []
            for hit in hits or []:
                if not isinstance(hit, dict) or "x" not in hit:
                    continue
                rebuilt.append(BlockHit(
                    query=hit.get("query", name), name=hit.get("name", name),
                    x=int(hit["x"]), y=int(hit["y"]), z=int(hit["z"]),
                    distance=float(hit.get("distance") or 0.0),
                    above=hit.get("above"), below=hit.get("below"), level=hit.get("level")))
            gs.resources[name] = rebuilt
        # The network's cells come back from JSON as lists; the state keeps tuples so they are
        # hashable where a full chest is tested against a set, and a torn entry from an old log
        # simply drops instead of guessing.
        gs.extra_chests = tuple(tuple(int(v) for v in cell)
                                for cell in (data.get("extra_chests") or ())
                                if isinstance(cell, (list, tuple)) and len(cell) == 3)
        gs.chests_full = frozenset(tuple(int(v) for v in cell)
                                   for cell in (data.get("chests_full") or ())
                                   if isinstance(cell, (list, tuple)) and len(cell) == 3)
        stores = []
        for entry in (data.get("banked_stores") or ()):
            if isinstance(entry, (list, tuple)) and len(entry) == 3:
                cell, name, count = entry
                if isinstance(cell, (list, tuple)) and len(cell) == 3:
                    stores.append((tuple(int(v) for v in cell), str(name), int(count)))
        gs.banked_stores = tuple(stores)
        return gs

    def narrative(self) -> str:
        parts: list[str] = []
        if self.dimension:
            parts.append(f"in the {self.dimension}")
        if self.health is not None:
            parts.append("health is full" if self.health >= 20
                         else f"health {self.health:g}/20, {_health_band(self.health)}")
        if self.food is not None:
            parts.append("hunger is full" if self.food >= 20
                         else f"hunger {self.food:g}/20, {_food_band(self.food)}")
        if self.phase:
            weather = " and raining" if self.raining else ""
            label = "daytime" if self.phase == "day" else self.phase
            asleep = " and the bot is asleep" if self.sleeping else ""
            parts.append(f"it is {label}{weather}{asleep}")
        if self.pos:
            _, y, _ = self.pos
            # `onGround` is the server's physics contact flag - true on a cave floor a hundred
            # blocks under the sky, which is why the run that mined at y=-38 read "(on the
            # ground)" to a human as "surface". Say what the flag means, nothing more.
            where = "standing on solid floor" if self.on_ground else "in the air"
            if self.in_water:
                where = "in water"
            # Raw x/z are per-world random and not transferable, so the head would have to learn
            # to ignore them - the same reason "players nearby" was cut. Only the depth band
            # survives, read against the global frame every world shares (bedrock ~-64, sea level
            # ~63, diamonds below 16); `as_state` still logs the absolute position.
            parts.append(f"{_depth_band(y)} ({where})")
        if self.base:
            bx, by, bz = self.base
            at_base = (self.pos and abs(self.pos[0] - (bx + 0.5)) <= 4
                       and abs(self.pos[2] - (bz + 0.5)) <= 4)
            if at_base:
                parts.append("at the home base")
            elif self.pos:
                # Relative to the bot, not absolute: the distance to walk and roughly where it
                # lies vertically is the whole of what a step keys on. Direction stays coarse -
                # the head needs the relation, not the metres.
                dist = math.dist(self.pos, (bx + 0.5, by + 0.5, bz + 0.5))
                vert = _rough_vertical((by + 0.5) - self.pos[1])
                tail = "" if vert == "level" else f" and {vert}"
                parts.append(f"home base {dist:.0f} blocks away{tail}")
            else:
                parts.append("home base set")
        if self.chest_pos:
            # The fullness of the network is a decision the model has to be able to see: run79
            # waited with a full pack beside a full chest because nothing in the state said the
            # chest could not take another item, and "build another chest" was never on offer.
            line = "a chest stands at the base"
            extra = [tuple(cell) for cell in self.extra_chests]
            if extra:
                line += f", plus {len(extra)} overflow chest{'' if len(extra) == 1 else 's'} beside it"
            full = set(self.chests_full)
            network = [tuple(self.chest_pos)] + extra
            if extra and all(cell in full for cell in extra) and tuple(self.chest_pos) in full:
                line += ": every one of them is full, so nothing more can be stowed until another chest stands"
            elif tuple(self.chest_pos) in full:
                line += " (it is full)" if not extra else " (the main one is full)"
            parts.append(line)
        inv = ", ".join(f"{c}x {n}" for n, c in sorted(self.inventory.items())) or "empty"
        parts.append(f"inventory: {inv}")
        if self.slots_used >= INVENTORY_NEAR_FULL:
            parts.append(f"the pack is full ({self.slots_used}/{INVENTORY_SLOTS} slots)")
        tools = ", ".join(f"{cls} {tier}".strip() for cls, tier in self.tools.items()
                          if tier and tier != "none") or "no tools"
        parts.append(f"tools: {tools}")
        worn_armor = ", ".join(f"{cls} {meta.get('tier')}"
                               for cls, meta in self.armor.items()
                               if meta.get("tier") and meta.get("tier") != "none")
        if worn_armor:
            parts.append(f"armour: {worn_armor}")
        if self.offhand:
            parts.append(f"off-hand: {self.offhand}")
        worn = []
        for cls, meta in self.tool_meta.items():
            left, mx = meta.get("left"), meta.get("max")
            if isinstance(left, (int, float)) and isinstance(mx, (int, float)) and mx > 0:
                pct = left / mx
                if pct <= 0.25:
                    worn.append(f"{meta.get('item') or cls} at {pct * 100:.0f}% durability")
        if worn:
            parts.append("low durability: " + ", ".join(worn))
        close = [(n, c, d) for n, c, d in self.hostiles if d is None or d <= 16]
        if close:
            desc = ", ".join(
                f"{n} x{c}" + (f" at {d:.0f} blocks" if d is not None else "")
                for n, c, d in close[:3]
            )
            parts.append(f"hostile mobs within 16 blocks: {desc}")
        else:
            parts.append("no hostile mobs within 16 blocks")
        # `players nearby: ...` used to be a sentence here and is gone on purpose. No candidate and no
        # rule has ever keyed on another player - the goal is a solo playthrough - so it was pure
        # noise that spent the model's attention on a fact it has to learn to ignore. `players`
        # stays on the state, so the log still records who was standing there.
        if self.drops:
            parts.append(f"{self.drops} mob(s) the reflex killed dropped items nearby")
        # Death is narrative only while the gear is still recoverable - inside the very window
        # `recover_drops` offers on (see `_can_recover`). Outside it the sentence is noise Laya
        # cannot act on: live, run78 read a 14-hours-old death in the *previous* world on every
        # single tick (the server remembers deaths across world changes and restarts), with
        # nothing left to recover and no site in this world.
        if self.death_pos is not None and self.last_death_age_s is not None and self.pos is not None:
            from .actions import RECOVER_RANGE, RECOVER_WINDOW_S  # lazy: `actions` imports this module
            if (self.last_death_age_s <= RECOVER_WINDOW_S
                    and math.dist(self.pos, self.death_pos) <= RECOVER_RANGE):
                dist = math.dist(self.pos, self.death_pos)
                cause = self.last_death_cause or "cause unreported"
                parts.append(f"last death: {self.last_death_age_s:.0f}s ago {cause} "
                             f"{dist:.0f} blocks away; gear dropped there can still be recovered")
        present = [(n, hits[0].distance) for n, hits in self.resources.items()
                   if hits and n in NAMED_RESOURCES]
        if present:
            present.sort(key=lambda t: t[1])
            parts.append("nearby resources: " + ", ".join(
                f"{n} at {d:.0f} blocks" for n, d in present[:4]))
        if self.unread:
            head = "; ".join(self.notifications[:2])
            parts.append(f"unread notifications ({self.unread}): {head}" if head
                         else f"unread notifications: {self.unread}")
        # Every clause is capitalised, not just the first. Capitalising only `parts[0]` left "Health 12/20,
        # wounded. hunger 9/20, below the eating floor." - the same category/measurement mix one level
        # down, with a sentence break sitting where a reader expects a new subject.
        for i, part in enumerate(parts):
            parts[i] = part[:1].upper() + part[1:] if part else part
        return phase_persona(self) + ". " + ". ".join(parts) + "."


def _norm(text: Any, limit: int = 80) -> str:
    return " ".join(str(text or "").split())[:limit]


def _summaries(events: list[dict], limit: int = 3) -> list[str]:
    return [_norm(e.get("summary") or e.get("type")) for e in events[-limit:] if isinstance(e, dict)]


def _previews(preview: Any) -> list[str]:
    return [_norm(p) for p in (preview or [])][:3]


def _count_drops(events: list[dict], now_iso: Any) -> int:
    """Mobs the survival reflex killed recently: their drops are lying nearby."""
    count = 0
    for e in events:
        if not isinstance(e, dict) or "target died" not in str(e.get("summary") or ""):
            continue
        age = _age_seconds(now_iso, e.get("observedAt"))
        if age is not None and age <= EVENT_REACT_WINDOW_S:
            count += 1
    return count


async def _drain_events(executor) -> tuple[list[dict], int, list[str]]:
    """Walk the unread event backlog to the present.

    `read_recent_events` returns the *oldest* unread page (max 100), so a deep backlog
    has to be walked page by page before this tick's fresh events appear. Bounded per
    tick so a runaway backlog cannot spin; the unread count also drains as we go.
    """
    events: list[dict] = []
    unread, preview = 0, []
    for _ in range(EVENT_DRAIN_PAGES):
        res = await executor.read(
            "read_recent_events", {"limit": EVENT_PAGE},
            "drain unread world events so reactions act on fresh ones",
        )
        data = coerce(res.data, res.text)
        result = get(data, "response", "data", "result", default={}) or {}
        page = [e for e in (result.get("events") or []) if isinstance(e, dict)]
        events.extend(page)
        got = get(data, "notifications", "unreadCount")
        if isinstance(got, int):
            unread = got
        prev = get(data, "notifications", "recentPreview")
        if prev:
            preview = prev
        remaining = result.get("remainingEventCount")
        if not page or not isinstance(remaining, int) or remaining <= 0:
            break
    return events, unread, preview


def parse_status(data: dict) -> GameState:
    gs = GameState()
    sit = get(data, "response", "data", "result", "situation", default={}) or {}
    if not isinstance(sit, dict):
        return gs

    vitals = sit.get("vitals") or {}
    gs.health = _num(vitals.get("health"))
    gs.food = _num(vitals.get("food"))
    gs.saturation = _num(vitals.get("saturation"))

    clock = sit.get("clock") or {}
    gs.phase = clock.get("phase")
    raining = clock.get("raining")
    gs.raining = bool(raining) if isinstance(raining, bool) else None
    sleeping = clock.get("sleeping")
    gs.sleeping = bool(sleeping) if isinstance(sleeping, bool) else None
    gs.minutes_to_change = _num(clock.get("minutesUntilChange"))

    pos = sit.get("position") or {}
    if all(k in pos for k in ("x", "y", "z")):
        gs.pos = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
    gs.on_ground = pos.get("onGround") if isinstance(pos.get("onGround"), bool) else None
    gs.in_water = pos.get("inWater") if isinstance(pos.get("inWater"), bool) else None
    gs.in_lava = pos.get("inLava") if isinstance(pos.get("inLava"), bool) else None

    gs.dimension = sit.get("dimension")

    inv = sit.get("inventory") or {}
    for name, cnt in _iter_items(inv.get("stacks")):
        gs.inventory[name] = gs.inventory.get(name, 0) + cnt

    tools = (sit.get("tools") or {}).get("tools") or []
    for t in tools:
        if isinstance(t, dict) and t.get("class"):
            cls = str(t["class"])
            gs.tools[cls] = str(t.get("tier") or "none")
            gs.tool_meta[cls] = {
                "tier": t.get("tier"),
                "item": t.get("item"),
                "left": _num(t.get("durabilityLeft")),
                "max": _num(t.get("maximumDurability")),
            }
    # Worn armour is the stack that sits in one of the four armour slots (location
    # head/torso/legs/feet); the off-hand item reports location off-hand. `situation.tools.armour`
    # lists only the carried set per class and says nothing about what is on the body, so an
    # equipped piece has to be read from the location. It leaves main/hotbar, so remove it from
    # the carried inventory the loop above added.
    gs.slots_used = 0
    for st in inv.get("stacks") or []:
        if not isinstance(st, dict):
            continue
        loc = str(st.get("location") or "")
        name = st.get("item_name") or st.get("item") or st.get("name")
        if not name:
            continue
        name = str(name)
        cls = ARMOR_SLOTS.get(loc)
        if cls:
            dur = st.get("durability") or {}
            gs.armor[cls] = {"tier": _armor_tier(name), "item": name,
                             "left": _num(dur.get("remaining")), "max": _num(dur.get("maximum"))}
        elif loc in OFFHAND_SLOTS:
            gs.offhand = name
        else:
            gs.slots_used += 1
            continue  # carried normally: the loop above already counted it
        # An equipped piece leaves the pack, so it is not also counted as carried.
        left = gs.inventory.get(name, 0) - int(st.get("count") or 1)
        if left > 0:
            gs.inventory[name] = left
        else:
            gs.inventory.pop(name, None)

    nearby = sit.get("nearby") or {}
    hostile_rows = nearby.get("hostiles")
    if not isinstance(hostile_rows, list) or not hostile_rows:
        hostile_rows = [m for m in (nearby.get("mobs") or [])
                        if isinstance(m, dict) and m.get("kind") == "hostile"]
    for m in hostile_rows:
        if not isinstance(m, dict):
            continue
        # `nearby.hostiles` rows carry `distance` directly and no `count`; `nearby.mobs` rows
        # nest it under `nearest`. Reading only the nested form dropped every hostile distance,
        # so the narrative said "spider x1" with no "at Xm" and the danger gate could not tell a
        # mob across the room from one in contact (live: it deferred every tick over a zombie
        # 13.5 m away in a cave while health was full, and the bot stood still).
        nearest = m.get("nearest")
        dist = m.get("distance")
        if dist is None and isinstance(nearest, dict):
            dist = nearest.get("distance")
        gs.hostiles.append((str(m.get("name") or "?"), int(m.get("count") or 1), _num(dist)))
    dists = [d for _, _, d in gs.hostiles if d is not None]
    gs.nearest_hostile = min(dists) if dists else None

    for p in nearby.get("players") or []:
        if isinstance(p, dict) and p.get("username"):
            gs.players.append(str(p["username"]))

    for m in nearby.get("mobs") or []:
        if not isinstance(m, dict) or not m.get("name"):
            continue
        nearest = m.get("nearest") or {}
        dist = _num(nearest.get("distance")) if isinstance(nearest, dict) else None
        gs.mobs.append((str(m["name"]), str(m.get("kind") or "?"),
                        int(m.get("count") or 1), dist))
        if m["name"] == "piglin" and str(m.get("age")) == "adult":
            eid = nearest.get("entityId") if isinstance(nearest, dict) else None
            if isinstance(eid, int):
                gs.piglins.append((eid, dist))

    death = sit.get("lastDeath") or {}
    if isinstance(death, dict) and death.get("cause"):
        gs.last_death_cause = str(death["cause"])
        gs.last_death_age_s = _age_seconds(sit.get("observedAt"), death.get("observedAt"))
        if isinstance(death.get("observedAt"), str):
            gs.last_death_at = death["observedAt"]
        dpos = death.get("position") or {}
        if all(k in dpos for k in ("x", "y", "z")):
            gs.death_pos = (float(dpos["x"]), float(dpos["y"]), float(dpos["z"]))
        gs.death_dim = death.get("dimension") if isinstance(death.get("dimension"), str) else None

    activity = sit.get("activity") or {}
    gs.activity_owner = activity.get("owner")
    gs.active_action = activity.get("activeAction")

    surv = get(data, "response", "data", "survival", default={}) or {}
    summary = surv.get("summary") if isinstance(surv, dict) else None
    if isinstance(summary, str) and summary not in ("unknown", ""):
        gs.survivors.append(summary)
    return gs


def parse_resources(data: dict, limit: int = RESOURCE_LIMIT,
                    radius: float = RESOURCE_RADIUS) -> dict[str, list[BlockHit]]:
    """view_blocks find -> {query_name: [BlockHit]} within `radius`, nearest first.

    `limit` caps the hits kept per query name. It has to match the `limit` the scan asked
    the server for, or the parse would throw away hits the state still needs -- the lava
    group especially, where a pour cell is only recognisable when the *source* beside it is
    in the list too (and that source is not among the nearest few cells).

    `radius` is the read bound; only the rescue's ground scan widens it to `RESCUE_RADIUS`.
    """
    blocks = get(data, "response", "data", "result", "blocks", default={}) or {}
    groups = blocks.get("find") if isinstance(blocks, dict) else None
    out: dict[str, list[BlockHit]] = {}
    for group in groups or []:
        if not isinstance(group, dict) or not group.get("name"):
            continue
        query = str(group["name"])
        hits: list[BlockHit] = []
        for e in group.get("listed") or []:
            if not isinstance(e, dict):
                continue
            d = _num(e.get("distance"))
            if d is None or d > radius or "x" not in e:
                continue
            hits.append(BlockHit(
                query=query,
                name=str(e.get("name") or query),
                x=int(e["x"]), y=int(e["y"]), z=int(e["z"]), distance=d,
                above=e.get("above"), below=e.get("below"),
                level=int(e["level"]) if isinstance(e.get("level"), (int, float)) else None,
            ))
        hits.sort(key=lambda h: h.distance)
        if hits:
            out[query] = hits[:limit]
    return out


def parse_resource_counts(data: dict) -> dict[str, int]:
    """view_blocks find -> {query_name: found} across every loaded chunk.

    The listed hits `parse_resources` keeps are radius-limited, but each group also carries a
    `found` total for the whole loaded world. Kept separately so a deposit that is present but
    past RESOURCE_RADIUS (diamond ore under a surface bot) can still justify a tunnel dig.
    """
    blocks = get(data, "response", "data", "result", "blocks", default={}) or {}
    groups = blocks.get("find") if isinstance(blocks, dict) else None
    out: dict[str, int] = {}
    for group in groups or []:
        if not isinstance(group, dict) or not group.get("name"):
            continue
        found = group.get("found")
        if isinstance(found, (int, float)) and found > 0:
            out[str(group["name"])] = int(found)
    return out


def parse_box(data: dict) -> dict[tuple[int, int, int], str]:
    """view_blocks box -> {(x, y, z): block_name} over every cell of the box.

    `layers` holds one entry per y, each with a row per z whose `blocks` list runs x from the
    centre's `halfWidth` to the west to the same to the east. Used only to find a flat base
    site, so a missing or malformed box just yields no grid.
    """
    box = get(data, "response", "data", "result", "blocks", "box", default=None)
    if not isinstance(box, dict):
        return {}
    centre = box.get("center")
    half = box.get("halfWidth")
    if not isinstance(centre, dict) or not isinstance(half, int):
        return {}
    try:
        cx = int(centre["x"])
    except (KeyError, TypeError, ValueError):
        return {}
    out: dict[tuple[int, int, int], str] = {}
    for layer in box.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        y = layer.get("y")
        for row in layer.get("rows") or []:
            if not isinstance(row, dict):
                continue
            z = row.get("z")
            blocks = row.get("blocks") or []
            if not isinstance(y, int) or not isinstance(z, int):
                continue
            for i, name in enumerate(blocks):
                out[(cx - half + i, y, z)] = str(name)
    return out


async def scan_ground(executor, pos, dy: int = 0,
                      why: str = "read the terrain around the bot to place the home base on level ground"
                      ) -> dict:
    """One `view_blocks box` read around the bot, for the flat base-site search.

    `half_width`/`half_height` are the server's maxima (4 and 3), which is exactly the reach
    the frame footprint needs: 5 cells along x, 5 layers up from the anchor's ground. `dy`
    lifts the read's centre, so a second read four above the bot's feet extends the scanned
    column from `feet-3` to `feet+7` and covers the level ground a bot standing in a dip has
    overhead. Returns the coerced result for `parse_box`.
    """
    cx, cy, cz = (int(math.floor(v)) for v in pos)
    res = await executor.read(
        "view_blocks",
        {"box": {"x": cx, "y": cy + dy, "z": cz, "half_width": 4, "half_height": 3}},
        why,
    )
    return coerce(res.data, res.text)


def parse_cells(data: dict) -> dict[tuple[int, int, int], str]:
    """view_blocks cells -> {(x, y, z): block_name} for every requested cell.

    `cells` holds one entry per coordinate with its `name` and `shape`. Used to read the portal
    opening cell by cell before the frame is built; a malformed reply just yields no grid.
    """
    cells = get(data, "response", "data", "result", "blocks", "cells", default=None)
    if not isinstance(cells, list):
        return {}
    out: dict[tuple[int, int, int], str] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        try:
            key = (int(cell["x"]), int(cell["y"]), int(cell["z"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = str(cell.get("name"))
    return out


def portal_interior(base: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """The six cells of the portal opening above the support layer.

    The frame is a west-east plane at z = bz with its bottom row on the ground (see
    `targets.flat_build_site`), so the opening is the 2x3 interior at (bx+2..bx+3, by+1..by+3).
    These cells must end up air for the portal to form and they sit *above* the support layer.
    """
    bx, by, bz = base
    return [(x, y, bz) for x in (bx + 2, bx + 3) for y in (by + 1, by + 2, by + 3)]


def frame_volume(base: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Every cell of the frame plane the build may touch: the 5-wide ground row and the 5 layers up.

    The frame is a west-east plane at z = bz (`targets.FRAME_WIDTH` by `FRAME_HEIGHT`, the bottom
    row on the ground). Its opening is the 2x3 middle (`portal_interior`); the rest is the border
    the build raises and the ground row it digs out and re-places as obsidian. The read has to
    cover the whole plane, not just the opening: a block the build refuses anywhere in it (a
    container in a border cell, see `Agent._note_frame_cells`) needs its name to be dug by
    `clear_frame`.
    """
    bx, by, bz = base
    return [(x, y, bz) for x in range(bx, bx + 5) for y in range(by, by + 5)]


def frame_obsidian_cells(base: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """The ten obsidian cells a finished nether portal frame holds.

    A portal needs obsidian in every block that touches the 2x3 opening; the four outer corners
    are free (the run fills them with cobblestone). So the ten touching cells are up both sides
    and along the top and bottom rows between the corners -- reading exactly these tells a
    finished frame from a partial one without guessing from the pack's obsidian count.
    """
    bx, by, bz = base
    cells = [(bx + 2, by, bz), (bx + 3, by, bz)]
    cells += [(bx + 1, by + dy, bz) for dy in (1, 2, 3)]
    cells += [(bx + 4, by + dy, bz) for dy in (1, 2, 3)]
    cells += [(bx + 2, by + 4, bz), (bx + 3, by + 4, bz)]
    return cells


def frame_wanted_blocks(base: tuple[int, int, int]) -> dict[tuple[int, int, int], str]:
    """The block each frame ring cell should hold: obsidian round the opening, cobblestone corners."""
    bx, by, bz = base
    wanted = {cell: "obsidian" for cell in frame_obsidian_cells(base)}
    wanted.update({cell: "cobblestone" for cell in (
        (bx + 1, by, bz), (bx + 4, by, bz), (bx + 1, by + 4, bz), (bx + 4, by + 4, bz))})
    return wanted


async def scan_frame_volume(executor, base: tuple[int, int, int]) -> dict:
    """One `view_blocks cells` read of the whole frame plane, for the frame-clearance steps.

    The base scan (`targets.flat_build_site`) only proves clearance for a site *it* chose; a
    base the loop adopted from a standing bed, or one taken from the `base_target` fallback,
    never got that check. This read lets `clear_portal` dig a stray block out of the opening
    before `build_portal`, and lets `clear_frame` name a cell the build itself refused to touch.
    Returns the coerced result for `parse_cells`.
    """
    cells = [{"x": x, "y": y, "z": z} for x, y, z in frame_volume(base)]
    res = await executor.read(
        "view_blocks",
        {"cells": cells},
        "read the portal frame plane for stray and refused cells before the frame is built",
    )
    return coerce(res.data, res.text)


def parse_portal_corner(data: dict) -> tuple[int, int, int] | None:
    """view_blocks find (nether_portal) -> the opening's corner (min x, min y, min z), or None.

    Deliberately *not* radius-limited: `parse_resources` keeps only hits inside `RESOURCE_RADIUS`,
    but a portal a previous run built can stand further off, and it is the one block that lets a
    restart re-anchor its home base. The frame is a west-east plane (`actions._portal_site`) whose
    2x3 opening starts two blocks east of, and one above, the base anchor at the frame's z, so the
    lowest, westernmost portal cell inverts the site: base = corner - (2, 1, 0).
    """
    blocks = get(data, "response", "data", "result", "blocks", default={}) or {}
    groups = blocks.get("find") if isinstance(blocks, dict) else None
    cells: list[tuple[int, int, int]] = []
    for group in groups or []:
        if not isinstance(group, dict) or group.get("name") != PORTAL_BLOCK:
            continue
        for e in group.get("listed") or []:
            if not isinstance(e, dict):
                continue
            try:
                cells.append((int(e["x"]), int(e["y"]), int(e["z"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not cells:
        return None
    return (min(c[0] for c in cells), min(c[1] for c in cells), min(c[2] for c in cells))


def parse_surface_height(text: str) -> int | None:
    """A `navigate` height-required answer -> the column's surface feet height, or None.

    A `navigate` without a height resolves an unambiguous outdoor column by itself, but refuses
    to guess between floors when one column carries several: it stops with
    `[NAVIGATION_HEIGHT_REQUIRED] ... observed standable feet heights: 74, -15, -54`. The floors
    are listed highest first, so the largest one is the open surface; that is the cell the bot
    has to climb back to before the base is anchored. Anything else (a real navigation, a
    different failure, an empty text) has no such line and returns None.
    """
    m = re.search(r"standable feet heights:\s*([^.]*)", text or "")
    if not m:
        return None
    floors = [int(v) for v in re.findall(r"-?\d+", m.group(1))]
    if not floors:
        return None
    # Prefer the highest floor, but the caller must verify it's actually dry/open
    return max(floors)


def parse_ground_height(text: str) -> int | None:
    """A `[NAVIGATION_TARGET_UNSUPPORTED]` answer -> the column's real ground height, or None.

    A `navigate` whose target cell has no footing refuses with
    `[NAVIGATION_TARGET_UNSUPPORTED] No usable footing was observed within 1 blocks of <x>, <y>,
    <z>. The ground in that column is at y=<ground>.` The refusal names the ground it *did* find,
    which is exactly the height a base anchor lost to its own digging has to be repaired to (live,
    run11: the bot quarried its home base to air, the anchor stayed at the old surface height, and
    every `go_home` answered the same message for 230 ticks). Only that line yields a value; any
    other failure, an unrelated navigation or an empty text returns None.
    """
    m = re.search(r"ground in that column is at y=\s*(-?\d+)", text or "")
    return int(m.group(1)) if m else None


def parse_block_above(text: str) -> tuple[int, int, int] | None:
    """A `[CHEST_BLOCKED]` answer -> the cell whose block stops the chest lid, or None.

    `use_container` refuses to open a chest that has an opaque full cube directly above it:
    `[CHEST_BLOCKED] The chest at (x, y, z) cannot open: cobblestone is directly above it at
    (x, y+1, z).` The refusal names both the chest and the offending block; the cell after
    "above it at" is the one the bot has to dig so the lid can swing. Only that phrase yields a
    value; any other failure or an empty text returns None.
    """
    m = re.search(r"above it at \((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", text or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse_refused_cells(text: str) -> list[tuple[int, int, int]]:
    """A `[BUILD_INCOMPLETE]` answer -> the cells the build refused, in the order it named them.

    `build_structure` reports a cell it could not work as `... 1 were refused (229,62,222:
    movement policy prohibits breaking this block).` The movement policy refuses a break it was
    not asked for, and a *container* always needs that explicit request, so a container standing
    in a frame cell is never dug by the build itself (see `Agent._note_frame_cells`). Only the
    "prohibits breaking" refusals are returned: a cell the route merely failed to reach is
    retried by the build and must not be dug out from under it. Any other failure yields [].
    """
    if "[BUILD_INCOMPLETE]" not in (text or ""):
        return []
    out: list[tuple[int, int, int]] = []
    for m in re.finditer(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+):([^)]*)\)", text):
        if "prohibits breaking" in m.group(4):
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


async def scan_portal_corner(executor) -> tuple[int, int, int] | None:
    """One `view_blocks find` for a lit portal anywhere in the loaded chunks -> its corner.

    Used while the loop has no base yet: a fresh run finds nothing and `establish_base` proceeds
    as before, while a restart finds the portal it left standing and adopts the base from it
    (`graph._adopt_portal_base`).
    """
    res = await executor.read(
        "view_blocks",
        {"find": {"block_names": [PORTAL_BLOCK], "limit": PORTAL_SCAN_LIMIT}},
        "find a lit portal a previous run left standing",
    )
    return parse_portal_corner(coerce(res.data, res.text))


def _age_seconds(now_iso: Any, then_iso: Any) -> float | None:
    if not (isinstance(now_iso, str) and isinstance(then_iso, str)):
        return None
    from datetime import datetime, timezone

    def parse(s: str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    now, then = parse(now_iso), parse(then_iso)
    if now and then:
        return max(0.0, (now - then).total_seconds())
    return None


class PerceptionBlind(RuntimeError):
    """The world could not be read, so there is nothing to decide from.

    Raised rather than returned. A `view_status` that answers with no position is the shape a dead
    Minecraft connection leaves behind, and everything downstream copes with it silently: `pos`,
    `health` and `food` stay None, every distance predicate reads false, and the candidate list
    collapses to whatever survives without any fact - so the loop keeps deciding. Live, run72: the
    connection ended, `read` reported it, and the next four ticks were logged in two seconds at
    `position: None`, `health: None`, all four `get_wood`, all four refusals.

    Deciding from an empty state is worse than stopping. The run has nothing to act on and every
    action it takes is a guess about a world it cannot see.
    """


async def build(executor) -> GameState:
    """Perceive through the MCP read tools (ungated, cheap)."""
    status = await executor.read("view_status", {}, "read live bot vitals and position")
    status_data = coerce(status.data, status.text)
    # Walk the event backlog to the newest events; reactions only use fresh ones.
    events, unread, preview = await _drain_events(executor)
    gs = parse_status(status_data)
    if gs.pos is None:
        raise PerceptionBlind(
            "view_status answered without a position; the Minecraft connection is gone or the "
            f"server is not answering (text: {(status.text or '')[:120]!r})")
    gs.events = _summaries(events)
    gs.unread = unread
    gs.notifications = _previews(preview)
    now_iso = get(status_data, "response", "data", "result", "situation", "observedAt")
    gs.drops = _count_drops(events, now_iso)
    # `view_blocks find` takes at most 8 block names per call, and the resource group alone
    # already uses all eight, so each group is scanned separately and merged; a failed group
    # just leaves its names out of the state.
    gs.resources = {}
    gs.rescue_ground = {}
    for names, limit, why in ((RESOURCE_NAMES, RESOURCE_LIMIT,
                               "scan nearby resources for the next step"),
                              (RESCUE_GROUND_NAMES, GROUND_LIMIT,
                               "scan standable ground for navigate targets"),
                              (CHAIN_NAMES, CHAIN_LIMIT,
                               "scan the lava to pour on and a chest at the base")):
        try:
            blocks = await executor.read(
                "view_blocks",
                {"find": {"block_names": list(names), "limit": limit}},
                why,
            )
            payload = coerce(blocks.data, blocks.text)
            parsed = parse_resources(payload, limit)
            if names == RESCUE_GROUND_NAMES:
                # The water rescue is the one read that wants the ground past 64 m (live,
                # run19b: the shore sat at 65.1 m and the 64 m scan left only a cave floor
                # to aim at) and the rock a cave pour leaves underfoot (live, run20). Same
                # payload, wider bound, kept in its own pool -- `resources` keeps the
                # soil-only view `explore` and `base_target` were built on.
                gs.rescue_ground = parse_resources(payload, limit, RESCUE_RADIUS)
                parsed = {q: hits for q, hits in parsed.items() if q in GROUND_NAMES}
            gs.resources.update(parsed)
            gs.resource_found.update(parse_resource_counts(payload))
        except Exception:
            pass
    # The Nether group is a fourth read and only earns its place down there: in the Overworld it
    # finds nothing, so `explore` gets no heading and `hunt_blaze` no route.
    if _in_nether_dimension(gs.dimension):
        try:
            blocks = await executor.read(
                "view_blocks",
                {"find": {"block_names": list(NETHER_CHAIN_NAMES), "limit": RESOURCE_LIMIT}},
                "scan the nether bricks that lead to a fortress and the spawners inside it",
            )
            payload = coerce(blocks.data, blocks.text)
            # Read to `FORTRESS_RADIUS`, not the 64 m of every other group: this one exists to
            # spot a fortress from far off, and 64 m is a proximity bound for a resource the bot
            # walks to. Same answer, wider keep.
            gs.resources.update(parse_resources(payload, RESOURCE_LIMIT, FORTRESS_RADIUS))
            gs.resource_found.update(parse_resource_counts(payload))
        except Exception:
            pass
        # The heading is a second read, off the server's own bookkeeping rather than the bot's
        # reasoning about stripes: it is what turns the search from one compass line into
        # coverage of the ground nobody has looked at yet. The bricks read above say what to walk
        # to; this says where to walk to make them show up.
        try:
            frontier = await executor.read(
                "query_bot_data",
                {"sql": FRONTIER_HEADING_SQL},
                "read the compass heading to the nearest chunk edge the world has not observed",
            )
            gs.frontier_heading = parse_frontier_heading(coerce(frontier.data, frontier.text))
        except Exception:
            gs.frontier_heading = None
    return gs


def _in_nether_dimension(dimension: Any) -> bool:
    """Whether a `view_status` dimension string names the Nether."""
    return "nether" in str(dimension or "").lower()


# The depth band the narrative reads out of a raw `y`, against the one frame every world shares:
# bedrock sits near -64, sea level near 63, and diamonds are below 16. The surface floor is set
# a little under sea level so a bot a few blocks into a hillside still reads as "at the surface";
# the deep floor is the diamond line, the one depth change a step actually keys on. The three
# bands are coarse on purpose - the head needs where it is, not the exact altitude.
SURFACE_Y = 55
DEEP_Y = 16
# Coarse vertical relation to a target (e.g. the home base): `level` within this many blocks, and
# `far above`/`far below` once the climb is more than a short shaft. Direction, not degrees.
VERTICAL_LEVEL = 4
VERTICAL_FAR = 20


def _depth_band(y: float) -> str:
    """Where a `y` sits in the shared global frame, in words a step can act on."""
    if y >= SURFACE_Y:
        return "at the surface"
    if y >= DEEP_Y:
        return "underground"
    return "deep underground"


def _rough_vertical(dy: float) -> str:
    """Coarse up/down relation for `dy = target_y - bot_y` (blocks); `level` when near even."""
    if dy >= VERTICAL_FAR:
        return "far above"
    if dy > VERTICAL_LEVEL:
        return "above"
    if dy <= -VERTICAL_FAR:
        return "far below"
    if dy < -VERTICAL_LEVEL:
        return "below"
    return "level"


def _health_band(health: float) -> str:
    """What being below full health means, in the words the rules would use about it.

    Only one band is claimed, because only one is true without knowing more: any damage is
    damage. The number carries the rest, and a ladder here would put words in the narrative that
    no rule in `actions` acts on.
    """
    return "wounded"


def _food_band(food: float) -> str:
    """The band a hunger value falls in, named after what the loop does with it.

    These are the same two floors the candidates use - `EAT_HUNGER` and `REGEN_HUNGER` - so the
    narrative cannot claim a state the loop is not acting on. The middle band is the one that
    costs a bot its health: between the two floors it neither eats nor regenerates, and it sits
    there until something hits it. A three-way "fine / middling / critical" wording folds that
    four-point gap into the top band and hides it.
    """
    if food < EAT_HUNGER:
        return "below the eating floor"
    if food < REGEN_HUNGER:
        return "not regenerating"
    return "regenerating"
