from __future__ import annotations

import asyncio
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
# The Nether's floor is in none of those names. Down there the ground is netherrack, crimson_nylium
# and basalt, so `_known_footings` answered empty in every Nether run and every `navigate` target
# beside a gate was picked blind: live, run198 it0-599, six hundred `leave_portal` ticks aimed at
# (5,74,13) - two blocks above the floor the server named as y=71 - each answered
# `[NAVIGATION_TARGET_UNSUPPORTED]`. Read as its own call, because the ground cast already sits at
# the eight names one `view_blocks find` accepts.
NETHER_GROUND_NAMES = ("netherrack", "crimson_nylium", "basalt")
NETHER_GROUND_LIMIT = 24
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
# The stew pantry, read in both dimensions: a bowl, a red and a brown mushroom brew a meal
# whose bowl comes back from every bite, and both mushrooms grow on netherrack as well as on
# Overworld soil. Its own `view_blocks find` group because the resource group already holds all
# eight names one call accepts.
MUSHROOM_NAMES = ("red_mushroom", "brown_mushroom")
# Eight, not four: `gather_mushrooms` asks for a batch, and a scan that loads only four mushrooms
# cannot tell the pantry whether the ground is worth a walk. The call accepts eight names, and
# this group has two.
MUSHROOM_LIMIT = 8
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
# The pearl hunt's own landmark. Endermen are the one hostile the warped forest fills on its own -
# every other Nether biome spawns them rarely, beside a pack that makes the hunt a fight - so the
# biome is what a pearl march should steer at. The server sampled the biome palette of every chunk
# it loaded (`frontier_chunk_biomes`, joined to a heading in `frontier_navigation`), so a warped
# forest the run has already stood in can be named instead of betting on a compass line. No row
# means this run has never loaded one; the march then falls back to the unexplored frontier.
WARPED_FOREST_SQL = ("SELECT heading_degrees, distance_blocks, chunk_x, chunk_z "
                     "FROM frontier_navigation WHERE dimension = 'the_nether' "
                     "AND dominant_biome = 'warped_forest' "
                     "ORDER BY distance_blocks ASC LIMIT 1")
# Groups whose nearest hit is worth naming in the narrative (Laya reads that text).
NAMED_RESOURCES = RESOURCE_NAMES + ("lava", "spawner", "nether_bricks", "nether_brick_fence", "nether_brick_stairs",
                                    "end_portal_frame", "red_mushroom", "brown_mushroom")
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
# The rescue's ground scan reads deeper than `GROUND_LIMIT`. In open ocean the nearest footholds
# are the seabed right under the bot (water overhead, no foothold) and the odd flooded cave with
# air above; the actual shore, the one cell a swimmer can finish at, sits past them and fell out
# of the nearest-16 cut, so it never reached the state and `nearest_ground` dove to the cave
# instead (live, run116: adrift at (-1,62,251), every `escape_water` aimed 30 m down at a cave
# floor, ~50 ticks of dive-and-surface until the bot drifted close enough for the shore to enter
# the scan on its own). Read far enough that a shore within `RESCUE_RADIUS` survives the cut.
RESCUE_GROUND_LIMIT = 64
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


# --- The durable ledger: what this world already contains ---
# `action_executions` and `events` are the server's own record of everything this bot has done in
# this world. They outlive a death, a dimension change and a restart, and unlike `view_blocks`
# they are not bounded by loaded chunks: a run that starts 300 m from its own base cannot see the
# gate it lit in a previous run, but the ledger can. `query_bot_data` caps a read at 100 rows and
# 64 KiB, so every query here is narrow and ordered, and the milestone it answers is decided in
# `progress.py`, not here.
#
# A lit portal block is only ever *recorded* in a chunk the server actually scanned, which is why
# the chunk table answers "does this world have a burning gate" even when the chunk is unloaded
# now: the material row was written the tick the portal was observed.
GATE_CHUNK_SQL = (
    "SELECT fc.dimension, fc.chunk_x, fc.chunk_z FROM frontier_chunk_materials m "
    "JOIN frontier_chunks fc ON fc.chunk_key = m.chunk_key "
    "WHERE m.material = 'nether_portal' ORDER BY fc.first_observed_at DESC LIMIT 20")

# `activate_portal` is the only action that reports the state of a frame it was asked to light:
# `portalAfter` for the Nether gate, `eyesAfter` for the twelve sockets of the end frame. Both
# are read from the world by the server, so they are the durable answer to "how far is the frame
# filled" that no scan and no per-run counter can give.
PORTAL_LEDGER_SQL = (
    "SELECT json_extract(record_json, '$.arguments') AS args, "
    "json_extract(record_json, '$.terminal.output.result') AS result "
    "FROM action_executions "
    "WHERE json_extract(record_json, '$.action') = 'activate_portal' "
    "ORDER BY json_extract(record_json, '$.admittedAt') DESC LIMIT 20")

# A bed the bot placed or slept in is a respawn point the world still holds. `sleep` reports the
# bed it laid; `place_block` reports the cell it put a `white_bed` in.
BED_LEDGER_SQL = (
    "SELECT json_extract(record_json, '$.arguments') AS args, "
    "json_extract(record_json, '$.terminal.output.result') AS result "
    "FROM action_executions "
    "WHERE json_extract(record_json, '$.action') IN ('place_block', 'sleep') "
    "ORDER BY json_extract(record_json, '$.admittedAt') DESC LIMIT 40")

# Every dimension the bot has ever been carried into. A run that starts in the Nether has no
# crossing to observe, but the server wrote one down when it happened.
DIMENSION_LEDGER_SQL = (
    "SELECT json_extract(payload_json, '$.to') AS to_dim FROM events "
    "WHERE event_type = 'player_dimension_change' "
    "ORDER BY observed_at DESC LIMIT 40")

# One saved eye-of-ender bearing is enough: `locate_stronghold` reuses the same durable
# `search_id`, so a run that already has one does not owe the estimate throw again.
BEARING_SQL = (
    "SELECT search_id, state, bearing_degrees FROM stronghold_throws "
    "WHERE search_id = 'stronghold' ORDER BY thrown_at DESC LIMIT 1")

# The frame the locate phase confirmed, as the server wrote it down. This is the stronghold's only
# durable address: the survey stops the moment a frame *loads*, so a confirmed locate leaves the bot
# beside the estimate rather than beside the frame, and without this row the loop has nowhere to
# walk to (run173: locate succeeded three times, `activate_end_portal` never once resolved).
STRONGHOLD_FRAME_SQL = (
    "SELECT payload_json FROM events WHERE event_type = 'stronghold_located' "
    "ORDER BY event_id DESC LIMIT 1")

# The checkpoint the loop writes itself (`graph.Agent._write_checkpoint`): the counters no ledger
# can prove, such as how many pearl waves this run has already spent.
CHECKPOINT_PREFIX = "mlaya checkpoint:"
CHECKPOINT_READ_SQL = (
    "SELECT note FROM notes WHERE note LIKE 'mlaya checkpoint%' "
    "ORDER BY note_id DESC LIMIT 5")


def _query_rows(payload: Any) -> list[Any]:
    """The rows of a `query_bot_data` answer, tolerant of the wrapper's shape."""
    rows = get(payload, "response", "data", "result", "query", "rows")
    return rows if isinstance(rows, list) else []


def _json_cell(value: Any) -> Any:
    """A SQLite column that holds JSON, read as the object it is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def parse_gate_chunks(payload: Any) -> list[tuple[str, int, int]]:
    """The chunks the server has seen a lit portal block in, as `(dimension, chunk_x, chunk_z)`."""
    out: list[tuple[str, int, int]] = []
    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        if len(cells) < 3:
            continue
        dim = cells[0] if isinstance(cells[0], str) else ""
        x, z = _num(cells[1]), _num(cells[2])
        if x is None or z is None:
            continue
        out.append((dim, int(x), int(z)))
    return out


def parse_portal_ledger(payload: Any) -> list[dict]:
    """Every `activate_portal` this bot has run, newest first, as `{args, result}` dicts."""
    out: list[dict] = []
    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        if len(cells) < 2:
            continue
        args = _json_cell(cells[0])
        result = _json_cell(cells[1])
        if not isinstance(result, dict):
            continue
        out.append({"args": args if isinstance(args, dict) else {}, "result": result})
    return out


def parse_bed_ledger(payload: Any) -> list[tuple[int, int, int]]:
    """Cells a bed was laid in, newest first. A bed the placement refused is not in the world."""
    out: list[tuple[int, int, int]] = []
    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        if len(cells) < 2:
            continue
        result = _json_cell(cells[1])
        if not isinstance(result, dict):
            continue
        bed = result.get("sleep", {}).get("bedPosition") if "sleep" in result else None
        if bed is None:
            placement = result.get("placement") or {}
            if placement.get("requestedBlock") not in ("bed", "white_bed"):
                continue
            if not (placement.get("placed") or placement.get("confirmed")):
                continue
            bed = placement.get("target")
        if not isinstance(bed, dict):
            continue
        x, y, z = (_num(bed.get("x")), _num(bed.get("y")), _num(bed.get("z")))
        if x is None or y is None or z is None:
            continue
        out.append((math.floor(x), math.floor(y), math.floor(z)))
    return out


def parse_dimension_ledger(payload: Any) -> list[str]:
    """The dimensions this bot has been carried into, newest first."""
    out: list[str] = []
    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        if cells and isinstance(cells[0], str):
            out.append(cells[0])
    return out


def parse_bearing(payload: Any) -> bool:
    """Whether a stronghold bearing is saved in the server's durable search.

    A row is written the moment a throw starts and only reaches `tracking`/`observed` once the
    eye's flight was recorded; `pending` is a throw that never landed, which is no bearing.
    """
    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        if len(cells) >= 2 and isinstance(cells[1], str) and cells[1] != "pending":
            return True
    return False


def parse_frame_confirmation(payload: Any) -> tuple[int, int, int] | None:
    """The end portal frame cell a locate confirmation carried, however it was nested.

    Read from both the tool's own answer and the `stronghold_located` ledger row, which is the same
    object the server writes. The frame is a block, so the cell is an integer triple; anything that
    is not a full coordinate (an estimate, which has no y) is not a cell and returns None.
    """
    def cell(value: Any) -> tuple[int, int, int] | None:
        if not isinstance(value, dict):
            return None
        position = value.get("position")
        if not isinstance(position, dict):
            return None
        x, y, z = position.get("x"), position.get("y"), position.get("z")
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y, z)):
            return (int(x), int(y), int(z))
        return None

    def walk(cur: Any) -> tuple[int, int, int] | None:
        found: tuple[int, int, int] | None = None
        stack = [cur]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if "confirmation" in node:
                    confirmed = cell(node["confirmation"])
                    if confirmed is not None:
                        return confirmed
                found = found or cell(node)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return found

    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        for raw in cells:
            frame = walk(_json_cell(raw))
            if frame is not None:
                return frame
    return walk(payload)


def parse_portal_result(payload: Any) -> dict | None:
    """The `portal` evidence a settled `activate_portal` carried, however the answer nested it.

    `portal.eyesAfter` is the server counting the eyes standing in the twelve sockets of the end
    frame -- the world's own number, read on the tick the call ran. It is also present on a
    refusal: `[PORTAL_MISSING_EYES]` reports the frame as it is, so a bot that carried too few
    eyes learns how many the frame already held. The wrapper has changed shape before, so the
    search is the same tolerant walk `_find_container` uses rather than one fixed path.
    """
    stack = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if isinstance(cur.get("portal"), dict):
                return cur["portal"]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def parse_checkpoint(payload: Any) -> dict:
    """The newest checkpoint note this loop wrote for this world, as a dict.

    Notes are the loop's own memory: the server's tables record actions, not decisions, so the
    counters that only the run knows (pearl waves spent, barter rolls made, the cells the powder
    was stowed into) live here. Newest first, and a note whose JSON has been cut off is no
    checkpoint -- an empty dict means "nothing to restore", which is the ordinary case on a world
    this loop has never run on.
    """
    for row in _query_rows(payload):
        cells = row if isinstance(row, list) else [row]
        if not cells or not isinstance(cells[0], str):
            continue
        note = cells[0]
        if not note.startswith(CHECKPOINT_PREFIX):
            continue
        try:
            data = json.loads(note[len(CHECKPOINT_PREFIX):].strip())
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


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


def parse_warped_forest(payload: Any) -> tuple[float, float, int, int] | None:
    """`WARPED_FOREST_SQL` -> (heading, distance, chunk_x, chunk_z) of the nearest recorded
    warped forest, or None when this run has never loaded one.

    The heading is the server's own compass bearing to that chunk, the same shape
    `parse_frontier_heading` returns, so `explore_frontier` takes it unchanged. No row is not an
    error: the biome has not been seen, and the march falls back to the unexplored frontier.
    """
    rows = get(payload, "response", "data", "result", "query", "rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        cells = row if isinstance(row, list) else [row]
        if len(cells) < 4:
            continue
        heading = _num(cells[0])
        distance = _num(cells[1])
        chunk_x = _num(cells[2])
        chunk_z = _num(cells[3])
        if None in (heading, distance, chunk_x, chunk_z):
            continue
        return (round(heading % 360.0, 1), float(distance), int(chunk_x), int(chunk_z))
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
PHASE_NETHER_PEARLS = "You are a Minecraft player hunting endermen for ender pearls"
PHASE_STRONGHOLD = "You are a Minecraft player finding the stronghold"
PHASE_END = "You are a Minecraft player fighting the ender dragon"

# One row per run phase: the persona sentence (System 1 head + System 2 framing), a stable slug
# for the logged state (the `phase` field the server fills is the *time of day*, null in the
# Nether, so System 2 was reading an empty box where the run phase is actually well defined), and
# a short brief of what the phase is trying to achieve - the preconditions System 2 needs so it
# can tell a stuck label from a step that still serves the goal.
_PHASE_ROWS = {
    "diamond_kit": (PHASE_TECH,
                    "mine and craft a full diamond kit: diamond pickaxe and sword to hand, "
                    "diamond armour worn on all four slots."),
    "prepare_nether": (PHASE_PREP,
                       "before crossing, finish the Nether kit: shield, a bow with arrows, a "
                       "cooked-food reserve, a bed, a chest standing at the base, and a lit "
                       "Overworld portal."),
    "nether_rods": (PHASE_NETHER,
                    "hunt blazes at the fortress for blaze rods, turn them into powder and bank "
                    "that powder in the chest at the gate, then leave the Nether with the haul."),
    "nether_pearls": (PHASE_NETHER_PEARLS,
                      "the rod haul is banked: barter gold with piglins and hunt endermen for the "
                      "pearls the eyes of ender cost, then leave the Nether with them."),
    "stronghold": (PHASE_STRONGHOLD,
                   "craft the eyes of ender, find the stronghold and activate the end portal."),
    "end_dragon": (PHASE_END, "enter the End and defeat the ender dragon."),
}


def _phase_key(source) -> str:
    """The run phase as a stable slug, from one source of truth (`GameState` or `as_state()` dict).

    The branch logic both `phase_persona` and `phase_brief` read, so neither interface can frame the
    run differently than the other. The phases follow the world: the dimension owns nether and end
    work; in the Overworld a finished diamond kit (worn, diamond pickaxe and sword to hand) says the
    climb is over and only the crossing is left. In the Overworld the stronghold follows the Nether
    haul: once the blaze powder is banked (`powder_stowed`) and until the end portal stands active
    (`end_portal_active`), the run is out for eyes of ender and the stronghold (docs/stronghold.md).
    Both are loop latches that only ever set, so the phase cannot flicker with the world.
    """
    def field(name: str, default=None):
        return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)
    dim = str(field("dimension") or "").lower()
    if "end" in dim:
        return "end_dragon"
    if "nether" in dim:
        # The Nether has two jobs, and the run is in one of them at a time: the rod haul that banks
        # the powder, and the pearl waves that pay for the eyes (docs/stronghold.md, "Nether in
        # Wellen"). `Agent._inject` derives `nether_trip` from `_pearl_wave_trip`, the same predicate
        # the planner's Nether block splits on, so the persona cannot call the trip a blaze hunt
        # while every offer down there is a pearl hunt (live, run220 it228: `search_enderman` chosen
        # on the tick, "killing blazes for blaze rods" in the sentence above it).
        return "nether_pearls" if field("nether_trip") else "nether_rods"
    if field("powder_stowed") and not field("end_portal_active"):
        return "stronghold"
    tools = field("tools") or {}
    armor = field("armor") or {}

    def worn(slot: str) -> bool:
        meta = armor.get(slot) if isinstance(armor, dict) else None
        return isinstance(meta, dict) and meta.get("tier") == "diamond"

    kit = (all(worn(slot) for slot in ("helmet", "chestplate", "leggings", "boots"))
           and tools.get("pickaxe") == "diamond" and tools.get("sword") == "diamond")
    return "prepare_nether" if kit else "diamond_kit"


def phase_persona(source) -> str:
    """The phase sentence for both model interfaces, from one source of truth (`_phase_key`)."""
    return _PHASE_ROWS[_phase_key(source)][0]


def phase_brief(source) -> tuple[str, str]:
    """The run phase as `(slug, goal)`: a stable name and what the phase is trying to achieve.

    System 2 reads both - the slug fills the `phase` the server leaves empty outside the Overworld
    day/night cycle, and the goal line tells it the phase's preconditions so it can judge whether a
    stuck label still serves them or should give way.
    """
    key = _phase_key(source)
    return key, _PHASE_ROWS[key][1]


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
    # The nearest recorded warped forest: (heading, distance, chunk_x, chunk_z). Endermen are what
    # that biome fills on its own, so the pearl march steers at it when the run has loaded one
    # (`WARPED_FOREST_SQL`, read in the Nether only).
    warped_forest: tuple[float, float, int, int] | None = None
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
    # Every loaded dropped-item entity the status scan lists, nearest first:
    # (item name, stack count, distance, position). `drops` above only counts kills from the last
    # `EVENT_REACT_WINDOW_S`; an arrow a skeleton dropped five minutes ago, or string from a spider
    # the reflex killed out of sight, is still worth the walk. The End fight is fought at range,
    # so the supply sweep keys on this list rather than on a fresh kill alone.
    dropped_items: list[tuple[str, int, float | None,
                              tuple[float, float, float] | None]] = field(default_factory=list)
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
    # The gate bed this run laid or adopted, remembered like `chest_pos`. `view_blocks find` only
    # reads loaded chunks, so a bot 280 m off base sees no bed at all and reads the respawn bed as
    # still owed (live, run121 it80: `craft_bed` fired at (228,72,-336) while the gate bed stood
    # three cells from the anchor). Per-run like the base itself.
    bed_pos: tuple[int, int, int] | None = None
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
    # Standing cells an `escape_water` swim was stopped at (`[NAVIGATION_STOPPED]`): the pathfinder
    # timed out before it could reach that cell from the pocket the body is floating in, so aiming
    # at it again can only time out again. The rescue passes these over and picks another foothold.
    rescue_skip: frozenset = frozenset()
    # Stances an `approach_lava` walk already ended on. A lava body the bot has stood beside and
    # still shows no level-0 source cannot produce obsidian from that footing, so the walk picks
    # another body instead of hopping back to it (live, run143: 94 `approach_lava` ticks bouncing
    # between two stances 5.6 blocks apart beside the same two flows, never a source).
    lava_walked: frozenset = frozenset()
    # Lava pools retired after `LAVA_POOL_RETIRE_AFTER` fruitless walks (see `_note_lava_tries`).
    # A walk that keeps ending short of its pour stance and never forms obsidian is not going to
    # start forming it by walking the same twenty-four blocks again, and the base work it steals
    # ticks from is what ends the loop (live, run213 it101-141). Keyed by the walk's target cell;
    # `_lava_retired` groups by `LAVA_POOL_RADIUS`. A countdown, so the pool returns.
    lava_retired: frozenset = frozenset()
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
    # The cell the base layout wants for the furnace has no footing (`[PLACE_NO_SUPPORT]`), so the
    # next attempt names no cell at all and lets the server stand it on one that has (see
    # `_furnace_args`). Only a refusal of *that* attempt gives the furnace up.
    furnace_site_refused: bool = False
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
    chest_above_name: str | None = None
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
    # Candidate cells whose block the ground will not clear: a `[PLACE_TARGET_OCCUPIED]` naming a
    # block `make_room` cannot break, or a dig `make_room` was itself refused at with "movement
    # policy prohibits breaking this block" (`Agent._note_place`). Unlike `overflow_skip`, which
    # is about the support under a cell, this is about what stands in it: the same cell named
    # again is the same refusal (live, run154: the gate bed on the chest's cell, six ticks).
    placement_refused: frozenset = frozenset()
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
    # Columns `(x, z)` whose surface probe answered `[NAVIGATION_TARGET_WATER]`: the top of that
    # column is a fluid and the server's shore search (8 blocks) found no dry footing. The loop
    # owns this per run and injects it, like `surface`, so `surface_up` aims at a column it has
    # not already proved wet instead of probing the same water every tick (live, run142 it55-57:
    # three identical refusals at (5.51, -19, -115.34), the body never moving).
    surface_wet: frozenset[tuple[int, int]] = frozenset()
    # Cells of the portal opening (`portal_interior`) that hold a stray block, read from one
    # `view_blocks cells` scan while a frame build is pending. The opening sits *above* the
    # support layer and must end up air: a block left there is built around and `activate_portal`
    # answers `[PORTAL_NO_FRAME]` on every tick (live, Nether run: a bed the generic sleep laid
    # "beside the bot" landed in the opening). `clear_portal` digs these cells before the build.
    portal_blocked: list[tuple[int, int, int]] = field(default_factory=list)
    # Cells of the home-base footprint that hold a block piled above ground level, which
    # `flatten_home_base` digs back to air (`actions.FLATTEN_BLOCKS`). Read from one `view_blocks
    # box` while the gate is lit and the bot is home by day (`Agent._scan_base_terrain`), lowest
    # tower cells and nearest cells first. Empty means the ground is already flat, and the leveling
    # step drops out of the offer.
    raised_cells: list[tuple[int, int, int]] = field(default_factory=list)
    # The cells `prepare_bed_pad` builds under the gate bed, as `(x, y, z, block_name)` with
    # `"air"` meaning dig: the three cells of the bed's row beside the frame, dug where a block
    # sits and cobblestone-stood where the floor is missing. Planned from the same box read as
    # `raised_cells` (`Agent._scan_base_terrain`), so it is empty when the pad is already flat or
    # the base chunks are unloaded, and the step drops out of the offer.
    bed_pad: list[tuple[int, int, int, str]] = field(default_factory=list)
    # The loop learned from the server that the frame build has nothing left to place: a
    # `build_portal` answered `ok` while spending no obsidian or cobblestone (`Agent._note_build`).
    # That success resets `_track_stuck`, so the "resume a partial frame" clause would offer the
    # step forever; while this stands `candidates` drops it so the lull work owns the tick (live,
    # run25: 163 identical `build_portal` ticks on an already finished frame, the bot frozen).
    frame_complete: bool = False
    # The base gate has been *seen* burning this run: the scan named a `nether_portal` block, or a
    # crossing proved it. `_portal_lit` is otherwise a scan read, so a gate whose chunk has unloaded
    # - or a bot standing on the other side of it, in the Nether - reads unlit, and the base chain
    # reopens for a gate that already stands (run124 it40-41). Same latch shape as `frame_complete`.
    portal_lit: bool = False
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
    # How many hops toward the Nether exit the terrain refused in a row, at the same cell (Agent
    # `_note_nether_hop`). A hop aimed at a cell with no footing is refused without the body moving,
    # so the same refusal repeats until the aim is shortened (live, run149: sixteen identical
    # `[NAVIGATION_TARGET_UNSUPPORTED]` on one 120 m hop).
    nether_hop_repeats: int = 0
    # Marches (`explore_frontier`) in a row that got no route at all (Agent `_note_explore`).
    # `[EXPLORATION_ROUTE_STOPPED]` leaves the body where it was, so the next tick asked for the
    # same heading and hit the same wall (live, run164 it159-161: three identical eastward marches
    # from (132, 41, 0) into the lava sea). The count turns the next bearing a quadrant
    # (`_clear_march_heading`) so a blocked line is approached from another side, and a pearl wave
    # that burns the whole turn budget goes home instead of marching in place.
    explore_stalls: int = 0
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
    slept_at_stronghold: bool = False
    stronghold_estimated: bool = False
    # The end portal frame the locate phase confirmed (`Agent._note_locate`, restored from the
    # server's `stronghold_located` row). The address the bot has to walk to: a locate "succeeds"
    # as soon as a frame chunk loads, so the confirmation can sit a hundred metres off the body and
    # the frame never appears in the 64 m scan `activate_end_portal` aims at.
    stronghold_frame: tuple[int, int, int] | None = None
    powder_exhausted: bool = False
    # Pearl trips the waves have made (Agent `_note_dimension`): a crossing counted once the rod
    # haul is already stowed (`powder_stowed`), so it never counts the rod trip itself. The waves
    # relaunch into the Nether to hunt pearls until the eyes are paid for or `PEARL_WAVE_MAX_TRIPS`
    # trips are spent, after which the Overworld enderman roam is the fallback. A per-run latch like
    # `barter_attempts`, reset on a cold restart (nothing in the world re-derives it); a restart
    # mid-hunt simply grants a fresh five, which is the safe direction - it never strands the bot.
    pearl_wave_trips: int = 0
    # Which Nether trip the body is on: "rods" (the blaze haul that banks the powder) or "pearls"
    # (the waves that pay for the eyes). Derived every tick by `Agent._inject` from
    # `_pearl_wave_trip`, the same predicate the planner's Nether block splits on, so the phase
    # persona and the offer can never describe two different trips. Empty means "not derived yet",
    # which reads as the rod haul - the trip every run starts with.
    nether_trip: str = ""
    # The cells a blaze-powder stow actually deposited into (Agent `_note_stow`): the storage
    # network's roomy chest, which is not always `chest_pos`. `take_blaze_powder` withdraws from
    # these remembered cells, and `powder_exhausted` only latches once every one of them has
    # come up empty - a top-up aimed at the main chest while the powder sits in an overflow
    # chest would send the bot back through the gate for nothing.
    powder_cells: tuple[tuple[int, int, int], ...] = ()
    # Blaze powder the loop booked into the chest at the gate, counted from the stow and withdraw
    # deltas (`Agent._note_stow`, `_note_take_powder`) rather than read back from the lid. The rod
    # haul and the top-up are sized on it: the Nether is only owed a trip when the powder it can
    # reach is short of the eyes the frame still needs.
    powder_bank: int = 0
    # Sockets of the end portal frame this run has filled, from the eyes `activate_portal` spent
    # (`Agent._note_portal`). The world's own pre-filled sockets are never reported, so the frame
    # is counted from what the bot put in and is only known complete when the server confirms the
    # portal active (`end_portal_active`).
    frame_eyes: int = 0
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
    # The run's milestones as one object, resolved by `progress.progress_of` from the durable
    # ledger, the live scan and the per-run latches in that order, with a `source` map saying
    # which of the three answered each field. Injected like `base`: perception cannot know it,
    # and a tick whose stand is not logged cannot be read afterwards.
    progress: Any = None
    # --- The End fight (Phase 5; parsed from `view_status.endFight`) ---
    # The loaded fight entities, and only the loaded ones: an empty list means "nothing is loaded
    # here", never "all dead" (the server's own schema says so). So the dragon's death is never read
    # from these going empty - it is latched from a kill receipt (`dragon_slain`), and these only aim
    # the next attack while the fight is in render.
    #
    # Each crystal as `(entity_id, x, y, z)`, nearest first; `destroy_end_crystal` needs the id.
    end_crystals: tuple[tuple[int, int, int, int], ...] = ()
    # The one dragon's loaded facts: its entity id (what every dragon tool takes), its health, the
    # server's phase name (`DRAGON_PHASE_NAMES`, e.g. `holding_pattern`/`landing`/`sitting_flaming`),
    # whether it is perched (sitting, and so arrow-immune), and the head cell to aim a perch melee at
    # (the current perched-head estimate, else the predicted landing head). None while unloaded.
    dragon_id: int | None = None
    dragon_health: float | None = None
    dragon_phase: str | None = None
    dragon_perched: bool = False
    perch_head: tuple[int, int, int] | None = None
    # The loaded dragon-breath clouds as `(x, y, z)` cells. The server aborts shelter/perch on breath
    # itself; the loop only logs the clouds so a recorded tick says what the arena held.
    dragon_breath: tuple[tuple[int, int, int], ...] = ()
    # Per-run latches the Agent sets (never perception): the End has been entered at least once
    # (`_note_end_entered`), the dragon has died (`_note_dragon`, from a kill receipt - the run's
    # goal), and the Overworld cell of the active end portal to step into (`_note_portal`, the
    # `activate_portal` centre). Logged so a replayed tick and a cold restart read them back.
    end_entered: bool = False
    dragon_slain: bool = False
    end_portal_cell: tuple[int, int, int] | None = None

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
            "dropped_items": [[name, count, dist, list(pos) if pos else None]
                              for name, count, dist, pos in self.dropped_items],
            "base": self.base,
            "chest_pos": self.chest_pos,
            "chest_above": list(self.chest_above) if self.chest_above else None,
            "chest_above_name": self.chest_above_name,
            "base_adopted": self.base_adopted,
            "furnace_pos": self.furnace_pos,
            "smoker_pos": self.smoker_pos,
            "bed_pos": self.bed_pos,
            "furnace_occupied": self.furnace_occupied,
            "furnace_contents": dict(self.furnace_contents),
            "furnace_blocked": self.furnace_blocked,
            "furnace_site_refused": self.furnace_site_refused,
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
            "portal_lit": self.portal_lit,
            "nether_arrival": self.nether_arrival,
            "nether_gate": self.nether_gate,
            # The march's own bearings: the heading to the unexplored frontier and the warped
            # forest the pearl hunt steers at. Logged so a recorded pearl tick can be replayed -
            # the offer turned on both, and a tick that does not say what it saw cannot answer why
            # it walked where it walked.
            "frontier_heading": self.frontier_heading,
            "warped_forest": self.warped_forest,
            "nether_hop_repeats": self.nether_hop_repeats,
            "explore_stalls": self.explore_stalls,
            "nether_ticks": self.nether_ticks,
            "nether_travel": self.nether_travel,
            "mobs": self.mobs,
            "piglins": self.piglins,
            "barter_attempts": self.barter_attempts,
            # The stronghold latches, logged for the same reason the Nether facts are.
            "powder_stowed": self.powder_stowed,
            "powder_exhausted": self.powder_exhausted,
            "pearl_wave_trips": self.pearl_wave_trips,
            "nether_trip": self.nether_trip,
            "stronghold_estimated": self.stronghold_estimated,
            "end_portal_active": self.end_portal_active,
            "slept_at_stronghold": self.slept_at_stronghold,
            "stronghold_frame": list(self.stronghold_frame) if self.stronghold_frame else None,
            "powder_cells": [list(cell) for cell in self.powder_cells],
            "powder_bank": self.powder_bank,
            "frame_eyes": self.frame_eyes,
            "nether_landmarks": [list(cell) for cell in self.nether_landmarks],
            # The End fight: the loaded combat read and the per-run latches, logged for the same
            # reason the stronghold facts are - a replayed tick and a cold restart both read them.
            "end_crystals": [list(c) for c in self.end_crystals],
            "dragon_id": self.dragon_id,
            "dragon_health": self.dragon_health,
            "dragon_phase": self.dragon_phase,
            "dragon_perched": self.dragon_perched,
            "perch_head": list(self.perch_head) if self.perch_head else None,
            "dragon_breath": [list(c) for c in self.dragon_breath],
            "end_entered": self.end_entered,
            "dragon_slain": self.dragon_slain,
            "end_portal_cell": list(self.end_portal_cell) if self.end_portal_cell else None,
            # The resolved milestones and the source of each. Logged for the same reason the
            # furnace's facts were: a run that started on a half-played world and rebuilt its
            # gate cannot be read afterwards unless the tick says what it believed about the
            # gate, the bed and the frame, and which of the three sources told it so.
            "progress": self.progress.as_dict() if self.progress is not None else None,
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
                      "base_adopted", "furnace_pos", "smoker_pos", "bed_pos",
                      "furnace_occupied", "furnace_blocked",
                      "sleeping", "resource_found", "mobs", "piglins", "barter_attempts", "banked",
                      "nether_entered", "portal_lit", "nether_arrival", "nether_gate", "nether_ticks",
                      "nether_travel", "powder_stowed", "powder_exhausted", "pearl_wave_trips",
                      "nether_trip",
                      "stronghold_estimated", "end_portal_active", "slept_at_stronghold",
                      "stronghold_frame", "powder_cells",
                      "powder_bank", "frame_eyes", "frontier_heading", "warped_forest",
                      "explore_stalls", "nether_landmarks",
                      "dragon_id", "dragon_health", "dragon_phase", "dragon_perched",
                      "end_entered", "dragon_slain"):
            value = data.get(field)
            if value is not None:
                setattr(gs, field, value)
        # The End combat reads come back from JSON as lists; the state keeps tuples of ints so a
        # predicate reads a replayed tick the same way as a live one.
        gs.end_crystals = tuple(tuple(int(v) for v in c) for c in (data.get("end_crystals") or ()))
        gs.dragon_breath = tuple(tuple(int(v) for v in c) for c in (data.get("dragon_breath") or ()))
        for field in ("perch_head", "end_portal_cell"):
            value = data.get(field)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                setattr(gs, field, tuple(int(v) for v in value))
        # A logged cell comes back from JSON as a list; the state keeps 3-tuples, and a cell that
        # stays a list is unhashable wherever a predicate puts it in a set (`_bed_reach_stance`
        # does `set(_bed_cells(gs))`, so a replayed tick raised where the live one never could).
        for field in ("base", "chest_pos", "furnace_pos", "smoker_pos", "bed_pos",
                      "nether_arrival", "nether_gate", "stronghold_frame"):
            value = getattr(gs, field)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                setattr(gs, field, tuple(int(v) for v in value))
        if gs.warped_forest is not None:
            # (heading, distance, chunk_x, chunk_z) as logged; the cells stay ints, the bearings
            # floats, so a replayed tick reads the same as the live one.
            heading, distance, chunk_x, chunk_z = gs.warped_forest
            gs.warped_forest = (float(heading), float(distance), int(chunk_x), int(chunk_z))
        for field in ("death_pos",):
            value = getattr(gs, field)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                setattr(gs, field, tuple(float(v) for v in value))
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
        # Dropped items are logged as `[name, count, distance, [x, y, z]]`; the state keeps the
        # position a 3-tuple so a report or a predicate reads it the same way as on the live tick.
        gs.dropped_items = []
        for entry in (data.get("dropped_items") or ()):
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                continue
            name, count, dist, pos = entry
            cell = None
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                cell = tuple(float(v) for v in pos)
            gs.dropped_items.append(
                (str(name), int(count or 1), None if dist is None else float(dist), cell))
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
        prog = data.get("progress")
        if isinstance(prog, dict):
            from .progress import Progress  # lazy: `progress` imports this module
            gs.progress = Progress(**{k: v for k, v in prog.items()
                                      if k in Progress.__dataclass_fields__})
        return gs

    def narrative(self) -> str:
        parts: list[str] = []
        # Where the run stands, before anything else: the model sees only the first window of a
        # long state, so the milestones have to be the first thing in it.
        if self.progress is not None:
            parts.append(self.progress.sentence())
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
            # survives, read against the frame of the dimension the bot is in (the Overworld's
            # bedrock ~-64 / sea level ~63 / diamond line, the Nether's own shell floor); `as_state`
            # still logs the absolute position.
            parts.append(f"{_depth_band(y, self.dimension)} ({where})")
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
        if self.slots_used >= INVENTORY_SLOTS:
            parts.append(f"the pack is full ({self.slots_used}/{INVENTORY_SLOTS} slots)")
        elif self.slots_used >= INVENTORY_NEAR_FULL:
            # "Full" at 34/36 is not full, and the reviewer read it as full: run220 tick 427, where
            # System 2 dropped junk instead of hunting the 18 endermen at 5 blocks because the line
            # said the pack had no room left. Say how much room there is.
            free = INVENTORY_SLOTS - self.slots_used
            parts.append(f"the pack is nearly full ({self.slots_used}/{INVENTORY_SLOTS} slots, "
                         f"{free} free)")
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
        for cls, meta in self.armor.items():
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
        # Older drops outlive the reflex window above, and the End reserve is built from exactly
        # these: arrows from skeletons and string from spiders. Only the supply items are named;
        # the raw drop list also holds every piece of junk a creeper scattered, which is noise.
        from .actions import DROP_SWEEP_RADIUS  # lazy: `actions` imports this module
        supply = [(n, c, d) for n, c, d, _ in self.dropped_items
                  if n in ("arrow", "string") and (d is None or d <= DROP_SWEEP_RADIUS)]
        if supply:
            desc = ", ".join(
                f"{c}x {n}" + (f" at {d:.0f} blocks" if d is not None else "")
                for n, c, d in supply[:3]
            )
            parts.append(f"dropped supply nearby: {desc}")
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
        # The stew pantry is the food this run can renew where it stands, and `gather_mushrooms`
        # is offered on it - but the resource line above names only the four nearest hits, so
        # mushrooms thirty blocks out never reach the state the head reads. Live, run210 it334:
        # `gather_mushrooms` lost to `explore` 0.128 to 0.872 with both on the shortlist; replayed
        # on the same tick with the pantry named, 0.755 to 0.245. The line appears exactly when the
        # walk is possible (a bowl carried, a stack below the cap, a mushroom loaded), so it never
        # promises a meal the pack cannot make.
        from .actions import BOWL_ITEM, STEW_MUSHROOMS, _mushroom_pantry_wanted  # lazy
        if _mushroom_pantry_wanted(self):
            bowls = int(self.inventory.get(BOWL_ITEM, 0))
            parts.append(f"stew pantry: {bowls} bowl{'s' if bowls != 1 else ''} carried, "
                         + ", ".join(f"{int(self.inventory.get(n, 0))} {n}" for n in STEW_MUSHROOMS))
        # The End fight, when the bot is down there. The crystals heal the dragon, so their count
        # is the first thing a step keys on; then the dragon's own health and phase - a sitting
        # dragon is arrow-immune and answered with a perch, a flying one with a bow. An empty arena
        # is "nothing loaded yet", the cue to walk to the fountain so the ring loads, never a kill.
        if "end" in str(self.dimension or "").lower():
            if self.end_crystals:
                parts.append(f"{len(self.end_crystals)} end crystal(s) still loaded on the towers, "
                             "healing the dragon until destroyed")
            if self.dragon_id is not None:
                hp = f"{self.dragon_health:g} health" if self.dragon_health is not None else "health unknown"
                phase = self.dragon_phase or "unknown phase"
                stance = "perched and arrow-immune" if self.dragon_perched else "flying"
                parts.append(f"the ender dragon is loaded: {hp}, {phase}, {stance}")
            elif not self.end_crystals:
                parts.append("no End fight entities are loaded; move to the central fountain so the "
                             "towers, crystals and dragon load")
            if self.dragon_breath:
                parts.append(f"{len(self.dragon_breath)} dragon-breath cloud(s) nearby")
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

    # Every loaded item entity, nearest first (`nearby.droppedItems`, no distance bound). This is
    # what lets the loop walk back for arrows and string after the reflex's own kill window has
    # closed, and it is the only read that sees a drop nobody just killed for.
    for d in nearby.get("droppedItems") or []:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        pos = d.get("position")
        cell = None
        if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
            cell = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        gs.dropped_items.append((str(d["name"]), int(d.get("count") or 1),
                                 _num(d.get("distance")), cell))

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

    # The End fight. Loaded entities only: an empty list is "nothing loaded", never "all dead", so
    # the dragon's death is not read here (that is the kill-receipt latch `dragon_slain`); these aim
    # the next attack while the fight is in render. One dragon at a time is all the arena holds.
    end_fight = sit.get("endFight") or {}
    if isinstance(end_fight, dict):
        crystals: list[tuple[int, int, int, int]] = []
        for c in end_fight.get("crystals") or []:
            if not isinstance(c, dict):
                continue
            eid = c.get("entityId")
            cp = c.get("position")
            if isinstance(eid, int) and isinstance(cp, dict) and all(k in cp for k in ("x", "y", "z")):
                crystals.append((eid, int(cp["x"]), int(cp["y"]), int(cp["z"])))
        gs.end_crystals = tuple(crystals)
        for dr in end_fight.get("dragons") or []:
            if not isinstance(dr, dict) or not isinstance(dr.get("entityId"), int):
                continue
            gs.dragon_id = dr["entityId"]
            gs.dragon_health = _num(dr.get("health"))
            gs.dragon_phase = dr.get("phaseName") if isinstance(dr.get("phaseName"), str) else None
            gs.dragon_perched = bool(dr.get("perched"))
            head = dr.get("headEstimate") or dr.get("landingHeadEstimate")
            if isinstance(head, dict) and all(k in head for k in ("x", "y", "z")):
                gs.perch_head = (int(head["x"]), int(head["y"]), int(head["z"]))
            break  # the arena holds one dragon; the first loaded entry is it
        clouds: list[tuple[int, int, int]] = []
        for cl in end_fight.get("clouds") or []:
            if isinstance(cl, dict) and all(k in cl for k in ("x", "y", "z")):
                clouds.append((int(cl["x"]), int(cl["y"]), int(cl["z"])))
        gs.dragon_breath = tuple(clouds)
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


async def scan_base_box(executor, base: tuple[int, int, int]) -> dict:
    """One `view_blocks box` read of the home-base footprint, for `flatten_home_base`.

    Centred on the anchor and lifted so the box spans the ground floor `by` up to `by+4`; the
    server caps `half_width` at 4, so the field is the 9x9 columns the base furniture and the gate
    all sit within (`actions.FLATTEN_RADIUS` is kept equal to it). Returns the coerced result for
    `parse_box`; the removable cells are filtered out of it by the loop (`Agent._scan_base_terrain`),
    which owns the block whitelist -- `perception` must not import the action layer.
    """
    bx, by, bz = base
    res = await executor.read(
        "view_blocks",
        {"box": {"x": bx, "y": by + 2, "z": bz, "half_width": 4, "half_height": 2}},
        "read the home-base footprint for stray blocks piled above ground level, to level it",
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


def parse_water_column(text: str) -> tuple[int, int] | None:
    """A `[NAVIGATION_TARGET_WATER]` answer -> the column it found water in, or None.

    A height-less `navigate` resolves a column's surface; when that surface is a fluid and no
    dry footing lies within the server's 8-block shore search, it refuses with
    `[NAVIGATION_TARGET_WATER] The ground at 5, -116 is water, and no dry ground the bot can
    stand on lies within 8 blocks of it ...`. The column named in that line is the one the probe
    aimed at, which is exactly what the loop has to remember to aim somewhere else next tick.
    Any other failure, an unrelated navigation or an empty text returns None.
    """
    m = re.search(r"The ground at (-?\d+), (-?\d+) is water", text or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


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


def parse_chest_blocker_name(text: str) -> str | None:
    """The block name in a `[CHEST_BLOCKED]` refusal, or None.

    `[CHEST_BLOCKED] The chest at (x, y, z) cannot open: crafting_table is directly above
    it at (x, y+1, z).` The word before "is directly above it" names the blocker; that is the
    block `collect_block` has to break, since `build_structure` refuses containers.
    """
    m = re.search(r":\s*(\w+)\s+is directly above it at", text or "")
    if not m:
        return None
    return m.group(1)


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


# A `view_status` that answers with a dead Minecraft connection is not always a dead run: the
# host is alive, the bot is in the world, and the server merely stalled. The read is retried
# before the loop declares itself blind (live, run145: one 30 s timeout ended a 200-tick plan at
# iteration 130). Only that shape is retried - a status that answers with something else and no
# position is a world that is really unreadable, and retrying only delays the honest stop.
PERCEPTION_RETRIES = 3
PERCEPTION_RETRY_DELAY = 3.0
DISCONNECT_RE = re.compile(r"MINECRAFT_DISCONNECTED|CONNECTION_LOST|client timed out", re.I)


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


async def _read_status(executor):
    """`view_status`, retried while the answer says the Minecraft connection dropped.

    The transport is answering (the host is up and the tool ran); only the bot's own connection
    stalled. That is the one failure worth a second try: a lag spike or a chunk load that ends in
    a few seconds should not end a run. A status that answers with something else and no position
    is not retried - that is the world really being unreadable.
    """
    status = await executor.read("view_status", {}, "read live bot vitals and position")
    for _ in range(PERCEPTION_RETRIES - 1):
        if parse_status(coerce(status.data, status.text)).pos is not None:
            break
        if not DISCONNECT_RE.search(status.text or ""):
            break
        await asyncio.sleep(PERCEPTION_RETRY_DELAY)
        status = await executor.read("view_status", {}, "read live bot vitals and position")
    return status


async def build(executor) -> GameState:
    """Perceive through the MCP read tools (ungated, cheap)."""
    status = await _read_status(executor)
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
    scan_groups = [(RESOURCE_NAMES, RESOURCE_LIMIT,
                    "scan nearby resources for the next step"),
                   (RESCUE_GROUND_NAMES, RESCUE_GROUND_LIMIT,
                    "scan standable ground for navigate targets"),
                   (CHAIN_NAMES, CHAIN_LIMIT,
                    "scan the lava to pour on and a chest at the base"),
                   (MUSHROOM_NAMES, MUSHROOM_LIMIT,
                    "scan the mushrooms that brew into stew when no animal is in reach")]
    if "nether" in str(gs.dimension or "").lower():
        scan_groups.append((NETHER_GROUND_NAMES, NETHER_GROUND_LIMIT,
                           "scan the Nether's own floor for standable footing"))
    for names, limit, why in scan_groups:
        try:
            blocks = await executor.read(
                "view_blocks",
                {"find": {"block_names": list(names), "limit": limit}},
                why,
            )
            payload = coerce(blocks.data, blocks.text)
            parsed = parse_resources(payload, limit)
            if names == NETHER_GROUND_NAMES:
                # The floor a Nether run stands on goes into the footing pool and nowhere else:
                # netherrack fills every loaded chunk, so counting it in `resource_found` would
                # turn every "found" reading into noise, and `explore`'s ground view is soil-only
                # by construction.
                gs.rescue_ground.update(parse_resources(payload, limit, RESOURCE_RADIUS))
                continue
            if names == RESCUE_GROUND_NAMES:
                # The water rescue is the one read that wants the ground past 64 m (live,
                # run19b: the shore sat at 65.1 m and the 64 m scan left only a cave floor
                # to aim at) and the rock a cave pour leaves underfoot (live, run20), and it
                # reads `RESCUE_GROUND_LIMIT` cells deep so a shore crowded out of the nearest
                # few by the seabed still lands in the pool (live, run116). Same payload, wider
                # bound and depth, kept in its own pool -- `resources` keeps the soil-only view
                # `explore` and `base_target` were built on, capped back at `GROUND_LIMIT`.
                gs.rescue_ground = parse_resources(payload, limit, RESCUE_RADIUS)
                parsed = {q: hits[:GROUND_LIMIT] for q, hits in parsed.items()
                          if q in GROUND_NAMES}
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
        # The pearl hunt's landmark, read in the same breath: a warped forest is the one place
        # endermen fill without the rest of the Nether's hostile pack, so `search_enderman` steers
        # at it instead of walking a compass line. No row means this run has not loaded one yet.
        try:
            warped = await executor.read(
                "query_bot_data",
                {"sql": WARPED_FOREST_SQL},
                "read where the nearest warped forest was recorded, the biome endermen fill",
            )
            gs.warped_forest = parse_warped_forest(coerce(warped.data, warped.text))
        except Exception:
            gs.warped_forest = None
    return gs


def _in_nether_dimension(dimension: Any) -> bool:
    """Whether a `view_status` dimension string names the Nether."""
    return "nether" in str(dimension or "").lower()


# The depth band the narrative reads out of a raw `y`, against the Overworld's frame: bedrock sits
# near -64, sea level near 63, and diamonds are below 16. The surface floor is set
# a little under sea level so a bot a few blocks into a hillside still reads as "at the surface";
# the deep floor is the diamond line, the one depth change a step actually keys on. The three
# bands are coarse on purpose - the head needs where it is, not the exact altitude.
SURFACE_Y = 55
DEEP_Y = 16
# The Nether is not on that frame. It is a closed shell - bedrock floor at 0, bedrock ceiling at
# ~127 - with no sea level and no diamond line, and its terrain floor sits around y=45 in every
# world the bot has walked (of 7,666 Nether ticks in runs 99-112, 5,701 stand in the 40s and 50s).
# Read against the Overworld bands, the open Nether floor is "underground": run112d said
# "Underground (in the air)" at y=47.92 while the bot stood on the Nether floor. Same three words,
# keyed to the shell's own floor and roof.
NETHER_SURFACE_Y = 32.0
NETHER_DEEP_Y = 8.0

# Coarse vertical relation to a target (e.g. the home base): `level` within this many blocks, and
# `far above`/`far below` once the climb is more than a short shaft. Direction, not degrees.
VERTICAL_LEVEL = 4
VERTICAL_FAR = 20


def _depth_band(y: float, dimension: Any = None) -> str:
    """Where a `y` sits in the frame of the world it is in, in words a step can act on."""
    if _in_nether_dimension(dimension):
        surface, deep = NETHER_SURFACE_Y, NETHER_DEEP_Y
    else:
        surface, deep = SURFACE_Y, DEEP_Y
    if y >= surface:
        return "at the surface"
    if y >= deep:
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
