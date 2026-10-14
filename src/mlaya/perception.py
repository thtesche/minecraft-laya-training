from __future__ import annotations

import json
import math
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
# The blocks the Nether / home-base chain has to find precisely rather than read off a count.
# `lava` is where the carried water bucket is poured: obsidian is *made*, not dug (tunnelling to
# the natural y=-55 deposit is the live trap that left the bot swimming in its own flooded
# tunnel). `chest` lets a run that did not place the chest itself re-adopt one already standing
# at the base. `furnace` lets a run re-adopt the furnace it placed at the base (so cooking
# reuses it instead of placing and tearing a temporary one down on every call). `gold_ore` is
# the Nether-prep dig (the gold helmet that keeps piglins neutral): like the diamonds, a deposit
# deep under the surface counts through `resource_found` so the `collect_block` tunnel is still
# offered. Kept as its own `view_blocks find` group: that call takes at most eight names.
CHAIN_NAMES = ("lava", "chest", "furnace", "gold_ore")
# Groups whose nearest hit is worth naming in the narrative (Laya reads that text).
NAMED_RESOURCES = RESOURCE_NAMES + ("lava",)
RESOURCE_RADIUS = 64.0
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

    def as_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y, "z": self.z,
                "distance": self.distance, "above": self.above, "below": self.below,
                "level": self.level}


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
    death_pos: tuple[float, float, float] | None = None
    death_dim: str | None = None
    activity_owner: str | None = None
    active_action: str | None = None
    events: list[str] = field(default_factory=list)
    unread: int = 0
    survivors: list[str] = field(default_factory=list)
    # name -> [BlockHit] nearest first, already radius-filtered
    resources: dict[str, list[BlockHit]] = field(default_factory=dict)
    sleeping: bool | None = None
    minutes_to_change: float | None = None
    # (name, kind, count, nearest distance) for every nearby mob species
    mobs: list[tuple[str, str, int, float | None]] = field(default_factory=list)
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
    # True while the base was re-adopted from the world (a lit portal a previous run left) and the
    # bot has not reached it yet. It is what justifies walking home once (`go_home`) so a restart
    # finishes at the portal it built instead of drifting off to explore.
    base_adopted: bool = False
    # Lava cells whose pour the server already refused (`[BUCKET_NO_FACE]`): no solid block
    # beside them faces the bot, so retrying there can only fail again. The loop records them
    # and the obsidian step passes them over, which is what moves the search from a stopped-up
    # surface pond to the next lava pool instead of looping on the nearest cell forever.
    pour_skip: frozenset = frozenset()
    # A `drop_junk` the terrain refused to give a disposal pit for (`[DROP_HOLE_UNAVAILABLE]`):
    # from then on the drop goes onto the open ground instead of into a dug hole. Both free the
    # slots the relief is for, and the open drop cannot fail, so the loop cannot stall on it.
    drop_hole_blocked: bool = False
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
    # The cell a placement was refused at because a block stood in it (`[PLACE_TARGET_OCCUPIED]`):
    # `make_room` digs exactly that cell, since the block in it is the whole cause and a pocket
    # around the bot need not touch it (the chest cell sits beside the base, not the bot). Live,
    # from-zero run: the cell held `grass_block` and `place_chest` answered the code every tick.
    blocked_cell: tuple[int, int, int] | None = None
    # Whether a block kind exists anywhere in the loaded world, even beyond RESOURCE_RADIUS.
    # `view_blocks find` reports a per-name `found` count across every loaded chunk, so a deep
    # deposit the 64 m scan cannot list still proves the ore is there to dig for. This is what
    # gates `mine_diamond`: standing on the surface, the nearest diamond ore (y<16) is always
    # out of scan range, so without the raw count the loop could never dig down to the diamond
    # armour piece it still needs once one is lost.
    resource_found: dict[str, int] = field(default_factory=dict)
    # A level, clear surface cell to raise the home base on, proposed by a one-off
    # `view_blocks box` scan while `base` is still unset. The loop owns the base, so this is
    # only the *site proposal*; `None` means no flat site was found and the base pick falls
    # back to the nearest standable ground. A frame raised on a slope leaves its cells holding
    # stone, which the build server will not dig (live: 15 of 20 frame cells stayed stone).
    build_site: tuple[int, int, int] | None = None
    # Cells of the portal opening (`portal_interior`) that hold a stray block, read from one
    # `view_blocks cells` scan while a frame build is pending. The opening sits *above* the
    # support layer and must end up air: a block left there is built around and `activate_portal`
    # answers `[PORTAL_NO_FRAME]` on every tick (live, Nether run: a bed the generic sleep laid
    # "beside the bot" landed in the opening). `clear_portal` digs these cells before the build.
    portal_blocked: list[tuple[int, int, int]] = field(default_factory=list)

    def resource_nearest(self, name: str) -> BlockHit | None:
        hits = self.resources.get(name)
        return hits[0] if hits else None

    def resource_distance(self, name: str) -> float | None:
        hit = self.resource_nearest(name)
        return hit.distance if hit else None

    def mob_nearest(self, name: str) -> float | None:
        ds = [d for n, _, _, d in self.mobs if n == name and d is not None]
        return min(ds) if ds else None

    def as_state(self) -> dict:
        """Structured view for logging and System-2 prompts (never sent raw to Laya)."""
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
            "events": self.events,
            "unread": self.unread,
            "notifications": self.notifications,
            "sleeping": self.sleeping,
            "resources": {n: [h.as_dict() for h in hits[:2]] for n, hits in self.resources.items() if hits},
            "resource_found": self.resource_found,
            "mobs": self.mobs,
        }

    def narrative(self) -> str:
        """Compact natural-language state. Must stay small: Laya truncates long states."""
        parts: list[str] = []
        if self.dimension:
            parts.append(f"in the {self.dimension}")
        if self.health is not None:
            parts.append("health is full" if self.health >= 20 else f"health is {self.health:g}/20")
        if self.food is not None:
            parts.append("hunger is full" if self.food >= 20 else f"hunger is {self.food:g}/20")
        if self.phase:
            weather = " and raining" if self.raining else ""
            label = "daytime" if self.phase == "day" else self.phase
            asleep = " and the bot is asleep" if self.sleeping else ""
            parts.append(f"it is {label}{weather}{asleep}")
        if self.pos:
            x, y, z = self.pos
            where = "on the ground" if self.on_ground else "in the air"
            if self.in_water:
                where = "in water"
            parts.append(f"at x={x:.0f}, y={y:.0f}, z={z:.0f} ({where})")
        if self.base:
            bx, by, bz = self.base
            here = ", at the base" if (self.pos and abs(self.pos[0] - (bx + 0.5)) <= 4
                                       and abs(self.pos[2] - (bz + 0.5)) <= 4) else ""
            parts.append(f"home base at x={bx}, y={by}, z={bz}{here}")
        if self.chest_pos:
            parts.append("a chest stands at the base")
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
                f"{n} x{c}" + (f" at {d:.0f}m" if d is not None else "")
                for n, c, d in close[:3]
            )
            parts.append(f"hostile mobs within 16 blocks: {desc}")
        else:
            parts.append("no hostile mobs within 16 blocks")
        if self.players:
            parts.append(f"players nearby: {', '.join(self.players[:3])}")
        if self.last_death_cause:
            age = f" {self.last_death_age_s:.0f}s ago" if self.last_death_age_s is not None else ""
            parts.append(f"last death:{age} {self.last_death_cause}")
        if self.drops:
            parts.append(f"{self.drops} mob(s) the reflex killed dropped items nearby")
        if self.death_pos and self.last_death_age_s is not None and self.last_death_age_s <= 90:
            parts.append("gear dropped at the last death site can still be recovered")
        present = [(n, hits[0].distance) for n, hits in self.resources.items()
                   if hits and n in NAMED_RESOURCES]
        if present:
            present.sort(key=lambda t: t[1])
            parts.append("nearby resources: " + ", ".join(
                f"{n} at {d:.0f}m" for n, d in present[:4]))
        if self.unread:
            head = "; ".join(self.notifications[:2])
            parts.append(f"unread notifications ({self.unread}): {head}" if head
                         else f"unread notifications: {self.unread}")
        parts[0] = parts[0][:1].upper() + parts[0][1:]
        return "You are a Minecraft bot. " + ". ".join(parts) + "."


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

    death = sit.get("lastDeath") or {}
    if isinstance(death, dict) and death.get("cause"):
        gs.last_death_cause = str(death["cause"])
        gs.last_death_age_s = _age_seconds(sit.get("observedAt"), death.get("observedAt"))
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


def parse_resources(data: dict, limit: int = RESOURCE_LIMIT) -> dict[str, list[BlockHit]]:
    """view_blocks find -> {query_name: [BlockHit]} within RESOURCE_RADIUS, nearest first.

    `limit` caps the hits kept per query name. It has to match the `limit` the scan asked
    the server for, or the parse would throw away hits the state still needs -- the lava
    group especially, where a pour cell is only recognisable when the *source* beside it is
    in the list too (and that source is not among the nearest few cells).
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
            if d is None or d > RESOURCE_RADIUS or "x" not in e:
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


async def scan_ground(executor, pos, dy: int = 0) -> dict:
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
        "read the terrain around the bot to place the home base on level ground",
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


async def scan_frame_volume(executor, base: tuple[int, int, int]) -> dict:
    """One `view_blocks cells` read of the portal opening, for the frame-clearance step.

    The base scan (`targets.flat_build_site`) only proves clearance for a site *it* chose; a
    base the loop adopted from a standing bed, or one taken from the `base_target` fallback,
    never got that check. This read is what lets `clear_portal` dig a stray block out of the
    opening before `build_portal`. Returns the coerced result for `parse_cells`.
    """
    cells = [{"x": x, "y": y, "z": z} for x, y, z in portal_interior(base)]
    res = await executor.read(
        "view_blocks",
        {"cells": cells},
        "read the portal opening for stray blocks before the frame is built",
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


async def build(executor) -> GameState:
    """Perceive through the MCP read tools (ungated, cheap)."""
    status = await executor.read("view_status", {}, "read live bot vitals and position")
    status_data = coerce(status.data, status.text)
    # Walk the event backlog to the newest events; reactions only use fresh ones.
    events, unread, preview = await _drain_events(executor)
    gs = parse_status(status_data)
    gs.events = _summaries(events)
    gs.unread = unread
    gs.notifications = _previews(preview)
    now_iso = get(status_data, "response", "data", "result", "situation", "observedAt")
    gs.drops = _count_drops(events, now_iso)
    # `view_blocks find` takes at most 8 block names per call, and the resource group alone
    # already uses all eight, so each group is scanned separately and merged; a failed group
    # just leaves its names out of the state.
    gs.resources = {}
    for names, limit, why in ((RESOURCE_NAMES, RESOURCE_LIMIT,
                               "scan nearby resources for the next step"),
                              (GROUND_NAMES, GROUND_LIMIT,
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
            gs.resources.update(parse_resources(payload, limit))
            gs.resource_found.update(parse_resource_counts(payload))
        except Exception:
            pass
    return gs
