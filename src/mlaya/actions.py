from __future__ import annotations

import math
from dataclasses import dataclass

from .perception import GameState

# Labels Laya may choose among; each maps to a real mine-ai MCP tool.
CRITERIA: dict[str, str] = {
    "eat_food": "eat a carried food item to restore hunger",
    "get_wood": "chop a tree and collect logs; the bot has no wood yet, so this comes first",
    "craft_pickaxe": "craft a wooden pickaxe from the logs the bot carries; this is the next step",
    "craft_table": "craft a crafting table from logs, required before any 3x3 recipe",
    "mine_stone": "mine stone for cobblestone using the pickaxe the bot carries",
    "craft_stone_pickaxe": "craft a stone pickaxe at a crafting table from cobblestone and sticks",
    "mine_iron": "mine iron ore for raw iron using the stone pickaxe the bot carries",
    "mine_coal": "mine coal ore to use as furnace fuel",
    "craft_furnace": "craft a furnace at a crafting table from eight cobblestone",
    "smelt_iron": "smelt raw iron into iron ingots in the furnace the bot carries",
    "craft_iron_pickaxe": "craft an iron pickaxe at a crafting table from three iron ingots and sticks",
    "mine_diamond": "mine diamond ore with the iron pickaxe to gather diamonds for diamond tools",
    "craft_diamond_pickaxe": "craft a diamond pickaxe from three diamonds and sticks; the best pickaxe the bot can make",
    "craft_diamond_sword": "craft a diamond sword from two diamonds and a stick, once the diamond pickaxe exists",
    "sleep": "sleep in a carried bed at night to skip to morning and set the respawn point",
    "craft_bed": "craft a bed from white wool (or string) and planks at a crafting table",
    "hunt_wool": "kill a nearby sheep to collect white wool for a bed",
    "hunt_string": "kill nearby spiders to collect string, which crafts into wool",
    "hunt_food": "hunt an animal within walking range (far only in daylight), or a lone zombie, to bank at least 16 edible items",
    "cook_food": "cook carried raw meat in the furnace so it feeds more hunger",
    "replace_tool": "craft a fresh pickaxe to replace the one that is nearly worn out",
    "collect_drops": "pick up items a mob dropped nearby when the survival reflex killed it",
    "recover_drops": "walk back to the last death site and recover the gear dropped there",
    "explore": "walk to new terrain to find missing resources; the last fallback",
    "make_room": "dig a small pocket out of the surrounding blocks so a carried workstation fits",
}

EDIBLE = {
    "bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_chicken",
    "cooked_mutton", "cooked_rabbit", "cooked_cod", "cooked_salmon", "carrot",
    "potato", "baked_potato", "beetroot", "melon_slice", "sweet_berries",
    "rotten_flesh", "dried_kelp", "cookie", "mushroom_stew", "beef", "porkchop",
    "chicken", "mutton", "rabbit", "cod", "salmon",
}

# Pickaxe tier by registry prefix (tools[] reports e.g. {"class": "pickaxe", "tier": "wooden"}).
TIER = {"none": 0, "wooden": 1, "golden": 1, "stone": 2, "iron": 3, "diamond": 4, "netherite": 5}
STONE_PICK_TIER = 2
IRON_PICK_TIER = 3
DIAMOND_PICK_TIER = 4

# Passive food animals and the raw drop `collect_mob_drop` should ask for.
FOOD_MOBS = {"cow": "beef", "pig": "porkchop", "chicken": "chicken",
             "sheep": "mutton", "rabbit": "rabbit"}
# Raw -> cooked; cooking needs a carried furnace and fuel.
COOKED = {"beef": "cooked_beef", "porkchop": "cooked_porkchop",
          "chicken": "cooked_chicken", "mutton": "cooked_mutton",
          "rabbit": "cooked_rabbit", "cod": "cooked_cod", "salmon": "cooked_salmon"}
# Minimum edible items to keep banked. 16 is the Nether floor (24 the real target):
# hunger drains fast while sprinting and fighting, and death drops everything.
FOOD_RESERVE = 16
# Food hunting is the one gather that walks to a moving target, so it is bounded by
# daylight: a long walk to a distant animal is cheap in daylight, but after dark the bot
# only tops up from animals already at hand. The old unguarded hunt roamed ~180 m at
# night and was killed twice, losing a fresh iron pickaxe.
FOOD_HUNT_RANGE_DAY = 64.0
FOOD_HUNT_RANGE_NIGHT = 16.0
# Quarries that fight back, so `collect_mob_drop` must run in the hunt stage.
HOSTILE_HUNTS = {"spider", "zombie", "skeleton", "creeper", "enderman",
                 "witch", "blaze", "piglin", "zombified_piglin"}

# M2d reactions. An active tool at or below this fraction of its maximum is replaced
# before it breaks; a death this recent, this close, is worth walking back to.
LOW_DURABILITY = 0.15
# The reaction only runs between ticks, but one long `collect_block` (a tunnel to a distant
# vein) can spend far more than 15% inside a single action - live, that broke an iron pickaxe
# mid-mine and dropped the bot to stone, forcing a full iron rebuild. So the active pickaxe is
# additionally backed up once its *absolute* headroom falls to this many uses: the fresh
# same-tier spare is then already carried, and the server swaps it in the instant the active
# one breaks, so the tier (and with it `mine_diamond`) never goes away.
MINING_HEADROOM = 80
RECOVER_WINDOW_S = 90.0
RECOVER_RANGE = 120.0


@dataclass
class Action:
    label: str
    tool: str
    args: dict
    rationale: str


def is_hostile_hunt(action: "Action") -> bool:
    """True when a hunt fights back, so it must run in the aggressive hunt stage."""
    return action.tool == "collect_mob_drop" and action.args.get("mob_name") in HOSTILE_HUNTS


def _logs(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n == "logs" or n.endswith("_log"))


def _planks(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n.endswith("_planks"))


def _first_edible(inv: dict[str, int]) -> str | None:
    return next((n for n in inv if n in EDIBLE), None)


def _edible_count(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n in EDIBLE)


def _pick_tier(inv: dict[str, int], tools: dict[str, str]) -> int:
    tier = tools.get("pickaxe")
    if tier:
        value = TIER.get(str(tier).lower())
        if value is not None:
            return value
    best = 0
    for n in inv:
        if n.endswith("_pickaxe"):
            best = max(best, TIER.get(n[: -len("_pickaxe")], 0))
    return best


def _has(inv: dict[str, int], name: str) -> bool:
    return inv.get(name, 0) > 0


def _can_sticks(inv: dict[str, int], logs: int, planks: int) -> bool:
    return inv.get("stick", 0) >= 2 or planks >= 2 or logs >= 1


def _nearest_food_mob(gs: GameState, max_range: float | None = None) -> str | None:
    best: str | None = None
    best_d: float | None = None
    for mob in FOOD_MOBS:
        d = gs.mob_nearest(mob)
        if d is None or (max_range is not None and d > max_range):
            continue
        if best_d is None or d < best_d:
            best, best_d = mob, d
    return best


def _food_hunt_range(gs: GameState) -> float:
    """How far a food hunt may walk: far in daylight, only at hand after dark."""
    return FOOD_HUNT_RANGE_NIGHT if gs.phase == "night" else FOOD_HUNT_RANGE_DAY


def _first_cookable(inv: dict[str, int]) -> str | None:
    return next((n for n in inv if n in COOKED and inv[n] > 0), None)


def _wool(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n.endswith("_wool"))


def _has_bed(inv: dict[str, int]) -> bool:
    return any(n.endswith("_bed") for n in inv)


def _safe_spider_hunt(gs: GameState) -> bool:
    """Only farm spider string when the fight is a lone, close, healthy one.

    Hunting hostiles is the single most expensive mistake this bot has made (a
    spider run walked it into a zombie/skeleton pack and cost an iron pickaxe), so
    string farming requires: one spider within 10m, no other hostile within 16m,
    and health at least 16.
    """
    spider = gs.mob_nearest("spider")
    if spider is None or spider > 10 or (gs.health or 0) < 16:
        return False
    others = [d for n, k, _, d in gs.mobs
              if k == "hostile" and n != "spider" and d is not None and d <= 16]
    return not others


def _safe_zombie_hunt(gs: GameState) -> bool:
    """Zombies are the skill's overworld food bank, but only in ones and twos.

    Rotten flesh is the most reliable local food, so fall back to it only when the
    target is close, health is comfortable, and no other hostile species is in the
    neighbourhood — a pack is how a hunt turns into a death.
    """
    nearest = gs.mob_nearest("zombie")
    if nearest is None or nearest > 24 or (gs.health or 0) < 18:
        return False
    hostiles = [(n, d) for n, k, _, d in gs.mobs
                if k == "hostile" and d is not None and d <= 16]
    return len(hostiles) <= 2 and all(n == "zombie" for n, _ in hostiles)


def _food_target(gs: GameState) -> tuple[str, str] | None:
    """The safest available food source: a passive animal in walking reach, else a lone zombie."""
    mob = _nearest_food_mob(gs, _food_hunt_range(gs))
    if mob is not None:
        return mob, FOOD_MOBS[mob]
    if _safe_zombie_hunt(gs):
        return "zombie", "rotten_flesh"
    return None


def _tool_durability(gs: GameState, cls: str) -> tuple[float, str] | None:
    """(fraction remaining, item name) for the active tool of a class, if known."""
    meta = gs.tool_meta.get(cls) or {}
    left, mx, item = meta.get("left"), meta.get("max"), meta.get("item")
    if item and isinstance(left, (int, float)) and isinstance(mx, (int, float)) and mx > 0:
        return (left / mx, str(item))
    return None


def _pickaxe_left(gs: GameState) -> float | None:
    """Absolute uses left on the active pickaxe, when the server reports it."""
    left = (gs.tool_meta.get("pickaxe") or {}).get("left")
    return float(left) if isinstance(left, (int, float)) else None


def _pickaxe_due_for_spare(gs: GameState) -> bool:
    """True when the active pickaxe should already have a fresh same-tier backup carried.

    Two triggers: the classic "about to break" fraction, and an absolute headroom reserve so
    that a single long dig cannot exhaust the tool before the next tick can react.
    """
    d = _tool_durability(gs, "pickaxe")
    if d is None:
        return False
    if d[0] <= LOW_DURABILITY:
        return True
    left = _pickaxe_left(gs)
    return left is not None and left <= MINING_HEADROOM


def _has_spare_pickaxe(gs: GameState, inv: dict[str, int], item: str) -> bool:
    """A same-tier backup is already carried (the active tool counts as the first copy)."""
    active = (gs.tool_meta.get("pickaxe") or {}).get("item")
    return inv.get(item, 0) >= (2 if item == active else 1)


def _replacement_pickaxe(gs: GameState, inv: dict[str, int]) -> str | None:
    """The best pickaxe still craftable when the active one needs a backup.

    Never a downgrade below what the bot already wields: diamond if diamonds are banked,
    else iron, else stone (which still mines iron ore). A wooden one is pointless here
    because `craft_pickaxe` only fires when the bot has no pickaxe at all. Returns None once
    a spare of that kind is already carried, so the reserve cannot craft the same spare twice
    (the server equips a same-tier fresh tool, but a lower-tier spare leaves the worn active
    one in hand - without this guard that would loop until the materials ran out).
    """
    tier = _pick_tier(inv, gs.tools)
    if tier < STONE_PICK_TIER:
        return None
    sticks = _can_sticks(inv, _logs(inv), _planks(inv))
    table = _has(inv, "crafting_table")
    item = None
    if tier >= DIAMOND_PICK_TIER and table and inv.get("diamond", 0) >= 3 and sticks:
        item = "diamond_pickaxe"
    elif tier >= IRON_PICK_TIER and table and inv.get("iron_ingot", 0) >= 3 and sticks:
        item = "iron_pickaxe"
    elif table and inv.get("cobblestone", 0) >= 3 and sticks:
        item = "stone_pickaxe"
    if item is None or _has_spare_pickaxe(gs, inv, item):
        return None
    return item


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _can_recover(gs: GameState) -> bool:
    """A fresh death close by: the dropped gear is still loaded and worth a walk back."""
    if gs.death_pos is None or gs.last_death_age_s is None or gs.pos is None:
        return False
    if gs.last_death_age_s > RECOVER_WINDOW_S:
        return False
    if gs.death_dim and gs.dimension and gs.death_dim != gs.dimension:
        return False
    return _dist(gs.pos, gs.death_pos) <= RECOVER_RANGE


def candidates(gs: GameState) -> list[str]:
    """The actions whose preconditions currently hold, in tech-tree priority order.

    Laya only ever chooses among these, so no choice can be a dead end.
    """
    inv = gs.inventory
    out: list[str] = []
    tier = _pick_tier(inv, gs.tools)
    logs = _logs(inv)
    planks = _planks(inv)
    cobble = inv.get("cobblestone", 0)
    table = _has(inv, "crafting_table")
    furnace = _has(inv, "furnace")
    coal = inv.get("coal", 0)
    raw_iron = inv.get("raw_iron", 0)
    iron = inv.get("iron_ingot", 0)
    diamond = inv.get("diamond", 0)

    eating = gs.food is not None and gs.food < 14 and bool(_first_edible(inv))

    # --- event-driven reactions (M2d) ---
    # A tool about to break, a mob the reflex just killed (its drops despawn in minutes),
    # or gear lying at a fresh death site are all time-sensitive, so the top reaction
    # owns the tick. Eating still rides along: survival outranks everything.
    reactions: list[str] = []
    if _can_recover(gs):
        reactions.append("recover_drops")
    if _pickaxe_due_for_spare(gs) and _replacement_pickaxe(gs, inv):
        reactions.append("replace_tool")
    if gs.drops > 0:
        reactions.append("collect_drops")
    if reactions:
        return (["eat_food"] if eating else []) + reactions[:1]

    if eating:
        out.append("eat_food")
    if _has(inv, "furnace") and inv.get("coal", 0) >= 1 and _first_cookable(inv):
        out.append("cook_food")
    # Food reserve: bank FOOD_RESERVE edible items. At full hunger the reflex cannot
    # eat, so gathered drops accumulate; when hunger drops it spends them. Never
    # compete with eating: feed to 20 first, bank after.
    if (_edible_count(inv) < FOOD_RESERVE and not eating
            and _food_target(gs) is not None):
        out.append("hunt_food")

    wool = _wool(inv)
    string = inv.get("string", 0)
    bed = _has_bed(inv)
    # Night + a carried bed: sleeping skips to morning and pins the respawn point.
    if gs.phase == "night" and not gs.sleeping and bed:
        out.append("sleep")

    # Bootstrap, then crafts/smelts (in tech-tree order), then gathering.
    need_wood = (tier == 0 or not table
                 or (not bed and (wool >= 3 or string >= 12))
                 or (_pickaxe_due_for_spare(gs) and not _can_sticks(inv, logs, planks)))
    if logs == 0 and planks == 0 and need_wood:
        out.append("get_wood")
    if (logs >= 1 or planks >= 5) and tier == 0:
        out.append("craft_pickaxe")
    if (logs >= 1 or planks >= 4) and not table:
        out.append("craft_table")
    if tier >= 1 and tier < STONE_PICK_TIER and table and cobble >= 3 and _can_sticks(inv, logs, planks):
        out.append("craft_stone_pickaxe")
    if tier >= 1 and table and not furnace and cobble >= 8:
        out.append("craft_furnace")
    if furnace and raw_iron >= 1 and coal >= 1 and iron < 3:
        out.append("smelt_iron")
    if table and iron >= 3 and tier < IRON_PICK_TIER and _can_sticks(inv, logs, planks):
        out.append("craft_iron_pickaxe")
    if table and diamond >= 3 and tier < DIAMOND_PICK_TIER and _can_sticks(inv, logs, planks):
        out.append("craft_diamond_pickaxe")
    # The sword waits for the pickaxe so the first three diamonds always go to the tool
    # that unlocks obsidian; it needs only two more.
    if (table and tier >= DIAMOND_PICK_TIER and diamond >= 2 and not _has(inv, "diamond_sword")
            and _can_sticks(inv, logs, planks)):
        out.append("craft_diamond_sword")
    # Bed chain: a bed lets `sleep` skip the night and set the respawn point.
    if not bed:
        if table and (wool >= 3 or string >= 12) and (logs >= 1 or planks >= 3):
            out.append("craft_bed")
        elif wool < 3 and string < 12:
            if gs.mob_nearest("sheep") is not None:
                out.append("hunt_wool")
            elif _safe_spider_hunt(gs):
                out.append("hunt_string")
    if tier >= STONE_PICK_TIER and tier < IRON_PICK_TIER and raw_iron + iron < 3 and gs.resource_distance("iron_ore") is not None:
        out.append("mine_iron")
    if tier >= STONE_PICK_TIER and tier < IRON_PICK_TIER and raw_iron >= 1 and coal < 1 and gs.resource_distance("coal_ore") is not None:
        out.append("mine_coal")
    # Diamonds are wanted for the pickaxe (three) and then for the sword (two more). There
    # is deliberately no upper tier bound here: bounding it at the diamond tier locked the
    # sword away, because a diamond pickaxe could never mine the two diamonds it still
    # needed (and `craft_diamond_sword` requires the pickaxe to already exist).
    wants_diamonds = (diamond < 3 if tier < DIAMOND_PICK_TIER
                      else diamond < 2 and not _has(inv, "diamond_sword"))
    if tier >= IRON_PICK_TIER and wants_diamonds and gs.resource_distance("diamond_ore") is not None:
        out.append("mine_diamond")
    if tier >= 1 and cobble < 12:
        out.append("mine_stone")

    # explore is the explicit last fallback: only when nothing productive is available.
    if not out:
        out.append("explore")
    return out


def resolve(label: str | None, gs: GameState, target=None) -> Action | None:
    """Map a Laya choice to a concrete MCP tool call, or None if preconditions fail.

    `target` is the M4 parameter-layer pick (a `targets.Target`); it is only used by a
    label whose step needs a destination (`explore`), and ignored elsewhere.
    """
    if not label:
        return None
    inv = gs.inventory
    tier = _pick_tier(inv, gs.tools)
    x, y, z = gs.pos or (None, None, None)

    if label == "get_wood":
        return Action(label, "collect_block", {"block_name": "logs", "count": 8},
                      "gather logs to start the tech tree")
    if label == "mine_stone":
        return Action(label, "collect_block", {"block_name": "stone", "count": 8},
                      "mine cobblestone to unlock stone tools")
    if label == "mine_iron":
        if tier < STONE_PICK_TIER:
            return None
        count = max(1, 3 - inv.get("raw_iron", 0))
        return Action(label, "collect_block", {"block_name": "iron_ore", "count": count},
                      "mine iron ore for iron tools")
    if label == "mine_coal":
        if tier < 1:
            return None
        return Action(label, "collect_block", {"block_name": "coal_ore", "count": 3},
                      "mine coal for furnace fuel")
    if label == "mine_diamond":
        if tier < IRON_PICK_TIER:
            return None
        need = 3 if tier < DIAMOND_PICK_TIER else 2  # the pickaxe first, then the sword
        count = max(1, need - inv.get("diamond", 0))
        return Action(label, "collect_block", {"block_name": "diamond_ore", "count": count},
                      "mine diamond ore for diamond tools")
    if label == "craft_pickaxe":
        if (_logs(inv) < 1 and _planks(inv) < 5) or tier != 0:
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "wooden_pickaxe"}]},
                      "craft a wooden pickaxe from carried logs")
    if label == "craft_table":
        if (_logs(inv) < 1 and _planks(inv) < 4) or _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "crafting_table"}]},
                      "craft a crafting table for 3x3 recipes")
    if label == "craft_stone_pickaxe":
        if inv.get("cobblestone", 0) < 3 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "stone_pickaxe"}], "temporary_workstation": True},
                      "craft a stone pickaxe at the carried crafting table")
    if label == "craft_furnace":
        if inv.get("cobblestone", 0) < 8 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "furnace"}], "temporary_workstation": True},
                      "craft a furnace to smelt iron")
    if label == "smelt_iron":
        if not _has(inv, "furnace") or inv.get("raw_iron", 0) < 1 or inv.get("coal", 0) < 1:
            return None
        count = min(inv.get("raw_iron", 0), 3)
        return Action(label, "smelt_item",
                      {"item_name": "raw_iron", "count": count, "fuel_item_name": "coal",
                       "temporary_workstation": True},
                      "smelt raw iron into ingots")
    if label == "craft_iron_pickaxe":
        if inv.get("iron_ingot", 0) < 3 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "iron_pickaxe"}], "temporary_workstation": True},
                      "craft an iron pickaxe for mining diamonds")
    if label == "craft_diamond_pickaxe":
        if (inv.get("diamond", 0) < 3 or not _has(inv, "crafting_table")
                or tier >= DIAMOND_PICK_TIER or not _can_sticks(inv, _logs(inv), _planks(inv))):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "diamond_pickaxe"}], "temporary_workstation": True},
                      "craft a diamond pickaxe from three diamonds and sticks")
    if label == "craft_diamond_sword":
        if (inv.get("diamond", 0) < 2 or not _has(inv, "crafting_table")
                or tier < DIAMOND_PICK_TIER or _has(inv, "diamond_sword")
                or not _can_sticks(inv, _logs(inv), _planks(inv))):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "diamond_sword"}], "temporary_workstation": True},
                      "craft a diamond sword from two diamonds and a stick")
    if label == "sleep":
        if gs.phase != "night" or gs.sleeping or not _has_bed(inv):
            return None
        return Action(label, "sleep", {},
                      "sleep through the night and set the respawn point")
    if label == "craft_bed":
        if not _has(inv, "crafting_table") or (_wool(inv) < 3 and inv.get("string", 0) < 12) \
                or (_logs(inv) < 1 and _planks(inv) < 3):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "white_bed"}], "temporary_workstation": True},
                      "craft a bed from wool or string and planks")
    if label == "hunt_wool":
        if _wool(inv) >= 3:
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "sheep", "drop_name": "white_wool", "count": 3},
                      "collect white wool from a sheep for a bed")
    if label == "hunt_string":
        if inv.get("string", 0) >= 12 or not _safe_spider_hunt(gs):
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "spider", "drop_name": "string", "count": 12,
                       "allow_without_shield": True},
                      "collect string from a lone spider to craft wool")
    if label == "hunt_food":
        target = _food_target(gs)
        if not target:
            return None
        mob, drop = target
        count = 6 if mob in FOOD_MOBS else 10
        return Action(label, "collect_mob_drop",
                      {"mob_name": mob, "drop_name": drop, "count": count,
                       "allow_without_shield": True},
                      f"hunt a {mob} for {drop}")
    if label == "cook_food":
        raw = _first_cookable(inv)
        if not raw or not _has(inv, "furnace") or inv.get("coal", 0) < 1:
            return None
        return Action(label, "smelt_item",
                      {"item_name": raw, "count": min(inv.get(raw, 0), 3),
                       "fuel_item_name": "coal", "temporary_workstation": True},
                      f"cook {raw} into {COOKED[raw]}")
    if label == "eat_food":
        food = _first_edible(inv)
        if not food:
            return None
        return Action(label, "eat_food", {"food_name": food},
                      f"eat {food} to restore hunger")
    if label == "replace_tool":
        item = _replacement_pickaxe(gs, inv)
        if not item or not _pickaxe_due_for_spare(gs):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": item}], "temporary_workstation": True},
                      f"carry a fresh {item} before the active one breaks mid-dig")
    if label == "collect_drops":
        if gs.drops <= 0:
            return None
        return Action(label, "pick_up_items", {"radius": 16},
                      "collect items a mob dropped when the reflex killed it")
    if label == "recover_drops":
        if not _can_recover(gs):
            return None
        return Action(label, "pick_up_items", {"recover_death_items": True},
                      "walk back to the death site and recover the dropped gear")
    if label == "make_room":
        if gs.pos is None:
            return None
        # `build_structure` with `block_name: "air"` digs cells clear. One small pocket at
        # the bot's own level is enough: each cleared side cell has the tunnel floor under
        # it, so the placement search finds a supported, replaceable cell within reach.
        # Kept to four cells on purpose -- the tool cost grows with the cell list.
        mx, my, mz = (math.floor(v) for v in gs.pos)
        cells = [{"x": mx + dx, "y": my, "z": mz + dz, "block_name": "air"}
                 for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        return Action(label, "build_structure", {"blocks": cells},
                      "dig a small pocket around the bot for the carried workstation")
    if label == "explore":
        if target is not None:
            tx, ty, tz = target.pos
            return Action(label, "navigate",
                          {"x": tx, "y": ty, "z": tz, "dig": True, "scaffold": True},
                          f"explore toward {target.name} at {tx},{ty},{tz}")
        if x is None or z is None:
            return None
        args: dict = {"x": int(x) + 16, "z": int(z), "dig": True, "scaffold": True}
        if y is not None:
            args["y"] = int(y)
        return Action(label, "navigate", args, "explore toward new terrain")
    return None
