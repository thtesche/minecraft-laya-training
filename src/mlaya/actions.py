from __future__ import annotations

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
    "sleep": "sleep in a carried bed at night to skip to morning and set the respawn point",
    "craft_bed": "craft a bed from white wool (or string) and planks at a crafting table",
    "hunt_wool": "kill a nearby sheep to collect white wool for a bed",
    "hunt_string": "kill nearby spiders to collect string, which crafts into wool",
    "explore": "walk to new terrain to find missing resources; the last fallback",
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


@dataclass
class Action:
    label: str
    tool: str
    args: dict
    rationale: str


def _logs(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n == "logs" or n.endswith("_log"))


def _planks(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n.endswith("_planks"))


def _first_edible(inv: dict[str, int]) -> str | None:
    return next((n for n in inv if n in EDIBLE), None)


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

    if gs.food is not None and gs.food < 14 and _first_edible(inv):
        out.append("eat_food")

    wool = _wool(inv)
    string = inv.get("string", 0)
    bed = _has_bed(inv)
    # Night + a carried bed: sleeping skips to morning and pins the respawn point.
    if gs.phase == "night" and not gs.sleeping and bed:
        out.append("sleep")

    # Bootstrap, then crafts/smelts (in tech-tree order), then gathering.
    need_wood = (tier == 0 or not table
                 or (not bed and (wool >= 3 or string >= 12)))
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
    if tier >= 1 and cobble < 12:
        out.append("mine_stone")

    # explore is the explicit last fallback: only when nothing productive is available.
    if not out:
        out.append("explore")
    return out


def resolve(label: str | None, gs: GameState) -> Action | None:
    """Map a Laya choice to a concrete MCP tool call, or None if preconditions fail."""
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
    if label == "eat_food":
        food = _first_edible(inv)
        if not food:
            return None
        return Action(label, "eat_food", {"food_name": food},
                      f"eat {food} to restore hunger")
    if label == "explore":
        if x is None or z is None:
            return None
        args: dict = {"x": int(x) + 16, "z": int(z), "dig": True, "scaffold": True}
        if y is not None:
            args["y"] = int(y)
        return Action(label, "navigate", args, "explore toward new terrain")
    return None
