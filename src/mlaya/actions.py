from __future__ import annotations

from dataclasses import dataclass

from .perception import GameState

# Labels Laya may choose among; each maps to a real mine-ai MCP tool.
CRITERIA: dict[str, str] = {
    "eat_food": "eat a carried food item to restore hunger",
    "get_wood": "chop a tree and collect logs; the bot has no wood yet, so this comes first",
    "craft_pickaxe": "craft a wooden pickaxe from the logs the bot carries; this is the next step",
    "mine_stone": "mine stone for cobblestone using the pickaxe the bot carries",
    "craft_stone_pickaxe": "craft a stone pickaxe from the cobblestone and sticks the bot carries",
    "explore": "walk to new terrain; only useful once the bot already has wood and a pickaxe",
}

EDIBLE = {
    "bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_chicken",
    "cooked_mutton", "cooked_rabbit", "cooked_cod", "cooked_salmon", "carrot",
    "potato", "baked_potato", "beetroot", "melon_slice", "sweet_berries",
    "rotten_flesh", "dried_kelp", "cookie", "mushroom_stew",
}


@dataclass
class Action:
    label: str
    tool: str
    args: dict
    rationale: str


def _logs(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n == "logs" or n.endswith("_log"))


def _first_edible(inv: dict[str, int]) -> str | None:
    return next((n for n in inv if n in EDIBLE), None)


def _has_pickaxe(inv: dict[str, int], tools: dict[str, str]) -> bool:
    if any(n.endswith("_pickaxe") for n in inv):
        return True
    tier = tools.get("pickaxe")
    return bool(tier and tier != "none")


def candidates(gs: GameState) -> list[str]:
    """The actions whose preconditions currently hold. Laya chooses among these."""
    inv = gs.inventory
    out: list[str] = []
    if gs.food is not None and gs.food < 14 and _first_edible(inv):
        out.append("eat_food")
    logs = _logs(inv)
    has_pick = _has_pickaxe(inv, gs.tools)
    has_stone_pick = any(n in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe",
                               "netherite_pickaxe") for n in inv)
    if logs == 0:
        out.append("get_wood")
    if logs >= 1 and not has_pick:
        out.append("craft_pickaxe")
    if has_pick and not has_stone_pick and inv.get("cobblestone", 0) >= 3 \
            and inv.get("stick", 0) >= 2:
        out.append("craft_stone_pickaxe")
    if has_pick:
        out.append("mine_stone")
    out.append("explore")
    return out


def resolve(label: str | None, gs: GameState) -> Action | None:
    """Map a Laya choice to a concrete MCP tool call, or None if preconditions fail."""
    if not label:
        return None
    inv = gs.inventory
    x, y, z = gs.pos or (None, None, None)

    if label == "get_wood":
        return Action(label, "collect_block", {"block_name": "logs", "count": 5},
                      "gather logs to start the tech tree")
    if label == "mine_stone":
        return Action(label, "collect_block", {"block_name": "stone", "count": 8},
                      "mine cobblestone to unlock stone tools")
    if label == "craft_pickaxe":
        if _logs(inv) < 1 or _has_pickaxe(inv, gs.tools):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "wooden_pickaxe"}]},
                      "craft a wooden pickaxe from carried logs")
    if label == "craft_stone_pickaxe":
        if inv.get("cobblestone", 0) < 3 or inv.get("stick", 0) < 2:
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "stone_pickaxe"}],
                       "temporary_workstation": True},
                      "craft a stone pickaxe for iron")
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
