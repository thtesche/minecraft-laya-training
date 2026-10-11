from __future__ import annotations

from .perception import GameState


def diff(before: GameState, after: GameState) -> dict:
    def pos_delta() -> float | None:
        if before.pos and after.pos:
            return round(sum((a - b) ** 2 for a, b in zip(after.pos, before.pos)) ** 0.5, 2)
        return None

    inv_delta: dict[str, int] = {}
    for k in set(before.inventory) | set(after.inventory):
        d = after.inventory.get(k, 0) - before.inventory.get(k, 0)
        if d:
            inv_delta[k] = d

    return {
        "pos_delta": pos_delta(),
        "health_delta": None if before.health is None or after.health is None
        else round(after.health - before.health, 1),
        "food_delta": None if before.food is None or after.food is None
        else round(after.food - before.food, 1),
        "inventory_delta": inv_delta,
        "on_ground": after.on_ground,
        "in_water": after.in_water,
    }


def progressed(label: str, d: dict) -> bool:
    inv_up = any(v > 0 for v in d["inventory_delta"].values())
    if label in ("get_wood", "mine_stone", "craft_pickaxe", "craft_stone_pickaxe"):
        return inv_up
    if label == "eat_food":
        return (d["food_delta"] or 0) > 0
    if label == "explore":
        return (d["pos_delta"] or 0) > 2
    return False
