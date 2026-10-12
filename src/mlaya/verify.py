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

    worn_before = {m.get("item") for m in before.armor.values() if m.get("item")}
    worn_after = {m.get("item") for m in after.armor.values() if m.get("item")}

    return {
        "pos_delta": pos_delta(),
        "health_delta": None if before.health is None or after.health is None
        else round(after.health - before.health, 1),
        "food_delta": None if before.food is None or after.food is None
        else round(after.food - before.food, 1),
        "inventory_delta": inv_delta,
        "armor_delta": sorted(worn_after - worn_before),
        "on_ground": after.on_ground,
        "in_water": after.in_water,
        "phase": after.phase,
        "sleeping": after.sleeping,
    }


# Labels whose progress shows up as a net inventory gain (a drop collected or a craft produced).
INVENTORY_LABELS = {
    "get_wood", "mine_stone", "mine_iron", "mine_coal", "craft_pickaxe", "craft_table",
    "craft_stone_pickaxe", "craft_furnace", "smelt_iron", "craft_iron_pickaxe", "craft_bed",
    "mine_diamond", "craft_diamond_pickaxe", "craft_diamond_sword",
    "craft_diamond_chestplate", "craft_diamond_leggings", "craft_diamond_boots",
    "craft_diamond_helmet",
}


def progressed(label: str, d: dict) -> bool:
    inv_up = any(v > 0 for v in d["inventory_delta"].values())
    armor_up = bool(d.get("armor_delta"))
    if label in INVENTORY_LABELS:
        # A craft normally lands in the pack; a server that auto-equips armour puts the piece
        # straight on the body, so armour_craft progress also shows as armor_delta.
        return inv_up or armor_up
    if label == "equip_armor":
        return armor_up  # the piece leaves the pack for a worn slot
    if label == "hunt_string":
        return d["inventory_delta"].get("string", 0) > 0
    if label == "hunt_wool":
        return any(k.endswith("_wool") and v > 0 for k, v in d["inventory_delta"].items())
    if label == "hunt_food":
        raw = {"beef", "porkchop", "chicken", "mutton", "rabbit", "rotten_flesh"}
        return any(d["inventory_delta"].get(x, 0) > 0 for x in raw)
    if label == "cook_food":
        cooked = {"cooked_beef", "cooked_porkchop", "cooked_chicken", "cooked_mutton",
                  "cooked_rabbit", "cooked_cod", "cooked_salmon"}
        return any(d["inventory_delta"].get(x, 0) > 0 for x in cooked)
    if label == "eat_food":
        return (d["food_delta"] or 0) > 0
    if label == "replace_tool":
        return any(k.endswith("_pickaxe") and v > 0 for k, v in d["inventory_delta"].items())
    if label in ("collect_drops", "recover_drops"):
        return inv_up
    if label == "sleep":
        return d.get("phase") != "night" or bool(d.get("sleeping"))
    if label == "explore":
        return (d["pos_delta"] or 0) > 2
    return False
