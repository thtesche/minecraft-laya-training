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
    off_before = {before.offhand} if before.offhand else set()
    off_after = {after.offhand} if after.offhand else set()

    return {
        "pos_delta": pos_delta(),
        "health_delta": None if before.health is None or after.health is None
        else round(after.health - before.health, 1),
        "food_delta": None if before.food is None or after.food is None
        else round(after.food - before.food, 1),
        "inventory_delta": inv_delta,
        "armor_delta": sorted(worn_after - worn_before),
        "offhand_delta": sorted(off_after - off_before),
        "on_ground": after.on_ground,
        "in_water": after.in_water,
        "phase": after.phase,
        "sleeping": after.sleeping,
        # World-block milestones the inventory cannot show: lighting touches no stack, and a
        # placed bed leaves the pack but exists as a block. The portal frame itself is judged
        # by the obsidian leaving the pack (below), because natural obsidian may already be
        # loaded and would mask the frame in a resource-presence check.
        "portal_lit": after.resource_distance("nether_portal") is not None,
        "bed_placed": after.resource_distance("white_bed") is not None,
    }


# Labels whose progress shows up as a net inventory gain (a drop collected or a craft produced).
INVENTORY_LABELS = {
    "get_wood", "mine_stone", "mine_iron", "mine_coal", "craft_pickaxe", "craft_table",
    "craft_stone_pickaxe", "craft_furnace", "smelt_iron", "craft_iron_pickaxe", "craft_bed",
    "mine_diamond", "craft_diamond_pickaxe", "craft_diamond_sword",
    "craft_diamond_chestplate", "craft_diamond_leggings", "craft_diamond_boots",
    "craft_diamond_helmet", "craft_shield",
    # Gold and the Nether prep. Each nets an item the same way (raw_gold, gold_ingot, a bow, an
    # arrow, a worn gold helmet, a feather), but they were missing here so `progressed` always
    # answered False - live, `smelt_gold` finished (coal -1, gold_ingot +3) and read as no
    # progress, and `mine_gold` gained 5 raw_gold that way too.
    "mine_gold", "smelt_gold", "craft_bow", "craft_arrows", "craft_gold_helmet",
}


def progressed(label: str, d: dict) -> bool:
    inv_up = any(v > 0 for v in d["inventory_delta"].values())
    armor_up = bool(d.get("armor_delta"))
    offhand_up = bool(d.get("offhand_delta"))
    if label in INVENTORY_LABELS:
        # A craft normally lands in the pack; a server that auto-equips armour puts the piece
        # straight on the body, so armour_craft progress also shows as armor_delta.
        return inv_up or armor_up or offhand_up
    if label == "equip_armor":
        return armor_up  # the piece leaves the pack for a worn slot
    if label == "equip_shield":
        return "shield" in d.get("offhand_delta", [])  # the shield reaches the off-hand slot
    if label == "hunt_string":
        return d["inventory_delta"].get("string", 0) > 0
    if label == "hunt_feather":
        return d["inventory_delta"].get("feather", 0) > 0
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
    # Overworld -> Nether gate. Each step's own item is the proof; the world-block steps use the
    # booleans above (lighting) or the obsidian leaving the pack (the frame).
    if label == "craft_bucket":
        return d["inventory_delta"].get("bucket", 0) > 0
    if label == "fill_bucket":
        return d["inventory_delta"].get("water_bucket", 0) > 0
    if label == "mine_obsidian":
        return d["inventory_delta"].get("obsidian", 0) > 0
    if label == "get_flint":
        return d["inventory_delta"].get("flint", 0) > 0
    if label == "craft_flint_and_steel":
        return d["inventory_delta"].get("flint_and_steel", 0) > 0
    if label == "build_portal":
        return d["inventory_delta"].get("obsidian", 0) < 0
    if label == "light_portal":
        return bool(d.get("portal_lit"))
    if label == "place_bed":
        return bool(d.get("bed_placed"))
    # Home base. Walking home (or up to the surface) shows as movement; the chest shows as the
    # item entering or leaving the pack; stowing/dropping shows as any carried item going down.
    if label in ("surface_up", "establish_base", "go_home"):
        return (d["pos_delta"] or 0) > 2
    if label == "escape_water":
        # The rescue worked if it actually moved the body out of the water; landing back on
        # ground shows in `on_ground`, but the walk itself is the proof it made progress.
        return (d["pos_delta"] or 0) > 2 or bool(d.get("on_ground"))
    if label == "craft_chest":
        return d["inventory_delta"].get("chest", 0) > 0
    if label == "place_chest":
        return d["inventory_delta"].get("chest", 0) < 0
    if label == "place_furnace":
        return d["inventory_delta"].get("furnace", 0) < 0
    if label in ("stow_surplus", "drop_junk"):
        return any(v < 0 for v in d["inventory_delta"].values())
    if label == "explore":
        return (d["pos_delta"] or 0) > 2
    return False
