from __future__ import annotations

import unittest

from mlaya.perception import parse_resource_counts, parse_status


def _status(situation: dict) -> dict:
    return {"response": {"data": {"result": {"situation": situation}}}}


class ResourceCountParseTest(unittest.TestCase):
    """`parse_resource_counts` keeps the per-name `found` total across every loaded chunk, which
    the radius-limited hit list drops. A surface bot needs it to know that loaded diamond ore
    exists below, so the change that offers the tunnel dig can fire without a nearby hit."""

    def test_found_total_is_kept_outside_the_hit_radius(self) -> None:
        data = {"response": {"data": {"result": {"blocks": {"find": [
            {"name": "diamond_ore", "found": 15479, "listed": [
                {"x": 0, "y": 8, "z": 0, "name": "diamond_ore", "distance": 109.6},
            ]},
            {"name": "iron_ore", "found": 0, "listed": []},
        ]}}}}}
        self.assertEqual(parse_resource_counts(data), {"diamond_ore": 15479})

    def test_missing_find_block_is_empty(self) -> None:
        self.assertEqual(parse_resource_counts({}), {})


class ArmourParseTest(unittest.TestCase):
    """Worn armour must be read from the inventory stack location (head/torso/legs/feet).
    `situation.tools.armour` lists the carried set per class whether or not the piece is on
    the body, so treating it as "worn" reported armour that was still in the backpack."""

    def test_carried_armour_stays_inventory(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 28, "location": "main", "name": "diamond_chestplate", "count": 1},
            ]},
            "tools": {"armour": [
                {"class": "chestplate", "tier": "diamond", "item": "diamond_chestplate",
                 "slot": 28},
            ]},
        })
        gs = parse_status(st)
        self.assertEqual(gs.inventory.get("diamond_chestplate"), 1)
        self.assertEqual(gs.armor, {})

    def test_worn_armour_is_not_also_carried(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 6, "location": "torso", "name": "diamond_chestplate", "count": 1,
                 "durability": {"remaining": 528, "maximum": 528}},
                {"slot": 5, "location": "head", "name": "diamond_helmet", "count": 1,
                 "durability": {"remaining": 363, "maximum": 363}},
            ]},
            "tools": {"armour": [
                {"class": "helmet", "tier": "diamond", "item": "diamond_helmet", "slot": 5},
                {"class": "chestplate", "tier": "diamond", "item": "diamond_chestplate",
                 "slot": 6},
            ]},
        })
        gs = parse_status(st)
        self.assertNotIn("diamond_chestplate", gs.inventory)
        self.assertNotIn("diamond_helmet", gs.inventory)
        self.assertEqual(gs.armor["chestplate"]["item"], "diamond_chestplate")
        self.assertEqual(gs.armor["chestplate"]["tier"], "diamond")
        self.assertEqual(gs.armor["chestplate"]["max"], 528)
        self.assertEqual(gs.armor["helmet"]["item"], "diamond_helmet")
        self.assertEqual(gs.armor["helmet"]["left"], 363)


class OffhandParseTest(unittest.TestCase):
    """The off-hand item reports a location of its own and leaves the carried inventory, exactly
    like armour. This is what tells the loop a shield is actually worn (`equip_shield` done)."""

    def test_offhand_item_is_not_also_carried(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 40, "location": "off-hand", "name": "shield", "count": 1},
            ]},
        })
        gs = parse_status(st)
        self.assertEqual(gs.offhand, "shield")
        self.assertNotIn("shield", gs.inventory)
        self.assertIn("off-hand: shield", gs.narrative())

    def test_carried_shield_leaves_the_offhand_empty(self) -> None:
        st = _status({
            "inventory": {"stacks": [
                {"slot": 20, "location": "main", "name": "shield", "count": 1},
            ]},
        })
        gs = parse_status(st)
        self.assertIsNone(gs.offhand)
        self.assertEqual(gs.inventory.get("shield"), 1)


if __name__ == "__main__":
    unittest.main()
