from __future__ import annotations

import unittest

from mlaya.perception import parse_status


def _status(situation: dict) -> dict:
    return {"response": {"data": {"result": {"situation": situation}}}}


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


if __name__ == "__main__":
    unittest.main()
