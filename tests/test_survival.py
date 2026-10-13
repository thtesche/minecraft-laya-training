"""Tests for the server survival-policy presets (`mlaya.survival`). Stdlib unittest.

The values live in one table that a run applies at start and after every death, so these
tests pin the deliberate choices: the tuned base combat band, and the hunt stage that drops
the floor for a requested hunt without changing anything else.
"""
from __future__ import annotations

import copy
import unittest

from mlaya import survival


class BasePresetTest(unittest.TestCase):
    def test_combat_band_is_the_tuned_one(self):
        combat = survival.BASE_CHANGES["combat"]
        self.assertEqual(combat["engage_min_health"], 14)
        self.assertEqual(combat["critical_health"], 10)
        self.assertEqual(combat["recover_to_health"], 16)

    def test_recover_to_health_is_not_below_the_pursuit_floor(self):
        combat = survival.BASE_CHANGES["combat"]
        self.assertGreaterEqual(combat["recover_to_health"], combat["engage_min_health"])

    def test_scaffold_blocks_include_the_deep_and_nether_names(self):
        blocks = survival.BASE_CHANGES["navigation"]["scaffold_blocks"]
        for name in ("cobbled_deepslate", "netherrack", "basalt", "end_stone"):
            self.assertIn(name, blocks)

    def test_presets_are_indexed_by_name(self):
        self.assertIs(survival.PRESETS["base"], survival.BASE_CHANGES)
        self.assertEqual(survival.PRESETS["hunt"]["combat"]["engage_min_health"], 0)


class HuntPresetTest(unittest.TestCase):
    def test_hunt_drops_only_the_floor_values(self):
        base = survival.BASE_CHANGES
        hunt = survival.HUNT_CHANGES
        self.assertEqual(hunt["combat"]["engage_min_health"], 0)
        self.assertEqual(hunt["combat"]["critical_health"], 0)
        self.assertFalse(hunt["combat"]["retreat"])
        # Everything but those three fields matches the base stage.
        expected = copy.deepcopy(base)
        expected["combat"].update(
            {"engage_min_health": 0, "critical_health": 0, "retreat": False})
        self.assertEqual(hunt, expected)


if __name__ == "__main__":
    unittest.main()
