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
        self.assertIs(survival.PRESETS["flee"], survival.FLEE_CHANGES)


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


class FleePresetTest(unittest.TestCase):
    """The flee stage is the exact inverse of the hunt stage: a pursuit floor past full health
    so the reflex evades instead of fighting and frees the body for the way home."""

    def test_pursuit_floor_is_past_full_health(self):
        combat = survival.FLEE_CHANGES["combat"]
        # Above the 20-point maximum, so the condition can never be met and the reflex never
        # pursues - the mirror of the hunt stage's zero floor.
        self.assertGreater(combat["engage_min_health"], survival.MAX_HEALTH)
        self.assertEqual(combat["engage_min_health"], survival.MAX_HEALTH + 1)

    def test_flee_keeps_the_emergency_floor_and_only_raises_the_pursuit_one(self):
        base = survival.BASE_CHANGES
        flee = survival.FLEE_CHANGES
        # Unlike the hunt, the critical-health self-defence floor is untouched: a cornered bot
        # still fights for its life below ten; only the pursuit of a winnable-looking fight stops.
        self.assertEqual(flee["combat"]["critical_health"], base["combat"]["critical_health"])
        self.assertTrue(flee["combat"]["retreat"])
        expected = copy.deepcopy(base)
        expected["combat"].update({"engage_min_health": survival.MAX_HEALTH + 1, "retreat": True})
        self.assertEqual(flee, expected)


if __name__ == "__main__":
    unittest.main()
