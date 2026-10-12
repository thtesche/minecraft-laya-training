"""Reward signal: `diff` + `progressed` for the overworld -> Nether gate.

The gate's later steps touch no simple inventory gain. Building the frame *spends* obsidian,
and lighting the portal changes no stack at all - it only turns the frame into a
`nether_portal` block. So the diff carries two world-block booleans and the frame is judged by
the obsidian leaving the pack, which is why these are pinned here.
"""
from __future__ import annotations

import unittest

from mlaya.perception import BlockHit, GameState
from mlaya.verify import diff, progressed


def gs(**kw) -> GameState:
    g = GameState()
    g.inventory = kw.pop("inventory", {})
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def _hit(name: str) -> list[BlockHit]:
    return [BlockHit(name, name, 0, -58, 0, 3.0)]


class NetherProgressTest(unittest.TestCase):
    def test_craft_bucket_needs_a_bucket(self) -> None:
        d = diff(gs(inventory={"iron_ingot": 3}), gs(inventory={"bucket": 1}))
        self.assertTrue(progressed("craft_bucket", d))
        self.assertFalse(progressed("craft_bucket",
                                    diff(gs(inventory={"iron_ingot": 3}),
                                         gs(inventory={"iron_ingot": 1}))))

    def test_fill_bucket_is_the_water_bucket_gain(self) -> None:
        d = diff(gs(inventory={"bucket": 1}), gs(inventory={"water_bucket": 1}))
        self.assertTrue(progressed("fill_bucket", d))

    def test_mine_obsidian_is_the_obsidian_gain(self) -> None:
        d = diff(gs(inventory={}), gs(inventory={"obsidian": 10}))
        self.assertTrue(progressed("mine_obsidian", d))
        # Cobblestone spent closing lava is not obsidian progress.
        self.assertFalse(progressed("mine_obsidian",
                                    diff(gs(inventory={"cobblestone": 5}),
                                         gs(inventory={"cobblestone": 3}))))

    def test_get_flint_is_the_flint_gain(self) -> None:
        self.assertTrue(progressed("get_flint", diff(gs(inventory={}),
                                                     gs(inventory={"flint": 1, "gravel": 8}))))
        self.assertFalse(progressed("get_flint", diff(gs(inventory={}),
                                                      gs(inventory={"gravel": 8}))))

    def test_craft_flint_and_steel_is_its_own_gain(self) -> None:
        d = diff(gs(inventory={"flint": 1, "iron_ingot": 1}),
                 gs(inventory={"flint_and_steel": 1}))
        self.assertTrue(progressed("craft_flint_and_steel", d))

    def test_build_portal_is_the_obsidian_leaving_the_pack(self) -> None:
        d = diff(gs(inventory={"obsidian": 10, "cobblestone": 21}),
                 gs(inventory={"cobblestone": 17}))
        self.assertTrue(progressed("build_portal", d))
        self.assertFalse(progressed("build_portal",
                                    diff(gs(inventory={"obsidian": 10}),
                                         gs(inventory={"obsidian": 10}))))

    def test_light_portal_is_the_portal_block_appearing(self) -> None:
        d = diff(gs(), gs(resources={"nether_portal": _hit("nether_portal")}))
        self.assertTrue(progressed("light_portal", d))
        self.assertFalse(progressed("light_portal", diff(gs(), gs())))

    def test_place_bed_is_the_bed_block_appearing(self) -> None:
        d = diff(gs(inventory={"white_bed": 1}),
                 gs(resources={"white_bed": _hit("white_bed")}))
        self.assertTrue(progressed("place_bed", d))
        self.assertFalse(progressed("place_bed",
                                    diff(gs(inventory={"white_bed": 1}),
                                         gs(inventory={}))))


class HomeBaseProgressTest(unittest.TestCase):
    """The base steps: movement for the walks, the chest entering/leaving the pack, and a
    falling inventory for stow/drop (either way pack space was freed)."""

    def test_walking_home_counts_as_movement(self) -> None:
        d = diff(gs(pos=(0.0, 64.0, 0.0)), gs(pos=(20.0, 64.0, 0.0)))
        self.assertTrue(progressed("establish_base", d))
        self.assertTrue(progressed("go_home", d))
        self.assertFalse(progressed("go_home", diff(gs(pos=(0.0, 64.0, 0.0)),
                                                    gs(pos=(1.0, 64.0, 0.0)))))

    def test_craft_chest_is_the_chest_gain(self) -> None:
        d = diff(gs(inventory={"oak_planks": 8}), gs(inventory={"chest": 1}))
        self.assertTrue(progressed("craft_chest", d))
        self.assertFalse(progressed("craft_chest",
                                    diff(gs(inventory={"oak_planks": 8}),
                                         gs(inventory={"oak_planks": 0}))))

    def test_place_chest_is_the_chest_leaving_the_pack(self) -> None:
        d = diff(gs(inventory={"chest": 1}), gs(inventory={}))
        self.assertTrue(progressed("place_chest", d))

    def test_stow_and_drop_are_any_item_going_down(self) -> None:
        d = diff(gs(inventory={"tuff": 64, "granite": 32}), gs(inventory={"tuff": 0, "granite": 32}))
        self.assertTrue(progressed("stow_surplus", d))
        self.assertTrue(progressed("drop_junk", d))
        self.assertFalse(progressed("stow_surplus",
                                    diff(gs(inventory={"tuff": 64}), gs(inventory={"tuff": 64}))))


if __name__ == "__main__":
    unittest.main()
