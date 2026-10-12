"""Action binding: label -> (tool, args) for the reactions and the food/bed chain.

The M2d reactions (`collect_drops`, `recover_drops`, `replace_tool`) and the bed chain
(`craft_bed`, `sleep`) are event-driven and hard to reach on demand in a live world, so
their precondition guards and argument binding are pinned here.
"""
from __future__ import annotations

import unittest

from mlaya.actions import Action, candidates, resolve
from mlaya.perception import BlockHit, GameState


def gs(**kw) -> GameState:
    g = GameState()
    g.inventory = kw.pop("inventory", {})
    g.tools = kw.pop("tools", {})
    g.mobs = kw.pop("mobs", [])
    for k, v in kw.items():
        setattr(g, k, v)
    return g


class CollectDropsTest(unittest.TestCase):
    def test_fresh_drop_is_picked_up(self) -> None:
        a = resolve("collect_drops", gs(drops=2))
        self.assertEqual(a, Action("collect_drops", "pick_up_items", {"radius": 16}, a.rationale))

    def test_no_drops_no_action(self) -> None:
        self.assertIsNone(resolve("collect_drops", gs(drops=0)))


class RecoverDropsTest(unittest.TestCase):
    def test_fresh_nearby_death(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(10.0, 64.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld")
        a = resolve("recover_drops", g)
        self.assertIsNotNone(a)
        self.assertEqual(a.tool, "pick_up_items")
        self.assertTrue(a.args["recover_death_items"])

    def test_too_old(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(10.0, 64.0, 0.0),
               last_death_age_s=200.0, dimension="overworld", death_dim="overworld")
        self.assertIsNone(resolve("recover_drops", g))

    def test_other_dimension(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(10.0, 64.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="the_nether")
        self.assertIsNone(resolve("recover_drops", g))

    def test_out_of_range(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(500.0, 64.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld")
        self.assertIsNone(resolve("recover_drops", g))


class ReplaceToolTest(unittest.TestCase):
    def _worn(self, left: float = 15.0, item: str = "stone_pickaxe",
              maximum: float = 131.0) -> dict:
        return {"pickaxe": {"tier": "stone", "item": item, "left": left, "max": maximum}}

    def test_worn_pickaxe_replaced_with_stone(self) -> None:
        g = gs(inventory={"crafting_table": 1, "cobblestone": 3, "acacia_log": 1},
               tools={"pickaxe": "stone"}, tool_meta=self._worn())
        a = resolve("replace_tool", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "stone_pickaxe"}])

    def test_upgrades_to_iron_when_banked(self) -> None:
        g = gs(inventory={"crafting_table": 1, "iron_ingot": 3, "acacia_log": 1, "cobblestone": 3},
               tools={"pickaxe": "iron"}, tool_meta=self._worn(item="iron_pickaxe", maximum=250.0))
        a = resolve("replace_tool", g)
        self.assertEqual(a.args["items"], [{"item_name": "iron_pickaxe"}])

    def test_diamond_pickaxe_stays_diamond(self) -> None:
        g = gs(inventory={"crafting_table": 1, "diamond": 3, "acacia_log": 1, "cobblestone": 3},
               tools={"pickaxe": "diamond"},
               tool_meta=self._worn(item="diamond_pickaxe", maximum=1561.0))
        a = resolve("replace_tool", g)
        self.assertEqual(a.args["items"], [{"item_name": "diamond_pickaxe"}])

    def test_healthy_pickaxe_is_left_alone(self) -> None:
        g = gs(inventory={"crafting_table": 1, "cobblestone": 3, "acacia_log": 1},
               tools={"pickaxe": "stone"}, tool_meta=self._worn(left=100.0))
        self.assertIsNone(resolve("replace_tool", g))


class BedChainTest(unittest.TestCase):
    def test_craft_bed_from_wool(self) -> None:
        g = gs(inventory={"crafting_table": 1, "white_wool": 3, "acacia_planks": 3})
        a = resolve("craft_bed", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "white_bed"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_craft_bed_from_string(self) -> None:
        g = gs(inventory={"crafting_table": 1, "string": 12, "acacia_log": 1})
        self.assertEqual(resolve("craft_bed", g).args["items"], [{"item_name": "white_bed"}])

    def test_craft_bed_needs_materials(self) -> None:
        g = gs(inventory={"crafting_table": 1, "acacia_planks": 3})
        self.assertIsNone(resolve("craft_bed", g))

    def test_sleep_at_night_with_bed(self) -> None:
        g = gs(inventory={"white_bed": 1}, phase="night", sleeping=False)
        a = resolve("sleep", g)
        self.assertEqual(a.tool, "sleep")
        self.assertEqual(a.args, {})

    def test_no_sleep_during_day(self) -> None:
        g = gs(inventory={"white_bed": 1}, phase="day", sleeping=False)
        self.assertIsNone(resolve("sleep", g))


class FoodChainTest(unittest.TestCase):
    def test_hunt_passive_animal(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 10.0)])
        a = resolve("hunt_food", g)
        self.assertEqual(a.args["mob_name"], "cow")
        self.assertEqual(a.args["drop_name"], "beef")
        self.assertEqual(a.args["count"], 6)

    def test_hunt_lone_zombie_for_rotten_flesh(self) -> None:
        g = gs(mobs=[("zombie", "hostile", 1, 10.0)], health=20.0)
        a = resolve("hunt_food", g)
        self.assertEqual(a.args["mob_name"], "zombie")
        self.assertEqual(a.args["drop_name"], "rotten_flesh")
        self.assertEqual(a.args["count"], 10)

    def test_no_food_target(self) -> None:
        self.assertIsNone(resolve("hunt_food", gs()))

    def test_night_hunt_ignores_far_animal(self) -> None:
        """After dark a distant animal is not worth the walk (the old 180 m night roam)."""
        g = gs(mobs=[("cow", "animal", 5, 40.0)], phase="night")
        self.assertIsNone(resolve("hunt_food", g))

    def test_night_hunt_takes_close_animal(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 12.0)], phase="night")
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")

    def test_day_hunt_reaches_far_animal(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 40.0)], phase="day")
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")

    def test_day_hunt_gives_up_beyond_range(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 80.0)], phase="day")
        self.assertIsNone(resolve("hunt_food", g))

    def test_night_hunt_boundary(self) -> None:
        self.assertIsNotNone(resolve("hunt_food", gs(mobs=[("pig", "animal", 1, 16.0)], phase="night")))
        self.assertIsNone(resolve("hunt_food", gs(mobs=[("pig", "animal", 1, 16.1)], phase="night")))

    def test_candidates_gate_hunt_food_by_range(self) -> None:
        far = gs(mobs=[("cow", "animal", 5, 40.0)], phase="night")
        self.assertNotIn("hunt_food", candidates(far))
        near = gs(mobs=[("cow", "animal", 5, 12.0)], phase="night")
        self.assertIn("hunt_food", candidates(near))

    def test_eat_first_edible(self) -> None:
        a = resolve("eat_food", gs(inventory={"cooked_beef": 1}))
        self.assertEqual(a.tool, "eat_food")
        self.assertEqual(a.args["food_name"], "cooked_beef")

    def test_cook_raw_meat_with_furnace(self) -> None:
        g = gs(inventory={"furnace": 1, "coal": 1, "beef": 2})
        a = resolve("cook_food", g)
        self.assertEqual(a.tool, "smelt_item")
        self.assertEqual(a.args["item_name"], "beef")
        self.assertEqual(a.args["fuel_item_name"], "coal")


class MakeRoomTest(unittest.TestCase):
    """The in-place unblock for a workstation that cannot be placed (a tight shaft)."""

    def test_digs_a_pocket_around_the_bot(self) -> None:
        a = resolve("make_room", gs(pos=(-277.5, 87.0, -51.6)))
        self.assertEqual(a.tool, "build_structure")
        cells = a.args["blocks"]
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(c["block_name"] == "air" for c in cells))
        self.assertEqual({(c["x"], c["y"], c["z"]) for c in cells},
                         {(-277, 87, -52), (-279, 87, -52),
                          (-278, 87, -51), (-278, 87, -53)})

    def test_none_without_position(self) -> None:
        self.assertIsNone(resolve("make_room", gs()))


class ReactionPriorityTest(unittest.TestCase):
    def _recoverable(self) -> GameState:
        return gs(pos=(0.0, 64.0, 0.0), death_pos=(1.0, 64.0, 0.0),
                  last_death_age_s=10.0, dimension="overworld", death_dim="overworld")

    def test_recover_outranks_replace_and_drops(self) -> None:
        g = self._recoverable()
        g.drops = 1
        g.tools = {"pickaxe": "stone"}
        g.tool_meta = {"pickaxe": {"tier": "stone", "item": "stone_pickaxe", "left": 10.0, "max": 131.0}}
        g.inventory = {"crafting_table": 1, "cobblestone": 3, "acacia_log": 1}
        self.assertEqual(candidates(g)[0], "recover_drops")

    def test_eating_rides_along_with_a_reaction(self) -> None:
        g = self._recoverable()
        g.food = 10.0
        g.inventory = {"cooked_beef": 1}
        self.assertEqual(candidates(g), ["eat_food", "recover_drops"])

    def test_sleep_not_offered_without_bed(self) -> None:
        g = gs(phase="night", sleeping=False)
        self.assertNotIn("sleep", candidates(g))


def _ore(name: str, dist: float = 12.0) -> dict:
    return {name: [BlockHit(name, name, 0, -58, 0, dist)]}


class DiamondChainTest(unittest.TestCase):
    def test_mine_diamond_needs_iron_pickaxe(self) -> None:
        near = _ore("diamond_ore")
        self.assertIsNone(resolve("mine_diamond", gs(tools={"pickaxe": "stone"}, resources=near)))
        a = resolve("mine_diamond", gs(tools={"pickaxe": "iron"}, resources=near))
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "diamond_ore", "count": 3})

    def test_mine_diamond_counts_remaining(self) -> None:
        g = gs(tools={"pickaxe": "iron"}, inventory={"diamond": 1}, resources=_ore("diamond_ore"))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 2)

    def test_craft_diamond_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "iron"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 3})
        a = resolve("craft_diamond_pickaxe", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_pickaxe"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_craft_diamond_pickaxe_needs_three(self) -> None:
        g = gs(tools={"pickaxe": "iron"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 2})
        self.assertIsNone(resolve("craft_diamond_pickaxe", g))

    def test_craft_diamond_sword_waits_for_pickaxe(self) -> None:
        base = {"crafting_table": 1, "stick": 2, "diamond": 2}
        self.assertIsNone(resolve("craft_diamond_sword", gs(tools={"pickaxe": "iron"}, inventory=base)))
        a = resolve("craft_diamond_sword", gs(tools={"pickaxe": "diamond"}, inventory=base))
        self.assertEqual(a.args["items"], [{"item_name": "diamond_sword"}])

    def test_candidates_offer_diamond_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "iron"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 3})
        self.assertIn("craft_diamond_pickaxe", candidates(g))

    def test_candidates_offer_mine_diamond_only_with_ore(self) -> None:
        self.assertIn("mine_diamond",
                      candidates(gs(tools={"pickaxe": "iron"}, resources=_ore("diamond_ore"))))
        self.assertNotIn("mine_diamond", candidates(gs(tools={"pickaxe": "iron"})))

    def test_candidates_no_diamond_sword_before_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "iron"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 2})
        self.assertNotIn("craft_diamond_sword", candidates(g))


if __name__ == "__main__":
    unittest.main()
