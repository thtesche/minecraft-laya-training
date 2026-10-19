"""Action binding: label -> (tool, args) for the reactions and the food/bed chain.

The M2d reactions (`collect_drops`, `recover_drops`, `replace_tool`) and the bed chain
(`craft_bed`, `sleep`) are event-driven and hard to reach on demand in a live world, so
their precondition guards and argument binding are pinned here.
"""
from __future__ import annotations

import math
import unittest

from mlaya.actions import (_all_nether_prep_steps, _in_portal, _leave_portal_target, Action,
                         _armor_to_wear, candidates, inventory_relief,
                           nether_expedition_complete, NETHER_HOPELESS_AFTER, resolve,
                           tech_tree_complete, WATER_RESCUE_AFTER)
from mlaya.perception import (BlockHit, GameState, FORTRESS_RADIUS, frame_obsidian_cells,
                              parse_frontier_heading, parse_resources)


def gs(**kw) -> GameState:
    g = GameState()
    g.inventory = kw.pop("inventory", {})
    g.tools = kw.pop("tools", {})
    g.mobs = kw.pop("mobs", [])
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def geared(**inv) -> GameState:
    """A state with the whole diamond kit worn and the shield in the off-hand (`_gear_ready`)."""
    base = {"crafting_table": 1}
    base.update(inv)
    g = gs(tools={"pickaxe": "diamond", "sword": "diamond"}, inventory=base)
    g.armor = {slot: {"item": f"diamond_{slot}"}
               for slot in ("helmet", "chestplate", "leggings", "boots")}
    g.offhand = "shield"
    return g


def combat_ready(g: GameState) -> GameState:
    """Mirror the pre-diamond combat kit (iron sword carried, full iron armour worn, shield in
    the off-hand) so `_combat_ready` holds and `mine_diamond` is allowed."""
    g.inventory = {**g.inventory, "iron_sword": 1}
    g.armor = {slot: {"item": f"iron_{slot}"}
               for slot in ("helmet", "chestplate", "leggings", "boots")}
    g.offhand = "shield"
    return g


def at_home(g: GameState, base: tuple = (-197, -47, -98)) -> GameState:
    """Stand the bot on the home base, so the base-only steps are eligible (`_at_base`)."""
    g.base = base
    g.pos = (base[0] + 0.5, base[1] + 1.0, base[2] + 0.5)
    return g


def lit_portal(g: GameState) -> GameState:
    """Load a lit nether portal beside the bot, so `_portal_lit` holds (the Nether lull)."""
    g.resources = {**(g.resources or {}),
                   "nether_portal": [BlockHit("nether_portal", "nether_portal",
                                              -196, -47, -98, 3.0)]}
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
               last_death_age_s=400.0, dimension="overworld", death_dim="overworld")
        self.assertIsNone(resolve("recover_drops", g))

    def test_far_but_recent_is_still_recovered(self) -> None:
        # The dropped stack's despawn timer pauses in an unloaded chunk, so a death a long walk
        # away (a respawn at the base, ~230 m) is still worth going back for.
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(230.0, 64.0, 0.0),
               last_death_age_s=120.0, dimension="overworld", death_dim="overworld")
        self.assertIsNotNone(resolve("recover_drops", g))

    def test_other_dimension(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(10.0, 64.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="the_nether")
        self.assertIsNone(resolve("recover_drops", g))

    def test_no_recovery_walk_in_the_nether_at_all(self) -> None:
        # A death in the Nether drops the whole kit where the bot fell -- a fortress ledge, a lava
        # sea, a ghast's reach. Walking back for it is the same ground that just killed the bot,
        # with nothing in hand, so the rule is the dimension and not the distance or the age.
        # Live: run43 died at (-23,57,134) and run44 was offered the walk straight back into it.
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(3.0, 64.0, 3.0),
               last_death_age_s=5.0, dimension="the_nether", death_dim="the_nether")
        self.assertIsNone(resolve("recover_drops", g))
        self.assertNotIn("recover_drops", candidates(g))

    def test_a_nether_death_is_not_recovered_from_the_overworld_either(self) -> None:
        # Same reason from the other side: the drops are in the Nether, and crossing back into the
        # dimension that killed the bot to fetch them is the trip we just refused.
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(10.0, 64.0, 0.0),
               last_death_age_s=5.0, dimension="overworld", death_dim="the_nether")
        self.assertIsNone(resolve("recover_drops", g))
        self.assertNotIn("recover_drops", candidates(g))

    def test_out_of_range(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(500.0, 64.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld")
        self.assertIsNone(resolve("recover_drops", g))

    def test_deep_death_waits_for_a_pickaxe(self) -> None:
        # A death well below the bot means digging down; by hand that loses the despawn race
        # (live, run12), so recovery waits for a pickaxe. The bot carries none here.
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(0.0, 0.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld")
        self.assertIsNone(resolve("recover_drops", g))
        self.assertNotIn("recover_drops", candidates(g))

    def test_deep_death_offered_once_a_pickaxe_is_carried(self) -> None:
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(0.0, 0.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld",
               tools={"pickaxe": "wooden"})
        self.assertIsNotNone(resolve("recover_drops", g))
        self.assertIn("recover_drops", candidates(g))

    def test_shallow_death_needs_no_pickaxe(self) -> None:
        # A death on roughly the bot's own level is a walk, not a dig, so it is offered unarmed.
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(6.0, 62.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld")
        self.assertIsNotNone(resolve("recover_drops", g))

    def test_a_written_off_site_is_not_offered_again(self) -> None:
        # A recovery walk that reached the site and found nothing writes off that cell
        # (`graph.Agent._note_recover`). Offering it again would only stand the bot at the
        # emptied site for the rest of the window (live, run14).
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(6.0, 62.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld",
               recover_written_off=frozenset({(6, 62, 0)}))
        self.assertIsNone(resolve("recover_drops", g))
        self.assertNotIn("recover_drops", candidates(g))

    def test_a_different_site_is_still_offered(self) -> None:
        # A later death elsewhere is a new cell, not written off, and is still walked.
        g = gs(pos=(0.0, 64.0, 0.0), death_pos=(6.0, 62.0, 0.0),
               last_death_age_s=30.0, dimension="overworld", death_dim="overworld",
               recover_written_off=frozenset({(50, 20, 50)}))
        self.assertIsNotNone(resolve("recover_drops", g))


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

    def test_iron_pickaxe_backed_up_before_a_long_dig(self) -> None:
        # 70/250 is 28%, above LOW_DURABILITY but below MINING_HEADROOM: a single long dig
        # could still outlast it inside one action, so reserve a fresh same-tier spare now.
        g = gs(inventory={"crafting_table": 1, "iron_ingot": 3, "acacia_log": 1,
                          "cobblestone": 3},
               tools={"pickaxe": "iron"},
               tool_meta=self._worn(item="iron_pickaxe", maximum=250.0, left=70.0))
        self.assertEqual(resolve("replace_tool", g).args["items"],
                         [{"item_name": "iron_pickaxe"}])

    def test_pickaxe_with_headroom_needs_no_backup(self) -> None:
        g = gs(inventory={"crafting_table": 1, "iron_ingot": 3, "acacia_log": 1,
                          "cobblestone": 3},
               tools={"pickaxe": "iron"},
               tool_meta=self._worn(item="iron_pickaxe", maximum=250.0, left=200.0))
        self.assertIsNone(resolve("replace_tool", g))

    def test_no_second_backup_when_a_spare_is_already_carried(self) -> None:
        g = gs(inventory={"crafting_table": 1, "iron_ingot": 3, "acacia_log": 1,
                          "cobblestone": 3, "iron_pickaxe": 2},
               tools={"pickaxe": "iron"},
               tool_meta=self._worn(item="iron_pickaxe", maximum=250.0, left=5.0))
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

    def test_sleep_at_the_base_needs_a_bed_standing_there(self) -> None:
        # Once the portal is lit the gate bed has to *stand at the base*: counting any loaded bed
        # sent the server to the leftover bed in the base's own shaft and pinned the respawn point
        # off the gate again (live, run19b it25: the bot slept nine blocks below the base).
        off = lit_portal(at_home(geared(cooked_beef=8, furnace=1)))
        off.phase = "night"
        off.resources = {**off.resources,
                         "white_bed": [BlockHit("white_bed", "white_bed", -196, -56, -98, 9.0)]}
        self.assertNotIn("sleep", candidates(off))
        self.assertIsNone(resolve("sleep", off))

    def test_sleep_at_the_base_works_on_the_base_bed(self) -> None:
        g = lit_portal(at_home(geared(cooked_beef=8, furnace=1)))
        g.phase = "night"
        g.resources = {**g.resources,
                       "white_bed": [BlockHit("white_bed", "white_bed", -198, -46, -98, 1.0)]}
        self.assertIn("sleep", candidates(g))
        self.assertIsNotNone(resolve("sleep", g))

    def test_a_carried_bed_walks_home_in_the_day(self) -> None:
        # `take_bed` can strike by day, but `place_bed` only lays the bed after dark. The bed
        # belongs home whatever the hour, so `go_home` offers in daylight too (live, run19b: the
        # bed was taken up in the day, no `go_home` came, and it was never laid).
        g = lit_portal(at_home(geared(cooked_beef=8, white_bed=1, furnace=1)))
        g.pos = (-196.5, -46.0, -60.5)
        g.phase = "day"
        self.assertIn("go_home", candidates(g))
        self.assertIsNotNone(resolve("go_home", g))


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

    def test_cook_with_wood_fuel_in_the_nether_lull(self) -> None:
        # After the portal is lit the daytime lull is food-prep time: raw meat cooks on any
        # fuel, so a bot carrying wood but no coal cooks instead of falling through to explore
        # (live, from-zero run: `coal` was None, so `cook_food` never fired). An animal in reach
        # keeps the food goal reachable, which is the other half of what cooking is offered for.
        g = lit_portal(at_home(geared(furnace=1, beef=7, acacia_planks=3)))
        g.mobs = [("cow", "animal", 1, 10.0)]
        self.assertIn("cook_food", candidates(g))
        a = resolve("cook_food", g)
        self.assertEqual(a.args["item_name"], "beef")
        self.assertEqual(a.args["fuel_item_name"], "acacia_planks")
        # A plank smelts 1.5 items, so one cooks exactly one item; coal would cook three (the
        # batch cap).
        self.assertEqual(a.args["count"], 1)

    def test_wood_fuel_waits_for_the_nether_lull(self) -> None:
        # Before the gate is open, cooking must not spend the wood the chain still wants and the
        # step is not even offered (only coal/charcoal may cook outside the lull).
        g = geared(furnace=1, beef=7, acacia_planks=3)
        self.assertIsNone(resolve("cook_food", g))
        self.assertNotIn("cook_food", candidates(g))

    def test_the_reserve_deepens_to_the_nether_target_in_the_lull(self) -> None:
        # 17 edibles is above the ordinary floor (16) but below the Nether target (64), so in the
        # lull the bot keeps stocking; outside it 17 is already enough (live: 17 edibles and no
        # work drove the bot to `explore`).
        lull = lit_portal(at_home(geared(beef=17, cooked_beef=0, cobblestone=12,
                                         flint_and_steel=1, white_bed=1, furnace=1)))
        lull.mobs = [("cow", "animal", 5, 10.0)]
        self.assertIn("hunt_food", candidates(lull))
        before = geared(beef=17, cooked_beef=0, cobblestone=12,
                        flint_and_steel=1, white_bed=1, furnace=1)
        before.mobs = [("cow", "animal", 5, 10.0)]
        self.assertNotIn("hunt_food", candidates(before))

    def test_hunt_food_stops_at_the_nether_target(self) -> None:
        # A full stack of cooked meat is the target; at it the bot stops hunting.
        g = lit_portal(at_home(geared(cooked_beef=64, cobblestone=12, flint_and_steel=1,
                                      white_bed=1, furnace=1)))
        g.mobs = [("cow", "animal", 5, 10.0)]
        self.assertIsNone(resolve("hunt_food", g))

    def test_eat_prefers_the_most_nutritious_food(self) -> None:
        # Cooked meat outlasts everything else and is spent first; among what is left, bread
        # (6.0) beats raw meat (1.8) and rotten flesh (0.8), so the worst food is a last resort.
        a = resolve("eat_food", gs(inventory={"bread": 3, "beef": 5, "cooked_beef": 1}))
        self.assertEqual(a.args["food_name"], "cooked_beef")
        b = resolve("eat_food", gs(inventory={"bread": 3, "beef": 5}))
        self.assertEqual(b.args["food_name"], "bread")
        c = resolve("eat_food", gs(inventory={"rotten_flesh": 2, "apple": 1}))
        self.assertEqual(c.args["food_name"], "apple")

    def test_a_wounded_bot_eats_to_reach_the_regeneration_threshold(self) -> None:
        # Food 17 is above the old hunger floor (14) but below the regeneration threshold (18):
        # the bot can neither eat nor heal, and bleeds out (live, run14: 2.7 HP at food 17 for
        # 17 ticks, then a mob finished it). A wound now makes it eat up to the regen band.
        g = gs(health=2.7, food=17.0, inventory={"rotten_flesh": 20})
        self.assertIn("eat_food", candidates(g))

    def test_a_full_health_bot_does_not_waste_food_banking(self) -> None:
        # The extra eat is only for a wound that cannot heal; at full health with food 17 there
        # is nothing to regenerate, so the food is kept.
        g = gs(health=20.0, food=17.0, inventory={"rotten_flesh": 20})
        self.assertNotIn("eat_food", candidates(g))

    def test_starving_still_eats_below_the_hunger_floor(self) -> None:
        # The old hunger floor is unchanged: hungry and unwounded, the bot still eats.
        g = gs(health=20.0, food=13.0, inventory={"rotten_flesh": 20})
        self.assertIn("eat_food", candidates(g))


class NetherLullWaitTest(unittest.TestCase):
    """The daytime lull after the portal is lit: the only work left is the night bed, so the bot
    holds at the base instead of wandering off on `explore` (live, from-zero run: 25 idle ticks,
    five of them wasted on `[NAVIGATION_STOPPED]` pathfinder timeouts)."""

    @staticmethod
    def _idle() -> GameState:
        # The Nether prep is already banked (bow, arrows, deep cobblestone stack, and a furnace
        # standing at the base), so the lull has no productive step left and the only sensible
        # tick is holding at the base.
        g = lit_portal(at_home(geared(flint_and_steel=1, cobblestone=64, white_bed=1,
                                      iron_ingot=1, cooked_beef=24,
                                      bow=1, arrow=16, diamond_pickaxe=2)))
        g.food = 20.0
        g.chest_pos = (-198, -47, -98)
        g.furnace_pos = (-198, -46, -99)
        return g

    def test_holding_at_the_base_beats_exploring(self) -> None:
        self.assertEqual(candidates(self._idle()), ["wait"])
        a = resolve("wait", self._idle())
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["z"]), (-197, -98))
        self.assertEqual(a.args["range"], 1)

    def test_before_the_portal_falls_back_to_explore(self) -> None:
        # No lit portal yet: the gate is still to do, so the bot explores rather than waits.
        g = at_home(geared(flint_and_steel=1, cobblestone=12, white_bed=1,
                           iron_ingot=1, cooked_beef=24, furnace=1))
        g.food = 20.0
        self.assertIn("explore", candidates(g))
        self.assertNotIn("wait", candidates(g))

    def test_an_open_reserve_with_no_hunt_in_reach_holds(self) -> None:
        # A reserve still short with no animal in hunt reach: the requirement drops out (the
        # reach test), so the lull holds at the base rather than roaming after it. Roaming for
        # food was tried live (run19b) and reverted - it carried the bot 150-200 m out.
        g = self._idle()
        g.mobs = [("cow", "animal", 3, 300.0)]
        self.assertEqual(candidates(g), ["wait"])

    def test_wait_needs_the_lit_portal(self) -> None:
        g = at_home(geared(flint_and_steel=1, cobblestone=12, white_bed=1, cooked_beef=24))
        self.assertIsNone(resolve("wait", g))

    def test_wait_needs_the_base(self) -> None:
        g = lit_portal(geared(flint_and_steel=1, cobblestone=12, white_bed=1, cooked_beef=24))
        self.assertIsNone(resolve("wait", g))


class NetherPrepTest(unittest.TestCase):
    """The post-portal lull stocks the Nether kit beyond the food: a bow with arrows, a gold
    helmet and a deep cobblestone stack (`_nether_prep_steps`). Each step is offered only while
    it can progress, so an unreachable material never blocks the run."""

    @staticmethod
    def _lull(**inv) -> GameState:
        base = {"flint_and_steel": 1, "white_bed": 1, "cooked_beef": 24, "furnace": 1,
                "cobblestone": 64, "acacia_planks": 4}
        base.update(inv)
        g = at_home(lit_portal(geared(**base)))
        g.food = 20.0
        return g

    # --- bow: three string from a lone spider plus sticks ---
    def test_craft_bow_from_string_and_sticks(self) -> None:
        a = resolve("craft_bow", self._lull(string=3))
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "bow"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_craft_bow_needs_three_string(self) -> None:
        self.assertIsNone(resolve("craft_bow", self._lull(string=2)))

    def test_bow_string_goal_drops_to_three_once_the_bed_exists(self) -> None:
        # Twelve string is the bed's price; with the bed placed the bow only wants three.
        g = self._lull()
        g.mobs = [("spider", "hostile", 1, 8.0)]
        g.health = 20.0
        a = resolve("hunt_string", g)
        self.assertEqual(a.args["count"], 3)

    # --- arrows: flint, stick and feather ---
    def test_craft_arrows_needs_flint_and_feather(self) -> None:
        a = resolve("craft_arrows", self._lull(bow=1, flint=1, feather=1))
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "arrow"}])

    def test_craft_arrows_needs_a_feather(self) -> None:
        self.assertIsNone(resolve("craft_arrows", self._lull(bow=1, flint=1)))

    def test_arrows_are_not_crafted_before_the_lull(self) -> None:
        # No lit portal: the prep is out of scope, so arrow craft stays closed.
        self.assertIsNone(resolve("craft_arrows", geared(flint=1, feather=1)))

    def test_feathers_come_from_a_chicken(self) -> None:
        g = self._lull(bow=1)
        g.mobs = [("chicken", "animal", 4, 6.0)]
        a = resolve("hunt_feather", g)
        self.assertEqual(a.args["mob_name"], "chicken")
        self.assertEqual(a.args["drop_name"], "feather")

    def test_flint_reopens_for_arrows_after_flint_and_steel(self) -> None:
        # Carrying flint and steel used to close the gravel dig for good; the bow reopens it.
        g = self._lull(bow=1)
        g.resources = {**g.resources,
                       "gravel": [BlockHit("gravel", "gravel", -196, -48, -98, 5.0)]}
        a = resolve("get_flint", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args["block_name"], "gravel")

    # --- gold helmet: mine, smelt, craft ---
    def test_mine_gold_for_the_helmet(self) -> None:
        g = self._lull()
        g.resources = {**g.resources,
                       "gold_ore": [BlockHit("gold_ore", "gold_ore", -200, 20, -100, 40.0)]}
        a = resolve("mine_gold", g)
        self.assertEqual(a.args["block_name"], "gold_ore")
        self.assertEqual(a.args["count"], 5)

    def test_mine_gold_needs_a_deposit(self) -> None:
        self.assertIsNone(resolve("mine_gold", self._lull()))

    def test_mine_gold_counts_raw_gold_already_banked(self) -> None:
        g = self._lull(raw_gold=2)
        g.resources = {**g.resources,
                       "gold_ore": [BlockHit("gold_ore", "gold_ore", -200, 20, -100, 40.0)]}
        self.assertIsNone(resolve("mine_gold", g))

    def test_smelt_gold(self) -> None:
        a = resolve("smelt_gold", self._lull(raw_gold=5, coal=1))
        self.assertEqual(a.tool, "smelt_item")
        self.assertEqual(a.args["item_name"], "raw_gold")
        self.assertEqual(a.args["fuel_item_name"], "coal")

    def test_craft_gold_helmet(self) -> None:
        a = resolve("craft_gold_helmet", self._lull(gold_ingot=5))
        self.assertEqual(a.args["items"], [{"item_name": "golden_helmet"}])

    def test_gold_helmet_needs_five_ingots(self) -> None:
        self.assertIsNone(resolve("craft_gold_helmet", self._lull(gold_ingot=4)))

    def test_a_carried_helmet_closes_the_gold_chain(self) -> None:
        self.assertIsNone(resolve("craft_gold_helmet", self._lull(gold_ingot=5, golden_helmet=1)))

    # --- candidates: the lull offers the prep, the chain does not ---
    def test_lull_offers_the_prep_steps(self) -> None:
        g = self._lull(string=3, bow=1, flint=1, feather=1)
        g.resources = {**g.resources,
                       "gold_ore": [BlockHit("gold_ore", "gold_ore", -200, 20, -100, 40.0)]}
        cand = candidates(g)
        self.assertIn("craft_arrows", cand)
        self.assertIn("mine_gold", cand)

    def test_no_prep_before_the_portal(self) -> None:
        g = geared(flint_and_steel=1, cobblestone=64, string=3)
        self.assertNotIn("craft_bow", candidates(g))

    def test_light_cobble_floor_before_the_portal(self) -> None:
        # Before the portal the floor stays twelve, so twelve carried blocks are enough.
        self.assertNotIn("mine_stone", candidates(geared(cobblestone=12)))

    def test_deep_cobble_stack_is_stocked_in_the_lull(self) -> None:
        g = self._lull(bow=1, arrow=16, cobblestone=12)
        self.assertIn("mine_stone", candidates(g))

    def test_mine_stone_is_not_listed_twice(self) -> None:
        # Both the light floor and the deep lull stack want it; the list keeps one occurrence.
        g = self._lull(bow=1, arrow=16, cobblestone=5)
        self.assertEqual(candidates(g).count("mine_stone"), 1)

    def test_the_gold_dig_waits_for_daylight(self) -> None:
        # At night the run sleeps; the deep gold dig is not offered then, but it stays pending
        # for the completion gate so nightfall cannot end the run early.
        from mlaya.actions import _all_nether_prep_steps, tech_tree_complete
        g = self._lull(bow=1, arrow=16)
        g.resources = {**g.resources,
                       "gold_ore": [BlockHit("gold_ore", "gold_ore", -200, 20, -100, 40.0)],
                       "white_bed": [BlockHit("white_bed", "white_bed", -196, -47, -97, 2.0)]}
        g.phase = "night"
        self.assertNotIn("mine_gold", candidates(g))
        self.assertIn("mine_gold", _all_nether_prep_steps(g))
        self.assertFalse(tech_tree_complete(g))
        g.phase = "day"
        self.assertIn("mine_gold", candidates(g))


class OverworldGateTest(unittest.TestCase):
    """A bed placed in the Nether explodes and water evaporates there, so the Overworld-only
    steps (`sleep`, `place_bed`, `fill_bucket`) are refused off-dimension."""

    def _night(self, dimension: str) -> GameState:
        g = lit_portal(at_home(geared(white_bed=1)))
        g.phase = "night"
        g.dimension = dimension
        return g

    def test_sleep_refused_in_the_nether(self) -> None:
        g = self._night("minecraft:the_nether")
        self.assertIsNone(resolve("sleep", g))
        self.assertNotIn("sleep", candidates(g))

    def test_place_bed_refused_in_the_nether(self) -> None:
        self.assertIsNone(resolve("place_bed", self._night("minecraft:the_nether")))

    def test_overworld_still_allows_the_bed(self) -> None:
        # The bed belongs in the Overworld; `place_bed` lays the carried bed at the gate there.
        self.assertIsNotNone(resolve("place_bed", self._night("minecraft:overworld")))

    def test_fill_bucket_refused_in_the_nether(self) -> None:
        g = geared(bucket=1)
        g.dimension = "the_nether"
        self.assertIsNone(resolve("fill_bucket", g))


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

    def test_digs_the_cell_a_placement_was_refused_at(self) -> None:
        # Live, from-zero run: `place_chest` targets the cell beside the base anchor, that cell
        # held `grass_block`, and a pocket around the bot would not have touched it.
        a = resolve("make_room", gs(pos=(-178.5, 63.0, -120.4), blocked_cell=(-180, 64, -118)))
        self.assertEqual(a.tool, "build_structure")
        self.assertEqual(a.args["blocks"],
                         [{"x": -180, "y": 64, "z": -118, "block_name": "air"}])


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


class IronOreRuleTest(unittest.TestCase):
    """Iron ore needs a stone pickaxe or better; a wooden one destroys it with no drop (there is
    no silk touch, so the drop is raw iron, which is what the smelt step wants). A break inside
    one dig drops the bot onto the swapped-in lower tier and destroys the remaining ore, so the
    dig is also refused when the pickaxe is nearly spent and no sufficient backup is carried."""

    def _meta(self, left: float, item: str, tier: str, maximum: float = 131.0) -> dict:
        return {"pickaxe": {"tier": tier, "item": item, "left": left, "max": maximum}}

    def test_mine_iron_needs_a_stone_pickaxe(self) -> None:
        near = _ore("iron_ore")
        for tier in ("none", "wooden", "golden"):
            self.assertIsNone(resolve("mine_iron", gs(tools={"pickaxe": tier}, resources=near)))
            self.assertNotIn("mine_iron",
                             candidates(gs(tools={"pickaxe": tier}, resources=near)))
        a = resolve("mine_iron", gs(tools={"pickaxe": "stone"}, resources=near))
        self.assertEqual(a.args, {"block_name": "iron_ore", "count": 3})

    def test_spent_pickaxe_without_a_backup_does_not_start(self) -> None:
        worn = gs(tools={"pickaxe": "stone"},
                  tool_meta=self._meta(50.0, "stone_pickaxe", "stone"),
                  resources=_ore("iron_ore"))
        self.assertNotIn("mine_iron", candidates(worn))
        self.assertIsNone(resolve("mine_iron", worn))

    def test_a_carried_backup_allows_the_dig(self) -> None:
        g = gs(tools={"pickaxe": "stone"}, inventory={"stone_pickaxe": 2},
               tool_meta=self._meta(50.0, "stone_pickaxe", "stone"),
               resources=_ore("iron_ore"))
        self.assertIn("mine_iron", candidates(g))
        self.assertIsNotNone(resolve("mine_iron", g))

    def test_a_spent_iron_pickaxe_can_remine_its_own_replacement(self) -> None:
        # Regression: the old `tier < IRON_PICK_TIER` bound left a spent iron pickaxe with no
        # way to gather the three iron its replacement needs.
        g = gs(tools={"pickaxe": "iron"}, inventory={"iron_pickaxe": 1, "stone_pickaxe": 1},
               tool_meta=self._meta(50.0, "iron_pickaxe", "iron", 250.0),
               resources=_ore("iron_ore"))
        self.assertIn("mine_iron", candidates(g))

    def test_a_fresh_iron_pickaxe_stops_mining_iron(self) -> None:
        # Nothing wants iron once the pickaxe is fresh and the whole combat kit (sword, armour,
        # shield) is in place; before the kit exists the iron is what builds it.
        g = gs(tools={"pickaxe": "iron"}, offhand="shield",
               inventory={"iron_sword": 1},
               armor={slot: {"item": f"iron_{slot}"}
                      for slot in ("helmet", "chestplate", "leggings", "boots")},
               tool_meta=self._meta(200.0, "iron_pickaxe", "iron", 250.0),
               resources=_ore("iron_ore"))
        self.assertNotIn("mine_iron", candidates(g))

    def test_a_missing_combat_piece_still_pulls_iron(self) -> None:
        # The pre-diamond kit is the reason to keep digging iron after the pickaxe exists.
        g = gs(tools={"pickaxe": "iron"}, offhand="shield",
               tool_meta=self._meta(200.0, "iron_pickaxe", "iron", 250.0),
               resources=_ore("iron_ore"))
        self.assertIn("mine_iron", candidates(g))

    def test_stone_is_not_guarded(self) -> None:
        # A spent wooden pickaxe must still be allowed to gather the cobble a stone replacement
        # needs - guarding stone would deadlock the bootstrap.
        g = gs(tools={"pickaxe": "wooden"}, inventory={"wooden_pickaxe": 1},
               tool_meta=self._meta(10.0, "wooden_pickaxe", "wooden", 59.0))
        self.assertIn("mine_stone", candidates(g))


class DiamondChainTest(unittest.TestCase):
    def test_mine_diamond_needs_iron_pickaxe(self) -> None:
        # Wood, stone and gold all destroy diamond ore without a drop, so neither the
        # candidate nor the binding may fire below iron (golden is tier 1, not "better").
        near = _ore("diamond_ore")
        for tier in ("wooden", "golden", "stone"):
            self.assertIsNone(
                resolve("mine_diamond", combat_ready(gs(tools={"pickaxe": tier}, resources=near))))
            self.assertNotIn(
                "mine_diamond",
                candidates(combat_ready(gs(tools={"pickaxe": tier}, resources=near))))
        a = resolve("mine_diamond", combat_ready(gs(tools={"pickaxe": "iron"}, resources=near)))
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "diamond_ore", "count": 3})

    def test_mine_diamond_counts_remaining(self) -> None:
        g = combat_ready(gs(tools={"pickaxe": "iron"}, inventory={"diamond": 1},
                            resources=_ore("diamond_ore")))
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
                      candidates(combat_ready(gs(tools={"pickaxe": "iron"},
                                                  resources=_ore("diamond_ore")))))
        self.assertNotIn("mine_diamond",
                         candidates(combat_ready(gs(tools={"pickaxe": "iron"}))))

    def test_mine_diamond_waits_for_the_combat_kit(self) -> None:
        # The dive for the diamonds is what killed the unarmed bot (live, run12): no `mine_diamond`
        # until an iron sword, a full iron armour set and the shield are in place.
        g = gs(tools={"pickaxe": "iron"}, resources=_ore("diamond_ore"))
        self.assertNotIn("mine_diamond", candidates(g))
        self.assertIsNone(resolve("mine_diamond", g))
        self.assertIn("mine_diamond", candidates(combat_ready(gs(
            tools={"pickaxe": "iron"}, resources=_ore("diamond_ore")))))

    def test_deep_unscanned_diamond_ore_still_offers_the_dig(self) -> None:
        # The resource scan is radius-limited (RESOURCE_RADIUS), so diamond ore more than 64 m
        # down never appears in `resources` while the bot stands on the surface over it. The raw
        # `found` count from `view_blocks` proves the ore is loaded and `collect_block` tunnels to
        # it, so the dig must stay on offer - otherwise a lost diamond armour piece (the live
        # case: the boots were destroyed and the whole kit sat at 33/36 slots) could never be
        # rebuilt and `_gear_ready` would never close.
        deep = combat_ready(gs(tools={"pickaxe": "diamond"},
                               inventory={"diamond_sword": 1, "diamond_chestplate": 1,
                                          "diamond_leggings": 1, "diamond_helmet": 1,
                                          "diamond": 1},
                               resource_found={"diamond_ore": 15479}))
        self.assertIn("mine_diamond", candidates(deep))
        self.assertEqual(resolve("mine_diamond", deep).args["count"], 3)

    def test_no_dig_offer_without_any_diamond_ore_loaded(self) -> None:
        self.assertNotIn("mine_diamond",
                         candidates(combat_ready(gs(tools={"pickaxe": "diamond"},
                                                    resource_found={"iron_ore": 900,
                                                                    "coal_ore": 400}))))

    def test_no_diamond_dig_on_a_pickaxe_that_needs_a_backup(self) -> None:
        # Breaking mid-dig would destroy the ore with the swapped-in lower tier, so do not
        # start while the pickaxe is due for a backup (unknown durability must not block).
        def meta(left):
            return {"pickaxe": {"tier": "iron", "item": "iron_pickaxe",
                                "left": left, "max": 250.0}}
        worn = combat_ready(gs(tools={"pickaxe": "iron"}, tool_meta=meta(70.0),
                               resources=_ore("diamond_ore")))
        self.assertNotIn("mine_diamond", candidates(worn))
        fresh = combat_ready(gs(tools={"pickaxe": "iron"}, tool_meta=meta(200.0),
                                resources=_ore("diamond_ore")))
        self.assertIn("mine_diamond", candidates(fresh))

    def test_candidates_no_diamond_sword_before_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "iron"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 2})
        self.assertNotIn("craft_diamond_sword", candidates(g))

    def test_candidates_mine_diamond_with_diamond_pickaxe_for_the_sword(self) -> None:
        # Regression: bounding mine_diamond at the diamond tier deadlocked the sword, since
        # the diamond pickaxe could no longer mine the two diamonds the sword still needs.
        g = combat_ready(gs(tools={"pickaxe": "diamond"}, resources=_ore("diamond_ore")))
        self.assertIn("mine_diamond", candidates(g))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 2)

    def test_candidates_mine_diamond_for_armour_after_the_sword(self) -> None:
        # After the pickaxe and the sword the diamond goal is the armour set, chestplate first
        # (eight diamonds), so mining continues; it stops once every piece is built.
        g = combat_ready(gs(tools={"pickaxe": "diamond"}, inventory={"diamond_sword": 1},
                            resources=_ore("diamond_ore")))
        self.assertIn("mine_diamond", candidates(g))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 8)
        done = combat_ready(gs(tools={"pickaxe": "diamond"},
                               inventory={"diamond_sword": 1, "diamond_chestplate": 1,
                                          "diamond_leggings": 1, "diamond_boots": 1,
                                          "diamond_helmet": 1},
                               resources=_ore("diamond_ore")))
        self.assertNotIn("mine_diamond", candidates(done))

    def test_candidates_offer_diamond_sword_after_two_diamonds(self) -> None:
        g = gs(tools={"pickaxe": "diamond"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 2},
               resources=_ore("diamond_ore"))
        self.assertIn("craft_diamond_sword", candidates(g))


class DiamondArmorTest(unittest.TestCase):
    """The armour tier: 24 diamonds (chestplate 8, leggings 7, boots 4, helmet 5), built one
    piece at a time in protection-first order, then worn with a single `equip` batch."""

    def _ready(self, resources=None, **inv):
        base = {"crafting_table": 1, "stick": 2, "diamond_sword": 1}
        base.update(inv)
        return gs(tools={"pickaxe": "diamond"}, inventory=base, resources=resources or {})

    def test_chestplate_is_the_first_piece(self) -> None:
        a = resolve("craft_diamond_chestplate", self._ready(diamond=8))
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_chestplate"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_armour_needs_no_sticks(self) -> None:
        g = gs(tools={"pickaxe": "diamond"},
               inventory={"crafting_table": 1, "diamond": 8, "diamond_sword": 1})
        self.assertIn("craft_diamond_chestplate", candidates(g))

    def test_only_the_next_missing_piece_is_offered(self) -> None:
        # 24 diamonds would cover the whole set, but the chain still asks for the chestplate.
        cands = candidates(self._ready(diamond=24))
        self.assertIn("craft_diamond_chestplate", cands)
        self.assertNotIn("craft_diamond_leggings", cands)

    def test_next_piece_after_the_chestplate_is_crafted(self) -> None:
        cands = candidates(self._ready(diamond=7, diamond_chestplate=1))
        self.assertIn("craft_diamond_leggings", cands)
        self.assertNotIn("craft_diamond_chestplate", cands)

    def test_mine_count_targets_the_next_piece_only(self) -> None:
        g = combat_ready(gs(tools={"pickaxe": "diamond"},
                            inventory={"diamond_sword": 1, "diamond": 3},
                            resources=_ore("diamond_ore")))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 5)  # chestplate costs 8

    def test_equip_armor_wears_the_crafted_pieces(self) -> None:
        g = self._ready(diamond_chestplate=1)
        self.assertIn("equip_armor", candidates(g))
        a = resolve("equip_armor", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_chestplate"}])

    def test_equip_armor_skips_worn_pieces(self) -> None:
        g = self._ready(diamond_chestplate=1, diamond_leggings=1)
        g.armor = {"chestplate": {"tier": "diamond", "item": "diamond_chestplate"}}
        a = resolve("equip_armor", g)
        self.assertEqual(a.args["items"], [{"item_name": "diamond_leggings"}])

    def test_worn_armour_is_not_rebuilt(self) -> None:
        g = self._ready(diamond=8)
        g.armor = {"chestplate": {"tier": "diamond", "item": "diamond_chestplate"}}
        self.assertNotIn("craft_diamond_chestplate", candidates(g))
        self.assertIn("craft_diamond_leggings", candidates(g))

    def test_full_set_ends_the_diamond_chain(self) -> None:
        g = self._ready(diamond=1, resources=_ore("diamond_ore"))
        g.armor = {slot: {"tier": "diamond", "item": f"diamond_{slot}"}
                   for slot in ("chestplate", "leggings", "boots", "helmet")}
        self.assertNotIn("mine_diamond", candidates(g))
        self.assertIsNone(resolve("mine_diamond", g))


class ShieldTest(unittest.TestCase):
    """The shield: 6 planks + 1 iron ingot, crafted at a table and then worn in the off-hand.
    Same lesson as the armour: `craft_item` only puts it in the pack, so it blocks nothing until
    an `equip` to the off-hand slot actually moves it there."""

    def _ready(self, **inv):
        base = {"crafting_table": 1, "iron_ingot": 1, "oak_planks": 6}
        base.update(inv)
        return gs(tools={"pickaxe": "iron"}, inventory=base)

    def test_shield_is_crafted_once_planks_and_iron_are_there(self) -> None:
        g = self._ready()
        self.assertIn("craft_shield", candidates(g))
        a = resolve("craft_shield", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "shield"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_shield_waits_for_the_iron_pickaxe(self) -> None:
        # Iron is scarce; the first ingots belong to the pickaxe that unlocks diamonds.
        g = gs(tools={"pickaxe": "stone"},
               inventory={"crafting_table": 1, "iron_ingot": 1, "oak_planks": 6})
        self.assertNotIn("craft_shield", candidates(g))
        self.assertIsNone(resolve("craft_shield", g))

    def test_shield_needs_six_planks(self) -> None:
        g = self._ready(oak_planks=1)
        self.assertNotIn("craft_shield", candidates(g))
        # Logs the bot converts into planks count toward the six.
        g.inventory["oak_log"] = 2
        self.assertIn("craft_shield", candidates(g))

    def test_shield_needs_an_ingot_and_a_table(self) -> None:
        self.assertNotIn("craft_shield", candidates(self._ready(iron_ingot=0)))
        g = gs(tools={"pickaxe": "iron"},
               inventory={"iron_ingot": 1, "oak_planks": 6})
        self.assertNotIn("craft_shield", candidates(g))

    def test_carried_shield_is_not_rebuilt(self) -> None:
        self.assertNotIn("craft_shield", candidates(self._ready(shield=1)))

    def test_equip_shield_puts_it_in_the_off_hand(self) -> None:
        g = self._ready(shield=1)
        self.assertIn("equip_shield", candidates(g))
        a = resolve("equip_shield", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "shield", "destination": "off-hand"}])

    def test_worn_shield_is_not_rebuilt_or_reworn(self) -> None:
        g = self._ready()
        g.offhand = "shield"  # equipped leaves the carried inventory
        self.assertNotIn("craft_shield", candidates(g))
        self.assertNotIn("equip_shield", candidates(g))
        self.assertIsNone(resolve("equip_shield", g))

    def test_equip_shield_needs_a_carried_shield(self) -> None:
        self.assertNotIn("equip_shield", candidates(self._ready()))
        self.assertIsNone(resolve("equip_shield", self._ready()))

    def test_missing_combat_pieces_pull_iron(self) -> None:
        # The kit (shield, sword, armour) is the reason to dig iron once the pickaxe exists; the
        # mine is capped to one batch, so a 27-ingot goal is not a single unbounded dig.
        g = gs(tools={"pickaxe": "diamond"},
               inventory={"crafting_table": 1, "oak_planks": 6}, resources=_ore("iron_ore"))
        self.assertIn("mine_iron", candidates(g))
        self.assertEqual(resolve("mine_iron", g).args["count"], 8)

    def test_complete_combat_kit_does_not_quarry_iron(self) -> None:
        g = gs(tools={"pickaxe": "diamond"},
               inventory={"crafting_table": 1, "iron_sword": 1},
               armor={slot: {"item": f"iron_{slot}"}
                      for slot in ("helmet", "chestplate", "leggings", "boots")},
               offhand="shield", resources=_ore("iron_ore"))
        self.assertNotIn("mine_iron", candidates(g))
        self.assertIsNone(resolve("mine_iron", g))


class CombatKitTest(unittest.TestCase):
    """The pre-diamond combat kit: an iron sword and a full iron armour set, crafted after the
    iron pickaxe and before the first diamond dive (`mine_diamond` is gated on `_combat_ready`)."""

    def _ready(self, **inv):
        base = {"crafting_table": 1, "iron_ingot": 8, "stick": 2}
        base.update(inv)
        return gs(tools={"pickaxe": "iron"}, inventory=base)

    def test_iron_sword_needs_the_pickaxe_and_two_ingots(self) -> None:
        g = self._ready()
        self.assertIn("craft_iron_sword", candidates(g))
        a = resolve("craft_iron_sword", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "iron_sword"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_iron_sword_waits_for_the_iron_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "stone"},
               inventory={"crafting_table": 1, "iron_ingot": 8, "stick": 2})
        self.assertNotIn("craft_iron_sword", candidates(g))
        self.assertIsNone(resolve("craft_iron_sword", g))

    def test_carried_sword_is_not_rebuilt(self) -> None:
        g = self._ready(iron_sword=1)
        self.assertNotIn("craft_iron_sword", candidates(g))

    def test_armour_is_built_most_protection_first(self) -> None:
        g = self._ready(iron_ingot=24)
        self.assertIn("craft_iron_chestplate", candidates(g))
        self.assertNotIn("craft_iron_leggings", candidates(g))

    def test_next_piece_after_the_chestplate(self) -> None:
        g = self._ready(iron_ingot=24, iron_chestplate=1)
        self.assertIn("craft_iron_leggings", candidates(g))
        self.assertNotIn("craft_iron_chestplate", candidates(g))

    def test_armour_needs_the_ingots(self) -> None:
        g = self._ready(iron_ingot=7)  # the leggings' cost, but the chestplate is next
        self.assertNotIn("craft_iron_chestplate", candidates(g))

    def test_equip_wears_the_crafted_iron_piece(self) -> None:
        g = self._ready(iron_chestplate=1)
        self.assertEqual(resolve("equip_armor", g).args["items"],
                         [{"item_name": "iron_chestplate"}])

    def test_equip_never_downgrades_worn_diamond(self) -> None:
        # The diamond set is crafted on top of the iron one; the displaced iron piece must not be
        # traded back over the diamond it is sitting under.
        g = self._ready(iron_chestplate=1, iron_helmet=1)
        g.armor = {"chestplate": {"item": "diamond_chestplate"}}
        self.assertEqual(resolve("equip_armor", g).args["items"],
                         [{"item_name": "iron_helmet"}])

    def test_partial_kit_still_blocks_the_diamond_dive(self) -> None:
        g = gs(tools={"pickaxe": "iron"}, inventory={"iron_sword": 1},
               resources=_ore("diamond_ore"))
        g.offhand = "shield"
        g.armor = {slot: {"item": f"iron_{slot}"}
                   for slot in ("chestplate", "leggings", "boots")}  # no helmet yet
        self.assertNotIn("mine_diamond", candidates(g))


class TechTreeCompleteTest(unittest.TestCase):
    """The auto-stop predicate: diamond tools + every armour piece actually worn + shield.

    Requiring the armour to be *worn* (not merely carried) is the point: `equip_armor` is still
    an eligible action while the set sits in the pack, so stopping then would end the run with
    the gear unequipped.
    """

    @staticmethod
    def _gate() -> dict:
        """The world blocks the gate adds: the lit portal and the bed beside it."""
        return {
            "nether_portal": [BlockHit("nether_portal", "nether_portal", -270, 64, -56, 3.0)],
            # The bed stands `BED_PORTAL_CLEARANCE` from the portal, as `place_bed` lays it: the
            # next cell south of the frame, not against the opening.
            "white_bed": [BlockHit("white_bed", "white_bed", -270, 64, -58, 4.0)],
        }

    def _complete(self, **overrides) -> GameState:
        base = dict(
            tools={"pickaxe": "diamond", "sword": "diamond"},
            # The Nether prep is banked too: a bow with arrows and the deep cobblestone stack
            # (`_nether_prep_steps`), so the completion gate is satisfied.
            inventory={"diamond_sword": 1, "bow": 1, "arrow": 16, "cobblestone": 64},
            armor={slot: {"item": f"diamond_{slot}"}
                   for slot in ("helmet", "chestplate", "leggings", "boots")},
            offhand="shield",
            resources=self._gate(),
        )
        base.update(overrides)
        return gs(**base)

    def test_a_full_kit_completes_the_tree(self) -> None:
        self.assertTrue(tech_tree_complete(self._complete()))

    def test_carried_but_unworn_armour_is_not_done(self) -> None:
        self.assertFalse(tech_tree_complete(self._complete(armor={})))
        # ...even with every piece waiting in the pack.
        g = self._complete(armor={}, inventory={"diamond_sword": 1, "diamond_helmet": 1})
        self.assertFalse(tech_tree_complete(g))

    def test_missing_shield_or_sword_is_not_done(self) -> None:
        self.assertFalse(tech_tree_complete(self._complete(offhand=None)))
        g = self._complete(tools={"pickaxe": "diamond"}, inventory={})
        self.assertFalse(tech_tree_complete(g))

    def test_diamond_pickaxe_is_required(self) -> None:
        g = self._complete(tools={"pickaxe": "iron", "sword": "diamond"})
        self.assertFalse(tech_tree_complete(g))

    def test_the_unlit_portal_is_not_done(self) -> None:
        # The frame built but not lit: the gate is not open, so the run keeps going.
        g = self._complete(resources={"white_bed": self._gate()["white_bed"]})
        self.assertFalse(tech_tree_complete(g))

    def test_a_lost_bow_is_not_done(self) -> None:
        # The prep step drops out when the string to rebuild the bow is out of reach, and an empty
        # step list reads as "prep done" - which ended run32 at its first tick with a goal it had
        # not reached. The bank is judged as carried, so a lost piece sends the run back to work.
        g = self._complete()
        self.assertTrue(tech_tree_complete(g))
        g.inventory = {k: v for k, v in g.inventory.items() if k not in ("bow", "arrow")}
        self.assertFalse(tech_tree_complete(g))

    def test_one_string_short_of_the_bow_is_still_a_lost_bow(self) -> None:
        # Two string and no spider in reach: the bow cannot be rebuilt, so it cannot count as
        # banked - which is the state run32 stopped in.
        g = self._complete(inventory={"diamond_sword": 1, "string": 2, "stick": 3,
                                      "cobblestone": 64})
        self.assertFalse(tech_tree_complete(g))

    def test_a_spider_in_reach_repairs_the_bow(self) -> None:
        # The old spider guard wanted one within 10 m in a clear field, which made a single missing
        # string unreachable and the bow unrepairable for good.
        g = self._complete(inventory={"diamond_sword": 1, "string": 2, "stick": 3,
                                      "cobblestone": 64}, health=20.0)
        g.mobs = [("spider", "hostile", 1, 18.0)]
        self.assertIn("hunt_string", candidates(g))

    def test_a_half_made_arrow_reserve_is_not_done(self) -> None:
        g = self._complete()
        g.inventory = {**g.inventory, "bow": 1, "arrow": 8}
        self.assertFalse(tech_tree_complete(g))

    def test_a_missing_bed_is_not_done(self) -> None:
        g = self._complete(resources={"nether_portal": self._gate()["nether_portal"]})
        self.assertFalse(tech_tree_complete(g))

    def test_a_bed_in_the_gate_is_not_done(self) -> None:
        # A bed within `BED_PORTAL_CLEARANCE` of the lit gate is not the gate bed, even at the
        # base: sleeping pins the respawn point to the bed's own cell, so it would drop the bot
        # into the portal on its next death, and the gate would throw it to the Nether. The run
        # takes the bed up (`take_bed`) and lays a fresh one clear of the gate.
        g = self._complete(
            base=(-270, 64, -57),
            resources={**self._gate(),
                       "white_bed": [BlockHit("white_bed", "white_bed", -269, 64, -56, 1.0)]},
        )
        self.assertFalse(tech_tree_complete(g))

    def test_a_bed_away_from_the_base_is_not_done(self) -> None:
        # A leftover bed from an earlier run satisfies `_bed_placed` but lies off the base, so the
        # run keeps working until `take_bed`/`place_bed` bring it home (live, run19: the run17 bed
        # sat at y=55 inside the base's gold shaft, ten below the base, and read as "done").
        g = self._complete(
            base=(-270, 64, -57),
            resources={**self._gate(),
                       "white_bed": [BlockHit("white_bed", "white_bed", -269, 55, -56, 9.0)]},
        )
        self.assertFalse(tech_tree_complete(g))

    def test_the_deep_cobble_stack_is_required(self) -> None:
        # Portal, bed and the whole kit stand, but the Nether shield stack is not banked.
        g = self._complete(inventory={"diamond_sword": 1, "bow": 1, "arrow": 16,
                                      "cobblestone": 6})
        self.assertFalse(tech_tree_complete(g))

    def test_pending_gold_prep_blocks_completion(self) -> None:
        res = {**self._gate(),
               "gold_ore": [BlockHit("gold_ore", "gold_ore", -200, 20, -100, 40.0)]}
        g = self._complete(resources=res,
                           inventory={"diamond_sword": 1, "bow": 1, "arrow": 16,
                                      "cobblestone": 64, "crafting_table": 1})
        self.assertFalse(tech_tree_complete(g))

    def test_unreachable_prep_does_not_block_completion(self) -> None:
        # No chicken, no gold ore: those prep steps drop out, so the run can still end instead of
        # stalling on an item it can never get. The bow and its arrows are carried, though - the
        # Nether trip is fought with them, so they are no longer "a step that might drop out".
        g = self._complete(inventory={"diamond_sword": 1, "bow": 1, "arrow": 16,
                                      "cobblestone": 64})
        self.assertTrue(tech_tree_complete(g))

    def test_the_food_reserve_is_required(self) -> None:
        # An animal stands in reach, but the deep Nether food reserve is not banked yet - the
        # gate holds the run instead of ending it on a nearly empty pack (live run13 stopped
        # with 4 cooked mutton).
        g = self._complete(mobs=[("cow", "animal", 3, 12.0)], inventory={
            "diamond_sword": 1, "bow": 1, "arrow": 16, "cobblestone": 64,
            "cooked_beef": 8})
        self.assertFalse(tech_tree_complete(g))

    def test_a_banked_food_reserve_completes(self) -> None:
        g = self._complete(mobs=[("cow", "animal", 3, 12.0)], inventory={
            "diamond_sword": 1, "bow": 1, "arrow": 16, "cobblestone": 64,
            "cooked_beef": 64})
        self.assertTrue(tech_tree_complete(g))

    def test_a_lone_zombie_counts_as_a_food_source(self) -> None:
        # Rotten flesh is the reliable local bank, so a zombie alone keeps the gate holding.
        g = self._complete(mobs=[("zombie", "hostile", 1, 10.0)], inventory={
            "diamond_sword": 1, "bow": 1, "arrow": 16, "cobblestone": 64,
            "cooked_beef": 8})
        self.assertFalse(tech_tree_complete(g))

    def test_a_source_beyond_reach_drops_the_reserve_requirement(self) -> None:
        # A mob merely *loaded* (here a cow past the daylight hunt range, and a zombie past melee
        # range) cannot hold the gate open: the hunt could not reach either, so the requirement
        # drops out like an unreachable prep step rather than stalling the run on a distant mob.
        g = self._complete(mobs=[("cow", "animal", 3, 100.0), ("zombie", "hostile", 1, 40.0)],
                           inventory={"diamond_sword": 1, "bow": 1, "arrow": 16,
                                      "cobblestone": 64, "cooked_beef": 8})
        self.assertTrue(tech_tree_complete(g))

    def test_no_food_source_drops_the_reserve_requirement(self) -> None:
        # No animal and no zombie anywhere: the requirement drops out like an unreachable prep
        # step, so a world without food can still let the run end rather than stall.
        g = self._complete(inventory={
            "diamond_sword": 1, "bow": 1, "arrow": 16, "cobblestone": 64,
            "cooked_beef": 2})
        self.assertTrue(tech_tree_complete(g))


class NetherGateTest(unittest.TestCase):
    """The overworld -> Nether gate: a bucket of water makes the obsidian, gravel supplies the
    flint, the flint and steel lights the frame, and a bed is placed beside the lit portal.

    The chain is gated on the whole diamond kit (`_gear_ready`). The obsidian step stands while
    no frame has been raised at the base (`_frame_started_at_base`): the obsidian already placed
    is the latch that stops `mine_obsidian` re-mining after `build_portal` has spent the ten
    blocks, and it keeps working when the pack is short and the igniter is already carried.
    """

    @staticmethod
    def _geared(**inv) -> GameState:
        # Two pickaxes: the gate no longer lights without the spare (`NETHER_PICKAXES` rides on
        # `_expedition_ready`), because a burning gate is an open door the pathfinder walks
        # through like any other block - a bot standing in its own gate crosses whether or not
        # it meant to (live, run53: teleported mid-`cook_food` with one pickaxe, `enter_portal`
        # never run).
        base = {"crafting_table": 1, "diamond_pickaxe": 2}
        base.update(inv)
        g = gs(tools={"pickaxe": "diamond", "sword": "diamond"}, inventory=base)
        g.armor = {slot: {"item": f"diamond_{slot}"}
                   for slot in ("helmet", "chestplate", "leggings", "boots")}
        g.offhand = "shield"
        return g

    @staticmethod
    def _blocks(*names: str) -> dict:
        return {n: [BlockHit(n, n, -270, 64, -56, 3.0)] for n in names}

    @staticmethod
    def _lava(source=(-270, 64, -57), target=(-270, 64, -56), distance=3.0) -> dict:
        """A pour-able pond: a level-0 lava *source* with a flow cell beside it.

        `target` is the cell the water goes into. It stands beside the source at the same
        level, open above and floored with stone -- the shape the server actually accepts a
        pour on, and the only shape that turns the source into obsidian.
        """
        sx, sy, sz = source
        tx, ty, tz = target
        return {"lava": [
            BlockHit("lava", "lava", tx, ty, tz, distance,
                     above="air", below="stone", level=2),
            BlockHit("lava", "lava", sx, sy, sz, distance + 1,
                     above="stone", below="stone", level=0),
        ]}

    @staticmethod
    def _at_home(g: GameState, base: tuple = (-197, -47, -98)) -> GameState:
        """Stand the bot on the home base, so the base-only steps (portal, bed) are eligible."""
        g.base = base
        g.pos = (base[0] + 0.5, base[1] + 1.0, base[2] + 0.5)
        return g

    def _lit_gate_with_bed(self, bed: BlockHit) -> GameState:
        """Stand the geared bot at the base with the portal lit and one bed standing at `bed`."""
        g = self._at_home(self._geared())
        g.resources = {"nether_portal": self._blocks("nether_portal")["nether_portal"],
                       "white_bed": [bed]}
        return g

    def _expedition(self, **inv) -> GameState:
        """A geared bot at the base with the frame standing, the prep banked and the gate bed
        down: `_expedition_ready` holds, so `light_portal` is offered.

        The frame is confirmed by an obsidian hit *inside the frame plane* (the signal
        `build_portal` leaves behind); the gate itself is still unlit, which is exactly the
        state `light_portal` is meant to run from.
        """
        base = {"flint_and_steel": 1, "bow": 1, "arrow": 16, "cobblestone": 64}
        base.update(inv)
        g = self._at_home(self._geared(**base))
        g.resources = {
            "obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)],
            "white_bed": [BlockHit("white_bed", "white_bed", -196, -47, -98, 1.0)],
        }
        return g

    def test_craft_bucket_from_three_iron(self) -> None:
        a = resolve("craft_bucket", self._geared(iron_ingot=3))
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "bucket"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_bucket_waits_for_the_diamond_kit(self) -> None:
        # Iron is scarce; the bucket belongs to the finished overworld, not the iron tier.
        g = gs(tools={"pickaxe": "iron"}, inventory={"crafting_table": 1, "iron_ingot": 3})
        self.assertNotIn("craft_bucket", candidates(g))
        self.assertIsNone(resolve("craft_bucket", g))

    def test_bucket_needs_three_ingots(self) -> None:
        self.assertNotIn("craft_bucket", candidates(self._geared(iron_ingot=2)))

    def test_the_bucket_pulls_three_iron(self) -> None:
        # With the kit worn, the bucket is the reason to quarry iron: three ingots.
        g = self._geared()
        g.resources = self._blocks("iron_ore")
        self.assertIn("mine_iron", candidates(g))
        self.assertEqual(resolve("mine_iron", g).args["count"], 3)

    def test_after_the_bucket_only_one_more_ingot(self) -> None:
        # The bucket is spent, so only the flint and steel's single ingot is left to pull.
        g = self._geared(water_bucket=1)
        g.resources = self._blocks("iron_ore")
        self.assertIn("mine_iron", candidates(g))
        self.assertEqual(resolve("mine_iron", g).args["count"], 1)

    def test_fill_bucket_scoops_water(self) -> None:
        a = resolve("fill_bucket", self._geared(bucket=1))
        self.assertEqual(a.tool, "use_bucket")
        self.assertEqual(a.args, {"action": "fill", "liquid": "water"})

    def test_fill_bucket_needs_an_empty_bucket(self) -> None:
        self.assertIsNone(resolve("fill_bucket", self._geared()))
        self.assertIsNone(resolve("fill_bucket", self._geared(water_bucket=1)))

    def test_fill_bucket_still_offered_while_the_obsidian_is_short(self) -> None:
        # The water is wanted exactly while obsidian is still being made.
        self.assertIn("fill_bucket", candidates(self._geared(bucket=1, obsidian=4)))

    def test_no_fill_or_craft_bucket_once_the_obsidian_is_banked(self) -> None:
        # The bucket serves only `mine_obsidian`. With the ten blocks banked the water is
        # finished, and offering the fill anyway stranded run9: an out-of-reach water column made
        # `fill_bucket` fail, `go_home` answered, and the bot traded the two between the base and
        # the water for 45 iterations while the portal stood ready to build. Neither the fill nor
        # the craft that feeds it may be offered once the obsidian is complete.
        g = self._geared(bucket=1, obsidian=10)
        self.assertNotIn("fill_bucket", candidates(g))
        self.assertNotIn("craft_bucket", candidates(g))

    def test_fill_bucket_stops_once_the_flint_and_steel_is_crafted(self) -> None:
        # The flint and steel is the latch that ends the obsidian epoch (it exists only to light
        # the frame). After it an empty bucket must not pull the bot back to the water -- at a
        # base whose water is out of reach that is the run9 ping-pong, and the only work left is
        # to raise the frame.
        g = self._at_home(self._geared(bucket=1, obsidian=10,
                                       cobblestone=4, flint_and_steel=1))
        self.assertNotIn("fill_bucket", candidates(g))
        self.assertIn("build_portal", candidates(g))

    def test_fill_bucket_returns_with_flint_and_steel_while_the_frame_is_short(self) -> None:
        # A restart can carry the igniter into a world whose frame is not up (a short pack, or a
        # base re-anchored away from the obsidian already placed). The igniter is not the latch
        # any more: while no frame stands at the base the water chain reopens, so the missing
        # obsidian can still be made.
        g = self._geared(bucket=1, obsidian=4, flint_and_steel=1)
        self.assertIn("fill_bucket", candidates(g))

    def test_no_fill_or_mine_once_the_frame_stands(self) -> None:
        # The frame itself is the latch: an obsidian block in the base's plane means the build
        # has begun, so the water and the mine step stand down and `build_portal` resumes with
        # the blocks still carried.
        g = self._at_home(self._geared(bucket=1, obsidian=4,
                                       cobblestone=4, flint_and_steel=1))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        self.assertNotIn("fill_bucket", candidates(g))
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))
        self.assertIn("build_portal", candidates(g))

    def test_mine_obsidian_pours_water_onto_lava(self) -> None:
        # Obsidian is *made*, not dug: with the bucket full and a usable lava cell within
        # pouring reach, the step empties the bucket into that cell.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = self._lava()
        self.assertIn("mine_obsidian", candidates(g))
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "use_bucket")
        self.assertEqual(a.args, {"action": "pour", "liquid": "water",
                                  "x": -270, "y": 64, "z": -56})

    def test_mine_obsidian_walks_to_far_lava_first(self) -> None:
        # Out of pouring reach the step walks to the cell *on top of* the block beside the lava,
        # never into the lava and never to the neighbour's own height: that neighbour is solid
        # rock at a real lava lake, the server refuses to dig it (it would open a route into the
        # liquid), and the pathfinder then stops a cell short and diagonally -- the stance every
        # live `[BUCKET_NO_FACE]` came from. The walk may dig (the lava lake sits far below the
        # surface) but never scaffolds, so the pathfinder cannot bridge over its target.
        g = self._geared(water_bucket=1)
        g.pos = (0.0, 64.0, 0.0)
        g.resources = self._lava()
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual(a.args["y"], 65)
        self.assertIn((a.args["x"], a.args["z"]),
                      [(-269, -56), (-271, -56), (-270, -55)])
        self.assertEqual(a.args["range"], 1)
        self.assertTrue(a.args["dig"])
        self.assertFalse(a.args["scaffold"])

    def test_mine_obsidian_needs_a_source_not_a_flow(self) -> None:
        # Water on flowing lava makes only cobblestone, so with no level-0 source in the
        # listing the pour cannot produce obsidian at all and the step stays out.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = {"lava": [BlockHit("lava", "lava", -270, 64, -56, 3.0,
                                         above="air", below="stone", level=2)]}
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_needs_an_open_pour_cell(self) -> None:
        # The surface pond's only usable cell is capped by stone, so no water can be placed
        # there: the step refuses instead of pouring into a wall.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0, above="stone", below="stone", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
        ]}
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_needs_a_floor_under_the_target(self) -> None:
        # A target cell over lava or air has no solid block to click beside it: that is the
        # live `[BUCKET_NO_FACE]` that repeated for a whole run on the pond.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0, above="air", below="lava", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
        ]}
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_accepts_a_source_under_a_cave_roof(self) -> None:
        # Live, from-zero run: both lava sources in scan reach sat under `cave_air`, and the step
        # required exactly `"air"` above the target, so `mine_obsidian` was never offered at all
        # and the bot wandered the surface with `explore` alone until the iteration limit.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0,
                     above="cave_air", below="stone", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0,
                     above="cave_air", below="stone", level=0),
        ]}
        self.assertIn("mine_obsidian", candidates(g))
        self.assertEqual(resolve("mine_obsidian", g).args,
                         {"action": "pour", "liquid": "water", "x": -270, "y": 64, "z": -56})

    def test_mine_obsidian_needs_solid_ground_not_cave_air(self) -> None:
        # The floor rule reads the same air variants: `cave_air` under the target is not a block
        # to click beside, so this cell stays out.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0,
                     above="air", below="cave_air", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0,
                     above="air", below="stone", level=0),
        ]}
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_skips_a_refused_pour_cell(self) -> None:
        # The loop remembers a cell a pour was refused at; the next tick goes to the next
        # source instead of retrying the stopped-up one forever.
        g = self._geared(water_bucket=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.pour_skip = frozenset({(-270, 64, -56)})
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0, above="air", below="stone", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
            BlockHit("lava", "lava", -260, 64, -56, 10.0, above="air", below="stone", level=2),
            BlockHit("lava", "lava", -260, 64, -57, 10.5, above="stone", below="stone", level=0),
        ]}
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-261, 65, -56))

    def test_mine_obsidian_skips_a_refused_stance(self) -> None:
        # Live, run16: the stance the walk picked sat over a deep lava lake, so the server
        # answered `[NAVIGATION_TARGET_UNSUPPORTED]` ("No ground was found within 64 blocks below
        # it") on 215 ticks. That is a property of the stance cell, so the next tick stands on
        # another neighbour of the same source instead of walking there again.
        g = self._geared(water_bucket=1)
        g.pos = (0.0, 64.0, 0.0)  # out of pouring reach, so a stance is chosen
        g.stance_skip = frozenset({(-269, 65, -56)})
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0, above="air", below="stone", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
        ]}
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-270, 65, -55))

    def test_mine_obsidian_passes_over_a_source_with_every_stance_refused(self) -> None:
        # Every stance of the near source is unusable, so the step moves to the next source --
        # the same move `pour_skip` makes for a refused pour.
        g = self._geared(water_bucket=1)
        g.pos = (0.0, 64.0, 0.0)
        g.stance_skip = frozenset({(-269, 65, -56), (-271, 65, -56), (-270, 65, -55)})
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0, above="air", below="stone", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
            BlockHit("lava", "lava", -260, 64, -56, 10.0, above="air", below="stone", level=2),
            BlockHit("lava", "lava", -260, 64, -57, 10.5, above="stone", below="stone", level=0),
        ]}
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-259, 65, -56))

    def test_a_source_with_every_stance_refused_is_passed_over(self) -> None:
        # No second source and the only one's stances are all unusable: the step stands down
        # instead of walking to a stance the server refuses forever.
        g = self._geared(water_bucket=1)
        g.pos = (0.0, 64.0, 0.0)
        g.stance_skip = frozenset({(-269, 65, -56), (-271, 65, -56), (-270, 65, -55)})
        g.resources = {"lava": [
            BlockHit("lava", "lava", -270, 64, -56, 3.0, above="air", below="stone", level=2),
            BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
        ]}
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_mines_what_the_pour_formed(self) -> None:
        # Obsidian the last pour formed lies at the bot's feet: mine it, even with no water left.
        g = self._geared(obsidian=4)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = self._blocks("obsidian")
        self.assertIn("mine_obsidian", candidates(g))
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "obsidian", "count": 6})

    def test_mine_obsidian_returns_with_flint_and_steel_while_the_frame_is_short(self) -> None:
        # Live, run17: an igniter carried into a world whose frame was never finished left
        # `mine_obsidian` switched off for good -- the obsidian chain keyed on the igniter, not
        # on the frame. With six blocks parked within reach (a partial frame the bot still
        # stands beside) it now mines them back and banks the ten.
        g = self._geared(obsidian=4, flint_and_steel=1)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = self._blocks("obsidian")
        self.assertIn("mine_obsidian", candidates(g))
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "obsidian", "count": 6})

    def test_mine_obsidian_still_needs_a_frame_or_a_way_to_make_one(self) -> None:
        # The igniter only reopens the chain; it does not conjure a target. With no lava to pour
        # on and no obsidian within reach the step stays out, exactly as for a fresh pack.
        g = self._geared(obsidian=4, flint_and_steel=1)
        g.pos = (0.0, 64.0, 0.0)
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_ignores_deep_natural_obsidian(self) -> None:
        # The natural deposit sits tens of blocks down under the same lava caves. With the bucket
        # full but no lava in range, the step refuses rather than tunnelling down to it (the live
        # trap left the bot swimming in its own flooded shaft).
        g = self._geared(water_bucket=1)
        g.pos = (0.0, 64.0, 0.0)
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", 0, -55, 0, 119.0)]}
        self.assertNotIn("mine_obsidian", candidates(g))
        self.assertIsNone(resolve("mine_obsidian", g))

    def test_mine_obsidian_needs_water_or_obsidian_in_reach(self) -> None:
        # Neither a full bucket nor obsidian at hand: nothing for this step to do.
        self.assertNotIn("mine_obsidian", candidates(self._geared()))

    def test_mine_obsidian_needs_a_diamond_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "iron"}, inventory={"water_bucket": 1})
        self.assertNotIn("mine_obsidian", candidates(g))

    def test_flint_from_gravel(self) -> None:
        g = self._geared()
        g.resources = self._blocks("gravel")
        self.assertIn("get_flint", candidates(g))
        a = resolve("get_flint", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args["block_name"], "gravel")

    def test_craft_flint_and_steel_is_two_by_two(self) -> None:
        # Flint and steel is a 2x2 recipe: it must NOT ask for the temporary workstation.
        g = self._geared(flint=1, iron_ingot=1, obsidian=10)
        self.assertIn("craft_flint_and_steel", candidates(g))
        a = resolve("craft_flint_and_steel", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args, {"items": [{"item_name": "flint_and_steel"}]})

    def test_flint_and_steel_waits_for_the_obsidian(self) -> None:
        # The igniter belongs to a banked pack: it is crafted once the ten blocks exist, just
        # before the build, rather than spending the iron early with nothing yet to light.
        g = self._geared(flint=1, iron_ingot=1)
        self.assertNotIn("craft_flint_and_steel", candidates(g))
        self.assertIsNone(resolve("craft_flint_and_steel", g))

    def test_build_portal_raises_the_frame_beside_the_bot(self) -> None:
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        self.assertIn("build_portal", candidates(g))
        a = resolve("build_portal", g)
        self.assertEqual(a.tool, "build_structure")
        self.assertEqual(a.args["portal_frame"], {"x": -195, "y": -46, "z": -98, "axis": "x"})
        self.assertTrue(a.args["remove_wrong_blocks"])

    def test_build_portal_waits_until_the_bot_is_at_the_base(self) -> None:
        g = self._geared(flint_and_steel=1, obsidian=10, cobblestone=4)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)  # geared but far from home
        self.assertNotIn("build_portal", candidates(g))

    def test_build_portal_needs_the_ten_blocks(self) -> None:
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=9, cobblestone=4))
        self.assertNotIn("build_portal", candidates(g))

    def test_build_portal_resumes_a_frame_it_already_started(self) -> None:
        # A partial build leaves exactly the cells it still needs in the pack; the obsidian in
        # the frame plane is the signal to re-send the same structure and finish it.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=7, cobblestone=4))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        self.assertIn("build_portal", candidates(g))
        a = resolve("build_portal", g)
        self.assertEqual(a.args["portal_frame"], {"x": -195, "y": -46, "z": -98, "axis": "x"})

    def test_a_finished_frame_is_not_rebuilt(self) -> None:
        # A build the server answers `ok` to while it spends nothing has placed every cell it
        # can: the frame stands complete. The "resume a partial frame" clause would otherwise
        # keep the step in the offer for as long as a leftover block sits in the pack, and the
        # no-op success resets the stuck guard, so nothing else would ever drop it (live, run25:
        # 163 identical `build_portal` ticks on a finished frame). `gs.frame_complete` is the latch
        # the loop sets from that answer (`Agent._note_build`).
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=9, cobblestone=22))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        self.assertIn("build_portal", candidates(g))  # a partial frame still resumes
        g.frame_complete = True
        self.assertNotIn("build_portal", candidates(g))
        self.assertIsNone(resolve("build_portal", g))

    def test_build_portal_stays_put_without_ten_blocks_or_a_frame(self) -> None:
        # Seven obsidian and no frame started yet: the bot must bank the rest first.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=7, cobblestone=4))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -260, 64, -56, 3.0)]}
        self.assertNotIn("build_portal", candidates(g))
        self.assertIsNone(resolve("build_portal", g))

    def test_stray_block_in_the_opening_is_dug_before_the_build(self) -> None:
        # A block left above the support layer (a bed the old sleep laid "beside the bot", a
        # leftover block) must go first: the frame would be built around it and `light_portal`
        # would answer `[PORTAL_NO_FRAME]` forever. The loop's opening read fills
        # `gs.portal_blocked`, and while it is non-empty the dig stands in front of the build.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        g.portal_blocked = [(-195, -46, -98), (-195, -45, -98)]
        self.assertIn("clear_portal", candidates(g))
        self.assertNotIn("build_portal", candidates(g))
        a = resolve("clear_portal", g)
        self.assertEqual(a.tool, "build_structure")
        self.assertEqual(a.args["blocks"], [
            {"x": -195, "y": -46, "z": -98, "block_name": "air"},
            {"x": -195, "y": -45, "z": -98, "block_name": "air"},
        ])

    def test_clear_portal_digs_at_most_four_cells_a_call(self) -> None:
        # `build_structure` runs one route search per cell and a long list watchdog-kills the
        # server; the opening is re-read every tick, so a bigger pile is dug over several ticks.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        g.portal_blocked = [(-195, -46, -98), (-195, -45, -98), (-195, -44, -98),
                            (-194, -46, -98), (-194, -45, -98), (-194, -44, -98)]
        a = resolve("clear_portal", g)
        self.assertEqual(len(a.args["blocks"]), 4)
        self.assertTrue(all(c["block_name"] == "air" for c in a.args["blocks"]))

    def test_clear_portal_stays_put_without_a_blocked_opening(self) -> None:
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        self.assertIsNone(resolve("clear_portal", g))
        self.assertIn("build_portal", candidates(g))
        self.assertNotIn("clear_portal", candidates(g))

    def test_clear_portal_waits_until_the_bot_is_at_the_base(self) -> None:
        g = self._geared(flint_and_steel=1, obsidian=10, cobblestone=4)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)  # geared but far from home
        g.portal_blocked = [(-195, -46, -98)]
        self.assertIsNone(resolve("clear_portal", g))

    def test_a_container_in_a_frame_cell_is_cleared_explicitly(self) -> None:
        # A container the build cannot dig in place (the movement policy refuses an unasked break)
        # is refused on every `build_portal`, so the frame never closes (live, run27: run26's
        # furnace stood on the new base's frame corner at the frame's west-bottom cell). The step
        # is a `collect_block` at that exact cell, whose requested break the policy does allow.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        g.refused_frame = ((-196, -47, -98),)
        g.frame_names = {(-196, -47, -98): "furnace"}
        self.assertIn("clear_frame", candidates(g))
        self.assertNotIn("build_portal", candidates(g))
        a = resolve("clear_frame", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args["block_name"], "furnace")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-196, -47, -98))

    def test_clear_frame_is_offered_with_the_obsidian_already_spent(self) -> None:
        # The frame is partly built and the bank is spent; clearing the cell needs no obsidian and
        # no cobble, so it still stands -- the bot clears it while it refills the pack (live,
        # run27: the build had spent the ten blocks when run26's furnace was refused).
        g = self._at_home(self._geared(flint_and_steel=1, cobblestone=4))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        g.refused_frame = ((-196, -47, -98),)
        g.frame_names = {(-196, -47, -98): "furnace"}
        self.assertIn("clear_frame", candidates(g))
        self.assertNotIn("build_portal", candidates(g))

    def test_clear_frame_waits_until_the_frame_read_names_the_cell(self) -> None:
        # `collect_block` matches on the block name, so a refused cell the frame read has not
        # named yet waits for the next read instead of being dug blind; the build stays offered.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        g.refused_frame = ((-196, -47, -98),)
        self.assertNotIn("clear_frame", candidates(g))
        self.assertIn("build_portal", candidates(g))
        self.assertIsNone(resolve("clear_frame", g))

    def test_the_frame_block_and_fluids_are_never_dug(self) -> None:
        # The refuse list is a set of cells, not of block kinds: the frame's own obsidian (already
        # correct) and a fluid (a `collect_block` cannot safely break it) are passed over.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=10, cobblestone=4))
        g.refused_frame = ((-196, -47, -98),)
        for name in ("nether_portal", "lava", "water"):
            g.frame_names = {(-196, -47, -98): name}
            self.assertNotIn("clear_frame", candidates(g))
            self.assertIsNone(resolve("clear_frame", g))

    def test_clear_frame_waits_until_the_bot_is_at_the_base(self) -> None:
        g = self._geared(flint_and_steel=1, obsidian=10, cobblestone=4)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)  # geared but far from home
        g.refused_frame = ((-196, -47, -98),)
        g.frame_names = {(-196, -47, -98): "furnace"}
        self.assertIsNone(resolve("clear_frame", g))

    def test_light_portal_strikes_the_nearest_frame_block(self) -> None:
        # The frame stands and the expedition is ready to go through; the nearest obsidian block
        # *inside the frame plane* is the one struck.
        g = self._expedition()
        self.assertIn("light_portal", candidates(g))
        a = resolve("light_portal", g)
        self.assertEqual(a.tool, "activate_portal")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-195, -47, -98))

    def test_light_portal_waits_until_a_bed_is_at_hand(self) -> None:
        # The frame stands and the flint and steel is carried, but no bed is at hand: lighting
        # folds the bed into the "ready" bundle, so it waits. This is the run22 bug - had the
        # gate been lit here it would have burned through the whole prep window beside the base.
        g = self._expedition()
        g.resources.pop("white_bed")
        self.assertNotIn("light_portal", candidates(g))
        self.assertIsNone(resolve("light_portal", g))

    def test_light_portal_waits_for_the_prep(self) -> None:
        # The bed is down but the deep cobble stack is short: still not ready to go through.
        g = self._expedition(cobblestone=6)
        self.assertNotIn("light_portal", candidates(g))
        self.assertIsNone(resolve("light_portal", g))

    def test_light_portal_waits_for_night_with_only_a_carried_bed(self) -> None:
        # A bed merely *carried* cannot be laid until the gate burns (a bed laid beside an unlit
        # frame can land in the open opening, which `clear_portal` cannot dig). So lighting with a
        # carried bed waits for night, when `place_bed` can run the very next tick; by day the
        # burning gate would sit beside the working bot.
        def carried(phase: str) -> GameState:
            g = self._at_home(self._geared(white_bed=1, flint_and_steel=1,
                                           bow=1, arrow=16, cobblestone=64))
            g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
            g.phase = phase
            return g

        day = carried("day")
        self.assertNotIn("light_portal", candidates(day))
        self.assertIsNone(resolve("light_portal", day))
        night = carried("night")
        self.assertIn("light_portal", candidates(night))
        self.assertEqual(resolve("light_portal", night).tool, "activate_portal")

    def test_light_portal_ignores_the_hour_with_a_bed_at_the_base(self) -> None:
        # A bed already standing at the base needs no night wait: lighting stops the run at once.
        g = self._expedition()
        g.phase = "day"
        self.assertIn("light_portal", candidates(g))

    def test_light_portal_ignores_obsidian_off_the_frame(self) -> None:
        # Natural obsidian near the base is not our frame: without an obsidian block in the
        # frame plane the light step must not fire (it would answer `[PORTAL_NO_FRAME]`).
        g = self._at_home(self._geared(flint_and_steel=1))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -260, 64, -56, 3.0)]}
        self.assertNotIn("light_portal", candidates(g))
        self.assertIsNone(resolve("light_portal", g))

    def test_light_portal_waits_until_the_frame_is_built(self) -> None:
        # Still carrying the ten obsidian: nothing has been placed yet.
        g = self._geared(flint_and_steel=1, obsidian=10)
        g.resources = self._blocks("obsidian")
        self.assertNotIn("light_portal", candidates(g))
        self.assertIsNone(resolve("light_portal", g))

    def test_light_portal_lights_a_finished_frame_with_the_obsidian_re_banked(self) -> None:
        # Live, run28: the build finished the frame, the loop then re-mined a fresh ten (the
        # nearby obsidian scan had dropped the frame's own hits) and `obsidian < PORTAL_OBSIDIAN`
        # never held again, so the finished gate could not be lit. The exact plane read proves
        # the frame is complete, so the pack's obsidian count no longer decides it.
        g = self._expedition(obsidian=10)
        g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)}
        self.assertIn("light_portal", candidates(g))
        self.assertEqual(resolve("light_portal", g).tool, "activate_portal")

    def test_light_portal_waits_when_a_re_banked_pack_has_only_a_partial_frame(self) -> None:
        # The count alone is not enough: a re-banked ten with one frame cell still missing must
        # not light, or the server answers `[PORTAL_NO_FRAME]` on every tick.
        g = self._expedition(obsidian=10)
        g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)[:-1]}
        self.assertNotIn("light_portal", candidates(g))
        self.assertIsNone(resolve("light_portal", g))

    def test_the_frame_plane_read_stops_the_obsidian_re_mine(self) -> None:
        # `gs.resources` keeps only the four nearest obsidian hits, so a deposit the bot is
        # walking to can crowd the frame's own obsidian out of it. The exact plane read still
        # sees the frame, and the obsidian chain stands down instead of re-mining a fresh ten
        # (live, run28: that re-banked ten then blocked `light_portal` forever).
        def state(frame: bool) -> GameState:
            g = self._at_home(self._geared(water_bucket=1, obsidian=1, flint_and_steel=1))
            g.resources = self._lava()
            g.resources["obsidian"] = [BlockHit("obsidian", "obsidian", -240, 20, -200, 2.0)]
            if frame:
                g.frame_names = {cell: "obsidian"
                                 for cell in frame_obsidian_cells(g.base)}
            return g
        self.assertIn("mine_obsidian", candidates(state(frame=False)))
        self.assertNotIn("mine_obsidian", candidates(state(frame=True)))

    def test_no_gate_actions_once_the_portal_is_lit(self) -> None:
        g = self._geared(flint_and_steel=1)
        g.resources = self._blocks("obsidian", "nether_portal")
        self.assertNotIn("light_portal", candidates(g))
        self.assertNotIn("mine_obsidian", candidates(g))

    def test_place_bed_beside_the_lit_portal(self) -> None:
        g = self._at_home(self._geared(white_bed=1))
        g.resources = self._blocks("nether_portal")
        g.phase = "night"
        self.assertIn("place_bed", candidates(g))
        a = resolve("place_bed", g)
        self.assertEqual(a.tool, "sleep")
        self.assertEqual(a.args, {})

    def test_place_bed_waits_for_night_to_set_the_spawn(self) -> None:
        # `sleep` is what actually sets the respawn point, and a bed can only be slept in at
        # night; placing it in daylight would leave the pause point unset.
        g = self._at_home(self._geared(white_bed=1))
        g.resources = self._blocks("nether_portal")
        g.phase = "day"
        self.assertNotIn("place_bed", candidates(g))

    def test_place_bed_waits_for_the_lit_gate(self) -> None:
        # The frame stands but the gate is not lit: a bed laid beside the open 2x3 opening can
        # land in it, and `clear_portal` cannot dig a bed out (the movement policy protects it),
        # so `light_portal` would answer `[PORTAL_NO_FRAME]` forever. The bed waits for the lit
        # gate, whose opening blocks are what keep the bed on the ground beside the base.
        g = self._at_home(self._geared(white_bed=1))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        g.phase = "night"
        self.assertNotIn("place_bed", candidates(g))
        self.assertIsNone(resolve("place_bed", g))

    def test_the_gate_lights_before_the_bed_goes_down(self) -> None:
        # A carried bed, the prep banked and night: the *gate* is offered, not the bed. Once it
        # lights the opening is solid, so `place_bed` lays the bed safely the very next tick and
        # the run stops - the burning gate is never left standing while the bot works.
        g = self._at_home(self._geared(white_bed=1, flint_and_steel=1, bow=1, arrow=16,
                                       cobblestone=64))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        g.phase = "night"
        self.assertIn("light_portal", candidates(g))
        self.assertNotIn("place_bed", candidates(g))

    def test_go_home_draws_the_bot_home_once_the_trip_is_stocked(self) -> None:
        # The trip is stocked but the gate is unlit, and `light_portal` is `_at_base`-gated, so a
        # bot that finished the prep away from the base (a gold shaft under it, a feather hunt
        # nearby) is drawn home rather than drifting until dusk with nothing left to do.
        g = self._expedition()
        g.pos = (-230.5, 40.0, -120.5)
        self.assertIn("go_home", candidates(g))

    def test_go_home_is_not_drawn_before_the_frame_stands(self) -> None:
        # `_all_nether_prep_steps` is empty before the prep window opens (the frame standing), so
        # without the frame clause in `_expedition_prepared` a geared bot with a bed and food
        # would read as fully stocked and be hauled home for a lighting that cannot happen yet.
        g = self._at_home(self._geared(white_bed=1))
        g.pos = (-230.5, 40.0, -120.5)
        self.assertNotIn("go_home", candidates(g))

    def test_place_bed_is_not_repeated(self) -> None:
        # A bed already standing at the base is not laid a second time.
        g = self._lit_gate_with_bed(BlockHit("white_bed", "white_bed", -196, -47, -98, 1.0))
        g.inventory["white_bed"] = 1
        g.phase = "night"
        self.assertNotIn("place_bed", candidates(g))

    def test_place_bed_lays_from_the_cell_the_bot_already_stands_on(self) -> None:
        # live, run29: the lit gate put `_bed_stance` one cell north of the bot. `navigate` settles
        # inside its arrival tolerance, so the bot never crossed that cell boundary and 17
        # identical `navigate` ticks left the carried bed unlaid, with the run unable to finish.
        # The bot's own cell clears the gate just as well, so the bed goes down from where it is.
        g = self._at_home(self._geared(white_bed=1), base=(226, 63, 222))
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  x, y, 222, 3.0)
                                         for x in (228, 229) for y in (64, 65, 66)]}
        g.phase = "night"
        g.pos = (225.62, 64.0, 221.48)
        self.assertIn("place_bed", candidates(g))
        a = resolve("place_bed", g)
        self.assertEqual(a.tool, "sleep")
        self.assertEqual(a.args, {})

    def test_place_bed_still_walks_off_a_cell_the_gate_reaches(self) -> None:
        # The bot's own cell is *not* clear of the gate - a bed laid from it would sit within
        # `BED_PORTAL_CLEARANCE` and respawn the bot into the portal - so the walk to a stance
        # that is clear is still what `place_bed` orders.
        g = self._at_home(self._geared(white_bed=1), base=(226, 63, 222))
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  x, y, 222, 3.0)
                                         for x in (228, 229) for y in (64, 65, 66)]}
        g.phase = "night"
        g.pos = (227.5, 64.0, 221.5)  # one cell east, next to the opening
        self.assertIn("place_bed", candidates(g))
        a = resolve("place_bed", g)
        self.assertEqual(a.tool, "navigate")
        self.assertNotEqual((a.args["x"], a.args["y"], a.args["z"]), (227, 64, 221))

    def test_place_bed_lays_the_carried_bed_at_a_base_without_one(self) -> None:
        # A bed standing *away* from the base is no gate bed: the carried bed goes down at the
        # base even while the leftover stands (the leftover is `take_bed`'s job, and the bot
        # carries a bed, so it cannot take one up just now).
        g = self._lit_gate_with_bed(BlockHit("white_bed", "white_bed", -270, 64, -56, 3.0))
        g.inventory["white_bed"] = 1
        g.phase = "night"
        self.assertIn("place_bed", candidates(g))

    def test_sleep_is_held_off_the_base_until_the_portal_is_lit(self) -> None:
        # The server lays the bed "beside the bot", and at the base that spot is the portal's
        # own interior, which the later frame build cannot clear (`remove_wrong_blocks` answers
        # "movement policy prohibits breaking this block" for a bed). The base bed is
        # `place_bed`'s job and waits for the lit portal; the generic night sleep must not
        # pre-empt it. Live, the bed fell into the opening and `light_portal` answered
        # `[PORTAL_NO_FRAME]` on every tick.
        g = self._at_home(self._geared(white_bed=1))
        g.phase = "night"
        self.assertNotIn("sleep", candidates(g))
        self.assertIsNone(resolve("sleep", g))

    def test_sleep_at_the_base_waits_for_the_bed_to_stand_there(self) -> None:
        # With the portal lit the gate bed must *stand* at the base: a carried bed is `place_bed`'s
        # job first (the server lays the bed beside the bot, so the respawn point is set at the
        # gate), and only then does `sleep` use it. A bed merely *loaded* used to send the bot to
        # sleep in whatever bed was nearest (live, run19b it25: nine blocks below the base).
        g = self._at_home(self._geared(white_bed=1))
        g.phase = "night"
        g.resources = self._blocks("nether_portal")
        self.assertNotIn("sleep", candidates(g))
        self.assertIn("place_bed", candidates(g))
        self.assertEqual(resolve("place_bed", g).tool, "sleep")

        g.resources = {**g.resources,
                       "white_bed": [BlockHit("white_bed", "white_bed", -198, -46, -98, 1.0)]}
        self.assertIn("sleep", candidates(g))
        self.assertEqual(resolve("sleep", g).tool, "sleep")

    def test_sleep_away_from_the_base_is_unaffected(self) -> None:
        # Away from the base a night bed is only a respawn convenience, never the portal.
        g = self._geared(white_bed=1)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        g.phase = "night"
        self.assertIn("sleep", candidates(g))

    def test_sleep_waits_at_the_base_once_the_portal_is_lit(self) -> None:
        # Post-portal the gate bed belongs at the base: sleeping off-base would lay a carried
        # bed wherever the bot happened to be and pin the respawn point away from the gate -
        # exactly the leftover `take_bed` then had to fetch back.
        g = self._geared(white_bed=1)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        g.phase = "night"
        g.resources = self._blocks("nether_portal")
        self.assertNotIn("sleep", candidates(g))
        self.assertIsNone(resolve("sleep", g))

    def test_take_bed_when_it_stands_away_from_the_base(self) -> None:
        # A bed left below the base is a leftover from an earlier base: it is taken up so the
        # base can lay it beside the portal. Live, run18: the run17 bed sat at y=55 inside the
        # shaft the surface base dug for gold, and `mine_gold` kept stumbling on its half-block
        # top (feet 55.56, no settled arrival) instead of reaching the ore.
        g = self._lit_gate_with_bed(BlockHit("white_bed", "white_bed", -196, -57, -98, 11.0))
        self.assertIn("take_bed", candidates(g))
        a = resolve("take_bed", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "white_bed", "count": 1})

    def test_take_bed_is_not_offered_when_the_bed_stands_at_the_base(self) -> None:
        g = self._lit_gate_with_bed(BlockHit("white_bed", "white_bed", -196, -46, -98, 2.0))
        self.assertNotIn("take_bed", candidates(g))
        self.assertIsNone(resolve("take_bed", g))

    def test_take_bed_leaves_a_bed_laid_at_the_base_alone(self) -> None:
        # `_at_base` allows the bot's feet `BASE_RADIUS + 1` above the anchor, and the bed
        # `place_bed` lays sits where the bot stood. A tighter box would fetch it straight back
        # and the two steps would trade the bed every tick.
        g = self._lit_gate_with_bed(BlockHit("white_bed", "white_bed", -197, -42, -98, 3.0))
        self.assertNotIn("take_bed", candidates(g))

    def test_take_bed_waits_until_the_frame_stands(self) -> None:
        g = self._geared()
        g.base = (-197, -47, -98)
        g.pos = (-196.5, -46.0, -98.5)
        g.resources = {"white_bed": [BlockHit("white_bed", "white_bed", -196, -57, -98, 11.0)]}
        self.assertNotIn("take_bed", candidates(g))
        self.assertIsNone(resolve("take_bed", g))

    def test_take_bed_waits_for_a_carried_bed_to_be_laid(self) -> None:
        # A bed already carried belongs at the base through `place_bed`; the standing leftover
        # is not fetched on top of it, which would leave two beds and re-offer every tick.
        g = self._lit_gate_with_bed(BlockHit("white_bed", "white_bed", -196, -57, -98, 11.0))
        g.inventory["white_bed"] = 1
        self.assertNotIn("take_bed", candidates(g))
        self.assertIsNone(resolve("take_bed", g))


class HomeBaseTest(unittest.TestCase):
    """The surface base: one anchor for the bed, the portal and the chest.

    `establish_base` picks the open ground overhead, `go_home` walks the bot back whenever base
    work is waiting, and a chest at the base stores surplus blocks before the pack ever reaches
    the server's `[INVENTORY_FULL]` refusal (`drop_junk` is only the fallback with no chest).
    """

    @staticmethod
    def _geared(**inv) -> GameState:
        base = {"crafting_table": 1}
        base.update(inv)
        g = gs(tools={"pickaxe": "diamond", "sword": "diamond"}, inventory=base)
        g.armor = {slot: {"item": f"diamond_{slot}"}
                   for slot in ("helmet", "chestplate", "leggings", "boots")}
        g.offhand = "shield"
        return g

    @staticmethod
    def _home(g: GameState, base: tuple = (-197, -47, -98)) -> GameState:
        g.base = base
        g.pos = (base[0] + 0.5, base[1] + 1.0, base[2] + 0.5)
        return g

    @staticmethod
    def _on_surface(g: GameState) -> GameState:
        """Mark the bot as standing on open surface ground, so `establish_base` is offered.

        A real run learns this cell from `surface_up` (a height-less `navigate`); the tests stand
        the bot there directly and record it, which is what `_at_surface` reads.
        """
        x, y, z = g.pos
        g.surface = (math.floor(x), math.floor(y), math.floor(z))
        return g

    def test_establish_base_walks_to_the_surface_ground(self) -> None:
        g = self._geared()
        g.pos = (-197.44, 64.0, -97.64)
        self._on_surface(g)
        g.resources = {"grass_block": [BlockHit("grass_block", "grass_block",
                                                -196, 70, -98, 5.0, above="air", below="dirt")]}
        self.assertIn("establish_base", candidates(g))
        a = resolve("establish_base", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-196, 71, -98))
        self.assertTrue(a.args["dig"] and a.args["scaffold"])

    def test_establish_base_needs_a_surface_target(self) -> None:
        g = self._geared()
        g.pos = (-197.44, 64.0, -97.64)
        self._on_surface(g)
        g.resources = {}
        self.assertIn("establish_base", candidates(g))
        self.assertIsNone(resolve("establish_base", g))

    def test_surface_up_is_offered_until_the_surface_is_known(self) -> None:
        # No surface learned yet: the run climbs out before it may anchor.
        g = self._geared()
        g.pos = (-197.44, 64.0, -97.64)
        self.assertIn("surface_up", candidates(g))
        self.assertNotIn("establish_base", candidates(g))

    def test_surface_up_is_offered_on_a_deep_cave_floor(self) -> None:
        # The base belongs under open sky. A mined tunnel's `air` reads the same as sky, so the
        # only test is how far below the probed surface the bot stands (live, run9: base y=-16
        # under a y=73 surface).
        g = self._geared()
        g.pos = (-47.5, -15.0, 4.6)
        g.surface = (-47, 74, 4)
        self.assertIn("surface_up", candidates(g))
        self.assertNotIn("establish_base", candidates(g))

    def test_no_surface_anchor_under_a_different_column(self) -> None:
        # Live, run10: the surface cell is learned for the probed column. A bot that mined its way
        # under a *different* column must not read as "at the surface" just because that other
        # column's height is close -- that fired `establish_base` on a cave floor and the walk to
        # it wedged the bot in a flooded shaft.
        g = self._geared()
        g.pos = (-56.45, 54.0, -0.5)  # cave floor, column (-57, -1), five below the cell below
        g.surface = (-68, 59, 3)      # a y=59 surface learned in another column
        self.assertIn("surface_up", candidates(g))
        self.assertNotIn("establish_base", candidates(g))

    def test_no_surface_anchor_five_below_its_own_column_surface(self) -> None:
        # The same column, but the feet rest on a cave floor five blocks under the learned surface.
        g = self._geared()
        g.pos = (-67.49, 54.0, 3.33)  # column (-68, 3)
        g.surface = (-68, 59, 3)
        self.assertIn("surface_up", candidates(g))
        self.assertNotIn("establish_base", candidates(g))

    def test_one_block_dip_still_counts_as_the_surface(self) -> None:
        # The margin that stays: a one-block scoop at the column's own surface still opens to sky.
        g = self._geared()
        g.pos = (-197.44, 63.0, -97.64)  # one below the learned y=64
        g.surface = (-198, 64, -98)
        self.assertIn("establish_base", candidates(g))
        self.assertNotIn("surface_up", candidates(g))

    def test_surface_up_probes_a_column_before_it_climbs(self) -> None:
        # First the height-less probe (the server names the surface, or walks the bot up when the
        # column is unambiguous) ...
        g = self._geared()
        g.pos = (-47.5, -15.0, 4.6)
        a = resolve("surface_up", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["z"]), (-48, 4))
        self.assertNotIn("y", a.args)
        # ... then the climb itself, carrying the height the probe named.
        g.surface = (-48, 74, 4)
        a = resolve("surface_up", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-48, 74, 4))
        self.assertTrue(a.args["dig"] and a.args["scaffold"])

    def test_surface_up_re_probes_after_a_move_to_a_new_column(self) -> None:
        # The surface of a new column can be tens of blocks higher or lower, so a stale height is
        # never reused: the probe runs again.
        g = self._geared()
        g.pos = (-20.5, -12.0, 30.5)
        g.surface = (-48, 74, 4)
        a = resolve("surface_up", g)
        self.assertEqual((a.args["x"], a.args["z"]), (-21, 30))
        self.assertNotIn("y", a.args)

    def test_no_base_work_once_the_base_is_set(self) -> None:
        self.assertNotIn("establish_base", candidates(self._home(self._geared())))

    def test_go_home_walks_back_when_base_work_waits(self) -> None:
        g = self._geared(flint_and_steel=1, obsidian=10, cobblestone=4)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        self.assertIn("go_home", candidates(g))
        a = resolve("go_home", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -98))

    def test_go_home_not_offered_without_base_work(self) -> None:
        g = self._geared()
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        self.assertNotIn("go_home", candidates(g))
        self.assertIsNone(resolve("go_home", g))

    def test_go_home_to_a_base_re_adopted_from_a_portal(self) -> None:
        # A restart re-anchors the base on the lit portal it left standing
        # (`graph._adopt_portal_base`) and walks back to it, even with no other base work pending:
        # otherwise the run can never reach the portal its end state needs.
        g = self._geared()
        g.base = (-197, -47, -98)
        g.pos = (-100.5, 64.0, -60.5)
        g.base_adopted = True
        self.assertIn("go_home", candidates(g))
        a = resolve("go_home", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -98))
        # Once the bot stands at the base, `_at_base` takes over and this offering stops.
        g.pos = (-196.5, -46.0, -98.5)
        self.assertNotIn("go_home", candidates(g))

    def test_craft_chest_from_eight_planks(self) -> None:
        g = self._geared(oak_planks=8)
        g.base = (-197, -47, -98)
        self.assertIn("craft_chest", candidates(g))
        a = resolve("craft_chest", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "chest"}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_craft_chest_needs_eight_planks(self) -> None:
        g = self._geared(oak_planks=7)
        g.base = (-197, -47, -98)
        self.assertNotIn("craft_chest", candidates(g))
        self.assertIsNone(resolve("craft_chest", g))

    def test_place_chest_at_the_base(self) -> None:
        g = self._home(self._geared(chest=1))
        self.assertIn("place_chest", candidates(g))
        a = resolve("place_chest", g)
        self.assertEqual(a.tool, "place_block")
        self.assertEqual(a.args["block_name"], "chest")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -97))

    def test_the_chest_chain_is_dropped_when_the_ground_cannot_hold_it(self) -> None:
        # `[PLACE_NO_SUPPORT]` says no solid block touches the cell beside the base, so the
        # chest cannot stand there (live: the run looped on it and never reached the portal).
        # Once the loop knows, neither half of the chain is offered again, so the portal gets
        # its turn and the relief falls back to `drop_junk`.
        g = self._home(self._geared(chest=1))
        g.chest_blocked = True
        self.assertNotIn("place_chest", candidates(g))
        self.assertIsNone(resolve("place_chest", g))
        g2 = self._home(self._geared(oak_planks=8))
        g2.chest_blocked = True
        self.assertNotIn("craft_chest", candidates(g2))
        self.assertIsNone(resolve("craft_chest", g2))

    def test_a_blocked_chest_does_not_justify_going_home(self) -> None:
        g = self._geared(chest=1)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        g.chest_blocked = True
        self.assertNotIn("go_home", candidates(g))
        self.assertIsNone(resolve("go_home", g))

    def test_stow_surplus_deposits_into_the_base_chest(self) -> None:
        g = self._home(self._geared(tuff=64))
        g.chest_pos = (-197, -46, -97)
        g.slots_used = 34
        self.assertIn("stow_surplus", candidates(g))
        a = resolve("stow_surplus", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "deposit")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -97))
        self.assertEqual(a.args["items"], [{"item_name": "tuff", "count": 64}])

    def test_stow_surplus_digs_the_block_off_the_chest_lid(self) -> None:
        # Live, run26: the base sat two blocks under the surface, a cobblestone came to rest
        # directly on the chest, and `use_container` answered `[CHEST_BLOCKED]` on every tick, so
        # the full pack could never drain. The remembered cell above the chest is dug first; the
        # deposit runs as soon as the lid is clear.
        g = self._home(self._geared(tuff=64))
        g.chest_pos = (-197, -46, -97)
        g.chest_above = (-197, -45, -97)
        g.slots_used = 34
        a = resolve("stow_surplus", g)
        self.assertEqual(a.tool, "build_structure")
        self.assertEqual(a.args["blocks"],
                         [{"x": -197, "y": -45, "z": -97, "block_name": "air"}])

    def test_inventory_relief_digs_the_lid_before_depositing(self) -> None:
        g = self._home(self._geared(tuff=64))
        g.chest_pos = (-197, -46, -97)
        g.chest_above = (-197, -45, -97)
        g.slots_used = 36
        self.assertEqual(inventory_relief(g).tool, "build_structure")

    def test_drop_junk_falls_back_without_a_chest(self) -> None:
        g = self._home(self._geared(tuff=64))
        g.slots_used = 34
        self.assertIn("drop_junk", candidates(g))
        self.assertNotIn("stow_surplus", candidates(g))
        a = resolve("drop_junk", g)
        self.assertEqual(a.tool, "drop_item")
        self.assertEqual(a.args["items"], [{"item_name": "tuff"}])

    def test_a_full_pack_of_spare_gear_still_sheds(self) -> None:
        # Live, run28: the pack was full of the kit the chain upgraded through (an iron sword,
        # wooden/stone/iron picks and iron armour, with the diamond set worn). `_junk_carried`
        # saw none of it, so `_sheddable` was False and every slot-needing action -- `craft_table`,
        # every mob drop -- answered `[INVENTORY_FULL]` on end. Spare gear has to be shed too.
        g = self._home(self._geared(iron_sword=1, iron_pickaxe=1, stone_pickaxe=1,
                                    wooden_pickaxe=1, iron_helmet=1, iron_chestplate=1,
                                    iron_leggings=1, iron_boots=1))
        g.slots_used = 36
        self.assertIn("drop_junk", candidates(g))
        a = resolve("drop_junk", g)
        self.assertEqual(a.tool, "drop_item")
        names = [i["item_name"] for i in a.args["items"]]
        self.assertIn("iron_sword", names)
        self.assertIn("iron_pickaxe", names)
        self.assertIn("wooden_pickaxe", names)
        self.assertIn("iron_chestplate", names)
        self.assertNotIn("diamond_pickaxe", names)

    def test_spare_gear_never_drops_the_gold_helmet(self) -> None:
        # The gold helmet keeps piglins off in the Nether: it is not spare even though the worn
        # diamond helmet outranks it, so alone it is never a reason to shed.
        g = self._home(self._geared(golden_helmet=1))
        g.slots_used = 36
        self.assertNotIn("drop_junk", candidates(g))

    def test_drop_junk_uses_a_hole_until_the_terrain_refuses_one(self) -> None:
        # `in_a_hole` digs an enclosed disposal pit so the dropped blocks stay put; where the
        # terrain cannot give one (live: a 1-wide tunnel answered `[DROP_HOLE_UNAVAILABLE]` on
        # every tick) the relief drops on the open ground instead, which still frees the slot.
        g = self._home(self._geared(tuff=64))
        g.slots_used = 34
        self.assertTrue(resolve("drop_junk", g).args["in_a_hole"])
        g.drop_hole_blocked = True
        self.assertFalse(resolve("drop_junk", g).args["in_a_hole"])

    def test_inventory_relief_stows_at_the_base_chest(self) -> None:
        g = self._home(self._geared(tuff=64))
        g.chest_pos = (-197, -46, -97)
        g.slots_used = 36
        self.assertEqual(inventory_relief(g).tool, "use_container")

    def test_inventory_relief_drops_without_a_chest(self) -> None:
        g = self._home(self._geared(tuff=64))
        g.slots_used = 36
        self.assertEqual(inventory_relief(g).tool, "drop_item")

    def test_inventory_relief_skips_a_full_chest(self) -> None:
        # After a deposit failed (the chest is full) the relief must drop, not retry depositing.
        g = self._home(self._geared(tuff=64))
        g.chest_pos = (-197, -46, -97)
        g.slots_used = 36
        self.assertEqual(inventory_relief(g, allow_stow=False).tool, "drop_item")


class BaseFurnaceTest(unittest.TestCase):
    """The furnace that stands at the base, so cooking does not tear a temporary one down.

    `smelt_item` with `temporary_workstation` places a carried furnace and collects it again
    before returning, so every `cook_food`/`smelt_gold` call in the lull set the furnace up and
    broke it down again. Standing the carried furnace at the base once (`place_furnace`) and
    naming its cell lets the same furnace be reused.
    """

    @staticmethod
    def _lull(**inv) -> GameState:
        return lit_portal(at_home(geared(**inv)))

    def test_place_furnace_at_the_base(self) -> None:
        g = self._lull(furnace=1, beef=2)
        self.assertIn("place_furnace", candidates(g))
        a = resolve("place_furnace", g)
        self.assertEqual(a.tool, "place_block")
        self.assertEqual(a.args["block_name"], "furnace")
        # One block out from the base on the side opposite the chest, off the portal plane (z=bz).
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -99))

    def test_place_furnace_waits_for_the_lull(self) -> None:
        # Before the portal is lit the chain still wants the furnace for smelting away from the
        # base, so it is not stood down permanently yet.
        g = at_home(geared(furnace=1, beef=2))
        self.assertNotIn("place_furnace", candidates(g))

    def test_a_furnace_already_standing_is_not_placed_again(self) -> None:
        g = self._lull(furnace=1, beef=2)
        g.furnace_pos = (-197, -46, -99)
        self.assertNotIn("place_furnace", candidates(g))
        self.assertIsNone(resolve("place_furnace", g))

    def test_place_furnace_dropped_when_the_ground_cannot_hold_it(self) -> None:
        g = self._lull(furnace=1, beef=2)
        g.furnace_blocked = True
        self.assertNotIn("place_furnace", candidates(g))
        self.assertIsNone(resolve("place_furnace", g))

    def test_go_home_to_stand_the_furnace_at_the_base(self) -> None:
        g = self._lull(furnace=1, beef=2)
        g.pos = (-250.5, 64.0, -120.5)   # away from the base
        self.assertIn("go_home", candidates(g))
        a = resolve("go_home", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -98))

    def test_cook_food_reuses_the_base_furnace(self) -> None:
        g = self._lull(furnace=0, beef=3, coal=1)
        g.furnace_pos = (-197, -46, -99)
        a = resolve("cook_food", g)
        self.assertEqual(a.tool, "smelt_item")
        self.assertNotIn("temporary_workstation", a.args)
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -99))

    def test_cook_food_places_a_temporary_furnace_without_a_base_one(self) -> None:
        a = resolve("cook_food", gs(inventory={"furnace": 1, "coal": 1, "beef": 2}))
        self.assertTrue(a.args["temporary_workstation"])
        self.assertNotIn("x", a.args)

    def test_the_furnace_keeps_clear_of_a_placed_bed(self) -> None:
        # A bed on the furnace cell would make the placement `[PLACE_TARGET_OCCUPIED]` and
        # `make_room` would dig the respawn bed away, so the furnace stands nowhere there and
        # cooking falls back to the temporary workstation.
        g = self._lull(furnace=1, beef=2, acacia_planks=3)
        g.resources = {**g.resources,
                       "white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -99, 2.0)]}
        self.assertNotIn("place_furnace", candidates(g))
        self.assertIsNone(resolve("place_furnace", g))
        self.assertTrue(resolve("cook_food", g).args["temporary_workstation"])

    def test_smelt_gold_reuses_the_base_furnace(self) -> None:
        g = self._lull(furnace=0, raw_gold=2, coal=1)
        g.furnace_pos = (-197, -46, -99)
        a = resolve("smelt_gold", g)
        self.assertEqual(a.tool, "smelt_item")
        self.assertNotIn("temporary_workstation", a.args)
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -99))

    def test_a_full_pack_holds_cooking_back(self) -> None:
        # Live, run26: the cook completed but the pack had no free slot for the output, so the
        # output stayed in the furnace and every later insert was refused (`[FURNACE_NOT_EMPTY]`).
        # One free slot keeps the recovery from stranding itself; at 36 the loop sheds first.
        g = self._lull(furnace=0, beef=2, coal=1)
        g.furnace_pos = (-197, -46, -99)
        # An animal in reach, so this is about the pack room and not about whether cooking can
        # advance the food goal at all (see `CookFoodReachTest`).
        g.mobs = [("cow", "animal", 1, 9.0)]
        g.slots_used = 36
        self.assertNotIn("cook_food", candidates(g))
        self.assertIsNone(resolve("cook_food", g))
        g.slots_used = 35
        self.assertIn("cook_food", candidates(g))
        self.assertIsNotNone(resolve("cook_food", g))

    def test_a_full_pack_holds_gold_smelting_back(self) -> None:
        g = self._lull(furnace=0, raw_gold=2, coal=1)
        g.furnace_pos = (-197, -46, -99)
        g.slots_used = 36
        self.assertNotIn("smelt_gold", candidates(g))
        self.assertIsNone(resolve("smelt_gold", g))


class EscapeWaterTest(unittest.TestCase):
    """The no-footing rescue: while the bot floats in water nothing else works.

    Live failure: the obsidian dig left the bot treading water off the ground and every
    following `collect_block` thrashed in place for minutes, while `navigate` still walked it
    clear. The rescue is forced into `candidates` rather than offered, because without footing
    no other label can make progress -- but only once the bot has spent
    `WATER_RESCUE_AFTER` decisions adrift, since the server swims and a crossing must be allowed
    to finish (live, run15).
    """

    def test_no_footing_in_water_owns_the_tick(self) -> None:
        g = gs(in_water=True, on_ground=False, water_ticks=WATER_RESCUE_AFTER)
        self.assertEqual(candidates(g), ["escape_water"])

    def test_the_first_decision_off_the_feet_is_left_to_the_goal(self) -> None:
        # The server navigates a swim, so a bot between two banks is working, not trapped: the
        # rescue must not tear it out of a crossing the moment an action comes back (run15).
        g = geared(iron_pickaxe=1, iron_sword=1)
        g.in_water, g.on_ground, g.water_ticks = True, False, WATER_RESCUE_AFTER - 1
        g.resources = {**g.resources,
                       "diamond_ore": [BlockHit("diamond_ore", "diamond_ore", 50, 7, -16, 55.0,
                                                above="diorite", below="diorite")]}
        labels = candidates(g)
        self.assertNotEqual(labels, ["escape_water"])
        self.assertIn("mine_diamond", labels)

    def test_a_grounded_bot_in_water_is_not_rescued(self) -> None:
        # Wading with the feet on the bottom (a water source, a pour that did not flood the bot)
        # is ordinary play, not the trap.
        g = gs(in_water=True, on_ground=True)
        self.assertNotIn("escape_water", candidates(g))

    def test_a_bot_in_the_air_is_not_rescued(self) -> None:
        # Mid-jump or mid-fall is not the trap either.
        g = gs(in_water=False, on_ground=False)
        self.assertNotIn("escape_water", candidates(g))

    def test_escape_prefers_near_dry_ground_over_a_far_deep_base(self) -> None:
        # Live run8: treading water, the bot drowned ten-plus times while `navigate` tried to dig
        # 70 m straight down to a base 300 m off. The nearest dry cell is the target it can reach.
        g = gs(in_water=True, on_ground=False, base=(305, -9, -177),
               pos=(332.5, 62.4, -208.5))
        g.resources = {"sand": [BlockHit("sand", "sand", 336, 62, -206, 4.2,
                                         above="air", below="sand")]}
        a = resolve("escape_water", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (336, 63, -206))
        self.assertTrue(a.args["dig"] and a.args["scaffold"])

    def test_escape_falls_back_to_the_base_without_dry_ground(self) -> None:
        # The base is the last resort, for open water with no standable cell in scan reach.
        g = gs(in_water=True, on_ground=False, base=(-197, -47, -98),
               pos=(-300.5, 60.0, -120.5))
        a = resolve("escape_water", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -98))
        self.assertTrue(a.args["dig"] and a.args["scaffold"])

    def test_escape_without_a_base_walks_to_dry_ground(self) -> None:
        g = gs(in_water=True, on_ground=False, pos=(-300.5, 60.0, -120.5))
        g.resources = {"grass_block": [BlockHit("grass_block", "grass_block",
                                                -300, 64, -121, 4.0, above="air", below="dirt")]}
        a = resolve("escape_water", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-300, 65, -121))

    def test_escape_without_ground_defers(self) -> None:
        g = gs(in_water=True, on_ground=False, pos=(-300.5, 60.0, -120.5))
        self.assertIsNone(resolve("escape_water", g))


class LeavePortalTest(unittest.TestCase):
    """The standing-in-a-portal rescue.

    The shared pathfinder walks a portal block like any open gate, so a `navigate` goal on the
    far side routes the body through the gate and the teleport fires a few seconds later - no
    step ever asked for the crossing (live, run22: the lit gate stood beside the base and the bot
    was in the Nether a few decisions after it was lit). The rescue owns the tick outright, so
    the body steps out before the teleport.
    """

    def test_standing_in_a_portal_owns_the_tick(self) -> None:
        # The bot's feet cell *is* a portal block: it is halfway through the gate.
        g = geared()
        g.base = (-197, -47, -98)
        g.pos = (-195.5, -46.0, -97.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -196, -46, -98, 0.6)]}
        self.assertEqual(candidates(g), ["leave_portal"])

    def test_leave_steps_back_to_the_base(self) -> None:
        # The frame plane sits two blocks east of the base, so the base is the natural step-out.
        g = geared()
        g.base = (-197, -47, -98)
        g.pos = (-195.5, -46.0, -97.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -196, -46, -98, 0.6)]}
        a = resolve("leave_portal", g)
        self.assertEqual(a.tool, "navigate")
        # Whole block cells: `navigate` validates `x` and `z` as integers, and a float was refused
        # at the door, so the rescue never fired and the run froze on the spot (run35).
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -98))
        for axis in ("x", "z"):
            self.assertIsInstance(a.args[axis], int)
        # Never dig or scaffold: the rescue must not damage the frame around the opening.
        self.assertFalse(a.args["dig"])
        self.assertFalse(a.args["scaffold"])

    def test_leave_without_a_base_steps_to_a_neighbour(self) -> None:
        g = geared()
        g.pos = (-195.5, 64.0, -97.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -196, 64, -98, 0.6)]}
        a = resolve("leave_portal", g)
        self.assertEqual(a.tool, "navigate")
        # The target is a horizontal neighbour, never the portal cell itself.
        cell = (math.floor(a.args["x"]), math.floor(a.args["y"]), math.floor(a.args["z"]))
        self.assertNotEqual(cell, (-196, 64, -98))

    def test_a_portal_beside_the_bot_is_not_the_rescue(self) -> None:
        # A lit gate a couple of blocks off is the normal case (the bot works at the base beside
        # it); only actually standing *in* it is the rescue.
        g = geared()
        g.base = (-197, -47, -98)
        g.pos = (-196.5, -46.0, -97.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -194, -46, -98, 3.0)]}
        self.assertNotEqual(candidates(g), ["leave_portal"])

    def test_head_only_in_a_portal_is_still_the_rescue(self) -> None:
        # Feet clear but the head block is the portal: still mid-teleport, still a rescue.
        g = geared()
        g.base = (-197, -47, -98)
        g.pos = (-196.5, -46.0, -97.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -197, -45, -98, 1.0)]}
        self.assertEqual(candidates(g), ["leave_portal"])


class NetherFrontierHeadingTest(unittest.TestCase):
    """The fortress search walks the world's unexplored edge, and keeps the axis as its fallback.

    A compass line is a bet on which way the fortresses are striped, and the sources disagree
    (east-west here, north-south there) while four Nether runs never found a fortress to measure.
    The server already knows which chunks it has observed, so it can name the edge instead: walking
    that heading covers new ground and turns by itself, which is the grid the bot never had.
    """

    NETHER = "minecraft:the_nether"

    def _down(self, x: float, arrival_x: float = 0.0, heading=None) -> GameState:
        g = gs(pos=(x, 65.0, 0.0), dimension=self.NETHER, health=20.0, food=18.0)
        g.base = (-197, -47, -98)
        g.nether_entered = True
        g.nether_arrival = (arrival_x, 65, 0)
        g.frontier_heading = heading
        return g

    def test_the_search_walks_the_unobserved_edge_when_the_server_names_one(self) -> None:
        # Live, run43: the bot came in at x=33 and ended at x=-33, on one heading for the whole
        # trip. North is the heading the frontier read reported for the run's own chunk table.
        g = self._down(10.0, heading=0.0)
        a = resolve("explore", g)
        self.assertEqual(a.tool, "explore_frontier")
        self.assertEqual(a.args["heading"], 0.0)

    def test_a_heading_in_any_direction_is_followed_verbatim(self) -> None:
        for heading in (0.0, 45.0, 137.5, 180.0, 254.0, 315.0):
            a = resolve("explore", self._down(10.0, heading=heading))
            self.assertEqual(a.args["heading"], heading)

    def test_the_frontier_answers_in_both_directions_where_the_axis_march_would_not(self) -> None:
        # The axis march is binary: past the arrival cell it always goes west. The frontier is not,
        # so this heading is reachable only through the new read.
        east_of_arrival = self._down(120.0, arrival_x=0.0, heading=180.0)
        self.assertEqual(resolve("explore", east_of_arrival).args["heading"], 180.0)

    def test_a_tick_without_a_frontier_read_falls_back_to_the_axis_march(self) -> None:
        east = self._down(120.0, arrival_x=0.0)
        self.assertEqual(resolve("explore", east).args["heading"], 270.0)
        west = self._down(-120.0, arrival_x=0.0)
        self.assertEqual(resolve("explore", west).args["heading"], 90.0)

    def test_a_fortress_still_wins_over_the_heading(self) -> None:
        g = self._down(10.0, heading=0.0)
        g.resources = {"nether_bricks": [BlockHit("nether_bricks", "nether_bricks",
                                                 31, 64, -8, 16.0)]}
        self.assertEqual(resolve("explore", g).tool, "navigate")


class FrontierHeadingParseTest(unittest.TestCase):
    """`frontier_navigation` arrives as `query.rows`, a list of row lists of one number."""

    def test_the_nearest_row_reads_as_a_compass_heading(self) -> None:
        payload = {"response": {"data": {"result": {"query": {
            "columns": ["heading_degrees"], "rows": [[254.0]], "returnedRows": 1}}}}}
        self.assertEqual(parse_frontier_heading(payload), 254.0)

    def test_a_heading_is_wrapped_into_the_compass_rose(self) -> None:
        payload = {"response": {"data": {"result": {"query": {
            "columns": ["heading_degrees"], "rows": [[361.5]], "returnedRows": 1}}}}}
        self.assertEqual(parse_frontier_heading(payload), 1.5)

    def test_no_rows_is_no_heading_rather_than_a_guess(self) -> None:
        # Either the ground around the bot is fully observed or the read failed. Both are the
        # caller's to answer with a fallback, so this is None and never a made-up direction.
        payload = {"response": {"data": {"result": {"query": {
            "columns": ["heading_degrees"], "rows": [], "returnedRows": 0}}}}}
        self.assertIsNone(parse_frontier_heading(payload))

    def test_a_failed_or_shapeless_answer_is_no_heading(self) -> None:
        self.assertIsNone(parse_frontier_heading({}))
        self.assertIsNone(parse_frontier_heading({"response": {"data": {"result": {"error": "x"}}}}))
        self.assertIsNone(parse_frontier_heading({"response": {"data": {"result": {"query": {
            "rows": [["eastish"]]}}}}}))


class NetherSearchHeadingTest(unittest.TestCase):
    """The fortress search marches the east-west axis instead of wandering.

    Nether fortresses spawn in parallel east-west stripes, so a straight line along that axis
    crosses them. The generic `explore` branch takes the best standable ground inside the scan
    radius, and in the Nether that is netherrack in every direction - live run37 walked 231 blocks
    to gain 31 blocks of ground, a 13 % straight line through open lava sea that crossed no stripe.
    """

    NETHER = "minecraft:the_nether"

    def _down(self, x: float, arrival_x: float = 0.0, **res) -> GameState:
        g = gs(pos=(x, 65.0, 0.0), dimension=self.NETHER, health=20.0, food=18.0)
        g.base = (-197, -47, -98)
        g.nether_entered = True
        g.nether_arrival = (arrival_x, 65, 0)
        g.resources = dict(res)
        return g

    def test_the_search_marches_the_axis_rather_than_hopping_to_nearby_ground(self) -> None:
        g = self._down(10.0)
        a = resolve("explore", g)
        self.assertEqual(a.tool, "explore_frontier")
        self.assertIn(a.args["heading"], (90.0, 270.0))

    def test_the_march_flips_once_it_is_past_where_it_came_in(self) -> None:
        # East until it is level with the arrival cell, then back west: a boustrophedon sweep
        # across the stripes, with no state of its own to keep.
        east = self._down(120.0, arrival_x=0.0)
        self.assertEqual(resolve("explore", east).args["heading"], 270.0)
        west = self._down(-120.0, arrival_x=0.0)
        self.assertEqual(resolve("explore", west).args["heading"], 90.0)

    def test_a_spawner_in_reach_is_walked_to_rather_than_marched_past(self) -> None:
        g = self._down(10.0, spawner=[BlockHit("spawner", "spawner", 40, 66, 4, 22.0)])
        a = resolve("explore", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (40, 66, 4))

    def test_fortress_bricks_win_over_marching(self) -> None:
        g = self._down(10.0, nether_bricks=[BlockHit("nether_bricks", "nether_bricks",
                                                     31, 64, -8, 16.0)])
        a = resolve("explore", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (31, 64, -8))

    def test_the_overworld_wander_is_untouched(self) -> None:
        # Home, a fortress is nothing to walk to - the walk-at-the-bricks is the Nether's, and
        # `explore` there keeps the ordinary branch it always had.
        g = at_home(geared(crafting_table=1, cobblestone=64), base=(-197, -47, -98))
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        g.nether_arrival = None
        g.resources = {"nether_bricks": [BlockHit("nether_bricks", "nether_bricks",
                                                  -180, 64, -90, 14.0)]}
        a = resolve("explore", g)
        self.assertNotEqual((a.args.get("x"), a.args.get("y"), a.args.get("z")), (-180, 64, -90))


class CraftBedDimensionTest(unittest.TestCase):
    """A bed is only worth making where one can be laid.

    `sleep`, `place_bed` and `fill_bucket` are Overworld-only (a bed explodes in the Nether, water
    evaporates there), and `craft_bed` was not: the trip was offered a bed it could not lay, so it
    spent wool and a slot on it while it should have been hunting for blaze rods (live, run37).
    """

    def _stocked(self, dimension: str) -> GameState:
        g = at_home(geared(crafting_table=1, white_wool=6, oak_log=4), base=(-197, -47, -98))
        g.dimension = dimension
        g.health = 20.0
        g.food = 18.0
        return g

    def test_the_bed_chain_still_works_in_the_overworld(self) -> None:
        g = self._stocked("minecraft:overworld")
        self.assertIn("craft_bed", candidates(g))
        self.assertIsNotNone(resolve("craft_bed", g))

    def test_no_bed_is_made_in_the_nether(self) -> None:
        g = self._stocked("minecraft:the_nether")
        self.assertNotIn("craft_bed", candidates(g))
        self.assertIsNone(resolve("craft_bed", g))

    def test_no_wood_is_chopped_in_the_nether(self) -> None:
        # The Nether has no trees. `get_wood` stayed in the offer all the way through run42 and
        # took every tick over `leave_nether`, so 600 iterations bought nothing.
        g = self._stocked("minecraft:the_nether")
        g.inventory.pop("crafting_table", None)
        g.inventory.pop("oak_log", None)
        self.assertNotIn("get_wood", candidates(g))
        self.assertIsNone(resolve("get_wood", g))

    def test_wood_is_still_chopped_in_the_overworld(self) -> None:
        g = self._stocked("minecraft:overworld")
        g.inventory.pop("crafting_table", None)
        g.inventory.pop("oak_log", None)
        self.assertIn("get_wood", candidates(g))

    def test_the_bed_chain_is_untouched_where_a_bed_already_stands(self) -> None:
        # The Nether rule adds to that one, it does not replace it.
        g = self._stocked("minecraft:overworld")
        g.resources = {"white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -100, 1.5)]}
        self.assertNotIn("craft_bed", candidates(g))


class GoldHelmetSlotTest(unittest.TestCase):
    """In the Nether the helmet slot belongs to the gold helmet.

    `swap_to_gold_helmet` wears the gold piece so piglins stay neutral and leaves the diamond
    helmet in the pack. Read as "a better piece is waiting for a bare slot", `equip_armor` puts it
    straight back on the head and the two steps undo each other (live, run36: alternating for
    eight ticks, every action a success, no progress at all).
    """

    BASE = (-197, -47, -98)
    NETHER = "minecraft:the_nether"

    def _in_the_nether(self, helmet: str, **inv) -> GameState:
        g = gs(pos=(12.5, 65.0, -3.5), dimension=self.NETHER, inventory=dict(inv),
               tools={"pickaxe": "diamond", "sword": "diamond"}, offhand="shield",
               health=20.0, food=18.0)
        g.armor = {slot: {"item": f"diamond_{slot}"} for slot in
                   ("helmet", "chestplate", "leggings", "boots")}
        g.armor["helmet"] = {"item": helmet}
        g.base = self.BASE
        g.nether_entered = True
        return g

    def test_the_diamond_helmet_is_left_in_the_pack_in_the_nether(self) -> None:
        # The swap has just freed it; wearing it again is what the ping-pong was.
        g = self._in_the_nether("golden_helmet", golden_helmet=1, diamond_helmet=1)
        self.assertNotIn("diamond_helmet", _armor_to_wear(g))
        self.assertNotIn("equip_armor", candidates(g))

    def test_the_other_slots_still_go_on_in_the_nether(self) -> None:
        # The helmet is the swap's business; a diamond chestplate waiting under an iron one is
        # plain `equip_armor` work and must not be skipped along with it.
        g = self._in_the_nether("golden_helmet", golden_helmet=1,
                                diamond_chestplate=1)
        g.armor["chestplate"] = {"item": "iron_chestplate"}
        self.assertIn("diamond_chestplate", _armor_to_wear(g))

    def test_the_overworld_still_wears_the_better_helmet(self) -> None:
        # Home, a diamond helmet waiting under the gold one is a better piece in a bare slot and
        # nothing else claims that slot, so it goes on.
        g = self._in_the_nether("golden_helmet", diamond_helmet=1)
        g.dimension = "minecraft:overworld"
        self.assertIn("diamond_helmet", _armor_to_wear(g))

    def test_the_gold_helmet_keeps_the_slot_while_it_is_on_the_head(self) -> None:
        # Once it is worn, the gold piece owns the slot for the trip: the diamond helmet waits in
        # the pack until the bot is home again.
        g = self._in_the_nether("golden_helmet", diamond_helmet=1)
        self.assertNotIn("diamond_helmet", _armor_to_wear(g))

    def test_without_a_gold_helmet_the_nether_keeps_the_diamond_one(self) -> None:
        # No gold piece carried and none worn: there is no swap in progress, so a better helmet
        # waiting for its slot is plain `equip_armor` work.
        g = self._in_the_nether("iron_helmet", diamond_helmet=1)
        self.assertIn("diamond_helmet", _armor_to_wear(g))


class NavigateArgumentsTest(unittest.TestCase):
    """Every `navigate` call must name whole block cells.

    The server validates `x` and `z` as integers and refuses a float before it looks at anything
    else. A float is not a small inaccuracy here: the call never runs, and where the step owns
    the tick - `leave_portal` does, through `_in_portal` - the run freezes silently (live, run35:
    23 identical `[expected int, received number at x]` refusals with no other candidate).
    """

    BASE = (-197, -47, -98)

    def _base_state(self, **inv) -> GameState:
        g = at_home(geared(furnace=1, crafting_table=1, cobblestone=64, **inv), base=self.BASE)
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        return g

    def _navigations(self, g: GameState) -> list:
        """Every label that resolves to a `navigate`, with the state as it stands."""
        from mlaya.actions import candidates
        return [a for a in (resolve(label, g) for label in candidates(g))
                if a is not None and a.tool == "navigate"]

    def test_a_base_state_navigates_only_to_whole_cells(self) -> None:
        g = self._base_state(coal=8, beef=3)
        for action in self._navigations(g):
            with self.subTest(label=action.label):
                self.assertIsInstance(action.args["x"], int)
                self.assertIsInstance(action.args["z"], int)

    def test_the_portal_rescue_navigates_to_a_whole_cell(self) -> None:
        # The rescue's own target: the base cell centre, which has to go out as that cell.
        g = self._base_state()
        g.pos = (-195.5, -46.0, -97.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -196, -46, -98, 0.6)]}
        a = resolve("leave_portal", g)
        self.assertEqual(a.tool, "navigate")
        self.assertIsInstance(a.args["x"], int)
        self.assertIsInstance(a.args["z"], int)


class CookFoodReachTest(unittest.TestCase):
    """Cooking only while a hunt could follow it.

    Cooking turns one raw item into one cooked one and both count the same in `_edible_count`,
    so it can never close the gap it is offered for. While it was the only step on offer it also
    suppressed the exploration fallback, which only fires on an empty list - so the bot stayed at
    its base cooking through its stock instead of looking for what it was short of (run34: a bow
    it could not rebuild, no spider in reach, `candidates: ['cook_food']`).
    """

    BASE = (-197, -47, -98)

    def _stocked(self, **inv) -> GameState:
        # Two pickaxes, because that is what a banked trip carries now (`NETHER_PICKAXES`); with
        # one, the spare-pickaxe step outranks everything the lull is being tested for.
        g = at_home(geared(furnace=1, coal=8, beef=5, crafting_table=1,
                            cobblestone=64, diamond_pickaxe=2, **inv), base=self.BASE)
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        return g

    def test_raw_food_is_cooked_while_an_animal_is_in_reach(self) -> None:
        # The useful case is unchanged: a hunt is available, so the reserve can still grow and
        # cooked food is worth the furnace.
        g = self._stocked()
        g.mobs = [("cow", "animal", 2, 12.0)]
        self.assertIn("cook_food", candidates(g))

    def test_cooking_drops_out_when_no_food_source_is_in_reach(self) -> None:
        g = self._stocked()
        self.assertNotIn("cook_food", candidates(g))

    def test_the_exploration_fallback_gets_the_tick_back(self) -> None:
        # The point of dropping it: an empty offer list is what lets `wait`/`explore` run, so the
        # bot can go and look for what it is missing instead of standing over a furnace.
        g = self._stocked()
        self.assertIn("explore", candidates(g))


class CollectSmeltedTest(unittest.TestCase):
    """Draining the base furnace, instead of abandoning its cell.

    A completed cook leaves its output in the furnace, and the next cook is refused with
    `[FURNACE_NOT_EMPTY]`. The loop used to give the cell up and cook in a carried furnace, which
    cost a furnace per batch and left the base with no oven at all. The server opens a furnace for
    `inspect`/`withdraw`, so the output is taken back into the pack first.
    """

    BASE = (-197, -47, -98)
    FURNACE = (-197, -46, -99)

    def _home(self, **inv) -> GameState:
        g = at_home(geared(**inv), base=self.BASE)
        g.dimension = "minecraft:overworld"
        g.furnace_pos = self.FURNACE
        g.health = 20.0
        g.food = 15.0
        return g

    def test_a_refused_cook_puts_the_furnace_on_the_list(self) -> None:
        g = self._home(beef=5, coal=4)
        g.furnace_occupied = True
        g.furnace_contents = {"cooked_beef": 2}
        self.assertIn("collect_smelted", candidates(g))
        a = resolve("collect_smelted", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "withdraw")
        self.assertEqual(a.args["items"], [{"item_name": "cooked_beef", "count": 2}])
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.FURNACE)

    def test_an_unlooked_furnace_is_inspected_before_it_is_reached_into(self) -> None:
        # The refusal says "occupied", not "with what": the first look is an inspect, so a cook is
        # only ever interrupted by a furnace that really holds something.
        g = self._home(beef=5, coal=4)
        g.furnace_occupied = True
        a = resolve("collect_smelted", g)
        self.assertEqual(a.args["operation"], "inspect")

    def test_a_furnace_nobody_refused_is_left_alone(self) -> None:
        # No refusal means no interruption: the step never reaches into a furnace mid-cook.
        g = self._home(beef=5, coal=4)
        self.assertNotIn("collect_smelted", candidates(g))
        self.assertIsNone(resolve("collect_smelted", g))

    def test_an_empty_furnace_has_nothing_to_withdraw(self) -> None:
        g = self._home(beef=5, coal=4)
        g.furnace_occupied = True
        g.furnace_contents = {"coal": 0}
        self.assertIsNone(resolve("collect_smelted", g))

    def test_a_full_pack_is_shed_before_the_furnace_is_drained(self) -> None:
        # The output has to land somewhere: recovering it onto a full pack is what stranded it in
        # the slot the first time (live, run26).
        g = self._home(beef=5, coal=4)
        g.furnace_occupied = True
        g.furnace_contents = {"cooked_beef": 2}
        g.slots_used = 36
        self.assertNotIn("collect_smelted", candidates(g))

    def test_a_cell_the_terrain_refused_is_not_drained(self) -> None:
        # `furnace_blocked` means there is no usable oven there at all; the flag is not stale
        # truth about an occupied one.
        g = self._home(beef=5, coal=4)
        g.furnace_occupied = True
        g.furnace_blocked = True
        g.furnace_contents = {"cooked_beef": 2}
        self.assertNotIn("collect_smelted", candidates(g))


class NetherExpeditionTest(unittest.TestCase):
    """The first Nether trip: nine blaze rods home, eighteen powder in the chest at the gate.

    The chain the user asked for: cross into the Nether, keep piglins neutral with the gold
    helmet, kill blazes for nine rods, come back over the exit portal, turn the rods into powder
    in the Overworld, and park the powder in the chest at the portal. Getting *back out* is part
    of the goal, so the return is offered whenever the haul is done or the bot can no longer
    afford to stay.
    """

    BASE = (-197, -47, -98)
    NETHER = "minecraft:the_nether"
    OVERWORLD = "minecraft:overworld"
    # The Overworld gate at the base: the frame plane sits two blocks east of it.
    GATE = (-195, -46, -98)
    # A Nether exit portal a few blocks from where the bot lands.
    EXIT = (12, 65, -4)

    def _home(self, **inv) -> GameState:
        """The bot at its base in the Overworld, gate lit and the expedition fully banked."""
        # `diamond_pickaxe=2`: banking the trip includes the spare (`NETHER_PICKAXES`). One
        # pickaxe is not a banked expedition any more - it breaks in the Nether and nothing is
        # recoverable there, so a bot holding one is still on its way to the gate, not through it.
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64,
                     golden_helmet=1, cooked_mutton=64, white_bed=1,
                     diamond_pickaxe=2)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                   *self.GATE, 2.0)],
                       "obsidian": [BlockHit("obsidian", "obsidian", -195, -46, -98, 2.0)],
                       # The gate bed, standing clear of the opening as `place_bed` lays it.
                       "white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -100, 2.5)]}
        g.dimension = self.OVERWORLD
        return g

    def _down(self, **inv) -> GameState:
        """The bot in the Nether, helmet worn, no exit portal loaded."""
        g = gs(pos=(12.5, 65.0, -3.5), dimension=self.NETHER, inventory=dict(inv),
               tools={"pickaxe": "diamond", "sword": "diamond"},
               armor={s: {"item": f"diamond_{s}"} for s in
                      ("helmet", "chestplate", "leggings", "boots")},
               offhand="shield", health=20.0, food=18.0)
        g.base = self.BASE
        g.nether_entered = True
        g.nether_arrival = self.EXIT
        return g

    def _with_exit(self, g: GameState) -> GameState:
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                   *self.EXIT, 2.0)]}
        return g

    # --- crossing in ---
    def test_the_gate_opens_once_the_expedition_is_stocked(self) -> None:
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=64)
        self.assertIn("enter_portal", candidates(g))
        a = resolve("enter_portal", g)
        self.assertEqual(a.tool, "enter_nether_portal")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.GATE)
        # The server pauses a crossing below 32 arrows; this run banks 16, so the crossing would
        # otherwise be refused forever (live, run41: `[PORTAL_LOW_SUPPLIES] 38/16 food 0/32
        # arrows` with a stocked expedition).
        self.assertTrue(a.args["allow_low_supplies"])

    def test_the_crossing_needs_no_low_supplies_override_to_leave_the_nether(self) -> None:
        # The guard only applies on the way in: coming home, the destination is the Overworld.
        g = self._with_exit(self._down(blaze_rod=9))
        a = resolve("leave_nether", g)
        self.assertNotIn("allow_low_supplies", a.args)

    def test_the_gate_stays_shut_before_the_expedition_is_stocked(self) -> None:
        # The trip needs the whole bank first: a bot that walks in with half an inventory loses
        # the kit and the trip with it.
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=12)
        self.assertNotIn("enter_portal", candidates(g))
        self.assertIsNone(resolve("enter_portal", g))

    def test_a_bow_less_bot_may_still_depart(self) -> None:
        # The bow is part of the bank, not a precondition for the trip: a diamond sword and shield
        # are what the bot fights blazes with, and holding the departure for a bow made run34 wait
        # at its own gate for an item nothing on the map could supply. The Overworld section is
        # still not "done" - the bank is short - so the run keeps working either way.
        g = self._home()
        g.inventory.pop("bow", None)
        g.inventory.pop("arrow", None)
        self.assertIn("enter_portal", candidates(g))
        self.assertFalse(tech_tree_complete(g))

    def test_the_gate_is_not_offered_twice(self) -> None:
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=64)
        g.nether_entered = True
        self.assertNotIn("enter_portal", candidates(g))

    def test_an_intended_crossing_stands_the_rescue_down(self) -> None:
        # The rescue exists to step a body out of a gate it wandered into; it must not fight the
        # one crossing the loop asked for, or the trip can never happen.
        g = self._home()
        g.pos = (self.GATE[0] + 0.5, float(self.GATE[1]), self.GATE[2] + 0.5)
        self.assertEqual(candidates(g), ["leave_portal"])
        g.enter_intent = True
        self.assertNotEqual(candidates(g), ["leave_portal"])

    # --- down there ---
    def test_the_gold_helmet_goes_on_first(self) -> None:
        g = self._down(golden_helmet=1)
        self.assertIn("swap_to_gold_helmet", candidates(g))
        a = resolve("swap_to_gold_helmet", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "golden_helmet", "destination": "head"}])

    def test_the_helmet_swap_waits_for_the_gold_piece(self) -> None:
        # Piglin neutrality needs a *worn* gold piece; without one there is nothing to swap to.
        g = self._down()
        self.assertNotIn("swap_to_gold_helmet", candidates(g))

    def test_a_blaze_in_reach_is_hunted(self) -> None:
        g = self._down()
        g.mobs = [("blaze", "hostile", 1, 8.0)]
        self.assertIn("hunt_blaze", candidates(g))
        a = resolve("hunt_blaze", g)
        self.assertEqual(a.tool, "collect_mob_drop")
        self.assertEqual((a.args["mob_name"], a.args["drop_name"]), ("blaze", "blaze_rod"))

    def test_a_blaze_beyond_reach_is_not_chased(self) -> None:
        # Blazes hover; the bot walks the fortress bricks first rather than sprinting at one.
        g = self._down()
        g.mobs = [("blaze", "hostile", 1, 90.0)]
        self.assertNotIn("hunt_blaze", candidates(g))

    def test_a_hurt_bot_does_not_start_a_blaze_fight(self) -> None:
        g = self._down()
        g.mobs = [("blaze", "hostile", 1, 6.0)]
        g.health = 9.0
        self.assertNotIn("hunt_blaze", candidates(g))

    def test_a_blaze_crowded_by_other_hostiles_is_left_alone(self) -> None:
        g = self._down()
        g.mobs = [("blaze", "hostile", 1, 6.0), ("piglin", "hostile", 2, 9.0)]
        self.assertNotIn("hunt_blaze", candidates(g))

    def test_the_trip_ends_with_the_haul(self) -> None:
        g = self._down(blaze_rod=9)
        g.mobs = [("blaze", "hostile", 1, 5.0)]
        self.assertIn("leave_nether", candidates(g))
        self.assertNotIn("hunt_blaze", candidates(g))

    def test_the_exit_portal_is_the_way_home(self) -> None:
        g = self._with_exit(self._down(blaze_rod=9))
        a = resolve("leave_nether", g)
        self.assertEqual(a.tool, "enter_nether_portal")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.EXIT)

    def test_with_no_portal_in_the_scan_the_bot_walks_back_to_its_arrival(self) -> None:
        # The exit portal only reaches the state while it is loaded. The cell the bot arrived on
        # is where it stands, so that is the heading home.
        g = self._down(blaze_rod=9)
        a = resolve("leave_nether", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.EXIT)

    def test_a_badly_hurt_bot_goes_home_rods_or_not(self) -> None:
        # Dying down there drops the kit and the haul both; the rods in the pack are already
        # worth the walk back.
        g = self._down(blaze_rod=2)
        g.health = 6.0
        self.assertIn("leave_nether", candidates(g))

    def test_a_full_pack_goes_home(self) -> None:
        g = self._down(blaze_rod=2)
        g.slots_used = 36
        self.assertIn("leave_nether", candidates(g))

    def test_a_bot_with_nothing_left_to_find_goes_home(self) -> None:
        # Hunted out, and the walk has shown no fortress anywhere: continuing is not a plan.
        g = self._down(blaze_rod=2)
        g.nether_ticks = NETHER_HOPELESS_AFTER
        self.assertIn("leave_nether", candidates(g))

    def test_open_nether_on_arrival_is_not_the_end_of_the_hunt(self) -> None:
        # The bot lands in open Nether, where a scan reaches nothing but lava and a fortress is
        # hundreds of blocks off. Turning round on the first tick would abandon every trip.
        g = self._down(blaze_rod=2)
        g.nether_ticks = 1
        self.assertNotIn("leave_nether", candidates(g))
        g.nether_ticks = NETHER_HOPELESS_AFTER - 1
        self.assertNotIn("leave_nether", candidates(g))

    def test_a_fortress_in_sight_keeps_the_hunt_open(self) -> None:
        g = self._down(blaze_rod=2)
        g.nether_ticks = NETHER_HOPELESS_AFTER + 10
        g.resources = {"nether_bricks": [BlockHit("nether_bricks", "nether_bricks",
                                                   40, 66, 10, 30.0)]}
        self.assertNotIn("leave_nether", candidates(g))

    def test_a_blaze_seen_but_out_of_reach_keeps_the_hunt_open(self) -> None:
        # A blaze 90 m off is not a hunt to start, but it is a reason to close in, not to leave.
        g = self._down(blaze_rod=2)
        g.nether_ticks = NETHER_HOPELESS_AFTER + 10
        g.mobs = [("blaze", "hostile", 1, 90.0)]
        self.assertNotIn("leave_nether", candidates(g))

    # --- home again ---
    def test_the_diamond_helmet_goes_back_on(self) -> None:
        g = self._home(blaze_rod=9, diamond_helmet=1)
        g.nether_entered = True
        g.armor["helmet"] = {"item": "golden_helmet"}
        self.assertIn("swap_to_diamond_helmet", candidates(g))
        a = resolve("swap_to_diamond_helmet", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_helmet", "destination": "head"}])

    def test_the_rods_become_eighteen_powder(self) -> None:
        g = self._home(blaze_rod=9, crafting_table=1)
        g.nether_entered = True
        self.assertIn("craft_blaze_powder", candidates(g))
        a = resolve("craft_blaze_powder", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "blaze_powder", "count": 18}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_powder_is_not_crafted_down_in_the_nether(self) -> None:
        # The Nether has no crafting table worth standing next to, and a temporary workstation
        # in a fortress is a bad trade. The powder is made at home.
        g = self._down(blaze_rod=9, crafting_table=1)
        self.assertNotIn("craft_blaze_powder", candidates(g))
        self.assertIsNone(resolve("craft_blaze_powder", g))

    def test_the_powder_goes_into_the_chest_at_the_gate(self) -> None:
        g = self._home(blaze_powder=18)
        g.nether_entered = True
        g.chest_pos = (-196, -46, -98)
        self.assertIn("stow_blaze_powder", candidates(g))
        a = resolve("stow_blaze_powder", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "deposit")
        self.assertEqual(a.args["items"], [{"item_name": "blaze_powder", "count": 18}])
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), g.chest_pos)

    def test_the_powder_needs_a_chest_to_go_into(self) -> None:
        g = self._home(blaze_powder=18)
        g.nether_entered = True
        self.assertNotIn("stow_blaze_powder", candidates(g))
        self.assertIsNone(resolve("stow_blaze_powder", g))

    def test_a_wandered_bot_is_walked_home_to_depart(self) -> None:
        # Nothing else pulls a stocked bot back once the frame is up, the bed is down and the
        # chest is empty: the gate is base-gated, so without this the trip can never start.
        g = self._home()
        g.pos = (-250.5, 40.0, -120.5)
        self.assertIn("go_home", candidates(g))

    def test_an_unstocked_bot_is_not_walked_home_for_the_gate(self) -> None:
        # The walk home is only owed to the trip once the trip can actually leave.
        g = self._home(cobblestone=12)
        g.pos = (-250.5, 40.0, -120.5)
        self.assertNotIn("go_home", candidates(g))

    def test_a_short_arrow_reserve_is_topped_up(self) -> None:
        # Sticks run out one per arrow and nothing else gathers them, so the arrow step used to
        # drop out at the three the bow left behind and the expedition left on eight arrows.
        g = self._home(flint_and_steel=1, bow=1, arrow=8, golden_helmet=1,
                       cooked_mutton=64, white_bed=1, flint=2, feather=12)
        g.resources = dict(g.resources)
        g.resources.pop("nether_portal")  # the frame stands unlit: the prep window
        g.resources["obsidian"] = [BlockHit("obsidian", "obsidian", -195, -46, -98, 2.0)]
        self.assertNotIn("craft_arrows", candidates(g))
        self.assertIn("get_wood", candidates(g))

    def test_the_goal_is_the_powder_in_the_chest(self) -> None:
        # Still carried is not done: the point of the trip is that the haul is safe at home.
        g = self._home(blaze_powder=18)
        self.assertFalse(nether_expedition_complete(g))
        g.powder_stowed = True
        self.assertTrue(nether_expedition_complete(g))

    def test_a_stowed_expedition_closes_the_chain(self) -> None:
        g = self._home()
        g.powder_stowed = True
        out = candidates(g)
        for label in ("enter_portal", "hunt_blaze", "leave_nether", "craft_blaze_powder",
                      "stow_blaze_powder"):
            self.assertNotIn(label, out)


class BurningGateUnstockedPrepTest(unittest.TestCase):
    """A gate that is already burning must not hold the prep hostage, and must not let the bot in.

    The case the user named: the bot was already in the Nether once, came back (a death, a missing
    resource), and finds the gate still burning. The lighting gate is then irrelevant - it is lit -
    so the prep has to be free to run, and the crossing has to stay shut until it is stocked.

    Pinned because the fix that put `NETHER_PICKAXES` on `_expedition_ready` could easily have cost
    the first half: the prep steps key on `_expedition_prepared`, which never reads the gate, while
    `enter_portal` still reads `_expedition_ready` on top of it.
    """

    BASE = (226, 63, 222)
    GATE = (228, 64, 222)

    def _burning_gate(self, **inv):
        """Gate lit. The expedition is stockable in whatever `inv` says it is missing."""
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64, white_bed=1,
                     golden_helmet=1, cooked_mutton=64, diamond_pickaxe=2, raw_gold=5)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        # The frame plane sits two blocks west of the gate and stands: the prep chain is scoped to
        # the window after the frame is up (`_nether_idle`), which is exactly the state a bot comes
        # home to after a trip - gate burning, frame standing, helmet gone.
        g.resources = {
            "nether_portal": [BlockHit("nether_portal", "nether_portal", *self.GATE, 2.0)],
            "white_bed": [BlockHit("white_bed", "white_bed", 225, 64, 223, 2.5)],
            "obsidian": [BlockHit("obsidian", "obsidian", 226, 64, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 65, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 66, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 67, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 68, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 69, 221, 2.0)],
            "furnace": [BlockHit("furnace", "furnace", 229, 62, 222, 4.5)],
            "gold_ore": [BlockHit("gold_ore", "gold_ore", 231, 60, 226, 8.0)],
        }
        g.furnace_pos = (229, 62, 222)   # the oven stands at the base, so no temporary one
        return g

    def test_a_burning_gate_does_not_stop_the_gold_chain(self) -> None:
        g = self._burning_gate(golden_helmet=0, raw_gold=0)
        self.assertIn("mine_gold", candidates(g))

    def test_a_burning_gate_does_not_stop_the_gold_helmet(self) -> None:
        # Five gold ingots and a table: the piece itself is reachable, so it is offered.
        g = self._burning_gate(golden_helmet=0, gold_ingot=5)
        self.assertIn("craft_gold_helmet", candidates(g))

    def test_the_gate_stays_shut_when_the_helmet_step_is_unreachable(self) -> None:
        # The sharp edge of the case the user named. With no coal there is no gold step in the
        # prep list at all, the empty list read as "prep done", and the crossing opened - into the
        # Nether, where piglins are hostile and the trip dies. The helmet is now read from the
        # pack, so the gate stays shut whatever the prep list happens to say.
        g = self._burning_gate(golden_helmet=0)
        self.assertEqual(_all_nether_prep_steps(g), [])
        self.assertNotIn("enter_portal", candidates(g))
        self.assertNotIn("light_portal", candidates(g))

    def test_the_spare_pickaxe_is_asked_for_once_the_other_prep_is_banked(self) -> None:
        # Order on purpose: `_spare_pickaxe_missing` reads `_expedition_prepared`, so the pickaxe
        # is the last thing before the crossing and not something to chase before the helmet is on.
        self.assertIn("craft_pickaxe_spare",
                      candidates(self._burning_gate(diamond_pickaxe=1)))

    def test_the_crossing_stays_shut_while_the_spare_is_missing(self) -> None:
        self.assertNotIn("enter_portal",
                         candidates(self._burning_gate(diamond_pickaxe=1)))

    def test_the_crossing_opens_on_a_burning_gate_once_the_kit_is_complete(self) -> None:
        self.assertIn("enter_portal", candidates(self._burning_gate()))

    def test_the_step_out_rescue_does_not_read_the_kit_at_all(self) -> None:
        # The protection that covers this case once the gate is already lit. Standing in a burning
        # gate teleports within seconds, so it owns the tick whatever the trip is missing - which
        # is why it must depend neither on the lighting gate nor on the kit.
        g = self._burning_gate(golden_helmet=0, diamond_pickaxe=1)
        g.pos = (self.GATE[0] + 0.5, float(self.GATE[1]), self.GATE[2] + 0.5)
        self.assertTrue(_in_portal(g))
        self.assertIn("leave_portal", candidates(g))
        self.assertEqual(resolve("leave_portal", g).tool, "navigate")


class BaseFurnaceIsAnOptimisationTest(unittest.TestCase):
    """Cooking and smelting never need the base; only draining a standing furnace does.

    `_smelt_args` names the standing furnace's cell when there is one and falls back to the
    carried furnace as a temporary workstation otherwise - which is the older behaviour the user
    remembered, and it still works. So the base furnace is an *optimisation* (reuse instead of
    set up and tear down on every call), never a precondition. What it does buy is `collect_smelted`,
    and that one does need the base: a temporary workstation is gone by the time the drain looks
    for it.
    """

    BASE = (226, 63, 222)
    FURNACE = (229, 62, 222)

    def _kit(self, furnace_pos=None, at_base=False):
        stock = dict(crafting_table=1, coal=4, raw_iron=3, beef=2, furnace=1)
        g = at_home(geared(**stock), base=self.BASE) if at_base else gs(**{
            "inventory": dict(stock), "tools": {"pickaxe": "diamond", "sword": "diamond"},
            "armor": {s: {"item": f"diamond_{s}"}
                      for s in ("helmet", "chestplate", "leggings", "boots")},
            "offhand": "shield", "dimension": "minecraft:overworld",
            "health": 20.0, "food": 18.0})
        if not at_base:
            g.base = self.BASE
            g.pos = (150.0, 64.0, 150.0)      # a long way from home, mid-run
        g.furnace_pos = furnace_pos
        g.furnace_blocked = False
        g.phase = "day"
        # The frame stands: `_nether_idle` keys on it, and it is what makes the base furnace an
        # optimisation rather than chain work. Ten obsidian in the pack plus the frame plane, as
        # the live base at (226,63,222) has it.
        stock["obsidian"] = 10
        g.inventory["obsidian"] = 10
        # The frame plane sits two blocks east of the base, on z == bz - that is what
        # `_portal_frame_started` checks, and where the live base has it.
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", 228, y, 222, 2.0)
                                     for y in range(64, 70)],
                       "furnace": [BlockHit("furnace", "furnace", *self.FURNACE, 4.5)]}
        return g

    def test_a_cook_away_from_the_base_uses_the_carried_furnace(self) -> None:
        a = resolve("smelt_iron", self._kit())
        self.assertIsNotNone(a)
        self.assertTrue(a.args["temporary_workstation"])
        self.assertNotIn("x", a.args)         # no base cell named

    def test_a_cook_at_the_base_reuses_the_standing_furnace(self) -> None:
        a = resolve("smelt_iron", self._kit(furnace_pos=self.FURNACE, at_base=True))
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.FURNACE)
        self.assertNotIn("temporary_workstation", a.args)

    def test_the_drain_is_the_one_thing_that_needs_the_standing_furnace(self) -> None:
        # There is nothing at the base to drain when the furnace was carried and temporary.
        away = self._kit()
        self.assertIsNone(resolve("collect_smelted", away))
        home = self._kit(furnace_pos=self.FURNACE, at_base=True)
        home.furnace_occupied = True
        home.furnace_contents = {"cooked_beef": 1}
        self.assertIsNotNone(resolve("collect_smelted", home))

    def test_the_walk_home_is_owed_exactly_when_the_placement_can_follow_it(self) -> None:
        # The half-open contract this closes: `go_home` and `place_furnace` were two copies of one
        # question. Run53 carried a furnace for 341 records and was never offered the placement,
        # because the walk was owed on conditions the placement did not read.
        away = self._kit()
        self.assertIn("go_home", candidates(away))
        away.phase = "night"                     # `_nether_idle` still holds: it keys on the kit
        self.assertIn("place_furnace", candidates(self._kit(at_base=True)))
        # The lull closed: the chain owns the bot, so neither the walk nor the placement is asked.
        early = self._kit(at_base=True)
        early.armor = {}
        self.assertNotIn("place_furnace", candidates(early))
        self.assertNotIn("go_home", candidates(self._kit(furnace_pos=self.FURNACE)))


class LeavePortalDimensionTest(unittest.TestCase):
    """Stepping out of the Nether's own arrival portal must aim at a cell in the Nether.

    `base` is one field for both worlds, and `_leave_portal_target` preferred it as the natural
    step-out. In the Nether that sent the bot toward *Overworld* coordinates, which the pathfinder
    can never reach - a total, repeating refusal, on a step that owns the tick outright, so the
    expedition never started: `[NAVIGATION_STOPPED] ... no path found, visited 101 nodes` twenty
    times running (live, run53, arriving at (33,55,34)).
    """

    NETHER = "minecraft:the_nether"
    OVERWORLD = "minecraft:overworld"

    @staticmethod
    def _gate(pos, dim):
        g = gs(pos=pos, dimension=dim, base=(226, 63, 222))
        g.resources = {"nether_portal": [
            BlockHit("nether_portal", "nether_portal", 33, 55, 34, 0.0),
            BlockHit("nether_portal", "nether_portal", 32, 55, 34, 1.0),
            BlockHit("nether_portal", "nether_portal", 33, 56, 34, 1.0),
        ]}
        return g

    def test_the_nether_arrival_steps_out_beside_itself(self) -> None:
        target = _leave_portal_target(self._gate((33.43, 55.0, 34.5), self.NETHER))
        self.assertIsNotNone(target)
        # A neighbour of the arrival cell, never the 190-block-away Overworld base.
        self.assertLess(abs(target[0] - 33.5) + abs(target[2] - 34.5), 8.0)
        a = resolve("leave_portal", self._gate((33.43, 55.0, 34.5), self.NETHER))
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (33, 55, 32))

    def test_the_step_out_target_is_out_of_reach_of_standing_still(self) -> None:
        # The whole bug. `navigate` calls the goal reached inside `range` (one block by default),
        # so a target one cell out is satisfied by the body not moving at all: 299 then 345 ticks
        # of `leave_portal` answering ok with the body on the same obsidian inside the gate. The
        # target has to be further away than the arrival it is escaping.
        g = self._gate((33.43, 55.0, 34.5), self.NETHER)
        target = _leave_portal_target(g)
        start = g.pos
        self.assertGreater(abs(target[0] - start[0]) + abs(target[2] - start[2]), 1.0)
        # Measured, and worth pinning: two cells out is what actually walked the body clear of the
        # gate (z 34.5 -> 33.66, the portal blocks all sitting on z 34).
        self.assertEqual((int(target[0]), int(target[2])), (33, 32))

    def test_the_overworld_gate_still_steps_out_to_the_base(self) -> None:
        g = gs(pos=(226.5, 64.0, 222.5), dimension=self.OVERWORLD, base=(226, 63, 222))
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                   228, 64, 222, 2.0)]}
        self.assertEqual(_leave_portal_target(g), (226.5, 64, 222.5))


class ExpeditionSparePickaxeTest(unittest.TestCase):
    """The trip carries two pickaxes, and the gate stays shut until it does.

    The expedition digs: netherrack, obsidian for a return gate if the arrival portal is gone, the
    frame if it has to re-light on the far side. A pickaxe that breaks out there ends the trip
    where it stands, and `_can_recover` refuses the Nether on purpose, so nothing can be fetched
    from where it fell. Live: the pack held exactly one `diamond_pickaxe` with the trip stocked and
    the gate lit.
    """

    BASE = (-197, -47, -98)
    GATE = (-195, -46, -98)

    def _stocked(self, **inv) -> GameState:
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64,
                     golden_helmet=1, cooked_mutton=64, white_bed=1, diamond_pickaxe=1)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                   *self.GATE, 2.0)],
                       "obsidian": [BlockHit("obsidian", "obsidian", -195, -46, -98, 2.0)],
                       "white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -100, 2.5)]}
        g.dimension = "minecraft:overworld"
        return g

    def test_one_pickaxe_holds_the_gate_shut(self) -> None:
        g = self._stocked()
        self.assertNotIn("enter_portal", candidates(g))

    def test_a_stocked_trip_with_one_pickaxe_asks_for_the_second(self) -> None:
        g = self._stocked()
        self.assertIn("craft_pickaxe_spare", candidates(g))
        self.assertNotIn("enter_portal", candidates(g))

    def test_two_pickaxes_open_the_gate(self) -> None:
        g = self._stocked(diamond_pickaxe=2)
        self.assertIn("enter_portal", candidates(g))
        self.assertNotIn("craft_pickaxe_spare", candidates(g))

    def test_the_spare_is_offered_before_the_diamonds_are_in_the_pack(self) -> None:
        # The gate is already lit at that point, so the run stands at its own portal waiting. The
        # step has to outrank idling there, and `mine_diamond` then fills the gap.
        g = self._stocked(diamond=0)
        self.assertIn("craft_pickaxe_spare", candidates(g))
        self.assertIsNone(resolve("craft_pickaxe_spare", g))

    def test_the_spare_needs_three_diamonds_and_two_sticks(self) -> None:
        a = resolve("craft_pickaxe_spare", self._stocked(diamond=3, stick=4))
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_pickaxe"}])
        self.assertIsNone(resolve("craft_pickaxe_spare", self._stocked(diamond=2)))
        self.assertIsNone(resolve("craft_pickaxe_spare", self._stocked(diamond=3, stick=0)))

    def test_the_spare_is_not_asked_for_before_the_trip_is_stocked(self) -> None:
        # An unstocked trip has to be stocked first: the pickaxe is the last thing before the
        # crossing, not the first step of it.
        self.assertNotIn("craft_pickaxe_spare",
                         candidates(self._stocked(cobblestone=12)))

    def test_the_nether_is_never_asked_to_craft_one(self) -> None:
        g = self._stocked()
        g.dimension = "minecraft:the_nether"
        g.nether_entered = True
        self.assertNotIn("craft_pickaxe_spare", candidates(g))

    def _unlit(self, **inv):
        # The frame stands, the gate does not burn: that is the state `light_portal` decides on.
        g = self._stocked(**inv)
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -46, -98, 2.0)],
                       "white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -100, 2.5)]}
        return g

    def test_the_gate_does_not_light_without_the_spare(self) -> None:
        # This is the placement that matters, and the live run is what corrected it. The gate used
        # to light on a stocked trip that still lacked the second pickaxe, and a burning gate is an
        # open door the pathfinder walks like any other block - so the bot crossed *by accident*,
        # teleported mid-action, with one pickaxe and no crossing step at all (live, run53:
        # `cook_food` at it1782388155 in the Overworld, `the_nether` at it1782388156).
        # Keeping the gate dark until the kit is complete means any crossing carries the kit.
        self.assertNotIn("light_portal", candidates(self._unlit(diamond_pickaxe=1)))

    def test_the_gate_lights_once_the_spare_is_banked(self) -> None:
        self.assertIn("light_portal", candidates(self._unlit(diamond_pickaxe=2)))


    def test_the_chain_step_that_would_have_done_it_is_never_offered(self) -> None:
        # `craft_diamond_pickaxe` exists and would craft exactly this, but it is a chain step that
        # fires only while `tier < DIAMOND_PICK_TIER`, and nothing ever appended it to the offer
        # at all. So there was no path to a second diamond pickaxe before this.
        self.assertNotIn("craft_diamond_pickaxe",
                         candidates(self._stocked(diamond=3, stick=4)))


if __name__ == "__main__":
    unittest.main()
