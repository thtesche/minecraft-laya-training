"""Action binding: label -> (tool, args) for the reactions and the food/bed chain.

The M2d reactions (`collect_drops`, `recover_drops`, `replace_tool`) and the bed chain
(`craft_bed`, `sleep`) are event-driven and hard to reach on demand in a live world, so
their precondition guards and argument binding are pinned here.
"""
from __future__ import annotations

import inspect
import math
import re
import unittest
from types import SimpleNamespace

from mlaya.actions import (_all_nether_prep_steps, _barter_pearls_wanted, _bed_at_base,
                          _bed_cells, _bed_reachable, _bed_reach_stance, _bed_stance,
                          _click_gap, _crowding_hostiles, _diamond_reason, _expedition_prepared,
                          _eyes_owed, _frame_empty, _nether_haul_home, _nether_topup_wanted,
                          _nether_pearl_wave_wanted, _pearls_owed, _powder_target, _rods_target,
                          _frame_needs_obsidian, _gold_reserve_short, _known_footings,
                          _centre,
                          _hoard_surplus, _idle_off_base, _nether_prep_steps,
                         _pearl_wave_trip, _search_enderman_wanted, _safe_enderman_hunt, _safe_blaze_hunt, _safe_zombie_hunt,
                          _safe_spider_hunt, _safe_string_hunt,
                         _enderman_in_reach, _combat_blocks_carried, ENDERMAN_HUNT_BLOCKS,
                         _in_portal, _leave_portal_target,
                         _portal_lit, Action, _armor_to_wear, candidates,
                         _base_leveling_wanted, FLATTEN_BATCH, FLATTEN_RADIUS, FLATTEN_UP,
                         BASE_RADIUS, BED_SLEEP_CLICK, INVENTORY_NEAR_FULL, inventory_relief, SLEEP_OFF_BASE_STEP,
                         NETHER_COBBLE, NETHER_COBBLE_SLACK,
                         BARTER_GOLD_BUDGET, BARTER_MAX_ATTEMPTS, BARTER_PEARL_BATCH,
                         BARTER_PIGLIN_RANGE, GOLD_BARTER_RESERVE,
                         NETHER_PEARL_SEARCH_M, PEARL_WAVE_MAX_TRIPS,
                         DEFER_QUIET_TICKS, HUNT_BATCH, HUNT_HURRY, ZOMBIE_FLESH_BATCH, _hunt_count,
                           nether_expedition_complete, nether_expedition_open, NETHER_SEARCH_M,
                           NETHER_HOPELESS_TICKS, NETHER_GATE_APPROACH, resolve, stronghold_complete,
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


class UnarmedHuntGateTest(unittest.TestCase):
    """run83's three deaths: the hunt walks up to kill something, and at hunger 20 with bare
    hands that walk is the death. it85 ran night, `pickaxe: none`, and `hunt_food` and
    `hunt_wool` alone on the candidate list; the food hunt died at (-86,71,14) to a zombie.
    Offer and refusal read one rule (`_hunt_ok`): a weapon carried, or hunger genuinely
    under way. Spider string keeps its own guard - the bow's three strings are earned on
    exactly that walk, so gating it would deny the expedition its only way to arm itself.
    """

    MOBS = [("cow", "animal", 5, 9.0), ("sheep", "animal", 2, 10.0)]

    @staticmethod
    def _state(**kw) -> GameState:
        """Full hunger, a cow nearest and a sheep still inside the bed chain's reach."""
        return gs(mobs=UnarmedHuntGateTest.MOBS, **kw)

    def test_the_unarmed_food_hunt_refuses(self) -> None:
        g = self._state()
        self.assertIsNone(resolve("hunt_food", g))
        self.assertNotIn("hunt_food", candidates(g))

    def test_the_unarmed_wool_hunt_is_not_offered(self) -> None:
        # run83's it85 list: night and no weapon left `hunt_food` and `hunt_wool` as the
        # only offers - both are the walk that killed the bot.
        g = self._state(phase="night")
        self.assertIsNone(resolve("hunt_wool", g))
        self.assertNotIn("hunt_wool", candidates(g))

    def test_a_sword_lets_the_same_walk_be_made(self) -> None:
        g = self._state(tools={"sword": "iron"})
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")
        self.assertEqual(resolve("hunt_wool", g).args["mob_name"], "sheep")
        # The offer path is really reachable in this state, so its refusal unarmed (above)
        # is the gate working, not the state failing to ask for a hunt at all.
        self.assertIn("hunt_food", candidates(g))
        self.assertIn("hunt_wool", candidates(g))

    def test_a_bow_lets_the_same_walk_be_made(self) -> None:
        g = self._state(tools={"bow": "bow"})
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")

    def test_hunger_under_way_beats_the_missing_weapon(self) -> None:
        # Only starvation opens the exception, and only the eating reflex's own floor is
        # close enough for it: at 12 the walk is still refused, at 8 it is made.
        self.assertIsNone(resolve("hunt_food", self._state(food=12.0)))
        self.assertEqual(
            resolve("hunt_food", self._state(food=8.0)).args["mob_name"], "cow")


class FoodChainTest(unittest.TestCase):
    # The hunt's weapon gate (`_hunt_ok`) is not what these fixtures test - range and target
    # choice are - so every one carries an iron sword and only carries that.
    SWORD = {"sword": "iron"}

    def test_hunt_passive_animal(self) -> None:
        # Five cows in view: five drops at most, one a kill being the least a cow is sure to give.
        g = gs(mobs=[("cow", "animal", 5, 10.0)], tools=self.SWORD)
        a = resolve("hunt_food", g)
        self.assertEqual(a.args["mob_name"], "cow")
        self.assertEqual(a.args["drop_name"], "beef")
        self.assertEqual(a.args["count"], 5)
        g.mobs = [("cow", "animal", 12, 10.0)]
        self.assertEqual(resolve("hunt_food", g).args["count"], HUNT_BATCH)

    def test_the_hunt_asks_no_more_than_the_animals_in_view(self) -> None:
        # run89 it15: six chicken asked for, four in view at 47 m. `collect_mob_drop` has no
        # range limit and retargeted until it had six - 591 blocks, 178 s, 276 m from the start,
        # hunger 8 -> 5. The count stops at the animals the bot can see.
        g = gs(mobs=[("chicken", "animal", 4, 47.5)], tools=self.SWORD, food=17.0, health=20.0)
        self.assertEqual(resolve("hunt_food", g).args["count"], 4)

    def test_a_hungry_or_wounded_hunt_is_short(self) -> None:
        # Two drops are a meal; the next tick `eat_food` gets it (`_must_eat`).
        g = gs(mobs=[("cow", "animal", 12, 10.0)], tools=self.SWORD, food=8.0, health=20.0)
        self.assertEqual(resolve("hunt_food", g).args["count"], HUNT_HURRY)
        g.food, g.health = 17.0, 8.5
        self.assertEqual(resolve("hunt_food", g).args["count"], HUNT_HURRY)
        g.mobs = [("zombie", "hostile", 12, 10.0)]
        self.assertEqual(_hunt_count(g, "zombie"), HUNT_HURRY)
        g.food, g.health = 20.0, 20.0
        self.assertEqual(_hunt_count(g, "zombie"), ZOMBIE_FLESH_BATCH)

    def test_a_chicken_is_the_last_choice(self) -> None:
        # The server's hunger reflex never eats raw chicken, and a chicken gives one drop: a cow
        # in reach wins even when the chickens stand closer.
        g = gs(mobs=[("chicken", "animal", 4, 12.0), ("cow", "animal", 2, 40.0)],
               tools=self.SWORD)
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")
        g.mobs = [("chicken", "animal", 4, 12.0), ("cow", "animal", 2, 90.0)]
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "chicken")

    def test_hunt_lone_zombie_for_rotten_flesh(self) -> None:
        g = gs(mobs=[("zombie", "hostile", 1, 10.0)], health=20.0, tools=self.SWORD)
        a = resolve("hunt_food", g)
        self.assertEqual(a.args["mob_name"], "zombie")
        self.assertEqual(a.args["drop_name"], "rotten_flesh")
        # One animal standing within the melee band: one drop. run102 it43 asked for the whole
        # `ZOMBIE_FLESH_BATCH` of ten from that one zombie, and the server went looking for the
        # other nine - 614 blocks in 181 s, no swing, `[BODY_HELD]`.
        self.assertEqual(a.args["count"], 1)

    def test_the_zombie_batch_is_cut_by_the_zombies_in_reach(self) -> None:
        # run102 it43: one zombie at 24 m, 24 more loaded across the chunks. The batch is cut by
        # what stands within the melee band, not by the loaded count, because `collect_mob_drop`
        # retargets to the next animal of the kind anywhere loaded until the count is met.
        g = gs(mobs=[("zombie", "hostile", 1, 24.0), ("zombie", "hostile", 24, 60.0)],
               health=20.0, food=17.0, tools=self.SWORD)
        self.assertEqual(_hunt_count(g, "zombie"), 1)
        g.mobs = [("zombie", "hostile", 12, 12.0)]
        self.assertEqual(_hunt_count(g, "zombie"), ZOMBIE_FLESH_BATCH)

    def test_a_well_fed_bot_does_not_hunt_a_zombie_for_flesh(self) -> None:
        # Rotten flesh is edible but never a provision, so a zombie hunt asked for as a reserve
        # can never close the gap it was offered for. run102 offered `hunt_food` at hunger 17 with
        # nothing but a zombie in reach; the hunt roamed, the body stayed held, and the run ended
        # three ticks later on `body_held`.
        g = gs(mobs=[("zombie", "hostile", 1, 10.0)], health=20.0, food=17.0, tools=self.SWORD)
        self.assertIsNone(resolve("hunt_food", g))
        self.assertNotIn("hunt_food", candidates(g))
        g.food = 12.0
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "zombie")
        self.assertIn("hunt_food", candidates(g))

    def test_no_food_target(self) -> None:
        self.assertIsNone(resolve("hunt_food", gs()))

    def test_night_hunt_ignores_far_animal(self) -> None:
        """After dark a distant animal is not worth the walk (the old 180 m night roam)."""
        g = gs(mobs=[("cow", "animal", 5, 40.0)], phase="night", tools=self.SWORD)
        self.assertIsNone(resolve("hunt_food", g))

    def test_night_hunt_takes_close_animal(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 12.0)], phase="night", tools=self.SWORD)
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")

    def test_day_hunt_reaches_far_animal(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 40.0)], phase="day", tools=self.SWORD)
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "cow")

    def test_day_hunt_gives_up_beyond_range(self) -> None:
        g = gs(mobs=[("cow", "animal", 5, 80.0)], phase="day", tools=self.SWORD)
        self.assertIsNone(resolve("hunt_food", g))

    def test_night_hunt_boundary(self) -> None:
        self.assertIsNotNone(resolve("hunt_food", gs(mobs=[("pig", "animal", 1, 16.0)], phase="night", tools=self.SWORD)))
        self.assertIsNone(resolve("hunt_food", gs(mobs=[("pig", "animal", 1, 16.1)], phase="night", tools=self.SWORD)))

    def test_candidates_gate_hunt_food_by_range(self) -> None:
        far = gs(mobs=[("cow", "animal", 5, 40.0)], phase="night", tools=self.SWORD)
        self.assertNotIn("hunt_food", candidates(far))
        near = gs(mobs=[("cow", "animal", 5, 12.0)], phase="night", tools=self.SWORD)
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


class RecoveryTest(unittest.TestCase):
    """Wounded with nothing to heal on: the loop takes the tick back for food or a retreat.

    run88 from it64: cave-spider bites took the bot to 8.5 health beside a spawner, hunger 17,
    not one edible item all run. Below the health band every tick deferred to the survival
    policy, which had nothing to do - 536 defers, health never moved, the run ended standing.
    """

    @staticmethod
    def _hurt(**kw) -> GameState:
        g = geared(cobblestone=179, raw_iron=3, furnace=1, coal=3, spider_eye=2)
        g.health, g.food, g.phase = 8.5, 17.0, "day"
        g.pos = (-233.4, 4.0, 35.3)
        g.nearest_hostile = 12.9
        g.defer_quiet = DEFER_QUIET_TICKS
        for k, v in kw.items():
            setattr(g, k, v)
        return g

    def test_an_animal_in_reach_is_hunted(self) -> None:
        # it66: a chicken 44 m off, `hunt_food` on offer - and deferred with every other tick.
        g = self._hurt(mobs=[("chicken", "animal", 4, 44.0)])
        self.assertEqual(candidates(g), ["hunt_food"])
        self.assertEqual(resolve("hunt_food", g).tool, "collect_mob_drop")

    def test_no_animal_and_no_base_climbs_out(self) -> None:
        g = self._hurt(mobs=[("chicken", "animal", 3, 115.5)])
        self.assertEqual(candidates(g), ["surface_up"])
        self.assertIsNotNone(resolve("surface_up", g))

    def test_a_base_draws_the_bot_home(self) -> None:
        g = self._hurt(base=(-60, 84, 44))
        self.assertEqual(candidates(g), ["go_home"])
        self.assertEqual(resolve("go_home", g).tool, "navigate")

    def test_recovery_needs_no_defers_first(self) -> None:
        # run89 it0: the same state with the gate saying `no` - the ordinary offer
        # (`smelt_iron`, `hunt_wool`, `get_flint`) ran fifteen ticks while hunger fell to 5.
        # The quiet defers only gate the release from a defer, not the offer itself.
        g = self._hurt(defer_quiet=0)
        self.assertEqual(candidates(g), ["surface_up"])

    def test_food_in_the_pack_is_eaten_first(self) -> None:
        # run89 it16-17: six raw chicken, hunger 5, health 8.5 - and `surface_up` then
        # `establish_base` won over `eat_food`. The drowned on the second walk killed the bot
        # with the chicken still in the pack. Now the meal is the tick's only step.
        g = self._hurt(food=5.0)
        g.inventory["chicken"] = 6
        self.assertEqual(candidates(g), ["eat_food"])
        self.assertEqual(resolve("eat_food", g).args, {"food_name": "chicken"})

    def test_food_in_the_pack_is_no_recovery_case(self) -> None:
        # Eating heals: the meal, not the retreat.
        g = self._hurt()
        g.inventory["cooked_beef"] = 3
        self.assertEqual(candidates(g), ["eat_food"])

    def test_starving_reaches_food_past_the_day_cap(self) -> None:
        # run107: the kit lost to a Nether death, then 365 ticks ping-ponging `explore` <->
        # `go_home` at the base while the nearest animal (a pig at 113 m) sat past the 64 m day
        # cap - `_recovery_steps` found no food target and looped the body between two cells until
        # it starved (hunger 6 -> 0, health 12 -> 1). Below the starvation floor the cap gives way
        # so the one step that heals (a walk to the far animal) is taken instead of the loop.
        g = self._hurt(food=6.0, health=12.0, base=(-36, -14, 122),
                       mobs=[("pig", "animal", 2, 113.0), ("chicken", "animal", 7, 87.0)])
        self.assertEqual(candidates(g), ["hunt_food"])
        self.assertEqual(resolve("hunt_food", g).args["mob_name"], "pig")

    def test_merely_hungry_keeps_the_day_cap(self) -> None:
        # Above the starvation floor the thrift still holds: a pig 113 m off is not worth the walk,
        # so the bot retreats home rather than roam (the old far-roam stayed reverted).
        g = self._hurt(food=12.0, health=12.0, base=(-36, -14, 122),
                       mobs=[("pig", "animal", 2, 113.0)])
        self.assertEqual(candidates(g), ["go_home"])


class NetherLullWaitTest(unittest.TestCase):
    """The lull after the portal is lit and the bed laid: nothing productive is left, so the bot
    holds at the base instead of wandering off on `explore` (live, from-zero run: 25 idle ticks,
    five of them wasted on `[NAVIGATION_STOPPED]` pathfinder timeouts)."""

    @staticmethod
    def _idle() -> GameState:
        # The Nether prep is already banked (bow, arrows, deep cobblestone stack, and a furnace
        # standing at the base) and the bed stands beside the gate, so the lull has no productive
        # step left and the only sensible tick is holding at the base. A bed still carried is no
        # longer such a tick: `place_bed` lays it by day as well.
        g = lit_portal(at_home(geared(flint_and_steel=1, cobblestone=64,
                                      iron_ingot=1, cooked_beef=24,
                                      bow=1, arrow=16, diamond_pickaxe=2)))
        g.resources["white_bed"] = [BlockHit("white_bed", "white_bed", -200, -47, -98, 3.0)]
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

    def test_one_scaffold_block_does_not_reopen_the_dig(self) -> None:
        # run87 it177-179: the drop left exactly 64, the walk off it laid one scaffold block,
        # and `mine_stone` came up for the 63rd - its `collect_block` hoovered the dropped pile
        # straight back into the pack. A batch of slack keeps the two lines apart.
        g = self._lull(bow=1, arrow=16, cobblestone=NETHER_COBBLE - 1)
        self.assertNotIn("mine_stone", candidates(g))
        g = self._lull(bow=1, arrow=16, cobblestone=NETHER_COBBLE - NETHER_COBBLE_SLACK - 1)
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

    def test_the_gate_chain_is_off_the_offer_below(self) -> None:
        # run124 it40-41: `['fill_bucket', 'get_flint', 'search_enderman']` at (-6.5, 73, 265)
        # in the Nether. `_portal_lit` is a scan read, not a latch, so down there the base gate
        # reads unlit and the obsidian chain reopens for a frame that already stands - a bucket
        # with no water to fill and gravel that does not exist in that dimension.
        g = geared(bucket=1)
        g.base = (-197, -47, -98)
        g.pos = (-6.5, 73.0, 265.3)
        g.dimension = "the_nether"
        g.resources = {"gravel": [BlockHit("gravel", "gravel", -7, 73, 266, 2.0)]}
        cand = candidates(g)
        for label in ("fill_bucket", "craft_bucket", "get_flint", "mine_obsidian",
                      "approach_lava", "craft_flint_and_steel"):
            self.assertNotIn(label, cand)

    def test_the_same_pack_still_offers_the_chain_in_the_overworld(self) -> None:
        # The gate is scoped, not switched off: in the Overworld with the frame still short, the
        # bucket and the flint dig are exactly what the chain wants.
        g = geared(bucket=1)
        g.resources = {"gravel": [BlockHit("gravel", "gravel", -196, -47, -97, 3.0)]}
        cand = candidates(g)
        self.assertIn("fill_bucket", cand)
        self.assertIn("get_flint", cand)


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

    def test_a_hungry_bot_eats_before_the_reaction(self) -> None:
        # Hungry, the meal comes first and alone (`_must_eat`): a second on the meal costs the
        # drops nothing, and offered beside other steps it lost (run89 it16-17).
        g = self._recoverable()
        g.food = 10.0
        g.inventory = {"cooked_beef": 1}
        self.assertEqual(candidates(g), ["eat_food"])

    def test_eating_rides_along_when_the_need_is_mild(self) -> None:
        # Fed above the floor and healthy enough, the meal is one offer among the others.
        g = self._recoverable()
        g.food = 16.0
        g.health = 18.0
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

    def test_a_lost_shield_asks_for_its_planks(self) -> None:
        # The dead zone this closes, from run98's gate state: a geared bot with a table, one ingot
        # and five planks had `mine_iron`/`place_chest` on offer and no wood step, so `craft_shield`
        # could never run - while `_gear_ready` holds the Nether departure on exactly that shield.
        g = at_home(geared(oak_planks=5, iron_ingot=1))
        g.offhand = None  # the shield is what was lost
        self.assertNotIn("craft_shield", candidates(g))
        self.assertIn("get_wood", candidates(g))

    def test_one_log_is_short_of_the_shields_six_planks(self) -> None:
        # `logs == 0` is not the question here: one log is four planks, the shield wants six.
        g = self._ready(oak_planks=0, oak_log=1)
        self.assertNotIn("craft_shield", candidates(g))
        self.assertIn("get_wood", candidates(g))

    def test_two_logs_are_enough_so_no_walk(self) -> None:
        g = self._ready(oak_planks=0, oak_log=2)
        self.assertIn("craft_shield", candidates(g))
        self.assertNotIn("get_wood", candidates(g))

    def test_a_worn_shield_asks_for_no_wood(self) -> None:
        g = self._ready(oak_planks=0)
        g.offhand = "shield"
        self.assertNotIn("get_wood", candidates(g))

    def test_a_stone_pickaxe_asks_for_no_wood_for_the_shield(self) -> None:
        # The craft itself waits for the iron pickaxe; gathering for it earlier is a walk to a
        # craft the offer cannot satisfy.
        g = gs(tools={"pickaxe": "stone"},
               inventory={"crafting_table": 1, "iron_ingot": 1})
        self.assertNotIn("craft_shield", candidates(g))
        self.assertNotIn("get_wood", candidates(g))

    def test_the_nether_never_asks_for_wood_for_the_shield(self) -> None:
        g = self._ready(oak_planks=0)
        g.dimension = "minecraft:the_nether"
        self.assertNotIn("get_wood", candidates(g))

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
        `build_portal` leaves behind) and by the exact plane read naming all ten obsidian
        cells (`_frame_complete_at_base` -- the light step only burns on a frame the read
        proves finished, run80); the gate itself is still unlit, which is exactly the
        state `light_portal` is meant to run from.
        """
        base = {"flint_and_steel": 1, "bow": 1, "arrow": 16, "cobblestone": 64}
        base.update(inv)
        g = self._at_home(self._geared(**base))
        g.resources = {
            "obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)],
            "white_bed": [BlockHit("white_bed", "white_bed", -196, -47, -98, 1.0)],
        }
        g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)}
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

    def _full_pack_at_home(self) -> GameState:
        # run84 it8-15: a 36/36 pack carrying 129 cobblestone, the chest at the base, the lava
        # 27 blocks down - the walk to it succeeded every time and `go_home` called it back.
        g = self._at_home(self._geared(water_bucket=1, cobblestone=129))
        g.chest_pos = (g.base[0], g.base[1] + 1, g.base[2] + 1)
        g.slots_used = 36
        g.resources = self._lava(source=(-193, -73, -94), target=(-193, -73, -93),
                                 distance=27.0)
        return g

    def test_a_full_pack_stows_before_it_walks_to_the_lava(self) -> None:
        g = self._full_pack_at_home()
        offered = candidates(g)
        self.assertNotIn("mine_obsidian", offered)
        self.assertIsNone(resolve("mine_obsidian", g))
        self.assertIn("stow_surplus", offered)

    def test_the_lava_walk_returns_once_the_pack_has_room(self) -> None:
        g = self._full_pack_at_home()
        g.slots_used = 20
        self.assertIn("mine_obsidian", candidates(g))
        self.assertEqual(resolve("mine_obsidian", g).tool, "navigate")

    def test_a_full_chest_network_does_not_hold_the_obsidian_back(self) -> None:
        # Nothing to shed into: the surplus is no debt the walk could wait for.
        g = self._full_pack_at_home()
        g.chests_full = [g.chest_pos]
        self.assertIn("mine_obsidian", candidates(g))

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

    def test_approach_lava_walks_to_the_lava_when_no_source_is_loaded(self) -> None:
        # Live, run114 it159-225: the base anchor sat 118 blocks from the pond that made the run's
        # three obsidian, and the base scan listed only two *flows* (level 2 and 8) 60 m away - no
        # level-0 source. `mine_obsidian` closed, every other step was already satisfied, and the
        # offer fell to the `explore`/`go_home` fallback for 66 ticks with seven blocks still owed.
        # A source only enters the scan once the bot is beside it, so the chain needs the walk.
        g = self._geared(water_bucket=1, obsidian=3, flint=4, cobblestone=60,
                        coal=1, raw_iron=1, iron_ingot=1)
        g.base = (-8, 67, -55)
        g.pos = (-7.4, 68.0, -55.6)
        g.resources = {"lava": [
            BlockHit("lava", "lava", -11, 38, -4, 60.1, above="stone", below="lava", level=2),
            BlockHit("lava", "lava", -11, 37, -4, 60.6, above="lava", below="lava", level=8),
        ]}
        offered = candidates(g)
        self.assertIn("approach_lava", offered)
        self.assertNotIn("mine_obsidian", offered)
        a = resolve("approach_lava", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual(a.args["y"], 39)  # on top of a neighbour, never in the lava
        self.assertIn((a.args["x"], a.args["z"]), [(-10, -4), (-12, -4), (-11, -3), (-11, -5)])
        self.assertTrue(a.args["dig"] and a.args["scaffold"])

    def test_approach_lava_yields_to_a_source_in_the_listing(self) -> None:
        # With a pour-able source in the scan the walk is already what `mine_obsidian` does;
        # offering both would hand Laya two names for one move.
        g = self._geared(water_bucket=1)
        g.pos = (0.0, 64.0, 0.0)
        g.resources = self._lava()
        offered = candidates(g)
        self.assertIn("mine_obsidian", offered)
        self.assertNotIn("approach_lava", offered)
        self.assertIsNone(resolve("approach_lava", g))

    def test_approach_lava_needs_the_water_and_the_short_pack(self) -> None:
        g = self._geared(obsidian=3)
        g.pos = (-7.4, 68.0, -55.6)
        g.resources = {"lava": [BlockHit("lava", "lava", -11, 38, -4, 60.1,
                                         above="stone", below="stone", level=2)]}
        self.assertNotIn("approach_lava", candidates(g))
        g.inventory["water_bucket"] = 1
        g.inventory["obsidian"] = 10
        self.assertNotIn("approach_lava", candidates(g))

    def test_approach_lava_skips_a_stance_the_server_refused(self) -> None:
        # Every stance of that lava body answered `[NAVIGATION_TARGET_UNSUPPORTED]` once; walking
        # to it again is the run16 215-tick loop, so the step stays out entirely.
        g = self._geared(water_bucket=1, obsidian=3)
        g.pos = (-7.4, 68.0, -55.6)
        g.resources = {"lava": [BlockHit("lava", "lava", -11, 38, -4, 60.1,
                                         above="stone", below="stone", level=2)]}
        g.stance_skip = {(-10, 39, -4), (-12, 39, -4), (-11, 39, -3), (-11, 39, -5)}
        self.assertNotIn("approach_lava", candidates(g))
        self.assertIsNone(resolve("approach_lava", g))

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
        # The cell is named exactly - run113 it84 shows why: a bare count let the server re-target
        # a natural deposit 124 blocks away and the bot tread the water pocket for 14 minutes.
        g = self._geared(obsidian=4)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = self._blocks("obsidian")
        self.assertIn("mine_obsidian", candidates(g))
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "obsidian", "x": -270, "y": 64, "z": -56})

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
        self.assertEqual(a.args, {"block_name": "obsidian", "x": -270, "y": 64, "z": -56})

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

    def test_a_cell_already_holding_its_frame_block_is_not_dug(self) -> None:
        # run86: the BUILD_STOPPED marking read the plane from before the build, so the cells the
        # build had just filled were marked as well, and `clear_frame` dug the cobblestone corner
        # and two fresh obsidian back out. A cell holding what the frame wants there is done.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=6, cobblestone=4))
        corner, bottom = (-196, -47, -98), (-195, -47, -98)
        g.refused_frame = (corner, bottom)
        g.frame_names = {corner: "cobblestone", bottom: "obsidian"}
        self.assertNotIn("clear_frame", candidates(g))
        self.assertIsNone(resolve("clear_frame", g))

    def test_the_real_blocker_is_dug_past_the_finished_cells(self) -> None:
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=6, cobblestone=4))
        bottom, side = (-195, -47, -98), (-196, -46, -98)
        g.refused_frame = (bottom, side)
        g.frame_names = {bottom: "obsidian", side: "spruce_leaves"}
        a = resolve("clear_frame", g)
        self.assertEqual(a.args["block_name"], "spruce_leaves")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), side)

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

    def test_a_finished_frame_burns_by_day_too(self) -> None:
        # Live, run80: the last build stopped the frame 3 obsidian short for 350 ticks and closed
        # it too late in the day - the run ended before night came, beside a finished, fully
        # geared gate that never burned. With the bed carried the gate lights by day too, and
        # `place_bed` lays the bed the very next tick, whatever the hour.
        def carried(phase: str) -> GameState:
            g = self._at_home(self._geared(white_bed=1, flint_and_steel=1,
                                           bow=1, arrow=16, cobblestone=64))
            g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
            g.phase = phase
            g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)}
            return g

        day = carried("day")
        self.assertIn("light_portal", candidates(day))
        self.assertEqual(resolve("light_portal", day).tool, "activate_portal")
        night = carried("night")
        self.assertIn("light_portal", candidates(night))

    def test_light_portal_waits_while_the_frame_is_only_partly_built(self) -> None:
        # run80 from it215: a build had spent 7 of the ten obsidian and stopped, the pack still
        # carried 3, and the plane read named the missing cells dirt and spruce. Every
        # `activate_portal` on that short frame answered `[PORTAL_NO_FRAME]` - 22 attempts. The
        # "obsidian left the pack" signal was not enough: lighting now waits for the plane read
        # to prove the ten obsidian stand, whatever the hour.
        def partial(phase: str) -> GameState:
            g = self._at_home(self._geared(white_bed=1, flint_and_steel=1, obsidian=3,
                                           bow=1, arrow=16, cobblestone=64))
            g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
            g.phase = phase
            cells = frame_obsidian_cells(g.base)
            names = {cell: "obsidian" for cell in cells[:7]}
            names[cells[7]] = "dirt"
            names[cells[8]] = "spruce_log"
            names[cells[9]] = "dirt"
            g.frame_names = names
            return g

        for phase in ("day", "night"):
            g = partial(phase)
            self.assertNotIn("light_portal", candidates(g))
            self.assertIsNone(resolve("light_portal", g))

    def test_a_block_left_in_the_opening_holds_the_lighting_back(self) -> None:
        # Even a frame the plane read proves complete cannot burn while a stray block stands in
        # the 2x3 opening (a bed the base `sleep` laid into it): `activate_portal` reads the
        # opening with the gate and answers `[PORTAL_NO_FRAME]` forever. The clear steps dig it
        # in front of the light; while `gs.portal_blocked` holds a cell, the light says nothing.
        g = self._expedition()
        g.phase = "day"
        self.assertIn("light_portal", candidates(g))
        g.portal_blocked = [(-195, -46, -98)]
        self.assertNotIn("light_portal", candidates(g))
        self.assertIsNone(resolve("light_portal", g))

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

    def test_place_bed_by_day_sets_the_spawn(self) -> None:
        # By day the `sleep` server lays the carried bed and uses it instead of sleeping, which
        # sets the respawn point all the same ("Respawn point set at the bed, but daytime prevents
        # sleeping"). Held back for the night, run86 stood ~300 `wait` ticks beside a gate lit by
        # day with nothing else left to do.
        g = self._at_home(self._geared(white_bed=1))
        g.resources = self._blocks("nether_portal")
        g.phase = "day"
        self.assertIn("place_bed", candidates(g))
        a = resolve("place_bed", g)
        self.assertEqual(a.tool, "sleep")

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
        # A carried bed and the prep banked: the *gate* is offered, not the bed. Once it
        # lights the opening is solid, so `place_bed` lays the bed safely the very next tick and
        # the run stops - the burning gate is never left standing while the bot works.
        g = self._at_home(self._geared(white_bed=1, flint_and_steel=1, bow=1, arrow=16,
                                       cobblestone=64))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)}
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

    def _run120_base(self) -> GameState:
        """The base run120 actually had, read live: the gate lit at (-8..-7, 70-71, -52), the bot
        on the base anchor cell itself, the bed in the pack, and the ground scan naming every
        column beside it - including the dug-out one at z=-53, whose floor sits a block lower.
        """
        g = self._geared(white_bed=1, cobblestone=64)
        g.base = (-10, 69, -52)
        g.pos = (-9.5, 70.0, -51.5)
        g.phase = "night"
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal", x, y, -52, 2.0)
                                         for x in (-8, -7) for y in (70, 71)]}
        g.rescue_ground = {
            "grass_block": [BlockHit("grass_block", "grass_block", x, y, z, 1.0, above="air")
                           for (x, y, z) in ((-11, 69, -52), (-12, 69, -52), (-10, 69, -51),
                                            (-10, 69, -50), (-11, 69, -50), (-12, 69, -51),
                                            (-13, 69, -52), (-11, 68, -53))],
            "cobblestone": [BlockHit("cobblestone", "cobblestone", -9, 69, -53, 2.4, above="air"),
                            BlockHit("cobblestone", "cobblestone", -11, 69, -54, 2.2, above="air")],
        }
        return g

    def test_place_bed_stance_is_ground_the_bot_can_stand_on(self) -> None:
        # run120 it11-15: the bot stood on the base anchor cell, the gate reached it, and the
        # stance the offsets offered was one cell west - a column the run had dug out, whose
        # floor sits at y=68. `navigate` to a cell with nothing under it is refused outright
        # ("[NAVIGATION_TARGET_UNSUPPORTED] ... the ground in that column is at y=69"), so the
        # bed never went down and the run could not finish. The stance is a cell the ground read
        # says is standable, and the dug-out column is not one of them.
        g = self._run120_base()
        stance = _bed_stance(g)
        self.assertEqual(stance, (-11, 70, -52))
        self.assertIn(stance, _known_footings(g))

    def test_place_bed_walk_demands_the_exact_stance_cell(self) -> None:
        # The other half of run120: the stance was standable, but `navigate`'s default arrival
        # tolerance of one block reported success from the cell next to it, so `_at_stance`
        # stayed false and five identical `navigate` ticks laid nothing. The bed goes down where
        # the bot stands, so the walk has to arrive *in* the stance cell.
        g = self._run120_base()
        a = resolve("place_bed", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-11, 70, -52))
        self.assertEqual(a.args["range"], 0)

    def test_place_bed_lays_the_bed_once_the_bot_is_in_the_stance(self) -> None:
        # The tick after the walk: the feet are in the stance cell, so the bed goes down.
        g = self._run120_base()
        g.pos = (-10.47, 70.0, -51.5)
        self.assertEqual(resolve("place_bed", g).tool, "sleep")

    def test_place_bed_stance_skips_a_column_with_no_floor(self) -> None:
        # Nothing at the anchor's level is standable in the nearest clear column, so the stance
        # takes the next clear cell the ground read proves out, rather than a cell in mid-air.
        g = self._at_home(self._geared(white_bed=1), base=(0, 69, 0))
        g.pos = (0.5, 70.0, 0.5)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal", x, y, 0, 2.0)
                                         for x in (2, 3) for y in (70, 71)]}
        g.rescue_ground = {"grass_block": [
            BlockHit("grass_block", "grass_block", -1, 69, 1, 1.5, above="air"),
            BlockHit("grass_block", "grass_block", -2, 69, 0, 2.0, above="air"),
        ]}
        stance = _bed_stance(g)
        self.assertEqual(stance, (-1, 70, 1))

    def test_place_bed_stance_falls_back_when_no_ground_was_read(self) -> None:
        # A state with no ground scan at all (a failed read, or a test fixture) still gets the
        # clearance-only stance, so the bed chain is not held hostage to the footing read.
        g = self._at_home(self._geared(white_bed=1), base=(226, 63, 222))
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  x, y, 222, 3.0)
                                         for x in (228, 229) for y in (64, 65, 66)]}
        g.pos = (227.5, 64.0, 221.5)
        self.assertEqual(_bed_stance(g), (225, 64, 222))

    def _run121_base(self, pos: tuple = (-9.3, 70.0, -52.5)) -> GameState:
        """The base run121 actually had, read live: the gate lit at (-8..-7, 70-71, -52), the
        gate bed standing at (-10,70,-51)/(-10,70,-50), the cobblestone floor the run laid at
        y=69, and the bot two cells off the bed.
        """
        g = self._geared()
        g.base = (-10, 69, -52)
        g.pos = pos
        g.phase = "night"
        g.resources = {
            "nether_portal": [BlockHit("nether_portal", "nether_portal", x, y, -52, 2.0)
                              for x in (-8, -7) for y in (70, 71)],
            "white_bed": [BlockHit("white_bed", "white_bed", -10, 70, -51, 1.4, above="cobblestone"),
                          BlockHit("white_bed", "white_bed", -10, 70, -50, 1.6, above="air")],
        }
        g.rescue_ground = {"cobblestone": [
            BlockHit("cobblestone", "cobblestone", x, 69, z, 1.0, above="air")
            for x in (-11, -10, -9) for z in (-52, -51, -50)
            if (x, z) not in ((-10, -51), (-10, -50))]}
        return g

    def test_sleep_walks_onto_a_bed_it_cannot_reach(self) -> None:
        # run121 it100-103: four identical `[BED_SLEEP_REJECTED] bot is not sleeping` at
        # (-9.3,70,-52.5) against a bed at (-10,70,-51). The server's bed search is 32 blocks, so
        # `_sleep_ready` was satisfied; Minecraft takes the bed click only from within two blocks
        # of the bed's top centre, and the body was 5.04 squared away. `sleep` now walks onto a
        # footing beside the bed before it clicks.
        g = self._run121_base()
        self.assertIn("sleep", candidates(g))
        self.assertFalse(_bed_reachable(g))
        a = resolve("sleep", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual(a.args["range"], 0)

    def test_the_sleep_walk_ends_beside_the_bed(self) -> None:
        # The tick after the walk: the feet are on a cell inside the click box, so the sleep runs.
        g = self._run121_base()
        stance = _bed_reach_stance(g)
        self.assertIsNotNone(stance)
        g.pos = (stance[0] + 0.5, stance[1], stance[2] + 0.5)
        self.assertTrue(_bed_reachable(g))
        self.assertEqual(resolve("sleep", g).tool, "sleep")

    def test_the_sleep_stance_is_ground_the_scan_proved(self) -> None:
        # `navigate` refuses a column with no floor outright, so the stance is a cell the ground
        # read says is standable, not the nearest cell next to the bed whatever is under it.
        g = self._run121_base()
        stance = _bed_reach_stance(g)
        self.assertIn(stance, _known_footings(g))
        self.assertNotIn(stance, _bed_cells(g))

    def test_the_sleep_stance_skips_a_column_the_run_dug_out(self) -> None:
        # The cell nearest the bed has no floor at the bed's level - the run dug that column out -
        # and `navigate` to a column with nothing under it is refused outright. The stance takes
        # the next cell the ground read proved, still inside the click box.
        g = self._run121_base()
        g.rescue_ground = {"cobblestone": [
            BlockHit("cobblestone", "cobblestone", x, 69, z, 1.0, above="air")
            for x in (-11, -10, -9) for z in (-52, -51, -50)
            if (x, z) not in ((-10, -51), (-10, -50), (-10, -52))]}
        stance = _bed_reach_stance(g)
        self.assertNotEqual(stance, (-10, 70, -52))
        self.assertIn(stance, _known_footings(g))
        self.assertTrue(_click_gap(_centre(stance), _bed_cells(g)) <= BED_SLEEP_CLICK ** 2)

    def test_the_sleep_stance_falls_back_when_no_ground_was_read(self) -> None:
        # No footing read at all: the ring of cells around the bed is the only information there
        # is, and the nearest of them is tried rather than giving up on the night.
        g = self._run121_base()
        g.rescue_ground = {}
        stance = _bed_reach_stance(g)
        self.assertIsNotNone(stance)
        self.assertNotIn(stance, _bed_cells(g))

    def test_a_bed_out_of_scan_range_is_not_crafted_again(self) -> None:
        # run121 it80: the bot stood at (228,72,-336), 280 m from the base, and the base chunks
        # were unloaded, so the scan saw no bed and `craft_bed` made a second one. The remembered
        # gate bed answers "is a bed owed" when the world cannot be seen.
        g = self._geared(white_wool=4, oak_log=2)
        g.base = (-10, 69, -52)
        g.pos = (228.5, 72.0, -336.4)
        g.bed_pos = (-10, 70, -51)
        g.resources = {}
        self.assertTrue(_bed_at_base(g))
        self.assertNotIn("craft_bed", candidates(g))

    def test_a_bed_taken_up_is_forgotten(self) -> None:
        # `take_bed` removes the block, so a remembered cell would keep `_bed_at_base` true over
        # empty ground and close the completion gate on a respawn point that no longer exists.
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent.bed_pos = (-10, 70, -51)
        agent._note_bed(SimpleNamespace(label="take_bed", tool="collect_block", args={}),
                        True, None)
        self.assertIsNone(agent.bed_pos)

    def test_the_sleep_response_names_the_bed_it_clicked(self) -> None:
        # The bed block is the respawn point, and the `sleep` answer is the only place the server
        # ever names it -- a failed sleep included, since it reports the cell it tried to use.
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent.bed_pos = None
        res = SimpleNamespace(data={"response": {"data": {"output": {"result": {
            "sleep": {"asleep": False, "respawnSet": False,
                      "bedPosition": {"x": -10, "y": 70, "z": -51}}}}}}})
        agent._note_bed(SimpleNamespace(label="place_bed", tool="sleep", args={}), False, res)
        self.assertEqual(agent.bed_pos, (-10, 70, -51))

    def test_a_bed_at_the_base_is_adopted_from_the_world(self) -> None:
        # A restart loses `bed_pos` with the base, but the bed is a real block: the tick the bot
        # walks back into its chunks, the scan names it and the respawn point is known again.
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent.base = (-10, 69, -52)
        agent.bed_pos = None
        agent._adopt_bed(self._run121_base())
        self.assertEqual(agent.bed_pos, (-10, 70, -51))

    def test_a_bed_the_base_scan_disagrees_with_is_forgotten(self) -> None:
        # The bot is standing at the base, the scan ran, and no bed is there: the remembered cell
        # is stale and has to go, or `sleep` keeps refusing and the run never ends.
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent.base = (-10, 69, -52)
        agent.bed_pos = (-10, 70, -51)
        g = self._run121_base()
        g.resources = {"nether_portal": g.resources["nether_portal"],
                       "stone": [BlockHit("stone", "stone", -12, 66, -51, 4.0)]}
        agent._adopt_bed(g)
        self.assertIsNone(agent.bed_pos)

    def test_a_bed_out_of_the_base_box_is_not_forgotten(self) -> None:
        # A bot 280 m off base cannot see its own bed either; forgetting there would throw away
        # the only record of the respawn point the run has.
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent.base = (-10, 69, -52)
        agent.bed_pos = (-10, 70, -51)
        g = self._geared()
        g.base = (-10, 69, -52)
        g.pos = (228.5, 72.0, -336.4)
        g.resources = {"stone": [BlockHit("stone", "stone", 226, 63, -336, 2.0)]}
        agent._adopt_bed(g)
        self.assertEqual(agent.bed_pos, (-10, 70, -51))

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
        # `[PORTAL_NO_FRAME]` on every tick. So at the base `sleep` only walks off it, west and
        # away from the frame; the bed goes down where the bot stands next tick.
        g = self._at_home(self._geared(white_bed=1))
        g.phase = "night"
        a = resolve("sleep", g)
        self.assertEqual(a.tool, "navigate")
        bx, _, bz = g.base
        self.assertLessEqual(a.args["x"], bx - BASE_RADIUS - 1)
        self.assertEqual(a.args["z"], bz)

    def test_a_night_at_the_base_walks_off_it_to_sleep(self) -> None:
        # run87 it147-158: night, the frame up but unlit (the spare pickaxe still owed), no
        # diamond ore in view and the bed in the pack. Sleeping at the base is forbidden and
        # nothing walked the bot anywhere else, so the night offered `wait` alone for twelve
        # ticks. Now `sleep` is offered and walks off the base first; off it, it sleeps.
        g = self._at_home(self._geared(white_bed=1, flint_and_steel=1, cobblestone=64,
                                       diamond_pickaxe=1))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)}
        g.phase = "night"
        self.assertNotIn("light_portal", candidates(g))
        self.assertIn("sleep", candidates(g))
        self.assertNotIn("wait", candidates(g))
        self.assertEqual(resolve("sleep", g).tool, "navigate")
        bx, by, bz = g.base
        g.pos = (bx - SLEEP_OFF_BASE_STEP + 0.5, by + 1.0, bz + 0.5)
        self.assertIn("sleep", candidates(g))
        self.assertEqual(resolve("sleep", g).tool, "sleep")

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
        # A leftover on open ground off the base waits for the lull; only a bed underground is
        # fetched at once (`test_a_cave_bed_is_fetched_at_once`).
        g = self._geared()
        g.base = (-197, -47, -98)
        g.pos = (-196.5, -46.0, -98.5)
        g.resources = {"white_bed": [BlockHit("white_bed", "white_bed", -230, -46, -98, 33.0)]}
        self.assertNotIn("take_bed", candidates(g))
        self.assertIsNone(resolve("take_bed", g))

    def test_a_cave_bed_is_fetched_at_once(self) -> None:
        # run87 it193-200: the night sleep laid the bed at y=-10 in a zombie cave, the game
        # refused the sleep but kept the respawn there, and from it198 `take_bed` stood beside
        # `drop_junk` and `craft_bow` while Laya picked the junk. A bed that far below the base
        # is now the tick's only step (eating aside), frame or no frame.
        g = self._geared(cobblestone=40)
        g.base = (-60, 84, 44)
        g.pos = (-60.5, -5.0, 22.5)
        g.resources = {"white_bed": [BlockHit("white_bed", "white_bed", -64, -10, 58, 36.0)]}
        self.assertEqual(candidates(g), ["take_bed"])
        a = resolve("take_bed", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "white_bed", "count": 1})

    def test_no_night_sleep_with_a_monster_at_hand(self) -> None:
        # The game refuses the sleep with hostiles near the bed but sets the respawn first, so
        # the bed is not laid at all while one is within `SLEEP_HOSTILE_CLEAR`.
        g = self._geared(white_bed=1)
        g.base = (-60, 84, 44)
        g.pos = (-120.5, 75.0, 40.5)
        g.phase = "night"
        g.nearest_hostile = 6.0
        self.assertNotIn("sleep", candidates(g))
        self.assertIsNone(resolve("sleep", g))
        g.nearest_hostile = 20.0
        self.assertIn("sleep", candidates(g))
        self.assertEqual(resolve("sleep", g).tool, "sleep")

    def test_no_night_sleep_underground(self) -> None:
        # run87 it184: the night found the bot at y=-1 after a hunt; the bed went down in the
        # cave. Below the surface the night changes nothing, and the bed would only move the
        # respawn into the rock.
        g = self._geared(white_bed=1)
        g.base = (-60, 84, 44)
        g.pos = (-64.5, -10.0, 57.5)
        g.phase = "night"
        self.assertNotIn("sleep", candidates(g))
        self.assertIsNone(resolve("sleep", g))

    def test_a_cave_bed_in_the_search_keeps_the_sleep_off(self) -> None:
        # The server sleeps in a standing bed before it lays a carried one, so a cave bed within
        # its 32 blocks would take the surface sleep back underground.
        g = self._geared(white_bed=1)
        g.base = (-60, 84, 44)
        g.pos = (-64.5, 70.0, 57.5)
        g.phase = "night"
        g.resources = {"white_bed": [BlockHit("white_bed", "white_bed", -64, 50, 58, 20.0)]}
        self.assertNotIn("sleep", candidates(g))

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

    def test_a_carried_chest_waits_for_a_filling_pack_before_it_walks_the_bot_home(self) -> None:
        # Live, run99 it153-165: a bot on 29 of the 36 slots left the base for the trip's food,
        # and the carried chest drew it straight back every time it reached the animals -
        # `go_home` and `hunt_food` swapped the same 70 m four times over. The chest is placed
        # for the room it frees, so a light pack owes it no walk; once the pack fills, it does.
        g = self._geared(chest=1)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        g.slots_used = INVENTORY_NEAR_FULL - 4
        self.assertNotIn("go_home", candidates(g))
        self.assertIsNone(resolve("go_home", g))
        g.slots_used = INVENTORY_NEAR_FULL
        self.assertIn("go_home", candidates(g))
        self.assertIsNotNone(resolve("go_home", g))

    def test_a_carried_chest_is_still_placed_at_the_base_with_a_light_pack(self) -> None:
        # The walk is deferred, not the placement: a bot that is home with a light pack still
        # gets the chest set down, whatever made it come home (run99 laid it at it167).
        g = self._home(self._geared(chest=1))
        g.slots_used = INVENTORY_NEAR_FULL - 4
        self.assertIn("place_chest", candidates(g))

    def test_other_base_work_still_walks_a_lightly_packed_bot_home(self) -> None:
        # Only the chest reason changed: a frame waiting to be raised pulls the same light pack
        # home.
        g = self._geared(flint_and_steel=1, obsidian=10, cobblestone=4)
        g.base = (-197, -47, -98)
        g.pos = (-250.5, 64.0, -120.5)
        g.slots_used = INVENTORY_NEAR_FULL - 4
        self.assertIn("go_home", candidates(g))

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


class StorageNetworkTest(unittest.TestCase):
    """run79's storage bottleneck: one chest at the base, and once it was full nothing could be
    shed anymore -- `[CONTAINER_BATCH_INCOMPLETE]` on every side, a 36/36 pack and 102 `wait`
    ticks. A network grows: the server's full message marks *which* chest closed, the deposit
    aims at the next one, and another chest is crafted and stood beside the base once every
    one of them is full. The banked provisions are looked for in whichever chest holds them,
    not in the main chest by assumption.
    """

    BASE = (-197, -47, -98)
    MAIN = (-197, -46, -97)            # (bx, by+1, bz+1), the main chest's cell
    OVERFLOW = (-197, -46, -95)        # the first free candidate: (0, 3) behind the main chest
    SECOND = (-198, -46, -96)          # the next candidate after that: (-1, 2)

    @staticmethod
    def _network(chests_full: frozenset, **inv) -> GameState:
        """A bot on its base with the main chest placed, the given cells called full, and the
        pack as named (the geared fixture already carries a crafting table)."""
        g = at_home(geared(**inv))
        g.chest_pos = StorageNetworkTest.MAIN
        g.chests_full = chests_full
        return g

    def test_a_full_main_chest_deposits_into_the_overflow_one(self) -> None:
        g = self._network(frozenset({StorageNetworkTest.MAIN}), tuff=64)
        g.extra_chests = (StorageNetworkTest.OVERFLOW,)
        g.slots_used = 36
        self.assertIn("stow_surplus", candidates(g))
        a = resolve("stow_surplus", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "deposit")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), StorageNetworkTest.OVERFLOW)

    def test_the_roomy_chest_across_the_base_is_walked_to_first(self) -> None:
        # Farther off than `use_container` reaches: the label walks to the chest under its own
        # name, the deposit runs once the bot stands beside it.
        g = self._network(frozenset({StorageNetworkTest.MAIN}), tuff=64)
        g.extra_chests = (StorageNetworkTest.OVERFLOW,)
        g.pos = (-190.5, -46.0, -90.5)
        a = resolve("stow_surplus", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), StorageNetworkTest.OVERFLOW)

    def test_a_chest_adopted_outside_the_base_works_from_its_lid(self) -> None:
        # run82's dead loop: the chest had been adopted eight blocks outside the base's
        # radius, so the walk over was offered at the anchor and the deposit nowhere else,
        # while the walk back was justified by the very surplus the lid waited for - 481
        # `go_home` and `stow_surplus` walks between (-45,71,15) and (-43,71,21), each one
        # "ok", a 35-slot pack never shedding a block. Standing at the chest within reach,
        # the deposit is what has to run; carrying the surplus back to the anchor is not.
        g = self._network(frozenset(), tuff=64)
        g.chest_pos = (-197, -46, -106)        # nine blocks off the anchor, out of its radius
        g.pos = (-196.6, -46.2, -105.6)        # at the lid, and not at the base
        g.slots_used = 35
        self.assertIn("stow_surplus", candidates(g))
        a = resolve("stow_surplus", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "deposit")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-197, -46, -106))
        self.assertNotIn("go_home", candidates(g))

    def test_surplus_still_justifies_the_walk_home_when_no_lid_is_in_reach(self) -> None:
        # The other anchor of the same pair: away from base and chest alike the walk back
        # stays worth it - the deposit is not within reach yet.
        g = self._network(frozenset(), tuff=64)
        g.chest_pos = (-197, -46, -106)
        g.pos = (-189.5, -46.0, -103.5)
        g.slots_used = 35
        self.assertIn("go_home", candidates(g))
        self.assertNotIn("stow_surplus", candidates(g))

    def test_a_full_distant_chest_still_sends_the_bot_home(self) -> None:
        # Its lid declared full, the relief is growing the network elsewhere and the surplus
        # is not stowable right here, so the walk back to the base keeps its justification.
        g = self._network(frozenset(), tuff=64)
        g.chest_pos = (-197, -46, -106)
        g.chests_full = frozenset({(-197, -46, -106)})
        g.pos = (-196.6, -46.2, -105.6)
        g.slots_used = 35
        self.assertNotIn("stow_surplus", candidates(g))
        self.assertIn("go_home", candidates(g))

    def test_a_full_network_asks_for_another_chest(self) -> None:
        # run79's deadlock offer: no room left anywhere, but a crafting table and eight planks
        # -- and the relief that could only ever have dropped is now a chest under construction.
        g = self._network(frozenset({StorageNetworkTest.MAIN}), spruce_planks=8)
        g.slots_used = 36
        self.assertIn("craft_chest", candidates(g))
        self.assertNotIn("stow_surplus", candidates(g))
        a = resolve("craft_chest", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "chest"}])

    def test_the_overflow_chest_is_stood_behind_the_main_one(self) -> None:
        g = self._network(frozenset({StorageNetworkTest.MAIN}), chest=1)
        self.assertIn("place_chest", candidates(g))
        a = resolve("place_chest", g)
        self.assertEqual(a.tool, "place_block")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), StorageNetworkTest.OVERFLOW)

    def test_cells_taken_or_refused_are_passed_over(self) -> None:
        # The first candidate already holds the run's second chest, the next one the ground
        # refused: the third candidate gets the third chest, not a refusal on every tick.
        g = self._network(frozenset({StorageNetworkTest.MAIN, StorageNetworkTest.OVERFLOW}),
                          chest=1)
        g.extra_chests = (StorageNetworkTest.OVERFLOW,)
        g.overflow_skip = frozenset({StorageNetworkTest.SECOND})
        a = resolve("place_chest", g)
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-196, -46, -96))

    def test_no_new_chest_while_a_network_chest_still_has_room(self) -> None:
        # One full chest beside one that is roomy is a deposit waiting, not a furniture project.
        g = self._network(frozenset({StorageNetworkTest.MAIN}), spruce_planks=8)
        g.extra_chests = (StorageNetworkTest.OVERFLOW,)
        self.assertNotIn("craft_chest", candidates(g))
        self.assertNotIn("place_chest", candidates(g))
        self.assertIsNone(resolve("craft_chest", g))
        self.assertIsNone(resolve("place_chest", g))

    def test_relief_builds_the_next_chest_when_the_network_is_full(self) -> None:
        # The run79 bottleneck with materials in the pack: neither deposit nor drop frees a
        # slot, but the chest does. `allow_stow` off -- the relief's own name for "the chest
        # is full" -- used to be a hard fall to `drop_junk`.
        g = self._network(frozenset({StorageNetworkTest.MAIN}), spruce_planks=8)
        g.slots_used = 36
        self.assertEqual(inventory_relief(g).tool, "craft_item")
        self.assertEqual(inventory_relief(g, allow_stow=False).tool, "craft_item")
        g2 = self._network(frozenset({StorageNetworkTest.MAIN}), chest=1)
        g2.slots_used = 36
        self.assertEqual(inventory_relief(g2).tool, "place_block")

    def test_provisions_standing_in_the_overflow_are_fetched_there(self) -> None:
        # `banked` aggregates the network -- and that is exactly the trap: provisions the main
        # chest does not hold cannot be withdrawn from its cell, and the aggregate does not say
        # which lid they stand behind. `banked_stores` does, and `take_stored_food` reaches in.
        g = at_home(geared())
        g.food = 12.0
        g.phase = "day"
        g.chest_pos = self.MAIN
        g.banked = {"cooked_beef": 16}
        g.banked_stores = ((self.OVERFLOW, "cooked_beef", 16),)
        self.assertIn("take_stored_food", candidates(g))
        a = resolve("take_stored_food", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "withdraw")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.OVERFLOW)
        self.assertEqual(a.args["items"], [{"item_name": "cooked_beef", "count": 16}])

    def test_a_store_far_off_is_walked_to_before_it_is_reached_into(self) -> None:
        g = at_home(geared())
        g.food = 12.0
        g.phase = "day"
        g.chest_pos = self.MAIN
        far = (40, -46, 30)
        g.banked = {"cooked_beef": 8}
        g.banked_stores = ((far, "cooked_beef", 8),)
        a = resolve("take_stored_food", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), far)

    def test_the_main_chest_leads_when_both_hold_provisions(self) -> None:
        # The bot stands at its base most of the time; the shortest walk is the one it skips.
        g = at_home(geared())
        g.food = 12.0
        g.phase = "day"
        g.chest_pos = self.MAIN
        g.banked = {"cooked_beef": 16}
        g.banked_stores = ((self.OVERFLOW, "cooked_beef", 8), (self.MAIN, "cooked_beef", 8))
        a = resolve("take_stored_food", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.MAIN)

    def test_a_store_of_junk_only_is_not_a_larder(self) -> None:
        # The run78 shuttle in its per-cell shape: a network chest full of rotten flesh is not
        # a provision to fetch, wherever it stands.
        g = at_home(geared())
        g.food = 12.0
        g.phase = "day"
        g.chest_pos = self.MAIN
        g.banked = {"rotten_flesh": 96}
        g.banked_stores = ((self.OVERFLOW, "rotten_flesh", 96),)
        self.assertNotIn("take_stored_food", candidates(g))
        self.assertIsNone(resolve("take_stored_food", g))


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


class CraftFurnaceReachTest(unittest.TestCase):
    """`craft_furnace` is gated on a furnace being *reachable*, not merely remembered.

    Live run116: the bot died at stone tier, respawned, and `explore`-drifted ~650 m from its old
    base. The server still remembered the base furnace (`furnace_pos`), so the old flag
    `_has(inv,'furnace') or _furnace_ready(gs)` read True and suppressed `craft_furnace` -- yet
    `smelt_iron` needs `_at_base`, which a drifted bot (`base None`) never meets. No furnace, no
    smelt, every tech step blocked, endless `explore`. The flag now mirrors `_smelting_reach`, so
    the craft is offered exactly when there is no furnace to smelt at.
    """

    def _drifted(self) -> GameState:
        # The it114-149 state, to the field: stone tier, cobble to spare, a crafting table, raw
        # iron and coal, but drifted far from base (`base None`) with only the old base furnace
        # remembered.
        g = gs(tools={"pickaxe": "stone"}, dimension="overworld",
               inventory={"cobblestone": 45, "crafting_table": 1, "raw_iron": 3,
                          "coal": 3, "stone_pickaxe": 2, "wooden_pickaxe": 2})
        g.pos = (32.5, 62.0, -706.5)
        g.base = None
        g.furnace_pos = (-6, 68, -56)   # remembered old base, ~650 m off
        g.chest_pos = (-8, 68, -54)
        return g

    def test_a_drifted_bot_crafts_a_furnace_it_cannot_reach_the_remembered_one(self) -> None:
        g = self._drifted()
        cands = candidates(g)
        self.assertIn("craft_furnace", cands)
        # And it cannot smelt at the unreachable base, so the two stay consistent.
        self.assertNotIn("smelt_iron", cands)

    def test_the_base_furnace_is_reused_not_recrafted(self) -> None:
        # Standing at the base with a furnace remembered there: no new craft, smelt at the base one.
        g = at_home(gs(tools={"pickaxe": "stone"}, dimension="overworld",
                       inventory={"cobblestone": 45, "crafting_table": 1, "raw_iron": 3,
                                  "coal": 3, "stone_pickaxe": 2}))
        g.furnace_pos = (g.base[0] - 1, g.base[1] + 1, g.base[2] - 1)
        cands = candidates(g)
        self.assertNotIn("craft_furnace", cands)
        self.assertIn("smelt_iron", cands)

    def test_a_carried_furnace_also_suppresses_the_craft(self) -> None:
        g = self._drifted()
        g.inventory["furnace"] = 1
        self.assertNotIn("craft_furnace", candidates(g))


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

    def test_a_landmark_underfoot_is_not_a_destination(self) -> None:
        # Live, run104 it155-266: 112 ticks on one spawner at (94,39,86). The bot stood on the cage,
        # `navigate` answered `ok` with 0 metres moved, and the identical cell came back every tick.
        # A landmark the body is already on is not somewhere to walk to.
        g = self._down(10.0, spawner=[BlockHit("spawner", "spawner", 11, 65, 0, 1.0)])
        self.assertEqual(resolve("explore", g).tool, "explore_frontier")

    def test_a_cage_the_search_already_walked_onto_is_written_off(self) -> None:
        # The ping-pong the write-off stops: the march carries the bot 22 m out, the cage is the
        # nearest landmark again, and the search walks straight back onto the block it just stood
        # on. A landmark that produced no step has been searched.
        g = self._down(10.0, spawner=[BlockHit("spawner", "spawner", 40, 66, 4, 22.0)])
        g.nether_landmarks = frozenset({(40, 66, 4)})
        self.assertEqual(resolve("explore", g).tool, "explore_frontier")

    def test_a_landmark_still_a_walk_away_is_walked_to(self) -> None:
        # The write-off only removes cells the bot has stood on; a cage it has not reached yet is
        # still the target the search is for.
        g = self._down(10.0, spawner=[BlockHit("spawner", "spawner", 40, 66, 4, 22.0)])
        g.nether_landmarks = frozenset({(94, 39, 86)})
        a = resolve("explore", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (40, 66, 4))

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


class WoodGateDeadZoneTest(unittest.TestCase):
    """`get_wood` and `craft_bed` must share one boundary, with no gap between them.

    Live run97 (it149-162): a geared bot at a lit gate held 4 wool, one leftover `jungle_planks`
    and no logs, no bed. `craft_bed` wants three planks or a log; `get_wood` fired only on an
    empty wood pack (`logs == 0 and planks == 0`). The single plank was too little to craft yet
    enough to clear the `== 0` gate, so neither step was offered, the candidate list emptied and
    the Nether lull fell through to 14 `wait` ticks at the base. The gate now asks whether the
    carried wood can actually pay for the craft it is for.
    """

    def _lull(self, **inv) -> GameState:
        # A geared bot at a lit gate (the Nether lull), by day, with no bed.
        g = lit_portal(at_home(geared(**inv)))
        g.phase = "day"
        g.health = 20.0
        g.food = 18.0
        return g

    def test_one_leftover_plank_still_asks_for_wood(self) -> None:
        g = self._lull(white_wool=4, jungle_planks=1)
        cand = candidates(g)
        self.assertIn("get_wood", cand)
        self.assertNotIn("craft_bed", cand)   # one plank cannot craft the bed yet

    def test_two_planks_still_ask_for_wood(self) -> None:
        self.assertIn("get_wood", candidates(self._lull(white_wool=4, jungle_planks=2)))

    def test_three_planks_craft_the_bed_and_stop_chopping(self) -> None:
        # The boundary: three planks pay for the bed, so the craft fires and `get_wood` stands down.
        g = self._lull(white_wool=4, jungle_planks=3)
        cand = candidates(g)
        self.assertIn("craft_bed", cand)
        self.assertNotIn("get_wood", cand)

    def test_a_single_log_pays_for_the_bed(self) -> None:
        # A log always suffices (the craft makes its own planks), so no `get_wood` on one log.
        g = self._lull(white_wool=4, oak_log=1)
        cand = candidates(g)
        self.assertIn("craft_bed", cand)
        self.assertNotIn("get_wood", cand)

    def test_no_wood_is_asked_for_once_a_bed_is_at_hand(self) -> None:
        # With the bed already carried the bed branch is satisfied, so the wood gate stays shut.
        g = self._lull(white_wool=4, jungle_planks=1, white_bed=1)
        self.assertNotIn("get_wood", candidates(g))


class FrameCompleteLatchTest(unittest.TestCase):
    """A finished frame stays finished when the bot roams off and the plane read goes blank.

    `_frame_needs_obsidian` reads the frame-plane scan, which only exists with the base chunks
    loaded. 290 m out it reads blank, "no frame" looks true, and the obsidian chain reopened and
    mined a second ten for a gate already built (live run97 it142-145). The `gs.frame_complete`
    latch - set from the plane the moment it showed the frame, and location-independent - keeps
    the chain shut wherever the bot stands.
    """

    def _away(self, **inv) -> GameState:
        g = geared(**inv)
        g.base = (-197, -47, -98)
        g.pos = (100.0, 64.0, 100.0)      # well off the base, where the plane read is blank
        return g

    def test_a_blank_plane_off_the_base_would_reopen_the_chain(self) -> None:
        # The bug the latch closes: no frame names, obsidian spent, not latched -> "needs obsidian".
        g = self._away()
        self.assertFalse(g.frame_names)
        self.assertTrue(_frame_needs_obsidian(g, g.inventory))

    def test_the_latch_keeps_the_chain_shut_off_the_base(self) -> None:
        g = self._away()
        g.frame_complete = True
        self.assertFalse(_frame_needs_obsidian(g, g.inventory))

    def test_the_latch_takes_mine_obsidian_and_fill_bucket_out_of_the_offer(self) -> None:
        g = self._away(flint_and_steel=1)
        g.frame_complete = True
        cand = candidates(g)
        self.assertNotIn("mine_obsidian", cand)
        self.assertNotIn("fill_bucket", cand)


class IdleWalksHomeTest(unittest.TestCase):
    """An idle geared bot off its base walks home, so `wait` can hold it there.

    `wait` is the empty-candidate fallback only *at* the base (`BASE_RADIUS` = 4 m). The moment an
    idle bot drifts off - a `collect_drops` a few metres out - the fallback used to flip to
    `explore`, which has no tether home: each frontier step pushed it further, `wait` never
    re-fired, and the base was lost (live run97 it163-193: a collect_drops to 10 m, then `explore`
    ratcheted 332 -> 652 m out and oscillated there to the run's end). The fallback now walks an
    idle geared Overworld bot home first (`_idle_off_base`), and `resolve` honours that offer so the
    two gates cannot disagree. `explore` stays the fallback in the Nether (the march to the
    fortress) and when a home walk keeps failing and is stuck-listed.
    """

    BASE = (-197, -47, -98)

    def _idle_off(self, **inv) -> GameState:
        # The daytime lull with nothing on offer: geared, every reserve met, no mob or resource in
        # reach - exactly where the old fallback fell through to `explore` (cf. `CookFoodReachTest`).
        g = geared(furnace=1, coal=8, beef=5, cobblestone=64, diamond_pickaxe=2, **inv)
        g.base = self.BASE
        g.pos = (100.0, 64.0, 100.0)      # well past BASE_RADIUS
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        return g

    def test_the_idle_bot_off_base_is_sent_home_not_out(self) -> None:
        g = self._idle_off()
        self.assertEqual(candidates(g), ["go_home"])
        self.assertTrue(_idle_off_base(g))

    def test_the_home_offer_resolves_and_is_not_refused(self) -> None:
        # The whole point of the shared reason: a fallback that offered `go_home` only to have
        # `resolve` refuse it would look like a dead end and burn the tick (cf. `_go_home_wanted`).
        g = self._idle_off()
        a = resolve("go_home", g)
        self.assertIsNotNone(a)
        self.assertEqual((a.args["x"], a.args["z"]), (self.BASE[0], self.BASE[2]))

    def test_at_the_base_the_fallback_is_unchanged(self) -> None:
        # Standing on the base is not the idle-off-base state; the base's own fallback owns the tick.
        g = self._idle_off()
        g.pos = (float(self.BASE[0]), float(self.BASE[1] + 1), float(self.BASE[2]))
        self.assertNotIn("go_home", candidates(g))
        self.assertFalse(_idle_off_base(g))

    def test_the_nether_keeps_exploring(self) -> None:
        # Down in the Nether the fallback `explore` is the march to the fortress, not wandering, so
        # the home walk must not take it: `base` names an Overworld cell no Nether path should chase.
        g = self._idle_off()
        g.dimension = "minecraft:the_nether"
        self.assertEqual(candidates(g), ["explore"])
        self.assertFalse(_idle_off_base(g))

    def test_a_stuck_home_walk_hands_the_tick_back_to_explore(self) -> None:
        # A home walk that keeps failing is stuck-listed like any step; the bot must not freeze
        # pathing to a base it cannot reach, so `explore` takes the tick instead.
        g = self._idle_off()
        g.stuck_labels = ("go_home",)
        self.assertEqual(candidates(g), ["explore"])


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

    def _assert_doorstep(self, a: Action, portal: tuple[int, int, int],
                         bot: tuple[float, float, float]) -> None:
        """The way-home walk aims at a standable doorstep, never the portal cell itself.

        `navigate` refuses a target that is an active portal cell (live, run99: 178 refusals,
        the walk never started). So the target must differ from the portal, sit at its height,
        stand `NETHER_GATE_APPROACH` blocks out on the bot's own side, and be closer to the bot
        than the portal is.
        """
        self.assertEqual(a.tool, "navigate")
        tx, ty, tz = a.args["x"], a.args["y"], a.args["z"]
        self.assertNotEqual((tx, tz), (portal[0], portal[2]))
        self.assertEqual(ty, portal[1])
        gap = math.dist((tx, tz), (portal[0], portal[2]))
        self.assertLessEqual(abs(gap - NETHER_GATE_APPROACH), 1.0)
        self.assertLess(math.dist((tx, tz), (bot[0], bot[2])),
                        math.dist((portal[0], portal[2]), (bot[0], bot[2])))

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

    def test_a_haul_home_holds_the_gate_shut(self) -> None:
        # A trip home with its rods still in the pack is processing work - craft the powder, stow
        # it - not a second crossing, so the gate is not re-offered while the haul waits. (The
        # gate used to be keyed on the one-way `nether_entered` latch instead, which never
        # cleared.)
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=64)
        g.nether_entered = True
        g.inventory["blaze_rod"] = 9
        self.assertNotIn("enter_portal", candidates(g))

    def test_an_empty_handed_death_return_reopens_the_gate(self) -> None:
        # The case run93 hung on: a death in the Nether drops the whole haul and kit, and the bot
        # respawns home to a gate still burning with no rods and no powder. The relaunch was
        # blocked by the one-way `nether_entered` latch - True for the rest of the run - so the
        # second crossing was never offered and the bot stood at the base on `wait` for ~25 min.
        # An empty pack (no rods, no powder) now reads as "no haul to process" and reopens it.
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=64)
        g.nether_entered = True
        self.assertIn("enter_portal", candidates(g))

    def test_an_intended_crossing_stands_the_rescue_down(self) -> None:
        # The rescue exists to step a body out of a gate it wandered into; it must not fight the
        # one crossing the loop asked for, or the trip can never happen.
        g = self._home()
        g.pos = (self.GATE[0] + 0.5, float(self.GATE[1]), self.GATE[2] + 0.5)
        self.assertEqual(candidates(g), ["leave_portal"])
        g.enter_intent = True
        self.assertNotEqual(candidates(g), ["leave_portal"])

    def test_the_gold_helmet_goes_on_before_the_crossing(self) -> None:
        # Live, run99: both entries (it231, it336) crossed wearing the diamond helmet with the
        # gold piece still in the pack, and the swap came only at it342 - six Nether ticks deep,
        # the bot piglin-exposed the whole gap. The helmet now goes on at the base, offered ahead
        # of the crossing so it is on the head before the bot steps through.
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=64)
        cand = candidates(g)
        self.assertIn("swap_to_gold_helmet", cand)
        self.assertLess(cand.index("swap_to_gold_helmet"), cand.index("enter_portal"))
        a = resolve("swap_to_gold_helmet", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "golden_helmet", "destination": "head"}])

    def test_the_gold_helmet_on_the_head_still_opens_the_gate(self) -> None:
        # Once the swap has run the gold piece is worn and the diamond helmet sits in the pack.
        # The departure must not read as unstocked just because the helmet moved to the head, or
        # the gate the swap was meant to walk through could never open. Worn gold counts as having
        # the helmet, so the crossing is still offered - and the swap is not, there being no bare
        # diamond helmet left to replace.
        g = self._home(flint_and_steel=1, bow=1, arrow=16, cobblestone=64)
        g.armor["helmet"] = {"item": "golden_helmet"}
        g.inventory["golden_helmet"] = 0
        g.inventory["diamond_helmet"] = 1
        cand = candidates(g)
        self.assertIn("enter_portal", cand)
        self.assertNotIn("swap_to_gold_helmet", cand)
        self.assertEqual(resolve("enter_portal", g).tool, "enter_nether_portal")

    # --- down there ---
    def test_the_gold_helmet_goes_on_first(self) -> None:
        g = self._down(golden_helmet=1)
        self.assertIn("swap_to_gold_helmet", candidates(g))
        a = resolve("swap_to_gold_helmet", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "golden_helmet", "destination": "head"}])

    def test_the_helmet_swap_is_the_only_choice_until_it_is_on(self) -> None:
        # Live, run104: the swap sat on the list beside `explore` and the model took `explore` on
        # 52 of the 66 Nether ticks it was offered, standing the bot piglin-exposed. A worn gold
        # piece is a hard precondition, so the swap owns the tick outright - nothing else is
        # offered while the diamond helmet is still on a gold piece's slot.
        g = self._down(golden_helmet=1)
        self.assertEqual(candidates(g), ["swap_to_gold_helmet"])
        # The exploration/search steps that shared the list before are gone until the swap runs.
        g.resources = {"netherrack": [BlockHit("netherrack", "netherrack", 20, 65, 0, 9.0)]}
        self.assertEqual(candidates(g), ["swap_to_gold_helmet"])

    def test_a_meal_the_body_needs_still_outranks_the_swap(self) -> None:
        # The helmet is a hard gate, but survival is a harder one: a bot that must eat eats first,
        # which the top of `candidates` already returns before the expedition branch is reached.
        g = self._down(golden_helmet=1, cooked_mutton=4)
        g.food = 2.0
        self.assertEqual(candidates(g), ["eat_food"])

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
        g.mobs = [("blaze", "hostile", 1, 6.0), ("wither_skeleton", "hostile", 2, 9.0)]
        self.assertNotIn("hunt_blaze", candidates(g))

    def test_piglins_are_not_company_for_a_blaze_hunt(self) -> None:
        # run119 it221-229: blaze 3.9 m, 53 zombified piglins at 9.1 m, health 20. The crowd that
        # kept the hunt shut for nine ticks is a fortress floor's normal population: piglins and
        # zombified piglins attack only when provoked, so they are no reason to leave a blaze
        # alone. A real hostile in reach still closes it.
        g = self._down()
        g.mobs = [("blaze", "hostile", 1, 3.9), ("zombified_piglin", "hostile", 53, 9.1),
                  ("piglin", "hostile", 1, 12.0), ("wither_skeleton", "hostile", 1, 21.8)]
        self.assertIn("hunt_blaze", candidates(g))
        g.mobs.append(("wither_skeleton", "hostile", 2, 9.0))
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

    def test_wounded_and_foodless_the_recovery_flees_the_nether(self) -> None:
        # run93 it229-320: 9.4 health falling, hunger 7, nine rods already in the pack and nothing
        # edible. `_cannot_heal` routes the tick to `_recovery_steps`, which runs *before* the
        # Nether expedition chain - so its old fall-through to `explore` (no food to hunt, no base
        # to walk to in the Nether) sent the bot wandering ~70 ticks until a blaze burned it down
        # at 4.2 health, the rods and the diamond kit lost with it. The way out of danger in the
        # Nether is the way home: `leave_nether`, offered alone.
        g = self._down(blaze_rod=9)
        g.health, g.food = 9.4, 7.0
        g.pos = (self.EXIT[0] + 40.5, float(self.EXIT[1]), self.EXIT[2] + 40.5)
        self.assertEqual(candidates(g), ["leave_nether"])
        a = resolve("leave_nether", g)
        self._assert_doorstep(a, self.EXIT, g.pos)

    def test_with_no_portal_in_the_scan_the_bot_walks_back_to_its_arrival(self) -> None:
        # The exit portal only reaches the state while it is loaded. The cell the bot arrived on
        # is where the way home is, so a bot still out in the Nether walks there first - but not
        # onto the portal cell itself: `navigate` refuses an active portal cell outright (live,
        # run99 it347-599), so the walk aims at a standable doorstep a few blocks short of it.
        g = self._down(blaze_rod=9)
        g.pos = (self.EXIT[0] + 40.5, float(self.EXIT[1]), self.EXIT[2] + 40.5)
        a = resolve("leave_nether", g)
        self._assert_doorstep(a, self.EXIT, g.pos)

    def test_far_from_the_gate_the_walk_never_aims_at_the_portal_cell(self) -> None:
        # Live, run99 it347-599: the bot sat ~220 blocks from its arrival gate with no portal in
        # the scan, and the way-home walk aimed straight at the portal cell - which `navigate`
        # refuses outright (`[NAVIGATION_PORTAL_INTENT_REQUIRED]`, prog=False, whatever the
        # `range`). 178 refusals, the walk never took a step, and the bot starved at a gate it
        # could see but never reach (~170 iterations lost). The target must never be the portal
        # cell: a doorstep on the bot's own side is a standable cell `navigate` accepts, and
        # closing that distance loads the portal so the crossing fires on the next tick.
        g = self._down(blaze_rod=9)
        g.pos = (self.EXIT[0] - 220.3, float(self.EXIT[1]), self.EXIT[2] + 18.7)
        a = resolve("leave_nether", g)
        self.assertEqual(a.tool, "navigate")
        self.assertNotEqual((a.args["x"], a.args["z"]), (self.EXIT[0], self.EXIT[2]))
        self._assert_doorstep(a, self.EXIT, g.pos)

    def test_standing_on_the_arrival_cell_crosses_instead_of_walking(self) -> None:
        # `navigate` into an active portal is refused outright - `[NAVIGATION_PORTAL_INTENT_REQUIRED]
        # Use enter_nether_portal to enter this active portal` - and the scan does not always list
        # the block the body is standing in. Live, run70: no `nether_portal` hit at all at
        # (226,49,222) and ten identical refusals, so the bot stood in its own gateway for the rest
        # of the run. Arrival first, then the crossing, is the server's own two-step.
        g = self._down(blaze_rod=9)
        self.assertEqual(resolve("leave_nether", g).tool, "enter_nether_portal")
        # The boundary is the two-block entry zone around the arrival cell: inside it the bot
        # crosses, outside it the bot walks to the doorstep first (the walk never aims at the
        # portal cell, which `navigate` refuses - the doorstep is a standable cell short of it).
        g.pos = (self.EXIT[0] + 2.5, float(self.EXIT[1]), self.EXIT[2])
        self.assertEqual(resolve("leave_nether", g).tool, "navigate")
        g.pos = (self.EXIT[0] + 1.5, float(self.EXIT[1]), self.EXIT[2])
        self.assertEqual(resolve("leave_nether", g).tool, "enter_nether_portal")

    def test_a_run_born_in_the_nether_walks_the_remembered_gate(self) -> None:
        # Cold start: this run never crossed, so there is no arrival - and the gate it did
        # arrive through is not a block in the scan either, 500 blocks unloaded (live, run74:
        # `view_blocks find nether_portal` answered `found=0` for the whole trip). What does
        # answer is the server's memory of the last crossing (`perception.REMEMBERED_GATE_SQL`),
        # outliving the death and the restart - run74 instead walked its own start position, a
        # cell of bare basalt 250 blocks from the real gate, and died waiting there.
        g = self._down(blaze_rod=9)
        g.nether_arrival = None
        g.nether_gate = self.EXIT
        g.pos = (self.EXIT[0] + 40.5, float(self.EXIT[1]), self.EXIT[2] + 40.5)
        a = resolve("leave_nether", g)
        self._assert_doorstep(a, self.EXIT, g.pos)
        # Standing beside the remembered gate is the crossing, not the walk: the same two-step
        # as the arrival, because `navigate` into an active portal is refused outright.
        g.pos = (self.EXIT[0] + 1.5, float(self.EXIT[1]), self.EXIT[2])
        self.assertEqual(resolve("leave_nether", g).tool, "enter_nether_portal")

    def test_no_arrival_and_no_gate_walks_no_one_to_a_maybe_portal(self) -> None:
        # Nothing knows the way in, so "walk back to where the trip started" would be a guess -
        # run74's guess was its own start cell, and it starved at that non-portal. No cell is
        # no offer: the scan, at least, may still find the gate somewhere the bot walks.
        g = self._down(blaze_rod=9)
        g.nether_arrival = None
        g.nether_gate = None
        self.assertIsNone(resolve("leave_nether", g))

    def test_the_arrival_this_run_saw_outranks_the_remembered_gate(self) -> None:
        # The gate is whatever the last crossing came through; the arrival is the portal this
        # trip actually walked in - a stale row cannot outvote a fact the run watched itself.
        g = self._down(blaze_rod=9)
        g.nether_gate = (self.EXIT[0] + 60, self.EXIT[1], self.EXIT[2] + 60)
        g.pos = (self.EXIT[0] + 40.5, float(self.EXIT[1]), self.EXIT[2] + 40.5)
        a = resolve("leave_nether", g)
        # The doorstep hugs the arrival portal, not the stale gate 60 blocks off: it stays within
        # the approach distance of the arrival and far from the remembered gate.
        self._assert_doorstep(a, self.EXIT, g.pos)
        self.assertGreater(math.dist((a.args["x"], a.args["z"]),
                                     (g.nether_gate[0], g.nether_gate[2])), 40.0)

    def test_the_axis_sweep_points_at_the_remembered_gate(self) -> None:
        # Without the world's frontier compass, the fallback march should at least point back
        # to the way out. Live, run74: its fake "arrival" stood west of the fortress, so the
        # sweep ran away from the real gate - east of the gate now reads as east of the
        # arrival, and the search walked the wrong way twice over.
        from mlaya.actions import _nether_search_heading
        g = self._down()
        g.nether_arrival = None
        g.nether_gate = self.EXIT
        g.pos = (self.EXIT[0] + 200.5, 70.0, self.EXIT[2])    # east of the gate
        self.assertEqual(_nether_search_heading(g), 270.0)    # due west: toward it
        g.pos = (self.EXIT[0] - 200.5, 70.0, self.EXIT[2])    # west of it
        self.assertEqual(_nether_search_heading(g), 90.0)     # due east
        g.nether_gate = None                                  # nothing known: plain east
        self.assertEqual(_nether_search_heading(g), 90.0)

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
        g.nether_travel = NETHER_SEARCH_M
        self.assertIn("leave_nether", candidates(g))

    def test_open_nether_on_arrival_is_not_the_end_of_the_hunt(self) -> None:
        # The bot lands in open Nether, where a scan reaches nothing but lava and a fortress is
        # hundreds of blocks off. Turning round on the first tick would abandon every trip.
        g = self._down(blaze_rod=2)
        g.nether_ticks = 1
        self.assertNotIn("leave_nether", candidates(g))
        g.nether_travel = NETHER_SEARCH_M - 1
        self.assertNotIn("leave_nether", candidates(g))

    def test_a_fortress_in_sight_keeps_the_hunt_open(self) -> None:
        g = self._down(blaze_rod=2)
        g.nether_travel = NETHER_SEARCH_M + 10
        g.resources = {"nether_bricks": [BlockHit("nether_bricks", "nether_bricks",
                                                   40, 66, 10, 30.0)]}
        self.assertNotIn("leave_nether", candidates(g))

    def test_a_blaze_seen_but_out_of_reach_keeps_the_hunt_open(self) -> None:
        # A blaze 90 m off is not a hunt to start, but it is a reason to close in, not to leave.
        g = self._down(blaze_rod=2)
        g.nether_travel = NETHER_SEARCH_M + 10
        g.mobs = [("blaze", "hostile", 1, 90.0)]
        self.assertNotIn("leave_nether", candidates(g))

    # --- the fortress search is a first-class step down there ---
    def test_the_fortress_search_is_offered_first_class(self) -> None:
        # In the Nether, with no blaze in reach and nothing to turn back for, the step the trip
        # is for is the search for the fortress. It is offered on its own, the way `mine_diamond`
        # is in the Overworld - not buried in the `explore` fallback (live: `explore` was offered
        # on 1 of 2499 Nether ticks, because the tech-tree candidates kept the list non-empty).
        g = self._down(blaze_rod=2)
        self.assertIn("explore", candidates(g))

    def test_the_fortress_search_is_not_offered_overworld(self) -> None:
        # The first-class `explore` is a Nether step; the Overworld keeps its own fallback.
        g = self._home()
        g.nether_entered = True
        self.assertNotIn("explore", candidates(g))

    def test_the_tech_tree_is_not_offered_in_the_nether(self) -> None:
        # A pack full of netherrack reads as an incomplete tech tree (no cobble, no table), so
        # without the Overworld gate the Nether offered `craft_table`/`mine_stone` every tick and
        # the search never ran. None of the surface work may be on offer down there.
        g = self._down(netherrack=64, oak_log=16)
        for label in ("craft_table", "mine_stone", "craft_pickaxe", "craft_furnace",
                      "craft_chest", "get_wood", "mine_iron", "mine_coal", "mine_diamond"):
            self.assertNotIn(label, candidates(g), label)

    def test_the_nether_prep_steps_are_not_offered_down_there(self) -> None:
        # `_nether_idle` (the prep's scope) is true in the Nether - the kit is done, the frame is
        # up - so the prep would dig from a base that does not exist down there.
        g = self._down(string=0, feather=0)
        g.inventory.pop("bow", None)
        g.inventory.pop("arrow", None)
        for label in ("craft_bow", "hunt_string", "craft_arrows", "get_flint", "hunt_feather"):
            self.assertNotIn(label, candidates(g), label)

    def test_a_pack_full_of_droppable_junk_sheds_before_it_leaves(self) -> None:
        # "Out of room" is a reason to turn back only when the room is full of something that
        # cannot be dropped. A pack of netherrack frees itself with `drop_junk`, so the bot sheds
        # it and keeps searching instead of coming home with nothing (live: 34/36 of netherrack,
        # `leave_nether` offered every tick).
        g = self._down(netherrack=256, cooked_mutton=32)
        g.slots_used = 34
        self.assertEqual(candidates(g), ["drop_junk"])
        self.assertNotIn("leave_nether", candidates(g))

    def test_the_shed_owns_the_tick_the_pack_is_full(self) -> None:
        # Offered beside `explore`, `drop_junk` was on the list 19 times in run118 (it92-114) and
        # taken 0, while the fortress march hoovered 3 -> 173 netherrack in fifteen ticks and the
        # pack sat at 36/36: the haul neither progressed nor ended. A blaze rod has nowhere to
        # land in a full pack, so the shed is a precondition of the march, not a second choice.
        g = self._down(netherrack=173, gravel=73, basalt=53, nether_bricks=49, blackstone=11,
                       blaze_rod=5, diamond_pickaxe=2, cooked_mutton=9)
        g.slots_used = 36
        self.assertEqual(candidates(g), ["drop_junk"])
        a = resolve("drop_junk", g)
        self.assertEqual(a.tool, "drop_item")
        self.assertIn("netherrack", {i["item_name"] for i in a.args["items"]})

    def test_a_pack_with_room_still_marches(self) -> None:
        # The shed only owns the tick at `INVENTORY_NEAR_FULL`; with slots to spare the search is
        # the step the trip is for, and dropping is not on offer.
        g = self._down(netherrack=173, gravel=73, basalt=53, nether_bricks=49, blackstone=11,
                       blaze_rod=5, diamond_pickaxe=2, cooked_mutton=9)
        g.slots_used = 30
        self.assertIn("explore", candidates(g))
        self.assertNotIn("drop_junk", candidates(g))

    def test_a_given_up_shed_ends_the_trip(self) -> None:
        # `_note_drop_undone` takes `drop_junk` off the offer for `STUCK_COOLDOWN` ticks once the
        # ground has refused a shed twice in a row. With the shed given up, a full pack is no
        # longer something the trip can work around: the bot comes home with the rods it has.
        g = self._down(netherrack=173, gravel=73, basalt=53, nether_bricks=49, blackstone=11,
                       blaze_rod=5, diamond_pickaxe=2, cooked_mutton=9)
        g.slots_used = 36
        g.stuck_labels = frozenset({"drop_junk"})
        self.assertIn("leave_nether", candidates(g))
        self.assertNotIn("drop_junk", candidates(g))

    def test_a_pack_full_of_undroppable_loot_still_leaves(self) -> None:
        # The same full-pack test with something that cannot be dropped (blaze rods over the
        # hoard cap is still loot, not junk): the trip ends and the bot comes home with it.
        g = self._down(blaze_rod=9)
        g.slots_used = 34
        self.assertIn("leave_nether", candidates(g))

    def test_the_walk_off_is_offered_while_the_drop_is_pending(self) -> None:
        # A junk drop that landed on open ground (no pit) is not finished until the bot has
        # walked off it, so while that walk is pending the offer is the walk alone - it owns the
        # tick, ahead of the search and the tech tree.
        g = self._down(netherrack=128)
        g.drop_open_pending = True
        g.drop_pos = (g.pos[0], g.pos[1], g.pos[2])
        self.assertEqual(candidates(g), ["step_away_drops"])

    def _agent_with_pending_drop(self):
        from mlaya.config import Settings
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent._drop_open_pending = True
        agent._drop_pos = (12.5, 65.0, -3.5)
        agent._step_away_fails = 0
        return agent

    def test_a_step_that_succeeds_closes_the_walk_off(self) -> None:
        # The happy path: the 3-block step goes through and the latch clears, so the offer is
        # gone and the bot is free to make its next choice.
        agent = self._agent_with_pending_drop()
        agent._note_step_away(Action("step_away_drops", "navigate", {}, ""), True)
        self.assertFalse(agent._drop_open_pending)
        self.assertIsNone(agent._drop_pos)

    def test_a_step_that_keeps_failing_releases_the_walk_off(self) -> None:
        # The walk off is a 3-block navigate that rough Nether terrain can refuse on every try.
        # A refusal keeps the junk at the bot's feet, so the latch must survive a couple of them
        # (the pathfinder may find footing soon) - but not all of them, or the bot is wedged on
        # the one failing candidate forever (live, run73: a fortress in view, `step_away_drops`
        # refused on every tick). After `STEP_AWAY_GIVE_UP` refusals the walk off is given up and
        # the bot moves on; the junk stays dropped because the bot marches away from it, not over
        # it.
        from mlaya.graph import STEP_AWAY_GIVE_UP
        agent = self._agent_with_pending_drop()
        act = Action("step_away_drops", "navigate", {}, "")
        for i in range(STEP_AWAY_GIVE_UP - 1):
            agent._note_step_away(act, False)
            self.assertTrue(agent._drop_open_pending, f"latch must survive refusal {i + 1}")
        agent._note_step_away(act, False)
        self.assertFalse(agent._drop_open_pending)
        self.assertIsNone(agent._drop_pos)
        self.assertEqual(agent._step_away_fails, 0)

    def test_the_loop_reports_refusals_through_note_result(self) -> None:
        # The give-up existed but never reached it in run81: the call sat under `_verify`'s
        # success guard, so a walk-off that the cliff top refused again and again was never
        # counted (live: 470 refusals in a row, `[NAVIGATION_TARGET_UNSUPPORTED]` every tick,
        # nothing progressing, the run cap burned on the one exclusive offer). The report now
        # goes both ways through `_note_result`, and this test rides that very path.
        from mlaya.graph import STEP_AWAY_GIVE_UP
        agent = self._agent_with_pending_drop()
        act = Action("step_away_drops", "navigate", {}, "")
        for i in range(STEP_AWAY_GIVE_UP - 1):
            agent._note_result(act, False)
            self.assertTrue(agent._drop_open_pending, f"give-up must come last (refusal {i + 1})")
        agent._note_result(act, False)
        self.assertFalse(agent._drop_open_pending)

    def test_note_result_closes_the_walk_off_on_success(self) -> None:
        # The happy path rides the same report: success closes the latch, and the refusal
        # counter goes with it.
        agent = self._agent_with_pending_drop()
        agent._note_result(Action("step_away_drops", "navigate", {}, ""), True)
        self.assertFalse(agent._drop_open_pending)
        self.assertEqual(agent._step_away_fails, 0)

    def test_a_refused_other_label_leaves_the_pending_walk_off_alone(self) -> None:
        # The other half of `_note_result`: a refusal is counted for the walk-off itself and
        # never for it, and `_note_placed` must not see the failure - the pending latch and
        # its counter stay exactly as they were.
        agent = self._agent_with_pending_drop()
        agent._note_result(Action("mine_iron", "dig", {}, ""), False)
        self.assertTrue(agent._drop_open_pending)
        self.assertEqual(agent._step_away_fails, 0)

    def test_explore_walks_to_a_visible_spawner(self) -> None:
        # The search closes in on the bricks once they are in view, the way a dig closes in on
        # the ore: `explore`'s Nether branch navigates to the nearest fortress landmark.
        g = self._down(blaze_rod=2)
        g.resources = {"spawner": [BlockHit("spawner", "spawner", 40, 66, 10, 30.0)]}
        self.assertIn("explore", candidates(g))
        a = resolve("explore", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (40, 66, 10))

    # --- home again ---
    def test_the_diamond_helmet_goes_back_on(self) -> None:
        g = self._home(blaze_rod=9, diamond_helmet=1)
        g.nether_entered = True
        g.armor["helmet"] = {"item": "golden_helmet"}
        self.assertIn("swap_to_diamond_helmet", candidates(g))
        a = resolve("swap_to_diamond_helmet", g)
        self.assertEqual(a.tool, "equip")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_helmet", "destination": "head"}])

    def test_the_rods_become_the_powder_the_eyes_owe(self) -> None:
        # Nine rods could make eighteen powder, but the chain only ever needs what the eyes still
        # owe: sixteen sockets, sixteen powder. The two rods to spare stay in the pack.
        g = self._home(blaze_rod=9, crafting_table=1)
        g.nether_entered = True
        self.assertIn("craft_blaze_powder", candidates(g))
        a = resolve("craft_blaze_powder", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "blaze_powder", "count": 16}])
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

    def test_a_stowed_expedition_closes_the_blaze_chain(self) -> None:
        # The rod trip is over: no blaze hunt, no powder craft, no stow. But the pearl waves
        # reopen the gate - the pearls the eyes owe are hunted down there, not in the Overworld.
        g = self._home()
        g.powder_stowed = True
        out = candidates(g)
        for label in ("hunt_blaze", "leave_nether", "craft_blaze_powder", "stow_blaze_powder"):
            self.assertNotIn(label, out)
        self.assertTrue(nether_expedition_open(g))
        self.assertIn("enter_portal", out)

    def test_a_stowed_expedition_with_pearls_in_hand_stays_closed(self) -> None:
        # Pearls just brought home are crafted into eyes before another wave launches; the
        # chain only reopens when the pack is empty of pearls again.
        g = self._home(ender_pearl=4)
        g.powder_stowed = True
        g.powder_bank = 16
        self.assertFalse(nether_expedition_open(g))
        self.assertNotIn("enter_portal", candidates(g))

    def test_the_pearl_waves_close_after_the_trip_cap(self) -> None:
        # Five waves spent and the pearls still owed: the gate stays shut and the Overworld
        # enderman roam takes the hunt over, so the goal is never stranded on an empty Nether.
        g = self._home()
        g.powder_stowed = True
        g.powder_bank = 16
        g.pearl_wave_trips = PEARL_WAVE_MAX_TRIPS
        self.assertFalse(nether_expedition_open(g))
        self.assertNotIn("enter_portal", candidates(g))


class PiglinBarterTest(unittest.TestCase):
    """Pearls from piglin barter: the gold reserve banked in the lull, traded down in the Nether.

    `barter` needs an exact loaded adult piglin id, gold ingots in the pack and a piglin close
    enough to walk to. The reserve is dug in the same lull as the helmet but is deliberately not
    part of the crossing gate: the helmet is the piglin pass, the reserve only the pearl budget,
    and a gate that waits for a stack of barter gold waits for a dig the trip does not need to
    survive (the same shape as the bow, which `_nether_bank_carried` keeps out of the gate).
    """

    NETHER = "minecraft:the_nether"

    def _lull(self, **inv) -> GameState:
        """The bot at its base in the prep lull, a gold deposit loaded, the helmet banked."""
        base = {"flint_and_steel": 1, "white_bed": 1, "cooked_beef": 24, "furnace": 1,
                "cobblestone": 64, "acacia_planks": 4, "coal": 4, "golden_helmet": 1,
                "diamond_pickaxe": 2}
        base.update(inv)
        g = at_home(lit_portal(geared(**base)))
        g.food = 20.0
        g.resources = {**(g.resources or {}),
                       "gold_ore": [BlockHit("gold_ore", "gold_ore", -200, 20, -100, 40.0)]}
        return g

    def _down(self, **inv) -> GameState:
        """The bot in the Nether, gold helmet worn, one adult piglin loaded twelve blocks off."""
        g = gs(pos=(12.5, 65.0, -3.5), dimension=self.NETHER, inventory=dict(inv),
               tools={"pickaxe": "diamond", "sword": "diamond"},
               armor={"helmet": {"item": "golden_helmet"},
                      "chestplate": {"item": "diamond_chestplate"},
                      "leggings": {"item": "diamond_leggings"},
                      "boots": {"item": "diamond_boots"}},
               offhand="shield", health=20.0, food=18.0)
        g.base = (-197, -47, -98)
        g.nether_entered = True
        g.nether_arrival = (12, 65, -4)
        g.piglins = [(412, 12.0)]
        return g

    # --- the reserve: dug for the wave that spends it, never gating the gate ---
    def test_the_reserve_is_dug_in_the_pearl_wave_not_before_it(self) -> None:
        # The user's waves put the barter gold in the trip that trades it: while the rod haul is
        # still to be made, the lull digs nothing for pearls.
        g = self._lull()
        self.assertFalse(_gold_reserve_short(g))
        self.assertNotIn("mine_gold_reserve", candidates(g))
        g.powder_stowed = True
        self.assertTrue(_gold_reserve_short(g))
        self.assertIn("mine_gold_reserve", candidates(g))
        a = resolve("mine_gold_reserve", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args["block_name"], "gold_ore")
        self.assertEqual(a.args["count"], GOLD_BARTER_RESERVE)

    def test_the_reserve_waits_for_the_helmet(self) -> None:
        # Without a worn gold piece the bot that walks up to trade gets torn apart, and the
        # helmet line already digs the gold it needs.
        g = self._lull(golden_helmet=0)
        g.powder_stowed = True
        self.assertNotIn("mine_gold_reserve", candidates(g))
        self.assertIn("mine_gold", candidates(g))
        self.assertIsNone(resolve("mine_gold_reserve", g))

    def test_the_reserve_is_not_dug_for_pearls_already_held(self) -> None:
        g = self._lull(ender_pearl=16)
        g.powder_stowed = True
        self.assertFalse(_gold_reserve_short(g))
        self.assertNotIn("mine_gold_reserve", candidates(g))
        self.assertIsNone(resolve("mine_gold_reserve", g))

    def test_the_reserve_needs_a_loaded_deposit(self) -> None:
        g = self._lull()
        g.powder_stowed = True
        g.resources = {k: v for k, v in g.resources.items() if k != "gold_ore"}
        self.assertIsNone(resolve("mine_gold_reserve", g))

    def test_the_reserve_does_not_hold_the_gate_shut(self) -> None:
        # The crossing gate reads `_all_nether_prep_steps`; the reserve rides on top of it in
        # `_nether_prep_steps`, so a bot short of barter gold still walks through the gate.
        g = self._lull()
        g.powder_stowed = True
        self.assertEqual(_all_nether_prep_steps(g), [])
        self.assertEqual(_nether_prep_steps(g), ["mine_gold_reserve"])
        self.assertTrue(_expedition_prepared(g))
        self.assertIn("enter_portal", candidates(g))

    def test_smelt_gold_covers_the_reserve_too(self) -> None:
        # The helmet is already crafted, so the five ingots went into it; the raw gold in the
        # pack still becomes barter money.
        g = self._lull(raw_gold=10, coal=4)
        g.powder_stowed = True
        self.assertIn("smelt_gold", candidates(g))
        a = resolve("smelt_gold", g)
        self.assertEqual(a.args["item_name"], "raw_gold")
        self.assertGreater(a.args["count"], 0)

    def test_smelt_gold_stops_once_the_reserve_is_banked(self) -> None:
        g = self._lull(raw_gold=GOLD_BARTER_RESERVE, coal=4)
        g.powder_stowed = True
        self.assertFalse(_gold_reserve_short(g))
        self.assertIsNone(resolve("smelt_gold", g))

    # --- the barter: gold, a piglin in reach, pearls still owed ---
    def test_barter_trades_gold_with_the_nearest_adult_piglin(self) -> None:
        g = self._down(blaze_rod=9, gold_ingot=GOLD_BARTER_RESERVE)
        g.powder_stowed = True
        self.assertTrue(_barter_pearls_wanted(g))
        a = resolve("barter_pearls", g)
        self.assertEqual(a.tool, "barter")
        self.assertEqual(a.args, {"piglin_id": 412, "item_name": "ender_pearl",
                                 "count": BARTER_PEARL_BATCH,
                                 "gold_budget": BARTER_GOLD_BUDGET})

    def test_barter_needs_gold_in_the_pack(self) -> None:
        g = self._down(blaze_rod=9)
        g.powder_stowed = True
        self.assertFalse(_barter_pearls_wanted(g))
        self.assertIsNone(resolve("barter_pearls", g))

    def test_barter_needs_a_piglin_within_reach(self) -> None:
        # `barter` navigates to the piglin; a walk to a distant one is the same off-base drift
        # that stalled the Overworld pearl hunt (run110), so the bot marches instead.
        g = self._down(blaze_rod=9, gold_ingot=GOLD_BARTER_RESERVE)
        g.powder_stowed = True
        g.piglins = [(412, BARTER_PIGLIN_RANGE + 8.0)]
        self.assertFalse(_barter_pearls_wanted(g))
        self.assertIsNone(resolve("barter_pearls", g))

    def test_barter_needs_pearls_still_owed(self) -> None:
        g = self._down(blaze_rod=9, gold_ingot=GOLD_BARTER_RESERVE, ender_pearl=16)
        g.powder_stowed = True
        self.assertFalse(_barter_pearls_wanted(g))
        self.assertIsNone(resolve("barter_pearls", g))

    def test_the_rod_trip_never_stops_to_barter(self) -> None:
        # Welle 1 hauls rods and comes straight home: the barter that used to ride the rods-done
        # tick (commit 584a51a) is now a pearl-wave job, so a full pack of rods walks back
        # through the gate even with gold and a piglin at hand.
        g = self._down(blaze_rod=9, gold_ingot=GOLD_BARTER_RESERVE)
        out = candidates(g)
        self.assertIn("leave_nether", out)
        self.assertNotIn("barter_pearls", out)

    def test_a_pearl_wave_barters_before_it_hunts(self) -> None:
        # The wave trades the reserve first - one reserve buys about a pearl - and only then
        # turns to the endermen.
        g = self._down(gold_ingot=GOLD_BARTER_RESERVE)
        g.powder_stowed = True
        out = candidates(g)
        self.assertIn("barter_pearls", out)
        self.assertNotIn("leave_nether", out)
        self.assertNotIn("hunt_blaze", out)

    def test_a_piglin_out_of_range_sends_the_wave_roaming(self) -> None:
        # A piglin beyond reach is not worth the walk in a pearl wave: the wave roams the warped
        # forest for endermen instead, and the next wave may find a piglin closer.
        g = self._down(gold_ingot=GOLD_BARTER_RESERVE)
        g.powder_stowed = True
        g.piglins = [(412, BARTER_PIGLIN_RANGE + 20.0)]
        out = candidates(g)
        self.assertIn("search_enderman", out)
        self.assertNotIn("barter_pearls", out)
        self.assertNotIn("explore", out)

    def test_the_bot_comes_home_when_the_gold_is_spent(self) -> None:
        # Pearls owed but no ingots left to offer: the trip is done, and the next wave (or the
        # Overworld hunt, once the waves are spent) carries on.
        g = self._down(blaze_rod=9, ender_pearl=4)
        out = candidates(g)
        self.assertIn("leave_nether", out)
        self.assertNotIn("barter_pearls", out)

    def test_the_reserve_is_not_dug_once_the_attempts_are_spent(self) -> None:
        # A 16-ingot reserve buys about one pearl; without the cap the lull would dig gold for a
        # barter that keeps missing until the run ran out of iterations.
        g = self._lull()
        g.powder_stowed = True
        g.barter_attempts = BARTER_MAX_ATTEMPTS
        self.assertFalse(_gold_reserve_short(g))
        self.assertNotIn("mine_gold_reserve", candidates(g))
        self.assertIsNone(resolve("mine_gold_reserve", g))

    def test_the_barter_closes_after_its_tries(self) -> None:
        g = self._down(blaze_rod=9, gold_ingot=GOLD_BARTER_RESERVE)
        g.powder_stowed = True
        g.barter_attempts = BARTER_MAX_ATTEMPTS
        self.assertFalse(_barter_pearls_wanted(g))
        self.assertIsNone(resolve("barter_pearls", g))
        out = candidates(g)
        # The trade is closed, but the wave is not over: with pearl budget left it roams for
        # endermen, and only when that is spent does it come home.
        self.assertIn("search_enderman", out)
        self.assertNotIn("leave_nether", out)
        self.assertNotIn("explore", out)
        g.nether_travel = NETHER_PEARL_SEARCH_M
        out = candidates(g)
        self.assertIn("leave_nether", out)
        self.assertNotIn("search_enderman", out)


class StrongholdPhaseTest(unittest.TestCase):
    """The phase after the haul: pearls, the banked powder, the eyes, the stronghold, the portal.

    The chain lives in `actions/stronghold.py` and follows the Nether shape: every step is
    offered only once the ones before it are done, the server owns the mechanics, and the loop
    only books the calls. The rare case these tests care about most is the powder top-up - the
    banked eighteen lost to a death or a raided chest - which has to reopen the gate the
    expedition closed.
    """

    BASE = (-197, -47, -98)
    CHEST = (-196, -46, -98)
    GATE = (-195, -46, -98)
    OVERWORLD = "minecraft:overworld"
    NETHER = "minecraft:the_nether"

    def _home(self, mobs=(), **inv) -> GameState:
        """A stocked bot at its base, haul banked, gate burning - the stronghold's starting state."""
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64,
                     golden_helmet=1, cooked_mutton=64, white_bed=1, diamond_pickaxe=2)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.chest_pos = self.CHEST
        g.powder_stowed = True
        g.powder_bank = 18
        g.health = 20.0
        g.food = 20.0
        g.dimension = self.OVERWORLD
        g.mobs = list(mobs)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  *self.GATE, 2.0)]}
        return g

    def test_the_phase_is_silent_before_the_haul(self) -> None:
        g = self._home(ender_pearl=16, blaze_powder=16)
        g.powder_stowed = False
        for label in ("hunt_enderman", "take_blaze_powder", "craft_ender_eye",
                      "locate_stronghold", "activate_end_portal"):
            self.assertNotIn(label, candidates(g))

    def test_pearls_and_powder_craft_the_eyes(self) -> None:
        g = self._home(ender_pearl=16, blaze_powder=16)
        self.assertIn("craft_ender_eye", candidates(g))
        a = resolve("craft_ender_eye", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "ender_eye", "count": 16}])
        self.assertTrue(a.args["temporary_workstation"])

    def test_the_craft_stops_at_the_eyes_owed(self) -> None:
        # Sixteen eyes is the whole phase; a pack with more pearls than it needs does not craft
        # a seventeenth, and the powder the eyes owe comes out of the chest, not the pack.
        g = self._home(ender_pearl=20, blaze_powder=20)
        self.assertEqual(resolve("craft_ender_eye", g).args["items"][0]["count"], 16)
        self.assertNotIn("take_blaze_powder", candidates(g))

    def test_the_powder_comes_back_out_of_the_chest(self) -> None:
        g = self._home(ender_pearl=16)
        self.assertIn("take_blaze_powder", candidates(g))
        a = resolve("take_blaze_powder", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "withdraw")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.CHEST)
        self.assertEqual(a.args["items"], [{"item_name": "blaze_powder", "count": 16}])

    def test_the_withdraw_walks_to_the_cell_the_powder_was_stowed_in(self) -> None:
        # `stow_blaze_powder` deposits into the network's roomy chest, which may be an overflow
        # chest rather than `chest_pos`; the withdraw has to look in the remembered cell, or a
        # full overflow reads as an empty bank and the bot walks back through the gate for
        # powder sitting a few blocks away.
        overflow = (-190, -46, -100)
        g = self._home(ender_pearl=16)
        g.powder_cells = (overflow,)
        a = resolve("take_blaze_powder", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), overflow)
        g.pos = (overflow[0] + 0.5, overflow[1] + 1.0, overflow[2] + 0.5)
        a = resolve("take_blaze_powder", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), overflow)

    def test_the_enderman_hunt_is_offered_when_the_fight_is_safe(self) -> None:
        g = self._home(mobs=[("enderman", "hostile", 12, 18.0)])
        self.assertIn("hunt_enderman", candidates(g))
        a = resolve("hunt_enderman", g)
        self.assertEqual(a.args["mob_name"], "enderman")
        self.assertEqual(a.args["drop_name"], "ender_pearl")
        self.assertEqual(a.args["count"], 3)  # one batch at a time, sixteen still owed

    def test_a_wounded_hunt_is_refused_but_a_crowded_one_is_not(self) -> None:
        # An enderman teleports behind the bot and hits for half a heart a swing, so a wounded
        # fight still waits. The crowding guard the blaze hunt keeps is dropped here: endermen
        # spawn at night with the usual pack, so the diamond kit takes the pearl fight even with
        # a zombie alongside rather than stalling at the base to the run limit.
        g = self._home(mobs=[("enderman", "hostile", 12, 18.0)])
        g.health = 10.0
        self.assertNotIn("hunt_enderman", candidates(g))
        g.health = 20.0
        g.mobs = [("enderman", "hostile", 12, 18.0), ("zombie", "hostile", 8, 10.0)]
        self.assertIn("hunt_enderman", candidates(g))

    def test_no_enderman_in_reach_roams_to_find_one(self) -> None:
        # Pearls still owed and no enderman loaded within reach. In the Overworld the roam is
        # only the fallback: while the pearl waves still have trips left, the bot relaunches
        # through the gate rather than wandering the base's frontier (run110's off-base drift
        # is the reason the hunt moved into the Nether at all).
        g = self._home(mobs=[])
        cand = candidates(g)
        self.assertNotIn("search_enderman", cand)
        self.assertIn("enter_portal", cand)
        # The five waves spent, the Overworld roam takes over.
        g.pearl_wave_trips = PEARL_WAVE_MAX_TRIPS
        cand = candidates(g)
        self.assertIn("search_enderman", cand)
        self.assertNotIn("hunt_enderman", cand)
        a = resolve("search_enderman", g)
        self.assertEqual(a.tool, "explore_frontier")
        # An enderman in reach hands the tick to the hunt, not another roam.
        g.mobs = [("enderman", "hostile", 12, 18.0)]
        cand = candidates(g)
        self.assertIn("hunt_enderman", cand)
        self.assertNotIn("search_enderman", cand)
        # A wounded bot neither hunts nor roams - it heals at the base first.
        g.mobs = []
        g.health = 10.0
        self.assertNotIn("search_enderman", candidates(g))

    def test_a_pearl_wave_hunts_the_enderman_it_find(self) -> None:
        # Down in the Nether the hunt is dimension-blind: `_safe_enderman_hunt` is the whole
        # precondition, so a warped-forest enderman in reach is fought the same way. The gold
        # helmet is worn - without it the piglin filter owns the tick before the pearl ladder.
        g = self._home(mobs=[("enderman", "hostile", 12, 18.0)])
        g.powder_stowed = True
        g.dimension = self.NETHER
        g.armor["helmet"] = {"item": "golden_helmet"}
        self.assertIn("hunt_enderman", candidates(g))
        self.assertNotIn("search_enderman", candidates(g))
        a = resolve("hunt_enderman", g)
        self.assertEqual(a.args["mob_name"], "enderman")
        self.assertEqual(a.args["drop_name"], "ender_pearl")

    def test_a_pearl_wave_short_on_blocks_comes_home_to_restock(self) -> None:
        # The fight roofs the bot over as the enderman teleports, and `collect_mob_drop` refuses
        # with `[COMBAT_BUILD_MATERIALS_MISSING]` when the scaffold runs short (run112d it240-243:
        # five netherrack left, roof wanted six, four refused ticks). Below the block floor the
        # hunt sits out, and with the enderman still loaded the wave neither hunts nor roams past
        # it - it leaves to restock and relaunch.
        g = self._home(mobs=[("enderman", "hostile", 12, 18.0)], cobblestone=5, netherrack=5)
        g.powder_stowed = True
        g.dimension = self.NETHER
        g.armor["helmet"] = {"item": "golden_helmet"}
        self.assertTrue(_enderman_in_reach(g))
        self.assertFalse(_safe_enderman_hunt(g))
        out = candidates(g)
        self.assertNotIn("hunt_enderman", out)
        self.assertNotIn("search_enderman", out)
        self.assertIn("leave_nether", out)

    def test_netherrack_counts_toward_the_hunt_block_floor(self) -> None:
        # Netherrack scaffolds the roof as readily as cobblestone, so the Nether stock and the
        # cobble stack are summed: topped to the floor out of netherrack alone, the hunt is on.
        g = self._home(mobs=[("enderman", "hostile", 12, 18.0)], cobblestone=0,
                       netherrack=ENDERMAN_HUNT_BLOCKS - 1)
        g.powder_stowed = True
        g.dimension = self.NETHER
        g.armor["helmet"] = {"item": "golden_helmet"}
        self.assertEqual(_combat_blocks_carried(g.inventory), ENDERMAN_HUNT_BLOCKS - 1)
        self.assertNotIn("hunt_enderman", candidates(g))
        g.inventory["netherrack"] = ENDERMAN_HUNT_BLOCKS
        self.assertTrue(_safe_enderman_hunt(g))
        self.assertIn("hunt_enderman", candidates(g))

    def test_the_wave_roams_until_the_pearl_budget_is_spent(self) -> None:
        # No enderman loaded: the wave roams new ground, bounded in metres like the fortress
        # hunt, so an empty stretch sends the bot home to stow and relaunch instead of burning
        # the trip wandering.
        g = self._home()
        g.powder_stowed = True
        g.dimension = self.NETHER
        g.armor["helmet"] = {"item": "golden_helmet"}
        g.nether_travel = 0.0
        self.assertTrue(_search_enderman_wanted(g))
        self.assertIn("search_enderman", candidates(g))
        g.nether_travel = NETHER_PEARL_SEARCH_M
        self.assertFalse(_search_enderman_wanted(g))
        out = candidates(g)
        self.assertNotIn("search_enderman", out)
        self.assertIn("leave_nether", out)

    def test_sixteen_eyes_send_the_bot_to_the_stronghold(self) -> None:
        g = self._home(ender_eye=16)
        self.assertIn("locate_stronghold", candidates(g))
        a = resolve("locate_stronghold", g)
        self.assertEqual(a.tool, "locate_stronghold")
        self.assertEqual(a.args["phase"], "estimate")
        self.assertTrue(a.args["continue_without_recommended_items"])
        g.stronghold_estimated = True
        self.assertEqual(resolve("locate_stronghold", g).args["phase"], "locate")

    def test_standing_at_the_frame_fills_it(self) -> None:
        g = self._home(ender_eye=16)
        g.resources["end_portal_frame"] = [BlockHit("end_portal_frame", "end_portal_frame",
                                                    100, 60, 200, 3.0)]
        g.resource_found = {"end_portal_frame": 12}
        self.assertIn("activate_end_portal", candidates(g))
        a = resolve("activate_end_portal", g)
        self.assertEqual(a.tool, "activate_portal")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (100, 60, 200))
        g.end_portal_active = True
        self.assertTrue(stronghold_complete(g))
        self.assertNotIn("activate_end_portal", candidates(g))

    def test_a_lost_bank_reopens_the_gate(self) -> None:
        # The top-up: the stowed powder is gone and the eyes still owe powder, so the tested
        # Nether chain runs again. The departure gate reads the expedition as open even though
        # the completion latch says it is done - without that, `candidates` reopens the chain
        # and `enter_portal` still refuses, and the bot stands home at an empty chest.
        g = self._home(ender_pearl=16)
        g.powder_exhausted = True
        self.assertTrue(nether_expedition_open(g))
        self.assertIn("enter_portal", candidates(g))
        a = resolve("enter_portal", g)
        self.assertEqual(a.tool, "enter_nether_portal")

    def test_a_full_bank_keeps_the_gate_shut(self) -> None:
        g = self._home(ender_pearl=16, blaze_powder=18)
        self.assertFalse(nether_expedition_open(g))
        self.assertNotIn("enter_portal", candidates(g))

    def test_the_enderman_hunt_sizes_to_the_eyes_still_owed(self) -> None:
        # Shattered eyes, not a cold start: the bot holds fourteen, owes two and has no pearls, so
        # the hunt asks for two more - not a fresh sixteen. Each enderman is a teleporting fight;
        # over-hunting a dozen of them to replace two eyes risks the diamond kit for nothing.
        g = self._home(ender_eye=14, mobs=[("enderman", "hostile", 12, 18.0)])
        self.assertIn("hunt_enderman", candidates(g))
        self.assertEqual(resolve("hunt_enderman", g).args["count"], 2)

    def test_the_pearls_already_owed_are_not_re_hunted(self) -> None:
        # Two eyes owed, two pearls already in the pack: the hunt is done even though the pack is
        # far short of the sixteen a cold start gathers.
        g = self._home(ender_eye=14, ender_pearl=2, mobs=[("enderman", "hostile", 12, 18.0)])
        self.assertNotIn("hunt_enderman", candidates(g))

    def test_a_shattered_eye_after_the_estimate_does_not_force_a_rebuild(self) -> None:
        # The estimate throw costs an eye. Before the estimate the bot builds the full sixteen;
        # after it, the frame needs only twelve, so a shaved eye lets the locate travel proceed
        # instead of dragging the bot back to re-craft (and maybe back to the Nether) for one eye.
        g = self._home(ender_eye=13)
        self.assertNotIn("locate_stronghold", candidates(g))
        g.stronghold_estimated = True
        self.assertIn("locate_stronghold", candidates(g))
        self.assertEqual(resolve("locate_stronghold", g).args["phase"], "locate")

    def test_a_top_up_hauls_only_the_rods_the_broken_eyes_owe(self) -> None:
        # Two eyes shattered and the bank is empty, so the top-up reopens the Nether. It owes two
        # powder = one rod, not the first trip's nine; the hunt asks for exactly that one rod.
        g = self._home(ender_eye=14, mobs=[("blaze", "hostile", 1, 5.0)])
        g.powder_exhausted = True
        g.dimension = self.NETHER
        self.assertTrue(nether_expedition_open(g))
        self.assertEqual(resolve("hunt_blaze", g).args["count"], 1)

    def test_a_top_up_crafts_only_the_powder_the_eyes_owe(self) -> None:
        # Home with the one rod the top-up hauled: it becomes two powder, not eighteen.
        g = self._home(ender_eye=14, blaze_rod=1, crafting_table=1)
        g.powder_exhausted = True
        g.nether_entered = True
        self.assertIn("craft_blaze_powder", candidates(g))
        self.assertEqual(resolve("craft_blaze_powder", g).args["items"][0]["count"], 2)

    def test_too_few_eyes_at_the_frame_re_supplies_instead_of_dead_ending(self) -> None:
        # The frame took more eyes than the bot carried in, so it now stands at the sockets with
        # none left. The old block re-offered an `activate` it could not complete; now it falls
        # through to procurement - hunt more pearls here, craft eyes, and `locate` carries the bot
        # back to finish the frame. No un-completable activate, and no locate on the frame itself.
        g = self._home(mobs=[("enderman", "hostile", 12, 18.0)])
        g.stronghold_estimated = True
        g.resource_found = {"end_portal_frame": 7}
        self.assertNotIn("activate_end_portal", candidates(g))
        self.assertNotIn("locate_stronghold", candidates(g))
        self.assertIn("hunt_enderman", candidates(g))

    def test_a_partial_fill_still_activates_while_eyes_remain(self) -> None:
        # Fewer eyes than empty sockets, but more than none: fill what the frame will take now.
        # Each pass fills >= 1 socket and the fill persists, so the activate <-> procurement loop
        # converges rather than waiting for a full set before placing anything.
        g = self._home(ender_eye=5)
        g.stronghold_estimated = True
        g.resource_found = {"end_portal_frame": 7}
        self.assertIn("activate_end_portal", candidates(g))
        self.assertNotIn("locate_stronghold", candidates(g))


class RodDemandSizedToTheFrameTest(unittest.TestCase):
    """How many rods the Nether owes: the eyes the frame still needs, and nothing more.

    The chain is rods -> powder -> pearls -> eyes -> frame, so the haul is a demand, not a goal
    number. Live, run123 (it23-31): one `blaze_rod` left in the pack from an earlier run, no
    powder, sixteen eyes owed. `_nether_haul_home` read the rod as a haul still on its way in,
    `craft_blaze_powder` demanded the whole nine-rod haul before it would craft, and
    `stow_blaze_powder` had no powder to deposit - the gate never opened, and ten ticks went to
    `wait` at (-10.4, 70, -51.5) with a burning portal and a complete kit.
    """

    BASE = (-197, -47, -98)
    CHEST = (-196, -46, -98)
    GATE = (-195, -46, -98)

    def _home(self, **inv) -> GameState:
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64, golden_helmet=1,
                     cooked_mutton=64, white_bed=1, diamond_pickaxe=2)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.chest_pos = self.CHEST
        g.dimension = "minecraft:overworld"
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  *self.GATE, 2.0)]}
        return g

    def test_a_leftover_rod_is_processed_then_banked(self) -> None:
        # The run123 tick, one step further: the rod becomes powder, the powder reaches the
        # chest, and the gate opens for the trip the eyes still owe.
        g = self._home(blaze_rod=1)
        self.assertTrue(_nether_haul_home(g))
        self.assertIn("craft_blaze_powder", candidates(g))
        self.assertEqual(resolve("craft_blaze_powder", g).args["items"][0]["count"], 2)
        g.inventory["blaze_rod"] = 0
        g.inventory["blaze_powder"] = 2
        self.assertIn("stow_blaze_powder", candidates(g))
        g.inventory["blaze_powder"] = 0
        g.powder_stowed = True
        g.powder_bank = 2
        self.assertFalse(_nether_haul_home(g))
        self.assertIn("enter_portal", candidates(g))

    def test_a_rod_that_cannot_be_crafted_is_not_a_haul(self) -> None:
        # No table in the pack: nothing can turn the rod into powder this tick, so holding the
        # gate shut on it buys a deadlock, not a haul.
        g = self._home(blaze_rod=1)
        g.inventory.pop("crafting_table")
        self.assertFalse(_nether_haul_home(g))
        self.assertIn("enter_portal", candidates(g))

    def test_paid_eyes_make_a_carried_rod_spare(self) -> None:
        # Sixteen eyes crafted: the rod in the pack is dead weight, and the trip is owed to the
        # frame, not to the pack.
        g = self._home(blaze_rod=1, crafting_table=1, ender_eye=16)
        self.assertFalse(_nether_haul_home(g))
        self.assertEqual(_rods_target(g), 0)

    def test_the_haul_is_sized_to_the_eyes_still_owed(self) -> None:
        g = self._home()
        self.assertEqual(_eyes_owed(g), 16)          # twelve sockets plus the throw buffer
        self.assertEqual(_rods_target(g), 8)         # two powder to a rod, not a flat nine
        g.inventory["ender_eye"] = 8
        self.assertEqual(_eyes_owed(g), 8)
        self.assertEqual(_rods_target(g), 4)

    def test_banked_powder_pays_for_the_eyes(self) -> None:
        g = self._home()
        g.powder_stowed = True
        g.powder_bank = 16
        self.assertEqual(_rods_target(g), 0)
        self.assertFalse(_nether_topup_wanted(g))
        # The gate is still open, but for pearls rather than rods: the wave ladder owns the
        # trip, not a second blaze hunt.
        self.assertTrue(_nether_pearl_wave_wanted(g))
        self.assertFalse(_pearl_wave_trip(g) is False and _nether_topup_wanted(g))

    def test_a_short_bank_reopens_the_gate(self) -> None:
        # Four eyes shattered on the throws and only two powder left in the chest: the frame
        # owes four powder, the bank covers two of them, so the top-up hauls one rod - not the
        # first trip's eight.
        g = self._home(ender_eye=8)
        g.powder_stowed = True
        g.powder_bank = 2
        g.stronghold_estimated = True
        self.assertEqual(_eyes_owed(g), 4)
        self.assertTrue(_nether_topup_wanted(g))
        self.assertEqual(_rods_target(g), 1)
        self.assertEqual(_powder_target(g), 4)

    def test_sockets_already_in_the_frame_are_not_owed_again(self) -> None:
        # The frame's own pre-filled sockets are never reported by the world, so the count is
        # booked from the eyes the bot put in; the portal opening ends the demand outright.
        g = self._home()
        g.stronghold_estimated = True
        g.frame_eyes = 9
        self.assertEqual(_frame_empty(g), 3)
        self.assertEqual(_eyes_owed(g), 3)
        self.assertEqual(_rods_target(g), 2)
        g.end_portal_active = True
        self.assertEqual(_eyes_owed(g), 0)
        self.assertEqual(_rods_target(g), 0)

    def test_the_pearl_hunt_stops_at_the_eyes_owed(self) -> None:
        g = self._home()
        g.stronghold_estimated = True
        g.frame_eyes = 10
        self.assertEqual(_pearls_owed(g), 2)

    def test_the_locate_carries_the_sockets_plus_the_buffer(self) -> None:
        g = self._home()
        g.powder_stowed = True
        g.powder_bank = 16
        g.inventory["ender_eye"] = 15
        self.assertNotIn("locate_stronghold", candidates(g))
        g.inventory["ender_eye"] = 16
        self.assertIn("locate_stronghold", candidates(g))
        g.frame_eyes = 4
        # Four sockets the frame already holds: twelve eyes are enough to travel.
        g.inventory["ender_eye"] = 12
        self.assertIn("locate_stronghold", candidates(g))


class DiamondsForTheSparePickaxeTest(unittest.TestCase):
    """The two gates for `mine_diamond` asked different questions, sixty-two times.

    Live: 62 escalations on `mine_diamond` between 08:51 and 12:38, one per tick, each handing a
    step to System 2 that provably could not run. `craft_pickaxe_spare` above it had six of the
    same between 08:38 and 08:49 and stopped the moment that fix went in, so the shape was known
    and simply not applied one line further up.

    The disagreement was not a missing check but a missing *reason*. The offer came from the
    Nether branch: the trip needs two pickaxes and the bot carries one, so the diamonds are wanted
    for the spare. `resolve` only knew the tech-tree order, which is finished - pickaxe, sword and
    the whole armour set - so it read "no next diamond item" and refused. Both reasons now live in
    `_mine_diamond_wanted`, and `resolve` names the same two when it counts.
    """

    BASE = (-197, -47, -98)
    GATE = (-195, 63, -97)

    def _stocked_but_for_the_pickaxe(self, **inv):
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64, white_bed=1,
                     golden_helmet=1, cooked_mutton=64, diamond_pickaxe=1, raw_gold=5,
                     diamond_sword=1, stick=4)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        g.nether_entered = False
        g.resources = {
            "nether_portal": [BlockHit("nether_portal", "nether_portal", *self.GATE, 2.0)],
            "white_bed": [BlockHit("white_bed", "white_bed", 225, 64, 223, 2.5)],
            "obsidian": [BlockHit("obsidian", "obsidian", 226, 64, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 65, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 66, 221, 2.0),
                         BlockHit("obsidian", "obsidian", 226, 67, 221, 2.0)],
            "diamond_ore": [BlockHit("diamond_ore", "diamond_ore", 190, 12, 190, 49.0)],
        }
        g.resource_found = {"diamond_ore": 5, "nether_portal": 6}
        return g

    def test_the_tech_tree_is_finished_and_the_dig_is_still_offered(self) -> None:
        # Pickaxe, sword and the whole armour set are done. The only diamonds left in the world are
        # the ones the trip needs, and the offer must say so.
        from mlaya.actions import _next_diamond_item
        g = self._stocked_but_for_the_pickaxe()
        self.assertIsNone(_next_diamond_item(g))
        self.assertIn("mine_diamond", candidates(g))

    def test_and_this_time_the_label_resolves(self) -> None:
        g = self._stocked_but_for_the_pickaxe()
        a = resolve("mine_diamond", g)
        self.assertIsNotNone(a)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args["block_name"], "diamond_ore")

    def test_it_asks_for_what_the_spare_pickaxe_still_costs(self) -> None:
        g = self._stocked_but_for_the_pickaxe()
        self.assertEqual(resolve("mine_diamond", g).args["count"], 3)
        g.inventory["diamond"] = 2
        self.assertEqual(resolve("mine_diamond", g).args["count"], 1)

    def test_with_both_pickaxes_nothing_is_wanted(self) -> None:
        from mlaya.actions import _mine_diamond_wanted
        g = self._stocked_but_for_the_pickaxe(diamond_pickaxe=2)
        self.assertFalse(_mine_diamond_wanted(g))
        self.assertIsNone(resolve("mine_diamond", g))

    def test_no_ore_in_reach_means_no_offer(self) -> None:
        from mlaya.actions import _mine_diamond_wanted
        g = self._stocked_but_for_the_pickaxe()
        g.resources.pop("diamond_ore")
        g.resource_found.pop("diamond_ore")
        self.assertFalse(_mine_diamond_wanted(g))
        self.assertNotIn("mine_diamond", candidates(g))

    def test_the_two_gates_cannot_drift_apart_again(self) -> None:
        # The shape that has now cost three escalations - `craft_pickaxe_spare`, `go_home` for the
        # crossing, `mine_diamond`. For every label the offer makes, the resolve must answer.
        from mlaya.actions import _mine_diamond_wanted
        for pickaxes in (1, 2):
            with self.subTest(diamond_pickaxes=pickaxes):
                g = self._stocked_but_for_the_pickaxe(diamond_pickaxe=pickaxes)
                self.assertEqual("mine_diamond" in candidates(g),
                                 resolve("mine_diamond", g) is not None)


class GoHomeIsAnOverworldOnlyStepTest(unittest.TestCase):
    """`go_home` names the base's coordinates, and `base` is one field for both worlds.

    Live, run70: the bot crossed into the Nether and `go_home` offered itself 24 times, navigating
    to the *Overworld* base cell - `(226,63,222)` read as a Nether coordinate, 778 m from where the
    body actually stood. The pathfinder found *a* path to *some* Nether cell with those numbers
    and reported success, so the run counted it as progress and marched further out.

    Then the loop: `go_home` succeeds and moves the body, `leave_nether` fails with
    `[NAVIGATION_STOPPED]`, and the pair alternates - `go_home` ok, `leave_nether` refused, from
    tick 30 to 47 without a fortress in sight and without a blaze rod in the pack.

    `_leave_portal_target` already carries this guard for exactly this reason and writes down why
    it was added after run53. `go_home` is the same mistake in the neighbouring label: the
    diagnosis existed, in the same file, and was not applied.
    """

    BASE = (-197, -47, -98)

    def _live_nether_tick(self) -> GameState:
        """The exact state run70 offered `go_home` from: Overworld base (226,63,222) still held,
        a base step genuinely owed, no bed in the Nether's sight, and the body at (207,207) - a
        Nether cell the Overworld's coordinates happen to name."""
        g = at_home(geared(golden_helmet=0, diamond_pickaxe=2, flint_and_steel=1,
                           cobblestone=64, white_bed=1, cooked_mutton=24), base=(226, 63, 222))
        g.dimension = "minecraft:the_nether"
        g.nether_entered = True
        g.nether_arrival = (33, 55, 34)
        g.pos = (207.5, 49.0, 207.5)
        # run70 had re-adopted this base from the lit gate, which is the first clause of
        # `_base_steps_ready` - so a base step really was owed, and that is precisely why the walk
        # was offered.
        g.base_adopted = True
        return g

    def test_no_offer_from_the_nether(self) -> None:
        from mlaya.actions import _go_home_wanted, _base_steps_ready
        g = self._live_nether_tick()
        # The premise, so the test cannot pass for the wrong reason: a base step really is owed,
        # which is exactly why the walk was offered - and why it was wrong.
        self.assertTrue(_base_steps_ready(g))
        self.assertFalse(_go_home_wanted(g))
        self.assertNotIn("go_home", candidates(g))

    def test_and_no_resolve_even_if_something_forced_the_label(self) -> None:
        self.assertIsNone(resolve("go_home", self._live_nether_tick()))

    def test_the_nether_way_home_is_the_portal_not_a_walk(self) -> None:
        # What replaces it: `leave_nether` crosses through the arrival cell.
        g = at_home(geared(white_bed=1))
        g.dimension = "minecraft:the_nether"
        g.nether_entered = True
        g.nether_arrival = (33, 55, 34)
        g.pos = (33.5, 55.0, 34.5)
        self.assertNotIn("go_home", candidates(g))

    def test_the_overworld_still_walks_home(self) -> None:
        # Night, thirty metres out, bed at the base: the live reason. Nothing is owed at the base
        # and no crossing is ready - thirty metres from home for no work is the case the run70
        # fix has to keep working, and the only thing that makes it wanted is the bed.
        from mlaya.actions import _go_home_wanted
        bx, by, bz = self.BASE
        g = at_home(geared(white_bed=1))
        g.dimension = "minecraft:overworld"
        g.phase = "night"
        g.resources = {"white_bed": [BlockHit("white_bed", 3.0, bx - 3, by + 1, bz + 2, 3.0)]}
        g.pos = (bx + 30.5, by + 1.0, bz)
        self.assertTrue(_go_home_wanted(g))
        a = resolve("go_home", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["z"]), (self.BASE[0], self.BASE[2]))

    def test_the_two_gates_agree_across_the_dimension(self) -> None:
        # Daytime and thirty metres out with no chest at the base: a base step is genuinely owed,
        # so the walk is the right answer and both gates must say so - in the Overworld, and must
        # both refuse it in the Nether, where those coordinates name somewhere else entirely.
        for dim in ("minecraft:overworld", "minecraft:the_nether"):
            with self.subTest(dimension=dim):
                if dim == "minecraft:the_nether":
                    g = self._live_nether_tick()
                else:
                    # The same owed base step, in the dimension where the coordinates mean it.
                    g = at_home(geared(golden_helmet=0, diamond_pickaxe=2, flint_and_steel=1,
                                       cobblestone=64, white_bed=1, cooked_mutton=24))
                    g.dimension = "minecraft:overworld"
                    g.phase = "day"
                    g.base_adopted = True
                    g.pos = (226.5 + 30.0, 64.0, 222.0)
                self.assertEqual("go_home" in candidates(g),
                                 resolve("go_home", g) is not None)


class NeutralCrowdingTest(unittest.TestCase):
    """The hunt guards count only hostiles that could actually join the fight.

    Live, run119 it221-229: a blaze 3.9 m away, 53 zombified piglins at 9.1 m, health 20. Every
    guard read that crowd as a second fight and refused the hunt for nine ticks, while the blaze
    shot at a bot that never asked for a fight. Piglins and zombified piglins turn on a player
    only when provoked - a hit, or a ghast fireball that bounces back - and a fortress floor is
    never empty of them, so counting them as company closes every hunt in the place where hunting
    is the whole point of the trip.
    """

    def test_piglins_do_not_close_the_blaze_hunt(self) -> None:
        g = gs(health=20.0, mobs=[("blaze", "hostile", 1, 3.9),
                                  ("zombified_piglin", "hostile", 53, 9.1),
                                  ("piglin", "hostile", 1, 12.0)])
        self.assertTrue(_safe_blaze_hunt(g))
        self.assertEqual(_crowding_hostiles(g, "blaze", 16), [])

    def test_a_hostile_that_does_turn_on_still_closes_it(self) -> None:
        g = gs(health=20.0, mobs=[("blaze", "hostile", 1, 3.9),
                                  ("wither_skeleton", "hostile", 2, 9.1)])
        self.assertFalse(_safe_blaze_hunt(g))

    def test_the_string_hunts_ignore_piglins(self) -> None:
        g = gs(health=20.0, mobs=[("spider", "hostile", 1, 6.0),
                                  ("zombified_piglin", "hostile", 40, 8.0)])
        self.assertTrue(_safe_spider_hunt(g))
        self.assertTrue(_safe_string_hunt(g))
        # A real pack still refuses the strict guard; the bow's wider rule allows a little company.
        g.mobs.append(("zombie", "hostile", 3, 9.0))
        self.assertFalse(_safe_spider_hunt(g))
        self.assertTrue(_safe_string_hunt(g))

    def test_the_zombie_food_hunt_ignores_piglins(self) -> None:
        g = gs(health=20.0, mobs=[("zombie", "hostile", 1, 5.0),
                                  ("zombified_piglin", "hostile", 12, 7.0)])
        self.assertTrue(_safe_zombie_hunt(g))
        g.mobs.append(("skeleton", "hostile", 3, 8.0))
        self.assertFalse(_safe_zombie_hunt(g))

    def test_a_hurt_bot_is_still_refused(self) -> None:
        # The exclusion is about who is crowding, not about lowering the health floor.
        g = gs(health=10.0, mobs=[("blaze", "hostile", 1, 3.9)])
        self.assertFalse(_safe_blaze_hunt(g))


class TheOfferAndTheResolveMustAgreeTest(unittest.TestCase):
    """Four labels have now been offered and refused on the same tick. This is the shape.

    An offered label that `resolve` cannot answer looks exactly like a dead end, and costs a model
    call per tick while it lasts. Four instances, all found by measuring which labels were offered
    and could not be built:

      craft_pickaxe_spare  six escalations, 08:38-08:49, stopped when that fix went in
      go_home              the crossing reason existed in `candidates` and not in `resolve`
      mine_diamond         62 escalations, 08:51-12:38, the same disease one line above
      hunt_string, drop_junk   offered through the wider guard, resolved through the stricter one

    `drop_junk` was mine, from today: the whitelist went into `_sheddable`, which gates the offer,
    but the resolve still named only `JUNK_ITEMS`. The whitelist may only ever feed the chest,
    because a drop is not undoable - so the drop gate has to name what the drop will name.
    """

    BASE = (-197, -47, -98)

    def test_a_string_hunt_offered_through_the_wide_guard_resolves(self) -> None:
        # A spider at 14 m with a friend: refused by the strict bed-chain guard, allowed by the
        # bow's own rule. The offer used the wide one and the resolve the strict one, so eight
        # ticks went to System 2 for a hunt that was fine.
        from mlaya.actions import _safe_spider_hunt, _safe_string_hunt
        g = at_home(geared(string=2, stick=4))
        g.health = 20.0
        g.mobs = [("spider", "hostile", 14.0, 14.0), ("zombie", "hostile", 15.0, 15.0)]
        self.assertTrue(_safe_string_hunt(g))
        self.assertFalse(_safe_spider_hunt(g))
        self.assertIsNotNone(resolve("hunt_string", g))

    def test_a_hurt_bot_is_still_refused(self) -> None:
        # The wide guard does not mean no guard: below 16 health a spider is not worth it.
        g = at_home(geared(string=2, stick=4))
        g.health = 12.0
        g.mobs = [("spider", "hostile", 6.0, 6.0)]
        self.assertNotIn("hunt_string", candidates(g))

    def test_the_whitelist_never_reaches_the_ground(self) -> None:
        from mlaya.actions import _droppable, _surplus_besides_keep
        # A full pack, no chest in reach, and the only thing spare is an unnamed item.
        # Only an unnamed item is spare. `tuff` would not do: it is on `JUNK_ITEMS`, so it is
        # legitimately droppable and the test would prove nothing.
        g = at_home(geared(oak_sapling=3))
        g.chest_pos = None
        g.slots_used = INVENTORY_NEAR_FULL
        self.assertTrue(_surplus_besides_keep(g, g.inventory))
        self.assertFalse(_droppable(g, g.inventory))
        self.assertNotIn("drop_junk", candidates(g))

    def test_named_junk_still_reaches_the_ground_without_a_chest(self) -> None:
        g = at_home(geared(tuff=64))
        g.chest_pos = None
        g.slots_used = INVENTORY_NEAR_FULL
        self.assertIn("drop_junk", candidates(g))
        a = resolve("drop_junk", g)
        self.assertEqual(a.tool, "drop_item")
        self.assertIn("tuff", [i["item_name"] for i in a.args["items"]])

    def test_the_same_whitelist_item_goes_to_the_chest_when_there_is_one(self) -> None:
        # The whole point of the split: same item, same moment, different fate - recoverable or
        # not. With a chest standing there the item is banked, and nothing is dropped.
        g = at_home(geared(tuff=64, oak_sapling=3))
        g.chest_pos = (self.BASE[0] + 2, self.BASE[1], self.BASE[2] + 2)
        g.slots_used = INVENTORY_NEAR_FULL
        self.assertIn("stow_surplus", candidates(g))
        self.assertNotIn("drop_junk", candidates(g))
        a = resolve("stow_surplus", g)
        self.assertIn("oak_sapling", [i["item_name"] for i in a.args["items"]])

    def test_the_nether_arrival_is_in_the_logged_state(self) -> None:
        # Not a label but the same lesson: `leave_nether` asks `resolve` for the cell it arrived
        # on when no portal block is in scan, and that cell was not written to the log - so a
        # rebuilt tick could never answer it and read as a broken gate rather than a missing field.
        from mlaya.perception import GameState
        gs = GameState(nether_entered=True, nether_arrival=(33, 55, 34), nether_ticks=4)
        rebuilt = GameState.from_state(gs.as_state())
        self.assertEqual(rebuilt.nether_arrival, (33, 55, 34))
        self.assertTrue(rebuilt.nether_entered)
        self.assertEqual(rebuilt.nether_ticks, 4)


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
        # Order on purpose: it reads `_expedition_prepared`, so the pickaxe is the last thing
        # before the crossing and not something to chase before the helmet is on. And it is asked
        # for only when it can be built - see the corrected test beside it.
        self.assertIn("craft_pickaxe_spare",
                      candidates(self._burning_gate(diamond_pickaxe=1, diamond=3, stick=4)))

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

    def test_a_cook_far_from_the_base_never_names_the_base_cell(self) -> None:
        # Knowing the cell is not standing at it. Live, run58: a `cook_food` at (357,70,362) with
        # `furnace_pos` (229,62,222) in the state - a smelt aimed 190 m away, answered ok. Away
        # from the base the carried furnace is the branch that was always there and always worked.
        g = self._kit(furnace_pos=self.FURNACE)      # away: pos is (150,64,150)
        a = resolve("smelt_iron", g)
        self.assertIsNotNone(a)
        self.assertTrue(a.args["temporary_workstation"])
        self.assertNotIn("x", a.args)

    def test_a_cook_one_step_outside_the_base_still_uses_a_temporary_furnace(self) -> None:
        # `_at_base` is the test, not the distance: the cell has to be reachable to be worth naming.
        g = self._kit(furnace_pos=self.FURNACE, at_base=True)
        g.pos = (self.BASE[0] + 0.5 + 12.0, float(self.BASE[1] + 1), self.BASE[2] + 0.5)
        a = resolve("smelt_iron", g)
        self.assertTrue(a.args["temporary_workstation"])

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


class PortalLitIsAWorldFactTest(unittest.TestCase):
    """A lit gate stays lit however far the bot has walked from it.

    The flag was a 64 m proximity read, which makes it a statement about where the bot stands
    rather than about the world - and two gates read it as a fact. `light_portal` is offered when
    it reads false, so a bot 190 m out believed it had an unlit gate and would have set about
    lighting one that was already burning. That is run22's shape in a different costume.

    The state carries two readings of the same block: the radius-filtered hits and `resource_found`,
    the raw count over every loaded chunk. The diamonds and the gold already use the second for
    exactly this reason - a deposit too far to list still proves the ore is there.
    """

    BASE = (226, 63, 222)
    GATE = (228, 64, 222)

    def _far_away(self):
        """At (357,70,362), 190 m off, with the gate outside the scan radius but in a loaded chunk."""
        g = at_home(geared(flint_and_steel=1, cobblestone=64, obsidian=10, diamond_pickaxe=2,
                            golden_helmet=1, bow=1, arrow=16, cooked_mutton=64, white_bed=1),
                    base=self.BASE)
        g.pos = (357.0, 70.0, 362.0)
        g.dimension = "minecraft:overworld"
        g.health = 20.0
        g.food = 18.0
        g.phase = "day"
        g.resources = {}                                   # nothing inside RESOURCE_RADIUS
        g.resource_found = {"nether_portal": 6}           # six blocks, counted in a loaded chunk
        return g

    def test_the_gate_is_lit_from_across_the_map(self) -> None:
        self.assertTrue(_portal_lit(self._far_away()))

    def test_light_portal_is_not_offered_for_a_gate_that_already_burns(self) -> None:
        g = self._far_away()
        g.obsidian = 10
        self.assertNotIn("light_portal", candidates(g))

    def test_a_gate_nobody_has_seen_is_not_claimed(self) -> None:
        # The other half: the count has to mean something. No portal in the scan and none counted
        # is an unlit gate, and the frame clause still owes the walk home.
        g = self._far_away()
        g.resource_found = {}
        self.assertFalse(_portal_lit(g))
        self.assertIn("go_home", candidates(g))


class SmokerBesideTheFurnaceTest(unittest.TestCase):
    """A smoker is not a better furnace, it is a different one, and it stands next to it.

    The user proposed it: a food station at the base, so the bot comes home and cooks in bulk what
    it gathered out there. The correction of the day was that a smoker processes food only - which
    is exactly why it belongs *beside* the furnace rather than replacing it. The tech tree still
    needs a furnace for iron, and a smoker would refuse the ore.

    Cooking is the only smelting this run does in bulk - live, twenty cooked items in twenty ticks,
    one per tick at eighteen seconds - and every one of them is food.
    """

    BASE = (-197, -47, -98)
    FURNACE = (-197, -46, -99)
    SMOKER = (-196, -46, -99)

    def _base_with_raw_meat(self, **inv) -> GameState:
        # The wood the run already hoards also carries the smoker recipe: server's recipe is
        # four logs plus a carried furnace, and one more furnace stays behind as station.
        g = at_home(geared(oak_log=12, **inv))
        g.chest_pos = (self.BASE[0] + 2, self.BASE[1], self.BASE[2] + 2)
        g.furnace_pos = self.FURNACE
        g.smoker_pos = None
        g.food = 20.0
        return g

    def test_the_smoker_stands_beside_the_furnace(self) -> None:
        # The furnace takes (bx, by+1, bz-1), the chest sits at bz+1 and the portal's frame plane
        # runs along z == bz. The one cell free of all three is next to the furnace on the other
        # side of x - which is where the user wanted the bot to come back and cook.
        from mlaya.actions import _smoker_site
        g = self._base_with_raw_meat()
        self.assertEqual(_smoker_site(g), self.SMOKER)
        self.assertNotEqual(_smoker_site(g), g.furnace_pos)

    def test_a_bed_on_that_cell_still_refuses(self) -> None:
        from mlaya.actions import _smoker_site
        g = self._base_with_raw_meat()
        g.resources = {"white_bed": [BlockHit("white_bed", 1.0,
                                              self.SMOKER[0], self.SMOKER[1],
                                              self.SMOKER[2], 1.0)]}
        self.assertIsNone(_smoker_site(g))

    def test_it_is_built_once_there_is_food_a_second_furnace_and_four_logs(self) -> None:
        # The server's recipe, not a fiction: four logs *and* one furnace go into the smoker,
        # which is why a second furnace has to be carried - run78's "four planks and a coal"
        # was answered `[CRAFT_MATERIALS_MISSING] oak_log x3` against one carried log.
        g = self._base_with_raw_meat(beef=12, furnace=2)
        self.assertIn("craft_smoker", candidates(g))
        a = resolve("craft_smoker", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "smoker"}])

    def test_one_carried_furnace_is_eaten_by_the_recipe_so_is_not_enough(self) -> None:
        # The furnace in the recipe is a material. With one furnace the smoker would stand
        # where the cook chain's station stood, so the craft that makes cooking faster takes
        # away the station cooking needs. Only the second furnace makes it a gain.
        g = self._base_with_raw_meat(beef=12, furnace=1)
        self.assertNotIn("craft_smoker", candidates(g))
        self.assertIsNone(resolve("craft_smoker", g))

    def test_the_recipe_wants_logs_and_a_planks_hoard_does_not_answer_for_them(self) -> None:
        # run78 carried oak_log 1 and the server named `oak_log x3` missing - four logs are
        # the recipe, and planks (however many `_plank_units` counts) are not part of it.
        g = self._base_with_raw_meat(beef=12, furnace=2)
        g.inventory["oak_log"] = 1
        self.assertNotIn("craft_smoker", candidates(g))
        self.assertIsNone(resolve("craft_smoker", g))

    def test_without_food_waiting_there_is_nothing_to_speed_up(self) -> None:
        g = self._base_with_raw_meat(cobblestone=40, furnace=2)
        self.assertNotIn("craft_smoker", candidates(g))

    def test_craft_first_then_place_then_never_again(self) -> None:
        # The order is the point: there is nothing to stand until the recipe has run, so
        # `place_smoker` is not offered against a base the bot is merely standing next to.
        g = self._base_with_raw_meat(beef=12, furnace=2)
        self.assertIn("craft_smoker", candidates(g))
        self.assertNotIn("place_smoker", candidates(g))

        g.inventory["smoker"] = 1                 # the recipe produced one
        self.assertIn("place_smoker", candidates(g))
        a = resolve("place_smoker", g)
        self.assertEqual(a.tool, "place_block")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.SMOKER)

        g.smoker_pos = self.SMOKER                 # and it is latched on that success
        self.assertNotIn("place_smoker", candidates(g))
        self.assertNotIn("craft_smoker", candidates(g))

    def test_food_goes_into_the_smoker_and_ore_still_goes_into_the_furnace(self) -> None:
        # The split that matters. A smoker would refuse raw iron, so passing its cell to
        # `smelt_iron` would trade a faster cook for a tech tree that cannot finish.
        g = self._base_with_raw_meat(beef=12, coal=3)
        g.smoker_pos = self.SMOKER
        cook = resolve("cook_food", g)
        self.assertEqual((cook.args["x"], cook.args["y"], cook.args["z"]), self.SMOKER)

        g.inventory["raw_iron"] = 3
        iron = resolve("smelt_iron", g)
        self.assertEqual((iron.args["x"], iron.args["y"], iron.args["z"]), self.FURNACE)

    def test_without_a_smoker_food_still_cooks_in_the_furnace(self) -> None:
        g = self._base_with_raw_meat(beef=12, coal=3)
        cook = resolve("cook_food", g)
        self.assertEqual((cook.args["x"], cook.args["y"], cook.args["z"]), self.FURNACE)

    def test_away_from_the_base_nothing_cooks_without_a_carried_furnace(self) -> None:
        # run78's eight `[WORKSTATION_NOT_CARRIED]`: the station stood at the base and the cook
        # ran where the bot stood, and the temporary workstation `_smelt_args` asks for there
        # is a carried furnace this state does not carry. Naming the remote cell is the run58
        # bug; pretending the portable one exists is the run78 bug. Both refuse here.
        g = self._base_with_raw_meat(beef=12, coal=3)
        g.smoker_pos = self.SMOKER
        g.pos = (120.5, 64.0, 120.5)
        self.assertNotIn("cook_food", candidates(g))
        self.assertIsNone(resolve("cook_food", g))

    def test_the_carried_furnace_is_the_station_that_travels(self) -> None:
        # The run58 lesson keeps its half: with the furnace in the pack, away from every
        # standing station is not a refusal but the portable branch that always worked.
        g = self._base_with_raw_meat(beef=12, coal=3, furnace=1)
        g.pos = (120.5, 64.0, 120.5)
        cook = resolve("cook_food", g)
        self.assertTrue(cook.args["temporary_workstation"])
        self.assertNotIn("x", cook.args)


class NightSendsItHomeTest(unittest.TestCase):
    """Home is where the bed is. That counts as a reason to be there with no work waiting.

    The sleep rule went in first and measured nothing: over nineteen night ticks in run67 the bot
    held ten or eleven metres from the base - `mine_stone` at sixty-one - and slept on none of
    them. `_sleep_ready` demands standing on the base with a lit gate, and `go_home` was only
    ever offered when base *work* was waiting or the Nether crossing was ready. Neither was true
    at night, so the candidates were `explore` and `craft_arrows`, and a skeleton found the bot
    eleven metres from home with a full diamond kit it never got to use.

    A half-measure: the rule was right and its precondition was impossible. Home is a place, not
    a to-do list.
    """

    BASE = (-197, -47, -98)

    def _out_at_night(self, **inv) -> GameState:
        g = lit_portal(at_home(geared(flint_and_steel=1, white_bed=1, oak_log=8, coal=4, **inv)))
        bx, by, bz = self.BASE
        g.chest_pos = (bx + 2, by, bz + 2)
        g.furnace_pos = (bx, by + 1, bz - 1)
        g.resources = {"white_bed": [BlockHit("white_bed", 3.0, bx - 3, by + 1, bz + 2, 3.0)],
                       "nether_portal": [BlockHit("nether_portal", 4.0,
                                                  bx + 2, by + 1, bz + 1, 4.0)]}
        g.resource_found = {"nether_portal": 6}
        g.phase = "night"
        g.food = 20.0
        g.pos = (bx + 11.5, by + 1.0, bz)          # eleven metres out, as live
        return g

    def test_eleven_metres_out_at_night_the_walk_home_is_offered(self) -> None:
        # The live case, to the metre. Nothing is owed at the base and no crossing is ready, so
        # before this change the list was `explore`.
        g = self._out_at_night()
        self.assertIn("go_home", candidates(g))

    def test_the_walk_home_actually_targets_the_base(self) -> None:
        g = self._out_at_night()
        a = resolve("go_home", g)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["z"]), (self.BASE[0], self.BASE[2]))

    def test_daytime_is_unchanged(self) -> None:
        # In daylight there is a reason to be out, and no reason invented here.
        g = self._out_at_night()
        g.phase = "day"
        self.assertNotIn("go_home", candidates(g))

    def test_the_night_reason_carries_it_on_its_own(self) -> None:
        # The live state, isolated: no base work owed, no crossing ready, and only the bed
        # making the walk worth taking. Daytime removes exactly that one reason.
        from mlaya.actions import _base_steps_ready, _nether_departure_ready, _go_home_wanted
        g = self._out_at_night()
        self.assertFalse(_base_steps_ready(g))
        self.assertFalse(_nether_departure_ready(g))
        self.assertTrue(_go_home_wanted(g))
        g.phase = "day"
        self.assertFalse(_go_home_wanted(g))

    def test_a_missing_bed_is_its_own_reason_to_come_home(self) -> None:
        # Not a counterexample but a second door: with the bed gone, `place_bed` becomes an owed
        # base step, so the walk is wanted for a different reason - and stays wanted.
        from mlaya.actions import _base_steps_ready
        g = self._out_at_night()
        g.resources = {"nether_portal": g.resources["nether_portal"]}
        self.assertTrue(_base_steps_ready(g))
        self.assertIn("go_home", candidates(g))

    def test_standing_on_the_base_needs_no_walk(self) -> None:
        g = self._out_at_night()
        g.pos = (self.BASE[0] + 0.5, self.BASE[1] + 1.0, self.BASE[2] + 0.5)
        self.assertNotIn("go_home", candidates(g))

    def test_arriving_home_is_what_unlocks_the_sleep(self) -> None:
        # The whole point, asserted end to end: away from the base the walk is offered, and on it
        # sleep is the only thing on the list.
        away = self._out_at_night()
        self.assertIn("go_home", candidates(away))
        away.pos = (self.BASE[0] + 0.5, self.BASE[1] + 1.0, self.BASE[2] + 0.5)
        self.assertEqual(candidates(away), ["sleep"])


class NightIsAClockNotACandidateTest(unittest.TestCase):
    """At night, with a bed underfoot, the run has nothing to choose between.

    Live, across 462 night ticks: 175 were sleep-ready, `sleep` was offered on 164, and it was
    taken nine times - four of those refused because `hostile_reflex` or `hunger_reflex` already
    owned the body. The bot chose `craft_arrows` on four ticks running with a bed under it.

    A night costs about forty decisions at eighteen seconds; a sleep costs one. Everything on that
    list has no deadline - it is all still waiting at dawn - while every tick spent on it is a
    tick standing in the dark beside mobs the reflex would otherwise have met in bed.

    Owning the tick outright is safe because everything genuinely time-critical has already
    returned by this point: the portal and water rescues above, and the event-driven reactions
    immediately before - drops before they despawn, a pickaxe about to break, gear at a death
    site. Those are `return`s, not appends, and that is the reason this can be a `return` too.
    """

    BASE = (-197, -47, -98)

    def _night_at_base(self, **inv) -> GameState:
        g = lit_portal(at_home(geared(flint_and_steel=1, white_bed=1, oak_log=8, **inv)))
        g.chest_pos = (self.BASE[0] + 2, self.BASE[1], self.BASE[2] + 2)
        g.furnace_pos = (self.BASE[0], self.BASE[1] + 1, self.BASE[2] - 1)
        # A bed *standing* at the base, not carried: `_bed_at_base` reads the scan.
        bx, by, bz = self.BASE
        # The bed has to stand clear of the lit gate: a bed within BED_PORTAL_CLEARANCE of a
        # burning portal respawns the bot into it, and `_bed_bad_spot` refuses it as the gate bed.
        g.resources = {"white_bed": [BlockHit("white_bed", 3.0, bx - 3, by + 1, bz + 2, 3.0)],
                       "nether_portal": [BlockHit("nether_portal", 4.0,
                                                  bx + 2, by + 1, bz + 1, 4.0)]}
        # `_portal_lit` reads both readings of a block: the radius-filtered hits and the raw
        # `resource_found` count. The gate being lit is what moves the bed to the base.
        g.resource_found = {"nether_portal": 6}
        g.phase = "night"
        g.food = 20.0
        return g

    def test_nothing_competes_with_sleep_at_night(self) -> None:
        # The live case: the arrow bench is right there and the bed is right here.
        g = self._night_at_base(stick=12, flint=6, feather=8, coal=4, beef=4)
        self.assertEqual(candidates(g), ["sleep"])

    def test_a_hungry_bot_eats_and_then_sleeps(self) -> None:
        # The meal owns the tick (`_must_eat`); the sleep follows once the hunger is fed.
        g = self._night_at_base(cooked_mutton=4)
        g.food = 9.0
        self.assertEqual(candidates(g), ["eat_food"])
        g.food = 20.0
        self.assertEqual(candidates(g), ["sleep"])

    def test_drops_still_outrank_sleep_before_they_despawn(self) -> None:
        # The reflex killed something and its items evaporate in minutes. A sleep would throw them
        # away for the sake of a night that is going to pass anyway.
        g = self._night_at_base()
        g.drops = 2
        self.assertEqual(candidates(g), ["collect_drops"])

    def test_a_pickaxe_about_to_break_still_outranks_sleep(self) -> None:
        # The reaction the mine depends on: a pickaxe that breaks mid-run takes the bot down to
        # stone and the whole chain with it. It has to be able to interrupt the night.
        g = self._night_at_base()
        g.tools = {"pickaxe": "stone", "sword": "stone"}
        g.tool_meta = {"pickaxe": {"item": "stone_pickaxe", "tier": "stone",
                                   "left": 15.0, "max": 131.0}}
        g.inventory = {"crafting_table": 1, "cobblestone": 3, "acacia_log": 1}
        self.assertEqual(candidates(g), ["replace_tool"])

    def test_daytime_is_untouched(self) -> None:
        g = self._night_at_base(coal=4)
        g.phase = "day"
        self.assertNotIn("sleep", candidates(g))

    def test_no_bed_means_no_clock(self) -> None:
        g = self._night_at_base()
        g.resources = {}
        self.assertNotIn("sleep", candidates(g))

    def test_the_nether_has_no_nights(self) -> None:
        g = self._night_at_base()
        g.dimension = "minecraft:the_nether"
        self.assertNotIn("sleep", candidates(g))


class SleepReachTest(unittest.TestCase):
    """The night offer asks for a bed the `sleep` server can find, and honours the withholding.

    run86 started at night at (-91,68,-5), the old base's bed 51 blocks away at (-43,71,12):
    the scan saw it, the server searches 32 blocks, and the exclusive night offer answered
    `[BED_NOT_FOUND]` sixteen ticks running. The stuck guard dropped `sleep` at it2, but the
    night offer returns ahead of the withholding filter, so the drop changed nothing.
    """

    def _night(self, bed_distance: float | None = None, **inv) -> GameState:
        g = geared(**inv)
        g.pos = (-91.0, 68.0, -5.0)
        g.phase = "night"
        g.food = 20.0
        if bed_distance is not None:
            g.resources = {"white_bed": [BlockHit("white_bed", "white_bed", -43, 71, 12,
                                                  bed_distance)]}
        return g

    def test_a_bed_beyond_the_servers_search_is_no_sleep(self) -> None:
        g = self._night(bed_distance=51.4)
        self.assertNotIn("sleep", candidates(g))
        self.assertIsNone(resolve("sleep", g))

    def test_a_bed_within_the_servers_search_owns_the_night(self) -> None:
        g = self._night(bed_distance=20.0)
        self.assertEqual(candidates(g), ["sleep"])

    def test_a_carried_bed_sleeps_wherever_the_scan_looks(self) -> None:
        g = self._night(bed_distance=51.4, white_bed=1)
        self.assertEqual(candidates(g), ["sleep"])

    def test_a_withheld_sleep_gives_the_night_back_to_the_offer(self) -> None:
        g = self._night(bed_distance=20.0)
        g.stuck_labels = frozenset({"sleep"})
        offered = candidates(g)
        self.assertNotIn("sleep", offered)
        self.assertTrue(offered)


class CoalIsCookingFuelTooTest(unittest.TestCase):
    """Coal smelts eight items; a log smelts one. That is the whole difference.

    `mine_coal` used to be gated on `raw_iron >= 1`, which made coal iron-only fuel. Live: the run
    went 4, 9, 17, 25, 32, 36, 44, 52, 55, 62, 69 oak logs - burned one per cooked meal - with
    `raw_iron` at 0 and `coal` at 0 on every tick of the window, while `coal_ore` sat inside the
    scan radius on all thirty of them. Not a missing idea: a gate that never opened.

    Whatever is about to be smelted now counts, not just ore. The smoker was considered and left
    out: it processes food only, and the carried furnace already covers both cases through
    `temporary_workstation`.
    """

    def _state(self, **inv) -> GameState:
        g = at_home(geared(**inv))
        g.tools = {"pickaxe": "stone", "sword": "stone"}
        g.resources = {"coal_ore": [BlockHit("coal_ore", 12.0, 12, 40, 41, 12.0)]}
        g.resource_found = {"coal_ore": 9}
        return g

    def test_raw_meat_in_the_pack_mints_the_coal_to_cook_it(self) -> None:
        # The live case: no iron anywhere, a stack of beef, and coal ore ten steps away.
        g = self._state(beef=20)
        self.assertNotIn("raw_iron", g.inventory)
        self.assertIn("mine_coal", candidates(g))

    def test_cooking_fuel_already_carried_stops_the_mining(self) -> None:
        for name in ("coal", "charcoal"):
            with self.subTest(fuel=name):
                g = self._state(beef=20, **{name: 3})
                self.assertNotIn("mine_coal", candidates(g))

    def test_iron_still_mints_its_own_fuel(self) -> None:
        # The original gate is untouched: ore needs a fire just as much as meat does.
        g = self._state(raw_iron=3)
        self.assertIn("mine_coal", candidates(g))

    def test_nothing_to_smelt_means_no_coal_run(self) -> None:
        # Neither a meal nor ore: coal would be a slot for nothing.
        g = self._state(cobblestone=30)
        self.assertNotIn("mine_coal", candidates(g))

    def test_no_coal_ore_in_reach_means_no_coal_run(self) -> None:
        g = self._state(beef=20)
        g.resources = {}
        g.resource_found = {}
        self.assertNotIn("mine_coal", candidates(g))

    def test_coal_is_not_surplus_the_next_tick(self) -> None:
        # It has to survive to the furnace. The whitelist would otherwise bank it on the very next
        # tick - it is a stack with no recipe behind it - and the run would be back to logs.
        from mlaya.actions import _surplus_besides_keep
        g = self._state(coal=3)
        self.assertNotIn("coal", _surplus_besides_keep(g, g.inventory))
        self.assertIsNone(resolve("stow_surplus", g))   # nothing worth banking at all


class PackDisciplineTest(unittest.TestCase):
    """The pack is thirty slots of twenty-nine stacks, so the problem is types, not volume.

    Two live findings put here. A stack of 68 cobblestone costs two slots for the sake of four
    blocks, and `HOARD_KEEP` capped it at 64 - but the cap only ever fired inside the *full pack*
    branch, so with the chest known and the bot standing on the base it drifted 60, 65, 67, 68, 72
    and nothing noticed. And the pack held six items no recipe wants that `JUNK_ITEMS` never named
    (`oak_sapling`, `crying_obsidian`, `light_gray_wool`, `leather`, `stone_bricks`,
    `stone_brick_slab`) - a blacklist can only name the rubbish it already thought of.

    The whitelist is deliberately one-way: it only ever feeds the chest. A wrong guess lands in a
    container three steps away whose contents the server keeps in its own database, and costs one
    `withdraw` later. `drop_junk` still goes by `JUNK_ITEMS` alone, because a drop is not undoable.
    """

    BASE = (-197, -47, -98)
    CHEST = (-195, -46, -97)

    def _at_chest(self, **inv) -> GameState:
        g = at_home(geared(**inv))
        g.chest_pos = self.CHEST
        g.slots_used = 12          # plenty of room: this is not a relief case
        return g

    def test_a_stack_over_its_cap_is_stowed_without_a_full_pack(self) -> None:
        g = self._at_chest(cobblestone=68)
        self.assertLess(g.slots_used, INVENTORY_NEAR_FULL)
        self.assertIn("stow_surplus", candidates(g))
        a = resolve("stow_surplus", g)
        self.assertEqual(a.args["operation"], "deposit")
        self.assertIn({"item_name": "cobblestone", "count": 4}, a.args["items"])

    def test_a_stack_at_its_cap_is_left_alone(self) -> None:
        # The whole point of the cap: 64 blocks is one slot, and taking the stack below that would
        # mean mining for material the expedition needs.
        g = self._at_chest(cobblestone=64)
        self.assertNotIn("stow_surplus", candidates(g))

    def test_an_overfull_stack_with_no_chest_is_left_on_the_ground(self) -> None:
        # Dropping over a cap is a different decision from dropping for space: there is no
        # `[INVENTORY_FULL]` to relieve and nothing to gain from losing blocks to the world.
        g = at_home(geared(cobblestone=68))
        g.chest_pos = None
        self.assertNotIn("stow_surplus", candidates(g))
        self.assertNotIn("drop_junk", candidates(g))

    def test_the_whitelists_unnamed_items_at_the_chest(self) -> None:
        for name in ("oak_sapling", "crying_obsidian", "light_gray_wool", "leather",
                     "stone_bricks", "stone_brick_slab"):
            with self.subTest(item=name):
                g = self._at_chest(**{name: 3})
                self.assertIn("stow_surplus", candidates(g))
                a = resolve("stow_surplus", g)
                self.assertIn({"item_name": name, "count": 3}, a.args["items"])

    def test_the_whitelist_never_names_what_the_run_still_needs(self) -> None:
        # Every one of these has cost the run a step when it went missing: the obsidian for the
        # frame, the bow kit for the Nether, the food for the crossing, the bed, the fuel.
        g = self._at_chest(obsidian=10, string=2, feather=12, flint=1, stick=6, iron_ingot=2,
                           cooked_mutton=30, oak_log=9, gravel=14, diamond=2, white_wool=6,
                           white_bed=1, crafting_table=1, bucket=1, flint_and_steel=1)
        self.assertNotIn("stow_surplus", candidates(g))

    def test_the_pickaxe_is_kept_even_before_the_inventory_is_parsed(self) -> None:
        # `tool_meta` is empty until the inventory is parsed, and a whitelist that filed the
        # diamond pickaxe as surplus in that state would bank the run's best pickaxe. `tools`
        # carries class and tier at all times, so `diamond` + `pickaxe` reconstructs the name.
        g = self._at_chest(oak_sapling=2, diamond_pickaxe=1)
        g.tool_meta = {}
        g.tools = {"pickaxe": "diamond", "sword": "diamond"}
        a = resolve("stow_surplus", g)
        self.assertNotIn("diamond_pickaxe", [i["item_name"] for i in a.args["items"]])

    def test_the_pickaxe_in_hand_is_never_surplus(self) -> None:
        # Gear is not written out by hand: whatever the bot has on or in hand is kept whatever it
        # is called. This is what stops a whitelist from shedding the pickaxe it is mining with.
        g = self._at_chest(oak_sapling=2)
        g.tool_meta = {"pickaxe": {"item": "stone_pickaxe", "tier": "stone"}}
        a = resolve("stow_surplus", g)
        self.assertNotIn("stone_pickaxe", [i["item_name"] for i in a.args["items"]])

    def test_a_full_pack_still_falls_back_to_dropping_without_a_chest(self) -> None:
        # The old relief path is untouched: away from the base there is no undo, so the
        # whitelist does not apply and only the blacklist may shed.
        g = at_home(geared(tuff=64))
        g.chest_pos = None
        g.slots_used = INVENTORY_NEAR_FULL
        self.assertIn("drop_junk", candidates(g))
        a = resolve("drop_junk", g)
        self.assertNotIn("oak_sapling", a.args.get("items", [])[0]["item_name"] if a.args.get("items") else [])


class BankedFoodEndsTheHuntTest(unittest.TestCase):
    """A chest full of food means no hunt, and the pack is refilled from that chest.

    The bot stood at its own base with a chest full of cooked mutton and hunted anyway, because
    `hunt_food` counted `_edible_count(inventory)` - the pack, and nothing else. Food banked at
    home was invisible to the only gate that decides whether a hunt is worth walking for.

    Carried and banked stay two numbers, deliberately. The chest is at the base and worth nothing
    in the Nether, so what stops a hunt is the *total* while what has to come along is the *pack*.
    Folding them into one would deadlock the run in the other direction: stop hunting, and then
    nothing could put food in the pack for the crossing.

    The server records a full `contents_json` for every container it has seen, and the home query
    was leaving that column unread. Now it is read, filtered to edibles, and logged.
    """

    BASE = (-197, -47, -98)
    CHEST = (-195, -46, -97)

    def _hungry_at_base(self, carried: int = 0, banked: int = 0) -> GameState:
        g = at_home(geared(cooked_mutton=carried))
        g.food = 12.0                     # hungry: the hunt is not competing with eating
        g.phase = "day"
        g.mobs = [("cow", "cow", 8.0, 8.0)]
        g.chest_pos = self.CHEST
        g.banked = {"cooked_mutton": banked} if banked else {}
        return g

    def test_a_full_chest_ends_the_hunt(self) -> None:
        g = self._hungry_at_base(carried=0, banked=32)
        self.assertNotIn("hunt_food", candidates(g))

    def test_an_empty_chest_still_offers_the_hunt(self) -> None:
        g = self._hungry_at_base(carried=0, banked=0)
        self.assertIn("hunt_food", candidates(g))

    def test_the_chest_is_where_the_pack_is_filled_from(self) -> None:
        # The other half, and the reason the two numbers are kept apart: a banked reserve that
        # stops the hunt would starve the crossing if nothing could carry it into the Nether.
        g = self._hungry_at_base(carried=0, banked=32)
        self.assertIn("take_stored_food", candidates(g))
        a = resolve("take_stored_food", g)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "withdraw")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.CHEST)

    def test_the_withdraw_names_what_the_chest_actually_holds(self) -> None:
        # A `withdraw` naming an item the chest does not hold takes nothing out, and the refusal
        # then looks exactly like a working chest.
        g = self._hungry_at_base(carried=0, banked=9)
        g.banked = {"cooked_mutton": 9}
        a = resolve("take_stored_food", g)
        self.assertEqual(a.args["items"], [{"item_name": "cooked_mutton", "count": 9}])

    def test_a_full_pack_is_not_asked_to_carry_more(self) -> None:
        g = self._hungry_at_base(carried=64, banked=32)
        self.assertNotIn("take_stored_food", candidates(g))

    def test_the_chest_in_the_nether_is_not_a_source(self) -> None:
        # `gs.chest_pos` is the Overworld chest; a withdraw aimed at it from the Nether is a
        # navigation failure wearing a policy's clothes.
        g = self._hungry_at_base(carried=0, banked=32)
        g.dimension = "minecraft:the_nether"
        self.assertNotIn("take_stored_food", candidates(g))

    def test_a_chest_of_rotten_flesh_does_not_end_the_hunt(self) -> None:
        # The live chest: 96 rotten flesh and nothing else. `EDIBLE` has to stay that broad,
        # because the reflex will eat this rather than starve - but a *reserve* is a different
        # question, and one list cannot answer both. Counting it withheld the hunt that fetches
        # the right sort, which is worse than the hunting it replaced.
        g = self._hungry_at_base(carried=0)
        g.banked = {"rotten_flesh": 96}
        self.assertIn("hunt_food", candidates(g))

    def test_a_bank_of_only_junk_is_an_empty_larder(self) -> None:
        # The fallback this test once pinned became a shuttle in run78: `stow_surplus` deposits
        # junk (rotten flesh is JUNK) into the chest, and `take_stored_food` withdrew that very
        # stack back on the next tick - ~400 iterations of pack and chest shedding 21 rotten
        # flesh onto each other, with the Nether trip waiting. Provisions the chest does not
        # hold are not fetchable; the hunt stays offered instead (see the test above).
        g = self._hungry_at_base(carried=0)
        g.banked = {"rotten_flesh": 96}
        self.assertNotIn("take_stored_food", candidates(g))
        self.assertIsNone(resolve("take_stored_food", g))
        from mlaya.actions import _provisioned
        self.assertEqual(_provisioned(g), 0)

    def test_provisions_are_preferred_over_the_emergency_pile(self) -> None:
        g = self._hungry_at_base(carried=0)
        g.banked = {"rotten_flesh": 96, "cooked_mutton": 8}
        a = resolve("take_stored_food", g)
        self.assertEqual(a.args["items"], [{"item_name": "cooked_mutton", "count": 8}])

    def test_provisions_are_a_strict_subset_of_what_is_edible(self) -> None:
        # Three named exclusions, each for a reason in the constant's comment: rotten flesh (4
        # hunger, mostly poison), dried kelp (1 hunger per stack), mushroom stew (does not stack,
        # so 64 of them are 64 slots).
        from mlaya.actions import EDIBLE, PROVISIONS
        self.assertTrue(PROVISIONS < EDIBLE)
        self.assertEqual(set(EDIBLE) - set(PROVISIONS),
                         {"rotten_flesh", "dried_kelp", "mushroom_stew"})

    def test_the_reserve_the_crossing_needs_is_still_counted_in_the_pack(self) -> None:
        # 64 banked and an empty pack must NOT count as a stocked expedition.
        from mlaya.actions import _food_reserve_met
        g = self._hungry_at_base(carried=0, banked=64)
        g.nether_idle = True
        self.assertFalse(_food_reserve_met(g))
        g.inventory["cooked_mutton"] = 64
        self.assertTrue(_food_reserve_met(g))


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
        # Three diamonds are what makes it askable; below that it asks for the digging instead.
        g = self._stocked(diamond=3, stick=4)
        self.assertIn("craft_pickaxe_spare", candidates(g))
        self.assertNotIn("enter_portal", candidates(g))

    def test_two_pickaxes_open_the_gate(self) -> None:
        g = self._stocked(diamond_pickaxe=2)
        self.assertIn("enter_portal", candidates(g))
        self.assertNotIn("craft_pickaxe_spare", candidates(g))

    def test_the_spare_is_not_offered_before_the_diamonds_are_in_the_pack(self) -> None:
        # Corrected, and the reason is a live cost rather than a taste. An offer the resolver
        # cannot satisfy is what escalates the tick to System 2: run57 spent four model calls a
        # second on this label with one diamond in the pack, each handed to a model that then chose
        # `place_bed` or `get_wood`. The offer has to describe something that can happen, and the
        # diamonds still in the ground are not it.
        g = self._stocked(diamond=0)
        self.assertNotIn("craft_pickaxe_spare", candidates(g))
        self.assertIsNone(resolve("craft_pickaxe_spare", g))

    def test_the_missing_diamonds_are_asked_for_as_their_own_step(self) -> None:
        # The other half, and it has to be explicit: `_next_diamond_item` returns None once the
        # armour and the first pickaxe are done, so the tech tree never asks for diamonds again.
        # Without this the expedition would have had no step at all for what it needs.
        g = self._stocked(diamond=0)
        g.resources["diamond_ore"] = [BlockHit("diamond_ore", "diamond_ore", -193, -40, -95, 3.0)]
        self.assertIn("mine_diamond", candidates(g))

    def test_no_diamond_step_without_a_mineable_deposit(self) -> None:
        # Nothing loaded and nothing reachable: then the spare is simply not asked for, and the run
        # goes on with what it can do rather than escalating the same label every tick.
        g = self._stocked(diamond=0)
        g.resources.pop("diamond_ore", None)
        g.resource_found.pop("diamond_ore", None)
        self.assertNotIn("mine_diamond", candidates(g))

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
        # The plane read names all ten obsidian cells: the light step needs the frame proven
        # finished (run80), only the pickaxe spare is decided by these tests.
        g = self._stocked(**inv)
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -46, -98, 2.0)],
                       "white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -100, 2.5)]}
        g.frame_names = {cell: "obsidian" for cell in frame_obsidian_cells(g.base)}
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


class PaidGoalEndsTheDiamondDigTest(unittest.TestCase):
    """A diamond goal the pack can already pay for ends the dig - offer and resolve alike.

    run77 ended on this exact hole: the spare-pickaxe branch offered `mine_diamond` from the
    *want* question alone and never asked whether the pack could already pay, so from the
    first crafted pickaxe on every single tick offered the dig - 140-odd diamonds mined one
    per tick while the armour and the second pickaxe, both long since paid for, sat uncrafted.
    The dig now has to stop exactly where the craft takes over; the craft itself is the step
    that pays for the goal.
    """

    BASE = (-197, -47, -98)
    GATE = (-195, -46, -98)

    def _stocked(self, **inv) -> GameState:
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64, golden_helmet=1,
                     cooked_mutton=64, white_bed=1, diamond_sword=1, diamond_pickaxe=1)
        stock.update(inv)
        g = at_home(geared(**stock), base=self.BASE)
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                   *self.GATE, 2.0)],
                       "obsidian": [BlockHit("obsidian", "obsidian", -195, -46, -98, 2.0)],
                       "white_bed": [BlockHit("white_bed", "white_bed", -197, -46, -100, 2.5)],
                       "diamond_ore": [BlockHit("diamond_ore", "diamond_ore",
                                                -193, -40, -95, 3.0)]}
        g.dimension = "minecraft:overworld"
        return g

    def test_a_paid_spare_stops_the_dig_offer_and_the_binding(self) -> None:
        g = self._stocked(diamond=3, stick=4)
        self.assertIn("craft_pickaxe_spare", candidates(g))
        self.assertNotIn("mine_diamond", candidates(g))
        self.assertIsNone(resolve("mine_diamond", g))

    def test_the_dig_is_still_asked_while_the_spare_is_short(self) -> None:
        g = self._stocked(diamond=1, stick=2)
        self.assertIn("mine_diamond", candidates(g))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 2)


class PaidArmourEndsTheDiamondDigTest(unittest.TestCase):
    """The tech-tree half of the same rule: a payable armour piece outranks the dig.

    Same run, same standstill one layer earlier: while boots and helmet were craftable from
    the pile, `mine_diamond` was still offered from the spare branch and Laya picked it every
    single tick - the offer had to stop competing with the craft it should have handed over to.
    """

    def _kit(self, **inv) -> GameState:
        base = {"crafting_table": 1, "diamond_pickaxe": 1, "diamond_sword": 1,
                "diamond_chestplate": 1, "diamond_leggings": 1, "diamond": 20}
        base.update(inv)
        return combat_ready(gs(tools={"pickaxe": "diamond"}, inventory=base,
                               resources=_ore("diamond_ore")))

    def test_paid_boots_stop_the_dig_offer(self) -> None:
        g = self._kit()
        self.assertIn("craft_diamond_boots", candidates(g))
        self.assertNotIn("mine_diamond", candidates(g))
        self.assertIsNone(resolve("mine_diamond", g))

    def test_short_boots_are_still_dug_for(self) -> None:
        g = self._kit(diamond=1)
        self.assertIn("mine_diamond", candidates(g))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 3)

    def test_no_dig_when_the_workstation_is_what_is_missing(self) -> None:
        # Armour really is table-bound: diamonds will not dig one up, so the dig must not
        # stand in for a step that goes looking for the workstation instead.
        g = self._kit(crafting_table=0)
        self.assertNotIn("mine_diamond", candidates(g))
        self.assertIsNone(resolve("mine_diamond", g))


class HoardSurplusKeepsGoalDiamondsTest(unittest.TestCase):
    """run77's `drop_junk` shed 123 diamonds onto the ground while goals still wanted them.

    The hoard cap counts what one wears, not what the next piece or the trip's second
    pickaxe still wants - a goal outlives a stack, so diamonds with a purpose are never
    surplus.
    """

    def test_diamonds_stay_while_a_goal_waits(self) -> None:
        g = combat_ready(gs(tools={"pickaxe": "diamond"},
                            inventory={"crafting_table": 1, "diamond": 190,
                                       "diamond_pickaxe": 1, "diamond_sword": 1}))
        self.assertTrue(_diamond_reason(g))
        self.assertNotIn("diamond", {s["item_name"] for s in _hoard_surplus(g, g.inventory)})

    def test_diamonds_are_surplus_once_every_reason_is_gone(self) -> None:
        g = geared(diamond=190, diamond_pickaxe=2, diamond_sword=1)
        self.assertFalse(_diamond_reason(g))
        self.assertIn("diamond", {s["item_name"] for s in _hoard_surplus(g, g.inventory)})


class StepAwayDropsWholeCellTest(unittest.TestCase):
    """`navigate` takes whole cells: run77 stood 120 ticks in one validation error.

    `step_away_drops` walked off a drop with float coordinates - `[expected int, received
    number at x]` owned the tick forever, exactly the freeze `leave_portal` had already
    learned to floor around.
    """

    def test_the_target_is_a_whole_cell(self) -> None:
        g = gs(pos=(10.3, 64.7, 20.9), drop_pos=(10.0, 64.0, 17.0),
               drop_open_pending=True)
        a = resolve("step_away_drops", g)
        self.assertEqual(a.tool, "navigate")
        self.assertTrue(all(isinstance(a.args[k], int) for k in ("x", "y", "z")), a.args)

    def test_a_cell_the_ground_read_proved_is_walked_to(self) -> None:
        # run86: the blind "three blocks away" target had no footing nine times running.
        g = gs(pos=(-89.5, 70.0, -4.5), drop_pos=(-89.5, 70.0, -4.5),
               drop_open_pending=True, drop_away=(-85, 70, -3))
        a = resolve("step_away_drops", g)
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-85, 70, -3))

    def test_a_refused_walk_marks_its_cell_for_the_next_pick(self) -> None:
        from mlaya.config import Settings
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent._drop_open_pending = True
        agent._drop_pos = (-89.5, 70.0, -4.5)
        agent._step_away_fails = 0
        agent._drop_refused = set()
        agent._note_step_away(Action("step_away_drops", "navigate",
                                     {"x": -85, "y": 70, "z": -3}, ""), False)
        self.assertEqual(agent._drop_refused, {(-85, 70, -3)})
        self.assertTrue(agent._drop_open_pending)


class DeferQuietTest(unittest.TestCase):
    """A defer counts as quiet only when it took no hit and no survival reflex acted."""

    def _agent(self):
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._defer_quiet = 0
        return agent

    def test_quiet_defers_add_up(self) -> None:
        agent = self._agent()
        for _ in range(3):
            agent._note_defer(gs(health=20.0), gs(health=20.0, events=["hunger: ate; cooked porkchop"]))
        self.assertEqual(agent._defer_quiet, 3)

    def test_a_hit_or_a_fight_starts_over(self) -> None:
        agent = self._agent()
        agent._defer_quiet = 4
        agent._note_defer(gs(health=20.0), gs(health=16.4))
        self.assertEqual(agent._defer_quiet, 0)
        agent._defer_quiet = 4
        agent._note_defer(gs(health=20.0),
                          gs(health=20.0, events=["hostile: target died; health 20 → 20/20"]))
        self.assertEqual(agent._defer_quiet, 0)

    def test_a_hit_while_working_hands_contact_back(self) -> None:
        agent = self._agent()
        agent._defer_quiet = 9
        agent._note_hurt(gs(health=20.0), gs(health=20.0))
        self.assertEqual(agent._defer_quiet, 9)
        agent._note_hurt(gs(health=20.0), gs(health=17.0))
        self.assertEqual(agent._defer_quiet, 0)


class RefreshAfterTest(unittest.TestCase):
    """Facts learned after `_verify`'s observation reach the state the next tick decides on."""

    def _agent(self):
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.surface = None
        agent._frame_complete = False
        agent._powder_stowed = False
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._powder_bank = 0
        agent._frame_eyes = 0
        agent._stuck_off = {}
        agent._defer_quiet = 0
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        return agent

    def test_the_probe_answer_climbs_on_the_very_next_tick(self) -> None:
        # run87 it58/59: the same height-less probe twice before the climb at it60.
        agent = self._agent()
        probe = Action("surface_up", "navigate", {"x": -58, "z": 45}, "")
        after = geared()
        after.pos = (-57.4, -13.0, 45.5)
        agent._note_surface(probe, False,
                            "[NAVIGATION_STOPPED] Pathfinder stopped: [NAVIGATION_HEIGHT_REQUIRED] "
                            "Column -58, 45 has highest block grass_block at y=84; observed "
                            "standable feet heights: 85, -13, -39. Supply y to choose a floor..",
                            after.pos)
        agent._refresh_after(after)
        self.assertEqual(after.surface, (-58, 85, 45))
        climb = resolve("surface_up", after)
        self.assertEqual((climb.args["x"], climb.args["y"], climb.args["z"]), (-58, 85, 45))

    def test_the_latches_and_the_withholding_are_handed_over(self) -> None:
        agent = self._agent()
        agent._frame_complete = True
        agent._powder_stowed = True
        agent._stuck_off = {"sleep": 15}
        agent._defer_quiet = 3
        after = gs()
        agent._refresh_after(after)
        self.assertTrue(after.frame_complete)
        self.assertTrue(after.powder_stowed)
        self.assertEqual(after.stuck_labels, frozenset({"sleep"}))
        self.assertEqual(after.defer_quiet, 3)

    def test_the_pocket_reaches_the_state_of_the_very_tick(self) -> None:
        # `_note_boxed` runs after the observation now (it needs the post-action cell), so the
        # tunnel it asks for has to be handed to that state, not to the one after it.
        from mlaya.graph import BOXED_REPEATS
        agent = self._agent()
        agent._boxed = ((182, 70, 181), BOXED_REPEATS)
        agent._dig_out_tries = 2
        after = gs(pos=(182.5, 70.0, 181.5))
        agent._refresh_after(after)
        self.assertTrue(after.boxed_in)
        self.assertEqual(after.dig_out_try, 2)
        self.assertEqual(candidates(after), ["dig_out"])


class WaitHoldTest(unittest.IsolatedAsyncioTestCase):
    """A successful `wait` holds `wait_hold_s` instead of burning an iteration per second."""

    def _agent(self, hold: float = 30.0):
        from mlaya.config import Settings
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent.settings.loop.wait_hold_s = hold
        return agent

    async def test_a_successful_wait_holds(self) -> None:
        from unittest import mock
        with mock.patch("mlaya.graph.asyncio.sleep", mock.AsyncMock()) as sleep:
            await self._agent()._hold_wait(Action("wait", "navigate", {}, ""), True)
        sleep.assert_awaited_once_with(30.0)

    async def test_other_steps_and_refused_waits_do_not_hold(self) -> None:
        from unittest import mock
        with mock.patch("mlaya.graph.asyncio.sleep", mock.AsyncMock()) as sleep:
            agent = self._agent()
            await agent._hold_wait(Action("wait", "navigate", {}, ""), False)
            await agent._hold_wait(Action("mine_coal", "collect_block", {}, ""), True)
            await self._agent(0)._hold_wait(Action("wait", "navigate", {}, ""), True)
        sleep.assert_not_awaited()


class StepAwayGroundReadTest(unittest.IsolatedAsyncioTestCase):
    """The ground around an open drop is read once per drop, and only while the walk is owed."""

    async def test_one_read_per_drop(self) -> None:
        from unittest import mock
        from mlaya.graph import Agent
        agent = Agent.__new__(Agent)
        agent.bed_pos = None
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.mcp = object()
        agent._drop_open_pending = True
        agent._drop_pos = (-89.5, 70.0, -4.5)
        agent._drop_grid, agent._drop_grid_at, agent._drop_refused = {}, None, {(1, 2, 3)}
        grid = {(-85, 70, -3): "air"}
        with mock.patch("mlaya.perception.scan_ground",
                        mock.AsyncMock(return_value={})) as scan, \
                mock.patch("mlaya.perception.parse_box", return_value=grid):
            await agent._scan_drop_ground(None)
            await agent._scan_drop_ground(None)
            self.assertEqual(scan.await_count, 1)
            self.assertEqual(agent._drop_grid, grid)
            self.assertEqual(agent._drop_refused, set())
            agent._drop_pos = (0.5, 64.0, 0.5)  # the next drop
            await agent._scan_drop_ground(None)
            self.assertEqual(scan.await_count, 2)
            agent._drop_open_pending = False
            await agent._scan_drop_ground(None)
            self.assertEqual(scan.await_count, 2)
            self.assertEqual(agent._drop_grid, {})


class DefencelessAtNightTest(unittest.TestCase):
    """A naked bot at night walks neither back to its gear nor home.

    run90 died nine times in the Overworld after the Nether took its kit, all at night with no
    tool and no armour, most of them on `recover_drops` or `go_home` walks back past the skeleton
    that had just shot it.
    """

    def _death(self, **kw) -> GameState:
        return gs(pos=(182.5, 72.0, 179.5), death_pos=(190.0, 72.0, 166.0),
                  last_death_age_s=20.0, dimension="overworld", death_dim="overworld", **kw)

    def test_no_walk_back_for_the_gear_at_night(self) -> None:
        self.assertIsNone(resolve("recover_drops", self._death(phase="night")))
        self.assertNotIn("recover_drops", candidates(self._death(phase="night")))

    def test_by_day_the_walk_is_offered_again(self) -> None:
        self.assertIsNotNone(resolve("recover_drops", self._death(phase="day")))

    def test_a_sword_or_armour_lifts_it(self) -> None:
        self.assertIsNotNone(resolve("recover_drops",
                                     self._death(phase="night", tools={"sword": "stone"})))
        worn = self._death(phase="night")
        worn.armor = {"chestplate": {"item": "leather_chestplate"}}
        self.assertIsNotNone(resolve("recover_drops", worn))

    def test_wounded_and_naked_at_night_defers_instead_of_walking_home(self) -> None:
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=9.8, food=10,
               base=(231, 71, 157), dimension="overworld")
        self.assertEqual(candidates(g), [])
        g.phase = "day"
        self.assertEqual(candidates(g), ["go_home"])

    def test_the_ordinary_offer_takes_no_long_walk_either(self) -> None:
        # run95 it274: 20 health, full hunger, no tool and no armour, and the night's `explore`
        # walked into a spider and a skeleton at 10-16 m - dead three ticks later. Standing still
        # is the night's work; the survival policy guards the body meanwhile.
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=20, food=20,
               dimension="overworld")
        self.assertEqual(candidates(g), ["wait"])
        g.phase = "day"
        self.assertEqual(candidates(g), ["get_wood"])

    def test_a_weapon_keeps_the_walk_on_offer(self) -> None:
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=20, food=20,
               dimension="overworld", tools={"sword": "stone"})
        self.assertIn("get_wood", candidates(g))

    def test_the_night_hold_is_where_the_bot_stands(self) -> None:
        # Not the base: a naked bot 500 m out walking home at night is run90's death again.
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=20, food=20,
               dimension="overworld", base=(231, 71, 157))
        # The base is set, but `go_home` is gated on `_gear_ready`: a bot that lost its kit never
        # sees it, so the offer is the in-place hold and not a walk home.
        self.assertEqual(candidates(g), ["wait"])
        a = resolve("wait", g)
        self.assertIsNotNone(a)
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (182, 72, 179))
        g.phase = "day"
        self.assertIsNone(resolve("wait", g))

    def test_a_naked_bot_at_night_chops_if_trees_are_in_reach(self) -> None:
        # Wounded and naked, but trees loaded: the bot may chop at night instead of deferring.
        # run112 it0–36: 44 ticks of defer while health 9,2 with no edible and no trees, then
        # the sun rose and healing started. With trees loaded the bot could have worked.
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=9.8, food=10,
               dimension="overworld", base=(231, 71, 157))
        self.assertEqual(candidates(g), [])
        g.resources["logs"] = [BlockHit("logs", "logs", 183, 72, 179, 1.2)]
        self.assertIn("get_wood", candidates(g))
        a = resolve("get_wood", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args["block_name"], "logs")

    def test_a_naked_bot_with_no_trees_defers(self) -> None:
        # Wounded, naked, no edible, and no logs loaded: the bot defers and waits for sunrise.
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=9.8, food=10,
               dimension="overworld", base=(231, 71, 157))
        self.assertEqual(candidates(g), [])

    def test_a_naked_bot_at_night_crafts_a_pickaxe_if_materials_exist(self) -> None:
        # Wounded, naked, but planks in the pack: craft a pickaxe instead of waiting.
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=9.8, food=10,
               dimension="overworld", base=(231, 71, 157), inventory={"birch_planks": 5})
        self.assertIn("craft_pickaxe", candidates(g))
        a = resolve("craft_pickaxe", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"][0]["item_name"], "wooden_pickaxe")

    def test_a_naked_bot_at_night_crafts_a_table_if_planks_exist(self) -> None:
        # Wounded, naked, planks in pack, no crafting table: craft one instead of waiting.
        g = gs(pos=(182.5, 72.0, 179.5), phase="night", health=9.8, food=10,
               dimension="overworld", base=(231, 71, 157), inventory={"birch_planks": 4})
        self.assertIn("craft_table", candidates(g))
        a = resolve("craft_table", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"][0]["item_name"], "crafting_table")


class DigOutTest(unittest.TestCase):
    """A bot the pathfinder cannot move gets a tunnel, not another walk.

    run90 stood 330 ticks at (182,70,181) and (182,72,180), every `explore`, `get_wood` and
    `recover_drops` answered `no path found` after some 70 nodes, until the iteration limit.
    """

    def test_shut_in_owns_the_tick(self) -> None:
        g = gs(pos=(182.5, 70.0, 181.5), boxed_in=True, inventory={"crafting_table": 1})
        self.assertEqual(candidates(g), ["dig_out"])
        self.assertNotIn("dig_out", candidates(gs(pos=(182.5, 70.0, 181.5))))
        self.assertIsNone(resolve("dig_out", gs(pos=(182.5, 70.0, 181.5))))

    def test_the_tunnel_runs_toward_the_base_at_feet_and_head(self) -> None:
        g = gs(pos=(182.5, 70.0, 181.5), boxed_in=True, base=(231, 71, 157))
        a = resolve("dig_out", g)
        self.assertEqual(a.tool, "build_structure")
        cells = {(c["x"], c["y"], c["z"]) for c in a.args["blocks"]}
        self.assertEqual(cells, {(182 + k, 70 + h, 181) for k in (1, 2, 3) for h in (0, 1)})
        self.assertTrue(all(c["block_name"] == "air" for c in a.args["blocks"]))

    def test_a_body_that_is_not_standing_does_not_dig(self) -> None:
        # `build_structure` air cells around a body that is not standing hung the runtime event
        # loop in the water pocket at (-0.5,62,281) (live run115 it527, run116 it0). The step that
        # gets the feet back owns that tick, not a tunnel.
        for water in (True, False):
            g = gs(pos=(-0.5, 62.4, 281.5), boxed_in=True, in_water=water, on_ground=False)
            self.assertIsNone(resolve("dig_out", g))
        # Standing and shut in, the tunnel still runs.
        self.assertIsNotNone(resolve("dig_out", gs(pos=(182.5, 70.0, 181.5), boxed_in=True,
                                                  on_ground=True)))

    def test_each_further_try_turns_to_another_side(self) -> None:
        sides = set()
        for attempt in range(4):
            g = gs(pos=(0.5, 64.0, 0.5), boxed_in=True, dig_out_try=attempt)
            first = resolve("dig_out", g).args["blocks"][0]
            sides.add((first["x"], first["z"]))
        self.assertEqual(sides, {(1, 0), (0, 1), (-1, 0), (0, -1)})


class FlattenHomeBaseTest(unittest.TestCase):
    """`flatten_home_base`: the daytime maintenance step that digs away the stray terrain and the
    reflex-placed defence scaffold piled above ground level around a lit home gate.

    The removable cells come ready from the loop's footprint scan (`gs.raised_cells`,
    `Agent._scan_base_terrain`, pinned in `test_graph.py`), so the action layer only has to offer
    the step while there is something to level and bind one `build_structure` of air cells, capped
    at `FLATTEN_BATCH` because the server runs a route search per cell.
    """

    BASE = (-50, 70, 20)

    def _lumpy(self, raised=None) -> GameState:
        """A geared bot home by day at a lit gate, with blocks piled above ground level."""
        bx, by, bz = self.BASE
        g = at_home(geared(), self.BASE)
        g.phase = "day"
        g.resources = {"nether_portal": [
            BlockHit("nether_portal", "nether_portal", bx + 2, by, bz, 2.0)]}
        g.raised_cells = raised if raised is not None else [
            (bx + 1, by + 1, bz), (bx + 2, by + 2, bz + 1), (bx - 3, by + 1, bz - 2)]
        return g

    def test_offered_at_a_lumpy_base_by_day(self) -> None:
        g = self._lumpy()
        self.assertTrue(_base_leveling_wanted(g))
        self.assertIn("flatten_home_base", candidates(g))

    def test_a_clean_base_drops_the_step(self) -> None:
        # The scan comes up empty the moment the ground is flat, and the step falls out of the offer.
        g = self._lumpy(raised=[])
        self.assertFalse(_base_leveling_wanted(g))
        self.assertNotIn("flatten_home_base", candidates(g))

    def test_night_holds_it_back(self) -> None:
        # Daytime only for now, to watch how it grips before it runs at night as well.
        g = self._lumpy()
        g.phase = "night"
        self.assertFalse(_base_leveling_wanted(g))
        self.assertNotIn("flatten_home_base", candidates(g))

    def test_off_base_is_not_offered(self) -> None:
        # Away from home the footprint is not the bot's to level.
        g = self._lumpy()
        g.pos = (g.base[0] + 40.0, g.base[1] + 1.0, g.base[2] + 40.0)
        self.assertFalse(_base_leveling_wanted(g))
        self.assertNotIn("flatten_home_base", candidates(g))

    def test_an_unlit_gate_is_not_offered(self) -> None:
        # Gated on the lit gate so the base geometry is final: the frame build and its clearance
        # steps own the tick before that, and leveling must not race them.
        g = self._lumpy()
        g.resources = {}
        self.assertFalse(_base_leveling_wanted(g))
        self.assertNotIn("flatten_home_base", candidates(g))

    def test_the_nether_has_no_base_to_level(self) -> None:
        g = self._lumpy()
        g.dimension = "the_nether"
        self.assertFalse(_base_leveling_wanted(g))

    def test_the_resolve_digs_the_raised_cells_to_air(self) -> None:
        g = self._lumpy()
        a = resolve("flatten_home_base", g)
        self.assertEqual(a.tool, "build_structure")
        self.assertTrue(all(c["block_name"] == "air" for c in a.args["blocks"]))
        self.assertEqual([(c["x"], c["y"], c["z"]) for c in a.args["blocks"]], g.raised_cells)

    def test_the_batch_is_capped(self) -> None:
        # One route search per cell, so a long list is dug a batch at a time; the scan re-reads each
        # tick and the rest falls to the next one (`clear_portal`'s reason).
        bx, by, bz = self.BASE
        raised = [(bx, by + 1, bz + i) for i in range(FLATTEN_BATCH + 5)]
        g = self._lumpy(raised=raised)
        a = resolve("flatten_home_base", g)
        self.assertEqual(len(a.args["blocks"]), FLATTEN_BATCH)
        self.assertEqual([(c["x"], c["y"], c["z"]) for c in a.args["blocks"]],
                         raised[:FLATTEN_BATCH])

    def test_the_offer_and_the_resolve_agree(self) -> None:
        # When the step is not wanted, `resolve` answers None too, so it is never offered and then
        # refused on the same tick (`_base_leveling_wanted` is the single gate for both).
        g = self._lumpy(raised=[])
        self.assertIsNone(resolve("flatten_home_base", g))


class LabelRegistryTest(unittest.TestCase):
    """Every label the planner can offer must carry a brain criterion.

    `brain.intent_question` builds the choice question as `{label: CRITERIA[label]}`, so a label
    the planner appends without one crashes the very tick it first appears on. Live, run115 it6:
    `KeyError: 'approach_lava'` in `decide` ended the run after five iterations.
    """

    def test_every_offered_label_has_a_criterion(self) -> None:
        from mlaya.actions import planner, stronghold
        from mlaya.actions.constants import CRITERIA

        offered: set[str] = set()
        for module in (planner, stronghold):
            src = inspect.getsource(module)
            offered |= set(re.findall(r'out\.append\("([a-z_]+)"\)', src))
            offered |= set(re.findall(r'return \["([a-z_]+)"\]', src))
            offered |= set(re.findall(r'\+ \["([a-z_]+)"\]', src))
        self.assertFalse(sorted(offered - set(CRITERIA)))


if __name__ == "__main__":
    unittest.main()
