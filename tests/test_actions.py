"""Action binding: label -> (tool, args) for the reactions and the food/bed chain.

The M2d reactions (`collect_drops`, `recover_drops`, `replace_tool`) and the bed chain
(`craft_bed`, `sleep`) are event-driven and hard to reach on demand in a live world, so
their precondition guards and argument binding are pinned here.
"""
from __future__ import annotations

import unittest

from mlaya.actions import Action, candidates, inventory_relief, resolve, tech_tree_complete
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
        # Nothing wants iron once the pickaxe is fresh and the shield is already worn (the shield
        # is the only other iron sink; see ShieldTest.test_missing_shield_pulls_one_iron).
        g = gs(tools={"pickaxe": "iron"}, offhand="shield",
               tool_meta=self._meta(200.0, "iron_pickaxe", "iron", 250.0),
               resources=_ore("iron_ore"))
        self.assertNotIn("mine_iron", candidates(g))

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
            self.assertIsNone(resolve("mine_diamond", gs(tools={"pickaxe": tier}, resources=near)))
            self.assertNotIn("mine_diamond",
                             candidates(gs(tools={"pickaxe": tier}, resources=near)))
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

    def test_deep_unscanned_diamond_ore_still_offers_the_dig(self) -> None:
        # The resource scan is radius-limited (RESOURCE_RADIUS), so diamond ore more than 64 m
        # down never appears in `resources` while the bot stands on the surface over it. The raw
        # `found` count from `view_blocks` proves the ore is loaded and `collect_block` tunnels to
        # it, so the dig must stay on offer - otherwise a lost diamond armour piece (the live
        # case: the boots were destroyed and the whole kit sat at 33/36 slots) could never be
        # rebuilt and `_gear_ready` would never close.
        deep = gs(tools={"pickaxe": "diamond"},
                  inventory={"diamond_sword": 1, "diamond_chestplate": 1,
                             "diamond_leggings": 1, "diamond_helmet": 1, "diamond": 1},
                  resource_found={"diamond_ore": 15479})
        self.assertIn("mine_diamond", candidates(deep))
        self.assertEqual(resolve("mine_diamond", deep).args["count"], 3)

    def test_no_dig_offer_without_any_diamond_ore_loaded(self) -> None:
        self.assertNotIn("mine_diamond",
                         candidates(gs(tools={"pickaxe": "diamond"},
                                       resource_found={"iron_ore": 900, "coal_ore": 400})))

    def test_no_diamond_dig_on_a_pickaxe_that_needs_a_backup(self) -> None:
        # Breaking mid-dig would destroy the ore with the swapped-in lower tier, so do not
        # start while the pickaxe is due for a backup (unknown durability must not block).
        def meta(left):
            return {"pickaxe": {"tier": "iron", "item": "iron_pickaxe",
                                "left": left, "max": 250.0}}
        worn = gs(tools={"pickaxe": "iron"}, tool_meta=meta(70.0),
                  resources=_ore("diamond_ore"))
        self.assertNotIn("mine_diamond", candidates(worn))
        fresh = gs(tools={"pickaxe": "iron"}, tool_meta=meta(200.0),
                   resources=_ore("diamond_ore"))
        self.assertIn("mine_diamond", candidates(fresh))

    def test_candidates_no_diamond_sword_before_pickaxe(self) -> None:
        g = gs(tools={"pickaxe": "iron"},
               inventory={"crafting_table": 1, "stick": 2, "diamond": 2})
        self.assertNotIn("craft_diamond_sword", candidates(g))

    def test_candidates_mine_diamond_with_diamond_pickaxe_for_the_sword(self) -> None:
        # Regression: bounding mine_diamond at the diamond tier deadlocked the sword, since
        # the diamond pickaxe could no longer mine the two diamonds the sword still needs.
        g = gs(tools={"pickaxe": "diamond"}, resources=_ore("diamond_ore"))
        self.assertIn("mine_diamond", candidates(g))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 2)

    def test_candidates_mine_diamond_for_armour_after_the_sword(self) -> None:
        # After the pickaxe and the sword the diamond goal is the armour set, chestplate first
        # (eight diamonds), so mining continues; it stops once every piece is built.
        g = gs(tools={"pickaxe": "diamond"}, inventory={"diamond_sword": 1},
               resources=_ore("diamond_ore"))
        self.assertIn("mine_diamond", candidates(g))
        self.assertEqual(resolve("mine_diamond", g).args["count"], 8)
        done = gs(tools={"pickaxe": "diamond"},
                  inventory={"diamond_sword": 1, "diamond_chestplate": 1,
                             "diamond_leggings": 1, "diamond_boots": 1, "diamond_helmet": 1},
                  resources=_ore("diamond_ore"))
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
        g = gs(tools={"pickaxe": "diamond"}, inventory={"diamond_sword": 1, "diamond": 3},
               resources=_ore("diamond_ore"))
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

    def test_missing_shield_pulls_one_iron(self) -> None:
        # Nothing else needs iron once the pickaxe exists, so the shield is the reason to dig:
        # one ingot is enough (live, the loop was offering only `explore` next to an iron vein).
        g = gs(tools={"pickaxe": "diamond"},
               inventory={"crafting_table": 1, "oak_planks": 6}, resources=_ore("iron_ore"))
        self.assertIn("mine_iron", candidates(g))
        self.assertEqual(resolve("mine_iron", g).args["count"], 1)

    def test_shield_done_does_not_quarry_iron(self) -> None:
        g = gs(tools={"pickaxe": "diamond"}, inventory={"crafting_table": 1},
               offhand="shield", resources=_ore("iron_ore"))
        self.assertNotIn("mine_iron", candidates(g))
        self.assertIsNone(resolve("mine_iron", g))


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
            "white_bed": [BlockHit("white_bed", "white_bed", -269, 64, -56, 4.0)],
        }

    def _complete(self, **overrides) -> GameState:
        base = dict(
            tools={"pickaxe": "diamond", "sword": "diamond"},
            inventory={"diamond_sword": 1},
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

    def test_a_missing_bed_is_not_done(self) -> None:
        g = self._complete(resources={"nether_portal": self._gate()["nether_portal"]})
        self.assertFalse(tech_tree_complete(g))


class NetherGateTest(unittest.TestCase):
    """The overworld -> Nether gate: a bucket of water makes the obsidian, gravel supplies the
    flint, the flint and steel lights the frame, and a bed is placed beside the lit portal.

    The chain is gated on the whole diamond kit (`_gear_ready`) and ordered so the obsidian is
    banked before the flint and steel is crafted - that craft is the latch that stops
    `mine_obsidian` from re-mining after `build_portal` has spent the ten blocks.
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

    def test_mine_obsidian_mines_what_the_pour_formed(self) -> None:
        # Obsidian the last pour formed lies at the bot's feet: mine it, even with no water left.
        g = self._geared(obsidian=4)
        g.pos = (-270.0, 64.0, -56.0)
        g.resources = self._blocks("obsidian")
        self.assertIn("mine_obsidian", candidates(g))
        a = resolve("mine_obsidian", g)
        self.assertEqual(a.tool, "collect_block")
        self.assertEqual(a.args, {"block_name": "obsidian", "count": 6})

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
        # The craft is the latch that keeps the obsidian from being mined twice; it may only
        # fire once the ten blocks are banked.
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

    def test_build_portal_stays_put_without_ten_blocks_or_a_frame(self) -> None:
        # Seven obsidian and no frame started yet: the bot must bank the rest first.
        g = self._at_home(self._geared(flint_and_steel=1, obsidian=7, cobblestone=4))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -260, 64, -56, 3.0)]}
        self.assertNotIn("build_portal", candidates(g))
        self.assertIsNone(resolve("build_portal", g))

    def test_light_portal_strikes_the_nearest_frame_block(self) -> None:
        # After the build the ten obsidian have left the pack; an obsidian hit *inside the
        # frame plane* confirms the frame is standing, and its nearest block is struck.
        g = self._at_home(self._geared(flint_and_steel=1))
        g.resources = {"obsidian": [BlockHit("obsidian", "obsidian", -195, -47, -98, 3.0)]}
        self.assertIn("light_portal", candidates(g))
        a = resolve("light_portal", g)
        self.assertEqual(a.tool, "activate_portal")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-195, -47, -98))

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

    def test_place_bed_waits_for_the_lit_portal(self) -> None:
        g = self._at_home(self._geared(white_bed=1))
        g.phase = "night"
        self.assertNotIn("place_bed", candidates(g))
        self.assertIsNone(resolve("place_bed", g))

    def test_place_bed_is_not_repeated(self) -> None:
        g = self._at_home(self._geared(white_bed=1))
        g.phase = "night"
        g.resources = self._blocks("nether_portal", "white_bed")
        self.assertNotIn("place_bed", candidates(g))


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

    def test_establish_base_walks_to_the_surface_ground(self) -> None:
        g = self._geared()
        g.pos = (-197.44, 64.0, -97.64)
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
        g.resources = {}
        self.assertIn("establish_base", candidates(g))
        self.assertIsNone(resolve("establish_base", g))

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

    def test_drop_junk_falls_back_without_a_chest(self) -> None:
        g = self._home(self._geared(tuff=64))
        g.slots_used = 34
        self.assertIn("drop_junk", candidates(g))
        self.assertNotIn("stow_surplus", candidates(g))
        a = resolve("drop_junk", g)
        self.assertEqual(a.tool, "drop_item")
        self.assertEqual(a.args["items"], [{"item_name": "tuff"}])

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


class EscapeWaterTest(unittest.TestCase):
    """The no-footing rescue: while the bot floats in water nothing else works.

    Live failure: the obsidian dig left the bot treading water off the ground and every
    following `collect_block` thrashed in place for minutes, while `navigate` still walked it
    clear. The rescue is forced into `candidates` rather than offered, because without footing
    no other label can make progress.
    """

    def test_no_footing_in_water_owns_the_tick(self) -> None:
        g = gs(in_water=True, on_ground=False)
        self.assertEqual(candidates(g), ["escape_water"])

    def test_a_grounded_bot_in_water_is_not_rescued(self) -> None:
        # Wading with the feet on the bottom (a water source, a pour that did not flood the bot)
        # is ordinary play, not the trap.
        g = gs(in_water=True, on_ground=True)
        self.assertNotIn("escape_water", candidates(g))

    def test_a_bot_in_the_air_is_not_rescued(self) -> None:
        # Mid-jump or mid-fall is not the trap either.
        g = gs(in_water=False, on_ground=False)
        self.assertNotIn("escape_water", candidates(g))

    def test_escape_walks_to_the_base(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
