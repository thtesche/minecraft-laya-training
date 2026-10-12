"""Action binding: label -> (tool, args) for the reactions and the food/bed chain.

The M2d reactions (`collect_drops`, `recover_drops`, `replace_tool`) and the bed chain
(`craft_bed`, `sleep`) are event-driven and hard to reach on demand in a live world, so
their precondition guards and argument binding are pinned here.
"""
from __future__ import annotations

import unittest

from mlaya.actions import Action, candidates, resolve, tech_tree_complete
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

    def _complete(self, **overrides) -> GameState:
        base = dict(
            tools={"pickaxe": "diamond", "sword": "diamond"},
            inventory={"diamond_sword": 1},
            armor={slot: {"item": f"diamond_{slot}"}
                   for slot in ("helmet", "chestplate", "leggings", "boots")},
            offhand="shield",
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


if __name__ == "__main__":
    unittest.main()
