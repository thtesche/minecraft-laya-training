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


class CollectSmeltedProgressTest(unittest.TestCase):
    """A refusal is not the quiet first half of a drain.

    Draining the base furnace is two calls: a look into it, which changes nothing, and the
    reach-in, which lands the output in the pack. So an empty inventory delta counts as progress
    on a call that succeeded -- but a *refusal* leaves the delta empty too, and reading that as
    progress reset the stuck count on every attempt. Live, run49: `collect_smelted` answered
    `[CONTAINER_NOT_PRESENT] The loaded target block is furnace, not a supported container` 26
    times running and was never dropped from the offer, because it never looked stuck.
    """

    REFUSAL = "[CONTAINER_NOT_PRESENT] The loaded target block is furnace, not a supported container"

    def test_the_quiet_look_into_an_open_furnace_is_progress(self) -> None:
        self.assertTrue(progressed("collect_smelted", diff(gs(), gs()), ok=True))

    def test_the_items_landing_in_the_pack_are_progress(self) -> None:
        d = diff(gs(inventory={"beef": 2}), gs(inventory={"beef": 2, "cooked_beef": 1}))
        self.assertTrue(progressed("collect_smelted", d))

    def test_a_refusal_is_not_progress_however_empty_the_delta(self) -> None:
        d = diff(gs(), gs())
        self.assertEqual(d["inventory_delta"], {})
        self.assertFalse(progressed("collect_smelted", d, ok=False))

    def test_four_refusals_in_a_row_then_get_dropped(self) -> None:
        # The shape that used to be impossible: the same code, counted four times, reaches the
        # threshold. `_track_stuck` is what reads this verdict, so the guard is what this pins.
        from mlaya.graph import Agent, STUCK_REPEATS
        agent = Agent.__new__(Agent)
        agent._nether_landmarks = []
        agent._stuck_reps = {}
        agent._stuck_code = {}
        agent._stuck_off = {}
        agent.log = []
        d = diff(gs(), gs())
        for _ in range(STUCK_REPEATS):
            ok = False
            agent._track_stuck("collect_smelted", ok, self.REFUSAL,
                               progressed("collect_smelted", d, ok))
        self.assertIn("collect_smelted", agent._stuck_off)


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

    def test_place_furnace_is_the_furnace_leaving_the_pack(self) -> None:
        d = diff(gs(inventory={"furnace": 1}), gs(inventory={}))
        self.assertTrue(progressed("place_furnace", d))

    def test_stow_and_drop_are_any_item_going_down(self) -> None:
        d = diff(gs(inventory={"tuff": 64, "granite": 32}), gs(inventory={"tuff": 0, "granite": 32}))
        self.assertTrue(progressed("stow_surplus", d))
        self.assertTrue(progressed("drop_junk", d))
        self.assertFalse(progressed("stow_surplus",
                                    diff(gs(inventory={"tuff": 64}), gs(inventory={"tuff": 64}))))

    def test_the_walk_to_the_roomy_chest_is_progress_too(self) -> None:
        # A network chest across the base is `stow_surplus` walking, `take_stored_food` walking
        # to it: the step toward the chest is progress as much as the deposit at it is -- and
        # until this a *successful* `withdraw` had no rule at all and always read as no progress.
        walked = diff(gs(pos=(0.0, 64.0, 0.0)), gs(pos=(20.0, 64.0, 0.0)))
        self.assertTrue(progressed("stow_surplus", walked))
        self.assertTrue(progressed("take_stored_food", walked))
        self.assertTrue(progressed("take_stored_food",
                                    diff(gs(inventory={}), gs(inventory={"cooked_beef": 12}))))
        self.assertFalse(progressed("stow_surplus", walked, False))
        self.assertFalse(progressed("take_stored_food", walked, False))


class GoldAndNetherPrepProgressTest(unittest.TestCase):
    """Gold and the Nether-prep steps each net an item, but they were absent from the
    inventory set, so `progressed` answered False even when they worked. Live, `smelt_gold`
    finished (coal -1, gold_ingot +3) and read as no progress."""

    def test_mine_gold_is_the_raw_gold_gain(self) -> None:
        d = diff(gs(inventory={}), gs(inventory={"raw_gold": 5}))
        self.assertTrue(progressed("mine_gold", d))

    def test_smelt_gold_is_the_gold_ingot_gain(self) -> None:
        d = diff(gs(inventory={"raw_gold": 3, "coal": 1}),
                 gs(inventory={"gold_ingot": 3, "coal": 0}))
        self.assertTrue(progressed("smelt_gold", d))

    def test_craft_bow_and_arrows_are_their_own_gain(self) -> None:
        self.assertTrue(progressed("craft_bow",
                                   diff(gs(inventory={"stick": 3}),
                                        gs(inventory={"stick": 3, "bow": 1}))))
        self.assertTrue(progressed("craft_arrows",
                                   diff(gs(inventory={"flint": 1, "stick": 1}),
                                        gs(inventory={"arrow": 4}))))

    def test_craft_gold_helmet_is_the_piece_arriving(self) -> None:
        carried = diff(gs(inventory={"gold_ingot": 5}),
                       gs(inventory={"gold_ingot": 5, "golden_helmet": 1}))
        self.assertTrue(progressed("craft_gold_helmet", carried))
        # A server that auto-equips puts the helmet on the body, not in the pack.
        worn = diff(gs(inventory={"gold_ingot": 5}),
                    gs(inventory={"gold_ingot": 5},
                       armor={"helmet": {"tier": "gold", "item": "golden_helmet"}}))
        self.assertTrue(progressed("craft_gold_helmet", worn))

    def test_hunt_feather_is_the_feather_gain(self) -> None:
        self.assertTrue(progressed("hunt_feather",
                                   diff(gs(inventory={}), gs(inventory={"feather": 2}))))
        self.assertFalse(progressed("hunt_feather",
                                    diff(gs(inventory={"feather": 2}),
                                         gs(inventory={"feather": 2}))))


class PearlHuntProgressTest(unittest.TestCase):
    """The stronghold pearl run had no rule at all, so a working hunt read as no progress: the
    kill that drops a pearl, the roam that loads fresh ground to find an enderman, the banked
    powder withdrawn, the eye assembled. A roam that reads as stuck gets dropped from the offer
    and the bot strands; an unverified tick is thrown to the offline teacher's opinion."""

    def test_hunt_enderman_is_the_pearl_the_kill_drops(self) -> None:
        self.assertTrue(progressed("hunt_enderman",
                                   diff(gs(inventory={}), gs(inventory={"ender_pearl": 1}))))
        # A hunt that moved the bot but dropped no pearl made no progress -- the pearl is the proof.
        self.assertFalse(progressed("hunt_enderman",
                                    diff(gs(pos=(0.0, 64.0, 0.0)), gs(pos=(20.0, 64.0, 0.0)))))

    def test_search_enderman_is_the_ground_it_roams(self) -> None:
        # Pure roam: nothing lands until a hunt, so movement to new ground is the proof, the same
        # rule `explore` uses -- a roam that read as no progress looked stuck and got dropped.
        self.assertTrue(progressed("search_enderman",
                                   diff(gs(pos=(0.0, 64.0, 0.0)), gs(pos=(20.0, 64.0, 0.0)))))
        # A pearl that lands while roaming counts too; standing still does not.
        self.assertTrue(progressed("search_enderman",
                                   diff(gs(inventory={}), gs(inventory={"ender_pearl": 1}))))
        self.assertFalse(progressed("search_enderman",
                                    diff(gs(pos=(0.0, 64.0, 0.0)), gs(pos=(1.0, 64.0, 0.0)))))

    def test_take_blaze_powder_is_the_powder_withdrawn(self) -> None:
        self.assertTrue(progressed("take_blaze_powder",
                                   diff(gs(inventory={}), gs(inventory={"blaze_powder": 6}))))

    def test_craft_ender_eye_is_the_eye_assembled(self) -> None:
        d = diff(gs(inventory={"ender_pearl": 1, "blaze_powder": 1}),
                 gs(inventory={"ender_eye": 1}))
        self.assertTrue(progressed("craft_ender_eye", d))


if __name__ == "__main__":
    unittest.main()
