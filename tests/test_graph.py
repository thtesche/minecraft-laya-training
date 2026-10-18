"""Failure recovery: an action blocked by the bot's surroundings is unblocked instead of
being retried in place.

Live-validated in the iron chain: after mining ore the bot sat in a 1x1 shaft and
`craft_furnace` answered `[WORKSTATION_PLACEMENT_FAILED] crafting_table: no_cell` on every
tick, so the run never reached the furnace. The fix is a code-level unlock: when the same
label that just failed this way comes up again, dig a small pocket where the bot stands so
the workstation fits (`make_room`), and only on a second try walk to open ground.
The armour run found the same class of block for `sleep`: `[BED_PLACEMENT_FAILED] ... no ...
flat ground` in that shaft, so the stub list now covers both codes.
"""
from __future__ import annotations

import unittest

from mlaya.actions import Action, candidates, resolve
from mlaya.brain import Decision
from mlaya.config import Settings
from mlaya.graph import (Agent, PORTAL_INTENT_TICKS, STUCK_COOLDOWN, STUCK_REPEATS,
                         blocked_by_surroundings, unblock_action)
from mlaya.perception import BlockHit, GameState


def gs(**kw) -> GameState:
    g = GameState()
    g.inventory = kw.pop("inventory", {})
    g.tools = kw.pop("tools", {})
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def _ground() -> dict:
    return {"grass_block": [
        BlockHit(query="grass_block", name="grass_block", x=-268, y=95, z=-56,
                 distance=13.4, above="air", below="dirt")]}


class BlockedBySurroundingsTest(unittest.TestCase):
    def test_detects_workstation_placement_failure(self) -> None:
        self.assertTrue(blocked_by_surroundings(
            "[WORKSTATION_PLACEMENT_FAILED] crafting_table: no_cell."))

    def test_detects_bed_placement_failure(self) -> None:
        # Live, armour run: in the same 1x1 shaft `sleep` answered
        # "[BED_PLACEMENT_FAILED] No nearby supported 2-block flat ground found to place a bed."
        self.assertTrue(blocked_by_surroundings(
            "[BED_PLACEMENT_FAILED] No nearby supported 2-block flat ground found to place a bed."))

    def test_ignores_other_failures_and_empty(self) -> None:
        self.assertFalse(blocked_by_surroundings("[HUNT_TARGET_NOT_LOADED] sheep"))
        self.assertFalse(blocked_by_surroundings(""))
        self.assertFalse(blocked_by_surroundings(None))

    def test_detects_a_pathfinder_that_cannot_move(self) -> None:
        # Live, run10: boxed in a 1-wide flooded shaft, the body did not move on any route the
        # pathfinder re-issued, so `navigate`, `collect_block` and `build_structure` all answered
        # this. Treating it as a surroundings block lets the unblock ladder fire instead of the
        # forced `escape_water` rescue spinning on it forever.
        self.assertTrue(blocked_by_surroundings(
            "[NAVIGATION_STOPPED] Pathfinder stopped: repeated_movement_failure: "
            "Server confirmation deadline elapsed."))

    def test_detects_gather_with_no_reachable_target(self) -> None:
        # Live, run12: the bot stood among 56 trees and answered `get_wood` with this every tick
        # (liquid isolation refused every block). The unblock ladder digs a pocket and then walks
        # to open ground, which is the change such a refusal needs.
        self.assertTrue(blocked_by_surroundings(
            "[NO_REACHABLE_MATCHING_TARGETS] Every loaded logs block was found but none could be "
            "reached. Last routing stop: liquid isolation needs a permitted building block or an "
            "empty bucket"))


class WaterTicksTest(unittest.TestCase):
    """The loop counts the decisions the bot spends off its feet in water (`Agent._inject`).

    `escape_water` owns the tick only from `WATER_RESCUE_AFTER` on, so the counter has to rise
    while the bot floats and fall the moment it has ground again (live, run15: the goal crossed a
    flooded flat and the rescue, which owned the tick on sight, walked it straight back).
    """

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._water_ticks = 0
        return agent

    def test_counts_the_decisions_off_the_feet(self) -> None:
        agent = self._agent()
        g = gs(in_water=True, on_ground=False)
        for expected in (1, 2, 3):
            agent._inject(g)
            self.assertEqual(g.water_ticks, expected)

    def test_ground_underwater_and_dry_land_both_reset_it(self) -> None:
        agent = self._agent()
        g = gs(in_water=True, on_ground=False)
        agent._inject(g)
        self.assertEqual(g.water_ticks, 1)
        for back in (gs(in_water=True, on_ground=True), gs(in_water=False, on_ground=True),
                     gs(in_water=False, on_ground=False)):
            agent._inject(back)
            self.assertEqual(back.water_ticks, 0)
        agent._inject(g)
        self.assertEqual(g.water_ticks, 1)  # and it starts again from one


class UnblockActionTest(unittest.TestCase):
    def test_first_attempt_digs_a_pocket_in_place(self) -> None:
        a = unblock_action(gs(pos=(-277.5, 87.0, -51.6)))
        self.assertEqual(a.tool, "build_structure")
        cells = a.args["blocks"]
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(c["block_name"] == "air" for c in cells))
        self.assertEqual({(c["x"], c["y"], c["z"]) for c in cells},
                         {(-277, 87, -52), (-279, 87, -52),
                          (-278, 87, -51), (-278, 87, -53)})

    def test_second_attempt_falls_back_to_open_ground(self) -> None:
        a = unblock_action(gs(pos=(-277.5, 87.0, -51.6), resources=_ground()), attempt=1)
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (-268, 95, -56))
        self.assertTrue(a.args["dig"] and a.args["scaffold"])

    def test_none_without_pos_or_ground(self) -> None:
        self.assertIsNone(unblock_action(gs()))                       # no pos: no pocket
        self.assertIsNone(unblock_action(gs(pos=(0.0, 60.0, 0.0)), attempt=1))


class GuardUnblockTest(unittest.IsolatedAsyncioTestCase):
    class _Log:
        def __init__(self) -> None:
            self.records: list = []

        def append(self, rec) -> None:
            self.records.append(rec)

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent.log = self._Log()
        agent._blocked_label = None
        agent._blocked_kind = "surroundings"
        agent._stow_blocked = False
        agent._unblock_attempts = 0
        agent._unblock_done = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        return agent

    async def test_first_block_digs_a_pocket(self) -> None:
        agent = self._agent()
        agent._blocked_label = "craft_furnace"
        state = {"before": gs(pos=(-277.5, 87.0, -51.6)),
                 "decision": Decision("craft_furnace")}
        out = await agent._guard(state)
        self.assertEqual(out["action"].tool, "build_structure")
        self.assertEqual(agent._unblock_attempts, 1)
        # The label is kept: it is only cleared once craft_furnace itself succeeds, otherwise
        # the stage-2 `explore` escape below is never reached.
        self.assertEqual(agent._blocked_label, "craft_furnace")
        self.assertEqual(agent.log.records[-1]["reason"], "unblock_surroundings")
        self.assertEqual(agent.log.records[-1]["via"], "make_room")

    async def test_second_block_falls_back_to_navigate(self) -> None:
        agent = self._agent()
        agent._blocked_label = "craft_furnace"
        agent._unblock_attempts = 1
        state = {"before": gs(pos=(-277.5, 87.0, -51.6), resources=_ground()),
                 "decision": Decision("craft_furnace")}
        out = await agent._guard(state)
        self.assertEqual(out["action"].tool, "navigate")
        self.assertEqual(agent.log.records[-1]["via"], "explore")

    async def test_the_label_gets_its_retry_after_the_pocket(self) -> None:
        # Regression, live from-zero run: `smelt_iron` on grass answered
        # `[WORKSTATION_PLACEMENT_FAILED] furnace: no_cell`; the guard then intercepted every
        # retry with `explore`, so the furnace never got to use the pocket `make_room` had just
        # dug and the label stayed blocked for the rest of the run.
        agent = self._agent()
        agent._blocked_label = "craft_furnace"
        first = await agent._guard({"before": gs(pos=(-277.5, 87.0, -51.6)),
                                    "decision": Decision("craft_furnace")})
        self.assertEqual(first["action"].tool, "build_structure")  # make_room
        self.assertTrue(agent._unblock_done)
        # The very next tick the label itself runs, not another unblock step.
        second = await agent._guard({"before": gs(inventory={"crafting_table": 1, "cobblestone": 8}),
                                     "decision": Decision("craft_furnace")})
        self.assertEqual(second["action"].tool, "craft_item")
        self.assertFalse(agent._unblock_done)

    async def test_a_failed_retry_reaches_the_explore_rung(self) -> None:
        agent = self._agent()
        agent._blocked_label = "craft_furnace"
        agent._unblock_attempts = 1
        agent._unblock_done = True  # the pocket ran; the label is retrying
        agent._note_block("craft_furnace", False, "[WORKSTATION_PLACEMENT_FAILED] furnace: no_cell")
        self.assertEqual(agent._blocked_label, "craft_furnace")
        self.assertEqual(agent._unblock_attempts, 1)
        self.assertFalse(agent._unblock_done)
        out = await agent._guard({"before": gs(pos=(-277.5, 87.0, -51.6), resources=_ground()),
                                  "decision": Decision("craft_furnace")})
        self.assertEqual(out["action"].tool, "navigate")
        self.assertEqual(agent.log.records[-1]["via"], "explore")

    async def test_keeps_label_when_not_blocked(self) -> None:
        agent = self._agent()
        state = {"before": gs(inventory={"crafting_table": 1, "cobblestone": 8}),
                 "decision": Decision("craft_furnace")}
        out = await agent._guard(state)
        self.assertEqual(out["action"].tool, "craft_item")

    async def test_a_deferral_with_nothing_blocked_stays_a_deferral(self) -> None:
        # A deferral carries `action_label = None`; `None == None` used to inject `make_room`
        # (live, from-zero run it5) instead of leaving the tick to the survival policy.
        agent = self._agent()
        out = await agent._guard({"before": gs(pos=(-277.5, 87.0, -51.6), resources=_ground()),
                                  "decision": Decision(None, deferred=True)})
        self.assertEqual(out, {"action": None, "deferred": True, "note": ""})

    def test_a_successful_unblock_keeps_the_label_and_the_count(self) -> None:
        # Regression: the shield run's sleep/make_room loop. Clearing the label when the
        # *unblock* succeeds reset the count each time, so the stage-2 explore was unreachable.
        agent = self._agent()
        agent._blocked_label = "sleep"
        agent._unblock_attempts = 1
        agent._note_block("make_room", True, "")
        self.assertEqual(agent._blocked_label, "sleep")
        self.assertEqual(agent._unblock_attempts, 1)

    def test_the_blocked_label_clears_only_when_it_succeeds(self) -> None:
        agent = self._agent()
        agent._blocked_label = "sleep"
        agent._unblock_attempts = 2
        agent._note_block("sleep", False, "[BED_PLACEMENT_FAILED] no flat ground")
        self.assertEqual(agent._blocked_label, "sleep")
        self.assertEqual(agent._unblock_attempts, 2)
        agent._note_block("sleep", True, "")
        self.assertIsNone(agent._blocked_label)
        self.assertEqual(agent._unblock_attempts, 0)

    def test_a_new_block_starts_a_fresh_sequence(self) -> None:
        agent = self._agent()
        agent._note_block("craft_furnace", False, "[WORKSTATION_PLACEMENT_FAILED] no_cell")
        self.assertEqual(agent._blocked_label, "craft_furnace")
        self.assertEqual(agent._unblock_attempts, 0)
        # An unrelated failure is not a surroundings block and must not clear a live one.
        agent._note_block("mine_stone", False, "[HUNT_TARGET_NOT_LOADED] sheep")
        self.assertEqual(agent._blocked_label, "craft_furnace")

    def test_a_full_pack_is_an_inventory_block(self) -> None:
        agent = self._agent()
        agent._note_block("mine_coal", False, "[INVENTORY_FULL] no space for coal_ore.")
        self.assertEqual(agent._blocked_label, "mine_coal")
        self.assertEqual(agent._blocked_kind, "inventory")

    async def test_inventory_block_stows_at_the_base_chest(self) -> None:
        agent = self._agent()
        agent._blocked_label = "mine_coal"
        agent._blocked_kind = "inventory"
        g = gs(pos=(-197.5, -46.0, -98.5), inventory={"tuff": 64}, slots_used=36,
               base=(-197, -47, -98), chest_pos=(-197, -46, -97))
        out = await agent._guard({"before": g, "decision": Decision("mine_coal")})
        self.assertEqual(out["action"].tool, "use_container")
        # The relief itself makes room, so the label is freed to run again next tick.
        self.assertIsNone(agent._blocked_label)
        self.assertEqual(agent.log.records[-1]["reason"], "unblock_inventory")

    async def test_inventory_block_drops_without_a_chest(self) -> None:
        agent = self._agent()
        agent._blocked_label = "mine_coal"
        agent._blocked_kind = "inventory"
        g = gs(pos=(-197.5, -46.0, -98.5), inventory={"tuff": 64}, slots_used=36,
               base=(-197, -47, -98))
        out = await agent._guard({"before": g, "decision": Decision("mine_coal")})
        self.assertEqual(out["action"].tool, "drop_item")

    def test_a_refused_placement_cell_is_remembered_and_cleared(self) -> None:
        # Live, from-zero run: `place_chest` targets the cell beside the base anchor, a
        # `grass_block` stood in it, and the server answered on every tick.
        agent = self._agent()
        act = Action("place_chest", "place_block",
                     {"block_name": "chest", "x": -180, "y": 64, "z": -118}, "place the chest")
        agent._note_place(act, False,
                          "[PLACE_TARGET_OCCUPIED] The target cell contains grass_block")
        self.assertEqual(agent._blocked_cell, (-180, 64, -118))
        dig = Action("make_room", "build_structure",
                     {"blocks": [{"x": -180, "y": 64, "z": -118, "block_name": "air"}]}, "dig it")
        agent._note_place(dig, True, "")
        self.assertIsNone(agent._blocked_cell)

    async def test_the_placement_block_digs_the_refused_cell(self) -> None:
        agent = self._agent()
        agent._blocked_label = "place_chest"
        agent._blocked_cell = (-180, 64, -118)
        g = gs(pos=(-178.5, 63.0, -120.4), blocked_cell=(-180, 64, -118))
        out = await agent._guard({"before": g, "decision": Decision("place_chest")})
        self.assertEqual(out["action"].tool, "build_structure")
        self.assertEqual(out["action"].args["blocks"],
                         [{"x": -180, "y": 64, "z": -118, "block_name": "air"}])


class PourSkipTest(unittest.TestCase):
    """A pour the server refuses for want of a clickable face is remembered, not retried.

    Live, obsidian run: the surface pond holds one usable cell, the bucket answered
    `[BUCKET_NO_FACE]` there on every tick, and the label was offered again and again. That
    answer is a property of the cell, so the loop has to move on to the next lava source.
    """

    class _Action:
        def __init__(self, tool: str, args: dict) -> None:
            self.tool = tool
            self.args = args

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent.furnace_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        return agent

    def test_remembers_the_refused_cell(self) -> None:
        agent = self._agent()
        agent._note_pour(
            self._Action("use_bucket", {"action": "pour", "x": -129, "y": 66, "z": -30}),
            False, "[BUCKET_NO_FACE] No solid block beside the target cell faces the bot.")
        self.assertEqual(agent._pour_skip, {(-129, 66, -30)})

    def test_ignores_success_other_tools_and_other_failures(self) -> None:
        agent = self._agent()
        pour = {"action": "pour", "x": 1, "y": 2, "z": 3}
        agent._note_pour(self._Action("use_bucket", pour), True, "[BUCKET_NO_FACE]")
        agent._note_pour(self._Action("use_bucket", {"action": "fill"}), False,
                         "[BUCKET_NO_FACE]")
        agent._note_pour(self._Action("navigate", pour), False, "[BUCKET_NO_FACE]")
        agent._note_pour(self._Action("use_bucket", pour), False,
                         "[BUCKET_EMPTY] the bucket carries no water")
        self.assertEqual(agent._pour_skip, set())

    def test_inject_hands_the_set_to_the_state(self) -> None:
        agent = self._agent()
        agent.base = None
        agent.chest_pos = None
        agent._pour_skip = {(-129, 66, -30)}
        self.assertEqual(agent._inject(gs()).pour_skip, frozenset({(-129, 66, -30)}))


def _pond_state(**inv):
    """A geared bot far from a two-cell lava pond, the shape a `mine_obsidian` walk is aimed at.

    Out of pouring reach (`pos` is the origin), so the step has to pick a *stance* beside the
    pond instead of pouring where it stands.
    """
    g = gs(pos=(0.0, 64.0, 0.0), tools={"pickaxe": "diamond", "sword": "diamond"},
           inventory={"crafting_table": 1, "water_bucket": 1, **inv})
    g.armor = {slot: {"item": f"diamond_{slot}"}
               for slot in ("helmet", "chestplate", "leggings", "boots")}
    g.offhand = "shield"
    g.resources = {"lava": [
        BlockHit("lava", "lava", -270, 64, -56, 3.0, above="air", below="stone", level=2),
        BlockHit("lava", "lava", -270, 64, -57, 4.0, above="stone", below="stone", level=0),
    ]}
    return g


class StanceSkipTest(unittest.TestCase):
    """A `mine_obsidian` walk the server refuses for want of footing is not walked again.

    Live, run16: the stance the step picked sat over a deep lava lake, so the server answered
    `[NAVIGATION_TARGET_UNSUPPORTED]` ("No ground was found within 64 blocks below it") on 215
    ticks and the run never made a single obsidian. That answer is a property of the stance cell,
    so the loop stands on another neighbour of the same source, and once every stance of a source
    is refused the step passes over the source itself -- the move `pour_skip` makes for a pour.
    """

    class _Action:
        def __init__(self, label: str, tool: str, args: dict | None = None) -> None:
            self.label = label
            self.tool = tool
            self.args = args or {}

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._water_ticks = 0
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent.furnace_pos = None
        agent.chest_pos = None
        agent.base = None
        agent.surface = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.log = []
        return agent

    def test_remembers_the_refused_stance(self) -> None:
        agent = self._agent()
        agent._note_stance(
            self._Action("mine_obsidian", "navigate", {"x": -187, "y": -36, "z": 18}),
            False,
            "[NAVIGATION_TARGET_UNSUPPORTED] No usable footing was observed within 1 blocks of "
            "-187, -36, 18. No ground was found within 64 blocks below it.")
        self.assertEqual(agent._stance_skip, {(-187, -36, 18)})

    def test_ignores_success_other_labels_tools_and_codes(self) -> None:
        agent = self._agent()
        stance = {"x": 1, "y": 2, "z": 3}
        refusal = "[NAVIGATION_TARGET_UNSUPPORTED] no footing"
        agent._note_stance(self._Action("mine_obsidian", "navigate", stance), True, refusal)
        agent._note_stance(self._Action("go_home", "navigate", stance), False, refusal)
        agent._note_stance(self._Action("mine_obsidian", "collect_block", stance), False, refusal)
        agent._note_stance(self._Action("mine_obsidian", "navigate", stance), False,
                           "[NAVIGATION_STOPPED] repeated_movement_failure")
        self.assertEqual(agent._stance_skip, set())

    def test_inject_hands_the_set_to_the_state(self) -> None:
        agent = self._agent()
        agent._stance_skip = {(-187, -36, 18)}
        self.assertEqual(agent._inject(gs()).stance_skip, frozenset({(-187, -36, 18)}))

    def test_the_next_tick_walks_to_another_stance(self) -> None:
        # End to end: the refused walk writes the stance off, `_inject` carries it, and the same
        # label now aims at a different neighbour of the source instead of the refused cell.
        agent = self._agent()
        agent._note_stance(
            self._Action("mine_obsidian", "navigate", {"x": -269, "y": 65, "z": -56}),
            False, "[NAVIGATION_TARGET_UNSUPPORTED] no footing")
        action = resolve("mine_obsidian", agent._inject(_pond_state()))
        self.assertEqual(action.tool, "navigate")
        self.assertEqual((action.args["x"], action.args["y"], action.args["z"]),
                         (-270, 65, -55))


class StuckOfferTest(unittest.TestCase):
    """A label that keeps answering the identical refusal is withheld, so the fallback moves.

    Live, run16: `mine_obsidian` was the *only* candidate and the server refused it identically
    on 215 ticks; because the offer stayed non-empty, the last fallback (`explore`/`wait`) never
    had a turn and the run burned to the iteration limit. The guard catches the repeated refusal
    generically (run11's `go_home` and run14's `recover_drops` stalled the same way, each needing
    a bespoke latch) and hands the tick to the fallback.
    """

    REFUSAL = ("[NAVIGATION_TARGET_UNSUPPORTED] No usable footing was observed within 1 blocks "
               "of -187, -36, 18.")

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._recover_written_off = set()
        agent._water_ticks = 0
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent.furnace_pos = None
        agent.chest_pos = None
        agent.base = None
        agent.surface = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.log = []
        return agent

    def test_withholds_the_label_after_repeated_identical_refusals(self) -> None:
        agent = self._agent()
        for _ in range(STUCK_REPEATS):
            agent._track_stuck("mine_obsidian", False, self.REFUSAL, False)
        self.assertEqual(agent._stuck_off.get("mine_obsidian"), STUCK_COOLDOWN)
        self.assertEqual([r["reason"] for r in agent.log], ["stuck_offer_dropped"])

    def test_progress_and_success_reset_the_run(self) -> None:
        agent = self._agent()
        agent._track_stuck("mine_obsidian", False, self.REFUSAL, False)
        agent._track_stuck("mine_obsidian", False, self.REFUSAL, True)  # the world moved on
        agent._track_stuck("mine_obsidian", False, self.REFUSAL, False)
        self.assertEqual(agent._stuck_reps.get("mine_obsidian"), 1)

    def test_a_different_code_restarts_the_run(self) -> None:
        agent = self._agent()
        agent._track_stuck("mine_obsidian", False, "[CODE_A] x", False)
        agent._track_stuck("mine_obsidian", False, "[CODE_B] x", False)
        self.assertEqual((agent._stuck_code.get("mine_obsidian"),
                          agent._stuck_reps.get("mine_obsidian")), ("CODE_B", 1))

    def test_each_label_keeps_its_own_run(self) -> None:
        agent = self._agent()
        agent._track_stuck("mine_obsidian", False, self.REFUSAL, False)
        agent._track_stuck("get_wood", False, self.REFUSAL, False)
        self.assertEqual(agent._stuck_reps, {"mine_obsidian": 1, "get_wood": 1})

    def test_a_withheld_label_does_not_hide_another_refusing_beside_it(self) -> None:
        # run17: while `build_portal` sat out its window the loop returned early for the whole
        # window, so `light_portal` refusing identically on every tick it got was never counted,
        # never withheld, and burned 150 ticks. Each label is now counted on its own.
        agent = self._agent()
        for _ in range(STUCK_REPEATS):
            agent._track_stuck("build_portal", False, self.REFUSAL, False)
        self.assertIn("build_portal", agent._stuck_off)
        for _ in range(STUCK_REPEATS):
            agent._track_stuck("light_portal", False, self.REFUSAL, False)
        self.assertIn("light_portal", agent._stuck_off)

    def test_the_window_expires_and_the_label_returns(self) -> None:
        agent = self._agent()
        for _ in range(STUCK_REPEATS):
            agent._track_stuck("mine_obsidian", False, self.REFUSAL, False)
        for _ in range(STUCK_COOLDOWN):
            agent._track_stuck("mine_obsidian", False, self.REFUSAL, False)
        self.assertEqual(agent._stuck_off, {})
        self.assertEqual([r["reason"] for r in agent.log][-1], "stuck_offer_restored")

    def test_inject_withholds_only_during_the_window(self) -> None:
        agent = self._agent()
        agent._stuck_off = {"mine_obsidian": 3}
        self.assertEqual(agent._inject(gs()).stuck_labels, frozenset({"mine_obsidian"}))
        agent._stuck_off = {}
        self.assertEqual(agent._inject(gs()).stuck_labels, frozenset())

    def test_candidates_drop_the_withheld_label_and_the_fallback_takes_over(self) -> None:
        # The run16 shape: `mine_obsidian` is the only candidate, so withholding it is what lets
        # the fallback own the tick instead of the run burning to the limit on the same call. The
        # empty base steps and full stone stack keep every other label out of the offer.
        g = _pond_state(cobblestone=20, furnace=1, water_bucket=1)
        g.base = (0, 63, 0)
        self.assertEqual(candidates(g), ["mine_obsidian"])
        g.stuck_labels = frozenset({"mine_obsidian"})
        self.assertEqual(candidates(g), ["explore"])


class FrameCompleteLatchTest(unittest.TestCase):
    """A frame build the server answers `ok` to while spending nothing is not re-offered.

    Live, run25: the frame was finished but the pack still carried nine obsidian, so the
    "resume a partial frame" clause in `candidates` kept offering `build_portal`, and the server
    answered that no-op with a plain success - which resets `_track_stuck`, so no stuck guard ever
    dropped it. The policy chose it on 163 ticks in a row and the bot stood frozen at its own gate.
    `_note_build` reads the answer as "the frame already stands complete"; `candidates` steps over
    the build from then on and the lull work (bow, gold helmet, cobble, food) owns the tick.
    """

    class _Action:
        def __init__(self, label: str) -> None:
            self.label = label

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent._portal_blocked = []
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_off = {}
        agent._recover_written_off = set()
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._water_ticks = 0
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        return agent

    def test_a_build_that_spends_nothing_latches_the_frame_complete(self) -> None:
        agent = self._agent()
        agent._note_build(self._Action("build_portal"), True, {"inventory_delta": {}})
        self.assertTrue(agent._frame_complete)
        # The loop hands the latch to the state, where `candidates`/`resolve` read it.
        self.assertTrue(agent._inject(gs()).frame_complete)

    def test_a_build_that_places_blocks_is_not_the_latch(self) -> None:
        agent = self._agent()
        agent._note_build(self._Action("build_portal"), True,
                          {"inventory_delta": {"obsidian": -2, "cobblestone": -1}})
        self.assertFalse(agent._frame_complete)
        self.assertFalse(agent._inject(gs()).frame_complete)

    def test_only_a_successful_build_portal_latches_it(self) -> None:
        agent = self._agent()
        agent._note_build(self._Action("mine_obsidian"), True, {"inventory_delta": {}})
        agent._note_build(self._Action("build_portal"), False, {"inventory_delta": {}})
        self.assertFalse(agent._frame_complete)


class FrameRefusalTest(unittest.TestCase):
    """A frame cell the build refused is remembered and cleared with an explicit dig.

    `build_structure` cannot dig a container in a frame cell in place: the movement policy
    refuses a break it was not asked for. The build then reports the cell refused on every
    attempt, the frame never closes, and `light_portal` answers `[PORTAL_NO_FRAME]` forever
    (live, run27: run26's furnace stood on the new base's frame corner). `_note_frame_cells`
    reads the cell from the answer so `clear_frame` can dig it with a `collect_block` -- the
    explicit break the policy does allow.
    """

    class _Action:
        def __init__(self, label: str, **args) -> None:
            self.label = label
            self.args = args

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_off = {}
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._water_ticks = 0
        return agent

    _REFUSAL = ("[BUILD_INCOMPLETE] 1 cell still wrong: 1 were refused (229,62,222: movement "
                "policy prohibits breaking this block).")

    def test_a_refused_frame_cell_is_remembered(self) -> None:
        agent = self._agent()
        agent._note_frame_cells(self._Action("build_portal"), False, self._REFUSAL)
        self.assertEqual(agent._refused_frame, {(229, 62, 222)})

    def test_the_cell_rides_into_the_state_with_its_name(self) -> None:
        agent = self._agent()
        agent._note_frame_cells(self._Action("build_portal"), False, self._REFUSAL)
        agent._frame_names = {(229, 62, 222): "furnace"}
        state = agent._inject(gs(base=(228, 62, 222)))
        self.assertEqual(state.refused_frame, ((229, 62, 222),))
        self.assertEqual(state.frame_names, {(229, 62, 222): "furnace"})

    def test_a_refusal_for_another_reason_is_not_remembered(self) -> None:
        agent = self._agent()
        agent._note_frame_cells(self._Action("build_portal"), False,
                                "[BUILD_INCOMPLETE] 1 cell still wrong: 1 were refused (5,6,7: no route).")
        self.assertEqual(agent._refused_frame, set())

    def test_a_finished_build_clears_the_list(self) -> None:
        agent = self._agent()
        agent._note_frame_cells(self._Action("build_portal"), False, self._REFUSAL)
        agent._note_frame_cells(self._Action("build_portal"), True, "")
        self.assertEqual(agent._refused_frame, set())

    def test_clearing_one_cell_drops_only_that_cell(self) -> None:
        agent = self._agent()
        agent._note_frame_cells(
            self._Action("build_portal"), False,
            "[BUILD_INCOMPLETE] 2 cells still wrong: 2 were refused "
            "(229,62,222: movement policy prohibits breaking this block), "
            "(229,66,222: movement policy prohibits breaking this block).")
        agent._note_frame_cells(self._Action("clear_frame", x=229, y=62, z=222), True, "")
        self.assertEqual(agent._refused_frame, {(229, 66, 222)})


class RecoverWriteOffTest(unittest.TestCase):
    """A recovery walk that reaches the death site and finds nothing is not retried.

    Live, run14: `pick_up_items(recover_death_items)` answered `[NO_MATCHING_ITEMS]` at the site
    on every tick while `recover_drops` owned the tick for the whole 300 s window; the bot stood
    there for 40 + 77 ticks, over half the run, instead of resuming the tech tree. The answer is
    definitive (the 16 m scan around the site found nothing), so the site is written off.
    """

    class _Action:
        def __init__(self, label: str) -> None:
            self.label = label

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._recover_written_off = set()
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent.base = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent.surface = None
        return agent

    def test_a_site_found_empty_is_written_off(self) -> None:
        agent = self._agent()
        agent._note_recover(gs(death_pos=(41.45, -22.0, -113.5)),
                            self._Action("recover_drops"), False,
                            "[NO_MATCHING_ITEMS] No matching loaded dropped items were observed.")
        self.assertEqual(agent._recover_written_off, {(41, -22, -114)})

    def test_a_missing_death_record_is_written_off(self) -> None:
        agent = self._agent()
        agent._note_recover(gs(death_pos=(1.0, 2.0, 3.0)),
                            self._Action("recover_drops"), False,
                            "[NO_RECORDED_DEATH] No retained death is available.")
        self.assertEqual(agent._recover_written_off, {(1, 2, 3)})

    def test_ignores_success_other_labels_and_other_failures(self) -> None:
        agent = self._agent()
        death = gs(death_pos=(1.0, 2.0, 3.0))
        # A successful walk (the gear was picked up) must not be written off.
        agent._note_recover(death, self._Action("recover_drops"), True, "[NO_MATCHING_ITEMS]")
        # Only the recovery label; a gather failure with a matching-looking text is ignored.
        agent._note_recover(death, self._Action("get_wood"), False, "[NO_MATCHING_ITEMS]")
        # A transient failure the walk can retry is not final.
        agent._note_recover(death, self._Action("recover_drops"), False,
                            "[RECOVERY_SITE_UNREACHABLE] no route")
        # No death pos to key on: nothing to write off.
        agent._note_recover(gs(), self._Action("recover_drops"), False, "[NO_MATCHING_ITEMS]")
        self.assertEqual(agent._recover_written_off, set())

    def test_inject_hands_the_set_to_the_state(self) -> None:
        agent = self._agent()
        agent._recover_written_off = {(1, 2, 3)}
        self.assertEqual(agent._inject(gs()).recover_written_off, frozenset({(1, 2, 3)}))

    def test_the_next_tick_stops_offering_the_walk(self) -> None:
        # End to end: the failed walk writes the site off, `_inject` carries it, and the same
        # death is no longer offered (the whole point -- the tech tree resumes instead).
        agent = self._agent()
        g = gs(pos=(41.0, -22.0, -113.0), death_pos=(41.45, -22.0, -113.5),
               last_death_age_s=12.0, dimension="overworld", death_dim="overworld")
        agent._note_recover(g, self._Action("recover_drops"), False,
                            "[NO_MATCHING_ITEMS] No matching loaded dropped items were observed.")
        injected = agent._inject(g)
        self.assertNotIn("recover_drops", candidates(injected))
        self.assertIsNone(resolve("recover_drops", injected))


class BuildSiteSurfaceGateTest(unittest.IsolatedAsyncioTestCase):
    """`_scan_build_site` only proposes a base site on open surface ground.

    A scan taken on a cave floor proposes the floor itself: `flat_build_site` cannot tell that the
    `air` over a mined tunnel is not open sky, which is how run9 anchored its base 89 m down. The
    scan therefore waits until `surface_up` has brought the bot out.
    """

    @staticmethod
    def _geared(**kw) -> GameState:
        g = gs(inventory={"crafting_table": 1, **kw.pop("inventory", {})}, **kw)
        g.tools = {"pickaxe": "diamond", "sword": "diamond"}
        g.armor = {s: {"item": f"diamond_{s}"}
                   for s in ("helmet", "chestplate", "leggings", "boots")}
        g.offhand = "shield"
        return g

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.base = None
        agent.build_site = None
        agent._build_site_scanned = None
        agent.mcp = object()  # any real read would touch this and blow up
        return agent

    async def test_no_scan_before_the_surface_is_known(self) -> None:
        agent = self._agent()
        await agent._scan_build_site(self._geared(pos=(-47.5, -15.0, 4.6)))
        self.assertIsNone(agent._build_site_scanned)

    async def test_no_scan_on_a_deep_cave_floor(self) -> None:
        agent = self._agent()
        g = self._geared(pos=(-47.5, -15.0, 4.6))
        g.surface = (-47, 74, 4)
        await agent._scan_build_site(g)
        self.assertIsNone(agent._build_site_scanned)


class SurfaceNoteTest(unittest.TestCase):
    """`_note_surface` learns the surface cell `surface_up` needs to stand on.

    A height-less `navigate` is the only surface probe the client has: an unambiguous column
    resolves and walks the bot out, an ambiguous one stops with `[NAVIGATION_HEIGHT_REQUIRED]` and
    names the surface. `_at_surface` then gates `establish_base` -- a mined tunnel's `air` overhead
    cannot be told from sky (live, run9: the base landed at y=-16 under a y=73 surface).
    """

    class _Action:
        def __init__(self, label: str, args: dict) -> None:
            self.label = label
            self.args = args

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.surface = None
        return agent

    def test_a_resolved_probe_records_the_cell_it_walked_to(self) -> None:
        agent = self._agent()
        agent._note_surface(self._Action("surface_up", {"x": -47, "z": 4}), True, "", (-46.55, 74.0, 3.63))
        self.assertEqual(agent.surface, (-47, 74, 3))

    def test_an_ambiguous_probe_records_the_answered_height(self) -> None:
        agent = self._agent()
        text = ("[NAVIGATION_STOPPED] [NAVIGATION_HEIGHT_REQUIRED] Column -47, 4 has highest "
                "block podzol at y=73; observed standable feet heights: 74, -15, -54.")
        agent._note_surface(self._Action("surface_up", {"x": -47, "z": 4}), False, text, None)
        self.assertEqual(agent.surface, (-47, 74, 4))

    def test_a_failed_climb_keeps_the_cell_for_a_retry(self) -> None:
        # The climb carries a height; a failure on the way (a bad path, a timeout) must not throw
        # the known surface away -- the next tick retries the same climb.
        agent = self._agent()
        agent.surface = (-47, 74, 4)
        agent._note_surface(self._Action("surface_up", {"x": -47, "y": 74, "z": 4}), False,
                            "[NAVIGATION_STOPPED] could not reach", None)
        self.assertEqual(agent.surface, (-47, 74, 4))

    def test_other_labels_are_left_alone(self) -> None:
        agent = self._agent()
        agent._note_surface(self._Action("establish_base", {"x": -47, "z": 4}), True, "",
                            (-47.0, 74.0, 4.0))
        self.assertIsNone(agent.surface)


class BaseAnchorRepairTest(unittest.TestCase):
    """A home base whose own ground was dug out is re-pinned to the ground the server reports.

    `go_home` navigates to `base.y + 1`; nothing re-checks the anchor against the live world once
    `establish_base` records it. Live, run11: the bot quarried its own base to open air, so the
    anchor kept the old surface height and every `go_home` answered
    `[NAVIGATION_TARGET_UNSUPPORTED] ... The ground in that column is at y=70.` for 230 further
    ticks. Re-pinning to that reported height heals it in one tick; a report that no longer moves
    the anchor drops the base so a fresh flat site is scanned.
    """

    class _Action:
        def __init__(self, label: str, args: dict) -> None:
            self.label = label
            self.args = args

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent.base = (419, 73, 594)
        agent.build_site = (419, 73, 594)
        agent._build_site_scanned = (419.0, 594.0)
        return agent

    _UNSUPPORTED = ("[NAVIGATION_TARGET_UNSUPPORTED] No usable footing was observed within 1 "
                    "blocks of 419, 74, 594. The ground in that column is at y=70. Choose an "
                    "observed supported height, or pass build: true to scaffold to this point "
                    "deliberately.")

    def test_the_anchor_is_repinned_to_the_reported_ground(self) -> None:
        agent = self._agent()
        agent._note_base(self._Action("go_home", {"x": 419, "y": 74, "z": 594}), False,
                         self._UNSUPPORTED)
        self.assertEqual(agent.base, (419, 70, 594))
        # The site is kept: only the height was wrong, so no re-scan is forced.
        self.assertEqual(agent.build_site, (419, 73, 594))

    def test_a_report_that_cannot_move_the_anchor_drops_the_base(self) -> None:
        agent = self._agent()
        agent.base = (419, 70, 594)
        agent._note_base(self._Action("go_home", {"x": 419, "y": 71, "z": 594}), False,
                         self._UNSUPPORTED)
        self.assertIsNone(agent.base)
        self.assertIsNone(agent.build_site)
        self.assertIsNone(agent._build_site_scanned)

    def test_a_refusal_at_another_column_leaves_the_base_alone(self) -> None:
        agent = self._agent()
        agent._note_base(self._Action("go_home", {"x": 413, "y": 58, "z": 601}), False,
                         self._UNSUPPORTED)
        self.assertEqual(agent.base, (419, 73, 594))

    def test_ignores_success_other_codes_and_unparseable_text(self) -> None:
        agent = self._agent()
        agent._note_base(self._Action("go_home", {"x": 419, "y": 74, "z": 594}), True,
                         self._UNSUPPORTED)
        agent._note_base(self._Action("go_home", {"x": 419, "y": 74, "z": 594}), False,
                         "[NAVIGATION_STOPPED] repeated_movement_failure")
        agent._note_base(self._Action("go_home", {"x": 419, "y": 74, "z": 594}), False,
                         "[NAVIGATION_TARGET_UNSUPPORTED] no height given")
        self.assertEqual(agent.base, (419, 73, 594))

    def test_no_base_yet_is_left_alone(self) -> None:
        agent = self._agent()
        agent.base = None
        agent._note_base(self._Action("establish_base", {"x": 419, "y": 74, "z": 594}), False,
                         self._UNSUPPORTED)
        self.assertIsNone(agent.base)


class DropHoleTest(unittest.TestCase):
    """A drop whose disposal pit the terrain refuses falls back to the open ground.

    Live, obsidian run: the bot sat in a 1-wide tunnel with a nearly full pack and chose
    `drop_junk` on every tick, but each call answered `[DROP_HOLE_UNAVAILABLE]` (no adjacent
    terrain can be dug into a dry, enclosed pit), so the run stood still. The answer is a
    property of the ground, so the loop remembers it and drops without a pit from then on.
    """

    class _Action:
        def __init__(self, label: str) -> None:
            self.label = label
            self.tool = "drop_item"
            self.args: dict = {}

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent.furnace_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        return agent

    def test_remembers_the_unavailable_pit(self) -> None:
        agent = self._agent()
        agent._note_drop(self._Action("drop_junk"), False,
                         "[DROP_HOLE_UNAVAILABLE] No adjacent terrain can be excavated into "
                         "a dry, enclosed disposal pit.")
        self.assertTrue(agent._drop_hole_blocked)

    def test_ignores_success_and_other_labels(self) -> None:
        agent = self._agent()
        agent._note_drop(self._Action("drop_junk"), True, "[DROP_HOLE_UNAVAILABLE]")
        agent._note_drop(self._Action("stow_surplus"), False, "[DROP_HOLE_UNAVAILABLE]")
        agent._note_drop(self._Action("drop_junk"), False, "[INVENTORY_FULL]")
        self.assertFalse(agent._drop_hole_blocked)

    def test_inject_hands_the_flag_to_the_state(self) -> None:
        agent = self._agent()
        agent.base = None
        agent.chest_pos = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = True
        self.assertTrue(agent._inject(gs()).drop_hole_blocked)


class ChestSupportTest(unittest.TestCase):
    """A chest the base's ground cannot hold is dropped from the chain, not retried forever.

    Live, Nether run: the base stood on a narrow ledge, the cell beside it had no supporting
    block, and `place_chest` answered `[PLACE_NO_SUPPORT]` on every tick, so the run never
    reached the portal. The answer is a property of the base site, so the loop remembers it and
    the inventory relief falls back to `drop_junk`.
    """

    class _Action:
        def __init__(self, label: str) -> None:
            self.label = label
            self.tool = "place_block"
            self.args = {"block_name": "chest", "x": 1, "y": 2, "z": 3}

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent.furnace_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        return agent

    def test_remembers_the_unsupported_cell(self) -> None:
        agent = self._agent()
        agent._note_chest(self._Action("place_chest"), False,
                          "[PLACE_NO_SUPPORT] No adjacent solid block can support placement "
                          "into the target cell.")
        self.assertTrue(agent._chest_blocked)

    def test_ignores_success_and_other_labels(self) -> None:
        agent = self._agent()
        agent._note_chest(self._Action("place_chest"), True, "[PLACE_NO_SUPPORT]")
        agent._note_chest(self._Action("place_bed"), False, "[PLACE_NO_SUPPORT]")
        agent._note_chest(self._Action("place_chest"), False, "[PLACE_BLOCK_FAILED]")
        self.assertFalse(agent._chest_blocked)

    def test_inject_hands_the_flag_to_the_state(self) -> None:
        agent = self._agent()
        agent.base = None
        agent.chest_pos = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = True
        self.assertTrue(agent._inject(gs()).chest_blocked)


class FurnaceSupportTest(unittest.TestCase):
    """The base furnace follows the chest: a ground that refuses it drops the chain.

    `smelt_item` with a temporary workstation places the carried furnace and collects it again
    before returning, so every cooking call in the lull set a furnace up and tore it down again.
    Standing it at the base once and naming its cell fixes that; when the cell cannot hold it,
    the chain is dropped and cooking falls back to the temporary workstation.
    """

    class _Action:
        def __init__(self, label: str, tool: str = "place_block") -> None:
            self.label = label
            self.tool = tool
            self.args = {"block_name": "furnace", "x": 1, "y": 2, "z": 3}

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent.furnace_pos = None
        agent.chest_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        return agent

    def test_remembers_the_unsupported_cell(self) -> None:
        agent = self._agent()
        agent._note_furnace(self._Action("place_furnace"), False,
                            "[PLACE_NO_SUPPORT] No adjacent solid block can support placement "
                            "into the target cell.")
        self.assertTrue(agent._furnace_blocked)

    def test_ignores_success_and_other_labels(self) -> None:
        agent = self._agent()
        agent._note_furnace(self._Action("place_furnace"), True, "[PLACE_NO_SUPPORT]")
        agent._note_furnace(self._Action("place_chest"), False, "[PLACE_NO_SUPPORT]")
        agent._note_furnace(self._Action("place_furnace"), False, "[PLACE_BLOCK_FAILED]")
        self.assertFalse(agent._furnace_blocked)

    def test_inject_hands_the_flag_and_position_to_the_state(self) -> None:
        agent = self._agent()
        agent.base = None
        agent.chest_pos = None
        agent.furnace_pos = (-197, -46, -99)
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._furnace_blocked = True
        state = agent._inject(gs())
        self.assertTrue(state.furnace_blocked)
        self.assertEqual(state.furnace_pos, (-197, -46, -99))

    def test_a_placed_furnace_is_remembered_from_the_scan(self) -> None:
        # After a restart the loop has no memory of the furnace, but it is a real block beside
        # the base: the scan re-adopts its cell, so cooking reuses it instead of a temporary one.
        agent = self._agent()
        agent.base = (-197, -47, -98)
        g = gs(resources={"furnace": [BlockHit("furnace", "furnace", -197, -46, -99, 2.0)]})
        agent._adopt_homebase(g)
        self.assertEqual(agent.furnace_pos, (-197, -46, -99))

    def test_a_far_furnace_is_not_adopted(self) -> None:
        agent = self._agent()
        agent.base = (-197, -47, -98)
        g = gs(resources={"furnace": [BlockHit("furnace", "furnace", -150, -46, -99, 40.0)]})
        agent._adopt_homebase(g)
        self.assertIsNone(agent.furnace_pos)

    def test_a_given_up_furnace_is_not_re_adopted(self) -> None:
        # Abandoning the occupied base furnace (`_furnace_blocked`, `furnace_pos` cleared) must
        # not be undone a tick later by the block scan re-adopting the same standing furnace --
        # cooking would target it again and answer `[FURNACE_NOT_EMPTY]` forever.
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent._furnace_blocked = True
        g = gs(resources={"furnace": [BlockHit("furnace", "furnace", -197, -46, -99, 2.0)]})
        agent._adopt_homebase(g)
        self.assertIsNone(agent.furnace_pos)

    def test_an_occupied_base_furnace_is_abandoned(self) -> None:
        # Live, run26: a completed cook could not recover its output (the pack was full), so the
        # furnace held pre-existing output and every later cook answered `[FURNACE_NOT_EMPTY]`.
        # No MCP tool can open a furnace to drain it (`use_container` handles only location-owned
        # containers), so the cell is dropped: `furnace_pos` cleared, `place_furnace` barred, and
        # cooking falls back to a furnace `smelt_item` carries for the one batch.
        agent = self._agent()
        agent.furnace_pos = (229, 62, 222)
        agent._note_furnace(self._Action("cook_food", "smelt_item"), False,
                            "[FURNACE_NOT_EMPTY] The furnace has pre-existing output; "
                            "nothing was inserted.")
        self.assertTrue(agent._furnace_blocked)
        self.assertIsNone(agent.furnace_pos)

    def test_a_normal_smelt_failure_leaves_the_furnace_standing(self) -> None:
        agent = self._agent()
        agent.furnace_pos = (229, 62, 222)
        agent._note_furnace(self._Action("cook_food", "smelt_item"), False,
                            "[SMELT_NO_FUEL] The furnace ran out of fuel.")
        self.assertFalse(agent._furnace_blocked)
        self.assertEqual(agent.furnace_pos, (229, 62, 222))


class ChestLidTest(unittest.TestCase):
    """A block standing on the chest's lid is dug off before the deposit is retried.

    Live, run26: the base sat two blocks under the surface, a cobblestone came to rest directly
    above the chest, and `use_container` answered `[CHEST_BLOCKED]` on every tick -- so the full
    pack could never be unloaded and every gather stayed refused with `[INVENTORY_FULL]`. The
    block is a property of the cell, not of the attempt, so the loop remembers it and the
    deposit's own resolve digs it clear first.
    """

    class _Action:
        def __init__(self, label: str, tool: str = "use_container") -> None:
            self.label = label
            self.tool = tool
            self.args = {"operation": "deposit"}

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._chest_above = None
        return agent

    def test_remembers_the_block_on_the_lid(self) -> None:
        agent = self._agent()
        agent._note_chest_lid(self._Action("stow_surplus"), False,
                              "[CHEST_BLOCKED] The chest at (229, 62, 224) cannot open: "
                              "cobblestone is directly above it at (229, 63, 224).")
        self.assertEqual(agent._chest_above, (229, 63, 224))

    def test_forgets_it_once_the_stow_succeeds(self) -> None:
        agent = self._agent()
        agent._chest_above = (229, 63, 224)
        agent._note_chest_lid(self._Action("stow_surplus"), True, "")
        self.assertIsNone(agent._chest_above)

    def test_ignores_other_labels_and_other_refusals(self) -> None:
        agent = self._agent()
        agent._note_chest_lid(self._Action("drop_junk", "drop_item"), False, "[CHEST_BLOCKED] x")
        agent._note_chest_lid(self._Action("stow_surplus"), False, "[INVENTORY_FULL] no slot")
        self.assertIsNone(agent._chest_above)

    def test_inject_hands_the_cell_to_the_state(self) -> None:
        agent = self._agent()
        agent.base = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._blocked_cell = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_above = (229, 63, 224)
        self.assertEqual(agent._inject(gs()).chest_above, (229, 63, 224))


class GoalStopTest(unittest.TestCase):
    """The run ends early once the tech tree is complete, unless that is switched off."""

    def _agent(self, stop_when_complete: bool = True) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent.settings = Settings()
        agent.settings.loop.max_iterations = 10
        agent.settings.loop.stop_when_complete = stop_when_complete
        return agent

    @staticmethod
    def _complete() -> GameState:
        # The full overworld goal: diamond kit worn, shield off-hand, the portal lit and the gate
        # bed laid clear of it, and the Nether prep banked (bow + arrows + the cobblestone stack).
        return gs(tools={"pickaxe": "diamond", "sword": "diamond"},
                  inventory={"diamond_sword": 1, "bow": 1, "arrow": 16, "cobblestone": 64},
                  armor={slot: {"item": f"diamond_{slot}"}
                         for slot in ("helmet", "chestplate", "leggings", "boots")},
                  offhand="shield",
                  resources={
                      "nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                 -270, 64, -56, 3.0)],
                      # Two blocks clear of the gate (`BED_PORTAL_CLEARANCE`), as `place_bed` lays
                      # it: a bed in the opening would respawn the bot into the portal.
                      "white_bed": [BlockHit("white_bed", "white_bed", -270, 64, -58, 4.0)],
                  })

    def test_the_goal_ends_the_run_early(self) -> None:
        self.assertEqual(self._agent()._stop_state(self._complete(), 3),
                         {"done": True, "stop": "goal"})

    def test_an_incomplete_run_keeps_going(self) -> None:
        self.assertEqual(self._agent()._stop_state(gs(), 3), {"done": False, "stop": ""})

    def test_the_iteration_limit_still_stops(self) -> None:
        self.assertEqual(self._agent()._stop_state(gs(), 10), {"done": True, "stop": "limit"})

    def test_the_goal_stop_can_be_switched_off(self) -> None:
        self.assertEqual(self._agent(stop_when_complete=False)._stop_state(self._complete(), 3),
                         {"done": False, "stop": ""})


class PortalOpeningTest(unittest.IsolatedAsyncioTestCase):
    """The loop reads the portal opening while a frame build is pending, so a stray block left
    above the support layer is dug before the build (a base the loop adopted from a standing bed,
    or one taken from the `base_target` fallback, never got the `flat_build_site` clearance
    check). Live, a block in the opening left `light_portal` answering `[PORTAL_NO_FRAME]`."""

    class _Read:
        def __init__(self, cells):
            self.data = {"response": {"data": {"result": {"blocks": {"cells": cells}}}}}
            self.text = None

    class _Mcp:
        def __init__(self, cells):
            self.cells = cells
            self.calls: list = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            return PortalOpeningTest._Read(self.cells)

    @staticmethod
    def _cells(blocked=()):
        return [{"x": x, "y": y, "z": -98,
                 "name": "cobblestone" if (x, y) in blocked else "air"}
                for x in (-195, -194) for y in (-46, -45, -44)]

    def _agent(self, cells) -> Agent:
        agent = Agent.__new__(Agent)
        agent.base = (-197, -47, -98)
        agent.chest_pos = None
        agent.furnace_pos = None
        agent._furnace_blocked = False
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        agent.mcp = self._Mcp(cells)
        return agent

    async def test_reads_the_opening_and_flags_a_stray_block(self) -> None:
        agent = self._agent(self._cells(blocked={(-195, -46)}))
        await agent._scan_portal_opening(gs(pos=(-196.5, -46.0, -98.5),
                                                  inventory={"flint_and_steel": 1}))
        self.assertEqual(agent._portal_blocked, [(-195, -46, -98)])
        self.assertEqual(agent.mcp.calls[0][0], "view_blocks")

    async def test_a_clear_opening_leaves_nothing_to_dig(self) -> None:
        agent = self._agent(self._cells())
        await agent._scan_portal_opening(gs(pos=(-196.5, -46.0, -98.5),
                                                  inventory={"flint_and_steel": 1}))
        self.assertEqual(agent._portal_blocked, [])

    async def test_skips_without_flint_and_steel_or_a_base(self) -> None:
        agent = self._agent(self._cells())
        await agent._scan_portal_opening(gs(pos=(-196.5, -46.0, -98.5)))
        self.assertEqual(agent.mcp.calls, [])
        agent.mcp.calls.clear()
        agent.base = None
        await agent._scan_portal_opening(gs(pos=(-196.5, -46.0, -98.5),
                                                  inventory={"flint_and_steel": 1}))
        self.assertEqual(agent.mcp.calls, [])

    async def test_skips_once_the_portal_is_lit(self) -> None:
        agent = self._agent(self._cells(blocked={(-195, -46)}))
        g = gs(pos=(-196.5, -46.0, -98.5), inventory={"flint_and_steel": 1},
               resources={"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                     -195, -46, -98, 3.0)]})
        await agent._scan_portal_opening(g)
        self.assertEqual(agent.mcp.calls, [])

    def test_inject_hands_the_cells_to_the_state(self) -> None:
        agent = self._agent(self._cells())
        agent._portal_blocked = [(-195, -46, -98)]
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent.furnace_pos = None
        agent.build_site = None
        self.assertEqual(agent._inject(gs()).portal_blocked, [(-195, -46, -98)])


class PortalBaseAdoptTest(unittest.IsolatedAsyncioTestCase):
    """A restart re-anchors the home base on the lit portal a previous run left standing.

    `base` is per-run state, but the portal is a real block; its opening corner inverts
    `_portal_site`, so the adopted base puts the frame plane back on the standing obsidian and
    `go_home` walks the bot to it. Without this the loop drifts off to `explore`, and it cannot
    rebuild a portal either (with flint and steel crafted and the obsidian spent, `mine_obsidian`
    is latched off)."""

    class _Read:
        def __init__(self, data):
            self.data = data
            self.text = None

    class _Mcp:
        def __init__(self, data):
            self.data = data
            self.calls: list = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            return PortalBaseAdoptTest._Read(self.data)

    # A lit portal at x=-154..-153, y=63..65, z=257, read at ~76 m - beyond RESOURCE_RADIUS.
    _FIND = {"response": {"data": {"result": {"blocks": {"find": [
        {"name": "nether_portal", "found": 6, "listed": [
            {"x": -154, "y": 63, "z": 257, "name": "nether_portal", "distance": 76.0},
            {"x": -153, "y": 65, "z": 257, "name": "nether_portal", "distance": 76.4},
        ]},
    ]}}}}}
    _NONE = {"response": {"data": {"result": {"blocks": {"find": [
        {"name": "nether_portal", "found": 0, "listed": []}]}}}}}

    def _agent(self, data) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent.base = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent._furnace_blocked = False
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        agent._blocked_cell = None
        agent._chest_above = None
        agent.mcp = self._Mcp(data)
        return agent

    @staticmethod
    def _geared(**inv) -> GameState:
        return gs(pos=(-178.5, 68.0, 329.5),
                  tools={"pickaxe": "diamond", "sword": "diamond"},
                  inventory={"crafting_table": 1, **inv},
                  armor={slot: {"item": f"diamond_{slot}"}
                         for slot in ("helmet", "chestplate", "leggings", "boots")},
                  offhand="shield")

    async def test_adopts_the_base_from_a_portal_past_the_scan(self) -> None:
        agent = self._agent(self._FIND)
        await agent._adopt_portal_base(self._geared())
        self.assertEqual(agent.base, (-156, 62, 257))   # corner - (2, 1, 0)
        self.assertTrue(agent._base_adopted)
        self.assertEqual(agent.mcp.calls[0][0], "view_blocks")

    async def test_no_lit_portal_leaves_the_base_unset(self) -> None:
        agent = self._agent(self._NONE)
        await agent._adopt_portal_base(self._geared())
        self.assertIsNone(agent.base)
        self.assertFalse(agent._base_adopted)

    async def test_skips_once_a_base_is_set(self) -> None:
        agent = self._agent(self._FIND)
        agent.base = (-1, 2, 3)
        await agent._adopt_portal_base(self._geared())
        self.assertEqual(agent.base, (-1, 2, 3))
        self.assertEqual(agent.mcp.calls, [])

    async def test_skips_before_the_diamond_kit(self) -> None:
        agent = self._agent(self._FIND)
        await agent._adopt_portal_base(gs(pos=(0.0, 64.0, 0.0)))
        self.assertIsNone(agent.base)
        self.assertEqual(agent.mcp.calls, [])

    def test_inject_hands_the_adopted_flag_to_the_state(self) -> None:
        agent = self._agent(self._FIND)
        agent._base_adopted = True
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent.furnace_pos = None
        agent.build_site = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        self.assertTrue(agent._inject(gs()).base_adopted)


class BedSiteScanTest(unittest.IsolatedAsyncioTestCase):
    """A standing bed becomes the base only once the loop has proven its ground carries the frame.

    Live, run17: the bot slept underground during the iron phase and `_adopt_homebase` took that
    bed as the base with no clearance check, so the frame was raised into a stone roof -- the top
    and western cells stayed stone and `light_portal` answered `[PORTAL_NO_FRAME]` on 150 ticks.
    The scan now reads the ground around the bed, exactly as `_scan_build_site` does for the bot,
    and only a bed whose site fits the frame is adopted.
    """

    class _Read:
        def __init__(self, data):
            self.data = data
            self.text = None

    class _Mcp:
        """A `view_blocks box` reader over a fixed sparse grid (missing cells read as air)."""

        def __init__(self, grid):
            self.grid = grid
            self.calls: list = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            box = args["box"]
            cx, cy, cz = box["x"], box["y"], box["z"]
            hw, hh = box["half_width"], box["half_height"]
            layers = []
            for y in range(cy - hh, cy + hh + 1):
                rows = []
                for z in range(cz - hw, cz + hw + 1):
                    rows.append({"z": z, "blocks": [
                        self.grid.get((x, y, z), "air")
                        for x in range(cx - hw, cx + hw + 1)]})
                layers.append({"y": y, "rows": rows})
            data = {"response": {"data": {"result": {"blocks": {"box": {
                "center": {"x": cx, "y": cy, "z": cz}, "halfWidth": hw,
                "layers": layers}}}}}}
            return BedSiteScanTest._Read(data)

    @staticmethod
    def _grid(ground: int = 63, roof: bool = False) -> dict:
        grid = {}
        for x in range(-6, 10):
            for z in range(-4, 5):
                for y in range(ground + 1, 72):
                    grid[(x, y, z)] = "stone" if roof else "air"
                grid[(x, ground, z)] = "grass_block"
                grid[(x, ground - 1, z)] = "dirt"
        grid[(0, ground + 1, 0)] = "white_bed"      # the bed the run laid
        return grid

    def _agent(self, grid) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent.base = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent._furnace_blocked = False
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent.mcp = self._Mcp(grid)
        return agent

    @staticmethod
    def _near_bed(**kw) -> GameState:
        return gs(pos=(0.5, 64.0, 0.5), resources={
            "white_bed": [BlockHit("white_bed", "white_bed", 0, 64, 0, 1.0)]}, **kw)

    async def test_a_clear_bed_site_is_recorded_and_adopted(self) -> None:
        agent = self._agent(self._grid())
        g = self._near_bed()
        await agent._scan_bed_site(g)
        self.assertEqual(agent._bed_base, (0, 63, 0))
        self.assertEqual(agent.mcp.calls[0][0], "view_blocks")
        agent._adopt_homebase(g)
        self.assertEqual(agent.base, (0, 63, 0))

    async def test_an_underground_bed_records_no_base(self) -> None:
        agent = self._agent(self._grid(roof=True))
        g = self._near_bed()
        await agent._scan_bed_site(g)
        self.assertIsNone(agent._bed_base)
        agent._adopt_homebase(g)
        self.assertIsNone(agent.base)  # left for `establish_base` to anchor a usable site

    async def test_the_read_is_paid_once_per_bed(self) -> None:
        agent = self._agent(self._grid())
        g = self._near_bed()
        await agent._scan_bed_site(g)
        calls = len(agent.mcp.calls)
        await agent._scan_bed_site(g)
        self.assertEqual(len(agent.mcp.calls), calls)

    async def test_skips_without_a_bed_in_reach(self) -> None:
        agent = self._agent(self._grid())
        await agent._scan_bed_site(gs(pos=(0.5, 64.0, 0.5)))
        self.assertIsNone(agent._bed_base)
        self.assertEqual(agent.mcp.calls, [])

    async def test_skips_once_a_base_is_set(self) -> None:
        agent = self._agent(self._grid())
        agent.base = (-1, 2, 3)
        await agent._scan_bed_site(self._near_bed())
        self.assertEqual(agent.mcp.calls, [])


class NetherCrossingTest(unittest.TestCase):
    """The loop side of the Nether trip: the intended crossing, and the trip's goal.

    Two things cannot live in `actions` alone. The `_in_portal` rescue has to stand down for a
    crossing the loop actually chose, and the powder that reaches the chest has to be remembered
    as a fact of the run rather than re-read out of a chest every tick.
    """

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent.furnace_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_arrival = None
        agent._powder_stowed = False
        agent._water_ticks = 0
        agent.settings = Settings()
        return agent

    def _in_gate(self) -> GameState:
        """The bot standing in the gate, in the state the loop hands `candidates`."""
        g = gs(pos=(-194.5, -46.0, -97.5), base=(-197, -47, -98), dimension="minecraft:overworld",
               health=20.0, food=18.0)
        # The feet and head cells are portal blocks (the mid-teleport case) and one block east is
        # the frame, so this reads as "halfway through the gate" exactly like a live tick does.
        g.resources = {"nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                  -195, -46, -98, 0.4),
                                         BlockHit("nether_portal", "nether_portal",
                                                  -195, -45, -98, 0.9)],
                       "obsidian": [BlockHit("obsidian", "obsidian", -194, -46, -98, 1.0)]}
        return g

    def test_a_chosen_crossing_stands_the_rescue_down(self) -> None:
        agent = self._agent()
        g = self._in_gate()
        # Untouched by any crossing: the rescue owns the tick, as it must.
        self.assertEqual(candidates(agent._inject(g)), ["leave_portal"])
        agent._note_crossing(Action("enter_portal", "enter_nether_portal",
                                    {"x": -195, "y": -46, "z": -98}, ""))
        self.assertNotEqual(candidates(agent._inject(self._in_gate())), ["leave_portal"])

    def test_the_rescue_comes_back_after_the_crossing_window(self) -> None:
        # The intent is a short countdown, not a switch: a gate the bot walks into later on must
        # be rescued again, or the rescue would be disabled for the rest of the run.
        agent = self._agent()
        agent._note_crossing(Action("enter_portal", "enter_nether_portal",
                                    {"x": -195, "y": -46, "z": -98}, ""))
        for _ in range(PORTAL_INTENT_TICKS):
            agent._note_dimension(self._in_gate())
        self.assertEqual(candidates(agent._inject(self._in_gate())), ["leave_portal"])

    def test_ordinary_actions_never_open_the_window(self) -> None:
        agent = self._agent()
        agent._note_crossing(Action("mine_stone", "collect_block",
                                    {"block_name": "stone", "count": 8}, ""))
        self.assertEqual(candidates(agent._inject(self._in_gate())), ["leave_portal"])

    def test_arriving_in_the_nether_is_remembered_with_its_cell(self) -> None:
        # `leave_nether` needs the arrival cell whenever no exit portal block is loaded, and the
        # crossing itself is only observed afterwards as a dimension change.
        agent = self._agent()
        g = gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether")
        agent._note_dimension(g)
        self.assertTrue(agent._nether_entered)
        self.assertEqual(agent._nether_arrival, (12, 65, -4))
        # Standing still down there must not walk the arrival cell along with the bot.
        agent._note_dimension(gs(pos=(40.5, 70.0, -9.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._nether_arrival, (12, 65, -4))

    def test_the_powder_reaching_the_chest_closes_the_goal(self) -> None:
        agent = self._agent()
        deposit = Action("stow_blaze_powder", "use_container",
                         {"operation": "deposit", "x": -196, "y": -46, "z": -98}, "")
        agent._note_stow(deposit, True, {"inventory_delta": {"blaze_powder": -18}})
        self.assertTrue(agent._powder_stowed)
        self.assertTrue(agent._inject(gs()).powder_stowed)

    def test_a_refused_deposit_leaves_the_goal_open(self) -> None:
        # A full chest or a blocked lid must not count as done, or the run stops with the haul
        # still in the pack where a death takes it.
        agent = self._agent()
        deposit = Action("stow_blaze_powder", "use_container",
                         {"operation": "deposit", "x": -196, "y": -46, "z": -98}, "")
        agent._note_stow(deposit, False, {"inventory_delta": {}})
        self.assertFalse(agent._powder_stowed)
        agent._note_stow(deposit, True, {"inventory_delta": {}})
        self.assertFalse(agent._powder_stowed)

    def test_the_stowed_expedition_ends_the_run(self) -> None:
        agent = self._agent()
        agent.settings.loop.stop_when_complete = True
        agent._powder_stowed = True
        self.assertTrue(agent._goal_reached(agent._inject(gs())))


if __name__ == "__main__":
    unittest.main()
