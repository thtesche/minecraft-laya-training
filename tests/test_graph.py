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

import json
from types import SimpleNamespace
import math
import unittest

from mlaya import perception
from mlaya.actions import (Action, candidates, nether_expedition_complete, resolve,
                          tech_tree_complete)
from mlaya.brain import Decision
from mlaya.config import Settings
from mlaya.graph import (Agent, BODY_HELD_STOP, BOXED_REPEATS, DIG_OUT_MAX_TRIES, HOME_ANCHOR_TICKS,
                         RECOVER_UNREACHABLE_REPEATS, shut_in, IDLE_REPEATS, PASS_TIME_LABELS,
                         ORBIT_WINDOW, PORTAL_INTENT_TICKS, STUCK_COOLDOWN, STUCK_REPEATS,
                         WANDER_RATIO, WANDER_REPEATS, WANDER_WALK,
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._nether_ticks = 0
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._water_ticks = 0
        agent._defer_quiet = 0
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

    def test_a_floating_body_swims_instead_of_digging(self) -> None:
        # Live run115 it527 and run116 it0: `make_room`'s `build_structure` hung the runtime event
        # loop in the water pocket at (-0.5,62,281) -- `build_structure` the only operation still
        # active at both `[RUNTIME_UNRESPONSIVE]` incidents. A body with no footing gets the water
        # rescue, not a dug pocket.
        a = unblock_action(gs(pos=(-0.5, 62.4, 281.5), in_water=True, on_ground=False,
                              resources={"stone": [BlockHit("stone", "stone", 4, 62, 281, 4.5,
                                                            above="air", below="stone")]}))
        self.assertEqual(a.label, "escape_water")
        self.assertEqual(a.tool, "navigate")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), (4, 63, 281))


class GuardUnblockTest(unittest.IsolatedAsyncioTestCase):
    class _Log:
        def __init__(self) -> None:
            self.records: list = []

        def append(self, rec) -> None:
            self.records.append(rec)

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
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
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
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
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._pour_skip = {(-129, 66, -30)}
        self.assertEqual(agent._inject(gs()).pour_skip, frozenset({(-129, 66, -30)}))


class NetherLandmarkWriteOffTest(unittest.TestCase):
    """A fortress landmark the explore walk never stepped towards is searched, not retried.

    Live, run104 it155-266: `explore` navigated to the spawner the bot was standing on, the server
    answered `ok` with 0 metres moved, and the identical cell was offered for 112 ticks - magma
    cubes under the feet, no blaze in reach, no rods to get.
    """

    class _Action:
        def __init__(self, label: str, tool: str, args: dict) -> None:
            self.label = label
            self.tool = tool
            self.args = args

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._blocked_cell = None
        agent._chest_above = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        agent.base = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._water_ticks = 0
        return agent

    def test_a_stepless_walk_writes_the_cell_off(self) -> None:
        agent = self._agent()
        agent._note_landmark(self._Action("explore", "navigate",
                                         {"x": 94, "y": 39, "z": 86, "dig": True, "scaffold": True}),
                            {"pos_delta": 0.0}, GameState(pos=(94.4, 40.0, 86.4)))
        self.assertEqual(agent._nether_landmarks, [(94, 39, 86)])

    def test_a_walk_that_reached_the_cage_writes_it_off(self) -> None:
        # Live, run105: the cage at (94,39,86) was re-acquired from 56-113 m out and the walk back
        # *did* move, so the 0-metre rule never fired - the bot ping-ponged 94<->150<->207. A walk
        # that ends standing on the landmark has reached it, however far it came.
        agent = self._agent()
        agent._note_landmark(self._Action("explore", "navigate", {"x": 94, "y": 39, "z": 86}),
                            {"pos_delta": 56.5}, GameState(pos=(94.4, 40.0, 86.4)))
        self.assertEqual(agent._nether_landmarks, [(94, 39, 86)])

    def test_a_walk_still_short_of_the_landmark_keeps_it_open(self) -> None:
        agent = self._agent()
        agent._note_landmark(self._Action("explore", "navigate", {"x": 40, "y": 66, "z": 4}),
                            {"pos_delta": 22.4}, GameState(pos=(20.0, 66.0, 4.0)))
        self.assertEqual(agent._nether_landmarks, [])

    def test_other_labels_and_tools_are_left_alone(self) -> None:
        agent = self._agent()
        on_cell = GameState(pos=(12.4, 65.0, -3.6))
        agent._note_landmark(self._Action("leave_nether", "navigate", {"x": 12, "y": 65, "z": -4}),
                            {"pos_delta": 0.0}, on_cell)
        agent._note_landmark(self._Action("explore", "explore_frontier", {"heading": 90.0}),
                            {"pos_delta": 0.0}, on_cell)
        agent._note_landmark(self._Action("explore", "navigate", {"heading": 90.0}),
                            {"pos_delta": 0.0}, on_cell)
        self.assertEqual(agent._nether_landmarks, [])

    def test_the_note_reaches_the_state(self) -> None:
        agent = self._agent()
        agent._nether_landmarks = [(94, 39, 86)]
        state = agent._inject(GameState())
        self.assertEqual(state.nether_landmarks, frozenset({(94, 39, 86)}))

    def test_coming_home_clears_the_written_off_cages(self) -> None:
        # A new trip searches a new stretch of the world; the cage the last one stood in is not
        # this one's target.
        agent = self._agent()
        agent._nether_landmarks = [(94, 39, 86)]
        agent._last_dimension = "minecraft:the_nether"
        agent._nether_entered = True
        agent._nether_ticks = 40
        agent._nether_travel = 300.0
        agent._nether_last_pos = (94.4, 86.4)
        agent._note_dimension(GameState(dimension="minecraft:overworld", pos=(-179.4, 66.0, 104.6)))
        self.assertEqual(agent._nether_landmarks, [])


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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
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


class RescueSkipTest(unittest.TestCase):
    """A water-rescue foothold the pathfinder could not swim to is not aimed at again.

    Live, run115: treading water at (-0.5,62,281) for 281 rescue walks, every one aimed at a cave
    floor 29 blocks below and every one answered `[NAVIGATION_STOPPED] ... closest node was
    -1,59,281` -- three blocks under the floating body, twenty-nine above the cell it was aiming
    at. The route to that cell does not exist from the pocket, so the rescue takes the next
    foothold instead of diving at the same one until the iteration limit.
    """

    STOPPED = ("[NAVIGATION_STOPPED] Pathfinder stopped: search timed out after 2000 ms compute "
               "(limit 2000 ms); no path or usable partial route found; closest node was -1,59,281.")

    class _Action:
        def __init__(self, label: str, tool: str, args: dict | None = None) -> None:
            self.label = label
            self.tool = tool
            self.args = args or {}

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.base = None
        agent.surface = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.log = []
        return agent

    def test_remembers_the_foothold_the_swim_timed_out_on(self) -> None:
        agent = self._agent()
        agent._note_rescue(
            self._Action("escape_water", "navigate", {"x": -1, "y": 34, "z": 281}),
            False, self.STOPPED)
        self.assertEqual(agent._rescue_skip, {(-1, 34, 281)})

    def test_ignores_success_other_labels_tools_and_codes(self) -> None:
        agent = self._agent()
        cell = {"x": 1, "y": 2, "z": 3}
        agent._note_rescue(self._Action("escape_water", "navigate", cell), True, self.STOPPED)
        agent._note_rescue(self._Action("explore", "navigate", cell), False, self.STOPPED)
        agent._note_rescue(self._Action("escape_water", "collect_block", cell), False, self.STOPPED)
        agent._note_rescue(self._Action("escape_water", "navigate", cell), False,
                           "[NAVIGATION_TARGET_UNSUPPORTED] no footing")
        self.assertEqual(agent._rescue_skip, set())

    def test_inject_hands_the_set_to_the_state(self) -> None:
        agent = self._agent()
        agent._rescue_skip = {(-1, 34, 281)}
        self.assertEqual(agent._inject(gs()).rescue_skip, frozenset({(-1, 34, 281)}))

    def test_the_next_swim_aims_at_the_other_foothold(self) -> None:
        # End to end, to the run115 block: the stopped walk writes the cell off, `_inject` carries
        # it, and the rescue now climbs to the dry ground above the pocket instead of diving.
        agent = self._agent()
        agent._note_rescue(
            self._Action("escape_water", "navigate", {"x": -1, "y": 34, "z": 281}),
            False, self.STOPPED)
        state = gs(in_water=True, on_ground=False, pos=(-0.5, 62.4, 281.5))
        state.rescue_ground = {
            "stone": [BlockHit("stone", "stone", -1, 33, 281, 29.4, above="cave_air")],
            "cobblestone": [BlockHit("cobblestone", "cobblestone", 4, 70, 281, 8.0, above="air")],
        }
        action = resolve("escape_water", agent._inject(state))
        self.assertEqual(action.tool, "navigate")
        self.assertEqual((action.args["x"], action.args["y"], action.args["z"]), (4, 71, 281))


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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
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


class RememberedHomeTest(unittest.IsolatedAsyncioTestCase):
    """A run that lost its base walks back to the chest it last kept.

    `base` is per-run state, so every restart loses it, and the two ways back are both bounded by
    what the world has loaded: the lit portal and the bed are found only within view distance. The
    bot died 300 m from home and the run answered by raising a second base and pouring a second
    portal (live, run47). The chest is the one anchor that is not a read of the world: the server
    keeps every container the bot opened in its own database, which outlives the death, the
    dimension change and the restart.
    """

    class _Read:
        def __init__(self, data):
            self.data = data
            self.text = None

    class _Mcp:
        def __init__(self, rows):
            self.rows = rows
            self.calls: list = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            payload = {"response": {"data": {"result": {"query": {
                "columns": ["block_x", "block_y", "block_z"], "rows": self.rows}}}}}
            return RememberedHomeTest._Read(payload)

    def _agent(self, **over):
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._home_anchor = None
        agent._home_ticks = 0
        agent.log = []
        for k, v in over.items():
            setattr(agent, k, v)
        return agent

    # The live run47 world: the chest at (229,62,224) beside a base at (226,63,222), seen from
    # (12,62,463) -- 300 m away, in a different dimension's worth of walking.
    ROWS = [[229, 62, 224]]

    async def test_a_lost_base_walks_the_bot_back_to_the_chest_it_kept(self) -> None:
        agent = self._agent()
        agent.mcp = self._Mcp(self.ROWS)
        await agent._walk_home_to_memory(gs(pos=(12.0, 62.0, 463.0),
                                            dimension="minecraft:overworld"))
        self.assertEqual(agent.chest_pos, (229, 62, 224))
        # The standing cell beside the chest: the same offset `place_chest` uses to put one there.
        self.assertEqual(agent.base, (229, 61, 225))
        self.assertEqual(agent._home_anchor, (229, 62, 224))

    async def test_the_read_happens_once_and_only_without_a_base(self) -> None:
        agent = self._agent()
        mcp = self._Mcp(self.ROWS)
        agent.mcp = mcp
        await agent._walk_home_to_memory(gs(pos=(12.0, 62.0, 463.0)))
        calls = len(mcp.calls)
        self.assertEqual(calls, 1)
        await agent._walk_home_to_memory(gs(pos=(13.0, 62.0, 463.0)))
        self.assertEqual(len(mcp.calls), calls)   # the base is set, so the rescue is done

    async def test_a_world_with_no_remembered_chest_is_left_alone(self) -> None:
        agent = self._agent()
        agent.mcp = self._Mcp([])
        await agent._walk_home_to_memory(gs(pos=(12.0, 62.0, 463.0)))
        self.assertIsNone(agent.base)
        self.assertIsNone(agent.chest_pos)
        self.assertIsNone(agent._home_anchor)

    async def test_the_nether_never_gets_an_overworld_base(self) -> None:
        # The remembered chest is in the Overworld. Guessing a base from a Nether position is how a
        # run ends up building where it cannot come back from.
        agent = self._agent()
        agent.mcp = self._Mcp(self.ROWS)
        await agent._walk_home_to_memory(gs(pos=(10.0, 65.0, 5.0),
                                            dimension="minecraft:the_nether"))
        self.assertIsNone(agent.base)
        self.assertEqual(agent.mcp.calls, [])

    async def test_a_failed_read_leaves_the_run_free_to_base_itself(self) -> None:
        class _Boom:
            async def read(self, *a):
                raise RuntimeError("ACTION_BUSY")
        agent = self._agent()
        agent.mcp = _Boom()
        await agent._walk_home_to_memory(gs(pos=(12.0, 62.0, 463.0)))
        self.assertIsNone(agent.base)

    def test_arriving_retires_the_guess_so_the_real_anchor_can_be_named(self) -> None:
        # The base was guessed from a chest. Keeping it would pour the frame plane at a cell no
        # scan ever checked; dropping it puts the lit portal and the bed back in reach.
        agent = self._agent(_home_anchor=(229, 62, 224), _home_ticks=HOME_ANCHOR_TICKS)
        agent.base = (229, 61, 225)
        agent._settle_home_anchor(gs(pos=(229.5, 61.0, 225.5), base=(229, 61, 225)))
        self.assertIsNone(agent.base)
        self.assertIsNone(agent._home_anchor)

    def test_the_chest_survives_the_guess_being_retired(self) -> None:
        # The chest is a real observation, not part of the guess: it is the one thing this anchor
        # got right, and `stow_surplus`/`stow_blaze_powder` need it to be the real one.
        agent = self._agent(_home_anchor=(229, 62, 224), _home_ticks=HOME_ANCHOR_TICKS)
        agent.base = (229, 61, 225)
        agent.chest_pos = (229, 62, 224)
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._settle_home_anchor(gs(pos=(229.5, 61.0, 225.5), base=(229, 61, 225)))
        self.assertEqual(agent.chest_pos, (229, 62, 224))

    def test_a_walk_that_never_arrives_gives_up_instead_of_stalling_forever(self) -> None:
        agent = self._agent(_home_anchor=(229, 62, 224), _home_ticks=1)
        agent.base = (229, 61, 225)
        agent._settle_home_anchor(gs(pos=(12.0, 62.0, 463.0), base=(229, 61, 225)))
        self.assertIsNone(agent.base)
        self.assertEqual([r["reason"] for r in agent.log], ["home_anchor_given_up"])

    def test_the_window_ticks_down_while_the_bot_is_still_walking(self) -> None:
        agent = self._agent(_home_anchor=(229, 62, 224), _home_ticks=HOME_ANCHOR_TICKS)
        agent.base = (229, 61, 225)
        agent._settle_home_anchor(gs(pos=(12.0, 62.0, 463.0), base=(229, 61, 225)))
        self.assertEqual(agent._home_ticks, HOME_ANCHOR_TICKS - 1)
        self.assertEqual(agent.base, (229, 61, 225))

    def test_a_new_base_drops_a_chest_stranded_at_the_base_it_moved_off(self) -> None:
        # run100: `_walk_home_to_memory` seeded `chest_pos` from a previous run's chest 322 blocks
        # away, the guess was given up (`_settle_home_anchor`) and `establish_base` raised a fresh
        # base. The stranded chest must not keep `_chest_ready` true at the new base, or no chest is
        # ever built there and `stow_surplus` walks the pack back to a phantom cell.
        agent = self._agent(_bed_base=None, _furnace_blocked=False, base=(-57, 70, 44),
                            chest_pos=(265, 72, 207), _chests_full={(265, 72, 207)},
                            _banked_stores=(((265, 72, 207), "rotten_flesh", 43),),
                            banked={"rotten_flesh": 43})
        agent._adopt_homebase(gs())
        self.assertIsNone(agent.chest_pos)
        self.assertEqual(agent._banked_stores, ())
        self.assertEqual(agent.banked, {})
        self.assertEqual(agent._chests_full, set())

    def test_a_chest_within_reach_of_the_base_is_kept(self) -> None:
        # The real network: the main chest beside the base at `(bx, by+1, bz+1)` and an overflow at
        # the far end of `STORAGE_SITES`. Both sit inside `ADOPT_CHEST_RADIUS`, so neither strands.
        agent = self._agent(_bed_base=None, _furnace_blocked=False, base=(-57, 70, 44),
                            chest_pos=(-57, 71, 45), _extra_chests=[(-55, 71, 48)])
        agent._adopt_homebase(gs())
        self.assertEqual(agent.chest_pos, (-57, 71, 45))
        self.assertEqual(agent._extra_chests, [(-55, 71, 48)])


class IdleOfferGuardTest(unittest.TestCase):
    """A step that keeps succeeding at nothing is dropped like any other unending offer.

    `_track_stuck` counts refusals and `_track_wander` counts walking that arrives nowhere. Neither
    sees a step the server reports as *succeeded* while the world does not change: live, run54, the
    bot standing in the Nether's arrival portal where the rescue owns the tick outright, and
    `leave_portal` answered ok 299 times in four minutes without moving a centimetre - the target
    cell is one step out and the call succeeds where it stands. 299 iterations, no refusals, no
    walking, nothing.
    """

    class _Delta(dict):
        pass

    def _agent(self):
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._stuck_off = {}
        agent._idle_reps = {}
        agent.log = []
        return agent

    @staticmethod
    def _nothing() -> dict:
        return {"inventory_delta": {}, "armor_delta": [], "offhand_delta": [],
                "food_delta": 0, "pos_delta": 0, "phase_changed": False,
                "dimension_changed": False}

    def _some_change(self, **over) -> dict:
        d = self._nothing()
        d.update(over)
        return d

    def test_three_successful_ticks_that_change_nothing_drop_the_label(self) -> None:
        agent = self._agent()
        for _ in range(IDLE_REPEATS):
            agent._track_idle("leave_portal", self._nothing())
        self.assertEqual(agent._stuck_off.get("leave_portal"), STUCK_COOLDOWN)
        self.assertEqual([r["reason"] for r in agent.log], ["idle_offer_dropped"])

    def test_one_tick_that_moved_the_bot_is_not_an_idle_tick(self) -> None:
        # A short walk is progress even under the two blocks `place_bed` counts, so the position
        # is compared exactly rather than by a threshold.
        agent = self._agent()
        for _ in range(IDLE_REPEATS + 2):
            agent._track_idle("leave_portal", self._some_change(pos_delta=1.0))
        self.assertEqual(agent._stuck_off, {})

    def test_a_run_of_idle_ticks_broken_by_one_change_starts_over(self) -> None:
        agent = self._agent()
        agent._track_idle("leave_portal", self._nothing())
        agent._track_idle("leave_portal", self._nothing())
        agent._track_idle("leave_portal", self._some_change(inventory_delta={"beef": 1}))
        self.assertEqual(agent._idle_reps, {})
        for _ in range(IDLE_REPEATS):
            agent._track_idle("leave_portal", self._nothing())
        self.assertIn("leave_portal", agent._stuck_off)

    def test_each_kind_of_change_clears_the_count(self) -> None:
        for delta in (self._some_change(armor_delta=["helmet"]),
                      self._some_change(offhand_delta=["shield"]),
                      self._some_change(food_delta=4),
                      self._some_change(pos_delta=2.0),
                      self._some_change(phase_changed=True),
                      self._some_change(dimension_changed=True)):
            agent = self._agent()
            agent._track_idle("go_home", self._nothing())
            agent._track_idle("go_home", delta)
            self.assertEqual(agent._idle_reps, {}, delta)

    def test_the_steps_that_exist_to_pass_time_are_exempt(self) -> None:
        # A lull really does change nothing. Dropping these would end the run's waiting with
        # nothing to do instead of waiting - the opposite of what the guard is for.
        for label in PASS_TIME_LABELS:
            agent = self._agent()
            for _ in range(IDLE_REPEATS * 3):
                agent._track_idle(label, self._nothing())
            self.assertEqual(agent._stuck_off, {}, label)

    def test_a_label_already_withheld_is_not_counted_again(self) -> None:
        # Same reasoning as the other two guards: while the offer leaves a label out, its ticks
        # say nothing about it.
        agent = self._agent()
        agent._stuck_off["leave_portal"] = STUCK_COOLDOWN
        for _ in range(IDLE_REPEATS * 2):
            agent._track_idle("leave_portal", self._nothing())
        self.assertEqual(agent._idle_reps, {})


class WanderGuardTest(unittest.TestCase):
    """An action that walks a long way and ends where it started is withheld from the offer.

    `_track_stuck` reads refusal codes, so it cannot see this: a wandering action *succeeds*, and
    the position moves every tick, so nothing looked stuck. Live, run43 and run45 each spent an
    action walking 459 blocks to finish 17 from the start -- 27 walked per block gained -- and the
    run managed six decisions in ten minutes while the bot visibly shuffled on the spot.
    """

    class _Res:
        def __init__(self, travelled=None, net=None, ok=True, data=True):
            self.ok = ok
            self.data = {}
            if data and travelled is not None:
                self.data = {"terminal": {"output": {"progress": {
                    "distanceTravelledBlocks": travelled,
                    "distanceFromStartBlocks": net,
                    "current": {"dimension": "minecraft:overworld", "x": 77.5, "z": 183.5},
                }}}}
            self.text = "collected 6x iron_ore"
            if not data and travelled is not None:
                # The server prints the same two numbers into the result text
                # (`formatProgress`), and a client must not depend on the payload keeping its
                # shape: this is the live run43 pair, 458.9 walked and 16.6 net.
                self.text = (f"### Final progress\n**Travel:** {travelled} blocks, including "
                             f"0 during reflexes. **Distance from start:** "
                             f"{'unavailable' if net is None else f'{net} blocks'}.")

    def _agent(self):
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._wander_reps = {}
        agent._stuck_off = {}
        agent.log = []
        return agent

    def test_withholds_after_two_walks_that_arrive_nowhere(self) -> None:
        agent = self._agent()
        agent._track_wander("mine_iron", self._Res(458.9, 16.6))   # live run43
        self.assertEqual(agent._stuck_off, {})                      # one is not yet a pattern
        self.assertEqual(agent._wander_reps, {"mine_iron": 1})
        agent._track_wander("mine_iron", self._Res(245.5, 24.2))   # live run43
        self.assertEqual(agent._stuck_off.get("mine_iron"), STUCK_COOLDOWN)
        self.assertEqual([r["reason"] for r in agent.log], ["wander_offer_dropped"])

    def test_a_walk_that_gets_somewhere_is_left_alone(self) -> None:
        # The east-west Nether march, live run43: 609 walked, 396 net. It is a long walk, and it
        # is progress; withholding it would break the search that depends on it.
        agent = self._agent()
        for _ in range(5):
            agent._track_wander("explore", self._Res(608.8, 396.1))
        self.assertEqual(agent._stuck_off, {})
        self.assertEqual(agent._wander_reps, {})

    def test_a_short_walk_is_never_wandering(self) -> None:
        # Crafting and smelting walk a few blocks at most. The floor keeps them out of the count
        # entirely, so no allowlist of labels is needed to exempt actions that do not travel.
        agent = self._agent()
        for _ in range(5):
            agent._track_wander("craft_item", self._Res(WANDER_WALK - 1, 0.0))
        self.assertEqual(agent._stuck_off, {})

    def test_one_walk_that_arrives_starts_the_run_over(self) -> None:
        agent = self._agent()
        agent._track_wander("mine_iron", self._Res(458.9, 16.6))
        agent._track_wander("mine_iron", self._Res(20.0, 18.0))
        self.assertEqual(agent._wander_reps, {})

    def test_no_progress_reported_is_not_evidence(self) -> None:
        # Not every action reports progress, and a read or a refusal carries none. That is not a
        # reason to withhold anything.
        agent = self._agent()
        for _ in range(4):
            agent._track_wander("mine_iron", self._Res())
        self.assertEqual(agent._stuck_off, {})
        self.assertEqual(agent._wander_reps, {})

    def test_a_withheld_label_is_not_counted_again(self) -> None:
        # Same reasoning as the refusal guard: while the offer leaves a label out, its ticks say
        # nothing about it, and counting them would extend the window on the bot's own silence.
        agent = self._agent()
        agent._stuck_off["mine_iron"] = STUCK_COOLDOWN
        agent._track_wander("mine_iron", self._Res(458.9, 16.6))
        agent._track_wander("mine_iron", self._Res(458.9, 16.6))
        self.assertEqual(agent._wander_reps, {})

    def test_the_window_is_a_countdown_and_the_offer_takes_the_label_back(self) -> None:
        agent = self._agent()
        for _ in range(WANDER_REPEATS):
            agent._track_wander("mine_iron", self._Res(458.9, 16.6))
        agent._stuck_reps = {}
        agent._stuck_code = {}
        agent._track_stuck("smelt_iron", True, "", False)   # any tick ticks the window down
        self.assertEqual(agent._stuck_off.get("mine_iron"), STUCK_COOLDOWN - 1)
        agent._stuck_off["mine_iron"] = 1
        agent._track_stuck("smelt_iron", True, "", False)
        self.assertEqual(agent._stuck_off, {})

    def test_the_net_floor_is_the_walked_over_ratio(self) -> None:
        # The floor is `walked / WANDER_RATIO`: exactly on it counts as arriving, a block under
        # it counts as a shamble. Both calls walk the same distance, so the ratio alone decides.
        agent = self._agent()
        at = WANDER_WALK / WANDER_RATIO
        agent._track_wander("explore", self._Res(WANDER_WALK, at))
        self.assertEqual(agent._wander_reps, {})
        agent._track_wander("explore", self._Res(WANDER_WALK, at - 1))
        self.assertEqual(agent._wander_reps, {"explore": 1})

    def test_the_same_walk_read_out_of_the_result_text_is_caught(self) -> None:
        # No progress in the payload at all: the numbers are still in the text, so the guard has
        # to fire on those too.
        agent = self._agent()
        for _ in range(WANDER_REPEATS):
            agent._track_wander("mine_iron", self._Res(458.9, 16.6, data=False))
        self.assertEqual(agent._stuck_off.get("mine_iron"), STUCK_COOLDOWN)

    def test_an_unavailable_distance_is_no_evidence(self) -> None:
        # The server prints "unavailable" when it could not sample the position. A walk with no
        # end point is not proof of a shamble.
        agent = self._agent()
        for _ in range(4):
            agent._track_wander("mine_iron", self._Res(458.9, None, data=False))
        self.assertEqual(agent._stuck_off, {})


class OrbitGuardTest(unittest.TestCase):
    """A step that moves every tick but keeps bouncing over the same cells is withheld.

    The cross-tick twin of the wander guard. A single `explore` tick walks only a few blocks
    straight, so its own travel-to-net ratio is fine and `_track_wander` never fires; the step
    succeeds and the position moves, so `_track_stuck` and `_track_idle` are blind too. Live,
    run112d it517-599: 83 straight `explore` ticks ping-ponging between (-450.5, 1757.6) and
    (-449.7, 1761.6), seven cells inside five blocks, every tick `progressed`, and the run ran out
    its budget there.
    """

    # The live ping-pong, (x, y, z): two cells about four blocks apart, visited alternately.
    CELL_A = (-450.5, 63.0, 1757.62)
    CELL_B = (-449.7, 63.0, 1761.64)

    def _agent(self):
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._orbit_path = {}
        agent._stuck_off = {}
        agent.log = []
        return agent

    @staticmethod
    def _nothing() -> dict:
        return {"inventory_delta": {}, "armor_delta": [], "offhand_delta": [],
                "food_delta": 0, "pos_delta": 4.0, "phase_changed": False,
                "dimension_changed": False}

    def _pingpong(self, agent, label, n):
        for i in range(n):
            agent._track_orbit(label, self._nothing(),
                               self.CELL_A if i % 2 == 0 else self.CELL_B)

    def test_a_full_window_of_pingpong_drops_the_label(self) -> None:
        agent = self._agent()
        self._pingpong(agent, "explore", ORBIT_WINDOW)
        self.assertEqual(agent._stuck_off.get("explore"), STUCK_COOLDOWN)
        self.assertEqual([r["reason"] for r in agent.log], ["orbit_offer_dropped"])

    def test_the_window_has_to_fill_before_it_fires(self) -> None:
        agent = self._agent()
        self._pingpong(agent, "explore", ORBIT_WINDOW - 1)
        self.assertEqual(agent._stuck_off, {})

    def test_a_straight_march_is_left_alone(self) -> None:
        # Each tick steps a few blocks the same way, so the window walks far and ends far: the
        # frontier roam the pearl search depends on must pass.
        agent = self._agent()
        for i in range(ORBIT_WINDOW * 2):
            agent._track_orbit("explore", self._nothing(), (float(i * 4), 63.0, 0.0))
        self.assertEqual(agent._stuck_off, {})

    def test_a_window_that_never_walks_far_is_not_orbiting(self) -> None:
        # Tiny steps around one spot sum to under the walk floor; that is the idle guard's job,
        # not this one, so nothing here fires.
        agent = self._agent()
        for i in range(ORBIT_WINDOW * 2):
            agent._track_orbit("explore", self._nothing(),
                               (0.1 if i % 2 == 0 else 0.0, 63.0, 0.0))
        self.assertEqual(agent._stuck_off, {})

    def test_a_tick_that_gains_inventory_starts_the_window_over(self) -> None:
        # A bot working a vein bounces over a few cells too; a gained item is real progress,
        # however it moved, so the window resets and the gather is never read as orbiting.
        agent = self._agent()
        for i in range(ORBIT_WINDOW - 1):
            agent._track_orbit("mine_diamond", self._nothing(),
                               self.CELL_A if i % 2 == 0 else self.CELL_B)
        delta = self._nothing()
        delta["inventory_delta"] = {"diamond": 1}
        agent._track_orbit("mine_diamond", delta, self.CELL_B)
        self.assertEqual(agent._orbit_path, {})
        self.assertEqual(agent._stuck_off, {})

    def test_a_dimension_crossing_starts_the_window_over(self) -> None:
        agent = self._agent()
        for i in range(ORBIT_WINDOW - 1):
            agent._track_orbit("leave_nether", self._nothing(),
                               self.CELL_A if i % 2 == 0 else self.CELL_B)
        delta = self._nothing()
        delta["dimension_changed"] = True
        agent._track_orbit("leave_nether", delta, self.CELL_B)
        self.assertEqual(agent._orbit_path, {})

    def test_the_steps_that_pass_time_are_exempt(self) -> None:
        for label in PASS_TIME_LABELS:
            agent = self._agent()
            self._pingpong(agent, label, ORBIT_WINDOW * 2)
            self.assertEqual(agent._stuck_off, {}, label)

    def test_a_withheld_label_is_not_counted_again(self) -> None:
        agent = self._agent()
        agent._stuck_off["explore"] = STUCK_COOLDOWN
        self._pingpong(agent, "explore", ORBIT_WINDOW * 2)
        self.assertEqual(agent._orbit_path, {})

    def test_no_position_is_no_evidence(self) -> None:
        agent = self._agent()
        for _ in range(ORBIT_WINDOW * 2):
            agent._track_orbit("explore", self._nothing(), None)
        self.assertEqual(agent._stuck_off, {})


class FurnaceDrainReadTest(unittest.TestCase):
    """The look into the furnace is how the drain gets its item list, so it must be read.

    `collect_smelted` is two calls: an `inspect` to learn what is in the cell, then a `withdraw`
    of exactly that. The second call can only be built from the first one's answer, and the loop
    was clearing its own state on the successful inspect -- as if looking had emptied the furnace.
    So the step inspected on every tick, sent no `withdraw` at all, and the cook behind it met
    `[FURNACE_NOT_EMPTY]` again (live, run50: five successful `collect_smelted`, nothing drained,
    nothing in the pack).
    """

    ANSWER = {"response": {"data": {"state": "settled", "output": {
        "result": {"status": "succeeded", "container": {
            "blockName": "furnace", "slotCount": 3,
            "contents": [{"slot": 2, "item": "cooked_beef", "count": 1}]}}}}}}

    class _Res:
        def __init__(self, data):
            self.ok = True
            self.text = ""
            self.data = data

    class _Action:
        def __init__(self, label, tool, args=None):
            self.label = label
            self.tool = tool
            self.args = args or {}

    def _agent(self):
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.log = []
        return agent

    def _inspect(self, operation="inspect"):
        return self._Action("collect_smelted", "use_container", {"operation": operation,
                                                                "x": 229, "y": 62, "z": 222})

    def test_the_inspect_is_remembered_as_what_the_furnace_holds(self) -> None:
        agent = self._agent()
        agent._note_furnace(self._inspect(), True, "", self._Res(self.ANSWER))
        self.assertEqual(agent._furnace_contents, {"cooked_beef": 1})
        self.assertTrue(agent._furnace_occupied)

    def test_the_withdraw_empties_the_cell(self) -> None:
        agent = self._agent()
        agent._note_furnace(self._inspect(), True, "", self._Res(self.ANSWER))
        agent._note_furnace(self._inspect("withdraw"), True, "", self._Res({}))
        self.assertEqual(agent._furnace_contents, {})
        self.assertFalse(agent._furnace_occupied)

    def test_an_empty_window_leaves_the_furnace_marked_empty(self) -> None:
        agent = self._agent()
        agent._furnace_occupied = True
        empty = {"response": {"data": {"result": {"container": {
            "blockName": "furnace", "contents": []}}}}}
        agent._note_furnace(self._inspect(), True, "", self._Res(empty))
        self.assertFalse(agent._furnace_occupied)

    def test_a_refused_inspect_changes_nothing(self) -> None:
        agent = self._agent()
        agent._furnace_occupied = True
        agent._note_furnace(self._inspect(), False, "[CONTAINER_NOT_PRESENT] not a container")
        self.assertTrue(agent._furnace_occupied)
        self.assertEqual(agent._furnace_contents, {})

    def test_the_second_call_is_now_buildable_from_what_the_first_learned(self) -> None:
        # The end of the whole thing: after the inspect the state names the item, so `resolve`
        # produces a withdraw of it rather than another look.
        from mlaya.actions import candidates, resolve
        agent = self._agent()
        agent._note_furnace(self._inspect(), True, "", self._Res(self.ANSWER))
        agent.furnace_pos = (229, 62, 222)
        agent.smoker_pos = None
        state = gs(inventory={}, base=agent.furnace_pos, furnace_pos=agent.furnace_pos,
                   furnace_occupied=agent._furnace_occupied,
                   furnace_contents=dict(agent._furnace_contents))
        state.base_adopted = frozenset()
        state.pos = (229.5, 62.0, 222.5)
        a = resolve("collect_smelted", state)
        self.assertIsNotNone(a)
        self.assertEqual(a.tool, "use_container")
        self.assertEqual(a.args["operation"], "withdraw")
        self.assertEqual(a.args["items"], [{"item_name": "cooked_beef", "count": 1}])


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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent._portal_blocked = []
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_off = {}
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
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

    def test_the_plane_read_latches_a_finished_frame(self) -> None:
        # The build that finishes the frame *spends* its obsidian, so `_note_build` does not latch
        # (that is the no-op build only). The plane read is what proves it finished, and it must
        # latch too: the read goes blank once the bot roams off the base and the chunks unload, and
        # a frame read back as unbuilt reopened the obsidian chain (live run97 it142-145, 290 m out).
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent._frame_names = {cell: "obsidian"
                              for cell in perception.frame_obsidian_cells(agent.base)}
        self.assertTrue(agent._inject(gs()).frame_complete)
        self.assertTrue(agent._frame_complete)

    def test_a_partial_plane_read_does_not_latch(self) -> None:
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent._frame_names = {cell: "obsidian"
                              for cell in perception.frame_obsidian_cells(agent.base)[:-1]}
        self.assertFalse(agent._inject(gs()).frame_complete)
        self.assertFalse(agent._frame_complete)

    def test_a_frame_that_lost_obsidian_unlatches(self) -> None:
        # run103: the build spent its ten obsidian at it114 and the plane read latched the frame
        # complete on the next tick; by it191 two of those cells were air again (the pack had
        # picked the two blocks back up at it140). With the latch still set `candidates` stepped
        # over `build_portal`, `light_portal` refused on the same read, and the bot `wait`ed at
        # its own gate with two obsidian in the pack for fifty ticks.
        agent = self._agent()
        agent.base = (-181, 64, 101)
        cells = perception.frame_obsidian_cells(agent.base)
        agent._frame_complete = True
        agent._frame_names = {cell: "obsidian" for cell in cells}
        for cell in cells[2:4]:
            agent._frame_names[cell] = "air"
        self.assertFalse(agent._inject(gs()).frame_complete)
        self.assertFalse(agent._frame_complete)

    def test_the_unlatch_reopens_the_build_with_the_two_blocks_left(self) -> None:
        agent = self._agent()
        agent.base = (-181, 64, 101)
        cells = perception.frame_obsidian_cells(agent.base)
        agent._frame_complete = True
        agent._frame_names = {cell: "obsidian" for cell in cells}
        for cell in cells[2:4]:
            agent._frame_names[cell] = "air"
        g = agent._inject(gs(base=agent.base, pos=(-181.5, 65.0, 101.5), offhand="shield",
                             inventory={"obsidian": 2, "cobblestone": 58, "flint_and_steel": 1},
                             tools={"pickaxe": "diamond", "sword": "diamond"},
                             armor={slot: {"item": f"diamond_{slot}"}
                                    for slot in ("helmet", "chestplate", "leggings", "boots")}))
        self.assertIn("build_portal", candidates(g))

    def test_a_blank_plane_read_keeps_the_latch(self) -> None:
        # The run97 protection stays: a bot roaming off the base reads the plane blank, and a
        # frame read back as unbuilt reopened the obsidian chain 290 m out. A read that names
        # nothing is not the frame talking about itself, so it cannot undo the latch.
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent._frame_complete = True
        agent._frame_names = {}
        self.assertTrue(agent._inject(gs()).frame_complete)

    def test_an_all_air_plane_read_keeps_the_latch(self) -> None:
        # No obsidian named in the plane at all: nothing here says "our frame is short", so the
        # latch holds and the lull work keeps the tick instead of a rebuild that cannot fill.
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent._frame_complete = True
        agent._frame_names = {cell: "air"
                             for cell in perception.frame_obsidian_cells(agent.base)}
        self.assertTrue(agent._inject(gs()).frame_complete)


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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_off = {}
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._nether_ticks = 0
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._water_ticks = 0
        agent._defer_quiet = 0
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent.base = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
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

    def test_a_site_unreachable_again_and_again_is_written_off(self) -> None:
        # run90: 231 walks from (182.5, 72, 179.5) to a site 50 m off, all
        # `[RECOVERY_SITE_UNREACHABLE]`, until the death record went stale.
        agent = self._agent()
        death = gs(death_pos=(228.74, 72.0, 156.7))
        refused = "[RECOVERY_SITE_UNREACHABLE] no path found after 2 ms compute"
        for _ in range(RECOVER_UNREACHABLE_REPEATS - 1):
            agent._note_recover(death, self._Action("recover_drops"), False, refused)
        # A reflex interruption between them neither counts nor starts over.
        agent._note_recover(death, self._Action("recover_drops"), False,
                            "[HOSTILE_CONTACT] fight response for skeleton#1")
        self.assertEqual(agent._recover_written_off, set())
        agent._note_recover(death, self._Action("recover_drops"), False, refused)
        self.assertEqual(agent._recover_written_off, {(228, 72, 156)})

    def test_a_new_death_site_starts_the_count_over(self) -> None:
        agent = self._agent()
        refused = "[RECOVERY_SITE_UNREACHABLE] no route"
        for _ in range(RECOVER_UNREACHABLE_REPEATS - 1):
            agent._note_recover(gs(death_pos=(1.0, 2.0, 3.0)), self._Action("recover_drops"),
                                False, refused)
        agent._note_recover(gs(death_pos=(9.0, 2.0, 3.0)), self._Action("recover_drops"),
                            False, refused)
        self.assertEqual(agent._recover_written_off, set())

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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._blocked_cell = None
        agent._chest_above = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        return agent

    def test_the_drop_keeps_the_way_the_bot_walked(self) -> None:
        # The toss flies the way the bot faces, which is the way it last walked; the walk off
        # the drop reads that heading (run87: dug west, tossed west, stepped west into the pile).
        agent = self._agent()
        agent._drop_hole_blocked = True
        agent._note_heading((-59.5, 85.0, 44.5), (-93.5, 6.0, 35.3))
        agent._note_heading((-93.5, 6.0, 35.3), (-93.6, 6.0, 35.3))
        g = GameState()
        g.pos = (-93.5, 6.0, 35.3)
        agent._note_drop(self._Action("drop_junk"), True, "", g)
        hx, hz = agent._drop_heading
        self.assertLess(hx, -0.9)
        self.assertAlmostEqual(math.hypot(hx, hz), 1.0)

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
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = True
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.chest_pos = None
        agent.banked = {}
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
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
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
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
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = (-197, -46, -99)
        agent.smoker_pos = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
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

    def test_an_occupied_base_furnace_is_drained_rather_than_abandoned(self) -> None:
        # Live, run26: a completed cook could not recover its output (the pack was full), so the
        # furnace held pre-existing output and every later cook answered `[FURNACE_NOT_EMPTY]`.
        # The old answer was to drop the cell and cook in a carried furnace, because no tool could
        # open a furnace. The server now opens one for `inspect`/`withdraw`, so the refusal is now
        # a flag to drain it - the cell stays, and the run keeps its base furnace.
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent.furnace_pos = (229, 62, 222)
        agent.smoker_pos = None
        agent._note_furnace(self._Action("cook_food", "smelt_item"), False,
                            "[FURNACE_NOT_EMPTY] The furnace has pre-existing output; "
                            "nothing was inserted.")
        self.assertTrue(agent._furnace_occupied)
        self.assertFalse(agent._furnace_blocked)
        self.assertEqual(agent.furnace_pos, (229, 62, 222))

    def test_a_server_that_cannot_open_a_furnace_still_gives_the_cell_up(self) -> None:
        # Without server-side furnace support the drain cannot work, and that is the one case no
        # retry fixes: back to a carried furnace for the batch.
        agent = self._agent()
        agent.furnace_pos = (229, 62, 222)
        agent.smoker_pos = None
        agent._note_furnace(self._Action("collect_smelted", "use_container"), False,
                            "[CONTAINER_OPERATION_UNSUPPORTED] furnace holds input, fuel and "
                            "output slots rather than storage...")
        self.assertTrue(agent._furnace_blocked)
        self.assertIsNone(agent.furnace_pos)

    def test_draining_the_furnace_clears_the_flag(self) -> None:
        agent = self._agent()
        agent.base = (-197, -47, -98)
        agent.furnace_pos = (229, 62, 222)
        agent.smoker_pos = None
        agent._furnace_occupied = True
        agent._note_furnace(self._Action("collect_smelted", "use_container"), True, "")
        self.assertFalse(agent._furnace_occupied)

    def test_a_normal_smelt_failure_leaves_the_furnace_standing(self) -> None:
        agent = self._agent()
        agent.furnace_pos = (229, 62, 222)
        agent.smoker_pos = None
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
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
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._blocked_cell = None
        agent.build_site = None
        agent._base_adopted = False
        agent._portal_base_scanned = None
        agent.surface = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_above = (229, 63, 224)
        self.assertEqual(agent._inject(gs()).chest_above, (229, 63, 224))


class GoalStopTest(unittest.TestCase):
    """The run ends early once *every* phase is done, unless that is switched off.

    The Overworld goal closes one tick before the crossing it prepares - lit portal, gear worn,
    bed at the base, prep banked, food reserve met, with `enter_portal` in the offer right behind
    it. Read as a goal on its own it ended every run there, so the Nether only ever ran from a
    wrecked state, the one where the gate had already been broken by a death (live, run29:
    `{"iterations": 180, "stop": "goal"}`, expedition never started).
    """

    def _agent(self, stop_when_complete: bool = True) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent.settings.loop.max_iterations = 10
        agent.settings.loop.stop_when_complete = stop_when_complete
        agent._held_ticks = 0
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
        g = self._complete()
        g.powder_stowed = True          # the expedition is what the overworld goal was preparing
        g.end_portal_active = True      # and the stronghold is what the expedition was for
        self.assertEqual(self._agent()._stop_state(g, 3),
                         {"done": True, "stop": "goal"})

    def test_the_stowed_powder_alone_keeps_the_run_going(self) -> None:
        # The Nether haul is no longer the last phase: the run carries on to the stronghold and
        # stops only once the end portal stands active as well (docs/stronghold.md).
        g = self._complete()
        g.powder_stowed = True
        self.assertEqual(self._agent()._stop_state(g, 3), {"done": False, "stop": ""})

    def test_the_overworld_goal_alone_keeps_the_run_going(self) -> None:
        # The whole point: this is the state the loop reaches one tick before `enter_portal`, and
        # ending there is what kept every run out of the Nether.
        g = self._complete()
        self.assertEqual(self._agent()._stop_state(g, 3), {"done": False, "stop": ""})

    def test_an_incomplete_run_keeps_going(self) -> None:
        self.assertEqual(self._agent()._stop_state(gs(), 3), {"done": False, "stop": ""})

    def test_the_iteration_limit_still_stops(self) -> None:
        self.assertEqual(self._agent()._stop_state(gs(), 10), {"done": True, "stop": "limit"})

    def test_the_goal_stop_can_be_switched_off(self) -> None:
        g = self._complete()
        g.powder_stowed = True
        self.assertEqual(self._agent(stop_when_complete=False)._stop_state(g, 3),
                         {"done": False, "stop": ""})


class ExpeditionDepartureChainTest(unittest.TestCase):
    """From a base found again after a death to a stocked crossing, in one chain.

    This is the question the run has to answer before it can go through the gate, and it is a
    chain rather than a predicate: every step is offered only while its predecessor is done, so a
    gap anywhere shows up as the next step missing rather than as a failure at the gate. The state
    is the live one - base (226,63,222), chest (229,62,224), gate at (228,64,222), ten obsidian,
    the full diamond kit worn - because the numbers are what the gates compare against.
    """

    BASE = (226, 63, 222)
    CHEST = (229, 62, 224)
    GATE = (228, 64, 222)
    NETHER = "minecraft:the_nether"
    OVERWORLD = "minecraft:overworld"

    def _standing(self, **inv):
        """The bot at its base with the expedition already banked, short of what it lacks."""
        stock = dict(flint_and_steel=1, bow=1, arrow=16, cobblestone=64,
                     golden_helmet=1, cooked_mutton=64, white_bed=1, diamond_pickaxe=2,
                     diamond=3, stick=4)
        stock.update(inv)
        # The full diamond kit worn and the shield in the off-hand (`_gear_ready`), not merely
        # carried: the expedition gate asks for the armour on the body.
        g = GameState()
        g.inventory = dict(stock)
        g.tools = {"pickaxe": "diamond", "sword": "diamond"}
        g.armor = {slot: {"item": f"diamond_{slot}"}
                   for slot in ("helmet", "chestplate", "leggings", "boots")}
        g.offhand = "shield"
        # Stand the bot on its base, so the base-only steps are eligible (`_at_base`).
        g.base = self.BASE
        g.pos = (self.BASE[0] + 0.5, self.BASE[1] + 1.0, self.BASE[2] + 0.5)
        g.resources = {
            "nether_portal": [BlockHit("nether_portal", "nether_portal", *self.GATE, 2.0)],
            "obsidian": [BlockHit("obsidian", "obsidian", 226, 64, 221, 2.0)],
            "white_bed": [BlockHit("white_bed", "white_bed", 225, 64, 223, 2.5)],
            "furnace": [BlockHit("furnace", "furnace", 229, 62, 222, 4.5)],
        }
        g.dimension = self.OVERWORLD
        g.chest_pos = self.CHEST
        g.furnace_pos = (229, 62, 222)
        g.health = 20.0
        g.food = 18.0
        return g

    def _agent(self):
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._home_anchor = None
        agent._home_ticks = 0
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        return agent

    def test_the_gate_is_offered_and_is_the_crossing(self) -> None:
        g = self._standing()
        out = candidates(g)
        self.assertIn("enter_portal", out)
        a = resolve("enter_portal", g)
        self.assertEqual(a.tool, "enter_nether_portal")
        self.assertEqual((a.args["x"], a.args["y"], a.args["z"]), self.GATE)
        # The server pauses a crossing below 32 arrows and this run banks 16, so the crossing
        # would otherwise be refused forever (live, run41).
        self.assertTrue(a.args["allow_low_supplies"])

    def test_the_second_pickaxe_is_the_only_thing_between_the_base_and_the_gate(self) -> None:
        # One pickaxe is not a banked expedition: it breaks in the Nether and nothing is
        # recoverable there (`_can_recover` refuses the dimension on purpose).
        g = self._standing(diamond_pickaxe=1)
        self.assertNotIn("enter_portal", candidates(g))
        self.assertIn("craft_pickaxe_spare", candidates(g))

    def test_the_spare_is_built_from_what_the_run_can_mine(self) -> None:
        g = self._standing(diamond_pickaxe=1, diamond=3, stick=4)
        a = resolve("craft_pickaxe_spare", g)
        self.assertEqual(a.tool, "craft_item")
        self.assertEqual(a.args["items"], [{"item_name": "diamond_pickaxe"}])

    def test_the_spare_fetches_wood_for_its_sticks_instead_of_waiting(self) -> None:
        # run100 it~285-319: one diamond pickaxe, three diamonds, a crafting table and a healthy
        # tool, but no wood. The spare cannot be crafted (no sticks), `mine_diamond` is not short,
        # and the `get_wood` that supplies the sticks was gated on the tool being worn
        # (`_pickaxe_due_for_spare`), not on the trip still wanting its second pickaxe
        # (`_spare_pickaxe_wanted`) -- so the lull fell through to `wait` and the gate was never lit.
        # The table is carried on purpose: without it the `craft_table` plank need masks the bug,
        # offering `get_wood` for the table rather than for the sticks, and the test proves nothing.
        g = self._standing(diamond_pickaxe=1, stick=0, crafting_table=1)
        g.inventory.pop("stick", None)
        out = candidates(g)
        self.assertIn("get_wood", out)
        self.assertNotEqual(out, ["wait"])
        self.assertNotIn("craft_pickaxe_spare", out)   # no sticks yet: offered once the wood is in

    def test_a_walked_away_bot_is_brought_back_for_the_crossing(self) -> None:
        # The gate is base-gated and nothing else pulls a stocked bot home, which is how run30
        # sat fully stocked and lit with `enter_portal` never offered.
        g = self._standing()
        g.pos = (-250.5, 40.0, -120.5)
        self.assertIn("go_home", candidates(g))

    def test_the_run_is_not_finished_by_standing_at_the_gate(self) -> None:
        # The state one tick before the crossing: the overworld goal holds, the expedition has
        # not happened, and the run has to keep going. Ending here is what kept every run out of
        # the Nether (run29, `{"stop": "goal"}` at iteration 180).
        agent = self._agent()
        g = agent._inject(self._standing())
        self.assertTrue(tech_tree_complete(g))
        self.assertFalse(nether_expedition_complete(g))
        self.assertFalse(agent._goal_reached(g))
        self.assertEqual(agent._stop_state(g, 3), {"done": False, "stop": ""})

    def test_the_run_finishes_when_the_portal_is_open_and_the_base_is_in_order(self) -> None:
        agent = self._agent()
        agent._powder_stowed = True
        agent._end_portal_active = True
        g = agent._inject(self._standing())
        self.assertTrue(agent._goal_reached(g))
        self.assertEqual(agent._stop_state(g, 3), {"done": True, "stop": "goal"})

    def test_the_search_down_there_walks_the_unexplored_edge(self) -> None:
        # The crossing is only the start of the trip: down there the step is the fortress search,
        # and it follows the world's frontier rather than a fixed compass line.
        g = gs(pos=(10.5, 65.0, 4.5), dimension=self.NETHER, health=20.0, food=18.0,
               inventory={"diamond_pickaxe": 2}, tools={"pickaxe": "diamond", "sword": "diamond"},
               armor={s: {"item": f"diamond_{s}"} for s in
                      ("helmet", "chestplate", "leggings", "boots")},
               offhand="shield", base=self.BASE)
        g.nether_entered = True
        g.nether_arrival = (10, 65, 4)
        g.frontier_heading = 180.0
        g.golden_helmet = True
        a = resolve("explore", g)
        self.assertEqual(a.tool, "explore_frontier")
        self.assertEqual(a.args["heading"], 180.0)


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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = (-197, -47, -98)
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
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
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        self.assertEqual(agent._inject(gs()).portal_blocked, [(-195, -46, -98)])


class StockedChestReadTest(unittest.IsolatedAsyncioTestCase):
    """The chest read must not be able to fail quietly into looking like an empty chest.

    Written after shipping it: the read was passed `{"query": ...}` where the tool wants `sql`,
    and the `except Exception: return` turned that into a perfectly ordinary empty `banked`. Live
    that reads as "no food is banked", which is the exact state the read exists to disprove - the
    run went on hunting beside a full chest, and nothing in the log said why.
    """

    FULL = {"response": {"data": {"result": {"query": {"rows": [
        [229, 62, 224, '[{"slot":0,"item":"cooked_mutton","count":21},'
                       '{"slot":1,"item":"obsidian","count":64}]']]}}}}}

    class _Read:
        """Stands in for the `ToolResult` the executor hands back, not for the payload."""

        def __init__(self, data):
            self.data, self.text = data, ""

    class _Mcp:
        def __init__(self, data=None, boom: bool = False):
            self.data, self.boom = data, boom
            self.calls: list[tuple[str, dict]] = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            if self.boom:
                raise RuntimeError("Invalid arguments: sql: Missing key")
            return StockedChestReadTest._Read(self.data)

    def _agent(self, mcp) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = (-197, -47, -98)
        agent.chest_pos = (229, 62, 224)
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._stock_read = None
        agent._stock_warned = False
        agent.log = []
        agent.mcp = mcp
        return agent

    def _state(self):
        return gs(pos=(0.5, 64.0, 0.5), dimension="minecraft:overworld")

    async def test_the_read_asks_for_the_column_it_needs(self) -> None:
        mcp = self._Mcp(self.FULL)
        agent = self._agent(mcp)
        await agent._note_stocked_chest(self._state())
        self.assertEqual(mcp.calls[0][1]["sql"], perception.REMEMBERED_STOCKS_SQL)
        self.assertEqual(agent.banked, {"cooked_mutton": 21})

    async def test_only_the_edibles_are_kept(self) -> None:
        # The obsidian is in the same column. Carrying it into every state, every narrative and
        # every logged tick would buy nothing: the field exists to answer one question.
        agent = self._agent(self._Mcp(self.FULL))
        await agent._note_stocked_chest(self._state())
        self.assertNotIn("obsidian", agent.banked)

    async def test_a_broken_read_is_logged_not_swallowed(self) -> None:
        agent = self._agent(self._Mcp(boom=True))
        await agent._note_stocked_chest(self._state())
        self.assertEqual(agent.banked, {})
        self.assertEqual([r["reason"] for r in agent.log], ["chest_stock_unreadable"])

    async def test_the_read_is_not_asked_again_from_the_same_cell(self) -> None:
        # It is a database read on every tick of a long run; the loop is the only writer, so the
        # same cell cannot have anything new to say.
        mcp = self._Mcp(self.FULL)
        agent = self._agent(mcp)
        await agent._note_stocked_chest(self._state())
        await agent._note_stocked_chest(self._state())
        self.assertEqual(len(mcp.calls), 1)
        await agent._note_stocked_chest(gs(pos=(40.5, 64.0, 0.5)))
        self.assertEqual(len(mcp.calls), 2)


class PowderStowReadoptionTest(unittest.IsolatedAsyncioTestCase):
    """The powder-stowed milestone must re-derive from the chest after a cold start.

    `_powder_stowed` (= `nether_expedition_complete`) is a loop latch set when a stow deposit
    succeeds; a restart forgets it and `nether_expedition_open` then marches the bot back through
    the gate for nine rods already banked in its own chest. The same database read that finds the
    banked food also sees the banked powder, so a chest holding blaze powder latches the milestone
    from the world instead of from a deposit this process never saw.
    """

    POWDER = {"response": {"data": {"result": {"query": {"rows": [
        [229, 62, 224, '[{"slot":0,"item":"blaze_powder","count":18},'
                       '{"slot":1,"item":"cooked_mutton","count":8}]']]}}}}}
    NO_POWDER = {"response": {"data": {"result": {"query": {"rows": [
        [229, 62, 224, '[{"slot":0,"item":"cooked_mutton","count":8}]']]}}}}}
    OTHER_BASE = {"response": {"data": {"result": {"query": {"rows": [
        [12, 70, 34, '[{"slot":0,"item":"blaze_powder","count":18}]']]}}}}}

    class _Read:
        def __init__(self, data):
            self.data, self.text = data, ""

    class _Mcp:
        def __init__(self, data):
            self.data = data
            self.calls: list[tuple[str, dict]] = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            return PowderStowReadoptionTest._Read(self.data)

    def _agent(self, mcp) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = (-197, -47, -98)
        agent.chest_pos = (229, 62, 224)
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent._stock_read = None
        agent._stock_warned = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._powder_cells = []
        agent.log = []
        agent.mcp = mcp
        return agent

    def _state(self):
        return gs(pos=(0.5, 64.0, 0.5), dimension="minecraft:overworld")

    async def test_powder_in_the_chest_latches_the_milestone(self) -> None:
        agent = self._agent(self._Mcp(self.POWDER))
        await agent._note_stocked_chest(self._state())
        self.assertTrue(agent._powder_stowed)
        # The cell is remembered so a withdraw after the restart looks in the right chest.
        self.assertIn((229, 62, 224), agent._powder_cells)
        self.assertEqual([r["reason"] for r in agent.log], ["powder_stow_readopted"])

    async def test_an_empty_chest_leaves_the_expedition_open(self) -> None:
        # No powder banked: the latch stays off so a genuine top-up can reopen the gate.
        agent = self._agent(self._Mcp(self.NO_POWDER))
        await agent._note_stocked_chest(self._state())
        self.assertFalse(agent._powder_stowed)
        self.assertEqual(agent._powder_cells, [])
        self.assertEqual(agent.log, [])

    async def test_powder_in_a_chest_this_base_does_not_own_is_ignored(self) -> None:
        # The database lists every container the bot ever opened, in any base or world. Only the
        # cells this base owns count, or a stale chest elsewhere would fake the milestone.
        agent = self._agent(self._Mcp(self.OTHER_BASE))
        await agent._note_stocked_chest(self._state())
        self.assertFalse(agent._powder_stowed)

    async def test_an_already_stowed_latch_is_left_alone(self) -> None:
        # Set this session by the deposit: re-deriving it is harmless and logs nothing new.
        agent = self._agent(self._Mcp(self.POWDER))
        agent._powder_stowed = True
        await agent._note_stocked_chest(self._state())
        self.assertTrue(agent._powder_stowed)
        self.assertEqual(agent.log, [])


class ContainerMissingTest(unittest.TestCase):
    """A cell the server calls "not a container" has to leave what the run remembers.

    run83 re-adopted its base from a bed a previous run left standing and took the chest
    cell banked beside it (-45,71,13) on adoption, without the cell ever being looked at;
    the world had given the chest up. Four deposits walked to the empty space, each answered
    `[CONTAINER_NOT_PRESENT] The loaded target block is air, not a chest`, and none of it
    landed: the phantom stayed the deposit's target until the run ended, every `stow_surplus`
    walked empty, and a `take_stored_food` would have paid the same walk for provisions that
    were believed to stand there. The server's word is the evidence: the cell leaves the
    network, its believed provisions leave with it, and the storage chain places a chest
    again -- the next adopted phantom dies the same first death.
    """

    MAIN = (-45, 71, 13)
    OVERFLOW = (-45, 71, 20)

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.chest_pos = ContainerMissingTest.MAIN
        agent._extra_chests = [ContainerMissingTest.OVERFLOW]
        agent._chests_full = {ContainerMissingTest.OVERFLOW}
        agent._banked_stores = ((ContainerMissingTest.MAIN, "cooked_beef", 4),)
        agent.banked = {"cooked_beef": 4}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        return agent

    def _deposit(self, cell: tuple[int, int, int]) -> Action:
        return Action("stow_surplus", "use_container",
                      {"operation": "deposit", "x": cell[0], "y": cell[1], "z": cell[2],
                       "items": []}, "deposit into the chest")

    def test_the_phantom_main_chest_leaves_the_network(self) -> None:
        agent = self._agent()
        agent._note_container_missing(self._deposit(self.MAIN), False,
                                      "[CONTAINER_NOT_PRESENT] The loaded target block is air, not a chest")
        self.assertIsNone(agent.chest_pos)
        self.assertEqual(agent._extra_chests, [ContainerMissingTest.OVERFLOW])
        self.assertEqual(agent._banked_stores, ())
        self.assertEqual(agent.banked, {})

    def test_the_overflow_cell_drops_from_full_and_keeps_the_main_one(self) -> None:
        agent = self._agent()
        agent._banked_stores = ((ContainerMissingTest.OVERFLOW, "cooked_beef", 4),)
        agent._note_container_missing(self._deposit(self.OVERFLOW), False,
                                      "[CONTAINER_NOT_PRESENT] ...")
        self.assertEqual(agent._extra_chests, [])
        self.assertEqual(agent._chests_full, set())
        self.assertEqual(agent._banked_stores, ())
        self.assertEqual(agent.banked, {})
        self.assertEqual(agent.chest_pos, ContainerMissingTest.MAIN)

    def test_a_furnace_answering_the_same_is_no_station_anymore(self) -> None:
        agent = self._agent()
        agent.furnace_pos = (-45, 71, 11)
        agent._furnace_contents = {"iron_ingot": 2}
        agent._furnace_occupied = True
        act = Action("collect_smelted", "use_container",
                     {"operation": "withdraw", "x": -45, "y": 71, "z": 11, "items": []},
                     "take the furnace output")
        agent._note_container_missing(act, False, "[CONTAINER_NOT_PRESENT] ...")
        self.assertIsNone(agent.furnace_pos)
        self.assertEqual(agent._furnace_contents, {})
        self.assertFalse(agent._furnace_occupied)

    def test_a_standing_furnace_the_server_cannot_open_stays(self) -> None:
        # run49: 119 drains answered this at a real furnace; forgetting it re-adopted the same
        # cell with the occupied flag wiped, and cook and drain refused each other in turn.
        agent = self._agent()
        agent.furnace_pos = (-45, 71, 11)
        agent._furnace_contents = {"iron_ingot": 2}
        agent._furnace_occupied = True
        act = Action("collect_smelted", "use_container",
                     {"operation": "withdraw", "x": -45, "y": 71, "z": 11, "items": []},
                     "take the furnace output")
        agent._note_container_missing(
            act, False, "[CONTAINER_NOT_PRESENT] The loaded target block is furnace, "
                        "not a supported location-owned container.")
        self.assertEqual(agent.furnace_pos, (-45, 71, 11))
        self.assertEqual(agent._furnace_contents, {"iron_ingot": 2})
        self.assertTrue(agent._furnace_occupied)

    def test_a_furnace_cell_turned_to_air_is_forgotten(self) -> None:
        agent = self._agent()
        agent.furnace_pos = (-45, 71, 11)
        agent._furnace_occupied = True
        act = Action("collect_smelted", "use_container",
                     {"operation": "inspect", "x": -45, "y": 71, "z": 11},
                     "look at the furnace")
        agent._note_container_missing(
            act, False, "[CONTAINER_NOT_PRESENT] The loaded target block is air, "
                        "not a supported location-owned container.")
        self.assertIsNone(agent.furnace_pos)
        self.assertFalse(agent._furnace_occupied)

    def test_a_chest_cell_holding_a_furnace_leaves_the_network(self) -> None:
        agent = self._agent()
        agent._note_container_missing(
            self._deposit(self.MAIN), False,
            "[CONTAINER_NOT_PRESENT] The loaded target block is furnace, "
            "not a supported location-owned container.")
        self.assertIsNone(agent.chest_pos)
        self.assertEqual(agent.banked, {})

    def test_a_chest_block_that_is_refused_stays_in_the_network(self) -> None:
        agent = self._agent()
        agent._note_container_missing(
            self._deposit(self.MAIN), False,
            "[CONTAINER_NOT_PRESENT] The loaded target block is chest, not loaded yet.")
        self.assertEqual(agent.chest_pos, ContainerMissingTest.MAIN)
        self.assertEqual(agent.banked, {"cooked_beef": 4})

    def test_a_successful_attempt_forgets_nothing(self) -> None:
        agent = self._agent()
        agent._note_container_missing(self._deposit(self.MAIN), True, "")
        self.assertEqual(agent.chest_pos, ContainerMissingTest.MAIN)

    def test_an_unrelated_cell_leaves_the_remembered_chest_alone(self) -> None:
        agent = self._agent()
        agent._note_container_missing(self._deposit((10, 64, 20)), False,
                                      "[CONTAINER_NOT_PRESENT] ...")
        self.assertEqual(agent.chest_pos, ContainerMissingTest.MAIN)
        self.assertEqual(agent._banked_stores, ((ContainerMissingTest.MAIN, "cooked_beef", 4),))
        self.assertEqual(agent.banked, {"cooked_beef": 4})


class StorageNetworkNotesTest(unittest.IsolatedAsyncioTestCase):
    """How the loop remembers what the server says about storage, chest by chest (run79: the
    `[CONTAINER_BATCH_INCOMPLETE]` deposit became a failed action, the one chest a dead end, and
    102 `wait` ticks followed -- because nothing recorded *which* cell was the one full, let alone
    that a second chest beside it would have had room)."""

    BASE = (-197, -47, -98)
    MAIN = (-197, -46, -97)
    OVERFLOW = (-197, -46, -95)
    FOREIGN = (40, 70, 40)          # a chest of some other base: the database has it, not us

    class _Action:
        def __init__(self, label: str, tool: str, args: dict | None = None) -> None:
            self.label = label
            self.tool = tool
            self.args = args or {}

    class _Read:
        def __init__(self, data):
            self.data, self.text = data, ""

    class _Mcp:
        def __init__(self, data):
            self.data = data
            self.calls: list[tuple[str, dict]] = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            return StorageNetworkNotesTest._Read(self.data)

    # Overflow chest first (the newest observation), then the main one, then a foreign base's.
    ROWS = {"response": {"data": {"result": {"query": {"rows": [
        [-197, -46, -95, '[{"item":"cooked_beef","count":12},{"item":"cobblestone","count":64}]'],
        [-197, -46, -97, '[{"item":"cooked_mutton","count":21},{"item":"rotten_flesh","count":96}]'],
        [40, 70, 40, '[{"item":"golden_carrot","count":3}]']]}}}}}

    def _agent(self, data=None) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = self.BASE
        agent.chest_pos = None
        agent.banked = {}
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent._stock_read = None
        agent._stock_warned = False
        agent._stow_blocked = False
        agent._chest_blocked = False
        agent.log = []
        # The rest of the state `_adopt_homebase` and `_inject` read, for the two tests that
        # walk through them.
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._base_adopted = False
        agent.surface = None
        agent.build_site = None
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_off = set()
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._blocked_cell = None
        agent._chest_above = None
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._portal_base_scanned = None
        agent.mcp = self._Mcp(data)
        return agent

    def _deposit(self, cell) -> _Action:
        return self._Action("stow_surplus", "use_container",
                            {"operation": "deposit", "x": cell[0], "y": cell[1], "z": cell[2]})

    def test_batch_incomplete_marks_which_chest_closed(self) -> None:
        agent = self._agent()
        agent._note_container(self._deposit(self.MAIN), False,
                              "[CONTAINER_BATCH_INCOMPLETE] Only part of the offered batch fit "
                              "the container. Observed item transfers: gunpowder 0/2, "
                              "cobblestone 0/16")
        self.assertIn(self.MAIN, agent._chests_full)

    def test_a_deposit_that_worked_opens_the_cell_again(self) -> None:
        agent = self._agent()
        agent._chests_full = {self.MAIN}
        agent._note_container(self._deposit(self.MAIN), True, "")
        self.assertEqual(agent._chests_full, set())

    def test_every_other_refusal_leaves_fullness_alone(self) -> None:
        # A deposit broken off by combat contact is not a chest that ran out of space; a
        # `withdraw` says nothing about space at all.
        agent = self._agent()
        agent._note_container(self._deposit(self.MAIN), False, "[HOSTILE_CONTACT]")
        self.assertEqual(agent._chests_full, set())
        withdraw = self._Action("take_stored_food", "use_container",
                                {"operation": "withdraw", "x": self.MAIN[0],
                                 "y": self.MAIN[1], "z": self.MAIN[2]})
        agent._note_container(withdraw, False, "[CONTAINER_BATCH_INCOMPLETE]")
        self.assertEqual(agent._chests_full, set())

    def test_only_the_full_message_blocks_the_stow_relief(self) -> None:
        # The old line blocked the deposit relief on any failure, combat included -- and the run
        # dropped what a chest standing right there could have stored.
        act = self._Action("stow_surplus", "navigate", {})
        agent = self._agent()
        agent._note_stow_blocked(act, False, "[HOSTILE_CONTACT]")
        self.assertFalse(agent._stow_blocked)
        agent._note_stow_blocked(act, False, "[CONTAINER_BATCH_INCOMPLETE]")
        self.assertTrue(agent._stow_blocked)
        deposit = self._deposit(self.MAIN)
        agent._note_stow_blocked(deposit, True, "")
        self.assertFalse(agent._stow_blocked)

    def test_an_overflow_chest_grows_the_network_not_the_main_cell(self) -> None:
        agent = self._agent()
        agent.chest_pos = self.MAIN
        agent._chests_full = {self.MAIN}
        place = self._Action("place_chest", "place_block",
                             {"block_name": "chest", "x": self.OVERFLOW[0],
                              "y": self.OVERFLOW[1], "z": self.OVERFLOW[2]})
        agent._note_placed(place)
        self.assertEqual(agent.chest_pos, self.MAIN)          # the main chest keeps its cell
        self.assertEqual(agent._extra_chests, [self.OVERFLOW])
        self.assertNotIn(self.OVERFLOW, agent._chests_full)   # a fresh chest is roomy by definition

    def test_the_first_chest_still_sets_the_main_cell(self) -> None:
        agent = self._agent()
        place = self._Action("place_chest", "place_block",
                             {"block_name": "chest", "x": self.MAIN[0],
                              "y": self.MAIN[1], "z": self.MAIN[2]})
        agent._note_placed(place)
        self.assertEqual(agent.chest_pos, self.MAIN)
        self.assertEqual(agent._extra_chests, [])

    def test_a_probe_that_walked_up_adopts_no_cell(self) -> None:
        # Live, run85: the heightless `surface_up` probe (`navigate` with `x` and `z` only)
        # answered not with `[NAVIGATION_HEIGHT_REQUIRED]` but with a walk -- the first column
        # the server could answer unambiguously -- and `_note_placed`, whose guard checked
        # only `"x"`, raised `KeyError: 'y'` in `verify` and broke the whole run at it31.
        # What names no height names no cell: the success passes by untouched.
        agent = self._agent()
        probe = self._Action("surface_up", "navigate", {"x": -36, "z": 13})
        agent._note_placed(probe)
        self.assertIsNone(agent.chest_pos)
        self.assertEqual(agent.base, self.BASE)
        self.assertEqual(agent._extra_chests, [])

    def test_the_grounds_refusal_at_an_overflow_cell_keeps_the_chain(self) -> None:
        # One candidate is refused: the next is picked. Only the main cell's refusal gives the
        # chest chain up (`_chest_blocked`), the old verdict on the base ground.
        agent = self._agent()
        agent.chest_pos = self.MAIN
        refused = self._Action("place_chest", "place_block",
                               {"block_name": "chest", "x": self.OVERFLOW[0],
                                "y": self.OVERFLOW[1], "z": self.OVERFLOW[2]})
        agent._note_chest(refused, False, "[PLACE_NO_SUPPORT] No adjacent solid block")
        self.assertFalse(agent._chest_blocked)
        self.assertEqual(agent._overflow_no_support, {self.OVERFLOW})

    def test_the_grounds_refusal_at_the_main_cell_ends_the_chain(self) -> None:
        agent = self._agent()
        refused = self._Action("place_chest", "place_block",
                               {"block_name": "chest", "x": self.MAIN[0],
                                "y": self.MAIN[1], "z": self.MAIN[2]})
        agent._note_chest(refused, False, "[PLACE_NO_SUPPORT] No adjacent solid block")
        self.assertTrue(agent._chest_blocked)
        self.assertEqual(agent._overflow_no_support, set())

    async def test_the_network_read_keeps_only_its_own_cells(self) -> None:
        # The database lists every chest this bot ever opened -- including the one a *foreign*
        # base left. What the loop does not own, it cannot bank, and would not know where to
        # walk to fetch.
        agent = self._agent(self.ROWS)
        agent.chest_pos = self.MAIN
        agent._extra_chests = [self.OVERFLOW]
        await agent._note_stocked_chest(gs(pos=(0.5, 64.0, 0.5)))
        self.assertEqual(agent.banked,
                         {"cooked_beef": 12, "cooked_mutton": 21, "rotten_flesh": 96})
        self.assertEqual(agent._banked_stores,
                         ((self.OVERFLOW, "cooked_beef", 12),
                          (self.MAIN, "cooked_mutton", 21), (self.MAIN, "rotten_flesh", 96)))

    async def test_an_unadopted_read_leaves_the_bank_empty_rather_than_guessing(self) -> None:
        # Fresh restart, network unknown: rows exist but no cell is owned -- the aggregate stays
        # empty instead of adopting some other base's chest into the bank.
        agent = self._agent(self.ROWS)
        await agent._note_stocked_chest(gs(pos=(0.5, 64.0, 0.5)))
        self.assertEqual(agent.banked, {})
        self.assertEqual(agent._banked_stores, ())

    def test_a_restart_adopts_the_overflow_chest_next_to_the_main_one(self) -> None:
        # The chest scan sees every chest near the base. Left alone, the network's overflow
        # would go blind after a restart: its provisions stay in the server's database behind a
        # cell the loop no longer owns.
        agent = self._agent()
        agent.chest_pos = self.MAIN
        g = gs(resources={"chest": [BlockHit("chest", "chest", self.MAIN[0], self.MAIN[1],
                                             self.MAIN[2], 1.0),
                                    BlockHit("chest", "chest", self.OVERFLOW[0], self.OVERFLOW[1],
                                             self.OVERFLOW[2], 5.0)]})
        agent._adopt_homebase(g)
        self.assertEqual(agent.chest_pos, self.MAIN)
        self.assertEqual(agent._extra_chests, [self.OVERFLOW])

    def test_inject_carries_the_network_into_the_state(self) -> None:
        agent = self._agent()
        agent.chest_pos = self.MAIN
        agent._extra_chests = [self.OVERFLOW]
        agent._chests_full = {self.MAIN}
        agent._overflow_no_support = {self.OVERFLOW}
        agent._banked_stores = ((self.MAIN, "cooked_mutton", 21),)
        g = agent._inject(gs())
        self.assertEqual(g.extra_chests, (self.OVERFLOW,))
        self.assertEqual(g.chests_full, frozenset({self.MAIN}))
        self.assertEqual(g.overflow_skip, frozenset({self.OVERFLOW}))
        self.assertEqual(g.banked_stores, ((self.MAIN, "cooked_mutton", 21),))


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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
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

    async def test_a_scan_that_found_nothing_does_not_lock_out_the_next_one(self) -> None:
        # Live, run59. The throttle is measured from the cell the bot last looked from, so a
        # *failed* look used to burn that cell - and with it every look from the eight cells
        # around it. The run started with no base, so it walked home, retired its guess, and then
        # `_adopt_portal_base` declined to look again because it had already looked four metres
        # away. The lit gate went unnamed and a second base was raised beside it.
        agent = self._agent(self._NONE)
        await agent._adopt_portal_base(self._geared())
        self.assertIsNone(agent.base)
        self.assertIsNone(agent._portal_base_scanned)     # a failed look is not a look

        # Same cell, gate now in view: the base is found.
        agent.mcp = self._Mcp(self._FIND)
        await agent._adopt_portal_base(self._geared())
        self.assertEqual(agent.base, (-156, 62, 257))

    async def test_the_throttle_still_holds_after_a_scan_that_did_find_the_gate(self) -> None:
        # The other half: the guard is not simply gone. A cell that was read successfully is not
        # re-read while the bot stands on it - and it is read again as soon as it walks off.
        agent = self._agent(self._FIND)
        near = self._geared()
        await agent._adopt_portal_base(near)
        self.assertEqual(agent.base, (-156, 62, 257))

        agent.base = None                     # the anchor retired the guess
        agent.mcp.calls.clear()
        await agent._adopt_portal_base(near)  # same cell, within 8
        self.assertEqual(agent.mcp.calls, [])

        far = gs(pos=(-178.5, 68.0, 341.5),         # 12 blocks on, same kit
                 tools={"pickaxe": "diamond", "sword": "diamond"},
                 inventory={"crafting_table": 1},
                 armor={slot: {"item": f"diamond_{slot}"}
                        for slot in ("helmet", "chestplate", "leggings", "boots")},
                 offhand="shield")
        await agent._adopt_portal_base(far)
        self.assertEqual(len(agent.mcp.calls), 1)
        self.assertEqual(agent.base, (-156, 62, 257))

    async def test_skips_once_a_base_is_set(self) -> None:
        agent = self._agent(self._FIND)
        agent.base = (-1, 2, 3)
        await agent._adopt_portal_base(self._geared())
        self.assertEqual(agent.base, (-1, 2, 3))
        self.assertEqual(agent.mcp.calls, [])

    async def test_skips_before_the_bot_can_light_a_gate(self) -> None:
        # No flint and steel, no diamond kit: nothing in this run has ever built a gate, so a lit
        # portal found now belongs to someone else and adopting it would move the base off a
        # half-built site of our own.
        agent = self._agent(self._FIND)
        await agent._adopt_portal_base(gs(pos=(0.0, 64.0, 0.0)))
        self.assertIsNone(agent.base)
        self.assertEqual(agent.mcp.calls, [])

    async def test_adopts_with_the_flint_and_steel_alone(self) -> None:
        # The case that lost run43 its base: dead in the Nether, no diamond kit, but the flint and
        # steel still in the pack - the gate and the chest a previous run built were standing.
        agent = self._agent(self._FIND)
        await agent._adopt_portal_base(gs(pos=(-178.5, 68.0, 329.5),
                                         inventory={"flint_and_steel": 1}))
        self.assertEqual(agent.base, (-156, 62, 257))
        self.assertTrue(agent._base_adopted)

    async def test_the_chest_beside_the_adopted_gate_comes_back_too(self) -> None:
        # The whole point of re-anchoring: the goal stows the blaze powder in the chest at the
        # portal, so a base adopted from the gate has to bring the chest and the furnace with it -
        # live, run43 died in the Nether and would have re-based on open ground, chest included.
        agent = self._agent(self._FIND)
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._furnace_blocked = False
        await agent._adopt_portal_base(gs(pos=(-178.5, 68.0, 329.5),
                                         inventory={"flint_and_steel": 1}))
        g = gs(pos=(-178.5, 68.0, 329.5), inventory={"flint_and_steel": 1},
               resources={"chest": [BlockHit("chest", "container", -154, 63, 258, 30.0)],
                          "furnace": [BlockHit("furnace", "container", -155, 63, 258, 30.0)]})
        agent._adopt_homebase(g)
        self.assertEqual(agent.chest_pos, (-154, 63, 258))
        self.assertEqual(agent.furnace_pos, (-155, 63, 258))

    def test_inject_hands_the_adopted_flag_to_the_state(self) -> None:
        agent = self._agent(self._FIND)
        agent._base_adopted = True
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
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
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
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


class SearchIsPaidForInMetresTest(unittest.IsolatedAsyncioTestCase):
    """A search budget has to be spent on searching.

    `_nether_hopeless` counted decisions - twenty-four of them - and a decision is not search.
    Live, run70: nineteen of the twenty-four ticks that ended the hunt were `body owner:
    hostile_reflex`. The bot stood at 108 m from its own portal and was shot at, and the budget
    was spent on being shot at. The hunt ended with zero blaze rods and no fortress in sight, on
    a trip that had barely begun.

    The unit is now ground covered, `nether_travel`, accumulated since the crossing and reset with
    it. `NETHER_HOPELESS_TICKS` stays as the backstop, because a bot the reflex has pinned covers
    no ground at all and something still has to end the trip.
    """

    async def test_standing_still_never_spends_the_budget(self) -> None:
        agent = self._agent()
        g = gs(pos=(40.5, 70.0, -9.5), dimension="minecraft:the_nether")
        agent._note_dimension(g)
        for _ in range(60):
            agent._note_dimension(g)          # the same cell: pinned by the reflex, no ground
        self.assertEqual(agent._nether_travel, 0.0)
        self.assertEqual(agent._nether_ticks, 61)

    async def test_walking_accumulates_ground_covered(self) -> None:
        agent = self._agent()
        agent._note_dimension(gs(pos=(40.5, 70.0, -9.5), dimension="minecraft:the_nether"))
        agent._note_dimension(gs(pos=(60.5, 70.0, -9.5), dimension="minecraft:the_nether"))
        agent._note_dimension(gs(pos=(80.5, 70.0, -9.5), dimension="minecraft:the_nether"))
        self.assertAlmostEqual(agent._nether_travel, 40.0)

    async def test_the_meter_counter_is_in_the_logged_state(self) -> None:
        # Without this the whole thing is unfalsifiable: there would be no way to say afterwards
        # whether the bot searched or stood still. The same rule as the furnace facts.
        from mlaya.perception import GameState
        from mlaya.actions import _nether_hopeless, NETHER_SEARCH_M
        g = GameState(nether_travel=NETHER_SEARCH_M + 1)
        rebuilt = GameState.from_state(g.as_state())
        self.assertEqual(rebuilt.nether_travel, NETHER_SEARCH_M + 1)
        self.assertTrue(_nether_hopeless(rebuilt))
        g.nether_travel = 0.0
        self.assertFalse(_nether_hopeless(GameState.from_state(g.as_state())))

    async def test_the_counter_resets_on_the_way_home(self) -> None:
        agent = self._agent()
        agent._nether_travel = 300.0
        agent._nether_last_pos = (10.0, 10.0)
        agent._note_dimension(gs(pos=(1.0, 64.0, 1.0), dimension="minecraft:overworld"))
        self.assertEqual(agent._nether_travel, 0.0)
        self.assertIsNone(agent._nether_last_pos)

    def _agent(self):
        from mlaya.graph import Agent
        from mlaya.config import Settings
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        return agent


class NetherCrossingTest(unittest.TestCase):
    """The loop side of the Nether trip: the intended crossing, and the trip's goal.

    Two things cannot live in `actions` alone. The `_in_portal` rescue has to stand down for a
    crossing the loop actually chose, and the powder that reaches the chest has to be remembered
    as a fact of the run rather than re-read out of a chest every tick.
    """

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._nether_ticks = 0
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._water_ticks = 0
        agent._defer_quiet = 0
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
        # crossing itself is only observed afterwards as a dimension change - overworld first,
        # Nether after: that is what a crossing looks like from inside a run.
        agent = self._agent()
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        g = gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether")
        agent._note_dimension(g)
        self.assertTrue(agent._nether_entered)
        self.assertEqual(agent._nether_arrival, (12, 65, -4))
        # Standing still down there must not walk the arrival cell along with the bot.
        agent._note_dimension(gs(pos=(40.5, 70.0, -9.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._nether_arrival, (12, 65, -4))

    def test_a_run_born_in_the_nether_records_no_arrival(self) -> None:
        # A first tick reading Nether saw no crossing, and the cell it stands says nothing about
        # a portal: it is wherever the last run left the bot. Live, run74: the loop recorded
        # (-469,57,421) as the "arrival" - bare basalt with no obsidian anywhere near - walked
        # the bot back to it from the fortress, and stood it there for `[PORTAL_TARGET] ...
        # observed air` until the bot died. The cold start's way home is the server's memory
        # (`_note_nether_gate`), not the start position.
        agent = self._agent()
        agent._note_dimension(gs(pos=(-469.2, 57.0, 421.5), dimension="minecraft:the_nether"))
        self.assertTrue(agent._nether_entered)              # down there, that much is real
        self.assertIsNone(agent._nether_arrival)            # but no crossing was ever seen
        for _ in range(3):
            agent._note_dimension(gs(pos=(-300.5, 63.0, 513.5), dimension="minecraft:the_nether"))
        self.assertIsNone(agent._nether_arrival)

    def test_the_next_crossing_records_its_own_arrival(self) -> None:
        # The second trip matters too: a run that comes up in the Nether, walks its bot home
        # and goes back down must know the gate it comes through now, not the cold start's
        # nothing. The arrival follows every crossing the run watches over, ever since it
        # stopped being the first Nether position seen.
        agent = self._agent()
        agent._note_dimension(gs(pos=(-469.2, 57.0, 421.5), dimension="minecraft:the_nether"))
        self.assertIsNone(agent._nether_arrival)
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._nether_arrival, (12, 65, -4))

    def test_a_crossing_after_the_haul_counts_a_pearl_trip(self) -> None:
        # The pearl waves are bounded by trips, not by time in the Nether. The rod trip is not
        # one of them - the powder is not banked yet - and every crossing made after it is.
        agent = self._agent()
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._pearl_wave_trips, 0)
        agent._powder_stowed = True
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._pearl_wave_trips, 1)
        # Roaming down there is not another trip; only the crossing is.
        for _ in range(5):
            agent._note_dimension(gs(pos=(30.5, 63.0, 10.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._pearl_wave_trips, 1)
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._pearl_wave_trips, 2)
        # The count reaches the state the planner reads, and survives the round trip through
        # the serialization the run log writes.
        self.assertEqual(agent._inject(gs()).pearl_wave_trips, 2)
        self.assertEqual(GameState.from_state(agent._inject(gs()).as_state()).pearl_wave_trips, 2)

    def test_a_powder_top_up_is_not_a_pearl_trip(self) -> None:
        # The top-up trip reopens the gate for blaze rods, not pearls; charging it against the
        # pearl cap would hand the hunt to the Overworld roam before it was ever tried.
        agent = self._agent()
        agent._powder_stowed = True
        agent._powder_exhausted = True
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._pearl_wave_trips, 0)
        # A fresh stow clears the exhaustion, and the next crossing is a pearl trip again.
        agent._powder_exhausted = False
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._pearl_wave_trips, 1)

    def test_the_nether_tick_count_resets_at_home(self) -> None:
        # `_nether_hopeless` counts decisions spent down there, so a bot that comes home for any
        # reason has to start the next trip from zero rather than leave on the first tick.
        agent = self._agent()
        for _ in range(5):
            agent._note_dimension(gs(pos=(12.5, 65.0, -3.5), dimension="minecraft:the_nether"))
        self.assertEqual(agent._nether_ticks, 5)
        agent._note_dimension(gs(pos=(226.5, 64.0, 222.5), dimension="minecraft:overworld"))
        self.assertEqual(agent._nether_ticks, 0)

    def test_the_powder_reaching_the_chest_closes_the_goal(self) -> None:
        agent = self._agent()
        deposit = Action("stow_blaze_powder", "use_container",
                         {"operation": "deposit", "x": -196, "y": -46, "z": -98}, "")
        agent._note_stow(deposit, True, {"inventory_delta": {"blaze_powder": -18}})
        self.assertTrue(agent._powder_stowed)
        self.assertTrue(agent._inject(gs()).powder_stowed)

    def test_a_barter_counts_toward_the_cap(self) -> None:
        # The gold reserve is dug only while the barter cap is not reached: a piglin rolls once
        # per ingot and pearls are the rare entry of that roll, so an uncapped route turns the
        # lull into a gold farm. A call the server refused is not an attempt.
        agent = self._agent()
        trade = Action("barter_pearls", "barter",
                       {"piglin_id": 412, "item_name": "ender_pearl", "count": 8,
                        "gold_budget": 8}, "")
        agent._note_barter(trade, True)
        agent._note_barter(trade, False)
        self.assertEqual(agent._inject(gs()).barter_attempts, 1)

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

    def test_a_stow_remembers_the_cell_it_filled(self) -> None:
        # `stow_blaze_powder` deposits into the network's roomy chest, not necessarily
        # `chest_pos`; the stronghold withdraw has to look in the same cell, so the loop
        # books it when the deposit lands.
        agent = self._agent()
        deposit = Action("stow_blaze_powder", "use_container",
                         {"operation": "deposit", "x": -190, "y": -46, "z": -100}, "")
        agent._note_stow(deposit, True, {"inventory_delta": {"blaze_powder": -18}})
        self.assertEqual(agent._powder_cells, [(-190, -46, -100)])
        self.assertEqual(agent._inject(gs()).powder_cells, ((-190, -46, -100),))

    def test_one_empty_lid_does_not_empty_the_bank(self) -> None:
        # A withdraw that comes up empty at the main chest says nothing about the overflow
        # chest the powder actually went into; the top-up only latches once every remembered
        # cell has come up empty.
        agent = self._agent()
        main = Action("stow_blaze_powder", "use_container",
                      {"operation": "deposit", "x": -196, "y": -46, "z": -98}, "")
        agent._note_stow(main, True, {"inventory_delta": {"blaze_powder": -18}})
        agent._powder_cells.append((-190, -46, -100))
        empty = Action("take_blaze_powder", "use_container",
                       {"operation": "withdraw", "x": -196, "y": -46, "z": -98,
                        "items": [{"item_name": "blaze_powder", "count": 16}]}, "")
        agent._note_take_powder(empty, False, "[NO_MATCHING_ITEMS]", {})
        self.assertFalse(agent._powder_exhausted)
        self.assertEqual(agent._powder_cells, [(-190, -46, -100)])
        # The overflow lid comes up empty too: now the bank is out and the top-up opens.
        other = Action("take_blaze_powder", "use_container",
                       {"operation": "withdraw", "x": -190, "y": -46, "z": -100,
                        "items": [{"item_name": "blaze_powder", "count": 16}]}, "")
        agent._note_take_powder(other, False, "[NO_MATCHING_ITEMS]", {})
        self.assertTrue(agent._powder_exhausted)
        self.assertEqual(agent._powder_cells, [])

    def test_a_withdraw_that_brought_powder_keeps_the_cell(self) -> None:
        # The chest may still hold more than the withdraw asked for; the cell stays in play
        # until a visit to it comes up empty.
        agent = self._agent()
        deposit = Action("stow_blaze_powder", "use_container",
                         {"operation": "deposit", "x": -196, "y": -46, "z": -98}, "")
        agent._note_stow(deposit, True, {"inventory_delta": {"blaze_powder": -18}})
        take = Action("take_blaze_powder", "use_container",
                      {"operation": "withdraw", "x": -196, "y": -46, "z": -98,
                       "items": [{"item_name": "blaze_powder", "count": 16}]}, "")
        agent._note_take_powder(take, True, "", {"inventory_delta": {"blaze_powder": 16}})
        self.assertEqual(agent._powder_cells, [(-196, -46, -98)])
        self.assertFalse(agent._powder_exhausted)

    def test_a_fresh_stow_clears_the_exhaustion(self) -> None:
        # The top-up ends where it began: the second haul parks powder in the chest again, the
        # latch reopens the eye-crafting, and the cell the deposit reached is remembered.
        agent = self._agent()
        agent._powder_exhausted = True
        deposit = Action("stow_blaze_powder", "use_container",
                         {"operation": "deposit", "x": -190, "y": -46, "z": -100}, "")
        agent._note_stow(deposit, True, {"inventory_delta": {"blaze_powder": -18}})
        self.assertFalse(agent._powder_exhausted)
        self.assertEqual(agent._powder_cells, [(-190, -46, -100)])

    def test_the_stowed_expedition_alone_keeps_the_run_going(self) -> None:
        # `powder_stowed` is a latch that only ever sets, while `tech_tree_complete` reads the
        # live world - a creepers' bed or a dropped tool turns it false again. So the run carries
        # on past the deposit until the home base is in order too, and stops on the first tick
        # where both hold. Leaving the world tidy is part of finishing a playthrough.
        agent = self._agent()
        agent.settings.loop.stop_when_complete = True
        agent._powder_stowed = True
        self.assertFalse(agent._goal_reached(agent._inject(gs())))


class RememberedGateTest(unittest.IsolatedAsyncioTestCase):
    """A cold Nether run takes the way home from the server's own memory.

    A run that starts with the bot in the Nether saw no crossing: the one that put the bot down
    there happened while another process was running, and the graph resets with every restart.
    Live, run74: the loop acted as if it knew anyway - it recorded the cell the bot happened to
    stand on at the start as its "arrival", walked 250 blocks to that non-portal, and waited at
    bare basalt for a gate that had never stood there, until the bot died. The server does hold
    the real fact: every crossing gets a `player_dimension_change` event with the position it
    delivered the bot at, kept in the database that outlives the death, the dimension change and
    the restart (`perception.REMEMBERED_GATE_SQL`). The same shape as the remembered chest, for
    the one way out.
    """

    class _Read:
        def __init__(self, data):
            self.data = data
            self.text = None

    class _Mcp:
        def __init__(self, rows):
            self.rows = rows
            self.calls: list = []

        async def read(self, tool, args, why):
            self.calls.append((tool, args))
            payload = {"response": {"data": {"result": {"query": {
                "columns": ["x", "y", "z"], "rows": self.rows}}}}}
            return RememberedGateTest._Read(payload)

    ROWS = [[33.42975, 55, 34.5]]   # the live crossing into the (33,55,34) gate

    async def test_a_cold_nether_run_reads_the_gate_once(self) -> None:
        agent = self._agent()
        mcp = self._Mcp(self.ROWS)
        agent.mcp = mcp
        await agent._note_nether_gate(gs(pos=(12.5, 65.0, -3.5),
                                         dimension="minecraft:the_nether"))
        self.assertEqual(agent._nether_gate, (33, 55, 34))
        self.assertEqual(mcp.calls[0][0], "query_bot_data")
        sql = mcp.calls[0][1]["sql"]
        self.assertIn("player_dimension_change", sql)
        self.assertIn("'the_nether'", sql)
        # One read of the memory per run, not one per tick down there.
        await agent._note_nether_gate(gs(pos=(13.5, 65.0, -3.5),
                                         dimension="minecraft:the_nether"))
        self.assertEqual(len(mcp.calls), 1)

    async def test_the_overworld_owes_no_gate_read(self) -> None:
        # The way out is used in the Nether; a tick at home must not spend the read.
        agent = self._agent()
        agent.mcp = self._Mcp(self.ROWS)
        await agent._note_nether_gate(gs(pos=(226.5, 64.0, 222.5),
                                         dimension="minecraft:overworld"))
        self.assertEqual(agent.mcp.calls, [])
        self.assertIsNone(agent._nether_gate)

    async def test_a_world_that_never_crossed_has_no_gate(self) -> None:
        agent = self._agent()
        agent.mcp = self._Mcp([])
        await agent._note_nether_gate(gs(pos=(12.5, 65.0, -3.5),
                                         dimension="minecraft:the_nether"))
        self.assertIsNone(agent._nether_gate)

    async def test_a_failed_read_leaves_the_run_its_own_sources(self) -> None:
        # Nothing is owed a gate: a scan the bot walks into, and the arrival of a crossing the
        # run watched, are still its own. A database that will not answer must not strand the
        # loop, and must not cost it another read on every tick either.
        class _Boom:
            async def read(self, *a):
                raise RuntimeError("database gone")
        agent = self._agent()
        agent.mcp = _Boom()
        await agent._note_nether_gate(gs(pos=(12.5, 65.0, -3.5),
                                         dimension="minecraft:the_nether"))
        self.assertIsNone(agent._nether_gate)
        self.assertTrue(agent._gate_read)   # read once means once, even into the failure

    async def test_the_gate_is_injected_into_the_state(self) -> None:
        # A fact the candidates turn on has to be in the state the candidates read, and in the
        # log afterwards: run74's wrong "arrival" was unanswerable from the recorded ticks.
        agent = self._agent()
        agent._nether_gate = (33, 55, 34)
        self.assertEqual(agent._inject(gs()).nether_gate, (33, 55, 34))
        rebuilt = GameState.from_state(agent._inject(gs()).as_state())
        self.assertEqual(rebuilt.nether_gate, (33, 55, 34))

    def _agent(self, **over):
        agent = Agent.__new__(Agent)  # bypass __init__: no Laya, no MCP session
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent.surface = None
        agent.chest_pos = None
        agent._extra_chests = []
        agent._chests_full = set()
        agent._overflow_no_support = set()
        agent._banked_stores = ()
        agent.banked = {}
        agent.furnace_pos = None
        agent.smoker_pos = None
        agent.build_site = None
        agent._base_adopted = False
        agent._pour_skip = set()
        agent._stance_skip = set()
        agent._rescue_skip = set()
        agent._stuck_code = {}
        agent._stuck_reps = {}
        agent._stuck_off = {}
        agent._bed_base = None
        agent._bed_base_scanned = None
        agent._recover_written_off = set()
        agent._recover_unreachable = (None, 0)
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        agent._drop_hole_blocked = False
        agent._drop_open_pending = False
        agent._defer_quiet = 0
        agent._drop_pos = None
        agent._held_ticks = 0
        agent._drop_heading = None
        agent._heading = None
        agent._chest_blocked = False
        agent._furnace_blocked = False
        agent._furnace_contents = {}
        agent._furnace_occupied = False
        agent._blocked_cell = None
        agent._chest_above = None
        agent._portal_blocked = []
        agent._frame_complete = False
        agent._refused_frame = set()
        agent._frame_names = {}
        agent._enter_intent_ticks = 0
        agent._nether_entered = False
        agent._nether_travel = 0.0
        agent._nether_last_pos = None
        agent._nether_ticks = 0
        agent._nether_arrival = None
        agent._last_dimension = None
        agent._nether_gate = None
        agent._gate_read = False
        agent._powder_stowed = False
        agent._barter_attempts = 0
        agent._end_portal_active = False
        agent._stronghold_estimated = False
        agent._powder_exhausted = False
        agent._powder_cells = []
        agent._water_ticks = 0
        agent._defer_quiet = 0
        agent._home_anchor = None
        agent._home_ticks = 0
        agent.log = []
        agent.mcp = None
        for k, v in over.items():
            setattr(agent, k, v)
        return agent


class StoppedBuildFrameMarkingTest(unittest.TestCase):
    """A `[BUILD_STOPPED]` that names no cell marks what the frame scan already knows.

    Live, run80: the answer "The build stopped: 2 passes changed nothing; last route
    completed." carries no cell, so `parse_refused_cells` remembered nothing, `clear_frame`
    was never offered and 350 `clear_portal` route digs changed nothing either -- the frame
    closed only at it364, too late in the day. The blocked opening cells and the frame ring
    cells the plane read has named a wrong block stand in for the missing answer:
    `clear_frame` digs each with the explicit `collect_block` the movement policy allows.
    """

    _STOPPED = ("[BUILD_STOPPED] The build stopped: 2 passes changed nothing; "
                "last route completed.")

    class _Action:
        def __init__(self, label: str, **args) -> None:
            self.label = label
            self.args = args

    @staticmethod
    def _agent(**over) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.base = None
        agent._portal_blocked = []
        agent._frame_names = {}
        agent._refused_frame = set()
        for k, v in over.items():
            setattr(agent, k, v)
        return agent

    def test_a_stopped_build_marks_the_blocked_opening(self) -> None:
        # The cells the scan flagged in the opening are marked for the explicit dig, with
        # nothing else: no base, no plane read, so no ring can be judged.
        agent = self._agent(base=(-45, 70, 10),
                           _portal_blocked=[(-43, 71, 10), (-42, 71, 10)])
        agent._note_frame_cells(self._Action("clear_portal"), False, self._STOPPED)
        self.assertEqual(agent._refused_frame, {(-43, 71, 10), (-42, 71, 10)})

    def test_a_stopped_build_marks_the_wrong_blocks_in_the_frame_ring(self) -> None:
        # run80's blockers were the frame's own cells: a tree in a side obsidian cell,
        # terrain where a cobblestone corner belongs. The plane read names each cell and the
        # ring says which block belongs where: cells already holding their right block, and
        # cells the read has not named, wait.
        agent = self._agent(base=(-45, 70, 10))
        cells = perception.frame_obsidian_cells(agent.base)
        names = {cell: "obsidian" for cell in cells}
        names[cells[2]] = "spruce_log"            # side cell holding a tree
        names[(-44, 73, 10)] = "dirt"             # side cell holding terrain
        names[(-44, 70, 10)] = "cobblestone"      # corner already right
        names[(-41, 70, 10)] = "dirt"             # corner still terrain
        agent._frame_names = names
        agent._note_frame_cells(self._Action("build_portal"), False, self._STOPPED)
        self.assertEqual(agent._refused_frame, {cells[2], (-44, 73, 10), (-41, 70, 10)})

    def test_a_stopped_build_without_a_base_or_scan_marks_nothing(self) -> None:
        agent = self._agent()
        agent._note_frame_cells(self._Action("clear_portal"), False, self._STOPPED)
        self.assertEqual(agent._refused_frame, set())

    def test_the_opening_marking_survives_a_base_without_a_scan(self) -> None:
        # No plane read yet: the blocked opening still stands, the ring simply cannot be
        # judged -- no cell is invented from thin air.
        agent = self._agent(base=(-45, 70, 10), _portal_blocked=[(-43, 72, 10)])
        agent._note_frame_cells(self._Action("clear_portal"), False, self._STOPPED)
        self.assertEqual(agent._refused_frame, {(-43, 72, 10)})

    def test_a_finished_build_clears_the_marked_cells(self) -> None:
        # Same as the refused container before it: the build closing the frame clears the
        # whole list, and one dug cell drops just itself.
        agent = self._agent(base=(-45, 70, 10), _portal_blocked=[(-43, 71, 10)])
        agent._note_frame_cells(self._Action("clear_portal"), False, self._STOPPED)
        agent._note_frame_cells(self._Action("clear_frame", x=-43, y=71, z=10), True, "")
        self.assertEqual(agent._refused_frame, set())
        agent._note_frame_cells(self._Action("build_portal"), True, "")
        self.assertEqual(agent._refused_frame, set())


class BodyHeldStopTest(unittest.TestCase):
    """A body the server's reflex will not hand back ends the run with a reason.

    run89: `hostile_reflex` held the body in `fight`/`hold` against a zombie 1.6 m off for nearly
    three hours, full health, no event. Each tick sat 31 and then 93 minutes in `wait_for_action`
    and the run neither moved nor ended. run95 spent three *pairs* of five-minute waits on one
    held action id each (it268/269, it279/280, it354/355) and never saw a third in a row.
    """

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent.settings = Settings()
        agent.settings.loop.max_iterations = 600
        agent.settings.loop.stop_when_complete = True
        agent._held_ticks = 0
        return agent

    def test_two_dead_held_calls_in_a_row_stop_the_run(self) -> None:
        agent = self._agent()
        held = SimpleNamespace(status="held")
        # run89's standoff: full health, no drop, the same lone zombie off the feet - a hold that
        # moves nothing on either side, which is the only kind that counts toward the stop.
        flat = dict(health=20.0, inventory={}, hostiles=[("zombie", 1, 1.6)])
        self.assertEqual(BODY_HELD_STOP, 2)
        for _ in range(BODY_HELD_STOP - 1):
            agent._note_held(held, gs(**flat), gs(**flat))
            self.assertFalse(agent._stop_state(gs(), 10)["done"])
        agent._note_held(held, gs(**flat), gs(**flat))
        self.assertEqual(agent._stop_state(gs(), 10), {"done": True, "stop": "body_held"})

    def test_a_result_that_came_back_starts_the_count_over(self) -> None:
        agent = self._agent()
        flat = dict(health=20.0, inventory={}, hostiles=[("zombie", 1, 1.6)])
        agent._note_held(SimpleNamespace(status="held"), gs(**flat), gs(**flat))
        agent._note_held(SimpleNamespace(status="held"), gs(**flat), gs(**flat))
        agent._note_held(SimpleNamespace(status="succeeded"), gs(**flat), gs(**flat))
        agent._note_held(SimpleNamespace(status="held"), gs(**flat), gs(**flat))
        self.assertFalse(agent._stop_state(gs(), 10)["done"])

    def test_a_winning_hold_does_not_count_toward_the_stop(self) -> None:
        # Live, run106: the held ticks at the blaze spawner ran HP 7->13->20 while the fortress
        # was being cleared, yet two of them ended the run on `body_held`. A hold that heals the
        # bot is the reflex winning the ground it holds, not run89's dead standoff, so it breaks
        # the streak rather than counting - two healing holds in a row never stop the run.
        agent = self._agent()
        held = SimpleNamespace(status="held")
        crowd = [("blaze", 11, 5.0)]
        agent._note_held(held, gs(health=7.0, inventory={}, hostiles=crowd),
                               gs(health=13.0, inventory={}, hostiles=crowd))
        self.assertEqual(agent._held_ticks, 0)
        agent._note_held(held, gs(health=13.0, inventory={}, hostiles=crowd),
                               gs(health=20.0, inventory={}, hostiles=crowd))
        self.assertFalse(agent._stop_state(gs(), 10)["done"])

    def test_a_rod_or_a_thinned_pack_is_progress_too(self) -> None:
        agent = self._agent()
        held = SimpleNamespace(status="held")
        crowd = [("blaze", 11, 5.0)]
        # A rod came in, health flat at full: the haul is still growing.
        agent._note_held(held, gs(health=20.0, inventory={"blaze_rod": 10}, hostiles=crowd),
                               gs(health=20.0, inventory={"blaze_rod": 11}, hostiles=crowd))
        self.assertEqual(agent._held_ticks, 0)
        # The pack thinned by two, no new rod yet: a kill landed.
        agent._note_held(held, gs(health=20.0, inventory={"blaze_rod": 11}, hostiles=crowd),
                               gs(health=20.0, inventory={"blaze_rod": 11}, hostiles=[("blaze", 9, 5.0)]))
        self.assertEqual(agent._held_ticks, 0)


class FleeStageTest(unittest.IsolatedAsyncioTestCase):
    """`leave_nether` crosses home under the flee stage, so the reflex stops fighting whatever is
    in reach and the walk is not held behind an endless spawner fight.

    Live, run106: two `leave_nether` ticks came back `[BODY_HELD]` 300 s each at a blaze spawner
    (owner hostile_reflex), and the run ended on `body_held` with twelve rods it never carried out.
    """

    class _Action:
        def __init__(self, label: str, tool: str, args: dict | None = None) -> None:
            self.label = label
            self.tool = tool
            self.args = args or {}
            self.rationale = "test"

    def _agent(self, order: list) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent.log = []

        class _MCP:
            async def call(self, tool, args, rationale):
                order.append(("call", tool))
                return SimpleNamespace(ok=True, text="", status="succeeded", action_id=None)

        agent.mcp = _MCP()
        from mlaya import graph
        for name in ("apply_flee", "apply_hunt", "restore_base"):
            tag = name.replace("apply_", "").replace("restore_base", "restore")

            async def staged(executor, reason, _tag=tag):
                order.append((_tag, reason))
                return {"ok": True}

            original = getattr(graph.survival, name)
            setattr(graph.survival, name, staged)
            self.addCleanup(setattr, graph.survival, name, original)
        return agent

    async def test_leave_nether_runs_under_the_flee_stage(self) -> None:
        order: list = []
        agent = self._agent(order)
        await agent._execute({"action": self._Action("leave_nether", "navigate", {"x": 1})})
        self.assertEqual([k for k, _ in order], ["flee", "call", "restore"])
        self.assertEqual(agent.log[0]["reason"], "flee_stage_on")

    async def test_an_ordinary_nether_walk_is_not_staged(self) -> None:
        order: list = []
        agent = self._agent(order)
        await agent._execute({"action": self._Action("explore", "navigate", {"heading": 90.0})})
        self.assertEqual([k for k, _ in order], ["call"])
        self.assertEqual(agent.log, [])


class DeathPolicyTest(unittest.IsolatedAsyncioTestCase):
    """The survival policy goes back on after every death, however close the next one follows.

    run90 died nine times in the Overworld, two of them 34 s apart and two 25 s apart, and the
    old age window re-applied the policy for three of the ten deaths.
    """

    def _agent(self) -> tuple[Agent, list]:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._death_seen = None
        agent.log = []
        agent.mcp = None
        applied: list[str] = []

        async def apply(executor, reason, preset="base"):
            applied.append(reason)
            return {"ok": True}

        from mlaya import graph
        original = graph.survival.apply_policy
        graph.survival.apply_policy = apply
        self.addCleanup(setattr, graph.survival, "apply_policy", original)
        return agent, applied

    async def test_two_deaths_in_quick_succession_both_reapply(self) -> None:
        agent, applied = self._agent()
        await agent._reapply_after_death(gs(last_death_at="2026-10-23T19:12:30Z",
                                            last_death_age_s=4.0))
        await agent._reapply_after_death(gs(last_death_at="2026-10-23T19:12:30Z",
                                            last_death_age_s=20.0))
        await agent._reapply_after_death(gs(last_death_at="2026-10-23T19:13:04Z",
                                            last_death_age_s=3.0))
        self.assertEqual(len(applied), 2)
        self.assertEqual([e["reason"] for e in agent.log], ["death", "death"])

    async def test_an_old_death_found_at_the_start_is_left_alone(self) -> None:
        agent, applied = self._agent()
        await agent._reapply_after_death(gs(last_death_at="2026-10-22T08:00:00Z",
                                            last_death_age_s=50462.0))
        await agent._reapply_after_death(gs())
        self.assertEqual(applied, [])


class ShutInTest(unittest.TestCase):
    """A pathfinder that cannot leave the pocket hands the tick to `dig_out`.

    run90 stood its last 330 ticks shut in; every walk answered `no path found` after 70-78
    nodes, the closest one beside the bot, where a search that truly runs out visits ~100,000.
    """

    POCKET = ("[NAVIGATION_STOPPED] Pathfinder stopped: no path found after 2 ms compute; "
              "visited 70 nodes, generated 78; closest node was 181,70,184.")
    # run95 it125: the walk began 105 m away and 52 blocks below the cell it was refused at.
    RUN95 = ("[NAVIGATION_STOPPED] Pathfinder stopped: no path found after 0 ms compute; "
             "visited 1 nodes, generated 1; closest node was -216,-2,-367.")
    FAR = ("[NAVIGATION_STOPPED] Pathfinder stopped: no path found after 900 ms compute; "
           "visited 101324 nodes, generated 120000; closest node was 240,70,150.")
    # run96: the body did not move on a re-issued route, reported with no node count for
    # `shut_in` to read. The first code is `[NAVIGATION_STOPPED]`, which is what used to reset
    # the count instead of raising it.
    STUCK = ("[NAVIGATION_STOPPED] Pathfinder stopped: repeated_movement_failure: "
             "Server confirmation deadline elapsed.")

    class _Action:
        def __init__(self, label: str) -> None:
            self.label = label

    def _agent(self) -> Agent:
        agent = Agent.__new__(Agent)
        agent._pearl_wave_trips = 0
        agent._nether_landmarks = []
        agent._boxed = (None, 0)
        agent._dig_out_tries = 0
        return agent

    def test_reads_the_pocket_and_not_a_long_search(self) -> None:
        here = (182.5, 70.0, 181.5)
        self.assertTrue(shut_in(self.POCKET, here))
        self.assertTrue(shut_in("no path found after 1 ms compute; visited 76 nodes, generated "
                                "105; closest node was 178,70,183", here))
        self.assertFalse(shut_in(self.FAR, here))
        # Few nodes but the closest one far off: a ledge to cross, not a pocket.
        self.assertFalse(shut_in("no path found after 24 ms compute; visited 402 nodes, "
                                 "generated 436; closest node was 223,100,16", (364.3, 101, -11.5)))

    def test_three_refusals_from_one_cell_shut_the_bot_in(self) -> None:
        agent = self._agent()
        here = gs(pos=(182.5, 70.0, 181.5))
        for label in ("explore", "get_wood", "explore")[:BOXED_REPEATS]:
            self.assertFalse(agent._boxed[1] >= BOXED_REPEATS)
            agent._note_boxed(here, self._Action(label), False, self.POCKET)
        self.assertEqual(agent._boxed, ((182, 70, 181), BOXED_REPEATS))

    def test_a_body_that_cannot_move_shuts_the_bot_in(self) -> None:
        # run96: 100 `explore` from one spawn cell walled in birch logs and leaves, every one
        # `repeated_movement_failure`, never a step. `shut_in` reads no node count off it, so the
        # count rose only once this refusal fed the tunnel directly; before, its
        # `[NAVIGATION_STOPPED]` code reset the count on the very tick that should have raised it.
        agent = self._agent()
        here = gs(pos=(-64.6, 73.0, 3.5))
        self.assertFalse(shut_in(self.STUCK, (-64.6, 73.0, 3.5)))
        for label in ("explore", "explore", "explore")[:BOXED_REPEATS]:
            self.assertFalse(agent._boxed[1] >= BOXED_REPEATS)
            agent._note_boxed(here, self._Action(label), False, self.STUCK)
        self.assertEqual(agent._boxed, ((-65, 73, 3), BOXED_REPEATS))

    def test_a_dug_tunnel_leaves_one_refusal_to_the_next_side(self) -> None:
        agent = self._agent()
        here = gs(pos=(182.5, 70.0, 181.5))
        agent._boxed = ((182, 70, 181), BOXED_REPEATS)
        agent._note_boxed(here, self._Action("dig_out"), True, "")
        self.assertEqual((agent._boxed[1], agent._dig_out_tries), (BOXED_REPEATS - 1, 1))
        agent._note_boxed(here, self._Action("explore"), False, self.POCKET)
        self.assertEqual(agent._boxed[1], BOXED_REPEATS)
        # A walk that got out clears it all: the body left the cell.
        agent._note_boxed(gs(pos=(190.5, 70.0, 181.5)), self._Action("explore"), True, "")
        self.assertEqual((agent._boxed, agent._dig_out_tries), ((None, 0), 0))

    def test_a_refused_dig_turns_to_the_next_side_then_gives_up(self) -> None:
        # run101 it6-181: boxed on a fractional-y perch on the old world's leftover frame, every
        # `dig_out` refused `[BUILD_INCOMPLETE] ... no path found; visited 1 node`. The side only
        # advanced on a dig that went through, so `dig_out_try` stayed 0 and the same east cells
        # were dug at 176 times to the iteration limit. A refused dig now turns to the next side,
        # and once every side has been tried round without giving the latch drops, so the tick is
        # no longer spent on `dig_out` alone.
        refused = ("[BUILD_INCOMPLETE] 3 cells still wrong: 3 were refused (-56,72,45: no path "
                   "found after 0 ms compute; visited 1 nodes, generated 1; closest node was "
                   "-57,72,45)")
        agent = self._agent()
        here = gs(pos=(-56.5, 72.56, 45.5))
        agent._boxed = ((-57, 72, 45), BOXED_REPEATS)
        for n in range(1, DIG_OUT_MAX_TRIES):
            agent._note_boxed(here, self._Action("dig_out"), False, refused)
            self.assertEqual(agent._dig_out_tries, n)   # the side turns on the refusal
            self.assertTrue(agent._boxed[1] >= BOXED_REPEATS)   # still shut in, dig_out still owns the tick
        # Every side tried round and none gave: the latch drops so the rest of the ladder runs.
        agent._note_boxed(here, self._Action("dig_out"), False, refused)
        self.assertEqual((agent._boxed, agent._dig_out_tries), ((None, 0), 0))

    def test_work_done_in_place_keeps_the_count(self) -> None:
        # run95 it126: the tick after the first refused `go_home`, `collect_drops` succeeded in
        # the very cell the bot was suffocating in. A pickup, a craft or a meal proves nothing
        # about whether the pathfinder can leave, so it must not start the count over.
        agent = self._agent()
        here = gs(pos=(-215.4, -2.0, -366.4))
        agent._note_boxed(here, self._Action("go_home"), False, self.RUN95)
        agent._note_boxed(here, self._Action("collect_drops"), True, "")
        self.assertEqual(agent._boxed, ((-216, -2, -367), 1))
        agent._note_boxed(here, self._Action("go_home"), False, self.RUN95)
        agent._note_boxed(here, self._Action("go_home"), False, self.RUN95)
        self.assertEqual(agent._boxed, ((-216, -2, -367), BOXED_REPEATS))

    def test_moving_or_a_long_search_starts_over(self) -> None:
        agent = self._agent()
        agent._boxed = ((182, 70, 181), 2)
        agent._dig_out_tries = 2
        agent._note_boxed(gs(pos=(190.5, 70.0, 181.5)), self._Action("explore"), False,
                          self.FAR)
        self.assertEqual((agent._boxed, agent._dig_out_tries), ((None, 0), 0))

    def test_the_refusal_counts_at_the_cell_the_walk_ended_in(self) -> None:
        # run95 it125: the tick began at (-299,-54,-402), `go_home` left the body inside the wall
        # and the refusal named that very cell. Read against the pre-action cell it looked like a
        # walk that had simply run out of path, the count stayed at 0 of 3, and the bot answered
        # `go_home` from inside the block until it suffocated.
        agent = self._agent()
        agent._note_boxed(gs(pos=(-215.4, -2.0, -366.4)), self._Action("go_home"), False,
                          self.RUN95)
        self.assertEqual(agent._boxed, ((-216, -2, -367), 1))
        agent._note_boxed(gs(pos=(-215.4, -2.0, -366.4)), self._Action("go_home"), False,
                          self.RUN95)
        self.assertEqual(agent._boxed[1], 2)

    def test_the_pre_action_cell_is_not_what_the_refusal_speaks_about(self) -> None:
        # The same refusal read against the cell the tick *began* in: nothing is near the closest
        # node, so this is the shape that hid the pocket from run95's loop.
        agent = self._agent()
        agent._note_boxed(gs(pos=(-299.4, -54.0, -402.4)), self._Action("go_home"), False,
                          self.RUN95)
        self.assertEqual(agent._boxed, (None, 0))

    def test_inject_hands_it_to_the_state(self) -> None:
        agent = RecoverWriteOffTest._agent(RecoverWriteOffTest())
        agent._boxed = ((182, 70, 181), BOXED_REPEATS)
        agent._dig_out_tries = 2
        agent._water_ticks = 0
        injected = agent._inject(gs(pos=(182.5, 70.0, 181.5)))
        self.assertTrue(injected.boxed_in)
        self.assertEqual(injected.dig_out_try, 2)


if __name__ == "__main__":
    unittest.main()
