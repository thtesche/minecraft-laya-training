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

from mlaya.brain import Decision
from mlaya.config import Settings
from mlaya.graph import Agent, blocked_by_surroundings, unblock_action
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
        agent._pour_skip = set()
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
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent._portal_blocked = []
        agent.build_site = None
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
        agent._portal_blocked = []
        agent.build_site = None
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
        agent._portal_blocked = []
        agent.build_site = None
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
        agent._drop_hole_blocked = False
        agent._chest_blocked = True
        self.assertTrue(agent._inject(gs()).chest_blocked)


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
        # The full overworld goal: diamond kit worn, shield off-hand, the portal lit and a bed
        # beside it (both read from the world-block resources, not the inventory).
        return gs(tools={"pickaxe": "diamond", "sword": "diamond"},
                  inventory={"diamond_sword": 1},
                  armor={slot: {"item": f"diamond_{slot}"}
                         for slot in ("helmet", "chestplate", "leggings", "boots")},
                  offhand="shield",
                  resources={
                      "nether_portal": [BlockHit("nether_portal", "nether_portal",
                                                 -270, 64, -56, 3.0)],
                      "white_bed": [BlockHit("white_bed", "white_bed", -269, 64, -56, 4.0)],
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
        agent._portal_blocked = []
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
        agent._drop_hole_blocked = False
        agent._chest_blocked = False
        agent.build_site = None
        self.assertEqual(agent._inject(gs()).portal_blocked, [(-195, -46, -98)])


if __name__ == "__main__":
    unittest.main()
