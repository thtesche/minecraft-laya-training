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
        agent._unblock_attempts = 0
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

    async def test_keeps_label_when_not_blocked(self) -> None:
        agent = self._agent()
        state = {"before": gs(inventory={"crafting_table": 1, "cobblestone": 8}),
                 "decision": Decision("craft_furnace")}
        out = await agent._guard(state)
        self.assertEqual(out["action"].tool, "craft_item")

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


if __name__ == "__main__":
    unittest.main()
