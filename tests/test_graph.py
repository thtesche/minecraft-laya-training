"""Failure recovery: an action blocked by the bot's surroundings is unblocked instead of
being retried in place.

Live-validated in the iron chain: after mining ore the bot sat in a 1x1 shaft and
`craft_furnace` answered `[WORKSTATION_PLACEMENT_FAILED] crafting_table: no_cell` on every
tick, so the run never reached the furnace. The fix is a code-level unlock: when the same
label that just failed this way comes up again, dig a small pocket where the bot stands so
the workstation fits (`make_room`), and only on a second try walk to open ground.
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
        self.assertIsNone(agent._blocked_label)
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


if __name__ == "__main__":
    unittest.main()
