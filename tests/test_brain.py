from __future__ import annotations

import unittest
from types import SimpleNamespace

from mlaya.brain import (DANGER_BAND_NOTE, DANGER_DEFER_HEALTH, DANGER_DEFER_M, DANGER_QID,
                         INTENT_QID, Brain, danger_matters)


class _Router:
    """Stands in for Laya: returns canned answers so the band can be tested on its own."""

    def __init__(self, answers: dict) -> None:
        self._answers = answers

    def predict(self, state: str, questions: dict, **kwargs) -> dict:
        return {"answers": self._answers}


def _brain(answers: dict) -> Brain:
    brain = object.__new__(Brain)
    brain._cfg = SimpleNamespace(model=None, max_len=None, min_confidence=None)
    brain.router = _Router(answers)
    return brain


def _gs(health: float = 20.0, nearest: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(health=health, nearest_hostile=nearest)


def _gate_yes(intent: str = "mine_diamond") -> dict:
    return {DANGER_QID: {"choice": "yes"}, INTENT_QID: {"choice": intent}}


class DangerBandTest(unittest.TestCase):
    """The gate defers unconditionally on `yes`, so a yes it drew from a merely distant mob
    froze run7b at diamond depth (health and hunger full, server policy idle). The band keeps
    the loop working unless the body is really at risk."""

    def test_far_mob_with_health_in_hand_keeps_working(self) -> None:
        decision = _brain(_gate_yes()).decide(
            "narrative", ["mine_diamond", "hunt_food"], _gs(20.0, 14.0))
        self.assertFalse(decision.deferred)
        self.assertEqual(decision.action_label, "mine_diamond")
        self.assertEqual(decision.note, DANGER_BAND_NOTE)

    def test_close_mob_hands_the_tick_to_the_policy(self) -> None:
        decision = _brain(_gate_yes()).decide(
            "narrative", ["mine_diamond", "hunt_food"], _gs(20.0, DANGER_DEFER_M))
        self.assertTrue(decision.deferred)
        self.assertIsNone(decision.action_label)

    def test_low_health_hands_the_tick_to_the_policy(self) -> None:
        decision = _brain(_gate_yes()).decide(
            "narrative", ["mine_diamond", "hunt_food"], _gs(DANGER_DEFER_HEALTH - 0.1, 14.0))
        self.assertTrue(decision.deferred)

    def test_no_state_keeps_the_unconditional_defer(self) -> None:
        decision = _brain(_gate_yes()).decide("narrative", ["mine_diamond", "hunt_food"])
        self.assertTrue(decision.deferred)

    def test_a_gate_no_is_not_banded(self) -> None:
        answers = {DANGER_QID: {"choice": "no"}, INTENT_QID: {"choice": "mine_diamond"}}
        decision = _brain(answers).decide(
            "narrative", ["mine_diamond", "hunt_food"], _gs(20.0, 14.0))
        self.assertFalse(decision.deferred)
        self.assertEqual(decision.note, "")


class DangerMattersTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertTrue(danger_matters(None))
        self.assertTrue(danger_matters(_gs(20.0, DANGER_DEFER_M)))
        self.assertFalse(danger_matters(_gs(20.0, DANGER_DEFER_M + 0.1)))
        self.assertTrue(danger_matters(_gs(DANGER_DEFER_HEALTH - 0.1, None)))
        self.assertFalse(danger_matters(_gs(DANGER_DEFER_HEALTH, None)))


if __name__ == "__main__":
    unittest.main()
