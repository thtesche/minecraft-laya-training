from __future__ import annotations

import unittest
from types import SimpleNamespace

from mlaya.actions import _must_eat
from mlaya.brain import (DANGER_BAND_NOTE, DANGER_DEFER_HEALTH, DANGER_DEFER_M, DANGER_QID,
                         DEFER_QUIET_TICKS, INTENT_QID, Brain, danger_matters)


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

    def test_a_mid_range_mob_keeps_working(self) -> None:
        # run13 froze ~60 ticks on a skeleton/creeper hovering 5-8 m: in shooting range, never
        # closing to a hit, so the survival policy idled while the loop stood aside. Only contact
        # range may hand the tick over now, so a mob at 6 m keeps the loop working.
        decision = _brain(_gate_yes()).decide(
            "narrative", ["mine_diamond", "hunt_food"], _gs(20.0, 6.0))
        self.assertFalse(decision.deferred)
        self.assertEqual(decision.action_label, "mine_diamond")
        self.assertEqual(decision.note, DANGER_BAND_NOTE)

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

    def test_quiet_defers_hand_contact_range_back_to_the_loop(self) -> None:
        # run86: blazes at 2 m behind the spawner fence, health full, the reflex silent - the
        # loop deferred 120+ ticks without moving.
        g = _gs(20.0, 2.0)
        g.defer_quiet = DEFER_QUIET_TICKS - 1
        self.assertTrue(danger_matters(g))
        g.defer_quiet = DEFER_QUIET_TICKS
        self.assertFalse(danger_matters(g))

    def test_low_health_defers_however_quiet_while_it_can_heal(self) -> None:
        # Food in the pack, or hunger high enough to regenerate: the policy heals the body, so
        # the defer holds however quiet the ticks are.
        g = _gs(DANGER_DEFER_HEALTH - 0.1, 2.0)
        g.defer_quiet = DEFER_QUIET_TICKS + 10
        g.food = 17.0
        g.inventory = {"cooked_beef": 3}
        self.assertTrue(danger_matters(g))
        g.inventory = {}
        g.food = 18.0
        self.assertTrue(danger_matters(g))

    def test_low_health_with_nothing_to_heal_on_goes_back_to_the_loop(self) -> None:
        # run88 from it64: 8.5 health after cave-spider bites, hunger 17, not one edible item,
        # the spiders held off at 10-16 m. The policy had nothing to do and 536 defers changed
        # nothing; after the quiet defers the loop takes the tick for its recovery steps.
        g = _gs(8.5, 10.0)
        g.food = 17.0
        g.inventory = {"cobblestone": 179, "spider_eye": 2}
        g.defer_quiet = DEFER_QUIET_TICKS - 1
        self.assertTrue(danger_matters(g))
        g.defer_quiet = DEFER_QUIET_TICKS
        self.assertFalse(danger_matters(g))

    def test_low_health_with_a_meal_nothing_in_range_lets_the_loop_eat(self) -> None:
        # run95 from it462: 1 health, hunger 6, two raw chicken carried, the nearest hostile 40 m
        # off, `candidates: ['eat_food']` - and 138 defers to the end of the run. An edible in the
        # pack makes `_cannot_heal` false, so the quiet defers never released the tick, and the
        # server's hunger reflex does not eat raw chicken.
        g = _gs(1.0, 40.0)
        g.food = 6.0
        g.inventory = {"chicken": 2, "feather": 3}
        g.defer_quiet = 0
        self.assertFalse(danger_matters(g))
        self.assertTrue(_must_eat(g))

    def test_a_meal_does_not_turn_the_back_on_a_mob_in_contact(self) -> None:
        # Eating with a hostile at 2 m is how the next hit finds the bot: in contact range the
        # policy keeps the tick (it fights, and it eats once the fight is over).
        g = _gs(1.0, DANGER_DEFER_M)
        g.food = 6.0
        g.inventory = {"chicken": 2}
        self.assertTrue(danger_matters(g))

    def test_a_shut_in_body_takes_the_tick_whatever_the_health(self) -> None:
        # run95 it125-130: suffocating in a wall at 9 health, nearest hostile 23 m, the offer one
        # `go_home` walk that the pathfinder could not start. The policy has no answer to a body
        # inside a block; the loop's `dig_out` has.
        g = _gs(9.0, 23.0)
        g.food = 20.0
        g.inventory = {"cobblestone": 30}
        g.boxed_in = True
        self.assertFalse(danger_matters(g))
        g.boxed_in = False
        self.assertTrue(danger_matters(g))


if __name__ == "__main__":
    unittest.main()
