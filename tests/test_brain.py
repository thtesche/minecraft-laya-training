from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from mlaya.actions import CRITERIA, _must_eat
from mlaya import brain
from mlaya.brain import (DANGER_BAND_NOTE, DANGER_DEFER_HEALTH, DANGER_DEFER_M, DANGER_QID,
                        DEFER_QUIET_TICKS, INTENT_QID, Brain, System2Dialogue,
                        _failure_diagnostics, _plan_steps,
                        _review_situation, _system2_state_view, danger_matters,
                        escalate_to_system2, escalation_system, escalation_tail, review_payload,
                        trajectory_digest)


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


def _rec(label: str, *, ok=False, progressed=False, status="failed", state=None) -> dict:
    rec = {"stage": "action", "label": label, "ok": ok, "progressed": progressed,
           "status": status}
    if state is not None:
        rec["state"] = state
    return rec


class TrajectoryDigestTest(unittest.TestCase):
    """The run's own decision log folded down to what a next-move decision needs."""

    def test_recent_labels_and_repeated_failures_surface(self) -> None:
        history = (
            [_rec("explore", ok=True, progressed=True, status="succeeded")]
            + [_rec("mine_obsidian") for _ in range(4)]
            + [_rec("light_portal") for _ in range(2)]
        )
        digest = trajectory_digest(history)
        self.assertEqual(digest["window"], 7)
        self.assertEqual(digest["recent"][-1]["label"], "light_portal")
        # Worst-first, and a label that failed once is not listed (threshold is 2).
        failing = [f["label"] for f in digest["repeated_failures"]]
        self.assertEqual(failing[0], "mine_obsidian")
        self.assertEqual(digest["repeated_failures"][0]["failed"], 4)
        self.assertNotIn("explore", failing)

    def test_the_window_bounds_how_far_back_it_reads(self) -> None:
        history = [_rec(f"l{i}") for i in range(40)]
        self.assertEqual(trajectory_digest(history, window=10)["window"], 10)

    def test_only_action_records_count_and_the_ledger_is_the_last_state(self) -> None:
        history = [
            {"stage": "policy", "reason": "stuck_offer_dropped", "label": "explore"},
            _rec("mine_iron", state={"phase": "iron", "base": [1, 2, 3], "chest_pos": None}),
        ]
        digest = trajectory_digest(history)
        self.assertEqual(digest["window"], 1)
        self.assertEqual(digest["ledger"]["phase"], "iron")
        self.assertEqual(digest["ledger"]["base"], [1, 2, 3])


class PlanStepsTest(unittest.TestCase):
    """The ordered labels a System-2 reply proposes, filtered to allowed set only."""

    def test_a_list_plan_is_kept_in_order_dropping_non_strings(self) -> None:
        self.assertEqual(_plan_steps({"plan": ["a", 3, "b"]}, ["a", "b"]), ["a", "b"])

    def test_labels_outside_allowed_set_are_dropped(self) -> None:
        # If the model proposes a label not in the candidate set, it must be filtered
        # out to prevent a mismatch between the escalation candidates and the action run.
        self.assertEqual(_plan_steps({"plan": ["craft_flint_and_steel", "get_wood"]}, ["get_wood"]), ["get_wood"])
        self.assertEqual(_plan_steps({"plan": ["invalid", "another_invalid"]}, ["get_wood"]), [])

    def test_a_legacy_single_action_becomes_a_one_step_plan(self) -> None:
        self.assertEqual(_plan_steps({"action": "mine_stone"}, ["mine_stone"]), ["mine_stone"])

    def test_an_error_or_empty_reply_has_no_steps(self) -> None:
        self.assertEqual(_plan_steps({"error": "boom"}, ["a"]), [])
        self.assertEqual(_plan_steps(None, ["a"]), [])


class EscalateContractTest(unittest.TestCase):
    """What `escalate_to_system2` guarantees without a model behind it."""

    def test_a_missing_key_is_reported_not_raised(self) -> None:
        out = escalate_to_system2(
            SimpleNamespace(api_key_env="STRATA_API_KEY"), None,
            {"phase": "early"}, {"trigger": "unresolved_label"}, ["get_wood"])
        self.assertEqual(out, {"error": "missing STRATA_API_KEY"})


class EscalationBriefTest(unittest.TestCase):
    """The escalation brief carries the sentence System 1 reads, not only the raw state.

    run252 it40: System 2's prompt had `position: [-7.5, 63.0, 60.5]` and `on_ground: true`
    and nothing that said "at surface altitude, open sky not confirmed" - the observer read
    the altitude band as "already out" while `surface_up` was re-probing a new column.
    """

    def _state(self) -> dict:
        return {
            "health": 20.0, "food": 20.0, "phase": "day", "raining": False,
            "position": [-7.5, 63.0, 60.5], "on_ground": True, "in_water": False,
            "dimension": "overworld", "inventory": {"cobblestone": 20}, "tools": {},
            "armor": {}, "hostiles": [], "base": None, "chest_pos": None, "slots_used": 3,
            "surface": None, "surface_wet": [],
        }

    def test_the_brief_carries_the_situation_laya_reads(self) -> None:
        tail = escalation_tail(self._state(), {"trigger": "stall"}, ["surface_up"])
        self.assertIn("At surface altitude (standing on solid floor, open sky not confirmed)",
                      tail["situation"])

    def test_a_probed_surface_reads_as_open_sky(self) -> None:
        state = self._state()
        state["surface"] = [-8, 63, 60]
        tail = escalation_tail(state, {}, ["surface_up"])
        self.assertIn("open sky above", tail["situation"])
        self.assertEqual(tail["state"]["surface"], [-8, 63, 60])

    def test_the_ready_set_replaces_the_full_catalogue(self) -> None:
        # The 108 descriptions live in the byte-stable spine; the appended turn only names which of
        # them are available this tick. That is the difference between a 34k-token brief and a tail.
        tail = escalation_tail(self._state(), {"label": "surface_up"}, ["surface_up"])
        self.assertEqual(tail["ready"], ["surface_up"])
        self.assertNotIn("actions", tail)
        self.assertNotIn("recent_history", tail)
        self.assertIn("diagnostics", tail["failure"])
        self.assertEqual(tail["phase"]["name"], "diamond_kit")

    def test_the_trajectory_is_attached_when_passed(self) -> None:
        tail = escalation_tail(self._state(), {}, ["surface_up"], trajectory={"window": 3})
        self.assertEqual(tail["trajectory"], {"window": 3})

    def test_the_spine_is_byte_stable_and_holds_the_catalogue(self) -> None:
        """The cached prefix must not change mid-run, so nothing tick-dependent is in it."""
        spine = escalation_system()
        self.assertEqual(spine, escalation_system())
        for label, desc in CRITERIA.items():
            self.assertIn(f"- {label}: {desc}", spine)
        self.assertIn("death traps", spine)
        self.assertNotIn("At surface altitude", spine)


class System2DialogueTest(unittest.TestCase):
    """The escalation conversation grows by appending, which is what makes the cache hit.

    `scripts/cache_probe.py` measured the rule: the slot reuses its prefix only for a pure
    extension. A conversation that is rebuilt every escalation - the old shape - prefills all
    34k tokens cold every time.
    """

    def _state(self) -> dict:
        return {"health": 20.0, "food": 20.0, "position": [0, 64, 0], "on_ground": True,
                "dimension": "overworld", "inventory": {}, "tools": {}, "armor": {},
                "hostiles": [], "base": None, "chest_pos": None, "slots_used": 1}

    def test_each_turn_is_a_pure_extension_of_the_last(self) -> None:
        seen: list[list[dict]] = []

        def fake_chat(cfg, api_key, messages):
            seen.append([dict(m) for m in messages])
            return json.dumps({"plan": ["get_wood"], "goal": "wood", "note": "n"}), \
                {"prompt_tokens": 1000, "cached_tokens": 900, "prefill_ms": 300}

        dlg = System2Dialogue(SimpleNamespace(model="m", max_tokens=4096, base_url="u"), "key")
        with mock.patch.object(brain, "_system2_chat", fake_chat):
            first = dlg.turn(self._state(), {"label": "mine_stone"}, ["get_wood"], None, None)
            second = dlg.turn(self._state(), {"label": "mine_stone"}, ["get_wood"], None,
                              {"iteration": 3, "steps": ["get_wood"], "ran": [], "resolved": False})

        self.assertEqual(seen[0], dlg.messages[:2])
        self.assertEqual(seen[1][:len(seen[0])], seen[0])   # turn 2 starts with all of turn 1
        self.assertEqual(seen[1][0]["content"], escalation_system())  # the spine never changes
        self.assertEqual(first[0]["plan"], ["get_wood"])
        self.assertEqual(first[1]["cached_tokens"], 900)
        self.assertIn("previous_plan", json.loads(seen[1][-1]["content"]))

    def test_the_reasoning_chain_is_not_kept_in_the_story(self) -> None:
        """Only the plan rides in the assistant turn: the thinking tokens are the bulk of it."""
        def fake_chat(cfg, api_key, messages):
            return json.dumps({"plan": ["get_wood"], "goal": "wood"}), {"prompt_tokens": 10}

        dlg = System2Dialogue(SimpleNamespace(model="m", max_tokens=4096, base_url="u"), "key")
        with mock.patch.object(brain, "_system2_chat", fake_chat):
            dlg.turn(self._state(), {}, ["get_wood"], None, None)
        self.assertEqual(json.loads(dlg.messages[-1]["content"]), {"plan": ["get_wood"], "goal": "wood"})

    def test_a_bad_reply_still_leaves_a_readable_plan(self) -> None:
        def fake_chat(cfg, api_key, messages):
            return "not json at all", {"prompt_tokens": 10}

        dlg = System2Dialogue(SimpleNamespace(model="m", max_tokens=4096, base_url="u"), "key")
        with mock.patch.object(brain, "_system2_chat", fake_chat):
            data, _ = dlg.turn(self._state(), {}, ["get_wood"], None, None)
        self.assertIn("error", data)
        self.assertEqual(data["plan"], [])
        self.assertIsNone(data["action"])


class System2StateViewTest(unittest.TestCase):
    """The System-2 view trims the noise and annotates what the model would otherwise misread."""

    def _nether_state(self) -> dict:
        return {
            "dimension": "the_nether",
            "phase": None,  # the server's time-of-day box, empty outside the Overworld
            "slots_used": 33,
            "players": ["LayaView"],
            "base": [-77, 69, -27],
            "dropped_items": [["gold_nugget", 1, 87.6, [-374.96, 32.0, 147.87]]],
        }

    def test_the_player_roll_call_is_dropped(self) -> None:
        self.assertNotIn("players", _system2_state_view(self._nether_state()))

    def test_the_run_phase_fills_the_empty_time_of_day_box(self) -> None:
        self.assertEqual(_system2_state_view(self._nether_state())["phase"], "nether_rods")

    def test_a_pearl_trip_is_named_as_one(self) -> None:
        # The Nether has two jobs. Once the rod haul is banked the trip hunts pearls, and the
        # slug System 2 reads has to say so, not keep calling it a blaze hunt.
        state = self._nether_state()
        state["nether_trip"] = "pearls"
        self.assertEqual(_system2_state_view(state)["phase"], "nether_pearls")

    def test_the_base_is_marked_as_the_overworld_landmark(self) -> None:
        self.assertEqual(_system2_state_view(self._nether_state())["base"],
                         {"pos": [-77, 69, -27], "dimension": "overworld"})

    def test_a_run_without_a_base_leaves_it_none(self) -> None:
        state = self._nether_state()
        state["base"] = None
        self.assertIsNone(_system2_state_view(state)["base"])

    def test_the_logged_state_is_not_mutated(self) -> None:
        state = self._nether_state()
        _system2_state_view(state)
        self.assertEqual(state["phase"], None)
        self.assertEqual(state["base"], [-77, 69, -27])
        self.assertIn("players", state)


class ReviewPayloadTest(unittest.TestCase):
    """The reviewer reads a digest, not the raw state dump and eight full records."""

    def _state(self) -> dict:
        return {
            "health": 20.0, "food": 15.0, "phase": "day", "raining": False,
            "position": [-288.49, 53.0, -378.79], "on_ground": True, "in_water": False,
            "dimension": "overworld", "inventory": {"oak_log": 3, "acacia_log": 5},
            "tools": {}, "armor": {}, "hostiles": [], "base": None, "chest_pos": None,
            "slots_used": 8, "progress": {"gate": False, "bed": False, "powder": 0, "eyes": 0},
        }

    def test_the_payload_is_situation_and_digest_not_raw_fields(self) -> None:
        payload = review_payload(self._state(), [], ["get_wood", "craft_pickaxe"])
        self.assertEqual(sorted(payload), ["candidates", "phase", "situation", "trajectory"])
        self.assertNotIn("state", payload)
        self.assertNotIn("recent_history", payload)

    def test_a_blocked_goal_rides_beside_the_candidates(self) -> None:
        # The gated-milestone hint is threaded through so the reviewer sees why a phase end-state
        # action is absent from the candidates, instead of reading its absence as a loop.
        blocked = {"action": "light_portal", "needs": ["spare diamond pickaxe missing (1/2)"]}
        payload = review_payload(self._state(), [], ["cook_food", "mine_diamond"], blocked)
        self.assertEqual(payload["blocked_goal"], blocked)
        # Ordered beside the candidates (the slowly-changing block), ahead of the per-tick situation.
        keys = list(payload)
        self.assertLess(keys.index("blocked_goal"), keys.index("situation"))

    def test_no_blocked_goal_key_without_one(self) -> None:
        payload = review_payload(self._state(), [], ["get_wood"])
        self.assertNotIn("blocked_goal", payload)

    def test_the_situation_is_the_paragraph_system_1_reads(self) -> None:
        payload = review_payload(self._state(), [], ["get_wood"])
        self.assertIn("3x oak_log", payload["situation"])
        self.assertIn("No hostile mobs within 16 blocks", payload["situation"])

    def test_the_history_is_a_label_ledger_not_full_records(self) -> None:
        history = [{"stage": "action", "label": "get_wood", "ok": True, "progressed": True,
                   "status": "succeeded", "state": {"phase": "day"}}]
        digest = review_payload(self._state(), history, ["get_wood"])["trajectory"]
        self.assertEqual(digest["window"], 1)
        self.assertEqual(digest["recent"][0]["label"], "get_wood")

    def test_the_review_window_is_twelve_ticks(self) -> None:
        history = [{"stage": "action", "label": f"l{i}", "ok": True, "progressed": True,
                   "status": "succeeded"} for i in range(40)]
        self.assertEqual(review_payload(self._state(), history, ["get_wood"])["trajectory"]["window"], 12)

    def test_the_digest_stays_small_enough_to_prefill_fast(self) -> None:
        import json
        # The raw view the digest replaced ran 88k characters on a mid-run tick; the reviewer's
        # brief has to be an order of magnitude smaller or the prefill dominates every call.
        payload = review_payload(self._state(), [{"stage": "action", "state": self._state()}] * 8,
                                 ["get_wood", "craft_pickaxe"])
        self.assertLess(len(json.dumps(payload)), 4000)

    def test_a_bare_state_still_reads_without_raising(self) -> None:
        situation = review_payload({"not_a_state": 1}, [], ["get_wood"])["situation"]
        self.assertEqual(_review_situation({"not_a_state": 1}), situation)
        self.assertIn("inventory: empty", situation.lower())


class FailureDiagnosticsTest(unittest.TestCase):
    """The computed reason a stuck reaction keeps refusing, so System 2 need not reverse-engineer it."""

    def test_a_pinned_pack_reads_as_full(self) -> None:
        diag = _failure_diagnostics({"slots_used": 36})
        self.assertEqual(diag["pack"], {"slots_used": 36, "slots_max": 36, "full": True})

    def test_a_pack_with_room_is_not_full(self) -> None:
        self.assertFalse(_failure_diagnostics({"slots_used": 33})["pack"]["full"])

    def test_the_nearest_drop_distance_is_surfaced(self) -> None:
        diag = _failure_diagnostics({
            "slots_used": 20,
            "dropped_items": [["gold_nugget", 1, 87.6, None], ["blaze_rod", 1, 5.2, None]],
        })
        self.assertEqual(diag["nearest_drop_m"], 5.2)

    def test_no_drops_means_no_distance_field(self) -> None:
        self.assertNotIn("nearest_drop_m", _failure_diagnostics({"slots_used": 20}))


if __name__ == "__main__":
    unittest.main()
