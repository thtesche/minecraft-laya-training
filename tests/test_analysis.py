"""The analyzer's numbers, checked against a log written by hand.

Every case here is a shape the live logs actually produce: the refusal code in brackets,
a run that walks in place on one label, a death record from the policy stage.
"""

from __future__ import annotations

import unittest

from mlaya import analysis


def action(iteration: int, label: str, *, status="failed", text="", pos=0.0,
           gained=0, health=20.0, food=20.0, progressed=None,
           water=False, swim=None, escalated=False, at=(0, 64, 0)) -> dict:
    return {
        "ts": 1000 + iteration,
        "stage": "action",
        "iteration": iteration,
        "label": label,
        "tool": "navigate",
        "status": status,
        "text": text,
        "escalated": escalated,
        "progressed": progressed if progressed is not None else (status == "succeeded"),
        "state": {"health": health, "food": food, "position": list(at), "in_water": water},
        "delta": {
            "pos_delta": pos,
            # The x/z plane the body actually crossed. A tick that only drifts up and back
            # down the same column has a `pos_delta` and no `swim_delta`.
            "swim_delta": pos if swim is None else swim,
            "inventory_delta": {"cobblestone": gained} if gained else {},
        },
    }


def escalation(iteration: int, unresolved: str, plan: dict, chosen: str | None) -> dict:
    """One System 2 call: the label System 1 offered but could not resolve, the model's plan,
    and the label the loop ended up running."""
    return {"ts": 1000 + iteration, "stage": "escalation", "iteration": iteration,
            "label": unresolved, "plan": plan, "chosen": chosen}


STALL = [action(i, "leave_nether", text="[NAVIGATION_TARGET_UNSUPPORTED] no path")
         for i in range(7)]
MOVING = [action(i, "explore", status="succeeded", pos=4.0) for i in range(7, 12)]


class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.records = STALL + MOVING + [
            {"stage": "policy", "reason": "death", "iteration": 12, "summary": "died at (-3,72,-9)"},
            {"stage": "policy", "reason": "world_state_restored", "iteration": 13,
             "facts": ["gate_lit", "frame_complete"]},
            {"stage": "defer", "iteration": 14},
        ]
        self.report = analysis.summarize(self.records)

    def test_refusals_count_and_locate(self):
        refusals = self.report["refusals"]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["code"], "NAVIGATION_TARGET_UNSUPPORTED")
        self.assertEqual(refusals[0]["count"], 7)
        self.assertEqual(refusals[0]["first_iteration"], 0)
        self.assertEqual(refusals[0]["labels"], ["leave_nether"])

    def test_stall_window_is_the_whole_ping_pong(self):
        stalls = self.report["stalls"]
        self.assertEqual(len(stalls), 1)
        self.assertEqual(stalls[0]["ticks"], 7)
        self.assertEqual(stalls[0]["start_iteration"], 0)
        self.assertEqual(stalls[0]["end_iteration"], 6)
        self.assertEqual(stalls[0]["codes"], ["NAVIGATION_TARGET_UNSUPPORTED"])

    def test_movement_separates_standing_from_walking(self):
        movement = self.report["movement"]
        self.assertEqual(movement["total_blocks"], 20.0)
        self.assertEqual(movement["still_ticks"], 7)
        self.assertEqual(movement["longest_still_streak"], 7)

    def test_a_tick_that_gains_without_walking_is_not_still(self):
        # run141 spent 8 ticks at the furnace: no step, one iron ingot each. Counting those
        # as standing put still_ratio at 0.84 for a run that was working the whole time.
        report = analysis.summarize([action(i, "smelt_iron", status="succeeded", gained=1)
                                     for i in range(8)])
        self.assertEqual(report["movement"]["still_ticks"], 0)
        self.assertEqual(report["movement"]["still_ratio"], 0.0)
        self.assertEqual(report["stalls"], [])

    def test_deaths_come_from_the_policy_record(self):
        self.assertEqual(self.report["vitals"]["deaths"], 1)
        self.assertEqual(self.report["vitals"]["death_iterations"], [12])

    def test_milestones_are_first_seen_facts(self):
        facts = {m["fact"] for m in self.report["milestones"]}
        self.assertEqual(facts, {"gate_lit", "frame_complete"})

    def test_markdown_carries_the_headline_numbers(self):
        text = analysis.to_markdown(self.report)
        self.assertIn("NAVIGATION_TARGET_UNSUPPORTED", text)
        self.assertIn("it0–it6", text)


class TestWater(unittest.TestCase):
    """How much of a run is spent off the feet, and how much of that swim goes nowhere.

    A body adrift in a flooded pocket travels three blocks straight up and back down on the
    same column every tick (live, run115: 281 rescue walks from (-0.5, 62, 281), `pos_delta`
    3.5 each, never on ground). `pos_delta` calls that movement; the x/z plane says it went
    nowhere, which is the defect the report has to show.
    """

    def setUp(self):
        self.report = analysis.summarize(
            [action(i, "escape_water", status="succeeded", pos=3.5, swim=0.0, water=True)
             for i in range(6)]
            + [action(6, "escape_water", status="succeeded", pos=4.0, swim=4.0, water=True),
               action(7, "mine_coal", status="succeeded", pos=5.0, gained=1)]
        )

    def test_water_ticks_are_counted_and_streaked(self):
        water = self.report["water"]
        self.assertEqual(water["ticks"], 7)
        self.assertEqual(water["ratio"], 0.875)
        self.assertEqual(water["longest_streak"], 7)
        self.assertEqual(water["escape_ticks"], 7)

    def test_adrift_is_a_swim_that_crosses_no_ground(self):
        water = self.report["water"]
        self.assertEqual(water["adrift_ticks"], 6)
        self.assertEqual(water["swim_blocks"], 4.0)

    def test_a_dry_run_reports_no_water(self):
        self.assertEqual(analysis.summarize(MOVING)["water"]["ticks"], 0)

    def test_markdown_names_the_water(self):
        text = analysis.to_markdown(self.report)
        self.assertIn("7 ticks off the feet", text)
        self.assertIn("6 of them adrift", text)


class TestLoops(unittest.TestCase):
    """Ticks that walk a long path and end where they began.

    `still_ratio` cannot see a bounce: a bot hopping between two stances moves five blocks every
    tick, which reads as movement. Live, run143 it63-78: `approach_lava` between (-66,8,-67) and
    (-66,7,-73) for 94 ticks while the portal frame still owed ten obsidian.
    """

    A = (-66.5, 8.0, -67.4)
    B = (-66.5, 7.0, -72.9)

    def setUp(self):
        self.report = analysis.summarize(
            [action(i, "approach_lava", status="succeeded", pos=5.6,
                    at=self.A if i % 2 else self.B) for i in range(12)]
            + [action(12, "mine_iron", status="succeeded", pos=6.0, gained=2, at=(10, 64, 10))]
        )

    def test_a_bounce_between_two_stances_is_a_loop(self):
        loops = self.report["loops"]
        self.assertEqual(loops["labels"], 1)
        self.assertEqual(loops["ticks"], 12)
        detail = loops["detail"][0]
        self.assertEqual(detail["label"], "approach_lava")
        self.assertEqual(detail["start_iteration"], 0)
        self.assertEqual(detail["end_iteration"], 11)

    def test_a_walk_that_gets_somewhere_is_not_a_loop(self):
        report = analysis.summarize([action(i, "explore", status="succeeded", pos=6.0,
                                           at=(6 * i, 64, 0)) for i in range(12)])
        self.assertEqual(report["loops"]["labels"], 0)

    def test_a_label_scattered_over_the_run_is_not_one_episode(self):
        # Eight sleep ticks twelve iterations apart hop between two cells, but they are not one
        # bounce: the bot went home, worked elsewhere, came back. The span is what makes an episode.
        report = analysis.summarize([action(i, "sleep", status="succeeded", pos=6.0,
                                           at=self.A if k % 2 else self.B)
                                     for k, i in enumerate(range(0, 100, 12))])
        self.assertEqual(report["loops"]["labels"], 0)

    def test_loops_show_in_the_markdown(self):
        text = analysis.to_markdown(self.report)
        self.assertIn("Loops (12 ticks walking, going nowhere)", text)
        self.assertIn("`approach_lava`", text)

    def test_a_loop_that_ends_shows_in_the_diff(self):
        dry = analysis.summarize([action(i, "mine_iron", status="succeeded", pos=6.0,
                                         gained=1, at=(6 * i, 64, 0)) for i in range(12)])
        diff = analysis.compare(dry, self.report)
        self.assertEqual(diff["deltas"]["loops.ticks"], -12)


class TestSystem2(unittest.TestCase):
    """System 2 calls: the escalation the loop pays when System 1 runs out.

    The tech tree can offer a label whose concrete action cannot be built from the state
    (`resolve` returns None); that is the only path to the generative model, and the run log
    records the call and the action it produced.
    """

    def setUp(self):
        self.report = analysis.summarize([
            escalation(5, "hunt_string", {"action": "hunt_string", "goal": "get string",
                                          "note": "spider nearby"}, "hunt_string"),
            action(5, "hunt_string", status="succeeded", pos=3.0, escalated=True),
            escalation(6, "cook_food", {"error": "missing STRATA_API_KEY"}, None),
            action(6, "explore", status="succeeded", pos=0.0, progressed=False, escalated=True),
        ])

    def test_calls_and_their_outcome_are_counted(self):
        s2 = self.report["system2"]
        self.assertEqual(s2["calls"], 2)
        self.assertEqual(s2["ticks"], 2)
        self.assertEqual(s2["progressed"], 1)
        self.assertEqual(s2["progress_rate"], 0.5)
        self.assertEqual(s2["fallbacks"], 1)
        self.assertEqual(s2["errors"], 1)

    def test_the_trigger_label_and_the_chosen_one_are_listed(self):
        s2 = self.report["system2"]
        self.assertEqual(s2["triggers"], [{"label": "hunt_string", "count": 1},
                                         {"label": "cook_food", "count": 1}])
        self.assertEqual(s2["chosen"][0], {"label": "hunt_string", "count": 1})
        self.assertEqual(s2["calls_detail"][0]["goal"], "get string")

    def test_a_run_that_never_escalates_reports_zero(self):
        self.assertEqual(analysis.summarize(MOVING)["system2"]["calls"], 0)

    def test_markdown_names_the_escalations(self):
        text = analysis.to_markdown(self.report)
        self.assertIn("System 2 escalations (2)", text)
        self.assertIn("`hunt_string` had no action -> `hunt_string`", text)


class TestCompare(unittest.TestCase):
    def test_diff_against_a_baseline(self):
        baseline = analysis.summarize(
            STALL + [action(i, "wait", status="succeeded") for i in range(7, 17)]
        )
        current = analysis.summarize(STALL + MOVING)
        diff = analysis.compare(current, baseline)
        # The baseline never moved at all (7 stuck ticks plus 10 waits); the current run
        # covers 20 blocks in 12 ticks, so the per-tick movement gains its full 1.67.
        self.assertEqual(diff["deltas"]["movement.blocks_per_tick"], 1.67)
        self.assertEqual(diff["deltas"]["waits.wait_ticks"], -10)
        self.assertEqual(diff["worse_refusals"], [])

    def test_water_gains_and_losses_show_in_the_diff(self):
        # A run that gets out of the water is a better run, and the diff has to say so.
        baseline = analysis.summarize([action(i, "escape_water", status="succeeded", pos=3.5,
                                             swim=0.0, water=True) for i in range(10)])
        current = analysis.summarize([action(i, "mine_iron", status="succeeded", pos=4.0,
                                            gained=1) for i in range(10)])
        diff = analysis.compare(current, baseline)
        self.assertEqual(diff["deltas"]["water.ratio"], -1.0)
        self.assertEqual(diff["deltas"]["water.adrift_ticks"], -10)


if __name__ == "__main__":
    unittest.main()
