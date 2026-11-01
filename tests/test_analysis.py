"""The analyzer's numbers, checked against a log written by hand.

Every case here is a shape the live logs actually produce: the refusal code in brackets,
a run that walks in place on one label, a death record from the policy stage.
"""

from __future__ import annotations

import unittest

from mlaya import analysis


def action(iteration: int, label: str, *, status="failed", text="", pos=0.0,
           gained=0, health=20.0, food=20.0, progressed=None,
           water=False, swim=None) -> dict:
    return {
        "ts": 1000 + iteration,
        "stage": "action",
        "iteration": iteration,
        "label": label,
        "tool": "navigate",
        "status": status,
        "text": text,
        "progressed": progressed if progressed is not None else (status == "succeeded"),
        "state": {"health": health, "food": food, "position": [0, 64, 0], "in_water": water},
        "delta": {
            "pos_delta": pos,
            # The x/z plane the body actually crossed. A tick that only drifts up and back
            # down the same column has a `pos_delta` and no `swim_delta`.
            "swim_delta": pos if swim is None else swim,
            "inventory_delta": {"cobblestone": gained} if gained else {},
        },
    }


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
