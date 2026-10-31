"""The analyzer's numbers, checked against a log written by hand.

Every case here is a shape the live logs actually produce: the refusal code in brackets,
a run that walks in place on one label, a death record from the policy stage.
"""

from __future__ import annotations

import unittest

from mlaya import analysis


def action(iteration: int, label: str, *, status="failed", text="", pos=0.0,
           gained=0, health=20.0, food=20.0, progressed=None) -> dict:
    return {
        "ts": 1000 + iteration,
        "stage": "action",
        "iteration": iteration,
        "label": label,
        "tool": "navigate",
        "status": status,
        "text": text,
        "progressed": progressed if progressed is not None else (status == "succeeded"),
        "state": {"health": health, "food": food, "position": [0, 64, 0]},
        "delta": {
            "pos_delta": pos,
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


if __name__ == "__main__":
    unittest.main()
