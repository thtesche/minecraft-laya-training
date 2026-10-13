"""Tests for the M3 dataset builder (`mlaya.dataset`). Stdlib unittest, no extra deps.

Run: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlaya import dataset as ds
from mlaya.actions import CRITERIA
from mlaya.brain import DANGER_Q, DANGER_QID, INTENT_QID, intent_question


def _rec(narrative, candidates, label, *, progressed=True, health=20.0,
         hostiles=None, answers=None, stage="action", note=None):
    return {
        "stage": stage,
        "narrative": narrative,
        "candidates": list(candidates),
        "label": label,
        "progressed": progressed,
        "state": {"health": health, "hostiles": hostiles or []},
        "answers": answers,
        "note": note,
    }


def _write(tmp: Path, records) -> Path:
    p = tmp / "decisions.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


class BuildRowsTest(unittest.TestCase):
    def test_multi_candidate_progressed_becomes_row(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("safe state", ["mine_stone", "get_wood"], "mine_stone"),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["state"], "safe state")
        self.assertEqual(row["expected"][INTENT_QID], "mine_stone")
        # The intent question must be byte-for-byte the inference rendering.
        self.assertEqual(row["questions"][INTENT_QID], intent_question(["mine_stone", "get_wood"]))
        # Danger row present, labelled from the (safe) state heuristic.
        self.assertEqual(row["expected"][DANGER_QID], "no")
        self.assertEqual(row["questions"][DANGER_QID], DANGER_Q)

    def test_single_candidate_is_deterministic_and_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("safe", ["mine_stone"], "mine_stone"),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(rows, [])

    def test_failed_tick_dropped_by_default_kept_with_flag(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("safe", ["mine_stone", "get_wood"], "get_wood", progressed=False),
            ])
            self.assertEqual(ds.build_rows(p), [])
            kept = ds.build_rows(p, include_failed=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["expected"][INTENT_QID], "get_wood")

    def test_a_refused_danger_tick_becomes_a_yes_row(self):
        # A refused action carries no usable intent, but at 0.3 health it is the one example
        # that teaches the gate the low-health `yes` - even with a logged `no`.
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("hurt", ["eat_food", "hunt_wool"], "eat_food", progressed=False,
                     health=0.3, answers={DANGER_QID: {"choice": "no"}}),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(rows, [{"state": "hurt",
                                 "questions": {DANGER_QID: DANGER_Q},
                                 "expected": {DANGER_QID: "yes"}}])

    def test_duplicate_state_and_candidates_deduped(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("same", ["mine_stone", "get_wood"], "mine_stone"),
                _rec("same", ["mine_stone", "get_wood"], "get_wood"),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expected"][INTENT_QID], "mine_stone")

    def test_unknown_candidates_and_labels_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("x", ["mine_stone", "not_a_real_label"], "not_a_real_label"),
            ])
            rows = ds.build_rows(p)
        # After filtering unknown labels only one candidate remains -> deterministic -> skipped.
        self.assertEqual(rows, [])

    def test_a_defer_without_a_danger_reason_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("safe", ["mine_stone", "get_wood"], "mine_stone", stage="defer",
                     answers={DANGER_QID: {"choice": "no"}}, note="no valid action"),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(rows, [])

    def test_a_danger_defer_becomes_a_yes_row(self):
        # A deferred tick produces no action at all, so an action-only dataset would drop it -
        # yet it is the one place a real gate `yes` survives. It becomes a danger-only row.
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("a skeleton is shooting me", ["mine_stone", "get_wood"], None,
                     stage="defer", answers={DANGER_QID: {"choice": "yes"}},
                     note="in danger -> survival policy"),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "a skeleton is shooting me")
        self.assertEqual(rows[0]["expected"], {DANGER_QID: "yes"})
        self.assertNotIn(INTENT_QID, rows[0]["questions"])

    def test_a_danger_defer_is_dropped_without_the_danger_flag(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("a skeleton is shooting me", ["mine_stone", "get_wood"], None,
                     stage="defer", note="in danger -> survival policy"),
            ])
            rows = ds.build_rows(p, with_danger=False)
        self.assertEqual(rows, [])

    def test_the_danger_note_alone_is_enough(self):
        # Older logs carry the reason in the note but no logged answers.
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("hurt", ["mine_stone", "get_wood"], None, stage="defer",
                     note="in danger -> survival policy"),
            ])
            rows = ds.build_rows(p)
        self.assertEqual(rows, [{"state": "hurt",
                                 "questions": {DANGER_QID: DANGER_Q},
                                 "expected": {DANGER_QID: "yes"}}])

    def test_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec(f"state {i}", ["mine_stone", "get_wood"], "mine_stone") for i in range(5)
            ])
            rows = ds.build_rows(p, limit=2)
        self.assertEqual(len(rows), 2)

    def test_no_danger_flag(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), [
                _rec("safe", ["mine_stone", "get_wood"], "mine_stone"),
            ])
            rows = ds.build_rows(p, with_danger=False)
        self.assertNotIn(DANGER_QID, rows[0]["questions"])


class DangerLabelTest(unittest.TestCase):
    def test_low_health_is_danger(self):
        self.assertEqual(ds.danger_label({"health": 5.0, "hostiles": []}, None), "yes")

    def test_close_hostile_is_danger(self):
        state = {"health": 20.0, "hostiles": [["zombie", 1, 3.0]]}
        self.assertEqual(ds.danger_label(state, None), "yes")

    def test_far_hostile_is_safe(self):
        state = {"health": 20.0, "hostiles": [["zombie", 1, 12.0]]}
        self.assertEqual(ds.danger_label(state, None), "no")

    def test_heuristic_yes_overrides_a_logged_no(self):
        # The logged `no` at fatal health is the mis-calibration the gate is trained to fix,
        # so the state heuristic wins over it.
        state = {"health": 3.0, "hostiles": []}  # heuristic says yes
        answers = {DANGER_QID: {"choice": "no"}}
        self.assertEqual(ds.danger_label(state, answers), "yes")

    def test_logged_yes_survives_a_safe_heuristic(self):
        # The gate under-triggers, so a logged `yes` is kept where the heuristic is silent.
        state = {"health": 20.0, "hostiles": [["zombie", 1, 12.0]]}
        answers = {DANGER_QID: {"choice": "yes"}}
        self.assertEqual(ds.danger_label(state, answers), "yes")

    def test_flat_choice_answer_accepted(self):
        answers = {DANGER_QID: "yes"}
        self.assertEqual(ds.danger_label({"health": 20.0}, answers), "yes")


class SplitTest(unittest.TestCase):
    def test_reproducible(self):
        rows = [{"state": f"s{i}", "questions": {}, "expected": {}} for i in range(10)]
        a = ds.split(rows, 0.3, seed=7)
        b = ds.split(rows, 0.3, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]) + len(a[1]), 10)

    def test_tiny_set_has_no_eval(self):
        rows = [{"state": f"s{i}", "questions": {}, "expected": {}} for i in range(3)]
        train, ev = ds.split(rows, 0.25)
        self.assertEqual(ev, [])
        self.assertEqual(len(train), 3)


class RealLogSmokeTest(unittest.TestCase):
    def test_questions_use_known_criteria(self):
        rows = ds.build_rows(Path("runs/decisions.jsonl"))
        for row in rows:
            q = row["questions"].get(INTENT_QID)
            if q is None:
                continue  # a danger-only row from a defer
            for label in q["criteria"]:
                self.assertIn(label, CRITERIA)


if __name__ == "__main__":
    unittest.main()
