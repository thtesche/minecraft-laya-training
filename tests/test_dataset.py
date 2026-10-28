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

    def test_logged_yes_survives_an_ambiguous_state(self):
        # No hostile distance recorded: the state cannot prove safety, so the logged `yes`
        # (the example the gate was missing) still stands.
        state = {"health": 20.0, "hostiles": []}
        answers = {DANGER_QID: {"choice": "yes"}}
        self.assertEqual(ds.danger_label(state, answers), "yes")

    def test_logged_yes_survives_a_hostile_inside_the_band(self):
        # In the ambiguous band between the floor (`DANGER_CLOSE_M`) and the ceiling
        # (`DANGER_SAFE_M`, boundary included) the state cannot decide, so the gate's `yes` is
        # kept.
        answers = {DANGER_QID: {"choice": "yes"}}
        for d in (4.5, 6.0, 8.0):
            state = {"health": 20.0, "hostiles": [["zombie", 1, d]]}
            self.assertEqual(ds.danger_label(state, answers), "yes", f"at {d} m")

    def test_logged_yes_is_dropped_when_the_state_proves_safe(self):
        # A far, known hostile at full health is the over-trigger the runtime band absorbs
        # (live run7b answered `yes` at 9-24 m): the ceiling labels it `no` despite the `yes`.
        state = {"health": 20.0, "hostiles": [["zombie", 1, 12.0]]}
        answers = {DANGER_QID: {"choice": "yes"}}
        self.assertEqual(ds.danger_label(state, answers), "no")

    def test_ceiling_needs_health_above_the_band(self):
        # Low-ish health keeps the logged `yes` even with the mob far away.
        state = {"health": 14.0, "hostiles": [["zombie", 1, 12.0]]}
        answers = {DANGER_QID: {"choice": "yes"}}
        self.assertEqual(ds.danger_label(state, answers), "yes")

    def test_flat_choice_answer_accepted(self):
        answers = {DANGER_QID: "yes"}
        self.assertEqual(ds.danger_label({"health": 20.0}, answers), "yes")


class RenderNarrativeTest(unittest.TestCase):
    """A tick is re-read through today's `narrative()` from its logged `state` (A4)."""

    def test_a_tick_with_state_is_rerendered(self):
        rec = {"narrative": "old absolute wording x=1, y=2, z=3",
               "state": {"health": 20.0, "food": 20.0, "position": [1000.0, 72.0, -500.0],
                         "on_ground": True, "hostiles": []}}
        out = ds.render_narrative(rec)
        self.assertIn("At the surface", out)
        self.assertNotIn("x=", out)

    def test_a_tick_without_state_keeps_its_logged_narrative(self):
        self.assertEqual(ds.render_narrative({"narrative": "logged only"}), "logged only")

    def test_rerender_changes_the_row_state(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _rec("old absolute x=5", ["mine_stone", "get_wood"], "mine_stone")
            rec["state"] = {"health": 20.0, "food": 20.0, "position": [5.0, 70.0, 5.0],
                            "on_ground": True, "hostiles": []}
            p = _write(Path(d), [rec])
            plain = ds.build_rows(p)
            rerendered = ds.build_rows(p, rerender=True)
        self.assertEqual(plain[0]["state"], "old absolute x=5")
        self.assertIn("At the surface", rerendered[0]["state"])
        self.assertNotIn("x=", rerendered[0]["state"])


class TeacherSourceTest(unittest.TestCase):
    """The teacher side-file fills no-progress ticks; the verifier always wins a collision (B4)."""

    def _tick(self, iteration, candidates, label, *, progressed, narrative, state=None):
        rec = _rec(narrative, candidates, label, progressed=progressed)
        rec["iteration"] = iteration
        if state is not None:
            rec["state"] = state
        return rec

    def _teacher(self, rec, judge_label):
        return {"iteration": rec["iteration"],
                "narrative_hash": ds._narrative_hash(ds.render_narrative(rec)),
                "candidates": rec["candidates"], "laya_label": rec["label"],
                "judge_label": judge_label}

    def test_a_no_progress_tick_is_filled_by_the_judge(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._tick(3, ["mine_stone", "get_wood"], "get_wood",
                             progressed=False, narrative="safe no-progress")
            log = _write(Path(d), [rec])
            tf = Path(d) / "teacher.jsonl"
            tf.write_text(json.dumps(self._teacher(rec, "mine_stone")) + "\n", encoding="utf-8")
            plain = ds.build_rows(log)
            taught = ds.build_rows(log, teacher=tf)
        self.assertEqual(plain, [])  # no progress, nothing from the verifier
        self.assertEqual(len(taught), 1)
        self.assertEqual(taught[0]["expected"][INTENT_QID], "mine_stone")

    def test_the_verifier_wins_a_progressed_tick(self):
        # Laya's action progressed: the grounded row stands, the judge's dissent is dropped.
        with tempfile.TemporaryDirectory() as d:
            rec = self._tick(3, ["mine_stone", "get_wood"], "get_wood",
                             progressed=True, narrative="safe progressed")
            log = _write(Path(d), [rec])
            tf = Path(d) / "teacher.jsonl"
            tf.write_text(json.dumps(self._teacher(rec, "mine_stone")) + "\n", encoding="utf-8")
            rows = ds.build_rows(log, teacher=tf)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expected"][INTENT_QID], "get_wood")

    def test_a_judge_label_equal_to_layas_is_not_a_fill(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._tick(3, ["mine_stone", "get_wood"], "get_wood",
                             progressed=False, narrative="safe")
            log = _write(Path(d), [rec])
            tf = Path(d) / "teacher.jsonl"
            tf.write_text(json.dumps(self._teacher(rec, "get_wood")) + "\n", encoding="utf-8")
            rows = ds.build_rows(log, teacher=tf)
        self.assertEqual([r for r in rows if INTENT_QID in r["expected"]], [])

    def test_the_join_survives_rerender(self):
        # Side-file hashes the canonical re-rendered narrative, so the join holds whether or not
        # the dataset itself is built with rerender.
        state = {"health": 20.0, "food": 20.0, "position": [5.0, 70.0, 5.0],
                 "on_ground": True, "hostiles": []}
        with tempfile.TemporaryDirectory() as d:
            rec = self._tick(3, ["mine_stone", "get_wood"], "get_wood",
                             progressed=False, narrative="old absolute x=5", state=state)
            log = _write(Path(d), [rec])
            tf = Path(d) / "teacher.jsonl"
            tf.write_text(json.dumps(self._teacher(rec, "mine_stone")) + "\n", encoding="utf-8")
            rows = ds.build_rows(log, teacher=tf, rerender=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expected"][INTENT_QID], "mine_stone")
        self.assertIn("At the surface", rows[0]["state"])


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


class LoopFilterTest(unittest.TestCase):
    """Ping-pong ticks and walk-only deposits teach no intent (run82, run84)."""

    def _act(self, n, label, inv=None):
        rec = _rec(f"state {n}", ["go_home", "stow_surplus", "mine_obsidian"], label)
        rec["delta"] = {"inventory_delta": inv or {}}
        return rec

    def _rows(self, records):
        with tempfile.TemporaryDirectory() as d:
            return ds.build_rows(_write(Path(d), records), with_danger=False)

    def test_an_alternation_with_nothing_gained_is_dropped(self):
        # run82: go_home <-> stow_surplus, every tick "ok", delta {}.
        recs = [self._act(i, "go_home" if i % 2 else "stow_surplus", {"cobblestone": -1} if i % 2 == 0 else {})
                for i in range(ds.LOOP_MIN_TICKS)]
        self.assertEqual(self._rows(recs), [])

    def test_gains_and_losses_that_cancel_are_still_a_loop(self):
        # run84: mine_obsidian +25 cobblestone, go_home -25.
        recs = [self._act(i, "mine_obsidian" if i % 2 == 0 else "go_home",
                          {"cobblestone": 25 if i % 2 == 0 else -25})
                for i in range(ds.LOOP_MIN_TICKS)]
        self.assertEqual(self._rows(recs), [])

    def test_a_short_alternation_is_kept(self):
        recs = [self._act(i, "mine_obsidian" if i % 2 == 0 else "go_home")
                for i in range(ds.LOOP_MIN_TICKS - 1)]
        self.assertEqual(len(self._rows(recs)), ds.LOOP_MIN_TICKS - 1)

    def test_an_alternation_that_gains_is_work(self):
        recs = [self._act(i, "mine_obsidian" if i % 2 == 0 else "go_home",
                          {"obsidian": 1} if i % 2 == 0 else {})
                for i in range(ds.LOOP_MIN_TICKS)]
        self.assertEqual(len(self._rows(recs)), ds.LOOP_MIN_TICKS)

    def test_a_stow_that_only_walked_is_no_deposit(self):
        rows = self._rows([self._act(0, "stow_surplus"),
                           self._act(1, "stow_surplus", {"cobblestone": -64}),
                           self._act(2, "go_home")])
        self.assertEqual([r["state"] for r in rows], ["state 1", "state 2"])

    def test_several_logs_are_read_as_one_set(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.jsonl", Path(d) / "b.jsonl"
            a.write_text(json.dumps(self._act(0, "go_home")) + "\n", encoding="utf-8")
            b.write_text(json.dumps(self._act(1, "mine_obsidian")) + "\n", encoding="utf-8")
            rows = ds.build_rows([a, b], with_danger=False)
        self.assertEqual([r["state"] for r in rows], ["state 0", "state 1"])


if __name__ == "__main__":
    unittest.main()
