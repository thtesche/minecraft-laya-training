"""Tests for the hold-out registry and the checkpoint comparison (M4). Stdlib unittest.

Run: PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlaya import cli
from mlaya import dataset as ds
from mlaya import holdout as ho
from mlaya.brain import INTENT_QID


def _rec(narrative, candidates, label, *, progressed=True, iteration=0):
    return {
        "stage": "action",
        "iteration": iteration,
        "narrative": narrative,
        "candidates": list(candidates),
        "label": label,
        "progressed": progressed,
        "state": {"health": 20.0, "hostiles": []},
        "answers": {},
    }


def _write(tmp: Path, records, name="decisions-run257.jsonl") -> Path:
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


class RegistryTest(unittest.TestCase):
    def test_add_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "holdout.json"
            ho.add_run(257, Path("runs/history/decisions-run257.jsonl"), "fresh world", path)
            ho.add_run(258, Path("runs/history/decisions-run258.jsonl"), "", path)
            runs = ho.load_registry(path)
            self.assertEqual([r["run"] for r in runs], [257, 258])
            self.assertEqual(runs[0]["note"], "fresh world")

    def test_add_is_idempotent_and_updates_the_note(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "holdout.json"
            ho.add_run(257, Path("a.jsonl"), "first", path)
            ho.add_run(257, Path("b.jsonl"), "second", path)
            runs = ho.load_registry(path)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["log"], "b.jsonl")
            self.assertEqual(runs[0]["note"], "second")

    def test_empty_registry(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ho.load_registry(Path(d) / "missing.json"), [])

    def test_holdout_logs_skips_runs_without_a_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = _write(Path(d), [_rec("safe state", ["mine_stone", "get_wood"], "mine_stone")])
            runs = [{"run": 257, "log": str(log)}, {"run": 9999, "log": ""}]
            self.assertEqual(ho.holdout_logs(runs), [log])

    def test_the_archived_log_beats_the_live_one(self):
        # A run registered while it is still running points at `runs/decisions.jsonl`, which the
        # next run overwrites. Once `decisions-runNN.jsonl` exists it must be the one used.
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            live = _write(d, [_rec("live state", ["mine_stone", "get_wood"], "mine_stone")],
                          name="decisions.jsonl")
            archived = _write(d, [_rec("archived state", ["cook_food", "hunt_food"], "cook_food")],
                              name="decisions-run257.jsonl")
            old = ho.run_log
            ho.run_log = lambda run: archived
            try:
                self.assertEqual(ho.holdout_logs([{"run": 257, "log": str(live)}]), [archived])
                self.assertEqual(ho.log_for({"run": 257, "log": str(live)}), archived)
                # An explicit archived path stays authoritative even when another is found.
                self.assertEqual(ho.log_for({"run": 257, "log": str(archived)}), archived)
            finally:
                ho.run_log = old


class EvalBuildTest(unittest.TestCase):
    def test_build_eval_writes_rows_and_counts_them(self):
        with tempfile.TemporaryDirectory() as d:
            log = _write(Path(d), [
                _rec("safe state one", ["mine_stone", "get_wood"], "mine_stone", iteration=1),
                _rec("safe state two", ["cook_food", "hunt_food"], "cook_food", iteration=2),
            ])
            out = Path(d) / "eval.jsonl"
            res = ho.build_eval(out, runs=[{"run": 257, "log": str(log)}])
            self.assertEqual(res["eval_rows"], 2)
            self.assertEqual(res["runs"], [257])
            rows = ds._read_records(out)
            self.assertTrue(all(INTENT_QID in r["expected"] for r in rows))
            # The hold-out target is what the verifier confirmed, never a teacher hedge.
            self.assertFalse(any(isinstance(r.get("gold"), dict) for r in rows))

    def test_no_runs_registered_reports_a_hint(self):
        res = ho.build_eval(Path("/nonexistent/eval.jsonl"), runs=[])
        self.assertIn("error", res)
        self.assertIn("holdout add", res["hint"])

    def test_overlap_with_the_training_file_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _rec("safe state one", ["mine_stone", "get_wood"], "mine_stone")
            log = _write(Path(d), [rec])
            # The training file holds dataset rows, not raw ticks: overlap compares states.
            train = Path(d) / "train.jsonl"
            ds.write_jsonl(train, ds.build_rows(log, rerender=True, reverify=True))
            res = ho.build_eval(Path(d) / "eval.jsonl", runs=[{"run": 257, "log": str(log)}],
                                train=train)
            self.assertEqual(res["overlap_with_train_dropped"], 1)
            self.assertEqual(res["eval_rows"], 0)

    def test_a_different_state_is_not_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            # Different health and position: after `rerender` these render to different states,
            # so the hold-out row is genuinely unseen by the training file.
            train_log = _write(Path(d), [_rec("other state", ["mine_stone", "get_wood"],
                                             "mine_stone")], name="train-ticks.jsonl")
            train = Path(d) / "train.jsonl"
            ds.write_jsonl(train, ds.build_rows(train_log, rerender=True, reverify=True))
            other = _rec("safe state one", ["mine_stone", "get_wood"], "mine_stone")
            other["state"] = {"health": 12.0, "hostiles": [{"type": "zombie", "distance": 3}]}
            log = _write(Path(d), [other], name="holdout-ticks.jsonl")
            res = ho.build_eval(Path(d) / "eval.jsonl", runs=[{"run": 257, "log": str(log)}],
                                train=train)
            self.assertEqual(res["overlap_with_train_dropped"], 0)
            self.assertEqual(res["eval_rows"], 1)

    def test_several_training_files_are_all_checked(self):
        with tempfile.TemporaryDirectory() as d:
            log = _write(Path(d), [
                _rec("state a", ["mine_stone", "get_wood"], "mine_stone", iteration=1),
                _rec("state b", ["cook_food", "hunt_food"], "cook_food", iteration=2),
            ])
            t1 = Path(d) / "train1.jsonl"
            t2 = Path(d) / "train2.jsonl"
            rows = ds.build_rows(log, rerender=True, reverify=True)
            ds.write_jsonl(t1, rows[:1])
            ds.write_jsonl(t2, rows[1:])
            res = ho.build_eval(Path(d) / "eval.jsonl", runs=[{"run": 257, "log": str(log)}],
                                train=[t1, t2])
            self.assertEqual(res["overlap_with_train_dropped"], 2)
            self.assertEqual(res["eval_rows"], 0)
            self.assertEqual(len(res["train_files_checked"]), 2)

    def test_per_run_cap_keeps_one_run_from_domination(self):
        with tempfile.TemporaryDirectory() as d:
            log = _write(Path(d), [
                _rec("state a", ["mine_stone", "get_wood"], "mine_stone", iteration=1),
                _rec("state b", ["cook_food", "hunt_food"], "cook_food", iteration=2),
                _rec("state c", ["mine_coal", "get_wood"], "mine_coal", iteration=3),
            ])
            res = ho.build_eval(Path(d) / "eval.jsonl", runs=[{"run": 257, "log": str(log)}],
                                per_run=2)
            self.assertEqual(res["eval_rows"], 2)
            self.assertEqual(res["per_run_cap"], 2)

    def test_the_sample_is_the_same_every_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            log = _write(Path(d), [
                _rec("state a", ["mine_stone", "get_wood"], "mine_stone", iteration=1),
                _rec("state b", ["cook_food", "hunt_food"], "cook_food", iteration=2),
                _rec("state c", ["mine_coal", "get_wood"], "mine_coal", iteration=3),
            ])
            runs = [{"run": 257, "log": str(log)}]
            first = ho.build_eval(Path(d) / "e1.jsonl", runs=runs, per_run=2)
            second = ho.build_eval(Path(d) / "e2.jsonl", runs=runs, per_run=2)
            self.assertEqual([r["state"] for r in ds._read_records(Path(d) / "e1.jsonl")],
                             [r["state"] for r in ds._read_records(Path(d) / "e2.jsonl")])
            self.assertEqual(first["eval_items"], second["eval_items"])

    def test_every_row_carries_the_run_it_came_from(self):
        with tempfile.TemporaryDirectory() as d:
            log = _write(Path(d), [_rec("safe state one", ["mine_stone", "get_wood"],
                                        "mine_stone")])
            ho.build_eval(Path(d) / "eval.jsonl", runs=[{"run": 257, "log": str(log)}])
            rows = ds._read_records(Path(d) / "eval.jsonl")
            self.assertEqual([r["run"] for r in rows], [257])


class DatasetGuardTest(unittest.TestCase):
    """`mlaya dataset` refuses a run the registry says is held out."""

    def test_a_registered_holdout_run_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            log = _write(d, [_rec("safe state one", ["mine_stone", "get_wood"], "mine_stone")],
                         name="decisions-run257.jsonl")
            reg = d / "holdout.json"
            ho.add_run(257, log, "", reg)
            old = ho.REGISTRY_PATH
            ho.REGISTRY_PATH = reg
            try:
                rc = cli.main(["dataset", "--log", str(log), "--out", str(d / "train.jsonl")])
            finally:
                ho.REGISTRY_PATH = old
            self.assertEqual(rc, 2)
            self.assertFalse((d / "train.jsonl").exists())

    def test_an_unregistered_run_passes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            log = _write(d, [_rec("safe state one", ["mine_stone", "get_wood"], "mine_stone")],
                         name="decisions-run900.jsonl")
            old = ho.REGISTRY_PATH
            ho.REGISTRY_PATH = d / "holdout.json"
            try:
                rc = cli.main(["dataset", "--log", str(log), "--out", str(d / "train.jsonl")])
            finally:
                ho.REGISTRY_PATH = old
            self.assertEqual(rc, 0)
            self.assertTrue((d / "train.jsonl").exists())


class StatisticsTest(unittest.TestCase):
    def test_mcnemar_counts_only_discordant_pairs(self):
        a = [True, True, False, False, True]
        b = [True, False, True, False, True]
        r = ho.mcnemar(a, b)
        self.assertEqual(r["a_wins"], 1)
        self.assertEqual(r["b_wins"], 1)
        self.assertEqual(r["discordant"], 2)
        self.assertAlmostEqual(r["p"], 1.0)

    def test_mcnemar_of_identical_predictions_is_one(self):
        r = ho.mcnemar([True, False, True], [True, False, True])
        self.assertEqual(r["discordant"], 0)
        self.assertEqual(r["p"], 1.0)

    def test_mcnemar_p_follows_the_imbalance_not_the_count(self):
        # Same number of discordant pairs: a 10/10 split says nothing, a 15/5 split does.
        balanced = ho.mcnemar([True] * 10 + [False] * 10, [False] * 10 + [True] * 10)
        imbalanced = ho.mcnemar([True] * 15 + [False] * 5, [False] * 15 + [True] * 5)
        self.assertEqual(balanced["discordant"], imbalanced["discordant"])
        self.assertEqual(balanced["p"], 1.0)
        self.assertLess(imbalanced["p"], 0.05)

    def test_mcnemar_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            ho.mcnemar([True], [True, False])

    def test_mean_sd_reports_the_spread_a_candidate_must_clear(self):
        s = ho.mean_sd([0.9369, 0.9279, 0.9324])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["spread"], 0.0090, places=4)
        self.assertGreater(s["sd"], 0.0)
        self.assertEqual(ho.mean_sd([])["n"], 0)


if __name__ == "__main__":
    unittest.main()
