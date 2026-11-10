"""Tests for the offline teacher path (`mlaya.teacher`). Stdlib unittest, no network.

The LLM calls are injected as plain callables, so the agreement logic, the blind pairing and the
verdict mapping are all exercised without a model. Run:
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

from mlaya import teacher as t


def _act(iteration, candidates, label, *, progressed=False, narrative="s", state=None):
    return {"stage": "action", "iteration": iteration, "candidates": list(candidates),
            "label": label, "progressed": progressed, "narrative": narrative,
            "state": state if state is not None else {"health": 20.0, "hostiles": []}}


class TeachableTest(unittest.TestCase):
    def test_a_two_candidate_action_is_teachable(self):
        self.assertEqual(t._teachable_candidates(_act(1, ["mine_stone", "get_wood"], "get_wood")),
                         ["mine_stone", "get_wood"])

    def test_a_single_candidate_teaches_nothing(self):
        self.assertEqual(t._teachable_candidates(_act(1, ["mine_stone"], "mine_stone")), [])

    def test_a_defer_is_not_an_action_tick(self):
        rec = _act(1, ["mine_stone", "get_wood"], "get_wood")
        rec["stage"] = "defer"
        self.assertEqual(t._teachable_candidates(rec), [])

    def test_an_unknown_label_drops_the_tick(self):
        self.assertEqual(t._teachable_candidates(_act(1, ["mine_stone", "x"], "x")), [])


class ReplayTest(unittest.TestCase):
    def _run(self, records, s2_fn, judge_fn):
        return t.replay(records, s2_fn=s2_fn, judge_fn=judge_fn)

    def test_agreement_needs_no_judge(self):
        calls = []
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("get_wood", "same pick"),
            judge_fn=lambda *a: (calls.append(a), ("A", None, ""))[1])
        self.assertEqual(len(judg), 1)
        self.assertTrue(judg[0]["agree"])
        self.assertIsNone(judg[0]["judge_verdict"])
        self.assertEqual(calls, [])  # the judge is never spent on an agreement

    def test_disagreement_runs_the_judge_blind_and_maps_the_verdict(self):
        seen = {}

        def judge(state, a_label, b_label, candidates):
            seen["pair"] = {a_label, b_label}
            # Side with mine_stone (System 2's pick) whichever slot it landed in.
            return ("A" if a_label == "mine_stone" else "B"), None, "stone"

        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("mine_stone", "stone is better"),
            judge_fn=judge)[0]
        self.assertFalse(judg["agree"])
        self.assertEqual(seen["pair"], {"mine_stone", "get_wood"})
        self.assertEqual(judg["judge_label"], "mine_stone")
        self.assertIn(judg["judge_verdict"], ("A", "B"))

    def test_unsure_carries_no_label(self):
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("mine_stone", ""),
            judge_fn=lambda *a: ("unsure", None, "cannot tell"))[0]
        self.assertEqual(judg["judge_verdict"], "unsure")
        self.assertIsNone(judg["judge_label"])

    def test_neither_names_a_third_allowed_action(self):
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood", "eat_food"], "get_wood")],
            s2_fn=lambda s, c, h: ("mine_stone", ""),
            judge_fn=lambda *a: ("neither", "eat_food", "eat first"))[0]
        self.assertEqual(judg["judge_verdict"], "neither")
        self.assertEqual(judg["judge_label"], "eat_food")

    def test_neither_with_an_out_of_set_action_becomes_unsure(self):
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("mine_stone", ""),
            judge_fn=lambda *a: ("neither", "fly_to_the_moon", ""))[0]
        self.assertEqual(judg["judge_verdict"], "unsure")
        self.assertIsNone(judg["judge_label"])

    def test_s2_off_the_candidate_set_is_no_disagreement(self):
        calls = []
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("not_a_label", ""),
            judge_fn=lambda *a: (calls.append(a), ("A", None, ""))[1])[0]
        self.assertIsNone(judg["s2_label"])
        self.assertFalse(judg["agree"])
        self.assertEqual(calls, [])  # no label to disagree with -> no judge

    def test_reasoning_is_carried_through_when_s2_returns_it(self):
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("get_wood", "same pick", "first I check the pack, then..."),
            judge_fn=lambda *a: ("A", None, ""))[0]
        self.assertEqual(judg["s2_reasoning"], "first I check the pack, then...")

    def test_a_two_tuple_s2_leaves_reasoning_empty(self):
        judg = self._run(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            s2_fn=lambda s, c, h: ("get_wood", "same pick"),
            judge_fn=lambda *a: ("A", None, ""))[0]
        self.assertEqual(judg["s2_reasoning"], "")

    def test_the_blind_pairing_is_stable_for_a_tick(self):
        # Same seed (the tick's narrative) -> same A/B order every time it is replayed.
        rec = _act(7, ["mine_stone", "get_wood"], "get_wood", state={"health": 20.0})
        a1 = t._blind_pair("get_wood", "mine_stone", t.render_narrative(rec) or "")
        a2 = t._blind_pair("get_wood", "mine_stone", t.render_narrative(rec) or "")
        self.assertEqual(a1, a2)


class SampleTest(unittest.TestCase):
    def _run(self, n):
        return [_act(i, ["mine_stone", "get_wood"], "get_wood") for i in range(n)]

    def test_n_at_or_above_the_count_reviews_all(self):
        recs = self._run(5)
        self.assertEqual(t.sample_teachable(recs, 0), {0, 1, 2, 3, 4})
        self.assertEqual(t.sample_teachable(recs, 5), {0, 1, 2, 3, 4})
        self.assertEqual(t.sample_teachable(recs, 9), {0, 1, 2, 3, 4})

    def test_the_sample_spans_the_run_endpoints_included(self):
        recs = self._run(100)
        picks = t.sample_teachable(recs, 5)
        self.assertEqual(len(picks), 5)
        self.assertIn(0, picks)       # first teachable tick
        self.assertIn(99, picks)      # last teachable tick

    def test_only_reviewed_ticks_produce_a_judgment_others_feed_history(self):
        recs = self._run(4)
        seen_history_lengths = []

        def s2(state, candidates, history):
            seen_history_lengths.append(len(history))
            return "mine_stone", ""

        judg = t.replay(recs, s2_fn=s2, judge_fn=lambda *a: ("A", None, ""),
                        review={0, 3})
        self.assertEqual([j["iteration"] for j in judg], [0, 3])
        # Tick 3 is reviewed after ticks 0..2 have all gone into history.
        self.assertEqual(seen_history_lengths, [0, 3])


class FlipTest(unittest.TestCase):
    """The flip test: the same disagreement judged with the blind pair's slots swapped."""

    def _judged(self, judge_fn):
        records = [_act(1, ["mine_stone", "get_wood"], "get_wood")]
        judgments = t.replay(records, s2_fn=lambda s, c, h: ("mine_stone", ""), judge_fn=judge_fn)
        return records, judgments

    def test_an_order_blind_judge_does_not_flip(self):
        # A judge that prefers the action, not the letter, gives the same label both ways.
        judge = lambda s, a, b, c: (("A" if a == "mine_stone" else "B"), None, "")
        records, judgments = self._judged(judge)
        flips = t.flip_verdicts(records, judgments, judge)
        self.assertEqual(len(flips) == 1, True)
        self.assertFalse(flips[0]["flip"])
        self.assertEqual(flips[0]["judge_label"], "mine_stone")
        self.assertEqual(flips[0]["flip_label"], "mine_stone")

    def test_a_slot_bias_flips(self):
        judge = lambda s, a, b, c: ("A", None, "A reads better")
        records, judgments = self._judged(judge)
        flips = t.flip_verdicts(records, judgments, judge)
        self.assertTrue(flips[0]["flip"])
        self.assertEqual({flips[0]["judge_label"], flips[0]["flip_label"]}
                         , {"mine_stone", "get_wood"})

    def test_an_unsure_flip_counts_as_a_flip(self):
        # A judge that cannot tell on the second pass cannot be trusted to have preferred
        # something on the first.
        records, judgments = self._judged(lambda s, a, b, c: ("A", None, ""))
        flips = t.flip_verdicts(records, judgments, lambda s, a, b, c: ("unsure", None, ""))
        self.assertTrue(flips[0]["flip"])
        self.assertIsNone(flips[0]["flip_label"])

    def test_agreements_are_not_rejudged(self):
        records = [_act(1, ["mine_stone", "get_wood"], "get_wood")]
        judgments = t.replay(records, s2_fn=lambda s, c, h: ("get_wood", ""),
                             judge_fn=lambda *a: ("A", None, ""))
        self.assertEqual(t.flip_verdicts(records, judgments, lambda *a: ("A", None, "")), [])


if __name__ == "__main__":
    unittest.main()
