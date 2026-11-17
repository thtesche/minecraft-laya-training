"""Tests for the offline teacher path (`mlaya.teacher`). Stdlib unittest, no network.

The LLM calls are injected as plain callables, so the agreement logic, the blind pairing and the
verdict mapping are all exercised without a model. Run:
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import types
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

        def judge(state, a_label, b_label, candidates, history):
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


class FlipMergeTest(unittest.TestCase):
    """The flip test's verdict has to land on the judgment the dataset reads, not a side file."""

    def test_the_flip_result_is_folded_into_the_judgment(self) -> None:
        judgments = [{"iteration": 3, "judge_label": "mine_stone"},
                     {"iteration": 9, "judge_label": "mine_stone"}]
        flips = [{"iteration": 3, "flip_verdict": "A", "flip_label": "get_wood", "flip": True},
                 {"iteration": 9, "flip_verdict": "B", "flip_label": "mine_stone", "flip": False}]
        merged = t.apply_flips(judgments, flips)
        self.assertTrue(merged[0]["flip"])
        self.assertEqual(merged[0]["flip_label"], "get_wood")
        self.assertFalse(merged[1]["flip"])

    def test_a_disagreement_the_flip_test_never_ran_stays_unmarked(self) -> None:
        merged = t.apply_flips([{"iteration": 3, "judge_label": "mine_stone"}], [])
        self.assertNotIn("flip", merged[0])


class FlipTest(unittest.TestCase):
    """The flip test: the same disagreement judged with the blind pair's slots swapped."""

    def _judged(self, judge_fn):
        records = [_act(1, ["mine_stone", "get_wood"], "get_wood")]
        judgments = t.replay(records, s2_fn=lambda s, c, h: ("mine_stone", ""), judge_fn=judge_fn)
        return records, judgments

    def test_an_order_blind_judge_does_not_flip(self):
        # A judge that prefers the action, not the letter, gives the same label both ways.
        judge = lambda s, a, b, c, h: (("A" if a == "mine_stone" else "B"), None, "")
        records, judgments = self._judged(judge)
        flips = t.flip_verdicts(records, judgments, judge)
        self.assertEqual(len(flips) == 1, True)
        self.assertFalse(flips[0]["flip"])
        self.assertEqual(flips[0]["judge_label"], "mine_stone")
        self.assertEqual(flips[0]["flip_label"], "mine_stone")

    def test_a_slot_bias_flips(self):
        judge = lambda s, a, b, c, h: ("A", None, "A reads better")
        records, judgments = self._judged(judge)
        flips = t.flip_verdicts(records, judgments, judge)
        self.assertTrue(flips[0]["flip"])
        self.assertEqual({flips[0]["judge_label"], flips[0]["flip_label"]}
                         , {"mine_stone", "get_wood"})

    def test_an_unsure_flip_counts_as_a_flip(self):
        # A judge that cannot tell on the second pass cannot be trusted to have preferred
        # something on the first.
        records, judgments = self._judged(lambda s, a, b, c, h: ("A", None, ""))
        flips = t.flip_verdicts(records, judgments, lambda s, a, b, c, h: ("unsure", None, ""))
        self.assertTrue(flips[0]["flip"])
        self.assertIsNone(flips[0]["flip_label"])

    def test_agreements_are_not_rejudged(self):
        records = [_act(1, ["mine_stone", "get_wood"], "get_wood")]
        judgments = t.replay(records, s2_fn=lambda s, c, h: ("get_wood", ""),
                             judge_fn=lambda *a: ("A", None, ""))
        self.assertEqual(t.flip_verdicts(records, judgments, lambda *a: ("A", None, "")), [])


class JudgeNoThinkTest(unittest.TestCase):
    """`--judge-no-think` must reach the wire as OpenRouter `reasoning: {enabled: false}`.

    The local Qwen reviewer has no thinking lever (measured: the Strata server ignores
    `/no_think`, `thinking_budget` and `enable_thinking`), so the reviewer must not send one.
    """

    def _fake_openai(self, content: str):
        import openai
        captured: dict = {}

        class FakeClient:
            def __init__(self, *a, **k):
                self._content = content
                self._captured = captured
                self.chat = types.SimpleNamespace(completions=self)

            def create(self, **kwargs):
                self._captured["kwargs"] = kwargs
                message = type("Msg", (), {"content": self._content, "reasoning_content": None})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        original = getattr(openai, "OpenAI", None)
        openai.OpenAI = lambda *a, **k: FakeClient()
        return captured, original

    def _restore(self, original):
        import openai
        if original is not None:
            openai.OpenAI = original

    def test_the_judge_disables_reasoning_when_asked(self):
        from mlaya.config import JudgeConfig
        captured, orig = self._fake_openai('{"verdict": "A", "note": "ok"}')
        try:
            judge = t.make_judge_fn(JudgeConfig(), "key", no_think=True)
            judge({"health": 20.0}, "get_wood", "mine_stone", ["get_wood", "mine_stone"], [])
            self.assertEqual(captured["kwargs"].get("extra_body"), {"reasoning": {"enabled": False}})
            self.assertNotIn("reasoning", captured["kwargs"])

            judge2 = t.make_judge_fn(JudgeConfig(), "key")
            judge2({"health": 20.0}, "get_wood", "mine_stone", ["get_wood", "mine_stone"], [])
            self.assertNotIn("reasoning", captured["kwargs"])
        finally:
            self._restore(orig)

    def test_the_reviewer_sends_no_think_directive(self):
        from mlaya.brain import review_with_system2
        from mlaya.config import LLMConfig
        captured, orig = self._fake_openai('{"action": "get_wood", "note": "ok"}')
        try:
            review_with_system2(LLMConfig(), "key", {"health": 20.0}, [], ["get_wood", "mine_stone"])
            user = captured["kwargs"]["messages"][-1]["content"]
            self.assertNotIn("no_think", user)
            self.assertNotIn("thinking_budget", captured["kwargs"])
        finally:
            self._restore(orig)


class ObserverTeacherTest(unittest.TestCase):
    """`replay_from_observer`: the live observer verdict is System 2, the picker names the
    alternative. No network - both LLMs are injected callables."""

    def _verdict(self, iteration, agree, better=None, note="loop", risk="loop: x"):
        return {"iteration": iteration, "verdict": {"phase": "prep", "agree": agree,
                                                    "risk": risk, "note": note, "better": better}}

    def test_observer_agreement_needs_no_picker(self):
        picker_calls = []
        judg = t.replay_from_observer(
            [_act(1, ["mine_stone", "get_wood"], "get_wood")],
            [self._verdict(1, True)],
            judge_fn=lambda *a: ("unsure", None, ""),
            picker_fn=lambda *a: picker_calls.append(a) or (None, ""))
        self.assertEqual(len(judg), 1)
        self.assertTrue(judg[0]["agree"])
        self.assertEqual(judg[0]["s2_label"], "get_wood")
        self.assertEqual(picker_calls, [])

    def test_a_better_field_is_adjudicated_by_the_blind_judge(self):
        seen = {}

        def judge(state, a_label, b_label, candidates, history):
            seen["pair"] = {a_label, b_label}
            return ("A" if a_label == "craft_bed" else "B"), None, "bed first"

        judg = t.replay_from_observer(
            [_act(1, ["surface_up", "craft_bed"], "surface_up")],
            [self._verdict(1, False, better="craft_bed")],
            judge_fn=judge, picker_fn=None)
        self.assertEqual(seen["pair"], {"surface_up", "craft_bed"})
        self.assertFalse(judg[0]["agree"])
        self.assertEqual(judg[0]["s2_label"], "craft_bed")
        self.assertEqual(judg[0]["teacher_source"], "observer")
        self.assertEqual(judg[0]["judge_label"], "craft_bed")

    def test_an_invalid_better_field_falls_back_to_the_picker(self):
        def picker(state, order, critique, history, blocked_goal=None):
            return "craft_bed", "bed first"

        judg = t.replay_from_observer(
            [_act(1, ["surface_up", "craft_bed"], "surface_up")],
            [self._verdict(1, False, better="mine_diamond")],  # not a candidate
            judge_fn=lambda *a: ("unsure", None, ""), picker_fn=picker)
        self.assertEqual(judg[0]["teacher_source"], "picker")
        self.assertEqual(judg[0]["judge_label"], "craft_bed")
        self.assertEqual(judg[0]["judge_verdict"], "pick")

    def test_a_picker_pick_that_flips_with_the_order_carries_no_label(self):
        # The picker reads the slot, not the action: it always names the first candidate it is shown.
        def picker(state, order, critique, history, blocked_goal=None):
            return order[0], "first listed"

        judg = t.replay_from_observer(
            [_act(1, ["mine_stone", "get_wood", "craft_table"], "mine_stone")],
            [self._verdict(1, False)],
            judge_fn=lambda *a: ("unsure", None, ""), picker_fn=picker)
        self.assertIsNone(judg[0]["s2_label"])
        self.assertIsNone(judg[0]["judge_label"])

    def test_a_picker_pick_of_layas_own_choice_is_no_disagreement(self):
        def picker(state, order, critique, history, blocked_goal=None):
            return "mine_stone", "laya was right"

        judg = t.replay_from_observer(
            [_act(1, ["mine_stone", "get_wood"], "mine_stone")],
            [self._verdict(1, False, note="detour")],
            judge_fn=lambda *a: ("unsure", None, ""), picker_fn=picker)
        self.assertIsNone(judg[0]["judge_label"])

    def test_repeats_count_the_same_counter_move_across_the_whole_run(self):
        # The repetition gate counts every verdict the observer wrote, including ticks the teacher
        # path never reviewed because the filter left only one candidate there.
        records = [_act(1, ["mine_stone", "get_wood"], "mine_stone"),
                   _act(2, ["mine_stone", "get_wood"], "mine_stone"),
                   _act(3, ["mine_stone"], "mine_stone"),
                   _act(4, ["mine_stone", "get_wood"], "get_wood")]
        verdicts = [self._verdict(1, False, better="get_wood"),
                    self._verdict(2, False, better="get_wood"),
                    self._verdict(3, False, better="get_wood"),   # single-candidate tick, unreviewed
                    self._verdict(4, True)]
        judg = t.replay_from_observer(records, verdicts,
                                      judge_fn=lambda *a: ("unsure", None, ""), picker_fn=None)
        self.assertEqual([j["iteration"] for j in judg], [1, 2, 4])
        self.assertEqual([j["better_repeats"] for j in judg], [3, 3, 0])
        self.assertFalse(judg[0]["agree"])

    def test_uncovered_ticks_get_no_judgment(self):
        judg = t.replay_from_observer(
            [_act(1, ["mine_stone", "get_wood"], "get_wood"),
             _act(2, ["mine_stone", "get_wood"], "get_wood")],
            [self._verdict(1, True)],
            judge_fn=lambda *a: ("unsure", None, ""), picker_fn=lambda *a: (None, ""))
        self.assertEqual([j["iteration"] for j in judg], [1])

    def test_the_pick_order_is_seeded_and_stable(self):
        cands = ["mine_stone", "get_wood", "craft_table", "hunt_food"]
        self.assertEqual(t._pick_order(cands, "seed", 0), t._pick_order(cands, "seed", 0))
        self.assertEqual(sorted(t._pick_order(cands, "seed", 0)), sorted(cands))
        self.assertNotEqual(t._pick_order(cands, "seed", 0), t._pick_order(cands, "seed", 1))

    def _fake_openai(self, responses):
        import openai
        captured: list = []

        class FakeClient:
            def __init__(self, *a, **k):
                self.i = 0
                self.chat = types.SimpleNamespace(completions=self)

            def create(self, **kwargs):
                captured.append(kwargs)
                content = responses[min(self.i, len(responses) - 1)]
                self.i += 1
                message = type("Msg", (), {"content": content})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        original = getattr(openai, "OpenAI", None)
        openai.OpenAI = lambda *a, **k: FakeClient()
        return captured, original

    def test_a_truncated_picker_reply_is_retried_without_reasoning(self):
        # Live, run253: Nemotron's thinking chain ate the whole max_tokens budget and returned
        # content null - four disagreements were lost to truncation, not to instability.
        from mlaya.config import JudgeConfig
        import openai
        captured, orig = self._fake_openai([None, '{"action": "craft_bed", "note": "bed first"}'])
        try:
            pick = t.make_picker_fn(JudgeConfig(), "key")
            action, _ = pick({"health": 20.0}, ["craft_bed", "mine_stone"], "loop", [])
            self.assertEqual(action, "craft_bed")
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[1].get("extra_body"), {"reasoning": {"enabled": False}})
        finally:
            if orig is not None:
                openai.OpenAI = orig

    def test_the_picker_forwards_the_blocked_goal_into_its_brief(self):
        # A gated end-state action (light_portal) is not among the candidates; the picker is told
        # why so it points at the candidate that clears the need instead of reading a loop.
        import json
        from mlaya.config import JudgeConfig
        import openai
        captured, orig = self._fake_openai(['{"action": "mine_diamond", "note": "get the spare"}'])
        try:
            pick = t.make_picker_fn(JudgeConfig(), "key")
            blocked = {"action": "light_portal",
                       "needs": ["spare diamond pickaxe missing (1/2)"]}
            action, _ = pick({"health": 20.0}, ["cook_food", "mine_diamond"], "loop", [], blocked)
            self.assertEqual(action, "mine_diamond")
            brief = json.loads(captured[0]["messages"][1]["content"])
            self.assertEqual(brief["blocked_goal"]["action"], "light_portal")
            self.assertIn("spare diamond pickaxe missing (1/2)", brief["blocked_goal"]["needs"])
        finally:
            if orig is not None:
                openai.OpenAI = orig

    def test_the_picker_brief_omits_blocked_goal_when_none(self):
        import json
        from mlaya.config import JudgeConfig
        import openai
        captured, orig = self._fake_openai(['{"action": "craft_bed"}'])
        try:
            pick = t.make_picker_fn(JudgeConfig(), "key")
            pick({"health": 20.0}, ["craft_bed", "mine_stone"], "loop", [])
            brief = json.loads(captured[0]["messages"][1]["content"])
            self.assertNotIn("blocked_goal", brief)
        finally:
            if orig is not None:
                openai.OpenAI = orig

    def test_an_already_reasoning_off_picker_does_not_retry(self):
        from mlaya.config import JudgeConfig
        import openai
        captured = []

        class FakeClient:
            def __init__(self, *a, **k):
                self.chat = types.SimpleNamespace(completions=self)

            def create(self, **kwargs):
                captured.append(kwargs)
                message = type("Msg", (), {"content": '{"action": "craft_bed"}'})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        orig = getattr(openai, "OpenAI", None)
        openai.OpenAI = lambda *a, **k: FakeClient()
        try:
            pick = t.make_picker_fn(JudgeConfig(), "key", no_think=True)
            action, _ = pick({"health": 20.0}, ["craft_bed", "mine_stone"], "loop", [])
            self.assertEqual(action, "craft_bed")
            self.assertEqual(len(captured), 1)
        finally:
            if orig is not None:
                openai.OpenAI = orig


if __name__ == "__main__":
    unittest.main()
