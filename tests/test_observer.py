"""Tests for the parallel observer (`mlaya.observer`). Stdlib unittest, no network.

The observer's LLM call is injected, so its whole logic - what counts as an event worth full
reasoning, the phase debounce against the derived spine's flaps, the budget trim, and the verdict
parse - is exercised offline here. Phase derivation itself (`_phase_of` -> `phase_brief`) has its
own tests in test_perception, so it is patched to a record field to keep these about the observer.
"""
from __future__ import annotations

import unittest
from unittest import mock

from mlaya.config import ObserverConfig
from mlaya.observer import Observer, _parse_verdict


def _rec(stage="action", it=0, phase="diamond_kit", label="get_wood",
         progressed=False, death=None, candidates=None):
    """One decision-log record in the shape the live loop writes (the fields the observer reads)."""
    return {
        "stage": stage, "iteration": it, "label": label,
        "candidates": candidates or [label], "progressed": progressed,
        "status": "ok", "narrative": "", "_phase": phase,
        "state": {"last_death": death} if death is not None else {},
    }


def _observer(cfg=None):
    """An Observer whose llm_fn records every call and returns a canned verdict + telemetry.

    `_phase_of` is patched to read the record's `_phase`, so a test controls the phase directly.
    `prompt_tokens` is driven low by default so the trim never fires unless a test asks for it.
    """
    calls = []
    out = []

    def llm(messages, reason):
        calls.append({"n": len(messages), "reason": reason})
        return ({"phase": None, "agree": True, "risk": None, "note": "ok"},
                {"prompt_tokens": 100})

    obs = Observer(cfg or ObserverConfig(), llm, out.append)
    return obs, calls, out


class EventDetectionTest(unittest.TestCase):
    """Only the moments worth stopping for get a full-reasoning turn; the rest stay routine."""

    def test_the_first_phase_is_an_event(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, out = _observer()
            res = obs.ingest(_rec())
        self.assertEqual(res["event"], "phase_change")
        self.assertTrue(res["reasoned"])

    def test_a_routine_tick_is_not_reasoned(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, _out = _observer()
            obs.ingest(_rec(it=0))                       # first phase - the event
            res = obs.ingest(_rec(it=1))                 # same phase, nothing new
        self.assertIsNone(res["event"])
        self.assertFalse(res["reasoned"])

    def test_an_escalation_record_is_an_event(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, _out = _observer()
            obs.ingest(_rec(it=0))
            res = obs.ingest(_rec(stage="escalation", it=1))
        self.assertEqual(res["event"], "escalation")

    def test_a_fresh_death_cause_is_an_event(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, _out = _observer()
            obs.ingest(_rec(it=0))
            res = obs.ingest(_rec(it=1, death="MineAI tried to swim in lava"))
        self.assertEqual(res["event"], "death")

    def test_the_same_death_cause_fires_once(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, _out = _observer()
            obs.ingest(_rec(it=0))
            obs.ingest(_rec(it=1, death="slain by Drowned"))
            res = obs.ingest(_rec(it=2, death="slain by Drowned"))   # same cause, still there
        self.assertIsNone(res["event"])

    def test_a_repeated_label_without_progress_is_a_stall(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, _out = _observer()
            res = None
            for i in range(7):                           # one label, never progressing
                res = obs.ingest(_rec(it=i, label="get_wood", progressed=False))
        self.assertEqual(res["event"], "stall")


class PhaseDebounceTest(unittest.TestCase):
    """The derived phase spine flaps for a tick; the story must not thrash on a single flap."""

    def test_a_single_tick_flap_does_not_change_phase(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, out = _observer()
            obs.ingest(_rec(it=0, phase="stronghold"))           # commit stronghold
            obs.ingest(_rec(it=1, phase="nether_pearls"))        # one flap - unconfirmed
            res = obs.ingest(_rec(it=2, phase="stronghold"))     # back; flap absorbed
        changes = [o for o in out if o["event"] == "phase_change"]
        self.assertEqual(len(changes), 1)                        # only the very first
        self.assertIsNone(res["event"])

    def test_a_sustained_change_is_accepted(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, out = _observer()
            obs.ingest(_rec(it=0, phase="stronghold"))
            obs.ingest(_rec(it=1, phase="nether_pearls"))        # candidate
            res = obs.ingest(_rec(it=2, phase="nether_pearls"))  # confirmed (phase_confirm=2)
        self.assertEqual(res["event"], "phase_change")
        self.assertEqual([o["event"] for o in out].count("phase_change"), 2)


class SamplingTest(unittest.TestCase):
    """When the tailer is behind, a routine tick may be dropped - an event never is."""

    def test_sampled_skips_routine_but_keeps_events(self) -> None:
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, calls, _out = _observer()
            obs.ingest(_rec(it=0))                                # first phase event (reasoned)
            skipped = obs.ingest(_rec(it=1), sampled=True)       # routine under pressure
            kept = obs.ingest(_rec(stage="escalation", it=2), sampled=True)
        self.assertIsNone(skipped)                               # dropped, no verdict
        self.assertEqual(kept["event"], "escalation")           # event survives sampling


class TrimTest(unittest.TestCase):
    """At the input budget the oldest turns fold into a summary; the system prefix is untouched."""

    def test_the_story_folds_when_it_passes_the_budget(self) -> None:
        cfg = ObserverConfig(input_budget=1000)

        def llm(messages, reason):
            return ({"phase": None, "agree": True, "risk": None, "note": "ok"},
                    {"prompt_tokens": 5000})                     # over budget every tick

        out = []
        obs = Observer(cfg, llm, out.append)
        head = obs.messages[0]
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            for i in range(60):
                obs.ingest(_rec(it=i))
        self.assertIs(obs.messages[0], head)                     # prefix preserved byte-for-byte
        self.assertEqual(obs.messages[0]["role"], "system")
        self.assertLess(len(obs.messages), 60)                   # older turns were folded away


class ParseVerdictTest(unittest.TestCase):
    """The reply is read as a verdict however the model dressed it - or degrades to a readable note."""

    def test_a_clean_json_object_is_read(self) -> None:
        v = _parse_verdict('{"phase": "stronghold", "agree": false, "risk": "loop", "note": "x"}')
        self.assertEqual(v["agree"], False)
        self.assertEqual(v["risk"], "loop")

    def test_reasoning_tags_are_stripped(self) -> None:
        v = _parse_verdict('<think>let me weigh it</think>{"agree": true, "note": "ok"}')
        self.assertEqual(v["agree"], True)

    def test_a_reasoning_leak_without_json_degrades_gracefully(self) -> None:
        # The smoke test saw /no_think ticks leak bare reasoning with no JSON and no closing brace.
        leak = "We need answer user's JSON observation. Need analyze the situation and"
        v = _parse_verdict(leak)
        self.assertIsNone(v["agree"])
        self.assertTrue(v["note"])                               # the text is kept, not lost


if __name__ == "__main__":
    unittest.main()
