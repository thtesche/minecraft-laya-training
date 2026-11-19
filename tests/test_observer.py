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
from mlaya.observer import Observer, _gear_line, _parse_verdict, system_prefix


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

    def test_the_fold_keeps_a_model_written_checkpoint(self) -> None:
        # The folded turns are replaced by the model's own "story so far", not a char-truncated
        # digest - the narrative thread the story observer lives on survives the fold.
        cfg = ObserverConfig(input_budget=1000)

        def llm(messages, reason):
            if messages[-1]["content"].startswith("STOP judging"):
                return ({"story": "passed diamond_kit; in prepare_nether; banking blaze powder"}, {})
            return ({"phase": None, "agree": True, "risk": None, "note": "ok"},
                    {"prompt_tokens": 5000})

        out = []
        obs = Observer(cfg, llm, out.append)
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            for i in range(60):
                obs.ingest(_rec(it=i))
        folded = obs.messages[1]
        self.assertEqual(folded["role"], "user")
        self.assertIn("STORY SO FAR", folded["content"])
        self.assertIn("banking blaze powder", folded["content"])

    def test_the_fold_falls_back_when_the_checkpoint_is_empty(self) -> None:
        # A checkpoint that returns nothing usable must not lose the thread: fall back to the
        # char-truncated digest of the dropped verdicts, never an empty fold.
        cfg = ObserverConfig(input_budget=1000)

        def llm(messages, reason):
            if messages[-1]["content"].startswith("STOP judging"):
                return ({"phase": None}, {})                     # no story, no note
            return ({"phase": None, "agree": True, "risk": None, "note": "ok"},
                    {"prompt_tokens": 5000})

        out = []
        obs = Observer(cfg, llm, out.append)
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            for i in range(60):
                obs.ingest(_rec(it=i))
        folded = obs.messages[1]
        self.assertIn("STORY SO FAR", folded["content"])
        self.assertIn("ok", folded["content"])                   # digest of the dropped verdicts


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


class PrefixTest(unittest.TestCase):
    """What the judge is taught before it ever sees a tick - the prefix is where a false flag
    either starts or stops."""

    def test_the_shield_is_taught_as_kit_work(self) -> None:
        # Live, run247: `risk="unnecessary detour", note="Shield is not part of diamond kit."` -
        # the phase line named the kit without the shield, so the judge invented a detour for the
        # one item the departure gate requires.
        content = system_prefix()["content"]
        self.assertIn("shield", content)
        self.assertIn("off-hand", content)

    def test_the_gear_field_is_taught(self) -> None:
        content = system_prefix()["content"]
        self.assertIn("`gear`", content)
        self.assertIn("durability", content)

    def test_the_better_field_is_taught(self) -> None:
        # The observer-as-teacher path needs a concrete counter-move, not just a critique:
        # a disagreement must name the better candidate or null.
        content = system_prefix()["content"]
        self.assertIn("`better`", content)
        self.assertIn("candidates", content)

    def test_the_blocked_goal_field_is_taught(self) -> None:
        # The gated-milestone hint has to tell the observer not to demand a withheld end state and
        # never to name it in `better`, or it reads an unlit gate as a loop (the run258 loop).
        content = system_prefix()["content"]
        self.assertIn("`blocked_goal`", content)
        self.assertIn("`needs`", content)

    def test_the_night_hold_before_the_crossing_is_taught(self) -> None:
        # run279 it0-16: the bot held at the frame for the night sleep that pins the respawn before
        # the first crossing, and the judge called that wait idle on every tick, because no phase
        # line named the sleep as the crossing's precondition.
        content = system_prefix()["content"]
        self.assertIn("end_dragon", content)
        self.assertIn("respawn", content)
        self.assertIn("night", content)


class GearLineTest(unittest.TestCase):
    """The judge reads the kit's real state - the shield in the off-hand, the armour worn, a tool
    wearing out - instead of inferring it from the label stream (the run247 shield false-flag)."""

    STATE = {
        "armor": {
            "helmet": {"tier": "diamond", "item": "diamond_helmet", "left": 330, "max": 363},
            "chestplate": {"tier": "diamond", "item": "diamond_chestplate",
                           "left": 680, "max": 680},
        },
        "tool_meta": {
            "pickaxe": {"tier": "diamond", "item": "diamond_pickaxe", "left": 300, "max": 1561},
            "sword": {"tier": "iron", "item": "iron_sword", "left": 100, "max": 250},
        },
        "offhand": "shield",
    }

    def test_worn_gear_and_durability_are_read(self) -> None:
        g = _gear_line(self.STATE)
        self.assertEqual(g["offhand"], "shield")
        self.assertEqual(g["armor"]["helmet"], "diamond 91%")       # 330/363
        self.assertEqual(g["armor"]["chestplate"], "diamond 100%")
        self.assertEqual(g["tools"]["pickaxe"], "diamond 19%")      # 300/1561
        self.assertEqual(g["tools"]["sword"], "iron 40%")           # 100/250

    def test_a_missing_shield_shows_as_null_not_omitted(self) -> None:
        # The other half of the fix: with armour on but no shield the judge should see the gap,
        # not an absence it has to guess at.
        g = _gear_line({"armor": self.STATE["armor"], "offhand": None})
        self.assertIn("offhand", g)
        self.assertIsNone(g["offhand"])

    def test_a_stateless_record_adds_no_gear(self) -> None:
        self.assertIsNone(_gear_line({}))
        self.assertIsNone(_gear_line(None))

    def test_the_observation_carries_the_gear(self) -> None:
        rec = {"iteration": 5, "stage": "action", "label": "fill_bucket",
               "candidates": ["fill_bucket"], "progressed": True, "status": "ok",
               "narrative": "", "state": self.STATE}
        with mock.patch("mlaya.observer._phase_of", lambda r: "prepare_nether"):
            payload = Observer.observation(rec, None)
        self.assertEqual(payload["gear"]["offhand"], "shield")
        self.assertEqual(payload["gear"]["armor"]["helmet"], "diamond 91%")

    def test_the_observation_carries_the_blocked_goal(self) -> None:
        # The loop's lighting hint rides into the payload so the observer sees why `light_portal`
        # is not a candidate this tick, instead of inferring a loop from its absence.
        rec = {"iteration": 5, "stage": "action", "label": "cook_food",
               "candidates": ["cook_food", "mine_diamond"], "progressed": True, "status": "ok",
               "narrative": "", "state": self.STATE,
               "blocked_goal": {"action": "light_portal",
                                "needs": ["spare diamond pickaxe missing (1/2)"]}}
        with mock.patch("mlaya.observer._phase_of", lambda r: "prepare_nether"):
            payload = Observer.observation(rec, None)
        self.assertEqual(payload["blocked_goal"]["action"], "light_portal")
        self.assertIn("spare diamond pickaxe missing (1/2)",
                      payload["blocked_goal"]["needs"])

    def test_no_blocked_goal_key_when_absent(self) -> None:
        rec = {"iteration": 5, "stage": "action", "label": "cook_food",
               "candidates": ["cook_food"], "progressed": True, "status": "ok",
               "narrative": "", "state": self.STATE}
        with mock.patch("mlaya.observer._phase_of", lambda r: "prepare_nether"):
            payload = Observer.observation(rec, None)
        self.assertNotIn("blocked_goal", payload)

    def test_the_situation_slice_keeps_the_surface_proof(self) -> None:
        # The judge's `situation` is the narrative cut to 360 characters. run252 it40 read
        # "Already at surface" from the altitude band alone; the longer proof sentence has to
        # still fit inside the slice, or the loop flag returns.
        rec = {"iteration": 40, "stage": "action", "label": "surface_up",
               "candidates": ["surface_up"], "progressed": False, "status": "succeeded",
               "narrative": ("You are a Minecraft player preparing for the nether. Milestones: "
                            "no gate burns yet, no bed is laid, no blaze powder is banked, the "
                            "end portal frame holds 0 of 12 eyes. In the overworld. Health is full. "
                            "Hunger 16/20, not regenerating. It is daytime. At surface altitude "
                            "(standing on solid floor, open sky not confirmed). Inventory: empty.")}
        payload = Observer.observation(rec, None)
        self.assertIn("open sky not confirmed", payload["situation"])
        self.assertLessEqual(len(payload["situation"]), 360)

    def test_the_logged_verdict_carries_the_gear_too(self) -> None:
        # The payload is not archived - only the verdict and its telemetry - so the record the
        # Phase-3 join reads has to carry the gear the judge was shown, or a gear-aware judgment
        # can never be audited against the world it was made from.
        rec = {"iteration": 5, "stage": "action", "label": "fill_bucket",
               "candidates": ["fill_bucket"], "progressed": True, "status": "ok",
               "narrative": "", "state": self.STATE, "_phase": "prepare_nether"}
        with mock.patch("mlaya.observer._phase_of", lambda r: r.get("_phase")):
            obs, _calls, out = _observer()
            res = obs.ingest(rec)
        self.assertEqual(res["gear"]["tools"]["sword"], "iron 40%")


if __name__ == "__main__":
    unittest.main()
