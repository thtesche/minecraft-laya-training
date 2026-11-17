"""Offline teacher path: replay an archived run log, ask System 2 as a teacher, adjudicate.

The verifier (`verify.progressed`) stays the grounded primary truth of the whole pipeline. The
teacher is *opinion*, not a rollout, and it fills only the gaps the verifier does not cover: the
disputed ticks and the no-progress ticks a verifier-only dataset throws away. A tick where the
world showed real progress is never overruled by an LLM's opinion (that adjudication lives in
`dataset.build_rows`; here we only record the opinions).

Per action tick with >= 2 eligible labels (only `intent`; the `in_danger` gate is untouched):

  1. System 1 (Laya) already chose a label at run time - it is in the log (`label`).
  2. System 2 is asked the same question over the same candidates, with the same state. If it
     agrees with Laya there is nothing to adjudicate.
  3. On disagreement a third model, the judge, sees both actions *blind* - it is never told
     which came from S1 and which from S2, so no authority bias creeps in - and rules: A better,
     B better, neither (a third allowed action is better), or unsure. Only a clear, confident
     verdict carries a label forward.

Everything runs on the archived log, so there are no latency or cost problems in the live loop
and the loop (`graph.py`) is not touched. The per-tick judgments are written to a side-file
(`runs/history/teacher-runNN.jsonl`) that `dataset.build_rows` reads as an extra label source.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any, Callable

from .actions import CRITERIA
from .brain import review_payload, review_with_system2
from .config import JudgeConfig, LLMConfig
from .dataset import _narrative_hash, render_narrative

# A teacher judgment is produced only for an action tick the head could actually learn from: more
# than one eligible label, and a label Laya really chose among them. The `in_danger` gate and its
# floor/ceiling heuristic in `dataset.py` are deliberately left alone - this path is about the
# intent choice only.
# System 2 returns its pick, a short reason, and optionally its thinking. The thinking is a third
# element the live wiring provides; a test fake may return just `(label, note)` and `replay` fills
# the reasoning with "".
S2Fn = Callable[[dict, list[str], list[dict]], tuple]
# The judge is handed the same trajectory System 2 reasoned over (the prior action ticks), so the
# blind A/B comparison is made on the same context the pick it is judging was made on.
JudgeFn = Callable[[dict, str, str, list[str], list[dict]], tuple[str | None, str | None, str]]


def _teachable_candidates(rec: dict) -> list[str]:
    """The eligible labels of an action tick, or [] when it teaches no intent.

    Mirrors `dataset.build_rows`' intent gate at the coarse level the teacher needs: known
    labels, Laya's choice among them, at least two of them. The finer verifier-side filters
    (ping-pong windows, walk-only stows) stay in the dataset builder, where the grounded primary
    rows are made; the teacher only needs to know this tick carried a real choice.
    """
    if rec.get("stage") != "action":
        return []
    candidates = [c for c in (rec.get("candidates") or []) if c in CRITERIA]
    label = rec.get("label")
    if len(candidates) >= 2 and label in candidates:
        return candidates
    return []


def _blind_pair(laya_label: str, s2_label: str, seed: str) -> tuple[str, str, bool]:
    """Order the two actions as (A, B) with a seeded coin, so the judge cannot read authority.

    Returns `(a_label, b_label, a_is_laya)`. The seed is the tick's narrative hash, so the same
    archived tick always presents the same way - the side-file stays auditable and a re-run is
    reproducible.
    """
    if random.Random(seed).random() < 0.5:
        return laya_label, s2_label, True
    return s2_label, laya_label, False


def judge_tick(rec: dict, laya_label: str, s2_label: str, candidates: list[str],
               judge_fn: JudgeFn, history: list[dict] | None = None, swap: bool = False) -> dict:
    """Run the blind judge for one disagreement and map its verdict back to a concrete label.

    `history` is the run up to this tick - the same trajectory System 2 read when it made the pick
    being judged - so the judge weighs the two actions against the same context, not the bare state.
    `swap` puts the same two actions in the opposite slots. That is the flip test: a verdict that
    survives the swap is a judgement about the actions, a verdict that flips is a reaction to which
    letter they were handed under.
    """
    narrative = render_narrative(rec) or ""
    a_label, b_label, a_is_laya = _blind_pair(laya_label, s2_label, narrative)
    if swap:
        a_label, b_label = b_label, a_label
    verdict, better, note = judge_fn(rec.get("state") or {}, a_label, b_label,
                                     candidates, history or [])
    label: str | None = None
    if verdict == "A":
        label = a_label
    elif verdict == "B":
        label = b_label
    elif verdict == "neither":
        # A third action the judge names has to be one the filter actually offered; otherwise it
        # is not a label the head can be taught to pick here.
        label = better if better in candidates else None
        if label is None:
            verdict = "unsure"
    # "unsure" (and a malformed verdict) carry no label forward.
    if verdict not in ("A", "B", "neither"):
        verdict = "unsure"
        label = None
    return {"judge_verdict": verdict, "judge_label": label, "judge_note": (note or "")[:300]}


def sample_teachable(records: list[dict], n: int) -> set[Any]:
    """The iterations of `n` teachable ticks spread evenly across the run (all of them if n<=0).

    One System-2 call takes ~2 minutes against the local reasoning server, so a rich run's few
    hundred teachable ticks is many hours. A representative sample - the same two-model question on
    ticks taken from the whole trajectory, not just a short run's handful - gives the quality read
    for a fraction of the calls. Even spacing (endpoints included) keeps the sample from clustering
    in whichever phase happened to carry the most choices.
    """
    teach = [r.get("iteration") for r in records if _teachable_candidates(r)]
    if n <= 0 or n >= len(teach):
        return set(teach)
    if n == 1:
        return {teach[0]}
    picks = {teach[round(i * (len(teach) - 1) / (n - 1))] for i in range(n)}
    return picks


def replay(records: list[dict], *, s2_fn: S2Fn, judge_fn: JudgeFn,
           review: set[Any] | None = None) -> list[dict]:
    """Produce one judgment per teachable action tick (pure; the LLM calls are injected).

    `s2_fn(state, candidates, history)` returns `(label, note)` - System 2's pick among the
    candidates (or None) and its reason. `judge_fn(state, a_label, b_label, candidates, history)`
    returns `(verdict, better_action, note)` - the judge reads the same trajectory S2 did. Keeping
    both injectable is what lets the agreement logic, the blind pairing and the verdict mapping be
    tested with no network.

    `review` restricts the LLM calls to the teachable ticks whose iteration it names (from
    `sample_teachable`); the ticks left out still feed the history S2 reads for context, they are
    just not themselves reviewed. `None` reviews every teachable tick.
    """
    out: list[dict] = []
    history: list[dict] = []
    for rec in records:
        if rec.get("stage") not in ("action", "defer"):
            continue
        candidates = _teachable_candidates(rec)
        if not candidates or (review is not None and rec.get("iteration") not in review):
            # A tick that is not reviewed - non-teachable, or outside the sample - still belongs
            # to the history S2 reads for context.
            if rec.get("stage") == "action":
                history.append(rec)
            continue
        narrative = render_narrative(rec) or ""
        laya_label = rec.get("label")
        res = s2_fn(rec.get("state") or {}, candidates, history)
        s2_label, s2_note = res[0], res[1]
        s2_reasoning = res[2] if len(res) > 2 else ""
        if s2_label not in candidates:
            s2_label = None
        agree = s2_label is not None and s2_label == laya_label
        entry: dict[str, Any] = {
            "iteration": rec.get("iteration"),
            "narrative_hash": _narrative_hash(narrative),
            "candidates": candidates,
            "laya_label": laya_label,
            "s2_label": s2_label,
            "agree": agree,
            "s2_note": (s2_note or "")[:300],
            # Six of the thirty sample ticks ran the thinking to the 4096-token output cap, so a
            # 4000-character cut would hide the part where the model actually commits to a pick -
            # the part we are reading the reasoning for.
            "s2_reasoning": (s2_reasoning or "")[:8000],
            "progressed": bool(rec.get("progressed")),
            "judge_verdict": None,
            "judge_label": None,
            "judge_note": "",
        }
        # The judge is spent only where there is a real disagreement to settle. It reads the same
        # history S2 just read (the prior action ticks), passed before this tick joins it below.
        if not agree and s2_label is not None:
            entry.update(judge_tick(rec, laya_label, s2_label, candidates, judge_fn, history))
        out.append(entry)
        history.append(rec)
    return out


def flip_verdicts(records: list[dict], judgments: list[dict], judge_fn: JudgeFn) -> list[dict]:
    """Re-judge every recorded disagreement with the blind pair's A/B slots swapped.

    The judge is blind by construction, but blindness is only a claim until it is tested: if the
    verdict flips when the same two actions are handed over in the opposite order, the verdict was
    about the slot, not the actions, and the label it produced is noise the dataset should not
    learn from. This costs one judge call per disagreement and nothing else - System 2 is not asked
    again, its pick is read from the side-file.

    Returns one entry per disagreement: the original verdict and label, the swapped verdict and
    label, and `flip` - whether the two entries name different actions. A verdict that comes back
    "unsure" on either side counts as a flip, since an unsure judge cannot be trusted to have
    preferred one of the two in the first place.
    """
    by_iteration = {r.get("iteration"): r for r in records}
    # The history each tick's judge saw, rebuilt exactly as `replay` accumulates it: every prior
    # action tick, snapshotted before the tick itself joins. The swap has to be judged on the same
    # context as the original, or a verdict change would conflate the slot flip with a context change.
    history_by_iteration: dict[Any, list[dict]] = {}
    hist: list[dict] = []
    for rec in records:
        if rec.get("stage") != "action":
            continue
        history_by_iteration[rec.get("iteration")] = list(hist)
        hist.append(rec)
    out: list[dict] = []
    for j in judgments:
        if j.get("s2_label") is None or j.get("agree"):
            continue
        rec = by_iteration.get(j.get("iteration"))
        if rec is None:
            continue
        flipped = judge_tick(rec, j["laya_label"], j["s2_label"], j.get("candidates") or [],
                             judge_fn, history_by_iteration.get(j.get("iteration")), swap=True)
        original = j.get("judge_label")
        new = flipped.get("judge_label")
        out.append({
            "iteration": j.get("iteration"),
            "narrative_hash": j.get("narrative_hash"),
            "laya_label": j.get("laya_label"),
            "s2_label": j.get("s2_label"),
            "progressed": j.get("progressed"),
            "judge_verdict": j.get("judge_verdict"),
            "judge_label": original,
            "flip_verdict": flipped.get("judge_verdict"),
            "flip_label": new,
            "flip_note": flipped.get("judge_note", ""),
            "flip": original != new,
        })
    return out


# --- live LLM wiring (used by the CLI; replaced by fakes in tests) ---------------------------

def make_system2_fn(cfg: LLMConfig, api_key: str | None) -> S2Fn:
    """A `replay` S2 callable backed by `brain.review_with_system2`.

    The review framing is neutral by design: no impasse ("your controller is stuck"), no failure
    or trajectory brief, and Laya's own choice is *not* handed to the model. S2 picks among the
    same candidates Laya weighed, over the same state - so the two systems answer the same
    question and the picks compare cleanly. The third return value is the model's thinking, carried
    through so the side-file records *how* S2 decided, not only what it chose.
    """
    def s2(state: dict, candidates: list[str],
           history: list[dict]) -> tuple[str | None, str, str]:
        out = review_with_system2(cfg, api_key, state, history, list(candidates))
        if not isinstance(out, dict):
            return None, "", ""
        if out.get("error"):
            return None, f"error: {out['error']}", str(out.get("reasoning") or "")
        label = out.get("action")
        return ((label if label in candidates else None),
                str(out.get("note") or ""), str(out.get("reasoning") or ""))
    return s2


JUDGE_SYSTEM = (
    "You are a Minecraft survival expert judging which of two proposed next actions, A and B, is "
    "better for a survival bot on its way to a full diamond kit, the Nether, and the End. You are "
    "given the run phase, a plain description of the situation, a digest of the ticks leading up to "
    "it, and a short description of every candidate action (A and B are among them). You are NOT "
    "told which agent proposed A or which proposed B - judge only on the merits for surviving and "
    "progressing right now. "
    'Reply with JSON {"verdict": "A"|"B"|"neither"|"unsure", "better_action": <one allowed '
    'candidate label when verdict is "neither", else null>, "note": <why, short>}. Use "neither" '
    "only when a different allowed candidate action is clearly better than both; use \"unsure\" "
    "when you cannot tell. Prefer \"unsure\" over a guess."
)


def make_judge_fn(cfg: JudgeConfig, api_key: str | None, no_think: bool = False) -> JudgeFn:
    """A `replay` judge callable backed by its own (blind) LLM call."""
    def judge(state: dict, a_label: str, b_label: str, candidates: list[str],
              history: list[dict]) -> tuple[str | None, str | None, str]:
        if not api_key:
            return "unsure", None, f"missing {cfg.api_key_env}"
        try:
            from openai import OpenAI

            client = OpenAI(base_url=cfg.base_url, api_key=api_key)
            # The judge reads the very brief the reviewer did: the phase, the situation as a
            # paragraph, the same trajectory ledger System 2 reasoned over, and the candidate
            # descriptions - not the raw field dump, which made every judgement a 30k-token prefill.
            brief = review_payload(state, history, candidates)
            brief["option_A"] = {"action": a_label, "what_it_does": CRITERIA.get(a_label, "")}
            brief["option_B"] = {"action": b_label, "what_it_does": CRITERIA.get(b_label, "")}
            r = client.chat.completions.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}],
                response_format={"type": "json_object"},
                # Nemotron is a reasoning model: left on its own it spends the whole
                # `max_tokens` budget on a thinking chain and returns `content: null`
                # (finish_reason "length"), which the parser reads as "unsure" - the judge
                # silently loses every verdict. `/no_think` is Qwen-only and does nothing
                # here, so the judge's lever is OpenRouter's `reasoning` control. It is not
                # an OpenAI SDK parameter, so it has to ride in `extra_body`.
                **({"extra_body": {"reasoning": {"enabled": False}}} if no_think else {}),
            )
            content = r.choices[0].message.content
            if not content:
                return "unsure", None, "empty judge response"
            data = json.loads(content)
            verdict = data.get("verdict")
            better = data.get("better_action")
            note = str(data.get("note") or "")
            if verdict not in ("A", "B", "neither", "unsure"):
                return "unsure", None, f"bad verdict: {verdict!r}"
            return verdict, (better if isinstance(better, str) else None), note
        except Exception as e:  # the teacher is offline tooling; never crash the batch
            return "unsure", None, f"error: {e}"
    return judge


def apply_flips(judgments: list[dict], flips: list[dict]) -> list[dict]:
    """Fold the flip test's verdicts into the judgments they re-judged.

    The side-file is what the dataset reads, so a verdict's stability has to live in it, not in a
    separate audit file the builder never sees. A judgment with `flip: True` is one the judge
    reversed when the same two actions were handed over under the other letter: the verdict was
    about the slot, and the dataset must not learn from it.
    """
    by_iteration = {f.get("iteration"): f for f in flips}
    for j in judgments:
        f = by_iteration.get(j.get("iteration"))
        if f:
            j["flip_verdict"] = f.get("flip_verdict")
            j["flip_label"] = f.get("flip_label")
            j["flip"] = bool(f.get("flip"))
    return judgments


def run_teacher(records: list[dict], settings, sample: int = 0, judge_no_think: bool = False) -> list[dict]:
    """Replay with the live LLMs wired from `settings`; `sample` caps the reviewed ticks (0 = all).

    `judge_no_think` turns off the blind judge's reasoning chain via OpenRouter's `reasoning`
    control - measured at ~30x faster per verdict. The local Qwen reviewer has no such lever:
    the Strata server ignores `/no_think`, `thinking_budget` and `enable_thinking`.
    """
    s2_fn = make_system2_fn(settings.llm, settings.api_key())
    judge_fn = make_judge_fn(settings.judge, settings.judge_api_key(), no_think=judge_no_think)
    review = sample_teachable(records, sample) if sample else None
    return replay(records, s2_fn=s2_fn, judge_fn=judge_fn, review=review)


# --- observer as System 2: the live verdict proposes, a picker names the alternative -----------

PICKER_SYSTEM = (
    "You are a Minecraft survival expert. A fast decision engine (Laya) chose one action at one "
    "tick of a survival run, and a watching observer flagged that tick as questionable. You are "
    "given the run phase, a plain description of the situation, a digest of the ticks leading up "
    "to it, the actions that were actually available at that tick, and the observer's short "
    "critique. Pick the single available action that best advances survival and progress right "
    "now; if Laya's own choice is the best one, pick it. "
    "A `blocked_goal` field, when present, names the phase's end-state action the engine is "
    "deliberately withholding this tick and the prerequisites still unmet (`needs`). That action "
    "is NOT available and is not in the list - do not treat its absence as a loop, and prefer the "
    "available action that advances the first unmet `needs` item. "
    'Reply with JSON {"action": <one of the available actions>, "note": <why, short>}.'
)


def _pick_order(candidates: list[str], seed: str, shift: int) -> list[str]:
    """A seeded shuffle of the candidate list - the best-of-N analogue of the blind A/B pairing.

    The picker sees the candidates in shuffled order so a verdict cannot be about which action the
    payload happened to list first; the second shuffle is the stability check, the picker's own
    flip test.
    """
    order = list(candidates)
    random.Random(f"{seed}|{shift}").shuffle(order)
    return order


def make_picker_fn(cfg: JudgeConfig, api_key: str | None, no_think: bool = False):
    """A best-of-N picker over the tick's own candidates, backed by the judge model."""
    def pick(state: dict, ordered_candidates: list[str], critique: str,
             history: list[dict], blocked_goal: dict | None = None) -> tuple[str | None, str]:
        if not api_key:
            return None, f"missing {cfg.api_key_env}"
        try:
            from openai import OpenAI

            client = OpenAI(base_url=cfg.base_url, api_key=api_key)
            brief = review_payload(state, history, ordered_candidates, blocked_goal)
            brief["available_actions"] = [
                {"action": c, "what_it_does": CRITERIA.get(c, "")} for c in ordered_candidates
            ]
            if critique:
                brief["observer_critique"] = critique
            messages = [{"role": "system", "content": PICKER_SYSTEM},
                        {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]

            def call(reasoning_off: bool):
                return client.chat.completions.create(
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **({"extra_body": {"reasoning": {"enabled": False}} if reasoning_off else {}}),
                )

            r = call(no_think)
            content = r.choices[0].message.content
            # Nemotron is a reasoning model: its thinking chain can eat the whole `max_tokens`
            # budget and return `content: null` (finish_reason "length"). That is a truncation,
            # not a verdict - one retry with the chain off recovers the pick instead of silently
            # dropping the tick. Measured on run253: 4 of 18 rejected disagreements were lost to
            # exactly this, not to genuine instability.
            if not content and not no_think:
                r = call(True)
                content = r.choices[0].message.content
            if not content:
                return None, "empty picker response"
            data = json.loads(content)
            action = data.get("action")
            if not isinstance(action, str) or action not in ordered_candidates:
                return None, f"bad pick: {action!r}"
            return action, str(data.get("note") or "")
        except Exception as e:  # the teacher is offline tooling; never crash the batch
            return None, f"error: {e}"
    return pick


def _observer_competing(rec: dict, verdict: dict, candidates: list[str], laya_label: str,
                       picker_fn, history: list[dict]) -> tuple[str | None, str]:
    """The action the observer thinks Laya should have done, or None when none can be named.

    The observer's own `better` field (the extended verdict schema) is preferred: it is the live
    System 2's concrete counter-move. Older logs have no such field, so the picker names one from
    the same candidate list - and only a pick that survives two different shuffles carries a
    label, the same slot-bias logic the blind judge's flip test enforces.
    """
    better = verdict.get("better")
    if isinstance(better, str) and better in candidates and better != laya_label:
        return better, "observer"
    if picker_fn is None:
        return None, ""
    seed = render_narrative(rec) or ""
    critique = str(verdict.get("note") or "")
    blocked_goal = rec.get("blocked_goal")
    a, note_a = picker_fn(rec.get("state") or {}, _pick_order(candidates, seed, 0), critique, history,
                          blocked_goal)
    b, _ = picker_fn(rec.get("state") or {}, _pick_order(candidates, seed, 1), critique, history,
                     blocked_goal)
    if a is not None and a == b and a != laya_label:
        return a, "picker"
    return None, note_a


def replay_from_observer(records: list[dict], verdicts: list[dict], *,
                         judge_fn: JudgeFn, picker_fn=None) -> list[dict]:
    """Side-file judgments where the live observer is System 2 instead of an offline S2 review.

    The observer already watched every tick during the run - its verdict rides in the observer
    log, so the ~77 s per tick System 2 replay is replaced by one picker call per disagreement.
    Where the observer agrees with Laya there is nothing to adjudicate. Where it disagrees, the
    competing action comes from its `better` field (then the blind judge and flip test adjudicate
    it exactly as in the S2 path) or, for logs without that field, from the picker - a blind
    best-of-N choice over the same candidates, kept only if two shuffles agree. Ticks the
    observer never covered get no judgment, so the dataset falls back to the verifier's label.
    """
    by_iter = {v.get("iteration"): v for v in verdicts}
    # How often the observer named each counter-move in this run. A critique it repeats across
    # ticks is a stable read of the state, not a one-tick whim - the repetition is what the
    # `rep-veto` teacher gate counts (see `dataset._teacher_label`).
    repeats: Counter[str] = Counter()
    for v in verdicts:
        verdict = v.get("verdict") or {}
        if verdict.get("agree") is not False:
            continue
        better = verdict.get("better")
        if isinstance(better, str) and better:
            repeats[better] += 1
    out: list[dict] = []
    history: list[dict] = []
    for rec in records:
        if rec.get("stage") not in ("action", "defer"):
            continue
        candidates = _teachable_candidates(rec)
        verdict_rec = by_iter.get(rec.get("iteration"))
        if not candidates or verdict_rec is None:
            if rec.get("stage") == "action":
                history.append(rec)
            continue
        verdict = verdict_rec.get("verdict") or {}
        laya_label = rec.get("label")
        narrative = render_narrative(rec) or ""
        entry: dict[str, Any] = {
            "iteration": rec.get("iteration"),
            "narrative_hash": _narrative_hash(narrative),
            "candidates": candidates,
            "laya_label": laya_label,
            "s2_label": None,
            "agree": False,
            "s2_note": "",
            "s2_reasoning": "",
            "progressed": bool(rec.get("progressed")),
            "teacher_source": "observer",
            "observer_agree": verdict.get("agree"),
            "observer_note": str(verdict.get("note") or "")[:300],
            "observer_risk": str(verdict.get("risk") or "")[:300],
            "judge_verdict": None,
            "judge_label": None,
            "judge_note": "",
            "better_repeats": 0,
        }
        if verdict.get("agree") is True:
            entry["agree"] = True
            entry["s2_label"] = laya_label
        elif verdict.get("agree") is False:
            competing, source = _observer_competing(rec, verdict, candidates, laya_label,
                                                   picker_fn, history)
            if competing is not None:
                entry["s2_label"] = competing
                entry["teacher_source"] = source
                entry["s2_note"] = str(verdict.get("note") or "")[:300]
                if source == "observer":
                    entry["better_repeats"] = repeats.get(competing, 0)
                    # The observer named the counter-move itself - the blind judge and the flip
                    # test then decide whether the disagreement carries a label, as in the S2 path.
                    entry.update(judge_tick(rec, laya_label, competing, candidates, judge_fn, history))
                else:
                    # A picker pick already survived two different shuffles, which is the same
                    # stability requirement the repetition gate counts for an observer label.
                    entry["better_repeats"] = 2
                    entry["judge_verdict"] = "pick"
                    entry["judge_label"] = competing
                    entry["judge_note"] = ""
        out.append(entry)
        history.append(rec)
    return out
