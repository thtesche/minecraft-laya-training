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
from typing import Any, Callable

from .actions import CRITERIA
from .brain import escalate_to_system2
from .config import JudgeConfig, LLMConfig
from .dataset import _narrative_hash, render_narrative

# A teacher judgment is produced only for an action tick the head could actually learn from: more
# than one eligible label, and a label Laya really chose among them. The `in_danger` gate and its
# floor/ceiling heuristic in `dataset.py` are deliberately left alone - this path is about the
# intent choice only.
S2Fn = Callable[[dict, list[str], list[dict]], tuple[str | None, str]]
JudgeFn = Callable[[dict, str, str, list[str]], tuple[str | None, str | None, str]]


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
               judge_fn: JudgeFn, swap: bool = False) -> dict:
    """Run the blind judge for one disagreement and map its verdict back to a concrete label.

    `swap` puts the same two actions in the opposite slots. That is the flip test: a verdict that
    survives the swap is a judgement about the actions, a verdict that flips is a reaction to which
    letter they were handed under.
    """
    narrative = render_narrative(rec) or ""
    a_label, b_label, a_is_laya = _blind_pair(laya_label, s2_label, narrative)
    if swap:
        a_label, b_label = b_label, a_label
    verdict, better, note = judge_fn(rec.get("state") or {}, a_label, b_label, candidates)
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


def replay(records: list[dict], *, s2_fn: S2Fn, judge_fn: JudgeFn) -> list[dict]:
    """Produce one judgment per teachable action tick (pure; the LLM calls are injected).

    `s2_fn(state, candidates, history)` returns `(label, note)` - System 2's pick among the
    candidates (or None) and its reason. `judge_fn(state, a_label, b_label, candidates)` returns
    `(verdict, better_action, note)`. Keeping both injectable is what lets the agreement logic,
    the blind pairing and the verdict mapping be tested with no network.
    """
    out: list[dict] = []
    history: list[dict] = []
    for rec in records:
        if rec.get("stage") not in ("action", "defer"):
            continue
        candidates = _teachable_candidates(rec)
        if not candidates:
            # A non-teachable tick still belongs to the history S2 reads for context.
            if rec.get("stage") == "action":
                history.append(rec)
            continue
        narrative = render_narrative(rec) or ""
        laya_label = rec.get("label")
        s2_label, s2_note = s2_fn(rec.get("state") or {}, candidates, history)
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
            "progressed": bool(rec.get("progressed")),
            "judge_verdict": None,
            "judge_label": None,
            "judge_note": "",
        }
        # The judge is spent only where there is a real disagreement to settle.
        if not agree and s2_label is not None:
            entry.update(judge_tick(rec, laya_label, s2_label, candidates, judge_fn))
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
    out: list[dict] = []
    for j in judgments:
        if j.get("s2_label") is None or j.get("agree"):
            continue
        rec = by_iteration.get(j.get("iteration"))
        if rec is None:
            continue
        flipped = judge_tick(rec, j["laya_label"], j["s2_label"],
                             j.get("candidates") or [], judge_fn, swap=True)
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
    """A `replay` S2 callable backed by `brain.escalate_to_system2`.

    Deliberately neutral: the failure brief says only that this is an offline review, and Laya's
    own choice is *not* handed to the model, so S2 forms its opinion over the same state and the
    same candidates Laya saw - not for or against a label it was shown.
    """
    def s2(state: dict, candidates: list[str], history: list[dict]) -> tuple[str | None, str]:
        plan = escalate_to_system2(
            cfg, api_key, state, history,
            {"triggers": ["offline_teacher_review"]}, list(candidates))
        if not isinstance(plan, dict):
            return None, ""
        if plan.get("error"):
            return None, f"error: {plan['error']}"
        label = plan.get("action")
        return (label if label in candidates else None), str(plan.get("note") or "")
    return s2


JUDGE_SYSTEM = (
    "You are a Minecraft survival expert judging which of two proposed next actions is better "
    "for a bot in a given situation. You are NOT told which agent proposed which option - judge "
    "only on the merits for surviving and progressing toward a diamond kit and the nether. "
    'Reply with JSON {"verdict": "A"|"B"|"neither"|"unsure", "better_action": <one allowed '
    'label when verdict is "neither", else null>, "note": <why, short>}. Use "neither" only '
    "when a different allowed action is clearly better than both; use \"unsure\" when you "
    "cannot tell. Prefer \"unsure\" over a guess."
)


def make_judge_fn(cfg: JudgeConfig, api_key: str | None) -> JudgeFn:
    """A `replay` judge callable backed by its own (blind) LLM call."""
    def judge(state: dict, a_label: str, b_label: str,
              candidates: list[str]) -> tuple[str | None, str | None, str]:
        if not api_key:
            return "unsure", None, f"missing {cfg.api_key_env}"
        try:
            from openai import OpenAI

            client = OpenAI(base_url=cfg.base_url, api_key=api_key)
            payload = {
                "state": state,
                "option_A": {"action": a_label, "what_it_does": CRITERIA.get(a_label, "")},
                "option_B": {"action": b_label, "what_it_does": CRITERIA.get(b_label, "")},
                "allowed_actions": {c: CRITERIA.get(c, "") for c in candidates},
            }
            r = client.chat.completions.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"},
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


def run_teacher(records: list[dict], settings) -> list[dict]:
    """Replay with the live LLMs wired from `settings`."""
    s2_fn = make_system2_fn(settings.llm, settings.api_key())
    judge_fn = make_judge_fn(settings.judge, settings.judge_api_key())
    return replay(records, s2_fn=s2_fn, judge_fn=judge_fn)
