"""Build a Laya training set from the run log (M3).

The run log (`runs/decisions.jsonl`) records, per executed tick, the compact state we
handed Laya (`narrative`), the eligible labels (`candidates`), the label Laya chose
(`label`), and whether that action made verifiable progress (`progressed`).

The verifier is the reward signal. A tick whose action *progressed* is a positive example
for that action in that state; a tick whose action did not progress carries no known-good
alternative, so it is dropped by default (`--include-failed` keeps it as a plain imitation
row). Training rows are emitted in Laya's own schema -- `{state, questions, expected}` --
using the *same* question builders inference uses (`brain.intent_question`, `brain.DANGER_Q`)
so the head is trained on exactly the rendering it meets at run time.

Why danger rows: Laya's head is selected by question *type* (the checkpoint carries a
`type_emb.`), so `in_danger` and `intent` share one `choice` head. An intent-only dataset
would silently un-train the gate. We therefore add an `in_danger` row per state, labelled
from a documented state heuristic that mirrors the criteria text ("a mob is attacking now or
health is very low"), with a logged Laya answer used only where the heuristic is silent.

The heuristic is a *floor*, not a fallback. A logged `no` while the state is unambiguously
dangerous is exactly the zero-shot gate mis-calibration this dataset exists to fix (live: a
tick at 0.3 health with a spider in range still answered `no`), so trusting that answer would
train the error in.

A *ceiling* bounds the label from above: a state that proves itself safe - health comfortably
high and the nearest hostile known to be beyond the runtime band - is a `no` even where the
gate answered `yes`. Without it a logged `yes` for a merely loaded mob is kept, training in the
very over-trigger the runtime band absorbs (`brain.danger_matters`): live run7b answered `yes`
at 9-24 m with health 20. Only a state that proves its own safety trips the ceiling; where the
nearest hostile is unknown (an empty `hostiles` list cannot tell "no mob" from "not recorded")
the logged answer still stands.

Where the `yes` examples come from: an action tick where the gate fired still runs (the guard
can override a defer), but the honest danger moments are the *defer* ticks - the bot stood
aside for the server survival reflex. At run time those produce no action, so an action-only
dataset drops every one of them and trains the gate on negatives alone. We therefore emit a
danger-only row for each deferred tick whose label is `yes`, and - because a refused action
in danger (`progressed=False`, e.g. a body owned by the hostile reflex) is the same honest
moment - for each such tick the heuristic calls dangerous, even though it carries no usable
intent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .actions import CRITERIA
# The danger ceiling reuses the runtime band's thresholds, so the label says exactly what the
# loop acts on (see brain.danger_matters).
from .brain import (DANGER_Q, INTENT_QID, DANGER_QID, intent_question,
                    DANGER_DEFER_M as DANGER_SAFE_M,
                    DANGER_DEFER_HEALTH as DANGER_SAFE_HEALTH)

# Heuristic `in_danger` label, used only when a state carries no logged Laya answer.
# Mirrors the criteria text: a hostile close enough to be hitting the bot, or health low
# enough that the next hit is fatal. Floor and ceiling (DANGER_SAFE_M / DANGER_SAFE_HEALTH,
# imported above) bound the label; between them the logged gate answer stands.
DANGER_CLOSE_M = 4.0
DANGER_LOW_HEALTH = 8.0


def _read_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _min_hostile_distance(state: dict) -> float | None:
    dists: list[float] = []
    for row in state.get("hostiles") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            d = row[2]
            if isinstance(d, (int, float)):
                dists.append(float(d))
    return min(dists) if dists else None


def _heuristic_danger(state: dict) -> bool:
    """State-only danger: fatal health or a hostile inside contact range."""
    health = state.get("health")
    if isinstance(health, (int, float)) and health < DANGER_LOW_HEALTH:
        return True
    d = _min_hostile_distance(state)
    return d is not None and d <= DANGER_CLOSE_M


def _clearly_safe(state: dict) -> bool:
    """State-only safety: health high and the nearest hostile known to be past the band.

    Requires a *known* distance beyond `DANGER_SAFE_M`: an empty or absent `hostiles` list
    cannot tell "no mob" from "not recorded", so it stays ambiguous and does not trip the
    ceiling (the logged answer stands).
    """
    health = state.get("health")
    if not isinstance(health, (int, float)) or health < DANGER_SAFE_HEALTH:
        return False
    d = _min_hostile_distance(state)
    return d is not None and d > DANGER_SAFE_M


def danger_label(state: dict, answers: dict | None) -> str | None:
    """Label for the `in_danger` question.

    The state heuristic is a *floor*: where it says danger the label is `yes` even if the logged
    gate answered `no`. A clearly safe state is a *ceiling*: where it proves safety the label is
    `no` even if the gate answered `yes`. Between the two the logged answer stands, so the gate
    gains the `yes` examples it was missing without learning the far-mob over-trigger. See the
    module docstring.
    """
    if _heuristic_danger(state):
        return "yes"
    if _clearly_safe(state):
        return "no"
    if isinstance(answers, dict):
        ans = answers.get(DANGER_QID)
        if isinstance(ans, dict) and ans.get("choice") in ("yes", "no"):
            return ans["choice"]
        if ans in ("yes", "no"):
            return ans
    return "no"


def _danger_defer_row(narrative: str, answers: dict | None, note: str | None) -> dict | None:
    """A danger-only row for a deferred tick, or None when the defer was not about danger.

    The defer reason is authoritative: either the gate's own logged `in_danger` answer said
    `yes`, or the decision carried the danger note. We deliberately do *not* fall back to the
    state heuristic here - a `no valid action` defer in a state the heuristic calls dangerous
    would otherwise be mislabelled.
    """
    logged = answers.get(DANGER_QID) if isinstance(answers, dict) else None
    choice = logged.get("choice") if isinstance(logged, dict) else logged
    if choice != "yes" and "danger" not in (note or ""):
        return None
    return {"state": narrative,
            "questions": {DANGER_QID: DANGER_Q},
            "expected": {DANGER_QID: "yes"}}


def build_rows(
    log_path: Path,
    *,
    include_failed: bool = False,
    with_danger: bool = True,
    limit: int | None = None,
) -> list[dict]:
    """Turn the run log into `{state, questions, expected}` training rows.

    Action ticks: only those with more than one eligible label carry an `intent` choice; a
    single-candidate tick is deterministic at run time (the filter already picked it), so it
    teaches nothing. Ticks are de-duplicated on (state, candidates).

    Defer ticks: a deferred tick is the bot standing aside for the server survival reflex.
    When the reason is danger that is a labelled `in_danger = yes` example - the case an
    action-only dataset never sees. It becomes a danger-only row. A defer for another reason
    ("no valid action") is ignored.
    """
    rows: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for rec in _read_records(log_path):
        stage = rec.get("stage")
        if stage not in ("action", "defer"):
            continue
        narrative = rec.get("narrative")
        if not isinstance(narrative, str) or not narrative:
            continue
        state = rec.get("state") if isinstance(rec.get("state"), dict) else {}

        if stage == "defer":
            if not with_danger:
                continue
            row = _danger_defer_row(narrative, rec.get("answers"), rec.get("note"))
            key = (narrative, ("in_danger",))
            if row is None or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
            continue

        candidates = tuple(c for c in (rec.get("candidates") or []) if c in CRITERIA)
        label = rec.get("label")
        teachable = len(candidates) >= 2 and label in candidates
        if teachable and (rec.get("progressed") or include_failed):
            key = (narrative, candidates)
            if key in seen:
                continue
            seen.add(key)

            questions: dict[str, Any] = {INTENT_QID: intent_question(list(candidates))}
            expected: dict[str, Any] = {INTENT_QID: label}
            if with_danger:
                dlabel = danger_label(state, rec.get("answers"))
                if dlabel:
                    questions[DANGER_QID] = DANGER_Q
                    expected[DANGER_QID] = dlabel

            rows.append({"state": narrative, "questions": questions, "expected": expected})
            if limit and len(rows) >= limit:
                break
            continue

        # A refused action (no progress) carries no usable intent, but in danger it is a `yes`
        # example for the gate - the low-health tick an action-only dataset would drop.
        if with_danger and _heuristic_danger(state):
            key = (narrative, ("in_danger",))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"state": narrative,
                         "questions": {DANGER_QID: DANGER_Q},
                         "expected": {DANGER_QID: "yes"}})
            if limit and len(rows) >= limit:
                break
    return rows


def _bucket(row: dict, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{row['state']}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def split(rows: list[dict], eval_frac: float = 0.25, seed: int = 1234) -> tuple[list[dict], list[dict]]:
    """Split into (train, eval) by a stable hash of the state, so it is reproducible.

    With very few rows the eval slice is skipped (< 4 rows needs no held-out copy; spending
    a quarter of a 5-row set on one example helps nobody).
    """
    if eval_frac <= 0 or len(rows) < 4:
        return list(rows), []
    ordered = sorted(rows, key=lambda r: _bucket(r, seed))
    n_eval = max(1, round(len(ordered) * eval_frac))
    return ordered[n_eval:], ordered[:n_eval]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
