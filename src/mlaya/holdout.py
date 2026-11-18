"""Held-out evaluation: runs the head never trained on (M4).

A split inside one set of runs measures recall on states the head has already seen in a
neighbouring tick. Live: the 252-256 corpus scored 0.9369 for a fine-tune and 0.9279 for the
base checkpoint, while on run257 - a run that appears in no training file - the same fine-tune
scored 0.8764 against the base's 0.8876. The in-corpus gain did not travel.

Two things make a comparison here meaningless, and both are measured rather than assumed:

  * **Pipeline jitter.** Three runs on identical data, config and seed scored 0.9338, 0.9373 and
    0.9408 on the 574-item set - a 0.70 pp spread. RLCD samples Gaussian noise around the logits
    every step and MPS is not deterministic, so one run is one draw and `holdout train --reps` is
    what measures the noise. The seed is not the source: seeds 0, 1 and 2 scored 0.9355, 0.9390
    and 0.9373, a 0.35 pp spread, smaller than the same-seed one.
  * **Item-level luck.** With 574 items a one-item difference is 0.17 pp. `compare` therefore
    reports McNemar's discordant counts (items A gets right and B wrong, and the reverse) next
    to the accuracy difference, so a "win" that is three items reads as three items.

The registry (`runs/holdout.json`) is the contract: a run added to it is never fed to
`mlaya train`. `build_eval` drops rows whose state also appears in any training file it is
given - a hold-out set that quietly contains training rows measures nothing - and reports how
many it removed.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from .config import ROOT
from . import dataset as ds

REGISTRY_PATH = ROOT / "runs" / "holdout.json"
DEFAULT_EVAL = ROOT / "data" / "holdout-eval.jsonl"
SAMPLE_SEED = 20261117


# --------------------------------------------------------------------------- registry

def registry_path(path: Path | None = None) -> Path:
    """Where the registry lives, resolved when called, not when imported.

    A default argument is bound at import time, so `path: Path = REGISTRY_PATH` would keep
    reading the shipped file even after a caller or a test repoints `REGISTRY_PATH` - the
    dataset guard would then wave through a run the registry says is held out.
    """
    return REGISTRY_PATH if path is None else Path(path)


def load_registry(path: Path | None = None) -> list[dict]:
    """The registered hold-out runs, oldest first."""
    path = registry_path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    runs = data.get("runs") if isinstance(data, dict) else data
    return [r for r in (runs or []) if isinstance(r, dict) and r.get("run") is not None]


def save_registry(runs: list[dict], path: Path | None = None) -> None:
    path = registry_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": sorted(runs, key=lambda r: r["run"])}, indent=2) + "\n")


def run_log(run: int) -> Path | None:
    """The archived decision log of `run`, or None when it is not archived yet.

    The live `runs/decisions.jsonl` is deliberately not a fallback: it belongs to whichever run
    is running now, and registering it under the wrong number would put a run in the hold-out set
    that is not the one meant.
    """
    for p in sorted((ROOT / "runs" / "history").glob(f"decisions-run{run}*.jsonl")):
        return p
    return None


def run_of_log(log: Path) -> int | None:
    """The run number a decision log belongs to, from its name."""
    m = re.search(r"decisions-run(\d+)", str(log))
    return int(m.group(1)) if m else None


def add_run(run: int, log: Path | None = None, note: str = "", path: Path | None = None) -> dict:
    """Register one run as hold-out. Idempotent: re-adding updates the note and the log path."""
    log = log or run_log(run)
    runs = load_registry(path)
    entry = next((r for r in runs if r["run"] == run), None)
    if entry is None:
        entry = {"run": run, "log": str(log) if log else "", "note": note}
        runs.append(entry)
    else:
        if log:
            entry["log"] = str(log)
        if note:
            entry["note"] = note
    save_registry(runs, path)
    return entry


def log_for(entry: dict) -> Path | None:
    """The decision log a registry entry points at.

    An explicit, archived path is authoritative. The live `runs/decisions.jsonl` is not: it
    belongs to whichever run is running now, so a run registered while it was still in progress
    resolves to its archived `decisions-runNN.jsonl` as soon as one exists - otherwise the
    hold-out set silently becomes the current run.
    """
    explicit = Path(entry["log"]) if entry.get("log") else None
    if explicit and explicit.exists() and explicit.name != "decisions.jsonl":
        return explicit
    return run_log(entry["run"]) or explicit


def holdout_logs(runs: list[dict]) -> list[Path]:
    out = []
    for r in runs:
        p = log_for(r)
        if p and p.exists():
            out.append(p)
    return out


# --------------------------------------------------------------------------- eval set

def _row_sig(row: dict) -> tuple[str, tuple[str, ...]]:
    return (row.get("state", ""), tuple(sorted((row.get("questions") or {}).keys())))


def overlap(train_rows: Iterable[dict], eval_rows: Iterable[dict]) -> int:
    """How many hold-out states also sit in the training file. A hold-out set with overlap
    is not held out."""
    seen = {_row_sig(r) for r in train_rows}
    return sum(1 for r in eval_rows if _row_sig(r) in seen)


def eval_path(path: Path | None = None) -> Path:
    """Where the hold-out eval JSONL is written, resolved when called (see `registry_path`)."""
    return DEFAULT_EVAL if path is None else Path(path)


def build_eval(out: Path | None = None, runs: list[dict] | None = None,
               train: Iterable[Path] | None = None, per_run: int = 0,
               seed: int = SAMPLE_SEED) -> dict:
    """Write one eval JSONL from every registered hold-out run.

    No teacher: the hold-out target is what the verifier confirmed Laya actually did, so the
    question stays "does the head predict behaviour it never saw", not "does it agree with the
    observer". Rows whose state also sits in any `train` file are dropped - a hold-out set that
    contains training rows measures nothing - and the count is reported.

    `per_run` caps how many rows each run contributes. Without a cap one long run dominates the
    set and the score stops being a measure of generalisation across worlds; with a cap every
    registered run carries the same weight, and the set stays small enough to score in seconds.
    The sample is seeded, so rebuilding gives the same rows.
    """
    runs = runs if runs is not None else load_registry()
    logs = holdout_logs(runs)
    if not logs:
        return {"error": "no hold-out runs registered", "hint": "mlaya holdout add --run 257"}

    seen: set[tuple[str, tuple[str, ...]]] = set()
    train_files = []
    if isinstance(train, (str, Path)):
        train = [train]
    for t in (train or []):
        t = Path(t)
        if t.exists():
            train_files.append(t)
            seen.update(_row_sig(r) for r in ds._read_records(t))

    rows, dropped, used_runs = [], 0, []
    for r in runs:
        log = log_for(r)
        if not log or not log.exists():
            continue
        used_runs.append(r["run"])
        built = ds.build_rows(Path(log), rerender=True, reverify=True)
        fresh = [x for x in built if _row_sig(x) not in seen]
        dropped += len(built) - len(fresh)
        if per_run and len(fresh) > per_run:
            fresh = random.Random(seed + r["run"]).sample(fresh, per_run)
        for x in fresh:
            x["run"] = r["run"]
        rows.extend(fresh)

    out = eval_path(out)
    ds.write_jsonl(out, rows)
    return {
        "runs": used_runs,
        "train_files_checked": [str(t) for t in train_files],
        "overlap_with_train_dropped": dropped,
        "per_run_cap": per_run,
        "eval_rows": len(rows),
        "eval_items": sum(len(r.get("questions") or {}) for r in rows),
        "eval_path": str(out),
    }


# --------------------------------------------------------------------------- scoring

def _load_and_score(checkpoint: str | Path) -> tuple[dict, list[bool], list[int | None]]:
    """`(metrics, per-item correct, run of each item)` for one checkpoint on DEFAULT_EVAL.

    The run attribution is only trustworthy when every labelled question became an item; if
    `items_from_rows` dropped any, the item order no longer lines up with the rows and the
    attribution is returned empty rather than wrong.
    """
    from laya.train import (load_checkpoint, read_data, items_from_rows,
                            calibration_records, evaluate_records, resolve_device,
                            uses_parallel_layout)

    rows = read_data(str(eval_path()))
    model, tok, cfg = load_checkpoint(str(checkpoint))
    dev = resolve_device("auto")
    model.to(dev)
    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    parallel = uses_parallel_layout(cfg)
    items, skipped = items_from_rows(tok, rows, max_len, head_max_len, label_smoothing=0.0)
    records = calibration_records(model, tok, items, dev, max_len, head_max_len, parallel=parallel)
    metrics = evaluate_records(records, cfg.get("temperature"), cfg.get("temperature_by_options"))
    correct = []
    for _qtype, logits, target, k in records:
        pred = int(max(range(k), key=lambda i: float(logits[i])))
        gold = int(max(range(len(target)), key=lambda i: float(target[i])))
        correct.append(pred == gold)

    runs = [row.get("run") for row in rows for _ in (row.get("questions") or {})]
    return metrics, correct, runs if len(runs) == len(correct) else []


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Discordant counts and the exact two-sided McNemar p-value.

    `a_wins` is the number of items a answers right and b wrong; `b_wins` the reverse. Only
    discordant pairs carry information - on the rest both agree.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("the two checkpoints were scored on different item counts")
    a_wins = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    b_wins = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = a_wins + b_wins
    if n == 0:
        return {"a_wins": 0, "b_wins": 0, "discordant": 0, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(a_wins, b_wins) + 1)) / 2 ** n
    return {"a_wins": a_wins, "b_wins": b_wins, "discordant": n, "p": min(1.0, 2 * tail)}


def compare(checkpoints: list[str | Path]) -> list[dict]:
    """Score every checkpoint on the hold-out set, each against the first one as reference."""
    scored = []
    for ckpt in checkpoints:
        metrics, correct, runs = _load_and_score(ckpt)
        scored.append({"checkpoint": str(ckpt), "metrics": metrics, "correct": correct,
                       "runs": runs})
    ref = scored[0]
    out = []
    for s in scored:
        row = {
            "checkpoint": s["checkpoint"],
            "accuracy": s["metrics"]["accuracy"],
            "brier_top1": s["metrics"]["brier_top1"],
            "ece": s["metrics"]["ece"],
            "mean_confidence": s["metrics"]["mean_confidence"],
            "items": s["metrics"]["items"],
        }
        if s is not ref:
            row["vs_reference"] = mcnemar(ref["correct"], s["correct"])
            row["delta_accuracy"] = s["metrics"]["accuracy"] - ref["metrics"]["accuracy"]
        if s["runs"]:
            tally: dict[int, list[int]] = {}
            for run, ok in zip(s["runs"], s["correct"]):
                if run is None:
                    continue
                t = tally.setdefault(run, [0, 0])
                t[0] += int(ok)
                t[1] += 1
            row["by_run"] = {run: acc / n for run, (acc, n) in sorted(tally.items())
                             if n}
        out.append(row)
    return out


def mean_sd(values: list[float]) -> dict:
    """Mean and sample standard deviation; `spread` is max - min, the number a candidate has
    to clear before a difference can be called an effect."""
    if not values:
        return {"mean": 0.0, "sd": 0.0, "spread": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "spread": max(values) - min(values),
        "n": len(values),
    }
