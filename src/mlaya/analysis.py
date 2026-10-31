"""Run analysis: the lab's second pair of eyes on a finished run.

The loop writes one record per tick to `runs/decisions.jsonl`. This module turns that log
into a defect report: how the run ended, what the server refused and how often, where the
bot stood still, which milestone cost how many ticks, and how all of that compares to the
run before it. It reads the log only (plus the run's own end JSON when given), so it works
on archived runs and never touches the live loop.

The organising idea is the one the stall guards already use: a tick that moves neither the
body nor the inventory taught nothing, and a run that reaches its goal in fewer ticks is a
better run regardless of how many ticks it had available.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

# A bracketed all-caps token in a tool reply is the server's error code.
CODE_RE = re.compile(r"\[([A-Z][A-Z_]{3,})\]")
# Six ticks on one or two labels with no gain is the same window the dataset filter calls
# a ping-pong; the report keeps the whole window so the iteration range is quotable.
STALL_TICKS = 6
# Below this a tick counts as standing still. A step is a block; 0.5 separates a real
# move from the jitter of a pathfinding attempt that never left its cell.
MOVE_EPS = 0.5


def read_records(path: Path) -> list[dict[str, Any]]:
    """One dict per line; a half-written line from a run still in progress is skipped."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _end_from_run_log(run_log: Path | None) -> dict[str, Any] | None:
    """The run's last line is the end JSON (`iterations`, `stop`, `note`, `policy`)."""
    if not run_log or not run_log.exists():
        return None
    for line in reversed(run_log.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _pos_delta(record: dict[str, Any]) -> float:
    delta = record.get("delta") or {}
    value = delta.get("pos_delta")
    if value is None:
        return 0.0
    return float(value)


def _gained(record: dict[str, Any]) -> int:
    delta = record.get("delta") or {}
    inventory = delta.get("inventory_delta") or {}
    return sum(v for v in inventory.values() if isinstance(v, (int, float)) and v > 0)


def _wasted(record: dict[str, Any]) -> bool:
    """The report's definition of a wasted tick: neither the body nor the inventory moved.
    A tick that smelts iron in front of a furnace gains an item without walking; calling
    that standing would make a productive run look idle."""
    return _pos_delta(record) < MOVE_EPS and _gained(record) == 0


def summarize(records: list[dict[str, Any]], run_log: Path | None = None) -> dict[str, Any]:
    actions = [r for r in records if r.get("stage") == "action"]
    defers = [r for r in records if r.get("stage") == "defer"]
    policies = [r for r in records if r.get("stage") == "policy"]

    refusals: Counter[str] = Counter()
    refusal_first: dict[str, int] = {}
    refusal_labels: dict[str, set[str]] = {}
    for r in actions:
        text = f"{r.get('text') or ''} {r.get('code') or ''}"
        for code in CODE_RE.findall(text):
            refusals[code] += 1
            refusal_first.setdefault(code, r.get("iteration"))
            refusal_labels.setdefault(code, set()).add(str(r.get("label")))

    moved = sum(_pos_delta(r) for r in actions)
    still = [r for r in actions if _wasted(r)]
    longest_still = 0
    streak = 0
    for r in actions:
        if _wasted(r):
            streak += 1
            longest_still = max(longest_still, streak)
        else:
            streak = 0

    stalls: list[dict[str, Any]] = []
    i = 0
    while i < len(actions):
        j = i
        while j < len(actions) and _wasted(actions[j]):
            j += 1
        window = actions[i:j]
        if len(window) >= STALL_TICKS:
            labels = sorted({str(w.get("label")) for w in window})
            if len(labels) <= 2:
                stalls.append({
                    "start_iteration": window[0].get("iteration"),
                    "end_iteration": window[-1].get("iteration"),
                    "ticks": len(window),
                    "labels": labels,
                    "codes": sorted({c for w in window
                                    for c in CODE_RE.findall(f"{w.get('text') or ''}")}),
                })
        i = j if j > i else i + 1

    healths = [r["state"]["health"] for r in actions
               if isinstance(r.get("state"), dict) and r["state"].get("health") is not None]
    foods = [r["state"]["food"] for r in actions
             if isinstance(r.get("state"), dict) and r["state"].get("food") is not None]

    milestones: list[dict[str, Any]] = []
    seen_facts: set[str] = set()
    death_ticks: list[int] = []
    tick = 0
    for r in records:
        if r.get("stage") == "action":
            tick += 1
        if r.get("stage") == "policy" and r.get("reason") == "death":
            death_ticks.append(tick)
        for fact in r.get("facts") or []:
            if fact not in seen_facts:
                seen_facts.add(fact)
                # Policy records carry no iteration of their own; the tick they fall between
                # is when the loop learned the fact.
                milestones.append({"fact": fact, "iteration": tick})

    # The chain the run is meant to close, in the order the loop works through it. The
    # iteration a fact first appears at is the cost of that phase, and the difference
    # between two runs is the whole point of the lab.
    chain = ["nether_entered", "powder_bank", "stronghold_estimated",
             "frame_complete", "end_portal_active"]
    reached = {fact: next((m["iteration"] for m in milestones if m["fact"] == fact), None)
               for fact in chain}
    reached_chain = [v for v in reached.values() if v is not None]

    label_stats: dict[str, dict[str, Any]] = {}
    for r in actions:
        label = str(r.get("label"))
        entry = label_stats.setdefault(label, {"label": label, "ticks": 0, "progressed": 0,
                                              "failed": 0, "blocks": 0.0})
        entry["ticks"] += 1
        entry["progressed"] += 1 if r.get("progressed") else 0
        entry["failed"] += 1 if r.get("status") == "failed" else 0
        entry["blocks"] += _pos_delta(r)

    wait_ticks = sum(1 for r in actions if r.get("label") == "wait")

    waste = sorted(
        (r for r in actions if not r.get("progressed")),
        key=lambda r: tuple(-refusals.get(c, 0) for c in CODE_RE.findall(str(r.get("text") or ""))),
    )[:10]

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "end": _end_from_run_log(run_log),
        "ticks": {"action": len(actions), "defer": len(defers), "policy": len(policies)},
        "progress": {
            "progressed": sum(1 for r in actions if r.get("progressed")),
            "succeeded": sum(1 for r in actions if r.get("status") == "succeeded"),
            "failed": sum(1 for r in actions if r.get("status") == "failed"),
            "progress_rate": round(sum(1 for r in actions if r.get("progressed")) / len(actions), 3)
            if actions else 0.0,
        },
        "movement": {
            "total_blocks": round(moved, 1),
            "blocks_per_tick": round(moved / len(actions), 2) if actions else 0.0,
            "still_ticks": len(still),
            "still_ratio": round(len(still) / len(actions), 3) if actions else 0.0,
            "longest_still_streak": longest_still,
        },
        "waits": {"wait_ticks": wait_ticks, "defer_ticks": len(defers)},
        "refusals": [
            {"code": code, "count": count, "first_iteration": refusal_first.get(code),
             "labels": sorted(refusal_labels.get(code, set()))}
            for code, count in refusals.most_common()
        ],
        "stalls": stalls,
        "vitals": {
            "min_health": min(healths) if healths else None,
            "min_food": min(foods) if foods else None,
            "deaths": len(death_ticks),
            "death_iterations": death_ticks,
        },
        "milestones": milestones,
        "goal": {
            "reached": reached,
            "ticks_to_stronghold": reached.get("stronghold_estimated"),
            "chain_closed": all(v is not None for v in reached.values()),
            "last_milestone_iteration": max(reached_chain) if reached_chain else None,
        },
        "labels": sorted(label_stats.values(), key=lambda e: -e["ticks"]),
        "top_waste": [
            {"iteration": r.get("iteration"), "label": r.get("label"), "tool": r.get("tool"),
             "status": r.get("status"), "text": str(r.get("text") or "")[:160]}
            for r in waste
        ],
    }


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The numbers that decide whether a fix helped, side by side with the run before."""
    def pick(report: dict[str, Any], path: str) -> Any:
        node: Any = report
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        return node

    keys = [
        "end.stop", "ticks.action", "progress.progress_rate",
        "movement.blocks_per_tick", "movement.still_ratio",
        "waits.wait_ticks", "waits.defer_ticks", "vitals.deaths",
    ]
    deltas: dict[str, Any] = {}
    for key in keys:
        a, b = pick(current, key), pick(baseline, key)
        if a is not None and b is not None:
            deltas[key] = round(a - b, 3) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    codes = {c["code"]: c["count"] for c in current.get("refusals", [])}
    base_codes = {c["code"]: c["count"] for c in baseline.get("refusals", [])}
    new_codes = sorted(c for c in codes if c not in base_codes)
    worse = sorted(
        ({"code": c, "count": codes[c], "was": base_codes[c]}
         for c in codes if c in base_codes and codes[c] > base_codes[c]),
        key=lambda e: -(e["count"] - e["was"]),
    )
    return {"deltas": deltas, "new_refusals": new_codes, "worse_refusals": worse[:8]}


def to_markdown(report: dict[str, Any], diff: dict[str, Any] | None = None) -> str:
    lines = [f"# Run report ({report['generated']})"]
    end = report.get("end") or {}
    lines.append(f"- stop: **{end.get('stop', 'unknown')}** · iterations: {end.get('iterations', report['ticks']['action'])}")
    lines.append(f"- progress rate: {report['progress']['progress_rate']} · failed calls: {report['progress']['failed']}")
    lines.append(f"- movement: {report['movement']['total_blocks']} blocks, "
                 f"{report['movement']['blocks_per_tick']}/tick, "
                 f"{report['movement']['still_ratio']:.0%} of ticks standing still")
    lines.append(f"- waits: {report['waits']['wait_ticks']} · defers: {report['waits']['defer_ticks']} · deaths: {report['vitals']['deaths']}")

    if report["refusals"]:
        lines.append("\n## Refusals")
        for r in report["refusals"][:10]:
            lines.append(f"- `{r['code']}` ×{r['count']} (first at it{r['first_iteration']}, labels: {', '.join(r['labels'])})")

    if report["stalls"]:
        lines.append("\n## Stalls")
        for s in report["stalls"][:10]:
            lines.append(f"- it{s['start_iteration']}–it{s['end_iteration']}: {s['ticks']} ticks on "
                         f"{', '.join(s['labels'])}{' · ' + ', '.join(s['codes']) if s['codes'] else ''}")

    if report["milestones"]:
        lines.append("\n## Milestones")
        for m in report["milestones"]:
            lines.append(f"- {m['fact']} at it{m['iteration']}")
        goal = report.get("goal", {})
        lines.append(f"- chain closed: {goal.get('chain_closed')} · "
                     f"ticks to stronghold: {goal.get('ticks_to_stronghold')}")

    if diff:
        lines.append("\n## Against the baseline")
        for key, value in diff["deltas"].items():
            lines.append(f"- {key}: {value:+}")
        if diff["new_refusals"]:
            lines.append(f"- new refusal codes: {', '.join(diff['new_refusals'])}")
        for w in diff["worse_refusals"]:
            lines.append(f"- worse: `{w['code']}` {w['was']} → {w['count']}")

    return "\n".join(lines)
