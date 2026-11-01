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
import math
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
# A label that walks every tick and ends where it began is a bounce, not work. The ratio is the
# one the loop's orbit guard uses; the window is shorter so the report also catches a hop the
# guard's 12-tick trail lets through (live, run143: 94 `approach_lava` ticks hopping between two
# stances 5.6 blocks apart beside the same two lava flows). `LOOP_SPAN` keeps a streak an episode:
# eight `sleep` ticks spread over 150 iterations are not a bounce, they are the run's shape.
LOOP_TICKS = 8
LOOP_WALK = 36.0
LOOP_RATIO = 5.0
LOOP_SPAN = 24

# The chain a run is meant to close, in the order the loop works through it. These are the keys of
# every tick's `progress` block, so a milestone is the tick the loop first believed the fact --
# which is the only measurement the goal metric can use: a run that reaches the stronghold in
# fewer ticks is a better run, and without these numbers "reached" is a guess.
CHAIN_FACTS = ("bed_at_base", "frame_complete", "gate_lit", "nether_entered",
               "powder_banked", "stronghold_estimated", "end_portal_active")


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


def _in_water(record: dict[str, Any]) -> bool:
    state = record.get("state") or {}
    return bool(state.get("in_water"))


def _swim(record: dict[str, Any]) -> float:
    """Horizontal distance the body actually crossed this tick.

    `pos_delta` counts vertical drift, and a body adrift in a flooded pocket travels three
    blocks straight up and back down on the same column every tick (live run115: 281 rescue
    walks at (-0.5, 62, 281), `pos_delta` 3.5 each, never on ground). Only the x/z plane says
    whether a swim went anywhere.
    """
    delta = record.get("delta") or {}
    value = delta.get("swim_delta")
    if value is None:
        return 0.0
    return float(value)


def _position(record: dict[str, Any]) -> tuple[float, float, float] | None:
    state = record.get("state") or {}
    pos = state.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        return None
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _death_causes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attribute each death: the cause the server names and the footing the body had.

    The policy line marks the tick the loop noticed the death. The server's cause string
    (`last_death`) lands in the state of the action records that follow -- run149 shows the
    cause appearing two ticks after the line, not the immediately next one -- so the scan
    forward takes the first following action record that carries it. Where the body died is
    read from the action record *before* the line: by the time the cause is reported the bot
    has already respawned at the bed, so the tick after says nothing about the water the body
    fell into. A cause naming drowning counts as a water death even without the footing read;
    a body standing in water at its last tick counts too, whatever the cause says.
    """
    out: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        if r.get("stage") != "policy" or r.get("reason") != "death":
            continue
        cause = ""
        for nxt in records[i + 1:i + 4]:
            if nxt.get("stage") == "action":
                cause = str((nxt.get("state") or {}).get("last_death") or "")
                if cause:
                    break
        in_water = False
        for prev in reversed(records[:i]):
            if prev.get("stage") == "action":
                in_water = bool((prev.get("state") or {}).get("in_water"))
                break
        out.append({
            "cause": cause or "unreported",
            "in_water": in_water,
            "water_death": "drown" in cause.lower() or in_water,
        })
    return out


def _target(record: dict[str, Any]) -> tuple[int, int, int] | None:
    """The cell an action aimed at, when the tick logged one."""
    args = record.get("args") or {}
    cell = (args.get("x"), args.get("y"), args.get("z"))
    return cell if all(isinstance(v, int) for v in cell) else None


def _loops(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Labels whose ticks walk a long path and end where they started.

    `still_ratio` cannot see this: a bot hopping between two stances moves five blocks every
    tick, which reads as movement. The test is the one the orbit guard uses, over a sliding
    window of `LOOP_TICKS` ticks of one label — a window that walked `LOOP_WALK` blocks yet
    ended within `path / LOOP_RATIO` of where it began is a bounce. Overlapping windows of the
    same label merge into one streak, so the report quotes the whole ping-pong, not one window,
    and a window only counts while it sits inside `LOOP_SPAN` iterations -- eight `sleep` ticks
    spread over 150 iterations are the run's shape, not a loop.
    """
    trails: dict[str, list[tuple[int, tuple[float, float, float]]]] = {}
    for r in actions:
        pos = _position(r)
        if pos is None:
            continue
        trails.setdefault(str(r.get("label")), []).append(
            (int(r.get("iteration", 0)), pos, _target(r)))

    loops = []
    for label, trail in trails.items():
        if len(trail) < LOOP_TICKS:
            continue
        windows = []
        for i in range(len(trail) - LOOP_TICKS + 1):
            window = trail[i:i + LOOP_TICKS]
            path = sum(math.dist(window[k][1], window[k + 1][1]) for k in range(len(window) - 1))
            net = math.dist(window[0][1], window[-1][1])
            if (path >= LOOP_WALK and net < path / LOOP_RATIO
                    and window[-1][0] - window[0][0] <= LOOP_SPAN):
                windows.append((i, path, net))
        if not windows:
            continue
        # Overlapping windows of the same bounce merge into one streak, but only while the streak
        # stays inside `LOOP_SPAN` iterations: a hop is an episode, not the run's whole shape.
        streaks = []
        for i, path, net in windows:
            if (streaks and i <= streaks[-1][1] - LOOP_TICKS + 1
                    and trail[i + LOOP_TICKS - 1][0] - trail[streaks[-1][0]][0] <= LOOP_SPAN):
                streaks[-1][1] = max(streaks[-1][1], i + LOOP_TICKS)
                streaks[-1][2] += path
                streaks[-1][3] += net
                streaks[-1][4] += 1
            else:
                streaks.append([i, i + LOOP_TICKS, path, net, 1])
        for start, end, path, net, count in streaks:
            cells = sorted({t for _, _, t in trail[start:end] if t is not None})
            loops.append({"label": label, "ticks": end - start,
                          "start_iteration": trail[start][0], "end_iteration": trail[end - 1][0],
                          "path_blocks": round(path / count, 1),
                          "net_blocks": round(net / count, 1),
                          # The cells the bouncing label kept aiming at: the hop is a stance
                          # problem, and the cell names are what make it fixable.
                          "targets": [list(c) for c in cells]})
    return sorted(loops, key=lambda e: -e["ticks"])


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

    # Water: how much of the run is spent off the feet, and how much of that swim goes
    # nowhere. A bot adrift in a flooded pocket answers `escape_water` on tick after tick
    # aimed at a foothold the pathfinder cannot reach (live, run115: 281 rescue walks from
    # (-0.5, 62, 281), each one "visited 30902 nodes ... closest node was -1, 59, 281").
    water = [r for r in actions if _in_water(r)]
    longest_water = 0
    streak = 0
    for r in actions:
        if _in_water(r):
            streak += 1
            longest_water = max(longest_water, streak)
        else:
            streak = 0
    adrift = [r for r in water if _swim(r) < MOVE_EPS]

    loops = _loops(actions)

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
        # The chain facts are on every tick's `progress` block, so a milestone is the tick the
        # loop first believed it. Only a restored world writes `facts`, which is why run143's
        # report had no Milestones section at all: the chain was never measured, only assumed.
        progress = (r.get("state") or {}).get("progress") or {}
        for fact, value in progress.items():
            if fact in CHAIN_FACTS and value and fact not in seen_facts:
                seen_facts.add(fact)
                milestones.append({"fact": fact, "iteration": tick})

    # The chain the run is meant to close, in the order the loop works through it. The
    # iteration a fact first appears at is the cost of that phase, and the difference
    # between two runs is the whole point of the lab.
    chain = list(CHAIN_FACTS)
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

    # System 2: the generative escalation, asked only where System 1 runs out -- a label the
    # tech tree offered but `resolve` could not turn into a concrete action. Every call is
    # logged (`stage: escalation`) with the label that failed, the model's plan and the label
    # the loop actually ran, and the action record that follows carries `escalated: true`.
    escalations = [r for r in records if r.get("stage") == "escalation"]
    esc_actions = [r for r in actions if r.get("escalated")]
    esc_trigger = Counter(str(e.get("label")) for e in escalations)
    esc_chosen = Counter(str(e.get("chosen")) for e in escalations)

    waste = sorted(
        (r for r in actions if not r.get("progressed")),
        key=lambda r: tuple(-refusals.get(c, 0) for c in CODE_RE.findall(str(r.get("text") or ""))),
    )[:10]

    death_causes = _death_causes(records)

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
        "water": {
            "ticks": len(water),
            "ratio": round(len(water) / len(actions), 3) if actions else 0.0,
            "longest_streak": longest_water,
            "adrift_ticks": len(adrift),
            "swim_blocks": round(sum(_swim(r) for r in water), 1),
            "escape_ticks": sum(1 for r in actions if r.get("label") == "escape_water"),
        },
        "loops": {
            "labels": len(loops),
            "ticks": sum(l["ticks"] for l in loops),
            "detail": loops,
        },
        "system2": {
            "calls": len(escalations),
            "ticks": len(esc_actions),
            "progressed": sum(1 for r in esc_actions if r.get("progressed")),
            "progress_rate": round(sum(1 for r in esc_actions if r.get("progressed"))
                                   / len(esc_actions), 3) if esc_actions else None,
            "fallbacks": sum(1 for e in escalations if not e.get("chosen")),
            "errors": sum(1 for e in escalations if (e.get("plan") or {}).get("error")),
            "triggers": [{"label": label, "count": count}
                         for label, count in esc_trigger.most_common()],
            "chosen": [{"label": label, "count": count}
                       for label, count in esc_chosen.most_common()],
            "calls_detail": [
                {"iteration": e.get("iteration"),
                 "unresolved": e.get("label"),
                 "chosen": e.get("chosen"),
                 "goal": str((e.get("plan") or {}).get("goal") or "")[:120],
                 "note": str((e.get("plan") or {}).get("note") or "")[:160]}
                for e in escalations[:10]
            ],
        },
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
            "death_causes": death_causes,
            "water_deaths": sum(1 for d in death_causes if d["water_death"]),
        },
        "milestones": milestones,
        "goal": {
            "reached": reached,
            "missing": [f for f in chain if reached[f] is None],
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
        "waits.wait_ticks", "waits.defer_ticks", "vitals.deaths", "vitals.water_deaths",
        "water.ratio", "water.adrift_ticks", "loops.ticks", "system2.calls",
        "goal.ticks_to_stronghold",
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
    # A run that died mid-tick has no `stop`: its end line is the error JSON. Naming the
    # failure here is what tells the reader the run was cut short rather than finished.
    stop = end.get("stop") or end.get("error") or "unknown"
    lines.append(f"- stop: **{stop}** · iterations: {end.get('iterations', report['ticks']['action'])}")
    if end.get("detail"):
        lines.append(f"- the run stopped reading the world: {end['detail'][:160]}")
    lines.append(f"- progress rate: {report['progress']['progress_rate']} · failed calls: {report['progress']['failed']}")
    lines.append(f"- movement: {report['movement']['total_blocks']} blocks, "
                 f"{report['movement']['blocks_per_tick']}/tick, "
                 f"{report['movement']['still_ratio']:.0%} of ticks standing still")
    lines.append(f"- waits: {report['waits']['wait_ticks']} · defers: {report['waits']['defer_ticks']} · deaths: {report['vitals']['deaths']}")
    vitals = report.get("vitals") or {}
    if vitals.get("death_causes"):
        detail = "; ".join(
            f"{d['cause']}{' (body in water)' if d['in_water'] else ''}"
            for d in vitals["death_causes"])
        lines.append(f"- death causes: {detail} · water deaths: {vitals.get('water_deaths', 0)}")
    water = report.get("water") or {}
    if water.get("ticks"):
        lines.append(f"- water: {water['ticks']} ticks off the feet ({water['ratio']:.0%}), longest "
                     f"streak {water['longest_streak']}, {water['adrift_ticks']} of them adrift, "
                     f"{water['escape_ticks']} rescue ticks")

    if report["refusals"]:
        lines.append("\n## Refusals")
        for r in report["refusals"][:10]:
            lines.append(f"- `{r['code']}` ×{r['count']} (first at it{r['first_iteration']}, labels: {', '.join(r['labels'])})")

    if report["stalls"]:
        lines.append("\n## Stalls")
        for s in report["stalls"][:10]:
            lines.append(f"- it{s['start_iteration']}–it{s['end_iteration']}: {s['ticks']} ticks on "
                         f"{', '.join(s['labels'])}{' · ' + ', '.join(s['codes']) if s['codes'] else ''}")

    loops = report.get("loops") or {}
    if loops.get("labels"):
        lines.append(f"\n## Loops ({loops['ticks']} ticks walking, going nowhere)")
        for l in loops["detail"]:
            lines.append(f"- it{l['start_iteration']}–it{l['end_iteration']}: {l['ticks']} ticks on "
                         f"`{l['label']}` — {l['path_blocks']} blocks walked per window, "
                         f"{l['net_blocks']} net")
            if l.get("targets"):
                cells = ", ".join(f"({c[0]},{c[1]},{c[2]})" for c in l["targets"][:6])
                lines.append(f"  aimed at {len(l['targets'])} cell(s): {cells}")

    s2 = report.get("system2") or {}
    if s2.get("calls"):
        lines.append(f"\n## System 2 escalations ({s2['calls']})")
        lines.append(f"- {s2['ticks']} ticks ran on an escalated action, progress rate "
                     f"{s2['progress_rate']} · {s2['fallbacks']} fell back to the default")
        for d in s2["calls_detail"]:
            lines.append(f"- it{d['iteration']}: `{d['unresolved']}` had no action -> "
                         f"`{d['chosen']}` — {d['goal']}")

    if report["milestones"]:
        lines.append("\n## Milestones")
        for m in report["milestones"]:
            lines.append(f"- {m['fact']} at it{m['iteration']}")
        goal = report.get("goal", {})
        lines.append(f"- chain closed: {goal.get('chain_closed')} · "
                     f"ticks to stronghold: {goal.get('ticks_to_stronghold')}")
        if goal.get("missing"):
            lines.append(f"- still owed: {', '.join(goal['missing'])}")

    if diff:
        base = diff.get("baseline_log", "")
        name = Path(base).name if base else "previous run"
        lines.append(f"\n## Against the baseline ({name})")
        for key, value in diff["deltas"].items():
            lines.append(f"- {key}: {value:+}")
        if diff["new_refusals"]:
            lines.append(f"- new refusal codes: {', '.join(diff['new_refusals'])}")
        for w in diff["worse_refusals"]:
            lines.append(f"- worse: `{w['code']}` {w['was']} → {w['count']}")

    return "\n".join(lines)
