#!/usr/bin/env python3
"""One monitoring pass over a running lab: cache, ledger, plan, observer, End-prep, health.

Reads ./scripts/lab.sh status plus runs/decisions.jsonl, runs/observer.jsonl and
runs/system2.jsonl. Prints six short blocks in the order the monitor asks for, each
with the number and the reading. Deltas against a previous run come from the archived
history files, so pass --baseline runs/history/decisions-run283.jsonl to compare.
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

KIT_LABELS = {"craft_diamond_sword", "craft_bow", "craft_arrows", "craft_diamond_pickaxe"}
FARM_LABELS = {"kill_chicken", "mine_gravel", "kill_skeleton", "farm_chickens", "collect_eggs"}
WOOD_LABELS = {"get_wood", "chop_tree", "collect_planks"}
STALL_LABELS = {"sleep", "wait_for_action", "wait"}

REGEAR_WINDOW = "prepare_end_dragon"


def status():
    out = subprocess.run(["./scripts/lab.sh", "status"], cwd=REPO,
                         capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


def records(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def cache_block(recs, name):
    calls = [r.get("telemetry") or {} for r in recs if r.get("telemetry")]
    if not calls:
        return f"{name}: keine Calls - nichts zu messen"
    first = calls[0]
    ratios = []
    for c in calls[1:]:
        prompt = c.get("prompt_tokens", 0)
        cached = c.get("cached_tokens", 0)
        if prompt:
            ratios.append(cached / prompt)
    low = [i for i, r in enumerate(ratios, start=2) if r < 0.60]
    mean = sum(ratios) / len(ratios) if ratios else 0.0
    return (f"{name}: {len(calls)} Calls, erste {first.get('cached_tokens', 0)}/{first.get('prompt_tokens', 0)} "
            f"({mean:.0%} ab Call 2), {len(low)} unter 60 % (Calls {low})")


def label_counter(recs):
    return Counter(r["label"] for r in recs if r.get("stage") == "action")


def refusals(recs):
    out = Counter()
    for r in recs:
        if r.get("stage") == "action" and not r.get("ok"):
            text = (r.get("text") or "").split("|")[0].strip()
            out[(r["label"], text[:60])] += 1
    return out


def death_lines(recs):
    """Deaths the loop noticed, with the cause the server names.

    The loop writes a `policy`/`death` line when it re-applies the survival preset after a
    respawn, and the cause lands in the `last_death` field of the state of the records that
    follow - the same read `analysis._death_causes` makes. A health delta cannot find it: by the
    time the next tick is logged the body is back at 20 (live, run288 it60: `explore` walked into
    lava, the whole pack went to the ground, and the delta read `health_delta 0.0`).
    """
    out = []
    for i, r in enumerate(recs):
        if r.get("stage") == "policy" and r.get("reason") == "death":
            cause = ""
            for nxt in recs[i + 1:]:
                cause = str((nxt.get("state") or {}).get("last_death") or "")
                if cause:
                    break
            out.append(cause or "cause unreported")
    return out


def early_milestones(recs):
    """Milestone latches already set on the first logged tick.

    In a freshly restored world these must be empty; in a world the run continues in they are
    the world's own history, so the pass has to say which of the two it is looking at.
    """
    acts = [r for r in recs if r.get("stage") == "action"]
    if not acts:
        return []
    st = acts[0].get("state") or {}
    flags = ("powder_stowed", "nether_entered", "end_portal_active", "slept_at_stronghold",
             "stronghold_frame", "dragon_slain", "base")
    return [f for f in flags if st.get(f)]


def longest_defer(recs):
    run = best = 0
    for r in recs:
        if r.get("stage") == "action" and not r.get("progressed"):
            run += 1
            best = max(best, run)
        elif r.get("stage") == "action":
            run = 0
    return best


def still_ratio(recs):
    acts = [r for r in recs if r.get("stage") == "action"]
    if not acts:
        return 0.0
    still = sum(1 for r in acts
                if abs((r.get("delta") or {}).get("pos_delta", 0.0)) < 0.01
                and not (r.get("delta") or {}).get("inventory_delta"))
    return still / len(acts)


def plan_quality(recs):
    """resolve_rate / recurrence / plan length from the escalation plans in system2.jsonl."""
    plans = [r for r in recs if "plan" in r]
    if not plans:
        return None
    lengths = [len(r.get("steps") or []) for r in plans]
    resolves = sum(1 for r in plans if r.get("resolved"))
    unresolved = Counter(u for r in plans for u in (r.get("unresolved") or []))
    recurrence = sum(c - 1 for c in unresolved.values() if c > 1)
    return {"plans": len(plans), "resolve_rate": resolves / len(plans),
            "avg_len": sum(lengths) / len(lengths) if lengths else 0,
            "recurrence": recurrence}


def main():
    st = status()
    dec = records(RUNS / "decisions.jsonl")
    obs = records(RUNS / "observer.jsonl")
    s2 = records(RUNS / "system2.jsonl")

    run = st["run"]
    print(f"run {run['number']}  it {run['iterations']}  phase {run['phase']}  "
          f"last {run['last_label']} ({run['last_status']})  ticks {run['ticks']}")

    print("\n1. Cache")
    print(cache_block(s2, "System 2"))
    print(cache_block(obs, "Observer"))

    print("\n2. Ledger")
    ledger = sum(1 for r in dec if r.get("stage") == "policy" and r.get("reason") == "plan_ledger")
    print(f"plan_ledger-Schritte: {ledger}")

    print("\n3. Planqualität")
    pq = plan_quality(s2)
    print(json.dumps(pq) if pq else "keine Pläne - System 2 nicht eskaliert")

    print("\n4. Observer")
    o = st["observer"]
    prog = sum(1 for r in obs if r.get("progressed"))
    print(f"judgments {o['judgments']}  overrides {o['overrides']}  progressed {prog}  "
          f"agreement {o['agreement']}  unoffered_gaps {o['unoffered_gaps']}")
    better = Counter(r["verdict"]["better"] for r in obs if r.get("verdict", {}).get("better"))
    if better:
        print("better-Labels: " + ", ".join(f"{k} x{v}" for k, v in better.most_common(6)))

    print("\n5. End-Vorbereitung")
    labels = label_counter(dec)
    offered = Counter()
    for r in dec:
        if r.get("stage") == "action" and "re-gearing" in (r.get("narrative") or ""):
            for c in r.get("candidates", []):
                offered[c] += 1
    kit_offered = {k: v for k, v in offered.items() if k in KIT_LABELS}
    print(f"Kit angeboten: {kit_offered or 'kein Kit-Schritt im Fenster'}")
    print("Kit ausgeführt: " + ", ".join(f"{k} x{labels[k]}" for k in KIT_LABELS if labels[k]) or "keiner")
    entry = [r for r in dec if r.get("stage") == "action" and r["label"] in
             {"enter_end_portal", "enter_the_end", "enter_end"}]
    print(f"End-Eintritte: {len(entry)}")
    print("Refusals: " + (", ".join(f"{l} x{n} [{t}]" for (l, t), n in refusals(dec).most_common(8)) or "keine"))
    wood = [r for r in dec if r.get("stage") == "action" and r["label"] in WOOD_LABELS]
    wood_args = Counter(json.dumps(r.get("args", {}))[:40] for r in wood)
    print(f"get_wood: {len(wood)} Calls ({dict(wood_args)})")
    farm = {k: v for k, v in labels.items() if k in FARM_LABELS}
    print(f"Farming im Fenster: {farm or 'keins'}")
    print(f"längste Deferserie: {longest_defer(dec)}  still_ratio {still_ratio(dec):.3f}")

    print("\n6. Gesundheit")
    print(f"health {run['health']}  food {run['food']}  anomalies {run['anomalies']}  "
          f"host runtime {st['host']['runtime']}  world {st['world']['running']}")
    stall = Counter(r["label"] for r in dec if r.get("stage") == "action" and r["label"] in STALL_LABELS)
    print(f"sleep/wait: {dict(stall)}")
    dead = death_lines(dec)
    print(f"Deaths: {len(dead)}" + (f" - {', '.join(dead)}" if dead else ""))
    print(f"Meilensteine beim ersten Tick: {early_milestones(dec) or 'keine'}")


if __name__ == "__main__":
    main()
