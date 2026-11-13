#!/usr/bin/env python3
"""One-shot MineAI run check: stand, progress, and the defects the lab cares about.

Reads ./scripts/lab.sh status and runs/decisions.jsonl. Prints a compact report;
only the lines that changed since the last check matter, everything else is stable.
"""

import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DECISIONS = REPO / "runs" / "decisions.jsonl"
MILESTONES = [
    "bed_at_base", "frame_complete", "gate_lit", "stronghold_estimated",
    "end_portal_active", "nether_entered", "slept_at_stronghold", "dragon_slain",
]
FRESH_FLAGS = [
    "gate_lit", "frame_complete", "stronghold_estimated", "stronghold_frame",
    "nether_entered", "pearl_wave_trips", "powder_cells", "powder_bank",
]


def status():
    out = subprocess.run(["./scripts/lab.sh", "status"], cwd=REPO,
                         capture_output=True, text=True, timeout=30)
    return json.loads(out.stdout)


def actions():
    acts = []
    if not DECISIONS.exists():
        return acts
    for line in DECISIONS.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("stage") == "action":
            acts.append(d)
    return acts


def milestone_firsts(acts):
    firsts = {}
    for a in acts:
        pr = a.get("state", {}).get("progress", {})
        for k in MILESTONES:
            if pr.get(k) and k not in firsts:
                firsts[k] = a.get("iteration")
    return firsts


def regressions(acts):
    """Milestone flags that flip back to false - the scan outranks the ledger."""
    seen = {}
    regs = []
    for a in acts:
        pr = a.get("state", {}).get("progress", {})
        for k in MILESTONES:
            val = bool(pr.get(k))
            if k in seen and seen[k] and not val:
                regs.append((k, a.get("iteration"), pr.get("source", {}).get(k, "?")))
            seen[k] = val
    return regs


def label_runs(acts):
    """Consecutive stretches of one label, with the net position moved."""
    runs, cur = [], None
    for a in acts:
        label = a.get("label")
        if cur and cur[0] == label:
            cur[1] += 1
            cur[2] += abs(a["delta"].get("pos_delta", 0.0))
        else:
            cur = [label, 1, abs(a["delta"].get("pos_delta", 0.0))]
            runs.append(cur)
    return runs


def main():
    st = status()
    run = st["run"]
    host = st["host"]
    acts = actions()
    n = len(acts)

    print(f"run {run['number']}  iteration {run['iterations']}  ticks {run['ticks']}")
    print(f"health {run['health']}  food {run['food']}  dimension {run['dimension']}  "
          f"last label {run['last_label']} ({run['last_status']})")
    print(f"world {st['world']['running']}  host {host['running']} "
          f"runtime {host['runtime']}  connected {host['connected']}  commit {host['commit']}")
    if run["anomalies"]:
        print("ANOMALIES:", run["anomalies"])

    if n:
        still = sum(1 for a in acts
                    if a["delta"].get("pos_delta", 0) == 0 and not a["delta"].get("inventory_delta"))
        print(f"still_ratio {still / n:.3f} ({still}/{n} ticks with no body or inventory change)")

        firsts = milestone_firsts(acts)
        reached = ", ".join(f"{k}@{v}" for k, v in sorted(firsts.items())) or "keiner"
        print(f"milestones: {reached}")
        for k, it, src in regressions(acts):
            print(f"REGRESSION: {k} false again at it{it} (source {src})")

        it0 = next((a for a in acts if a.get("iteration") == 0), None)
        if it0:
            pr = it0["state"]["progress"]
            stale = {k: pr.get(k) for k in FRESH_FLAGS if pr.get(k)}
            if stale:
                print(f"STALE LEDGER at it0: {stale}")

        bed_rejects = sum(1 for a in acts if "BED_SLEEP_REJECTED" in (a.get("text") or ""))
        if bed_rejects:
            print(f"BED_SLEEP_REJECTED: {bed_rejects}x")

        for label, length, moved in label_runs(acts):
            if length >= 20 and moved < 1.0:
                print(f"OSCILLATION: {label} {length} ticks, {moved:.1f} blocks net")

        water = [a for a in acts[:30] if a.get("label") == "escape_water"]
        if len(water) >= 4:
            net = sum(abs(a["delta"].get("pos_delta", 0.0)) for a in water)
            print(f"WATER AT SPAWN: escape_water {len(water)} ticks, {net:.1f} blocks")

        pickaxe = [a for a in acts
                   if a.get("label") == "craft_pickaxe" and "CRAFTING_TABLE_REQUIRED" in (a.get("text") or "")]
        if len(pickaxe) >= 5:
            print(f"CRAFT_PICKAXE STALL: {len(pickaxe)} ticks with CRAFTING_TABLE_REQUIRED")

        # `last_death` stays in the state after a death, so counting ticks that carry it counts
        # every tick since the death, and two deaths with the same cause collapse into one entry
        # (run228: two lava deaths at the same Nether pond). The respawn signature is the pack
        # going empty: a tick whose item total falls to near zero from a stocked one.
        deaths = []
        prev_total = None
        prev_dim = None
        for a in acts:
            st = a["state"]
            total = sum((st.get("inventory") or {}).values())
            if prev_total is not None and total <= 10 and prev_total >= 20:
                # The tick that carries the death already respawned, so its dimension is the
                # world the bot came back to; the world it died in is the one before.
                deaths.append((a.get("iteration"), st.get("last_death"), prev_dim,
                              st.get("death_pos")))
            prev_total = total
            prev_dim = st.get("dimension")
        if deaths:
            print(f"DEATHS: {len(deaths)}")
            for it, cause, dim, pos in deaths:
                pos = tuple(round(v, 1) for v in (pos or ()))
                print(f"  it{it} in {dim} at {pos}: {cause}")

        top = Counter(a.get("label") for a in acts).most_common(6)
        print("top labels:", ", ".join(f"{l} {c}" for l, c in top))

        pos = acts[-1]["state"]["position"]
        print(f"position {pos}")


if __name__ == "__main__":
    main()
