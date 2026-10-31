---
name: MineAI Lab Run
description: Start, monitor and analyse a MineAI run end to end — the world container, the MCP host from the bot-run branch, the mlaya loop, and the defect report it produces.
---

# Running the lab

The lab is one command with four verbs. Everything else in this skill is the discipline
around them.

```bash
./scripts/lab.sh up [--n 600] [--fresh] [--ignore-goal]   # world + host + run
./scripts/lab.sh status                                   # JSON: world, host, run, last tick
./scripts/lab.sh analyze [--baseline runs/history/…]      # defect report
./scripts/lab.sh down [--run-only]                        # stop run, host, world
```

`lab up` refuses to start a second run while one is running, refuses to start the host on
any branch other than `bot-run`, and only starts the loop once the world answers on 25565
and the host answers on `/health`. Tool conventions, argument gotchas and the failure table
live in the `mineai-minecraft-run` skill; this skill is the sequence, not the tool manual.

## The sequence

1. **Preflight.** `lab.sh status`. If the world or the host is down, `lab.sh up` starts it.
   Never assume the host is running because a previous tick succeeded.
2. **Run.** `lab.sh up --n N`. The run number is chosen by the lab; the previous
   `runs/decisions.jsonl` is archived to `runs/history/decisions-runNN.jsonl` first.
3. **Watch, do not hover.** `lab.sh status` every few minutes. It reports the last tick,
   health, food, dimension and an `anomalies` list. Notify the user immediately on:
   health ≤ 8, hunger ≤ 6, a stall window, or `host.runtime: failed`.
4. **Analyse.** `lab.sh analyze --baseline runs/history/decisions-runNN.jsonl`. Read the
   refusals, the stalls, the movement per tick and the milestone timeline.
5. **Attribute.** Each defect belongs to one of two codebases:
   - the **brain** (`src/mlaya/…`) — candidates, predicates, planner, verifier, policy;
   - the **server** (`/Users/thtesche/VibeCoding/mine-ai-mcp`, branch `bot-run`) — tool
     behaviour, foreground gating, the survival reflexes.
   The host commit is recorded in `runs/.lab/mcp_commit`; a defect that reproduces across
   two runs with the same commit is a brain defect, not a server one.
6. **Fix or ask.** Inside the boundary below, fix and re-run. Outside it, present the
   finding and wait for GO.
7. **Re-run and compare.** The report's diff against the baseline is the only evidence
   that a fix worked.

## What may be fixed without asking

- thresholds and constants, candidate filters in `actions/predicates.py`, error handling in
  `mcp.py`, survival policy values in `survival.py`, analyzer metrics, tests, docs.

## What needs the user's GO

- control flow in `graph.py`, goal and tech-tree definitions, planner structure, dataset or
  training, anything that wipes world progress (seed, `--fresh`, `config/settings.toml`).

Hard limits: at most three autonomous fixes per loop; every fix carries a test that fails
without it; the same defect in two consecutive runs stops the loop and goes to the user;
code is never changed while a run is in flight.

## Server-side fixes (mine-ai-mcp, branch `bot-run`)

The host always runs from `bot-run`. A patch to the server follows the repo's own path:

```bash
cd /Users/thtesche/VibeCoding/mine-ai-mcp
git checkout -b patch-<defect> bot-run
# fix, then
bun run test          # typecheck + bun test over src and tests
```

Merge back into `bot-run`, restart the host (`lab.sh down --run-only` then `lab.sh up`),
and re-run. A server patch that is not merged into `bot-run` is not in effect for the next
run, no matter how it looks in the editor.

## Rechte in der Container-Welt

A dedicated server has no operators by default — unlike a singleplayer LAN world, where the
host is automatically OP. Nothing is cheated: the server is vanilla 1.21.4 with
`online-mode=false` (offline authentication, so MineAI can join) and `spawn-protection=0`.

To get command rights, use the server console:

```bash
./scripts/lab.sh console "op <dein Name>"     # or: ./scripts/lab.sh op <dein Name>
./scripts/lab.sh console "list"              # who is online
```

The viewer needs the same: `LayaView` must be spectator, otherwise it counts as a living
player and the bot can no longer sleep through the night.

## The ledger decides what the world "already has"

`bootstrap_progress` reads the server's SQLite (`action_executions`, `events`,
`stronghold_throws`) once per run, and those facts outrank a fresh block scan. The lab keeps
that ledger under `runs/.lab/bot-data` so `--fresh` wipes world and ledger together. A run
that reports `gate_lit`, `frame_complete` and `stronghold_estimated` at tick 0 in a world
that has none of them is reading an old ledger, not the world.

## The metric the loop optimises

The goal is the whole chain to the stronghold in as few iterations as possible, with the bot
moving on every tick. The report measures exactly that:

- `goal.ticks_to_stronghold` — the iteration `stronghold_estimated` first appears at;
- `movement.blocks_per_tick` and `movement.still_ratio` — a tick that moves neither the body
  nor the inventory is waste;
- `waits.wait_ticks` and `waits.defer_ticks` — every wait is a tick the bot did not use;
- `stalls` — windows of six or more ticks on one or two labels with no gain.

A run is better when the chain closes in fewer ticks **and** the still ratio goes down. Both
have to move; a run that reaches the stronghold in fewer ticks by standing still is not a
better run.
