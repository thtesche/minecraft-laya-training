# minecraft-laya-training

A Minecraft agent whose **System-1 decisions are made by
[Laya](https://github.com/NandhaKishorM/laya)** — a non-autoregressive
typed-decision engine (`choice` / `score` / `noul`) — and which acts through the
**mine-ai MCP server**. A generative LLM (OpenRouter) serves as **System 2**
for rare escalations.

Laya produces no text and no tool arguments. The code therefore filters the currently
**valid options** (preconditions/tech tree), and Laya decides in **one** forward pass
over a binary danger gate plus the choice among those options; a separate Python
**executor** calls the MCP tools and respects the server's foreground gating.

## How it works

The loop is a LangGraph: `perceive → decide → guard → execute → verify`, with a
`defer` node when the server's survival reflex owns the body. The same split runs
through every layer — **code filters, Laya chooses**:

| Layer | File | Job |
|---|---|---|
| Perception | `src/mlaya/perception.py` | `view_status` + events + `view_blocks` → `GameState` and a compact English narrative |
| Brain (S1) | `src/mlaya/brain.py` | one Laya `predict`: the `in_danger` gate, then the `intent` choice among the valid labels (and, for `explore`, the target cell) |
| Actions | `src/mlaya/actions.py` | `candidates(gs)`: valid labels by precondition; `resolve(label, gs)`: label → MCP tool + arguments |
| Verify | `src/mlaya/verify.py` | the target/actual delta decides whether an action made progress |
| Memory | `src/mlaya/memory.py` | every decision + outcome in `runs/decisions.jsonl` |
| Survival | `src/mlaya/survival.py` | sets the server policy (`base` / `hunt`) at start and after every death |
| System 2 | `brain.escalate_to_system2` | OpenRouter picks a valid alternative when a label cannot be resolved |

The executor's gating state machine (submission IDs, `ACTION_BUSY` /
`RESULT_NOT_RETRIEVED` recovery, foreground waits) lives in `src/mlaya/mcp.py`. Design
rationale and the zero-shot findings behind this split are in
[`docs/architecture.md`](docs/architecture.md).

## Status

| Milestone | State |
|---|---|
| M1 — one real MCP call per decision | reached |
| M2 — wood → stone → iron → **diamond** chain (tools, full armour, shield, iron combat kit) | reached, validated live |
| M2b — bed / respawn, sleep at night, notifications | reached, validated live |
| M2c — survival policy + food acquisition/eating | reached, validated live |
| M2d — reacting to event/notification types (durability, drops, death) | reached, validated live |
| M3 — run log → dataset → RLCD head fine-tune | reached (OOD zero-shot accuracy 0.934) |
| M4 — parameter layer (target choice over `view_blocks`) | reached, validated live |
| Homebase — surface anchor with bed, chest and furnace | reached, validated live |
| Nether portal — bucket → obsidian → flint & steel → frame → lit portal | reached, validated live |
| Nether prep — bow + arrows, gold helmet, cobblestone/food reserve | implemented, part of the auto-stop |
| Storage network — chests per cell, overflow chests, provisions fetched back | reached, validated live |
| Stall guards — refusals, wandering, idle steps, quiet defers, `wait` hold | implemented (see [`docs/architecture.md`](docs/architecture.md#stall-guards)) |
| Nether visit (phase 3) — crossing, fortress search, blaze rods home as powder | implemented; live up to the fortress spawner (run86), no rod home yet — see [`docs/nether-prep.md`](docs/nether-prep.md) |
| M3, second round — run79–86 head on top of the 2026-10-14 checkpoint | trained (held-out accuracy 0.767 → 0.885), not yet run live |

The run **stops itself** once every phase stands (`Agent._goal_reached`): the overworld goal
(`actions.tech_tree_complete` — full diamond kit **worn** + shield in the off-hand, a bed at the
base, all Nether prep steps done, the food reserve banked and the gate lit) **and** the Nether
trip (`actions.nether_expedition_complete` — the blaze powder deposited in the chest at the
gate). The portal is lit **last** in the overworld: `light_portal` waits for the whole stocked
trip, so the gate never burns through the prep window. It normally lights at night, when the
carried bed can be laid the very next tick; a frame the plane read proves finished may also
burn by day, and the bot then holds at the base (`wait`, 30 s per decision) until night lays
the bed and opens the crossing. The run then reports `"stop": "goal"`; a run that never gets
the powder home ends with `"stop": "limit"`, and `--ignore-goal` (or
`[loop] stop_when_complete = false`) always runs to the iteration limit.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env        # enter OPENROUTER_API_KEY (only for System-2 escalation)
```

`.env` is loaded automatically on every CLI run (real environment variables take
precedence) and is gitignored. Prerequisite: the mine-ai MCP server running at
`http://localhost:25575/mcp`.

## Usage

```bash
.venv/bin/python -m mlaya.cli probe      # health + tools/list
.venv/bin/python -m mlaya.cli decide     # read state + decide once (dry run)
.venv/bin/python -m mlaya.cli policy     # show the server survival policy
.venv/bin/python -m mlaya.cli policy --apply              # set the `base` stage
.venv/bin/python -m mlaya.cli policy --apply --preset hunt  # set the `hunt` stage
.venv/bin/python -m mlaya.cli run -n 10  # decision loop (executes real actions)
.venv/bin/python -m mlaya.cli run --ignore-goal  # keep running past the goal
```

Configuration: [`config/settings.toml`](config/settings.toml) — among others
`[loop] wait_hold_s` (seconds a successful `wait` holds, default 30). Every decision is logged
to `runs/decisions.jsonl`; move it to `runs/history/decisions-runNN.jsonl` before the next run.

### M3: logs → dataset → fine-tuning

```bash
# 1) Build a Laya training dataset from the run log (train + held-out eval)
.venv/bin/python -m mlaya.cli dataset            # -> data/train.jsonl + data/eval.jsonl
.venv/bin/python -m mlaya.cli dataset --include-failed   # also keep ticks without progress
.venv/bin/python -m mlaya.cli dataset --no-danger        # without `in_danger` rows
# several archived runs as one set, into their own files
.venv/bin/python -m mlaya.cli dataset --log runs/history/decisions-run79.jsonl \
    --log runs/history/decisions-run80.jsonl --out data/run79-80-train.jsonl --eval data/run79-80-eval.jsonl

# 2) Check the dataset format (loads no weights), then retrain the head (RLCD)
.venv/bin/python -m mlaya.cli train --dry-run
.venv/bin/python -m mlaya.cli train              # -> runs/checkpoints/minecraft-<time>
# the base is [train] base_model resolved through LAYA_EXTRA_MODELS, so with the variable set
# the head trains on top of the checkpoint the loop serves; --base names one explicitly
.venv/bin/python -m mlaya.cli train --base runs/checkpoints/minecraft-<time> --data data/run79-80-train.jsonl

# 3) Register the checkpoint under `english` and run the loop with it
export LAYA_EXTRA_MODELS='{"english": "'"$PWD"'/runs/checkpoints/minecraft-<time>"}'
.venv/bin/python -m mlaya.cli run -n 10
```

`LAYA_EXTRA_MODELS` is a JSON object `{name: source}`; a name that matches a built-in
checkpoint (`english`) **re-points it** — the fine-tune is served without a code change.
The reward is the verifier: only ticks whose action demonstrably made progress become
positive examples (`--include-failed` for pure imitation). Ticks inside a ping-pong — six or more
ticks repeating one label or alternating two with no item gained — teach no intent, and a
`stow_surplus` counts only when an item left the pack (see `docs/architecture.md`, M3).

Tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — design, layers, decisions, and the
  milestone-by-milestone findings from the live runs (M1–M4, homebase, Nether chain).
- [`docs/nether-prep.md`](docs/nether-prep.md) — the Nether preparation design and the
  phase 3 visit, with its live status.
- [`docs/operations.md`](docs/operations.md) — the runbook: world and MCP setup, how a run is
  started and archived, how a fix lands, how a head is trained.
- [`viewer/README.md`](viewer/README.md) — the live browser view of the bot (a spectator
  client with prismarine-viewer).
