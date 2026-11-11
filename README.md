# minecraft-laya-training

A Minecraft agent whose **System-1 decisions are made by
[Laya](https://github.com/NandhaKishorM/laya)** — a non-autoregressive
typed-decision engine (`choice` / `score` / `noul`) — and which acts through the
**mine-ai MCP server**. A generative LLM (the local Strata server) serves as **System 2**
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
| Actions | `src/mlaya/actions/` | `candidates(gs)`: valid labels by precondition; `resolve(label, gs)`: label → MCP tool + arguments (split into `constants` / `predicates` / `planner`) |
| Verify | `src/mlaya/verify.py` | the target/actual delta decides whether an action made progress |
| Memory | `src/mlaya/memory.py` | every decision + outcome in `runs/decisions.jsonl` |
| Survival | `src/mlaya/survival.py` | sets the server policy (`base` / `hunt`) at start and after every death |
| System 2 | `brain.escalate_to_system2` | Strata (local) picks a valid alternative when a label cannot be resolved |

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
| Nether safety (staged exit, food security, per-trip requirement) | implemented; staged exit (120 m hops), food security gate (`NETHER_FOOD_MIN_RESERVE = 24`), per-trip checks — see [`docs/nether-prep.md`](docs/nether-prep.md) |
| Storage network — chests per cell, overflow chests, provisions fetched back | reached, validated live |
| Stall guards — refusals, wandering, idle steps, quiet defers, `wait` hold | implemented (see [`docs/architecture.md`](docs/architecture.md#stall-guards)) |
| Equipment durability audit and replacement crafting | implemented; worn-out armor (below 15%) triggers replacement craft from reserve — see [`docs/architecture.md`](docs/architecture.md#equipment-durability-audit-and-replacement-crafting) |
| Nether visit (phase 3) — crossing, fortress search, blaze rods home as powder | implemented, **closed live**: run187 lit the gate (it510), crossed (it511) and banked the powder in the gate chest (it540) — see [`docs/nether-prep.md`](docs/nether-prep.md) |
| Stronghold → End portal (phase 4) — eyes of ender, stronghold search, end-portal activation, demand-sized re-supply, respawn bed beside the frame | implemented, walked live (run184 filled the frame, run186 laid and slept in the frame bed, run208 activated the portal); the whole chain has not yet been closed from an empty world — see [`docs/stronghold.md`](docs/stronghold.md) |
| The End → dragon (phase 5) — pumpkin and MLG water, enter the portal, crystals first, bow for the flying dragon and a melee perch, `end_complete` as the final goal | implemented, **not yet walked live** (no run has entered the End) — see [`docs/architecture.md`](docs/architecture.md#phase-5--the-end-and-the-dragon-2026-11-08-71c9b63) |
| M3, second round — run79–86 head on top of the 2026-10-14 checkpoint | trained (held-out accuracy 0.767 → 0.885), served live since run87 via `LAYA_EXTRA_MODELS` |
| M5 — stronger training signal: relative narrative + offline teacher/judge labels | planned (see [`docs/architecture.md`](docs/architecture.md#next-milestones)) |

The run **stops itself** once every phase stands (`Agent._goal_reached`): the overworld goal
(`actions.tech_tree_complete` — full diamond kit **worn** + shield in the off-hand, a bed at the
base, all Nether prep steps done, the food reserve banked and the gate lit) **and** the Nether trip
(`actions.nether_expedition_complete` — the blaze powder deposited in the chest at the gate) **and**
the stronghold phase (`stronghold_complete` — the end portal active — **plus** the sleep latch that
proves the respawn was pinned at the frame; the live bed scan is deliberately not read there, since
it is false the moment the body is in the End) **and** Phase 5 (`end_complete` — the dragon dead,
latched from the kill receipt). The portal is lit
**last** in the overworld: `light_portal` waits for the whole stocked trip, so the gate never burns
through the prep window. The carried bed is laid the very next tick, by day or by night — the
`sleep` server sets the respawn point at a bed used in daylight too — and that opens the crossing.
The run then reports `"stop": "goal"`; a run that never gets the powder home ends with `"stop":
"limit"`, and `--ignore-goal` (or `[loop] stop_when_complete = false`) always runs to the iteration
limit.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env        # enter STRATA_API_KEY (only for System-2 escalation)
./scripts/install-hooks.sh  # install pre-commit hook (runs tests before each commit)
```

`.env` is loaded automatically on every CLI run (real environment variables take
precedence) and is gitignored. Prerequisite: the mine-ai MCP server running at
`http://localhost:25575/mcp`.

Tests run automatically on every `git commit` (via pre-commit hook) and on every push
via GitHub Actions (CI).

## Usage

The lab is the everyday entry point — it starts the Docker world, the MCP host from the
`bot-run` branch, the run, and writes the report when the run ends:

```bash
./scripts/lab.sh up --n 600 --fresh     # fresh world + ledger, then a run
./scripts/lab.sh status                 # world, host, run, last tick as JSON
./scripts/lab.sh report 187             # the finished run's report
./scripts/lab.sh down                   # stop run, host, world
```

The bare CLI, for probing and single decisions against a host that is already up:

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
to `runs/decisions.jsonl`; `lab.sh` archives it to `runs/history/decisions-runNN.jsonl` the moment
a new run starts. To read that log while a run is live: `tail -f runs/decisions.jsonl |
scripts/dec.py` (one key per line at level 1, everything deeper compact — see
[`docs/operations.md`](docs/operations.md#reading-the-log-while-it-is-written)).
Every System-2 escalation also writes a record to its own side-file `runs/system2.jsonl` (the full
brief that was sent, the plan that came back, and whether it resolved), archived per run to
`runs/history/system2-runNN.jsonl`; `mlaya.analysis` joins it back to the decision log by iteration
to score the escalations (resolve / recurrence / error rate).

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

### Publishing to Hugging Face

The served checkpoint and the training corpora are public:

- **[thtesche/laya-minecraft-english](https://huggingface.co/thtesche/laya-minecraft-english)** — the current checkpoint (2026-11-11), full model card, lineage of all seven fine-tunes
- **[datasets/thtesche/laya-minecraft-training-data](https://huggingface.co/datasets/thtesche/laya-minecraft-training-data)** — every train/eval JSONL with schema and reward semantics

A Hub id is a valid `LAYA_EXTRA_MODELS` source, so the loop can run straight from the Hub:

```bash
export LAYA_EXTRA_MODELS='{"english": "thtesche/laya-minecraft-english"}'
```

To publish a new checkpoint: `hf upload thtesche/laya-minecraft-english runs/checkpoints/minecraft-<time> --exclude "checkpoint_latest/*"`, then refresh the card (`hf/model-card.md`). Weights must go to a **model** repo — laya's loader defaults to `repo_type="model"` and does not resolve dataset repos.

Tests: `uv run pytest tests/ -q` (1755 tests) — the same command the pre-commit hook and CI run.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — design, layers, decisions, and the
  milestone-by-milestone findings from the live runs (M1–M4, homebase, Nether chain, the stronghold
  hand-off and the End phase).
- [`docs/nether-prep.md`](docs/nether-prep.md) — the Nether preparation design and the
  phase 3 visit, with its live status.
- [`docs/stronghold.md`](docs/stronghold.md) — the phase after the Nether haul: eyes of ender,
  the stronghold, the end portal, and the demand-sized re-supply.
- [`docs/restart-recovery.md`](docs/restart-recovery.md) — what survives a cold start (world,
  inventory, chests) and what the loop re-derives (the milestone latches), with the stronghold
  phase as the worked example.
- [`docs/operations.md`](docs/operations.md) — the runbook: world and MCP setup, how a run is
  started and archived, how a fix lands, how a head is trained.
- [`viewer/README.md`](viewer/README.md) — the live browser view of the bot (a spectator
  client with prismarine-viewer).

## License

[Apache License 2.0](LICENSE) throughout — the agent code in this repository as well as the
published Laya checkpoints and training corpora on the Hub, consistent with the
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) base model they derive from.
