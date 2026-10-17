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
| Nether visit (phase 3) | **open** — see [`docs/nether-prep.md`](docs/nether-prep.md) |

The run **stops itself** once the overworld goal stands
(`actions.tech_tree_complete`): full diamond kit **worn** + shield in the off-hand, a bed
at the base, all Nether prep steps done and the food reserve banked. The portal is lit
**last** — `light_portal` waits for that whole bundle (`_expedition_ready`) and for a night
that can lay the carried bed at once, so the gate never burns while the bot still works at
the base beside it; the bed goes down the very next tick and the run ends. It then reports
`"stop": "goal"`; with `--ignore-goal` (or
`[loop] stop_when_complete = false`) it runs to the iteration limit instead.

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

Configuration: [`config/settings.toml`](config/settings.toml). Every decision is logged
to `runs/decisions.jsonl`.

### M3: logs → dataset → fine-tuning

```bash
# 1) Build a Laya training dataset from the run log (train + held-out eval)
.venv/bin/python -m mlaya.cli dataset            # -> data/train.jsonl + data/eval.jsonl
.venv/bin/python -m mlaya.cli dataset --include-failed   # also keep ticks without progress
.venv/bin/python -m mlaya.cli dataset --no-danger        # without `in_danger` rows

# 2) Check the dataset format (loads no weights), then retrain the head (RLCD)
.venv/bin/python -m mlaya.cli train --dry-run
.venv/bin/python -m mlaya.cli train              # -> runs/checkpoints/minecraft-<time>

# 3) Register the checkpoint under `english` and run the loop with it
export LAYA_EXTRA_MODELS='{"english": "'"$PWD"'/runs/checkpoints/minecraft-<time>"}'
.venv/bin/python -m mlaya.cli run -n 10
```

`LAYA_EXTRA_MODELS` is a JSON object `{name: source}`; a name that matches a built-in
checkpoint (`english`) **re-points it** — the fine-tune is served without a code change.
The reward is the verifier: only ticks whose action demonstrably made progress become
positive examples (`--include-failed` for pure imitation).

Tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — design, layers, decisions, and the
  milestone-by-milestone findings from the live runs (M1–M4, homebase, Nether chain).
- [`docs/nether-prep.md`](docs/nether-prep.md) — the Nether preparation design and the
  open phase 3 (the actual visit).
