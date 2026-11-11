---
license: apache-2.0
base_model: convaiinnovations/laya
tags:
  - laya
  - rlcd
  - system-one
  - minecraft
  - fine-tuned
  - decision-making
---

# Laya fine-tuned for Minecraft System-1 decisions

**A domain-specialised Laya checkpoint that decides what a Minecraft agent does next — in one forward pass, with calibrated probabilities.**

This is an RLCD fine-tune of the English Laya checkpoint ([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), ModernBERT-large) on decision ticks recorded by a live Minecraft agent. It is the System-1 brain of the agent: code filters the *valid* actions (preconditions, tech tree), and this model chooses among them — plus a binary danger gate — in a single non-autoregressive pass. It produces no text and no tool arguments, so there is nothing to parse and nothing to hallucinate.

- **Agent code:** [github.com/thtesche/minecraft-laya-training](https://github.com/thtesche/minecraft-laya-training)
- **Training corpora:** [datasets/thtesche/laya-minecraft-training-data](https://huggingface.co/datasets/thtesche/laya-minecraft-training-data)
- **This checkpoint:** the one served live by the agent (since 2026-11-11).

## How the agent uses it

The agent loop is a LangGraph: `perceive → decide → guard → execute → verify`.

1. **Perception** turns the MCP server's state (`view_status`, events, block scans) into a compact English narrative — milestones, inventory, vitals, threats, position.
2. **This model** answers two `choice` questions per tick:
   - `in_danger` — the binary safety gate (act now vs. work calmly),
   - `intent` — the next action among the *currently valid* options (the code has already removed impossible ones).
3. A Python executor resolves the chosen label into MCP tool calls; a verifier compares target vs. actual state.

Only ticks where the verifier confirmed real progress become positive training examples — the reward is the world, not a human label.

## Training

| | |
|---|---|
| Method | RLCD (Reinforcement Learning for Calibrated Decisions) — REINFORCE with group-mean baseline, strictly proper scoring rules |
| Base | `convaiinnovations/laya` (english), encoder frozen |
| Data | 6,459 decision-tick items from live runs 187–219, plus run 220 folded in through an offline teacher (flip-stable judge corrections) |
| Epochs / batch | 4 epochs, micro-batch 8, 3,232 optimizer updates |
| Learning rates | head 1e-4, encoder 2.5e-5 (cosine, min 1e-6) |
| Calibration | one temperature per (question type, option count), fitted on 400 held-out items (choice temperature 1.197) |

**Reward design details** that make the data trustworthy: ticks inside a ping-pong (six or more ticks repeating one label with no item gained) teach no intent and are filtered out; `stow_surplus` counts as progress only when an item actually left the pack.

**Offline teacher (run 220):** a second, text-generating model reviewed each decision after the fact under a neutral framing; a blind judge then compared its pick against the agent's. Only corrections that survived an A/B slot-swap flip test (the verdict does not reverse when the two actions change positions) were kept as labels — ~13 flip-stable corrections where the agent's choice made no progress. The whole corpus is re-rendered through today's narrative + criteria wording, so the head is trained on exactly the text it meets at inference.

## Evaluation

415 items from runs 206–208 (a period the training corpus does not cover), scored against this checkpoint's own frozen base (`2026-10-28`):

| Metric | Before fine-tune | After fine-tune |
|---|---|---|
| Accuracy | 0.9181 | 0.9108 |
| Mean confidence | 0.868 | 0.900 |
| ECE (lower is better) | 0.0585 | **0.0262** |
| Brier (lower is better) | 0.1470 | **0.1322** |
| Brier top-1 (lower is better) | 0.0765 | **0.0647** |

This fine-tune trades a small amount of top-1 accuracy (−0.7 pt on this set) for a large calibration gain: ECE drops by more than half and both Brier scores fall. Its headline value is not this held-out accuracy — it restores train=inference after a criteria-wording change and folds in the flip-stable teacher corrections (both of which live outside runs 206–208, so they cannot show here). Measured as deployed on the same re-rendered eval, the prior `2026-11-07` head scores 0.9205 accuracy / 0.0396 ECE; this one 0.9108 / 0.0262.

> **Honest caveat:** 6 of the 415 eval items overlap the training/calibration data, so this is not a strictly held-out evaluation. The full `train_report.json` (including abstention thresholds `choice:2 → 0.5181`, `choice:3-5 → 0.5907`) is in the repo.

## Checkpoint lineage

The agent has fine-tuned Laya six times as its playthroughs got harder. Earlier checkpoints are in the training-data dataset's history; this repo holds the current one:

| Checkpoint | Eval items | Accuracy | Held-out |
|---|---|---|---|
| 2026-10-11 | 6 | 0.833 | yes (smoke) |
| 2026-10-12 | 9 | 1.000 | yes (smoke) |
| 2026-10-14 | 61 | 0.934 | yes |
| 2026-10-23 | 279 | 0.885 | yes (runs 79–86) |
| 2026-10-28 | 522 | 0.918 | yes |
| 2026-11-07 | 415 | 0.933 | overlapping |
| **2026-11-11 (this repo)** | 415 | **0.911** | overlapping |

## Usage

**With the agent (drop-in):** the `laya` router lets a Hub ID re-point the built-in `english` checkpoint — no code change:

```bash
export LAYA_EXTRA_MODELS='{"english": "thtesche/laya-minecraft-english"}'
.venv/bin/python -m mlaya.cli run -n 10
```

**Directly with `laya`:**

```python
import laya
agent = laya.load("thtesche/laya-minecraft-english")   # downloads only model.safetensors, tokenizer/, encoder/
result = agent.predict(
    "Health is full. Hunger is high. Inventory: 3x iron_ingot, 1x diamond. No hostile mobs nearby.",
    {
        "intent": {"type": "choice", "instructions": "The bot is safe enough to work. What should it do next?",
                   "criteria": {"craft_sword": "craft an iron sword from the ingots carried",
                                "eat": "eat food to relieve hunger",
                                "mine_diamond": "mine more diamonds"}},
        "in_danger": {"type": "choice", "instructions": "Is the bot in immediate danger right now?",
                      "criteria": {"no": "no mob is attacking and health is high",
                                   "yes": "a mob is attacking now"}},
    },
)
print(result["answers"]["intent"]["choice"], result["answers"]["intent"]["confidence"])
```

**Pin a revision** for reproducible deployments: `LAYA_REVISION=<commit-sha>` (or pass `revision=` to `laya.load`).

## Repository layout

Exactly the layout `laya` expects:

```
model.safetensors          842 MB  fine-tuned weights (encoder + decision head)
rl_agent_config.json               head config: max_len 512, head_max_len 192, temperatures
encoder/config.json                ModernBERT-large encoder config
tokenizer/                         tokenizer.json + config
train_report.json                  full before/after metrics, calibration, training args
questions.json                     question signatures from training
```

## Limitations

- **Domain-bound:** the labels are Minecraft survival actions (craft, mine, navigate, sleep, flee…). It is not a general-purpose decision model.
- **Two questions only:** `intent` and `in_danger`; `score` and `noul` primitives are untrained here (0 items).
- **Confidence is calibrated for this domain** — the fitted temperature (1.197 for choice questions) applies to these question shapes, not arbitrary new schemas.
- **Endgame blind spot:** earlier checkpoints had no endgame rows; this one covers the Nether/stronghold chain only as far as runs 187–220 went.

## Citation

```bibtex
@misc{tesche2026layaminecraft,
  title = {Minecraft System-1 decisions with Laya: fine-tuned checkpoint and training corpora},
  author = {T. Tesche},
  year = {2026},
  url = {https://huggingface.co/thtesche/laya-minecraft-english},
  note = {Agent code: https://github.com/thtesche/minecraft-laya-training}
}
```
