---
license: apache-2.0
tags:
  - minecraft
  - rlcd
  - decision-making
  - fine-tuning-data
---

# Minecraft System-1 decision logs — Laya fine-tuning corpora

Decision ticks recorded by a Minecraft agent whose System-1 brain is [Laya](https://huggingface.co/convaiinnovations/laya). Each line is one tick of the agent loop (`perceive → decide → execute → verify`): a compact English narrative of the world state, the typed questions asked of the model, and the label the verifier accepted.

**Reward semantics:** only ticks whose action demonstrably made progress become positive examples (pure imitation with `--include-failed` is possible in the builder). Ping-pong ticks (six or more ticks repeating one label with no item gained) teach no intent and are filtered out.

## Schema (JSONL, one object per line)

```json
{
  "state": "You are a Minecraft player ... Inventory: 4x acacia_log ...",
  "questions": {
    "intent":     {"type": "choice", "instructions": "...", "criteria": {"craft_pickaxe": "...", "craft_table": "..."}},
    "in_danger":  {"type": "choice", "instructions": "...", "criteria": {"no": "...", "yes": "..."}}
  },
  "expected": {"intent": "craft_pickaxe", "in_danger": "no"}
}
```

## Files

| File | Items | What it is |
|---|---|---|
| `train.jsonl` | 4,631 | Main corpus (runs up to Oct 2026) |
| `eval.jsonl` | 12 | Small held-out eval |
| `ood-eval.jsonl` | 33 | Out-of-distribution eval (endgame states the base model has never seen) |
| `run6.jsonl`, `run6-eval.jsonl`, `run6-preview.jsonl` | 56 / 18 / 56 | Earliest smoke dataset (run 6) |
| `run79-86-train.jsonl`, `run79-86-eval.jsonl` | 512 / 171 | M3 second round (runs 79–86) |
| `train-187-194.jsonl`, `eval-187-194.jsonl` | 609 / 203 | Runs 187–194 |
| `train-187-205.jsonl`, `eval-206-208.jsonl` | 1,509 / 219 | Runs 187–205 (basis of the 2026-11-07 checkpoint) |
| `train-187-220-rr.jsonl`, `eval-206-208-rr.jsonl` | 3,587 / 219 | Runs 187–219 + run 220 (offline-teacher corrected), re-rendered through today's narrative + criteria wording — basis of the **2026-11-11** checkpoint |
| `relative-train.jsonl`, `relative-eval.jsonl` | 849 / 283 | Relative-narrative experiment (M5) |
| `train-smoke.jsonl`, `train-test.jsonl` | 38 / 116 | Smoke/test sets for the training pipeline |

## How the data was built

Run logs (`decisions.jsonl` per run) are turned into these corpora with the agent's dataset builder:

```bash
.venv/bin/python -m mlaya.cli dataset --log runs/history/decisions-run79.jsonl \
    --log runs/history/decisions-run80.jsonl --out train.jsonl --eval eval.jsonl
```

Agent code: [github.com/thtesche/minecraft-laya-training](https://github.com/thtesche/minecraft-laya-training) · Fine-tuned weights: [thtesche/laya-minecraft-english](https://huggingface.co/thtesche/laya-minecraft-english)
