from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from . import perception
from . import survival
from .actions import candidates, resolve
from .config import ROOT, load_env, load_settings
from .graph import Agent
from .mcp import MCPExecutor


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


async def _health(settings) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(settings.mcp.health_url)
        r.raise_for_status()
        return r.json()


async def cmd_probe(args) -> int:
    settings = load_settings()
    key = settings.api_key()
    env_file = "yes" if (ROOT / ".env").exists() else "no"
    print(f"env  .env={env_file}  {settings.llm.api_key_env}="
          f"{'set' if key else 'MISSING (System-2 escalation disabled)'}")
    health = await _health(settings)
    mc = health.get("minecraft", {})
    print(f"health ok  transport={health.get('transport')}  bot={mc.get('username')} "
          f"pos={mc.get('position')} food={mc.get('food')} busy={mc.get('busy')}")
    async with MCPExecutor(settings.mcp) as executor:
        tools = await executor.list_tools()
        print(f"tools: {len(tools)}")
        for t in tools:
            print("  -", t.name)
    return 0


async def cmd_decide(args) -> int:
    """Perceive and decide once, without executing a world action."""
    settings = load_settings()
    await _health(settings)
    from .brain import Brain

    async with MCPExecutor(settings.mcp) as executor:
        gs = await perception.build(executor)
        cands = candidates(gs)
        decision = Brain(settings.laya).decide(gs.narrative(), cands)
        action = resolve(decision.action_label, gs)
    print(json.dumps({
        "narrative": gs.narrative(),
        "state": gs.as_state(),
        "candidates": cands,
        "deferred": decision.deferred,
        "note": decision.note,
        "label": decision.action_label,
        "answers": decision.answers,
        "resolved_tool": action.tool if action else None,
        "resolved_args": action.args if action else None,
        "dry_run": "no action executed",
    }, indent=2, ensure_ascii=False, default=str))
    return 0


async def cmd_run(args) -> int:
    settings = load_settings()
    iterations = getattr(args, "iterations", 0) or settings.loop.max_iterations
    settings.loop.max_iterations = iterations
    if getattr(args, "ignore_goal", False):
        settings.loop.stop_when_complete = False
    await _health(settings)
    async with MCPExecutor(settings.mcp) as executor:
        policy = await survival.apply_policy(executor, "apply policy at run start")
        agent = Agent(settings, executor)
        final = await agent.run()
    print(json.dumps({
        "iterations": final.get("iteration"),
        "stop": final.get("stop"),
        "note": final.get("note"),
        "policy": policy,
        "log": str(settings.log_path()),
    }, default=str))
    return 0


async def cmd_policy(args) -> int:
    """Show the server survival policy, optionally applying the working set."""
    settings = load_settings()
    await _health(settings)
    async with MCPExecutor(settings.mcp) as executor:
        applied = None
        if getattr(args, "apply", False):
            applied = await survival.apply_policy(executor, "manual survival policy apply",
                                                  preset=getattr(args, "preset", "base"))
        policy = await survival.read_policy(executor)
    print(json.dumps({
        "applied": applied,
        "revision": policy.get("revision"),
        "lastChange": policy.get("lastChange"),
        "effective": policy.get("effective"),
        "overrides": policy.get("overrides"),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_dataset(args) -> int:
    """Build Laya training JSONL from the run log (M3)."""
    from . import dataset as ds

    settings = load_settings()
    rows = ds.build_rows(
        settings.log_path(),
        include_failed=args.include_failed,
        with_danger=not args.no_danger,
        limit=args.limit or None,
    )
    train, ev = ds.split(rows, args.eval_frac, args.seed)

    out = _resolve(args.out)
    ds.write_jsonl(out, train)

    n_danger_yes = sum(1 for r in rows if r["expected"].get(ds.DANGER_QID) == "yes")
    labels: dict[str, int] = {}
    for r in rows:
        lab = r["expected"].get(ds.INTENT_QID)
        labels[lab] = labels.get(lab, 0) + 1

    result = {
        "log": str(settings.log_path()),
        "rows": len(rows),
        "train": len(train),
        "eval": len(ev),
        "train_path": str(out),
        "danger_yes": n_danger_yes,
        "intent_labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
    }
    if ev:
        ev_path = _resolve(args.eval or (settings.train.data_dir + "/eval.jsonl"))
        ds.write_jsonl(ev_path, ev)
        result["eval_path"] = str(ev_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    """Fine-tune a Laya checkpoint head on the built dataset (M3)."""
    from laya.train import TrainConfig, dry_run, finetune, resolve_checkpoint_dir

    settings = load_settings()
    tcfg = settings.train
    data = _resolve(args.data if args.data else (tcfg.data_dir + "/train.jsonl"))
    if not data.exists():
        print(json.dumps({"error": f"dataset not found: {data}",
                          "hint": "run `mlaya dataset` first"}))
        return 2

    eval_path = args.eval
    if eval_path is None:
        default_eval = _resolve(tcfg.data_dir + "/eval.jsonl")
        eval_path = str(default_eval) if default_eval.exists() else None
    elif eval_path == "":
        eval_path = None
    else:
        eval_path = str(_resolve(eval_path))

    epochs = args.epochs or tcfg.epochs
    loss = args.loss or tcfg.loss
    cfg = TrainConfig(
        epochs=epochs,
        micro_batch=tcfg.micro_batch,
        grad_accum=tcfg.grad_accum,
        loss=loss,
        freeze_encoder=not args.train_encoder,
        shuffle_options=("choice",) if not args.no_shuffle else (),
        eval_data=eval_path,
    )

    if args.dry_run:
        base = resolve_checkpoint_dir(tcfg.base_model)
        print(json.dumps({"dry_run": True, "data": str(data), "eval_data": eval_path,
                          "base": base, "config": {"epochs": epochs, "loss": loss,
                                                   "freeze_encoder": cfg.freeze_encoder}},
                         indent=2, ensure_ascii=False))
        print(json.dumps(dry_run(str(data), base, cfg), indent=2, ensure_ascii=False, default=str))
        return 0

    base = resolve_checkpoint_dir(tcfg.base_model)
    out = _resolve(args.out) if args.out else _resolve(
        f"{tcfg.out_dir}/minecraft-{time.strftime('%Y%m%d-%H%M%S')}")
    summary = finetune(str(data), base, str(out), cfg, device=args.device or "auto")

    register = json.dumps({tcfg.base_model: str(out)})
    print(json.dumps({
        "base": base,
        "checkpoint": str(out),
        "summary": summary,
        # A name matching a built-in checkpoint re-points it, so the loop picks up the
        # fine-tune under `english` with no code change.
        "register": f"LAYA_EXTRA_MODELS='{register}'",
    }, indent=2, ensure_ascii=False, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser("mlaya")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("probe", help="health check + tools/list")
    sub.add_parser("decide", help="perceive + decide once (dry run, no action)")
    run = sub.add_parser("run", help="run the decision loop")
    run.add_argument("-n", "--iterations", type=int, default=0)
    run.add_argument("--ignore-goal", action="store_true",
                     help="keep looping past a completed tech tree (dataset collection)")
    pol = sub.add_parser("policy", help="show/apply the server survival policy")
    pol.add_argument("--apply", action="store_true", help="apply a policy stage")
    pol.add_argument("--preset", choices=["base", "hunt"], default="base",
                     help="base (cautious travel) or hunt (aggressive combat floor)")

    dset = sub.add_parser("dataset", help="build Laya training JSONL from the run log")
    dset.add_argument("--out", default="data/train.jsonl", help="training JSONL path")
    dset.add_argument("--eval", default="", help="held-out JSONL path (empty = write only train)")
    dset.add_argument("--eval-frac", type=float, default=0.25, help="eval split fraction")
    dset.add_argument("--seed", type=int, default=1234, help="split seed")
    dset.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    dset.add_argument("--include-failed", action="store_true",
                      help="also keep ticks whose action made no progress (plain imitation)")
    dset.add_argument("--no-danger", action="store_true",
                      help="omit in_danger rows (only if the gate is trained elsewhere)")

    trn = sub.add_parser("train", help="fine-tune a Laya checkpoint head on the dataset")
    trn.add_argument("--data", default="", help="training JSONL (default data/train.jsonl)")
    trn.add_argument("--eval", default=None, help="held-out JSONL (default auto-detect)")
    trn.add_argument("--out", default="", help="checkpoint output dir")
    trn.add_argument("--epochs", type=int, default=0)
    trn.add_argument("--loss", choices=["rlcd", "soft-ce"], default="")
    trn.add_argument("--device", default="", help="auto|cpu|mps|cuda")
    trn.add_argument("--train-encoder", action="store_true",
                     help="also train the encoder (default: heads only)")
    trn.add_argument("--no-shuffle", action="store_true",
                     help="disable option-order augmentation for choice questions")
    trn.add_argument("--dry-run", action="store_true",
                     help="inspect the dataset and build items without loading weights")

    args = parser.parse_args(argv)
    cmd = args.cmd or "run"
    if cmd == "probe":
        return asyncio.run(cmd_probe(args))
    if cmd == "decide":
        return asyncio.run(cmd_decide(args))
    if cmd == "policy":
        return asyncio.run(cmd_policy(args))
    if cmd == "dataset":
        return cmd_dataset(args)
    if cmd == "train":
        return cmd_train(args)
    return asyncio.run(cmd_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
