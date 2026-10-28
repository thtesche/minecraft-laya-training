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
        decision = Brain(settings.laya).decide(gs.narrative(), cands, gs)
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
    try:
        async with MCPExecutor(settings.mcp) as executor:
            policy = await survival.apply_policy(executor, "apply policy at run start")
            agent = Agent(settings, executor)
            final = await agent.run()
    except perception.PerceptionBlind as exc:
        # One line, not a traceback: the world stopped being readable and the run stopped with it.
        # Anything else would send the user looking through a stack for a sentence about sockets.
        print(json.dumps({"error": "perception_blind", "detail": str(exc),
                          "log": str(settings.log_path())}, ensure_ascii=False))
        return 2
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
    logs = [_resolve(p) for p in args.log] if args.log else [settings.log_path()]
    missing = [str(p) for p in logs if not p.exists()]
    if missing:
        print(json.dumps({"error": "run log not found", "missing": missing}))
        return 2
    rows = ds.build_rows(
        logs,
        include_failed=args.include_failed,
        with_danger=not args.no_danger,
        limit=args.limit or None,
        rerender=args.rerender,
        reverify=args.reverify,
        teacher=[_resolve(p) for p in args.teacher] if args.teacher else None,
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
        "logs": [str(p) for p in logs],
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


def _teacher_out_path(log: Path) -> Path:
    """Where a run log's teacher judgments land: `decisions-runNN.jsonl` -> `teacher-runNN.jsonl`,
    beside the log, so the side-file sits next to the run it reviews."""
    name = log.name
    out = ("teacher-" + name[len("decisions-"):]) if name.startswith("decisions-") \
        else f"teacher-{log.stem}.jsonl"
    return log.parent / out


def cmd_teacher(args) -> int:
    """Replay an archived run log through the offline teacher path (System 2 + judge).

    Opinion only, written to a side-file `dataset` reads as an extra label source; the verifier
    stays primary. No world action, no touch of the live loop.
    """
    from . import dataset as ds
    from . import teacher as tmod

    settings = load_settings()
    log = _resolve(args.log)
    if not log.exists():
        print(json.dumps({"error": "run log not found", "path": str(log)}))
        return 2
    s2_key = "set" if settings.api_key() else f"MISSING ({settings.llm.api_key_env})"
    judge_key = "set" if settings.judge_api_key() else f"MISSING ({settings.judge.api_key_env})"
    records = ds._read_records(log)
    judgments = tmod.run_teacher(records, settings)

    out = _resolve(args.out) if args.out else _teacher_out_path(log)
    ds.write_jsonl(out, judgments)

    agree = sum(1 for j in judgments if j.get("agree"))
    disagree = sum(1 for j in judgments if j.get("s2_label") is not None and not j.get("agree"))
    verdicts: dict[str, int] = {}
    for j in judgments:
        v = j.get("judge_verdict")
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1
    labelled = sum(1 for j in judgments if j.get("judge_label"))
    print(json.dumps({
        "log": str(log),
        "teacher_path": str(out),
        "s2_key": s2_key,
        "judge_key": judge_key,
        "teachable_ticks": len(judgments),
        "agree": agree,
        "disagree": disagree,
        "judge_verdicts": verdicts,
        "judge_labelled": labelled,
    }, indent=2, ensure_ascii=False))
    return 0


def _train_base(settings, name: str = "") -> str:
    """The checkpoint a fine-tune starts from: `--base`, else `[train] base_model`.

    A name the run re-points through `LAYA_EXTRA_MODELS` resolves the same way here, so the
    head is trained on top of the checkpoint the loop actually served. Resolving the bare name
    instead fell back to the built-in `english` from the Hub, discarding every earlier fine-tune.
    """
    name = name or settings.train.base_model
    return str(settings.laya.extra_models.get(name, name))


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
        base = resolve_checkpoint_dir(_train_base(settings, args.base))
        print(json.dumps({"dry_run": True, "data": str(data), "eval_data": eval_path,
                          "base": base, "config": {"epochs": epochs, "loss": loss,
                                                   "freeze_encoder": cfg.freeze_encoder}},
                         indent=2, ensure_ascii=False))
        print(json.dumps(dry_run(str(data), base, cfg), indent=2, ensure_ascii=False, default=str))
        return 0

    base = resolve_checkpoint_dir(_train_base(settings, args.base))
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
    dset.add_argument("--log", action="append", default=[],
                      help="run log to read; repeat for several (default: the configured log)")
    dset.add_argument("--out", default="data/train.jsonl", help="training JSONL path")
    dset.add_argument("--eval", default="", help="held-out JSONL path (empty = write only train)")
    dset.add_argument("--eval-frac", type=float, default=0.25, help="eval split fraction")
    dset.add_argument("--seed", type=int, default=1234, help="split seed")
    dset.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    dset.add_argument("--include-failed", action="store_true",
                      help="also keep ticks whose action made no progress (plain imitation)")
    dset.add_argument("--no-danger", action="store_true",
                      help="omit in_danger rows (only if the gate is trained elsewhere)")
    dset.add_argument("--rerender", action="store_true",
                      help="re-render each tick's narrative from its logged state with today's "
                           "renderer (rebuild an archived corpus in the current wording)")
    dset.add_argument("--reverify", action="store_true",
                      help="recompute progressed from each tick's logged delta with today's "
                           "verifier (a rule added after the run re-labels the archived corpus)")
    dset.add_argument("--teacher", action="append", default=[],
                      help="teacher side-file (runs/history/teacher-runNN.jsonl) as an extra "
                           "label source; repeat for several. The verifier stays primary")

    tch = sub.add_parser("teacher", help="replay a run log through the offline teacher path "
                                         "(System 2 + blind judge) into a side-file")
    tch.add_argument("--log", required=True, help="archived run log to review")
    tch.add_argument("--out", default="",
                     help="teacher side-file (default: teacher-<run>.jsonl beside the log)")

    trn = sub.add_parser("train", help="fine-tune a Laya checkpoint head on the dataset")
    trn.add_argument("--data", default="", help="training JSONL (default data/train.jsonl)")
    trn.add_argument("--base", default="",
                     help="checkpoint to start from: a name or a directory (default [train] base_model)")
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
    if cmd == "teacher":
        return cmd_teacher(args)
    if cmd == "train":
        return cmd_train(args)
    return asyncio.run(cmd_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
