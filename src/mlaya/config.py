from __future__ import annotations

import os
import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "settings.toml"
ENV_PATH = ROOT / ".env"


def load_env(path: Path | None = None, override: bool = False) -> bool:
    """Load ROOT/.env into os.environ. Returns True if a file was loaded.

    Real environment variables win (override=False); the .env is a convenience
    for local runs. Safe to call repeatedly.
    """
    from dotenv import load_dotenv

    p = path or ENV_PATH
    if not p.exists():
        return False
    load_dotenv(p, override=override)
    return True


class LLMConfig(BaseModel):
    # System 2 escalation: the local Strata server (OpenAI-compatible), mirroring the
    # provider configured in OpenCode. The key comes from the environment (.env).
    model: str = "qwen3.8-flash-next-q2_0"
    api_key_env: str = "STRATA_API_KEY"
    max_tokens: int = 4096
    base_url: str = "http://192.168.0.109:8080/v1"


class JudgeConfig(BaseModel):
    # The third LLM in the offline teacher path (`teacher.py`): it adjudicates, blind, only
    # when System 1 (Laya) and System 2 disagree. A separate model from `[llm]` is the point -
    # a different judge removes the bias of one model grading its own earlier opinion. The
    # default is OpenRouter's free Nemotron 3.5 Super VL (`nvidia/nemotron-3.5-super-vl:free`), a
    # genuinely independent model from the local Strata System 2, reached over its OpenAI-compatible
    # endpoint. The key comes from the environment (.env, `OPENROUTER_API_KEY`); without it the
    # judge returns "unsure" and the teacher path still runs. `config/settings.toml` overrides this.
    model: str = "nvidia/nemotron-3.5-super-vl:free"
    api_key_env: str = "OPENROUTER_API_KEY"
    max_tokens: int = 4096
    base_url: str = "https://openrouter.ai/api/v1"


class ObserverConfig(BaseModel):
    # The parallel observer: a persistent LLM that narrates and judges the run in parallel to Laya,
    # read-only, never touching the bot. It keeps ONE append-only "story" conversation so the Strata
    # prefix cache stays warm tick after tick (scripts/cache_probe.py: a branch caches 0, an append
    # reuses the whole prior prompt). The ceilings below are the generous ones from the Strata model
    # definition in OpenChamber (context 200k / input 185k / output 16384, reasoning high) - the live
    # System-2 `[llm]` stays at its own, tighter budget.
    model: str = "qwen3.8-flash-next-q2_0"
    api_key_env: str = "STRATA_API_KEY"
    base_url: str = "http://192.168.0.109:8080/v1"
    max_tokens: int = 16_384          # the model's output ceiling, for the full-reasoning event turns
    # Routine /no_think ticks still write a full JSON verdict; the smoke test saw them reach ~480
    # completion tokens and a 512 cap truncated the JSON mid-reply (an unparseable verdict). This
    # leaves headroom without inviting the model to ramble past the terse note the prompt asks for.
    routine_max_tokens: int = 1_024
    context_tokens: int = 200_000     # the model's context window
    # Fold the oldest story turns into a summary before this. Set well below the model's 185k input
    # ceiling on purpose: at ~200-250 tokens/tick a run rarely reaches 185k, so a budget up there
    # would fold ~never and the prompt would just grow unbounded. 80k keeps the live prompt lean -
    # it folds periodically (every ~300 teachable ticks) at the price of a few more cache misses,
    # each cheap: only the kept tail re-prefills, the byte-stable system prefix never does.
    input_budget: int = 80_000
    # How many of the most recent story turns a fold keeps verbatim behind the checkpoint. The
    # folded-away older turns are replaced by one model-written "story so far" hand-off.
    keep_turns: int = 40
    reasoning_effort: str = "high"    # for the full-reasoning event turns; routine ticks run /no_think
    log_path: str = "runs/observer.jsonl"
    # How far behind the decisions log the tailer may fall before routine ticks are sampled rather
    # than each observed (events - phase change, escalation, stall, death - are never skipped).
    max_lag: int = 8
    poll_s: float = 1.0


class MCPConfig(BaseModel):
    url: str = "http://localhost:25575/mcp"
    health_url: str = "http://localhost:25575/health"
    tool_timeout_ms: int = 3_600_000
    initial_wait_ms: int = 5_000
    poll_ms: int = 2_000
    max_polls: int = 900
    # How long one foreground call may wait in total for its action - and for any earlier action
    # it has to drain first - before the tick gives up with `[BODY_HELD]`. run89 sat in
    # `wait_for_action` for 31 and then 93 minutes per tick: the server's hostile reflex held the
    # body in a fight it could not finish, and every drain ran the full `max_polls`, up to three
    # times over.
    action_wait_ms: int = 300_000
    max_sse_event_bytes: int = 16_777_216
    submission_prefix: str = ""


class LayaConfig(BaseModel):
    model: str = "multilingual"
    preload: bool = False
    device: str = ""
    min_confidence: float | None = None
    max_len: int = 0
    # Checkpoint registrations, normally fed from the LAYA_EXTRA_MODELS env var
    # (a JSON object {name: source}); a name that matches a built-in re-points it,
    # which is how a fine-tune is served under `english` with no code change.
    extra_models: dict[str, Any] = {}
    # Craft the pre-diamond iron armour set before the first diamond dive. Default off: early runs
    # died to mobs without it, which is why the kit used to be mandatory - but with the diamond
    # armour that follows, the detour cost a run most of its budget (run246 reached diamond only at
    # the buzzer). Off, the kit is the iron sword and shield only and the dive goes behind those,
    # then the run wears diamond armour. Turned on per run by `--iron-armor` (cli.cmd_run), injected
    # onto the GameState by the loop (graph._inject), read by actions._combat_ready and the planner.
    iron_armor: bool = False


class TrainConfig(BaseModel):
    """Defaults for the M3 fine-tune (see `laya.train.TrainConfig`)."""

    base_model: str = "english"
    out_dir: str = "runs/checkpoints"
    data_dir: str = "data"
    epochs: int = 4
    micro_batch: int = 8
    # Laya's notebook uses grad_accum=8 (effective batch 64) for large corpora. Our log is
    # tiny, so one accumulation step keeps the optimizer moving.
    grad_accum: int = 1
    loss: str = "rlcd"  # rlcd | soft-ce
    freeze_encoder: bool = True


class LoopConfig(BaseModel):
    max_iterations: int = 10
    # No per-run cap on System-2 escalations: every stall trigger (the stall-loop and the idle
    # pass-time guard) asks the model whenever a label keeps looping, and the plan ledger runs the
    # remaining steps of an already-paid plan before a new call is spent. A run that keeps tripping
    # the same loop is now escalated every time rather than being throttled.
    hitl: bool = False
    # End the loop as soon as the overworld tech tree is complete (diamond tools + armour
    # worn + shield + lit portal + bed + the Nether prep banked), instead of walking the full
    # iteration budget. See actions.tech_tree_complete and actions._all_nether_prep_steps.
    stop_when_complete: bool = True
    # Seconds a successful `wait` holds before the next decision. The step settles at once (a
    # navigate to the anchor it already stands on), so without the hold every tick was one
    # second and run86 spent ~300 of its 600 iterations waiting for one night.
    wait_hold_s: float = 30.0
    log_path: str = "runs/decisions.jsonl"
    # The dedicated side-file for System-2 calls: one record per escalation with the full brief
    # that was sent, the plan that came back, and whether it resolved. Kept apart from the main
    # decision log so the escalations can be read and scored on their own (`mlaya.analysis` joins
    # the two by iteration), and archived per run alongside `decisions.jsonl`.
    system2_log_path: str = "runs/system2.jsonl"
    memory_path: str = "MEMORY.md"
    # Opt-in (`--use-observer-better`): run the parallel observer in-process and, on a disagree,
    # take its proposed `better` label on the *next* tick instead of only logging it for the
    # training set. Off, the observer stays an out-of-process, read-only narrator and the live loop
    # never pays its per-tick call. The override is capped and resolvable-only (see graph._guard /
    # _observer_override), so the worst case is the very action Laya would have run anyway.
    use_observer_better: bool = False


class Settings(BaseModel):
    llm: LLMConfig = LLMConfig()
    judge: JudgeConfig = JudgeConfig()
    observer: ObserverConfig = ObserverConfig()
    mcp: MCPConfig = MCPConfig()
    laya: LayaConfig = LayaConfig()
    loop: LoopConfig = LoopConfig()
    train: TrainConfig = TrainConfig()

    @property
    def root(self) -> Path:
        return ROOT

    def api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env)

    def judge_api_key(self) -> str | None:
        return os.environ.get(self.judge.api_key_env)

    def observer_api_key(self) -> str | None:
        return os.environ.get(self.observer.api_key_env)

    def observer_log_path(self) -> Path:
        p = Path(self.observer.log_path)
        return p if p.is_absolute() else ROOT / p

    def log_path(self) -> Path:
        p = Path(self.loop.log_path)
        return p if p.is_absolute() else ROOT / p

    def system2_log_path(self) -> Path:
        p = Path(self.loop.system2_log_path)
        return p if p.is_absolute() else ROOT / p

    def memory_path(self) -> Path:
        p = Path(self.loop.memory_path)
        return p if p.is_absolute() else ROOT / p


def _extra_models_from_env() -> dict[str, Any]:
    """Checkpoint registrations from LAYA_EXTRA_MODELS (a JSON object of name -> source).

    A malformed value has no harmless fallback: a checkpoint that silently fails to
    register would auto-route to another one, so this raises rather than drop it (the same
    choice laya.serve makes).
    """
    raw = os.environ.get("LAYA_EXTRA_MODELS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"invalid LAYA_EXTRA_MODELS: not JSON ({e})") from None
    if not isinstance(data, dict):
        raise ValueError("invalid LAYA_EXTRA_MODELS: a JSON object {name: source} is required")
    return data


def load_settings(path: Path | None = None) -> Settings:
    load_env()
    path = path or CONFIG_PATH
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = Settings(**data)
    # Environment wins over the file: LAYA_EXTRA_MODELS is the documented registration
    # mechanism, so a checkpoint named there must reach the Router.
    extra = _extra_models_from_env()
    if extra:
        settings.laya.extra_models = extra
    return settings
