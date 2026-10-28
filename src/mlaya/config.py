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
    # a different judge removes the bias of one model grading its own earlier opinion - but it
    # defaults to the same Strata server so the path works out of the box. The key comes from
    # the environment (.env) like System 2's, its own env name so the two can differ.
    model: str = "qwen3.8-flash-next-q2_0"
    api_key_env: str = "STRATA_API_KEY"
    max_tokens: int = 4096
    base_url: str = "http://192.168.0.109:8080/v1"


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
    max_escalations: int = 3
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
    memory_path: str = "MEMORY.md"


class Settings(BaseModel):
    llm: LLMConfig = LLMConfig()
    judge: JudgeConfig = JudgeConfig()
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

    def log_path(self) -> Path:
        p = Path(self.loop.log_path)
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
