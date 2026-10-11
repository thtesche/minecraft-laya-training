from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "settings.toml"


class LLMConfig(BaseModel):
    model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    api_key_env: str = "OPENROUTER_API_KEY"
    max_tokens: int = 16384
    base_url: str = "https://openrouter.ai/api/v1"


class MCPConfig(BaseModel):
    url: str = "http://localhost:25575/mcp"
    health_url: str = "http://localhost:25575/health"
    tool_timeout_ms: int = 3_600_000
    initial_wait_ms: int = 5_000
    poll_ms: int = 2_000
    max_polls: int = 900
    max_sse_event_bytes: int = 16_777_216
    submission_prefix: str = ""


class LayaConfig(BaseModel):
    model: str = "multilingual"
    preload: bool = False
    device: str = ""
    min_confidence: float | None = None
    max_len: int = 0


class LoopConfig(BaseModel):
    max_iterations: int = 10
    max_escalations: int = 3
    hitl: bool = False
    log_path: str = "runs/decisions.jsonl"
    memory_path: str = "MEMORY.md"


class Settings(BaseModel):
    llm: LLMConfig = LLMConfig()
    mcp: MCPConfig = MCPConfig()
    laya: LayaConfig = LayaConfig()
    loop: LoopConfig = LoopConfig()

    @property
    def root(self) -> Path:
        return ROOT

    def api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env)

    def log_path(self) -> Path:
        p = Path(self.loop.log_path)
        return p if p.is_absolute() else ROOT / p

    def memory_path(self) -> Path:
        p = Path(self.loop.memory_path)
        return p if p.is_absolute() else ROOT / p


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_PATH
    data: dict = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Settings(**data)
