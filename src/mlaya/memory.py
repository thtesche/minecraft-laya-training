from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MemoryLog:
    """Append-only JSONL decision log plus a simple MEMORY.md writer."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        rec = {"ts": time.time(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def write_memory(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
