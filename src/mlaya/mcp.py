from __future__ import annotations

import asyncio
import json
import re
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import MCPConfig

SETTLED = {
    "succeeded", "success", "failed", "error", "cancelled", "canceled",
    "completed", "complete", "settled", "done",
}


def clip(text: str, limit: int = 200) -> str:
    """One-line string, hard-capped so the server's maxLength(200) is honoured."""
    text = " ".join(str(text).split())
    return text[:limit]


@dataclass
class ToolResult:
    ok: bool
    text: str
    data: dict | None
    action_id: str | None
    settled: bool
    status: str | None = None
    raw: Any = None


def _find(obj: Any, *keys: str) -> Any:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k in keys:
                if k in cur and cur[k] is not None:
                    return cur[k]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _extract(res: Any) -> ToolResult:
    data = getattr(res, "structured_content", None)
    if data is None:
        data = getattr(res, "structuredContent", None)
    text = ""
    for c in getattr(res, "content", None) or []:
        t = getattr(c, "text", None)
        if t:
            text += t
    if data is None and text.lstrip().startswith("{"):
        try:
            parsed = json.loads(text)
            data = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            data = None
    ok = not bool(getattr(res, "is_error", False) or getattr(res, "isError", False))

    action_id = None
    status = None
    if isinstance(data, dict):
        action_id = _find(data, "actionId", "action_id")
        status = _find(data, "status")
    if action_id is None:
        m = re.search(r"Action ID[: ]+([0-9a-fA-F-]{8,})", text)
        if m:
            action_id = m.group(1)
    if status is None:
        m = re.search(r"Status:\W{0,4}([A-Za-z_]+)", text)
        if m:
            status = m.group(1).lower()
    settled = action_id is None or (status in SETTLED)
    return ToolResult(ok, text, data, action_id, settled, status, res)


class MCPExecutor:
    """Streamable-HTTP MCP client with the foreground gating state machine."""

    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg
        self._stack: AsyncExitStack | None = None
        self._sess: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._read_timeout = cfg.tool_timeout_ms / 1000

    async def __aenter__(self) -> "MCPExecutor":
        self._stack = AsyncExitStack()
        streams = await self._stack.enter_async_context(
            streamable_http_client(self.cfg.url, max_sse_event_size=self.cfg.max_sse_event_bytes)
        )
        read, write = streams
        self._sess = await self._stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=self._read_timeout)
        )
        await self._sess.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def list_tools(self) -> list:
        assert self._sess is not None
        return (await self._sess.list_tools()).tools

    async def _raw_call(self, name: str, args: dict) -> Any:
        assert self._sess is not None
        return await self._sess.call_tool(name, args, read_timeout_seconds=self._read_timeout)

    def _sub(self, tool: str) -> str:
        return f"{self.cfg.submission_prefix}{tool}-{uuid.uuid4().hex[:8]}"

    async def read(self, tool: str, args: dict, rationale: str) -> ToolResult:
        """Ungated read (view_status, read_recent_events, view_blocks, query_bot_data)."""
        payload = {**args, "rationale": clip(rationale), "response_format": "json"}
        return _extract(await self._raw_call(tool, payload))

    async def call(self, tool: str, args: dict, rationale: str) -> ToolResult:
        """Foreground action: serialised, with submission_id and result retrieval."""
        async with self._lock:
            payload = {
                **args,
                "submission_id": self._sub(tool),
                "rationale": clip(rationale),
                "wait_timeout_ms": self.cfg.initial_wait_ms,
                "response_format": "json",
            }
            res = _extract(await self._raw_call(tool, payload))
            if not res.ok:
                res = await self._recover(res)
            if res.action_id and not res.settled:
                res = await self._wait_settled(res.action_id)
            return res

    async def _recover(self, res: ToolResult) -> ToolResult:
        """ACTION_BUSY / lost action id: wait on whatever the server points at."""
        m = re.search(r"([0-9a-fA-F-]{8,})", res.text)
        if m and ("ACTION_BUSY" in res.text or "RESULT_NOT_RETRIEVED" in res.text
                  or "unretrieved" in res.text.lower()):
            return await self._wait_settled(m.group(1))
        return res

    async def _wait_settled(self, action_id: str) -> ToolResult:
        res = ToolResult(False, "", None, action_id, False)
        for _ in range(self.cfg.max_polls):
            res = _extract(await self._raw_call("wait_for_action", {
                "action_id": action_id,
                "timeout_ms": self.cfg.poll_ms,
                "rationale": "retrieve the preceding foreground result",
                "response_format": "json",
            }))
            if res.settled:
                return res
        return res
