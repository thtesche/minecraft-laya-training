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


def _held(res: ToolResult) -> ToolResult:
    """The refusal a foreground call ends with when its action did not settle in time.

    Names who holds the body and in what, as far as the last progress answer says: run89's hunt
    sat behind `hostile_reflex` / `fight` / `hold` for 55,712 ticks against a zombie 1.6 m off.
    """
    data = res.data if isinstance(res.data, dict) else {}
    survival = _find(data, "survival") if data else None
    owner = response = None
    if isinstance(survival, dict):
        owner = (survival.get("owner") or {}).get("current")
        response = survival.get("response") or None
    parts = [f"owner {owner or 'unknown'}"]
    if isinstance(response, dict):
        parts.append(f"{response.get('kind')}/{response.get('phase')}")
        if response.get("phaseTicks") is not None:
            parts.append(f"{response.get('phaseTicks')} ticks")
    text = (f"[BODY_HELD] action {res.action_id} still running after the wait budget; "
            + ", ".join(parts))
    return ToolResult(False, text, res.data, res.action_id, False, status="held", raw=res.raw)


def _gated_id(res: ToolResult) -> tuple[str, str | None]:
    """(code, action_id) when a foreground submission was refused until another result is retrieved.

    The server answers `RESULT_NOT_RETRIEVED` with the id in the *structured* data
    (`unretrievedActionId`), and `ACTION_BUSY` with `activeActionId`; the human-readable text only
    names the parameter, so a regex over the text finds nothing. Live, a `navigate` whose result was
    never retrieved blocked every later tick and the loop stood still.
    """
    data = res.data if isinstance(res.data, dict) else {}
    code = _find(data, "code") if data else None
    for candidate, key in (("RESULT_NOT_RETRIEVED", "unretrievedActionId"),
                           ("ACTION_BUSY", "activeActionId")):
        if candidate not in (res.text or "") and code != candidate:
            continue
        pending = _find(data, key) if data else None
        if isinstance(pending, str):
            return candidate, pending
        m = re.search(r"([0-9a-fA-F]{8}-[0-9a-fA-F-]{4,})", res.text or "")
        return candidate, (m.group(1) if m else None)
    return "", None


class MCPExecutor:
    """Streamable-HTTP MCP client with the foreground gating state machine."""

    # How a malformed response is answered. A read is side-effect free and retried; a foreground
    # action is not retried, because it may already be running out there.
    _READ_ATTEMPTS = 3
    _RETRY_BACKOFF_S = 1.0
    _MAX_TRANSPORT_FAILURES = 5

    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg
        self._stack: AsyncExitStack | None = None
        self._sess: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._read_timeout = cfg.tool_timeout_ms / 1000
        self._transport_failures = 0

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
        """Ungated read (view_status, read_recent_events, view_blocks, query_bot_data).

        Retried, because a read has no effect on the world: there is nothing to double up and
        nothing to leave half-done, so a blip costs a retry and nothing else. Run59 died on
        `MCPError: Unexpected content type:` - one malformed HTTP response on the way to
        `view_blocks`, and a 600-tick run gone with it. After the last attempt the error is
        re-raised rather than swallowed: a server that cannot be read from is not a server the
        run should keep talking to, and the loop needs perception to decide anything at all.
        """
        payload = {**args, "rationale": clip(rationale), "response_format": "json"}
        for attempt in range(self._READ_ATTEMPTS):
            try:
                res = _extract(await self._raw_call(tool, payload))
            except Exception:
                if attempt == self._READ_ATTEMPTS - 1:
                    self._transport_failures += 1
                    self._check_transport()
                    raise
                await asyncio.sleep(self._RETRY_BACKOFF_S * (attempt + 1))
                continue
            self._transport_failures = 0
            return res
        raise AssertionError("unreachable")

    def _check_transport(self) -> None:
        """Let a run die loudly once the transport has failed outright, never quietly forever.

        A contained write leaves the tick recorded as a refusal, which is what keeps a single
        blip from ending the run. If the server is genuinely gone, every tick would then fail the
        same way and the run would spin to its tick limit with no error to show for it. Five in a
        row is a dead server, not weather.
        """
        if self._transport_failures >= self._MAX_TRANSPORT_FAILURES:
            raise RuntimeError(
                f"[TRANSPORT_DOWN] {self._transport_failures} consecutive transport failures; "
                "the server is not answering")

    async def call(self, tool: str, args: dict, rationale: str) -> ToolResult:
        """Foreground action: serialised, with submission_id and result retrieval.

        A transport failure here is contained rather than retried, and the asymmetry is the whole
        point. The request may well have reached the server: `read` can be repeated because a
        read changes nothing, while a foreground action is already running out there and a second
        submission - which would carry a second `submission_id`, since `_payload` mints a fresh
        one per attempt - could start a second one. So the tick is recorded as a refusal and the
        run carries on; the next tick's perception shows what the world actually did.
        """
        async with self._lock:
            # One budget for the whole call: the drains of earlier actions and our own wait.
            deadline = asyncio.get_running_loop().time() + self.cfg.action_wait_ms / 1000
            res = ToolResult(False, "", None, None, False)
            for _ in range(3):
                try:
                    res = _extract(await self._raw_call(tool, self._payload(tool, args, rationale)))
                except Exception as exc:
                    self._transport_failures += 1
                    self._check_transport()
                    return ToolResult(False, f"[TRANSPORT] {clip(str(exc), 160)}", None, None,
                                      True, status="transport")
                self._transport_failures = 0
                _, pending = _gated_id(res)
                if pending:
                    # The refusal means *our* action never started, so draining the named result
                    # and resubmitting cannot double-run anything.
                    drained = await self._wait_settled(pending, deadline)
                    if not drained.settled:
                        return _held(drained if drained.data else res)
                    continue
                if not res.ok:
                    res = await self._recover(res, deadline)
                break
            if res.action_id and not res.settled:
                waited = await self._wait_settled(res.action_id, deadline)
                if not waited.settled:
                    return _held(waited if waited.data else res)
                res = waited
            return res

    def _payload(self, tool: str, args: dict, rationale: str) -> dict:
        return {
            **args,
            "submission_id": self._sub(tool),
            "rationale": clip(rationale),
            "wait_timeout_ms": self.cfg.initial_wait_ms,
            "response_format": "json",
        }

    async def _recover(self, res: ToolResult, deadline: float | None = None) -> ToolResult:
        """ACTION_BUSY / lost action id: wait on whatever the server points at."""
        m = re.search(r"([0-9a-fA-F-]{8,})", res.text)
        if m and "ACTION_BUSY" in res.text:
            return await self._wait_settled(m.group(1), deadline)
        return res

    async def _wait_settled(self, action_id: str, deadline: float | None = None) -> ToolResult:
        """Poll an action until it settles, `max_polls` runs out, or `deadline` passes."""
        res = ToolResult(False, "", None, action_id, False)
        loop = asyncio.get_running_loop()
        for _ in range(self.cfg.max_polls):
            if deadline is not None and loop.time() >= deadline:
                return res
            res = _extract(await self._raw_call("wait_for_action", {
                "action_id": action_id,
                "timeout_ms": self.cfg.poll_ms,
                "rationale": "retrieve the preceding foreground result",
                "response_format": "json",
            }))
            if res.settled:
                return res
        return res
