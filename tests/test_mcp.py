"""The foreground gating state machine: a refused submission must be drained and retried.

Live, a `navigate` whose result was never retrieved made the server answer every later
submission with RESULT_NOT_RETRIEVED. The action id lives only in the structured data
(`unretrievedActionId`), so the old text-regex recovery never fired and the run stood still.
"""

from __future__ import annotations

import unittest

from mlaya.config import MCPConfig
from mlaya.mcp import MCPExecutor, _extract, _gated_id

PENDING = "58d1d3e1-8a65-49b8-812a-6eb4587c72f9"
MINE = "c0ffee00-1111-2222-3333-444444444444"


class _Content:
    def __init__(self, text: str) -> None:
        self.text = text


class _Res:
    """Stands in for an mcp CallToolResult."""

    def __init__(self, structured=None, text: str = "", is_error: bool = False) -> None:
        self.structured_content = structured
        self.content = [_Content(text)] if text else []
        self.is_error = is_error


def _refused(pending: str) -> _Res:
    return _Res(
        structured={"response": {"format": "json", "data": {
            "state": "refused",
            "code": "RESULT_NOT_RETRIEVED",
            "error": "No new action started. Call wait_for_action with unretrievedActionId to "
                     "retrieve the preceding full result.",
            "unretrievedActionId": pending,
        }}},
        is_error=True,
        text="No new action started. Call wait_for_action with unretrievedActionId to retrieve "
             "the preceding full result.",
    )


def _settled(action_id: str, status: str = "succeeded") -> _Res:
    return _Res(structured={"response": {"format": "json", "data": {
        "state": "settled",
        "actionId": action_id,
        "output": {"action": "drop_item", "result": {"status": status}},
    }}})


class GatedIdTest(unittest.TestCase):
    def test_pending_id_comes_from_structured_data_not_text(self) -> None:
        res = _extract(_refused(PENDING))
        self.assertFalse(res.ok)
        # The text names only the parameter; the id lives in the structured data.
        self.assertNotIn(PENDING, res.text)
        self.assertEqual(_gated_id(res), ("RESULT_NOT_RETRIEVED", PENDING))

    def test_settled_result_is_not_gated(self) -> None:
        self.assertEqual(_gated_id(_extract(_settled(MINE))), ("", None))


class DrainTest(unittest.IsolatedAsyncioTestCase):
    def _exec(self, script: list) -> tuple[MCPExecutor, list]:
        ex = MCPExecutor(MCPConfig())
        calls: list[tuple[str, dict]] = []

        async def fake(name: str, args: dict):
            calls.append((name, args))
            return script.pop(0)

        ex._raw_call = fake  # type: ignore[assignment]
        return ex, calls

    async def test_refused_submission_drains_then_resubmits(self) -> None:
        ex, calls = self._exec([_refused(PENDING), _settled(PENDING), _settled(MINE)])
        res = await ex.call("drop_item", {"items": [{"item_name": "andesite"}]}, "test")
        self.assertTrue(res.ok)
        self.assertEqual(res.action_id, MINE)
        self.assertEqual([n for n, _ in calls], ["drop_item", "wait_for_action", "drop_item"])
        # each submission carries a fresh submission_id
        self.assertNotEqual(calls[0][1]["submission_id"], calls[2][1]["submission_id"])

    async def test_clean_submission_is_not_retried(self) -> None:
        ex, calls = self._exec([_settled(MINE)])
        res = await ex.call("drop_item", {}, "test")
        self.assertTrue(res.ok)
        self.assertEqual([n for n, _ in calls], ["drop_item"])


if __name__ == "__main__":
    unittest.main()
