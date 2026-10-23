"""The foreground gating state machine: a refused submission must be drained and retried.

Live, a `navigate` whose result was never retrieved made the server answer every later
submission with RESULT_NOT_RETRIEVED. The action id lives only in the structured data
(`unretrievedActionId`), so the old text-regex recovery never fired and the run stood still.
"""

from __future__ import annotations

import asyncio
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


def _held_progress(action_id: str) -> _Res:
    """A `wait_for_action` answer for an action the hostile reflex has suspended (run89)."""
    return _Res(structured={"response": {"format": "json", "data": {
        "state": "pending",
        "wakeReason": "timeout",
        "actionId": action_id,
        "survival": {"owner": {"current": "hostile_reflex"},
                     "response": {"capability": "combat", "kind": "fight", "phase": "hold",
                                  "phaseTicks": 55712}},
    }}})


class WaitBudgetTest(unittest.IsolatedAsyncioTestCase):
    """One foreground call waits at most `action_wait_ms`, drains included.

    run89: the hostile reflex held the body in a fight it could not finish, and each tick sat in
    `wait_for_action` for the full `max_polls` - 31 minutes, and 93 when a refused submission
    drained the held action three times over.
    """

    def _exec(self, answer) -> tuple[MCPExecutor, list]:
        cfg = MCPConfig(action_wait_ms=0)
        ex = MCPExecutor(cfg)
        calls: list[str] = []

        async def fake(name: str, args: dict):
            calls.append(name)
            return answer(name)

        ex._raw_call = fake  # type: ignore[assignment]
        return ex, calls

    async def test_a_held_action_ends_the_call_with_the_holder_named(self) -> None:
        ex, calls = self._exec(lambda name: _held_progress(MINE))
        res = await ex.call("collect_mob_drop", {"mob_name": "sheep"}, "test")
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "held")
        self.assertIn("[BODY_HELD]", res.text)
        self.assertIn("hostile_reflex", res.text)
        self.assertIn("fight/hold", res.text)
        self.assertIn("waited 0 s", res.text)
        self.assertEqual(calls, ["collect_mob_drop"])

    async def test_a_drain_that_does_not_settle_is_not_resubmitted(self) -> None:
        ex, calls = self._exec(lambda name: _refused(PENDING))
        res = await ex.call("collect_mob_drop", {"mob_name": "sheep"}, "test")
        self.assertEqual(res.status, "held")
        self.assertEqual(calls, ["collect_mob_drop"])

    async def test_the_budget_bounds_the_polls(self) -> None:
        cfg = MCPConfig(action_wait_ms=50, poll_ms=10)
        ex = MCPExecutor(cfg)
        polls = 0

        async def fake(name: str, args: dict):
            nonlocal polls
            if name == "wait_for_action":
                polls += 1
                await asyncio.sleep(0.02)
            return _held_progress(MINE)

        ex._raw_call = fake  # type: ignore[assignment]
        res = await ex.call("collect_mob_drop", {}, "test")
        self.assertEqual(res.status, "held")
        self.assertLess(polls, 10)


def _pending(action_id: str) -> _Res:
    """A `wait_for_action` answer for an action still running, its progress naming a status."""
    return _Res(structured={"response": {"format": "json", "data": {
        "state": "pending",
        "wakeReason": "timeout",
        "actionId": action_id,
        "progress": {"actionId": action_id, "action": "collect_mob_drop",
                     "progress": {"status": "succeeded"}},
    }}})


class SettledStateTest(unittest.IsolatedAsyncioTestCase):
    """The envelope's `state` says whether an action settled, not a nested `status`.

    run90 it76: a hunt that collected two of six chicken settled with result status `partial`,
    which the depth-first `status` search read as still running. Every `wait_for_action` answered
    at once with the same settlement, 900 polls ran out in two minutes, and the tick ended
    `[BODY_HELD] ... owner unknown` while nothing held the body.
    """

    def test_a_partial_settlement_is_settled(self) -> None:
        res = _extract(_settled(MINE, status="partial"))
        self.assertTrue(res.settled)
        self.assertEqual(res.status, "partial")

    def test_a_pending_answer_is_running_whatever_its_progress_says(self) -> None:
        self.assertFalse(_extract(_pending(MINE)).settled)

    def test_an_acceptance_is_running(self) -> None:
        res = _extract(_Res(structured={"response": {"format": "json", "data": {
            "state": "accepted", "actionId": MINE, "action": "collect_mob_drop",
            "admittedAt": "2026-10-23T21:00:00Z"}}}))
        self.assertFalse(res.settled)

    def test_a_markdown_answer_still_reads_its_status_line(self) -> None:
        res = _extract(_Res(text=f"Action ID: {MINE}\nStatus: partial"))
        self.assertTrue(res.settled)

    async def test_a_partial_hunt_is_retrieved_once(self) -> None:
        script = [_pending(MINE), _settled(MINE, status="partial")]
        ex = MCPExecutor(MCPConfig())
        calls: list[str] = []

        async def fake(name: str, args: dict):
            calls.append(name)
            return script.pop(0)

        ex._raw_call = fake  # type: ignore[assignment]
        res = await ex.call("collect_mob_drop", {"mob_name": "chicken"}, "test")
        self.assertNotEqual(res.status, "held")
        self.assertEqual(res.status, "partial")
        self.assertEqual(calls, ["collect_mob_drop", "wait_for_action"])


class TransportTest(unittest.IsolatedAsyncioTestCase):
    """A malformed HTTP response is answered, not obeyed.

    Live, run59: `MCPError: Unexpected content type:` on the way to `view_blocks` ended a
    600-tick run outright. `_raw_call` had no exception handling at all, so one unreadable
    response took the process down.

    The asymmetry is the whole design. A read has no effect on the world, so it is retried - there
    is nothing to double up. A foreground action may already be running out there, and `_payload`
    mints a fresh `submission_id` per attempt, so a resubmission could start a second one. That
    one is contained instead: the tick is recorded as a refusal and the run goes on.
    """

    def _exec(self, script: list, failures: int = 0) -> tuple[MCPExecutor, list]:
        ex = MCPExecutor(MCPConfig())
        ex._RETRY_BACKOFF_S = 0.0          # no waiting in tests
        calls: list[tuple[str, dict]] = []

        left = [failures]

        async def fake(name: str, args: dict):
            calls.append((name, args))
            if left[0] > 0:
                left[0] -= 1
                raise RuntimeError("Unexpected content type:")
            return script.pop(0)

        ex._raw_call = fake  # type: ignore[assignment]
        return ex, calls

    async def test_a_read_survives_one_malformed_response(self) -> None:
        ex, calls = self._exec([_settled(MINE)], failures=1)
        res = await ex.read("view_status", {}, "test")
        self.assertTrue(res.ok)
        self.assertEqual(len(calls), 2)      # asked again, got an answer

    async def test_a_read_that_never_answers_still_raises(self) -> None:
        # Contained forever would be worse: perception is what every decision is made from, and a
        # run talking to a server that cannot answer it is a run spinning to its tick limit.
        ex, calls = self._exec([], failures=99)
        with self.assertRaises(RuntimeError):
            await ex.read("view_status", {}, "test")
        self.assertEqual(len(calls), MCPExecutor._READ_ATTEMPTS)

    async def test_a_write_that_loses_its_answer_is_never_submitted_twice(self) -> None:
        ex, calls = self._exec([], failures=99)
        res = await ex.call("drop_item", {}, "test")
        self.assertFalse(res.ok)
        self.assertTrue(res.text.startswith("[TRANSPORT]"))
        # One call. A second submission would carry a second submission_id and could start a
        # second drop for a request the server may already have accepted.
        self.assertEqual([n for n, _ in calls], ["drop_item"])

    async def test_five_blind_writes_end_the_run_loudly(self) -> None:
        # The containment must not become a spin: a dead server fails every tick the same way,
        # and a run that reaches its tick limit with nothing to show for it hides the cause.
        ex, calls = self._exec([], failures=99)
        for _ in range(MCPExecutor._MAX_TRANSPORT_FAILURES - 1):
            self.assertFalse((await ex.call("drop_item", {}, "test")).ok)
        with self.assertRaises(RuntimeError) as caught:
            await ex.call("drop_item", {}, "test")
        self.assertIn("[TRANSPORT_DOWN]", str(caught.exception))
        self.assertEqual(len(calls), MCPExecutor._MAX_TRANSPORT_FAILURES)


if __name__ == "__main__":
    unittest.main()
