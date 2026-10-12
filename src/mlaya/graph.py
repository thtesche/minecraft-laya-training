from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from . import perception
from . import survival
from . import verify as vmod
from .actions import candidates, is_hostile_hunt, resolve
from .brain import Brain, escalate_to_system2
from .memory import MemoryLog
from .config import Settings
from .targets import ground_targets

# Safe default when a chosen label cannot be resolved (should be rare; candidates are filtered).
FALLBACK_LABEL = "get_wood"

# Labels whose step needs a destination; the M4 parameter layer picks it from `view_blocks`.
TARGET_LABELS = frozenset({"explore"})

# Tool-result codes that mean "the bot's surroundings block the action". `craft_item` and
# `smelt_item` place a carried workstation beside the bot; in a 1x1 excavation shaft there
# is no replaceable cell, so the tool answers `[WORKSTATION_PLACEMENT_FAILED] ... no_cell`.
# Retrying in place loops forever, so the fix is to change the surroundings and try again.
BLOCKED_CODES = ("WORKSTATION_PLACEMENT_FAILED",)


def blocked_by_surroundings(text: str | None) -> bool:
    """True when a failed tool result says the surroundings prevent the action."""
    return any(code in (text or "") for code in BLOCKED_CODES)


def unblock_action(gs, attempt: int = 0):
    """The step that clears a surroundings block, or None when nothing can.

    First choice is to dig a small pocket where the bot stands (`make_room`): it is cheap
    and keeps the bot next to what it was working on. Only if that pocket did not help
    does the second attempt fall back to walking to open ground (`explore`, M4 target pool).
    """
    if attempt < 1:
        action = resolve("make_room", gs)
        if action is not None:
            return action
    targets = ground_targets(gs)
    return resolve("explore", gs, targets[0]) if targets else None


def _chosen(answers: dict) -> dict:
    """Compact record of Laya's own answers for the log: {qid: choice or score}.

    Kept small on purpose (probabilities are dropped): the M3 dataset needs the `in_danger`
    gate label and the `intent` pick, both of which are here.
    """
    out: dict = {}
    for qid, ans in (answers or {}).items():
        if not isinstance(ans, dict):
            continue
        if "choice" in ans:
            out[qid] = ans["choice"]
        elif "score" in ans:
            out[qid] = ans["score"]
    return out


class AgentState(TypedDict, total=False):
    iteration: int
    before: Any
    decision: Any
    action: Any
    result: Any
    escalated: bool
    deferred: bool
    note: str
    done: bool


class Agent:
    def __init__(self, settings: Settings, executor):
        self.settings = settings
        self.mcp = executor
        self.brain = Brain(settings.laya)
        self.log = MemoryLog(settings.log_path())
        self.history: list[dict] = []
        # The server restores default policy on death; re-apply once per death.
        self._policy_for_death_done = False
        # Label whose last attempt failed because the surroundings blocked it; the next
        # tick changes the surroundings before retrying it (`blocked_by_surroundings`).
        self._blocked_label: str | None = None
        # How many unblock steps have been spent on that label: pocket first, then explore.
        self._unblock_attempts = 0
        self.graph = self._build()

    async def _perceive(self, state: AgentState) -> dict:
        gs = await perception.build(self.mcp)
        return {"iteration": 0, "before": gs, "done": False, "note": "", "escalated": False,
                "deferred": False}

    async def _decide(self, state: AgentState) -> dict:
        gs = state["before"]
        decision = self.brain.decide(gs.narrative(), candidates(gs))
        if not decision.deferred and decision.action_label in TARGET_LABELS:
            # M4 parameter layer: turn "where?" into a Laya choice over grounded candidates.
            targets = ground_targets(gs)
            if targets:
                target, meta = self.brain.choose_target(gs.narrative(), targets)
                decision.target = target
                decision.target_meta = meta
        return {"decision": decision, "escalated": False, "deferred": False}

    async def _guard(self, state: AgentState) -> dict:
        gs = state["before"]
        decision = state["decision"]
        if self._blocked_label == decision.action_label:
            # The same label failed because the surroundings blocked it. Clear a pocket
            # (or, on a second try, move to open ground) first; the label stays in
            # `candidates`, so the next tick retries it there.
            action = unblock_action(gs, self._unblock_attempts)
            if action is not None:
                self._unblock_attempts += 1
                self._blocked_label = None
                self.log.append({"stage": "policy", "reason": "unblock_surroundings",
                                 "blocked": decision.action_label, "via": action.label})
                return {"action": action, "note": f"unblock -> {action.label}"}
        action = resolve(decision.action_label, gs, decision.target)
        if action is not None:
            return {"action": action}
        if decision.deferred:
            # In danger or nothing valid: the server-side survival policy owns this tick.
            return {"action": None, "deferred": True, "note": decision.note}
        plan = escalate_to_system2(
            self.settings.llm, self.settings.api_key(), gs.as_state(), self.history,
            {"triggers": ["unresolved_label"], "label": decision.action_label},
            list(candidates(gs)),
        )
        label = (plan or {}).get("action")
        alt = resolve(label, gs) if label else None
        self.log.append({"stage": "escalation", "label": decision.action_label,
                         "plan": plan, "chosen": label})
        if alt is not None:
            return {"action": alt, "escalated": True, "note": f"escalated -> {label}"}
        return {"action": resolve(FALLBACK_LABEL, gs), "escalated": True,
                "note": f"fallback -> {FALLBACK_LABEL}"}

    async def _execute(self, state: AgentState) -> dict:
        action = state["action"]
        # A hostile hunt runs in the aggressive hunt stage; the base combat floor
        # would otherwise block the engagement ([HUNT_CAPABILITY_BLOCKED]). Restore
        # immediately afterwards: the zero floor would otherwise hold the body and
        # block navigation until the next policy change.
        staged = is_hostile_hunt(action)
        if staged:
            await survival.apply_hunt(self.mcp, f"open the hunt stage for {action.label}")
            self.log.append({"stage": "policy", "reason": "hunt_stage_on",
                             "label": action.label})
        try:
            res = await self.mcp.call(action.tool, action.args, action.rationale)
        finally:
            if staged:
                await survival.restore_base(
                    self.mcp, f"restore the base stage after {action.label}")
        return {"result": res}

    async def _defer(self, state: AgentState) -> dict:
        decision = state["decision"]
        self.log.append({
            "stage": "defer",
            "iteration": state.get("iteration", 0),
            "state": state["before"].as_state(),
            "candidates": decision.candidates,
            "note": decision.note,
        })
        self.history.append({"stage": "defer", "note": decision.note})
        iteration = state.get("iteration", 0) + 1
        gs = state["before"]
        if iteration < self.settings.loop.max_iterations:
            await asyncio.sleep(self.settings.mcp.poll_ms / 1000)
            gs = await perception.build(self.mcp)
        return {"before": gs, "iteration": iteration,
                "done": iteration >= self.settings.loop.max_iterations}

    async def _verify(self, state: AgentState) -> dict:
        before = state["before"]
        action = state["action"]
        res = state["result"]
        # Remember a surroundings block so the next tick changes the surroundings before
        # retrying; any successful action clears it and the attempt count.
        self._blocked_label = (
            action.label
            if not getattr(res, "ok", False)
            and blocked_by_surroundings(getattr(res, "text", ""))
            else None
        )
        if self._blocked_label is None:
            self._unblock_attempts = 0
        after = await perception.build(self.mcp)
        await self._reapply_after_death(after)
        delta = vmod.diff(before, after)
        progressed = vmod.progressed(action.label, delta)
        record = {
            "stage": "action",
            "iteration": state.get("iteration", 0),
            "state": before.as_state(),
            "narrative": before.narrative(),
            "candidates": state["decision"].candidates,
            "label": action.label,
            "tool": action.tool,
            "escalated": state.get("escalated", False),
            "ok": bool(getattr(res, "ok", False)),
            "status": getattr(res, "status", None),
            "text": (getattr(res, "text", "") or "")[:400],
            "delta": delta,
            "progressed": progressed,
            "answers": _chosen(getattr(state["decision"], "answers", {})),
        }
        target = getattr(state["decision"], "target", None)
        if target is not None:
            record["target"] = getattr(target, "label", str(target))
            meta = getattr(state["decision"], "target_meta", None)
            if meta:
                record["target_meta"] = meta
        self.log.append(record)
        self.history.append(record)
        iteration = state.get("iteration", 0) + 1
        return {"before": after, "iteration": iteration,
                "done": iteration >= self.settings.loop.max_iterations}

    async def _reapply_after_death(self, gs) -> None:
        """The server restores default policy on death; re-apply once per death."""
        age = getattr(gs, "last_death_age_s", None)
        if age is None or age > 120:
            self._policy_for_death_done = False
            return
        if age < 90 and not self._policy_for_death_done:
            self._policy_for_death_done = True
            summary = await survival.apply_policy(self.mcp, "re-apply after death")
            self.log.append({"stage": "policy", "reason": "death", "summary": summary})

    def _after_guard(self, state: AgentState) -> str:
        return "execute" if state.get("action") is not None else "defer"

    def _route(self, state: AgentState) -> str:
        return END if state.get("done") else "decide"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("perceive", self._perceive)
        graph.add_node("decide", self._decide)
        graph.add_node("guard", self._guard)
        graph.add_node("execute", self._execute)
        graph.add_node("defer", self._defer)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("perceive")
        graph.add_edge("perceive", "decide")
        graph.add_edge("decide", "guard")
        graph.add_conditional_edges("guard", self._after_guard,
                                    {"execute": "execute", "defer": "defer"})
        graph.add_edge("execute", "verify")
        graph.add_conditional_edges("verify", self._route, {"decide": "decide", END: END})
        graph.add_conditional_edges("defer", self._route, {"decide": "decide", END: END})
        return graph.compile(checkpointer=MemorySaver())

    async def run(self) -> dict:
        config = {
            "configurable": {"thread_id": "mlaya"},
            "recursion_limit": 6 * self.settings.loop.max_iterations + 10,
        }
        return await self.graph.ainvoke({}, config)
