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

# Safe default when a chosen label cannot be resolved (should be rare; candidates are filtered).
FALLBACK_LABEL = "get_wood"


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
        self.graph = self._build()

    async def _perceive(self, state: AgentState) -> dict:
        gs = await perception.build(self.mcp)
        return {"iteration": 0, "before": gs, "done": False, "note": "", "escalated": False,
                "deferred": False}

    async def _decide(self, state: AgentState) -> dict:
        gs = state["before"]
        decision = self.brain.decide(gs.narrative(), candidates(gs))
        return {"decision": decision, "escalated": False, "deferred": False}

    async def _guard(self, state: AgentState) -> dict:
        gs = state["before"]
        decision = state["decision"]
        action = resolve(decision.action_label, gs)
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
        }
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
