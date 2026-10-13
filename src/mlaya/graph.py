from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from . import perception
from . import survival
from . import verify as vmod
from .actions import candidates, inventory_relief, is_hostile_hunt, resolve, tech_tree_complete
from .brain import Brain, escalate_to_system2
from .memory import MemoryLog
from .config import Settings
from .targets import ground_targets

# Safe default when a chosen label cannot be resolved (should be rare; candidates are filtered).
FALLBACK_LABEL = "get_wood"

# Labels whose step needs a destination; the M4 parameter layer picks it from `view_blocks`.
TARGET_LABELS = frozenset({"explore"})

# A placed bed within this range of the bot means the bot is standing at a home base a previous
# run (or a manual build) left behind: that bed's ground cell is re-adopted as `base`, so the run
# continues there instead of raising a second base somewhere else.
ADOPT_BED_RADIUS = 12.0
# How far a chest may sit from the re-adopted base and still belong to it (one block of slack
# over the base radius, plus the vertical step a chest is placed at).
ADOPT_CHEST_RADIUS = 4

# Tool-result codes that mean "the bot's surroundings block the action". `craft_item` and
# `smelt_item` place a carried workstation beside the bot; in a 1x1 excavation shaft there
# is no replaceable cell, so the tool answers `[WORKSTATION_PLACEMENT_FAILED] ... no_cell`.
# `sleep` places the carried bed the same way and needs two supported cells, so the same
# shaft answers `[BED_PLACEMENT_FAILED] ... no ... flat ground` (live, armour run: three
# wasted ticks). Retrying in place loops forever, so the fix is to change the surroundings
# and try again.
BLOCKED_CODES = ("WORKSTATION_PLACEMENT_FAILED", "BED_PLACEMENT_FAILED", "PLACE_BLOCK_FAILED")
# A gathering action is refused with `[INVENTORY_FULL]` once all 36 carried slots are taken; the
# unblock there is not terrain but pack space (stow at the base, else drop), so it routes through
# a separate relief step.
INVENTORY_CODES = ("INVENTORY_FULL",)


def blocked_by_surroundings(text: str | None) -> bool:
    """True when a failed tool result says the surroundings prevent the action."""
    return any(code in (text or "") for code in BLOCKED_CODES)


def blocked_by_inventory(text: str | None) -> bool:
    """True when a failed tool result says the pack is full."""
    return any(code in (text or "") for code in INVENTORY_CODES)


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
    stop: str  # why the loop ended: "goal" (tech tree complete) or "limit"


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
        # What kind of block that was: "surroundings" (dig a pocket / walk to open ground) or
        # "inventory" (make pack room). It chooses the unblock step and when the label clears.
        self._blocked_kind = "surroundings"
        # A deposit into the base chest failed (the chest is full): the inventory relief then
        # drops instead of retrying a deposit that cannot free anything.
        self._stow_blocked = False
        # How many unblock steps have been spent on that label: pocket first, then explore.
        self._unblock_attempts = 0
        # Home base: the surface anchor the loop chooses once (see `establish_base`) and the
        # placed chest there. Both are per-run only and injected into each GameState.
        self.base: tuple[int, int, int] | None = None
        self.chest_pos: tuple[int, int, int] | None = None
        # Lava cells a pour already failed on with `[BUCKET_NO_FACE]`. That answer is
        # structural -- no solid block beside that cell faces the bot -- so retrying it can
        # only fail again; remembering it lets the next tick pour elsewhere (live: the
        # surface pond holds one usable cell and the obsidian would never be made there).
        self._pour_skip: set[tuple[int, int, int]] = set()
        # A `drop_junk` whose disposal pit the terrain refused (`[DROP_HOLE_UNAVAILABLE]`): from
        # then on the relief drops the surplus on the open ground instead. That still frees the
        # slots it is for, and unlike the pit dig it cannot fail, so the loop cannot stall on it.
        self._drop_hole_blocked = False
        self.graph = self._build()

    def _inject(self, gs):
        """Carry the loop-owned facts perception cannot see into the GameState."""
        self._adopt_homebase(gs)
        gs.base = self.base
        gs.chest_pos = self.chest_pos
        gs.pour_skip = frozenset(self._pour_skip)
        gs.drop_hole_blocked = self._drop_hole_blocked
        return gs

    def _adopt_homebase(self, gs) -> None:
        """Re-adopt a home base and chest a previous run left standing in the world.

        `base` and `chest_pos` are per-run state, so a restart loses them - but the bed and the
        chest are real blocks. When the bot stands at a placed bed, that bed's ground cell becomes
        the base again, and a chest a few blocks away is remembered with it, so the home-base
        steps (sleep, stow) keep working at the old base instead of building a second one.
        """
        bed = gs.resource_nearest("white_bed")
        if self.base is None and bed is not None and bed.distance <= ADOPT_BED_RADIUS:
            self.base = (bed.x, bed.y - 1, bed.z)
        if self.chest_pos is None and self.base is not None:
            bx, by, bz = self.base
            for hit in gs.resources.get("chest", []):
                if (abs(hit.x - bx) <= ADOPT_CHEST_RADIUS
                        and abs(hit.z - bz) <= ADOPT_CHEST_RADIUS
                        and abs(hit.y - (by + 1)) <= ADOPT_CHEST_RADIUS):
                    self.chest_pos = hit.pos
                    break

    async def _perceive(self, state: AgentState) -> dict:
        gs = self._inject(await perception.build(self.mcp))
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
            if self._blocked_kind == "inventory":
                # The pack was full. Make room (deposit at the base chest, else drop the
                # surplus) and let the blocked label run again next tick: unlike a terrain
                # fix, this step removes the cause itself, so the label must not stay blocked.
                action = inventory_relief(gs, allow_stow=not self._stow_blocked)
                if action is not None:
                    self._blocked_label = None
                    self._unblock_attempts = 0
                    self.log.append({"stage": "policy", "reason": "unblock_inventory",
                                     "blocked": decision.action_label, "via": action.label})
                    return {"action": action, "note": f"unblock -> {action.label}"}
            else:
                # The same label failed because the surroundings blocked it. Clear a pocket
                # (or, on a second try, move to open ground) first; the label stays in
                # `candidates`, so the next tick retries it there.
                action = unblock_action(gs, self._unblock_attempts)
                if action is not None:
                    self._unblock_attempts += 1
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
            "narrative": state["before"].narrative(),
            "candidates": decision.candidates,
            "note": decision.note,
            "answers": _chosen(getattr(decision, "answers", {})),
        })
        self.history.append({"stage": "defer", "note": decision.note})
        iteration = state.get("iteration", 0) + 1
        gs = state["before"]
        stop = self._stop_state(gs, iteration)
        if not stop["done"]:
            await asyncio.sleep(self.settings.mcp.poll_ms / 1000)
            gs = self._inject(await perception.build(self.mcp))
        return {"before": gs, "iteration": iteration, **stop}

    def _goal_reached(self, gs) -> bool:
        """Whether the run's goal is reached (see `actions.tech_tree_complete`)."""
        return bool(self.settings.loop.stop_when_complete) and tech_tree_complete(gs)

    def _stop_state(self, gs, iteration: int) -> dict:
        """`done`/`stop` for the next state: the goal ends the run early, else the limit does."""
        goal = self._goal_reached(gs)
        limit = iteration >= self.settings.loop.max_iterations
        return {"done": goal or limit,
                "stop": "goal" if goal else ("limit" if limit else "")}

    def _note_block(self, label: str, ok: bool, text: str | None) -> None:
        """Update the surroundings-block bookkeeping from one action's result.

        The blocked label is cleared only once *it* finally succeeds. A successful unblock
        (`make_room`) must not clear it: otherwise the stage-2 `explore` escape is unreachable
        and `make_room` retries forever (live: `sleep` in a 1x1 shaft alternated
        sleep/make_room for the rest of the run, and the shield run burned 12 of 16 ticks on it).
        """
        if label == self._blocked_label:
            if ok:
                self._blocked_label = None
                self._unblock_attempts = 0
        elif not ok and blocked_by_inventory(text):
            # A full pack is not terrain: the unblock makes room (stow at the base, else drop),
            # and that frees the label to run again on the next tick.
            self._blocked_label = label
            self._blocked_kind = "inventory"
            self._unblock_attempts = 0
        elif not ok and blocked_by_surroundings(text):
            self._blocked_label = label
            self._blocked_kind = "surroundings"
            self._unblock_attempts = 0

    def _note_pour(self, action, ok: bool, text: str | None) -> None:
        """Remember a lava cell a pour was refused at, so it is never poured at again.

        `[BUCKET_NO_FACE]` says no solid block beside that cell faces the bot: a property of
        the cell and the terrain around it, not of the attempt. The obsidian step reads the set
        back out of the GameState (`pour_skip`) and looks for the next lava source, which is
        what carries the run from a stopped-up surface pond to the lava lake further off.
        """
        args = action.args or {}
        if ok or action.tool != "use_bucket" or args.get("action") != "pour":
            return
        if "BUCKET_NO_FACE" not in (text or ""):
            return
        pos = (args.get("x"), args.get("y"), args.get("z"))
        if all(isinstance(v, int) for v in pos):
            self._pour_skip.add(pos)

    def _note_drop(self, action, ok: bool, text: str | None) -> None:
        """Remember that no disposal pit can be dug where the bot stands, so the next drop does
        not ask for one.

        `[DROP_HOLE_UNAVAILABLE]` says no adjacent terrain can be excavated into a dry, enclosed
        pit. That is a property of the ground, not of the attempt (live, obsidian run: a 1-wide
        tunnel answered it on every tick while `drop_junk` retried in place). Dropping on the open
        ground instead still frees the slot, which is all the relief is for.
        """
        if ok or action.label != "drop_junk":
            return
        if "DROP_HOLE_UNAVAILABLE" in (text or ""):
            self._drop_hole_blocked = True

    async def _verify(self, state: AgentState) -> dict:
        before = state["before"]
        action = state["action"]
        res = state["result"]
        # Remember a surroundings block so the next tick changes the surroundings before
        # retrying it (see `_note_block` for the clearing rule).
        _ok = getattr(res, "ok", False)
        _text = getattr(res, "text", "")
        self._note_block(action.label, _ok, _text)
        self._note_pour(action, _ok, _text)
        self._note_drop(action, _ok, _text)
        if action.label == "stow_surplus":
            # A failed deposit means the chest is full; stop offering it as the relief.
            self._stow_blocked = not _ok
        if _ok:
            # `establish_base` reports the surface cell through its navigate target; the base
            # anchor is the ground block under the feet it walked to. `place_chest` names the
            # cell it placed into, remembered so `use_container` can reach it later.
            if action.label == "establish_base" and "x" in action.args:
                self.base = (int(action.args["x"]), int(action.args["y"]) - 1,
                             int(action.args["z"]))
            elif action.label == "place_chest" and "x" in action.args:
                self.chest_pos = (int(action.args["x"]), int(action.args["y"]),
                                  int(action.args["z"]))
        after = self._inject(await perception.build(self.mcp))
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
                **self._stop_state(after, iteration)}

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
