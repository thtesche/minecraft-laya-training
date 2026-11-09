from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .actions import (CRITERIA, DEFER_QUIET_TICKS, RECOVER_HEALTH, _must_eat,
                      _recovery_owed)
from .config import LLMConfig, LayaConfig
from .perception import INVENTORY_SLOTS, phase_brief, phase_persona

# Two decisions per tick, one forward pass:
#   in_danger  - binary gate; a reliable zero-shot signal. Yes -> defer to the survival policy.
#   intent     - choose among the actions whose preconditions already hold (built at runtime).
# A `danger` score is added purely for logging / later head training.
DANGER_QID = "in_danger"
INTENT_QID = "intent"
TARGET_QID = "target"
DANGER_Q = {
    "type": "choice",
    "instructions": "Is the bot in immediate danger right now?",
    "criteria": {
        "no": "no: no mob is attacking and health is comfortably high - a hostile that is not "
              "within a few blocks and closing in does not count",
        "yes": "yes: a mob is within a few blocks and attacking now, or health is very low",
    },
}
DANGER_SCORE = {
    "type": "score",
    "instructions": "How dangerous is the situation?",
    "criteria": ["none", "low", "high", "critical"],
}

# The zero-shot gate answers from the narrative, which can only say a hostile is "within 16
# blocks" - it cannot tell a mob across the room from one in contact, so it reads a merely
# loaded mob as danger. Live (run7b) a zombie 9-15 m away in a cave kept it answering `yes`,
# and `yes` defers the tick unconditionally: the server policy had nothing to do (health and
# hunger full, no contact) while the bot stood still at diamond depth tick after tick. The band
# below is the runtime guard: a `yes` only yields the tick when the body is genuinely at risk,
# so the loop keeps working otherwise. The server reflexes still guard contact independently of
# this gate, so the loop only has to stand aside in *contact* range, or when health is low.
#
# run13 widened this to 8 m and paid for it: a skeleton/creeper hovering 5-8 m - close enough to
# shoot, never closing to a hit - deferred ~60 ticks while the policy idled and the loop froze in
# a cave. So the defer radius is *contact* range, matching the dataset's own danger floor
# (`DANGER_CLOSE_M`); only the labelling ceiling stays wider (see `dataset.DANGER_SAFE_M`).
DANGER_DEFER_M = 4.0
DANGER_DEFER_HEALTH = RECOVER_HEALTH
DANGER_BAND_NOTE = "danger band: no mob in range -> keep working"
# Contact range is still no proof of a fight. run86 stood at a fortress blaze spawner with
# blazes at 2-4 m that no longer attacked: the server's reflex fell silent at 20:25:33 (health
# and hunger full, `busy` false) and the loop deferred 120+ ticks without moving until the run
# cap. After this many defers in a row that took no damage and saw no reflex act, the band
# lets the loop work again; the first hit hands the tick back. Under the health band the quiet
# defers release the tick only when nothing heals in place (`actions._recovery_owed`), and the
# loop then offers food or a retreat alone. `DEFER_QUIET_TICKS` lives in `actions` beside it.
#
# run95 paid for two more holes in the band, both of them a defer that could never end:
#   - the bot suffocated in a wall at 9 health with the nearest hostile 23 m off. The policy has
#     no answer to a body inside a block, so the only step that helps is the loop's own `dig_out`
#     (`gs.boxed_in`), and the band held it for every one of the ten ticks to the death.
#   - at 1 health, hunger 6, two raw chicken in the pack, the offer was `['eat_food']` for the
#     last 138 ticks: an edible in the pack makes `_cannot_heal` false, so the quiet defers never
#     released it, while the server's hunger reflex left the chicken alone (`FOOD_MOBS_LAST`).
#     It died with the meal still carried.
# A meal the body needs now (`actions._must_eat`) and a pocket the pathfinder cannot leave are
# both things the loop can answer and the policy cannot, so each releases the band - the meal
# only with no hostile in contact range, because eating with a mob at 2 m is how the next hit
# finds the bot. `_recovery_owed` stays as it was: it is the case with nothing to eat at all.


def danger_matters(gs: Any) -> bool:
    """Whether a gate `yes` should really hand the tick to the survival policy.

    True when health has dropped below the band or a hostile is inside the defer radius. A mob
    further out, or none parsed at all, with health in hand does not hold the loop back. A
    `None` state (a bare `decide` call, or a caller with no perception) keeps the old,
    unconditional defer.

    Under the band the defer holds while the policy can still heal the body - except when the
    loop owns the only step that helps: `_recovery_owed` (quiet defers, no food carried and
    hunger too low to regenerate - run88's 536 ticks at 8.5 health), a meal the body needs now
    with nothing in contact range (`_must_eat`), or a body shut in a pocket (`gs.boxed_in`).
    `_defer_quiet` is no help in any of the three: a health loss resets it, and a suffocating or
    starving body loses health every tick without the policy doing anything.
    """
    if gs is None:
        return True
    nearest = gs.nearest_hostile
    contact = nearest is not None and nearest <= DANGER_DEFER_M
    if gs.health is not None and gs.health < DANGER_DEFER_HEALTH:
        if _recovery_owed(gs) or getattr(gs, "boxed_in", False):
            return False
        return not (_must_eat(gs) and not contact)
    if getattr(gs, "defer_quiet", 0) >= DEFER_QUIET_TICKS:
        return False
    return contact


INTENT_INSTRUCTIONS = "The bot is safe enough to work. What should it do next?"


def intent_question(candidates: list[str]) -> dict:
    """The `intent` choice question for a set of eligible labels.

    Shared by inference (`Brain._questions`) and the M3 dataset builder, so a fine-tuned
    head is trained on exactly the rendering it meets at inference.
    """
    return {
        "type": "choice",
        "instructions": INTENT_INSTRUCTIONS,
        "criteria": {label: CRITERIA[label] for label in candidates},
    }


TARGET_INSTRUCTIONS = ("Pick the destination to walk to. Prefer open ground that is easy "
                       "to stand on and leads toward new terrain.")
# The brain shortlists targets down to this many before Laya answers (head budget 192
# tokens across options; `predict_shortlist` keeps a handful legible).
TARGET_K = 6


def target_question(targets) -> dict:
    """The M4 `target` choice question over `targets.Target` candidates."""
    return {
        "type": "choice",
        "instructions": TARGET_INSTRUCTIONS,
        "criteria": {t.label: t.text for t in targets},
    }


@dataclass
class Decision:
    action_label: str | None
    deferred: bool = False
    candidates: list[str] = field(default_factory=list)
    answers: dict = field(default_factory=dict)
    probabilities: dict = field(default_factory=dict)
    note: str = ""
    # M4 parameter layer: the chosen destination for a label whose step needs one.
    target: Any = None
    target_meta: dict = field(default_factory=dict)


class Brain:
    """System 1: Laya in-process, one forward pass per tick."""

    def __init__(self, cfg: LayaConfig):
        from laya import Router

        kwargs: dict[str, Any] = {"preload": cfg.preload, "default": cfg.model or "multilingual"}
        if cfg.device:
            kwargs["device"] = cfg.device
        if cfg.extra_models:
            # Register (or re-point) checkpoints, e.g. a fine-tune served under `english`.
            kwargs["models"] = dict(cfg.extra_models)
        self._cfg = cfg
        self.router = Router(**kwargs)
        self._embed_fn = None

    def _questions(self, candidates: list[str]) -> dict[str, dict]:
        questions: dict[str, dict] = {DANGER_QID: DANGER_Q, "danger": DANGER_SCORE}
        if len(candidates) > 1:
            questions[INTENT_QID] = intent_question(candidates)
        return questions

    def decide(self, state: str, candidates: list[str], gs: Any = None) -> Decision:
        """One forward pass: the danger gate, then the intent over the eligible labels.

        `gs` (the perceived state) bounds the gate: a `yes` from a merely distant mob is
        rewritten to `no` so the loop keeps working (see `danger_matters`). Callers without a
        state, and tests, keep the unconditional defer by leaving it `None`.
        """
        questions = self._questions(candidates)
        kwargs: dict[str, Any] = {}
        if self._cfg.model:
            kwargs["model"] = self._cfg.model
        if self._cfg.max_len:
            kwargs["max_len"] = self._cfg.max_len
        if self._cfg.min_confidence:
            kwargs["min_confidence"] = self._cfg.min_confidence

        res = self.router.predict(state, questions, **kwargs)
        answers = res.get("answers", {}) if isinstance(res, dict) else {}
        probabilities = {
            qid: ans.get("probabilities")
            for qid, ans in answers.items()
            if isinstance(ans, dict) and ans.get("probabilities")
        }

        in_danger = answers.get(DANGER_QID, {}).get("choice") == "yes"
        band = in_danger and not danger_matters(gs)
        if band:
            # The gate over-read a distant mob; keep working instead of standing aside.
            in_danger = False
        if in_danger and "enter_portal" not in candidates:
            # Danger defers to the survival policy, except when entering the portal is ready:
            # the crossing is the escape route from the Overworld and must not be blocked.
            return Decision(None, deferred=True, candidates=candidates,
                            answers=answers, probabilities=probabilities,
                            note="in danger -> survival policy")
        note = DANGER_BAND_NOTE if band else ""
        if not candidates:
            return Decision(None, deferred=True, answers=answers,
                            probabilities=probabilities, note="no valid action")
        if len(candidates) == 1:
            return Decision(candidates[0], candidates=candidates, note=note,
                            answers=answers, probabilities=probabilities)
        label = answers.get(INTENT_QID, {}).get("choice")
        if label not in candidates:
            label = candidates[0]
        return Decision(label, candidates=candidates, note=note,
                        answers=answers, probabilities=probabilities)

    def _target_embed_fn(self):
        """Mean-pooled encoder embedder for the shortlist, built once and cached."""
        if self._embed_fn is None:
            from laya.shortlist import cached_embed_fn, embed_fn_from_agent

            agent = self.router.load(self._cfg.model or "multilingual")
            self._embed_fn = cached_embed_fn(embed_fn_from_agent(agent))
        return self._embed_fn

    def choose_target(self, state: str, targets) -> tuple[Any, dict]:
        """Pick one destination from the candidates (M4 parameter layer).

        One candidate needs no model call. Up to `TARGET_K` go through a single normal
        pass; a larger pool is narrowed with `predict_shortlist`, which keeps the option
        texts legible inside the shared head budget. Returns `(target, shortlist_meta)`.
        """
        if not targets:
            return None, {}
        if len(targets) == 1:
            return targets[0], {}

        by_label = {t.label: t for t in targets}
        question = {TARGET_QID: target_question(targets)}
        kwargs: dict[str, Any] = {}
        if self._cfg.model:
            kwargs["model"] = self._cfg.model
        if self._cfg.max_len:
            kwargs["max_len"] = self._cfg.max_len

        meta: dict = {}
        if len(targets) <= TARGET_K:
            res = self.router.predict(state, question, **kwargs)
        else:
            from laya.shortlist import predict_shortlist

            res = predict_shortlist(self.router, state, question, self._target_embed_fn(),
                                    k=TARGET_K, **kwargs)
            meta = (res.get("shortlist") or {}).get(TARGET_QID, {}) if isinstance(res, dict) else {}

        answers = res.get("answers", {}) if isinstance(res, dict) else {}
        choice = answers.get(TARGET_QID, {}).get("choice")
        target = by_label.get(choice)
        if target is None:
            # Fall back to the farthest candidate (ground_targets sorts farthest first).
            target = targets[0]
            meta = {**meta, "fallback": True}
        return target, meta


# How many of the run's own action ticks the trajectory digest looks back over. Enough to show
# a loop forming (a label refused a dozen times) without carrying the whole run into the prompt.
DIGEST_WINDOW = 24


def trajectory_digest(history: list[dict], window: int = DIGEST_WINDOW) -> dict:
    """A compact read of where the run has been, built from the decision log System 1 already keeps.

    System 2 used to see only the last five raw records, which is too little to tell a one-off
    refusal from a structural loop: the same label refused twenty times in a row looked the same
    as a label tried once. The digest is the content of `runs/decisions.jsonl` folded down to what
    a next-move decision needs - the recent labels and how each one fared, which labels keep
    failing, and the base/chest ledger - so the model is told the shape of the impasse, not asked
    to re-derive it from a handful of ticks.
    """
    actions = [r for r in history if r.get("stage") == "action"][-window:]
    recent = [
        {
            "label": r.get("label"),
            "ok": bool(r.get("ok")),
            "progressed": bool(r.get("progressed")),
            "status": r.get("status"),
        }
        for r in actions
    ]
    attempts: dict[str, int] = {}
    failures: dict[str, int] = {}
    for r in actions:
        label = str(r.get("label"))
        attempts[label] = attempts.get(label, 0) + 1
        if not r.get("progressed") and r.get("status") != "succeeded":
            failures[label] = failures.get(label, 0) + 1
    # Labels that keep failing without progress are the structural ones - the thing a richer brief
    # is for. Listed worst-first so the model reads the stuck label before the noise.
    repeated_failures = [
        {"label": label, "failed": count, "attempts": attempts.get(label, count)}
        for label, count in sorted(failures.items(), key=lambda kv: -kv[1])
        if count >= 2
    ]
    last_state = next((r.get("state") for r in reversed(actions)
                       if isinstance(r.get("state"), dict)), None) or {}
    ledger = {
        "phase": last_state.get("phase"),
        "dimension": last_state.get("dimension"),
        "base": last_state.get("base"),
        "chest_pos": last_state.get("chest_pos"),
        "furnace_pos": last_state.get("furnace_pos"),
        "banked": last_state.get("banked"),
    }
    return {
        "window": len(actions),
        "recent": recent,
        "repeated_failures": repeated_failures,
        "ledger": ledger,
    }


def _plan_steps(plan: dict | None, allowed_labels: list[str]) -> list[str]:
    """The ordered labels a System-2 reply proposes, from `plan` (a list) or the legacy `action`.

    Filters to only allowed labels; a label outside the set is skipped, not run.
    """
    if not isinstance(plan, dict):
        return []
    steps = plan.get("plan")
    if isinstance(steps, list):
        ordered = [str(s) for s in steps if isinstance(s, str) and str(s) in allowed_labels]
    else:
        action = plan.get("action")
        ordered = [str(action)] if isinstance(action, str) and str(action) in allowed_labels else []
    return ordered


# Fields the escalation state carries for the log and System 1 but that only add noise to a System-2
# brief: the viewer/other player roll call never bears on the next move, and the time-of-day `phase`
# the server fills is null outside the Overworld, where it reads as "phase unknown" over a run phase
# that is in fact well defined. Both are dropped from the System-2 view and the run phase is set
# from `phase_brief` instead.
_SYSTEM2_DROP_FIELDS = ("players",)


def _system2_state_view(state: dict) -> dict:
    """The state as System 2 should read it: the logged view, trimmed and annotated.

    `as_state()` is the log's own vocabulary and must stay whole (`from_state` rebuilds a tick from
    it), so the tailoring happens on a copy. The run phase replaces the empty time-of-day box, the
    home base is marked as the Overworld landmark it always is (System 2 mid-Nether must not read
    the base coordinates as somewhere nearby), and the fields that only add noise are dropped.
    """
    view = {k: v for k, v in state.items() if k not in _SYSTEM2_DROP_FIELDS}
    view["phase"] = phase_brief(state)[0]
    base = state.get("base")
    if base is not None:
        view["base"] = {"pos": list(base), "dimension": "overworld"}
    return view


def _failure_diagnostics(state: dict) -> dict:
    """Why a reaction cannot make progress, computed so System 2 does not re-derive it from raw state.

    The stuck reactions are the ones that need a free slot or a drop in reach, so the brief names the
    pack's fullness and the nearest loaded drop outright: a pinned pack says "shed first", a drop
    tens of blocks out says "walk to it or let it lie", and neither should cost System 2 a guess.
    """
    slots_used = state.get("slots_used")
    diag: dict[str, Any] = {}
    if isinstance(slots_used, int):
        diag["pack"] = {"slots_used": slots_used, "slots_max": INVENTORY_SLOTS,
                        "full": slots_used >= INVENTORY_SLOTS}
    dists = [drop[2] for drop in (state.get("dropped_items") or [])
             if len(drop) > 2 and drop[2] is not None]
    if dists:
        diag["nearest_drop_m"] = round(min(dists), 1)
    return diag


def escalate_to_system2(
    cfg: LLMConfig,
    api_key: str | None,
    state: dict,
    history: list[dict],
    failure: dict,
    allowed_labels: list[str],
    trajectory: dict | None = None,
) -> dict | None:
    """System 2: ask a generative model for a way out when System 1 cannot act (rare).

    The reply is normalised so every caller reads the same shape: `plan` is the ordered list of
    labels the model proposes (best first) and `action` is its first step, kept for the
    offline teacher and the older log readers. `trajectory` is the digest from
    `trajectory_digest`; passing it is what turns "pick a label" into "break this impasse".
    System 2 sees all available actions, not just the currently-ready ones, so it can propose
    a sequence that achieves preconditions before executing the stuck label, and the `failure`
    brief carries the computed reason (pack fullness, nearest drop) so the model is not left to
    reverse-engineer why the stuck label keeps refusing.
    """
    if not api_key:
        return {"error": f"missing {cfg.api_key_env}"}
    try:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.base_url, api_key=api_key)
        phase_slug, phase_goal = phase_brief(state)
        system = (
            # The same phase sentence System 1 reads at the head of every state - System 2 is
            # asked about this run, not about Minecraft in general, and should not have to
            # re-derive mid-Nether from `state.dimension` whether the next thing is a pickaxe.
            # The phase goal spells out the preconditions the run is working toward, so the model
            # can tell a stuck label from a step that still serves the goal.
            f"{phase_persona(state)}. The run phase is `{phase_slug}`: {phase_goal} "
            "You are the Minecraft expert advising this bot on its "
            "next move. Its own controller is stuck: read `failure` (which names the stuck label and "
            "the computed reason it keeps refusing), the `trajectory` of what it has been doing, and "
            "what each entry in `actions` does. "
            "Some actions are immediately available (ready: true), others need preconditions met first. "
            "Propose the shortest ordered sequence of actions that breaks the impasse, "
            "including setup steps if needed. The sequence may include actions that are not currently "
            "ready, if getting them ready (e.g. shedding a full pack with drop_junk, or moving to a "
            "distant drop) would solve the stuck label. "
            "Reply with JSON "
            '{"plan": [<action labels, best first>], "goal": <short string>, '
            '"note": <why this breaks the loop, short>}.'
        )
        failure = {**failure, "diagnostics": _failure_diagnostics(state)}
        payload = {
            "state": _system2_state_view(state),
            "failure": failure,
            # All actions with descriptions, plus which are currently ready.
            "actions": {
                label: {"description": CRITERIA.get(label, ""), "ready": label in allowed_labels}
                for label in CRITERIA
            },
            "recent_history": history[-8:],
        }
        if trajectory is not None:
            payload["trajectory"] = trajectory
        user = json.dumps(payload, ensure_ascii=False)
        r = client.chat.completions.create(
            model=cfg.model,
            # Reasoning model: thinking tokens are emitted before the JSON (the local
            # Strata server reports them as `reasoning_content`), so the cap must leave
            # room for both. `[llm] max_tokens` follows the model's 4096 output limit;
            # a user can raise it if the server allows more.
            max_tokens=cfg.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        choice = r.choices[0]
        content = choice.message.content
        if not content:
            return {"error": "empty model response", "finish_reason": choice.finish_reason}
        data = json.loads(content)
        if not isinstance(data, dict):
            return {"error": f"non-object model response: {type(data).__name__}"}
        steps = _plan_steps(data, CRITERIA.keys())
        data["plan"] = steps
        data["action"] = steps[0] if steps else None
        return data
    except Exception as e:  # escalation must never crash the loop
        return {"error": str(e)}
