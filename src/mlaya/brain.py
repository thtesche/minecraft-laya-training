from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .actions import (CRITERIA, DEFER_QUIET_TICKS, RECOVER_HEALTH, _must_eat,
                      _recovery_owed)
from .config import LLMConfig, LayaConfig
from .perception import phase_persona

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
        if in_danger:
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


def escalate_to_system2(
    cfg: LLMConfig,
    api_key: str | None,
    state: dict,
    history: list[dict],
    failure: dict,
    allowed_labels: list[str],
) -> dict | None:
    """System 2: ask a generative model to pick a valid next action (rare)."""
    if not api_key:
        return {"error": f"missing {cfg.api_key_env}"}
    try:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.base_url, api_key=api_key)
        system = (
            # The same phase sentence System 1 reads at the head of every state - System 2 is
            # asked about this run, not about Minecraft in general, and should not have to
            # re-derive mid-Nether from `state.dimension` whether the next thing is a pickaxe.
            phase_persona(state) + ". You are the Minecraft expert advising this bot on its "
            "next move. Pick exactly one next action from allowed_actions and reply with JSON "
            '{"action": <one allowed label or null>, "goal": <short string>, '
            '"note": <why, short>}.'
        )
        user = json.dumps(
            {
                "state": state,
                "recent_history": history[-5:],
                "failure": failure,
                "allowed_actions": allowed_labels,
            },
            ensure_ascii=False,
        )
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
        return json.loads(content)
    except Exception as e:  # escalation must never crash the loop
        return {"error": str(e)}
