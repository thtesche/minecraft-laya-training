from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .actions import CRITERIA
from .config import LLMConfig, LayaConfig

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
        "no": "no: no mob is attacking and health is comfortably high",
        "yes": "yes: a mob is attacking now or health is very low",
    },
}
DANGER_SCORE = {
    "type": "score",
    "instructions": "How dangerous is the situation?",
    "criteria": ["none", "low", "high", "critical"],
}
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

    def decide(self, state: str, candidates: list[str]) -> Decision:
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
        if in_danger:
            return Decision(None, deferred=True, candidates=candidates,
                            answers=answers, probabilities=probabilities,
                            note="in danger -> survival policy")
        if not candidates:
            return Decision(None, deferred=True, answers=answers,
                            probabilities=probabilities, note="no valid action")
        if len(candidates) == 1:
            return Decision(candidates[0], candidates=candidates,
                            answers=answers, probabilities=probabilities)
        label = answers.get(INTENT_QID, {}).get("choice")
        if label not in candidates:
            label = candidates[0]
        return Decision(label, candidates=candidates,
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
            "You are a Minecraft expert advising an autonomous bot. "
            "Pick exactly one next action from allowed_actions and reply with JSON "
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
            # Generous budget: nemotron is a reasoning model, so reasoning tokens
            # are spent before the JSON is emitted (a small cap yields empty content).
            max_tokens=min(cfg.max_tokens, 4096),
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
