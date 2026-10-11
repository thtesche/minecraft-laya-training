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


@dataclass
class Decision:
    action_label: str | None
    deferred: bool = False
    candidates: list[str] = field(default_factory=list)
    answers: dict = field(default_factory=dict)
    probabilities: dict = field(default_factory=dict)
    note: str = ""


class Brain:
    """System 1: Laya in-process, one forward pass per tick."""

    def __init__(self, cfg: LayaConfig):
        from laya import Router

        kwargs: dict[str, Any] = {"preload": cfg.preload, "default": cfg.model or "multilingual"}
        if cfg.device:
            kwargs["device"] = cfg.device
        self._cfg = cfg
        self.router = Router(**kwargs)

    def _questions(self, candidates: list[str]) -> dict[str, dict]:
        questions: dict[str, dict] = {"in_danger": DANGER_Q, "danger": DANGER_SCORE}
        if len(candidates) > 1:
            questions["intent"] = {
                "type": "choice",
                "instructions": "The bot is safe enough to work. What should it do next?",
                "criteria": {label: CRITERIA[label] for label in candidates},
            }
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

        in_danger = answers.get("in_danger", {}).get("choice") == "yes"
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
        label = answers.get("intent", {}).get("choice")
        if label not in candidates:
            label = candidates[0]
        return Decision(label, candidates=candidates,
                        answers=answers, probabilities=probabilities)


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
            max_tokens=min(cfg.max_tokens, 1024),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return json.loads(r.choices[0].message.content or "{}")
    except Exception as e:  # escalation must never crash the loop
        return {"error": str(e)}
