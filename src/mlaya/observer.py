"""The parallel observer: an LLM that builds its own "story" of the run alongside Laya.

Not a per-tick judge. A persistent narrator with its own plan: it watches the decision log the
loop already writes, keeps a running read of *what it would do to play this game*, *what Laya is
doing*, and *whether it agrees*, and it always carries the current game phase. It is read-only -
it reads `runs/decisions.jsonl` (and may be given read-only MCP later), never touches the bot or
any log the loop reads, and writes its running judgment to `runs/observer.jsonl`.

The whole thing is ONE append-only conversation (`Story`). That is the design, not an accident:
`scripts/cache_probe.py` showed the Strata endpoint reuses a prefix only for a pure extension (a
branch caches 0), so an append-only story is the only shape that stays cheap - each tick reuses
the whole prior conversation and prefills only the ~200 new tokens. Routine ticks run light
(`/no_think`, ~1-3 s); the model spends full reasoning only on *events* - a phase change, an
escalation, a stall, a death. The system prefix is byte-stable for the whole run (editing it
mid-story throws the cache away), so the thinking toggle lives in the appended turn, never in it.

The LLM call is injected (`llm_fn`), exactly like the offline teacher, so the event logic, the
story bookkeeping and the trim can be tested with no network.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import ObserverConfig
from .perception import _PHASE_ROWS, phase_brief

# llm_fn(messages, reason) -> (verdict_dict, telemetry). `reason` asks for a full-reasoning turn.
LLMFn = Callable[[list[dict], bool], "tuple[dict, dict]"]

# The death traps the pure-MCP sessions only learned after dying (gold armour in the Nether, etc.).
# They sit in the byte-stable system prefix so the observer can flag a risk BEFORE it bites, and so
# the prefix the cache holds never changes. Kept here, close to the prompt, rather than in a data
# file for now - the set is small and changes with the prompt it rides in.
DEATH_TRAPS = (
    "In the Nether, piglins turn hostile unless gold armour is worn - never provoke them bare.",
    "Never dig straight down: lava or a long fall is one block away.",
    "Keep a bed to reset spawn and a cooked-food reserve before crossing to the Nether.",
    "Ghasts and blazes hit from range; crossing lava without a bridge or fire resistance is fatal.",
)

# Each phase goal in _PHASE_ROWS names its END STATE, not the ordered precursor steps that reach it.
# Without the precursors the observer mistakes a legitimate buildup step for a detour - live (run245)
# it judged "mine_iron" and "smelt_iron" in diamond_kit as "loop: mine diamond ore for the kit",
# though an iron pickaxe is the precondition for mining diamond at all. These lines teach the
# within-phase progression so a precursor reads as progress. They live here, not in _PHASE_ROWS, so
# the live loop's own phase spine stays untouched.
# The turn appended when the story is folded at the budget: the model writes its own hand-off
# instead of leaving a char-truncated list of past verdicts behind. It asks for the same JSON shape
# the verdict parser already reads (`story` field), so no second parse path is needed, and it names
# the four things the next half of the run actually needs carried: where the run stands, what risk
# is still open, the current plan, and the gear trend. `/think` - a fold is rare and worth real
# reasoning. Kept here by the prompt it folds, like the prefix.
CHECKPOINT_REQUEST = (
    "STOP judging for one turn. The story so far is about to be folded away to save context. "
    "Write a compact hand-off for your future self in <=120 words: the phases already passed and "
    "where the run stands now, any risk or loop still unresolved, your current plan for this "
    'phase, and how the gear has trended. Reply with ONLY {"story": "<the hand-off>"}. /think'
)


PHASE_PROGRESSIONS = (
    "diamond_kit: the tool ladder is wood -> stone -> IRON -> diamond. An iron pickaxe is REQUIRED "
    "to mine diamond ore at all, so mining/smelting iron and crafting iron tools IS progress toward "
    "the diamond kit, never a detour. The shield belongs to the same kit: the run is not allowed to "
    "leave the base without one, so crafting it and keeping it in the off-hand is kit work, not a "
    "side quest - and body armour is optional, so a kit without it is still a kit.",
    "prepare_nether: shield, bow+arrows, cooked food, a bed, a chest at the base, a lit portal are "
    "separate items - gathering any one of them is progress.",
    "nether_rods: kill blazes for rods, craft powder, bank it at the gate - each hunt serves the haul.",
    "nether_pearls: barter gold with piglins and hunt endermen for pearls - both pay for the eyes.",
    "stronghold: craft eyes of ender, throw to locate the stronghold, activate the end portal.",
)


def system_prefix() -> dict:
    """The observer's byte-stable story spine: its role, the six phases, the traps, the schema.

    Built once and never changed during a run - the whole prefix cache hangs off it. The reply
    schema is stated once, plainly; the thinking toggle is appended per observation, not here.
    """
    phases = "\n".join(f"- {key}: {goal}" for key, (_, goal) in _PHASE_ROWS.items())
    progress = "\n".join(f"- {p}" for p in PHASE_PROGRESSIONS)
    traps = "\n".join(f"- {t}" for t in DEATH_TRAPS)
    content = (
        "You are a Minecraft survival expert watching a bot play, in parallel to its System-1 "
        "controller (Laya, a fast typed decision engine). You keep your OWN running plan for the "
        "current phase, you watch what Laya actually does each step, and you say whether you agree. "
        "You never control the bot - you narrate and judge.\n\n"
        "The run works through these phases in order, each with its goal (the END STATE):\n"
        + phases + "\n\n"
        "A phase's goal is reached through ordered precursor steps. Judge each step by whether it "
        "serves the phase, NOT by whether it names the phase's end state:\n" + progress + "\n"
        "A loop is the SAME label repeating WITHOUT progress. When a step reports progressed=true it "
        "advanced the world - never call that a loop.\n\n"
        "Laya's labels are high-level intents; the tool layer carries out each one's own sub-steps - "
        "placing a crafting table, pathing, gathering an immediate prerequisite. Judge the STRATEGY "
        "(is this the right aim for the phase?), not the mechanics: a step with status=succeeded did "
        "what it set out to do, so do not raise a missing-precondition risk (no crafting table, no "
        "tool) for an action that already succeeded.\n\n"
        "Each observation may carry `gear`: the armour worn per slot and the pickaxe and sword "
        "held, each with its durability in %, plus the off-hand item (a shield, or null). Read it "
        "before you judge the kit - never flag acquiring or 'focusing on' a piece the gear already "
        "shows worn or held; a shield in the off-hand IS the kit's shield. A piece the phase still "
        "needs worn down to a low durability (roughly <=20%) is a real risk worth the note.\n\n"
        "Some observations carry `blocked_goal`: the phase's end-state action (e.g. lighting the "
        "portal) that the engine is DELIBERATELY withholding this tick, with `needs` listing the "
        "prerequisites still unmet. That action is not a legal move now and will not be in "
        "`candidates` - so do NOT flag a loop or risk for 'never doing it', and NEVER name it in "
        "`better`. The right move is whatever candidate advances the first unmet `needs` item; judge "
        "Laya against that, and if a candidate clears a `needs` item while Laya chose something idle, "
        "that candidate is the better move.\n\n"
        "Known death traps - flag the risk BEFORE it bites, do not wait for the death:\n" + traps +
        "\n\nIf you disagree, name the alternative: in `better` give the single label from the "
        "`candidates` list that would have been the better move at this tick, or null when none of "
        "the candidates is better than what Laya did.\n\n"
        "Reply with ONLY one JSON object and nothing else - no preamble, no reasoning outside "
        "it:\n"
        '{"phase": <current phase slug>, "agree": true|false, '
        '"risk": null|<short string naming a danger or a loop>, '
        '"note": <<=12 words: what you would do / why you (dis)agree>, '
        '"better": null|<one label from the candidates list>}.'
    )
    return {"role": "system", "content": content}


# --- reading the decision log ------------------------------------------------------------------

def _phase_of(rec: dict) -> str | None:
    state = rec.get("state")
    return phase_brief(state)[0] if isinstance(state, dict) and state else None


def _gear_line(state: dict | None) -> dict | None:
    """What the bot wears and holds, with durability in %, so the judge reads the kit's real
    state instead of inferring it from the action stream.

    Without it the observer judged the kit blind: live (run247) it told the bot to "focus on
    shield" and flagged a step as a detour while the shield sat in the off-hand the whole time,
    and it could never see a tool wearing out before it broke. `None` for a stateless record so
    nothing is added when there is nothing to say.
    """
    if not isinstance(state, dict):
        return None

    def worn_pct(meta: dict) -> str:
        left, mx = meta.get("left"), meta.get("max")
        if isinstance(left, (int, float)) and isinstance(mx, (int, float)) and mx > 0:
            return f" {left / mx * 100:.0f}%"
        return ""

    gear: dict[str, Any] = {}
    armor = state.get("armor") or {}
    worn = {cls: f"{meta.get('tier')}{worn_pct(meta)}"
            for cls, meta in armor.items()
            if isinstance(meta, dict) and meta.get("tier") and meta.get("tier") != "none"}
    if worn:
        gear["armor"] = worn
    tool_meta = state.get("tool_meta") or {}
    tools = state.get("tools") or {}
    held = {}
    for cls in ("pickaxe", "sword"):
        meta = tool_meta.get(cls)
        meta = meta if isinstance(meta, dict) else {}
        tier = meta.get("tier") or tools.get(cls)
        if tier and tier != "none":
            held[cls] = f"{tier}{worn_pct(meta)}"
    if held:
        gear["tools"] = held
    # The off-hand is the whole point of the shield false-flag, so it rides along whenever there
    # is any gear at all - "offhand": null then says the shield is genuinely missing.
    offhand = state.get("offhand")
    if offhand or gear:
        gear["offhand"] = offhand
    return gear or None


@dataclass
class Observer:
    """Follows the decision log and drives one `Story`, writing a judgment per observation."""

    cfg: ObserverConfig
    llm_fn: LLMFn
    sink: Callable[[dict], None]
    messages: list[dict] = field(default_factory=list)
    # How many consecutive observations a differing phase must hold before the story accepts it as
    # a new phase. The loop's phase spine (`phase_brief`) is derived from live state, not latched,
    # so it flaps for a tick between neighbours (stronghold <-> nether_pearls) when inventory reads
    # ambiguously; without this a single flap would fire a spurious phase_change and a wasted
    # full-reasoning turn, and the narrative would thrash.
    phase_confirm: int = 2
    # Rolling context used to decide what is an *event* worth full reasoning.
    _phase: str | None = None          # the phase the story currently believes it is in
    _cand: str | None = None           # a differing phase awaiting confirmation
    _cand_n: int = 0
    _recent: list[dict] = field(default_factory=list)
    _death: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [system_prefix()]

    # --- event detection (pure; reads only the log records) ------------------------------------

    def _event(self, rec: dict) -> str | None:
        """Name the reason this record deserves a full-reasoning turn, or None for a routine tick.

        Events are the moments the story should actually stop and think: the run changing phase,
        the loop escalating to System 2, a stall (few labels, no progress over a window), or a
        death. Everything else is a light update that only keeps the narrative current.
        """
        # Always run phase confirmation so the debounce counter stays current even on records that
        # resolve to another event first (an escalation record carries only sparse state).
        phase_changed = self._confirm_phase(_phase_of(rec))
        if rec.get("stage") == "escalation":
            return "escalation"
        if phase_changed:
            return "phase_change"
        state = rec.get("state")
        # The logged state carries `last_death` - the death-cause string ("... swim in lava"),
        # the same text for every look at one death and a new text for the next - so a change in
        # it marks a fresh death. (The dataclass's `last_death_at` is not in the logged dict.)
        death = state.get("last_death") if isinstance(state, dict) else None
        if death and death != self._death:
            return "death"
        if self._is_stall():
            return "stall"
        return None

    def _confirm_phase(self, phase: str | None) -> bool:
        """Commit `phase` as the current one and report whether that is a *new* phase.

        A sparse-state record (phase None) is ignored, keeping any pending candidate alive. A phase
        that matches the committed one clears the candidate. A differing phase must hold for
        `phase_confirm` consecutive observations before it is accepted - except the very first
        phase of the run, which commits at once (entering the first phase is a real event).
        """
        if not phase:
            return False
        if phase == self._phase:
            self._cand, self._cand_n = None, 0
            return False
        self._cand_n = self._cand_n + 1 if phase == self._cand else 1
        self._cand = phase
        if self._phase is None or self._cand_n >= self.phase_confirm:
            self._phase = phase
            self._cand, self._cand_n = None, 0
            return True
        return False

    def _is_stall(self) -> bool:
        """Few distinct labels and no progress across the recent action window - the loop's own tell."""
        acts = [r for r in self._recent if r.get("stage") == "action"][-6:]
        if len(acts) < 6:
            return False
        labels = {a.get("label") for a in acts}
        progressed = any(a.get("progressed") for a in acts)
        return len(labels) <= 2 and not progressed

    # --- the observation the model reads -------------------------------------------------------

    @staticmethod
    def observation(rec: dict, event: str | None) -> dict:
        """One compact user turn: what Laya just did, in the words a next-move read needs."""
        payload = {
            "iteration": rec.get("iteration"),
            "phase": _phase_of(rec),
            "stage": rec.get("stage"),
            "laya_label": rec.get("label"),
            "candidates": rec.get("candidates"),
            "progressed": rec.get("progressed"),
            "status": rec.get("status"),
            "situation": (rec.get("narrative") or "")[:360],
        }
        gear = _gear_line(rec.get("state"))
        if gear:
            payload["gear"] = gear
        blocked = rec.get("blocked_goal")
        if blocked:
            payload["blocked_goal"] = blocked
        if event:
            payload["event"] = event
        return payload

    # --- the live loop over new records --------------------------------------------------------

    def ingest(self, rec: dict, *, sampled: bool = False) -> dict | None:
        """Observe one log record; return the judgment written, or None when the tick was skipped.

        `sampled` is set by the tailer when it is behind: a routine tick may then be dropped to
        catch up, but an event is never skipped - the story must not miss a phase change or a death
        just because perception outran the model.
        """
        if rec.get("stage") not in ("action", "defer", "escalation"):
            return None
        self._recent.append(rec)
        self._recent = self._recent[-24:]
        event = self._event(rec)
        if sampled and event is None:
            self._advance(rec)
            return None

        obs = self.observation(rec, event)
        think = event is not None
        self.messages.append({"role": "user",
                              "content": json.dumps(obs, ensure_ascii=False) +
                                         (" /think" if think else " /no_think")})
        verdict, tel = self.llm_fn(self.messages, think)
        # Store only the verdict in history (reasoning stripped) so the prefix stays small and the
        # next tick is a pure extension of this one.
        self.messages.append({"role": "assistant",
                              "content": json.dumps(verdict, ensure_ascii=False)})
        self._trim(tel)

        out = {
            "ts": time.time(),
            "iteration": rec.get("iteration"),
            "phase": obs["phase"],
            "event": event,
            "reasoned": think,
            "laya_label": rec.get("label"),
            "progressed": rec.get("progressed"),
            # The gear the judge was shown, logged with the verdict: without it the Phase-3 join
            # cannot tell a gear-aware judgment from a guess, because the payload itself is not
            # archived (only the verdict and its telemetry are).
            "gear": obs.get("gear"),
            "verdict": verdict,
            "telemetry": tel,
        }
        self.sink(out)
        self._advance(rec)
        return out

    def _advance(self, rec: dict) -> None:
        # Phase commitment is owned by `_confirm_phase` (debounced); here only the death marker is
        # advanced, so the next differing death-cause string reads as a fresh death.
        state = rec.get("state")
        death = state.get("last_death") if isinstance(state, dict) else None
        if death:
            self._death = death

    def _trim(self, tel: dict) -> None:
        """Fold the oldest story turns into one checkpoint when the prompt nears the input budget.

        A trim is the one moment the prefix after the system block changes, so the cache takes a
        one-off miss - but only the kept tail re-prefills, never the byte-stable system prefix. The
        budget (80k, well under the 185k input ceiling) is set to fold periodically rather than
        almost-never, so the live prompt stays lean; each fold is a cheap, infrequent miss. What is
        folded away is replaced by a model-written `_checkpoint`, not a char-truncated digest, so
        the narrative thread the story observer lives on survives the fold.
        """
        if (tel.get("prompt_tokens") or 0) < self.cfg.input_budget:
            return
        head = self.messages[0]
        body = self.messages[1:]
        if len(body) <= self.cfg.keep_turns:
            return
        keep = body[-self.cfg.keep_turns:]
        dropped = body[:-self.cfg.keep_turns]
        self.messages = [head, self._checkpoint(dropped), *keep]

    def _checkpoint(self, dropped: list[dict]) -> dict:
        """One model-written "story so far" turn that replaces the folded `dropped` turns.

        The request rides the `head + dropped` prefix the cache already holds, so only its own
        tokens prefill - the fold's one miss is still just the rebuilt `[head, summary, keep]` the
        next tick diverges into. A fold must never crash the follow loop or lose the thread, so a
        failed or empty checkpoint falls back to the char-truncated digest of the dropped verdicts.
        """
        story = ""
        try:
            req = {"role": "user", "content": CHECKPOINT_REQUEST}
            verdict, _tel = self.llm_fn([self.messages[0], *dropped, req], True)
            story = (verdict.get("story") or verdict.get("note") or "").strip()
        except Exception:
            story = ""
        if not story:
            story = json.dumps([m.get("content", "")[:120] for m in dropped
                                if m.get("role") == "assistant"][-20:], ensure_ascii=False)
        return {"role": "user", "content": "STORY SO FAR (older turns folded): " + story}


# --- live LLM wiring (replaced by a fake in tests) ---------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _parse_verdict(text: str) -> dict:
    """The JSON object the model replied with, reasoning stripped; a shape it can always be read as."""
    cleaned = _THINK_RE.sub("", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"phase": None, "agree": None, "risk": None, "note": cleaned[:200]}


def make_llm_fn(cfg: ObserverConfig, api_key: str | None) -> LLMFn:
    """An `Observer.llm_fn` backed by the Strata endpoint, append-only and cache-friendly.

    `cache_prompt=true` keeps the slot's prefix KV warm across turns; the generous `max_tokens`
    (the model's 16384 output ceiling) leaves room for a full reasoning turn without truncating
    the JSON the way a 64-token cap did in the probe. Reasoning is requested only when `reason`
    is set - the routine path rides on the `/no_think` directive the caller appended.
    """
    from openai import OpenAI

    client = OpenAI(base_url=cfg.base_url, api_key=api_key) if api_key else None

    def llm(messages: list[dict], reason: bool) -> tuple[dict, dict]:
        if client is None:
            return {"phase": None, "agree": None, "risk": None, "note": f"missing {cfg.api_key_env}"}, {}
        extra: dict[str, Any] = {"cache_prompt": True}
        if reason and cfg.reasoning_effort:
            extra["reasoning_effort"] = cfg.reasoning_effort
        t0 = time.time()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens if reason else cfg.routine_max_tokens,
                messages=messages,
                extra_body=extra,
            )
        except Exception as e:  # the observer is side tooling; never crash the follow loop
            return {"phase": None, "agree": None, "risk": None, "note": f"error: {e}"}, {"error": str(e)}
        dt = time.time() - t0
        body = raw.http_response.json()
        usage = body.get("usage") or {}
        timings = body.get("timings") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        msg = body["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
        tel = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached,
            "fresh_tokens": (usage.get("prompt_tokens") or 0) - cached,
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_ms": round(timings.get("prompt_ms") or 0),
            "latency_s": round(dt, 2),
            "reasoned": reason,
        }
        return _parse_verdict(text), tel

    return llm
