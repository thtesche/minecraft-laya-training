"""Phase-0 telemetry probe + cache A/B for the parallel observer.

Answers two questions against the live Strata endpoint (`settings.llm`), read-only:

  --spike       What cache/timing fields does the endpoint return, and does it reuse a
                prefix across requests? Fires a cold call, an identical repeat, and a
                branching variant, and dumps usage + llama.cpp `timings`.

  --story LOG   Does the observer's shape (one append-only "story" conversation) stay
                cheap tick after tick? Replays N teachable action ticks from an archived
                decision log as a growing conversation and reports, per tick, how much of
                the prompt was served from cache vs freshly prefilled.

Findings this tool established (run224, qwen3.8-flash-next-q2_0):
  * The endpoint reports `usage.prompt_tokens_details.cached_tokens` AND llama.cpp
    `timings.cache_n`/`prompt_ms` - caching is fully measurable.
  * Pass `extra_body={"cache_prompt": true}`: without it, reuse across a *branching*
    request is unreliable (0 cached); with it, reuse is immediate and reliable.
  * Append-only is the reuse path: each tick reuses the whole prior story and prefills
    only the ~150 new tokens (~1.1 s) instead of the whole prompt cold (~5 s).
  * The system prefix must be BYTE-STABLE for the whole run. Editing it mid-story (e.g.
    to toggle reasoning) diverges the prompt at token 0 and throws the cache away - toggle
    thinking in the appended turn, never in the system block.
  * Prefill gets cheap, but the model's reasoning (predicted tokens, ~27 ms/token) does
    not cache. Routine ticks must run light; spend full reasoning only on events.

Nothing here touches the bot or any log the loop reads. Telemetry -> runs/cache_probe.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlaya.config import load_settings  # noqa: E402
from mlaya.perception import _PHASE_ROWS, phase_brief  # noqa: E402


def _client(cfg, api_key):
    from openai import OpenAI
    return OpenAI(base_url=cfg.base_url, api_key=api_key)


def _call(client, model, messages, max_tokens):
    """One completion via with_raw_response, returning (text, telemetry) with cache numbers.

    `cache_prompt=true` is the knob the spike found: it makes the server reuse the slot's
    prefix KV reliably, not just on an exact repeat.
    """
    t0 = time.time()
    raw = client.chat.completions.with_raw_response.create(
        model=model, max_tokens=max_tokens, messages=messages,
        extra_body={"cache_prompt": True})
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
        "predict_ms": round(timings.get("predicted_ms") or 0),
        "latency_s": round(dt, 2),
    }
    return text, tel


def _log(settings, record: dict) -> None:
    out = settings.root / "runs" / "cache_probe.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**record, "ts": time.time()}) + "\n")


def spike(settings) -> None:
    client = _client(settings.llm, settings.api_key())
    prefix = "You are a Minecraft survival narrator. " + \
             " ".join(f"Note {i}: stay consistent." for i in range(1, 200)) + " /no_think"
    base = [{"role": "system", "content": prefix}, {"role": "user", "content": "Say ONE."}]
    branch = [{"role": "system", "content": prefix}, {"role": "user", "content": "Say TWO instead."}]
    print(f"model={settings.llm.model}  base_url={settings.llm.base_url}")
    for name, msgs in (("cold", base), ("identical-repeat", base), ("branch", branch)):
        _, tel = _call(client, settings.llm.model, msgs, 8)
        print(f"  {name:16s}: {json.dumps(tel)}")
        _log(settings, {"mode": "spike", "variant": name, **tel})


def _system_prefix() -> dict:
    """The observer's byte-stable story spine: the six phases and the reply schema.

    Kept identical for the whole run so the cached prefix is never invalidated.
    """
    phases = "\n".join(f"- {k}: {v[1]}" for k, v in _PHASE_ROWS.items())
    return {"role": "system", "content":
            "You narrate and judge a Minecraft bot's run in parallel to its System-1 (Laya). "
            "Keep your own plan for the current phase, watch what Laya does, and say whether you "
            f"agree.\nPhases and goals:\n{phases}\n"
            'Reply with JSON {"phase":..,"agree":true|false,"risk":null|"..","next":".."}. /no_think'}


def _observation(rec: dict) -> dict:
    state = rec.get("state") or {}
    return {"role": "user", "content": json.dumps({
        "phase": phase_brief(state)[0] if state else "?",
        "laya_label": rec.get("label"),
        "candidates": rec.get("candidates"),
        "progressed": rec.get("progressed"),
        "status": rec.get("status"),
        "situation": (rec.get("narrative") or "")[:300],
    }, ensure_ascii=False)}


def story(settings, log_path: Path, n: int) -> None:
    client = _client(settings.llm, settings.api_key())
    recs = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    acts = [r for r in recs if r.get("stage") == "action"]
    picks = acts[:: max(1, len(acts) // n)][:n]
    conv = [_system_prefix()]
    print(f"STORY on {log_path.name}: {len(picks)} ticks, append-only, cache_prompt=true")
    for i, rec in enumerate(picks):
        conv.append(_observation(rec))
        text, tel = _call(client, settings.llm.model, conv, 64)
        # Strip reasoning from stored history (the appended turn is the verdict only) to bound
        # context; the reuse still covers everything up to this observation.
        conv.append({"role": "assistant", "content": text or "{}"})
        print(f"  tick{i:2d}: cached={tel['cached_tokens']:5d} fresh={tel['fresh_tokens']:4d} "
              f"prefill={tel['prefill_ms']:5d}ms total={tel['latency_s']}s")
        _log(settings, {"mode": "story", "tick": i, "iteration": rec.get("iteration"), **tel})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", action="store_true")
    ap.add_argument("--story", type=str, default="", help="archived decision log to replay")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    settings = load_settings()
    if not settings.api_key():
        print(f"missing {settings.llm.api_key_env}", file=sys.stderr)
        sys.exit(2)
    if args.spike:
        spike(settings)
    if args.story:
        story(settings, Path(args.story), args.n)
    if not args.spike and not args.story:
        ap.error("pass --spike and/or --story LOG")


if __name__ == "__main__":
    main()
