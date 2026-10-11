from __future__ import annotations

# The server-side survival policy. These rules govern what the bot does on its own
# whenever the model is not driving the body (danger deferrals, reflex eating, path
# scaffolding), so they are configuration, not code that Laya re-implements.
#
# The values mirror the "Survival policy that works" set from the mine-ai skill:
# deepslate/tuff scaffold blocks for underground travel, a cautious combat floor,
# and a food reflex that eats raw food instead of hoarding it until starvation.
#
# The whole block is passed on every `set` because a partial block replaces the
# policy table rather than merging into it.
POLICY_CHANGES: dict = {
    "navigation": {
        "scaffold_blocks": [
            "dirt", "cobblestone", "cobbled_deepslate",
            "netherrack", "basalt", "end_stone",
        ],
        "hostile_avoidance_multiplier": 3.0,
        "bucket_fall_save": True,
        "bucket_drops": True,
    },
    "combat": {
        "engagement": "respond_to_threats",
        "hide": "when_recovery_possible",
        "recover": "when_possible",
        "retreat": True,
        "melee": True,
        "bow": True,
        "shield": True,
        "terrain": {"dig": True, "place": True},
        "engage_min_health": 17,
        "critical_health": 14,
        "recover_to_health": 18,
        "protected_wait_ticks": 300,
        # 40 (not the 300 default) or enderman hunts never finish.
        "enderman_wait_ticks": 40,
        "volley_wait_ticks": 100,
        "recovery_timeout_ms": 30000,
        "evade_timeout_ms": 30000,
        "evade_safe_range": 36,
    },
    "food": {
        # "always" so hunger never stalls a run: the reflex eats gathered food
        # instead of banking it until the bot starves.
        "raw": {"allow": "always", "hunger_at_most": 20, "health_below": 10},
    },
}

POLICY_LIFETIME = {"kind": "session"}


def _policy_node(data: dict) -> dict:
    node = (data or {}).get("response", {}).get("data", {}) or {}
    pol = node.get("survivalPolicy")
    return pol if isinstance(pol, dict) else {}


async def read_policy(executor) -> dict:
    """Return the server's current survivalPolicy block (revision, defaults, overrides)."""
    res = await executor.read("view_status", {}, "read the effective survival policy")
    return _policy_node(res.data or {})


async def apply_policy(executor, reason: str) -> dict:
    """Set the full working survival policy against the current revision.

    `expected_revision` is optimistic concurrency: we read it, then set. The server
    restores defaults on every death and dimension change, so this must be re-applied
    after each one (see `graph.Agent._reapply_after_death`).
    """
    policy = await read_policy(executor)
    revision = policy.get("revision")
    if not revision:
        return {"applied": False, "error": "no revision in view_status"}
    res = await executor.read("set_survival_policy", {
        "operation": "set",
        "expected_revision": revision,
        "changes": POLICY_CHANGES,
        "lifetime": POLICY_LIFETIME,
        "reason": reason[:200],
    }, "apply the working survival policy (scaffold, combat floor, raw food)")
    effective = _policy_node(res.data or {}).get("effective") or {}
    return {
        "applied": bool(res.ok),
        "revision": revision,
        "status": res.status,
        "raw_food": ((effective.get("food") or {}).get("raw")),
        "error": None if res.ok else (res.text or "")[:300],
    }
