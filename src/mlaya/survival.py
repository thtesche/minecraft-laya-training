from __future__ import annotations

import copy

# The server-side survival policy. These rules govern what the bot does on its own
# whenever the model is not driving the body (danger deferrals, reflex eating, path
# scaffolding), so they are configuration, not code that Laya re-implements.
#
# Two stages ("Stufen"):
#   base  - cautious travel/progression, safe to leave on: don't *pursue* below 14 health,
#           heal back to 16 between fights, eat raw food rather than hoard. (Was 17/18; a
#           lone melee spider at 16.5 health fell just under the pursuit floor, so the bot
#           evaded instead of killing it, then bled out below the old emergency line - see
#           the tuning note on `combat` below.)
#   hunt  - an explicitly requested hostile hunt runs with the floor dropped to zero
#           so the engagement actually happens, then the base stage is restored.
# The auto-eating stage is `food.raw.allow`: "always" (base/hunt) vs "emergency_only"
# (the server default, which only eats raw below hunger 6 AND health 10).
#
# The whole block is passed on every `set` because a partial block replaces the
# policy table rather than merging into it.
BASE_CHANGES: dict = {
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
        # Tuned 2026-10-13: the server treats immediate contact defence as separate from this
        # pursuit floor, so a health just under 17 turned a lone spider into an evasion and a
        # hide. 14 opens a fighting band; 10 keeps emergency protection for the truly critical.
        "engage_min_health": 14,
        "critical_health": 10,
        "recover_to_health": 16,
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
        # instead of banking it until the bot starves. At full hunger it cannot eat,
        # so drops still accumulate into the reserve instead of being spent.
        "raw": {"allow": "always", "hunger_at_most": 20, "health_below": 10},
    },
}

# The hunt stage. `combat.engage_min_health` above the bot's current health blocks an
# explicitly requested hunt ([HUNT_CAPABILITY_BLOCKED]); the skill's confirmed fix is
# to drop the floor to zero for the duration of the hunt and restore it afterwards.
# The zero floor is deliberately temporary: with it the hostile reflex will hold the
# body and block navigation, so it is restored the moment the hunt ends.
HUNT_CHANGES = copy.deepcopy(BASE_CHANGES)
HUNT_CHANGES["combat"].update({
    "engage_min_health": 0,
    "critical_health": 0,
    "retreat": False,
})

# Full health in Minecraft: ten hearts, twenty points. An engagement floor above it can
# never be met, so the reflex never *pursues* - the mirror of the hunt stage's zero floor.
MAX_HEALTH = 20

# The flee stage. The exact inverse of the hunt stage: where the hunt drops the pursuit
# floor to zero so the reflex fights, the flee raises it past full health so the reflex
# never pursues - it evades to `evade_safe_range` and hands the body back. That is what
# frees a `navigate` the model issued (the way home, haul done) from being held behind an
# endless spawner fight. Live, run106: `leave_nether` came back `[BODY_HELD]` twice at a
# blaze spawner (eleven-twelve blazes at five metres, owner hostile_reflex, 300 s each),
# and the run stopped on `body_held` with twelve rods it never carried out. `critical_health`
# is left at the base floor so a cornered bot still defends itself below ten; only the
# pursuit of a fight it should be walking away from is switched off. Restored to base the
# moment the crossing ends, like the hunt stage.
FLEE_CHANGES = copy.deepcopy(BASE_CHANGES)
FLEE_CHANGES["combat"].update({
    "engage_min_health": MAX_HEALTH + 1,
    "retreat": True,
})

PRESETS: dict[str, dict] = {"base": BASE_CHANGES, "hunt": HUNT_CHANGES, "flee": FLEE_CHANGES}

# The base combat floor; a hunt at or above it would pass unmodified.
BASE_ENGAGE_MIN_HEALTH = BASE_CHANGES["combat"]["engage_min_health"]

POLICY_LIFETIME = {"kind": "session"}


def _policy_node(data: dict) -> dict:
    node = (data or {}).get("response", {}).get("data", {}) or {}
    pol = node.get("survivalPolicy")
    return pol if isinstance(pol, dict) else {}


async def read_policy(executor) -> dict:
    """Return the server's current survivalPolicy block (revision, defaults, overrides)."""
    res = await executor.read("view_status", {}, "read the effective survival policy")
    return _policy_node(res.data or {})


async def apply_policy(executor, reason: str, preset: str = "base") -> dict:
    """Set a survival-policy preset against the current revision.

    `expected_revision` is optimistic concurrency: we read it, then set. The server
    restores defaults on every death and dimension change, so this must be re-applied
    after each one (see `graph.Agent._reapply_after_death`).
    """
    changes = PRESETS.get(preset)
    if changes is None:
        return {"applied": False, "error": f"unknown preset {preset!r}"}
    policy = await read_policy(executor)
    revision = policy.get("revision")
    if not revision:
        return {"applied": False, "error": "no revision in view_status"}
    res = await executor.read("set_survival_policy", {
        "operation": "set",
        "expected_revision": revision,
        "changes": changes,
        "lifetime": POLICY_LIFETIME,
        "reason": reason[:200],
    }, f"apply the '{preset}' survival policy stage")
    effective = _policy_node(res.data or {}).get("effective") or {}
    return {
        "applied": bool(res.ok),
        "preset": preset,
        "revision": revision,
        "status": res.status,
        "engage_min_health": (effective.get("combat") or {}).get("engage_min_health"),
        "raw_food": (effective.get("food") or {}).get("raw"),
        "error": None if res.ok else (res.text or "")[:300],
    }


async def apply_hunt(executor, reason: str) -> dict:
    """Enter the hunt stage: aggressive combat floor for a deliberately requested hunt."""
    return await apply_policy(executor, reason, preset="hunt")


async def apply_flee(executor, reason: str) -> dict:
    """Enter the flee stage: a pursuit floor past full health, so the reflex evades instead of
    fighting and frees the body for the way home."""
    return await apply_policy(executor, reason, preset="flee")


async def restore_base(executor, reason: str) -> dict:
    """Leave the hunt stage and re-arm the cautious base floor."""
    return await apply_policy(executor, reason, preset="base")
