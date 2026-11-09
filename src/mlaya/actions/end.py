from __future__ import annotations

from .predicates import *  # noqa: F401,F403

# --- Phase 5: the Ender Dragon fight (the step after the stronghold) ---
# Design: the Phase 5 plan. The server owns the hard combat - destroying a crystal, leading a bow
# shot, building and timing a perch, the MLG-water reflex, the blast-safety arithmetic - exactly as
# the Nether and stronghold phases leave the hard mechanics to it. This module only reads the loaded
# fight (`endFight` in perception), filters which tool the moment allows, books it on the loaded
# entity, and latches the one goal: the dragon dead. The phase stops there; the return path (egg,
# exit portal, homecoming) is a later phase (memo part2-return-path-deferred).
#
# It mirrors `stronghold.py`: `END_LABELS` the module owns, `end_candidates` the planner splices,
# `resolve_end` the main resolver tries first, and `end_complete` the goal reads.

END_LABELS = frozenset({
    "carve_pumpkin", "enter_the_end", "position_end", "gather_end_stone",
    "destroy_crystal", "shoot_dragon", "prepare_perch", "attack_perch",
})


def end_complete(gs: GameState) -> bool:
    """The Phase 5 goal: the dragon is dead.

    Latched from a kill receipt (`Agent._note_dragon`), the same shape as `powder_stowed`, never
    read from the fight entities going unloaded - an empty `endFight` list is "nothing loaded", not
    "all dead", so only an observed death closes the run.
    """
    return bool(getattr(gs, "dragon_slain", False))


def _end_phase_active(gs: GameState) -> bool:
    """The run owns the End fight: the end portal stands (stronghold done), the respawn is pinned by
    a stronghold sleep, and the dragon is not yet dead.

    True in both windows the phase wears: the Overworld beside the portal (the first crossing and
    every death-respawn re-entry) and the End itself. `end_portal_active` and the `slept_at_stronghold`
    latch are the durable proofs the stronghold phase left behind; the live bed scan is not read here
    (it is False the moment the bot steps into the End), so the phase stays owned across the crossing.
    """
    return (bool(gs.end_portal_active)
            and bool(getattr(gs, "slept_at_stronghold", False))
            and not end_complete(gs))


def _pumpkin_wearable(gs: GameState) -> bool:
    """A carved pumpkin carried but not yet worn.

    Worn as a helmet it hides the enderman gaze so a look never provokes one. Best-effort: with
    none carried the step never appears and the crossing proceeds on the server's own gaze guard.
    A successful equip spends the pumpkin out of the pack (worn armour leaves the inventory), so
    `_has` falls False and the step drops out on its own - it never loops.
    """
    return _has(gs.inventory, CARVED_PUMPKIN) and not _worn(gs, CARVED_PUMPKIN)


def _end_should_eat(gs: GameState) -> bool:
    """Below the End's health floor with food to eat: heal before booking another attack.

    The acute tick is already `_must_eat`'s (it owns the loop outright); this is the softer floor
    the fight keeps above, because there are no potions or golden apples by design and a meal is the
    only buffer the diamond kit and the MLG water do not already carry.
    """
    return (gs.health is not None and gs.health < END_HEALTH_FLOOR
            and bool(_first_edible(gs.inventory)))


def _end_gather_safe(gs: GameState) -> bool:
    """Whether mining end stone is worth the tick: not wounded below the floor.

    Mirrors `gather_scaffold`'s 'only when healthy' gate - a dig is not a fight, so it waits out a
    wound and lets the heal or the reflex own the low-health tick. The server interrupts the dig
    itself if a hostile reaches it, so no finer danger test is needed here.
    """
    return gs.health is None or gs.health >= END_HEALTH_FLOOR


def end_candidates(gs: GameState) -> list[str]:
    """The Phase 5 steps whose preconditions hold, in the order the fight walks them.

    Two faces behind one function, dispatched on the dimension, the way `stronghold_candidates` is
    Overworld-gated: in the End it is the fight; in the Overworld after the stronghold sleep it is
    the short entry/re-entry window. The planner early-returns this whole list for the End and
    splices it after the stronghold chain for the Overworld; both reach the same function here.
    """
    if _in_end(gs):
        return _end_combat_candidates(gs)
    if not _overworld(gs) or not _end_phase_active(gs):
        return []
    return _end_entry_candidates(gs)


def _end_entry_candidates(gs: GameState) -> list[str]:
    """The Overworld entry/re-entry window: best-effort pumpkin, top up the MLG water, step through."""
    inv = gs.inventory
    out: list[str] = []
    # Best-effort: wear a carried carved pumpkin so the enderman gaze never provokes one.
    if _pumpkin_wearable(gs):
        out.append("carve_pumpkin")
    # The MLG water bucket is what the survival reflex pours for a fall in the End. It rides down
    # from the Nether prep; if only an empty bucket is carried, top it up at a source here. Never a
    # gate - the entry proceeds on the supply reserve the stronghold phase already met.
    if _has(inv, BUCKET_ITEM) and not _has(inv, WATER_BUCKET):
        out.append("fill_bucket")
    # Step through the active portal. The cell is the `activate_portal` centre (`end_portal_cell`),
    # or the stronghold frame itself as a fallback; the supply gate (16 food + 32 arrows) and the
    # respawn guard are both already met by the stronghold phase, so no low-supplies override.
    if gs.end_portal_cell is not None or gs.stronghold_frame is not None:
        out.append("enter_the_end")
    return out


def _end_combat_candidates(gs: GameState) -> list[str]:
    """The fight itself, in the End: heal, recover a death's drops, crystals first, then the dragon."""
    out: list[str] = []
    # Heal below the floor: a meal is the only buffer, so it comes before any attack.
    if _end_should_eat(gs):
        out.append("eat_food")
    # A death in the End dropped the whole pack on the island; recover it before fighting on. The
    # shared `recover_drops` resolver (`pick_up_items recover_death_items`) carries its own gate
    # `_can_recover`, which already restricts recovery to a death in the current dimension.
    if _can_recover(gs):
        out.append("recover_drops")
    if _crystals_loaded(gs):
        # Crystals heal the dragon, so they fall first. Keep end stone on hand for the server's
        # melee staircase to a caged crystal: it reports the shortfall before building and re-audits
        # the same layout after an interruption, so a topped-up reserve is never wasted.
        if _end_stone_short(gs) and _end_gather_safe(gs):
            out.append("gather_end_stone")
        out.append("destroy_crystal")
    elif _dragon_alive(gs):
        if _dragon_perched(gs):
            # Sitting: arrow-immune. Strike the head this window; prepare_perch rides along to
            # resume and re-verify the notch if a breath cloud or a hit interrupted it.
            out.append("attack_perch")
            out.append("prepare_perch")
        else:
            # Flying: the bow is the answer. prepare_perch rides along so the staging notch is dug
            # and verified ahead of the next landing rather than scrambled for when it begins.
            out.append("shoot_dragon")
            out.append("prepare_perch")
    else:
        # Nothing loaded, and the dragon not slain (or the goal would have closed): walk to the
        # fountain so the towers, their crystals and the dragon load as entities to attack.
        out.append("position_end")
    return out


def resolve_end(label: str, gs: GameState) -> Action | None:
    """Map an End label to its MCP call, or None if its preconditions fail.

    `planner.resolve` tries this first for the End labels; a label it does not own returns None and
    falls through to the main resolver unchanged (the shared `eat_food`, `fill_bucket`,
    `recover_drops` the End candidates also offer are resolved there, not here).
    """
    inv = gs.inventory
    if label == "carve_pumpkin":
        if not _pumpkin_wearable(gs):
            return None
        return Action(label, "equip",
                      {"items": [{"item_name": CARVED_PUMPKIN, "destination": "head"}]},
                      "wear the carved pumpkin so looking at an enderman never provokes it")
    if label == "enter_the_end":
        cell = gs.end_portal_cell or gs.stronghold_frame
        if cell is None:
            return None
        x, y, z = cell
        return Action(label, "enter_end_portal",
                      {"x": int(x), "y": int(y), "z": int(z)},
                      "step into the active end portal to travel to the End")
    if label == "position_end":
        if not _in_end(gs):
            return None
        x, y, z = END_FOUNTAIN_CELL
        return Action(label, "navigate",
                      {"x": x, "y": y, "z": z, "dig": True, "scaffold": True},
                      "walk to the central fountain so the fight entities load")
    if label == "gather_end_stone":
        if not _in_end(gs) or not _end_stone_short(gs):
            return None
        carried = int(inv.get(END_STONE, 0))
        return Action(label, "collect_block",
                      {"block_name": END_STONE, "count": END_STONE_RESERVE - carried},
                      f"mine end stone for the crystal staircases ({carried}/{END_STONE_RESERVE})")
    if label == "destroy_crystal":
        if not _crystals_loaded(gs):
            return None
        entity_id = gs.end_crystals[0][0]
        return Action(label, "destroy_end_crystal",
                      {"entity_id": entity_id, "weapon": "auto", "approach": "staircase"},
                      "destroy a loaded end crystal before it can heal the dragon")
    if label == "shoot_dragon":
        if not _dragon_alive(gs) or _dragon_perched(gs) or _crystals_loaded(gs):
            return None
        return Action(label, "shoot_dragon",
                      {"entity_id": gs.dragon_id, "hitbox_margin": 0.5},
                      "fire a full-charge bow shot at the flying dragon")
    if label == "prepare_perch":
        if not _dragon_alive(gs) or _crystals_loaded(gs):
            return None
        return Action(label, "prepare_dragon_perch",
                      {"entity_id": gs.dragon_id},
                      "build a low staging notch beside the fountain for the next landing")
    if label == "attack_perch":
        if not _dragon_perched(gs) or _crystals_loaded(gs):
            return None
        return Action(label, "attack_dragon_perch",
                      {"entity_id": gs.dragon_id},
                      "strike the perched dragon's head, then withdraw before takeoff")
    return None


# Re-export the whole module namespace (public and private) so a downstream
# `from .end import *` carries every constant and helper by its original name.
__all__ = [_name for _name in list(globals()) if not _name.startswith("__")]
