from __future__ import annotations

from .predicates import *  # noqa: F401,F403

# --- The stronghold phase (the step after the Nether haul) ---
# Design: docs/stronghold.md. The server owns the hard mechanics (throwing an eye, triangulating,
# travelling, filling sockets); the agent only procures the eyes, filters the valid steps and
# books the calls, exactly as the Nether phase does. The phase stops at an *activated* end portal:
# entering the End and the dragon fight are a later phase the server already carries the tools for.

ENDERMAN_MOB = "enderman"
# The frame block the perception scan finds once the locate phase has stood the bot at the
# stronghold; `activate_portal` is given any one of its cells and fills the whole frame.
END_PORTAL_FRAME = "end_portal_frame"
# The eye/pearl item names and the `ENDER_EYES` / `ENDER_PEARLS` counts live in `constants` -
# the Nether gates read them for the top-up predicate (`predicates.nether_expedition_open`).

# An enderman teleports and hits hard, and the hunt happens in the Overworld where the run would
# otherwise sleep. Same safe-hunt shape as the blaze: one target in reach, the bot healthy, no
# other hostile crowding the fight. The eyes are not worth the diamond kit.
ENDERMAN_HUNT_RANGE = 24.0
ENDERMAN_HUNT_HEALTH = 14.0

# The end portal frame has twelve sockets; `activate_portal` needs a carried eye for every one not
# already generated filled. Twelve is the floor the locate travel has to keep in hand, while
# `ENDER_EYES` (sixteen) is the build target - the extra four a buffer for eyes shattered on a
# throw. Once the estimate throw is behind the bot, a shaved eye must not drag it back to sixteen.
END_PORTAL_SOCKETS = 12

# The labels this module owns; `planner.resolve` hands each of them to `resolve_stronghold`.
STRONGHOLD_LABELS = frozenset({
    "hunt_enderman", "search_enderman", "take_blaze_powder", "craft_ender_eye",
    "locate_stronghold", "activate_end_portal",
})


def _ender_pearls(inv: dict[str, int]) -> int:
    """Ender pearls carried in the pack."""
    return int(inv.get(ENDER_PEARL_ITEM, 0))


def _ender_eyes(inv: dict[str, int]) -> int:
    """Eyes of ender carried in the pack."""
    return int(inv.get(ENDER_EYE_ITEM, 0))


def _eyes_owed(gs: GameState) -> int:
    """How many more eyes of ender the phase still has to craft."""
    return max(0, ENDER_EYES - _ender_eyes(gs.inventory))


def _safe_enderman_hunt(gs: GameState) -> bool:
    """Whether an enderman is a fight worth starting right now.

    One enderman in reach and the bot healthy. The "no other hostile crowding the fight" gate
    the blaze hunt keeps is dropped here on purpose: endermen spawn at night alongside the usual
    mob pack, so that gate almost never clears in the dark and the pearl hunt would stall at the
    base to the run limit (run109 it312-599). The diamond kit and shield are expected to carry
    the fight even with zombies or skeletons around - a crowded hunt is the price of the pearls.
    """
    ender = gs.mob_nearest(ENDERMAN_MOB)
    if ender is None or ender > ENDERMAN_HUNT_RANGE or (gs.health or 0) < ENDERMAN_HUNT_HEALTH:
        return False
    return True


def _search_enderman_wanted(gs: GameState) -> bool:
    """Whether to roam looking for an enderman: pearls still short, none in reach, bot healthy.

    `hunt_enderman` is reactive, the way the blaze hunt waits on a loaded fortress - it only
    fires once an enderman is already within `ENDERMAN_HUNT_RANGE`. Nothing sent the bot out to
    meet one, so a bot that banked its powder sat at the base on `wait` to the run limit, no
    enderman ever wandering the twenty-four metres into range (run109 it312-599: the stronghold
    phase was open, `stronghold_candidates` empty, only `wait` on offer). This is the overworld
    mirror of the Nether fortress march: roam new ground until an enderman loads, then
    `_safe_enderman_hunt` turns the tick over to `hunt_enderman`. Held to a healthy bot so a
    wounded one heals at the base rather than wandering off after a teleporting mob.
    """
    return (_overworld(gs)
            and _ender_pearls(gs.inventory) < _eyes_owed(gs)
            and not _safe_enderman_hunt(gs)
            and (gs.health or 0) >= ENDERMAN_HUNT_HEALTH)


def _at_stronghold(gs: GameState) -> bool:
    """Whether the bot stands at the stronghold's end portal frame.

    Read off the world, not a latch: the locate phase travels the bot to the frame and the frame
    blocks load into the scan there (`end_portal_frame` is in the Overworld scan group). This is
    what hands the tick from `locate_stronghold` to `activate_end_portal`, the same way a loaded
    `nether_portal` hands `leave_nether` its target - so a cancelled locate that is re-run simply
    resumes, and nothing has to be parsed out of the tool's prose.
    """
    return (gs.resource_distance(END_PORTAL_FRAME) is not None
            or gs.resource_found.get(END_PORTAL_FRAME, 0) > 0)


def _frame_cell(gs: GameState) -> tuple[int, int, int] | None:
    """The nearest end portal frame block, the cell `activate_portal` is pointed at."""
    hits = gs.resources.get(END_PORTAL_FRAME) or []
    if not hits:
        return None
    hit = hits[0]
    return (int(hit.x), int(hit.y), int(hit.z))


def _take_powder_wanted(gs: GameState) -> bool:
    """Whether the banked blaze powder has to come back out of the chest to craft eyes.

    The Nether haul parked the powder in the gate chest (`stow_blaze_powder`); the eyes are
    crafted from it here. Offered while the phase still owes eyes, a pearl is in hand to pair the
    powder with, the carried powder is short of the eyes owed, and the chest has not already been
    emptied of powder (`gs.powder_exhausted`, latched when a withdraw comes up empty).
    """
    inv = gs.inventory
    if _in_nether(gs) or not _chest_ready(gs) or gs.powder_exhausted:
        return False
    return (_eyes_owed(gs) > 0 and _ender_pearls(inv) > 0
            and _blaze_powder(inv) < _eyes_owed(gs))


def _can_craft_ender_eye(gs: GameState) -> bool:
    """Carried a pearl, a powder and a table, with eyes still owed."""
    inv = gs.inventory
    return (not _in_nether(gs) and _eyes_owed(gs) > 0
            and _ender_pearls(inv) > 0 and _blaze_powder(inv) > 0
            and _has(inv, "crafting_table"))


def stronghold_complete(gs: GameState) -> bool:
    """True when the end portal stands activated.

    Latched on `activate_portal`'s own success (`gs.end_portal_active`, `Agent._note_portal`),
    the same shape as `powder_stowed`: reading the world back would cost an inspection every
    tick, and the action's result already confirms the active portal.
    """
    return bool(gs.end_portal_active)


def stronghold_candidates(gs: GameState) -> list[str]:
    """The stronghold steps whose preconditions hold, in the order the bot walks them.

    Offered by `planner.candidates` once the Nether haul stands and the portal is not yet active.
    One chain: gather pearls, bring the banked powder back out, craft the eyes, find and travel to
    the stronghold, fill the frame. Each step only appears once the ones before it are done.
    """
    inv = gs.inventory
    out: list[str] = []
    # The Overworld side only: the hunt, the chest and the crafting all belong up here. A powder
    # top-up runs the bot back through the gate, but that chain is the Nether block's, reopened by
    # `_nether_topup_wanted`, not this one.
    if not _overworld(gs):
        return out
    # Before the first (estimate) throw the buffer is built to ENDER_EYES; once the estimate is
    # saved, the frame needs only END_PORTAL_SOCKETS, so an eye shattered on that throw does not
    # send the bot back to re-craft - the locate travel proceeds on that floor.
    locate_need = END_PORTAL_SOCKETS if gs.stronghold_estimated else ENDER_EYES
    if _at_stronghold(gs) and not gs.end_portal_active:
        # Standing at the frame. With eyes in hand, fill it and nothing else. With none left -
        # the frame took more eyes than the bot carried in - do not re-offer an activate it cannot
        # complete: fall through to procurement so the bot gathers more, and `locate` (durable
        # search_id) carries it back to finish the frame. Each pass fills >= 1 of the twelve
        # sockets and the fill persists, so the activate <-> procurement loop is self-terminating.
        if _ender_eyes(inv) > 0:
            out.append("activate_end_portal")
            return out
    # Not reached at the stronghold itself: the block above has already matched there (with eyes it
    # returned, with none it fell through to procurement below), so `locate` is never re-offered on
    # the frame it just arrived at.
    elif _ender_eyes(inv) >= locate_need:
        # The eyes are crafted; go find the stronghold.
        out.append("locate_stronghold")
        return out
    # Procurement, all offered together while any still has work; the order sets the priority. The
    # pearl target is the eyes still owed, not a flat sixteen: a re-supply for a couple of shattered
    # eyes must not over-hunt a dozen teleporting endermen it does not need.
    if _ender_pearls(inv) < _eyes_owed(gs) and _safe_enderman_hunt(gs):
        out.append("hunt_enderman")
    # No enderman in reach but pearls still owed: roam to find one rather than idle at the base.
    # `_search_enderman_wanted` already excludes the in-reach case, so this and `hunt_enderman`
    # never offer together.
    if _search_enderman_wanted(gs):
        out.append("search_enderman")
    if _take_powder_wanted(gs):
        out.append("take_blaze_powder")
    if _can_craft_ender_eye(gs):
        out.append("craft_ender_eye")
    return out


def resolve_stronghold(label: str, gs: GameState) -> Action | None:
    """Map a stronghold label to its MCP call, or None if its preconditions fail.

    `planner.resolve` tries this first for the stronghold labels; a label it does not own returns
    None and falls through to the main resolver unchanged.
    """
    inv = gs.inventory
    if label == "hunt_enderman":
        if not _overworld(gs) or not _safe_enderman_hunt(gs):
            return None
        want = _eyes_owed(gs) - _ender_pearls(inv)
        return Action(label, "collect_mob_drop",
                      {"mob_name": ENDERMAN_MOB, "drop_name": ENDER_PEARL_ITEM,
                       "count": min(3, want), "allow_without_shield": True},
                      f"kill an enderman for its ender pearl ({want} still needed)")
    if label == "search_enderman":
        if not _search_enderman_wanted(gs):
            return None
        want = _eyes_owed(gs) - _ender_pearls(inv)
        # No fortress-style landmark to steer by - endermen are just mobs that load in the dark -
        # so march the remembered frontier into new terrain, the same `explore_frontier` roam the
        # Overworld explore falls back to. Each march loads fresh ground; once an enderman comes
        # within reach `hunt_enderman` takes over on the next tick.
        return Action(label, "explore_frontier",
                      {"heading": frontier_heading(gs), "chunks": EXPLORE_FRONTIER_CHUNKS},
                      f"roam new ground to find an enderman ({want} pearls still needed)")
    if label == "take_blaze_powder":
        if not _take_powder_wanted(gs):
            return None
        # The powder was stowed into the network's roomy chest (`stow_blaze_powder`), which is
        # not always `chest_pos`: the loop remembers the cells a deposit actually reached
        # (`gs.powder_cells`, `Agent._note_stow`) and the withdraw works those, falling back to
        # the main chest for a state that never recorded a stow. A cell that comes up empty is
        # dropped from the list by `_note_take_powder`, and the top-up only latches once every
        # remembered cell has come up empty. The banked tracking only follows food, not powder,
        # so the exact count is not read: the withdraw asks for what the eyes owe and takes what
        # the chest holds.
        cells = [tuple(c) for c in gs.powder_cells] or [tuple(gs.chest_pos)]
        cell = next((c for c in cells if _near_cell(gs, c)), cells[0])
        if not _near_cell(gs, cell):
            cx, cy, cz = cell
            return Action(label, "navigate", {"x": cx, "y": cy, "z": cz},
                          "walk over to the chest that holds the banked blaze powder")
        want = _eyes_owed(gs) - _blaze_powder(inv)
        cx, cy, cz = cell
        return Action(label, "use_container",
                      {"operation": "withdraw", "x": cx, "y": cy, "z": cz,
                       "items": [{"item_name": BLAZE_POWDER_ITEM, "count": want}]},
                      f"take {want} blaze powder back out of the chest to craft eyes of ender")
    if label == "craft_ender_eye":
        if not _can_craft_ender_eye(gs):
            return None
        want = min(_ender_pearls(inv), _blaze_powder(inv), _eyes_owed(gs))
        if want <= 0:
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": ENDER_EYE_ITEM, "count": want}],
                       "temporary_workstation": True},
                      f"craft {want} eyes of ender, one ender pearl and one blaze powder each")
    if label == "locate_stronghold":
        need = END_PORTAL_SOCKETS if gs.stronghold_estimated else ENDER_EYES
        if not _overworld(gs) or _ender_eyes(inv) < need or _at_stronghold(gs):
            return None
        # Two phases, keyed by a loop latch: `estimate` saves the eye-of-ender triangulation
        # without travelling, then `locate` reuses the same durable `search_id` to travel there
        # and confirm the frame. `continue_without_recommended_items` waves off the dragon-fight
        # supply check - the dragon is a later phase; this one stops at an activated portal.
        phase = "locate" if gs.stronghold_estimated else "estimate"
        why = ("travel to the stronghold the eye-of-ender bearing points to and stand at its frame"
               if phase == "locate"
               else "throw an eye of ender to estimate where the stronghold lies")
        return Action(label, "locate_stronghold",
                      {"phase": phase, "search_id": "stronghold",
                       "continue_without_recommended_items": True}, why)
    if label == "activate_end_portal":
        if not _at_stronghold(gs) or gs.end_portal_active:
            return None
        cell = _frame_cell(gs)
        if cell is None:
            return None
        x, y, z = cell
        return Action(label, "activate_portal", {"x": x, "y": y, "z": z},
                      "fill the end portal frame's empty sockets with the carried eyes of ender")
    return None


# Re-export the whole module namespace (public and private) so a downstream
# `from .stronghold import *` carries every constant and helper by its original name.
__all__ = [_name for _name in list(globals()) if not _name.startswith("__")]
