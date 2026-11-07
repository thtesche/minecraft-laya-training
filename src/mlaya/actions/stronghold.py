from __future__ import annotations

from .predicates import *  # noqa: F401,F403

# --- The stronghold phase (the step after the Nether haul) ---
# Design: docs/stronghold.md. The server owns the hard mechanics (throwing an eye, triangulating,
# travelling, filling sockets); the agent only procures the eyes, filters the valid steps and
# books the calls, exactly as the Nether phase does. The phase stops at an *activated* end portal:
# entering the End and the dragon fight are a later phase the server already carries the tools for.

ENDERMAN_MOB = "enderman"
# The frame block the perception scan finds once the bot is near the stronghold; `activate_portal`
# is given any one of its cells and fills the whole frame.
END_PORTAL_FRAME = "end_portal_frame"
# How close `travel_to_stronghold` has to get before the fill can start. The frame is a solid block
# the bot cannot stand in, so the navigate goal is a distance rather than a cell; three blocks puts
# the body in the frame's neighbourhood - inside the 64 m scan, and inside the server's own approach
# range for `activate_portal`.
STRONGHOLD_ARRIVAL_RANGE = 3
# The eye/pearl item names and the counts (`ENDER_EYES`, `END_PORTAL_SOCKETS`, `EYE_THROW_BUFFER`)
# live in `constants`: the frame has twelve sockets, `activate_portal` needs a carried eye for
# every one not already generated filled, and the extra four are the throw margin - the estimate
# spends an eye and about one throw in five shatters. The Nether gates read the same numbers for
# the top-up predicate (`predicates.nether_expedition_open`).

# An enderman teleports and hits hard, and the hunt happens in the Overworld where the run would
# otherwise sleep. Same safe-hunt shape as the blaze: one target in reach, the bot healthy, no
# other hostile crowding the fight. The eyes are not worth the diamond kit.
# 24 -> 32: the march's own `hostiles` read names endermen out to ~140 m, so the bot kept walking
# past fightable ones. run168 it105-108 held an enderman at 24.5 and 23.9 m while the offer was
# `drop_junk`/`leave_nether`, and it161 stood 5.5 m from a warped forest with one at 28.1 m - just
# outside this gate - and turned home. Eight metres of extra reach is one second of walking, and
# `collect_mob_drop` navigates to the mob itself, so the hunt starts from the same place either way.
ENDERMAN_HUNT_RANGE = 32.0
ENDERMAN_HUNT_HEALTH = 14.0

# The labels this module owns; `planner.resolve` hands each of them to `resolve_stronghold`.
STRONGHOLD_LABELS = frozenset({
    "hunt_enderman", "search_enderman", "gather_scaffold", "take_blaze_powder",
    "craft_ender_eye", "locate_stronghold", "travel_to_stronghold", "activate_end_portal",
    "craft_bed_stronghold", "place_bed_stronghold", "sleep_stronghold",
})


def _ender_pearls(inv: dict[str, int]) -> int:
    """Ender pearls carried in the pack."""
    return int(inv.get(ENDER_PEARL_ITEM, 0))


def _ender_eyes(inv: dict[str, int]) -> int:
    """Eyes of ender carried in the pack."""
    return int(inv.get(ENDER_EYE_ITEM, 0))


def _enderman_in_reach(gs: GameState) -> bool:
    """Whether an enderman stands within the hunt range, blocks and health aside.

    The reach is what the roam steers by: an enderman this close is one to deal with here rather
    than march past, whether or not the fight can be *started* this tick. `_search_enderman_wanted`
    reads this (not `_safe_enderman_hunt`) so that a bot short on building blocks does not roam off
    and leave a loaded enderman behind - it comes home to restock instead.
    """
    ender = gs.mob_nearest(ENDERMAN_MOB)
    return ender is not None and ender <= ENDERMAN_HUNT_RANGE


def _safe_enderman_hunt(gs: GameState) -> bool:
    """Whether an enderman is a fight worth starting right now.

    One enderman in reach, the bot healthy, and the building blocks the fight needs in the pack.
    The "no other hostile crowding the fight" gate the blaze hunt keeps is dropped here on purpose:
    endermen spawn at night alongside the usual mob pack, so that gate almost never clears in the
    dark and the pearl hunt would stall at the base to the run limit (run109 it312-599). The
    diamond kit and shield are expected to carry the fight even with zombies or skeletons around -
    a crowded hunt is the price of the pearls.

    The block floor is the one piece that is not optional: `collect_mob_drop` roofs the bot over as
    the enderman teleports and refuses with `[COMBAT_BUILD_MATERIALS_MISSING]` when the scaffold
    runs short (run112d it240-243: five netherrack left, roof wanted six, four refused ticks). Short
    of `ENDERMAN_HUNT_BLOCKS` the hunt sits out, and the wave mines netherrack where it stands
    (`gather_scaffold`) rather than walking home for cobblestone.
    """
    if not _enderman_in_reach(gs) or (gs.health or 0) < ENDERMAN_HUNT_HEALTH:
        return False
    return _combat_blocks_carried(gs.inventory) >= ENDERMAN_HUNT_BLOCKS


def _search_enderman_wanted(gs: GameState) -> bool:
    """Whether to roam looking for an enderman: pearls still short, none in reach, bot healthy.

    `hunt_enderman` is reactive, the way the blaze hunt waits on a loaded fortress - it only
    fires once an enderman is already within `ENDERMAN_HUNT_RANGE`. Nothing sent the bot out to
    meet one, so a bot that banked its powder sat at the base on `wait` to the run limit, no
    enderman ever wandering the twenty-four metres into range (run109 it312-599: the stronghold
    phase was open, `stronghold_candidates` empty, only `wait` on offer). This roams new ground
    until an enderman loads, then `_safe_enderman_hunt` turns the tick over to `hunt_enderman`.
    Held to a healthy bot so a wounded one heals rather than wandering off after a teleporting mob.

    The in-reach exclusion reads `_enderman_in_reach`, not `_safe_enderman_hunt`: an enderman is
    already loaded, so roaming past it to cover more ground is wrong even when the fight cannot
    start for want of building blocks. The planner then falls to `gather_scaffold` (`_scaffold_short`)
    and the bot digs the roof it is short of rather than abandoning a mob it walked this way to find.

    The roam happens in two places, and the dimension picks which. In the *Nether*, during a pearl
    wave, it is the warped-forest search the waves depend on - endermen live there, not at the
    fortress the rod trip targeted - bounded by `NETHER_PEARL_SEARCH_M` of ground covered so an
    empty stretch sends the bot home to stow and relaunch rather than burn the trip (the pearl
    twin of `_nether_hopeless`'s `NETHER_SEARCH_M`). In the *Overworld*, it is only the fallback
    once the pearl waves are spent (`pearl_wave_trips >= PEARL_WAVE_MAX_TRIPS`): before that the
    pearls come from the Nether, and an Overworld roam here is the off-base drift that laundered
    Laya's "walk away from the full base" picks into verifier positives (the reason this path was
    moved into the Nether at all).
    """
    inv = gs.inventory
    if (_ender_pearls(inv) >= _eyes_owed(gs) or _enderman_in_reach(gs)
            or (gs.health or 0) < ENDERMAN_HUNT_HEALTH):
        return False
    if _in_nether(gs):
        # A stopped march leaves the body where it stood, so the metre budget cannot fill: the
        # stall count is what ends a wave that can find no route at all (`PEARL_MARCH_STALL_LIMIT`).
        return (gs.nether_travel < NETHER_PEARL_SEARCH_M
                and int(getattr(gs, "explore_stalls", 0) or 0) < PEARL_MARCH_STALL_LIMIT)
    return gs.pearl_wave_trips >= PEARL_WAVE_MAX_TRIPS


def _scaffold_short(gs: GameState) -> bool:
    """Whether a pearl wave should mine its own scaffold blocks rather than walk home for them.

    Two things spend them. The fight does: `collect_mob_drop` rebuilds the defensive roof every
    time the enderman teleports (run169 it121-122 spent 68 and then 60 cobblestone on two hunts),
    and the wave crosses with `NETHER_COBBLE` (64), so the second hunt is already on the floor
    `_safe_enderman_hunt` needs. Travel does too: `navigate` and `explore_frontier` place them as
    `scaffold`, so a wave that spent the stack on a lava crossing cannot path anywhere - run169
    it187 bridged with all 122 it carried, and it188-196 then stood nine ticks at the same cell
    with `[NAVIGATION_STOPPED] search timed out after 2000 m`, 6 pearls in the pack and no way
    home. Netherrack is the Nether's own floor: `collect_block` digs the closest reachable matches
    without a coordinate, netherrack roofs a fight as readily as cobblestone and bridges as
    readily again. So the restock happens where the wave stands.

    Every pearl-wave tick is travelling until the pearls are in hand - the roof's need and the
    bridge's need are the same shortage, and a wave that comes home for cobblestone risks the
    pearls it already won on a 300 m walk each way. Once the pearls are met the trip is over and
    the wave leaves with whatever it has (an urgent exit already outranks this, see the ladder).
    """
    if not _in_nether(gs) or (gs.health or 0) < ENDERMAN_HUNT_HEALTH:
        return False
    if _combat_blocks_carried(gs.inventory) >= ENDERMAN_HUNT_BLOCKS:
        return False
    if not _can_mine(gs, SCAFFOLD_BLOCK):
        return False
    return _ender_pearls(gs.inventory) < _eyes_owed(gs)


def _frame_distance(gs: GameState) -> float | None:
    """How far the end portal frame is, in blocks, or None when the loop has no address for it.

    The scan's own number first; outside `RESOURCE_RADIUS` the scan has no hit, and the coordinates
    the locate phase confirmed are the only address the run holds. A locate "succeeds" the moment a
    frame chunk loads (run173 it75: confirmed at (-269,-36,-1819) with the bot 108 m off, 98 blocks
    above it), so the confirmed cell can be far from the body and still be the frame's position.
    """
    distance = gs.resource_distance(END_PORTAL_FRAME)
    if distance is not None:
        return distance
    if gs.pos is None or gs.stronghold_frame is None:
        return None
    x, y, z = gs.pos
    fx, fy, fz = gs.stronghold_frame
    return math.hypot(x - fx, y - fy, z - fz)


def _at_stronghold(gs: GameState) -> bool:
    """Whether the bot stands close enough to the frame to fill it.

    Read off the world, not a latch: `end_portal_frame` is in the Overworld scan group, so the frame
    appears in `gs.resources` the moment the body is within `RESOURCE_RADIUS` of it, and the
    confirmed cell gives the same answer once the travel step has walked the bot in. The
    `resource_found` half of this test used to be enough on its own, and it is what broke the hand
    from `locate_stronghold` to `activate_end_portal`: a frame loads in a chunk a hundred metres
    away, the label came onto the offer, resolved to nothing, and the tick went to `eat_food`
    instead (run173 it72, it76).
    """
    distance = _frame_distance(gs)
    return distance is not None and distance <= STRONGHOLD_ARRIVAL_RANGE


def _frame_cell(gs: GameState) -> tuple[int, int, int] | None:
    """The nearest end portal frame block, the cell `activate_portal` is pointed at."""
    hits = gs.resources.get(END_PORTAL_FRAME) or []
    if hits:
        hit = hits[0]
        return (int(hit.x), int(hit.y), int(hit.z))
    return gs.stronghold_frame


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


def _stronghold_phase(gs: GameState) -> bool:
    """The run owns the stronghold: a frame address is known and the only work left is at it.

    Two windows wear the same suppression. Before the fill: eyes in hand, portal not yet active,
    walking the ~1300 blocks out to fill the frame. After it fires: the run still owes a bed beside
    the frame and the sleep that pins the respawn there, and a death that respawns the bot at base
    strands it the same ~1300 blocks off (live, run184: two deaths after the 10:40 fill, then it
    idled at base with no path back), so it must walk back rather than settle at the old base spawn.

    From here to the goal nothing is worth walking home for. The frameless steps that leak onto the
    offer in this window - `go_home` (night plus the bed at base), `mine_stone` for cobblestone,
    `stow_surplus` back to a chest - are pure detours that pull the bot the wrong way down the trek,
    and at night `go_home` outranks the travel step outright (live, run176: the very first action was
    the home walk). It mirrors `stronghold_candidates`: a frame distance known and either eyes to
    fill the frame or a bed still owed beside it, which is exactly when that function offers the trek.
    """
    if not _overworld(gs) or _frame_distance(gs) is None:
        return False
    if stronghold_complete(gs):
        # The portal stands; the goal is the bed beside the frame and the sleep that pins the
        # spawn - both, not the sleep alone. A bed taken up after the sleep leaves the respawn
        # unpinned (live, run186 it10-13: `take_bed` hauled the frame bed into the pack and
        # `go_home` followed), so the phase stays open until the bed stands again and is slept
        # in, exactly as `_goal_reached` counts it.
        return not (getattr(gs, "slept_at_stronghold", False) and _bed_at_stronghold(gs))
    return nether_expedition_complete(gs) and _ender_eyes(gs.inventory) >= _eyes_in_hand(gs)


def _striding_to_stronghold(gs: GameState) -> bool:
    """The pure march to a confirmed frame: in the stronghold phase, body not yet at it.

    In this window the run only walks and digs toward the frame - to fill it, or, once the portal
    is lit, to lay the respawn bed beside it - so a full pack drops its clutter on the ground
    (`_march_shed`) instead of walking to a chest. `_stronghold_phase` carries the rest of the
    suppression (the offer-level blocks in `planner`), which also covers standing at the frame.
    """
    return _stronghold_phase(gs) and not _at_stronghold(gs)


def _march_junk(name: str) -> bool:
    """Whether a block is clutter the stronghold march drops on the way (beyond `JUNK_ITEMS`).

    The families (`JUNK_SUFFIXES`) are junk everywhere now; they stay here so the march shed still
    names them, and `resolve("drop_junk")` dedupes the two lists before the call.
    """
    return (name in STRONGHOLD_MARCH_JUNK
            or any(name.endswith(suffix) for suffix in JUNK_SUFFIXES))


def _march_shed(gs: GameState) -> list[str]:
    """Carried march clutter to drop, in inventory order. Empty off the march, so the wider shed
    list only ever applies while the bot is trekking to the frame and never returning to base."""
    if not _striding_to_stronghold(gs):
        return []
    inv = gs.inventory
    return [n for n, cnt in inv.items() if cnt and n not in JUNK_ITEMS and _march_junk(n)]


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

    # After the portal fires: lay a bed beside the stronghold frame and sleep once to pin the
    # respawn there instead of the old base. That is the new end of the run. On the happy path the
    # bot still stands at the frame it just lit, but a death respawns it at base ~1300 blocks off
    # (live, run184: two deaths after the 10:40 fill, then it idled at base with no path back). So
    # this is a trek like the pre-fill one: secure a bed's wool, walk to the frame, lay it, sleep.
    if gs.end_portal_active:
        # The bed already stands beside the frame: one sleep left to pin the respawn (a daytime
        # place may have pinned it already, hence the `slept_at_stronghold` guard). Checked first,
        # not "is a bed carried", because `place_bed_stronghold` spends the bed out of the pack as
        # it lays it, so a has-bed-first test would read the just-placed bed as gone and craft a
        # second one instead of reaching the sleep.
        if _bed_at_stronghold(gs):
            if getattr(gs, "slept_at_stronghold", False):
                return out
            # The sleep that pins the respawn only runs at night (and refuses with a hostile
            # close, or mid-sleep), so the offer is gated on exactly what its resolve needs - the
            # offer-on-the-bed-alone it used to be resolved to nothing every daytime tick and
            # escalated. By day the one step left is still that sleep, so the bot must not wander
            # off the frame: `wait` holds it on the bed's own cell until night (the `wait` resolve
            # carries the stronghold hold). `_sleep_safe` mirrors the resolve; `go_home`/`explore`
            # are shut for this phase, which is what would otherwise march the body off.
            if gs.phase == "night" and not gs.sleeping and _sleep_safe(gs):
                out.append("sleep_stronghold")
            else:
                out.append("wait")
            return out
        # No bed and no wool to make one: gather it before the trek, the same gate the pre-fill
        # phase keeps ahead of travel - a bed at the base is the old spawn being left, so the
        # carried-materials test, not `_bed_available`. With neither a sheep nor a spider in reach,
        # an empty `out` falls through to `explore` to roam for one.
        if not _has_bed(inv) and not _bed_materials_ready(gs):
            if gs.mob_nearest("sheep") is not None and _hunt_ok(gs):
                out.append("hunt_wool")
            elif _safe_spider_hunt(gs):
                out.append("hunt_string")
            return out
        # Bed or its materials in hand but the body is not at the frame (a death put it at base):
        # walk back before the precise bedside navigate, dropping a full pack's march clutter first
        # just as the pre-fill travel does. `place_bed_stronghold`'s stance only resolves within
        # reach of the frame, so the trek has to close the distance first.
        if not _at_stronghold(gs):
            if _frame_distance(gs) is not None:
                if gs.slots_used >= INVENTORY_NEAR_FULL and (_droppable(gs, inv) or _march_shed(gs)):
                    out.append("drop_junk")
                out.append("travel_to_stronghold")
            return out
        # Standing at the frame: lay the carried bed, or craft one from the carried wool.
        if _has_bed(inv):
            out.append("place_bed_stronghold")
        elif _bed_materials_ready(gs) and _has(inv, "crafting_table"):
            # The craft builds its workstation from a carried table (`temporary_workstation`), so
            # its resolve refuses with no table in the pack - the offer used to appear on the wool
            # alone and escalated. Gated on the table too, so a tableless pack drops this offer and
            # the tech-tree chain (`craft_table`, else `get_wood`, both already on the planner's
            # list in this phase) builds one first instead of a dead offer sitting unresolved.
            out.append("craft_bed_stronghold")
        return out

    # Before the first (estimate) throw the buffer is built on top of the frame's empty sockets;
    # once the estimate is saved, only the sockets are owed, so an eye shattered on that throw does
    # not send the bot back to re-craft - the locate travel proceeds on that floor.
    locate_need = _eyes_in_hand(gs)
    if _at_stronghold(gs) and not gs.end_portal_active:
        # Standing at the frame. With eyes in hand, fill it and nothing else. With none left -
        # the frame took more eyes than the bot carried in - do not re-offer an activate it cannot
        # complete: fall through to procurement so the bot gathers more, and `locate` (durable
        # search_id) carries it back to finish the frame. Each pass fills >= 1 of the twelve
        # sockets and the fill persists, so the activate <-> procurement loop is self-terminating.
        if _ender_eyes(inv) > 0:
            out.append("activate_end_portal")
            return out
    # Not reached at the stronghold itself: the block above already matched there (with eyes it
    # returned, with none it fell through to procurement below), so nothing is re-offered on the
    # frame the bot is standing at.
    elif _ender_eyes(inv) >= locate_need:
        # The eyes are crafted, but the trek pins the run's new spawn with a bed beside the frame,
        # so the means to set it has to ride along before walking ~1300 blocks out. A base bed is
        # the old spawn the run is leaving, so the general bed chain (gated on *any* bed being
        # available) stays shut here and this gathers a fresh bed's wool itself. Offered ahead of
        # the locate/travel, which `resolve_stronghold` refuses until a bed or its materials are
        # carried; with neither, roam (empty `out` -> `explore`) until a sheep or spider turns up.
        if not _has_bed(inv) and not _bed_materials_ready(gs):
            if gs.mob_nearest("sheep") is not None and _hunt_ok(gs):
                out.append("hunt_wool")
            elif _safe_spider_hunt(gs):
                out.append("hunt_string")
            return out
        # Spend the carried materials on the End reserve before the one-way trek. Only the
        # deterministic crafts are listed here (`_stronghold_supply_steps`), so this cannot stall
        # the march on a chicken, a gravel patch or a skeleton that never loads.
        out.extend(_stronghold_supply_steps(gs))
        # A frame the locate phase confirmed but the body never reached has to be walked to: the
        # survey stops the moment a frame chunk loads, so re-sending `locate` answers in one tick
        # and moves nobody (run173 it75 and it79: two identical successes, the bot 108 m from the
        # frame and 98 blocks above it).
        if _frame_distance(gs) is not None:
            # A full pack cannot pick up what the descent digs, and the march never doubles back to
            # base, so the relief is a drop on the ground here rather than the `stow_surplus` the
            # home-storage block would offer (suppressed for this phase in `planner`). Offered ahead
            # of the walk so the pack clears first; the march block below sheds the wider clutter
            # list, which is why the offer gate reads `_march_shed` as well as `_droppable`.
            if gs.slots_used >= INVENTORY_NEAR_FULL and (_droppable(gs, inv) or _march_shed(gs)):
                out.append("drop_junk")
            out.append("travel_to_stronghold")
            return out
        # No frame the loop knows the position of yet: throw the eyes and let the server find it.
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
        # No dimension gate: the reactive hunt is good wherever an enderman is in reach - the
        # Overworld, and now the Nether during a pearl wave (`planner`'s pearl ladder offers it
        # there). `_safe_enderman_hunt` carries the whole precondition (one in reach, bot healthy).
        if not _safe_enderman_hunt(gs):
            return None
        want = _eyes_owed(gs) - _ender_pearls(inv)
        return Action(label, "collect_mob_drop",
                      {"mob_name": ENDERMAN_MOB, "drop_name": ENDER_PEARL_ITEM,
                       "count": min(3, want), "allow_without_shield": True},
                      f"kill an enderman for its ender pearl ({want} still needed)")
    if label == "gather_scaffold":
        if not _scaffold_short(gs):
            return None
        # The wave's own restock: netherrack underfoot, no coordinate, one call's maximum. The
        # label stays on offer while the reserve is still under `ENDERMAN_HUNT_BLOCKS`, so a wave
        # that spent everything simply digs again; the hunt takes the next tick once it is stocked.
        carried = _combat_blocks_carried(inv)
        return Action(label, "collect_block",
                      {"block_name": SCAFFOLD_BLOCK, "count": SCAFFOLD_MINE_BATCH},
                      f"mine {SCAFFOLD_BLOCK} for the roof and the bridge "
                      f"({carried} carried, {ENDERMAN_HUNT_BLOCKS} wanted)")
    if label == "search_enderman":
        if not _search_enderman_wanted(gs):
            return None
        want = _eyes_owed(gs) - _ender_pearls(inv)
        # Endermen fill the warped forest - it is the one Nether biome whose hostile spawn is the
        # enderman itself - so a forest this world has already recorded is the place to walk to.
        # Without one, the march heads for ground nobody has observed (`gs.frontier_heading`, the
        # server's own bearing to the nearest frontier chunk), which sweeps a new sector as the
        # frontier moves. It used to take `frontier_heading(gs)` - the away-from-base bearing -
        # which is the same compass line on every trip, because the base does not move: run164
        # walked three pearl waves down the same crimson corridor and came home with no pearl.
        if _in_nether(gs):
            base = (gs.warped_forest[0] if gs.warped_forest is not None
                    else _nether_search_heading(gs))
            heading = _clear_march_heading(gs, base)
            # A recorded forest is a destination, not a bearing to pass by. `explore_frontier`
            # takes an optional `biome` and only succeeds once that biome is observed at the
            # grounded bot's feet, so the march stops *in* the forest instead of walking a ray
            # through it and out the far side - which is what the plain heading did: run168 mapped
            # 192 warped chunks, stood in one on 2 of ~120 Nether ticks, and came home from four
            # trips with no pearl. The tool stops the leg early on the first entry, so a forest
            # sighted 30 m off the line still counts.
            #
            # Inside `WARPED_ARRIVAL_M` the biome target cannot be used: the call would succeed on
            # its own first check ("including at the start") and re-offer on the next tick without
            # the body moving. There the march keeps short legs on the same bearing, sweeping the
            # forest it already stands in.
            if gs.warped_forest is not None and gs.warped_forest[1] > WARPED_ARRIVAL_M:
                return Action(label, "explore_frontier",
                              {"heading": heading, "chunks": PEARL_MARCH_CHUNKS,
                               "biome": "warped_forest"},
                              f"march into the recorded warped forest for endermen "
                              f"({want} pearls still needed)")
            return Action(label, "explore_frontier",
                          {"heading": heading, "chunks": 1},
                          f"sweep the warped forest already underfoot ({want} pearls needed)")
        heading = _clear_march_heading(gs, frontier_heading(gs))
        # No fortress-style landmark to steer by - endermen are just mobs that load in the dark -
        # so march the remembered frontier into new terrain, the same `explore_frontier` roam the
        # Overworld explore falls back to. Each march loads fresh ground; once an enderman comes
        # within reach `hunt_enderman` takes over on the next tick.
        return Action(label, "explore_frontier",
                      {"heading": heading, "chunks": EXPLORE_FRONTIER_CHUNKS},
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
        need = _eyes_in_hand(gs)
        if not _overworld(gs) or _ender_eyes(inv) < need or _at_stronghold(gs):
            return None
        # Before the stronghold trek, the bot must carry the means to set a fresh respawn at the
        # frame: either a finished bed, or the wool and wood to craft one there. Without either
        # there is nothing to pin the new spawn and the run cannot finish at the stronghold, so the
        # trek waits while `stronghold_candidates` gathers the wool. A bed already standing at the
        # base does not count - that is the old spawn the run is leaving - which is why the gate
        # reads the carried bed, not `_bed_available`.
        if not _has_bed(inv) and not _bed_materials_ready(gs):
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
    if label == "travel_to_stronghold":
        # Reopened after activation: the post-fill phase walks the bot back to the frame to lay the
        # respawn bed when a death left it at base. `_at_stronghold` still closes the step - the
        # frame reached, there is nothing left to walk to.
        if _at_stronghold(gs):
            return None
        cell = gs.stronghold_frame
        if cell is None:
            return None
        x, y, z = cell
        why = (f"walk back to the end portal frame at {x},{y},{z} to lay the respawn bed beside it"
               if gs.end_portal_active
               else f"walk to the confirmed end portal frame at {x},{y},{z} to fill its sockets")
        return Action(label, "navigate", {"x": x, "y": y, "z": z,
                                         "range": STRONGHOLD_ARRIVAL_RANGE}, why)
    if label == "activate_end_portal":
        if not _at_stronghold(gs) or gs.end_portal_active:
            return None
        cell = _frame_cell(gs)
        if cell is None:
            return None
        x, y, z = cell
        return Action(label, "activate_portal", {"x": x, "y": y, "z": z},
                      "fill the end portal frame's empty sockets with the carried eyes of ender")
    if label == "craft_bed_stronghold":
        if not _overworld(gs):
            return None
        if not _has(inv, "crafting_table") or not _bed_materials_ready(gs):
            return None
        bed_item = _bed_item_to_craft(inv)
        if bed_item is None:
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": bed_item}], "temporary_workstation": True},
                      f"craft a {bed_item} for the stronghold respawn point")
    if label == "place_bed_stronghold":
        if not _overworld(gs) or not _has_bed(inv) or _bed_at_stronghold(gs):
            return None
        cell = _bed_stronghold_stance(gs)
        if cell is None:
            return None
        sx, sy, sz = cell
        if not _at_stance(gs, cell):
            return Action(label, "navigate",
                          {"x": sx, "y": sy, "z": sz, "range": 0,
                           "dig": True, "scaffold": True},
                          "walk to a clear spot beside the stronghold frame for the bed")
        return Action(label, "sleep", {},
                      "place the bed beside the stronghold frame and set the respawn point")
    if label == "sleep_stronghold":
        if gs.phase != "night" or gs.sleeping or not _overworld(gs) or not _sleep_safe(gs):
            return None
        if not _bed_at_stronghold(gs):
            return None
        return Action(label, "sleep", {},
                      "sleep at the stronghold to pin the respawn point there")
    return None


# Re-export the whole module namespace (public and private) so a downstream
# `from .stronghold import *` carries every constant and helper by its original name.
__all__ = [_name for _name in list(globals()) if not _name.startswith("__")]
