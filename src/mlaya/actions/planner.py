from __future__ import annotations

from .predicates import *  # noqa: F401,F403
from .stronghold import *  # noqa: F401,F403


def candidates(gs: GameState) -> list[str]:
    """The actions whose preconditions currently hold, in tech-tree priority order.

    Laya only ever chooses among these, so no choice can be a dead end.
    """
    inv = gs.inventory
    out: list[str] = []
    tier = _pick_tier(inv, gs.tools)
    logs = _logs(inv)
    planks = _planks(inv)
    cobble = inv.get("cobblestone", 0)
    table = _has(inv, "crafting_table")
    # Whether a furnace is reachable *now*, not merely remembered. This gates `craft_furnace`
    # (offered only when none is reachable) and must be the same test `smelt_iron` uses to accept
    # one (`_smelting_reach`), so the two can never disagree -- craft a furnace exactly when there
    # is none to smelt at. `_furnace_ready` alone (a bare `furnace_pos is not None`) was the run116
    # deadlock: a bot that died and drifted ~650 m from base still carried the server's memory of
    # the old base furnace, so `furnace` read True, `craft_furnace` was suppressed, yet `smelt_iron`
    # needs `_at_base` (base None out there) -- no furnace, no smelt, every tech step blocked, endless
    # `explore`. Gating on reach lets the drifted bot craft a temporary-workstation furnace instead.
    furnace = _smelting_reach(gs)
    coal = inv.get("coal", 0)
    charcoal = inv.get("charcoal", 0)
    raw_iron = inv.get("raw_iron", 0)
    iron = inv.get("iron_ingot", 0)
    diamond = inv.get("diamond", 0)

    # Eat when hungry (the survival floor) or when a wound cannot heal: natural regeneration
    # needs food >= 18, so a bot parked at 14-17 neither eats (the old floor was 14) nor
    # regenerates, and simply bleeds out to the next hit (live, run14: 17 ticks at 2.7 HP and
    # food 17, then a mob finished it). Feeding to the regeneration threshold is what heals.
    starving = gs.food is not None and gs.food < EAT_HUNGER
    wounded = (gs.health is not None and gs.health < HEALTH_MAX
               and gs.food is not None and gs.food < REGEN_HUNGER)
    eating = (starving or wounded) and bool(_first_edible(inv))

    # --- standing in a lit portal (rescue) ---
    # The pathfinder walks a portal block like any other open gate, so a `navigate` goal on the
    # far side routes the body through it and the teleport fires a few seconds later - no step
    # ever asked for the crossing (live, run22: the lit gate stood beside the base and the bot was
    # in the Nether a few decisions after it was lit). Nothing and nobody intends to be in the
    # gate here, so the rescue owns the tick outright (no Laya choice), exactly like the water
    # rescue. Phase 3, the intended crossing, must bypass this.
    if _in_portal(gs):
        return ["leave_portal"]

    # --- no footing in water (rescue) ---
    # Live failure: the obsidian dig left the bot treading water off the ground. Off-ground
    # breaks the shared pathfinder -- `collect_block` then thrashed for minutes without moving a
    # block -- while `navigate` still works from a floating stance. Nothing else can run until
    # the bot has ground under it again, so the rescue owns the tick outright (no Laya choice).
    # But "nothing else can run" is only true of work that needs the feet planted. A body that is
    # merely between two banks is swimming, which the server navigates on its own, so owning the
    # tick on sight tore the bot out of every crossing the moment an action came back (live,
    # run15). The rescue therefore waits `WATER_RESCUE_AFTER` decisions off the feet, which
    # leaves a goal that crosses water the room to finish, and still takes over a bot that has
    # been adrift for longer than any crossing needs.
    if gs.in_water and not gs.on_ground and gs.water_ticks >= WATER_RESCUE_AFTER:
        return ["escape_water"]

    # --- shut in (rescue) ---
    # Every walk ends `no path found` after a few dozen nodes, the closest one beside the bot:
    # the pathfinder cannot leave the pocket it stands in, so nothing that walks can run. Live,
    # run90: 330 ticks at (182,70,181) and (182,72,180), recover_drops, explore and get_wood
    # refused alike, until the iteration limit. The tunnel owns the tick (`Agent._note_boxed`).
    if gs.boxed_in and gs.pos is not None:
        return ["dig_out"]

    # --- junk dropped on open ground, before it can be picked back up ---
    # When the terrain refused a disposal pit, `drop_junk` drops onto the ground where the bot
    # stands, and dropped items are picked up again by walking within reach of them. The pit
    # path already ends with the server plugging the hole and stepping back; the open drop has
    # no such ending, so it would undo itself on the very next step (live: no run has yet had
    # this offer, so the walk back is untested until one does).
    if gs.drop_open_pending and gs.pos is not None and gs.drop_pos is not None:
        dx, dz = gs.pos[0] - gs.drop_pos[0], gs.pos[2] - gs.drop_pos[2]
        if dx * dx + dz * dz < 3.0 * 3.0:
            return ["step_away_drops"]

    # A meal the body needs now owns the tick (`_must_eat`); wounded with nothing to eat, the
    # tick goes to food or safety, nothing else (`_cannot_heal`).
    if _must_eat(gs) and "eat_food" not in gs.stuck_labels:
        return ["eat_food"]
    if _cannot_heal(gs):
        if _stronghold_phase(gs):
            # Wounded and starving on the stronghold trek: `_recovery_steps` would walk the bot
            # ~1300 blocks home to a base that holds no food either, and the phase gate refuses
            # that `go_home` - which is a dead end, escalated tick after tick (live, run177 it10-11).
            # The only heal is food the trek can reach; with none, the march itself is the move.
            out = ["eat_food"] if eating else []
            if _hunt_ok(gs) and _food_target(gs) is not None:
                out.append("hunt_food")
            elif _mushroom_pantry_wanted(gs):
                out.append("gather_mushrooms")
            out.extend(stronghold_candidates(gs))
            return out
        return _recovery_steps(gs)

    # --- event-driven reactions (M2d) ---
    # A tool about to break, a mob the reflex just killed (its drops despawn in minutes),
    # or gear lying at a fresh death site are all time-sensitive, so the top reaction
    # owns the tick. Eating still rides along: survival outranks everything.
    reactions: list[str] = []
    if _can_recover(gs):
        reactions.append("recover_drops")
    if _pickaxe_due_for_spare(gs) and _replacement_pickaxe(gs, inv):
        reactions.append("replace_tool")
    if gs.drops > 0:
        reactions.append("collect_drops")
    if reactions:
        return (["eat_food"] if eating else []) + reactions[:1]

    if eating:
        out.append("eat_food")
    # A furnace that stands at the base is what stops `smelt_item` placing and tearing a
    # temporary workstation down on every call. In the lull the carried furnace is stood at the
    # base once (walking home first if the bot is away), after which cooking reuses its cell.
    if _furnace_worth_placing(gs):
        # The furnace is stood at the base once, and the walk home for it is the same ping-pong as
        # `go_home` itself: never on the stronghold march (run176 started on exactly that walk).
        if _at_base(gs):
            out.append("place_furnace")
        elif not _stronghold_phase(gs):
            out.append("go_home")
    # The smoker is the same idea one step on: it does not replace the furnace (ore still needs
    # that one) but it doubles the rate for the only thing this run smelts in bulk. Offered at the
    # base only, because the whole point is cooking *at* the base rather than wherever the bot
    # happened to gather the meat.
    if _smoker_worth_placing(gs) and _has(inv, "smoker"):
        out.append("place_smoker")
    if ((_has(inv, "furnace") or _furnace_ready(gs)) and _first_cookable(inv) and not eating
            and _furnace_room(gs) and _food_source_in_reach(gs)):
        # Cook raw meat so it feeds far more hunger. Outside the Nether lull only coal/charcoal
        # may be spent on it (as before, so it never steals wood from the chain); during the lull
        # any carried fuel does, so a bot with raw meat and wood still cooks.
        #
        # The step needs a source in reach, by the same rule as `_food_reserve_met`. Cooking
        # turns one raw item into one cooked one and both count the same in `_edible_count`, so
        # it cannot close the gap it is offered for - and while it is the only thing on offer the
        # exploration fallback never runs, because that fallback only fires on an empty list. The
        # bot stood at its base cooking its way through its stock and never went looking for the
        # thing it was actually short of (live, run34: a bow it could not rebuild, no spider
        # within 24 m, and `candidates: ['cook_food']` with nothing else on it).
        fuel = _furnace_fuel(inv)
        if fuel is not None and _cooking_reach(gs):
            if fuel[0] in ("coal", "charcoal"):
                out.append("cook_food")
            elif _nether_idle(gs) and _provisioned(gs) < _food_goal(gs):
                # Wood is chain material and one unit cooks one item, so it is only worth
                # spending while the reserve is actually short. The lull branch used to cook on
                # any carried fuel regardless: live, run164 it122-131 spent its last two acacia
                # logs on six raw porkchops beside 66 cooked meals, `[SMELT_FUEL_STARVED]`
                # alternating with the one-item successes until the fuel ran out - eight ticks
                # the stall detector flagged (`stalled: cook_food/leave_portal`). Coal is cheap
                # and stays unconditional; wood waits for a reserve that needs it.
                out.append("cook_food")
    # Food reserve: bank edible items. At full hunger the reflex cannot eat, so gathered drops
    # accumulate; when hunger drops it spends them. Never compete with eating: feed to 20 first,
    # bank after. The goal deepens to the Nether target once the gate frame stands.
    if (_provisioned(gs) < _food_goal(gs) and not eating
            and _food_target(gs) is not None
            and _hunt_ok(gs)):
        out.append("hunt_food")
    # No animal to hunt: the stew pantry is the fallback the user asked for - a bowl and one red
    # plus one brown mushroom brew a meal whose bowl comes back, so there is food where there is
    # no herd. Offered only when the hunt has nothing to walk to (`_food_source_in_reach`), so
    # animals keep their priority, and only below the renewable floor, so it cannot brew the day
    # away. Live, run153: 23 edibles and no animal in reach, where `wait` took the rest of the run.
    if (not eating and not _food_source_in_reach(gs)
            and _edible_count(inv) < NETHER_FOOD_RENEWABLE_RESERVE
            and _stew_source_in_reach(gs)):
        out.append("craft_stew")
    # The pantry is stocked before it is brewed. A mushroom walk is the cheapest food in the game:
    # the mushroom does not run away, does not despawn, and regrows on the spot - in the Nether on
    # netherrack, at home on soil. Live, run175: 44 action ticks, no mushroom picked, because
    # `craft_stew` only ever asked for the one mushroom a single stew was short of. The step is
    # offered beside the hunt only when the hunt has nothing to walk to (animals keep their
    # priority) and only while the reserve is short, so it cannot become the tick's whole life;
    # whether to take it is Laya's call.
    if (not eating and not _food_source_in_reach(gs)
            and _provisioned(gs) < _food_goal(gs)
            and _mushroom_pantry_wanted(gs)
            and "gather_mushrooms" not in gs.stuck_labels):
        out.append("gather_mushrooms")
    if (_pack_food_short(gs) and not _in_nether(gs) and _at_base(gs)
            and _chest_ready(gs)):
        out.append("take_stored_food")

    # A bed standing underground holds the respawn point in a cave (`_unsafe_beds`), so it is
    # taken back up before anything else - not left beside `drop_junk` for Laya to pass over, as
    # run87 did from it198 on with the bot at 0.2 health among the zombies of that cave.
    if (_unsafe_beds(gs) and not _has_bed(inv) and _overworld(gs)
            and "take_bed" not in gs.stuck_labels):
        return (["eat_food"] if eating else []) + ["take_bed"]

    bed = _bed_available(gs)
    bed_ready = _bed_leaves_ready(inv)
    # Night + a bed: sleeping skips to morning and pins the respawn point. `_sleep_ready` owns
    # *where* it may run - away from the base before the gate is lit (the server lays the bed
    # "beside the bot", and at the base that spot is the portal's own interior, which the later
    # frame build - or the still-open unlit opening - cannot clear: live, the bed landed in the
    # opening and `light_portal` answered `[PORTAL_NO_FRAME]` on every tick), and at the base on a
    # bed actually standing there, gate burning or not (run158 held 23 night ticks on `wait`
    # because the standing bed only counted once the portal lit). `_bed_in_opening` keeps a bed
    # inside the frame's own opening out of it. `place_bed` owns the base bed, `take_bed` brings a
    # leftover home.
    # The night offer returns ahead of the withholding filter at the end, so it asks for itself:
    # run86 dropped `sleep` at it2 for `[BED_NOT_FOUND]` and the offer took the next 13 ticks
    # anyway, the window never reaching it.
    if (gs.phase == "night" and not gs.sleeping and _overworld(gs) and _sleep_safe(gs)
            and (_sleep_ready(gs) or _sleep_off_base(gs))
            and "sleep" not in gs.stuck_labels):
        # Sleeping is not a step among others, it is the clock. A night costs about forty decisions
        # at eighteen seconds each; a sleep costs one. So at night, with a bed underfoot, nothing
        # else on this list is time-critical: the work will still be waiting at dawn, and every
        # tick spent on it instead is a tick standing in the dark next to mobs the reflex would
        # otherwise have to fight on its own.
        #
        # Live: 462 night ticks, 175 of them sleep-ready, `sleep` offered on 164 - and taken nine
        # times, half of those refused because `hostile_reflex` or `hunger_reflex` already owned
        # the body. The bot chose `craft_arrows` four ticks running with a bed under it.
        #
        # Safe to own the tick outright, because everything time-critical has already returned by
        # the time this line is reached: the portal and water rescues return above it, and the
        # event-driven reactions - recovering drops before they despawn, replacing a pickaxe that
        # is about to break, gear at a fresh death site - return right here, the tick before. What
        # is left on `out` is work with no deadline.
        return (["eat_food"] if eating else []) + ["sleep"]

    # A bed standing away from the base is a leftover the new base did not bring along. Take it
    # up in the lull and lay it again beside the portal: the respawn point belongs at the gate,
    # and a bed left in the base's own dig shaft is what `mine_gold` keeps stumbling on. Offered
    # with the prep, so the bed is cleared before the dig starts rather than mid-tunnel.
    if _nether_idle(gs) and _overworld(gs) and not _has_bed(inv) and _bed_needs_moving(gs):
        out.append("take_bed")

    if not _in_nether(gs):
        # Bootstrap, then crafts/smelts (in tech-tree order), then gathering.
        # The planks each wood craft below needs when no log is carried - a log always suffices,
        # since the craft makes its own planks from it. `need_planks` is the deepest of the crafts
        # still owed, so `get_wood` and the craft it feeds share one boundary with no gap between
        # them. The old gate (`logs == 0 and planks == 0`) left exactly such a gap: a single
        # leftover plank was too little for the bed (three planks) yet cleared the `== 0` test, so
        # neither `get_wood` nor `craft_bed` fired and the Nether lull fell through to `wait` (live
        # run97 it149-162: 4 wool, 1 jungle_planks, 0 logs, no bed, 14 `wait` ticks at the base).
        need_planks = 0
        if tier == 0:
            need_planks = max(need_planks, 5)           # craft_pickaxe
        if not table:
            need_planks = max(need_planks, 4)           # craft_table
        if not bed and bed_ready:
            need_planks = max(need_planks, 3)           # craft_bed
        # The spare pickaxe's sticks, due either because the worn tool wants a same-tier backup
        # (`_pickaxe_due_for_spare`) or because the banked trip is still short of its second pickaxe
        # (`_spare_pickaxe_wanted`, `NETHER_PICKAXES`). The second need was missing here, so a lull
        # that had the diamonds but no wood could not fetch it: live run100 stood at the base with one
        # diamond pickaxe, three diamonds and a healthy tool but no logs -- `craft_pickaxe_spare`
        # could not build (no sticks), `mine_diamond` was not short, and nothing supplied the wood,
        # so the candidates fell through to `wait` for 300+ ticks and the gate was never lit.
        if (_pickaxe_due_for_spare(gs) or _spare_pickaxe_wanted(gs)) and not _can_sticks(inv, logs, planks):
            need_planks = max(need_planks, 2)           # sticks for the spare pickaxe
        # A lost shield is a hard gate on the Nether departure (`_gear_ready`), and its six planks
        # are a *unit* need, not a "no log carried" one: two logs make eight, one log only four,
        # and the `logs == 0` gate below stays shut for both. Without this a bot with a table, an
        # ingot and one log had only `mine_iron`/`place_chest` on offer and never re-armed
        # (checked against `candidates()` on run98's gate state, shield dropped, 5 planks).
        shield_short = (not _has_shield(gs) and table and tier >= IRON_PICK_TIER
                        and _plank_units(inv) < SHIELD_PLANKS)
        if shield_short:
            need_planks = max(need_planks, SHIELD_PLANKS)
        # A full pack at a chestless base is the moment to build one: the surplus has nowhere to
        # rest, so it sheds on the ground (`drop_junk`) and, come the Nether chain, the blaze rods
        # have no home either. The eight planks a chest costs are a *unit* need like the shield's,
        # and run191 showed they are never in hand at a base: it settled at it45 and never carried
        # more than six plank units after, so `craft_chest` could never fire and the pack shed on
        # the ground seven times. Gated on exactly what makes `craft_chest` eligible, plus a pack
        # full enough to shed -- so it rides alongside `drop_junk` as the build-instead-of-drop
        # option, and stays out of the idle `wait`/`go_home`/`explore` fallbacks a light pack owns.
        chest_short = (gs.base is not None and not _chest_ready(gs) and not gs.chest_blocked
                       and _chest_site(gs) is not None and not _has(inv, CHEST_ITEM)
                       and gs.slots_used >= INVENTORY_NEAR_FULL
                       and _plank_units(inv) < CHEST_PLANKS)
        if chest_short:
            need_planks = max(need_planks, CHEST_PLANKS)
        # The stew bowl is the last wood the lull owes: the Nether has no trees, so the bowl a
        # stew comes back from has to be carried in. A pack out of planks cannot craft it, and
        # without this the bowl step would simply drop out of `_all_nether_prep_steps` and the
        # gate would light without one (run171 crossed with both mushrooms and no bowl, and
        # never brewed a meal). Daylight only: chopping a tree is a walk, and the night's walks
        # are the ones that have killed the kit.
        if (gs.phase != "night" and _nether_idle(gs) and not _has(inv, BOWL_ITEM)
                and _plank_units(inv) < 3):
            need_planks = max(need_planks, 3)
        if _overworld(gs) and ((logs == 0 and planks < need_planks) or shield_short or chest_short):
            # Overworld only, like `craft_bed` and `place_bed`: the Nether has no trees, so a chop-a-tree
            # step can never succeed there and burns every tick it is offered. The bed chain is what
            # asks for wood most often (wool in the pack, no bed yet), which is why it kept winning the
            # offer over `leave_nether` - live, run42 spent all 600 iterations on `get_wood` in the
            # Nether and came home with nothing.
            out.append("get_wood")
        if (logs >= 1 or planks >= 5) and tier == 0:
            out.append("craft_pickaxe")
        if (logs >= 1 or planks >= 4) and not table:
            out.append("craft_table")
        if tier >= 1 and tier < STONE_PICK_TIER and table and cobble >= 3 and _can_sticks(inv, logs, planks):
            out.append("craft_stone_pickaxe")
        if tier >= 1 and table and not furnace and cobble >= 8:
            out.append("craft_furnace")
        if (tier >= 1 and table and gs.smoker_pos is None and _at_base(gs)
                and _smoker_craft_ready(gs) and _first_cookable(inv) is not None):
            out.append("craft_smoker")
        if (_smelting_reach(gs) and raw_iron >= 1 and coal >= 1 and iron < _iron_goal(gs)
                and _furnace_room(gs)):
            out.append("smelt_iron")
        if table and iron >= 3 and tier < IRON_PICK_TIER and _can_sticks(inv, logs, planks):
            out.append("craft_iron_pickaxe")
        # Bed chain: a bed lets `sleep` skip the night and set the respawn point. Overworld only, with
        # `sleep` and `place_bed`: a bed cannot be laid in the Nether (`place_bed` is refused there),
        # so crafting one down there spends wool and a slot on something the trip cannot use (live,
        # run37: `craft_bed` offered while hunting for blaze rods).
        if not bed and _overworld(gs):
            if table and bed_ready and (logs >= 1 or planks >= 3):
                out.append("craft_bed")
            elif not bed_ready:
                # Wool is killed for, so the walk needs the weapon too - run83's it85 stood
                # with `pickaxe: none` at hunger 20 and had only `hunt_food` and `hunt_wool`
                # on offer, two deaths later. String keeps its own guard: the spider walk is
                # how the bow's three strings are earned, and gate it and the expedition
                # cannot arm itself anymore.
                #
                # The guard is the *white* fleece, not the wool total: run160 carried
                # `gray_wool 1 + light_gray_wool 2` and no bed, which the old `wool >= 3` read
                # as ready. The server's plan for `white_bed` needs three white wool or the
                # twelve string that recipe makes them from, so the craft was offered 22 times
                # and refused 13 (`[CRAFT_MATERIALS_MISSING] Missing leaf materials: string x4`)
                # while the two hunts that could have supplied it stayed shut behind the
                # `wool < 3` test - the candidate list narrowed to `craft_bed` alone and then
                # to `wait` (it336-345).
                if gs.mob_nearest("sheep") is not None and _hunt_ok(gs):
                    out.append("hunt_wool")
                elif _safe_spider_hunt(gs):
                    out.append("hunt_string")
        # Iron is wanted for the pickaxe, again to rebuild one that is nearly worn, and for the
        # shield. A stone pickaxe is already enough to mine it (wood is not - it breaks the ore),
        # so this no longer stops at the iron tier; the upper bound was what left a worn iron
        # pickaxe with no way back. `_iron_goal` is the one place that decides how much.
        if raw_iron + iron < _iron_goal(gs) and _can_mine(gs, "iron_ore") \
                and gs.resource_distance("iron_ore") is not None:
            out.append("mine_iron")
        if (raw_iron >= 1 or _first_cookable(inv)) and coal < 1 and charcoal < 1 \
                and _can_mine(gs, "coal_ore") and gs.resource_distance("coal_ore") is not None:
            # The gate used to be `raw_iron >= 1`, which made coal iron-only fuel and left every cooked
            # meal burning a log: `_furnace_fuel` smelts eight items per coal and one per log, and live
            # the run went 4, 9, 17, 25, 32, 36, 44, 52, 55, 62, 69 oak logs with `raw_iron` at 0 and
            # `coal` at 0 on every single tick - while `coal_ore` sat inside the scan radius on all
            # thirty of them. Whatever is about to be smelted now counts, not just ore.
            out.append("mine_coal")
        # Shield (6 planks + 1 iron ingot, 3x3 so it needs the table): it only blocks while it sits
        # in the off-hand, so the craft is immediately followed by `equip_shield`. Gated behind the
        # iron pickaxe so it never spends the first ingots the pickaxe needs.
        if (not _has_shield(gs) and table and tier >= IRON_PICK_TIER
                and _plank_units(inv) >= SHIELD_PLANKS and iron >= SHIELD_IRON):
            out.append("craft_shield")
        if _has(inv, SHIELD_ITEM) and not _offhand(gs, SHIELD_ITEM):
            out.append("equip_shield")
        # The rest of the pre-diamond combat kit: an iron sword and a full iron armour set, so the
        # dive to the diamonds is not the unarmed, unarmoured trip that got the bot killed (live,
        # run12). Gated behind the iron pickaxe, like the shield, so it never spends the first
        # ingots that pickaxe needs; the armour is built one piece at a time, most protection first.
        if (table and tier >= IRON_PICK_TIER and not _has_iron_sword(gs)
                and iron >= IRON_SWORD_INGOTS and _can_sticks(inv, logs, planks)):
            out.append("craft_iron_sword")
        iron_armor_nxt = _next_iron_armor_item(gs)
        if (table and tier >= IRON_PICK_TIER and iron_armor_nxt is not None
                and iron >= iron_armor_nxt[1]):
            out.append("craft_" + iron_armor_nxt[0])
        # A crafted armour piece sits in the inventory until it is worn; `equip` slots armour even
        # without a destination, so one batch puts the whole set on. Wear it before crafting more.
        if _armor_to_wear(gs):
            out.append("equip_armor")
        # Diamond milestones, one at a time: pickaxe (three diamonds), then the sword (two, and only
        # once the pickaxe exists - bounding that away deadlocked the sword), then the armour set
        # (24, most protection per diamond first). `_next_diamond_item` names the next piece and its
        # cost, so the dig count and the craft gate share one source.
        # The dig itself waits for the combat kit (`_combat_ready`): the diamonds are what the
        # whole trip is for, and it is the trip that killed the unarmed bot.
        # `_can_mine` refuses a dig on an almost-broken pickaxe: breaking mid-action has the server
        # finish the ore with the swapped-in lower tier, destroying it without a drop.
        nxt = _next_diamond_item(gs)
        if nxt is not None:
            item, cost = nxt
            if diamond < cost:
                if _mine_diamond_wanted(gs):
                    out.append("mine_diamond")
            elif _diamond_craft_ready(gs, item):
                out.append("craft_" + item)

    # --- Home base ---
    # One surface anchor owns the bed, the portal frame and the chest. It is established once
    # the diamond kit is complete, and after that the base steps are only eligible while the
    # bot is there; `go_home` walks it back when base work is waiting.
    if _gear_ready(gs):
        if gs.base is None and _overworld(gs):
            # The base, the bed and the portal all belong under open sky. A bot that mined its
            # way down stands on a cave floor whose mined `air` reads exactly like sky, so it
            # climbs back out first: `surface_up` probes the surface and walks up to it, and only
            # then is `establish_base` offered.
            #
            # The Nether has no such climb. Its "surface" is the bedrock ceiling at y=127, so the
            # probe answered `[NAVIGATION_HEIGHT_REQUIRED] ... highest block bedrock` and the
            # climb never ran (live, run163 it0-1, and it6 again after the shutdown guard
            # released it). The base is an Overworld anchor; a run that starts down there leaves
            # through the gate (`leave_nether`) instead.
            out.append("establish_base" if _at_surface(gs) else "surface_up")
        elif not _at_base(gs) and _go_home_wanted(gs) and not _stronghold_phase(gs):
            # Three reasons to walk home: some base step is waiting, or the Nether gate is lit,
            # stocked and the bot's next job - which only it can do standing on the base - or it
            # is night and the bed is at the base.
            #
            # The third one was missing, and it made the sleep rule dead code. Live, run67: over
            # nineteen night ticks the bot held ten or eleven metres from the base - `mine_stone`
            # at sixty-one - and slept on none of them, because `_sleep_ready` demands standing on
            # the base with a lit gate. The candidates on those ticks were `explore` and
            # `craft_arrows`, and a skeleton found it eleven metres from home. Home is where the
            # bed is; that counts as a reason to be there on its own, with no work waiting.
            out.append("go_home")

    # --- Overworld -> Nether gate ---
    # Gated on the whole diamond kit (`_gear_ready`) so a fresh run finishes the tier first.
    # The obsidian step stands while no frame has been raised at the base: the obsidian already
    # in the frame plane is the latch that stops `mine_obsidian` re-mining after `build_portal`
    # has spent the ten blocks. It is the *frame* that tells our fresh build apart from the
    # natural obsidian that lines the very lava the bot walks to, so the carried count alone
    # never has to decide it -- and a pack left short by a restart can still mine what it needs
    # even with the igniter already in hand.
    # `build_portal` and `place_bed` run only at the base (`_at_base`), so the gate stands at
    # one fixed spot the respawn point can be pinned to.
    # The chain is Overworld work: the frame, the water and the gravel all belong to the base.
    # `_portal_lit` is a scan read, not a latch (its own docstring says so), so a bot standing in
    # the Nether reads "no lit gate" and the bucket and flint steps come back onto the offer -
    # live, run124 it40-41: `['fill_bucket', 'get_flint', 'search_enderman']` at (-6.5, 73, 265)
    # in the Nether, where there is no water to scoop and no gravel to dig.
    if _overworld(gs) and _gear_ready(gs) and not _portal_lit(gs):
        has_bucket = _has(inv, BUCKET_ITEM)
        has_water = _has(inv, WATER_BUCKET)
        has_flint = _has(inv, FLINT_ITEM)
        has_fs = _has(inv, FLINT_STEEL_ITEM)
        obsidian = inv.get("obsidian", 0)
        # The bucket is only ever for *making* obsidian: `mine_obsidian` pours the carried water
        # onto lava and mines what forms, scooping a fresh bucket of water between blocks. The
        # water is wanted exactly while a frame is still to be banked, so the chain (and the
        # step that spends the water) closes together: at a base whose water column sits out of
        # reach, offering the fill anyway strands the run -- the step fails (`[BUCKET_FAILED]`),
        # the base steps send the bot home, and the two trade it between the base and the water
        # for the rest of the run. Live, run9 spent 45 iterations on fill_bucket/gohome after the
        # obsidian was banked, the portal ready to build throughout.
        if _frame_needs_obsidian(gs, inv):
            if not has_bucket and not has_water and table and iron >= BUCKET_IRON:
                out.append("craft_bucket")
            elif has_bucket and not has_water:
                out.append("fill_bucket")
        if not has_flint and not has_fs and gs.resource_distance("gravel") is not None:
            out.append("get_flint")
        # Obsidian is *made*, not dug: the step is offered while the bucket is full and a lava
        # source can actually be worked, or while freshly formed obsidian lies within reach
        # (to mine it). "Worked" has two meanings: a cell the bucket can reach directly
        # (`_lava_pour_target`), or a source the dig can open for itself (`_lava_cast_pool`).
        # The reach test on the mine side keeps the dig shallow - the natural deposit under the
        # same lava caves is tens of blocks down and never comes that close.
        obsidian_open = (_frame_needs_obsidian(gs, inv) and _can_mine(gs, "obsidian")
                         and not _shed_owed(gs, inv))
        if obsidian_open and (_obsidian_within_reach(gs)
                              or (has_water and (_lava_pour_target(gs) is not None
                                                 or _lava_cast_pool(gs) is not None))):
            out.append("mine_obsidian")
        elif (obsidian_open and has_water and _lava_walk_target(gs) is not None):
            # The obsidian is made at a lava source, and a source is only in the scan when the bot
            # stands near it. With none loaded, the chain has no step at all and the offer falls to
            # the `explore`/`go_home` fallback (live, run114 it159-225: 66 ticks of 55-block
            # ping-pong between base and frontier with seven obsidian still owed). Walking to the
            # lava the scan *can* see is what reopens it. Once a source is in the listing the dig
            # walks to it itself, so the walk yields to `mine_obsidian`.
            out.append("approach_lava")
        # The obsidian gate here means "the frame's obsidian is secured before iron is spent on the
        # igniter" - either carried for a build still to come, or already standing in the frame. A
        # restart onto a world whose frame a previous run finished (`_frame_started_at_base`) carries
        # no ten obsidian and never will, so the raw count alone would hold the igniter off forever
        # and the dark gate could never be re-lit (run204: complete frame, two obsidian in the pack).
        if (not has_fs and has_flint and iron >= FLINT_STEEL_IRON and table
                and (obsidian >= PORTAL_OBSIDIAN or _frame_started_at_base(gs))):
            out.append("craft_flint_and_steel")
        # `build_portal` runs while the bot carries ten obsidian, or to *resume* a frame it
        # already started: a partial build leaves exactly the cells it still needs in the pack,
        # and re-sending the same structure resumes it (the server takes the interior corner).
        # A frame the build already finished is stepped over: a leftover obsidian in the pack
        # would otherwise keep the resume clause live on a finished frame, and the server answers
        # the no-op with a plain success, which no stuck guard counts (`gs.frame_complete`,
        # `Agent._note_build`; live, run25: 163 identical `build_portal` ticks, the bot frozen).
        can_build = (obsidian >= PORTAL_OBSIDIAN
                     or (obsidian >= 1 and _portal_frame_started(gs)))
        if _at_base(gs) and has_fs and not gs.frame_complete and _frame_blocker(gs) is not None:
            # A block the build cannot remove -- a container in a frame cell, which the movement
            # policy only digs when the break is asked for (`Agent._note_frame_cells`) -- clears
            # first, with the explicit `collect_block` the build never sends (live, run27: run26's
            # furnace on the new base's frame corner). Clearing needs no obsidian and no cobble,
            # so it stands ahead of the build gate: the cell is cleared while the pack is refilled.
            out.append("clear_frame")
        elif (_at_base(gs) and has_fs and can_build and cobble >= PORTAL_CORNERS
                and not gs.frame_complete):
            # A block left standing in the opening (above the support layer) has to go first:
            # the frame would be built around it and `light_portal` would answer
            # `[PORTAL_NO_FRAME]` forever (live, Nether run: a bed the generic sleep laid
            # "beside the bot" landed in the opening). `gs.portal_blocked` is filled by the
            # loop's opening read; while it is non-empty the dig stands in front of the build.
            out.append("clear_portal" if gs.portal_blocked else "build_portal")
        # The frame is proven finished by the plane read (`_frame_complete_at_base`): the ten
        # obsidian cells the gate needs stand, named by the frame scan. The old
        # "obsidian < PORTAL_OBSIDIAN" alone was not enough - live, run80 from it215: a build had
        # spent 7 of the ten blocks and gone quiet for 350 ticks, and `light_portal` fired 22
        # times on the still-short frame, the server answering `[PORTAL_NO_FRAME]` each time.
        # Lighting also stands down while a stray block still sits in the opening
        # (`gs.portal_blocked`): the clear steps dig it first. And the trip must be stocked
        # (`_expedition_ready`; live, run22 walked into its own early-lit gate). There is no hour:
        # `place_bed` follows the lit gate the very next tick, by day too (live, run80: the last
        # build closed the frame late in the day, and the run ended beside a gate that never burned).
        if (_at_base(gs) and has_fs and not gs.portal_blocked
                and _frame_complete_at_base(gs)
                and _portal_frame_started(gs) and _expedition_ready(gs)):
            out.append("light_portal")
    # The bed goes down at the base whatever the hour: `sleep` lays it, and by day the server
    # uses the bed instead of sleeping in it, which sets the respawn point all the same ("Respawn
    # point set at the bed, but daytime prevents sleeping"). Holding it back for the night left
    # the run nothing to do between a gate lit by day and dark (run86: ~300 `wait` ticks). It waits for the *lit*
    # gate (`_portal_lit`), not the bare frame: the server lays a carried bed on the nearest flat
    # footprint, and beside an unlit frame that is the open 2x3 opening, which `clear_portal`
    # cannot dig (a bed is protected from a route dig) and `light_portal` then fails on. Once the
    # gate burns the opening is solid and the bed falls on the ground beside the base.
    # The bed's ground is built before the bed is laid: digging alone can never give the bed a
    # floor, so a footprint with a hole under it stays refused however often the ground around it
    # is cleared (`_bed_pad_wanted`, `gs.bed_pad`, planned from the same footprint scan).
    if _bed_pad_wanted(gs):
        out.append("prepare_bed_pad")
    if (_at_base(gs) and _portal_lit(gs) and _has_bed(inv) and not _bed_at_base(gs)
            and _overworld(gs)):
        # A bed already standing outside the base box is the one `sleep` will click: the server
        # searches a 32-block radius for the nearest bed and walks to it, so the carried bed is
        # never laid where the bot stands (live, run156 it214-600: 131 `place_bed` ticks, each
        # walking twelve blocks to the run155 bed at (-31,71,172) and `go_home` walking back).
        # Taking the stray up first leaves the server nothing to find, and the next tick lays the
        # carried bed at the stance.
        if _bed_needs_moving(gs) and "take_bed" not in gs.stuck_labels:
            out.append("take_bed")
        out.append("place_bed")

    # --- The Nether expedition (phase 3) ---
    # One chain, read in the order the bot has to walk it: cross, keep piglins off, kill blazes
    # for nine rods, come home with them, swap the helmet back, turn the rods into powder, and
    # park the powder in the chest at the gate. Each step only appears while the ones before it are
    # done, so no step is ever offered out of turn.
    #
    # The same chain reopens for a stronghold-phase powder top-up (`nether_expedition_open`): the
    # rare case where the banked powder was lost and the eyes cannot be crafted. A fresh stow
    # closes the expedition again and clears the exhaustion latch, so the eye-crafting resumes.
    if nether_expedition_open(gs):
        if _in_nether(gs):
            # The helmet first, and *only* the helmet: piglins attack on sight without a worn gold
            # piece, and the whole trip is fought around them, so until it is on there is no safe
            # second choice to leave on the list. Offered alongside `explore` the model took the
            # explore 52 of the 66 Nether ticks it was on the list (run104), and training cannot
            # fix that: a non-progressing `explore` carries no known-good alternative, so the
            # dataset drops it and never learns the swap was the move (`dataset.py`). A worn gold
            # piece is a hard precondition, not a preference, so it is a filter and not a choice -
            # the swap owns the tick outright, like the rescues at the top of this function.
            # (Urgent survival - a meal the body needs, a drowning, a shut-in - has already
            # returned above, so nothing it would preempt is lost here.)
            if _gold_helmet_needed(gs) and "swap_to_gold_helmet" not in gs.stuck_labels:
                return ["swap_to_gold_helmet"]
            # The early return above skips the stuck filter at the end of this function, so a swap
            # the server keeps refusing owns every tick forever: run124 it18-36, eighteen ticks of
            # "No new action started; body owner: hostile_reflex" with nothing else on the list.
            # Once the guard has withheld it, the tick falls to the rest of the Nether chain.
            # A pack full of droppable junk (netherrack) reads as "out of room" and would send the
            # bot home with nothing. The junk frees itself with `drop_junk`; the room does not. So
            # while the pack is full of something that can be dropped, the bot sheds it and keeps
            # searching instead of turning back (live: the pack sat at 34/36 of netherrack and
            # `_nether_must_return` fired on the slots, `leave_nether` offered every tick).
            # The shed has to be *on offer* for that to hold. `_note_drop_undone` takes it off for
            # `STUCK_COOLDOWN` ticks once the ground has refused it twice in a row, and a pack that
            # cannot be emptied is a reason to come home with the rods already got, not to keep
            # marching with nowhere for the next one (run118: 36/36 for thirteen ticks straight,
            # five of nine rods).
            junk_full = (gs.slots_used >= INVENTORY_NEAR_FULL and _droppable(gs, inv)
                         and "drop_junk" not in gs.stuck_labels)
            # The gate the bot came through is the only way back, and the Nether side of it can go
            # out while the run is still down there: a frame that stands but does not burn makes
            # every `leave_nether` a `[PORTAL_TARGET]` refusal, and with nothing else on the offer
            # the tick went to `explore` (live, run195 it410-413: four identical refusals at the
            # bot's own exit frame, then 270 blocks of marching away from it). `light_portal` used
            # to be an Overworld step only - `_at_base` and `_portal_lit` gate it - so down here
            # there was no step that could relight it at all.
            # Relighting is a precondition of the return, not a preference between two steps, so
            # it owns the tick, and the stuck guard keeps a refused relight from owning the run.
            if (_nether_exit_ignite_cell(gs) is not None
                    and "light_portal" not in gs.stuck_labels):
                if _has(inv, FLINT_STEEL_ITEM):
                    return (["eat_food"] if eating else []) + ["light_portal"]
                # No igniter: the craft needs a free slot, so a pack full of junk sheds first
                # (run195 ended 36/36 with netherrack and no flint and steel), and the shed owns
                # the tick while the ground still accepts it.
                if not junk_full and _igniter_craftable(gs, inv):
                    return (["eat_food"] if eating else []) + ["craft_flint_and_steel"]
            if _pearl_wave_trip(gs):
                # --- Pearl wave (Welle 2+) ---
                # The rod haul is already home and stowed, so the blaze hunt and the fortress march
                # are behind the bot: this trip is down here only for the pearls the eyes still
                # owe. Barter the gold reserve with a piglin first (one reserve buys about a pearl),
                # then kill an enderman in reach, then mine the roof that fight needs when the pack
                # is down to the block floor, then roam a warped forest to find one within the
                # pearl budget; out of budget, out of health, or forced back, come home and stow -
                # a death down here would drop the whole wave, which is why the waves exist.
                # `barter_pearls` moved here from the rod trip (it used to ride the rods-done tick,
                # commit 584a51a): the user's waves keep the risky trades off the rod haul.
                if _safe_enderman_hunt(gs) and gs.slots_used < INVENTORY_SLOTS:
                    # An enderman in reach is the whole point of the trip, so it outranks the
                    # near-full walk home. run168 it107-108 held one at 23.9 m with 63 netherrack
                    # in the pack and `slots_used 33` - `_nether_must_return` fired first and sent
                    # the wave home past it, three trips running. One free slot is all a pearl
                    # needs; the near-full latch stays for the case where there is none.
                    out.append("hunt_enderman")
                elif _nether_must_return(gs) and not junk_full:
                    out.append("leave_nether" if _way_home_known(gs) else "explore")
                elif junk_full:
                    # A pack full of droppable junk reads as "out of room" and would send the bot
                    # home empty; shed it first so the next pearl has somewhere to land.
                    out.append("drop_junk")
                elif _barter_pearls_wanted(gs):
                    out.append("barter_pearls")
                elif _scaffold_short(gs):
                    # The scaffold is spent twice over: the fight's roof runs down to the floor
                    # `_safe_enderman_hunt` needs (run169 spent 128 blocks on two hunts), and
                    # `navigate`/`explore_frontier` spend the same stack as bridging material.
                    # run169 it187 bridged with all 122 it carried and then stood nine ticks at
                    # the same cell with `[NAVIGATION_STOPPED] search timed out after 2000 m` - no
                    # way home with 6 pearls in the pack. Netherrack is the Nether's own floor, so
                    # the restock is a dig here instead of a ~300 m walk home for cobblestone.
                    out.append("gather_scaffold")
                elif _nether_haul_leashed(gs):
                    # Pearls or rods are aboard and the gate is further than the leash: the wave
                    # stops covering ground and walks back. run203 roamed 700 m out on a budget
                    # that counts metres walked, not metres from home, and died there with 16
                    # pearls and 9 rods when the return route closed under it.
                    out.append("leave_nether" if _way_home_known(gs) else "explore")
                elif _search_enderman_wanted(gs):
                    # No enderman in reach but the pearl budget still has ground to cover: roam new
                    # terrain to load one, the `explore_frontier` march `search_enderman` resolves
                    # to. `_search_enderman_wanted` carries the budget (`NETHER_PEARL_SEARCH_M`).
                    out.append("search_enderman")
                else:
                    # Pearls still owed but nothing to do for them here - none in reach, the budget
                    # spent, the trade closed: home with what pearls were got, stow, and (if trips
                    # remain) the next wave relaunches from the base. Without a known exit the
                    # march is the offer: standing still is not a way home (see `_way_home_known`).
                    out.append("leave_nether" if _way_home_known(gs) else "explore")
            else:
                # --- Rod wave (Welle 1) ---
                rods = _blaze_rods(inv)
                if rods >= _rods_target(gs):
                    # The rod haul is complete: walk back through the gate and stow it. The pearls
                    # are a later wave's job, not this trip's - a death with the rods in the pack
                    # would lose the whole haul, so securing them at home comes first. With no
                    # known exit the march is the offer (`_way_home_known`): a step that resolves
                    # to None would only defer the tick.
                    out.append("leave_nether" if _way_home_known(gs) else "explore")
                elif _safe_blaze_hunt(gs):
                    out.append("hunt_blaze")
                elif (_nether_must_return(gs) and not junk_full) or _nether_hopeless(gs):
                    # Out of health, out of room, or out of chances: come home with what was got
                    # rather than die down there. The rods in the pack are already worth the trip.
                    # "Out of room" is suppressed while the room is full of droppable junk: that is
                    # `drop_junk`'s job, not a reason to abandon the trip. No known exit: march.
                    out.append("leave_nether" if _way_home_known(gs) else "explore")
                elif _nether_haul_leashed(gs) and _way_home_known(gs):
                    # Rods in the pack and the gate further than the leash: the fortress march
                    # stops where the haul still has a way back to it (see `_nether_haul_leashed`).
                    out.append("leave_nether")
                else:
                    # No blaze in reach, no reason to turn back: the fortress search. It is offered
                    # first-class here, the way `mine_diamond` is in the Overworld - not buried in
                    # the `explore` fallback, which a pack full of netherrack (the tech tree reading
                    # as incomplete) kept from ever firing, so the bot never marched to the bricks
                    # it could see (live: `found: 0` for every fortress block, `explore` offered on
                    # 1 of 2499 Nether ticks). `explore`'s Nether branch walks to the spawner/bricks
                    # when they are in view and marches the frontier toward them when they are not.
                    if junk_full:
                        # The shed owns the tick. Offered beside `explore` it was on the list 19
                        # times in run118 and taken 0, while the march hoovered netherrack 3 -> 173
                        # in fifteen ticks and the pack stayed at 36/36: a rod has nowhere to land
                        # in a full pack, so emptying it is a precondition of the haul, not a
                        # preference between two steps - the same reason the pearl wave above sheds
                        # alone, and the same reason the gold helmet swap owns its tick.
                        # The shed owns the tick, but not a tick the server refuses: an early return
                        # here skips the stuck filter at the end of the function, so a shed the ground
                        # keeps rejecting would own the run (the shape run118's 36/36 pack had).
                        shed = [] if "drop_junk" in gs.stuck_labels else ["drop_junk"]
                        return (["eat_food"] if eating else []) + shed
                    out.append("explore")
        else:
            # Overworld side: the gate is the way in, and it opens once the expedition is stocked.
            if _spare_pickaxe_missing(gs):
                out.append("craft_pickaxe_spare")
            elif _mine_diamond_wanted(gs):
                # The trip needs the second pickaxe and the pack cannot pay for it yet, so the
                # next step is the digging - asked for in the offer, where it is visible, instead
                # of leaving an unbuildable label to be escalated away by System 2 every tick.
                out.append("mine_diamond")
            if _at_base(gs) and _nether_departure_ready(gs):
                # The gold helmet goes on at the base, before the crossing - a diamond helmet
                # worn into the Nether leaves the bot piglin-exposed until it swaps down there
                # (live, run99: the swap came six Nether ticks after each entry). Offered ahead of
                # `enter_portal` so the helmet is on the head before the bot steps through.
                if _gold_helmet_needed(gs):
                    out.append("swap_to_gold_helmet")
                out.append("enter_portal")
            # The haul, back in the Overworld: the helmet comes off, the rods become powder, the
            # powder goes into the chest. `nether_entered` is this run's latch, so a rod left in
            # the pack by an earlier run sat here unprocessed while `_nether_haul_home` read it as
            # a haul still on its way in - the gate shut on a trip that could never start
            # (run123, ten `wait` ticks at the base). Carried rods or powder are a haul whatever
            # run carried them across.
            if not _in_nether(gs) and (gs.nether_entered or _blaze_rods(inv) or _blaze_powder(inv)):
                # The gold piece is the piglin pass for a crossing that is about to happen; the
                # diamond helmet is the better headgear for everything else the base does. Both
                # preconditions were satisfiable on the same tick, and that is a loop: run127
                # it137-174, at the base with the gate lit and the trip stocked, the model took
                # `swap_to_gold_helmet` 61 times and `swap_to_diamond_helmet` 40 times in
                # thirty-eight ticks - each swap made the other one wanted again - and never
                # stepped through. A pending departure owns the helmet slot; the diamond swap
                # waits until no crossing is on the list.
                if (_worn(gs, GOLD_HELMET_ITEM) and (_has(inv, "diamond_helmet")
                                                     or _worn(gs, "diamond_helmet"))
                        and not _nether_departure_ready(gs)):
                    out.append("swap_to_diamond_helmet")
                if _blaze_rods(inv) > 0 and _has(inv, "crafting_table") \
                        and _blaze_powder(inv) < _powder_target(gs):
                    out.append("craft_blaze_powder")
                # Powder the eye chain can spend from the pack is not a haul to bank: offered
                # beside `take_blaze_powder` it became a lid the model kept crossing in both
                # directions (run125 it16-47, thirty-two ticks of the same two powder).
                if _blaze_powder(inv) > 0 and _chest_ready(gs) and _storage_target(gs) is not None \
                        and not _powder_is_eye_fuel(gs):
                    out.append("stow_blaze_powder")

    # --- The stronghold (the phase after the Nether haul) ---
    # Offered once the Nether haul stands: gather pearls, bring the banked powder back out,
    # craft the eyes, find and travel to the stronghold, fill the frame. Once the portal is
    # active the same function offers the bed chain instead (craft, place beside the frame,
    # sleep to reset the spawn). The chain lives in `stronghold.stronghold_candidates`,
    # Overworld-gated there, so it stays silent while a powder top-up runs the bot through the gate.
    if nether_expedition_complete(gs):
        out.extend(stronghold_candidates(gs))

    # --- Home base storage ---
    # A chest at the base is where a filling pack sheds surplus, well before the server's
    # `[INVENTORY_FULL]` refusal; `drop_junk` is only the fallback when no chest is there.
    # --- Home base storage ---
    # A cook refused with `[FURNACE_NOT_EMPTY]` means the base furnace still holds output. Empty it
    # first: the alternative the loop used to take - abandon the cell and cook in a carried furnace
    # - cost a furnace per batch (live, run26, and run32 it blocked the expedition's cooking).
    if _furnace_needs_draining(gs) and _at_base(gs) and _furnace_room(gs):
        out.append("collect_smelted")

    # --- Home base storage ---
    if (gs.base is not None and not _chest_ready(gs) and not gs.chest_blocked
            and _chest_site(gs) is not None):
        if _has(inv, CHEST_ITEM) and _at_base(gs):
            out.append("place_chest")
        elif table and not _has(inv, CHEST_ITEM) and _plank_units(inv) >= CHEST_PLANKS:
            out.append("craft_chest")
    elif _chest_ready(gs) and not gs.chest_blocked and _at_base(gs) and _storage_full(gs):
        # run79's storage deadlock: one chest at the base, then `[CONTAINER_BATCH_INCOMPLETE]`
        # on every side and the pack had to keep everything. The same chain that raised the
        # first chest opens again once the server has called every chest in the network full:
        # another one is crafted and stood beside the base, and the relief deposits into it.
        # The ground's own refusal (`chest_blocked`) closes this too -- a base whose first
        # chest had to be given up on will not hold a second one somewhere nearby.
        relief = None
        if _has(inv, CHEST_ITEM) and _overflow_site(gs) is not None:
            relief = "place_chest"
        elif table and not _has(inv, CHEST_ITEM) and _plank_units(inv) >= CHEST_PLANKS:
            relief = "craft_chest"
        # A pack at the relief threshold with nowhere to put its surplus is not a choice
        # between the chest and the next step. Live, run200 it8: `craft_chest` sat on the offer
        # beside `enter_portal` and the crossing took the tick sixteen times, so `powder_stowed`
        # never latched, the pearl wave stayed shut and the run spent its iterations walking
        # through its own gate. The stuck guard keeps a refused site from owning the run.
        if relief and gs.slots_used >= INVENTORY_NEAR_FULL and relief not in gs.stuck_labels:
            return [relief]
        if relief:
            out.append(relief)
    storage_target = _storage_target(gs)
    # The walk-over deposit works the chest where it stands: `at_chest` is the base or the
    # lid's reach, not only the base - a chest adopted outside the base radius was walked to
    # but never worked from the chest side, and the bot went home for the very surplus the
    # chest was waiting for (run82's 481-tick ping-pong).
    at_chest = (gs.base is not None and _chest_ready(gs) and storage_target is not None
                and (_at_base(gs) or _near_cell(gs, storage_target)))
    # The stronghold march owns its own relief (`stronghold_candidates` drops on the ground) and
    # must never be sent home to a chest: a `stow_surplus` ~2000 blocks back is exactly the base
    # ping-pong this phase has to avoid (run175). So the home-storage relief stands down there.
    if _striding_to_stronghold(gs):
        pass
    elif gs.slots_used >= INVENTORY_NEAR_FULL:
        if at_chest:
            if _sheddable(gs, inv):
                out.append("stow_surplus")
        elif _droppable(gs, inv):
            # No chest to hold it: shed the surplus on the ground rather than stall the gather.
            # `_droppable`, not `_sheddable` - the whitelist may only ever feed the chest, because
            # a drop is not undoable. Asking `_sheddable` here offered `drop_junk` for items the
            # resolve then refused to name, which is an escalation per tick.
            out.append("drop_junk")
    elif at_chest and (_junk_carried(inv) or _hoard_surplus(gs, inv)
                        or _surplus_besides_keep(gs, inv) or _bank_surplus_food(gs, inv)):
        # A stack that crossed its cap with room to spare, and a chest standing right there.
        # Live, cobblestone drifted 60, 65, 67, 68, 72 with `chest_pos` known and the bot on
        # the base the whole time, because `HOARD_KEEP` capped at 64 only ever fired *inside*
        # the full-pack branch above. A stack of 68 costs two slots for the sake of four blocks,
        # and nothing in the loop ever noticed. With no chest in reach the surplus is left alone:
        # dropping it on the ground over a cap is a different decision.
        out.append("stow_surplus")

    # Overworld only: the cobble a tech tree spends belongs to the surface, and a pack full of
    # netherrack reads as short of it, so without this the Nether offered `mine_stone` on every
    # tick and the `explore` fallback (the only thing that marched toward a fortress) never fired.
    # Withheld on the stronghold march: the trek to the frame spends no cobblestone, so the step is
    # pure noise that competes with `travel_to_stronghold` and pulls the bot off the line (run175).
    if cobble < 12 and _can_mine(gs, "stone") and _overworld(gs) and not _stronghold_phase(gs):
        out.append("mine_stone")

    # --- Nether prep (the daytime lull once the gate frame stands) ---
    # The bow, its arrows, the gold helmet and the deep cobblestone stack, stocked once the
    # gate frame stands (still unlit: the gate only lights when all of this is done). `_nether_prep_steps`
    # is the single source for both this list and the
    # completion gate, so the offered steps and `tech_tree_complete` never disagree.
    # Offered Overworld only: `_nether_idle` (its scope) is true down there too, and a prep step
    # that digs from the base has no base in the Nether to dig from.
    # Not on the stronghold march: the lull's gold dig and base crafts are the same detour as
    # `mine_stone` - the trek never spends them, and the frame is the only job left before the
    # portal (run176). The crossing is already behind the run when the phase opens.
    if _overworld(gs) and not _stronghold_phase(gs):
        out.extend(_nether_prep_steps(gs))

    # Home-base ground work. Once the gate is lit, a daytime maintenance step digs away the stray
    # terrain and reflex-placed defence scaffold that pile up above ground level around the base, so
    # the home ground stays flat (`_base_leveling_wanted`, `gs.raised_cells`). Offered below the prep
    # and the gate chain but above `wait` (the empty-list fallback), so an idle bot at a lumpy base
    # levels it instead of waiting, and it drops out the moment the scan comes up empty.
    if _base_leveling_wanted(gs):
        out.append("flatten_home_base")

    # The same label can arrive twice (the light `mine_stone` above and the deep lull stack in
    # `_nether_prep_steps`); keep the first occurrence so the priority order is stable.
    out = list(dict.fromkeys(out))

    # The labels the loop is withholding (`Agent._track_stuck`): each kept answering the identical
    # refusal without progress, so it is taken out of the offer for a cooling-off window and the
    # fallback below gets its turn. Without this a single-candidate step the server refuses
    # forever owns every tick and the fallback never fires (live, run16: `mine_obsidian` ->
    # `[NAVIGATION_TARGET_UNSUPPORTED]` on 215 ticks as the only candidate).
    if gs.stuck_labels:
        out = [label for label in out if label not in gs.stuck_labels]

    # A naked bot at night takes no long walk (`_defenceless_at_night`): the recovery steps already
    # stand still in that state (`_recovery_steps` returns nothing at all), and the ordinary offer
    # must too. run95 it274 explored at 20 health with no tool and no armour on, walked into a
    # spider and a skeleton at 10-16 m, and was dead within three ticks; the same night's
    # `get_wood` walks had already spent two ten-minute `[BODY_HELD]` waits. `go_home` is left in,
    # not out of an analogy with the recovery steps: it is gated on `_gear_ready`, so a bot that
    # lost its kit (run95's naked body) never sees it, and a still-geared bot's walk home is the
    # walk to the bed that ends the night.
    still = _defenceless_at_night(gs)
    if still:
        out = [label for label in out if label not in DEFENCELESS_WALK_LABELS]

    # The explicit last fallback when nothing productive is available. In the lull once the gate
    # frame stands the only work left is the prep, and wandering off just burned
    # five pathfinder timeouts and left the bot away from the base at dusk (live, from-zero run), so
    # hold at the base and wait it out rather than explore. Roaming to fill an open food reserve
    # was tried live (run19b) and reverted: the roam carried the bot 150-200 m out, where the
    # base-range latches re-open the portal chain and it stranded in water for 76 ticks.
    #
    # `wait` holds only *at* the base, though, so an idle bot that has drifted off it (a
    # `collect_drops` a few metres out) fell through to `explore` and marched away with no tether
    # home - `wait` never re-fired and the base was lost (live run97 it163-193: `explore` ratcheted
    # 332 -> 652 m out and oscillated there to the end). So an idle geared Overworld bot off its
    # base walks home first (`_idle_off_base`), where `wait` can hold it; `explore` stays the
    # fallback in the Nether (the march to the fortress) and before a base or a kit is in hand. A
    # home walk that keeps failing is stuck-listed like any other step, and then `explore` takes
    # over so the bot is not frozen pathing to a base it cannot reach.
    if not out:
        if still or (_at_base(gs) and _nether_idle(gs)):
            out.append("wait")
        elif (_idle_off_base(gs) and "go_home" not in gs.stuck_labels
              and not _stronghold_phase(gs)):
            out.append("go_home")
        else:
            out.append("explore")
    return out


def resolve(label: str | None, gs: GameState, target=None) -> Action | None:
    """Map a Laya choice to a concrete MCP tool call, or None if preconditions fail.

    `target` is the M4 parameter-layer pick (a `targets.Target`); it is only used by a
    label whose step needs a destination (`explore`), and ignored elsewhere.
    """
    if not label:
        return None
    # The stronghold labels are resolved in their own module (see docs/stronghold.md); a label it
    # does not own never reaches here, so the main dispatch below is untouched.
    if label in STRONGHOLD_LABELS:
        return resolve_stronghold(label, gs)
    inv = gs.inventory
    tier = _pick_tier(inv, gs.tools)
    x, y, z = gs.pos or (None, None, None)

    if label == "escape_water":
        # Get ground underfoot *now*: the nearest dry standable cell is the one target a floating
        # body can actually reach, so it wins outright. A far or underground home base is only the
        # fallback -- from treading water it cannot be walked to (live run8: the bot drowned ten-plus
        # times trying to dig 70 m down to a base 300 m off, lost its diamond kit, and the runtime
        # died). `navigate` is the one tool that still moves a floating body.
        target = nearest_ground(gs)
        if target is not None:
            tx, ty, tz = target.pos
            return Action(label, "navigate",
                          {"x": tx, "y": ty + 1, "z": tz, "dig": True, "scaffold": True},
                          "climb out of the water onto the nearest dry ground")
        if gs.base is not None:
            bx, by, bz = gs.base
            return Action(label, "navigate",
                          {"x": bx, "y": by + 1, "z": bz, "dig": True, "scaffold": True},
                          "swim clear of the water and climb back up to the dry home base")
        return None
    if label == "get_wood":
        if not _overworld(gs):
            # Gated here as well as in the offer: a label the policy still holds must not be able
            # to smuggle an impossible action through (same reason as `craft_bed`).
            return None
        return Action(label, "collect_block", {"block_name": "logs", "count": 8},
                      "gather logs to start the tech tree")
    if label == "mine_stone":
        if not _can_mine(gs, "stone"):
            return None
        return Action(label, "collect_block", {"block_name": "stone", "count": 8},
                      "mine cobblestone to unlock stone tools")
    if label == "mine_iron":
        if not _can_mine(gs, "iron_ore"):
            return None
        if inv.get("raw_iron", 0) + inv.get("iron_ingot", 0) >= _iron_goal(gs):
            return None
        count = max(1, min(_iron_goal(gs) - inv.get("raw_iron", 0), IRON_MINE_BATCH))
        return Action(label, "collect_block", {"block_name": "iron_ore", "count": count},
                      "mine iron ore for iron tools")
    if label == "mine_coal":
        if not _can_mine(gs, "coal_ore"):
            return None
        return Action(label, "collect_block", {"block_name": "coal_ore", "count": 3},
                      "mine coal for furnace fuel")
    if label == "mine_diamond":
        if not _can_mine(gs, "diamond_ore"):
            return None
        if not _combat_ready(gs):
            return None
        if not _diamond_dig_short(gs):
            # The pack can pay for the next diamond goal right now: the dig would only
            # pile up diamonds the run has no un-built reason for (run77's 140-diamond
            # standstill), and the craft that pays for it is the tick's real step.
            return None
        nxt = _next_diamond_item(gs)
        if nxt is not None:
            cost = nxt[1]
        elif _spare_pickaxe_wanted(gs):
            # The tech tree is finished - pickaxe, sword and the whole armour set - but the trip
            # still needs its second pickaxe, and that is a diamond reason too. Before this the
            # offer named it and the resolve refused it, every tick, into System 2.
            cost = SPARE_PICKAXE_COST
        else:
            return None
        count = max(1, cost - inv.get("diamond", 0))
        return Action(label, "collect_block", {"block_name": "diamond_ore", "count": count},
                      "mine diamond ore for diamond tools and armour")
    if label == "craft_pickaxe":
        if (_logs(inv) < 1 and _planks(inv) < 5) or tier != 0:
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "wooden_pickaxe"}]},
                      "craft a wooden pickaxe from carried logs")
    if label == "craft_table":
        if (_logs(inv) < 1 and _planks(inv) < 4) or _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "crafting_table"}]},
                      "craft a crafting table for 3x3 recipes")
    if label == "craft_stone_pickaxe":
        if inv.get("cobblestone", 0) < 3 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "stone_pickaxe"}], "temporary_workstation": True},
                      "craft a stone pickaxe at the carried crafting table")
    if label == "craft_smoker":
        if not _smoker_craft_ready(gs) or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "smoker"}], "temporary_workstation": True},
                      "craft a smoker so food is cooked faster at the base")
    if label == "place_smoker":
        cell = _smoker_site(gs)
        if (gs.smoker_pos is not None or not _smoker_worth_placing(gs) or cell is None
                or not _has(inv, "smoker")):
            return None
        sx, sy, sz = cell
        return Action(label, "place_block",
                      {"block_name": "smoker", "x": sx, "y": sy, "z": sz},
                      "place the carried smoker beside the furnace at the base")
    if label == "craft_furnace":
        if inv.get("cobblestone", 0) < 8 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "furnace"}], "temporary_workstation": True},
                      "craft a furnace to smelt iron")
    if label == "smelt_iron":
        if (not _smelting_reach(gs) or inv.get("raw_iron", 0) < 1
                or inv.get("coal", 0) < 1 or not _furnace_room(gs)):
            return None
        count = min(inv.get("raw_iron", 0), 3)
        return Action(label, "smelt_item", _smelt_args(gs, "raw_iron", count, "coal"),
                      "smelt raw iron into ingots")
    if label == "craft_iron_pickaxe":
        if inv.get("iron_ingot", 0) < 3 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "iron_pickaxe"}], "temporary_workstation": True},
                      "craft an iron pickaxe for mining diamonds")
    if label == "craft_shield":
        if (inv.get("iron_ingot", 0) < SHIELD_IRON or _plank_units(inv) < SHIELD_PLANKS
                or not _has(inv, "crafting_table") or tier < IRON_PICK_TIER or _has_shield(gs)):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": SHIELD_ITEM}], "temporary_workstation": True},
                      "craft a shield from planks and an iron ingot")
    if label == "craft_iron_sword":
        if (inv.get("iron_ingot", 0) < IRON_SWORD_INGOTS or not _has(inv, "crafting_table")
                or tier < IRON_PICK_TIER or _has_iron_sword(gs)
                or not _can_sticks(inv, _logs(inv), _planks(inv))):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "iron_sword"}], "temporary_workstation": True},
                      "craft an iron sword from two ingots and a stick")
    iron_slot = label[len("craft_iron_"):] if label.startswith("craft_iron_") else None
    if iron_slot in IRON_ARMOR_COST:
        cost = IRON_ARMOR_COST[iron_slot]
        if (inv.get("iron_ingot", 0) < cost or not _has(inv, "crafting_table")
                or tier < IRON_PICK_TIER):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": f"iron_{iron_slot}"}],
                       "temporary_workstation": True},
                      f"craft an iron {iron_slot} from {cost} ingots")
    if label == "craft_bow":
        if (_has(inv, BOW_ITEM) or not _has(inv, "crafting_table")
                or inv.get("string", 0) < BOW_STRING
                or not _can_sticks(inv, _logs(inv), _planks(inv))):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": BOW_ITEM}], "temporary_workstation": True},
                      "craft a bow from string and sticks for the ranged answer to ghasts")
    if label == "craft_arrows":
        if (not _nether_idle(gs) or not _has(inv, "crafting_table")
                or not _has(inv, FLINT_ITEM) or not _has(inv, FEATHER_ITEM)
                or not _can_sticks(inv, _logs(inv), _planks(inv))
                or inv.get(ARROW_ITEM, 0) >= NETHER_ARROWS):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": ARROW_ITEM}], "temporary_workstation": True},
                      "craft arrows from flint, sticks and feathers for the carried bow")
    if label == "equip_shield":
        if not _has(inv, SHIELD_ITEM) or _offhand(gs, SHIELD_ITEM):
            return None
        return Action(label, "equip",
                      {"items": [{"item_name": SHIELD_ITEM, "destination": "off-hand"}]},
                      "put the shield in the off-hand so it blocks attacks")
    if label == "craft_pickaxe_spare":
        if not _spare_pickaxe_missing(gs):
            return None
        if inv.get("diamond", 0) < 3 or not _can_sticks(inv, _logs(inv), _planks(inv)):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "diamond_pickaxe"}], "temporary_workstation": True},
                      "craft the second diamond pickaxe to take into the Nether")
    if label == "craft_diamond_pickaxe":
        if (inv.get("diamond", 0) < 3 or not _has(inv, "crafting_table")
                or tier >= DIAMOND_PICK_TIER or not _can_sticks(inv, _logs(inv), _planks(inv))):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "diamond_pickaxe"}], "temporary_workstation": True},
                      "craft a diamond pickaxe from three diamonds and sticks")
    if label == "craft_diamond_sword":
        if (inv.get("diamond", 0) < 2 or not _has(inv, "crafting_table")
                or tier < DIAMOND_PICK_TIER or _has(inv, "diamond_sword")
                or not _can_sticks(inv, _logs(inv), _planks(inv))):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "diamond_sword"}], "temporary_workstation": True},
                      "craft a diamond sword from two diamonds and a stick")
    if label == "equip_armor":
        items = [{"item_name": item} for item in _armor_to_wear(gs)]
        if not items:
            return None
        return Action(label, "equip", {"items": items},
                      "wear the crafted armour so its protection is active")
    armor_slot = label[len("craft_diamond_"):] if label.startswith("craft_diamond_") else None
    if armor_slot in DIAMOND_ARMOR_COST:
        cost = DIAMOND_ARMOR_COST[armor_slot]
        if inv.get("diamond", 0) < cost or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": f"diamond_{armor_slot}"}],
                       "temporary_workstation": True},
                      f"craft a diamond {armor_slot} from {cost} diamonds")
    if label == "craft_bucket":
        if (not _gear_ready(gs) or _has(inv, BUCKET_ITEM) or _has(inv, WATER_BUCKET)
                or inv.get("iron_ingot", 0) < BUCKET_IRON or not _has(inv, "crafting_table")):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": BUCKET_ITEM}], "temporary_workstation": True},
                      "craft a bucket from three iron ingots")
    if label == "fill_bucket":
        if not _has(inv, BUCKET_ITEM) or _has(inv, WATER_BUCKET) or not _overworld(gs):
            return None
        return Action(label, "use_bucket", {"action": "fill", "liquid": "water"},
                      "fill the carried bucket with water to pour onto lava")
    if label == "mine_obsidian":
        if (not _can_mine(gs, "obsidian") or _frame_started_at_base(gs)
                or inv.get("obsidian", 0) >= PORTAL_OBSIDIAN or _shed_owed(gs, inv)):
            return None
        # Phase 1: obsidian already lies within reach (what the last pour formed) - mine it.
        # The cell is named exactly, one block per tick: a bare `count` lets the server
        # re-target the nearest obsidian it can find once the formed block is gone. Live,
        # run113 it84: the pour formed one block, the action asked for eight, and the bot
        # spent 14 minutes treading the water pocket beside the lava on the way back to a
        # natural deposit 124 blocks away - the exact flooded-shaft walk phase 2 rules out.
        if _obsidian_within_reach(gs):
            hit = gs.resource_nearest("obsidian")
            return Action(label, "collect_block",
                          {"block_name": "obsidian", "x": hit.x, "y": hit.y, "z": hit.z},
                          "mine the obsidian the poured water just formed at the bot's feet")
        # Phase 2: no obsidian in reach, so make some - either by pouring the carried water into a
        # lava cell beside a real source (`_lava_pour_target`), or by letting the dig open a capped
        # pool itself. Natural obsidian deep under that lava is deliberately *not* a target:
        # tunnelling to it is what left the bot treading water in a flooded shaft.
        if not _has(inv, WATER_BUCKET):
            return None
        lava = _lava_pour_target(gs)
        if lava is None:
            if _lava_cast_pool(gs) is None:
                return None
            # The pool is capped: no cell beside the source is open to the bucket from here. The
            # server's mine process owns the cast - it routes to the pool, breaks the blocks
            # standing between the eye and the source, pours, scoops the bucket back and mines what
            # formed. No coordinates: naming a cell would name the lid the process is there to
            # remove.
            return Action(label, "collect_block",
                          {"block_name": "obsidian",
                           "count": min(PORTAL_OBSIDIAN - inv.get("obsidian", 0),
                                        OBSIDIAN_CAST_BATCH)},
                          "let the dig open the capped lava pool, pour the carried water into it "
                          "and mine the obsidian that forms")
        if _at_pour_stance(gs, lava):
            return Action(label, "use_bucket",
                          {"action": "pour", "liquid": "water",
                           "x": lava.x, "y": lava.y, "z": lava.z},
                          "pour the carried water onto the lava source to form obsidian")
        stance = _pour_stance(gs, lava)
        if stance is None:
            # Every neighbour of the cell is unusable footing, but the bot is already inside the
            # pool's open pocket: the pour is worth one try from where it stands, which is where
            # the server's own approach leaves it (live, run160 it67: a pour from three blocks off,
            # at the target's own level, formed the run's first obsidian).
            if _within_reach(gs, lava, POUR_REACH):
                return Action(label, "use_bucket",
                              {"action": "pour", "liquid": "water",
                               "x": lava.x, "y": lava.y, "z": lava.z},
                              "pour the carried water into the lava from the open pocket the bot "
                              "already stands in")
            return None
        sx, sy, sz = stance
        return Action(label, "navigate",
                      {"x": sx, "y": sy, "z": sz, "range": 1,
                       "dig": True, "scaffold": False},
                      "walk to the lava source, stand on the block beside it and pour")
    if label == "approach_lava":
        if (not _has(inv, WATER_BUCKET) or not _frame_needs_obsidian(gs, inv)
                or _obsidian_within_reach(gs) or _lava_pour_target(gs) is not None
                or _lava_cast_pool(gs) is not None):
            return None
        stance = _lava_walk_target(gs)
        if stance is None:
            return None
        sx, sy, sz = stance
        return Action(label, "navigate",
                      {"x": sx, "y": sy, "z": sz, "range": 1, "dig": True, "scaffold": True},
                      "walk to the lava the scan can see: the source that forms obsidian only "
                      "shows up once the bot stands beside it")
    if label == "get_flint":
        if _has(inv, FLINT_ITEM):
            return None
        # Flint feeds the flint and steel; afterwards the only sink is the Nether arrows, so the
        # dig reopens in the lull while the arrow reserve is short.
        if _has(inv, FLINT_STEEL_ITEM) and not _arrows_pending(gs):
            return None
        return Action(label, "collect_block", {"block_name": "gravel", "count": FLINT_GRAVEL},
                      "dig gravel for flint to make flint and steel")
    if label == "craft_flint_and_steel":
        # The obsidian floor is the Overworld gate's: an igniter is only worth making while the
        # frame still has to be built. In the Nether the frame is already standing and the
        # obsidian is spent, so the unlit exit read is the reason (live, run195 it451: 0 obsidian
        # in the pack at its own unlit exit frame, no igniter to strike it with).
        if (not _has(inv, FLINT_ITEM) or inv.get("iron_ingot", 0) < FLINT_STEEL_IRON
                or not _has(inv, "crafting_table") or _has(inv, FLINT_STEEL_ITEM)
                or (inv.get("obsidian", 0) < PORTAL_OBSIDIAN
                    and _nether_exit_ignite_cell(gs) is None)):
            return None
        return Action(label, "craft_item", {"items": [{"item_name": FLINT_STEEL_ITEM}]},
                      "craft flint and steel from flint and an iron ingot")
    if label == "clear_frame":
        # A frame cell the build refused (`Agent._note_frame_cells`): a container there is dug by
        # an explicit `collect_block`, the only break the movement policy allows. `_frame_blocker`
        # yields the cell and its name; a cell the frame read has not named yet waits.
        if not _at_base(gs) or _portal_lit(gs) or gs.frame_complete:
            return None
        blocker = _frame_blocker(gs)
        if blocker is None:
            return None
        (x, y, z), name = blocker
        return Action(label, "collect_block",
                      {"block_name": name, "count": 1, "x": x, "y": y, "z": z},
                      "break the block the frame build cannot remove from its cell")
    if label == "clear_portal":
        if (not _at_base(gs) or _portal_lit(gs) or not _has(inv, FLINT_STEEL_ITEM)
                or not gs.portal_blocked):
            return None
        # At most four cells per call: `build_structure` runs one route search per cell and a
        # long list has watchdog-killed the server. The opening is re-read every tick, so a
        # bigger pile is dug over several ticks.
        cells = [{"x": x, "y": y, "z": z, "block_name": "air"}
                 for x, y, z in gs.portal_blocked[:4]]
        return Action(label, "build_structure", {"blocks": cells},
                      "dig the stray blocks out of the portal opening before the frame is built")
    if label == "build_portal":
        site = _portal_site(gs)
        obsidian = inv.get("obsidian", 0)
        can_build = (obsidian >= PORTAL_OBSIDIAN
                     or (obsidian >= 1 and _portal_frame_started(gs)))
        if (site is None or not _at_base(gs) or not _has(inv, FLINT_STEEL_ITEM)
                or _portal_lit(gs) or not can_build or gs.frame_complete
                or inv.get("cobblestone", 0) < PORTAL_CORNERS):
            return None
        return Action(label, "build_structure",
                      {"portal_frame": site, "remove_wrong_blocks": True},
                      "build the obsidian portal frame beside the bot")
    if label == "light_portal":
        if not _has(inv, FLINT_STEEL_ITEM):
            return None
        if _in_nether(gs):
            # The Nether side of the gate: the frame the bot came through, standing unlit. No base
            # and no expedition check - down here the reason to strike it is the way home, and the
            # obsidian is already in the frame, not in the pack.
            cell = _nether_exit_ignite_cell(gs)
            if cell is None:
                return None
            return Action(label, "activate_portal", {"x": cell[0], "y": cell[1], "z": cell[2]},
                          "relight the Nether-side frame that went out, to open the way home")
        cell = _portal_ignite_cell(gs)
        if (_portal_lit(gs) or not _at_base(gs)
                or not _frame_complete_at_base(gs) or gs.portal_blocked
                or cell is None or not _portal_frame_started(gs) or not _expedition_ready(gs)):
            return None
        return Action(label, "activate_portal", {"x": cell[0], "y": cell[1], "z": cell[2]},
                      "light the built portal frame with the carried flint and steel")
    if label == "leave_portal":
        target = _leave_portal_target(gs)
        if target is None:
            return None
        # `navigate` takes whole block cells (`x` and `z` are integers); the step-out target is a
        # cell centre. Sending the float was refused at argument validation, every time, so the
        # rescue that owns the tick could never fire - and because `_in_portal` hands it the tick
        # outright, the run froze on the spot (live, run35: `[Invalid arguments for tool navigate:
        # expected int, received number at x]`, 23 identical ticks with no other candidate).
        tx, ty, tz = (math.floor(v) for v in target)
        # `build: true` is the part the rescue was missing. A step-out beside a Nether gate is
        # often mid-air: the frame stands on the floor, and the cell two blocks off the body hangs
        # above it. The server refuses a target with nothing under it unless the caller says the
        # bot means to scaffold there on purpose (live, run198 it0-599: six hundred identical
        # `[NAVIGATION_TARGET_UNSUPPORTED] ... The ground in that column is at y=71`, the body
        # never leaving the obsidian it stood on). Digging stays off: the frame around the opening
        # must not be broken.
        return Action(label, "navigate",
                      {"x": tx, "y": ty, "z": tz, "dig": False, "scaffold": True, "build": True},
                      "step out of the lit portal the body is standing in, before it teleports")
    if label == "take_bed":
        if not _overworld(gs):
            return None
        # Two reasons to take a bed up: it stands too close to the lit gate to respawn on, or an
        # earlier base left it outside the box. The second used to require carrying no bed, on the
        # assumption `place_bed` could lay a fresh one; `sleep` disagrees, finding the standing bed
        # within 32 blocks and walking to it, so a carried bed stays carried and the run trades
        # `place_bed` and `go_home` for hundreds of ticks (live, run156 it214-600).
        if not (_unsafe_beds(gs)
                or (_bed_needs_moving(gs)
                    and (_nether_idle(gs) or (_portal_lit(gs) and _at_base(gs))))):
            return None
        return Action(label, "collect_block",
                      {"block_name": "white_bed", "count": 1},
                      "take the standing respawn bed back up so it can be laid at the base")
    if label == "place_bed":
        stance = _bed_stance(gs)
        if (not _has_bed(inv) or not _portal_lit(gs) or _bed_at_base(gs)
                or stance is None
                or not _at_base(gs) or not _overworld(gs)):
            return None
        if _bed_needs_moving(gs) and "take_bed" not in gs.stuck_labels:
            # `sleep` would walk to the bed already standing off base instead of laying the
            # carried one at the stance; the take step runs first.
            return None
        if not _at_stance(gs, stance):
            # The `sleep` server lays the carried bed on the flat footprint beside the bot, so the
            # bot has to *stand* where the bed may go before it may sleep: walking to the clear
            # stance is half the step, and the very next tick lays the bed there.
            sx, sy, sz = stance
            # `range: 0` because the bed goes down *where the bot stands*: with the server's
            # default arrival tolerance of one block, a bot already next to the stance is
            # reported as arrived without ever crossing into it, and the next tick's `sleep`
            # lays the bed beside the wrong cell (live, run120: 5 `place_bed` ticks at the same
            # position, then the idle guard withheld the label and the run spent its remaining
            # 40 ticks on `wait`).
            return Action(label, "navigate",
                          {"x": sx, "y": sy, "z": sz, "range": 0,
                           "dig": True, "scaffold": True},
                          "walk onto the spot clear of the gate, where the respawn bed goes")
        return Action(label, "sleep", {},
                      "place the carried bed clear of the gate and set the respawn point")
    if label == "sleep":
        if gs.phase != "night" or gs.sleeping or not _overworld(gs) or not _sleep_safe(gs):
            return None
        if _sleep_off_base(gs):
            stance = _sleep_off_base_stance(gs)
            if stance is None:
                return None
            sx, sy, sz = stance
            return Action(label, "navigate",
                          {"x": sx, "y": sy, "z": sz, "range": 1,
                           "dig": True, "scaffold": True},
                          "walk off the base, away from the unlit gate frame, to sleep there")
        if not _sleep_ready(gs):
            return None
        if _bed_at_base(gs) and not _bed_reachable(gs):
            # The bed is there; the body is not next to it. `sleep` clicks the bed block, and
            # vanilla only takes that click from within two blocks of the bed's top centre, so a
            # bot standing two cells off answers `[BED_SLEEP_REJECTED] bot is not sleeping` while
            # the server names the bed it tried to use (live, run121 it100-103: four identical
            # refusals at (-9.3,70,-52.5) against a bed at (-10,70,-51)). `range: 0` for the same
            # reason as `place_bed`'s stance walk: the arrival has to be *in* that cell.
            stance = _bed_reach_stance(gs)
            if stance is None:
                return None
            sx, sy, sz = stance
            return Action(label, "navigate",
                          {"x": sx, "y": sy, "z": sz, "range": 0,
                           "dig": True, "scaffold": True},
                          "step onto the footing beside the gate bed, close enough to click it")
        return Action(label, "sleep", {},
                      "sleep through the night and set the respawn point")
    if label == "craft_bed":
        if not _overworld(gs):
            # Nothing can lay a bed in the Nether, so there is no reason to make one there.
            return None
        if not _has(inv, "crafting_table") or not _bed_leaves_ready(inv) \
                or (_logs(inv) < 1 and _planks(inv) < 3):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "white_bed"}], "temporary_workstation": True},
                      "craft a bed from white wool or string and planks")
    if label == "hunt_wool":
        if _bed_leaves_ready(inv):
            return None
        if not _hunt_ok(gs):
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "sheep", "drop_name": "white_wool", "count": 3},
                      "collect white wool from a sheep for a bed")
    if label == "hunt_string":
        goal = _string_goal(gs)
        # The widest guard that offers this label, not the strictest. `_safe_string_hunt` is the
        # bow's own rule (one spider, wider reach, a little company) and the bed chain offers
        # through it too; `resolve` used to require the stricter `_safe_spider_hunt` instead, so
        # every offer the bow chain made was refused on the same tick and escalated.
        if goal <= 0 or inv.get("string", 0) >= goal or not _safe_string_hunt(gs):
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "spider", "drop_name": "string",
                       "count": max(1, goal - inv.get("string", 0)),
                       "allow_without_shield": True},
                      "collect string from a lone spider")
    if label == "hunt_feather":
        if not _arrows_pending(gs) or _has(inv, FEATHER_ITEM) or gs.mob_nearest("chicken") is None:
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "chicken", "drop_name": FEATHER_ITEM, "count": 4,
                       "allow_without_shield": True},
                      "collect feathers from a chicken for the Nether arrows")
    if label == "mine_gold":
        if (not _can_mine(gs, GOLD_ORE) or not _gold_ore_available(gs) or _has(inv, RAW_GOLD)
                or inv.get(GOLD_INGOT, 0) >= GOLD_HELMET_INGOTS):
            return None
        return Action(label, "collect_block",
                      {"block_name": GOLD_ORE, "count": _gold_goal(inv)},
                      "mine gold ore for the helmet that keeps piglins neutral")
    if label == "mine_gold_reserve":
        if (not _gold_reserve_short(gs)
                or not (_has(gs.inventory, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM))
                or not _can_mine(gs, GOLD_ORE) or not _gold_ore_available(gs)):
            return None
        want = GOLD_BARTER_RESERVE - inv.get(GOLD_INGOT, 0) - inv.get(RAW_GOLD, 0)
        if want <= 0:
            return None
        return Action(label, "collect_block",
                      {"block_name": GOLD_ORE, "count": want},
                      f"mine gold ore for the {want} ingots the piglin pearl barter needs")
    if label == "smelt_gold":
        fuel = _furnace_fuel(inv)
        if inv.get(RAW_GOLD, 0) < 1 or fuel is None:
            return None
        if not _furnace_room(gs):
            return None
        if inv.get(GOLD_INGOT, 0) + inv.get(RAW_GOLD, 0) >= _gold_ingots_wanted(gs):
            return None
        fuel_name, capacity = fuel
        # Outside the lull only coal/charcoal may be spent (never wood the chain still wants).
        if fuel_name not in ("coal", "charcoal") and not _nether_idle(gs):
            return None
        count = min(inv.get(RAW_GOLD, 0), 3, capacity,
                    max(0, _gold_ingots_wanted(gs) - inv.get(GOLD_INGOT, 0)))
        if count < 1:
            return None
        # A carried furnace (temporary workstation) always wins: no walking home, no waiting
        # for a base furnace that may be hundreds of blocks away or not standing at all.
        if _has(inv, "furnace"):
            return Action(label, "smelt_item",
                          _smelt_args(gs, RAW_GOLD, count, fuel_name),
                          f"smelt raw gold with carried furnace using {fuel_name}")
        # No portable furnace: at base, use the base furnace if its position is known.
        if _at_base(gs) and gs.furnace_pos is not None:
            fx, fy, fz = gs.furnace_pos
            return Action(label, "smelt_item",
                          {"item_name": RAW_GOLD, "count": count, "fuel_item_name": fuel_name,
                           "x": fx, "y": fy, "z": fz},
                          f"smelt raw gold at base furnace using {fuel_name}")
        # Otherwise there is nothing to smelt at: craft a furnace first.
        return resolve("craft_furnace", gs)
    if label == "craft_gold_helmet":
        if (inv.get(GOLD_INGOT, 0) < GOLD_HELMET_INGOTS or not _has(inv, "crafting_table")
                or _has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": GOLD_HELMET_ITEM}], "temporary_workstation": True},
                      "craft the gold helmet that keeps piglins neutral in the Nether")
    if label == "enter_portal":
        cell = _exit_portal_cell(gs)
        if cell is None or not _nether_departure_ready(gs) or not _at_base(gs):
            return None
        x, y, z = cell
        # `allow_low_supplies` on purpose. The server pauses a crossing below 16 food *and* 32
        # arrows (`portal-policy.ts`), and this run banks `NETHER_ARROWS` = 16 - half of what the
        # gate wants - so without the override the crossing is refused forever and the expedition
        # cannot start at all (live, run41: `[PORTAL_LOW_SUPPLIES] 38/16 food ... 0/32 arrows`).
        # The food half of that guard is not given away: `_expedition_prepared` already holds the
        # run to the deep reserve, and the arrow half is the defensive reserve for a trip whose
        # goal - blaze rods - is fought with a sword and a shield.
        return Action(label, "enter_nether_portal",
                      {"x": x, "y": y, "z": z, "allow_low_supplies": True},
                      "cross into the Nether through the lit gate at the base")
    if label == "swap_to_gold_helmet":
        if not _gold_helmet_needed(gs):
            return None
        return Action(label, "equip", {"items": [{"item_name": GOLD_HELMET_ITEM,
                                                  "destination": "head"}]},
                      "wear the gold helmet so piglins stay neutral; the diamond one goes to the pack")
    if label == "hunt_blaze":
        if not _in_nether(gs) or not _safe_blaze_hunt(gs):
            return None
        want = _rods_target(gs) - _blaze_rods(inv)
        return Action(label, "collect_mob_drop",
                      {"mob_name": BLAZE_MOB, "drop_name": BLAZE_ROD_ITEM,
                       "count": min(3, want), "allow_without_shield": True},
                      f"kill a blaze for its blaze rod ({want} still needed)")
    if label == "barter_pearls":
        if not _barter_pearls_wanted(gs):
            return None
        near = gs.piglin_nearest()
        if near is None:
            return None
        piglin_id, _ = near
        want = min(_pearls_owed(gs), BARTER_PEARL_BATCH)
        budget = min(gs.inventory.get(GOLD_INGOT, 0), BARTER_GOLD_BUDGET)
        if want <= 0 or budget <= 0:
            return None
        return Action(label, "barter",
                      {"piglin_id": piglin_id, "item_name": ENDER_PEARL_ITEM,
                       "count": want, "gold_budget": budget},
                      f"trade gold with a piglin for {want} ender pearls ({_pearls_owed(gs)} still owed)")
    if label == "leave_nether":
        if not _in_nether(gs):
            return None
        cell = _exit_portal_cell(gs)
        if cell is not None:
            x, y, z = cell
            return Action(label, "enter_nether_portal", {"x": x, "y": y, "z": z},
                          "cross back through the Nether exit portal with the rods")
        # No portal block in the scan: the way home is the cell the crossing left the bot
        # standing on - this run watched that crossing (`nether_arrival`), or the server
        # remembers it (`nether_gate`, `_note_nether_gate`). Walk there first; the crossing
        # itself waits until the portal is loaded again. Neither known is no way home walked:
        # run74 walked 250 blocks to its own start position and waited at a non-portal until
        # the bot died - a guess is not the gate, and standing at the wrong cell is how the
        # trip never came back.
        home = _nether_way_home(gs)
        if home is None:
            return None
        ax, ay, az = home
        # Already standing on it: cross, do not walk. `navigate` into an active portal is refused
        # outright - `[NAVIGATION_PORTAL_INTENT_REQUIRED] Use enter_nether_portal to enter this
        # active portal` - and the scan does not always list the block the body is standing in
        # (live, run70: no `nether_portal` hit at all at (226,49,222), and ten identical
        # refusals, so the bot stood in its own gateway for the rest of the run). Arrival first,
        # then the crossing, is the server's own two-step.
        if gs.pos is not None and math.dist((gs.pos[0], gs.pos[2]), (ax, az)) <= 2.0:
            return Action(label, "enter_nether_portal", {"x": ax, "y": ay, "z": az},
                          "cross back through the Nether exit portal with the rods")
        # The walk never aims at the portal cell itself: `navigate` refuses an active portal cell
        # outright - `[NAVIGATION_PORTAL_INTENT_REQUIRED]`, whatever the `range` - so a target
        # equal to the arrival cell is refused at step zero and the bot never moves (live, run99
        # it347-599: 178 refusals, the way home never walked, the bot starving at a gate it could
        # see but never reach). Aim `NETHER_GATE_APPROACH` blocks short on the bot's own side: a
        # standable doorstep clear of the portal's two-wide interior and frame, close enough that
        # the portal block loads into the scan and `_exit_portal_cell` crosses on the next tick.
        # When far from the gate (> NETHER_EXIT_STAGING_DISTANCE), stage the exit in hops rather
        # than one long walk: pathfinder search times out on 450+ block walks through lava/sparse
        # terrain, so navigate in ~120 m hops. Each hop is a tick toward the gate, reloading the
        # world and resetting the search budget.
        # `build: True` is what makes a hop a step instead of a refusal. A hop aims at the height
        # the body stands on, and in the Nether the cell 120 m along that line is usually air over
        # a lava sea: run203 logged 23 `[NAVIGATION_TARGET_UNSUPPORTED] ... No ground was found
        # within 64 blocks below it` on `leave_nether` alone, the body never moving. The server
        # accepts a mid-air target only when the caller says the bot means to scaffold to it (the
        # same lesson `leave_portal` learned in run198).
        tx, tz, ty = float(ax), float(az), ay
        if gs.pos is not None:
            dx, dz = gs.pos[0] - ax, gs.pos[2] - az
            span = math.hypot(dx, dz)
            if span > 1e-6:
                hop = _exit_hop(gs)
                if span > hop:
                    # A hop aims at the height the body is standing on, not at the height the gate
                    # came in at: live, run149 the 120 m hop aimed at y 66 while the bot stood at
                    # y 37-77, and the terrain refused the same cell sixteen times.
                    # `dx`/`dz` point from the gate to the body, so the hop *subtracts* them:
                    # adding walked the bot away from the exit (live, run151 it13-41: ten
                    # "hops" from (189,299) to (859,1187), each ~100 blocks deeper into the
                    # Nether, the gate at (1,74,16) behind it).
                    tx = gs.pos[0] - dx / span * hop
                    tz = gs.pos[2] - dz / span * hop
                    ty = math.floor(gs.pos[1])
                else:
                    tx = ax + dx / span * NETHER_GATE_APPROACH
                    tz = az + dz / span * NETHER_GATE_APPROACH
        return Action(label, "navigate",
                      {"x": round(tx), "y": ty, "z": round(tz),
                       "range": 2, "dig": True, "scaffold": True, "build": True},
                      "walk toward the Nether exit portal (staged in hops if far away)")
    if label == "swap_to_diamond_helmet":
        # The offer and the resolve agree: while a crossing is pending the gold piece stays on,
        # so an escalated diamond swap cannot put the bot back in front of the piglins.
        if _in_nether(gs) or not _worn(gs, GOLD_HELMET_ITEM) or _nether_departure_ready(gs):
            return None
        if not (_has(inv, "diamond_helmet") or _worn(gs, "diamond_helmet")):
            return None
        return Action(label, "equip", {"items": [{"item_name": "diamond_helmet",
                                                  "destination": "head"}]},
                      "wear the diamond helmet again, now that the bot is back in the Overworld")
    if label == "craft_blaze_powder":
        # Any rod in the pack is worth crafting while the eyes still owe powder: the old gate
        # demanded the whole haul (`_rods_target`) before it would craft, so a single rod left
        # over from an earlier run could never become powder, and with it never leave the pack -
        # the gate stayed shut on a haul that could not be processed (run123, ten `wait` ticks
        # at the base with one rod and a burning portal).
        if (_in_nether(gs) or _blaze_rods(inv) <= 0 or not _has(inv, "crafting_table")
                or _blaze_powder(inv) >= _powder_target(gs)):
            return None
        want = min(2 * _blaze_rods(inv), _powder_target(gs) - _blaze_powder(inv))
        if want <= 0:
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": BLAZE_POWDER_ITEM, "count": want}],
                       "temporary_workstation": True},
                      f"craft the carried blaze rods into {want} blaze powder, one rod to two")
    if label == "stow_blaze_powder":
        if _in_nether(gs) or _blaze_powder(inv) <= 0 or not _chest_ready(gs):
            return None
        # The pack's powder is the next eye, not a haul: banking it here is the tick the model
        # would have spent crafting, and the pair of offers either way is the ping-pong.
        if _powder_is_eye_fuel(gs):
            return None
        # The blaze powder belongs in a chest of the network that can still take it: the main
        # one may well be the one a stow filled up, and powder deposited into a full chest is
        # powder held in the pack on the way back through the gate.
        cell = _storage_target(gs)
        if cell is None:
            # Every chest in the network has been called full. Depositing into a closed lid is a
            # refusal the stuck guard counts (run124 it54: `[CONTAINER_BATCH_INCOMPLETE] ...
            # blaze_powder 0/2`), and the powder in the pack is supply the eyes can spend directly
            # rather than a haul the network still owes.
            return None
        if not _near_cell(gs, cell):
            wx, wy, wz = cell
            return Action(label, "navigate", {"x": wx, "y": wy, "z": wz},
                          "walk over to the chest at the portal to bank the blaze powder")
        cx, cy, cz = cell
        powder = _blaze_powder(inv)
        return Action(label, "use_container",
                      {"operation": "deposit", "x": cx, "y": cy, "z": cz,
                       "items": [{"item_name": BLAZE_POWDER_ITEM, "count": powder}]},
                      f"deposit {powder} blaze powder into the chest at the portal")
    if label == "take_stored_food":
        if _in_nether(gs) or not _at_base(gs) or not _chest_ready(gs):
            return None
        want = FOOD_BANK_KEEP - _edible_count(inv)
        if want <= 0:
            return None
        # Named from what a chest last held, not from a wish list: a `withdraw` naming an item
        # the chest does not hold takes nothing out, and the tick is then a refusal that looks
        # exactly like a working chest. And now per chest rather than in the aggregate: with a
        # network the provisions can stand in the overflow, not in the main chest, and emptying
        # the aggregate into one cell's coordinates withdraws from the wrong lid.
        stores = _banked_stores(gs)
        if stores:
            cell, held = stores[0]
            if not _near_cell(gs, cell):
                cx, cy, cz = cell
                return Action(label, "navigate", {"x": cx, "y": cy, "z": cz},
                              "walk over to the chest at the base that holds the banked provisions")
        else:
            # Before the first per-cell read (or on an old logged state): what the run knew
            # before networks existed -- the aggregate, taken at the one chest.
            stored = _banked_provisions(gs)
            if not stored:
                return None
            cell, held = tuple(gs.chest_pos), stored
            if not _near_cell(gs, cell):
                cx, cy, cz = cell
                return Action(label, "navigate", {"x": cx, "y": cy, "z": cz},
                              "walk over to the chest at the base that holds the banked provisions")
        items: list[dict] = []
        left = want
        for name, count in held.items():
            if left <= 0:
                break
            take = min(int(count), left, 64)
            if take > 0:
                items.append({"item_name": name, "count": take})
                left -= take
        if not items:
            return None
        cx, cy, cz = cell
        return Action(label, "use_container",
                      {"operation": "withdraw", "x": cx, "y": cy, "z": cz, "items": items},
                      f"take {sum(i['count'] for i in items)} edible items out of the chest at the "
                      "base so the pack is fed for the next trip without hunting again")
    if label == "hunt_food":
        if _provisioned(gs) >= _food_goal(gs):
            return None
        # The offer's weapon rule lives here too, so a forced label cannot walk the unarmed
        # hunt that the offer itself refuses (`_hunt_ok`).
        if not _hunt_ok(gs):
            return None
        target = _food_target(gs)
        if not target:
            return None
        mob, drop = target
        count = _hunt_count(gs, mob)
        return Action(label, "collect_mob_drop",
                      {"mob_name": mob, "drop_name": drop, "count": count,
                       "allow_without_shield": True},
                      f"hunt a {mob} for {drop}")
    if label == "craft_bowl":
        # The same test as the offer in `_all_nether_prep_steps`, so a forced label cannot craft
        # a bowl out of planks the pack no longer has.
        if _has(inv, BOWL_ITEM) or _plank_units(inv) < 3 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": BOWL_ITEM}], "temporary_workstation": True},
                      "craft the stew bowl at the base before crossing: the Nether has no wood, "
                      "and every mushroom meal needs a bowl that comes back")
    if label == "craft_stew":
        # Two steps in one label: fetch the mushrooms the batch still lacks, or brew once the
        # pack covers it. The bowl is never fetched separately - `craft_item` plans the bowl from
        # the carried planks on its way to the stew, and the server's own planner knows that
        # recipe. That bowl craft is the only part of the plan that owes a workstation: the stew
        # itself is shapeless and fits the 2x2 grid, so a carried bowl needs no table.
        #
        # The batch the pack is asked for is capped by what it holds. Asking for `STEW_BATCH`
        # with one red and one brown carried is the dead end of run170 it32-35: four identical
        # `[CRAFT_MATERIALS_MISSING] Missing leaf materials: brown_mushroom x1, red_mushroom x1`
        # ticks until the stuck guard took `craft_stew` off the offer. A short pack picks the
        # shortfall instead, so the pantry is grown rather than refused.
        need = _stew_need(gs) or 1
        missing = _stew_missing(gs, need)
        if missing is not None:
            hit = gs.resource_nearest(missing)
            brew = _stew_brewable(gs)
            if hit is not None:
                short = max(1, need - int(inv.get(missing, 0) or 0))
                return Action(label, "collect_block",
                              {"block_name": missing, "count": short},
                              f"pick {short} {missing} for the stew batch; the bowl and the "
                              "other mushroom are already in the pack")
            if brew <= 0:
                return None
        else:
            brew = _stew_brewable(gs)
            if brew <= 0:
                return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": STEW_ITEM, "count": brew}],
                       **({"temporary_workstation": True} if not _has(inv, BOWL_ITEM) else {})},
                      "brew mushroom stew from the bowl and both mushrooms - the bowl comes back "
                      "from every meal, so this renews itself")
    if label == "gather_mushrooms":
        kind = _mushroom_short(gs)
        if kind is None or not _bowl_available(gs):
            return None
        if gs.resource_nearest(kind) is None:
            return None
        carried = int(inv.get(kind, 0) or 0)
        # The batch is the most one call can dig and the most the stack can still take, never a
        # demand for the cap: the server mines what the loaded ground actually holds and reports
        # what came back, and the next tick decides whether to keep digging. Until the kit brews
        # (a bowl and one of each kind) the batch is a seed, not a stack: filling one kind first
        # is how run193 spent its last free slot on 32 red mushrooms while the brown that would
        # have completed the recipe stood 41 m away.
        batch = MUSHROOM_BATCH if _stew_kit(gs) else MUSHROOM_SEED
        count = max(1, min(batch, MUSHROOM_CAP - carried))
        return Action(label, "collect_block",
                      {"block_name": kind, "count": count},
                      f"pick up to {count} {kind} for the stew pantry; they regrow where they "
                      "stand and every stew returns its bowl")
    if label == "collect_smelted":
        if not _furnace_needs_draining(gs) or not _at_base(gs) or not _furnace_room(gs):
            return None
        fx, fy, fz = gs.furnace_pos
        if not gs.furnace_contents:
            # Nothing known to be sitting in it yet: look before reaching in, so a cook is only
            # interrupted by a furnace that really is occupied.
            return Action(label, "use_container",
                          {"operation": "inspect", "x": fx, "y": fy, "z": fz},
                          "look at the furnace at the base to see whether it still holds output")
        items = [{"item_name": n, "count": int(c)} for n, c in gs.furnace_contents.items()
                 if int(c) > 0]
        if not items:
            return None
        return Action(label, "use_container",
                      {"operation": "withdraw", "x": fx, "y": fy, "z": fz, "items": items[:12]},
                      "take everything out of the furnace at the base so the next cook can start")
    if label == "cook_food":
        raw = _first_cookable(inv)
        fuel = _furnace_fuel(inv)
        if not raw or fuel is None or not _furnace_room(gs):
            return None
        # The station rule lives in `cooking_reach`, the same one the offer asks, so the two
        # cannot drift: run78's eight `[WORKSTATION_NOT_CARRIED]` came precisely from this
        # line accepting a furnace standing 200 m away while the cook asked for a temporary
        # workstation it could not carry.
        if not _cooking_reach(gs):
            return None
        station = _cooking_station(gs)
        # Outside the lull only coal/charcoal may be spent (never wood the chain still wants).
        if fuel[0] not in ("coal", "charcoal") and not _nether_idle(gs):
            return None
        # Wood in the lull is spent only on a reserve that is actually short — the same gate the
        # offer carries, so the two cannot drift (see the `cook_food` offer).
        if (fuel[0] not in ("coal", "charcoal")
                and _provisioned(gs) >= _food_goal(gs)):
            return None
        fuel_name, capacity = fuel
        return Action(label, "smelt_item",
                      _smelt_args(gs, raw, min(inv.get(raw, 0), 3, capacity), fuel_name, station),
                      f"cook {raw} into {COOKED[raw]} using {fuel_name}")
    if label == "eat_food":
        food = _first_edible(inv)
        if not food:
            return None
        return Action(label, "eat_food", {"food_name": food},
                      f"eat {food} to restore hunger")
    if label == "replace_tool":
        item = _replacement_pickaxe(gs, inv)
        if not item or not _pickaxe_due_for_spare(gs):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": item}], "temporary_workstation": True},
                      f"carry a fresh {item} before the active one breaks mid-dig")
    if label == "collect_drops":
        if gs.drops <= 0:
            return None
        return Action(label, "pick_up_items", {"radius": 16},
                      "collect items a mob dropped when the reflex killed it")
    if label == "recover_drops":
        if not _can_recover(gs):
            return None
        return Action(label, "pick_up_items", {"recover_death_items": True},
                      "walk back to the death site and recover the dropped gear")
    if label == "make_room":
        if gs.pos is None:
            return None
        # A placement the terrain refused at a named cell (`[PLACE_TARGET_OCCUPIED]`): dig that
        # exact cell. The block standing in it is the whole cause, and the pocket below would
        # not even touch it (the chest cell sits beside the base, not beside the bot).
        if gs.blocked_cell is not None and gs.blocked_cell not in gs.placement_refused:
            x, y, z = gs.blocked_cell
            cells = [{"x": x, "y": y, "z": z, "block_name": "air"}]
            return Action(label, "build_structure", {"blocks": cells},
                          "dig the block out of the cell the placement was refused at")
        # Otherwise one small pocket at the bot's own level is enough: each cleared side cell
        # has the tunnel floor under it, so the placement search finds a supported, replaceable
        # cell within reach. Kept to four cells on purpose -- the tool cost grows with the list.
        mx, my, mz = (math.floor(v) for v in gs.pos)
        cells = [{"x": mx + dx, "y": my, "z": mz + dz, "block_name": "air"}
                 for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        return Action(label, "build_structure", {"blocks": cells},
                      "dig a small pocket around the bot for the carried workstation or bed")
    if label == "dig_out":
        if gs.pos is None or not gs.boxed_in:
            return None
        if gs.on_ground is False:
            # The tunnel is the same `build_structure` call that hung the runtime event loop
            # around a body the server reported as not standing (see `graph.unblock_action`,
            # run115 it527 and run116 it0): a dig needs feet, and the step that gets them back
            # owns the tick.
            return None
        mx, my, mz = (math.floor(v) for v in gs.pos)
        # Toward the base first, where one is anchored; each further try at the same spot turns
        # to the next side, so a wall that will not give is not dug at twice.
        sides = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if gs.base is not None:
            bx, bz = gs.base[0] - mx, gs.base[2] - mz
            sides.sort(key=lambda d: -(d[0] * bx + d[1] * bz))
        dx, dz = sides[gs.dig_out_try % len(sides)]
        # Feet and head height, three cells long: a corridor the bot walks out along, short
        # enough to dig by hand (the shut-in bot of run90 carried no tool at all).
        cells = [{"x": mx + dx * k, "y": my + h, "z": mz + dz * k, "block_name": "air"}
                 for k in (1, 2, 3) for h in (0, 1)]
        return Action(label, "build_structure", {"blocks": cells},
                      "dig a short tunnel out of the pocket the pathfinder cannot leave")
    if label == "surface_up":
        if (gs.base is not None or not _gear_ready(gs) or gs.pos is None or _at_surface(gs)
                or _in_nether(gs)):
            return None
        # Don't try to build base from water - need to get to dry land first
        if gs.in_water:
            return None
        mx, mz = math.floor(gs.pos[0]), math.floor(gs.pos[2])
        # A column the probe already answered `[NAVIGATION_TARGET_WATER]` has no way up: its
        # top block is a fluid and the server's shore search reaches only 8 blocks, so the same
        # probe fails the same way every tick (live, run142 it55-57: three identical refusals at
        # (5.51, -19, -115.34), the body never moving, until an unrelated `get_flint` broke the
        # loop). The climb then belongs to the nearest column the ground scan shows has dry
        # open-sky footing, reached by digging and scaffolding the way up to it.
        if (mx, mz) in gs.surface_wet:
            foot = _dry_surface_foot(gs, mx, mz)
            if foot is None:
                return None
            fx, fy, fz = foot
            return Action(label, "navigate", {"x": fx, "y": fy, "z": fz,
                                             "dig": True, "scaffold": True},
                          "climb out onto the nearest dry column's open surface ground")
        # A `navigate` without a height asks the server for a column's surface. It walks the bot
        # up to open sky when the answer is unambiguous, and otherwise stops with
        # `[NAVIGATION_HEIGHT_REQUIRED]`, naming the surface it found. That answer is read back
        # in `graph._verify`, so the climb itself runs on the next tick -- once with the height,
        # which the pathfinder needs. Re-probe whenever the bot has moved to a new column: the
        # surface of a different column can be tens of blocks higher or lower.
        if gs.surface is None or (mx, mz) != (gs.surface[0], gs.surface[2]):
            return Action(label, "navigate", {"x": mx, "z": mz},
                          "ask the server for this column's surface height")
        sx, sy, sz = gs.surface
        # Don't climb to a surface that's far below current level if we're near water
        # (could be seabed, not suitable for base)
        if gs.pos is not None and sy < gs.pos[1] - 10:
            # If surface is much lower, it's likely not the open air surface we want
            # Clear it and re-probe
            gs.surface = None
            return Action(label, "navigate", {"x": mx, "z": mz},
                          "ask the server for this column's surface height (re-probe, likely seabed)")
        return Action(label, "navigate",
                      {"x": sx, "y": sy, "z": sz, "dig": True, "scaffold": True},
                      "climb back onto open surface ground before the base is anchored")
    if label == "establish_base":
        if gs.base is not None or not _gear_ready(gs) or not _at_surface(gs):
            return None
        # Prefer the level, clear site the base scan found (`flat_build_site`): the portal
        # frame only fits on ground that is flat along its whole bottom row. Fall back to the
        # nearest standable surface when no such site was in scan reach.
        if gs.build_site is not None:
            bx, by, bz = gs.build_site
            return Action(label, "navigate",
                          {"x": bx, "y": by + 1, "z": bz, "dig": True, "scaffold": True},
                          f"walk to the level ground at {bx},{by + 1},{bz} and set the base")
        target = base_target(gs)
        if target is None:
            return None
        tx, ty, tz = target.pos
        return Action(label, "navigate",
                      {"x": tx, "y": ty + 1, "z": tz, "dig": True, "scaffold": True},
                      f"walk up to the surface at {tx},{ty + 1},{tz} and set the base")
    if label == "go_home":
        if gs.base is None or _at_base(gs) or _in_nether(gs) or _stronghold_phase(gs):
            # The stronghold phase never walks home: a night at a base with a bed is the strongest
            # `go_home` reason there is, and it is exactly the ping-pong that keeps the trek from
            # arriving (live, run176: the first action after the restart was the home walk).
            return None
        # Two reasons walk the bot home: base work waits (`_go_home_wanted`), or it is the idle
        # geared Overworld bot the empty-candidate fallback sends home (`_idle_off_base`). The
        # fallback reason is honoured only when it is in fact the fallback - `go_home` the single
        # thing on offer - so a tick with real work the bot was *not* sent home for cannot bind a
        # home walk here (the `_go_home_wanted` two-gates note). `candidates` is pure and never
        # calls `resolve`, so re-reading it in this one narrow branch is safe.
        if not _go_home_wanted(gs) and not (
                _idle_off_base(gs) and candidates(gs) == ["go_home"]):
            return None
        bx, by, bz = gs.base
        return Action(label, "navigate",
                      {"x": bx, "y": by + 1, "z": bz, "dig": True, "scaffold": True},
                      "walk back to the home base to place the bed, the portal or a chest")
    if label == "craft_chest":
        if (gs.base is None or _has(inv, CHEST_ITEM) or gs.chest_blocked
                or not _has(inv, "crafting_table") or _plank_units(inv) < CHEST_PLANKS):
            return None
        if _chest_ready(gs) and not _storage_full(gs):
            # A chest still stands with room somewhere in the network: the surplus goes there,
            # and a third chest would only gather dust beside the two that work.
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": CHEST_ITEM}], "temporary_workstation": True},
                      ("craft another chest from eight planks: the storage network at the base "
                       "is full through" if _chest_ready(gs) else
                       "craft a chest from eight planks to store surplus at the base"))
    if label == "place_chest":
        if gs.base is None or gs.chest_blocked or not _has(inv, CHEST_ITEM):
            return None
        cell = _chest_site(gs)
        if cell is None:
            return None
        cx, cy, cz = cell
        if not _chest_ready(gs):
            return Action(label, "place_block",
                          {"block_name": CHEST_ITEM, "x": cx, "y": cy, "z": cz},
                          "place the chest at the home base")
        if not _storage_full(gs):
            # The first chest is not even up yet, or one of them has room: an overflow chest
            # placed with nobody waiting for it is an item spent on furniture.
            return None
        return Action(label, "place_block",
                      {"block_name": CHEST_ITEM, "x": cx, "y": cy, "z": cz},
                      "set up another chest beside the base: the storage network is full through")
    if label == "place_furnace":
        args = _furnace_args(gs)
        if (_furnace_ready(gs) or gs.furnace_blocked or not _has(inv, "furnace")
                or args is None):
            return None
        return Action(label, "place_block", args,
                      "place the carried furnace at the home base so cooking reuses it")
    if label == "stow_surplus":
        items = [{"item_name": n, "count": inv[n]}
                 for n in _junk_carried(inv) + _surplus_besides_keep(gs, inv)
                 + _spare_gear(gs, inv)]
        items += _hoard_surplus(gs, inv)
        items += _bank_surplus_food(gs, inv)
        if not _chest_ready(gs) or not items:
            return None
        # A block sitting on the chest stops its lid (`[CHEST_BLOCKED]`): dig that exact cell
        # first, then open the chest on the next tick. Reusing the label keeps the relief to one
        # step; the deposit runs as soon as the cell above the chest is clear. A container
        # (crafting_table, furnace, chest) cannot be removed by `build_structure` (the server
        # answers `[BUILD_INCOMPLETE]`), so it is broken with `collect_block` instead, the same
        # tool `clear_frame` uses for a refused frame cell.
        if gs.chest_above is not None:
            ax, ay, az = gs.chest_above
            name = getattr(gs, "chest_above_name", None)
            if name:
                return Action(label, "collect_block",
                              {"block_name": name, "count": 1, "x": ax, "y": ay, "z": az},
                              f"break the {name} at {ax},{ay},{az} that blocks the chest lid")
            return Action(label, "build_structure",
                          {"blocks": [{"x": ax, "y": ay, "z": az, "block_name": "air"}]},
                          f"dig the block at {ax},{ay},{az} that blocks the chest lid")
        target = _storage_target(gs)
        if target is None:
            # Every chest in the network is full: depositing anywhere would only earn the same
            # refusal again. The offer grows the network instead (run79's 8×
            # `[CONTAINER_BATCH_INCOMPLETE]` and the `wait` chain that followed).
            return None
        if not _near_cell(gs, target):
            tx, ty, tz = target
            return Action(label, "navigate", {"x": tx, "y": ty, "z": tz},
                          "walk over to the storage chest at the base that still has room")
        cx, cy, cz = target
        return Action(label, "use_container",
                      {"operation": "deposit", "x": cx, "y": cy, "z": cz, "items": items[:12]},
                      "deposit the surplus blocks into the chest at the base")
    if label == "prepare_bed_pad":
        if not _bed_pad_wanted(gs):
            return None
        # At most six cells - three digs and three fills - so no batch cap: `build_structure` runs
        # one route search per cell, and the row is re-read with the footprint scan every tick.
        cells = [{"x": x, "y": y, "z": z, "block_name": name}
                 for x, y, z, name in gs.bed_pad]
        if not cells:
            return None
        return Action(label, "build_structure", {"blocks": cells},
                      "build the pad the gate bed lies on: dig the blocks in its footprint, stand cobblestone under the cells with no floor")
    if label == "flatten_home_base":
        if not _base_leveling_wanted(gs):
            return None
        cells = [{"x": x, "y": y, "z": z, "block_name": "air"}
                 for x, y, z in gs.raised_cells[:FLATTEN_BATCH]]
        if not cells:
            return None
        # One route search per cell, so the batch is capped (`clear_portal`'s reason); the scan
        # re-reads each tick and the rest falls to the next one.
        return Action(label, "build_structure", {"blocks": cells},
                      "level the home base: dig the stray blocks piled above ground level around the gate")
    if label == "step_away_drops":
        if not gs.drop_open_pending or gs.drop_pos is None or gs.pos is None:
            return None
        if gs.drop_away is not None:
            # A cell the ground read proved standable, far enough that the junk cannot be in
            # reach whichever way the toss flew (`targets.step_away_cell`).
            ax, ay, az = gs.drop_away
            return Action(label, "navigate",
                          {"x": ax, "y": ay, "z": az, "range": 1, "dig": True,
                           "scaffold": True},
                          "step onto firm ground away from the junk just dropped, so it stays dropped")
        # No read, or no standable cell in it: walk off the drop, straight away from the cell the junk landed on. Three blocks is the
        # reach items are gathered over again, so standing that far away ends the pickup risk;
        # the pathfinder finds footing on that line rather than the line being assumed walkable.
        # Away means back the way the bot came, when it is known: the toss flew the way it faced.
        dx, dz = gs.drop_pos[0] - gs.pos[0], gs.drop_pos[2] - gs.pos[2]
        if math.hypot(dx, dz) < 0.5 and gs.drop_heading is not None:
            dx, dz = gs.drop_heading
        angle = math.atan2(-dz, -dx)
        # `navigate` wants whole cells and says so with a validation error that owns the tick
        # forever (live, run77: 120 identical `[expected int, received number at x]` refusals
        # while the bot stood on the junk it had just dropped). Floor the float target here;
        # the pathfinder still finds footing on that line.
        tx = math.floor(gs.pos[0] + 3.0 * math.cos(angle))
        tz = math.floor(gs.pos[2] + 3.0 * math.sin(angle))
        return Action(label, "navigate",
                      {"x": tx, "y": math.floor(gs.pos[1]), "z": tz, "range": 1, "dig": True,
                       "scaffold": True},
                      "step away from the junk dropped on open ground, so it stays dropped")
    if label == "drop_junk":
        # On the stronghold march the shed list widens to the biome clutter and Nether leftovers the
        # descent cannot spend (`_march_shed`): the trek never returns to base, so this is the only
        # way a full pack clears, and the extra names apply only while `_striding_to_stronghold`.
        # `_pack_clutter` is the whole-stack half of the same idea: a slot opens only when a stack
        # is emptied, so the shed also names the stacks whose use the pack has already covered.
        # The march list and the family list can name the same block (a sapling is both), and a
        # repeated name in one call is a `[NOT_CARRIED]` on the second entry.
        names: list[str] = []
        for n in _junk_carried(inv) + _march_shed(gs) + _spare_gear(gs, inv) + _pack_clutter(gs, inv):
            if n not in names:
                names.append(n)
        items = [{"item_name": n} for n in names]
        items += _hoard_surplus(gs, inv)
        if not items:
            return None
        # Drop on open ground in the Nether. `in_a_hole` excavates a cell first, and the Nether's
        # floor is netherrack - the very block the hoard surplus is about to shed - so the dig
        # handed the pack back what the drop had just freed. Live, run168 it90-117: eight
        # `drop_junk` calls, netherrack 54 -> 96 -> 65 -> 99 -> 67, slots pinned at 34/36, until a
        # real home stop (craft + stow) freed the space. Dropping on the ground costs a
        # `step_away_drops` at worst; it cannot undo itself.
        # `allow_equipped` is what makes the list executable. The server refuses to toss the item
        # in hand (`[DROP_HELD]`) unless it is told to, and the Nether's scaffold block is
        # netherrack - so a bot walking with netherrack in hand answered
        # `[DROP_INCOMPLETE] 1 of 1 item could not be dropped` on every shed (live, run195
        # it441-443). Nothing in this list is gear the chain needs: `_spare_gear` only names a
        # tier strictly below the best carried, and the rest are blocks.
        return Action(label, "drop_item",
                      {"items": items[:12],
                       "in_a_hole": not gs.drop_hole_blocked and not _in_nether(gs),
                       "allow_equipped": True},
                      "drop low-value blocks so the full pack has room again")
    if label == "wait":
        # Hold at the base: navigate to the anchor, which is at most a step away, so the tick
        # settles at once and the bot does not wander off while daylight burns.
        if gs.base is not None and _nether_idle(gs):
            bx, by, bz = gs.base
            return Action(label, "navigate",
                          {"x": bx, "y": by + 1, "z": bz, "range": 1},
                          "hold at the home base while nothing productive is left in the lull")
        # A naked bot at night holds where it stands: `candidates` offers this instead of
        # `explore`/`get_wood` (`_defenceless_at_night`). The walk home is exactly the long one
        # the night forbids, so the anchor is the bot's own cell - a navigate to where it already
        # stands, which settles at once and spends `[loop] wait_hold_s` instead of the dark.
        if _defenceless_at_night(gs) and gs.pos is not None:
            mx, my, mz = (math.floor(v) for v in gs.pos)
            return Action(label, "navigate", {"x": mx, "y": my, "z": mz, "range": 1},
                          "stand still until sunrise - no weapon and no armour at night")
        # Daytime at the stronghold frame with the respawn bed laid: the one step left is the sleep
        # that pins the spawn, and that waits for night (`sleep_stronghold` offers `wait` here). The
        # base-hold branch above cannot help - the base is ~1300 blocks off - and the phase shuts
        # `go_home`/`explore`, so hold on the bed's own cell, a navigate that settles at once rather
        # than a march off the frame it has to sleep at.
        if _bed_at_stronghold(gs) and gs.pos is not None:
            mx, my, mz = (math.floor(v) for v in gs.pos)
            return Action(label, "navigate", {"x": mx, "y": my, "z": mz, "range": 1},
                          "hold at the stronghold bed until night, when the respawn sleep can run")
        return None
    if label == "explore":
        if x is None or z is None:
            return None
        if _in_nether(gs):
            # The Nether has no chain to work through, so `explore` is not wandering here - it is
            # the fortress search. Walk at the bricks when they are in reach, otherwise march the
            # east-west axis that fortresses are striped along. The generic branch below takes
            # the best standable ground inside the scan radius, which in the Nether is netherrack
            # in every direction: 231 blocks of walking bought 31 blocks of ground in run37, a
            # 13 % straight line through open lava sea that crossed no stripe at all.
            fortress = _fortress_hit(gs)
            if fortress is not None:
                fx, fy, fz = fortress.pos
                return Action(label, "navigate",
                              {"x": fx, "y": fy, "z": fz, "dig": True, "scaffold": True},
                              f"walk to the {fortress.name} at {fx},{fy},{fz}, where blazes spawn")
            heading = _clear_march_heading(gs, _nether_search_heading(gs))
            return Action(label, "explore_frontier",
                          {"heading": heading, "chunks": EXPLORE_FRONTIER_CHUNKS},
                          "march east-west along the fortresses' axis, where blazes spawn")
        if target is not None and target.hit.distance >= EXPLORE_MIN_M:
            tx, ty, tz = target.pos
            return Action(label, "navigate",
                          {"x": tx, "y": ty, "z": tz, "dig": True, "scaffold": True},
                          f"explore toward {target.name} at {tx},{ty},{tz}")
        # No real destination: the best `ground_targets` can offer is a cell at the bot's feet,
        # and a `navigate` between two of those ping-pongs in place (see `targets.boxed_in`).
        # Extend the remembered frontier in a real heading instead, so the bot leaves terrain it
        # has already worked and can find what `explore` is out for (live: lava for the portal).
        return Action(label, "explore_frontier",
                      {"heading": _clear_march_heading(gs, frontier_heading(gs)),
                       "chunks": EXPLORE_FRONTIER_CHUNKS},
                      "explore: no outward ground in reach, extend the frontier toward new terrain")
    return None


def inventory_relief(gs: GameState, allow_stow: bool = True) -> "Action | None":
    """The step that frees pack space after an `[INVENTORY_FULL]` refusal.

    Prefers the chest at the base (deposit) when the bot is standing there; once every chest of
    the network is full it grows the network instead, and only where there is neither chest nor
    room left does it drop into a disposal pit. Either way the pack gains room so the blocked
    gather can run again on the next tick. `allow_stow` is False once a deposit has already
    failed into a full chest, so the loop builds another chest or drops instead of retrying a
    deposit that cannot work.
    """
    if _chest_ready(gs) and _at_base(gs) and gs.base is not None:
        if allow_stow:
            action = resolve("stow_surplus", gs)
            if action is not None:
                return action
        # The network itself is the relief once every chest in it is full -- including the
        # case this function exists for (`allow_stow` off: a deposit just failed into a full
        # chest). Without this the run79 deadlock had no third way out: pack full, chest full,
        # no junk left to drop, and `wait` owned the tick.
        action = resolve("place_chest", gs) or resolve("craft_chest", gs)
        if action is not None:
            return action
    return resolve("drop_junk", gs)


# Re-export the whole module namespace (public and private) so a downstream
# `from .<module> import *` carries every constant and helper by its original name,
# and `mlaya.actions.__init__` can hand the full surface back out unchanged.
__all__ = [_name for _name in list(globals()) if not _name.startswith("__")]
