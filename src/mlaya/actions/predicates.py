from __future__ import annotations

from .constants import *  # noqa: F401,F403


def is_hostile_hunt(action: "Action") -> bool:
    """True when a hunt fights back, so it must run in the aggressive hunt stage."""
    return action.tool == "collect_mob_drop" and action.args.get("mob_name") in HOSTILE_HUNTS


def _logs(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n == "logs" or n.endswith("_log"))


def _planks(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n.endswith("_planks"))


def _plank_units(inv: dict[str, int]) -> int:
    """Planks available once carried logs are accounted for (one log converts into four)."""
    return _planks(inv) + 4 * _logs(inv)


def _first_edible(inv: dict[str, int]) -> str | None:
    """The most nutritious carried food, so the bot eats cooked meat before bread or raw meat."""
    foods = [n for n in inv if n in EDIBLE and inv.get(n, 0) > 0]
    if not foods:
        return None
    return max(foods, key=lambda n: FOOD_SATURATION.get(n, 0.0))


def _edible_count(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n in EDIBLE)


def _pick_tier(inv: dict[str, int], tools: dict[str, str]) -> int:
    tier = tools.get("pickaxe")
    if tier:
        value = TIER.get(str(tier).lower())
        if value is not None:
            return value
    best = 0
    for n in inv:
        if n.endswith("_pickaxe"):
            best = max(best, TIER.get(n[: -len("_pickaxe")], 0))
    return best


def _has(inv: dict[str, int], name: str) -> bool:
    return inv.get(name, 0) > 0


def _can_sticks(inv: dict[str, int], logs: int, planks: int) -> bool:
    return inv.get("stick", 0) >= 2 or planks >= 2 or logs >= 1


# How many drops one food hunt asks for, and how many when the meal is urgent. `collect_mob_drop`
# has no range limit: it retargets to the next animal of that kind anywhere loaded until the
# count is met. run89 asked for six chicken with four in sight; the server walked 591 blocks in
# one 178-second action, ending 276 m out, and the hunger fell from 8 to 5 on the way. So the
# count never exceeds the animals of that kind the bot can see, and a hungry or wounded bot asks
# for two - enough to eat, short enough that `eat_food` gets the next tick.
HUNT_BATCH = 6
HUNT_HURRY = 2
ZOMBIE_FLESH_BATCH = 10
# A chicken is hunted only when no other food animal is in reach: one raw chicken per kill, and
# the server's hunger reflex never eats it (`REFLEX_AVOIDED_FOODS`, food poisoning), so a pack of
# raw chicken does not feed a bot whose loop is deferring - run89 died with six of them carried.
FOOD_MOBS_LAST = ("chicken",)


def _nearest_food_mob(gs: GameState, max_range: float | None = None) -> str | None:
    """The nearest food animal in reach, a chicken only when nothing else is (`FOOD_MOBS_LAST`)."""
    best: tuple[bool, float, str] | None = None
    for mob in FOOD_MOBS:
        d = gs.mob_nearest(mob)
        if d is None or (max_range is not None and d > max_range):
            continue
        key = (mob in FOOD_MOBS_LAST, d, mob)
        if best is None or key < best:
            best = key
    return best[2] if best else None


def _mob_count(gs: GameState, name: str) -> int:
    """How many animals of one kind the observation lists (all loaded, nearest first)."""
    return sum(count for n, _, count, _ in gs.mobs if n == name)


def _mob_in_reach(gs: GameState, name: str, max_range: float) -> int:
    """How many of this kind stand within `max_range` of the body, not merely loaded somewhere.

    The distinction is what a hunt turns on: `collect_mob_drop` has no range limit of its own and
    retargets to the next animal of the kind anywhere loaded until the count is met, so a count
    cut by the loaded area is a count the server walks off to find.
    """
    return sum(count for n, _, count, d in gs.mobs
               if n == name and d is not None and d <= max_range)


def _hunt_reach(gs: GameState, mob: str) -> float:
    """The band a hunt of this kind may work in: the food range for an animal, the melee band for
    the zombie `hunt_food` falls back to."""
    return _food_hunt_range(gs) if mob in FOOD_MOBS else ZOMBIE_HUNT_RANGE


def _hunt_count(gs: GameState, mob: str) -> int:
    """Drops one hunt asks for: the batch, or `HUNT_HURRY` when hungry or wounded, and never more
    than the animals of that kind standing within the hunt's own reach, so the server has no
    reason to walk past them.

    The reach cap used to cover `FOOD_MOBS` only, which is how a zombie hunt escaped it: run102
    it43, hunger 17, health 20, no animal within the night's 16 m, one zombie at 24 m, and the
    hunt asked for `ZOMBIE_FLESH_BATCH` = 10 rotten flesh from it. The server walked the body 614
    blocks in 181 s from zombie to zombie, swung at none, and the tick came back `[BODY_HELD]`.
    """
    hungry = gs.food is not None and gs.food < EAT_HUNGER
    hurt = gs.health is not None and gs.health < RECOVER_HEALTH
    if hungry or hurt:
        want = HUNT_HURRY
    else:
        want = HUNT_BATCH if mob in FOOD_MOBS else ZOMBIE_FLESH_BATCH
    return min(want, max(1, _mob_in_reach(gs, mob, _hunt_reach(gs, mob))))


def _food_hunt_range(gs: GameState) -> float:
    """How far a food hunt may walk: far in daylight, only at hand after dark."""
    return FOOD_HUNT_RANGE_NIGHT if gs.phase == "night" else FOOD_HUNT_RANGE_DAY


# The offer's own long walks, withheld while `_defenceless_at_night` holds (`candidates`). The
# recovery path already stands still in that state; the ordinary offer did not, and run95 it274
# spent a night's `explore` at 20 health and no gear walking straight into a spider and a skeleton.
# Only the walks that go looking for something are here: `go_home` is a walk back to shelter.
DEFENCELESS_WALK_LABELS = ("explore", "get_wood")


def _defenceless_at_night(gs: GameState) -> bool:
    """Night, no weapon in hand and nothing worn: every walk is a walk past the mobs.

    run90 died nine times in the Overworld after the Nether took its kit, every one of them at
    night with no tool and no armour, and most on a walk the loop chose: `recover_drops` back to
    the site a skeleton had just shot it at, or `go_home` to a base where another stood. The
    skeletons and zombies that did it burn at sunrise, so a naked bot waits for the morning
    instead (`_can_recover`, `_recovery_steps`).
    """
    return (gs.phase == "night" and not _armed(gs)
            and not any((m or {}).get("item") for m in gs.armor.values()))


def _armed(gs: GameState) -> bool:
    """Whether the bot carries a weapon a hunt can be walked up with.

    A hunt ends by killing something, and bare hands turn that walk into the run's death:
    run83 offered `hunt_food` and `hunt_wool` at hunger 20, at night, with no tool at all
    (it85), and the food hunt died at (-86,71,14) to a zombie, the day before a skeleton on
    the same kind of walk (with an iron pickaxe but no sword) had already cost the whole
    pack. The bow counts because it shoots a hunt down from distance; a pickaxe does not,
    however often it was carried when the hunt started.
    """
    for kind in ("sword", "bow"):
        tool = str(gs.tools.get(kind) or "none").lower()
        if tool != "none":
            return True
    return False


def _cannot_heal(gs: GameState) -> bool:
    """Wounded, and nothing that heals where the bot stands: no food carried, no regeneration.

    A body with no food in the pack and hunger under `REGEN_HUNGER` neither eats nor regenerates,
    so every tick of work at that health is a tick waiting for the next hit. Whenever the loop has
    the tick in that state it offers `_recovery_steps` alone. run89 began in run88's state - 8.5
    health, hunger 15, no food, in the spider cave - and the gate's `no` handed it the ordinary
    offer (`smelt_iron`, `hunt_wool`, `get_flint`) for fifteen ticks while the hunger fell to 5.
    """
    food = getattr(gs, "food", None)
    return (gs.health is not None and gs.health < RECOVER_HEALTH
            and not _first_edible(getattr(gs, "inventory", None) or {})
            and food is not None and food < REGEN_HUNGER)


def _recovery_owed(gs: GameState) -> bool:
    """`_cannot_heal`, and the survival policy idle: the defer below the band lets go.

    Below `RECOVER_HEALTH` the loop defers every hostile `yes` to the policy, which fights and
    eats - with nothing to eat it does nothing. run88 took cave-spider bites down to 8.5 health at
    it63, hunger 17, not one edible item all run, and deferred the remaining 536 ticks beside the
    spawner: health never moved, and the `hunt_food` offered on it65-76 for a chicken 44 m off was
    deferred with the rest. After `DEFER_QUIET_TICKS` quiet defers `brain.danger_matters` hands
    the tick back for `_recovery_steps`.
    """
    return _cannot_heal(gs) and getattr(gs, "defer_quiet", 0) >= DEFER_QUIET_TICKS


def _must_eat(gs: GameState) -> bool:
    """Food carried, and the body needs it now: hungry, or wounded below the band without the
    hunger to regenerate. Then `eat_food` is the tick's only step.

    Offered beside the work, the meal lost: run89 carried six raw chicken from it16, hunger 5,
    health 8.5, and Laya chose `surface_up` and then `establish_base`; the second walk met a
    drowned at night, and the chicken was still in the pack when the bot died at it21.
    """
    food = getattr(gs, "food", None)
    if food is None or not _first_edible(getattr(gs, "inventory", None) or {}):
        return False
    return food < EAT_HUNGER or (gs.health is not None and gs.health < RECOVER_HEALTH
                                 and food < REGEN_HUNGER)


def _recovery_steps(gs: GameState) -> list[str]:
    """The one step that leads toward food or safety, while `_cannot_heal` holds.

    Food first, because a meal is what heals: the hunt when an animal is in reach. Otherwise
    away from where the bot was hurt - home to the base, or up out of the cave when there is none
    - and with neither, `explore` toward new ground and new animals. No work rides along: a
    `smelt_iron` at 8.5 health beside a spawner is how the next bite finds the bot.

    A naked bot at night may chop trees, hunt if the target is in reach, or craft tools - no
    navigation, just in-place work until sunrise. If no such work is available, wait for sunrise.
    """
    if _hunt_ok(gs) and _food_target(gs) is not None:
        return ["hunt_food"]
    if _defenceless_at_night(gs):
        inv = gs.inventory
        # In-place work only: gather, hunt, craft. No navigation away from this spot.
        if gs.resources.get("logs"):
            return ["get_wood"]
        # Can we craft tools with what we have?
        planks = _planks(inv)
        logs = _logs(inv)
        tier = _pick_tier(inv, gs.tools)
        table = inv.get("crafting_table", 0)
        if (logs >= 1 or planks >= 5) and tier == 0:
            return ["craft_pickaxe"]
        if planks >= 4 and not table:
            return ["craft_table"]
        return []
    # In the Nether the way out of danger is the way home: cross back through the exit portal
    # rather than explore deeper. A death down there drops the whole diamond kit and every rod,
    # and there is no food to hunt and no base to retreat to, so a wounded bot flees with what it
    # has. run93 stood at 9.4 health, hunger 7, nine rods already in the pack and nothing edible:
    # recovery fell through to `explore`, the bot wandered the Nether ~70 ticks, and a blaze
    # burned it down at 4.2 health - the rods and the diamond kit lost with it.
    if _in_nether(gs):
        return ["leave_nether"]
    if gs.base is not None and not _at_base(gs):
        return ["go_home"]
    if gs.base is None and _gear_ready(gs) and gs.pos is not None and not _at_surface(gs):
        return ["surface_up"]
    return ["explore"]


def _hunt_ok(gs: GameState) -> bool:
    """Whether a hunt may be walked: armed, or bare-handed only while starvation bites.

    Unknown hunger is no licence for the unarmed walk - the exception is earned by a meter
    actually at `HUNT_STARVE`, which real states always carry, and none is worth risking a
    death for a meal that the eating reflex would not have needed anyway.
    """
    return _armed(gs) or (gs.food is not None and gs.food <= HUNT_STARVE)




def _first_cookable(inv: dict[str, int]) -> str | None:
    return next((n for n in inv if n in COOKED and inv[n] > 0), None)


def _furnace_fuel(inv: dict[str, int]) -> tuple[str, int] | None:
    """The best carried furnace fuel and how many items one unit of it smelts, or None.

    Coal and charcoal smelt eight items each and are preferred, so their use for iron smelting
    is not stolen; a plank or log smelts one and a half, so a single one cooks exactly one item.
    Requiring coal alone was too strict: a bot carrying a stack of raw meat and plenty of wood
    still could not cook (live, the from-zero run: `coal` was None, so `cook_food` never fired
    and the daytime lull after the gate frame went up fell through to `explore`).
    """
    for name in ("coal", "charcoal"):
        if inv.get(name, 0) > 0:
            return name, 8
    for name in inv:
        if name == "logs" or name.endswith("_log") or name.endswith("_planks"):
            return name, 1
    return None


def _wool(inv: dict[str, int]) -> int:
    return sum(c for n, c in inv.items() if n.endswith("_wool"))


def _has_bed(inv: dict[str, int]) -> bool:
    return any(n.endswith("_bed") for n in inv)


def _safe_spider_hunt(gs: GameState) -> bool:
    """Only farm spider string when the fight is a lone, close, healthy one.

    Hunting hostiles is the single most expensive mistake this bot has made (a
    spider run walked it into a zombie/skeleton pack and cost an iron pickaxe), so
    string farming requires: one spider within 10m, no other hostile within 16m,
    and health at least 16.
    """
    spider = gs.mob_nearest("spider")
    if spider is None or spider > 10 or (gs.health or 0) < 16:
        return False
    others = [d for n, k, _, d in gs.mobs
              if k == "hostile" and n != "spider" and d is not None and d <= 16]
    return not others


def _safe_string_hunt(gs: GameState) -> bool:
    """Whether one spider is worth walking to for a single missing string.

    `_safe_spider_hunt` guards the *bed* chain, where string comes in threes and a bad fight
    costs the kit. The Nether bow needs one string to be repairable at all, and the strict guard
    made that unreachable: with two string banked and no spider within 10 m of a clear field, the
    bow step dropped out for good (live, run32) and the run could neither fight nor finish. One
    spider is cheap next to a bow-less trip, so this widens the reach and allows a little company,
    while still refusing a mob cluster and a hurt bot.
    """
    spider = gs.mob_nearest("spider")
    if spider is None or spider > STRING_HUNT_RANGE or (gs.health or 0) < 16:
        return False
    others = [d for n, k, _, d in gs.mobs
              if k == "hostile" and n != "spider" and d is not None and d <= STRING_HUNT_RANGE]
    return len(others) <= 2


def _safe_zombie_hunt(gs: GameState) -> bool:
    """Zombies are the skill's overworld food bank, but only in ones and twos.

    Rotten flesh is the most reliable local food, so fall back to it only when the
    target is close, health is comfortable, and no other hostile species is in the
    neighbourhood — a pack is how a hunt turns into a death.
    """
    nearest = gs.mob_nearest("zombie")
    if nearest is None or nearest > ZOMBIE_HUNT_RANGE or (gs.health or 0) < 18:
        return False
    hostiles = [(n, d) for n, k, _, d in gs.mobs
                if k == "hostile" and d is not None and d <= 16]
    return len(hostiles) <= 2 and all(n == "zombie" for n, _ in hostiles)


def _eating_now(gs: GameState) -> bool:
    """Whether the pack needs a meal this tick, as opposed to a reserve for a later trip.

    Rotten flesh is `EDIBLE` but never `PROVISIONS` (see the chest of 96 that read as a full
    larder), so a zombie hunt cannot move `_provisioned` toward `_food_goal`: asked for as
    banking, it is a hunt that can never end. run102 it43-45: hunger 17, health 20, nothing but a
    zombie in reach; the hunt roamed 614 blocks, came back `[BODY_HELD]`, and the two ticks that
    followed came back `[BODY_HELD]` too - the run ended at 46 iterations on `body_held`.
    """
    return gs.food is None or gs.food <= EAT_HUNGER


def _food_target(gs: GameState) -> tuple[str, str] | None:
    """The safest available food source: a passive animal in walking reach, else a zombie close
    enough to eat off right now.

    The zombie is the emergency branch, not the reserve branch: it is only taken when the pack is
    at or under the eating floor, because its flesh does not count as a provision.
    """
    mob = _nearest_food_mob(gs, _food_hunt_range(gs))
    if mob is not None:
        return mob, FOOD_MOBS[mob]
    # Starving in daylight with nothing inside the ordinary reach: drop the range cap and walk to
    # the nearest loaded food animal. The cap is a thrift rule, not a safety one - `collect_mob_drop`
    # retargets to any loaded animal regardless of distance - and below the starvation floor a far
    # walk to food is the only way off a starve. run107 lost its kit to a Nether death, then burned
    # 365 ticks ping-ponging `explore` <-> `go_home` at its base while the nearest animal (a pig at
    # 113 m) sat past the 64 m day cap: `_cannot_heal` routed every tick to `_recovery_steps`, which
    # found no food target and degenerated into the two-cell loop; hunger fell 6 -> 0, health to 1,
    # and the run starved. Held to daylight so it never reopens the old far night roam.
    if gs.food is not None and gs.food <= HUNT_STARVE and gs.phase != "night":
        mob = _nearest_food_mob(gs)
        if mob is not None:
            return mob, FOOD_MOBS[mob]
    if _safe_zombie_hunt(gs) and _eating_now(gs):
        return "zombie", "rotten_flesh"
    return None


def _tool_durability(gs: GameState, cls: str) -> tuple[float, str] | None:
    """(fraction remaining, item name) for the active tool of a class, if known."""
    meta = gs.tool_meta.get(cls) or {}
    left, mx, item = meta.get("left"), meta.get("max"), meta.get("item")
    if item and isinstance(left, (int, float)) and isinstance(mx, (int, float)) and mx > 0:
        return (left / mx, str(item))
    return None


def _pickaxe_left(gs: GameState) -> float | None:
    """Absolute uses left on the active pickaxe, when the server reports it."""
    left = (gs.tool_meta.get("pickaxe") or {}).get("left")
    return float(left) if isinstance(left, (int, float)) else None


def _pickaxe_due_for_spare(gs: GameState) -> bool:
    """True when the active pickaxe should already have a fresh same-tier backup carried.

    Two triggers: the classic "about to break" fraction, and an absolute headroom reserve so
    that a single long dig cannot exhaust the tool before the next tick can react.
    """
    d = _tool_durability(gs, "pickaxe")
    if d is None:
        return False
    if d[0] <= LOW_DURABILITY:
        return True
    left = _pickaxe_left(gs)
    return left is not None and left <= MINING_HEADROOM


def _has_spare_pickaxe(gs: GameState, inv: dict[str, int], item: str) -> bool:
    """A same-tier backup is already carried (the active tool counts as the first copy)."""
    active = (gs.tool_meta.get("pickaxe") or {}).get("item")
    return inv.get(item, 0) >= (2 if item == active else 1)


def _replacement_pickaxe(gs: GameState, inv: dict[str, int]) -> str | None:
    """The best pickaxe still craftable when the active one needs a backup.

    Never a downgrade below what the bot already wields: diamond if diamonds are banked,
    else iron, else stone (which still mines iron ore). A wooden one is pointless here
    because `craft_pickaxe` only fires when the bot has no pickaxe at all. Returns None once
    a spare of that kind is already carried, so the reserve cannot craft the same spare twice
    (the server equips a same-tier fresh tool, but a lower-tier spare leaves the worn active
    one in hand - without this guard that would loop until the materials ran out).
    """
    tier = _pick_tier(inv, gs.tools)
    if tier < STONE_PICK_TIER:
        return None
    sticks = _can_sticks(inv, _logs(inv), _planks(inv))
    table = _has(inv, "crafting_table")
    item = None
    if tier >= DIAMOND_PICK_TIER and table and inv.get("diamond", 0) >= 3 and sticks:
        item = "diamond_pickaxe"
    elif tier >= IRON_PICK_TIER and table and inv.get("iron_ingot", 0) >= 3 and sticks:
        item = "iron_pickaxe"
    elif table and inv.get("cobblestone", 0) >= 3 and sticks:
        item = "stone_pickaxe"
    if item is None or _has_spare_pickaxe(gs, inv, item):
        return None
    return item


def _tier_after_break(gs: GameState) -> int:
    """Best pickaxe tier still carried once the active one has broken."""
    active = (gs.tool_meta.get("pickaxe") or {}).get("item")
    counts = dict(gs.inventory)
    if active and counts.get(active, 0) > 0:
        counts[active] -= 1
    tiers = [TIER.get(name[: -len("_pickaxe")], 0)
             for name, n in counts.items() if n > 0 and name.endswith("_pickaxe")]
    return max(tiers, default=0)


def _can_mine(gs: GameState, block: str) -> bool:
    """Whether the active pickaxe may start a dig on `block`.

    Two conditions: the active tool meets the block's minimum tier, and a break *inside* the
    dig cannot leave the bot below it - the server finishes the block with the swapped-in lower
    tier, which destroys it without a drop. Stone and coal are exempt from the second check:
    they are cheap, and guarding them would deadlock the bootstrap, where a worn wooden pickaxe
    must still be allowed to gather the cobble a stone replacement needs.
    """
    need = ORE_MIN_TIER.get(block, 0)
    if _pick_tier(gs.inventory, gs.tools) < need:
        return False
    if need < STONE_PICK_TIER:
        return True
    return not _pickaxe_due_for_spare(gs) or _tier_after_break(gs) >= need


def _worn(gs: GameState, item: str) -> bool:
    """Whether `item` is currently on the armour slots (equipped armour leaves the inventory)."""
    return any((m or {}).get("item") == item for m in gs.armor.values())


def _armor_present(gs: GameState, slot: str) -> bool:
    """A diamond piece counts as done whether it is worn or still waiting in the inventory."""
    item = f"diamond_{slot}"
    return _worn(gs, item) or _has(gs.inventory, item)


def _armor_rank(item: str | None) -> int:
    """The material tier of an armour item name (`iron_chestplate` -> 3), 0 when absent."""
    if not item:
        return 0
    return TIER.get(item.split("_", 1)[0], 0)


def _armor_slot_covered(gs: GameState, slot: str) -> bool:
    """Armour on that slot at iron tier or better, worn or still carried.

    The pre-diamond kit is iron, but a diamond piece already on the slot counts too - the point
    is that the slot is protected, not which tier protects it.
    """
    for tier in ("iron", "diamond", "netherite"):
        item = f"{tier}_{slot}"
        if _worn(gs, item) or _has(gs.inventory, item):
            return True
    return False


def _armor_to_wear(gs: GameState) -> list[str]:
    """Carried armour whose slot is bare or holds a weaker piece, best tier per slot.

    The diamond set is crafted on top of the iron one, so a diamond chestplate worn while the
    iron one sits back in the pack must not be traded away: only a *better* piece is worn, and
    the iron piece it displaced is skipped rather than re-equipped on every tick.

    The helmet slot has a second owner. The gold helmet replaces the diamond one
    (`swap_to_gold_helmet`, so piglins stay neutral) and the swap leaves the diamond helmet in
    the pack; read as "a better piece is waiting", that puts it straight back on the head and the
    two steps undo each other every tick (live, run36: `equip_armor` and `swap_to_gold_helmet`
    alternating, each action succeeding, none of them progress). So the helmet is skipped while
    the gold piece owns the slot: the whole Nether trip, and at the base from the moment the bot
    is about to cross (`_nether_departure_ready`), where the gold piece now goes on before the
    crossing - otherwise the same ping-pong plays out on the Overworld side. On the way home the
    gate is no longer departure-ready (the haul holds it shut), so the diamond helmet goes back on.
    """
    inv = gs.inventory
    out: list[str] = []
    slots = [slot for slot, _ in IRON_ARMOR]
    if ((_in_nether(gs) or _nether_departure_ready(gs))
            and (inv.get(GOLD_HELMET_ITEM, 0) > 0 or _worn(gs, GOLD_HELMET_ITEM))):
        slots = [slot for slot in slots if slot != "helmet"]
    for slot in slots:
        worn = (gs.armor.get(slot) or {}).get("item")
        for tier in ("netherite", "diamond", "iron"):
            item = f"{tier}_{slot}"
            if _has(inv, item):
                if TIER.get(tier, 0) > _armor_rank(worn):
                    out.append(item)
                break
    return out


def _offhand(gs: GameState, item: str) -> bool:
    """Whether `item` currently sits in the off-hand slot (there it leaves the inventory)."""
    return gs.offhand == item


def _has_shield(gs: GameState) -> bool:
    """A shield counts as done whether it is in the off-hand or still waiting in the inventory."""
    return _offhand(gs, SHIELD_ITEM) or _has(gs.inventory, SHIELD_ITEM)


def _has_iron_sword(gs: GameState) -> bool:
    """An iron sword or better, worn (the `sword` tool slot) or still carried."""
    if TIER.get(str(gs.tools.get("sword", "")).lower(), 0) >= IRON_PICK_TIER:
        return True
    return any(_has(gs.inventory, f"{tier}_sword")
               for tier in ("iron", "diamond", "netherite"))


def _combat_ready(gs: GameState) -> bool:
    """The pre-diamond safety kit: a sword, a shield and armour on every slot.

    `mine_diamond` is gated on this, so the bot never dives to y<16 for diamonds unarmed and
    unarmoured again (live, run12: a naked dive died to a zombie and a creeper, and the dropped
    gear was lost to the despawn). The pieces are iron, which is the only pre-diamond sword and
    armour the tree crafts.
    """
    if not _has_iron_sword(gs):
        return False
    if not _has_shield(gs):
        return False
    return all(_armor_slot_covered(gs, slot) for slot, _ in IRON_ARMOR)


def _gear_ready(gs: GameState) -> bool:
    """The Nether-prerequisite gear: diamond pickaxe, diamond sword, the whole armour set
    actually worn, and the shield in the off-hand. The portal chain hangs off this, so a fresh
    run finishes the diamond tier before it ever starts banking obsidian."""
    inv = gs.inventory
    if _pick_tier(inv, gs.tools) < DIAMOND_PICK_TIER:
        return False
    if not (_has(inv, "diamond_sword") or str(gs.tools.get("sword", "")).lower() == "diamond"):
        return False
    for slot, _ in DIAMOND_ARMOR:
        if _worn(gs, f"diamond_{slot}"):
            continue
        # The helmet slot is shared with the gold piece the bot wears for piglin neutrality. Once
        # it puts the gold helmet on at the base before crossing (`swap_to_gold_helmet`), the
        # diamond helmet sits in the pack - held in reserve, not missing - so the kit still counts
        # as ready. Without this the gold swap would read as a bare slot and collapse the whole
        # departure chain the swap hangs off (gear ready -> expedition ready -> the gate itself).
        if slot == "helmet" and _worn(gs, GOLD_HELMET_ITEM) and _has(inv, "diamond_helmet"):
            continue
        return False
    return _offhand(gs, SHIELD_ITEM)


def _portal_lit(gs: GameState) -> bool:
    """A lit nether portal is standing - the frame is on and the gate is open.

    The raw `resource_found` count counts every hit in a loaded chunk, not just those inside
    `RESOURCE_RADIUS`, and that second reading is what this needs. The proximity read alone makes
    the flag a statement about where the bot is standing rather than about the world: live, run58,
    `portal_lit` came back false while the gate stood lit at `(228,64,222)` and the bot cooked at
    `(357,70,362)`. Two of the gates read this as a fact - `light_portal` is offered when it is
    false - so a bot 190 m out believed it had an unlit gate and would have set about lighting one
    that was already burning. That is run22's shape again, in a different costume.

    What it is not: a latch. A gate whose chunk has unloaded still reads false, and the honest fix
    for that is a remembered flag of the kind `_frame_complete` and `powder_stowed` already are.
    Scoped deliberately, because this is the difference between a claim the loop can make and one
    it should remember.
    """
    return (gs.resource_distance(PORTAL_BLOCK) is not None
            or gs.resource_found.get(PORTAL_BLOCK, 0) > 0)


def _portal_built(gs: GameState) -> bool:
    """Whether the gate structure stands at the base - the frame is up, lit or not.

    The whole daytime lull used to key on `_portal_lit`, which meant the gate had to be
    *burning* before the run would stock the Nether prep - and so it burned through the entire
    prep window while the bot worked at the base beside its open interior (live, run22: lit at
    it208, in the Nether by it245). The frame standing is the earlier, honest signal: the prep
    belongs to the window the frame is up, and only `light_portal` now waits for the expedition
    to be ready (`_expedition_ready`). A lit portal always has its frame, so this still holds for
    a gate a previous run left standing.
    """
    return _portal_lit(gs) or _frame_started_at_base(gs)


def _in_portal(gs: GameState) -> bool:
    """Whether the bot's own body occupies a lit portal block.

    The shared pathfinder treats a portal block as open, so a `navigate` goal on the far side
    routes the body straight through the gate, and the teleport fires a few seconds later - no
    step ever asked for the crossing (live, run22: the lit gate stood beside the base and the
    bot was in the Nether a few decisions after it was lit). The rescue owns the tick while the
    body is in the gate, so it steps out before the teleport.

    `gs.enter_intent` is the one crossing that *was* asked for: the loop sets it the tick it
    resolves `enter_portal` or `leave_nether` and holds it while the body is in the gate, so the
    rescue stands down for the intended trip and keeps owning every accidental one (Agent
    `_enter_intent`).
    """
    if gs.enter_intent or gs.pos is None:
        return False
    fx, fy, fz = (math.floor(v) for v in gs.pos)
    cells = {(fx, fy, fz), (fx, fy + 1, fz)}
    return any(hit.pos in cells for hit in gs.resources.get(PORTAL_BLOCK, []))


def _leave_portal_target(gs: GameState) -> tuple[float, float, float] | None:
    """A standable cell just outside the gate the bot stands in, or None.

    The home base is the natural step-out (the frame plane sits two blocks east of it), so it wins
    when it is clear of the portal; otherwise the nearest horizontal neighbour that is not itself a
    portal block. No digging: the frame around the opening must not be damaged by the rescue.

    The base is only a step-out in the dimension it is in. `base` is one field for both worlds, so
    a bot standing in the Nether's own arrival portal was sent toward the *Overworld* base's
    coordinates - which the pathfinder can never reach from there. That refusal is total and
    repeats: `[NAVIGATION_STOPPED] ... no path found, visited 101 nodes` twenty times running,
    with `leave_portal` owning the tick outright, so the expedition never started (live, run53).
    """
    if gs.pos is None:
        return None
    fx, fy, fz = (math.floor(v) for v in gs.pos)
    portal = {hit.pos for hit in gs.resources.get(PORTAL_BLOCK, [])}
    if gs.base is not None and _overworld(gs):
        bx, by, bz = gs.base
        if (bx, by + 1, bz) not in portal and (bx, by + 2, bz) not in portal:
            return (bx + 0.5, by + 1, bz + 0.5)
    # The neighbours start at *two* cells, not one. `navigate` calls the goal reached when the body
    # is within `range` of it - one block by default - so a target one cell away is satisfied by
    # standing exactly where it already stands: the call answers success in five milliseconds
    # without moving anything. Live, run54 and run55 burned 299 then 345 ticks of `leave_portal`
    # all reporting ok, the body on the same obsidian inside the gate, the run a step from the
    # iteration limit. With the target two cells out the identical call walks - measured: the body
    # went from z 34.5 to z 33.66, clear of the portal blocks, which all sit on z 34.
    for dx, dz in ((0, -2), (0, 2), (-2, 0), (2, 0), (0, -3), (0, 3), (-3, 0), (3, 0)):
        if (fx + dx, fy, fz + dz) not in portal:
            return (fx + dx + 0.5, fy, fz + dz + 0.5)
    return None


# Candidate stances for the gate bed, as offsets from the base anchor in the horizontal plane,
# nearest first. `place_bed` walks the bot onto the first one whose own cell and four orthogonal
# neighbours all clear `BED_PORTAL_CLEARANCE`: the `sleep` server lays the carried bed on the flat
# footprint "beside the bot" (the bed's two cells are two of those five), so proving the whole
# 3x3-less-the-diagonals block clear is what makes the bed clear whichever way it lands. The
# anchor (0,0) is tried first so a base whose gate is already far enough is left untouched.
_BED_STANCE_OFFSETS = tuple(sorted(
    ((dx, dz) for dx in range(-4, 5) for dz in range(-4, 5)),
    key=lambda o: (max(abs(o[0]), abs(o[1])), abs(o[0]) + abs(o[1]), o[0], o[1])))


def _gate_gap(cell: tuple[int, int, int], portal: list) -> int:
    """Chebyshev distance in the horizontal plane from a cell to the nearest lit portal block.

    `portal` is the `nether_portal` hit list; an empty list (nothing lit or loaded) reads as a
    gap no stance can fail on. The vertical axis is ignored on purpose: the hazard the clearance
    guards against is a bed that respawns the bot inside the gate's own footprint, and
    `BED_PORTAL_CLEARANCE` is the plan-view margin the bed keeps from the opening.
    """
    if not portal:
        return 1 << 30
    x, _, z = cell
    return min(max(abs(x - h.pos[0]), abs(z - h.pos[2])) for h in portal)


def _bed_bad_spot(gs: GameState) -> bool:
    """A bed stands within `BED_PORTAL_CLEARANCE` of the lit gate, wherever the base is.

    Sleeping pins the respawn point to the bed's own cell, so such a bed is not a valid gate bed
    even when it sits inside the base box (`_bed_at_base`): the respawn can drop the bot into the
    portal, which teleports it to the Nether within seconds. Live: the bed sat in the opening, the
    bot respawned at the gate, and the Nether followed three seconds later. `take_bed` clears it
    and `place_bed` lays a fresh one clear of the gate. Keyed on the gate being lit - before that
    there is no portal block to be close to, so the prep bed away from the base is untouched.
    """
    portal = gs.resources.get(PORTAL_BLOCK) or []
    if not portal:
        return False
    return any(_gate_gap(hit.pos, portal) < BED_PORTAL_CLEARANCE
               for hit in gs.resources.get("white_bed") or [])


def _bed_needs_moving(gs: GameState) -> bool:
    """A standing bed that `take_bed` should pick up: a leftover away from the base, or one laid
    too close to the gate (`_bed_bad_spot`). Both leave a respawn point the gate bed must not be.
    """
    return _bed_off_base(gs) or _bed_bad_spot(gs)


def _clear_of_gate(cell: tuple[int, int, int], portal: list) -> bool:
    """Whether a bed may be laid with the bot standing on `cell`.

    The `sleep` server lays the carried bed on the flat footprint beside the bot, so the cell
    the bot occupies and its four orthogonal neighbours - the two cells a bed can reach from it
    - must all keep `BED_PORTAL_CLEARANCE` from the lit gate.
    """
    x, _, z = cell
    cells = ((x, z), (x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1))
    return all(_gate_gap((cx, cell[1], cz), portal) >= BED_PORTAL_CLEARANCE
               for cx, cz in cells)


def _bed_stance(gs: GameState) -> tuple[int, int, int] | None:
    """The feet cell the bot must stand on for `place_bed`, clear of the gate, or None.

    See `_BED_STANCE_OFFSETS`. The bot's *own* cell is tried first, because it needs no walk at
    all. Asking the bot to step into a different legal neighbour is not safe to do: `navigate`
    settles inside its arrival tolerance, so a bot already close enough to a neighbouring stance
    reports arrival without ever crossing that cell's boundary (live, run29: the lit gate put the
    stance at `(225,64,222)`, the bot stood on the equally legal `(225,64,221)`, and 17 identical
    `navigate` ticks left the carried bed unlaid and the run unable to finish). The clearance test
    is the same either way (`_clear_of_gate`), so a bed laid from the cell the bot already holds is
    exactly as clear of the gate as one laid after a walk.

    Otherwise the offsets are tried nearest-first from the base anchor, which leaves a base whose
    gate is already far enough untouched. None with no base, or when no cell within the base box
    is clear - `place_bed` then stands down rather than lay the bed against the gate.
    """
    if gs.base is None:
        return None
    bx, by, bz = gs.base
    portal = gs.resources.get(PORTAL_BLOCK) or []
    if gs.pos is not None:
        cell = (math.floor(gs.pos[0]), math.floor(gs.pos[1]), math.floor(gs.pos[2]))
        # Inside the base box only: the bot must stay where `_at_base` holds and the bed must land
        # within that box (`_bed_off_base` would otherwise take the fresh bed straight back up).
        if (abs(cell[0] - bx) <= BASE_RADIUS and abs(cell[2] - bz) <= BASE_RADIUS
                and abs(cell[1] - (by + 1)) <= BASE_RADIUS + 1
                and _clear_of_gate(cell, portal)):
            return cell
    for dx, dz in _BED_STANCE_OFFSETS:
        cell = (bx + dx, by + 1, bz + dz)
        if _clear_of_gate(cell, portal):
            return cell
    return None


def _at_stance(gs: GameState, cell: tuple[int, int, int]) -> bool:
    """Whether the bot's feet occupy the stance cell, so `sleep` lays the bed exactly there."""
    if gs.pos is None:
        return False
    return (math.floor(gs.pos[0]), math.floor(gs.pos[1]),
            math.floor(gs.pos[2])) == cell


def _bed_placed(gs: GameState) -> bool:
    """A bed block stands nearby - the bed the run places next to the portal."""
    return gs.resource_distance("white_bed") is not None


def _bed_off_base(gs: GameState) -> bool:
    """A bed stands nearby, but not at the base.

    The base bed belongs beside the gate frame, so a bed lying anywhere else is a leftover - the
    respawn bed of an earlier base, or of an earlier run in the same world, that the new base did
    not bring along. Live, run18: the run17 bed sat at y=55 inside the shaft the surface base dug
    for gold, and every `collect_block` stumbled on its half-block top (feet 55.56, no settled
    arrival) while `mine_gold` re-dug the same shaft. `take_bed` picks it up and `place_bed` lays
    it at the base again.

    The box is the one `_at_base` allows the bot itself, so a bed `place_bed` laid where the bot
    stood reads as at the base; a tighter box would make place and take trade the bed every tick.
    The shaft bed is clear of it (live, run18: base y=64, bed y=55, ten below the feet level).
    """
    if gs.base is None:
        return False
    hits = gs.resources.get("white_bed") or []
    if not hits:
        return False
    bx, by, bz = gs.base
    return not any(abs(hit.pos[0] - (bx + 0.5)) <= BASE_RADIUS
                   and abs(hit.pos[1] - (by + 1)) <= BASE_RADIUS + 1
                   and abs(hit.pos[2] - (bz + 0.5)) <= BASE_RADIUS
                   for hit in hits)


def _bed_at_base(gs: GameState) -> bool:
    """A bed stands at the base - the gate bed `place_bed` lays beside the gate frame.

    The one clause the completion gate, `sleep` and `place_bed` all key off: a bed merely
    *loaded* is not it (`_bed_placed`), because the leftover in the base's own shaft would send
    the server to that bed and pin the respawn point off the gate again (live, run19b it25). A
    bed inside the base box but too close to the gate is not it either (`_bed_bad_spot`): it
    would respawn the bot into the portal, so the run takes it up and lays it clear.
    """
    return _bed_placed(gs) and not _bed_off_base(gs) and not _bed_bad_spot(gs)


def _nether_idle(gs: GameState) -> bool:
    """The daytime lull once the gate frame stands: the kit is done, the frame is up and only
    the prep (bow, gold helmet, deep food and cobble) and the gate bed are left. The extra
    Nether-prep work (deepening the food reserve, cooking on any fuel, holding at the base) is
    scoped to this window; before it the bot is busy with the chain and must not be pulled off
    it. It keys on the frame (`_portal_built`), not on the lit gate: the gate is only lit when
    the expedition is ready (`_expedition_ready`), so the prep is precisely the work that runs
    *before* the portal burns rather than beside the open interior."""
    return _gear_ready(gs) and _portal_built(gs)


def _food_goal(gs: GameState) -> int:
    """How many edible items to keep banked: the ordinary floor, or the deeper Nether target
    stocked during the lull so the next trip starts with a full supply."""
    return NETHER_FOOD_RESERVE if _nether_idle(gs) else FOOD_RESERVE


def _food_source_in_reach(gs: GameState) -> bool:
    """Whether food the daylight hunt could actually walk to is present: a passive animal within
    the daylight reach, or a zombie within melee reach for its rotten flesh.

    Distance-bounded, unlike the Nether prep steps' "is it loaded anywhere?" test, because a hunt
    must *close the distance* while a mine can tunnel to a merely loaded deposit: keying the
    completion gate off a mob anywhere could hold the run open with nothing it can do (the lull
    then offers only `wait`, and the run stalls). The *daylight* range is used regardless of the
    current phase, so the gate does not read the shorter night range as "no source ever": the run
    sleeps the night and hunts by day, and that `sleep` is what keeps the gate from idling.

    Testing the loaded area instead, to stop the gate flipping when the bot stands in a tunnel,
    was tried live (run19b) and reverted: a gate no hunt within reach can satisfy pushed the run
    into the far hunt below, which is the far worse failure.
    """
    return (_nearest_food_mob(gs, FOOD_HUNT_RANGE_DAY) is not None
            or (gs.mob_nearest("zombie") or float("inf")) <= ZOMBIE_HUNT_RANGE)


def _provision_count(counts: dict[str, int]) -> int:
    """How many of these items can actually feed a trip, as opposed to merely be eaten."""
    return sum(c for n, c in counts.items() if n in PROVISIONS)


def _total_edible(gs: GameState) -> int:
    """Every edible item the bot can lay hands on: the pack plus what the chest last held.

    This is the number that decides whether a hunt is worth walking for, and it was the pack
    alone. The bot stood at its own base with a chest full of food and hunted anyway, because
    `_edible_count(inventory)` cannot see across the room from the pack into the chest.

    The two must not be folded into one. What stops a hunt is the total; what has to come along
    into the Nether is the pack, and the chest is at the base - so `_food_reserve_met` still counts
    carried food only, and `take_stored_food` is what refills the pack from the bank.
    """
    return _edible_count(gs.inventory) + sum(gs.banked.values())


def _provisioned(gs: GameState) -> int:
    """Provisions carried plus banked: the count that answers "is a hunt still worth walking?".

    Carried and banked are one number here, because a hunt can serve either - what goes into the
    pack is decided afterwards, by `take_stored_food`. What is *not* one number is the two kinds
    of food. The chest at the base held 96 rotten flesh and nothing else, and counting it as a
    reserve withheld the hunt that fetches the right sort - a crossing stocked with emergency
    food is worse than the hunting it replaced.
    """
    return _provision_count(gs.inventory) + _provision_count(gs.banked)


def _banked_provisions(gs: GameState) -> dict[str, int]:
    """What the chest holds that counts as trip provisions - never the junk banked beside them.

    Rotten flesh can be eaten but cannot be banked as food (see `_provisioned` for the same
    split). run77's successor ran on this hole: `stow_surplus` deposited 21 rotten flesh into
    the chest, `take_stored_food` withdrew that very stack again, and for ~400 iterations pack
    and chest shed the pile back and forth while the Nether trip it was offered for waited. A
    bank that holds only junk is an empty larder, and provisions that do not exist are not
    fetchable.
    """
    return {n: int(c) for n, c in (gs.banked or {}).items() if n in PROVISIONS and c > 0}


def _food_reserve_met(gs: GameState) -> bool:
    """Whether the banked food reserve is met, or out of reach.

    The gate holds the run for `_food_goal` - the deep `NETHER_FOOD_RESERVE` in the lull once the
    gate frame stands - but lets the requirement drop out once no source is in reach, the same "unreachable
    material drops out" rule the Nether prep steps follow. So a stocked pack is now part of "the
    overworld is done": if the gate holds, a hunt is offered in the same reach (or the night's
    `sleep` is), so it never stalls; and a world with no animal near can never leave the run
    unable to end.
    """
    return (_edible_count(gs.inventory) >= _food_goal(gs)
            or not _food_source_in_reach(gs))


def _overworld(gs: GameState) -> bool:
    """Whether the bot is in the Overworld. A bed placed in the Nether explodes and water
    evaporates there, so `sleep`, `place_bed` and `fill_bucket` are Overworld-only."""
    dim = str(gs.dimension or "").lower()
    return "nether" not in dim and "end" not in dim


# The fortress bricks and spawners a Nether search recognises, in the order a search prefers them:
# the cage itself, then the structure around it.
FORTRESS_BLOCKS = ("spawner", "nether_bricks", "nether_brick_fence", "nether_brick_stairs")


def _fortress_hit(gs: GameState):
    """The nearest fortress landmark worth walking to, or None.

    Blazes exist only in a Nether fortress, so the bricks and the spawner cage are the landmark
    worth walking to rather than the local greedy step `explore` would otherwise take.

    Two readings the live runs forced. A landmark underfoot is not a destination: run104 stood on a
    spawner at (94,39,86) for 100+ ticks, `navigate` answering `ok` with 0 metres moved. And a cage
    the bot has already walked onto and got nothing out of is searched - offering it again next tick
    is the same 0-metre step (`gs.nether_landmarks`, Agent `_note_landmark`).
    """
    for name in FORTRESS_BLOCKS:
        hit = gs.resource_nearest(name)
        if hit is None or hit.distance < FORTRESS_MIN_M:
            continue
        if hit.pos in gs.nether_landmarks:
            continue
        return hit
    return None


def _nether_search_heading(gs: GameState) -> float:
    """Compass heading for the fortress search: the world's unexplored edge, else due east or west.

    The server remembers which chunks it has observed, so it can name where the unexplored starts
    and point a compass heading at it (`FRONTIER_HEADING_SQL`). Walking that heading is the grid
    search: the frontier moves as the bot covers it, so successive headings turn, sweep and turn
    again -- coverage of the ground nobody has looked at, without a waypoint list or a leg length
    to keep, and without betting on an axis. The stripes were described east-west in one source and
    north-south in another and the loop had no measurement, since it never found a fortress.

    The axis march is the fallback for a tick without a frontier read (no rows, or the read
    failed): east until the bot is level with where the trip came in - or, when the run itself
    saw no crossing, with where the server remembers the last one arriving (`_nether_way_home`) -
    then west. That sweeps one line, which is worse, but it is a line rather than standing still.
    """
    if gs.frontier_heading is not None:
        return gs.frontier_heading
    way_home = _nether_way_home(gs)
    if way_home is None or gs.pos is None:
        return 90.0
    return 270.0 if gs.pos[0] >= way_home[0] else 90.0


def _in_nether(gs: GameState) -> bool:
    """Whether the bot stands in the Nether. The mirror of `_overworld`."""
    return "nether" in str(gs.dimension or "").lower()


def _blaze_rods(inv: dict[str, int]) -> int:
    """Blaze rods carried, counting neither powder nor a chest full of either."""
    return int(inv.get(BLAZE_ROD_ITEM, 0))


def _blaze_powder(inv: dict[str, int]) -> int:
    """Blaze powder carried in the pack."""
    return int(inv.get(BLAZE_POWDER_ITEM, 0))


def _nether_haul_home(gs: GameState) -> bool:
    """A Nether trip is home with its haul still to turn into stowed powder.

    The relaunch path - rebuild the spare pickaxe (`_spare_pickaxe_wanted`), light the gate and
    cross (`_nether_departure_ready`) - used to be gated on `not gs.nether_entered`, so a trip
    already made was neither re-prepared nor re-entered. But that latch is one-way per run and
    never clears: a bot that died in the Nether and respawned home empty-handed still read as
    "trip done", and every relaunch step stayed off. run93 stood at the base on `wait` for ~25
    min (~50 ticks) with one diamond pickaxe left of the kit, no food banked and no bow - the
    second pickaxe it needed to relaunch was never offered because the latch said the trip was
    already behind it.

    What the gate really wants is "a haul is waiting to be processed": rods or powder in the
    pack, turned into the stowed powder that `nether_expedition_complete` latches. A Nether death
    drops both, so an empty-handed return reopens the relaunch; a successful return still holds
    it shut until the powder is crafted and stowed. Rods only ever come from the Nether, so
    before the first trip this is false - identical to the old `not nether_entered` there.
    """
    return _blaze_rods(gs.inventory) > 0 or _blaze_powder(gs.inventory) > 0


def _nether_topup_wanted(gs: GameState) -> bool:
    """Whether the stronghold phase has to send the bot back to the Nether for more powder.

    The fallback the user named: top up blaze rods from the Nether only if needed. It is not a
    normal step - the first expedition banks eighteen powder, more than the sixteen the eyes
    cost - so this stays shut unless the banked powder was lost (the chest raided, a death on
    the way back). The trigger is the gate chest coming up empty of powder (`gs.powder_exhausted`)
    while the phase still owes eyes it has no powder to craft. Reopening reuses the whole tested
    Nether chain; a fresh stow clears `powder_exhausted` and the eye-crafting resumes.

    It lives here rather than in `stronghold` because `nether_expedition_open` below reads it:
    the gate that reopens the chain has to be visible to the Nether predicates themselves, and
    `predicates` cannot import `stronghold` without a cycle.
    """
    return (nether_expedition_complete(gs) and not gs.end_portal_active
            and gs.powder_exhausted
            and _eyes_still_owed(gs) > _blaze_powder(gs.inventory))


def _nether_pearl_wave_wanted(gs: GameState) -> bool:
    """Whether a pearl wave should run: the rod haul is stowed and the eyes are still owed, with
    pearls to hunt for them. The second reopener of `nether_expedition_open`, beside the powder
    top-up.

    The waves are the user's structure for a Nether that kills easily (docs/stronghold.md, "Nether
    in Wellen"): Welle 1 hauls the rods and banks the powder (`nether_expedition_complete`), and
    only then do the pearl trips begin - gold dug fresh in the Overworld, carried down to barter,
    and endermen hunted in the Nether. The trip is split from the rod haul so a death down there
    drops at most one wave's pearls, not the whole run's progress.

    Two faces, because the trip count means one thing at home and another below. In the Nether the
    wave stays open until the bot comes home, whatever the count, so `leave_nether` is always on the
    pearl ladder (the count is already spent on the crossing). In the Overworld it only relaunches
    with *empty hands* - pearls just brought home are crafted into eyes first, the eye chain owning
    that tick - and only while trips remain: after `PEARL_WAVE_MAX_TRIPS` the Overworld enderman
    roam takes the pearl hunt over (`stronghold._search_enderman_wanted`), so the goal is never
    stranded on an empty Nether.
    """
    if not (nether_expedition_complete(gs) and not gs.end_portal_active
            and _eyes_still_owed(gs) > 0):
        return False
    if _in_nether(gs):
        return not gs.powder_exhausted
    return (int(gs.inventory.get(ENDER_PEARL_ITEM, 0)) == 0
            and gs.pearl_wave_trips < PEARL_WAVE_MAX_TRIPS)


def _pearl_wave_trip(gs: GameState) -> bool:
    """Whether the bot's Nether trip is a pearl trip rather than a rod trip.

    The planner's Nether block splits on this: a trip made after the rod haul is banked hunts
    pearls, a trip before it (or a powder top-up, where the bank was lost) hunts blazes. The
    top-up is the exception that keeps the rod chain: `powder_stowed` is still latched there, so
    `nether_expedition_complete` reads done, but the eyes cannot be crafted without fresh powder,
    and the trip that reopened the gate is a blaze hunt, not a pearl hunt.
    """
    return nether_expedition_complete(gs) and not gs.powder_exhausted


def _eyes_still_owed(gs: GameState) -> int:
    """Eyes of ender the stronghold phase still owes, read here without importing `stronghold`.

    `stronghold._eyes_owed` is the same number; the Nether haul can only size itself on it through
    a local copy, since `predicates` cannot import `stronghold` without a cycle - the same reason
    `_nether_topup_wanted` lives here.
    """
    return max(0, ENDER_EYES - int(gs.inventory.get(ENDER_EYE_ITEM, 0)))


def _rods_target(gs: GameState) -> int:
    """How many blaze rods the current expedition should haul.

    The first trip hauls the full `BLAZE_RODS` (nine, two powder to spare). A stronghold top-up is
    a *replacement*, not a re-haul: it only owes the powder the shattered eyes cost, so it is sized
    to `ceil((eyes_owed - carried_powder) / 2)`, clamped to `[1, BLAZE_RODS]`. Going back for nine
    rods to replace two broken eyes is seven needless fortress fights, each one another fireball
    between the bot and its own portal.
    """
    if not _nether_topup_wanted(gs):
        return BLAZE_RODS
    short = max(0, _eyes_still_owed(gs) - _blaze_powder(gs.inventory))
    return max(1, min(BLAZE_RODS, -(-short // 2)))


def _powder_target(gs: GameState) -> int:
    """How much blaze powder the current expedition should end up with (two powder to a rod)."""
    return BLAZE_POWDER if not _nether_topup_wanted(gs) else _rods_target(gs) * 2


def _safe_blaze_hunt(gs: GameState) -> bool:
    """Whether a blaze is a fight worth starting right now.

    A blaze answers melee by igniting itself and answering with fireballs, and a fortress has
    more of them than one bot can survive at once, so the same guard `_safe_spider_hunt` uses
    applies: one target in reach, the bot healthy, and no other hostile crowding the fight.
    """
    blaze = gs.mob_nearest(BLAZE_MOB)
    if blaze is None or blaze > BLAZE_HUNT_RANGE or (gs.health or 0) < BLAZE_HUNT_HEALTH:
        return False
    others = [d for n, k, _, d in gs.mobs
              if k == "hostile" and n != BLAZE_MOB and d is not None and d <= 16]
    return not others


def _nether_must_return(gs: GameState) -> bool:
    """Whether the Nether trip has to end now, rods or no rods.

    A death down there drops the whole haul and the diamond kit, so the trip ends on the first
    sign that the bot cannot keep fighting: badly hurt, starving, in lava, or so full that the
    next blaze drop has nowhere to go. The rods already carried are worth the walk home.
    """
    if gs.in_lava:
        return True
    if gs.health is not None and gs.health < NETHER_RETURN_HEALTH:
        return True
    if gs.food is not None and gs.food < NETHER_RETURN_FOOD:
        return True
    return gs.slots_used >= INVENTORY_NEAR_FULL


def _nether_hopeless(gs: GameState) -> bool:
    """Whether the hunt is over because no blaze is left to reach.

    Blazes only ever reach the state while one is loaded, and a fortress can sit a long walk away
    with nothing between. Hunted out and with fortress bricks nowhere in sight, continuing to
    explore the Nether is not a plan - the bot goes home with the rods it has and finishes the
    chain there (a short haul still becomes powder, and the trip can be run again later).

    The wait is the point: the bot arrives in open Nether, where a `view_blocks find` reaches
    nothing but lava and the fortress is hundreds of blocks off. Judged on the first tick it would
    turn straight round (live, run31: `leave_nether` offered the tick it landed), so the bot has to
    have actually been down there, walking, before nothing counts as hopeless.
    """
    if _safe_blaze_hunt(gs) or gs.mob_nearest(BLAZE_MOB) is not None:
        return False
    if gs.resource_distance("nether_bricks") is not None:
        return False
    if gs.resource_found.get("nether_bricks", 0) > 0:
        return False
    if gs.nether_travel >= NETHER_SEARCH_M:
        return True
    # The backstop. Search is bounded by ground covered, but a bot pinned by the combat reflex
    # covers none, and something has to end the trip anyway.
    return gs.nether_ticks >= NETHER_HOPELESS_TICKS


def _spare_pickaxe_missing(gs: GameState) -> bool:
    """The trip is stocked enough to start mining diamonds for the second pickaxe.

    `craft_pickaxe_spare` only, and only when it can actually be built. Offering it before the
    diamonds were in the pack meant an unresolvable label, which is what escalates the tick to
    System 2: live, run57 burned four model calls per second on a step that could not run, each one
    handed off to a model that then chose `place_bed` or `get_wood`.

    The digging is the other half, and the tech tree never asks for it once the armour and the first
    pickaxe are done - `_next_diamond_item` returns None then - so without an explicit `mine_diamond`
    the expedition would have had no step at all for the diamonds it needs.
    """
    return _spare_pickaxe_craftable(gs)


def _spare_pickaxe_wanted(gs: GameState) -> bool:
    """Whether the banked trip still needs its second pickaxe (`NETHER_PICKAXES`).

    Split from "is it buildable" because the two have to be different questions. Offered while the
    diamonds were still in the ground, the label could not be resolved - and an unresolvable label
    is exactly what escalates the tick to System 2, one model call per second for a step that
    provably cannot run. Live, run57: four escalations on `craft_pickaxe_spare` with one diamond in
    the pack, each handed to a model that then chose `place_bed` or `get_wood`. The right answer was
    neither: the offer should have said "go and mine diamonds".
    """
    inv = gs.inventory
    return (_overworld(gs) and not _nether_haul_home(gs)
            and _expedition_prepared(gs)
            and inv.get("diamond_pickaxe", 0) < NETHER_PICKAXES)


SPARE_PICKAXE_COST = 3        # a diamond pickaxe, like the tech-tree one


def _spare_pickaxe_craftable(gs: GameState) -> bool:
    """Wanted *and* buildable right now: three diamonds and two sticks.

    `craft_diamond_pickaxe` could not do this job - it is a chain step that fires only while
    `tier < DIAMOND_PICK_TIER`, and it was never offered at all.
    """
    inv = gs.inventory
    return (_spare_pickaxe_wanted(gs) and inv.get("diamond", 0) >= SPARE_PICKAXE_COST
            and _can_sticks(inv, _logs(inv), _planks(inv)))


def _diamond_reason(gs: GameState) -> bool:
    """Whether carried diamonds still have a named purpose (a goal not built yet).

    The purpose decides what is *not* surplus: run77 shed 123 diamonds in one `drop_junk`
    while the next armour piece and the trip's second pickaxe were still waiting on them -
    the hoard cap counts them as surplus over one stack, but a goal can outlive a stack.
    """
    return _next_diamond_item(gs) is not None or _spare_pickaxe_wanted(gs)


def _diamond_dig_short(gs: GameState) -> bool:
    """Whether the next diamond goal cannot be paid for yet - so digging earns something.

    run77 was the counter-example: from the first crafted pickaxe on, `mine_diamond` stayed
    offered on every tick - the spare-pickaxe branch asked the *want* question
    (`_spare_pickaxe_wanted`) but never the *pay* question - and the bot mined 140-odd
    diamonds one-per-tick while the armour and the second pickaxe it had already paid for
    sat uncrafted. Digging is a step exactly while the goal is short of its diamonds; a
    goal that is paid for but waiting on a workstation is not cured by more digging
    either - the pack already holds what the dig would bring.
    """
    inv = gs.inventory
    nxt = _next_diamond_item(gs)
    if nxt is not None:
        return inv.get("diamond", 0) < nxt[1]
    if _spare_pickaxe_wanted(gs):
        return inv.get("diamond", 0) < SPARE_PICKAXE_COST
    return False


def _mine_diamond_wanted(gs: GameState) -> bool:
    """One answer to "is digging for diamonds the next thing", for both gates to use.

    `candidates` and `resolve` answered this differently, and the tick that found out was
    escalated to System 2 - an offered label that cannot be resolved looks exactly like a dead
    end, and costs a model call per second for a step that provably cannot run.

    The disagreement was not a missing check but a missing *reason*. The offer came from the
    Nether branch: the trip needs two pickaxes and the bot carries one, so the diamonds are wanted
    for the spare. `resolve` only knew the tech-tree order, which is finished - pickaxe, sword and
    the whole armour set done - so it read "no next diamond item" and refused. Both are now in
    this one predicate, and `resolve` names the same two reasons when it counts the diamonds.
    """
    return (_can_mine(gs, "diamond_ore") and _combat_ready(gs)
            and _diamond_ore_available(gs)
            and (_next_diamond_item(gs) is not None or _spare_pickaxe_wanted(gs))
            and _diamond_dig_short(gs))


def _nether_departure_ready(gs: GameState) -> bool:
    """Whether the Overworld gate is the bot's next job and it has to be standing at the base.

    `enter_portal` is base-gated, and none of the base steps reach a bot that wandered off once
    the expedition is stocked: the frame is already up (`_base_steps_ready`'s build clause needs an
    unlit gate), the bed is already at the base, and the chest has nothing to store. So a bot that
    drifts off after stocking the trip had no way back and the departure could never happen
    (live, run30: fully stocked, portal burning, bot 40 m off in a hole, `enter_portal` never
    offered and no step willing to walk it home).

    The gold helmet counts whether it is in the pack or already worn, not through
    `_all_nether_prep_steps`. That list drops any step whose material it cannot reach *right now*,
    so a bot with no coal has no gold step in it, the empty list reads as "prep done", and the
    crossing opens - into the Nether, where piglins are hostile and the trip dies. A burning gate
    is what makes this reachable: after a death down there the bot comes home to a gate that
    already burns and may have lost the helmet on the way. Worn counts because the bot now puts
    the gold piece on at the base before crossing (`swap_to_gold_helmet`, so no bare diamond
    helmet shows to the piglins on arrival); reading only the pack would make the gate read as
    unstocked the moment the helmet moved to the head and the departure could never finish.

    The bow is deliberately *not* treated this way, and the difference is supply, not importance:
    string needs a spider in reach and run34 waited at its own gate for hours for one, while gold
    ore is minable on demand (`mine_gold`). A gate that waits for gold waits for something that
    always arrives; a gate that waits for a bow waits for luck.

    The spare pickaxe (`NETHER_PICKAXES`) rides on `_expedition_ready`, which gates the *lighting*,
    not only this step - the opposite of where I first put it, and the live run corrected it. A
    burning portal is an open door and the pathfinder walks it like any other block, so a gate that
    lights before the kit is complete is a gate the bot crosses *by accident*: run53 stood in its
    own gate while cooking, the teleport fired mid-action, and it went down with one pickaxe and
    no crossing step at all (`cook_food` at it1782388155 in the Overworld, `the_nether` at
    it1782388156, `enter_portal` never executed). Keeping the gate dark until the trip is stocked
    means any crossing - planned or accidental - carries the kit.
    """
    return (nether_expedition_open(gs) and not _nether_haul_home(gs)
            and _overworld(gs) and _portal_lit(gs) and _expedition_ready(gs)
            and (_has(gs.inventory, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)))


def _exit_portal_cell(gs: GameState) -> tuple[int, int, int] | None:
    """The nearest `nether_portal` block cell in reach, the Nether exit as much as the way in."""
    hits = gs.resources.get(PORTAL_BLOCK) or []
    return hits[0].pos if hits else None


def _nether_way_home(gs: GameState) -> tuple[int, int, int] | None:
    """The cell the way out of the Nether stands on, when no portal block is loaded.

    Two sources, and they are not equally honest. `nether_arrival` is what this run watched
    cross - since run74 it is recorded off an observed dimension flip only, so it really is the
    portal cell, but it dies with the run. `nether_gate` is the server's memory of the last
    crossing (`perception.REMEMBERED_GATE_SQL`): it outlives the restart and the death, and it
    is the only one that answers when a run begins in the Nether having seen no way in. Neither
    answering means no way home is known, which is not a walk: run74's "arrival" was its own
    start position, and the loop walked the bot 250 blocks to a cell of bare basalt and waited
    there for a gate that never stood on it.
    """
    if gs.nether_arrival is not None:
        return gs.nether_arrival
    return gs.nether_gate


def _furnace_needs_draining(gs: GameState) -> bool:
    """Whether the standing furnace at the base holds output that blocks the next cook.

    `smelt_item` refuses a furnace whose output slot is occupied (`[FURNACE_NOT_EMPTY]`), and the
    loop's old answer was to abandon the cell and cook the next batch in a carried furnace
    (live, run26: a completed cook at a full pack left its output behind and every `cook_food`
    after answered the same refusal, burning a furnace per batch). The output is ordinary items in
    a three-slot window, so `use_container` can take them out - the server accepts `inspect` and
    `withdraw` against a furnace and refuses `deposit`/`organize` for it.

    Keyed on the refusal itself (`gs.furnace_occupied`) rather than on a guess, so the step never
    interrupts a cook that could have run: the loop only looks once the server has said the
    furnace is occupied.
    """
    return gs.furnace_pos is not None and gs.furnace_occupied and not gs.furnace_blocked


def _gold_helmet_needed(gs: GameState) -> bool:
    """Whether the diamond helmet still stands where a gold one has to go.

    One worn piece of gold armour is enough to keep piglins from attacking on sight, and the
    helmet is the slot it has to share with the diamond piece, so this is the swap both ways.

    The swap is wanted on both sides of the crossing. In the Nether it keeps a trip already down
    there neutral; at the base it goes on *before* the bot crosses, so no bare diamond helmet
    shows to the piglins on arrival (live, run99: both entries - it231, it336 - crossed in the
    diamond helmet and the gold piece went on only at it342, six Nether ticks deep, the bot
    piglin-exposed the whole gap). `_nether_departure_ready` already requires the gold helmet in
    hand, so it only ever reads true where the gate is stocked and about to open.
    """
    return ((_in_nether(gs) or _nether_departure_ready(gs)) and _worn(gs, "diamond_helmet")
            and (gs.inventory.get(GOLD_HELMET_ITEM, 0) > 0 or _worn(gs, GOLD_HELMET_ITEM)))


def _string_goal(gs: GameState) -> int:
    """String still wanted: twelve for a bed, or three for the Nether bow once the bed exists."""
    if not _bed_available(gs):
        return 12
    return 0 if _has(gs.inventory, BOW_ITEM) else BOW_STRING


def _arrows_pending(gs: GameState) -> bool:
    """The bow is built in the lull but the arrow reserve is not, so `craft_arrows` keeps
    working; also what lets `get_flint` reopen the gravel dig after flint and steel already
    exists (the only other flint sink)."""
    return (_nether_idle(gs) and _has(gs.inventory, BOW_ITEM)
            and gs.inventory.get(ARROW_ITEM, 0) < NETHER_ARROWS)


def _gold_ore_available(gs: GameState) -> bool:
    """Gold ore to dig for: within the scan, or merely present in a loaded chunk.

    Like `_diamond_ore_available`: a surface bot never lists the gold tens of blocks below it,
    but `collect_block` will tunnel to a named target, so a loaded deposit still justifies the
    dig. Without this the gold helmet could never be built from the surface.
    """
    return (gs.resource_distance(GOLD_ORE) is not None
            or gs.resource_found.get(GOLD_ORE, 0) > 0)


def _gold_goal(inv: dict[str, int]) -> int:
    """Gold ingots still wanted for the helmet, counting raw gold already banked."""
    return max(0, GOLD_HELMET_INGOTS - inv.get(GOLD_INGOT, 0) - inv.get(RAW_GOLD, 0))


def _pearls_owed(gs: GameState) -> int:
    """Ender pearls still needed to craft the eyes of ender, counting the pearls in the pack.

    The pearl half of an eye comes from piglin barter in the Nether (docs/stronghold.md), so the
    expedition sizes its gold reserve and its barter calls on this number.
    """
    return max(0, ENDER_PEARLS - int(gs.inventory.get(ENDER_PEARL_ITEM, 0)))


def _gold_ingots_wanted(gs: GameState) -> int:
    """Ingots the pack has to hold for the trip: the helmet piece, plus the barter reserve.

    The helmet is counted only while it is still in the pack - once it is crafted the five ingots
    are gone into it, and only the barter reserve remains to be banked.
    """
    inv = gs.inventory
    want = 0
    if not (_has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
        want += GOLD_HELMET_INGOTS
    if _pearls_owed(gs) > 0:
        want += GOLD_BARTER_RESERVE
    return want


def _gold_reserve_short(gs: GameState) -> bool:
    """The piglin barter reserve of gold ingots is not yet banked.

    Deliberately *not* part of `_all_nether_prep_steps`: that list is the crossing gate, and a gate
    that waits for a stack of barter gold waits for a dig the trip does not need to survive. The
    helmet is the piglin pass; the reserve is only the pearl budget. So the reserve is offered in
    the same lull but never holds the gate shut.

    The attempt cap closes the dig: a 16-ingot reserve buys about one pearl, and without the cap
    the lull would dig gold for a barter that keeps missing until the run ran out of iterations.

    Gated on the rod haul being home (`nether_expedition_complete`), because the user wants the
    barter gold dug *in the pearl wave* that spends it, not banked ahead during the rod trip's prep
    ("Gold für Welle 2 erst in Welle 2 abbauen"): the dig runs in the lull before each pearl
    crossing, so a rod trip that dies never dropped a reserve it had not yet needed.
    """
    inv = gs.inventory
    return (nether_expedition_complete(gs)
            and gs.barter_attempts < BARTER_MAX_ATTEMPTS
            and _pearls_owed(gs) > 0
            and inv.get(GOLD_INGOT, 0) + inv.get(RAW_GOLD, 0) < GOLD_BARTER_RESERVE)


def _barter_pearls_wanted(gs: GameState) -> bool:
    """Whether to trade gold with a piglin for pearls right now.

    In the Nether, pearls still owed, gold ingots in the pack, and an adult piglin within
    `BARTER_PIGLIN_RANGE`. The range is the whole point: `barter` navigates to the piglin, and a
    walk to a distant one is the same off-base drift that stalled the Overworld pearl hunt (run110)
    - so beyond the range the bot marches the frontier instead of chasing a single mob.
    """
    if not _in_nether(gs) or _pearls_owed(gs) <= 0:
        return False
    if gs.inventory.get(GOLD_INGOT, 0) <= 0:
        return False
    if gs.barter_attempts >= BARTER_MAX_ATTEMPTS:
        # The route has had its tries; the enderman hunt is the pearl source again.
        return False
    near = gs.piglin_nearest()
    return near is not None and near[1] <= BARTER_PIGLIN_RANGE


def _gold_reserve_steps(gs: GameState) -> list[str]:
    """The barter reserve dig, in order, or [] - extra ingots to trade with piglins for pearls.

    Only once the helmet is in hand: a bot that walks up to a piglin without a worn gold piece gets
    torn apart, and the helmet line already digs the gold it needs. Smelting comes before digging,
    so raw gold already in the pack becomes the ingots the barter needs first.
    """
    if not _nether_idle(gs) or not _gold_reserve_short(gs):
        return []
    inv = gs.inventory
    if not (_has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
        return []
    if (_has(inv, RAW_GOLD) and _smelting_reach(gs)
            and _furnace_fuel(inv) is not None and _furnace_room(gs)):
        return ["smelt_gold"]
    if _can_mine(gs, GOLD_ORE) and _gold_ore_available(gs):
        return ["mine_gold_reserve"]
    return []


def _nether_prep_steps(gs: GameState) -> list[str]:
    """The Nether prep to offer in the daytime lull once the gate frame stands, in order, or [].

    Each step appears only while it can actually make progress, so an unreachable material (no
    spider, no chicken, no gold ore loaded) drops that step out and never blocks the run -- the
    same shape as `hunt_food` needing a `_food_target`. Scoped to `_nether_idle`, so it never
    competes with the chain.

    The gold reserve rides on top of `_all_nether_prep_steps` rather than inside it: the reserve is
    offered in the lull, but it is not part of the crossing gate (see `_gold_reserve_short`).
    """
    steps = _all_nether_prep_steps(gs) + _gold_reserve_steps(gs)
    # The gold dig is a daylight job: it tunnels down from the base, so it is not offered at
    # night (the run sleeps then). It stays in `_all_nether_prep_steps` so the completion gate
    # still counts it as pending -- otherwise nightfall would end the run before the dig.
    if gs.phase == "night":
        steps = [s for s in steps if s not in ("mine_gold", "mine_gold_reserve")]
    return steps


def _all_nether_prep_steps(gs: GameState) -> list[str]:
    """Every prep step whose preconditions hold, without the day/night filter.

    This is the one source both `candidates` (through `_nether_prep_steps`) and
    `tech_tree_complete` read, so the offered steps and the completion gate cannot disagree on
    what the expedition still needs.
    """
    if not _nether_idle(gs):
        return []
    inv = gs.inventory
    table = _has(inv, "crafting_table")
    logs = _logs(inv)
    planks = _planks(inv)
    sticks = _can_sticks(inv, logs, planks)
    out: list[str] = []
    # Bow: three string from spiders and three sticks, crafted at the table.
    if not _has(inv, BOW_ITEM):
        if table and inv.get("string", 0) >= BOW_STRING and sticks:
            out.append("craft_bow")
        elif _safe_string_hunt(gs):
            out.append("hunt_string")
    elif inv.get(ARROW_ITEM, 0) < NETHER_ARROWS:
        # Arrows: one flint, one stick and one feather each, crafted at the table.
        if table and _has(inv, FLINT_ITEM) and _has(inv, FEATHER_ITEM) and sticks:
            out.append("craft_arrows")
        else:
            if not _has(inv, FLINT_ITEM) and gs.resource_distance("gravel") is not None:
                out.append("get_flint")
            if not _has(inv, FEATHER_ITEM) and gs.mob_nearest("chicken") is not None:
                out.append("hunt_feather")
            # Sticks run out long before the arrow reserve does - one per arrow - and nothing else
            # in the chain gathers them, so the step silently dropped out at the three sticks the
            # bow left behind and the expedition departed on half the arrows it is meant to
            # carry (live, run30: 8 arrows, nothing on offer, and no way to top them up).
            if (table and _has(inv, FLINT_ITEM) and _has(inv, FEATHER_ITEM) and not sticks
                    and logs == 0 and planks == 0):
                out.append("get_wood")
    # Gold helmet: mine the ore, smelt it, craft the piece (worn later, in the Nether).
    if not (_has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
        if inv.get(GOLD_INGOT, 0) >= GOLD_HELMET_INGOTS:
            if table:
                out.append("craft_gold_helmet")
        elif _has(inv, RAW_GOLD):
            if (_smelting_reach(gs) and _furnace_fuel(inv) is not None and _furnace_room(gs)):
                out.append("smelt_gold")
        elif _can_mine(gs, GOLD_ORE) and _gold_ore_available(gs):
            out.append("mine_gold")
    # Cobblestone: a deep stack for bridging over lava and gaps in the Nether (the arrival portal
    # is deliberately left unprotected, docs/nether-prep.md "(B) After arrival").
    if inv.get("cobblestone", 0) < NETHER_COBBLE - NETHER_COBBLE_SLACK and _can_mine(gs, "stone"):
        out.append("mine_stone")
    return out


def _nether_bank_carried(gs: GameState) -> bool:
    """The banked trip kit, judged as *carried* rather than as "no step is pending".

    `_all_nether_prep_steps` drops a step out whose material it cannot reach right now, which is
    what keeps an unreachable item from blocking the chain - but an empty list then reads as "prep
    done" to both `_expedition_prepared` and `tech_tree_complete`. Losing the bow therefore read as
    a finished Overworld goal and ended the run at its first tick (live, run32: `bow` gone, two
    string left, no spider near enough, `tech_tree_complete` True, `{"iterations": 1,
    "stop": "goal"}`). The bow and its arrow reserve are therefore checked as the inventory they
    are, so a lost one sends the run back to work rather than home.

    The gold helmet stays on the pending-step side on purpose: an unreachable material must not be
    able to block completion (`test_unreachable_prep_does_not_block_completion`), and unlike the
    bow it is not a piece the run quietly carries the whole trip without.

    This gates the *completion claim*, not the trip: `_expedition_prepared` deliberately does not
    read it, so a bot that cannot rebuild its bow can still walk through the gate. Holding the
    departure for a bow made the run wait at its own gate for an item nothing on the map could
    supply (live, run34: no spider within reach for hours), while a diamond sword and shield are
    what it actually fights blazes with.
    """
    inv = gs.inventory
    return _has(inv, BOW_ITEM) and inv.get(ARROW_ITEM, 0) >= NETHER_ARROWS


def _expedition_prepared(gs: GameState) -> bool:
    """The trip is fully stocked - only the lighting (and the bed laid right after) is left.

    The diamond kit, a bed at hand, every prep step banked and the deep food reserve met. The
    frame must stand (`_portal_built`): `_all_nether_prep_steps` reports an empty list both once
    every step is done *and* before the prep window opens, so the frame is what tells a stocked
    trip apart from one still before its lull.
    """
    return (_gear_ready(gs) and _portal_built(gs) and _bed_available(gs)
            and not _all_nether_prep_steps(gs)
            and _food_reserve_met(gs))


def _expedition_ready(gs: GameState) -> bool:
    """Whether the gate may be lit now.

    `_expedition_prepared` (the whole trip stocked, a bed at hand) and the spare pickaxe. No hour:
    `place_bed` lays the carried bed the very next tick, by day as well as by night, so no burning
    gate stands unused. `light_portal` requires this, so lighting is the last Overworld act: with
    the frame merely built, unlit, the bot can work at the base for as long as the prep takes
    without ever standing at an open gate. The hour it used to wait for cost run86 some 300
    `wait` ticks between a gate lit by day and the night its bed was held back for.

    The bed is laid *after* the gate lights, not before. The server lays a carried bed on the
    nearest flat two-block footprint, and beside an *unlit* frame that footprint is often the
    open 2x3 opening itself; `clear_portal` cannot remove it (the movement policy protects a bed
    from a route dig), so `light_portal` would answer `[PORTAL_NO_FRAME]` forever. With the gate
    lit the opening holds portal blocks and the bed falls on the ground beside the base instead.
    So the gate lights once a bed is *at hand*, and the bed is laid at once. After that the
    completion gate needs only the bed at the base.
    """
    return (_expedition_prepared(gs)
            and gs.inventory.get("diamond_pickaxe", 0) >= NETHER_PICKAXES)


# What stays in the pack while a chest stands open behind it.
#
# `JUNK_ITEMS` above is a blacklist, and a blacklist can only name the rubbish it already thought
# of: live, the pack held `oak_sapling`, `crying_obsidian`, `light_gray_wool`, `leather`,
# `stone_bricks` and `stone_brick_slab` - six slots, every one of them unusable, none of them on
# the list. The other half is that a blacklist only ever grows when someone notices, while a
# whitelist is the direction the problem runs in: anything *not* named here is surplus by default.
#
# So this is a whitelist, and it is deliberately conservative - every entry is an item that costs
# a slot to replace or a trip to fetch. The two rules that keep it from being dangerous:
#
#   * It only ever feeds `stow_surplus`, a deposit. Wrongly shedding puts the item in the chest,
#     a few blocks away and readable from the server's own database - not in a disposal pit.
#   * Away from the base the whitelist does not apply at all; `drop_junk` still goes by
#     `JUNK_ITEMS` alone, because a drop is not undoable.
KEEP_MATERIALS = frozenset({
    # Bauzeug. Cobblestone is capped at `NETHER_COBBLE` separately - one stack, one slot.
    "cobblestone", "stone", "diorite", "andesite", "granite", "tuff", "dirt", "coarse_dirt",
    "obsidian", "gravel", "sand", "clay",
    # Holz, als Brennstoff und als Bretter. `cook_food` burns a log per item.
    "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
    "oak_planks", "birch_planks", "spruce_planks", "planks",
    # Stationen und Werkzeug, die man nicht nachbauen kann, ohne die Kette zu verlieren.
    "crafting_table", "furnace", "chest", "bucket", "shield", "flint_and_steel", "bow",
    # Bogen und Pfeile.
    "arrow", "string", "feather", "flint", "stick",
    # Verarbeitetes und Erze.
    "iron_ingot", "gold_ingot", "copper_ingot", "diamond", "emerald", "lapis_lazuli",
    "coal", "charcoal", "redstone",
    # Nether-Vorbereitung.
    "blaze_rod", "blaze_powder", "raw_gold", "raw_iron", "netherite_scrap", "ancient_debris",
    # Betten. The item is `white_bed`, not `bed` - writing `bed` here kept nothing and would have
    # filed the run's only bed as surplus, which is the exact failure this whitelist has to not
    # have. The wrong colours of *wool* are not kept, but a bed always is: it cannot be made
    # again without a sheep in reach, and the Nether prep stands down until one is laid.
    "white_bed", "red_bed", "blue_bed", "black_bed", "white_wool", "red_wool", "blue_wool",
    "black_wool",
})


def _keep_in_pack(gs: GameState) -> frozenset[str]:
    """`KEEP_MATERIALS`, plus everything the bot is wearing, holding or has decided to keep.

    Gear is not written out by hand: an item the bot has on or in hand is in this set whatever it
    is called, which is what keeps a whitelist from ever shedding the pickaxe it is mining with.

    Gear is read from *two* fields on purpose. `tool_meta` names the exact item, but it is empty
    whenever the inventory has not been parsed yet, and a whitelist that filed `diamond_pickaxe`
    as surplus in that state would bank the run's best pickaxe. `tools` maps class to tier and is
    always present, so `diamond` + `pickaxe` reconstructs the name. Either field alone is a hole.
    """
    held = {meta.get("item") for meta in list(gs.armor.values()) + list(gs.tool_meta.values())}
    held |= {f"{tier}_{cls}" for cls, tier in (gs.tools or {}).items()
             if tier and tier != "none"}
    return KEEP_MATERIALS | set(EDIBLE) | set(_GEAR_KEEP) | {n for n in held if n}


def _surplus_besides_keep(gs: GameState, inv: dict[str, int]) -> list[str]:
    """Carried items that are neither on the whitelist nor worth naming twice.

    Only ever handed to the chest, never to the drop: the chest is three steps away and the
    server keeps its contents in its own database, so a wrong guess here costs one `withdraw`
    later. That asymmetry is the whole reason this function exists next to `JUNK_ITEMS` rather
    than replacing it.
    """
    keep = _keep_in_pack(gs)
    return [n for n in inv if inv.get(n, 0) > 0 and n not in keep and n not in JUNK_ITEMS]


def _go_home_wanted(gs: GameState) -> bool:
    """One reason-list for `go_home`, so the candidate gate and `resolve` cannot disagree.

    Two gates answering the same question differently is how a step gets offered and then refused
    on the same tick - and the refusal looks exactly like a dead end, which is what sends the loop
    to System 2 to explain itself. That is what happened to `craft_pickaxe_spare` in run53, and it
    was already true here for the crossing: `candidates` offered `go_home` when
    `_nether_departure_ready` held, while `resolve` knew only about `_base_steps_ready`.

    Four reasons, in the order they matter:
      some base step is waiting
      the Nether gate is lit and stocked, and the bot's next job only it can do standing there
      it is night and the bed is at the base
      wounded with nothing to heal on (`_cannot_heal`), so the base is the way out
    """
    # Never in the Nether. `base` is one field for both worlds, so its coordinates name a
    # different place down there - 778 m from where the bot actually stood, live, run70. The
    # Nether's way home is `leave_nether`, which crosses through the arrival portal; `navigate`
    # to an Overworld cell is not that, and the pathfinder "succeeds" at it because a path to
    # *some* Nether cell with those numbers exists. The same guard `_leave_portal_target` carries
    # for the same reason, written down there after run53.
    if _in_nether(gs):
        return False
    return (_base_steps_ready(gs) or _nether_departure_ready(gs)
            or (gs.phase == "night" and _bed_at_base(gs)) or _cannot_heal(gs))


def _idle_off_base(gs: GameState) -> bool:
    """A geared Overworld bot standing off its base - the state the empty-candidate fallback walks
    home from, rather than letting `explore` march it to the horizon.

    `wait` holds an idle bot only *at* the base (`BASE_RADIUS` = 4 m). The moment a step drifts it
    off - a `collect_drops` a few metres out - the empty-candidate fallback used to flip to
    `explore`, which has no tether home: each frontier step pushed the bot further, `wait` never
    re-fired, and the base was lost (live run97 it163-193: a collect_drops to 10 m, then `explore`
    ratcheted 332 -> 652 m out and oscillated there to the run's end). Walking home lands the bot
    back where `wait` can hold it. Once geared with a base, there is nothing in the Overworld worth
    walking *outward* for - the next phase waits at the gate the base holds - so home is right
    whenever the lull has run dry. The Nether is excluded: there the fallback `explore` is the
    march to the fortress, not wandering, so `_overworld` keeps this to the Overworld.

    This is a second, narrower reason than `_go_home_wanted` and shared the same way: the fallback
    offers `go_home` exactly in this state and `resolve` accepts it exactly here, so the two gates
    cannot disagree. It is deliberately read only by the empty-candidate fallback, never added to
    the ordinary offer, so no non-empty candidate set the policy was trained on changes.
    """
    return _gear_ready(gs) and gs.base is not None and not _at_base(gs) and _overworld(gs)


def _junk_carried(inv: dict[str, int]) -> list[str]:
    """Carried low-value blocks no recipe wants, in the fixed shedding order."""
    return [n for n in JUNK_ITEMS if inv.get(n, 0) > 0]


def _hoard_surplus(gs: GameState, inv: dict[str, int]) -> list[dict[str, int]]:
    """Stacks carried far past their keep/cap; drop_item supports optional count.

    Diamonds with a purpose are never surplus: the cap knows how much one keeps wearing,
    not how much the next piece or the trip's second pickaxe still wants. Without the
    exception run77's `drop_junk` shed 123 diamonds onto the ground it then walked back
    to a stack over a cap that counts wearing, not goals.
    """
    out: list[dict[str, int]] = []
    goal_named = _diamond_reason(gs)
    for name, keep in HOARD_KEEP.items():
        cnt = inv.get(name, 0)
        if cnt > keep:
            out.append({"item_name": name, "count": cnt - keep})
    for name, cnt in inv.items():
        if cnt > HOARD_CAP and name not in HOARD_KEEP:
            if name == "diamond" and goal_named:
                continue
            out.append({"item_name": name, "count": cnt - HOARD_CAP})
    return out


def _spare_gear(gs: GameState, inv: dict[str, int]) -> list[str]:
    """Carried tools and armour below the best the bot already carries, wears or holds.

    The pack fills with the kit the chain upgraded through (wooden/stone/iron picks, an iron
    sword and iron armour once the diamond set is on). No recipe here wants them and
    `_junk_carried` cannot see them, so a full pack of nothing else never sheds. The "best"
    counts carried, worn and held gear, so the bot only drops pieces a strictly better one
    replaces, and `_GEAR_KEEP` (the gold helmet) is never dropped.
    """
    out: list[str] = []
    for kinds, held in ((_TOOL_KINDS, gs.tools), (_ARMOR_KINDS, {})):
        for kind in kinds:
            best = 0
            for material, rank in _GEAR_RANK.items():
                name = f"{material}_{kind}"
                if (inv.get(name, 0) or _worn(gs, name)
                        or held.get(kind) == material):
                    best = max(best, rank)
            for material, rank in _GEAR_RANK.items():
                name = f"{material}_{kind}"
                if rank < best and name not in _GEAR_KEEP and inv.get(name, 0):
                    out.append(name)
    return out


def _droppable(gs: GameState, inv: dict[str, int]) -> bool:
    """What may go on the ground: named junk, spare gear, a crossed cap. Never the whitelist.

    The one-way rule of `KEEP_MATERIALS` - it feeds `stow_surplus` and nothing else - is not a
    preference, it is what makes the whitelist safe. A wrongly shed item lands in a chest three
    steps away and costs one `withdraw`; the same item dropped is gone for good. So the gate that
    offers `drop_junk` has to name exactly the items its resolve will name, and the whitelist is
    not among them.
    """
    return bool(_junk_carried(inv) or _spare_gear(gs, inv) or _hoard_surplus(gs, inv))


def _sheddable(gs: GameState, inv: dict[str, int]) -> bool:
    return bool(_junk_carried(inv) or _hoard_surplus(gs, inv) or _spare_gear(gs, inv)
                or _surplus_besides_keep(gs, inv))


def _combat_blocks_carried(inv: dict[str, int]) -> int:
    """How many blocks the bot can scaffold a fight with (`COMBAT_SCAFFOLD_BLOCKS`).

    The server builds a defensive roof out of these while fighting, and refuses the engagement with
    `[COMBAT_BUILD_MATERIALS_MISSING]` when too few are carried. Netherrack counts as readily as
    cobblestone, so the Nether stock and the Overworld stack are summed together.
    """
    return sum(int(inv.get(name, 0)) for name in COMBAT_SCAFFOLD_BLOCKS)


def _chest_ready(gs: GameState) -> bool:
    """A chest has been placed at the base (its coordinates are remembered)."""
    return gs.chest_pos is not None


def _chest_network(gs: GameState) -> list[tuple[int, int, int]]:
    """Every chest cell the base keeps: the main one first, then the overflow in order."""
    cells: list[tuple[int, int, int]] = []
    if gs.chest_pos is not None:
        cells.append(tuple(gs.chest_pos))
    for cell in gs.extra_chests:
        c = tuple(cell)
        if c not in cells:
            cells.append(c)
    return cells


def _storage_target(gs: GameState) -> tuple[int, int, int] | None:
    """The network chest that should hold the next deposit: the first one the server has not
    already called full. `None` means the network is full through -- a deposit anywhere in it
    would only earn the same `[CONTAINER_BATCH_INCOMPLETE]` again, and the relief builds
    another chest instead of pouring into closed lids.
    """
    full = set(gs.chests_full)
    for cell in _chest_network(gs):
        if cell not in full:
            return cell
    return None


def _storage_reachable(gs: GameState) -> bool:
    """Whether the network's next roomy chest stands within deposit reach of the bot.

    `stow_surplus` works a chest where it stands (`_near_cell`, not the base anchor), so a
    surplus this close is stored now - and is no reason to walk to the base. run82 ran 481
    `go_home` and `stow_surplus` walks between (-45,71,15) and (-43,71,21), every one "ok"
    and walking, a 35-slot pack never shed a block: the deposit was only ever offered at the
    base anchor, while the chest it walked to stood outside that anchor's radius.
    """
    target = _storage_target(gs)
    return target is not None and gs.pos is not None and _near_cell(gs, target)


def _shed_owed(gs: GameState, inv: dict[str, int]) -> bool:
    """Whether a near-full pack owes the roomy chest its surplus before the next trip out.

    The same condition that calls the bot home (`_base_steps_ready`'s surplus clause), so a
    step that walks away from the base must not be offered while it holds: run84 walked
    `mine_obsidian` 27 blocks down to the lava at a full 36/36 pack, `go_home` called it straight
    back on the surplus before the pour, and at the base Laya chose the walk down again over
    `stow_surplus` - a `navigate` that succeeds every time, so no guard ever counted it.
    """
    return (_chest_ready(gs) and gs.slots_used >= INVENTORY_NEAR_FULL
            and _sheddable(gs, inv) and not _storage_full(gs))


def _storage_full(gs: GameState) -> bool:
    """Whether every chest standing at the base the server has declared full."""
    network = _chest_network(gs)
    return bool(network) and all(cell in set(gs.chests_full) for cell in network)


def _overflow_site(gs: GameState) -> tuple[int, int, int] | None:
    """A cell for the next overflow chest, or None once none can stand.

    The candidates march behind the base in +z, past the furnace, the smoker and the frame
    plane; a cell that already has a chest, a bed or the frame's obsidian on it is skipped, as
    is one the ground refused a chest at (`[PLACE_NO_SUPPORT]`, kept per cell so the whole
    chain stays open -- the main ground's refusal is that, a bed the frame cannot use is this).
    A cell merely *occupied* is not this pick's problem: `make_room` digs it free and the same
    cell retries.
    """
    if gs.base is None:
        return None
    bx, by, bz = gs.base
    taken = set(_chest_network(gs))
    for name in ("white_bed", "obsidian"):
        taken.update(tuple(hit.pos) for hit in gs.resources.get(name, []))
    for station in (gs.furnace_pos, gs.smoker_pos):
        if station is not None:
            taken.add(tuple(station))
    taken |= set(gs.overflow_skip)
    for dx, dz in STORAGE_SITES:
        cell = (bx + dx, by + 1, bz + dz)
        if cell not in taken:
            return cell
    return None


def _near_cell(gs: GameState, cell: tuple[int, int, int],
               reach: float = STORAGE_REACH) -> bool:
    """Whether the bot stands over a cell's centre close enough to work it directly.

    A chest's cell sits at the bot's feet level (`by + 1` is the floor the bot walks on), so the
    height test is loose -- a chest one step up a ledge is still in reach of the lid.
    """
    if gs.pos is None:
        return False
    cx, cy, cz = cell
    return (abs(gs.pos[0] - (cx + 0.5)) <= reach and abs(gs.pos[2] - (cz + 0.5)) <= reach
            and abs(gs.pos[1] - cy) <= 2.5)


def _banked_stores(gs: GameState) -> list[tuple[tuple[int, int, int], dict[str, int]]]:
    """The provisions each network chest holds, as `[(cell, {item: count}), ...]`.

    The main chest leads the list -- it is where the bot stands most of the time -- and junk
    stays out of it the same way `_banked_provisions` keeps it out of the aggregate: a store of
    rotten flesh alone is not a larder, and a `take` that walks to fetch it is the round trip
    `stow_surplus` would immediately undo.
    """
    stores: dict[tuple[int, int, int], dict[str, int]] = {}
    for entry in (gs.banked_stores or ()):
        try:
            cell, name, count = entry
            cell = tuple(int(v) for v in cell)
            count = int(count)
        except (TypeError, ValueError):
            continue
        if count > 0 and name in PROVISIONS:
            stores.setdefault(cell, {})[name] = stores.setdefault(cell, {}).get(name, 0) + count
    ordered: list[tuple[tuple[int, int, int], dict[str, int]]] = []
    if gs.chest_pos is not None and tuple(gs.chest_pos) in stores:
        main = tuple(gs.chest_pos)
        ordered.append((main, stores.pop(main)))
    ordered.extend((cell, counts) for cell, counts in stores.items())
    return ordered


def _furnace_site(gs: GameState) -> tuple[int, int, int] | None:
    """The cell the base furnace goes on, or None when it cannot stand there.

    One block out from the base on the side opposite the chest, clear of the portal's frame
    plane (which runs along z == bz). The one refusal is a placed bed on that exact cell: the
    placement would answer `[PLACE_TARGET_OCCUPIED]` and `make_room` would dig the respawn bed
    away. The bed's cells are scanned (`white_bed`), so the furnace simply stands nowhere there
    and cooking falls back to the temporary workstation.
    """
    if gs.base is None:
        return None
    bx, by, bz = gs.base
    cell = (bx, by + 1, bz - 1)
    if any(hit.pos == cell for hit in gs.resources.get("white_bed", [])):
        return None
    return cell


def _smoker_site(gs: GameState) -> tuple[int, int, int] | None:
    """The cell the base smoker goes on: beside the furnace, clear of the chest and the frame.

    The furnace takes `(bx, by+1, bz-1)`, the chest sits at `(bx, by+1, bz+1)` and the portal's
    frame plane runs along `z == bz`, so the cell next to the furnace on the other side of x is
    the one that is free of all three. The bed refusal is the furnace's: a bed on that cell would
    answer `[PLACE_TARGET_OCCUPIED]`, and `make_room` would dig away the respawn bed to clear it.
    """
    if gs.base is None:
        return None
    bx, by, bz = gs.base
    cell = (bx + 1, by + 1, bz - 1)
    if any(hit.pos == cell for hit in gs.resources.get("white_bed", [])):
        return None
    return cell


def _smoker_worth_placing(gs: GameState) -> bool:
    """Whether standing a smoker at the base is the next thing to do there.

    Food only, and only once there is food to cook and a way to light it. Cooking is the one
    thing this run does in bulk - live, twenty cooked items in twenty ticks, one per tick at
    eighteen seconds - and it is all food, so the station that doubles the smelt rate is worth
    the recipe. The materials are asked by `craft_smoker` alone (`_smoker_craft_ready`): the
    placement step must not refuse a smoker it already carries because of a material question
    it no longer answers.
    """
    if gs.base is None or gs.smoker_pos is not None or not _at_base(gs):
        return False
    return _first_cookable(gs.inventory) is not None


def _smoker_craft_ready(gs: GameState) -> bool:
    """The server's smoker recipe: four logs *and* a carried furnace - and a second furnace.

    "four planks and a coal" was never a smoker: live, run78 offered the craft with one log
    carried and the server answered `[CRAFT_MATERIALS_MISSING] oak_log x3` - the recipe wants
    four logs plus one furnace, and that furnace is a material too. Requiring a second carried
    furnace is not greed but the cook chain: with the recipe's furnace consumed, the station
    for `cook_food` and the iron of `smelt_iron` is gone with it.
    """
    inv = gs.inventory
    return (_has(inv, "crafting_table") and _logs(inv) >= SMOKER_LOGS
            and inv.get("furnace", 0) >= SMOKER_FURNACES)


def _furnace_worth_placing(gs: GameState) -> bool:
    """Whether standing the carried furnace at the base is the next thing to do there.

    One predicate for both halves of that job, so the two cannot disagree. `candidates` offers
    `place_furnace` at the base and `go_home` away from it; `_base_steps_ready` separately owes
    the walk home. Those were two independent copies of the same question, and they answered it
    differently: `go_home` fired on the first condition, `place_furnace` on the second, so the bot
    was walked home for a step that was then never offered - 341 of 377 records carrying a furnace
    and not one `place_furnace` to show for it (live, run53).

    The furnace is an optimisation, not a requirement: `_smelt_args` falls back to a temporary
    workstation, so cooking and smelting work anywhere with the carried furnace. That is why the
    window is the lull (`_nether_idle`) - before it the bot is busy with the chain - and why it
    needs no second gate: a walk home is owed exactly when the placement can follow it.
    """
    inv = gs.inventory
    return (_nether_idle(gs) and gs.base is not None and not _furnace_ready(gs)
            and not gs.furnace_blocked and _has(inv, "furnace")
            and _furnace_site(gs) is not None)


def _furnace_ready(gs: GameState) -> bool:
    """A furnace stands at the base (its coordinates are remembered).

    When it does, `smelt_item` gets the furnace's coordinates and never places a temporary
    workstation, so the same furnace is reused instead of being set up and torn down on every
    `cook_food`/`smelt_gold` call (live: the bot visibly tore the furnace down between batches).
    """
    return gs.furnace_pos is not None


def _furnace_room(gs: GameState) -> bool:
    """Whether the pack has a free slot for the output `smelt_item` has to recover.

    A smelt that completes with no room leaves its cooked output stuck in the furnace (live,
    run26), and a furnace with pre-existing output refuses the next insert forever. Requiring a
    free slot keeps a cook from stranding itself; when the pack is full the loop sheds
    (`stow_surplus`/`drop_junk`) first.
    """
    return gs.slots_used < INVENTORY_SLOTS


def _cooking_station(gs: GameState) -> tuple[int, int, int] | None:
    """The cell food should be smelted at: the smoker if one stands at the base, else the furnace.

    A smoker is not a better furnace, it is a different one: it processes *food only* and runs at
    double speed. The tech tree still needs the furnace for iron, so the smoker stands beside it
    rather than replacing it - and that is the user's shape for the base, the bot coming home to
    cook in bulk what it gathered out there, instead of cooking one item per tick wherever it
    happens to stand.
    """
    if not _at_base(gs):
        return None
    return gs.smoker_pos if gs.smoker_pos is not None else gs.furnace_pos


def _cooking_reach(gs: GameState) -> bool:
    """Where a cook can actually run: standing at the base with a station standing there, or
    carrying the furnace. The same options `resolve` will bind, one rule for offer and binding.

    run78: eight `cook_food` in a row ran into the server's `[WORKSTATION_NOT_CARRIED]`.
    `_furnace_ready` answers "a furnace stands at the base" even when the bot is 200 m out on a
    hunt and carries nothing - but `_smelt_args` names the station's cell only where the body
    stands at it, and otherwise asks for a *temporary* workstation, which is a carried furnace.
    """
    return _has(gs.inventory, "furnace") or _cooking_station(gs) is not None


def _smelting_reach(gs: GameState) -> bool:
    """The same distinction for ore: a smoker would refuse raw iron, so only the furnace counts."""
    return _has(gs.inventory, "furnace") or (_at_base(gs) and gs.furnace_pos is not None)


def _smelt_args(gs: GameState, item_name: str, count: int, fuel_name: str,
                station: tuple[int, int, int] | None = None) -> dict:
    """The `smelt_item` arguments: the cell of the station that suits this input, or a portable one.

    With a station standing at the base *and the bot standing there too*, the call names its exact
    cell and reuses it; otherwise the carried furnace is placed and recovered for this one call.

    Both conditions, not one. Knowing the cell is not the same as standing at it, and naming a
    furnace 190 m away sends the cook to a target the bot has no reason to walk to: live, run58, a
    `cook_food` at `(357,70,362)` with `furnace_pos` `(229,62,222)` in the state - a smelt aimed at
    the base from the far side of the map, answered ok. The temporary workstation is the branch that
    was always there and always worked, so away from the base it is simply the right one.

    `station` is passed rather than inferred: iron has to go in the furnace and food in the
    smoker, and a smoker would refuse the ore.
    """
    args: dict = {"item_name": item_name, "count": count, "fuel_item_name": fuel_name}
    if station is None:
        station = gs.furnace_pos if _at_base(gs) else None
    if station is not None and _at_base(gs):
        fx, fy, fz = station
        args.update({"x": fx, "y": fy, "z": fz})
    else:
        args["temporary_workstation"] = True
    return args


def _obsidian_within_reach(gs: GameState) -> bool:
    """Obsidian close enough to mine: what the last water pour formed lies at the bot's feet."""
    hit = gs.resource_nearest("obsidian")
    return hit is not None and hit.distance <= OBSIDIAN_REACH


def _within_reach(gs: GameState, hit, reach: float) -> bool:
    """Whether the bot stands within `reach` of a `view_blocks` hit (needs a known position)."""
    return gs.pos is not None and _dist(gs.pos, hit.pos) <= reach


# The four horizontal steps around a lava cell: the bucket clicks a solid block *beside* the
# pour target, so only a lateral neighbour can be the stance the bot pours from.
_LATERAL = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))


def _lava_pour_target(gs: GameState) -> BlockHit | None:
    """The nearest lava cell the carried water can actually turn into obsidian, or None.

    Three rules, each paid for live on the surface pond:
      * The water must meet a lava *source* (`level == 0`); flowing lava yields cobblestone.
        Pouring straight *into* a source deletes it, so the target is a lava cell *beside* a
        source -- the poured water then flows onto the source and makes obsidian.
      * The bucket clicks a solid block beside the target cell, so a target over lava or air
        answers `[BUCKET_NO_FACE]`; its `below` must be solid (no air variant and not lava).
      * The cell must be open above: `above` is one of the server's air variants, since
        `cave_air` under an overhang is as good as open `air`. Requiring exactly `"air"` threw
        every cave source away and the run never offered `mine_obsidian` at all (live, from-zero
        run: both lava sources in scan reach sat under `cave_air`, so the bot wandered the
        surface with `explore` alone while the portal chain waited for obsidian).
    Cells the server already refused (`gs.pour_skip`) are passed over, so a stopped-up pond
    does not starve the run of the lava lake further off.
    """
    hits = gs.resources.get("lava") or []
    if not hits:
        return None
    sources = {h.pos for h in hits if h.level == 0}
    if not sources:
        return None
    for h in hits:  # already nearest first
        if (h.pos in gs.pour_skip or h.above not in AIR_NAMES
                or h.below in (None, "lava") or h.below in AIR_NAMES):
            continue
        if any((h.x + dx, h.y, h.z + dz) in sources for dx, _, dz in _LATERAL):
            # A source whose every walkable stance the server already refused has no footing to
            # pour from (see `_pour_stance`), so the search moves to the next source -- the same
            # move `pour_skip` makes for a refused pour. A cell within pouring reach is poured at
            # directly, so it is not passed over.
            if not _within_reach(gs, h, POUR_REACH):
                options = _stance_options(gs, h)
                if options and all(o in gs.stance_skip for o in options):
                    continue
            return h
    return None


def _stance_options(gs: GameState, target: BlockHit) -> list[tuple[int, int, int]]:
    """The cells to stand in to pour into `target`: *on top of* a lateral neighbour.

    The bucket clicks a solid block *beside* the pour target, so only a lateral neighbour can be
    the stance the bot pours from. Lava neighbours are skipped: the bot may not stand there, and
    there is nothing to stand on either.
    """
    lava = {h.pos for h in (gs.resources.get("lava") or [])}
    return [(target.x + dx, target.y + 1, target.z + dz)
            for dx, _, dz in _LATERAL
            if (target.x + dx, target.y, target.z + dz) not in lava]


def _pour_stance(gs: GameState, target: BlockHit) -> tuple[int, int, int] | None:
    """The cell to stand in to pour into `target`: *on top of* a lateral neighbour.

    Standing at the target's own height is what the live lake pour failed on. The lateral cell
    beside a lava pool is solid rock, and the server refuses to dig that last block next to the
    liquid ("breaking this block would open the route into liquid"), so the pathfinder stopped a
    cell short and diagonally -- where the reference block's face points past the bot and every
    pour answers `[BUCKET_NO_FACE]`. Standing on top of the neighbour is reachable, and that is
    the stance the live lake pour formed obsidian from. Stances the server already refused
    (`gs.stance_skip`, `[NAVIGATION_TARGET_UNSUPPORTED]`: no footing there) are passed over, so a
    stance over a deep lava lake is not walked to twice (live, run16: 215 identical refusals).
    """
    options = [p for p in _stance_options(gs, target) if p not in gs.stance_skip]
    if not options:
        return None
    if gs.pos is None:
        return options[0]
    return min(options, key=lambda p: _dist(gs.pos, p))


def _lava_walk_target(gs: GameState) -> tuple[int, int, int] | None:
    """The stance to walk to when the scan holds lava but no pour-able source.

    A lava *source* only enters the scan once the bot is near it, and `mine_obsidian` is offered
    only while one is. Live, run114: the base anchor sat 118 blocks from the pond that made the
    run's three obsidian, and the base scan listed only two *flows* (level 2 and level 8) 60 m
    away - no level-0 source, so `_lava_pour_target` answered None, `mine_obsidian` closed, and the
    offer fell to the `explore`/`go_home` fallback for 66 ticks while the frame still owed seven
    blocks. The walk is the chain's missing link: standing beside the lava loads the source, and
    the pour step opens on the next tick.
    """
    for h in gs.resources.get("lava") or []:  # already nearest first
        if _within_reach(gs, h, POUR_REACH):
            continue  # already at the lava: `mine_obsidian` owns that tick
        if h.pos in gs.pour_skip:
            continue
        stance = _pour_stance(gs, h)
        if stance is not None:
            return stance
    return None


def _bed_available(gs: GameState) -> bool:
    """A bed is at hand for `sleep`: carried, or already standing beside the base.

    Counting a *placed* bed matters once the bed item has left the pack (a manual base build, or
    a restart): a carried-bed-only test would send the loop off to hunt wool for a second bed
    while one already stands at the base.
    """
    return _has_bed(gs.inventory) or _bed_placed(gs)


def _sleep_ready(gs: GameState) -> bool:
    """Whether `sleep` may run where the bot stands.

    Before the gate is lit the bot sleeps *away* from the base: the server lays the bed "beside
    the bot", and at the base that spot is the portal's own interior, which the later frame build
    or `light_portal` cannot clear (`remove_wrong_blocks` answers "movement policy prohibits
    breaking this block" for a bed). Once the gate is lit the gate bed belongs at the base, and it
    has to be a bed *standing at the base*: counting any loaded bed sent the server to the leftover
    bed in the base's own shaft and pinned the respawn point off the gate again (live, run19b
    it25: the bot stood at the base and slept in the bed nine blocks below). A carried bed is
    `place_bed`'s job first; `take_bed` brings a leftover one home. The switch is the *lit* gate
    (`_portal_lit`), not the bare frame: with the frame up but unlit the opening is still open
    air, so a bed laid there blocks `light_portal`; only once it burns is a base bed safe.
    """
    if _portal_lit(gs):
        return _at_base(gs) and _bed_at_base(gs)
    return (not _at_base(gs) and not (gs.pos is not None and _underground(gs, gs.pos[1]))
            and _bed_in_sleep_reach(gs))


# How far west of the base anchor the night walk goes before a carried bed may be slept in
# (`_sleep_off_base`). The frame stands east of the anchor (`perception.frame_obsidian_cells`,
# x+1..x+4), so west leads away from it, and three blocks past `BASE_RADIUS` is off the base
# box with the navigate's arrival tolerance to spare.
SLEEP_OFF_BASE_STEP = int(BASE_RADIUS) + 3


def _sleep_off_base(gs: GameState) -> bool:
    """Night at the base with an unlit frame and a carried bed: walk off the base to sleep.

    `_sleep_ready` forbids sleeping at the base before the gate burns (the bed would land in the
    open 2x3 opening) and allows it anywhere else - but nothing walked the bot anywhere else. With
    the gate still waiting for its spare pickaxe and no diamond ore in view, the night offered
    `wait` alone: run87 stood twelve ticks at its base, bed in the pack, until dawn (it147-158).
    The walk west leaves the frame behind; the bed it lays there is a leftover `take_bed` brings
    home in the day, as it does for every bed slept in away from the base. A gate that may be lit
    now (`_expedition_ready`) keeps the bot at the base: lighting it and `place_bed` beside it is
    the night's work, and the end of the Overworld.
    """
    return (_overworld(gs) and not _portal_lit(gs) and _at_base(gs)
            and _has_bed(gs.inventory) and gs.base is not None
            and not _expedition_ready(gs))


def _underground(gs: GameState, feet_y: float) -> bool:
    """Whether feet (or a bed) at `feet_y` lie well below the surface the base stands on."""
    floor = gs.base[1] + 1 if gs.base is not None else SURFACE_FALLBACK_Y
    return feet_y < floor - SLEEP_DEPTH_TOL


def _sleep_safe(gs: GameState) -> bool:
    """No hostile close enough for the game to refuse the sleep (`SLEEP_HOSTILE_CLEAR`)."""
    return gs.nearest_hostile is None or gs.nearest_hostile > SLEEP_HOSTILE_CLEAR


def _unsafe_beds(gs: GameState) -> list:
    """Standing beds laid underground: each one may hold the respawn point inside a cave."""
    return [hit for hit in gs.resources.get("white_bed") or [] if _underground(gs, hit.y)]


def _bed_in_sleep_reach(gs: GameState) -> bool:
    """A bed the `sleep` server can find from here: carried, or standing within its search.

    `_bed_available` counts any bed the scan sees, which is right for "is a second bed owed",
    and wrong for the sleep itself: run86 started at night 51 blocks from the old base's bed,
    the server searches 32, and the exclusive night offer answered `[BED_NOT_FOUND]` on sixteen
    ticks until the morning.
    """
    if any(hit.distance <= SLEEP_BED_REACH for hit in _unsafe_beds(gs)):
        # The server sleeps in a standing bed before it lays the carried one, so a cave bed in
        # its search would take the sleep - and the respawn - back underground.
        return False
    if _has_bed(gs.inventory):
        return True
    distance = gs.resource_distance("white_bed")
    return distance is not None and distance <= SLEEP_BED_REACH


def _at_base(gs: GameState) -> bool:
    """Whether the bot stands at the home base, close enough to lay blocks out there."""
    if gs.base is None or gs.pos is None:
        return False
    bx, by, bz = gs.base
    x, y, z = gs.pos
    return (abs(x - (bx + 0.5)) <= BASE_RADIUS
            and abs(z - (bz + 0.5)) <= BASE_RADIUS
            and abs(y - (by + 1)) <= BASE_RADIUS + 1)


def _at_surface(gs: GameState) -> bool:
    """Whether the bot stands at open surface ground, high enough to anchor the base.

    `gs.surface` is the surface cell the loop last learned for one column (`x, feet_y, z`).
    A cave floor sits far below it, while a mined tunnel's `air` overhead is indistinguishable
    from sky, so the base gate is this depth test rather than any `view_blocks` reading: the bot
    is at the surface only when it stands in that same column and its feet rest on the learned
    floor. Unknown surface, a different column, or feet below that floor is never "at the
    surface" -- the climb has to prove it first.

    The column test is load-bearing: the cell was learned for the probed column, and a bot that
    mined its way under a *different* column must not read as "at the surface" merely because
    that column's height happens to be close. Live, run10: feet on a y=54 cave floor sat within
    the old eight-block tolerance of a y=59 surface learned in another column, so
    `establish_base` fired underground and the walk to it wedged the bot in a flooded shaft.
    `BASE_SURFACE_TOL` now only absorbs the odd slab or one-block dip at the column's own
    surface.
    """
    if gs.surface is None or gs.pos is None:
        return False
    # Don't consider it "at surface" if in water - could be seabed
    if getattr(gs, "in_water", False):
        return False
    sx, sy, sz = gs.surface
    if (math.floor(gs.pos[0]), math.floor(gs.pos[2])) != (sx, sz):
        return False
    # If surface is much below us and we're near water, it's likely seabed
    if sy < gs.pos[1] - 15:
        return False
    return abs(gs.pos[1] - sy) <= BASE_SURFACE_TOL


def _base_steps_ready(gs: GameState) -> bool:
    """Whether some base step is ready to run once the bot is back at the base.

    Used only to justify `go_home`: the bot walks back exactly when there is work that needs
    the base (a frame to raise, a bed to lay, a chest to place once the pack fills, surplus to
    store), never as an idle stroll.
    """
    inv = gs.inventory
    # A base re-adopted from a lit portal a previous run left standing is not reached yet: walk
    # back to it once, so the run finishes at its own gate instead of drifting off to explore.
    # Cleared on arrival (`graph._observe`), so it cannot pull the bot off its work later.
    if gs.base_adopted:
        return True
    can_build = (inv.get("obsidian", 0) >= PORTAL_OBSIDIAN
                 or (inv.get("obsidian", 0) >= 1 and _portal_frame_started(gs)))
    if (_has(inv, FLINT_STEEL_ITEM) and can_build
            and inv.get("cobblestone", 0) >= PORTAL_CORNERS and not _portal_lit(gs)):
        return True
    if _portal_lit(gs) and _has_bed(inv) and not _bed_at_base(gs):
        # Not gated on the night: a bed carried out of the base's own shaft (taken up by
        # `take_bed`) belongs home whatever the hour, and `place_bed` lays it by day. Gated on the
        # night, the bot carried it around all day and never came back (live, run19b it39-40:
        # `take_bed` succeeded in daylight, `go_home` was not offered, and the bed was never laid).
        # Keyed on the *lit* gate, not the bare frame: a carried bed cannot be laid until the gate
        # burns (see `_sleep_ready`), so drawing it home through the unlit prep window would only
        # pull the bot off `mine_gold` and the hunts that leave the base.
        return True
    # The whole trip is stocked and only the lighting is left: it happens at the base (and the
    # bed laid the very next tick), so walk home rather than drift. `light_portal` itself is
    # `_at_base`-gated, so without this the bot that finished the prep away from the base (a gold
    # shaft under it, a feather hunt nearby) would never come back to light the gate. Keyed on
    # `_expedition_prepared`, not `_expedition_ready`: the walk home is worth it before the spare
    # pickaxe the lighting also waits for is made.
    if not _portal_lit(gs) and _expedition_prepared(gs):
        return True
    # In the lull, a carried furnace belongs at the base: walking back lets `place_furnace`
    # stand it there, after which cooking reuses it rather than tearing a temporary one down.
    # The same predicate the offer reads, so the walk is never owed for a step that would not
    # follow it (see `_furnace_worth_placing`: run53 walked home 341 records over a furnace and
    # was never offered the placement).
    if _furnace_worth_placing(gs):
        return True
    # A carried chest only pulls the bot home once the pack is filling. The chest is placed for
    # the room it frees, not for its own sake, and a light pack owes it nothing: live, run99
    # it153-165, a bot on 29 of 36 slots hunted the trip's food 70 m out, and the carried chest
    # walked it back every time it arrived - `go_home` and `hunt_food` handed each other the same
    # 70 m four times over. At the base the chest is still offered (`place_chest`), so the early
    # placement a bot that happens to be home can make is not lost; only the walk is deferred.
    if (not gs.chest_blocked and _has(inv, CHEST_ITEM) and not _chest_ready(gs)
            and gs.slots_used >= INVENTORY_NEAR_FULL):
        return True
    if (_chest_ready(gs) and gs.slots_used >= INVENTORY_NEAR_FULL and _sheddable(gs, inv)
            and not _storage_reachable(gs)):
        # Surplus within reach of the roomy chest is stored now, not carried home first: the
        # walk back and the walk over share one surplus, and where the chest stands outside
        # the base's radius the two hand the bot off to each other forever (run82, 481 ticks).
        return True
    return False


def _portal_iron_goal(gs: GameState) -> int:
    """Ingots the Nether gate still wants: three for the bucket, then one for flint and steel.

    The bucket is checked first because it is the larger spend and consumes its ingots before
    the flint and steel is crafted; each is wanted until that item is carried.
    """
    inv = gs.inventory
    if not (_has(inv, BUCKET_ITEM) or _has(inv, WATER_BUCKET)):
        return BUCKET_IRON
    if not _has(inv, FLINT_STEEL_ITEM):
        return FLINT_STEEL_IRON
    return 0


def _combat_iron_missing(gs: GameState) -> int:
    """Ingots still to bank for the pre-diamond kit: the shield, the sword and each missing piece.

    Counts only what is not built yet, so the goal falls as the pieces are crafted and no ingot
    is quarried twice.
    """
    need = 0
    if not _has_shield(gs):
        need += SHIELD_IRON
    if not _has_iron_sword(gs):
        need += IRON_SWORD_INGOTS
    for slot, cost in IRON_ARMOR:
        if not _armor_slot_covered(gs, slot):
            need += cost
    return need


def _iron_goal(gs: GameState) -> int:
    """Ingots (or raw iron) still wanted: three build an iron pickaxe, or its fresh spare.

    Once that pickaxe exists nothing needs iron any more, so the loop would stop digging it -
    but the shield wants one ingot, and once the whole kit is worn the portal gate wants more
    (a bucket, then flint and steel). This is the single source both `candidates` and
    `resolve` use, so those pulls never reopen a mindless iron quarry.
    """
    inv = gs.inventory
    tier = _pick_tier(inv, gs.tools)
    goal = 3 if (tier < IRON_PICK_TIER or _pickaxe_due_for_spare(gs)) else 0
    if tier >= IRON_PICK_TIER:
        # The pre-diamond kit (shield, iron sword, iron armour) is the reason to keep digging
        # iron after the pickaxe exists; `_combat_iron_missing` is 0 once it is complete.
        goal = max(goal, _combat_iron_missing(gs))
    # `_gear_ready` (not just "has a shield") is the gate, so banking the bucket's three
    # ingots waits until the diamond kit is complete - matching where `craft_bucket` appears.
    if _gear_ready(gs) and not _portal_lit(gs):
        goal = max(goal, _portal_iron_goal(gs))
    return goal


def _portal_site(gs: GameState) -> dict | None:
    """Where to raise the portal frame: its six interior cells start two blocks east of the
    base anchor (or, with no base yet, the bot's own feet), one block above the ground the
    frame stands on. The frame then runs from one block west of that corner out to the east,
    with the bot standing just clear of its western face, so it is never built into itself."""
    if gs.base is not None:
        bx, by, bz = gs.base
        return {"x": bx + 2, "y": by + 1, "z": bz, "axis": "x"}
    if gs.pos is None:
        return None
    mx, my, mz = (math.floor(v) for v in gs.pos)
    return {"x": mx + 2, "y": my, "z": mz, "axis": "x"}


def _portal_frame_started(gs: GameState) -> bool:
    """Whether obsidian already stands where our frame goes.

    The frame is a fixed plane relative to the base (`_portal_site`), so one obsidian hit
    inside that plane means a build ran -- possibly only partly. That is the signal which lets
    `build_portal` resume with what the bot still carries: after a partial build the carried
    count equals the cells left, so the frame can always be finished with the remainder,
    without first re-banking ten blocks (which is impossible once flint and steel is crafted).
    """
    site = _portal_site(gs)
    if site is None:
        return False
    sx, sy, sz = site["x"], site["y"], site["z"]
    # The exact frame-plane read first: `gs.resources` keeps only the four nearest obsidian hits
    # inside `RESOURCE_RADIUS`, so a deep deposit (or an older frame one block over) crowds our
    # own obsidian out and the loop believes no frame stands (live, run28: it re-mined ten
    # obsidian after the frame was already up, then could never light the finished gate).
    if gs.base is not None and gs.frame_names:
        if any(gs.frame_names.get(cell) == "obsidian"
               for cell in frame_obsidian_cells(gs.base)):
            return True
    for hit in gs.resources.get("obsidian", []):
        if hit.z == sz and sx - 1 <= hit.x <= sx + 2 and sy - 1 <= hit.y <= sy + 3:
            return True
    return False


def _frame_blocker(gs: GameState) -> tuple[tuple[int, int, int], str] | None:
    """The first frame cell the build refused that `clear_frame` can dig, with its block name.

    `build_structure` digs a wrong frame cell in place, but the movement policy refuses a break it
    was not asked for, and a *container* always needs that explicit request, so the build reports
    the cell refused on every attempt and the frame never closes (see `Agent._note_frame_cells`).
    The name comes from the frame-plane read (`gs.frame_names`): `collect_block` matches on it, so
    a cell the scan has not named yet -- or one holding the frame's own block, a fluid, or bedrock
    -- waits rather than being dug blind.

    A cell that now holds the block the frame wants there is done, whatever marked it: the
    `[BUILD_STOPPED]` marking reads the plane as it stood *before* the build, so the cells that
    build had just filled were marked too, and run86 dug its own cobblestone corner (-89,68,-6)
    and two freshly laid obsidian (-88,68,-6), (-86,69,-6) back out of the frame.
    """
    wanted = frame_wanted_blocks(gs.base) if gs.base is not None else {}
    for cell in gs.refused_frame:
        name = gs.frame_names.get(cell)
        if name is None or name in AIR_NAMES or name in FRAME_UNDIGGABLE:
            continue
        if wanted.get(cell) == name:
            continue
        return cell, name
    return None


def _frame_started_at_base(gs: GameState) -> bool:
    """Whether our own frame already stands in the base's portal plane.

    `_portal_frame_started` answers against the plane the *base* pins; with no base yet that
    plane is only the bot's own feet, where the answer means nothing, so it stays false here.
    """
    return gs.base is not None and _portal_frame_started(gs)


def _frame_complete_at_base(gs: GameState) -> bool:
    """All ten frame obsidian stand in the base's plane, from the exact frame-plane read.

    `light_portal` used to prove "the build spent its bank" by `obsidian < PORTAL_OBSIDIAN`, but
    the pack can carry a fresh ten again (a restart, or the loop re-mining) and then a *finished*
    frame could never be lit (live, run28). The plane read is the frame's own state, so it needs
    no guess from the pack. The four cobblestone corners are not required: a nether portal needs
    obsidian only in the blocks that touch the opening.
    """
    if gs.base is None or not gs.frame_names:
        return False
    return all(gs.frame_names.get(cell) == "obsidian"
               for cell in frame_obsidian_cells(gs.base))


def _frame_short_at_base(gs: GameState) -> bool:
    """The frame the plane read shows is short: some of its obsidian stands, some does not.

    `Agent._frame_complete` is a one-way latch, and a frame can lose blocks after it fires. Live,
    run103: the build spent its ten obsidian at it114, the plane read latched the frame complete on
    the next tick, and by it191 two of those cells were air again (the pack had picked the two
    blocks back up at it140). With the latch still set `candidates` stepped over `build_portal`,
    `light_portal` refused on the same plane read that now said the frame was *not* finished, and
    the bot stood at its own gate with two obsidian in the pack for fifty ticks. A read that names
    the frame's own obsidian is the frame talking about itself, so it is allowed to undo the latch;
    a blank read (the base chunks unloaded, run97) names nothing and leaves it standing.
    """
    if gs.base is None or not gs.frame_names:
        return False
    cells = frame_obsidian_cells(gs.base)
    standing = sum(1 for cell in cells if gs.frame_names.get(cell) == "obsidian")
    return 0 < standing < len(cells)


def _portal_ignite_cell(gs: GameState) -> tuple[int, int, int] | None:
    """A frame obsidian cell to strike with the flint and steel.

    Prefers the nearest obsidian the scan reports; when the four nearest hits are a deposit or an
    older frame and ours is not among them, the exact frame-plane read names one of our cells.
    """
    hit = gs.resource_nearest("obsidian")
    if hit is not None:
        return (hit.x, hit.y, hit.z)
    if gs.base is not None:
        for cell in frame_obsidian_cells(gs.base):
            if gs.frame_names.get(cell) == "obsidian":
                return cell
    return None


def _frame_needs_obsidian(gs: GameState, inv: dict) -> bool:
    """Whether the obsidian chain still wants to bank blocks for a frame that is not standing.

    The chain used to close the moment `flint_and_steel` existed (`not has_fs`), on the
    assumption that the igniter is only ever crafted *after* the ten blocks are banked. A
    restart breaks that: an igniter carried back into a world whose frame is short -- or whose
    base was re-anchored away from the obsidian already placed -- left `mine_obsidian` switched
    off for the rest of the run, with no way to raise the frame. The frame plane itself tells
    "our frame is up" apart from "we still owe obsidian" where the igniter cannot, so the chain
    gates on that instead: mine while no frame stands at the base and the pack is short.

    `gs.frame_complete` is read ahead of the plane: the plane scan is blank once the base chunks
    unload, so a bot that finished the frame and roamed off read "no frame" and mined a second ten
    for a built gate (live run97 it142-145, 290 m out). The latch is set from the plane the moment
    it showed the frame and is location-independent, so it keeps the chain shut wherever the bot is.
    """
    return (not gs.frame_complete and not _frame_started_at_base(gs)
            and inv.get("obsidian", 0) < PORTAL_OBSIDIAN)


def _next_diamond_item(gs: GameState) -> tuple[str, int] | None:
    """The next diamond thing worth building, and what it costs in diamonds.

    The order is the tech tree: pickaxe (3, unlocks obsidian), then sword (2, needs the
    pickaxe to exist), then the armour set - most protection per diamond first. `None` means
    the diamond goals are met.
    """
    inv = gs.inventory
    if _pick_tier(inv, gs.tools) < DIAMOND_PICK_TIER:
        return "diamond_pickaxe", 3
    if not _has(inv, "diamond_sword"):
        return "diamond_sword", 2
    for slot, cost in DIAMOND_ARMOR:
        if not _armor_present(gs, slot):
            return f"diamond_{slot}", cost
    return None


def _next_iron_armor_item(gs: GameState) -> tuple[str, int] | None:
    """The next iron armour piece to build and its ingot cost, most protection first.

    The mirror of `_next_diamond_item` for the pre-diamond kit, so the craft gate and the iron
    goal share one source.
    """
    for slot, cost in IRON_ARMOR:
        if not _armor_slot_covered(gs, slot):
            return f"iron_{slot}", cost
    return None


def nether_expedition_complete(gs: GameState) -> bool:
    """True when the blaze powder from the first Nether trip stands in the Overworld chest.

    The goal the user set for the first visit: nine blaze rods, eighteen blaze powder, parked in
    the chest at the portal for now. Reading the chest back would cost a `use_container`
    inspection on every tick of a long run, so the deposit is latched by the loop when the action
    reports success (`gs.powder_stowed`, `Agent._note_stow`) - the same shape as `frame_complete`.

    Powder still in the pack does not count: the point is that the haul is safe at home, and a bot
    that dies on the way back loses the inventory in one step.
    """
    return bool(gs.powder_stowed)


def nether_expedition_open(gs: GameState) -> bool:
    """Whether the Nether chain is still being worked, or has been reopened for a powder top-up.

    `nether_expedition_complete` is a one-way latch: once the powder stands in the chest the
    expedition is done for the phase-3 goal, and every gate that reads it - the departure gate,
    the `enter_portal` offer - shuts. But the stronghold phase can need the gate again: the
    banked powder lost, the eyes uncraftable (`_nether_topup_wanted`). Without this predicate a
    top-up could never relaunch - `candidates` reopened the chain, yet `enter_portal` refused on
    the completion latch, and the bot stood home at an empty chest with an open goal.

    The pearl waves are the second reopener (`_nether_pearl_wave_wanted`): once the rod haul is
    stowed the chain reopens for the pearl trips, the Nether side running the pearl ladder rather
    than the blaze hunt (see `planner`).
    """
    return (not nether_expedition_complete(gs)
            or _nether_topup_wanted(gs)
            or _nether_pearl_wave_wanted(gs))


def _diamond_craft_ready(gs: GameState, item: str) -> bool:
    """Carried materials for a diamond craft: always a table, sticks only for tool and sword."""
    if not _has(gs.inventory, "crafting_table"):
        return False
    if item.endswith("_pickaxe") or item.endswith("_sword"):
        inv = gs.inventory
        return _can_sticks(inv, _logs(inv), _planks(inv))
    return True  # armour is diamonds only


def tech_tree_complete(gs: GameState) -> bool:
    """True when the overworld goal the loop works toward is fully reached.

    The prerequisite gear (`_gear_ready`): a diamond pickaxe, a diamond sword, all four diamond
    armour pieces actually *worn* - not merely carried, since it is `equip_armor` that puts them
    on - and the shield in the off-hand. Then the overworld -> Nether gate itself: a bed stands
    *at the base* beside the frame to respawn at (a bed merely loaded is not enough: a leftover
    bed from an earlier run satisfies `_bed_placed` while `_bed_off_base` still reports it, so
    the run keeps working until `take_bed`/`place_bed` bring it home; the bed is laid through
    `sleep`, which sets the respawn point by day as well as by night), the Nether prep is finished
    (a bow with arrows, a gold helmet and a deep cobblestone stack are banked, `_all_nether_prep_steps`)
    and the deep food reserve is banked (`_food_reserve_met`, the `NETHER_FOOD_RESERVE`). The
    gate is lit with all of that in place but the bed still carried (`light_portal` requires
    `_expedition_ready`), so the only burning-portal moment before the bed is laid is one tick,
    immediately followed by `place_bed`; only then does this gate close. It must never be
    lit early: the bot works at the base beside the frame, and the pathfinder walks an open gate
    like any block, so a portal burning through the prep window walks the body through it (live,
    run22: lit at it208, the bot stood in its own portal and was in the Nether by it245, with no
    step asking for the crossing). A prep step whose material has no source simply drops out of
    the list - and so does the food reserve once no source is in the loaded area - so an
    unreachable item never leaves the run unable to end.

    Requiring the armour to be worn (not just present) is what lets the loop finish
    `equip_armor` first instead of stopping with the set still in the pack.
    """
    return (_portal_lit(gs) and _gear_ready(gs) and _bed_at_base(gs)
            and not _all_nether_prep_steps(gs) and _nether_bank_carried(gs)
            and _food_reserve_met(gs))


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _cell(pos: tuple[float, float, float]) -> tuple[int, int, int]:
    """The block cell a float position names, for keying loop-owned per-cell facts."""
    return (int(math.floor(pos[0])), int(math.floor(pos[1])), int(math.floor(pos[2])))


def _can_recover(gs: GameState) -> bool:
    """A fresh death close by: the dropped gear is still loaded and worth a walk back."""
    if gs.death_pos is None or gs.last_death_age_s is None or gs.pos is None:
        return False
    # A site already walked and found empty is written off (see `Agent._note_recover`): the drop
    # is gone, so the walk can never pay off and offering it again only stalls the loop.
    if _cell(gs.death_pos) in gs.recover_written_off:
        return False
    if gs.last_death_age_s > RECOVER_WINDOW_S:
        return False
    if gs.death_dim and gs.dimension and gs.death_dim != gs.dimension:
        return False
    # Never walk back for a drop in the Nether, whatever the distance or the age. A death down
    # there drops the whole kit on the spot where the bot fell, which in the one place it ever
    # dies is the worst place to walk back into: a fortress ledge, a lava sea, a ghast's reach.
    # The gear is not worth a second trip through the dimension that just killed the bot for it,
    # and the walk back is what the recovery *is* - there is no partial recovery, it is the same
    # dangerous ground with the bot carrying nothing. The cost of the rule is the whole kit, paid
    # in the Overworld where a death costs a walk; the cost of breaking it is a run that cannot
    # finish an expedition twice.
    if _in_nether(gs) or _in_nether_dimension(gs.death_dim):
        return False
    # The mob that dropped the bot is still standing over the gear in the dark.
    if _defenceless_at_night(gs):
        return False
    if _dist(gs.pos, gs.death_pos) > RECOVER_RANGE:
        return False
    # A death well below the bot has to be dug down to. By hand that is several times slower
    # than with a pickaxe, so the dropped stack can despawn mid-walk (live, run12: the bot
    # respawned on the surface, hand-dug toward a death ~60 blocks down, lost the race and then
    # looped on `get_wood`). Require a pickaxe first; until it is crafted the normal chain
    # (gather wood, craft one) runs, and recovery is offered the moment it exists again.
    if (gs.death_pos[1] < gs.pos[1] - RECOVER_DIG_DEPTH
            and _pick_tier(gs.inventory, gs.tools) < 1):
        return False
    return True


def _diamond_ore_available(gs: GameState) -> bool:
    """Diamond ore to dig for: within the 64 m scan, or merely present in a loaded chunk.

    The listed resource hits are radius-limited, so the nearest diamond ore under a bot standing
    on the surface (y<16) never appears while the bot is over it. `resource_found` counts every
    loaded chunk, and `collect_block` will tunnel to a named target, so a deep deposit still
    justifies offering the dig - otherwise the loop can never leave the surface toward the
    diamonds its next armour piece needs.
    """
    return (gs.resource_distance("diamond_ore") is not None
            or gs.resource_found.get("diamond_ore", 0) > 0)




# Re-export the whole module namespace (public and private) so a downstream
# `from .<module> import *` carries every constant and helper by its original name,
# and `mlaya.actions.__init__` can hand the full surface back out unchanged.
__all__ = [_name for _name in list(globals()) if not _name.startswith("__")]
