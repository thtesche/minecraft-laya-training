from __future__ import annotations

import math
from dataclasses import dataclass

from .perception import INVENTORY_NEAR_FULL, GameState
from .targets import base_target

# Labels Laya may choose among; each maps to a real mine-ai MCP tool.
CRITERIA: dict[str, str] = {
    "escape_water": "swim clear of the water and climb back onto dry ground; with no footing under the bot nothing else can work",
    "eat_food": "eat a carried food item to restore hunger",
    "get_wood": "chop a tree and collect logs; the bot has no wood yet, so this comes first",
    "craft_pickaxe": "craft a wooden pickaxe from the logs the bot carries; this is the next step",
    "craft_table": "craft a crafting table from logs, required before any 3x3 recipe",
    "mine_stone": "mine stone for cobblestone using the pickaxe the bot carries",
    "craft_stone_pickaxe": "craft a stone pickaxe at a crafting table from cobblestone and sticks",
    "mine_iron": "mine iron ore for raw iron using the stone pickaxe the bot carries",
    "mine_coal": "mine coal ore to use as furnace fuel",
    "craft_furnace": "craft a furnace at a crafting table from eight cobblestone",
    "smelt_iron": "smelt raw iron into iron ingots in the furnace the bot carries",
    "craft_iron_pickaxe": "craft an iron pickaxe at a crafting table from three iron ingots and sticks",
    "mine_diamond": "mine diamond ore with an iron or better pickaxe to gather diamonds for diamond tools and armour",
    "craft_diamond_pickaxe": "craft a diamond pickaxe from three diamonds and sticks; the best pickaxe the bot can make",
    "craft_diamond_sword": "craft a diamond sword from two diamonds and a stick, once the diamond pickaxe exists",
    "equip_armor": "wear the diamond armour pieces the bot crafted, so their protection is active",
    "craft_shield": "craft a shield from six planks and one iron ingot; it blocks attacks while held in the off-hand",
    "equip_shield": "put the crafted shield into the off-hand slot, where it actually blocks attacks",
    "craft_diamond_chestplate": "craft a diamond chestplate from eight diamonds; the most protection per diamond",
    "craft_diamond_leggings": "craft diamond leggings from seven diamonds",
    "craft_diamond_boots": "craft diamond boots from four diamonds",
    "craft_diamond_helmet": "craft a diamond helmet from five diamonds",
    "sleep": "sleep in a carried bed at night to skip to morning and set the respawn point",
    "craft_bed": "craft a bed from white wool (or string) and planks at a crafting table",
    "hunt_wool": "kill a nearby sheep to collect white wool for a bed",
    "hunt_string": "kill nearby spiders to collect string, which crafts into wool",
    "hunt_food": "hunt an animal within walking range (far only in daylight), or a lone zombie, to bank at least 16 edible items",
    "cook_food": "cook carried raw meat in the furnace so it feeds more hunger",
    "replace_tool": "craft a fresh pickaxe to replace the one that is nearly worn out",
    "collect_drops": "pick up items a mob dropped nearby when the survival reflex killed it",
    "recover_drops": "walk back to the last death site and recover the gear dropped there",
    "explore": "walk to new terrain to find missing resources; the last fallback",
    "make_room": "dig a small pocket out of the surrounding blocks so a carried workstation or bed fits",
    # Home base: one surface anchor the bed, the portal and the chests all stand at.
    "establish_base": "walk up to open surface ground and set up the home base there",
    "go_home": "walk back to the home base to place the bed, the portal or store surplus there",
    "craft_chest": "craft a chest from eight planks to store surplus blocks at the base",
    "place_chest": "place the crafted chest at the home base",
    "stow_surplus": "deposit the surplus blocks of a full pack into the chest at the base",
    # Overworld -> Nether gate (the portal frame and the bed beside it).
    "craft_bucket": "craft a bucket from three iron ingots; a water bucket is how obsidian is made",
    "fill_bucket": "fill the carried bucket with water at a source, ready to pour onto lava",
    "mine_obsidian": "carry a water bucket and mine obsidian: the dig pours the water onto lava and mines what forms",
    "get_flint": "dig gravel for flint, the spark the flint and steel needs",
    "craft_flint_and_steel": "craft flint and steel from flint and an iron ingot to light the portal",
    "build_portal": "build the nether portal frame from the ten banked obsidian and four cobblestone corners",
    "light_portal": "strike the built portal frame with the carried flint and steel to light it",
    "place_bed": "place the carried bed beside the lit portal so the respawn point sits at the gate",
    "drop_junk": "drop low-value blocks from the full pack so a gathering action has room again",
}

EDIBLE = {
    "bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_chicken",
    "cooked_mutton", "cooked_rabbit", "cooked_cod", "cooked_salmon", "carrot",
    "potato", "baked_potato", "beetroot", "melon_slice", "sweet_berries",
    "rotten_flesh", "dried_kelp", "cookie", "mushroom_stew", "beef", "porkchop",
    "chicken", "mutton", "rabbit", "cod", "salmon",
}

# Pickaxe tier by registry prefix (tools[] reports e.g. {"class": "pickaxe", "tier": "wooden"}).
TIER = {"none": 0, "wooden": 1, "golden": 1, "stone": 2, "iron": 3, "diamond": 4, "netherite": 5}
STONE_PICK_TIER = 2
IRON_PICK_TIER = 3
DIAMOND_PICK_TIER = 4

# The game's rule: a block needs a minimum pickaxe tier or it breaks with no drop. There is no
# silk touch here, so iron ore drops raw iron - exactly what `smelt_iron` consumes (the mining
# bindings therefore ask for raw_iron, never for the ore block).
ORE_MIN_TIER = {
    "stone": 1,                      # cobblestone: any pickaxe
    "coal_ore": 1,                   # coal: any pickaxe
    "iron_ore": STONE_PICK_TIER,     # wood breaks it; stone already yields raw iron
    "diamond_ore": IRON_PICK_TIER,   # wood/stone/gold destroy it
    "obsidian": DIAMOND_PICK_TIER,   # only diamond or netherite drop it
}

# Diamond armour in the order the bot should build it: most protection per diamond first
# (chestplate 8 armour / 8 diamonds, leggings 6/7, boots 3/4, helmet 3/5), so a partial set
# is already as useful as it can be. The number is the recipe's diamond cost; the item is
# `diamond_<slot>`. Armour needs no sticks, only a crafting table.
DIAMOND_ARMOR = (("chestplate", 8), ("leggings", 7), ("boots", 4), ("helmet", 5))
DIAMOND_ARMOR_COST = dict(DIAMOND_ARMOR)

# Shield: 6 planks + 1 iron ingot (3x3, so it needs a carried crafting table). It only blocks
# while it sits in the off-hand, so the craft is followed by an `equip` to that slot. Gated
# behind the iron pickaxe so it never spends the first three ingots that pickaxe needs.
SHIELD_ITEM = "shield"
SHIELD_PLANKS = 6
SHIELD_IRON = 1

# Overworld -> Nether gate. A portal frame is ten obsidian with four cobblestone corners; the
# server's `build_structure portal_frame` builds exactly that from a carried stack, so the
# obsidian is never placed by hand. Obsidian is the one block the bot *makes* rather than finds:
# `mine_obsidian` walks to the nearest lava and pours the carried water onto it, then mines the
# obsidian that forms. The water itself needs a bucket (three iron) and the portal needs flint
# and steel (one flint and one ingot) to light.
PORTAL_OBSIDIAN = 10
PORTAL_CORNERS = 4
PORTAL_BLOCK = "nether_portal"
BUCKET_ITEM = "bucket"
WATER_BUCKET = "water_bucket"
FLINT_ITEM = "flint"
FLINT_STEEL_ITEM = "flint_and_steel"
BUCKET_IRON = 3
FLINT_STEEL_IRON = 1
# Gravel dug for one flint; the drop is a one-in-ten roll, so a batch and a repeat is the plan.
FLINT_GRAVEL = 16
# How close obsidian must be to be mined rather than assumed missing. Water poured on lava forms
# obsidian at the bot's feet; the natural deposit sits tens of blocks down in the same lava caves
# and never comes inside this reach, so the test keeps the dig shallow instead of tunnelling to it.
OBSIDIAN_REACH = 6.0
# How near the bot must be to a lava cell before the bucket can be emptied into it.
POUR_REACH = 4.5

# Low-value carried blocks no recipe here wants. When the pack fills up a gathering action is
# refused with `[INVENTORY_FULL]`, so the loop sheds these to make room. Deliberately excludes
# anything a recipe consumes: cobblestone (portal corners, furnace), planks/logs/sticks, wool,
# string, coal, iron, flint, gravel, obsidian and food.
JUNK_ITEMS = ("tuff", "calcite", "diorite", "andesite", "granite", "cobbled_deepslate",
              "deepslate", "dirt", "leather", "bone", "raw_copper", "glow_ink_sac")

# Home base. The run keeps one anchor on the surface; the bed, the portal frame and the chest
# all stand there, and the bot walks back to it to deploy them. Eight planks make a chest, which
# is how a full pack sheds surplus blocks without scattering them on the ground.
CHEST_ITEM = "chest"
CHEST_PLANKS = 8
# How close the bot must be to the base to lay blocks out there: feet within this many blocks
# of the anchor in x/z (one more vertically, so a step up or down still counts).
BASE_RADIUS = 4.0

# Passive food animals and the raw drop `collect_mob_drop` should ask for.
FOOD_MOBS = {"cow": "beef", "pig": "porkchop", "chicken": "chicken",
             "sheep": "mutton", "rabbit": "rabbit"}
# Raw -> cooked; cooking needs a carried furnace and fuel.
COOKED = {"beef": "cooked_beef", "porkchop": "cooked_porkchop",
          "chicken": "cooked_chicken", "mutton": "cooked_mutton",
          "rabbit": "cooked_rabbit", "cod": "cooked_cod", "salmon": "cooked_salmon"}
# Minimum edible items to keep banked. 16 is the Nether floor (24 the real target):
# hunger drains fast while sprinting and fighting, and death drops everything.
FOOD_RESERVE = 16
# Food hunting is the one gather that walks to a moving target, so it is bounded by
# daylight: a long walk to a distant animal is cheap in daylight, but after dark the bot
# only tops up from animals already at hand. The old unguarded hunt roamed ~180 m at
# night and was killed twice, losing a fresh iron pickaxe.
FOOD_HUNT_RANGE_DAY = 64.0
FOOD_HUNT_RANGE_NIGHT = 16.0
# Quarries that fight back, so `collect_mob_drop` must run in the hunt stage.
HOSTILE_HUNTS = {"spider", "zombie", "skeleton", "creeper", "enderman",
                 "witch", "blaze", "piglin", "zombified_piglin"}

# M2d reactions. An active tool at or below this fraction of its maximum is replaced
# before it breaks; a death this recent, this close, is worth walking back to.
LOW_DURABILITY = 0.15
# The reaction only runs between ticks, but one long `collect_block` (a tunnel to a distant
# vein) can spend far more than 15% inside a single action - live, that broke an iron pickaxe
# mid-mine and dropped the bot to stone, forcing a full iron rebuild. So the active pickaxe is
# additionally backed up once its *absolute* headroom falls to this many uses: the fresh
# same-tier spare is then already carried, and the server swaps it in the instant the active
# one breaks, so the tier (and with it `mine_diamond`) never goes away.
MINING_HEADROOM = 80
# Gear recovery walks back to the last death site. The dropped stack's despawn timer only ticks
# while its chunk is loaded, so a death that happened while the bot was far away (a respawn at the
# base, hundreds of blocks off) is still worth the walk: the window is the ambient 5-minute item
# despawn, the range one long trek out. Live, a 230 m death was written off as a total loss.
RECOVER_WINDOW_S = 300.0
RECOVER_RANGE = 320.0


@dataclass
class Action:
    label: str
    tool: str
    args: dict
    rationale: str


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
    return next((n for n in inv if n in EDIBLE), None)


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


def _nearest_food_mob(gs: GameState, max_range: float | None = None) -> str | None:
    best: str | None = None
    best_d: float | None = None
    for mob in FOOD_MOBS:
        d = gs.mob_nearest(mob)
        if d is None or (max_range is not None and d > max_range):
            continue
        if best_d is None or d < best_d:
            best, best_d = mob, d
    return best


def _food_hunt_range(gs: GameState) -> float:
    """How far a food hunt may walk: far in daylight, only at hand after dark."""
    return FOOD_HUNT_RANGE_NIGHT if gs.phase == "night" else FOOD_HUNT_RANGE_DAY


def _first_cookable(inv: dict[str, int]) -> str | None:
    return next((n for n in inv if n in COOKED and inv[n] > 0), None)


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


def _safe_zombie_hunt(gs: GameState) -> bool:
    """Zombies are the skill's overworld food bank, but only in ones and twos.

    Rotten flesh is the most reliable local food, so fall back to it only when the
    target is close, health is comfortable, and no other hostile species is in the
    neighbourhood — a pack is how a hunt turns into a death.
    """
    nearest = gs.mob_nearest("zombie")
    if nearest is None or nearest > 24 or (gs.health or 0) < 18:
        return False
    hostiles = [(n, d) for n, k, _, d in gs.mobs
                if k == "hostile" and d is not None and d <= 16]
    return len(hostiles) <= 2 and all(n == "zombie" for n, _ in hostiles)


def _food_target(gs: GameState) -> tuple[str, str] | None:
    """The safest available food source: a passive animal in walking reach, else a lone zombie."""
    mob = _nearest_food_mob(gs, _food_hunt_range(gs))
    if mob is not None:
        return mob, FOOD_MOBS[mob]
    if _safe_zombie_hunt(gs):
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


def _offhand(gs: GameState, item: str) -> bool:
    """Whether `item` currently sits in the off-hand slot (there it leaves the inventory)."""
    return gs.offhand == item


def _has_shield(gs: GameState) -> bool:
    """A shield counts as done whether it is in the off-hand or still waiting in the inventory."""
    return _offhand(gs, SHIELD_ITEM) or _has(gs.inventory, SHIELD_ITEM)


def _gear_ready(gs: GameState) -> bool:
    """The Nether-prerequisite gear: diamond pickaxe, diamond sword, the whole armour set
    actually worn, and the shield in the off-hand. The portal chain hangs off this, so a fresh
    run finishes the diamond tier before it ever starts banking obsidian."""
    inv = gs.inventory
    if _pick_tier(inv, gs.tools) < DIAMOND_PICK_TIER:
        return False
    if not (_has(inv, "diamond_sword") or str(gs.tools.get("sword", "")).lower() == "diamond"):
        return False
    if any(not _worn(gs, f"diamond_{slot}") for slot, _ in DIAMOND_ARMOR):
        return False
    return _offhand(gs, SHIELD_ITEM)


def _portal_lit(gs: GameState) -> bool:
    """A lit nether portal block is loaded nearby (the frame is on and the gate is open)."""
    return gs.resource_distance(PORTAL_BLOCK) is not None


def _bed_placed(gs: GameState) -> bool:
    """A bed block stands nearby - the bed the run places next to the portal."""
    return gs.resource_distance("white_bed") is not None


def _junk_carried(inv: dict[str, int]) -> list[str]:
    """Carried low-value blocks no recipe wants, in the fixed shedding order."""
    return [n for n in JUNK_ITEMS if inv.get(n, 0) > 0]


def _chest_ready(gs: GameState) -> bool:
    """A chest has been placed at the base (its coordinates are remembered)."""
    return gs.chest_pos is not None


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
        answers `[BUCKET_NO_FACE]`; its `below` must be solid.
      * The cell must be open above (`above == "air"`), or the water cannot be placed at all.
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
        if (h.pos in gs.pour_skip or h.above != "air"
                or h.below in (None, "lava", "air")):
            continue
        if any((h.x + dx, h.y, h.z + dz) in sources for dx, _, dz in _LATERAL):
            return h
    return None


def _pour_stance(gs: GameState, target: BlockHit) -> tuple[int, int, int] | None:
    """The cell to stand in to pour into `target`: *on top of* a lateral neighbour.

    Standing at the target's own height is what the live lake pour failed on. The lateral cell
    beside a lava pool is solid rock, and the server refuses to dig that last block next to the
    liquid ("breaking this block would open the route into liquid"), so the pathfinder stopped a
    cell short and diagonally -- where the reference block's face points past the bot and every
    pour answers `[BUCKET_NO_FACE]`. Standing on top of the neighbour is reachable, and that is
    the stance the live lake pour formed obsidian from. Lava neighbours are skipped: the bot may
    not stand there, and there is nothing to stand on either.
    """
    lava = {h.pos for h in (gs.resources.get("lava") or [])}
    options = [(target.x + dx, target.y + 1, target.z + dz)
               for dx, _, dz in _LATERAL
               if (target.x + dx, target.y, target.z + dz) not in lava]
    if not options:
        return None
    if gs.pos is None:
        return options[0]
    return min(options, key=lambda p: _dist(gs.pos, p))


def _bed_available(gs: GameState) -> bool:
    """A bed is at hand for `sleep`: carried, or already standing beside the base.

    Counting a *placed* bed matters once the bed item has left the pack (a manual base build, or
    a restart): a carried-bed-only test would send the loop off to hunt wool for a second bed
    while one already stands at the base.
    """
    return _has_bed(gs.inventory) or _bed_placed(gs)


def _at_base(gs: GameState) -> bool:
    """Whether the bot stands at the home base, close enough to lay blocks out there."""
    if gs.base is None or gs.pos is None:
        return False
    bx, by, bz = gs.base
    x, y, z = gs.pos
    return (abs(x - (bx + 0.5)) <= BASE_RADIUS
            and abs(z - (bz + 0.5)) <= BASE_RADIUS
            and abs(y - (by + 1)) <= BASE_RADIUS + 1)


def _base_steps_ready(gs: GameState) -> bool:
    """Whether some base step is ready to run once the bot is back at the base.

    Used only to justify `go_home`: the bot walks back exactly when there is work that needs
    the base (a frame to raise, a bed to lay, a chest to place, surplus to store), never as an
    idle stroll.
    """
    inv = gs.inventory
    if (_has(inv, FLINT_STEEL_ITEM) and inv.get("obsidian", 0) >= PORTAL_OBSIDIAN
            and inv.get("cobblestone", 0) >= PORTAL_CORNERS and not _portal_lit(gs)):
        return True
    if (_portal_lit(gs) and _has_bed(inv) and not _bed_placed(gs)
            and gs.phase == "night"):
        return True
    if _has(inv, CHEST_ITEM) and not _chest_ready(gs):
        return True
    if _chest_ready(gs) and gs.slots_used >= INVENTORY_NEAR_FULL and _junk_carried(inv):
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
    if tier >= IRON_PICK_TIER and not _has_shield(gs):
        goal = max(goal, SHIELD_IRON)
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
    on - and the shield in the off-hand. Then the overworld -> Nether gate itself: the portal is
    lit and a bed stands beside it to respawn at (the bed is only laid at night, so `sleep`
    actually ran and the respawn point is set). At that point the eligible actions collapse to
    `explore`/`sleep`, so a long unattended run can end itself instead of spending hours on
    nothing. Requiring the armour to be worn (not just present) is what lets the loop finish
    `equip_armor` first instead of stopping with the set still in the pack.
    """
    return _gear_ready(gs) and _portal_lit(gs) and _bed_placed(gs)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _can_recover(gs: GameState) -> bool:
    """A fresh death close by: the dropped gear is still loaded and worth a walk back."""
    if gs.death_pos is None or gs.last_death_age_s is None or gs.pos is None:
        return False
    if gs.last_death_age_s > RECOVER_WINDOW_S:
        return False
    if gs.death_dim and gs.dimension and gs.death_dim != gs.dimension:
        return False
    return _dist(gs.pos, gs.death_pos) <= RECOVER_RANGE


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
    furnace = _has(inv, "furnace")
    coal = inv.get("coal", 0)
    raw_iron = inv.get("raw_iron", 0)
    iron = inv.get("iron_ingot", 0)
    diamond = inv.get("diamond", 0)

    eating = gs.food is not None and gs.food < 14 and bool(_first_edible(inv))

    # --- no footing in water (rescue) ---
    # Live failure: the obsidian dig left the bot treading water off the ground. Off-ground
    # breaks the shared pathfinder -- `collect_block` then thrashed for minutes without moving a
    # block -- while `navigate` still works from a floating stance. Nothing else can run until
    # the bot has ground under it again, so the rescue owns the tick outright (no Laya choice).
    if gs.in_water and not gs.on_ground:
        return ["escape_water"]

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
    if _has(inv, "furnace") and inv.get("coal", 0) >= 1 and _first_cookable(inv):
        out.append("cook_food")
    # Food reserve: bank FOOD_RESERVE edible items. At full hunger the reflex cannot
    # eat, so gathered drops accumulate; when hunger drops it spends them. Never
    # compete with eating: feed to 20 first, bank after.
    if (_edible_count(inv) < FOOD_RESERVE and not eating
            and _food_target(gs) is not None):
        out.append("hunt_food")

    wool = _wool(inv)
    string = inv.get("string", 0)
    bed = _bed_available(gs)
    # Night + a bed (carried or already standing): sleeping skips to morning and pins the
    # respawn point.
    if gs.phase == "night" and not gs.sleeping and bed:
        out.append("sleep")

    # Bootstrap, then crafts/smelts (in tech-tree order), then gathering.
    need_wood = (tier == 0 or not table
                 or (not bed and (wool >= 3 or string >= 12))
                 or (_pickaxe_due_for_spare(gs) and not _can_sticks(inv, logs, planks)))
    if logs == 0 and planks == 0 and need_wood:
        out.append("get_wood")
    if (logs >= 1 or planks >= 5) and tier == 0:
        out.append("craft_pickaxe")
    if (logs >= 1 or planks >= 4) and not table:
        out.append("craft_table")
    if tier >= 1 and tier < STONE_PICK_TIER and table and cobble >= 3 and _can_sticks(inv, logs, planks):
        out.append("craft_stone_pickaxe")
    if tier >= 1 and table and not furnace and cobble >= 8:
        out.append("craft_furnace")
    if furnace and raw_iron >= 1 and coal >= 1 and iron < 3:
        out.append("smelt_iron")
    if table and iron >= 3 and tier < IRON_PICK_TIER and _can_sticks(inv, logs, planks):
        out.append("craft_iron_pickaxe")
    # Bed chain: a bed lets `sleep` skip the night and set the respawn point.
    if not bed:
        if table and (wool >= 3 or string >= 12) and (logs >= 1 or planks >= 3):
            out.append("craft_bed")
        elif wool < 3 and string < 12:
            if gs.mob_nearest("sheep") is not None:
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
    if raw_iron >= 1 and coal < 1 and _can_mine(gs, "coal_ore") and gs.resource_distance("coal_ore") is not None:
        out.append("mine_coal")
    # Shield (6 planks + 1 iron ingot, 3x3 so it needs the table): it only blocks while it sits
    # in the off-hand, so the craft is immediately followed by `equip_shield`. Gated behind the
    # iron pickaxe so it never spends the first ingots the pickaxe needs.
    if (not _has_shield(gs) and table and tier >= IRON_PICK_TIER
            and _plank_units(inv) >= SHIELD_PLANKS and iron >= SHIELD_IRON):
        out.append("craft_shield")
    if _has(inv, SHIELD_ITEM) and not _offhand(gs, SHIELD_ITEM):
        out.append("equip_shield")
    # A crafted armour piece sits in the inventory until it is worn; `equip` slots armour even
    # without a destination, so one batch puts the whole set on. Wear it before crafting more.
    if any(_has(inv, f"diamond_{slot}") and not _worn(gs, f"diamond_{slot}")
           for slot, _ in DIAMOND_ARMOR):
        out.append("equip_armor")
    # Diamond milestones, one at a time: pickaxe (three diamonds), then the sword (two, and only
    # once the pickaxe exists - bounding that away deadlocked the sword), then the armour set
    # (24, most protection per diamond first). `_next_diamond_item` names the next piece and its
    # cost, so the dig count and the craft gate share one source.
    # `_can_mine` refuses a dig on an almost-broken pickaxe: breaking mid-action has the server
    # finish the ore with the swapped-in lower tier, destroying it without a drop.
    nxt = _next_diamond_item(gs)
    if nxt is not None:
        item, cost = nxt
        if diamond < cost:
            if _can_mine(gs, "diamond_ore") and _diamond_ore_available(gs):
                out.append("mine_diamond")
        elif _diamond_craft_ready(gs, item):
            out.append("craft_" + item)

    # --- Home base ---
    # One surface anchor owns the bed, the portal frame and the chest. It is established once
    # the diamond kit is complete, and after that the base steps are only eligible while the
    # bot is there; `go_home` walks it back when base work is waiting.
    if _gear_ready(gs):
        if gs.base is None:
            out.append("establish_base")
        elif not _at_base(gs) and _base_steps_ready(gs):
            out.append("go_home")

    # --- Overworld -> Nether gate ---
    # Gated on the whole diamond kit (`_gear_ready`) so a fresh run finishes the tier first.
    # The order banks the obsidian *before* the flint and steel is crafted: that craft is the
    # latch that stops `mine_obsidian` re-mining after `build_portal` has spent the ten blocks.
    # Without it there is no way to tell our fresh frame from the natural obsidian that lines
    # the very lava the bot walks to, so the carried count alone cannot decide it.
    # `build_portal` and `place_bed` run only at the base (`_at_base`), so the gate stands at
    # one fixed spot the respawn point can be pinned to.
    if _gear_ready(gs) and not _portal_lit(gs):
        has_bucket = _has(inv, BUCKET_ITEM)
        has_water = _has(inv, WATER_BUCKET)
        has_flint = _has(inv, FLINT_ITEM)
        has_fs = _has(inv, FLINT_STEEL_ITEM)
        obsidian = inv.get("obsidian", 0)
        if not has_bucket and not has_water and table and iron >= BUCKET_IRON:
            out.append("craft_bucket")
        elif has_bucket and not has_water:
            out.append("fill_bucket")
        if not has_flint and not has_fs and gs.resource_distance("gravel") is not None:
            out.append("get_flint")
        # Obsidian is *made*, not dug: the step is offered while the bucket is full and a lava
        # source can actually be poured on, or while freshly formed obsidian lies within reach
        # (to mine it). The reach test on the mine side keeps the dig shallow - the natural
        # deposit under the same lava caves is tens of blocks down and never comes that close.
        if (obsidian < PORTAL_OBSIDIAN and not has_fs and _can_mine(gs, "obsidian")
                and ((has_water and _lava_pour_target(gs) is not None)
                     or _obsidian_within_reach(gs))):
            out.append("mine_obsidian")
        if (not has_fs and has_flint and iron >= FLINT_STEEL_IRON and table
                and obsidian >= PORTAL_OBSIDIAN):
            out.append("craft_flint_and_steel")
        if (_at_base(gs) and has_fs and obsidian >= PORTAL_OBSIDIAN
                and cobble >= PORTAL_CORNERS):
            out.append("build_portal")
        # After the build the ten obsidian have left the pack (`obsidian < PORTAL_OBSIDIAN`)
        # while the flint and steel remains - that is the "frame is standing" signal, and the
        # frame's own blocks are the nearest obsidian, so the light step can name one.
        if (has_fs and obsidian < PORTAL_OBSIDIAN
                and gs.resource_nearest("obsidian") is not None):
            out.append("light_portal")
    # The bed goes down at the base at night, because `sleep` is what actually sets the respawn
    # point: placing it in daylight would leave the pause point unset.
    if (_at_base(gs) and _portal_lit(gs) and _has_bed(inv) and not _bed_placed(gs)
            and gs.phase == "night"):
        out.append("place_bed")

    # --- Home base storage ---
    # A chest at the base is where a filling pack sheds surplus, well before the server's
    # `[INVENTORY_FULL]` refusal; `drop_junk` is only the fallback when no chest is there.
    if gs.base is not None and not _chest_ready(gs):
        if _has(inv, CHEST_ITEM) and _at_base(gs):
            out.append("place_chest")
        elif table and not _has(inv, CHEST_ITEM) and _plank_units(inv) >= CHEST_PLANKS:
            out.append("craft_chest")
    if gs.slots_used >= INVENTORY_NEAR_FULL and _junk_carried(inv):
        if gs.base is not None and _chest_ready(gs) and _at_base(gs):
            out.append("stow_surplus")
        elif gs.base is None or not _chest_ready(gs):
            # No chest to hold it: shed the surplus on the ground rather than stall the gather.
            out.append("drop_junk")

    if cobble < 12 and _can_mine(gs, "stone"):
        out.append("mine_stone")

    # explore is the explicit last fallback: only when nothing productive is available.
    if not out:
        out.append("explore")
    return out


def resolve(label: str | None, gs: GameState, target=None) -> Action | None:
    """Map a Laya choice to a concrete MCP tool call, or None if preconditions fail.

    `target` is the M4 parameter-layer pick (a `targets.Target`); it is only used by a
    label whose step needs a destination (`explore`), and ignored elsewhere.
    """
    if not label:
        return None
    inv = gs.inventory
    tier = _pick_tier(inv, gs.tools)
    x, y, z = gs.pos or (None, None, None)

    if label == "escape_water":
        # Swim clear of the water and climb back onto dry ground: the base when one is set,
        # otherwise the nearest standable cell. `navigate` is the one tool that still moves a
        # floating body (live: a `collect_block` loop left the bot treading water, and every
        # later dig thrashed in place while `navigate` walked it home).
        if gs.base is not None:
            bx, by, bz = gs.base
            return Action(label, "navigate",
                          {"x": bx, "y": by + 1, "z": bz, "dig": True, "scaffold": True},
                          "swim clear of the water and climb back up to the dry home base")
        target = base_target(gs)
        if target is None:
            return None
        tx, ty, tz = target.pos
        return Action(label, "navigate",
                      {"x": tx, "y": ty + 1, "z": tz, "dig": True, "scaffold": True},
                      "climb out of the water onto the nearest dry ground")
    if label == "get_wood":
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
        count = max(1, _iron_goal(gs) - inv.get("raw_iron", 0))
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
        nxt = _next_diamond_item(gs)
        if nxt is None:  # pickaxe, sword and the whole armour set are done
            return None
        count = max(1, nxt[1] - inv.get("diamond", 0))
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
    if label == "craft_furnace":
        if inv.get("cobblestone", 0) < 8 or not _has(inv, "crafting_table"):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "furnace"}], "temporary_workstation": True},
                      "craft a furnace to smelt iron")
    if label == "smelt_iron":
        if not _has(inv, "furnace") or inv.get("raw_iron", 0) < 1 or inv.get("coal", 0) < 1:
            return None
        count = min(inv.get("raw_iron", 0), 3)
        return Action(label, "smelt_item",
                      {"item_name": "raw_iron", "count": count, "fuel_item_name": "coal",
                       "temporary_workstation": True},
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
    if label == "equip_shield":
        if not _has(inv, SHIELD_ITEM) or _offhand(gs, SHIELD_ITEM):
            return None
        return Action(label, "equip",
                      {"items": [{"item_name": SHIELD_ITEM, "destination": "off-hand"}]},
                      "put the shield in the off-hand so it blocks attacks")
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
        items = [{"item_name": f"diamond_{slot}"} for slot, _ in DIAMOND_ARMOR
                 if _has(inv, f"diamond_{slot}") and not _worn(gs, f"diamond_{slot}")]
        if not items:
            return None
        return Action(label, "equip", {"items": items},
                      "wear the crafted diamond armour so its protection is active")
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
        if not _has(inv, BUCKET_ITEM) or _has(inv, WATER_BUCKET):
            return None
        return Action(label, "use_bucket", {"action": "fill", "liquid": "water"},
                      "fill the carried bucket with water to pour onto lava")
    if label == "mine_obsidian":
        if (not _can_mine(gs, "obsidian") or _has(inv, FLINT_STEEL_ITEM)
                or inv.get("obsidian", 0) >= PORTAL_OBSIDIAN):
            return None
        # Phase 1: obsidian already lies within reach (what the last pour formed) - mine it.
        if _obsidian_within_reach(gs):
            count = max(1, PORTAL_OBSIDIAN - inv.get("obsidian", 0))
            return Action(label, "collect_block", {"block_name": "obsidian", "count": count},
                          "mine the obsidian the poured water just formed at the bot's feet")
        # Phase 2: no obsidian in reach, so make some - pour the carried water into a lava
        # cell that lies beside a real lava source (see `_lava_pour_target`). Natural obsidian
        # deep under that lava is deliberately *not* a target: tunnelling to it is what left
        # the bot treading water in a flooded shaft.
        if not _has(inv, WATER_BUCKET):
            return None
        lava = _lava_pour_target(gs)
        if lava is None:
            return None
        if _within_reach(gs, lava, POUR_REACH):
            return Action(label, "use_bucket",
                          {"action": "pour", "liquid": "water",
                           "x": lava.x, "y": lava.y, "z": lava.z},
                          "pour the carried water onto the lava source to form obsidian")
        stance = _pour_stance(gs, lava)
        if stance is None:
            return None
        sx, sy, sz = stance
        return Action(label, "navigate",
                      {"x": sx, "y": sy, "z": sz, "range": 1,
                       "dig": True, "scaffold": False},
                      "walk to the lava source, stand on the block beside it and pour")
    if label == "get_flint":
        if _has(inv, FLINT_ITEM) or _has(inv, FLINT_STEEL_ITEM):
            return None
        return Action(label, "collect_block", {"block_name": "gravel", "count": FLINT_GRAVEL},
                      "dig gravel for flint to make flint and steel")
    if label == "craft_flint_and_steel":
        if (not _has(inv, FLINT_ITEM) or inv.get("iron_ingot", 0) < FLINT_STEEL_IRON
                or not _has(inv, "crafting_table") or _has(inv, FLINT_STEEL_ITEM)
                or inv.get("obsidian", 0) < PORTAL_OBSIDIAN):
            return None
        return Action(label, "craft_item", {"items": [{"item_name": FLINT_STEEL_ITEM}]},
                      "craft flint and steel from flint and an iron ingot")
    if label == "build_portal":
        site = _portal_site(gs)
        if (site is None or not _at_base(gs) or not _has(inv, FLINT_STEEL_ITEM)
                or _portal_lit(gs)
                or inv.get("obsidian", 0) < PORTAL_OBSIDIAN
                or inv.get("cobblestone", 0) < PORTAL_CORNERS):
            return None
        return Action(label, "build_structure",
                      {"portal_frame": site, "remove_wrong_blocks": True},
                      "build the obsidian portal frame beside the bot")
    if label == "light_portal":
        hit = gs.resource_nearest("obsidian")
        if (not _has(inv, FLINT_STEEL_ITEM) or _portal_lit(gs)
                or inv.get("obsidian", 0) >= PORTAL_OBSIDIAN or hit is None):
            return None
        return Action(label, "activate_portal", {"x": hit.x, "y": hit.y, "z": hit.z},
                      "light the built portal frame with the carried flint and steel")
    if label == "place_bed":
        if (not _has_bed(inv) or not _portal_lit(gs) or _bed_placed(gs)
                or not _at_base(gs) or gs.phase != "night"):
            return None
        return Action(label, "sleep", {},
                      "place the carried bed beside the lit portal and set the respawn point")
    if label == "sleep":
        if gs.phase != "night" or gs.sleeping or not _bed_available(gs):
            return None
        return Action(label, "sleep", {},
                      "sleep through the night and set the respawn point")
    if label == "craft_bed":
        if not _has(inv, "crafting_table") or (_wool(inv) < 3 and inv.get("string", 0) < 12) \
                or (_logs(inv) < 1 and _planks(inv) < 3):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "white_bed"}], "temporary_workstation": True},
                      "craft a bed from wool or string and planks")
    if label == "hunt_wool":
        if _wool(inv) >= 3:
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "sheep", "drop_name": "white_wool", "count": 3},
                      "collect white wool from a sheep for a bed")
    if label == "hunt_string":
        if inv.get("string", 0) >= 12 or not _safe_spider_hunt(gs):
            return None
        return Action(label, "collect_mob_drop",
                      {"mob_name": "spider", "drop_name": "string", "count": 12,
                       "allow_without_shield": True},
                      "collect string from a lone spider to craft wool")
    if label == "hunt_food":
        target = _food_target(gs)
        if not target:
            return None
        mob, drop = target
        count = 6 if mob in FOOD_MOBS else 10
        return Action(label, "collect_mob_drop",
                      {"mob_name": mob, "drop_name": drop, "count": count,
                       "allow_without_shield": True},
                      f"hunt a {mob} for {drop}")
    if label == "cook_food":
        raw = _first_cookable(inv)
        if not raw or not _has(inv, "furnace") or inv.get("coal", 0) < 1:
            return None
        return Action(label, "smelt_item",
                      {"item_name": raw, "count": min(inv.get(raw, 0), 3),
                       "fuel_item_name": "coal", "temporary_workstation": True},
                      f"cook {raw} into {COOKED[raw]}")
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
        # `build_structure` with `block_name: "air"` digs cells clear. One small pocket at
        # the bot's own level is enough: each cleared side cell has the tunnel floor under
        # it, so the placement search finds a supported, replaceable cell within reach.
        # Kept to four cells on purpose -- the tool cost grows with the cell list.
        mx, my, mz = (math.floor(v) for v in gs.pos)
        cells = [{"x": mx + dx, "y": my, "z": mz + dz, "block_name": "air"}
                 for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        return Action(label, "build_structure", {"blocks": cells},
                      "dig a small pocket around the bot for the carried workstation or bed")
    if label == "establish_base":
        if gs.base is not None or not _gear_ready(gs):
            return None
        target = base_target(gs)
        if target is None:
            return None
        tx, ty, tz = target.pos
        return Action(label, "navigate",
                      {"x": tx, "y": ty + 1, "z": tz, "dig": True, "scaffold": True},
                      f"walk up to the surface at {tx},{ty + 1},{tz} and set the base")
    if label == "go_home":
        if gs.base is None or _at_base(gs) or not _base_steps_ready(gs):
            return None
        bx, by, bz = gs.base
        return Action(label, "navigate",
                      {"x": bx, "y": by + 1, "z": bz, "dig": True, "scaffold": True},
                      "walk back to the home base to place the bed, the portal or a chest")
    if label == "craft_chest":
        if (gs.base is None or _has(inv, CHEST_ITEM) or _chest_ready(gs)
                or not _has(inv, "crafting_table") or _plank_units(inv) < CHEST_PLANKS):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": CHEST_ITEM}], "temporary_workstation": True},
                      "craft a chest from eight planks to store surplus at the base")
    if label == "place_chest":
        if gs.base is None or _chest_ready(gs) or not _has(inv, CHEST_ITEM):
            return None
        bx, by, bz = gs.base
        return Action(label, "place_block",
                      {"block_name": CHEST_ITEM, "x": bx, "y": by + 1, "z": bz + 1},
                      "place the chest at the home base")
    if label == "stow_surplus":
        items = [{"item_name": n, "count": inv[n]} for n in _junk_carried(inv)]
        if not _chest_ready(gs) or not items:
            return None
        cx, cy, cz = gs.chest_pos
        return Action(label, "use_container",
                      {"operation": "deposit", "x": cx, "y": cy, "z": cz, "items": items[:12]},
                      "deposit the surplus blocks into the chest at the base")
    if label == "drop_junk":
        items = [{"item_name": n} for n in _junk_carried(inv)]
        if not items:
            return None
        return Action(label, "drop_item",
                      {"items": items[:12], "in_a_hole": not gs.drop_hole_blocked},
                      "drop low-value blocks so the full pack has room again")
    if label == "explore":
        if target is not None:
            tx, ty, tz = target.pos
            return Action(label, "navigate",
                          {"x": tx, "y": ty, "z": tz, "dig": True, "scaffold": True},
                          f"explore toward {target.name} at {tx},{ty},{tz}")
        if x is None or z is None:
            return None
        args: dict = {"x": int(x) + 16, "z": int(z), "dig": True, "scaffold": True}
        if y is not None:
            args["y"] = int(y)
        return Action(label, "navigate", args, "explore toward new terrain")
    return None


def inventory_relief(gs: GameState, allow_stow: bool = True) -> "Action | None":
    """The step that frees pack space after an `[INVENTORY_FULL]` refusal.

    Prefers the chest at the base (deposit) when the bot is standing there; otherwise it drops
    the surplus into a disposal pit. Either way the pack gains room so the blocked gather can
    run again on the next tick. `allow_stow` is False once a deposit has already failed (a full
    chest), so the loop falls back to dropping instead of retrying a deposit that cannot work.
    """
    if allow_stow and _chest_ready(gs) and _at_base(gs):
        action = resolve("stow_surplus", gs)
        if action is not None:
            return action
    return resolve("drop_junk", gs)
