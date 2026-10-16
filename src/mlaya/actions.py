from __future__ import annotations

import math
from dataclasses import dataclass

from .perception import AIR_NAMES, INVENTORY_NEAR_FULL, GameState
from .targets import (EXPLORE_FRONTIER_CHUNKS, EXPLORE_MIN_M, base_target,
                      frontier_heading, nearest_ground)

# Labels Laya may choose among; each maps to a real mine-ai MCP tool.
CRITERIA: dict[str, str] = {
    "escape_water": "swim clear of the water and climb back onto dry ground; the body has no footing, so this owns the tick once the bot has spent a decision off its feet",
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
    "equip_armor": "wear the armour pieces the bot crafted (iron first, then diamond), so their protection is active",
    "craft_shield": "craft a shield from six planks and one iron ingot; it blocks attacks while held in the off-hand",
    "equip_shield": "put the crafted shield into the off-hand slot, where it actually blocks attacks",
    "craft_iron_sword": "craft an iron sword from two iron ingots and a stick; the pre-diamond weapon",
    "craft_iron_chestplate": "craft an iron chestplate from eight iron ingots; the most protection per ingot",
    "craft_iron_leggings": "craft iron leggings from seven iron ingots",
    "craft_iron_boots": "craft iron boots from four iron ingots",
    "craft_iron_helmet": "craft an iron helmet from five iron ingots",
    "craft_diamond_chestplate": "craft a diamond chestplate from eight diamonds; the most protection per diamond",
    "craft_diamond_leggings": "craft diamond leggings from seven diamonds",
    "craft_diamond_boots": "craft diamond boots from four diamonds",
    "craft_diamond_helmet": "craft a diamond helmet from five diamonds",
    "sleep": "sleep in a carried bed at night to skip to morning and set the respawn point",
    "craft_bed": "craft a bed from white wool (or string) and planks at a crafting table",
    "hunt_wool": "kill a nearby sheep to collect white wool for a bed",
    "hunt_string": "kill nearby spiders to collect string, which crafts into wool",
    "hunt_food": "hunt an animal within walking range (far only in daylight), or a lone zombie, to bank food; once the portal is lit, stock the deeper Nether reserve",
    "cook_food": "cook carried raw meat in the furnace at the base (or the carried furnace, set up just for the batch) so it feeds more hunger; cooked meat lasts far longer than bread, and before the Nether trip any carried fuel will do",
    # Nether prep, stocked in the daytime lull after the portal is lit (see `_nether_prep_steps`).
    "craft_bow": "craft a bow from string and sticks; the server's combat layer fires it for ranged defence against ghasts and blazes",
    "craft_arrows": "craft arrows from flint, sticks and feathers so the carried bow has ammunition",
    "hunt_feather": "kill a nearby chicken to collect feathers for the Nether arrows",
    "mine_gold": "mine gold ore for the helmet that keeps piglins neutral in the Nether",
    "smelt_gold": "smelt raw gold into gold ingots in the furnace for the gold helmet",
    "craft_gold_helmet": "craft the gold helmet that keeps piglins from attacking; carry it until the Nether trip",
    "replace_tool": "craft a fresh pickaxe to replace the one that is nearly worn out",
    "collect_drops": "pick up items a mob dropped nearby when the survival reflex killed it",
    "recover_drops": "walk back to the last death site and recover the gear dropped there",
    "explore": "walk to new terrain to find missing resources; the last fallback",
    "wait": "hold position at the home base until night falls; the only work left is the bed, which cannot be placed in daylight",
    "make_room": "dig the block a placement was refused at, or a small pocket, so a carried workstation or bed fits",
    # Home base: one surface anchor the bed, the portal and the chests all stand at.
    "surface_up": "climb back out to open surface ground; the home base, the bed and the portal all belong under open sky, never on a cave floor",
    "establish_base": "walk up to open surface ground and set up the home base there",
    "go_home": "walk back to the home base to place the bed, the portal or store surplus there",
    "craft_chest": "craft a chest from eight planks to store surplus blocks at the base",
    "place_chest": "place the crafted chest at the home base",
    "place_furnace": "place the furnace the bot carries at the home base, so cooking and smelting reuse it instead of setting up a temporary one every time",
    "stow_surplus": "deposit the surplus blocks of a full pack into the chest at the base",
    # Overworld -> Nether gate (the portal frame and the bed beside it).
    "craft_bucket": "craft a bucket from three iron ingots; a water bucket is how obsidian is made",
    "fill_bucket": "fill the carried bucket with water at a source, ready to pour onto lava",
    "mine_obsidian": "carry a water bucket and mine obsidian: the dig pours the water onto lava and mines what forms",
    "get_flint": "dig gravel for flint, the spark the flint and steel needs",
    "craft_flint_and_steel": "craft flint and steel from flint and an iron ingot to light the portal",
    "clear_portal": "dig stray blocks out of the portal opening so the frame can be built and lit",
    "build_portal": "build the nether portal frame from the ten banked obsidian and four cobblestone corners",
    "light_portal": "strike the built portal frame with the carried flint and steel to light it",
    "place_bed": "place the carried bed beside the lit portal so the respawn point sits at the gate",
    "take_bed": "take the standing respawn bed back up when it lies away from the base, so it can be laid beside the portal",
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
    "gold_ore": IRON_PICK_TIER,      # stone breaks it; iron or better yields raw gold
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

# Pre-diamond combat kit. The loop used to dive to y<16 for diamonds with only an iron pickaxe;
# live (run12) that naked dive met a zombie and a creeper and died, and the recovery walk could
# not outrun the dropped stack's despawn. Before the first dive the bot now carries an iron
# sword and wears a full iron armour set, with the shield in the off-hand - iron is the only
# pre-diamond sword/armour the tree has. The pieces are ordered most protection per ingot first,
# exactly like the diamond set, so a partial set is already as useful as it can be.
IRON_SWORD_INGOTS = 2
IRON_ARMOR = (("chestplate", 8), ("leggings", 7), ("boots", 4), ("helmet", 5))
IRON_ARMOR_COST = dict(IRON_ARMOR)
# One `mine_iron` is a single `collect_block`, so the goal of a whole iron kit must not become one
# unbounded dig: cap the per-action count and let the loop re-offer the step for the rest.
IRON_MINE_BATCH = 8

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

# Decisions a bot may spend off its feet in water before `escape_water` owns the tick. The rescue
# exists because work that needs the feet planted cannot run from a floating stance, not because a
# floating body can run nothing at all: the server navigates a swim now, so a goal that crosses
# water must get to finish (live, run15: the goal walked east into a flooded flat and the rescue
# walked it straight back -- 155 rescue walks against 30 diamond digs, 157 cancellations, no
# diamonds).
WATER_RESCUE_AFTER = 2

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
# How far the bot's feet may sit from the surface cell it last learned for *its own* column and
# still count as "on the surface" for the base. The base belongs under open sky, and a mined
# tunnel's `air` reads the same as sky, so the depth below the probed surface is the only reliable
# test. A one-block scoop or slab at the column's own surface still opens to the sky, so a small
# margin stays; anything more is a cave floor, and a cell learned for a *different* column never
# counts at all (`_at_surface`).
BASE_SURFACE_TOL = 2

# Passive food animals and the raw drop `collect_mob_drop` should ask for.
FOOD_MOBS = {"cow": "beef", "pig": "porkchop", "chicken": "chicken",
             "sheep": "mutton", "rabbit": "rabbit"}
# Raw -> cooked; cooking needs a carried furnace and fuel.
COOKED = {"beef": "cooked_beef", "porkchop": "cooked_porkchop",
          "chicken": "cooked_chicken", "mutton": "cooked_mutton",
          "rabbit": "cooked_rabbit", "cod": "cooked_cod", "salmon": "cooked_salmon"}
# Saturation each edible restores (Minecraft's hidden hunger buffer): cooked meat keeps the bot
# fed far longer than bread or raw meat, and rotten flesh barely counts. `_first_edible` eats the
# richest carried food first, so a full pack is spent on steak before bread. Golden carrots
# (14.4, the best food in the game) would top the list, but they need carrots and gold, which
# this bot neither farms nor mines -- cooked meat is the richest food within its reach.
FOOD_SATURATION = {
    "cooked_beef": 12.8, "cooked_porkchop": 12.8, "cooked_mutton": 9.6, "cooked_salmon": 9.6,
    "cooked_chicken": 7.2, "mushroom_stew": 7.2,
    "bread": 6.0, "baked_potato": 6.0,
    "cooked_rabbit": 5.0, "cooked_cod": 5.0,
    "carrot": 3.6, "apple": 2.4,
    "beef": 1.8, "porkchop": 1.8, "rabbit": 1.8,
    "chicken": 1.2, "mutton": 1.2, "beetroot": 1.2, "melon_slice": 1.2,
    "rotten_flesh": 0.8,
    "potato": 0.6, "sweet_berries": 0.6, "dried_kelp": 0.6,
    "cod": 0.4, "salmon": 0.4, "cookie": 0.4,
}
# Minimum edible items to keep banked. 16 is the floor held through the tech tree; the Nether
# wants far more -- hunger drains fast while sprinting and fighting there and a death drops the
# whole pack -- so the target deepens in the daytime lull after the portal is lit (see
# `_food_goal`). The usual advice is 64-128 units (one to two stacks) of high-saturation food;
# a stack of cooked meat is a single inventory slot, so 64 is cheap to carry.
FOOD_RESERVE = 16
NETHER_FOOD_RESERVE = 64

# Hunger floors for eating. `EAT_HUNGER` is the survival floor the reflex has always used; the
# bot eats below it to keep hunger off empty. `REGEN_HUNGER` is Minecraft's natural-regeneration
# threshold: health only regenerates while food is at least 18, so a wounded bot whose food sits
# in the 14-17 gap eats nothing and heals nothing until it is hit again. Hunger is 0-20.
HEALTH_MAX = 20.0
EAT_HUNGER = 14.0
REGEN_HUNGER = 18.0

# --- Nether prep (the daytime lull after the portal) ---
# Beyond the deep food reserve, the expedition wants a ranged answer, a piglin pass and building
# material. All of it is stocked in the same lull, so the run reaches the gate fully prepared:
#   * a bow with arrows -- the server's combat layer fires the bow itself (`survival.BASE_CHANGES`
#     `combat.bow`), so the bot only has to *carry* one and keep arrows in the pack; string comes
#     from spiders, feathers from chickens, flint from gravel, sticks from wood.
#   * a gold helmet -- one worn piece of gold armour is all it takes to keep piglins neutral; it
#     is carried until the Nether trip (worn it would displace the diamond helmet).
#   * a deep cobblestone stack for a ghast-proof shelter around the arrival portal and for bridging.
# Every step is offered only while it can actually progress, so a material with no source (no
# spider, no chicken, no gold ore loaded) drops that step out and never blocks the run.
BOW_ITEM = "bow"
ARROW_ITEM = "arrow"
BOW_STRING = 3
NETHER_ARROWS = 16
FEATHER_ITEM = "feather"
GOLD_HELMET_ITEM = "golden_helmet"
GOLD_HELMET_INGOTS = 5
GOLD_ORE = "gold_ore"
RAW_GOLD = "raw_gold"
GOLD_INGOT = "gold_ingot"
NETHER_COBBLE = 64
# Food hunting is the one gather that walks to a moving target, so it is bounded by
# daylight: a long walk to a distant animal is cheap in daylight, but after dark the bot
# only tops up from animals already at hand. The old unguarded hunt roamed ~180 m at
# night and was killed twice, losing a fresh iron pickaxe. Widening the day reach to 128 to
# make a stricter food gate reachable cost far more than it bought: live (run19b) the hunt
# walked the bot 150-200 m from the base, where the base-range latches re-open the portal
# chain and the bot stranded in water for 76 ticks, so the reach is back to the bound the
# death above produced.
FOOD_HUNT_RANGE_DAY = 64.0
FOOD_HUNT_RANGE_NIGHT = 16.0
# A zombie hunted for its rotten flesh has to be this close (shared with `_safe_zombie_hunt`,
# which also uses it as the source test for the completion gate).
ZOMBIE_HUNT_RANGE = 24.0
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
# A death this far below the bot has to be dug down to. By hand that is several times slower than
# with a pickaxe, so the dropped stack can despawn while the bot digs toward it (live, run12: the
# bot respawned on the surface, hand-dug toward a death ~60 blocks down, lost the race and then
# looped on `get_wood`). Recovery is only offered for such a death once a pickaxe is carried;
# until then the normal chain (gather wood, craft one) runs first and recovery is offered again.
RECOVER_DIG_DEPTH = 4


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


def _furnace_fuel(inv: dict[str, int]) -> tuple[str, int] | None:
    """The best carried furnace fuel and how many items one unit of it smelts, or None.

    Coal and charcoal smelt eight items each and are preferred, so their use for iron smelting
    is not stolen; a plank or log smelts one and a half, so a single one cooks exactly one item.
    Requiring coal alone was too strict: a bot carrying a stack of raw meat and plenty of wood
    still could not cook (live, the from-zero run: `coal` was None, so `cook_food` never fired
    and the daytime lull after the portal fell through to `explore`).
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
    """
    inv = gs.inventory
    out: list[str] = []
    for slot, _ in IRON_ARMOR:
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
    if any(not _worn(gs, f"diamond_{slot}") for slot, _ in DIAMOND_ARMOR):
        return False
    return _offhand(gs, SHIELD_ITEM)


def _portal_lit(gs: GameState) -> bool:
    """A lit nether portal block is loaded nearby (the frame is on and the gate is open)."""
    return gs.resource_distance(PORTAL_BLOCK) is not None


def _bed_placed(gs: GameState) -> bool:
    """A bed block stands nearby - the bed the run places next to the portal."""
    return gs.resource_distance("white_bed") is not None


def _bed_off_base(gs: GameState) -> bool:
    """A bed stands nearby, but not at the base.

    The base bed belongs beside the lit portal, so a bed lying anywhere else is a leftover - the
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
    """A bed stands at the base - the gate bed `place_bed` lays beside the lit portal.

    The one clause the completion gate, `sleep` and `place_bed` all key off: a bed merely
    *loaded* is not it (`_bed_placed`), because the leftover in the base's own shaft would send
    the server to that bed and pin the respawn point off the gate again (live, run19b it25).
    """
    return _bed_placed(gs) and not _bed_off_base(gs)


def _nether_idle(gs: GameState) -> bool:
    """The daytime lull after the portal is lit: the kit is done, the gate is open and only the
    night bed is left. The extra Nether-prep work (deepening the food reserve, cooking on any
    fuel, holding at the base) is scoped to this window; before it the bot is busy with the
    chain and must not be pulled off it."""
    return _gear_ready(gs) and _portal_lit(gs)


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


def _food_reserve_met(gs: GameState) -> bool:
    """Whether the banked food reserve is met, or out of reach.

    The gate holds the run for `_food_goal` - the deep `NETHER_FOOD_RESERVE` in the post-portal
    lull - but lets the requirement drop out once no source is in reach, the same "unreachable
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


def _nether_prep_steps(gs: GameState) -> list[str]:
    """The Nether prep to offer in the daytime lull after the portal, in order, or [].

    Each step appears only while it can actually make progress, so an unreachable material (no
    spider, no chicken, no gold ore loaded) drops that step out and never blocks the run -- the
    same shape as `hunt_food` needing a `_food_target`. Scoped to `_nether_idle`, so it never
    competes with the chain.
    """
    steps = _all_nether_prep_steps(gs)
    # The gold dig is a daylight job: it tunnels down from the base, so it is not offered at
    # night (the run sleeps then). It stays in `_all_nether_prep_steps` so the completion gate
    # still counts it as pending -- otherwise nightfall would end the run before the dig.
    if gs.phase == "night":
        steps = [s for s in steps if s != "mine_gold"]
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
    sticks = _can_sticks(inv, _logs(inv), _planks(inv))
    out: list[str] = []
    # Bow: three string from spiders and three sticks, crafted at the table.
    if not _has(inv, BOW_ITEM):
        if table and inv.get("string", 0) >= BOW_STRING and sticks:
            out.append("craft_bow")
        elif _safe_spider_hunt(gs):
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
    # Gold helmet: mine the ore, smelt it, craft the piece (worn later, in the Nether).
    if not (_has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
        if inv.get(GOLD_INGOT, 0) >= GOLD_HELMET_INGOTS:
            if table:
                out.append("craft_gold_helmet")
        elif _has(inv, RAW_GOLD):
            if (_has(inv, "furnace") or _furnace_ready(gs)) and _furnace_fuel(inv) is not None:
                out.append("smelt_gold")
        elif _can_mine(gs, GOLD_ORE) and _gold_ore_available(gs):
            out.append("mine_gold")
    # Cobblestone: a deep stack for the ghast-proof shelter around the arrival portal.
    if inv.get("cobblestone", 0) < NETHER_COBBLE and _can_mine(gs, "stone"):
        out.append("mine_stone")
    return out


def _junk_carried(inv: dict[str, int]) -> list[str]:
    """Carried low-value blocks no recipe wants, in the fixed shedding order."""
    return [n for n in JUNK_ITEMS if inv.get(n, 0) > 0]


def _chest_ready(gs: GameState) -> bool:
    """A chest has been placed at the base (its coordinates are remembered)."""
    return gs.chest_pos is not None


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


def _furnace_ready(gs: GameState) -> bool:
    """A furnace stands at the base (its coordinates are remembered).

    When it does, `smelt_item` gets the furnace's coordinates and never places a temporary
    workstation, so the same furnace is reused instead of being set up and torn down on every
    `cook_food`/`smelt_gold` call (live: the bot visibly tore the furnace down between batches).
    """
    return gs.furnace_pos is not None


def _smelt_args(gs: GameState, item_name: str, count: int, fuel_name: str) -> dict:
    """The `smelt_item` arguments: the base furnace's coordinates, or a temporary workstation.

    With a furnace standing at the base the call names its exact cell and reuses it; otherwise
    it falls back to the carried furnace placed and recovered for this one call.
    """
    args: dict = {"item_name": item_name, "count": count, "fuel_item_name": fuel_name}
    if gs.furnace_pos is not None:
        fx, fy, fz = gs.furnace_pos
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


def _bed_available(gs: GameState) -> bool:
    """A bed is at hand for `sleep`: carried, or already standing beside the base.

    Counting a *placed* bed matters once the bed item has left the pack (a manual base build, or
    a restart): a carried-bed-only test would send the loop off to hunt wool for a second bed
    while one already stands at the base.
    """
    return _has_bed(gs.inventory) or _bed_placed(gs)


def _sleep_ready(gs: GameState) -> bool:
    """Whether `sleep` may run where the bot stands.

    Before the portal is lit the bot sleeps *away* from the base: the server lays the bed "beside
    the bot", and at the base that spot is the portal's own interior, which the later frame build
    cannot clear (`remove_wrong_blocks` answers "movement policy prohibits breaking this block"
    for a bed). Once the portal is lit the gate bed belongs at the base, and it has to be a bed
    *standing at the base*: counting any loaded bed sent the server to the leftover bed in the
    base's own shaft and pinned the respawn point off the gate again (live, run19b it25: the bot
    stood at the base and slept in the bed nine blocks below). A carried bed is `place_bed`'s job
    first; `take_bed` brings a leftover one home.
    """
    if _portal_lit(gs):
        return _at_base(gs) and _bed_at_base(gs)
    return not _at_base(gs) and _bed_available(gs)


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
    sx, sy, sz = gs.surface
    if (math.floor(gs.pos[0]), math.floor(gs.pos[2])) != (sx, sz):
        return False
    return abs(gs.pos[1] - sy) <= BASE_SURFACE_TOL


def _base_steps_ready(gs: GameState) -> bool:
    """Whether some base step is ready to run once the bot is back at the base.

    Used only to justify `go_home`: the bot walks back exactly when there is work that needs
    the base (a frame to raise, a bed to lay, a chest to place, surplus to store), never as an
    idle stroll.
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
        # Not gated on the night: `place_bed` only runs after dark, but a bed carried out of the
        # base's own shaft (taken up by `take_bed`) belongs home whatever the hour. Gated on the
        # night, the bot carried it around all day and never came back (live, run19b it39-40:
        # `take_bed` succeeded in daylight, `go_home` was not offered, and the bed was never laid).
        return True
    # In the lull, a carried furnace belongs at the base: walking back lets `place_furnace`
    # stand it there, after which cooking reuses it rather than tearing a temporary one down.
    if (_nether_idle(gs) and not _furnace_ready(gs) and not gs.furnace_blocked
            and _has(inv, "furnace") and _furnace_site(gs) is not None):
        return True
    if not gs.chest_blocked and _has(inv, CHEST_ITEM) and not _chest_ready(gs):
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
    for hit in gs.resources.get("obsidian", []):
        if hit.z == sz and sx - 1 <= hit.x <= sx + 2 and sy - 1 <= hit.y <= sy + 3:
            return True
    return False


def _frame_started_at_base(gs: GameState) -> bool:
    """Whether our own frame already stands in the base's portal plane.

    `_portal_frame_started` answers against the plane the *base* pins; with no base yet that
    plane is only the bot's own feet, where the answer means nothing, so it stays false here.
    """
    return gs.base is not None and _portal_frame_started(gs)


def _frame_needs_obsidian(gs: GameState, inv: dict) -> bool:
    """Whether the obsidian chain still wants to bank blocks for a frame that is not standing.

    The chain used to close the moment `flint_and_steel` existed (`not has_fs`), on the
    assumption that the igniter is only ever crafted *after* the ten blocks are banked. A
    restart breaks that: an igniter carried back into a world whose frame is short -- or whose
    base was re-anchored away from the obsidian already placed -- left `mine_obsidian` switched
    off for the rest of the run, with no way to raise the frame. The frame plane itself tells
    "our frame is up" apart from "we still owe obsidian" where the igniter cannot, so the chain
    gates on that instead: mine while no frame stands at the base and the pack is short.
    """
    return not _frame_started_at_base(gs) and inv.get("obsidian", 0) < PORTAL_OBSIDIAN


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
    lit and a bed stands *at the base* beside it to respawn at (a bed merely loaded is not
    enough: a leftover bed from an earlier run satisfies `_bed_placed` while `_bed_off_base`
    still reports it, so the run keeps working until `take_bed`/`place_bed` bring it home). The
    bed is only laid at night, so `sleep` actually ran and the respawn point is set. Then the
    Nether prep must be finished: the run keeps stocking the lull (`_all_nether_prep_steps`)
    until a bow with arrows, a gold helmet and a deep cobblestone stack are banked, *and* the
    food reserve is banked (`_food_reserve_met`, the deep `NETHER_FOOD_RESERVE`), so the trip
    starts prepared rather than stopping the moment the bed is laid. A prep step whose material
    has no source simply drops out of the list - and so does the food reserve once no source is
    in the loaded area - so an unreachable item never leaves the run unable to end.

    Requiring the armour to be worn (not just present) is what lets the loop finish
    `equip_armor` first instead of stopping with the set still in the pack.
    """
    return (_gear_ready(gs) and _portal_lit(gs) and _bed_at_base(gs)
            and not _all_nether_prep_steps(gs) and _food_reserve_met(gs))


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
    furnace = _has(inv, "furnace") or _furnace_ready(gs)
    coal = inv.get("coal", 0)
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
    if (_nether_idle(gs) and gs.base is not None and not _furnace_ready(gs)
            and not gs.furnace_blocked and _has(inv, "furnace")
            and _furnace_site(gs) is not None):
        out.append("place_furnace" if _at_base(gs) else "go_home")
    if (_has(inv, "furnace") or _furnace_ready(gs)) and _first_cookable(inv) and not eating:
        # Cook raw meat so it feeds far more hunger. Outside the Nether lull only coal/charcoal
        # may be spent on it (as before, so it never steals wood from the chain); during the lull
        # any carried fuel does, so a bot with raw meat and wood still cooks.
        fuel = _furnace_fuel(inv)
        if fuel is not None and (fuel[0] in ("coal", "charcoal") or _nether_idle(gs)):
            out.append("cook_food")
    # Food reserve: bank edible items. At full hunger the reflex cannot eat, so gathered drops
    # accumulate; when hunger drops it spends them. Never compete with eating: feed to 20 first,
    # bank after. The goal deepens to the Nether target in the lull after the portal is lit.
    if (_edible_count(inv) < _food_goal(gs) and not eating
            and _food_target(gs) is not None):
        out.append("hunt_food")

    wool = _wool(inv)
    string = inv.get("string", 0)
    bed = _bed_available(gs)
    # Night + a bed: sleeping skips to morning and pins the respawn point. `_sleep_ready` owns
    # *where* it may run - away from the base before the portal is lit (the server lays the bed
    # "beside the bot", and at the base that spot is the portal's own interior, which the later
    # frame build then cannot clear: live, the bed landed in the opening and `light_portal`
    # answered `[PORTAL_NO_FRAME]` on every tick), and at the base on a bed actually standing
    # there once it is lit. `place_bed` owns the base bed, `take_bed` brings a leftover one home.
    if gs.phase == "night" and not gs.sleeping and _overworld(gs) and _sleep_ready(gs):
        out.append("sleep")

    # A bed standing away from the base is a leftover the new base did not bring along. Take it
    # up in the lull and lay it again beside the portal: the respawn point belongs at the gate,
    # and a bed left in the base's own dig shaft is what `mine_gold` keeps stumbling on. Offered
    # with the prep, so the bed is cleared before the dig starts rather than mid-tunnel.
    if _nether_idle(gs) and _overworld(gs) and not _has_bed(inv) and _bed_off_base(gs):
        out.append("take_bed")

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
    if furnace and raw_iron >= 1 and coal >= 1 and iron < _iron_goal(gs):
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
            if (_combat_ready(gs) and _can_mine(gs, "diamond_ore")
                    and _diamond_ore_available(gs)):
                out.append("mine_diamond")
        elif _diamond_craft_ready(gs, item):
            out.append("craft_" + item)

    # --- Home base ---
    # One surface anchor owns the bed, the portal frame and the chest. It is established once
    # the diamond kit is complete, and after that the base steps are only eligible while the
    # bot is there; `go_home` walks it back when base work is waiting.
    if _gear_ready(gs):
        if gs.base is None:
            # The base, the bed and the portal all belong under open sky. A bot that mined its
            # way down stands on a cave floor whose mined `air` reads exactly like sky, so it
            # climbs back out first: `surface_up` probes the surface and walks up to it, and only
            # then is `establish_base` offered.
            out.append("establish_base" if _at_surface(gs) else "surface_up")
        elif not _at_base(gs) and _base_steps_ready(gs):
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
    if _gear_ready(gs) and not _portal_lit(gs):
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
        # source can actually be poured on, or while freshly formed obsidian lies within reach
        # (to mine it). The reach test on the mine side keeps the dig shallow - the natural
        # deposit under the same lava caves is tens of blocks down and never comes that close.
        if (_frame_needs_obsidian(gs, inv) and _can_mine(gs, "obsidian")
                and ((has_water and _lava_pour_target(gs) is not None)
                     or _obsidian_within_reach(gs))):
            out.append("mine_obsidian")
        if (not has_fs and has_flint and iron >= FLINT_STEEL_IRON and table
                and obsidian >= PORTAL_OBSIDIAN):
            out.append("craft_flint_and_steel")
        # `build_portal` runs while the bot carries ten obsidian, or to *resume* a frame it
        # already started: a partial build leaves exactly the cells it still needs in the pack,
        # and re-sending the same structure resumes it (the server takes the interior corner).
        can_build = (obsidian >= PORTAL_OBSIDIAN
                     or (obsidian >= 1 and _portal_frame_started(gs)))
        if (_at_base(gs) and has_fs and can_build and cobble >= PORTAL_CORNERS):
            # A block left standing in the opening (above the support layer) has to go first:
            # the frame would be built around it and `light_portal` would answer
            # `[PORTAL_NO_FRAME]` forever (live, Nether run: a bed the generic sleep laid
            # "beside the bot" landed in the opening). `gs.portal_blocked` is filled by the
            # loop's opening read; while it is non-empty the dig stands in front of the build.
            out.append("clear_portal" if gs.portal_blocked else "build_portal")
        # After the build the ten obsidian have left the pack (`obsidian < PORTAL_OBSIDIAN`)
        # while the flint and steel remains - that is the "frame is standing" signal. It is
        # confirmed against the frame plane (`_portal_frame_started`) so natural obsidian near
        # the base cannot be mistaken for our frame.
        if has_fs and obsidian < PORTAL_OBSIDIAN and _portal_frame_started(gs):
            out.append("light_portal")
    # The bed goes down at the base at night, because `sleep` is what actually sets the respawn
    # point: placing it in daylight would leave the pause point unset.
    if (_at_base(gs) and _portal_lit(gs) and _has_bed(inv) and not _bed_at_base(gs)
            and gs.phase == "night" and _overworld(gs)):
        out.append("place_bed")

    # --- Home base storage ---
    # A chest at the base is where a filling pack sheds surplus, well before the server's
    # `[INVENTORY_FULL]` refusal; `drop_junk` is only the fallback when no chest is there.
    if gs.base is not None and not _chest_ready(gs) and not gs.chest_blocked:
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

    # --- Nether prep (the daytime lull after the portal) ---
    # The bow, its arrows, the gold helmet and the deep cobblestone stack, stocked once the
    # portal is lit. `_nether_prep_steps` is the single source for both this list and the
    # completion gate, so the offered steps and `tech_tree_complete` never disagree.
    out.extend(_nether_prep_steps(gs))

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

    # The explicit last fallback when nothing productive is available. In the lull after the
    # portal is lit the only work left is the night bed, and wandering off just burned five
    # pathfinder timeouts and left the bot away from the base at dusk (live, from-zero run), so
    # hold at the base and wait it out rather than explore. Roaming to fill an open food reserve
    # was tried live (run19b) and reverted: the roam carried the bot 150-200 m out, where the
    # base-range latches re-open the portal chain and it stranded in water for 76 ticks.
    if not out:
        out.append("wait" if _at_base(gs) and _nether_idle(gs) else "explore")
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
        if (not (_has(inv, "furnace") or _furnace_ready(gs)) or inv.get("raw_iron", 0) < 1
                or inv.get("coal", 0) < 1):
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
        if _has(inv, FLINT_ITEM):
            return None
        # Flint feeds the flint and steel; afterwards the only sink is the Nether arrows, so the
        # dig reopens in the lull while the arrow reserve is short.
        if _has(inv, FLINT_STEEL_ITEM) and not _arrows_pending(gs):
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
                or _portal_lit(gs) or not can_build
                or inv.get("cobblestone", 0) < PORTAL_CORNERS):
            return None
        return Action(label, "build_structure",
                      {"portal_frame": site, "remove_wrong_blocks": True},
                      "build the obsidian portal frame beside the bot")
    if label == "light_portal":
        hit = gs.resource_nearest("obsidian")
        if (not _has(inv, FLINT_STEEL_ITEM) or _portal_lit(gs)
                or inv.get("obsidian", 0) >= PORTAL_OBSIDIAN or hit is None
                or not _portal_frame_started(gs)):
            return None
        return Action(label, "activate_portal", {"x": hit.x, "y": hit.y, "z": hit.z},
                      "light the built portal frame with the carried flint and steel")
    if label == "take_bed":
        if (_has_bed(inv) or not _nether_idle(gs) or not _overworld(gs)
                or not _bed_off_base(gs)):
            return None
        return Action(label, "collect_block",
                      {"block_name": "white_bed", "count": 1},
                      "take the standing respawn bed back up so it can be laid at the base")
    if label == "place_bed":
        if (not _has_bed(inv) or not _portal_lit(gs) or _bed_at_base(gs)
                or not _at_base(gs) or gs.phase != "night" or not _overworld(gs)):
            return None
        return Action(label, "sleep", {},
                      "place the carried bed beside the lit portal and set the respawn point")
    if label == "sleep":
        if (gs.phase != "night" or gs.sleeping or not _overworld(gs)
                or not _sleep_ready(gs)):
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
        goal = _string_goal(gs)
        if goal <= 0 or inv.get("string", 0) >= goal or not _safe_spider_hunt(gs):
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
    if label == "smelt_gold":
        fuel = _furnace_fuel(inv)
        if (not (_has(inv, "furnace") or _furnace_ready(gs)) or inv.get(RAW_GOLD, 0) < 1
                or fuel is None or _has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
            return None
        fuel_name, capacity = fuel
        # Outside the lull only coal/charcoal may be spent (never wood the chain still wants).
        if fuel_name not in ("coal", "charcoal") and not _nether_idle(gs):
            return None
        count = min(inv.get(RAW_GOLD, 0), 3, capacity,
                    max(0, GOLD_HELMET_INGOTS - inv.get(GOLD_INGOT, 0)))
        if count < 1:
            return None
        return Action(label, "smelt_item", _smelt_args(gs, RAW_GOLD, count, fuel_name),
                      f"smelt raw gold into ingots using {fuel_name}")
    if label == "craft_gold_helmet":
        if (inv.get(GOLD_INGOT, 0) < GOLD_HELMET_INGOTS or not _has(inv, "crafting_table")
                or _has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": GOLD_HELMET_ITEM}], "temporary_workstation": True},
                      "craft the gold helmet that keeps piglins neutral in the Nether")
    if label == "hunt_food":
        if _edible_count(inv) >= _food_goal(gs):
            return None
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
        fuel = _furnace_fuel(inv)
        if not raw or not (_has(inv, "furnace") or _furnace_ready(gs)) or fuel is None:
            return None
        # Outside the lull only coal/charcoal may be spent (never wood the chain still wants).
        if fuel[0] not in ("coal", "charcoal") and not _nether_idle(gs):
            return None
        fuel_name, capacity = fuel
        return Action(label, "smelt_item",
                      _smelt_args(gs, raw, min(inv.get(raw, 0), 3, capacity), fuel_name),
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
        if gs.blocked_cell is not None:
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
    if label == "surface_up":
        if gs.base is not None or not _gear_ready(gs) or gs.pos is None or _at_surface(gs):
            return None
        mx, mz = math.floor(gs.pos[0]), math.floor(gs.pos[2])
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
        if gs.base is None or _at_base(gs) or not _base_steps_ready(gs):
            return None
        bx, by, bz = gs.base
        return Action(label, "navigate",
                      {"x": bx, "y": by + 1, "z": bz, "dig": True, "scaffold": True},
                      "walk back to the home base to place the bed, the portal or a chest")
    if label == "craft_chest":
        if (gs.base is None or _has(inv, CHEST_ITEM) or _chest_ready(gs) or gs.chest_blocked
                or not _has(inv, "crafting_table") or _plank_units(inv) < CHEST_PLANKS):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": CHEST_ITEM}], "temporary_workstation": True},
                      "craft a chest from eight planks to store surplus at the base")
    if label == "place_chest":
        if gs.base is None or _chest_ready(gs) or gs.chest_blocked or not _has(inv, CHEST_ITEM):
            return None
        bx, by, bz = gs.base
        return Action(label, "place_block",
                      {"block_name": CHEST_ITEM, "x": bx, "y": by + 1, "z": bz + 1},
                      "place the chest at the home base")
    if label == "place_furnace":
        cell = _furnace_site(gs)
        if (_furnace_ready(gs) or gs.furnace_blocked or not _has(inv, "furnace")
                or cell is None):
            return None
        fx, fy, fz = cell
        return Action(label, "place_block",
                      {"block_name": "furnace", "x": fx, "y": fy, "z": fz},
                      "place the carried furnace at the home base so cooking reuses it")
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
    if label == "wait":
        # Hold at the base: navigate to the anchor, which is at most a step away, so the tick
        # settles at once and the bot does not wander off while daylight burns.
        if gs.base is None or not _nether_idle(gs):
            return None
        bx, by, bz = gs.base
        return Action(label, "navigate",
                      {"x": bx, "y": by + 1, "z": bz, "range": 1},
                      "hold at the home base until night falls and the bed can be placed")
    if label == "explore":
        if x is None or z is None:
            return None
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
                      {"heading": frontier_heading(gs), "chunks": EXPLORE_FRONTIER_CHUNKS},
                      "explore: no outward ground in reach, extend the frontier toward new terrain")
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
