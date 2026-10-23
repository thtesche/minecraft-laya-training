from __future__ import annotations

import math
from dataclasses import dataclass

from .perception import (AIR_NAMES, EAT_HUNGER, INVENTORY_NEAR_FULL, INVENTORY_SLOTS,
                         PORTAL_BLOCK, REGEN_HUNGER, GameState, _in_nether_dimension,
                         frame_obsidian_cells, frame_wanted_blocks)
from .targets import (EXPLORE_FRONTIER_CHUNKS, EXPLORE_MIN_M, base_target,
                      frontier_heading, nearest_ground)

# Labels Laya may choose among; each maps to a real mine-ai MCP tool.
CRITERIA: dict[str, str] = {
    "escape_water": "swim clear of the water and climb back onto dry ground; the body has no footing, so this owns the tick once the bot has spent a decision off its feet",
    "leave_portal": "step out of a lit nether portal the body is standing in; standing in the gate teleports the bot within seconds, so this owns the tick outright",
    "eat_food": "eat a carried food item to restore hunger",
    "get_wood": "chop a tree and collect logs; the bot has no wood yet, so this comes first",
    "craft_pickaxe": "craft a wooden pickaxe from the logs the bot carries; this is the next step",
    "craft_table": "craft a crafting table from logs, required before any 3x3 recipe",
    "mine_stone": "mine stone for cobblestone using the pickaxe the bot carries",
    "craft_stone_pickaxe": "craft a stone pickaxe at a crafting table from cobblestone and sticks",
    "mine_iron": "mine iron ore for raw iron using the stone pickaxe the bot carries",
    "mine_coal": "mine coal ore to use as furnace fuel",
    "craft_smoker": "craft a smoker from the carried furnace and four logs, so food is cooked at double speed at the base; a second carried furnace stays behind as the workstation",
    "place_smoker": "stand the carried smoker beside the furnace at the base, so cooking reuses it",
    "craft_furnace": "craft a furnace at a crafting table from eight cobblestone",
    "smelt_iron": "smelt raw iron into iron ingots in the furnace the bot carries",
    "craft_iron_pickaxe": "craft an iron pickaxe at a crafting table from three iron ingots and sticks",
    "mine_diamond": "mine diamond ore with an iron or better pickaxe to gather diamonds for diamond tools and armour",
    "craft_diamond_pickaxe": "craft a diamond pickaxe from three diamonds and sticks; the best pickaxe the bot can make",
    "craft_pickaxe_spare": "craft the second diamond pickaxe the Nether trip needs; a pickaxe that breaks out there ends the expedition and nothing can be recovered",
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
    "hunt_food": "hunt an animal within walking range (far only in daylight), or a lone zombie, to bank food; once the gate frame stands, stock the deeper Nether reserve - the chest at the base counts towards this, so a full chest means no hunt at all",
    "take_stored_food": "take cooked food out of the chest at the base and put it in the pack, so the Nether trip is fed from what is already banked rather than hunted again",
    "cook_food": "cook carried raw meat in the furnace at the base (or the carried furnace, set up just for the batch) so it feeds more hunger; cooked meat lasts far longer than bread, and before the Nether trip any carried fuel will do",
    "collect_smelted": "take what a cook left sitting in the furnace at the base back into the pack, so the next cook can start",
    # Nether prep, stocked in the daytime lull once the gate frame stands (see `_nether_prep_steps`).
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
    "wait": "hold position at the home base; nothing productive is left in the lull before the expedition",
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
    "clear_frame": "break a container or other block the frame build cannot remove from a frame cell, then finish the frame",
    "build_portal": "build the nether portal frame from the ten banked obsidian and four cobblestone corners",
    "light_portal": "strike the built portal frame with the carried flint and steel to light it",
    "place_bed": "place the carried bed beside the gate frame so the respawn point sits at the gate",
    "take_bed": "take the standing respawn bed back up when it lies away from the base, so it can be laid beside the portal",
    "drop_junk": "drop low-value blocks from the full pack so a gathering action has room again",
    "step_away_drops": "walk a few blocks off the junk just dropped on open ground, so the bot cannot pick it up again",
    "enter_portal": "step into the lit gate and cross into the Nether for the blaze rods",
    "swap_to_gold_helmet": "wear the gold helmet instead of the diamond one, so piglins stay neutral in the Nether",
    "hunt_blaze": "kill a blaze and take its blaze rod, nine of which make eighteen blaze powder",
    "leave_nether": "step back into the Nether exit portal and cross home with the rods",
    "swap_to_diamond_helmet": "wear the diamond helmet again now that the bot is back in the Overworld",
    "craft_blaze_powder": "craft the carried blaze rods into blaze powder, one rod to two powder",
    "stow_blaze_powder": "deposit the blaze powder into the chest at the Overworld portal",
}

EDIBLE = {
    "bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_chicken",
    "cooked_mutton", "cooked_rabbit", "cooked_cod", "cooked_salmon", "carrot",
    "potato", "baked_potato", "beetroot", "melon_slice", "sweet_berries",
    "rotten_flesh", "dried_kelp", "cookie", "mushroom_stew", "beef", "porkchop",
    "chicken", "mutton", "rabbit", "cod", "salmon",
}

# `EDIBLE` answers "can this be eaten at all", and it has to stay that broad: the reflex will eat
# rotten flesh rather than starve. But a *reserve* is a different question, and one list cannot
# answer both. Live: the chest at the base held 96 rotten flesh and nothing else, so the first cut
# of the banked-chest read read that as a full larder and withheld every hunt - a Nether crossing
# stocked with emergency food, which is worse than the hunting it replaced.
#
# So the three items that cannot carry a trip are named rather than counted:
#   rotten_flesh    4 hunger, and most of it poisons
#   dried_kelp       1 hunger, for a whole stack
#   mushroom_stew    6 hunger but it does not stack: 64 of them are 64 slots
PROVISIONS = frozenset(EDIBLE) - {"rotten_flesh", "dried_kelp", "mushroom_stew"}

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
# --- The Nether expedition (phase 3) ---
# The goal the user set for the first trip: nine blaze rods, which are eighteen blaze powder (one
# rod makes two). The rods are only ever *carried* out of the Nether and turned into powder back
# in the Overworld, where a crafting table exists and the bot is not standing in a lava sea.
BLAZE_RODS = 9
BLAZE_POWDER = BLAZE_RODS * 2
BLAZE_MOB = "blaze"
BLAZE_ROD_ITEM = "blaze_rod"
BLAZE_POWDER_ITEM = "blaze_powder"
# How far a blaze may be for `hunt_blaze` to commit. Blazes hover, so `collect_mob_drop` has to
# close that distance itself; past this the bot is better off walking the fortress bricks first.
BLAZE_HUNT_RANGE = 24.0
# Blazes set themselves on fire when struck and answer with fireballs, so a hunt only starts from a
# healthy bot that is not already in a fight. This mirrors `_safe_spider_hunt`.
BLAZE_HUNT_HEALTH = 14.0
# When the Nether trip has to end. Blazes hit hard, so the bot leaves at a quarter of its health
# and well before it is hungry; both thresholds are checked on the state, never guessed at runtime.
NETHER_RETURN_HEALTH = 8.0
NETHER_RETURN_FOOD = 8.0
# How far the bot has to search before "no blaze and no fortress anywhere" counts as the end of
# the hunt rather than as open Nether seen for the first time (`_nether_hopeless`).
#
# Metres, not decisions. `NETHER_HOPELESS_AFTER = 24` counted ticks, and a tick is not search:
# live, run70, nineteen of the twenty-four ticks that ended the hunt were `body owner:
# hostile_reflex` - the bot stood at 108 m from its own portal and was shot at, and the budget
# was spent on being shot at.
#
# 700, because the old world measured the cost: its nearest fortress stood 580 m straight from
# the gate at (33,34) to (-300,513), and 400 stopped a hunt that had walked the whole way there
# and was still owed the last 180 - `nether_travel` only counts metres covered, and hostile
# avoidance, lava detours and reflex retreats make the walked line longer than the straight one
# anyway. A fortress per region is the vanilla promise, so a few hundred metres is the ordinary
# price and the long cases are not rare. The budget is a floor, not a leash: a fortress loaded
# into scan suspends the judgement, and the health and food gates end a trip that fights badly
# long before this number matters.
NETHER_SEARCH_M = 700.0
NETHER_HOPELESS_TICKS = 120
# How far a lone spider may be for a *single* string (`_safe_string_hunt`, repairing the Nether
# bow). Wider than the bed-chain's spider guard, which is about three strings at once.
STRING_HUNT_RANGE = 24.0
# Cells `clear_frame` will not dig even when the build refused them: the frame's own block and the
# fluids and bedrock a `collect_block` cannot safely break. A container the movement policy would
# not dig unasked (a furnace, a chest) is exactly what the step is for, so it is not listed here.
FRAME_UNDIGGABLE = frozenset({"nether_portal", "lava", "flowing_lava", "water", "flowing_water",
                              "fire", "soul_fire", "bedrock"})
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
              "deepslate", "dirt", "coarse_dirt", "pointed_dripstone", "lapis_lazuli",
              "leather", "bone", "raw_copper", "glow_ink_sac",
              "egg", "rotten_flesh", "mossy_stone_brick_slab", "sand", "dripstone_block")

# Geometry of the gear the tech chain upgrades through, weakest material first. A carried tool
# or armour piece of a lower tier than the best the bot already carries, wears or holds is dead
# weight no recipe here wants, but `_junk_carried` cannot see it either (the name is a real tool),
# so a full pack of nothing else never sheds and every slot-needing action answers
# `[INVENTORY_FULL]` forever (live, run28: a pack of the iron kit and the stone/wooden picks
# stalled on `craft_table` and on every mob drop). The gold helmet keeps piglins off in the
# Nether, so it is never shed even though a diamond helmet outranks it.
_TOOL_KINDS = ("pickaxe", "axe", "shovel", "hoe", "sword")
_ARMOR_KINDS = ("helmet", "chestplate", "leggings", "boots")
_GEAR_RANK = {"leather": 1, "golden": 1, "wooden": 1, "stone": 2, "chainmail": 2, "iron": 3,
              "diamond": 4, "netherite": 5}
_GEAR_KEEP = ("golden_helmet",)

# Home base. The run keeps one anchor on the surface; the bed, the portal frame and the chest
# all stand there, and the bot walks back to it to deploy them. Eight planks make a chest, which
# is how a full pack sheds surplus blocks without scattering them on the ground.
CHEST_ITEM = "chest"
SMOKER_LOGS = 4        # the server's smoker recipe: four logs *and* one carried furnace
SMOKER_FURNACES = 2    # one goes into the craft; the second stays as station for cook and iron
CHEST_PLANKS = 8
# The overflow cells of the base's storage network, in the order they are filled: behind and
# beside the main chest at `(bx, by+1, bz+1)`, marching in +z -- away from the furnace and the
# smoker (both at `bz-1`) and from the portal's frame plane (which runs along `z == bz`, so no
# candidate ever touches it).
STORAGE_SITES = ((0, 3), (-1, 2), (1, 2), (-1, 1), (1, 1), (-2, 2), (2, 2),
                 (0, 4), (-2, 3), (2, 3), (-2, 1), (2, 1), (-1, 3), (1, 3),
                 (-2, 4), (2, 4))
# How close to a chest's centre the bot must stand for `use_container` to work the cell itself;
# farther off, the storage label walks there under its own name first.
STORAGE_REACH = 4.0
# Hunger at which a hunt is worth walking even with bare hands: below it the starvation
# damage is certain, while the unarmed hunt is only a risk. run83 died twice at hunger 20,
# where nothing at all forced the unarmed walk - so the number sits well under the eating
# reflex (`EAT_HUNGER` 14), which would have let the same walk through at half hunger.
HUNT_STARVE = 8.0
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
# How far the respawn bed must stand from the lit gate, measured in blocks (Chebyshev) in the
# horizontal plane between a bed cell and any `nether_portal` block. Sleeping sets the respawn
# point to the bed's *own* cell, so a bed laid in the opening - or against the frame - drops the
# bot straight into the gate on its next death, which teleports it to the Nether within seconds
# (live: the bed sat in the opening, respawn at the gate, the Nether three seconds later, and the
# bed then stood behind the lit gate where it could not be reached). Two blocks is the clearance
# that keeps the bed on the ground *beside* the frame; the `sleep` server lays a carried bed on
# the nearest flat footprint, so `place_bed` walks the bot to a stance whose whole footprint is
# this far from the gate rather than trusting the anchor it happens to stand on.
BED_PORTAL_CLEARANCE = 2

# How far the `sleep` server looks for a standing bed: "[BED_NOT_FOUND] No unoccupied bed found
# within 32 blocks". A bed the scan sees beyond it cannot be slept in from where the bot stands.
SLEEP_BED_REACH = 32.0

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
# whole pack -- so the target deepens in the daytime lull once the gate frame stands (see
# `_food_goal`). The usual advice is 64-128 units (one to two stacks) of high-saturation food;
# a stack of cooked meat is a single inventory slot, so 64 is cheap to carry.
FOOD_RESERVE = 16
NETHER_FOOD_RESERVE = 64

# Hunger floors for eating live in `perception` now, so the narrative and these gates cannot
# drift apart: the sentence that tells the model "not regenerating" has to mean the same number
# as the gate that withholds nothing. EAT_HUNGER is the survival floor the reflex has always
# used; REGEN_HUNGER is Minecraft's natural regeneration threshold.
HEALTH_MAX = 20.0

# --- Nether prep (the daytime lull once the gate frame stands) ---
# Beyond the deep food reserve, the expedition wants a ranged answer, a piglin pass and building
# material. All of it is stocked in the same lull, so the run reaches the gate fully prepared:
#   * a bow with arrows -- the server's combat layer fires the bow itself (`survival.BASE_CHANGES`
#     `combat.bow`), so the bot only has to *carry* one and keep arrows in the pack; string comes
#     from spiders, feathers from chickens, flint from gravel, sticks from wood.
#   * a gold helmet -- one worn piece of gold armour is all it takes to keep piglins neutral; it
#     is carried until the Nether trip (worn it would displace the diamond helmet).
#   * a deep cobblestone stack for bridging over lava and gaps in the Nether. It is deliberately
#     *not* spent on a shelter around the arrival portal (decided 2026-10-18, docs/nether-prep.md
#     "(B) After arrival"): the exit stays unprotected for now, the way back is what matters.
# Every step is offered only while it can actually progress, so a material with no source (no
# spider, no chicken, no gold ore loaded) drops that step out and never blocks the run.
BOW_ITEM = "bow"
ARROW_ITEM = "arrow"
BOW_STRING = 3
NETHER_ARROWS = 16
FEATHER_ITEM = "feather"
# Pickaxes carried into the Nether, the active one included. The expedition digs: netherrack for
# cover, obsidian for a return gate if the arrival portal is gone, and the frame if the bot has to
# re-light on the far side. A pickaxe that breaks out there ends the trip where it stands, and with
# `recover_drops` refusing the Nether (`_can_recover`) there is no way back to the tools - so the
# second one is made here, in the Overworld, where a broken pickaxe costs one craft_item. Live: the
# pack held exactly one `diamond_pickaxe` with the trip stocked and the gate lit.
NETHER_PICKAXES = 2
GOLD_HELMET_ITEM = "golden_helmet"
GOLD_HELMET_INGOTS = 5
GOLD_ORE = "gold_ore"
RAW_GOLD = "raw_gold"
GOLD_INGOT = "gold_ingot"
NETHER_COBBLE = 64
# How far the Nether stack may run below `NETHER_COBBLE` before `mine_stone` refills it. The surplus
# drop leaves exactly `NETHER_COBBLE` (`HOARD_KEEP`), and the next scaffold block took it to 63:
# `mine_stone` came up for the one block, and its `collect_block` hoovered up the whole pile just
# dropped beside the bot (live, run88 it177-179: 63 cobblestone and all the junk back in the pack,
# dropped again the next tick). One `mine_stone` batch of slack keeps the two lines apart.
NETHER_COBBLE_SLACK = 8
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
# Blocks carried far past any plausible need: the run keeps at most this many of each item type
# when shedding surplus (general cap). The hoard keep is explicit for cobblestone so the banked
# Nether stack (`NETHER_COBBLE`) is never dropped.
HOARD_KEEP = {"cobblestone": NETHER_COBBLE}
HOARD_CAP = 64

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

    The helmet slot has a second owner. In the Nether the gold helmet replaces the diamond one
    (`swap_to_gold_helmet`, so piglins stay neutral) and the swap leaves the diamond helmet in
    the pack; read as "a better piece is waiting", that puts it straight back on the head and the
    two steps undo each other every tick (live, run36: `equip_armor` and `swap_to_gold_helmet`
    alternating, each action succeeding, none of them progress). So the helmet is skipped there
    while the gold piece is to hand, and the swap owns that slot for the whole trip.
    """
    inv = gs.inventory
    out: list[str] = []
    slots = [slot for slot, _ in IRON_ARMOR]
    if _in_nether(gs) and (inv.get(GOLD_HELMET_ITEM, 0) > 0 or _worn(gs, GOLD_HELMET_ITEM)):
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
    if any(not _worn(gs, f"diamond_{slot}") for slot, _ in DIAMOND_ARMOR):
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
    """The nearest fortress landmark in reach, or None.

    Blazes exist only in a Nether fortress, so the bricks and the spawner cage are the landmark
    worth walking to rather than the local greedy step `explore` would otherwise take.
    """
    for name in FORTRESS_BLOCKS:
        hit = gs.resource_nearest(name)
        if hit is not None:
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
    return (_overworld(gs) and not gs.nether_entered
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

    The gold helmet is read from the pack, not through `_all_nether_prep_steps`. That list drops
    any step whose material it cannot reach *right now*, so a bot with no coal has no gold step in
    it, the empty list reads as "prep done", and the crossing opens - into the Nether, where piglins
    are hostile and the trip dies. A burning gate is what makes this reachable: after a death down
    there the bot comes home to a gate that already burns and may have lost the helmet on the way.

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
    return (not nether_expedition_complete(gs) and not gs.nether_entered
            and _overworld(gs) and _portal_lit(gs) and _expedition_ready(gs)
            and _has(gs.inventory, GOLD_HELMET_ITEM))


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
    """
    return (_in_nether(gs) and _worn(gs, "diamond_helmet")
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


def _nether_prep_steps(gs: GameState) -> list[str]:
    """The Nether prep to offer in the daytime lull once the gate frame stands, in order, or [].

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

    Three reasons, in the order they matter:
      some base step is waiting
      the Nether gate is lit and stocked, and the bot's next job only it can do standing there
      it is night and the bed is at the base
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
            or (gs.phase == "night" and _bed_at_base(gs)))


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
    return not _at_base(gs) and _bed_in_sleep_reach(gs)


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
    `wait` alone: run88 stood twelve ticks at its base, bed in the pack, until dawn (it147-158).
    The walk west leaves the frame behind; the bed it lays there is a leftover `take_bed` brings
    home in the day, as it does for every bed slept in away from the base. A gate that may be lit
    now (`_expedition_ready`) keeps the bot at the base: lighting it and `place_bed` beside it is
    the night's work, and the end of the Overworld.
    """
    return (_overworld(gs) and not _portal_lit(gs) and _at_base(gs)
            and _has_bed(gs.inventory) and gs.base is not None
            and not _expedition_ready(gs))


def _bed_in_sleep_reach(gs: GameState) -> bool:
    """A bed the `sleep` server can find from here: carried, or standing within its search.

    `_bed_available` counts any bed the scan sees, which is right for "is a second bed owed",
    and wrong for the sleep itself: run86 started at night 51 blocks from the old base's bed,
    the server searches 32, and the exclusive night offer answered `[BED_NOT_FOUND]` on sixteen
    ticks until the morning.
    """
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
    if not gs.chest_blocked and _has(inv, CHEST_ITEM) and not _chest_ready(gs):
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
        out.append("place_furnace" if _at_base(gs) else "go_home")
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
        if fuel is not None and _cooking_reach(gs) and (fuel[0] in ("coal", "charcoal") or _nether_idle(gs)):
            out.append("cook_food")
    # Food reserve: bank edible items. At full hunger the reflex cannot eat, so gathered drops
    # accumulate; when hunger drops it spends them. Never compete with eating: feed to 20 first,
    # bank after. The goal deepens to the Nether target once the gate frame stands.
    if (_provisioned(gs) < _food_goal(gs) and not eating
            and _food_target(gs) is not None
            and _hunt_ok(gs)):
        out.append("hunt_food")
    if (_edible_count(inv) < _food_goal(gs) and _banked_provisions(gs)
            and not _in_nether(gs) and _at_base(gs) and _chest_ready(gs)):
        out.append("take_stored_food")

    wool = _wool(inv)
    string = inv.get("string", 0)
    bed = _bed_available(gs)
    # Night + a bed: sleeping skips to morning and pins the respawn point. `_sleep_ready` owns
    # *where* it may run - away from the base before the gate is lit (the server lays the bed
    # "beside the bot", and at the base that spot is the portal's own interior, which the later
    # frame build - or the still-open unlit opening - cannot clear: live, the bed landed in the
    # opening and `light_portal` answered `[PORTAL_NO_FRAME]` on every tick), and at the base on a
    # bed actually standing there once the gate is lit. `place_bed` owns the base bed, `take_bed`
    # brings a leftover home.
    # The night offer returns ahead of the withholding filter at the end, so it asks for itself:
    # run86 dropped `sleep` at it2 for `[BED_NOT_FOUND]` and the offer took the next 13 ticks
    # anyway, the window never reaching it.
    if (gs.phase == "night" and not gs.sleeping and _overworld(gs)
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
        need_wood = (tier == 0 or not table
                     or (not bed and (wool >= 3 or string >= 12))
                     or (_pickaxe_due_for_spare(gs) and not _can_sticks(inv, logs, planks)))
        if logs == 0 and planks == 0 and need_wood and _overworld(gs):
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
            if table and (wool >= 3 or string >= 12) and (logs >= 1 or planks >= 3):
                out.append("craft_bed")
            elif wool < 3 and string < 12:
                # Wool is killed for, so the walk needs the weapon too - run83's it85 stood
                # with `pickaxe: none` at hunger 20 and had only `hunt_food` and `hunt_wool`
                # on offer, two deaths later. String keeps its own guard: the spider walk is
                # how the bow's three strings are earned, and gate it and the expedition
                # cannot arm itself anymore.
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
        if gs.base is None:
            # The base, the bed and the portal all belong under open sky. A bot that mined its
            # way down stands on a cave floor whose mined `air` reads exactly like sky, so it
            # climbs back out first: `surface_up` probes the surface and walks up to it, and only
            # then is `establish_base` offered.
            out.append("establish_base" if _at_surface(gs) else "surface_up")
        elif not _at_base(gs) and _go_home_wanted(gs):
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
                and not _shed_owed(gs, inv)
                and ((has_water and _lava_pour_target(gs) is not None)
                     or _obsidian_within_reach(gs))):
            out.append("mine_obsidian")
        if (not has_fs and has_flint and iron >= FLINT_STEEL_IRON and table
                and obsidian >= PORTAL_OBSIDIAN):
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
    if (_at_base(gs) and _portal_lit(gs) and _has_bed(inv) and not _bed_at_base(gs)
            and _overworld(gs)):
        out.append("place_bed")

    # --- The Nether expedition (phase 3) ---
    # One chain, read in the order the bot has to walk it: cross, keep piglins off, kill blazes
    # for nine rods, come home with them, swap the helmet back, turn the rods into powder, and
    # park the powder in the chest at the gate. Each step only appears while the ones before it are
    # done, so no step is ever offered out of turn.
    if not nether_expedition_complete(gs):
        if _in_nether(gs):
            # The helmet first: piglins attack on sight without a worn gold piece, and the whole
            # trip is fought around them.
            if _gold_helmet_needed(gs):
                out.append("swap_to_gold_helmet")
            # A pack full of droppable junk (netherrack) reads as "out of room" and would send the
            # bot home with nothing. The junk frees itself with `drop_junk`; the room does not. So
            # while the pack is full of something that can be dropped, the bot sheds it and keeps
            # searching instead of turning back (live: the pack sat at 34/36 of netherrack and
            # `_nether_must_return` fired on the slots, `leave_nether` offered every tick).
            junk_full = (gs.slots_used >= INVENTORY_NEAR_FULL and _droppable(gs, inv))
            rods = _blaze_rods(inv)
            if rods >= BLAZE_RODS:
                # The haul is complete; every extra blaze down there is another fireball between
                # the bot and its own portal.
                out.append("leave_nether")
            elif _safe_blaze_hunt(gs):
                out.append("hunt_blaze")
            elif (_nether_must_return(gs) and not junk_full) or _nether_hopeless(gs):
                # Out of health, out of room, or out of chances: come home with what was got
                # rather than die down there. The rods in the pack are already worth the trip.
                # "Out of room" is suppressed while the room is full of droppable junk: that is
                # `drop_junk`'s job, not a reason to abandon the trip.
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
                    # Shed the netherrack first, so the next tick's slots are free and the march
                    # does not start from a full pack. Offered ahead of `explore` on purpose.
                    out.append("drop_junk")
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
                out.append("enter_portal")
            if gs.nether_entered and not _in_nether(gs):
                if _worn(gs, GOLD_HELMET_ITEM) and (_has(inv, "diamond_helmet")
                                                    or _worn(gs, "diamond_helmet")):
                    out.append("swap_to_diamond_helmet")
                if _blaze_rods(inv) >= BLAZE_RODS and _has(inv, "crafting_table") \
                        and _blaze_powder(inv) < BLAZE_POWDER:
                    out.append("craft_blaze_powder")
                if _blaze_powder(inv) > 0 and _chest_ready(gs):
                    out.append("stow_blaze_powder")

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
    if gs.base is not None and not _chest_ready(gs) and not gs.chest_blocked:
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
        if _has(inv, CHEST_ITEM) and _overflow_site(gs) is not None:
            out.append("place_chest")
        elif table and not _has(inv, CHEST_ITEM) and _plank_units(inv) >= CHEST_PLANKS:
            out.append("craft_chest")
    storage_target = _storage_target(gs)
    # The walk-over deposit works the chest where it stands: `at_chest` is the base or the
    # lid's reach, not only the base - a chest adopted outside the base radius was walked to
    # but never worked from the chest side, and the bot went home for the very surplus the
    # chest was waiting for (run82's 481-tick ping-pong).
    at_chest = (gs.base is not None and _chest_ready(gs) and storage_target is not None
                and (_at_base(gs) or _near_cell(gs, storage_target)))
    if gs.slots_used >= INVENTORY_NEAR_FULL:
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
                        or _surplus_besides_keep(gs, inv)):
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
    if cobble < 12 and _can_mine(gs, "stone") and _overworld(gs):
        out.append("mine_stone")

    # --- Nether prep (the daytime lull once the gate frame stands) ---
    # The bow, its arrows, the gold helmet and the deep cobblestone stack, stocked once the
    # gate frame stands (still unlit: the gate only lights when all of this is done). `_nether_prep_steps`
    # is the single source for both this list and the
    # completion gate, so the offered steps and `tech_tree_complete` never disagree.
    # Offered Overworld only: `_nether_idle` (its scope) is true down there too, and a prep step
    # that digs from the base has no base in the Nether to dig from.
    if _overworld(gs):
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

    # The explicit last fallback when nothing productive is available. In the lull once the gate
    # frame stands the only work left is the prep, and wandering off just burned
    # five pathfinder timeouts and left the bot away from the base at dusk (live, from-zero run), so
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
        cell = _portal_ignite_cell(gs)
        if (not _has(inv, FLINT_STEEL_ITEM) or _portal_lit(gs) or not _at_base(gs)
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
        return Action(label, "navigate",
                      {"x": tx, "y": ty, "z": tz, "dig": False, "scaffold": False},
                      "step out of the lit portal the body is standing in, before it teleports")
    if label == "take_bed":
        if (_has_bed(inv) or not _nether_idle(gs) or not _overworld(gs)
                or not _bed_needs_moving(gs)):
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
        if not _at_stance(gs, stance):
            # The `sleep` server lays the carried bed on the flat footprint beside the bot, so the
            # bot has to *stand* where the bed may go before it may sleep: walking to the clear
            # stance is half the step, and the very next tick lays the bed there.
            sx, sy, sz = stance
            return Action(label, "navigate",
                          {"x": sx, "y": sy, "z": sz, "dig": True, "scaffold": True},
                          "walk onto the spot clear of the gate, where the respawn bed goes")
        return Action(label, "sleep", {},
                      "place the carried bed clear of the gate and set the respawn point")
    if label == "sleep":
        if gs.phase != "night" or gs.sleeping or not _overworld(gs):
            return None
        if _sleep_off_base(gs):
            bx, by, bz = gs.base
            return Action(label, "navigate",
                          {"x": bx - SLEEP_OFF_BASE_STEP, "y": by + 1, "z": bz, "range": 1,
                           "dig": True, "scaffold": True},
                          "walk off the base, away from the unlit gate frame, to sleep there")
        if not _sleep_ready(gs):
            return None
        return Action(label, "sleep", {},
                      "sleep through the night and set the respawn point")
    if label == "craft_bed":
        if not _overworld(gs):
            # Nothing can lay a bed in the Nether, so there is no reason to make one there.
            return None
        if not _has(inv, "crafting_table") or (_wool(inv) < 3 and inv.get("string", 0) < 12) \
                or (_logs(inv) < 1 and _planks(inv) < 3):
            return None
        return Action(label, "craft_item",
                      {"items": [{"item_name": "white_bed"}], "temporary_workstation": True},
                      "craft a bed from wool or string and planks")
    if label == "hunt_wool":
        if _wool(inv) >= 3:
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
    if label == "smelt_gold":
        fuel = _furnace_fuel(inv)
        if (not _smelting_reach(gs) or inv.get(RAW_GOLD, 0) < 1
                or fuel is None or not _furnace_room(gs)
                or _has(inv, GOLD_HELMET_ITEM) or _worn(gs, GOLD_HELMET_ITEM)):
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
        want = BLAZE_RODS - _blaze_rods(inv)
        return Action(label, "collect_mob_drop",
                      {"mob_name": BLAZE_MOB, "drop_name": BLAZE_ROD_ITEM,
                       "count": min(3, want), "allow_without_shield": True},
                      f"kill a blaze for its blaze rod ({want} still needed)")
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
        return Action(label, "navigate",
                      {"x": ax, "y": ay, "z": az, "range": 2, "dig": True, "scaffold": True},
                      "walk back to where the Nether trip started, where the exit portal stands")
    if label == "swap_to_diamond_helmet":
        if _in_nether(gs) or not _worn(gs, GOLD_HELMET_ITEM):
            return None
        if not (_has(inv, "diamond_helmet") or _worn(gs, "diamond_helmet")):
            return None
        return Action(label, "equip", {"items": [{"item_name": "diamond_helmet",
                                                  "destination": "head"}]},
                      "wear the diamond helmet again, now that the bot is back in the Overworld")
    if label == "craft_blaze_powder":
        if (_in_nether(gs) or _blaze_rods(inv) < BLAZE_RODS or not _has(inv, "crafting_table")
                or _blaze_powder(inv) >= BLAZE_POWDER):
            return None
        want = BLAZE_POWDER - _blaze_powder(inv)
        return Action(label, "craft_item",
                      {"items": [{"item_name": BLAZE_POWDER_ITEM, "count": want}],
                       "temporary_workstation": True},
                      f"craft the carried blaze rods into {want} blaze powder, one rod to two")
    if label == "stow_blaze_powder":
        if _in_nether(gs) or _blaze_powder(inv) <= 0 or not _chest_ready(gs):
            return None
        # The blaze powder belongs in a chest of the network that can still take it: the main
        # one may well be the one a stow filled up, and powder deposited into a full chest is
        # powder held in the pack on the way back through the gate.
        cell = _storage_target(gs) or tuple(gs.chest_pos)
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
        want = _food_goal(gs) - _edible_count(inv)
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
        count = 6 if mob in FOOD_MOBS else 10
        return Action(label, "collect_mob_drop",
                      {"mob_name": mob, "drop_name": drop, "count": count,
                       "allow_without_shield": True},
                      f"hunt a {mob} for {drop}")
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
        if (gs.base is None or _at_base(gs) or _in_nether(gs)
                or not _go_home_wanted(gs)):
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
        bx, by, bz = gs.base
        if not _chest_ready(gs):
            return Action(label, "place_block",
                          {"block_name": CHEST_ITEM, "x": bx, "y": by + 1, "z": bz + 1},
                          "place the chest at the home base")
        if not _storage_full(gs):
            # The first chest is not even up yet, or one of them has room: an overflow chest
            # placed with nobody waiting for it is an item spent on furniture.
            return None
        cell = _overflow_site(gs)
        if cell is None:
            return None
        ox, oy, oz = cell
        return Action(label, "place_block",
                      {"block_name": CHEST_ITEM, "x": ox, "y": oy, "z": oz},
                      "set up another chest beside the base: the storage network is full through")
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
        items = [{"item_name": n, "count": inv[n]}
                 for n in _junk_carried(inv) + _surplus_besides_keep(gs, inv)
                 + _spare_gear(gs, inv)]
        items += _hoard_surplus(gs, inv)
        if not _chest_ready(gs) or not items:
            return None
        # A block sitting on the chest stops its lid (`[CHEST_BLOCKED]`): dig that exact cell
        # first, then open the chest on the next tick. Reusing the label keeps the relief to one
        # step; the deposit runs as soon as the cell above the chest is clear.
        if gs.chest_above is not None:
            ax, ay, az = gs.chest_above
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
        items = [{"item_name": n} for n in _junk_carried(inv) + _spare_gear(gs, inv)]
        items += _hoard_surplus(gs, inv)
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
                      "hold at the home base while nothing productive is left in the lull")
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
            heading = _nether_search_heading(gs)
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
                      {"heading": frontier_heading(gs), "chunks": EXPLORE_FRONTIER_CHUNKS},
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
