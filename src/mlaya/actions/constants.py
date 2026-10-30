from __future__ import annotations

import math
from dataclasses import dataclass

from ..perception import (AIR_NAMES, EAT_HUNGER, INVENTORY_NEAR_FULL, INVENTORY_SLOTS,
                          PORTAL_BLOCK, REGEN_HUNGER, GameState, _in_nether_dimension,
                          frame_obsidian_cells, frame_wanted_blocks)
from ..targets import (EXPLORE_FRONTIER_CHUNKS, EXPLORE_MIN_M, base_target,
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
    "mine_gold_reserve": "mine extra gold ore to trade with piglins in the Nether for ender pearls",
    "smelt_gold": "smelt raw gold into gold ingots in the furnace for the gold helmet",
    "craft_gold_helmet": "craft the gold helmet that keeps piglins from attacking; carry it until the Nether trip",
    "replace_tool": "craft a fresh pickaxe to replace the one that is nearly worn out",
    "collect_drops": "pick up items a mob dropped nearby when the survival reflex killed it",
    "recover_drops": "walk back to the last death site and recover the gear dropped there",
    "explore": "walk to new terrain to find missing resources; the last fallback",
    "wait": "hold position at the home base; nothing productive is left in the lull before the expedition",
    "make_room": "dig the block a placement was refused at, or a small pocket, so a carried workstation or bed fits",
    "dig_out": "dig a short tunnel out of the pocket the bot is shut in, so it can walk anywhere again",
    # Home base: one surface anchor the bed, the portal and the chests all stand at.
    "surface_up": "climb back out to open surface ground; the home base, the bed and the portal all belong under open sky, never on a cave floor",
    "establish_base": "walk up to open surface ground and set up the home base there",
    "go_home": "walk back to the home base to place the bed, the portal or store surplus there",
    "craft_chest": "craft a chest from eight planks to store surplus blocks at the base",
    "place_chest": "place the crafted chest at the home base",
    "place_furnace": "place the furnace the bot carries at the home base, so cooking and smelting reuse it instead of setting up a temporary one every time",
    "stow_surplus": "deposit the surplus blocks of a full pack into the chest at the base",
    "flatten_home_base": "level the home base: dig away the stray terrain and the reflex-placed defence blocks that pile up above ground level around the lit gate, so the home ground stays flat",
    # Overworld -> Nether gate (the portal frame and the bed beside it).
    "craft_bucket": "craft a bucket from three iron ingots; a water bucket is how obsidian is made",
    "fill_bucket": "fill the carried bucket with water at a source, ready to pour onto lava",
    "mine_obsidian": "carry a water bucket and mine obsidian: the dig pours the water onto lava and mines what forms",
    "approach_lava": "walk to the lava the scan can see, so the source that forms obsidian comes into reach",
    "get_flint": "dig gravel for flint, the spark the flint and steel needs",
    "craft_flint_and_steel": "craft flint and steel from flint and an iron ingot to light the portal",
    "clear_portal": "dig stray blocks out of the portal opening so the frame can be built and lit",
    "clear_frame": "break a container or other block the frame build cannot remove from a frame cell, then finish the frame",
    "build_portal": "build the nether portal frame from the ten banked obsidian and four cobblestone corners",
    "light_portal": "strike the built portal frame with the carried flint and steel to light it",
    "place_bed": "place the carried bed beside the gate frame so the respawn point sits at the gate",
    "take_bed": "take the standing respawn bed back up when it lies away from the base or underground, so it can be laid beside the portal",
    "drop_junk": "drop low-value blocks from the full pack so a gathering action has room again",
    "step_away_drops": "walk a few blocks off the junk just dropped on open ground, so the bot cannot pick it up again",
    "enter_portal": "step into the lit gate and cross into the Nether for the blaze rods",
    "swap_to_gold_helmet": "wear the gold helmet instead of the diamond one, so piglins stay neutral in the Nether",
    "hunt_blaze": "kill a blaze and take its blaze rod, nine of which make eighteen blaze powder",
    "barter_pearls": "trade gold ingots with an adult piglin for ender pearls, the other half of an eye of ender",
    "leave_nether": "step back into the Nether exit portal and cross home with the rods",
    "swap_to_diamond_helmet": "wear the diamond helmet again now that the bot is back in the Overworld",
    "craft_blaze_powder": "craft the carried blaze rods into blaze powder, one rod to two powder",
    "stow_blaze_powder": "deposit the blaze powder into the chest at the Overworld portal",
    # The stronghold phase (the step after the Nether haul); see docs/stronghold.md.
    "hunt_enderman": "kill an enderman and take its ender pearl, one of the two halves of an eye of ender",
    "search_enderman": "roam out to new ground to find an enderman, when pearls are still needed and none is in reach to hunt",
    "take_blaze_powder": "take the banked blaze powder back out of the chest at the portal, so eyes of ender can be crafted from it",
    "craft_ender_eye": "craft an eye of ender from one ender pearl and one blaze powder; twelve of them open the end portal",
    "locate_stronghold": "throw an eye of ender to find the stronghold, then travel to it and stand at the end portal frame",
    "activate_end_portal": "fill the end portal frame's empty sockets with the carried eyes of ender until the portal opens",
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
# How far a fortress landmark has to be to be worth walking to. Live, run104: the bot stood on a
# spawner at (94,39,86) for 100+ ticks, `navigate` answering `ok` with 0 metres moved, because the
# landmark it was told to walk to was the block under its own feet.
FORTRESS_MIN_M = 2.0
# --- The stronghold phase (phase 4) ---
# Twelve sockets activate the end portal. Up to four may generate pre-filled, the estimate and the
# locate each throw an eye, and about one throw in five shatters; sixteen eyes is the safe upper
# bound that still leaves enough after the throws to fill twelve empty sockets. The counts live
# here because the Nether gates need them: `nether_expedition_open` asks whether a top-up trip is
# wanted, and that predicate cannot reach into `stronghold` without an import cycle.
#
# The rod haul is sized on these numbers, not on a flat nine: the chain is rods -> powder -> pearls
# -> eyes -> frame, so the rods owed are the ones the *empty sockets* still need, and a rod carried
# from an earlier run is supply the demand already counts (see `_rods_target`).
END_PORTAL_SOCKETS = 12
# The eyes carried beyond the frame's empty sockets: the estimate throw spends one, and roughly one
# throw in five shatters. Four is the working margin; once the estimate is saved the buffer is no
# longer owed and the frame's sockets alone decide the count.
EYE_THROW_BUFFER = 4
ENDER_PEARL_ITEM = "ender_pearl"
ENDER_EYE_ITEM = "ender_eye"
ENDER_EYES = END_PORTAL_SOCKETS + EYE_THROW_BUFFER
ENDER_PEARLS = ENDER_EYES
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
# How many blocks short of the exit portal the way-home walk aims. `navigate` refuses a target
# that is an active portal cell outright (`[NAVIGATION_PORTAL_INTENT_REQUIRED] Use
# enter_nether_portal to enter this active portal`), whatever its `range` - so the walk aims at a
# standable doorstep this many blocks out on the bot's own side instead, clear of the two-wide
# interior and its obsidian frame. Three keeps the stop at the gate's threshold yet well inside
# the scan radius, so the portal block loads and `_exit_portal_cell` crosses on the next tick
# (live, run99 it347-599: the old aim straight at the portal cell was refused 178 times, the
# walk never started, and the bot starved at a gate it could see but never reach).
NETHER_GATE_APPROACH = 3
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
# How close the body has to be for that search to end in a sleep. Minecraft accepts the bed click
# only within two blocks of the bed's top centre (`distanceToSquared(x+0.5, y+1, z+0.5) > 4`
# refuses it), and mineflayer's `sleep` reaches the same check before activating the block. Live,
# run121 it100-103: the bot stood at (-9.3,70,-52.5) with the bed at (-10,70,-51) - 1.7 blocks off
# in plan, but a full block above the bed's own level, 5.04 squared - and four identical
# `[BED_SLEEP_REJECTED] bot is not sleeping` ticks followed.
BED_SLEEP_CLICK = 2.0
# A night sleep away from the base needs open ground and no monster at hand. Minecraft refuses a
# sleep with hostiles near the bed but sets the respawn point first, so a refused sleep still
# pins the respawn wherever the bed was laid: run87 slept at y=-10 in a zombie cave (it193-197,
# five `[BED_HOSTILES_NEARBY]`), the bed stayed, and the bot fought down to 0.2 health with its
# respawn among the zombies. The hostile radius is wider than the game's own eight blocks, so the
# bed is never laid for a sleep that will be refused. Underground means feet (or a bed) more than
# `SLEEP_DEPTH_TOL` below the base floor - or below `SURFACE_FALLBACK_Y` while there is no base -
# where the night changes nothing and a bed is only a respawn point inside rock. The margin is
# wide because the base can stand on a hill above a valley's open ground; reading a valley as a
# cave only skips one sleep, reading a cave as ground lays the respawn in it.
SLEEP_HOSTILE_CLEAR = 12.0
SLEEP_DEPTH_TOL = 16
SURFACE_FALLBACK_Y = 56

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
# Health under this hands a hostile `yes` to the survival policy (`brain.danger_matters`): the
# body is at risk, so the loop stands aside while the server's reflexes fight and eat.
RECOVER_HEALTH = 15.0
# Defers in a row that took no damage and saw no reflex act (`graph.Agent._note_defer`). After
# this many the policy is not doing anything with the tick: above `RECOVER_HEALTH` the loop
# works again (run86, blazes at a spawner that no longer attacked), and below it the loop takes
# the tick only to recover (`_recovery_owed`).
DEFER_QUIET_TICKS = 5

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
# --- Piglin barter for ender pearls (docs/stronghold.md) ---
# The pearl half of an eye of ender comes from trading gold ingots with adult piglins in the
# Nether rather than hunting endermen in the Overworld. Barter is random: a piglin throws back
# one item from its loot pool per ingot offered, so the reserve is a budget, not a guarantee, and
# each call caps the offers it is allowed to make. The gold is smelted at the base furnace before
# the crossing (there is no coal to smelt in the Nether) and carried down as ingots.
GOLD_BARTER_RESERVE = 16
BARTER_GOLD_BUDGET = 8
BARTER_PEARL_BATCH = 8
# How many barter calls one run may make before the route is written off. A piglin rolls once per
# ingot and the pearl entry of that roll is about 2% for 2-4 pearls, so a 16-ingot reserve buys
# roughly one pearl; without a cap the lull would keep digging gold for a barter that keeps
# missing, and the run would end on its iteration budget with the pearls still owed. Once the cap
# is reached the enderman hunt is the pearl source again.
BARTER_MAX_ATTEMPTS = 2
# A piglin must be this close before the bot trades with it: the `barter` call navigates to the
# piglin itself, and a long walk to a distant one is the same off-base drift the Overworld pearl
# hunt stalled on (run110). Beyond this range the bot marches the frontier instead.
BARTER_PIGLIN_RANGE = 32
# --- The pearl waves (docs/stronghold.md, "Nether in Wellen") ---
# The pearls are hunted in waves: the first Nether trip hauls only blaze rods and comes home to
# stow them (a death down there drops the whole pack), and only then do the pearl trips begin -
# gold dug fresh, carried down to barter, and endermen hunted in the Nether. Barter buys roughly
# one pearl a reserve, so most of the sixteen come from kills, and endermen live in the Nether's
# warped forests rather than at the fortress the rod trip targeted - so a pearl trip may have to
# roam to find one. This budget bounds that roam the way `NETHER_SEARCH_M` bounds the fortress
# hunt: metres of ground covered, not decisions, so the reflex ticks the bot stood still being
# shot at do not spend it. Shorter than the fortress budget on purpose - a warped forest is not a
# guaranteed region feature, so an empty stretch should send the bot home to stow and relaunch
# rather than burn the whole trip wandering.
NETHER_PEARL_SEARCH_M = 400.0
# How many pearl trips the waves make before the Overworld enderman roam takes over as the
# fallback. Each trip crosses, hunts, and comes home to stow whatever pearls it got, so the cap
# bounds the round trips, not the time in any one of them (`_nether_must_return` already does
# that). Five, because a trip that barters once and kills a few endermen brings home two or three
# pearls, so five waves cover the sixteen with room for the ones a trip comes home empty.
PEARL_WAVE_MAX_TRIPS = 5
NETHER_COBBLE = 64
# How far the Nether stack may run below `NETHER_COBBLE` before `mine_stone` refills it. The surplus
# drop leaves exactly `NETHER_COBBLE` (`HOARD_KEEP`), and the next scaffold block took it to 63:
# `mine_stone` came up for the one block, and its `collect_block` hoovered up the whole pile just
# dropped beside the bot (live, run87 it177-179: 63 cobblestone and all the junk back in the pack,
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

# Mobs the server files under `kind: hostile` that do not start a fight on their own. Piglins and
# zombified piglins turn on a player only when provoked - by a hit, or by a ghast fireball that
# bounces back - and a fortress is full of them, so counting them as company closes every hunt
# guard. Live, run119 it221-224: a blaze 3.9 m away was never hunted because 53 zombified piglins
# stood 9.1 m off, and the bot took fireballs standing still with `hunt_blaze` never offered.
NEUTRAL_HOSTILES = frozenset({"piglin", "zombified_piglin"})

# The blocks the server will place while fighting - the `navigation.scaffold_blocks` the survival
# policy admits (`survival.BASE_CHANGES`). An enderman hunt roofs itself over as the mob teleports,
# and `collect_mob_drop` refuses with `[COMBAT_BUILD_MATERIALS_MISSING]` when fewer than six of
# these are carried at the build cell. Kept in step with `survival.py` by hand - predicates must
# not import the policy module.
COMBAT_SCAFFOLD_BLOCKS = frozenset({
    "dirt", "cobblestone", "cobbled_deepslate", "netherrack", "basalt", "end_stone",
})
# How many scaffold blocks the bot must carry before an enderman hunt is offered. The server's
# defensive roof needs six at the selected cell; the fight rebuilds the roof each teleport, so a
# hunt started on bare six runs dry on the first one. Sixteen is a working reserve well under the
# `NETHER_COBBLE` (64) stack the wave carries, so a normally stocked trip clears it and only a
# depleted bot sits the hunt out (live, run112d it240-243: five netherrack left, the roof wanted
# six, and `hunt_enderman` refused four ticks running until the stuck guard dropped it). Short of
# the reserve with an enderman in reach, the pearl wave comes home to restock and relaunch.
ENDERMAN_HUNT_BLOCKS = 16

# --- Home-base leveling (`flatten_home_base`) ---
# The survival reflex stacks scaffold blocks during a fight, and a buried-and-raised base leaves
# stray terrain, and neither is ever cleared, so the home ground turns lumpy and sometimes hides a
# tunnel mouth. Once the gate is lit and the base is settled, a daytime maintenance step digs these
# back to air. The region is the base footprint, read in one `view_blocks box` whose half-width the
# server caps at 4 (a 9x9 column field centred on the anchor, which is why `FLATTEN_RADIUS` is 4 and
# must match `perception.scan_base_box`); `FLATTEN_UP` is how many rows above the ground floor `by`
# are cleared, enough for a scaffold tower the reflex roofs itself over with.
FLATTEN_RADIUS = 4
FLATTEN_UP = 4
# At most this many cells dug per call: `build_structure` runs one route search per cell and a long
# list has watchdog-killed the server, so the scan re-reads and the rest falls to the next tick (the
# `clear_portal` four-cell cap, a touch higher since these sit right at the bot's feet).
FLATTEN_BATCH = 8
# The blocks `flatten_home_base` will dig: ordinary terrain and the scaffold the combat reflex
# places (`COMBAT_SCAFFOLD_BLOCKS`). A whitelist, not a blacklist, so the base's own furniture is
# safe by construction -- a chest, furnace, smoker, crafting table, bed, the obsidian frame and a
# lit portal block are none of these and are never dug. Cobblestone *is* listed on purpose: the
# portal's four cobblestone corners are only needed while the frame is built and the gate works
# without them, so a corner left standing above ground level is cleared like any other bump
# (user call, 2026-10-30). Only cells above the floor `by` are ever dug, so the ground the bed and
# furniture stand on is never undercut, and the two lower corners (at `by`) stay bedded in the soil.
FLATTEN_BLOCKS = COMBAT_SCAFFOLD_BLOCKS | frozenset({
    "dirt", "coarse_dirt", "rooted_dirt", "grass_block", "podzol", "mud", "clay",
    "stone", "cobblestone", "cobbled_deepslate", "deepslate", "tuff", "calcite",
    "andesite", "diorite", "granite", "dripstone_block", "gravel", "sand", "red_sand",
    "sandstone", "snow_block", "moss_block",
})

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


# Re-export the whole module namespace (public and private) so a downstream
# `from .<module> import *` carries every constant and helper by its original name,
# and `mlaya.actions.__init__` can hand the full surface back out unchanged.
__all__ = [_name for _name in list(globals()) if not _name.startswith("__")]
