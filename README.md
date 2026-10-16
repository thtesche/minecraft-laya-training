# minecraft-laya-training

A Minecraft agent whose **System-1 decisions are made by
[Laya](https://github.com/NandhaKishorM/laya)** — a non-autoregressive
typed-decision engine (`choice` / `score` / `noul`) — and which acts through the
**mine-ai MCP server**. A generative LLM (OpenRouter) serves as **System 2**
for rare escalations.

Laya produces no text and no tool arguments. The code therefore filters the currently
**valid options** (preconditions/tech tree), and Laya decides in **one** forward pass
over a binary danger gate plus the choice among those options; a separate Python **executor**
calls the MCP tools and respects the server's foreground gating.

Current state (**M2/M2b/M2c/M2d/M3/M4 + homebase/Nether**): progression runs from wood through a crafting table and stone tools
up to the **iron pickaxe** (furnace + smelting) and on via `mine_diamond` to the **diamond pickaxe**
(+ diamond sword and **a full diamond armour set**: helmet/chestplate/leggings/boots). `view_blocks`
feeds the nearby resources
(wood/stone/iron/coal/diamond with distance) into the state. On the surface the agent lays out a
**homebase** (the anchor for bed, portal and chest): it builds a **bed** (from sheep wool or
spider string), **sleeps in it at night** (skips the night, sets the respawn point), makes
**obsidian** from a water bucket, lights a **Nether portal** with it, and keeps surplus in a
**chest**. It reads the pending **notifications** and at start sets the
**server survival policy** (scaffold blocks, a cautious combat window, `food.raw.allow: always`);
eating happens through the server reflex, food is procured via `hunt_food`/`cook_food`, which holds a
**minimum reserve of 16** edible items. The policy runs in two **stages** (`base` for travel/
progression, `hunt` for a hostile hunt). The agent **reacts to event types**: a nearly
worn-out pickaxe is replaced (`replace_tool`), drops from a mob killed by the reflex are
collected (`collect_drops`), and freshly lost gear is recovered at the death site
(`recover_drops`). Since **M3** a Laya training dataset can be built from the run log and the
`choice` head be retrained (RLCD); the checkpoint is registered under `english` via
`LAYA_EXTRA_MODELS` — without a code change. **M4** adds a **parameter layer**: where a label says only
*what* (e.g. `explore`), Laya picks the *where* from the `view_blocks` candidates — real standing cells
with `above: air` — and binds them to a `navigate` target; with many candidates
`predict_shortlist` narrows them down first.

The M2/M2b/M2c/M2d paths were tested live: `replace_tool` (worn pickaxe → replacement), `hunt_food`
(animal hunted, reserve topped up), `collect_drops` (reflex kill → drops collected),
`recover_drops` (death in place → gear recovered) as well as the **bed chain** `craft_table` →
`hunt_wool` (sheep) → `craft_bed` → at night `sleep` (the night was skipped to day) all ran.

The **iron chain** also ran live (tier 0 → 3): `craft_pickaxe` → `mine_stone` →
`craft_stone_pickaxe` → `mine_iron` → `mine_coal` → `craft_furnace` → `smelt_iron` → `craft_iron_pickaxe`.
In the process a blocker was found and fixed: in a 1×1 shaft after mining the ore,
`craft_item` failed with `[WORKSTATION_PLACEMENT_FAILED] … no_cell` (no room for the carried
crafting table) —
an **endless retry** in the loop. `graph.Agent` remembers the blocked label and inserts an
**unblock**: first `make_room` (digs four cells free with `build_structure`, right at the
spot), then `explore` into open terrain. Reproduced live in a deep shaft and fixed
(`no_cell` → `make_room` → the same craft goes through). Pinned in `tests/test_graph.py`.
The same mechanism now also covers the **bed spot**: in the armour run, `sleep` in the
1×1 shaft answered three times with `[BED_PLACEMENT_FAILED] … no … flat ground` (three wasted ticks).
`BLOCKED_CODES` therefore also contains `BED_PLACEMENT_FAILED` → the unblock first digs a pocket
(`make_room`) in which the bed finds two supported cells.

The **diamond chain** has now also run live: `mine_iron`/`mine_coal`/`smelt_iron` →
`craft_iron_pickaxe` (replacement, after the iron pickaxe had broken in the tunnel) → `mine_diamond` →
`craft_diamond_pickaxe` → two more diamonds → `craft_diamond_sword` (final state tier 4,
pickaxe **and** sword of diamond). In the process a **deadlock** was found and fixed: `mine_diamond` was
capped at `tier < diamond` and was therefore not allowed to mine, with the diamond pickaxe itself, the two diamonds
for the sword — the sword was thus unreachable. The cap is removed (regression tests in
`tests/test_actions.py::DiamondChainTest`). Second finding: the iron pickaxe can break *within* a
long `collect_block` (the `replace_tool` reaction only checks between two ticks). **Hardening:**
`replace_tool` now additionally reserves a fresh replacement of the same tier at ≤ `MINING_HEADROOM = 80`
remaining uses; if the active pickaxe breaks, the server pushes the replacement in, and the tier is preserved.
A guard prevents multiple crafting of the same replacement. On top of that an explicit ore rule
(`ORE_MIN_TIER`): cobblestone/coal need any pickaxe, iron a stone one, diamond an
iron pickaxe (gold counts as tier 1, so it destroys both). `_can_mine` starts a mining run
only if a break *midway* would also not fall below the minimum tier — otherwise the
lower tier pushed in afterwards destroys the remaining ore without a drop. Stone and coal are exempt, otherwise
the bootstrap would block (a wooden pickaxe must still be allowed to collect the cobblestone for the stone
replacement). And the earlier upper bound `tier < IRON_PICK_TIER` for `mine_iron` is gone: a worn iron pickaxe can
now mine the iron for its own replacement (a fresh one still mines no iron).

**Diamond armour (validated live):** `_next_diamond_item` extends the
diamond goals beyond the tool chain + sword to the complete armour set
(`DIAMOND_ARMOR`): chestplate (8), leggings (7), boots (4), helmet (5) = **24 diamonds**,
ordered by **protection per diamond** (chestplate first), so that a partial set is already as useful
as possible. Only ever **the next missing piece** is offered (`mine_diamond` with
`count = cost − worn diamonds`, otherwise `craft_diamond_<slot>`); an `equip_armor`
(an `equip` batch, without a destination) puts crafted pieces on. Armour needs **no sticks**,
only the crafting table, and counts as "present" whether **worn** or still in the inventory — so that no
piece is crafted twice. **Ran live** (25 ticks): `mine_diamond`
(8/7/4/5 = exactly 24) → `craft_diamond_chestplate` → `leggings` → `boots` → `helmet`, final state
all four pieces in the inventory. **Correction:** the server does **not** equip by itself — the pieces
initially stay in the inventory (`location: main`). Only what an `equip` (`equip_armor`)
moves into an armour slot is worn. `situation.tools.armour` lists **carried** armour per class
(whether on or not); the **worn** set comes from the inventory `location`
(`head`/`torso`/`legs`/`feet`).

**Shield & iron combat kit:** The shield (6 planks + 1 iron ingot, 3×3 → crafting table) is a
small tier of its own: `craft_shield` → `equip_shield`. It blocks only **in the off-hand**, and a
crafted shield stays — like the armour — initially in the backpack; only an `equip` with
`destination: "off-hand"` puts it there. `view_status` reports the off-hand item as an
inventory stack with `location: "off-hand"`; `GameState.offhand` reads that, and `verify.diff` counts
`offhand_delta` as progress of `equip_shield`. **Live finding (run12):** the bot only dived to the diamonds with
an iron pickaxe, met a zombie and a creeper and died; the subsequent run to the
death site lost the race against the item despawn. Hence now an **iron combat kit** before the first
diamond trip: `craft_iron_sword` + `craft_iron_<slot>` (chestplate 8, leggings 7, boots 4, helmet 5,
**protection per ingot** first, only ever the next missing piece) → `equip_armor`, and `mine_diamond`
is gated behind `_combat_ready` (iron sword or better, shield, all four slots filled with iron or
better). `equip_armor` puts on iron **and** diamond and never swaps in a worse
piece. `_iron_goal` is the single iron source: 3 ingots for the (spare) pickaxe, otherwise shield 1 +
sword 2 + missing armour pieces; `mine_iron`/`smelt_iron` share it, `mine_iron` is per action
capped at `IRON_MINE_BATCH = 8`. All iron crafts wait for the iron pickaxe so they
do not steal the first ingots.

**Deep deaths (run12):** `recover_drops` now waits for a **pickaxe** when the death lies
clearly below the bot (`RECOVER_DIG_DEPTH = 4`): the way back has to dig down, and with
a bare hand the bot loses the race against the 5-minute despawn (live: hand-dug from the surface
to a death ~60 blocks deeper → lost → endless loop on `get_wood`). In addition,
`[NO_REACHABLE_MATCHING_TARGETS]` (the server found target blocks but reached none — e.g. a
tree it refuses because of liquid isolation) now counts towards `BLOCKED_CODES`: instead of
hammering in place, the unblock digs a pocket and then falls back to open terrain.

**Two holes after the first death (run14):** after the textbook build (full
diamond kit at it66), run14 faced two bugs that share the same pattern — a state nobody
leaves again. **(1) `recover_drops` never gave up:** the server answers `[NO_MATCHING_ITEMS]` when
it reaches the death site and its 16 m scan finds nothing anymore (drops despawned/burned) — a
**definitive** answer. `_can_recover` still offered the trip for the whole 300 s window, the
reaction held the body every tick and the bot stood at the empty spot (live 40 + 77
ticks, over half the run; the second block only ended when the window expired). `_note_recover`
now writes off the death cell at the first final code (`RECOVER_FINAL_CODES`; the transient
`RECOVERY_SITE_UNREACHABLE` **not**), and `_can_recover` rejects it — a later death elsewhere
is a new cell and is still walked. **(2) Healing dead zone 14–17:** natural
regeneration requires hunger ≥ 18, but `eat_food` only kicked in at hunger < 14 — a bot with hunger 17
and 2.7 HP never ate and never healed (live 17 ticks, then killed by a mob). `eat_food` now kicks in
also at an injury below the regen threshold (`health < 20 and food < 18`), but eats only when **injured**,
so that a healthy bot spares its reserve. Pinned in `tests/test_actions.py` and
`tests/test_graph.py`. The third finding (the `craft_shield` tick was the only candidate while
the server reflex fought on the spot) is untouched.

**Homebase & Nether portal:** The base stands under **open sky**. A tunnel the bot dug itself
leaves `air`, not `cave_air`, and is therefore indistinguishable from real sky by *no*
`view_blocks` value (run9 anchored the base this way at `y = -16` under a
`y = 73` surface). Only the **depth below the surface** is reliable: as long as the
surface is unknown or the bot stands too deep, the candidates offer `surface_up` — a
`navigate` **without `y`** asks the server for the surface of the column (it climbs up itself when the column is unambiguous,
otherwise it stops with `[NAVIGATION_HEIGHT_REQUIRED] … at y=N`), and the next
tick climbs with `dig`/`scaffold` to exactly that cell (live: `y = -15` → `y = 74` in ~92 s).
Only then is `establish_base` allowed (`_at_surface`: the **bot's own** column of the learned surface
and feet within `BASE_SURFACE_TOL = 2`; the earlier tolerance 8 without a column check put
run10's base underground and wedged the bot into a flooded shaft); the
base scan `_scan_build_site` also only runs at the surface. Once the diamond kit stands,
`establish_base` looks for a flat,
free spot for the portal frame: two `view_blocks box` reads around the bot — its own level and
four above it, so that flat terrain over a hollow is found too — (`perception.scan_ground`
→ `targets.flat_build_site`) require a continuous floor row at the same height with air
above **and solid ground one level below**: the server digs out the floor row and re-lays it
as obsidian, so a cell over a hole would wrongly stay "without a placement surface" and the
opening would never close (live, `light_portal` hung in `[PORTAL_NO_FRAME]` this way); likewise the frame
would otherwise get stuck in stone on sloping ground. If the scan finds nothing, the
choice falls back to the next open surface above the bot (`targets.base_target`). `go_home`
only walks back when work waits there (`_base_steps_ready`). Bed, portal frame and chest all stand
at this anchor (`build_portal`/`place_bed` only at `_at_base`). The **portal opening** is
additionally checked and cleared **before the build**: the loop reads the six inner cells
(`perception.scan_frame_volume`) and `clear_portal` digs away a foreign block there before
`build_portal` runs (at most four cells per call). This is needed for the bases that had *no*
scan — an adopted one or one chosen via `base_target` — because otherwise a block above the
support layer is rebuilt, and `light_portal` answers `[PORTAL_NO_FRAME]` forever (live, the
generic night sleep threw a bed right there). The Nether path:
`craft_bucket` (3 iron ingots) → `fill_bucket` → `mine_obsidian` → `get_flint` (gravel) →
`craft_flint_and_steel` (2×2) → `clear_portal` → `build_portal` → `light_portal` → `place_bed`.
`mine_obsidian` **creates** the obsidian: the run pours the water bucket onto the nearest
lava (`use_bucket pour` onto the lava cell, if need be walking there first) and mines exactly the blocks
that formed. `cave_air` also counts as "open" above the pour cell (`perception.AIR_NAMES`): underground
lava lakes are covered by cave air instead of `air`, and the pour check rejected such cells
at first — fixed live in the from-zero run. Natural obsidian lies only deep below the same lava lakes (`y ≈ -55`) and is deliberately
not mined — a deep-drilling attempt ended in the **water trap** (floating without ground, every
following `collect_block` ran empty). The **water escape reflex** handles that: `in_water and
not on_ground` makes `escape_water` the only candidate after `WATER_RESCUE_AFTER` (= 2)
decisions without ground and navigates back to the nearest dry ground (the base only as a
last resort). The first decisions are left to the goal: the server navigates a
swim leg itself, and a crossing the bot aborts in the water was otherwise pulled straight back
(**live run15**: 155 rescues against 30 diamond digs, 157 breath aborts,
0 diamonds). The rescue reads the **wider** ground radius (`RESCUE_RADIUS` = 128, and prefers a
cell at the bot's own level within `RESCUE_Y_BAND`): the point is the nearest water-safe cell, and
the nearest dry block a 64 m read sees can be a *cave floor* far below (live run19b: the rescue
aimed 31 blocks straight down at a cavity floor, `closest node was -182,59,-85`, while the shore
that would have ended it sat at 65.1 m — 1.1 m past the 64 m read). A **wedged** bot (server stop `repeated_movement_failure`)
counts as blocked surroundings, so the unblock ladder kicks in instead of an endless loop. The key to the chain remains the **obsidian latch**, but it is now the **frame** rather than the igniter: `mine_obsidian` stands while no obsidian has been raised in the base's portal plane (`_frame_started_at_base`), so it does not dig again after the build (which yields the 10 blocks) — and, unlike the old `not has_fs` latch, it still mines when a restart carries the flint & steel into a world whose pack is short or whose base was re-anchored away from the old frame. The bed is
placed **only at night** — only sleeping sets the respawn point. At the
base the generic night sleep is **suppressed** in front of the lit portal: the server places
the bed "next to the bot", and there that would be the portal interior; the frame build cannot
remove a bed, so live `light_portal` hung at `[PORTAL_NO_FRAME]` every tick. The base bed
is therefore placed by `place_bed` after the portal burns. From that point `sleep` is also
restricted to the base, and a bed found standing **away** from it — a leftover from an earlier base
or an earlier run — is taken up (`take_bed`) and laid again at the base, so the gate bed does not
sit in the base's own dig shafts (live, run18: the run17 bed at y=55 tripped every `mine_gold`
tunnel with a "no observed settled arrival" on its half-block top).
Surplus is handled by a **chest** (`craft_chest` from 8 planks → `place_chest` → `stow_surplus`) from
33/36 slots; `drop_junk` only throws as a **last resort** when there is no chest or it is full. An
`[INVENTORY_FULL]` refusal triggers a relief stage of its own (`inventory_relief`). Base and chest
count **per run** and are injected into every `GameState` by the loop; if a
bed (and a chest) is already in reach at start, they are adopted so that a restart at
the same homebase continues instead of building a second one — but a bed is adopted only once the
loop has proven its ground can carry the frame (`_scan_bed_site` → `targets.bed_frame_site`): a bed
laid underground or on a slope is refused, so the run anchors a usable base instead of pinning the
frame where its cells stay stone (live, run17). If the chest cannot stand at the chosen spot — the
server answers `[PLACE_NO_SUPPORT]` when no solid block borders the cell (live, the base
stood on a narrow edge and the run hung on `place_chest` every tick instead of building the portal) — the loop drops the chest chain: the relief then simply throws the surplus
on the ground (`drop_junk`), which frees the same slots and cannot fail. If, on the other hand, the
target cell is **occupied** (`[PLACE_TARGET_OCCUPIED]` — live in the from-zero run: a grass block in
the chest cell), the loop remembers the cell (`gs.blocked_cell`) and points the unblock exactly
at it: `make_room` digs out the refused cell, after which `place_chest` goes through (it10 → it11 →
it12). Pinned in `tests/test_actions.py::MakeRoomTest`.

**Defused:** `hunt_food` used to roam unprotected to the nearest creature (a night run wandered
159–190 m and was killed twice, total loss of the inventory). It is now tied to a
**daylight barrier** (`FOOD_HUNT_RANGE_DAY = 64` / `FOOD_HUNT_RANGE_NIGHT = 16`): at night
only animals in reach, during the day also more distant ones. Widening the day reach to 128 to
make a stricter food gate reachable was tried live (run19b) and reverted — it walked the bot
150–200 m from the base, where the base-range latches re-open the portal chain and it stranded
in water. It still holds that at night the
server `hostile_reflex` temporarily holds the body ("body owner: …"), and mobs in the list (≤ ~150 m)
must first be **loaded** to be huntable (`[HUNT_TARGET_NOT_LOADED]`).
Details + 22 determinism tests: [`docs/architecture.md`](docs/architecture.md#m2d--reached-reacting-to-eventnotification-types).

**Lull after the portal (found live):** Once the kit stands and the portal burns, during the day only
the night bed is still open — the bot fell into pure `explore` (25 idle ticks, 5 of them
`[NAVIGATION_STOPPED]` timeouts). This lull is now Nether preparation: the food reserve is topped up to
the **Nether target `NETHER_FOOD_RESERVE = 64`** (one full scoop, ~1 inventory slot)
(`_food_goal`; live the bot had 17), and `cook_food` may then cook with **any** fuel
(`_furnace_fuel`: coal first, otherwise planks/logs) — before, it required coal, and without coal the
raw meat stayed put. So that the furnace is not constantly torn down in the process — `smelt_item` with
`temporary_workstation` places the carried furnace per call and picks it up again —,
`place_furnace` sets it down once at the base (coordinates as `gs.furnace_pos`, adopted from the
`furnace` scan on a restart) and `cook_food`/`smelt_gold` then use the same furnace. Eating is by
**saturation** (`FOOD_SATURATION`): steak before bread before raw
meat before `rotten_flesh`. Golden carrots (the best Nether food) need carrots and gold, which
this bot neither farms nor mines — cooked meat is the best within reach. If no
work is found at all, `wait` holds the position at the base instead of running away.

**Nether preparation (post-portal step, implemented):** The same lull now also re-equips the
remaining Nether gear (`_all_nether_prep_steps`, a single place for offering **and**
auto-stop): a **bow with ≥ 16 arrows** (`craft_bow`; 3 string from `hunt_string` + sticks, arrows
from flint + stick + feather — `get_flint` is reopened for that after the flint and steel, `hunt_feather`
collects feathers), a **gold helmet** (`mine_gold` → `smelt_gold` → `craft_gold_helmet`; one piece of
golden armour keeps piglins neutral, worn only in the Nether) and a **deep cobblestone reserve**
(`NETHER_COBBLE = 64`). The bow is the actual gap: the **server combat fires it itself**
(`combat.bow`), the bot only has to carry it. `tech_tree_complete` now additionally requires that
no prep step is still open **and the food reserve is banked** (`_food_reserve_met` → the
64 Nether units from `_food_goal`; live run13 otherwise ended the run with only 4× cooked_mutton) —
the run thus prepares itself fully instead of ending with the bed on the first night. Each
step drops out of the list as soon as it gets no further (no spawn, no ore loaded), and
the food demand drops away as soon as no food animal is within the **daylight hunt reach**
and no zombie is within melee reach (24 m) — so an unreachable item never blocks the
run. The auto-stop also insists the gate bed lies **at the base**
(`_bed_at_base`): a leftover bed merely *loaded* satisfies `_bed_placed`, so without the
clause the run could stop with the bed still in an earlier run's shaft (live, run19). The
same condition guards the bed work: `sleep` at the base needs a bed *standing* there (`place_bed`
lays a carried one first), `take_bed` brings a leftover one home, and a carried bed walks home in
daylight too, because `place_bed` only lays it after dark (live, run19b: the bed was taken up by
day, no `go_home` came, and it was never laid). The
deep dig for gold is not on offer at night (`_nether_prep_steps` filters it), but stays
open for the auto-stop.

**Nether safety gates:** `sleep`, `place_bed` and `fill_bucket` are restricted to the overworld via `_overworld(gs)` — a bed in the Nether explodes, water evaporates there.

The design, a reconciliation of the optimal Nether inventory with what the bot can do (fire resistance as a
chicken-and-egg, elytra/totem/shulker out of reach; decided: the gold helmet replaces the
diamond helmet on entry), is in [`docs/nether-prep.md`](docs/nether-prep.md).

Architecture + zero-shot findings: [`docs/architecture.md`](docs/architecture.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env        # enter OPENROUTER_API_KEY (only for System-2 escalation)
```

`.env` is loaded automatically on every CLI run (real environment variables take precedence)
and is gitignored. The status can be checked with `probe`.

Prerequisite: the mine-ai MCP server runs at `http://localhost:25575/mcp`.

## Usage

```bash
.venv/bin/python -m mlaya.cli probe      # health + tools/list
.venv/bin/python -m mlaya.cli decide     # read state + decide once (dry run)
.venv/bin/python -m mlaya.cli policy     # show the server survival policy
.venv/bin/python -m mlaya.cli policy --apply              # set the `base` stage
.venv/bin/python -m mlaya.cli policy --apply --preset hunt  # set the `hunt` stage
.venv/bin/python -m mlaya.cli run -n 10  # decision loop (executes real actions)
.venv/bin/python -m mlaya.cli run --ignore-goal  # keep running past the tech-tree goal
```

Configuration: [`config/settings.toml`](config/settings.toml).
Every decision is logged to `runs/decisions.jsonl`.

The `run` loop **stops by itself** once the tech-tree goal stands: diamond pickaxe +
diamond sword, all four diamond armour pieces *worn*, the shield in the off-hand, **the
Nether portal lit and a bed placed at the homebase**
(`actions.tech_tree_complete`). It then reports `"stop": "goal"`; without a goal it ends at the
iteration limit with `"stop": "limit"`. Turning it off (e.g. for dataset collection past the
goal) is done via `[loop] stop_when_complete = false` or `run --ignore-goal`.

## M3: logs → dataset → fine-tuning

```bash
# 1) Build a Laya training dataset from the run log (train + held-out eval)
.venv/bin/python -m mlaya.cli dataset            # -> data/train.jsonl + data/eval.jsonl
.venv/bin/python -m mlaya.cli dataset --include-failed   # also keep ticks without progress
.venv/bin/python -m mlaya.cli dataset --no-danger        # without `in_danger` rows (see below)

# 2) Check the dataset format (loads no weights), then retrain the head (RLCD)
.venv/bin/python -m mlaya.cli train --dry-run
.venv/bin/python -m mlaya.cli train              # -> runs/checkpoints/minecraft-<time>

# 3) Register the checkpoint under `english` and run the loop with it
export LAYA_EXTRA_MODELS='{"english": "'"$PWD"'/runs/checkpoints/minecraft-<time>"}'
.venv/bin/python -m mlaya.cli run -n 10
```

`LAYA_EXTRA_MODELS` is a JSON object `{name: source}`; a name that matches a built-in checkpoint
(`english`) **re-points it** — exactly how the fine-tune is served without a code change.
The reward is the verifier: only ticks whose action demonstrably made progress become
positive examples (`--include-failed` for pure imitation). Tests:
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`.

**Result of the fresh-world run** (48 ticks, tech tree from nothing → `stop: "goal"`):
35 dataset rows (26 train / 9 held-out). The held-out comparison shows **no** accuracy gain
(1.000 → 1.000) — the split is in-distribution, every row is a state Laya solved in the
run itself; the fine-tune shows itself in **loss 0.734 → 0.532**, **ECE 0.501 → 0.392** and
**Brier 0.412 → 0.284** (better-calibrated confidence).

**Result over a whole held-out run (OOD, 2026-10-14):** Corpus = all from-zero runs
except the stall runs and the over-trigger run `run7b`, **252 train rows** (`in_danger=yes`
28, only ticks with progress). Held-out is the **complete** run `run7d` (**61 items**, no
state in training — the state dedup was empty too). The zero-shot accuracy on this
out-of-distribution split is **0.934** — so **no** 1.0 ceiling anymore. The fine-tune (4 epochs,
`rlcd`, encoder frozen, MPS) leaves the accuracy unchanged (the labels are the logged
gate choice; imitation cannot fix their errors) and improves the calibration:
**loss 0.474 → 0.372**, **ECE 0.250 → 0.132**, **Brier 0.261 → 0.187**, `mean_confidence`
0.691 → 0.891.

**Danger rows:** An `in_danger = yes` tick ends as a **`defer`** (no action) — exactly there
sits the real `yes` case. The builder now takes such defer ticks along as a pure `in_danger` row.
Conversely a **cap** cuts a `yes` for a merely loaded mob: a state with
health ≥ 15 and the nearest hostile known beyond 8 m (`DANGER_SAFE_M`) is labelled as `no`.
This cap is **deliberately wider** than the runtime band `brain.danger_matters` (since run13
only contact range `DANGER_DEFER_M = 4 m`): the loop must not stand still for a mid-distance mob, but
in the same undecided band between the lower bound (≤ 4 m) and the upper bound (> 8 m)
the logged gate answer should stand, instead of forcing `no`. As long as no
`yes` case is in the log yet, build with `--no-danger`, otherwise the training pulls the shared
`choice` head towards "always no".

## M4: parameter layer (target selection)

`src/mlaya/targets.py` builds a candidate list of
**standing cells** (`above: air`) from the `view_blocks` hits, `brain.choose_target` lets Laya choose among them (with more than
`TARGET_K = 6` candidates via `predict_shortlist`), and `actions.resolve(label, gs, target)`
binds the target to the tool arguments. The first beneficiary is `explore` → `navigate` with a
real `(x, y, z)` position instead of a fixed `x + 16` offset. Also new is that the
`view_blocks` query additionally searches for `grass_block`/`sand`/`gravel`/`dirt` and
`perception.BlockHit` keeps the `above`/`below` annotation of the hits.
