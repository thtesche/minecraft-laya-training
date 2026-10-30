# Architecture — minecraft-laya-training

A Minecraft agent whose **System-1 decisions come from [Laya](https://github.com/NandhaKishorM/laya)**
(a non-autoregressive typed decision engine, `choice`/`score`/`noul`, a single forward pass).
Laya produces **no** text and **no** tool arguments — therefore "playing" is broken down into
**decisions among already-valid options**, and a separate **executor** talks to the
mine-ai-MCP.

## Decisions (fixed)

| Axis | Choice |
|---|---|
| Language | Python ≥3.10, virtual environment via `uv` |
| System 1 | Laya **in-process** (`laya.Router`), **zero-shot**, checkpoint `english` |
| System 2 | Strata (local llama.cpp) `qwen3.8-flash-next-q2_0` (`max_tokens=4096`) |
| Orchestration | **LangGraph** (async, `MemorySaver`) |
| MCP | Streamable HTTP `http://localhost:25575/mcp` |
| Knowledge | zero-shot, in the `criteria` texts; no knowledge file |

### Why the `english` checkpoint

The state and all questions are English. In a direct comparison, `english` reads the state clearly
better than `multilingual` (e.g. binary gate: `SAFE→safe 0.96`, `DANGER→in_danger 0.92` versus
`0.89/0.76`). Dict state (JSON) dropped noticeably; **compact string state** wins.

## Zero-shot finding (important)

Systematic tests before building produced:

- **Multi-class `choice` is unreliable zero-shot** (first-option bias): a 3-way `macro`
  (`emergency`/`survival`/`progression`) chose `emergency` in the safe state; even 5-way
  progression was noisy. `noul` is also poorly calibrated (a long state drifts upward).
- **Binary `choice` works**: "Is the bot in danger?" hit `SAFE→no`, `DANGER→yes`.
- **A long state is truncated by Laya to the first window** → state must stay short.

Consequence: The code **filters the options by their preconditions** (tech tree), and Laya
only decides among these valid options. This means there are no dead ends, and every
decision is a real choice among feasible actions. That is exactly the path that
later head fine-tuning (M3) improves.

## Layers

```
[1] MCPExecutor   (src/mlaya/mcp.py)
    Streamable HTTP client + gating state machine:
    submission_id factory, rationale guard (<=200), action-ID parsing,
    wait_for_action (initial_wait_ms, then poll_ms x max_polls),
    ACTION_BUSY / RESULT_NOT_RETRIEVED recovery: the blocking action ID is read from the
    structured data (`unretrievedActionId` / `activeActionId`; the text only names
    the parameter name), its result fetched and the own submission repeated - the
    rejection means that the own action never started. Reads are ungated.
[2] Perception    (src/mlaya/perception.py)
    view_status + read_recent_events + view_blocks (all free) -> GameState
    -> compact NL state (`narrative()`); resources (logs/stone/iron_ore/coal_ore/
    diamond_ore) with distance within a radius of 64 blocks (resource and ground scan are
    two `view_blocks` groups, because `find` allows at most 8 block names); time of day/
    sleep state, mob types (incl. passive) and unread notifications (drain)
    flow into the state.
[3] Brain (S1)    (src/mlaya/brain.py)
    Laya router, ONE predict() pass:
      in_danger  (binary gate)      -> defer to the survival policy; `danger_matters` limits it
                                       (hostile ≤ 4 m contact or health < 15, otherwise the loop keeps working)
      danger     (score, logging only)
      intent     (choice)            -> among the filtered candidates
[4] Actions       (src/mlaya/actions/ package)
    candidates(gs): valid labels by preconditions (tech-tree order)
    resolve(label, gs): label -> real MCP tool + arguments
    Cascade: get_wood -> craft_pickaxe -> craft_table -> mine_stone ->
    craft_stone_pickaxe -> craft_furnace -> mine_iron -> mine_coal ->
    smelt_iron -> craft_iron_pickaxe -> craft_shield -> equip_shield ->
    craft_iron_sword -> craft_iron_<slot> -> equip_armor (iron combat kit, BEFORE the
    diamond run) -> mine_diamond -> craft_diamond_pickaxe
    (-> craft_diamond_sword) -> craft_diamond_<slot> (chestplate/leggings/boots/helmet)
    -> equip_armor; bed chain (hunt_wool/hunt_string ->
    craft_bed -> sleep at night) and eat_food when hungry; home base
    (surface_up -> establish_base -> go_home, craft_chest/place_chest/stow_surplus) and
    Nether chain (craft_bucket -> fill_bucket -> mine_obsidian -> get_flint ->
    craft_flint_and_steel -> clear_portal -> build_portal -> light_portal -> place_bed); explore
    only as the last fallback (when nothing productive is possible).
[5] Verify        (src/mlaya/verify.py)
    read view_status again; target/actual delta (inventory, position, health, hunger,
    phase/sleep); progress per action
[6] Memory        (src/mlaya/memory.py)
    runs/decisions.jsonl (every decision + outcome)
[7] System 2      (brain.escalate_to_system2)
    structural trigger (label not resolvable) -> Strata chooses a valid alternative
[8] Survival      (src/mlaya/survival.py)
    sets the server survival policy in two stages (`base`/`hunt`) at start and again
    after every death; `read_policy` reads revision/`effective`/`overrides`
```

## LangGraph

```
START -> perceive -> decide -> guard --+-> execute -> verify -+-> decide (loop)
                        |              |                       +-> END (goal reached
                        |              +-> defer ---------------+    or max_iterations)
                        |                                          +-> END
                   (System-2 escalation on a non-resolvable label)
```

`perceive`/`decide`/`guard`/`execute`/`defer`/`verify` are async nodes in `src/mlaya/graph.py`.
`execute` calls `MCPExecutor.call` and blocks until settle (~40–110 s); the `MemorySaver`
checkpointer keeps the run resumable. `defer` skips the foreground call when the server
survival reflex is responsible (danger / no valid action), and re-perceives.

## Start of this codebase (M1) — reached

**Criterion:** Laya chooses an action from a real `view_status`, the executor performs **one**
real MCP call incl. gating/retrieval, the verifier reads `view_status` again; N iterations
without a crash. Every decision lands in `runs/decisions.jsonl`.

Verified run (`run -n 4`): `craft_pickaxe` → `craft_item` (succeeded, `+wooden_pickaxe`),
then `mine_stone` → `collect_block` (succeeded, `+8 cobblestone`) — all gated, no crash.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m mlaya.cli probe     # health + tools/list
.venv/bin/python -m mlaya.cli decide    # perceive + decide (dry run)
.venv/bin/python -m mlaya.cli run -n 10 # real loop (mutates the world)
```

## M2 — reached

**Criterion:** include nearby resources in the state (`view_blocks`) and drive the progression
through stone tools and furnace up to the **iron pickaxe** (since the diamond tier up to the
**diamond pickaxe**).

Verified run against the real world:

```
[0] craft_table          craft_item    +crafting_table, -1 acacia_log
[1] craft_stone_pickaxe  craft_item    +stone_pickaxe, -3 cobblestone, -2 stick
[2] mine_iron            collect_block +3 raw_iron (+cobblestone, +dirt)
[3] mine_coal            collect_block +3 coal
[4] craft_furnace        craft_item    +furnace, -8 cobblestone
[5] smelt_iron           smelt_item    -1 coal, -3 raw_iron, +3 iron_ingot
[6] craft_iron_pickaxe   craft_item    +iron_pickaxe, -3 iron_ingot, -2 acacia_planks
```

The state now shows e.g.:
`nearby resources: stone at 1m, coal_ore at 11m, iron_ore at 14m, logs at 16m.`

Important implementation details:

- 3×3 recipes (stone pickaxe, furnace, iron pickaxe, diamond pickaxe/sword, the four
  diamond armour pieces) run via `temporary_workstation: true` and need a **carried**
  `crafting_table` (`craft_table` beforehand); smelting needs a carried `furnace`.
  Armour needs **no sticks**, only the crafting table.
- `mine_iron`/`mine_coal`/`mine_diamond` are only offered when ore is actually
  loaded within the radius (otherwise the cascade falls back to `explore`). `mine_diamond` additionally
  needs tier ≥ iron **and the completed iron combat kit** (`_combat_ready`: iron sword + all four
  iron pieces worn + shield in the off-hand — see the "Combat kit" section). The diamond goals come
  from **one** source, `_next_diamond_item(gs)`
  (pickaxe 3 → sword 2, only once the diamond pickaxe stands → armour in the order
  `DIAMOND_ARMOR` = chestplate 8, leggings 7, boots 4, helmet 5, i.e. **protection per diamond** first).
  Only **the next missing piece** is ever active: `mine_diamond` with
  `count = cost − diamonds present`, otherwise `craft_diamond_<slot>`. **No upper limit on the
  tier** (live-found deadlock, see below): a condition `tier < diamond` would have forbidden that the
  diamond pickaxe itself mines the two diamonds for the sword (and later the 24 for the armour)
  — both would have been unreachable.
- A crafted armour piece initially sits in the inventory; `equip_armor` sends it to the body with **one**
  `equip` batch (without `destination` — armour finds its own slot). A piece
  counts as "present" whether **worn** or in the inventory, so that nothing is crafted twice.
  Attention: `situation.tools.armour` is only a **convenience list of the carried** armour
  per class — it says nothing about whether a piece is equipped. The **worn** set is read from
  the inventory `location` (`head`/`torso`/`legs`/`feet`; an equipped piece leaves
  `main`/`hotbar`). Previously the parser misinterpreted `tools.armour` as "worn", whereby
  `equip_armor` was suppressed.
- **Shield** (`craft_shield` → `equip_shield`): 6 planks + 1 iron ingot (3×3, needs the
  crafting table). It only blocks in the **off-hand**, but a craft only lands in the backpack — only an
  `equip` with `destination: "off-hand"` puts it there. The server reports the off-hand stack with
  `location: "off-hand"`; `GameState.offhand` reads that (the armour slots do the same), and
  `verify.diff`/`progressed` record `offhand_delta`.
- **Iron combat kit (before the diamond run):** `craft_iron_sword` (2 ingots + stick, 3×3) and
  `craft_iron_<slot>` (chestplate 8, leggings 7, boots 4, helmet 5 — `IRON_ARMOR`, as with the
  diamond set **protection per ingot** first; always only the next missing piece via
  `_next_iron_armor_item`), then `equip_armor`. `equip_armor` equips iron **and** diamond and
  only swaps against a **better** piece (`_armor_to_wear`), otherwise the iron piece
  displaced by the diamond piece would immediately be put back over it in the backpack. `mine_diamond` is gated behind
  `_combat_ready` (iron sword **or better**, shield, all four slots filled with iron **or
  better**). **Reason (live, run12):** The bot descended to the
  diamonds (y<16) with only an iron pickaxe, met a zombie and creeper and died; the run to the death site lost the race against
  the item despawn. The iron tier is the only sword/armour tier before diamond that the tree
  builds. **Iron consumption:** `_iron_goal` is the only source for "how much iron is still needed". It is
  `3` for the (replacement) pickaxe, otherwise `_combat_iron_missing` (shield 1 + sword 2 + missing
  armour pieces), and with a completed diamond kit additionally `_portal_iron_goal` (bucket, flint and steel).
  `mine_iron` **and** `smelt_iron` (ingots < goal) share it; `mine_iron` is capped per action at
  `IRON_MINE_BATCH = 8`, so that an armour goal does not become a single unlimited mining
  operation. All iron crafts are gated behind the iron pickaxe, so that they do not steal
  the first three ingots.
- `explore` is deliberately the **last fallback** and only in `candidates` when no
  productive step is possible — otherwise it dilutes the (zero-shot noisy) choice. Exception:
  in the lull after lighting the portal (see eating/auto-stop) `wait` stands in its place —
  the bot holds at the base instead of running away at dusk (live: 5 pathfinder timeouts).
- **Environment blocker (live-found & fixed):** In a 1×1 shaft after ore mining
  `craft_item`/`smelt_item` answers with `[WORKSTATION_PLACEMENT_FAILED] … no_cell` (no
  cell for the carried crafting table). In the loop that would be an **endless retry**. The
  `graph.Agent` remembers the blocked label (`blocked_by_surroundings`) and inserts an
  **unblock**: first `make_room` (`build_structure` with `block_name: "air"` digs four
  cells free at bot height — cheap, directly at the ore), and only as a second instance an
  `explore` navigation to open terrain (the same M4 target pool). Then the craft takes effect.
  Live: dug deep enough, reproduced `craft_furnace` `no_cell`; `make_room` (4 cells)
  → the same `craft_furnace` went through (`furnace` 3 → 4). Pinned in `tests/test_graph.py`
  and `tests/test_actions.py::MakeRoomTest`.
  **Extended (armour run):** `sleep` failed in the same 1×1 shaft three times with
  `[BED_PLACEMENT_FAILED] … no … flat ground` (three wasted ticks, the bed needs two
  supported cells). `BLOCKED_CODES` now also contains `BED_PLACEMENT_FAILED`, the unblock
  first digs a pocket (`make_room`) into which the bed fits.
  **Fix (shield run):** The planned escalation "first `make_room`, then `explore`" was
  **unreachable**: `_verify` cleared the blocked label and zeroed `_unblock_attempts` already
  when the *unblock* succeeded — so it stayed at `sleep` → `make_room` → `sleep` → … (the run
  burned 12 of 16 iterations). Now the label is only cleared when **the blocked
  label itself** succeeds (`_note_block`); a successful unblock leaves counter and label
  in place, so that the second attempt really takes `explore`. Pinned in
  `tests/test_graph.py::GuardUnblockTest`.
  **Addendum:** The unblock only fires when a label is really blocked
  (`_blocked_label is not None`). A pure danger deferral carries `action_label=None` and must not
  re-arm the ladder — otherwise an unblockable tick would push a pointless `make_room` afterwards.
  **Addendum (run12):** `BLOCKED_CODES` also contains `NO_REACHABLE_MATCHING_TARGETS` — the server
  reports that when it *found* the target blocks but has none *reachable* (`contract.ts`:
  "Every loaded X block was found but none could be reached", e.g. a tree it refuses because of
  liquid isolation). In place it repeats endlessly (live: the bot stood under 56
  trees and answered every tick with `get_wood`); the unblock ladder (pocket, then
  open terrain) is exactly the change such a refusal needs. Pinned in
  `tests/test_graph.py::BlockedBySurroundingsTest`.
- **Observation:** A 2×2 craft (`craft_pickaxe`) put the carried crafting table into the world once
  instead of keeping it. The chain heals itself because `craft_table` is offered again
  as long as no crafting table is carried.
- **Danger: a tool breaks *during* a long mining run (live-found).** The
  `replace_tool` reaction only fires at `LOW_DURABILITY = 0.15`, and only *between* two ticks.
  A single `collect_block` digging a long tunnel to the diamond ore can consume more
  durability than is left until the next check — in the diamond run so the
  **iron pickaxe** broke in the middle of `mine_diamond` (`status: partial`, 2 diamonds, then tier 2).
  Afterwards `mine_diamond` is no longer offered (tier < iron), and the chain must run the
  iron rebuild completely (`mine_iron → mine_coal → smelt_iron → craft_iron_pickaxe`).
  That works (live run through), but is expensive — the same "self-healing but expensive"
  class as the crafting-table observation above.
  **Hardening (three parts):**
  1. `replace_tool` now additionally triggers at an absolute remaining durability
     ≤ `MINING_HEADROOM = 80` uses, so that the fresh replacement of the same tier is already carried
     *before* a long digging run can use it up; if the active pickaxe then breaks, the
     server automatically pushes the replacement in and the tier is preserved. An
     `_has_spare_pickaxe` guard prevents the reserve from crafting the same replacement multiple times.
  2. `ORE_MIN_TIER` makes the game rule explicit (cobblestone: any pickaxe, coal: any pickaxe, iron:
     `STONE_PICK_TIER`, diamond: `IRON_PICK_TIER`; gold counts as tier 1 and therefore destroys iron
     as well as diamond ore). `_can_mine(gs, block)` additionally checks that a break *within* the dig
     does not fall below the minimum tier — otherwise the server pulls the pushed-in lower
     tier in and destroys the remaining ore without reward. Stone and coal are
     excepted: they are cheap, and a prohibition would block the bootstrap (a worn
     wooden pickaxe must still be allowed to mine the cobblestone for the stone replacement). `candidates` **and**
     `resolve` use the same check, `mine_stone`/`mine_iron`/`mine_coal`/`mine_diamond` are
     thereby aligned.
  3. The old upper limit `tier < IRON_PICK_TIER` for `mine_iron` dropped away: since a stone pickaxe
     already mines iron (wood destroys it), a worn iron pickaxe can now mine the iron for
     its own replacement (`iron_wanted` additionally triggers at
     `_pickaxe_due_for_spare`). A *fresh* iron pickaxe still does not mine iron.
  The diamond pickaxe (1561 durability) makes the risk small afterwards.

## M2b — reached: bed/respawn, sleeping at night, notifications

**Criterion:** Build a bed, sleep in it at night (skips the night **and** sets the
respawn point), and actually read the unread notifications and lift them into the state.

- **Bed chain** in `actions.py`: `hunt_wool` (sheep) or `hunt_string` (spider) →
  `craft_bed` (`white_bed`, 3×3 → `temporary_workstation: true`) → `sleep`.
  `sleep` is only offered at night and with a carried bed; `sleep` finds/places the bed
  itself (`view_crafting_requirements` confirms: `white_bed` = 3 `white_wool` + 3 planks,
  `white_wool` = 4 `string`).
- **`white_wool` is craftable from `string`** — the server computes the chain recursively
  (`craft_item white_bed` first makes wool from `string`, then the bed).
- **Validation (live):** the full chain ran through: `craft_table` (during the day) → roamed during the day via
  `navigate` to a mob group, until a sheep was loaded within simulation range →
  `hunt_wool` (`collect_mob_drop sheep/white_wool ×3`, yielded **4 wool**) → `craft_bed` (`white_bed`)
  → waited for night → `sleep` **`succeeded`**, phase **night → day** (night skipped).
  `sleep` **places the bed into the world** (the inventory loses it; the bed afterwards stood at
  (−324, 124, −6) at the feet). Two edge cases from the live run:
  - **`[HUNT_TARGET_NOT_LOADED]`** — the mob list reaches up to ~150 m (`nearest.position` present),
    but only **loaded** (simulated) mobs are huntable (`rangeBlocks: 16`); so you must first
    walk over until the sheep is loaded.
  - **Body ownership:** at night the server `hostile_reflex` (or the **hunger** reflex) temporarily holds
    the body ("No new action started; body owner: …"), so a foreground craft has to wait.
    `craft_bed` came back as `cancelled` (hunger reflex at hunger 13) — the bed was still
    in the inventory afterwards, i.e. the craft had already completed before the interruption.
- **Safety gate for hostile hunting:** A spider run led the bot into a
  zombie/skeleton/creeper field and cost it an iron pickaxe. `hunt_string` is
  therefore only allowed when **exactly one** spider ≤10 m, **no further hostile ≤16 m** and
  health ≥16 (`_safe_spider_hunt`); otherwise it stays out of it.
- **Notifications:** `read_recent_events` is now read with `limit: 50` (drain of the
  event/notification backlog) instead of 5; the newest `events` summaries **and** the
  `notifications.recentPreview` are in the state (`narrative()`: "unread notifications (N): …").
  (In M2d replaced by a full backlog drain, see below.)

## M2c — reached: survival policy & eating

**Criterion:** Set the server survival policy explicitly (instead of falling back to defaults) and add
a **food acquisition** that fits "actually survive".

- **Finding:** `eat_food` existed, but there was **no** acquisition — in a fresh run
  it could practically never fire. Worse: the survival policy was **never** set. The server
  ran on default: `food.raw.allow: "emergency_only"`, `hunger_at_most: 6`, `health_below: 10` —
  the own hunger reflex ate raw only at hunger ≤ 6 **and** health < 10.
- **`survival.py`** holds the policy as **two stages** (presets), along the documented
  enums:
  - **`base`** (traversal/progression): `navigation.scaffold_blocks` incl. deepslate/tuff,
    `hostile_avoidance_multiplier: 3`, the combat window
    (`engagement: respond_to_threats`, `engage_min_health 14`, `critical_health 10`,
    `recover_to_health 16`, `retreat: true`, `enderman_wait_ticks 40`, timeouts) and
    `food.raw.allow: "always"` with `hunger_at_most: 20`.
    **Tuned (2026-10-13):** previously `17/14/18`. The server description names `engage_min_health`
    explicitly as the threshold for a **pursuing** action ("immediate contact defence remains
    separate") and `critical_health` the emergency threshold **before** contact. Live the bot fell
    just under the 17 pursuit threshold at 16.5 HP, avoided a single spider instead of
    killing it and bled below the old emergency threshold 14 (hp 16.5 → 0.3, "hide, walled 9").
    The window `14/10` lets it fight at 14–16 HP and lifts the emergency protective reaction
    for < 10. `recover_to_health 16` keeps healing between fights conservative.
  - **`hunt`** (combat stage): the same base with **`engage_min_health 0`, `critical_health 0`,
    `retreat: false`** — otherwise the combat floor blocks an explicitly requested hunt with
    `[HUNT_CAPABILITY_BLOCKED]`. The zero stage is deliberately **temporary**: with it the
    hostile reflex holds the body and blocks navigation, hence straight back to `base` after.
  - The third stage is **auto-eating**: `food.raw.allow` (`emergency_only` = server default,
    which only eats at hunger ≤ 6 **and** health < 10, `always` = here) with the thresholds
    `hunger_at_most`/`health_below`.
- Wiring: `apply_policy` at the **start** of `run` (`base`); the graph reads `last_death_age_s`
  and sets `base` again after every death (the server resets to defaults on death/dimension change).
  The `execute` node switches to `hunt` for a **hostile** hunt (`is_hostile_hunt`) and
  back to `base` in the `finally`. `mlaya.cli policy [--apply] [--preset base|hunt]`.
- **Eating** in `actions.py`: `hunt_food` acquires the **minimum stock** (`FOOD_RESERVE = 16`;
  nether goal 64, see lull) — next passive mob (`FOOD_MOBS = cow/pig/chicken/sheep/rabbit`) **within
  a daylight-dependent range** (`FOOD_HUNT_RANGE_DAY = 64` / `FOOD_HUNT_RANGE_NIGHT = 16`:
  at night only what is already near) or, as a fallback, a **single zombie** (`rotten_flesh`,
  gated: ≤2 hostiles ≤16 m and all zombies, health ≥18). `cook_food` fries raw meat
  (`smelt_item` in the carried furnace). `hunt_food` is offered as long as the stock is < `_food_goal`
  **and** not currently eating ("feed to 20 first, then bank"); the action is marked with
  `allow_without_shield: true`. Eating happens via the server reflex (raw allowed), `eat_food`
  the manual lookup remains.
  **Two thresholds (run14, live):** `eat_food` triggers at hunger < `EAT_HUNGER = 14` (the old
  survival threshold) **or** on a wound it cannot heal: natural regeneration
  requires hunger ≥ `REGEN_HUNGER = 18`, so a bot between 14 and 17 neither ate nor healed and
  simply bled out (live: 17 ticks at 2.7 HP and hunger 17, then killed by the mob). It eats
  only at `health < HEALTH_MAX = 20`, so an uninjured bot at hunger 17 keeps its reserve.
  **Lull follow-up (`_nether_idle`, live-found):** As soon as the diamond kit stands **and** the gate
  frame stands, only the prep (bow, gold helmet, deep food and cobble) and the gate bed are left
  open — the bot fell into pure `explore` during the day (live in
  the from-zero run: 25 idle ticks, of which 5 × `[NAVIGATION_STOPPED]` timeouts). This lull is now used for
  nether preparation: `_food_goal` raises the reserve target to `NETHER_FOOD_RESERVE = 64` (one
  full shulker; live the bot had 17 edibles). The usual rule of thumb for the Nether is **64–128
  units** of high-saturation food, because hunger falls faster when sprinting and fighting and death
  costs the whole inventory; a shulker of fried meat is only one inventory slot, so cheap.
  **The reserve is part of the auto-stop:** `tech_tree_complete` now requires `_food_reserve_met`
  (the 64 units from `_food_goal`), and the requirement drops — like a prep step — as soon as no
  food source is in the **hunt's walking reach** (`_food_source_in_reach`: a passive animal within
  the daylight range, 64 m, or a zombie within melee reach, 24 m), so that a world with nothing
  reachable never blocks the run. Distance-bounded on purpose, unlike the "is it loaded anywhere?"
  test of the Nether prep steps: a hunt must *close the distance* to a moving target, so a source
  merely loaded far away cannot hold the gate open with nothing the bot can do. The stricter
  "is any food loaded?" test plus a day reach widened to 128 m were tried live (run19b) and
  reverted: the far hunt walked the bot 150–200 m from the base, where the base-range latches
  re-open the portal chain, and it stranded in water for 76 ticks. The auto-stop also insists the
  gate bed lies **at the base** (`_bed_at_base`): a leftover bed merely *loaded* satisfies
  `_bed_placed`, so without the clause the run could stop with the bed still in an earlier run's
  shaft (live, run19). The same clause governs `sleep` at the base once the gate is lit (a bed
  merely loaded sent the server to the leftover bed and pinned the respawn point off the gate —
  live, run19b it25: nine blocks below the base) and `place_bed`, which fills a base without a bed.
  Live, run13 would otherwise have ended with only 4× `cooked_mutton`.
  `cook_food` may work with **any** fuel in the lull (`_furnace_fuel`: coal/charcoal
  first, else planks/logs — one log fries exactly one item). It eats by **saturation**
  (`FOOD_SATURATION`): fried meat before bread before raw meat before `rotten_flesh`. Golden
  carrots (14.4 saturation — the best food in the game) need carrots and gold, which this bot
  neither farms nor mines, and are therefore out of reach. Outside the lull it stays with the
  old behaviour (only coal), so that frying never steals wood the chain still needs. If no
  work is found at all, `wait` (see below) holds position. Tested in
  `tests/test_actions.py::FoodChainTest`.
- **Danger (live, now defused):** `hunt_food` used to choose the **nearest** mob
  regardless of distance, and `collect_mob_drop` roams there. A nightly `run` thus wandered
  **159–190 m** and was **killed twice** (among others "slain by Enderman"), which cost **the complete
  inventory** — including the just-crafted iron pickaxe. `hunt_string` also failed
  three times, once fatally. The `in_danger` gate said **`no`** in these ticks (health 10.5, skeleton
  9.6 m) — the zero-shot gate is poorly calibrated for danger (known finding, see above). The
  server policy only fights from health ≥ 17 (`respond_to_threats`), so it does not intervene below.
  **Fix:** `hunt_food` is now bound to a **daylight/distance bound**; at night only
  what is already in reach is hunted. Live cross-checked with the same mobs (35/38/75/103 m):
  **at night** (`range=16`) → `_food_target = None`, `hunt_food` missing from `candidates`; **during the day**
  (`range=64`) → `hunt_food` offered and executed (`collect_mob_drop sheep/mutton ×6` → 6 mutton +
  4 wool). Residual risk: during the day longer paths up to 64 m remain possible — deliberate, because the day path
  is cheap.

## M2d — reached: reacting to event/notification types

**Criterion:** not only lift events into the state but react to their **types**
("durability low" → craft a replacement tool, "target died" → harvest drops, death → recovery).

- **Finding:** `view_status` already delivers the **durability of the active tool**
  (`tools.tools[].durabilityLeft/maximumDurability/item`) and the **death position**
  (`lastDeath.position`/`dimension`). So for durability and death **no** event is needed —
  only for "target died", whose drops lie in the world.
- **Event drain:** `read_recent_events` delivers the **oldest** unread events (max. 100),
  not the newest — with an 841-deep backlog the state therefore only saw old events. `build()`
  now walks until `remainingEventCount == 0` (cap `EVENT_DRAIN_PAGES = 12` per tick), so that every
  tick sees the **fresh** events; `notifications.unreadCount` is 0 afterwards.
- **Reactions** (`actions.py`): three new labels, each **exclusive** for the tick (time-critical),
  `eat_food` keeps running along. Priority: recover > replace > drops.
  - **`replace_tool`** — the active pickaxe needs a replacement at `LOW_DURABILITY = 0.15`
    **or** at absolute remaining durability ≤ `MINING_HEADROOM = 80` uses (the reaction runs
    only *between* ticks, a single long `collect_block` can consume more, see above), and a
    replacement is craftable (`_replacement_pickaxe`: diamond at ≥3 diamonds, else iron, else stone;
    never an unnecessary downgrade). `_has_spare_pickaxe` prevents a second replacement of the same kind
    (otherwise the reserve would craft endlessly if the server leaves a *lower* tier
    lying around). → `craft_item` with `temporary_workstation`.
  - **`collect_drops`** — fresh "target died" event (`gs.drops > 0`, age ≤
    `EVENT_REACT_WINDOW_S = 120` s) → `pick_up_items {radius: 16}`.
  - **`recover_drops`** — death ≤ `RECOVER_WINDOW_S = 300` s, same dimension, ≤ `RECOVER_RANGE =
    320` m → `pick_up_items {recover_death_items: true}` (the server itself runs to the
    death position). Additionally a **deep** death (`death_y < bot_y −
    RECOVER_DIG_DEPTH = 4`) needs a **pickaxe** (`_can_recover`): the way back has to
    dig down, and by hand the bot loses the race against the despawn (live, run12:
    hand-dug from the surface to a death ~60 blocks deeper, lost, afterwards endless loop
    on `get_wood`). Until the pickaxe is crafted, the normal chain (wood → pickaxe) runs first.
    After every death and every dimension change the graph re-arms the policy (`_rearm_policy`).
    **Write-off (run14, live):** The server answers `[NO_MATCHING_ITEMS]` when it reaches the
    death site and its 16 m scan finds nothing more (the drops are despawned/burned)
    — a **definitive** answer, not a transient one. `_can_recover` still offered the path the whole
    window, so that the reaction held the body every tick and the bot stood still at the empty
    spot until the window ran out (and a new death reset it): live 40 + 77
    ticks in two such loops, over half the run, after the diamond kit had been
    lost anyway. `Agent._note_recover` therefore memorizes the death cell (`actions._cell`), as soon as
    a final code comes (`RECOVER_FINAL_CODES` = `NO_MATCHING_ITEMS`, `NO_RECORDED_DEATH`,
    `DEATH_DIMENSION_MISMATCH`, `DEATH_RECORD_STALE`; but not the transient
    `RECOVERY_SITE_UNREACHABLE`), `GameState.recover_written_off` carries it and `_can_recover`
    rejects this cell. A later death elsewhere is a new cell and is run as usual.
- **State:** `GameState.tool_meta` (class → `{tier,item,left,max}`), `death_pos`, `death_dim`,
  `drops`; `narrative()` appends "low durability: …", "… mob(s) the reflex killed dropped items
  nearby" and "gear … can still be recovered". `verify.progressed` checks `replace_tool` (new
  `*_pickaxe`) and `collect_drops`/`recover_drops` (net inventory).
- **Validation (live, all three reactions):**
  - `replace_tool` **end-to-end** — the stone pickaxe checked at 15/131 (`≤ LOW_DURABILITY`)
    worn → `candidates == ["replace_tool"]` → Laya chooses it → `_guard` binds
    `craft_item {items:[{item_name:"stone_pickaxe"}], temporary_workstation:true}` → `succeeded`,
    afterwards 131/131.
  - `collect_drops` — with the aggressive `hunt` policy the server reflex killed a mob (event
    `hostile: target died`) → `drops = 1` → `candidates == ["collect_drops"]` → Laya chooses it →
    `_guard` binds `pick_up_items {radius: 16}`. The execution answered `[NO_MATCHING_ITEMS]`
    (the drops were already unloaded/gone by this tick), so the **decision + binding** is
    documented, not the actual collection.
  - `recover_drops` **end-to-end** — provoked death in place at (−90.6, 71, −14.9), respawn
    (−52.5, 68, 8.5) ≈ 45 m away, 8 s ago → `candidates == ["recover_drops"]` → Laya chooses it →
    `_guard` binds `pick_up_items {recover_death_items: true}` → `succeeded`; the inventory
    (dirt/cobble/logs/planks/sticks + string) came back. The `hunt` policy is thereby after the death
    automatically reset to `base` (`_rearm_policy`).
  - **Observed event wording:** `hostile: target died`, `hostile: contact ended`,
    `MineAI died at <pos> …`, `<tool> broke.` — `_count_drops` keys off `"target died"`.
  - 22 unit tests (`tests/test_actions.py`) additionally pin all bindings/guards as well as the
    bed and food chain deterministically.

## M3 — reached: logs → dataset → RLCD head fine-tuning

**Criterion:** Build a training dataset from the run log, retrain Laya on its own decisions
(head fine-tuning) and register the checkpoint via `LAYA_EXTRA_MODELS`.

- **Dataset** (`src/mlaya/dataset.py`, CLI `mlaya dataset`): `runs/decisions.jsonl` →
  lines `{state, questions, expected}`. `state` is the `narrative()` string that Laya saw in
  operation; `questions` uses **exactly the same** builders as inference
  (`brain.intent_question`, `brain.DANGER_Q`), so that the head is trained on the same rendering
  it later sees.
- **Reward = verifier.** A tick whose action is `progressed` is a positive example for
  exactly that action in that state; a tick without progress has no known good
  alternative candidate and is discarded (`--include-failed` keeps it as pure imitation).
  Only ticks with **more than one** candidate carry an `intent` choice — a single-candidate tick
  is deterministic (the filter already chose it) and teaches nothing. Dedup over
  `(state, candidates)`.
- **Why danger rows:** Laya selects the head via the **question type** (the checkpoint carries a
  `type_emb.`), so `in_danger` and `intent` **share one** `choice` head. A pure
  intent dataset would silently unlearn the gate. Hence one `in_danger` row per
  state: a documented heuristic ("a mob attacks or health very low",
  `DANGER_CLOSE_M = 4`, `DANGER_LOW_HEALTH = 8`) is the **floor**, a logged Laya answer
  only applies where the heuristic is silent. **Tuned (2026-10-13):** previously the logged
  answer took precedence. Live the gate still answered `no` at **0.3 HP with a spider in reach**
  — exactly the miscalibration the training is supposed to fix; now the heuristic wins.
  **Upper bound (2026-10-14):** A state that proves itself safe — health ≥
  `DANGER_SAFE_HEALTH = 15` **and** the nearest hostile **known** beyond `DANGER_SAFE_M = 8`
  m (the same thresholds as the runtime band, see below) — is labelled `no`, even if the
  gate said `yes`. Otherwise the over-firing for a merely loaded mob would be trained in
  (live: run7b `yes` at 9–24 m, health 20). Only a state that proves its own safety
  flips the upper bound; if the hostile distance is unknown (empty `hostiles` list),
  the logged answer still applies.
  `--no-danger` drops the rows. `graph._verify` now logs compact `answers` for this
  (`{qid: choice|score}`), so that future runs provide the true gate label.
- **Split:** stable hash over the state (`--eval-frac`, `--seed`), reproducible; < 4 rows
  ⇒ no eval slice.
- **Training** (`mlaya train`, wrapper around `laya.train.finetune`): `--loss rlcd` (GRPO-like + soft-CE)
  or `soft-ce`, `--train-encoder` (default: heads only, `--freeze-encoder`), option-order augmentation
  for `choice` (`shuffle_options=("choice",)` against position bias), `--dry-run` checks the dataset
  without weights.
- **Registering:** `LAYA_EXTRA_MODELS` is a JSON object `{name: source}`; a name that hits a
  built-in checkpoint **retargets it**. `config.load_settings` parses the variable into
  `laya.extra_models` and `Brain` passes it on as `models=` to the `Router`. Thereby the
  loop loads the fine-tune under `english` **without a code change**.
- **Validation (fresh world run, 48 ticks → goal):** complete tech tree from nothing,
  `stop: "goal"`. Dataset: **35 rows** (26 train / 9 held-out), 17 intent labels across the
  whole chain (`get_wood` … `equip_armor`). Head fine-tuning (4 epochs, `rlcd`, encoder
  frozen) → 842 MB checkpoint. Held-out (9 items): `choice` accuracy **1.000 → 1.000**,
  loss **0.734 → 0.532 (−0.20)**, ECE **0.501 → 0.392**, Brier **0.412 → 0.284**, `mean_confidence`
  0.499 → 0.608. The `Router` loads the checkpoint under `english` without a code change. 24 unit tests
  (`tests/test_dataset.py`, stdlib `unittest`).
- **Finding (accuracy ceiling):** The held-out accuracy is **already 1.0 zero-shot** — the split is
  in-distribution (every row is a state Laya solved itself in the run). An
  accuracy gain is not measurable here; the fine-tune shows in loss, Brier and ECE. For
  a real accuracy comparison it needs an **out-of-distribution** split (states from
  another run), not more in-distribution rows.
- **OOD result (2026-10-14):** dataset across **multiple runs** instead of a single one: 252
  train rows (only ticks with progress, deduplicated across the runs; stall runs and the
  over-firing run `run7b` out, `in_danger=yes` 28). Held-out is a **complete run**
  (`run7d`, 61 items, no state in the training). The zero-shot accuracy on this OOD set is
  **0.934** (4 errors) — i.e. *no* 1.0 ceiling. The fine-tune (4 epochs, `rlcd`, encoder
  frozen, MPS) leaves the accuracy unchanged — the labels are the logged gate choice, and
  imitation cannot fix their errors — but improves the calibration markedly:
  **loss 0.474 → 0.372, ECE 0.250 → 0.132, Brier 0.261 → 0.187**, `mean_confidence` 0.691 → 0.891.
  (So the head becomes confidently correct, not more correct.)
- **Danger ticks now land in the dataset:** An `in_danger=yes` tick leads to a **defer** (no
  `action` record), but the builder only saw `action` → `danger_yes: 0`, the gate would have only
  learned `no`. `graph._defer` now logs `narrative` + `answers`, and the builder turns a
  defer with gate `yes` (or the danger note) into a **pure danger row**. A defer for another
  reason ("no valid action") stays ignored; the state heuristic is deliberately **not**
  used there. Tested in `tests/test_dataset.py` (defer cases). For the fresh run the fix came
  too late — its 10 danger ticks are old `defer` records without `narrative`; the dataset was
  therefore built with `--no-danger` and the gate stays zero-shot.
- **Rejected danger ticks (2026-10-13):** An `action`-tick with `progressed=false` (e.g. the
  body belonged to the hostile reflex) was previously discarded **entirely** — those were exactly the
  toughest states live (hp 0.3–6.5 with a spider). For such a tick the builder now emits a
  **pure danger row** (`in_danger: yes`) whenever the heuristic classifies it as dangerous: no
  usable intent, but an honest gate positive. Tested in
  `tests/test_dataset.py::BuildRowsTest::test_a_refused_danger_tick_becomes_a_yes_row`.

- **Danger band at runtime (2026-10-13, tightened 2026-10-15):** The gate answers from the
  narrative, which can only say "hostile mobs within 16 blocks" — it does not see the distance and reads
  a merely loaded mob as a threat. `yes` defers **unconditionally**, but the server policy had
  nothing to do (hp/hunger full, no contact): run7b stood still tick by tick at diamond depth. Two
  fixes apply: (1) `perception.parse_status` now reads the hostile distance **directly** from
  `nearby.hostiles` (previously from `nearest.distance`, where it is missing → every distance was `None`, the
  narrative without "at Xm"); (2) `brain.danger_matters` bounds the gate: a `yes` passes the tick
  **only** if a hostile is ≤ `DANGER_DEFER_M` m away or health < `DANGER_DEFER_HEALTH = 15`
  (with the in-place releases listed further down: `_recovery_owed`, `_must_eat`, `boxed_in`).
  `DANGER_DEFER_M` was 8 m ("a wider threshold than the training floor"), but run13 showed the
  price: a skeleton/creeper holding 5–8 m (in shot range, never in contact) deferred for ~60
  ticks while the policy was idle — the loop froze in a cave. The threshold is therefore now
  **contact range** (`DANGER_DEFER_M = 4`), equal to the training floor `DANGER_CLOSE_M`;
  the server reflexes protect contact anyway independently, the loop only needs to step aside for real
  bodily danger (contact or low HP). The builder deliberately keeps its **upper bound**
  (`DANGER_SAFE_M = 8` / `DANGER_SAFE_HEALTH = 15`) **wider** than the runtime (its own
  constant instead of the earlier import), so that in the undecided band between the lower bound (≤ 4 m)
  and the upper bound (> 8 m) the logged gate answer stays, instead of forcing `no`.
  The `DANGER_Q` criteria were also sharpened ("within a few blocks and closing in"). Tested
  in `tests/test_brain.py` and `tests/test_perception.py`.
- **Quiet defers hand the tick back (2026-10-22, `e0e7687`):** contact range is still no proof
  of a fight. run86 stood at a fortress blaze spawner with blazes 2–4 m away behind the spawner
  fence; the server's hostile reflex fell silent at 20:25:33 (health and hunger full, `busy`
  false), and the loop deferred **120+ ticks** without moving until the iteration cap.
  `Agent._note_defer` now counts the defers in a row that took **no damage** and saw **no
  survival reflex act** (event summaries starting with `hostile:`, `fire:`, `lava:`, … —
  `graph.SURVIVAL_EVENT_PREFIXES`; eating does not count). Once `gs.defer_quiet` reaches
  `brain.DEFER_QUIET_TICKS = 5`, `danger_matters` lets the loop work again; health below the
  band still defers, and the first hit — also during the loop's own action (`_note_hurt`) —
  resets the count. Tested in `tests/test_brain.py::DangerMattersTest` and
  `tests/test_actions.py::DeferQuietTest`.
- **The band releases what only the loop can do (2026-10-25, run95):** two more deferrals that
  could not end. (1) The bot suffocated in a wall at 9 health, nearest hostile 23 m: the policy
  has no answer to a body inside a block, and the band held the tick for the ten ticks to the
  death while the one useful step, `dig_out`, waited on `boxed_in`. `gs.boxed_in` now releases
  the band whatever the health. (2) At 1 health and hunger 6 it carried two raw chicken; the
  offer was `['eat_food']` for the last 138 ticks, because an edible in the pack makes
  `_cannot_heal` (and so `_recovery_owed`) false, while the server's hunger reflex does not eat
  raw chicken (`FOOD_MOBS_LAST`). `actions._must_eat` now releases the band too, unless a
  hostile is inside `DANGER_DEFER_M`: eating with a mob at 2 m is how the next hit finds the bot.
  Both are the same principle as `_recovery_owed` — the defer is only worth its tick while the
  policy has something to do with it. Tested in `tests/test_brain.py::DangerMattersTest`.
- **Second training round: run79–86 (2026-10-23, `775d0d2`):** three changes to the pipeline.
  (1) **Several logs:** `mlaya dataset --log A --log B …` reads them as one set (dedup across
  them). (2) **Ping-pong filter:** the verifier reads a walk as progress, so the loops of the
  October runs would have been taught as good decisions — run82's 481 `go_home`↔`stow_surplus`
  ticks and run84's `mine_obsidian`↔`go_home` swaps (cobblestone +25/−25). `dataset._loop_ticks`
  drops every window of `LOOP_MIN_TICKS = 6` or more action ticks that repeats one label or
  strictly alternates two with **no item gained net** over the window, and a `stow_surplus`
  counts only when an item actually left the pack (`_deposited`). Effect on run79–86:
  `stow_surplus` 48 → 12 rows, `go_home` 81 → 56. (3) **The base is the served checkpoint:**
  `mlaya train` resolved `[train] base_model` straight through `laya.train.resolve_checkpoint_dir`,
  which ignores `LAYA_EXTRA_MODELS` — every fine-tune therefore started from the Hub's built-in
  `english` and discarded the earlier one. `cli._train_base` now resolves the name through
  `laya.extra_models` exactly as the loop does, and `--base` names a checkpoint explicitly.
  Dataset: 683 rows (512 train / 171 held-out, `data/run79-86-*.jsonl`), 306 `in_danger=yes`;
  dry run 843 valid items, 380 optimizer updates over 4 epochs, base
  `runs/checkpoints/minecraft-20261014-104114`. Tested in `tests/test_dataset.py::LoopFilterTest`
  and `tests/test_cli.py`.
  **Result** (4 epochs, `rlcd`, encoder frozen → `runs/checkpoints/minecraft-20261023-100828`,
  log `runs/train-run79-86.log`), held-out 279 items: accuracy **0.767 → 0.885**, loss **1.224 →
  0.309**, ECE **0.255 → 0.065**, Brier **0.483 → 0.189**; calibration slice accuracy 0.893. Unlike
  the 2026-10-14 round the accuracy moves — but this held-out set is a hash split of the **same**
  runs, not a whole unseen run as `run7d` was, so it is closer to in-distribution and the number
  is an upper bound until a live run confirms it. Served with
  `LAYA_EXTRA_MODELS='{"english": "<repo>/runs/checkpoints/minecraft-20261023-100828"}'`.

## M4 — reached: parameter layer (goal choice over `view_blocks` candidates, `predict_shortlist`)

**Criterion:** A label says *what* (e.g. "explore"), but not *where*. The parameter layer fills
the arguments from the grounded state: instead of a hard-wired offset, Laya picks a
concrete goal from the `view_blocks` candidates, and `predict_shortlist` keeps large candidate sets
within the head budget. It is the same division as in the label layer — **code filters, Laya chooses** —
just one level deeper.

- **Perception** (`src/mlaya/perception.py`): `view_blocks` hits are now kept as `BlockHit`
  (`name`, `x/y/z`, `distance`, `above`, `below`). The server annotates every hit with
  `above`/`below`; `above == "air"` is **real standable ground** — per the run skill the reliable way
  to choose a `navigate` goal. In addition to the resource search, `build()` requests the same
  `view_blocks` query for `GROUND_NAMES` (`grass_block`, `sand`, `gravel`, `dirt`).
- **`src/mlaya/targets.py`**: `Target` (label + description text + `BlockHit`) and
  `ground_targets(gs)`. It keeps only standable ground from `DESTINATION_QUERIES` (floor + `stone`),
  deduplicates per `(x, z)` column, discards terrain **underfoot** (`EXPLORE_MIN_M = 4`) and
  sorts **furthest first** — so an explore pick is a real step outside; if the
  bot is enclosed, the nearest ground is offered instead. The texts stay short (distance +
  compass rose), because all options share a `head_max_len` budget.
- **Brain** (`src/mlaya/brain.py`): `TARGET_QID = "target"`, `target_question(targets)` and
  `choose_target(state, targets)`. Up to `TARGET_K = 6` candidates a normal `predict` runs; a
  larger pool is narrowed via `laya.shortlist.predict_shortlist`, whose `embed_fn` comes from the
  already loaded checkpoint (`embed_fn_from_agent(router.load(model))`, cached once).
  No hit ⇒ fallback to the most distant candidate.
- **Graph/Actions** (`src/mlaya/graph.py`, `src/mlaya/actions/`): `TARGET_LABELS = {"explore"}`. `_decide`
  builds `ground_targets` for such a label and records the choice as `Decision.target` (+
  `target_meta`); `_guard` calls `resolve(label, gs, target)`. `explore` then binds
  `navigate {x, y, z, dig, scaffold}` to the **real** goal position (with `y`, otherwise `navigate`
  guesses over roofs/foliage); without candidates the old `x + 16` fallback remains. The action log now carries
  `target` and `target_meta`.
- **Validation (live):** `view_blocks` returned 8 groups × 4 hits; from these 7 standable candidates,
  after the `min_distance` filter 4 (`grass_block` ~8 m north). `ground_targets` labels them
  correctly (distance + direction), `predict_shortlist` (k=2 forced) narrows 7 → 2, Laya picks
  `grass_block@-42,10`, and `_guard` binds `navigate {x:-42, y:65, z:10, dig:true, scaffold:true}`.
  The path without a shortlist (pool ≤ `TARGET_K`) and the fallback without candidates are checked too.
  13 unit tests (`tests/test_targets.py`, stdlib `unittest`).
- **Caveat 1 (reach):** `view_blocks find` lists the **nearest** hits per query (limit 4).
  The candidates are therefore nearby terrain, not a far goal — the parameter layer chooses *direction*, not
  a journey across the map. True long-range scouting would need `view_blocks box` in a ring or
  `view_frontier` and thus a query of its own.
- **Caveat 2 (frequency):** `explore` is the **last fallback** and fires rarely; the mechanism is
  deliberately generic (`TARGET_LABELS`, `resolve(..., target)`), further labels can adopt it
  later.

## Storage network — reached (2026-10-21, `5135ef0`)

run79 ended in the deadlock this exists to break: the one chest at the base filled up, every
deposit after it answered `[CONTAINER_BATCH_INCOMPLETE]`, the pack had nothing left to shed, and
102 `wait` ticks followed. The base now keeps a **network** of chests:

- **Fullness per cell** (`Agent._note_container`): a deposit the server could not finish marks
  that cell full; a successful deposit or withdraw opens it again.
- **Overflow chests** grow beside the main one (`STORAGE_SITES` candidates, refused cells
  skipped) once every chest of the network is full. `stow_surplus` deposits into the first chest
  with room, `take_stored_food` reaches into whichever network chest the server's database last
  saw holding provisions (`banked_stores` per cell, own cells only).
- **The deposit works where the chest stands (`0d24d78`):** a chest adopted outside the base's
  radius was walked to 481 times and worked never (run82) — `stow_surplus` was only allowed at
  the base anchor and `go_home` walked the bot back on the very surplus the chest was waiting
  for. `at_chest` is now the base **or** the chest lid's reach (`STORAGE_REACH`), and
  `_storage_reachable` silences `go_home`'s surplus clause while a roomy chest is in reach.
- **Phantom cells leave the network (`9f5b62e`, `3f5b908`):** run83 adopted a chest cell that
  the world had given up; four deposits answered `[CONTAINER_NOT_PRESENT] The loaded target block
  is air` and the phantom stayed the target to the end. `Agent._note_container_missing` drops a
  cell — with the provisions believed in it — when the server names a block that is **not** what
  the cell was remembered as (air, or a furnace where a chest was). A standing furnace the server
  merely cannot open (`... is furnace, not a supported container`, 119× in run49) is **kept**:
  forgetting it would re-adopt it with its occupied flag wiped.

## Stall guards

Most October runs were not lost to deaths but to a loop that kept the run busy without
progress. The loop now has one guard per kind of stall; each withholds the label for
`STUCK_COOLDOWN = 15` ticks, after which it returns on its own.

| Stall | Guard | Trigger |
|---|---|---|
| the same refusal again and again | `_track_stuck` | `STUCK_REPEATS = 4` identical failure codes |
| walking far, arriving nowhere | `_track_wander` | ≥ `WANDER_WALK = 80` m walked at a ratio worse than `WANDER_RATIO = 6`, `WANDER_REPEATS = 2`× |
| succeeding at nothing | `_track_idle` | `IDLE_REPEATS = 3` ticks with no change at all (`wait`/`sleep` exempt) |
| a walk off dropped junk that cannot be made | `_note_step_away` | `STEP_AWAY_GIVE_UP = 3` refusals in a row |
| a body the server's reflex will not hand back | `MCPExecutor.call` / `_note_held` | `[mcp] action_wait_ms = 300000` per call, then `[BODY_HELD]`; `BODY_HELD_STOP = 2` of them in a row end the run with `"stop": "body_held"` |
| deferring to a reflex that does nothing | `_note_defer` | `DEFER_QUIET_TICKS = 5` quiet defers (see M3); under the health band only when nothing heals in place (`_recovery_owed`), the loop owes a meal it can eat with nothing in contact range (`_must_eat`), or the body is shut in (`boxed_in`) |
| a death site the walk cannot reach | `_note_recover` | `RECOVER_UNREACHABLE_REPEATS = 3` `[RECOVERY_SITE_UNREACHABLE]` from one site write it off |
| shut in a pocket the pathfinder cannot leave | `_note_boxed` / `dig_out` | `BOXED_REPEATS = 3` `no path found` after ≤ `BOXED_NODES = 200` nodes, closest ≤ 5 m, from one cell — counted at the cell the action *ended* in, and not reset by work done in place |

Four lessons from the runs behind them:

- **Exclusive offers must ask for the withholding themselves.** A rescue or the night `sleep`
  returns from `candidates` before the final `gs.stuck_labels` filter, so a withheld label kept
  the tick anyway (run86: `sleep` dropped at it2, chosen 13 more times). The night offer now checks
  `"sleep" not in gs.stuck_labels` itself (`6d3d164`).
- **Two offers that walk the bot in opposite directions are one predicate.** A walk that
  succeeds every tick is invisible to all three guards above. The fix is always the same: the
  step that walks *away* must not be offered while the step that calls it *back* holds —
  `_storage_reachable` (run82, `go_home`↔`stow_surplus`) and `_shed_owed` (run84, a full pack
  walking `mine_obsidian` 27 blocks down to the lava while `go_home` called it back on the
  surplus, `251df45`).
- **A guard must see both outcomes.** `STEP_AWAY_GIVE_UP` existed but `_note_step_away` sat under
  the success guard, so run81's 470 cliff-top refusals never reached it (`c48da18`).
- **A one-way latch can silence a whole chain.** `_nether_entered` never clears, so after a Nether
  death the relaunch gates offered nothing and every tick fell through to `wait` — run93's ~88
  waits at the base. A latch that marks "has been done" must be re-opened by "is still owed"
  (`_nether_haul_home`, `e812d83`).

`wait` itself is no longer a stall: it navigates to the anchor the bot already stands on and
settled in about a second, so run86 — its gate lit by day — spent ~300 of 600 iterations waiting
for one night. A successful `wait` now holds `[loop] wait_hold_s` (default 30 s,
`Agent._hold_wait`, `834f5b0`), which makes a ten-minute day some twenty decisions. The night
itself is no longer waited for: `place_bed` lays the gate bed by day as well (see **Homebase**
under the next milestones).

## Live findings run80–86 (2026-10-21 … 22)

| Run | Result | What held it back | Fix |
|---|---|---|---|
| run80 | 600/600, 0 deaths | `[BUILD_STOPPED]` names no cell, ~350 `clear_portal` ticks; 22 lightings on a partial frame; finished frame never lit by day | `017f69c`, `86f2846` |
| run81 | 600/600, 0 deaths | 470× `step_away_drops` refused on a cliff top, give-up never saw the refusals | `c48da18` |
| run82 | 600/600, 0 deaths, best early game so far | 481× `go_home`↔`stow_surplus`, the chest outside the base radius | `0d24d78` |
| run83 | aborted at it524 (`SSE stream ended`), 4 deaths | unarmed hunts at night; a phantom chest from earlier runs | `91c1cb5`, `9f5b62e`/`3f5b908` |
| run84 | stopped early | `mine_obsidian`↔`go_home` with a full pack | `251df45` |
| run85 | crashed at it31 | `KeyError: 'y'` in `_note_placed` after a heightless `surface_up` probe | `c7b3c38` |
| run86 | 600/600 `limit`, 0 deaths, **first crossing of this series** | a night of `[BED_NOT_FOUND]` (bed 51 m away, server searches 32); `clear_frame` dug its own fresh frame blocks; junk walk-off aimed due west into no footing; gate lit by day → ~300 `wait` ticks; at the fortress spawner 120+ quiet defers | `6d3d164`, `c63b1db`, `3907457`, `834f5b0`, `e0e7687` |

Details of the run86 fixes:

- **Sleep reach (`6d3d164`):** `_bed_available` counts any bed the scan sees — right for "is a
  second bed owed", wrong for the sleep itself. `_bed_in_sleep_reach` counts a carried bed or one
  within `SLEEP_BED_REACH = 32` (the server's search), and the night offer honours the withholding.
- **Frame cells that are done (`c63b1db`):** the `[BUILD_STOPPED]` marking reads the frame plane
  from **before** the build, so the cells that build had just filled were marked too, and
  `clear_frame` dug the fresh cobblestone corner and two just-laid obsidian back out.
  `perception.frame_wanted_blocks(base)` is now the one table of which block each ring cell
  wants, and `_frame_blocker` skips a cell that holds it.
- **Walk-off to proven footing (`3907457`):** the drop position is the bot's own feet, so "three
  blocks away from the drop" was a zero vector — `atan2(-0, -0)`, always due west, at the bot's
  own height. One `view_blocks box` around the drop (`Agent._scan_drop_ground`) now feeds
  `targets.step_away_cell`: solid footing (no lava/magma/cactus), room for the body, at least
  `STEP_AWAY_MIN = 4` blocks from the drop — past the reach of a toss in any direction, since
  `drop_item` cannot aim — nearest to the bot on its own level; a refused cell gives way to the
  next from the same read.

### run87 (2026-10-23): standing still, and a bed in a cave

- **The walk off a drop went the way of the toss.** In a one-wide tunnel the cells four blocks
  east and west of the drop tie, and the tie went to the lower x — west. The bot had dug west,
  so it faced west and tossed west, and the walk led straight through the pile: four rounds of
  `drop_junk` / `step_away_drops` picked the same junk back up (it171–176). The loop now keeps
  the way the bot last walked (`Agent._note_heading`, a move of at least a block), stores it with
  the drop (`drop_heading`), and `step_away_cell` prefers any cell behind it; the read-less
  fallback walks back along it too.
- **One scaffold block reopened the dig.** The surplus drop leaves exactly `NETHER_COBBLE`, the
  walk off it spent one block as scaffold, and `mine_stone` came up for the 63rd — its
  `collect_block` hoovered the dropped pile back into the pack (it177–179). The lull refills the
  stack only below `NETHER_COBBLE - NETHER_COBBLE_SLACK` (64 − 8).
- **A night at the base with nothing allowed.** The frame stood unlit (the spare pickaxe still
  owed), no diamond ore was in view, and sleeping at the base is forbidden before the gate burns,
  so the night offered `wait` alone for twelve ticks (it147–158). `_sleep_off_base` now offers
  `sleep` there: it walks `SLEEP_OFF_BASE_STEP` blocks west of the anchor, away from the frame,
  and sleeps off the base next tick; `take_bed` brings the bed home in the day. A gate that may be
  lit now keeps the bot at the base.
- **A night sleep in a zombie cave.** A night hunt left the bot at y=−1; the night offer laid the
  bed at (−64, −10, 58) and the game refused five sleeps for the zombies around it — but sets
  the respawn point before it checks for monsters, so the respawn stayed in the cave while the
  bot fought down to 0.2 health. `take_bed` was on offer from it198, beside `drop_junk`, and
  Laya kept picking the junk. Now the night sleep needs no hostile within
  `SLEEP_HOSTILE_CLEAR = 12` (wider than the game's eight blocks), and away from the base the
  bot's feet must not lie more than `SLEEP_DEPTH_TOL = 16` below the base floor (`_underground`);
  a standing bed that deep in the server's 32-block search keeps the sleep off too, since the
  server would sleep in it before laying the carried one. A bed already standing underground
  (`_unsafe_beds`) makes `take_bed` the tick's only step, eating aside, frame or no frame.

### run88 (2026-10-23): wounded, foodless, deferring

The fastest chain of the series — the whole diamond set worn by it55 — and then nothing: at it63
cave-spider bites during `mine_iron` beside a spawner took the bot to 8.5 health. It had not
carried one edible item all run, and at hunger 17 the game does not regenerate (`REGEN_HUNGER`
is 18). Below `RECOVER_HEALTH = 15` every hostile `yes` deferred to the survival policy, which
had nothing to fight and nothing to eat; the quiet-defer release only applied above the band.
536 defers to the iteration limit, health fixed at 8.5, while the spawner's spiders gathered to
89 at 10–16 m. `hunt_food` for a chicken 44 m away was on offer on it65–76 and deferred with the
rest.

`actions._recovery_owed` names the case: under the band, `DEFER_QUIET_TICKS` quiet defers, no
food carried, hunger under `REGEN_HUNGER`. `brain.danger_matters` then hands the tick back, and
`candidates` offers `_recovery_steps` alone: `hunt_food` with an animal in reach, else `go_home`
to the base, else `surface_up` out of the cave when there is no base, else `explore`. While the
body can still heal — food in the pack, or hunger at 18 — the defer holds as before.

### run89 (2026-10-23): the chicken stayed in the pack

run89 began in run88's state — 8.5 health, hunger 15, no food, in the spider cave — and died at
it21. Two gaps:

- **The recovery waited for defers.** The gate said `no`, so the loop had the tick and offered
  the ordinary work (`smelt_iron`, `hunt_wool`, `get_flint`) for fifteen ticks while a failing
  `surface_up` let the hunger fall to 5. `actions._cannot_heal` (wounded below
  `RECOVER_HEALTH`, no food carried, hunger under `REGEN_HUNGER`) now sends every tick in that
  state to `_recovery_steps`; the quiet defers (`_recovery_owed`) only gate the release from a
  defer in `brain.danger_matters`.
- **The meal rode along and lost.** `hunt_food` brought six raw chicken at it15, and on it16–17
  `eat_food` stood beside `surface_up` and `establish_base` at hunger 5 and 8.5 health; Laya chose
  the walks, the second met a drowned at night, and the bot died with the chicken still carried.
  `_must_eat` — food carried, and hunger under `EAT_HUNGER`, or health under the band with hunger
  under `REGEN_HUNGER` — makes `eat_food` the tick's only step. A milder need still rides along.

- **The hunt asked for more than it could see.** `hunt_food` sent `collect_mob_drop` for six
  chicken with four in view at 47 m. The server's hunt has no range limit and retargets to the
  next animal of the kind anywhere loaded until the count is met: six target changes, one 81.6 m
  out, 591 blocks in one 178-second action, 276 m from the start, hunger 8 → 5. `_hunt_count`
  now asks for at most as many drops as animals of that kind are in view (one per kill is the
  least each gives), `HUNT_BATCH = 6` at most, and `HUNT_HURRY = 2` when the bot is hungry or
  wounded, so the meal comes back fast and `eat_food` takes the next tick. And a chicken is the
  last choice (`FOOD_MOBS_LAST`): the server's hunger reflex never eats raw chicken
  (`REFLEX_AVOIDED_FOODS`, food poisoning), so the six run89 carried fed nothing while it
  deferred.

A range limit on `collect_mob_drop` itself would belong in `mine-ai-mcp`.

### run89, later (2026-10-23): three hours in a fight that never ended

At 16:11 the server's `hostile_reflex` took the body during a `hunt_wool` at y=−54 and stayed in
`fight`, phase `hold`, against a zombie 1.6 m away — 55,712 ticks, three swings, full health,
not one event after 16:11:36. The loop could only wait: every foreground call polled
`wait_for_action` up to `max_polls` (900 × 2 s), and a refused submission drained the held action
up to three times, so it311 came 31 minutes after it310 and it317 93 minutes after it316.

`MCPExecutor.call` now shares one deadline, `[mcp] action_wait_ms` (five minutes), between its
drains and its own wait. A call that runs out of it returns `[BODY_HELD]` with the holder the
last progress answer named (owner, response kind/phase, phase ticks) and status `held`, without
resubmitting. `Agent._note_held` counts such calls in a row — a defer between them does not reset
the count — and `BODY_HELD_STOP = 2` of them end the run with `"stop": "body_held"`. Nothing the
loop can send moves a body the reflex owns; the fix for the endless `hold` itself belongs in
`mine-ai-mcp`. The second call is what makes the report timely: run95 spent three *pairs* of
five-minute waits on one held action id each (it268/269, it279/280, it354/355) and never saw a
third in a row, so 18% of that run went into waiting on a body nobody was going to release.

Whether an answer has settled comes from the envelope's `response.data.state` (`accepted` and
`pending` run, everything else is final), not from the first `status` a depth-first search finds.
run90's it76 hunt settled `partial` (two of six chicken), was read as still running, and 900
polls — each answered at once with the same settlement — ran out in two minutes as a
`[BODY_HELD] ... owner unknown`. The refusal now also says how long the call waited.

### run90 (2026-10-23): the Nether reached, and ten deaths after it

The best opening so far: by it140 the bot had built and lit the portal, crafted the gold
helmet and a spare diamond pickaxe, and crossed with food. At it154 piglin brutes caught it at
(-76, 35, -317), and it burned to death; `take_bed` had carried the bed along (it128), so it
woke at world spawn, not the base, with nothing. Nine Overworld deaths followed, all at night,
no tool and no armour, most on walks the loop chose. Four fixes:

- **The policy after every death.** `_rearm_policy` re-armed only once a death was over
  120 s old, so deaths 25–34 s apart left the server's default policy in force (the reflex
  `fight`ing zombies naked); three of ten deaths were answered. A death is now told by its
  server timestamp (`GameState.last_death_at`), and the defer path checks it too.
- **An unreachable death site.** 231 `recover_drops` from one spot answered
  `[RECOVERY_SITE_UNREACHABLE]` until the record went stale; three in a row now write it off.
- **Shut in.** The last 330 ticks stood at (182, 70, 181): every walk `no path found` after
  70–78 nodes (a real search exhausted visits ~100,000). Three such refusals from one cell make
  `dig_out` the only step: a three-cell corridor at feet and head height, toward the base first
  and then each other side in turn.
- **Naked at night.** With no weapon and nothing worn, at night, `recover_drops` is not offered
  and `_recovery_steps` offers nothing, so the tick defers to the survival policy until the mobs
  burn at sunrise (`_defenceless_at_night`). The ordinary offer follows the same rule
  (`DEFENCELESS_WALK_LABELS`): `explore` and `get_wood` are withheld, and an otherwise empty
  offer is `wait` rather than the fallback `explore` — holding **where it stands** (`resolve`
  anchors on the bot's own cell, not the base), since the walk home is the same long walk. run95
  it274 was 20 health and full hunger with no gear, and the night's `explore` walked into a
  spider and a skeleton at 10–16 m.

The two `[BODY_HELD]`s (it76 `hunt_food`, it503 `get_wood`) were both `partial` results read as
still running, fixed in `5377fc7`.

### run91–92 (2026-10-24): `craft_table` offered with no logs

Both runs looped on `craft_table` — 43× (run91, 600/600 limit) and 28× (run92, killed by an MCP
host restart: exit 1, `MCPError: SSE stream ended without a response`) — every call ok=False
`[CRAFT_MATERIALS_MISSING] "Missing leaf materials"`, with `explore` wandering between the
refusals. The world was not fresh: run91 carried over obsidian ×10, raw_gold ×5, diamond tools
and a water bucket, but only oak_planks ×1 and stripped_oak_log ×1 stood within reach of the
craft. The attempt to hold the offer until a `pale_oak_log` is found (MC 1.21+, `dd6dd0e`) was
reverted the same morning (`9dc2c19`); the class stays open, and run94's fresh world crafted the
table at it4 without trouble.

### run93 (2026-10-24): the crossing, the blaze, and the closed relaunch gate

The first crossing since run86, and the fastest opening of the series: full diamond armour by
it48, `craft_flint_and_steel` it92, frame complete it95 (401.5,73,-320.4), respawn bed it107,
`light_portal` it163 and `enter_portal` it164 — in the Nether by 12:09. Nine blaze rods reached
the pack (it167 +7, it207 +1, it216 +1). Then the recovery gap: at 9.4 health and hunger 7 the
`_cannot_heal` gate handed the tick to `_recovery_steps`, which knew only Overworld moves —
`hunt_food` (no animals in the Nether), `go_home` (excluded there), `surface_up` (base exists) —
and fell through to `explore`. The bot wandered the Nether ~70 ticks (it296–319, `explore`
ok=False "body owner: hostile_reflex"), a blaze burned it down at 4.2 health (12:40:45,
32.5,53,84.3), and the rods, the diamond kit, the gold helmet and the food were lost with it.
Fix: `_recovery_steps` now offers `leave_nether` in the Nether (`985d420`) — the way out of
danger there is the way home.

The death then exposed the relaunch latch. `_nether_entered` is set on the first crossing and
never cleared, and both relaunch gates — `_spare_pickaxe_wanted` (the trip needs two diamond
pickaxes, one survived) and `_nether_departure_ready` (`enter_portal`) — hang off
`not nether_entered`. With the latch set, the Overworld chain offered nothing, candidates fell
through to the `wait` fallback, and the bot stood at the base ~25 minutes (~88 waits, ~78 of
them in a row, by day). Fix (`e812d83`): the question is "is a haul still waiting to be
processed?" — `_nether_haul_home` (rods or powder in the pack, or the stowed powder that
completes the trip) replaces the latch at both gates; an empty-handed return from a death
reopens the expedition. Before the first trip the helper is false, identical to the old latch.

Codes: BODY_HELD ×4, HOSTILE ×7, NAVIGATION ×2 (`[NAVIGATION_PORTAL_INTENT_REQUIRED]`,
`[NAVIGATION_STOPPED]`), NO_MATCHING ×3, CONTAINER ×1. The run ended at it479/600 (13:45:35,
exit 2) not through the loop but through `perception_blind`: `view_status` answered
`[MINECRAFT_DISCONNECTED] ... socketClosed`. The gold helmet was crafted (it116) but never
worn — `swap_to_gold_helmet` was offered while the reflex held the body.

## Live findings run94–107 (2026-10-25 … 27)

Distilled from the commit log (`ceace70..HEAD`); the detailed per-run subsections above stop at
run93. Two structural changes land in this window and the layer references elsewhere in this doc
still name the old shapes: **`actions.py` was split into an `actions/` package** (`constants` /
`predicates` / `planner`, `f09edd4`), and the **stronghold → end-portal phase shipped**
(`34079f5`: hunt endermen, craft eyes, locate and activate the end portal; re-supply sized to the
eyes actually owed, `acd4795`/`7158fe0`; documented in `docs/stronghold.md`, `95f1d44`). System 2
escalation moved from OpenRouter to the local Strata server (`fd709db`).

Overworld / base fixes (no run number in several, symptom-named): a roaming bot no longer re-mines
a built gate (frame-complete latch, `c303844`), an idle geared bot off its base walks home instead
of exploring away (`3970d05`), a lost shield asks for its six planks instead of waiting (`eff72bd`),
a carried chest only walks the bot home once the pack is filling (`986938a`), a fresh base stops
sending `stow_surplus` to a phantom cross-run chest cell (`383cf26`), the base lull fetches wood for
the spare pickaxe's sticks on expedition need (`8db2424`), a refused `dig_out` turns to the next
side and gives up after a full round (`ed6ae22`), a hunt never asks for more drops than the animals
within reach and a zombie is only hunted when the pack needs eating now (`bb980df`). run95 closed
four danger-band holes (`040f589`), `repeated_movement_failure` now escalates to `dig_out`
(`c9f11ba`), and run97's bed-wood dead zone that stalled on `wait` is closed by `get_wood`
(`dccf3f6`). run103 rebuilt a gate it lost obsidian from instead of waiting at an unlit portal
(`0c5366a`).

Nether / fortress series — the arc toward the first powder home:
- **run104** (`d763e44`, `89c3a71`): 438 explore ticks, 112 on one magma-cube spawner at (94,39,86),
  no blaze loaded — a fortress landmark underfoot was treated as an explore target and a reached cage
  written off at 0 m moved; the gold-helmet swap lost 52 of 66 Nether ticks to `explore` and now
  owns the tick outright.
- **run105** (`8e7d824`): the reverse write-off — a landmark the walk *reached* was discarded even
  when the step moved, so the bot ping-ponged 94↔150↔207 instead of re-navigating straight back.
- **run106** (`38c4e8c`): the closest yet, **12 rods carried out**, but the run ended on `body_held`
  — two `leave_nether` ticks held 300 s each at the spawner under the hostile reflex. The flee stage
  now raises the pursuit floor past full health so the reflex evades and frees `navigate`, and a hold
  that healed the bot, brought in a rod or thinned the pack no longer counts as the end.
- **run107** (`e902b7e`): the kit was lost to a Nether death (it131), then 365 ticks of
  `explore`↔`go_home` at the base until the bot nearly starved — the nearest animal (a pig at 113 m)
  sat past the day-hunt range cap, so `_recovery_steps` found no food target and degenerated into a
  two-cell loop (hunger 6→0, health 12→1). Below `HUNT_STARVE` in daylight the cap now drops so
  recovery reaches the far animal.

Net: crossing, fortress search and the rod haul all run live, but **no run has carried the blaze
powder back to the chest at the gate** — `nether_expedition_complete` is still unmet live.

## Live findings run108–127 (2026-10-28 … 30)

Ticks below are quoted from the archived logs in `runs/history/`, read with `scripts/dec.py`
(`8dcb2ed`), the level-1 reader for `decisions.jsonl`.

**A body with no footing cannot dig.** Both `[RUNTIME_UNRESPONSIVE]` incidents of the window hung
on the same call: `build_structure` digging air cells around a floating body at (-0.5, 62, 281) —
run115 it527 after `[WORKSTATION_PLACEMENT_FAILED] crafting_table: no_cell`, run116 it0 on the
first tick of the run, the dig the only operation still active in each incident's activity record.
`unblock_action` now hands a body with no footing to the swim and `dig_out` refuses to tunnel from
one (`5861fef`); run116 restarted and crashed on the same call with `inWater` false, so the guard
widened from "in water" to any body the server reports as not standing (`dce8da7`). The condition
is footing, not water.

**The water rescue climbs, it does not dive.** run115 it166-527 answered 281 identical
`escape_water` refusals, each aimed 29 blocks down at a cave floor and each stopping three blocks
under the body (`[NAVIGATION_STOPPED] … closest node was -1,59,281`). `nearest_ground` now passes
over cells the pathfinder already timed out at (`rescue_skip`, `32c9970`) and takes a dry cell
**above** the swimmer before one below it; run116's open-ocean case (`62c726c`) reads
`RESCUE_GROUND_LIMIT` (64) cells so a shore past the nearest-16 cut survives the scan, and the
last-resort "any standable cell at any level" fallback — which only ever produced the deep dive —
is gone. The progress rule measures the swim across the water (`swim_delta`, x/z only): the
vertical bob of a body in one column is what made 281 identical refusals read as progress and kept
the stuck guard from ever withholding the label.

**A foreground action that travels far and ends where it began is cancelled.** run113 it84:
`mine_obsidian` asked the server for 8 obsidian by count, the pour formed one block at the bot's
feet, and the action re-targeted a natural deposit 124 blocks away — 14 minutes, 2491 blocks
travelled, still 2 blocks from where it started, checkpoint stuck at 1/8, and the loop powerless
while the foreground owns the body. `_wait_settled` now reads the progress facts (≥ 240 s elapsed,
≥ 400 blocks travelled, ≤ 10 blocks from the start) and cancels through the server, retrieving the
settlement in the same breath (`add2381`). Elapsed time alone is not the tell — a long navigate
legitimately runs for minutes — the tell is travel that ends where it began. Phase 1 of
`mine_obsidian` names the exact cell the pour formed (`020fc34`), the same bug from the other side:
a bare `count` lets the server re-target the nearest natural obsidian once the formed block is gone.

**An `explore` that ping-pongs in place is now visible.** run112d it517-599 burned its last 83
iterations between two cells about four blocks apart. Every guard was blind to it: the step
succeeds (no code for `_track_stuck`), the position changes every tick (not idle for
`_track_idle`), and one short `explore` walks straight, so its own travel-to-net ratio looks fine
to `_track_wander`. The going-nowhere only shows across ticks, so `_track_orbit` (`0224073`) keeps
the feet-cells a label ends on over `ORBIT_WINDOW` ticks and reads the wander ratio there; a tick
that gains inventory, armour or off-hand, or crosses a dimension, resets the window.

**The furnace flag is a reach, not a memory.** run116: a bot that died at stone tier and drifted
~650 m from its old base still carried the server's memory of the base furnace (`furnace_pos`), so
`craft_furnace` was suppressed while `smelt_iron`, which needs `_at_base` and has no base out
there, could not run — no furnace, nothing to smelt at, every tech step blocked, endless `explore`
(it114+). The flag now mirrors `_smelting_reach` (`83e96b8`): craft a furnace exactly when there is
none to smelt at.

**The pack, not the plan, is the Nether bottleneck.** run117's two most-taken labels of 723 ticks
were `drop_junk` (131) and `step_away_drops` (126): one `[DROP_HOLE_UNAVAILABLE]` at it88 in a
gravel pocket at (-86, 1, -339) latched the disposal pit for the whole run, so 125 of the run's
drops landed on open ground and 115 of them were walked back over within three ticks — 257 of the
run's ticks (131 attempts, 126 walk-offs) went to dropping and walking off it. `84d0170` makes
both facts local: the pit
refusal is the ground's answer at one cell and releases 12 m on (`DROP_HOLE_MEMORY`), and an open
drop that comes back is counted as no drop at all (`DROP_UNDONE_GIVE_UP` = 2 in a row withholds
`drop_junk` for the stuck window). run118 showed the other half: through it92-151 `candidates` was
`['drop_junk', 'explore']` 27 times and Laya took `explore` every time, while the fortress march
hoovered netherrack 3 → 188 and the pack sat at 35-36/36. `junk_full` kept suppressing
`leave_nether` on the assumption that the pack empties itself; at it151 the bot died at
(-277, 85, -143) — 300 m from the gate, 0.5 health, five rods in a pack with no room for the
sixth. `e940e4c` gives the shed the tick (the shape the gold-helmet swap and the pearl wave already
use) and makes `junk_full` hold only while `drop_junk` is actually on offer, so a shed the ground
has refused twice sends the bot home with the rods it has instead of 15 ticks of marching with
nowhere for the next one.

**A fortress crowd is not a fight.** run119 it221-229: the bot stood at (-134, 70, 78) while a
blaze 3.9 m away shot at it for nine ticks. `_safe_blaze_hunt` asks for one target in reach, a
healthy bot and no other hostile within 16 m — and 53 zombified piglins stood 9.1 m off, so
`hunt_blaze` was never offered and the body stayed with `hostile_reflex` (`[BODY_HELD]`, phase
`approach`, 163 phaseTicks, 0 attacks). Piglins and zombified piglins turn on a player only when
provoked - a hit, or a ghast fireball that bounces back - and a fortress floor is never empty of
them, so counting them as company closes every hunt in the dimension the hunt is the point of.
`NEUTRAL_HOSTILES` and the shared `_crowding_hostiles` now leave them out of the blaze, spider,
string and zombie guards; a wither skeleton at 9 m still closes one.

**A crossing resets the policy, not only a death.** run119 it217: `enter_portal` put the bot in the
Nether, and the server's own `lastChange` read "Death or dimension change; defaults restored" -
`engage_min_health` 12, `hostile_avoidance_multiplier` 1, raw food `emergency_only`. Only deaths
were re-armed, so the eight ticks that followed at a blaze spawner ran on the cautious default with
the hunt closed. `_rearm_policy` (formerly `_reapply_after_death`) now tracks the dimension and
re-applies the base stage on every change, the first tick of a run excepted - whatever dimension
the bot starts in, the policy the run applied at its start already stands there.

**The bed stance must be a cell the walk can actually arrive in.** run120 it11-15 and it31-33:
eight identical `place_bed` ticks, the bot at (-9.5, 70, -51.5) and the carried bed still in the
pack. `_bed_stance` chose the nearest cell clear of the gate, `(-11, 70, -52)`, one block west -
and `navigate`'s default arrival tolerance is one block, so the call succeeded from where the bot
already stood, `_at_stance` stayed false, and `sleep` never ran. `_track_idle` withheld the label
after `IDLE_REPEATS = 3` such ticks, and the run's remaining 27 ticks were `wait`: with no bed at
the base, `tech_tree_complete` can never hold. Two changes, both read live against that base: the
stance walk now asks for `range: 0`, and `_bed_stance` only offers cells the ground scan proves
standable (a `view_blocks` hit with `air` above it, from the soil group and the wider rock cast in
`gs.rescue_ground`, since a base floor is usually cobblestone the run laid itself) - the offset
grid walks onto the anchor's level whatever the ground there is, and live the column at
`(-11, 70, -53)` is dug out to y=68, which `navigate` refuses outright with
`[NAVIGATION_TARGET_UNSUPPORTED] ... the ground in that column is at y=69`. With no ground read at
all the grid is still the only information there is, so it stays the fallback.

**A bed the scan cannot see is still the respawn point.** run121 it80: 280 m off base the base
chunks unload, `view_blocks find white_bed` returns nothing, and `craft_bed` was offered as if no
bed existed - a second bed, one more slot in a pack that was already at 27. `bed_pos` now rides in
the GameState like `chest_pos`: the `sleep` response names the cell it clicked (`bedPosition`, the
only place the server ever reports a respawn point - `view_status` has no spawn field and the bot's
SQLite has no respawn table), `_adopt_bed` re-adopts it from a scan at the base after a restart, and
it is forgotten when the bot is *at* the base, the scan ran, and no bed stands there.
`take_bed` clears it in the same breath.

**Sleeping needs the body beside the bed.** run121 it100-103: four identical
`[BED_SLEEP_REJECTED] bot is not sleeping` at (-9.3, 70, -52.5) with the bed at (-10, 70, -51) and
`respawnSet: false`. The server's bed search is 32 blocks, so `_sleep_ready` was satisfied; what
fails is vanilla's click box - the player must be within two blocks of the bed's top centre, and
1.7 blocks off in plan plus one block of height is 5.04 squared. `sleep` now resolves to a
`navigate` with `range: 0` onto a footing inside that box (`_bed_reachable`, `_bed_reach_stance`),
the same shape as `place_bed`'s stance walk, and the walk counts as progress for the stuck guard.
The box is not the whole story: the later refusals in the same run stood at (-9.5, 71, -51.5), 1.06
squared from that centre, and still did not sleep. mineflayer activates the bed block it finds, and
vanilla only resolves a bed click on the *head* half, or on the foot when the player faces the head -
so the heading the walk leaves the body on is the next thing to pin down, which is what
`scripts/bed_probe.py` measures stance by stance.

**A leftover blaze rod held the gate shut.** run123 it23-60: thirty-eight `wait` ticks at
(-10.45, 70, -51.47) with the gate portal lit three blocks away, full diamond kit, full hunger, and
the only offer in the shortlist being `wait`. The pack carried **one** `blaze_rod` from an earlier
run in the same world and no powder. `_nether_haul_home` read that rod as "a haul still on its way
in" and closed the departure gate (`_nether_departure_ready`), while `craft_blaze_powder` would only
grind at the flat nine-rod target and `stow_blaze_powder` had nothing to deposit — the haul could
neither be processed nor be replaced by a new trip. Two changes: the haul now means *a haul the
chain still needs and can actually bank* (the eyes owed > 0, and a crafting table for the rods or
powder plus a chest for the deposit), and the powder target is the eyes the frame owes rather than
`BLAZE_POWDER`, so one rod in the pack grinds into two powder and gets stowed instead of waiting for
eight more.

**A gate the scan cannot see is not an unlit gate.** run124 it16 vs it40-41: the bot crossed its
own base portal at it16, and by it40 sat 271 m into the Nether where `_portal_lit` read false again
- `_portal_lit` is a scan read, and a portal only appears in a loaded chunk. The base chain then
reopened for a gate that already stands and put `fill_bucket` and `get_flint` back on the offer in
a dimension where there is no water to scoop and no gravel to dig. Two fixes: the gate chain is
scoped to the Overworld, and `gs.portal_lit` is now a latch, seeded by the scan, by `light_portal`
and by any crossing - the same shape `frame_complete` and `powder_stowed` already have.

**The pearl wave banked its pearls where no step could reach them.** run124 it53: `stow_surplus`
deposited the wave's four ender pearls into the base chest - `KEEP_MATERIALS` names blaze rods and
powder but not the pearl or the eye, and the whitelist's rule is that anything unnamed is surplus.
No step withdraws pearls, so `craft_ender_eye` read an empty pack and `_pearls_owed` sent the bot
back to the endermen. Pearls and eyes are now on the whitelist.

**A haul the network cannot take is not a haul.** run124 it54, one tick after the same chest
swallowed the surplus: `stow_blaze_powder` fell back to the main chest's coordinates even though the
server had just called that lid full, and answered `[CONTAINER_BATCH_INCOMPLETE] ... blaze_powder
0/2`. The deposit now needs a chest with room, and `_nether_haul_home` counts powder as a pending
haul only while some lid can take it - otherwise the carried powder is supply the eyes can spend
directly, and the departure gate is not shut on it.

**An exclusive offer must still obey the stuck guard.** run124 it18-36: eighteen ticks whose only
candidate was `swap_to_gold_helmet`, each answered "No new action started; body owner:
hostile_reflex". The helmet swap returns its list early, before the `gs.stuck_labels` filter at the
end of `candidates`, so a withheld label stayed the sole offer forever. The same early return that
owns the shed tick (`drop_junk`) is guarded now.

**A base is adopted from what the run could have built, not from the whole kit.** run125: the bot
wore the full diamond set but carried no shield and no flint and steel, so `_gear_ready` read false
and `_adopt_portal_base` never ran - `base` stayed unset for all 600 ticks while six lit portal
blocks stood twelve metres away at `(-7..-8, 70-72, -52)`. Without an anchor there is no `go_home`,
no `enter_portal`, no `flatten_home_base` and no chest relief: the run sat at its own gate. The bed
at `(-10,70,-50)` could not anchor it either - `bed_frame_site` checks the frame plane the bed's
ground would carry, and the standing frame is a different plane three blocks off. Adoption now asks
the narrower question: flint and steel in the pack, or the diamond pickaxe obsidian is mined with.

**The chest relief chain fires live.** run126 it33-35: the main chest at `(-10,70,-49)` and the
overflow at `(-8,68,-54)` both came back full, `craft_chest` then `place_chest` raised a third at
`(-11,70,-50)`, and `_storage_target` moved to it. A network that has run out of lids builds a new
one rather than refusing the deposit.

**Powder the eyes can spend is not a haul to bank.** run125 it16-529: one pearl and two blaze
powder in the pack, sixteen eyes owed. `stow_blaze_powder` (the haul block) and `take_blaze_powder`
(the stronghold block) were both live on the same tick, and the model crossed the lid in one
direction and back in the other for five hundred ticks - thirty of each. Every transfer is its own
label's progress in `verify`, so the stuck guard never counted a single one. The deposit is offered
only when the carried powder is surplus to the eyes owed now: a pearl in hand, a table in the pack
and no more powder than the frame costs means the pack holds the next eye, not a haul.

**A haul is home when the bot is back at the base, not when the lid takes it.** run126 it16-185:
the fix above worked - the ping-pong stopped - but the same rule read the other way left the run
with nothing to do. Two powder and one pearl in the pack meant the deposit was refused, so
`powder_stowed` never latched, `nether_expedition_complete` stayed false, the stronghold block
never opened, and `craft_ender_eye` was never offered: 102 of 186 ticks were `wait`. The latch is
now "powder stowed **or** back in the Overworld with powder in the pack", and `_nether_haul_home`
excludes powder the eye chain can spend from the pack - processed supply is not a trip still
pending. Replaying run126's it100 snapshot through the new code gives
`['swap_to_gold_helmet', 'enter_portal', 'take_blaze_powder', 'craft_ender_eye', 'mine_gold_reserve']`
where the live run offered `['wait']`.

**A food source has to be one the offer can serve.** `_food_source_in_reach` counted any zombie
inside melee range as provisions, but rotten flesh is `EDIBLE` and never `PROVISIONS`, so a zombie
only feeds the reserve when the pack is at the eating floor. With the pack full and a zombie 20 m
off, the gate held the expedition on a hunt that was never on the offer. The zombie branch now
carries `_eating_now(gs)`, the same condition `_food_target`'s emergency branch uses.

**The wander guard must not close the way out of the Nether.** run127 it13-22: the rod haul was
complete (8 rods, target 7) and the only step left was `leave_nether`. Two exit walks of 35 blocks
each, both toward the gate, were interrupted by a blaze in contact (`[HOSTILE_CONTACT]`, 3.8 m and
3.6 m), and the pair read out as `walked 346.4, net 35.0` - the ratio the wander guard is built for.
`leave_nether` came off the offer for fifteen ticks, the Nether block fell through to `explore`,
the bot walked deeper into the fortress and died at it22 with eight rods, the powder, the pearl and
the diamond kit. The exit is retried, not pruned: `_track_wander` now skips `leave_nether` while the
bot is still in the Nether, and `_track_stuck` still owns a crossing that stops moving at all.

Net: the crossing, the fortress march and the rod haul all run live, and the powder reaches the
chest at the gate (run124 it6, run125 it1-3, `powder_stowed` latched); what no run has done yet is
turn that powder into an eye of ender. The water, footing and foreground guards closed the
runtime-crash and the walk-back-and-forth classes. run125 never crossed at all - it had no base
anchor to walk to - and its remaining stall was a chest lid rather than a fight; run126 crossed the
lid and stalled on the latch that was supposed to say the haul was done.

## Next milestones

- **Open after run86:** (1) the October fixes above are unit-tested but not yet seen live;
  (2) the gate may still be lit by day (`86f2846`), which leaves the bot beside an open portal
  for up to a day — the `wait` hold makes it cheap, not safe; (3) `mine_obsidian` re-made ten
  obsidian far from a base whose frame already stood (run86 it64–66), presumably because the
  frame-plane read only exists near the base; (4) other steps that leave the base (`mine_diamond`,
  `mine_gold`) are not yet held back by `_shed_owed`; (5) the server's `observed_containers`
  table was empty in the run86 world, so `_walk_home_to_memory` cannot fire there.

- **M3+ (open):** The **out-of-distribution** held-out split is now built (full run
  `run7d`): the zero-shot accuracy there is **0.934**, the fine-tune improves loss/ECE/Brier,
  but **not** the accuracy — the rows imitate the logged gate choice, and imitation cannot fix its
  errors. A real accuracy gain would need a stronger signal than
  imitation (e.g. verifier reward directly) or more/more diverse runs. Both `in_danger` classes
  are represented with the defer fix; the new upper bound only caps the distant `yes`.
- **M5 — stronger training signal (built): relative narrative + offline teacher/judge.**
  Extends the M3+ gap above: imitating the logged choice cannot fix the choice's own errors, so
  the head needs a signal stronger than imitation. Two parts, both **offline** — the live loop
  (`graph.py`) is untouched, so this is a post-run analysis step, not a change to the running bot.
  The **code** is in; the retrain + live validation are a run-the-user-owns step (below).
  - *Relative narrative (done).* `perception.narrative()` now renders the bot, the base and the
    death site **relative**: the bot's `x/z` are dropped and `y` becomes a **depth band** against
    the frame of the dimension it is in (`_depth_band`: Overworld bedrock ≈ −64, sea level ≈ 63,
    diamonds < 16; the Nether is a shell from bedrock floor 0 to ceiling ≈ 127, so its open floor
    in the 40s reads as surface, not "underground") →
    "at the surface"/"underground"/"deep underground"); the base is a distance + a coarse vertical
    word (`_rough_vertical`: level/above/below/far); the death site a distance. Absolute x/z are
    per-world random and carry no transferable signal — the head would only *learn to ignore* them,
    the same reasoning that deleted the `players nearby` line. The distance word is unified to
    `blocks` (1 block = 1 m), no legend sentence — the head learns the scale from consistent data.
    `as_state` stays **absolute** (diagnosis, S2 rebuild, `from_state`). **Retrofittable:** the log
    stores both `narrative` and `state`, so `dataset --rerender` (`dataset.render_narrative` =
    `GameState.from_state(state).narrative()`) rebuilds a whole archived corpus in the new wording
    before a retrain — training = inference. (The death clause cannot be reconstructed, its age is
    not in `as_state`, so a re-rendered death tick drops that one sentence; everything a step keys
    on survives.)
  - *Offline teacher/judge (done).* `mlaya teacher --log …` (`teacher.py`) replays an archived log;
    for each action tick with ≥ 2 candidates it asks System 2 (`[llm]`, via
    `brain.escalate_to_system2`, Laya's own pick withheld so S2 is not biased) over the **same**
    `narrative`/`candidates` and logs agreement + S2's reason. On disagreement a third model, the
    judge (`[judge]` — its own model, so it does not grade its own earlier opinion), sees the two
    actions **blind** (A/B order seeded by the tick's narrative hash, not told which is S1/S2) and
    rules A/B/neither/unsure; only a clear verdict carries a label. Output:
    `runs/history/teacher-runNN.jsonl`, keyed `(iteration, narrative_hash)`. `dataset.build_rows`
    reads it with `--teacher` as an extra label source: a **no-progress** teachable tick whose
    judge clearly sided against Laya's pick becomes a positive row for the judge's label. The
    **verifier stays the grounded primary truth** — a tick that demonstrably progressed is not
    overridden, and any de-dup collision goes to the verifier. Scope: `intent` only; the
    `in_danger` gate's floor/ceiling heuristic in `dataset.py` is untouched.
  - *Still owed (user step):* rebuild the corpus (`dataset --rerender [--teacher …]`), train, and
    validate against **real run outcomes** (goal, deaths, ticks-to-milestone), not judge agreement —
    otherwise the head is tuned to please the big model rather than to survive. Go live via
    `LAYA_EXTRA_MODELS` only once it at least matches.
- **Iron chain (re-)validated live (tier 0 → 3):** `craft_pickaxe` → `mine_stone` →
  `craft_stone_pickaxe` → `mine_iron` → `mine_coal` → `craft_furnace` → `smelt_iron` →
  `craft_iron_pickaxe` has **run through live**, including the new unblock recovery
  (`[WORKSTATION_PLACEMENT_FAILED]` → `make_room` digs a pocket → craft takes; `explore`
  as a second instance).
- **M2/M2b/M2c/M2d validated live:** `replace_tool`, `hunt_food`, `collect_drops`
  (decision/binding), `recover_drops` (see M2c/M2d) as well as `craft_table`, `hunt_wool`, `craft_bed`,
  `sleep` (see M2b) have **run through live**. Thus in M2/M2b/M2c/M2d no path remains without a
  live run; the bed/food chains are additionally pinned by the determinism tests.
- **Diamond tier (validated live):** `mine_diamond` → `craft_diamond_pickaxe` → (2 more
  diamonds) → `craft_diamond_sword` has **run through live**; final state
  `tools = {"pickaxe": "diamond", "sword": "diamond"}` (tier 4), `tests/test_actions.py::DiamondChainTest`.
  Two points found live: (a) the iron pickaxe broke *inside* a long
  `collect_block` (see above), the chain healed itself by re-crafting the iron; (b) an original
  tier cap on `mine_diamond` had made the sword **unreachable** (deadlock) — the cap
  is removed, `mine_diamond` now knows the target quantities from `_next_diamond_item`
  (3 pickaxe, 2 sword, then the armour set).
- **Diamond armour (validated live):** `DIAMOND_ARMOR` (chestplate 8, leggings 7, boots 4,
  helmet 5 = 24 diamonds) hooks onto the same chain via `_next_diamond_item`; one missing
  piece per tick, `craft_diamond_<slot>`, then `equip_armor` (one `equip` batch, without
  `destination`). Perception reads the **worn** set from the inventory `location`
  (`head`/`torso`/`legs`/`feet`); `situation.tools.armour` is only the list of carried
  armour per class. `verify` counts a worn piece as progress via `armor_delta`.
  Unit-tested in `tests/test_actions.py::DiamondArmorTest` (9 tests) and
  `tests/test_perception.py::ArmourParseTest` (2 tests). **Run through live** (25 ticks):
  `mine_diamond` (count 8 → 7 → 4 → 5, together exactly the 24 diamonds) →
  `craft_diamond_chestplate` → `craft_diamond_leggings` → `craft_diamond_boots` →
  `craft_diamond_helmet`; `tools = {pickaxe, sword: diamond}`, hp 20. **Corrected finding:** the
  server does **not** equip by itself — the crafted pieces stay in the inventory (`location: main`)
  until an `equip` puts them into an armour slot. The first final state "all four in `armor`" was a
  parser artefact (see above); after an explicit `equip` the stacks show `head`/`torso`/`legs`/`feet`
  and only then does `tools.armour` show the worn set.
  **Second finding:** `sleep` failed 3× in the 1×1 shaft with `[BED_PLACEMENT_FAILED]`; the
  unblock mechanism now covers this code as well (see M2).
- **Shield (validated live):** `craft_shield` (6 planks + 1 iron ingot, `temporary_workstation`)
  → `equip_shield` (`equip` with `destination: "off-hand"`). Perception reads the off-hand stack
  via `location: "off-hand"` into `GameState.offhand`; `verify` books `offhand_delta`. The finding
  that first made the tier reachable: with a standing diamond pickaxe `iron_wanted` was wrong,
  so the loop offered only `['explore']` live, although `iron_ore at 4m` was in reach —
  `_iron_goal` (3 for the pickaxe/spare, otherwise 1 for the shield, otherwise 0) pulls the iron. Tested
  in `tests/test_actions.py::ShieldTest` and `tests/test_perception.py::OffhandParseTest`.
- **Auto-stop on completed tech tree (implemented):** `actions.tech_tree_complete(gs)` is true,
  once the full diamond kit stands — diamond pickaxe + diamond sword, **all four**
  diamond armour pieces **worn** (`_worn`, not just in the inventory — otherwise the run would end
  before `equip_armor` takes effect) and the shield in the off-hand — **and** the overworld goal stands. The
  goal is split in two. `_expedition_ready` is the prepared trip minus the burning gate and the laid
  bed: **a bed at hand** (`_bed_available`, carried or standing — a leftover bed merely *loaded* still
  needs `take_bed`/`place_bed` to bring it home), **no Nether prep step left open**
  (`_all_nether_prep_steps`: bow+arrows, gold helmet, deep cobblestone) **and the food reserve
  banked** (`_food_reserve_met`, the 64 units from `_food_goal`) — and, because the bed is laid
  *after* the gate burns (see the bed note below), with a bed at hand, so `place_bed` runs the
  very next tick, by day or by night. `light_portal` requires `_expedition_ready`, so **the gate is lit last**: the portal
  burning is the last Overworld act, and `tech_tree_complete` — `_portal_lit` **and** `_gear_ready`
  **and** `_bed_at_base` **and** `not _all_nether_prep_steps` **and** `_food_reserve_met` — closes in
  the same tick the bed goes down. It must never light early — the bot works at the base right
  beside the frame and the pathfinder walks an open portal like any block, so a gate burning through
  the prep window walks the body through it (live, run22: lit at it208, the bot stood in its own
  portal and was in the Nether by it245, with no step asking for the crossing). Prep steps like the
  food demand drop away as soon as their raw material is unreachable (no spawn, no ore, no
  food animal in the loaded area and no zombie in melee reach), so an unreachable item never
  blocks the run. `[loop]
  stop_when_complete` (default `true`) lets `verify`/`defer` end the run with `stop: "goal"`
  instead of working through the iteration budget; otherwise it ends with `stop: "limit"`.
  `run --ignore-goal` disables the stop condition for a run (dataset collection beyond the
  goal). Tested in `tests/test_actions.py::TechTreeCompleteTest` and
  `tests/test_graph.py::GoalStopTest`.
- **Nether chain (implemented):** Nine labels in `actions.py` lead from the finished diamond kit to the
  open portal: `craft_bucket` (3 iron ingots) → `fill_bucket` → `mine_obsidian` → `get_flint`
  (from gravel) → `craft_flint_and_steel` (2×2, **no** crafting table) → `clear_portal` (blocking blocks
  from the opening, see homebase) → `build_portal`
  (`build_structure portal_frame`, 10 obsidian + 4 cobblestone corners) → `light_portal`
  (`activate_portal`) → `place_bed` (`sleep`). The whole chain hangs on `_gear_ready` (a fresh
  run first completes the diamond tier).
  **Obsidian is made, not mined:** `mine_obsidian` has two phases in `resolve`. If obsidian
  lies in reach (`OBSIDIAN_REACH` = 6 m — whatever the last pour formed), it is
  mined (`collect_block`, exactly the missing blocks up to 10). Otherwise the water bucket is poured onto the
  next lava cell (`use_bucket pour` with `x,y,z`, from `POUR_REACH` = 4.5 m; otherwise first
  navigate there — **without** `dig`/`scaffold`, so the pathfinder does not tunnel into the lava).
  As "open" above the pour cell, besides `air`, `cave_air` also counts (`perception.AIR_NAMES`):
  underground lava lakes are covered by cave air instead of `air`, and the pour check discarded
  such cells first (fixed live in the from-zero run).
  Natural obsidian lies only deep (`y ≈ -55`) under the same lava lakes and never falls under
  `OBSIDIAN_REACH`; digging down there is the **water trap** (see reflex below). The step is only
  offered when "bucket full **and** lava in reach" or "obsidian in reach".
  **Obsidian latch (key design):** the chain closes on the **frame**, not the igniter. It used
  to require obsidian to be banked **before** flint and steel — that craft was the lock that kept
  `mine_obsidian` from re-mining after the build (which yields the 10 blocks), because a pure
  "resource present" check would otherwise mask the fresh frame. But a restart can carry the
  igniter into a world whose pack is short, and then the igniter latch locked the chain out
  completely (live, run17: 4 obsidian + flint and steel, no way to raise the frame). So the lock
  is now `_frame_started_at_base`: `mine_obsidian` (and the water chain that feeds it) stands
  while **no** obsidian lies in the base's plane (`_portal_frame_started`, with a base set), which
  tells our fresh frame apart from natural obsidian where the igniter count cannot. So that a
  **partially** built frame does not run into a dead end,
  `_portal_frame_started` checks the fixed frame plane (`_portal_site` ± neighbours) for obsidian: if any lies
  there, `build_portal` may **continue** with the rest in the inventory (after a partial build the
  remaining quantity is exactly the still-missing cells), and `light_portal` only fires when obsidian really
  stands **in the frame** (`has_fs and obsidian < 10 and _portal_frame_started`) **and** the trip
  is fully prepared (`_expedition_ready`, see the auto-stop above); natural obsidian
  next to the base no longer counts.
  Tested in `tests/test_actions.py::NetherGateTest` and `tests/test_verify.py`.
- **Home-base leveling (implemented, 2026-10-30):** the survival reflex stacks defence scaffold
  during fights and a buried-and-raised base leaves stray terrain; neither was ever cleared, so the
  home ground turned lumpy and sometimes hid a tunnel mouth. `flatten_home_base` is a daytime
  maintenance step that digs those bumps back to air (`build_structure` with `block_name: "air"`).
  The removable cells come from the loop's own footprint read (`Agent._scan_base_terrain` →
  `gs.raised_cells`): one `view_blocks box` over the 9×9 column field centred on the anchor (the
  server caps `half_width` at 4, so `FLATTEN_RADIUS` = 4 is matched in `perception.scan_base_box`),
  keeping every cell **above** the floor `by` up to `by + FLATTEN_UP` (= 4) that holds a block in
  the `FLATTEN_BLOCKS` **whitelist** (terrain + `COMBAT_SCAFFOLD_BLOCKS`). A whitelist, not a
  blacklist, so the base's own furniture is safe by construction — a chest, furnace, smoker,
  crafting table, bed, the obsidian frame and a lit `nether_portal` block are none of these and are
  never dug; the floor `by` itself is never touched, so nothing the base stands on is undercut.
  Cobblestone **is** on the whitelist on purpose: the portal's four cobblestone corners are only
  needed while the frame is built and the gate works without them, so a corner left standing above
  ground level is cleared like any other bump (the two lower corners sit at `by` and stay bedded).
  The dig batch is capped at `FLATTEN_BATCH` (= 8, one route search per cell — `clear_portal`'s
  reason); the scan re-reads each tick and the rest falls to the next. Gated through
  `_base_leveling_wanted`: at the base, Overworld, gate lit (so the base geometry is final), the
  list non-empty, and **daytime only for now** — to watch how it grips before it runs at night as
  well. Offered below the Nether prep and the gate chain but above `wait`, so an idle bot at a lumpy
  base levels it and the step drops out the moment the scan comes up empty. `verify.progressed`
  counts a dug block that lands in the pack or a cleared cell that dropped nothing (the
  `collect_smelted` shape: a refusal's empty delta is not progress). Tested in
  `tests/test_actions.py::FlattenHomeBaseTest`, `tests/test_graph.py::BaseTerrainScanTest`,
  `tests/test_verify.py::FlattenHomeBaseProgressTest`.
- **Water escape reflex (implemented):** `candidates` checks `gs.in_water and not gs.on_ground` as
  the very first thing (even before the M2d reactions) and then returns **only** `["escape_water"]` — here
  Laya does not choose, the state is a rescue. The loop tracks `gs.water_ticks` for this (`Agent._inject`)
  and holds the rescue back for `WATER_RESCUE_AFTER` = 2 decisions: "without ground nothing can move"
  applies only to work that needs **feet on the ground** — a body between two banks
  **swims**, and the server navigates that itself. Had the rescue triggered at the first water
  contact, it would have yanked the bot out of every crossing as soon as an action came back (**live run15**: the goal ran
  east into a flooded seagrass flat, the rescue ran it back — 155 rescue paths against
  30 diamond digs, 157 breath aborts, 0 diamonds). `resolve("escape_water")` navigates **first to the
  nearest dry, standable ground** (`targets.nearest_ground`: a cell at the bot's own level — the
  shoreline the swimming body can really reach) with `dig`/`scaffold`. The pick went wrong twice in
  **live run19b**: taken as the outright nearest, the aim was a *cave floor* 32 m below the swimming
  body (a dive — `[NAVIGATION_STOPPED] ... closest node was -182,59,-85`), and the shore that would
  have ended it (65.1 m, at the bot's own level) sat a whisker past the 64 m read the list was
  bounded by, so the aim fell through to the base. `nearest_ground` now prefers a cell within
  `RESCUE_Y_BAND` of the bot's level out of a ground pool read to `RESCUE_RADIUS`
  (`perception`, `gs.rescue_ground`: the same `view_blocks` payload, a wider bound — every other
  group keeps its 64 m read), with the nearest open-sky cell at any level as the fallback, so a
  world with only a low dry ledge behaves as before. That pool also carries the rescue's **own,
  wider cast** (`RESCUE_GROUND_NAMES`: the four soil names plus `stone`, `cobblestone`,
  `deepslate`, `andesite` — the rock a cave pour leaves underfoot; `gs.resources` keeps the
  soil-only view `explore` and `base_target` were built on), and a cell **at the bot's own
  waterline may be roofed** (`cave_air` above, `BlockHit.rescue_standable`): a bot that pours its
  bucket on a cave lava pool stands on cobblestone, and a rescue reading the soil names only saw no
  ledge beside it at all — **live run20**, the escape cell sat 0.6 m off while the aim fell on
  grass 37 m away / 34 blocks up, and every path stalled 20 s a tick for ~15 ticks. A roofed cell
  is taken **only** at the waterline, so the far cave-floor dive stays out. The homebase is
  **only the stopgap** when no ground lies within scan range.
  Reason: without ground under the feet the shared pathfinder breaks — a `collect_block` ran live
  370 blocks wide and a net 0.18 m — while `navigate` keeps working from the swimming posture.
  (**Live run8:** the bot drowned ten times or more because the reflex stubbornly navigated to the
  underground base — 70 m straight down, 300 m away — instead of to the nearby coast; the
  base-first rule was the error, not the model.) If the bot stands on the bottom (water trench, pond,
  a pour that does not flood it), the reflex does **not** engage. Tested in
  `tests/test_actions.py::EscapeWaterTest` and `tests/test_graph.py::WaterTicksTest`.
- **Portal escape reflex (implemented, live run22):** The shared pathfinder treats a **lit nether
  portal block as open ground**, so a `navigate` goal on the far side of the gate routes the body
  straight through it — and a body standing in the gate teleports a few seconds later, with no step
  ever asking for the crossing. Live run22: the gate was lit at it208 beside the base, the bot kept
  working there, and by it245 it stood in the Nether. Two defences: (1) the gate is **only lit once
  the trip is ready** (`light_portal` requires `_expedition_ready`, see the auto-stop above), so no
  lit portal exists during the long prep window at all; (2) a reflex for the residual case — an
  adopted gate a previous run left standing, or the instant just after lighting. `candidates` checks
  `_in_portal` (the bot's own feet or head cell is a loaded `nether_portal` block) as the very first
  thing, ahead of the water rescue — the teleport is faster than drowning — and returns **only**
  `["leave_portal"]`, another rescue where Laya does not choose. `resolve("leave_portal")` navigates
  one step out with **no** `dig`/`scaffold` (the frame must not be damaged by the rescue), preferring
  the home base (the frame plane sits two blocks east of it) and otherwise the nearest horizontal
  neighbour that is not itself a portal cell. Phase 3, the intended crossing, must bypass this.
  Tested in `tests/test_actions.py::LeavePortalTest`.
  **Stuck (`repeated_movement_failure`):** The server reports this stop when the body does **not**
  get anywhere on a re-issued route — the bot is physically enclosed.
  Live run10: in a 1-block-wide, flooded shaft, `navigate`, `collect_block` and
  `build_structure` all answered like this, and the forced `escape_water` rescue ran **endlessly** against it. That
  code counts as `blocked_by_surroundings`, so the unblock ladder engages first — dig a pocket
  (`make_room`), otherwise to open ground (`explore`). But neither a feet-pocket nor another walk
  frees a bot walled in on every side: **live run96** spawned in birch logs and leaves and answered
  **100 `explore`** from the one cell (-64.6, 73, 3.5), never a step, the ladder cycling
  make_room/explore for 220 iterations. So `repeated_movement_failure` now also feeds the shut-in
  count (`_note_boxed`), exactly like a short `no path` search — `shut_in` cannot read it (no node
  count in the message, and its `[NAVIGATION_STOPPED]` code used to *reset* the count). Once
  `BOXED_REPEATS` such refusals from one cell have shown the ladder cannot free the bot, `dig_out`
  owns the tick and carves a tunnel out, one side per try. Tested in
  `tests/test_graph.py::ShutInTest::test_a_body_that_cannot_move_shuts_the_bot_in`.
  (In the run10 case the server additionally rejected digging/building with `Underwater stance
  unavailable` as long as the head stood in water — a server-side guard, `src/navigation/
  diving.ts`; only when the head reaches into air is work allowed.) Tested in
  `tests/test_graph.py::BlockedBySurroundingsTest`.
  **Dive backstop (fixed live, run15):** The branch that, without a sight corridor, immediately reported
  `300` ticks of breath ("rescue at once") also hit a body that was **standing on the bottom** in shallow water
  or whose breath was **full** anyway. `surface()` aborts at once with a full bar
  (`air_full`, no movement), but the owned dive continuation still returns the running action
  — the "rescue" therefore only declined. That is exactly how the livelock arose: seagrass cells are not
  source water, `openWaterSurface` found no corridor, `backstop()` reported 300 against 300 ticks of breath
  — 157 breath aborts. `Diving.backstop()` now checks for **standable ground** just like `beforeWork`
  (`isSafeSupport`) and leaves a **full** bar alone; the rescue only fires when
  breath is really used up. Tested in `src/navigation/diving.test.ts`.
- **Homebase (implemented):** An anchor on the surface carries bed, portal frame and chest.
  **First climb up, then anchor (`surface_up`):** The only reliable statement "this is
  surface" is the **depth below the surface** — not the block above the head. A tunnel dug by the
  bot leaves `air`, not `cave_air`, and is therefore indistinguishable from open sky by
  *no* `view_blocks` value (`_OPEN_SKY` like `above == "air"` only measure "there is
  support", not "this is sky"). **Live run9:** the base landed at `y = -16` under a
  `y = 73` surface, 89 blocks deep, because the bot had descended to the diamond digging and the
  scan found a cave floor there. Therefore `establish_base` is bound to **`_at_surface`**: it is only
  placed when the bot stands **in the same column** as the learned surface **and** `|gs.pos[1] −
  gs.surface[1]| <= BASE_SURFACE_TOL` (2) holds. The column check carries: the cell
  was learned for *one* column, a bot that digs under a *different* column must not
  count as "at the surface" merely because of a similar height. **Live run10:** feet on a
  `y = 54` cave floor lay within the old tolerance 8 of a `y = 59` surface learned in another column,
  `establish_base` fired underground, and the way there wedged the bot into
  a flooded shaft. Tolerance 2 now only catches the single slab or a
  one-block hollow at the bot's *own* surface. As long as the
  surface is unknown or the bot stands too deep, the candidates offer **`surface_up`**:
  a `navigate` **without `y`** asks the server for the column's surface — with an
  unambiguous column it climbs up itself and otherwise stops with `[NAVIGATION_HEIGHT_REQUIRED] …
  highest block … at y=N` (`perception.parse_surface_height` takes the highest of the listed
  stand heights). `Agent._note_surface` reads the answer into `self.surface = (x, feet_y, z)`; the
  next tick climbs with `dig`/`scaffold` to exactly this cell (measured live: from `y = -15`
  to `y = 74` in ~92 s, 12 cobblestone scaffold). If the column changes, it is checked anew (another
  column can lie dozens of blocks higher/lower); a failed *ascent* leaves the cell standing
  and is simply repeated. `Agent._scan_build_site` also now scans only at `_at_surface`,
  otherwise it would propose the cave floor itself. Live: `fill_bucket`↔`go_home` oscillated 45
  iterations, because the bucket was to be filled even after banking the obsidian (fix: the
  bucket chain is bound to the frame still missing, like `mine_obsidian` — see the obsidian
  latch above).
  `establish_base` then runs onto a **level, free** spot: two
  `view_blocks box` reads around the bot — its level and four above — (`Agent._scan_build_site` →
  `perception.scan_ground` → `targets.flat_build_site`) demand a continuous floor row at the
  same height with **open sky air** (`_OPEN_SKY`) above **and solid ground one level
  below**: the server digs out the
  floor row and re-places it as obsidian, so a cell one above a hole would wrongly stay "without
  a landing surface" and the opening would never close (live, `light_portal` hung like this in
  `[PORTAL_NO_FRAME]`). This makes the frame fit the terrain (live the old pick
  — the *lowest* standable cell — ran into a hollow, and 15 of 20 frame cells were stuck in stone). If no spot is within
  scan range, `targets.base_target` applies as before (only `GROUND_NAMES`, at/above the
  bot level, nearest first). `_REPLACEABLE` (incl. `cave_air`) stays for the
  opening check `_clear`, where enclosed air is correct. `establish_base` records the anchor
  as `GameState.base`; `go_home`
  only walks back when base work is really waiting (`_base_steps_ready`). `build_portal`/`place_bed`
  are restricted to `_at_base`, `_portal_site` anchors on the base block instead of the feet.
  **Anchor repair (`Agent._note_base`):** The anchor is set once at `establish_base` from the
  navigation goal (as a block coordinate: `args.y − 1`); nothing checks it afterwards against the
  living world. If the bot digs away its own base floor (or a creeper blows it up, sand
  falls), every `go_home` answers the now unsupported goal `base.y + 1` with
  `[NAVIGATION_TARGET_UNSUPPORTED] No usable footing … The ground in that column is at y=N.` — **Live
  run11:** after `establish_base` the bot had dismantled its base to open air, and the next
  **230** `go_home` ticks answered the same refusal, without the run making progress (Laya chose
  `go_home` again and again, because it wanted to get rid of a carried chest and `_base_steps_ready`
  stayed True for that). But the refusal names the height the server *actually* found, so
  the anchor is simply re-pinned to it — the same "read the height back from the server answer"
  move as `_note_surface` (`perception.parse_ground_height`). If the answer does not move the anchor
  any more (the same height it already carries), the spot is not repairable in place, and the base
  is **discarded** (`base`, `build_site`, `_build_site_scanned` = `None`), so that the loop anchors
  anew on a fresh `flat_build_site`. Both paths are bounded: one tick to heal, one to
  reset, never the endless retry. `tests/test_graph.py::BaseAnchorRepairTest`,
  `tests/test_perception.py::ParseGroundHeightTest`.
  **Clear the opening before the build:** The scan clears the glade only for the base *it*
  chose. An adopted base (`_adopt_homebase`, whose own bed site is now checked by
  `_scan_bed_site`/`bed_frame_site`) or one chosen via `base_target` never had
  this check. Therefore, during the build window (flint and steel carried, portal
  unlit), the loop reads the six interior cells of the opening (`perception.scan_frame_volume` → `parse_cells` →
  `perception.portal_interior`); if a non-replaceable block lies there, the candidates show
  `clear_portal` instead of `build_portal` and digs it away (`build_structure` with `block_name: "air"`,
  at most four cells per call — the server starts a route search per cell). A block above
  the support layer would otherwise be moved and `light_portal` would answer `[PORTAL_NO_FRAME]` forever
  (live the generic night sleep threw a bed exactly there).
  **Sleep sets the respawn, by day too:** `place_bed` (via `sleep`) is offered whatever the hour.
  By day the server lays the bed and uses it instead of sleeping, which sets the respawn point
  all the same ("Respawn point set at the bed, but daytime prevents sleeping"). It used to wait
  for the night, and run86 stood ~300 `wait` ticks beside a gate lit by day.
  `_bed_available` also counts a **placed** bed (not just a carried one): after a
  manual base build or a restart the bed otherwise remains as a block and the loop would
  needlessly look for wool for a second one.
  **Bed once the gate is lit:** Before the lit gate the generic `sleep` at the
  base is **suppressed** (`_at_base` and *not* `_portal_lit`). The server places the bed "next to the
  bot", and at the base that is exactly the portal interior — `build_portal` cannot
  clear a bed there (`remove_wrong_blocks` answers "movement policy prohibits breaking this block"). Live
  the bed landed in the opening and `light_portal` answered in every tick with
  `[PORTAL_NO_FRAME]`. The base bed is `place_bed`'s job and waits for the *lit* gate: with the
  frame up but unlit the opening is still open air and a bed laid there blocks the lighting, so the
  gate goes first. `light_portal` requires `_expedition_ready` — which holds once a bed is **at
  hand** (see the auto-stop) — and `place_bed`
  lays it the very next tick, when the opening is solid. Away from the base the night
  sleep stays unchanged **until the gate is lit**; a night that finds the bot at the base with a
  carried bed and a gate not yet ready to light walks it off the base first (`_sleep_off_base`). After that
  `sleep` is restricted to the base (`_at_base` *iff* `_portal_lit`): anywhere else it would pin the
  respawn point away from the gate, and a carried bed laid wherever the bot happened to stand is
  exactly the leftover `take_bed` then had to fetch. A bed standing **away from the base** — the
  respawn bed of an earlier base, or of an earlier run in the same world — is taken up in the lull
  (`take_bed` → `collect_block {block_name: "white_bed", count: 1}`, once the frame stands)
  and laid again at the base by `place_bed`, so the gate bed ends up where it belongs. The
  auto-stop insists on it too: `tech_tree_complete` adds `not _bed_off_base`, so the run cannot
  end with the gate bed still sitting in an earlier run's shaft (live, run19: the goal read
  complete with the run17 bed at y=55). Live, run18:
  the run17 bed sat at y=55 inside the shaft the new surface base dug for gold, and every
  `collect_block` stumbled on its half-block top (feet 55.56, no settled arrival) while `mine_gold`
  re-dug the same shaft.
  A **chest** (8 planks, 3×3) holds surplus: `craft_chest` → `place_chest` (coordinates are
  remembered) → `stow_surplus` (`use_container deposit`) at ≥33/36 slots; `drop_junk` is only the
  stopgap without a chest. `drop_junk` sheds the low-value `JUNK_ITEMS` list plus, since `df3ca56`,
  any single item carried above `HOARD_CAP` (64): a pack packed with one cheap block (live, run21:
  ~306 cobblestone, no chest) can still make room, while `HOARD_KEEP` keeps a floor under the
  banked cobblestone reserve (`NETHER_COBBLE`). A shed either goes into a disposal pit the server
  digs and plugs behind the bot, or, where the ground refuses one (`[DROP_HOLE_UNAVAILABLE]`), onto
  open ground at the bot's feet — and that drop is not finished until `step_away_drops` walks three
  blocks off it. Both are per-cell facts (`84d0170`): the pit refusal is remembered at the cell that
  answered and released 12 m on, and an open drop walked back over is counted as no drop at all,
  with the label withheld for the stuck window after two in a row. In the Nether rod wave the shed
  owns the tick outright when the pack is full (`e940e4c`) — offered beside `explore` it was taken 0
  of 27 times in run118 while the march filled the pack to 36/36 — and `junk_full` only suppresses
  `leave_nether` while `drop_junk` is actually on offer. An `[INVENTORY_FULL]` refusal triggers its own relief level
  (`blocked_by_inventory` → `inventory_relief`: chest, otherwise throw; a full chest switches
  `allow_stow` off). Base/chest are **per run** and are injected by the loop into every `GameState`.
  If the chest cannot stand at the spot — the server answers `[PLACE_NO_SUPPORT]` when no
  solid block borders the cell (live, Nether run: the base lay on a narrow edge and
  `place_chest` ran up in every tick instead of building the portal) — then
  `Agent._note_chest` notes this and drops the whole chest chain (`gs.chest_blocked`); the relief
  then falls back to `drop_junk`, which frees the same slots and cannot fail.
  **Occupied target cell (`[PLACE_TARGET_OCCUPIED]`, live in the from-zero run):** if the chest
  cell held a foreign block (grass), the server refused the build. `_note_place` writes the
  refused cell to `gs.blocked_cell`, and the unblock chooses exactly this cell: `make_room`
  digs it free, afterwards `place_chest` passes (it10 → it11 → it12). Before, the run ran into
  an endless retry on the same occupied cell.
  `Agent._adopt_homebase` adopts at tick start an already **standing** bed in reach
  (`ADOPT_BED_RADIUS` = 12 m) as anchor and a **chest** near the anchor (`ADOPT_CHEST_RADIUS` = 4)
  as `chest_pos` — so a restart continues at the same homebase instead of building a second
  (the chest is listed for that in `perception.CHAIN_NAMES` in the `view_blocks` scan). The bed is
  only adopted once its ground is proven to carry the frame: `Agent._scan_bed_site` reads the ground
  around the bed (its own level and four above, as `_scan_build_site` does for the bot) and
  `targets.bed_frame_site` applies `flat_build_site`'s criteria to the frame's own four columns — a
  bed the run laid underground or on a slope is **refused**, so `establish_base` anchors a site the
  frame can stand on (live, run17: the base was pinned nine blocks under the surface and the frame's
  top and western cells stayed stone, so `light_portal` answered `[PORTAL_NO_FRAME]` on 150 ticks).
  **Anchor from the standing portal (live, continuing run7c):** Bed adoption has a
  blind spot — the bed can stand far from the portal (here: underground at y=28), and `base`
  is per run only in memory. After a restart the bot stood 76 m from its own, already
  glowing portal, outside the 64 m resource reach, so `_portal_lit` was *false*:
  `establish_base` would have built a second base (and thus a second portal). When this was
  first found that was impossible only because `flint_and_steel` locked `mine_obsidian` and the
  obsidian was used up; now that the obsidian latch keys on the frame instead (see above), a
  duplicate portal is reachable rather than impossible, so adoption matters as much as ever.
  `Agent._adopt_portal_base` reads — as long as `base` is still empty and the kit
  stands — once a `nether_portal` **beyond** `RESOURCE_RADIUS`
  (`perception.scan_portal_corner`/`parse_portal_corner`, deliberately *without* a radius filter) and sets
  `base = corner − (2, 1, 0)`: The opening corner inverts `_portal_site`, so that the frame plane
  again lies on the standing obsidian. `gs.base_adopted` travels into the state and
  `_base_steps_ready` therefore answers `True`, so that `go_home` leads the bot back; on arrival
  (`_at_base`) the flag is cleared. `_scan_portal_opening` recognises an already glowing portal
  and does not report its blocks as a "foreign block in the opening".
  **Furnace at the base (live, feedback from the run):** `smelt_item` with `temporary_workstation`
  places the carried furnace *for that one call* and collects it again afterwards — while
  cooking in the lull (3 pieces per call) the bot thus kept tearing the furnace down between batches
  again. Therefore `place_furnace` places the carried furnace once at the base (cell
  `(bx, by+1, bz-1)`, next to the chest and clear of the portal frame plane `z == bz`), remembers the
  coordinates as `gs.furnace_pos` (`Agent._verify` notes them, `_adopt_homebase` adopts on
  restart a furnace near the anchor from the `furnace` scan in `CHAIN_NAMES`). `cook_food`,
  `smelt_gold` and `smelt_iron` then give `smelt_item` these coordinates and *without*
  `temporary_workstation`, so that the same furnace stays standing. The step runs only in the lull
  (`_nether_idle`) — before the gate frame stands the chain still needs the carried furnace for smelting
  away from the base —, and `_furnace_site` refuses the cell if a bed already stands there
  (the `[PLACE_TARGET_OCCUPIED]` unblock would otherwise dig away the respawn bed); if the floor
  cannot support the furnace (`[PLACE_NO_SUPPORT]`), the chain drops away (`Agent._note_furnace` →
  `gs.furnace_blocked`) and cooking uses the temporary build again.
  Tested in `tests/test_actions.py::HomeBaseTest`, `tests/test_actions.py::BaseFurnaceTest`,
  `tests/test_targets.py::BaseTargetTest`,
  `tests/test_graph.py::GuardUnblockTest`, `tests/test_graph.py::PortalBaseAdoptTest`,
  `tests/test_graph.py::FurnaceSupportTest`,
  `tests/test_perception.py::PortalCornerParseTest` and `tests/test_verify.py::HomeBaseProgressTest`.
