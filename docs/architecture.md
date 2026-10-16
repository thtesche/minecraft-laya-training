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
| System 2 | OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` (`max_tokens=16384`) |
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
[4] Actions       (src/mlaya/actions.py)
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
    structural trigger (label not resolvable) -> OpenRouter chooses a valid alternative
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
  **Lull follow-up (`_nether_idle`, live-found):** As soon as the diamond kit stands **and** the portal
  burns, only the night bed is left open — the bot fell into pure `explore` during the day (live in
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
  shaft (live, run19). The same clause governs `sleep` at the base once the portal is lit (a bed
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
    After every death the graph resets the policy (`_reapply_after_death`).
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
    automatically reset to `base` (`_reapply_after_death`).
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
  **only** if a hostile is ≤ `DANGER_DEFER_M` m away or health < `DANGER_DEFER_HEALTH = 15`.
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
- **Graph/Actions** (`src/mlaya/graph.py`, `actions.py`): `TARGET_LABELS = {"explore"}`. `_decide`
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

## Next milestones

- **M3+ (open):** The **out-of-distribution** held-out split is now built (full run
  `run7d`): the zero-shot accuracy there is **0.934**, the fine-tune improves loss/ECE/Brier,
  but **not** the accuracy — the rows imitate the logged gate choice, and imitation cannot fix its
  errors. A real accuracy gain would need a stronger signal than
  imitation (e.g. verifier reward directly) or more/more diverse runs. Both `in_danger` classes
  are represented with the defer fix; the new upper bound only caps the distant `yes`.
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
  before `equip_armor` takes effect) and the shield in the off-hand — **and** the overworld goal stands: the
  portal is lit, a bed placed **at the base** beside it (`_bed_placed` *and* `not _bed_off_base`: a
  leftover bed merely *loaded* satisfies the first clause, so the run keeps working until
  `take_bed`/`place_bed` bring it home), **no Nether prep step left open**
  (`_all_nether_prep_steps`: bow+arrows, gold helmet, deep cobblestone) **and the food reserve
  banked** (`_food_reserve_met`, the 64 units from `_food_goal`). Prep steps like the
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
  stands **in the frame** (`has_fs and obsidian < 10 and _portal_frame_started`); natural obsidian
  next to the base no longer counts.
  Tested in `tests/test_actions.py::NetherGateTest` and `tests/test_verify.py`.
- **Water escape reflex (implemented):** `candidates` checks `gs.in_water and not gs.on_ground` as
  the very first thing (even before the M2d reactions) and then returns **only** `["escape_water"]` — here
  Laya does not choose, the state is a rescue. The loop tracks `gs.water_ticks` for this (`Agent._inject`)
  and holds the rescue back for `WATER_RESCUE_AFTER` = 2 decisions: "without ground nothing can move"
  applies only to work that needs **feet on the ground** — a body between two banks
  **swims**, and the server navigates that itself. Had the rescue triggered at the first water
  contact, it would have yanked the bot out of every crossing as soon as an action came back (**live run15**: the goal ran
  east into a flooded seagrass flat, the rescue ran it back — 155 rescue paths against
  30 diamond digs, 157 breath aborts, 0 diamonds). `resolve("escape_water")` navigates **first to the
  nearest dry, standable ground** (`targets.nearest_ground`, any level — the dry cell the
  swimming body can really reach may lie *below* the water line) with
  `dig`/`scaffold`; the homebase is **only the stopgap** when no ground lies within scan range.
  Reason: without ground under the feet the shared pathfinder breaks — a `collect_block` ran live
  370 blocks wide and a net 0.18 m — while `navigate` keeps working from the swimming posture.
  (**Live run8:** the bot drowned ten times or more because the reflex stubbornly navigated to the
  underground base — 70 m straight down, 300 m away — instead of to the nearby coast; the
  base-first rule was the error, not the model.) If the bot stands on the bottom (water trench, pond,
  a pour that does not flood it), the reflex does **not** engage. Tested in
  `tests/test_actions.py::EscapeWaterTest` and `tests/test_graph.py::WaterTicksTest`.
  **Stuck (`repeated_movement_failure`):** The server reports this stop when the body does **not**
  get anywhere on a re-issued route — the bot is physically enclosed.
  Live run10: in a 1-block-wide, flooded shaft, `navigate`, `collect_block` and
  `build_structure` all answered like this, and the forced `escape_water` rescue ran **endlessly** against it. That
  code now counts as `blocked_by_surroundings`, so the existing unblock ladder engages — dig a pocket
  (`make_room`), otherwise to open ground (`explore`) — instead of an endless loop without a
  Laya choice. (In the same case the server additionally rejected digging/building with `Underwater stance
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
  **Sleep sets the respawn:** `place_bed` (via `sleep`)
  is offered **only at night** — by day the bed would only be placed, the spawn point would stay unset.
  `_bed_available` also counts a **placed** bed (not just a carried one): after a
  manual base build or a restart the bed otherwise remains as a block and the loop would
  needlessly look for wool for a second one.
  **Bed only after the portal:** Before the lit portal the generic `sleep` at the
  base is **suppressed** (`_at_base` and *not* `_portal_lit`). The server places the bed "next to the
  bot", and at the base that is exactly the portal interior — `build_portal` cannot
  clear a bed there (`remove_wrong_blocks` answers "movement policy prohibits breaking this block"). Live
  the bed landed in the opening and `light_portal` answered in every tick with
  `[PORTAL_NO_FRAME]`. The base bed is `place_bed`'s job and waits for the *lit*
  portal; away from the base the night sleep stays unchanged **until the portal burns**. After that
  `sleep` is restricted to the base (`_at_base` *iff* `_portal_lit`): anywhere else it would pin the
  respawn point away from the gate, and a carried bed laid wherever the bot happened to stand is
  exactly the leftover `take_bed` then had to fetch. A bed standing **away from the base** — the
  respawn bed of an earlier base, or of an earlier run in the same world — is taken up in the lull
  (`take_bed` → `collect_block {block_name: "white_bed", count: 1}`, only once the portal is lit)
  and laid again at the base by `place_bed`, so the gate bed ends up where it belongs. The
  auto-stop insists on it too: `tech_tree_complete` adds `not _bed_off_base`, so the run cannot
  end with the gate bed still sitting in an earlier run's shaft (live, run19: the goal read
  complete with the run17 bed at y=55). Live, run18:
  the run17 bed sat at y=55 inside the shaft the new surface base dug for gold, and every
  `collect_block` stumbled on its half-block top (feet 55.56, no settled arrival) while `mine_gold`
  re-dug the same shaft.
  A **chest** (8 planks, 3×3) holds surplus: `craft_chest` → `place_chest` (coordinates are
  remembered) → `stow_surplus` (`use_container deposit`) at ≥33/36 slots; `drop_junk` is only the
  stopgap without a chest. An `[INVENTORY_FULL]` refusal triggers its own relief level
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
  (`_nether_idle`) — before the portal the chain still needs the carried furnace for smelting
  away from the base —, and `_furnace_site` refuses the cell if a bed already stands there
  (the `[PLACE_TARGET_OCCUPIED]` unblock would otherwise dig away the respawn bed); if the floor
  cannot support the furnace (`[PLACE_NO_SUPPORT]`), the chain drops away (`Agent._note_furnace` →
  `gs.furnace_blocked`) and cooking uses the temporary build again.
  Tested in `tests/test_actions.py::HomeBaseTest`, `tests/test_actions.py::BaseFurnaceTest`,
  `tests/test_targets.py::BaseTargetTest`,
  `tests/test_graph.py::GuardUnblockTest`, `tests/test_graph.py::PortalBaseAdoptTest`,
  `tests/test_graph.py::FurnaceSupportTest`,
  `tests/test_perception.py::PortalCornerParseTest` and `tests/test_verify.py::HomeBaseProgressTest`.
