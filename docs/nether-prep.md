# Nether preparation (pre-gate step)

Status: **phases 1 + 2 implemented** (bow + arrows, cobblestone reserve, gold helmet; safety
gates). Phase 3 (the actual Nether visit) is still open. Details below.

## Idea

Between "the frame stands" (`build_portal`, the gate still **unlit**) and "the bot enters the
Nether" a **preparation stage** is missing: the bot should only step through the gate once armour, tools,
food and building material really are right. Just as with eating (see `NETHER_FOOD_RESERVE = 64`
in [`architecture.md`](architecture.md)): better properly prepared than quickly in and
losing everything. The gate is lit **only once all of this stands** (`light_portal` requires
`_expedition_ready`: the prep done, the food reserved, a bed at hand; no hour), so the portal never burns through the preparation window beside the base —
the bot used to walk into its own lit gate there (live, run22). The bed is laid the very next tick
after lighting, once the opening holds the portal's own blocks; laid beside the *unlit* frame it can
fall into the still-open opening, which `clear_portal` cannot dig out.

Preparation runs in **the same daytime lull** as eating (see `_nether_idle`) and is
now part of the **auto-stop**: `tech_tree_complete` additionally requires that no preparation
step is still open (`_all_nether_prep_steps`) **and the food reserve is banked**
(`_food_reserve_met` → the `NETHER_FOOD_RESERVE = 64` from `_food_goal`, see below). That way
the run prepares itself fully instead of ending with the bed on the first night. Each step drops out of the
list as soon as it gets no further (no spawn, no ore), and the food demand drops away as
soon as no animal and no zombie is in reach — so an unreachable item never blocks the run.

The Nether is a different game from the overworld:

- **Hunger falls faster** (sprinting, fights), and **death costs the whole inventory** —
  hence the large food reserve.
- **Ghasts destroy blocks**, the portal frame too → flint and steel to relight, cobblestone
  as protection.
- **Water evaporates**, **beds explode**, **clocks/compasses spin**.

The step has two halves: **(A) prepare the inventory** (before the passage) and **(B) after
arrival** (shelter, way back). This design covers both, but prioritises the inventory
preparation, because it hangs off the existing chain.

## (A) Optimal Nether inventory — reconciled with what the bot can do

Legend: ✅ present · 🟡 feasible (proposal) · ⛔ outside this bot's reach.

| Recommendation (hotbar) | Bot state | Assessment |
|---|---|---|
| Sword (diamond) | ✅ `craft_diamond_sword` (worn) | done |
| Pickaxe | ✅ `craft_diamond_pickaxe` (worn) | done |
| Shield | ✅ `craft_shield` + `equip_shield` (offhand) | done |
| Food (steak) | ✅ 64 × cooked meat (`NETHER_FOOD_RESERVE`) | done |
| Building blocks (cobblestone) | ✅ `NETHER_COBBLE = 64` in the lull | done |
| Bow + arrows | ✅ `craft_bow` + `craft_arrows` (≥ 16) | done — server fires |
| Axe / elytra | – / ⛔ | axe unnecessary (no wood cutting in the Nether), elytra = End |
| Fire resistance potion / water bucket | ⛔ | chicken-and-egg or evaporates — see below |
| Firework | ⛔ | needs elytra + gunpowder |

| Recommendation (offhand/inventory) | Bot state | Assessment |
|---|---|---|
| Golden armour against piglins | ✅ `mine_gold` → `smelt_gold` → `craft_gold_helmet` (ready) | done (worn only in the Nether) |
| Flint and steel (to relight the portal) | ✅ `craft_flint_and_steel` (carried) | done |
| Totem of undying | ⛔ | evoker/raid |
| Shulker box | ⛔ | End city |
| Full armour (diamond) | ✅ all 4 pieces worn | done (enchanting ⛔) |

### The bow is the actual gap

The **server combat already fires the bow itself** (`survival.BASE_CHANGES["combat"]["bow"] = True`);
the bot therefore only has to *have* it and carry **arrows**. That makes this no new
combat AI, but pure procurement — just like eating:

- **Bow** = 3 sticks + 3 string. String comes from hunting spiders (`hunt_string`, target in the lull 3).
- **Arrows** = flint + stick + feather. Flint (`get_flint`, reopened after the flint and steel) and
  feathers (`hunt_feather`, chickens) are part of the chains.

The server scaffold already knows `netherrack`/`basalt` (`scaffold_blocks`), so the
Nether terrain is already provided for in the server reflex.

### Why fire resistance & water bucket do not work

- **Water bucket** evaporates in the Nether — a reserve of it is pointless.
- **Fire resistance potion** needs a brewing stand and **blaze powder from the Nether** — that is a
  **chicken-and-egg**: the potion can only exist *after* the first Nether visit. So not a pre-Nether
  item; it becomes relevant at the earliest after the first blaze route.
- **Enchanting** (fire protection/protection) would need an enchanting table + lapis + XP — the bot
  never enchants.

## (B) After arrival

1. **No shelter around the arrival portal** — decided 2026-10-18, the portal stays **unprotected**.
   The first Nether visit does not build a cobblestone shell around the exit portal (the portal as
   the door). Instead the bot keeps the way back clear, treats the exit as a hot spot it uses to
   refuel and get out, and spends the cobblestone stack on bridging instead. Revisit only if ghasts
   break the exit apart in practice.
2. Keep the **way back** open (light/secure the portal first).
3. **Safety gate in the Nether** (see below).

## Safety gates (implemented with the step)

- **Never sleep / place a bed in the Nether.** Beds explode there. `sleep`, `place_bed` (and
  `fill_bucket`, since water evaporates) are restricted to the overworld via `_overworld(gs)`
  (`gs.dimension`; `apply_policy` automatically reapplies the server policy on a dimension change).
- **Check food and cobblestone** before the passage (minimum reserve), so that the bot does not "by
  accident" walk in with half an inventory.
- **Gold dig only during the day:** the deep dig for gold is not on offer at night (the run
  sleeps then), but stays open for the auto-stop — otherwise the night would end the run too early.

## Implementation in stages

**Phase 0 — done (reference):** diamond kit, shield, flint and steel, 64 food, cobblestone.

**Phase 1 — done:**
- **Bow + arrows** (`craft_bow`, `craft_arrows`, `hunt_feather`): bow = 3 string (spiders,
  `hunt_string`) + 3 sticks; arrows = flint (`get_flint`, reopened after the flint and steel) +
  stick + feather (chickens). Goal: bow present ∧ ≥ 16 arrows (`NETHER_ARROWS`). The server fires
  the bow itself (`combat.bow`).
- **Cobblestone minimum reserve** raised to 64 in the lull (`NETHER_COBBLE`), refilled only once
  it falls below 56 (`NETHER_COBBLE_SLACK`), so a scaffold block does not send `mine_stone` to
  collect a pile just dropped; the light threshold 12 continues to apply outside the lull. Before the passage the bot carries a spare-pickaxe reflex
  (`replace_tool`).

**Phase 2 — done (gold helmet, carried ready):**
- **Gold helmet** (`mine_gold` → `smelt_gold` → `craft_gold_helmet`), one piece of golden armour keeps
  piglins neutral. Gold ore is found like diamond via `resource_found` (`gold_ore` in
  `perception.CHAIN_NAMES`) and mined via a `collect_block` tunnel; an iron pickaxe suffices.
- The helmet is kept **in the pack** ready; it is only worn on the passage (phase 3), because
  it occupies the same slot as the diamond helmet — decided: **the gold helmet replaces the diamond helmet**
  (`_swap_to_gold_helmet` on entry, see below).
- **Barter reserve** (`mine_gold_reserve` → `smelt_gold`): once the helmet is banked, the lull digs
  `GOLD_BARTER_RESERVE` (16) more ingots to trade with piglins for ender pearls (docs/stronghold.md).
  It is offered in the same window but is *not* part of the crossing gate — `_gold_reserve_steps`
  rides on top of `_all_nether_prep_steps`, so a bot short of barter gold still walks through the gate.

**Phase 3 — done (the first Nether visit: nine blaze rods, eighteen powder):**

The goal the user set (2026-10-18): find and kill blazes in the Nether, take **9 blaze rods** home
(→ **18 blaze powder**, one rod to two), and park the powder in the **chest at the Overworld
portal** for now. Getting back out is part of the goal, not a detail.

- **`enter_portal`** (`enter_nether_portal`): the crossing itself. It must **bypass the portal
  escape reflex** (`_in_portal` → `leave_portal`, see architecture): the rescue steps the body out
  of a lit gate, so the intended crossing needs an explicit intent the rescue respects. That is
  `gs.enter_intent` — a **countdown** (`PORTAL_INTENT_TICKS`), not a latch: the crossing action
  blocks until it settles and the teleport needs a moment after, and a gate the bot merely walked
  into later must still be rescued. Offered only with the whole expedition stocked
  (`_expedition_ready`) and never twice (`gs.nether_entered`).
- **Armour swap on entry:** diamond helmet off, gold helmet on (`swap_to_gold_helmet`, one `equip`),
  so piglins stay neutral. The diamond helmet stays in the pack, and `swap_to_diamond_helmet` puts
  it back on once the bot is home.
- **`hunt_blaze`** (`collect_mob_drop` on `blaze` → `blaze_rod`): only a blaze within
  `BLAZE_HUNT_RANGE` of a healthy bot that is not already in a fight (`_safe_blaze_hunt`, the
  `_safe_spider_hunt` guard — blazes ignite themselves when struck and answer with fireballs).
  Without a blaze in reach the bot walks the fortress bricks instead: the Nether scan group
  (`nether_bricks`, `nether_brick_fence`, `nether_brick_stairs`, `spawner`) is read **only** in the
  Nether and is what gives `explore` a heading rather than a random walk.
- **`leave_nether`** (the way out): the exit portal when a `nether_portal` block is loaded, and
  otherwise a walk back to `gs.nether_arrival` — the cell the bot landed on, where the exit portal
  stands. Offered when the nine rods are in, **and also** when the bot cannot afford to stay
  (`_nether_must_return`: badly hurt, starving, in lava, or so full the next drop has nowhere to go)
  or when there is nothing left to find (`_nether_hopeless`: no blaze, no fortress anywhere).
  A short haul still becomes powder.
- **Home:** `craft_blaze_powder` (9 rods → 18 powder, Overworld only — a temporary workstation in
  a fortress is a bad trade) and `stow_blaze_powder` (deposit into the chest at the gate).
- **Goal:** `nether_expedition_complete` — the powder standing in the chest, latched by
  `Agent._note_stow` on the deposit's own success (the same shape as `frame_complete`; reading the
  chest back would cost an inspection every tick). Powder still in the pack does **not** count:
  the point is that the haul is safe at home, and one death on the way back takes the inventory.

**Live status (run86, 2026-10-22):** the first crossing of the October series. The gate was lit
at it112, the bot waited at the base for the night, crossed with `enter_portal` at it420 and found
a fortress within four decisions — at it425 it stood beside a blaze spawner at (-24,70,-54) in the
Nether. The server's hostile reflex killed blazes there, but no rod reached the pack: after
20:25:33 the reflex fell silent while blazes stayed within 4 m, and the loop deferred for the rest
of the run (fixed by the quiet-defer guard, `e0e7687`, see architecture/M3). Two gaps showed:
`swap_to_gold_helmet` was offered on every Nether tick but never ran (the defers took them all),
and `hunt_blaze` was not offered at all, because `_safe_blaze_hunt` refuses while any other
hostile is within 16 m — at a fortress a zombified piglin or a wither skeleton almost always is.
The ~300 iterations the bot spent waiting for the night are gone: `place_bed` now lays the bed by
day as well, and the `sleep` server sets the respawn point at a bed used in daylight, so the
crossing follows the lit gate at once.

**Live status (run93, 2026-10-24):** the crossing followed the lit gate the same day — frame
it95, `light_portal` it163, `enter_portal` it164 — and **nine blaze rods reached the pack**
(it167/207/216), the first rods of the series. None came home: the bot died fighting a blaze
before the return (32.5,53,84.3), losing the rods, the diamond kit and the food. Two gaps closed
since: a wounded, foodless bot in the Nether now recovers with `leave_nether` instead of falling
through to `explore` (`985d420`), and a death return reopens the expedition instead of a ~88-tick
`wait` loop — `_nether_haul_home` replaced the one-way `_nether_entered` latch at the relaunch
gates (`e812d83`, see architecture/run93). The gold helmet is still not worn on crossing: it was
crafted (it116), but every swap attempt landed while the reflex held the body.

**Live status (run104–107, 2026-10-26 … 27):** the fortress search and the walk home matured; the
powder is still not banked. run104 spent 438 explore ticks, 112 of them on one magma-cube spawner
at (94,39,86) with no blaze ever loaded — a landmark underfoot was treated as an explore target and
a reached cage was written off at 0 m moved (`d763e44`); the gold-helmet swap also lost 52 of 66
Nether ticks to `explore` and now owns the tick outright (`89c3a71`). run105 closed the reverse
case: a landmark the walk *reached* was written off even when the step moved, so the bot
ping-ponged 94↔150↔207 instead of re-navigating (`8e7d824`). run106 was the closest yet — **12
rods carried out** — but the run ended on `body_held`: two `leave_nether` ticks held 300 s each at
the spawner under the hostile reflex, so the walk home never started; the flee stage now raises the
pursuit floor past full health so the reflex evades and frees `navigate`, and a hold that healed
the bot or brought in a rod no longer ends the run (`38c4e8c`). run107 lost the whole kit to a
Nether death (it131) and then ping-ponged `explore`↔`go_home` at the base until it nearly starved,
because the nearest animal sat past the day-hunt range cap; below `HUNT_STARVE` the cap now drops so
recovery reaches the far animal (`e902b7e`). So the crossing, the fortress and the rod haul all run
live, but no run has yet carried the powder back to the chest at the gate.

## Decided: the gold helmet replaces the diamond helmet

One piece of golden armour keeps piglins neutral, but the bot wears a **full diamond set** —
the gold helmet occupies the same slot as the diamond helmet. **Decision (agreed with the user):
on entering the Nether the diamond helmet is taken off and the gold helmet worn** (piglin
neutrality is worth the somewhat lower head protection). The diamond helmet stays in the pack and goes
back on on the way back to the overworld (phase 3, `_swap_to_gold_helmet`). Until then the bot
wears the diamond helmet and only carries the gold helmet.

## Out of reach (deliberately not built)

Elytra + firework (End/gunpowder), totem (evoker/raid), shulker box (End), fire resistance potion
(Nether — chicken-and-egg), enchanting (table/lapis/XP), golden carrots (farming + gold),
crossbow (like a bow, but more involved — a bow suffices).
