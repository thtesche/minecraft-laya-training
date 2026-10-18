# Nether preparation (pre-gate step)

Status: **phases 1 + 2 implemented** (bow + arrows, cobblestone reserve, gold helmet; safety
gates). Phase 3 (the actual Nether visit) is still open. Details below.

## Idea

Between "the frame stands" (`build_portal`, the gate still **unlit**) and "the bot enters the
Nether" a **preparation stage** is missing: the bot should only step through the gate once armour, tools,
food and building material really are right. Just as with eating (see `NETHER_FOOD_RESERVE = 64`
in [`architecture.md`](architecture.md)): better properly prepared than quickly in and
losing everything. The gate is lit **only once all of this stands** (`light_portal` requires
`_expedition_ready`: the prep done, the food reserved, a bed at hand, and it night or the bed
already at the base), so the portal never burns through the preparation window beside the base —
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
- **Cobblestone minimum reserve** raised to 64 in the lull (`NETHER_COBBLE`); the light threshold 12
  continues to apply outside the lull. Before the passage the bot carries a spare-pickaxe reflex
  (`replace_tool`).

**Phase 2 — done (gold helmet, carried ready):**
- **Gold helmet** (`mine_gold` → `smelt_gold` → `craft_gold_helmet`), one piece of golden armour keeps
  piglins neutral. Gold ore is found like diamond via `resource_found` (`gold_ore` in
  `perception.CHAIN_NAMES`) and mined via a `collect_block` tunnel; an iron pickaxe suffices.
- The helmet is kept **in the pack** ready; it is only worn on the passage (phase 3), because
  it occupies the same slot as the diamond helmet — decided: **the gold helmet replaces the diamond helmet**
  (`_swap_to_gold_helmet` on entry, see below).

**Phase 3 — open (the actual Nether visit):**
- **`enter_portal`**: navigate in (`dimension → nether`). This must **bypass the portal escape
  reflex** (`_in_portal` → `leave_portal`, see architecture): the rescue steps the body out of a
  lit gate, so the intended crossing needs an explicit intent the rescue respects.
- **Armour swap on entry:** diamond helmet off, gold helmet on (`_swap_to_gold_helmet`), so that
  piglins stay neutral. The diamond helmet stays in the pack for the way back.
- Build **no** shelter around the arrival portal (decided 2026-10-18, see (B)), secure the **way
  back**, then the first Nether resource (blaze/netherrack) as the goal of the following chain.

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
