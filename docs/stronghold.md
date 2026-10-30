# Stronghold phase (the step after the Nether haul)

Status: **implemented and committed** — the `actions` package split landed first
(behaviour-preserving, `f09edd4`), then the stronghold chain on top of it (`34079f5`); the suite
is 997 green. `PHASE_STRONGHOLD` now has its real trigger in `phase_persona` (see Perception).
The re-supply after shattered eyes is now demand-sized, and both robustness gaps are closed (see
the section near the end).

Scope of this phase, as the user set it: **gather eyes of ender, navigate to the stronghold and
remember it, activate the end portal, and top up blaze rods from the Nether only if needed.** It
stops at an *activated* end portal. Entering the End and the dragon fight are a later phase (the
server already carries `enter_end_portal`, `shoot_dragon`, `attack_dragon_perch`,
`destroy_end_crystal` for it).

## What the mine-ai server already does for us

Unlike the boat transport, the hard client mechanics are server tools — the agent only filters
valid options and books the call, exactly as everywhere else:

| User step | Server tool | What the tool handles itself |
|---|---|---|
| navigate to the stronghold, remember it | `locate_stronghold` | two phases: `estimate` saves an eye-of-ender triangulation and returns the approximate distance and recommended dragon-fight supplies **without travelling**; `locate` reuses the same `search_id`, travels there, refines with a local throw and confirms the loaded `end_portal_frame` blocks. Recovers dropped eyes; shattered eyes do not invalidate the bearing. The `search_id` is durable per bot+world and survives cancellation and restart. |
| activate the stronghold | `activate_portal` | validates the whole end_portal frame, fills the missing sockets from carried `ender_eye`, confirms every interior portal block, skips filled sockets and already-active portals (so a repeat after cancellation resumes). |
| (later) enter the End | `enter_end_portal` | requires a personal respawn bed within `respawn_within` (default 128) confirmed through this session's `sleep`; `allow_distant_respawn` bypasses only that guard. |

So the agent's work is procurement (eyes) + the phase/candidates/resolve wiring + the goal + the
durable latches — no new navigation or triangulation code.

## The resource that is actually new: ender pearls

An **eye of ender = 1 ender pearl + 1 blaze powder** (`craft_item ender_eye`). The arithmetic for a
target of **16 eyes** (12 to activate + a buffer for throws and for sockets that may be missing an
eye; each socket has a 10% chance of generating pre-filled, but 16 is the safe upper bound):

| Need | Count | Source | Already covered? |
|---|---|---|---|
| Eyes of ender | ≈16 | `craft_item ender_eye` | — |
| Blaze powder | 16 | one blaze rod → two powder | **yes** — the Nether expedition already banks 18 powder (`BLAZE_POWDER`), 2 to spare. The powder is *withdrawn from the gate chest* for crafting; a second Nether trip is only needed if the stock was lost. |
| Ender pearls | 16 | endermen (`collect_mob_drop enderman → ender_pearl`); enderman is already in `HOSTILE_HUNTS`. Overworld at night, or the Nether. | **no** — the one genuinely new chain. |

### Pearls from piglin barter (added 2026-10-28)

The server exposes `barter(piglin_id, item_name, count, gold_budget)`: the bot walks to a loaded
**adult** piglin and offers gold ingots one at a time until the requested net gain is in the pack or
the budget is spent. That is the route the user asked for, and it is now the pearl source the
expedition aims at: the reserve is dug in the lull before a pearl wave, smelted at the base furnace
(there is no coal to smelt with down there), carried through the gate as ingots, and traded in that
same wave - `leave_nether` only takes the tick once the pearl budget is spent or the roam is out of
ground to cover.

The arithmetic the reserve is sized against is unforgiving. A piglin rolls once per ingot, the pearl
entry of that roll is about 2% for a stack of 2-4, so a barter nets roughly **0.06 pearls per ingot**
and sixteen pearls costs on the order of **250 ingots**. `GOLD_BARTER_RESERVE` (16) is therefore a
first experiment, not a supply: it buys about one pearl. The enderman hunt stays as the fallback for
whatever the gold cannot pay for.

Two things the implementation had to learn: `barter` demands an exact loaded adult `entityId`, which
`nearby.mobs` rows carry under `nearest` (perception now keeps those ids in `GameState.piglins`,
adults only - the server refuses a baby), and a piglin has to be within `BARTER_PIGLIN_RANGE` (32 m)
for the trade to be worth a walk, because `barter` navigates to the piglin itself. Beyond that range
the bot marches the frontier instead of chasing one mob - the same off-base drift that stalled the
Overworld pearl hunt (run110).

The reserve is deliberately *not* part of the crossing gate: `_gold_reserve_steps` rides on top of
`_all_nether_prep_steps` rather than inside it, so a bot short of barter gold still walks through the
gate. The helmet is the piglin pass; the reserve is only the pearl budget.

And the route is capped: `BARTER_MAX_ATTEMPTS` (2) barter calls per run, counted by
`graph._note_barter` and read by `_gold_reserve_short` and `_barter_pearls_wanted`. Without the cap
the lull would keep digging gold for a trade that keeps missing until the run ran out of
iterations; once the cap is reached the enderman hunt is the pearl source again.

### The pearl waves (added 2026-10-28)

A death in the Nether drops the whole pack, so the pearl hunt is not bolted onto the rod trip - it
runs in **waves** that come home in between (the user's structure, reversing commit 584a51a, which
had put the barter inside the rods trip because "the bot is already down there"):

- **Welle 1 - rods only.** The expedition banks the blaze powder and closes (`powder_stowed` →
  `nether_expedition_complete`). Nothing else is attempted down there: no barter, no endermen. The
  haul is safe at home before any risk is taken again.
- **Welle 2+ - gold, barter, endermen.** `_nether_pearl_wave_wanted` reopens `nether_expedition_open`
  the way `_nether_topup_wanted` does, and the Nether block in `planner.candidates` splits on
  `_pearl_wave_trip`: the pearl ladder is `barter_pearls` → `hunt_enderman` → `search_enderman` →
  `leave_nether`. The gold for the trade is dug in the lull *before that wave*, not banked ahead of
  the rod trip (`_gold_reserve_short` is gated on `nether_expedition_complete`).
- **The cap.** `graph._note_dimension` counts a crossing as a pearl trip only once the powder is
  stowed and the bank is not exhausted (a powder top-up is a rod trip, not a pearl trip). After
  `PEARL_WAVE_MAX_TRIPS` (5) crossings the waves shut and the Overworld `search_enderman` roam
  becomes the fallback, so the goal is never stranded on an empty Nether.

The roam is bounded in metres, not ticks, like the fortress hunt: `NETHER_PEARL_SEARCH_M` (400 m) of
ground covered. Endermen spawn in the Nether's warped forests, not at the fortress the rod trip
targeted, so an empty stretch has to send the bot home to stow and relaunch rather than burn the trip
wandering.

### Why no second Nether trip by default

The chain is rods → powder → pearls → eyes → frame, so the haul is sized on the eyes the frame still
owes, not on the nine-rod goal: sixteen eyes owed means eight rods and sixteen powder, and a frame
that needs nine sockets needs five rods. So "top up blaze rods from the Nether if needed" is a
*fallback*, not a normal step: it reopens only when the powder the chain can reach (carried plus
banked) cannot pay for the eyes it owes (a death on the way back, the chest raided). The relaunch
latch the Nether phase already grew (`_nether_haul_home`, `e812d83`) is the hook for reopening the
expedition.

## Safety gates (carried over, plus new ones)

- **Endermen are dangerous** — they teleport and hit hard, and the hunt happens at night (when the
  run otherwise sleeps). So `hunt_enderman` keeps the health floor (`ENDERMAN_HUNT_HEALTH`) and the
  bounded range of the `_safe_blaze_hunt` shape — but **not** its "no other hostile crowding the
  fight" gate. Endermen spawn at night alongside the usual mob pack, so that gate almost never
  cleared in the dark and the hunt stalled at the base to the run limit (run109 it312-599); the
  diamond kit and shield are expected to carry a crowded pearl fight.
- **No enderman in reach → `search_enderman`, not sleep.** `hunt_enderman` is reactive (it needs an
  enderman already loaded within range, the way `hunt_blaze` needs a loaded fortress). The dedicated
  `search_enderman` roam (`explore_frontier` along the remembered frontier, same as the Overworld
  `explore` fallback) sends a healthy bot out to load fresh ground until one comes into range, then
  the hunt takes the tick. Without it a bot that banked its powder sat on `wait` to the run limit
  (run109). A wounded bot (`< ENDERMAN_HUNT_HEALTH`) does neither and heals at the base first. The
  roam is dimension-blind now, and the dimension decides its budget: in a pearl wave it is bounded by
  `NETHER_PEARL_SEARCH_M`; in the Overworld it only runs once the waves are spent.
- **The respawn bed at the stronghold** — for the *later* End phase, `enter_end_portal` needs a bed
  within 128 blocks. When that phase lands, the chain lays and sleeps a bed at the stronghold before
  entering. This phase stops before that, so the bed is noted here only as the seam to the next one.

## The chain, in the order the bot walks it

Offered only once the Overworld goal and the Nether haul stand (`tech_tree_complete` and
`nether_expedition_complete`), so the stronghold work never competes with the tech tree:

1. **`hunt_enderman`** (`collect_mob_drop enderman → ender_pearl`) until `_pearls >= _eyes_goal`.
   Overworld, safe-hunt gated (health floor + range, crowding tolerated). When no enderman is in
   reach it hands off to **`search_enderman`** (`explore_frontier`), which roams new ground until
   one loads; the two never offer together (`_search_enderman_wanted` excludes the in-reach case).
2. **`take_blaze_powder`** — withdraw the banked powder from the gate chest into the pack when the
   carried powder is short of the eyes the phase owes (the mirror of `stow_blaze_powder`).
3. **`craft_ender_eye`** (`craft_item ender_eye`, pearl + powder, needs a crafting table) until
   `_ender_eyes >= _eyes_goal`.
4. **`locate_stronghold`** — `phase: estimate` first (saves the throw, no travel), then
   `phase: locate` with the same `search_id` (travels, refines, confirms the frame). One label; the
   phase is chosen by a loop latch (`gs.stronghold_estimated`, `_note_locate`). The frame is read
   straight off the scan (`_at_stronghold`; `end_portal_frame` is in `CHAIN_NAMES`/`NAMED_RESOURCES`)
   and its cell comes from `_frame_cell`, so a cancelled locate simply resumes — no frame latch.
5. **`activate_end_portal`** (`activate_portal` at `gs.stronghold_frame`) — fills the sockets from
   the carried eyes and confirms the active portal.

## Goal

`stronghold_complete(gs)` — the end portal is active, latched on `activate_portal`'s own success
(`gs.end_portal_active`, `Agent._note_portal`, the same shape as `powder_stowed`); reading the
world back would cost an inspection every tick. `_goal_reached` becomes a three-way AND:
`tech_tree_complete ∧ nether_expedition_complete ∧ stronghold_complete`, keeping the existing rule
that a playthrough stops only when *every* phase stands, never the first one that closes.

## Perception

`phase_persona` returns `PHASE_STRONGHOLD` once `nether_expedition_complete` holds and the end
portal is not yet active and the bot is in the Overworld — the first real trigger for the stub. The
head is trained on exactly this rendering, so the sentence has to match the chain the bot is on.

## The `actions` package refactor (lands first)

`actions.py` is 4279 lines and is the file every phase edits; adding the stronghold chain to it
would push it past 5000. Before the feature, `actions.py` becomes a package, **behaviour-preserving**
(the 972-test suite is the gate, and no test monkeypatches `mlaya.actions` internals, so direct
cross-module imports are safe):

- `actions/constants.py` — the module imports, `CRITERIA`, `EDIBLE`, the tier/recipe tables, every
  tuning constant and the `Action` dataclass (current lines ~1–459).
- `actions/predicates.py` — the ~140 `_`-prefixed state predicates and the goal functions
  `tech_tree_complete` / `nether_expedition_complete` (current lines ~460–2822).
- `actions/planner.py` — `candidates()`, `resolve()`, `inventory_relief()` (current lines
  ~2823–4279): the churn surface for any new phase.
- `actions/stronghold.py` — the new phase's constants, predicates, `stronghold_candidates(gs)`,
  `resolve_stronghold(...)` and `stronghold_complete(gs)`; `planner` splices these in.
- `actions/__init__.py` — re-exports the full public **and private** surface of every submodule
  (a `dir()` copy), so `from mlaya.actions import <anything>` keeps working unchanged.

Each submodule re-exports its whole namespace via `__all__ = [n for n in globals() if not
n.startswith("__")]`, so a downstream `from .predicates import *` brings the private helpers too and
no call site is rewritten. Import graph: `constants → predicates → stronghold → planner`, no cycle.
Further splitting `predicates.py` is a possible follow-up, deliberately left for later to keep this
step's risk at zero.

## Resolved: the re-supply is demand-sized (review 2026-10-25, fixed `acd4795` + follow-up)

The initial build was correct, but the *re-supply* after thrown eyes shatter was sized to fixed
constants, not to the eyes still owed. All three parts are now fixed:

- **The Nether top-up is sized to the deficit.** `predicates._rods_target` / `_powder_target`
  return `BLAZE_RODS` / `BLAZE_POWDER` on the first trip (the top-up latch `_nether_topup_wanted`
  is off) and, during a stronghold top-up, `ceil((eyes_owed - carried_powder) / 2)` rods clamped to
  `[1, BLAZE_RODS]` with the matching powder. The Nether chain in `planner` (`hunt_blaze`,
  `leave_nether`, both `craft_blaze_powder` sites) reads these helpers instead of the flat
  constants, so a top-up for two broken eyes kills **one** blaze, not nine. The helpers live in
  `predicates` (not `stronghold`) because `predicates` cannot import `stronghold` without a cycle —
  `_eyes_still_owed` is a local copy of `stronghold._eyes_owed`.
- **`hunt_enderman` targets the eyes owed.** `stronghold_candidates` offers it while
  `_ender_pearls < _eyes_owed(gs)`, and the resolver asks for `_eyes_owed - pearls`, so a re-supply
  no longer over-hunts a dozen teleporting endermen. (`take_blaze_powder` and `craft_ender_eye`
  were already `_eyes_owed`-sized.)
- **A shaved eye after the estimate no longer forces a rebuild.** `END_PORTAL_SOCKETS` (12) is the
  locate-travel floor: the buffer is still built to `ENDER_EYES` (16) before the first `estimate`
  throw, but once `stronghold_estimated` is set the frame needs only twelve, so an eye shattered on
  that throw lets the `locate` travel proceed instead of dragging the bot back to re-craft (and
  maybe back to the Nether). Both the candidate gate and the `locate_stronghold` resolver read
  this floor.
- **The frame no longer dead-ends.** `stronghold_candidates` still returns only `activate_end_portal`
  while the bot carries eyes, but if heavy breakage left it at the frame with **none**, it falls
  through to procurement (hunt pearls, walk back to the gate chest for powder via `take_blaze_powder`
  — `_chest_ready` is distance-independent — craft eyes) and `locate` (durable `search_id`) carries
  it back to finish the frame. `activate_portal` only latches `end_portal_active` on a *full* frame,
  so a partial fill stays not-done; each pass fills ≥ 1 of the twelve sockets and the fill persists,
  so the activate ↔ procurement loop is self-terminating.

## The demand is the frame, not the constant (2026-10-30)

Every step of the chain now sizes itself on one derived number, `predicates._eyes_owed`:

```
eyes owed = (12 - sockets already filled) + throw buffer - eyes in the pack
```

- **`_eyes_in_hand`** is the frame's empty sockets plus the `EYE_THROW_BUFFER` (4) — the buffer only
  while the `estimate` throw has not been made yet, since after it the frame is the only thing the
  eyes are for. `_eyes_owed` subtracts the eyes already carried. `stronghold.py` no longer keeps its
  own copy of this count; it reads the same function the Nether gates do.
- **The sockets already in the frame cannot be read.** `view_blocks` names the block, not its `eye`
  state, and `activate_portal` only reports that it skipped the filled ones. So `gs.frame_eyes` is
  booked from the `ender_eye` inventory delta of each `activate_portal` call (`Agent._note_portal`),
  capped at `END_PORTAL_SOCKETS`. Until the server confirms an open portal, twelve empty sockets is
  the honest upper bound: the chain never under-supplies, and it stops the tick the portal lights.
- **`_rods_target` / `_powder_target`** are `ceil(eyes_owed / 2)` and `min(BLAZE_POWDER, eyes_owed)`
  — two powder to a rod, one powder to an eye. `craft_blaze_powder` now grinds whatever rods the pack
  holds (`min(2 * rods, powder_target - powder)`) instead of waiting for nine.
- **`gs.powder_bank`** is the powder booked into the chest at the gate: seeded from the chest scan,
  then adjusted by every stow and withdraw delta. `powder_exhausted` is the world saying the bank is
  empty, and it wins over the book. The top-up trigger is the demand, not that latch: it opens
  whenever `eyes_owed > carried powder + banked powder`.
- **`_nether_haul_home` means "a haul the chain still needs and can actually bank".** A rod left over
  from an earlier run is supply the demand already counts, and a rod the pack cannot process (no
  crafting table, no chest) is dead weight, not a pending haul. run123 is the case this closed: one
  `blaze_rod` in the pack, no powder, sixteen eyes owed — the leftover rod kept the departure gate
  shut, `craft_blaze_powder` wanted nine rods before it would grind, and the bot spent ten `wait`
  ticks at the lit gate with a complete kit.

## Out of reach / later (deliberately not in this phase)

- **Entering the End and the dragon fight** — the next phase; the server tools exist, the bed-at-
  stronghold seam is noted above.
- **Bartering for pearls** — endermen are faster; `barter` stays a fallback only if the enderman
  hunt proves unreachable in a world.
- **Ender chests, blaze farms, enchanting** — outside this bot's tree, as in the Nether prep.
