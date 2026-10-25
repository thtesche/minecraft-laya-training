# Stronghold phase (the step after the Nether haul)

Status: **implemented and committed** — the `actions` package split landed first
(behaviour-preserving, `f09edd4`), then the stronghold chain on top of it (`34079f5`); the suite
is 990 green. `PHASE_STRONGHOLD` now has its real trigger in `phase_persona` (see Perception).
One open design point remains: the re-supply is not demand-sized (see the section near the end).

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

The server's `barter` tool can also buy pearls from piglins, but its own doc says hunting endermen
is usually faster; the overworld enderman hunt mirrors the existing `hunt_blaze` pattern and is the
first cut. Pearls are the only reason a stronghold run needs new gathering at all.

### Why no second Nether trip by default

The first expedition already targets nine rods / eighteen powder, which is more than the sixteen
powder sixteen eyes cost. So "top up blaze rods from the Nether if needed" is a *fallback*, not a
normal step: it reopens only when the banked powder plus carried powder cannot pay for the eyes the
phase still owes (a death on the way back, the chest raided). The relaunch latch the Nether phase
already grew (`_nether_haul_home`, `e812d83`) is the hook for reopening the expedition.

## Safety gates (carried over, plus new ones)

- **Endermen are dangerous** — they teleport and hit hard, and the hunt happens at night (when the
  run otherwise sleeps). So `hunt_enderman` follows the `_safe_blaze_hunt`/`_safe_spider_hunt`
  shape: a healthy bot (`>= BLAZE_HUNT_HEALTH`-style floor), not already in a fight, the enderman
  within a bounded range. **Never provoke an enderman the bot cannot win**; the eyes are not worth
  the diamond kit.
- **Sleep still owns the night** where no enderman is in safe reach: the pearl hunt does not replace
  the clock, it rides the same night the way `hunt_blaze` rides the Nether lull.
- **The respawn bed at the stronghold** — for the *later* End phase, `enter_end_portal` needs a bed
  within 128 blocks. When that phase lands, the chain lays and sleeps a bed at the stronghold before
  entering. This phase stops before that, so the bed is noted here only as the seam to the next one.

## The chain, in the order the bot walks it

Offered only once the Overworld goal and the Nether haul stand (`tech_tree_complete` and
`nether_expedition_complete`), so the stronghold work never competes with the tech tree:

1. **`hunt_enderman`** (`collect_mob_drop enderman → ender_pearl`) until `_pearls >= _eyes_goal`.
   Overworld, safe-hunt gated. Falls back to `explore` at night when no enderman is in reach.
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

## Known limitation: the re-supply is not demand-sized (review 2026-10-25)

The initial build is correct, but the *re-supply* after thrown eyes shatter is sized to fixed
constants, not to the eyes still owed:

- **The Nether top-up hauls a fixed nine rods / eighteen powder** (`BLAZE_RODS` / `BLAZE_POWDER`,
  read in `planner` at the `hunt_blaze`, `leave_nether`, `craft_blaze_powder` and
  `stow_blaze_powder` steps), whatever the deficit. A top-up fired to replace two broken eyes
  still sends the bot to kill nine blazes — each one "another fireball between the bot and its own
  portal", so needless risk to the diamond kit.
- **`hunt_enderman` targets a fixed sixteen pearls** (`ENDER_PEARLS`), not the eyes owed, so a
  re-supply over-hunts endermen the same way. (`take_blaze_powder` and `craft_ender_eye` *are*
  already `_eyes_owed`-sized, so the pattern is only half applied.)

Two related robustness gaps share the root cause:

- `locate_stronghold` is offered only at a full sixteen eyes (`_ender_eyes >= ENDER_EYES`), so an
  eye thrown in the `estimate` phase forces a rebuild to sixteen — possibly a whole top-up — before
  the `locate` travel can run.
- At the frame, `stronghold_candidates` returns *only* `activate_end_portal`; if heavy breakage on
  the way left too few eyes to fill the frame, there is no path back to craft or hunt more, so the
  phase can stall.

Fix sketch: a demand-sized target (`BLAZE_RODS` / `ENDER_PEARLS` normally, `ceil(eyes_short / 2)`
rods and `_eyes_owed - pearls` during a top-up), used wherever the Nether chain and `hunt_enderman`
read the constants; offer `locate` once there are enough eyes for the frame rather than the full
sixteen; and let the frame step fall back to procurement when eyes are short. Not yet done.

## Out of reach / later (deliberately not in this phase)

- **Entering the End and the dragon fight** — the next phase; the server tools exist, the bed-at-
  stronghold seam is noted above.
- **Bartering for pearls** — endermen are faster; `barter` stays a fallback only if the enderman
  hunt proves unreachable in a world.
- **Ender chests, blaze farms, enchanting** — outside this bot's tree, as in the Nether prep.
