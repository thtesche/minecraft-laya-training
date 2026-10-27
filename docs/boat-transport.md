# Boat transport — feasibility study

Status: **study only**, nothing implemented. Question: can a Mineflayer-side plugin give the bot
boat travel, exposed either as a new navigation kind (`navigate_water`, when the caller knows the
route crosses water) or chosen automatically by `navigate`?

**Short answer: feasible, medium effort.** Placing, boarding and leaving a boat are already
nearly there in Mineflayer. The real work is that **a boat is steered by its client, not by the
server**: Mineflayer cannot move a boat today, so the plugin has to simulate the boat itself and
report its position every tick. On top of that comes a small surface-water planner and the
board/leave/pick-up choreography. Recommended path: a time-boxed spike (does the server accept
our simulated boat?), then an explicit `navigate_water` tool, and automatic choice inside
`navigate` only after that runs reliably.

## Why bother

- **Swimming is slow and dangerous.** The server's own swim pricing is
  `SWIM_HORIZONTAL_TICKS = 14` per block (`mine-ai-mcp/src/navigation/world/swimming.ts`), about
  1.4 m/s. A boat on open water reaches roughly 0.36 blocks/tick (≈ 7 m/s, see below): about
  **five times faster than swimming** and faster than sprinting (4 ticks/block).
- **Water has cost this agent repeatedly.** The archived decision logs hold **603 `escape_water`
  decisions**; run8 drowned treading water, run15 stranded in a flooded flat, the base-range
  latches once left it in water for 76 ticks, and run89 was killed by a drowned
  (see the comments around `WATER_RESCUE_AFTER` in `src/mlaya/actions/`). A seated body does
  not lose breath on the surface and is not pushed under.
- **Exploration stops at coasts.** `explore`/`view_frontier` already know `surface_water`, but an
  ocean or a wide lake is effectively a wall for a walking bot, or a long swim at night.

## What exists today

### Mineflayer (fork `aibengineering/mineflayer`, 4.39.0, game 1.21.4)

| Piece | State | Finding |
|---|---|---|
| `bot.placeEntity` for boats | 🟡 broken on 1.21.4 | `place_entity.js` maps every `*_boat` to the entity name `boat`, but since 1.21.2 each wood is its own entity (`oak_boat`, `spruce_boat`, …, `bamboo_raft`). The spawn listener also unsubscribes after the **first** entity spawn of any kind. Worse, on the official **npm 4.39.0** (2026-09-06) the boat's `use_item` lacks the `rotation` field 1.21.2 added, so the serializer throws and **disconnects the bot** — fixed on master by [PrismarineJS/mineflayer#4085](https://github.com/PrismarineJS/mineflayer/pull/4085) (merged 2026-09-21, not yet released). The plugin places with `lookAt` + `activateItem` and finds the boat itself. |
| `bot.mount(entity)` | ✅ | a plain `use_entity` interact; `bot.vehicle` and the `mount`/`dismount` events are maintained in `entities.js`. |
| `bot.dismount()` | ⛔ on 1.21.2+ | sends `player_input { jump }`, but vanilla leaves a vehicle on **sneak**; the server never reads the jump for this. Upstream fix open: [PrismarineJS/mineflayer#4093](https://github.com/PrismarineJS/mineflayer/pull/4093) (holds sneak until the server reports the dismount). Until then the plugin sends the sneak itself. |
| `bot.moveVehicle(left, forward)` | ⛔ for boats | only sends `player_input`. That steers server-driven vehicles; a **boat ridden by a player is simulated by that player's client**, which reports the result with `vehicle_move`. Mineflayer neither sends `vehicle_move` nor handles the clientbound correction. |
| Physics while mounted | ⛔ | `physics.js` switches its own physics off on `mount`; prismarine-physics knows no boats, so `bot.entity.position` goes stale in a boat (open upstream: [#3615](https://github.com/PrismarineJS/mineflayer/issues/3615), and [#3726](https://github.com/PrismarineJS/mineflayer/issues/3726) asks for mounted control in general). |

The 1.21.4 protocol has everything needed: serverbound `vehicle_move {x, y, z, yaw, pitch,
onGround}`, `steer_boat {leftPaddle, rightPaddle}` (paddle animation only) and `player_input`.

### Prior art on GitHub (searched 2026-10-23)

**No reusable plugin exists**: no `mineflayer-boat` on npm, no Mineflayer boat or vehicle
repository. But at least five AI-agent projects have built the same thing *inside* their bot, and
all of them took the approach this study proposes — simulate the boat on the client and send
`vehicle_move` every tick:

| Project | License | What it shows |
|---|---|---|
| [mrubens/jevcraft](https://github.com/mrubens/jevcraft) `src/boats.js` | MIT | the most complete: vanilla boat tick (0.9 drag, 0.04/0.005 thrust, inertial turning, buoyancy), BFS route over source water with the 1.375 box sampled per cell, dock search on both shores, braking, abort on the first server correction, boat recovery by attack + pickup, dismount by sneak; land legs via `mineflayer-pathfinder` |
| [Im-neko/AILoveShen](https://github.com/Im-neko/AILoveShen) `minecraft-bridge/src/boat.mjs` | none | kinematic model at 0.35 blocks/tick; its comment says **Paper 1.21.4 accepted it without position corrections** (tested 2026-09-27) |
| [rmalde/minecraft-agent](https://github.com/rmalde/minecraft-agent) `boat.mjs` | none | compact client prediction, speed capped at 0.30, resync from clientbound `vehicle_move`, launch site by line of sight |
| [thegauravgiri/minecraft-agents](https://github.com/thegauravgiri/minecraft-agents) `src/skills/boat.js` | MIT | crafts the boat from the plank type on hand, refusal reasons like `no_boat` |
| [wehos/mc-agent-neko](https://github.com/wehos/mc-agent-neko) `skills/boatCross.js` | MIT | mindcraft fork; boat as deep-water rescue reflex, the same idea as the `escape_water` rescue below |

Consequences: the core risk of the spike (does the server accept a simulated boat?) is much
smaller than first assumed, and the MIT projects may be read and adapted with attribution; the
two without a license may only be read. The gap a standalone plugin fills is real: everyone has
rebuilt this privately.

### mine-ai-mcp server

- Its **own navigation engine** (incremental A* + continuous route executor), no
  `mineflayer-pathfinder`; eleven movement kinds including `swim`. A boat does not fit the
  catalogue as one more edge: it changes the body (size 1.375 × 0.5625, no own physics, different
  speeds) for a whole leg.
- **Every action starts with `prepare-body.ts`, which dismounts** (`if (bot.entity.vehicle)
  bot.dismount()`). A boat trip therefore has to be **one foreground action from shore to shore**;
  it cannot be spread over several MCP calls.
- The survival reflexes (fight, footing, hunger) own the body when they fire; they assume a
  walking body today.

## The technical core: client-side boat kinematics

Because the client is authoritative, the plugin runs a small boat simulation at 20 Hz and sends
each result as `vehicle_move`. It does not need the full vanilla boat; it needs **positions the
server accepts**.

**Vanilla boat on water** (`Boat.java`, values to confirm in the spike): paddling forward adds
0.04 blocks/tick along the yaw, momentum on water is 0.9 per tick → terminal speed
0.04·0.9/(1−0.9) ≈ **0.36 blocks/tick**; turning adds ±1° to a rotation rate that decays with the
same momentum. The boat floats with its bottom just below the water surface.

**What the 1.21.4 server checks** in `handleMoveVehicle`:

1. *Moved too quickly*: a jump far beyond the vehicle's current motion (squared distance budget
   of 100) is rejected and the boat is sent back. Steps of ≤ 0.4 blocks per tick are far below it.
2. *Moved wrongly*: the server replays the step with its own collision; a deviation of more than
   0.0625 blocks sends a correction. Simulated positions must therefore respect block collision —
   staying on open water with the boat's box clear of banks is enough.
3. *Floating*: a vehicle with nothing but air around it for 80 ticks gets the player kicked
   ("flying is not enabled"). Water is not air, so a boat on water passes; a boat driven over a
   drop does not.

**Proposed kinematic model** (simpler than vanilla, chosen to be accepted rather than authentic):

- Speed ramps to a cruising speed below vanilla terminal speed (e.g. 0.3 blocks/tick), slows to
  0.1 in narrow or turning sections, and stops on the waypoint.
- Yaw turns toward the next waypoint at a bounded rate; the boat moves along its yaw.
- `y` is held at the water surface of the cell below (`surface + boat offset`), never integrated.
- Each candidate step is collision-tested against the boat's box (1.375 wide) before it is sent.
- A clientbound `vehicle_move` (server correction) resets the simulated state to the server's
  position; three corrections in a row abort the leg (`boat_rejected`).
- `bot.entity.position` is set from the boat each tick so perception, `view_status` and the
  combat code see the real body.
- `steer_boat` paddles are sent for honesty in other players' view; they are not needed for
  movement.

**Hazards the model must refuse or handle**: flowing water (pushes the boat; treat as cost, not
as forbidden), waterfalls and drops, bubble columns (pull or push the boat, eject below the
surface), ice (much faster physics — out of scope for v1), lily pads (broken by a boat — fine),
a boat dragged under water ejects its passenger after a few seconds.

## Route planning: a separate surface-water leg

The boat leg needs its own **2D A* over water-surface cells**, not the 3D movement catalogue:

- A cell is passable when its block is water (source or flowing), the block above is air, and the
  boat box fits: **corridors at least two cells wide** (1.375 does not pass a one-wide river or
  a one-block gap between banks); diagonal clearance checked like the land diagonals.
- Cost in ticks: distance / cruising speed, plus a penalty for flowing water against the heading
  and for each turn; the same tick units as the land heuristic, so legs can be compared.
- Only loaded chunks are planned; at ~7 m/s the boat outruns a LAN world's loading quickly, so the
  leg is re-planned as chunks arrive (the existing incremental search does the same on land).

**Composite trip** (one foreground action):

```
land leg      navigate to a launch cell: dry, beside a water-surface cell the boat fits in
launch        equip boat, place it on the water (custom placeEntity that knows oak_boat & co.)
board         mount, wait for the mount event, start the boat loop
boat leg      surface A*, kinematic loop, corrections, re-plan on new chunks
landing       stop beside a landing cell, dismount, confirm the feet are on dry ground
recover boat  attack the empty boat until it breaks, pick up the dropped boat item
land leg      navigate the rest of the way
```

If the boat cannot be recovered (drifted, broken), the trip still counts as arrived and the
result says `boat_lost`; one boat costs only five planks.

## API: explicit tool first, automatic choice second

**Option A — explicit `navigate_water`** (recommended for v1). Same target arguments as
`navigate` (`x`, `y?`, `z`, `range`), plus `keep_boat` (default true). Refuses with a reason when
no boat is carried (`no_boat`), no launch cell is near (`no_launch_point`), or the far shore has
no landing (`no_landing`). The caller — in this project Laya's resolver — decides when to use it.
Pros: no risk to the well-tuned `navigate`, clear refusals to learn from, testable alone.

**Option B — `navigate { vehicle: "auto" | "boat" | "none" }`.** `auto` lets the planner compare
a land/swim route with a boat route. A cheap rule is enough:

- consider a boat only when one is carried (or 5 planks + a reachable crafting step, later);
- estimate the water share of the straight line from the loaded surface map; below ~15 blocks
  of water, never;
- otherwise price both: land search result (or its failure) versus
  `land→launch + boat leg + landing→target + ~60 ticks overhead` (place, mount, dismount,
  recover); take the cheaper.

B is the convenient end state, but it couples the boat to every `navigate` call — including the
survival reflexes and `collect_block` approaches that must never board. It should come only after
A has live runs behind it, and default to `none` everywhere except `navigate`/`explore_frontier`.

## Where the code lives: a standalone plugin with three seams

Decision: the boat is a **standalone Mineflayer plugin** (working name `mineflayer-boat`), and
mine-ai-mcp keeps only a thin adapter. About four fifths of the work is plain Mineflayer and
belongs in the plugin; three things belong to whoever owns the body and the land routes, and the
plugin reaches them through seams instead of rebuilding them.

**In the plugin:**

- the boat kinematics (`vehicle_move` at 20 Hz, server corrections, `bot.entity.position` sync);
- placement that knows `oak_boat` & co., mount, dismount, breaking and picking the boat back up;
- the surface-water planner over `bot.blockAt` and the launch/landing search;
- the whole trip as one call: `await bot.boat.travel(target, options)`.

**Through seams, owned by the host (mine-ai-mcp here):**

| Seam | Why the host owns it | Shape |
|---|---|---|
| `walkTo` | the land legs need a pathfinder; mine-ai-mcp has its own engine, other hosts use `mineflayer-pathfinder` | `travel(target, { walkTo: (pos) => runtime.navigate(...) })` |
| `signal` | the `NavigationRuntime` arbitrates the body between navigation, reflexes and digging; a hostile or low health must be able to end the trip | an `AbortSignal`; on abort the plugin steers to the nearest shore, lands and resolves with `aborted` |
| `estimate` | only the host can price the land route it would compare against | `bot.boat.estimate(from, to)` → ticks (land→launch excluded), or `null` when no boat route exists |

**In mine-ai-mcp:** the `navigate_water` MCP tool (arguments, refusals, telemetry) and, in
phase B, the cost comparison inside `navigate`.

The price is some duplication: the plugin brings its own simpler collision and water tests
instead of the server's `block-geometry.ts`, and its own tests without the server (about two to
three extra days). In return it is reusable (also upstream at PrismarineJS), and the boat logic
stays apart from the server's tuned navigation.

## Integration into this agent (Laya side)

- **Tech tree:** a cheap `craft_boat` label (five planks of one wood, 2×3 shape — needs the
  crafting table, which the bot carries) once wood is plentiful; the boat is exempt from
  `stow_surplus` and from junk drops.
- **Use:** the resolvers that send the bot far — `explore`, `go_home`, the return to a storage
  cell — pick `navigate_water` when a boat is carried and the target lies across water (or the
  last `navigate` there failed with a water-related refusal). No new Laya decision is needed: it
  stays a *resolve* choice, consistent with "code filters, Laya chooses".
- **Rescue:** `escape_water` gains a better first move — a carried boat placed on the surface and
  boarded lifts the bot out of the water at once, then `navigate_water` to the nearest shore.
- **State:** the narrative gets "carries a boat" / "in a boat"; `verify` treats a completed boat
  trip like a `navigate` (position delta), and `boat_lost` as an inventory delta.
- **Nether:** no water, so the whole feature is Overworld-only (`_overworld(gs)` guard).

## Risks

| Risk | Weight | Mitigation |
|---|---|---|
| Server rejects the simulated movement (corrections, kick) | high, decides the study | spike first; conservative speed and collision-tested steps |
| Reflexes in a boat: combat, hunger, footing assume a walking body | medium | the boat action owns the body for the leg; a hostile in reach or low health → steer to the nearest shore and land, then hand back (attacking from the seat works in vanilla, but keep v1 simple) |
| `prepare-body` dismounts at the next action | low | the trip is one action; landing is part of it |
| Drowned / guardians in deep ocean | medium | prefer routes along the coast (cost for distance to shore); never at night in v1 |
| Chunk loading at boat speed | medium | plan only on loaded chunks, slow down at the loaded edge |
| Boat recognition across versions (`oak_boat` vs `boat`) | low | own placement helper keyed on the held item's name |
| Upstream fit | low | the plugin is its own `mineflayer-boat` package; the fork's `moveVehicle` stays untouched |

## Effort (rough, agent sessions)

| Step | Size |
|---|---|
| **Spike:** place, mount, drive a straight 50-block line with `vehicle_move`, dismount, recover — on the real LAN world | 1–2 days |
| Plugin core: kinematic loop, corrections, position sync, placement/recovery helpers | 3–5 days |
| Standalone packaging: own collision/water tests, seams (`walkTo`, `signal`, `estimate`), tests without the server | 2–3 days |
| Surface-water planner + launch/landing search | 3–4 days |
| `navigate_water` MCP tool, refusals, telemetry; mine-labs scenarios (lake crossing, river bend, one-wide river refused) | 2–3 days |
| Laya integration (`craft_boat`, resolver choice, rescue, tests) | 1–2 days |
| Option B (`vehicle: "auto"`) | +3–4 days, after live runs |

**Go/no-go for the spike:** the boat crosses 50 blocks of still water at ≥ 0.25 blocks/tick with
at most occasional server corrections and no kick, and the bot ends on the far shore with the
boat item back in the inventory.

## Open questions

1. Is a simplified boat model accepted on the LAN (integrated) server exactly like on a dedicated
   one? The integrated server skips some movement checks only for the world's owner, not for
   the bot. AILoveShen reports Paper 1.21.4 accepting 0.35 blocks/tick without corrections;
   the LAN world remains to confirm in the spike.
2. ~~Should the plugin live inside `mine-ai-mcp/src/navigation` or as a separate Mineflayer
   plugin?~~ Decided: separate plugin with three seams — see
   [Where the code lives](#where-the-code-lives-a-standalone-plugin-with-three-seams).
3. Is `bamboo_raft` (bamboo forests) and `*_chest_boat` worth supporting? v1: any carried boat,
   chest boats treated as plain boats.
4. Night travel and deep ocean: forbid in v1, or allow with a shore-distance cost?
