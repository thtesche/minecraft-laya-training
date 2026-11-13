# `tunnel_to` — design

Status: draft. Doc-only; nothing in `~/VibeCoding/mine-ai-mcp` changes until the user gives the word.

## 1. Problem

The End pipeline is `locate_stronghold` → `activate_portal` → `enter_end_portal`.
`locate_stronghold` confirms the loaded `end_portal_frame` blocks and finishes — but
when the stronghold is under deep ocean it confirms the frame from the water surface,
~100 blocks straight up, and the bot has no dry, walkable way down to it. `activate_portal`
assumes the bot can stand at the frame. That gap is where the chain breaks.

`tunnel_to` fills exactly that gap: **frame confirmed → bot standing in the portal room,
on a dry walkable staircase, having touched neither water nor lava.** The outward dig only;
the return path is deferred (see §9).

The technique is not speculative. A manual hand-pilot carved it end-to-end on seed
`97996358`: trek to the south shore, swim the ocean, carve a fully dry ~1:1 diagonal
staircase from y68 down to the portal room at the frame `(-270,-36,-1817)`, arrive at HP 20,
then climb back out. `tunnel_to` encapsulates that pilot as one action. See the project
memory `shore-diagonal-staircase-validated`.

## 2. Scope

Option **B**: a single self-contained action that owns decomposition, lava scan, auto-y,
ladder placement and the landmark write — rather than a thin orchestrator leaning on the
model to sequence `navigate`/`view_blocks`/`dig` calls (Option A). B keeps the dangerous
liquid-breach logic inside one guarded code path and makes the seeded test deterministic.

In scope now:

- Find the nearest dry shore to the confirmed frame.
- Navigate there (swimming the ocean is allowed; the staircase itself is dry).
- Carve a dry diagonal staircase from the shore down to the portal room.
- Place walkable infrastructure (ladders on any vertical run) so the staircase is
  two-way walkable for a later return.
- Leave the bot standing at the frame interior.
- Persist two landmarks (mouth, portal-exit) folded into the `locate_stronghold`
  `search_id` record.

Out of scope now (later extensions):

- **Return / homecoming.** Deferred — see the project memory `part2-return-path-deferred`.
- **Stronghold-under-land.** A pure vertical shaft with ladders, no shore hop. Parked;
  the trigger (§8) will branch to it later.
- **Food / supply gating.** Omitted deliberately. We accept the Normal-difficulty
  starvation HP floor (~10) and handle food with the End scope.

## 3. Trigger (where it plugs in)

Inside `locate_stronghold`, phase `locate`, after the local eye throw and frame
confirmation. Branch to `tunnel_to` when **all** hold:

1. The local eye throw descends ~vertically — `|Δy|` dominates the horizontal — i.e. the
   bot is essentially directly above the stronghold.
2. The bot is over or in water (feet/standing cell is water, or the surface column below
   is water).
3. The confirmed frame is deep below the bot (large positive descent, e.g. `> ~20` blocks).

When they hold, `locate_stronghold` has a confirmed frame but no footing. It hands the
confirmed frame position to `tunnel_to`. (The stronghold-under-land case fails condition 2
and is left for the vertical-shaft extension.)

This doc specifies `tunnel_to` as a standalone action with its own contract, callable
directly for the seeded test; the `locate_stronghold` branch is the production caller.

**Built (Option A — advisory next-step).** Rather than a compound call, `locate_stronghold`
stays a discrete step and recommends the next one, mirroring its existing `estimate→locate`
`next` hint. On a confirmed frame it sets a nullable result field

```ts
reachByTunnel: { searchId: string; target: Point; descentBlocks: number } | null
```

populated only when all three conditions hold: the loop's `verticalEyeDescent` flag (a local
throw that sank ~straight down), `overWater(bot)` (feet or the cell below is water), and
`descentBlocks > 20` (`TUNNEL_TRIGGER_DESCENT`). When set, `formatResult` appends an advisory
pointing the agent at `tunnel_to` with the shared `search_id`; the catalogue description notes
the handoff too. Null when the frame is reachable on foot (the under-land case), left for the
vertical-shaft extension. Covered by two unit tests (deep-water → flagged, dry-land → null) in
`locate-stronghold.test.ts`; the actions stay decoupled, each a single step.

## 4. Action contract

```ts
export const TUNNEL_TO = "tunnel_to" as const;

export const tunnelToInputSchema = z.strictObject({
  // Confirmed target the staircase aims at. Defaults to the loaded end_portal_frame
  // interior found near the bot, so the locate_stronghold branch can call with no args.
  target: pointSchema.optional().describe(
    "Block to reach, walkable. Defaults to the interior of the loaded end_portal_frame nearest the bot.",
  ),
  search_id: z.string().trim().min(1).default("stronghold").describe(
    "Durable search name; landmarks fold into this locate_stronghold record.",
  ),
  // Guard budgets, defaulted; exposed for the test and for tuning.
  max_shore_distance: z.number().int().min(16).default(256).describe(
    "Max horizontal blocks to search for dry land before giving up.",
  ),
  segment_length: z.number().int().min(4).max(13).default(13).describe(
    "Blocks per navigate dig-hop. A* times out on the full descent, so decompose.",
  ),
  silverfish_budget: z.number().int().min(0).default(4).describe(
    "Infested blocks the break-in may accept before preferring another face.",
  ),
});
```

Result schema (mirrors `actionResultSchema` house style — `success`, `message`, plus
payload):

```ts
export const tunnelToResultSchema = actionResultSchema({
  target: pointSchema,                       // where we aimed
  arrived: pointSchema.nullable(),           // where the bot actually stands
  reachedTarget: z.boolean(),                // standing at/adjacent to target interior
  mouth: pointSchema.nullable(),             // dry shore entry (landmark)
  portalExit: pointSchema.nullable(),        // last walkable step in the room (landmark)
  shoreDistanceBlocks: z.number().nonnegative().nullable(),
  staircase: z.strictObject({
    segments: z.number().int().nonnegative(),  // dig-hops carved
    descent: z.number().int().nonnegative(),   // blocks of y dropped
    laddersPlaced: z.number().int().nonnegative(),
  }),
  touchedWater: z.boolean(),                  // invariant: false on success
  touchedLava: z.boolean(),                   // invariant: false on success
  hazards: z.array(z.strictObject({           // what the scan found / how handled
    kind: z.enum(["lava", "silverfish"]),
    position: pointSchema,
    action: z.enum(["rerouted", "avoided", "accepted"]),
  })),
});
```

Success means `reachedTarget && !touchedWater && !touchedLava`. On a guarded abort
(e.g. lava everywhere, no dry break-in within budget) the action returns `success: false`
with the partial staircase recorded, the bot parked safely on the last dry step, and the
mouth landmark written so a retry resumes.

## 5. Planner algorithm

### 5.1 Nearest dry shore

From the confirmed frame `(fx, _, fz)`, pick the nearest land cell:

- Read `frontier_chunks.surface_water_fraction` from the bot DB. Candidate chunks are
  those with `surface_water_fraction == 0` (fully dry), ordered by horizontal distance to
  `(fx, fz)`, within `max_shore_distance`.
- **Verified (2026-11-12):** the host does populate `surface_water_fraction`
  (`bot-data/frontier.ts` `scanAndRecord` → `measureSurfaceWaterFraction`: the fraction of
  the 256 surface cells whose highest non-air block is water/bubble, so `== 0` is exactly
  "fully dry surface"). **But** it is written once per chunk, the first time that chunk is
  *loaded and scanned* (`INSERT ... ON CONFLICT (chunk_key) DO NOTHING`), and never
  updated. So the table only covers chunks the bot has actually observed. When `tunnel_to`
  triggers from the ocean surface ~108 blocks from shore, the shore chunk is **not
  guaranteed** to be recorded yet (it depends on render distance and the `locate_stronghold`
  approach path).
- Therefore the `view_blocks` fallback is **not a backstop but the primary path whenever the
  nearest shore is unobserved**: scan the surface outward from above the frame until a dry
  standable column is found. Use the DB query first (cheap, and `locate_stronghold`'s trek
  often has loaded nearby land); on an empty/too-far result, fall back to the scan.
- The mouth is the dry standable surface cell at the chosen chunk closest to the frame on
  the straight line toward it. For seed `97996358` this resolves to the south shore chunk
  `(-312,-1720)` (water 0%, d≈108), matching the hand-pilot.

### 5.2 Line → segments

A straight 3-D line from the mouth to the target. A* over the full descent explores
~70k nodes and times out (~2s budget in `navigation/navigate.ts`), so decompose:

- Split the line into `≤ segment_length` (default 13) descending dig-hops.
- Each hop is a `navigate` with `{ dig: true, scaffold: true, range: 1 }` to the next
  waypoint — the validated dig-hop shape. `scaffold: true` lets it bridge a 1-block gap
  without placing a tower; the liquid-breach guards in
  `navigation/mineflayer/movement-policy.ts` keep each hop dry.
- Target a ~1:1 diagonal (one block out per one block down) so the staircase self-supports
  and stays walkable both ways — the hand-pilot slope. The 1-block-per-step drop is also
  what keeps each hop under the policy's `DEFAULT_MAXIMUM_DROP = 3` (unassisted descent cap
  without a water bucket); a steeper or vertical descent would exceed it, which is exactly
  why vertical runs get ladders (§2).

### 5.3 Auto-y (foliage / headroom retry)

A dig-hop can fail with `NAVIGATION_HEIGHT_REQUIRED` when the waypoint's feet height is
unstandable (water column, overhang, too-low ceiling). The error payload lists
`standable feet heights`. On that failure:

- Parse the `standable feet heights`.
- Pick the height **closest to the current y** (keeps the ~1:1 slope; avoids lurching up
  or down).
- Reissue the hop to the adjusted waypoint.
- Bounded retries (e.g. 3) before treating the segment as blocked and invoking §6.

### 5.4 Hazard scan per deep segment

Before committing a deep dig-hop, scan ahead with a `view_blocks` box
(`half_width ≤ 4`, `half_height ≤ 3`) centred on the waypoint. Classify lava and
`infested_*` blocks and feed §6. Scanning only the deep segments (not shallow shore hops)
keeps the call count down.

### 5.5 Break-in to the portal room

The final segments breach the stronghold shell into the portal room. Prefer a face that is
**not** `infested_stone_bricks`. For seed `97996358` the north wall (`z-1812`) is infested
and the lava pool sits at y-38 under the frame (`x-271..-269, z-1816..-1818`) — so approach
the frame interior from a side that avoids both, arriving at the frame y-level
(`y-36`) rather than dropping onto the lava.

### 5.6 Place the bot

End with the bot standing at the frame interior (`target`), or the nearest walkable cell
adjacent to it if the exact interior is occupied. Record it as `arrived` / `portalExit`.

## 6. Hazard policy

| Hazard | Policy | Behaviour |
| --- | --- | --- |
| **Lava** | Hard guard | Never breach into or under lava. If a scan finds lava on the planned hop, reroute (shift the waypoint, re-slope) ; if no dry route exists within budget, abort safely and record `touchedLava: false`. Never a "cross a little lava" path. **Already enforced by the movement policy** (`AVOID_BLOCKS` + `lavaFlowsInto` refuse standing in/beside lava; `opensIntoLiquid` refuses breaking into it) — the `view_blocks` pre-scan is a *routing-quality* guard that reroutes before a hop dead-ends on that refusal, not the last line of safety. |
| **Silverfish** | Soft preference | Prefer a non-infested break-in face. Accept up to `silverfish_budget` (default 4) infested breaks if the dry route needs them; the host's hostile-mob reflex handles spawned silverfish. Each accepted block is recorded in `hazards`. |
| **Water** | Hard guard | The staircase is dry by construction; the liquid-breach guards keep every hop dry. `touchedWater: true` on any breach is a failure, not a nuisance. (Swimming the open ocean *to* the mouth is separate and allowed; the dig starts at the dry mouth.) |

## 7. Landmarks

Fold two coordinates into the existing `locate_stronghold` `search_id` record rather than a
standalone store — the mirror of `StrongholdThrows` (`stronghold_throws` table,
`throw-store.ts`), keyed `(bot_id, dimension, search_id)`.

- `mouth` — the dry shore entry where the staircase begins.
- `portal_exit` — the last walkable step inside the room (the re-entry foothold for the
  deferred return path).

Two shapes are possible; pick at implementation time against the host's store conventions:

- **Columns on the existing throw/search record:** add `mouth_json`, `portal_exit_json`
  to the stronghold search record (one row per `search_id`), set by `tunnel_to` on success.
- **A sibling table** `stronghold_landmarks(bot_id, dimension, search_id, kind,
  position_json, created_at)` with `kind ∈ {mouth, portal_exit}`, following the exact
  `StrongholdThrows` begin/observe/finish transaction idiom.

The sibling table is cleaner (no schema change to the throw record, multiple landmarks
extensible for the return path), so it is the default recommendation. `locate_stronghold`'s
result reads them back so the End chain and a resumed `tunnel_to` can reuse the mouth
without re-searching.

## 8. State machine

```
             confirmed frame (from locate_stronghold)
                          │
                 ┌────────▼────────┐
                 │ FIND_SHORE      │ surface_water_fraction / fallback scan
                 └────────┬────────┘
                          │ mouth  ──► write mouth landmark
                 ┌────────▼────────┐
                 │ GOTO_SHORE      │ navigate (swim allowed), end dry at mouth
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐   per segment
                 │ CARVE_SEGMENT   │◄── view_blocks hazard scan
                 │  navigate dig   │    auto-y retry (§5.3)
                 │  ladder if vert │    lava → reroute / abort
                 └────────┬────────┘
                          │ more? loop
                 ┌────────▼────────┐
                 │ BREAK_IN        │ prefer non-infested face
                 └────────┬────────┘
                 ┌────────▼────────┐
                 │ STAND_AT_FRAME  │ write portal_exit landmark
                 └────────┬────────┘
                          ▼
                 success: reachedTarget && !water && !lava
```

## 9. Return path (deferred, noted)

Not built now. The ladders placed on vertical runs (§2) and the `portal_exit` landmark
(§7) are the hooks a later return action needs: climb the ladders, walk the dry staircase
back to the `mouth`, then surface. Design conditions parked in the project memory
`part2-return-path-deferred`.

## 10. Test setup

Seeded, headless, deterministic. One canonical seed as the acceptance test; more seeds
added later as regression.

- **Seed `97996358`** — the ocean stronghold. Frame `~(-270,-36,-1817)`, 12
  `end_portal_frame` at y-36 ringing a 3×3 hole; lava at y-38 directly under; north wall
  silverfish; under ~100 blocks of deep ocean (all surrounding chunks 100% water). Nearest
  dry land: south shore chunk `(-312,-1720)`, d≈108.
- **Harness:** extend `tests/minecraft/scenario-bot.ts` (real server, Mine-Labs-arranged
  player). Place the bot near the ocean surface above the frame, call `tunnel_to` with the
  known frame as `target`, run to completion headless.
- **World substrate:** the seeded vanilla 1.21.4 world in this repo's
  `infra/docker-compose.yml` (SEED `97996358`, bind-mounted `./world`); drive it with
  `scripts/lab.sh` (`up`/`down`/`console`). Note: the pre-commit hook tests the main
  checkout only — run the suite yourself for worktree commits (memory
  `precommit-hook-tests-main-checkout`).
- **Assertions:**
  1. `reachedTarget === true` — bot stands at/adjacent to the frame interior.
  2. `touchedWater === false` and `touchedLava === false` — the dry/lava invariants.
  3. The staircase is walkable both ways: replay mouth→frame and frame→mouth over the
     carved cells with the same movement policy; no liquid cell, every step standable,
     ladders present on vertical runs.
  4. Landmarks `mouth` and `portal_exit` persisted and reread from the `search_id` record.
- **Unit coverage** (no server): planner math on `MemoryWorld`/`openArena` — line→segment
  split respects `segment_length`; auto-y picks the height closest to current y; lava scan
  classifies a seeded lava box as a hard reroute; silverfish over budget flips the break-in
  face.

## 11. Dependencies — verification status (checked 2026-11-12, static code read)

1. **`surface_water_fraction` populated — ✅ with a caveat.** Written by
   `bot-data/frontier.ts` per observed chunk, `== 0` means fully dry surface. But it is
   insert-once (`ON CONFLICT DO NOTHING`), so it covers only chunks the bot has loaded —
   the nearest shore is not guaranteed present at trigger time. The §5.1 `view_blocks`
   fallback is consequently load-bearing, not optional. Resolved in §5.1.
2. **`view_blocks` box limits — ✅ exactly as assumed.** `src/actions/view-blocks/contract.ts`:
   `MAX_BOX_HALF_WIDTH = 4`, `MAX_BOX_HALF_HEIGHT = 3` (a 9×9×7 box). The action also offers
   `find` (nearest blocks of named kinds, anywhere loaded) and `cells` (≤32 exact cells) —
   `find: ["lava", "infested_stone_bricks"]` is a cleaner hazard probe than a box for the
   "is there lava anywhere near this hop" question, worth using alongside the box.
3. **`navigate` dig-hop liquid guards — ✅ closed statically.** The guard is
   `opensIntoLiquid` → `OPENS_INTO_LIQUID` in `navigation/mineflayer/movement-policy.ts`,
   a callback on the single `MovementPolicy` every `navigate` call uses. It decides on block
   state + world geometry only — there is no hand-vs-programmatic path, so the hand-pilot
   proof carries over. Critically, `evaluateBreak` applies `opensIntoLiquid` **last, even to
   a requested break** (`dig: true`): if removing a cell would let water or lava in, the dig
   is refused regardless of the caller asking for it. That is the mechanism that keeps the
   staircase dry, and it holds for the action's calls identically. Lava is doubly guarded:
   `AVOID_BLOCKS` + `lavaFlowsInto` refuse *standing* in/next-to lava, `opensIntoLiquid`
   refuses *breaking* into it. (The seeded test still exercises it end-to-end, but the
   invariant no longer rests on an unverified assumption.)
4. **Landmark store shape — ✅ sibling table confirmed viable.** `StrongholdThrows`
   (`throw-store.ts`) is keyed `(bot_id, dimension, search_id)` with begin/observe/finish
   transactions; a sibling `stronghold_landmarks(bot_id, dimension, search_id, kind,
   position_json, created_at)` mirrors it cleanly and is the recommended shape (§7).
