# Stronghold phase — implementation handoff

> **Status (2026-10-25): DONE.** The phase is implemented, tested and committed
> (`34079f5`, suite 990 green); both open bugs recorded below are fixed. The body
> of this file is the historical handoff record — follow the dated updates at the
> end. One open *design* point remains (the re-supply is not demand-sized): see the
> 2026-10-25 update below and `docs/stronghold.md`.

This document originally handed off a **mid-implementation, uncommitted** change: the
Stronghold-search phase for the survival bot. It was written so the work could
continue in a different harness. Read `docs/stronghold.md` first for the design;
this file records *how the code got to where it is* and *what is still open*.

## Status at handoff

- Design doc `docs/stronghold.md`: **written and committed.**
- Refactor of `actions.py` → `actions/` package: **committed** as `f09edd4`
  (NOT pushed; branch is 2 commits ahead of `origin/main`). Plan step 1 done.
- Stronghold phase code: **uncommitted, mid-implementation.** Do not assume it
  works — there is a known failing test (see below).

### Uncommitted working tree

```
 M src/mlaya/actions/__init__.py     (+1)   stronghold import, order constants→predicates→stronghold→planner
 M src/mlaya/actions/constants.py    (+6)   new eye/pearl CRITERIA entries
 M src/mlaya/actions/planner.py      (+19)  gate extension, candidate block, resolve_stronghold dispatch
?? src/mlaya/actions/stronghold.py   (new, 240 lines, untracked)
```

Nothing in `graph.py` or `perception.py` is edited yet. That is the gap that
causes the failing test.

## Known failing test (start here)

Baseline before this work: **972 tests, all green.**
Now: **972 tests, 1 error.**

```
AttributeError: 'GameState' object has no attribute 'end_portal_active'
  test_a_stowed_expedition_closes_the_chain  (tests/test_actions.py ~4037)
    → candidates()            planner.py:412
    → _nether_topup_wanted()  stronghold.py:123
    → stronghold_complete()   stronghold.py:134  →  return bool(gs.end_portal_active)
```

Run tests with:

```
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

The phase code reads a `GameState.end_portal_active` field that does not exist
yet. **Next Move step 1 adds it and makes the suite green again.**

## What the new module already does (`actions/stronghold.py`)

A self-contained phase module, mirroring the Nether-haul structure:

- **constants** — eye target = 16, pearl/eye resource names, `STRONGHOLD_LABELS`.
- **predicates** —
  - `stronghold_complete(gs)` (line ~134): `return bool(gs.end_portal_active)` —
    the phase goal latch.
  - `_nether_topup_wanted(gs)` (line ~113): true when the Nether haul is done,
    the stronghold is not yet complete, and powder has run short — drives the
    *fallback* second Nether trip via `_nether_haul_home`.
- **`stronghold_candidates(gs)`** (line ~137): the label generator for this
  phase (hunt endermen → craft eyes → locate → activate).
- **`resolve_stronghold(label, gs)`** (fragment ~232): resolves the
  `activate_portal` label into the real server call (absolute `x/y/z` of a
  frame block). Server signatures confirmed (see design doc).
- `__all__` built from module globals (same pattern as the other submodules).

## Wiring already spliced into `planner.py`

- Import of the stronghold module.
- The phase gate in `candidates()` (~line 402/412) now also considers the
  stronghold phase and the top-up fallback.
- `resolve()` (~line 559) dispatches to `resolve_stronghold(label, gs)` when
  `label in STRONGHOLD_LABELS`, guarded so the main dispatch is untouched.

## Next moves (in order)

### 1. Add the `end_portal_active` field — fixes the failing test

In `src/mlaya/perception.py`, `GameState`:

- Declare `end_portal_active: bool = False` **after** `powder_stowed` (line 781)
  and **before** `recover_written_off` (line 786).
- Persist it in `as_state` — add to the Nether block, after `"mobs": self.mobs`
  (~line 884).
- Rehydrate it in `from_state` — it rebuilds via `gs = cls()` (~line 902) then a
  field loop, so adding the field to the dataclass + `as_state` is enough; verify
  the loop picks it up.

Re-run the suite — it should return to **972 green**.

### 2. Wire the goal latch (loop side)

The goal is an AND over all phases in `_goal_reached` (`graph.py:1139`), with
each phase's latch set loop-side via a `_note_*` helper and mirrored onto
GameState (deliberate asymmetry: latch vs. read).

- Add `Agent._note_portal`, mirroring `_note_stow` (`graph.py:1393`, sets
  `self._powder_stowed = True` at 1402). It sets `self._end_portal_active = True`.
- Call it from `_apply_notes` (which calls `_note_stow` at `graph.py:1924`).
- Init `self._end_portal_active = False` near `_powder_stowed` (`graph.py:468`).
- Mirror it in `as_state` / `_inject` alongside `_powder_stowed`
  (`graph.py:520` / `1107`).
- Extend `_goal_reached` (`graph.py:1139`) to the three-way AND:
  `tech_tree_complete AND nether_expedition_complete AND stronghold_complete`.

### 3. Phase persona + perception

- Add a `PHASE_STRONGHOLD` trigger in `phase_persona` (`perception.py` ~507).
- Add `end_portal_frame` to a perception scan group (`NAMED_RESOURCES` ~78 /
  scan ~1540-1646) so an `_at_stronghold`-style predicate can see frame blocks.

### 4. `take_blaze_powder` label

Blaze powder is **not** in the edible-only `_banked_stores` filter
(`predicates.py` ~1594 / `graph.py` EDIBLE_NAMES ~737). The `take_blaze_powder`
label must search the **raw** stores for powder itself — mirror
`stow_blaze_powder`. Confirm the second-Nether fallback routes through
`_nether_haul_home` (`predicates.py` ~907).

### 5. Tests + commit

- Add unit tests mirroring the Nether-label patterns in `tests/test_actions.py`
  and `tests/test_graph.py` (goal/latch tests ~3501-3528; the three-way-AND test).
- Keep the suite green.
- Commit as a **separate unit** from the refactor. Push only if the user asks.

## Guardrails

- Behavior-preserving discipline: the refactor was a pure move; this phase adds
  wiring only — the server owns all endgame mechanics (`locate_stronghold`,
  `activate_portal`, etc.). The agent never implements client-side mechanics.
- No live runs or hot-patching; the user hosts the world and starts/stops runs.
- The phase is offered **only** once `tech_tree_complete` AND
  `nether_expedition_complete` both hold. It stops at an **activated end portal**
  — bed-at-stronghold, End entry, and the dragon fight are a later phase.

## Key file map

| File | What's here |
|---|---|
| `src/mlaya/actions/stronghold.py` | new phase module (untracked) |
| `src/mlaya/actions/planner.py` | `candidates()` gate ~402/412, `resolve()` ~559, `resolve_stronghold` guard |
| `src/mlaya/actions/predicates.py` | `_nether_haul_home` ~907, `_banked_stores` ~1594 (edible-only) |
| `src/mlaya/actions/constants.py` | CRITERIA ~12/96, `BLAZE_*` ~176-180, `HOSTILE_HUNTS` ~425 (has `enderman`) |
| `src/mlaya/graph.py` | `_goal_reached` 1139, latch init 468, `as_state`/`_inject` 520/1107, `_note_stow` 1393, `_apply_notes` 1924 |
| `src/mlaya/perception.py` | `GameState` latches (`powder_stowed` 781), `as_state` ~884, `from_state` ~902, scan groups ~1540-1646, `phase_persona` ~507 |
| `docs/stronghold.md` | phase design doc |
| `docs/nether-prep.md` | style/phase-gating template |

## Update (second session, 2026-10-24)

Done, suite green (974 tests): steps 1-3. `GameState` gained `end_portal_active`,
`stronghold_estimated`, `powder_exhausted` (also logged, with `powder_stowed`, in `as_state` /
`from_state`). `graph.py`: latches + `_note_take_powder` / `_note_locate` / `_note_portal`, a fresh
stow clears `_powder_exhausted`, `_goal_reached` is the three-way AND. `phase_persona` returns
`PHASE_STRONGHOLD` in the Overworld while `powder_stowed and not end_portal_active`;
`end_portal_frame` joined `CHAIN_NAMES` and `NAMED_RESOURCES`. Goal/persona tests updated.

Open bugs found in step 4 (not fixed yet):
- **The top-up can never relaunch.** `_nether_departure_ready` and `_spare_pickaxe_wanted`
  (predicates.py ~1000/1109) require `not nether_expedition_complete(gs)`, which stays true
  during a top-up, so `enter_portal` is never offered. Needs an "expedition open" predicate
  (not complete OR top-up wanted) without a predicates→stronghold import cycle.
- **The withdraw only looks in `chest_pos`**, but `stow_blaze_powder` deposits into
  `_storage_target(gs)` (possibly an overflow chest). Remember the stowed cells loop-side and
  withdraw from them, latching `powder_exhausted` only once all came up empty.

Still to do: step 5 tests for the stronghold labels/latches, then a separate commit.

## Update (third session, 2026-10-24)

Both open bugs are fixed, the step-5 tests are in, and the phase is committed as a separate
unit from the refactor. Suite: 990 green.

- **The top-up relaunches.** `_nether_topup_wanted` moved from `stronghold.py` to
  `predicates.py` (the Nether gates need it, and `predicates` cannot import `stronghold`
  without a cycle); the new `nether_expedition_open(gs)` - "not complete OR top-up wanted" -
  replaced the completion latch in `_nether_departure_ready` and in the `candidates` gate, so
  `enter_portal` is offered during a top-up. The eye/pearl item names and `ENDER_EYES` moved
  to `constants.py` for the same reason.
- **The withdraw remembers the cells.** `_note_stow` books the cell the deposit actually
  reached into `gs.powder_cells` (mirrored in `as_state`/`from_state`); `take_blaze_powder`
  works those cells, falling back to `chest_pos`, and `_note_take_powder` drops a cell that
  comes up empty, latching `powder_exhausted` only once every remembered cell has.
- **Tests.** `StrongholdPhaseTest` (test_actions.py) pins the five labels' offer and binding,
  the safe-hunt guard, the frame hand-off and the top-up gate; `NetherCrossingTest`
  (test_graph.py) pins the cell bookkeeping (stow remembers, one empty lid does not close the
  bank, a fresh stow clears exhaustion); the `powder_cells` round-trip is tested in
  test_perception.py.

## Update (fourth session, 2026-10-25) — review finding

A code review raised the re-supply sizing, now recorded as a known limitation in
`docs/stronghold.md` ("the re-supply is not demand-sized"):

- The Nether **top-up** still hauls a fixed nine rods / eighteen powder
  (`BLAZE_RODS` / `BLAZE_POWDER`), and `hunt_enderman` a fixed sixteen pearls
  (`ENDER_PEARLS`), regardless of how many eyes actually broke. `take_blaze_powder`
  and `craft_ender_eye` are already `_eyes_owed`-sized, so the pattern is only half applied.
- Two robustness gaps share the cause: `locate_stronghold` is gated on a full sixteen
  eyes (a throw in the `estimate` phase forces a rebuild before `locate` travels), and
  `stronghold_candidates` offers only `activate_end_portal` at the frame (no path to make
  more eyes if breakage left too few).

Not yet fixed — awaiting the user's word. The fix sketch is in the design doc.
