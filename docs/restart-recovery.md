# Restart recovery — what survives a cold start, what the loop re-derives

Reviewed 2026-10-25. The question this answers: the game is paused and the MCP server stopped;
on a fresh start with the **old Minecraft world** and a **freshly launched MCP server**, does the
whole decision logic still work?

**Short answer.** The per-tick logic runs and corrupts nothing — it is built to read its world
fresh every tick rather than trust persisted memory. But a cold start of the *agent process*
forgets every loop-owned latch. World-anchored facts re-adopt themselves; the milestone latches do
not, and one of them (`powder_stowed`) gates the whole stronghold phase, so a restart mid-phase
**rewinds the bot to "re-do the Nether haul."** Everything re-converges because the inventory and
the world blocks persist; the cost is redundant work, not a broken state.

## The design: world-read facts vs. in-memory latches

There is **no live state file** the agent loads on start. `runs/decisions.jsonl` is the decision
log for training, and `GameState.from_state` only rebuilds a state from a log row for
replay/training — neither resumes a live run. Every latch in `graph.Agent.__init__` starts at its
default (`None` / `False` / `[]`).

Two kinds of fact, two fates on restart:

- **World-anchored facts re-adopt.** The inventory is read from the server every tick. The lit
  Nether gate is re-adopted as the base (`_adopt_portal_base`), and once the bot stands near it the
  chest, overflow chests, furnace and smoker re-adopt from the scan within `ADOPT_CHEST_RADIUS`
  (`_adopt_homebase`). The end-portal frame — and any eyes already seated in its sockets — load
  into the scan near the stronghold and drive `_at_stronghold` the same way a loaded `nether_portal`
  drives `leave_nether`. **Caveat:** re-adoption is location-dependent. A bot that restarts far
  from base (e.g. deep in the stronghold) is briefly "homeless" until it walks back to its portal.

- **Milestone latches reset and are *not* read back.** `stronghold_complete` and
  `nether_expedition_complete` are deliberately pure latches — the docstring is explicit: *"reading
  the world back would cost an inspection every tick."* On a cold start they all read `False`:
  `powder_stowed`, `stronghold_estimated`, `end_portal_active`, `powder_exhausted`, `powder_cells`.

## The stronghold phase, latch by latch

| Loop latch | After restart | Recovery | Cost |
|---|---|---|---|
| inventory / gear | — (server-side) | read fresh every tick | none |
| gate, chests, furnace, frame, filled sockets | — (world / container DB) | re-adopted near base, else world scan at the stronghold | none, but only once near base |
| `powder_stowed` (= `nether_expedition_complete`) | `False` | **not** read back from the chest | **stronghold chain gated off; Nether trip re-runs** |
| `stronghold_estimated` | `False` | re-latches after a fresh `estimate` throw | one eye; `locate_need` back to 16 |
| `powder_exhausted` / `powder_cells` | `False` / `[]` | self-heals — an empty withdraw re-latches | none |
| `end_portal_active` (= `stronghold_complete`) | `False` | **not** read back | re-activation attempt on an already-active frame |

### The dominant effect: the Nether haul re-runs

`planner.candidates` offers the stronghold chain only under
`if nether_expedition_complete(gs) and not stronghold_complete(gs)` (`planner.py:476`), and
`nether_expedition_complete` is just `bool(gs.powder_stowed)`. With the latch reset,
`nether_expedition_open` reads `True` again (`planner.py:412`), so the bot re-enters the Nether and
re-hauls nine rods / eighteen powder — **even though the powder already sits in the gate chest**
(the chest then holds thirty-six). Wasteful, not harmful. Once the fresh stow re-latches
`powder_stowed`, the stronghold chain resumes and runs on the eyes/pearls still in the inventory.

### The server-side bearing is durable (the one thing that does persist off-agent)

`locate_stronghold` keeps the triangulation under a durable `search_id: "stronghold"` on the MCP
server, durable per bot+world across cancellation and restart (`docs/stronghold.md`). So even
though the agent's `stronghold_estimated` latch resets and it re-issues an `estimate` phase, the
server reuses the kept bearing if it still has it, or the throw rebuilds it if the fresh MDC server
lost it. Either way it re-converges.

### The one real soft spot: a forgotten *active* portal

`end_portal_active` is never read back from the world. If the portal was already activated before
the restart, the bot forgets it and — once it is back at the frame with eyes — issues
`activate_portal` again. Whether that closes cleanly depends on how the server answers
`activate_portal` on a frame whose twelve sockets are already full: a re-confirming `ok` self-heals
in one tick, a refusal leaves the bot gathering eyes it does not need until it re-activates. This
is harmless for the current scope — the End entry and dragon fight are a later phase that does not
exist in the bot yet — so it stops at an (already) active portal regardless.

## If resuming mid-phase ever has to be free

The loop is built for one continuous run, not for resuming a phase across a process restart. If
that changes, the targeted hardening is to read a few facts back from the world instead of trusting
the latch, each once on the first tick that can see them:

- `powder_stowed` — from the gate chest's contents (a `use_container` inspection) rather than the
  deposit latch, so a restart with powder already banked does not re-run the Nether trip.
- `end_portal_active` — from a single frame inspection on the first `_at_stronghold` after start,
  so an already-active portal is recognised instead of re-activated.

`stronghold_estimated` needs no fix: the durable server bearing already carries that work, and a
re-estimate is cheap. Not built — runs start only on the user's word, and this is a deliberate
tradeoff, not a bug.
