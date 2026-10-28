# Restart recovery — what survives a cold start, what the loop re-derives

Reviewed 2026-10-28. The question this answers: the game is paused and the MCP server stopped;
on a fresh start with the **old Minecraft world** and a **freshly launched MCP server**, does the
whole decision logic still work?

**Short answer.** The per-tick logic runs and corrupts nothing — it is built to read its world
fresh every tick rather than trust persisted memory. A cold start of the *agent process* forgets
every loop-owned latch, but the two that mattered re-derive themselves: world-anchored facts
re-adopt, and `powder_stowed` — the latch that gates the whole stronghold phase — is now read back
from the banked chest (`_adopt_powder_stow`), so a restart mid-phase no longer **rewinds the bot to
"re-do the Nether haul."** Everything re-converges because the inventory and the world blocks
persist; the only residual cost is a possible re-activation of an already-active end portal.

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

- **Milestone latches reset on a cold start; most re-derive, one does not.** The latches in
  `graph.Agent.__init__` all start `False` / `[]`: `powder_stowed`, `stronghold_estimated`,
  `end_portal_active`, `powder_exhausted`, `powder_cells`. `powder_stowed` is read back from the
  banked powder (`_adopt_powder_stow`, below), `stronghold_estimated` re-latches off the durable
  server bearing, `powder_exhausted`/`powder_cells` self-heal — only `end_portal_active` is still a
  pure latch, and its reset is cheap (one re-activation of an already-full frame).

## The stronghold phase, latch by latch

| Loop latch | After restart | Recovery | Cost |
|---|---|---|---|
| inventory / gear | — (server-side) | read fresh every tick | none |
| gate, chests, furnace, frame, filled sockets | — (world / container DB) | re-adopted near base, else world scan at the stronghold | none, but only once near base |
| `powder_stowed` (= `nether_expedition_complete`) | `False` | **read back** from the banked chest (`_adopt_powder_stow`) | none — the stronghold chain resumes without a re-haul |
| `stronghold_estimated` | `False` | re-latches after a fresh `estimate` throw | one eye; `locate_need` back to 16 |
| `powder_exhausted` / `powder_cells` | `False` / `[]` | self-heals — an empty withdraw re-latches | none |
| `end_portal_active` (= `stronghold_complete`) | `False` | **not** read back | re-activation attempt on an already-active frame |

### The haul no longer re-runs: `powder_stowed` re-derives from the chest

`planner.candidates` offers the stronghold chain only under
`if nether_expedition_complete(gs) and not stronghold_complete(gs)` (`planner.py:476`), and
`nether_expedition_complete` is just `bool(gs.powder_stowed)`. A reset latch used to flip
`nether_expedition_open` back to `True` (`planner.py:412`), so the bot re-entered the Nether and
re-hauled nine rods / eighteen powder even though the powder already sat in the gate chest — the
most dangerous stretch of the run, re-run for nothing (the chest then held thirty-six).

That is now closed. `_note_stocked_chest` already reads every network chest's full contents from
the server's container database (`REMEMBERED_STOCKS_SQL`) to find the banked food — a read of the
server's memory, not of the world, so it answers from any distance and across a restart, at no
`use_container` cost on the tick. `_adopt_powder_stow` consumes the same rows: blaze powder standing
in a network chest latches `powder_stowed` (and remembers the cells, so a later withdraw opens the
right lid). It is one-way — a chest drained of its powder leaves the latch untouched, which is
correct, because `_note_take_powder` / `_nether_topup_wanted` own reopening a genuine top-up. The
read is filtered to the cells this base owns (`chest_pos` + `_extra_chests`), so a stale chest from
another base or world cannot fake the milestone. A bot that spawns on its own base adopts the chest
after the tick's stock read has run, so `_adopt_homebase` clears the `_stock_read` throttle to force
one fresh read on the next tick.

### The server-side bearing is durable (the one thing that does persist off-agent)

`locate_stronghold` keeps the triangulation under a durable `search_id: "stronghold"` on the MCP
server, durable per bot+world across cancellation and restart (`docs/stronghold.md`). So even
though the agent's `stronghold_estimated` latch resets and it re-issues an `estimate` phase, the
server reuses the kept bearing if it still has it, or the throw rebuilds it if the fresh MDC server
lost it. Either way it re-converges.

## The one latch still not read back

`powder_stowed` is now re-derived (above), so the expensive re-haul is gone. The one remaining pure
latch is `end_portal_active`: it is never read back from the world, so if the portal was already
activated before the restart, the bot forgets it and — once it is back at the frame with eyes —
issues `activate_portal` again. Whether that closes cleanly depends on how the server answers on a
frame whose twelve sockets are already full: a re-confirming `ok` self-heals in one tick, a refusal
leaves the bot gathering eyes it does not need until it re-activates. This is harmless for the
current scope — the End entry and dragon fight are a later phase that does not exist in the bot yet
— so it stops at an (already) active portal regardless. The targeted hardening, if it is ever worth
it, is a single frame inspection on the first `_at_stronghold` after start to recognise an active
portal instead of re-activating it.

`stronghold_estimated` needs no fix: the durable server bearing already carries that work, and a
re-estimate is cheap.
