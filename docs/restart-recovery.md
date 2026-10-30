# Restart recovery — what survives a cold start, what the loop re-derives

Reviewed 2026-10-30. The question this answers: the game is paused and the MCP server stopped;
on a fresh start with the **old Minecraft world** and a **freshly launched MCP server**, does the
whole decision logic still work?

**Short answer.** It does, and it no longer has to guess. The per-tick logic is built to read its
world fresh every tick rather than trust persisted memory, so a cold start corrupts nothing. What
used to be the problem is that "where am I" was answered by ~40 per-run latches in
`graph.Agent.__init__`, all of which come back `False`/`[]`, plus two reads that only work inside
loaded chunks. Now the run reads the server's own durable record of the world once at start
(`_bootstrap_world` → `progress.bootstrap_progress`) and resolves every milestone through one
object, `Progress`, with one precedence:

    durable ledger fact  >  world scan  >  run latch  >  False

So a bot that restarts 300 m from its base knows its gate burns, its frame stands, its bed is laid
and its powder is banked — without walking back to prove any of it.

## The design: three sources, one order

There is **no live state file** the agent loads on start. `runs/decisions.jsonl` is the decision log
for training, and `GameState.from_state` only rebuilds a state from a log row for replay — neither
resumes a live run. Every latch in `graph.Agent.__init__` starts at its default.

Three sources with different half-lives:

- **The ledger** — the MCP server's own SQLite (`action_executions`, `events`,
  `frontier_chunks` + `frontier_chunk_materials`, `stronghold_throws`, `observed_containers`,
  `notes`). It outlives a death, a dimension change and a restart, and it answers from 300 blocks
  away, where no block scan reaches. `query_bot_data` caps a read at 100 rows and 64 KiB, so every
  query in `perception.py` is narrow and ordered.
- **The scan** — `view_status` and `view_blocks`: true right now, only for loaded chunks.
- **The latch** — the loop's own flags, reset by every cold start.

A latch may *set* a milestone, never refute one the world answered. The one exception is the frame,
which the scan may still undo: a plane that names its own obsidian as air is the frame talking about
itself (`_frame_short_at_base`, live run103); a blank plane names nothing and leaves the ledger
standing.

`Progress` is not a new kind of truth — it is the same predicates in `actions/predicates.py` read in
one order. Those predicates stay as they are; they are the adapters `progress_of` is built from, and
rewriting them would rewrite the tests that pin them.

## The bootstrap: what the ledger is read for

`_bootstrap_world` runs once per run, before the first decision, and writes what it finds into the
same latches the actions set. Each read is guarded on its own: an absent table, a schema the server
has not created yet and a transport blip all look the same — an absent fact — and the milestone then
falls back to the scan and the latch exactly as it did before. A run that cannot read the ledger is
a run that behaves the way it always did, not a run that stops.

| Question | Table | Query constant |
|---|---|---|
| Does this world have a burning gate? | `frontier_chunk_materials` joined to `frontier_chunks`, `material = 'nether_portal'` | `GATE_CHUNK_SQL` |
| Was the gate ever lit, and where? | `action_executions`, `action = 'activate_portal'`, `portal.kind = "nether"`, `portalAfter > 0` | `PORTAL_LEDGER_SQL` |
| How many sockets does the end frame hold? | same rows, `portal.kind = "end"`, `eyesAfter` | `PORTAL_LEDGER_SQL` |
| Is the end portal open? | same rows, `portalAfter >= 9` or `activated` | `PORTAL_LEDGER_SQL` |
| Which bed does the world still hold? | `action_executions`, `action IN ('place_block','sleep')` | `BED_LEDGER_SQL` |
| Has this bot ever been in the Nether / the End? | `events`, `event_type = 'player_dimension_change'` | `DIMENSION_LEDGER_SQL` |
| Is a stronghold bearing saved? | `stronghold_throws`, `state != 'pending'` | `BEARING_SQL` |
| What counters did the last run of this loop leave? | `notes`, `note LIKE 'mlaya checkpoint%'` | `CHECKPOINT_READ_SQL` |

Three of these are worth naming because they close a specific hole:

**A lit gate is recorded even when its chunk is unloaded.** A portal block only ever enters the
chunk table on the tick the server actually saw it, so the material row survives the unload. That is
the difference between "I cannot see the gate" and "there is no gate" — run47 poured a second
portal beside a lit one because nothing could tell it the first existed.

**A frame that lit is a frame that stands.** `activate_portal` refuses `[PORTAL_NO_FRAME]` before it
ever reaches the flint and steel, so a `portalAfter > 0` row is durable proof the ten obsidian are
in place. `build_structure` reporting `complete` is *not* kept as proof: a build that dug its cells
clean also reports success, and only the lit portal distinguishes the two.

**The sockets are counted from the world, not the pack.** `eyesAfter` is the server counting the
eyes standing in the twelve sockets, including the ones the world generated filled — and it is
present on a refusal too: `[PORTAL_MISSING_EYES] 7 empty sockets` names the frame as it is. Before
this, `frame_eyes` counted only the eyes this bot happened to put in, so a frame that arrived with
five seated eyes read as empty and the rod haul and the pearl waves were sized for twelve eyes the
phase never owed. `_note_portal` now reads `portal.eyesAfter`, then the text line, then the
inventory delta, in that order.

## The stronghold phase, latch by latch

| Loop latch | After restart | Source | Recovery | Cost |
|---|---|---|---|---|
| inventory / gear | — (server-side) | scan | read fresh every tick | none |
| gate, chests, furnace, frame, filled sockets | — (world / container DB) | ledger, then scan | bootstrapped from the ledger; re-adopted near base (`_adopt_homebase`) | none |
| `powder_stowed` (= `nether_expedition_complete`) | `False` | ledger | **read back** from the powder chest (`_adopt_powder_stow`) | none — the chain resumes without a re-haul |
| `stronghold_estimated` | `False` | ledger | **read back** from the durable `stronghold_throws` bearing | none |
| `end_portal_active` (= `stronghold_complete`) | `False` | ledger | **read back** from `activate_portal`'s `portalAfter`/`activated` | none |
| `powder_exhausted` / `powder_cells` | `False` / `[]` | ledger + checkpoint | self-heals; the cells come from the checkpoint note | none |
| `pearl_wave_trips`, `barter_attempts` | `0` | checkpoint note | **read back** from the loop's own note | none |

### Which chest counts as the base's chest

`_note_stocked_chest` filters the container rows to the cells this run owns (`chest_pos` +
`_extra_chests`) — that filter is untouched, and a milestone is not a licence to raid a stranger's
lid. For the *milestone* only, when the run has no base yet (a bot that starts in the Nether never
adopts one, because `_walk_home_to_memory` steps out down there), a chest counts as the base's when
it holds powder and stands within `progress.GATE_CHEST_RADIUS` (6) of `_gate_anchor()`: the run's
own chest, then its base, then the gate cell the ledger recorded lighting.

### The checkpoint note: the counters no ledger can prove

The server's tables record actions, not decisions. Pearl waves spent, barter rolls made, the cells
the powder was stowed into — the loop is the only writer of those, so it writes them itself, as a
`note_save` note prefixed `mlaya checkpoint:` (durable per bot and world, read back by
`bootstrap_progress`).

`note_save` is a foreground action, so the note is written only where it is worth a call: when the
milestone set changes (`_checkpoint_key` — gate, frame, bed, powder, bearing, portal, and each
quarter of the frame), and once at the end of a run. Never while a reflex is holding the body
(`res.status == "held"`), and never per tick. A failure is logged once and dropped.

## What the run sees of its own stand

Every tick `_inject` resolves one `Progress` and puts it in `GameState.progress`, and that object
does three things:

- **It leads the narrative.** Right after the phase persona, the state opens with
  `Milestones: the gate burns, the bed is laid, the blaze powder is banked, the end portal frame
  holds 5 of 12 eyes` — before the inventory, because Laya only sees the first window of a long
  state (`docs/architecture.md`).
- **It records its provenance.** `progress.source` says which of the three answered each field
  (`ledger` / `scan` / `latch` / `default`), and `as_state()` logs it. A run that started on a
  half-played world and rebuilt its gate can be read afterwards: the tick says what it believed
  about the gate and which source told it so.
- **It derives `eyes_owed`** — empty sockets plus the four eyes the throws need, minus the eyes in
  the pack — the number `_eyes_owed` and the rod target are sized on.

## The residual cost

None of the old ones remain. The gate, the bed, the frame, the sockets, the bearing and the portal
are all read from the ledger before the first decision, so a restart mid-phase does not rewind the
bot to "re-do the Nether haul" or "re-light a gate that burns".

The one thing the ledger cannot say is a fact a *different* bot recorded: every table here lives in
the per-bot database
(`~/.mine-ai/bot-data/server-<host>-port-<port>-seedHash-<hash>/bots/<bot>/bot-data.sqlite`), so a
gate lit by another bot in the same world is not in this bot's rows. If the ledger is silent, the
run behaves as it always did — it falls back to the scan, then the latch, and walks to the gate to
look.
