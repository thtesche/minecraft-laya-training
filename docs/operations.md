# Operations — running, fixing and training

The working rules of this project: how a run is set up and archived, how a fix lands, and how a
head is trained. Run-by-run findings live in [`architecture.md`](architecture.md) (the
"Live findings" and milestone sections); this page is the procedure around them.

## The world and the bot

- **The world is a vanilla 1.21.4 dedicated server in Docker** (`infra/docker-compose.yml`, seed
  `97996358`, world folder bind-mounted at `infra/world`, port pinned to `127.0.0.1:25565`).
  `./scripts/lab.sh up` starts it; `--fresh` wipes the world folder *and* the bot's ledger first,
  and stops the container before wiping it — deleting a bind-mounted folder under a running server
  leaves the process alive with no working directory, so the run looks healthy in an empty world.
  The user's own client can join the same server, and so can the viewer.
- **The MCP server** (`mine-ai-mcp`, a separate repo at `~/VibeCoding/mine-ai-mcp`) connects the
  bot `MineAI` to that world and serves MCP on `http://127.0.0.1:25575/mcp`:
  `bun src/server/host.ts --minecraft-host 127.0.0.1 --minecraft-port 25565 --username MineAI --version 1.21.4 --data-root runs/.lab/bot-data`.
  `lab.sh` always starts it from the `bot-run` branch (the patches the runs live on) and restarts
  it when the data root changed or when the host answers `/health` without a connected Minecraft
  client — a host with a dead world connection is not a host.
  `.venv/bin/python -m mlaya.cli probe` answers whether the loop can reach it.
- **The bot's database** lives under `runs/.lab/bot-data/server-host-<host>-port-<port>-seedHash-<seed>/bots/<uuid>/bot-data.sqlite`,
  one per world. It holds every action request (`action_requests`, with arguments), the server's
  events (`events`, including the survival reflexes' outcomes) and the containers the bot has
  opened (`observed_containers`). Read it read-only (`sqlite3 "file:<path>?mode=ro"`); it is the
  place to find the arguments of an action the run log does not record. It sits next to the world
  on purpose: a fresh world is only fresh if this ledger is wiped with it.
- **Operator rights are reseeded on every world start.** A dedicated server has an empty ops list
  (a LAN world makes its host OP automatically), so `lab.sh` writes `Sanskrita` and `LayaView`
  into `ops.json` before each start, remembering the UUID the server actually assigns
  (`runs/.lab/ops-known.json`) — the name-derived offline UUID is not the one a real client uses.
- **The live view** is the [viewer](../viewer/README.md): a spectator client in the browser,
  started by `lab.sh up` and stopped by `lab.sh down` (`--no-viewer` skips it, `lab.sh viewer
  up|down|status` drives it on its own). It must stay a spectator, or the bot can no longer sleep
  through the night.

## A run

One command: `./scripts/lab.sh up [--n 1000] [--fresh] [--ignore-goal] [--no-viewer]
[--iron-armor] [--observe | --use-observer-better]`. `--iron-armor` opts into the pre-diamond iron
armour set (default: skip it); `--observe` and `--use-observer-better` are the two mutually exclusive
observer modes (see below). The other verbs: `status` (JSON: world, viewer, host, run, observer,
last tick, anomalies), `report [run]`,
`analyze [--baseline runs/history/…]`, `dataset [flags]`, `viewer [up|down|status]`,
`console "op LayaView"`, `down [--run-only]`.

The lab does the bookkeeping that used to be manual:

1. It refuses a second run while one is running (`runs/.lab/run.pid`), and starts only when the
   world is ready, the host is connected, and `mlaya.cli probe` passes.
2. The previous log is archived on its own: `runs/decisions.jsonl` →
   `runs/history/decisions-runNN.jsonl` (a run that did not reach its limit gets a `-partial` /
   `-crashed` suffix by hand).
3. A watcher writes `runs/history/report-runNN.md` and `summary-runNN.json` the moment the loop's
   process exits, diffing against the previous run's summary — nobody has to ask for the analysis.
4. The live view rides with the world: `up` starts LayaView, `down` stops it, and `status` reports
   what the viewer writes to `runs/.lab/viewer-state.json` — connected, gamemode, camera, distance
   to MineAI. The viewer is an add-on: node missing or `viewer/node_modules` not installed is a
   line in the log, never a reason to refuse a run.

`status` names the anomalies worth a look without reading the log: health ≤ 8, food ≤ 6, a host
that answers `/health` but is not connected to the world, and a stall (six action ticks with at
most two distinct labels and under 3 blocks moved).

Optional, only when a trained head is meant to run:
`export LAYA_EXTRA_MODELS='{"english": "<repo>/runs/checkpoints/minecraft-<time>"}'` before `up`.
Without it the loop runs the base `english` checkpoint. The head the lab serves by default is
registered in the gitignored `.env` — currently `minecraft-20261111-103647` (fine-tuned on the
runs 187–219 corpus plus run 220's flip-stable teacher corrections, re-rendered, over the
2026-10-28 checkpoint). A real environment variable still wins over
the file, so an A/B run can override it without touching `.env`.

**Watchdog:** `scripts/watchdog.sh` runs during every `up` and monitors the server log for freezes.
If the server fails to tick (detected via "Can't keep up" in the log), the watchdog terminates the
run immediately, preventing the bot from waiting hours for a dead server. This guard is essential
because the Java Watchdog does not fire until the process wakes — a frozen container or host sleep
can leave the server stalled without crashing, and the agent blind.

Manual start, only when the lab is not usable:
`.venv/bin/python -m mlaya.cli run -n 600 > runs/runNN.log 2>&1 &`.

Never patch code under a running run: a run keeps the code it started with, and a fix
applies from the next run. A run that is stuck is stopped by the user, not by the tooling.
When a run ends, the last line of `runs/runNN.log` is the end JSON (`iterations`, `stop`
`goal`/`limit`/`body_held`/`body_busy`, policy; `body_held` means two *non-progressing*
`action_wait_ms` budgets in a row (a reflex still inside its release budget is progress, and a reflex
that never releases is capped separately by `BODY_HELD_LIVE_STOP = 6`), `body_busy` means it answered
`No new action started; body owner: …` `BODY_BUSY_STOP` times running without ever handing the
body over), and `runs/decisions.jsonl` holds one record per tick
(`action`, `defer`, `policy` stages) with the state, the candidates, the chosen label, the result
text and the delta.

### Reading the log while it is written

A record is one long line, and `state` alone (with `resources` and `mobs`) runs to a few
thousand characters, so `jq .` buries the decision under it. `scripts/dec.py` prints a record
with one key per line at level 1 and everything deeper on a single line:

```bash
tail -f runs/decisions.jsonl | scripts/dec.py                                   # live, level 1
tail -n 5 runs/decisions.jsonl | scripts/dec.py --keys ts,stage,iteration,tool,label,ok,status,answers,delta
scripts/dec.py --depth 2 --file runs/history/decisions-runNN.jsonl              # one level more unfolded
scripts/dec.py --no-color < runs/decisions.jsonl | less                         # into less or a file
```

`--depth` is the number of levels that get their own lines (default 1), `--keys` keeps only the
named top-level keys, and a half-written record from a run still in progress is skipped with a
note on stderr rather than ending the reader.

## Parallel observer (opt-in)

A second LLM watches every tick alongside Laya and writes its own judgment to `runs/observer.jsonl`.
It is **read-only** — it tails `runs/decisions.jsonl`, never touches the bot or any log the loop
reads — and it is **off by default** (no per-tick model call unless enabled). It is one append-only
conversation ("story"): a branch caches 0 tokens on this endpoint, a pure extension reuses the whole
prior prefix, so the system prefix is byte-stable for the run and the thinking toggle rides in each
appended turn (`scripts/cache_probe.py` measured this). Routine ticks run light (`/no_think`,
~1–3 s); full reasoning is spent only on *events* — phase change, escalation, stall (few labels and
no progress across the recent window), death. Config: `ObserverConfig` in `config.py` (no `[observer]`
section in `config/settings.toml` yet — add one to override): Strata `qwen3.8-flash-next-q2_0`,
`STRATA_API_KEY`, 16 384 output tokens for events / 1 024 for
routine turns, 185 k input budget before the oldest turns are folded into a summary, `max_lag = 8`
ticks before routine ticks are sampled rather than each observed; events are never skipped).

Two ways to run it, **never together** (both write `runs/observer.jsonl`):

- `./scripts/lab.sh up --observe` — a detached **sidecar** (`mlaya.cli observe`), read-only. It
  follows the live log until stopped; `lab.sh` drains it before the report reads the logs and
  archives it as `runs/history/observer-runNN.jsonl` beside `decisions-runNN.jsonl`. `status`
  reports `observer.running`, `judgments` and the last `iteration`/`phase`/`event`/`verdict`.
- `./scripts/lab.sh up --use-observer-better` — the observer runs **in-process** in the loop
  (`settings.loop.use_observer_better`); it writes `runs/observer.jsonl` itself, so no sidecar. On a
  *disagree* whose `better` is one of the tick's own candidates and differs from Laya's label, that
  label is staged and run on the **next** tick. Guarded: one-shot (cleared on read, a fresh verdict
  sets the next), never the same alternative more than `OBSERVER_BETTER_REPEATS = 3` ticks running,
  cleared at once by an agree or a defer (danger ticks belong to survival), and reset by progress on
  the forced label — so the worst case is the action Laya would have run anyway. The decision record
  carries `observer_forced` and the log a `policy` line with `reason: observer_better`, so the report
  can tell an override from a Laya choice. It is placed after the deterministic inventory unblock
  and stall escalation, so it never pre-empts them.

Manual/replay: `mlaya observe --log runs/decisions.jsonl [--replay] [--out runs/observer.jsonl]`.
Replay walks an archived log once; follow tails until interrupted. `--out` keeps an archived replay
out of the live sidecar log. Each verdict line carries `ts`, `iteration`, `phase`, `event`,
`reasoned`, `laya_label`, `progressed`, `gear`, the `verdict` object and `telemetry`. The verdict
schema is `{"phase", "agree", "risk", "note", "better"}`; `better` is the single label from the
tick's `candidates` that would have been the better move, or null. `gear` is the worn armour per
slot and pickaxe/sword with durability in %, plus the off-hand — shown so the judge does not flag a
kit it cannot see (run247 told the bot to "focus on shield" while the shield was in the off-hand
the whole time). Note that `progressed` in this log is **copied from the action record**: for an
`explore` tick it only means the body covered ground (`verify.moved`), not that the phase advanced
— see the reward note in `architecture.md`.

`mlaya analyze [--observer <log>]` joins the two logs by iteration; the report's `observer` block
shows cache `hit_rate` (the payoff of the append-only story), latency split by reasoned/routine
ticks, agreement/disagreement with the tick's real outcome, risks and stalls (`stalls.borne_out`
counts flags whose tick did *not* progress), and how often gear was actually shown.

## A fix

- **Every fix comes with tests**, written from the live state that showed the defect (the cell,
  the label, the refusal text), and the new test must fail without the fix.
- Tests: `.venv/bin/python -m pytest -q` — that is what the pre-commit hook runs (`pytest` is in
  the venv; `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` gives the same 1967
  tests). There is no bare `python` on the machine; always `.venv/bin/python`.
- **Commits:** small and thematic, straight on `main`, a one-line summary with a scope prefix —
  `fix(runNNN):`, `feat(end):`, `docs(operations):` — that names the live case behind it (run,
  cell, count). Push only when asked.
- Agent fixtures in `tests/test_graph.py` build `Agent` with `Agent.__new__` and set each field by
  hand; a new field that `_inject` reads needs adding to those fixtures.

## Training a head

1. Build the dataset from the archived runs into its own files:
   `.venv/bin/python -m mlaya.cli dataset --log runs/history/decisions-runA.jsonl --log … --out data/<name>-train.jsonl --eval data/<name>-eval.jsonl`.
   Ping-pong windows and walk-only deposits are filtered out (see architecture, M3). Add
   `--rerender` to re-read each archived tick's narrative through today's `narrative()` (rebuild
   an old corpus in the current wording — needed after the relative-narrative switch, so the head
   trains on the rendering it meets at run time). The absolute cells stay in the log's `state`;
   only the narrative Laya reads is re-rendered.
2. Dry run: `.venv/bin/python -m mlaya.cli train --dry-run --data … --eval …` — check that `base` is the checkpoint you
   mean. The base is `[train] base_model` resolved through `LAYA_EXTRA_MODELS`, or `--base`.
3. Train: `.venv/bin/python -m mlaya.cli train --data … --eval … > runs/train-<name>.log 2>&1` → a new
   `runs/checkpoints/minecraft-<time>`; the end JSON carries the held-out before/after and the
   `LAYA_EXTRA_MODELS` value that serves it.
4. A held-out split from the training runs is close to in-distribution; only a live run (or a
   whole unseen run as held-out) shows whether the head decides better.
5. Optional — publish to the Hub: `hf upload thtesche/laya-minecraft-english runs/checkpoints/minecraft-<time> --exclude "checkpoint_latest/*"` and refresh the card from `hf/model-card.md`. The Hub id then works directly as a `LAYA_EXTRA_MODELS` source (`{"english": "thtesche/laya-minecraft-english"}`); corpora go to the dataset repo (`hf upload thtesche/laya-minecraft-training-data data --repo-type dataset`, card in `hf/dataset-card.md`). Weights belong in a **model** repo — laya's `snapshot_download` defaults to `repo_type="model"` and cannot resolve dataset repos.

## Hold-out set (M4)

A split drawn from the same runs the head was trained on is nearly in-distribution. Measured:
the 252–256 corpus scored 0.9369 for a fine-tune and 0.9279 for the base checkpoint; on run257,
a run in no training file, the same fine-tune scored 0.8764 against the base's 0.8876. The gain
inside the corpus did not travel.

Two things have to be measured before a checkpoint is called better:

* **Pipeline jitter.** Three runs on the same data, config and seed scored 0.9338 / 0.9373 /
  0.9408 on the 574-item hold-out — a 0.70 pp spread. RLCD samples Gaussian noise around the
  logits each step and MPS is not deterministic, so one run is one draw. The seed is not the
  source: seeds 0, 1, 2 scored 0.9355 / 0.9390 / 0.9373, a 0.35 pp spread, *smaller* than the
  same-seed one. `holdout train --reps` repeats the run, not the seed.
* **Item-level luck.** On 574 items one item is 0.17 pp. `holdout eval` reports McNemar's
  discordant counts (items A right and B wrong, and the reverse) beside the accuracy difference,
  so a "win" of three items reads as three items, not as a trend.

A candidate has to clear the jitter, not the item count: across nine hard fine-tunes the mean
delta against the base is −0.04 pp with a 0.24 pp standard deviation and a 0.70 pp spread — the
teacher path is inside the noise. Only soft + soft-ce left it: −2.09 pp, McNemar 18/6, p = 0.023.

The registry now holds 33 runs (221–251, 257 and 258): everything after the base checkpoint's
training corpus, none of it in any file a candidate was trained on. `--per-run 10` caps each
run's contribution so a 330-tick run does not outweigh a 9-tick one — 296 rows, 574 items.

1. Register a finished run: `.venv/bin/python -m mlaya.cli holdout add --run 257`. The log is
   found in `runs/history/decisions-run257*.jsonl`; the live `runs/decisions.jsonl` is not a
   fallback, because it belongs to whichever run is running now.
2. Build the eval file: `.venv/bin/python -m mlaya.cli holdout build --per-run 10 --train
   data/train-187-220-rr.jsonl --train data/s5-rep-veto-hard.jsonl …` — repeat `--train` for
   **every** corpus any candidate was trained on. Rows whose state also appears in one of them
   are dropped and the count reported (15 of 311 here) — a hold-out set containing training rows
   measures nothing. No teacher is applied: the target is what the verifier confirmed Laya
   actually did.
3. Score checkpoints: `.venv/bin/python -m mlaya.cli holdout eval --ckpt <base> --ckpt <candidate> …`
   — the first `--ckpt` is the McNemar reference.
4. Train a config several times: `.venv/bin/python -m mlaya.cli holdout train --data … --base …
   --loss rlcd --reps 3` writes `<out>-s0…-sN`, scores each against the hold-out and prints
   `candidate_spread` — the mean and standard deviation of the deltas, the number a candidate
   has to clear.

## Teacher path (System 2 + blind judge, offline or from the observer)

The verifier (`progressed`) is the grounded primary label source and stays that way. The teacher
is *opinion* and fills only the gaps the verifier leaves — disputed ticks and the no-progress
ticks a verifier-only dataset throws away. It never overrules a tick the world showed real
progress on. Everything runs on archived logs; the live loop (`graph.py`) is untouched (the
separate `--use-observer-better` override above is the opt-in live exception).

1. Pick the System 2 source. The original path replays it per tick:
   `.venv/bin/python -m mlaya.cli teacher --log runs/history/decisions-runNN.jsonl`
   asks `[llm]` (Strata `qwen3.8-flash-next-q2_0`, `STRATA_API_KEY`) the same question over the same
   candidates on every action tick with ≥2 eligible labels — a rich run is hours at ~77 s a call.
   Since run256 the cheaper path reuses the parallel observer, which already judged every tick:
   `teacher --log <decisions log> --from-observer runs/history/observer-runNN.jsonl`. The observer's
   verdict decides agreement; only its *disagreements* cost a call — to a blind picker over the
   tick's own candidates when the verdict has no `better`, otherwise the judge adjudicates the named
   counter-move directly. Live run256 (436 iterations): 401 verdicts, 129 disagreements, 35 with a
   valid counter-move. Ticks the observer never covered get no judgment, so the dataset falls back
   to the verifier's label. Both paths write `runs/history/teacher-runNN.jsonl`; the side-file
   records `teacher_source` (`observer`/`picker`) and `better_repeats` (how often that counter-move
   recurred in the run). `--judge-no-think` turns off the judge's reasoning chain (OpenRouter
   `reasoning off`, ~30x faster); the local Qwen reviewer ignores `/no_think`, `thinking_budget`
   and `enable_thinking`, so it has no such lever.
2. On disagreement a third model — the judge (`[judge]`, OpenRouter's free
   `nvidia/nemotron-3.5-super-vl:free`, its own model so it does not grade its own earlier opinion,
   `OPENROUTER_API_KEY`) — sees both actions *blind* (not told which came from S1/S2) and rules
   A/B/neither/unsure. Only a clear, confident verdict carries a label forward.
3. Fold it into a dataset: add `--teacher runs/history/teacher-runNN.jsonl` to the `dataset`
   build (pair it with `--rerender`, as the side-file keys on the re-rendered narrative). By default
   a no-progress tick whose judge clearly sided against Laya's pick becomes a positive row for the
   judge's label; on any de-dup collision with a verifier row the verifier wins. The join hash is a
   function of the renderer, so a side-file recorded before a wording change no longer hashes to the
   same tick — the builder then falls back to the tick's identity (`iteration`, Laya's label,
   the eligible labels) rather than silently adding zero rows. Regenerating the side-file is still
   the right thing after a renderer change; the fallback just keeps an expensive batch usable.
4. **Which counter-move gets through:** `--teacher-gate` decides the evidence a teacher
   counter-move needs to become a label on a no-progress tick: `judge` (blind judge sided against
   Laya and survived the flip test), `rep` (observer named it ≥2× in the run), `union` (either) and
   the default `rep-veto` (it recurs and the judge never clearly preferred Laya's own action — the
   judge may veto but need not confirm). Repetition alone is not independent proof because the
   observer is a rolling conversation with its own earlier verdicts in context; the judge is the
   only model with no shared context, so it keeps the veto. Side-files written before `better`
   existed have no repetition to count and fall back to the judge gate, so no paid judgment is
   thrown away. On runs 252–254 that is 15 labels through the judge gate, 22 through `rep-veto`.
   Soft `gold` targets (`--teacher-soft`) are gone: on the hold-out set the hedged rows made the
   head 2.1 pp worse (McNemar 18/6, p = 0.023 over 574 items) and every lost item was an intent
   question — 16 lost, 2 gained, `in_danger` untouched — while the hard path sat inside the
   run-to-run noise band. A label is one action or it is no label.
5. Test the blindness: `teacher --log <run log> --flip-from runs/history/teacher-runNN.jsonl`
   re-runs the blind judge with the A/B slots swapped (System 2 is not asked again; its pick is read
   from the side-file) and writes `teacher-runNN-flip.jsonl` with the two verdicts side by side. It
   also writes the flip result back into the side-file, so an unstable verdict can be filtered where
   labels are taken. A verdict that survives the swap is a judgement about the actions; one that
   flips was a reaction to which letter they arrived under, and its label is noise the dataset
   should not learn from. Measured on run111: **1 flip in 29** (3.4%) — the judge is order-blind in
   practice, and the tick that flipped (`search_enderman` vs `go_home` at it38) is a genuinely close
   call rather than a slot artifact. On the larger run220 (digest payload) the rate was far higher —
   **36 flips in 180** disagreements (20%) — so the gate is not cosmetic: it kept 36 letter-reactions
   out of the labels and left ~13 flip-stable no-progress corrections.
6. Validate against **real run outcomes** (goal/deaths/ticks-to-milestone), never against
   judge-agreement — optimising Laya to please the big model instead of to survive is the failure
   mode this ordering exists to avoid. Go live only once it at least matches.
