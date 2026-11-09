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

One command: `./scripts/lab.sh up [--n 600] [--fresh] [--ignore-goal] [--no-viewer]`. The other
verbs: `status` (JSON: world, viewer, host, run, last tick, anomalies), `report [run]`,
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
registered in the gitignored `.env` — currently `minecraft-20261107-132042` (fine-tuned on the
runs 187–205 corpus over the 2026-10-28 checkpoint). A real environment variable still wins over
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
`goal`/`limit`/`body_held`/`body_busy`, policy; `body_held` means the server's reflex kept the
body through two `action_wait_ms` budgets in a row, `body_busy` means it answered
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

## A fix

- **Every fix comes with tests**, written from the live state that showed the defect (the cell,
  the label, the refusal text), and the new test must fail without the fix.
- Tests: `.venv/bin/python -m pytest -q` — that is what the pre-commit hook runs (`pytest` is in
  the venv; `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` gives the same 1755
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

## Teacher path (offline System 2 + judge)

The verifier (`progressed`) is the grounded primary label source and stays that way. The teacher
is *opinion* and fills only the gaps the verifier leaves — disputed ticks and the no-progress
ticks a verifier-only dataset throws away. It never overrules a tick the world showed real
progress on. Everything runs on the archived log; the live loop (`graph.py`) is untouched.

1. Review a run: `.venv/bin/python -m mlaya.cli teacher --log runs/history/decisions-runNN.jsonl`
   → `runs/history/teacher-runNN.jsonl`. Per action tick with ≥2 eligible labels it asks System 2
   (`[llm]`) the same question over the same candidates; on disagreement a third model — the judge
   (`[judge]`, its own model so it does not grade its own earlier opinion) — sees both actions
   *blind* (not told which came from S1/S2) and rules A/B/neither/unsure. Only a clear, confident
   verdict carries a label forward. Both keys come from `.env` (`STRATA_API_KEY` by default).
2. Fold it into a dataset: add `--teacher runs/history/teacher-runNN.jsonl` to the `dataset`
   build (pair it with `--rerender`, as the side-file keys on the re-rendered narrative). A
   no-progress tick whose judge clearly sided against Laya's pick becomes a positive row for the
   judge's label; on any de-dup collision with a verifier row the verifier wins. The join hash is
   a function of the renderer, so a side-file recorded before a wording change no longer hashes to
   the same tick — the builder then falls back to the tick's identity (`iteration`, Laya's label,
   the eligible labels) rather than silently adding zero rows. Regenerating the side-file is still
   the right thing after a renderer change; the fallback just keeps an expensive batch usable.
 3. Test the blindness: `teacher --log <run log> --flip-from runs/history/teacher-runNN.jsonl`
    re-judges every recorded disagreement with the blind pair's A/B slots swapped (System 2 is not
    asked again; its pick is read from the side-file) and writes `teacher-runNN-flip.jsonl` with the
    two verdicts side by side. A verdict that survives the swap is a judgement about the actions;
    one that flips was a reaction to which letter they arrived under, and its label is noise the
    dataset should not learn from. Measured on run111: **1 flip in 29** (3.4%) — the judge is
    order-blind in practice, and the tick that flipped (`search_enderman` vs `go_home` at it38) is
    a genuinely close call rather than a slot artifact.
4. Validate at step 4 above against **real run outcomes** (goal/deaths/ticks-to-milestone), never
   against judge-agreement — optimising Laya to please the big model instead of to survive is the
   failure mode this ordering exists to avoid. Go live only once it at least matches.
