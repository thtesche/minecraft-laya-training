#!/usr/bin/env bash
# The lab's lifecycle: world (Docker), MCP host (bun), run (mlaya), viewer (node), report (analyze).
#
#   ./scripts/lab.sh up [--n 1000] [--fresh] [--ignore-goal]  start world, host, viewer, run
#   ./scripts/lab.sh up --no-viewer                           the same, without the live view
#   ./scripts/lab.sh status                                    JSON: world, viewer, host, run, last tick
#   ./scripts/lab.sh analyze [--baseline runs/history/…]       report for the newest log
#   ./scripts/lab.sh report [run number]                        write the finished run's report
#   ./scripts/lab.sh dataset [flags]                            fine-tuning corpus from every archived run
#   ./scripts/lab.sh viewer [up|down|status]                   the LayaView spectator client alone
#   ./scripts/lab.sh console "op LayaView"                      server console command
#   ./scripts/lab.sh down [--run-only]                         stop run, host, viewer, world
#
# Rules the lab enforces: one run at a time (lock file), a run only starts when the world
# and the host are both healthy, the previous decisions log is archived first, and a run that
# ends writes its own report -- nobody has to ask for the analysis. A watchdog monitors the
# server log for freezes (Watchdog tick timeouts) and terminates the run if detected, preventing
# multi-hour hangs when the server becomes unresponsive.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MCP_REPO="/Users/thtesche/VibeCoding/mine-ai-mcp"
# The host always runs from our own server branch: the patches the runs need live there.
MCP_BRANCH="bot-run"
# The bot's ledger (action_executions, events, frontier) is what the loop reads back as
# "this world already stands". Keeping it beside the world makes a fresh world actually fresh.
MCP_DATA_ROOT="$REPO/runs/.lab/bot-data"
# Players that get operator rights on every world start. A dedicated server has an empty
# ops list, unlike a singleplayer LAN world where the host is automatically OP. The UUID a
# client uses is not the name-derived offline one, so the lab remembers the UUID the server
# actually assigns and reseeds ops.json before each world start.
OPS_USERS=("Sanskrita" "LayaView")
COMPOSE="$REPO/infra/docker-compose.yml"
STATE="$REPO/runs/.lab"
OPS_KNOWN="$STATE/ops-known.json"
PY="$REPO/.venv/bin/python"
MCP_PORT=25575
MC_PORT=25565
# The live view: a spectator client (LayaView) that watches the bot in the browser. It is an
# add-on, never a dependency of a run - see start_viewer.
VIEWER="$REPO/viewer"
VIEWER_PORT=3006
VIEWER_PID="$STATE/viewer.pid"
VIEWER_STATE="$STATE/viewer-state.json"

mkdir -p "$STATE"

log() { printf '%s\n' "$*"; }

world_up() {
  # `timeout` because a stalled daemon must not hang the lab: an unreachable container is
  # treated as down, and `up` starts it again.
  timeout 15 docker compose -f "$COMPOSE" ps --format '{{.Name}} {{.State}}' 2>/dev/null | grep -q running
}

host_up() {
  pgrep -f 'src/server/host.ts' >/dev/null 2>&1
}

# The host must run from `bot-run`. A clean tree is switched silently; a dirty one is not
# touched, because a half-applied patch under a running bot is worse than no patch.
ensure_mcp_branch() {
  branch=$(git -C "$MCP_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)
  if [[ "$branch" == "$MCP_BRANCH" ]]; then
    return 0
  fi
  if [[ -z "$(git -C "$MCP_REPO" status --porcelain)" ]]; then
    git -C "$MCP_REPO" checkout "$MCP_BRANCH" >/dev/null
    log "MCP repo switched from $branch to $MCP_BRANCH"
    return 0
  fi
  log "MCP repo is on '$branch' with uncommitted changes - the host must run from $MCP_BRANCH"
  return 1
}

mcp_commit() {
  git -C "$MCP_REPO" rev-parse --short HEAD 2>/dev/null
}

run_up() {
  [[ -f "$STATE/run.pid" ]] && kill -0 "$(cat "$STATE/run.pid")" 2>/dev/null
}

# Operator rights for the lab's players. Being in ops.json is not enough: the entry needs a
# permission level above 0, and the UUID must be the one the client actually presents.
seed_ops() {
  mkdir -p "$REPO/infra/world"
  [[ -f "$OPS_KNOWN" ]] || return 0
  "$PY" - "$OPS_KNOWN" "$REPO/infra/world/ops.json" <<'EOF'
import json, sys
known = json.load(open(sys.argv[1]))
ops = [{"uuid": u, "name": n, "level": 4, "bypassesPlayerLimit": False}
       for n, u in known.items()]
open(sys.argv[2], "w").write(json.dumps(ops, indent=2) + "\n")
EOF
  log "ops seeded for $(IFS=' '; echo "${OPS_USERS[*]}")"
}

# The world's own log file is bind-mounted, so it is a faster and steadier readiness signal
# than the Docker log API, which stalls when the daemon is busy. Either source is enough.
# A host that is connected to the world is the third proof: the server is up and playable,
# whatever the log file happens to hold (live: `latest.log` rotated to its tail and the gate
# refused a world the bot was playing in).
world_ready() {
  nc -z -w 1 127.0.0.1 "$MC_PORT" 2>/dev/null || return 1
  grep -q 'Done (' "$REPO/infra/world/logs/latest.log" 2>/dev/null && return 0
  timeout 15 docker compose -f "$COMPOSE" logs world 2>/dev/null | grep -q 'Done (' && return 0
  host_connected
}

wait_world() {
  for ((i = 0; i < 120; i++)); do
    if world_ready; then
      return 0
    fi
    sleep 1
  done
  # Say which half of the gate failed: a closed port is a container that is not up,
  # a missing "Done (" is a server that is still generating the world or crashed.
  log "world gate failed after 120 s: port $(nc -z -w 1 127.0.0.1 "$MC_PORT" >/dev/null 2>&1 && echo open || echo closed), 'Done (' in the log $(world_ready && echo yes || echo no)"
  return 1
}

# A host that answers /health but is not connected to the world is not usable: the
# transport is up, the bot is not. `ok` and `minecraft.connected` are the real gate.
host_connected() {
  body=$(curl -s -m 3 "http://127.0.0.1:$MCP_PORT/health")
  [[ -n "$body" ]] && printf '%s' "$body" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if d.get("ok") and (d.get("minecraft") or {}).get("connected") else 1)
'
}

wait_host() {
  for ((i = 0; i < 60; i++)); do
    if host_connected; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_host() {
  # A killed host releases port 25575 only when its process is actually gone. Starting
  # too early fails with "port in use" and leaves the lab with no host at all.
  for ((i = 0; i < 20; i++)); do
    nc -z -w 1 127.0.0.1 "$MCP_PORT" 2>/dev/null || break
    sleep 1
  done
  # NAV_FAILURE_TIMEOUT_MS / NAV_PRIMARY_TIMEOUT_MS raise the pathfinder's inline
  # search budgets for a run, to test whether more compute clears the long-route
  # `search timed out ... no path` stops. Unset, the host keeps its own defaults.
  local search_flags=()
  [[ -n "${NAV_FAILURE_TIMEOUT_MS:-}" ]] && search_flags+=(--search-failure-timeout-ms "$NAV_FAILURE_TIMEOUT_MS")
  [[ -n "${NAV_PRIMARY_TIMEOUT_MS:-}" ]] && search_flags+=(--search-primary-timeout-ms "$NAV_PRIMARY_TIMEOUT_MS")
  # `${arr[@]}` on an empty array is an unbound variable in bash 3.2 (the macOS default)
  # under `set -u`, so the host would never start when no search budgets are overridden.
  # The `${arr[@]+...}` form expands to nothing when the array is empty in every bash.
  (cd "$MCP_REPO" && bun src/server/host.ts --minecraft-host 127.0.0.1 \
    --minecraft-port "$MC_PORT" --username MineAI --version 1.21.4 \
    --data-root "$MCP_DATA_ROOT" ${search_flags[@]+"${search_flags[@]}"} \
    > "$REPO/runs/mcp.log" 2>&1 &)
  [[ ${#search_flags[@]} -gt 0 ]] && log "MCP host search budgets overridden: ${search_flags[*]}"
  echo "$MCP_DATA_ROOT" > "$STATE/host_args"
  log "MCP host starting from $MCP_BRANCH @ $(mcp_commit) (log: runs/mcp.log)"
}

viewer_up() {
  [[ -f "$VIEWER_PID" ]] && kill -0 "$(cat "$VIEWER_PID")" 2>/dev/null
}

# LayaView joins the world as a spectator and prismarine-viewer renders it on :3006. The viewer
# reconnects by itself when the world restarts, so it outlives a `--fresh` without help.
# It is an add-on: node missing or the deps not installed is a line in the log, never a reason
# to refuse a run.
start_viewer() {
  if viewer_up; then
    log "viewer already running (pid $(cat "$VIEWER_PID")) - http://localhost:$VIEWER_PORT"
    return 0
  fi
  if ! command -v node >/dev/null 2>&1; then
    log "viewer not started: node is not on the path"
    return 1
  fi
  if [[ ! -d "$VIEWER/node_modules" ]]; then
    log "viewer not started: viewer/node_modules is missing - run: cd viewer && npm install"
    return 1
  fi
  # The braces keep the command substitution's pipe closed: without them the background job
  # inherits it and `$( ... )` waits for a viewer that never exits.
  pid=$( cd "$VIEWER" && { node viewer.mjs --host 127.0.0.1 --port "$MC_PORT" \
    --view-port "$VIEWER_PORT" --state "$VIEWER_STATE" > "$REPO/runs/viewer.log" 2>&1 & echo $!; } )
  echo "$pid" > "$VIEWER_PID"
  log "viewer starting as LayaView (spectator) -> http://localhost:$VIEWER_PORT, log: runs/viewer.log"
  return 0
}

stop_viewer() {
  if viewer_up; then
    kill "$(cat "$VIEWER_PID")" 2>/dev/null
    rm -f "$VIEWER_PID"
    log "viewer stopped"
  fi
}

next_run_number() {  max=0
  for f in "$REPO"/runs/run*.log "$REPO"/runs/history/decisions-run*.jsonl; do
    n=$(basename "$f" | sed -n 's/.*run\([0-9]*\).*/\1/p')
    [[ -n "$n" && "$n" -gt "$max" ]] && max="$n"
  done
  echo $((max + 1))
}

# The comparison run is the newest archived log that is long enough to mean something, and
# preferably one that ran to its iteration limit. An aborted run is not a baseline: run 145
# reported against run 144, which died at iteration 5, and every line of that diff said
# "+124 ticks" about a run that never started; run 148 diffed against run 146, cut off at 40
# ticks by the shell timeout that killed it.
BASELINE_MIN_TICKS=100
pick_baseline() {
  fallback=""
  for ((n = $1 - 1; n >= 1; n--)); do
    f="$REPO/runs/history/decisions-run$n.jsonl"
    [[ -f "$f" ]] || continue
    [[ "$(wc -l < "$f")" -ge $BASELINE_MIN_TICKS ]] || continue
    # The run's own end line says whether it finished or was cut short.
    if grep -q '"stop": *"limit"' "$REPO/runs/run$n.log" 2>/dev/null; then
      echo "$f"; return 0
    fi
    [[ -z "$fallback" ]] && fallback="$f"
  done
  [[ -n "$fallback" ]] && { echo "$fallback"; return 0; }
  return 1
}

# A finished run reports itself. The JSON summary and the markdown go next to the archived log,
# so the next `up` can diff against it without anyone remembering to ask. `analyze` reads the
# end JSON from the newest runs/run*.log, which is still this run's at that moment.
report_run() {
  run_no="${1:-$(cat "$STATE/run_number" 2>/dev/null)}"
  [[ -n "$run_no" ]] || return 0
  log_file="$REPO/runs/decisions.jsonl"
  archived="$REPO/runs/history/decisions-run$run_no.jsonl"
  # Asked for a run by number: report the archived log, not the one the next run is writing.
  if [[ $# -ge 1 && -f "$archived" ]]; then log_file="$archived"; fi
  baseline=$(pick_baseline "$run_no") || baseline=""
  out_md="$REPO/runs/history/report-run$run_no.md"
  out_json="$REPO/runs/history/summary-run$run_no.json"
  # The run's own log carries the end JSON; without it the report reads the newest run's.
  run_log="$REPO/runs/run$run_no.log"
  extra=()
  [[ -f "$run_log" ]] && extra=(--run-log "$run_log")
  if [[ -f "$baseline" ]]; then
    "$PY" -m mlaya.cli analyze --log "$log_file" "${extra[@]}" \
      --baseline "$baseline" --out "$out_md" > "$out_json" 2>&1
  else
    "$PY" -m mlaya.cli analyze --log "$log_file" "${extra[@]}" \
      --out "$out_md" > "$out_json" 2>&1
  fi
  log "run $run_no reported: runs/history/report-run$run_no.md"
}

cmd_up() {
  n=1000
  fresh=0
  ignore_goal=0
  no_viewer=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --n) n="${2:-1000}"; shift 2 ;;
      --n=*) n="${1#--n=}"; shift ;;
      --fresh) fresh=1; shift ;;
      --ignore-goal) ignore_goal=1; shift ;;
      --no-viewer) no_viewer=1; shift ;;
      *) log "unknown argument: $1"; return 1 ;;
    esac
  done

  if run_up; then
    log "a run is already running (pid $(cat "$STATE/run.pid")) - lab up refuses a second one"
    return 1
  fi

  if [[ $fresh == 1 ]]; then
    # The world folder is bind-mounted into the running server. Wiping it while the
    # server is up deletes its working directory under it: the process keeps running,
    # every write fails, and the run looks alive with an empty world. Stop first.
    if world_up; then
      docker compose -f "$COMPOSE" down
      log "world container stopped for the wipe"
    fi
    rm -rf "$REPO/infra/world" "$MCP_DATA_ROOT"
    log "world and the bot's ledger wiped: fresh world from the seed in infra/docker-compose.yml"
  fi

  if ! world_up; then
    seed_ops
    # A stalled daemon must not hang the lab forever (live: `up` sat on this call for 22 h).
    timeout 60 docker compose -f "$COMPOSE" up -d
    log "world container starting"
  fi
  if ! wait_world; then
    log "world did not become ready in 120 s - check: docker compose -f infra/docker-compose.yml logs world"
    return 1
  fi
  log "world ready on 127.0.0.1:$MC_PORT"

  if ! ensure_mcp_branch; then
    return 1
  fi
  # The host loads its code when it starts, so a patch committed after it came up is not in
  # the running bot. `mcp_commit` records what the host started from: a tree ahead of it means
  # a fix nobody restarted for sits in the repo while the bot plays the old one.
  started_commit=$(cat "$STATE/mcp_commit" 2>/dev/null)
  echo "$(mcp_commit)" > "$STATE/mcp_commit"
  if ! host_up; then
    start_host
  elif [[ "$started_commit" != "$(mcp_commit)" ]]; then
    log "MCP host is running $started_commit but the tree is at $(mcp_commit) - restarting it for the patch"
    pkill -f 'src/server/host.ts'
    sleep 2
    start_host
  elif [[ "$(cat "$STATE/host_args" 2>/dev/null)" != "$MCP_DATA_ROOT" ]]; then
    log "MCP host is running with a stale bot ledger - restarting with $MCP_DATA_ROOT"
    pkill -f 'src/server/host.ts'
    sleep 2
    start_host
  elif ! host_connected; then
    # The world container can restart under a running host; the mineflayer client does not
    # reconnect on its own, and a host with a dead Minecraft connection is not a host.
    log "MCP host is running but not connected to the world - restarting it"
    pkill -f 'src/server/host.ts'
    sleep 2
    start_host
  fi
  if ! wait_host; then
    log "MCP host did not answer /health with a connected bot in 60 s - check runs/mcp.log"
    tail -5 "$REPO/runs/mcp.log" 2>/dev/null
    return 1
  fi
  log "MCP host healthy on 127.0.0.1:$MCP_PORT"

  # The live view rides with the lab: started with the world, stopped with it. A viewer the
  # lab does not manage is a viewer nobody remembers to stop.
  if [[ $no_viewer == 1 ]]; then
    stop_viewer
  else
    start_viewer
  fi

  if ! "$PY" -m mlaya.cli probe; then
    log "probe failed - the host is not usable, the run is not started"
    return 1
  fi

  run_no=$(next_run_number)
  if [[ -f "$REPO/runs/decisions.jsonl" ]]; then
    mv "$REPO/runs/decisions.jsonl" "$REPO/runs/history/decisions-run$((run_no - 1)).jsonl"
    log "previous log archived to runs/history/decisions-run$((run_no - 1)).jsonl"
  fi
  # The System-2 side-file rides along with the decision log so a run's escalations stay paired
  # with it (`mlaya.analysis` joins the two by iteration).
  if [[ -f "$REPO/runs/system2.jsonl" ]]; then
    mv "$REPO/runs/system2.jsonl" "$REPO/runs/history/system2-run$((run_no - 1)).jsonl"
    log "previous System-2 log archived to runs/history/system2-run$((run_no - 1)).jsonl"
  fi

  touch "$REPO/runs/decisions.jsonl"
  args=(-n "$n")
  [[ $ignore_goal == 1 ]] && args+=(--ignore-goal)
  cd "$REPO"
  nohup "$PY" -m mlaya.cli run "${args[@]}" > "$REPO/runs/run$run_no.log" 2>&1 &
  run_pid=$!
  echo "$run_pid" > "$STATE/run.pid"
  echo "$run_no" > "$STATE/run_number"
  touch "$STATE/lock"
  # The report writes itself once the run's process is gone: nobody has to remember to ask
  # for the analysis, and the next `up` can diff against it straight away.
  ( while kill -0 "$run_pid" 2>/dev/null; do sleep 5; done; report_run "$run_no" ) &
  echo $! > "$STATE/report.pid"
  # Watchdog monitors the server log for freezes (Can't keep up) and aborts the run.
  "$REPO/scripts/watchdog.sh" "$run_no" "$REPO/infra/world/logs/latest.log" "$STATE" &
  echo $! > "$STATE/watchdog.pid"
  log "run $run_no started (${args[*]}) -> runs/run$run_no.log, report written on completion"
  return 0
}

cmd_status() {
  "$PY" - "$REPO" "$(world_up && echo yes || echo no)" "$VIEWER_PORT" <<'EOF'
import json, os, subprocess, sys, time
from pathlib import Path

repo = Path(sys.argv[1])
world_running = sys.argv[2] == "yes"
viewer_port = int(sys.argv[3]) if len(sys.argv) > 3 else 3006
state = repo / "runs" / ".lab"

def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0

def pid_alive(path):
    try:
        os.kill(int(path.read_text().strip()), 0)
        return True
    except Exception:
        return False

import urllib.request
try:
    health = json.loads(urllib.request.urlopen("http://127.0.0.1:25575/health", timeout=5).read())
except Exception as exc:
    health = {"error": str(exc)}

records = []
log = repo / "runs" / "decisions.jsonl"
if log.exists():
    for line in log.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass

actions = [r for r in records if r.get("stage") == "action"]
last = actions[-1] if actions else {}
state_ = last.get("state", {})

# The host's code is fixed at its start; the repo moves under it. Reporting the tree's HEAD
# as the host's commit would say a patch is in play when the bot is still running the old one.
started_commit = (state / "mcp_commit").read_text().strip() if (state / "mcp_commit").exists() else None
tree_commit = subprocess.run(["git", "-C", "/Users/thtesche/VibeCoding/mine-ai-mcp",
                              "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

anomalies = []
if running("src/server/host.ts") and started_commit and tree_commit and started_commit != tree_commit:
    anomalies.append(f"host runs MCP {started_commit} while the tree is at {tree_commit} - restart it for the patch")
if running("src/server/host.ts") and not (health.get("minecraft") or {}).get("connected"):
    # The run reads the world through the host. A host that answers /health but is not
    # connected to Minecraft is blind, and every tick after that is a failed perception.
    anomalies.append("host is not connected to the world")
hp, food = state_.get("health"), state_.get("food")
if hp is not None and hp <= 8:
    anomalies.append(f"low health {hp}")
if food is not None and food <= 6:
    anomalies.append(f"low food {food}")
if len(actions) >= 6:
    window = actions[-6:]
    labels = {a.get("label") for a in window}
    moved = sum(abs((a.get("delta") or {}).get("pos_delta", 0)) for a in window)
    if len(labels) <= 2 and moved < 3:
        anomalies.append("stalled: " + "/".join(sorted(str(l) for l in labels)))

run_no = (state / "run_number").read_text().strip() if (state / "run_number").exists() else ""

# The viewer reports itself through --state, so status can tell "the process is alive" apart
# from "LayaView is actually in the world with a camera on the bot".
viewer = {"running": pid_alive(state / "viewer.pid"), "url": f"http://localhost:{viewer_port}"}
try:
    vstate = json.loads((state / "viewer-state.json").read_text())
except Exception:
    vstate = {}
for key in ("connected", "gamemode", "dimension", "camera", "target", "distance", "updatedAt"):
    viewer[key] = vstate.get(key)
if viewer["running"] and not viewer.get("connected"):
    viewer["note"] = "process is up but LayaView is not in the world (reconnecting)"

print(json.dumps({
    "world": {"running": world_running},
    "viewer": viewer,
    "host": {"running": running("src/server/host.ts"),
             "branch": subprocess.run(["git", "-C", "/Users/thtesche/VibeCoding/mine-ai-mcp",
                                       "rev-parse", "--abbrev-ref", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
             "commit": started_commit or tree_commit,
             "tree_commit": tree_commit,
             "runtime": (health.get("runtime") or {}).get("state"),
             "connected": (health.get("minecraft") or {}).get("connected"),
             "foreground": health.get("foreground"),
             "bot": (health.get("minecraft") or {}).get("position")},
    "run": {"running": pid_alive(state / "run.pid"),
            "number": run_no or None,
            "log": str(repo / "runs" / f"run{run_no}.log") if run_no else None,
            "iterations": last.get("iteration"),
            "last_label": last.get("label"),
            "last_status": last.get("status"),
            "health": hp, "food": food,
            "dimension": state_.get("dimension"),
            "ticks": len(records),
            "anomalies": anomalies},
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
}, indent=2, ensure_ascii=False, default=str))
EOF
}

cmd_analyze() {
  baseline=""
  if [[ ${1:-} == --baseline ]]; then baseline="$2"; fi
  if [[ -n "$baseline" ]]; then
    "$PY" -m mlaya.cli analyze --log "$REPO/runs/decisions.jsonl" --baseline "$baseline"
  else
    "$PY" -m mlaya.cli analyze --log "$REPO/runs/decisions.jsonl"
  fi
}

cmd_report() {
  report_run "${1:-}"
}

# The live view on its own: restart a viewer that dropped, or run one against a world that is
# already up, without touching the host or starting a run.
cmd_viewer() {
  case "${1:-status}" in
    up) start_viewer ;;
    down) stop_viewer ;;
    status)
      if viewer_up; then
        echo "viewer running (pid $(cat "$VIEWER_PID")) - http://localhost:$VIEWER_PORT"
        cat "$VIEWER_STATE" 2>/dev/null || echo '{"connected": null}'
      else
        echo "viewer not running"
      fi
      ;;
    *) echo "usage: lab.sh viewer up|down|status"; return 1 ;;
  esac
}

# The fine-tuning corpus: every archived run plus the live log, rebuilt through today's
# renderer so the head is trained on exactly the wording the loop serves. Extra flags
# (--rerender, --include-failed, --limit …) go straight to `mlaya.cli dataset`.
cmd_dataset() {
  logs=()
  for f in "$REPO"/runs/history/decisions-run*.jsonl; do logs+=(--log "$f"); done
  if [[ -f "$REPO/runs/decisions.jsonl" ]]; then logs+=(--log "$REPO/runs/decisions.jsonl"); fi
  "$PY" -m mlaya.cli dataset "${logs[@]}" --out "$REPO/data/train.jsonl" "$@"
}

cmd_down() {
  if run_up; then
    kill "$(cat "$STATE/run.pid")" 2>/dev/null
    kill "$(cat "$STATE/watchdog.pid")" 2>/dev/null
    rm -f "$STATE/run.pid" "$STATE/report.pid" "$STATE/watchdog.pid" "$STATE/lock"
    log "run stopped"
  fi
  if [[ ${1:-} != --run-only ]]; then
    stop_viewer
    if host_up; then
      pkill -f 'src/server/host.ts'
      log "MCP host stopped"
    fi
    docker compose -f "$COMPOSE" down
    log "world container stopped"
  fi
}

cmd_host_test() {
  ensure_mcp_branch || return 1
  log "running the server test suite on $MCP_BRANCH @ $(mcp_commit)"
  (cd "$MCP_REPO" && bun run test)
}

cmd_op() {
  if [[ $# -eq 0 ]]; then
    echo "usage: lab.sh op <player>   (the player must be online)"
    return 1
  fi
  cmd_console "op $*"
  "$PY" - "$REPO/infra/world/ops.json" "$OPS_KNOWN" "$*" <<'EOF'
import json, os, sys
world, known_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
known = json.load(open(known_path)) if os.path.exists(known_path) else {}
try:
    entries = json.load(open(world))
except Exception:
    entries = []
for e in entries:
    if e.get("name") == name and e.get("level", 0) > 0:
        known[name] = e["uuid"]
        print(f"{name} is op (uuid {e['uuid']}); remembered for every future world start")
        break
else:
    print(f"{name} is not in ops.json - the console grants op only to a player who is online")
json.dump(known, open(known_path, "w"), indent=2)
EOF
}

cmd_console() {
  if [[ $# -eq 0 ]]; then
    echo "usage: lab.sh console \"op LayaView\""
    return 1
  fi
  docker compose -f "$COMPOSE" exec world sh -c 'printf "%s\n" "$1" > /tmp/minecraft-input' sh "$*"
}

case "${1:-status}" in
  up) shift; cmd_up "$@" ;;
  status) cmd_status ;;
  analyze) shift; cmd_analyze "$@" ;;
  report) shift; cmd_report "$@" ;;
  dataset) shift; cmd_dataset "$@" ;;
  viewer) shift; cmd_viewer "$@" ;;
  console) shift; cmd_console "$@" ;;
  op) shift; cmd_op "$*" ;;
  host-test) cmd_host_test ;;
  down) shift; cmd_down "$@" ;;
  *) echo "usage: lab.sh up|status|analyze|report|viewer|host-test|down" ; exit 1 ;;
esac
