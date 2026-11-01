#!/usr/bin/env bash
# The lab's lifecycle: world (Docker), MCP host (bun), run (mlaya), report (analyze).
#
#   ./scripts/lab.sh up [--n 600] [--fresh] [--ignore-goal]   start world, host, run
#   ./scripts/lab.sh status                                    JSON: world, host, run, last tick
#   ./scripts/lab.sh analyze [--baseline runs/history/…]       report for the newest log
#   ./scripts/lab.sh report [run number]                        write the finished run's report
#   ./scripts/lab.sh console "op LayaView"                      server console command
#   ./scripts/lab.sh down [--run-only]                         stop run, host, world
#
# Rules the lab enforces: one run at a time (lock file), a run only starts when the world
# and the host are both healthy, the previous decisions log is archived first, and a run that
# ends writes its own report -- nobody has to ask for the analysis.
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
  (cd "$MCP_REPO" && bun src/server/host.ts --minecraft-host 127.0.0.1 \
    --minecraft-port "$MC_PORT" --username MineAI --version 1.21.4 \
    --data-root "$MCP_DATA_ROOT" \
    > "$REPO/runs/mcp.log" 2>&1 &)
  echo "$MCP_DATA_ROOT" > "$STATE/host_args"
  log "MCP host starting from $MCP_BRANCH @ $(mcp_commit) (log: runs/mcp.log)"
}

next_run_number() {  max=0
  for f in "$REPO"/runs/run*.log "$REPO"/runs/history/decisions-run*.jsonl; do
    n=$(basename "$f" | sed -n 's/.*run\([0-9]*\).*/\1/p')
    [[ -n "$n" && "$n" -gt "$max" ]] && max="$n"
  done
  echo $((max + 1))
}

# The comparison run is the newest archived log long enough to mean something. The run right
# before is usually it, but an aborted one is not: run 145 reported against run 144, which died
# at iteration 5, and every line of that diff said "+124 ticks" about a run that never started.
BASELINE_MIN_TICKS=30
pick_baseline() {
  for ((n = $1 - 1; n >= 1; n--)); do
    f="$REPO/runs/history/decisions-run$n.jsonl"
    [[ -f "$f" ]] || continue
    [[ "$(wc -l < "$f")" -ge $BASELINE_MIN_TICKS ]] && { echo "$f"; return 0; }
  done
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
  n=600
  fresh=0
  ignore_goal=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --n) n="${2:-600}"; shift 2 ;;
      --n=*) n="${1#--n=}"; shift ;;
      --fresh) fresh=1; shift ;;
      --ignore-goal) ignore_goal=1; shift ;;
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
  echo "$(mcp_commit)" > "$STATE/mcp_commit"
  if ! host_up; then
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

  if ! "$PY" -m mlaya.cli probe; then
    log "probe failed - the host is not usable, the run is not started"
    return 1
  fi

  run_no=$(next_run_number)
  if [[ -f "$REPO/runs/decisions.jsonl" ]]; then
    mv "$REPO/runs/decisions.jsonl" "$REPO/runs/history/decisions-run$((run_no - 1)).jsonl"
    log "previous log archived to runs/history/decisions-run$((run_no - 1)).jsonl"
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
  log "run $run_no started (${args[*]}) -> runs/run$run_no.log, report written on completion"
  return 0
}

cmd_status() {
  "$PY" - "$REPO" "$(world_up && echo yes || echo no)" <<'EOF'
import json, os, subprocess, sys, time
from pathlib import Path

repo = Path(sys.argv[1])
world_running = sys.argv[2] == "yes"
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

anomalies = []
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
print(json.dumps({
    "world": {"running": world_running},
    "host": {"running": running("src/server/host.ts"),
             "branch": subprocess.run(["git", "-C", "/Users/thtesche/VibeCoding/mine-ai-mcp",
                                       "rev-parse", "--abbrev-ref", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
             "commit": subprocess.run(["git", "-C", "/Users/thtesche/VibeCoding/mine-ai-mcp",
                                       "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
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

cmd_down() {
  if run_up; then
    kill "$(cat "$STATE/run.pid")" 2>/dev/null
    rm -f "$STATE/run.pid" "$STATE/report.pid" "$STATE/lock"
    log "run stopped"
  fi
  if [[ ${1:-} != --run-only ]]; then
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
  console) shift; cmd_console "$@" ;;
  op) shift; cmd_op "$*" ;;
  host-test) cmd_host_test ;;
  down) shift; cmd_down "$@" ;;
  *) echo "usage: lab.sh up|status|analyze|report|host-test|down" ; exit 1 ;;
esac
