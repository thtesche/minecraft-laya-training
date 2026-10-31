#!/usr/bin/env bash
# The lab's lifecycle: world (Docker), MCP host (bun), run (mlaya), report (analyze).
#
#   ./scripts/lab.sh up [--n 600] [--fresh] [--ignore-goal]   start world, host, run
#   ./scripts/lab.sh status                                    JSON: world, host, run, last tick
#   ./scripts/lab.sh analyze [--baseline runs/history/…]       report for the newest log
#   ./scripts/lab.sh console "op LayaView"                      server console command
#   ./scripts/lab.sh down [--run-only]                         stop run, host, world
#
# Rules the lab enforces: one run at a time (lock file), a run only starts when the world
# and the host are both healthy, and the previous decisions log is archived first.
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
  docker compose -f "$COMPOSE" ps --format '{{.Name}} {{.State}}' 2>/dev/null | grep -q running
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

wait_world() {
  for ((i = 0; i < 120; i++)); do
    if nc -z -w 1 127.0.0.1 "$MC_PORT" 2>/dev/null && \
       docker compose -f "$COMPOSE" logs world 2>/dev/null | grep -q 'Done ('; then
      return 0
    fi
    sleep 1
  done
  # Say which half of the gate failed: a closed port is a container that is not up,
  # a missing "Done (" is a server that is still generating the world or crashed.
  log "world gate failed after 120 s: port $(nc -z -w 1 127.0.0.1 "$MC_PORT" >/dev/null 2>&1 && echo open || echo closed), 'Done (' in the log $(docker compose -f "$COMPOSE" logs world 2>/dev/null | grep -q 'Done (' && echo yes || echo no)"
  return 1
}

# A host that answers /health but is not connected to the world is not usable: the
# transport is up, the bot is not. `ok` and `minecraft.connected` are the real gate.
wait_host() {
  for ((i = 0; i < 60; i++)); do
    body=$(curl -s -m 3 "http://127.0.0.1:$MCP_PORT/health")
    if [[ -n "$body" ]] && printf '%s' "$body" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if d.get("ok") and (d.get("minecraft") or {}).get("connected") else 1)
'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_host() {
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
    docker compose -f "$COMPOSE" up -d
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
  fi
  if ! wait_host; then
    log "MCP host did not answer /health with a connected bot in 60 s - check runs/mcp.log"
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
  echo $! > "$STATE/run.pid"
  echo "$run_no" > "$STATE/run_number"
  touch "$STATE/lock"
  log "run $run_no started (${args[*]}) -> runs/run$run_no.log"
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

cmd_down() {
  if run_up; then
    kill "$(cat "$STATE/run.pid")" 2>/dev/null
    rm -f "$STATE/run.pid" "$STATE/lock"
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
  console) shift; cmd_console "$@" ;;
  op) shift; cmd_op "$*" ;;
  host-test) cmd_host_test ;;
  down) shift; cmd_down "$@" ;;
  *) echo "usage: lab.sh up|status|analyze|host-test|down" ; exit 1 ;;
esac
