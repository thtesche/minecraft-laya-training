#!/usr/bin/env bash
# Watchdog for server freezes: monitors latest.log for "Can't keep up" and terminates
# the run if detected, to prevent the bot from hanging for hours waiting for a dead server.
#
# Usage: watchdog.sh <run_number> <world_log_path> <state_dir> <check_interval_s>
#
# The watchdog runs until the run process exits, or a freeze is detected and handled.

set -uo pipefail

run_no="$1"
log_path="$2"
state_dir="$3"
check_interval="${4:-10}"

log() { printf '[watchdog-run%s] %s\n' "$run_no" "$*"; }

# Marker files to prevent repeated restarts
freeze_detected_file="$state_dir/freeze-detected-run$run_no"

while true; do
  # Check if the run process is still alive
  if ! kill -0 "$(cat "$state_dir/run.pid" 2>/dev/null)" 2>/dev/null; then
    log "run process has exited - watchdog done"
    break
  fi

  # Check for freeze signals in latest.log
  if [[ -f "$log_path" ]]; then
    # "Can't keep up!" is the Watchdog's warning before a tick timeout
    if grep -q "Can't keep up" "$log_path"; then
      # Only log and abort once per run, even if multiple ticks lag
      if [[ ! -f "$freeze_detected_file" ]]; then
        log "detected 'Can't keep up' — server is frozen"
        touch "$freeze_detected_file"
        # Extract the worst tick time from the log for the report
        tick_time=$(grep "A single server tick took" "$log_path" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        [[ -n "$tick_time" ]] && log "worst tick time: ${tick_time}s"
        # Kill the run; the report subprocess will pick up when it exits
        log "terminating run $run_no"
        kill "$(cat "$state_dir/run.pid")" 2>/dev/null || true
        break
      fi
    fi
  fi

  sleep "$check_interval"
done

exit 0
