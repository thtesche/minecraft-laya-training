#!/usr/bin/env bash
# Run a command in a session of its own.
#
# The lab's long-lived processes (world client host, the run, the observer, the watchdog, the
# report watcher, the viewer) must not live in the process group of whoever started them. A
# shell that closes, a harness that SIGKILLs a group it spawned, an app that restarts - any of
# those end every process of that group at once, and two runs (251, 252) died exactly that way:
# five processes of one `lab.sh up`, no traceback, no end JSON, `stop: unknown`. The world
# container and the viewer, which lived in other groups, kept running.
#
# macOS has no util-linux `setsid`, so the new session is made in python before the exec: the
# setsid call, the exec and the target all share one pid, which keeps `$!` in lab.sh the pid of
# the process the caller actually wants to track and to `kill` later.
#
# Usage: detach.sh <command> [args...]
set -uo pipefail

if [[ $# -lt 1 ]]; then
  printf 'usage: detach.sh <command> [args...]\n' >&2
  exit 2
fi

exec /usr/bin/env python3 -c '
import os, sys

try:
    os.setsid()
except OSError as exc:  # already a group leader: run anyway, only the isolation is lost
    sys.stderr.write("[detach] setsid failed: %s\n" % exc)
os.execvp(sys.argv[1], sys.argv[1:])
' "$@"
