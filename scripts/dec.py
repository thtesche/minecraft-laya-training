#!/usr/bin/env python3
"""Pretty-print decisions.jsonl: keys at --depth get their own line, deeper values stay compact.

Usage:
  tail -f runs/decisions.jsonl | scripts/dec.py
  tail -n 5 runs/decisions.jsonl | scripts/dec.py --keys ts,stage,tool,label,ok,status
  scripts/dec.py --depth 2 < runs/decisions.jsonl
"""

import argparse
import json
import re
import sys

TOKEN = re.compile(r'("(?:[^"\\]|\\.)*")|\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?')

def colorize(compact: str) -> str:
    out, i = [], 0
    for m in TOKEN.finditer(compact):
        out.append(compact[i:m.start()])
        t = m.group(0)
        if t.startswith('"'):
            out.append(f"\033[0;32m{t}\033[0m")      # string -> green
        elif t in ("true", "false", "null"):
            out.append(f"\033[0;90m{t}\033[0m")      # bool/null -> dim
        else:
            out.append(f"\033[0;39m{t}\033[0m")      # number
        i = m.end()
    out.append(compact[i:])
    return "".join(out)

def render(value, depth, indent, color: bool) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    if depth <= 0 or not isinstance(value, (dict, list)) or not value:
        compact = json.dumps(value, separators=(",", ":"))
        return colorize(compact) if color else compact
    if isinstance(value, dict):
        parts = [
            f'{inner}"{k}": {render(v, depth - 1, indent + 2, color)}'
            for k, v in value.items()
        ]
        open_, close_ = "{", "}"
    else:
        parts = [inner + render(v, depth - 1, indent + 2, color) for v in value]
        open_, close_ = "[", "]"
    body = ",\n".join(parts)
    if color:
        body = re.sub(r'^( +)"([^"]*)":', rf'\1\033[1;34m"\2"\033[0m:', body, flags=re.M)
    return f"{open_}\n{body}\n{pad}{close_}"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=1, help="levels to unfold (default 1)")
    ap.add_argument("--keys", help="comma-separated top-level keys to keep")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--file", help="read this file instead of stdin")
    args = ap.parse_args()
    keys = args.keys.split(",") if args.keys else None

    sys.stdout.reconfigure(line_buffering=True)
    stream = open(args.file) if args.file else sys.stdin
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"skip: {e}", file=sys.stderr)
            continue
        if keys:
            obj = {k: obj[k] for k in keys if k in obj}
        print(render(obj, args.depth, 0, not args.no_color))

if __name__ == "__main__":
    main()
