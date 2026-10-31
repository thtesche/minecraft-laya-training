---
description: Start a lab run: world container, MCP host from bot-run, mlaya loop
---
Run the lab: `./scripts/lab.sh up --n ${1:-600}`.

Then watch it with `./scripts/lab.sh status` every few minutes, and when the run ends
analyse it with `./scripts/lab.sh analyze --baseline runs/history/decisions-runNN.jsonl`.
Report the refusals, stalls, movement per tick and the milestone timeline, and say which of
the findings you fixed on your own and which need my GO.
