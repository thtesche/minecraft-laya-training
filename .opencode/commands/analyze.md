---
description: Analyse the newest run log and compare it with an earlier run
---
Run `./scripts/lab.sh analyze ${1:+--baseline $1}` and read the report it writes.

Give me the findings as a ranked list: symptom, the iterations that show it, the hypothesis,
the file to change, and the test that would fail without the fix. Mark each one as
autonomous or needing my GO, and say which belong to the brain and which to the
mine-ai-mcp server on `bot-run`.
