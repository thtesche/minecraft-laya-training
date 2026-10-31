#!/bin/bash
# Minecraft Laya Run Monitor
# Usage: ./monitor.sh (call it every minute via cron or manually)

cd /Users/thtesche/VibeCoding/minecraft-laya-training
LOG=$(ls -t runs/run[0-9]*.log 2>/dev/null | head -1)
RUNNAME=$(basename "$LOG" .log)

if [ -z "$LOG" ]; then
  echo "[ERROR] No run logs found"
  exit 1
fi

if ! pgrep -f 'mlaya.cli run' > /dev/null 2>&1; then
  echo "[$RUNNAME] ✓ COMPLETE"
  tail -2 "$LOG" | grep -E "passed|error|died" || tail -1 "$LOG"
  exit 0
fi

# Run is active
LAST_ENTRY=$(tail -1 runs/decisions.jsonl 2>/dev/null)

ITER=$(echo "$LAST_ENTRY" | jq -r '.iteration // "?"' 2>/dev/null)
LABEL=$(echo "$LAST_ENTRY" | jq -r '.label // "?"' 2>/dev/null)
HP=$(echo "$LAST_ENTRY" | jq -r '.state.health // "?"' 2>/dev/null)
FOOD=$(echo "$LAST_ENTRY" | jq -r '.state.food // "?"' 2>/dev/null)
POS=$(echo "$LAST_ENTRY" | jq -r '.state.position[0:3] // []' 2>/dev/null | tr '\n' ' ')
DIM=$(echo "$LAST_ENTRY" | jq -r '.state.dimension // "?"' 2>/dev/null)
CANDS=$(echo "$LAST_ENTRY" | jq -r '.candidates // []' 2>/dev/null | jq -c 'if length > 3 then .[0:3] + ["..."] else . end' 2>/dev/null)

# Check anomalies
ANOMALY=""
if [ "$HP" != "?" ]; then
  HP_NUM=$(echo "$HP" | cut -d. -f1)
  if [ "$HP_NUM" -le 8 ]; then ANOMALY="⚠️ LOW HP: $HP"; fi
fi
if [ "$FOOD" != "?" ]; then
  FOOD_NUM=$(echo "$FOOD" | cut -d. -f1)
  if [ "$FOOD_NUM" -le 8 ]; then ANOMALY="$ANOMALY ⚠️ LOW FOOD: $FOOD"; fi
fi

if [ ! -z "$ANOMALY" ]; then
  echo "[$RUNNAME] $ANOMALY"
  echo "  it $ITER $LABEL | $DIM | pos $POS"
else
  echo "[$RUNNAME] it $ITER | $LABEL | HP=$HP food=$FOOD | $DIM | cands=$CANDS"
fi
