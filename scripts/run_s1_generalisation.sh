#!/usr/bin/env bash
# S1 - generalisation to five boroughs beyond the three benchmark ones.
# Sequential (one GPU job at a time). Each borough builds its own graph
# and POI cache on first run, so early boroughs are slower.
cd "$(dirname "$0")/.."
for B in "Camden" "Kensington and Chelsea" "Wandsworth" "Brent" "City of London"; do
  SLUG=$(echo "$B" | tr '[:upper:] ' '[:lower:]_')
  echo "=== S1 START $B at $(date +%H:%M:%S) ==="
  ./.venv/Scripts/python.exe scripts/run_ucl_comparison_multiyear.py "$B" 2>&1 | grep -E "AccHR|Written|Traceback|Error|Built" || true
  echo "=== S1 DONE $B at $(date +%H:%M:%S) ==="
done
echo "S1_ALL_BOROUGHS_COMPLETE"
