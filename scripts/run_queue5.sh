#!/usr/bin/env bash
# Waits for the S5 multi-seed sweep, then regenerates the headline
# multi-seed figures with per-window output on current code. R5: sequential.
cd "$(dirname "$0")/.."
while ! grep -q "QUEUE4_COMPLETE" /tmp/queue4.log 2>/dev/null; do sleep 30; done
echo "=== QUEUE5 START headline multiseed at $(date +%H:%M:%S) ==="
./.venv/Scripts/python.exe scripts/run_headline_multiseed.py > /tmp/headline_multiseed.log 2>&1
echo "=== QUEUE5 DONE (exit $?) at $(date +%H:%M:%S) ==="
grep -E "seeds, mean|Written to" /tmp/headline_multiseed.log | tail -6 || true
echo "QUEUE5_COMPLETE"
