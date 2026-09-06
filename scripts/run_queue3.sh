#!/usr/bin/env bash
# Waits for the S5 multi-seed, then the cheap reproducibility check.
# Sequential: one GPU job at a time (R5).
cd "$(dirname "$0")/.."
echo "=== waiting for S5 multiseed at $(date +%H:%M:%S) ==="
while ! grep -q "QUEUE2_COMPLETE" /tmp/queue2.log 2>/dev/null; do sleep 30; done
echo "=== S5 finished at $(date +%H:%M:%S) ==="
echo "=== QUEUE3 START reproducibility check at $(date +%H:%M:%S) ==="
./.venv/Scripts/python.exe scripts/run_reproducibility_check.py > /tmp/repro_check.log 2>&1
echo "=== QUEUE3 DONE (exit $?) at $(date +%H:%M:%S) ==="
grep -E "current code|V8 run log|committed CSV|VERDICT" /tmp/repro_check.log || true
echo "QUEUE3_COMPLETE"
