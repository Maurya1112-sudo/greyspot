#!/usr/bin/env bash
# Waits for the reproducibility check, then restarts the S5 multi-seed
# sweep (which resumes from seed 42's completed windows). R5: sequential.
cd "$(dirname "$0")/.."
while ! grep -q "QUEUE3_COMPLETE" /tmp/queue3.log 2>/dev/null; do sleep 15; done
echo "=== QUEUE4 START S5 multiseed (resuming) at $(date +%H:%M:%S) ==="
./.venv/Scripts/python.exe scripts/run_s5_multiseed.py >> /tmp/s5_multiseed.log 2>&1
echo "=== QUEUE4 DONE (exit $?) at $(date +%H:%M:%S) ==="
echo "QUEUE4_COMPLETE"
