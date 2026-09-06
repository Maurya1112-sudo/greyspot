#!/usr/bin/env bash
# Waits for the headline regeneration, then probes determinism. R5: sequential.
cd "$(dirname "$0")/.."
while ! grep -q "QUEUE5_COMPLETE" /tmp/queue5.log 2>/dev/null; do sleep 30; done
echo "=== QUEUE6 START determinism probe at $(date +%H:%M:%S) ==="
./.venv/Scripts/python.exe scripts/run_determinism_probe.py 3 > /tmp/determinism.log 2>&1
echo "=== QUEUE6 DONE (exit $?) at $(date +%H:%M:%S) ==="
grep -E "spread|VERDICT|DETERMINISTIC|NOT deterministic" /tmp/determinism.log | tail -12 || true
echo "QUEUE6_COMPLETE"
