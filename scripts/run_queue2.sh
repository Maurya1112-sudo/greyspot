#!/usr/bin/env bash
# Waits for the in-flight Wandsworth run, then multi-seeds S5.
# Sequential: one GPU job at a time (R5).
cd "$(dirname "$0")/.."
PIDFILE=/tmp/greyspot_queue.pids
: > "$PIDFILE"; echo "queue $$" >> "$PIDFILE"

echo "=== waiting for Wandsworth at $(date +%H:%M:%S) ==="
while ! grep -q "Written to\|Traceback\|PoiDownloadError" /tmp/wandsworth_tiled.log 2>/dev/null; do sleep 20; done
echo "=== Wandsworth finished at $(date +%H:%M:%S) ==="

echo "=== QUEUE START S5 multiseed at $(date +%H:%M:%S) ==="
./.venv/Scripts/python.exe scripts/run_s5_multiseed.py > /tmp/s5_multiseed.log 2>&1 &
child=$!; echo "s5_multiseed $child" >> "$PIDFILE"; wait $child
echo "=== QUEUE DONE S5 multiseed at $(date +%H:%M:%S) (exit $?) ==="
grep -E "seeds: mean|Written to" /tmp/s5_multiseed.log | tail -4 || true
echo "QUEUE2_COMPLETE at $(date +%H:%M:%S)"
