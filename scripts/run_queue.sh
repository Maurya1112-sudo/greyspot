#!/usr/bin/env bash
# Sequential GPU work queue (rule R5: one GPU job at a time).
#
# Writes its own PID and each child's PID to /tmp/greyspot_queue.pids so a
# kill can be verified rather than assumed - rule R16, added after a
# `pkill -f` silently failed to kill a shell loop on 2026-09-05 and left
# two GPU jobs running concurrently.
#
# To stop it:  bash scripts/kill_queue.sh
set -u
cd "$(dirname "$0")/.."
PIDFILE=/tmp/greyspot_queue.pids
echo "queue $$" > "$PIDFILE"

wait_for() {  # wait_for <logfile> <marker>
  while ! grep -q "$2" "$1" 2>/dev/null; do
    sleep 20
    # if the log stops existing, give up rather than spin forever
    [ -f "$1" ] || return 1
  done
}

run() {  # run <label> <logfile> <script> <args...>
  local label="$1" log="$2"; shift 2
  echo "=== QUEUE START $label at $(date +%H:%M:%S) ==="
  ./.venv/Scripts/python.exe "$@" > "$log" 2>&1 &
  local pid=$!
  echo "child $pid $label" >> "$PIDFILE"
  wait $pid
  echo "=== QUEUE DONE $label at $(date +%H:%M:%S) (exit $?) ==="
  grep -E "AccHR@20 across|PoiDownloadError|Traceback" "$log" | tail -8 || true
}

# 1. Wait for the in-flight C2 Tower Hamlets run to finish.
echo "=== waiting for C2 (Tower Hamlets) to finish at $(date +%H:%M:%S) ==="
wait_for /tmp/c2_th.log "Written to"
echo "=== C2 finished; queue starting at $(date +%H:%M:%S) ==="

# 2. Brent - POI now cached (download proven complete by two byte-identical
#    runs), so this needs no Overpass.
run "S1 Brent" /tmp/brent_retry.log     scripts/run_ucl_comparison_multiyear.py "Brent"

# 3. S5 deep-history null on a SECOND borough. The finding - that the GNN
#    gains nothing from history beyond 5 years while a trivial sort gains
#    +2.95 - currently rests on Lambeth alone, and single-borough results
#    are exactly what this project has repeatedly watched fail. Westminster
#    has all caches, so no network needed.
run "S5 deep history Westminster" /tmp/s5_westminster.log     scripts/run_s5_deep_history.py "Westminster"

# 4. Wandsworth LAST: still needs Overpass, which is degraded (resets
#    connections mid-transfer). Most likely to fail, so it blocks nothing.
run "S1 Wandsworth" /tmp/wandsworth_retry.log     scripts/run_ucl_comparison_multiyear.py "Wandsworth"

echo "QUEUE_COMPLETE at $(date +%H:%M:%S)"
