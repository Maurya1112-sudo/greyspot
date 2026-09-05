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

# 2. C6 head-to-head on Tower Hamlets. Needs no network - the borough's
#    graph and POI caches already exist. Closes the substitution-effect
#    claim, which currently rests on Westminster alone and is in the
#    preprint.
run "C6 head-to-head Tower Hamlets" /tmp/c6_h2h_th2.log \
    scripts/run_headtohead_stzitd.py "Tower Hamlets"

# 3. Brent - POI now cached (download proven complete by two byte-identical
#    runs), so this needs no Overpass either.
run "S1 Brent" /tmp/brent_retry.log \
    scripts/run_ucl_comparison_multiyear.py "Brent"

# 4. Wandsworth LAST: it still needs Overpass, which is degraded (resets
#    connections mid-transfer). Most likely to fail, so it blocks nothing.
run "S1 Wandsworth" /tmp/wandsworth_retry.log \
    scripts/run_ucl_comparison_multiyear.py "Wandsworth"

echo "QUEUE_COMPLETE at $(date +%H:%M:%S)"
