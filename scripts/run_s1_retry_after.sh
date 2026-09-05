#!/usr/bin/env bash
# Waits for the in-flight S1 resume to finish, then re-runs the two boroughs
# it could not complete:
#   Wandsworth - Overpass dropped the `amenity` category (3rd occurrence);
#                category downloads now retry 4x with exponential backoff.
#   Brent      - refused by a density floor calibrated on inner-London
#                boroughs only; the download was proven complete by two
#                byte-identical runs, and the floor is now 40, not 75.
# Waits rather than running concurrently: one GPU job at a time (rule R5).
cd "$(dirname "$0")/.."
while ! grep -q "S1_RESUME_COMPLETE" /tmp/s1_resume.log 2>/dev/null; do sleep 20; done
echo "=== prior S1 finished; starting retries at $(date +%H:%M:%S) ==="
for B in "Wandsworth" "Brent"; do
  echo "=== RETRY START $B at $(date +%H:%M:%S) ==="
  ./.venv/Scripts/python.exe scripts/run_ucl_comparison_multiyear.py "$B" 2>&1 \
    | grep -E "AccHR|Written|Traceback|PoiDownloadError|Error|density check|retrying" || true
  echo "=== RETRY DONE $B at $(date +%H:%M:%S) ==="
done
echo "S1_RETRY_COMPLETE"
