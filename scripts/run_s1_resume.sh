#!/usr/bin/env bash
# S1 RESUME - the three boroughs blocked by the 2026-09-05 Overpass outage.
# Camden and Kensington & Chelsea already completed and are NOT re-run.
# Wandsworth's truncated POI cache was quarantined to
# CORRUPT_wandsworth_poi_partial_download.csv.bak and Brent's zero-POI cache
# deleted, so both re-download - now behind the density guard in
# greyspot.ingest.poi, which refuses to cache an incomplete response.
cd "$(dirname "$0")/.."
for B in "Wandsworth" "Brent" "City of London"; do
  echo "=== S1 START $B at $(date +%H:%M:%S) ==="
  ./.venv/Scripts/python.exe scripts/run_ucl_comparison_multiyear.py "$B" 2>&1 \
    | grep -E "AccHR|Written|Traceback|PoiDownloadError|Error|Built|density check" || true
  echo "=== S1 DONE $B at $(date +%H:%M:%S) ==="
done
echo "S1_RESUME_COMPLETE"
