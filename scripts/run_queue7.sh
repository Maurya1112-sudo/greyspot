#!/usr/bin/env bash
# Waits for the head-to-head multi-seed, then multi-seeds the encoder-order
# ablation - the most seed-exposed surviving claim. R5: sequential.
cd "$(dirname "$0")/.."
while ! grep -q "tower_hamlets.headtohead_multiseed_per_window" /tmp/h2h_multiseed.log 2>/dev/null; do sleep 30; done
echo "=== QUEUE7 START encoder multiseed at $(date +%H:%M:%S) ==="
./.venv/Scripts/python.exe scripts/run_encoder_multiseed.py > /tmp/encoder_multiseed.log 2>&1
echo "=== QUEUE7 DONE (exit $?) at $(date +%H:%M:%S) ==="
grep -E "seeds, mean|Written to" /tmp/encoder_multiseed.log | tail -4 || true
echo "QUEUE7_COMPLETE"
