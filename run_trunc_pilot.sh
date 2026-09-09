#!/bin/bash
# Denoise-truncation CPU pilot: smoke then two paired arms (see TRUNC_PILOT_PREREG.md).
set -u
cd /home/jovyan/work/himoe-vla/himoe-route-capture
OUT=/home/jovyan/work/himoe-vla/runs-trunc-pilot
LOG=$OUT/logs
mkdir -p "$LOG"
SCENES=0,3,7,10,13,16,20,23,26,29,33,36,39,42,46,49

run () { # id rounds scene_ids draws
  local id=$1 r=$2 sc=$3 dr=$4
  mkdir -p "$OUT/$id"
  cp "$OUT/_tasks.json" "$OUT/$id/_tasks.json"
  echo "[$(date +%H:%M:%S)] start $id (r=$r)"
  python3 -u run_corpus_capture.py \
    --root "$OUT/$id" --gpu cpu --port 8461 \
    --benchmarks libero_10 --tasks 8 \
    --scene-ids "$sc" --draws "$dr" \
    --inference-timeout 1800 \
    --trunc-rounds "$r" > "$LOG/$id.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] done  $id rc=$rc"
  return $rc
}

run smoke-r10 10 0 2 || { echo "SMOKE r10 FAILED"; exit 1; }
run smoke-r6  6  0 2 || { echo "SMOKE r6 FAILED";  exit 1; }
run arm-r10 10 "$SCENES" 8 || { echo "ARM r10 FAILED"; exit 1; }
run arm-r6  6  "$SCENES" 8 || { echo "ARM r6 FAILED";  exit 1; }
echo "[$(date +%H:%M:%S)] pilot complete"
