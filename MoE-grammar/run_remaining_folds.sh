#!/usr/bin/env bash
# Run the outstanding full-40 folds one at a time, waiting for room in the shared
# 8 GiB cgroup. Other jobs share this limit, so starting without headroom just gets
# the fold OOM-killed several minutes in.
set -u
cd /home/jovyan/work/himoe-vla/MoE-grammar

LIMIT=$(cat /sys/fs/cgroup/memory.max)
NEEDED=$((4300 * 1024 * 1024))   # observed fold peak plus margin
MAX_ATTEMPTS=6

wait_for_room() {
  for _ in $(seq 1 120); do
    local used free
    used=$(cat /sys/fs/cgroup/memory.current)
    free=$((LIMIT - used))
    if [ "$free" -ge "$NEEDED" ]; then
      echo "$(date +%H:%M:%S) free=$((free / 1024 / 1024))MB - starting"
      return 0
    fi
    sleep 30
  done
  echo "$(date +%H:%M:%S) gave up waiting for memory"
  return 1
}

for fold in "$@"; do
  if [ -f "results-full40-v2-fold$fold/summary.json" ]; then
    echo "fold $fold already complete"
    continue
  fi
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    wait_for_room || break
    echo "$(date +%H:%M:%S) fold $fold attempt $attempt"
    if CUDA_VISIBLE_DEVICES=0 python -u -m moe_grammar.run_full40_audit \
        --fold "$fold" \
        --features-dir artifacts/features-full40-v2 \
        --output-dir "results-full40-v2-fold$fold" \
        > "logs/final-fold$fold.log" 2>&1; then
      echo "$(date +%H:%M:%S) fold $fold done"
      break
    fi
    echo "$(date +%H:%M:%S) fold $fold attempt $attempt failed (likely OOM); retrying"
    rm -rf "results-full40-v2-fold$fold"
    sleep 60
  done
done
echo "REMAINING_FOLDS_FINISHED"
