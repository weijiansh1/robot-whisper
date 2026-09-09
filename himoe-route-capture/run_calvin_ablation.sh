#!/usr/bin/env bash
# Does the HB-MoE *routing decision* matter on CALVIN, or only the block's presence?
#
# The paper's appendix already compares "with MoE" against a parameter-matched
# dense baseline (Sum 4.012 vs 3.801) -- that is the block_off contrast.  It never
# tests whether the top-4 choice carries any of that gain.  On LIBERO-Goal we
# found it does not (random routing p=0.83, either branch alone p=1.000/0.815,
# whole block -18 points).  These arms ask the same question on CALVIN.
#
# Arms are paired: the evaluator's 1000-sequence universe is fixed and each
# sequence has a deterministic initial condition, so arm-vs-arm is a matched
# comparison by sequence index.  The control is re-run on the same MIG slices as
# the ablations -- the released 1000-run from 2026-08-04 used a different GPU
# configuration and cross-partition trajectories are not bit-reproducible.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
ART=/home/jovyan/work/himoe-vla-cache/himoe-calvin-alignment/formal-artifacts
BR=/home/jovyan/.cache/himoe-libero-bridge
CAL=/home/jovyan/.cache/himoe-calvin-alignment
GPU2G=MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a
GPU1G=MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3
LOG=/tmp/cal-abl
N_SEQ=200
mkdir -p "$LOG"

# One server per MIG slice.  Measured footprint is 15.7 GB per HiMoE CALVIN
# server against a 33.28 GB slice, so two servers on one slice leaves under 2 GB
# of headroom and the OOM killer takes them out mid-load (observed: 2 of 3 killed).
# shard : port : gpu : start : end
SHARDS=("s0:8200:$GPU2G:0:100" "s1:8201:$GPU1G:100:200")

stop_servers() {
  ps -eo pid,cmd | grep "[s]erve_ablated_calvin.py" | awk '{print $1}' > "$LOG/pids" || true
  while read -r p; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; done < "$LOG/pids"
  sleep 6
  while read -r p; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done < "$LOG/pids"
  sleep 3
}

for MODE in none router_random block_off; do
  DONE=1
  for s in "${SHARDS[@]}"; do
    IFS=: read -r name port gpu start end <<< "$s"
    f="$ART/abl-calvin-$MODE-$name/sequences.jsonl"
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq $((end-start)) ] || DONE=0
  done
  if [ "$DONE" = "1" ]; then echo "[$(date +%H:%M:%S)] $MODE already complete"; continue; fi

  echo "[$(date +%H:%M:%S)] ===== arm $MODE ====="
  stop_servers

  for s in "${SHARDS[@]}"; do
    IFS=: read -r name port gpu start end <<< "$s"
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONPATH=/home/jovyan/work/himoe-flower-calvin/src:$ROOT \
    setsid nohup "$BR/envs/model/bin/python" -u "$ROOT/serve_ablated_calvin.py" \
      --checkpoint-dir "$CAL/checkpoints/HiMoE-VLA-CALVIN-D" \
      --upstream-root "$BR/upstream/HiMoE-VLA" \
      --calvin-root "$CAL/upstream/calvin" \
      --port "$port" --mode "$MODE" \
      > "$LOG/srv-$MODE-$name.log" 2>&1 < /dev/null &
  done

  for s in "${SHARDS[@]}"; do
    IFS=: read -r name port gpu start end <<< "$s"
    ok=0
    for _ in $(seq 1 60); do
      sleep 5
      grep -q "server listening" "$LOG/srv-$MODE-$name.log" 2>/dev/null && { ok=1; break; }
      grep -qi "Traceback" "$LOG/srv-$MODE-$name.log" 2>/dev/null && { echo "server $name FAILED"; tail -20 "$LOG/srv-$MODE-$name.log"; exit 1; }
    done
    [ "$ok" = "1" ] || { echo "server $name never ready"; exit 1; }
  done
  echo "[$(date +%H:%M:%S)] $MODE: all 4 servers up"

  for s in "${SHARDS[@]}"; do
    IFS=: read -r name port gpu start end <<< "$s"
    OUT="$ART/abl-calvin-$MODE-$name"; mkdir -p "$OUT"
    cd /home/jovyan/work/himoe-flower-calvin
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/jovyan/work/himoe-flower-calvin/src \
    setsid nohup bash scripts/calvin.sh \
      --dataset-root "$CAL/datasets/task_D_D" --calvin-root "$CAL/upstream/calvin" \
      --himoe-root "$BR/upstream/HiMoE-VLA" --output-dir "$OUT" \
      --host 127.0.0.1 --port "$port" --policy-backend himoe \
      --start-index "$start" --end-index "$end" \
      > "$LOG/cli-$MODE-$name.log" 2>&1 < /dev/null &
    sleep 2
  done

  while pgrep -f "himoe-calvin-eval" > /dev/null; do sleep 30; done
  n=0
  for s in "${SHARDS[@]}"; do
    IFS=: read -r name port gpu start end <<< "$s"
    n=$((n + $(wc -l < "$ART/abl-calvin-$MODE-$name/sequences.jsonl" 2>/dev/null || echo 0)))
  done
  echo "[$(date +%H:%M:%S)] $MODE done: $n / $N_SEQ sequences"
done

stop_servers
echo "[$(date +%H:%M:%S)] ===== all arms done ====="
