#!/usr/bin/env bash
# 64 flow-noise draws on one fixed (task, init state), with full 32-dim router
# probabilities captured server-side.  Replicates the within64-s24 design on
# other LIBERO-Goal tasks so the mid-window finding (steps 4-10: routing beats
# proprioception) can be tested outside the single scene it was found in.
#
# usage: run_within64_task.sh <taskA> <initA> <taskB> <initB>
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/mt64
mkdir -p "$LOG"

JOBS=("$1:$2:8310:MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a" "$3:$4:8311:MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3")

for j in "${JOBS[@]}"; do
  IFS=: read -r task init port gpu <<< "$j"
  OUT="$ROOT/runs/within64-t${task}s${init}"; mkdir -p "$OUT"
  cd "$ROOT"
  # NOTE: do not also export CUDA_VISIBLE_DEVICES here -- --gpu sets it inside,
  # and pre-restricting to the same MIG UUID makes it unresolvable and the
  # process hangs in CUDA init with an empty log.
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
  MOEVLA_DATA_HOME=$BR/moevla-data \
  setsid nohup "$BR/envs/model/bin/python" -u serve_with_recorder.py \
    --port "$port" --gpu "$gpu" --suite goal \
    --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
    --upstream-root "$BR/upstream/HiMoE-VLA" \
    --libero-wrist-layout released-left --store-full-probs \
    --out "$OUT" > "$LOG/srv-t${task}s${init}.log" 2>&1 < /dev/null &
done

for j in "${JOBS[@]}"; do
  IFS=: read -r task init port gpu <<< "$j"
  ok=0
  for _ in $(seq 1 60); do
    sleep 5
    grep -q "serving on ws" "$LOG/srv-t${task}s${init}.log" 2>/dev/null && { ok=1; break; }
    grep -qi "Traceback" "$LOG/srv-t${task}s${init}.log" 2>/dev/null && { echo "t$task s$init server FAILED"; tail -20 "$LOG/srv-t${task}s${init}.log"; exit 1; }
  done
  [ "$ok" = 1 ] || { echo "t$task s$init server never ready"; exit 1; }
  echo "[$(date +%H:%M:%S)] t$task s$init server ready on $port"
done

for j in "${JOBS[@]}"; do
  IFS=: read -r task init port gpu <<< "$j"
  OUT="$ROOT/runs/within64-t${task}s${init}-client"; mkdir -p "$OUT"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
  setsid nohup "$BR/envs/libero/bin/python" -u rollout_with_routes.py \
    --port "$port" --task-id "$task" --benchmark libero_goal \
    --init-state-ids "$init" --repeats 64 --noise-seed-base 1000 \
    --no-routing-capture --label "within64-t${task}s${init}" --out "$OUT" \
    --libero-root "$BR/upstream/LIBERO" \
    > "$LOG/cli-t${task}s${init}.log" 2>&1 < /dev/null &
  sleep 2
done

while pgrep -f "rollout_with_routes.py --port 831" > /dev/null; do sleep 30; done
sleep 20
# SIGTERM, never a plain exit: the writer flushes in the signal handler, and at
# interpreter exit zarr's async executor is already torn down so the last
# partial chunk would be silently lost.
ps -eo pid,cmd | grep "[s]erve_with_recorder.py --port 831" | awk '{print $1}' | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 15
for j in "${JOBS[@]}"; do
  IFS=: read -r task init port gpu <<< "$j"
  echo "[$(date +%H:%M:%S)] $(grep -E '^within64-t' "$LOG/cli-t${task}s${init}.log" | tail -1)"
done
echo "[$(date +%H:%M:%S)] done"
