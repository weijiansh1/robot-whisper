#!/usr/bin/env bash
# Capture 64 flow-noise draws on one (task, init state) with the object state attached.
#
#   usage: run_objstate_capture.sh <task_id> <init_state_id> [port] [gpu_uuid]
#
# Seeds match the original within64-* captures (seed 7, flow noise 1000..1063), so the
# recorded success count is a sanity check against them -- though note the pipeline is
# not bit-reproducible, so agreement is approximate rather than exact.
#
# Output goes to runs/objstate-t<task>s<init>{,-client}; runs/within64-* is never touched.
set -uo pipefail

TASK=${1:?task id}
INIT=${2:?init state id}
PORT=${3:-8320}
# 4g.71gb: the slice with the most free memory and the most SMs.  All three MIG
# slices carry another workload now, so "empty" is no longer an option; this one
# leaves the widest margin.
GPU=${4:-MIG-63b1c8d1-da8b-5f89-acd2-359db55eb1da}

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/objstate/t${TASK}s${INIT}
OUT="$ROOT/runs/objstate-t${TASK}s${INIT}"

if [ -e "$OUT/routes.zarr" ]; then
  echo "refusing to overwrite existing capture at $OUT"; exit 1
fi
mkdir -p "$LOG" "$OUT" "${OUT}-client"
cd "$ROOT"

PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
MOEVLA_DATA_HOME=$BR/moevla-data \
setsid nohup "$BR/envs/model/bin/python" -u serve_with_recorder.py \
  --port "$PORT" --gpu "$GPU" --suite goal \
  --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
  --upstream-root "$BR/upstream/HiMoE-VLA" \
  --libero-wrist-layout released-left --store-full-probs \
  --out "$OUT" > "$LOG/server.log" 2>&1 < /dev/null &

ok=0
for _ in $(seq 1 60); do
  sleep 5
  grep -q "serving on ws" "$LOG/server.log" 2>/dev/null && { ok=1; break; }
  grep -qi "Traceback" "$LOG/server.log" 2>/dev/null && {
    echo "t${TASK}s${INIT} server FAILED"; tail -30 "$LOG/server.log"; exit 1; }
done
[ "$ok" = 1 ] || { echo "server did not come up in 300s"; tail -30 "$LOG/server.log"; exit 1; }
echo "[$(date +%H:%M:%S)] t${TASK}s${INIT} server up on :$PORT"

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u rollout_with_routes.py \
  --port "$PORT" --task-id "$TASK" --benchmark libero_goal \
  --init-state-ids "$INIT" --repeats 64 --noise-seed-base 1000 \
  --no-routing-capture --label "objstate-t${TASK}s${INIT}" --out "${OUT}-client" \
  --libero-root "$BR/upstream/LIBERO" > "$LOG/client.log" 2>&1

echo "[$(date +%H:%M:%S)] t${TASK}s${INIT} client done: $(grep -E '^objstate-t' "$LOG/client.log" | tail -1)"

# SIGTERM, never a plain exit: the writer flushes in the signal handler, and at
# interpreter exit zarr's async executor is already torn down so the last partial
# chunk would be silently lost.
ps -eo pid,cmd | grep "[s]erve_with_recorder.py --port $PORT" | awk '{print $1}' \
  | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 15
echo "[$(date +%H:%M:%S)] t${TASK}s${INIT} done"
