#!/usr/bin/env bash
# Commitment curve: branch the rollout at k control steps, vary only the future
# flow noise, and ask whether the outcome can still change.
#
# k is chosen to bracket the routing decode window.  Steps 11-12 are where the
# routing state beats the full physical world state (robot qpos + every object
# joint); step 13 onward is a pure state readout.  If rows at k=11 are still
# mixed, the routing signal arrives while the outcome is genuinely undecided.
# If they are already stable, the 0.675 decode is post-hoc and the window means
# nothing for control.  k=4 and k=8 are the undecided anchors, k=14 the decided one.
#
# 6 branches x 8 prefixes x 8 future streams = 384 rollouts, ~3 h.
# The planner visits every branch inside each 2x2 row/col block, so an interrupted
# run still holds a balanced sample across k.  Re-run with --resume to continue.
#
# SM count changes cuBLAS kernel selection and with it the per-episode trajectory,
# so the slice has to be recorded and held fixed for the whole grid.  2g.35gb is
# what the pilot and the pre-registered pre-action protocol used; 4g.71gb is what
# was actually free.  Either is internally consistent -- the grid's own prefix
# replay is checked bit-exactly -- but cells from different slices cannot be pooled.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/prefix-k6
GPU=${PREFIX_GPU:-MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a}   # default 2g.35gb, 16 SM
PORT=${PREFIX_PORT:-8330}
TASK=0
INIT=24
OUT="$ROOT/runs/prefix-commitment-k6-s24-mig2g"

mkdir -p "$LOG" "${OUT}-server" "${OUT}-client"
cd "$ROOT"

PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
MOEVLA_DATA_HOME=$BR/moevla-data \
setsid nohup "$BR/envs/model/bin/python" -u serve_with_recorder.py \
  --port "$PORT" --gpu "$GPU" --suite goal \
  --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
  --upstream-root "$BR/upstream/HiMoE-VLA" \
  --libero-wrist-layout released-left --store-full-probs \
  --out "${OUT}-server" > "$LOG/server.log" 2>&1 < /dev/null &

ok=0
for _ in $(seq 1 60); do
  sleep 5
  grep -q "serving on ws" "$LOG/server.log" 2>/dev/null && { ok=1; break; }
  grep -qi "Traceback" "$LOG/server.log" 2>/dev/null && {
    echo "server FAILED"; tail -30 "$LOG/server.log"; exit 1; }
done
[ "$ok" = 1 ] || { echo "server did not come up in 300s"; tail -30 "$LOG/server.log"; exit 1; }
echo "[$(date +%H:%M:%S)] server up on :$PORT"
grep -E "n_suffix|discovered" "$LOG/server.log"

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u rollout_prefix_commitment.py \
  --port "$PORT" --task-id "$TASK" --benchmark libero_goal --init-state-id "$INIT" \
  --branch-controls 4 8 10 11 12 14 --n-prefix 8 --n-future 8 \
  --label "prefix-commitment-k6-s${INIT}" --out "${OUT}-client" \
  --libero-root "$BR/upstream/LIBERO" "$@" > "$LOG/client.log" 2>&1

echo "[$(date +%H:%M:%S)] client exit=$? : $(tail -1 "$LOG/client.log")"

# SIGTERM, never a plain exit: the Zarr writer flushes in the signal handler and
# at interpreter exit the async executor is already gone, losing the last chunk.
ps -eo pid,cmd | grep "[s]erve_with_recorder.py --port $PORT" | awk '{print $1}' \
  | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 15
echo "[$(date +%H:%M:%S)] done"
