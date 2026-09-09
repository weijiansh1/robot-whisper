#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/work/himoe-vla/himoe-route-capture
BRIDGE=/home/jovyan/.cache/himoe-libero-bridge
LOG_DIR=${ADAPTIVE_LOG_DIR:-/tmp/adaptive-denoise-stop}
PORT=${ADAPTIVE_PORT:-8482}
OUT=${ADAPTIVE_OUT:-$ROOT/runs/adaptive-denoise-stop-t0s24-seed2000}
THRESHOLD=${ADAPTIVE_THRESHOLD:-0.0047}
ARMS=${ADAPTIVE_ARMS:-baseline,adaptive}
SUITE=${ADAPTIVE_SUITE:-goal}
BENCHMARK=${ADAPTIVE_BENCHMARK:-libero_goal}
CHECKPOINT=${ADAPTIVE_CHECKPOINT:-HiMoE-VLA-Libero-Goal}
TASK_ID=${ADAPTIVE_TASK_ID:-0}
INIT_STATE_ID=${ADAPTIVE_INIT_STATE_ID:-24}
FLOW_NOISE_SEEDS=${ADAPTIVE_FLOW_NOISE_SEEDS:-2000}
MAX_STEPS=${ADAPTIVE_MAX_STEPS:-300}
ROUTE_METRIC=${ADAPTIVE_ROUTE_METRIC:-probability_hellinger}
SERVER_PID=""

if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing output: $OUT" >&2
  exit 1
fi
mkdir -p "$LOG_DIR" "$OUT"

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT"
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
MOEVLA_DATA_HOME=$BRIDGE/moevla-data \
"$BRIDGE/envs/model/bin/python" -u serve_adaptive_denoise.py \
  --port "$PORT" --gpu cpu --threads 64 --suite "$SUITE" \
  --checkpoint-dir "$BRIDGE/checkpoints/$CHECKPOINT" \
  --upstream-root "$BRIDGE/upstream/HiMoE-VLA" \
  --libero-wrist-layout checkpoint-right \
  --threshold "$THRESHOLD" --min-steps 8 --consecutive 2 \
  --route-metric "$ROUTE_METRIC" \
  --verify-baseline --out "$OUT/server" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 60); do
  sleep 5
  if grep -q "serving on ws" "$LOG_DIR/server.log" 2>/dev/null; then
    ready=1
    break
  fi
  if grep -qi "Traceback" "$LOG_DIR/server.log" 2>/dev/null; then
    tail -80 "$LOG_DIR/server.log"
    exit 1
  fi
done
if [ "$ready" -ne 1 ]; then
  tail -80 "$LOG_DIR/server.log"
  exit 1
fi

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BRIDGE/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BRIDGE/upstream/LIBERO:$BRIDGE/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BRIDGE/envs/libero/bin/python" -u rollout_adaptive_denoise.py \
  --port "$PORT" --benchmark "$BENCHMARK" --task-id "$TASK_ID" \
  --init-state-id "$INIT_STATE_ID" --flow-noise-seeds "$FLOW_NOISE_SEEDS" \
  --max-steps "$MAX_STEPS" --out "$OUT/client" \
  --arms "$ARMS" \
  --libero-root "$BRIDGE/upstream/LIBERO" 2>&1 | tee "$LOG_DIR/client.log"

cleanup
SERVER_PID=""
echo "adaptive denoise pilot saved to $OUT"
