#!/usr/bin/env bash
# Transplant control-step-11 HB routing between rollouts.
#
# 16 recipients x (1 recording pass + 4 arms) = 80 rollouts, ~45 min.
# Read the self-patch identity line at the end first: if the self arm does not
# reproduce phase A bit-exactly, every other arm's action delta is contaminated
# by that much drift and the numbers mean nothing.
#
# Same MIG slice, wrist layout and init state as the decode window it is testing
# (released-left, init 24) -- the 0.675 residual decode at step 11 was measured
# there, so a causal test has to run in the same regime.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/patch-probe
GPU=${PATCH_GPU:-MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3}   # 1g.35gb
PORT=${PATCH_PORT:-8340}
BRANCH=${PATCH_BRANCH:-11}
OUT="$ROOT/runs/patch-probe-k${BRANCH}-s24"

mkdir -p "$LOG" "$OUT"
cd "$ROOT"

PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
MOEVLA_DATA_HOME=$BR/moevla-data \
setsid nohup "$BR/envs/model/bin/python" -u serve_patched_router.py \
  --port "$PORT" --gpu "$GPU" --suite goal \
  --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
  --upstream-root "$BR/upstream/HiMoE-VLA" \
  --libero-wrist-layout released-left \
  --out "$OUT" > "$LOG/server.log" 2>&1 < /dev/null &

ok=0
for _ in $(seq 1 60); do
  sleep 5
  grep -q "serving on ws" "$LOG/server.log" 2>/dev/null && { ok=1; break; }
  grep -qi "Traceback" "$LOG/server.log" 2>/dev/null && {
    echo "server FAILED"; tail -30 "$LOG/server.log"; exit 1; }
done
[ "$ok" = 1 ] || { echo "server did not come up in 300s"; tail -30 "$LOG/server.log"; exit 1; }
echo "[$(date +%H:%M:%S)] server up on :$PORT"
grep "HB gates" "$LOG/server.log"

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u rollout_patch_probe.py \
  --port "$PORT" --task-id 0 --benchmark libero_goal --init-state-id 24 \
  --branch-control "$BRANCH" --n-recipients 16 --noise-seed-base 1000 \
  --label "patch-probe-k${BRANCH}-s24" --out "$OUT" \
  --libero-root "$BR/upstream/LIBERO" "$@" 2>&1 | tee "$LOG/client.log"

ps -eo pid,cmd | grep "[s]erve_patched_router.py --port $PORT" | awk '{print $1}' \
  | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 10
echo "[$(date +%H:%M:%S)] done"
