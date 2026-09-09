#!/usr/bin/env bash
# Measure R_pi(s_11): the routing the policy reaches at one state by resampling only
# its own flow noise, so the transplant's flips can be tested for membership.
#
# 16 seeds x (11 prefix calls + 64 resampled calls) = ~1200 inferences, ~20 min.
# No environment step happens between the resampled calls, so every draw sees the
# identical state.
#
# Must run on the same slice as the transplant it is interpreting (4g.71gb): a
# different SM count moves the trajectory, so s_11 would not be the same state.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/routing-cloud
GPU=${CLOUD_GPU:-MIG-63b1c8d1-da8b-5f89-acd2-359db55eb1da}   # 4g.71gb
PORT=${CLOUD_PORT:-8350}
BRANCH=${CLOUD_BRANCH:-11}
OUT="$ROOT/runs/routing-cloud-k${BRANCH}-s24"
PATCH_SUMMARIES="$ROOT/runs/patch-probe-k${BRANCH}-s24/summaries.json"

[ -f "$PATCH_SUMMARIES" ] || { echo "missing $PATCH_SUMMARIES"; exit 1; }
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

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u rollout_routing_cloud.py \
  --port "$PORT" --task-id 0 --benchmark libero_goal --init-state-id 24 \
  --branch-control "$BRANCH" --n-recipients 16 --n-cloud 64 --noise-seed-base 1000 \
  --label "routing-cloud-k${BRANCH}-s24" --out "$OUT" \
  --libero-root "$BR/upstream/LIBERO" "$@" 2>&1 | tee "$LOG/client.log"

ps -eo pid,cmd | grep "[s]erve_patched_router.py --port $PORT" | awk '{print $1}' \
  | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 10

for m in jaccard tv; do
  "$BR/envs/libero/bin/python" analyze_routing_cloud.py \
    --cloud-dir "$OUT" --patch-summaries "$PATCH_SUMMARIES" --metric "$m" \
    --out "$ROOT/analysis/routing-cloud-k${BRANCH}-s24/${m}.json"
done
echo "[$(date +%H:%M:%S)] done"
