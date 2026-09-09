#!/usr/bin/env bash
# Same-state candidate forking pilot: one server, one client, then the analysis.
#
#   usage: run_fork_pilot.sh [port] [gpu_uuid] [label]
#
# The server records full router probabilities for every query it answers, in call
# order, and fork_pilot.py logs which call belongs to which (snapshot, candidate).
# Nothing else may talk to this server while it runs, or the row mapping breaks.
#
# 32 candidates, not the 8 the first pilot used.  At 8 the per-snapshot Spearman has an
# SE of 1/sqrt(7) = 0.38, so 20 snapshots resolve only rho >~ 0.15 -- and that run came
# back null for the *action* baseline too, which in a deterministic simulator is the
# thing that literally causes the outcome.  A null there means the design lacked power,
# not that routing carries nothing, so the candidate count is what had to rise.
set -uo pipefail

PORT=${1:-8330}
GPU=${2:-MIG-63b1c8d1-da8b-5f89-acd2-359db55eb1da}
LABEL=${3:-fork-pilot-n32}

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/objstate/$LABEL
OUT="$ROOT/runs/$LABEL"

if [ -e "$OUT/routes.zarr" ]; then
  echo "refusing to overwrite existing pilot at $OUT"; exit 1
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
    echo "server FAILED"; tail -30 "$LOG/server.log"; exit 1; }
done
[ "$ok" = 1 ] || { echo "server did not come up in 300s"; tail -30 "$LOG/server.log"; exit 1; }
echo "[$(date +%H:%M:%S)] fork-pilot server up on :$PORT"

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u fork_pilot.py \
  --port "$PORT" --client-dir "$ROOT/runs/objstate-t0s24-client" \
  --libero-root "$BR/upstream/LIBERO" \
  --episodes 0,1,2,3,4 --fork-steps 6,10,11,12 --n-candidates 32 \
  --restore-mode hard \
  --out "${OUT}-client" 2>&1 | tee "$LOG/client.log"

ps -eo pid,cmd | grep "[s]erve_with_recorder.py --port $PORT" | awk '{print $1}' \
  | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 15
echo "[$(date +%H:%M:%S)] fork-pilot capture done"
