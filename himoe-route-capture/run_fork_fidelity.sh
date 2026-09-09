#!/usr/bin/env bash
# Fork fidelity probe: is a same-state branch trustworthy at the contact phases?
#
#   usage: run_fork_fidelity.sh [port] [gpu_uuid]
#
# Gates the N=32 fork pilot.  No routing is recorded here -- the probe only needs the
# action chunks and the simulator, so the plain recorder server is fine and its zarr
# output is ignored.
set -uo pipefail

PORT=${1:-8340}
GPU=${2:-MIG-63b1c8d1-da8b-5f89-acd2-359db55eb1da}

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/fork-fidelity
OUT="$ROOT/runs/fork-fidelity"

mkdir -p "$LOG" "$OUT"
cd "$ROOT"

PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
MOEVLA_DATA_HOME=$BR/moevla-data \
setsid nohup "$BR/envs/model/bin/python" -u serve_with_recorder.py \
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
echo "[$(date +%H:%M:%S)] fidelity server up on :$PORT"

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u probe_fork_fidelity.py \
  --port "$PORT" --libero-root "$BR/upstream/LIBERO" \
  --task-id 0 --init-state-id 24 --phases 6,12 \
  --n-candidates 16 --repeat-every 5 --seed-base 5000 \
  --out "$ROOT/analysis/fork-fidelity/fidelity_replay.json" 2>&1 | tee "$LOG/client.log"

# SIGTERM, never a plain exit: the Zarr writer flushes in the signal handler.
ps -eo pid,cmd | grep "[s]erve_with_recorder.py --port $PORT" | awk '{print $1}' \
  | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 15
echo "[$(date +%H:%M:%S)] fidelity probe done"
