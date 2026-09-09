#!/usr/bin/env bash
# Routing Lead Test capture: K sibling candidates per query state, with the flow
# trajectory and the full 32-dim router softmax at every Euler step.
#
# 16 episodes x ~12 control steps x 16 candidates ~= 3000 inferences, ~35 min.
#
# Slice: 2g.35gb.  This experiment is self-contained -- it never compares against
# the 4g.71gb patch-probe / routing-cloud batches -- so the usual "same slice as
# the thing you are interpreting" constraint does not bind here.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-vla/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/flow-lead
GPU=${LEAD_GPU:-MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a}   # 2g.35gb
PORT=${LEAD_PORT:-8460}
OUT="$ROOT/runs/flow-lead-t0s24"
PIDFILE="$LOG/server.pid"

mkdir -p "$LOG" "$OUT"
cd "$ROOT"

PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
MOEVLA_DATA_HOME=$BR/moevla-data \
setsid nohup "$BR/envs/model/bin/python" -u serve_flow_trace.py \
  --port "$PORT" --gpu "$GPU" --suite goal \
  --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
  --upstream-root "$BR/upstream/HiMoE-VLA" \
  --libero-wrist-layout checkpoint-right \
  --out "$OUT" > "$LOG/server.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

ok=0
for _ in $(seq 1 60); do
  sleep 5
  grep -q "serving on ws" "$LOG/server.log" 2>/dev/null && { ok=1; break; }
  grep -qi "Traceback" "$LOG/server.log" 2>/dev/null && {
    echo "server FAILED"; tail -40 "$LOG/server.log"; exit 1; }
done
[ "$ok" = 1 ] || { echo "server did not come up in 300s"; tail -40 "$LOG/server.log"; exit 1; }
echo "[$(date +%H:%M:%S)] server up on :$PORT (pid $(cat $PIDFILE))"

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
"$BR/envs/libero/bin/python" -u rollout_flow_lead.py \
  --port "$PORT" --task-id 0 --benchmark libero_goal --init-state-id 24 \
  --n-episodes "${LEAD_EPISODES:-16}" --n-candidates "${LEAD_K:-16}" \
  --noise-seed-base 2000 --label flow-lead-t0s24 --out "$OUT" \
  --libero-root "$BR/upstream/LIBERO" "$@" 2>&1 | tee "$LOG/client.log"

# PID file, never pkill -f: a pattern on the script name matches this script too
kill -TERM "$(cat "$PIDFILE")" 2>/dev/null
sleep 12
echo "[$(date +%H:%M:%S)] capture done -> $OUT"
