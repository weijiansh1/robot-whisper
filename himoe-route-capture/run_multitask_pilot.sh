#!/usr/bin/env bash
# Pilot: find a mixed-outcome init state for each LIBERO-Goal task.
#
# The 64-draw design holds the scene fixed and varies only flow noise, so it
# needs an init state whose per-state success rate is away from 0 and 1.  Under
# the correct wrist layout the whole suite sits at 97.8% (hardest task 92%), so
# there is no dynamic range there; these runs use released-left, the same
# operating point as within64-s24.  That is a deliberate mis-fed input and the
# external-validity caveat carries over.
#
# 8 init states x 4 noise draws per task = 32 episodes.  A state splitting 1-3 of
# 4 is a candidate for the full 64-draw run.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/mt-pilot
mkdir -p "$LOG"
INIT_STATES=0,5,9,14,19,24,31,38
REPEATS=4

# task : port : gpu
JOBS=("$1:8300:MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a" "$2:8301:MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3")

for j in "${JOBS[@]}"; do
  IFS=: read -r task port gpu <<< "$j"
  cd "$ROOT"
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
  MOEVLA_DATA_HOME=$BR/moevla-data \
  setsid nohup "$BR/envs/model/bin/python" -u serve_ablated_router.py \
    --port "$port" --gpu "$gpu" --suite goal \
    --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
    --upstream-root "$BR/upstream/HiMoE-VLA" \
    --libero-wrist-layout released-left --mode none \
    > "$LOG/srv-t$task.log" 2>&1 < /dev/null &
done

for j in "${JOBS[@]}"; do
  IFS=: read -r task port gpu <<< "$j"
  ok=0
  for _ in $(seq 1 60); do
    sleep 5
    grep -q "serving on ws" "$LOG/srv-t$task.log" 2>/dev/null && { ok=1; break; }
    grep -qi "Traceback" "$LOG/srv-t$task.log" 2>/dev/null && { echo "task $task server FAILED"; tail -20 "$LOG/srv-t$task.log"; exit 1; }
  done
  [ "$ok" = 1 ] || { echo "task $task server never ready"; exit 1; }
  echo "[$(date +%H:%M:%S)] task $task server ready on $port"
done

for j in "${JOBS[@]}"; do
  IFS=: read -r task port gpu <<< "$j"
  OUT="$ROOT/runs/pilot-goal-t$task"; mkdir -p "$OUT"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
  setsid nohup "$BR/envs/libero/bin/python" -u rollout_with_routes.py \
    --port "$port" --task-id "$task" --benchmark libero_goal \
    --init-state-ids "$INIT_STATES" --repeats "$REPEATS" --noise-seed-base 1000 \
    --no-routing-capture --label "pilot-t$task" --out "$OUT" \
    --libero-root "$BR/upstream/LIBERO" \
    > "$LOG/cli-t$task.log" 2>&1 < /dev/null &
  sleep 2
done

while pgrep -f "rollout_with_routes.py --port 830" > /dev/null; do sleep 20; done
for j in "${JOBS[@]}"; do
  IFS=: read -r task port gpu <<< "$j"
  echo "[$(date +%H:%M:%S)] task $task: $(grep -E '^pilot-t' "$LOG/cli-t$task.log" | tail -1)"
done
ps -eo pid,cmd | grep "[s]erve_ablated_router.py --port 830" | awk '{print $1}' | while read p; do kill -TERM "$p" 2>/dev/null; done
sleep 5
echo "[$(date +%H:%M:%S)] pilot done for tasks $1 $2"
