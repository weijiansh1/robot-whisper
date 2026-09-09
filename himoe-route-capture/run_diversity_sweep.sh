#!/usr/bin/env bash
# Candidate-diversity screen across all 10 LIBERO-Goal tasks.
# Zero rollouts: each state is probed by branching 16 candidates for one chunk.
set -uo pipefail
ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
LOG=/tmp/div-sweep; mkdir -p "$LOG"
PHASES=0,4,8,12
run_task() {
  local task=$1 port=$2
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
  timeout 900 "$BR/envs/libero/bin/python" -u "$ROOT/screen_candidate_diversity.py" \
    --port "$port" --task-id "$task" --init-state-id 0 --phases $PHASES \
    --n-candidates 16 --libero-root "$BR/upstream/LIBERO" \
    --out "$ROOT/runs/diversity-t${task}s0" > "$LOG/t$task.log" 2>&1
  echo "[$(date +%H:%M:%S)] task $task 完成"
}
for t in 0 1 2 3 4; do run_task $t 8400; done &
for t in 5 6 7 8 9; do run_task $t 8401; done &
wait
echo "[$(date +%H:%M:%S)] 全部完成"
