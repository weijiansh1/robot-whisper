#!/usr/bin/env bash
set -u
B=/home/jovyan/.cache/himoe-libero-bridge
LIBERO=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$LIBERO:/home/jovyan/work/himoe-vla/himoe-libero-bridge/src:$RC"
exec "$B/envs/libero/bin/python" "$RC/routing_rescue_collect.py" \
  --host 127.0.0.1 --port 9500 --benchmark libero_10 --task-id 8 \
  --init-state-id 7 --worker-id 0 --k 8 --seed 20260830 \
  --libero-root "$LIBERO" \
  --out /home/jovyan/work/himoe-vla/analysis_triggered_fork/rescue_i7
