#!/usr/bin/env bash
# Launch the simulation client against the remote policy server.
#
# The client runs here because this box has the LIBERO stack (the remote A100 only serves
# the model).  The venv's own editable libero install is broken, hence the explicit
# PYTHONPATH; the bundled system-libs supply libEGL for offscreen rendering.
set -euo pipefail

B=/home/jovyan/.cache/himoe-libero-bridge
LIBERO=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO
RC=/home/jovyan/work/himoe-vla/himoe-route-capture

export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$LIBERO:/home/jovyan/work/himoe-vla/himoe-libero-bridge/src:$RC"

exec "$B/envs/libero/bin/python" "$RC/triggered_fork_collect.py" \
  --host "${HOST:?set HOST}" --port "${PORT:-9500}" \
  --benchmark libero_10 --task-id 8 \
  --init-state-id "${INIT:-0}" --worker-id 0 --k "${K:-8}" \
  --libero-root "$LIBERO" \
  --out "/home/jovyan/work/himoe-vla/analysis_triggered_fork/run${INIT:-0}"
