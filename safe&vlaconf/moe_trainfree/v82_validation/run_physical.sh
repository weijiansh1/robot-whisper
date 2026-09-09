#!/usr/bin/env bash
set -euo pipefail

TASK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TASK_ROOT=$(cd "$TASK_DIR/../../.." && pwd)
BRIDGE_CACHE="$TASK_ROOT/himoe-vla-cache/himoe-libero-bridge/cache"
LIBERO_ROOT="$BRIDGE_CACHE/upstream/LIBERO"
export CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=/tmp/himoe-physical-failure-libero-config
export LD_LIBRARY_PATH="$BRIDGE_CACHE/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TASK_ROOT/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$LIBERO_ROOT:${PYTHONPATH:-}"
exec "$BRIDGE_CACHE/envs/libero/bin/python" "$TASK_DIR/physical_controls.py" "$@"
