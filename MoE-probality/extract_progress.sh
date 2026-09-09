#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd "$SCRIPT_DIR/.." && pwd)
BRIDGE_CACHE="$WORKSPACE/himoe-vla-cache/himoe-libero-bridge/cache"
LIBERO_ROOT="$BRIDGE_CACHE/upstream/LIBERO"

export CUDA_VISIBLE_DEVICES=
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=/tmp/himoe-probability-progress-libero-config
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$BRIDGE_CACHE/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WORKSPACE/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$LIBERO_ROOT:${PYTHONPATH:-}"

exec "$BRIDGE_CACHE/envs/libero/bin/python" -u "$SCRIPT_DIR/probability/extract_progress.py" \
  --hub "$WORKSPACE/VLA_MUI_HUB" --libero-root "$LIBERO_ROOT" "$@"
