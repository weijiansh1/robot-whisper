#!/usr/bin/env bash
# Pin the CPU-only LIBERO render environment, then exec the given command.
#
# osmesa rather than egl: every GPU on this host is fully occupied by processes
# outside this container, and the counterfactual builder needs no GPU at all.
# run_corpus_capture.py already documents osmesa as the backend that avoids
# sporadic EGL client aborts.
set -euo pipefail

BR=/home/jovyan/.cache/himoe-libero-bridge
ROOT=/home/jovyan/work/himoe-vla

export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
# LIBERO rewrites config.yaml whenever it builds an environment; keep this run's
# copy private so a concurrent client cannot observe it mid-write.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/moe-coupling-circuit-0906/results/.libero_config}"
export PYTHONPATH="/home/jovyan/work/himoe-libero-wrist-fix/src:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src"

mkdir -p "$LIBERO_CONFIG_PATH"
cd "$ROOT"
exec "$BR/envs/libero/bin/python" "$@"
