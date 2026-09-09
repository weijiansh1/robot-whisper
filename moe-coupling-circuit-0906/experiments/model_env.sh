#!/usr/bin/env bash
# Pin the model environment, then exec the given command.
#
# CUDA_VISIBLE_DEVICES is deliberately not pre-restricted: run_corpus_capture.py
# records that pre-restricting it hangs CUDA init, and the policy sets it itself.
set -euo pipefail

BR=/home/jovyan/.cache/himoe-libero-bridge
ROOT=/home/jovyan/work/himoe-vla

export PYTHONPATH="/home/jovyan/work/himoe-libero-wrist-fix/src"
export MOEVLA_DATA_HOME="$BR/moevla-data"

cd "$ROOT"
exec "$BR/envs/model/bin/python" -u "$@"
