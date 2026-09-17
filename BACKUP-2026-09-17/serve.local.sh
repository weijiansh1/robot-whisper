#!/bin/bash
# 远程 /data/serve.sh 的本机版：路径 /data -> $DATA，其余参数保持一致
DATA=${DATA:-/home/swj/data}
cd $DATA/srv
export PYTHONPATH=$DATA/srv/src:$DATA/srv/packages/openpi-client/src:$DATA/srv
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${HIMOE_SERVER_THREADS:-4} MKL_NUM_THREADS=${HIMOE_SERVER_THREADS:-4}
export HIMOE_UPSTREAM_COMMIT=27a2c46932d8b6373ca0074eb997f299bcd4f6f5
git config --global --add safe.directory $DATA/srv 2>/dev/null
git config --global --add safe.directory '*' 2>/dev/null
OUT=${OUT:-$DATA/run0/server}
mkdir -p $OUT
exec $DATA/venv311/bin/python serve_with_recorder.py \
  --host 0.0.0.0 --port ${PORT:-9500} --gpu ${GPU:-${CUDA_VISIBLE_DEVICES:-0}} \
  --suite long \
  --checkpoint-dir $DATA/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10 \
  --upstream-root $DATA/srv \
  --libero-wrist-layout released-left \
  --out $OUT \
  --store-full-probs
