#!/bin/bash
# usage: shard.sh <shard 0-3> <port>
# Equal-budget arms: every branch gets FORK_BUDGET queries from its own fork point.
# Trunks are distinguished by SEED (worker_id does NOT enter trunk_noise/branch_noise).
BR=/home/jovyan/.cache/himoe-libero-bridge
ROOT=/home/jovyan/work/himoe-vla/trap-recovery-depth-20260904
SHARD=$1; PORT=$2; i=0
FORK_BUDGET=16
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/home/jovyan/work/himoe-vla/himoe-route-capture:/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src
# 初态顺序：中等失败率(25-75%)优先 —— 失败率 100% 的初态注定失败、任何时点都救不回
# (i2/i12 实测三臂全 0/4)，0% 的又全被 --failures-only 跳过。集合不变，跑完仍覆盖 50 个。
INITS="10 11 15 27 47 0 1 8 16 40 13 19 26 30 42 2 12 38 41 3 21 39 48 49 4 23 29 35 37 44 5 6 7 9 14 17 18 20 22 24 25 28 31 32 33 34 36 43 45 46"
for init in $INITS; do
  for w in 0 1 2 3; do
    if [ $((i % 4)) -eq "$SHARD" ]; then
      OUT=$ROOT/runs/i${init}_s${w}
      if [ ! -f "$OUT/manifest.json" ]; then
        rm -rf "$OUT"
        echo "=== i${init}_s${w} $(date +%H:%M:%S)"
        timeout 2400 $BR/envs/libero/bin/python -u $ROOT/code/delayed_fork_collect.py \
          --host 127.0.0.1 --port $PORT --benchmark libero_10 --task-id 8 \
          --init-state-id $init --worker-id $w --max-steps 1200 --max-trunk-queries 80 \
          --seed $((20260904 + w * 1000)) \
          --k 4 --delay-offsets 4,8 --fork-budget-queries $FORK_BUDGET \
          --failures-only --control-arm \
          --libero-root $BR/upstream/LIBERO --out "$OUT" 2>&1 | tail -16
      fi
    fi
    i=$((i+1))
  done
done
echo "SHARD $SHARD DONE $(date +%H:%M:%S)"
