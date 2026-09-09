#!/usr/bin/env bash
# Which HB-MoE branch carries the 18-point drop?
#
# Four arms on the SAME MIG slice, paired on the same 50 init states and flow
# noise seeds (init_state_id = flow_noise_seed = 0..49, the protocol the
# published left50 / abl-left-* batches used):
#
#   none        both branches            -> re-baselines this slice
#   shared_off  routed branch only
#   routed_off  shared branch only
#   block_off   neither                  -> positive control, should land near 10%
#
# The baseline is re-run rather than reused: the published numbers come from the
# 1g.35gb slice, and a different SM count changes cuBLAS kernel selection, so
# trajectories are not bit-reproducible across slices and a paired test against
# the old batch would be comparing two hardware configs as well as two ablations.
set -uo pipefail

ROOT=/home/jovyan/work/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
GPUID=MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a   # 2g.35gb; never the 4g.71gb slice
PORT=8121
EPISODES=50
LAYOUT=released-left

cd "$ROOT"
mkdir -p runs/logs

for MODE in none shared_off routed_off block_off; do
  OUT="$ROOT/runs/abl2-$MODE"
  if [ -f "$OUT/summaries.json" ] && [ "$(python3 -c "import json;print(len(json.load(open('$OUT/summaries.json'))))" 2>/dev/null)" = "$EPISODES" ]; then
    echo "[$(date +%H:%M:%S)] $MODE already complete, skipping"
    continue
  fi
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] === $MODE : starting server ==="

  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
  MOEVLA_DATA_HOME=$BR/moevla-data \
  setsid nohup "$BR/envs/model/bin/python" -u serve_ablated_branch.py \
    --port $PORT --gpu "$GPUID" --suite goal \
    --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-Goal" \
    --upstream-root "$BR/upstream/HiMoE-VLA" \
    --libero-wrist-layout "$LAYOUT" --mode "$MODE" \
    > "runs/logs/srv-abl2-$MODE.log" 2>&1 < /dev/null &

  SRV=""
  for i in $(seq 1 60); do
    sleep 5
    SRV=$(pgrep -f "serve_ablated_branch.py --port $PORT" | head -1)
    grep -q "serving on ws" "runs/logs/srv-abl2-$MODE.log" 2>/dev/null && break
    if grep -qi "Traceback" "runs/logs/srv-abl2-$MODE.log" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] $MODE : SERVER FAILED"; tail -20 "runs/logs/srv-abl2-$MODE.log"; exit 1
    fi
  done
  if ! grep -q "serving on ws" "runs/logs/srv-abl2-$MODE.log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] $MODE : server never became ready"; exit 1
  fi
  echo "[$(date +%H:%M:%S)] $MODE : server pid $SRV ready; $(grep 'HB layers' runs/logs/srv-abl2-$MODE.log)"

  CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
  "$BR/envs/libero/bin/python" -u rollout_with_routes.py \
    --port $PORT --task-id 0 --benchmark libero_goal \
    --episodes $EPISODES --no-routing-capture \
    --label "abl2-$MODE" --out "$OUT" --libero-root "$BR/upstream/LIBERO" \
    > "runs/logs/cli-abl2-$MODE.log" 2>&1

  echo "[$(date +%H:%M:%S)] $MODE : $(grep -E '^abl2-' runs/logs/cli-abl2-$MODE.log | tail -1)"

  [ -n "$SRV" ] && kill -TERM "$SRV" 2>/dev/null
  for i in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
  kill -0 "$SRV" 2>/dev/null && kill -9 "$SRV" 2>/dev/null
  sleep 5
done

echo "[$(date +%H:%M:%S)] === all arms done ==="
for MODE in none shared_off routed_off block_off; do
  python3 -c "
import json
s=json.load(open('$ROOT/runs/abl2-$MODE/summaries.json'))
print('%-16s %2d/%d = %.0f%%' % ('$MODE', sum(e['success'] for e in s), len(s), 100*sum(e['success'] for e in s)/len(s)))
" 2>/dev/null
done
