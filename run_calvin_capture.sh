#!/bin/bash
# CALVIN-D capture: the same route recorder as LIBERO, on a different simulator,
# a different action space (calvin_d_joint -> the other AS signature) and a
# 15-dim robot_obs that carries the end-effector pose AND the seven joint angles.
# That last part is the point: it can separate "the routing encodes the arm" from
# "the routing encodes the end effector", which LIBERO's 8-dim state cannot.
set -u
C=/home/jovyan/.cache/himoe-calvin-alignment
H=/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA
M=/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python
SRC=/home/jovyan/work/himoe-vla/himoe-calvin-alignment/src
WF=/home/jovyan/work/himoe-libero-wrist-fix/src
OUT=/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/calvin_d/task_D_D/routes-v1
LOG=/home/jovyan/work/himoe-vla/logs-calvin
mkdir -p "$LOG" "$OUT"

CUDA_VISIBLE_DEVICES=MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a \
PYTHONPATH="$SRC:$WF:$H/packages/openpi-client/src" \
$M -u /home/jovyan/work/himoe-vla/himoe-route-capture/serve_calvin_with_recorder.py \
  --checkpoint-dir $C/checkpoints/HiMoE-VLA-CALVIN-D \
  --upstream-root $H --calvin-root $C/upstream/calvin \
  --host 127.0.0.1 --port 8501 --out "$OUT/server" \
  > "$LOG/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV"
for i in $(seq 60); do grep -q "serving on ws" "$LOG/server.log" && break; sleep 5; done
grep -q "serving on ws" "$LOG/server.log" || { echo "server failed"; tail -5 "$LOG/server.log"; exit 1; }
echo "[$(date +%H:%M:%S)] server ready, starting client"

cd /home/jovyan/work/himoe-vla/himoe-calvin-alignment
bash scripts/calvin.sh \
  --dataset-root $C/datasets/task_D_D --calvin-root $C/upstream/calvin \
  --himoe-root $H --output-dir "$OUT/client" \
  --host 127.0.0.1 --port 8501 --max-new-sequences 80 \
  > "$LOG/client.log" 2>&1
echo "[$(date +%H:%M:%S)] client done, flushing server"
kill -TERM $SRV; wait $SRV 2>/dev/null
echo "[$(date +%H:%M:%S)] all done"
