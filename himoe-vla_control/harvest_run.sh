#!/bin/bash
# Phase-1 trunk harvest launcher: one dedicated route-recorder server (legacy,
# full probs) + one trunk_harvest client.
#   bash harvest_run.sh <gpu> <tag> <cells> <uid_base>
# e.g. bash harvest_run.sh 3 pilot "10:0,10:1,11:0,11:1,15:0,15:1,27:0,27:1,47:0,47:1,1:0,1:1" 300
set -u
GPU=${1:?gpu}
TAG=${2:?tag}
CELLS=${3:?cells init:stream,...}
UID_BASE=${4:?trunk uid base}
# Private port band 93xx: 88xx/89xx are used by other people's servers on this
# box (a collision there silently pointed the client at someone else's model).
PORT=$((9310 + GPU + 10 * ${WORKER:-0}))

HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
B=/home/jovyan/.cache/himoe-libero-bridge
M=$B/envs/model/bin/python
L=$B/envs/libero/bin/python
RUN="$HERE/runs/harvest-$TAG"
mkdir -p "$RUN/logs"

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4

if ss -tln 2>/dev/null | grep -q "127.0.0.1:$PORT "; then
    echo "FATAL: port $PORT already in use — pick another band"; exit 1
fi

nohup $M -u "$RC/serve_with_recorder.py" \
    --host 127.0.0.1 --port $PORT --gpu $GPU --suite long \
    --checkpoint-dir $B/checkpoints/HiMoE-VLA-Libero-10 \
    --upstream-root $B/upstream/HiMoE-VLA \
    --libero-wrist-layout checkpoint-right \
    --store-full-probs \
    --out "$RUN/server" \
    > "$RUN/logs/server.log" 2>&1 &
echo $! > "$RUN/logs/server.pid"
echo "[$(date +%H:%M:%S)] server gpu$GPU :$PORT (pid $!)"

python3 - "$PORT" <<'EOF' || { echo "server never opened port"; exit 1; }
import socket, sys, time
port = int(sys.argv[1]); deadline = time.time() + 900
while time.time() < deadline:
    try:
        socket.create_connection(("127.0.0.1", port), 2).close(); sys.exit(0)
    except OSError:
        time.sleep(5)
sys.exit(1)
EOF
echo "[$(date +%H:%M:%S)] server ready; starting harvest client"

env LD_LIBRARY_PATH=$B/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    $L -u "$HERE/trunk_harvest.py" \
    --port $PORT --benchmark libero_10 --task-id 8 \
    --cells "$CELLS" --trunk-uid-base "$UID_BASE" \
    --out "$RUN/trunks" \
    > "$RUN/logs/client.log" 2>&1
RC_CLIENT=$?
echo "[$(date +%H:%M:%S)] client rc=$RC_CLIENT; flushing server"
kill -TERM "$(cat "$RUN/logs/server.pid")" 2>/dev/null
sleep 15
echo "[$(date +%H:%M:%S)] done: $RUN"
