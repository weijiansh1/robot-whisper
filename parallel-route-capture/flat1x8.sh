#!/bin/bash
# Flat corpus: every LIBERO task x 1 init state (scene 0) x 8 flow-noise draws,
# via the proven run_corpus_capture pipeline (same geometry family as right-16x32).
# One instance per GPU, 5 tasks each, sequential within instance (one model in
# RAM/GPU at a time -> no cold-start stampede). Writes hub layout through the
# cache_new shim; resumable (completed tasks are skipped on rerun).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
RUN_ID=right-1x8-20260903
HUB="$HERE/hubshim"
LOGS="$HERE/runs/flat1x8/logs"; PIDS="$HERE/runs/flat1x8/pids"
mkdir -p "$LOGS" "$PIDS"

launch() { # gpu port benchmark tasks
    local gpu=$1 port=$2 bench=$3 tasks=$4
    nohup python3 -u "$RC/run_corpus_capture.py" \
        --hub-root "$HUB" --run-id "$RUN_ID" \
        --gpu "$gpu" --port "$port" \
        --benchmarks "$bench" --tasks "$tasks" \
        --scene-ids 0 --draws 8 \
        > "$LOGS/gpu$gpu-$bench.log" 2>&1 &
    echo $! > "$PIDS/gpu$gpu.pid"
    echo "[$(date +%H:%M:%S)] gpu$gpu: $bench tasks $tasks (pid $!)"
}

case "${1:-start}" in
start)
    launch 0 8700 libero_10      0,1,2,3,4
    launch 1 8701 libero_10      5,6,7,8,9
    launch 2 8702 libero_goal    0,1,2,3,4
    launch 3 8703 libero_goal    5,6,7,8,9
    launch 4 8704 libero_spatial 0,1,2,3,4
    launch 5 8705 libero_spatial 5,6,7,8,9
    launch 6 8706 libero_object  0,1,2,3,4
    launch 7 8707 libero_object  5,6,7,8,9
    echo "8 instances up; status: bash flat1x8.sh status"
    ;;
status)
    for f in "$PIDS"/gpu*.pid; do
        pid=$(cat "$f"); g=$(basename "$f" .pid)
        state=$([ -d "/proc/$pid" ] && echo RUN || echo DONE)
        log=$(ls "$LOGS"/$g-*.log 2>/dev/null | head -1)
        last=$(tail -1 "$log" 2>/dev/null | cut -c1-110)
        echo "$g [$state] $last"
    done
    python3 - "$RUN_ID" <<'EOF'
import glob, json, sys
n = t = 0
for p in glob.glob("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_*/*/%s/client/summaries.json" % sys.argv[1]):
    s = json.load(open(p)); n += len(s); t += 1
print(f"episodes captured: {n}/320 across {t}/40 tasks")
EOF
    ;;
stop)
    for f in "$PIDS"/gpu*.pid; do kill -TERM "$(cat "$f")" 2>/dev/null; done
    sleep 3; pkill -TERM -f '[s]erve_with_recorder.py'
    echo "stopped"
    ;;
esac
