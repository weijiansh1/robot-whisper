#!/bin/bash
# FULL GRID corpus (filename kept for the chainer; geometry is 50x8):
# every LIBERO task x ALL 50 official init states x 8 flow-noise draws
# = 16,000 episodes with full route capture.
#
# libero_10 episodes cost ~3x the other suites in inference, so the 10 long
# tasks are spread across all 8 GPUs; 22 single-benchmark instances total,
# 2-3 per GPU (~45-68 GiB VRAM each GPU; CALVIN shares GPU 7).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
RUN_ID=right-50x8-20260903
HUB="$HERE/hubshim"
LOGS="$HERE/runs/flat50x8/logs"; PIDS="$HERE/runs/flat50x8/pids"
mkdir -p "$LOGS" "$PIDS"

launch() { # tag gpu port benchmark tasks
    local tag=$1 gpu=$2 port=$3 bench=$4 tasks=$5
    nohup python3 -u "$RC/run_corpus_capture.py" \
        --hub-root "$HUB" --run-id "$RUN_ID" \
        --gpu "$gpu" --port "$port" \
        --benchmarks "$bench" --tasks "$tasks" \
        --scenes 50 --draws 8 \
        > "$LOGS/$tag.log" 2>&1 &
    echo $! > "$PIDS/$tag.pid"
    echo "[$(date +%H:%M:%S)] $tag gpu$gpu: $bench tasks $tasks (pid $!)"
}

case "${1:-start}" in
start)
    launch g0-long1 0 8710 libero_10      0
    launch g0-long2 0 8711 libero_10      8
    launch g0-obj   0 8712 libero_object  0,1,2
    launch g1-long1 1 8713 libero_10      1
    launch g1-long2 1 8714 libero_10      9
    launch g1-obj   1 8715 libero_object  3,4,5
    launch g2-long  2 8716 libero_10      2
    launch g2-goal  2 8717 libero_goal    0,1,2
    launch g2-obj   2 8718 libero_object  6,7
    launch g3-long  3 8719 libero_10      3
    launch g3-goal  3 8720 libero_goal    3,4,5
    launch g3-obj   3 8721 libero_object  8,9
    launch g4-long  4 8722 libero_10      4
    launch g4-goal  4 8723 libero_goal    6,7,8
    launch g4-spa   4 8724 libero_spatial 0,1
    launch g5-long  5 8725 libero_10      5
    launch g5-goal  5 8726 libero_goal    9
    launch g5-spa   5 8727 libero_spatial 2,3,4
    launch g6-long  6 8728 libero_10      6
    launch g6-spa   6 8729 libero_spatial 5,6,7
    launch g7-long  7 8730 libero_10      7
    launch g7-spa   7 8731 libero_spatial 8,9
    # disk guard: TERM everything if work fs drops under 3 GB free
    nohup bash -c 'while :; do
        f=$(df -BG --output=avail '"$HERE"' | tail -1 | tr -dc 0-9)
        [ "$f" -lt 3 ] && { echo "DISK LOW ${f}G"; bash '"$HERE"'/flat50x1.sh stop; exit 1; }
        sleep 120; done' > "$LOGS/diskguard.log" 2>&1 &
    echo $! > "$PIDS/diskguard.pid"
    echo "22 instances + disk guard up; status: bash flat50x1.sh status"
    ;;
status)
    for f in "$PIDS"/g*.pid; do
        pid=$(cat "$f"); g=$(basename "$f" .pid)
        state=$([ -d "/proc/$pid" ] && echo RUN || echo DONE)
        last=$(tail -1 "$LOGS/$g.log" 2>/dev/null | cut -c1-100)
        echo "$g [$state] $last"
    done
    python3 - "$RUN_ID" <<'EOF'
import glob, json, sys
n = t = 0
for p in glob.glob("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_*/*/%s/client/summaries.json" % sys.argv[1]):
    s = json.load(open(p)); n += len(s); t += 1
print(f"episodes captured: {n}/16000 across {t}/40 tasks")
EOF
    df -h /home/jovyan/work | tail -1
    ;;
stop)
    for f in "$PIDS"/g*.pid "$PIDS"/diskguard.pid; do
        [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null
    done
    sleep 3; pkill -TERM -f '[s]erve_with_recorder.py'
    echo "stopped"
    ;;
esac
