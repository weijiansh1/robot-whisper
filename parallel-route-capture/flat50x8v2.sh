#!/bin/bash
# 50x8 grid, v2 packing: ONE task per instance, 5 instances per GPU, so every
# card keeps ~5 inference streams queued (GPU 7 runs 4 + CALVIN; its 5th task
# starts automatically when the CALVIN client exits).
#
# Clients render with OSMesa: 39 concurrent EGL clients aborted sporadically
# (~1.5%/episode, libc++abi) under GL driver contention; OSMesa measured the
# same speed here (29-32 s/episode) and is crash-immune. Launches are staggered
# so servers never cold-start all at once (600 s readiness timeout).
set -u
export CORPUS_MUJOCO_GL=osmesa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4
HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
RUN_ID=right-50x8-20260903
HUB="$HERE/hubshim"
LOGS="$HERE/runs/flat50x8/logs"; PIDS="$HERE/runs/flat50x8/pids"
mkdir -p "$LOGS" "$PIDS"

launch() { # tag gpu port benchmark task
    local tag=$1 gpu=$2 port=$3 bench=$4 task=$5
    nohup python3 -u "$RC/run_corpus_capture.py" \
        --hub-root "$HUB" --run-id "$RUN_ID" \
        --gpu "$gpu" --port "$port" \
        --benchmarks "$bench" --tasks "$task" \
        --scenes 50 --draws 8 \
        > "$LOGS/$tag.log" 2>&1 &
    echo $! > "$PIDS/$tag.pid"
    echo "[$(date +%H:%M:%S)] $tag gpu$gpu: $bench task $task (pid $!)"
    sleep 15
}

case "${1:-start}" in
start)
    launch v0a 0 8740 libero_10      0
    launch v0b 0 8741 libero_10      8
    launch v0c 0 8742 libero_object  0
    launch v0d 0 8743 libero_object  1
    launch v0e 0 8744 libero_object  2
    launch v1a 1 8745 libero_10      1
    launch v1b 1 8746 libero_10      9
    launch v1c 1 8747 libero_object  3
    launch v1d 1 8748 libero_object  4
    launch v1e 1 8749 libero_object  5
    launch v2a 2 8750 libero_10      2
    launch v2b 2 8751 libero_goal    0
    launch v2c 2 8752 libero_goal    1
    launch v2d 2 8753 libero_goal    2
    launch v2e 2 8754 libero_object  6
    launch v3a 3 8755 libero_10      3
    launch v3b 3 8756 libero_goal    3
    launch v3c 3 8757 libero_goal    4
    launch v3d 3 8758 libero_goal    5
    launch v3e 3 8759 libero_object  7
    launch v4a 4 8760 libero_10      4
    launch v4b 4 8761 libero_goal    6
    launch v4c 4 8762 libero_goal    7
    launch v4d 4 8763 libero_spatial 0
    launch v4e 4 8764 libero_spatial 1
    launch v5a 5 8765 libero_10      5
    launch v5b 5 8766 libero_goal    8
    launch v5c 5 8767 libero_goal    9
    launch v5d 5 8768 libero_spatial 2
    launch v5e 5 8769 libero_spatial 3
    launch v6a 6 8770 libero_10      6
    launch v6b 6 8771 libero_spatial 4
    launch v6c 6 8772 libero_spatial 5
    launch v6d 6 8773 libero_spatial 6
    launch v6e 6 8774 libero_object  8
    launch v7a 7 8775 libero_10      7
    launch v7b 7 8776 libero_spatial 7
    launch v7c 7 8777 libero_spatial 8
    launch v7d 7 8778 libero_object  9
    # deferred 40th task: spatial 9 starts on GPU 7 once the CALVIN client exits
    nohup bash -c '
        while pgrep -f "serve_calvin_with_recorder" > /dev/null; do sleep 60; done
        echo "[deferred] calvin gone; starting spatial task 9 on gpu 7"
        cd '"$HERE"'
        nohup python3 -u '"$RC"'/run_corpus_capture.py \
            --hub-root '"$HUB"' --run-id '"$RUN_ID"' \
            --gpu 7 --port 8779 --benchmarks libero_spatial --tasks 9 \
            --scenes 50 --draws 8 > '"$LOGS"'/v7e.log 2>&1 &
        echo $! > '"$PIDS"'/v7e.pid
    ' > "$LOGS/deferred.log" 2>&1 &
    echo $! > "$PIDS/deferred.pid"
    # disk guard
    nohup bash -c 'while :; do
        f=$(df -BG --output=avail '"$HERE"' | tail -1 | tr -dc 0-9)
        [ "$f" -lt 3 ] && { echo "DISK LOW ${f}G"; bash '"$HERE"'/flat50x8v2.sh stop; exit 1; }
        sleep 120; done' > "$LOGS/diskguard.log" 2>&1 &
    echo $! > "$PIDS/diskguard.pid"
    echo "39+1(deferred) instances + disk guard up"
    ;;
status)
    done_n=0; run_n=0
    for f in "$PIDS"/v*.pid; do
        pid=$(cat "$f"); g=$(basename "$f" .pid)
        if [ -d "/proc/$pid" ]; then run_n=$((run_n+1)); else done_n=$((done_n+1)); fi
    done
    echo "instances: $run_n running, $done_n exited"
    grep -h "\[ok\]\|\[fail\]" "$LOGS"/v*.log 2>/dev/null | tail -5
    python3 - "$RUN_ID" <<'EOF'
import glob, json, sys
n = t = s = 0
for p in glob.glob("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_*/*/%s/client/summaries.json" % sys.argv[1]):
    d = json.load(open(p)); n += len(d); s += sum(1 for e in d if e.get("success")); t += 1
print(f"episodes: {n}/16000 ({s} success) across {t}/40 finished tasks")
EOF
    df -h /home/jovyan/work | tail -1
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
    ;;
stop)
    for f in "$PIDS"/v*.pid "$PIDS"/deferred.pid "$PIDS"/diskguard.pid; do
        [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null
    done
    sleep 3; pkill -TERM -f '[s]erve_with_recorder.py'
    echo "stopped"
    ;;
esac
