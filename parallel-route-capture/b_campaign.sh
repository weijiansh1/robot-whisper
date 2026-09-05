#!/bin/bash
# Seed-extension campaign: right-50x8b (noise seeds 1008-1015) -> corpus grows
# to 50x16 per task when merged with right-50x8 at analysis time.
#   bash b_campaign.sh start1     # wave 1: 27 tasks on free slots (avoid GPU 4/6)
#   bash b_campaign.sh start2     # wave 2: remaining 13 (run when slots free)
#   bash b_campaign.sh supervise  # self-healing loop (covers launched tasks)
#   bash b_campaign.sh status|stop
set -u
export CORPUS_MUJOCO_GL=osmesa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4
HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
RUN_ID=right-50x8b-20260903
SEED_BASE=1008
HUB="$HERE/hubshim"
LOGS="$HERE/runs/flat50x8b/logs"; PIDS="$HERE/runs/flat50x8b/pids"
CN=/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA
MAX_ATTEMPTS=4
mkdir -p "$LOGS" "$PIDS"

# tag -> "wave gpu port benchmark task"
declare -A SPEC=(
  [b-lng0]="1 3 8850 libero_10 0"      [b-lng1]="1 3 8851 libero_10 1"
  [b-lng2]="1 3 8852 libero_10 2"      [b-lng3]="1 3 8853 libero_10 3"
  [b-lng4]="1 3 8854 libero_10 4"      [b-lng5]="1 5 8855 libero_10 5"
  [b-lng6]="1 5 8856 libero_10 6"      [b-lng7]="1 5 8857 libero_10 7"
  [b-lng8]="1 5 8858 libero_10 8"      [b-lng9]="1 5 8859 libero_10 9"
  [b-gl0]="1 1 8860 libero_goal 0"     [b-gl1]="1 1 8861 libero_goal 1"
  [b-gl2]="1 1 8862 libero_goal 2"     [b-gl3]="1 1 8863 libero_goal 3"
  [b-gl4]="1 1 8864 libero_goal 4"     [b-gl5]="1 2 8865 libero_goal 5"
  [b-gl6]="1 2 8866 libero_goal 6"     [b-gl7]="1 2 8867 libero_goal 7"
  [b-gl8]="1 2 8868 libero_goal 8"     [b-gl9]="1 7 8869 libero_goal 9"
  [b-sp0]="1 7 8870 libero_spatial 0"  [b-sp1]="1 7 8871 libero_spatial 1"
  [b-sp2]="1 7 8872 libero_spatial 2"  [b-sp3]="1 0 8873 libero_spatial 3"
  [b-sp4]="1 0 8874 libero_spatial 4"  [b-sp5]="1 0 8875 libero_spatial 5"
  [b-sp6]="1 0 8876 libero_spatial 6"
  [b-sp7]="2 1 8877 libero_spatial 7"  [b-sp8]="2 1 8878 libero_spatial 8"
  [b-sp9]="2 1 8879 libero_spatial 9"  [b-ob0]="2 2 8880 libero_object 0"
  [b-ob1]="2 2 8881 libero_object 1"   [b-ob2]="2 2 8882 libero_object 2"
  [b-ob3]="2 3 8883 libero_object 3"   [b-ob4]="2 3 8884 libero_object 4"
  [b-ob5]="2 3 8885 libero_object 5"   [b-ob6]="2 5 8886 libero_object 6"
  [b-ob7]="2 5 8887 libero_object 7"   [b-ob8]="2 0 8890 libero_object 8"
  [b-ob9]="2 7 8889 libero_object 9"
)

hub_dir() { [ "$1" = libero_10 ] && echo libero_long || echo "$1"; }
say() { echo "[$(date +%H:%M:%S)] $*"; }
alive() { local p; p=$(cat "$PIDS/$1.pid" 2>/dev/null); [ -n "$p" ] && [ -d "/proc/$p" ]; }

launch_one() { # tag [force]
    local tag=$1
    local mode=${2:-}
    read -r wave gpu port bench task <<< "${SPEC[$tag]}"
    alive "$tag" && { say "$tag already running — skip"; return; }
    local extra=()
    [ "$mode" = force ] && extra=(--force)
    nohup setsid python3 -u "$RC/run_corpus_capture.py" \
        --hub-root "$HUB" --run-id "$RUN_ID" \
        --gpu "$gpu" --port "$port" \
        --benchmarks "$bench" --tasks "$task" \
        --scenes 50 --draws 8 --noise-seed-base $SEED_BASE \
        "${extra[@]}" \
        > "$LOGS/$tag.log" 2>&1 &
    echo $! > "$PIDS/$tag.pid"
    say "$tag gpu$gpu :$port $bench task $task (pid $!)"
    sleep 15
}

start_wave() { # wave-number
    for tag in $(echo "${!SPEC[@]}" | tr ' ' '\n' | sort); do
        read -r wave gpu port bench task <<< "${SPEC[$tag]}"
        [ "$wave" = "$1" ] && launch_one "$tag"
    done
    say "wave $1 launched"
}

supervise() {
    say "b-supervisor up (max $MAX_ATTEMPTS attempts/task, seed base $SEED_BASE)"
    while :; do
        all_done=1
        for tag in "${!SPEC[@]}"; do
            [ -f "$PIDS/$tag.pid" ] || continue          # not launched yet: ignore
            pid=$(cat "$PIDS/$tag.pid")
            if [ -d "/proc/$pid" ]; then
                all_done=0
            elif grep -q "0 failure" "$LOGS/$tag.log" 2>/dev/null; then
                :
            elif grep -qE "failure|already exists" "$LOGS/$tag.log" 2>/dev/null; then
                all_done=0
                n=$(ls "$LOGS/$tag".attempt*.log 2>/dev/null | wc -l)
                if [ "$n" -ge $((MAX_ATTEMPTS - 1)) ]; then
                    say "$tag exhausted — manual review"
                    continue
                fi
                read -r wave gpu port bench task <<< "${SPEC[$tag]}"
                say "repair $tag (attempt $((n+2)))"
                opid=$(pgrep -f "serve_with_recorder.py.*--port $port " | head -1)
                [ -n "${opid:-}" ] && { kill -TERM "$opid"; sleep 5; }
                tname=$(python3 -c "
import json
p = json.load(open('$HERE/plan.json'))
print(p['task_names']['$bench'][$task])")
                [ -n "$tname" ] && rm -rf "$CN/$(hub_dir "$bench")/$tname/$RUN_ID"
                mv "$LOGS/$tag.log" "$LOGS/$tag.attempt$((n+1)).log"
                launch_one "$tag"
                sleep 15
            else
                all_done=0
            fi
        done
        # auto-launch at most ONE pending wave-2 task per cycle, gated on a
        # pid-based slot count (free-memory lags server allocation by ~2 min
        # and caused a launch storm on 2026-09-03 evening).
        for tag in $(echo "${!SPEC[@]}" | tr ' ' '\n' | sort); do
            [ -f "$PIDS/$tag.pid" ] && continue
            read -r wave gpu port bench task <<< "${SPEC[$tag]}"
            [ "$wave" = 2 ] || continue
            slots=0
            for t2 in "${!SPEC[@]}"; do
                read -r w2 g2 _ _ _ <<< "${SPEC[$t2]}"
                [ "$g2" = "$gpu" ] && alive "$t2" && slots=$((slots+1))
            done
            for f in "$HERE"/runs/flat50x8/pids/v${gpu}*.pid; do
                [ -f "$f" ] && [ -d "/proc/$(cat "$f")" ] && slots=$((slots+1))
            done
            if [ "$slots" -lt 5 ]; then
                say "gpu$gpu has $slots/5 slots used -> launching wave-2 $tag"
                launch_one "$tag"
                break
            fi
        done
        # done only when every task in SPEC has completed cleanly
        launched=$(ls "$PIDS"/b-*.pid 2>/dev/null | wc -l)
        [ "$all_done" -eq 1 ] && [ "$launched" -eq ${#SPEC[@]} ] && { say "B CAMPAIGN COMPLETE"; exit 0; }
        sleep 180
    done
}

case "${1:-status}" in
start1) start_wave 1 ;;
start2) start_wave 2 ;;
    supervise) supervise ;;
    rescue-ob8)
        # Port 8888 belongs to JupyterHub.  Preserve the fourth failed log and
        # overwrite only the zero-episode failed run on the corrected port.
        [ -f "$LOGS/b-ob8.log" ] && [ ! -f "$LOGS/b-ob8.attempt4-port8888.log" ] && \
            mv "$LOGS/b-ob8.log" "$LOGS/b-ob8.attempt4-port8888.log"
        launch_one b-ob8 force
        ;;
status)
    run=0; done_n=0; dead=0
    for tag in $(echo "${!SPEC[@]}" | tr ' ' '\n' | sort); do
        [ -f "$PIDS/$tag.pid" ] || continue
        if alive "$tag"; then run=$((run+1));
        elif grep -q "0 failure" "$LOGS/$tag.log" 2>/dev/null; then done_n=$((done_n+1));
        else dead=$((dead+1)); echo "  dead: $tag"; fi
    done
    echo "b-campaign: $run running, $done_n complete, $dead dead, $(( ${#SPEC[@]} - $(ls "$PIDS"/b-*.pid 2>/dev/null | wc -l) )) not yet launched"
    python3 - "$RUN_ID" <<'EOF'
import glob, json, sys
n = t = 0
for p in glob.glob("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_*/*/%s/client/summaries.json" % sys.argv[1]):
    d = json.load(open(p)); n += len(d); t += 1
print(f"episodes: {n}/16000 across {t}/40 tasks")
EOF
    ;;
stop)
    for f in "$PIDS"/b-*.pid; do [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null; done
    say "b-campaign stop signals sent (servers flush via orchestrator teardown)"
    ;;
esac
