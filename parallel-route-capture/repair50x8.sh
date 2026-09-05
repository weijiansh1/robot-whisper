#!/bin/bash
# Relaunch 50x8 instances that died on the 600 s server-readiness timeout during
# the 39-way cold-start stampede: kill the orphaned server, remove the stub run
# dir (the run_id clash guard refuses to reuse it), relaunch staggered.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
RUN_ID=right-50x8-20260903
HUB="$HERE/hubshim"
LOGS="$HERE/runs/flat50x8/logs"; PIDS="$HERE/runs/flat50x8/pids"
CN=/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA

declare -A SPEC=(
  [v0a]="0 8740 libero_10 0"      [v0b]="0 8741 libero_10 8"
  [v0c]="0 8742 libero_object 0"  [v0d]="0 8743 libero_object 1"  [v0e]="0 8744 libero_object 2"
  [v1a]="1 8745 libero_10 1"      [v1b]="1 8746 libero_10 9"
  [v1c]="1 8747 libero_object 3"  [v1d]="1 8748 libero_object 4"  [v1e]="1 8749 libero_object 5"
  [v2a]="2 8750 libero_10 2"      [v2b]="2 8751 libero_goal 0"    [v2c]="2 8752 libero_goal 1"
  [v2d]="2 8753 libero_goal 2"    [v2e]="2 8754 libero_object 6"
  [v3a]="3 8755 libero_10 3"      [v3b]="3 8756 libero_goal 3"    [v3c]="3 8757 libero_goal 4"
  [v3d]="3 8758 libero_goal 5"    [v3e]="3 8759 libero_object 7"
  [v4a]="4 8760 libero_10 4"      [v4b]="4 8761 libero_goal 6"    [v4c]="4 8762 libero_goal 7"
  [v4d]="4 8763 libero_spatial 0" [v4e]="4 8764 libero_spatial 1"
  [v5a]="5 8765 libero_10 5"      [v5b]="5 8766 libero_goal 8"    [v5c]="5 8767 libero_goal 9"
  [v5d]="5 8768 libero_spatial 2" [v5e]="5 8769 libero_spatial 3"
  [v6a]="6 8770 libero_10 6"      [v6b]="6 8771 libero_spatial 4" [v6c]="6 8772 libero_spatial 5"
  [v6d]="6 8773 libero_spatial 6" [v6e]="6 8774 libero_object 8"
  [v7a]="7 8775 libero_10 7"      [v7b]="7 8776 libero_spatial 7" [v7c]="7 8777 libero_spatial 8"
  [v7d]="7 8778 libero_object 9"  [v7e]="7 8779 libero_spatial 9"
)

hub_dir() { [ "$1" = libero_10 ] && echo libero_long || echo "$1"; }

repaired=0
for tag in "${!SPEC[@]}"; do
    [ -f "$PIDS/$tag.pid" ] || continue
    pid=$(cat "$PIDS/$tag.pid")
    [ -d "/proc/$pid" ] && continue                       # still running
    grep -q "server not ready" "$LOGS/$tag.log" 2>/dev/null || continue
    read -r gpu port bench task <<< "${SPEC[$tag]}"
    echo "[repair] $tag gpu$gpu $bench task $task (port $port)"
    opid=$(pgrep -f "serve_with_recorder.*--port $port\b" | head -1)
    [ -n "${opid:-}" ] && { echo "  kill orphan server $opid"; kill -TERM "$opid"; sleep 5; }
    # find the task's run dir via the plan line in the old log, then remove stub
    tname=$(grep -oP "t0?${task}__\K\S+" "$LOGS/$tag.log" | head -1)
    stub="$CN/$(hub_dir "$bench")/$tname/$RUN_ID"
    if [ -n "$tname" ] && [ -d "$stub" ] && [ ! -f "$stub/client/summaries.json" ]; then
        echo "  rm stub $stub"; rm -rf "$stub"
    fi
    mv "$LOGS/$tag.log" "$LOGS/$tag.attempt1.log"
    nohup python3 -u "$RC/run_corpus_capture.py" \
        --hub-root "$HUB" --run-id "$RUN_ID" \
        --gpu "$gpu" --port "$port" \
        --benchmarks "$bench" --tasks "$task" \
        --scenes 50 --draws 8 \
        > "$LOGS/$tag.log" 2>&1 &
    echo $! > "$PIDS/$tag.pid"
    echo "  relaunched (pid $!)"
    repaired=$((repaired+1))
    sleep 45   # stagger: no second stampede
done
echo "repaired $repaired instance(s)"
