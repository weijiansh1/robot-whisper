#!/bin/bash
# Held-out validation campaign.  The first two runs use flow-noise seeds
# 1000-1007 and 1008-1015; this run adds 1016-1031 while keeping env seed 7.
#
# Large artifacts live on the roomier /home/jovyan filesystem.  Stable links
# under VLA_MUI_HUB/cache_new preserve the normal hub layout and can later be
# materialized onto the work filesystem without changing analysis paths.
set -u

export CORPUS_MUJOCO_GL=osmesa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4

HERE="$(cd "$(dirname "$0")" && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
PLAN="$HERE/plan.json"
RUN_ID=right-50x16c-20260904
SEED_BASE=1016
DRAWS=16
EPISODES=800
SPILL_HUB=/home/jovyan/cache_new_spill_20260904
CANON=/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new
RUN="$HERE/runs/flat50x16c"
LOGS="$RUN/logs"
PIDS="$RUN/pids"
TASKS="$RUN/tasks.tsv"
RECOVERY="$SPILL_HUB/recovery_backups/$RUN_ID"
SUPERVISOR_PID="$PIDS/supervisor.pid"
B_OB8_PID="$HERE/runs/flat50x8b/pids/b-ob8.pid"
# Tasks keep their original home GPU in tasks.tsv, but newly released cards can
# steal pending work.  GPU 6 remains reserved for the online experiment.
GPUS=(0 1 2 3 4 7)
MAX_PER_GPU=6
MIN_FREE_MIB=23500
POWER_TARGET_W=300
HALT_FREE_GB=8
MAX_ZERO_EPISODE_ATTEMPTS=3

say() { echo "[$(date +%H:%M:%S)] $*"; }

alive_pidfile() {
    local file=$1 pid
    [ -f "$file" ] || return 1
    pid=$(cat "$file" 2>/dev/null)
    [ -n "$pid" ] && [ -d "/proc/$pid" ]
}

prepare() {
    mkdir -p "$LOGS" "$PIDS" "$SPILL_HUB/cache"
    python3 - "$PLAN" "$TASKS" "$SPILL_HUB" "$CANON" "$RUN_ID" <<'PY'
import json
import os
import pathlib
import sys

plan_path, tasks_path, spill_hub, canonical, run_id = sys.argv[1:]
plan = json.loads(pathlib.Path(plan_path).read_text())
tasks_path = pathlib.Path(tasks_path)
spill_hub = pathlib.Path(spill_hub)
canonical = pathlib.Path(canonical)

manifest = spill_hub / "manifest.json"
source_manifest = pathlib.Path(plan_path).parents[1] / "VLA_MUI_HUB" / "manifest.json"
if manifest.is_symlink():
    if manifest.resolve(strict=False) != source_manifest.resolve(strict=False):
        raise SystemExit(f"wrong manifest link: {manifest} -> {os.readlink(manifest)}")
elif manifest.exists():
    raise SystemExit(f"refusing to replace non-link {manifest}")
else:
    manifest.symlink_to(source_manifest)

gpus = [0, 2, 4, 7]
suites = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]
abbr = {"libero_10": "lng", "libero_goal": "gl", "libero_spatial": "sp", "libero_object": "ob"}
rows = []
ordinal = 0
for suite in suites:
    hub_suite = "libero_long" if suite == "libero_10" else suite
    for task_id, task_name in enumerate(plan["task_names"][suite]):
        gpu = gpus[ordinal % len(gpus)]
        tag = f"c-{abbr[suite]}{task_id}"
        port = 8900 + ordinal
        target = spill_hub / "cache" / "HiMoE-VLA" / hub_suite / task_name / run_id
        link = canonical / "HiMoE-VLA" / hub_suite / task_name / run_id
        target.parent.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve(strict=False) != target.resolve(strict=False):
                raise SystemExit(f"wrong run link: {link} -> {os.readlink(link)}")
        elif link.exists():
            raise SystemExit(f"refusing to replace existing run path {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
        rows.append((tag, gpu, port, suite, task_id, hub_suite, task_name, str(target)))
        ordinal += 1

text = "\n".join("\t".join(map(str, row)) for row in rows) + "\n"
if tasks_path.exists() and tasks_path.read_text() != text:
    raise SystemExit(f"task manifest changed unexpectedly: {tasks_path}")
tasks_path.write_text(text)
print(f"prepared {len(rows)} task links under {canonical}")
print(f"physical data root: {spill_hub / 'cache'}")
PY
}

spec() {
    awk -F '\t' -v tag="$1" '$1 == tag {print; exit}' "$TASKS"
}

episode_count() {
    local target=$1
    python3 - "$target/client/summaries.json" <<'PY'
import json
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
try:
    print(len(json.loads(p.read_text())))
except (OSError, ValueError):
    print(0)
PY
}

task_complete() {
    local target=$1 n
    n=$(episode_count "$target")
    [ "$n" -ge "$EPISODES" ] && grep -q '"status": "complete"' "$target/meta.json" 2>/dev/null
}

active_on_gpu() {
    local gpu=$1 count=0 file tag assigned
    for file in "$PIDS"/c-*.pid; do
        [ -f "$file" ] || continue
        alive_pidfile "$file" || continue
        tag=$(basename "$file" .pid)
        assigned=$(cat "$PIDS/$tag.gpu" 2>/dev/null)
        [ "$assigned" = "$gpu" ] && count=$((count + 1))
    done
    if [ "$gpu" = 0 ] && alive_pidfile "$B_OB8_PID"; then
        count=$((count + 1))
    fi
    echo "$count"
}

free_mib() {
    nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -dc '0-9'
}

launch_one() {
    local tag=$1 gpu=$2 line port bench task hub_suite task_name target
    line=$(spec "$tag")
    [ -n "$line" ] || { say "unknown task $tag"; return 1; }
    IFS=$'\t' read -r _ _ port bench task hub_suite task_name target <<< "$line"
    task_complete "$target" && return 0
    alive_pidfile "$PIDS/$tag.pid" && return 0

    local free
    free=$(free_mib "$gpu")
    if [ -z "$free" ] || [ "$free" -lt "$MIN_FREE_MIB" ]; then
        say "gpu$gpu has ${free:-unknown} MiB free; need $MIN_FREE_MIB before another model"
        return 1
    fi
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"; then
        say "$tag port $port is already in use"
        return 1
    fi

    local attempts=0 force=() log
    [ -f "$PIDS/$tag.attempts" ] && attempts=$(cat "$PIDS/$tag.attempts")
    attempts=$((attempts + 1))
    echo "$attempts" > "$PIDS/$tag.attempts"
    [ -e "$target" ] && force=(--force)
    log="$LOGS/$tag.attempt${attempts}.log"
    nohup setsid python3 -u "$RC/run_corpus_capture.py" \
        --hub-root "$SPILL_HUB" --run-id "$RUN_ID" \
        --gpu "$gpu" --port "$port" \
        --benchmarks "$bench" --tasks "$task" \
        --scenes 50 --draws "$DRAWS" --noise-seed-base "$SEED_BASE" \
        "${force[@]}" > "$log" 2>&1 &
    echo $! > "$PIDS/$tag.pid"
    echo "$gpu" > "$PIDS/$tag.gpu"
    say "$tag gpu$gpu :$port $bench task $task attempt $attempts (pid $!)"
}

next_pending_for_gpu() {
    local gpu=$1 tag assigned target n attempts
    while IFS=$'\t' read -r tag assigned _ _ _ _ _ target; do
        [ -z "$gpu" ] || [ "$assigned" = "$gpu" ] || continue
        task_complete "$target" && continue
        alive_pidfile "$PIDS/$tag.pid" && continue
        if [ -f "$PIDS/$tag.pid" ]; then
            n=$(episode_count "$target")
            attempts=$(cat "$PIDS/$tag.attempts" 2>/dev/null || echo 1)
            # Startup failures are safe to retry because they have no episodes.
            [ "$n" -eq 0 ] && [ "$attempts" -lt "$MAX_ZERO_EPISODE_ATTEMPTS" ] || continue
        fi
        echo "$tag"
        return 0
    done < "$TASKS"
    return 1
}

recover_config_races() {
    local tag target n attempts log stamp dest
    while IFS=$'\t' read -r tag _ _ _ _ _ _ target; do
        [ -f "$PIDS/$tag.pid" ] || continue
        alive_pidfile "$PIDS/$tag.pid" && continue
        [ -d "$target" ] || continue
        n=$(episode_count "$target")
        [ "$n" -gt 0 ] || continue
        log="$target/logs/client.log"
        grep -q "TypeError: 'NoneType' object is not iterable" "$log" 2>/dev/null || continue
        grep -q 'get_libero_path' "$log" 2>/dev/null || continue
        attempts=$(cat "$PIDS/$tag.attempts" 2>/dev/null || echo 1)
        if [ "$attempts" -ge "$MAX_ZERO_EPISODE_ATTEMPTS" ]; then
            say "$tag config-race recovery exhausted after $attempts attempts"
            continue
        fi
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
        dest="$RECOVERY/${tag}.attempt${attempts}.${n}episodes.$stamp"
        mkdir -p "$RECOVERY"
        if [ -e "$dest" ]; then
            say "$tag recovery destination already exists: $dest"
            continue
        fi
        mv -- "$target" "$dest"
        say "$tag archived $n partial episodes at $dest; queued full recapture"
    done < "$TASKS"
}

halt_jobs() {
    local file pid
    for file in "$PIDS"/c-*.pid; do
        [ -f "$file" ] || continue
        pid=$(cat "$file" 2>/dev/null)
        [ -n "$pid" ] || continue
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    done
}

status() {
    prepare >/dev/null
    python3 - "$TASKS" "$EPISODES" "$PIDS" <<'PY'
import collections
import json
import pathlib
import sys

tasks = pathlib.Path(sys.argv[1])
want = int(sys.argv[2])
pids = pathlib.Path(sys.argv[3])
states = collections.Counter()
episodes = 0
for line in tasks.read_text().splitlines():
    tag, gpu, port, bench, task, hub, name, target = line.split("\t")
    target = pathlib.Path(target)
    try:
        rows = json.loads((target / "client/summaries.json").read_text())
    except (OSError, ValueError):
        rows = []
    episodes += len(rows)
    try:
        meta = json.loads((target / "meta.json").read_text())
        state = meta.get("status", "unknown")
    except (OSError, ValueError):
        state = "pending"
    if len(rows) >= want and state == "complete":
        state = "complete"
    states[state] += 1
    if state not in {"complete", "pending"}:
        actual_gpu = pids / f"{tag}.gpu"
        if actual_gpu.is_file():
            gpu = actual_gpu.read_text().strip() or gpu
        print(f"{tag:8s} g{gpu} {len(rows):3d}/{want} {state:8s} {hub}/{name}")
print(f"tasks: {dict(sorted(states.items()))}")
print(f"episodes: {episodes}/{40 * want}")
PY
    local gpu power used util active
    for gpu in "${GPUS[@]}"; do
        IFS=, read -r power used util < <(nvidia-smi -i "$gpu" \
            --query-gpu=power.draw,memory.used,utilization.gpu \
            --format=csv,noheader,nounits | tr -d ' ')
        active=$(active_on_gpu "$gpu")
        printf 'gpu%s jobs=%s power=%sW memory=%sMiB util=%s%%' "$gpu" "$active" "$power" "$used" "$util"
        awk -v p="$power" -v target="$POWER_TARGET_W" 'BEGIN {if (p + 0 < target) printf "  (< %.0fW target)", target}'
        printf '\n'
    done
    df -h "$SPILL_HUB" "$CANON" | awk 'NR == 1 || !seen[$1]++'
}

supervise() {
    trap 'rc=$?; say "C supervisor exit rc=$rc"' EXIT
    prepare
    say "C supervisor: seeds $SEED_BASE-$((SEED_BASE + DRAWS - 1)), GPUs ${GPUS[*]}, max $MAX_PER_GPU jobs/GPU"
    local cycles=0 gpu slots tag launched complete failed n attempts free_gb
    while :; do
        free_gb=$(df -BG --output=avail "$SPILL_HUB" | tail -n 1 | tr -dc '0-9')
        if [ "$free_gb" -lt "$HALT_FREE_GB" ]; then
            say "spill disk low: ${free_gb}G < ${HALT_FREE_GB}G; stopping C jobs"
            halt_jobs
            exit 1
        fi

        recover_config_races

        launched=0
        for gpu in "${GPUS[@]}"; do
            slots=$(active_on_gpu "$gpu")
            [ "$slots" -lt "$MAX_PER_GPU" ] || continue
            tag=$(next_pending_for_gpu "$gpu" || true)
            # GPUs added after the campaign started have no home rows; they and
            # any primary GPU with an exhausted home queue steal global work.
            [ -n "$tag" ] || tag=$(next_pending_for_gpu "" || true)
            [ -n "$tag" ] || continue
            if launch_one "$tag" "$gpu"; then
                launched=1
                sleep 15
            fi
        done

        complete=0; failed=0
        while IFS=$'\t' read -r tag _ _ _ _ _ _ target; do
            if task_complete "$target"; then
                complete=$((complete + 1))
            elif [ -f "$PIDS/$tag.pid" ] && ! alive_pidfile "$PIDS/$tag.pid"; then
                n=$(episode_count "$target")
                attempts=$(cat "$PIDS/$tag.attempts" 2>/dev/null || echo 1)
                if [ "$n" -gt 0 ] || [ "$attempts" -ge "$MAX_ZERO_EPISODE_ATTEMPTS" ]; then
                    failed=$((failed + 1))
                fi
            fi
        done < "$TASKS"
        [ "$complete" -eq 40 ] && { say "C CAMPAIGN COMPLETE"; exit 0; }

        cycles=$((cycles + 1))
        if [ "$launched" -eq 0 ] || [ $((cycles % 10)) -eq 0 ]; then
            say "$complete/40 complete, $failed manual-review failures"
        fi
        sleep 30
    done
}

case "${1:-status}" in
    prepare) prepare ;;
    start)
        prepare
        if alive_pidfile "$SUPERVISOR_PID"; then
            say "supervisor already running (pid $(cat "$SUPERVISOR_PID"))"
            exit 0
        fi
        nohup setsid bash "$0" supervise >> "$LOGS/supervisor.log" 2>&1 &
        echo $! > "$SUPERVISOR_PID"
        say "supervisor launched (pid $!)"
        ;;
    supervise) supervise ;;
    status) status ;;
    stop)
        # Stop the scheduler first so it cannot fill a slot while jobs are
        # being drained.  Task children are then signalled before their parent
        # orchestrators, which lets route writers flush normally.
        alive_pidfile "$SUPERVISOR_PID" && kill -TERM "$(cat "$SUPERVISOR_PID")" 2>/dev/null || true
        halt_jobs
        say "C campaign stop signals sent; GPU 6 untouched"
        ;;
    *) echo "usage: $0 {prepare|start|supervise|status|stop}"; exit 2 ;;
esac
