#!/bin/bash
# Persistent HiMoE-VLA matrix for online experiments.
#
# One shared host process keeps a complete five-model task bundle on every GPU.
# Sharing the Python/PyTorch runtime is required by this container's 8 GiB host
# memory limit; model parameters remain isolated and each GPU has one serialized
# CUDA worker shared by its five task endpoints.
#
#             goal  spatial  object  long  calvin
#   GPU 0     8800    8801    8802  8803    8804
#   GPU 1     8810    8811    8812  8813    8814
#   GPU 2     8820    8821    8822  8823    8824
#   GPU 3     8830    8831    8832  8833    8834
#   GPU 4     8840    8841    8842  8843    8844
#   GPU 5     8850    8851    8852  8853    8854
#   GPU 6     8860    8861    8862  8863    8864
#   GPU 7     8870    8871    8872  8873    8874
#
# Usage: bash online_servers.sh start|status|verify|stop [gpu ...]
set -u

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072

HERE="$(cd "$(dirname "$0")" && pwd)"
M=/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python
SESSION=himoe-vla-online-matrix
PID_FILE="$HERE/logs/matrix.pid"
GPU_FILE="$HERE/logs/matrix.gpus"
LOG_FILE="$HERE/logs/matrix.log"
OLD_SESSION_PREFIX=himoe-vla-online-bundle-gpu
GPUS="0 1 2 3 4 5 6 7"
MODELS="goal spatial object long calvin"
NEED_FREE_MIB=95000
mkdir -p "$HERE/logs"

port_for() { # physical-gpu model-offset
    echo $((8800 + $1 * 10 + $2))
}

matrix_pid() {
    cat "$PID_FILE" 2>/dev/null
}

matrix_alive() {
    local pid
    pid=$(matrix_pid)
    [ -n "$pid" ] && [ -d "/proc/$pid" ] && \
        tmux has-session -t "=$SESSION" 2>/dev/null
}

port_ready() {
    ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .
}

bundle_ready() {
    local gpu=$1 offset
    for offset in 0 1 2 3 4; do
        port_ready "$(port_for "$gpu" "$offset")" || return 1
    done
}

gpu_selected() {
    local gpu=$1 selected
    selected=$(cat "$GPU_FILE" 2>/dev/null)
    case " $selected " in
        *" $gpu "*) return 0 ;;
        *) return 1 ;;
    esac
}

validate_gpu() {
    case "$1" in
        0|1|2|3|4|5|6|7) return 0 ;;
        *) echo "refusing GPU $1; managed GPUs are 0-7" >&2; return 1 ;;
    esac
}

validate_targets() {
    local gpu seen=" "
    for gpu in $1; do
        validate_gpu "$gpu" || return 1
        case "$seen" in
            *" $gpu "*) echo "GPU $gpu was requested more than once" >&2; return 1 ;;
        esac
        seen="$seen$gpu "
    done
}

stop_old_servers() {
    local name session pid running="" any gpu
    for name in goal-a spatial-a object-a long-a calvin-a; do
        session="himoe-vla-online-$name"
        if tmux has-session -t "=$session" 2>/dev/null; then
            pid=$(tmux display-message -p -t "=$session:0.0" '#{pane_pid}')
            [ -n "$pid" ] && [ -d "/proc/$pid" ] && kill -TERM "$pid" 2>/dev/null
            running="$running $session"
            echo "stopping legacy $name (${pid:-unknown})"
        fi
    done
    for gpu in $GPUS; do
        session="$OLD_SESSION_PREFIX$gpu"
        if tmux has-session -t "=$session" 2>/dev/null; then
            pid=$(tmux display-message -p -t "=$session:0.0" '#{pane_pid}')
            [ -n "$pid" ] && [ -d "/proc/$pid" ] && kill -TERM "$pid" 2>/dev/null
            running="$running $session"
            echo "stopping old gpu$gpu bundle (${pid:-unknown})"
        fi
    done
    [ -z "$running" ] && return 0
    for _ in $(seq 1 60); do
        any=0
        for session in $running; do
            tmux has-session -t "=$session" 2>/dev/null && any=1
        done
        [ "$any" -eq 0 ] && return 0
        sleep 1
    done
    echo "old servers did not finish shutdown within 60 seconds" >&2
    return 1
}

wait_ready() { # physical-gpu matrix-pid
    local gpu=$1 pid=$2 attempt
    for attempt in $(seq 1 900); do
        if [ -z "$pid" ] || ! [ -d "/proc/$pid" ]; then
            echo "model matrix exited before gpu$gpu became ready" >&2
            tail -80 "$LOG_FILE" >&2
            return 1
        fi
        if bundle_ready "$gpu"; then
            echo "[$(date +%H:%M:%S)] gpu$gpu all five models ready"
            return 0
        fi
        sleep 2
    done
    echo "gpu$gpu did not become ready within 1800 seconds" >&2
    tail -80 "$LOG_FILE" >&2
    return 1
}

start_matrix() { # space-separated physical GPU indices
    local targets=$1 gpu offset port free_mib csv command pid
    validate_targets "$targets" || return 1
    if matrix_alive; then
        for gpu in $targets; do
            if ! gpu_selected "$gpu"; then
                echo "running matrix does not include gpu$gpu; stop it before changing GPU set" >&2
                return 1
            fi
        done
        pid=$(matrix_pid)
        echo "model matrix already running (pid $pid)"
        for gpu in $targets; do wait_ready "$gpu" "$pid" || return 1; done
        return 0
    fi

    stop_old_servers || return 1
    for gpu in $targets; do
        for offset in 0 1 2 3 4; do
            port=$(port_for "$gpu" "$offset")
            if port_ready "$port"; then
                echo "port $port is already occupied; refusing to start matrix" >&2
                return 1
            fi
        done
        free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free \
            --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ -z "$free_mib" ] || [ "$free_mib" -lt "$NEED_FREE_MIB" ]; then
            echo "gpu$gpu has ${free_mib:-unknown} MiB free; need $NEED_FREE_MIB MiB" >&2
            return 1
        fi
    done

    tmux has-session -t "=$SESSION" 2>/dev/null && tmux kill-session -t "=$SESSION"
    csv=$(echo "$targets" | tr ' ' ',')
    cmd=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$csv" \
        "$M" -u "$HERE/serve_model_matrix.py" \
        --gpus "$csv" --host 127.0.0.1 --port-base 8800)
    printf -v command '%q ' "${cmd[@]}"
    printf -v command 'exec %s> %q 2>&1' "$command" "$LOG_FILE"
    : > "$LOG_FILE"
    tmux new-session -d -s "$SESSION" -c "$HERE" "$command"
    pid=$(tmux display-message -p -t "=$SESSION:0.0" '#{pane_pid}')
    echo "$pid" > "$PID_FILE"
    echo "$targets" > "$GPU_FILE"
    echo "[$(date +%H:%M:%S)] model matrix loading GPUs $targets (pid $pid)"
    for gpu in $targets; do wait_ready "$gpu" "$pid" || return 1; done
}

stop_matrix() {
    local pid
    if ! matrix_alive; then
        tmux has-session -t "=$SESSION" 2>/dev/null && tmux kill-session -t "=$SESSION"
        echo "model matrix already stopped"
        return 0
    fi
    pid=$(matrix_pid)
    kill -TERM "$pid"
    echo "stopping model matrix ($pid)"
    for _ in $(seq 1 60); do
        [ -d "/proc/$pid" ] || return 0
        sleep 1
    done
    echo "model matrix did not stop within 60 seconds" >&2
    return 1
}

show_status() {
    local gpu offset model port proc rss used current maximum pid
    printf "%-5s %-8s %-8s %-8s %-8s %-8s %-12s %-10s\n" \
        GPU GOAL SPATIAL OBJECT LONG CALVIN MATRIX_RSS VRAM_MIB
    rss="-"
    if matrix_alive; then
        pid=$(matrix_pid)
        rss=$(awk '/VmRSS/ {print int($2 / 1024)}' "/proc/$pid/status")
    fi
    for gpu in $GPUS; do
        proc=down
        matrix_alive && gpu_selected "$gpu" && proc=loading
        bundle_ready "$gpu" && proc=UP
        set --
        offset=0
        for model in $MODELS; do
            port=$(port_for "$gpu" "$offset")
            if port_ready "$port"; then set -- "$@" ":$port"; else set -- "$@" "-"; fi
            offset=$((offset + 1))
        done
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
            --format=csv,noheader,nounits | head -1 | tr -d ' ')
        printf "%-5s %-8s %-8s %-8s %-8s %-8s %-12s %-10s # %s\n" \
            "$gpu" "$1" "$2" "$3" "$4" "$5" "$rss" "$used" "$proc"
    done
    current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    maximum=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown)
    awk -v current="$current" -v maximum="$maximum" \
        'BEGIN { printf "cgroup memory: %.2f GiB / ", current / 1073741824; if (maximum == "max" || maximum == "unknown") print maximum; else printf "%.2f GiB\n", maximum / 1073741824 }'
}

command=${1:-status}
shift || true
targets="${*:-$GPUS}"

case "$command" in
start)
    start_matrix "$targets"
    result=$?
    show_status
    exit "$result"
    ;;
status)
    show_status
    ;;
verify)
    validate_targets "$targets" || exit 1
    csv=$(echo "$targets" | tr ' ' ',')
    matrix_csv=$(tr ' ' ',' < "$GPU_FILE")
    exec "$M" -u "$HERE/verify_model_bundles.py" \
        --gpus "$csv" --matrix-gpus "$matrix_csv"
    ;;
stop)
    if [ "$targets" != "$GPUS" ]; then
        echo "the shared matrix cannot stop one GPU independently; run stop without GPU arguments" >&2
        exit 1
    fi
    stop_matrix
    ;;
*)
    echo "usage: bash online_servers.sh start|status|verify|stop [gpu ...]" >&2
    exit 2
    ;;
esac
