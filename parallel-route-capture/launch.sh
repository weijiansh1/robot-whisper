#!/bin/bash
# Parallel MoE route-capture orchestrator (per-task servers, VLA_MUI_HUB cache_new layout).
#
#   bash launch.sh auto    # gate -> smoke -> full  (safe to start while GPUs are busy)
#   bash launch.sh gate    # block until every GPU has enough free memory
#   bash launch.sh smoke   # 1 server + 1 worker, 1 snapshot, verify routes.zarr
#   bash launch.sh full    # start all servers + workers + calvin + watchdog
#   bash launch.sh stop    # SIGTERM clients first, then servers (flush path)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN=$(python3 -c "import json;print(json.load(open('$HERE/plan.json'))['paths']['run_root'])")
DEADLINE_H=$(python3 -c "import json;print(json.load(open('$HERE/plan.json'))['deadline_hours'])")
LOGS="$RUN/logs"; PIDS="$RUN/pids"
mkdir -p "$LOGS" "$PIDS"
NEED_FREE_MIB=120000  # 5 servers/GPU x ~22.6 GiB + EGL contexts

say() { echo "[$(date +%H:%M:%S)] $*"; }

wait_port() { # host port timeout_s
    python3 - "$1" "$2" "$3" <<'EOF'
import socket, sys, time
host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        socket.create_connection((host, port), 2).close(); sys.exit(0)
    except OSError:
        time.sleep(5)
sys.exit(1)
EOF
}

gate() {
    say "waiting until all GPUs have >= ${NEED_FREE_MIB} MiB free ..."
    while true; do
        min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
        [ "$min_free" -ge "$NEED_FREE_MIB" ] && { say "GPUs free (min ${min_free} MiB)"; return 0; }
        sleep 60
    done
}

start_bg() { # name script logpath  (logpath absolute, or relative to $LOGS)
    local log="$3"
    case "$log" in /*) ;; *) log="$LOGS/$log" ;; esac
    mkdir -p "$(dirname "$log")"
    nohup bash "$HERE/cmds/$2" > "$log" 2>&1 &
    echo $! > "$PIDS/$1.pid"
    say "started $1 (pid $!)"
}

smoke() {
    say "SMOKE: server + 1 worker, 1 snapshot, K=16"
    rm -rf "$RUN/smoke"
    start_bg smoke-server server-smoke.sh smoke-server.log
    port=$(python3 -c "import json;print(json.load(open('$HERE/plan.json'))['smoke']['port'])")
    if ! wait_port 127.0.0.1 "$port" 900; then
        say "SMOKE FAILED: server never opened port $port"; tail -20 "$LOGS/smoke-server.log"
        kill "$(cat "$PIDS/smoke-server.pid")" 2>/dev/null; return 1
    fi
    say "smoke server ready; running worker"
    STOP_BEFORE_UNIX=0 bash "$HERE/cmds/worker-smoke.sh" > "$LOGS/smoke-worker.log" 2>&1
    rc=$?
    kill -TERM "$(cat "$PIDS/smoke-server.pid")" 2>/dev/null
    sleep 10
    [ $rc -ne 0 ] && { say "SMOKE FAILED: worker rc=$rc"; tail -20 "$LOGS/smoke-worker.log"; return 1; }
    python3 - "$RUN" <<'EOF' || return 1
import sys, pathlib, zarr
run = pathlib.Path(sys.argv[1])
z = zarr.open_group(str(run / "smoke/server/routes.zarr"), mode="r")
rows = z["episode_id"].shape[0]
snaps = len(list((run / "smoke/shard").glob("snapshot_*/manifest.json")))
print(f"smoke verified: routes rows={rows}, snapshots={snaps}")
assert rows > 0 and snaps >= 1
EOF
    say "SMOKE PASSED"
}

alive_any() { # pid-name...
    for n in "$@"; do
        [ -f "$PIDS/$n.pid" ] && [ -d "/proc/$(cat "$PIDS/$n.pid")" ] && return 0
    done
    return 1
}

full() {
    local man="$HERE/cmds/manifest.json"
    local n_waves
    n_waves=$(python3 -c "import json;print(json.load(open('$man'))['n_waves'])")
    export STOP_BEFORE_UNIX=$(( $(date +%s) + DEADLINE_H * 3600 ))
    say "FULL: $n_waves waves; worker deadline $(date -d @$STOP_BEFORE_UNIX '+%F %T')"
    [ -f "$PIDS/watchdog.pid" ] && kill "$(cat "$PIDS/watchdog.pid")" 2>/dev/null
    nohup bash "$HERE/monitor.sh" watchdog > "$LOGS/watchdog.log" 2>&1 &
    echo $! > "$PIDS/watchdog.pid"

    # CALVIN spans all waves on its pinned GPU slot; on a relaunch after a
    # partial failure the healthy calvin pair keeps running — don't double-start.
    IFS=$'\t' read -r cname cport cscript clog cclient_log < <(python3 -c "
import json
c = json.load(open('$man'))['calvin']
print('\t'.join([c['name'], str(c['port']), c['script'], c['log'], c['client_log']]))")
    if alive_any calvin-client; then
        say "calvin client already running — leaving the pair untouched"
    else
        start_bg "$cname" "$cscript" "$clog"
        if wait_port 127.0.0.1 "$cport" 1800; then
            start_bg calvin-client calvin-client.sh "$cclient_log"
        else
            say "WARN: calvin server never opened :$cport — continuing with LIBERO waves"
            tail -5 "$clog"
        fi
    fi

    for (( w=0; w<n_waves; w++ )); do
        say "=== wave $w: starting servers ==="
        while IFS=$'\t' read -r name port script log; do
            start_bg "$name" "$script" "$log"
        done < <(python3 -c "
import json
for s in json.load(open('$man'))['servers']:
    if s['wave'] == $w:
        print('\t'.join([s['name'], str(s['port']), s['script'], s['log']]))")
        while IFS=$'\t' read -r name port log; do
            if ! wait_port 127.0.0.1 "$port" 1800; then
                say "FATAL: $name never opened port $port"; tail -20 "$log"
                stop; return 1
            fi
        done < <(python3 -c "
import json
for s in json.load(open('$man'))['servers']:
    if s['wave'] == $w:
        print('\t'.join([s['name'], str(s['port']), s['log']]))")
        say "wave $w servers up; starting workers"
        wave_workers=()
        while IFS=$'\t' read -r wname log; do
            start_bg "$wname" "$wname.sh" "$log"
            wave_workers+=("$wname")
            sleep 2
        done < <(python3 -c "
import json
for sh in json.load(open('$man'))['shards']:
    if sh['wave'] == $w:
        print('\t'.join([sh['name'], sh['log']]))")
        say "wave $w: ${#wave_workers[@]} workers running; waiting for completion"
        while alive_any "${wave_workers[@]}"; do
            [ "$(date +%s)" -ge "$STOP_BEFORE_UNIX" ] && { say "deadline hit"; break; }
            sleep 30
        done
        say "wave $w workers done; flushing wave servers"
        while read -r name; do
            [ -f "$PIDS/$name.pid" ] && kill -TERM "$(cat "$PIDS/$name.pid")" 2>/dev/null
        done < <(python3 -c "
import json
for s in json.load(open('$man'))['servers']:
    if s['wave'] == $w:
        print(s['name'])")
        sleep 15
    done

    say "all LIBERO waves done; waiting for calvin client"
    while alive_any calvin-client; do
        [ "$(date +%s)" -ge "$STOP_BEFORE_UNIX" ] && { say "deadline hit"; break; }
        sleep 60
    done
    [ -f "$PIDS/$cname.pid" ] && kill -TERM "$(cat "$PIDS/$cname.pid")" 2>/dev/null
    [ -f "$PIDS/watchdog.pid" ] && kill "$(cat "$PIDS/watchdog.pid")" 2>/dev/null
    say "CAMPAIGN COMPLETE; final status:"
    bash "$HERE/monitor.sh" status
}

stop() {
    say "stopping clients first (workers flush per-snapshot; servers flush on TERM)"
    for f in "$PIDS"/worker-*.pid "$PIDS"/calvin-client.pid; do
        [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null
    done
    sleep 20
    for f in "$PIDS"/server-*.pid "$PIDS"/smoke-server.pid "$PIDS"/watchdog.pid; do
        [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null
    done
    say "stop signals sent"
}

case "${1:-auto}" in
    gate) gate ;;
    smoke) smoke ;;
    full) full ;;
    stop) stop ;;
    auto) gate && smoke && full ;;
    *) echo "usage: launch.sh [auto|gate|smoke|full|stop]"; exit 2 ;;
esac
