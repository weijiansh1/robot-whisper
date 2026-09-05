#!/bin/bash
# Progress + health for the parallel capture campaign (cache_new per-task layout).
#   bash monitor.sh            # one-shot status table
#   bash monitor.sh watchdog   # loop: halt everything if disk free < threshold
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN=$(python3 -c "import json;print(json.load(open('$HERE/plan.json'))['paths']['run_root'])")
HALT_GB=$(python3 -c "import json;print(json.load(open('$HERE/plan.json'))['disk_halt_free_gb'])")

status() {
    python3 - "$HERE" "$RUN" <<'EOF'
import json, pathlib, subprocess, sys
here, run = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
man = json.load(open(here / "cmds/manifest.json"))
pids = {p.stem: int(p.read_text()) for p in (run / "pids").glob("*.pid")} if (run / "pids").exists() else {}

def alive(name):
    pid = pids.get(name)
    if pid is None:
        return "-"
    return "UP" if pathlib.Path(f"/proc/{pid}").exists() else "DEAD"

total_snap = total_done = 0
print(f"{'shard':58s} {'gpu':3s} {'proc':5s} {'snapshots':>10s}")
for sh in man["shards"]:
    done = len(list(pathlib.Path(sh["out"]).glob("snapshot_*/manifest.json")))
    total_snap += sh["mtq"]; total_done += min(done, sh["mtq"])
    state = alive(sh["name"])
    mark = "DONE" if done >= sh["mtq"] else state
    print(f"{sh['label']:58s} g{sh['gpu']}  {mark:5s} {done:>5d}/{sh['mtq']}")
print(f"\nLIBERO snapshots: {total_done}/{total_snap} "
      f"(={total_done*16}/{total_snap*16} branches)")
up = sum(1 for s in man["servers"] if alive(s["name"]) == "UP")
dead = [s["name"] for s in man["servers"] if alive(s["name"]) == "DEAD"]
print(f"task servers up now: {up}/{len(man['servers'])} (waves run in sequence)"
      + (f"  DEAD: {dead}" if dead else ""))
cal = man.get("calvin")
if cal:
    seqf = pathlib.Path(cal["run_dir"]) / "client/sequences.jsonl"
    n = sum(1 for _ in open(seqf)) if seqf.exists() else 0
    print(f"calvin sequences: {n}  server {alive(cal['name'])}  client {alive('calvin-client')}")
free = subprocess.run(["df", "-BG", "--output=avail", str(here)],
                      capture_output=True, text=True).stdout.split()[-1]
print(f"disk free: {free}")
EOF
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/gpu /'
}

watchdog() {
    while true; do
        free_gb=$(df -BG --output=avail "$HERE" | tail -1 | tr -dc '0-9')
        if [ "$free_gb" -lt "$HALT_GB" ]; then
            echo "[$(date +%H:%M:%S)] DISK LOW (${free_gb}G < ${HALT_GB}G) - halting campaign"
            bash "$HERE/launch.sh" stop
            exit 1
        fi
        sleep 120
    done
}

case "${1:-status}" in
    status) status ;;
    watchdog) watchdog ;;
    *) echo "usage: monitor.sh [status|watchdog]"; exit 2 ;;
esac
