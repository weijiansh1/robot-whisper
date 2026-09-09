#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 RUN_ROOT LIBERO_ROOT AUDITOR" >&2
    exit 2
fi

run_root=$1
libero_root=$2
auditor=$3
status_file="${run_root}/FINALIZER_STATUS.txt"

write_status() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$status_file"
}

write_status "waiting_for_collectors"
while pgrep -f "rolling_star_collect.py.*${run_root}/formal/worker" >/dev/null; do
    sleep 20
done

incomplete=$(
    find "${run_root}/formal" -mindepth 2 -maxdepth 2 -type d \
        -name 'snapshot_*' ! -exec test -f '{}/manifest.json' ';' -print
)
if [[ -n "$incomplete" ]]; then
    write_status "error_incomplete_snapshots ${incomplete//$'\n'/ }"
    exit 1
fi

write_status "collectors_stopped_auditing"
env \
    PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/himoe-route-capture \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    /home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python \
    "$auditor" \
    --run-root "$run_root" \
    --libero-root "$libero_root" \
    > "${run_root}/formal/physical_audit/final_audit_stdout.log" 2>&1

mapfile -t server_pids < <(
    pgrep -f \
        "^/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python -u .*serve_with_recorder.py.*--out ${run_root}/formal/server$" \
        || true
)
if [[ ${#server_pids[@]} -ne 1 ]]; then
    write_status "error_expected_one_server found=${#server_pids[@]}"
    exit 1
fi
write_status "audit_passed_stopping_server pid=${server_pids[0]}"
kill -TERM "${server_pids[0]}"
for _ in $(seq 1 90); do
    if ! kill -0 "${server_pids[0]}" 2>/dev/null; then
        break
    fi
    sleep 2
done
if kill -0 "${server_pids[0]}" 2>/dev/null; then
    write_status "error_server_did_not_exit_after_sigterm"
    exit 1
fi
if [[ ! -s "${run_root}/formal/server/capture_summary.json" ]]; then
    write_status "error_missing_capture_summary"
    exit 1
fi

tmux send-keys -t rs_gpu C-c 2>/dev/null || true
sleep 2
write_status "writers_stopped_hashing"
(
    cd "$run_root"
    find formal logs -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        > raw_sha256sums.txt.tmp
    mv raw_sha256sums.txt.tmp raw_sha256sums.txt
)
write_status "complete manifests=$(find "${run_root}/formal" -name manifest.json | wc -l)"
touch "${run_root}/FINALIZED"
