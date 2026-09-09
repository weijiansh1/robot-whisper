#!/usr/bin/env bash
# 采集：libero_10/t08 的完整 flow 轨迹 + 结局标签，逐初态一个 run 目录。
#
#   bash code/capture.sh smoke        # 1 初态 x 2 集，量速度
#   bash code/capture.sh lane A       # 初态 0,6,12,18,24,30,36,42
#   bash code/capture.sh lane B       # 初态 3,9,15,21,27,33,39,45
#   bash code/capture.sh status
#
# 服务端与客户端环境照 run_flow_lead.sh（已知可用）。线程上限必须设：本机 240 核，
# OpenBLAS/llvmpipe 默认按核数开线程会把 numpy 拖慢 100 倍。
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
BR=/home/jovyan/.cache/himoe-libero-bridge
CAP="$HERE/capture"
LOGS="$HERE/capture/_logs"
PIDS="$HERE/capture/_pids"
mkdir -p "$CAP" "$LOGS" "$PIDS"

GPU=6
SUITE=long
BENCH=libero_10
TASK=8
WRIST=checkpoint-right          # 与既有 flow 语料一致；绝不可与 paper-right 同表
SEED_BASE=7000
N_EP=8
MAXSTEPS=520

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4

# 一个初态 = 一个服务端 + 一个客户端，串行跑完 N_EP 集后落盘退出。
run_one() { # port init_state n_ep tag
    local port=$1 init=$2 nep=$3 tag=$4
    local out="$CAP/$tag"
    if [ -f "$out/episodes.json" ]; then echo "[skip] $tag 已完成"; return 0; fi
    rm -rf "$out"; mkdir -p "$out"

    PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
    MOEVLA_DATA_HOME=$BR/moevla-data \
    setsid nohup "$BR/envs/model/bin/python" -u "$RC/serve_flow_trace.py" \
        --port "$port" --gpu "$GPU" --suite "$SUITE" \
        --checkpoint-dir "$BR/checkpoints/HiMoE-VLA-Libero-10" \
        --upstream-root "$BR/upstream/HiMoE-VLA" \
        --libero-wrist-layout "$WRIST" \
        --out "$out" > "$LOGS/$tag.server.log" 2>&1 < /dev/null &
    local spid=$!
    echo "$spid" > "$PIDS/$tag.server.pid"

    local ok=0
    for _ in $(seq 1 120); do
        sleep 5
        grep -q "serving on ws" "$LOGS/$tag.server.log" 2>/dev/null && { ok=1; break; }
        grep -qi "Traceback" "$LOGS/$tag.server.log" 2>/dev/null && break
    done
    if [ "$ok" != 1 ]; then
        echo "[FAIL] $tag 服务端未就绪"; tail -20 "$LOGS/$tag.server.log"
        kill -TERM "$spid" 2>/dev/null; return 1
    fi
    echo "[$(date +%H:%M:%S)] $tag 服务端就绪 :$port (pid $spid)"

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
    LD_LIBRARY_PATH=$BR/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
    "$BR/envs/libero/bin/python" -u "$RC/rollout_flow_lead.py" \
        --port "$port" --benchmark "$BENCH" --task-id "$TASK" \
        --init-state-id "$init" --n-episodes "$nep" --n-candidates 1 \
        --noise-seed-base "$SEED_BASE" --max-steps "$MAXSTEPS" \
        --label "$tag" --out "$out" \
        --libero-root "$BR/upstream/LIBERO" > "$LOGS/$tag.client.log" 2>&1
    local rc=$?

    # 服务端最后一块数据在写缓冲里，必须走 TERM 才 flush 落盘
    kill -TERM "$spid" 2>/dev/null
    sleep 12
    echo "[$(date +%H:%M:%S)] $tag 结束 rc=$rc -> $out"
    return $rc
}

case "${1:-}" in
smoke)
    run_one 8890 0 2 smoke-init00
    ;;
lane)
    lane="${2:?需要 A 或 B}"
    case "$lane" in
        A) inits=(0 6 12 18 24 30 36 42); port=8891 ;;
        B) inits=(3 9 15 21 27 33 39 45); port=8892 ;;
        *) echo "lane 只能是 A 或 B"; exit 1 ;;
    esac
    for i in "${inits[@]}"; do
        run_one "$port" "$i" "$N_EP" "$(printf 'init%02d' "$i")"
    done
    echo "[$(date +%H:%M:%S)] lane $lane 全部结束"
    ;;
status)
    tot=0; ok=0; done_n=0
    for f in "$CAP"/init*/episodes.json; do
        [ -f "$f" ] || continue
        done_n=$((done_n+1))
        read n s < <(python3 -c "
import json,sys; d=json.load(open('$f'))
print(len(d), sum(1 for e in d if e['success']))")
        tot=$((tot+n)); ok=$((ok+s))
    done
    echo "完成初态 $done_n/16   集数 $tot/128   成功 $ok"
    ls "$CAP" 2>/dev/null | grep -c '^init' | xargs echo "run 目录数:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '7p'
    ;;
stop)
    for f in "$PIDS"/*.pid; do [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null; done
    echo stopped
    ;;
*)
    echo "用法: capture.sh {smoke|lane A|lane B|status|stop}"; exit 1 ;;
esac
