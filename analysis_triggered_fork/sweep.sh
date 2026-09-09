#!/usr/bin/env bash
# Hunt for a rescue: trunk fails, detector fires, at least one resampled branch succeeds.
#
# Init states are ordered by how often resampling at q0 changed the outcome in
# rolling-star, because a rescue needs both a failing trunk and a rescuable state:
#   init 7  (w3) 13-15/16 succeed  -- resampling nearly always works, but the trunk
#                                     rarely fails, so most draws will be unusable
#   init 0  (w0)  1-7/16           -- the mixed regime; the live question lives here
#   init 3  (w2)  0-2/16           -- near-hopeless, kept as a negative control
# Seeds vary the trunk's own noise stream, so the same init state gives a different
# rollout each draw.
set -u
B=/home/jovyan/.cache/himoe-libero-bridge
LIBERO=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO
RC=/home/jovyan/work/himoe-vla/himoe-route-capture
OUT=/home/jovyan/work/himoe-vla/analysis_triggered_fork
export LD_LIBRARY_PATH="$B/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$LIBERO:/home/jovyan/work/himoe-vla/himoe-libero-bridge/src:$RC"

for spec in "7:20260830" "7:20260831" "0:20260831" "7:20260832" "0:20260832" "3:20260830"; do
  init=${spec%%:*}; seed=${spec##*:}
  tag="i${init}_s${seed}"
  echo "=== $tag ==="
  "$B/envs/libero/bin/python" "$RC/triggered_fork_collect.py" \
      --host 127.0.0.1 --port 9500 --benchmark libero_10 --task-id 8 \
      --init-state-id "$init" --worker-id 0 --k 8 --seed "$seed" \
      --libero-root "$LIBERO" --out "$OUT/$tag" 2>&1 | grep -E "ALARM|trunk:|fork q|RESCUE|no rescue|cannot answer|succeeded"
  m="$OUT/$tag/manifest.json"
  if [ -f "$m" ]; then
    n=$(python3 -c "import json;print(json.load(open('$m'))['triggered']['successes'])" 2>/dev/null || echo 0)
    echo "  >>> $tag 重采样成功 $n / 8"
    [ "${n:-0}" -gt 0 ] && { echo "★★★ 找到救回来的:$tag"; break; }
  fi
done
echo SWEEP_DONE
