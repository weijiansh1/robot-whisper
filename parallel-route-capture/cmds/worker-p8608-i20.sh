#!/bin/bash
# worker worker-p8608-i20: libero_10 task 9 init 20
cd /home/jovyan/work/himoe-vla/himoe-route-capture
exec env PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src \
  LD_LIBRARY_PATH=/home/jovyan/.cache/himoe-libero-bridge/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 \
  /home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python -u rolling_star_collect.py \
  --host 127.0.0.1 --port 8608 \
  --benchmark libero_10 --task-id 9 \
  --init-state-id 20 --worker-id 1 \
  --k 16 --seed 20260903 \
  --environment-seed 7 \
  --settle-steps 10 --max-steps 520 \
  --replan-steps 10 \
  --max-trunk-queries 6 \
  --stop-before-unix ${STOP_BEFORE_UNIX:-0} \
  --libero-root /home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO \
  --out /home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_long/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/rolling-star-k16-20260903/client/init-20
