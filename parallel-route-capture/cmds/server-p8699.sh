#!/bin/bash
# CALVIN-D route-recorder server: gpu=7 port=8699
exec env CUDA_VISIBLE_DEVICES=7 \
  PYTHONPATH=/home/jovyan/work/himoe-vla/himoe-calvin-alignment/src:/home/jovyan/work/himoe-libero-wrist-fix/src:/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src \
  /home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python -u /home/jovyan/work/himoe-vla/himoe-route-capture/serve_calvin_with_recorder.py \
  --checkpoint-dir /home/jovyan/.cache/himoe-calvin-alignment/checkpoints/HiMoE-VLA-CALVIN-D \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA --calvin-root /home/jovyan/.cache/himoe-calvin-alignment/upstream/calvin \
  --host 127.0.0.1 --port 8699 \
  --out /home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/calvin_d/task_D_D/routes-v2-20260903/server
