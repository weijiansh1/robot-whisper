#!/bin/bash
# route-recorder server: suite=long gpu=1 port=8609
# serve_with_recorder sets CUDA_VISIBLE_DEVICES itself from --gpu, overriding
# any outer env var — so the PHYSICAL gpu id must go through --gpu.
exec /home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python -u /home/jovyan/work/himoe-vla/himoe-route-capture/serve_with_recorder.py \
  --host 127.0.0.1 --port 8609 --gpu 1 --suite long \
  --checkpoint-dir /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-10 \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --libero-wrist-layout checkpoint-right \
  --store-full-probs \
  --out /home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/rolling-star-k16-20260903/server
