#!/bin/bash
# route-recorder server: suite=object gpu=6 port=8630
# serve_with_recorder sets CUDA_VISIBLE_DEVICES itself from --gpu, overriding
# any outer env var — so the PHYSICAL gpu id must go through --gpu.
exec /home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python -u /home/jovyan/work/himoe-vla/himoe-route-capture/serve_with_recorder.py \
  --host 127.0.0.1 --port 8630 --gpu 6 --suite object \
  --checkpoint-dir /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Object \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --libero-wrist-layout checkpoint-right \
  --store-full-probs \
  --out /home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket/rolling-star-k16-20260903/server
