#!/bin/bash
# CALVIN client: fresh corpus, 200 sequences
cd /home/jovyan/work/himoe-vla/himoe-calvin-alignment
exec bash scripts/calvin.sh \
  --dataset-root /home/jovyan/.cache/himoe-calvin-alignment/datasets/task_D_D --calvin-root /home/jovyan/.cache/himoe-calvin-alignment/upstream/calvin \
  --himoe-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA --output-dir /home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA/calvin_d/task_D_D/routes-v2-20260903/client \
  --host 127.0.0.1 --port 8699 \
  --max-new-sequences 200
