#!/usr/bin/env bash
# Rebuild the global-prefix causal-Transformer audit on the corrected features.
#
# The archived results-global-prefix/ was trained from results-full40/full40_model.joblib,
# i.e. the pre-fix 2,187-D descriptors over the non-permutation-invariant layer feature.
# Dataset preparation loads the whole corpus and must wait for room in the shared 8 GiB
# cgroup; training is small enough to run one fold per GPU concurrently.
set -u
cd /home/jovyan/work/himoe-vla/MoE-grammar

LIMIT=$(cat /sys/fs/cgroup/memory.max)
NEEDED=$((4300 * 1024 * 1024))
GPUS=(1 2 3 4 5)
BATCH=${BATCH:-4096}

wait_for_room() {
  for _ in $(seq 1 240); do
    local free=$((LIMIT - $(cat /sys/fs/cgroup/memory.current)))
    if [ "$free" -ge "$NEEDED" ]; then
      echo "$(date +%H:%M:%S) free=$((free / 1024 / 1024))MB"
      return 0
    fi
    sleep 30
  done
  return 1
}

mkdir -p artifacts results-global-prefix-v2
for fold in 0 1 2 3 4; do
  out="artifacts/global-prefix-v2-fold$fold.npz"
  [ -f "$out" ] && { echo "dataset fold $fold exists"; continue; }
  for attempt in 1 2 3 4; do
    wait_for_room || break
    echo "$(date +%H:%M:%S) preparing fold $fold (attempt $attempt)"
    if python -u -m moe_grammar.prepare_global_prefix_dataset \
        --features-dir artifacts/features-full40-v2 \
        --grammar-model "results-full40-v2-fold$fold/full40_model.joblib" \
        --output "$out" > "logs/gp-prepare-fold$fold.log" 2>&1; then
      echo "$(date +%H:%M:%S) dataset fold $fold done"
      break
    fi
    echo "$(date +%H:%M:%S) dataset fold $fold failed; retrying"
    rm -f "$out"
    sleep 45
  done
done

echo "$(date +%H:%M:%S) launching ${#GPUS[@]} concurrent trainings (batch=$BATCH)"
for fold in 0 1 2 3 4; do
  gpu=${GPUS[$fold]}
  mkdir -p "results-global-prefix-v2/fold$fold"
  CUDA_VISIBLE_DEVICES=$gpu python -u -m moe_grammar.train_healthy_prefix_lm \
    --dataset "artifacts/global-prefix-v2-fold$fold.npz" \
    --output-dir "results-global-prefix-v2/fold$fold" \
    --device cuda:0 --batch-size "$BATCH" \
    > "logs/gp-train-fold$fold.log" 2>&1 &
done
wait
echo "$(date +%H:%M:%S) training finished"

python -u -m moe_grammar.evaluate_global_prefix \
  --datasets artifacts/global-prefix-v2-fold{0,1,2,3,4}.npz \
  --prediction-dirs results-global-prefix-v2/fold{0,1,2,3,4} \
  --output-dir results-global-prefix-v2 > logs/gp-evaluate.log 2>&1 \
  && echo "$(date +%H:%M:%S) evaluation done" || echo "evaluation failed"
echo GLOBAL_PREFIX_V2_FINISHED
