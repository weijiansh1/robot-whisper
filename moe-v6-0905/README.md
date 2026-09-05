# MoE Spatial Guard v6 - 2026-09-05

This directory contains a separate GPU-searched optimization of
`moe-v4-0904`. It does not modify the frozen v4/v5 snapshot.

The optimization keeps the high-precision long-suite guard, restores v4 for
goal and object tasks, and searches an expanded spatial-only causal lock head.
See `method/ONLINE_SPATIAL_GUARD_V6_PROTOCOL.md` for the frozen selection
constraints and validity limits.

## Headline result

| External 8B profile | TP / FP | Recall | Early-4 recall | Precision | Non-risk FPR |
|---|---:|---:|---:|---:|---:|
| v4 baseline | 410 / 81 | 72.70% | 58.51% | 83.50% | 0.539% |
| v5 lead guard | 402 / 73 | 71.28% | 61.35% | 84.63% | 0.486% |
| **v6 spatial guard** | **408 / 45** | **72.34%** | **62.77%** | **90.07%** | **0.299%** |

The selected spatial lock is `back_median`, low mobility, width 1, two
confirmations, q80. Long tasks retain v5 `L12/W4/K8/q70`; goal and object
tasks retain v4 `all_median/W4/K4/q75`. The L5 instability head is unchanged.

External 8B is not a pristine holdout. These numbers establish deterministic
replay and a better post-hoc operating point, not confirmed generalization.

See `docs/SPATIAL_GUARD_V6_REPORT_ZH.md` for the full comparison and caveats.

## Layout

- `experiments/`: two-GPU development sweep and frozen external evaluator.
- `method/`: frozen protocol and single-rollout runtime monitor.
- `results/spatial_guard_v6/`: all 51,200 candidates, CPU shortlist, sealed
  alarms, deployment profiles, metrics, and provenance hashes.
- `tests/`: profile, metric, provenance, and raw-Zarr runtime replay checks.

## Run

```bash
cd /home/jovyan/work/himoe-vla/moe-v6-0905
CUDA_VISIBLE_DEVICES=6,7 \
  python experiments/evaluate_spatial_guard_v6_gpu.py
pytest -q tests
```

The evaluator uses both visible GPUs for the development-only sweep. External
labels are loaded only after the selected rule, deployment profile, and first
alarms have been written.

The verified environment is Python 3.13.12, CUDA 12.4, and the exact package
versions in `requirements.txt`. The CUDA-tagged PyTorch wheel may require the
matching PyTorch package index when rebuilding the environment.
