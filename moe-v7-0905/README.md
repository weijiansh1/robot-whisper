# MoE Intrinsic Guard v7 - 2026-09-05

This directory is a standalone, task-agnostic and train-free iteration of
`moe-v4-0904`. It does not modify v4-v6 and does not duplicate raw Zarr data.

The monitor uses only intrinsic MoE routing dynamics. A rollout calibrates its
scale from its own early prefix, while three scalar thresholds come from one
pooled unlabeled reference corpus. No task ID, suite ID, task prototype,
per-task threshold, outcome, model fitting, or learned weight is used online.

The primary external result is **439 TP / 80 FP**: 77.84% risk recall,
58.69% early-4 recall, 84.59% precision, and 0.532% timely FPR. The median
lead among detected risks is 12 queries. This operating point improves over
the old task-prototype cold-start detector by +115 TP and -7 FP.

That result is measured on **tasks the calibration corpus already contains**:
all 39 external tasks are inside the 40-task reference. Under leave-one-suite-out
recalibration the guard reaches **73.40% recall and 78.26% precision** (micro,
414 TP / 115 FP), or 62.93% / 88.04% macro-averaged over the four folds. See
`docs/LOSO_VALIDATION_REPORT_ZH.md`. Both numbers stand; each must be quoted
with its condition.

See `REPORT_ZH.md` for the mechanism audit, limitations, comparisons, and raw
two-GPU replay. See `method/ONLINE_INTRINSIC_GUARD_V7_PROTOCOL.md` for the
frozen equations.

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python experiments/select_operating_point.py
python experiments/evaluate_intrinsic_guard_v7.py
CUDA_VISIBLE_DEVICES=6,7 python experiments/verify_raw_causal_gpu.py
pytest -q tests
```

The selection script audits a finite Boolean-rule grid against development
outcomes; it is not model training and never opens the external cohort. The
evaluator calibrates scalar threshold values from all 16,000 unlabeled
reference trajectories, seals external alarms, and only then opens outcome
files. The external cohort is not claimed as a pristine holdout because it was
already inspected during earlier v3-v6 work.
