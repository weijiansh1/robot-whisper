# MoE v4 - 2026-09-04

This directory is the consolidated, reproducible snapshot of the train-free,
MoE-only dual-regime online alarm and its experiments.

The v5 iteration adds two optional task-conditional operating profiles. They do
not overwrite the frozen v4 baseline, and their external 8B replay is not a
pristine holdout.

## Headline results

| Cohort | Trajectories | Recall | Precision | Non-risk FPR |
|---|---:|---:|---:|---:|
| `cache_new/right-50x8` development | 14,800 | 76.39% | 84.74% | 0.468% |
| `cache_new/right-50x8b` external replay | 15,600 | 72.70% | 83.50% | 0.539% |
| `cache/right-16x32` frozen transfer | 2,560 | 74.59% | 94.63% | 0.577% |

The `cache_new` target treats persistent failures and `+10` late successes as
risk.  The older `cache/right-16x32` cohort has no continuation audit and uses
raw original-horizon failure.

## V5 operating profiles

| External 8B profile | Recall | Early-4 recall | Precision | Non-risk FPR |
|---|---:|---:|---:|---:|
| v4 baseline | 72.70% | 58.51% | 83.50% | 0.539% |
| long guard (high precision) | 69.33% | 55.67% | 92.00% | 0.226% |
| lead guard (earlier warning) | 71.28% | 61.35% | 84.63% | 0.486% |

See `docs/V5_ITERATION_REPORT_ZH.md` for selection constraints, per-suite
tradeoffs, the rejected persistent-score calibration, and validity limits.

## Layout

- `method/`: frozen protocol and single-rollout runtime monitor.
- `experiments/`: feature extraction, development sweep, v3 ablation, final v4
  evaluation, and frozen transfer evaluation.
- `results/development_sweep/`: all 13,440 development candidates and Pareto
  tables.
- `results/cache_new_v4/`: thresholds, first alarms, episode outputs, and metrics
  for the 14,800/15,600 cohorts.
- `results/cache16x32_v4/`: frozen-rule results for five `right-16x32` tasks.
- `results/layerwise_mobility/`: extracted eight-layer causal mobility tensors.
- `results/persistent_calibration_v5/`: rejected exact-statistic calibration.
- `results/long_guard_v5/`: high-precision task profiles and sealed replay.
- `results/lead_guard_v5/`: lead-aware task profiles and sealed replay.
- `analysis/`: normalized first-alarm timing analysis and per-task tables.
- `tests/`: runtime replay, fixed-metric, provenance, and timing checks.

## Alarm timing

Timing uses the post-hoc normalized phase

```text
100 * zero_based_first_alarm_query / (observed_rollout_length - 1)
```

This phase is only for analysis and is never an alarm input.  On external 8B,
the median first-alarm phase is 71.43%, with an interquartile range of
62.50%-86.21%.  See `analysis/ALARM_TIMING_REPORT_ZH.md` and
`analysis/results/alarm_timing_by_task_percent.csv`.

## Unknown-task cold start

`method/unknown_task_monitor.py` removes the runtime task-ID requirement. It
uses only the first four `hb_router_probs` snapshots to self-calibrate the lock
threshold and retrieve a routing-similar instability threshold. The strict
leave-one-task-out external result is 57.45% recall, 78.83% precision, and
0.579% non-risk FPR; lock-only gives 44.50% recall and 85.67% precision. See
`docs/UNKNOWN_TASK_COLD_START_REPORT_ZH.md` for the protocol and limitations.

## Source data

- `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache_new/HiMoE-VLA`
  - run `right-50x8-20260903`: development and threshold reference routes.
  - run `right-50x8b-20260903`: external route replay.
- `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA`
  - run `right-16x32`: five-task frozen transfer replay.
- Clean `+10` labels remain at
  `/home/jovyan/work/himoe-vla/double-selete/trainfree/results/timeout_extension_plus10`.

Raw Hub routes and the clean label source are not duplicated.  All method code,
derived route features, thresholds, first alarms, and reported tables are copied
into this directory.

## Verification

```bash
python moe-v4-0904/analysis/analyze_alarm_timing.py
pytest -q moe-v4-0904/tests
```

The copied final evaluators also use this directory's result layout by default:

```bash
python moe-v4-0904/experiments/evaluate_dual_regime_v4.py
python moe-v4-0904/experiments/evaluate_dual_regime_v4_cache16x32.py
python moe-v4-0904/experiments/evaluate_persistent_calibration_v5.py
python moe-v4-0904/experiments/evaluate_long_guard_v5.py
python moe-v4-0904/experiments/evaluate_lead_guard_v5.py
```

The first command reads the original clean label CSVs; the second reads the raw
Hub routes.  Neither command changes the frozen detector parameters.
