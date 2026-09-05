# Pure-MoE persistent-phenotype alarm: round-two report

## Bottom line

The result is useful but specialized. Adding a four-query persistent MoE lock
phenotype to the existing low-mobility cascade raises online risk recall in both
cohorts while keeping successful-rollout false alarms below 0.8% for warnings
and below 0.34% for hard alarms. Runtime uses only router probabilities and
expert IDs from the current rollout.

The gain is not yet general across trap types. Most additional detections come
from `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`, where the old
mobility detector missed a stable lock phenotype. Task-bootstrap intervals
therefore include zero for three of the four recall comparisons. This round
shows that persistent expert-support structure adds real information beyond
mobility, but not that a universal pure-MoE alarm has been obtained.

## Detector

The mobility branch is unchanged:

```text
warning: four-query route mobility is below its outcome-blind q95 threshold
         for two consecutive queries
hard:    the same condition lasts four consecutive queries
```

The new branch uses two equal-weight, same-task/same-query empirical-rank heads:

```text
lock_in = mean(low mobility, high recurrence,
               narrow top-4 union, low token disagreement)

flat_narrow_support = mean(high gate entropy, low top-1/top-2 margin,
                           narrow top-4 union)
```

If either same head stays above 0.80 for four consecutive queries, it triggers
the persistent static branch. The complete rule is:

```text
warning = mobility_K2 OR static_K4
hard    = mobility_K4 OR static_K4
```

All windows are trailing and causal. A single peak, a centered window, future
length, state/action values, simulator state, and success flags are absent from
the runtime path.

## Overall results

The risk target is every trajectory that failed to finish inside its original
horizon, including the 71 trajectories that succeeded during the ten-query
extension. Timely success is the only negative class.

| Cohort | Alarm | TP / risk | Timely FP | Risk recall | Timely FPR | Precision |
|---|---|---:|---:|---:|---:|---:|
| Main | mobility K2 | 234 / 487 | 88 / 14,313 | 48.05% | 0.615% | 72.67% |
| Main | **K2 OR static K4** | **276 / 487** | **108 / 14,313** | **56.67%** | **0.755%** | **71.88%** |
| External 8b | mobility K2 | 252 / 564 | 104 / 15,036 | 44.68% | 0.692% | 70.79% |
| External 8b | **K2 OR static K4** | **289 / 564** | **120 / 15,036** | **51.24%** | **0.798%** | **70.66%** |
| Main | mobility K4 | 177 / 487 | 21 / 14,313 | 36.34% | 0.147% | 89.39% |
| Main | **K4 OR static K4** | **228 / 487** | **46 / 14,313** | **46.82%** | **0.321%** | **83.21%** |
| External 8b | mobility K4 | 189 / 564 | 29 / 15,036 | 33.51% | 0.193% | 86.70% |
| External 8b | **K4 OR static K4** | **239 / 564** | **51 / 15,036** | **42.38%** | **0.339%** | **82.41%** |

For warning, recall improves by 8.62 and 6.56 percentage points while precision
changes by -0.80 and -0.13 points. For hard alarm, recall improves by 10.47 and
8.87 points; precision falls by 6.18 and 4.28 points but remains above the
predefined 80% floor.

## Late and persistent outcomes

| Cohort | Alarm | Late recall | Persistent recall |
|---|---|---:|---:|
| Main | mobility K2 | 46.67% | 48.14% |
| Main | K2 OR static K4 | **66.67%** | **56.02%** |
| External 8b | mobility K2 | 46.34% | 44.55% |
| External 8b | K2 OR static K4 | **53.66%** | **51.05%** |
| Main | mobility K4 | 36.67% | 36.32% |
| Main | K4 OR static K4 | **63.33%** | **45.73%** |
| External 8b | mobility K4 | 46.34% | 32.50% |
| External 8b | K4 OR static K4 | **53.66%** | **41.49%** |

Late-success recall rises rather than being hidden as a false positive. This is
consistent with the interpretation that those trajectories were genuinely at
deadline risk even though extra execution time rescued them.

## Alarm timing

The additional static alarms are not endpoint-only alarms. Their median lead
over original termination is 17 queries in both cohorts, and all 42/37 warning
incremental risk detections occur at least four queries before termination.

The combined warning median lead changes from 7 to 10 queries on main and from
8 to 12 on external. The combined hard-alarm median changes from 6 to 12.5 and
from 7 to 13 queries. Early-four-query risk recall becomes:

| Cohort | Mobility warning | Multihead warning | Mobility hard | Multihead hard |
|---|---:|---:|---:|---:|
| Main | 34.29% | **42.92%** | 24.44% | **34.70%** |
| External 8b | 34.22% | **41.13%** | 25.18% | **33.87%** |

## Concentration and limitation

The added warning recall occurs entirely in `libero_long`. For the hard alarm:

- main adds 51 risk detections, of which 48 (94.1%) are from the two-moka-pot
  task;
- external adds 50, of which 42 (84.0%) are from the same task;
- object tasks gain no TP and add five/two successful false alarms;
- spatial tasks are unchanged;
- goal tasks add one TP in each cohort.

The task-cluster bootstrap 95% interval for hard recall change is `[0, 21.5]`
percentage points on main and `[0.25, 17.35]` on external. Warning intervals
start at zero in both cohorts. The external flow-seed replication makes the
moka-pot result credible for that task, but it does not establish task-wide
generality.

## Interpretation

This result answers a narrower question positively:

```text
low route mobility alone misses some trajectories whose route is not globally
extreme, but whose expert support, recurrence, and token agreement jointly stay
in a persistent lock phenotype.
```

It also confirms why adding instantaneous derivatives directly was harmful:
healthy phase changes produce isolated routing spikes. Requiring four causal
queries of one coherent phenotype filters those spikes.

The next pure-MoE round should target failure families outside `libero_long`,
especially spatial/object cases, and should use a new run for independent
confirmation. Adding more OR branches on these already-inspected outcomes would
risk fitting false positives rather than learning a transferable trap signal.

## Artifacts and validity

- Runtime: `moe_only_cascade_monitor.py`
- Evaluator: `evaluate_moe_only_cascade_v2.py`
- Protocol: `ONLINE_MOE_ONLY_CASCADE_V2_PROTOCOL.md`
- Results: `results/online_moe_only_cascade_v2/`

The implementation is train-free at runtime and calibration is outcome-free.
The rule was nevertheless chosen after earlier outcome inspection, so both
cohorts are post-hoc evidence for this round. No intervention or causal success
improvement is claimed.
