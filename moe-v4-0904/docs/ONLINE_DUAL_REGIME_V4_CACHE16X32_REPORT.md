# Frozen v4 Transfer to VLA_MUI_HUB/cache

## Result

The frozen v4 alarm reproduces on the five complete LIBERO `right-16x32` runs in
`VLA_MUI_HUB/cache`.  These runs contain 2,560 trajectories: 16 initial states by
32 flow-noise seeds for each task.

| Detector | Failure recall | Precision | Success FPR | TP | FP | Median lead |
|---|---:|---:|---:|---:|---:|---:|
| Frozen legacy q95/K2 | 22.80% | 97.22% | 0.089% | 70 | 2 | 3 queries |
| Frozen v4 | **74.59%** | **94.63%** | **0.577%** | **229** | **13** | **15 queries** |

There are 307 original-horizon failures and 2,253 successes.  The v4 alarm adds
159 true detections at the cost of 11 additional false alarms.  Recall with at
least four queries of lead is 61.24%.

No v4 representation, quantile, confirmation count, or threshold was adjusted on
this cohort.  Both legacy and v4 thresholds were loaded from the earlier
`cache_new/right-50x8` task profiles.

## Per-task check

The four tasks containing failures have v4 recalls of 97.62%, 69.91%, 100%, and
67.57%.  The fifth task contains 512 successes and no failures; none are alarmed.
The moka-pot task supplies 216 of the cohort's 307 failures and 151 of the 229
detected failures, so aggregate estimates remain weighted toward that task.

## Interpretation limits

- `right-16x32` stores only original `success` outcomes.  It has no matching `+10`
  continuation audit, so slow successes remain counted as failures.  Its target
  is therefore not identical to the cleaned `cache_new` target.
- Only five normal LIBERO runs under `cache` have complete `right-16x32` data.
  Pin-on/off/base intervention runs and CALVIN data are different experimental
  domains and are not mixed into this result.
- This old cache has been used in earlier exploratory analyses.  It is an
  independent sampling grid for the frozen v4 rule, but not a pristine dataset
  never examined by the project.
- The L5 sustained-instability head fires zero times here.  The reproduced result
  is entirely due to the all-layer-median lock-in head.  Thus the lock-in signal
  transfers; the second phenotype does not transfer to this cohort.
