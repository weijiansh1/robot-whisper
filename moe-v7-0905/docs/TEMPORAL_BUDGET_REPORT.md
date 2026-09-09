# Fixed temporal boundaries: a small change, not a substantial improvement

## Conclusion

The fixed early-strict/late-loose schedule changes external results from
419 TP / 92 FP to 420 TP / 91 FP. The reverse schedule produces 415 TP / 95 FP.
These are small changes, not evidence that global elapsed-time relaxation
solves the detection or early-warning problem. Keep the existing method and
the constant automatic calibration intact; no replacement is selected here.

The elapsed-time-only diagnostic is important: it detects every external
long-suite failure at query 37 or 38 (zero-based), without looking at MoE.
It has substantially worse precision and misses every failure in the three
shorter suites, but its aggregate early-8 recall exceeds all MoE variants.
Endpoint-based early recall alone therefore cannot establish MoE-specific
early-warning value on these data.

## What Was Fixed Before Outcomes

The protocol is `method/TEMPORAL_BUDGET_PROTOCOL.md`. The new seal timestamp
is `2026-09-06T14:33:20.892753+00:00`. Calibration uses all 16,000 unlabeled
historical main-plus-extra trajectories. No external features, outcomes,
task/suite identities, or published thresholds enter the new calibration.
Related outcomes were already known, so this is exploratory, not a blind test.

The three schedules retain original v7 features, baseline windows, smoothing,
persistence, and latched head composition. Their boundaries are
`b_h + sign * a_h * tau / (tau + q)`, with sign 0, +1, and -1, respectively.
The query q is elapsed time, never current query divided by final episode
length. The globally calibrated time scale is 6.5 queries, half the median
unlabeled historical length. There was no time-scale or amplitude sweep.

Shared amplitudes, from the IQR of historical smoothed-score peaks:

| Head | Amplitude |
| --- | ---: |
| Freeze | 0.2290258012 |
| Acceleration | 0.0747486223 |
| Periodicity | 0.4591891067 |

Offsets are subtracted from each smoothed observation before persistence.
Relaxing a later boundary cannot retroactively turn an earlier observation
into an exceedance against that later boundary. First alarms are actual
confirmation queries, not backdated window starts. Existing initialization
semantics are preserved and verified at every observed prefix.

Each schedule independently recalibrates its intercepts to the same observed
episode-alarm cap. All three, and the time-only diagnostic, alarm on exactly
480 / 16,000 reference episodes (3%). This is not a successful-episode FPR
guarantee and does not guarantee 3% on external episodes.

| Schedule | Freeze intercept | Acceleration intercept | Periodicity intercept |
| --- | ---: | ---: | ---: |
| Constant | 0.76633531 | 0.07495879 | 0.24855959 |
| Early strict | 0.71604180 | 0.05451363 | 0.16062361 |
| Early loose | 0.81419367 | 0.10059650 | 0.33038336 |

Periodicity normalization is unchanged at 0.0207076669. The constant profile
and every reference/main/external head's first alarm match the previous
automatic calibration exactly.

The time-only profile alarms at zero-based q=37 when its episode-start random
priority exceeds 0.595465861, otherwise q=38, provided the episode reaches
that query. The fixed seeds and saved priorities are in the protocol and
sealed artifacts. This tie rule uses neither identity nor outcomes. It is
a single randomized diagnostic, not a proposed MoE-only detector.

## External Results

The same 15,600 episodes contain 564 original-horizon failures and 15,036
timely successes. All values below are percentages except TP and FP.

| Detector | TP | FP | Precision | Recall | Success FPR | Alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Published v7 | 439 | 80 | 84.59 | 77.84 | 0.532 | 3.327 |
| Constant automatic | 419 | 92 | 82.00 | 74.29 | 0.612 | 3.276 |
| Early strict / late loose | 420 | 91 | 82.19 | 74.47 | 0.605 | 3.276 |
| Early loose / late strict | 415 | 95 | 81.37 | 73.58 | 0.632 | 3.269 |
| Time only | 274 | 213 | 56.26 | 48.58 | 1.417 | 3.122 |

Published v7 is a historical comparator with label-selected thresholds; it
does not meet the new outcome-free threshold-calibration requirement.

| Detector | Early-4 recall | Early-8 recall | Early-12 recall | Before-half TP |
| --- | ---: | ---: | ---: | ---: |
| Published v7 | 58.69 | 44.50 | 39.18 | 46 |
| Constant automatic | 56.21 | 45.57 | 39.36 | 47 |
| Early strict / late loose | 56.74 | 45.92 | 39.36 | 42 |
| Early loose / late strict | 56.03 | 45.57 | 39.72 | 50 |
| Time only | 48.58 | 48.58 | 48.58 | 0 |

Early-k means at least k queries before the recorded original endpoint,
divided by all 564 failures. It is not localization of physical failure onset.
The early-strict schedule adds three early-4 detections (317 to 320), two
early-8 detections (257 to 259), and no early-12 detections (222 to 222).

## Paired and Suite Checks

Against constant automatic calibration, early strict gains 3 TP, loses 2 TP,
adds 8 FP, and removes 9 FP. Of the 417 failures detected by both, 21 alarm
earlier, 27 later, and 369 at the same query. The small aggregate early-recall
increase is not a general improvement in per-episode alarm timing.

Early loose gains 4 TP, loses 8 TP, adds 17 FP, and removes 14 FP. Of 411
common detected failures, 26 alarm earlier, 36 later, and 349 are unchanged.

| External suite | Failures | Constant TP/FP | Early strict TP/FP | Early loose TP/FP | Time-only TP/FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Goal | 106 | 62 / 1 | 62 / 1 | 61 / 1 | 0 / 0 |
| Long | 274 | 237 / 85 | 239 / 84 | 233 / 88 | 274 / 213 |
| Object | 44 | 22 / 3 | 22 / 3 | 22 / 3 | 0 / 0 |
| Spatial | 140 | 98 / 3 | 97 / 3 | 99 / 3 | 0 / 0 |

The time-only diagnostic gets all its true and false alarms in the long
suite. Its result illustrates a duration/suite confound, not that MoE is
uninformative. MoE achieves much better precision and detects shorter-suite
failures that this global clock boundary cannot reach.

## Main Development Results

The same 14,800 main episodes contain 487 original-horizon failures.

| Detector | TP | FP | Precision | Recall | Early-4 recall | Early-8 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Published v7 | 382 | 67 | 85.08 | 78.44 | 55.85 | 42.51 |
| Constant automatic | 350 | 76 | 82.16 | 71.87 | 51.54 | 42.30 |
| Early strict / late loose | 351 | 77 | 82.01 | 72.07 | 51.95 | 42.92 |
| Early loose / late strict | 346 | 79 | 81.41 | 71.05 | 51.54 | 42.09 |
| Time only | 222 | 206 | 51.87 | 45.59 | 45.59 | 45.59 |

Main episodes belong to the reference corpus, so these are development
diagnostics, not independent generalization measurements.

## Interpretation

Budget recalibration compensates for much of a simple global temporal shift.
For example, freeze boundaries at q=20 are 0.76634 (constant), 0.77222 (early
strict), and 0.75802 (early loose). Despite larger startup differences, many
operating boundaries and resulting decisions remain close. This explains
part of the small observed change; it does not disprove all temporal methods.

Do not tune stronger slopes or switch to the observed best schedule on this
already inspected external set. A subsequent independently specified test
could condition relaxation on accumulating MoE evidence rather than elapsed
time alone. It would still need a clock-only comparison and matching workload
and false-alarm reporting, with a genuinely untouched evaluation cohort.

## Verification and Artifacts

- All source and prediction hashes in the new seal were verified (29 artifacts).
- Metrics were independently recomputed from saved per-episode predictions
  and labels across all 30,400 evaluated episodes and all five detectors.
- Raw CPU replay covered 22 external episodes and 736 unique queries, with
  66 episode/schedule replays. Every schedule included 14 alarming episodes.
  Samples cover alarm/no-alarm and disagreement groups without outcome labels.
- All raw feature/score tolerances, exact first alarms, and future-mutation
  prefix-invariance checks passed. Raw replay was completed before the new
  outcome evaluation.
- Tests cover budget replay, constant equivalence, padding, serialization,
  every-prefix decisions, persistence against historical boundaries, no
  backdating, label access prohibition, and tamper detection before evaluation.
  The complete bundle test suite passed: 93 tests in 9.43 seconds.
- No previous v7, ordinal-MDL, or constant automatic artifacts were replaced.

Implementation: `method/temporal_budget_guard.py`.
Evaluator: `experiments/evaluate_temporal_budget.py`.
Raw verifier: `experiments/verify_temporal_budget_raw.py`.
All predictions, profiles, calibration details, paired changes, bootstrap
intervals, suite/task metrics and raw audit are under
`results/temporal_budget_guard/`.

For a separate reproduction, use a new output directory; the seal stage
refuses to overwrite an existing directory:

```bash
python moe-v7-0905/experiments/evaluate_temporal_budget.py --output /tmp/temporal_budget_reproduction --stage seal
python moe-v7-0905/experiments/verify_temporal_budget_raw.py --output /tmp/temporal_budget_reproduction
python moe-v7-0905/experiments/evaluate_temporal_budget.py --output /tmp/temporal_budget_reproduction --stage evaluate
```

Runtime uses `TemporalProfile.load(...)` and `TemporalGuardMonitor(profile)`
with `update(hb_router_probs)` once per observed query. Temporal profiles
have a separate schema and are intentionally rejected by the old static
profile loader. Reported head scores are time-adjusted scores; compare them
with the saved intercepts, not with a second time-varying boundary.
