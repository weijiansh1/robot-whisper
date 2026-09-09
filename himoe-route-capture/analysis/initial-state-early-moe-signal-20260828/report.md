# Initial-state prior versus early HiMoE signal

## Bottom line

- The initial-state prior is strong and reusable across held-out flow seeds.
  For any failure, it reduces Brier loss from
  0.1025 to
  0.0632, a
  38.3% reduction.  All six target
  intervals favor the initial-state prior.
- q0 routing does not show a stable increment beyond that prior.  For any
  failure the change is +0.0003, with a 95% clustered
  interval [-0.0007, 0.0014].  Targets
  with an interval wholly above zero: none.
- Routing also fails to improve prediction for complete unseen initial states;
  targets with an interval wholly above zero: none.
  Extending the fixed early window to q0:q2 does not rescue this result
  (no positive lower bounds).
- Therefore the defensible result is: the initial configuration is a strong
  early risk prior, but this readout provides no evidence that the first three
  MoE routing queries already identify the seed-specific eventual failure or
  transfer that risk to unseen geometry.

## Scope

The primary input is routing from the first policy query (`q0`), before its
emitted action chunk is executed.  The sensitivity input concatenates the fixed
first three queries (`q0:q2`); q1 and q2 may reflect new observations after
earlier chunks, but every rollout contributes exactly those same absolute query
indices.  No episode duration, remaining time, terminal window, action vector,
simulator trajectory, physical-failure annotation, or outcome-derived feature
is available to a predictor.

The five behavioral labels overlap and are each treated one-versus-rest.  A
task is evaluated for a label only when it contains at least
5 positives and 5 negatives:

- `any_failure`: 307/2048, 4 task(s)
- `stagnation`: 171/1536, 3 task(s)
- `regrasp_or_drop`: 67/2048, 4 task(s)
- `goal_regression`: 49/1024, 2 task(s)
- `residual_none`: 86/1536, 3 task(s)
- `active_return`: 39/512, 1 task(s)

## Experimental controls

1. `heldout_seed`: every fold leaves four complete flow seeds out of all 16
   initial states.  The training side still sees each test initial state.  This
   is the direct test of whether routing adds information beyond that state's
   empirical difficulty.
2. `heldout_init`: every fold leaves two complete initial states out, including
   all 32 flow seeds.  This asks whether routing transfers to unseen geometry.
3. Models are task-local.  Router descriptors are expert-permutation invariant;
   gate-input hidden summaries are kept separate.  Scaling and 12-component
   PCA are fit on the outer training fold only.  Each feature model is a
   regularized correction to the corresponding prior in log-odds space, so a
   zero correction reproduces its baseline exactly.  Regularization is fixed
   at `C=0.1`; `C=0.03` and `C=0.3` are saved as sensitivity checks.
4. The initial-state probability uses an 8-episode shrinkage prior.
   In `heldout_init` it must reduce exactly to the task prior, because the test
   state has no training outcomes.

## Main q0 results

All entries below are cross-fitted Brier loss; lower is better.

| target | positive/n | task prior | init prior | init+q0 routing | unseen-init q0 routing |
| --- | --- | --- | --- | --- | --- |
| any_failure | 307/2048 | 0.1025 | 0.0632 | 0.0629 | 0.1192 |
| stagnation | 171/1536 | 0.0850 | 0.0523 | 0.0530 | 0.1026 |
| regrasp_or_drop | 67/2048 | 0.0311 | 0.0232 | 0.0235 | 0.0355 |
| goal_regression | 49/1024 | 0.0450 | 0.0311 | 0.0314 | 0.0521 |
| residual_none | 86/1536 | 0.0509 | 0.0453 | 0.0462 | 0.0525 |
| active_return | 39/512 | 0.0705 | 0.0625 | 0.0627 | 0.0752 |

The next table expresses paired Brier improvement as `baseline - candidate`.
Positive is useful.  Intervals resample complete initial-state clusters
(5,000 stratified bootstrap draws); they quantify sampling
uncertainty, not causality.

| target | comparison | gain | relative | 95% cluster CI |
| --- | --- | --- | --- | --- |
| any_failure | initial_prior | 0.0393 | 0.3833 | [0.0230, 0.0614] |
| any_failure | router_beyond_init | 0.0003 | 0.0050 | [-0.0007, 0.0014] |
| any_failure | router_unseen_init | -0.0109 | -0.1010 | [-0.0269, 0.0037] |
| stagnation | initial_prior | 0.0327 | 0.3846 | [0.0162, 0.0551] |
| stagnation | router_beyond_init | -0.0007 | -0.0139 | [-0.0026, 0.0011] |
| stagnation | router_unseen_init | -0.0116 | -0.1276 | [-0.0238, -0.0000] |
| regrasp_or_drop | initial_prior | 0.0079 | 0.2527 | [0.0002, 0.0226] |
| regrasp_or_drop | router_beyond_init | -0.0003 | -0.0109 | [-0.0008, 0.0005] |
| regrasp_or_drop | router_unseen_init | -0.0033 | -0.1032 | [-0.0088, -0.0000] |
| goal_regression | initial_prior | 0.0139 | 0.3089 | [0.0010, 0.0386] |
| goal_regression | router_beyond_init | -0.0004 | -0.0113 | [-0.0016, 0.0011] |
| goal_regression | router_unseen_init | -0.0052 | -0.1115 | [-0.0148, 0.0005] |
| residual_none | initial_prior | 0.0056 | 0.1102 | [0.0022, 0.0104] |
| residual_none | router_beyond_init | -0.0009 | -0.0191 | [-0.0015, -0.0004] |
| residual_none | router_unseen_init | -0.0010 | -0.0199 | [-0.0029, 0.0006] |
| active_return | initial_prior | 0.0080 | 0.1130 | [0.0024, 0.0167] |
| active_return | router_beyond_init | -0.0001 | -0.0021 | [-0.0015, 0.0012] |
| active_return | router_unseen_init | -0.0036 | -0.0507 | [-0.0088, 0.0010] |

Interpret the three comparisons literally:

- `initial_prior`: how strong the reusable initial-state difficulty prior is
  when new flow seeds are run from a known state.
- `router_beyond_init`: whether q0 routing separates different seeds from the
  same initial state beyond that prior.
- `router_unseen_init`: whether q0 routing predicts risk for an entirely unseen
  initial state, relative to task prevalence alone.

## Fixed false-trigger operating point

The threshold is the 90th percentile of training-fold negative scores, applied
unchanged to held-out data.  This is included to show practical coverage and
the realized false-trigger rate; ties are triggered only when strictly above
the threshold.

| target | split | model | positive_coverage_at_train_negative_q90 | negative_trigger_rate_at_train_negative_q90 |
| --- | --- | --- | --- | --- |
| active_return | heldout_init | task_router | 0.0000 | 0.0127 |
| active_return | heldout_seed | task_init_router | 0.3846 | 0.1121 |
| any_failure | heldout_init | task_router | 0.0847 | 0.0787 |
| any_failure | heldout_seed | task_init_router | 0.5928 | 0.0936 |
| goal_regression | heldout_init | task_router | 0.0612 | 0.0390 |
| goal_regression | heldout_seed | task_init_router | 0.6122 | 0.1108 |
| regrasp_or_drop | heldout_init | task_router | 0.0597 | 0.0399 |
| regrasp_or_drop | heldout_seed | task_init_router | 0.4478 | 0.0914 |
| residual_none | heldout_init | task_router | 0.0116 | 0.0172 |
| residual_none | heldout_seed | task_init_router | 0.4535 | 0.0869 |
| stagnation | heldout_init | task_router | 0.0819 | 0.1048 |
| stagnation | heldout_seed | task_init_router | 0.6550 | 0.0886 |

## q0:q2 sensitivity

The three-query window is still absolute and early: exactly queries 0, 1, and
2 for every rollout, regardless of eventual duration.  It cannot see a final
window.

| target | comparison | window | improvement | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| active_return | router_beyond_init | q0 | -0.0001 | -0.0015 | 0.0012 |
| active_return | router_beyond_init | q0_to_q2 | -0.0001 | -0.0017 | 0.0016 |
| any_failure | router_beyond_init | q0 | 0.0003 | -0.0007 | 0.0014 |
| any_failure | router_beyond_init | q0_to_q2 | -0.0004 | -0.0016 | 0.0009 |
| goal_regression | router_beyond_init | q0 | -0.0004 | -0.0016 | 0.0011 |
| goal_regression | router_beyond_init | q0_to_q2 | -0.0002 | -0.0013 | 0.0010 |
| regrasp_or_drop | router_beyond_init | q0 | -0.0003 | -0.0008 | 0.0005 |
| regrasp_or_drop | router_beyond_init | q0_to_q2 | -0.0002 | -0.0007 | 0.0005 |
| residual_none | router_beyond_init | q0 | -0.0009 | -0.0015 | -0.0004 |
| residual_none | router_beyond_init | q0_to_q2 | -0.0011 | -0.0018 | -0.0005 |
| stagnation | router_beyond_init | q0 | -0.0007 | -0.0026 | 0.0011 |
| stagnation | router_beyond_init | q0_to_q2 | -0.0011 | -0.0026 | 0.0005 |
| active_return | router_unseen_init | q0 | -0.0036 | -0.0088 | 0.0010 |
| active_return | router_unseen_init | q0_to_q2 | -0.0044 | -0.0096 | -0.0006 |
| any_failure | router_unseen_init | q0 | -0.0109 | -0.0269 | 0.0037 |
| any_failure | router_unseen_init | q0_to_q2 | -0.0109 | -0.0297 | 0.0052 |
| goal_regression | router_unseen_init | q0 | -0.0052 | -0.0148 | 0.0005 |
| goal_regression | router_unseen_init | q0_to_q2 | -0.0020 | -0.0044 | 0.0001 |
| regrasp_or_drop | router_unseen_init | q0 | -0.0033 | -0.0088 | -0.0000 |
| regrasp_or_drop | router_unseen_init | q0_to_q2 | -0.0016 | -0.0029 | -0.0003 |
| residual_none | router_unseen_init | q0 | -0.0010 | -0.0029 | 0.0006 |
| residual_none | router_unseen_init | q0_to_q2 | -0.0037 | -0.0071 | -0.0003 |
| stagnation | router_unseen_init | q0 | -0.0116 | -0.0238 | -0.0000 |
| stagnation | router_unseen_init | q0_to_q2 | -0.0112 | -0.0264 | 0.0039 |


## 32-component capacity sensitivity

Increasing each feature block from 12 to 32 training-fold PCA components did
not uncover a hidden improvement: among the 24 prespecified
target/comparison/window cells, 0 had a positive point
estimate.  The q0 results are:

| target | comparison | improvement | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| active_return | router_beyond_init | -0.0051 | -0.0110 | -0.0005 |
| any_failure | router_beyond_init | -0.0009 | -0.0028 | 0.0010 |
| goal_regression | router_beyond_init | -0.0006 | -0.0016 | 0.0003 |
| regrasp_or_drop | router_beyond_init | -0.0002 | -0.0008 | 0.0006 |
| residual_none | router_beyond_init | -0.0032 | -0.0051 | -0.0014 |
| stagnation | router_beyond_init | -0.0024 | -0.0047 | -0.0001 |
| active_return | router_unseen_init | -0.0160 | -0.0321 | -0.0041 |
| any_failure | router_unseen_init | -0.0163 | -0.0358 | 0.0031 |
| goal_regression | router_unseen_init | -0.0046 | -0.0103 | -0.0006 |
| regrasp_or_drop | router_unseen_init | -0.0030 | -0.0063 | -0.0007 |
| residual_none | router_unseen_init | -0.0040 | -0.0098 | 0.0010 |
| stagnation | router_unseen_init | -0.0131 | -0.0347 | 0.0081 |

The complete high-capacity rerun, including predictions and independent
checksums, is in `../initial-state-early-moe-signal-pca32-20260828/`.


## Interpretation guardrails

- A strong `initial_prior` is an early risk signal, but it does not establish
  that a routing expert recognizes a future failure.  It can reflect geometry,
  task difficulty, or another stable property of that initial state.
- A positive `router_beyond_init` is the cleanest evidence here for seed-level
  early information, because outcome variation is compared after the initial
  state has already been identified statistically.
- A positive `router_unseen_init` is stronger evidence of transferable early
  structure.  Failure there, alongside success on known states, is consistent
  with initial-state memorization or a non-transferable geometry proxy.
- `hidden` is the input presented to a gate.  Hidden-only gains do not show
  expert selection or expert output contribution.  Those controls are retained
  in the machine-readable tables, but the main claim is based on routing.
- This corpus has one rollout per `(initial state, flow seed)`, not branched
  counterfactual continuations.  Statistical prediction cannot identify the
  intervention that would prevent a failure.

## Artifacts

- `metrics.csv`: Brier loss, log loss, calibration mean, and fixed-threshold
  coverage for every target/split/window/model/regularization setting.
- `paired_deltas.csv`: paired loss improvements and clustered uncertainty.
- `predictions.csv.gz`: every held-out score and threshold.
- `target_support.csv`: included and excluded task-label cells.
- `fallback_fits.csv`: model fits that reverted to their prior because the
  training fold had a single class.
- `brier_gain_q0.png`: visual summary of the three main comparisons.
- `summary.json` and `checksums.sha256`: audit metadata.

Fallback fits: 20.  They are all repetitions of one data split:
drawer-task stagnation in unseen-initial-state fold 5.  That fold holds the
only stagnation-positive drawer initial state out of training.  Removing the
drawer task leaves the pooled unseen-state stagnation point estimate negative.
