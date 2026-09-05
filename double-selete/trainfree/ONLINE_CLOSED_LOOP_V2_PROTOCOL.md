# Train-free censor-aware closed-loop Trap alarm v2

Status: frozen before v2 per-query scores, thresholds, and outcome tables are
generated.

## Objective

Reduce alarms on trajectories that complete successfully while increasing
alarms on trajectories that eventually time out. The primary output is a
causal, latched online alarm. A typed MoE/physical phenotype is also emitted for
later recovery-policy work, but no intervention is evaluated here.

This is an exploratory follow-up to the v1 multi-head alarm. The 14,800-rollout
cohort and its v1 outcome results have already been inspected. In particular,
an audit established that every rollout ending before its suite query limit
completed and every rollout reaching that limit failed. This termination
structure motivates the censor-aware design below. It also means that this
experiment is train-free but not an independent, purely MoE validation.

## Cohort and online observables

- Source: the frozen 37-task `right-50x8-20260903` cohort under
  `VLA_MUI_HUB/cache_new/HiMoE-VLA`.
- Each task has 50 initial states and 8 flow-noise draws per state.
- Suite query limits are fixed by the collection configuration:
  `libero_spatial=22`, `libero_object=28`, `libero_goal=30`, and
  `libero_long=52`.
- At online query `q`, the detector may use routing tensors and robot state up
  through `q`, plus action chunks generated through `q`. The realized motion
  from `q-1` to `q` is aligned only with the already-issued chunk at `q-1`.
- `sim_state`, images, success, failure subtype, future queries, and the
  held-out rollout's eventual length are unavailable to an online update.

All evaluation uses leave-one-initial-state-out folds within task. The eight
rollouts of the held-out initial state are test rollouts. Historical references
are the 392 rollouts from the other 49 states.

## Censor-aware reference bank

A historical rollout ending before the configured query limit is an observed
completion. A historical rollout reaching the limit is treated as
administratively right-censored and is excluded from the healthy reference
bank. This rule reads termination time and the configured limit, never a
`success` field.

For a live rollout with elapsed count `e=q+1`, define completion-delay risk as

```text
completion_delay(q) = ECDF of completed reference durations evaluated at e
```

using mid-ranks for ties. A high value means that most comparable historical
rollouts had already completed. This is causal at deployment because the live
input is only the current elapsed count; completed/censored status is used only
when constructing the historical bank.

The exact equivalence between censoring and outcome in this cohort makes this a
strong operational baseline and a weak source of mechanistic evidence. Results
must therefore separate eventual detection from genuinely early detection.

## MoE heads

The twelve causal routing features and four typed heads are identical to
`ONLINE_MULTIHEAD_PROTOCOL.md`, except for two calibration changes:

1. directed empirical percentiles use only completed historical references;
2. percentiles use mid-ranks, so a tied or constant feature does not receive an
   artificial percentile of one.

The four three-query-smoothed heads are `instability`, `lock_in`,
`flat_narrow_support`, and `feedback_decoupling`. Define

```text
route_any = max(instability,
                lock_in,
                flat_narrow_support,
                feedback_decoupling)
```

The first eligible routing score remains query 4.

## Deployable state/action confirmation heads

Only `state` and `actions` from the client episode record are used. At query
`q`, define causal raw signals:

- `eef_motion_w3`: mean EEF-position displacement over transitions ending at
  `q-2:q`;
- `gripper_motion_w3`: mean L2 change of the two gripper-state coordinates over
  the same transitions;
- `prior_command_w3`: mean translational magnitude in action chunks that led to
  those transitions;
- `eef_reversal_w3`: fraction of consecutive EEF displacement pairs with cosine
  below -0.25 in the recent window; steps below 1 mm are ignored;
- `gripper_flip_w4`: fraction of sign changes among the mean gripper commands in
  the four previously issued chunks;
- `state_return_w4`: largest positive difference between distance to the
  immediately preceding state and distance to a state at lag 2--4. EEF
  positions are scaled by 5 cm and the two gripper coordinates by 4 cm before
  distance is computed.

Each raw signal becomes a same-task, same-query directed mid-rank percentile
against completed references that reached that query. At least 32 references
are required. The confirmation heads are

```text
motion_stall = mean(low eef_motion_w3,
                    high prior_command_w3)

physical_cycle = mean(high eef_reversal_w3,
                      high gripper_flip_w4,
                      high state_return_w4)
```

`gripper_motion_w3` is retained as an audit feature but is not included in
`motion_stall`, because many healthy manipulation phases intentionally keep the
gripper fixed.

## Two-stage detector

The typed mechanism confirmations are geometric means:

```text
loop_confirm = sqrt(instability * physical_cycle)

static_confirm = sqrt(max(lock_in, flat_narrow_support) * motion_stall)

feedback_confirm = sqrt(feedback_decoupling * motion_stall)

mechanism_any = max(loop_confirm,
                    static_confirm,
                    feedback_confirm)
```

The detectors are:

```text
completion_delay
route_only       = route_any
mechanism_only   = mechanism_any
delay_route      = max(completion_delay,
                       sqrt(completion_delay * route_any))
delay_confirmed  = max(completion_delay,
                       sqrt(completion_delay * mechanism_any))
```

`delay_confirmed` is primary. If a confirmation score is unavailable because
too few completed references reached a late query, it falls back to
`completion_delay`. All formulas and weights are fixed; no outcome label fits a
parameter.

At the first primary alarm, the largest of `loop_confirm`, `static_confirm`, and
`feedback_confirm` is emitted as a recovery phenotype when available. The type
does not affect whether the alarm fires.

## Threshold calibration and sealing

For each fold and detector, take the maximum causal score over every completed
historical reference trajectory. The q90, q95, and q97.5 higher quantiles of
these maxima are frozen thresholds. A strict threshold exceedance triggers a
latched alarm. The primary operating point is q95.

Before outcomes are loaded, write and hash:

- the feature cache;
- every per-query detector and typed-confirmation score;
- per-fold thresholds;
- per-query latched alarms;
- source and protocol hashes;
- an explicit list of fields used during scoring.

The scorer must assert that neither `success` nor any physical label was read.
It must separately disclose that reference termination times and configured
query limits were used to determine censoring.

## Evaluation

After sealing, reveal outcome labels and report for all three operating points:

- failure recall, success false-positive rate, and alarm precision;
- recall at least 2, 4, 8, and 12 queries before termination;
- median first-alarm lead among detected failures;
- the same metrics by suite and task;
- paired task-bootstrap differences versus v1 `multi_max`, v1 `clock`, and
  `completion_delay`.

The central MoE question is not whether `delay_confirmed` eventually detects
timeouts. That is structurally easy in this cohort. It is whether confirmation
increases lead over `completion_delay` at comparable realized success FPR.

## Interpretation rule

- `completion_delay` may be called an effective train-free operational alarm,
  but not a MoE Trap signature.
- MoE/state confirmation is useful only if `delay_confirmed` improves early
  recall or lead over `completion_delay` without increasing success FPR by more
  than one percentage point at q95.
- Because this protocol was designed after inspecting the main cohort, any
  positive v2 result is exploratory until frozen and tested on new tasks or new
  initial states.
- No claim about corrective intervention is permitted in this experiment.
