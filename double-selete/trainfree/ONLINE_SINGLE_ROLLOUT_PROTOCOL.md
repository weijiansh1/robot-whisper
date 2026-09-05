# Train-free single-rollout streaming Trap alarm

Status: frozen before the new streaming scores and thresholds are generated.

## Runtime contract

The deployment unit is one live rollout. At VLA query `q`, the monitor receives
only:

- the current HB router probabilities and selected expert IDs;
- the current 8-D robot state;
- the action chunk generated at `q`;
- a frozen task calibration profile built before deployment.

It retains a bounded prefix history and immediately emits `normal`, `suspect`,
or a latched `alarm` plus a phenotype. It never receives the rollout's eventual
length, success, timeout status, future query, simulator state, another live
rollout, or the other flow-noise draws for the same initial state.

Historical data are allowed only to freeze empirical feature normalizers and
alarm thresholds. Once that profile exists, runtime inference is strictly
single-rollout and streaming.

## Offline reference protocol

- Cohort: the 37-task, 14,800-rollout `right-50x8-20260903` cohort.
- Evaluation: leave one initial state and all eight of its draws out. The other
  392 trajectories in the same task form the historical profile.
- All 392 historical trajectories are retained. No filtering by success,
  failure, termination time, censoring, or physical behavior is permitted.
- At query `q`, a raw feature is compared only with historical trajectories
  that reached `q`. Query index is allowed for phase normalization, but elapsed
  duration is not itself a detector feature.
- Directed empirical percentiles use mid-ranks. At least 32 historical values
  are required.

This experiment follows earlier inspection of this cohort and is exploratory,
not independent confirmation.

## Causal instantaneous heads

The twelve MoE route features and four three-query-smoothed route heads are the
same as in `ONLINE_MULTIHEAD_PROTOCOL.md`:

- `instability`;
- `lock_in`;
- `flat_narrow_support`;
- `feedback_decoupling`.

The deployable physical features are the same as in
`ONLINE_CLOSED_LOOP_V2_PROTOCOL.md`. They form:

```text
motion_stall = mean(low eef_motion_w3,
                    high prior_command_w3)

physical_cycle = mean(high eef_reversal_w3,
                      high gripper_flip_w4,
                      high state_return_w4)
```

Neither head reads object `sim_state`.

## Streaming phenotype state machine

Let

```text
static_route(q) = max(lock_in(q), flat_narrow_support(q))

loop_physical(q) = sqrt(instability(q) * physical_cycle(q))
static_physical(q) = sqrt(static_route(q) * motion_stall(q))
feedback_physical(q) = sqrt(feedback_decoupling(q) * motion_stall(q))
```

The state machine distinguishes a one-query correction from sustained or
ordered evidence:

```text
loop_switch(q) = sqrt(max(instability(q-3:q-1)) * static_route(q))

loop_state(q) = max(loop_switch(q),
                    mean(loop_physical(q-2:q)))

static_state(q) = mean(static_physical(q-2:q))

feedback_state(q) = mean(feedback_physical(q-2:q))

phenotype_state(q) = max(loop_state(q),
                         static_state(q),
                         feedback_state(q))
```

Windows are trailing and never centered. A value is unavailable until its full
history exists. The candidate alarm evidence requires two consecutive high
phenotype states:

```text
single_rollout_multi(q) = min(phenotype_state(q-1),
                              phenotype_state(q))
```

This is the primary detector. Its winner at `q` is the largest current typed
state and is emitted as `loop`, `static`, or `feedback`.

The frozen ablations are:

```text
instant_route(q) = max(the four route heads at q)

persistent_route(q) = min(instant_route(q-1), instant_route(q))

multi_state(q) = phenotype_state(q)
```

The typed `loop_state`, `static_state`, and `feedback_state` are also evaluated
as individual alarms. No completion-delay or clock feature is present.

## Calibration and alarm semantics

For each held-out fold and detector, compute the detector causally on the 392
historical trajectories and take one maximum per trajectory. q90, q95, and
q97.5 higher quantiles of those outcome-blind maxima are frozen thresholds.
Composites are calibrated separately so adding heads does not receive a larger
nominal alarm budget.

At runtime:

- `normal`: primary evidence does not exceed its task threshold;
- `suspect`: `multi_state` exceeds its threshold but the two-query persistence
  condition has not yet fired;
- `alarm`: `single_rollout_multi` strictly exceeds its threshold; alarm and
  first phenotype are latched for the rest of the rollout.

The primary operating point is q95.

## Sealing and evaluation

Before reading outcomes, seal per-query scores, thresholds, alarms, first alarm
phenotypes, source/code/protocol hashes, and `labels_used: []`. The manifest
must explicitly assert that final length, censoring, and timeout are absent
from detector features and calibration filtering.

After sealing, evaluate all rollouts, not only alarmed ones:

- failure recall and success false-positive rate;
- precision;
- recall at least 2, 4, 8, and 12 queries before termination;
- median lead among detected failures;
- physical-onset precursor/timely/reaction recall where proxy onsets exist;
- paired task-bootstrap differences against `instant_route`,
  `persistent_route`, and v1 `multi_max`.

Call the state machine useful only if q95 improves failure and onset recall over
`instant_route` at no more than one percentage point additional success FPR.
Otherwise retain its typed output only as a diagnostic.

No intervention claim is allowed.
