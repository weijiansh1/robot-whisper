# Train-free single-rollout route-derivative alarm

Status: frozen before derivative scores, thresholds, and alarms are generated.

## Question

Does explicitly combining the instantaneous MoE phenotype level with its first
and second derivatives across VLA queries improve strict single-rollout online
Trap alarms?

This is distinct from `route_acceleration`, which measures curvature across the
ten denoising flow steps inside one forward. Here derivatives run along the
rollout query axis and use only prior/current forwards.

## Runtime and reference contract

The runtime unit is one rollout. Query `q` may use only raw routing from queries
`<=q` and a frozen per-task profile. It cannot use state, actions, elapsed-time
risk, eventual length, timeout, outcome, simulator state, future routing, or
another live rollout.

Evaluation holds out one initial state and all eight of its flow-noise draws.
All 392 other-initial-state trajectories in the task enter the reference bank,
without filtering by outcome, duration, censoring, or physical behavior.
Feature percentiles and thresholds are task-specific and outcome-blind.

This is an exploratory follow-up designed after inspecting earlier results on
the same cohort, not independent confirmation.

## Instantaneous phenotype vector

At each query, compute the four three-query-smoothed route heads from
`ONLINE_MULTIHEAD_PROTOCOL.md` using directed same-task, same-query empirical
mid-rank percentiles:

```text
h(q) = [instability,
        lock_in,
        flat_narrow_support,
        feedback_decoupling]

level(q) = max(h(q))
```

At least 32 historical trajectories must have reached a query for it to be
scorable.

## Cross-query derivatives

With four heads, normalize Euclidean differences to the unit interval:

```text
velocity_raw(q) = ||h(q) - h(q-1)||_2 / 2

acceleration_raw(q) = ||h(q) - 2 h(q-1) + h(q-2)||_2 / 4
```

The divisors are the respective maximum Euclidean magnitudes for vectors in
`[0,1]^4`. Raw velocity and acceleration are converted to same-query empirical
mid-rank percentiles against the outcome-blind historical reference bank:

```text
velocity(q) = ECDF_q(velocity_raw(q))
acceleration(q) = ECDF_q(acceleration_raw(q))
```

For reproducible equality between batched replay and the scalar runtime, the
two raw derivative magnitudes are rounded to five decimal places before the
ECDF lookup. This is only a numerical contract; it does not change the causal
inputs or introduce a fitted parameter.

These are causal derivatives of the multi-head phenotype, not finite
differences of eventual trajectory scores.

## Frozen detectors

```text
instant_level(q) = level(q)

persistent_level(q) = min(level(q-1), level(q))

level_velocity(q) = sqrt(level(q) * velocity(q))

level_acceleration(q) = sqrt(level(q) * acceleration(q))

derivative_mean(q) = mean(level(q), velocity(q), acceleration(q))

derivative_fusion(q) = max(persistent_level(q),
                           level_velocity(q),
                           level_acceleration(q))
```

`derivative_fusion` is primary. Its winning branch is emitted as `persistent`,
`velocity`, or `acceleration`. All detectors begin at query 7 so comparisons
with the sealed single-rollout persistence experiment use the same warm-up.

## Calibration

For every fold and detector, take one maximum causal score per each of the 392
historical trajectories. Freeze the q90, q95, and q97.5 higher quantiles of
those maxima. `derivative_fusion` receives its own threshold; it does not inherit
the thresholds or false-alarm budgets of its three branches. A strict exceedance
triggers and latches the online alarm.

## Evaluation rule

Seal scores, thresholds, alarms, branch identities, profile, code hashes, and
`labels_used: []` before outcome reveal. Then report failure recall, success
false-positive rate, precision, early recall, lead, and physical-onset metrics.

Call derivatives useful only if q95 `derivative_fusion` improves failure recall
over `persistent_level` without increasing success FPR by more than one
percentage point, and the task-bootstrap 95% interval for recall improvement
excludes zero. Otherwise retain `persistent_level` as the deployment rule.

No intervention claim is allowed.
