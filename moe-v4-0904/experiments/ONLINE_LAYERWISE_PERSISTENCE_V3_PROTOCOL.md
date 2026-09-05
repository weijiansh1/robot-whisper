# Online MoE-only Layerwise Persistence Alarm (v3)

## Purpose

Detect deadline-risk rollouts online from one rollout's MoE router outputs.  No
robot state, action, image embedding, task progress, force, outcome, or horizon
counter is used at runtime.

The risk target is the original rollout failure: both persistent failures and
rollouts that succeed only after the `+10` query extension are positives.  A
success within the original task horizon is negative.

## Runtime signal

For each query, use `hb_router_probs` from all eight HB layers at final flow step
9 and action-token positions 1 through 10.  Normalize the 32 expert weights and
compute the mean tokenwise Hellinger distance from the previous query separately
for each layer:

```text
m_l(q) = mean_action Hellinger(route_l(q), route_l(q - 1))
```

Combine layers by the median, then smooth causally:

```text
m(q)  = median_l m_l(q)
u4(q) = mean(m(q - 3), ..., m(q))
```

The median is fixed before external evaluation.  It avoids both failure of one
layer being diluted by a mean and an isolated noisy layer controlling the alarm.

## Outcome-blind calibration

For each task, process a historical reference route cohort using the same
features.  For every reference trajectory, take the maximum instantaneous lock
score `max_q -u4(q)`.  The task threshold is the higher empirical quantile of
these trajectory maxima.  Outcomes are neither loaded nor filtered during this
calibration.

Two operating points are frozen:

- Primary: q75 threshold.
- Sensitivity analysis: q70 threshold.

For development replay, all eight noise seeds belonging to the evaluated initial
state are excluded from its 392-trajectory calibration set.  For external 8B
replay, thresholds come only from the earlier independent route cohort.

## Causal alarm

At query `q`, alarm if `u4` is below the frozen threshold for four consecutive
queries.  Equivalently, the persistent score
`-max(u4(q - 3), ..., u4(q))` must exceed the calibrated lock threshold.  The
alarm latches after its first trigger.

The first possible alarm is query 7: four mobility observations are needed for
`u4`, followed by four-point persistence.  No future query is used.

## Separation of selection and testing

The representation and hyperparameters were selected using the 14,800-rollout
development cohort.  This is detector design, not a claim of a fully
outcome-blind method-selection process.  The per-task deployment threshold itself
remains train-free and outcome-blind.

This protocol and its primary q75 operating point are frozen before reading the
external 8B outcomes.  The external result must be reported even if it is worse.
