# Online MoE-only Dual-Regime Alarm (v4)

## Fixed heads

The detector is the causal OR of two train-free MoE routing heads.

### Lock-in head

This is the frozen v3 primary head:

- Mean tokenwise adjacent-query Hellinger mobility for each of all eight HB
  layers, using final flow step 9 and action tokens 1 through 10.
- Median across the eight layers.
- Four-query causal mean.
- Low-mobility threshold: task-specific q75 of reference-trajectory K1 maxima.
- Four consecutive threshold crossings.

### Instability head

This head uses the same route slice and mobility definition, but only HB layer
`L5`:

- Four-query causal mean of L5 mobility.
- High-mobility threshold: task-specific q80 of reference-trajectory K1 maxima.
- Eight consecutive threshold crossings.

The instability head is deliberately long-duration.  It represents sustained
route switching, not a single noisy spike.  The lock-in and instability heads
cover opposite routing regimes; no fitted weight combines them.

## Alarm semantics

The first alarm query is the earlier first trigger of the two heads.  Once either
head triggers, the alarm latches.  Both heads are prefix-only and operate on one
rollout.  Runtime inputs contain only MoE router probabilities; no outcome,
horizon, robot state, task progress, action, or cross-rollout state is used.

## Calibration and evaluation

Each threshold is an empirical order statistic from a historical same-task route
cohort, with outcomes neither loaded nor filtered.  Development replay excludes
all eight seeds of the held-out initial state.  External replay uses the earlier
independent route cohort.

The L5/q80/eight-confirmation instability head was selected on the 14,800-rollout
development set for incremental coverage over the frozen v3 q75 lock-in head.
This complete v4 rule is frozen before computing its external-8B alarms.  Its
external result must be reported without retuning.

The positive target remains original-horizon deadline risk, including both
persistent failures and `+10` late successes.  Timely successes are negatives.
