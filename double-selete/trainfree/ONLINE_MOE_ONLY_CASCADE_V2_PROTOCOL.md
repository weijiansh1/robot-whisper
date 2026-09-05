# Pure-MoE persistent-phenotype cascade protocol

Status: post-hoc exploratory. The rule is fixed before the consolidated replay
implemented by `evaluate_moe_only_cascade_v2.py`, but outcomes from both route
cohorts were inspected during earlier experiments. Runtime scoring and every
reference statistic remain outcome-free; method selection is not outcome-free.

## Question

Can an online alarm recover failures missed by the low-route-mobility cascade
without losing its low false-positive rate, using only current and past MoE
routing from one rollout?

This is an alarm experiment. It does not rank trajectories, allocate extra
queries, read robot state, or execute an intervention.

## Outcomes used only for evaluation

The timeout-extension audit defines three mutually exclusive outcomes:

- `timely`: success inside the original task horizon;
- `late`: no success inside the horizon, but success within ten extra VLA
  queries;
- `persistent`: no success within that extension.

The online risk target is `late OR persistent`, because both reached the
original deadline without completing. A late success is a true risk alarm, not
a false alarm. Only `timely` trajectories are negatives.

## Runtime inputs

One update receives only the current HiMoE router probabilities and expert IDs:

```text
router_prob: [8 layers, 10 flow steps, 11 tokens, 32 experts]
expert_ids:  [8 layers, 10 flow steps, 11 tokens, 4 selected experts]
```

The monitor may retain its own routing history for the current rollout. It may
not read EEF pose, proprioception, issued actions, images, simulator state,
success flags, eventual length, future queries, or other live rollouts.

## Head 1: low-mobility evidence

Let `m(q)` be the adjacent-query Hellinger mobility of final-flow action-token
routing in HB layers 12--15, averaged over layers and action tokens. Define the
four-query trailing mean `u4(q)`.

For each task, the suspect threshold is the q95 higher empirical quantile of
the per-trajectory maximum of `-u4(q)` in an outcome-blind historical reference
bank. The same suspect threshold is shared by both persistence levels:

```text
mobility warning: -u4(q) exceeds the threshold for 2 consecutive queries
mobility hard:    -u4(q) exceeds the threshold for 4 consecutive queries
```

## Heads 2 and 3: persistent lock phenotype

Every raw component is converted at the same task and query index to an
empirical midrank using the outcome-blind historical reference bank. Component
weights are fixed and equal. The causal three-query trailing head values are:

```text
lock_in = mean(low route_mobility,
               high lag_recurrence,
               low top4_union,
               low token_disagreement)

flat_narrow_support = mean(high gate_entropy,
                           low top12_margin,
                           low top4_union)
```

A phenotype branch fires only if either *the same lock head* or *the same flat
support head* is above 0.80 for four consecutive queries. Alternation between
the two heads does not satisfy persistence. This combines instantaneous gate
and support information with cross-query mobility and recurrence, while
rejecting isolated peaks caused by healthy task-phase transitions.

## State machine

```text
warning = mobility_warning OR persistent_lock_phenotype
hard    = mobility_hard    OR persistent_lock_phenotype
```

Both outputs latch for the remainder of the rollout. `hard` is always a subset
of `warning`. The first triggering branch is emitted as `mobility`, `lock_in`,
`flat_narrow_support`, or a tie.

## Calibration and replay separation

- Development main: leave one initial-state ID and all eight of its flow draws
  out. The other 392 same-task trajectories form the outcome-blind reference.
- External 8b: use only the corresponding 400 trajectories from the older
  `right-50x8-20260903` run as the reference for
  `right-50x8b-20260903`.
- The head gate (0.80), persistence (4), mobility threshold source (K1 q95),
  and OR logic are shared across every task and cohort.

The cached route heads and mobility alarms were sealed without outcomes by the
prior online experiments. Outcome tables are joined only by the evaluator.

## Endpoints

Report separately for warning and hard alarm:

- risk recall and timely-success false-positive rate;
- precision among alarms;
- late and persistent recall;
- recall at least 2, 4, 8, and 12 queries before original termination;
- median lead time;
- branch composition and per-suite/per-task concentration;
- paired task-bootstrap changes relative to mobility-only K2/K4.

Call the new warning useful only if both cohorts have higher risk recall,
timely FPR at most 1%, and alarm precision no more than two percentage points
below mobility K2. Call the hard alarm useful only if both cohorts improve
recall by at least five percentage points, keep timely FPR at most 0.5%, and
retain at least 80% alarm precision.

No causal intervention or independent-confirmation claim is allowed.
