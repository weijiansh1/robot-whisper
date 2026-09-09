# v7 With Unlabeled Alarm-Budget Calibration

## Scope

Keep the existing `IntrinsicGuardMonitor` unchanged: raw features, per-layer
initial normalization, smoothing, persistence, latching, and Boolean mechanism.
Replace only the procedure that chooses the three global scalar thresholds.
Do not load the published thresholds in the new calibration procedure.

The user accepts an automatically calculated decision boundary from unlabeled
historical MoE. The only operating choice is a total episode-alarm budget,
default 0.03 (three alarms per 100 reference trajectories). This is a declared
workload ceiling, NOT a successful-episode false-positive rate or failure
probability. It is adjustable by the operator, not optimized using outcomes.

The author has already seen v7 and related outcomes. The inherited features and
mechanisms were previously developed with outcome feedback. This experiment
removes label-guided THRESHOLD selection, not the history of label-informed
method design. External_8b is not a pristine holdout.

## Calibration data

Use all 16,000 historical 7B reference trajectories (main plus extra), regardless
of their outcomes. Do not group, filter, or weight by task, suite, outcome, or
observed episode length. Validity masks only identify which queries exist.
Pad absent queries with NaN before any feature statistic. Keep v7's global
periodicity scale: the q75 absolute finite periodicity value, without outcomes.

Compute the original v7 score streams. For each reference trajectory retain:

- The maximum freeze score.
- The maximum smoothed acceleration and smoothed periodicity scores.
- The maximum persistent acceleration and persistent periodicity scores.

Calibrate on complete historical trajectories, then use frozen thresholds on
new prefixes. No future query of the monitored trajectory is an online input.
The full-trajectory maxima account for all crossing opportunities observed in
each reference rollout; they do not establish an out-of-distribution guarantee.

## Deterministic budget allocation

Let N be the number of reference trajectories and B = floor(N * alarm_budget).
There are TWO Boolean branches: freeze, and acceleration AND periodicity.
Use the same marginal reference alarm allowance m for these two branches.
Search m only over integers 0 through B, using no outcome data.

For each m:

1. Freeze: use the smallest observed freeze-peak order statistic whose strict
   exceedance count is at most m. Ties are never broken by episode identity.
2. Turbulence: use the SAME quantile level for its two smoothed-peak reference
   distributions. The available levels are j/N for j=0..N, with exact integer
   `higher` order-statistic indices. Choose the smallest j for which at most m
   reference trajectories have BOTH persistent peaks above their respective
   thresholds. Requiring both peaks is exactly equivalent to the original
   separately latched conjunction over a complete reference trajectory.
3. Count the actual UNION of the two reference alarm sets. Their overlap is
   observed directly; do not assume independent heads or branches.

Both alarm sets are nested as m increases. Binary search finds the largest m
whose union has at most B members. This uses the budget as fully as possible
within the fixed shared-allowance family; it does not globally optimize all
three threshold combinations. Ties may leave unused budget. No precision,
recall, F1, failure label, or label-derived constraint is an objective.

The turbulence quantile lookup is implemented exactly using integer rank
clearances, not a coarse numerical parameter grid. The calibrator returns a
standard four-scalar `GlobalIntrinsicProfile`, loadable by unchanged v7 code.
The deployment monitor never sees the reference corpus, identities, or budget
search. Numerical thresholds still exist, but no hand-selected per-head alarm
quantile remains in the new calibration procedure.

## What is guaranteed and what is not

The saved profile must trigger on at most B of these N reference trajectories.
Audit this by replaying the actual float32 v7 score comparisons after profile
serialization. This is a deterministic in-reference constraint, not a theorem
about new trajectories. Calibration data contain both successful and failed
episodes, so this budget is NOT a class-conditional false-positive guarantee.
The reference distribution and its horizon mixture still matter.

No fitted failure model, gradients, learned weights, task-specific cutoffs, or
label-guided budget search are introduced. Order-statistic calibration is still
data-dependent statistical estimation, which the user has now authorized.

## Fixed experiment and ablations

Freeze this protocol, implementation, budget, profile, and all alarm arrays
before loading any new outcome CSV in the evaluator. The primary detector is
always the fully automatic profile, independent of its eventual performance.
Also seal and report these diagnostic ablations, without selecting among them:

- Published v7, unchanged.
- Only the freeze threshold replaced; other thresholds are published v7.
- Only acceleration/periodicity thresholds replaced; freeze is published v7.
- All three thresholds automatically calibrated: the primary new detector.

Partial replacements still contain label-selected published thresholds and
must NOT be advertised as fully label-free calibration. Ablations isolate which
branch changes performance; they are not candidates for outcome-based selection.

Evaluate the same 14,800 development and 15,600 external episodes as v7, with
the same label alignment and risk definition. Development overlaps calibration;
only the external rows are outside the calibration corpus. Report actual total
alarm rates, TP/FP, precision, recall, per-suite results, and endpoint-relative
lead. Do not confuse a predicted earlier change location with alarm time, or
an endpoint lead with the onset of an irreversible physical failure.

Unit tests must verify budget limits, ties, row-order and padding invariance,
monotonicity, exact search against brute-force counts, profile round trips,
unchanged v7 scores, streaming causality, and no outcome access while sealing.
Raw replay must sample both alarming and non-alarming episodes without reading
their labels. Preserve existing v7 and ordinal-MDL files and results unchanged.
