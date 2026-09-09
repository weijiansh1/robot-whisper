# Fixed temporal boundaries under an unlabeled episode-alarm budget

Status: exploratory follow-up. Related outcomes have already been inspected.
This is not a pristine holdout experiment. Freeze this protocol, implementation,
profiles and predictions before loading outcomes for the new variants.

## Scope

Keep the v7 continuous MoE features, initial baselines, trailing means,
confirmation counts, strict comparisons, latched conjunction and disjunction.
No outcome labels, task identity, successful-only selection, failure-model
training, or external features enter calibration. Use all 16,000 historical
main-plus-extra episodes. Default workload budget: 3% of entire episodes,
not 3% per query and not a successful-episode false-positive guarantee.

## Predeclared variants

For zero-based elapsed query q, set g(q) = tau / (tau + q). The one global tau
is half the median historical episode length, floored at one query. This uses
completed historical episodes only; a monitored episode's eventual length,
task-specific timeout and progress fraction are never inputs to its boundary.

For head h, amplitude a_h is the interquartile range of its historical
per-episode smoothed-score peaks, floored at the existing numerical epsilon.
Amplitudes are shared by all variants. There is no amplitude or time-scale
sweep. Independently calibrate intercepts b_h for these three schedules:

- constant: boundary_h(q) = b_h;
- early_strict: boundary_h(q) = b_h + a_h * g(q);
- early_loose: boundary_h(q) = b_h - a_h * g(q).

The latter two are respectively early-strict/late-loose and the reverse
within a trajectory. They need not be uniformly stricter/looser than constant
at any particular query after independent budget recalibration.

Subtract the time-dependent offset from each smoothed observation BEFORE
taking the persistence-window minimum. Thus historical observations are
compared against their own query's boundary, never against a newly relaxed
current boundary. Preserve v7 initialization: initial normalization can be
computed only once its baseline is available; no actual head decision occurs
before that availability. Recompute the prefix as the existing runtime does.
The first alarm is the actual confirmation query, never the window start.

Apply the existing unlabeled monotone branch-allowance calibration separately
to each variant's adjusted peaks. Replay all reference paths and verify their
union alarm count is at most floor(0.03 * 16000) = 480. The constant variant
must reproduce the previous automatic calibration and all its first alarms
exactly. None of the three schedules is selected using outcome metrics.

## Elapsed-time-only diagnostic

Calibrate a single global elapsed-query boundary from historical lengths.
To avoid an unfair zero-budget baseline when many trajectories have the same
length, break ties using independent uniform priorities assigned at episode
start, without task identity, MoE, or outcomes. Sort reference pairs
(last_query, priority); take order index N - capacity - 1. A trajectory alarms
at boundary_query if its priority exceeds the saved tie priority, otherwise
at boundary_query + 1, provided that query is actually observed. This yields
the exact observed reference cap when priorities are unique. It does not use
the current episode's eventual length to decide when to alarm.

Use NumPy PCG64 seed 2026090601 for reference priorities, reuse its main prefix
for the development replay, and seed 2026090602 for external priorities.
These seeds are fixed here, not compared or selected. Save priorities and
the boundary before outcomes. This randomized diagnostic is not a proposed
MoE-only detector. No out-of-reference alarm-rate guarantee is claimed.

## Evaluation and verification

Seal profile, reference counts, priorities, first alarms, metadata, source
hashes and feature-cache hashes before reading labels. Evaluate every fixed
variant, elapsed-time-only, and the already published v7 on the same aligned
main and external cohorts. Report TP, FP, precision, failure recall, success
FPR, actual alarm rate, and recall at least 4/8/12 queries before the recorded
original endpoint. Endpoint lead is not physical failure-onset localization.
Do not conceal negative results or retune after opening outcomes.

Check constant-profile equivalence, padding invariance, query-prefix
invariance, persistence under changing thresholds, no backdating, budget
replay and serialization. Replay raw external MoE episodes sampled without
outcomes from each schedule's alarm/no-alarm sets and their disagreement
sets. Confirm feature/score agreement, exact first alarms, and invariance to
changes confined to future router probabilities. Preserve older artifacts.
