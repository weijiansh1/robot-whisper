# Direct Persistent-Score Calibration (v5 candidate)

## Scope

This candidate changes calibration and model selection, not the online route
representation.  The runtime heads remain the v4 all-layer-median lock head and
the sustained-high L5 instability head.  Runtime input is still only the current
rollout's `hb_router_probs`.

## Exact deployed statistics

For every query, both heads first compute the same four-query causal mean used by
v4.  The lock head orients low mobility as positive and requires four consecutive
threshold crossings.  The instability head orients high L5 mobility as positive
and requires eight consecutive crossings.

Unlike v4, calibration is performed on the exact deployed persistent statistic:

```text
trajectory calibration score = max_query persistent_score(query)
```

A trajectory too short to form the persistent statistic receives a score of
negative infinity.  It remains part of the empirical reference population because
it also cannot alarm online.

Development thresholds leave out all eight routes from the evaluated initial
state and use the other 392 same-task trajectories.  External thresholds use all
400 routes from the earlier same-task cohort.  Outcomes are never loaded or used
to construct a threshold.

## Two operating points

1. `max_reference_fwer` sets each head threshold to the maximum reference score.
   With strict threshold crossings, exchangeability gives a per-head false-alarm
   bound of at most `1 / (n_reference + 1)` and a two-head union bound of at most
   `2 / (n_reference + 1)`.
2. `lead_selected` scans a fixed empirical-quantile grid on development only.  A
   candidate is eligible when development timely-success FPR is at most 0.5%,
   every suite FPR is at most 1%, every task FPR is at most 2.5%, and precision is
   at least 75%.  Eligible candidates are ranked by recall with at least four
   queries of lead, then recall with at least eight queries of lead, total recall,
   worst-task recall, suite FPR, and overall FPR.

The selected quantile pair is frozen before its external first-alarm array is
joined to external outcomes.  External 8B remains a previously inspected replay,
not a pristine confirmation cohort.

## Reporting

Overall recall is secondary to `early4` and `early8` risk recall.  Results must
also include per-suite and per-task tables so aggregate FPR cannot hide a
long-horizon task concentration.
