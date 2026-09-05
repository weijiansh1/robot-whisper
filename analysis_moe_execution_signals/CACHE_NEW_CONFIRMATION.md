# Frozen cache_new confirmation protocol

Frozen on 2026-09-05 after inspecting corpora A/B and before calculating any
event label or execution feature for the two `cache_new` runs below.

## Untouched confirmation data

- `right-50x8-20260903`, flow seeds 1000-1007;
- `right-50x8b-20260903`, flow seeds 1008-1015;
- LIBERO-Long task 8, 50 initial states, 16 candidates per state, 800 complete
  episodes in total.

## Frozen primary family

Two scalar signals only:

1. cross-query Jaccard churn of the actual stored Top-4 support;
2. cross-query sparse Hellinger churn of actual normalized Top-4 execution
   weights.

Both use final denoise, HB layers 12-15, action tokens 1-10, followed by the
same trailing four-query mean as the A/B onset analysis.  Each signal is
rank-residualized within initial-state and absolute-query strata against the
two previously established F0 loop signals: late-flow full-softmax volatility
and full-softmax route acceleration.  No outcome label enters feature
extraction or residualization.

The frozen event cells are the Cartesian product:

```text
event:   aggregate Trap, static
lead:    -4, -2 policy queries
signal:  support churn, sparse execution-weight churn
```

The expected direction is lower churn in event episodes.  The report retains
direction-free AUC and uses one group-level sign-flip maxT family across all
eight cells.  A cell confirms only if its direction is event-low and its maxT
`p <= 0.05`.  The query-resolution physical onset is the same proxy already
validated on corpus A and used for corpus B; it is not relabeled as dense
ground truth.

Raw scores are descriptive.  No token position, alternate lead, loop endpoint,
new signal, threshold, or intervention result may be promoted from this holdout.
