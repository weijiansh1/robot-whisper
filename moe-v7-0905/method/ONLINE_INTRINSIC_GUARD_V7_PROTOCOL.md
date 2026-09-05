# Online Intrinsic Routing Guard v7

## Contract

The runtime monitor receives only the current `hb_router_probs` tensor with
shape `[8, 10, 11, 32]`. It has no task or suite identity, task prototype,
outcome, horizon, simulator state, future query, peer rollout, or fitted model.

One global profile is calibrated from an unlabeled routing reference corpus.
The profile contains three scalar thresholds and one robust scale. It contains
no task-indexed array.

## Internal mechanisms

The rule separates two route-space mechanisms.

1. **Relative freeze.** Compute final-flow action-token Hellinger mobility for
   each HB layer. Divide each layer by its own mean mobility at q1..q4, take the
   median log contraction over back layers L12..L15, and apply a causal width-6
   mean. This tests whether routing energy collapses relative to the rollout's
   own initial routing regime.
2. **Confirmed turbulence.** Compute curvature of the soft expert route along
   the ten denoising-flow iterations. A high-curvature event must persist for
   eight queries. It is accepted only after a second head observes a persistent
   loss of long-lag route recurrence relative to lag 1. This rejects ordinary
   short replanning peaks.

The final alarm is:

```text
relative_freeze OR (persistent_flow_acceleration AND persistent_recurrence_loss)
```

The conjunction is prefix-causal: once each component has occurred, the
confirmed-turbulence branch latches. The first possible freeze alarm is q6 and
the first possible confirmed-turbulence alarm is q10.

## Fixed formulas

```text
freeze baseline       q1..q4, per-layer mean
freeze aggregation    median(L12..L15)
freeze smoothing      W6
freeze threshold       pooled unlabeled trajectory-peak q97.5

acceleration baseline q2..q7, median
acceleration smoothing W3
acceleration persistence K8
acceleration threshold pooled unlabeled trajectory-peak q70

recurrence baseline   q2..q6, median
recurrence smoothing  W6
recurrence persistence K4
recurrence threshold  pooled unlabeled trajectory-peak q65
```

All logarithms and distances are computed in float32. Thresholds use the
empirical `higher` order statistic. Every reference trajectory is retained;
outcomes are not read or filtered during calibration.

## Selection and evaluation

The formulas and this operating point were selected from a small, fixed grid of
interpretable Boolean rules using development outcomes. This is rule selection,
so v7 does not claim outcome-blind method discovery. Threshold *values* are
still empirical order statistics of the full unlabeled routing reference; no
outcome is used to fit a score. There are no learned weights, gradient updates,
classifiers, task centroids, or per-task cutoffs.

The external-8B cohort had already been inspected by v3-v6 and is not a pristine
holdout. The evaluator must nevertheless write and hash the global profile and
all external first-alarm arrays before it loads external outcomes.
