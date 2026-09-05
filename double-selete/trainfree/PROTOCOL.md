# Train-free two-head selector protocol

Status: frozen before the first label-aware evaluation on 2026-09-03.

## Question

Can two mechanism-specific, train-free routing heads improve Trap trajectory
selection and provide separate loop/static intervention targets?

This experiment is **train-free**, not discovery-free. The signal directions
were fixed from the preceding mechanism analysis on this task. No model is fit,
no coefficient or threshold is calibrated, and Trap labels are unavailable to
the scoring program. Consequently, results on corpus B are descriptive and a
new task is required for confirmatory generalization.

## Data available to the scorer

- Corpus: VLA_MUI_HUB moka-pot `right-16x32`, 512 trajectories in 16 pools of
  32 action candidates.
- Inputs: MoE routing probabilities/selected IDs at or before query `q`, the
  candidate's pool ID, and validity/length.
- Forbidden: success, physical state, actions, loop/static/trap labels, event
  onset, learned parameters, label-selected signs, and calibrated thresholds.
- Every component is converted to a percentile rank among currently valid
  candidates in the same 32-candidate initial-state pool. Ties receive their
  average rank. All components then have a fixed `[0, 1]` scale without fitting.

## Frozen probability-only heads (primary)

Let `R+(x)` be the within-pool percentile rank of `x` and `R-(x)=R+(-x)`.
All temporal windows are trailing and causal.

Loop precursor subhead:

```text
mean(
  R+(2-query change in gate concentration),
  R+(4-query mean late-flow volatility),
  R+(4-query mean route acceleration)
)
```

Loop lock-in subhead:

```text
mean(
  R+(2-query change in deep-layer soft cross-token consensus),
  R+(2-query change in deep-layer aggregate-distribution concentration)
)
```

The loop head is the maximum of the precursor and lock-in subheads. This fixed
OR reflects two temporally adjacent loop phases rather than averaging away a
short precursor once lock-in starts.

Static head:

```text
mean(
  R-(4-query mean gate concentration),
  R-(2-query change in gate concentration),
  R+(L15 d9 aggregate-distribution concentration),
  R+(L15 d9 soft cross-token consensus),
  R+(L15 concentration at d9 minus d8)
)
```

The train-free double score is `max(loop_head, static_head)`. An information-
matched one-head control is `(loop_head + static_head) / 2`; it uses exactly
the same component scores but collapses the incompatible modes before ranking.
The established generic train-free control is low 4-query route mobility.

## Hard-support diagnostic

A parallel diagnostic replaces the soft lock-in/support terms with recorded
Top-4 support concentration, maximum token occupancy, and inverse support size.
It is secondary because fourth/fifth-expert ties in fp16 can change hard IDs.

## Frozen evaluation

- Primary snapshot: query 34, the last query observed for all 512 candidates.
- Sensitivity snapshot: query 30.
- Independent-head budget: loop Top-8 and static Top-8 in each pool. Their
  natural union is compared with each one-head control at exactly the same
  per-pool union size.
- Budget sensitivities: Top-4 and Top-16 per head.
- Head-specific comparisons: loop-head loop recall and static-head static
  recall, each against a one-head control at the same K.
- Ranking metrics: group-balanced AUC for loop, static, and their union.
- Event alignment: loop head at loop onset minus two queries; static head at
  static onset. Each event is compared only with same-pool, same-query
  trajectories without that event type. Onsets choose evaluation rows only.
- Uncertainty: paired bootstrap over the 16 initial-state pools, 20,000 draws,
  fixed seed 20260903.

No formula, direction, weight, query, or primary metric may be changed after
labels are opened. Any later alternatives must be marked post-hoc.
