# Rich functional event pilot protocol

Frozen before loading any event-level functional snapshot values on 2026-09-05.

## Question

Do functional HB-MoE quantities add a matched event/control difference at four or
two policy queries before the frozen static-onset proxy?

## Inputs and unit of analysis

- Use `rich_event_plan.json` without changing its selected episodes, controls,
  leads, or query indices.
- The matched `pair_id` is the statistical unit. Each comparison is event versus
  control within the same initial state and absolute policy-query index.
- Apply the restoration tolerances already frozen in
  `rich_event_inputs_audit.json`. Exclude a pair at a lead if either row fails;
  do not relax a tolerance after capture.
- Capture transparency is a hard validity gate. A recorder-on action must be
  bitwise identical to its recorder-disabled action for every captured row.
- No row is excluded based on a functional feature value or the rerun action's
  agreement with the historical action.

## Primary feature family

For scalar tensors, average over batch, the last four stored HB layers, all ten
denoise steps, and action tokens 1-10 (excluding the state token):

1. routed authority;
2. expert cancellation;
3. normalized expert disagreement;
4. routed/shared cosine;
5. relative adjacent-denoise change of the 32-dimensional routed-output sketch.

For feature 5, compute each adjacent-flow value as
`||s_f-s_(f-1)|| / (||s_f||+||s_(f-1)||+1e-12)` before aggregation.
The ten primary tests are these five features at leads -4 and -2. Directions are
not prespecified; all tests are two-sided.

## Inference

- Report the paired event-minus-control mean, median, standardized paired effect,
  event-greater fraction, and descriptive pooled AUC for every cell.
- Obtain raw two-sided p-values and family-wise maxT p-values from 100,000 joint
  matched-pair sign flips using seed 20260905. The same sign is applied to all
  leads and features belonging to one `pair_id` on each permutation.
- The primary claim threshold is maxT `p <= 0.05` across all ten cells.
- Report 95% paired bootstrap confidence intervals for mean differences using
  20,000 pair resamples and seed 20260906. These intervals are descriptive and do
  not replace maxT inference.

## Secondary outputs

Layer, token, denoise-step maps; route/function correlations; Top-4/5 boundary,
tail mass, execution entropy; and safe-prefix-like token profiles are descriptive
only. They cannot support the pilot's primary claim or a control intervention.

## Interpretation limits

This is a frozen-state observational pilot. A significant difference supports a
candidate precursor at the reconstructed states; it does not establish router
causality, recovery benefit, or safe-prefix effectiveness. Historical-action
reproduction is reported as an alignment diagnostic and never repaired by
post-hoc sample selection.
