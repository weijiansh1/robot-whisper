# Single-chunk MoE rollout trend: frozen exploratory plan

Frozen on 2026-08-27 before reconstructing routed contributions for control
queries `k=1..8`. Query-0 contribution results and query-0..4 router/hidden
screens were already inspected, so this is explicitly exploratory rather than
an untouched confirmation study.

## Question

Do eventually failed rollouts progressively diverge from successful rollouts
in the HB-MoE computation observed inside each successive action chunk?

The analysis compares population curves over query index. It never concatenates
features from different queries: the predictor at query `k` contains only the
ten flow-denoise rounds executed inside query `k`.

## Cohort and window

- Four outcome-variable LIBERO right-16x32 tasks are included.
- Every task has all 512 rollouts active through query `k=8`, so a fixed cohort
  is used for `k=0..8`.
- Query `k=9` is excluded because one spatial task has already terminated 221
  successful rollouts, which would create outcome-dependent survival bias.
- Final rollout failure is the label. It is not assumed to equal an annotated
  trap onset.

## MoE representations

For HB layers `{2, 5, 12, 15}`, recorded top-4 expert IDs and normalized
selected weights are combined with checkpoint expert MLPs:

`C = sum_e normalized_weight_e * expert_e(hidden)`.

State-token and mean action-token contributions are retained separately for all
ten denoise rounds. Direction is reduced with a fixed data-independent
CountSketch. Routed magnitude, expert disagreement/cancellation, expert norm
variation, routed/shared ratio, and routed/shared conflict are a separate
scalar block.

Matched controls are the shared-expert output and the pre-MoE input hidden
state, with the same layer, token-group, denoise, and projection capacity.

## Frozen primary curves

1. **Failure decodability.** At each `k`, cross-fit a fixed PCA-12 plus L2
   logistic model from routed contribution only. Hold out complete groups of
   eight noise seeds and calculate AUC only within the same initial state and
   held-out fold.
2. **Distance from the successful computation manifold.** At each `k` and
   fold, fit preprocessing on training seeds and form a successful centroid
   separately for every initial state. Score held-out rollouts by RMS distance
   to that centroid, then compute the same conditional AUC.
3. **Routed specificity.** Compare routed curves with matched hidden and shared
   curves, and compare `hidden + shared + routed` with `hidden + shared`.
4. **Longitudinal contrast.** The primary trend contrast is `k8 - k0`. A trend
   is called stable only if its scene-bootstrap 95% interval excludes zero and
   at least three of four task point contrasts have the same sign.

Leave-one-initial-state-out curves are secondary generalization checks.
Uncertainty resamples initial states inside task and averages tasks equally.

## Descriptive computation-state curves

For success and failure groups, report conditional standardized gaps over `k`
for routed RMS, expert cancellation, routed/shared conflict, within-query
direction path length, and d0-to-d9 commitment. These are descriptive and no
individual metric is promoted after inspection.

## Interpretation limits

- Later query states are descendants of earlier actions, not tree-sampled
  counterfactual snapshots. A growing curve is evidence of trajectory-level
  separation, not proof that the earlier MoE state caused the later state.
- Hidden/shared controls distinguish routed-specific signal from general model
  state separation, but do not remove all physical-state confounding.
- Offline vectors are reconstructed from fp16 hidden captures and are not
  runtime-exact. Recorded expert IDs and weights remain authoritative.
- A causal claim still requires snapshot-, observation-, and noise-matched
  activation intervention.
