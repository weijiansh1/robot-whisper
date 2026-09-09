# MoE state impact: pilot-informed confirmation plan

Frozen on 2026-08-26 after inspecting only the existing LIBERO-Long t08
contribution cache.  Goal-top and the two outcome-variable spatial tasks are
the confirmation set.  Results are appended elsewhere; this file is not
edited after confirmation features are computed.

## Question

Does the routed HB expert contribution inside the first policy query form an
action commitment that is propagated to the physical state reached after the
10-action chunk?

The environment never reads an activation directly.  The proposed pathway is

`routed contribution -> emitted action chunk -> next physical state -> outcome`.

Action and next state are therefore mediators/readouts, not nuisance variables
to remove from the primary test.

## Data and scope

- Four outcome-variable 16-initial-state x 32-noise-seed tasks are reported.
- LIBERO-Long t08 is discovery because its contribution cache and a geometry
  pilot were inspected before this document.
- Goal-top, spatial-ramekin, and spatial-stove are untouched confirmation tasks.
- Only control query `k=0` is used.  Every seed sibling has exactly the same
  proprioceptive and simulator state within an initial-state stratum.
- Internal coordinates are four representative HB layers `{2, 5, 12, 15}`,
  all 10 flow-denoise positions, and all 11 suffix tokens.

## Representations

Primary state:

`routed = sum_e normalized_top4_weight_e * expert_e(hidden)`.

Matched controls are the shared-expert output and the HB expert-input hidden
state.  Each 1024-vector is direction-normalized and projected with the same
data-independent per-token CountSketch.  Routed magnitude, disagreement,
cancellation, norm CV, routed/shared ratio, and routed/shared conflict form a
separate scalar block.

## Frozen confirmation endpoints

1. **Action commitment slope.**  Within each initial state, correlate pairwise
   routed distance with pairwise normalized final-action distance.  Primary
   contrast: Spearman rho at denoise `d=8` minus `d=0`.
2. **One-chunk state-impact slope.**  Repeat endpoint 1 with the simulator
   transition `sim_state[1] - sim_state[0]`, excluding the time coordinate and
   scaling nonconstant coordinates within task.  Primary contrast: `d8-d0`.
3. **Routed versus shared formation.**  Compare the routed action-commitment
   slope to the matched shared-output slope.
4. **Outcome association.**  Contribution-only failure prediction is evaluated
   at the prespecified `d=0` and `d=8`.  Models use four whole-seed holdouts,
   train-only PCA-12 per block, fixed L2 logistic `C=0.1`, scene fixed effects,
   and AUC pairs restricted to the same scene and held-out fold.

All ten denoise positions and hidden/shared controls are descriptive curves.
Uncertainty resamples initial states within each confirmation task and averages
tasks equally.  A confirmation endpoint is called positive only if its paired
95% bootstrap interval excludes zero (or 0.5 for an AUC) and at least two of
three confirmation task point estimates have the predicted sign.

## Exploratory recovery proxy

Using only training seeds, define a scene-specific successful next-state
centroid and its 75th-percentile successful distance.  A held-out rollout past
that threshold is `off-success-manifold`.  Among those held-out rollouts,
success is called recovery and failure is called compounding.  Report whether
the cross-fitted `d8` routed risk score separates them.  This is not an estimate
of `Q_escape`, because every later state has only one recorded continuation.

## Interpretation limits

- Geometry and outcome decoding are observational total associations.
- Different noise seeds also alter non-MoE computation, so only an activation
  intervention with snapshot and noise fixed can establish causal influence.
- Final failure is not assumed to equal a trap in all four tasks.
