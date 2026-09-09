# Full cached-state closure audit of the q0+8 result

## Design

- Task scope: long/SCENE8 only.
- Exact clean-444 landmark cohort from the fixed-prefix audit.
- Positive label: later stagnation-core or active-return failure; every labelled physical event is after the cut.
- Leave one initial state out; AUC counts only success/failure pairs inside the same held-out initial state.
- Every model imputer, scaler and PCA is fit on training initial states only. The landmark/goal reference is transductive and frozen before these folds.
- Hard expert IDs are excluded. State/action routing means fp16 soft gate probabilities.
- Practical effect threshold was fixed at +0.05 within-init AUC.

## Cohort

`n=444`, failures `158`, successes `286`, initial states `16`.

## Ladder

| model | within-init AUC | pooled AUC | feature components |
|---|---:|---:|---|
| legacy_M2 | 0.470 | 0.650 | legacy_physical, action |
| legacy_route_only | 0.637 | 0.699 | action_route |
| legacy_joint | 0.580 | 0.719 | legacy_physical, action, action_route |
| M0_basic_proprio | 0.804 | 0.818 | basic |
| M1_full_geometry | 0.762 | 0.902 | full_geometry |
| M2_full_geometry_action | 0.748 | 0.896 | full_geometry, action |
| M3_M2_state_route | 0.756 | 0.891 | full_geometry, action, state_route |
| M4_M2_action_route | 0.734 | 0.885 | full_geometry, action, action_route |
| M5_M2_hidden | 0.801 | 0.904 | full_geometry, action, hidden |

## Increment over M2

| added representation | delta AUC | 95% init-bootstrap CI | approximate MDE80 | point estimate >= +0.05 |
|---|---:|---:|---:|---|
| M3_M2_state_route | +0.008 | [-0.017, +0.025] | 0.031 | no |
| M4_M2_action_route | -0.014 | [-0.037, +0.013] | 0.037 | no |
| M5_M2_hidden | +0.053 | [+0.014, +0.085] | 0.053 | yes |

## Specification sensitivity

Across C=0.03/0.1/0.3 and geometry PCA widths 32/64/128/256, state-route deltas span [-0.003, +0.027] and action-route deltas span [-0.018, -0.003]. Hidden deltas span [+0.050, +0.060].

## Verdict

The action-soft-routing increment over the full cached physical state plus action is -0.014; the state-soft-routing increment is +0.008. Interpret these against their initial-state bootstrap intervals and the reported MDE, not against pooled AUC.

The legacy-like `route_only - physical+action` contrast is shown only as a diagnostic. It does not reproduce the published +0.237 protocol because hard-occupancy features, scaling and dimensionality differ. It is not an additive routing effect; the valid comparisons here are M3/M4 minus M2.

The hidden point estimate exceeds +0.05, but its 95% interval includes effects well below +0.05. This is a tentative nonzero signal, not evidence that the true increment reaches the practical threshold.

`hidden` is the router input, not an expert output contribution. The full physical block contains every recorded sim-state coordinate, but the cache still lacks RGB/contact/force and is not complete environment geometry. The global successful-terminal goal reference is retained to match the published cohort, so this is a transductive cohort audit rather than strict unseen-init evaluation.

Bootstrap intervals resample fixed OOF test-cluster scores; they do not refit the full training pipeline inside each bootstrap draw. MDE is therefore conditional on the fitted OOF models.
