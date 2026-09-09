# First-query MoE state impact

## Bottom line

There is a same-state MoE association, but not a confirmed routed-expert failure precursor. At recorded denoise position d0, routed-contribution geometry tracks the final action (rho 0.387) and the one-chunk simulator transition (rho 0.132). However, shared-output and pre-MoE hidden controls are stronger, so this is not routed-specific evidence.

The prespecified formation effect went in the opposite direction: routed action alignment changed by -0.078 from d0 to d8 and routed next-state alignment by -0.032. Contribution-only failure prediction remained at chance (d0 AUC 0.484; d8 AUC 0.506).

## Design

- The input is only query `k=0`; all 32 seed siblings have the exact same physical initial state.
- Routed top-4 expert contributions are reconstructed offline at layers 2/5/12/15 for every denoise position and token.
- LIBERO-Long is discovery. Goal-top and two spatial tasks are the frozen confirmation set.
- Pairwise geometry is computed only within initial state. Outcome models hold out complete seed groups.
- Shared-expert output and the expert-input hidden state are matched-capacity controls.

## Confirmation endpoints

| endpoint | estimate | 95% scene-bootstrap CI | positive tasks |
|---|---:|---:|---:|
| routed action-alignment slope d8-d0 | -0.078 | [-0.093, -0.061] | 0/3 |
| routed next-sim alignment slope d8-d0 | -0.032 | [-0.049, -0.015] | 1/3 |
| shared action-alignment slope d8-d0 | -0.086 | [-0.095, -0.078] | 0/3 |
| routed slope minus shared slope | +0.009 | [-0.010, +0.028] | 2/3 |

## Denoise curve on confirmation tasks

| d | routed-action rho | shared-action rho | routed-next-sim rho | routed-full failure AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.387 | 0.491 | 0.132 | 0.484 [0.398, 0.578] |
| 1 | 0.380 | 0.492 | 0.126 | 0.417 [0.334, 0.499] |
| 2 | 0.384 | 0.489 | 0.123 | 0.508 [0.408, 0.618] |
| 3 | 0.388 | 0.490 | 0.140 | 0.507 [0.397, 0.611] |
| 4 | 0.381 | 0.488 | 0.151 | 0.522 [0.444, 0.587] |
| 5 | 0.381 | 0.486 | 0.151 | 0.477 [0.373, 0.562] |
| 6 | 0.373 | 0.477 | 0.154 | 0.526 [0.434, 0.594] |
| 7 | 0.336 | 0.443 | 0.124 | 0.522 [0.426, 0.606] |
| 8 | 0.309 | 0.404 | 0.101 | 0.506 [0.421, 0.566] |
| 9 | 0.216 | 0.367 | 0.056 | 0.473 [0.365, 0.561] |

## Prespecified outcome readouts

| representation | d0 AUC | d8 AUC | d8 tasks above chance |
|---|---:|---:|---:|
| routed_identity | 0.460 [0.400, 0.526] | 0.399 [0.300, 0.480] | 0/3 |
| routed_scalar | 0.493 [0.404, 0.602] | 0.492 [0.407, 0.555] | 1/3 |
| routed_full | 0.484 [0.398, 0.578] | 0.506 [0.421, 0.566] | 1/3 |
| shared_identity | 0.507 [0.400, 0.605] | 0.477 [0.375, 0.579] | 1/3 |
| hidden_identity | 0.491 [0.388, 0.590] | 0.552 [0.452, 0.654] | 3/3 |

## Recovery proxy

`off-success-manifold` is defined from training-seed successful next states only. It is a descriptive subgroup, not `Q_escape`.

| task | adverse | recovered | compounded | physical-score AUC | routed-risk AUC within adverse | pairs |
|---|---:|---:|---:|---:|---:|---:|
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 154 | 134 | 20 | 0.422 | 0.381 | 21 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 193 | 103 | 90 | 0.498 | 0.431 | 123 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 178 | 174 | 4 | 0.449 | 0.400 | 5 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 149 | 138 | 11 | 0.492 | 0.231 | 13 |

## Interpretation boundary

The geometry endpoints establish whether routed computation is organized like the emitted action and resulting one-chunk state transition. They do not establish a causal expert effect: seed noise also changes non-MoE computation. A snapshot-and-noise-fixed activation intervention is still required for causality.

The vectors are recomputed from stored fp16 hidden states, recorded expert IDs/weights, and checkpoint MLPs; they are not runtime-exact. Recorded selected probabilities reproduce with maximum MAE 0.00014. The fp16 gate audit recovers at least 73.5% of top-4 sets; recorded IDs, rather than recomputed IDs, are used for every contribution.

The recovery subgroup is underpowered on confirmation tasks (only 5-21 within-stratum failure/success pairs), so its below-chance point estimates are not treated as evidence of an inverse effect.
