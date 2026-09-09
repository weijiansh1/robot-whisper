# Success/failure MoE trend across single chunks

## Bottom line

Each point uses only one query's internal ten-denoise computation. There is no robust k0 precursor. At k1, routed direction alone has a weak same-state signal (AUC 0.572, 95% CI [0.503, 0.648], 4/4 tasks above chance), but routed-full remains at 0.490 and the direction signal is 0.499 under unseen-initial-state evaluation.

The primary routed-full separation appears at k2/k3, disappears at k4, and becomes consistently strong only from k6 through k8 (0.668, 0.678, 0.713).

Within the same initial states, routed-full AUC changes from 0.463 at k0 to 0.713 at k8 (delta +0.251, 95% CI [+0.145, +0.339], 4/4 task contrasts positive).

Distance from the successful routed-computation manifold changes from 0.502 to 0.725 (delta +0.223, 95% CI [+0.157, +0.304], 4/4 tasks positive).

This is not a routed-expert-specific effect: at k8 hidden and base AUC are 0.699 and 0.726, while adding routed features to hidden+shared changes AUC from 0.726 to 0.716. The secondary unseen-initial-state test is also not stable for routed-full (k0 0.496, k8 0.575; delta +0.078, 95% CI [-0.014, +0.160], 1/4 task contrasts positive).

## Scope

- Fixed 2048-rollout cohort: four tasks, 16 initial states x 32 seeds per task.
- Queries k0..k8 are evaluated separately; no predictor reads another chunk.
- All rollouts are active through k8. k9 is excluded to avoid success-dependent termination.
- Label: eventual failure, not annotated trap onset.
- Primary evaluation holds out complete seed groups and compares scores only within the same initial state and fold.
- Routed contributions use recorded top-4 IDs/weights and checkpoint expert outputs at layers 2/5/12/15.

## Primary trends (k8 minus k0)

| readout | representation | delta | 95% CI | positive tasks |
|---|---|---:|---:|---:|
| failure classifier | routed_full | +0.251 | [+0.145, +0.339] | 4/4 |
| failure classifier | hidden_identity | +0.147 | [+0.036, +0.243] | 2/4 |
| failure classifier | shared_identity | +0.193 | [+0.060, +0.301] | 3/4 |
| failure classifier | base | +0.196 | [+0.074, +0.304] | 3/4 |
| failure classifier | base_routed | +0.186 | [+0.072, +0.281] | 4/4 |
| success-manifold distance | routed_full | +0.223 | [+0.157, +0.304] | 4/4 |
| success-manifold distance | hidden_identity | +0.195 | [+0.122, +0.271] | 3/4 |
| success-manifold distance | shared_identity | +0.183 | [+0.101, +0.272] | 2/4 |
| success-manifold distance | base | +0.193 | [+0.112, +0.281] | 3/4 |
| success-manifold distance | base_routed | +0.200 | [+0.113, +0.292] | 3/4 |

## Single-chunk curves

| k | routed direction | routed full | hidden | base | base+routed | routed manifold | hidden manifold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.548 | 0.463 | 0.553 | 0.530 | 0.530 | 0.502 | 0.501 |
| 1 | 0.572 | 0.490 | 0.443 | 0.431 | 0.464 | 0.487 | 0.477 |
| 2 | 0.554 | 0.572 | 0.534 | 0.549 | 0.566 | 0.503 | 0.531 |
| 3 | 0.538 | 0.595 | 0.524 | 0.541 | 0.582 | 0.493 | 0.511 |
| 4 | 0.498 | 0.498 | 0.469 | 0.495 | 0.474 | 0.526 | 0.603 |
| 5 | 0.565 | 0.545 | 0.537 | 0.541 | 0.532 | 0.559 | 0.566 |
| 6 | 0.645 | 0.668 | 0.654 | 0.686 | 0.713 | 0.654 | 0.679 |
| 7 | 0.696 | 0.678 | 0.697 | 0.703 | 0.704 | 0.697 | 0.735 |
| 8 | 0.712 | 0.713 | 0.699 | 0.726 | 0.716 | 0.725 | 0.696 |

## Secondary unseen-state check

This classifier is trained on 15 initial states and evaluated on the held-out state. It is a harder generalization test than the repeated-rollout comparison above.

| k | routed direction | routed full | routed scalar | hidden | base | base+routed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.528 | 0.496 | 0.478 | 0.510 | 0.488 | 0.457 |
| 1 | 0.499 | 0.467 | 0.452 | 0.404 | 0.378 | 0.426 |
| 2 | 0.481 | 0.481 | 0.494 | 0.501 | 0.495 | 0.519 |
| 3 | 0.502 | 0.523 | 0.538 | 0.452 | 0.440 | 0.471 |
| 4 | 0.472 | 0.505 | 0.525 | 0.430 | 0.426 | 0.476 |
| 5 | 0.445 | 0.435 | 0.474 | 0.423 | 0.407 | 0.467 |
| 6 | 0.628 | 0.575 | 0.503 | 0.577 | 0.636 | 0.609 |
| 7 | 0.620 | 0.666 | 0.656 | 0.612 | 0.613 | 0.638 |
| 8 | 0.573 | 0.575 | 0.634 | 0.600 | 0.638 | 0.620 |

## Direct success/failure computation gaps

Positive values mean the metric is higher in failed rollouts after task-wide standardization.

| metric | k0 gap | k4 gap | k8 gap | k8-k0 (95% CI) |
|---|---:|---:|---:|---:|
| routed_rms | -0.024 | +0.054 | +0.074 | +0.097 [-0.072, +0.272] |
| cancellation | +0.060 | +0.017 | -0.101 | -0.162 [-0.323, +0.006] |
| shared_conflict | +0.030 | +0.012 | +0.120 | +0.090 [-0.071, +0.251] |
| within_query_path | +0.161 | +0.176 | -0.031 | -0.192 [-0.449, +0.059] |
| d0_to_d9_commitment | -0.028 | -0.040 | -0.094 | -0.066 [-0.262, +0.143] |

## Per-task routed trend

| task | classifier k0 | classifier k8 | delta | manifold k0 | manifold k8 | delta |
|---|---:|---:|---:|---:|---:|---:|
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 0.320 | 0.570 | +0.250 | 0.492 | 0.523 | +0.031 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.492 | 0.513 | +0.021 | 0.504 | 0.578 | +0.074 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.551 | 0.872 | +0.321 | 0.487 | 0.833 | +0.346 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.487 | 0.898 | +0.411 | 0.523 | 0.964 | +0.442 |

## Interpretation boundary

A growing curve means failed and successful closed-loop trajectories become more separable in single-query MoE computation. It does not show that an earlier MoE state caused the later separation, because later observations are descendants of earlier actions.

Hidden/shared controls test routed specificity. If they grow equally or more, the defensible interpretation is general model-state divergence rather than an extra routed-expert precursor.

Contributions are reconstructed from fp16 hidden captures and are not runtime-exact. Recorded IDs and weights are authoritative; selected-probability reconstruction has maximum MAE 4.2e-05 and the fp16 gate audit recovers at least 63.5% of top-4 sets.

This is exploratory because earlier query-0 contribution and query-0..4 hidden/router results were already inspected.

The supported conclusion is therefore trajectory-level internal-state divergence after several closed-loop interactions. These offline data do not support the stronger claim that the initial MoE state is already wrong, uniquely predicts failure, or causally pushes the system into it.
