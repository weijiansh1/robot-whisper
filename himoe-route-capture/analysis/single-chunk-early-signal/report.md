# Early single-query outcome signal

## Scope

- Unit: one rollout at one control query. Features never cross query boundaries.
- Queries 0 through 4 are evaluated independently.
- Query 0 compares exact shared initial states. Later queries share only the rollout origin; they are not tree-sampled counterfactuals.
- Positive label: eventual rollout failure. The all-success task is excluded.
- Evaluations: (1) leave one complete initial state out; (2) hold out complete noise seeds while retaining other seeds from the same state.
- Conditional AUC always counts only failure/success pairs from the same initial state and the same fitted fold.
- Control: current simulator state, proprioception, and all 10x7 emitted action values.
- Each control/router/hidden block is standardized and reduced to 12 train-only PCA components; logistic C=0.1 is fixed.
- Uncertainty: initial-state bootstrap, conditional on these tasks and recorded rollouts.

`hidden` means the HB router-input representation. It is not the selected experts' output contribution.


## Unseen-initial-state result

AUC 0.5 is chance. Delta is `control + router + hidden` minus the control AUC.

| chunk | control AUC | hidden-only AUC | MoE-only AUC | control+MoE AUC | delta | tasks with positive delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.443 [0.374, 0.501] | 0.531 [0.476, 0.596] | 0.488 [0.440, 0.537] | 0.435 [0.388, 0.487] | -0.009 [-0.088, 0.087] | 1/4 |
| 1 | 0.517 [0.445, 0.591] | 0.452 [0.389, 0.516] | 0.399 [0.341, 0.457] | 0.403 [0.345, 0.471] | -0.114 [-0.180, -0.041] | 0/4 |
| 2 | 0.514 [0.438, 0.593] | 0.428 [0.372, 0.478] | 0.436 [0.371, 0.498] | 0.442 [0.377, 0.512] | -0.072 [-0.147, -0.005] | 2/4 |
| 3 | 0.498 [0.436, 0.563] | 0.414 [0.338, 0.495] | 0.459 [0.393, 0.530] | 0.435 [0.372, 0.505] | -0.064 [-0.131, 0.004] | 0/4 |
| 4 | 0.479 [0.398, 0.571] | 0.448 [0.381, 0.513] | 0.391 [0.324, 0.449] | 0.418 [0.354, 0.483] | -0.060 [-0.135, 0.007] | 1/4 |

## Same-initial-state, held-out-seed result

AUC 0.5 is chance. Delta is `control + router + hidden` minus the control AUC.

| chunk | control AUC | hidden-only AUC | MoE-only AUC | control+MoE AUC | delta | tasks with positive delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.428 [0.362, 0.485] | 0.537 [0.471, 0.623] | 0.510 [0.460, 0.576] | 0.464 [0.402, 0.541] | +0.037 [-0.032, 0.126] | 3/4 |
| 1 | 0.507 [0.434, 0.582] | 0.508 [0.430, 0.588] | 0.493 [0.410, 0.573] | 0.485 [0.411, 0.562] | -0.023 [-0.096, 0.060] | 1/4 |
| 2 | 0.542 [0.468, 0.648] | 0.508 [0.429, 0.579] | 0.525 [0.442, 0.599] | 0.528 [0.448, 0.612] | -0.014 [-0.111, 0.051] | 2/4 |
| 3 | 0.566 [0.491, 0.642] | 0.474 [0.414, 0.554] | 0.492 [0.426, 0.577] | 0.492 [0.419, 0.578] | -0.074 [-0.156, 0.014] | 0/4 |
| 4 | 0.528 [0.449, 0.597] | 0.490 [0.388, 0.582] | 0.474 [0.373, 0.560] | 0.525 [0.432, 0.601] | -0.002 [-0.092, 0.088] | 2/4 |

## Bottom line

The fixed screen does not show a stable early HB-MoE outcome signal. The only repeatable hint is query-0 `hidden` alone: unseen-state macro AUC 0.531 [0.476, 0.596] and same-state held-out-seed macro AUC 0.537 [0.471, 0.623]. Both intervals include chance. After adding router features and the state/action control, the same-state AUC is 0.464 and the incremental delta is +0.037 [-0.032, 0.126], so the hint is not incremental under this model. No later query has a consistently positive increment.

## Per-task control+MoE result

| evaluation | task | chunk | failures | mixed states | control AUC | hidden AUC | control+MoE AUC | delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| scene_loso | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 0 | 42 | 7 | 0.516 | 0.692 | 0.612 | +0.096 [-0.026, 0.258] |
| scene_loso | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 1 | 42 | 7 | 0.534 | 0.486 | 0.436 | -0.098 [-0.304, 0.135] |
| scene_loso | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 2 | 42 | 7 | 0.555 | 0.379 | 0.368 | -0.187 [-0.392, -0.020] |
| scene_loso | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 3 | 42 | 7 | 0.418 | 0.368 | 0.322 | -0.096 [-0.177, -0.008] |
| scene_loso | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 4 | 42 | 7 | 0.363 | 0.454 | 0.297 | -0.066 [-0.248, 0.101] |
| scene_loso | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0 | 216 | 13 | 0.455 | 0.478 | 0.428 | -0.026 [-0.113, 0.037] |
| scene_loso | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 1 | 216 | 13 | 0.501 | 0.505 | 0.480 | -0.021 [-0.112, 0.070] |
| scene_loso | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 2 | 216 | 13 | 0.478 | 0.424 | 0.487 | +0.009 [-0.046, 0.064] |
| scene_loso | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 3 | 216 | 13 | 0.559 | 0.504 | 0.465 | -0.095 [-0.153, -0.031] |
| scene_loso | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 4 | 216 | 13 | 0.513 | 0.506 | 0.518 | +0.005 [-0.059, 0.070] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0 | 12 | 8 | 0.358 | 0.464 | 0.327 | -0.031 [-0.286, 0.266] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 1 | 12 | 8 | 0.556 | 0.411 | 0.318 | -0.237 [-0.335, -0.155] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 2 | 12 | 8 | 0.570 | 0.444 | 0.439 | -0.131 [-0.309, 0.018] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 3 | 12 | 8 | 0.556 | 0.374 | 0.531 | -0.025 [-0.142, 0.134] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 4 | 12 | 8 | 0.531 | 0.394 | 0.494 | -0.036 [-0.190, 0.091] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0 | 37 | 12 | 0.446 | 0.489 | 0.371 | -0.075 [-0.201, 0.098] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 1 | 37 | 12 | 0.478 | 0.407 | 0.378 | -0.100 [-0.194, 0.021] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 2 | 37 | 12 | 0.454 | 0.466 | 0.474 | +0.020 [-0.102, 0.132] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 3 | 37 | 12 | 0.460 | 0.408 | 0.421 | -0.039 [-0.265, 0.149] |
| scene_loso | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 4 | 37 | 12 | 0.509 | 0.438 | 0.364 | -0.145 [-0.292, -0.010] |
| seed_heldout | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 0 | 42 | 7 | 0.393 | 0.631 | 0.484 | +0.090 [-0.058, 0.316] |
| seed_heldout | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 1 | 42 | 7 | 0.533 | 0.607 | 0.533 | +0.000 [-0.216, 0.264] |
| seed_heldout | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 2 | 42 | 7 | 0.705 | 0.418 | 0.508 | -0.197 [-0.429, -0.092] |
| seed_heldout | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 3 | 42 | 7 | 0.508 | 0.459 | 0.410 | -0.098 [-0.281, 0.097] |
| seed_heldout | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 4 | 42 | 7 | 0.467 | 0.492 | 0.475 | +0.008 [-0.162, 0.206] |
| seed_heldout | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0 | 216 | 13 | 0.495 | 0.430 | 0.392 | -0.103 [-0.164, -0.044] |
| seed_heldout | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 1 | 216 | 13 | 0.505 | 0.531 | 0.538 | +0.033 [-0.075, 0.120] |
| seed_heldout | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 2 | 216 | 13 | 0.462 | 0.538 | 0.545 | +0.082 [-0.025, 0.174] |
| seed_heldout | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 3 | 216 | 13 | 0.568 | 0.434 | 0.448 | -0.120 [-0.199, -0.027] |
| seed_heldout | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 4 | 216 | 13 | 0.514 | 0.500 | 0.486 | -0.028 [-0.109, 0.053] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0 | 12 | 8 | 0.429 | 0.548 | 0.500 | +0.071 [-0.100, 0.351] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 1 | 12 | 8 | 0.417 | 0.417 | 0.381 | -0.036 [-0.159, 0.054] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 2 | 12 | 8 | 0.524 | 0.560 | 0.500 | -0.024 [-0.190, 0.184] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 3 | 12 | 8 | 0.607 | 0.536 | 0.607 | +0.000 [-0.157, 0.214] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 4 | 12 | 8 | 0.679 | 0.488 | 0.643 | -0.036 [-0.317, 0.222] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0 | 37 | 12 | 0.394 | 0.539 | 0.482 | +0.088 [-0.062, 0.175] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 1 | 37 | 12 | 0.575 | 0.477 | 0.487 | -0.088 [-0.207, 0.055] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 2 | 37 | 12 | 0.477 | 0.518 | 0.560 | +0.083 [-0.165, 0.200] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 3 | 37 | 12 | 0.580 | 0.466 | 0.503 | -0.078 [-0.274, 0.094] |
| seed_heldout | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 4 | 37 | 12 | 0.451 | 0.482 | 0.497 | +0.047 [-0.074, 0.201] |

## Interpretation boundary

This screen uses eventual outcome labels, not annotated trap-onset labels. A positive result establishes only early outcome decodability under the recorded policy. It does not identify a trap mechanism, recoverability, or a causal expert contribution. The five chunk indices and seven model families are exploratory and are not multiplicity-corrected.
