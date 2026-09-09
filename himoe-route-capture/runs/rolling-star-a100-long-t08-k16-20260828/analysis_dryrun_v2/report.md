# Rolling-star K=16 experiment

## Dataset

- 48 terminal branches from 3 committed snapshots; 22 success and 26 failure.
- 2 snapshots contain matched success/failure siblings.
- 2234 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 16
- `single_subtask_omission`: 4
- `goal_contact_near_miss`: 2
- `subtask_undo`: 2
- `drop_or_regrasp`: 1
- `active_retry`: 1

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00111.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (snapshots held out together):

| prefix_queries | model | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | snapshot_rate_only | 48 | 2 | 0.4316 | 1.0969 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1729 | 0.0737 | 0.2628 |
| 1 | noise | 48 | 2 | 0.4503 | 1.1437 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0187 | -0.0433 | -0.0187 | -0.0386 | 0.0011 | 0.1542 | 0.0693 | 0.2569 |
| 1 | action | 48 | 2 | 0.6475 | 2.2621 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.2159 | -0.5001 | -0.2159 | -0.4938 | -0.0503 | -0.0430 | -0.3303 | 0.1983 |
| 1 | noise+action | 48 | 2 | 0.6045 | 1.8242 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1729 | -0.4005 | -0.1729 | -0.2815 | -0.1036 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action | 48 | 2 | 0.7724 | 4.4938 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.3407 | -0.7894 | -0.3407 | -0.3883 | -0.2932 | -0.1678 | -0.2387 | -0.0791 |
| 1 | moe_state | 48 | 2 | 0.1667 | 2.3026 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | 0.2650 | 0.6139 | 0.2650 | 0.0306 | 0.5127 | 0.4379 | 0.0025 | 0.7415 |
| 1 | moe_action+state | 48 | 2 | 0.4666 | 5.4351 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0349 | -0.0809 | -0.0349 | -0.3349 | 0.2350 | 0.1380 | -0.1167 | 0.3806 |
| 1 | noise+action+moe_action | 48 | 2 | 0.7694 | 4.1479 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.3377 | -0.7824 | -0.3377 | -0.3962 | -0.2504 | -0.1648 | -0.2397 | -0.0781 |
| 1 | noise+action+moe_action+state | 48 | 2 | 0.4606 | 5.1998 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0290 | -0.0672 | -0.0290 | -0.3348 | 0.3616 | 0.1439 | -0.1046 | 0.3924 |
| 3 | snapshot_rate_only | 48 | 2 | 0.4316 | 1.0969 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2261 | 0.1768 | 0.2753 |
| 3 | noise | 48 | 2 | 0.4696 | 1.1974 | 0.5000 | 0.6875 | -0.1875 | -0.9375 | 0.5625 | -0.0380 | -0.0880 | -0.0380 | -0.0650 | -0.0097 | 0.1881 | 0.1377 | 0.2796 |
| 3 | action | 48 | 2 | 0.7062 | 3.3811 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2745 | -0.6361 | -0.2745 | -0.4068 | -0.1422 | -0.0485 | -0.2763 | 0.1628 |
| 3 | noise+action | 48 | 2 | 0.6577 | 2.1106 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2261 | -0.5238 | -0.2261 | -0.2894 | -0.1416 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action | 48 | 2 | 0.6927 | 2.4753 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2611 | -0.6049 | -0.2611 | -0.3140 | -0.2082 | -0.0350 | -0.0421 | -0.0279 |
| 3 | moe_state | 48 | 2 | 0.5829 | 3.1375 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1513 | -0.3505 | -0.1513 | -0.1904 | -0.0956 | 0.0748 | -0.0245 | 0.1472 |
| 3 | moe_action+state | 48 | 2 | 0.5989 | 4.8289 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1673 | -0.3875 | -0.1673 | -0.3341 | 0.0498 | 0.0588 | -0.0681 | 0.1818 |
| 3 | noise+action+moe_action | 48 | 2 | 0.6893 | 2.5192 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2577 | -0.5969 | -0.2577 | -0.3286 | -0.1753 | -0.0316 | -0.0499 | -0.0080 |
| 3 | noise+action+moe_action+state | 48 | 2 | 0.6008 | 4.8249 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1691 | -0.3918 | -0.1691 | -0.3612 | 0.0071 | 0.0570 | -0.0641 | 0.2915 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | back_12_15 | 8 | top1_mass | 0.1250 | 0.6250 | -0.5000 | -0.7500 | -0.2500 | 0.0297 | 0.8119 | 8 | 8 |
| 1 | action | back_12_15 | 9 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1089 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 7 | entropy | 0.5000 | 0.1250 | 0.3750 | -0.2500 | 1.0000 | 0.1089 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 4 | top1_mass | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.2500 | 0.1287 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 7 | top1_mass | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.2500 | 0.1485 | 1.0000 | 8 | 8 |
| 1 | state | front_2_5 | 9 | top1_mass | 0.2500 | 0.6250 | -0.3750 | -0.5000 | -0.2500 | 0.1485 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 0 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.2500 | 0.1584 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 6 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1584 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 7 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1584 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 8 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1782 | 1.0000 | 8 | 8 |
| 1 | action | front_2_5 | 9 | token_dispersion | 0.5000 | 0.2500 | 0.2500 | -0.2500 | 0.7500 | 0.2970 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 0 | top1_mass | 0.1250 | 0.3750 | -0.2500 | -0.5000 | 0.0000 | 0.3069 | 1.0000 | 8 | 8 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | n_failure_branches | n_type_positive | brier | log_loss | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loop_or_cycling | 1 | failure_rate_only | 26 | 19 | 0.3879 | 1.0340 | 0.0000 | 0.0000 | -0.0741 |
| loop_or_cycling | 1 | noise+action | 26 | 19 | 0.3138 | 0.8366 | 0.0741 | 0.1911 | 0.0000 |
| loop_or_cycling | 1 | moe_action | 26 | 19 | 0.7287 | 3.1195 | -0.3408 | -0.8785 | -0.4149 |
| loop_or_cycling | 1 | moe_state | 26 | 19 | 0.2308 | 2.5842 | 0.1572 | 0.4051 | 0.0830 |
| loop_or_cycling | 1 | moe_action+state | 26 | 19 | 0.2676 | 3.0035 | 0.1203 | 0.3101 | 0.0461 |
| loop_or_cycling | 1 | noise+action+moe_action | 26 | 19 | 0.6975 | 2.6272 | -0.3095 | -0.7980 | -0.3837 |
| loop_or_cycling | 1 | noise+action+moe_action+state | 26 | 19 | 0.2741 | 3.0166 | 0.1138 | 0.2933 | 0.0396 |
| loop_or_cycling | 3 | failure_rate_only | 26 | 19 | 0.3879 | 1.0340 | 0.0000 | 0.0000 | -0.0966 |
| loop_or_cycling | 3 | noise+action | 26 | 19 | 0.2913 | 0.7761 | 0.0966 | 0.2491 | 0.0000 |
| loop_or_cycling | 3 | moe_action | 26 | 19 | 0.5786 | 1.8000 | -0.1907 | -0.4915 | -0.2873 |
| loop_or_cycling | 3 | moe_state | 26 | 19 | 0.3628 | 3.5402 | 0.0251 | 0.0647 | -0.0715 |
| loop_or_cycling | 3 | moe_action+state | 26 | 19 | 0.5295 | 3.6757 | -0.1416 | -0.3651 | -0.2383 |
| loop_or_cycling | 3 | noise+action+moe_action | 26 | 19 | 0.5639 | 1.7062 | -0.1760 | -0.4537 | -0.2726 |
| loop_or_cycling | 3 | noise+action+moe_action+state | 26 | 19 | 0.5266 | 3.6312 | -0.1387 | -0.3576 | -0.2353 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.5000 | 1.0000 | -0.5000 | -1.0000 | -0.2500 | 0.0198 | 0.8812 | 6 | 6 |
| loop_or_cycling | 1 | state | back_12_15 | 3 | top1_mass | 0.5000 | 1.0000 | -0.5000 | -1.0000 | -0.2500 | 0.0396 | 0.8812 | 6 | 6 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | entropy | 1.0000 | 0.5000 | 0.5000 | 0.2500 | 1.0000 | 0.0495 | 0.8812 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 1 | entropy | 0.5000 | 1.0000 | -0.5000 | -1.0000 | -0.2500 | 0.0495 | 0.8812 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 2 | consensus_distance | 0.6667 | 1.0000 | -0.3333 | -0.5000 | -0.2500 | 0.1485 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 2 | top1_mass | 0.6667 | 1.0000 | -0.3333 | -0.5000 | -0.2500 | 0.1485 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | consensus_distance | 0.6667 | 1.0000 | -0.3333 | -1.0000 | 0.0000 | 0.1683 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 3 | consensus_distance | 0.6667 | 1.0000 | -0.3333 | -0.5000 | -0.2500 | 0.1683 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 3 | top1_mass | 0.6667 | 1.0000 | -0.3333 | -0.5000 | -0.2500 | 0.1683 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 4 | consensus_distance | 0.6667 | 1.0000 | -0.3333 | -0.5000 | -0.2500 | 0.1683 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 9 | consensus_distance | 0.6667 | 1.0000 | -0.3333 | -1.0000 | 0.0000 | 0.1683 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | state | front_2_5 | 9 | entropy | 0.6667 | 1.0000 | -0.3333 | -1.0000 | 0.0000 | 0.1683 | 1.0000 | 6 | 6 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Noise and first action chunks are explicit controls; `noise+action+moe_action+state` must beat `noise+action` before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
