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

| prefix_queries | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | snapshot_rate_only | absolute | 48 | 2 | 0.4316 | 1.0969 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1729 | 0.0737 | 0.2628 |
| 1 | noise | absolute | 48 | 2 | 0.4503 | 1.1437 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0187 | -0.0433 | noise+action | -0.0187 | -0.0386 | 0.0011 | 0.1542 | 0.0693 | 0.2569 |
| 1 | action | absolute | 48 | 2 | 0.6475 | 2.2621 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.2159 | -0.5001 | noise+action | -0.2159 | -0.4938 | -0.0503 | -0.0430 | -0.3303 | 0.1983 |
| 1 | noise+action | absolute | 48 | 2 | 0.6045 | 1.8242 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1729 | -0.4005 | noise+action | -0.1729 | -0.2815 | -0.1036 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action | absolute | 48 | 2 | 0.7724 | 4.4938 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.3407 | -0.7894 | noise+action | -0.3407 | -0.3883 | -0.2932 | -0.1678 | -0.2387 | -0.0791 |
| 1 | moe_state | absolute | 48 | 2 | 0.1667 | 2.3026 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | 0.2650 | 0.6139 | noise+action | 0.2650 | 0.0306 | 0.5127 | 0.4379 | 0.0025 | 0.7415 |
| 1 | moe_action+state | absolute | 48 | 2 | 0.4666 | 5.4351 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0349 | -0.0809 | noise+action | -0.0349 | -0.3349 | 0.2350 | 0.1380 | -0.1167 | 0.3806 |
| 1 | noise+action+moe_action | absolute | 48 | 2 | 0.7694 | 4.1479 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.3377 | -0.7824 | noise+action | -0.3377 | -0.3962 | -0.2504 | -0.1648 | -0.2397 | -0.0781 |
| 1 | noise+action+moe_action+state | absolute | 48 | 2 | 0.4606 | 5.1998 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0290 | -0.0672 | noise+action | -0.0290 | -0.3348 | 0.3616 | 0.1439 | -0.1046 | 0.3924 |
| 1 | noise_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4451 | 1.1335 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0135 | -0.0313 | noise+action_within_snapshot | -0.0135 | -0.0275 | -0.0015 | 0.0021 | -0.0119 | 0.0135 |
| 1 | action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4470 | 1.1517 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0153 | -0.0355 | noise+action_within_snapshot | -0.0153 | -0.0295 | -0.0077 | 0.0002 | -0.0032 | 0.0040 |
| 1 | noise+action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4472 | 1.1440 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0156 | -0.0360 | noise+action_within_snapshot | -0.0156 | -0.0254 | -0.0057 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4660 | 1.2679 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0343 | -0.0795 | noise+action_within_snapshot | -0.0343 | -0.0670 | -0.0146 | -0.0188 | -0.0355 | -0.0082 |
| 1 | moe_state_within_snapshot | within_snapshot_centered | 48 | 2 | 0.7088 | 4.8899 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.2771 | -0.6421 | noise+action_within_snapshot | -0.2771 | -0.3423 | -0.2120 | -0.2616 | -0.3207 | -0.2025 |
| 1 | moe_action+state_within_snapshot | within_snapshot_centered | 48 | 2 | 0.7297 | 5.3984 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.2981 | -0.6906 | noise+action_within_snapshot | -0.2981 | -0.3601 | -0.2341 | -0.2825 | -0.3286 | -0.2415 |
| 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4658 | 1.2585 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0342 | -0.0792 | noise+action_within_snapshot | -0.0342 | -0.0605 | -0.0144 | -0.0186 | -0.0328 | -0.0087 |
| 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 48 | 2 | 0.7278 | 5.3790 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.2962 | -0.6862 | noise+action_within_snapshot | -0.2962 | -0.3530 | -0.2472 | -0.2806 | -0.3262 | -0.1999 |
| 3 | snapshot_rate_only | absolute | 48 | 2 | 0.4316 | 1.0969 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2261 | 0.1584 | 0.2753 |
| 3 | noise | absolute | 48 | 2 | 0.4696 | 1.1974 | 0.5000 | 0.6875 | -0.1875 | -0.9375 | 0.5625 | -0.0380 | -0.0880 | noise+action | -0.0380 | -0.0909 | -0.0103 | 0.1881 | 0.1283 | 0.2796 |
| 3 | action | absolute | 48 | 2 | 0.7062 | 3.3811 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2745 | -0.6361 | noise+action | -0.2745 | -0.5235 | -0.1422 | -0.0485 | -0.2763 | 0.0979 |
| 3 | noise+action | absolute | 48 | 2 | 0.6577 | 2.1106 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2261 | -0.5238 | noise+action | -0.2261 | -0.2894 | -0.1416 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action | absolute | 48 | 2 | 0.6927 | 2.4753 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2611 | -0.6049 | noise+action | -0.2611 | -0.3351 | -0.1915 | -0.0350 | -0.0421 | -0.0262 |
| 3 | moe_state | absolute | 48 | 2 | 0.5829 | 3.1375 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1513 | -0.3505 | noise+action | -0.1513 | -0.1885 | -0.1141 | 0.0748 | -0.0488 | 0.1472 |
| 3 | moe_action+state | absolute | 48 | 2 | 0.5989 | 4.8289 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1673 | -0.3875 | noise+action | -0.1673 | -0.3612 | 0.0498 | 0.0588 | -0.0641 | 0.2970 |
| 3 | noise+action+moe_action | absolute | 48 | 2 | 0.6893 | 2.5192 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2577 | -0.5969 | noise+action | -0.2577 | -0.3286 | -0.2020 | -0.0316 | -0.0530 | -0.0080 |
| 3 | noise+action+moe_action+state | absolute | 48 | 2 | 0.6008 | 4.8249 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1691 | -0.3918 | noise+action | -0.1691 | -0.3342 | 0.0443 | 0.0570 | -0.0641 | 0.2915 |
| 3 | noise_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4591 | 1.1722 | 0.5000 | 0.6875 | -0.1875 | -0.9375 | 0.5625 | -0.0274 | -0.0636 | noise+action_within_snapshot | -0.0274 | -0.0443 | -0.0024 | -0.0164 | -0.0253 | -0.0024 |
| 3 | action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4368 | 1.1344 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0051 | -0.0119 | noise+action_within_snapshot | -0.0051 | -0.0090 | -0.0022 | 0.0059 | -0.0002 | 0.0120 |
| 3 | noise+action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4427 | 1.1263 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0110 | -0.0256 | noise+action_within_snapshot | -0.0110 | -0.0237 | -0.0015 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4425 | 1.1306 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0108 | -0.0251 | noise+action_within_snapshot | -0.0108 | -0.0129 | -0.0088 | 0.0002 | -0.0060 | 0.0091 |
| 3 | moe_state_within_snapshot | within_snapshot_centered | 48 | 2 | 0.7202 | 5.1848 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2885 | -0.6685 | noise+action_within_snapshot | -0.2885 | -0.3481 | -0.2399 | -0.2775 | -0.3309 | -0.2241 |
| 3 | moe_action+state_within_snapshot | within_snapshot_centered | 48 | 2 | 0.5495 | 3.5172 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1179 | -0.2731 | noise+action_within_snapshot | -0.1179 | -0.3154 | -0.0150 | -0.1068 | -0.3154 | 0.0107 |
| 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 48 | 2 | 0.4424 | 1.1296 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0108 | -0.0250 | noise+action_within_snapshot | -0.0108 | -0.0133 | -0.0069 | 0.0002 | -0.0069 | 0.0117 |
| 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 48 | 2 | 0.5494 | 3.5160 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1178 | -0.2729 | noise+action_within_snapshot | -0.1178 | -0.2188 | -0.0125 | -0.1067 | -0.3154 | 0.0022 |

Snapshot difficulty (one row per rolling state; leave-one-snapshot-out):

| model | snapshots | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initial_sim_state | 3 | 10.0000 | 0.0733 | 0.2501 | 0.2567 | 0.1121 | 0.4239 |
| mean_output_action | 3 | 10.0000 | 0.3239 | 0.4548 | 0.0062 | -0.2609 | 0.2732 |
| mean_moe_action | 3 | 10.0000 | 0.3287 | 0.5550 | 0.0014 | -0.0931 | 0.1548 |
| initial_policy_state | 3 | 10.0000 | 0.3292 | 0.5253 | 0.0009 | -0.2504 | 0.3662 |
| mean_rate | 3 | NA | 0.3301 | 0.4792 | 0.0000 | 0.0000 | 0.0000 |
| as_moe | 3 | 10.0000 | 0.3301 | 0.4792 | 0.0000 | 0.0000 | 0.0000 |
| sim_state+mean_moe_action+state | 3 | 10.0000 | 0.3498 | 0.5574 | -0.0197 | -0.1904 | 0.3140 |
| mean_moe_action+state | 3 | 10.0000 | 0.3517 | 0.5586 | -0.0216 | -0.1896 | 0.2341 |
| mean_moe_state | 3 | 10.0000 | 0.3594 | 0.4877 | -0.0293 | -0.3623 | 0.4649 |

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
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
