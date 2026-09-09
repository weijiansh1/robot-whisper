# Rolling-star K=16 experiment

## Dataset

- 64 terminal branches from 4 committed snapshots; 22 success and 42 failure.
- 2 snapshots contain matched success/failure siblings.
- 3066 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 22
- `single_subtask_omission`: 12
- `goal_contact_near_miss`: 3
- `drop_or_regrasp`: 2
- `subtask_undo`: 2
- `active_retry`: 1

Stagnation or loop/cycling covers 35/42 failures; 7 failures require other physical mechanisms: {"active_retry": 1, "goal_contact_near_miss": 3, "single_subtask_omission": 3}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00111.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (snapshots held out together):

| prefix_queries | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | snapshot_rate_only | absolute | 64 | 2 | 0.3418 | 0.9402 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2189 | -0.0667 | 0.2962 |
| 1 | noise | absolute | 64 | 2 | 0.3598 | 1.0409 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0180 | -0.0526 | noise+action | -0.0180 | -0.0356 | -0.0007 | 0.2009 | 0.0074 | 0.4790 |
| 1 | action | absolute | 64 | 2 | 0.6050 | 2.7474 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2632 | -0.7700 | noise+action | -0.2632 | -0.5986 | -0.0194 | -0.0443 | -0.1119 | 0.0033 |
| 1 | noise+action | absolute | 64 | 2 | 0.5607 | 2.0359 | 1.0000 | 0.6875 | 0.3125 | 0.1187 | 0.5625 | -0.2189 | -0.6404 | noise+action | -0.2189 | -0.4046 | -0.0204 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action | absolute | 64 | 2 | 0.4050 | 3.5222 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0063 | -0.0632 | -0.1849 | noise+action | -0.0632 | -0.1093 | 0.0260 | 0.1557 | -0.0703 | 0.3008 |
| 1 | moe_state | absolute | 64 | 2 | 0.3701 | 1.9325 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0283 | -0.0829 | noise+action | -0.0283 | -0.2741 | 0.1813 | 0.1906 | 0.0277 | 0.4003 |
| 1 | moe_action+state | absolute | 64 | 2 | 0.4226 | 3.7116 | 0.0000 | 0.6875 | -0.6875 | -0.8812 | -0.4375 | -0.0808 | -0.2363 | noise+action | -0.0808 | -0.2334 | -0.0192 | 0.1381 | -0.0174 | 0.2444 |
| 1 | noise+action+moe_action | absolute | 64 | 2 | 0.4106 | 3.5913 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4937 | -0.0688 | -0.2014 | noise+action | -0.0688 | -0.1627 | 0.0623 | 0.1500 | -0.0500 | 0.4485 |
| 1 | noise+action+moe_action+state | absolute | 64 | 2 | 0.4239 | 3.7460 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4937 | -0.0821 | -0.2403 | noise+action | -0.0821 | -0.1998 | 0.1240 | 0.1367 | -0.0313 | 0.2927 |
| 1 | noise_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3582 | 1.0318 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0164 | -0.0480 | noise+action_within_snapshot | -0.0164 | -0.0671 | 0.0022 | -0.0012 | -0.0057 | 0.0014 |
| 1 | action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3506 | 0.9840 | 0.5000 | 0.6875 | -0.1875 | -0.3812 | 0.0625 | -0.0088 | -0.0258 | noise+action_within_snapshot | -0.0088 | -0.0133 | -0.0030 | 0.0064 | -0.0059 | 0.0236 |
| 1 | noise+action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3570 | 1.0233 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0152 | -0.0444 | noise+action_within_snapshot | -0.0152 | -0.0310 | -0.0034 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3609 | 1.0746 | 1.0000 | 0.6875 | 0.3125 | 0.1187 | 0.5625 | -0.0191 | -0.0558 | noise+action_within_snapshot | -0.0191 | -0.0287 | -0.0071 | -0.0039 | -0.0147 | 0.0091 |
| 1 | moe_state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.5035 | 5.2285 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1617 | -0.4730 | noise+action_within_snapshot | -0.1617 | -0.2642 | -0.0616 | -0.1465 | -0.2090 | -0.0672 |
| 1 | moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.4654 | 2.7762 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.1236 | -0.3615 | noise+action_within_snapshot | -0.1236 | -0.1439 | -0.0753 | -0.1084 | -0.1876 | -0.0485 |
| 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3607 | 1.0761 | 0.5000 | 0.6875 | -0.1875 | -0.3812 | 0.0625 | -0.0189 | -0.0553 | noise+action_within_snapshot | -0.0189 | -0.0384 | -0.0086 | -0.0037 | -0.0074 | 0.0004 |
| 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.4594 | 2.7635 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5063 | -0.1176 | -0.3441 | noise+action_within_snapshot | -0.1176 | -0.1767 | -0.0112 | -0.1024 | -0.1780 | -0.0841 |
| 3 | snapshot_rate_only | absolute | 64 | 2 | 0.3418 | 0.9402 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2113 | 0.0615 | 0.4043 |
| 3 | noise | absolute | 64 | 2 | 0.3786 | 1.1362 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0368 | -0.1075 | noise+action | -0.0368 | -0.0896 | -0.0096 | 0.1746 | -0.0060 | 0.2210 |
| 3 | action | absolute | 64 | 2 | 0.5848 | 2.2552 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2430 | -0.7110 | noise+action | -0.2430 | -0.4750 | 0.0367 | -0.0317 | -0.0814 | 0.0192 |
| 3 | noise+action | absolute | 64 | 2 | 0.5531 | 2.0944 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2113 | -0.6183 | noise+action | -0.2113 | -0.3137 | -0.0711 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action | absolute | 64 | 2 | 0.3574 | 2.5818 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0063 | -0.0156 | -0.0457 | noise+action | -0.0156 | -0.1438 | 0.1606 | 0.1957 | -0.0524 | 0.4075 |
| 3 | moe_state | absolute | 64 | 2 | 0.5997 | 3.7915 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2579 | -0.7546 | noise+action | -0.2579 | -0.4718 | 0.0726 | -0.0466 | -0.4059 | 0.3169 |
| 3 | moe_action+state | absolute | 64 | 2 | 0.3830 | 2.9899 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0063 | -0.0412 | -0.1204 | noise+action | -0.0412 | -0.1374 | 0.1945 | 0.1702 | -0.0888 | 0.6143 |
| 3 | noise+action+moe_action | absolute | 64 | 2 | 0.3568 | 2.8989 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0063 | -0.0150 | -0.0438 | noise+action | -0.0150 | -0.0721 | 0.0922 | 0.1963 | -0.0364 | 0.4212 |
| 3 | noise+action+moe_action+state | absolute | 64 | 2 | 0.3884 | 3.0546 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0466 | -0.1364 | noise+action | -0.0466 | -0.1683 | 0.1790 | 0.1647 | -0.1078 | 0.3079 |
| 3 | noise_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3755 | 1.1548 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5063 | -0.0337 | -0.0985 | noise+action_within_snapshot | -0.0337 | -0.0883 | 0.0019 | -0.0242 | -0.0497 | 0.0010 |
| 3 | action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3444 | 0.9518 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0026 | -0.0075 | noise+action_within_snapshot | -0.0026 | -0.0039 | -0.0005 | 0.0069 | -0.0058 | 0.0176 |
| 3 | noise+action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3513 | 0.9833 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0095 | -0.0278 | noise+action_within_snapshot | -0.0095 | -0.0271 | 0.0017 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3511 | 0.9947 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0093 | -0.0271 | noise+action_within_snapshot | -0.0093 | -0.0142 | -0.0010 | 0.0002 | -0.0044 | 0.0037 |
| 3 | moe_state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.4780 | 3.7599 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.1362 | -0.3986 | noise+action_within_snapshot | -0.1362 | -0.2026 | -0.0676 | -0.1267 | -0.2396 | -0.0606 |
| 3 | moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3498 | 0.9925 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0080 | -0.0233 | noise+action_within_snapshot | -0.0080 | -0.0126 | -0.0004 | 0.0015 | -0.0057 | 0.0154 |
| 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3513 | 0.9948 | 0.5000 | 0.6875 | -0.1875 | -0.3812 | 0.0625 | -0.0095 | -0.0278 | noise+action_within_snapshot | -0.0095 | -0.0209 | -0.0021 | 0.0000 | -0.0023 | 0.0067 |
| 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3501 | 0.9910 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0083 | -0.0242 | noise+action_within_snapshot | -0.0083 | -0.0166 | -0.0018 | 0.0012 | -0.0062 | 0.0044 |

Snapshot difficulty (one row per rolling state; leave-one-snapshot-out):

| model | snapshots | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initial_policy_state | 4 | 10.0000 | 0.1425 | 0.3298 | 0.1231 | 0.0676 | 0.1948 |
| mean_moe_action | 4 | 10.0000 | 0.1985 | 0.3966 | 0.0672 | 0.0181 | 0.0808 |
| mean_moe_action+state | 4 | 10.0000 | 0.2336 | 0.4336 | 0.0320 | -0.0589 | 0.1676 |
| sim_state+mean_moe_action+state | 4 | 10.0000 | 0.2337 | 0.4334 | 0.0320 | -0.0664 | 0.1187 |
| initial_sim_state | 4 | 10.0000 | 0.2414 | 0.4095 | 0.0242 | -0.1150 | 0.1308 |
| as_moe | 4 | 10.0000 | 0.2656 | 0.4583 | 0.0000 | -0.0000 | 0.0000 |
| mean_rate | 4 | NA | 0.2656 | 0.4583 | 0.0000 | 0.0000 | 0.0000 |
| mean_moe_state | 4 | 10.0000 | 0.2798 | 0.4713 | -0.0142 | -0.1561 | 0.0745 |
| mean_output_action | 4 | 10.0000 | 0.5127 | 0.6206 | -0.2471 | -0.3694 | 0.1174 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | back_12_15 | 8 | top1_mass | 0.1250 | 0.6250 | -0.5000 | -0.7500 | -0.5000 | 0.0909 | 0.8182 | 8 | 8 |
| 1 | action | back_12_15 | 0 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.3750 | 0.1818 | 1.0000 | 8 | 8 |
| 1 | action | front_2_5 | 9 | entropy | 0.1250 | 0.3750 | -0.2500 | -0.7500 | -0.2500 | 0.1818 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 6 | consensus_distance | 0.2500 | 0.5000 | -0.2500 | -0.2500 | -0.2500 | 0.1818 | 1.0000 | 8 | 8 |
| 1 | state | front_2_5 | 0 | consensus_distance | 0.3750 | 0.1250 | 0.2500 | 0.2500 | 0.2500 | 0.1818 | 1.0000 | 8 | 8 |
| 1 | state | front_2_5 | 1 | entropy | 0.2500 | 0.5000 | -0.2500 | -0.2500 | -0.2500 | 0.1818 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 4 | top1_mass | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.3750 | 0.2727 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 8 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | -0.3750 | 0.2727 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 5 | top1_mass | 0.5000 | 0.2500 | 0.2500 | 0.2500 | 0.5000 | 0.2727 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 6 | entropy | 0.1250 | 0.3750 | -0.2500 | -0.2500 | -0.2500 | 0.2727 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 7 | entropy | 0.5000 | 0.1250 | 0.3750 | 0.3750 | 1.0000 | 0.2727 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 7 | top1_mass | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.3750 | 0.2727 | 1.0000 | 8 | 8 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | feature_scope | n_failure_branches | n_type_positive | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 1 | failure_rate_only | absolute | 42 | 11 | 0.2911 | 0.8949 | noise+action | 0.0000 | 0.0000 | 0.0372 |
| stagnation | 1 | noise+action | absolute | 42 | 11 | 0.3283 | 2.5010 | noise+action | -0.0372 | -0.1278 | 0.0000 |
| stagnation | 1 | moe_action | absolute | 42 | 11 | 0.2623 | 3.5060 | noise+action | 0.0287 | 0.0987 | 0.0659 |
| stagnation | 1 | moe_state | absolute | 42 | 11 | 0.6658 | 6.0374 | noise+action | -0.3747 | -1.2874 | -0.3375 |
| stagnation | 1 | moe_action+state | absolute | 42 | 11 | 0.2682 | 3.6481 | noise+action | 0.0229 | 0.0787 | 0.0601 |
| stagnation | 1 | noise+action+moe_action | absolute | 42 | 11 | 0.2624 | 3.5408 | noise+action | 0.0286 | 0.0984 | 0.0658 |
| stagnation | 1 | noise+action+moe_action+state | absolute | 42 | 11 | 0.2702 | 3.6491 | noise+action | 0.0208 | 0.0716 | 0.0580 |
| stagnation | 1 | noise+action_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3050 | 0.9852 | noise+action_within_snapshot | -0.0139 | -0.0477 | 0.0000 |
| stagnation | 1 | moe_action_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3438 | 2.5259 | noise+action_within_snapshot | -0.0527 | -0.1811 | -0.0388 |
| stagnation | 1 | moe_state_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3812 | 4.4803 | noise+action_within_snapshot | -0.0901 | -0.3096 | -0.0762 |
| stagnation | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3868 | 4.5064 | noise+action_within_snapshot | -0.0957 | -0.3290 | -0.0819 |
| stagnation | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3428 | 2.5646 | noise+action_within_snapshot | -0.0517 | -0.1777 | -0.0378 |
| stagnation | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3865 | 4.5057 | noise+action_within_snapshot | -0.0955 | -0.3280 | -0.0816 |
| stagnation | 3 | failure_rate_only | absolute | 42 | 11 | 0.2911 | 0.8949 | noise+action | 0.0000 | 0.0000 | 0.0264 |
| stagnation | 3 | noise+action | absolute | 42 | 11 | 0.3175 | 3.2467 | noise+action | -0.0264 | -0.0907 | 0.0000 |
| stagnation | 3 | moe_action | absolute | 42 | 11 | 0.2920 | 3.7463 | noise+action | -0.0010 | -0.0033 | 0.0254 |
| stagnation | 3 | moe_state | absolute | 42 | 11 | 0.3498 | 3.8278 | noise+action | -0.0588 | -0.2020 | -0.0324 |
| stagnation | 3 | moe_action+state | absolute | 42 | 11 | 0.2835 | 3.7069 | noise+action | 0.0075 | 0.0259 | 0.0339 |
| stagnation | 3 | noise+action+moe_action | absolute | 42 | 11 | 0.2926 | 3.7479 | noise+action | -0.0015 | -0.0053 | 0.0249 |
| stagnation | 3 | noise+action+moe_action+state | absolute | 42 | 11 | 0.2818 | 3.6990 | noise+action | 0.0092 | 0.0316 | 0.0356 |
| stagnation | 3 | noise+action_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3029 | 0.9610 | noise+action_within_snapshot | -0.0118 | -0.0405 | 0.0000 |
| stagnation | 3 | moe_action_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3139 | 1.1355 | noise+action_within_snapshot | -0.0228 | -0.0784 | -0.0110 |
| stagnation | 3 | moe_state_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3857 | 4.3964 | noise+action_within_snapshot | -0.0946 | -0.3251 | -0.0828 |
| stagnation | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3248 | 1.8575 | noise+action_within_snapshot | -0.0337 | -0.1158 | -0.0219 |
| stagnation | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3148 | 1.1482 | noise+action_within_snapshot | -0.0237 | -0.0815 | -0.0119 |
| stagnation | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 11 | 0.3252 | 2.0371 | noise+action_within_snapshot | -0.0341 | -0.1172 | -0.0223 |
| loop_or_cycling | 1 | failure_rate_only | absolute | 42 | 26 | 0.3227 | 0.8471 | noise+action | 0.0000 | 0.0000 | -0.0233 |
| loop_or_cycling | 1 | noise+action | absolute | 42 | 26 | 0.2994 | 0.7961 | noise+action | 0.0233 | 0.0723 | 0.0000 |
| loop_or_cycling | 1 | moe_action | absolute | 42 | 26 | 0.3849 | 1.0055 | noise+action | -0.0622 | -0.1926 | -0.0855 |
| loop_or_cycling | 1 | moe_state | absolute | 42 | 26 | 0.6118 | 6.6994 | noise+action | -0.2891 | -0.8959 | -0.3125 |
| loop_or_cycling | 1 | moe_action+state | absolute | 42 | 26 | 0.3903 | 1.2748 | noise+action | -0.0676 | -0.2094 | -0.0909 |
| loop_or_cycling | 1 | noise+action+moe_action | absolute | 42 | 26 | 0.3729 | 0.9676 | noise+action | -0.0502 | -0.1554 | -0.0735 |
| loop_or_cycling | 1 | noise+action+moe_action+state | absolute | 42 | 26 | 0.3815 | 1.1426 | noise+action | -0.0588 | -0.1821 | -0.0821 |
| loop_or_cycling | 1 | noise+action_within_snapshot | within_snapshot_centered | 42 | 26 | 0.3418 | 0.9032 | noise+action_within_snapshot | -0.0191 | -0.0593 | 0.0000 |
| loop_or_cycling | 1 | moe_action_within_snapshot | within_snapshot_centered | 42 | 26 | 0.3726 | 0.9777 | noise+action_within_snapshot | -0.0498 | -0.1544 | -0.0307 |
| loop_or_cycling | 1 | moe_state_within_snapshot | within_snapshot_centered | 42 | 26 | 0.5223 | 4.8392 | noise+action_within_snapshot | -0.1996 | -0.6186 | -0.1805 |
| loop_or_cycling | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 26 | 0.4746 | 4.1732 | noise+action_within_snapshot | -0.1518 | -0.4705 | -0.1327 |
| loop_or_cycling | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 26 | 0.3720 | 0.9759 | noise+action_within_snapshot | -0.0493 | -0.1527 | -0.0301 |
| loop_or_cycling | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 26 | 0.4767 | 4.1792 | noise+action_within_snapshot | -0.1539 | -0.4770 | -0.1348 |
| loop_or_cycling | 3 | failure_rate_only | absolute | 42 | 26 | 0.3227 | 0.8471 | noise+action | 0.0000 | 0.0000 | 0.0137 |
| loop_or_cycling | 3 | noise+action | absolute | 42 | 26 | 0.3364 | 0.9966 | noise+action | -0.0137 | -0.0425 | 0.0000 |
| loop_or_cycling | 3 | moe_action | absolute | 42 | 26 | 0.4445 | 1.2281 | noise+action | -0.1218 | -0.3775 | -0.1081 |
| loop_or_cycling | 3 | moe_state | absolute | 42 | 26 | 0.6597 | 5.9951 | noise+action | -0.3370 | -1.0443 | -0.3233 |
| loop_or_cycling | 3 | moe_action+state | absolute | 42 | 26 | 0.5950 | 2.7856 | noise+action | -0.2723 | -0.8438 | -0.2586 |
| loop_or_cycling | 3 | noise+action+moe_action | absolute | 42 | 26 | 0.4383 | 1.2066 | noise+action | -0.1156 | -0.3581 | -0.1018 |
| loop_or_cycling | 3 | noise+action+moe_action+state | absolute | 42 | 26 | 0.6008 | 2.7833 | noise+action | -0.2781 | -0.8616 | -0.2643 |
| loop_or_cycling | 3 | noise+action_within_snapshot | within_snapshot_centered | 42 | 26 | 0.3960 | 1.0632 | noise+action_within_snapshot | -0.0733 | -0.2271 | 0.0000 |
| loop_or_cycling | 3 | moe_action_within_snapshot | within_snapshot_centered | 42 | 26 | 0.4246 | 1.1044 | noise+action_within_snapshot | -0.1019 | -0.3158 | -0.0286 |
| loop_or_cycling | 3 | moe_state_within_snapshot | within_snapshot_centered | 42 | 26 | 0.5938 | 5.7887 | noise+action_within_snapshot | -0.2711 | -0.8402 | -0.1979 |
| loop_or_cycling | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 26 | 0.4809 | 2.5683 | noise+action_within_snapshot | -0.1582 | -0.4902 | -0.0849 |
| loop_or_cycling | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 26 | 0.4269 | 1.1101 | noise+action_within_snapshot | -0.1042 | -0.3229 | -0.0309 |
| loop_or_cycling | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 26 | 0.4842 | 2.4904 | noise+action_within_snapshot | -0.1615 | -0.5005 | -0.0882 |
| single_subtask_omission | 1 | failure_rate_only | absolute | 42 | 18 | 0.4800 | 1.2420 | noise+action | 0.0000 | 0.0000 | 0.1770 |
| single_subtask_omission | 1 | noise+action | absolute | 42 | 18 | 0.6570 | 3.1721 | noise+action | -0.1770 | -0.3688 | 0.0000 |
| single_subtask_omission | 1 | moe_action | absolute | 42 | 18 | 0.5178 | 2.9463 | noise+action | -0.0379 | -0.0789 | 0.1391 |
| single_subtask_omission | 1 | moe_state | absolute | 42 | 18 | 0.8247 | 7.8403 | noise+action | -0.3448 | -0.7184 | -0.1678 |
| single_subtask_omission | 1 | moe_action+state | absolute | 42 | 18 | 0.5863 | 4.2627 | noise+action | -0.1063 | -0.2215 | 0.0707 |
| single_subtask_omission | 1 | noise+action+moe_action | absolute | 42 | 18 | 0.5122 | 3.0647 | noise+action | -0.0323 | -0.0672 | 0.1448 |
| single_subtask_omission | 1 | noise+action+moe_action+state | absolute | 42 | 18 | 0.5750 | 4.4100 | noise+action | -0.0950 | -0.1980 | 0.0820 |
| single_subtask_omission | 1 | noise+action_within_snapshot | within_snapshot_centered | 42 | 18 | 0.5200 | 1.4427 | noise+action_within_snapshot | -0.0401 | -0.0835 | 0.0000 |
| single_subtask_omission | 1 | moe_action_within_snapshot | within_snapshot_centered | 42 | 18 | 0.5180 | 1.4136 | noise+action_within_snapshot | -0.0381 | -0.0793 | 0.0020 |
| single_subtask_omission | 1 | moe_state_within_snapshot | within_snapshot_centered | 42 | 18 | 0.5415 | 4.6022 | noise+action_within_snapshot | -0.0616 | -0.1282 | -0.0215 |
| single_subtask_omission | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 18 | 0.4928 | 3.9658 | noise+action_within_snapshot | -0.0128 | -0.0267 | 0.0273 |
| single_subtask_omission | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 18 | 0.5204 | 1.4197 | noise+action_within_snapshot | -0.0405 | -0.0843 | -0.0004 |
| single_subtask_omission | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 18 | 0.4932 | 3.9126 | noise+action_within_snapshot | -0.0133 | -0.0277 | 0.0268 |
| single_subtask_omission | 3 | failure_rate_only | absolute | 42 | 18 | 0.4800 | 1.2420 | noise+action | 0.0000 | 0.0000 | -0.0102 |
| single_subtask_omission | 3 | noise+action | absolute | 42 | 18 | 0.4698 | 1.9106 | noise+action | 0.0102 | 0.0212 | 0.0000 |
| single_subtask_omission | 3 | moe_action | absolute | 42 | 18 | 0.4609 | 2.5193 | noise+action | 0.0190 | 0.0397 | 0.0089 |
| single_subtask_omission | 3 | moe_state | absolute | 42 | 18 | 0.7563 | 4.2946 | noise+action | -0.2764 | -0.5759 | -0.2865 |
| single_subtask_omission | 3 | moe_action+state | absolute | 42 | 18 | 0.5150 | 3.1756 | noise+action | -0.0350 | -0.0730 | -0.0452 |
| single_subtask_omission | 3 | noise+action+moe_action | absolute | 42 | 18 | 0.4478 | 2.4654 | noise+action | 0.0322 | 0.0671 | 0.0220 |
| single_subtask_omission | 3 | noise+action+moe_action+state | absolute | 42 | 18 | 0.4894 | 3.0841 | noise+action | -0.0094 | -0.0196 | -0.0196 |
| single_subtask_omission | 3 | noise+action_within_snapshot | within_snapshot_centered | 42 | 18 | 0.6202 | 2.2692 | noise+action_within_snapshot | -0.1402 | -0.2921 | 0.0000 |
| single_subtask_omission | 3 | moe_action_within_snapshot | within_snapshot_centered | 42 | 18 | 0.6991 | 3.7178 | noise+action_within_snapshot | -0.2191 | -0.4566 | -0.0789 |
| single_subtask_omission | 3 | moe_state_within_snapshot | within_snapshot_centered | 42 | 18 | 0.5016 | 3.3297 | noise+action_within_snapshot | -0.0217 | -0.0452 | 0.1185 |
| single_subtask_omission | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 18 | 0.6239 | 4.0850 | noise+action_within_snapshot | -0.1440 | -0.3000 | -0.0038 |
| single_subtask_omission | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 18 | 0.7012 | 3.7483 | noise+action_within_snapshot | -0.2213 | -0.4611 | -0.0811 |
| single_subtask_omission | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 18 | 0.6258 | 4.0697 | noise+action_within_snapshot | -0.1459 | -0.3039 | -0.0057 |
| non_stagnation_non_loop | 1 | failure_rate_only | absolute | 42 | 7 | 0.1633 | 0.5427 | noise+action | 0.0000 | 0.0000 | 0.0729 |
| non_stagnation_non_loop | 1 | noise+action | absolute | 42 | 7 | 0.2362 | 0.6960 | noise+action | -0.0729 | -0.4465 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action | absolute | 42 | 7 | 0.6420 | 2.1637 | noise+action | -0.4788 | -2.9326 | -0.4059 |
| non_stagnation_non_loop | 1 | moe_state | absolute | 42 | 7 | 0.8560 | 6.4535 | noise+action | -0.6927 | -4.2430 | -0.6198 |
| non_stagnation_non_loop | 1 | moe_action+state | absolute | 42 | 7 | 0.7134 | 6.4315 | noise+action | -0.5502 | -3.3698 | -0.4773 |
| non_stagnation_non_loop | 1 | noise+action+moe_action | absolute | 42 | 7 | 0.6188 | 2.0262 | noise+action | -0.4556 | -2.7905 | -0.3827 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state | absolute | 42 | 7 | 0.7085 | 6.3076 | noise+action | -0.5452 | -3.3396 | -0.4723 |
| non_stagnation_non_loop | 1 | noise+action_within_snapshot | within_snapshot_centered | 42 | 7 | 0.1631 | 0.6249 | noise+action_within_snapshot | 0.0001 | 0.0007 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action_within_snapshot | within_snapshot_centered | 42 | 7 | 0.1872 | 1.0971 | noise+action_within_snapshot | -0.0240 | -0.1468 | -0.0241 |
| non_stagnation_non_loop | 1 | moe_state_within_snapshot | within_snapshot_centered | 42 | 7 | 0.3791 | 4.4963 | noise+action_within_snapshot | -0.2158 | -1.3219 | -0.2159 |
| non_stagnation_non_loop | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 7 | 0.3825 | 4.7573 | noise+action_within_snapshot | -0.2193 | -1.3430 | -0.2194 |
| non_stagnation_non_loop | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 7 | 0.1870 | 1.1680 | noise+action_within_snapshot | -0.0238 | -0.1456 | -0.0239 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 7 | 0.3826 | 4.7562 | noise+action_within_snapshot | -0.2193 | -1.3433 | -0.2194 |
| non_stagnation_non_loop | 3 | failure_rate_only | absolute | 42 | 7 | 0.1633 | 0.5427 | noise+action | 0.0000 | 0.0000 | 0.1894 |
| non_stagnation_non_loop | 3 | noise+action | absolute | 42 | 7 | 0.3526 | 0.9853 | noise+action | -0.1894 | -1.1599 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action | absolute | 42 | 7 | 0.3307 | 0.9492 | noise+action | -0.1675 | -1.0257 | 0.0219 |
| non_stagnation_non_loop | 3 | moe_state | absolute | 42 | 7 | 0.6487 | 6.2574 | noise+action | -0.4854 | -2.9735 | -0.2961 |
| non_stagnation_non_loop | 3 | moe_action+state | absolute | 42 | 7 | 0.5420 | 3.9000 | noise+action | -0.3787 | -2.3196 | -0.1893 |
| non_stagnation_non_loop | 3 | noise+action+moe_action | absolute | 42 | 7 | 0.3357 | 0.9639 | noise+action | -0.1725 | -1.0565 | 0.0169 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state | absolute | 42 | 7 | 0.5394 | 3.8570 | noise+action | -0.3761 | -2.3037 | -0.1867 |
| non_stagnation_non_loop | 3 | noise+action_within_snapshot | within_snapshot_centered | 42 | 7 | 0.1720 | 0.6723 | noise+action_within_snapshot | -0.0087 | -0.0535 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action_within_snapshot | within_snapshot_centered | 42 | 7 | 0.1700 | 0.6361 | noise+action_within_snapshot | -0.0067 | -0.0411 | 0.0020 |
| non_stagnation_non_loop | 3 | moe_state_within_snapshot | within_snapshot_centered | 42 | 7 | 0.4074 | 3.8898 | noise+action_within_snapshot | -0.2442 | -1.4956 | -0.2354 |
| non_stagnation_non_loop | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 42 | 7 | 0.2827 | 2.3763 | noise+action_within_snapshot | -0.1195 | -0.7317 | -0.1107 |
| non_stagnation_non_loop | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 42 | 7 | 0.1695 | 0.6303 | noise+action_within_snapshot | -0.0063 | -0.0384 | 0.0025 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 42 | 7 | 0.2820 | 2.2708 | noise+action_within_snapshot | -0.1188 | -0.7276 | -0.1101 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_subtask_omission | 1 | state | back_12_15 | 6 | top1_mass | 1.0000 | 0.3333 | 0.6667 | 0.5000 | 1.0000 | 0.0909 | 0.3636 | 6 | 6 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 0 | top1_mass | 0.4000 | 0.0000 | 0.4000 | 0.2500 | 0.4000 | 0.0909 | 0.4545 | 10 | 10 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 1 | top1_mass | 0.4000 | 0.0000 | 0.4000 | 0.2500 | 0.4000 | 0.0909 | 0.4545 | 10 | 10 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 2 | top1_mass | 0.4000 | 0.0000 | 0.4000 | 0.2500 | 0.4000 | 0.0909 | 0.4545 | 10 | 10 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | entropy | 0.9000 | 0.4000 | 0.5000 | 0.4225 | 0.9438 | 0.0909 | 0.6364 | 10 | 10 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.4000 | 0.9000 | -0.5000 | -0.9437 | -0.4225 | 0.0909 | 0.6364 | 10 | 10 |
| loop_or_cycling | 3 | action | front_2_5 | 0 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.4812 | -0.3075 | 0.0909 | 0.7273 | 10 | 10 |
| stagnation | 1 | state | front_2_5 | 4 | top1_mass | 0.1667 | 0.8333 | -0.6667 | -0.7500 | -0.5375 | 0.0909 | 0.8182 | 6 | 6 |
| stagnation | 1 | state | front_2_5 | 7 | top1_mass | 0.3333 | 1.0000 | -0.6667 | -0.9250 | -0.5000 | 0.0909 | 0.8182 | 6 | 6 |
| single_subtask_omission | 3 | action | back_12_15 | 0 | entropy | 1.0000 | 0.5000 | 0.5000 | 0.3063 | 1.0000 | 0.0909 | 0.8182 | 6 | 6 |
| single_subtask_omission | 3 | action | back_12_15 | 0 | top1_mass | 0.5000 | 1.0000 | -0.5000 | -1.0000 | -0.3062 | 0.0909 | 0.8182 | 6 | 6 |
| single_subtask_omission | 3 | action | back_12_15 | 1 | top1_mass | 0.5000 | 1.0000 | -0.5000 | -1.0000 | -0.3062 | 0.0909 | 0.8182 | 6 | 6 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
