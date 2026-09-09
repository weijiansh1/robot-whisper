# Rolling-star K=16 experiment

## Dataset

- 96 terminal branches from 6 committed snapshots; 41 success and 55 failure.
- 4 snapshots contain matched success/failure siblings.
- 4467 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 30
- `single_subtask_omission`: 12
- `drop_or_regrasp`: 6
- `goal_contact_near_miss`: 3
- `subtask_undo`: 3
- `active_retry`: 1

Stagnation or loop/cycling covers 48/55 failures; 7 failures require other physical mechanisms: {"active_retry": 1, "goal_contact_near_miss": 3, "single_subtask_omission": 3}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00111.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (snapshots held out together):

| prefix_queries | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | snapshot_rate_only | absolute | 96 | 4 | 0.3211 | 0.8439 | 0.6406 | 0.6406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1152 | -0.1511 | 0.4046 |
| 1 | noise | absolute | 96 | 4 | 0.3576 | 0.9399 | 0.2500 | 0.6406 | -0.3906 | -0.7656 | -0.0156 | -0.0365 | -0.1136 | noise+action | -0.0365 | -0.0607 | -0.0133 | 0.0787 | -0.2160 | 0.3713 |
| 1 | action | absolute | 96 | 4 | 0.4153 | 1.3956 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0942 | -0.2933 | noise+action | -0.0942 | -0.3608 | 0.2031 | 0.0210 | -0.0203 | 0.0611 |
| 1 | noise+action | absolute | 96 | 4 | 0.4363 | 1.5162 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.1152 | -0.3588 | noise+action | -0.1152 | -0.3982 | 0.1572 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action | absolute | 96 | 4 | 0.3104 | 0.9968 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | 0.0107 | 0.0333 | noise+action | 0.0107 | -0.1199 | 0.1452 | 0.1259 | -0.1132 | 0.4043 |
| 1 | moe_state | absolute | 96 | 4 | 0.3955 | 1.1502 | 0.5000 | 0.6406 | -0.1406 | -0.3906 | 0.0625 | -0.0744 | -0.2317 | noise+action | -0.0744 | -0.2570 | 0.1007 | 0.0408 | -0.3672 | 0.4779 |
| 1 | moe_action+state | absolute | 96 | 4 | 0.4309 | 1.6350 | 0.5000 | 0.6406 | -0.1406 | -0.3906 | 0.0625 | -0.1098 | -0.3420 | noise+action | -0.1098 | -0.3045 | 0.0946 | 0.0054 | -0.3435 | 0.4199 |
| 1 | noise+action+moe_action | absolute | 96 | 4 | 0.3174 | 1.0159 | 0.5000 | 0.6406 | -0.1406 | -0.3684 | 0.0625 | 0.0037 | 0.0115 | noise+action | 0.0037 | -0.1328 | 0.1425 | 0.1189 | -0.1123 | 0.4289 |
| 1 | noise+action+moe_action+state | absolute | 96 | 4 | 0.4391 | 1.6883 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.1180 | -0.3675 | noise+action | -0.1180 | -0.2977 | 0.0711 | -0.0028 | -0.3547 | 0.4076 |
| 1 | noise_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3283 | 0.8639 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0072 | -0.0224 | noise+action_within_snapshot | -0.0072 | -0.0115 | -0.0030 | -0.0025 | -0.0050 | -0.0003 |
| 1 | action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3299 | 0.8682 | 0.5000 | 0.6406 | -0.1406 | -0.3125 | 0.0625 | -0.0088 | -0.0275 | noise+action_within_snapshot | -0.0088 | -0.0140 | -0.0035 | -0.0041 | -0.0101 | 0.0020 |
| 1 | noise+action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3258 | 0.8591 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0047 | -0.0146 | noise+action_within_snapshot | -0.0047 | -0.0093 | -0.0007 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3398 | 0.8942 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.3594 | -0.0187 | -0.0581 | noise+action_within_snapshot | -0.0187 | -0.0297 | -0.0093 | -0.0140 | -0.0236 | -0.0049 |
| 1 | moe_state_within_snapshot | within_snapshot_centered | 96 | 4 | 0.5147 | 3.2481 | 0.7500 | 0.6406 | 0.1094 | -0.1875 | 0.5781 | -0.1936 | -0.6031 | noise+action_within_snapshot | -0.1936 | -0.3677 | -0.0196 | -0.1889 | -0.3625 | -0.0285 |
| 1 | moe_action+state_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3878 | 1.2046 | 0.7500 | 0.6406 | 0.1094 | -0.0938 | 0.4375 | -0.0667 | -0.2076 | noise+action_within_snapshot | -0.0667 | -0.1316 | -0.0138 | -0.0620 | -0.1220 | -0.0105 |
| 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3393 | 0.8929 | 0.2500 | 0.6406 | -0.3906 | -0.7902 | -0.0625 | -0.0183 | -0.0568 | noise+action_within_snapshot | -0.0183 | -0.0291 | -0.0090 | -0.0135 | -0.0225 | -0.0046 |
| 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3850 | 1.1753 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0639 | -0.1990 | noise+action_within_snapshot | -0.0639 | -0.1233 | -0.0124 | -0.0592 | -0.1191 | -0.0074 |
| 3 | snapshot_rate_only | absolute | 96 | 4 | 0.3211 | 0.8439 | 0.6406 | 0.6406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1801 | -0.0377 | 0.3872 |
| 3 | noise | absolute | 96 | 4 | 0.3492 | 0.9214 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0281 | -0.0875 | noise+action | -0.0281 | -0.0499 | -0.0049 | 0.1520 | -0.0961 | 0.4075 |
| 3 | action | absolute | 96 | 4 | 0.3833 | 1.1284 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0622 | -0.1938 | noise+action | -0.0622 | -0.3229 | 0.1457 | 0.1178 | 0.0501 | 0.1793 |
| 3 | noise+action | absolute | 96 | 4 | 0.5012 | 1.6405 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.1801 | -0.5608 | noise+action | -0.1801 | -0.4206 | 0.0668 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action | absolute | 96 | 4 | 0.2974 | 1.3271 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | 0.0237 | 0.0738 | noise+action | 0.0237 | -0.2044 | 0.2195 | 0.2038 | -0.0471 | 0.5518 |
| 3 | moe_state | absolute | 96 | 4 | 0.5118 | 3.4171 | 0.7500 | 0.6406 | 0.1094 | -0.3125 | 0.5781 | -0.1907 | -0.5940 | noise+action | -0.1907 | -0.3819 | 0.0284 | -0.0107 | -0.3818 | 0.2837 |
| 3 | moe_action+state | absolute | 96 | 4 | 0.3981 | 1.4675 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0770 | -0.2398 | noise+action | -0.0770 | -0.2670 | 0.1121 | 0.1031 | -0.2159 | 0.5228 |
| 3 | noise+action+moe_action | absolute | 96 | 4 | 0.2939 | 1.5324 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | 0.0272 | 0.0847 | noise+action | 0.0272 | -0.2369 | 0.2398 | 0.2073 | -0.0425 | 0.5443 |
| 3 | noise+action+moe_action+state | absolute | 96 | 4 | 0.3957 | 1.4196 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0746 | -0.2325 | noise+action | -0.0746 | -0.2728 | 0.1493 | 0.1054 | -0.2135 | 0.4658 |
| 3 | noise_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3337 | 0.8754 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0126 | -0.0393 | noise+action_within_snapshot | -0.0126 | -0.0194 | -0.0058 | -0.0069 | -0.0139 | -0.0001 |
| 3 | action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3276 | 0.8635 | 1.0000 | 0.6406 | 0.3594 | 0.0625 | 0.6562 | -0.0065 | -0.0204 | noise+action_within_snapshot | -0.0065 | -0.0134 | 0.0002 | -0.0009 | -0.0058 | 0.0044 |
| 3 | noise+action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3268 | 0.8584 | 0.7500 | 0.6406 | 0.1094 | -0.1719 | 0.4375 | -0.0057 | -0.0177 | noise+action_within_snapshot | -0.0057 | -0.0098 | -0.0015 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3313 | 0.8709 | 0.5000 | 0.6406 | -0.1406 | -0.3906 | 0.0625 | -0.0102 | -0.0318 | noise+action_within_snapshot | -0.0102 | -0.0181 | -0.0028 | -0.0045 | -0.0109 | 0.0016 |
| 3 | moe_state_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3414 | 0.9727 | 1.0000 | 0.6406 | 0.3594 | 0.0625 | 0.6562 | -0.0203 | -0.0632 | noise+action_within_snapshot | -0.0203 | -0.0541 | -0.0004 | -0.0146 | -0.0472 | 0.0037 |
| 3 | moe_action+state_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3322 | 0.8737 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0111 | -0.0345 | noise+action_within_snapshot | -0.0111 | -0.0277 | 0.0010 | -0.0054 | -0.0203 | 0.0030 |
| 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3307 | 0.8694 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0096 | -0.0300 | noise+action_within_snapshot | -0.0096 | -0.0173 | -0.0025 | -0.0039 | -0.0107 | 0.0014 |
| 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 96 | 4 | 0.3319 | 0.8731 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0108 | -0.0338 | noise+action_within_snapshot | -0.0108 | -0.0260 | 0.0004 | -0.0052 | -0.0204 | 0.0031 |

Snapshot difficulty (one row per rolling state; leave-one-snapshot-out):

| model | snapshots | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high | mse_gain_vs_policy_state | mse_gain_vs_policy_state_ci95_low | mse_gain_vs_policy_state_ci95_high | mse_gain_vs_sim_state | mse_gain_vs_sim_state_ci95_low | mse_gain_vs_sim_state_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean_moe_action | 6 | 10.0000 | 0.0976 | 0.2948 | 0.1225 | 0.0230 | 0.2219 | 0.1787 | 0.0422 | 0.3219 | 0.0668 | -0.0143 | 0.1530 |
| mean_moe_action+state | 6 | 10.0000 | 0.1373 | 0.3343 | 0.0828 | 0.0005 | 0.1729 | 0.1390 | 0.0404 | 0.2648 | 0.0271 | -0.0476 | 0.1145 |
| policy_state+mean_moe_action+state | 6 | 10.0000 | 0.1373 | 0.3343 | 0.0828 | -0.0047 | 0.1743 | 0.1390 | 0.0363 | 0.2648 | 0.0271 | -0.0534 | 0.1153 |
| sim_state+mean_moe_action+state | 6 | 10.0000 | 0.1379 | 0.3348 | 0.0823 | 0.0004 | 0.1642 | 0.1385 | 0.0347 | 0.2551 | 0.0265 | -0.0480 | 0.1141 |
| initial_sim_state | 6 | 10.0000 | 0.1644 | 0.3358 | 0.0558 | -0.0713 | 0.1779 | 0.1119 | -0.0475 | 0.3088 | 0.0000 | 0.0000 | 0.0000 |
| mean_moe_state | 6 | 10.0000 | 0.1823 | 0.3550 | 0.0378 | -0.0354 | 0.1396 | 0.0940 | 0.0083 | 0.1676 | -0.0179 | -0.1392 | 0.0811 |
| as_moe | 6 | 10.0000 | 0.2202 | 0.4125 | 0.0000 | -0.0000 | 0.0000 | 0.0562 | -0.0509 | 0.1625 | -0.0558 | -0.1684 | 0.0657 |
| mean_rate | 6 | NA | 0.2202 | 0.4125 | 0.0000 | 0.0000 | 0.0000 | 0.0562 | -0.0443 | 0.1533 | -0.0558 | -0.1834 | 0.0661 |
| initial_policy_state | 6 | 10.0000 | 0.2763 | 0.4706 | -0.0562 | -0.1659 | 0.0413 | 0.0000 | 0.0000 | 0.0000 | -0.1119 | -0.3207 | 0.0531 |
| mean_output_action | 6 | 10.0000 | 0.3175 | 0.5016 | -0.0973 | -0.4280 | 0.1479 | -0.0412 | -0.3809 | 0.2425 | -0.1531 | -0.3177 | -0.0191 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | back_12_15 | 8 | top1_mass | 0.1875 | 0.5000 | -0.3125 | -0.6250 | -0.0625 | 0.0339 | 0.8822 | 16 | 16 |
| 1 | state | back_12_15 | 7 | entropy | 0.4375 | 0.1875 | 0.2500 | -0.1250 | 0.7828 | 0.0838 | 0.9980 | 16 | 16 |
| 1 | state | front_2_5 | 8 | entropy | 0.4375 | 0.1875 | 0.2500 | -0.1250 | 0.6250 | 0.0978 | 0.9980 | 16 | 16 |
| 1 | state | back_12_15 | 3 | entropy | 0.5000 | 0.2500 | 0.2500 | 0.1250 | 0.4375 | 0.0998 | 0.9980 | 16 | 16 |
| 1 | state | back_12_15 | 0 | top1_mass | 0.1875 | 0.4375 | -0.2500 | -0.5000 | 0.0000 | 0.1078 | 0.9980 | 16 | 16 |
| 1 | action | back_12_15 | 0 | entropy | 0.2500 | 0.5000 | -0.2500 | -0.4375 | -0.0625 | 0.1138 | 0.9980 | 16 | 16 |
| 1 | action | front_2_5 | 7 | top1_mass | 0.4375 | 0.2500 | 0.1875 | 0.0000 | 0.3750 | 0.2036 | 1.0000 | 16 | 16 |
| 1 | action | front_2_5 | 6 | top1_mass | 0.4375 | 0.2500 | 0.1875 | 0.0000 | 0.3750 | 0.2236 | 1.0000 | 16 | 16 |
| 1 | action | back_12_15 | 6 | top1_mass | 0.1875 | 0.3750 | -0.1875 | -0.3750 | 0.0000 | 0.2375 | 1.0000 | 16 | 16 |
| 1 | action | front_2_5 | 9 | token_dispersion | 0.5000 | 0.3125 | 0.1875 | -0.1250 | 0.5953 | 0.2395 | 1.0000 | 16 | 16 |
| 1 | state | back_12_15 | 2 | top1_mass | 0.3125 | 0.5000 | -0.1875 | -0.3750 | 0.0000 | 0.2415 | 1.0000 | 16 | 16 |
| 1 | state | back_12_15 | 8 | consensus_distance | 0.4375 | 0.2500 | 0.1875 | -0.1250 | 0.5953 | 0.2435 | 1.0000 | 16 | 16 |

Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):

| prefix_queries | token_family | layer_group | denoise_step | expert | failure_minus_success_probability | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_mixed_snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | action | front_2_5 | 0 | 0 | 0.0004 | -0.0002 | 0.0010 | 0.0659 | 0.3253 | 4 |
| 1 | action | front_2_5 | 1 | 0 | 0.0003 | -0.0002 | 0.0010 | 0.0878 | 0.4351 | 4 |
| 1 | action | front_2_5 | 4 | 14 | -0.0003 | -0.0005 | -0.0002 | 0.0020 | 0.4431 | 4 |
| 1 | action | front_2_5 | 9 | 12 | 0.0003 | -0.0000 | 0.0008 | 0.0938 | 0.5389 | 4 |
| 1 | action | front_2_5 | 5 | 14 | -0.0003 | -0.0005 | -0.0002 | 0.0040 | 0.5409 | 4 |
| 1 | action | front_2_5 | 7 | 12 | 0.0003 | -0.0000 | 0.0008 | 0.1038 | 0.6467 | 4 |
| 1 | action | front_2_5 | 3 | 14 | -0.0003 | -0.0005 | -0.0002 | 0.0080 | 0.6747 | 4 |
| 1 | action | front_2_5 | 8 | 12 | 0.0003 | -0.0001 | 0.0008 | 0.1357 | 0.7106 | 4 |
| 1 | action | front_2_5 | 2 | 14 | -0.0003 | -0.0004 | -0.0002 | 0.0220 | 0.8343 | 4 |
| 1 | action | front_2_5 | 6 | 14 | -0.0003 | -0.0004 | -0.0001 | 0.0160 | 0.8423 | 4 |
| 1 | action | front_2_5 | 6 | 12 | 0.0003 | 0.0000 | 0.0006 | 0.1038 | 0.8443 | 4 |
| 1 | action | front_2_5 | 9 | 31 | -0.0002 | -0.0005 | -0.0001 | 0.0279 | 0.8583 | 4 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | feature_scope | n_failure_branches | n_type_positive | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 1 | failure_rate_only | absolute | 55 | 14 | 0.2359 | 0.7042 | noise+action | 0.0000 | 0.0000 | 0.0587 |
| stagnation | 1 | noise+action | absolute | 55 | 14 | 0.2946 | 1.2180 | noise+action | -0.0587 | -0.2488 | 0.0000 |
| stagnation | 1 | moe_action | absolute | 55 | 14 | 0.2912 | 1.5926 | noise+action | -0.0553 | -0.2345 | 0.0034 |
| stagnation | 1 | moe_state | absolute | 55 | 14 | 0.3202 | 1.4574 | noise+action | -0.0842 | -0.3571 | -0.0255 |
| stagnation | 1 | moe_action+state | absolute | 55 | 14 | 0.2941 | 1.7460 | noise+action | -0.0582 | -0.2465 | 0.0005 |
| stagnation | 1 | noise+action+moe_action | absolute | 55 | 14 | 0.2904 | 1.6162 | noise+action | -0.0544 | -0.2307 | 0.0043 |
| stagnation | 1 | noise+action+moe_action+state | absolute | 55 | 14 | 0.2985 | 1.7752 | noise+action | -0.0626 | -0.2654 | -0.0039 |
| stagnation | 1 | noise+action_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2506 | 0.7666 | noise+action_within_snapshot | -0.0147 | -0.0621 | 0.0000 |
| stagnation | 1 | moe_action_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2509 | 0.7349 | noise+action_within_snapshot | -0.0150 | -0.0634 | -0.0003 |
| stagnation | 1 | moe_state_within_snapshot | within_snapshot_centered | 55 | 14 | 0.3192 | 3.6377 | noise+action_within_snapshot | -0.0833 | -0.3530 | -0.0686 |
| stagnation | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2872 | 3.0310 | noise+action_within_snapshot | -0.0513 | -0.2174 | -0.0366 |
| stagnation | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2532 | 0.7477 | noise+action_within_snapshot | -0.0172 | -0.0730 | -0.0026 |
| stagnation | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2873 | 3.0316 | noise+action_within_snapshot | -0.0514 | -0.2179 | -0.0368 |
| stagnation | 3 | failure_rate_only | absolute | 55 | 14 | 0.2359 | 0.7042 | noise+action | 0.0000 | 0.0000 | 0.0433 |
| stagnation | 3 | noise+action | absolute | 55 | 14 | 0.2792 | 1.4803 | noise+action | -0.0433 | -0.1833 | 0.0000 |
| stagnation | 3 | moe_action | absolute | 55 | 14 | 0.2584 | 2.8887 | noise+action | -0.0225 | -0.0953 | 0.0208 |
| stagnation | 3 | moe_state | absolute | 55 | 14 | 0.4478 | 3.6121 | noise+action | -0.2119 | -0.8981 | -0.1686 |
| stagnation | 3 | moe_action+state | absolute | 55 | 14 | 0.3003 | 3.6678 | noise+action | -0.0643 | -0.2727 | -0.0211 |
| stagnation | 3 | noise+action+moe_action | absolute | 55 | 14 | 0.2606 | 2.9901 | noise+action | -0.0246 | -0.1044 | 0.0186 |
| stagnation | 3 | noise+action+moe_action+state | absolute | 55 | 14 | 0.2956 | 3.6574 | noise+action | -0.0596 | -0.2528 | -0.0164 |
| stagnation | 3 | noise+action_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2449 | 0.7607 | noise+action_within_snapshot | -0.0090 | -0.0381 | 0.0000 |
| stagnation | 3 | moe_action_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2502 | 0.7588 | noise+action_within_snapshot | -0.0143 | -0.0604 | -0.0053 |
| stagnation | 3 | moe_state_within_snapshot | within_snapshot_centered | 55 | 14 | 0.3264 | 3.4132 | noise+action_within_snapshot | -0.0905 | -0.3836 | -0.0815 |
| stagnation | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2790 | 2.7708 | noise+action_within_snapshot | -0.0430 | -0.1824 | -0.0340 |
| stagnation | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2501 | 0.7628 | noise+action_within_snapshot | -0.0141 | -0.0600 | -0.0051 |
| stagnation | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 14 | 0.2787 | 2.6329 | noise+action_within_snapshot | -0.0428 | -0.1812 | -0.0338 |
| loop_or_cycling | 1 | failure_rate_only | absolute | 55 | 39 | 0.2593 | 0.7365 | noise+action | 0.0000 | 0.0000 | 0.0858 |
| loop_or_cycling | 1 | noise+action | absolute | 55 | 39 | 0.3451 | 1.1012 | noise+action | -0.0858 | -0.3309 | 0.0000 |
| loop_or_cycling | 1 | moe_action | absolute | 55 | 39 | 0.3276 | 1.2850 | noise+action | -0.0683 | -0.2633 | 0.0175 |
| loop_or_cycling | 1 | moe_state | absolute | 55 | 39 | 0.1989 | 0.6144 | noise+action | 0.0604 | 0.2328 | 0.1462 |
| loop_or_cycling | 1 | moe_action+state | absolute | 55 | 39 | 0.2448 | 0.7430 | noise+action | 0.0145 | 0.0560 | 0.1003 |
| loop_or_cycling | 1 | noise+action+moe_action | absolute | 55 | 39 | 0.3299 | 1.2454 | noise+action | -0.0706 | -0.2723 | 0.0152 |
| loop_or_cycling | 1 | noise+action+moe_action+state | absolute | 55 | 39 | 0.2384 | 0.7221 | noise+action | 0.0209 | 0.0805 | 0.1067 |
| loop_or_cycling | 1 | noise+action_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2746 | 0.7906 | noise+action_within_snapshot | -0.0154 | -0.0592 | 0.0000 |
| loop_or_cycling | 1 | moe_action_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2957 | 0.8466 | noise+action_within_snapshot | -0.0364 | -0.1405 | -0.0211 |
| loop_or_cycling | 1 | moe_state_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2225 | 2.2722 | noise+action_within_snapshot | 0.0368 | 0.1420 | 0.0522 |
| loop_or_cycling | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 39 | 0.3879 | 4.0181 | noise+action_within_snapshot | -0.1286 | -0.4960 | -0.1132 |
| loop_or_cycling | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2954 | 0.8565 | noise+action_within_snapshot | -0.0361 | -0.1393 | -0.0208 |
| loop_or_cycling | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 39 | 0.3871 | 4.1098 | noise+action_within_snapshot | -0.1278 | -0.4931 | -0.1125 |
| loop_or_cycling | 3 | failure_rate_only | absolute | 55 | 39 | 0.2593 | 0.7365 | noise+action | 0.0000 | 0.0000 | 0.0578 |
| loop_or_cycling | 3 | noise+action | absolute | 55 | 39 | 0.3171 | 1.1367 | noise+action | -0.0578 | -0.2230 | 0.0000 |
| loop_or_cycling | 3 | moe_action | absolute | 55 | 39 | 0.3224 | 1.8971 | noise+action | -0.0631 | -0.2435 | -0.0053 |
| loop_or_cycling | 3 | moe_state | absolute | 55 | 39 | 0.5182 | 3.8474 | noise+action | -0.2589 | -0.9985 | -0.2011 |
| loop_or_cycling | 3 | moe_action+state | absolute | 55 | 39 | 0.4715 | 2.9428 | noise+action | -0.2122 | -0.8184 | -0.1544 |
| loop_or_cycling | 3 | noise+action+moe_action | absolute | 55 | 39 | 0.3267 | 1.9100 | noise+action | -0.0674 | -0.2601 | -0.0096 |
| loop_or_cycling | 3 | noise+action+moe_action+state | absolute | 55 | 39 | 0.4713 | 3.0314 | noise+action | -0.2120 | -0.8177 | -0.1542 |
| loop_or_cycling | 3 | noise+action_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2740 | 0.8339 | noise+action_within_snapshot | -0.0147 | -0.0567 | 0.0000 |
| loop_or_cycling | 3 | moe_action_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2874 | 0.8430 | noise+action_within_snapshot | -0.0281 | -0.1085 | -0.0134 |
| loop_or_cycling | 3 | moe_state_within_snapshot | within_snapshot_centered | 55 | 39 | 0.3641 | 3.3149 | noise+action_within_snapshot | -0.1048 | -0.4043 | -0.0901 |
| loop_or_cycling | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2816 | 2.3834 | noise+action_within_snapshot | -0.0223 | -0.0859 | -0.0076 |
| loop_or_cycling | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2871 | 0.8407 | noise+action_within_snapshot | -0.0278 | -0.1072 | -0.0131 |
| loop_or_cycling | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 39 | 0.2813 | 2.3386 | noise+action_within_snapshot | -0.0220 | -0.0850 | -0.0073 |
| drop_or_regrasp | 1 | failure_rate_only | absolute | 55 | 9 | 0.1447 | 0.4756 | noise+action | 0.0000 | 0.0000 | 0.0314 |
| drop_or_regrasp | 1 | noise+action | absolute | 55 | 9 | 0.1761 | 0.5640 | noise+action | -0.0314 | -0.2168 | 0.0000 |
| drop_or_regrasp | 1 | moe_action | absolute | 55 | 9 | 0.1698 | 0.5599 | noise+action | -0.0251 | -0.1732 | 0.0063 |
| drop_or_regrasp | 1 | moe_state | absolute | 55 | 9 | 0.2286 | 1.5009 | noise+action | -0.0839 | -0.5794 | -0.0525 |
| drop_or_regrasp | 1 | moe_action+state | absolute | 55 | 9 | 0.2387 | 2.4720 | noise+action | -0.0939 | -0.6491 | -0.0626 |
| drop_or_regrasp | 1 | noise+action+moe_action | absolute | 55 | 9 | 0.1575 | 0.5305 | noise+action | -0.0128 | -0.0882 | 0.0186 |
| drop_or_regrasp | 1 | noise+action+moe_action+state | absolute | 55 | 9 | 0.2366 | 2.3599 | noise+action | -0.0918 | -0.6343 | -0.0604 |
| drop_or_regrasp | 1 | noise+action_within_snapshot | within_snapshot_centered | 55 | 9 | 0.1562 | 0.5617 | noise+action_within_snapshot | -0.0114 | -0.0789 | 0.0000 |
| drop_or_regrasp | 1 | moe_action_within_snapshot | within_snapshot_centered | 55 | 9 | 0.1612 | 0.5570 | noise+action_within_snapshot | -0.0165 | -0.1138 | -0.0051 |
| drop_or_regrasp | 1 | moe_state_within_snapshot | within_snapshot_centered | 55 | 9 | 0.3780 | 4.1848 | noise+action_within_snapshot | -0.2332 | -1.6115 | -0.2218 |
| drop_or_regrasp | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 9 | 0.3720 | 3.9593 | noise+action_within_snapshot | -0.2273 | -1.5701 | -0.2158 |
| drop_or_regrasp | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 9 | 0.1628 | 0.5604 | noise+action_within_snapshot | -0.0181 | -0.1247 | -0.0066 |
| drop_or_regrasp | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 9 | 0.3717 | 3.9531 | noise+action_within_snapshot | -0.2270 | -1.5683 | -0.2156 |
| drop_or_regrasp | 3 | failure_rate_only | absolute | 55 | 9 | 0.1447 | 0.4756 | noise+action | 0.0000 | 0.0000 | 0.0151 |
| drop_or_regrasp | 3 | noise+action | absolute | 55 | 9 | 0.1599 | 0.5124 | noise+action | -0.0151 | -0.1046 | 0.0000 |
| drop_or_regrasp | 3 | moe_action | absolute | 55 | 9 | 0.1463 | 0.5240 | noise+action | -0.0015 | -0.0105 | 0.0136 |
| drop_or_regrasp | 3 | moe_state | absolute | 55 | 9 | 0.1881 | 0.7076 | noise+action | -0.0433 | -0.2993 | -0.0282 |
| drop_or_regrasp | 3 | moe_action+state | absolute | 55 | 9 | 0.1528 | 0.7282 | noise+action | -0.0081 | -0.0557 | 0.0071 |
| drop_or_regrasp | 3 | noise+action+moe_action | absolute | 55 | 9 | 0.1454 | 0.5182 | noise+action | -0.0007 | -0.0046 | 0.0145 |
| drop_or_regrasp | 3 | noise+action+moe_action+state | absolute | 55 | 9 | 0.1536 | 0.7584 | noise+action | -0.0089 | -0.0612 | 0.0063 |
| drop_or_regrasp | 3 | noise+action_within_snapshot | within_snapshot_centered | 55 | 9 | 0.1585 | 0.5564 | noise+action_within_snapshot | -0.0138 | -0.0953 | 0.0000 |
| drop_or_regrasp | 3 | moe_action_within_snapshot | within_snapshot_centered | 55 | 9 | 0.1641 | 0.6506 | noise+action_within_snapshot | -0.0193 | -0.1335 | -0.0055 |
| drop_or_regrasp | 3 | moe_state_within_snapshot | within_snapshot_centered | 55 | 9 | 0.3343 | 3.4058 | noise+action_within_snapshot | -0.1896 | -1.3099 | -0.1758 |
| drop_or_regrasp | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 9 | 0.2308 | 0.8006 | noise+action_within_snapshot | -0.0861 | -0.5946 | -0.0723 |
| drop_or_regrasp | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 9 | 0.1637 | 0.6512 | noise+action_within_snapshot | -0.0190 | -0.1309 | -0.0052 |
| drop_or_regrasp | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 9 | 0.3213 | 1.2057 | noise+action_within_snapshot | -0.1766 | -1.2199 | -0.1628 |
| single_subtask_omission | 1 | failure_rate_only | absolute | 55 | 19 | 0.3517 | 0.9800 | noise+action | 0.0000 | 0.0000 | 0.1750 |
| single_subtask_omission | 1 | noise+action | absolute | 55 | 19 | 0.5267 | 2.7215 | noise+action | -0.1750 | -0.4976 | 0.0000 |
| single_subtask_omission | 1 | moe_action | absolute | 55 | 19 | 0.4681 | 2.8755 | noise+action | -0.1164 | -0.3311 | 0.0586 |
| single_subtask_omission | 1 | moe_state | absolute | 55 | 19 | 0.5776 | 2.7966 | noise+action | -0.2260 | -0.6425 | -0.0510 |
| single_subtask_omission | 1 | moe_action+state | absolute | 55 | 19 | 0.5161 | 2.5389 | noise+action | -0.1644 | -0.4675 | 0.0106 |
| single_subtask_omission | 1 | noise+action+moe_action | absolute | 55 | 19 | 0.4679 | 3.2978 | noise+action | -0.1163 | -0.3306 | 0.0587 |
| single_subtask_omission | 1 | noise+action+moe_action+state | absolute | 55 | 19 | 0.5120 | 2.7065 | noise+action | -0.1604 | -0.4560 | 0.0146 |
| single_subtask_omission | 1 | noise+action_within_snapshot | within_snapshot_centered | 55 | 19 | 0.3781 | 1.1566 | noise+action_within_snapshot | -0.0264 | -0.0750 | 0.0000 |
| single_subtask_omission | 1 | moe_action_within_snapshot | within_snapshot_centered | 55 | 19 | 0.3783 | 1.0753 | noise+action_within_snapshot | -0.0266 | -0.0756 | -0.0002 |
| single_subtask_omission | 1 | moe_state_within_snapshot | within_snapshot_centered | 55 | 19 | 0.3855 | 4.1193 | noise+action_within_snapshot | -0.0338 | -0.0961 | -0.0074 |
| single_subtask_omission | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 19 | 0.3472 | 2.9862 | noise+action_within_snapshot | 0.0044 | 0.0126 | 0.0308 |
| single_subtask_omission | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 19 | 0.3793 | 1.0824 | noise+action_within_snapshot | -0.0277 | -0.0786 | -0.0013 |
| single_subtask_omission | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 19 | 0.3456 | 2.9818 | noise+action_within_snapshot | 0.0060 | 0.0172 | 0.0324 |
| single_subtask_omission | 3 | failure_rate_only | absolute | 55 | 19 | 0.3517 | 0.9800 | noise+action | 0.0000 | 0.0000 | 0.0441 |
| single_subtask_omission | 3 | noise+action | absolute | 55 | 19 | 0.3957 | 1.5136 | noise+action | -0.0441 | -0.1253 | 0.0000 |
| single_subtask_omission | 3 | moe_action | absolute | 55 | 19 | 0.3805 | 1.8304 | noise+action | -0.0289 | -0.0821 | 0.0152 |
| single_subtask_omission | 3 | moe_state | absolute | 55 | 19 | 0.6206 | 2.8882 | noise+action | -0.2690 | -0.7649 | -0.2249 |
| single_subtask_omission | 3 | moe_action+state | absolute | 55 | 19 | 0.5315 | 4.0775 | noise+action | -0.1798 | -0.5113 | -0.1357 |
| single_subtask_omission | 3 | noise+action+moe_action | absolute | 55 | 19 | 0.3575 | 1.6788 | noise+action | -0.0058 | -0.0165 | 0.0383 |
| single_subtask_omission | 3 | noise+action+moe_action+state | absolute | 55 | 19 | 0.5237 | 4.1197 | noise+action | -0.1720 | -0.4892 | -0.1280 |
| single_subtask_omission | 3 | noise+action_within_snapshot | within_snapshot_centered | 55 | 19 | 0.4246 | 1.6852 | noise+action_within_snapshot | -0.0729 | -0.2074 | 0.0000 |
| single_subtask_omission | 3 | moe_action_within_snapshot | within_snapshot_centered | 55 | 19 | 0.4322 | 2.8104 | noise+action_within_snapshot | -0.0806 | -0.2291 | -0.0076 |
| single_subtask_omission | 3 | moe_state_within_snapshot | within_snapshot_centered | 55 | 19 | 0.4003 | 3.0713 | noise+action_within_snapshot | -0.0487 | -0.1384 | 0.0243 |
| single_subtask_omission | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 19 | 0.4522 | 3.7235 | noise+action_within_snapshot | -0.1005 | -0.2858 | -0.0276 |
| single_subtask_omission | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 19 | 0.4334 | 2.8584 | noise+action_within_snapshot | -0.0817 | -0.2324 | -0.0088 |
| single_subtask_omission | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 19 | 0.4530 | 3.6528 | noise+action_within_snapshot | -0.1013 | -0.2880 | -0.0284 |
| non_stagnation_non_loop | 1 | failure_rate_only | absolute | 55 | 7 | 0.1232 | 0.4444 | noise+action | 0.0000 | 0.0000 | 0.1525 |
| non_stagnation_non_loop | 1 | noise+action | absolute | 55 | 7 | 0.2757 | 0.9572 | noise+action | -0.1525 | -1.2381 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action | absolute | 55 | 7 | 0.1040 | 0.4733 | noise+action | 0.0192 | 0.1557 | 0.1717 |
| non_stagnation_non_loop | 1 | moe_state | absolute | 55 | 7 | 0.3228 | 0.9860 | noise+action | -0.1996 | -1.6199 | -0.0470 |
| non_stagnation_non_loop | 1 | moe_action+state | absolute | 55 | 7 | 0.3423 | 2.5008 | noise+action | -0.2191 | -1.7784 | -0.0666 |
| non_stagnation_non_loop | 1 | noise+action+moe_action | absolute | 55 | 7 | 0.1135 | 0.5237 | noise+action | 0.0097 | 0.0788 | 0.1622 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state | absolute | 55 | 7 | 0.3444 | 2.2729 | noise+action | -0.2212 | -1.7952 | -0.0686 |
| non_stagnation_non_loop | 1 | noise+action_within_snapshot | within_snapshot_centered | 55 | 7 | 0.1265 | 0.5201 | noise+action_within_snapshot | -0.0033 | -0.0266 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action_within_snapshot | within_snapshot_centered | 55 | 7 | 0.1462 | 0.9468 | noise+action_within_snapshot | -0.0230 | -0.1870 | -0.0198 |
| non_stagnation_non_loop | 1 | moe_state_within_snapshot | within_snapshot_centered | 55 | 7 | 0.2529 | 2.9900 | noise+action_within_snapshot | -0.1297 | -1.0527 | -0.1264 |
| non_stagnation_non_loop | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 7 | 0.2950 | 3.5207 | noise+action_within_snapshot | -0.1718 | -1.3947 | -0.1685 |
| non_stagnation_non_loop | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 7 | 0.1445 | 0.9591 | noise+action_within_snapshot | -0.0213 | -0.1729 | -0.0180 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 7 | 0.2948 | 3.5076 | noise+action_within_snapshot | -0.1716 | -1.3927 | -0.1683 |
| non_stagnation_non_loop | 3 | failure_rate_only | absolute | 55 | 7 | 0.1232 | 0.4444 | noise+action | 0.0000 | 0.0000 | 0.2353 |
| non_stagnation_non_loop | 3 | noise+action | absolute | 55 | 7 | 0.3585 | 1.1708 | noise+action | -0.2353 | -1.9099 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action | absolute | 55 | 7 | 0.2319 | 1.5561 | noise+action | -0.1088 | -0.8828 | 0.1265 |
| non_stagnation_non_loop | 3 | moe_state | absolute | 55 | 7 | 0.4141 | 4.3008 | noise+action | -0.2909 | -2.3615 | -0.0556 |
| non_stagnation_non_loop | 3 | moe_action+state | absolute | 55 | 7 | 0.4013 | 3.3781 | noise+action | -0.2781 | -2.2573 | -0.0428 |
| non_stagnation_non_loop | 3 | noise+action+moe_action | absolute | 55 | 7 | 0.2461 | 1.6059 | noise+action | -0.1229 | -0.9979 | 0.1124 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state | absolute | 55 | 7 | 0.3981 | 3.6087 | noise+action | -0.2749 | -2.2316 | -0.0396 |
| non_stagnation_non_loop | 3 | noise+action_within_snapshot | within_snapshot_centered | 55 | 7 | 0.1240 | 0.4883 | noise+action_within_snapshot | -0.0008 | -0.0064 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action_within_snapshot | within_snapshot_centered | 55 | 7 | 0.1328 | 1.0595 | noise+action_within_snapshot | -0.0097 | -0.0784 | -0.0089 |
| non_stagnation_non_loop | 3 | moe_state_within_snapshot | within_snapshot_centered | 55 | 7 | 0.2979 | 3.6853 | noise+action_within_snapshot | -0.1747 | -1.4179 | -0.1739 |
| non_stagnation_non_loop | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 55 | 7 | 0.2787 | 1.5105 | noise+action_within_snapshot | -0.1555 | -1.2624 | -0.1547 |
| non_stagnation_non_loop | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 55 | 7 | 0.1327 | 1.0992 | noise+action_within_snapshot | -0.0095 | -0.0770 | -0.0087 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 55 | 7 | 0.2572 | 1.2816 | noise+action_within_snapshot | -0.1340 | -1.0878 | -0.1332 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_subtask_omission | 1 | state | back_12_15 | 6 | top1_mass | 0.7778 | 0.2222 | 0.5556 | 0.3333 | 1.0000 | 0.0020 | 0.1657 | 9 | 9 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 1 | top1_mass | 0.4000 | 0.0000 | 0.4000 | 0.2500 | 1.0000 | 0.0160 | 0.2834 | 10 | 10 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 0 | top1_mass | 0.4000 | 0.0000 | 0.4000 | 0.2500 | 1.0000 | 0.0180 | 0.2834 | 10 | 10 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 2 | top1_mass | 0.4000 | 0.0000 | 0.4000 | 0.2500 | 1.0000 | 0.0200 | 0.2834 | 10 | 10 |
| non_stagnation_non_loop | 1 | state | back_12_15 | 8 | top1_mass | 0.0000 | 0.4000 | -0.4000 | -1.0000 | -0.2500 | 0.0060 | 0.4731 | 10 | 10 |
| stagnation | 3 | action | back_12_15 | 0 | entropy | 0.6667 | 0.1111 | 0.5556 | 0.3333 | 0.7500 | 0.0299 | 0.5070 | 9 | 9 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | entropy | 0.9000 | 0.4000 | 0.5000 | 0.2500 | 1.0000 | 0.0140 | 0.6627 | 10 | 10 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.4000 | 0.9000 | -0.5000 | -1.0000 | -0.2500 | 0.0259 | 0.6627 | 10 | 10 |
| single_subtask_omission | 1 | state | back_12_15 | 4 | top1_mass | 0.7778 | 0.3333 | 0.4444 | 0.2500 | 1.0000 | 0.0220 | 0.8044 | 9 | 9 |
| single_subtask_omission | 1 | state | front_2_5 | 6 | consensus_distance | 0.6667 | 0.2222 | 0.4444 | 0.3333 | 0.5000 | 0.0299 | 0.8044 | 9 | 9 |
| single_subtask_omission | 1 | state | back_12_15 | 3 | consensus_distance | 0.7778 | 0.3333 | 0.4444 | 0.2500 | 1.0000 | 0.0319 | 0.8044 | 9 | 9 |
| single_subtask_omission | 1 | state | back_12_15 | 6 | entropy | 0.3333 | 0.7778 | -0.4444 | -0.5000 | -0.3333 | 0.0319 | 0.8044 | 9 | 9 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
