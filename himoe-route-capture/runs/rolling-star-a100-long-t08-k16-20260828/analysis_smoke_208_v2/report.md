# Rolling-star K=16 experiment

## Dataset

- 208 terminal branches from 13 committed snapshots; 77 success and 131 failure.
- 9 snapshots contain matched success/failure siblings.
- 9669 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.
- Candidate-ID negative control: max failure-rate deviation 0.168, within-snapshot permutation p=0.7451.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 77
- `single_subtask_omission`: 31
- `drop_or_regrasp`: 12
- `subtask_undo`: 5
- `goal_contact_near_miss`: 3
- `stagnation`: 1
- `timeout_other`: 1
- `active_retry`: 1

Stagnation or loop/cycling covers 113/131 failures; 18 failures require other physical mechanisms: {"active_retry": 1, "drop_or_regrasp": 1, "goal_contact_near_miss": 3, "single_subtask_omission": 11, "subtask_undo": 1, "timeout_other": 1}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00596.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Pilot-frozen two-signal family (the transition set is excluded from independent validation):

| signal | split | primary_statistic | primary_p_one_sided | validation_family_size | primary_p_bonferroni | branches | snapshots | mixed_snapshots | successes | failures | failure_minus_success_value | mean_effect_ci95_low | mean_effect_ci95_high | mean_effect_permutation_p_one_sided | top_quartile_failure_rate | bottom_quartile_failure_rate | quartile_failure_rate_difference | quartile_effect_ci95_low | quartile_effect_ci95_high | quartile_effect_permutation_p_one_sided | lowest_value_selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | selection_gain_permutation_p_one_sided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q0_action_front_d0_expert0 | discovery | mean_failure_minus_success | 0.0196 | 2 | 0.0392 | 192 | 12 | 8 | 62 | 130 | 0.0004 | 0.0000 | 0.0007 | 0.0196 | 0.5625 | 0.4688 | 0.0938 | -0.1180 | 0.2742 | 0.1373 | 0.3333 | 0.3229 | 0.0104 | -0.1099 | 0.2082 | 0.7255 |
| q0_action_front_d0_expert0 | transition_excluded | mean_failure_minus_success | 0.6863 | 2 | 1.0000 | 16 | 1 | 1 | 15 | 1 | -0.0003 | -0.0003 | -0.0003 | 0.6863 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7647 | 1.0000 | 0.9375 | 0.0625 | 0.0625 | 0.0625 | 1.0000 |
| q0_action_front_d0_expert0 | validation | mean_failure_minus_success | NA | 2 | NA | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| q0_state_front_d1_top1_mass | discovery | top_minus_bottom_quartile_failure_rate | 0.0196 | 2 | 0.0392 | 192 | 12 | 8 | 62 | 130 | 0.0000 | -0.0000 | 0.0000 | 0.2745 | 0.6250 | 0.3125 | 0.3125 | 0.1875 | 0.4688 | 0.0196 | 0.5833 | 0.3229 | 0.2604 | 0.1020 | 0.4797 | 0.0392 |
| q0_state_front_d1_top1_mass | transition_excluded | top_minus_bottom_quartile_failure_rate | 0.3529 | 2 | 0.7059 | 16 | 1 | 1 | 15 | 1 | 0.0007 | 0.0007 | 0.0007 | 0.6078 | 0.2500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.3529 | 1.0000 | 0.9375 | 0.0625 | 0.0625 | 0.0625 | 0.9608 |
| q0_state_front_d1_top1_mass | validation | top_minus_bottom_quartile_failure_rate | NA | 2 | NA | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

Both coordinates and positive failure directions were frozen before their validation boundaries; only `validation` rows are prospective tests, and `primary_p_bonferroni` corrects the two-signal family.

Cross-validated models (both snapshot-held-out and core worker-held-out checks):

| prefix_queries | cv_scheme | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | leave_one_snapshot_out | snapshot_rate_only | absolute | 208 | 9 | 0.2410 | 0.6756 | 0.5347 | 0.5347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1602 | 0.0889 | 0.2675 |
| 1 | leave_one_snapshot_out | noise | absolute | 208 | 9 | 0.2470 | 0.6894 | 0.5556 | 0.5347 | 0.0208 | -0.2113 | 0.3054 | -0.0060 | -0.0250 | noise+action | -0.0060 | -0.0115 | 0.0005 | 0.1542 | 0.0768 | 0.2530 |
| 1 | leave_one_snapshot_out | action | absolute | 208 | 9 | 0.4946 | 1.5857 | 0.3333 | 0.5347 | -0.2014 | -0.4405 | -0.0602 | -0.2537 | -1.0527 | noise+action | -0.2537 | -0.3825 | -0.1292 | -0.0935 | -0.1566 | 0.0722 |
| 1 | leave_one_snapshot_out | noise+action | absolute | 208 | 9 | 0.4012 | 1.1035 | 0.6667 | 0.5347 | 0.1319 | -0.0247 | 0.3634 | -0.1602 | -0.6649 | noise+action | -0.1602 | -0.2706 | -0.0980 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action | absolute | 208 | 9 | 0.3191 | 1.0183 | 0.4444 | 0.5347 | -0.0903 | -0.2036 | 0.0069 | -0.0782 | -0.3244 | noise+action | -0.0782 | -0.1864 | 0.0560 | 0.0820 | -0.0491 | 0.2907 |
| 1 | leave_one_snapshot_out | moe_state | absolute | 208 | 9 | 0.2219 | 0.7194 | 0.4444 | 0.5347 | -0.0903 | -0.2314 | 0.0177 | 0.0190 | 0.0790 | noise+action | 0.0190 | -0.1004 | 0.1355 | 0.1792 | -0.0047 | 0.3347 |
| 1 | leave_one_snapshot_out | moe_action+state | absolute | 208 | 9 | 0.3538 | 1.2516 | 0.4444 | 0.5347 | -0.0903 | -0.2083 | 0.0076 | -0.1128 | -0.4683 | noise+action | -0.1128 | -0.3335 | 0.0450 | 0.0474 | -0.1157 | 0.2042 |
| 1 | leave_one_snapshot_out | noise+action+moe_action | absolute | 208 | 9 | 0.3314 | 1.0538 | 0.4444 | 0.5347 | -0.0903 | -0.1774 | -0.0016 | -0.0904 | -0.3751 | noise+action | -0.0904 | -0.2463 | 0.0320 | 0.0698 | -0.0813 | 0.2100 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 208 | 9 | 0.3661 | 1.3274 | 0.4444 | 0.5347 | -0.0903 | -0.1944 | -0.0101 | -0.1251 | -0.5194 | noise+action | -0.1251 | -0.2478 | 0.0257 | 0.0351 | -0.1213 | 0.2134 |
| 1 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2434 | 0.6810 | 0.4444 | 0.5347 | -0.0903 | -0.2153 | 0.0007 | -0.0024 | -0.0102 | noise+action_within_snapshot | -0.0024 | -0.0040 | -0.0011 | -0.0020 | -0.0042 | 0.0009 |
| 1 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2432 | 0.6801 | 0.4444 | 0.5347 | -0.0903 | -0.3040 | 0.1566 | -0.0022 | -0.0091 | noise+action_within_snapshot | -0.0022 | -0.0073 | 0.0022 | -0.0017 | -0.0041 | 0.0013 |
| 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2415 | 0.6767 | 0.4444 | 0.5347 | -0.0903 | -0.3425 | 0.1095 | -0.0005 | -0.0021 | noise+action_within_snapshot | -0.0005 | -0.0028 | 0.0016 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2452 | 0.6852 | 0.4444 | 0.5347 | -0.0903 | -0.2299 | -0.0085 | -0.0042 | -0.0174 | noise+action_within_snapshot | -0.0042 | -0.0077 | -0.0004 | -0.0037 | -0.0092 | 0.0009 |
| 1 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3577 | 1.7247 | 0.5556 | 0.5347 | 0.0208 | -0.1720 | 0.1882 | -0.1167 | -0.4844 | noise+action_within_snapshot | -0.1167 | -0.2759 | -0.0053 | -0.1162 | -0.2042 | -0.0154 |
| 1 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2461 | 0.6867 | 0.4444 | 0.5347 | -0.0903 | -0.2236 | -0.0208 | -0.0052 | -0.0215 | noise+action_within_snapshot | -0.0052 | -0.0102 | -0.0012 | -0.0047 | -0.0111 | -0.0000 |
| 1 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2450 | 0.6849 | 0.4444 | 0.5347 | -0.0903 | -0.2191 | -0.0186 | -0.0041 | -0.0169 | noise+action_within_snapshot | -0.0041 | -0.0087 | -0.0001 | -0.0036 | -0.0081 | 0.0024 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2457 | 0.6859 | 0.4444 | 0.5347 | -0.0903 | -0.1790 | -0.0085 | -0.0048 | -0.0198 | noise+action_within_snapshot | -0.0048 | -0.0095 | -0.0008 | -0.0043 | -0.0083 | 0.0007 |
| 3 | leave_one_snapshot_out | snapshot_rate_only | absolute | 208 | 9 | 0.2410 | 0.6756 | 0.5347 | 0.5347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1025 | 0.0279 | 0.2297 |
| 3 | leave_one_snapshot_out | noise | absolute | 208 | 9 | 0.2419 | 0.6779 | 0.6667 | 0.5347 | 0.1319 | -0.0524 | 0.3094 | -0.0010 | -0.0041 | noise+action | -0.0010 | -0.0046 | 0.0042 | 0.1015 | 0.0314 | 0.2347 |
| 3 | leave_one_snapshot_out | action | absolute | 208 | 9 | 0.3018 | 0.9790 | 0.4444 | 0.5347 | -0.0903 | -0.1967 | 0.0069 | -0.0608 | -0.2524 | noise+action | -0.0608 | -0.1583 | 0.0320 | 0.0417 | -0.0174 | 0.0995 |
| 3 | leave_one_snapshot_out | noise+action | absolute | 208 | 9 | 0.3435 | 1.0108 | 0.6667 | 0.5347 | 0.1319 | -0.0139 | 0.2986 | -0.1025 | -0.4253 | noise+action | -0.1025 | -0.1918 | -0.0153 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action | absolute | 208 | 9 | 0.1935 | 0.5952 | 0.5556 | 0.5347 | 0.0208 | -0.1844 | 0.3526 | 0.0475 | 0.1971 | noise+action | 0.0475 | -0.0254 | 0.1281 | 0.1500 | 0.0590 | 0.2488 |
| 3 | leave_one_snapshot_out | moe_state | absolute | 208 | 9 | 0.1642 | 0.4944 | 0.6667 | 0.5347 | 0.1319 | 0.0170 | 0.4019 | 0.0768 | 0.3185 | noise+action | 0.0768 | 0.0145 | 0.1367 | 0.1792 | 0.1018 | 0.2813 |
| 3 | leave_one_snapshot_out | moe_action+state | absolute | 208 | 9 | 0.2283 | 0.7243 | 0.6667 | 0.5347 | 0.1319 | -0.1304 | 0.3925 | 0.0127 | 0.0525 | noise+action | 0.0127 | -0.1106 | 0.1147 | 0.1152 | -0.0015 | 0.2636 |
| 3 | leave_one_snapshot_out | noise+action+moe_action | absolute | 208 | 9 | 0.2009 | 0.6172 | 0.5556 | 0.5347 | 0.0208 | -0.2314 | 0.3363 | 0.0400 | 0.1662 | noise+action | 0.0400 | -0.0378 | 0.1365 | 0.1425 | 0.0710 | 0.2500 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 208 | 9 | 0.2385 | 0.7618 | 0.6667 | 0.5347 | 0.1319 | -0.1203 | 0.3387 | 0.0025 | 0.0104 | noise+action | 0.0025 | -0.1168 | 0.1270 | 0.1050 | -0.0013 | 0.2339 |
| 3 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2417 | 0.6773 | 0.6667 | 0.5347 | 0.1319 | -0.1234 | 0.4344 | -0.0007 | -0.0030 | noise+action_within_snapshot | -0.0007 | -0.0036 | 0.0011 | 0.0004 | -0.0024 | 0.0037 |
| 3 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2468 | 0.6885 | 0.6667 | 0.5347 | 0.1319 | -0.0493 | 0.3464 | -0.0059 | -0.0244 | noise+action_within_snapshot | -0.0059 | -0.0098 | -0.0030 | -0.0048 | -0.0098 | -0.0011 |
| 3 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2421 | 0.6782 | 0.6667 | 0.5347 | 0.1319 | -0.0833 | 0.4274 | -0.0011 | -0.0045 | noise+action_within_snapshot | -0.0011 | -0.0026 | 0.0003 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2423 | 0.6785 | 0.6667 | 0.5347 | 0.1319 | -0.0979 | 0.4382 | -0.0013 | -0.0055 | noise+action_within_snapshot | -0.0013 | -0.0046 | 0.0024 | -0.0002 | -0.0046 | 0.0028 |
| 3 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2454 | 0.6854 | 0.4444 | 0.5347 | -0.0903 | -0.1951 | 0.0354 | -0.0045 | -0.0185 | noise+action_within_snapshot | -0.0045 | -0.0081 | -0.0016 | -0.0034 | -0.0061 | -0.0006 |
| 3 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2424 | 0.6787 | 0.6667 | 0.5347 | 0.1319 | -0.1241 | 0.4189 | -0.0015 | -0.0061 | noise+action_within_snapshot | -0.0015 | -0.0064 | 0.0011 | -0.0004 | -0.0046 | 0.0034 |
| 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2420 | 0.6778 | 0.6667 | 0.5347 | 0.1319 | -0.0833 | 0.4250 | -0.0010 | -0.0043 | noise+action_within_snapshot | -0.0010 | -0.0047 | 0.0022 | 0.0001 | -0.0033 | 0.0036 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.2423 | 0.6784 | 0.6667 | 0.5347 | 0.1319 | -0.1304 | 0.4566 | -0.0014 | -0.0056 | noise+action_within_snapshot | -0.0014 | -0.0041 | 0.0009 | -0.0003 | -0.0042 | 0.0027 |
| 1 | leave_one_worker_out | snapshot_rate_only | absolute | 208 | 9 | 0.3786 | 1.0602 | 0.5347 | 0.5347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2188 | 0.0484 | 0.5095 |
| 1 | leave_one_worker_out | noise+action | absolute | 208 | 9 | 0.5974 | 2.1587 | 0.6667 | 0.5347 | 0.1319 | 0.0417 | 0.2465 | -0.2188 | -0.5779 | noise+action | -0.2188 | -0.4588 | -0.0499 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action | absolute | 208 | 9 | 0.5613 | 2.2768 | 0.6667 | 0.5347 | 0.1319 | -0.0938 | 0.2578 | -0.1827 | -0.4824 | noise+action | -0.1827 | -0.3345 | -0.0590 | 0.0362 | -0.0730 | 0.1531 |
| 1 | leave_one_worker_out | moe_state | absolute | 208 | 9 | 0.4626 | 1.8411 | 0.6667 | 0.5347 | 0.1319 | 0.0500 | 0.4062 | -0.0840 | -0.2219 | noise+action | -0.0840 | -0.1572 | -0.0206 | 0.1348 | -0.1088 | 0.3700 |
| 1 | leave_one_worker_out | noise+action+moe_action+state | absolute | 208 | 9 | 0.5810 | 2.4436 | 0.5556 | 0.5347 | 0.0208 | -0.0807 | 0.0612 | -0.2024 | -0.5346 | noise+action | -0.2024 | -0.3924 | -0.0502 | 0.0164 | -0.0773 | 0.1250 |
| 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3831 | 1.0927 | 0.5556 | 0.5347 | 0.0208 | -0.0357 | 0.0568 | -0.0045 | -0.0118 | noise+action_within_snapshot | -0.0045 | -0.0097 | -0.0001 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3852 | 1.1140 | 0.4444 | 0.5347 | -0.0903 | -0.2422 | 0.0312 | -0.0065 | -0.0173 | noise+action_within_snapshot | -0.0065 | -0.0080 | -0.0049 | -0.0021 | -0.0059 | 0.0028 |
| 1 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.4492 | 2.4526 | 0.6667 | 0.5347 | 0.1319 | 0.0417 | 0.2500 | -0.0705 | -0.1863 | noise+action_within_snapshot | -0.0705 | -0.1415 | 0.0059 | -0.0661 | -0.1533 | 0.0087 |
| 1 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3880 | 1.1156 | 0.4444 | 0.5347 | -0.0903 | -0.2917 | 0.0312 | -0.0094 | -0.0248 | noise+action_within_snapshot | -0.0094 | -0.0152 | -0.0048 | -0.0049 | -0.0067 | -0.0033 |
| 3 | leave_one_worker_out | snapshot_rate_only | absolute | 208 | 9 | 0.3786 | 1.0602 | 0.5347 | 0.5347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2247 | 0.0854 | 0.4188 |
| 3 | leave_one_worker_out | noise+action | absolute | 208 | 9 | 0.6033 | 2.1625 | 0.6667 | 0.5347 | 0.1319 | -0.0762 | 0.3486 | -0.2247 | -0.5934 | noise+action | -0.2247 | -0.3496 | -0.0696 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action | absolute | 208 | 9 | 0.5451 | 4.1672 | 0.6667 | 0.5347 | 0.1319 | -0.1875 | 0.4062 | -0.1665 | -0.4397 | noise+action | -0.1665 | -0.2653 | -0.0748 | 0.0582 | -0.0591 | 0.1961 |
| 3 | leave_one_worker_out | moe_state | absolute | 208 | 9 | 0.6617 | 5.0177 | 0.5556 | 0.5347 | 0.0208 | -0.0807 | 0.0568 | -0.2830 | -0.7476 | noise+action | -0.2830 | -0.5617 | -0.1373 | -0.0584 | -0.1315 | 0.0637 |
| 3 | leave_one_worker_out | noise+action+moe_action+state | absolute | 208 | 9 | 0.6590 | 5.3449 | 0.6667 | 0.5347 | 0.1319 | -0.0938 | 0.2561 | -0.2804 | -0.7405 | noise+action | -0.2804 | -0.5183 | -0.1075 | -0.0557 | -0.1278 | 0.0930 |
| 3 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3834 | 1.0870 | 0.4444 | 0.5347 | -0.0903 | -0.2917 | 0.0312 | -0.0048 | -0.0127 | noise+action_within_snapshot | -0.0048 | -0.0083 | -0.0007 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3833 | 1.1078 | 0.6667 | 0.5347 | 0.1319 | 0.0417 | 0.2500 | -0.0047 | -0.0125 | noise+action_within_snapshot | -0.0047 | -0.0099 | -0.0001 | 0.0001 | -0.0032 | 0.0033 |
| 3 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3869 | 1.1196 | 0.4444 | 0.5347 | -0.0903 | -0.2805 | 0.0555 | -0.0083 | -0.0219 | noise+action_within_snapshot | -0.0083 | -0.0139 | -0.0038 | -0.0035 | -0.0047 | -0.0027 |
| 3 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 208 | 9 | 0.3848 | 1.1092 | 0.4444 | 0.5347 | -0.0903 | -0.2422 | 0.0312 | -0.0062 | -0.0163 | noise+action_within_snapshot | -0.0062 | -0.0142 | -0.0001 | -0.0014 | -0.0034 | 0.0028 |

Snapshot difficulty (one row per rolling state; snapshot and worker holdouts):

| cv_scheme | model | snapshots | heldout_units | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high | mse_gain_vs_policy_state | mse_gain_vs_policy_state_ci95_low | mse_gain_vs_policy_state_ci95_high | mse_gain_vs_geometry | mse_gain_vs_geometry_ci95_low | mse_gain_vs_geometry_ci95_high | mse_gain_vs_sim_state | mse_gain_vs_sim_state_ci95_low | mse_gain_vs_sim_state_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_snapshot_out | initial_geometry | 13 | 13 | 10.0000 | 0.0407 | 0.1827 | 0.1437 | 0.1040 | 0.2148 | 0.0704 | -0.0041 | 0.1457 | 0.0000 | 0.0000 | 0.0000 | 0.0302 | -0.0006 | 0.0567 |
| leave_one_snapshot_out | initial_sim_state | 13 | 13 | 10.0000 | 0.0709 | 0.2176 | 0.1135 | 0.0478 | 0.1613 | 0.0402 | -0.0040 | 0.0925 | -0.0302 | -0.0702 | -0.0006 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_snapshot_out | mean_moe_action | 13 | 13 | 10.0000 | 0.0839 | 0.2484 | 0.1005 | 0.0351 | 0.1788 | 0.0272 | -0.0377 | 0.0847 | -0.0432 | -0.0842 | -0.0031 | -0.0130 | -0.0552 | 0.0214 |
| leave_one_snapshot_out | sim_state+mean_moe_action+state | 13 | 13 | 10.0000 | 0.0910 | 0.2580 | 0.0934 | 0.0351 | 0.1884 | 0.0201 | -0.0391 | 0.0793 | -0.0503 | -0.0950 | -0.0078 | -0.0201 | -0.0747 | 0.0179 |
| leave_one_snapshot_out | geometry+mean_moe_action+state | 13 | 13 | 10.0000 | 0.0916 | 0.2587 | 0.0928 | 0.0248 | 0.1710 | 0.0195 | -0.0482 | 0.1070 | -0.0508 | -0.0919 | -0.0092 | -0.0206 | -0.0768 | 0.0192 |
| leave_one_snapshot_out | policy_state+mean_moe_action+state | 13 | 13 | 10.0000 | 0.0917 | 0.2590 | 0.0927 | 0.0092 | 0.1694 | 0.0194 | -0.0667 | 0.0903 | -0.0509 | -0.0819 | -0.0132 | -0.0207 | -0.0767 | 0.0455 |
| leave_one_snapshot_out | mean_moe_action+state | 13 | 13 | 10.0000 | 0.0923 | 0.2598 | 0.0921 | 0.0196 | 0.1502 | 0.0188 | -0.0691 | 0.0945 | -0.0516 | -0.0981 | -0.0047 | -0.0214 | -0.0780 | 0.0269 |
| leave_one_snapshot_out | initial_policy_state | 13 | 13 | 10.0000 | 0.1111 | 0.2634 | 0.0733 | -0.0221 | 0.1311 | 0.0000 | 0.0000 | 0.0000 | -0.0704 | -0.1442 | -0.0138 | -0.0402 | -0.1019 | 0.0119 |
| leave_one_snapshot_out | mean_moe_state | 13 | 13 | 10.0000 | 0.1159 | 0.2764 | 0.0685 | 0.0026 | 0.1359 | -0.0048 | -0.0906 | 0.0822 | -0.0751 | -0.1326 | -0.0232 | -0.0449 | -0.1031 | 0.0033 |
| leave_one_snapshot_out | as_moe | 13 | 13 | 10.0000 | 0.1844 | 0.3894 | 0.0000 | -0.0000 | 0.0000 | -0.0733 | -0.1427 | 0.0104 | -0.1437 | -0.2025 | -0.0928 | -0.1135 | -0.1894 | -0.0645 |
| leave_one_snapshot_out | mean_rate | 13 | 13 | NA | 0.1844 | 0.3894 | 0.0000 | 0.0000 | 0.0000 | -0.0733 | -0.1551 | 0.0062 | -0.1437 | -0.2354 | -0.0799 | -0.1135 | -0.1792 | -0.0577 |
| leave_one_snapshot_out | mean_output_action | 13 | 13 | 10.0000 | 0.4717 | 0.6330 | -0.2872 | -0.4385 | -0.1306 | -0.3605 | -0.4946 | -0.2293 | -0.4309 | -0.5656 | -0.2362 | -0.4007 | -0.5441 | -0.2716 |
| leave_one_worker_out | initial_policy_state | 13 | 4 | 10.0000 | 0.2390 | 0.4285 | 0.0636 | -0.0844 | 0.2067 | 0.0000 | 0.0000 | 0.0000 | 0.0305 | -0.0602 | 0.1644 | 0.0463 | -0.0757 | 0.1035 |
| leave_one_worker_out | initial_geometry | 13 | 4 | 10.0000 | 0.2695 | 0.4091 | 0.0331 | -0.0353 | 0.1469 | -0.0305 | -0.1571 | 0.1187 | 0.0000 | 0.0000 | 0.0000 | 0.0158 | -0.1567 | 0.1760 |
| leave_one_worker_out | initial_sim_state | 13 | 4 | 10.0000 | 0.2853 | 0.4281 | 0.0173 | -0.2776 | 0.2088 | -0.0463 | -0.1096 | 0.0525 | -0.0158 | -0.1839 | 0.1283 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_worker_out | as_moe | 13 | 4 | 10.0000 | 0.3026 | 0.4858 | 0.0000 | 0.0000 | 0.0000 | -0.0636 | -0.2031 | 0.1168 | -0.0331 | -0.1102 | 0.0506 | -0.0173 | -0.2149 | 0.2263 |
| leave_one_worker_out | mean_rate | 13 | 4 | NA | 0.3026 | 0.4858 | 0.0000 | 0.0000 | 0.0000 | -0.0636 | -0.1763 | 0.1168 | -0.0331 | -0.1575 | 0.0477 | -0.0173 | -0.1990 | 0.2776 |
| leave_one_worker_out | mean_moe_action | 13 | 4 | 10.0000 | 0.3698 | 0.5271 | -0.0672 | -0.1743 | -0.0112 | -0.1308 | -0.2600 | -0.0208 | -0.1003 | -0.2117 | 0.0610 | -0.0845 | -0.2348 | 0.1010 |
| leave_one_worker_out | mean_moe_state | 13 | 4 | 10.0000 | 0.3748 | 0.5197 | -0.0722 | -0.2080 | 0.0590 | -0.1358 | -0.2656 | -0.0188 | -0.1053 | -0.2122 | 0.0120 | -0.0896 | -0.2190 | 0.0629 |
| leave_one_worker_out | geometry+mean_moe_action+state | 13 | 4 | 10.0000 | 0.3813 | 0.5311 | -0.0787 | -0.2287 | 0.0210 | -0.1422 | -0.2551 | -0.0157 | -0.1118 | -0.2345 | 0.0285 | -0.0960 | -0.2222 | 0.0086 |
| leave_one_worker_out | sim_state+mean_moe_action+state | 13 | 4 | 10.0000 | 0.3818 | 0.5316 | -0.0792 | -0.1616 | 0.0211 | -0.1428 | -0.2504 | -0.0190 | -0.1123 | -0.2066 | -0.0076 | -0.0965 | -0.2030 | 0.0491 |
| leave_one_worker_out | policy_state+mean_moe_action+state | 13 | 4 | 10.0000 | 0.3824 | 0.5322 | -0.0798 | -0.2274 | 0.0318 | -0.1434 | -0.2337 | -0.0470 | -0.1129 | -0.2231 | -0.0076 | -0.0972 | -0.2041 | 0.0407 |
| leave_one_worker_out | mean_moe_action+state | 13 | 4 | 10.0000 | 0.3829 | 0.5326 | -0.0803 | -0.2102 | 0.0349 | -0.1439 | -0.2564 | -0.0653 | -0.1134 | -0.2371 | -0.0098 | -0.0976 | -0.2068 | 0.0610 |
| leave_one_worker_out | mean_output_action | 13 | 4 | 10.0000 | 0.5513 | 0.7023 | -0.2487 | -0.5925 | -0.0219 | -0.3122 | -0.4413 | -0.2219 | -0.2818 | -0.4864 | -0.0711 | -0.2660 | -0.3484 | -0.2195 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | front_2_5 | 1 | top1_mass | 0.5833 | 0.2778 | 0.3056 | 0.1944 | 0.4382 | 0.0196 | 0.0196 | 36 | 36 |
| 1 | state | front_2_5 | 0 | consensus_distance | 0.5278 | 0.2500 | 0.2778 | 0.1451 | 0.4167 | 0.0196 | 0.0588 | 36 | 36 |
| 1 | state | front_2_5 | 0 | top1_mass | 0.6111 | 0.3333 | 0.2778 | 0.1174 | 0.4104 | 0.0196 | 0.0588 | 36 | 36 |
| 1 | state | front_2_5 | 2 | top1_mass | 0.5833 | 0.3611 | 0.2222 | 0.0278 | 0.3333 | 0.0196 | 0.5294 | 36 | 36 |
| 1 | state | front_2_5 | 7 | consensus_distance | 0.5556 | 0.3333 | 0.2222 | 0.0833 | 0.3549 | 0.0196 | 0.5294 | 36 | 36 |
| 1 | state | front_2_5 | 3 | top1_mass | 0.5556 | 0.3333 | 0.2222 | 0.0063 | 0.3611 | 0.0392 | 0.5294 | 36 | 36 |
| 1 | state | front_2_5 | 1 | consensus_distance | 0.5278 | 0.3056 | 0.2222 | 0.0681 | 0.3611 | 0.0588 | 0.5294 | 36 | 36 |
| 1 | state | front_2_5 | 8 | entropy | 0.5556 | 0.3333 | 0.2222 | 0.0278 | 0.3889 | 0.0588 | 0.5294 | 36 | 36 |
| 1 | state | front_2_5 | 5 | top1_mass | 0.5278 | 0.3333 | 0.1944 | 0.0618 | 0.2993 | 0.0196 | 0.8235 | 36 | 36 |
| 1 | state | front_2_5 | 7 | entropy | 0.5556 | 0.3611 | 0.1944 | 0.0278 | 0.3333 | 0.0196 | 0.8235 | 36 | 36 |
| 1 | state | front_2_5 | 4 | entropy | 0.5278 | 0.3333 | 0.1944 | 0.0618 | 0.3271 | 0.0392 | 0.8235 | 36 | 36 |
| 1 | state | front_2_5 | 6 | top1_mass | 0.5556 | 0.3611 | 0.1944 | 0.0833 | 0.2500 | 0.0392 | 0.8235 | 36 | 36 |

Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):

| prefix_queries | token_family | layer_group | denoise_step | expert | failure_minus_success_probability | ci95_low | ci95_high | ci95_worker_low | ci95_worker_high | snapshot_effect_min | snapshot_effect_max | snapshot_effect_sd | snapshots_effect_positive | snapshots_effect_negative | permutation_p_raw | permutation_p_maxT | n_mixed_snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | action | front_2_5 | 0 | 0 | 0.0003 | 0.0001 | 0.0006 | -0.0000 | 0.0006 | -0.0004 | 0.0013 | 0.0006 | 6 | 3 | 0.0392 | 0.0588 | 9 |
| 1 | action | front_2_5 | 1 | 0 | 0.0003 | 0.0000 | 0.0006 | 0.0000 | 0.0006 | -0.0004 | 0.0013 | 0.0006 | 6 | 3 | 0.0392 | 0.0784 | 9 |
| 1 | action | front_2_5 | 9 | 31 | -0.0002 | -0.0003 | -0.0001 | -0.0003 | -0.0001 | -0.0006 | 0.0000 | 0.0002 | 2 | 7 | 0.0196 | 0.7647 | 9 |
| 1 | action | front_2_5 | 2 | 0 | 0.0002 | -0.0000 | 0.0004 | 0.0000 | 0.0005 | -0.0006 | 0.0010 | 0.0006 | 5 | 4 | 0.0784 | 0.7647 | 9 |
| 1 | action | front_2_5 | 0 | 12 | -0.0002 | -0.0004 | 0.0001 | -0.0003 | -0.0001 | -0.0008 | 0.0003 | 0.0004 | 4 | 5 | 0.2157 | 0.8235 | 9 |
| 1 | action | back_12_15 | 0 | 21 | 0.0002 | 0.0001 | 0.0004 | -0.0001 | 0.0004 | -0.0001 | 0.0006 | 0.0003 | 5 | 4 | 0.0196 | 0.8824 | 9 |
| 1 | action | front_2_5 | 2 | 11 | -0.0002 | -0.0002 | -0.0001 | -0.0002 | -0.0001 | -0.0004 | 0.0000 | 0.0001 | 1 | 8 | 0.0196 | 0.9020 | 9 |
| 1 | action | front_2_5 | 1 | 12 | -0.0002 | -0.0004 | 0.0001 | -0.0003 | -0.0001 | -0.0008 | 0.0004 | 0.0004 | 4 | 5 | 0.2745 | 0.9020 | 9 |
| 1 | action | back_12_15 | 2 | 21 | 0.0002 | 0.0001 | 0.0003 | -0.0000 | 0.0004 | -0.0001 | 0.0005 | 0.0002 | 5 | 4 | 0.0196 | 0.9216 | 9 |
| 1 | action | front_2_5 | 9 | 19 | -0.0002 | -0.0003 | -0.0001 | -0.0003 | -0.0001 | -0.0006 | 0.0001 | 0.0002 | 1 | 8 | 0.0392 | 0.9216 | 9 |
| 1 | action | front_2_5 | 1 | 11 | -0.0002 | -0.0002 | -0.0001 | -0.0002 | -0.0001 | -0.0005 | 0.0000 | 0.0002 | 1 | 8 | 0.0196 | 0.9412 | 9 |
| 1 | action | front_2_5 | 8 | 9 | 0.0002 | 0.0001 | 0.0002 | 0.0000 | 0.0002 | -0.0000 | 0.0004 | 0.0001 | 7 | 2 | 0.0196 | 0.9412 | 9 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | cv_scheme | model | feature_scope | n_failure_branches | n_type_positive | heldout_units | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_improvement_vs_rate_ci95_low | brier_improvement_vs_rate_ci95_high | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action | brier_improvement_vs_noise_action_ci95_low | brier_improvement_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_or_regrasp | 1 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 17 | 13 | 0.1144 | 0.3925 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0074 | -0.0013 | 0.0163 |
| drop_or_regrasp | 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 17 | 13 | 0.1147 | 0.4002 | noise+action_within_snapshot | -0.0003 | -0.0077 | 0.0105 | -0.0023 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 1 | leave_one_snapshot_out | noise+action | absolute | 131 | 17 | 13 | 0.1218 | 0.4697 | noise+action | -0.0074 | -0.0202 | 0.0017 | -0.0646 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 3 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 17 | 13 | 0.1144 | 0.3925 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0067 | -0.0007 | 0.0162 |
| drop_or_regrasp | 3 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 131 | 17 | 13 | 0.1199 | 0.4512 | noise+action_within_snapshot | -0.0055 | -0.0195 | 0.0024 | -0.0476 | 0.0035 | -0.0015 | 0.0091 |
| drop_or_regrasp | 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 17 | 13 | 0.1200 | 0.4529 | noise+action_within_snapshot | -0.0056 | -0.0195 | 0.0017 | -0.0487 | 0.0033 | -0.0023 | 0.0087 |
| goal_regression | 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 6 | 13 | 0.0442 | 0.1955 | noise+action_within_snapshot | 0.0004 | -0.0014 | 0.0013 | 0.0094 | 0.0000 | 0.0000 | 0.0000 |
| goal_regression | 1 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 6 | 13 | 0.0446 | 0.1975 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0020 | -0.0004 | 0.0046 |
| goal_regression | 1 | leave_one_snapshot_out | noise+action | absolute | 131 | 6 | 13 | 0.0467 | 0.3095 | noise+action | -0.0020 | -0.0046 | 0.0005 | -0.0455 | 0.0000 | 0.0000 | 0.0000 |
| goal_regression | 3 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 131 | 6 | 13 | 0.0428 | 0.3223 | noise+action_within_snapshot | 0.0018 | 0.0005 | 0.0043 | 0.0400 | 0.0030 | 0.0003 | 0.0064 |
| goal_regression | 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 6 | 13 | 0.0431 | 0.3249 | noise+action_within_snapshot | 0.0016 | 0.0001 | 0.0029 | 0.0353 | 0.0028 | 0.0009 | 0.0055 |
| goal_regression | 3 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 6 | 13 | 0.0446 | 0.1975 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | -0.0001 | 0.0039 |
| loop_or_cycling | 1 | leave_one_snapshot_out | noise+action | absolute | 131 | 92 | 13 | 0.2154 | 0.6557 | noise+action | 0.0170 | -0.0276 | 0.0470 | 0.0732 | 0.0000 | 0.0000 | 0.0000 |
| loop_or_cycling | 1 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 92 | 13 | 0.2324 | 0.6652 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0170 | -0.0494 | 0.0299 |
| loop_or_cycling | 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 92 | 13 | 0.2411 | 0.6910 | noise+action_within_snapshot | -0.0088 | -0.0221 | -0.0012 | -0.0377 | 0.0000 | 0.0000 | 0.0000 |
| loop_or_cycling | 3 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 131 | 92 | 13 | 0.2172 | 1.1292 | noise+action | 0.0152 | -0.0512 | 0.0479 | 0.0652 | 0.0213 | -0.0299 | 0.0561 |
| loop_or_cycling | 3 | leave_one_snapshot_out | moe_action+state | absolute | 131 | 92 | 13 | 0.2173 | 1.2386 | noise+action | 0.0150 | -0.0422 | 0.0637 | 0.0648 | 0.0212 | -0.0225 | 0.0599 |
| loop_or_cycling | 3 | leave_one_snapshot_out | noise+action+moe_action | absolute | 131 | 92 | 13 | 0.2175 | 0.7527 | noise+action | 0.0149 | -0.0225 | 0.0621 | 0.0639 | 0.0210 | -0.0081 | 0.0537 |
| non_stagnation_non_loop | 1 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 18 | 13 | 0.1211 | 0.4108 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0107 | 0.0038 | 0.0248 |
| non_stagnation_non_loop | 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 18 | 13 | 0.1246 | 0.4348 | noise+action_within_snapshot | -0.0036 | -0.0107 | 0.0015 | -0.0294 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 1 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 18 | 13 | 0.1303 | 0.4656 | noise+action_within_snapshot | -0.0092 | -0.0206 | -0.0023 | -0.0762 | -0.0057 | -0.0110 | 0.0003 |
| non_stagnation_non_loop | 3 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 18 | 13 | 0.1211 | 0.4108 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0139 | 0.0027 | 0.0279 |
| non_stagnation_non_loop | 3 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 18 | 13 | 0.1223 | 0.4310 | noise+action_within_snapshot | -0.0012 | -0.0083 | 0.0037 | -0.0102 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 18 | 13 | 0.1323 | 0.4991 | noise+action_within_snapshot | -0.0113 | -0.0250 | 0.0004 | -0.0932 | -0.0100 | -0.0210 | -0.0039 |
| single_subtask_omission | 1 | leave_one_snapshot_out | noise+action+moe_action | absolute | 131 | 46 | 13 | 0.2471 | 1.1041 | noise+action | 0.0379 | -0.1333 | 0.1537 | 0.1330 | 0.0574 | -0.0854 | 0.2047 |
| single_subtask_omission | 1 | leave_one_snapshot_out | moe_action | absolute | 131 | 46 | 13 | 0.2502 | 1.0946 | noise+action | 0.0348 | -0.0864 | 0.1457 | 0.1221 | 0.0543 | -0.0722 | 0.1990 |
| single_subtask_omission | 1 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 131 | 46 | 13 | 0.2717 | 0.8417 | noise+action_within_snapshot | 0.0133 | -0.0301 | 0.0454 | 0.0465 | 0.0208 | -0.0232 | 0.0976 |
| single_subtask_omission | 3 | leave_one_snapshot_out | noise+action+moe_action | absolute | 131 | 46 | 13 | 0.2519 | 1.0640 | noise+action | 0.0331 | -0.0745 | 0.1509 | 0.1162 | 0.0817 | -0.0019 | 0.1815 |
| single_subtask_omission | 3 | leave_one_snapshot_out | moe_action | absolute | 131 | 46 | 13 | 0.2552 | 1.0931 | noise+action | 0.0297 | -0.0880 | 0.1253 | 0.1044 | 0.0784 | 0.0029 | 0.2131 |
| single_subtask_omission | 3 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 131 | 46 | 13 | 0.2782 | 2.6402 | noise+action | 0.0068 | -0.0884 | 0.1334 | 0.0237 | 0.0554 | -0.0145 | 0.2287 |
| stagnation | 1 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 131 | 27 | 13 | 0.1698 | 0.6285 | noise+action_within_snapshot | 0.0054 | -0.0068 | 0.0252 | 0.0311 | 0.0152 | 0.0012 | 0.0281 |
| stagnation | 1 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 131 | 27 | 13 | 0.1699 | 0.6286 | noise+action_within_snapshot | 0.0054 | -0.0069 | 0.0161 | 0.0308 | 0.0152 | 0.0061 | 0.0264 |
| stagnation | 1 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 27 | 13 | 0.1753 | 0.5459 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0182 | -0.0040 | 0.0354 |
| stagnation | 3 | leave_one_snapshot_out | failure_rate_only | absolute | 131 | 27 | 13 | 0.1753 | 0.5459 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0324 | 0.0045 | 0.0623 |
| stagnation | 3 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 131 | 27 | 13 | 0.1792 | 0.6580 | noise+action_within_snapshot | -0.0039 | -0.0106 | 0.0036 | -0.0221 | 0.0096 | -0.0025 | 0.0164 |
| stagnation | 3 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 131 | 27 | 13 | 0.1793 | 0.6577 | noise+action_within_snapshot | -0.0040 | -0.0136 | 0.0034 | -0.0230 | 0.0094 | 0.0006 | 0.0190 |
| drop_or_regrasp | 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 17 | 4 | 0.1243 | 0.4655 | noise+action_within_snapshot | 0.0005 | -0.0105 | 0.0133 | 0.0042 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 1 | leave_one_worker_out | failure_rate_only | absolute | 131 | 17 | 4 | 0.1248 | 0.4376 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0103 | 0.0044 | 0.0138 |
| drop_or_regrasp | 1 | leave_one_worker_out | noise+action | absolute | 131 | 17 | 4 | 0.1351 | 0.5289 | noise+action | -0.0103 | -0.0141 | -0.0050 | -0.0826 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 3 | leave_one_worker_out | noise+action | absolute | 131 | 17 | 4 | 0.1190 | 0.4874 | noise+action | 0.0058 | -0.0031 | 0.0168 | 0.0465 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 3 | leave_one_worker_out | failure_rate_only | absolute | 131 | 17 | 4 | 0.1248 | 0.4376 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0058 | -0.0156 | 0.0026 |
| drop_or_regrasp | 3 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 17 | 4 | 0.1290 | 0.4853 | noise+action_within_snapshot | -0.0042 | -0.0118 | 0.0025 | -0.0339 | 0.0000 | 0.0000 | 0.0000 |
| goal_regression | 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 6 | 4 | 0.0448 | 0.2151 | noise+action_within_snapshot | 0.0006 | -0.0008 | 0.0017 | 0.0125 | 0.0000 | 0.0000 | 0.0000 |
| goal_regression | 1 | leave_one_worker_out | failure_rate_only | absolute | 131 | 6 | 4 | 0.0454 | 0.2103 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | -0.0006 | 0.0032 |
| goal_regression | 1 | leave_one_worker_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 6 | 4 | 0.0457 | 0.3453 | noise+action_within_snapshot | -0.0003 | -0.0043 | 0.0059 | -0.0063 | -0.0009 | -0.0053 | 0.0037 |
| goal_regression | 3 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 131 | 6 | 4 | 0.0454 | 0.3559 | noise+action_within_snapshot | 0.0000 | -0.0037 | 0.0021 | 0.0006 | 0.0005 | -0.0011 | 0.0019 |
| goal_regression | 3 | leave_one_worker_out | failure_rate_only | absolute | 131 | 6 | 4 | 0.0454 | 0.2103 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | -0.0010 | 0.0027 |
| goal_regression | 3 | leave_one_worker_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 6 | 4 | 0.0455 | 0.3659 | noise+action_within_snapshot | -0.0001 | -0.0018 | 0.0018 | -0.0018 | 0.0004 | -0.0009 | 0.0016 |
| loop_or_cycling | 1 | leave_one_worker_out | noise+action | absolute | 131 | 92 | 4 | 0.2231 | 0.6612 | noise+action | 0.0415 | -0.0122 | 0.0648 | 0.1569 | 0.0000 | 0.0000 | 0.0000 |
| loop_or_cycling | 1 | leave_one_worker_out | noise+action+moe_action | absolute | 131 | 92 | 4 | 0.2476 | 0.7285 | noise+action | 0.0170 | -0.0193 | 0.0469 | 0.0642 | -0.0245 | -0.0509 | 0.0048 |
| loop_or_cycling | 1 | leave_one_worker_out | moe_state | absolute | 131 | 92 | 4 | 0.2518 | 1.0069 | noise+action | 0.0128 | -0.0620 | 0.0990 | 0.0483 | -0.0287 | -0.0928 | 0.0452 |
| loop_or_cycling | 3 | leave_one_worker_out | noise+action | absolute | 131 | 92 | 4 | 0.2532 | 0.7516 | noise+action | 0.0113 | -0.0325 | 0.0319 | 0.0428 | 0.0000 | 0.0000 | 0.0000 |
| loop_or_cycling | 3 | leave_one_worker_out | noise+action+moe_action | absolute | 131 | 92 | 4 | 0.2601 | 0.7748 | noise+action | 0.0045 | -0.0274 | 0.0120 | 0.0169 | -0.0069 | -0.0211 | 0.0155 |
| loop_or_cycling | 3 | leave_one_worker_out | failure_rate_only | absolute | 131 | 92 | 4 | 0.2646 | 0.7465 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0113 | -0.0312 | 0.0356 |
| non_stagnation_non_loop | 1 | leave_one_worker_out | failure_rate_only | absolute | 131 | 18 | 4 | 0.1276 | 0.4381 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0380 | 0.0067 | 0.0686 |
| non_stagnation_non_loop | 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 18 | 4 | 0.1332 | 0.4792 | noise+action_within_snapshot | -0.0056 | -0.0143 | 0.0025 | -0.0439 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 1 | leave_one_worker_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 18 | 4 | 0.1349 | 0.5169 | noise+action_within_snapshot | -0.0073 | -0.0240 | 0.0012 | -0.0576 | -0.0017 | -0.0127 | 0.0126 |
| non_stagnation_non_loop | 3 | leave_one_worker_out | failure_rate_only | absolute | 131 | 18 | 4 | 0.1276 | 0.4381 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0158 | 0.0000 | 0.0430 |
| non_stagnation_non_loop | 3 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 131 | 18 | 4 | 0.1286 | 0.4740 | noise+action_within_snapshot | -0.0010 | -0.0084 | 0.0049 | -0.0081 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 3 | leave_one_worker_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 18 | 4 | 0.1402 | 0.5670 | noise+action_within_snapshot | -0.0127 | -0.0260 | -0.0021 | -0.0994 | -0.0116 | -0.0212 | 0.0003 |
| single_subtask_omission | 1 | leave_one_worker_out | noise+action+moe_action | absolute | 131 | 46 | 4 | 0.3188 | 1.5852 | noise+action | 0.0619 | -0.0451 | 0.2254 | 0.1626 | 0.0686 | -0.0063 | 0.1082 |
| single_subtask_omission | 1 | leave_one_worker_out | moe_action | absolute | 131 | 46 | 4 | 0.3278 | 1.5418 | noise+action | 0.0529 | -0.0698 | 0.2044 | 0.1389 | 0.0596 | 0.0252 | 0.0899 |
| single_subtask_omission | 1 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 131 | 46 | 4 | 0.3386 | 1.2701 | noise+action_within_snapshot | 0.0420 | -0.0213 | 0.1097 | 0.1104 | 0.0490 | -0.0180 | 0.1495 |
| single_subtask_omission | 3 | leave_one_worker_out | moe_action+state | absolute | 131 | 46 | 4 | 0.3594 | 4.4341 | noise+action | 0.0212 | -0.1492 | 0.2293 | 0.0558 | 0.0403 | -0.0182 | 0.1183 |
| single_subtask_omission | 3 | leave_one_worker_out | noise+action+moe_action+state | absolute | 131 | 46 | 4 | 0.3604 | 4.4384 | noise+action | 0.0203 | -0.1580 | 0.2327 | 0.0532 | 0.0394 | -0.0182 | 0.1269 |
| single_subtask_omission | 3 | leave_one_worker_out | moe_action | absolute | 131 | 46 | 4 | 0.3611 | 2.1890 | noise+action | 0.0195 | -0.1340 | 0.1911 | 0.0513 | 0.0386 | -0.0064 | 0.1071 |
| stagnation | 1 | leave_one_worker_out | failure_rate_only | absolute | 131 | 27 | 4 | 0.2084 | 0.6724 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0159 | -0.0266 | 0.0591 |
| stagnation | 1 | leave_one_worker_out | moe_action | absolute | 131 | 27 | 4 | 0.2131 | 1.6988 | noise+action | -0.0047 | -0.0586 | 0.0582 | -0.0224 | 0.0112 | -0.0022 | 0.0316 |
| stagnation | 1 | leave_one_worker_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 27 | 4 | 0.2137 | 0.6870 | noise+action_within_snapshot | -0.0053 | -0.0117 | 0.0038 | -0.0255 | 0.0070 | -0.0112 | 0.0291 |
| stagnation | 3 | leave_one_worker_out | moe_action+state | absolute | 131 | 27 | 4 | 0.2056 | 2.0397 | noise+action | 0.0028 | -0.0523 | 0.0756 | 0.0137 | 0.0260 | -0.0030 | 0.0652 |
| stagnation | 3 | leave_one_worker_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 131 | 27 | 4 | 0.2063 | 0.6821 | noise+action_within_snapshot | 0.0021 | -0.0039 | 0.0094 | 0.0101 | 0.0086 | -0.0026 | 0.0148 |
| stagnation | 3 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 131 | 27 | 4 | 0.2066 | 0.6845 | noise+action_within_snapshot | 0.0018 | -0.0041 | 0.0087 | 0.0085 | 0.0082 | -0.0040 | 0.0139 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 3 | action | back_12_15 | 2 | consensus_distance | 0.2273 | 0.5455 | -0.3182 | -0.4387 | -0.2196 | 0.0196 | 0.4118 | 22 | 22 |
| stagnation | 3 | action | back_12_15 | 6 | consensus_distance | 0.2273 | 0.5455 | -0.3182 | -0.4070 | -0.1596 | 0.0196 | 0.4118 | 22 | 22 |
| goal_regression | 3 | action | back_12_15 | 9 | entropy | 0.0000 | 0.3077 | -0.3077 | -0.3333 | -0.2857 | 0.0784 | 0.5098 | 13 | 13 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.4783 | 0.8261 | -0.3478 | -0.6025 | -0.1690 | 0.0196 | 0.5882 | 23 | 23 |
| drop_or_regrasp | 1 | state | front_2_5 | 1 | top1_mass | 0.3636 | 0.0909 | 0.2727 | 0.1262 | 0.4737 | 0.0784 | 0.6667 | 22 | 22 |
| non_stagnation_non_loop | 1 | state | back_12_15 | 8 | top1_mass | 0.0435 | 0.3043 | -0.2609 | -0.4635 | -0.1250 | 0.0588 | 0.7647 | 23 | 23 |
| non_stagnation_non_loop | 1 | action | front_2_5 | 8 | token_dispersion | 0.3043 | 0.0435 | 0.2609 | 0.1926 | 0.3182 | 0.0784 | 0.7647 | 23 | 23 |
| stagnation | 3 | action | back_12_15 | 0 | entropy | 0.4091 | 0.1364 | 0.2727 | 0.1068 | 0.4051 | 0.0392 | 0.8039 | 22 | 22 |
| stagnation | 3 | action | back_12_15 | 9 | token_dispersion | 0.1364 | 0.4091 | -0.2727 | -0.3771 | -0.1429 | 0.0392 | 0.8039 | 22 | 22 |
| stagnation | 3 | action | front_2_5 | 9 | entropy | 0.5000 | 0.2273 | 0.2727 | 0.1394 | 0.3606 | 0.0392 | 0.8039 | 22 | 22 |
| stagnation | 3 | state | front_2_5 | 0 | entropy | 0.1818 | 0.4545 | -0.2727 | -0.4501 | -0.1200 | 0.0392 | 0.8039 | 22 | 22 |
| stagnation | 3 | state | front_2_5 | 1 | entropy | 0.1818 | 0.4545 | -0.2727 | -0.4501 | -0.1200 | 0.0392 | 0.8039 | 22 | 22 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
