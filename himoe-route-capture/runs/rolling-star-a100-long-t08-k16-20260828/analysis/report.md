# Rolling-star K=16 experiment

## Dataset

- 352 terminal branches from 22 committed snapshots; 117 success and 235 failure.
- 15 snapshots contain matched success/failure siblings.
- 16158 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.
- Candidate-ID negative control: max failure-rate deviation 0.122, within-snapshot permutation p=0.6893.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 138
- `single_subtask_omission`: 55
- `drop_or_regrasp`: 22
- `subtask_undo`: 8
- `goal_contact_near_miss`: 5
- `stagnation`: 4
- `timeout_other`: 2
- `active_retry`: 1

Stagnation or loop/cycling covers 199/235 failures; 36 failures require other physical mechanisms: {"active_retry": 1, "drop_or_regrasp": 3, "goal_contact_near_miss": 5, "single_subtask_omission": 23, "subtask_undo": 2, "timeout_other": 2}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00596.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Pilot-frozen two-signal family (the transition set is excluded from independent validation):

| signal | split | primary_statistic | primary_p_one_sided | validation_family_size | primary_p_bonferroni | branches | snapshots | mixed_snapshots | successes | failures | failure_minus_success_value | mean_effect_ci95_low | mean_effect_ci95_high | mean_effect_permutation_p_one_sided | top_quartile_failure_rate | bottom_quartile_failure_rate | quartile_failure_rate_difference | quartile_effect_ci95_low | quartile_effect_ci95_high | quartile_effect_permutation_p_one_sided | lowest_value_selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | selection_gain_permutation_p_one_sided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q0_action_front_d0_expert0 | discovery | mean_failure_minus_success | 0.0050 | 2 | 0.0100 | 192 | 12 | 8 | 62 | 130 | 0.0004 | 0.0000 | 0.0008 | 0.0050 | 0.5625 | 0.4688 | 0.0938 | -0.1250 | 0.2812 | 0.1812 | 0.3333 | 0.3229 | 0.0104 | -0.1198 | 0.1771 | 0.6471 |
| q0_action_front_d0_expert0 | transition_excluded | mean_failure_minus_success | 0.9400 | 2 | 1.0000 | 80 | 5 | 4 | 31 | 49 | -0.0004 | -0.0007 | -0.0001 | 0.9400 | 0.4375 | 0.6250 | -0.1875 | -0.3750 | 0.0000 | 0.9724 | 0.2000 | 0.3875 | -0.1875 | -0.5000 | 0.0125 | 0.9890 |
| q0_action_front_d0_expert0 | validation | mean_failure_minus_success | 0.4219 | 2 | 0.8438 | 80 | 5 | 3 | 24 | 56 | 0.0000 | -0.0003 | 0.0002 | 0.4219 | 0.5000 | 0.5000 | 0.0000 | -0.5000 | 0.2500 | 0.5877 | 0.4000 | 0.3000 | 0.1000 | -0.2000 | 0.4500 | 0.4681 |
| q0_state_front_d1_top1_mass | discovery | top_minus_bottom_quartile_failure_rate | 0.0002 | 2 | 0.0004 | 192 | 12 | 8 | 62 | 130 | 0.0000 | -0.0000 | 0.0000 | 0.2863 | 0.6250 | 0.3125 | 0.3125 | 0.1875 | 0.4688 | 0.0002 | 0.5833 | 0.3229 | 0.2604 | 0.0677 | 0.4844 | 0.0046 |
| q0_state_front_d1_top1_mass | transition_excluded | top_minus_bottom_quartile_failure_rate | 0.0714 | 2 | 0.1428 | 80 | 5 | 4 | 31 | 49 | 0.0002 | -0.0000 | 0.0005 | 0.1506 | 0.6250 | 0.4375 | 0.1875 | -0.1250 | 0.4375 | 0.0714 | 0.6000 | 0.3875 | 0.2125 | -0.0125 | 0.5500 | 0.1418 |
| q0_state_front_d1_top1_mass | validation | top_minus_bottom_quartile_failure_rate | 0.5807 | 2 | 1.0000 | 80 | 5 | 3 | 24 | 56 | -0.0000 | -0.0000 | 0.0000 | 0.4255 | 0.4167 | 0.4167 | 0.0000 | -0.2500 | 0.2500 | 0.5807 | 0.2000 | 0.3000 | -0.1000 | -0.2750 | 0.0500 | 0.9402 |

Both coordinates and positive failure directions were frozen before their validation boundaries; only `validation` rows are prospective tests, and `primary_p_bonferroni` corrects the two-signal family.

Cross-validated models (snapshot-grouped 5-fold and core leave-one-worker-out checks):

| prefix_queries | cv_scheme | cv_folds | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | grouped_5fold_snapshot | 5 | snapshot_rate_only | absolute | 352 | 15 | 0.2232 | 0.6389 | 0.4875 | 0.4875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0011 | -0.0228 | 0.0272 |
| 1 | grouped_5fold_snapshot | 5 | noise | absolute | 352 | 15 | 0.2280 | 0.6495 | 0.4667 | 0.4875 | -0.0208 | -0.1333 | 0.1042 | -0.0047 | -0.0212 | noise+action | -0.0047 | -0.0103 | -0.0002 | -0.0037 | -0.0255 | 0.0189 |
| 1 | grouped_5fold_snapshot | 5 | action | absolute | 352 | 15 | 0.2822 | 0.8544 | 0.3333 | 0.4875 | -0.1542 | -0.2875 | -0.0417 | -0.0589 | -0.2639 | noise+action | -0.0589 | -0.1257 | 0.0028 | -0.0579 | -0.1052 | -0.0116 |
| 1 | grouped_5fold_snapshot | 5 | noise+action | absolute | 352 | 15 | 0.2243 | 0.6430 | 0.4667 | 0.4875 | -0.0208 | -0.1917 | 0.1583 | -0.0011 | -0.0048 | noise+action | -0.0011 | -0.0269 | 0.0224 | 0.0000 | 0.0000 | 0.0000 |
| 1 | grouped_5fold_snapshot | 5 | moe_action | absolute | 352 | 15 | 0.1411 | 0.4594 | 0.7333 | 0.4875 | 0.2458 | 0.0500 | 0.4458 | 0.0821 | 0.3679 | noise+action | 0.0821 | 0.0131 | 0.1480 | 0.0832 | 0.0107 | 0.1599 |
| 1 | grouped_5fold_snapshot | 5 | moe_state | absolute | 352 | 15 | 0.1609 | 0.5542 | 0.5333 | 0.4875 | 0.0458 | -0.0708 | 0.1792 | 0.0623 | 0.2792 | noise+action | 0.0623 | -0.0134 | 0.1374 | 0.0634 | -0.0122 | 0.1362 |
| 1 | grouped_5fold_snapshot | 5 | moe_action+state | absolute | 352 | 15 | 0.1656 | 0.5547 | 0.5333 | 0.4875 | 0.0458 | -0.0958 | 0.2125 | 0.0577 | 0.2583 | noise+action | 0.0577 | -0.0258 | 0.1312 | 0.0587 | -0.0260 | 0.1405 |
| 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action | absolute | 352 | 15 | 0.1413 | 0.4592 | 0.7333 | 0.4875 | 0.2458 | 0.0500 | 0.4542 | 0.0820 | 0.3672 | noise+action | 0.0820 | 0.0145 | 0.1484 | 0.0830 | 0.0083 | 0.1568 |
| 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state | absolute | 352 | 15 | 0.1667 | 0.5592 | 0.5333 | 0.4875 | 0.0458 | -0.0917 | 0.2126 | 0.0566 | 0.2534 | noise+action | 0.0566 | -0.0248 | 0.1330 | 0.0576 | -0.0316 | 0.1378 |
| 1 | grouped_5fold_snapshot | 5 | noise_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2243 | 0.6413 | 0.4000 | 0.4875 | -0.0875 | -0.1792 | 0.0000 | -0.0010 | -0.0046 | noise+action_within_snapshot | -0.0010 | -0.0029 | 0.0005 | 0.0009 | -0.0010 | 0.0028 |
| 1 | grouped_5fold_snapshot | 5 | action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2281 | 0.6498 | 0.4000 | 0.4875 | -0.0875 | -0.2792 | 0.0958 | -0.0048 | -0.0217 | noise+action_within_snapshot | -0.0048 | -0.0073 | -0.0025 | -0.0030 | -0.0051 | -0.0009 |
| 1 | grouped_5fold_snapshot | 5 | noise+action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2251 | 0.6430 | 0.4000 | 0.4875 | -0.0875 | -0.1833 | 0.0000 | -0.0019 | -0.0085 | noise+action_within_snapshot | -0.0019 | -0.0038 | -0.0002 | 0.0000 | 0.0000 | 0.0000 |
| 1 | grouped_5fold_snapshot | 5 | moe_action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2289 | 0.6524 | 0.6000 | 0.4875 | 0.1125 | -0.0583 | 0.3083 | -0.0056 | -0.0252 | noise+action_within_snapshot | -0.0056 | -0.0096 | -0.0022 | -0.0037 | -0.0067 | -0.0008 |
| 1 | grouped_5fold_snapshot | 5 | moe_state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2229 | 0.6370 | 0.4000 | 0.4875 | -0.0875 | -0.2583 | 0.1000 | 0.0004 | 0.0016 | noise+action_within_snapshot | 0.0004 | -0.0129 | 0.0137 | 0.0022 | -0.0112 | 0.0156 |
| 1 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2265 | 0.6465 | 0.5333 | 0.4875 | 0.0458 | -0.1583 | 0.2667 | -0.0032 | -0.0143 | noise+action_within_snapshot | -0.0032 | -0.0063 | -0.0005 | -0.0013 | -0.0041 | 0.0012 |
| 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2288 | 0.6523 | 0.6000 | 0.4875 | 0.1125 | -0.0625 | 0.3042 | -0.0056 | -0.0250 | noise+action_within_snapshot | -0.0056 | -0.0095 | -0.0022 | -0.0037 | -0.0066 | -0.0009 |
| 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2265 | 0.6465 | 0.5333 | 0.4875 | 0.0458 | -0.1667 | 0.2708 | -0.0032 | -0.0144 | noise+action_within_snapshot | -0.0032 | -0.0064 | -0.0006 | -0.0013 | -0.0041 | 0.0012 |
| 3 | grouped_5fold_snapshot | 5 | snapshot_rate_only | absolute | 352 | 15 | 0.2232 | 0.6389 | 0.4875 | 0.4875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0227 | -0.0099 | 0.0587 |
| 3 | grouped_5fold_snapshot | 5 | noise | absolute | 352 | 15 | 0.2253 | 0.6436 | 0.4667 | 0.4875 | -0.0208 | -0.2125 | 0.1667 | -0.0021 | -0.0093 | noise+action | -0.0021 | -0.0076 | 0.0028 | 0.0206 | -0.0149 | 0.0562 |
| 3 | grouped_5fold_snapshot | 5 | action | absolute | 352 | 15 | 0.2781 | 0.8197 | 0.5333 | 0.4875 | 0.0458 | -0.0958 | 0.2000 | -0.0549 | -0.2459 | noise+action | -0.0549 | -0.1185 | 0.0092 | -0.0322 | -0.0927 | 0.0257 |
| 3 | grouped_5fold_snapshot | 5 | noise+action | absolute | 352 | 15 | 0.2459 | 0.6974 | 0.6000 | 0.4875 | 0.1125 | -0.0250 | 0.2708 | -0.0227 | -0.1017 | noise+action | -0.0227 | -0.0578 | 0.0103 | 0.0000 | 0.0000 | 0.0000 |
| 3 | grouped_5fold_snapshot | 5 | moe_action | absolute | 352 | 15 | 0.1204 | 0.4144 | 0.4667 | 0.4875 | -0.0208 | -0.2125 | 0.1667 | 0.1028 | 0.4606 | noise+action | 0.1028 | 0.0581 | 0.1493 | 0.1255 | 0.0579 | 0.2001 |
| 3 | grouped_5fold_snapshot | 5 | moe_state | absolute | 352 | 15 | 0.1151 | 0.3934 | 0.4000 | 0.4875 | -0.0875 | -0.2458 | 0.0833 | 0.1082 | 0.4845 | noise+action | 0.1082 | 0.0574 | 0.1612 | 0.1309 | 0.0598 | 0.2099 |
| 3 | grouped_5fold_snapshot | 5 | moe_action+state | absolute | 352 | 15 | 0.1171 | 0.4036 | 0.5333 | 0.4875 | 0.0458 | -0.1667 | 0.2583 | 0.1061 | 0.4753 | noise+action | 0.1061 | 0.0597 | 0.1551 | 0.1288 | 0.0595 | 0.2022 |
| 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action | absolute | 352 | 15 | 0.1207 | 0.4146 | 0.4667 | 0.4875 | -0.0208 | -0.2167 | 0.1626 | 0.1026 | 0.4594 | noise+action | 0.1026 | 0.0576 | 0.1490 | 0.1253 | 0.0596 | 0.1994 |
| 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state | absolute | 352 | 15 | 0.1173 | 0.4040 | 0.5333 | 0.4875 | 0.0458 | -0.1625 | 0.2542 | 0.1059 | 0.4746 | noise+action | 0.1059 | 0.0600 | 0.1520 | 0.1286 | 0.0575 | 0.2043 |
| 3 | grouped_5fold_snapshot | 5 | noise_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2239 | 0.6404 | 0.6000 | 0.4875 | 0.1125 | -0.0542 | 0.2958 | -0.0006 | -0.0029 | noise+action_within_snapshot | -0.0006 | -0.0024 | 0.0010 | 0.0024 | -0.0005 | 0.0054 |
| 3 | grouped_5fold_snapshot | 5 | action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2281 | 0.6498 | 0.5333 | 0.4875 | 0.0458 | -0.1501 | 0.2458 | -0.0049 | -0.0218 | noise+action_within_snapshot | -0.0049 | -0.0083 | -0.0008 | -0.0018 | -0.0039 | 0.0004 |
| 3 | grouped_5fold_snapshot | 5 | noise+action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2263 | 0.6457 | 0.4000 | 0.4875 | -0.0875 | -0.2542 | 0.0708 | -0.0031 | -0.0137 | noise+action_within_snapshot | -0.0031 | -0.0057 | -0.0004 | 0.0000 | 0.0000 | 0.0000 |
| 3 | grouped_5fold_snapshot | 5 | moe_action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2273 | 0.6483 | 0.4000 | 0.4875 | -0.0875 | -0.2500 | 0.0875 | -0.0041 | -0.0184 | noise+action_within_snapshot | -0.0041 | -0.0069 | -0.0017 | -0.0010 | -0.0044 | 0.0022 |
| 3 | grouped_5fold_snapshot | 5 | moe_state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2226 | 0.6381 | 0.4667 | 0.4875 | -0.0208 | -0.1458 | 0.1292 | 0.0006 | 0.0028 | noise+action_within_snapshot | 0.0006 | -0.0029 | 0.0045 | 0.0037 | 0.0002 | 0.0075 |
| 3 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2249 | 0.6431 | 0.4667 | 0.4875 | -0.0208 | -0.1458 | 0.1292 | -0.0016 | -0.0072 | noise+action_within_snapshot | -0.0016 | -0.0049 | 0.0015 | 0.0015 | -0.0020 | 0.0052 |
| 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2275 | 0.6485 | 0.4000 | 0.4875 | -0.0875 | -0.2500 | 0.0833 | -0.0042 | -0.0189 | noise+action_within_snapshot | -0.0042 | -0.0070 | -0.0017 | -0.0012 | -0.0044 | 0.0019 |
| 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.2249 | 0.6431 | 0.4667 | 0.4875 | -0.0208 | -0.1500 | 0.1333 | -0.0016 | -0.0072 | noise+action_within_snapshot | -0.0016 | -0.0048 | 0.0015 | 0.0015 | -0.0018 | 0.0052 |
| 1 | leave_one_worker_out | 4 | snapshot_rate_only | absolute | 352 | 15 | 0.3313 | 0.9491 | 0.4875 | 0.4875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1024 | 0.0321 | 0.2112 |
| 1 | leave_one_worker_out | 4 | noise+action | absolute | 352 | 15 | 0.4336 | 1.2637 | 0.4667 | 0.4875 | -0.0208 | -0.0938 | 0.2292 | -0.1024 | -0.3091 | noise+action | -0.1024 | -0.2112 | -0.0321 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | 4 | moe_action | absolute | 352 | 15 | 0.3961 | 1.4319 | 0.4000 | 0.4875 | -0.0875 | -0.1042 | -0.0729 | -0.0648 | -0.1956 | noise+action | -0.0648 | -0.1345 | 0.0049 | 0.0376 | -0.0510 | 0.0854 |
| 1 | leave_one_worker_out | 4 | moe_state | absolute | 352 | 15 | 0.5044 | 2.0644 | 0.6000 | 0.4875 | 0.1125 | -0.0938 | 0.5625 | -0.1731 | -0.5226 | noise+action | -0.1731 | -0.3491 | -0.0552 | -0.0707 | -0.1665 | 0.0250 |
| 1 | leave_one_worker_out | 4 | noise+action+moe_action+state | absolute | 352 | 15 | 0.5083 | 2.0414 | 0.4667 | 0.4875 | -0.0208 | -0.1042 | 0.0729 | -0.1770 | -0.5343 | noise+action | -0.1770 | -0.3803 | -0.0337 | -0.0746 | -0.1957 | 0.0465 |
| 1 | leave_one_worker_out | 4 | noise+action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3332 | 0.9613 | 0.5333 | 0.4875 | 0.0458 | -0.1042 | 0.0938 | -0.0019 | -0.0059 | noise+action_within_snapshot | -0.0019 | -0.0031 | -0.0008 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | 4 | moe_action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3366 | 0.9886 | 0.4000 | 0.4875 | -0.0875 | -0.1042 | -0.0729 | -0.0054 | -0.0162 | noise+action_within_snapshot | -0.0054 | -0.0082 | -0.0025 | -0.0034 | -0.0051 | -0.0017 |
| 1 | leave_one_worker_out | 4 | moe_state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3398 | 0.9826 | 0.3333 | 0.4875 | -0.1542 | -0.2604 | -0.0729 | -0.0085 | -0.0257 | noise+action_within_snapshot | -0.0085 | -0.0168 | 0.0015 | -0.0066 | -0.0150 | 0.0036 |
| 1 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3381 | 0.9860 | 0.4667 | 0.4875 | -0.0208 | -0.1042 | 0.0729 | -0.0069 | -0.0208 | noise+action_within_snapshot | -0.0069 | -0.0119 | -0.0023 | -0.0049 | -0.0093 | -0.0014 |
| 3 | leave_one_worker_out | 4 | snapshot_rate_only | absolute | 352 | 15 | 0.3313 | 0.9491 | 0.4875 | 0.4875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2004 | 0.0378 | 0.4085 |
| 3 | leave_one_worker_out | 4 | noise+action | absolute | 352 | 15 | 0.5316 | 1.7350 | 0.6000 | 0.4875 | 0.1125 | -0.1042 | 0.2396 | -0.2004 | -0.6048 | noise+action | -0.2004 | -0.4085 | -0.0378 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | 4 | moe_action | absolute | 352 | 15 | 0.4244 | 1.7821 | 0.4667 | 0.4875 | -0.0208 | -0.1042 | 0.0729 | -0.0931 | -0.2810 | noise+action | -0.0931 | -0.1907 | 0.0045 | 0.1072 | -0.0547 | 0.2635 |
| 3 | leave_one_worker_out | 4 | moe_state | absolute | 352 | 15 | 0.5734 | 3.0723 | 0.4667 | 0.4875 | -0.0208 | -0.1042 | 0.0938 | -0.2422 | -0.7310 | noise+action | -0.2422 | -0.5358 | -0.0684 | -0.0418 | -0.1619 | 0.1156 |
| 3 | leave_one_worker_out | 4 | noise+action+moe_action+state | absolute | 352 | 15 | 0.5264 | 2.8551 | 0.4000 | 0.4875 | -0.0875 | -0.2396 | 0.0729 | -0.1951 | -0.5890 | noise+action | -0.1951 | -0.4509 | -0.0160 | 0.0052 | -0.1303 | 0.1664 |
| 3 | leave_one_worker_out | 4 | noise+action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3387 | 0.9860 | 0.4000 | 0.4875 | -0.0875 | -0.1042 | -0.0729 | -0.0074 | -0.0223 | noise+action_within_snapshot | -0.0074 | -0.0131 | -0.0025 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | 4 | moe_action_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3423 | 1.0127 | 0.4000 | 0.4875 | -0.0875 | -0.1042 | -0.0729 | -0.0111 | -0.0334 | noise+action_within_snapshot | -0.0111 | -0.0200 | -0.0035 | -0.0037 | -0.0069 | 0.0004 |
| 3 | leave_one_worker_out | 4 | moe_state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3404 | 1.0010 | 0.5333 | 0.4875 | 0.0458 | -0.1042 | 0.0938 | -0.0091 | -0.0274 | noise+action_within_snapshot | -0.0091 | -0.0190 | -0.0031 | -0.0017 | -0.0058 | 0.0020 |
| 3 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 352 | 15 | 0.3402 | 1.0011 | 0.4667 | 0.4875 | -0.0208 | -0.1042 | 0.0938 | -0.0090 | -0.0271 | noise+action_within_snapshot | -0.0090 | -0.0202 | -0.0016 | -0.0016 | -0.0070 | 0.0022 |

Snapshot difficulty (one row per rolling state; snapshot and worker holdouts):

| cv_scheme | model | snapshots | heldout_units | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high | mse_gain_vs_policy_state | mse_gain_vs_policy_state_ci95_low | mse_gain_vs_policy_state_ci95_high | mse_gain_vs_geometry | mse_gain_vs_geometry_ci95_low | mse_gain_vs_geometry_ci95_high | mse_gain_vs_sim_state | mse_gain_vs_sim_state_ci95_low | mse_gain_vs_sim_state_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_snapshot_out | initial_geometry | 22 | 22 | 10.0000 | 0.0207 | 0.1214 | 0.1311 | 0.0777 | 0.1887 | 0.1057 | 0.0514 | 0.1629 | 0.0000 | 0.0000 | 0.0000 | 0.0103 | 0.0008 | 0.0218 |
| leave_one_snapshot_out | initial_sim_state | 22 | 22 | 10.0000 | 0.0310 | 0.1429 | 0.1208 | 0.0681 | 0.1814 | 0.0954 | 0.0455 | 0.1472 | -0.0103 | -0.0213 | -0.0009 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_snapshot_out | mean_moe_action | 22 | 22 | 10.0000 | 0.0542 | 0.1992 | 0.0975 | 0.0436 | 0.1565 | 0.0722 | 0.0199 | 0.1276 | -0.0336 | -0.0538 | -0.0155 | -0.0233 | -0.0392 | -0.0092 |
| leave_one_snapshot_out | geometry+mean_moe_action+state | 22 | 22 | 10.0000 | 0.0544 | 0.1986 | 0.0973 | 0.0440 | 0.1561 | 0.0720 | 0.0179 | 0.1297 | -0.0338 | -0.0578 | -0.0133 | -0.0235 | -0.0440 | -0.0055 |
| leave_one_snapshot_out | sim_state+mean_moe_action+state | 22 | 22 | 10.0000 | 0.0548 | 0.1992 | 0.0970 | 0.0420 | 0.1548 | 0.0716 | 0.0158 | 0.1296 | -0.0341 | -0.0584 | -0.0140 | -0.0238 | -0.0443 | -0.0062 |
| leave_one_snapshot_out | policy_state+mean_moe_action+state | 22 | 22 | 10.0000 | 0.0550 | 0.1995 | 0.0968 | 0.0423 | 0.1544 | 0.0714 | 0.0157 | 0.1287 | -0.0343 | -0.0583 | -0.0148 | -0.0240 | -0.0447 | -0.0058 |
| leave_one_snapshot_out | mean_moe_action+state | 22 | 22 | 10.0000 | 0.0551 | 0.1998 | 0.0967 | 0.0422 | 0.1566 | 0.0713 | 0.0168 | 0.1277 | -0.0344 | -0.0586 | -0.0137 | -0.0241 | -0.0454 | -0.0060 |
| leave_one_snapshot_out | mean_moe_state | 22 | 22 | 10.0000 | 0.0644 | 0.2026 | 0.0873 | 0.0303 | 0.1484 | 0.0619 | -0.0016 | 0.1288 | -0.0438 | -0.0798 | -0.0134 | -0.0335 | -0.0664 | -0.0047 |
| leave_one_snapshot_out | initial_policy_state | 22 | 22 | 10.0000 | 0.1264 | 0.2888 | 0.0254 | -0.0290 | 0.0766 | 0.0000 | 0.0000 | 0.0000 | -0.1057 | -0.1641 | -0.0528 | -0.0954 | -0.1487 | -0.0466 |
| leave_one_snapshot_out | mean_rate | 22 | 22 | NA | 0.1518 | 0.3420 | 0.0000 | 0.0000 | 0.0000 | -0.0254 | -0.0753 | 0.0291 | -0.1311 | -0.1876 | -0.0791 | -0.1208 | -0.1770 | -0.0672 |
| leave_one_snapshot_out | as_moe | 22 | 22 | 10.0000 | 0.1518 | 0.3420 | -0.0000 | -0.0000 | 0.0000 | -0.0254 | -0.0768 | 0.0283 | -0.1311 | -0.1886 | -0.0781 | -0.1208 | -0.1789 | -0.0677 |
| leave_one_snapshot_out | mean_output_action | 22 | 22 | 10.0000 | 0.2250 | 0.4042 | -0.0733 | -0.1424 | -0.0095 | -0.0986 | -0.1799 | -0.0158 | -0.2043 | -0.2972 | -0.1201 | -0.1940 | -0.2858 | -0.1118 |
| leave_one_worker_out | initial_geometry | 22 | 4 | 10.0000 | 0.2221 | 0.3731 | 0.0255 | -0.0537 | 0.1267 | 0.0971 | -0.0331 | 0.2685 | 0.0000 | 0.0000 | 0.0000 | 0.0649 | -0.0396 | 0.2491 |
| leave_one_worker_out | as_moe | 22 | 4 | 10.0000 | 0.2476 | 0.4278 | 0.0000 | 0.0000 | 0.0000 | 0.0716 | -0.0577 | 0.3081 | -0.0255 | -0.1267 | 0.0537 | 0.0394 | -0.1335 | 0.2579 |
| leave_one_worker_out | mean_rate | 22 | 4 | NA | 0.2476 | 0.4278 | 0.0000 | 0.0000 | 0.0000 | 0.0716 | -0.0577 | 0.3081 | -0.0255 | -0.1267 | 0.0537 | 0.0394 | -0.1335 | 0.2579 |
| leave_one_worker_out | initial_sim_state | 22 | 4 | 10.0000 | 0.2870 | 0.4330 | -0.0394 | -0.2579 | 0.1335 | 0.0322 | -0.0279 | 0.0987 | -0.0649 | -0.2491 | 0.0396 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_worker_out | mean_moe_action | 22 | 4 | 10.0000 | 0.3030 | 0.4767 | -0.0554 | -0.2042 | 0.0335 | 0.0162 | -0.0435 | 0.1059 | -0.0808 | -0.1866 | 0.0164 | -0.0160 | -0.1145 | 0.0825 |
| leave_one_worker_out | initial_policy_state | 22 | 4 | 10.0000 | 0.3192 | 0.4902 | -0.0716 | -0.3081 | 0.0577 | 0.0000 | 0.0000 | 0.0000 | -0.0971 | -0.2685 | 0.0331 | -0.0322 | -0.0987 | 0.0279 |
| leave_one_worker_out | mean_output_action | 22 | 4 | 10.0000 | 0.3617 | 0.5306 | -0.1140 | -0.2966 | 0.0265 | -0.0424 | -0.1068 | 0.0220 | -0.1395 | -0.2710 | -0.0251 | -0.0747 | -0.1513 | -0.0129 |
| leave_one_worker_out | geometry+mean_moe_action+state | 22 | 4 | 10.0000 | 0.3684 | 0.5113 | -0.1207 | -0.3805 | 0.0154 | -0.0491 | -0.0908 | -0.0023 | -0.1462 | -0.3567 | 0.0187 | -0.0814 | -0.1539 | 0.0196 |
| leave_one_worker_out | sim_state+mean_moe_action+state | 22 | 4 | 10.0000 | 0.3691 | 0.5118 | -0.1214 | -0.3827 | 0.0152 | -0.0498 | -0.0919 | -0.0030 | -0.1469 | -0.3588 | 0.0186 | -0.0821 | -0.1549 | 0.0197 |
| leave_one_worker_out | mean_moe_action+state | 22 | 4 | 10.0000 | 0.3712 | 0.5141 | -0.1236 | -0.3836 | 0.0120 | -0.0520 | -0.0931 | -0.0035 | -0.1491 | -0.3597 | 0.0169 | -0.0842 | -0.1601 | 0.0191 |
| leave_one_worker_out | policy_state+mean_moe_action+state | 22 | 4 | 10.0000 | 0.3712 | 0.5139 | -0.1236 | -0.3842 | 0.0122 | -0.0520 | -0.0933 | -0.0037 | -0.1491 | -0.3603 | 0.0169 | -0.0842 | -0.1600 | 0.0192 |
| leave_one_worker_out | mean_moe_state | 22 | 4 | 10.0000 | 0.3873 | 0.5310 | -0.1397 | -0.3905 | 0.0032 | -0.0681 | -0.1180 | -0.0147 | -0.1652 | -0.3736 | 0.0057 | -0.1003 | -0.1737 | 0.0057 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | front_2_5 | 0 | top1_mass | 0.6167 | 0.3833 | 0.2333 | 0.1000 | 0.3667 | 0.0008 | 0.0572 | 60 | 60 |
| 1 | state | front_2_5 | 1 | top1_mass | 0.5833 | 0.3667 | 0.2167 | 0.0833 | 0.3500 | 0.0014 | 0.1244 | 60 | 60 |
| 1 | state | front_2_5 | 3 | top1_mass | 0.6167 | 0.4333 | 0.1833 | 0.0667 | 0.3167 | 0.0062 | 0.4477 | 60 | 60 |
| 1 | state | front_2_5 | 2 | top1_mass | 0.6167 | 0.4333 | 0.1833 | 0.0500 | 0.3000 | 0.0066 | 0.4477 | 60 | 60 |
| 1 | state | front_2_5 | 7 | entropy | 0.6000 | 0.4333 | 0.1667 | 0.0000 | 0.3167 | 0.0120 | 0.6805 | 60 | 60 |
| 1 | state | front_2_5 | 8 | entropy | 0.5833 | 0.4167 | 0.1667 | 0.0333 | 0.3000 | 0.0140 | 0.6805 | 60 | 60 |
| 1 | state | front_2_5 | 8 | consensus_distance | 0.5833 | 0.4333 | 0.1500 | 0.0333 | 0.2667 | 0.0280 | 0.8792 | 60 | 60 |
| 1 | state | back_12_15 | 7 | consensus_distance | 0.6333 | 0.4833 | 0.1500 | 0.0000 | 0.2833 | 0.0314 | 0.8792 | 60 | 60 |
| 1 | state | front_2_5 | 7 | consensus_distance | 0.5667 | 0.4167 | 0.1500 | 0.0167 | 0.2833 | 0.0326 | 0.8792 | 60 | 60 |
| 1 | state | front_2_5 | 9 | entropy | 0.6167 | 0.4667 | 0.1500 | 0.0167 | 0.2667 | 0.0344 | 0.8792 | 60 | 60 |
| 1 | state | front_2_5 | 6 | consensus_distance | 0.5500 | 0.4167 | 0.1333 | 0.0167 | 0.2500 | 0.0570 | 0.9786 | 60 | 60 |
| 1 | state | front_2_5 | 4 | top1_mass | 0.6000 | 0.4667 | 0.1333 | 0.0333 | 0.2333 | 0.0592 | 0.9786 | 60 | 60 |

Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):

| prefix_queries | token_family | layer_group | denoise_step | expert | failure_minus_success_probability | ci95_low | ci95_high | ci95_worker_low | ci95_worker_high | snapshot_effect_min | snapshot_effect_max | snapshot_effect_sd | snapshots_effect_positive | snapshots_effect_negative | permutation_p_raw | permutation_p_maxT | n_mixed_snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | action | back_12_15 | 0 | 4 | 0.0001 | 0.0000 | 0.0002 | -0.0000 | 0.0002 | -0.0002 | 0.0004 | 0.0002 | 12 | 3 | 0.0010 | 0.7708 | 15 |
| 1 | action | front_2_5 | 0 | 4 | 0.0001 | -0.0000 | 0.0003 | -0.0002 | 0.0003 | -0.0003 | 0.0006 | 0.0003 | 10 | 5 | 0.0262 | 0.8518 | 15 |
| 1 | action | front_2_5 | 2 | 11 | -0.0001 | -0.0002 | -0.0001 | -0.0002 | -0.0000 | -0.0004 | 0.0001 | 0.0002 | 4 | 11 | 0.0150 | 0.8524 | 15 |
| 1 | action | front_2_5 | 4 | 11 | -0.0001 | -0.0002 | -0.0001 | -0.0002 | 0.0000 | -0.0004 | 0.0001 | 0.0001 | 4 | 11 | 0.0076 | 0.8604 | 15 |
| 1 | action | front_2_5 | 9 | 31 | -0.0001 | -0.0002 | -0.0000 | -0.0002 | -0.0001 | -0.0006 | 0.0001 | 0.0002 | 5 | 10 | 0.0044 | 0.8704 | 15 |
| 1 | action | front_2_5 | 4 | 14 | -0.0001 | -0.0003 | 0.0000 | -0.0003 | 0.0002 | -0.0006 | 0.0005 | 0.0003 | 6 | 9 | 0.0366 | 0.8938 | 15 |
| 1 | action | back_12_15 | 0 | 21 | 0.0001 | -0.0000 | 0.0002 | -0.0001 | 0.0003 | -0.0003 | 0.0006 | 0.0003 | 9 | 6 | 0.0006 | 0.8950 | 15 |
| 1 | action | front_2_5 | 1 | 11 | -0.0001 | -0.0002 | -0.0000 | -0.0002 | -0.0000 | -0.0005 | 0.0001 | 0.0002 | 3 | 12 | 0.0224 | 0.9140 | 15 |
| 1 | action | front_2_5 | 3 | 14 | -0.0001 | -0.0002 | 0.0000 | -0.0003 | 0.0002 | -0.0006 | 0.0004 | 0.0003 | 5 | 10 | 0.0524 | 0.9206 | 15 |
| 1 | action | front_2_5 | 0 | 0 | 0.0001 | -0.0002 | 0.0004 | -0.0002 | 0.0006 | -0.0007 | 0.0013 | 0.0006 | 9 | 6 | 0.2939 | 0.9372 | 15 |
| 1 | action | back_12_15 | 7 | 1 | 0.0001 | 0.0000 | 0.0002 | -0.0000 | 0.0002 | -0.0001 | 0.0006 | 0.0002 | 9 | 6 | 0.0048 | 0.9422 | 15 |
| 1 | action | front_2_5 | 0 | 11 | -0.0001 | -0.0002 | -0.0000 | -0.0002 | -0.0000 | -0.0004 | 0.0001 | 0.0001 | 3 | 12 | 0.0280 | 0.9504 | 15 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | cv_scheme | cv_folds | model | feature_scope | n_failure_branches | n_type_positive | heldout_units | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_improvement_vs_rate_ci95_low | brier_improvement_vs_rate_ci95_high | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action | brier_improvement_vs_noise_action_ci95_low | brier_improvement_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_or_regrasp | 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action | absolute | 235 | 30 | 22 | 0.1084 | 0.3907 | noise+action | 0.0043 | -0.0056 | 0.0138 | 0.0384 | 0.0080 | 0.0002 | 0.0175 |
| drop_or_regrasp | 1 | grouped_5fold_snapshot | 5 | moe_action | absolute | 235 | 30 | 22 | 0.1084 | 0.3912 | noise+action | 0.0043 | -0.0051 | 0.0137 | 0.0384 | 0.0080 | -0.0002 | 0.0173 |
| drop_or_regrasp | 1 | grouped_5fold_snapshot | 5 | failure_rate_only | absolute | 235 | 30 | 22 | 0.1127 | 0.3880 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0036 | -0.0035 | 0.0106 |
| drop_or_regrasp | 3 | grouped_5fold_snapshot | 5 | moe_action | absolute | 235 | 30 | 22 | 0.1089 | 0.3818 | noise+action | 0.0037 | -0.0051 | 0.0112 | 0.0333 | 0.0087 | 0.0022 | 0.0156 |
| drop_or_regrasp | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action | absolute | 235 | 30 | 22 | 0.1090 | 0.3818 | noise+action | 0.0036 | -0.0049 | 0.0114 | 0.0323 | 0.0086 | 0.0026 | 0.0156 |
| drop_or_regrasp | 3 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 30 | 22 | 0.1112 | 0.4017 | noise+action_within_snapshot | 0.0015 | -0.0043 | 0.0064 | 0.0131 | 0.0077 | 0.0002 | 0.0175 |
| goal_regression | 1 | grouped_5fold_snapshot | 5 | noise+action_within_snapshot | within_snapshot_centered | 235 | 9 | 22 | 0.0365 | 0.1610 | noise+action_within_snapshot | 0.0008 | -0.0004 | 0.0024 | 0.0209 | 0.0000 | 0.0000 | 0.0000 |
| goal_regression | 1 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 9 | 22 | 0.0367 | 0.1682 | noise+action_within_snapshot | 0.0006 | -0.0033 | 0.0051 | 0.0163 | -0.0002 | -0.0041 | 0.0040 |
| goal_regression | 1 | grouped_5fold_snapshot | 5 | failure_rate_only | absolute | 235 | 9 | 22 | 0.0373 | 0.1689 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 0.0001 | 0.0021 |
| goal_regression | 3 | grouped_5fold_snapshot | 5 | failure_rate_only | absolute | 235 | 9 | 22 | 0.0373 | 0.1689 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0009 | -0.0002 | 0.0021 |
| goal_regression | 3 | grouped_5fold_snapshot | 5 | moe_action+state | absolute | 235 | 9 | 22 | 0.0376 | 0.2946 | noise+action | -0.0003 | -0.0021 | 0.0011 | -0.0086 | 0.0006 | -0.0005 | 0.0017 |
| goal_regression | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state | absolute | 235 | 9 | 22 | 0.0376 | 0.2952 | noise+action | -0.0003 | -0.0022 | 0.0011 | -0.0092 | 0.0006 | -0.0005 | 0.0016 |
| loop_or_cycling | 1 | grouped_5fold_snapshot | 5 | moe_state | absolute | 235 | 163 | 22 | 0.2014 | 0.6075 | noise+action | 0.0202 | -0.0077 | 0.0467 | 0.0912 | 0.0249 | -0.0052 | 0.0560 |
| loop_or_cycling | 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state | absolute | 235 | 163 | 22 | 0.2088 | 0.6201 | noise+action | 0.0128 | -0.0151 | 0.0384 | 0.0579 | 0.0175 | -0.0125 | 0.0482 |
| loop_or_cycling | 1 | grouped_5fold_snapshot | 5 | moe_action+state | absolute | 235 | 163 | 22 | 0.2088 | 0.6205 | noise+action | 0.0128 | -0.0162 | 0.0382 | 0.0578 | 0.0175 | -0.0146 | 0.0499 |
| loop_or_cycling | 3 | grouped_5fold_snapshot | 5 | moe_action | absolute | 235 | 163 | 22 | 0.1935 | 0.5877 | noise+action | 0.0281 | -0.0033 | 0.0558 | 0.1269 | 0.0359 | 0.0036 | 0.0678 |
| loop_or_cycling | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action | absolute | 235 | 163 | 22 | 0.1939 | 0.5876 | noise+action | 0.0277 | -0.0041 | 0.0540 | 0.1250 | 0.0355 | 0.0033 | 0.0672 |
| loop_or_cycling | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state | absolute | 235 | 163 | 22 | 0.1944 | 0.5953 | noise+action | 0.0273 | -0.0023 | 0.0517 | 0.1230 | 0.0350 | 0.0053 | 0.0643 |
| non_stagnation_non_loop | 1 | grouped_5fold_snapshot | 5 | failure_rate_only | absolute | 235 | 36 | 22 | 0.1311 | 0.4333 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0027 | -0.0068 | 0.0126 |
| non_stagnation_non_loop | 1 | grouped_5fold_snapshot | 5 | noise+action_within_snapshot | within_snapshot_centered | 235 | 36 | 22 | 0.1324 | 0.4439 | noise+action_within_snapshot | -0.0013 | -0.0051 | 0.0021 | -0.0099 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 1 | grouped_5fold_snapshot | 5 | noise+action | absolute | 235 | 36 | 22 | 0.1337 | 0.4555 | noise+action | -0.0027 | -0.0122 | 0.0069 | -0.0203 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 3 | grouped_5fold_snapshot | 5 | moe_state_within_snapshot | within_snapshot_centered | 235 | 36 | 22 | 0.1220 | 0.4172 | noise+action_within_snapshot | 0.0091 | -0.0009 | 0.0218 | 0.0695 | 0.0058 | -0.0034 | 0.0168 |
| non_stagnation_non_loop | 3 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 36 | 22 | 0.1222 | 0.4150 | noise+action_within_snapshot | 0.0089 | -0.0004 | 0.0194 | 0.0680 | 0.0056 | -0.0028 | 0.0153 |
| non_stagnation_non_loop | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 36 | 22 | 0.1223 | 0.4153 | noise+action_within_snapshot | 0.0088 | -0.0004 | 0.0195 | 0.0668 | 0.0054 | -0.0030 | 0.0150 |
| single_subtask_omission | 1 | grouped_5fold_snapshot | 5 | moe_state | absolute | 235 | 84 | 22 | 0.2129 | 0.6033 | noise+action | 0.0433 | -0.0129 | 0.1024 | 0.1691 | 0.0516 | -0.0419 | 0.1446 |
| single_subtask_omission | 1 | grouped_5fold_snapshot | 5 | moe_action+state | absolute | 235 | 84 | 22 | 0.2306 | 0.7104 | noise+action | 0.0256 | -0.0548 | 0.0948 | 0.1000 | 0.0339 | -0.0705 | 0.1336 |
| single_subtask_omission | 1 | grouped_5fold_snapshot | 5 | moe_action | absolute | 235 | 84 | 22 | 0.2317 | 0.7259 | noise+action | 0.0246 | -0.0547 | 0.0923 | 0.0959 | 0.0329 | -0.0604 | 0.1248 |
| single_subtask_omission | 3 | grouped_5fold_snapshot | 5 | moe_state | absolute | 235 | 84 | 22 | 0.2034 | 0.5976 | noise+action | 0.0528 | -0.0082 | 0.1092 | 0.2062 | 0.0492 | 0.0151 | 0.0818 |
| single_subtask_omission | 3 | grouped_5fold_snapshot | 5 | moe_action | absolute | 235 | 84 | 22 | 0.2456 | 0.7672 | noise+action | 0.0106 | -0.0652 | 0.0786 | 0.0414 | 0.0070 | -0.0884 | 0.0942 |
| single_subtask_omission | 3 | grouped_5fold_snapshot | 5 | moe_state_within_snapshot | within_snapshot_centered | 235 | 84 | 22 | 0.2465 | 0.6889 | noise+action_within_snapshot | 0.0098 | -0.0069 | 0.0309 | 0.0382 | 0.0067 | -0.0053 | 0.0197 |
| stagnation | 1 | grouped_5fold_snapshot | 5 | moe_state_within_snapshot | within_snapshot_centered | 235 | 49 | 22 | 0.1630 | 0.5119 | noise+action_within_snapshot | 0.0086 | -0.0075 | 0.0226 | 0.0499 | 0.0125 | -0.0055 | 0.0274 |
| stagnation | 1 | grouped_5fold_snapshot | 5 | failure_rate_only | absolute | 235 | 49 | 22 | 0.1715 | 0.5314 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0272 | 0.0077 | 0.0480 |
| stagnation | 1 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 49 | 22 | 0.1741 | 0.5355 | noise+action_within_snapshot | -0.0026 | -0.0102 | 0.0051 | -0.0152 | 0.0013 | -0.0093 | 0.0118 |
| stagnation | 3 | grouped_5fold_snapshot | 5 | failure_rate_only | absolute | 235 | 49 | 22 | 0.1715 | 0.5314 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0314 | 0.0076 | 0.0600 |
| stagnation | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 49 | 22 | 0.1732 | 0.5376 | noise+action_within_snapshot | -0.0016 | -0.0071 | 0.0033 | -0.0096 | 0.0006 | -0.0058 | 0.0069 |
| stagnation | 3 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 49 | 22 | 0.1733 | 0.5380 | noise+action_within_snapshot | -0.0018 | -0.0072 | 0.0036 | -0.0102 | 0.0005 | -0.0064 | 0.0071 |
| subtask_undo | 1 | grouped_5fold_snapshot | 5 | noise+action_within_snapshot | within_snapshot_centered | 235 | 8 | 22 | 0.0323 | 0.1422 | noise+action_within_snapshot | 0.0008 | -0.0004 | 0.0025 | 0.0244 | 0.0000 | 0.0000 | 0.0000 |
| subtask_undo | 1 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 22 | 0.0325 | 0.1550 | noise+action_within_snapshot | 0.0007 | -0.0033 | 0.0050 | 0.0208 | -0.0001 | -0.0040 | 0.0041 |
| subtask_undo | 1 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 22 | 0.0331 | 0.1569 | noise+action_within_snapshot | 0.0000 | -0.0055 | 0.0050 | 0.0003 | -0.0008 | -0.0059 | 0.0041 |
| subtask_undo | 3 | grouped_5fold_snapshot | 5 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 22 | 0.0322 | 0.2487 | noise+action_within_snapshot | 0.0009 | -0.0009 | 0.0030 | 0.0283 | 0.0032 | 0.0005 | 0.0069 |
| subtask_undo | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 22 | 0.0324 | 0.2484 | noise+action_within_snapshot | 0.0007 | -0.0010 | 0.0027 | 0.0226 | 0.0030 | 0.0004 | 0.0067 |
| subtask_undo | 3 | grouped_5fold_snapshot | 5 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 235 | 8 | 22 | 0.0331 | 0.1852 | noise+action_within_snapshot | 0.0001 | -0.0036 | 0.0043 | 0.0022 | 0.0024 | -0.0023 | 0.0076 |
| drop_or_regrasp | 1 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 30 | 4 | 0.1239 | 0.4382 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0071 | -0.0046 | 0.0202 |
| drop_or_regrasp | 1 | leave_one_worker_out | 4 | noise+action_within_snapshot | within_snapshot_centered | 235 | 30 | 4 | 0.1247 | 0.4563 | noise+action_within_snapshot | -0.0008 | -0.0087 | 0.0122 | -0.0063 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 1 | leave_one_worker_out | 4 | noise+action | absolute | 235 | 30 | 4 | 0.1310 | 0.4923 | noise+action | -0.0071 | -0.0202 | 0.0046 | -0.0573 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 3 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 30 | 4 | 0.1239 | 0.4382 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0199 | 0.0039 | 0.0434 |
| drop_or_regrasp | 3 | leave_one_worker_out | 4 | noise+action_within_snapshot | within_snapshot_centered | 235 | 30 | 4 | 0.1260 | 0.4579 | noise+action_within_snapshot | -0.0021 | -0.0096 | 0.0017 | -0.0171 | 0.0000 | 0.0000 | 0.0000 |
| drop_or_regrasp | 3 | leave_one_worker_out | 4 | moe_state_within_snapshot | within_snapshot_centered | 235 | 30 | 4 | 0.1282 | 0.4768 | noise+action_within_snapshot | -0.0043 | -0.0215 | 0.0047 | -0.0344 | -0.0021 | -0.0119 | 0.0041 |
| goal_regression | 1 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 9 | 4 | 0.0374 | 0.3010 | noise+action_within_snapshot | 0.0012 | -0.0011 | 0.0035 | 0.0299 | 0.0011 | -0.0002 | 0.0032 |
| goal_regression | 1 | leave_one_worker_out | 4 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 9 | 4 | 0.0374 | 0.2973 | noise+action_within_snapshot | 0.0011 | -0.0009 | 0.0033 | 0.0294 | 0.0011 | -0.0001 | 0.0031 |
| goal_regression | 1 | leave_one_worker_out | 4 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 235 | 9 | 4 | 0.0385 | 0.2918 | noise+action_within_snapshot | 0.0000 | -0.0004 | 0.0004 | 0.0008 | 0.0000 | -0.0010 | 0.0010 |
| goal_regression | 3 | leave_one_worker_out | 4 | noise+action+moe_action+state | absolute | 235 | 9 | 4 | 0.0384 | 0.5315 | noise+action | 0.0001 | -0.0018 | 0.0012 | 0.0035 | 0.0010 | -0.0002 | 0.0023 |
| goal_regression | 3 | leave_one_worker_out | 4 | noise+action+moe_action | absolute | 235 | 9 | 4 | 0.0384 | 0.4822 | noise+action | 0.0001 | -0.0018 | 0.0012 | 0.0034 | 0.0010 | -0.0001 | 0.0023 |
| goal_regression | 3 | leave_one_worker_out | 4 | moe_action+state | absolute | 235 | 9 | 4 | 0.0384 | 0.5290 | noise+action | 0.0001 | -0.0018 | 0.0012 | 0.0030 | 0.0010 | -0.0001 | 0.0023 |
| loop_or_cycling | 1 | leave_one_worker_out | 4 | moe_state_within_snapshot | within_snapshot_centered | 235 | 163 | 4 | 0.2393 | 0.6904 | noise+action_within_snapshot | 0.0202 | 0.0153 | 0.0276 | 0.0779 | 0.0286 | 0.0252 | 0.0334 |
| loop_or_cycling | 1 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 163 | 4 | 0.2596 | 0.7263 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0178 | -0.0300 | 0.0662 |
| loop_or_cycling | 1 | leave_one_worker_out | 4 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 163 | 4 | 0.2622 | 0.7421 | noise+action_within_snapshot | -0.0026 | -0.0067 | 0.0001 | -0.0101 | 0.0057 | 0.0036 | 0.0074 |
| loop_or_cycling | 3 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 163 | 4 | 0.2596 | 0.7263 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0526 | -0.0311 | 0.1151 |
| loop_or_cycling | 3 | leave_one_worker_out | 4 | moe_state | absolute | 235 | 163 | 4 | 0.2618 | 1.3023 | noise+action | -0.0022 | -0.1108 | 0.0679 | -0.0084 | 0.0505 | -0.0249 | 0.1386 |
| loop_or_cycling | 3 | leave_one_worker_out | 4 | moe_action | absolute | 235 | 163 | 4 | 0.2720 | 0.8540 | noise+action | -0.0124 | -0.0713 | 0.0596 | -0.0478 | 0.0402 | 0.0146 | 0.0767 |
| non_stagnation_non_loop | 1 | leave_one_worker_out | 4 | noise+action | absolute | 235 | 36 | 4 | 0.1407 | 0.4889 | noise+action | 0.0006 | -0.0078 | 0.0067 | 0.0044 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 1 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 36 | 4 | 0.1413 | 0.4716 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0006 | -0.0067 | 0.0078 |
| non_stagnation_non_loop | 1 | leave_one_worker_out | 4 | noise+action_within_snapshot | within_snapshot_centered | 235 | 36 | 4 | 0.1447 | 0.4980 | noise+action_within_snapshot | -0.0034 | -0.0101 | 0.0004 | -0.0242 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 3 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 36 | 4 | 0.1413 | 0.4716 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0105 | -0.0031 | 0.0275 |
| non_stagnation_non_loop | 3 | leave_one_worker_out | 4 | noise+action_within_snapshot | within_snapshot_centered | 235 | 36 | 4 | 0.1434 | 0.4823 | noise+action_within_snapshot | -0.0021 | -0.0092 | 0.0021 | -0.0150 | 0.0000 | 0.0000 | 0.0000 |
| non_stagnation_non_loop | 3 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 36 | 4 | 0.1494 | 0.5348 | noise+action_within_snapshot | -0.0081 | -0.0205 | 0.0073 | -0.0575 | -0.0060 | -0.0187 | 0.0054 |
| single_subtask_omission | 1 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 84 | 4 | 0.3705 | 1.0172 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0356 | -0.0409 | 0.1153 |
| single_subtask_omission | 1 | leave_one_worker_out | 4 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 84 | 4 | 0.3755 | 1.1148 | noise+action_within_snapshot | -0.0050 | -0.0130 | 0.0071 | -0.0135 | 0.0039 | 0.0031 | 0.0055 |
| single_subtask_omission | 1 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 84 | 4 | 0.3760 | 1.1151 | noise+action_within_snapshot | -0.0055 | -0.0136 | 0.0067 | -0.0148 | 0.0034 | 0.0025 | 0.0051 |
| single_subtask_omission | 3 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 84 | 4 | 0.3705 | 1.0172 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0622 | -0.0456 | 0.1747 |
| single_subtask_omission | 3 | leave_one_worker_out | 4 | moe_action | absolute | 235 | 84 | 4 | 0.3723 | 2.0540 | noise+action | -0.0018 | -0.1139 | 0.1568 | -0.0049 | 0.0604 | 0.0136 | 0.1112 |
| single_subtask_omission | 3 | leave_one_worker_out | 4 | noise+action+moe_action | absolute | 235 | 84 | 4 | 0.3788 | 2.2092 | noise+action | -0.0083 | -0.1260 | 0.1533 | -0.0223 | 0.0540 | 0.0094 | 0.1077 |
| stagnation | 1 | leave_one_worker_out | 4 | moe_state_within_snapshot | within_snapshot_centered | 235 | 49 | 4 | 0.1805 | 0.5687 | noise+action_within_snapshot | 0.0107 | -0.0089 | 0.0235 | 0.0558 | 0.0171 | -0.0083 | 0.0310 |
| stagnation | 1 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 49 | 4 | 0.1912 | 0.5916 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0227 | -0.0120 | 0.0493 |
| stagnation | 1 | leave_one_worker_out | 4 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 49 | 4 | 0.1939 | 0.6228 | noise+action_within_snapshot | -0.0027 | -0.0117 | 0.0046 | -0.0143 | 0.0037 | -0.0111 | 0.0127 |
| stagnation | 3 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 49 | 4 | 0.1912 | 0.5916 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0382 | -0.0084 | 0.0705 |
| stagnation | 3 | leave_one_worker_out | 4 | moe_action_within_snapshot | within_snapshot_centered | 235 | 49 | 4 | 0.1930 | 0.6077 | noise+action_within_snapshot | -0.0018 | -0.0044 | 0.0024 | -0.0095 | 0.0020 | -0.0018 | 0.0079 |
| stagnation | 3 | leave_one_worker_out | 4 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 235 | 49 | 4 | 0.1931 | 0.6086 | noise+action_within_snapshot | -0.0019 | -0.0050 | 0.0025 | -0.0101 | 0.0019 | -0.0016 | 0.0073 |
| subtask_undo | 1 | leave_one_worker_out | 4 | moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 4 | 0.0339 | 0.2681 | noise+action_within_snapshot | 0.0002 | -0.0006 | 0.0012 | 0.0058 | 0.0004 | -0.0003 | 0.0014 |
| subtask_undo | 1 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 4 | 0.0339 | 0.2716 | noise+action_within_snapshot | 0.0002 | -0.0008 | 0.0013 | 0.0049 | 0.0004 | -0.0003 | 0.0015 |
| subtask_undo | 1 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 8 | 4 | 0.0341 | 0.1734 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | -0.0004 | 0.0013 |
| subtask_undo | 3 | leave_one_worker_out | 4 | moe_state_within_snapshot | within_snapshot_centered | 235 | 8 | 4 | 0.0341 | 0.3975 | noise+action_within_snapshot | 0.0000 | -0.0016 | 0.0009 | 0.0001 | 0.0003 | -0.0004 | 0.0009 |
| subtask_undo | 3 | leave_one_worker_out | 4 | failure_rate_only | absolute | 235 | 8 | 4 | 0.0341 | 0.1734 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 | -0.0001 | 0.0016 |
| subtask_undo | 3 | leave_one_worker_out | 4 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 235 | 8 | 4 | 0.0341 | 0.3879 | noise+action_within_snapshot | -0.0000 | -0.0016 | 0.0009 | -0.0001 | 0.0003 | -0.0004 | 0.0008 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loop_or_cycling | 3 | action | front_2_5 | 2 | top1_mass | 0.4762 | 0.7857 | -0.3095 | -0.4865 | -0.1628 | 0.0020 | 0.1238 | 42 | 42 |
| loop_or_cycling | 3 | action | back_12_15 | 7 | entropy | 0.7619 | 0.4524 | 0.3095 | 0.1000 | 0.5238 | 0.0032 | 0.1238 | 42 | 42 |
| loop_or_cycling | 3 | action | front_2_5 | 0 | top1_mass | 0.4286 | 0.7381 | -0.3095 | -0.4390 | -0.1591 | 0.0032 | 0.1238 | 42 | 42 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 2 | top1_mass | 0.3421 | 0.0789 | 0.2632 | 0.1220 | 0.4324 | 0.0034 | 0.1680 | 38 | 38 |
| loop_or_cycling | 3 | action | front_2_5 | 4 | top1_mass | 0.5238 | 0.8095 | -0.2857 | -0.4286 | -0.1250 | 0.0056 | 0.2248 | 42 | 42 |
| loop_or_cycling | 3 | action | back_12_15 | 8 | entropy | 0.7857 | 0.5000 | 0.2857 | 0.1429 | 0.4359 | 0.0072 | 0.2248 | 42 | 42 |
| drop_or_regrasp | 3 | action | front_2_5 | 4 | top1_mass | 0.0294 | 0.2647 | -0.2353 | -0.4138 | -0.0833 | 0.0136 | 0.3093 | 34 | 34 |
| single_subtask_omission | 1 | action | back_12_15 | 0 | entropy | 0.4054 | 0.6757 | -0.2703 | -0.4516 | -0.1081 | 0.0044 | 0.3145 | 37 | 37 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 0 | top1_mass | 0.3421 | 0.1053 | 0.2368 | 0.1026 | 0.3714 | 0.0126 | 0.3511 | 38 | 38 |
| loop_or_cycling | 1 | state | back_12_15 | 5 | entropy | 0.4524 | 0.7381 | -0.2857 | -0.4651 | -0.1000 | 0.0052 | 0.3581 | 42 | 42 |
| loop_or_cycling | 3 | action | back_12_15 | 9 | entropy | 0.7381 | 0.5000 | 0.2381 | 0.0555 | 0.4286 | 0.0220 | 0.5667 | 42 | 42 |
| loop_or_cycling | 3 | action | front_2_5 | 1 | top1_mass | 0.5000 | 0.7381 | -0.2381 | -0.4048 | -0.0465 | 0.0230 | 0.5667 | 42 | 42 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
