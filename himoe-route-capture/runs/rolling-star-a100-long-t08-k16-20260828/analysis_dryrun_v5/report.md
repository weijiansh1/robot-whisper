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

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00111.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (snapshots held out together):

| prefix_queries | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | snapshot_rate_only | absolute | 64 | 2 | 0.3418 | 0.9402 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2189 | -0.0008 | 0.5039 |
| 1 | noise | absolute | 64 | 2 | 0.3598 | 1.0409 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0180 | -0.0526 | noise+action | -0.0180 | -0.0356 | 0.0130 | 0.2009 | -0.0443 | 0.4175 |
| 1 | action | absolute | 64 | 2 | 0.6050 | 2.7474 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2632 | -0.7700 | noise+action | -0.2632 | -0.5289 | 0.0569 | -0.0443 | -0.1214 | 0.0047 |
| 1 | noise+action | absolute | 64 | 2 | 0.5607 | 2.0359 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2189 | -0.6404 | noise+action | -0.2189 | -0.4285 | 0.0945 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action | absolute | 64 | 2 | 0.4050 | 3.5222 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0632 | -0.1849 | noise+action | -0.0632 | -0.2160 | 0.0645 | 0.1557 | -0.0530 | 0.4937 |
| 1 | moe_state | absolute | 64 | 2 | 0.3701 | 1.9325 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0283 | -0.0829 | noise+action | -0.0283 | -0.2551 | 0.2080 | 0.1906 | -0.0283 | 0.6072 |
| 1 | moe_action+state | absolute | 64 | 2 | 0.4226 | 3.7116 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0808 | -0.2363 | noise+action | -0.0808 | -0.2572 | 0.0957 | 0.1381 | -0.0202 | 0.4400 |
| 1 | noise+action+moe_action | absolute | 64 | 2 | 0.4106 | 3.5913 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0688 | -0.2014 | noise+action | -0.0688 | -0.2211 | 0.0939 | 0.1500 | -0.0622 | 0.4736 |
| 1 | noise+action+moe_action+state | absolute | 64 | 2 | 0.4239 | 3.7460 | 0.0000 | 0.6875 | -0.6875 | -0.9375 | -0.4375 | -0.0821 | -0.2403 | noise+action | -0.0821 | -0.2577 | 0.1234 | 0.1367 | -0.0197 | 0.3629 |
| 1 | noise_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3582 | 1.0318 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0164 | -0.0480 | noise+action_within_snapshot | -0.0164 | -0.0531 | 0.0029 | -0.0012 | -0.0075 | 0.0015 |
| 1 | action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3506 | 0.9840 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0088 | -0.0258 | noise+action_within_snapshot | -0.0088 | -0.0169 | -0.0033 | 0.0064 | -0.0068 | 0.0297 |
| 1 | noise+action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3570 | 1.0233 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0152 | -0.0444 | noise+action_within_snapshot | -0.0152 | -0.0470 | 0.0018 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3609 | 1.0746 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0191 | -0.0558 | noise+action_within_snapshot | -0.0191 | -0.0370 | -0.0065 | -0.0039 | -0.0154 | 0.0086 |
| 1 | moe_state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.5035 | 5.2285 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.1617 | -0.4730 | noise+action_within_snapshot | -0.1617 | -0.2563 | -0.0232 | -0.1465 | -0.2472 | -0.0236 |
| 1 | moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.4654 | 2.7762 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.1236 | -0.3615 | noise+action_within_snapshot | -0.1236 | -0.2213 | -0.0352 | -0.1084 | -0.2220 | -0.0278 |
| 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3607 | 1.0761 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0189 | -0.0553 | noise+action_within_snapshot | -0.0189 | -0.0392 | -0.0060 | -0.0037 | -0.0141 | 0.0089 |
| 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.4594 | 2.7635 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.1176 | -0.3441 | noise+action_within_snapshot | -0.1176 | -0.2184 | -0.0248 | -0.1024 | -0.2303 | -0.0253 |
| 3 | snapshot_rate_only | absolute | 64 | 2 | 0.3418 | 0.9402 | 0.6875 | 0.6875 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2113 | -0.0262 | 0.4816 |
| 3 | noise | absolute | 64 | 2 | 0.3786 | 1.1362 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0368 | -0.1075 | noise+action | -0.0368 | -0.0910 | -0.0050 | 0.1746 | -0.0627 | 0.3725 |
| 3 | action | absolute | 64 | 2 | 0.5848 | 2.2552 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2430 | -0.7110 | noise+action | -0.2430 | -0.4921 | 0.0887 | -0.0317 | -0.1069 | 0.0445 |
| 3 | noise+action | absolute | 64 | 2 | 0.5531 | 2.0944 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.2113 | -0.6183 | noise+action | -0.2113 | -0.5120 | 0.0691 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action | absolute | 64 | 2 | 0.3574 | 2.5818 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0156 | -0.0457 | noise+action | -0.0156 | -0.2164 | 0.1725 | 0.1957 | -0.0583 | 0.4469 |
| 3 | moe_state | absolute | 64 | 2 | 0.5997 | 3.7915 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.2579 | -0.7546 | noise+action | -0.2579 | -0.6046 | 0.0579 | -0.0466 | -0.5448 | 0.5944 |
| 3 | moe_action+state | absolute | 64 | 2 | 0.3830 | 2.9899 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0412 | -0.1204 | noise+action | -0.0412 | -0.2735 | 0.1930 | 0.1702 | -0.1253 | 0.6574 |
| 3 | noise+action+moe_action | absolute | 64 | 2 | 0.3568 | 2.8989 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0150 | -0.0438 | noise+action | -0.0150 | -0.1941 | 0.1674 | 0.1963 | -0.0391 | 0.5360 |
| 3 | noise+action+moe_action+state | absolute | 64 | 2 | 0.3884 | 3.0546 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0466 | -0.1364 | noise+action | -0.0466 | -0.2742 | 0.1857 | 0.1647 | -0.1815 | 0.6402 |
| 3 | noise_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3755 | 1.1548 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0337 | -0.0985 | noise+action_within_snapshot | -0.0337 | -0.0694 | 0.0030 | -0.0242 | -0.0748 | 0.0014 |
| 3 | action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3444 | 0.9518 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0026 | -0.0075 | noise+action_within_snapshot | -0.0026 | -0.0052 | -0.0003 | 0.0069 | -0.0066 | 0.0233 |
| 3 | noise+action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3513 | 0.9833 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0095 | -0.0278 | noise+action_within_snapshot | -0.0095 | -0.0204 | 0.0018 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3511 | 0.9947 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0093 | -0.0271 | noise+action_within_snapshot | -0.0093 | -0.0209 | -0.0001 | 0.0002 | -0.0054 | 0.0049 |
| 3 | moe_state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.4780 | 3.7599 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.1362 | -0.3986 | noise+action_within_snapshot | -0.1362 | -0.2699 | -0.0005 | -0.1267 | -0.2508 | -0.0001 |
| 3 | moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3498 | 0.9925 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0080 | -0.0233 | noise+action_within_snapshot | -0.0080 | -0.0140 | 0.0000 | 0.0015 | -0.0076 | 0.0114 |
| 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3513 | 0.9948 | 0.5000 | 0.6875 | -0.1875 | -0.4375 | 0.0625 | -0.0095 | -0.0278 | noise+action_within_snapshot | -0.0095 | -0.0232 | -0.0004 | 0.0000 | -0.0042 | 0.0066 |
| 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 64 | 2 | 0.3501 | 0.9910 | 1.0000 | 0.6875 | 0.3125 | 0.0625 | 0.5625 | -0.0083 | -0.0242 | noise+action_within_snapshot | -0.0083 | -0.0157 | -0.0000 | 0.0012 | -0.0099 | 0.0138 |

Snapshot difficulty (one row per rolling state; leave-one-snapshot-out):

| model | snapshots | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initial_policy_state | 4 | 10.0000 | 0.1425 | 0.3298 | 0.1231 | 0.0302 | 0.1839 |
| mean_moe_action | 4 | 10.0000 | 0.1985 | 0.3966 | 0.0672 | 0.0005 | 0.1471 |
| mean_moe_action+state | 4 | 10.0000 | 0.2336 | 0.4336 | 0.0320 | -0.0743 | 0.1397 |
| sim_state+mean_moe_action+state | 4 | 10.0000 | 0.2337 | 0.4334 | 0.0320 | -0.0763 | 0.1192 |
| initial_sim_state | 4 | 10.0000 | 0.2414 | 0.4095 | 0.0242 | -0.1436 | 0.1751 |
| as_moe | 4 | 10.0000 | 0.2656 | 0.4583 | 0.0000 | -0.0000 | 0.0000 |
| mean_rate | 4 | NA | 0.2656 | 0.4583 | 0.0000 | 0.0000 | 0.0000 |
| mean_moe_state | 4 | 10.0000 | 0.2798 | 0.4713 | -0.0142 | -0.1688 | 0.1688 |
| mean_output_action | 4 | 10.0000 | 0.5127 | 0.6206 | -0.2471 | -0.6096 | 0.0792 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | back_12_15 | 8 | top1_mass | 0.1250 | 0.6250 | -0.5000 | -0.7500 | -0.2500 | 0.0198 | 0.7822 | 8 | 8 |
| 1 | state | back_12_15 | 7 | entropy | 0.5000 | 0.1250 | 0.3750 | -0.2500 | 1.0000 | 0.0990 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 7 | top1_mass | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.2500 | 0.0990 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 9 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1089 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 4 | top1_mass | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.2500 | 0.1188 | 1.0000 | 8 | 8 |
| 1 | state | front_2_5 | 9 | top1_mass | 0.2500 | 0.6250 | -0.3750 | -0.5000 | -0.2500 | 0.1188 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 0 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.5000 | -0.2500 | 0.1287 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 6 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1584 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 7 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1584 | 1.0000 | 8 | 8 |
| 1 | action | back_12_15 | 8 | entropy | 0.1250 | 0.5000 | -0.3750 | -0.7500 | 0.0000 | 0.1683 | 1.0000 | 8 | 8 |
| 1 | action | front_2_5 | 9 | entropy | 0.1250 | 0.3750 | -0.2500 | -0.7500 | 0.2500 | 0.2574 | 1.0000 | 8 | 8 |
| 1 | state | back_12_15 | 6 | entropy | 0.1250 | 0.3750 | -0.2500 | -0.2500 | -0.2500 | 0.2871 | 1.0000 | 8 | 8 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | n_failure_branches | n_type_positive | brier | log_loss | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 1 | failure_rate_only | 42 | 11 | 0.2911 | 0.8949 | 0.0000 | 0.0000 | 0.0372 |
| stagnation | 1 | noise+action | 42 | 11 | 0.3283 | 2.5010 | -0.0372 | -0.1278 | 0.0000 |
| stagnation | 1 | moe_action | 42 | 11 | 0.2623 | 3.5060 | 0.0287 | 0.0987 | 0.0659 |
| stagnation | 1 | moe_state | 42 | 11 | 0.6658 | 6.0374 | -0.3747 | -1.2874 | -0.3375 |
| stagnation | 1 | moe_action+state | 42 | 11 | 0.2682 | 3.6481 | 0.0229 | 0.0787 | 0.0601 |
| stagnation | 1 | noise+action+moe_action | 42 | 11 | 0.2624 | 3.5408 | 0.0286 | 0.0984 | 0.0658 |
| stagnation | 1 | noise+action+moe_action+state | 42 | 11 | 0.2702 | 3.6491 | 0.0208 | 0.0716 | 0.0580 |
| stagnation | 3 | failure_rate_only | 42 | 11 | 0.2911 | 0.8949 | 0.0000 | 0.0000 | 0.0264 |
| stagnation | 3 | noise+action | 42 | 11 | 0.3175 | 3.2467 | -0.0264 | -0.0907 | 0.0000 |
| stagnation | 3 | moe_action | 42 | 11 | 0.2920 | 3.7463 | -0.0010 | -0.0033 | 0.0254 |
| stagnation | 3 | moe_state | 42 | 11 | 0.3498 | 3.8278 | -0.0588 | -0.2020 | -0.0324 |
| stagnation | 3 | moe_action+state | 42 | 11 | 0.2835 | 3.7069 | 0.0075 | 0.0259 | 0.0339 |
| stagnation | 3 | noise+action+moe_action | 42 | 11 | 0.2926 | 3.7479 | -0.0015 | -0.0053 | 0.0249 |
| stagnation | 3 | noise+action+moe_action+state | 42 | 11 | 0.2818 | 3.6990 | 0.0092 | 0.0316 | 0.0356 |
| loop_or_cycling | 1 | failure_rate_only | 42 | 26 | 0.3227 | 0.8471 | 0.0000 | 0.0000 | -0.0233 |
| loop_or_cycling | 1 | noise+action | 42 | 26 | 0.2994 | 0.7961 | 0.0233 | 0.0723 | 0.0000 |
| loop_or_cycling | 1 | moe_action | 42 | 26 | 0.3849 | 1.0055 | -0.0622 | -0.1926 | -0.0855 |
| loop_or_cycling | 1 | moe_state | 42 | 26 | 0.6118 | 6.6994 | -0.2891 | -0.8959 | -0.3125 |
| loop_or_cycling | 1 | moe_action+state | 42 | 26 | 0.3903 | 1.2748 | -0.0676 | -0.2094 | -0.0909 |
| loop_or_cycling | 1 | noise+action+moe_action | 42 | 26 | 0.3729 | 0.9676 | -0.0502 | -0.1554 | -0.0735 |
| loop_or_cycling | 1 | noise+action+moe_action+state | 42 | 26 | 0.3815 | 1.1426 | -0.0588 | -0.1821 | -0.0821 |
| loop_or_cycling | 3 | failure_rate_only | 42 | 26 | 0.3227 | 0.8471 | 0.0000 | 0.0000 | 0.0137 |
| loop_or_cycling | 3 | noise+action | 42 | 26 | 0.3364 | 0.9966 | -0.0137 | -0.0425 | 0.0000 |
| loop_or_cycling | 3 | moe_action | 42 | 26 | 0.4445 | 1.2281 | -0.1218 | -0.3775 | -0.1081 |
| loop_or_cycling | 3 | moe_state | 42 | 26 | 0.6597 | 5.9951 | -0.3370 | -1.0443 | -0.3233 |
| loop_or_cycling | 3 | moe_action+state | 42 | 26 | 0.5950 | 2.7856 | -0.2723 | -0.8438 | -0.2586 |
| loop_or_cycling | 3 | noise+action+moe_action | 42 | 26 | 0.4383 | 1.2066 | -0.1156 | -0.3581 | -0.1018 |
| loop_or_cycling | 3 | noise+action+moe_action+state | 42 | 26 | 0.6008 | 2.7833 | -0.2781 | -0.8616 | -0.2643 |
| single_subtask_omission | 1 | failure_rate_only | 42 | 18 | 0.4177 | 1.0757 | 0.0000 | 0.0000 | 0.1443 |
| single_subtask_omission | 1 | noise+action | 42 | 18 | 0.5620 | 1.8471 | -0.1443 | -0.3455 | 0.0000 |
| single_subtask_omission | 1 | moe_action | 42 | 18 | 0.4464 | 2.1599 | -0.0287 | -0.0687 | 0.1156 |
| single_subtask_omission | 1 | moe_state | 42 | 18 | 0.7020 | 6.0665 | -0.2843 | -0.6806 | -0.1400 |
| single_subtask_omission | 1 | moe_action+state | 42 | 18 | 0.4931 | 3.2029 | -0.0754 | -0.1806 | 0.0688 |
| single_subtask_omission | 1 | noise+action+moe_action | 42 | 18 | 0.4482 | 2.1765 | -0.0305 | -0.0730 | 0.1138 |
| single_subtask_omission | 1 | noise+action+moe_action+state | 42 | 18 | 0.4910 | 3.1126 | -0.0733 | -0.1754 | 0.0710 |
| single_subtask_omission | 3 | failure_rate_only | 42 | 18 | 0.4177 | 1.0757 | 0.0000 | 0.0000 | -0.1746 |
| single_subtask_omission | 3 | noise+action | 42 | 18 | 0.2431 | 0.7734 | 0.1746 | 0.4181 | 0.0000 |
| single_subtask_omission | 3 | moe_action | 42 | 18 | 0.2210 | 0.7271 | 0.1967 | 0.4709 | 0.0221 |
| single_subtask_omission | 3 | moe_state | 42 | 18 | 0.7012 | 5.8813 | -0.2835 | -0.6787 | -0.4581 |
| single_subtask_omission | 3 | moe_action+state | 42 | 18 | 0.4281 | 2.0752 | -0.0104 | -0.0250 | -0.1850 |
| single_subtask_omission | 3 | noise+action+moe_action | 42 | 18 | 0.2122 | 0.7952 | 0.2055 | 0.4920 | 0.0309 |
| single_subtask_omission | 3 | noise+action+moe_action+state | 42 | 18 | 0.2804 | 1.1615 | 0.1373 | 0.3287 | -0.0373 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_subtask_omission | 3 | action | back_12_15 | 5 | entropy | 0.7000 | 0.2000 | 0.5000 | 0.2500 | 0.8812 | 0.0198 | 0.2277 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 1 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.8219 | -0.2500 | 0.0198 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 4 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.8219 | -0.2500 | 0.0198 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 0 | entropy | 0.6000 | 0.2000 | 0.4000 | 0.0000 | 0.8812 | 0.0297 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 3 | entropy | 0.7000 | 0.3000 | 0.4000 | 0.2500 | 0.8219 | 0.0297 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 3 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.8219 | -0.2500 | 0.0297 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 1 | entropy | 0.7000 | 0.3000 | 0.4000 | 0.2500 | 0.8219 | 0.0396 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 2 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.8219 | -0.2500 | 0.0396 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 5 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.8219 | -0.2500 | 0.0396 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 0 | top1_mass | 0.3000 | 0.7000 | -0.4000 | -0.8219 | -0.2500 | 0.0495 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 2 | entropy | 0.7000 | 0.3000 | 0.4000 | 0.2500 | 0.8219 | 0.0495 | 0.5347 | 10 | 10 |
| single_subtask_omission | 3 | action | back_12_15 | 4 | entropy | 0.7000 | 0.3000 | 0.4000 | 0.2500 | 0.8219 | 0.0495 | 0.5347 | 10 | 10 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
