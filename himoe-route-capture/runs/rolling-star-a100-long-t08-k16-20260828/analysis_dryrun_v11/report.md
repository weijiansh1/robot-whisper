# Rolling-star K=16 experiment

## Dataset

- 112 terminal branches from 7 committed snapshots; 41 success and 71 failure.
- 4 snapshots contain matched success/failure siblings.
- 5283 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 44
- `single_subtask_omission`: 13
- `drop_or_regrasp`: 6
- `goal_contact_near_miss`: 3
- `subtask_undo`: 3
- `stagnation`: 1
- `active_retry`: 1

Stagnation or loop/cycling covers 63/71 failures; 8 failures require other physical mechanisms: {"active_retry": 1, "goal_contact_near_miss": 3, "single_subtask_omission": 4}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00111.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (both snapshot-held-out and core worker-held-out checks):

| prefix_queries | cv_scheme | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | leave_one_snapshot_out | snapshot_rate_only | absolute | 112 | 4 | 0.2954 | 0.8016 | 0.6406 | 0.6406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.0856 | 0.5166 |
| 1 | leave_one_snapshot_out | noise | absolute | 112 | 4 | 0.3262 | 0.8938 | 0.5000 | 0.6406 | -0.1406 | -0.3684 | -0.0379 | -0.0308 | -0.1043 | noise+action | -0.0308 | -0.0425 | -0.0119 | 0.2192 | 0.0026 | 0.4143 |
| 1 | leave_one_snapshot_out | action | absolute | 112 | 4 | 0.5020 | 1.6350 | 0.5000 | 0.6406 | -0.1406 | -0.3289 | 0.0625 | -0.2067 | -0.6997 | noise+action | -0.2067 | -0.4256 | -0.0503 | 0.0434 | 0.0114 | 0.0845 |
| 1 | leave_one_snapshot_out | noise+action | absolute | 112 | 4 | 0.5454 | 1.8397 | 0.5000 | 0.6406 | -0.1406 | -0.3066 | 0.0031 | -0.2500 | -0.8465 | noise+action | -0.2500 | -0.4286 | -0.0639 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action | absolute | 112 | 4 | 0.3159 | 1.2102 | 0.5000 | 0.6406 | -0.1406 | -0.3125 | 0.0625 | -0.0205 | -0.0695 | noise+action | -0.0205 | -0.0680 | 0.0491 | 0.2295 | 0.0495 | 0.4313 |
| 1 | leave_one_snapshot_out | moe_state | absolute | 112 | 4 | 0.4480 | 1.6680 | 0.5000 | 0.6406 | -0.1406 | -0.2656 | 0.0254 | -0.1527 | -0.5168 | noise+action | -0.1527 | -0.3474 | -0.0123 | 0.0974 | -0.1047 | 0.3604 |
| 1 | leave_one_snapshot_out | moe_action+state | absolute | 112 | 4 | 0.4278 | 1.8767 | 0.5000 | 0.6406 | -0.1406 | -0.3312 | -0.0156 | -0.1324 | -0.4483 | noise+action | -0.1324 | -0.2716 | -0.0565 | 0.1176 | -0.0539 | 0.3180 |
| 1 | leave_one_snapshot_out | noise+action+moe_action | absolute | 112 | 4 | 0.3212 | 1.1779 | 0.5000 | 0.6406 | -0.1406 | -0.3125 | 0.0625 | -0.0259 | -0.0876 | noise+action | -0.0259 | -0.1295 | 0.0338 | 0.2242 | 0.1123 | 0.5155 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 112 | 4 | 0.4296 | 1.8147 | 0.5000 | 0.6406 | -0.1406 | -0.2680 | 0.0254 | -0.1342 | -0.4543 | noise+action | -0.1342 | -0.3150 | 0.0404 | 0.1158 | -0.0105 | 0.3086 |
| 1 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 112 | 4 | 0.2983 | 0.8112 | 0.7500 | 0.6406 | 0.1094 | -0.3125 | 0.5781 | -0.0030 | -0.0100 | noise+action_within_snapshot | -0.0030 | -0.0072 | 0.0005 | 0.0019 | -0.0016 | 0.0043 |
| 1 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.2996 | 0.8133 | 0.5000 | 0.6406 | -0.1406 | -0.3781 | 0.0625 | -0.0042 | -0.0143 | noise+action_within_snapshot | -0.0042 | -0.0066 | -0.0005 | 0.0006 | -0.0006 | 0.0023 |
| 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3002 | 0.8146 | 0.5000 | 0.6406 | -0.1406 | -0.3289 | 0.0254 | -0.0049 | -0.0165 | noise+action_within_snapshot | -0.0049 | -0.0073 | -0.0018 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3031 | 0.8241 | 0.5000 | 0.6406 | -0.1406 | -0.3535 | 0.0254 | -0.0078 | -0.0263 | noise+action_within_snapshot | -0.0078 | -0.0154 | -0.0015 | -0.0029 | -0.0076 | 0.0016 |
| 1 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.5407 | 3.2701 | 0.5000 | 0.6406 | -0.1406 | -0.3066 | 0.0625 | -0.2453 | -0.8305 | noise+action_within_snapshot | -0.2453 | -0.3207 | -0.1570 | -0.2404 | -0.3596 | -0.1504 |
| 1 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3107 | 0.8743 | 0.7500 | 0.6406 | 0.1094 | -0.0938 | 0.3781 | -0.0153 | -0.0518 | noise+action_within_snapshot | -0.0153 | -0.0290 | -0.0065 | -0.0104 | -0.0192 | -0.0029 |
| 1 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3030 | 0.8236 | 0.5000 | 0.6406 | -0.1406 | -0.3051 | -0.0379 | -0.0076 | -0.0259 | noise+action_within_snapshot | -0.0076 | -0.0127 | -0.0028 | -0.0028 | -0.0079 | 0.0009 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3102 | 0.8786 | 0.7500 | 0.6406 | 0.1094 | -0.0938 | 0.3125 | -0.0148 | -0.0501 | noise+action_within_snapshot | -0.0148 | -0.0297 | 0.0002 | -0.0099 | -0.0220 | -0.0022 |
| 3 | leave_one_snapshot_out | snapshot_rate_only | absolute | 112 | 4 | 0.2954 | 0.8016 | 0.6406 | 0.6406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2804 | 0.1079 | 0.4510 |
| 3 | leave_one_snapshot_out | noise | absolute | 112 | 4 | 0.3269 | 0.9194 | 0.5000 | 0.6406 | -0.1406 | -0.2902 | 0.0625 | -0.0315 | -0.1067 | noise+action | -0.0315 | -0.0555 | 0.0026 | 0.2489 | 0.1017 | 0.4692 |
| 3 | leave_one_snapshot_out | action | absolute | 112 | 4 | 0.5964 | 2.6542 | 0.5000 | 0.6406 | -0.1406 | -0.2902 | 0.0625 | -0.3011 | -1.0194 | noise+action | -0.3011 | -0.5513 | -0.1694 | -0.0207 | -0.0611 | 0.0352 |
| 3 | leave_one_snapshot_out | noise+action | absolute | 112 | 4 | 0.5757 | 2.6443 | 0.5000 | 0.6406 | -0.1406 | -0.2902 | 0.0254 | -0.2804 | -0.9493 | noise+action | -0.2804 | -0.4434 | -0.0877 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action | absolute | 112 | 4 | 0.2696 | 2.3729 | 0.7500 | 0.6406 | 0.1094 | -0.2531 | 0.4965 | 0.0258 | 0.0872 | noise+action | 0.0258 | -0.0583 | 0.1769 | 0.3061 | 0.0909 | 0.5725 |
| 3 | leave_one_snapshot_out | moe_state | absolute | 112 | 4 | 0.5246 | 2.3889 | 0.7500 | 0.6406 | 0.1094 | -0.2531 | 0.3469 | -0.2292 | -0.7762 | noise+action | -0.2292 | -0.4916 | -0.0258 | 0.0511 | -0.0877 | 0.1290 |
| 3 | leave_one_snapshot_out | moe_action+state | absolute | 112 | 4 | 0.2814 | 2.8069 | 0.7500 | 0.6406 | 0.1094 | -0.0625 | 0.4309 | 0.0139 | 0.0471 | noise+action | 0.0139 | -0.1766 | 0.1318 | 0.2943 | 0.0272 | 0.4694 |
| 3 | leave_one_snapshot_out | noise+action+moe_action | absolute | 112 | 4 | 0.2790 | 2.4085 | 0.7500 | 0.6406 | 0.1094 | -0.2531 | 0.4965 | 0.0164 | 0.0554 | noise+action | 0.0164 | -0.2104 | 0.1567 | 0.2968 | 0.1190 | 0.4441 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 112 | 4 | 0.2788 | 2.8117 | 0.7500 | 0.6406 | 0.1094 | -0.1875 | 0.4965 | 0.0165 | 0.0560 | noise+action | 0.0165 | -0.1564 | 0.1076 | 0.2969 | 0.0511 | 0.5084 |
| 3 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3003 | 0.8162 | 0.5000 | 0.6406 | -0.1406 | -0.3684 | 0.0625 | -0.0050 | -0.0169 | noise+action_within_snapshot | -0.0050 | -0.0080 | -0.0006 | -0.0011 | -0.0037 | 0.0022 |
| 3 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.2999 | 0.8241 | 1.0000 | 0.6406 | 0.3594 | 0.0625 | 0.6094 | -0.0046 | -0.0154 | noise+action_within_snapshot | -0.0046 | -0.0091 | 0.0023 | -0.0007 | -0.0055 | 0.0034 |
| 3 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.2992 | 0.8135 | 0.5000 | 0.6406 | -0.1406 | -0.3066 | 0.0254 | -0.0039 | -0.0131 | noise+action_within_snapshot | -0.0039 | -0.0099 | -0.0006 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.2959 | 0.8056 | 0.5000 | 0.6406 | -0.1406 | -0.3684 | -0.0156 | -0.0006 | -0.0019 | noise+action_within_snapshot | -0.0006 | -0.0015 | 0.0006 | 0.0033 | -0.0010 | 0.0071 |
| 3 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3109 | 0.8811 | 1.0000 | 0.6406 | 0.3594 | 0.1219 | 0.5969 | -0.0156 | -0.0527 | noise+action_within_snapshot | -0.0156 | -0.0342 | 0.0006 | -0.0117 | -0.0269 | -0.0002 |
| 3 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3039 | 0.8344 | 0.5000 | 0.6406 | -0.1406 | -0.2844 | -0.0379 | -0.0085 | -0.0288 | noise+action_within_snapshot | -0.0085 | -0.0160 | -0.0007 | -0.0046 | -0.0188 | 0.0034 |
| 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.2960 | 0.8057 | 0.5000 | 0.6406 | -0.1406 | -0.3090 | -0.0156 | -0.0006 | -0.0021 | noise+action_within_snapshot | -0.0006 | -0.0020 | 0.0007 | 0.0033 | -0.0006 | 0.0073 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3046 | 0.8379 | 0.5000 | 0.6406 | -0.1406 | -0.2656 | 0.0031 | -0.0092 | -0.0312 | noise+action_within_snapshot | -0.0092 | -0.0252 | -0.0021 | -0.0053 | -0.0244 | 0.0035 |
| 1 | leave_one_worker_out | snapshot_rate_only | absolute | 112 | 4 | 0.3654 | 1.0034 | 0.6406 | 0.6406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2396 | 0.1385 | 0.5642 |
| 1 | leave_one_worker_out | noise+action | absolute | 112 | 4 | 0.6050 | 2.7438 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.2396 | -0.6556 | noise+action | -0.2396 | -0.5417 | -0.1226 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action | absolute | 112 | 4 | 0.4998 | 2.4889 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.1343 | -0.3676 | noise+action | -0.1343 | -0.1891 | -0.0847 | 0.1052 | -0.0087 | 0.4361 |
| 1 | leave_one_worker_out | moe_state | absolute | 112 | 4 | 0.5235 | 2.1201 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.1581 | -0.4325 | noise+action | -0.1581 | -0.2463 | -0.0763 | 0.0815 | -0.0808 | 0.4172 |
| 1 | leave_one_worker_out | noise+action+moe_action+state | absolute | 112 | 4 | 0.5503 | 2.9211 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.1848 | -0.5058 | noise+action | -0.1848 | -0.2651 | -0.1047 | 0.0547 | -0.0708 | 0.4524 |
| 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3707 | 1.0415 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0053 | -0.0145 | noise+action_within_snapshot | -0.0053 | -0.0113 | 0.0002 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3757 | 1.0955 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0103 | -0.0281 | noise+action_within_snapshot | -0.0103 | -0.0155 | -0.0062 | -0.0050 | -0.0086 | -0.0024 |
| 1 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.5388 | 2.9178 | 1.0000 | 0.6406 | 0.3594 | 0.0625 | 0.6562 | -0.1734 | -0.4745 | noise+action_within_snapshot | -0.1734 | -0.2110 | -0.1343 | -0.1681 | -0.2831 | -0.0977 |
| 1 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3968 | 1.1745 | 0.7500 | 0.6406 | 0.1094 | 0.0625 | 0.1562 | -0.0313 | -0.0858 | noise+action_within_snapshot | -0.0313 | -0.0540 | -0.0084 | -0.0260 | -0.0468 | -0.0094 |
| 3 | leave_one_worker_out | snapshot_rate_only | absolute | 112 | 4 | 0.3654 | 1.0034 | 0.6406 | 0.6406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2512 | 0.1753 | 0.4749 |
| 3 | leave_one_worker_out | noise+action | absolute | 112 | 4 | 0.6166 | 2.4571 | 0.7500 | 0.6406 | 0.1094 | 0.0625 | 0.1562 | -0.2512 | -0.6874 | noise+action | -0.2512 | -0.4705 | -0.1297 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action | absolute | 112 | 4 | 0.5083 | 2.8256 | 0.7500 | 0.6406 | 0.1094 | 0.0625 | 0.1562 | -0.1428 | -0.3909 | noise+action | -0.1428 | -0.2072 | -0.0242 | 0.1083 | -0.0352 | 0.4479 |
| 3 | leave_one_worker_out | moe_state | absolute | 112 | 4 | 0.6594 | 3.3941 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.2940 | -0.8045 | noise+action | -0.2940 | -0.3962 | -0.2214 | -0.0428 | -0.1687 | 0.1007 |
| 3 | leave_one_worker_out | noise+action+moe_action+state | absolute | 112 | 4 | 0.5718 | 3.7448 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.2064 | -0.5649 | noise+action | -0.2064 | -0.2959 | -0.0396 | 0.0448 | -0.2246 | 0.3414 |
| 3 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3788 | 1.0678 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0134 | -0.0366 | noise+action_within_snapshot | -0.0134 | -0.0278 | -0.0025 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3732 | 1.0653 | 0.5000 | 0.6406 | -0.1406 | -0.3438 | 0.0625 | -0.0078 | -0.0214 | noise+action_within_snapshot | -0.0078 | -0.0155 | -0.0020 | 0.0056 | 0.0007 | 0.0120 |
| 3 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3885 | 1.1298 | 0.7500 | 0.6406 | 0.1094 | 0.0625 | 0.1562 | -0.0230 | -0.0630 | noise+action_within_snapshot | -0.0230 | -0.0490 | -0.0089 | -0.0096 | -0.0372 | 0.0080 |
| 3 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 112 | 4 | 0.3819 | 1.0880 | 0.7500 | 0.6406 | 0.1094 | 0.0625 | 0.1562 | -0.0165 | -0.0451 | noise+action_within_snapshot | -0.0165 | -0.0293 | -0.0057 | -0.0031 | -0.0126 | 0.0020 |

Snapshot difficulty (one row per rolling state; snapshot and worker holdouts):

| cv_scheme | model | snapshots | heldout_units | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high | mse_gain_vs_policy_state | mse_gain_vs_policy_state_ci95_low | mse_gain_vs_policy_state_ci95_high | mse_gain_vs_sim_state | mse_gain_vs_sim_state_ci95_low | mse_gain_vs_sim_state_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_snapshot_out | initial_sim_state | 7 | 7 | 10.0000 | 0.1294 | 0.2939 | 0.0793 | -0.0323 | 0.1521 | 0.0740 | -0.0486 | 0.1806 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_snapshot_out | mean_moe_action | 7 | 7 | 10.0000 | 0.1331 | 0.3363 | 0.0757 | -0.0357 | 0.1938 | 0.0703 | -0.0649 | 0.1963 | -0.0037 | -0.0861 | 0.0446 |
| leave_one_snapshot_out | sim_state+mean_moe_action+state | 7 | 7 | 10.0000 | 0.2021 | 0.4026 | 0.0066 | -0.1162 | 0.1091 | 0.0013 | -0.1349 | 0.1408 | -0.0727 | -0.1606 | 0.0190 |
| leave_one_snapshot_out | policy_state+mean_moe_action+state | 7 | 7 | 10.0000 | 0.2023 | 0.4029 | 0.0065 | -0.0657 | 0.1157 | 0.0011 | -0.0750 | 0.1147 | -0.0729 | -0.2370 | 0.0226 |
| leave_one_snapshot_out | mean_moe_action+state | 7 | 7 | 10.0000 | 0.2025 | 0.4031 | 0.0062 | -0.0716 | 0.1255 | 0.0009 | -0.1134 | 0.1117 | -0.0731 | -0.1783 | 0.0519 |
| leave_one_snapshot_out | initial_policy_state | 7 | 7 | 10.0000 | 0.2034 | 0.3793 | 0.0053 | -0.0712 | 0.0772 | 0.0000 | 0.0000 | 0.0000 | -0.0740 | -0.2285 | 0.0395 |
| leave_one_snapshot_out | mean_rate | 7 | 7 | NA | 0.2088 | 0.4048 | 0.0000 | 0.0000 | 0.0000 | -0.0053 | -0.0677 | 0.0815 | -0.0793 | -0.1944 | 0.0341 |
| leave_one_snapshot_out | as_moe | 7 | 7 | 10.0000 | 0.2088 | 0.4048 | -0.0000 | -0.0000 | 0.0000 | -0.0053 | -0.0575 | 0.0487 | -0.0793 | -0.1489 | 0.0675 |
| leave_one_snapshot_out | mean_output_action | 7 | 7 | 10.0000 | 0.2925 | 0.4900 | -0.0837 | -0.3270 | 0.1266 | -0.0890 | -0.2582 | 0.1451 | -0.1630 | -0.2476 | -0.0308 |
| leave_one_snapshot_out | mean_moe_state | 7 | 7 | 10.0000 | 0.2926 | 0.4564 | -0.0839 | -0.2959 | 0.0465 | -0.0892 | -0.2132 | 0.0959 | -0.1632 | -0.2964 | -0.0270 |
| leave_one_worker_out | initial_sim_state | 7 | 4 | 10.0000 | 0.2016 | 0.3284 | 0.0852 | -0.0212 | 0.2285 | 0.0601 | -0.1262 | 0.1341 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_worker_out | initial_policy_state | 7 | 4 | 10.0000 | 0.2617 | 0.4382 | 0.0251 | -0.0497 | 0.0657 | 0.0000 | 0.0000 | 0.0000 | -0.0601 | -0.1288 | 0.0315 |
| leave_one_worker_out | as_moe | 7 | 4 | 10.0000 | 0.2867 | 0.4628 | 0.0000 | -0.0000 | 0.0000 | -0.0251 | -0.1185 | 0.0434 | -0.0852 | -0.2285 | 0.0437 |
| leave_one_worker_out | mean_rate | 7 | 4 | NA | 0.2867 | 0.4628 | 0.0000 | 0.0000 | 0.0000 | -0.0251 | -0.0842 | 0.0256 | -0.0852 | -0.1674 | 0.0685 |
| leave_one_worker_out | mean_moe_action | 7 | 4 | 10.0000 | 0.3162 | 0.4777 | -0.0295 | -0.0478 | -0.0163 | -0.0546 | -0.1647 | 0.0103 | -0.1147 | -0.2337 | 0.0297 |
| leave_one_worker_out | sim_state+mean_moe_action+state | 7 | 4 | 10.0000 | 0.3478 | 0.5136 | -0.0611 | -0.0901 | -0.0254 | -0.0862 | -0.1549 | -0.0132 | -0.1463 | -0.2156 | -0.0288 |
| leave_one_worker_out | policy_state+mean_moe_action+state | 7 | 4 | 10.0000 | 0.3487 | 0.5148 | -0.0620 | -0.0930 | -0.0356 | -0.0871 | -0.1738 | -0.0445 | -0.1472 | -0.2058 | -0.0300 |
| leave_one_worker_out | mean_moe_action+state | 7 | 4 | 10.0000 | 0.3490 | 0.5150 | -0.0622 | -0.0881 | -0.0330 | -0.0873 | -0.1768 | -0.0199 | -0.1474 | -0.2651 | -0.0087 |
| leave_one_worker_out | mean_moe_state | 7 | 4 | 10.0000 | 0.4153 | 0.5708 | -0.1286 | -0.2023 | -0.0832 | -0.1537 | -0.2339 | -0.0905 | -0.2138 | -0.2709 | -0.0670 |
| leave_one_worker_out | mean_output_action | 7 | 4 | 10.0000 | 0.4809 | 0.6080 | -0.1942 | -0.3842 | -0.0708 | -0.2193 | -0.5360 | -0.0422 | -0.2794 | -0.4486 | -0.1600 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | back_12_15 | 8 | top1_mass | 0.1875 | 0.5000 | -0.3125 | -0.5656 | -0.0922 | 0.1818 | 0.9091 | 16 | 16 |
| 1 | action | back_12_15 | 7 | top1_mass | 0.1875 | 0.3750 | -0.1875 | -0.2500 | 0.0000 | 0.0909 | 1.0000 | 16 | 16 |
| 1 | action | front_2_5 | 9 | token_dispersion | 0.5000 | 0.3125 | 0.1875 | -0.1578 | 0.4406 | 0.0909 | 1.0000 | 16 | 16 |
| 1 | state | back_12_15 | 0 | top1_mass | 0.1875 | 0.4375 | -0.2500 | -0.3750 | 0.0000 | 0.0909 | 1.0000 | 16 | 16 |
| 1 | state | back_12_15 | 3 | entropy | 0.5000 | 0.2500 | 0.2500 | 0.1219 | 0.3750 | 0.0909 | 1.0000 | 16 | 16 |
| 1 | state | front_2_5 | 0 | consensus_distance | 0.3750 | 0.1875 | 0.1875 | 0.0922 | 0.2500 | 0.0909 | 1.0000 | 16 | 16 |
| 1 | state | front_2_5 | 0 | top1_mass | 0.5000 | 0.3125 | 0.1875 | 0.0922 | 0.2500 | 0.0909 | 1.0000 | 16 | 16 |
| 1 | action | back_12_15 | 5 | top1_mass | 0.1875 | 0.3750 | -0.1875 | -0.2500 | -0.0922 | 0.1818 | 1.0000 | 16 | 16 |
| 1 | action | back_12_15 | 6 | top1_mass | 0.1875 | 0.3750 | -0.1875 | -0.3453 | -0.0922 | 0.1818 | 1.0000 | 16 | 16 |
| 1 | state | back_12_15 | 7 | entropy | 0.4375 | 0.1875 | 0.2500 | -0.1578 | 0.5984 | 0.1818 | 1.0000 | 16 | 16 |
| 1 | state | front_2_5 | 1 | top1_mass | 0.4375 | 0.2500 | 0.1875 | 0.0922 | 0.2500 | 0.1818 | 1.0000 | 16 | 16 |
| 1 | action | back_12_15 | 0 | entropy | 0.2500 | 0.5000 | -0.2500 | -0.4078 | -0.0922 | 0.2727 | 1.0000 | 16 | 16 |

Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):

| prefix_queries | token_family | layer_group | denoise_step | expert | failure_minus_success_probability | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_mixed_snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | action | front_2_5 | 0 | 0 | 0.0004 | -0.0001 | 0.0008 | 0.0909 | 0.4545 | 4 |
| 1 | action | front_2_5 | 1 | 0 | 0.0003 | -0.0001 | 0.0007 | 0.0909 | 0.4545 | 4 |
| 1 | action | front_2_5 | 4 | 14 | -0.0003 | -0.0004 | -0.0002 | 0.0909 | 0.4545 | 4 |
| 1 | action | front_2_5 | 5 | 14 | -0.0003 | -0.0005 | -0.0002 | 0.0909 | 0.4545 | 4 |
| 1 | action | front_2_5 | 7 | 12 | 0.0003 | 0.0000 | 0.0006 | 0.0909 | 0.4545 | 4 |
| 1 | action | front_2_5 | 9 | 12 | 0.0003 | 0.0000 | 0.0006 | 0.0909 | 0.4545 | 4 |
| 1 | action | front_2_5 | 3 | 14 | -0.0003 | -0.0004 | -0.0002 | 0.0909 | 0.5455 | 4 |
| 1 | action | front_2_5 | 8 | 12 | 0.0003 | -0.0000 | 0.0006 | 0.0909 | 0.7273 | 4 |
| 1 | action | front_2_5 | 2 | 14 | -0.0003 | -0.0003 | -0.0002 | 0.0909 | 0.8182 | 4 |
| 1 | action | front_2_5 | 6 | 12 | 0.0003 | 0.0000 | 0.0005 | 0.0909 | 0.8182 | 4 |
| 1 | action | front_2_5 | 6 | 14 | -0.0003 | -0.0004 | -0.0002 | 0.0909 | 0.8182 | 4 |
| 1 | action | front_2_5 | 9 | 31 | -0.0002 | -0.0004 | -0.0001 | 0.0909 | 0.8182 | 4 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | feature_scope | n_failure_branches | n_type_positive | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 1 | failure_rate_only | absolute | 71 | 15 | 0.1954 | 0.6155 | noise+action | 0.0000 | 0.0000 | 0.0751 |
| stagnation | 1 | noise+action | absolute | 71 | 15 | 0.2704 | 0.9741 | noise+action | -0.0751 | -0.3844 | 0.0000 |
| stagnation | 1 | moe_action | absolute | 71 | 15 | 0.2705 | 2.3139 | noise+action | -0.0751 | -0.3846 | -0.0000 |
| stagnation | 1 | moe_state | absolute | 71 | 15 | 0.2419 | 1.2030 | noise+action | -0.0465 | -0.2380 | 0.0286 |
| stagnation | 1 | moe_action+state | absolute | 71 | 15 | 0.2813 | 2.3734 | noise+action | -0.0859 | -0.4399 | -0.0108 |
| stagnation | 1 | noise+action+moe_action | absolute | 71 | 15 | 0.2651 | 2.1912 | noise+action | -0.0697 | -0.3569 | 0.0054 |
| stagnation | 1 | noise+action+moe_action+state | absolute | 71 | 15 | 0.2752 | 2.3170 | noise+action | -0.0798 | -0.4085 | -0.0047 |
| stagnation | 1 | noise+action_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2120 | 0.7054 | noise+action_within_snapshot | -0.0166 | -0.0852 | 0.0000 |
| stagnation | 1 | moe_action_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2260 | 0.7382 | noise+action_within_snapshot | -0.0306 | -0.1568 | -0.0140 |
| stagnation | 1 | moe_state_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2099 | 2.3566 | noise+action_within_snapshot | -0.0146 | -0.0745 | 0.0021 |
| stagnation | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2310 | 2.4007 | noise+action_within_snapshot | -0.0357 | -0.1826 | -0.0190 |
| stagnation | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2254 | 0.7316 | noise+action_within_snapshot | -0.0301 | -0.1540 | -0.0134 |
| stagnation | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2305 | 2.3984 | noise+action_within_snapshot | -0.0352 | -0.1799 | -0.0185 |
| stagnation | 3 | failure_rate_only | absolute | 71 | 15 | 0.1954 | 0.6155 | noise+action | 0.0000 | 0.0000 | 0.0251 |
| stagnation | 3 | noise+action | absolute | 71 | 15 | 0.2205 | 1.2206 | noise+action | -0.0251 | -0.1286 | 0.0000 |
| stagnation | 3 | moe_action | absolute | 71 | 15 | 0.2023 | 2.2320 | noise+action | -0.0070 | -0.0356 | 0.0182 |
| stagnation | 3 | moe_state | absolute | 71 | 15 | 0.2700 | 2.7730 | noise+action | -0.0747 | -0.3823 | -0.0496 |
| stagnation | 3 | moe_action+state | absolute | 71 | 15 | 0.2226 | 2.1898 | noise+action | -0.0273 | -0.1397 | -0.0022 |
| stagnation | 3 | noise+action+moe_action | absolute | 71 | 15 | 0.1997 | 2.2430 | noise+action | -0.0044 | -0.0223 | 0.0208 |
| stagnation | 3 | noise+action+moe_action+state | absolute | 71 | 15 | 0.2217 | 2.2693 | noise+action | -0.0264 | -0.1350 | -0.0013 |
| stagnation | 3 | noise+action_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2040 | 0.6817 | noise+action_within_snapshot | -0.0086 | -0.0441 | 0.0000 |
| stagnation | 3 | moe_action_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2268 | 0.7963 | noise+action_within_snapshot | -0.0314 | -0.1607 | -0.0228 |
| stagnation | 3 | moe_state_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2316 | 2.4101 | noise+action_within_snapshot | -0.0362 | -0.1854 | -0.0276 |
| stagnation | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2294 | 1.6529 | noise+action_within_snapshot | -0.0340 | -0.1741 | -0.0254 |
| stagnation | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2236 | 0.7818 | noise+action_within_snapshot | -0.0282 | -0.1444 | -0.0196 |
| stagnation | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 15 | 0.2287 | 1.4983 | noise+action_within_snapshot | -0.0333 | -0.1705 | -0.0247 |
| loop_or_cycling | 1 | failure_rate_only | absolute | 71 | 53 | 0.2217 | 0.6575 | noise+action | 0.0000 | 0.0000 | 0.0398 |
| loop_or_cycling | 1 | noise+action | absolute | 71 | 53 | 0.2615 | 0.8940 | noise+action | -0.0398 | -0.1793 | 0.0000 |
| loop_or_cycling | 1 | moe_action | absolute | 71 | 53 | 0.3077 | 1.0636 | noise+action | -0.0860 | -0.3881 | -0.0463 |
| loop_or_cycling | 1 | moe_state | absolute | 71 | 53 | 0.2063 | 0.7227 | noise+action | 0.0154 | 0.0694 | 0.0551 |
| loop_or_cycling | 1 | moe_action+state | absolute | 71 | 53 | 0.2422 | 0.7758 | noise+action | -0.0205 | -0.0925 | 0.0193 |
| loop_or_cycling | 1 | noise+action+moe_action | absolute | 71 | 53 | 0.2979 | 0.9918 | noise+action | -0.0762 | -0.3438 | -0.0365 |
| loop_or_cycling | 1 | noise+action+moe_action+state | absolute | 71 | 53 | 0.2429 | 0.7967 | noise+action | -0.0212 | -0.0958 | 0.0185 |
| loop_or_cycling | 1 | noise+action_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2360 | 0.7458 | noise+action_within_snapshot | -0.0143 | -0.0647 | 0.0000 |
| loop_or_cycling | 1 | moe_action_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2514 | 0.8997 | noise+action_within_snapshot | -0.0297 | -0.1338 | -0.0153 |
| loop_or_cycling | 1 | moe_state_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2166 | 2.2221 | noise+action_within_snapshot | 0.0051 | 0.0228 | 0.0194 |
| loop_or_cycling | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2327 | 2.2509 | noise+action_within_snapshot | -0.0110 | -0.0494 | 0.0034 |
| loop_or_cycling | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2489 | 0.8789 | noise+action_within_snapshot | -0.0272 | -0.1226 | -0.0128 |
| loop_or_cycling | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2314 | 2.2475 | noise+action_within_snapshot | -0.0097 | -0.0436 | 0.0047 |
| loop_or_cycling | 3 | failure_rate_only | absolute | 71 | 53 | 0.2217 | 0.6575 | noise+action | 0.0000 | 0.0000 | 0.0087 |
| loop_or_cycling | 3 | noise+action | absolute | 71 | 53 | 0.2304 | 0.8037 | noise+action | -0.0087 | -0.0391 | 0.0000 |
| loop_or_cycling | 3 | moe_action | absolute | 71 | 53 | 0.2620 | 1.0383 | noise+action | -0.0403 | -0.1819 | -0.0317 |
| loop_or_cycling | 3 | moe_state | absolute | 71 | 53 | 0.4048 | 3.2817 | noise+action | -0.1831 | -0.8260 | -0.1745 |
| loop_or_cycling | 3 | moe_action+state | absolute | 71 | 53 | 0.4151 | 2.5136 | noise+action | -0.1934 | -0.8722 | -0.1847 |
| loop_or_cycling | 3 | noise+action+moe_action | absolute | 71 | 53 | 0.2548 | 1.0249 | noise+action | -0.0331 | -0.1495 | -0.0245 |
| loop_or_cycling | 3 | noise+action+moe_action+state | absolute | 71 | 53 | 0.4125 | 2.7229 | noise+action | -0.1908 | -0.8607 | -0.1822 |
| loop_or_cycling | 3 | noise+action_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2218 | 0.7114 | noise+action_within_snapshot | -0.0001 | -0.0005 | 0.0000 |
| loop_or_cycling | 3 | moe_action_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2472 | 0.8489 | noise+action_within_snapshot | -0.0255 | -0.1152 | -0.0254 |
| loop_or_cycling | 3 | moe_state_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2893 | 2.2539 | noise+action_within_snapshot | -0.0676 | -0.3048 | -0.0675 |
| loop_or_cycling | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2842 | 1.6233 | noise+action_within_snapshot | -0.0625 | -0.2820 | -0.0624 |
| loop_or_cycling | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2466 | 0.8426 | noise+action_within_snapshot | -0.0249 | -0.1124 | -0.0248 |
| loop_or_cycling | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 53 | 0.2818 | 1.3722 | noise+action_within_snapshot | -0.0601 | -0.2713 | -0.0600 |
| drop_or_regrasp | 1 | failure_rate_only | absolute | 71 | 9 | 0.1171 | 0.4108 | noise+action | 0.0000 | 0.0000 | 0.0467 |
| drop_or_regrasp | 1 | noise+action | absolute | 71 | 9 | 0.1639 | 0.6061 | noise+action | -0.0467 | -0.3988 | 0.0000 |
| drop_or_regrasp | 1 | moe_action | absolute | 71 | 9 | 0.1927 | 0.6417 | noise+action | -0.0755 | -0.6449 | -0.0288 |
| drop_or_regrasp | 1 | moe_state | absolute | 71 | 9 | 0.2915 | 1.0329 | noise+action | -0.1743 | -1.4881 | -0.1276 |
| drop_or_regrasp | 1 | moe_action+state | absolute | 71 | 9 | 0.2031 | 1.2357 | noise+action | -0.0859 | -0.7335 | -0.0392 |
| drop_or_regrasp | 1 | noise+action+moe_action | absolute | 71 | 9 | 0.1777 | 0.6004 | noise+action | -0.0605 | -0.5168 | -0.0138 |
| drop_or_regrasp | 1 | noise+action+moe_action+state | absolute | 71 | 9 | 0.1991 | 1.1216 | noise+action | -0.0820 | -0.6996 | -0.0352 |
| drop_or_regrasp | 1 | noise+action_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1241 | 0.4802 | noise+action_within_snapshot | -0.0069 | -0.0591 | 0.0000 |
| drop_or_regrasp | 1 | moe_action_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1349 | 0.6713 | noise+action_within_snapshot | -0.0178 | -0.1515 | -0.0108 |
| drop_or_regrasp | 1 | moe_state_within_snapshot | within_snapshot_centered | 71 | 9 | 0.2669 | 2.9017 | noise+action_within_snapshot | -0.1497 | -1.2782 | -0.1428 |
| drop_or_regrasp | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 9 | 0.2786 | 2.9041 | noise+action_within_snapshot | -0.1615 | -1.3783 | -0.1545 |
| drop_or_regrasp | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1350 | 0.6661 | noise+action_within_snapshot | -0.0179 | -0.1526 | -0.0110 |
| drop_or_regrasp | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 9 | 0.2725 | 2.6914 | noise+action_within_snapshot | -0.1554 | -1.3263 | -0.1484 |
| drop_or_regrasp | 3 | failure_rate_only | absolute | 71 | 9 | 0.1171 | 0.4108 | noise+action | 0.0000 | 0.0000 | 0.0187 |
| drop_or_regrasp | 3 | noise+action | absolute | 71 | 9 | 0.1359 | 0.5349 | noise+action | -0.0187 | -0.1597 | 0.0000 |
| drop_or_regrasp | 3 | moe_action | absolute | 71 | 9 | 0.1463 | 0.7707 | noise+action | -0.0292 | -0.2492 | -0.0105 |
| drop_or_regrasp | 3 | moe_state | absolute | 71 | 9 | 0.2321 | 1.1089 | noise+action | -0.1150 | -0.9814 | -0.0962 |
| drop_or_regrasp | 3 | moe_action+state | absolute | 71 | 9 | 0.1656 | 0.9024 | noise+action | -0.0484 | -0.4134 | -0.0297 |
| drop_or_regrasp | 3 | noise+action+moe_action | absolute | 71 | 9 | 0.1445 | 0.8073 | noise+action | -0.0273 | -0.2334 | -0.0086 |
| drop_or_regrasp | 3 | noise+action+moe_action+state | absolute | 71 | 9 | 0.1668 | 0.8984 | noise+action | -0.0496 | -0.4236 | -0.0309 |
| drop_or_regrasp | 3 | noise+action_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1249 | 0.5141 | noise+action_within_snapshot | -0.0078 | -0.0663 | 0.0000 |
| drop_or_regrasp | 3 | moe_action_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1312 | 0.7077 | noise+action_within_snapshot | -0.0140 | -0.1196 | -0.0062 |
| drop_or_regrasp | 3 | moe_state_within_snapshot | within_snapshot_centered | 71 | 9 | 0.2974 | 3.3688 | noise+action_within_snapshot | -0.1802 | -1.5384 | -0.1724 |
| drop_or_regrasp | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1911 | 0.9122 | noise+action_within_snapshot | -0.0739 | -0.6310 | -0.0661 |
| drop_or_regrasp | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1309 | 0.7184 | noise+action_within_snapshot | -0.0138 | -0.1174 | -0.0060 |
| drop_or_regrasp | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 9 | 0.1872 | 0.8865 | noise+action_within_snapshot | -0.0701 | -0.5980 | -0.0623 |
| single_subtask_omission | 1 | failure_rate_only | absolute | 71 | 20 | 0.2777 | 0.8277 | noise+action | 0.0000 | 0.0000 | 0.0922 |
| single_subtask_omission | 1 | noise+action | absolute | 71 | 20 | 0.3700 | 1.2923 | noise+action | -0.0922 | -0.3321 | 0.0000 |
| single_subtask_omission | 1 | moe_action | absolute | 71 | 20 | 0.3724 | 2.1357 | noise+action | -0.0947 | -0.3410 | -0.0025 |
| single_subtask_omission | 1 | moe_state | absolute | 71 | 20 | 0.4079 | 1.7428 | noise+action | -0.1302 | -0.4686 | -0.0379 |
| single_subtask_omission | 1 | moe_action+state | absolute | 71 | 20 | 0.3594 | 2.7793 | noise+action | -0.0817 | -0.2940 | 0.0106 |
| single_subtask_omission | 1 | noise+action+moe_action | absolute | 71 | 20 | 0.3727 | 2.4373 | noise+action | -0.0950 | -0.3419 | -0.0027 |
| single_subtask_omission | 1 | noise+action+moe_action+state | absolute | 71 | 20 | 0.3571 | 3.1074 | noise+action | -0.0794 | -0.2857 | 0.0129 |
| single_subtask_omission | 1 | noise+action_within_snapshot | within_snapshot_centered | 71 | 20 | 0.2959 | 1.0290 | noise+action_within_snapshot | -0.0182 | -0.0656 | 0.0000 |
| single_subtask_omission | 1 | moe_action_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3128 | 1.0599 | noise+action_within_snapshot | -0.0350 | -0.1261 | -0.0168 |
| single_subtask_omission | 1 | moe_state_within_snapshot | within_snapshot_centered | 71 | 20 | 0.2984 | 3.3602 | noise+action_within_snapshot | -0.0206 | -0.0743 | -0.0024 |
| single_subtask_omission | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3040 | 2.7380 | noise+action_within_snapshot | -0.0263 | -0.0947 | -0.0081 |
| single_subtask_omission | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3111 | 1.0499 | noise+action_within_snapshot | -0.0333 | -0.1200 | -0.0151 |
| single_subtask_omission | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3203 | 2.8136 | noise+action_within_snapshot | -0.0426 | -0.1534 | -0.0244 |
| single_subtask_omission | 3 | failure_rate_only | absolute | 71 | 20 | 0.2777 | 0.8277 | noise+action | 0.0000 | 0.0000 | 0.0094 |
| single_subtask_omission | 3 | noise+action | absolute | 71 | 20 | 0.2871 | 1.2508 | noise+action | -0.0094 | -0.0337 | 0.0000 |
| single_subtask_omission | 3 | moe_action | absolute | 71 | 20 | 0.2962 | 2.2153 | noise+action | -0.0185 | -0.0665 | -0.0091 |
| single_subtask_omission | 3 | moe_state | absolute | 71 | 20 | 0.4530 | 3.4366 | noise+action | -0.1753 | -0.6310 | -0.1659 |
| single_subtask_omission | 3 | moe_action+state | absolute | 71 | 20 | 0.3331 | 3.3278 | noise+action | -0.0553 | -0.1992 | -0.0460 |
| single_subtask_omission | 3 | noise+action+moe_action | absolute | 71 | 20 | 0.3020 | 2.2302 | noise+action | -0.0243 | -0.0875 | -0.0150 |
| single_subtask_omission | 3 | noise+action+moe_action+state | absolute | 71 | 20 | 0.3556 | 3.4509 | noise+action | -0.0779 | -0.2803 | -0.0685 |
| single_subtask_omission | 3 | noise+action_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3162 | 1.2773 | noise+action_within_snapshot | -0.0385 | -0.1386 | 0.0000 |
| single_subtask_omission | 3 | moe_action_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3338 | 2.1213 | noise+action_within_snapshot | -0.0560 | -0.2018 | -0.0175 |
| single_subtask_omission | 3 | moe_state_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3425 | 3.1407 | noise+action_within_snapshot | -0.0648 | -0.2333 | -0.0263 |
| single_subtask_omission | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3477 | 3.2384 | noise+action_within_snapshot | -0.0700 | -0.2519 | -0.0315 |
| single_subtask_omission | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3351 | 2.1190 | noise+action_within_snapshot | -0.0573 | -0.2065 | -0.0189 |
| single_subtask_omission | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 20 | 0.3499 | 3.2101 | noise+action_within_snapshot | -0.0722 | -0.2598 | -0.0337 |
| non_stagnation_non_loop | 1 | failure_rate_only | absolute | 71 | 8 | 0.1093 | 0.4133 | noise+action | 0.0000 | 0.0000 | 0.1489 |
| non_stagnation_non_loop | 1 | noise+action | absolute | 71 | 8 | 0.2582 | 0.9403 | noise+action | -0.1489 | -1.3622 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action | absolute | 71 | 8 | 0.1428 | 0.8503 | noise+action | -0.0334 | -0.3059 | 0.1155 |
| non_stagnation_non_loop | 1 | moe_state | absolute | 71 | 8 | 0.2767 | 0.9735 | noise+action | -0.1674 | -1.5313 | -0.0185 |
| non_stagnation_non_loop | 1 | moe_action+state | absolute | 71 | 8 | 0.2212 | 0.9924 | noise+action | -0.1119 | -1.0232 | 0.0371 |
| non_stagnation_non_loop | 1 | noise+action+moe_action | absolute | 71 | 8 | 0.1543 | 0.8869 | noise+action | -0.0450 | -0.4116 | 0.1039 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state | absolute | 71 | 8 | 0.2363 | 1.0724 | noise+action | -0.1270 | -1.1618 | 0.0219 |
| non_stagnation_non_loop | 1 | noise+action_within_snapshot | within_snapshot_centered | 71 | 8 | 0.1159 | 0.5367 | noise+action_within_snapshot | -0.0066 | -0.0601 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action_within_snapshot | within_snapshot_centered | 71 | 8 | 0.1210 | 0.8873 | noise+action_within_snapshot | -0.0116 | -0.1064 | -0.0051 |
| non_stagnation_non_loop | 1 | moe_state_within_snapshot | within_snapshot_centered | 71 | 8 | 0.2289 | 2.4995 | noise+action_within_snapshot | -0.1195 | -1.0934 | -0.1130 |
| non_stagnation_non_loop | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 8 | 0.2260 | 2.6462 | noise+action_within_snapshot | -0.1167 | -1.0670 | -0.1101 |
| non_stagnation_non_loop | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 8 | 0.1170 | 0.8867 | noise+action_within_snapshot | -0.0077 | -0.0704 | -0.0011 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 8 | 0.2256 | 2.6655 | noise+action_within_snapshot | -0.1162 | -1.0632 | -0.1097 |
| non_stagnation_non_loop | 3 | failure_rate_only | absolute | 71 | 8 | 0.1093 | 0.4133 | noise+action | 0.0000 | 0.0000 | 0.1789 |
| non_stagnation_non_loop | 3 | noise+action | absolute | 71 | 8 | 0.2882 | 1.1132 | noise+action | -0.1789 | -1.6364 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action | absolute | 71 | 8 | 0.2637 | 1.5579 | noise+action | -0.1544 | -1.4121 | 0.0245 |
| non_stagnation_non_loop | 3 | moe_state | absolute | 71 | 8 | 0.4833 | 2.9421 | noise+action | -0.3740 | -3.4206 | -0.1951 |
| non_stagnation_non_loop | 3 | moe_action+state | absolute | 71 | 8 | 0.3472 | 3.2236 | noise+action | -0.2379 | -2.1758 | -0.0590 |
| non_stagnation_non_loop | 3 | noise+action+moe_action | absolute | 71 | 8 | 0.2821 | 1.6593 | noise+action | -0.1728 | -1.5803 | 0.0061 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state | absolute | 71 | 8 | 0.3304 | 3.4961 | noise+action | -0.2210 | -2.0219 | -0.0421 |
| non_stagnation_non_loop | 3 | noise+action_within_snapshot | within_snapshot_centered | 71 | 8 | 0.1082 | 0.4324 | noise+action_within_snapshot | 0.0011 | 0.0104 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action_within_snapshot | within_snapshot_centered | 71 | 8 | 0.1109 | 0.7330 | noise+action_within_snapshot | -0.0015 | -0.0140 | -0.0027 |
| non_stagnation_non_loop | 3 | moe_state_within_snapshot | within_snapshot_centered | 71 | 8 | 0.2244 | 2.6656 | noise+action_within_snapshot | -0.1151 | -1.0529 | -0.1162 |
| non_stagnation_non_loop | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 71 | 8 | 0.2318 | 2.1090 | noise+action_within_snapshot | -0.1225 | -1.1203 | -0.1236 |
| non_stagnation_non_loop | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 71 | 8 | 0.1097 | 0.6937 | noise+action_within_snapshot | -0.0003 | -0.0032 | -0.0015 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 71 | 8 | 0.2287 | 1.7990 | noise+action_within_snapshot | -0.1194 | -1.0918 | -0.1205 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 3 | action | back_12_15 | 0 | entropy | 0.5385 | 0.0769 | 0.4615 | 0.2862 | 0.6154 | 0.0909 | 0.2727 | 13 | 13 |
| single_subtask_omission | 1 | state | back_12_15 | 6 | top1_mass | 0.5385 | 0.1538 | 0.3846 | 0.2000 | 0.6641 | 0.0909 | 0.2727 | 13 | 13 |
| non_stagnation_non_loop | 1 | state | back_12_15 | 8 | top1_mass | 0.0000 | 0.3571 | -0.3571 | -0.6050 | -0.2500 | 0.0909 | 0.2727 | 14 | 14 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 0 | top1_mass | 0.3571 | 0.0000 | 0.3571 | 0.2500 | 0.6050 | 0.0909 | 0.2727 | 14 | 14 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 1 | top1_mass | 0.3571 | 0.0000 | 0.3571 | 0.2500 | 0.6050 | 0.0909 | 0.2727 | 14 | 14 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 2 | top1_mass | 0.3571 | 0.0000 | 0.3571 | 0.2500 | 0.6050 | 0.0909 | 0.2727 | 14 | 14 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.5000 | 0.9286 | -0.4286 | -0.5833 | -0.2797 | 0.0909 | 0.4545 | 14 | 14 |
| loop_or_cycling | 3 | action | front_2_5 | 0 | top1_mass | 0.4286 | 0.7857 | -0.3571 | -0.4333 | -0.2500 | 0.0909 | 0.5455 | 14 | 14 |
| loop_or_cycling | 1 | state | back_12_15 | 3 | top1_mass | 0.5000 | 0.8571 | -0.3571 | -0.5000 | -0.2500 | 0.0909 | 0.8182 | 14 | 14 |
| loop_or_cycling | 1 | state | front_2_5 | 4 | entropy | 0.5000 | 0.8571 | -0.3571 | -0.6429 | -0.2313 | 0.0909 | 0.8182 | 14 | 14 |
| loop_or_cycling | 1 | state | front_2_5 | 9 | consensus_distance | 0.5714 | 0.9286 | -0.3571 | -0.5833 | -0.1547 | 0.0909 | 0.8182 | 14 | 14 |
| drop_or_regrasp | 3 | action | back_12_15 | 7 | top1_mass | 0.3077 | 0.0000 | 0.3077 | 0.2579 | 0.4208 | 0.0909 | 0.9091 | 13 | 13 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
