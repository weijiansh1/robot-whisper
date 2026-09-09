# Rolling-star K=16 experiment

## Dataset

- 192 terminal branches from 12 committed snapshots; 62 success and 130 failure.
- 8 snapshots contain matched success/failure siblings.
- 9074 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.
- Candidate-ID negative control: max failure-rate deviation 0.177, within-snapshot permutation p=0.6287.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 77
- `single_subtask_omission`: 31
- `drop_or_regrasp`: 11
- `subtask_undo`: 5
- `goal_contact_near_miss`: 3
- `stagnation`: 1
- `timeout_other`: 1
- `active_retry`: 1

Stagnation or loop/cycling covers 112/130 failures; 18 failures require other physical mechanisms: {"active_retry": 1, "drop_or_regrasp": 1, "goal_contact_near_miss": 3, "single_subtask_omission": 11, "subtask_undo": 1, "timeout_other": 1}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00394.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (both snapshot-held-out and core worker-held-out checks):

| prefix_queries | cv_scheme | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | leave_one_snapshot_out | snapshot_rate_only | absolute | 192 | 8 | 0.2326 | 0.6610 | 0.4844 | 0.4844 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1611 | 0.0512 | 0.2721 |
| 1 | leave_one_snapshot_out | noise | absolute | 192 | 8 | 0.2417 | 0.6850 | 0.5000 | 0.4844 | 0.0156 | -0.1250 | 0.1797 | -0.0090 | -0.0388 | noise+action | -0.0090 | -0.0177 | -0.0013 | 0.1521 | 0.0594 | 0.2635 |
| 1 | leave_one_snapshot_out | action | absolute | 192 | 8 | 0.3461 | 1.1567 | 0.3750 | 0.4844 | -0.1094 | -0.2266 | -0.0078 | -0.1135 | -0.4879 | noise+action | -0.1135 | -0.2019 | -0.0269 | 0.0476 | -0.0364 | 0.1596 |
| 1 | leave_one_snapshot_out | noise+action | absolute | 192 | 8 | 0.3938 | 1.1784 | 0.5000 | 0.4844 | 0.0156 | -0.2031 | 0.2891 | -0.1611 | -0.6925 | noise+action | -0.1611 | -0.2714 | -0.0635 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action | absolute | 192 | 8 | 0.2921 | 0.9457 | 0.3750 | 0.4844 | -0.1094 | -0.2266 | -0.0078 | -0.0595 | -0.2556 | noise+action | -0.0595 | -0.2070 | 0.0595 | 0.1016 | -0.0248 | 0.2341 |
| 1 | leave_one_snapshot_out | moe_state | absolute | 192 | 8 | 0.2767 | 0.8883 | 0.6250 | 0.4844 | 0.1406 | -0.1562 | 0.4301 | -0.0441 | -0.1895 | noise+action | -0.0441 | -0.1812 | 0.0652 | 0.1170 | -0.0602 | 0.2774 |
| 1 | leave_one_snapshot_out | moe_action+state | absolute | 192 | 8 | 0.3099 | 1.1287 | 0.5000 | 0.4844 | 0.0156 | -0.1953 | 0.2656 | -0.0773 | -0.3322 | noise+action | -0.0773 | -0.2242 | 0.0832 | 0.0838 | -0.0580 | 0.2331 |
| 1 | leave_one_snapshot_out | noise+action+moe_action | absolute | 192 | 8 | 0.2925 | 0.9504 | 0.3750 | 0.4844 | -0.1094 | -0.2266 | 0.0000 | -0.0599 | -0.2573 | noise+action | -0.0599 | -0.1883 | 0.0744 | 0.1013 | -0.0166 | 0.2421 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 192 | 8 | 0.3098 | 1.1177 | 0.5000 | 0.4844 | 0.0156 | -0.1797 | 0.3010 | -0.0771 | -0.3316 | noise+action | -0.0771 | -0.2387 | 0.0636 | 0.0840 | -0.0795 | 0.2407 |
| 1 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2345 | 0.6656 | 0.6250 | 0.4844 | 0.1406 | -0.0469 | 0.4025 | -0.0019 | -0.0080 | noise+action_within_snapshot | -0.0019 | -0.0047 | 0.0003 | -0.0009 | -0.0037 | 0.0018 |
| 1 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2354 | 0.6679 | 0.3750 | 0.4844 | -0.1094 | -0.3947 | 0.1484 | -0.0028 | -0.0120 | noise+action_within_snapshot | -0.0028 | -0.0073 | 0.0013 | -0.0018 | -0.0050 | 0.0016 |
| 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2336 | 0.6641 | 0.5000 | 0.4844 | 0.0156 | -0.1250 | 0.1875 | -0.0010 | -0.0043 | noise+action_within_snapshot | -0.0010 | -0.0046 | 0.0012 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2387 | 0.6778 | 0.3750 | 0.4844 | -0.1094 | -0.2422 | 0.0000 | -0.0060 | -0.0259 | noise+action_within_snapshot | -0.0060 | -0.0144 | -0.0004 | -0.0050 | -0.0136 | 0.0016 |
| 1 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2969 | 1.6413 | 0.6250 | 0.4844 | 0.1406 | -0.1525 | 0.4572 | -0.0642 | -0.2761 | noise+action_within_snapshot | -0.0642 | -0.1788 | 0.0157 | -0.0632 | -0.1962 | 0.0190 |
| 1 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2391 | 0.6770 | 0.3750 | 0.4844 | -0.1094 | -0.2344 | 0.0000 | -0.0064 | -0.0277 | noise+action_within_snapshot | -0.0064 | -0.0142 | -0.0012 | -0.0054 | -0.0144 | -0.0003 |
| 1 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2383 | 0.6768 | 0.3750 | 0.4844 | -0.1094 | -0.2307 | 0.0000 | -0.0057 | -0.0245 | noise+action_within_snapshot | -0.0057 | -0.0128 | -0.0001 | -0.0047 | -0.0133 | 0.0019 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2387 | 0.6763 | 0.3750 | 0.4844 | -0.1094 | -0.2422 | 0.0000 | -0.0061 | -0.0261 | noise+action_within_snapshot | -0.0061 | -0.0153 | -0.0011 | -0.0051 | -0.0132 | 0.0005 |
| 3 | leave_one_snapshot_out | snapshot_rate_only | absolute | 192 | 8 | 0.2326 | 0.6610 | 0.4844 | 0.4844 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1297 | 0.0325 | 0.2270 |
| 3 | leave_one_snapshot_out | noise | absolute | 192 | 8 | 0.2344 | 0.6665 | 0.5000 | 0.4844 | 0.0156 | -0.1328 | 0.1760 | -0.0018 | -0.0076 | noise+action | -0.0018 | -0.0089 | 0.0048 | 0.1280 | 0.0244 | 0.2326 |
| 3 | leave_one_snapshot_out | action | absolute | 192 | 8 | 0.3092 | 1.0219 | 0.3750 | 0.4844 | -0.1094 | -0.2344 | -0.0078 | -0.0765 | -0.3290 | noise+action | -0.0765 | -0.2209 | 0.0609 | 0.0532 | -0.0385 | 0.1490 |
| 3 | leave_one_snapshot_out | noise+action | absolute | 192 | 8 | 0.3624 | 1.0989 | 0.6250 | 0.4844 | 0.1406 | -0.0703 | 0.3635 | -0.1297 | -0.5577 | noise+action | -0.1297 | -0.2409 | -0.0331 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action | absolute | 192 | 8 | 0.2096 | 0.6925 | 0.6250 | 0.4844 | 0.1406 | -0.2541 | 0.5000 | 0.0230 | 0.0989 | noise+action | 0.0230 | -0.0765 | 0.1151 | 0.1528 | 0.0524 | 0.2590 |
| 3 | leave_one_snapshot_out | moe_state | absolute | 192 | 8 | 0.2039 | 0.5716 | 0.6250 | 0.4844 | 0.1406 | -0.1172 | 0.4338 | 0.0287 | 0.1234 | noise+action | 0.0287 | -0.1009 | 0.1313 | 0.1585 | 0.0363 | 0.2740 |
| 3 | leave_one_snapshot_out | moe_action+state | absolute | 192 | 8 | 0.2270 | 0.7459 | 0.6250 | 0.4844 | 0.1406 | -0.1525 | 0.4297 | 0.0057 | 0.0244 | noise+action | 0.0057 | -0.1026 | 0.1085 | 0.1354 | 0.0576 | 0.2275 |
| 3 | leave_one_snapshot_out | noise+action+moe_action | absolute | 192 | 8 | 0.2125 | 0.6968 | 0.6250 | 0.4844 | 0.1406 | -0.2775 | 0.5119 | 0.0201 | 0.0865 | noise+action | 0.0201 | -0.0730 | 0.1100 | 0.1499 | 0.0589 | 0.2385 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 192 | 8 | 0.2359 | 0.7624 | 0.6250 | 0.4844 | 0.1406 | -0.1373 | 0.4416 | -0.0032 | -0.0138 | noise+action | -0.0032 | -0.1153 | 0.1052 | 0.1265 | 0.0296 | 0.2172 |
| 3 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2340 | 0.6647 | 0.5000 | 0.4844 | 0.0156 | -0.3672 | 0.4033 | -0.0013 | -0.0058 | noise+action_within_snapshot | -0.0013 | -0.0049 | 0.0015 | -0.0001 | -0.0034 | 0.0029 |
| 3 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2387 | 0.6762 | 0.5000 | 0.4844 | 0.0156 | -0.1328 | 0.1994 | -0.0061 | -0.0262 | noise+action_within_snapshot | -0.0061 | -0.0107 | -0.0020 | -0.0049 | -0.0092 | -0.0006 |
| 3 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2339 | 0.6646 | 0.6250 | 0.4844 | 0.1406 | -0.0469 | 0.3672 | -0.0012 | -0.0052 | noise+action_within_snapshot | -0.0012 | -0.0034 | 0.0007 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2346 | 0.6670 | 0.6250 | 0.4844 | 0.1406 | -0.1250 | 0.4494 | -0.0020 | -0.0084 | noise+action_within_snapshot | -0.0020 | -0.0095 | 0.0037 | -0.0007 | -0.0081 | 0.0051 |
| 3 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2385 | 0.6751 | 0.3750 | 0.4844 | -0.1094 | -0.2266 | 0.0000 | -0.0059 | -0.0252 | noise+action_within_snapshot | -0.0059 | -0.0089 | -0.0031 | -0.0047 | -0.0078 | -0.0016 |
| 3 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2351 | 0.6678 | 0.5000 | 0.4844 | 0.0156 | -0.1875 | 0.2969 | -0.0024 | -0.0103 | noise+action_within_snapshot | -0.0024 | -0.0079 | 0.0020 | -0.0012 | -0.0063 | 0.0028 |
| 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2344 | 0.6665 | 0.7500 | 0.4844 | 0.2656 | -0.0234 | 0.6135 | -0.0018 | -0.0076 | noise+action_within_snapshot | -0.0018 | -0.0078 | 0.0036 | -0.0006 | -0.0078 | 0.0045 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.2350 | 0.6676 | 0.5000 | 0.4844 | 0.0156 | -0.2072 | 0.3281 | -0.0023 | -0.0100 | noise+action_within_snapshot | -0.0023 | -0.0073 | 0.0019 | -0.0011 | -0.0056 | 0.0031 |
| 1 | leave_one_worker_out | snapshot_rate_only | absolute | 192 | 8 | 0.3257 | 0.9329 | 0.4844 | 0.4844 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2175 | 0.0519 | 0.5094 |
| 1 | leave_one_worker_out | noise+action | absolute | 192 | 8 | 0.5432 | 2.0366 | 0.6250 | 0.4844 | 0.1406 | -0.2917 | 0.9062 | -0.2175 | -0.6677 | noise+action | -0.2175 | -0.5196 | -0.0519 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action | absolute | 192 | 8 | 0.5126 | 2.0208 | 0.5000 | 0.4844 | 0.0156 | -0.0938 | 0.0556 | -0.1868 | -0.5736 | noise+action | -0.1868 | -0.2761 | -0.0719 | 0.0307 | -0.1109 | 0.2108 |
| 1 | leave_one_worker_out | moe_state | absolute | 192 | 8 | 0.4040 | 1.5861 | 0.5000 | 0.4844 | 0.0156 | -0.2917 | 0.4062 | -0.0783 | -0.2403 | noise+action | -0.0783 | -0.1482 | -0.0349 | 0.1392 | -0.0794 | 0.4604 |
| 1 | leave_one_worker_out | noise+action+moe_action+state | absolute | 192 | 8 | 0.5097 | 2.0509 | 0.5000 | 0.4844 | 0.0156 | -0.0938 | 0.0625 | -0.1840 | -0.5649 | noise+action | -0.1840 | -0.3408 | -0.0703 | 0.0335 | -0.1105 | 0.1788 |
| 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3285 | 0.9562 | 0.5000 | 0.4844 | 0.0156 | -0.0938 | 0.0625 | -0.0028 | -0.0085 | noise+action_within_snapshot | -0.0028 | -0.0079 | 0.0004 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3314 | 0.9751 | 0.3750 | 0.4844 | -0.1094 | -0.2917 | 0.0625 | -0.0057 | -0.0174 | noise+action_within_snapshot | -0.0057 | -0.0069 | -0.0048 | -0.0029 | -0.0059 | 0.0028 |
| 1 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.4089 | 2.2641 | 0.5000 | 0.4844 | 0.0156 | -0.0938 | 0.0592 | -0.0831 | -0.2553 | noise+action_within_snapshot | -0.0831 | -0.2346 | 0.0167 | -0.0804 | -0.2426 | 0.0136 |
| 1 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3335 | 0.9756 | 0.3750 | 0.4844 | -0.1094 | -0.2917 | 0.0234 | -0.0078 | -0.0239 | noise+action_within_snapshot | -0.0078 | -0.0157 | -0.0026 | -0.0050 | -0.0078 | -0.0024 |
| 3 | leave_one_worker_out | snapshot_rate_only | absolute | 192 | 8 | 0.3257 | 0.9329 | 0.4844 | 0.4844 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.1983 | 0.0284 | 0.4345 |
| 3 | leave_one_worker_out | noise+action | absolute | 192 | 8 | 0.5241 | 2.0347 | 0.6250 | 0.4844 | 0.1406 | -0.0938 | 0.3750 | -0.1983 | -0.6089 | noise+action | -0.1983 | -0.3969 | -0.0284 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action | absolute | 192 | 8 | 0.5213 | 3.7399 | 0.6250 | 0.4844 | 0.1406 | 0.0417 | 0.4062 | -0.1955 | -0.6003 | noise+action | -0.1955 | -0.2715 | -0.0893 | 0.0028 | -0.1913 | 0.1969 |
| 3 | leave_one_worker_out | moe_state | absolute | 192 | 8 | 0.6386 | 4.4359 | 0.5000 | 0.4844 | 0.0156 | -0.0938 | 0.0625 | -0.3129 | -0.9606 | noise+action | -0.3129 | -0.5772 | -0.1100 | -0.1146 | -0.2553 | 0.0630 |
| 3 | leave_one_worker_out | noise+action+moe_action+state | absolute | 192 | 8 | 0.6416 | 5.0100 | 0.6250 | 0.4844 | 0.1406 | -0.0938 | 0.3750 | -0.3159 | -0.9697 | noise+action | -0.3159 | -0.6110 | -0.1092 | -0.1175 | -0.2586 | 0.0891 |
| 3 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3285 | 0.9521 | 0.3750 | 0.4844 | -0.1094 | -0.2917 | 0.0625 | -0.0028 | -0.0086 | noise+action_within_snapshot | -0.0028 | -0.0090 | 0.0009 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3296 | 0.9669 | 0.6250 | 0.4844 | 0.1406 | -0.2917 | 0.9062 | -0.0038 | -0.0118 | noise+action_within_snapshot | -0.0038 | -0.0104 | 0.0034 | -0.0010 | -0.0044 | 0.0037 |
| 3 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3369 | 0.9870 | 0.3750 | 0.4844 | -0.1094 | -0.2917 | 0.0625 | -0.0112 | -0.0343 | noise+action_within_snapshot | -0.0112 | -0.0173 | -0.0050 | -0.0084 | -0.0132 | -0.0044 |
| 3 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 192 | 8 | 0.3315 | 0.9710 | 0.3750 | 0.4844 | -0.1094 | -0.2917 | 0.0625 | -0.0058 | -0.0178 | noise+action_within_snapshot | -0.0058 | -0.0135 | -0.0001 | -0.0030 | -0.0056 | -0.0004 |

Snapshot difficulty (one row per rolling state; snapshot and worker holdouts):

| cv_scheme | model | snapshots | heldout_units | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high | mse_gain_vs_policy_state | mse_gain_vs_policy_state_ci95_low | mse_gain_vs_policy_state_ci95_high | mse_gain_vs_geometry | mse_gain_vs_geometry_ci95_low | mse_gain_vs_geometry_ci95_high | mse_gain_vs_sim_state | mse_gain_vs_sim_state_ci95_low | mse_gain_vs_sim_state_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_snapshot_out | initial_geometry | 12 | 12 | 10.0000 | 0.0559 | 0.2068 | 0.1121 | 0.0436 | 0.1801 | 0.0479 | -0.0328 | 0.1150 | 0.0000 | 0.0000 | 0.0000 | 0.0071 | -0.0468 | 0.0469 |
| leave_one_snapshot_out | initial_sim_state | 12 | 12 | 10.0000 | 0.0630 | 0.1949 | 0.1050 | 0.0283 | 0.2052 | 0.0408 | -0.0148 | 0.1023 | -0.0071 | -0.0561 | 0.0421 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_snapshot_out | mean_moe_action | 12 | 12 | 10.0000 | 0.0987 | 0.2684 | 0.0693 | -0.0086 | 0.1596 | 0.0050 | -0.0861 | 0.0961 | -0.0429 | -0.0880 | -0.0005 | -0.0358 | -0.1035 | 0.0245 |
| leave_one_snapshot_out | initial_policy_state | 12 | 12 | 10.0000 | 0.1037 | 0.2439 | 0.0643 | -0.0224 | 0.1584 | 0.0000 | 0.0000 | 0.0000 | -0.0479 | -0.1246 | 0.0350 | -0.0408 | -0.1000 | 0.0157 |
| leave_one_snapshot_out | sim_state+mean_moe_action+state | 12 | 12 | 10.0000 | 0.1174 | 0.2902 | 0.0506 | -0.0310 | 0.1355 | -0.0137 | -0.1125 | 0.0760 | -0.0616 | -0.1178 | -0.0130 | -0.0545 | -0.1301 | 0.0090 |
| leave_one_snapshot_out | policy_state+mean_moe_action+state | 12 | 12 | 10.0000 | 0.1186 | 0.2913 | 0.0494 | -0.0208 | 0.1425 | -0.0149 | -0.1139 | 0.0802 | -0.0628 | -0.1202 | -0.0117 | -0.0557 | -0.1366 | 0.0051 |
| leave_one_snapshot_out | geometry+mean_moe_action+state | 12 | 12 | 10.0000 | 0.1187 | 0.2911 | 0.0493 | -0.0281 | 0.1292 | -0.0150 | -0.1106 | 0.0741 | -0.0629 | -0.1147 | -0.0139 | -0.0558 | -0.1291 | 0.0098 |
| leave_one_snapshot_out | mean_moe_action+state | 12 | 12 | 10.0000 | 0.1196 | 0.2921 | 0.0484 | -0.0283 | 0.1393 | -0.0158 | -0.1278 | 0.0767 | -0.0637 | -0.1196 | -0.0136 | -0.0566 | -0.1462 | 0.0089 |
| leave_one_snapshot_out | mean_moe_state | 12 | 12 | 10.0000 | 0.1528 | 0.3154 | 0.0152 | -0.0629 | 0.1096 | -0.0491 | -0.1528 | 0.0468 | -0.0970 | -0.1774 | -0.0351 | -0.0899 | -0.1740 | -0.0186 |
| leave_one_snapshot_out | mean_rate | 12 | 12 | NA | 0.1680 | 0.3561 | 0.0000 | 0.0000 | 0.0000 | -0.0643 | -0.1616 | 0.0197 | -0.1121 | -0.1805 | -0.0459 | -0.1050 | -0.1897 | -0.0253 |
| leave_one_snapshot_out | as_moe | 12 | 12 | 10.0000 | 0.1680 | 0.3561 | -0.0000 | -0.0000 | -0.0000 | -0.0643 | -0.1550 | 0.0204 | -0.1121 | -0.1882 | -0.0395 | -0.1050 | -0.1870 | -0.0270 |
| leave_one_snapshot_out | mean_output_action | 12 | 12 | 10.0000 | 0.3147 | 0.4644 | -0.1467 | -0.3210 | 0.0006 | -0.2110 | -0.3796 | -0.0531 | -0.2589 | -0.4393 | -0.1054 | -0.2518 | -0.4225 | -0.1147 |
| leave_one_worker_out | initial_policy_state | 12 | 4 | 10.0000 | 0.2158 | 0.3967 | 0.0325 | -0.1134 | 0.1634 | 0.0000 | 0.0000 | 0.0000 | 0.0120 | -0.1169 | 0.1409 | 0.0543 | -0.0549 | 0.1900 |
| leave_one_worker_out | initial_geometry | 12 | 4 | 10.0000 | 0.2278 | 0.3708 | 0.0205 | -0.0579 | 0.1421 | -0.0120 | -0.1735 | 0.1169 | 0.0000 | 0.0000 | 0.0000 | 0.0423 | -0.1064 | 0.2795 |
| leave_one_worker_out | as_moe | 12 | 4 | 10.0000 | 0.2483 | 0.4271 | 0.0000 | -0.0000 | 0.0000 | -0.0325 | -0.1605 | 0.1134 | -0.0205 | -0.1294 | 0.0579 | 0.0218 | -0.1914 | 0.2691 |
| leave_one_worker_out | mean_rate | 12 | 4 | NA | 0.2483 | 0.4271 | 0.0000 | 0.0000 | 0.0000 | -0.0325 | -0.1634 | 0.1134 | -0.0205 | -0.1197 | 0.0579 | 0.0218 | -0.1914 | 0.2999 |
| leave_one_worker_out | initial_sim_state | 12 | 4 | 10.0000 | 0.2701 | 0.4124 | -0.0218 | -0.2691 | 0.1914 | -0.0543 | -0.1702 | 0.0549 | -0.0423 | -0.2402 | 0.1458 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_worker_out | mean_moe_action | 12 | 4 | 10.0000 | 0.3211 | 0.4850 | -0.0728 | -0.1775 | -0.0034 | -0.1053 | -0.2250 | -0.0124 | -0.0933 | -0.1848 | 0.0169 | -0.0510 | -0.2508 | 0.1304 |
| leave_one_worker_out | mean_moe_state | 12 | 4 | 10.0000 | 0.3274 | 0.4783 | -0.0791 | -0.2288 | 0.0275 | -0.1116 | -0.2171 | -0.0267 | -0.0996 | -0.2137 | 0.0145 | -0.0573 | -0.1923 | 0.0776 |
| leave_one_worker_out | geometry+mean_moe_action+state | 12 | 4 | 10.0000 | 0.3384 | 0.4955 | -0.0901 | -0.2551 | 0.0122 | -0.1226 | -0.2318 | -0.0248 | -0.1106 | -0.2292 | 0.0172 | -0.0683 | -0.2123 | 0.0757 |
| leave_one_worker_out | sim_state+mean_moe_action+state | 12 | 4 | 10.0000 | 0.3389 | 0.4960 | -0.0907 | -0.2406 | 0.0126 | -0.1231 | -0.2318 | -0.0348 | -0.1111 | -0.2313 | 0.0372 | -0.0689 | -0.2120 | 0.0743 |
| leave_one_worker_out | policy_state+mean_moe_action+state | 12 | 4 | 10.0000 | 0.3395 | 0.4966 | -0.0913 | -0.2394 | 0.0165 | -0.1237 | -0.2240 | -0.0338 | -0.1117 | -0.2216 | 0.0162 | -0.0694 | -0.2139 | 0.0750 |
| leave_one_worker_out | mean_moe_action+state | 12 | 4 | 10.0000 | 0.3399 | 0.4969 | -0.0917 | -0.2398 | 0.0103 | -0.1241 | -0.2332 | -0.0371 | -0.1121 | -0.2306 | -0.0020 | -0.0698 | -0.2144 | 0.0748 |
| leave_one_worker_out | mean_output_action | 12 | 4 | 10.0000 | 0.3853 | 0.5296 | -0.1371 | -0.4591 | 0.0581 | -0.1695 | -0.3213 | -0.0428 | -0.1575 | -0.4072 | 0.0549 | -0.1152 | -0.1790 | -0.0221 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | front_2_5 | 0 | consensus_distance | 0.5938 | 0.2812 | 0.3125 | 0.1875 | 0.4688 | 0.0020 | 0.0439 | 32 | 32 |
| 1 | state | front_2_5 | 1 | top1_mass | 0.6250 | 0.3125 | 0.3125 | 0.1875 | 0.4688 | 0.0020 | 0.0439 | 32 | 32 |
| 1 | state | front_2_5 | 0 | top1_mass | 0.6562 | 0.3750 | 0.2812 | 0.1562 | 0.4375 | 0.0040 | 0.1776 | 32 | 32 |
| 1 | state | front_2_5 | 8 | entropy | 0.6250 | 0.3438 | 0.2812 | 0.0938 | 0.4688 | 0.0040 | 0.1776 | 32 | 32 |
| 1 | state | front_2_5 | 1 | consensus_distance | 0.5938 | 0.3438 | 0.2500 | 0.0938 | 0.4375 | 0.0060 | 0.4012 | 32 | 32 |
| 1 | state | front_2_5 | 7 | entropy | 0.6250 | 0.3750 | 0.2500 | 0.0625 | 0.3750 | 0.0100 | 0.4012 | 32 | 32 |
| 1 | state | front_2_5 | 3 | top1_mass | 0.6250 | 0.3750 | 0.2500 | 0.0312 | 0.4688 | 0.0120 | 0.4012 | 32 | 32 |
| 1 | state | front_2_5 | 4 | consensus_distance | 0.6250 | 0.4062 | 0.2188 | 0.0000 | 0.4062 | 0.0140 | 0.7086 | 32 | 32 |
| 1 | state | front_2_5 | 2 | top1_mass | 0.6250 | 0.4062 | 0.2188 | 0.0312 | 0.4062 | 0.0180 | 0.7086 | 32 | 32 |
| 1 | state | front_2_5 | 5 | top1_mass | 0.5938 | 0.3750 | 0.2188 | 0.0938 | 0.3750 | 0.0200 | 0.7086 | 32 | 32 |
| 1 | state | front_2_5 | 7 | consensus_distance | 0.5938 | 0.3750 | 0.2188 | 0.0312 | 0.3750 | 0.0279 | 0.7086 | 32 | 32 |
| 1 | state | front_2_5 | 4 | entropy | 0.5938 | 0.3750 | 0.2188 | 0.0625 | 0.3750 | 0.0319 | 0.7086 | 32 | 32 |

Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):

| prefix_queries | token_family | layer_group | denoise_step | expert | failure_minus_success_probability | ci95_low | ci95_high | ci95_worker_low | ci95_worker_high | snapshot_effect_min | snapshot_effect_max | snapshot_effect_sd | snapshots_effect_positive | snapshots_effect_negative | permutation_p_raw | permutation_p_maxT | n_mixed_snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | action | front_2_5 | 0 | 0 | 0.0004 | 0.0001 | 0.0008 | 0.0001 | 0.0009 | -0.0004 | 0.0013 | 0.0006 | 6 | 2 | 0.0020 | 0.0080 | 8 |
| 1 | action | front_2_5 | 1 | 0 | 0.0004 | 0.0001 | 0.0008 | 0.0001 | 0.0008 | -0.0004 | 0.0013 | 0.0006 | 6 | 2 | 0.0020 | 0.0100 | 8 |
| 1 | action | front_2_5 | 2 | 0 | 0.0003 | -0.0000 | 0.0007 | 0.0001 | 0.0008 | -0.0003 | 0.0010 | 0.0005 | 5 | 3 | 0.0120 | 0.0878 | 8 |
| 1 | action | front_2_5 | 3 | 0 | 0.0003 | -0.0000 | 0.0006 | 0.0001 | 0.0007 | -0.0003 | 0.0008 | 0.0005 | 6 | 2 | 0.0240 | 0.2994 | 8 |
| 1 | action | front_2_5 | 0 | 12 | -0.0002 | -0.0005 | 0.0000 | -0.0004 | -0.0002 | -0.0008 | 0.0003 | 0.0004 | 3 | 5 | 0.0579 | 0.3892 | 8 |
| 1 | action | front_2_5 | 1 | 12 | -0.0002 | -0.0005 | 0.0000 | -0.0004 | -0.0002 | -0.0008 | 0.0004 | 0.0004 | 3 | 5 | 0.0519 | 0.4112 | 8 |
| 1 | action | front_2_5 | 9 | 31 | -0.0002 | -0.0004 | -0.0001 | -0.0003 | -0.0002 | -0.0006 | 0.0000 | 0.0002 | 1 | 7 | 0.0040 | 0.4172 | 8 |
| 1 | action | front_2_5 | 4 | 0 | 0.0002 | -0.0000 | 0.0005 | 0.0001 | 0.0006 | -0.0004 | 0.0007 | 0.0004 | 6 | 2 | 0.0479 | 0.4970 | 8 |
| 1 | action | front_2_5 | 9 | 19 | -0.0002 | -0.0003 | -0.0001 | -0.0004 | -0.0001 | -0.0006 | -0.0000 | 0.0002 | 0 | 8 | 0.0279 | 0.7246 | 8 |
| 1 | action | front_2_5 | 2 | 12 | -0.0002 | -0.0004 | 0.0000 | -0.0004 | -0.0001 | -0.0006 | 0.0003 | 0.0004 | 3 | 5 | 0.0439 | 0.7505 | 8 |
| 1 | action | front_2_5 | 2 | 11 | -0.0002 | -0.0003 | -0.0001 | -0.0003 | -0.0001 | -0.0004 | 0.0000 | 0.0001 | 1 | 7 | 0.0100 | 0.8044 | 8 |
| 1 | action | front_2_5 | 7 | 19 | -0.0002 | -0.0004 | -0.0001 | -0.0005 | -0.0001 | -0.0007 | 0.0001 | 0.0002 | 1 | 7 | 0.0419 | 0.8064 | 8 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | feature_scope | n_failure_branches | n_type_positive | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 1 | failure_rate_only | absolute | 130 | 27 | 0.1793 | 0.5571 | noise+action | 0.0000 | 0.0000 | 0.0184 |
| stagnation | 1 | noise+action | absolute | 130 | 27 | 0.1978 | 0.7331 | noise+action | -0.0184 | -0.1027 | 0.0000 |
| stagnation | 1 | moe_action | absolute | 130 | 27 | 0.2258 | 1.0532 | noise+action | -0.0465 | -0.2591 | -0.0281 |
| stagnation | 1 | moe_state | absolute | 130 | 27 | 0.1960 | 0.8537 | noise+action | -0.0167 | -0.0930 | 0.0017 |
| stagnation | 1 | moe_action+state | absolute | 130 | 27 | 0.2177 | 1.0490 | noise+action | -0.0384 | -0.2138 | -0.0199 |
| stagnation | 1 | noise+action+moe_action | absolute | 130 | 27 | 0.2274 | 1.1413 | noise+action | -0.0481 | -0.2682 | -0.0297 |
| stagnation | 1 | noise+action+moe_action+state | absolute | 130 | 27 | 0.2175 | 1.1008 | noise+action | -0.0381 | -0.2125 | -0.0197 |
| stagnation | 1 | noise+action_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1915 | 0.6008 | noise+action_within_snapshot | -0.0122 | -0.0680 | 0.0000 |
| stagnation | 1 | moe_action_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1966 | 0.5999 | noise+action_within_snapshot | -0.0172 | -0.0961 | -0.0050 |
| stagnation | 1 | moe_state_within_snapshot | within_snapshot_centered | 130 | 27 | 0.2185 | 1.5762 | noise+action_within_snapshot | -0.0392 | -0.2183 | -0.0270 |
| stagnation | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1824 | 0.6721 | noise+action_within_snapshot | -0.0030 | -0.0170 | 0.0092 |
| stagnation | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1956 | 0.5974 | noise+action_within_snapshot | -0.0162 | -0.0905 | -0.0040 |
| stagnation | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1806 | 0.6370 | noise+action_within_snapshot | -0.0012 | -0.0069 | 0.0110 |
| stagnation | 3 | failure_rate_only | absolute | 130 | 27 | 0.1793 | 0.5571 | noise+action | 0.0000 | 0.0000 | 0.0198 |
| stagnation | 3 | noise+action | absolute | 130 | 27 | 0.1991 | 0.7644 | noise+action | -0.0198 | -0.1104 | 0.0000 |
| stagnation | 3 | moe_action | absolute | 130 | 27 | 0.1957 | 0.9571 | noise+action | -0.0164 | -0.0914 | 0.0034 |
| stagnation | 3 | moe_state | absolute | 130 | 27 | 0.2667 | 1.6363 | noise+action | -0.0874 | -0.4873 | -0.0676 |
| stagnation | 3 | moe_action+state | absolute | 130 | 27 | 0.2176 | 1.2247 | noise+action | -0.0383 | -0.2135 | -0.0185 |
| stagnation | 3 | noise+action+moe_action | absolute | 130 | 27 | 0.1974 | 0.9769 | noise+action | -0.0181 | -0.1008 | 0.0017 |
| stagnation | 3 | noise+action+moe_action+state | absolute | 130 | 27 | 0.2141 | 1.2527 | noise+action | -0.0348 | -0.1938 | -0.0150 |
| stagnation | 3 | noise+action_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1912 | 0.6059 | noise+action_within_snapshot | -0.0118 | -0.0659 | 0.0000 |
| stagnation | 3 | moe_action_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1827 | 0.5724 | noise+action_within_snapshot | -0.0034 | -0.0188 | 0.0084 |
| stagnation | 3 | moe_state_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1919 | 0.6575 | noise+action_within_snapshot | -0.0126 | -0.0700 | -0.0007 |
| stagnation | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1854 | 0.5943 | noise+action_within_snapshot | -0.0060 | -0.0335 | 0.0058 |
| stagnation | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1838 | 0.5754 | noise+action_within_snapshot | -0.0045 | -0.0250 | 0.0073 |
| stagnation | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 27 | 0.1864 | 0.5969 | noise+action_within_snapshot | -0.0070 | -0.0392 | 0.0048 |
| loop_or_cycling | 1 | failure_rate_only | absolute | 130 | 91 | 0.2305 | 0.6604 | noise+action | 0.0000 | 0.0000 | -0.0181 |
| loop_or_cycling | 1 | noise+action | absolute | 130 | 91 | 0.2124 | 0.6270 | noise+action | 0.0181 | 0.0784 | 0.0000 |
| loop_or_cycling | 1 | moe_action | absolute | 130 | 91 | 0.2888 | 1.2008 | noise+action | -0.0583 | -0.2529 | -0.0764 |
| loop_or_cycling | 1 | moe_state | absolute | 130 | 91 | 0.2304 | 1.1139 | noise+action | 0.0001 | 0.0003 | -0.0180 |
| loop_or_cycling | 1 | moe_action+state | absolute | 130 | 91 | 0.2816 | 1.6395 | noise+action | -0.0511 | -0.2219 | -0.0692 |
| loop_or_cycling | 1 | noise+action+moe_action | absolute | 130 | 91 | 0.2837 | 1.1718 | noise+action | -0.0532 | -0.2310 | -0.0713 |
| loop_or_cycling | 1 | noise+action+moe_action+state | absolute | 130 | 91 | 0.2812 | 1.6625 | noise+action | -0.0507 | -0.2200 | -0.0688 |
| loop_or_cycling | 1 | noise+action_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2388 | 0.6890 | noise+action_within_snapshot | -0.0083 | -0.0360 | 0.0000 |
| loop_or_cycling | 1 | moe_action_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2390 | 0.6932 | noise+action_within_snapshot | -0.0086 | -0.0372 | -0.0003 |
| loop_or_cycling | 1 | moe_state_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2526 | 1.4218 | noise+action_within_snapshot | -0.0222 | -0.0962 | -0.0139 |
| loop_or_cycling | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2356 | 0.7044 | noise+action_within_snapshot | -0.0052 | -0.0224 | 0.0031 |
| loop_or_cycling | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2392 | 0.6928 | noise+action_within_snapshot | -0.0088 | -0.0381 | -0.0005 |
| loop_or_cycling | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2346 | 0.7029 | noise+action_within_snapshot | -0.0041 | -0.0178 | 0.0042 |
| loop_or_cycling | 3 | failure_rate_only | absolute | 130 | 91 | 0.2305 | 0.6604 | noise+action | 0.0000 | 0.0000 | 0.0164 |
| loop_or_cycling | 3 | noise+action | absolute | 130 | 91 | 0.2469 | 0.7402 | noise+action | -0.0164 | -0.0713 | 0.0000 |
| loop_or_cycling | 3 | moe_action | absolute | 130 | 91 | 0.2485 | 0.8576 | noise+action | -0.0180 | -0.0781 | -0.0016 |
| loop_or_cycling | 3 | moe_state | absolute | 130 | 91 | 0.3349 | 2.8460 | noise+action | -0.1044 | -0.4530 | -0.0880 |
| loop_or_cycling | 3 | moe_action+state | absolute | 130 | 91 | 0.3137 | 1.4907 | noise+action | -0.0833 | -0.3613 | -0.0668 |
| loop_or_cycling | 3 | noise+action+moe_action | absolute | 130 | 91 | 0.2428 | 0.8234 | noise+action | -0.0124 | -0.0537 | 0.0041 |
| loop_or_cycling | 3 | noise+action+moe_action+state | absolute | 130 | 91 | 0.3024 | 1.3656 | noise+action | -0.0720 | -0.3123 | -0.0555 |
| loop_or_cycling | 3 | noise+action_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2539 | 0.7382 | noise+action_within_snapshot | -0.0234 | -0.1017 | 0.0000 |
| loop_or_cycling | 3 | moe_action_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2560 | 0.7468 | noise+action_within_snapshot | -0.0255 | -0.1108 | -0.0021 |
| loop_or_cycling | 3 | moe_state_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2499 | 0.8476 | noise+action_within_snapshot | -0.0194 | -0.0842 | 0.0040 |
| loop_or_cycling | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2558 | 0.7706 | noise+action_within_snapshot | -0.0253 | -0.1098 | -0.0019 |
| loop_or_cycling | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2564 | 0.7486 | noise+action_within_snapshot | -0.0259 | -0.1124 | -0.0025 |
| loop_or_cycling | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 91 | 0.2553 | 0.7678 | noise+action_within_snapshot | -0.0248 | -0.1076 | -0.0014 |
| drop_or_regrasp | 1 | failure_rate_only | absolute | 130 | 16 | 0.1092 | 0.3786 | noise+action | 0.0000 | 0.0000 | 0.0091 |
| drop_or_regrasp | 1 | noise+action | absolute | 130 | 16 | 0.1183 | 0.4385 | noise+action | -0.0091 | -0.0833 | 0.0000 |
| drop_or_regrasp | 1 | moe_action | absolute | 130 | 16 | 0.1570 | 0.5790 | noise+action | -0.0478 | -0.4378 | -0.0387 |
| drop_or_regrasp | 1 | moe_state | absolute | 130 | 16 | 0.2389 | 0.8473 | noise+action | -0.1297 | -1.1883 | -0.1206 |
| drop_or_regrasp | 1 | moe_action+state | absolute | 130 | 16 | 0.3369 | 2.1994 | noise+action | -0.2277 | -2.0859 | -0.2186 |
| drop_or_regrasp | 1 | noise+action+moe_action | absolute | 130 | 16 | 0.1569 | 0.5664 | noise+action | -0.0478 | -0.4377 | -0.0387 |
| drop_or_regrasp | 1 | noise+action+moe_action+state | absolute | 130 | 16 | 0.3395 | 2.1915 | noise+action | -0.2304 | -2.1103 | -0.2213 |
| drop_or_regrasp | 1 | noise+action_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1092 | 0.3863 | noise+action_within_snapshot | -0.0000 | -0.0002 | 0.0000 |
| drop_or_regrasp | 1 | moe_action_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1226 | 0.4300 | noise+action_within_snapshot | -0.0135 | -0.1232 | -0.0134 |
| drop_or_regrasp | 1 | moe_state_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1904 | 1.6334 | noise+action_within_snapshot | -0.0812 | -0.7442 | -0.0812 |
| drop_or_regrasp | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1338 | 0.5126 | noise+action_within_snapshot | -0.0247 | -0.2258 | -0.0246 |
| drop_or_regrasp | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1213 | 0.4253 | noise+action_within_snapshot | -0.0121 | -0.1108 | -0.0121 |
| drop_or_regrasp | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1329 | 0.5136 | noise+action_within_snapshot | -0.0238 | -0.2176 | -0.0237 |
| drop_or_regrasp | 3 | failure_rate_only | absolute | 130 | 16 | 0.1092 | 0.3786 | noise+action | 0.0000 | 0.0000 | -0.0017 |
| drop_or_regrasp | 3 | noise+action | absolute | 130 | 16 | 0.1075 | 0.3961 | noise+action | 0.0017 | 0.0155 | 0.0000 |
| drop_or_regrasp | 3 | moe_action | absolute | 130 | 16 | 0.1379 | 0.5600 | noise+action | -0.0287 | -0.2630 | -0.0304 |
| drop_or_regrasp | 3 | moe_state | absolute | 130 | 16 | 0.2694 | 1.6133 | noise+action | -0.1602 | -1.4673 | -0.1619 |
| drop_or_regrasp | 3 | moe_action+state | absolute | 130 | 16 | 0.2521 | 0.9981 | noise+action | -0.1429 | -1.3092 | -0.1446 |
| drop_or_regrasp | 3 | noise+action+moe_action | absolute | 130 | 16 | 0.1322 | 0.5229 | noise+action | -0.0230 | -0.2105 | -0.0247 |
| drop_or_regrasp | 3 | noise+action+moe_action+state | absolute | 130 | 16 | 0.2321 | 0.8905 | noise+action | -0.1229 | -1.1260 | -0.1246 |
| drop_or_regrasp | 3 | noise+action_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1145 | 0.4128 | noise+action_within_snapshot | -0.0053 | -0.0486 | 0.0000 |
| drop_or_regrasp | 3 | moe_action_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1139 | 0.4436 | noise+action_within_snapshot | -0.0047 | -0.0431 | 0.0006 |
| drop_or_regrasp | 3 | moe_state_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1953 | 1.4052 | noise+action_within_snapshot | -0.0861 | -0.7885 | -0.0808 |
| drop_or_regrasp | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1123 | 0.4647 | noise+action_within_snapshot | -0.0032 | -0.0291 | 0.0021 |
| drop_or_regrasp | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1144 | 0.4453 | noise+action_within_snapshot | -0.0052 | -0.0480 | 0.0001 |
| drop_or_regrasp | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 16 | 0.1123 | 0.4632 | noise+action_within_snapshot | -0.0031 | -0.0284 | 0.0022 |
| goal_regression | 1 | failure_rate_only | absolute | 130 | 6 | 0.0456 | 0.2117 | noise+action | 0.0000 | 0.0000 | 0.0008 |
| goal_regression | 1 | noise+action | absolute | 130 | 6 | 0.0464 | 0.3097 | noise+action | -0.0008 | -0.0174 | 0.0000 |
| goal_regression | 1 | moe_action | absolute | 130 | 6 | 0.0521 | 0.5174 | noise+action | -0.0065 | -0.1415 | -0.0057 |
| goal_regression | 1 | moe_state | absolute | 130 | 6 | 0.0613 | 0.4087 | noise+action | -0.0157 | -0.3444 | -0.0149 |
| goal_regression | 1 | moe_action+state | absolute | 130 | 6 | 0.0569 | 0.5311 | noise+action | -0.0113 | -0.2469 | -0.0105 |
| goal_regression | 1 | noise+action+moe_action | absolute | 130 | 6 | 0.0503 | 0.5359 | noise+action | -0.0047 | -0.1029 | -0.0039 |
| goal_regression | 1 | noise+action+moe_action+state | absolute | 130 | 6 | 0.0579 | 0.5296 | noise+action | -0.0123 | -0.2697 | -0.0115 |
| goal_regression | 1 | noise+action_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0460 | 0.2421 | noise+action_within_snapshot | -0.0003 | -0.0074 | 0.0000 |
| goal_regression | 1 | moe_action_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0508 | 0.3072 | noise+action_within_snapshot | -0.0052 | -0.1139 | -0.0049 |
| goal_regression | 1 | moe_state_within_snapshot | within_snapshot_centered | 130 | 6 | 0.1474 | 0.8236 | noise+action_within_snapshot | -0.1018 | -2.2308 | -0.1014 |
| goal_regression | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0575 | 0.4489 | noise+action_within_snapshot | -0.0119 | -0.2605 | -0.0115 |
| goal_regression | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0498 | 0.3092 | noise+action_within_snapshot | -0.0042 | -0.0910 | -0.0038 |
| goal_regression | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0553 | 0.4456 | noise+action_within_snapshot | -0.0096 | -0.2111 | -0.0093 |
| goal_regression | 3 | failure_rate_only | absolute | 130 | 6 | 0.0456 | 0.2117 | noise+action | 0.0000 | 0.0000 | 0.0013 |
| goal_regression | 3 | noise+action | absolute | 130 | 6 | 0.0469 | 0.3038 | noise+action | -0.0013 | -0.0284 | 0.0000 |
| goal_regression | 3 | moe_action | absolute | 130 | 6 | 0.0697 | 0.6292 | noise+action | -0.0241 | -0.5274 | -0.0228 |
| goal_regression | 3 | moe_state | absolute | 130 | 6 | 0.0575 | 0.5754 | noise+action | -0.0119 | -0.2598 | -0.0106 |
| goal_regression | 3 | moe_action+state | absolute | 130 | 6 | 0.0499 | 0.5944 | noise+action | -0.0043 | -0.0936 | -0.0030 |
| goal_regression | 3 | noise+action+moe_action | absolute | 130 | 6 | 0.0770 | 0.6512 | noise+action | -0.0314 | -0.6881 | -0.0301 |
| goal_regression | 3 | noise+action+moe_action+state | absolute | 130 | 6 | 0.0482 | 0.6016 | noise+action | -0.0026 | -0.0568 | -0.0013 |
| goal_regression | 3 | noise+action_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0464 | 0.2440 | noise+action_within_snapshot | -0.0007 | -0.0163 | 0.0000 |
| goal_regression | 3 | moe_action_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0451 | 0.4010 | noise+action_within_snapshot | 0.0005 | 0.0113 | 0.0013 |
| goal_regression | 3 | moe_state_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0575 | 0.4503 | noise+action_within_snapshot | -0.0119 | -0.2602 | -0.0111 |
| goal_regression | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0489 | 0.4607 | noise+action_within_snapshot | -0.0033 | -0.0716 | -0.0025 |
| goal_regression | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0452 | 0.4070 | noise+action_within_snapshot | 0.0004 | 0.0095 | 0.0012 |
| goal_regression | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 6 | 0.0517 | 0.4667 | noise+action_within_snapshot | -0.0061 | -0.1339 | -0.0054 |
| single_subtask_omission | 1 | failure_rate_only | absolute | 130 | 46 | 0.2820 | 0.7718 | noise+action | 0.0000 | 0.0000 | 0.0254 |
| single_subtask_omission | 1 | noise+action | absolute | 130 | 46 | 0.3074 | 1.0823 | noise+action | -0.0254 | -0.0900 | 0.0000 |
| single_subtask_omission | 1 | moe_action | absolute | 130 | 46 | 0.2372 | 0.9628 | noise+action | 0.0448 | 0.1589 | 0.0702 |
| single_subtask_omission | 1 | moe_state | absolute | 130 | 46 | 0.3080 | 1.2347 | noise+action | -0.0260 | -0.0921 | -0.0006 |
| single_subtask_omission | 1 | moe_action+state | absolute | 130 | 46 | 0.2775 | 1.4648 | noise+action | 0.0045 | 0.0159 | 0.0299 |
| single_subtask_omission | 1 | noise+action+moe_action | absolute | 130 | 46 | 0.2411 | 1.0660 | noise+action | 0.0409 | 0.1450 | 0.0663 |
| single_subtask_omission | 1 | noise+action+moe_action+state | absolute | 130 | 46 | 0.2824 | 1.7960 | noise+action | -0.0004 | -0.0012 | 0.0250 |
| single_subtask_omission | 1 | noise+action_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2850 | 0.7843 | noise+action_within_snapshot | -0.0030 | -0.0105 | 0.0000 |
| single_subtask_omission | 1 | moe_action_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2990 | 0.8302 | noise+action_within_snapshot | -0.0169 | -0.0601 | -0.0140 |
| single_subtask_omission | 1 | moe_state_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2813 | 1.0954 | noise+action_within_snapshot | 0.0007 | 0.0026 | 0.0037 |
| single_subtask_omission | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2719 | 0.8536 | noise+action_within_snapshot | 0.0102 | 0.0360 | 0.0131 |
| single_subtask_omission | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2990 | 0.8293 | noise+action_within_snapshot | -0.0170 | -0.0604 | -0.0141 |
| single_subtask_omission | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2742 | 0.8596 | noise+action_within_snapshot | 0.0078 | 0.0276 | 0.0108 |
| single_subtask_omission | 3 | failure_rate_only | absolute | 130 | 46 | 0.2820 | 0.7718 | noise+action | 0.0000 | 0.0000 | 0.0243 |
| single_subtask_omission | 3 | noise+action | absolute | 130 | 46 | 0.3064 | 1.0792 | noise+action | -0.0243 | -0.0863 | 0.0000 |
| single_subtask_omission | 3 | moe_action | absolute | 130 | 46 | 0.2658 | 1.0349 | noise+action | 0.0163 | 0.0577 | 0.0406 |
| single_subtask_omission | 3 | moe_state | absolute | 130 | 46 | 0.3097 | 1.6858 | noise+action | -0.0277 | -0.0981 | -0.0033 |
| single_subtask_omission | 3 | moe_action+state | absolute | 130 | 46 | 0.2380 | 1.1757 | noise+action | 0.0440 | 0.1562 | 0.0684 |
| single_subtask_omission | 3 | noise+action+moe_action | absolute | 130 | 46 | 0.2697 | 1.0656 | noise+action | 0.0123 | 0.0437 | 0.0367 |
| single_subtask_omission | 3 | noise+action+moe_action+state | absolute | 130 | 46 | 0.2483 | 1.1614 | noise+action | 0.0337 | 0.1196 | 0.0581 |
| single_subtask_omission | 3 | noise+action_within_snapshot | within_snapshot_centered | 130 | 46 | 0.3062 | 0.8328 | noise+action_within_snapshot | -0.0242 | -0.0857 | 0.0000 |
| single_subtask_omission | 3 | moe_action_within_snapshot | within_snapshot_centered | 130 | 46 | 0.3075 | 0.8429 | noise+action_within_snapshot | -0.0255 | -0.0905 | -0.0013 |
| single_subtask_omission | 3 | moe_state_within_snapshot | within_snapshot_centered | 130 | 46 | 0.2695 | 0.8278 | noise+action_within_snapshot | 0.0125 | 0.0444 | 0.0367 |
| single_subtask_omission | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 46 | 0.3040 | 0.8352 | noise+action_within_snapshot | -0.0220 | -0.0780 | 0.0022 |
| single_subtask_omission | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 46 | 0.3082 | 0.8443 | noise+action_within_snapshot | -0.0262 | -0.0930 | -0.0020 |
| single_subtask_omission | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 46 | 0.3040 | 0.8347 | noise+action_within_snapshot | -0.0220 | -0.0781 | 0.0021 |
| non_stagnation_non_loop | 1 | failure_rate_only | absolute | 130 | 18 | 0.1215 | 0.4110 | noise+action | 0.0000 | 0.0000 | 0.0069 |
| non_stagnation_non_loop | 1 | noise+action | absolute | 130 | 18 | 0.1283 | 0.4612 | noise+action | -0.0069 | -0.0566 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action | absolute | 130 | 18 | 0.1977 | 0.9076 | noise+action | -0.0763 | -0.6281 | -0.0694 |
| non_stagnation_non_loop | 1 | moe_state | absolute | 130 | 18 | 0.2478 | 1.3218 | noise+action | -0.1264 | -1.0404 | -0.1195 |
| non_stagnation_non_loop | 1 | moe_action+state | absolute | 130 | 18 | 0.2856 | 1.7093 | noise+action | -0.1642 | -1.3515 | -0.1573 |
| non_stagnation_non_loop | 1 | noise+action+moe_action | absolute | 130 | 18 | 0.1947 | 0.8973 | noise+action | -0.0732 | -0.6030 | -0.0664 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state | absolute | 130 | 18 | 0.2919 | 1.7679 | noise+action | -0.1705 | -1.4035 | -0.1636 |
| non_stagnation_non_loop | 1 | noise+action_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1268 | 0.4474 | noise+action_within_snapshot | -0.0053 | -0.0436 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1351 | 0.4962 | noise+action_within_snapshot | -0.0136 | -0.1123 | -0.0084 |
| non_stagnation_non_loop | 1 | moe_state_within_snapshot | within_snapshot_centered | 130 | 18 | 0.2095 | 1.6860 | noise+action_within_snapshot | -0.0881 | -0.7249 | -0.0828 |
| non_stagnation_non_loop | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1407 | 0.5360 | noise+action_within_snapshot | -0.0193 | -0.1587 | -0.0140 |
| non_stagnation_non_loop | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1352 | 0.4964 | noise+action_within_snapshot | -0.0137 | -0.1127 | -0.0084 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1387 | 0.5335 | noise+action_within_snapshot | -0.0173 | -0.1423 | -0.0120 |
| non_stagnation_non_loop | 3 | failure_rate_only | absolute | 130 | 18 | 0.1215 | 0.4110 | noise+action | 0.0000 | 0.0000 | 0.0101 |
| non_stagnation_non_loop | 3 | noise+action | absolute | 130 | 18 | 0.1316 | 0.4710 | noise+action | -0.0101 | -0.0835 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action | absolute | 130 | 18 | 0.2387 | 0.9331 | noise+action | -0.1172 | -0.9649 | -0.1071 |
| non_stagnation_non_loop | 3 | moe_state | absolute | 130 | 18 | 0.3067 | 2.5671 | noise+action | -0.1852 | -1.5247 | -0.1751 |
| non_stagnation_non_loop | 3 | moe_action+state | absolute | 130 | 18 | 0.3222 | 1.7751 | noise+action | -0.2007 | -1.6524 | -0.1906 |
| non_stagnation_non_loop | 3 | noise+action+moe_action | absolute | 130 | 18 | 0.2308 | 0.9098 | noise+action | -0.1093 | -0.9001 | -0.0992 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state | absolute | 130 | 18 | 0.3258 | 1.7910 | noise+action | -0.2043 | -1.6820 | -0.1942 |
| non_stagnation_non_loop | 3 | noise+action_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1267 | 0.4490 | noise+action_within_snapshot | -0.0053 | -0.0433 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1339 | 0.5140 | noise+action_within_snapshot | -0.0125 | -0.1025 | -0.0072 |
| non_stagnation_non_loop | 3 | moe_state_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1412 | 0.6123 | noise+action_within_snapshot | -0.0197 | -0.1625 | -0.0145 |
| non_stagnation_non_loop | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1390 | 0.5491 | noise+action_within_snapshot | -0.0175 | -0.1441 | -0.0122 |
| non_stagnation_non_loop | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1340 | 0.5120 | noise+action_within_snapshot | -0.0125 | -0.1033 | -0.0073 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 130 | 18 | 0.1386 | 0.5364 | noise+action_within_snapshot | -0.0171 | -0.1409 | -0.0118 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_regression | 3 | action | back_12_15 | 9 | entropy | 0.0000 | 0.3077 | -0.3077 | -0.3333 | -0.2667 | 0.0160 | 0.4531 | 13 | 13 |
| stagnation | 3 | action | back_12_15 | 2 | consensus_distance | 0.2273 | 0.5455 | -0.3182 | -0.4750 | -0.1667 | 0.0379 | 0.5469 | 22 | 22 |
| stagnation | 3 | action | back_12_15 | 6 | consensus_distance | 0.2273 | 0.5455 | -0.3182 | -0.4750 | -0.1818 | 0.0379 | 0.5469 | 22 | 22 |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.4783 | 0.8261 | -0.3478 | -0.5789 | -0.1667 | 0.0100 | 0.5968 | 23 | 23 |
| drop_or_regrasp | 1 | state | front_2_5 | 1 | top1_mass | 0.3636 | 0.0909 | 0.2727 | 0.1276 | 0.4524 | 0.0240 | 0.6986 | 22 | 22 |
| non_stagnation_non_loop | 1 | state | back_12_15 | 8 | top1_mass | 0.0435 | 0.3043 | -0.2609 | -0.4646 | -0.1250 | 0.0180 | 0.7166 | 23 | 23 |
| non_stagnation_non_loop | 1 | action | front_2_5 | 8 | token_dispersion | 0.3043 | 0.0435 | 0.2609 | 0.1739 | 0.3500 | 0.0240 | 0.7166 | 23 | 23 |
| goal_regression | 1 | action | front_2_5 | 5 | entropy | 0.0000 | 0.3077 | -0.3077 | -0.5385 | -0.0833 | 0.0259 | 0.7904 | 13 | 13 |
| goal_regression | 1 | action | front_2_5 | 6 | entropy | 0.0000 | 0.3077 | -0.3077 | -0.5385 | -0.0833 | 0.0279 | 0.7904 | 13 | 13 |
| stagnation | 3 | action | back_12_15 | 0 | entropy | 0.4091 | 0.1364 | 0.2727 | 0.0870 | 0.5000 | 0.0499 | 0.7964 | 22 | 22 |
| stagnation | 3 | state | front_2_5 | 0 | entropy | 0.1818 | 0.4545 | -0.2727 | -0.5238 | -0.0816 | 0.0519 | 0.7964 | 22 | 22 |
| stagnation | 3 | state | front_2_5 | 1 | entropy | 0.1818 | 0.4545 | -0.2727 | -0.5238 | -0.0816 | 0.0519 | 0.7964 | 22 | 22 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
