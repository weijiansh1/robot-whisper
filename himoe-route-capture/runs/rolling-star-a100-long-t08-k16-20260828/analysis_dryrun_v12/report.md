# Rolling-star K=16 experiment

## Dataset

- 128 terminal branches from 8 committed snapshots; 42 success and 86 failure.
- 5 snapshots contain matched success/failure siblings.
- 6095 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 49
- `single_subtask_omission`: 20
- `drop_or_regrasp`: 7
- `subtask_undo`: 4
- `goal_contact_near_miss`: 3
- `stagnation`: 1
- `timeout_other`: 1
- `active_retry`: 1

Stagnation or loop/cycling covers 76/86 failures; 10 failures require other physical mechanisms: {"active_retry": 1, "goal_contact_near_miss": 3, "single_subtask_omission": 5, "timeout_other": 1}.

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route maximum probability span within matched siblings is 0.00394.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (both snapshot-held-out and core worker-held-out checks):

| prefix_queries | cv_scheme | model | feature_scope | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | noise_action_reference | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | leave_one_snapshot_out | snapshot_rate_only | absolute | 128 | 5 | 0.2475 | 0.6968 | 0.5250 | 0.5250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2035 | 0.0794 | 0.3668 |
| 1 | leave_one_snapshot_out | noise | absolute | 128 | 5 | 0.2580 | 0.7291 | 0.4000 | 0.5250 | -0.1250 | -0.3000 | 0.0375 | -0.0105 | -0.0426 | noise+action | -0.0105 | -0.0227 | 0.0008 | 0.1930 | 0.0674 | 0.3766 |
| 1 | leave_one_snapshot_out | action | absolute | 128 | 5 | 0.3683 | 1.0531 | 0.2000 | 0.5250 | -0.3250 | -0.6625 | -0.0500 | -0.1208 | -0.4883 | noise+action | -0.1208 | -0.2969 | 0.0365 | 0.0827 | -0.0921 | 0.2308 |
| 1 | leave_one_snapshot_out | noise+action | absolute | 128 | 5 | 0.4510 | 1.4799 | 0.2000 | 0.5250 | -0.3250 | -0.6375 | -0.0625 | -0.2035 | -0.8225 | noise+action | -0.2035 | -0.3731 | -0.0905 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action | absolute | 128 | 5 | 0.2347 | 0.8311 | 0.2000 | 0.5250 | -0.3250 | -0.6506 | -0.0500 | 0.0128 | 0.0515 | noise+action | 0.0128 | -0.0990 | 0.1310 | 0.2163 | 0.0484 | 0.4595 |
| 1 | leave_one_snapshot_out | moe_state | absolute | 128 | 5 | 0.2959 | 1.0397 | 0.6000 | 0.5250 | 0.0750 | -0.1250 | 0.3375 | -0.0484 | -0.1956 | noise+action | -0.0484 | -0.2229 | 0.0983 | 0.1551 | -0.0770 | 0.4097 |
| 1 | leave_one_snapshot_out | moe_action+state | absolute | 128 | 5 | 0.2953 | 1.1929 | 0.2000 | 0.5250 | -0.3250 | -0.6316 | -0.0500 | -0.0479 | -0.1934 | noise+action | -0.0479 | -0.2121 | 0.0938 | 0.1557 | -0.0501 | 0.4236 |
| 1 | leave_one_snapshot_out | noise+action+moe_action | absolute | 128 | 5 | 0.2318 | 0.7979 | 0.2000 | 0.5250 | -0.3250 | -0.6822 | -0.0559 | 0.0156 | 0.0630 | noise+action | 0.0156 | -0.0833 | 0.1272 | 0.2191 | 0.0429 | 0.4632 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 128 | 5 | 0.2885 | 1.1529 | 0.2000 | 0.5250 | -0.3250 | -0.6625 | -0.0500 | -0.0411 | -0.1659 | noise+action | -0.0411 | -0.1924 | 0.0961 | 0.1625 | -0.0427 | 0.3946 |
| 1 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2507 | 0.7070 | 0.4000 | 0.5250 | -0.1250 | -0.3250 | 0.0375 | -0.0033 | -0.0131 | noise+action_within_snapshot | -0.0033 | -0.0062 | -0.0001 | 0.0004 | -0.0018 | 0.0024 |
| 1 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2564 | 0.7228 | 0.4000 | 0.5250 | -0.1250 | -0.5756 | 0.2750 | -0.0089 | -0.0360 | noise+action_within_snapshot | -0.0089 | -0.0140 | -0.0037 | -0.0052 | -0.0088 | -0.0021 |
| 1 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2511 | 0.7079 | 0.6000 | 0.5250 | 0.0750 | -0.1441 | 0.3625 | -0.0037 | -0.0148 | noise+action_within_snapshot | -0.0037 | -0.0072 | -0.0001 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2606 | 0.7407 | 0.2000 | 0.5250 | -0.3250 | -0.6625 | -0.0559 | -0.0132 | -0.0532 | noise+action_within_snapshot | -0.0132 | -0.0241 | -0.0037 | -0.0095 | -0.0194 | -0.0005 |
| 1 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2843 | 1.0951 | 0.8000 | 0.5250 | 0.2750 | 0.0125 | 0.5441 | -0.0369 | -0.1489 | noise+action_within_snapshot | -0.0369 | -0.0687 | -0.0062 | -0.0332 | -0.0725 | -0.0043 |
| 1 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3275 | 1.0024 | 0.6000 | 0.5250 | 0.0750 | -0.1250 | 0.3375 | -0.0801 | -0.3236 | noise+action_within_snapshot | -0.0801 | -0.1989 | -0.0075 | -0.0764 | -0.1955 | -0.0041 |
| 1 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2605 | 0.7396 | 0.2000 | 0.5250 | -0.3250 | -0.6316 | -0.0625 | -0.0130 | -0.0527 | noise+action_within_snapshot | -0.0130 | -0.0235 | -0.0039 | -0.0094 | -0.0185 | -0.0001 |
| 1 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3277 | 1.0038 | 0.6000 | 0.5250 | 0.0750 | -0.1316 | 0.3625 | -0.0802 | -0.3243 | noise+action_within_snapshot | -0.0802 | -0.2032 | -0.0085 | -0.0766 | -0.1946 | -0.0056 |
| 3 | leave_one_snapshot_out | snapshot_rate_only | absolute | 128 | 5 | 0.2475 | 0.6968 | 0.5250 | 0.5250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2322 | 0.0929 | 0.3969 |
| 3 | leave_one_snapshot_out | noise | absolute | 128 | 5 | 0.2524 | 0.7169 | 0.6000 | 0.5250 | 0.0750 | -0.1441 | 0.3125 | -0.0050 | -0.0202 | noise+action | -0.0050 | -0.0224 | 0.0114 | 0.2272 | 0.0927 | 0.3877 |
| 3 | leave_one_snapshot_out | action | absolute | 128 | 5 | 0.3671 | 1.1320 | 0.4000 | 0.5250 | -0.1250 | -0.3250 | 0.0375 | -0.1196 | -0.4834 | noise+action | -0.1196 | -0.2368 | -0.0308 | 0.1125 | -0.0401 | 0.2950 |
| 3 | leave_one_snapshot_out | noise+action | absolute | 128 | 5 | 0.4796 | 1.4249 | 0.6000 | 0.5250 | 0.0750 | -0.2625 | 0.4500 | -0.2322 | -0.9383 | noise+action | -0.2322 | -0.3903 | -0.1114 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action | absolute | 128 | 5 | 0.2516 | 0.8616 | 0.6000 | 0.5250 | 0.0750 | -0.2625 | 0.5072 | -0.0042 | -0.0169 | noise+action | -0.0042 | -0.1407 | 0.1018 | 0.2280 | 0.0622 | 0.4174 |
| 3 | leave_one_snapshot_out | moe_state | absolute | 128 | 5 | 0.3236 | 1.3036 | 0.4000 | 0.5250 | -0.1250 | -0.2875 | 0.0125 | -0.0762 | -0.3079 | noise+action | -0.0762 | -0.2324 | 0.0542 | 0.1560 | -0.0356 | 0.4020 |
| 3 | leave_one_snapshot_out | moe_action+state | absolute | 128 | 5 | 0.3042 | 1.3025 | 0.4000 | 0.5250 | -0.1250 | -0.2875 | 0.0375 | -0.0568 | -0.2295 | noise+action | -0.0568 | -0.2107 | 0.0812 | 0.1754 | -0.0133 | 0.4046 |
| 3 | leave_one_snapshot_out | noise+action+moe_action | absolute | 128 | 5 | 0.2599 | 0.9276 | 0.6000 | 0.5250 | 0.0750 | -0.2625 | 0.5250 | -0.0124 | -0.0502 | noise+action | -0.0124 | -0.1421 | 0.1211 | 0.2198 | 0.0364 | 0.4276 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state | absolute | 128 | 5 | 0.2971 | 1.2713 | 0.4000 | 0.5250 | -0.1250 | -0.2941 | 0.0125 | -0.0497 | -0.2008 | noise+action | -0.0497 | -0.2054 | 0.0867 | 0.1825 | -0.0093 | 0.3954 |
| 3 | leave_one_snapshot_out | noise_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2496 | 0.7040 | 0.4000 | 0.5250 | -0.1250 | -0.3000 | 0.0375 | -0.0021 | -0.0085 | noise+action_within_snapshot | -0.0021 | -0.0066 | 0.0021 | -0.0009 | -0.0029 | 0.0013 |
| 3 | leave_one_snapshot_out | action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2563 | 0.7203 | 0.6000 | 0.5250 | 0.0750 | -0.1441 | 0.3375 | -0.0089 | -0.0360 | noise+action_within_snapshot | -0.0089 | -0.0158 | -0.0034 | -0.0077 | -0.0151 | -0.0007 |
| 3 | leave_one_snapshot_out | noise+action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2486 | 0.7002 | 0.6000 | 0.5250 | 0.0750 | -0.1316 | 0.3375 | -0.0012 | -0.0048 | noise+action_within_snapshot | -0.0012 | -0.0056 | 0.0023 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_snapshot_out | moe_action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2519 | 0.7118 | 0.6000 | 0.5250 | 0.0750 | -0.2625 | 0.5250 | -0.0045 | -0.0180 | noise+action_within_snapshot | -0.0045 | -0.0093 | 0.0001 | -0.0033 | -0.0093 | 0.0015 |
| 3 | leave_one_snapshot_out | moe_state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3400 | 1.3401 | 0.6000 | 0.5250 | 0.0750 | -0.1500 | 0.3000 | -0.0926 | -0.3741 | noise+action_within_snapshot | -0.0926 | -0.2462 | -0.0012 | -0.0914 | -0.2364 | 0.0016 |
| 3 | leave_one_snapshot_out | moe_action+state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2960 | 0.8276 | 0.6000 | 0.5250 | 0.0750 | -0.1375 | 0.3375 | -0.0485 | -0.1961 | noise+action_within_snapshot | -0.0485 | -0.1271 | -0.0040 | -0.0473 | -0.1306 | -0.0004 |
| 3 | leave_one_snapshot_out | noise+action+moe_action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2518 | 0.7114 | 0.4000 | 0.5250 | -0.1250 | -0.3000 | 0.0375 | -0.0044 | -0.0177 | noise+action_within_snapshot | -0.0044 | -0.0092 | 0.0005 | -0.0032 | -0.0093 | 0.0016 |
| 3 | leave_one_snapshot_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.2945 | 0.8218 | 0.6000 | 0.5250 | 0.0750 | -0.1250 | 0.3375 | -0.0470 | -0.1901 | noise+action_within_snapshot | -0.0470 | -0.1213 | -0.0043 | -0.0459 | -0.1225 | -0.0018 |
| 1 | leave_one_worker_out | snapshot_rate_only | absolute | 128 | 5 | 0.3308 | 0.9360 | 0.5250 | 0.5250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2313 | 0.0225 | 0.5873 |
| 1 | leave_one_worker_out | noise+action | absolute | 128 | 5 | 0.5621 | 2.0453 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.2313 | -0.6992 | noise+action | -0.2313 | -0.5873 | 0.0189 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action | absolute | 128 | 5 | 0.3973 | 1.5105 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0375 | -0.0665 | -0.2011 | noise+action | -0.0665 | -0.1405 | 0.0020 | 0.1648 | -0.0766 | 0.5548 |
| 1 | leave_one_worker_out | moe_state | absolute | 128 | 5 | 0.3358 | 1.1326 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.0050 | -0.0150 | noise+action | -0.0050 | -0.1433 | 0.1334 | 0.2263 | -0.0293 | 0.6246 |
| 1 | leave_one_worker_out | noise+action+moe_action+state | absolute | 128 | 5 | 0.3850 | 1.4882 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.0542 | -0.1639 | noise+action | -0.0542 | -0.1711 | 0.0627 | 0.1771 | -0.0458 | 0.5887 |
| 1 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3349 | 0.9750 | 0.6000 | 0.5250 | 0.0750 | -0.0625 | 0.1562 | -0.0041 | -0.0125 | noise+action_within_snapshot | -0.0041 | -0.0136 | 0.0032 | 0.0000 | 0.0000 | 0.0000 |
| 1 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3374 | 0.9966 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.0066 | -0.0200 | noise+action_within_snapshot | -0.0066 | -0.0106 | -0.0045 | -0.0025 | -0.0083 | 0.0030 |
| 1 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.4277 | 2.0896 | 0.6000 | 0.5250 | 0.0750 | -0.0625 | 0.1562 | -0.0969 | -0.2929 | noise+action_within_snapshot | -0.0969 | -0.1274 | -0.0653 | -0.0928 | -0.1266 | -0.0589 |
| 1 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3498 | 1.0539 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0375 | -0.0191 | -0.0576 | noise+action_within_snapshot | -0.0191 | -0.0374 | -0.0007 | -0.0149 | -0.0265 | -0.0033 |
| 3 | leave_one_worker_out | snapshot_rate_only | absolute | 128 | 5 | 0.3308 | 0.9360 | 0.5250 | 0.5250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | noise+action | 0.0000 | 0.0000 | 0.0000 | 0.2298 | 0.0604 | 0.5347 |
| 3 | leave_one_worker_out | noise+action | absolute | 128 | 5 | 0.5606 | 1.9287 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.2298 | -0.6948 | noise+action | -0.2298 | -0.5276 | -0.0604 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action | absolute | 128 | 5 | 0.4872 | 2.1355 | 0.6000 | 0.5250 | 0.0750 | -0.3438 | 0.9375 | -0.1564 | -0.4728 | noise+action | -0.1564 | -0.2443 | -0.0680 | 0.0734 | -0.1563 | 0.4732 |
| 3 | leave_one_worker_out | moe_state | absolute | 128 | 5 | 0.5635 | 2.1398 | 0.6000 | 0.5250 | 0.0750 | -0.0625 | 0.1562 | -0.2327 | -0.7035 | noise+action | -0.2327 | -0.3569 | -0.0999 | -0.0029 | -0.1600 | 0.1719 |
| 3 | leave_one_worker_out | noise+action+moe_action+state | absolute | 128 | 5 | 0.5203 | 2.2761 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.1895 | -0.5728 | noise+action | -0.1895 | -0.2141 | -0.1634 | 0.0404 | -0.1163 | 0.3264 |
| 3 | leave_one_worker_out | noise+action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3381 | 0.9745 | 0.4000 | 0.5250 | -0.1250 | -0.3438 | 0.0625 | -0.0073 | -0.0222 | noise+action_within_snapshot | -0.0073 | -0.0175 | -0.0005 | 0.0000 | 0.0000 | 0.0000 |
| 3 | leave_one_worker_out | moe_action_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3339 | 0.9704 | 0.6000 | 0.5250 | 0.0750 | -0.3438 | 0.9375 | -0.0031 | -0.0095 | noise+action_within_snapshot | -0.0031 | -0.0092 | 0.0031 | 0.0042 | -0.0013 | 0.0101 |
| 3 | leave_one_worker_out | moe_state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3532 | 1.0633 | 0.6000 | 0.5250 | 0.0750 | -0.0625 | 0.1562 | -0.0224 | -0.0676 | noise+action_within_snapshot | -0.0224 | -0.0502 | 0.0007 | -0.0150 | -0.0438 | 0.0033 |
| 3 | leave_one_worker_out | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 128 | 5 | 0.3349 | 0.9695 | 0.6000 | 0.5250 | 0.0750 | -0.0625 | 0.1562 | -0.0041 | -0.0125 | noise+action_within_snapshot | -0.0041 | -0.0112 | 0.0004 | 0.0032 | -0.0016 | 0.0080 |

Snapshot difficulty (one row per rolling state; snapshot and worker holdouts):

| cv_scheme | model | snapshots | heldout_units | ridge_alpha | failure_rate_mse | failure_rate_mae | mse_gain_vs_mean_rate | mse_gain_vs_mean_rate_ci95_low | mse_gain_vs_mean_rate_ci95_high | mse_gain_vs_policy_state | mse_gain_vs_policy_state_ci95_low | mse_gain_vs_policy_state_ci95_high | mse_gain_vs_sim_state | mse_gain_vs_sim_state_ci95_low | mse_gain_vs_sim_state_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_snapshot_out | mean_moe_action | 8 | 8 | 10.0000 | 0.0676 | 0.2394 | 0.1208 | 0.0155 | 0.2346 | 0.0958 | 0.0190 | 0.1867 | 0.0622 | 0.0129 | 0.1156 |
| leave_one_snapshot_out | mean_moe_action+state | 8 | 8 | 10.0000 | 0.0906 | 0.2684 | 0.0978 | 0.0081 | 0.1952 | 0.0728 | 0.0190 | 0.1461 | 0.0393 | -0.0000 | 0.0977 |
| leave_one_snapshot_out | policy_state+mean_moe_action+state | 8 | 8 | 10.0000 | 0.0906 | 0.2683 | 0.0978 | -0.0051 | 0.2079 | 0.0728 | 0.0116 | 0.1341 | 0.0393 | -0.0001 | 0.0941 |
| leave_one_snapshot_out | sim_state+mean_moe_action+state | 8 | 8 | 10.0000 | 0.0907 | 0.2683 | 0.0978 | 0.0037 | 0.1990 | 0.0728 | 0.0160 | 0.1342 | 0.0392 | 0.0002 | 0.0961 |
| leave_one_snapshot_out | mean_moe_state | 8 | 8 | 10.0000 | 0.1200 | 0.2846 | 0.0685 | -0.0305 | 0.1538 | 0.0435 | -0.0124 | 0.0983 | 0.0099 | -0.0346 | 0.0762 |
| leave_one_snapshot_out | initial_sim_state | 8 | 8 | 10.0000 | 0.1299 | 0.3142 | 0.0586 | -0.0348 | 0.1578 | 0.0336 | -0.0371 | 0.0978 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_snapshot_out | initial_policy_state | 8 | 8 | 10.0000 | 0.1634 | 0.3449 | 0.0250 | -0.0326 | 0.0798 | 0.0000 | 0.0000 | 0.0000 | -0.0336 | -0.1087 | 0.0360 |
| leave_one_snapshot_out | mean_output_action | 8 | 8 | 10.0000 | 0.1855 | 0.4005 | 0.0029 | -0.1581 | 0.1621 | -0.0221 | -0.1484 | 0.1020 | -0.0556 | -0.1508 | 0.0140 |
| leave_one_snapshot_out | mean_rate | 8 | 8 | NA | 0.1885 | 0.3795 | 0.0000 | 0.0000 | 0.0000 | -0.0250 | -0.0746 | 0.0385 | -0.0586 | -0.1599 | 0.0428 |
| leave_one_snapshot_out | as_moe | 8 | 8 | 10.0000 | 0.1885 | 0.3795 | -0.0000 | -0.0000 | 0.0000 | -0.0250 | -0.0859 | 0.0356 | -0.0586 | -0.1595 | 0.0496 |
| leave_one_worker_out | initial_policy_state | 8 | 4 | 10.0000 | 0.2182 | 0.4055 | 0.0364 | -0.0391 | 0.1168 | 0.0000 | 0.0000 | 0.0000 | 0.0428 | -0.0671 | 0.2273 |
| leave_one_worker_out | as_moe | 8 | 4 | 10.0000 | 0.2546 | 0.4349 | 0.0000 | 0.0000 | 0.0000 | -0.0364 | -0.1168 | 0.0440 | 0.0063 | -0.1693 | 0.2182 |
| leave_one_worker_out | mean_rate | 8 | 4 | NA | 0.2546 | 0.4349 | 0.0000 | 0.0000 | 0.0000 | -0.0364 | -0.1168 | 0.0440 | 0.0063 | -0.1693 | 0.2182 |
| leave_one_worker_out | mean_moe_action | 8 | 4 | 10.0000 | 0.2572 | 0.4321 | -0.0026 | -0.0859 | 0.0747 | -0.0391 | -0.1024 | 0.0243 | 0.0037 | -0.1093 | 0.1543 |
| leave_one_worker_out | initial_sim_state | 8 | 4 | 10.0000 | 0.2609 | 0.3973 | -0.0063 | -0.2182 | 0.1693 | -0.0428 | -0.2288 | 0.0642 | 0.0000 | 0.0000 | 0.0000 |
| leave_one_worker_out | sim_state+mean_moe_action+state | 8 | 4 | 10.0000 | 0.2654 | 0.4511 | -0.0108 | -0.1150 | 0.0896 | -0.0472 | -0.0977 | 0.0033 | -0.0044 | -0.1044 | 0.1460 |
| leave_one_worker_out | policy_state+mean_moe_action+state | 8 | 4 | 10.0000 | 0.2655 | 0.4521 | -0.0109 | -0.1071 | 0.0880 | -0.0473 | -0.0964 | 0.0017 | -0.0046 | -0.1048 | 0.1511 |
| leave_one_worker_out | mean_moe_action+state | 8 | 4 | 10.0000 | 0.2656 | 0.4522 | -0.0110 | -0.1071 | 0.0880 | -0.0475 | -0.0967 | 0.0018 | -0.0047 | -0.1051 | 0.1474 |
| leave_one_worker_out | mean_moe_state | 8 | 4 | 10.0000 | 0.2872 | 0.4782 | -0.0326 | -0.1723 | 0.1072 | -0.0690 | -0.1365 | 0.0187 | -0.0262 | -0.1663 | 0.1577 |
| leave_one_worker_out | mean_output_action | 8 | 4 | 10.0000 | 0.4144 | 0.5614 | -0.1598 | -0.5667 | 0.0989 | -0.1963 | -0.5678 | 0.0064 | -0.1535 | -0.3484 | -0.0416 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | state | back_12_15 | 7 | entropy | 0.5500 | 0.3000 | 0.2500 | -0.0500 | 0.6500 | 0.0448 | 0.9652 | 20 | 20 |
| 1 | state | front_2_5 | 8 | entropy | 0.5500 | 0.3000 | 0.2500 | 0.0500 | 0.5500 | 0.0448 | 0.9652 | 20 | 20 |
| 1 | state | back_12_15 | 8 | top1_mass | 0.3500 | 0.6000 | -0.2500 | -0.5000 | -0.0500 | 0.0547 | 0.9652 | 20 | 20 |
| 1 | state | front_2_5 | 1 | top1_mass | 0.5500 | 0.3500 | 0.2000 | 0.1000 | 0.2500 | 0.0796 | 1.0000 | 20 | 20 |
| 1 | state | front_2_5 | 0 | consensus_distance | 0.5000 | 0.3000 | 0.2000 | 0.1000 | 0.2500 | 0.1194 | 1.0000 | 20 | 20 |
| 1 | action | back_12_15 | 0 | entropy | 0.4000 | 0.6000 | -0.2000 | -0.3500 | -0.0500 | 0.1294 | 1.0000 | 20 | 20 |
| 1 | action | back_12_15 | 5 | top1_mass | 0.3000 | 0.5000 | -0.2000 | -0.2500 | -0.1000 | 0.1294 | 1.0000 | 20 | 20 |
| 1 | action | back_12_15 | 6 | top1_mass | 0.3000 | 0.5000 | -0.2000 | -0.3500 | -0.0500 | 0.1393 | 1.0000 | 20 | 20 |
| 1 | state | back_12_15 | 0 | top1_mass | 0.3500 | 0.5500 | -0.2000 | -0.4000 | 0.0000 | 0.1493 | 1.0000 | 20 | 20 |
| 1 | action | front_2_5 | 9 | consensus_distance | 0.5500 | 0.4000 | 0.1500 | 0.0500 | 0.2500 | 0.2289 | 1.0000 | 20 | 20 |
| 1 | action | back_12_15 | 9 | entropy | 0.4000 | 0.5500 | -0.1500 | -0.4500 | 0.0000 | 0.2488 | 1.0000 | 20 | 20 |
| 1 | action | front_2_5 | 9 | token_dispersion | 0.6000 | 0.4500 | 0.1500 | -0.1000 | 0.4500 | 0.2537 | 1.0000 | 20 | 20 |

Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):

| prefix_queries | token_family | layer_group | denoise_step | expert | failure_minus_success_probability | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_mixed_snapshots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | action | front_2_5 | 0 | 0 | 0.0005 | -0.0001 | 0.0010 | 0.0249 | 0.0348 | 5 |
| 1 | action | front_2_5 | 1 | 0 | 0.0004 | -0.0001 | 0.0010 | 0.0249 | 0.0597 | 5 |
| 1 | action | front_2_5 | 2 | 0 | 0.0004 | -0.0002 | 0.0009 | 0.0348 | 0.2139 | 5 |
| 1 | action | front_2_5 | 3 | 0 | 0.0003 | -0.0002 | 0.0007 | 0.0647 | 0.5124 | 5 |
| 1 | action | front_2_5 | 0 | 12 | -0.0003 | -0.0007 | 0.0001 | 0.0597 | 0.5821 | 5 |
| 1 | action | front_2_5 | 7 | 12 | 0.0003 | -0.0000 | 0.0007 | 0.1045 | 0.6567 | 5 |
| 1 | action | front_2_5 | 9 | 6 | 0.0003 | 0.0001 | 0.0004 | 0.0199 | 0.6667 | 5 |
| 1 | action | front_2_5 | 8 | 12 | 0.0003 | -0.0000 | 0.0007 | 0.1095 | 0.6816 | 5 |
| 1 | action | front_2_5 | 9 | 31 | -0.0003 | -0.0004 | -0.0001 | 0.0249 | 0.7015 | 5 |
| 1 | action | front_2_5 | 9 | 19 | -0.0003 | -0.0004 | -0.0001 | 0.0448 | 0.7313 | 5 |
| 1 | action | front_2_5 | 0 | 29 | 0.0002 | 0.0001 | 0.0004 | 0.0597 | 0.7811 | 5 |
| 1 | action | front_2_5 | 8 | 6 | 0.0002 | 0.0000 | 0.0004 | 0.0597 | 0.7861 | 5 |

Failure-subtype models, evaluated only among failed branches:

| failure_type | prefix_queries | model | feature_scope | n_failure_branches | n_type_positive | brier | log_loss | noise_action_reference | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_improvement_vs_noise_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | 1 | failure_rate_only | absolute | 86 | 21 | 0.2039 | 0.6100 | noise+action | 0.0000 | 0.0000 | 0.0298 |
| stagnation | 1 | noise+action | absolute | 86 | 21 | 0.2337 | 0.7904 | noise+action | -0.0298 | -0.1461 | 0.0000 |
| stagnation | 1 | moe_action | absolute | 86 | 21 | 0.3072 | 2.3008 | noise+action | -0.1034 | -0.5070 | -0.0736 |
| stagnation | 1 | moe_state | absolute | 86 | 21 | 0.2278 | 1.0087 | noise+action | -0.0239 | -0.1172 | 0.0059 |
| stagnation | 1 | moe_action+state | absolute | 86 | 21 | 0.2957 | 2.4454 | noise+action | -0.0918 | -0.4504 | -0.0620 |
| stagnation | 1 | noise+action+moe_action | absolute | 86 | 21 | 0.3049 | 2.3860 | noise+action | -0.1010 | -0.4954 | -0.0712 |
| stagnation | 1 | noise+action+moe_action+state | absolute | 86 | 21 | 0.2962 | 2.4374 | noise+action | -0.0923 | -0.4528 | -0.0625 |
| stagnation | 1 | noise+action_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2151 | 0.6583 | noise+action_within_snapshot | -0.0112 | -0.0551 | 0.0000 |
| stagnation | 1 | moe_action_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2171 | 0.6560 | noise+action_within_snapshot | -0.0132 | -0.0650 | -0.0020 |
| stagnation | 1 | moe_state_within_snapshot | within_snapshot_centered | 86 | 21 | 0.1956 | 1.2875 | noise+action_within_snapshot | 0.0083 | 0.0405 | 0.0195 |
| stagnation | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2112 | 0.8376 | noise+action_within_snapshot | -0.0073 | -0.0360 | 0.0039 |
| stagnation | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2160 | 0.6503 | noise+action_within_snapshot | -0.0122 | -0.0597 | -0.0009 |
| stagnation | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2073 | 0.8117 | noise+action_within_snapshot | -0.0035 | -0.0170 | 0.0078 |
| stagnation | 3 | failure_rate_only | absolute | 86 | 21 | 0.2039 | 0.6100 | noise+action | 0.0000 | 0.0000 | 0.0330 |
| stagnation | 3 | noise+action | absolute | 86 | 21 | 0.2369 | 0.9872 | noise+action | -0.0330 | -0.1618 | 0.0000 |
| stagnation | 3 | moe_action | absolute | 86 | 21 | 0.2564 | 1.0605 | noise+action | -0.0525 | -0.2577 | -0.0196 |
| stagnation | 3 | moe_state | absolute | 86 | 21 | 0.3068 | 2.4474 | noise+action | -0.1029 | -0.5050 | -0.0700 |
| stagnation | 3 | moe_action+state | absolute | 86 | 21 | 0.2858 | 2.5909 | noise+action | -0.0819 | -0.4018 | -0.0489 |
| stagnation | 3 | noise+action+moe_action | absolute | 86 | 21 | 0.2594 | 1.0525 | noise+action | -0.0555 | -0.2723 | -0.0225 |
| stagnation | 3 | noise+action+moe_action+state | absolute | 86 | 21 | 0.2867 | 2.6771 | noise+action | -0.0828 | -0.4061 | -0.0498 |
| stagnation | 3 | noise+action_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2200 | 0.7265 | noise+action_within_snapshot | -0.0161 | -0.0792 | 0.0000 |
| stagnation | 3 | moe_action_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2158 | 0.7990 | noise+action_within_snapshot | -0.0119 | -0.0585 | 0.0042 |
| stagnation | 3 | moe_state_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2255 | 1.0991 | noise+action_within_snapshot | -0.0217 | -0.1063 | -0.0055 |
| stagnation | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2191 | 0.7483 | noise+action_within_snapshot | -0.0153 | -0.0749 | 0.0009 |
| stagnation | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2192 | 0.8053 | noise+action_within_snapshot | -0.0154 | -0.0754 | 0.0008 |
| stagnation | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 21 | 0.2188 | 0.7564 | noise+action_within_snapshot | -0.0150 | -0.0735 | 0.0012 |
| loop_or_cycling | 1 | failure_rate_only | absolute | 86 | 60 | 0.2328 | 0.6660 | noise+action | 0.0000 | 0.0000 | 0.0239 |
| loop_or_cycling | 1 | noise+action | absolute | 86 | 60 | 0.2566 | 0.7781 | noise+action | -0.0239 | -0.1025 | 0.0000 |
| loop_or_cycling | 1 | moe_action | absolute | 86 | 60 | 0.2221 | 0.6521 | noise+action | 0.0107 | 0.0458 | 0.0345 |
| loop_or_cycling | 1 | moe_state | absolute | 86 | 60 | 0.2349 | 0.7149 | noise+action | -0.0021 | -0.0091 | 0.0217 |
| loop_or_cycling | 1 | moe_action+state | absolute | 86 | 60 | 0.2257 | 0.6400 | noise+action | 0.0070 | 0.0302 | 0.0309 |
| loop_or_cycling | 1 | noise+action+moe_action | absolute | 86 | 60 | 0.2222 | 0.6420 | noise+action | 0.0106 | 0.0455 | 0.0344 |
| loop_or_cycling | 1 | noise+action+moe_action+state | absolute | 86 | 60 | 0.2257 | 0.6492 | noise+action | 0.0070 | 0.0302 | 0.0309 |
| loop_or_cycling | 1 | noise+action_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2574 | 0.7512 | noise+action_within_snapshot | -0.0247 | -0.1059 | 0.0000 |
| loop_or_cycling | 1 | moe_action_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2658 | 0.7731 | noise+action_within_snapshot | -0.0331 | -0.1421 | -0.0084 |
| loop_or_cycling | 1 | moe_state_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2361 | 1.4125 | noise+action_within_snapshot | -0.0033 | -0.0143 | 0.0213 |
| loop_or_cycling | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2400 | 0.8460 | noise+action_within_snapshot | -0.0073 | -0.0312 | 0.0174 |
| loop_or_cycling | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2647 | 0.7672 | noise+action_within_snapshot | -0.0319 | -0.1372 | -0.0073 |
| loop_or_cycling | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2376 | 0.8237 | noise+action_within_snapshot | -0.0049 | -0.0210 | 0.0198 |
| loop_or_cycling | 3 | failure_rate_only | absolute | 86 | 60 | 0.2328 | 0.6660 | noise+action | 0.0000 | 0.0000 | 0.0299 |
| loop_or_cycling | 3 | noise+action | absolute | 86 | 60 | 0.2627 | 0.9518 | noise+action | -0.0299 | -0.1285 | 0.0000 |
| loop_or_cycling | 3 | moe_action | absolute | 86 | 60 | 0.2109 | 0.6306 | noise+action | 0.0219 | 0.0939 | 0.0518 |
| loop_or_cycling | 3 | moe_state | absolute | 86 | 60 | 0.4175 | 1.9088 | noise+action | -0.1847 | -0.7937 | -0.1548 |
| loop_or_cycling | 3 | moe_action+state | absolute | 86 | 60 | 0.3734 | 1.2983 | noise+action | -0.1406 | -0.6041 | -0.1107 |
| loop_or_cycling | 3 | noise+action+moe_action | absolute | 86 | 60 | 0.2107 | 0.6197 | noise+action | 0.0221 | 0.0947 | 0.0520 |
| loop_or_cycling | 3 | noise+action+moe_action+state | absolute | 86 | 60 | 0.3707 | 1.2761 | noise+action | -0.1379 | -0.5925 | -0.1080 |
| loop_or_cycling | 3 | noise+action_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2547 | 0.7809 | noise+action_within_snapshot | -0.0219 | -0.0943 | 0.0000 |
| loop_or_cycling | 3 | moe_action_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2707 | 0.8186 | noise+action_within_snapshot | -0.0379 | -0.1629 | -0.0160 |
| loop_or_cycling | 3 | moe_state_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2945 | 1.0383 | noise+action_within_snapshot | -0.0617 | -0.2652 | -0.0398 |
| loop_or_cycling | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 60 | 0.3006 | 0.9706 | noise+action_within_snapshot | -0.0678 | -0.2914 | -0.0459 |
| loop_or_cycling | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 60 | 0.2735 | 0.8355 | noise+action_within_snapshot | -0.0407 | -0.1750 | -0.0188 |
| loop_or_cycling | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 60 | 0.3009 | 0.9768 | noise+action_within_snapshot | -0.0682 | -0.2928 | -0.0462 |
| drop_or_regrasp | 1 | failure_rate_only | absolute | 86 | 11 | 0.1162 | 0.4041 | noise+action | 0.0000 | 0.0000 | 0.0154 |
| drop_or_regrasp | 1 | noise+action | absolute | 86 | 11 | 0.1316 | 0.6153 | noise+action | -0.0154 | -0.1323 | 0.0000 |
| drop_or_regrasp | 1 | moe_action | absolute | 86 | 11 | 0.2338 | 1.0508 | noise+action | -0.1176 | -1.0125 | -0.1023 |
| drop_or_regrasp | 1 | moe_state | absolute | 86 | 11 | 0.1396 | 0.5196 | noise+action | -0.0234 | -0.2014 | -0.0080 |
| drop_or_regrasp | 1 | moe_action+state | absolute | 86 | 11 | 0.1833 | 0.8549 | noise+action | -0.0671 | -0.5773 | -0.0517 |
| drop_or_regrasp | 1 | noise+action+moe_action | absolute | 86 | 11 | 0.2219 | 1.0270 | noise+action | -0.1057 | -0.9096 | -0.0903 |
| drop_or_regrasp | 1 | noise+action+moe_action+state | absolute | 86 | 11 | 0.1661 | 0.7901 | noise+action | -0.0500 | -0.4301 | -0.0346 |
| drop_or_regrasp | 1 | noise+action_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1224 | 0.4474 | noise+action_within_snapshot | -0.0063 | -0.0538 | 0.0000 |
| drop_or_regrasp | 1 | moe_action_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1333 | 0.8038 | noise+action_within_snapshot | -0.0172 | -0.1477 | -0.0109 |
| drop_or_regrasp | 1 | moe_state_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1767 | 1.1909 | noise+action_within_snapshot | -0.0605 | -0.5209 | -0.0543 |
| drop_or_regrasp | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1338 | 0.8228 | noise+action_within_snapshot | -0.0176 | -0.1513 | -0.0113 |
| drop_or_regrasp | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1319 | 0.7984 | noise+action_within_snapshot | -0.0157 | -0.1350 | -0.0094 |
| drop_or_regrasp | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1402 | 0.8482 | noise+action_within_snapshot | -0.0240 | -0.2066 | -0.0177 |
| drop_or_regrasp | 3 | failure_rate_only | absolute | 86 | 11 | 0.1162 | 0.4041 | noise+action | 0.0000 | 0.0000 | 0.0092 |
| drop_or_regrasp | 3 | noise+action | absolute | 86 | 11 | 0.1254 | 0.5624 | noise+action | -0.0092 | -0.0794 | 0.0000 |
| drop_or_regrasp | 3 | moe_action | absolute | 86 | 11 | 0.1295 | 0.7466 | noise+action | -0.0133 | -0.1146 | -0.0041 |
| drop_or_regrasp | 3 | moe_state | absolute | 86 | 11 | 0.2313 | 1.2975 | noise+action | -0.1152 | -0.9912 | -0.1059 |
| drop_or_regrasp | 3 | moe_action+state | absolute | 86 | 11 | 0.1296 | 0.7947 | noise+action | -0.0134 | -0.1153 | -0.0042 |
| drop_or_regrasp | 3 | noise+action+moe_action | absolute | 86 | 11 | 0.1294 | 0.7546 | noise+action | -0.0133 | -0.1142 | -0.0040 |
| drop_or_regrasp | 3 | noise+action+moe_action+state | absolute | 86 | 11 | 0.1300 | 0.7793 | noise+action | -0.0139 | -0.1192 | -0.0046 |
| drop_or_regrasp | 3 | noise+action_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1265 | 0.5522 | noise+action_within_snapshot | -0.0104 | -0.0892 | 0.0000 |
| drop_or_regrasp | 3 | moe_action_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1250 | 0.8977 | noise+action_within_snapshot | -0.0088 | -0.0759 | 0.0015 |
| drop_or_regrasp | 3 | moe_state_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1544 | 1.0181 | noise+action_within_snapshot | -0.0382 | -0.3291 | -0.0279 |
| drop_or_regrasp | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1347 | 1.0250 | noise+action_within_snapshot | -0.0185 | -0.1596 | -0.0082 |
| drop_or_regrasp | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1259 | 0.9178 | noise+action_within_snapshot | -0.0097 | -0.0839 | 0.0006 |
| drop_or_regrasp | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 11 | 0.1409 | 1.0674 | noise+action_within_snapshot | -0.0247 | -0.2123 | -0.0143 |
| single_subtask_omission | 1 | failure_rate_only | absolute | 86 | 32 | 0.2950 | 0.7958 | noise+action | 0.0000 | 0.0000 | 0.0278 |
| single_subtask_omission | 1 | noise+action | absolute | 86 | 32 | 0.3227 | 1.1326 | noise+action | -0.0278 | -0.0941 | 0.0000 |
| single_subtask_omission | 1 | moe_action | absolute | 86 | 32 | 0.2717 | 1.0565 | noise+action | 0.0232 | 0.0788 | 0.0510 |
| single_subtask_omission | 1 | moe_state | absolute | 86 | 32 | 0.2887 | 1.0826 | noise+action | 0.0063 | 0.0212 | 0.0340 |
| single_subtask_omission | 1 | moe_action+state | absolute | 86 | 32 | 0.2602 | 1.1068 | noise+action | 0.0347 | 0.1177 | 0.0625 |
| single_subtask_omission | 1 | noise+action+moe_action | absolute | 86 | 32 | 0.2747 | 1.1186 | noise+action | 0.0203 | 0.0687 | 0.0480 |
| single_subtask_omission | 1 | noise+action+moe_action+state | absolute | 86 | 32 | 0.2635 | 1.2081 | noise+action | 0.0314 | 0.1066 | 0.0592 |
| single_subtask_omission | 1 | noise+action_within_snapshot | within_snapshot_centered | 86 | 32 | 0.2998 | 0.8187 | noise+action_within_snapshot | -0.0049 | -0.0165 | 0.0000 |
| single_subtask_omission | 1 | moe_action_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3039 | 0.8340 | noise+action_within_snapshot | -0.0090 | -0.0304 | -0.0041 |
| single_subtask_omission | 1 | moe_state_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3281 | 2.3571 | noise+action_within_snapshot | -0.0331 | -0.1122 | -0.0282 |
| single_subtask_omission | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3078 | 0.8927 | noise+action_within_snapshot | -0.0128 | -0.0435 | -0.0080 |
| single_subtask_omission | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3030 | 0.8301 | noise+action_within_snapshot | -0.0081 | -0.0273 | -0.0032 |
| single_subtask_omission | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3049 | 0.8792 | noise+action_within_snapshot | -0.0099 | -0.0336 | -0.0051 |
| single_subtask_omission | 3 | failure_rate_only | absolute | 86 | 32 | 0.2950 | 0.7958 | noise+action | 0.0000 | 0.0000 | -0.0436 |
| single_subtask_omission | 3 | noise+action | absolute | 86 | 32 | 0.2514 | 0.9365 | noise+action | 0.0436 | 0.1478 | 0.0000 |
| single_subtask_omission | 3 | moe_action | absolute | 86 | 32 | 0.2121 | 1.0917 | noise+action | 0.0828 | 0.2808 | 0.0392 |
| single_subtask_omission | 3 | moe_state | absolute | 86 | 32 | 0.2915 | 0.8501 | noise+action | 0.0034 | 0.0116 | -0.0402 |
| single_subtask_omission | 3 | moe_action+state | absolute | 86 | 32 | 0.2237 | 1.7848 | noise+action | 0.0713 | 0.2418 | 0.0277 |
| single_subtask_omission | 3 | noise+action+moe_action | absolute | 86 | 32 | 0.2006 | 1.0792 | noise+action | 0.0944 | 0.3199 | 0.0508 |
| single_subtask_omission | 3 | noise+action+moe_action+state | absolute | 86 | 32 | 0.2166 | 1.6954 | noise+action | 0.0784 | 0.2658 | 0.0348 |
| single_subtask_omission | 3 | noise+action_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3317 | 0.8893 | noise+action_within_snapshot | -0.0368 | -0.1246 | 0.0000 |
| single_subtask_omission | 3 | moe_action_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3316 | 0.8928 | noise+action_within_snapshot | -0.0367 | -0.1243 | 0.0001 |
| single_subtask_omission | 3 | moe_state_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3842 | 1.8749 | noise+action_within_snapshot | -0.0892 | -0.3024 | -0.0524 |
| single_subtask_omission | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3400 | 0.9165 | noise+action_within_snapshot | -0.0450 | -0.1526 | -0.0083 |
| single_subtask_omission | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3327 | 0.8940 | noise+action_within_snapshot | -0.0377 | -0.1278 | -0.0009 |
| single_subtask_omission | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 32 | 0.3418 | 0.9188 | noise+action_within_snapshot | -0.0468 | -0.1586 | -0.0100 |
| non_stagnation_non_loop | 1 | failure_rate_only | absolute | 86 | 10 | 0.1043 | 0.3675 | noise+action | 0.0000 | 0.0000 | 0.0118 |
| non_stagnation_non_loop | 1 | noise+action | absolute | 86 | 10 | 0.1161 | 0.5227 | noise+action | -0.0118 | -0.1127 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action | absolute | 86 | 10 | 0.1322 | 0.4509 | noise+action | -0.0279 | -0.2675 | -0.0162 |
| non_stagnation_non_loop | 1 | moe_state | absolute | 86 | 10 | 0.1372 | 0.5059 | noise+action | -0.0329 | -0.3153 | -0.0211 |
| non_stagnation_non_loop | 1 | moe_action+state | absolute | 86 | 10 | 0.1230 | 0.4763 | noise+action | -0.0187 | -0.1793 | -0.0070 |
| non_stagnation_non_loop | 1 | noise+action+moe_action | absolute | 86 | 10 | 0.1341 | 0.4564 | noise+action | -0.0298 | -0.2857 | -0.0181 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state | absolute | 86 | 10 | 0.1332 | 0.5027 | noise+action | -0.0289 | -0.2770 | -0.0171 |
| non_stagnation_non_loop | 1 | noise+action_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1113 | 0.4245 | noise+action_within_snapshot | -0.0070 | -0.0667 | 0.0000 |
| non_stagnation_non_loop | 1 | moe_action_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1188 | 0.5717 | noise+action_within_snapshot | -0.0145 | -0.1388 | -0.0075 |
| non_stagnation_non_loop | 1 | moe_state_within_snapshot | within_snapshot_centered | 86 | 10 | 0.3204 | 2.8148 | noise+action_within_snapshot | -0.2161 | -2.0715 | -0.2091 |
| non_stagnation_non_loop | 1 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1270 | 0.8325 | noise+action_within_snapshot | -0.0226 | -0.2171 | -0.0157 |
| non_stagnation_non_loop | 1 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1187 | 0.5634 | noise+action_within_snapshot | -0.0144 | -0.1381 | -0.0075 |
| non_stagnation_non_loop | 1 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1271 | 0.7809 | noise+action_within_snapshot | -0.0228 | -0.2186 | -0.0158 |
| non_stagnation_non_loop | 3 | failure_rate_only | absolute | 86 | 10 | 0.1043 | 0.3675 | noise+action | 0.0000 | 0.0000 | 0.0161 |
| non_stagnation_non_loop | 3 | noise+action | absolute | 86 | 10 | 0.1204 | 0.4812 | noise+action | -0.0161 | -0.1543 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action | absolute | 86 | 10 | 0.1087 | 0.4581 | noise+action | -0.0044 | -0.0423 | 0.0117 |
| non_stagnation_non_loop | 3 | moe_state | absolute | 86 | 10 | 0.2314 | 0.7855 | noise+action | -0.1271 | -1.2182 | -0.1110 |
| non_stagnation_non_loop | 3 | moe_action+state | absolute | 86 | 10 | 0.2383 | 0.7794 | noise+action | -0.1340 | -1.2844 | -0.1179 |
| non_stagnation_non_loop | 3 | noise+action+moe_action | absolute | 86 | 10 | 0.1137 | 0.4933 | noise+action | -0.0094 | -0.0903 | 0.0067 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state | absolute | 86 | 10 | 0.2418 | 0.8030 | noise+action | -0.1375 | -1.3176 | -0.1214 |
| non_stagnation_non_loop | 3 | noise+action_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1069 | 0.3878 | noise+action_within_snapshot | -0.0026 | -0.0245 | 0.0000 |
| non_stagnation_non_loop | 3 | moe_action_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1110 | 0.4750 | noise+action_within_snapshot | -0.0066 | -0.0637 | -0.0041 |
| non_stagnation_non_loop | 3 | moe_state_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1727 | 0.6724 | noise+action_within_snapshot | -0.0684 | -0.6553 | -0.0658 |
| non_stagnation_non_loop | 3 | moe_action+state_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1222 | 0.6060 | noise+action_within_snapshot | -0.0178 | -0.1710 | -0.0153 |
| non_stagnation_non_loop | 3 | noise+action+moe_action_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1106 | 0.4732 | noise+action_within_snapshot | -0.0063 | -0.0602 | -0.0037 |
| non_stagnation_non_loop | 3 | noise+action+moe_action+state_within_snapshot | within_snapshot_centered | 86 | 10 | 0.1221 | 0.5930 | noise+action_within_snapshot | -0.0178 | -0.1708 | -0.0153 |

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | token_family | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loop_or_cycling | 1 | state | back_12_15 | 4 | top1_mass | 0.4706 | 0.9412 | -0.4706 | -0.6923 | -0.3000 | 0.0050 | 0.0995 | 17 | 17 |
| single_subtask_omission | 1 | state | back_12_15 | 6 | top1_mass | 0.6250 | 0.2500 | 0.3750 | 0.1579 | 0.6667 | 0.0050 | 0.2836 | 16 | 16 |
| single_subtask_omission | 1 | state | back_12_15 | 3 | consensus_distance | 0.6250 | 0.2500 | 0.3750 | 0.1500 | 0.7226 | 0.0149 | 0.2836 | 16 | 16 |
| loop_or_cycling | 3 | action | front_2_5 | 0 | top1_mass | 0.3529 | 0.7647 | -0.4118 | -0.5714 | -0.2778 | 0.0199 | 0.2985 | 17 | 17 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 0 | top1_mass | 0.3529 | 0.0588 | 0.2941 | 0.1176 | 0.5714 | 0.0100 | 0.3582 | 17 | 17 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 2 | top1_mass | 0.2941 | 0.0000 | 0.2941 | 0.1176 | 0.5714 | 0.0149 | 0.3582 | 17 | 17 |
| non_stagnation_non_loop | 3 | action | front_2_5 | 1 | top1_mass | 0.2941 | 0.0000 | 0.2941 | 0.1176 | 0.5714 | 0.0199 | 0.3582 | 17 | 17 |
| loop_or_cycling | 1 | state | front_2_5 | 4 | entropy | 0.4118 | 0.8235 | -0.4118 | -0.6397 | -0.1875 | 0.0050 | 0.4030 | 17 | 17 |
| loop_or_cycling | 1 | state | front_2_5 | 9 | consensus_distance | 0.4706 | 0.8824 | -0.4118 | -0.6923 | -0.1538 | 0.0100 | 0.4030 | 17 | 17 |
| non_stagnation_non_loop | 1 | state | back_12_15 | 8 | top1_mass | 0.0000 | 0.2941 | -0.2941 | -0.5385 | -0.1176 | 0.0050 | 0.4726 | 17 | 17 |
| non_stagnation_non_loop | 1 | action | front_2_5 | 0 | top1_mass | 0.2941 | 0.0000 | 0.2941 | 0.2500 | 0.3571 | 0.0100 | 0.4726 | 17 | 17 |
| non_stagnation_non_loop | 1 | action | front_2_5 | 1 | top1_mass | 0.2941 | 0.0000 | 0.2941 | 0.2500 | 0.3571 | 0.0149 | 0.4726 | 17 | 17 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.
- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
