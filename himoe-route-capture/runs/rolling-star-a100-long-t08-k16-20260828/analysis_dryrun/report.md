# Rolling-star K=16 experiment

## Dataset

- 32 terminal branches from 2 committed snapshots; 7 success and 25 failure.
- 1 snapshots contain matched success/failure siblings.
- 1601 candidate query rows with full HB probabilities; hidden state stored: False.
- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.

## Physical failure labels

Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.

- `loop_or_cycling`: 16
- `single_subtask_omission`: 4
- `goal_contact_near_miss`: 2
- `subtask_undo`: 2
- `drop_or_regrasp`: 1

## Early MoE signal

No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.

The q0 state-token route negative-control span within matched siblings is 0.00111.
The q0 AS-MoE within-snapshot span is 0; a zero span confirms that AS is a constant negative control on this task.

Cross-validated models (snapshots held out together):

| prefix_queries | model | n_candidates | n_mixed_snapshots_for_selection | brier | log_loss | selected_success_rate | random_success_rate | selection_gain | selection_gain_ci95_low | selection_gain_ci95_high | brier_improvement_vs_rate | brier_relative_improvement_vs_rate | brier_gain_vs_rate | brier_gain_vs_rate_ci95_low | brier_gain_vs_rate_ci95_high | brier_gain_vs_noise_action | brier_gain_vs_noise_action_ci95_low | brier_gain_vs_noise_action_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | snapshot_rate_only | 32 | 1 | 0.3145 | 3.3098 | 0.4375 | 0.4375 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1906 | 0.0000 | 0.3812 |
| 1 | noise | 32 | 1 | 0.3163 | 3.3133 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0018 | -0.0057 | -0.0018 | -0.0036 | 0.0000 | 0.1888 | 0.0000 | 0.3776 |
| 1 | action | 32 | 1 | 0.5969 | 4.0420 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.2825 | -0.8983 | -0.2825 | -0.5650 | 0.0000 | -0.0919 | -0.1838 | 0.0000 |
| 1 | noise+action | 32 | 1 | 0.5051 | 3.7295 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.1906 | -0.6061 | -0.1906 | -0.3812 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | moe_action | 32 | 1 | 0.5701 | 4.0368 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.2556 | -0.8129 | -0.2556 | -0.5112 | 0.0000 | -0.0650 | -0.1300 | 0.0000 |
| 1 | moe_state | 32 | 1 | 0.7187 | 8.8484 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.4043 | -1.2857 | -0.4043 | -0.8086 | 0.0000 | -0.2137 | -0.4274 | 0.0000 |
| 1 | moe_action+state | 32 | 1 | 0.2191 | 3.0317 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | 0.0954 | 0.3033 | 0.0954 | 0.0000 | 0.1907 | 0.2860 | 0.0000 | 0.5719 |
| 1 | noise+action+moe_action | 32 | 1 | 0.5537 | 3.9487 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.2393 | -0.7609 | -0.2393 | -0.4785 | 0.0000 | -0.0487 | -0.0973 | 0.0000 |
| 1 | noise+action+moe_action+state | 32 | 1 | 0.2223 | 3.0574 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | 0.0922 | 0.2930 | 0.0922 | 0.0000 | 0.1843 | 0.2827 | 0.0000 | 0.5655 |
| 3 | snapshot_rate_only | 32 | 1 | 0.3145 | 3.3098 | 0.4375 | 0.4375 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0691 | 0.0000 | 0.1383 |
| 3 | noise | 32 | 1 | 0.3163 | 3.3135 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0018 | -0.0059 | -0.0018 | -0.0037 | 0.0000 | 0.0673 | 0.0000 | 0.1346 |
| 3 | action | 32 | 1 | 0.6855 | 4.7348 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.3711 | -1.1800 | -0.3711 | -0.7421 | 0.0000 | -0.3019 | -0.6039 | 0.0000 |
| 3 | noise+action | 32 | 1 | 0.3836 | 3.4491 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0691 | -0.2199 | -0.0691 | -0.1383 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | moe_action | 32 | 1 | 0.3240 | 3.3291 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0096 | -0.0304 | -0.0096 | -0.0191 | 0.0000 | 0.0596 | 0.0000 | 0.1192 |
| 3 | moe_state | 32 | 1 | 0.7187 | 9.9122 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.4043 | -1.2857 | -0.4043 | -0.8086 | 0.0000 | -0.3352 | -0.6703 | 0.0000 |
| 3 | moe_action+state | 32 | 1 | 0.3757 | 3.4331 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0613 | -0.1949 | -0.0613 | -0.1226 | 0.0000 | 0.0079 | 0.0000 | 0.0157 |
| 3 | noise+action+moe_action | 32 | 1 | 0.3247 | 3.3306 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0103 | -0.0326 | -0.0103 | -0.0205 | 0.0000 | 0.0589 | 0.0000 | 0.1178 |
| 3 | noise+action+moe_action+state | 32 | 1 | 0.3689 | 3.4193 | 0.0000 | 0.4375 | -0.4375 | -0.4375 | -0.4375 | -0.0545 | -0.1732 | -0.0545 | -0.1089 | 0.0000 | 0.0147 | 0.0000 | 0.0294 |

Strongest layer/denoise quartile contrasts (maxT corrects the full screen):

| prefix_queries | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | front_2_5 | 9 | token_dispersion | 1.0000 | 0.2500 | 0.7500 | 0.7500 | 0.7500 | 0.0392 | 0.8431 | 4 | 4 |
| 1 | back_12_15 | 8 | entropy | 0.2500 | 1.0000 | -0.7500 | -0.7500 | -0.7500 | 0.0588 | 0.8431 | 4 | 4 |
| 1 | front_2_5 | 9 | entropy | 0.0000 | 0.7500 | -0.7500 | -0.7500 | -0.7500 | 0.0784 | 0.8431 | 4 | 4 |
| 1 | back_12_15 | 6 | entropy | 0.2500 | 1.0000 | -0.7500 | -0.7500 | -0.7500 | 0.0980 | 0.8431 | 4 | 4 |
| 1 | back_12_15 | 7 | entropy | 0.2500 | 1.0000 | -0.7500 | -0.7500 | -0.7500 | 0.0980 | 0.8431 | 4 | 4 |
| 1 | back_12_15 | 9 | entropy | 0.2500 | 1.0000 | -0.7500 | -0.7500 | -0.7500 | 0.1176 | 0.8431 | 4 | 4 |
| 1 | front_2_5 | 0 | token_dispersion | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.1765 | 1.0000 | 4 | 4 |
| 1 | back_12_15 | 9 | consensus_distance | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.2353 | 1.0000 | 4 | 4 |
| 1 | back_12_15 | 9 | token_dispersion | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.2745 | 1.0000 | 4 | 4 |
| 1 | front_2_5 | 2 | top1_mass | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.2745 | 1.0000 | 4 | 4 |
| 1 | back_12_15 | 5 | entropy | 0.2500 | 0.7500 | -0.5000 | -0.5000 | -0.5000 | 0.2941 | 1.0000 | 4 | 4 |
| 1 | back_12_15 | 0 | entropy | 0.2500 | 0.7500 | -0.5000 | -0.5000 | -0.5000 | 0.3333 | 1.0000 | 4 | 4 |

Failure-subtype models, evaluated only among failed branches:

Insufficient support for subtype cross-validation.

Strongest subtype-specific route contrasts:

| failure_type | prefix_queries | layer_group | denoise_step | metric | top_quartile_failure_rate | bottom_quartile_failure_rate | failure_rate_difference | ci95_low | ci95_high | permutation_p_raw | permutation_p_maxT | n_top | n_bottom |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loop_or_cycling | 1 | front_2_5 | 8 | top1_mass | 0.5000 | 0.8333 | -0.3333 | -0.5000 | -0.2500 | 0.2157 | 0.9412 | 6 | 6 |
| loop_or_cycling | 3 | front_2_5 | 2 | top1_mass | 0.5000 | 0.8333 | -0.3333 | -0.5000 | -0.2500 | 0.1765 | 1.0000 | 6 | 6 |
| loop_or_cycling | 3 | front_2_5 | 9 | entropy | 0.6667 | 1.0000 | -0.3333 | -1.0000 | 0.0000 | 0.1765 | 1.0000 | 6 | 6 |
| loop_or_cycling | 3 | front_2_5 | 0 | top1_mass | 0.5000 | 0.8333 | -0.3333 | -0.5000 | -0.2500 | 0.1961 | 1.0000 | 6 | 6 |
| loop_or_cycling | 3 | front_2_5 | 1 | top1_mass | 0.5000 | 0.8333 | -0.3333 | -0.5000 | -0.2500 | 0.1961 | 1.0000 | 6 | 6 |
| loop_or_cycling | 3 | front_2_5 | 7 | top1_mass | 1.0000 | 0.6667 | 0.3333 | 0.0000 | 1.0000 | 0.2353 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | front_2_5 | 6 | top1_mass | 0.6667 | 0.8333 | -0.1667 | -0.2500 | 0.0000 | 0.5686 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | back_12_15 | 3 | token_dispersion | 0.6667 | 0.8333 | -0.1667 | -0.2500 | 0.0000 | 0.5882 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | front_2_5 | 0 | entropy | 0.8333 | 0.6667 | 0.1667 | 0.0000 | 0.2500 | 0.5882 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | front_2_5 | 1 | entropy | 0.8333 | 0.6667 | 0.1667 | 0.0000 | 0.2500 | 0.5882 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | front_2_5 | 2 | entropy | 0.8333 | 0.6667 | 0.1667 | 0.0000 | 0.2500 | 0.5882 | 1.0000 | 6 | 6 |
| loop_or_cycling | 1 | front_2_5 | 3 | entropy | 0.8333 | 0.6667 | 0.1667 | 0.0000 | 0.2500 | 0.5882 | 1.0000 | 6 | 6 |

## Interpretation guardrails

- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.
- Noise and first action chunks are explicit controls; `noise+action+moe_action+state` must beat `noise+action` before claiming incremental MoE information.
- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.
- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.
