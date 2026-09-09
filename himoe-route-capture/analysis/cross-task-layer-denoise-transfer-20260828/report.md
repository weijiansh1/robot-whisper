# Cross-task layer and denoise routing transfer

## Direct result

No aggregate routing feature survives the task-wide max-T scan in any validation task at the first policy query or in the strict pre-anchor window. Scene8-selected directions also conflict across validation tasks, and selected expert probabilities do not transfer robustly between the two tasks sharing the Spatial checkpoint.

The strongest validation-task effects occur after the interaction anchor. In the stove task, selected effects are small through the anchor and grow at relative queries +2 to +4. In the ramekin task, the clearest divergence also starts at +2. This timing supports concurrent physical-state encoding rather than a generic early failure code.

## Cohorts

Every task contributes all 512 rollouts. `preanchor4` contains the last four decisions ending at the physical anchor; `phase5` contains five queries starting at that anchor. No rollout is removed because it ends early. Effects are failure minus success within initial state, measured in within-init standard deviations.

| short_task | successes | failures | mixed_initial_states | anchor_min | anchor_median | anchor_max | anchor_effect_sigma | anchor_permutation_p | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_moka_pots | 296 | 216 | 13 | 15 | 18 | 22 | 0.1132 | 0.3938 | cdc2b21f9ef6 |
| top_drawer_then_bowl | 470 | 42 | 7 | 6 | 6 | 7 | -0.0974 | 1 | 98ee29d09d18 |
| bowl_from_ramekin | 500 | 12 | 8 | 3 | 4 | 7 | -0.2388 | 0.2874 | 1029d0827030 |
| bowl_from_stove | 475 | 37 | 12 | 3 | 4 | 5 | 0.1324 | 0.6122 | 1029d0827030 |

Expert IDs are not compared across the Long, Goal, and Spatial checkpoints. The two spatial tasks share one checkpoint, so their expert indices are directly comparable.

## Frozen Scene8 aggregate transfer

The rows below were selected on Scene8 or fixed by the previous active-return analysis before examining validation-task effects. `match` means the failure-success direction agrees; Bonferroni p corrects the fixed validation family.

| selection | task | window | family | location | metric | effect_sigma | match | validation_bonferroni_p | loso_effect_min | loso_effect_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene8_preanchor4_top_per_family | top_drawer_then_bowl | preanchor4 | state_layer_summary | L13 | entropy_mean | -0.04231 | yes | 1 | -0.2478 | 0.1887 |
| scene8_preanchor4_top_per_family | bowl_from_ramekin | preanchor4 | state_layer_summary | L13 | entropy_mean | -0.4314 | yes | 1 | -0.6712 | -0.1126 |
| scene8_preanchor4_top_per_family | bowl_from_stove | preanchor4 | state_layer_summary | L13 | entropy_mean | 0.02578 | no | 1 | -0.1122 | 0.1219 |
| scene8_preanchor4_top_per_family | top_drawer_then_bowl | preanchor4 | state_group_summary | back_12_15 | top1_mean | 0.599 | yes | 0.8321 | 0.3576 | 0.7689 |
| scene8_preanchor4_top_per_family | bowl_from_ramekin | preanchor4 | state_group_summary | back_12_15 | top1_mean | 0.01956 | yes | 1 | -0.1599 | 0.1366 |
| scene8_preanchor4_top_per_family | bowl_from_stove | preanchor4 | state_group_summary | back_12_15 | top1_mean | 0.1863 | yes | 1 | 0.1097 | 0.264 |
| scene8_preanchor4_top_per_family | top_drawer_then_bowl | preanchor4 | action_layer_denoise | L14/d2 | denoise_delta_mean | -0.4239 | no | 1 | -0.4785 | -0.3328 |
| scene8_preanchor4_top_per_family | bowl_from_ramekin | preanchor4 | action_layer_denoise | L14/d2 | denoise_delta_mean | -0.04602 | no | 1 | -0.1845 | 0.1249 |
| scene8_preanchor4_top_per_family | bowl_from_stove | preanchor4 | action_layer_denoise | L14/d2 | denoise_delta_mean | -0.1014 | no | 1 | -0.147 | 0.01713 |
| scene8_preanchor4_top_per_family | top_drawer_then_bowl | preanchor4 | action_group_denoise | back_12_15/d2 | denoise_delta_mean | -0.2394 | no | 1 | -0.2808 | -0.2029 |
| scene8_preanchor4_top_per_family | bowl_from_ramekin | preanchor4 | action_group_denoise | back_12_15/d2 | denoise_delta_mean | 0.03323 | yes | 1 | -0.1621 | 0.3269 |
| scene8_preanchor4_top_per_family | bowl_from_stove | preanchor4 | action_group_denoise | back_12_15/d2 | denoise_delta_mean | 0.2263 | yes | 1 | 0.1405 | 0.3125 |
| scene8_phase5_top_per_family | top_drawer_then_bowl | phase5 | state_layer_summary | L4 | entropy_mean | -0.2062 | no | 1 | -0.4076 | -0.1046 |
| scene8_phase5_top_per_family | bowl_from_ramekin | phase5 | state_layer_summary | L4 | entropy_mean | 1.106 | yes | 0.04498 | 0.796 | 1.35 |
| scene8_phase5_top_per_family | bowl_from_stove | phase5 | state_layer_summary | L4 | entropy_mean | -0.2556 | no | 1 | -0.479 | -0.1768 |
| scene8_phase5_top_per_family | top_drawer_then_bowl | phase5 | state_group_summary | front_2_5 | top1_mean | 0.2262 | no | 1 | -0.05347 | 0.4363 |
| scene8_phase5_top_per_family | bowl_from_ramekin | phase5 | state_group_summary | front_2_5 | top1_mean | 0.1256 | no | 1 | -0.3019 | 0.6178 |
| scene8_phase5_top_per_family | bowl_from_stove | phase5 | state_group_summary | front_2_5 | top1_mean | -0.4867 | yes | 0.5172 | -0.6662 | -0.1577 |
| scene8_phase5_top_per_family | top_drawer_then_bowl | phase5 | action_layer_denoise | L15/d0 | entropy_mean | 0.1125 | yes | 1 | 0.05814 | 0.2199 |
| scene8_phase5_top_per_family | bowl_from_ramekin | phase5 | action_layer_denoise | L15/d0 | entropy_mean | -0.3449 | no | 1 | -0.5235 | -0.1684 |
| scene8_phase5_top_per_family | bowl_from_stove | phase5 | action_layer_denoise | L15/d0 | entropy_mean | -3.311 | no | 0.02249 | -3.648 | -2.875 |
| scene8_phase5_top_per_family | top_drawer_then_bowl | phase5 | action_group_denoise | back_12_15/d0 | entropy_std_query | -0.1815 | yes | 1 | -0.3923 | 0.1162 |
| scene8_phase5_top_per_family | bowl_from_ramekin | phase5 | action_group_denoise | back_12_15/d0 | entropy_std_query | 0.6219 | no | 1 | 0.06825 | 1.098 |
| scene8_phase5_top_per_family | bowl_from_stove | phase5 | action_group_denoise | back_12_15/d0 | entropy_std_query | -0.8701 | yes | 0.02249 | -1.168 | -0.5067 |
| scene8_query0_top_per_family | top_drawer_then_bowl | query0 | q0_action_layer_denoise | L14/d0 | top1_mean | -0.3764 | yes | 1 | -0.4496 | -0.2002 |
| scene8_query0_top_per_family | bowl_from_ramekin | query0 | q0_action_layer_denoise | L14/d0 | top1_mean | 0.8387 | no | 0.1799 | 0.6547 | 1.044 |
| scene8_query0_top_per_family | bowl_from_stove | query0 | q0_action_layer_denoise | L14/d0 | top1_mean | -0.05868 | yes | 1 | -0.1248 | 0.01394 |
| scene8_query0_top_per_family | top_drawer_then_bowl | query0 | q0_action_group_denoise | back_12_15/d7 | p4p5_gap_mean | 0.2705 | no | 1 | 0.2272 | 0.3526 |
| scene8_query0_top_per_family | bowl_from_ramekin | query0 | q0_action_group_denoise | back_12_15/d7 | p4p5_gap_mean | -0.1301 | yes | 1 | -0.2376 | 0.02589 |
| scene8_query0_top_per_family | bowl_from_stove | query0 | q0_action_group_denoise | back_12_15/d7 | p4p5_gap_mean | 0.05151 | no | 1 | -0.04434 | 0.1525 |
| prior_active_return_9q | top_drawer_then_bowl | phase5 | state_layer_summary | L4 | entropy_std_query | -0.08645 | yes | 1 | -0.201 | 0.08515 |
| prior_active_return_9q | bowl_from_ramekin | phase5 | state_layer_summary | L4 | entropy_std_query | -1.18 | yes | 0.02249 | -1.38 | -0.885 |
| prior_active_return_9q | bowl_from_stove | phase5 | state_layer_summary | L4 | entropy_std_query | 1.143 | no | 0.02249 | 0.9851 | 1.254 |
| prior_active_return_9q | top_drawer_then_bowl | phase5 | state_layer_summary | L14 | query_speed_mean | 0.09368 | no | 1 | -0.1059 | 0.1799 |
| prior_active_return_9q | bowl_from_ramekin | phase5 | state_layer_summary | L14 | query_speed_mean | -0.8597 | yes | 0.2924 | -1.218 | -0.7255 |
| prior_active_return_9q | bowl_from_stove | phase5 | state_layer_summary | L14 | query_speed_mean | -0.05246 | yes | 1 | -0.3151 | 0.1628 |
| prior_active_return_9q | top_drawer_then_bowl | phase5 | action_group_denoise | back_12_15/d2 | entropy_std_query | -0.3556 | yes | 1 | -0.5972 | -0.04636 |
| prior_active_return_9q | bowl_from_ramekin | phase5 | action_group_denoise | back_12_15/d2 | entropy_std_query | 0.68 | no | 1 | 0.1106 | 1.069 |
| prior_active_return_9q | bowl_from_stove | phase5 | action_group_denoise | back_12_15/d2 | entropy_std_query | -1.227 | yes | 0.02249 | -1.604 | -0.7287 |
| prior_active_return_9q | top_drawer_then_bowl | phase5 | action_group_denoise | back_12_15/d9 | top1_std_query | -0.3698 | yes | 1 | -0.5502 | -0.06714 |
| prior_active_return_9q | bowl_from_ramekin | phase5 | action_group_denoise | back_12_15/d9 | top1_std_query | -0.01362 | yes | 1 | -0.2962 | 0.147 |
| prior_active_return_9q | bowl_from_stove | phase5 | action_group_denoise | back_12_15/d9 | top1_std_query | 0.02796 | no | 1 | -0.7094 | 0.2107 |
| prior_active_return_9q | top_drawer_then_bowl | phase5 | action_layer_denoise | L15/d0 | entropy_mean | 0.1125 | yes | 1 | 0.05814 | 0.2199 |
| prior_active_return_9q | bowl_from_ramekin | phase5 | action_layer_denoise | L15/d0 | entropy_mean | -0.3449 | no | 1 | -0.5235 | -0.1684 |
| prior_active_return_9q | bowl_from_stove | phase5 | action_layer_denoise | L15/d0 | entropy_mean | -3.311 | no | 0.02249 | -3.648 | -2.875 |

## Spatial-checkpoint expert transfer

| selection | task | window | family | location | effect_sigma | match | validation_bonferroni_p | loso_effect_min | loso_effect_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stove_preanchor4_expert_top | bowl_from_ramekin | preanchor4 | state_expert_probability | L15/e24 | -0.1222 | yes | 1 | -0.2844 | 0.001695 |
| stove_preanchor4_expert_top | bowl_from_ramekin | preanchor4 | action_expert_probability | L13/d6/e24 | 0.5293 | no | 0.4398 | 0.2932 | 0.7501 |
| stove_phase5_expert_top | bowl_from_ramekin | phase5 | state_expert_probability | L13/e31 | 0.3663 | yes | 0.9845 | 0.07052 | 0.6223 |
| stove_phase5_expert_top | bowl_from_ramekin | phase5 | action_expert_probability | L2/d8/e9 | 0.3738 | yes | 0.8946 | 0.1173 | 0.638 |
| stove_query0_expert_top | bowl_from_ramekin | query0 | q0_action_expert_probability | L5/d9/e28 | 0.2893 | no | 1 | -0.07632 | 0.4603 |

## Temporal localization

These rows decompose already-selected phase effects; they are descriptive, not a second confirmatory search. Query 0 is the anchor decision before its action is executed. For query-speed profiles, the relative-query index names the later endpoint of the transition.

| task | profile | pre_or_anchor_query | pre_or_anchor_effect_sigma | post_query | post_effect_sigma | post_loso_min | post_loso_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top_drawer_then_bowl | state_L14_E21_probability | 0 | 0.584 | 3 | 0.3872 | 0.1898 | 0.5167 |
| bowl_from_ramekin | state_L4_top1 | 0 | 0.2402 | 2 | -0.8643 | -1.076 | -0.6382 |
| bowl_from_ramekin | action_L13_d9_query_speed | -1 | 0.2313 | 2 | -1.462 | -1.717 | -0.655 |
| bowl_from_ramekin | action_L15_d3_E21_probability | 0 | -0.5166 | 2 | -1.604 | -1.828 | -1.198 |
| bowl_from_ramekin | action_L15_d3_E7_probability | -1 | -0.5265 | 3 | 1.862 | 1.603 | 2.127 |
| bowl_from_stove | state_L2_query_speed | -2 | 0.3648 | 3 | -0.9845 | -1.164 | -0.5296 |
| bowl_from_stove | action_L15_d0_top1 | -1 | -0.2872 | 4 | 4.574 | 4.254 | 5.138 |
| bowl_from_stove | action_back_d7_top1 | -3 | -0.3073 | 4 | 4.28 | 3.824 | 4.87 |
| bowl_from_stove | action_L2_d8_E9_probability | -2 | -0.2668 | 4 | 4.145 | 3.872 | 4.568 |

## Task-specific exploratory peaks

These are selected separately inside each task and are not transfer evidence. Family/global max-T p-values account for scanning the displayed feature family/all features in that task-window.

| task | window | family | location | metric | effect_sigma | permutation_p_max_family | permutation_p_max_global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowl_from_ramekin | phase5 | action_group_denoise | back_12_15/d9 | query_speed_mean | -1.264 | 0.005997 | 0.08146 |
| bowl_from_ramekin | phase5 | action_layer_denoise | L13/d9 | query_speed_mean | -1.352 | 0.01199 | 0.03448 |
| bowl_from_ramekin | phase5 | state_group_summary | back_12_15 | query_speed_mean | -0.5795 | 0.3438 | 1 |
| bowl_from_ramekin | phase5 | state_layer_summary | L4 | top1_std_query | -1.322 | 0.001499 | 0.04548 |
| bowl_from_stove | phase5 | action_group_denoise | back_12_15/d7 | top1_mean | 3.728 | 0.0004998 | 0.0004998 |
| bowl_from_stove | phase5 | action_layer_denoise | L15/d0 | top1_mean | 3.584 | 0.0004998 | 0.0004998 |
| bowl_from_stove | phase5 | state_group_summary | front_2_5 | query_speed_mean | -1.268 | 0.0004998 | 0.0004998 |
| bowl_from_stove | phase5 | state_layer_summary | L2 | query_speed_mean | -1.578 | 0.0004998 | 0.0004998 |
| top_drawer_then_bowl | phase5 | action_group_denoise | back_12_15/d6 | entropy_std_query | -0.6025 | 0.5112 | 0.996 |
| top_drawer_then_bowl | phase5 | action_layer_denoise | L14/d5 | top1_std_query | -0.8705 | 0.1039 | 0.2819 |
| top_drawer_then_bowl | phase5 | state_group_summary | back_12_15 | entropy_mean | 0.5576 | 0.1379 | 0.9995 |
| top_drawer_then_bowl | phase5 | state_layer_summary | L12 | top1_std_query | -0.7953 | 0.03248 | 0.5567 |
| two_moka_pots | phase5 | action_group_denoise | back_12_15/d0 | entropy_std_query | -0.5296 | 0.005497 | 0.06397 |
| two_moka_pots | phase5 | action_layer_denoise | L15/d0 | entropy_mean | 0.6159 | 0.0009995 | 0.004998 |
| two_moka_pots | phase5 | state_group_summary | front_2_5 | top1_mean | -0.7482 | 0.0004998 | 0.0004998 |
| two_moka_pots | phase5 | state_layer_summary | L4 | entropy_mean | 0.7454 | 0.0004998 | 0.0004998 |
| bowl_from_ramekin | preanchor4 | action_group_denoise | back_12_15/d1 | top1_std_query | 1.01 | 0.04098 | 0.4353 |
| bowl_from_ramekin | preanchor4 | action_layer_denoise | L15/d2 | top1_std_query | 0.901 | 0.5077 | 0.8176 |
| bowl_from_ramekin | preanchor4 | state_group_summary | front_2_5 | p4p5_gap_mean | 0.342 | 0.8496 | 1 |
| bowl_from_ramekin | preanchor4 | state_layer_summary | L13 | entropy_mean | -0.4314 | 0.9615 | 1 |
| bowl_from_stove | preanchor4 | action_group_denoise | back_12_15/d1 | query_speed_mean | -0.4702 | 0.6622 | 1 |
| bowl_from_stove | preanchor4 | action_layer_denoise | L13/d3 | top1_std_query | 0.6864 | 0.1534 | 0.5117 |
| bowl_from_stove | preanchor4 | state_group_summary | back_12_15 | p4p5_gap_mean | 0.5017 | 0.09745 | 0.9985 |
| bowl_from_stove | preanchor4 | state_layer_summary | L13 | p4p5_gap_mean | 0.5314 | 0.2234 | 0.996 |
| top_drawer_then_bowl | preanchor4 | action_group_denoise | back_12_15/d9 | entropy_mean | 0.7238 | 0.1679 | 0.8171 |
| top_drawer_then_bowl | preanchor4 | action_layer_denoise | L4/d5 | top1_mean | -0.9752 | 0.02099 | 0.07496 |
| top_drawer_then_bowl | preanchor4 | state_group_summary | back_12_15 | top1_std_query | 0.6192 | 0.06897 | 0.993 |
| top_drawer_then_bowl | preanchor4 | state_layer_summary | L12 | top1_std_query | 0.693 | 0.06997 | 0.903 |
| two_moka_pots | preanchor4 | action_group_denoise | back_12_15/d2 | denoise_delta_mean | 0.4865 | 0.01099 | 0.1634 |
| two_moka_pots | preanchor4 | action_layer_denoise | L14/d2 | denoise_delta_mean | 0.631 | 0.0004998 | 0.002499 |
| two_moka_pots | preanchor4 | state_group_summary | back_12_15 | top1_mean | 0.6109 | 0.0004998 | 0.004998 |
| two_moka_pots | preanchor4 | state_layer_summary | L13 | entropy_mean | -0.7409 | 0.0004998 | 0.0004998 |
| bowl_from_ramekin | query0 | q0_action_group_denoise | back_12_15/d2 | entropy_mean | -0.7478 | 0.3823 | 0.992 |
| bowl_from_ramekin | query0 | q0_action_layer_denoise | L12/d0 | p4p5_gap_mean | 1.09 | 0.05597 | 0.2294 |
| bowl_from_stove | query0 | q0_action_group_denoise | back_12_15/d4 | p4p5_gap_mean | -0.4237 | 0.6112 | 1 |
| bowl_from_stove | query0 | q0_action_layer_denoise | L15/d8 | top1_mean | 0.435 | 0.965 | 0.9995 |
| top_drawer_then_bowl | query0 | q0_action_group_denoise | front_2_5/d5 | p4p5_gap_mean | 0.745 | 0.07746 | 0.7051 |
| top_drawer_then_bowl | query0 | q0_action_layer_denoise | L5/d3 | p4p5_gap_mean | 0.7186 | 0.3853 | 0.7916 |
| two_moka_pots | query0 | q0_action_group_denoise | back_12_15/d7 | p4p5_gap_mean | -0.2612 | 0.7096 | 1 |
| two_moka_pots | query0 | q0_action_layer_denoise | L14/d0 | top1_mean | -0.3802 | 0.3488 | 0.7616 |

## Integrity

State-token probability maximum difference over denoise steps: `0.00000000`.
Query-0 state-token maximum within-init probability span: `0.00000000`.
Query-0 therefore tests action routing only. The analysis uses soft routing probabilities rather than argmax expert labels, and uses no remaining time, episode length, or future-event timing as a feature.

The ramekin task has only 12 failures, so its effect estimates are visibly less stable. All findings remain conditional on these four tasks and three checkpoint families.
