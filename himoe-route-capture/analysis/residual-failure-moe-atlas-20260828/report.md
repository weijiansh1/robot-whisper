# Residual-failure MoE atlas

## Direct result

The repeated response is overwhelmingly persistence rather than a literal loop: 3/307 failures pass the late route-cycle rule, 1 also pass the physical-state rule, and 1 passes the full route+state+action rule. The independent physical oscillation label is 0/307.

The fixed-point family is strongest for stagnation, but it is not the whole signal. Regrasp/drop, goal regression, active return, and the previously unruled residual cohort peak in different confidence, denoising, token-disagreement, or state/action-mismatch families. The cross-fitted coverage table below quantifies what those extra families add under both failure-specific and dual failure/success control thresholds, never an episode-timeout threshold.

## Protocol

Features use only relative phase 0.5 through 0.9. The 1224-dimensional formal bank is expert-permutation invariant. It contains the 516 frozen front/back summaries of state/action confidence, action-token dispersion, state/action gap, within-query denoising motion and expert retention, adjacent-query persistence and nonlocal recurrence, plus 708 raw-route features that resolve action routing at d0 through d9 and each adjacent denoise transition. Episode length, remaining time, the final 10% of the rollout, physical state, action values, and labels are absent from the feature matrix.

All formal effects compare failures with other failures inside task x initial-state strata. Therefore every comparison is also within one task-specific failure horizon. Effects are target-minus-control differences in within-stratum residual standard deviations. Primary-label p-values use max-T over all 1224 features and then correct across the five prespecified modes.

State-token routing is identical over all ten denoise forwards (maximum raw probability difference 0), so a state d_i scan would only duplicate the same feature ten times. The per-step extension therefore resolves action-token routing only. d0 noisiest at flow time 1.0; d9 last recorded forward at 0.1.

The primary coverage experiment permits nominal calibration for a known initial state, but avoids reusing its evaluation controls: a deterministic half of held-out-state successes sets its shrunken success reference, while the other half alone estimates held-out success triggering. A second zero-reference sensitivity run uses raw routing features and all held-out successes, so it has no held-out-state calibration at all.

## Loop control

| failures | taxonomy_eef_oscillation | late_route_cycle_candidates | late_route_plus_state_cycle_candidates | late_joint_route_state_action_loop_candidates |
| --- | --- | --- | --- | --- |
| 307 | 0 | 3 | 1 | 1 |

## Mode support

| mode | primary | positive_total | positives_in_mixed_strata | controls_in_mixed_strata | mixed_strata |
| --- | --- | --- | --- | --- | --- |
| stagnation | True | 166 | 109 | 86 | 13 |
| regrasp_or_drop | True | 58 | 37 | 86 | 8 |
| goal_regression | True | 49 | 34 | 69 | 6 |
| residual_none | True | 88 | 71 | 136 | 15 |
| active_return | True | 39 | 39 | 127 | 9 |
| gripper_cycling | False | 9 | 5 | 88 | 5 |
| subtask_undo | False | 8 | 8 | 65 | 4 |

## Strongest invariant signature per primary mode

| mode | family | feature | effect_sigma | permutation_p_max_all_features | primary_label_fwer_p | loso_effect_min | loso_effect_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| active_return | state_confidence | state_entropy|front_2_5|denoise_mean|phase_late | -4.188 | 0.0004998 | 0.002499 | -4.747 | -3.878 |
| goal_regression | denoise_soft_motion | action_denoise_motion_at_i|front_2_5|d8_to_d9|phase_std | 1.337 | 0.007996 | 0.03998 | 1.049 | 2.214 |
| regrasp_or_drop | denoise_soft_motion | action_denoise_motion|back_12_15|denoise_final|phase_late | 1.514 | 0.005497 | 0.02749 | 1.216 | 1.787 |
| residual_none | state_action_mismatch | state_action_gap_at_d|front_2_5|d4|phase_mean | 2.413 | 0.0004998 | 0.002499 | 2.206 | 2.68 |
| stagnation | query_persistence | hard_route|back_12_15|adjacent_mean | -3.313 | 0.0004998 | 0.002499 | -3.861 | -3 |

## Strongest denoise-step localization per primary mode

These rows are selected only from features that preserve an individual action-routing denoise step or transition. Their p-values still use the joint formal max-T scan, not a smaller post-hoc search.

| mode | family | feature | effect_sigma | permutation_p_max_all_features | primary_label_fwer_p |
| --- | --- | --- | --- | --- | --- |
| active_return | state_action_mismatch | state_action_gap_at_d|front_2_5|d9|phase_slope | 3.537 | 0.0004998 | 0.002499 |
| goal_regression | denoise_soft_motion | action_denoise_motion_at_i|front_2_5|d8_to_d9|phase_std | 1.337 | 0.007996 | 0.03998 |
| regrasp_or_drop | denoise_soft_motion | action_denoise_motion_at_i|front_2_5|d8_to_d9|query_change | 1.38 | 0.01649 | 0.08246 |
| residual_none | state_action_mismatch | state_action_gap_at_d|front_2_5|d4|phase_mean | 2.413 | 0.0004998 | 0.002499 |
| stagnation | action_confidence | action_top1_at_d|front_2_5|d7|phase_mean | -2.725 | 0.0004998 | 0.002499 |

A peak alone can be misleading, so the next table decomposes the selected metric across every denoise step. `peak_to_median_abs` near one means a trajectory-wide offset; a large value means genuine step localization.

| mode | metric | layer_group | phase_statistic | peak_location | peak_effect_sigma | effect_min | effect_max | peak_to_median_abs | direction_consistent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| active_return | state_action_gap_at_d | front_2_5 | phase_slope | d9 | 3.537 | 3.469 | 3.537 | 1.017 | True |
| goal_regression | action_denoise_motion_at_i | front_2_5 | phase_std | d8_to_d9 | 1.337 | 0.1497 | 1.337 | 4.454 | True |
| regrasp_or_drop | action_denoise_motion_at_i | front_2_5 | query_change | d8_to_d9 | 1.38 | 0.1584 | 1.38 | 1.981 | True |
| residual_none | state_action_gap_at_d | front_2_5 | phase_mean | d4 | 2.413 | 2.404 | 2.413 | 1.001 | True |
| stagnation | action_top1_at_d | front_2_5 | phase_mean | d7 | -2.725 | -2.725 | -1.764 | 1.261 | True |

## Cross-fitted coverage

For each held-out task x initial-state stratum, features and directions are selected only on the other strata. This detector table deliberately stays on the frozen 516-feature all-outcome cache; the raw per-step extension is used for localization, not silently added without matching success controls. `other_failure_q90` uses the training 90th percentile of other-failure scores. The stricter `dual_control_q90` uses the larger of the other-failure and success 90th percentiles. `net_coverage` is target coverage minus held-out other-failure trigger rate. Success rates below use only the disjoint held-out evaluation half.

| mode | block | calibration | target_n | target_coverage | other_failure_trigger_rate | net_coverage | net_coverage_ci_low | net_coverage_ci_high | success_trigger_rate | operational_net_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | trap_only | other_failure_q90 | 109 | 0.8349 | 0.2907 | 0.5442 | 0.3314 | 0.7141 | 0.1574 | 0.5442 |
| stagnation | trap_only | dual_control_q90 | 109 | 0.8073 | 0.1512 | 0.6562 | 0.4119 | 0.8396 | 0.05556 | 0.6562 |
| stagnation | trap_plus_denoise | other_failure_q90 | 109 | 0.8716 | 0.1628 | 0.7088 | 0.4658 | 0.8731 | 0.463 | 0.4086 |
| stagnation | trap_plus_denoise | dual_control_q90 | 109 | 0.7615 | 0.02326 | 0.7382 | 0.485 | 0.8906 | 0.09259 | 0.6689 |
| stagnation | nontrap_only | other_failure_q90 | 109 | 0.8532 | 0.1279 | 0.7253 | 0.4845 | 0.8802 | 0.9167 | -0.06346 |
| stagnation | nontrap_only | dual_control_q90 | 109 | 0.7523 | 0.02326 | 0.729 | 0.4711 | 0.8878 | 0.07407 | 0.6782 |
| stagnation | full_invariant | other_failure_q90 | 109 | 0.8532 | 0.1279 | 0.7253 | 0.4786 | 0.8889 | 0.9074 | -0.0542 |
| stagnation | full_invariant | dual_control_q90 | 109 | 0.7615 | 0.02326 | 0.7382 | 0.4711 | 0.8913 | 0.06481 | 0.6967 |
| regrasp_or_drop | trap_only | other_failure_q90 | 37 | 0.1892 | 0.2674 | -0.07825 | -0.3836 | 0.4986 | 0.3438 | -0.1546 |
| regrasp_or_drop | trap_only | dual_control_q90 | 37 | 0.08108 | 0.1628 | -0.08171 | -0.255 | 0.2441 | 0.0625 | -0.08171 |
| regrasp_or_drop | trap_plus_denoise | other_failure_q90 | 37 | 0.1351 | 0.2442 | -0.1091 | -0.3836 | 0.2957 | 0.1406 | -0.1091 |
| regrasp_or_drop | trap_plus_denoise | dual_control_q90 | 37 | 0.1081 | 0.1279 | -0.0198 | -0.1924 | 0.3472 | 0.1094 | -0.0198 |
| regrasp_or_drop | nontrap_only | other_failure_q90 | 37 | 0.7568 | 0.1628 | 0.594 | 0.08865 | 0.7831 | 0.0625 | 0.594 |
| regrasp_or_drop | nontrap_only | dual_control_q90 | 37 | 0.7568 | 0.1628 | 0.594 | 0.07014 | 0.7793 | 0.0625 | 0.594 |
| regrasp_or_drop | full_invariant | other_failure_q90 | 37 | 0.2973 | 0.2093 | 0.08799 | -0.1556 | 0.4223 | 0.03125 | 0.08799 |
| regrasp_or_drop | full_invariant | dual_control_q90 | 37 | 0.2973 | 0.2093 | 0.08799 | -0.1524 | 0.4 | 0.03125 | 0.08799 |
| goal_regression | trap_only | other_failure_q90 | 34 | 0.08824 | 0.2899 | -0.2016 | -0.3487 | 0.3817 | 0.3333 | -0.2451 |
| goal_regression | trap_only | dual_control_q90 | 34 | 0.02941 | 0.1159 | -0.08653 | -0.2817 | 0.2281 | 0.02381 | -0.08653 |
| goal_regression | trap_plus_denoise | other_failure_q90 | 34 | 0.1765 | 0.2174 | -0.04092 | -0.206 | 0.5805 | 0.2619 | -0.08543 |
| goal_regression | trap_plus_denoise | dual_control_q90 | 34 | 0.1176 | 0.07246 | 0.04518 | -0.07143 | 0.6976 | 0.04762 | 0.04518 |
| goal_regression | nontrap_only | other_failure_q90 | 34 | 0.4706 | 0.3043 | 0.1662 | -0.1059 | 0.38 | 0.09524 | 0.1662 |
| goal_regression | nontrap_only | dual_control_q90 | 34 | 0.3824 | 0.2609 | 0.1215 | -0.172 | 0.3436 | 0.07143 | 0.1215 |
| goal_regression | full_invariant | other_failure_q90 | 34 | 0.2941 | 0.2464 | 0.04774 | -0.1262 | 0.6566 | 0.2619 | 0.03221 |
| goal_regression | full_invariant | dual_control_q90 | 34 | 0.1471 | 0.1159 | 0.03112 | -0.1264 | 0.4474 | 0.09524 | 0.03112 |
| residual_none | trap_only | other_failure_q90 | 71 | 0.1972 | 0.2794 | -0.08223 | -0.2552 | 0.102 | 0.1343 | -0.08223 |
| residual_none | trap_only | dual_control_q90 | 71 | 0.1831 | 0.2647 | -0.08161 | -0.2528 | 0.09466 | 0.04478 | -0.08161 |
| residual_none | trap_plus_denoise | other_failure_q90 | 71 | 0.2535 | 0.25 | 0.003521 | -0.1406 | 0.1596 | 0.0597 | 0.003521 |
| residual_none | trap_plus_denoise | dual_control_q90 | 71 | 0.2535 | 0.25 | 0.003521 | -0.1311 | 0.1757 | 0.0597 | 0.003521 |
| residual_none | nontrap_only | other_failure_q90 | 71 | 0.7324 | 0.1103 | 0.6221 | 0.3395 | 0.7771 | 0.06716 | 0.6221 |
| residual_none | nontrap_only | dual_control_q90 | 71 | 0.7324 | 0.1103 | 0.6221 | 0.3629 | 0.7799 | 0.05224 | 0.6221 |
| residual_none | full_invariant | other_failure_q90 | 71 | 0.7465 | 0.1029 | 0.6435 | 0.3946 | 0.7907 | 0.02985 | 0.6435 |
| residual_none | full_invariant | dual_control_q90 | 71 | 0.7465 | 0.1029 | 0.6435 | 0.3875 | 0.793 | 0.02985 | 0.6435 |
| active_return | trap_only | other_failure_q90 | 39 | 0.7949 | 0.1496 | 0.6453 | 0.4208 | 0.8377 | 0.7586 | 0.03625 |
| active_return | trap_only | dual_control_q90 | 39 | 0.1538 | 0 | 0.1538 | 0 | 0.3001 | 0.1034 | 0.0504 |
| active_return | trap_plus_denoise | other_failure_q90 | 39 | 0.7692 | 0.1024 | 0.6669 | 0.4556 | 0.8689 | 0.1207 | 0.6485 |
| active_return | trap_plus_denoise | dual_control_q90 | 39 | 0.7692 | 0.07874 | 0.6905 | 0.4865 | 0.9103 | 0.1207 | 0.6485 |
| active_return | nontrap_only | other_failure_q90 | 39 | 0.9487 | 0.1102 | 0.8385 | 0.7245 | 0.9149 | 0.2414 | 0.7073 |
| active_return | nontrap_only | dual_control_q90 | 39 | 0.9487 | 0.1024 | 0.8464 | 0.7435 | 0.9242 | 0.1034 | 0.8453 |
| active_return | full_invariant | other_failure_q90 | 39 | 0.9487 | 0.1102 | 0.8385 | 0.7363 | 0.9263 | 0.2759 | 0.6729 |
| active_return | full_invariant | dual_control_q90 | 39 | 0.9487 | 0.1024 | 0.8464 | 0.7394 | 0.9231 | 0.1034 | 0.8453 |

Full invariant block with dual controls only:

| mode | target_coverage | other_failure_trigger_rate | coverage_minus_other_failure | success_trigger_rate | operational_net_coverage |
| --- | --- | --- | --- | --- | --- |
| stagnation | 0.7615 | 0.02326 | 0.7382 | 0.06481 | 0.6967 |
| regrasp_or_drop | 0.2973 | 0.2093 | 0.08799 | 0.03125 | 0.08799 |
| goal_regression | 0.1471 | 0.1159 | 0.03112 | 0.09524 | 0.03112 |
| residual_none | 0.7465 | 0.1029 | 0.6435 | 0.02985 | 0.6435 |
| active_return | 0.9487 | 0.1024 | 0.8464 | 0.1034 | 0.8453 |

Union coverage counts an identifiable failed episode once when any detector for one of its true primary labels fires.

| block | calibration | identifiable_failure_episodes | covered_failure_episodes | coverage | supported_success_episodes | success_union_trigger_rate |
| --- | --- | --- | --- | --- | --- | --- |
| trap_only | other_failure_q90 | 212 | 138 | 0.6509 | 166 | 0.4819 |
| trap_plus_denoise | other_failure_q90 | 212 | 142 | 0.6698 | 166 | 0.4337 |
| nontrap_only | other_failure_q90 | 212 | 176 | 0.8302 | 166 | 0.6566 |
| full_invariant | other_failure_q90 | 212 | 173 | 0.816 | 166 | 0.6807 |
| trap_only | dual_control_q90 | 212 | 108 | 0.5094 | 166 | 0.1325 |
| trap_plus_denoise | dual_control_q90 | 212 | 133 | 0.6274 | 166 | 0.1687 |
| nontrap_only | dual_control_q90 | 212 | 174 | 0.8208 | 166 | 0.1325 |
| full_invariant | dual_control_q90 | 212 | 168 | 0.7925 | 166 | 0.1265 |

Under dual controls, the trap-only union covers 108/212 identifiable failures (50.9%) with a 13.3% union trigger rate on supported successes. The full invariant union covers 168/212 (79.2%) at 12.7%: a net 60 additional failures while aggregate success triggering remains essentially unchanged.

The requested trap-removal ablation is `nontrap_only`: both query-persistence and recurrence families are excluded. It still covers 174/212 failures (82.1%) with 13.3% success triggering.

Each block retrains a four-feature selector. Therefore `full_invariant` is not the set-theoretic union of the fitted nontrap and trap detectors: allowing trap families can displace a useful nontrap family from that four-feature budget. The nontrap block outperforming the full block is evidence of selector interference under fixed capacity, not evidence that adding a feature can intrinsically destroy information.

## Trap-removal overlap

Against trap-only, the nontrap detector overlap is: both 101, nontrap-only 73, trap-only 7, and neither 31.

| view | primary_physical_pattern | n |
| --- | --- | --- |
| added_by_nontrap | return_to_completed_pot2_proxy | 28 |
| added_by_nontrap | active_at_unfinished_pot1 | 25 |
| added_by_nontrap | bowl_transport_loss_proxy | 8 |
| added_by_nontrap | placed_object_goal_loss | 4 |
| added_by_nontrap | drawer_subgoal_regression | 3 |
| added_by_nontrap | other_long_incomplete | 3 |
| added_by_nontrap | ongoing_bowl_placement_failure | 1 |
| added_by_nontrap | ramekin_displacement_interference_proxy | 1 |
| missed_by_nontrap | bowl_never_transported | 19 |
| missed_by_nontrap | other_long_incomplete | 9 |
| missed_by_nontrap | drawer_subgoal_regression | 3 |
| missed_by_nontrap | placed_object_goal_loss | 3 |
| missed_by_nontrap | active_at_unfinished_pot1 | 2 |
| missed_by_nontrap | bowl_transport_loss_proxy | 2 |

## What the extra families add, and what remains

The episode-level overlap is: both blocks 100, full-only 68, trap-only 8, and neither 36. Thus the full block leaves 44/212 identifiable failures untriggered. Another 95/307 failures are not in this coverage denominator because their primary mode lacks within-initial-state positive/control support.

| view | primary_physical_pattern | n |
| --- | --- | --- |
| added_by_full | return_to_completed_pot2_proxy | 28 |
| added_by_full | active_at_unfinished_pot1 | 25 |
| added_by_full | bowl_transport_loss_proxy | 5 |
| added_by_full | other_long_incomplete | 3 |
| added_by_full | placed_object_goal_loss | 3 |
| added_by_full | drawer_subgoal_regression | 2 |
| added_by_full | ongoing_bowl_placement_failure | 1 |
| added_by_full | ramekin_displacement_interference_proxy | 1 |
| missed_by_full | bowl_never_transported | 19 |
| missed_by_full | other_long_incomplete | 9 |
| missed_by_full | bowl_transport_loss_proxy | 5 |
| missed_by_full | drawer_subgoal_regression | 5 |
| missed_by_full | placed_object_goal_loss | 4 |
| missed_by_full | active_at_unfinished_pot1 | 2 |

## Zero-reference sensitivity

This run removes even nominal success calibration from the held-out initial state. It is deliberately harsher: raw routing levels must transfer to an unseen state without a local baseline.

| block | identifiable_failure_episodes | covered_failure_episodes | coverage | supported_success_episodes | success_union_trigger_rate |
| --- | --- | --- | --- | --- | --- |
| trap_only | 212 | 37 | 0.1745 | 340 | 0.2059 |
| trap_plus_denoise | 212 | 120 | 0.566 | 340 | 0.1882 |
| nontrap_only | 212 | 149 | 0.7028 | 340 | 0.2147 |
| full_invariant | 212 | 160 | 0.7547 | 340 | 0.2559 |

| mode | target_coverage | other_failure_trigger_rate | success_trigger_rate | operational_net_coverage |
| --- | --- | --- | --- | --- |
| stagnation | 0.7523 | 0.1628 | 0.1357 | 0.5895 |
| regrasp_or_drop | 0.05405 | 0.04651 | 0.1429 | -0.0888 |
| goal_regression | 0.08824 | 0.05797 | 0.03371 | 0.03026 |
| residual_none | 0.7183 | 0.09559 | 0.08425 | 0.6227 |
| active_return | 0.9487 | 0.07087 | 0.1066 | 0.8422 |

## Task-specific physical modes

The table reports the strongest descriptive invariant feature only for physical patterns supported by at least three mixed initial states and eight positives in those states.

| task | physical_pattern | n | mixed_initial_states | cross_init_status | family | feature | effect_sigma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | active_at_unfinished_pot1 | 45 | 12 | estimable | query_recurrence | soft_denoise_spread|back_12_15|return_advantage_mean | 1.358 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | other_long_incomplete | 14 | 7 | estimable | denoise_soft_motion | action_denoise_motion_at_i|front_2_5|d6_to_d7|phase_late | 2.654 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | placed_object_goal_loss | 9 | 5 | estimable | denoise_soft_motion | action_denoise_motion_at_i|front_2_5|d8_to_d9|phase_std | 2.183 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | return_to_completed_pot2_proxy | 39 | 9 | estimable | state_confidence | state_entropy|front_2_5|denoise_mean|phase_late | -4.188 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | stalled_at_unfinished_pot1 | 109 | 9 | estimable | action_confidence | action_top4_at_d|back_12_15|d5|phase_std | -3.796 |

The full support audit is in `physical_mode_support.csv`. In particular, ramekin interference and stove never-transported are mostly initial-state-separated from their within-task failure controls; they remain strong outcome-vs-success observations in the earlier anchor analysis, but this dataset cannot establish a cross-initial-state mode-specific signature for them.

## Interpretation limits

- This is an observational state-reading experiment, not a causal intervention and not necessarily an early-warning experiment.
- Physical labels are kinematic proxies. Multi-label overlap is retained rather than forced into one mechanism.
- The primary cross-fit holds out all failures in an initial-state stratum but uses half of that stratum's successes for nominal calibration; the zero-reference run is the stricter unseen-state sensitivity. Both remain conditional on these tasks and checkpoints.
- A missing routing signature does not prove that the VLA lacks the information; expert hidden values, attention, visual tokens, and sampled action geometry are outside this routing-only bank.
