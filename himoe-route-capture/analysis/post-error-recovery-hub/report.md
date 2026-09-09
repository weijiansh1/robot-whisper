# Post-loss proxy eventual-outcome feasibility: Gate 0

Transport targets are frozen from the BDDL goals before outcomes are read.
The landmark is the first query after a conservative kinematic transport-loss
proxy. The proxy is not a dense-contact slip annotation.

## Event inventory

| task | events | terminal success after proxy | terminal failure after proxy |
|---|---:|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 0 | 0 | 0 |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 22 | 0 | 22 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 2 | 0 | 2 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 2 | 1 | 1 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 7 | 1 | 6 |

Total: 33 events (2 later terminal successes, 31 later terminal failures), 2 mixed task x initial-state clusters, 2 conditional pairs.

## Gate decision

**FAIL.** The event cohort lacks independent within-initial-state outcome support.

| requirement | observed/pass |
|---|---:|
| tasks_with_both_outcomes | false |
| mixed_task_initial_state_clusters | false |
| mixed_task_initial_state_phase_clusters | false |
| terminal_success_after_proxy_events | false |
| terminal_failure_after_proxy_events | false |
| minimum_followup_queries | true |
| all_double_holdout_folds_evaluable | false |

The physical -> +route -> +action -> +hidden classifier was not fit.
With only 2 later terminal successes, an AUC or p-value would be dominated by task, initial state, and phase rather than post-error adaptation.

## Phase and follow-up diagnostics

- Mixed task x initial-state x 5-query phase clusters: 2.
- Conditional pairs within those phase clusters: 2.
- Minimum remaining queries after the landmark: 6.
- Absolute univariate AUC for query phase alone: 0.976.

## Descriptive recurrence

Values below compare the post-error query with the chunk that caused
the detected loss. They are descriptive and carry no inferential claim.

| outcome | n | action-route Hellinger similarity | action top-4 Jaccard | action cosine | hidden-action cosine |
|---|---:|---:|---:|---:|---:|
| terminal_success_after_proxy | 2 | 0.935 | 0.113 | -0.035 | 0.556 |
| terminal_failure_after_proxy | 31 | 0.967 | 0.147 | 0.852 | 0.758 |

## Threshold sensitivity

All strict/loose variants are joint 20% perturbations unless marked hold-distance-only.

| rule | events | success | failure | mixed clusters | identity Jaccard | shifted query |
|---|---:|---:|---:|---:|---:|---:|
| joint_strict | 31 | 6 | 25 | 2 | 0.730 | 0 |
| main | 33 | 2 | 31 | 2 | 1.000 | 0 |
| joint_loose | 37 | 2 | 35 | 2 | 0.892 | 0 |
| hold_distance_strict_only | 34 | 6 | 28 | 2 | 0.811 | 0 |
| hold_distance_loose_only | 35 | 2 | 33 | 2 | 0.943 | 0 |

## Causal boundary

- `drop_query` is after the first detected physical loss and before the next emitted chunk.
- Recurrence at that query cannot predict the loss that already occurred.
- It could prognose later recovery only after a larger, contact-validated cohort is collected.
- The binary label is eventual task outcome, not verified regrasp, second slip, or trap.
- `hb_hidden` is router-input contextual hidden state, not expert contribution.
