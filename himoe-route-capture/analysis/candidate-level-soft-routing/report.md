# Query-0 candidate-level soft-routing audit

## Result

This is a candidate-level closing experiment, not a rollout clustering result. Every comparison is among the 32 flow-noise candidates of one task and one initial state at query 0. Complete initial-state pools are held out from fitting. Because features are centered using the observed K=32 pool, the estimand is transductive ranking within a complete candidate pool, not inductive prediction for one candidate in isolation.

Directly, action-token soft-routing distance and normalized action-chunk distance have mean within-pool Spearman 0.175 (cluster-bootstrap 95% CI 0.167 to 0.182). This is association, not proof that routing causes the action difference: routing and action can both be downstream of the flow seed and action hidden state.

The cross-fitted route decoder explains 0.605 of centered action-chunk variance. For x0-to-x1 EEF response, action explains 0.999, route alone 0.626, and the blockwise joint action-plus-route model 0.998. Thus routing clearly tracks which action candidate was sampled. The action block retains all 70 standardized chunk coordinates; route receives a separate 16-PC training-fold reduction. The joint model therefore retains exactly the full action representation used by action-only and adds the route subspace. Its response increment is -0.000369 (95% CI -0.000508 to -0.000277), so routing adds no useful prediction of this observed response once the complete action chunk is known.

For final success, route-only conditional AUC is 0.400 and the hidden comparator is 0.450. The learned score direction is not a usable selector on held-out initial states; the route AUC below 0.5 is reverse transfer, not proof that the representation contains no separable outcome information. This analysis is retrospective and conditional because mixed pools are selected using final outcomes. It therefore closes only this prespecified decoder as a quality selector. Action-plus-route versus full action changes AUC by +0.030 (95% CI -0.010 to +0.070; approximate MDE80 0.057). Broader success informativeness remains unresolved, and no additional endpoint search is triggered.

The first-chunk EEF endpoint is retained as a short-horizon physical-response endpoint. It is not labelled task quality. EEF-to-object approach is additionally available only for the three object-first tasks. Object lift, drawer displacement, and task-progress changes are reported below as identifiability audits because they are nearly or exactly degenerate over the first 0.5 seconds.

## Endpoint identifiability

| task | successes | mixed_outcome_initial_states | action_within_init_coordinate_sd | eef_step_within_init_sd_m | approach_within_init_sd_m | lift_within_init_sd_m | task_progress_within_init_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| middle drawer | 512 | 0 | 0.04117 | 0.0005253 |  |  | 0 |
| top drawer + bowl | 470 | 7 | 0.03936 | 0.0006281 |  |  | 0 |
| two moka pots | 296 | 13 | 0.0484 | 0.0007761 | 0.0006786 | 0 | 0 |
| bowl on ramekin | 500 | 8 | 0.09186 | 0.001412 | 0.001515 | 0 | 0 |
| bowl on stove | 475 | 12 | 0.09496 | 0.00158 | 0.001738 | 0 | 0 |

The query-0 robot state and full simulator state are bit-identical within every task-by-init pool (maximum difference 0), and x1 is the next policy boundary 0.5 seconds after x0. RGB frames were not archived in these NPZ files, so there is no retrospective pixel hash. Equality is established for cached simulator and proprio state, not independently for pixels.

## Cross-fitted estimates

| endpoint | metric | model | estimate | ci95_low | ci95_high | tasks | initial_states |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_chunk | pairwise | action_soft_route | 0.5685 | 0.5609 | 0.5753 | 5 | 80 |
| action_chunk | pairwise | hidden_upper | 0.7078 | 0.6969 | 0.7185 | 5 | 80 |
| action_chunk | pairwise | seed_sentinel | 0.3826 | 0.3626 | 0.4032 | 5 | 80 |
| action_chunk | pairwise | state_route_control |  |  |  | 5 | 80 |
| action_chunk | r2 | action_soft_route | 0.6047 | 0.5976 | 0.6102 | 5 | 80 |
| action_chunk | r2 | hidden_upper | 0.5742 | 0.5639 | 0.5838 | 5 | 80 |
| action_chunk | r2 | seed_sentinel | 0.4821 | 0.4723 | 0.4916 | 5 | 80 |
| action_chunk | r2 | state_route_control | 8.513e-09 | -3.669e-10 | 1.758e-08 | 5 | 80 |
| eef_delta_0p5s | r2 | action | 0.9985 | 0.9981 | 0.9988 | 5 | 80 |
| eef_delta_0p5s | r2 | action_plus_route | 0.9981 | 0.9976 | 0.9985 | 5 | 80 |
| eef_delta_0p5s | r2 | action_soft_route | 0.6261 | 0.6203 | 0.6312 | 5 | 80 |
| eef_delta_0p5s | r2 | hidden_upper | 0.8267 | 0.8176 | 0.8353 | 5 | 80 |
| eef_delta_0p5s | r2 | seed_sentinel | 0.5106 | 0.494 | 0.5277 | 5 | 80 |
| eef_delta_0p5s | r2 | state_route_control | 1.188e-08 | 3.256e-09 | 2.037e-08 | 5 | 80 |
| eef_target_approach_0p5s | r2 | action | 0.9941 | 0.9923 | 0.9956 | 3 | 48 |
| eef_target_approach_0p5s | r2 | action_plus_route | 0.9938 | 0.992 | 0.9954 | 3 | 48 |
| eef_target_approach_0p5s | r2 | action_soft_route | 0.647 | 0.6353 | 0.6583 | 3 | 48 |
| eef_target_approach_0p5s | r2 | hidden_upper | 0.8359 | 0.8202 | 0.8504 | 3 | 48 |
| eef_target_approach_0p5s | r2 | seed_sentinel | 0.5223 | 0.4875 | 0.5577 | 3 | 48 |
| eef_target_approach_0p5s | r2 | state_route_control | 4.118e-09 | -7.621e-09 | 1.547e-08 | 3 | 48 |
| eef_target_approach_0p5s | rank | action | 0.9941 | 0.9929 | 0.9951 | 3 | 48 |
| eef_target_approach_0p5s | rank | action_plus_route | 0.9942 | 0.9929 | 0.9953 | 3 | 48 |
| eef_target_approach_0p5s | rank | action_soft_route | 0.7838 | 0.7771 | 0.7902 | 3 | 48 |
| eef_target_approach_0p5s | rank | hidden_upper | 0.9045 | 0.8955 | 0.9134 | 3 | 48 |
| eef_target_approach_0p5s | rank | seed_sentinel | 0.7167 | 0.6944 | 0.7398 | 3 | 48 |
| eef_target_approach_0p5s | rank | state_route_control |  |  |  | 3 | 48 |
| final_success_mixed_pools | auc | action | 0.3334 | 0.2764 | 0.3894 | 4 | 40 |
| final_success_mixed_pools | auc | action_plus_route | 0.3632 | 0.2981 | 0.4265 | 4 | 40 |
| final_success_mixed_pools | auc | action_soft_route | 0.3998 | 0.3268 | 0.4681 | 4 | 40 |
| final_success_mixed_pools | auc | hidden_upper | 0.4499 | 0.3782 | 0.5219 | 4 | 40 |
| final_success_mixed_pools | auc | seed_sentinel | 0.4312 | 0.3689 | 0.4929 | 4 | 40 |
| final_success_mixed_pools | auc | state_route_control | 0.5 | 0.5 | 0.5 | 4 | 40 |

All learned feature scaling and PCA are fitted inside the training fold. The action block retains all 70 standardized coordinates; every other block has at most 16 components. Eight grouped folds are used when possible. The action target and physical targets are centered within init, so R2 is relative to predicting the held-out pool mean. Pairwise/rank metrics do not depend on target centering. Feature centering uses the complete observed K=32 pool and therefore defines the transductive scope stated above.

## Model contrasts and resolution

| endpoint | metric | larger_model | smaller_model | estimate | ci95_low | ci95_high | mde_80pct_two_sided_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_chunk | r2 | action_soft_route | seed_sentinel | 0.1225 | 0.1136 | 0.1306 | 0.0122 |
| action_chunk | r2 | action_soft_route | state_route_control | 0.6047 | 0.5976 | 0.6102 | 0.009111 |
| action_chunk | pairwise_spearman | action_soft_route | seed_sentinel | 0.1859 | 0.1644 | 0.2071 | 0.03011 |
| eef_delta_0p5s | r2 | action | state_route_control | 0.9985 | 0.9981 | 0.9988 | 0.0005394 |
| eef_delta_0p5s | r2 | action_soft_route | state_route_control | 0.6261 | 0.6203 | 0.6312 | 0.00782 |
| eef_delta_0p5s | r2 | action_plus_route | action | -0.0003691 | -0.0005078 | -0.0002774 | 0.0001689 |
| eef_target_approach_0p5s | r2 | action_plus_route | action | -0.0002762 | -0.0006904 | 2.688e-05 | 0.0005169 |
| eef_target_approach_0p5s | within_init_rank | action_plus_route | action | 0.0001756 | -0.0002367 | 0.0006035 | 0.0005928 |
| final_success_mixed_pools | conditional_auc | action_soft_route | state_route_control | -0.1002 | -0.1732 | -0.03195 | 0.1002 |
| final_success_mixed_pools | conditional_auc | action_plus_route | action | 0.02978 | -0.009948 | 0.0703 | 0.05747 |

The reported MDE is 2.801585 times the paired cluster-bootstrap standard deviation. Both intervals and MDE resample the frozen OOF prediction table; they do not refit folds and therefore omit learning-procedure uncertainty from overlapping CV training sets. They describe conditional resolution of this OOF table, not prospective study power and not an observed effect.

## Final success, secondary endpoint

Final success retrospectively uses only the 40 initial states containing both success and failure (1280 candidates; success rate 0.785). It is deliberately secondary: labels occur far after query 0, fixed all-success or all-failure pools contain no within-init candidate signal, and eligibility cannot be known at query 0. These AUCs are not a deployable prospective screen.

## Controls and interpretation

- seed_sentinel tests whether a flow-noise seed identity generalizes across held-out initial states.
- state_route_control is the state-token route negative control. It is candidate-invariant at query 0.
- action is the direct sampled action chunk and is the positive control for x0-to-x1 response.
- hidden_upper is retained as an artifact key, but the feature is a lossy RMS/moment hidden summary, not a true hidden-state upper bound.
- action_plus_route retains the same full 70-coordinate action block as action-only and adds a separately fitted 16-PC route block. Its held-init score difference is an additive generalization test, not a causal effect.

No hard expert IDs are used. Bootstrap resampling is by initial state within each fixed task and all reported predictions are frozen out-of-fold predictions.
