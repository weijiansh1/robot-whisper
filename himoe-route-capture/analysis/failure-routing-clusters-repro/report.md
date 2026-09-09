# Normalized middle/late routing clusters of all failed rollouts

Run class: `formal`.

The statistical unit is one failed rollout. Each rollout is independently
mapped from relative phase 0.5 to 1.0 onto ten fixed anchors; absolute query
index and episode length are absent from every clustering representation.

## Coverage

| task | failures | failure lengths | checkpoint |
|---|---:|---|---|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 0 | none | `98ee29d09d18` |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 42 | 30 | `98ee29d09d18` |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 52 | `cdc2b21f9ef6` |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 22 | `1029d0827030` |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 22 | `1029d0827030` |

Total: 307 failed rollouts and 13570 recorded queries. There are 3 distinct checkpoints.

## How many clusters?

`selected K` is reported only when all frozen stability gates pass. When it
does not, `exploratory K` is the best size-valid silhouette partition.

| view | role | cluster status | interpretation | selected K | exploratory K | mean subsample silhouette | median ARI | ARI p10 | PAC | sizes | task NMI | mode NMI excess |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| geometry | primary invariant | stable_partition | task_shadowed+representation_specific | 2 | 2 | 0.619 | 1.000 | 1.000 | 0.000 | 49/258 | 0.655 | -0.000 |
| change | primary invariant | no_stable_partition | exploratory_only+task_shadowed+representation_specific | NA | 2 | 0.371 | 0.920 | 0.411 | 0.102 | 16/291 | 0.034 | -0.001 |
| recurrence | primary invariant | stable_partition | task_shadowed+representation_specific+phase_sensitive | 6 | 3 | 0.315 | 0.833 | 0.726 | 0.105 | 64/26/37/42/89/49 | 0.621 | 0.020 |
| expert_occupancy | secondary ID-based | stable_partition | task_shadowed+secondary_expert_id_view | 4 | 4 | 0.264 | 0.914 | 0.822 | 0.131 | 127/37/40/103 | 0.696 | 0.020 |

## Cross-view agreement

ARI compares each view's reported partition; low values mean there is no single taxonomy.

| view | geometry | change | recurrence | expert_occupancy |
|---|---:|---:|---:|---:|
| geometry | 1.000 | -0.063 | 0.160 | 0.208 |
| change | -0.063 | 1.000 | -0.016 | 0.004 |
| recurrence | 0.160 | -0.016 | 1.000 | 0.599 |
| expert_occupancy | 0.208 | 0.004 | 0.599 | 1.000 |

## Post-cluster associations

These variables were revealed only after all route-only labels were frozen.
The failure-mode p-value uses task x initial-state permutations.

| view | factor | NMI | null NMI | excess | permutation p | BH q | permutation |
|---|---|---:|---:|---:|---:|---:|---|
| geometry | task | 0.655 | 0.008 | 0.647 | 0.000 | 0.000 | global |
| geometry | checkpoint | 0.702 | 0.005 | 0.696 | 0.000 | 0.000 | global |
| geometry | episode_length | 0.702 | 0.005 | 0.696 | 0.000 | 0.000 | global |
| geometry | initial_state_within_task | 0.253 | 0.253 | -0.000 | 1.000 | 1.000 | within_task |
| geometry | flow_seed_within_task_initial_state | 0.187 | 0.187 | 0.000 | 1.000 | 1.000 | within_task_initial_state |
| geometry | failure_mode_proxy | 0.405 | 0.405 | -0.000 | 1.000 | 1.000 | within_task_initial_state |
| change | task | 0.034 | 0.010 | 0.024 | 0.009 | 0.013 | global |
| change | checkpoint | 0.037 | 0.007 | 0.030 | 0.006 | 0.010 | global |
| change | episode_length | 0.037 | 0.007 | 0.030 | 0.006 | 0.010 | global |
| change | initial_state_within_task | 0.044 | 0.025 | 0.019 | 0.000 | 0.001 | within_task |
| change | flow_seed_within_task_initial_state | 0.030 | 0.031 | -0.001 | 0.658 | 0.789 | within_task_initial_state |
| change | failure_mode_proxy | 0.031 | 0.032 | -0.001 | 0.892 | 1.000 | within_task_initial_state |
| recurrence | task | 0.621 | 0.020 | 0.601 | 0.000 | 0.000 | global |
| recurrence | checkpoint | 0.643 | 0.013 | 0.630 | 0.000 | 0.000 | global |
| recurrence | episode_length | 0.643 | 0.013 | 0.630 | 0.000 | 0.000 | global |
| recurrence | initial_state_within_task | 0.510 | 0.371 | 0.139 | 0.000 | 0.000 | within_task |
| recurrence | flow_seed_within_task_initial_state | 0.327 | 0.319 | 0.009 | 0.064 | 0.085 | within_task_initial_state |
| recurrence | failure_mode_proxy | 0.503 | 0.483 | 0.020 | 0.000 | 0.000 | within_task_initial_state |
| expert_occupancy | task | 0.696 | 0.014 | 0.681 | 0.000 | 0.000 | global |
| expert_occupancy | checkpoint | 0.640 | 0.010 | 0.630 | 0.000 | 0.000 | global |
| expert_occupancy | episode_length | 0.640 | 0.010 | 0.630 | 0.000 | 0.000 | global |
| expert_occupancy | initial_state_within_task | 0.450 | 0.364 | 0.086 | 0.000 | 0.000 | within_task |
| expert_occupancy | flow_seed_within_task_initial_state | 0.295 | 0.295 | 0.000 | 0.453 | 0.572 | within_task_initial_state |
| expert_occupancy | failure_mode_proxy | 0.547 | 0.528 | 0.020 | 0.030 | 0.043 | within_task_initial_state |

## Cluster composition

### geometry (task_shadowed+representation_specific, reported K=2)

| cluster | n | dominant task fraction | task x init groups | task counts | length counts | checkpoints | failure-mode proxy counts | mean speed | late speed | drift | backward return |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| C0 | 258 | 0.837 | 21 | open_the_top_drawer_and_put_the_bowl_inside=42; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=216 | 30=42; 52=216 | 98ee29d0=42; cdc2b21f=216 | lifted_not_placed=36; misplaced=12; partial=198; reached_then_lost=12 | 0.012 | 0.011 | 0.020 | 0.254 |
| C1 | 49 | 0.755 | 20 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=12; pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=37 | 22=49 | 1029d082=49 | lifted_not_placed=6; misplaced=13; never_grasped=29; reached_then_lost=1 | 0.019 | 0.013 | 0.041 | 0.005 |
### change (exploratory_only+task_shadowed+representation_specific, reported K=2)

| cluster | n | dominant task fraction | task x init groups | task counts | length counts | checkpoints | failure-mode proxy counts | mean speed | late speed | drift | backward return |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| C0 | 291 | 0.687 | 40 | open_the_top_drawer_and_put_the_bowl_inside=42; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=200; pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=12; pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=37 | 22=49; 30=42; 52=200 | 1029d082=49; 98ee29d0=42; cdc2b21f=200 | lifted_not_placed=42; misplaced=25; never_grasped=29; partial=183; reached_then_lost=12 | 0.013 | 0.011 | 0.024 | 0.202 |
| C1 | 16 | 1.000 | 8 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=16 | 52=16 | cdc2b21f=16 | partial=15; reached_then_lost=1 | 0.020 | 0.020 | 0.018 | 0.445 |
### recurrence (task_shadowed+representation_specific+phase_sensitive, reported K=6)

| cluster | n | dominant task fraction | task x init groups | task counts | length counts | checkpoints | failure-mode proxy counts | mean speed | late speed | drift | backward return |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| C0 | 64 | 1.000 | 7 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=64 | 52=64 | cdc2b21f=64 | partial=64 | 0.007 | 0.006 | 0.017 | 0.260 |
| C1 | 37 | 1.000 | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=37 | 52=37 | cdc2b21f=37 | partial=37 | 0.008 | 0.006 | 0.020 | 0.206 |
| C2 | 42 | 1.000 | 7 | open_the_top_drawer_and_put_the_bowl_inside=42 | 30=42 | 98ee29d0=42 | lifted_not_placed=36; misplaced=6 | 0.011 | 0.008 | 0.021 | 0.045 |
| C3 | 26 | 1.000 | 9 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=26 | 52=26 | cdc2b21f=26 | misplaced=1; partial=25 | 0.012 | 0.016 | 0.023 | 0.240 |
| C4 | 89 | 1.000 | 13 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=89 | 52=89 | cdc2b21f=89 | misplaced=5; partial=72; reached_then_lost=12 | 0.018 | 0.018 | 0.022 | 0.374 |
| C5 | 49 | 0.755 | 20 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=12; pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=37 | 22=49 | 1029d082=49 | lifted_not_placed=6; misplaced=13; never_grasped=29; reached_then_lost=1 | 0.019 | 0.013 | 0.041 | 0.005 |
### expert_occupancy (task_shadowed+secondary_expert_id_view, reported K=4)

| cluster | n | dominant task fraction | task x init groups | task counts | length counts | checkpoints | failure-mode proxy counts | mean speed | late speed | drift | backward return |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|
| C0 | 127 | 1.000 | 9 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=127 | 52=127 | cdc2b21f=127 | misplaced=1; partial=125; reached_then_lost=1 | 0.008 | 0.008 | 0.019 | 0.240 |
| C1 | 40 | 1.000 | 7 | open_the_top_drawer_and_put_the_bowl_inside=40 | 30=40 | 98ee29d0=40 | lifted_not_placed=34; misplaced=6 | 0.010 | 0.007 | 0.021 | 0.034 |
| C2 | 103 | 0.864 | 22 | open_the_top_drawer_and_put_the_bowl_inside=2; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=89; pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=12 | 22=12; 30=2; 52=89 | 1029d082=12; 98ee29d0=2; cdc2b21f=89 | lifted_not_placed=3; misplaced=15; partial=73; reached_then_lost=12 | 0.018 | 0.017 | 0.024 | 0.330 |
| C3 | 37 | 1.000 | 12 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=37 | 22=37 | 1029d082=37 | lifted_not_placed=5; misplaced=3; never_grasped=29 | 0.019 | 0.013 | 0.041 | 0.000 |

## Relative-phase sensitivity

Each primary view is re-fit at its reported K after changing only the
relative phase window/grid.

| view | configuration | fixed K | ARI to primary | silhouette | sizes | min-size pass | own best size-valid K |
|---|---|---:|---:|---:|---|---|---:|
| geometry | grid8 | 2 | 1.000 | 0.618 | 49/258 | yes | 2 |
| geometry | grid12 | 2 | 1.000 | 0.613 | 49/258 | yes | 2 |
| geometry | late60 | 2 | 1.000 | 0.581 | 49/258 | yes | 2 |
| geometry | truncate90 | 2 | 1.000 | 0.621 | 49/258 | yes | 2 |
| change | grid8 | 2 | 0.684 | 0.348 | 25/282 | yes | 2 |
| change | grid12 | 2 | 0.785 | 0.343 | 16/291 | yes | 4 |
| change | late60 | 2 | 0.696 | 0.372 | 27/280 | yes | 4 |
| change | truncate90 | 2 | 0.602 | 0.340 | 32/275 | yes | 3 |
| recurrence | grid8 | 6 | 0.837 | 0.325 | 49/31/70/88/43/26 | yes | 2 |
| recurrence | grid12 | 6 | 0.712 | 0.322 | 41/71/49/40/32/74 | yes | 3 |
| recurrence | late60 | 6 | 0.788 | 0.461 | 91/15/34/101/43/23 | no | 5 |
| recurrence | truncate90 | 6 | 0.592 | 0.312 | 74/46/25/24/73/65 | yes | 3 |

## Interpretation limits

- Phase normalization removes absolute chunk position from the representation. Lengths 22/30/52 remain perfectly nested within task here, so horizon and task effects are not separately identifiable.
- Geometry/change/recurrence are expert-permutation invariant. Expert occupancy is a checkpoint-shadow diagnostic only.
- Failure-mode labels are external kinematic proxies loaded after clustering, not ground truth and not cluster inputs.
- Stability is conditional on ordinary episode subsampling; it does not establish generalization to held-out tasks, initial-state groups, or seeds.
- Rows marked exploratory_only failed the frozen stability gate; their labels and profiles are descriptive rather than a supported taxonomy.
- A stable routing regime is an association, not evidence that routing caused the failure.
