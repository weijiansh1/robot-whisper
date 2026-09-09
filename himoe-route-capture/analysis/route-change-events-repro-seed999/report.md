# Routing change-event organization

Each episode is represented by event counts, strengths, timings, and
cross-layer agreement for speed peaks, distance-matrix change points,
nonlocal returns, and low-speed runs. Raw phase traces, task, length, and
outcome are not clustering features.

## Stability

Status: `stable_partition`; reported K=4; sizes=[214, 1643, 205, 498].

| K | min size | silhouette | ARI med/p10 | PAC | stable |
|---:|---:|---:|---:|---:|:---:|
| 2 | 703 | 0.317 | 0.815 / 0.792 | 0.170 | true |
| 3 | 214 | 0.350 | 0.829 / 0.800 | 0.163 | true |
| 4 | 205 | 0.368 | 0.859 / 0.833 | 0.128 | true |
| 5 | 205 | 0.331 | 0.866 / 0.840 | 0.090 | true |
| 6 | 50 | 0.330 | 0.856 / 0.678 | 0.174 | false |
| 7 | 50 | 0.319 | 0.849 / 0.679 | 0.168 | false |
| 8 | 50 | 0.317 | 0.927 / 0.905 | 0.082 | false |

## Cluster profiles

| C | n | failure rate | peak count | change | joint return | return strength | stasis run | late stasis | layer sync | tasks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 214 | 0.977 | 0.110 | 0.665 | 0.078 | 0.099 | 0.073 | 0.850 | 0.936 | open_the_top_drawer_and_put_the_bowl_inside=39; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=129; pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=12; pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=34 |
| 1 | 1643 | 0.014 | 0.176 | 0.605 | 0.005 | 0.008 | 0.064 | 0.004 | 0.615 | open_the_middle_drawer_of_the_cabinet=512; open_the_top_drawer_and_put_the_bowl_inside=134; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=19; pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=500; pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=478 |
| 2 | 205 | 0.117 | 0.031 | 0.295 | 0.116 | 0.257 | 0.088 | 0.000 | 0.953 | open_the_top_drawer_and_put_the_bowl_inside=173; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=32 |
| 3 | 498 | 0.102 | 0.037 | 0.122 | 0.104 | 0.105 | 0.004 | 0.000 | 0.638 | open_the_top_drawer_and_put_the_bowl_inside=166; KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=332 |

## Confounding

| variable | NMI | null NMI | excess | Cramer's V | p |
|---|---:|---:|---:|---:|---:|
| task | 0.382 | 0.002 | 0.380 | 0.552 | 0.001 |
| checkpoint | 0.339 | 0.001 | 0.338 | 0.559 | 0.001 |
| episode_length | 0.464 | 0.010 | 0.454 | 0.853 | 0.001 |
| outcome_within_task_initial_state | 0.315 | 0.125 | 0.190 | 0.806 | 0.001 |

Event features predict task with accuracy 0.932 (majority 0.200) and episode length with R2 0.862 / MAE 3.404 queries.

ARI against the phase-pair recurrence K=6 partition: 0.242.

## Outcome increment

Predictions use 16 task-initial-state x flow-seed double-holdout folds.

| model | ROC AUC | average precision | log loss | Brier |
|---|---:|---:|---:|---:|
| task | 0.765 | 0.278 | 0.303 | 0.089 |
| task_length | 1.000 | 0.988 | 0.044 | 0.010 |
| task_events | 0.991 | 0.957 | 0.072 | 0.018 |
| task_length_events | 0.997 | 0.986 | 0.036 | 0.008 |

- `events_over_task`: AUC increment 0.226 (cluster-bootstrap 95% CI 0.126 to 0.355); AP increment 0.679 (CI 0.510 to 0.790).

- `events_over_task_length`: AUC increment -0.002 (cluster-bootstrap 95% CI -0.005 to 0.000); AP increment -0.002 (CI -0.024 to 0.022).

## Limits

- Events are derived from ten normalized phase anchors, not dense physical contact labels.
- A return is a nonlocal route-distance event; low-speed plateaus can satisfy it without a true control loop.
- Successful phase 1.0 is completion, while failed phase 1.0 is timeout. This is descriptive, not causal forecasting.
- Task and length are excluded from clustering but can remain recoverable from route dynamics.
