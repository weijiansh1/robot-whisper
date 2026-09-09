# Aligned route-signature landmark experiment

Run class: `formal_exploratory`.

No block in the primary task-residual partition passed the frozen cross-task success/failure criterion.

This is an exploratory posthoc analysis. Task residualization is explicit;
clusters represent deviations from each task's median route shape.

## Model selection and association

| variant | role | status | K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | cross-task failure/success blocks |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| raw | sensitivity | stable_partition | 6 | 512/252/470/471/500/355 | 0.495 | 0.991/0.977 | 0.007 | 0.887 | 0.155 | 0.0002 | 1/none |
| task_residual | primary | no_stable_partition | 2 | 2014/546 | 0.166 | 0.800/0.627 | 0.220 | 0.062 | 0.004 | 0.0002 | none/none |
| task_init_residual | sensitivity | no_stable_partition | 2 | 1948/612 | 0.262 | 0.764/0.503 | 0.280 | 0.045 | 0.001 | 0.2118 | none/none |

## Primary block composition

| block | n | success/failure | failure rate | dominant task | dominant fraction | length median | eligible tasks | direction |
|---:|---:|---|---:|---|---:|---:|---:|---|
| C0 | 2014 | 1753/261 | 0.130 | open_the_middle_drawer_of_the_cabinet | 0.254 | 13.0 | 4 | none |
| C1 | 546 | 500/46 | 0.084 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.342 | 13.0 | 4 | none |

### Per-task failure-rate deltas

| block | task | n | failure rate | task baseline | delta |
|---:|---|---:|---:|---:|---:|
| C0 | open_the_top_drawer_and_put_the_bowl_inside | 432 | 0.097 | 0.082 | 0.015 |
| C0 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 398 | 0.430 | 0.422 | 0.008 |
| C0 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 325 | 0.034 | 0.023 | 0.010 |
| C0 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 348 | 0.106 | 0.072 | 0.034 |
| C1 | open_the_top_drawer_and_put_the_bowl_inside | 80 | 0.000 | 0.082 | -0.082 |
| C1 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 114 | 0.395 | 0.422 | -0.027 |
| C1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 187 | 0.005 | 0.023 | -0.018 |
| C1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 164 | 0.000 | 0.072 | -0.072 |

## Per-task outcome association

| variant | task | failures | excess NMI | p | BH q |
|---|---|---:|---:|---:|---:|
| raw | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.712 | 0.0002 | 0.0003 |
| raw | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.419 | 0.0002 | 0.0003 |
| raw | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.803 | 0.0002 | 0.0003 |
| raw | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.797 | 0.0002 | 0.0003 |
| task_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.039 | 0.0002 | 0.0003 |
| task_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | -0.000 | 0.3351 | 0.3643 |
| task_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.010 | 0.0738 | 0.0984 |
| task_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.062 | 0.0002 | 0.0003 |
| task_init_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.042 | 0.0002 | 0.0003 |
| task_init_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.001 | 0.3643 | 0.3643 |
| task_init_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.009 | 0.1080 | 0.1296 |
| task_init_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.057 | 0.0002 | 0.0003 |

## Interpretation limits

- The primary representation uses task labels for median residualization; it tests shared relative deviations, not task-independent absolute router states.
- The route-only landmark basis is fixed before subsampling, so stability is conditional on that basis.
- Shift alignment can remove real timing differences, while ten-anchor interpolation can smooth short trajectories.
- Relative completion and timeout semantics, episode length, and checkpoint remain potential outcome proxies.
- This exploratory analysis does not establish causal mechanisms or unseen-task generalization.
