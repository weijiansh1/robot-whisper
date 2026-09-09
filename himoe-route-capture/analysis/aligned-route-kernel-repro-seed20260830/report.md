# Aligned route-signature landmark experiment

Run class: `nonformal`.

No block in the primary task-residual partition passed the frozen cross-task success/failure criterion.

This is an exploratory posthoc analysis. Task residualization is explicit;
clusters represent deviations from each task's median route shape.

## Model selection and association

| variant | role | status | K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | cross-task failure/success blocks |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| raw | sensitivity | stable_partition | 6 | 512/260/470/471/502/345 | 0.478 | 0.982/0.970 | 0.010 | 0.887 | 0.154 | 0.0002 | 1/none |
| task_residual | primary | no_stable_partition | 2 | 1987/573 | 0.166 | 0.835/0.613 | 0.228 | 0.067 | 0.001 | 0.0161 | none/none |
| task_init_residual | sensitivity | no_stable_partition | 2 | 2006/554 | 0.250 | 0.673/0.528 | 0.230 | 0.056 | -0.000 | 0.5731 | none/none |

## Primary block composition

| block | n | success/failure | failure rate | dominant task | dominant fraction | length median | eligible tasks | direction |
|---:|---:|---|---:|---|---:|---:|---:|---|
| C0 | 1987 | 1735/252 | 0.127 | open_the_middle_drawer_of_the_cabinet | 0.257 | 13.0 | 4 | none |
| C1 | 573 | 518/55 | 0.096 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.349 | 13.0 | 4 | none |

### Per-task failure-rate deltas

| block | task | n | failure rate | task baseline | delta |
|---:|---|---:|---:|---:|---:|
| C0 | open_the_top_drawer_and_put_the_bowl_inside | 436 | 0.096 | 0.082 | 0.014 |
| C0 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 385 | 0.421 | 0.422 | -0.001 |
| C0 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 312 | 0.035 | 0.023 | 0.012 |
| C0 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 343 | 0.108 | 0.072 | 0.036 |
| C1 | open_the_top_drawer_and_put_the_bowl_inside | 76 | 0.000 | 0.082 | -0.082 |
| C1 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 127 | 0.425 | 0.422 | 0.003 |
| C1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 200 | 0.005 | 0.023 | -0.018 |
| C1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 169 | 0.000 | 0.072 | -0.072 |

## Per-task outcome association

| variant | task | failures | excess NMI | p | BH q |
|---|---|---:|---:|---:|---:|
| raw | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.711 | 0.0002 | 0.0003 |
| raw | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.447 | 0.0002 | 0.0003 |
| raw | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.637 | 0.0002 | 0.0003 |
| raw | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.797 | 0.0002 | 0.0003 |
| task_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.038 | 0.0002 | 0.0003 |
| task_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | -0.001 | 1.0000 | 1.0000 |
| task_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.012 | 0.0464 | 0.0619 |
| task_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.064 | 0.0002 | 0.0003 |
| task_init_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.054 | 0.0002 | 0.0003 |
| task_init_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | -0.006 | 0.8958 | 1.0000 |
| task_init_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | -0.003 | 1.0000 | 1.0000 |
| task_init_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.059 | 0.0002 | 0.0003 |

## Interpretation limits

- The primary representation uses task labels for median residualization; it tests shared relative deviations, not task-independent absolute router states.
- The route-only landmark basis is fixed before subsampling, so stability is conditional on that basis.
- Shift alignment can remove real timing differences, while ten-anchor interpolation can smooth short trajectories.
- Relative completion and timeout semantics, episode length, and checkpoint remain potential outcome proxies.
- This exploratory analysis does not establish causal mechanisms or unseen-task generalization.
