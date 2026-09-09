# Aligned route-signature landmark experiment

Run class: `nonformal`.

No block in the primary task-residual partition passed the frozen cross-task success/failure criterion.

This is an exploratory posthoc analysis. Task residualization is explicit;
clusters represent deviations from each task's median route shape.

## Model selection and association

| variant | role | status | K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | cross-task failure/success blocks |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| raw | sensitivity | stable_partition | 6 | 512/264/470/471/495/348 | 0.482 | 0.991/0.977 | 0.006 | 0.888 | 0.150 | 0.0002 | 1/none |
| task_residual | primary | no_stable_partition | 2 | 2205/355 | 0.165 | 0.671/0.441 | 0.346 | 0.072 | 0.000 | 0.1909 | none/none |
| task_init_residual | sensitivity | no_stable_partition | 2 | 2056/504 | 0.261 | 0.737/0.576 | 0.240 | 0.047 | -0.001 | 0.6657 | none/none |

## Primary block composition

| block | n | success/failure | failure rate | dominant task | dominant fraction | length median | eligible tasks | direction |
|---:|---:|---|---:|---|---:|---:|---:|---|
| C0 | 2205 | 1934/271 | 0.123 | open_the_middle_drawer_of_the_cabinet | 0.232 | 13.0 | 4 | none |
| C1 | 355 | 319/36 | 0.101 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.417 | 13.0 | 4 | none |

### Per-task failure-rate deltas

| block | task | n | failure rate | task baseline | delta |
|---:|---|---:|---:|---:|---:|
| C0 | open_the_top_drawer_and_put_the_bowl_inside | 489 | 0.086 | 0.082 | 0.004 |
| C0 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 472 | 0.383 | 0.422 | -0.038 |
| C0 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 364 | 0.030 | 0.023 | 0.007 |
| C0 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 368 | 0.101 | 0.072 | 0.028 |
| C1 | open_the_top_drawer_and_put_the_bowl_inside | 23 | 0.000 | 0.082 | -0.082 |
| C1 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 40 | 0.875 | 0.422 | 0.453 |
| C1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 148 | 0.007 | 0.023 | -0.017 |
| C1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 144 | 0.000 | 0.072 | -0.072 |

## Per-task outcome association

| variant | task | failures | excess NMI | p | BH q |
|---|---|---:|---:|---:|---:|
| raw | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.712 | 0.0002 | 0.0003 |
| raw | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.399 | 0.0002 | 0.0003 |
| raw | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.674 | 0.0002 | 0.0003 |
| raw | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.813 | 0.0002 | 0.0003 |
| task_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.013 | 0.1600 | 0.1745 |
| task_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.057 | 0.0002 | 0.0003 |
| task_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.006 | 0.1490 | 0.1745 |
| task_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.056 | 0.0002 | 0.0003 |
| task_init_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.044 | 0.0002 | 0.0003 |
| task_init_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | -0.009 | 0.8892 | 0.8892 |
| task_init_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.008 | 0.1234 | 0.1645 |
| task_init_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.056 | 0.0002 | 0.0003 |

## Interpretation limits

- The primary representation uses task labels for median residualization; it tests shared relative deviations, not task-independent absolute router states.
- The route-only landmark basis is fixed before subsampling, so stability is conditional on that basis.
- Shift alignment can remove real timing differences, while ten-anchor interpolation can smooth short trajectories.
- Relative completion and timeout semantics, episode length, and checkpoint remain potential outcome proxies.
- This exploratory analysis does not establish causal mechanisms or unseen-task generalization.
