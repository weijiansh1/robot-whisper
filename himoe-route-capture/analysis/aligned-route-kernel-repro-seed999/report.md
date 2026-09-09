# Aligned route-signature landmark experiment

Run class: `nonformal`.

No block in the primary task-residual partition passed the frozen cross-task success/failure criterion.

This is an exploratory posthoc analysis. Task residualization is explicit;
clusters represent deviations from each task's median route shape.

## Model selection and association

| variant | role | status | K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | cross-task failure/success blocks |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| raw | sensitivity | stable_partition | 6 | 512/255/470/472/500/351 | 0.480 | 0.990/0.978 | 0.013 | 0.890 | 0.153 | 0.0002 | 1/none |
| task_residual | primary | no_stable_partition | 2 | 2077/483 | 0.182 | 0.790/0.614 | 0.282 | 0.054 | -0.000 | 0.4367 | none/none |
| task_init_residual | sensitivity | no_stable_partition | 2 | 2281/279 | 0.228 | 0.678/0.502 | 0.275 | 0.036 | -0.002 | 0.9084 | none/none |

## Primary block composition

| block | n | success/failure | failure rate | dominant task | dominant fraction | length median | eligible tasks | direction |
|---:|---:|---|---:|---|---:|---:|---:|---|
| C0 | 2077 | 1820/257 | 0.124 | open_the_middle_drawer_of_the_cabinet | 0.247 | 13.0 | 4 | none |
| C1 | 483 | 433/50 | 0.104 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.342 | 13.0 | 4 | none |

### Per-task failure-rate deltas

| block | task | n | failure rate | task baseline | delta |
|---:|---|---:|---:|---:|---:|
| C0 | open_the_top_drawer_and_put_the_bowl_inside | 426 | 0.099 | 0.082 | 0.017 |
| C0 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 422 | 0.396 | 0.422 | -0.026 |
| C0 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 347 | 0.032 | 0.023 | 0.008 |
| C0 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 370 | 0.100 | 0.072 | 0.028 |
| C1 | open_the_top_drawer_and_put_the_bowl_inside | 86 | 0.000 | 0.082 | -0.082 |
| C1 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 90 | 0.544 | 0.422 | 0.123 |
| C1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 165 | 0.006 | 0.023 | -0.017 |
| C1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 142 | 0.000 | 0.072 | -0.072 |

## Per-task outcome association

| variant | task | failures | excess NMI | p | BH q |
|---|---|---:|---:|---:|---:|
| raw | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.711 | 0.0002 | 0.0004 |
| raw | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.413 | 0.0002 | 0.0004 |
| raw | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.711 | 0.0002 | 0.0004 |
| raw | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.840 | 0.0002 | 0.0004 |
| task_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.041 | 0.0002 | 0.0004 |
| task_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.005 | 0.1568 | 0.1710 |
| task_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.008 | 0.1092 | 0.1456 |
| task_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.055 | 0.0002 | 0.0004 |
| task_init_residual | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.016 | 0.1828 | 0.1828 |
| task_init_residual | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.020 | 0.1032 | 0.1456 |
| task_init_residual | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.012 | 0.1500 | 0.1710 |
| task_init_residual | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.038 | 0.0040 | 0.0069 |

## Interpretation limits

- The primary representation uses task labels for median residualization; it tests shared relative deviations, not task-independent absolute router states.
- The route-only landmark basis is fixed before subsampling, so stability is conditional on that basis.
- Shift alignment can remove real timing differences, while ten-anchor interpolation can smooth short trajectories.
- Relative completion and timeout semantics, episode length, and checkpoint remain potential outcome proxies.
- This exploratory analysis does not establish causal mechanisms or unseen-task generalization.
