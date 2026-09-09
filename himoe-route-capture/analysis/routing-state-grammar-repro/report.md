# Symbolic MoE routing-state grammar

The ten normalized phase anchors are quantized into shared, label-blind route
states. Episode vectors contain normalized state occupancies, directed Markov
transitions, coarse phase occupancy, and return motifs; raw phase descriptors
are not flattened into the episode representation.

## State vocabulary

- Primary states: 8.
- State counts: 1722/7708/4538/2578/1346/4341/2658/709.
- Seed-refit ARI median/min: 0.997/0.996.
- Grammar dimensions: 187.

## Episode grammar partition

Status: `stable_partition`; reported K=6; sizes=462/576/485/464/505/68.
Reported-K subsample ARI median/P10=0.975/0.959, PAC=0.023.

| association | NMI | null | excess | p | BH q |
|---|---:|---:|---:|---:|---:|
| outcome_within_task_initial_state | 0.133 | 0.088 | 0.046 | 0.0002 | 0.0002 |
| task | 0.887 | 0.002 | 0.884 | 0.0002 | 0.0002 |
| checkpoint | 0.730 | 0.001 | 0.728 | 0.0002 | 0.0002 |
| episode_length | 0.598 | 0.014 | 0.584 | 0.0002 | 0.0002 |
| episode_length_within_task | 0.598 | 0.550 | 0.048 | 0.0002 | 0.0002 |

### Outcome association within each task

| task | episodes | failures | NMI excess | p | BH q |
|---|---:|---:|---:|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 512 | 0 | no outcome variation | - | - |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 512 | 42 | 0.156 | 0.0002 | 0.0003 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 512 | 216 | 0.009 | 0.0662 | 0.0662 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 512 | 12 | 0.255 | 0.0002 | 0.0003 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 512 | 37 | 0.708 | 0.0002 | 0.0003 |

Interpretation flags: `task_shadowed`.

## Cluster composition

| cluster | n | success | failure | failure rate | dominant task | length min/med/max |
|---:|---:|---:|---:|---:|---:|---|
| C0 | 462 | 462 | 0 | 0.000 | 1.000 | 11/12.0/15 |
| C1 | 576 | 553 | 23 | 0.040 | 0.889 | 12/13.0/52 |
| C2 | 485 | 482 | 3 | 0.006 | 1.000 | 9/10.0/22 |
| C3 | 464 | 442 | 22 | 0.047 | 0.961 | 12/19.0/52 |
| C4 | 505 | 293 | 212 | 0.420 | 0.990 | 30/40.0/52 |
| C5 | 68 | 21 | 47 | 0.691 | 0.588 | 9/22.0/52 |

## Grammar scalar outcome tests

| diagnostic | success | failure | residual effect SD | BH q |
|---|---:|---:|---:|---:|
| state_entropy | 0.334 | 0.230 | 0.018 | 0.7634 |
| transition_entropy | 0.250 | 0.186 | 0.089 | 0.2045 |
| switch_rate | 0.235 | 0.243 | 0.102 | 0.1848 |
| unique_state_fraction | 0.241 | 0.216 | 0.102 | 0.1848 |
| nonlocal_revisit_rate | 0.089 | 0.128 | 0.066 | 0.2658 |
| longest_run_fraction | 0.567 | 0.675 | 0.084 | 0.2045 |
| aba_return_rate | 0.038 | 0.066 | 0.097 | 0.1848 |
| abc_progression_rate | 0.013 | 0.060 | 0.489 | 0.0016 |

## Fixed-representation probes

| outcome model | fold-mean AUC | mixed-task macro AUC |
|---|---:|---:|
| task_only | 0.793 | 0.500 |
| occupancy_only | 0.893 | 0.865 |
| grammar_only | 0.982 | 0.977 |
| task_plus_occupancy | 0.939 | 0.917 |
| task_plus_grammar | 0.984 | 0.979 |
| task_plus_length | 0.998 | 0.999 |
| task_length_occupancy | 0.998 | 0.998 |
| task_length_grammar | 0.995 | 0.997 |

Task balanced accuracy: occupancy-only 0.991; full grammar 0.998.

| length model | R2 | MAE queries |
|---|---:|---:|
| task_only | 0.922 | 2.301 |
| occupancy_only | 0.932 | 2.180 |
| grammar_only | 0.960 | 1.463 |
| task_plus_grammar | 0.965 | 1.387 |

## Sensitivity

| configuration | states | anchors | fixed episode K | ARI | sizes | min-size |
|---|---:|---:|---:|---:|---|---|
| state_k6 | 6 | 10 | 6 | 0.847 | 466/487/478/502/69/558 | yes |
| state_k10 | 10 | 10 | 6 | 0.881 | 533/442/539/54/520/472 | yes |
| phase_grid5 | 8 | 5 | 6 | 0.843 | 468/529/503/55/463/542 | yes |
| truncate90 | 8 | 10 | 6 | 0.783 | 614/510/463/493/50/430 | no |

## Limits

- The vocabulary is expert-permutation-invariant but can still encode task, checkpoint, and visual-state structure.
- Relative phase and normalized transition counts remove explicit trajectory length, not completion-versus-timeout semantics.
- Failure horizon has almost no overlap with successful lengths, so task/length-adjusted probes diagnose rather than identify an outcome-specific mechanism.
- Episode K is searched only through six; selection at K=6 is an upper-bound result and may hide finer grammar states.
- The cross-validation probes use a vocabulary fitted label-blind on the full cohort and are descriptive, not held-out vocabulary generalization.
