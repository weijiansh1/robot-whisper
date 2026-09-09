# Joint success/failure routing blocks

Run class: `nonformal`.

No primary view simultaneously passed the frozen stability and conditional outcome-structure gates.

For the stronger question of whether adding successes changes the old failure
taxonomy, the confirmatory same-K comparisons are: `geometry`=preserved (ARI 1.000); `recurrence`=moderately_changed (ARI 0.723); `expert_occupancy`=reorganized (ARI 0.000).
Outcome is tested only after route-only clustering, with permutations inside
task x initial-state strata. This is an association, not a causal outcome signal.

## Coverage

| task | success | failure | queries | length min/median/max | checkpoint |
|---|---:|---:|---:|---|---|
| open_the_middle_drawer_of_the_cabinet | 512 | 0 | 6452 | 12/13.0/14 | `98ee29d09d18` |
| open_the_top_drawer_and_put_the_bowl_inside | 470 | 42 | 10018 | 17/19.0/30 | `98ee29d09d18` |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 296 | 216 | 22883 | 35/40.0/52 | `cdc2b21f9ef6` |
| pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 500 | 12 | 5139 | 9/10.0/22 | `1029d0827030` |
| pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 475 | 37 | 6816 | 11/13.0/22 | `1029d0827030` |

Total: 2560 episodes (2253 successes, 307 failures), 51308 queries. Every task
passes the complete 16 initial-state x 32 flow-seed design guard.

## Joint Blocks

The reported K is selected only among partitions passing all frozen
subsample/PAC/size gates. Otherwise it is explicitly exploratory.

| view | status | K | sizes | silhouette | ARI med/p10 | PAC | outcome NMI | null NMI | excess | p / BH q | outcome blocks | task NMI | flags |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---|---:|---|
| geometry | stable_partition | 2 | 1536/1024 | 0.402 | 1.000/1.000 | 0.000 | 0.035 | 0.035 | 0.000 | 1.0000/1.0000 | no | 0.590 | task_shadowed |
| change | stable_partition | 2 | 2304/256 | 0.336 | 0.932/0.894 | 0.074 | 0.010 | 0.011 | -0.001 | 0.5535/0.5904 | no | 0.193 | task_shadowed |
| recurrence | stable_partition | 6 | 101/874/231/525/268/561 | 0.349 | 0.969/0.945 | 0.050 | 0.105 | 0.078 | 0.028 | 0.0002/0.0002 | no | 0.693 | task_shadowed |
| expert_occupancy | stable_partition | 6 | 512/147/470/472/496/463 | 0.259 | 0.982/0.962 | 0.008 | 0.222 | 0.129 | 0.093 | 0.0002/0.0002 | yes | 0.883 | outcome_structured+task_shadowed+secondary_expert_id_view |

## Block Composition

| view | block | n | success | failure | failure rate | dominant-task fraction | length min/median/max |
|---|---:|---:|---:|---:|---:|---:|---|
| geometry | C0 | 1536 | 1278 | 258 | 0.168 | 0.333 | 12/19.0/52 |
| geometry | C1 | 1024 | 975 | 49 | 0.048 | 0.500 | 9/12.0/22 |
| change | C0 | 2304 | 2009 | 295 | 0.128 | 0.222 | 9/13.0/52 |
| change | C1 | 256 | 244 | 12 | 0.047 | 1.000 | 36/39.0/52 |
| recurrence | C0 | 101 | 0 | 101 | 1.000 | 1.000 | 52/52.0/52 |
| recurrence | C1 | 874 | 807 | 67 | 0.077 | 0.586 | 12/13.0/52 |
| recurrence | C2 | 231 | 222 | 9 | 0.039 | 1.000 | 9/9.0/22 |
| recurrence | C3 | 525 | 510 | 15 | 0.029 | 0.924 | 11/13.0/22 |
| recurrence | C4 | 268 | 243 | 25 | 0.093 | 0.899 | 10/10.0/22 |
| recurrence | C5 | 561 | 471 | 90 | 0.160 | 0.684 | 19/39.0/52 |
| expert_occupancy | C0 | 512 | 512 | 0 | 0.000 | 1.000 | 12/13.0/14 |
| expert_occupancy | C1 | 147 | 10 | 137 | 0.932 | 1.000 | 44/52.0/52 |
| expert_occupancy | C2 | 470 | 470 | 0 | 0.000 | 1.000 | 17/19.0/21 |
| expert_occupancy | C3 | 472 | 472 | 0 | 0.000 | 1.000 | 11/12.0/15 |
| expert_occupancy | C4 | 496 | 496 | 0 | 0.000 | 1.000 | 9/10.0/13 |
| expert_occupancy | C5 | 463 | 293 | 170 | 0.367 | 0.788 | 13/39.0/52 |

## Failure Taxonomy Change

`same K` is the primary estimand: the all-outcome dendrogram is cut at the
old failure-only K before restricting to the same 307 failures. `selected K`
is auxiliary because it also includes model-selection changes.

| view | old status/K | joint status/K | selected-K ARI | same-K ARI | Jaccard | split bits | merge bits | same-K class |
|---|---|---|---:|---:|---:|---:|---:|---|
| geometry | stable_partition/2 | stable_partition/2 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | preserved |
| change | no_stable_partition/2 | stable_partition/2 | -0.042 | -0.042 | 0.454 | 0.235 | 0.292 | reorganized |
| recurrence | stable_partition/6 | stable_partition/6 | 0.723 | 0.723 | 0.456 | 0.293 | 0.585 | moderately_changed |
| expert_occupancy | stable_partition/4 | stable_partition/6 | 0.540 | 0.000 | 0.414 | 0.000 | 1.806 | reorganized |

### Same-K transition details

#### geometry

| old failure block | joint C0 | joint C1 |
|---|---:|---:|
| old C0 | 258 | 0 |
| old C1 | 0 | 49 |

Within-task same-K ARI:

Not estimable: no task contains at least two blocks in both partitions.
#### change

| old failure block | joint C0 | joint C1 |
|---|---:|---:|
| old C0 | 279 | 12 |
| old C1 | 16 | 0 |

Within-task same-K ARI:

| task | failures | ARI |
|---|---:|---:|
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | -0.058 |
#### recurrence

| old failure block | joint C0 | joint C1 | joint C2 | joint C3 | joint C4 | joint C5 |
|---|---:|---:|---:|---:|---:|---:|
| old C0 | 64 | 0 | 0 | 0 | 0 | 0 |
| old C1 | 37 | 0 | 0 | 0 | 0 | 0 |
| old C2 | 0 | 42 | 0 | 0 | 0 | 0 |
| old C3 | 0 | 24 | 0 | 0 | 0 | 2 |
| old C4 | 0 | 1 | 0 | 0 | 0 | 88 |
| old C5 | 0 | 0 | 9 | 15 | 25 | 0 |

Within-task same-K ARI:

| task | failures | ARI |
|---|---:|---:|
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.747 |
#### expert_occupancy

| old failure block | joint C3 |
|---|---:|
| old C0 | 127 |
| old C1 | 40 |
| old C2 | 103 |
| old C3 | 37 |

Within-task same-K ARI:

Not estimable: no task contains at least two blocks in both partitions.

### Support and termination controls

The matched controls use 157 unique failure-success pairs with identical
task x initial-state support. They are descriptive for this supported subset.

| view | C2 shared tasks | support/refit | full +success | remove failure tail | prefix +success |
|---|---:|---:|---:|---:|---:|
| geometry | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| change | -0.038 | 0.436 | -0.044 | 0.264 | 0.925 |
| recurrence | 0.750 | 0.709 | 0.884 | 0.516 | 0.803 |
| expert_occupancy | 0.000 | 0.961 | 0.751 | 0.637 | 0.603 |

## Per-task Outcome Association

| view | task | failures | outcome NMI excess | p | BH q |
|---|---|---:|---:|---:|---:|
| geometry | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.000 | 1.0000 | 1.0000 |
| geometry | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.000 | 1.0000 | 1.0000 |
| geometry | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.000 | 1.0000 | 1.0000 |
| geometry | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.000 | 1.0000 | 1.0000 |
| change | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.000 | 1.0000 | 1.0000 |
| change | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.347 | 0.0002 | 0.0004 |
| change | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.000 | 1.0000 | 1.0000 |
| change | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.000 | 1.0000 | 1.0000 |
| recurrence | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.057 | 0.0002 | 0.0004 |
| recurrence | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.247 | 0.0002 | 0.0004 |
| recurrence | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.029 | 0.0002 | 0.0004 |
| recurrence | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.515 | 0.0002 | 0.0004 |
| expert_occupancy | open_the_top_drawer_and_put_the_bowl_inside | 42 | 0.711 | 0.0002 | 0.0004 |
| expert_occupancy | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 216 | 0.212 | 0.0002 | 0.0004 |
| expert_occupancy | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 12 | 0.733 | 0.0002 | 0.0004 |
| expert_occupancy | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37 | 0.836 | 0.0002 | 0.0004 |

## Relative-phase Sensitivity

| view | configuration | K | ARI to primary | sizes | min-size pass | outcome excess | BH q |
|---|---|---:|---:|---|---|---:|---:|
| geometry | grid5 | 2 | 1.000 | 1536/1024 | yes | 0.000 | 1.0000 |
| geometry | truncate90 | 2 | 1.000 | 1536/1024 | yes | 0.000 | 1.0000 |
| change | grid5 | 2 | 0.841 | 2265/295 | yes | -0.012 | 1.0000 |
| change | truncate90 | 2 | 0.860 | 2302/258 | yes | -0.005 | 1.0000 |
| recurrence | grid5 | 6 | 0.656 | 101/1027/222/550/252/408 | yes | 0.052 | 0.0006 |
| recurrence | truncate90 | 6 | 0.721 | 110/1036/489/462/390/73 | yes | 0.085 | 0.0006 |

## Cross-view Agreement

| view | geometry | change | recurrence | expert_occupancy |
|---|---:|---:|---:|---:|
| geometry | 1.000 | -0.026 | 0.427 | 0.312 |
| change | -0.026 | 1.000 | 0.051 | 0.045 |
| recurrence | 0.427 | 0.051 | 1.000 | 0.548 |
| expert_occupancy | 0.312 | 0.045 | 0.548 | 1.000 |

## Interpretation Limits

- Relative phase removes absolute chunk index and episode length from the feature vector, but it does not make success completion and failure timeout semantically equivalent.
- Failure horizons are almost perfectly outcome-linked in this corpus. The matched-prefix controls reduce this confound only on 157 supported pairs; they do not identify a causal outcome effect for all failures.
- A task-shadowed partition can still have conditioned outcome association, but it is not a task-independent routing taxonomy.
- `change` results remain descriptive if either the old or joint partition fails the frozen stability gate. `expert_occupancy` is an expert-ID-based secondary view.
- Stability is conditional on episode subsampling and does not establish generalization to unseen tasks, states, or noise draws.
