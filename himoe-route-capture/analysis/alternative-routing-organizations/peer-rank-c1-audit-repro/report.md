# Peer-rank HDBSCAN C1 audit

## Verdict

C1 is a real, persistently high peer-deviation density island and its core
replicates after terminal truncation. It is descriptively failure-enriched
after task x initial-state conditioning, but the enrichment vanishes after
conditioning on exact episode length. The unqualified phrase `cross-task
high-deviation failure-enriched block` therefore overstates the result.

A defensible description is: `a cross-task, high peer-deviation density
island whose membership is associated with failure in this corpus, largely
through longer or timeout trajectories`.

## Conditional enrichment

| view | C1 n | failures | precision | task-init expected | ratio | delta | p | task-init-length expected | ratio | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 245 | 70 | 0.286 | 34.19 | 2.048 | 0.146 | 0.00002 | 69.50 | 1.007 | 0.49783 |
| truncate90 | 255 | 69 | 0.271 | 31.66 | 2.180 | 0.146 | 0.00002 | 68.25 | 1.011 | 0.25087 |

The task-init test uses conditional hypergeometric permutations, preserving
the failure count in every one of the 80 task x initial-state cells.
Exact length is outcome-proximal and is included as a leakage/confounding
diagnostic, not as a causal adjustment.

## Per-task precision

| view | task | C1 n | C1 failures | C1 rate | task rate | delta | failure recall | Fisher p | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | open_the_middle_drawer_of_the_cabinet | 20 | 0 | 0.000 | 0.000 | 0.000 | NA | 1.00000 | 1.00000 |
| full | open_the_top_drawer_and_put_the_bowl_inside | 95 | 12 | 0.126 | 0.082 | 0.044 | 0.286 | 0.06707 | 0.20122 |
| full | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 51 | 25 | 0.490 | 0.422 | 0.068 | 0.116 | 0.18597 | 0.37194 |
| full | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 41 | 12 | 0.293 | 0.023 | 0.269 | 1.000 | 0.00000 | 0.00000 |
| full | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 38 | 21 | 0.553 | 0.072 | 0.480 | 0.568 | 0.00000 | 0.00000 |
| truncate90 | open_the_middle_drawer_of_the_cabinet | 26 | 0 | 0.000 | 0.000 | 0.000 | NA | 1.00000 | 1.00000 |
| truncate90 | open_the_top_drawer_and_put_the_bowl_inside | 87 | 15 | 0.172 | 0.082 | 0.090 | 0.357 | 0.00175 | 0.00525 |
| truncate90 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 46 | 21 | 0.457 | 0.422 | 0.035 | 0.097 | 0.36401 | 0.72802 |
| truncate90 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 58 | 12 | 0.207 | 0.023 | 0.183 | 1.000 | 0.00000 | 0.00000 |
| truncate90 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 38 | 21 | 0.553 | 0.072 | 0.480 | 0.568 | 0.00000 | 0.00000 |

## Full versus truncate90

| subset | full n | trunc n | intersection | Jaccard | full retained | trunc confirmed |
|---|---:|---:|---:|---:|---:|---:|
| all_episodes | 245 | 255 | 189 | 0.608 | 0.771 | 0.741 |
| failures_only | 70 | 69 | 62 | 0.805 | 0.886 | 0.899 |
| successes_only | 175 | 186 | 127 | 0.543 | 0.726 | 0.683 |

## Feature curves

The raw curve values are within-task x initial-state peer ranks. A value
near 0.84 means that C1 stays near the high-deviation end of its 32-rollout
sibling group; it is not an absolute routing-distance measurement.

| view | curve | C1 mean rank | non-C1 | delta | min phase delta | first | last |
|---|---|---:|---:|---:|---:|---:|---:|
| full | point | 0.842 | 0.464 | 0.378 | 0.325 | 0.814 | 0.794 |
| full | speed | 0.843 | 0.464 | 0.379 | 0.334 | 0.826 | 0.802 |
| full | bend | 0.838 | 0.464 | 0.373 | 0.333 | 0.844 | 0.801 |
| truncate90 | point | 0.848 | 0.461 | 0.387 | 0.334 | 0.801 | 0.838 |
| truncate90 | speed | 0.839 | 0.463 | 0.376 | 0.353 | 0.818 | 0.842 |
| truncate90 | bend | 0.812 | 0.465 | 0.347 | 0.306 | 0.801 | 0.824 |

All three curves are elevated at every phase. Full C1 weakens slightly at
the terminal anchor instead of being created by a terminal-only spike; the
truncate90 curves remain elevated through phase 0.90.

## HDBSCAN sensitivity

Each candidate below is selected label-blind as the discovered cluster with
the highest mean raw peer-rank curve. The full grid is in `sensitivity.csv`.

| view | viable/grid | Jaccard to default median (range) | size range | failure-rate range | all viable cover 5 tasks |
|---|---:|---:|---:|---:|:---:|
| full | 31/42 | 0.938 (0.698-1.000) | 241-351 | 0.231-0.289 | true |
| truncate90 | 39/42 | 0.957 (0.812-1.000) | 220-312 | 0.228-0.314 | true |

Across 31 settings where both views produce clusters, the full/truncate90 high-block Jaccard has median 0.610 (range 0.582-0.637); failure-only Jaccard has median 0.808 (range 0.787-0.854).

Some conservative settings return no cluster at all. With
`allow_single_cluster=False`, raising min_cluster_size above the small
low-deviation companion island can invalidate the entire two-island fit,
even though the high-deviation candidate itself is much larger.

## Limits

- This is a targeted post-hoc audit of a named block, not a discovery p-value.
- Four tasks contain failures; the middle-drawer task has zero failures and cannot validate enrichment.
- The strong evidence is concentrated in the two spatial bowl tasks; direction is positive but weaker in top-drawer and moka-pot.
- Full and truncate90 use relative-phase anchors with different terminal semantics.
- No unseen task is available, so cross-task means within-corpus coverage, not task generalization.
