# Truncate-90 aligned route-kernel sensitivity

Run class: `nonformal`.

Yes. Full-window raw C1 strongly persists after removing the final 10% of relative phase and remains a cross-task failure block.

## Raw C1 persistence

| comparison | value |
|---|---:|
| full raw K6 vs truncate-90 raw K6 ARI | 0.950 |
| matched truncate-90 cluster | C1 |
| reference episodes/failures | 252/246 |
| candidate episodes/failures | 249/219 |
| episode precision/recall/Jaccard | 0.851/0.841/0.734 |
| failure precision/recall/Jaccard | 0.945/0.841/0.802 |
| strong persistence | true |
| ordinary cross-task failure block retained | true |

### Per-task composition

| task | reference n/fail | truncate-90 n/fail | overlap n/fail | reference recall | candidate failure-rate delta |
|---|---:|---:|---:|---:|---:|
| open_the_top_drawer_and_put_the_bowl_inside | 42/42 | 41/41 | 41/41 | 0.976 | 0.918 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 160/158 | 136/133 | 123/121 | 0.769 | 0.556 |
| pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 13/11 | 35/11 | 12/11 | 0.923 | 0.291 |
| pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 37/35 | 37/34 | 36/34 | 0.973 | 0.847 |

## Refit model selection

| variant | status | selected/reported K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | failure/success blocks | full-label ARI |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---:|
| raw | stable_partition | 6/6 | 512/249/470/472/475/382 | 0.528 | 0.985/0.934 | 0.020 | 0.876 | 0.124 | 0.0002 | 1/none | 0.950 |
| task_residual | no_stable_partition | none/2 | 1879/681 | 0.170 | 0.883/0.559 | 0.277 | 0.368 | 0.064 | 0.0002 | none/none | -0.012 |

## Task-residual sensitivity

The fixed K2 full-vs-truncate ARI is -0.012. The truncate-90 partition status is `no_stable_partition` and its cross-task failure/success blocks are []/[].

## Interpretation limits

- Raw persistence is a robustness result for the episode set, not evidence that the block is task-independent.
- The candidate remains subject to task, episode-length, checkpoint, and timeout proxy explanations.
- Reference matching is posthoc but occurs only after truncate-90 labels are frozen.
- Stability is conditional on full-cohort anchor-PCA and landmark bases.
