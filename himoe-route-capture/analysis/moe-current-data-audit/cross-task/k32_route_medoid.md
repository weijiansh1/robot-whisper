# K32 full-route medoid shadow audit

All values use one state as the statistical unit. Random is the exact K32 pool mean, not a sampled baseline.

| method | selected success | random mean | delta | state-bootstrap 95% CI | unique seeds | max seed share |
|---|---:|---:|---:|---:|---:|---:|
| route probability medoid d0 | 0.887 | 0.880 | +0.007 | [-0.037, +0.051] | 3 | 0.912 |
| route probability medoid d0-2 | 0.887 | 0.880 | +0.007 | [-0.037, +0.052] | 2 | 0.975 |
| route probability medoid full | 0.887 | 0.880 | +0.007 | [-0.038, +0.051] | 3 | 0.975 |
| expert-ID medoid d0 | 0.875 | 0.880 | -0.005 | [-0.048, +0.035] | 11 | 0.263 |
| expert-ID medoid d0-2 | 0.887 | 0.880 | +0.007 | [-0.030, +0.043] | 9 | 0.338 |
| expert-ID medoid full | 0.863 | 0.880 | -0.018 | [-0.068, +0.031] | 7 | 0.362 |

## Per-task deltas

| method | libero_goal/open_the_middle_drawer_of_the_cabinet | libero_goal/open_the_top_drawer_and_put_the_bowl_inside | libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate |
|---|---:|---:|---:|---:|---:|
| route probability medoid d0 | +0.000 | -0.105 | +0.047 | +0.023 | +0.072 |
| route probability medoid d0-2 | +0.000 | -0.105 | +0.047 | +0.023 | +0.072 |
| route probability medoid full | +0.000 | -0.105 | +0.047 | +0.023 | +0.072 |
| expert-ID medoid d0 | +0.000 | +0.020 | -0.016 | -0.039 | +0.010 |
| expert-ID medoid d0-2 | +0.000 | +0.020 | +0.047 | +0.023 | -0.053 |
| expert-ID medoid full | +0.000 | +0.020 | -0.016 | +0.023 | -0.115 |

## Dispersion and opportunity

The opportunity target is `max(success) - mean(success)` in each K32 state pool. Correlation is undefined for a task whose opportunity is constant.

| method | finite-task Spearman macro | task-heldout MAE | train-mean baseline MAE |
|---|---:|---:|---:|
| route probability medoid d0 | 0.052 | 0.188 | 0.161 |
| route probability medoid d0-2 | 0.089 | 0.188 | 0.161 |
| route probability medoid full | 0.025 | 0.188 | 0.161 |
| expert-ID medoid d0 | -0.217 | 0.162 | 0.161 |
| expert-ID medoid d0-2 | -0.207 | 0.157 | 0.161 |
| expert-ID medoid full | -0.100 | 0.147 | 0.161 |

## Interpretation boundary

This is an episode-start shadow association. Each candidate has one outcome; the first query and all later replanning noise share the candidate seed stream. There are no continuation repeats or common random numbers, and no medoid was executed online. A selected-success delta therefore is not a first-chunk causal effect.
