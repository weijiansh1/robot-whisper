# Replanning-reset trap audit

This audit first matches each query to its physically nearest eligible
history query, then asks whether aligned HB routing and the 10x7 action
chunk also return without progress.

## Coverage

| task | episodes | failures | shared prefix | targets |
|---|---:|---:|---:|---|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 512 | 0 | 12 | none |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 512 | 42 | 17 | akita_black_bowl_1_joint0 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 512 | 216 | 35 | moka_pot_1_joint0, moka_pot_2_joint0 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 512 | 12 | 9 | akita_black_bowl_1_joint0 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 512 | 37 | 11 | akita_black_bowl_1_joint0 |

## Primary equal-prefix result

AUC above 0.5 means the score is larger in failures after comparing
only within task x initial-state strata.

| endpoint | failure AUC | cluster 95% CI | p one-sided | family-wise p |
|---|---:|---|---:|---:|
| physical return | 0.662 | [0.576, 0.733] | 0.0002 | 0.0002 |
| state-token route return | 0.673 | [0.579, 0.747] | 0.0002 | 0.0002 |
| action-token route return | 0.733 | [0.649, 0.805] | 0.0002 | 0.0002 |
| action-chunk return | 0.762 | [0.690, 0.822] | 0.0002 | 0.0002 |
| physical + route + no-progress | 0.741 | [0.642, 0.822] | 0.0002 | 0.0002 |
| physical + action + no-progress | 0.704 | [0.605, 0.791] | 0.0002 | 0.0002 |
| joint replanning-reset score | 0.720 | [0.616, 0.804] | 0.0002 | 0.0002 |

**Primary reading:** The prespecified joint-return endpoint is enriched in failures within the equal prefix.

## Backward-return topology control

A matched history point is a true backward return only if it is more
similar to the current query than the immediately preceding query.
This control was added after the continuous primary score was seen.

The physical nearest-neighbor search selected the minimum allowed lag
(lag 2) for 91.6% of all pairs, 91.6% of successes, and 91.8% of failures.

| topology endpoint | failure AUC | 95% CI | p one-sided | family-wise p |
|---|---:|---|---:|---:|
| physical return excess over lag 1 | 0.600 | [0.521, 0.671] | 0.0004 | 0.0012 |
| route return excess over lag 1 | 0.509 | [0.419, 0.592] | 0.3737 | 0.7702 |
| action return excess over lag 1 | 0.604 | [0.497, 0.699] | 0.0002 | 0.0008 |
| joint backward-return minimum excess | 0.626 | [0.535, 0.706] | 0.0002 | 0.0002 |
| strict backward-loop candidate rate | 0.504 | [0.494, 0.514] | 0.3999 | 0.8882 |

Strict backward-loop candidates are not enriched. The positive
continuous return score should therefore be read as persistence/
low-motion geometry, not as a confirmed nonlocal routing loop.

In the equal-prefix windows, the strict rule selected 8/8175 failure
pairs across 7/307 episodes and 17/29713 success pairs across 17/2253 episodes.

## Increment and per-task checks

| AUC contrast | delta | 95% CI | p(left > right) |
|---|---:|---|---:|
| action_route_minus_physical | +0.071 | [+0.013, +0.133] | 0.0176 |
| joint_minus_physical | +0.058 | [-0.007, +0.119] | 0.0198 |
| joint_minus_action | -0.042 | [-0.118, +0.022] | 0.9494 |
| action_route_minus_action | -0.029 | [-0.085, +0.026] | 0.8502 |

| task | joint AUC | strict backward-loop AUC |
|---|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | all success | - |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 0.605 | 0.500 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.782 | 0.510 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.179 | 0.480 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.877 | 0.500 |

## Search-range sensitivity

| history search | joint AUC | 95% CI | p one-sided |
|---|---:|---|---:|
| lag2_8 | 0.720 | [0.613, 0.809] | 0.0002 |
| lag3_8 | 0.723 | [0.627, 0.802] | 0.0002 |
| lag2_16 | 0.720 | [0.616, 0.807] | 0.0002 |
| lag2_all | 0.720 | [0.614, 0.807] | 0.0002 |

## Same-length terminal window

This window can see late slips but is phase-confounded and is not a
prediction test.

Joint-return failure AUC 0.979 [0.956, 0.996], p=0.0002.

## Physically recurrent pair quadrants

Thresholds are calibrated from successful equal-prefix pairs. Values
are mean within-episode rates conditional on physical recurrence.

| outcome | route repeat / action repeat | route repeat / action different | route different / action repeat | route different / action different | joint no-progress rate |
|---|---:|---:|---:|---:|---:|
| success | 0.101 | 0.141 | 0.124 | 0.595 | 0.028 |
| failure | 0.303 | 0.116 | 0.108 | 0.473 | 0.106 |

## Failure modes

The failure-only comparisons are descriptive because event-like modes are rare.

| mode | n | joint-return mean | physical-return mean | action-route mean |
|---|---:|---:|---:|---:|
| lifted_not_placed | 42 | 0.765 | 0.991 | 0.377 |
| misplaced | 25 | 0.789 | 0.993 | 0.411 |
| never_grasped | 29 | 0.692 | 0.971 | 0.401 |
| partial | 198 | 0.775 | 0.994 | 0.389 |
| reached_then_lost | 13 | 0.748 | 0.984 | 0.361 |

## Slip and repeated-attempt events

- Off-goal lift-loss proxy events: n=92 (59 failure, 33 success).
  Same-seed return minus cross-seed physical-match median: route -0.271 (p=1.0000), action -0.627 (p=1.0000).
  Median physical distance: within-episode return 0.834 versus cross-seed control 0.074.
- Repeated lift attempts: n=3 (0 failure, 3 success).
  Same-seed return minus cross-seed physical-match median: route -0.238 (p=0.9456), action -0.154 (p=0.9456).
  Median physical distance: within-episode return 1.074 versus cross-seed control 0.232.

## Interpretation limits

- The physical proxy is MuJoCo qpos, not a stored RGB or vision embedding.
- This model has recorded state-token and action-token HB routing, not an independent visual-token MoE stream.
- Every within-episode comparison holds the rollout noise seed fixed, but exact snapshot re-inference across multiple seeds requires images or simulator replay that are absent here.
- Lift-loss is a chunk-level kinematic proxy, not a semantic slip annotation.
- Routing/action recurrence is associative. A causal memory intervention would require online replay.
- Terminal-window and failure-mode results must not be called early warning.
