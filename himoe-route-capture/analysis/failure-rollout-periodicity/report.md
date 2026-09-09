# Failed-rollout MoE periodicity audit

Data source: `VLA_MUI_HUB/cache/HiMoE-VLA`, run `right-16x32`.
The source stores were opened read-only. AUC > 0.5 means the metric is
larger on failures after comparing only within task x init-state strata.

## Bottom line

The available data support **failure-associated route persistence**, not a
general periodic MoE loop. In the equal-prefix test, failure AUC is 0.802 for actual top-4 lag-1 overlap and 0.720 for soft-route lag-1 similarity. Both become still stronger in terminal windows (0.979 and 0.931).

Periodic prominence is lower on failures in the primary equal-prefix test (AUC 0.379 hard, 0.371 soft), and no equal-prefix or same-length terminal episode passes the aligned hard+soft cycle criterion. An exploratory failure-half scan finds only 3/307 route-cycle candidates, all in one task; candidate details and matched success calibration are below.

## Coverage

| task | episodes | failures | equal prefix | tested periods | AS gate |
|---|---:|---:|---:|---:|---|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 512 | 0 | 12 | 2-3 | constant |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 512 | 42 | 17 | 2-5 | constant |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 512 | 216 | 35 | 2-8 | constant |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 512 | 12 | 9 | 2-2 | constant |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 512 | 37 | 11 | 2-3 | constant |

## Equal-prefix primary test

| metric | failure AUC | permutation p | BH q | raw failure mean | raw success mean |
|---|---:|---:|---:|---:|---:|
| HB actual top-4 lag-1 overlap | 0.802 | 0.0002 | 0.0003 | 0.2940 | 0.2509 |
| HB actual top-4 periodic prominence | 0.379 | 0.0002 | 0.0003 | -0.0003 | -0.0059 |
| HB actual top-4 return above lag 1 | 0.456 | 0.1102 | 0.1446 | -0.0672 | -0.0613 |
| HB soft route lag-1 similarity | 0.720 | 0.0002 | 0.0003 | -0.7979 | -0.9733 |
| HB soft route periodic prominence | 0.371 | 0.0002 | 0.0003 | -0.0088 | -0.0420 |
| HB soft route return above lag 1 | 0.657 | 0.0002 | 0.0003 | -0.3135 | -0.3173 |
| hard+soft aligned route-cycle candidate | 0.500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| route cycle + aligned physical return | 0.500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| route+state+action aligned loop candidate | 0.500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| top-4 lag-1 residual after behavior | 0.651 | 0.0002 | 0.0003 | 0.0019 | -0.0005 |
| top-4 periodic residual after behavior | 0.390 | 0.0002 | 0.0003 | -0.0005 | 0.0002 |
| soft lag-1 residual after behavior | 0.519 | 0.4811 | 0.5943 | 0.0009 | -0.0002 |
| soft periodic residual after behavior | 0.372 | 0.0002 | 0.0003 | -0.0007 | 0.0002 |

Mixed-stratum sample: 275 failures + 1005 successes in 40 task-scene strata; 5000 within-stratum permutations.
Raw means are pooled without stratification and can disagree in direction
with the task-scene-stratified AUC when task scales differ.

## Same-length terminal-window test

| metric | failure AUC | permutation p | BH q | raw failure mean | raw success mean |
|---|---:|---:|---:|---:|---:|
| HB actual top-4 lag-1 overlap | 0.979 | 0.0002 | 0.0004 | 0.3818 | 0.2548 |
| HB actual top-4 periodic prominence | 0.551 | 0.0564 | 0.0911 | 0.0017 | -0.0052 |
| HB actual top-4 return above lag 1 | 0.635 | 0.0002 | 0.0004 | -0.0529 | -0.0630 |
| HB soft route lag-1 similarity | 0.931 | 0.0002 | 0.0004 | -0.5232 | -0.9474 |
| HB soft route periodic prominence | 0.506 | 0.8208 | 0.9576 | -0.0054 | -0.0429 |
| HB soft route return above lag 1 | 0.862 | 0.0002 | 0.0004 | -0.2357 | -0.3235 |
| hard+soft aligned route-cycle candidate | 0.500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| route cycle + aligned physical return | 0.500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| route+state+action aligned loop candidate | 0.500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| top-4 lag-1 residual after behavior | 0.671 | 0.0002 | 0.0004 | 0.0032 | -0.0009 |
| top-4 periodic residual after behavior | 0.553 | 0.0510 | 0.0892 | 0.0003 | -0.0001 |
| soft lag-1 residual after behavior | 0.517 | 0.5387 | 0.6654 | 0.0002 | -0.0001 |
| soft periodic residual after behavior | 0.536 | 0.1788 | 0.2346 | 0.0024 | -0.0007 |

Mixed-stratum sample: 275 failures + 1005 successes in 40 task-scene strata; 5000 within-stratum permutations.
Raw means are pooled without stratification and can disagree in direction
with the task-scene-stratified AUC when task scales differ.

## Failed episodes: early vs late

Each failed episode is split into two non-overlapping equal halves. Positive
delta means the metric increased near timeout.

| task | half window | top-4 lag1 delta (p) | top-4 periodic delta (p) | soft periodic delta (p) |
|---|---:|---:|---:|---:|
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 15 | 0.1645 (0.0000) | 0.0021 (0.0211) | 0.0093 (0.0001) |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 26 | 0.1095 (0.0000) | 0.0018 (0.0000) | -0.0020 (0.0001) |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 11 | 0.2183 (0.0022) | -0.0002 (0.8139) | 0.0165 (0.0029) |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 11 | 0.1383 (0.0000) | 0.0027 (0.0328) | 0.0198 (0.0000) |

High-specificity aligned candidates in the late failure half:

| task | route only | route + physical state | route + state + action |
|---|---|---|---|
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | [] (0/42) vs success [] (0/470) (p=1.0000) | [] (0/42) vs success [] (0/470) (p=1.0000) | [] (0/42) vs success [] (0/470) (p=1.0000) |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | [142, 155, 494] (3/216) vs success [] (0/296) (p=0.0745) | [494] (1/216) vs success [] (0/296) (p=0.4219) | [494] (1/216) vs success [] (0/296) (p=0.4219) |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | [] (0/12) vs success [] (0/38) (p=1.0000) | [] (0/12) vs success [] (0/38) (p=1.0000) | [] (0/12) vs success [] (0/38) (p=1.0000) |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | [] (0/37) vs success [] (0/475) (p=1.0000) | [] (0/37) vs success [] (0/475) (p=1.0000) | [] (0/37) vs success [] (0/475) (p=1.0000) |

Candidate detail (`top1` equality is over the eight HB layers on the
state token; stability lists are terminal window lengths that remain
positive):

| episode | scene / noise | hard return | soft return | top1 lag1 -> lag2 | route-stable windows | joint-stable windows |
|---:|---|---:|---:|---:|---|---|
| 142 | 13 / 1014 | 0.0060 | 0.0008 | 0.790 -> 0.823 | [18, 20, 22, 24, 26] | [18, 24] |
| 155 | 13 / 1027 | 0.0063 | 0.0051 | 0.820 -> 0.880 | [18, 20, 22, 24, 26] | [] |
| 494 | 49 / 1014 | 0.0059 | 0.0009 | 0.890 -> 0.938 | [18, 20, 22, 24, 26] | [18, 20, 22, 24, 26] |

## Interpretation constraints

- `periodic prominence` is a local autocorrelation peak at lag 2-8; the
  maximum lag is shortened when an episode window cannot show three cycles.
- `return above lag 1` must be positive for a route to resemble a past
  state more than its immediately previous state. The aligned candidate
  flags require agreement between actual top-4 and soft routing, then
  optionally simulator state and action.
- The terminal comparison is descriptive: successful and failed episodes
  end in different task phases. The equal-prefix test is the primary one.
- Association is not causal. Routing and outcome can both reflect the same
  physical trajectory; the behavior-residual rows are only a control for
  the measured action and simulator recurrence.
