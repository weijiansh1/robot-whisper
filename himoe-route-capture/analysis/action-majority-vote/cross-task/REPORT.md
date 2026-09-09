# Action-space majority voting on the five-task 16x32 grid

Post-hoc shadow audit. The selector uses only the first action chunk of each
candidate episode; the outcome is eventual rollout success. One seed fixes both
the first query and every later replanning query, so a selector here chooses a
noise stream and success is an episode-start shadow label.

| task | suite | pool success rate | mixed states |
|---|---|---:|---:|
| open_the_middle_drawer_of_the_cabinet | libero_goal | 1.000 | 0/16 |
| open_the_top_drawer_and_put_the_bowl_inside | libero_goal | 0.918 | 7/16 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | libero_long | 0.578 | 13/16 |
| pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | libero_spatial | 0.977 | 8/16 |
| pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | libero_spatial | 0.928 | 12/16 |

| budget N | selected - random | 95% CI | full-pool oracle - random | unique seeds | max seed share |
|---:|---:|---|---:|---:|---:|
| 2 | -0.15 pp | [-0.60, +0.27] pp | +10.74 pp | 31 | 0.062 |
| 4 | -0.38 pp | [-1.10, +0.31] pp | +10.74 pp | 32 | 0.082 |
| 8 | -0.77 pp | [-2.17, +0.59] pp | +10.74 pp | 32 | 0.158 |
| 16 | -1.34 pp | [-3.88, +1.12] pp | +10.74 pp | 29 | 0.291 |
| 32 | -0.51 pp | [-5.78, +4.65] pp | +10.74 pp | 6 | 0.512 |

`N=2` is degenerate by construction: the two-candidate medoid is a tie and
reduces to the lowest-seed pick.

