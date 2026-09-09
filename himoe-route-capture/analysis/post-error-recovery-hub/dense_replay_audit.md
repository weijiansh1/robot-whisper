# Dense-contact audit of selected post-loss proxies

This purposive audit covers every terminal-success candidate in the main
query-level cohort plus five failure candidates spanning all four event-bearing
tasks. It is not a random sample and must not be used to estimate prevalence.

A proxy is contact-confirmed only when the target has bilateral finger contact
for at least three consecutive simulator substeps before `drop_query`, followed
by no target-finger contact at that query's first physical point. Regrasp uses
the same three-substep bilateral criterion after the landmark.

| task | ep | outcome | pre bilateral run | confirmed loss | future run | regrasp | verdict |
|---|---:|---|---:|---:|---:|---:|---|
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 47 | failure | 25 | true | 0 | false | confirmed_loss_no_regrasp_terminal_failure |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 449 | failure | 1 | false | 0 | false | rejected_no_stable_bilateral_precontact |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 457 | success | 0 | false | 56 | true | rejected_no_stable_bilateral_precontact |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 3 | success | 14 | true | 57 | true | confirmed_loss_regrasp_terminal_success |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 28 | failure | 12 | true | 0 | false | confirmed_loss_no_regrasp_terminal_failure |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 204 | failure | 12 | true | 0 | false | confirmed_loss_no_regrasp_terminal_failure |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 463 | failure | 23 | true | 0 | false | confirmed_loss_no_regrasp_terminal_failure |

## Result

- Audited: 7; contact-confirmed loss: 5; rejected proxy: 2.
- Among confirmed losses: 1 terminal success, 4 terminal failure.
- Contact-confirmed regrasp after a confirmed loss: 1.
- Maximum query-state replay drift: 1.189e-07 (tolerance 1.0e-05).

The ramekin success candidate later performs a real bilateral grasp, but
its query-level landmark was not preceded by stable bilateral contact; it
therefore cannot validate an initial slip. The stove episode 3 candidate is
the only audited contact-confirmed loss followed by regrasp and task success.

## Contact-validated matched micro-case

Failure-minus-success recurrence differences are shown only for a
contact-confirmed pair sharing task and initial state.

| task | init | pair | route Hellinger | top-4 Jaccard | action cosine | hidden cosine |
|---|---:|---|---:|---:|---:|---:|
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0 | 28 vs 3 | +0.0033 | +0.0095 | +0.6600 | +0.1367 |

There is only one such pair. It is a case study, not an effect estimate.
The post-loss physical states are not matched, so it does not isolate routing.
