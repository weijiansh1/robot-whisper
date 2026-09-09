# Failed versus successful rollout MoE curves

Each point reads only the current action chunk. Shaded intervals in the plot bootstrap initial states within task and average the four tasks equally.

## Bottom line

The clearest growing comparison is distance from the successful MoE manifold: the failure-minus-success percentile gap is not distinguishable from zero through k4, becomes +0.113 at k5 (95% CI [+0.022, +0.200]), and grows to +0.245 at k8 (95% CI [+0.198, +0.290]).

A decoder direction frozen at k8 does not separate the groups through k6 (gap +0.001, 95% CI [-0.331, +0.300]). It separates only at k7 and k8, reaching +1.196 at k8 (95% CI [+0.722, +1.722]). Thus anomaly relative to the appropriate success state accumulates, but one fixed late-failure direction is not already growing in the early chunks.

## Population curves

| k | risk success | risk failure | risk gap | manifold success | manifold failure | manifold gap | frozen-axis success | frozen-axis failure | frozen gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.513 | 0.454 | -0.059 | 0.508 | 0.484 | -0.025 | -0.058 | -0.013 | +0.045 |
| 1 | 0.489 | 0.469 | -0.020 | 0.498 | 0.460 | -0.038 | -1.677 | -1.716 | -0.039 |
| 2 | 0.488 | 0.586 | +0.098 | 0.496 | 0.485 | -0.011 | -0.664 | -0.776 | -0.113 |
| 3 | 0.483 | 0.564 | +0.081 | 0.505 | 0.515 | +0.009 | -1.206 | -1.352 | -0.146 |
| 4 | 0.507 | 0.514 | +0.007 | 0.491 | 0.532 | +0.041 | -1.490 | -1.512 | -0.022 |
| 5 | 0.495 | 0.502 | +0.008 | 0.489 | 0.602 | +0.113 | -1.254 | -1.425 | -0.170 |
| 6 | 0.480 | 0.605 | +0.125 | 0.480 | 0.608 | +0.128 | -0.046 | -0.045 | +0.001 |
| 7 | 0.477 | 0.621 | +0.144 | 0.471 | 0.669 | +0.198 | +0.480 | +1.092 | +0.612 |
| 8 | 0.462 | 0.631 | +0.169 | 0.460 | 0.705 | +0.245 | -0.108 | +1.088 | +1.196 |

## Reading the signals

- Per-query risk percentile uses an independently cross-fitted routed-full decoder at each k, then ranks scores only within the same initial state and held-out seed fold.
- Success-manifold anomaly is label-free at scoring time: larger means farther from the successful routed-computation centroid learned from training seeds at that k.
- Frozen k8 failure axis fits a routed-full decoder at k8 inside each task and fold, then applies that exact preprocessing and direction to every earlier chunk. It tests whether the same within-task late-failure direction grows over time.
- Percentile curves show relative separation, not raw MoE magnitude. The frozen-axis panel retains one common score direction but remains an associative offline readout.

The k8 frozen-axis implementation reproduces the ordinary cross-fitted k8 routed-full probabilities with maximum absolute difference 0.
