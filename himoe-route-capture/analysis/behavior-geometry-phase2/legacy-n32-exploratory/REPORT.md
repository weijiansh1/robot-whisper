# Exploratory route--outcome geometry check

This is a legacy-proxy feasibility check, not confirmatory evidence.

| Action stratum | Eligible snapshots | Same pairs | Different pairs | Mean `different - same` d_R | 95% snapshot bootstrap CI | One-sided sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| far | 3 | 658 | 588 | 0.000281 | [-0.000947, 0.001838] | 0.500000 |
| near | 1 | 1385 | 55 | 0.000968 | CI unavailable | 0.500000 |

Both inequalities hold at the point-estimate level: **true**.
Both snapshot-bootstrap intervals are above zero: **false**.

## Metric and inference

`d_R` is RMS Hellinger distance over aligned 8 HB layers x 10 denoise rounds x 10 action tokens, using each site's normalized full 32-way router distribution. Contrasts are computed within snapshot and snapshots receive equal weight.

## Limitations

- Outcome sameness/difference is the phase-1 drawer-endpoint quantile proxy, not repeated-continuation Q equivalence.
- Action and outcome thresholds were fit post hoc on these same 20 snapshots.
- All snapshots come from one task and only five source episodes; there is no held-out task validation.
- Several snapshots have degenerate proxy outcomes, reducing the number eligible for within-snapshot contrasts.
- Pair mining and this follow-up reuse the same sample; selected effects require independent confirmation.
