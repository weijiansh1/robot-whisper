# Published-style action-space majority voting on same-state fork pools

Retrospective endpoint-proxy audit of already-executed candidates. The
selector picks the candidate closest to the consensus of the other sampled
candidates; no outcome is used to fit anything. Every contrast is reduced
within snapshot before the snapshot bootstrap.

## Pool `k32` (32 candidates x 20 snapshots)

Primary outcome: `drawer_progress_after_continuation` (higher is better).

| budget N | selector | selected - random | 95% CI | oracle recovery | unique picks | max share |
|---:|---|---:|---|---:|---:|---:|
| 2 | action_medoid | +0.000218 | [-0.000337, +0.000909] | +0.045 | 31 | 0.062 |
| 2 | action_centroid_nn | +0.000003 | [-0.000273, +0.000311] | +0.001 | 32 | 0.040 |
| 2 | action_rms_medoid | +0.000218 | [-0.000337, +0.000909] | +0.045 | 31 | 0.062 |
| 4 | action_medoid | +0.000695 | [+0.000034, +0.001539] | +0.080 | 32 | 0.044 |
| 4 | action_centroid_nn | +0.000899 | [+0.000092, +0.001918] | +0.104 | 32 | 0.045 |
| 4 | action_rms_medoid | +0.000760 | [+0.000125, +0.001559] | +0.088 | 32 | 0.042 |
| 8 | action_medoid | +0.001530 | [-0.000369, +0.003837] | +0.131 | 32 | 0.058 |
| 8 | action_centroid_nn | +0.001807 | [-0.000112, +0.004187] | +0.155 | 32 | 0.062 |
| 8 | action_rms_medoid | +0.002004 | [+0.000106, +0.004354] | +0.172 | 32 | 0.053 |
| 16 | action_medoid | +0.001789 | [-0.001227, +0.005240] | +0.128 | 32 | 0.083 |
| 16 | action_centroid_nn | +0.002027 | [-0.000915, +0.005478] | +0.145 | 32 | 0.091 |
| 16 | action_rms_medoid | +0.002956 | [-0.000205, +0.006857] | +0.212 | 32 | 0.070 |
| 32 | action_medoid | +0.003171 | [-0.001848, +0.010175] | +0.200 | 14 | 0.150 |
| 32 | action_centroid_nn | +0.001543 | [-0.001932, +0.005530] | +0.097 | 14 | 0.150 |
| 32 | action_rms_medoid | +0.003383 | [-0.001745, +0.010420] | +0.213 | 13 | 0.150 |

Budget scaling: per-snapshot slope of the effect on `log2(N)` over `N=[4, 8, 16, 32]`.

| selector | slope per log2(N) | 95% CI |
|---|---:|---|
| action_medoid | +0.000769 | [-0.000703, +0.002831] |
| action_centroid_nn | +0.000215 | [-0.000793, +0.001303] |
| action_rms_medoid | +0.000882 | [-0.000630, +0.002934] |

| budget N | oracle - random | 95% CI | subsets | exhaustive |
|---:|---:|---|---:|---|
| 2 | +0.004826 | [+0.001810, +0.008304] | 496 | True |
| 4 | +0.008653 | [+0.003271, +0.014851] | 35960 | True |
| 8 | +0.011671 | [+0.004387, +0.020126] | 20000 | False |
| 16 | +0.013940 | [+0.005223, +0.024238] | 20000 | False |
| 32 | +0.015873 | [+0.005931, +0.027694] | 1 | True |

### Full-pool action medoid on every declared outcome

| outcome | higher is better | selected - random | 95% CI |
|---|---|---:|---|
| `drawer_delta_after_continuation` | False | -0.003171 | [-0.010175, +0.001848] |
| `drawer_delta_chunk` | False | -0.000000 | [-0.000002, +0.000001] |
| `drawer_progress_after_continuation` | True | +0.003171 | [-0.001848, +0.010175] |
| `object_qpos_moved` | False | +0.000856 | [+0.000009, +0.001718] |
| `success_in_window` | True | +0.014063 | [+0.000000, +0.042188] |

### Outcome support

| outcome | mixed snapshots |
|---|---:|
| `drawer_delta_after_continuation` | 14 |
| `drawer_delta_chunk` | 4 |
| `drawer_progress_after_continuation` | 14 |
| `object_qpos_moved` | 20 |
| `success_in_window` | 1 |

### Degeneracy and dispersion

Median within-pool action distance: mean `0.1730`, range `0.1183`-`0.2293`.

Dispersion versus full-pool medoid effect: Spearman `-0.329` (`p=0.1564`, 20 snapshots).

## Pool `k8` (8 candidates x 20 snapshots)

Primary outcome: `drawer_progress_after_continuation` (higher is better).

| budget N | selector | selected - random | 95% CI | oracle recovery | unique picks | max share |
|---:|---|---:|---|---:|---:|---:|
| 2 | action_medoid | -0.001481 | [-0.003205, -0.000091] | -0.280 | 7 | 0.250 |
| 2 | action_centroid_nn | -0.000476 | [-0.001476, +0.000232] | -0.090 | 8 | 0.173 |
| 2 | action_rms_medoid | -0.001481 | [-0.003205, -0.000091] | -0.280 | 7 | 0.250 |
| 4 | action_medoid | +0.000421 | [-0.000909, +0.002026] | +0.044 | 8 | 0.188 |
| 4 | action_centroid_nn | +0.000055 | [-0.001099, +0.001521] | +0.006 | 8 | 0.201 |
| 4 | action_rms_medoid | +0.000644 | [-0.001266, +0.002766] | +0.067 | 8 | 0.144 |
| 8 | action_medoid | +0.002981 | [-0.002552, +0.009692] | +0.233 | 8 | 0.250 |
| 8 | action_centroid_nn | +0.001526 | [-0.004636, +0.008986] | +0.119 | 8 | 0.200 |
| 8 | action_rms_medoid | +0.004583 | [-0.001677, +0.011602] | +0.358 | 8 | 0.300 |

Budget scaling: per-snapshot slope of the effect on `log2(N)` over `N=[4, 8]`.

| selector | slope per log2(N) | 95% CI |
|---|---:|---|
| action_medoid | +0.002559 | [-0.001976, +0.008248] |
| action_centroid_nn | +0.001471 | [-0.003609, +0.007809] |
| action_rms_medoid | +0.003940 | [-0.000723, +0.009905] |

| budget N | oracle - random | 95% CI | subsets | exhaustive |
|---:|---:|---|---:|---|
| 2 | +0.005287 | [+0.001971, +0.009046] | 28 | True |
| 4 | +0.009586 | [+0.003684, +0.016418] | 70 | True |
| 8 | +0.012820 | [+0.005155, +0.022063] | 1 | True |

### Full-pool action medoid on every declared outcome

| outcome | higher is better | selected - random | 95% CI |
|---|---|---:|---|
| `drawer_delta_after_continuation` | False | -0.002981 | [-0.009692, +0.002552] |
| `drawer_delta_chunk` | False | +0.000018 | [-0.000001, +0.000054] |
| `drawer_progress_after_continuation` | True | +0.002981 | [-0.002552, +0.009692] |
| `success_in_window` | True | +0.012500 | [+0.000000, +0.037500] |

### Outcome support

| outcome | mixed snapshots |
|---|---:|
| `drawer_delta_after_continuation` | 15 |
| `drawer_delta_chunk` | 4 |
| `drawer_progress_after_continuation` | 15 |
| `success_in_window` | 1 |

### Degeneracy and dispersion

Median within-pool action distance: mean `0.1921`, range `0.1448`-`0.2209`.

Dispersion versus full-pool medoid effect: Spearman `-0.106` (`p=0.6563`, 20 snapshots).

## Interpretation limits

- `N=2` medoid is a tie by construction and reduces to the lowest-id pick;
  its row is reported only to keep the budget axis complete.
- Each candidate carries one shared-CRN continuation realization, not a
  repeated-continuation Monte Carlo `Q`.
- Binary success is mixed in almost no snapshot, so the drawer proxies carry
  the entire primary signal and no closed-loop success claim is available.
- Both pools come from one LIBERO-Goal task; there is no held-out task.
- Subset expectations at large `N` are Monte Carlo over common subsets; the
  random and oracle baselines remain exact.

