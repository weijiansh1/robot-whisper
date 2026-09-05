# Scene8 HB-MoE token dynamics pilot

## Experimental unit and split

- Task: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`.
- Discovery: init state 0, 32 rollouts, 14 success / 18 timeout failure.
- Confirmation: init state 42, 32 rollouts, 14 success / 18 timeout failure.
- A rollout is one statistical sample. Control steps, 10 denoise iterations, layers, and tokens are repeated measurements.
- Three fixed horizons are tested with an 8-control trailing window: t7, t12, and t34.
- Features include aligned T0-T10 lag 1-5 distances, action mean/std, T10-T1, Gram spectrum, and period-2 to period-5 curvature.

## Main result

| Horizon | Meaning | Aggregate discovery maxT | Fixed-direction confirmation | Per-cell confirmation |
|---:|---|---:|---:|---:|
| t7 | before displayed pose split | p=0.655 | AUC=0.500, p=0.511 | AUC=0.575, Bonferroni p=0.722 |
| t12 | before displayed route split | p=0.162 | AUC=0.468, p=0.622 | AUC=0.546, Bonferroni p=1.000 |
| t34 | before length/survivor split only | p<0.0001 | AUC=0.726, p=0.014 | AUC=0.893, Bonferroni p=0.00015 |

The t7/t12 null remains after removing the layer/flow average and scanning 91,361 variable cells. The aggregate did not hide a stable early local signal.

At t34, timeout failures have substantially lower longitudinal movement. Examples:

- Top-4 action-token routing `action_std`, lag 2: confirmation oriented AUC 0.921.
- Cell-localized Top-4 routing `T10 / L12 / f2`, lag 3: confirmation oriented AUC 0.893.
- Cell-localized hidden `action_std / L12 / f9`, lag 5: confirmation oriented AUC 0.849.

This is a broad late-stage low-change or stalled-trajectory phenotype, not an early precursor.

## No periodic return

The apparent period-2/period-3 features are curvature scores. The underlying hidden T0 distances remain monotone:

| State | Outcome | lag1 | lag2 | lag3 | lag4 | lag5 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | success | 0.2367 | 0.5167 | 0.7222 | 0.8543 | 0.9250 |
| 0 | timeout failure | 0.0799 | 0.1542 | 0.2145 | 0.2668 | 0.3006 |
| 42 | success | 0.2545 | 0.5848 | 0.8097 | 0.9181 | 0.9466 |
| 42 | timeout failure | 0.1572 | 0.3390 | 0.4815 | 0.5709 | 0.6404 |

No failure curve returns toward an earlier state. The best localized periodicity cell has confirmation p=0.006 before correction, p=0.054 after the nine category-by-horizon tests, and its lag curve is still monotone. The supported description is slowing/stagnation, not oscillation.

## Behavior control

Simple action dynamics at t34 are also highly predictive. Two behavior variables (proprio speed and action-mean speed) explain:

- 95.1% / 88.9% of the discovery/confirmation variance of the Top-4 `action_std` score;
- 97.2% / 95.8% of the soft-route T0 lag-1 score.

After this post-hoc linear control, neither feature has a discovery association (p=0.698 and p=0.690). Thus the late MoE signal is largely a readout of the already-stalled physical/action trajectory.

## Conclusion

1. There is no confirmed HB-MoE precursor before the displayed success/failure trajectory separation in these two initial states.
2. Timeout failures do enter a strong late low-change regime across aligned tokens, layers, and denoise iterations.
3. The data do not support a period-2 to period-5 MoE cycle.
4. MoE-only state can detect the late stalled phenotype here, but it cannot establish a MoE-specific or causal terminal trap.

Both states reuse noise seed IDs 1000-1031. The confirmation is cross-state, not unseen-seed confirmation.
