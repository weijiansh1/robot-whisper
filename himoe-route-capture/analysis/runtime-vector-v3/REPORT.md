# Runtime-vector v3 frozen-grid report

> **Withdrawn as an inferential result (2026-08-23).** A post-run independent
> audit found that the primary target is almost determined by the exact
> non-MoE scalar `RMS_live7(x1-x0)` (within-state macro Spearman `.987`), while
> B1 supplied only signed `x0/x1` coordinates through a 128-dimensional linear
> CountSketch.  A linear ridge cannot construct that norm, and the eight B1
> families were compressed into the same 128 coordinates.  The resulting B1
> RMSE is 6--11 times worse than the exact d0-RMS control.  On that minimal
> corrected control, adding S_LOO worsens RMSE in all five tasks.  Therefore the
> frozen gate below is retained only as an audit trail; it is not a valid test of
> commitment, expert increment, or pruning.  The replacement protocol is in
> `../raw-internal-activation-correction/PREREGISTRATION.md`.

## Superseded Decision

The preregistered width-128 S_LOO increment does **not** pass. None of the five
frozen projection seeds satisfies all of: 5/5 task improvement, at least 2%
equal-task macro relative RMSE improvement, positive two-axis bootstrap lower
bound, and 5/5 superiority to the matched exact11 routed-RMS control.

This is a negative result for the frozen readout protocol. It is not evidence
that expert combinations have no information under every nonlinear model or
under a different target.

## Post-run Baseline Audit

Using the same Cartesian state/candidate split (`split_seed=20260823`), a
one-coordinate exact baseline containing only
`RMS_live7(x1-x0)` gives the following held-out errors.  Appending the exact
11-coordinate S_LOO block makes every task worse.

| task | withdrawn B1 RMSE | exact d0-RMS RMSE | exact d0-RMS + S_LOO | relative S_LOO improvement |
|---|---:|---:|---:|---:|
| `libero_10:8` | .01215 | .001787 | .002719 | -52.16% |
| `libero_goal:0` | .01089 | .001462 | .001748 | -19.51% |
| `libero_goal:3` | .01374 | .001192 | .001266 | -6.22% |
| `libero_spatial:5` | .01126 | .001707 | .003492 | -104.53% |
| `libero_spatial:7` | .01285 | .001518 | .001591 | -4.82% |

This audit is not the replacement commitment experiment.  It demonstrates why
the old target/B1 pairing was invalid and provides a direct check that the old
S_LOO feature is not rescued by the missing nonlinear control.

## Admission

- Tasks, in report order: `libero_10:8`, `libero_goal:0`, `libero_goal:3`,
  `libero_spatial:5`, `libero_spatial:7`.
- Each task contains the exact `8 states x 16 repeated candidates` grid.
- Three checkpoints are evaluated task-locally; expert-ID coordinates are never
  pooled across checkpoints.
- Full `x0[10,24]` repeat error across states/tasks/checkpoints: `0.0`.
- Maximum runtime top-k weight-sum error: `0.005859375`; S_LOO/Gram geometry
  renormalizes weights before probability calculations.

## Frozen Gate

Task order in every vector below is the admission order above.

| sketch seed | task relative RMSE improvements | macro | 95% state x candidate bootstrap CI | tasks improved | S_LOO better than matched control | pass |
|---:|---|---:|---|---:|---:|---|
| 20260823 | `+1.42%, +0.12%, +0.73%, -12.19%, +0.00%` | -1.98% | [-4.40%, -0.03%] | 4/5 | 2/5 | no |
| 20260824 | `+0.79%, -3.64%, +1.86%, +0.90%, +0.89%` | +0.16% | [-1.39%, +1.93%] | 4/5 | 3/5 | no |
| 20260825 | `-0.43%, +1.26%, +0.09%, +1.83%, +2.29%` | +1.01% | [-0.66%, +2.50%] | 4/5 | 4/5 | no |
| 20260826 | `-6.57%, -0.83%, -0.63%, +3.01%, +4.77%` | -0.05% | [-4.56%, +3.29%] | 2/5 | 0/5 | no |
| 20260827 | `+5.57%, -3.36%, +0.34%, -2.11%, +1.78%` | +0.45% | [-3.17%, +3.90%] | 3/5 | 3/5 | no |

The worst projection-seed macro effect is `-1.98%`. Width 64/256 sensitivity
was intentionally not run in this first formal pass (`--no-sensitivity`).

## Standalone Mechanism

These are untrained within-state Spearman correlations with primary future-path
energy, averaged equally over states and then tasks. They do not replace the B1
incremental gate.

| scalar | per-task Spearman | equal-task macro |
|---|---|---:|
| d0 velocity live7 RMS | `.987, .979, .989, .986, .993` | `.987` |
| S_LOO chunk | `.684, .354, .330, -.046, -.034` | `.258` |
| runtime merged-routed RMS chunk | `.779, .302, .303, -.118, -.115` | `.230` |

The primary target is therefore almost monotonic in the already-observed d0
velocity scale on this grid. S_LOO is mixed across checkpoints and is slightly
negative on both Spatial tasks.

## K8 Boundary

K8 remains unresolved and non-deployable. OOF predictions from different held
seed folds are not compared directly: candidates are allocated to fold-local PAM
selections by largest remainder and then unioned. This is a crossfit-ensemble
diagnostic, not a single model scoring a K16 cloud. Descriptive K8 macro changes
range from `-0.40%` to `+0.34%` across sketch seeds and never meet the provisional
five-task/+1% threshold. No pruning claim is made.

No p-value or common-seed cross-task permutation was computed. Full machine
output, folds, paired error grids, bootstrap gates, endpoint metrics, and
diagnostics are in `summary.json`.
