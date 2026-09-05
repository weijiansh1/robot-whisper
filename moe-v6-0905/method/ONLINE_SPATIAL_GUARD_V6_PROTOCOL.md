# Online Spatial Guard v6 Protocol

## Scope

This iteration keeps the frozen v5 long-suite guard, restores the frozen v4
lock on `libero_goal` and `libero_object`, and searches a replacement lock head
only for `libero_spatial`. The L5 instability head remains frozen at v4.

The restriction is motivated by the v5 observation that the global non-long
L15 profile improved spatial lead time while creating concentrated object and
goal false positives. External 8B has already been inspected and is not a
pristine holdout; all v6 external numbers are post-hoc replay evidence.

## Development-only selection

The sweep reads only the 14,800-trajectory development cohort and its labels.
For every task and held-out initial state, thresholds use the other 49 initial
states (392 trajectories). No outcome is used for threshold calibration.

Candidate representations include individual layers, front/back/all layer
aggregates, layer dispersion, front-back gap, and causal absolute derivatives.
Mobility representations are searched in both low- and high-mobility
directions. The grid is:

```text
64 oriented representations
x 5 smoothing widths       (1, 2, 3, 4, 6)
x 8 confirmation counts    (1, 2, 3, 4, 6, 8, 10, 12)
x 20 calibration quantiles (0.50 ... 0.995)
= 51,200 candidates
```

Width 8 is excluded before scoring because short-rollout tasks leave fewer
than 32 finite cross-fit calibration peaks. No missing peak is imputed.

Only the spatial lock is replaced for candidate scoring. Long tasks use the
v5 long guard; goal and object tasks use v4; every suite retains the v4 L5
instability head.

Eligibility is frozen before external replay:

```text
overall non-risk FPR <= 0.25%
precision            >= 90%
risk recall          >= 75%
max suite FPR        <= 1%
max task FPR         <= 2.5%
```

Eligible candidates are ranked by early-4 recall, total recall, precision,
FPR, and deterministic simplicity tie-breakers. GPU results only shortlist
candidates; the top shortlist is recomputed with the NumPy float32 reference
path before selection.

## External replay

After `selection.json`, `deployment_profiles.npz`, and first-alarm arrays are
written, the evaluator joins external outcomes. Each external task threshold
is calibrated from its 400-trajectory outcome-blind reference cache. The online
monitor consumes only the current `hb_router_probs` snapshot and task profile.

## Compute

The full candidate grid is split across two CUDA workers. The intended launch
maps logical CUDA devices 0 and 1 to physical GPUs 6 and 7:

```bash
CUDA_VISIBLE_DEVICES=6,7 python experiments/evaluate_spatial_guard_v6_gpu.py
```
