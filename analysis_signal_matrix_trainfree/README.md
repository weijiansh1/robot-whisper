# Train-free MoE trap signal matrix

This directory evaluates eight pre-specified routing signals on the two
HiMoE-VLA long-horizon corpora without fitting an outcome predictor:

1. cross-chunk weighted-Jaccard recurrence;
2. lag-2/3/4 periodicity excess over lag 1;
3. late-flow routing volatility;
4. routing acceleration in Hellinger coordinates;
5. normalized gate entropy;
6. top-1/top-2 probability margin;
7. state-action routing gap;
8. deterministic macro-state stickiness.

Labels are used only for grouped evaluation, permutation tests, and physical
onset alignment. Corpus A uses dense physical Trap onsets. Corpus B uses a
query-resolution proxy that is first validated against corpus A's dense
onsets. No logistic model, neural probe, PCA, or k-means is fit.

The primary onset control is every trajectory without the event at the same
absolute query and within the same snapshot/init-state group. A separate
success-only analysis is retained as a sensitivity check. Loop/static tests
use one maxT family across both event types, all eight signals, and all five
pre-registered leads. Each routing signal is also rank-residualized against
the existing d9 Hellinger mobility baseline.

Run from the workspace root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python analysis_signal_matrix_trainfree/run.py
```

The first run streams the full routing Zarr stores and writes small scalar
caches. Pass `--rebuild-cache` to regenerate them. The main outputs are
`report.zh.md`, `loop_precursor_vs_actions.png`,
`onset_aligned_four_signals.png`, and `signal_matrix.png`. Exact aggregate and
type-stratified tests are in `onset_alignment.csv` and
`type_onset_alignment.csv`; their group-level inputs are retained alongside
them for audit.

The train-free physical Snapshot-Fork collector is
`recovery_window_collect.py`. It identifies loop onset from physical query
trajectories, restores the exact simulator/controller state at pre-specified
offsets, and evaluates paired fresh-noise continuations. The completed
init-state 3 run is under `recovery_window/scan_init3_seed20260903`; summary
outcomes and signal-control percentiles are in `recovery_window_summary.csv`
and `recovery_signal_audit.csv`. This run observed 0/8 recoveries at each of
offsets -4, -2, and 0, so it supports the routing signal as a sensor but does
not establish fresh-noise recovery or timing benefit.
