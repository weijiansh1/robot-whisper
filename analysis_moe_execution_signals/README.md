# Actual-execution MoE signal test

This analysis tests the first low-cost part of the proposed MoE information
ladder on the same A/B long-horizon corpora and physical onsets as
`analysis_signal_matrix_trainfree`.

- F0: frozen full-softmax routing signals.
- F1: actual Top-4/Top-5 boundary, mass outside Top-4, and normalized Top-4
  execution entropy.
- F2: support Jaccard and sparse execution-weight Hellinger churn, using the
  IDs selected by the runtime.
- F3: all-HB-layer summaries, action-token slopes, and early-state versus
  late-action cross-query propagation.

The scalar detector capacity remains one.  Features never read labels.  Labels
are used only by same-group, same-absolute-query matched AUC, group bootstrap,
and group sign-flip maxT tests.  The residual variant first removes the two
previously replicated F0 loop signals (`late_flow_volatility` and
`route_acceleration`) with label-free within-group rank regression.

Run from the workspace root:

```bash
PYTHONPATH=. python analysis_moe_execution_signals/run.py --nperm 2000
```

Against an already-loaded server, verify that the actual Top-4 trace is
available and that enabling it leaves the fixed-noise action bitwise unchanged:

```bash
PYTHONPATH=. python analysis_moe_execution_signals/audit_live_dispatch.py \
  --port 8803 \
  --out analysis_moe_execution_signals/live_dispatch_gpu0_long.json
```

The online protocol returns action-token Top-4 IDs and weights only.  It does
not replace `himoe-route-capture/himoe_functional_recorder.py` for state-token,
logit, or functional-output capture.

Analyze one or more rich snapshots with the route/function four-quadrant audit:

```bash
PYTHONPATH=. python analysis_moe_execution_signals/analyze_functional_snapshots.py \
  himoe-route-capture/runs/functional-noop-gpu0/snapshot.npz \
  himoe-route-capture/runs/functional-noop-gpu1/snapshot.npz \
  --out analysis_moe_execution_signals/functional_snapshot_summary.json \
  --report analysis_moe_execution_signals/functional_snapshot_report.md
```

Use `--rebuild-cache` only when feature definitions or source Zarr stores
change.  Every Zarr store is opened read-only.  Stored expert IDs, not a new
Top-K over fp16 probabilities, define actual support.

This stage cannot measure routed/shared authority, expert cancellation, or
functional expert geometry because the existing route stores do not contain
runtime expert outputs.  It also cannot estimate the intervention effect of
safe-prefix truncation; its per-token result is the gate for a paired snapshot
rollout rather than a correction claim.
