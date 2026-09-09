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

## Rich functional event pilot

The request-gated follow-up restores 96 frozen states from task 8 in
`cache_new`, captures functional HB state for matched static-event and healthy
episodes at leads -4 and -2, and analyzes 23 qualified pairs per lead.  The
analysis was frozen in `RICH_EVENT_PROTOCOL.md` before opening the functional
snapshot values.

Capture with one model migrated across three physical GPUs:

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  analysis_moe_execution_signals/capture_rich_event_features.py \
  --gpus 0,1,2 \
  --inputs analysis_moe_execution_signals/rich_event_inputs.npz \
  --input-audit analysis_moe_execution_signals/rich_event_inputs_audit.json \
  --out-dir analysis_moe_execution_signals/rich_event_functional_32d \
  --sketch-dim 32
```

The capture command refuses to start unless every selected GPU's power limit
equals its hardware maximum and anonymous-memory headroom is at least 3 GiB.
In the recorded run, GPUs 0-2 were all at 500 W / 500 W, all 96 recorder-on
actions were bitwise equal to recorder-disabled actions, and `oom_kill` stayed
at 13. Two rows failed the independently frozen simulator-restoration tolerance
and were excluded pairwise.

Run the frozen inference:

```bash
python analysis_moe_execution_signals/analyze_rich_event_features.py
```

Eight of ten primary cells passed joint matched-pair maxT at 0.05. At both
leads, static-event states had lower routed authority, cancellation, and
normalized expert disagreement, plus higher routed/shared cosine. Relative
32-D routed-output flow change did not pass at either lead. The four successful
features are strongly coupled and should be interpreted as one candidate
"weak routed branch / routed-shared convergence" axis, not four independent
discoveries. This frozen-state result is observational and does not establish
rerouting benefit or a safe action prefix.

RGB frames were regenerated from saved float32 MuJoCo state because the compact
episode NPZ files do not contain source images. Therefore this is a frozen-state
recapture, not an exact historical-request replay. Historical action agreement
is retained as a diagnostic: 95/96 rows had maximum absolute error below 0.042,
while one near-threshold control row flipped its gripper action.

## MoE-triggered short-horizon recovery pilot

`RECOVERY_HORIZON_PROTOCOL.md` freezes a paired causal test at the lead -4
task-8 states. Each of 23 eligible static-event states and its matched healthy
control receives both potential outcomes from the same full simulator and
controller snapshot:

- `h10`: execute every predicted 10-action chunk in full;
- `h2_burst10`: reobserve after two actions for the first ten physical actions,
  then return to `h10`.

Both arms have a 200-action opportunity budget and use common flow noise at
shared physical offsets. The clean run is in `horizon_recovery_clean/`; its 46
states and 92 arms were collected as three sequential shards on GPU 0 at 500 W
/ 500 W. Every paired state passed full-state, policy-input, first-noise, and
first-action equality checks. All shard-local `oom_kill` counters remained 24.
An earlier shared-cgroup-OOM attempt is marked invalid and is not analyzed.

Run the frozen analysis with:

```bash
PYTHONPATH=. python analysis_moe_execution_signals/analyze_horizon_recovery.py \
  --run-root analysis_moe_execution_signals/horizon_recovery_clean \
  --num-shards 3 \
  --out-dir analysis_moe_execution_signals
```

The ten-step short burst did not recover any selected Trap state: both arms
succeeded on 0/23 event states, for a paired effect of 0.0 percentage points
(95% bootstrap CI 0.0 to 0.0; joint maxT p=1). On healthy controls, `h10`
succeeded on 21/23 and the short burst on 19/23. The nominal +8.7-point
event-minus-control difference in differences is therefore caused entirely by
two control harms, not by a rescue (maxT p=0.5).

The frozen routed/shared-cosine alarm covered 19/23 events and falsely alarmed
on 3/23 controls. Applying the burst only on alarm still produced 0/23 event
successes, while reducing control success from 21/23 to 20/23. It did not lower
event recurrence either: both arms had 2/23, with one recurrence prevented and
a different one induced. Mean query cost increased by 4.0 calls on events and
4.70 on controls.

This separates detection from treatment response. Routed/shared convergence is
a strong prognostic sensor in these frozen states, but it supplies neither a
recovery direction nor evidence that a brief replanning burst is effective.
The negative result is specific to five h=2 replans over the first ten actions;
it does not test a longer alarm cooldown, persistent receding-horizon control,
retraction, or an intervention that changes expert computation. The next
recovery study should fit and validate treatment benefit separately from Trap
risk, with healthy-state harm and query cost kept as explicit constraints.
