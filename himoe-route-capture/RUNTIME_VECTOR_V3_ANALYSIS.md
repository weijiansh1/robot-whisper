# Runtime Vector v3 remaining-correction analysis

This protocol is frozen around the full-vector probe in
`himoe_hb_activation_flow_v3`. It is retrospective predictive evidence only.
It does not establish that an expert causes a correction or validate an online
stopping rule.

## Estimand and clock

The HB5/d0 activation is computed with `x0` as the flow-state input. The
operational decision is made only after d0 completes, when `x1` is also
observable. The primary target is future live-7 path energy after that decision:

```text
y = sqrt(sum_{tau=1..9,token,live7} (x[tau+1]-x[tau])^2 / (9*10*7))
```

The complete 70-dimensional live-7 `x10-x1` net vector is a separately fitted
secondary endpoint. It reports vector RMSE, within-state geometry Spearman,
net-RMS RMSE, and K8 geometry coverage. `x1-x0` remains instrumentation only.

## Frozen feature ladder

The operational feature sets are nested:

1. B0: full `10x24` `x0 + x1` (including the 17 latent flow dimensions).
2. Diagnostic steps add HB5 input hidden, then HB5 shared output.
3. B1: B0 + hidden + shared + full 32-way router probabilities, selected
   expert-ID one-hot, selected weights scattered into expert-ID coordinates,
   and the actual merged routed vector `r_runtime=sum_k w_runtime,k v_k`.
4. Task-local extraction arm A appends expert-ID raw vectors, raw RMS, and
   explicit contributions `c_k=w_k v_k`. This arm is secondary because expert
   IDs do not align across checkpoints.
5. Expert probability geometry first sets `w_norm=w_runtime/sum(w_runtime)`.
   Primary invariant arm B appends the absolute-unit statistic
   `S_LOO,t=sqrt(sum_k w_norm,k ||delta_k||^2/1024)` for all ten tokens and
   `sqrt(mean_t S_LOO,t^2)` for the chunk, where
   `delta_k=(r_norm-w_norm,k v_k)/(1-w_norm,k)-r_norm`.
6. Secondary invariant Gram arm appends descending eigenvalues of raw `Gv` and
   normalized-contribution `Gc` (four each), routed RMS `R`, cancellation `C`,
   and `S_LOO`, retaining the token axis. Runtime weight-sum error is audited;
   it cannot become a probability-geometry feature.

Here `v_k` is the signed expert MLP output after `down_proj` and before gate
weighting. It is not an intermediate neuron activation from `gate_proj` or
`up_proj`.

Raw expert coordinates are `(action token, expert ID, hidden coordinate)`.
Top-k slots are neither exchangeable examples nor an ordinal expert encoding.
Permuting slots together with IDs and vectors leaves the representation
unchanged.

High-dimensional B1 and ID-aware families use target-independent signed
CountSketch. The 11-dimensional S_LOO block and 110-dimensional Gram block are
kept exact, without projection or collisions. B1 is one frozen block. An expert
arm is independently represented and concatenated, so the first W columns of
every augmented design are exactly B1. Scaling is training-only and every arm
selects its own ridge alpha by inner group-disjoint CV.

Because exact S_LOO adds 11 readout coordinates, a matched non-expert control
adds exact runtime merged-routed RMS for ten tokens plus its chunk RMS. S_LOO
must beat this equal-width control; otherwise the result is only a
capacity/scale-extraction effect.

The report repeats high-dimensional projections for five frozen seeds and
requires the decision gate at every seed; median seed selection is forbidden.
Width 128 is preregistered. Widths 64 and 256 are descriptive sensitivity runs;
an effect confined to compressed B1 is only a low-capacity extraction advantage.

## Split and admission contract

The state group is `(benchmark, task_id, init_state_id)`. The repeated-noise
identity is `(flow_noise_seed, candidate_id)`, not just `flow_noise_seed`: the
current K16 grid uses one client RNG seed and repeats candidate draw positions
across states. Admission verifies the complete `x0[10,24]` is identical within
each seed identity to `1e-6`. Outer and inner folds are Cartesian: training
shares neither state nor repeated-noise identity with held rows.

Inference requires at least three crossed state pools and three seed groups,
with each state observed under at least three seeds and each seed under at least
three states. Five task identities, at least eight states per task, and at least
exactly 16 repeated candidate identities are required. The task set is frozen
to `libero_goal:{0,3}`, `libero_10:8`, and `libero_spatial:{5,7}`. Each
task/checkpoint is
projected, tuned, fitted, and evaluated independently; only task metrics are
then equally macro-averaged. No expert-ID coordinate ever pools across
checkpoints. The current K1 smoke is therefore a schema check only, and the CLI
returns `REFUSED` without writing a summary.

For several client invocations sharing one server store, give each rollout a
non-overlapping `rollout_flow_lead.py --query-base`. The complete grid maps
clients to three suite stores automatically:

```bash
python analyze_runtime_vector_v3.py \
  --grid-root runs/runtime-vector-grid-v3 \
  --no-sensitivity \
  --out analysis/runtime-vector-v3/summary.json
```

`--grid-root` maps client configs to
`server/{goal,spatial,long}/activation_flow.zarr` and checks shared-store query
IDs for collisions.

The preregistered primary effect is task-local
`(RMSE_B1-RMSE_B1+SLOO)/RMSE_B1`. It requires 5/5 task effects positive, an
equal-task macro effect of at least 2%, and a task-stratified state-bootstrap
95% CI lower bound above zero, at every projection seed. The bootstrap resamples
both state and candidate axes from the fixed 8x16 squared-error grid. One common
candidate-column draw is reused across all five tasks, while state draws remain
task-local. S_LOO must also beat the exact11 matched control in all tasks.
No p-value or common-seed cross-task permutation is computed or claimed.

Secondary action geometry and K8 use
`pred_final=x1_live7+predicted_correction` against `actual_final=x10_live7`.
K8 references deterministic PAM BUILD+SWAP on actual final-action geometry; it
is not called an exact oracle. Current K8 is a stratified crossfit-ensemble
diagnostic because candidates in different held-seed folds have different
models. It is not deployable and cannot satisfy a pruning gate without a new
independent train-noise grid.

An untrained mechanism section reports within-state Spearman for S_LOO chunk,
runtime merged-routed RMS chunk, and d0 velocity RMS against future-path energy.
It is descriptive and never substitutes for the B1 incremental gate.

No result is bundled because no admissible five-task v3 capture exists yet.

## Verification

`test_runtime_vector_v3.py` covers both target axes, state/seed leakage, repeated
`x0`, expert-ID scattering, exact S_LOO/permutation invariance, strict appended
block nesting, held-target isolation, and K1 refusal.
`test_rollout_flow_lead.py` covers non-overlapping query-base allocation.
