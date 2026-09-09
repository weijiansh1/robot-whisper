# Runtime HB5/d0 expert-vector test

> **Post-run invalidation (2026-08-23):** the frozen analysis was executed as
> written, then independently audited and withdrawn.  Its scalar path-energy
> target is almost a deterministic nonlinear function of the already observed
> d0 velocity, but B1 did not contain that exact norm and its linear CountSketch
> readout could not construct it.  This document remains immutable protocol
> history, not a valid positive or null result.  See
> `../raw-internal-activation-correction/PREREGISTRATION.md` for the corrected
> target and raw-internal-activation definition.

This protocol is frozen before collecting the five-task runtime grid. The prior
K=1 Goal smoke test was used only to validate tensor identities and numerical
reconstruction; it cannot inform any scientific metric below.

## Estimand

At the end of denoise step zero, test whether the actual HB5 expert decomposition
improves prediction of the natural, unperturbed future path energy

`y_path = sqrt(sum_(tau=1..9) ||x_(tau+1)-x_tau||^2 / (9*10*7))`.

The secondary vector target is the live-dimension correction `x10 - x1`.
Both are beyond observables that are already available after round zero.
`x1 - x0` is excluded as a scientific target because the probed expert output
helps construct that same update.

Amendment before grid collection: a blind method audit changed path energy from
secondary to primary because net displacement can hide later corrections that
cancel. No admissible runtime-grid row had been collected or inspected at the
time of this amendment.

## Fresh grid

- Five task/checkpoint combinations: Goal task 0, Goal task 3, Libero-10 task 8,
  Spatial task 5, and Spatial task 7.
- Eight fixed initial states per task: 1, 5, 11, 18, 25, 32, 40, and 48.
- Sixteen common candidate-noise columns per state, generated from RNG seed 8000.
  The repeated-noise identity is `(RNG seed, candidate_id)`; the episode-level
  `flow_noise_seed` alone is constant and must not be used as the seed group.
- One exact-observation query per state; no environment action is taken between
  sibling candidates.
- Probe cell fixed to HB5, denoise step 0, ten action tokens.

## Feature hierarchy

1. Action baseline: all 24 model dimensions of `x0 + x1` (equivalently current
   latent plus the observed d0 update). The 17 non-executed dimensions remain
   model inputs and therefore cannot be omitted from a fair baseline.
2. Strong non-expert baseline: action baseline plus pre-MoE hidden, shared-branch
   output, full router probabilities, selected IDs/weights, and the already
   merged routed vector `r`. Including `r` asks whether decomposition adds
   information beyond the ordinary MoE block output.
3. Expert model: strong baseline plus actual selected, pre-gate, signed expert
   output vectors. Expert-ID-aware and expert-permutation-invariant blocks are
   evaluated separately; checkpoint-specific expert IDs are never pooled.
4. Descriptive expert geometry: routed mixture, unweighted output magnitude,
   disagreement, cancellation, routed/shared conflict, and contribution
   concentration. These scalars cannot replace the main incremental comparison.

The frozen primary expert feature is the expert-permutation-invariant local
leave-one-out sensitivity. For each action token, with `v_k = E_k(h)`,
`c_k = w_k v_k`, and `r = sum_k c_k`, define

`delta_k = w_k (r - v_k) / (1 - w_k)`

and `S_LOO^2 = sum_k w_k ||delta_k||^2 / 1024`. The ten token values and their
predefined chunk RMS are appended to the strong baseline in absolute raw units.

The broader expert model explicitly forms `c_k = w_k E_k(h)` before any linear readout.
It also includes the raw and weighted 4 x 4 expert Gram matrices and exact local
drop-one changes `(r - c_k) / (1 - w_k) - r`, summarized without depending on
top-k slot order. Supplying `w_k` and `E_k(h)` as separate linear features is not
an adequate test because a linear ridge cannot construct these products or
pairwise directions.

Every feature transform, centering/scaling statistic, and ridge regularizer must
be fitted using the training fold only. Feature blocks use fixed training-fold
scale normalization. Hyperparameters are chosen using group-aware inner folds.
The primary high-dimensional readout width is fixed to 128 before collection;
64 and 256 are descriptive capacity sensitivities and cannot replace the
primary result. A gain that vanishes at width 256 is labelled a low-capacity
extraction advantage, not additional information beyond hidden state.
Because appending exact `S_LOO` also adds 11 readout coordinates, an equal-width
nonexpert control is required: ten tokenwise merged-routed RMS values plus their
chunk RMS. `S_LOO` must outperform this matched control before its gain can be
attributed to expert decomposition rather than added capacity or exact scale.

## Generalization split

Every outer prediction holds out both an initial-state identity and a candidate
noise block. Neither identity may occur in training. Models are fitted separately
for each task/checkpoint combination. Metrics are computed within exact-observation
pools and then aggregated first over states and then equally over tasks. A
state-only split is reported only as a repeated-seed diagnostic.

## Primary metrics and decision

The primary metric is held-out error for scalar future-path energy. The expert
claim requires strong-baseline-plus-`S_LOO` to reduce normalized error in all
five tasks, by at least 2% in the equal-task macro, with a task/state-cluster
bootstrap 95% interval excluding zero. Secondary metrics are within-pool
pair-distance Spearman for the live 10 x 7 remaining-correction vector and K=8
coverage of the final `x10` candidate cloud. Coverage is an operational
corroboration, not a substitute primary endpoint: it must improve in all five
tasks and by at least 1% in the equal-task macro before claiming a pruning rule.

The bootstrap resamples both the eight state rows and sixteen repeated-noise
columns within each task, then averages task effects equally. A state-only
interval is conditional on this fixed noise grid and cannot establish new-noise
generalization.

Amendment before inspecting outcomes: Cartesian state+seed OOF predictions for
different seed blocks come from different fitted probes, so they cannot be
combined into one deployable K16 geometry. This first grid can test natural path
energy and within-block geometry, but the K8 pruning gate remains unresolved.
A deployable K8 test requires a second independent 16-noise training grid so one
probe, trained with both state and noise identities excluded, predicts the whole
held K16 pool. No cross-model K8 union may satisfy the pruning gate.

Failure of this gate is a null result for natural early commitment and forbids a
candidate-pruning claim. Mechanistic prediction of an explicit expert
intervention may then be tested as a separately labelled causal-fragility target;
it must not be substituted post hoc for this endpoint.
