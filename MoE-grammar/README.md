# MoE Routing Grammar

This directory turns the proposal in `idea.md` into leakage-resistant experiments on the
`VLA_MUI_HUB` routing corpora. The legacy audit maps each 10-step query to a discrete word; the
current audit keeps a continuous multitrack query phenotype and models the complete causal
prefix. It uses all 40 LIBERO tasks in the two `right-50x8*-20260903` segments; the original
five-task `right-16x32` experiment remains as a smaller reference.

The implementation is read-only with respect to the source hub. Generated features and models
stay under this directory.

## Corrected Continuous Audit (Current)

The feedback-driven v2 audit supersedes the earlier claim that Trap monitoring can be reduced
to high healthy-grammar surprisal.

- Cross-layer expert numbers are no longer compared directly. The replacement measures
  dispersion of permutation-invariant entropy, margin, mass, consensus, and rank profiles.
- The main path uses 22 continuous query phenotypes. It does not re-differentiate native
  velocity/acceleration, whiten 2,187 dimensions into PCA-24, or commit to hard GMM words.
- A continuous VAR(4) gains 0.2174 bits/phenotype over a current-query Gaussian on held-out
  successes. A 12-state soft HMM gains only 0.0118 bits/phenotype over its clock-only control.
  History is predictive, but most HMM phase prediction is not prefix-specific.
- At a matched 4.7% Scene8 success-episode FPR, the three-channel observer detects 5.1% of 197
  physical stasis events by q-3 and 5.1% by onset. Current-query innovation reaches 5.1% by
  onset and has the best matched AUC (0.592). The return model has only 0.544 AUC for healthy
  three-query persistence. Freeze, recurrence, and low-surprisal components have matched AUC
  0.497, 0.479, and 0.420. This is a detector no-go, not evidence of reliable early warning.
- Same-snapshot candidate reranking has a preliminary positive emission signal: full-prefix
  scoring selects 3 successful candidates versus 1.5 expected at random across 28 K=4
  snapshots. However, only three independent trunks are informative, the one-sided exact
  sign-flip p-value is 0.125, and full-prefix, clock-only, last-query-only, and no-prefix all
  have exactly the same paired AUC (0.786). The signal cannot be attributed to global syntax.

The continuous observer report is in
[results-three-channel-v2/REPORT.zh.md](results-three-channel-v2/REPORT.zh.md). The causal
candidate audit is in
[results-candidate-reranking-v2/REPORT.zh.md](results-candidate-reranking-v2/REPORT.zh.md).
The sections below retain prior audits for provenance rather than presenting them as the final
monitoring result.

## Expanded Full-40 Result

The expanded corpus contains 32,000 episodes, 508,023 queries, 30,904 successes, and 1,096
failures. Five state-blocked folds train on 30 states, calibrate on 10, and test on 10, so every
state and all 16 of its noise branches are tested exactly once.

- Healthy task+position+ordered-history lowers continuous mixture NLL by 0.2091 bits/query
  relative to task+position; the five-fold range is [0.2040, 0.2169].
- The specifically ordered second-word effect is real but small: ordered context beats the same
  two-word bag by 0.00371 bits/query, and every fold's state-blocked interval excludes zero.
- A fixed-tokenizer learning curve shows why more data matters: with roughly 3,081 healthy
  episodes the ordered effect is only 0.00021 bits/query and appears in 3/5 folds; with roughly
  18,542 it is 0.00371 and appears in 5/5 folds. First-word dependence was already stable at the
  smaller scale.
- The old pooled-percentile CUSUM was miscalibrated for nonstationary task/position scores.
  Task+position empirical CDFs followed by normal-score Page-CUSUM improve q7 within-task/state
  AUC from 0.563 to 0.600 and q12 from 0.587 to 0.622.
- Almost all of that detection gain is phase calibration, not grammar: history improves over a
  phase-only detector by only 0.006 AUC at both q7 and q12; the grammar residual reaches only
  0.560/0.577.
- q3 is effectively chance. At q7, Page-CUSUM has 0.600 AUC, 7.15% healthy FPR, and 13.78%
  failure recall. At q12 it has 0.622 AUC and 33.76% recall at 7.56% FPR, but q12 excludes early
  successful completions and is survivor-conditioned.
- A three-seed supervised causal GRU does not uncover a large routing-only precursor on one
  strict test split: routing AUC is 0.594 at q7 and 0.646 at q12, while behavior-only reaches
  0.633 and 0.672. Routing has no demonstrated increment over behavior.

The five-fold result is in
[results-full40-crossfit/REPORT.zh.md](results-full40-crossfit/REPORT.zh.md). Per-fold models and
measurements are under `results-full40*`; the supervised upper-bound audit is in
[results-prefix/REPORT.zh.md](results-prefix/REPORT.zh.md).

## Global-Prefix Grammar Result

The local two-word audit was followed by a healthy-only causal Transformer over every preceding
query (maximum episode length 52). At position q it predicts the current routing word from
`0..q-1`; after observing q it predicts q+1. Its continuous score marginalizes over all GMM words,
so a prediction of a nearby routing prototype is penalized less than an unrelated word.

- Across all 30,904 cross-fitted successful episodes, full-prefix prediction improves over the
  task+position+local-two-word model by 0.338 bits/query from q1 onward and 0.376 bits/query from
  q4 onward (state-blocked 95% CI [-0.390, -0.362] for q4+).
- Holding the latest two words fixed while reversing only the older prefix worsens q4..q12 NLL
  by 0.139 bits/query (95% CI [0.132, 0.147]). This is direct evidence for ordered information
  beyond the local two-word context.
- This predictive syntax does **not** add failure discrimination. At q7, full-prefix current-word
  surprisal has AUC 0.591 versus 0.592 for local-2 (difference -0.002, 95% CI [-0.009, 0.006]); at
  q12 the difference is -0.006 (95% CI [-0.018, 0.005]).
- q3 remains chance. At q7 the full-prefix detector recalls only 10.6% of eventual failures at
  4.6% healthy FPR; q12 recalls 13.4% at 3.4% FPR. It is a weak risk signal, not a reliable online
  intervention trigger.
- CUSUM is not used in this audit. Query scores are calibrated against train-success at the same
  task and absolute position; the full prefix energy is the causal cumulative predictive NLL,
  and cal-success alone determines each horizon threshold.

The complete protocol and results are in
[results-global-prefix/REPORT.zh.md](results-global-prefix/REPORT.zh.md), with machine-readable
measurements in `results-global-prefix/summary.json`.

## Open-World Observer Result

The monitoring audit is now formulated as healthy-dynamics modeling rather than a closed
Loop/Static classifier. Every query is represented by 22 continuous MoE phenotypes covering
sharpness, token consensus, effective rank, switching, acceleration, and layer disagreement.
The generic observer is fit on successful episodes only; the physically labeled Scene8 stasis
events are kept as a separate known anchor. The corpus contains no reliable Loop onset, so no
Loop labels are invented.

A non-neural phase-HMM maintains a forward belief over the complete observed routing prefix and
scores the current chord as `-log p(z_q | z_<q)`. A clock-only HMM is the matched no-history
control. K in `{4, 8, 12}` is selected solely by cross-fitted held-out-success next-chord NLL;
K=12 wins. Relative to the clock, the full prefix improves q4+ density by 0.0165
bits/phenotype, and its routing-only filtered phase has Spearman 0.897 with offline normalized
progress. Thus a measurable healthy prefix structure exists without a Transformer.

That structure is not yet an effective early-warning detector:

- The strict global version uses only MoE phenotypes online: no task ID, action, vision,
  outcome, or future query. At q3 its within-task/state AUC is 0.505 (95% CI [0.487, 0.523])
  with 6.2% failure recall at 4.8% healthy FPR.
- At q7 AUC is 0.529 with 6.9% recall at 5.0% FPR. At q12 AUC reaches 0.593 and 30.0% recall
  at 5.1% FPR, but q12 is a later survivor-conditioned readout.
- On 197 cross-fitted physical Scene8 stasis events, the full-prefix EWMA detects 10.2% by the
  physical onset and 11.7% by onset+3, at 7.1% success-episode FPR. The known-stasis template is
  not better (5.6% onset recall at 9.5% FPR).
- Providing task ID improves q7/q12 AUC to 0.563/0.610, confirming substantial task nuisance,
  but that diagnostic is not counted as strict MoE-only deployment performance.
- Persistent off-manifold windows are clustered only into candidate routing phenotypes. A miss
  is `observer-unseen`, not evidence that the event is intrinsically MoE-silent.

The selected protocol and compact conclusion are in
[results-open-world-selected/REPORT.zh.md](results-open-world-selected/REPORT.zh.md). The full
strict audit is in `results-open-world-global-k12`; task-conditioned results are under
`results-open-world-task-k12`.

## Initial Five-Task Result

The data support **multiscale temporal structure and within-task conditional sequence grammar**,
with important limits on invariance and control use.

- On 2,253 cross-fitted healthy test episodes, task+position+ordered-history NLL is 1.056
  bits/token versus 1.178 for task+position alone. The paired gain is 0.123 bits/token with a
  state-blocked 95% CI of [0.096, 0.151], providing held-out evidence for history information
  beyond task and query clock.
- The gain decomposes cleanly: the previous word contributes 0.079 bits/token, adding a second
  word contributes 0.044, and retaining its order beats the same two-word bag by a smaller 0.0052
  bits/token (95% CI [0.0016, 0.0096]).
- A crossed state+noise-seed holdout reproduces the full effect at 0.141 bits/token (two-axis 95%
  CI [0.120, 0.163]) and the ordered-over-bag increment at 0.0057 bits/token (95% CI
  [0.0029, 0.0086]); all 16/16 held-out cells agree in direction for both.
- Zero-shot task transfer fails. With the entire test task excluded from PCA, tokenizer, and
  grammar fitting, position+history is 0.179 bits/token worse than position alone (task-bootstrap
  95% CI [0.022, 0.389]); only 1/5 tasks improves. The ordered-over-bag direction remains negative
  in 4/5 tasks, but its CI crosses zero.
- PST NLL is 1.413 bits/token versus 4.855 for a unigram, 2.754 for absolute position, and 1.488
  for an unordered recent-word bag.
- Real query order beats within-episode query shuffle by 5.442 bits/token. PST beats the bag
  control by 0.0747 bits/token (`p=0.00032`, exact state-blocked sign test).
- A true raw-flow shuffle, followed by feature recomputation, changes 92.6% of word assignments
  and raises lexical NLL by 138.3 nats/query.
- This supports a task-conditioned grammar, not a task-invariant one. The model establishing the
  primary effect receives task identity as an oracle; the unconditioned PST still mixes task and
  phase effects.
- Better healthy prediction does not imply useful anomaly monitoring. At q7/q12,
  task+phase+history changes failure AUC relative to task+phase by -0.001/0.000; scene8
  task-calibrated pre-onset recall is 0.096 at 0.078 healthy FPR.
- Duration worsens healthy next-word NLL. Its apparent gain is confined to late q34 failure
  readout and does not provide useful pre-onset scene8 detection.
- In scene8 at q34, cross-query flattening and terminal L15/f9 support synchronization are both
  associated with physical stasis (within-state AUC 0.768 and 0.717), while within-query flow
  flattening points in the opposite direction (AUC 0.268). This supports the multitrack
  `Fl_query || Sy_L15,f9` representation rather than a literal `Fl Fl Sy Sy` string. It is a late
  association, not an early precursor or causal MoE effect.

The full phase-conditioned result and failure controls are in
[results/REPORT.zh.md](results/REPORT.zh.md). The independent split audit is in
[results-dual-axis/REPORT.zh.md](results-dual-axis/REPORT.zh.md); both directories include
machine-readable `summary.json` files. Zero-shot cross-task transfer is isolated in
[results-task-holdout/REPORT.zh.md](results-task-holdout/REPORT.zh.md).

## Experiment Contract

For the current full-40 audit, the two non-overlapping seed segments provide 50 states x 16 seeds
for each of 40 tasks. Compact float16 extraction uses all 508,023 queries. In each fold, a
balanced sample of at most 2,000 healthy queries per task fits RobustScaler/PCA24/GMM64, while
all roughly 18.5k healthy training episodes fit task+position grammars. Train-success fits the
conditional score CDF, calibration-success sets thresholds, and unseen test states determine
all reported detection metrics.

The following contract describes the original five-task audit.

The source corpus contains 5 tasks, 16 initial states, and 32 flow-noise branches per state:
2,560 episodes and 51,308 routing queries in total. There are 2,253 successes and 307 failures.

Splitting is by global `init_state_id`. Each of four folds has disjoint train, calibration, and
test states, so all 32 sibling noise branches from a `task + init_state` root stay together.

- Robust scaling, PCA, GMM codebooks, and grammar counts use successful train-state episodes only.
- `K in {16, 32}` is selected using healthy calibration NLL only. All four folds select K=32.
- Detection CDFs and CUSUM thresholds use successful calibration-state episodes only.
- Every reported episode prediction is cross-fitted on an unseen initial state.
- Statistical intervals resample the 16 initial-state blocks, not individual sibling rollouts.

The main split prevents root-branch leakage. A second, stricter audit partitions both axes into
four groups and evaluates all 16 state-group x seed-group test cells. For each cell, training
excludes every episode sharing a test state or a test seed. It predicts every successful episode
exactly once and confirms that repeated seed IDs do not explain the history gain.

A third audit leaves out each complete task from preprocessing, tokenization, and grammar fitting.
It does not support zero-shot transfer, so the positive result should be described as a robust
within-task conditional grammar, not a task-invariant language.

## Representation

The corrected main representation is the 22-dimensional continuous phenotype described above.
Its cross-layer term compares invariant layer profiles, so independent expert permutations in
different layers leave it unchanged.

### Legacy Discrete Representation

For each `[query, layer, flow, token, expert]` routing tensor, the extractor computes:

- normalized entropy, Top-1/Top-2 margin, Top-1 mass, and Top-4 mass;
- soft Hellinger token consensus and authoritative Top-4 support consensus;
- token-expert effective rank;
- flow Hellinger velocity, Top-4 switching, and acceleration;
- cross-layer invariant-profile dispersion (the old raw expert-ID disagreement was invalid).

The legacy path yields a `10 x 81` chord tensor per query. It then applies levels, first
differences, and second differences to every track, including already-derived velocity and
acceleration, forming a 2,187-dimensional descriptor. This path remains only for reproducing
the older reports; it is not the corrected detector representation.

The sequence comparison includes unigram, absolute position, position+history, bigram, fixed
fourth-order Markov, unordered bag-context, PST, PST plus explicit duration, and task-conditioned
oracle controls. The primary nested model first conditions on task and absolute position, then
uses up to two previous words; unsupported contexts fall back to the clock-only distribution.
One-word and unordered two-word ablations separate local dependence, context size, and order.

## Reproduce

The default paths point to `/home/jovyan/work/himoe-vla/VLA_MUI_HUB`. First inspect the current
task indexes:

```bash
python -m moe_grammar.extract_features --list
```

The checked-in results used GPUs 0, 1, and 2 with batch size 2048. Run these in three terminals:

```bash
python -u -m moe_grammar.extract_features --task-indexes 2 --device cuda:0 --batch-size 2048
```

```bash
python -u -m moe_grammar.extract_features --task-indexes 0,1 --device cuda:1 --batch-size 2048
```

```bash
python -u -m moe_grammar.extract_features --task-indexes 3,4 --device cuda:2 --batch-size 2048
```

The extractor will not overwrite existing features unless `--force` is supplied. Run the formal
cross-fitted experiment with:

```bash
python -u -m moe_grammar.run_experiments
```

Run the pre-specified K=32 crossed state+noise-seed audit with:

```bash
python -u -m moe_grammar.run_dual_axis_audit
```

Run the fixed K=32 leave-one-task-out audit with:

```bash
python -u -m moe_grammar.run_task_holdout_audit
```

Extract the expanded corpus compactly across three GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m moe_grammar.extract_features --cache-name cache_new --run-id 'right-50x8*-20260903' --output-dir artifacts/features-full40 --ordered-only --storage-dtype float16 --batch-size 8192 --num-shards 3 --shard-index 0
```

Repeat with `CUDA_VISIBLE_DEVICES=1 --shard-index 1` and
`CUDA_VISIBLE_DEVICES=2 --shard-index 2`. Run a fold and aggregate all five completed folds with:

```bash
python -u -m moe_grammar.run_full40_audit --fold 0
python -m moe_grammar.aggregate_full40_audit
```

The supervised causal-prefix control is reproduced with
`prepare_prefix_dataset`, three `train_prefix_gru` seeds, and `evaluate_prefix_ensemble`.

For the healthy-only full-prefix audit, prepare one compact dataset per existing full-40 fold,
train the five models (the training commands can be distributed over GPUs 0/1/2), then aggregate:

```bash
python -m moe_grammar.prepare_global_prefix_dataset \
  --grammar-model results-full40/full40_model.joblib \
  --output artifacts/global-prefix-fold0.npz
python -m moe_grammar.train_healthy_prefix_lm \
  --dataset artifacts/global-prefix-fold0.npz \
  --output-dir results-global-prefix/fold0 --device cuda:0 --batch-size 1024
python -m moe_grammar.evaluate_global_prefix --output-dir results-global-prefix
```

Repeat the first two commands for folds 1..4 with the matching `results-full40-foldN` model.

Build and audit the healthy-only open-world observer with GPU-accelerated batched HMM forward
filtering:

```bash
python -m moe_grammar.prepare_open_world_dataset
python -m moe_grammar.run_open_world_audit \
  --output-dir results-open-world-global-k12 \
  --conditioning global --phase-states 12 --device cuda:0
python -m moe_grammar.select_open_world_audit \
  --task-conditioned results-open-world-task-k12
```

The K=4/8/12 healthy-only selection runs can be assigned independently to GPUs 0/1/2. On the
current H20 host, one full 508k-query HMM forward takes about 2.2 seconds; the complete five-fold
audit, including covariance fitting and 1,000 state-block bootstrap draws, takes about two
minutes per candidate.

Run the corrected continuous observer on any idle GPU, then audit routed same-snapshot
candidates on another idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m moe_grammar.run_three_channel_audit \
  --device cuda:0 --output-dir results-three-channel-v2
CUDA_VISIBLE_DEVICES=1 python -m moe_grammar.run_candidate_reranking_audit \
  --models-dir results-three-channel-v2/models \
  --split-summary results-open-world-global-k12/summary.json \
  --device cuda:0 --output-dir results-candidate-reranking-v2
```

The continuous VAR Ridge solve, prediction, and Gaussian quadratic forms use CUDA in this path;
small indexing, calibration quantiles, and report aggregation remain on CPU.

Verification:

```bash
ruff check moe_grammar tests
python -m pytest -q
```

## Outputs

- `artifacts/features/*.npz`: ordered and recomputed flow-shuffle chord tensors, metadata, and
  small behavior-control features (267 MiB total).
- `results/models/fold_*.joblib`: selected train-only preprocessing, K=32 codebook, phase-aware
  grammars, calibration references, and thresholds.
- `results/word_prototypes.json`: fold-0 word prototypes with explicitly heuristic phenotype tags.
- `results/overview.png`: healthy prediction and fixed-horizon outcome overview.
- `results/summary.json`: complete machine-readable protocol and measurements.
- `results/REPORT.zh.md`: concise Chinese interpretation and guardrails.
- `results-dual-axis/summary.json`: complete 16-cell state+seed holdout protocol and effects.
- `results-dual-axis/REPORT.zh.md`: concise dual-axis replication report.
- `results-task-holdout/summary.json`: leave-one-task-out effects and per-task diagnostics.
- `results-task-holdout/REPORT.zh.md`: zero-shot task-transfer audit.
- `artifacts/features-full40/*.npz`: 80 compact float16 segments for all 40 tasks (615 MiB).
- `results-full40-crossfit/summary.json`: five-fold aggregate grammar and sequential detection.
- `results-prefix/summary.json`: three-seed causal-GRU routing/behavior/combined controls.
- `artifacts/global-prefix-fold*.npz`: fold-specific projected words and matched local controls.
- `results-global-prefix/fold*/healthy_prefix_lm.pt`: healthy-only full-prefix checkpoints.
- `results-global-prefix/summary.json`: five-fold next-word, remote-order, and detection audit.
- `artifacts/open-world-phenotypes.npz`: compact 22-dimensional query phenotypes for full-40 and
  the physical Scene8 anchor corpus.
- `results-open-world-global-k{4,8,12}/`: strict MoE-only healthy-model candidates selected only
  by held-out-success NLL.
- `results-open-world-selected/`: automatic K selection, strict online metrics, and the
  task-conditioned nuisance diagnostic.
- `artifacts/open-world-phenotypes-v2.npz`: corrected, independently layer-permutation-invariant
  22-dimensional full40 and Scene8 phenotypes.
- `results-three-channel-v2/`: continuous VAR/soft-HMM, innovation/over-regularity/return scores,
  matched-FPR physical onset evaluation, and deploy-threshold transfer diagnostics.
- `results-candidate-reranking-v2/`: exact route-matched same-snapshot candidate scores,
  selections, paired ablations, and trunk-cluster uncertainty.

## Unsupported Next Steps

This corpus has no reliable loop onset and no exhaustive open-world Trap labels. Same-snapshot
candidate forks now provide a small offline reranking audit, but only three independent trunks
contain both successful and failed candidates, and they do not isolate a prefix benefit. The
data still cannot validate `<END>` extendability, online grammar-selected execution,
short-chunk intervention, or training-time grammar regularization. Healthy sequence structure
is real; the absent physical early-monitoring gain supports keeping grammar as a mechanism
analysis and candidate-emission hypothesis rather than deploying it as a controller.
