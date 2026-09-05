# MoE Routing Grammar

This directory turns the proposal in `idea.md` into a leakage-resistant experiment on the
`VLA_MUI_HUB/right-16x32` corpus. It treats each denoise step as a multitrack routing chord,
one 10-step query as a learned word, and consecutive queries as a sequence.

The implementation is read-only with respect to the source hub. Generated features and models
stay under this directory.

## Result

The data support **multiscale temporal structure and within-task conditional sequence grammar**,
with important limits on invariance and control use.

- On 2,253 cross-fitted healthy test episodes, task+position+history NLL is 1.071 bits/token versus
  1.178 for task+position alone. The paired gain is 0.108 bits/token with a state-blocked 95% CI
  of [0.076, 0.140]. This directly establishes history information beyond task and query clock.
- A crossed state+noise-seed holdout reproduces the effect: 0.134 bits/token, two-axis bootstrap
  95% CI [0.109, 0.162], with all 16/16 held-out cells agreeing in direction.
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
machine-readable `summary.json` files.

## Experiment Contract

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

## Representation

For each `[query, layer, flow, token, expert]` routing tensor, the extractor computes:

- normalized entropy, Top-1/Top-2 margin, Top-1 mass, and Top-4 mass;
- soft Hellinger token consensus and authoritative Top-4 support consensus;
- token-expert effective rank;
- flow Hellinger velocity, Top-4 switching, and acceleration;
- cross-layer Hellinger disagreement.

This yields a `10 x 81` chord tensor per query. Levels, first differences, and second differences
form a 2,187-dimensional ordered descriptor. Train-only robust scaling and PCA reduce it to 24
dimensions before a diagonal GMM produces query words.

The sequence comparison includes unigram, absolute position, position+history, bigram, fixed
fourth-order Markov, unordered bag-context, PST, PST plus explicit duration, and task-conditioned
oracle controls. The primary nested model first conditions on task and absolute position, then
uses up to two previous words; unsupported contexts fall back to the clock-only distribution.

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

## Unsupported Next Steps

This corpus has no reliable loop/static/mixed labels, no extended timeout rollouts, and no
candidate-fork intervention outcomes. It therefore cannot validate `<END>` extendability,
grammar-selected action candidates, short-chunk intervention, or training-time grammar
regularization. The healthy sequence result is now robust, but the absent early-monitoring gain
still supports keeping grammar as an analysis prior rather than deploying it as a controller.
