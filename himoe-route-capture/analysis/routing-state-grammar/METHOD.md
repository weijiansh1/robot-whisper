# Routing-state grammar: frozen method

Frozen before the formal run on 2026-08-27.

## Question and cohort

- Reuse the exact 2,560-episode all-outcome formal cache.
- Recover the ten relative-phase query states from the cached
  expert-permutation-invariant geometry array `[episode, phase, descriptor]`.
- Episode length, task, checkpoint, initial state, flow seed, and outcome do
  not enter prototype discovery or grammar clustering.
- The primary phase window is `[0.5, 1.0]`. A five-anchor grid and a
  `[0.5, 0.9]` endpoint-truncated window are sensitivity checks.

## Shared route states

- Give every episode equal weight through the same number of phase anchors.
- Robustly scale all query descriptors, retain up to 30 PCA components, and
  quantize the 25,600 query states with K-means.
- Use eight shared route states in the primary analysis. Six and ten states
  are fixed sensitivity settings. State IDs are canonicalized by PCA-centroid
  order and have no expert identity semantics.
- Report K-means seed stability by ARI against five independent refits.

## Episode grammar

Do not flatten the ten query descriptors. Convert each episode to a symbolic
state sequence and summarize it with normalized structured counts:

- state occupancy;
- directed first-order transition probabilities;
- early/middle/late state occupancy;
- start and terminal state indicators;
- five exhaustive three-state motif classes: `AAA`, `AAB`, `ABB`, `ABA`,
  and `ABC`;
- state-specific `A -> B -> A` return counts;
- normalized state/transition entropy, switch rate, unique-state fraction,
  longest-run fraction, and nonlocal revisit rate.

All count blocks are divided by their number of opportunities, so neither the
original query count nor absolute chunk index enters the grammar scale.

## Unsupervised episode analysis

- Robustly scale and PCA-reduce the grammar vector, then fit Ward partitions
  for episode K=2..6.
- Require at least `max(16, ceil(0.02 N))=52` episodes per full-data cluster.
- Run 50 ordinary 80% episode subsamples and refit scaling, PCA, and Ward in
  every subsample.
- A stable K requires median ARI >=0.75, ARI P10 >=0.50, PAC <=0.20, and the
  minimum-size gate. Select the stable K with highest mean subsample
  silhouette.

## Posthoc evaluation

- Reveal labels only after the grammar partition is frozen.
- Test outcome association by permutation within task x initial-state strata.
- Test task globally and length both globally and by permutation within task.
- Report NMI, permutation-null NMI, excess NMI, p, and BH q.
- Call the grammar `outcome_structured` only at outcome excess NMI >=0.05 and
  BH q <0.05. Flag `task_shadowed` at task NMI >=0.50 or if any episode
  cluster is at least 90% one task.
- Test scalar grammar diagnostics with outcome permutations inside task x
  initial-state strata and BH correction.
- Fixed-representation eight-fold initial-state-group cross-validation compares task-only, occupancy-only,
  full grammar, task+grammar, task+length, and task+length+grammar probes.
  AUC is computed inside each held-out fold before aggregation. These probes
  are descriptive because the unsupervised state vocabulary is
  fitted once on the full label-blind cohort.

## Sensitivity and limits

- Refit the entire state vocabulary and grammar for state K=6/10, five phase
  anchors, and the endpoint-truncated window; compare fixed episode-K labels
  to the primary partition by ARI.
- Relative phase and normalized counts remove explicit length scale but do not
  make successful completion and failure timeout semantically equivalent.
- Failure horizon and outcome have almost no common support. Length-adjusted
  models therefore diagnose confounding; they do not identify an independent
  routing effect.
- A route-state grammar is observational and does not show that routing caused
  the physical outcome.
