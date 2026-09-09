# Failure-rollout routing clusters: frozen method

Frozen before running `analyze_failure_routing_clusters.py` on 2026-08-27.

## Scope and time axis

- Use every terminal-failure episode from the five complete LIBERO
  `right-16x32` runs, with one episode as one statistical unit.
- Require the exact five-run, 512-episode-per-run formal lineage and the
  frozen per-task failure/query counts (307 failures, 13,570 queries total).
- For an episode with `T` queries, query `q` has relative phase
  `q / (T - 1)`. Absolute query/chunk index and `T` are not cluster features.
- The primary window is relative phase `[0.5, 1.0]`, linearly resampled to ten
  equally spaced phase anchors. Window `[0.6, 1.0]`, truncated window
  `[0.5, 0.9]`, and 8/12-anchor grids are sensitivity checks.
- Task, checkpoint, initial state, flow seed, physical state, action, and known
  failure-mode proxies are hidden until cluster labels are frozen.

## Representations

All primary representations retain the normalized phase order.

1. `geometry`: expert-permutation-invariant entropy, top-mass, action-token
   dispersion, state/action gap, denoise Hellinger, and hard-retention
   descriptors at each phase anchor.
2. `change`: the phase derivative of the same geometry trajectory.
3. `recurrence`: within-episode pairwise soft Hellinger and selected-expert
   occupancy distances between all phase anchors. This is invariant to a
   checkpoint-wide permutation of expert IDs.
4. `expert_occupancy`: square-root action-token top-4 occupancy by phase,
   layer, and expert. This secondary view is not expert-permutation invariant
   across checkpoints and is used to diagnose task/checkpoint shadowing.

## Clustering and selection

- Globally preprocess each representation without task centering, retain up
  to 30 PCA components, and cluster with Ward linkage.
- Evaluate `K=2..6`. A full-data partition must have at least 16 episodes per
  cluster.
- Run 200 ordinary 80% episode subsamples. Refit active-column selection,
  scaling, PCA, and Ward linkage independently inside every subsample. For
  each K report silhouette, median and 10th-percentile ARI to the full
  partition, matched cluster Jaccard, consensus PAC, and consensus dispersion.
- A stable K requires minimum size >=16, median ARI >=0.75, ARI P10 >=0.50,
  and PAC <=0.20. Choose the stable K with the highest mean subsample
  silhouette.
- If no K passes, report `no_stable_partition`; the best size-valid
  silhouette partition may be shown only as exploratory.

## Post-cluster interpretation

After labels are frozen, report task/checkpoint, length, initial-state, and
flow-seed composition. Compare labels with the existing physical failure-mode
proxies only as external description; assess that proxy by permutation within
task x initial-state strata. Raw NMI, its permutation-null mean and excess NMI
are reported together. Flow-seed association uses the same task x initial-state
strata, and all post-cluster permutation p-values receive Benjamini-Hochberg
correction.

The following interpretation gates are frozen before looking at results and do
not alter K or labels:

- `task_shadowed` if task NMI is at least 0.50 or any cluster is at least 90%
  one task;
- `representation_specific` if a primary view has ARI below 0.50 with every
  other stable primary view; no stable peer is reported separately;
- `phase_sensitive` if any 8/12-anchor, `[0.6,1.0]`, or `[0.5,0.9]`
  sensitivity partition has ARI below 0.50 to that view's primary partition or
  contains a cluster with fewer than 16 episodes;
- only a stable primary view with none of those flags is a
  `shared_taxonomy_candidate`.

All three primary views receive the phase-grid/window sensitivity checks.
Because failure lengths 22/30/52 are perfectly nested within task, normalized
phase removes explicit absolute-chunk scale but cannot identify task and
horizon effects separately.

The stability result is conditional on ordinary episode subsampling. It does
not establish generalization to held-out tasks, initial-state groups, or flow
seeds.
