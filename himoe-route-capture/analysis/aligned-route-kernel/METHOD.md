# Aligned route-signature landmark experiment

This is an exploratory follow-up to the frozen all-outcome Ward analysis. It
was designed after inspecting that analysis and is not confirmatory.

## Source cohort

- Use the exact 2,560 episodes and the immutable feature-cache manifest from
  `analysis/all-outcome-routing-clusters`.
- The episode remains the statistical unit. Outcome is hidden until all three
  trajectory partitions have been selected and frozen.
- Reuse the ten relative-phase anchors over `[0.5, 1.0]` and the 320-dimensional
  expert-permutation-invariant `geometry` signature at each anchor.

## Aligned landmark representation

1. Robustly scale the 25,600 anchor signatures and retain 32 PCA components.
2. Construct three route-only trajectory variants:
   - `raw`: no design residualization;
   - `task_residual` (primary): subtract the task median trajectory;
   - `task_init_residual`: subtract the task x initial-state median trajectory.
3. Center every component over phase within each episode and normalize the
   resulting trajectory to unit Frobenius norm. This deliberately tests shape,
   not absolute route level.
4. Select 256 landmarks by deterministic k-means++ without using outcome.
5. For every episode-landmark pair, take the maximum cosine similarity over
   integer shifts `-2..2` anchors, renormalizing on the overlap. Distance is
   `1 - max_similarity`. The 256 landmark distances form a scalable
   dissimilarity representation; this is not claimed to be an exact PSD
   Nystrom kernel.

## Clustering and selection

- Robustly scale each landmark-distance representation, retain up to 30 PCA
  components, and fit Ward partitions for `K=2..8`.
- Require cluster size `max(16, ceil(0.02 N))`.
- Use 50 ordinary 80% episode subsamples. The landmark basis remains fixed;
  distance-column scaling, PCA, and Ward are refit within each subsample.
- A stable K requires median ARI >=0.75, ARI P10 >=0.50, and 512-probe PAC
  <=0.20. Among stable K, select the highest mean sampled silhouette.

## Posthoc tests

- After labels are frozen, test outcome association by 5,000 permutations
  within task x initial-state, and report raw/null/excess NMI with BH q.
- Also report task and episode-length association, per-task outcome tests, and
  cluster-by-task failure-rate deltas from each task baseline.
- A `cross_task_failure_block` or `cross_task_success_block` requires:
  conditional outcome excess NMI >=0.05 and BH q <0.05 for the partition, plus
  at least three mixed-outcome tasks contributing at least ten episodes to the
  block with failure-rate deltas all in the same direction.

## Interpretation limits

- `task_residual` clusters describe deviations from each task's median route
  shape, not absolute shared router states.
- Integer shift alignment is coarse and can erase true timing differences.
- Relative phase still equates success completion with failure timeout, and
  short episodes are interpolated onto ten anchors.
- Landmarks and anchor PCA are fitted once on the full route-only cohort;
  subsample stability is conditional on that fixed basis.
- This experiment cannot establish causal outcome mechanisms or unseen-task
  generalization.
