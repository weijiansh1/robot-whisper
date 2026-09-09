# Truncate-90 aligned route-kernel sensitivity

This analysis is a frozen sensitivity follow-up to
`analysis/aligned-route-kernel`. It asks whether the full-window raw failure
block C1 (252 episodes, 246 failures) survives when the final 10% of relative
episode phase is excluded. It does not replace the full-window result.

## Cohort and representation

- Use the identical 2,560-episode cohort and the completion-verified feature
  manifest from `analysis/all-outcome-routing-clusters`.
- Replace only `feature_primary_geometry` (ten anchors over relative phase
  `[0.5, 1.0]`) with `feature_truncate90_geometry` (ten anchors over
  `[0.5, 0.9]`).
- Keep the aligned landmark protocol unchanged: robust anchor scaling, 32 PCA
  components, phase centering, unit Frobenius normalization, 256 deterministic
  k-means++ landmarks, and maximum cosine similarity over integer shifts
  `-2..2`.
- Analyze `raw` and the primary `task_residual` representation. Outcome remains
  hidden until all truncate-90 labels are frozen.

## Partition selection

- Refit landmark-distance scaling, PCA, and Ward partitions for `K=2..8`.
- Use 50 ordinary 80% episode subsamples with the landmark and anchor-PCA bases
  fixed, applying the same minimum-size, ARI, PAC, and silhouette rules as the
  full-window analysis.
- In addition to the selected partition, freeze a truncate-90 raw `K=6`
  partition and task-residual `K=2` partition. These permit like-for-like ARI
  comparisons with the corresponding full-window reported labels even if
  sensitivity model selection changes K.

## Full raw C1 persistence

The reference target is full-window raw C1, fixed before this run because it is
the only full-window `cross_task_failure_block`. Match it to the truncate-90
raw `K=6` cluster with maximum episode Jaccard overlap, breaking ties by overlap
count and then the lower cluster label.

Report episode-level and failure-only precision, recall, F1, and Jaccard, plus
per-task reference, candidate, and overlap composition. Before inspecting the
truncate-90 result, define `strong_persistence` as all four episode/failure
precision and recall values being at least 0.80, with the matched truncate-90
cluster retaining positive failure-rate deltas in at least three mixed-outcome
tasks that each contribute at least ten candidate episodes.

The ordinary cross-task block criterion remains unchanged: the fixed `K=6`
partition itself must be stable, conditional outcome excess NMI must be at
least 0.05 with BH q below 0.05, and the matched block must meet the same
cross-task cell/direction rule.

## Posthoc tests and limits

- After label freeze, use 5,000 permutations and BH correction for the same
  task, episode-length, initial-state, and outcome associations used in the
  parent analysis.
- Reference matching is explicitly posthoc and does not affect truncate-90
  label construction or K selection.
- Resampling stability remains conditional on full-cohort landmark and
  anchor-PCA bases. Ten-anchor interpolation and integer-shift alignment can
  smooth or erase timing differences.
- Persistence would show robustness to removing the final 10% of relative
  phase, not causality, unseen-task generalization, or independence from task
  and episode-length proxies.
