# Within-task label-blind loop motifs

Frozen before running `analyze_within_task_unsupervised_loops.py` on
2026-08-27.

## Motivation

The pooled unsupervised loop audit produced cluster-to-task purity 0.832.
This suggests that a cross-task cluster can primarily learn task identity.
This analysis therefore learns a separate motif vocabulary inside every task.

## Label isolation

Input geometry comes from the label-blind `seed_pairs.csv` produced by
`analyze_unsupervised_replanning_loops.py`. Although that audit table also has
an outcome column for later review, this analysis sanitizes every row into a
new record that excludes the outcome field. Outcomes are re-read only after
all task-specific models, cluster counts, centroids, and assignments are
complete.

No success/failure value is used for event inclusion, feature scaling, cluster
count selection, initialization selection, centroid fitting, cluster naming,
or full-trajectory assignment.

## Fit population

Each task is handled independently. The fit population contains every seed
pair that:

- lies in the longest prefix shared by all episodes of that task; and
- has positive physical, routing, and action return topology.

The equal prefix gives every episode the same number of opportunities. The
positive-topology gate is label-free and does not impose the earlier q99 score
threshold.

## Geometry features

Primary clustering uses twelve within-task features:

- the seven favorable empirical percentiles used by the label-blind loop
  score: physical seed, route gain, action gain, two-query X replay, two-query
  route replay, two-query action replay, and repeated physical effect;
- physical closure fraction;
- routing recovery fraction;
- action recovery fraction;
- log physical excursion in adjacent-step units;
- path inefficiency.

Task name, episode identity, initial state, flow seed, query phase, period,
outcome, and semantic failure proxies are excluded from fitting. Phase and
period are reported only after assignment.

Features are z-scored from that task's fit population. Constant dimensions are
dropped.

## Model and cluster-count selection

For every task and `K=2..8`:

1. run k-means++ five times with 50 Lloyd iterations;
2. retain the initialization with minimum squared error;
3. compute mean pairwise adjusted Rand index across the five assignments;
4. compute silhouette on a deterministic sample of at most 1500 fit events;
5. require every cluster to contain at least `max(20, 2% of fit events)`.

Choose the K with maximum silhouette among models passing the size guard and
mean ARI >= 0.8. If none passes stability, choose the best silhouette among
size-valid models and mark low stability. If none passes size, retain K=2 and
mark the failed guard.

Cluster IDs are made deterministic by ordering centroids by median minimum
X/R/A recovery fraction and then median loop score. Every positive-topology
event in the full task trajectory is assigned to the nearest frozen centroid.

## Outcome unblinding

After all task models and assignments are complete, outcome labels are loaded.
For every task and cluster, each episode receives an equal-prefix cluster rate:

`number of assigned events in cluster / number of eligible seed endpoints`.

The primary external endpoint is failure AUC for each cluster rate, comparing
only within initial-state strata. A 5000-draw within-stratum permutation test
and initial-state cluster bootstrap are used. The max-statistic family-wise
p-value covers all cluster-rate endpoints inside that task.

Cluster presence, maximum loop score, and maximum recovery are secondary
descriptions. Full-trajectory outcome composition is descriptive because
episode length and terminal phase differ by outcome.

## Interpretation

- A significant cluster rate identifies a task-internal geometric motif
  associated with eventual outcome without requiring a failure-cause label.
- A cluster with tiny recovery fractions remains a micro-return motif, even if
  outcome-associated.
- Clusters are not semantic failure categories and do not establish causality.
