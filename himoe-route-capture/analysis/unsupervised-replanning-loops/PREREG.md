# Label-blind discovery of replanning loops

Frozen before running `analyze_unsupervised_replanning_loops.py` on
2026-08-27.

## Question

Can trajectory-internal recurrence reveal closed physical-routing-action
motifs without knowing whether an episode succeeds, when a failure happens,
or why it happens?

Discovery follows

`leave X/R/A state -> return to X/R/A state -> replay short consequence`.

Episode outcomes are not used to select pairs, normalize features, set a
threshold, suppress duplicate events, select the cluster count, or fit the
clusters. Outcomes are read only after discovery is complete.

## Data and representations

- Scope: every complete `right-16x32` run under
  `VLA_MUI_HUB/cache/HiMoE-VLA`.
- `X`: canonicalized MuJoCo qpos, standardized within task using every query.
- `R`: recorded top-4 HB expert sets at all aligned 8 layer x 10 denoise x 10
  action-token cells. Client NPZ routing stubs are not used.
- `A`: the recorded 10 x 7 action chunk, standardized within task.
- RGB, goal poses, inferred failure modes, and successful terminal states are
  excluded from discovery.

## Physical seed pair

Primary minimum period is three control queries, with no maximum period. The
current endpoint `k` must have a following query so that replay can be checked.
For every eligible `k` and every `j <= k - 3`:

1. Let `m` be the interior query farthest from `X_j`.
2. Let `c = d_X(j,k)`, `e = d_X(j,m)`, and let `p` be the qpos path length
   from `j` through `k`.
3. Define label-free physical seed strength as

   `max(1-c/e,0) * max(1-c/p,0) * (1-exp(-e/s_X))`,

   where `s_X` is the task-wide median positive adjacent qpos distance.

Choose the `j` with maximum physical seed strength. This yields at most one
seed pair for each current endpoint and prevents route or outcome information
from choosing the physical recurrence.

A minimum-period-four extraction is frozen as a sensitivity analysis.

## Route, action, and replay features

For the selected `(j,k)` pair:

- physical closure fraction and path inefficiency;
- route return overlap `O_R(j,k)` and route return gain above the least-similar
  interior route `O_R(j,k) - min_t O_R(j,t)`;
- action return distance and action return gain below the farthest interior
  action `max_t d_A(j,t) - d_A(j,k)`;
- aligned two-query replay distances for X and A and aligned route overlap for
  R, comparing `(j,j+1)` with `(k,k+1)`;
- repeated physical effect distance between `X[j+1]-X[j]` and
  `X[k+1]-X[k]`;
- period, relative phase, and excursion magnitude for later description.

## Label-free score and candidates

For each task, empirical CDFs are fitted only on seed pairs whose current
endpoint lies in the longest prefix shared by every episode of that task.
The CDFs do not use outcomes. Seven favorable percentiles are computed:

1. physical seed strength;
2. route return gain;
3. action return gain;
4. low aligned two-query X distance;
5. high aligned two-query route overlap;
6. low aligned two-query action distance;
7. low repeated-effect distance.

The loop score is their geometric mean. The primary candidate threshold is
the task-specific 99th percentile of equal-prefix scores and additionally
requires positive route and action return gains. Thresholds at 99.5%, 98%,
and 95% are frozen sensitivity checks.

Within an episode, score-ordered nonmaximum suppression removes candidates
whose start and return endpoints are both within one query of an already kept
candidate.

## Unsupervised motif clusters

Primary full-trajectory candidates are clustered without outcomes. Ward
hierarchical clustering uses standardized discovery features: period fraction,
phase, physical closure, excursion, path inefficiency, the seven score
percentiles, and raw route/action return quantities.

The cluster count is selected from 2 through 6 by maximum mean silhouette,
subject to every cluster containing at least `max(5, 2% of candidates)` events.
If no candidate count passes that constraint, two clusters are retained and
the failure of the size guard is reported. Cluster descriptions remain
geometric; they are not assigned semantic failure names.

## Outcome unblinding

Only after thresholds, candidates, nonmaximum suppression, cluster count, and
cluster assignments are frozen in memory are `success` fields read again.

External validation uses equal-prefix episode summaries and compares terminal
failure versus success only within task x initial-state strata. The frozen
endpoints are maximum loop score, 90th-percentile loop score, and primary
candidate presence. A within-stratum permutation test and initial-state
cluster bootstrap use 5000 draws. Full-trajectory and cluster outcome rates
are descriptive because episode length and terminal phase differ by outcome.

## Interpretation

- Discovery can establish recurrent trajectory geometry without knowing the
  failure cause.
- Outcome enrichment can establish association with eventual failure.
- Neither result identifies a semantic failure type or proves that routing
  recurrence caused the outcome.
- The physical representation is qpos, not a stored visual embedding.
