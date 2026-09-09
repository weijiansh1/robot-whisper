# All-outcome normalized routing clusters: frozen method

Frozen before running `analyze_all_outcome_routing_clusters.py` on 2026-08-27.

## Question and cohort

- Jointly cluster all 2,560 episodes from the exact five complete LIBERO
  `right-16x32` runs: 2,253 successes and 307 terminal failures, totaling
  51,308 recorded queries.
- One episode is one statistical unit. Outcome, task, checkpoint, initial
  state, flow seed, episode length, and the existing failure-only labels are
  hidden until every joint route-only label is frozen.
- For an episode with `T` queries, query `q` has relative phase
  `q / (T - 1)`. Absolute query index and `T` are not clustering features.
- The primary window is relative phase `[0.5,1.0]`, linearly interpolated to
  ten anchors. A five-anchor version tests sensitivity to upsampling short
  successful episodes. Window `[0.5,0.9]` tests terminal-endpoint dependence.

## Representations

Use the same four representations as the failure-only analysis:

1. expert-permutation-invariant `geometry` trajectories;
2. their phase derivative, `change`;
3. expert-permutation-invariant pairwise phase `recurrence` geometry;
4. secondary, expert-ID-based `expert_occupancy` trajectories.

No task centering or outcome residualization is used.

## Joint clustering

- Robustly preprocess each representation, retain up to 30 PCA components,
  and fit Ward partitions for `K=2..6`, matching the failure-only analysis.
- Require at least `max(16, ceil(0.02 N))` episodes per full-data cluster: 52
  for the 2,560-episode joint cohort.
- Run 50 ordinary 80% episode subsamples. Refit active-column selection,
  scaling, PCA, and Ward inside every subsample. Silhouette is evaluated on a
  fixed-size sample for scalability; consensus PAC uses a fixed 512-episode
  probe.
- A stable K requires median ARI >=0.75, ARI P10 >=0.50, PAC <=0.20, and the
  minimum-size gate. Select the stable K with highest mean subsample
  silhouette. If none passes, label the best size-valid silhouette K as
  exploratory only.

## Direct answer tests

After joint labels are frozen:

- report success/failure counts and failure rate in every block;
- test outcome association by permutation within `task x initial_state`, and
  report raw NMI, permutation-null NMI, excess NMI, p, and BH q;
- call a view `outcome_structured` only if conditional excess NMI >=0.05 and
  BH q <0.05;
- restrict each joint partition to the same 307 failures and compare it with
  the prior failure-only labels. The primary comparison cuts the joint
  dendrogram at the old failure-only K, so a change in model-selected K cannot
  by itself create an apparent reorganization. Report same-K ARI, matched
  cluster Jaccard, old-to-joint split entropy, joint-to-old merge entropy, the
  full transition table, and within-task same-K ARI. The reported joint
  partition's ARI is auxiliary. Call either ARI `preserved` for ARI >=0.80,
  `moderately_changed` for 0.50-0.80, and `reorganized` for ARI <0.50;
- flag `task_shadowed` at task NMI >=0.50 or when any cluster is >=90% one
  task;
- compare the five-anchor and `[0.5,0.9]` partitions with the primary labels.
  ARI <0.50 or a cluster below the corresponding 2% size gate marks phase
  sensitivity.
- report outcome association separately within every task that contains both
  outcomes, with one BH correction across all estimated view x task tests.

## Support and termination controls

After the primary joint partition is frozen, run fixed-K descriptive controls:

- `C2`: remove the all-success middle-drawer task;
- `C3`: use the maximum set of unique task x initial-state matched pairs. The
  current corpus supports 157 failures and 157 successes without duplicating
  episodes. Within each stratum, sorted flow-seed ranks are sampled at evenly
  spaced quantiles as a deterministic coverage rule, not as a claim that
  nearby seed numbers are physically similar;
- `C3F`: refit the 157 supported full failure trajectories alone;
- `C4F`: truncate each matched failure to its paired success query length and
  cluster the 157 truncated failures;
- `C4`: jointly cluster those same truncated failures and 157 successes.

First report `ARI(old restricted failure labels, C3F)` as the support/refit
effect and `ARI(C3F, C3 restricted to failure)` as the full-trajectory effect
of adding balanced successes. Then report `ARI(C3F, C4F)` as the failure-tail
effect and `ARI(C4F, C4 restricted to failure)` as the net effect of adding
success under the matched query budget. These controls are conditional on the
157 supported pairs and do not identify effects for the other 150 failures.

## Interpretation limits

- Successful 100% phase denotes task completion; failed 100% phase denotes
  timeout. Joint clustering can therefore expose terminal semantics rather
  than a causal precursor of success or failure.
- Episode length is excluded explicitly but remains outcome- and task-linked
  in the observed corpus. Relative normalization does not identify these
  effects separately.
- Ordinary episode subsampling does not establish generalization to unseen
  tasks, initial states, or flow seeds.
