# MoE candidate-connectivity graph preregistration

Status: frozen design, 2026-08-23. This document fixes the analysis before the
candidate-connectivity results are computed. It concerns candidate graphs, not
the older expert co-selection graphs under `analysis/routing-graph*`.

## Question and claim boundary

For one exact observation, the policy is run with the same set of 32 flow-noise
seeds, producing 32 candidate action chunks. The analysis asks two questions:

1. Are candidates adjacent in early HB routing geometry also adjacent in the
   geometry of the complete first action chunk?
2. Are eventual successes locally connected in that routing graph?

A positive result means that the early route distribution is a readable
candidate-geometry interface. It does not establish that routing causes the
action or the outcome, that it contains information absent from the complete
hidden state, or that it is already a safe stopping/pruning rule.

## Frozen data scope

The source is the five lossless first-inference slices in
`raw/hub-first-inference/<task>/first_inference_hb5_d0.npz` from the result
bundle. The task aliases are:

- `goal-middle`
- `goal-top`
- `long-t08`
- `spatial-ramekin`
- `spatial-stove`

Each task contains 16 exact-observation pools and each pool contains the same 32
flow-noise seeds, 1000 through 1031. Thus the unit of graph construction is one
`task x init_state_id` pool with `K=32`; candidates from different states or
tasks are never joined by an edge.

The fixed MoE cell is HB layer 5, denoising step 0, at the first policy
inference of each episode. Only the ten action tokens, array positions `1:11`,
are used. The state token at position 0 and AS routing are exactly invariant
across candidates in these pools and are excluded from candidate geometry.

Required raw fields are:

- identity: `episode_index`, `init_state_id`, `flow_noise_seed`;
- endpoints: `actions`, `success`;
- primary routing: `hb_router_probs`;
- controls: `hb_hidden`, `hb_expert_ids`, `hb_selected_prob`.

Before analysis, every task must pass these checks:

- exactly 16 pools and exactly the canonical 32 seeds once per pool;
- `state` and `sim_state` are bitwise constant inside each pool;
- all action, hidden, and routing values are finite;
- every full 32-way routing site has positive mass after impossible negative
  fp16 round-off is clamped to zero;
- the normalized routing mass at every site sums to one;
- gathering `hb_router_probs` at `hb_expert_ids` reproduces
  `hb_selected_prob` within the stored fp16 precision.

Failure of an identity or alignment check stops the whole analysis. A pool is
not removed because its graph or endpoint looks unusual.

## Primary route distance

Let `p[i,t,e]` be the full 32-expert probability vector for candidate `i`,
action token `t`, and expert `e`. At each site, first compute

```text
p = maximum(p, 0)
p = p / sum_e p
```

The squared Hellinger distance at one aligned token site is

```text
H2(i,j,t) = 0.5 * sum_e (sqrt(p[i,t,e]) - sqrt(p[j,t,e]))^2.
```

The candidate distance is the equal-token root mean:

```text
D_route(i,j) = sqrt(mean_t H2(i,j,t)),  t = 1,...,10.
```

Expert coordinates are compared only within this one layer. Full probabilities,
not expert IDs, define the primary geometry.

## Action distance

`actions` is the complete first `10 x 7` chunk. For each benchmark/checkpoint,
load the frozen seven-dimensional `actions.std` vector from its checkpoint
`meta/stats.json` and compute

```text
a_std[i,t,d] = actions[i,t,d] / action_std[d]
D_action(i,j) = RMS(a_std[i] - a_std[j]) over all 10 x 7 coordinates.
```

Subtracting the action mean is unnecessary for pair distances. No dimension is
dropped and no time step is selected after inspecting the results.

For an extracted result bundle where the recorded checkpoint path is absent, a
local `metadata/client/normalization_stats.json` copy is permitted only when its
SHA-256 exactly matches the frozen `normalization_stats_sha256` in
`server_metadata.json`. The embedded seven values and the copied file must also
match exactly; otherwise analysis stops.

## Directed kNN graphs and AUK8

For each pool, sort the other 31 candidates of node `i` by `D_route(i,j)`.
Distance ties are resolved by ascending `flow_noise_seed`. Let `N_route(i,k)`
be the first `k` neighbors. The inferential analysis uses the nested directed
graphs for every `k=1,...,8`; no distance threshold or best `k` is selected.

For action geometry, rank `D_action(i,j)` among the 31 peers of each node, using
midranks for action-distance ties, and normalize the rank to

```text
u_action(i,j) = (rank_action(i,j) - 1) / 30.
```

For pool `p`, define the action-neighbor gain

```text
G_action,p(k) = 1 - 2 * mean_{i, j in N_route(i,k)} u_action(i,j).
AUK8_action,p = mean_{k=1,...,8} G_action,p(k).
```

The random-neighbor expectation is zero; positive values mean route neighbors
are closer than random peers in action space. The confirmatory action statistic
is the equal-task macro average:

```text
T_action = mean over 5 tasks of mean over that task's 16 pools of AUK8_action,p.
```

For visualization only, the displayed graph is the directed `k=4` graph, with
an optional undirected edge when either direction is present. Its appearance
does not determine the inferential `k` range.

## Success connectivity

A success analysis is possible only in pools containing both outcomes. The
fixed primary subset is all 40 mixed pools (`0 < successes < 32`):

| task | mixed pools |
|---|---:|
| goal-middle | 0 |
| goal-top | 7 |
| long-t08 | 13 |
| spatial-ramekin | 8 |
| spatial-stove | 12 |

`goal-middle` is 512/512 successful and therefore supplies no success
connectivity information. It remains in the action-geometry endpoint.

For node `i`, define the leave-node-out graph vote

```text
q_i(k) = mean_{j in N_route(i,k)} success_j.
```

Within each mixed pool, compute the tie-aware AUC of `q_i(k)` for predicting
`success_i`, then define

```text
G_success,p(k) = AUC_p(k) - 0.5
AUK8_success,p = mean_{k=1,...,8} G_success,p(k).
```

First average mixed pools within each informative task, then average the four
task means with equal weight:

```text
T_success = mean over 4 informative tasks of their mixed-pool mean AUK8_success.
```

The 16 pools with 4 through 28 successes (`balanced16`) are a frozen descriptive
sensitivity subset, not an alternative primary subset. It cannot replace the
mixed40 result based on which one is more favorable.

## Common seed-column permutation

The 32 seed columns recur in every state and every task and are therefore not 80
independent candidate assignments. The null uses one common seed-column
permutation per draw.

For each of 9,999 draws from `numpy.random.default_rng(20260823)`:

1. draw one permutation `pi` of the sorted 32 seed columns;
2. keep all route graphs and graph baselines fixed;
3. jointly relabel `(actions, success)` by the same `pi` in every pool of every
   task;
4. recompute action ranks, neighbor votes, pool statistics, and task macros.

This preserves every pool's class count, the action-success pairing, and all
cross-state/cross-task structure tied to a seed column while breaking the
candidate-level alignment with routing. Independent permutations by pool are
forbidden. Monte Carlo p-values use

```text
p = (1 + number of null statistics >= observed statistic) / 10000.
```

All tests are one-sided in the preregistered positive direction.

## Confirmatory family and maxT

The confirmatory family contains exactly two endpoints:

- `T_action`;
- `T_success`.

Because their scales differ, standardize each observed statistic and its null
draws using that endpoint's permutation-null mean and standard deviation. For
each draw, take the maximum standardized statistic across the two endpoints.
The maxT-FWER p-value for endpoint `e` is

```text
p_e = (1 + count_b[max_e Z_null[b,e] >= Z_observed[e]]) / 10000.
```

The decision threshold is maxT-FWER `p < 0.05`. Unadjusted permutation p-values
may also be reported but cannot support the confirmatory claim.

The `k=1,...,8` curves are components of the single AUK8 endpoint, not eight
tests. No layer, denoising step, token subset, action horizon, task subset, or
graph threshold may be selected from these data. Any later scan must be labeled
new exploratory work or include its entire search family in a new maxT null.

Effect sizes are always reported per task as well as in the macro average.
Leave-one-task-out macro estimates are descriptive cross-task stability checks;
task-specific p-values are not computed.

## Frozen baselines

Each baseline replaces `D_route` when forming the same nested directed graph and
the same AUK8 endpoints. Pool definitions, tie breaking, aggregation, and common
permutation draws remain identical.

### Complete hidden-state baseline

Use the ten action-token tensors from `hb_hidden[:,1:11,:]`:

```text
D_hidden(i,j) = RMS(hb_hidden[i,1:11] - hb_hidden[j,1:11]).
```

This is the essential information-boundary control because routing at this cell
is computed from the hidden state. Route performance that does not exceed this
baseline can still support a compact monitoring-interface claim, but not an
incremental-information claim.

### Initial-noise baselines

Reconstruct the identity-checked flow noise for each seed with

```text
default_rng(seed).standard_normal((10,24)).astype(float32).
```

The primary noise baseline uses the seven live action dimensions and RMS
distance. Full 24-dimensional noise is reported as a sensitivity baseline.

### Order-invariant top-k baseline

Scatter each site's four `hb_selected_prob` values into a 32-coordinate zero
vector at the corresponding `hb_expert_ids`, renormalize the selected mass, and
compute the same site-mean Hellinger distance. This is the probability-weighted
top-4 baseline. An unweighted top-4-set Jaccard distance may be reported as a
secondary sensitivity.

The baseline comparison reports paired pool/task effect differences from the
full-probability route graph. These controls are secondary: they cannot rescue
a failed primary endpoint. Any inferential claim that the full route graph
beats a baseline must use a separately declared maxT family over every claimed
baseline contrast; otherwise the contrasts remain effect-size comparisons.

## MST secondary analysis

As a threshold-free graph sensitivity, construct the minimum spanning tree of
the complete undirected `D_route` graph in each pool. Kruskal ties are resolved
lexicographically by the two flow-noise seeds.

For action geometry, count each tree edge in both directions and reuse the
node-wise normalized action ranks:

```text
G_action,MST = 1 - 2 * mean over directed tree-edge copies of u_action(i,j).
```

For success, report tree-edge homophily minus its fixed-class-count expectation:

```text
G_success,MST = mean_edges 1[y_i = y_j]
                - (n1*(n1-1) + n0*(n0-1)) / (32*31).
```

MST results use the same common permutations but are secondary and cannot
replace AUK8 or be chosen only when favorable.

## D/C/Q and block-fragility exploration

The five aligned
`analysis/expert-activation-hidden-matched/<task>/candidate_proxy_values.npz`
files contain:

- `scene_ids`, `seed_ids`;
- `disagreement`, `cancellation`, `conflict`;
- `block_sensitivity`, `routed_sensitivity`, `input_sensitivity`.

The six candidate values are aligned to the raw slice strictly by
`(scene_id, seed_id)`. For each scalar, an exploratory graph-smoothness curve can
reuse the action-rank construction with `abs(z_i-z_j)` in place of action
distance. These values may also color graph nodes.

D/C/Q analyses are explicitly exploratory. The proxy files contain neither raw
runtime expert outputs nor enough information to reconstruct route, action, or
success geometry by themselves. No D/C/Q result establishes a causal action or
outcome commitment effect. If p-values are shown, all six scalar families and
all reported graph constructions must be corrected together; otherwise only
effect sizes are reported.

## Withdrawal of the old `ids[...,0]` top-1 result

All previous candidate-connectivity numbers or conclusions that treated
`hb_expert_ids[...,0]` (or generic `ids[...,0]`) as the top-1 expert are
withdrawn and must not be cited. The stored top-k IDs are a selected set whose
slot order is not an authoritative probability ranking; the first slot is not
guaranteed to be `argmax`.

If a descriptive top-1 value is ever needed, it must be recomputed as
`argmax(hb_router_probs)` from the normalized full 32-way probabilities. Top-1
is not part of this preregistration. The primary result uses full-distribution
Hellinger geometry, and the top-k control is explicitly order-invariant.

## Required reporting

The final result must include:

- all validation checks and exact eligible pool counts;
- per-task and macro `AUK8_action` and `AUK8_success` effect sizes;
- the full `k=1,...,8` curves without selecting a favorable point;
- unadjusted and maxT-FWER permutation p-values for the two primary endpoints;
- hidden, live-noise, full-noise, and probability-weighted top-4 baselines;
- MST secondary effects;
- mixed40 primary and balanced16 descriptive success results;
- explicit limitations: one cell, one first inference, five tasks, three
  checkpoint suites, observational geometry, and no runtime intervention.

The wording "routing predicts/encodes success" is not permitted from graph
homophily alone. The allowed wording after a positive corrected result is that
eventual outcomes are locally associated with the early route graph within the
observed same-state candidate pools.
