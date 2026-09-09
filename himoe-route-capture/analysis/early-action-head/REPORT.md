# Early d0 action-head probe

## Result

The task-local hidden head reconstructs candidate action geometry, but the repeated-seed template explains nearly all of that result. The seed-disjoint sensitivity is the relevant check for a reusable MoE-derived signal.

## All 80 state pools

| method | distance Spearman (median) | K8 coverage / exact random |
|---|---:|---:|
| raw initial flow noise (10x24) | 0.234 | 0.971 |
| raw initial flow noise (live 10x7) | 0.475 | 0.970 |
| raw Early hidden mean | 0.461 | 0.981 |
| raw Early router (layers pooled, 32) | 0.316 | 0.992 |
| raw Early router (layers retained, 4x32) | 0.415 | 0.991 |
| seed-only action template | 0.919 | 0.947 |
| initial-noise linear head (state LOSO) | 0.919 | 0.947 |
| hidden linear head (state LOSO) | 0.930 | 0.946 |
| router-32 linear head (state LOSO) | 0.786 | 0.968 |
| router-4x32 linear head (state LOSO) | 0.912 | 0.947 |
| true-action PAM oracle | 1.000 | 0.934 |
| exact uniform random K8 | - | 1.000 |

## State + seed-disjoint sensitivity

The four heads have different target centers, so cross-block distances are not calibrated. Correlation is therefore computed only inside each held-eight block. Selection uses PAM K2 inside each block and unions the four pairs into K8.

| method | within-held8 Spearman (median) | stratified K8 coverage / exact stratified random |
|---|---:|---:|
| noise head (state + seed disjoint) | 0.013 | 1.007 |
| hidden head (state + seed disjoint) | 0.542 | 0.987 |
| router-32 head (state + seed disjoint) | 0.234 | 0.995 |
| router-4x32 head (state + seed disjoint) | 0.446 | 0.986 |
| exact random: two from each held8 | - | 1.000 |

## K8 action voting

After pruning, the final vote is the ordinary unweighted medoid of the eight true completed action chunks. Regret is measured by its K32-global centrality gap divided by the pool median pair distance.

| method | normalized global-medoid regret | exact K32 medoid reproduction |
|---|---:|---:|
| raw initial flow noise (10x24) | 0.011 | 0.500 |
| raw initial flow noise (live 10x7) | 0.020 | 0.225 |
| raw Early hidden mean | 0.039 | 0.188 |
| raw Early router (layers pooled, 32) | 0.030 | 0.350 |
| raw Early router (layers retained, 4x32) | 0.042 | 0.138 |
| seed-only action template | 0.017 | 0.300 |
| initial-noise linear head (state LOSO) | 0.015 | 0.312 |
| hidden linear head (state LOSO) | 0.015 | 0.350 |
| router-32 linear head (state LOSO) | 0.016 | 0.450 |
| router-4x32 linear head (state LOSO) | 0.014 | 0.412 |
| noise head (state + seed disjoint) | 0.051 | 0.037 |
| hidden head (state + seed disjoint) | 0.029 | 0.150 |
| router-32 head (state + seed disjoint) | 0.035 | 0.188 |
| router-4x32 head (state + seed disjoint) | 0.030 | 0.250 |
| true-action PAM oracle | 0.008 | 0.537 |
| MC uniform random K8 | 0.039 | 0.188 |
| MC random: two from each held8 | 0.038 | 0.189 |

Random voting baselines use 10,000 common subsets per pool (seeds 20260821 and 20260822); coverage and descriptive success retention remain exact, not Monte Carlo.

## Balanced-min4 16 pools: descriptive success retention

| method | any success retained | successful-candidate recall |
|---|---:|---:|
| raw initial flow noise (10x24) | 0.938 | 0.269 |
| raw initial flow noise (live 10x7) | 1.000 | 0.277 |
| raw Early hidden mean | 0.938 | 0.250 |
| raw Early router (layers pooled, 32) | 1.000 | 0.265 |
| raw Early router (layers retained, 4x32) | 1.000 | 0.279 |
| seed-only action template | 1.000 | 0.266 |
| initial-noise linear head (state LOSO) | 1.000 | 0.259 |
| hidden linear head (state LOSO) | 0.938 | 0.224 |
| router-32 linear head (state LOSO) | 0.938 | 0.255 |
| router-4x32 linear head (state LOSO) | 0.938 | 0.207 |
| noise head (state + seed disjoint) | 0.938 | 0.266 |
| hidden head (state + seed disjoint) | 1.000 | 0.272 |
| router-32 head (state + seed disjoint) | 0.938 | 0.251 |
| router-4x32 head (state + seed disjoint) | 1.000 | 0.285 |
| true-action PAM oracle | 0.875 | 0.216 |
| exact uniform random K8 | 0.946 | 0.250 |
| exact random: two from each held8 | 0.954 | 0.250 |

Here `balanced-min4` means 4-28 successes among K32, not a 16/16 split. Success is the eventual episode outcome and this saturated, descriptive retention check does not label the first chunk as correct. Success is never used to train a head or choose a medoid. The true-action oracle is an action-coverage oracle, not a success oracle.

## Protocol

- Target: the complete client `actions[0]` 10x7 chunk, divided by the official checkpoint action std for each of Goal, Spatial, or Libero-10.
- Features: first server row, d0, ten action tokens, mean over HB layers 2-5 and tokens. Hidden is 1024-D. Router is tested both after pooling layers to 32-D and with layer identity retained as 4x32.
- Noise baseline: the exact first request noise is reconstructed as NumPy PCG64 `standard_normal((10,24))`, cast to float32. Both all 24 model dimensions and the seven live action dimensions are shown.
- Semantics: captured hidden values are HB-MLP/gate inputs after layer attention, not expert outputs or post-HB representations.
- Main CV: a complete K32 init-state pool is held out within each task/checkpoint. Ridge alpha is chosen by inner state-group CV.
- Seed sensitivity: seeds are split by sorted position modulo four. Each held eight-seed block and held state are absent from training. Target centering uses only the 24 allowed seeds; feature centering uses all K32 and is explicitly transductive and seed-label-disjoint, not strictly feature-disjoint.
- Coverage: mean true-action distance to the nearest selected K8 medoid. Random coverage is its exact combinatorial expectation.
- Voting: choose the ordinary unweighted medoid among the selected eight true action chunks. No cluster weights are used. Random vote endpoints use fixed common-subset Monte Carlo; coverage stays exact.
- Scope: exploratory shadow analysis only; no candidates were actually stopped early and no closed-loop latency or success was measured.
