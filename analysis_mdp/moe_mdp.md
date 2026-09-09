# MoE routing Markov/MDP model -- running log
## 0. Corpus shape, length sentinel, baseline reproduction
**Corpus A**: 352 branches, Tmax=52, Tmin=33, 117 success / 235 fail, 4 groups, 16158 chunks.
  sentinel `T == cap(52)`: fires 42, precision(fail)=1.0000, recall=0.1787
  t=20  n_alive=352  npairs=2933  |  BASELINE hell|prev/mean8 det_auc=0.5943 (auc_succ=0.4057)  |  sentinel cap-bit det_auc=0.5668  |  oracle length T det_auc=0.9894
  t=25  n_alive=352  npairs=2933  |  BASELINE hell|prev/mean8 det_auc=0.6345 (auc_succ=0.6345)  |  sentinel cap-bit det_auc=0.5668  |  oracle length T det_auc=0.9894
  t=30  n_alive=352  npairs=2933  |  BASELINE hell|prev/mean8 det_auc=0.7988 (auc_succ=0.7988)  |  sentinel cap-bit det_auc=0.5668  |  oracle length T det_auc=0.9894
**Corpus B**: 512 branches, Tmax=52, Tmin=35, 296 success / 216 fail, 16 groups, 22883 chunks.
  sentinel `T == cap(52)`: fires 217, precision(fail)=0.9954, recall=1.0000
  t=20  n_alive=512  npairs=2008  |  BASELINE hell|prev/mean8 det_auc=0.6444 (auc_succ=0.6444)  |  sentinel cap-bit det_auc=0.9993  |  oracle length T det_auc=0.9993
  t=25  n_alive=512  npairs=2008  |  BASELINE hell|prev/mean8 det_auc=0.5249 (auc_succ=0.5249)  |  sentinel cap-bit det_auc=0.9993  |  oracle length T det_auc=0.9993
  t=30  n_alive=512  npairs=2008  |  BASELINE hell|prev/mean8 det_auc=0.7510 (auc_succ=0.7510)  |  sentinel cap-bit det_auc=0.9993  |  oracle length T det_auc=0.9993

> Reproduction target: A 0.795 / B 0.759 at t=30.
## 1. NUISANCE CHECK -- is the routing state a clock or a length code?
Label-free global fit (PCA20 + k-means on every chunk). NMI over all valid chunks between the state assignment and: t (absolute query index), T (branch length), phase t/T (10 bins), remaining T-t, and the at-cap bit.
| corpus | feat | K | NMI(state,t) | NMI(state,T) | NMI(state,phase) | NMI(state,T-t) | NMI(state,atcap) | AMI(state,T) |
|---|---|---|---|---|---|---|---|---|
| A | cell1280 | 4 | 0.161 | 0.006 | 0.190 | 0.104 | 0.002 | 0.005 |
| A | cell1280 | 8 | 0.237 | 0.030 | 0.270 | 0.186 | 0.004 | 0.028 |
| A | cell1280 | 16 | 0.277 | 0.051 | 0.307 | 0.252 | 0.005 | 0.048 |
| A | cell1280 | 32 | 0.313 | 0.070 | 0.345 | 0.311 | 0.006 | 0.064 |
| A | cell1280 | 64 | 0.320 | 0.084 | 0.341 | 0.340 | 0.009 | 0.073 |
| A | mean32 | 4 | 0.151 | 0.011 | 0.185 | 0.121 | 0.000 | 0.010 |
| A | mean32 | 8 | 0.188 | 0.026 | 0.218 | 0.173 | 0.002 | 0.025 |
| A | mean32 | 16 | 0.260 | 0.043 | 0.288 | 0.241 | 0.004 | 0.040 |
| A | mean32 | 32 | 0.290 | 0.066 | 0.320 | 0.290 | 0.007 | 0.060 |
| A | mean32 | 64 | 0.308 | 0.094 | 0.326 | 0.334 | 0.010 | 0.084 |
| B | cell1280 | 4 | 0.291 | 0.072 | 0.264 | 0.172 | 0.094 | 0.071 |
| B | cell1280 | 8 | 0.423 | 0.058 | 0.354 | 0.261 | 0.069 | 0.057 |
| B | cell1280 | 16 | 0.513 | 0.062 | 0.419 | 0.343 | 0.072 | 0.060 |
| B | cell1280 | 32 | 0.542 | 0.060 | 0.433 | 0.378 | 0.064 | 0.055 |
| B | cell1280 | 64 | 0.540 | 0.069 | 0.421 | 0.391 | 0.068 | 0.061 |
| B | mean32 | 4 | 0.232 | 0.070 | 0.170 | 0.133 | 0.092 | 0.069 |
| B | mean32 | 8 | 0.298 | 0.059 | 0.221 | 0.185 | 0.069 | 0.058 |
| B | mean32 | 16 | 0.381 | 0.054 | 0.300 | 0.252 | 0.059 | 0.052 |
| B | mean32 | 32 | 0.427 | 0.063 | 0.336 | 0.298 | 0.065 | 0.058 |
| B | mean32 | 64 | 0.467 | 0.071 | 0.352 | 0.337 | 0.067 | 0.064 |


### 1b. Reference ceilings and the at-fixed-t view
`clock_ceiling` = NMI(K-quantile-bins of t, t): the score a state that is nothing but a K-level clock would get. `frac_of_clock` = NMI(state,t)/clock_ceiling.
| corpus | feat | K | NMI(s,t) | clock_ceil | frac_of_clock | NMI(s,T) | len_ceil | frac_of_len | t=30: NMI(s_t,T) | t=30: NMI(s_t,succ) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | cell1280 | 4 | 0.161 | 0.510 | 0.315 | 0.006 | 0.634 | 0.010 | 0.263 | 0.228 |
| A | cell1280 | 8 | 0.237 | 0.666 | 0.356 | 0.030 | 0.741 | 0.041 | 0.304 | 0.189 |
| A | cell1280 | 16 | 0.277 | 0.777 | 0.356 | 0.051 | 0.757 | 0.068 | 0.335 | 0.186 |
| A | cell1280 | 32 | 0.313 | 0.849 | 0.369 | 0.070 | 0.735 | 0.095 | 0.396 | 0.232 |
| A | cell1280 | 64 | 0.320 | 0.865 | 0.370 | 0.084 | 0.701 | 0.119 | 0.411 | 0.211 |
| A | mean32 | 4 | 0.151 | 0.510 | 0.296 | 0.011 | 0.634 | 0.017 | 0.112 | 0.004 |
| A | mean32 | 8 | 0.188 | 0.666 | 0.281 | 0.026 | 0.741 | 0.035 | 0.259 | 0.182 |
| A | mean32 | 16 | 0.260 | 0.777 | 0.334 | 0.043 | 0.757 | 0.057 | 0.278 | 0.148 |
| A | mean32 | 32 | 0.290 | 0.849 | 0.342 | 0.066 | 0.735 | 0.090 | 0.343 | 0.183 |
| A | mean32 | 64 | 0.308 | 0.865 | 0.356 | 0.094 | 0.701 | 0.134 | 0.389 | 0.195 |
| B | cell1280 | 4 | 0.291 | 0.508 | 0.572 | 0.072 | 0.609 | 0.119 | 0.379 | 0.323 |
| B | cell1280 | 8 | 0.423 | 0.667 | 0.635 | 0.058 | 0.617 | 0.094 | 0.378 | 0.280 |
| B | cell1280 | 16 | 0.513 | 0.779 | 0.659 | 0.062 | 0.614 | 0.101 | 0.389 | 0.302 |
| B | cell1280 | 32 | 0.542 | 0.849 | 0.639 | 0.060 | 0.569 | 0.105 | 0.364 | 0.239 |
| B | cell1280 | 64 | 0.540 | 0.869 | 0.621 | 0.069 | 0.523 | 0.131 | 0.355 | 0.207 |
| B | mean32 | 4 | 0.232 | 0.508 | 0.456 | 0.070 | 0.609 | 0.114 | 0.299 | 0.268 |
| B | mean32 | 8 | 0.298 | 0.667 | 0.447 | 0.059 | 0.617 | 0.096 | 0.328 | 0.219 |
| B | mean32 | 16 | 0.381 | 0.779 | 0.489 | 0.054 | 0.614 | 0.088 | 0.347 | 0.228 |
| B | mean32 | 32 | 0.427 | 0.849 | 0.503 | 0.063 | 0.569 | 0.110 | 0.347 | 0.207 |
| B | mean32 | 64 | 0.467 | 0.869 | 0.538 | 0.071 | 0.523 | 0.136 | 0.342 | 0.204 |


## 2. CHAIN STRUCTURE -- is there an absorbing / trap set at all?
### 2a. Self-transition and closed-class decomposition (eps = min edge prob kept)
| corpus | feat | K | max P(s->s) | mean P(s->s) | #SCC eps=0 | maxSCC eps=0 | #closed eps=0 | #closed eps=.02 | #closed eps=.05 | states in closed eps=.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | cell1280 | 4 | 0.850 | 0.752 | 1 | 4 | 1 | 1 | 1 | 4 |
| A | cell1280 | 8 | 0.913 | 0.740 | 1 | 8 | 1 | 1 | 2 | 7 |
| A | cell1280 | 16 | 0.911 | 0.588 | 1 | 16 | 1 | 1 | 1 | 1 |
| A | cell1280 | 32 | 0.905 | 0.466 | 1 | 32 | 1 | 1 | 2 | 2 |
| A | cell1280 | 64 | 0.896 | 0.281 | 7 | 58 | 1 | 1 | 2 | 2 |
| A | mean32 | 4 | 0.800 | 0.761 | 1 | 4 | 1 | 1 | 1 | 3 |
| A | mean32 | 8 | 0.731 | 0.667 | 1 | 8 | 1 | 1 | 1 | 7 |
| A | mean32 | 16 | 0.743 | 0.580 | 1 | 16 | 1 | 1 | 1 | 16 |
| A | mean32 | 32 | 0.844 | 0.499 | 1 | 32 | 1 | 1 | 2 | 32 |
| A | mean32 | 64 | 0.881 | 0.376 | 1 | 64 | 1 | 1 | 2 | 2 |
| B | cell1280 | 4 | 0.937 | 0.828 | 1 | 4 | 1 | 1 | 2 | 4 |
| B | cell1280 | 8 | 0.937 | 0.676 | 1 | 8 | 1 | 1 | 2 | 6 |
| B | cell1280 | 16 | 0.937 | 0.516 | 4 | 13 | 1 | 1 | 1 | 1 |
| B | cell1280 | 32 | 0.917 | 0.335 | 6 | 27 | 1 | 1 | 2 | 2 |
| B | cell1280 | 64 | 0.882 | 0.241 | 16 | 49 | 1 | 1 | 3 | 3 |
| B | mean32 | 4 | 0.923 | 0.743 | 1 | 4 | 1 | 1 | 2 | 4 |
| B | mean32 | 8 | 0.933 | 0.569 | 1 | 8 | 1 | 1 | 1 | 1 |
| B | mean32 | 16 | 0.915 | 0.499 | 1 | 16 | 1 | 1 | 2 | 2 |
| B | mean32 | 32 | 0.883 | 0.372 | 1 | 32 | 1 | 1 | 2 | 2 |
| B | mean32 | 64 | 0.887 | 0.264 | 1 | 64 | 1 | 1 | 4 | 4 |

### 2b. Absorption probability h(s) = P(eventual failure | state s), and whether it is just a clock
| corpus | feat | K | h range | h std | E[remaining steps] range | Spearman(h, E[steps]) |
|---|---|---|---|---|---|---|
| A | cell1280 | 4 | 0.565-0.666 | 0.044 | 43.3-47.0 | +0.000 |
| A | cell1280 | 8 | 0.575-0.775 | 0.064 | 35.3-45.9 | +0.143 |
| A | cell1280 | 16 | 0.589-0.782 | 0.063 | 32.8-45.9 | -0.168 |
| A | cell1280 | 32 | 0.415-0.775 | 0.070 | 24.6-45.7 | -0.274 |
| A | cell1280 | 64 | 0.230-0.743 | 0.084 | 14.3-45.3 | -0.236 |
| A | mean32 | 4 | 0.609-0.676 | 0.026 | 41.8-45.8 | +0.800 |
| A | mean32 | 8 | 0.579-0.692 | 0.033 | 40.7-46.0 | +0.167 |
| A | mean32 | 16 | 0.574-0.722 | 0.033 | 36.4-46.2 | -0.318 |
| A | mean32 | 32 | 0.348-0.731 | 0.069 | 22.7-45.5 | -0.274 |
| A | mean32 | 64 | 0.275-0.719 | 0.067 | 18.3-45.0 | -0.457 |
| B | cell1280 | 4 | 0.396-0.700 | 0.125 | 36.6-44.4 | -0.800 |
| B | cell1280 | 8 | 0.340-0.682 | 0.101 | 31.0-47.2 | +0.429 |
| B | cell1280 | 16 | 0.273-0.699 | 0.089 | 21.8-43.9 | +0.291 |
| B | cell1280 | 32 | 0.252-0.720 | 0.097 | 19.4-43.6 | +0.312 |
| B | cell1280 | 64 | 0.183-0.633 | 0.074 | 15.5-44.6 | +0.222 |
| B | mean32 | 4 | 0.418-0.656 | 0.102 | 39.2-46.6 | -1.000 |
| B | mean32 | 8 | 0.360-0.689 | 0.094 | 37.2-47.5 | -0.119 |
| B | mean32 | 16 | 0.376-0.682 | 0.087 | 34.8-47.1 | +0.241 |
| B | mean32 | 32 | 0.267-0.675 | 0.083 | 24.8-43.7 | -0.015 |
| B | mean32 | 64 | 0.157-0.632 | 0.078 | 14.9-44.1 | +0.109 |

### 2c. Dwell-time distributions, eventual success vs eventual failure
| corpus | feat | K | mean dwell S | mean dwell F | P(dwell>=7) S | P(dwell>=7) F | run-level AUC | MWU p | max-dwell-by-t30 det AUC | dwell>=7 rule: fires / fail rate (base) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | cell1280 | 4 | 3.85 | 4.94 | 0.0915 | 0.1875 | 0.573 | 7.79e-13 | 0.607 | 228 / 0.860 (0.668) |
| A | cell1280 | 8 | 3.56 | 3.95 | 0.0663 | 0.0675 | 0.510 | 3.12e-01 | 0.543 | 128 / 0.492 (0.668) |
| A | cell1280 | 16 | 2.24 | 2.64 | 0.0035 | 0.0262 | 0.525 | 7.38e-04 | 0.591 | 35 / 0.857 (0.668) |
| A | cell1280 | 32 | 1.82 | 2.05 | 0.0020 | 0.0180 | 0.497 | 6.20e-01 | 0.538 | 19 / 0.947 (0.668) |
| A | cell1280 | 64 | 1.31 | 1.52 | 0.0006 | 0.0126 | 0.508 | 8.85e-02 | 0.629 | 16 / 0.875 (0.668) |
| A | mean32 | 4 | 3.55 | 4.71 | 0.1124 | 0.2538 | 0.557 | 8.45e-09 | 0.508 | 346 / 0.662 (0.668) |
| A | mean32 | 8 | 2.73 | 3.05 | 0.0624 | 0.0511 | 0.526 | 1.44e-03 | 0.601 | 68 / 0.897 (0.668) |
| A | mean32 | 16 | 2.15 | 2.53 | 0.0280 | 0.0292 | 0.533 | 5.09e-06 | 0.577 | 50 / 0.900 (0.668) |
| A | mean32 | 32 | 1.89 | 2.12 | 0.0043 | 0.0201 | 0.499 | 8.85e-01 | 0.537 | 27 / 0.741 (0.668) |
| A | mean32 | 64 | 1.51 | 1.74 | 0.0010 | 0.0155 | 0.511 | 4.39e-02 | 0.636 | 19 / 0.895 (0.668) |
| B | cell1280 | 4 | 5.00 | 6.03 | 0.3183 | 0.3128 | 0.507 | 4.33e-01 | 0.632 | 497 / 0.408 (0.422) |
| B | cell1280 | 8 | 2.93 | 3.90 | 0.0050 | 0.0674 | 0.517 | 1.35e-02 | 0.574 | 9 / 0.222 (0.422) |
| B | cell1280 | 16 | 2.23 | 2.93 | 0.0023 | 0.0449 | 0.500 | 9.95e-01 | 0.606 | 1 / 1.000 (0.422) |
| B | cell1280 | 32 | 1.44 | 1.99 | 0.0012 | 0.0312 | 0.524 | 1.19e-08 | 0.555 | 0 / nan (0.422) |
| B | cell1280 | 64 | 1.23 | 1.70 | 0.0010 | 0.0264 | 0.534 | 6.24e-25 | 0.566 | 0 / nan (0.422) |
| B | mean32 | 4 | 3.32 | 4.17 | 0.1558 | 0.1573 | 0.523 | 1.27e-03 | 0.557 | 397 / 0.368 (0.422) |
| B | mean32 | 8 | 1.97 | 2.78 | 0.0024 | 0.0453 | 0.543 | 6.85e-15 | 0.554 | 2 / 0.500 (0.422) |
| B | mean32 | 16 | 1.80 | 2.52 | 0.0015 | 0.0372 | 0.542 | 4.53e-16 | 0.500 | 0 / nan (0.422) |
| B | mean32 | 32 | 1.48 | 1.98 | 0.0011 | 0.0297 | 0.524 | 3.49e-08 | 0.539 | 0 / nan (0.422) |
| B | mean32 | 64 | 1.26 | 1.70 | 0.0011 | 0.0260 | 0.521 | 2.09e-09 | 0.589 | 0 / nan (0.422) |


## 3. DETECTION -- absorption probability vs the scalar baseline
### 3a. Two control state spaces
| corpus | control | max raw det AUC over its 39x5 cells | max residual det AUC | reading |
|---|---|---|---|---|
| A | clock | 0.555 | 0.578 | pure clock, zero routing |
| A | shuf | 0.928 | 0.920 | ACAUSAL within-branch shuffle -- leakage sentinel |
| B | clock | 0.558 | 0.575 | pure clock, zero routing |
| B | shuf | 0.964 | 0.896 | ACAUSAL within-branch shuffle -- leakage sentinel |

The clock control is *degenerate by construction*: the protocol fixes the query index t, so every branch alive at t sits in the same clock state and the score is constant. That is the point -- the fixed-t protocol is already immune to the clock component of the state, so the NMI(state,t) of 0.32-0.54 cannot be what produces any AUC below.
The acausal shuffle is NOT a control, it is a leakage sentinel: permuting a branch's chunk order mixes post-t states into the pre-t window and reaches det AUC 0.93 (A) / 0.96 (B). Any pipeline that touches the whole episode gets ~0.95 for free. All model cells below are strictly causal (window [t-7, t]).
### 3b. Family-wise null over the whole model grid (390 cells: 2 featurisations x 5 K x 3 t x 13 variants), 200 shared within-group branch-label permutations
| corpus | statistic | observed | null mean | null q95 | null max | family-wise p |
|---|---|---|---|---|---|---|
| A | max RAW det AUC | 0.796 | 0.660 | 0.711 | 0.739 | 0.0050 |
| A | max RESIDUAL det AUC | 0.724 | 0.664 | 0.720 | 0.751 | 0.0448 |
| A | cells with per-cell p_raw < 0.05 | 145/390 = 0.372 | 0.05 | | | |
| A | cells with per-cell p_res < 0.05 | 96/390 = 0.246 | 0.05 | | | |
| B | max RAW det AUC | 0.814 | 0.622 | 0.656 | 0.673 | 0.0050 |
| B | max RESIDUAL det AUC | 0.725 | 0.625 | 0.657 | 0.668 | 0.0050 |
| B | cells with per-cell p_raw < 0.05 | 155/390 = 0.397 | 0.05 | | | |
| B | cells with per-cell p_res < 0.05 | 78/390 = 0.200 | 0.05 | | | |

### 3c. Cross-corpus replication -- the decisive test
Residual sign agreement across corpora, over 390 matched cells: **observed 0.492**, null mean 0.605, null q95 0.705, permutation p = 0.965.
Per-t breakdown (residual sign agreement; 0.5 = chance):
| t | n cells | observed agreement | null mean | null q95 | p |
|---|---|---|---|---|---|
| 20 | 130 | 0.485 | 0.606 | 0.723 | 0.925 |
| 25 | 130 | 0.500 | 0.611 | 0.731 | 0.891 |
| 30 | 130 | 0.492 | 0.599 | 0.738 | 0.866 |

Cells clearing residual p<0.05 in BOTH corpora with matching sign: **11** of 390. Expected if the two corpora were independent draws with the same per-corpus clearance rates: **9.6**.
Variants supplying those joint cells:
| variant | joint cells | share of grid |
|---|---|---|
| emp_pool_w8 | 6 | 1/13 |
| esteps_w8 | 3 | 1/13 |
| nll_w8 | 1 | 1/13 |
| emp_time | 1 | 1/13 |

### 3d. Per-variant summary at t=30 (the only t where the baseline is strong)
| variant | A raw (best K) | A res | A p_res | B raw (best K) | B res | B p_res | beats baseline in both? |
|---|---|---|---|---|---|---|---|
| absorb | 0.700 (K=64) | 0.503 | 0.965 | 0.770 (K=64) | 0.672 | 0.005 | no |
| absorb_drift | 0.652 (K=64) | 0.524 | 0.726 | 0.763 (K=64) | 0.676 | 0.005 | no |
| absorb_max_w8 | 0.628 (K=32) | 0.554 | 0.418 | 0.738 (K=64) | 0.662 | 0.005 | no |
| absorb_w8 | 0.693 (K=64) | 0.516 | 0.836 | 0.760 (K=16) | 0.653 | 0.005 | no |
| dwell | 0.684 (K=64) | 0.533 | 0.562 | 0.681 (K=64) | 0.555 | 0.184 | no |
| emp_pool | 0.757 (K=64) | 0.579 | 0.249 | 0.781 (K=64) | 0.701 | 0.005 | no |
| emp_pool_w8 | 0.796 (K=64) | 0.636 | 0.035 | 0.814 (K=64) | 0.725 | 0.005 | no |
| emp_time | 0.733 (K=64) | 0.586 | 0.209 | 0.772 (K=64) | 0.693 | 0.005 | no |
| esteps | 0.621 (K=64) | 0.537 | 0.507 | 0.739 (K=4) | 0.625 | 0.005 | no |
| esteps_w8 | 0.635 (K=8) | 0.617 | 0.020 | 0.689 (K=4) | 0.594 | 0.010 | no |
| frac_hirisk_w8 | 0.648 (K=8) | 0.607 | 0.050 | 0.776 (K=32) | 0.703 | 0.005 | no |
| nll_w8 | 0.681 (K=8) | 0.557 | 0.259 | 0.733 (K=4) | 0.712 | 0.005 | no |
| stayp_w8 | 0.673 (K=4) | 0.666 | 0.005 | 0.750 (K=32) | 0.647 | 0.005 | no |

### 3e. Does K matter?  det AUC of the two leading variants vs K, t=30, cell1280
| corpus | variant | K=4 | K=8 | K=16 | K=32 | K=64 |
|---|---|---|---|---|---|---|
| A | absorb_w8 | 0.506 | 0.560 | 0.611 | 0.638 | 0.693 |
| A | emp_pool_w8 | 0.650 | 0.734 | 0.697 | 0.740 | 0.796 |
| B | absorb_w8 | 0.699 | 0.682 | 0.760 | 0.759 | 0.731 |
| B | emp_pool_w8 | 0.699 | 0.722 | 0.746 | 0.721 | 0.805 |

## 4. Does MEMORY help? first order vs second order vs semi-Markov (cell1280, t=30)
| corpus | variant | order1 best | order2 best | semi best | baseline |
|---|---|---|---|---|---|
| A | absorb_w8 | 0.693 | 0.601 | 0.640 | 0.799 |
| A | emp_pool_w8 | 0.796 | 0.694 | 0.831 | 0.799 |
| B | absorb_w8 | 0.760 | 0.759 | 0.753 | 0.751 |
| B | emp_pool_w8 | 0.805 | 0.782 | 0.793 | 0.751 |


## 5. Seed stability of the ONE cell that beat the scalar baseline in both corpora
semi-Markov (state x dwell bucket), K=64, emp_pool_w8. Five independent redraws of the folds and of the k-means initialisation.
| corpus | t | baseline | mode | det AUC per seed | mean | min | seeds beating baseline | per-seed perm p_res |
|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.594 | order1 | 0.719, 0.727, 0.680, 0.730, 0.718 | 0.715 | 0.680 | 5/5 | 0.005, 0.005, 0.020, 0.010, 0.005 |
| A | 20 | 0.594 | semi | 0.726, 0.708, 0.690, 0.743, 0.713 | 0.716 | 0.690 | 5/5 | 0.005, 0.005, 0.015, 0.005, 0.005 |
| A | 25 | 0.635 | order1 | 0.694, 0.748, 0.681, 0.751, 0.700 | 0.715 | 0.681 | 5/5 | 0.005, 0.005, 0.005, 0.005, 0.005 |
| A | 25 | 0.635 | semi | 0.696, 0.760, 0.690, 0.766, 0.693 | 0.721 | 0.690 | 5/5 | 0.005, 0.005, 0.005, 0.005, 0.005 |
| A | 30 | 0.799 | order1 | 0.796, 0.782, 0.796, 0.822, 0.732 | 0.786 | 0.732 | 1/5 | 0.035, 0.030, 0.015, 0.005, 0.164 |
| A | 30 | 0.799 | semi | 0.831, 0.822, 0.830, 0.839, 0.770 | 0.818 | 0.770 | 4/5 | 0.015, 0.010, 0.005, 0.010, 0.119 |
| B | 20 | 0.644 | order1 | 0.558, 0.500, 0.552, 0.559, 0.516 | 0.537 | 0.500 | 0/5 | 0.831, 0.821, 0.527, 0.801, 0.960 |
| B | 20 | 0.644 | semi | 0.600, 0.540, 0.551, 0.570, 0.516 | 0.555 | 0.516 | 0/5 | 0.368, 0.876, 0.741, 0.627, 0.851 |
| B | 25 | 0.525 | order1 | 0.560, 0.530, 0.562, 0.598, 0.502 | 0.550 | 0.502 | 4/5 | 0.542, 0.622, 0.294, 0.085, 0.960 |
| B | 25 | 0.525 | semi | 0.562, 0.549, 0.582, 0.607, 0.514 | 0.563 | 0.514 | 4/5 | 0.463, 0.363, 0.129, 0.030, 0.806 |
| B | 30 | 0.751 | order1 | 0.805, 0.776, 0.755, 0.776, 0.750 | 0.772 | 0.750 | 4/5 | 0.005, 0.005, 0.005, 0.005, 0.015 |
| B | 30 | 0.751 | semi | 0.793, 0.781, 0.760, 0.771, 0.754 | 0.772 | 0.754 | 5/5 | 0.005, 0.005, 0.005, 0.005, 0.005 |

## 6. Integrity checks

* **No shared prefixes in corpus A.** The rolling-star design could have let a training
  branch and a test branch share early chunks, which would break branch-level cross-fitting.
  Checked all 15,440 within-worker branch pairs: the longest identical leading run is
  **0 chunks** for every pair. Branch-level folds are clean.
* **Every model quantity that touches the label is cross-fitted** (5 branch-level folds,
  stratified by group only). PCA and k-means are also fitted on training folds only.
* **Acausal-shuffle leakage sentinel**: permuting a branch's chunk order (so that post-t
  states enter the pre-t window) lifts the same pipeline to det AUC 0.93 (A) / 0.96 (B).
  Nothing reported here uses any chunk after the query index.
* **No hard top-4 expert IDs anywhere.** All states are clustered from the 32-dim
  probability simplices only.

## 7. VERDICT

**The Markov framing adds essentially nothing over the scalar threshold, and the reason is
that the routing chain has no trap: it is a single ergodic communicating class whose
absorption probabilities span a band narrower than the corpus base rate, so the only usable
signal left in the state sequence is how often the state changes -- which is the scalar
baseline.**

Supporting decomposition of that one line:

1. **There is no absorbing or near-absorbing set.** At eps=0 the transient graph is one
   strongly connected component in 15 of 20 configurations, and the only closed class is the
   entire state space. The "closed" singletons that appear at eps=0.05 still leak 3-12% of
   their mass per step (expected escape in 8-30 chunks). "Stopped" and "trapped" are simply
   not native to this chain -- the concepts the framing promised to make native do not exist
   in the object.
2. **Stickiness is not failure.** The near-closed singleton at A/cell1280/K=64 has 107
   visitors and a 13% failure rate against a 67% base; at K=8 another has 88 visitors and a
   94% failure rate. High self-transition occurs in both outcome directions.
3. **The chain computation is strictly worse than a lookup table, and extra memory is a
   coin flip.** The absorption solve (`absorb`, `absorb_w8`) never beats the memoryless
   per-state empirical failure rate (`emp_pool_w8`) at any K, feat, t or corpus -- solving
   for absorption probabilities on the estimated chain is uniformly worse than just reading
   a cross-fitted failure rate off the current state. Adding memory does nothing either:
   at **matched K**, second order wins 37 of 72 comparisons with a median delta of
   **+0.002**, and the dwell-augmented semi-Markov state wins 55 of 120 with a median of
   **-0.004** (A +0.001, B -0.005). Second order's best is lower than first order's only
   because the pair state space cannot be pushed past K=16 without blowing up.
4. **What little increment exists is not chain structure, it is supervised discretisation.**
   The only cell that beats the scalar baseline in both corpora is semi-Markov K=64
   `emp_pool_w8` at t=30: A 0.818 +/- 0.028 (range 0.770-0.839) vs 0.799, B 0.772 +/- 0.016
   (range 0.754-0.793) vs 0.751 over five seed redraws, i.e. **+0.019 / +0.021**. The same cell *loses* to the baseline in B at
   t=20 (0.555 vs 0.644, permutation p_res 0.37-0.88, null) and is null in B at t=25. It is
   a one-timepoint, two-hundredths-of-AUC effect, found after searching ~570 cells.
5. **The state structure does not replicate across corpora.** Cross-corpus residual sign
   agreement over the 390-cell grid is **0.492 against a null mean of 0.605 (p = 0.965)** --
   below chance. No individual variant exceeds its own null (all p >= 0.124). The
   pre-specified absorption variant is the worst offender at 0.300 against a null of 0.763:
   whatever `absorb` reads beyond the baseline, it reads with *opposite* sign in the two
   corpora. Joint clearance is 11 of 390 cells against 9.6 expected by chance.
6. **K does not stabilise.** `emp_pool_w8` rises monotonically to the largest K tested
   (A 0.650 -> 0.796, B 0.699 -> 0.805 from K=4 to K=64) while mean dwell falls to 1.2-1.5
   chunks. The model gets better precisely as it stops being a state machine and becomes a
   nearest-prototype regressor on the routing vector. There is no stable K range because
   there is no scale at which the discretisation is the right description.

### Why the state structure does not help (the interesting part)

The dwell-time result is the whole story in miniature. Mean dwell is longer for eventual
failures in **20 of 20** (corpus x featurisation x K) configurations -- A/cell1280/K=4:
4.94 vs 3.85 chunks; B/cell1280/K=64: 1.70 vs 1.23 -- and this is the single most reliable
structural difference in the entire analysis. But longer dwell *is* fewer state changes *is*
lower chunk-to-chunk route movement, which is exactly what the Hellinger-to-previous-chunk
baseline measures, in continuous form rather than quantised through a k-means boundary.
Discretising first throws away the magnitude of the movement and keeps only whether it
crossed a cell boundary; the run-level dwell AUC is 0.50-0.57 and the branch-level
longest-dwell AUC is 0.50-0.64, against a baseline of 0.75-0.80. The Markov model then
spends its remaining capacity estimating transition probabilities among states between which
the branch is, at the useful K, moving almost every chunk.

So the chain does not fail because route change is uninformative. It fails because
**chunk-to-chunk route change is the only informative thing in the sequence, and quantising
the routing vector into states is a lossy way to measure it.** This is the same single
shared factor the two brute-force sweeps found across all layers and both token families,
now recovered a third way: an absorption model estimated on 64 routing states reduces, in
the end, to a noisier estimate of the rate at which the route moves.

### Scope note

Per project convention the matched controls are recorded but not made the headline. In
corpus B the one-bit sentinel `episode_length == step_cap` reaches within-group det AUC
0.9993 (precision 0.9954, recall 1.0000) and the outcome is effectively the length bit; in
corpus A the bit is precise but low-recall (1.000 / 0.179, det AUC 0.567) while the oracle
length still reaches 0.989. Every MoE number above is far below both. The routing states are
*not* a length code (frac-of-length-ceiling 0.01-0.14), which is what killed the 2026-08-28
clustering; they are partly a clock (frac-of-clock-ceiling 0.28-0.37 in A, 0.45-0.66 in B),
but the fixed-t protocol is immune to that by construction -- a pure clock state space scores
a maximum of 0.555 (A) / 0.558 (B) over its entire block.
