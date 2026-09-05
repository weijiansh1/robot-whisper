# Terminal window ablation

`W` is the number of terminal distance pairs, not the number of raw control rows.
For lag 1, W7 exactly matches the prior eight-control-row window.
Every fixed curve uses aligned T1-T10 routes and averages over token, layer, and flow.

## t34 fixed lag-1 action-token curves

Each cell is confirmation oriented AUC (one-sided permutation p). Direction is fixed in state 0 and evaluated in state 42.

| metric | W1 | W2 | W3 | W4 | W5 | W6 | W7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| top4_ja | 0.718 (0.0193) | 0.774 (0.0046) | 0.722 (0.0182) | 0.726 (0.0160) | 0.722 (0.0180) | 0.806 (0.0013) | 0.802 (0.0017) |
| top4_cos | 0.702 (0.0270) | 0.760 (0.0059) | 0.718 (0.0195) | 0.726 (0.0160) | 0.718 (0.0195) | 0.798 (0.0017) | 0.802 (0.0018) |
| soft_wj | 0.500 (0.5007) | 0.516 (0.4430) | 0.528 (0.3978) | 0.583 (0.2182) | 0.587 (0.2050) | 0.615 (0.1395) | 0.575 (0.2400) |
| soft_cos | 0.405 (0.8180) | 0.448 (0.6894) | 0.504 (0.4874) | 0.571 (0.2533) | 0.607 (0.1576) | 0.631 (0.1090) | 0.587 (0.2096) |
| selected_wj | 0.714 (0.0215) | 0.770 (0.0050) | 0.722 (0.0182) | 0.726 (0.0160) | 0.722 (0.0180) | 0.798 (0.0018) | 0.802 (0.0017) |
| selected_cos | 0.706 (0.0248) | 0.758 (0.0061) | 0.718 (0.0196) | 0.726 (0.0159) | 0.726 (0.0162) | 0.802 (0.0015) | 0.802 (0.0017) |

## Discovery-selected window per metric and horizon

Only W is selected in state 0; lag is fixed at 1 and comparison at action_all.

| metric | horizon | selected W | discovery AUC | confirmation AUC | p | Bonferroni over 18 metric-horizon families |
|---|---:|---:|---:|---:|---:|---:|
| top4_ja | t7 | W2 | 0.7063 | 0.6429 | 0.090045 | 1.000000 |
| top4_ja | t12 | W7 | 0.6230 | 0.5952 | 0.188241 | 1.000000 |
| top4_ja | t34 | W7 | 0.0159 | 0.8016 | 0.001750 | 0.031498 |
| top4_cos | t7 | W2 | 0.7063 | 0.6508 | 0.076746 | 1.000000 |
| top4_cos | t12 | W7 | 0.6230 | 0.5833 | 0.220439 | 1.000000 |
| top4_cos | t34 | W7 | 0.0198 | 0.8016 | 0.001800 | 0.032398 |
| soft_wj | t7 | W4 | 0.4167 | 0.4087 | 0.811909 | 1.000000 |
| soft_wj | t12 | W1 | 0.4127 | 0.5516 | 0.326284 | 1.000000 |
| soft_wj | t34 | W7 | 0.1786 | 0.5754 | 0.239988 | 1.000000 |
| soft_cos | t7 | W5 | 0.4008 | 0.4167 | 0.788261 | 1.000000 |
| soft_cos | t12 | W2 | 0.3968 | 0.5238 | 0.426529 | 1.000000 |
| soft_cos | t34 | W7 | 0.2103 | 0.5873 | 0.209640 | 1.000000 |
| selected_wj | t7 | W2 | 0.6944 | 0.6389 | 0.096845 | 1.000000 |
| selected_wj | t12 | W7 | 0.6190 | 0.5913 | 0.198690 | 1.000000 |
| selected_wj | t34 | W7 | 0.0159 | 0.8016 | 0.001750 | 0.031498 |
| selected_cos | t7 | W2 | 0.7103 | 0.6429 | 0.091345 | 1.000000 |
| selected_cos | t12 | W7 | 0.6230 | 0.5754 | 0.244288 | 1.000000 |
| selected_cos | t34 | W7 | 0.0198 | 0.8016 | 0.001700 | 0.030598 |

## Discovery-selected window and lag (all action tokens)

W and lag are selected in state 0; T1-T10 are always averaged.

| metric | horizon | selected W/lag | discovery maxT p | confirmation AUC | p | Bonferroni over 18 |
|---|---:|---|---:|---:|---:|---:|
| top4_ja | t7 | W2 / lag1 | 0.538173 | 0.6429 | 0.089196 | 1.000000 |
| top4_ja | t12 | W2 / lag4 | 0.299935 | 0.3611 | 0.908205 | 1.000000 |
| top4_ja | t34 | W5 / lag5 | 0.000050 | 0.8373 | 0.000550 | 0.009900 |
| top4_cos | t7 | W2 / lag1 | 0.523574 | 0.6508 | 0.075546 | 1.000000 |
| top4_cos | t12 | W2 / lag4 | 0.351732 | 0.3790 | 0.873456 | 1.000000 |
| top4_cos | t34 | W5 / lag5 | 0.000050 | 0.8452 | 0.000300 | 0.005400 |
| soft_wj | t7 | W2 / lag3 | 0.252837 | 0.3413 | 0.936053 | 1.000000 |
| soft_wj | t12 | W2 / lag4 | 0.005250 | 0.4206 | 0.772711 | 1.000000 |
| soft_wj | t34 | W7 / lag4 | 0.002650 | 0.5913 | 0.204440 | 1.000000 |
| soft_cos | t7 | W2 / lag3 | 0.268037 | 0.3571 | 0.917654 | 1.000000 |
| soft_cos | t12 | W1 / lag4 | 0.006250 | 0.5675 | 0.266487 | 1.000000 |
| soft_cos | t34 | W7 / lag5 | 0.005400 | 0.5556 | 0.306885 | 1.000000 |
| selected_wj | t7 | W2 / lag1 | 0.634468 | 0.6389 | 0.094945 | 1.000000 |
| selected_wj | t12 | W2 / lag4 | 0.298085 | 0.3571 | 0.917404 | 1.000000 |
| selected_wj | t34 | W5 / lag5 | 0.000050 | 0.8254 | 0.000850 | 0.015299 |
| selected_cos | t7 | W2 / lag1 | 0.503825 | 0.6429 | 0.089096 | 1.000000 |
| selected_cos | t12 | W2 / lag4 | 0.332333 | 0.3690 | 0.899605 | 1.000000 |
| selected_cos | t34 | W5 / lag5 | 0.000050 | 0.8373 | 0.000400 | 0.007200 |

## Joint window-lag-token scan

W1-W7, lag1-lag5, and T0-T10/action_all are selected in state 0 and tested in state 42.

| metric | horizon | selected feature | discovery maxT p | confirmation AUC | p | Bonferroni over 18 |
|---|---:|---|---:|---:|---:|---:|
| top4_ja | t7 | `top4_ja|h07|W1|lag2|T9` | 0.246088 | 0.5694 | 0.258287 | 1.000000 |
| top4_ja | t12 | `top4_ja|h12|W4|lag4|T2` | 0.155442 | 0.4643 | 0.635818 | 1.000000 |
| top4_ja | t34 | `top4_ja|h34|W2|lag5|T1` | 0.000050 | 0.8333 | 0.000650 | 0.011699 |
| top4_cos | t7 | `top4_cos|h07|W1|lag2|T9` | 0.187741 | 0.5298 | 0.391180 | 1.000000 |
| top4_cos | t12 | `top4_cos|h12|W4|lag4|T2` | 0.128194 | 0.4802 | 0.575521 | 1.000000 |
| top4_cos | t34 | `top4_cos|h34|W2|lag5|T1` | 0.000050 | 0.8393 | 0.000400 | 0.007200 |
| soft_wj | t7 | `soft_wj|h07|W2|lag1|T0` | 0.217239 | 0.5198 | 0.430978 | 1.000000 |
| soft_wj | t12 | `soft_wj|h12|W2|lag4|action_all` | 0.057897 | 0.4206 | 0.778211 | 1.000000 |
| soft_wj | t34 | `soft_wj|h34|W7|lag1|T0` | 0.000050 | 0.8849 | 0.000150 | 0.002700 |
| soft_cos | t7 | `soft_cos|h07|W3|lag1|T0` | 0.924954 | 0.6310 | 0.111194 | 1.000000 |
| soft_cos | t12 | `soft_cos|h12|W1|lag4|action_all` | 0.068097 | 0.5675 | 0.272836 | 1.000000 |
| soft_cos | t34 | `soft_cos|h34|W7|lag2|T0` | 0.000100 | 0.6984 | 0.029699 | 0.534573 |
| selected_wj | t7 | `selected_wj|h07|W1|lag2|T9` | 0.248938 | 0.5714 | 0.256637 | 1.000000 |
| selected_wj | t12 | `selected_wj|h12|W4|lag4|T2` | 0.134143 | 0.4603 | 0.654567 | 1.000000 |
| selected_wj | t34 | `selected_wj|h34|W5|lag5|T10` | 0.000050 | 0.7381 | 0.010799 | 0.194390 |
| selected_cos | t7 | `selected_cos|h07|W1|lag2|T9` | 0.220339 | 0.5317 | 0.389281 | 1.000000 |
| selected_cos | t12 | `selected_cos|h12|W4|lag4|T2` | 0.116294 | 0.4762 | 0.596720 | 1.000000 |
| selected_cos | t34 | `selected_cos|h34|W5|lag5|T10` | 0.000050 | 0.7222 | 0.018099 | 0.325784 |
