# HB-MoE route distance comparisons

## Definitions

- `cross`: compare the same token slot between control chunks t and t-k.
- `within`: compare two token slots inside one control chunk; no control-step distance is used.
- `top4_ja`: set Jaccard distance over authoritative Top-4 IDs.
- `top4_cos`: cosine distance over the authoritative Top-4 indicator.
- `soft_wj`: weighted-Jaccard distance over all 32 router probabilities.
- `soft_cos`: cosine distance over all 32 router probabilities.
- `selected_wj`: weighted-Jaccard distance over normalized actual Top-4 combine weights.
- `selected_cos`: cosine distance over normalized actual Top-4 combine weights.

## Fixed aggregate tests (no token or lag selection)

- `cross`: lag 1, average the aligned T1-T10 distances.
- `within`: average all 45 action-token pair distances in the same chunk.

| metric | mode | horizon | discovery AUC | discovery maxT p | confirmation oriented AUC | p | Bonferroni over 36 |
|---|---|---:|---:|---:|---:|---:|---:|
| top4_ja | cross | t7 | 0.5754 | 0.999100 | 0.5357 | 0.376281 | 1.000000 |
| top4_ja | cross | t12 | 0.6230 | 0.956302 | 0.5952 | 0.191940 | 1.000000 |
| top4_ja | cross | t34 | 0.0159 | 0.000050 | 0.8016 | 0.001700 | 0.061197 |
| top4_cos | cross | t7 | 0.5754 | 0.999100 | 0.5476 | 0.332733 | 1.000000 |
| top4_cos | cross | t12 | 0.6230 | 0.956302 | 0.5833 | 0.217189 | 1.000000 |
| top4_cos | cross | t34 | 0.0198 | 0.000050 | 0.8016 | 0.001700 | 0.061197 |
| soft_wj | cross | t7 | 0.4405 | 0.999950 | 0.3889 | 0.855607 | 1.000000 |
| soft_wj | cross | t12 | 0.5278 | 1.000000 | 0.5119 | 0.458627 | 1.000000 |
| soft_wj | cross | t34 | 0.1786 | 0.021249 | 0.5754 | 0.240338 | 1.000000 |
| soft_cos | cross | t7 | 0.4167 | 0.998150 | 0.3929 | 0.851057 | 1.000000 |
| soft_cos | cross | t12 | 0.4841 | 1.000000 | 0.4762 | 0.594770 | 1.000000 |
| soft_cos | cross | t34 | 0.2103 | 0.062197 | 0.5873 | 0.212739 | 1.000000 |
| selected_wj | cross | t7 | 0.5794 | 0.998750 | 0.5317 | 0.387031 | 1.000000 |
| selected_wj | cross | t12 | 0.6190 | 0.965752 | 0.5913 | 0.198590 | 1.000000 |
| selected_wj | cross | t34 | 0.0159 | 0.000050 | 0.8016 | 0.001900 | 0.068397 |
| selected_cos | cross | t7 | 0.5635 | 0.999900 | 0.5476 | 0.333683 | 1.000000 |
| selected_cos | cross | t12 | 0.6230 | 0.956302 | 0.5754 | 0.243938 | 1.000000 |
| selected_cos | cross | t34 | 0.0198 | 0.000050 | 0.8016 | 0.002150 | 0.077396 |
| top4_ja | within | t7 | 0.3849 | 0.973451 | 0.5952 | 0.191190 | 1.000000 |
| top4_ja | within | t12 | 0.6349 | 0.917554 | 0.2897 | 0.980551 | 1.000000 |
| top4_ja | within | t34 | 0.3770 | 0.956302 | 0.5992 | 0.176341 | 1.000000 |
| top4_cos | within | t7 | 0.3690 | 0.933003 | 0.6190 | 0.135243 | 1.000000 |
| top4_cos | within | t12 | 0.6429 | 0.882606 | 0.2698 | 0.987851 | 1.000000 |
| top4_cos | within | t34 | 0.3254 | 0.685066 | 0.6349 | 0.104745 | 1.000000 |
| soft_wj | within | t7 | 0.2937 | 0.445378 | 0.4325 | 0.744763 | 1.000000 |
| soft_wj | within | t12 | 0.4087 | 0.995700 | 0.4921 | 0.538623 | 1.000000 |
| soft_wj | within | t34 | 0.3254 | 0.685066 | 0.6032 | 0.173741 | 1.000000 |
| soft_cos | within | t7 | 0.2817 | 0.360832 | 0.4127 | 0.802460 | 1.000000 |
| soft_cos | within | t12 | 0.3373 | 0.770111 | 0.4921 | 0.541673 | 1.000000 |
| soft_cos | within | t34 | 0.3452 | 0.819409 | 0.5516 | 0.315234 | 1.000000 |
| selected_wj | within | t7 | 0.3849 | 0.973451 | 0.5952 | 0.186291 | 1.000000 |
| selected_wj | within | t12 | 0.6310 | 0.933003 | 0.2937 | 0.978501 | 1.000000 |
| selected_wj | within | t34 | 0.3849 | 0.973451 | 0.6032 | 0.167192 | 1.000000 |
| selected_cos | within | t7 | 0.3810 | 0.965752 | 0.6230 | 0.121444 | 1.000000 |
| selected_cos | within | t12 | 0.6389 | 0.901255 | 0.2619 | 0.990250 | 1.000000 |
| selected_cos | within | t34 | 0.3373 | 0.770111 | 0.6429 | 0.089096 | 1.000000 |

## Mode by horizon

| mode | horizon | selected feature | discovery maxT p | confirmation oriented AUC | p | Bonferroni over 6 |
|---|---:|---|---:|---:|---:|---:|
| cross | t7 | `soft_wj|cross|h07|lag4|T0` | 0.677766 | 0.6429 | 0.088296 | 0.529774 |
| cross | t12 | `selected_cos|cross|h12|lag4|T2` | 0.098495 | 0.4762 | 0.597220 | 1.000000 |
| cross | t34 | `selected_wj|cross|h34|lag1|T7` | 0.000100 | 0.7262 | 0.015349 | 0.092095 |
| within | t7 | `soft_cos|within|h07|level|T5_T9` | 0.759412 | 0.5079 | 0.480076 | 1.000000 |
| within | t12 | `top4_cos|within|h12|level|T4_T7` | 0.112944 | 0.3175 | 0.960202 | 1.000000 |
| within | t34 | `top4_cos|within|h34|level|T1_T10` | 0.001350 | 0.6429 | 0.086046 | 0.516274 |

## Metric, mode, and horizon

| metric | mode | horizon | selected comparison | discovery AUC | confirmation oriented AUC | p | Bonferroni over 36 |
|---|---|---:|---|---:|---:|---:|---:|
| top4_ja | cross | t7 | `T5` | 0.2738 | 0.3810 | 0.873106 | 1.000000 |
| top4_ja | cross | t12 | `T2` | 0.1548 | 0.4643 | 0.630168 | 1.000000 |
| top4_ja | cross | t34 | `T7` | 0.0159 | 0.7302 | 0.012599 | 0.453577 |
| top4_ja | within | t7 | `T4_T5` | 0.2738 | 0.5159 | 0.448828 | 1.000000 |
| top4_ja | within | t12 | `T4_T7` | 0.8294 | 0.3175 | 0.960002 | 1.000000 |
| top4_ja | within | t34 | `T1_T10` | 0.0714 | 0.6508 | 0.081146 | 1.000000 |
| top4_cos | cross | t7 | `T5` | 0.2738 | 0.3750 | 0.886156 | 1.000000 |
| top4_cos | cross | t12 | `T2` | 0.1508 | 0.4802 | 0.574871 | 1.000000 |
| top4_cos | cross | t34 | `T7` | 0.0198 | 0.7401 | 0.009150 | 0.329384 |
| top4_cos | within | t7 | `T4_T5` | 0.2679 | 0.5040 | 0.485226 | 1.000000 |
| top4_cos | within | t12 | `T4_T7` | 0.8353 | 0.3175 | 0.960552 | 1.000000 |
| top4_cos | within | t34 | `T1_T10` | 0.0714 | 0.6429 | 0.089796 | 1.000000 |
| soft_wj | cross | t7 | `T0` | 0.7857 | 0.6429 | 0.087496 | 1.000000 |
| soft_wj | cross | t12 | `T0` | 0.7579 | 0.5278 | 0.409080 | 1.000000 |
| soft_wj | cross | t34 | `T0` | 0.0159 | 0.8849 | 0.000050 | 0.001800 |
| soft_wj | within | t7 | `T2_T9` | 0.2540 | 0.5317 | 0.395830 | 1.000000 |
| soft_wj | within | t12 | `T3_T9` | 0.2302 | 0.6111 | 0.154542 | 1.000000 |
| soft_wj | within | t34 | `T2_T9` | 0.1032 | 0.6667 | 0.055547 | 1.000000 |
| soft_cos | cross | t7 | `T0` | 0.2778 | 0.6667 | 0.057897 | 1.000000 |
| soft_cos | cross | t12 | `T7` | 0.2540 | 0.5952 | 0.189691 | 1.000000 |
| soft_cos | cross | t34 | `T0` | 0.0476 | 0.6944 | 0.033398 | 1.000000 |
| soft_cos | within | t7 | `T5_T9` | 0.2381 | 0.5079 | 0.481476 | 1.000000 |
| soft_cos | within | t12 | `T0_T1` | 0.7579 | 0.4048 | 0.822009 | 1.000000 |
| soft_cos | within | t34 | `T2_T6` | 0.0913 | 0.6667 | 0.059547 | 1.000000 |
| selected_wj | cross | t7 | `T5` | 0.2698 | 0.3849 | 0.866957 | 1.000000 |
| selected_wj | cross | t12 | `T2` | 0.1508 | 0.4603 | 0.647768 | 1.000000 |
| selected_wj | cross | t34 | `T7` | 0.0159 | 0.7262 | 0.015549 | 0.559772 |
| selected_wj | within | t7 | `T4_T5` | 0.2778 | 0.5119 | 0.460677 | 1.000000 |
| selected_wj | within | t12 | `T4_T7` | 0.8333 | 0.3214 | 0.957852 | 1.000000 |
| selected_wj | within | t34 | `T1_T10` | 0.0794 | 0.6349 | 0.101795 | 1.000000 |
| selected_cos | cross | t7 | `T3` | 0.2738 | 0.4563 | 0.670616 | 1.000000 |
| selected_cos | cross | t12 | `T2` | 0.1468 | 0.4762 | 0.600020 | 1.000000 |
| selected_cos | cross | t34 | `T7` | 0.0198 | 0.7381 | 0.011749 | 0.422979 |
| selected_cos | within | t7 | `T4_T5` | 0.2659 | 0.5000 | 0.510324 | 1.000000 |
| selected_cos | within | t12 | `T4_T7` | 0.8294 | 0.3135 | 0.965952 | 1.000000 |
| selected_cos | within | t34 | `T1_T10` | 0.0794 | 0.6389 | 0.096845 | 1.000000 |
