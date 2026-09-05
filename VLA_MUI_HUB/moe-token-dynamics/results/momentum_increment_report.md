# Momentum incremental-information audit

## Direction residuals after route speed

| source | target | variance explained discovery / confirmation | residual confirmation AUC | p |
|---|---|---:|---:|---:|
| hard_route | `hard_route|h34|W7|straightness|action_all` | 0.734 / 0.808 | 0.4960 | 0.521224 |
| soft_route | `soft_route|h34|W4|straightness|action_all` | 0.142 / 0.074 | 0.6825 | 0.044498 |
| selected_route | `selected_route|h34|W7|straightness|action_all` | 0.705 / 0.799 | 0.4762 | 0.597970 |

## Fixed-W7 cross-state classifiers

Regularization is selected by four-fold CV in state 0; AUC is evaluated only in state 42.

| horizon | model | dimensions | C | confirmation AUC | p | Bonferroni over 15 |
|---:|---|---:|---:|---:|---:|---:|
| t7 | hard_speed | 1 | 0.001 | 0.5357 | 0.373581 | 1.000000 |
| t7 | hard_momentum | 8 | 10 | 0.6111 | 0.153892 | 1.000000 |
| t7 | soft_momentum | 8 | 100 | 0.5516 | 0.317134 | 1.000000 |
| t7 | selected_momentum | 8 | 10 | 0.6389 | 0.097345 | 1.000000 |
| t7 | all_momentum | 24 | 10 | 0.5397 | 0.365632 | 1.000000 |
| t12 | hard_speed | 1 | 0.001 | 0.5794 | 0.232438 | 1.000000 |
| t12 | hard_momentum | 8 | 0.1 | 0.5040 | 0.488126 | 1.000000 |
| t12 | soft_momentum | 8 | 1 | 0.5040 | 0.486626 | 1.000000 |
| t12 | selected_momentum | 8 | 1 | 0.4921 | 0.543223 | 1.000000 |
| t12 | all_momentum | 24 | 0.01 | 0.5119 | 0.457527 | 1.000000 |
| t34 | hard_speed | 1 | 0.001 | 0.8016 | 0.001800 | 0.026999 |
| t34 | hard_momentum | 8 | 0.001 | 0.8016 | 0.001200 | 0.017999 |
| t34 | soft_momentum | 8 | 100 | 0.8056 | 0.001200 | 0.017999 |
| t34 | selected_momentum | 8 | 0.001 | 0.8056 | 0.001250 | 0.018749 |
| t34 | all_momentum | 24 | 0.1 | 0.8452 | 0.000350 | 0.005250 |

## All-momentum gain over hard-route speed

| horizon | speed AUC | all-momentum AUC | delta | paired p | bootstrap delta 95% CI |
|---:|---:|---:|---:|---:|---:|
| t7 | 0.5357 | 0.5397 | 0.0040 | 0.502925 | [-0.2976, 0.3016] |
| t12 | 0.5794 | 0.5119 | -0.0675 | 0.765712 | [-0.2381, 0.0952] |
| t34 | 0.8016 | 0.8452 | 0.0437 | 0.100095 | [-0.0159, 0.1270] |
