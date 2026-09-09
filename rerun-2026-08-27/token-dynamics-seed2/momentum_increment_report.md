# Momentum incremental-information audit

## Direction residuals after route speed

| source | target | variance explained discovery / confirmation | residual confirmation AUC | p |
|---|---|---:|---:|---:|
| hard_route | `hard_route|h34|W7|straightness|action_all` | 0.734 / 0.808 | 0.4960 | 0.518824 |
| soft_route | `soft_route|h34|W4|straightness|action_all` | 0.142 / 0.074 | 0.6825 | 0.041848 |
| selected_route | `selected_route|h34|W7|straightness|action_all` | 0.705 / 0.799 | 0.4762 | 0.596370 |

## Fixed-W7 cross-state classifiers

Regularization is selected by four-fold CV in state 0; AUC is evaluated only in state 42.

| horizon | model | dimensions | C | confirmation AUC | p | Bonferroni over 15 |
|---:|---|---:|---:|---:|---:|---:|
| t7 | hard_speed | 1 | 0.001 | 0.5357 | 0.375681 | 1.000000 |
| t7 | hard_momentum | 8 | 10 | 0.6111 | 0.149793 | 1.000000 |
| t7 | soft_momentum | 8 | 100 | 0.5516 | 0.317634 | 1.000000 |
| t7 | selected_momentum | 8 | 10 | 0.6389 | 0.094395 | 1.000000 |
| t7 | all_momentum | 24 | 10 | 0.5397 | 0.361732 | 1.000000 |
| t12 | hard_speed | 1 | 0.001 | 0.5794 | 0.230838 | 1.000000 |
| t12 | hard_momentum | 8 | 0.1 | 0.5040 | 0.496875 | 1.000000 |
| t12 | soft_momentum | 8 | 1 | 0.5040 | 0.490025 | 1.000000 |
| t12 | selected_momentum | 8 | 1 | 0.4921 | 0.541373 | 1.000000 |
| t12 | all_momentum | 24 | 0.01 | 0.5119 | 0.465527 | 1.000000 |
| t34 | hard_speed | 1 | 0.001 | 0.8016 | 0.001700 | 0.025499 |
| t34 | hard_momentum | 8 | 0.001 | 0.8016 | 0.001350 | 0.020249 |
| t34 | soft_momentum | 8 | 100 | 0.8056 | 0.001600 | 0.023999 |
| t34 | selected_momentum | 8 | 0.001 | 0.8056 | 0.001400 | 0.020999 |
| t34 | all_momentum | 24 | 0.1 | 0.8452 | 0.000100 | 0.001500 |

## All-momentum gain over hard-route speed

| horizon | speed AUC | all-momentum AUC | delta | paired p | bootstrap delta 95% CI |
|---:|---:|---:|---:|---:|---:|
| t7 | 0.5357 | 0.5397 | 0.0040 | 0.493175 | [-0.2937, 0.3095] |
| t12 | 0.5794 | 0.5119 | -0.0675 | 0.765562 | [-0.2341, 0.0952] |
| t34 | 0.8016 | 0.8452 | 0.0437 | 0.101045 | [-0.0159, 0.1270] |
