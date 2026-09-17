# Validation of the grid

## Region settings: task AUROC at each radius scale (top 10 selected at scale 1.0)

| setting | s=0.5 | s=0.75 | s=1.0 | s=1.25 | s=1.5 | s=2.0 |
|---|---:|---:|---:|---:|---:|---:|
| total|layer3 3 8 0.9 2 16 returned | 0.500 | 0.494 | 0.747 | 0.500 | 0.500 | 0.500 |
| total|layer3 3 8 0.95 2 16 returned | 0.500 | 0.494 | 0.735 | 0.500 | 0.500 | 0.500 |
| route|layer6 3 12 0.99  16 eligible | 0.500 | 0.487 | 0.272 | 0.500 | 0.500 | 0.500 |
| total|layer3 3 8 0.99 2 16 returned | 0.500 | 0.494 | 0.723 | 0.500 | 0.500 | 0.500 |
| route|layer1 2 8 0.95 2 16 outside_fraction | 0.500 | 0.417 | 0.277 | 0.476 | 0.500 | 0.500 |
| route|layer1 2 8 0.95 3 16 outside_fraction | 0.500 | 0.417 | 0.277 | 0.476 | 0.500 | 0.500 |
| route|layer1 2 8 0.95 4 16 outside_fraction | 0.500 | 0.417 | 0.277 | 0.476 | 0.500 | 0.500 |
| route|layer1 2 8 0.99 2 16 outside_fraction | 0.500 | 0.421 | 0.278 | 0.476 | 0.500 | 0.500 |
| route|layer1 2 8 0.99 3 16 outside_fraction | 0.500 | 0.421 | 0.278 | 0.476 | 0.500 | 0.500 |
| route|layer1 2 8 0.99 4 16 outside_fraction | 0.500 | 0.421 | 0.278 | 0.476 | 0.500 | 0.500 |

Region settings with |task AUROC| >= 0.65 in a consistent direction at >= 4 of 6 scales: 0 / 25740.

## Permutation null (30 within-task label shuffles): max oriented task AUROC per feature family

| family | observed max | null mean of max | null 95% | null max | P(null >= observed) |
|---|---:|---:|---:|---:|---:|
| simple | 0.865 | 0.777 | 0.835 | 0.847 | 0.00 |
| persistence | 0.848 | 0.766 | 0.809 | 0.818 | 0.00 |
| region | 0.750 | 0.760 | 0.807 | 0.838 | 0.60 |
| v82 | 0.728 | 0.662 | 0.720 | 0.751 | 0.03 |

## select_libero10_eval_plus (top 20 per family by task AUROC)

| family | selected mean | transferred mean (task) | transferred min | transferred pooled |
|---|---:|---:|---:|---:|
| simple | 0.817 | 0.795 | 0.685 | 0.814 |
| persistence | 0.815 | 0.778 | 0.663 | 0.745 |
| region | 0.746 | 0.656 | 0.547 | 0.637 |
| v82 | 0.599 | 0.615 | 0.383 | 0.633 |

- simple: shared|back_last      24 net_path -> 0.141/0.194; route|layer5      16 net_path -> 0.164/0.248; shared|layer2      20 lag4 -> 0.832/0.742
- persistence: shared|full_path 3     16 h1_lifetime -> 0.842/0.829; input|back_last 3     24 h0_largest -> 0.839/0.792; shared|full_path 3     16 h1_normalized -> 0.833/0.792
- region: total|state_back 3 12 0.95 1.0 2 16 outside_fraction -> 0.755/0.588; total|state_back 3 12 0.95 1.0 4 16 outside_fraction -> 0.755/0.588; total|state_back 3 12 0.95 1.0 3 16 outside_fraction -> 0.755/0.588
- v82: v82      16 acceleration_score -> 0.206/0.346; v82      24 acceleration_score -> 0.252/0.331; v82      12 acceleration_score -> 0.317/0.485

## select_plus_eval_libero10 (top 20 per family by task AUROC)

| family | selected mean | transferred mean (task) | transferred min | transferred pooled |
|---|---:|---:|---:|---:|
| simple | 0.921 | 0.742 | 0.606 | 0.749 |
| persistence | 0.898 | 0.671 | 0.456 | 0.743 |
| region | 0.861 | 0.565 | 0.504 | 0.641 |
| v82 | 0.630 | 0.597 | 0.481 | 0.609 |

- simple: shared|layer0      20 lag1 -> 0.948/0.659; input|layer7      16 net_path -> 0.056/0.241; input|back_path      16 net_path -> 0.058/0.211
- persistence: shared|layer4 5     16 h1_normalized -> 0.923/0.615; input|back_path 5     16 h1_normalized -> 0.910/0.727; shared|layer4 5     16 h1_lifetime -> 0.910/0.619
- region: route|layer1 2 8 0.99 1.0 2 16 outside_fraction -> 0.107/0.406; route|layer1 2 8 0.99 1.0 4 16 outside_fraction -> 0.107/0.406; route|layer1 2 8 0.99 1.0 3 16 outside_fraction -> 0.107/0.406
- v82: v82      16 freeze_score -> 0.265/0.405; v82      20 freeze_score -> 0.277/0.407; v82      12 freeze_score -> 0.290/0.386

## Leave-one-base-task-out: best feature chosen on 9 tasks, AUROC on the held-out task

| family | held-out mean | held-out min | folds |
|---|---:|---:|---:|
| simple | 0.680 | 0.190 | 10 |
| persistence | 0.652 | 0.406 | 10 |
| region | 0.620 | 0.333 | 10 |
| v82 | 0.675 | 0.470 | 10 |

- simple: task00: input|layer6   16 net_path -> 0.64; task01: shared|back_last   24 net_path -> 0.43; task02: input|back_last   20 diameter -> 0.59; task03: input|layer6   16 net_path -> 0.70; task04: input|back_path   16 net_path -> 0.91; task05: input|layer1   20 lag2 -> 0.19; task06: shared|layer5   20 diameter -> 0.77; task07: input|back_last   20 diameter -> 0.99; task08: shared|layer5   20 diameter -> 0.69; task09: shared|layer5   20 diameter -> 0.89
- persistence: task00: input|back_last 3  24 h0_largest -> 0.56; task01: shared|layer3 3  24 h1_lifetime -> 1.00; task02: shared|layer3 3  24 h1_lifetime -> 0.70; task03: shared|layer3 3  24 h1_lifetime -> -; task04: shared|layer3 3  24 h1_lifetime -> 0.67; task05: input_total|layer4 2  20 h0_largest -> 0.48; task06: total|layer6 2  24 h0_largest -> 0.54; task07: shared|full_path 3  16 h1_lifetime -> 0.75; task08: shared|full_path 3  16 h1_lifetime -> 0.41; task09: shared|layer3 3  24 h1_lifetime -> 0.77
- region: task00: input|state_back 5 8 12 outside_fraction -> 0.63; task01: total|state_back 3 12 16 outside_fraction -> 1.00; task02: total|state_back 3 12 16 outside_fraction -> 0.70; task03: input|state_back 5 8 12 outside_fraction -> 0.60; task04: route|layer6 3 12 16 eligible -> 0.58; task05: total|state_back 2 12 16 outside_fraction -> 0.33; task06: total|state_back 3 12 16 outside_fraction -> 0.60; task07: total|state_back 3 12 16 outside_fraction -> 0.83; task08: total|layer3 3 8 16 returned -> 0.34; task09: total|layer3 3 8 16 returned -> 0.58
- v82: task00: v82   16 acceleration_score -> 0.70; task01: v82   16 acceleration_score -> 1.00; task02: v82   16 acceleration_score -> 0.52; task03: v82   24 acceleration_score -> -; task04: v82   24 acceleration_score -> 0.83; task05: v82   16 acceleration_score -> 0.52; task06: v82   16 acceleration_score -> 0.62; task07: v82   24 acceleration_score -> 0.67; task08: v82   16 acceleration_score -> 0.75; task09: v82   24 acceleration_score -> 0.47