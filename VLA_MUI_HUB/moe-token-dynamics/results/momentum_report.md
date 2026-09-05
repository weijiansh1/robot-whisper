# HB-MoE route momentum analysis

Velocity is v_t=x_t-x_(t-1); acceleration is a_t=v_t-v_(t-1).
W is the number of terminal velocity samples. State 0 selects a feature and state 42 confirms it.

## All aggregate motion features

Only action_all features are eligible; window and statistic are selected in discovery.

| source | horizon | selected feature | discovery maxT p | confirmation AUC | p | Bonferroni over 9 |
|---|---:|---|---:|---:|---:|---:|
| hard_route | t7 | `hard_route|h07|W2|speed_delta|action_all` | 0.451627 | 0.5476 | 0.338083 | 1.000000 |
| hard_route | t12 | `hard_route|h12|W3|ema08_magnitude|action_all` | 0.620219 | 0.5556 | 0.304535 | 1.000000 |
| hard_route | t34 | `hard_route|h34|W7|speed|action_all` | 0.000050 | 0.8016 | 0.001300 | 0.011699 |
| soft_route | t7 | `soft_route|h07|W2|speed_delta|action_all` | 0.819259 | 0.5040 | 0.491675 | 1.000000 |
| soft_route | t12 | `soft_route|h12|W4|straightness|action_all` | 0.044248 | 0.5714 | 0.262987 | 1.000000 |
| soft_route | t34 | `soft_route|h34|W4|straightness|action_all` | 0.000100 | 0.7262 | 0.016449 | 0.148043 |
| selected_route | t7 | `selected_route|h07|W1|acceleration|action_all` | 0.402980 | 0.6627 | 0.061197 | 0.550772 |
| selected_route | t12 | `selected_route|h12|W3|ema08_magnitude|action_all` | 0.645268 | 0.5675 | 0.269687 | 1.000000 |
| selected_route | t34 | `selected_route|h34|W7|speed|action_all` | 0.000050 | 0.8056 | 0.001800 | 0.016199 |

## Direction-only features

Only alignment and straightness are eligible, excluding speed magnitude.

| source | horizon | selected feature | discovery maxT p | confirmation AUC | p | Bonferroni over 9 |
|---|---:|---|---:|---:|---:|---:|
| hard_route | t7 | `hard_route|h07|W1|alignment|action_all` | 0.290485 | 0.6389 | 0.093995 | 0.845958 |
| hard_route | t12 | `hard_route|h12|W7|straightness|action_all` | 0.653567 | 0.5556 | 0.306035 | 1.000000 |
| hard_route | t34 | `hard_route|h34|W7|straightness|action_all` | 0.000100 | 0.7738 | 0.003950 | 0.035548 |
| soft_route | t7 | `soft_route|h07|W6|straightness|action_all` | 0.923504 | 0.4841 | 0.561872 | 1.000000 |
| soft_route | t12 | `soft_route|h12|W4|straightness|action_all` | 0.017699 | 0.5714 | 0.255637 | 1.000000 |
| soft_route | t34 | `soft_route|h34|W4|straightness|action_all` | 0.000050 | 0.7262 | 0.015899 | 0.143093 |
| selected_route | t7 | `selected_route|h07|W1|alignment|action_all` | 0.292035 | 0.6746 | 0.051097 | 0.459877 |
| selected_route | t12 | `selected_route|h12|W7|straightness|action_all` | 0.629569 | 0.5556 | 0.300385 | 1.000000 |
| selected_route | t34 | `selected_route|h34|W7|straightness|action_all` | 0.000200 | 0.7659 | 0.005700 | 0.051297 |

## Full token scan

Window, statistic, and T0-T10/action_all are selected in discovery.

| source | horizon | selected feature | discovery maxT p | confirmation AUC | p | Bonferroni over 9 |
|---|---:|---|---:|---:|---:|---:|
| hard_route | t7 | `hard_route|h07|W2|ema08_magnitude|T9` | 0.342033 | 0.5159 | 0.450427 | 1.000000 |
| hard_route | t12 | `hard_route|h12|W6|ema08_magnitude|T6` | 0.498725 | 0.3611 | 0.906755 | 1.000000 |
| hard_route | t34 | `hard_route|h34|W7|speed|T7` | 0.000050 | 0.7262 | 0.015949 | 0.143543 |
| soft_route | t7 | `soft_route|h07|W2|speed_delta|T7` | 0.407430 | 0.4603 | 0.653067 | 1.000000 |
| soft_route | t12 | `soft_route|h12|W7|ema08_magnitude|T8` | 0.069297 | 0.5952 | 0.187541 | 1.000000 |
| soft_route | t34 | `soft_route|h34|W7|speed|T0` | 0.000050 | 0.6508 | 0.076096 | 0.684866 |
| selected_route | t7 | `selected_route|h07|W2|ema08_magnitude|T9` | 0.294235 | 0.5317 | 0.394680 | 1.000000 |
| selected_route | t12 | `selected_route|h12|W6|ema08_magnitude|T6` | 0.442028 | 0.3730 | 0.889256 | 1.000000 |
| selected_route | t34 | `selected_route|h34|W7|speed|T7` | 0.000050 | 0.7262 | 0.015449 | 0.139043 |
