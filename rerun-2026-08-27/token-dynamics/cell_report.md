# Layer-flow cell localization

The discovery scan is intentionally treated only as feature selection. Its p-values are not used. Each selected cell is tested with a fixed direction in state 42.

## Horizon leads

| horizon | selected cell | discovery AUC | confirmation oriented AUC | confirmation p | Bonferroni over 3 horizons |
|---:|---|---:|---:|---:|---:|
| t7 | `route_top4|h07|lag4|T3|L14|f8` | 0.8671 | 0.5754 | 0.240738 | 0.722214 |
| t12 | `route_top4|h12|lag1|T10|L4|f3` | 0.9484 | 0.5456 | 0.336133 | 1.000000 |
| t34 | `route_top4|h34|lag3|T10|L12|f2` | 0.0000 | 0.8929 | 0.000050 | 0.000150 |

## Source by horizon

| horizon | source | selected cell | discovery AUC | confirmation oriented AUC | p |
|---:|---|---|---:|---:|---:|
| t7 | route_soft | `route_soft|h07|lag5|T5|L13|f5` | 0.8611 | 0.6429 | 0.093045 |
| t7 | route_top4 | `route_top4|h07|lag4|T3|L14|f8` | 0.8671 | 0.5754 | 0.233138 |
| t7 | hidden | `hidden|h07|lag3|T0|L12|f0` | 0.8571 | 0.6944 | 0.032598 |
| t12 | route_soft | `route_soft|h12|period3|T9|L12|f8` | 0.1230 | 0.5913 | 0.200990 |
| t12 | route_top4 | `route_top4|h12|lag1|T10|L4|f3` | 0.9484 | 0.5456 | 0.336233 |
| t12 | hidden | `hidden|h12|period4|action_std|L3|f5` | 0.1587 | 0.6667 | 0.057597 |
| t34 | route_soft | `route_soft|h34|lag1|T0|L4|f0` | 0.0079 | 0.7262 | 0.014349 |
| t34 | route_top4 | `route_top4|h34|lag3|T10|L12|f2` | 0.0000 | 0.8929 | 0.000050 |
| t34 | hidden | `hidden|h34|lag5|action_std|L12|f9` | 0.0000 | 0.8492 | 0.000150 |

## Category by horizon

| horizon | category | selected cell | discovery AUC | confirmation oriented AUC | p | Bonferroni over 9 category-horizon tests |
|---:|---|---|---:|---:|---:|---:|
| t7 | token_lag | `route_top4|h07|lag4|T3|L14|f8` | 0.8671 | 0.5754 | 0.237438 | 1.000000 |
| t7 | descriptor_lag | `route_top4|h07|lag1|action_std|L15|f2` | 0.8571 | 0.5714 | 0.256837 | 1.000000 |
| t7 | periodicity | `route_top4|h07|period5|T2|L12|f9` | 0.1409 | 0.5556 | 0.296485 | 1.000000 |
| t12 | token_lag | `route_top4|h12|lag1|T10|L4|f3` | 0.9484 | 0.5456 | 0.334833 | 1.000000 |
| t12 | descriptor_lag | `route_top4|h12|lag1|endpoint_delta|L4|f2` | 0.9127 | 0.5774 | 0.234638 | 1.000000 |
| t12 | periodicity | `route_top4|h12|period4|T10|L2|f2` | 0.8929 | 0.6310 | 0.109645 | 0.986801 |
| t34 | token_lag | `route_top4|h34|lag3|T10|L12|f2` | 0.0000 | 0.8929 | 0.000100 | 0.000900 |
| t34 | descriptor_lag | `route_top4|h34|lag5|action_mean|L4|f6` | 0.0000 | 0.8095 | 0.001000 | 0.009000 |
| t34 | periodicity | `hidden|h34|period3|T10|L14|f9` | 0.9881 | 0.7579 | 0.006000 | 0.053997 |

The same 32 noise seed IDs occur in discovery and confirmation states. A replicated cell is cross-state evidence, not unseen-seed evidence.
