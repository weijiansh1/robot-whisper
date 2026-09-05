# Early action-output control

All values are pair-weighted within-initial-state AUC under init-state + seed double holdout. The target keeps non-stasis failures as negatives.

| feature | t7 | t12 | t20 | t27 | t34 |
|---|---:|---:|---:|---:|---:|
| action_history | 0.438 | 0.556 | 0.514 | 0.727 | 0.838 |
| hidden_identity | 0.476 | 0.624 | 0.569 | 0.757 | 0.885 |
| hidden_history | 0.448 | 0.618 | 0.552 | 0.797 | 0.888 |
| action_plus_hidden_identity | 0.430 | 0.555 | 0.531 | 0.775 | 0.878 |
| action_plus_hidden_history | 0.457 | 0.563 | 0.500 | 0.793 | 0.889 |
| sim_history | 0.523 | 0.552 | 0.536 | 0.681 | 0.847 |

The emitted action chunk is not a strong t12 baseline. Concatenating action and hidden blocks under the fixed linear readout does not improve on hidden alone, so this audit does not establish conditional hidden-state value beyond actions.
