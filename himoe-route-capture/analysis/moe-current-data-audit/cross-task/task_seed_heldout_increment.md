# Task + seed-held-out incremental audit

This is an exploratory shadow probe. Every outer prediction excludes the complete test task and the test K8 seed identities.

| features | K8 AUC | selected success | random mean | delta | state-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| noise | 0.476 | 0.878 | 0.880 | -0.002 | [-0.022, +0.019] |
| action | 0.561 | 0.887 | 0.880 | +0.007 | [-0.017, +0.032] |
| hidden | 0.479 | 0.878 | 0.880 | -0.002 | [-0.026, +0.022] |
| route | 0.478 | 0.872 | 0.880 | -0.008 | [-0.029, +0.012] |
| action + noise | 0.533 | 0.884 | 0.880 | +0.004 | [-0.021, +0.029] |
| action + noise + route | 0.510 | 0.881 | 0.880 | +0.001 | [-0.021, +0.023] |
| action + noise + hidden | 0.449 | 0.878 | 0.880 | -0.002 | [-0.027, +0.022] |
| action + noise + hidden + route | 0.440 | 0.872 | 0.880 | -0.008 | [-0.035, +0.017] |

## Conditional route increments

- `route_over_action_noise`: selected-success -0.003 (95% CI [-0.028, +0.019]); pooled evaluable-pool AUC increment -0.023.
- `route_over_action_noise_hidden`: selected-success -0.006 (95% CI [-0.031, +0.019]); pooled evaluable-pool AUC increment -0.009.

## Interpretation boundary

The label is eventual episode success from one rollout per seed. The seed also controls later replanning noise, so conditional predictive increment cannot be read as route information about the causal value of the first action chunk.
