# MoE routing opportunity shadow audit

## Scope

This is a CPU-only retrospective analysis of the five available `16 x 32` grids. The target is empirical full-rollout success headroom, not candidate-chunk Q. A flow-noise seed also controls later replanning calls, so no causal candidate-selection claim is permitted.

## Data

- Tasks: `5`
- Initial-state pools: `80`
- Full rollouts: `2560`
- Mixed-outcome pools: `40`
- State-token maximum candidate variation (Hellinger): `0.00000003`

## Route dispersion versus empirical headroom

| MoE observable | task-macro Spearman | 95% hierarchical bootstrap CI | stratified p |
|---|---:|---:|---:|
| `action_route_dispersion_d0` | +0.052 | [-0.282, +0.371] | 0.6856 |
| `action_route_dispersion_d0_d2` | +0.089 | [-0.243, +0.391] | 0.4866 |
| `action_route_dispersion_full` | +0.025 | [-0.302, +0.347] | 0.8535 |
| `action_route_entropy_d0` | -0.320 | [-0.599, +0.024] | 0.0126 |
| `action_route_entropy_full` | -0.198 | [-0.504, +0.152] | 0.1246 |

Primary full-route result: task-macro Spearman `+0.025`, 95% CI `[-0.302, +0.347]`, permutation `p=0.8535`.

## Task-held-out diagnostic probes

| Features | OOF R2 | OOF Spearman |
|---|---:|---:|
| action-token route | -0.173 | -0.585 |
| state-token route | -1.207 | -0.151 |
| action + state route | -1.331 | -0.152 |

These probes use fixed ridge regularization and hold out a complete task. They are diagnostics, not tuned predictive models.

## Decision

The analysis cannot establish a MoE opportunity predictor from current data. A positive retrospective association would still require repeated CRN continuation Q before it could authorize adaptive K.
