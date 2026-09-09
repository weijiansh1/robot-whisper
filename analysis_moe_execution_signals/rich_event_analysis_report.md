# Rich HB-MoE matched event pilot

## Validity gates

- Functional capture: `true`; recorder-transparent rows: 96/96.
- GPU power limits: all selected cards were at their hardware maximum; OOM kills during capture: 13 -> 13.
- Restoration-qualified rows: 94/96; complete pairs: -4=23, -2=23.
- Historical action reproduction is diagnostic only: exact 0/96, maximum absolute difference 1.9697.

## Preregistered results

| Lead | Feature | N | Event-control | Paired d | AUC | raw p | maxT p |
|---:|---|---:|---:|---:|---:|---:|---:|
| -4 | `routed_authority` | 23 | -0.0121 | -1.1920 | 0.155 | 1e-05 | 0.00021 |
| -4 | `expert_cancellation` | 23 | -0.0158 | -1.4239 | 0.091 | 1e-05 | 1e-05 |
| -4 | `expert_disagreement_ratio` | 23 | -0.0146 | -1.6561 | 0.083 | 1e-05 | 1e-05 |
| -4 | `routed_shared_cosine` | 23 | 0.0475 | 1.9061 | 0.922 | 1e-05 | 1e-05 |
| -4 | `functional_flow_delta` | 23 | -0.0058 | -0.3660 | 0.340 | 0.09164 | 0.3806 |
| -2 | `routed_authority` | 23 | -0.0130 | -0.8807 | 0.189 | 0.00028 | 0.00288 |
| -2 | `expert_cancellation` | 23 | -0.0134 | -1.1733 | 0.117 | 7e-05 | 0.00021 |
| -2 | `expert_disagreement_ratio` | 23 | -0.0112 | -1.3698 | 0.098 | 1e-05 | 3e-05 |
| -2 | `routed_shared_cosine` | 23 | 0.0605 | 1.9407 | 0.951 | 1e-05 | 1e-05 |
| -2 | `functional_flow_delta` | 23 | -0.0033 | -0.2128 | 0.405 | 0.3214 | 0.8424 |

## Conclusion

- 8/10 primary cells pass family-wise maxT at 0.05: routed_authority@-4, expert_cancellation@-4, expert_disagreement_ratio@-4, routed_shared_cosine@-4, routed_authority@-2, expert_cancellation@-2, expert_disagreement_ratio@-2, routed_shared_cosine@-2.
- This is a frozen-state observational result, not evidence that rerouting improves control or that a token prefix is safe.
- Layer/token maps and route-function correlations are saved as descriptive secondary outputs only.
