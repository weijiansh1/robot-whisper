# Hidden-matched expert-activation proxy

This is a retrospective proxy on one captured task, not a runtime causal test.
Source capture: `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_goal/open_the_top_drawer_and_put_the_bowl_inside/right-16x32`.
The primary cell was fixed to HB L5/d0.

## Hidden-matched future-action divergence

Each exact-observation K32 pool contributes its 50 nearest pre-MoE-hidden pairs;
the lowest/highest ten final-action distances are the stable/divergent classes.
The literal current-q10/final-global-q10-q90 rule produced 229 stable and 0 divergent pairs,
so it is reported as underpowered rather than used as the primary balanced comparison.

| score | macro AUC | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| hidden_rms | 0.576 | [0.536, 0.614] | control |
| hidden_cosine | 0.574 | [0.536, 0.614] | control |
| noise_rms | 0.585 | [0.544, 0.624] | control |
| router_tv | 0.505 | [0.464, 0.549] | control |
| shared_cosine | 0.672 | [0.629, 0.714] | control |
| routed_cosine | 0.704 | [0.661, 0.744] | control |
| dcq_level | 0.752 | [0.717, 0.786] | 0.0472 |
| dcq_diff | 0.416 | [0.382, 0.456] | 0.9996 |
| disagreement_level | 0.684 | [0.639, 0.728] | 0.1822 |
| cancellation_level | 0.661 | [0.612, 0.706] | 0.2661 |
| conflict_level | 0.795 | [0.763, 0.828] | 0.0180 |

DCQ-level AUC minus controls:

- hidden_rms: +0.176 [0.128, 0.223]
- hidden_cosine: +0.178 [0.129, 0.226]
- noise_rms: +0.167 [0.112, 0.220]
- router_tv: +0.247 [0.198, 0.297]
- shared_cosine: +0.080 [0.027, 0.134]

## Offline rank-5-8 block sensitivity

F_block is the immediate relative change in routed+shared output after replacing
the recorded top-4 with the four highest-probability unselected experts.
It is not a final-action intervention.

| score | scene-macro Spearman | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| negative_top4_margin | -0.047 | [-0.107, 0.010] | control |
| router_entropy | -0.207 | [-0.263, -0.151] | control |
| hidden_rms | -0.239 | [-0.289, -0.186] | control |
| expert_mass | +0.271 | [0.192, 0.345] | control |
| routed_over_shared | +0.615 | [0.571, 0.662] | control |
| disagreement | +0.109 | [0.057, 0.163] | 0.3981 |
| cancellation | +0.009 | [-0.051, 0.069] | 0.6841 |
| conflict | +0.559 | [0.494, 0.622] | 0.0002 |
| dcq | +0.280 | [0.236, 0.330] | 0.0558 |

Primary-target Spearman contrasts:

- conflict_minus_routed_over_shared: -0.056 [-0.148, 0.039]
- conflict_minus_hidden_rms: +0.798 [0.733, 0.864]
- conflict_minus_negative_top4_margin: +0.606 [0.516, 0.691]
- dcq_minus_routed_over_shared: -0.335 [-0.412, -0.250]
- dcq_minus_hidden_rms: +0.519 [0.473, 0.563]
- dcq_minus_negative_top4_margin: +0.327 [0.253, 0.405]

Denominator sensitivity (scene-macro Spearman):

| target | disagreement | cancellation | conflict | DCQ | routed/shared |
|---|---:|---:|---:|---:|---:|
| post_relative | +0.109 | +0.009 | +0.559 | +0.280 | +0.615 |
| routed_relative | +0.435 | +0.479 | +0.280 | +0.445 | -0.520 |
| input_relative | -0.053 | -0.151 | +0.093 | -0.030 | +0.332 |

## Boundary

The expert outputs were reconstructed from fp16 stored hidden values with CPU fp32
checkpoint evaluation. Recorded ids and weights are authoritative, but vectors are
not runtime-exact. Exact x_tau matching, AS outputs, and final-action sensitivity
require a new instrumented rollout with common-future interventions.
