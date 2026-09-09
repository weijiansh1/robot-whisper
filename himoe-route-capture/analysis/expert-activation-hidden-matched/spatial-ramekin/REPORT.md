# Hidden-matched expert-activation proxy

This is a retrospective proxy on one captured task, not a runtime causal test.
Source capture: `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate/right-16x32`.
The primary cell was fixed to HB L5/d0.

## Hidden-matched future-action divergence

Each exact-observation K32 pool contributes its 50 nearest pre-MoE-hidden pairs;
the lowest/highest ten final-action distances are the stable/divergent classes.
The literal current-q10/final-global-q10-q90 rule produced 247 stable and 18 divergent pairs,
so it is reported as underpowered rather than used as the primary balanced comparison.

| score | macro AUC | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| hidden_rms | 0.737 | [0.709, 0.766] | control |
| hidden_cosine | 0.736 | [0.708, 0.765] | control |
| noise_rms | 0.625 | [0.603, 0.651] | control |
| router_tv | 0.464 | [0.426, 0.502] | control |
| shared_cosine | 0.636 | [0.612, 0.661] | control |
| routed_cosine | 0.507 | [0.467, 0.546] | control |
| dcq_level | 0.677 | [0.625, 0.733] | 0.1108 |
| dcq_diff | 0.552 | [0.490, 0.615] | 0.6483 |
| disagreement_level | 0.667 | [0.616, 0.719] | 0.1330 |
| cancellation_level | 0.669 | [0.619, 0.718] | 0.1296 |
| conflict_level | 0.615 | [0.556, 0.676] | 0.3147 |

DCQ-level AUC minus controls:

- hidden_rms: -0.060 [-0.124, 0.009]
- hidden_cosine: -0.059 [-0.125, 0.006]
- noise_rms: +0.052 [-0.006, 0.113]
- router_tv: +0.213 [0.134, 0.297]
- shared_cosine: +0.041 [-0.018, 0.102]

## Offline rank-5-8 block sensitivity

F_block is the immediate relative change in routed+shared output after replacing
the recorded top-4 with the four highest-probability unselected experts.
It is not a final-action intervention.

| score | scene-macro Spearman | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| negative_top4_margin | -0.013 | [-0.090, 0.056] | control |
| router_entropy | -0.116 | [-0.157, -0.077] | control |
| hidden_rms | -0.311 | [-0.341, -0.277] | control |
| expert_mass | -0.041 | [-0.094, 0.014] | control |
| routed_over_shared | +0.476 | [0.419, 0.538] | control |
| disagreement | +0.017 | [-0.046, 0.075] | 0.6241 |
| cancellation | +0.009 | [-0.052, 0.066] | 0.6465 |
| conflict | +0.369 | [0.314, 0.423] | 0.0028 |
| dcq | +0.111 | [0.053, 0.163] | 0.3203 |

Primary-target Spearman contrasts:

- conflict_minus_routed_over_shared: -0.108 [-0.191, -0.029]
- conflict_minus_hidden_rms: +0.680 [0.624, 0.734]
- conflict_minus_negative_top4_margin: +0.381 [0.300, 0.471]
- dcq_minus_routed_over_shared: -0.365 [-0.469, -0.277]
- dcq_minus_hidden_rms: +0.422 [0.361, 0.479]
- dcq_minus_negative_top4_margin: +0.124 [0.028, 0.227]

Denominator sensitivity (scene-macro Spearman):

| target | disagreement | cancellation | conflict | DCQ | routed/shared |
|---|---:|---:|---:|---:|---:|
| post_relative | +0.017 | +0.009 | +0.369 | +0.111 | +0.476 |
| routed_relative | +0.451 | +0.432 | +0.375 | +0.480 | -0.522 |
| input_relative | -0.455 | -0.448 | -0.072 | -0.383 | +0.461 |

## Boundary

The expert outputs were reconstructed from fp16 stored hidden values with CPU fp32
checkpoint evaluation. Recorded ids and weights are authoritative, but vectors are
not runtime-exact. Exact x_tau matching, AS outputs, and final-action sensitivity
require a new instrumented rollout with common-future interventions.
