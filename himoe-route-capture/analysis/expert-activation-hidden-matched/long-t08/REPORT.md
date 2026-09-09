# Hidden-matched expert-activation proxy

This is a retrospective proxy on one captured task, not a runtime causal test.
Source capture: `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32`.
The primary cell was fixed to HB L5/d0.

## Hidden-matched future-action divergence

Each exact-observation K32 pool contributes its 50 nearest pre-MoE-hidden pairs;
the lowest/highest ten final-action distances are the stable/divergent classes.
The literal current-q10/final-global-q10-q90 rule produced 187 stable and 9 divergent pairs,
so it is reported as underpowered rather than used as the primary balanced comparison.

| score | macro AUC | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| hidden_rms | 0.601 | [0.567, 0.638] | control |
| hidden_cosine | 0.601 | [0.568, 0.639] | control |
| noise_rms | 0.538 | [0.503, 0.572] | control |
| router_tv | 0.542 | [0.494, 0.583] | control |
| shared_cosine | 0.715 | [0.680, 0.746] | control |
| routed_cosine | 0.462 | [0.421, 0.502] | control |
| dcq_level | 0.553 | [0.511, 0.593] | 0.6369 |
| dcq_diff | 0.543 | [0.486, 0.601] | 0.6789 |
| disagreement_level | 0.565 | [0.522, 0.606] | 0.5767 |
| cancellation_level | 0.529 | [0.488, 0.574] | 0.7359 |
| conflict_level | 0.574 | [0.544, 0.603] | 0.5343 |

DCQ-level AUC minus controls:

- hidden_rms: -0.048 [-0.112, 0.014]
- hidden_cosine: -0.048 [-0.114, 0.014]
- noise_rms: +0.016 [-0.054, 0.084]
- router_tv: +0.011 [-0.054, 0.089]
- shared_cosine: -0.162 [-0.208, -0.109]

## Offline rank-5-8 block sensitivity

F_block is the immediate relative change in routed+shared output after replacing
the recorded top-4 with the four highest-probability unselected experts.
It is not a final-action intervention.

| score | scene-macro Spearman | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| negative_top4_margin | -0.086 | [-0.172, -0.002] | control |
| router_entropy | -0.060 | [-0.090, -0.030] | control |
| hidden_rms | -0.206 | [-0.238, -0.175] | control |
| expert_mass | +0.090 | [0.052, 0.128] | control |
| routed_over_shared | +0.542 | [0.501, 0.582] | control |
| disagreement | +0.225 | [0.179, 0.266] | 0.1178 |
| cancellation | +0.176 | [0.131, 0.216] | 0.2052 |
| conflict | +0.552 | [0.514, 0.590] | 0.0002 |
| dcq | +0.339 | [0.300, 0.378] | 0.0176 |

Primary-target Spearman contrasts:

- conflict_minus_routed_over_shared: +0.010 [-0.046, 0.074]
- conflict_minus_hidden_rms: +0.758 [0.715, 0.800]
- conflict_minus_negative_top4_margin: +0.637 [0.549, 0.734]
- dcq_minus_routed_over_shared: -0.202 [-0.259, -0.144]
- dcq_minus_hidden_rms: +0.545 [0.497, 0.592]
- dcq_minus_negative_top4_margin: +0.425 [0.324, 0.529]

Denominator sensitivity (scene-macro Spearman):

| target | disagreement | cancellation | conflict | DCQ | routed/shared |
|---|---:|---:|---:|---:|---:|
| post_relative | +0.225 | +0.176 | +0.552 | +0.339 | +0.542 |
| routed_relative | +0.484 | +0.507 | +0.402 | +0.518 | -0.490 |
| input_relative | -0.455 | -0.512 | -0.232 | -0.451 | +0.576 |

## Boundary

The expert outputs were reconstructed from fp16 stored hidden values with CPU fp32
checkpoint evaluation. Recorded ids and weights are authoritative, but vectors are
not runtime-exact. Exact x_tau matching, AS outputs, and final-action sensitivity
require a new instrumented rollout with common-future interventions.
