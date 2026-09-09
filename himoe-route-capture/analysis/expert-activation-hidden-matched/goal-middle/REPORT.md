# Hidden-matched expert-activation proxy

This is a retrospective proxy on one captured task, not a runtime causal test.
Source capture: `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_goal/open_the_middle_drawer_of_the_cabinet/right-16x32`.
The primary cell was fixed to HB L5/d0.

## Hidden-matched future-action divergence

Each exact-observation K32 pool contributes its 50 nearest pre-MoE-hidden pairs;
the lowest/highest ten final-action distances are the stable/divergent classes.
The literal current-q10/final-global-q10-q90 rule produced 271 stable and 10 divergent pairs,
so it is reported as underpowered rather than used as the primary balanced comparison.

| score | macro AUC | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| hidden_rms | 0.643 | [0.617, 0.671] | control |
| hidden_cosine | 0.642 | [0.616, 0.671] | control |
| noise_rms | 0.554 | [0.525, 0.584] | control |
| router_tv | 0.587 | [0.568, 0.606] | control |
| shared_cosine | 0.743 | [0.708, 0.774] | control |
| routed_cosine | 0.584 | [0.542, 0.623] | control |
| dcq_level | 0.709 | [0.648, 0.769] | 0.0976 |
| dcq_diff | 0.486 | [0.431, 0.539] | 0.9738 |
| disagreement_level | 0.651 | [0.575, 0.725] | 0.2735 |
| cancellation_level | 0.557 | [0.479, 0.634] | 0.7502 |
| conflict_level | 0.721 | [0.681, 0.759] | 0.0736 |

DCQ-level AUC minus controls:

- hidden_rms: +0.066 [0.010, 0.123]
- hidden_cosine: +0.067 [0.010, 0.122]
- noise_rms: +0.155 [0.093, 0.220]
- router_tv: +0.122 [0.065, 0.177]
- shared_cosine: -0.034 [-0.086, 0.024]

## Offline rank-5-8 block sensitivity

F_block is the immediate relative change in routed+shared output after replacing
the recorded top-4 with the four highest-probability unselected experts.
It is not a final-action intervention.

| score | scene-macro Spearman | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| negative_top4_margin | -0.133 | [-0.192, -0.073] | control |
| router_entropy | -0.296 | [-0.345, -0.251] | control |
| hidden_rms | -0.218 | [-0.264, -0.167] | control |
| expert_mass | +0.309 | [0.247, 0.371] | control |
| routed_over_shared | +0.637 | [0.600, 0.671] | control |
| disagreement | +0.002 | [-0.083, 0.087] | 0.7355 |
| cancellation | -0.084 | [-0.160, -0.008] | 0.9138 |
| conflict | +0.361 | [0.296, 0.425] | 0.0054 |
| dcq | +0.131 | [0.048, 0.212] | 0.3131 |

Primary-target Spearman contrasts:

- conflict_minus_routed_over_shared: -0.276 [-0.354, -0.202]
- conflict_minus_hidden_rms: +0.579 [0.477, 0.675]
- conflict_minus_negative_top4_margin: +0.494 [0.424, 0.574]
- dcq_minus_routed_over_shared: -0.506 [-0.609, -0.412]
- dcq_minus_hidden_rms: +0.349 [0.235, 0.461]
- dcq_minus_negative_top4_margin: +0.264 [0.168, 0.363]

Denominator sensitivity (scene-macro Spearman):

| target | disagreement | cancellation | conflict | DCQ | routed/shared |
|---|---:|---:|---:|---:|---:|
| post_relative | +0.002 | -0.084 | +0.361 | +0.131 | +0.637 |
| routed_relative | +0.399 | +0.455 | +0.160 | +0.403 | -0.505 |
| input_relative | -0.026 | -0.106 | -0.083 | -0.081 | +0.372 |

## Boundary

The expert outputs were reconstructed from fp16 stored hidden values with CPU fp32
checkpoint evaluation. Recorded ids and weights are authoritative, but vectors are
not runtime-exact. Exact x_tau matching, AS outputs, and final-action sensitivity
require a new instrumented rollout with common-future interventions.
