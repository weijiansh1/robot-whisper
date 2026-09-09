# Hidden-matched expert-activation proxy

This is a retrospective proxy on one captured task, not a runtime causal test.
Source capture: `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate/right-16x32`.
The primary cell was fixed to HB L5/d0.

## Hidden-matched future-action divergence

Each exact-observation K32 pool contributes its 50 nearest pre-MoE-hidden pairs;
the lowest/highest ten final-action distances are the stable/divergent classes.
The literal current-q10/final-global-q10-q90 rule produced 221 stable and 8 divergent pairs,
so it is reported as underpowered rather than used as the primary balanced comparison.

| score | macro AUC | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| hidden_rms | 0.556 | [0.508, 0.604] | control |
| hidden_cosine | 0.556 | [0.507, 0.603] | control |
| noise_rms | 0.499 | [0.463, 0.538] | control |
| router_tv | 0.401 | [0.364, 0.436] | control |
| shared_cosine | 0.546 | [0.504, 0.589] | control |
| routed_cosine | 0.478 | [0.429, 0.527] | control |
| dcq_level | 0.607 | [0.556, 0.656] | 0.3975 |
| dcq_diff | 0.611 | [0.569, 0.652] | 0.3803 |
| disagreement_level | 0.563 | [0.505, 0.622] | 0.5909 |
| cancellation_level | 0.564 | [0.506, 0.621] | 0.5885 |
| conflict_level | 0.693 | [0.646, 0.743] | 0.1174 |

DCQ-level AUC minus controls:

- hidden_rms: +0.051 [0.001, 0.099]
- hidden_cosine: +0.051 [0.001, 0.099]
- noise_rms: +0.108 [0.058, 0.154]
- router_tv: +0.206 [0.131, 0.280]
- shared_cosine: +0.061 [0.014, 0.110]

## Offline rank-5-8 block sensitivity

F_block is the immediate relative change in routed+shared output after replacing
the recorded top-4 with the four highest-probability unselected experts.
It is not a final-action intervention.

| score | scene-macro Spearman | pool-bootstrap 95% CI | maxT FWER p |
|---|---:|---:|---:|
| negative_top4_margin | -0.051 | [-0.122, 0.019] | control |
| router_entropy | -0.084 | [-0.120, -0.047] | control |
| hidden_rms | -0.307 | [-0.342, -0.270] | control |
| expert_mass | -0.022 | [-0.062, 0.018] | control |
| routed_over_shared | +0.406 | [0.353, 0.455] | control |
| disagreement | +0.052 | [0.003, 0.100] | 0.4873 |
| cancellation | +0.040 | [-0.008, 0.089] | 0.5271 |
| conflict | +0.445 | [0.408, 0.480] | 0.0004 |
| dcq | +0.168 | [0.120, 0.215] | 0.1626 |

Primary-target Spearman contrasts:

- conflict_minus_routed_over_shared: +0.039 [-0.033, 0.108]
- conflict_minus_hidden_rms: +0.752 [0.699, 0.803]
- conflict_minus_negative_top4_margin: +0.496 [0.428, 0.560]
- dcq_minus_routed_over_shared: -0.239 [-0.316, -0.158]
- dcq_minus_hidden_rms: +0.475 [0.429, 0.521]
- dcq_minus_negative_top4_margin: +0.218 [0.126, 0.310]

Denominator sensitivity (scene-macro Spearman):

| target | disagreement | cancellation | conflict | DCQ | routed/shared |
|---|---:|---:|---:|---:|---:|
| post_relative | +0.052 | +0.040 | +0.445 | +0.168 | +0.406 |
| routed_relative | +0.548 | +0.548 | +0.432 | +0.577 | -0.623 |
| input_relative | -0.409 | -0.395 | -0.087 | -0.346 | +0.362 |

## Boundary

The expert outputs were reconstructed from fp16 stored hidden values with CPU fp32
checkpoint evaluation. Recorded ids and weights are authoritative, but vectors are
not runtime-exact. Exact x_tau matching, AS outputs, and final-action sensitivity
require a new instrumented rollout with common-future interventions.
