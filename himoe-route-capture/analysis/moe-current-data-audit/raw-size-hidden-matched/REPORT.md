# Raw-size hidden-matched audit

This report formalizes a retrospective proxy. It does not establish that MoE
output size is model confidence, and it does not support candidate pruning.

## Frozen protocol

- Five existing complete captures: 16 exact-observation pools per task and 32 common seed columns per pool.
- Cell fixed to HB5/denoise-0.
- Each pool has 496 candidate pairs. Pair distance uses the full 10 x 1024 pre-MoE action-token hidden tensor.
- Keep the 50 nearest-hidden pairs; within those, the lowest/highest ten final-action distances are stable/divergent.
- Test pair level and absolute pair difference for `raw_mean(d0)` and gate-weighted `expert_mass(d0)`.
- AUC > 0.5 means a larger score predicts the divergent class.
- The four-score family uses 5000 common seed-column permutations and a two-sided maxT correction.

The expert quantities are absolute checkpoint units. No hidden, shared, routed,
post-MoE, d0, or cross-task normalization is applied.

## Source and alignment audit

| task | state axis | seed axis | hidden shape | action error | mean-shift distance error |
|---|---|---|---|---:|---:|
| goal-middle | match | match | 16x32x10x1024 | 1.2e-07 | 2.1e-17 |
| goal-top | match | match | 16x32x10x1024 | 1.2e-07 | 1.4e-17 |
| long-t08 | match | match | 16x32x10x1024 | 1.2e-07 | 2.8e-17 |
| spatial-ramekin | match | match | 16x32x10x1024 | 1.2e-07 | 4.2e-17 |
| spatial-stove | match | match | 16x32x10x1024 | 6e-08 | 4.2e-17 |

Archive formula checks: `raw_mean` max error `2.76e-08`, `expert_mass` max error `2.69e-08`, selected-weight sum max error `4.47e-08`.
Subtracting the action normalization mean cannot change a within-pool pair
difference; the numerical audit above confirms this directly.

## Four-score result

| score | goal-middle | goal-top | long-t08 | spatial-ramekin | spatial-stove | macro | five-task same direction | raw p | maxT4 p |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| raw_level | 0.5106 | 0.3994 | 0.5962 | 0.3331 | 0.5594 | 0.4797 | no | 0.7588 | 0.9288 |
| mass_level | 0.5138 | 0.4012 | 0.6006 | 0.3262 | 0.5537 | 0.4791 | no | 0.7518 | 0.9256 |
| raw_diff | 0.6306 | 0.5663 | 0.4994 | 0.3869 | 0.5194 | 0.5205 | no | 0.6609 | 0.9280 |
| mass_diff | 0.6350 | 0.5619 | 0.4969 | 0.3844 | 0.5188 | 0.5194 | no | 0.6855 | 0.9356 |

`level` is the mean size of the two candidates; `diff` is their absolute
size difference. None of the four scores has the same direction in all five
tasks, and none survives the four-score two-sided maxT test.

## Matching diagnostics

| task | near/all hidden median | stable hidden | divergent hidden | hidden AUC | stable action | divergent action |
|---|---:|---:|---:|---:|---:|---:|
| goal-middle | 0.893 | 0.60304 | 0.60870 | 0.643 | 0.02250 | 0.03284 |
| goal-top | 0.892 | 0.60551 | 0.60711 | 0.576 | 0.01953 | 0.02618 |
| long-t08 | 0.894 | 0.61860 | 0.62308 | 0.601 | 0.03268 | 0.05290 |
| spatial-ramekin | 0.892 | 0.60420 | 0.61242 | 0.737 | 0.04527 | 0.06608 |
| spatial-stove | 0.892 | 0.60034 | 0.60244 | 0.556 | 0.04601 | 0.06941 |

Nearest-50 matching reduces hidden separation but does not create exact twins.
Stable/divergent labels are also defined retrospectively from the final action,
so these AUCs are association diagnostics rather than causal effects.

## Verdict

Pair-level raw size is slightly below chance in the equal-task macro, while
absolute size difference is slightly above chance. The task signs are mixed and
the corrected p-values are null. Therefore HB5/d0 absolute selected-expert size
does not provide a reproducible early action-basin or confidence signal here.
It must not be used for pruning, early stopping, or action commitment.

Boundaries:

- The selected-expert norms are offline reconstructions from stored fp16 hidden states; ids and gate weights are captured.
- Matching is nearest-neighbor matching among 496 pairs, not exact hidden equality.
- The same data informed the retrospective question and this audit; there is no prospective holdout.
- Absolute units are checkpoint-specific; cross-task magnitudes are not directly comparable.
- Pairwise final-action distance is invariant to subtracting a common action mean.

Auditable pair selections and null draws are in `arrays.npz`; full precision results are in `summary.json`.
