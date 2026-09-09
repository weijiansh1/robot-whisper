# Cross-task hidden-matched expert-activation proxy

This report combines five retrospective captures. Each capture contains 16
exact-observation pools with 32 common flow-noise seeds. The cell was fixed at
HB layer 5, denoise step 0 before the formal 5,000-permutation runs. These are
offline proxies, not runtime interventions continued through the remaining
flow steps.

## Completion audit

The earlier `expert-activation-v2/long-t08` run is complete: its feature file,
summary, report, post-hoc control, and clean run log are present. It should not
be restarted. The planned exact first-output capture was never launched: the
recorder and tests exist, but no `first_control_outputs.zarr` or partial runtime
log exists. The strict divergence-twin and continued expert-intervention arms
also had no previous implementation or result to resume. The analyses below
are the completed CPU-feasible retrospective replacement for those missing
arms.

## Future-action divergence after hidden matching

Within every K=32 pool, the 50 nearest pairs by the full ten-action-token
pre-MoE hidden tensor were retained. The ten lowest and ten highest final-action
distances within that set formed the stable and divergent classes.

| task | full-hidden AUC | shared AUC | conflict Q AUC | DCQ AUC | DCQ maxT p | exact global-q90 divergent pairs |
|---|---:|---:|---:|---:|---:|---:|
| goal-middle | 0.643 | 0.743 | 0.721 | 0.709 | 0.0976 | 10 |
| goal-top | 0.576 | 0.672 | 0.795 | 0.752 | 0.0472 | 0 |
| long-t08 | 0.601 | 0.715 | 0.574 | 0.553 | 0.6369 | 9 |
| spatial-ramekin | 0.737 | 0.636 | 0.615 | 0.677 | 0.1108 | 18 |
| spatial-stove | 0.556 | 0.546 | 0.693 | 0.607 | 0.3975 | 8 |
| task mean | 0.623 | 0.662 | 0.680 | 0.660 | - | 9 |

Conflict and DCQ are above chance in all five tasks, but the expert-specific
increment is not stable. Across tasks, conflict minus full hidden is +0.057 and
conflict minus shared is only +0.017. A task-and-pool hierarchical bootstrap
gives intervals of [-0.053, 0.162] and [-0.079, 0.111], respectively. DCQ minus
full hidden is +0.037 [-0.040, 0.118], while DCQ minus shared is -0.003
[-0.090, 0.069].

The one nominally passing task (`goal-top`, p=0.0472) is corrected over the
eight D/C/Q discovery scores inside that task, but not over the five tasks. A
conservative five-task Bonferroni adjustment makes it 0.236. This is therefore
suggestive of a structured internal monitor, not evidence of an extraction
advantage over these simple full-hidden/shared distance controls. In principle,
the decomposition cannot contain information absent from the complete hidden
state because it is computed from that state.

The literal current-q10 and global-final-q90 intersection contains only 45
divergent pairs across all 80 pools, including zero in `goal-top`. The balanced
proxy is usable for exploration, but it is not the strict same-token twin test.
Its matched hidden median remains about 89.3% of the all-pair median, illustrating
high-dimensional distance concentration.

## Offline expert-replacement sensitivity

The recorded top four experts were replaced offline by probability ranks 5-8.
The table reports task-mean Spearman correlations with three normalizations of
the immediate routed-vector change.

| predictor | delta / post-MoE | delta / routed | delta / input hidden |
|---|---:|---:|---:|
| disagreement D | +0.081 | +0.463 | -0.280 |
| cancellation C | +0.030 | +0.484 | -0.322 |
| routed/shared conflict Q | +0.457 | +0.330 | -0.076 |
| DCQ composite | +0.206 | +0.485 | -0.258 |
| routed/shared norm ratio | +0.535 | -0.532 | +0.420 |

D, C, and DCQ reproducibly predict replacement size relative to the original
routed vector: all five tasks have maxT-FWER p=0.0002 for each of those three
scores. This is a real but close-to-mechanistic result because predictors and
target are computed from the same expert outputs. For the frozen primary
post-MoE denominator, Q is strong in every task (rho 0.361-0.559), but it does
not beat the simpler routed/shared norm ratio on average (-0.078). With input
hidden as denominator, the D/C/DCQ association reverses. Thus there is no
denominator-invariant scalar that can yet be called action fragility.

Offline reconstruction also has a numerical boundary: checkpoint-router top-4
sets match the captured sets in 86.8%-90.0% of sites (mean 88.5%), although
selected-probability MAE is only 5.98e-5. Runtime output capture is needed for
an exact geometry result.

## Decision

Do not rerun `expert-activation-v2`; its completed outputs answer a different,
route-ID-heavy question. A new live experiment is justified only as a focused
causal confirmation. It should record current flow latent/action token, exact
selected expert outputs, routed/shared/post-MoE vectors, then apply a fixed
rank-5-8 or drop-one intervention at L5/d0 and continue both branches with the
same observation and future randomness. Held-out tasks and perturbation-built
sibling pairs are needed because naturally occurring strict twins are too
sparse.

Current evidence supports a weak Level-1 claim: MoE decomposition exposes how
internally sensitive the current block is. It does not establish Level-2 hidden
action commitment, Level-3 success prediction, or a stopping/pruning rule.
