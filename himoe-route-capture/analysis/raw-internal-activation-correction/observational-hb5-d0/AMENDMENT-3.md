# Amendment 3: equal-family kernel implementation correction

Time: `2026-08-23T13:35:28Z` (UTC).

The first outcome run had been inspected before this amendment. It is retained
verbatim as `summary-overweighted-kernel.json` and
`REPORT-overweighted-kernel.md`. That run produced an equal-task macro relative
RMSE change of `-4.481%` versus B1 (95% paired state/candidate bootstrap CI
`[-5.935%, -3.236%]`), with improvement in `0/5` tasks and improvement over the
matched post-down control in `1/5` tasks.

The implementation added the new feature kernel directly to a B1 kernel that
was already the mean of eight equal-family kernels. This accidentally gave the
new feature eight times the weight of each individual B1 family. Both pre-down
and matched post-down exact40 arms degraded by nearly the same amount, exposing
the generic kernel-weighting failure.

The only correction is the pre-specified equal-family additive formula:

`K_augmented = (8 * K_B1 + K_feature) / 9`.

Target, exact40 feature, matched control, state/candidate splits, alpha grid,
bootstrap, and all interpretation rules remain unchanged. This is an
implementation-bug correction, not outcome-driven feature selection.
