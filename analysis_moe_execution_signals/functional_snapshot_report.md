# HB-MoE functional snapshot audit

## Gate

- GPU snapshots: 2; byte-identical: `true`.
- One snapshot contains 8 HB layers x 10 denoise steps x 11 tokens, and occupies 259.6 KiB before NPZ compression.
- This is a fixed synthetic observation no-op audit, not an event/outcome sample.

## Functional ranges

- Routed authority: median 0.324, q10-q90 0.129-0.410.
- Expert cancellation: median 0.461, q10-q90 0.335-0.489.
- Normalized disagreement: median 0.719, q10-q90 0.688-0.743.
- Routed/shared cosine: median 0.165, q10-q90 0.046-0.300.
- Exact Top-4/5 logit ties: 19.2% of sites.

## Route versus function

Median-split site fractions (descriptive only): stable 49.4%, cosmetic churn 0.6%, within-support functional change 39.0%, route+functional change 11.0%.
Support churn/function delta Spearman rho 0.821; execution-weight churn/function delta rho 0.856.

## Token and sketch checks

- State/action routed authority: 0.435 / 0.289.
- 16-D sketch routed-norm relative error median 12.6%, q90 31.5%; shared median 13.7%, q90 28.9%.

## Interpretation

The new quantities have non-degenerate dynamic range and the capture is numerically transparent on two physical GPUs.  This audit cannot establish Trap prediction or causality.  VLA_MUI_HUB stores no RGB observations, so event-level rich capture must reconstruct simulator snapshots and render fresh observations.
