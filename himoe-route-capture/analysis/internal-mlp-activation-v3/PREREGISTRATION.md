# HB5/d0 Pre-down Internal Activation: Frozen Protocol

This protocol was written before reconstructing or inspecting any `m_e`
features from the 640-row runtime-vector-v3 grid.

## Quantities

- Router weight: selection and mixing probability.
- Pre-down internal activation: `m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`.
- Post-down expert output: `E_e(h) = down_proj_e(m_e)`.

The primary features are computed from `m_e`, not from the already stored
`E_e(h)`. Raw RMS, raw mean absolute value, and raw signed mean remain in
checkpoint units; they are not divided by hidden, shared, routed, or expert
norms. Train-fold feature standardization is a readout fitting operation only.

## Frozen Feature Block

At each of 10 action tokens and four selected experts, compute exactly:

1. raw RMS of `m_e`;
2. raw L1 mean of `m_e`;
3. raw signed mean of `m_e`;
4. active fraction `|m_e| > 1e-3`;
5. closed-gate fraction `gate_proj_e(h) <= -4`;
6. open-gate fraction `gate_proj_e(h) >= 4`.

For each token and scalar, compute the normalized-router-weighted mean and
weighted scalar standard deviation across the selected experts. This yields
`10 * 6 * 2 = 120` exact features. The expert activation values themselves are
not norm-normalized.

Pre-down neuron coordinates are independently learned for each expert. No
cross-expert vector mean, angle, channelwise dispersion, or cancellation is
defined or tested. Aggregation occurs only after reducing within each expert to
the six scalars above, making it invariant to top-k slot order and global expert
ID permutation.

## Matched Control and Baseline

The strong baseline is B1 from `analyze_runtime_vector_v3.py`: x0, x1, HB5
input hidden, shared output, full router probabilities, selected IDs, weights,
and actual merged routed output. It is reused verbatim.

The matched control adds exactly 120 scalar features produced by applying six
fixed post-down vector summaries to the selected `E_e(h)` vectors, followed by
the same weighted mean and scalar dispersion aggregation. This asks whether a
gain is specific to pre-down internals rather than extra exact expert-summary
dimensions or readout capacity.

## Target and Split

Primary target: natural future path energy after d0,
`RMS_{tau=1..9,token,live7}(x[tau+1] - x[tau])`.

Each task is fit separately. Nested Cartesian CV holds out both observation
state and repeated-noise candidate identity. The five fixed CountSketch seeds,
ridge grid, and B1 construction are inherited unchanged from runtime-v3.

## Claim Gate

Every one of the five frozen B1 projection seeds must satisfy all of:

- lower held-out RMSE in all five tasks;
- at least 2% equal-task macro relative RMSE improvement over B1;
- paired state/candidate bootstrap 95% CI lower bound above zero;
- lower held-out RMSE than the exact120 post-down matched control in all five
  tasks.

Failure of any condition means no five-task incremental pre-down signal under
this protocol. This is HB5/d0 only and cannot establish increasing confidence
over denoising time, action success, or causality.
