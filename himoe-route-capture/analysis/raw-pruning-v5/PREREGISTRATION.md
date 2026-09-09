# Runtime-exact raw-internal pruning v5

This schema and all formulas are frozen before inspecting the first v5 smoke
outcome.  It implements the causal-compute portion of
`analysis/raw-internal-activation-correction/PREREGISTRATION.md`; it does not
replace or reinterpret the independent v4 intervention store.

## Runtime cell and raw quantity

The only intervention cell is HB layer 5, denoise round 0, on the ten action
suffix tokens.  The state suffix token is never changed.

For every actually selected expert and token, a forward-pre-hook on that
expert's `down_proj` records the exact tensor received at runtime:

`m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`.

The primary size is `L2(m_e) = sqrt(sum_j m_ej^2)` in checkpoint units.  The
stored descriptive companions are `L1 = sum_j |m_ej|`, signed mean, positive
fraction, and `Linf`.  Reductions are accumulated in fp32 from the runtime
tensor.  No value is divided by hidden, routed, shared, gate, or expert-output
magnitude.  Every selected `(action token, top-k slot)` must be filled exactly
once or the query is rejected.

The pre-gate expert output remains
`v_e = down_proj_e(m_e)`, and the deployed routed mixture is
`r = sum_e w_e v_e`.

## Policies

Each request supplies one policy.  The fixed names and definitions are:

- `baseline`: no routed change and no dropped slot;
- `min_internal_l2`, `max_internal_l2`: argmin/argmax of absolute `L2(m_e)`;
- `min_weight`, `max_weight`: argmin/argmax of returned top-k weight `w_e`;
- `min_raw_output`, `max_raw_output`: argmin/argmax of `L2(v_e)`;
- `uniform_random`: a selected slot drawn uniformly using the stable hash of
  `(pair_id, draw_id, action_token_index, "uniform")`;
- `categorical_weighted`: a selected slot drawn from row-normalized `w_e` using
  the corresponding stable hash, retained as a compatibility control.

Argmin/argmax ties choose the lowest stored slot index.  `uniform_random` is
not gate-weighted.  The preregistered deployable comparison uses
`baseline`, `min_internal_l2`, `min_weight`, `min_raw_output`, and
`uniform_random`; maximum policies are controls.

For every non-baseline policy and action token, if slot `k` is selected:

`r_drop = (r - w_k v_k) / (1 - w_k)`

`delta = r_drop - r`.

The executed routed value is `r + delta`.  This is a post-hoc exact block
counterfactual: all four expert outputs are measured in the instrumented run.
It establishes removal damage, not end-to-end latency saving.

## Pairing and row schema

The independent store is `himoe_hb5_d0_raw_pruning_v5` in
`hb5_raw_pruning_v5.zarr`.  One row is one
`(pair_id, draw_id, policy)` request and stores:

- complete model-normalized `x_traj`, `x0` through `x10`;
- observation and explicit-flow-noise SHA-256 digests;
- policy code, pair/draw/query identities;
- original HB5/d0 action-token input, router probabilities, selected IDs and
  weights, shared output, signed `v_e`, and original routed value;
- runtime-exact internal L2, L1, signed mean, positive fraction, and Linf;
- dropped slot/expert ID, executed routed value, intervention delta, and the
  original routed reconstruction audit.

Within `(pair_id, draw_id)`, all policies must have byte-identical physical
observation, explicit flow noise, `x0`, and every listed pre-intervention
quantity, including all internal statistics.  Dropped slot/ID and all
downstream routes are outcomes and are never pairing invariants.  Baseline must
be an elementwise no-op.

## Frozen endpoints

Against the same-pair baseline, for policy `p`:

`F_final(p) = RMS_live7(x10_p - x10_baseline)`

`F_rem(p) = RMS_live7((x10_p-x1_p) - (x10_baseline-x1_baseline))`.

The v5 K=1 smoke only validates instrumentation, pairing, and numerical
identities.  Its endpoint values cannot support a policy or scientific claim.
