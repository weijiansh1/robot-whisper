# Raw internal activation correction

This note is frozen before inspecting any pre-`down_proj` activation result or
any HB5 drop-one outcome.  It replaces the earlier interpretation of
`E_e(h)`-output RMS as "raw expert activation".

## What is measured

For selected expert `e`, distinguish four different tensors:

1. router probability `w_e`;
2. internal MLP activation
   `m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`;
3. pre-gate expert output `v_e = down_proj_e(m_e)`;
4. routed contribution `c_e = w_e * v_e`.

The raw-activation question refers to item 2.  Its primary size is the absolute
`L2(m_e)` in checkpoint units.  `L1(m_e)`, signed mean, positive fraction, and
`Linf(m_e)` are fixed descriptive companions.  No value is divided by hidden,
shared, routed, post-MoE, or gate magnitude.  RMS may be reported only as the
exact fixed-width rescaling `L2 / sqrt(intermediate_width)` and cannot replace
the raw-L2 result.  Feature scaling inside a training fold is numerical
conditioning, not a redefinition of the raw quantity.

Checkpoint magnitudes are not pooled.  Every predictive model and every raw
comparison is task/checkpoint local; cross-task evidence is an equal-task
summary of task-local effects and signs.

Raw pre-down size is parameterization-dependent: multiplying `m_e` by a
constant and dividing the corresponding `down_proj` columns by that constant
can preserve the expert function.  Therefore raw L2 is tested as a monitoring
or pruning heuristic, not assumed to be functional importance.  The matched
post-down output and paired removal effect are required controls.

## Observational question

HB5/denoise-0 is an exploratory screen on the already collected fresh
`5 tasks x 8 states x 16 candidate noises` grid.  It asks whether the fixed raw
internal statistics add held-out prediction of `x10 - x1` beyond full `x0`,
`x1`, HB5 input hidden, shared output, full router, selected IDs/weights, and
the actual merged routed vector.  State identity and candidate-noise identity
are both absent from training for every test row.

`future path energy` is not a commitment endpoint: on this grid it is almost a
monotone function of the already observed denoise-0 velocity.  A d0-only result
also cannot establish that the model becomes increasingly certain over time.

## Causal compute question

The operational test uses identical observation and explicit `x0` for every
paired arm.  At HB5/denoise-0, one selected expert is omitted for each of the ten
action tokens and the three surviving gate weights are renormalized.  The same
number of expert calls is omitted under every policy.

The fixed deployable policies are:

- smallest raw internal `L2(m_e)`;
- smallest router weight `w_e`;
- smallest pre-gate output `L2(v_e)`;
- deterministic uniform-random selected slot.

The primary utility contrast is raw-internal-minimum versus router-weight-minimum.
Raw activation is useful for pruning only if it causes less live-7 final-action
deviation in all five tasks and a positive equal-task paired effect.  Random and
largest-magnitude removals are controls, not alternative primary endpoints.

For arm `a`, the mechanism endpoints are

`F_final(a) = RMS_live7(x10_a - x10_baseline)`

and

`F_rem(a) = RMS_live7((x10_a - x1_a) - (x10_baseline - x1_baseline))`.

Candidate, state, and task remain paired.  Resampling uses state rows and common
candidate-noise columns, never the 40 token/expert sites as independent samples.
This block-level intervention can establish causal fragility or a low-damage
drop rule, but it cannot by itself establish environment success or end-to-end
latency savings.

## Commitment experiment still required

Increasing certainty requires a separate all-denoise capture.  At each round,
the analysis must condition on the complete current latent and current velocity,
then test future `x10` correction or future basin-rank changes.  Candidate-pair
statistics must be cross-fitted and resampled on state/candidate axes; the 120
pairs in a K=16 pool are not independent observations.  No d0-only result will
be relabelled as a temporal commitment result.
