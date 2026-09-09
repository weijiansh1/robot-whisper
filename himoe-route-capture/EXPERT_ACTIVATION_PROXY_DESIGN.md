# Retrospective hidden-matched expert-activation proxy

Frozen before running `analyze_expert_activation_proxy.py` on 2026-08-22.
This is an exploratory follow-up on already inspected Long-t08 data, not a new
confirmatory experiment.

## Scope

The existing capture does not contain the intermediate flow latent `x_tau` or
runtime expert outputs. Therefore this analysis cannot implement an exact
surface-token twin or a final-action intervention. It uses the full pre-MoE HB
hidden tensor as a stricter nearest-neighbor proxy and reconstructs expert
outputs offline from fp16 hidden values and checkpoint weights.

The primary cell is fixed to HB layer 5, denoise step 0. Layer 5 is the last
early HB block and step 0 is the earliest point at which a useful warning could
be emitted. No layer/time search is part of this analysis.

## Arm 1: hidden-matched future-action divergence

For each of the 16 exact-observation K=32 pools:

1. Compute pairwise RMS distance between the ten action-token pre-MoE hidden
   tensors at L5/d0.
2. Keep the closest 10% of the 496 candidate pairs (50 pairs).
3. Within those 50 pairs, call the lowest 20% final-action distances stable and
   the highest 20% divergent (10 pairs per class per pool).
4. Report scene-macro AUC for fixed scalar scores. The discovery family is the
   level and pair-difference form of D, C, Q, and their standardized DCQ mean.

The exact proposal (current q10 AND global final q10/q90) is also counted. It is
not substituted silently if it has too few divergent pairs.

Inference uses 5,000 candidate-label permutations. In each permutation, one
common permutation of the 32 repeated seed columns reassigns final actions in
all pools; the hidden match set and internal scores remain fixed. The maximum
AUC excess over 0.5
across the eight discovery scores gives one-sided FWER p-values. Confidence
intervals resample the 16 pools.

## Arm 2: offline immediate block sensitivity

At the same L5/d0 action-token sites, replace the recorded top-4 experts with
the four highest-probability experts not in the recorded set. Renormalize their
stored router probabilities and compute

`F_block = RMS(r_alt - r_true) / RMS(r_true + shared)`.

This is an immediate one-block sensitivity proxy, not the final-action
counterfactual from a live intervention. Candidate scores are averaged over the
ten action tokens. D, C, Q, and standardized DCQ form the discovery family.
Significance uses 5,000 common seed-column permutations across all 16 pools,
which preserves the repeated-seed structure. Confidence intervals resample
pools.

## Interpretation rules

- A matched-pair result above noise but not above hidden/shared controls is an
  internal monitoring interface, not expert-specific information.
- A D/C/Q association with `F_block` supports internal block fragility only.
- Neither arm establishes extra information beyond the complete hidden state.
- Runtime-exact output geometry, AS outputs, intermediate `x_tau`, and final
  action effects require a new instrumented rollout.

## Dry-run deviation record

After a 50-permutation implementation dry run, before the final 5,000-
permutation run, one denominator control was added. `Q = 1-cos(r,s)` changes
the inner-product term inside `RMS(r+s)`, which is the primary `F_block`
denominator, so their association can be partly algebraic. The frozen primary
target remains unchanged. Two sensitivity
targets, `RMS(delta r)/RMS(r)` and `RMS(delta r)/RMS(h)`, are now reported to
show whether an effect survives without the shared post-MoE denominator. No
cell, matching fraction, discovery feature, or primary decision rule changed.
