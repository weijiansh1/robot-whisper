# Pre-registration — state channel and within-chunk arc formation

Written 2026-09-06, **before** any external-8b outcome was read for any quantity
declared below.  Anything added after the first external read is labelled
`post-hoc` in the report and in `results/manifest.json`.

## 0. Frozen machinery (not re-implemented)

`experiments/protocol.py` in `moe-flow-semantics-0906` is imported verbatim
(`sys.dont_write_bytecode = True`, so nothing is written into that directory).
It carries the frozen detector protocol: `WIDTH = 4`, `CONFIRMATIONS = 4`,
the 14-point quantile grid, the leave-one-initial-state-out cross-fitted
`per_task` thresholds, the pooled `global` threshold, the survival prior and
the `low_prior < 0.25` early band.

**Anchors that must reproduce exactly, else the run aborts:**

| head | mode | expected |
|---|---|---|
| `mobility` step 9, L12, low, q0.975 | global | 195 TP / 17 FP, precision 0.9198, lift 1.7641 |
| `mobility` step 9, L2, low, q0.700 | per_task | 272 TP / 57 FP |
| `load_entropy` step 9, L3, low, q0.850 (= published `expert_load_effective_rank`) | per_task | 370 TP / 93 FP, lift 1.5479 |

## 1. Question A — within-chunk arc formation

### A.0 Algebraic decomposition (no labels, no selection)

For each (query, layer, denoising step), with `a_1..a_10` the unit-norm
square-root routing vectors of the ten action tokens and `s` that of the state
token, define `c_t = <a_t, s>` and `G_ij = <a_i, a_j>`.

Declared identities to verify numerically against raw Zarr:

* `conditional_energy` (the published Schur-complement diagonal) `= 1 - mean_t(c_t^2)`
* centred configuration size `trace(H G H)/10 = 0.9 * (1 - action_consensus)`
* along-state centred energy `= var_t(c_t) = 1 - conditional_energy - state_action_alignment^2`

If these hold, the "lower `conditional_energy` vs higher centred configuration
size" contradiction is decided by algebra rather than by fitting, and the
resolution is reported as such.

### A.1 Descriptive (development only, no thresholds)

Per denoising step 0..9, per layer, report the population medians of:
`pc1_index_corr`, `pc1_share`, `fiedler_index_rho`, `arc_stretch`,
`neighbour_ratio`, `centred_energy`, `along_state_energy`,
`along_state_fraction`, `pc1_state_cos`, `state_index_corr`,
`token_differentiation`, `state_action_alignment`.
Report front (L2-L5) versus back (L12-L15) separately.  Report at which step the
arc ordering first exceeds a fixed reference level.

### A.2 Predictive — declared detector quantities

Every quantity below is a per-(episode, query, layer) scalar computed from the
routing tensor of query `q` alone (all ten denoising steps of `q` precede the
emitted action, so this is causal), then fed unchanged to the frozen protocol.
`slope` always means the OLS slope of the quantity against denoising step index
0..9 within one query; `late_early` means mean(steps 7,8,9) - mean(steps 0,1,2).

**Primary family A (formation slope)** — the pre-declared object of the study:

| id | quantity |
|---|---|
| F1 | `td_slope` — slope of `token_differentiation` |
| F2 | `td_late_early` — late-early of `token_differentiation` |
| F3 | `pc1corr_slope` — slope of `pc1_index_corr` |
| F4 | `pc1share_slope` — slope of `pc1_share` |
| F5 | `stretch_slope` — slope of `arc_stretch` |
| F6 | `centred_slope` — slope of `centred_energy` |

**Secondary family A (endpoint / level)**:

| id | quantity |
|---|---|
| E1 | `pc1_index_corr_s9` |
| E2 | `arc_stretch_s9` |
| E3 | `fiedler_index_rho_s9` |
| E4 | `along_state_energy_s9` |
| E5 | `along_state_fraction_s9` |
| E6 | `pc1_state_cos_s9` |

**Negative control**:

| id | quantity |
|---|---|
| N1 | `saa_slope` — slope of `state_action_alignment`, which the controller probe reports as flat |

## 2. Question B — the state channel

| id | quantity |
|---|---|
| S1 | `state_mobility_s9` — cross-query Hellinger of the **state token** route (primary) |
| S2 | `state_mobility_mean` — mean over the ten steps |
| S3 | `state_minus_action_mobility` — `state_mobility_s9 - mobility_s9` |
| S4 | `state_over_action_logratio` — `log(state_mobility_s9 + 1e-6) - log(mobility_s9 + 1e-6)` |

Reference-frame independence is measured as **false-alarm dependence**:
`P(both fire | survivor) / [P(A fires | survivor) P(B fires | survivor)]`,
i.e. observed co-occurrence over the product of marginals, computed on
timely (non-risk) episodes only.  1.0 means independent reference frames.
Raw overlap is *not* reported as evidence of independence.

`libero_object` is inspected separately for the reported anti-predictive
prefix coupling; this is declared exploratory, n = 44 external risks.

## 3. Selection and evaluation rule (identical for every quantity)

Selection reads **development_main outcomes only**.  For each quantity and each
threshold mode independently:

```
eligible  : timely_fpr <= 0.005  AND  low_prior_precision >= 0.60
rank by   : low_prior_tp, then low_prior_precision
```

over the 11 published representations (L2, L3, L4, L5, L12, L13, L14, L15,
front_median, back_median, all_median) x {low, high} x the 14-point quantile
grid.  The single winner per (quantity, mode) is rebuilt on external with
development-derived thresholds and scored **once**.

`per_task` and `global` are reported separately.  Only `global` is treated as
evidence that the signal is MoE-side.

## 4. Multiplicity, declared in advance

17 quantities x 2 threshold modes = 34 external heads in one pass, plus the
false-alarm dependence table.  The pre-declared primary comparisons are
family A (F1-F6) for question A and S1 for question B.  The best head over all
34 is optimistically biased and will not be claimed as a tuned detector.

## 5. Success criteria, declared in advance

* Question A is answered **positively** only if at least one member of family
  A (F1-F6) reaches, on external and in `global` mode, `lift` whose bootstrap
  interval excludes 1.0 **and** `low_prior_tp >= 20`.
* Question B is answered **positively** only if S1 in `global` mode reaches
  `lift` above 1.0 with a bootstrap interval excluding 1.0, and its false-alarm
  dependence with the published `mobility` head is below 3.0.
* Anything weaker is reported as a negative result with its numbers.
