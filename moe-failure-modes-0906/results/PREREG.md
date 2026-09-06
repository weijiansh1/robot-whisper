# Prereg — MoE routing detectors vs. physical failure modes

Written 2026-09-06, before any head search on development and before any
external evaluation of a detector built in this bundle.

## Prior exposure (disclosed)

The controller supplied an external recall-by-physical-mode table for 5 existing
detectors, and the external per-mode risk counts. Those numbers were therefore
already seen before this bundle started. Consequences:

- Step 1 (verify the probe) is a **reproduction**, not an independent test. It is
  labelled as such everywhere.
- Step 2 (confound control) uses statistics that were **not** part of the prior
  exposure (stratified permutation nulls, standardised recalls). It is
  confirmatory for the *existence* question and exploratory for effect sizes.
- Step 4 (the multi-head) is selected on **development only** and evaluated on
  external **once**. No external number influences its composition.

## Terms

- `risk` = `original_failure` in the outcome labels: the episode did not finish
  before the suite horizon cap. Caps: goal 30, long 52, object 28, spatial 22.
- `mode` = `primary_failure_reason` from the simulator-backed physical labels.
  It is a **failure-mode label, not proof of internal causal mechanism**
  (per the labelling authors). Used for ANALYSIS AND VALIDATION ONLY.
- `persistent` risk = risk and not `late_success_plus10_queries`.
- Survival prior = per-suite P(risk | still running at chunk q). `early` /
  `low_prior` = alarm chunk where that prior < 0.25.
- Detector scores read routing tensors derived from `hb_router_probs` only.

## Q1 — is the mode-differential recall real, or a confound?

Null hypothesis H0(d): for detector `d`, whether a risk episode is caught is
independent of its physical mode, **given the stratum**.

Two strata are pre-declared and both are reported:

- `suite` (4 levels) — the confound the controller named.
- `task` (39 external / 37 development levels) — strictly finer; also absorbs the
  per-task threshold channel.

Test: stratified permutation. Permute the `mode` vector among risk episodes
**within each stratum**, 20,000 times, fixed seed 20260906. Statistics:

1. Omnibus per detector: `T = sum_m n_m * (r_dm - r_d|stratum-matched)^2`,
   computed as the stratified chi-square-like deviance of the mode x fired table.
2. Per (detector, mode): the standardised recall `r_dm` under direct
   standardisation to a common suite/task mix; two-sided permutation p.

Multiplicity: Benjamini-Hochberg at q = 0.05, applied over all
(detector, mode) cells within each stratification.

Timing control. Every risk episode runs to its suite cap, so within a suite all
risk episodes expose the detector to the identical number of chunks; there is no
differential opportunity within stratum. The residual timing question is
therefore reported descriptively and as a restricted analysis:

- mean alarm chunk and mean survival prior at alarm, by (detector, mode);
- recall restricted to the early band (prior at alarm < 0.25), by mode;
- a prior-matched recall: recall computed with alarms censored at the chunk where
  the suite prior first reaches 0.25, so all detectors are compared on the same
  survival-prior budget.

Reporting rule: every rate is reported with its numerator and denominator. Cells
with fewer than 30 risk episodes are marked `small` and no conclusion is drawn
from them alone.

## Q2 — selectivity, and dev -> external stability

Selectivity is defined relative to the detector's own overall recall:

    S(d, m) = log2( r*(d,m) / r*(d) )

with `r*` the suite-standardised recall (common suite mix = the pooled risk suite
distribution of the cohort). A mode is `selective` for `d` if S > 0 and the
BH-adjusted permutation p < 0.05 under **both** stratifications.

Stability: compute S on development and external independently; report Spearman
rank correlation across (d, m) cells, sign-agreement rate, and the per-detector
argmax-mode agreement. Pre-declared bar for "stable": sign agreement >= 0.7 over
cells with >= 30 risks in both cohorts.

## Q3/Q4 — the multi-head detector

Head grid, pre-declared and finite: the 24 detectors already frozen in
`moe-hb-front-back-0905/results/frame_survey/external_detectors.csv`
(12 routing quantities x {`global`, `per_task`}), rebuilt on development with the
identical representation / direction / quantile / width-4 trailing mean /
4-confirmation machinery, plus the 5 v7 heads. No new thresholds are fitted.

Selection objective (frozen here, evaluated on development only):

    maximise development TP subject to development FP <= FP_budget

with FP_budget matched to the operating point being challenged, searching only
over OR-combinations of at most 4 heads, greedily (add the head with the largest
development TP gain that keeps FP within budget; stop when no head helps).
Separate bundles for `global` and `per_task`, never mixed, because only `global`
proves the signal is MoE-side.

Three budgets are declared, one per external operating point to beat:
- `global`: development FP budget scaled from external 17 FP (best single
  `mobility|global`) and 13 FP (best frozen combination), using the development
  non-risk count. Concretely FP_budget_dev = round(FP_ext * 14313 / 15036).
- `per_task`: from external 93 FP (best single) and 57 FP (best combination).
- `v7`: from external 80 FP.

Naming rule (the anti-circularity rule). A head may be named after a physical
mode only if, on **development**, that mode's suite-standardised selectivity
S(d, m) is the head's maximum, is > 0, and its permutation p is significant under
both stratifications. The name is then **validated on external** by checking the
same mode is still the argmax with S > 0. Names are never taken from clustering
detector outputs. If no mode qualifies, the head is named `unselective`.

Comparison at matched false alarms: for each mode, the multi-head bundle is
compared against the best single head **after** censoring the single head's
alarms so its external FP count is <= the bundle's. Censoring rule declared here:
raise the single head's confirmation count until its FP is at or below the
bundle's, which is a causal, train-free operation on the same score.

## Hard constraints (restated, binding)

- No physical label enters any alarm decision, not even indirectly.
- Alarm at chunk q reads queries 0..q only.
- No fitted weights, no classifier, no gradient updates.
- `per_task` and `global` reported separately.
- Every rate is scored against the survival base rate; lift = precision / mean
  matched prior.
