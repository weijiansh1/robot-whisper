# Prereg — the four logged routing channels nobody has opened

Written 2026-09-06, before any external outcome label is read in this bundle and
before any detector built here is evaluated on `right-50x8b-20260903`.

## Scope

The Zarr route logs hold five arrays. Every quantity in this project so far
derives from one of them, `hb_router_probs`. This bundle opens the other four:

| array | shape | never read before |
|---|---|---|
| `hb_expert_ids` | (query, 8, 10, 11, 4) | except `analysis_trap_taxonomy` |
| `hb_selected_prob` | (query, 8, 10, 11, 4) | yes |
| `hb_entropy` | (query, 8, 10, 11) | yes |
| `as_probs` | (query, 4, 3) | yes |
| `as_expert_ids` | (query, 4) | yes |

## Prior exposure (disclosed)

Before this prereg was written the following were already seen and are therefore
**not** independent evidence produced here:

1. The controller's brief, which supplied three external anchors
   (`mobility|L12|low|q0.975|global` 195/17 lift 1.7641,
   `mobility|L2|low|q0.700|per_task` 272/57,
   `expert_load_effective_rank|L3|low|q0.85|per_task` 370/93 lift 1.5479),
   the 48–82% within-frame vs 14–31% cross-frame false-alarm dependence figures,
   the survival caps and prior crossings, and the two audit warnings.
2. A **development-only** probe of channel semantics, run before this file was
   written, on `right-50x8-20260903` only, reading no outcome labels. It
   established the four verification facts in §1 and the `as_*` constancy in §2.
   It is a **reproduction target**, not an independent test. It is labelled as
   such throughout. No external route file and no outcome label was opened.
3. `analysis_trap_taxonomy/results/expert_identity_summary.json`, supplied by the
   controller: top-4 occupancy AUC vector correlation 0.798 vs 0.846 soft for the
   loop event; expert identity does not replicate across runs (top-10 overlap
   4–6 of 10).

## Terms (inherited unchanged)

- `risk` = `original_failure`: the episode did not finish before the suite
  horizon cap. Caps: goal 30, long 52, object 28, spatial 22.
- Survival prior = per-suite P(risk | still running at chunk q); `low_prior` /
  `early` = alarm chunk where that prior < 0.25 (crossings at chunk 18/26/17/13).
- `lift` = precision / mean survival prior matched to each alarm's own chunk.
- Causal: a score at chunk q uses chunks 0..q only; all ten denoising steps of
  chunk q are available.
- Frozen detector protocol: `trailing_mean` W=4, `persistent_score` K=4,
  `row_max` peak, `quantile_higher`, `first_query`, the 14 `QUANTILES`, and the
  11 `REPRESENTATIONS` (L2..L5, L12..L15, front_median, back_median, all_median),
  both `per_task` (crossfit, leave-one-initial-state-out) and `global` (one
  pooled quantile over development main + extra, no task identity) modes.
  Selection rule applied identically to every quantity:
  `timely_fpr <= 0.005` and `low_prior_precision >= 0.60`, ranked by
  `low_prior_tp` then `low_prior_precision`.

## Step 0 — anchors

Before anything else, rebuild the three anchor heads from the frozen caches and
assert exact TP/FP and lift to 4 decimals. Any mismatch aborts the bundle.

## Step 1 — channel semantics (descriptive, outcome-free, both run_ids)

Full pass over every query row of both run_ids. Predeclared checks:

- **C1** `hb_selected_prob == take_along_axis(hb_router_probs, hb_expert_ids)`.
  Report max absolute deviation over all cells.
- **C2** `hb_expert_ids` is the top-4 of `hb_router_probs`. Because the log is
  stored in float16, exact ties at the k/k+1 boundary are expected; a mismatch
  counts as a **genuine violation** only if the 4th and 5th largest stored
  probabilities are *not* equal. Report both counts separately, plus
  `max |sum(hb_selected_prob) - sum(top-4 of hb_router_probs)|`, which is 0 iff
  the selected set always attains the maximal 4-subset mass.
- **C3** `hb_entropy == H(hb_router_probs)` over the full 32-way axis. Report max
  absolute deviation and compare it to the float16 ULP at that magnitude. Also
  compare `mean over the ten action tokens of hb_entropy` against the frozen
  `token_entropy` column of `moe-flow-semantics-0906/results/step_profiles`,
  which would make `hb_entropy` an already-swept quantity.
- **C4** Placeholder check, per `analysis_trap_taxonomy/REPORT.zh.md` line 90
  (`expert_ids=0` is a placeholder in some caches): count cells and rows where
  `hb_expert_ids` is all-zero, and confirm all 32 expert indices appear.
- **C5** `as_probs` and `as_expert_ids`: within-episode, within-task,
  within-suite and across-suite variation; number of distinct values; whether
  `as_expert_ids == argmax(as_probs)`.

**Predeclared consequence.** If C1–C3 hold, `hb_selected_prob`, `hb_entropy` and
(up to float16 ties) `hb_expert_ids` are deterministic functions of a channel the
project already exhausts. Per the controller's compounding argument they can then
only lose information, never add it, and **no detector sweep is run over
discrete-selection statistics.** A sweep is run only if C2 fails with genuine
(non-tie) violations.

## Step 2 — is `as_*` a fifth reference frame? (primary question)

Predeclared decision tree, evaluated on development first:

- **2a** Measure the within-episode variance of every `as_*`-derived per-query
  scalar. If it is exactly zero for every episode, `as_*` admits no
  chunk-resolved score and cannot be a reference frame in the sense the frame
  survey uses. Report that, and continue to 2b only to make the consequence
  concrete; do not present 2b as evidence that `as_*` is a detector family.
- **2b** Run the frozen sweep anyway on the six `as_*` scalars defined below,
  both modes, and report exactly what a per-episode-constant quantity does under
  this protocol. Select on development, replay on external **once**.
- **2c** False-alarm dependence (observed co-occurrence over the product of
  marginals, on timely episodes only) of every surviving `as_*` head against the
  24 existing heads in
  `moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz`.
  A fifth frame requires cross-frame ratios in the published 14–31% band, i.e.
  clearly below the 48–82% within-frame band.
- **2d** Relation to `hb_*`: at whatever granularity `as_*` actually varies.

`as_*` scalars, declared now (per query, per AS layer l in 0..3, from
`as_probs[q, l, :]` over the 3 AS experts and `as_expert_ids[q, l]`):

| name | definition |
|---|---|
| `as_entropy` | H(as_probs[l]) |
| `as_top1` | max_e as_probs[l, e] |
| `as_margin` | largest minus second-largest |
| `as_switch` | 1 if `as_expert_ids[q,l] != as_expert_ids[q-1,l]` else 0 |
| `as_selected_prob` | as_probs[l, as_expert_ids[l]] |
| `as_layer_disagreement` | mean pairwise total-variation distance over the 6 AS-layer pairs (a single scalar, broadcast to all four "layers") |

The four AS layers stand in for the eight HB layers; the three aggregates
(`front_median` over layers 0–1, `back_median` over layers 2–3, `all_median`) are
formed the same way, giving 7 representations instead of 11. This is a
**smaller** search than the established sweep, so the search-scale inflation
floor measured in Step 3 is an upper bound for it.

## Step 3 — negative controls under the identical sweep

The controller's warning: a control known flat by construction still reached lift
1.56 under an 11 x 2 x 14 sweep. Two controls, both seeded 20260906:

- `null_iid` — i.i.d. U(0,1) per (episode, chunk, one of 8 synthetic layers),
  masked by the real `valid`. Full 11-representation sweep. This is the
  inflation floor for the established sweep size.
- `null_episode_constant` — one i.i.d. U(0,1) draw per episode, held constant
  across chunks, 8 synthetic layers. This is the **matched** control for `as_*`,
  which is episode-constant by observation.

Whatever these achieve is reported next to every `as_*` number. An `as_*` head
that does not clear its matched control is reported as not clearing it.

## Step 4 — task-stratified effect sizes

Every claim about a quantity carrying information is accompanied by a
survival-conditioned AUC at chunks 4,6,8,10,12,14,16,18,20,24,28,32 under three
stratifications — pooled, within-suite, within-task — using the exact
Mann-Whitney weighted-pooling estimator of
`moe-audit-0906/experiments/recheck_stratification.py` with `MIN_STRATUM = 30`.
Suite stratification is treated as anti-conservative. Only the within-task column
supports a claim.

## What would make each answer

- `as_*` **is** a fifth frame: it varies within episodes, produces a head that
  passes the selection rule on development, replays on external with lift above
  its matched control, and shows cross-frame dependence ratios against all 24
  existing heads in the 14–31% band.
- `as_*` **is not** a fifth frame: it is constant within episodes, or its head
  fails to clear the matched control, or its dependence ratios sit in the
  within-frame band.
- The discrete selection **adds** something: C2 fails with genuine non-tie
  violations, i.e. `hb_expert_ids` is not recoverable from `hb_router_probs`.
- The discrete selection **adds nothing**: C1–C3 hold; the three arrays are
  deterministic functions of an exhausted channel.

## Outputs

`results/` only. Nothing outside `moe-unused-channels-0906/` is written. No git
command is run.
