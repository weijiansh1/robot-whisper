# Preregistration — two-tier (WATCH / ACT) MoE routing monitor

Written and frozen **before** any combination rule was scored on `external_8b`.
Only `experiments/build_alarms.py` had touched external at the time of writing,
and only to check that the rebuilt per-head arrays reproduce the already
published `moe-hb-front-back-0905` survey arrays bit for bit (they do, all 24).
Per-head external numbers were therefore already public before this study; what
is sealed here is **which combination rule gets built and how it is judged**.

## 0. Standing constraints inherited, not re-derived

- Scores read routing tensors only (`hb_router_probs` -> layer graphs -> the
  twelve quantities). No physical state, no vision, no reward.
- Representations, causal trailing mean (W=4), consecutive confirmations (K=4),
  cross-fitted per-task thresholds, pooled global thresholds and the survival
  prior scoring are the frozen v4 / hb-front-back code, imported not copied.
- Train-free. No weights are fitted. Selection is a search over a finite,
  predeclared grid, on development outcomes.

## 1. Head pool (frozen)

**Primary pool** — the 24 frame-survey heads, one per (quantity x threshold
mode), with the representation / direction / quantile the survey's own
per-quantity development rule already picked. Frames as declared by the survey
before it ran:

| frame | quantities |
|---|---|
| `adjacent_query` | mobility, conditional_query_d1, partial_query_d1 |
| `denoising_steps` | flow_path, flow_endpoint, flow_settling_log_ratio |
| `token_graph` | action_consensus, state_action_alignment, conditional_energy, conditional_effective_rank, partial_edge_std |
| `expert_load` | expert_load_effective_rank |

`per_task` rules draw only from the 12 per_task heads; `global` rules only from
the 12 global heads. The two modes are selected and reported separately.

**Secondary pool** — primary pool + `v7_guard|global`
(`moe-v7-0905` sealed guard, 439 TP / 80 FP external). It is declared to the
`adjacent_query` frame: all four of its components (freeze, acceleration,
periodicity, turbulence) compare query q to the same episode's recent past.
It is *not* an independent frame, and because it is itself a 4-head OR it is
reported only as a labelled sensitivity check, never as the headline rule.

Heads whose development *and* external alarm arrays are bitwise identical to
another head are flagged as duplicates in the output; a rule containing a
duplicate pair is reported but is not credited with extra frame diversity.

## 2. Causal combination

A `k`-of-`n` rule alarms at the chunk where the `k`-th component alarm arrives,
i.e. the `k`-th smallest component first-alarm chunk, and only if at least `k`
components ever fire. `k=1` is OR (min); `k=n` is AND (max). By construction a
combined alarm at chunk q uses only component alarms at chunks <= q.

## 3. Scoring (identical for every rule, both splits)

- TP / FP / precision / risk recall / timely FPR.
- Survival base rate: per suite, `P(risk | length > q)`. Each alarm is matched
  to the prior at its own alarm chunk; `lift = precision / mean matched prior`.
- Early band: alarms whose matched prior is below 0.25 (`low_prior_tp`,
  `low_prior_fp`). Reported for every rule.
- Physical-mode coverage: fraction of that mode's risk episodes that alarm.
  **Worst-mode coverage** = min over scored modes; **mode CV** = std/mean over
  scored modes. A mode is *scored* if it has >= 10 risk episodes in that split.
  On both splits that admits the same seven modes and excludes
  `no_meaningful_target_progress` (1 dev / 2 ext) and
  `mechanism_threshold_not_reached` (6 dev / 1 ext). Every rate is reported
  next to its n.

## 4. WATCH — coverage guarantee

Objective: **no physical failure mode is systematically invisible.** Recall is
not the target.

Shared feasibility constraint for every WATCH candidate:
combined development `timely_fpr <= 0.005` (the project's standing v4-level
false-alarm cap; = 71 development false alarms).

**Variant (a) — frame-diverse, unsupervised.** Uses development *binary*
outcomes only; physical-mode labels are never read.
1. Within each frame, take the head with the highest development TP among heads
   whose own development `timely_fpr <= 0.005`. Ties: higher development
   precision, then alphabetical.
2. OR the four per-frame winners.
3. If the 4-way OR busts the shared FP cap, drop frames one at a time in
   ascending order of development TP until it fits, and report both.

**Variant (b) — minimax mode coverage.** Exhaustive search over all subsets of
size 2..5 of the pool that span >= 2 frames; pick the subset maximising
development worst-mode coverage subject to the shared FP cap. Ties: higher
development mean-mode coverage, then fewer heads, then lower development FP.
**This variant puts physical-mode labels into rule selection.** That is a step
beyond this project's convention of selecting on binary development outcomes
only, and is declared as such wherever it is reported. It is never proposed as
a deployable selection procedure.

**Same-frame control.** For each variant's chosen size, the best subset drawn
from a *single* frame under the identical objective and constraint, so the
cross-frame claim has a matched comparison.

**The question of interest**, declared now: does (a) approximate (b)? Judged on
external by (i) head-set overlap, (ii) worst-mode coverage gap, (iii) mode-CV
gap. If (a)'s external worst-mode coverage is within 0.05 of (b)'s, the
unsupervised criterion is said to approximate the supervised one; otherwise the
"blind-spot elimination is free" claim is reported as **not supported**.

## 5. ACT — high-confidence trigger

**Selected independently of WATCH.** No head is inherited.

Objective: **maximise development precision subject to development TP >= 50**
(~10% of the 487 development risk episodes; declared now as "useful volume").

Grid: all pairs from the pool with `k=2`, and all triples with `k=2` and `k=3`,
restricted to head sets spanning >= 2 distinct frames. Ties: higher development
TP, then fewer heads, then earlier median development alarm chunk.

If no candidate reaches 50 development TP, that is reported as **ACT is not
viable at the declared volume**, the constraint is relaxed once to TP >= 25 and
the relaxation is labelled post-hoc.

## 6. Joint evaluation

- Episode counts: WATCH-only, ACT (and therefore WATCH if nested), neither.
- False alarm budget at each tier.
- Is `ACT alarms ⊆ WATCH alarms`? Reported as a measured containment rate, not
  assumed. ACT alarm chunk vs WATCH alarm chunk on shared episodes.
- Precision and lift at each tier.

## 7. Anchors that must reproduce (assert, do not report otherwise)

- `mobility|L12|low|q0.975|global` = 195 TP / 17 FP, precision 0.9198, lift 1.7641
- `mobility|L2|low|q0.700|per_task` = 272 TP / 57 FP
- `expert_load_effective_rank|L3|low|q0.85|per_task` = 370 TP / 93 FP, lift 1.5479

## 8. Claims that are forbidden in advance

- **No claim about an individual head's failure-mode specialisation.** A
  companion agent showed that apparent per-head mode selectivity is largely a
  task confound: 20/261 cells survive suite-stratified permutation nulls but
  only 2/261 survive task-stratified ones, and this is not a power problem.
  Any mode-related claim here must be about the *combination's* coverage or the
  *increment*, and must state which stratification it survives.
- **No claim that combining raises information density.** Across this project
  lift falls monotonically as heads are combined (1.764 single -> 1.488 OR ->
  1.290 k=2). If a tier's lift is at or below the single-head 1.764, the report
  must say plainly that it bought coverage, not information.

## 9. Post-hoc labelling

Anything decided after external numbers are seen is labelled `post_hoc: true`
in the machine-readable output and marked in the report.

---

## 10. ADDENDUM — declared after one look at development, before external

The sections above are the original seal. Running section 4 and 5 on
development exposed two defects in the *original* objectives. Following this
project's own convention (`moe-hb-front-back-0905/experiments/select_early_lock.py`
objective B), the repaired objectives are added here rather than substituted,
both are scored, and neither is presented as an independent confirmation of the
other. **External was still closed when this addendum was written.** The only
external contact so far remains the per-head reproduction check.

### 10.1 WATCH variant (a) was underspecified

"Best head per frame by solo development TP" ranks a frame's heads by volume,
so it picks the frame's most false-alarm-hungry head; the 4-frame OR then busts
the shared cap and the drop rule collapses it to two heads. That is a property
of the ranking, not of frame diversity, and it would confound the question of
interest. Three unsupervised variants are therefore scored, all of which read
binary development outcomes only and never physical-mode labels:

- **(a1) solo-best, then drop** — exactly as section 4 declared.
- **(a2) solo-best under an equal budget share** — within each frame take the
  highest development-TP head whose own development FP is at most `cap/4`
  (one quarter of the shared budget, one quarter per frame). Keeps four frames
  by construction.
- **(a3) frame-diverse greedy marginal gain** — start empty; repeatedly add the
  head, from a frame not yet represented, that maximises the *increase* in
  development TP of the running OR, subject to the running OR staying inside
  the shared cap; stop when no addition helps or five heads are reached. Also
  run without the one-head-per-frame restriction as `a3_free`.

(a3) is the variant that actually operationalises "use frames to buy coverage":
marginal gain is what a coverage tier should be greedy about, and it is still
computable from binary outcomes alone.

### 10.2 A size-matched supervised comparator

WATCH (b) is free to pick any size in 2..5, so an (a)-vs-(b) gap could be a size
gap. **(b4)** repeats the minimax search restricted to exactly four heads, the
size (a2) and (a3) produce, giving a like-for-like comparison.

### 10.3 Fixed reference rules

Scored, not selected: the controller's cross-frame trio
`mobility|global + flow_path|global + expert_load_effective_rank|global`, and
the best *same-frame* OR at each reported size under the identical objective and
cap.

### 10.4 ACT objective A rewards firing late

Every top-precision AND on development reaches precision 1.000 with lift 1.0-1.2
and 0-3 early true positives: the precision is the survival base rate, obtained
by firing at chunk 19-40 when almost everything still running is already
failing. An ACT tier that only fires after the fact is not an ACT tier.

- **ACT-A** — as declared in section 5: maximise development precision subject
  to development TP >= 50.
- **ACT-B** — maximise development `low_prior_tp` subject to development
  precision >= 0.95 and development TP >= 50. Ties: higher lift, then higher
  TP, then fewer heads. Same grid, same independence from WATCH.

ACT-B is the headline ACT rule; ACT-A is reported beside it as the thing the
naive objective would have chosen, and the difference between them is itself a
result.

### 10.5 Increment concentration test (evaluation, on external)

For the WATCH OR, the *increment* is the set of risk episodes the OR catches
that its highest-TP single member does not. Statistic: the coefficient of
variation of the increment's per-mode coverage rates over the seven scored
modes. Null: permute the increment indicator among risk episodes **within
stratum**, preserving each stratum's increment count; 10,000 draws; one-sided
p = P(null CV >= observed CV). Run twice, stratifying by suite and by task.
Any mode-related claim states which stratification it survives. No claim is
made about an individual head's mode selectivity.

### 10.6 Decision rule for "does (a) approximate (b)?"

Unchanged from section 4: on external, (a) approximates (b) if the external
worst-mode coverage gap is within 0.05. Evaluated for each of a1/a2/a3
separately, and reported with the head-set Jaccard overlap.
