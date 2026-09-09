# Online Intrinsic Routing Guard v8

## Contract

Unchanged from v7. The runtime monitor receives only the current
`hb_router_probs` tensor with shape `[8, 10, 11, 32]`. It has no task or suite
identity, task prototype, outcome, **horizon**, simulator state, future query,
peer rollout, or fitted model.

One global profile is calibrated from an unlabeled routing reference corpus. It
contains v7's three scalar thresholds and one robust scale, plus **two new
scalar thresholds**. It contains no task-indexed array.

v8 adds no new input. Both new heads are functions of one quantity already
present in the tensor:

```text
action_root      = sqrt(p)[:, :, 1:11, :]                      # action tokens
flow_speed[l, k] = ||action_root[l, k+1] - action_root[l, k]|| / sqrt(2),
                   averaged over the ten action tokens          # k = 0..8
```

The entire v8 data contract is one `[layer, 9]` array per query. `layer_graphs`
and `step_profiles` are not required; `flow_path = sum_k flow_speed` reproduces
the published cache to 2.4e-4 and the extraction to 9.6e-6.

## Internal mechanisms

v7's three heads are unchanged:

1. **Relative freeze** — final-flow action-token Hellinger mobility per layer,
   divided by the same layer's mean mobility at q1..q4, median log contraction
   over L12..L15, causal width-6 mean.
2. **Confirmed turbulence** — curvature of the soft expert route along the ten
   denoising iterations, persisting 8 queries, accepted only after a second
   head observes persistent loss of long-lag route recurrence relative to lag 1
   (persisting 4 queries).

v8 adds two:

3. **Front/back flow inversion.** The front layers respond to new observation
   while the back layers hold a stable relational shape: the front/back
   cross-query conditional-graph response ratio is 7.35 / 7.41 / 6.38 across
   the three cohorts, in the same direction in 40/40, 39/39 and 5/5 tasks, and
   the front/back flow-path ratio is 1.34 / 1.33 / 1.34. This head fires when
   that ordering **inverts** — when the front stops moving more than the back.

   ```text
   score = causal_W6_mean( log( mean_front sum_k flow_speed
                              / mean_back  sum_k flow_speed ) )
   fires when score <= threshold_frontback
   ```

   This head is **not** self-baselined. The ratio is already normalised across
   layers, and forcing v7's self-baseline onto it measurably destroys it
   (+169 TP for +1417 FP versus +93 TP for +40 FP without). It is the only
   head in the guard without a self-baseline, and that exception is deliberate.

4. **Three-step curvature.** The denoising axis is rank-1 for every functional
   measured (participation ratio 1.00–1.52 of 10), so v7's ten-step curvature
   is over-sampled. This head reads **steps 1, 5 and 9 only**.

   ```text
   curvature = mean_{back layers} | second difference of flow_speed
                                    over steps (1, 5, 9) |
   score     = causal_W6_mean( log( curvature / its own q1..q4 mean ) )
   fires when score >= threshold_curvature
   ```

   Measured: 3 steps gives 380/564 external at 75 FP, 9 steps gives 388/564 at
   90 FP — equal within noise, fewer false alarms, and a 3x cheaper read.

## Final alarm

```text
v8 = v7_guard
     OR confirmed(front/back flow inversion)
     OR confirmed(three-step curvature)
```

where `confirmed(h)` requires the crossing to hold for **K = 2 consecutive
queries**. Both new heads are compared with `>=` / `<=`, never strict `>` or
`<`: strict comparison silently drops the entire tie group at the threshold
(measured loss: 89.3% for `set_dwell`, 30.1% for `query_top1_churn`).

Earliest possible alarm: q6 for freeze and for both new heads (width-6 causal
mean), q10 for confirmed turbulence.

## Fixed values

```text
frontback quantile        0.005 (lower tail)   threshold  0.000000
curvature quantile        0.995 (upper tail)   threshold  0.389000
confirmation              K = 2 consecutive queries
curvature denoising steps 1, 5, 9
smoothing                 W6, causal
curvature baseline        q1..q4, per episode
```

`threshold_frontback = 0.0` is not hand-set: it is the 0.005 order statistic of
the unlabeled development scores and lands on zero because the head fires
exactly when the front/back ratio crosses 1.

Both thresholds are empirical order statistics of the **unlabeled**
`development_main` reference. Outcomes are read only to score, never to fit.
There are no learned weights, gradients, classifiers, task centroids, or
per-task cutoffs.

## Results

Cap-free protocol: the alarm must leave at least 4 chunks of lead
(`length − alarm_chunk >= 4`). No metric uses the horizon cap.

| cohort | episodes | risks | v7 | **v8** |
|---|---:|---:|---|---|
| development_main | 14,800 | 487 | 303/487, 38 FP | **331/487, 44 FP** |
| external_8b | 15,600 | 564 | 347/564, 57 FP | **377/564, 66 FP** |
| legacy_main16x32 | 2,560 | 307 | 220/307, 16 FP | **224/307, 16 FP** |
| **total** | **32,960** | **1,358** | **870/1,358, 111 FP** | **932/1,358, 126 FP** |

Precision 0.887 → 0.881. Per suite, pooled over all three cohorts:

| suite | risks | v7 | v8 |
|---|---:|---|---|
| goal | 250 | 159/250, 1 FP | 181/250, 10 FP |
| long | 712 | 591/712, 99 FP | 592/712, 105 FP |
| object | 81 | 38/81, 7 FP | 41/81, 7 FP |
| spatial | 315 | 82/315, 4 FP | **118/315, 4 FP** |

The gain is concentrated in the short-horizon suites, which is where a
horizon-length rule ("alarm on everything still running at chunk 37") catches
**0 of 290** by construction.

Lead-time trade, pooled:

| arm | lead≥0 | lead≥2 | lead≥4 | lead≥8 | lead≥12 |
|---|---|---|---|---|---|
| v7 | 1078/172 | 1019/152 | 870/111 | 638/82 | 564/60 |
| v8 | 1128/192 | 1075/168 | 932/126 | 689/90 | 590/61 |

**Null control.** Shuffling both new heads across episodes within each chunk
(preserving the per-chunk marginal, hence the alarm rate) and leaving v7
untouched gives 1,210/1,358 TP at **4,418 FP**. v8 reaches 932/1,358 at 126 FP:
the null needs 35x the false alarms.

## Selection and evaluation

The two heads and their quantiles were chosen on `development_main` by
maximising development union TP subject to development union FP ≤ 50.

**Honesty:** that selection rule was written after an unconstrained external
table for the same candidate heads had already been inspected. The rule is
development-only; the exploration that produced it was not. `external_8b` and
`legacy_main16x32` were each scored once under the frozen rule.
`legacy_main16x32` is the cleanest test: nothing in v7 or v8 was ever fitted on
it, and it uses a different sampling design (16 initial states × 32 noise seeds
versus 50 × 8).

As with v7, this is rule selection, not outcome-blind method discovery. The
external cohort had already been inspected by v3–v7 and is not a pristine
holdout.

## Reproduction

```bash
python moe-v8-0906/experiments/extract_flow_speed.py      # all three cohorts
python moe-v8-0906/experiments/evaluate_full_corpus.py
pytest -q moe-v8-0906/tests
```
