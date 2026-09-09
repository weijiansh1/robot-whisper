# Online Intrinsic Routing Guard v8.2

## Contract

Unchanged from v7 and v8. The runtime monitor receives only the current
`hb_router_probs` tensor, shape `[8, 10, 11, 32]`. No task or suite identity, no
task prototype, no outcome, **no horizon**, no simulator state, no future query,
no peer rollout, no fitted model.

Nothing is trained. The profile is five scalars: v7's three thresholds and one
robust scale, plus v8's two thresholds — all empirical order statistics of an
**unlabeled** reference corpus — and one scalar slope. The rule is boolean.

Entire data contract for everything v8/v8.2 adds over v7: one `[layer, 9]`
array per query,

```text
action_root      = sqrt(p)[:, :, 1:11, :]
flow_speed[l,k]  = ||action_root[l,k+1] − action_root[l,k]|| / sqrt(2),
                   averaged over the ten action tokens,  k = 0..8
```

verified against the published caches to 9.6e-6.

## Heads

v7's three are unchanged: relative freeze, persistent flow acceleration,
persistent recurrence loss. v8 adds two:

**Front/back flow inversion.** The front layers respond to new observation
while the back layers hold a stable relational shape — front/back cross-query
conditional-graph response is 7.35 / 7.41 / 6.38 across the three cohorts, same
direction in 40/40, 39/39 and 5/5 tasks, and the flow-path ratio is
1.34 / 1.33 / 1.34. The head fires when that ordering **inverts**.

```text
score = causal_W6_mean( log( mean_front sum_k flow_speed
                           / mean_back  sum_k flow_speed ) )
```

Not self-baselined. The ratio already normalises across layers; forcing v7's
self-baseline onto it measurably destroys it (+169 TP for +1417 FP versus
+93 TP for +40 FP without). This is the guard's only head without a
self-baseline and the exception is deliberate.

**Three-step curvature.** The denoising axis is rank-1 for every functional
measured (participation ratio 1.00–1.52 of 10), so v7's ten-step curvature is
over-sampled. This head reads **steps 1, 5 and 9 only** — 380/564 external at
75 FP versus 388/564 at 90 FP for nine steps, equal within noise with fewer
false alarms and a 3x cheaper read.

```text
curvature = mean_{L12..L15} | second difference of flow_speed over steps (1,5,9) |
score     = causal_W6_mean( log( curvature / its own q1..q4 mean ) )
```

## What v8.2 changes: a threshold that relaxes with the chunk index

```text
threshold(q) = threshold_0 + slope · q     (sign-flipped for `low` heads,
                                            so slope < 0 always relaxes)
slope = -0.0015
```

Among episodes still running, the fraction that will fail rises from about 0.05
to about 0.44 across the window, so identical evidence is worth more later than
earlier. A constant threshold does not encode that.

Crucially the slope is a **deterministic function of a counter**, so it cannot
be dragged along by the anomaly. Four adaptive alternatives were tested and all
lost, for exactly that reason:

| variant | result |
|---|---|
| running median + MAD of the episode's own past | 935/1358 at **963 FP** (7.6x) |
| expanding self-baseline instead of fixed q1..q4 | 920/1358 at 117 FP (fewer TP) |
| per-chunk z-normalisation against the reference | 933/1358 at **596 FP** (4.7x) |
| chunk-gated per-step mobility head, 5,712 configs | +8 TP / +3 FP |

v7's fixed q1..q4 baseline works because it is a *reference regime* — "compared
with how this episode opened" — not a *current* one.

Three richer shapes for the relaxation were tested at a matched development
budget — `−log S(q)`, `1 − S(q)` and `sqrt(q)`, where S is the label-free
survival fraction — and all landed at 1050/1358 with 177–189 FP against
linear's 1050/1358 with 173 FP. **The shape does not matter, only the amount**,
so the simplest form is kept.

## Fixed values

```text
frontback quantile   0.005 lower tail    threshold  0.000000
curvature quantile   0.995 upper tail    threshold  0.389000
slope                -0.0015
confirmation         K = 2 consecutive queries
curvature steps      1, 5, 9
smoothing            W6, causal          curvature baseline  q1..q4
earliest alarm       q6 (freeze and both new heads), q10 (confirmed turbulence)
comparison           >= / <=, never strict
```

`threshold_frontback = 0.0` is not hand-set: it is the 0.005 order statistic of
the unlabeled development scores, and it lands on zero because the head fires
exactly when the front/back ratio crosses 1.

## Results — 32,960 episodes, 1,358 risks, 31,602 successes

Cap-free: the alarm must leave at least 4 chunks of lead. No metric uses the
horizon.

| arm | lead≥0 | lead≥2 | **lead≥4** | lead≥8 | lead≥12 | lead≥16 | lead≥20 |
|---|---|---|---|---|---|---|---|
| v7 | 1078/172 | 1019/152 | 870/111 | 638/82 | 564/60 | 513/29 | 341/14 |
| v8 | 1128/192 | 1075/168 | 932/126 | 689/90 | 590/61 | 520/30 | 346/14 |
| **v8.2** | **1165/209** | **1104/182** | **966/137** | **715/101** | **610/66** | **528/30** | **352/14** |

**v8.2 dominates v8 at every lead cutoff** — that is a selection constraint, not
an observation, so aggregate recall cannot be bought by giving up early
detections.

At lead ≥ 4: recall **966/1358 = 0.711**, **137 false alarms = 4.3 per 1,000
successful episodes**, precision **0.876**. Against v8: **+34 TP for +11 FP**,
exchange 3.1.

| cohort | v8 | v8.2 |
|---|---|---|
| development_main | 331/487, 44 FP | 345/487, 47 FP |
| external_8b | 377/564, 66 FP | 396/564, 73 FP |
| legacy_main16x32 | 224/307, 16 FP | 225/307, 17 FP |

| suite | v7 | v8 | v8.2 |
|---|---|---|---|
| libero_goal | 159/250, 1 FP | 181/250, 10 FP | **187/250, 13 FP** |
| libero_long | 591/712, 99 FP | 592/712, 105 FP | **597/712, 112 FP** |
| libero_object | 38/81, 7 FP | 41/81, 7 FP | **49/81, 7 FP** |
| libero_spatial | 82/315, 4 FP | 118/315, 4 FP | **133/315, 5 FP** |

The gain concentrates in the short-horizon suites, which is where a
horizon-length rule ("alarm on everything still running at chunk 37") catches
**0 of 290** by construction.

**Null control.** Shuffling both new heads across episodes within each chunk —
preserving the per-chunk marginal, hence the alarm rate — and leaving v7
untouched: 1,225/1358 TP at **4,565 FP**. v8.2 reaches 966/1358 at 137 FP; the
null needs 33x the false alarms.

## Selection and evaluation

Grid: baseline ∈ {2,3,4}, width ∈ {3,4,6}, confirm ∈ {2,3},
slope ∈ {−0.0010, −0.0015, −0.0020, −0.0025}, 72 configurations. Rule declared
before scoring the sealed cohorts: on `development_main` only, maximise TP at
lead ≥ 4 subject to (a) development false alarms ≤ 47 and (b) dominating v8 at
every lead cutoff. Twelve configurations qualified. `external_8b` and
`legacy_main16x32` were each scored once.

**Selection is at lead ≥ 4, never at a long lead.** Risk episodes run to the cap
and successes do not, so a long-lead filter removes false alarms far faster than
true ones. A previous attempt produced "222/487 TP with 1 FP at lead ≥ 12" that
was really 156 false alarms, 154 of which the filter had removed.

**Honesty.** The grid and the selection rule were written after unconstrained
external tables for related candidates had been inspected during exploration.
The rule is development-only; the exploration that produced it was not. As with
v7, this is rule selection, not outcome-blind method discovery, and `external_8b`
was already inspected by v3–v8 so it is not a pristine holdout.
**`legacy_main16x32` is the only clean cohort** — nothing in v7, v8 or v8.2 was
ever fitted on it and it uses a different sampling design (16 initial states ×
32 noise seeds versus 50 × 8). Its 225/307 with 17 FP is the most trustworthy
single number here, and its gain over v8 is the smallest (+1/+1).

## Reproduction

```bash
python moe-v8-0906/experiments/extract_flow_speed.py
python moe-v8-0906/experiments/evaluate_full_corpus.py   # v8 anchors
python moe-v8-0906/experiments/freeze_v82.py
pytest -q moe-v8-0906/tests
```
