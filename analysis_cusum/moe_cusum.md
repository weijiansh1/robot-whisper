# Sequential (CUSUM / SPRT) detector on the MoE routing signal

Run started 2026-08-29 18:34.  Code in
`/home/jovyan/work/himoe-vla/analysis_cusum/`, rows in `moe_cusum.csv`.

## 0. Score and baseline verification

Per-step score = Hellinger between consecutive control steps, per (layer, token)
cell then averaged over the 40 cells (back-block HB layers 12-15, action tokens
1-10, denoise 9).  This is the V2 divergence-first placement of
`analysis_agg_axis/lib.py` `hell|prev`.  No hard top-4 expert IDs anywhere.

Reproduction check, within-group paired AUC of the negated 8-step window mean:

| t | corpus A | corpus B |
|---|---|---|
| 25 | 0.6345 | 0.5249 |
| 30 | **0.7988** | **0.7510** |

Published convention values are A 0.799 / B 0.751 at t=30 -> exact match.
Direction confirmed: eventual failure goes with a LOW routing change rate.

`/tmp/soft.pkl` does NOT hold this score.  Its `M` is the cell-mean
Bhattacharyya COEFFICIENT (max abs difference from a recomputed cell-mean BC =
1.8e-5), so `sqrt(1 - M[i,i+1])` is the similarity-first ordering, not the
divergence-first V2 convention.  It gives AUC 0.792 / 0.751 and reproduces the
published false-alarm rate exactly (A 16/117 = 13.68%, median phase 0.604) but
not the published detection rate (65.5% vs 70.6%).  Everything below is
rebuilt from the zarr slices (`analysis_stat_axis/{A,B}_main.npy`) with the
V2 score; soft.pkl is not used.

Fixed rule as published (W=8 mean, thr=0.0348, K=3, t in [8,35)), V2 score:

| corpus | FA | det | median phase | median t |
|---|---|---|---|---|
| A | 0.1795 (21/117) | 0.7106 (167/235) | 0.592 | 29 |
| B | 0.0608 (18/296) | 0.7454 (161/216) | 0.615 | 32 |

vs the quoted A 13.7%/70.6%, B 5.1%/74.1%.  Detection matches to within one
branch; the false-alarm rates differ by 3-5 branches because of the placement
ordering.  All comparisons below therefore re-calibrate BOTH rules to the same
nominal false-alarm rate under the same leave-one-group-out protocol, and the
literal thr=0.0348 point is reported alongside.

## 1. Sentinels

| corpus | score | pipeline | rule | peak AUC | FA | det | med phase | med t |
|---|---|---|---|---|---|---|---|---|
| A | real | TS | cusum | 0.596 | 0.214 | 0.664 | 0.347 | 17.5 |
| A | real | TH | cusum | 0.696 | 0.239 | 0.736 | 0.417 | 20.0 |
| A | real | TS_sprt | sprt | 0.539 | 0.188 | 0.562 | 0.250 | 13.0 |
| A | real | TH_sprt | sprt | 0.602 | 0.205 | 0.562 | 0.235 | 12.0 |
| A | shuffle11 | TS | cusum | 0.698 | 0.137 | 0.506 | 0.255 | 13.0 |
| A | shuffle11 | TH | cusum | 0.702 | 0.137 | 0.511 | 0.250 | 13.0 |
| A | shuffle11 | TS_sprt | sprt | 0.710 | 0.154 | 0.498 | 0.212 | 11.0 |
| A | shuffle11 | TH_sprt | sprt | 0.703 | 0.120 | 0.468 | 0.184 | 9.0 |
| A | shuffle12 | TS | cusum | 0.744 | 0.145 | 0.502 | 0.238 | 12.0 |
| A | shuffle12 | TH | cusum | 0.751 | 0.171 | 0.536 | 0.250 | 12.0 |
| A | shuffle12 | TS_sprt | sprt | 0.783 | 0.188 | 0.464 | 0.231 | 11.0 |
| A | shuffle12 | TH_sprt | sprt | 0.782 | 0.188 | 0.464 | 0.216 | 11.0 |
| A | shuffle13 | TS | cusum | 0.717 | 0.137 | 0.515 | 0.245 | 12.0 |
| A | shuffle13 | TH | cusum | 0.698 | 0.120 | 0.566 | 0.260 | 13.0 |
| A | shuffle13 | TS_sprt | sprt | 0.770 | 0.120 | 0.587 | 0.167 | 8.0 |
| A | shuffle13 | TH_sprt | sprt | 0.766 | 0.128 | 0.545 | 0.186 | 9.0 |
| A | clock | TS | cusum | 0.500 | 0.000 | 0.000 | nan | nan |
| A | clock | TH | cusum | 0.500 | 0.214 | 0.302 | 0.224 | 11.0 |
| A | clock | TS_sprt | sprt | 0.500 | 0.000 | 0.000 | nan | nan |
| A | clock | TH_sprt | sprt | 0.500 | 0.214 | 0.302 | 0.694 | 34.0 |
| A | clock_oracle | direct | t>=h | 0.512 | 0.000 | 0.000 | nan | nan |
| B | real | TS | cusum | 0.665 | 0.068 | 0.269 | 0.558 | 29.0 |
| B | real | TH | cusum | 0.731 | 0.061 | 0.625 | 0.615 | 32.0 |
| B | real | TS_sprt | sprt | 0.605 | 0.068 | 0.255 | 0.558 | 29.0 |
| B | real | TH_sprt | sprt | 0.722 | 0.051 | 0.597 | 0.615 | 32.0 |
| B | shuffle11 | TS | cusum | 0.790 | 0.051 | 0.648 | 0.202 | 10.5 |
| B | shuffle11 | TH | cusum | 0.787 | 0.054 | 0.657 | 0.212 | 11.0 |
| B | shuffle11 | TS_sprt | sprt | 0.774 | 0.054 | 0.671 | 0.154 | 8.0 |
| B | shuffle11 | TH_sprt | sprt | 0.780 | 0.051 | 0.671 | 0.154 | 8.0 |
| B | shuffle12 | TS | cusum | 0.789 | 0.051 | 0.694 | 0.173 | 9.0 |
| B | shuffle12 | TH | cusum | 0.792 | 0.051 | 0.694 | 0.173 | 9.0 |
| B | shuffle12 | TS_sprt | sprt | 0.755 | 0.051 | 0.685 | 0.154 | 8.0 |
| B | shuffle12 | TH_sprt | sprt | 0.754 | 0.054 | 0.671 | 0.154 | 8.0 |
| B | shuffle13 | TS | cusum | 0.790 | 0.054 | 0.671 | 0.192 | 10.0 |
| B | shuffle13 | TH | cusum | 0.787 | 0.054 | 0.681 | 0.173 | 9.0 |
| B | shuffle13 | TS_sprt | sprt | 0.782 | 0.054 | 0.681 | 0.154 | 8.0 |
| B | shuffle13 | TH_sprt | sprt | 0.784 | 0.051 | 0.676 | 0.154 | 8.0 |
| B | clock | TS | cusum | 0.500 | 0.000 | 0.000 | nan | nan |
| B | clock | TH | cusum | 0.500 | 0.078 | 0.042 | 0.596 | 31.0 |
| B | clock | TS_sprt | sprt | 0.500 | 0.000 | 0.000 | nan | nan |
| B | clock | TH_sprt | sprt | 0.500 | 0.095 | 0.167 | 0.154 | 8.0 |
| B | clock_oracle | direct | t>=h | 0.500 | 0.000 | 0.000 | nan | nan |

## 5. Serial correlation and bound calibration

| corpus | whiten | acf1(succ) | nominal alpha | achieved FA (Wald bound) | achieved FA (empirical LOGO) |
|---|---|---|---|---|---|
| A | none | 0.522 | 0.020 | 0.162 | 0.103 |
| A | none | 0.522 | 0.050 | 0.248 | 0.137 |
| A | none | 0.522 | 0.100 | 0.316 | 0.145 |
| A | none | 0.522 | 0.137 | 0.342 | 0.154 |
| A | none | 0.522 | 0.200 | 0.402 | 0.179 |
| A | ar1 | 0.211 | 0.020 | 0.103 | 0.137 |
| A | ar1 | 0.211 | 0.050 | 0.120 | 0.137 |
| A | ar1 | 0.211 | 0.100 | 0.179 | 0.145 |
| A | ar1 | 0.211 | 0.137 | 0.214 | 0.162 |
| A | ar1 | 0.211 | 0.200 | 0.274 | 0.171 |
| B | none | 0.756 | 0.020 | 0.186 | 0.024 |
| B | none | 0.756 | 0.050 | 0.226 | 0.057 |
| B | none | 0.756 | 0.100 | 0.294 | 0.111 |
| B | none | 0.756 | 0.137 | 0.345 | 0.159 |
| B | none | 0.756 | 0.200 | 0.412 | 0.206 |
| B | ar1 | 0.425 | 0.020 | 0.105 | 0.020 |
| B | ar1 | 0.425 | 0.050 | 0.152 | 0.057 |
| B | ar1 | 0.425 | 0.100 | 0.220 | 0.098 |
| B | ar1 | 0.425 | 0.137 | 0.257 | 0.145 |
| B | ar1 | 0.425 | 0.200 | 0.345 | 0.220 |
corpus A: acf1(z|succ) = 0.514, fitted AR(1) phi = 0.473
corpus B: acf1(z|succ) = 0.427, fitted AR(1) phi = 0.419

## 2. Matched-false-alarm comparison

Config grid: 88 sequential configurations shared by both corpora.
Spearman rank correlation of detection-at-matched-FA between corpus A and corpus B across configs: **-0.046**; of median alarm phase: **0.371**.
Best config on A: {'method': 'sequential', 'W': np.int64(1), 'rule': 'cusum', 'dens': 'gauss', 'whiten': 'none', 'tstart': np.int64(16)} -> A det 0.877, but on B det 0.353.
Best config on B: {'method': 'sequential', 'W': np.int64(1), 'rule': 'cusum', 'dens': 'tvd', 'whiten': 'none', 'tstart': np.int64(20)} -> B det 0.750, but on A det 0.547.

| corpus | detector | FA | det | phase p25 | **phase p50** | phase p75 | t p50 | ARL0 | ARL1 | modes | dip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | seq_cusum_primary | 0.137 | 0.500 | 0.427 | **0.555** | 0.611 | 27.5 | 183.4 | 46.2 | 0.06 | 0.35 |
| A | seq_sprt_primary | 0.137 | 0.269 | 0.330 | **0.520** | 0.632 | 25.5 | 178.4 | 91.2 | 0.46 | 0.59 |
| A | seq_cusum_W8 | 0.137 | 0.248 | 0.301 | **0.547** | 0.654 | 27.9 | 178.3 | 98.6 | 0.97 | 0.61 |
| A | seq_xcorpus | 0.137 | 0.547 | 0.516 | **0.581** | 0.624 | 29.4 | 192.6 | 43.8 | 0.26 | 0.29 |
| A | fixed_W8K3 | 0.137 | 0.572 | 0.391 | **0.587** | 0.637 | 28.9 | 207.3 | 40.2 | 0.83 | 0.34 |
| A | fixed_W12K3 | 0.137 | 0.476 | 0.519 | **0.620** | 0.654 | 30.3 | 185.0 | 50.2 | 0.90 | 0.32 |
| B | seq_cusum_primary | 0.051 | 0.493 | 0.614 | **0.615** | 0.635 | 32.0 | 525.5 | 53.0 | 0.36 | nan |
| B | seq_sprt_primary | 0.051 | 0.504 | 0.589 | **0.608** | 0.632 | 31.6 | 529.5 | 52.0 | 0.35 | nan |
| B | seq_cusum_W8 | 0.051 | 0.448 | 0.594 | **0.624** | 0.643 | 32.5 | 524.8 | 59.4 | 0.31 | nan |
| B | seq_xcorpus | 0.051 | 0.353 | 0.528 | **0.568** | 0.620 | 29.5 | 531.2 | 72.3 | 0.06 | nan |
| B | fixed_W8K3 | 0.051 | 0.698 | 0.586 | **0.615** | 0.635 | 32.0 | 545.3 | 35.3 | 1.00 | nan |
| B | fixed_W12K3 | 0.051 | 0.671 | 0.615 | **0.635** | 0.654 | 33.0 | 533.7 | 38.9 | 1.00 | nan |
# Sequential (CUSUM / SPRT) detector on the MoE routing signal — result

Code and CSVs: `/home/jovyan/work/himoe-vla/analysis_cusum/` (mirrored from `/tmp`).
Row-level output: `moe_cusum.csv`, `sweep_full.csv`, `sweep_tvd.csv`, `final.csv`,
`sentinel2.csv`, `accum.csv`, `breadth2.csv`, `calibration.csv`, `headline.csv`.

**Verdict: the sequential test does not beat the fixed-threshold rule, and the
alarm phase does not move earlier. This is the negative branch the brief
anticipated, and the reason is measurable: the discriminating information is
concentrated in a short late window (t >= 26), not spread thinly over the
prefix, so there is almost nothing to accumulate.**

---

## 0. Score, and what reproduces

Per-step score = Hellinger between consecutive control steps, per (layer, token)
cell on the 32-expert vector then averaged over the 40 cells (back-block HB
layers 12–15, action tokens 1–10, denoise 9) — the V2 divergence-first
placement of `analysis_agg_axis/lib.py` `hell|prev`. No hard top-4 IDs.

Within-group paired AUC of the negated 8-step window mean: **t=30 A 0.7988 /
B 0.7510** — an exact match to the documented convention values 0.799 / 0.751.
t=25: A 0.6345, B 0.5249. Failure goes with a **low** change rate.

`/tmp/soft.pkl` is **not** this score. Its `M` is the cell-mean Bhattacharyya
*coefficient* (max |diff| vs a recomputed cell-mean BC = 1.8e-5), so
`sqrt(1-M[i,i+1])` is the similarity-first ordering. It gives AUC 0.792/0.751,
reproduces the published false-alarm rate exactly (A 16/117 = 13.68%, median
phase 0.604) but not the published detection rate (65.5% vs 70.6%). Rebuilt
from the zarr slices instead; soft.pkl is not used anywhere below.

Fixed rule as published (W=8 mean, thr=0.0348, K=3, t in [8,35)) on the V2 score:
A FA 0.179 / det 0.711 / median phase 0.592; B FA 0.061 / det 0.745 / phase 0.615
(published: A 0.137/0.706, B 0.051/0.741, phase ~0.60). Detection matches to one
branch; the FA differs by 3–5 branches because of the placement ordering.
Every comparison below therefore re-calibrates **both** rules under the same
leave-one-group-out protocol to the same false-alarm rate.

**Corpus A group structure is severe and matters for calibration**: worker
success rates are 26% / **0%** / 6% / 91%. Worker 1 contributes zero successes,
so LOGO bound calibration in A is fitted on very few, very heterogeneous
successes.

---

## 1. Sentinels

### Shuffle sentinel (chunk order permuted inside each branch, whole pipeline rerun)

Peak causal statistic, within-group paired AUC (3 permutation seeds):

| corpus | detector | real | shuffle x3 | verdict |
|---|---|---|---|---|
| A | fixed_W8K3 | **0.730** | 0.588 / 0.659 / 0.638 | real above ceiling |
| A | seq_cusum (primary) | 0.622 | 0.690 / 0.728 / 0.731 | **real below** |
| A | seq_sprt (primary) | 0.534 | 0.687 / 0.775 / 0.772 | **real far below** |
| A | seq_xcorpus | 0.715 | 0.725 / 0.638 / 0.641 | borderline |
| B | fixed_W8K3 | **0.732** | 0.618 / 0.647 / 0.642 | real above ceiling |
| B | seq_cusum (primary) | 0.756 | 0.784 / 0.792 / 0.790 | **real below all three** |
| B | seq_sprt (primary) | 0.765 | 0.773 / 0.755 / 0.779 | indistinguishable |

Detection at matched FA tells the same story more bluntly:

| corpus | detector | real | shuffle x3 |
|---|---|---|---|
| A | fixed_W8K3 | **0.549** | 0.260 / 0.319 / 0.302 |
| A | seq_cusum | 0.477 | 0.451 / 0.447 / 0.489 |
| B | fixed_W8K3 | **0.718** | 0.560 / 0.579 / 0.588 |
| B | seq_cusum | 0.486 | 0.639 / 0.699 / 0.653 |

The causal pipeline never exceeds the leakage ceiling, so there is no leak. But
the pre-specified **sequential** detector performs the same on order-destroyed
data (A) or *worse* than on order-destroyed data (B). A cumulative sum is
order-invariant by construction; the only thing shuffling changes inside a
truncated window is *which* chunks land in it, and pulling late chunks forward
helps. The fixed rule, by contrast, is clearly damaged by shuffling — it is
genuinely reading the temporal arrangement.

The ceiling here is 0.59–0.79, not the 0.93–0.96 quoted for whole-episode
pipelines, because the detection window truncates at t=35: shuffling can only
import chunks from the same episode into the first 34 slots.

### Clock sentinel (score = pure function of t, nothing else)

| corpus | detector | peak AUC | FA | det |
|---|---|---|---|---|
| A / B | every detector above | **0.500** | 0.000 | 0.000 |

Fully protected — but *not* by construction, and it is worth saying why. The
window [8,35) closes before any branch dies (A T_min=33, B T_min=35), so
survival carries no information inside it; and the per-index standardisation
(`mu_t`, `sd_t` fitted on training folds) removes the drift. Remove the
standardisation and run the *identical* machinery time-homogeneously, and the
pure clock scores: **A CUSUM FA 0.214 / det 0.302; A SPRT FA 0.214 / det 0.302;
B SPRT FA 0.095 / det 0.167**. So the cumulative statistic *can* drift into a
clock, exactly as the brief warned. All reported results use the time-varying
form. Both were run; see `moe_cusum.csv` section `sentinel`.

---

## 2. Matched-false-alarm comparison

Both rules calibrated identically: the bound (sequential) or the threshold
(fixed) is chosen on the **training folds' success branches** to hit the nominal
FA, then applied unchanged to the held-out group. Group = worker (A, 4) /
init_state_id (B, 16). No group centring anywhere. Alarms in t in [8,35).

Config selection matters and does not transfer. Across the **88 sequential
configurations** run on both corpora (W in {1,2,4,8} x {CUSUM, SPRT} x
{Gaussian, KDE, time-varying Gaussian} x {no whitening, AR(1)} x accumulation
start in {1,8,16,20}), the Spearman rank correlation of detection-at-matched-FA
between corpus A and corpus B is **-0.046** (phase: 0.371). The best config on A
(det 0.877) gets 0.353 on B; the best on B (0.750) gets 0.574 on A. Numbers
below are therefore reported for a **pre-specified** config (W=1 raw increment,
CUSUM/SPRT, time-varying class Gaussians, AR(1) innovations, accumulate from
t=1) and for the config **selected on the other corpus** (honest transfer), with
the same-corpus best kept only as an explicitly selection-optimistic ceiling.

### Corpus A (target FA 0.137)

| detector | FA | det | phase p25 | **phase p50** | phase p75 | t p50 | ARL0 | ARL1 |
|---|---|---|---|---|---|---|---|---|
| fixed W8 K3 (LOGO) | 0.162 | 0.549 | 0.451 | **0.588** | 0.638 | 29 | 151 | 41 |
| fixed literal 0.0348 | 0.179 | 0.711 | 0.335 | **0.592** | 0.620 | 29 | 137 | 29 |
| seq CUSUM (pre-specified) | 0.128 | 0.477 | 0.436 | **0.576** | 0.617 | 28 | 195 | 49 |
| seq SPRT (pre-specified) | 0.137 | 0.217 | 0.317 | **0.500** | 0.629 | 24 | 180 | 114 |
| seq, config from corpus B | 0.154 | 0.574 | 0.505 | **0.577** | 0.620 | 29 | 165 | 41 |
| *seq, best in-sample (ceiling)* | *0.128* | *0.872* | *0.385* | *0.500* | *0.569* | *24* | *199* | *21* |

### Corpus B (target FA 0.051)

| detector | FA | det | phase p25 | **phase p50** | phase p75 | t p50 | ARL0 | ARL1 |
|---|---|---|---|---|---|---|---|---|
| fixed W8 K3 (LOGO) | 0.044 | 0.718 | 0.577 | **0.615** | 0.635 | 32 | 608 | 34 |
| fixed literal 0.0348 | 0.061 | 0.745 | 0.577 | **0.615** | 0.635 | 32 | 435 | 32 |
| seq CUSUM (pre-specified) | 0.051 | 0.486 | 0.615 | **0.615** | 0.635 | 32 | 529 | 54 |
| seq SPRT (pre-specified) | 0.054 | 0.509 | 0.577 | **0.596** | 0.635 | 31 | 495 | 50 |
| seq, config from corpus A | 0.047 | 0.347 | 0.538 | **0.558** | 0.615 | 29 | 563 | 73 |
| *seq, best in-sample (ceiling)* | *0.051* | *0.750* | *0.558* | *0.596* | *0.615* | *31* | *530* | *33* |

### Failure-class breakout (corpus A; corpus B has no physical labels)

Classes are exhaustive and disjoint over the 235 A failures: stagnation 49,
loop-not-stagnation 150, non-stagnation-non-loop 36.

| detector | FA | stagnation (49) | loop (150) | other (36) |
|---|---|---|---|---|
| fixed W8 K3 (LOGO) | 0.162 | 0.796 | 0.433 | 0.694 |
| fixed literal 0.0348 | 0.179 | 0.857 | 0.660 | 0.722 |
| seq CUSUM | 0.128 | **0.898** | 0.313 | 0.583 |
| seq SPRT | 0.137 | 0.408 | 0.133 | 0.306 |
| seq, config from B | 0.154 | **0.939** | 0.407 | 0.778 |

The one place the sequential rule wins is **stagnation**: 0.90–0.94 vs 0.80 for
the fixed rule at a *lower* false-alarm rate. Stagnation is the failure mode
whose routing signature is a sustained low change rate, i.e. the one mode where
weak evidence really is spread over time. It loses badly on loop/cycling
(0.31–0.41 vs 0.43), which is the majority class.

---

## 3. Alarm-phase and alarm-index distributions

Phase = alarm index / branch length, over failures that alarmed.

| corpus | detector | p10 | p25 | **p50** | p75 | p90 |
|---|---|---|---|---|---|---|
| A | fixed W8 K3 | 0.216 | 0.451 | **0.588** | 0.638 | 0.660 |
| A | seq CUSUM | 0.392 | 0.436 | **0.576** | 0.617 | 0.660 |
| A | seq SPRT | 0.280 | 0.317 | **0.500** | 0.629 | 0.653 |
| A | seq (config from B) | 0.412 | 0.505 | **0.577** | 0.620 | 0.640 |
| B | fixed W8 K3 | 0.558 | 0.577 | **0.615** | 0.635 | 0.635 |
| B | seq CUSUM | 0.596 | 0.615 | **0.615** | 0.635 | 0.654 |
| B | seq SPRT | 0.558 | 0.577 | **0.596** | 0.635 | 0.654 |

**The headline number does not move.** A: 0.588 -> 0.576 (CUSUM) or 0.577
(transferred config), a shift of ~1 pp. B: 0.615 -> 0.615, no shift at all. The
one detector that alarms genuinely earlier — A SPRT at median phase 0.500 — pays
for it with detection 0.217 vs 0.549. That is the ordinary speed/power trade,
not extra information: it alarms on less evidence.

Note the sequential rule *compresses* the phase distribution rather than
shifting it. The fixed rule's p10 is 0.216–0.235 (it has a real early tail); the
CUSUM's p10 is 0.392. The sequential rule removes early alarms rather than
adding them.

### Alarm-index histograms and bimodality

Counts of alarming failures by alarm index bin:

| corpus | detector | 8–12 | 12–16 | 16–20 | 20–24 | 24–28 | 28–32 | 32–35 | modes | dip |
|---|---|---|---|---|---|---|---|---|---|---|
| A | fixed literal 0.0348 | 13 | 29 | 0 | 6 | 9 | 79 | 31 | 3 | **0.94** |
| A | fixed W8 K3 (LOGO) | 17 | 12 | 1 | 4 | 15 | 48 | 32 | 2 | **0.88** |
| A | seq CUSUM | 0 | 6 | 5 | 20 | 21 | 35 | 25 | 3 | **0.04** |
| A | seq (config from B) | 0 | 0 | 0 | 31 | 23 | 56 | 25 | 2 | 0.28 |
| B | fixed W8 K3 | 0 | 11 | 2 | 0 | 0 | 52 | 90 | 3 | **1.00** |
| B | seq CUSUM | 0 | 0 | 0 | 0 | 0 | 26 | 79 | 1 | — |

The fixed rule's bimodality reproduces exactly as described: two peaks near
t=12–16 and t=28–32 with an essentially empty gap (dip 0.88–1.00, i.e. the
density between the peaks falls to ~5% of the lower peak). **The sequential rule
removes the bimodality** — the CUSUM's alarm index is unimodal-to-flat with dip
0.04, filling t=20–28. This is a real qualitative change, but it comes from
deleting the early mode, not from moving mass earlier.

---

## 4. Delay relative to the frozen physical loop onset (corpus A)

Onsets exist for 172/352 branches (150/150 loop, 15/49 stagnation, 5/36 other,
2/117 success). Distribution of `alarm_t - onset_query` over alarming failures
with a finite onset:

| detector | n | %early | %late | p10 | p25 | **p50** | p75 | p90 |
|---|---|---|---|---|---|---|---|---|
| fixed literal 0.0348 | 116 | 41% | 59% | -14 | -12 | **+8** | +9 | +9 |
| fixed W8 K3 (LOGO) | 79 | 35% | 65% | -12 | -10 | **+8** | +10 | +10 |
| seq CUSUM | 62 | 31% | 66% | -9 | -2 | **+5.5** | +9 | +10 |
| seq SPRT | 33 | **58%** | 42% | -8 | -5 | **-2** | +9 | +12 |
| seq (config from B) | 77 | 30% | 70% | -3 | -2 | **+7** | +8 | +10 |

The published 34% early / 66% late reproduces (fixed W8 K3: 35% / 65%). The
delay is strongly **bimodal, not centred**: the fixed rule is either ~10–12
queries early or ~8–10 late, with little mass in between; the CUSUM narrows the
early arm (p25 goes -10 -> -2) without moving the late arm much. Median delay
improves from +8 to +5.5 queries — real but small. Only SPRT is median-early
(-2 queries), at detection 0.217.

**`cls_other` caveat**: the "100% early, median 13 queries before the physical
predicate" statement rests on **n = 1 to 3 branches**. Only 5 of the 36
non-stagnation-non-loop failures have a loop onset at all, and their median
onset is query 42 — outside the detection window. Any alarm inside [8,35) is
early by construction. This cell should not be quoted as a finding.

---

## 5. ARL0 / ARL1

Censoring-aware hazard estimate: (total steps observed in the window, truncated
at the alarm) / (number of alarms), i.e. mean steps between false alarms in a
stream of successes (ARL0) and mean steps to detection in a stream of failures
(ARL1). `arl0_expo` / `arl1_expo` in `final.csv` give the exposure.

| corpus | detector | ARL0 | ARL1 | ARL0/ARL1 |
|---|---|---|---|---|
| A | fixed W8 K3 | 151 | 41 | 3.7 |
| A | seq CUSUM | 195 | 49 | 4.0 |
| A | seq SPRT | 180 | 114 | 1.6 |
| A | seq (config from B) | 165 | 41 | 4.1 |
| B | fixed W8 K3 | 608 | 34 | 18.1 |
| B | seq CUSUM | 529 | 54 | 9.8 |
| B | seq SPRT | 495 | 50 | 9.9 |

On the ratio, corpus A is a near-tie (3.7 vs 4.0–4.1) and corpus B is a clear
loss for the sequential rule (18.1 vs 9.8). Mean alarm index among alarming
successes is 19.0 (fixed) vs 19.5 (CUSUM) in A — the sequential rule's false
alarms are not later either.

---

## 6. Serial correlation: was the correction needed?

Lag-1 autocorrelation of the per-index-standardised score, success class:
**A 0.514, B 0.427**. Fitted AR(1) coefficients (success branches, training
folds only): **A phi=0.473, B phi=0.419**. So yes, the score is strongly
autocorrelated.

Achieved false-alarm rate under three bounds at the same nominal alpha
(W=1, time-varying densities, SPRT):

| corpus | whitening | acf1 of the LLR series (succ) | nominal 0.05 -> achieved, Wald bound | nominal 0.05 -> achieved, empirical LOGO |
|---|---|---|---|---|
| A | none | 0.522 | **0.248** (5.0x) | 0.137 |
| A | AR(1) | 0.211 | **0.120** (2.4x) | 0.137 |
| B | none | 0.756 | **0.226** (4.5x) | 0.057 |
| B | AR(1) | 0.425 | **0.152** (3.0x) | 0.057 |

Full grid in `calibration.csv`.

**Conclusions.** (i) The i.i.d. Wald bound `log((1-beta)/alpha)` is badly
optimistic — 4.5–5x too many false alarms unwhitened. (ii) AR(1) whitening
halves the LLR-series autocorrelation (A 0.52 -> 0.21, B 0.76 -> 0.43) and
halves the inflation (5.0x -> 2.4x, 4.5x -> 3.0x) **but does not fix it**;
residual dependence and non-Gaussian tails remain. (iii) Only the **empirical
leave-one-group-out calibration** produces a usable rate, and it works in B
(nominal 0.02/0.05/0.10/0.137/0.20 -> achieved 0.024/0.057/0.111/0.159/0.206)
but **fails in A at small alpha** (nominal 0.02 -> achieved 0.103, 0.05 ->
0.137). That failure is not serial correlation: it is that corpus A has four
groups with success rates 26/0/6/91%, so the held-out group's success
distribution is not the training folds'. All headline comparisons are therefore
matched on the **achieved** out-of-fold FA, for both rules alike.

---

## 7. Is the t=25 information accumulable, or concentrated?

**Concentrated — in time, not in branches.**

### It is not spread over the prefix

Within-group paired AUC at each index t, causal statistics only (`final.csv`,
section `perstep`). Corpus A:

| t | single step | W=2 | W=4 | W=8 | W=12 | mean of the whole prefix 1..t |
|---|---|---|---|---|---|---|
| 12 | 0.419 | 0.480 | 0.501 | 0.404 | 0.429 | 0.429 |
| 16 | 0.499 | 0.470 | 0.448 | 0.446 | 0.401 | 0.389 |
| 20 | 0.557 | 0.544 | 0.509 | 0.406 | 0.471 | 0.418 |
| 25 | 0.607 | 0.646 | 0.669 | 0.635 | 0.655 | **0.530** |
| 30 | 0.714 | 0.757 | 0.780 | 0.799 | 0.807 | 0.743 |

A **single step** at t=25 already carries 0.607 of the 0.635 the 8-step window
carries. Averaging 2–4 steps adds ~0.03. Averaging the **entire prefix** from
t=1 destroys it (0.530). Everything before t≈21 is at or below chance
within-group (0.37–0.56) — and below-chance means the pooled early direction
*reverses* within group, so early steps do not merely add noise, they add
wrongly-signed evidence out of fold. Corpus B is worse: chance from t=21 to
t=27, with the entire signal appearing at t>=28.

### The evidence trajectory confirms it

Mean per-step out-of-fold LLR by class (`breadth2.csv`, section
`evidence_traj`). Summed class separation `E[llr|fail] - E[llr|succ]`:

| corpus | t = 8..25 (18 steps) | t >= 26 (9 steps) | early share |
|---|---|---|---|
| A | 0.738 nats | 2.670 nats | **22%** |
| B | 0.191 nats | 8.300 nats | **2%** |

In A there is a small genuinely accumulable component (~0.04 nats/step from
t≈14), but it is worth about 0.7 nats total against the 2.7 nats delivered by
the last nine steps, and buying it costs the variance of 18 extra noisy terms.
In B there is essentially nothing before t=27.

### But it is *not* a property of a few branches

At t=30, restricted to groups with >=8 successes and >=8 failures, the fraction
of failures below their own group's success median is **A 92.5% (W=8) / 81.3%
(W=1)** and **B 84.2% / 89.5%**; per-group AUC ranges 0.85–0.92 (A) and
0.55–0.93 (B). The late separation is broad across branches.

So the brief's dichotomy resolves as a third option: the evidence is **weak
evidence spread over many branches but confined to a short late window**. A
sequential test cannot help with that, because there is no long run of weakly
informative steps to sum — and the fixed rule's 8-step window is already about
the right integration length for the window that exists. The median phase of
0.60 is not "waiting for the threshold"; it is waiting for evidence that has not
arrived yet.

---

## 8. What would and would not be honest to claim

Supported:

- At matched out-of-fold false-alarm rate, a properly specified sequential test
  (per-index standardisation, AR(1) innovations, time-varying class densities,
  empirically calibrated bound) is a **wash on corpus A** (det 0.574 vs 0.549 at
  FA 0.154 vs 0.162) and **clearly worse on corpus B** (0.347 vs 0.718) when its
  configuration is chosen honestly on the other corpus.
- **Median alarm phase does not improve**: A 0.588 -> 0.576/0.577, B 0.615 ->
  0.615.
- The sequential rule does change the *shape* of the alarm distribution: it
  removes the fixed rule's early mode (dip 0.88–1.00 -> 0.04) and narrows the
  early arm of the delay-to-onset distribution (p25 -10 -> -2).
- One real sub-result: on **stagnation** failures the sequential rule detects
  0.90–0.94 vs 0.80 for the fixed rule at a lower FA. That is the failure mode
  with a sustained signature.
- The i.i.d. Wald bound inflates the false-alarm rate 4.5–5x; AR(1) whitening
  halves it; only empirical calibration is usable, and it still breaks in
  corpus A because of extreme group heterogeneity.

Not supported:

- Any claim from a single best configuration. Cross-corpus rank correlation of
  detection across 88 configurations is **-0.046**; the in-sample best (A det
  0.872, B det 0.750) is selection, not signal.
- Any claim that the sequential rule alarms earlier.
- The `cls_other` early-alarm result at n=1–3.
