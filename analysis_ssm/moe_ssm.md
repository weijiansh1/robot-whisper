# Linear-Gaussian state-space (Kalman) model of MoE routing, action chunk as control input

Run 2026-08-29. Scripts in `analysis_ssm/`; raw appended log in `moe_ssm_log.md`;
long-format results in `moe_ssm.csv` / `moe_ssm_pivot.csv` / `moe_ssm_probe.csv`
(mirrored to `/tmp`).

## Model actually fitted

    y_k = clr(R_k)                 88 (layer, token) sites x 32 experts, denoise step 9
    x_k = PCA_r(y_k)               whitened, basis fit on TRAIN groups only
    z_{k+1} = F z_k + B u_k + b + w_k     u_k = flatten(executed 10x7 action chunk)
    x_k     = W z_k + v_k
    e_k = x_k - W z_{k|k-1}        innovation;  NIS_k = e_k' S_k^{-1} e_k

Fit by EM (log-likelihood monotone in every fold x config, both corpora), leave-one-group-out
(A: 4 workers; B: 16 init_state_ids). PCA, EM and the filter are label-free; labels enter only
at AUC time. The filter is forward-only, so every score at query t uses chunks 0..t of that
branch and parameters that branch never contributed to.
Latent grid: r in {8,16,32,64} x d in {4,8,16,32}, 9 configs. **Main config r=32, d=8, declared
before any AUC was computed.**

Model health check (so the negative is not a broken fit): held-out mean NIS is 34.8 (A) and 31.8
(B) against the calibrated value r=32, i.e. the innovation covariance S_k transfers correctly to
branches from unseen groups; EM log-likelihood is monotone in every fold x config; the learned
latent is a near-unit-root drift (max |eig F| = 0.97-1.03) with a non-trivial action coupling
(||B|| = 0.7-9.3, growing with d).

---

## 1. Baseline reproduction — PASS (exact)

Hellinger to previous chunk per (layer, token), mean over the 40 back-block/action-token cells,
denoise 9, 8-step window mean, within-group AUC pooled by pair count:

| t | A (352 branches, 4 workers, 2933 pairs) | B (512 branches, 16 init states, 2008 pairs) |
|---|---|---|
| 20 | 0.5943 | 0.6444 |
| 25 | 0.6345 | 0.5249 |
| **30** | **0.7988** (target 0.799) | **0.7510** (target 0.751) |

---

## 2. The three sentinels (reported before any headline)

### Sentinel 1 — shuffle (acausal control, whole pipeline re-run on within-branch permuted chunks)

Routing and action chunk permuted together; PCA, EM and scoring all redone on the shuffled data.

| corpus | t | family max REAL (144 SSM cells) | family max SHUFFLE | cells with real>shuffle | median(real-shuffle) |
|---|---|---|---|---|---|
| A | 20 | 0.730 | **0.716** | 0.44 | -0.009 |
| A | 25 | 0.720 | **0.728** | 0.58 | +0.023 |
| A | 30 | 0.807 | **0.714** | 0.77 | +0.050 |
| B | 20 | 0.681 | **0.905** | 0.32 | -0.031 |
| B | 25 | 0.698 | **0.911** | 0.40 | -0.022 |
| B | 30 | 0.804 | **0.891** | 0.78 | +0.080 |

The leakage ceiling on corpus B is **0.891-0.911**, close to the 0.96 reported previously; on
corpus A it is **0.714-0.728**, well below the 0.93 reported previously (my baseline aggregation
is an 8-chunk window, not a whole-history statistic: the shuffled *baseline* only reaches
0.508/0.546/0.606 in A and 0.615/0.598/0.576 in B).

The damaging part is not the ceiling but the comparison: **on B the order-destroyed pipeline
beats the causal one at every t, and on A the two are tied at t=20 and t=25.** Only at t=30 does
the causal pipeline gain a clear median advantage (+0.050 A, +0.080 B). Whatever the SSM
read-outs are measuring is, at t<=25, obtainable without temporal order at all.

### Sentinel 2 — clock

Mutual information between the filtered latent (32-state k-means quantisation of z_{k|k},
main config) and the clock, over all valid (branch, chunk) with k<=30:

| corpus | H(t) | MI(latent, t) | MI/H(t) | MI(latent, T) | MI/H(T) | MI(latent, t/T) | MI/H(t/T) | per-dim max MI(z_j, t) |
|---|---|---|---|---|---|---|---|---|
| A | 3.986 b | 1.696 b | **0.425** | 0.460 b | 0.168 | 1.810 b | 0.453 | 0.793 b |
| B | 3.986 b | 2.433 b | **0.610** | 0.155 b | 0.087 | 1.975 b | 0.495 | 1.305 b |

The latent is substantially a clock — it recovers 43% (A) / 61% (B) of the entropy of the chunk
index. Episode length T is much less encoded (9-17%). Within-group Spearman of the score at t=30
with T: nis_w8 -0.068 (A) / -0.242 (B); the baseline itself is -0.031 (A) / **-0.475** (B).

### Sentinel 3 — redundancy with the scalar baseline (within group, t=30)

| stat | A rho | A range | B rho | B range |
|---|---|---|---|---|
| innov_w8 | +0.690 | [+0.484, +0.798] | +0.534 | [-0.213, +0.875] |
| nis_w8 | +0.706 | [+0.409, +0.862] | +0.437 | [-0.288, +0.861] |
| nis_w12 | +0.662 | [+0.407, +0.874] | +0.369 | [-0.284, +0.842] |
| dzf_w8 (filtered-state change) | +0.864 | [+0.733, +0.950] | +0.696 | [+0.315, +0.951] |
| pcaz_w8 (zeroth-order in PCA space) | +0.881 | [+0.792, +0.960] | +0.784 | [+0.270, +0.956] |

**|rho| < 0.9 for the innovation statistics in both corpora, so the innovation is NOT the
baseline repackaged** — this is the one place where the latent-variable framing does something
new. The filtered-state *change* and the PCA-space zeroth-order predictor are, as expected,
near-duplicates of the baseline (0.70-0.88).

---

## 3. Innovation detection vs baseline (main config r=32, d=8)

`auc_succ` = P(score | success > score | failure). `det` = max(auc, 1-auc), i.e. the sign is
chosen with the label, as in the previous rounds. 200-draw within-group branch-level label
permutation.

| corpus | t | baseline det | stat | auc_succ | raw det | p raw | resid det | p resid | rho vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.594 | innov_w8 | 0.340 | 0.660 | 0.005 | 0.646 | 0.005 | -0.008 |
| A | 20 | 0.594 | nis_w8 | 0.483 | 0.517 | 0.706 | 0.547 | 0.308 | -0.239 |
| A | 20 | 0.594 | nis_w12 | 0.558 | 0.558 | 0.264 | 0.537 | 0.478 | +0.170 |
| A | 25 | 0.635 | innov_w8 | 0.502 | 0.502 | 0.985 | 0.567 | 0.249 | +0.363 |
| A | 25 | 0.635 | nis_w8 | 0.679 | 0.679 | 0.005 | 0.646 | 0.010 | +0.279 |
| A | 25 | 0.635 | nis_w12 | 0.666 | 0.666 | 0.005 | 0.675 | 0.005 | -0.175 |
| A | 30 | **0.799** | innov_w8 | 0.693 | 0.693 | 0.005 | 0.507 | 0.896 | +0.690 |
| A | 30 | **0.799** | nis_w8 | 0.695 | 0.695 | 0.005 | 0.529 | 0.587 | +0.706 |
| A | 30 | **0.799** | nis_w12 | 0.712 | 0.712 | 0.005 | 0.562 | 0.249 | +0.662 |
| B | 20 | 0.644 | innov_w8 | 0.498 | 0.502 | 0.960 | 0.521 | 0.532 | +0.368 |
| B | 20 | 0.644 | nis_w8 | 0.413 | 0.587 | 0.015 | 0.587 | 0.020 | +0.241 |
| B | 20 | 0.644 | nis_w12 | 0.401 | 0.599 | 0.010 | 0.575 | 0.035 | +0.187 |
| B | 25 | 0.525 | innov_w8 | 0.390 | 0.610 | 0.020 | 0.651 | 0.005 | +0.538 |
| B | 25 | 0.525 | nis_w8 | 0.382 | 0.618 | 0.010 | 0.633 | 0.005 | +0.488 |
| B | 25 | 0.525 | nis_w12 | 0.335 | 0.665 | 0.005 | 0.689 | 0.005 | +0.428 |
| B | 30 | **0.751** | innov_w8 | 0.699 | 0.699 | 0.005 | 0.559 | 0.129 | +0.534 |
| B | 30 | **0.751** | nis_w8 | 0.599 | 0.599 | 0.010 | 0.520 | 0.557 | +0.437 |
| B | 30 | **0.751** | nis_w12 | 0.535 | 0.535 | 0.393 | 0.562 | 0.095 | +0.369 |

Family-corrected (max |AUC-0.5| over all 144 SSM cells; the baseline is NOT in the family, the
increment is measured on the residualised score):

| corpus | t | baseline det | best residual cell | det | family null 95% | p_family |
|---|---|---|---|---|---|---|
| A | 20 | 0.594 | r64d8 zfnorm_w8 | 0.697 | 0.669 | 0.010 |
| A | 25 | 0.635 | r64d32 innov_w3 | 0.715 | 0.670 | 0.005 |
| A | 30 | 0.799 | r16d4 zfnorm_w8 | 0.675 | 0.673 | 0.050 |
| B | 20 | 0.644 | r64d8 nis_cum | 0.620 | 0.615 | 0.035 |
| B | 25 | 0.525 | r16d8 innov_w12 | 0.704 | 0.626 | 0.005 |
| B | 30 | 0.751 | r32d4 zfnorm_w8 | 0.754 | 0.630 | 0.005 |

**The increments do not transfer.** Cross-corpus sign agreement of the residualised cells is
0.486 (t=20), **0.292** (t=25) and 0.625 (t=30) — at t=25 the two corpora systematically
*disagree*. The per-corpus winners transfer as:

| best cell | in its own corpus | in the other corpus |
|---|---|---|
| A t=25 r64d32 innov_w3 | auc 0.715 | B auc 0.563 |
| B t=25 r16d8 innov_w12 | auc 0.296 | A auc 0.471 |
| B t=20 r64d8 nis_cum | auc 0.380 | A auc 0.499 |
| A t=30 r16d4 zfnorm_w8 | auc 0.325 | B auc 0.266 (agrees) |
| B t=30 r32d4 zfnorm_w8 | auc 0.246 | A auc 0.339 (agrees) |

The sign flip is specific to the innovation: at t=25 and t=30 the *baseline* has the same sign in
both corpora (auc_succ > 0.5), while nis_w12 has auc_succ 0.666 in A and 0.335 in B. The only
cell that transfers with a consistent sign is `zfnorm_w8` at t=30 — the **filtered-state norm**,
not the innovation — and Sentinel 1 disqualifies it (its shuffled twin scores 0.71-0.84, i.e. it
is *better* with time scrambled).

### Action-input ablation (does B u_k earn its place?)

Same model, control input zeroed, everything else identical:

| corpus | held-out predictive nll/chunk with action | action zeroed | delta | detection effect |
|---|---|---|---|---|
| A | 33.194 | 34.658 | **-1.464** | det changes by -0.070..+0.063, no consistent sign |
| B | 31.696 | 33.521 | **-1.825** | det changes by -0.021..+0.034, no consistent sign |

The executed action chunk **is** genuinely predictive of the next routing state (1.5-1.8 nats per
chunk of held-out one-step predictive likelihood, out of ~33). It buys a better *model* and
exactly nothing in *detection*.

### No-model comparator

A static Mahalanobis score — `||x_k||`, the whitened distance of the clr routing vector from the
training-fold mean, same 8-chunk window, no dynamics, no filter, no action input:

| corpus | t | det (real) | resid det | det (shuffled) |
|---|---|---|---|---|
| A | 20 | 0.718 | 0.685 | 0.531 |
| A | 25 | 0.533 | 0.621 | 0.548 |
| A | 30 | 0.696 | 0.514 | 0.564 |
| B | 20 | 0.531 | 0.549 | 0.550 |
| B | 25 | 0.575 | 0.615 | 0.544 |
| B | 30 | 0.701 | 0.557 | 0.543 |

This matches or beats the whole state-space apparatus (A t=20: 0.718 vs family max 0.730;
A t=30: 0.696 vs nis_w12 0.712; B t=30: 0.701 vs nis_w8 0.599) and, unlike `zfnorm_w8`, it does
depend on temporal order (shuffled 0.53-0.56).

---

## 4. The two falsifiable predictions

### P1 — "innovation largest when ||A_k|| is large and routing change is small": FALSIFIED, with the same sign in both corpora.

Pooled within-branch-standardised regression over held-out (branch, chunk), k=1..30:

| corpus | target | b(&#124;&#124;A_k&#124;&#124;) | b(route change) | b(interaction) | rho with &#124;&#124;A&#124;&#124; | partial rho given route change | n |
|---|---|---|---|---|---|---|---|
| A | innovation norm | **-0.288** | +0.366 | -0.175 | -0.252 | **-0.324** | 10560 |
| A | NIS | **-0.313** | +0.467 | -0.098 | -0.268 | **-0.370** | 10560 |
| B | innovation norm | **-0.215** | +0.593 | -0.157 | -0.152 | **-0.253** | 15360 |
| B | NIS | **-0.243** | +0.607 | -0.113 | -0.177 | **-0.289** | 15360 |

Quadrant means of standardised NIS (splits at within-branch medians):

| corpus | high &#124;&#124;A&#124;&#124;, low route change | high, high | low, low | low &#124;&#124;A&#124;&#124;, high route change |
|---|---|---|---|---|
| A | **-0.472** | -0.044 | -0.248 | **+0.942** |
| B | **-0.487** | +0.211 | -0.480 | **+1.007** |

The predicted cell (large action, small routing change) is the cell with the *smallest*
innovation. The innovation is largest when the action is small and the routing moved a lot. So
the innovation is not "the robot commanded motion and the scene did not respond"; it is
dominated by raw routing change, with action magnitude entering with the opposite sign.

### P2 — "gain concentrates in branches whose best fixed window differs from 8": not supported (and underpowered).

Per-group best baseline window W* and the nis_w8 gain over the baseline at t=30
(`pred2_A.csv`, `pred2_B.csv`; groups with 0% or 100% success are inestimable):

| corpus | estimable groups | mean gain, W*=8 | mean gain, W*!=8 |
|---|---|---|---|
| A | 3 | -0.149 (n=2) | +0.003 (n=1) |
| B | 13 | -0.080 (n=2) | -0.042 (n=11) |

The direction is nominally as predicted (less negative where W* != 8) but only 2 groups per
corpus have W* = 8, the gains are overwhelmingly negative, and B's largest single gain
(+0.345, group 23, W*=12) sits next to B's largest loss (-0.290, group 0, W*=5).
No evidence either way.

---

## 5. Sensitivity to latent rank

Range of raw det AUC across the 9 (r, d) configs:

| corpus | t | innov_w8 | nis_w8 | nis_w12 |
|---|---|---|---|---|
| A | 20 | 0.521-0.707 | 0.512-0.625 | 0.527-0.665 |
| A | 25 | 0.502-0.690 | 0.588-0.720 | 0.596-0.682 |
| A | 30 | 0.595-0.717 | 0.637-0.722 | 0.660-0.718 |
| B | 20 | 0.502-0.573 | 0.516-0.594 | 0.511-0.617 |
| B | 25 | 0.504-0.664 | 0.518-0.619 | 0.539-0.665 |
| B | 30 | 0.635-0.727 | 0.556-0.670 | 0.502-0.644 |

There is no monotone or even stable dependence on rank: the argmax config differs by corpus,
by t and by statistic (A t=25 favours r=8 d=4 for nis_w8 but r=64 d=32 for innov_w8; B t=25
favours r=16 d=8, B t=30 favours r=32 d=16). The spread across ranks (0.10-0.19) is of the same
order as the entire claimed effect. The pre-declared r=32 d=8 is never the best cell in any
corpus/t — the rank behaves as a free parameter of the same unstable kind as the window length
found in the earlier round.

---

## 6. Verdict

**Modelling the emission does not beat compressing it.** The Kalman innovation is genuinely a
different measurement from the scalar baseline — |rho| = 0.44-0.71, comfortably under the 0.9
redundancy line, so the latent-variable framing is not the baseline in disguise — and the
executed action chunk really does improve the fitted model (1.5-1.8 nats/chunk of held-out
predictive likelihood). But at the fixed protocol query t=30 the innovation loses to the scalar
in both corpora (A 0.71 vs 0.799; B 0.60-0.70 vs 0.751); its residual increments are large only
where the baseline happens to be weak (B t=25, baseline 0.525); those increments have **opposite
sign in the two corpora** (sign agreement 0.292 at t=25); the acausal shuffle control matches or
beats the causal pipeline everywhere except t=30; the latent recovers 43-61% of the entropy of
the chunk index; and a static Mahalanobis distance with no dynamics at all reproduces the whole
result. The one falsifiable prediction that could have distinguished the innovation from a
change rate — large action, small routing change — comes out backwards in both corpora with
matching sign.

Why the scalar survives again: the innovation is still, empirically, a function of routing change
(b(route change) = +0.37 to +0.61 dominates b(||A||) = -0.22 to -0.31), and the extra degrees of
freedom the model adds — latent rank, action coupling, filter memory — behave as free parameters
that each corpus re-tunes rather than as a shared mechanism. That is the same failure mode as the
646 slicings and the 13,494 aggregations arriving through a different door: the tensor's one
degree of freedom is not an artefact of reading it with a scalar, because a model that reads it
in 2 728 dimensions with learned dynamics and the action as input still ends up tracking the same
quantity, plus a clock.
