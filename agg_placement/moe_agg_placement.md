# Aggregation-method brute force on the HiMoE-VLA MoE routing tensor

Placement x operator x axis-scope, with divergence = Hellinger and temporal aggregation = mean over an 8-step window held fixed (sibling agent's axis).

## 0. Grid definition

```
layer   : scope {all(8), back 12-15, front 2-5} x place {Before, After} x op x6  = 36
denoise : scope all(10)  x place {B,A} x op x6  +  scope d9 (size 1, degenerate) = 13
token   : scope {all11, act 1-10} x place {B,A} x op x6  +  state (size 1)      = 25
total   : 36 x 13 x 25 = 11700 cells  x  t in [20, 25, 30]  x  2 corpora
ops     : mean, max, min, median, std, q75
place=B : aggregate the 32-dim probability vectors along that axis (re-normalise),
          then take ONE Hellinger against the previous chunk.
place=A : one Hellinger per element of that axis, then aggregate the scalars.
order   : within the Before group and within the After group, axes are reduced
          layer -> denoise -> token (see agg_order_probe.py for order sensitivity).
```

Baseline cell = `L[back/A/mean]_D[d9/-/id]_T[act/A/mean]` (index 6168): divergence-first over layer and token, back block, action tokens, denoise 9.

| corpus | t | baseline raw auc | baseline det AUC | brief target |
|---|---|---|---|---|
| A | 20 | 0.5943 | 0.5943 |  |
| A | 25 | 0.3655 | 0.6345 |  |
| A | 30 | 0.2012 | 0.7988 | 0.795 |
| B | 20 | 0.3556 | 0.6444 |  |
| B | 25 | 0.4751 | 0.5249 |  |
| B | 30 | 0.2490 | 0.7510 | 0.759 |

Reproduction check at t=30: A 0.7988 vs 0.795, B 0.7510 vs 0.759 -> both within 0.01. Convention: raw `auc` = P(feature_fail > feature_succ); detection AUC = max(auc, 1-auc). Route change is LOWER on failures at t=30 (raw auc < 0.5) in both corpora.

## 1. The full grid

### RAW detection AUC, distribution over all 11700 cells

| corpus | t | min | q25 | median | q75 | max | baseline | #cells > baseline |
|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.500 | 0.524 | 0.553 | 0.593 | 0.746 | 0.594 | 2819 |
| A | 25 | 0.500 | 0.558 | 0.601 | 0.641 | 0.771 | 0.635 | 3333 |
| A | 30 | 0.500 | 0.638 | 0.751 | 0.787 | 0.848 | 0.799 | 1907 |
| B | 20 | 0.500 | 0.536 | 0.572 | 0.609 | 0.713 | 0.644 | 1071 |
| B | 25 | 0.500 | 0.516 | 0.534 | 0.560 | 0.743 | 0.525 | 7310 |
| B | 30 | 0.500 | 0.657 | 0.697 | 0.731 | 0.817 | 0.751 | 1230 |

### RESIDUALISED on the baseline detection AUC, distribution over all 11700 cells

| corpus | t | min | q25 | median | q75 | max | baseline | #cells > baseline |
|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.500 | 0.521 | 0.546 | 0.577 | 0.732 | 0.594 | 1760 |
| A | 25 | 0.500 | 0.528 | 0.557 | 0.593 | 0.768 | 0.635 | 908 |
| A | 30 | 0.500 | 0.526 | 0.555 | 0.592 | 0.710 | 0.799 | 0 |
| B | 20 | 0.500 | 0.512 | 0.527 | 0.548 | 0.630 | 0.644 | 0 |
| B | 25 | 0.500 | 0.514 | 0.530 | 0.555 | 0.720 | 0.525 | 6691 |
| B | 30 | 0.500 | 0.530 | 0.553 | 0.584 | 0.713 | 0.751 | 0 |

## 2. Divergence-before vs divergence-after

Mean detection AUC by 3-letter placement code (layer, denoise, token; `-` = size-1 scope so the choice does not exist). Restricted to the fully crossed subgrid (denoise scope = all, token scope != state).

**t = 20**

| placement | n | A raw | A resid | B raw | B resid |
|---|---|---|---|---|---|
| AAA | 1296 | 0.5728 | 0.5476 | 0.5770 | 0.5317 |
| AAB | 1296 | 0.5479 | 0.5483 | 0.5801 | 0.5304 |
| ABA | 1296 | 0.5723 | 0.5507 | 0.5797 | 0.5321 |
| ABB | 1296 | 0.5496 | 0.5456 | 0.5794 | 0.5279 |
| BAA | 1296 | 0.5766 | 0.5622 | 0.5635 | 0.5314 |
| BAB | 1296 | 0.5582 | 0.5545 | 0.5673 | 0.5337 |
| BBA | 1296 | 0.5740 | 0.5624 | 0.5663 | 0.5342 |
| BBB | 1296 | 0.5562 | 0.5545 | 0.5651 | 0.5331 |

**t = 25**

| placement | n | A raw | A resid | B raw | B resid |
|---|---|---|---|---|---|
| AAA | 1296 | 0.5850 | 0.5531 | 0.5434 | 0.5390 |
| AAB | 1296 | 0.6243 | 0.5653 | 0.5449 | 0.5394 |
| ABA | 1296 | 0.5854 | 0.5517 | 0.5430 | 0.5365 |
| ABB | 1296 | 0.6191 | 0.5611 | 0.5416 | 0.5370 |
| BAA | 1296 | 0.5796 | 0.5775 | 0.5394 | 0.5358 |
| BAB | 1296 | 0.6053 | 0.5664 | 0.5408 | 0.5369 |
| BBA | 1296 | 0.5816 | 0.5707 | 0.5405 | 0.5358 |
| BBB | 1296 | 0.5989 | 0.5600 | 0.5414 | 0.5385 |

**t = 30**

| placement | n | A raw | A resid | B raw | B resid |
|---|---|---|---|---|---|
| AAA | 1296 | 0.6728 | 0.5587 | 0.6935 | 0.5611 |
| AAB | 1296 | 0.7467 | 0.5594 | 0.7104 | 0.5717 |
| ABA | 1296 | 0.6853 | 0.5580 | 0.6867 | 0.5596 |
| ABB | 1296 | 0.7491 | 0.5646 | 0.6962 | 0.5620 |
| BAA | 1296 | 0.6685 | 0.5653 | 0.6810 | 0.5551 |
| BAB | 1296 | 0.7316 | 0.5630 | 0.6779 | 0.5595 |
| BBA | 1296 | 0.6770 | 0.5614 | 0.6728 | 0.5538 |
| BBB | 1296 | 0.7266 | 0.5659 | 0.6611 | 0.5494 |

Marginal effect of each axis's placement (mean det AUC, crossed subgrid, t=30):

| axis | corpus | Before | After | delta (B-A) |
|---|---|---|---|---|
| layer | A | 0.7009 | 0.7135 | -0.0126 |
| layer | B | 0.6732 | 0.6967 | -0.0235 |
| denoise | A | 0.7095 | 0.7049 | +0.0046 |
| denoise | B | 0.6792 | 0.6907 | -0.0115 |
| token | A | 0.7385 | 0.6759 | +0.0626 |
| token | B | 0.6864 | 0.6835 | +0.0029 |

## 3. Which choice matters most (one-way eta^2 on detection AUC)

Fraction of the grid's variance in detection AUC explained by each design factor alone, on the fully-crossed subgrid.

```
                 A                                         B                                   
               raw                  res                  raw                  res              
                20     25     30     20     25     30     20     25     30     20     25     30
layer_scope  0.016  0.045  0.192  0.015  0.002  0.043  0.050  0.001  0.068  0.004  0.001  0.011
layer_place  0.004  0.013  0.004  0.017  0.015  0.002  0.022  0.001  0.038  0.003  0.000  0.015
layer_op     0.049  0.057  0.040  0.018  0.011  0.012  0.024  0.008  0.051  0.001  0.011  0.030
den_place    0.000  0.000  0.001  0.000  0.003  0.000  0.000  0.000  0.009  0.000  0.000  0.006
den_op       0.030  0.022  0.064  0.002  0.014  0.031  0.042  0.027  0.114  0.008  0.012  0.012
tok_scope    0.000  0.024  0.055  0.003  0.012  0.002  0.101  0.081  0.000  0.017  0.084  0.005
tok_place    0.052  0.077  0.104  0.004  0.000  0.001  0.000  0.000  0.001  0.000  0.000  0.002
tok_op       0.099  0.019  0.019  0.038  0.017  0.015  0.035  0.092  0.054  0.015  0.093  0.018
```

## 4. Top cells

**RAW grid, corpus A, top 10** (other corpus shown for the same cell)

```
  L[all/B/min]_D[all/B/max]_T[all11/B/std]             t=30  det=0.8483 pw_p=0.005  other=0.6868 same
  L[all/B/min]_D[all/B/q75]_T[all11/B/std]             t=30  det=0.8469 pw_p=0.005  other=0.6967 same
  L[back/B/q75]_D[all/A/max]_T[all11/B/std]            t=30  det=0.8462 pw_p=0.005  other=0.7276 same
  L[all/B/median]_D[all/A/median]_T[act/B/mean]        t=30  det=0.8435 pw_p=0.005  other=0.6439 same
  L[all/B/min]_D[all/A/median]_T[all11/B/std]          t=30  det=0.8418 pw_p=0.005  other=0.7047 same
  L[all/B/min]_D[all/A/min]_T[all11/B/std]             t=30  det=0.8415 pw_p=0.005  other=0.7062 same
  L[all/B/min]_D[all/A/q75]_T[all11/B/std]             t=30  det=0.8404 pw_p=0.005  other=0.6957 same
  L[all/B/median]_D[all/B/median]_T[act/B/mean]        t=30  det=0.8404 pw_p=0.005  other=0.6360 same
  L[all/B/min]_D[all/A/max]_T[all11/B/std]             t=30  det=0.8404 pw_p=0.005  other=0.6858 same
  L[back/B/q75]_D[all/B/mean]_T[all11/B/std]           t=30  det=0.8404 pw_p=0.005  other=0.7206 same
```

**RAW grid, corpus B, top 10** (other corpus shown for the same cell)

```
  L[back/B/min]_D[all/B/min]_T[act/B/max]              t=30  det=0.8167 pw_p=0.005  other=0.7600 same
  L[back/B/min]_D[d9/-/id]_T[act/B/max]                t=30  det=0.8088 pw_p=0.005  other=0.7600 same
  L[front/A/max]_D[all/B/std]_T[act/B/max]             t=30  det=0.8063 pw_p=0.005  other=0.7184 same
  L[front/A/std]_D[d9/-/id]_T[all11/A/median]          t=30  det=0.8043 pw_p=0.005  other=0.7457 same
  L[back/B/min]_D[all/B/max]_T[all11/A/mean]           t=30  det=0.8033 pw_p=0.005  other=0.8251 same
  L[back/B/min]_D[all/A/max]_T[act/B/max]              t=30  det=0.8028 pw_p=0.005  other=0.7760 same
  L[all/A/max]_D[d9/-/id]_T[act/B/std]                 t=30  det=0.8013 pw_p=0.005  other=0.7954 same
  L[all/A/max]_D[all/A/median]_T[act/B/mean]           t=30  det=0.7993 pw_p=0.005  other=0.7644 same
  L[all/A/max]_D[all/A/max]_T[act/B/std]               t=30  det=0.7988 pw_p=0.005  other=0.7862 same
  L[all/A/max]_D[all/A/mean]_T[act/B/std]              t=30  det=0.7983 pw_p=0.005  other=0.8159 same
```

**RESIDUAL grid, corpus A, top 10** (other corpus shown for the same cell)

```
  L[all/B/min]_D[d9/-/id]_T[all11/B/mean]              t=25  det=0.7682 pw_p=0.005  other=0.5329 same
  L[all/B/min]_D[all/A/max]_T[all11/B/mean]            t=25  det=0.7583 pw_p=0.005  other=0.5418 same
  L[all/A/mean]_D[all/A/max]_T[act/B/std]              t=25  det=0.7579 pw_p=0.005  other=0.5737 same
  L[all/A/mean]_D[d9/-/id]_T[act/B/std]                t=25  det=0.7538 pw_p=0.005  other=0.5842 same
  L[all/B/min]_D[all/A/min]_T[all11/B/min]             t=25  det=0.7419 pw_p=0.005  other=0.5946 same
  L[front/B/median]_D[all/B/q75]_T[all11/B/min]        t=25  det=0.7412 pw_p=0.005  other=0.6399 same
  L[all/B/min]_D[all/B/min]_T[all11/B/mean]            t=25  det=0.7409 pw_p=0.005  other=0.5239 same
  L[all/B/min]_D[all/B/median]_T[all11/B/min]          t=25  det=0.7402 pw_p=0.005  other=0.5926 same
  L[all/B/min]_D[all/A/mean]_T[all11/B/min]            t=25  det=0.7395 pw_p=0.005  other=0.5906 same
  L[front/B/median]_D[all/B/median]_T[all11/B/min]     t=25  det=0.7392 pw_p=0.005  other=0.6504 same
```

**RESIDUAL grid, corpus B, top 10** (other corpus shown for the same cell)

```
  L[all/B/q75]_D[all/A/median]_T[all11/B/min]          t=25  det=0.7201 pw_p=0.005  other=0.6403 same
  L[all/B/q75]_D[all/A/max]_T[all11/B/min]             t=25  det=0.7181 pw_p=0.005  other=0.6396 same
  L[all/B/q75]_D[all/B/q75]_T[all11/B/min]             t=25  det=0.7136 pw_p=0.005  other=0.6434 same
  L[front/A/max]_D[all/B/std]_T[act/B/max]             t=30  det=0.7126 pw_p=0.005  other=0.5305 same
  L[all/B/q75]_D[all/A/mean]_T[all11/B/min]            t=25  det=0.7126 pw_p=0.005  other=0.6410 same
  L[all/B/q75]_D[all/A/q75]_T[all11/B/min]             t=25  det=0.7122 pw_p=0.005  other=0.6420 same
  L[all/B/q75]_D[all/B/median]_T[all11/B/min]          t=25  det=0.7112 pw_p=0.005  other=0.6437 same
  L[all/B/q75]_D[all/B/mean]_T[all11/B/min]            t=25  det=0.7112 pw_p=0.005  other=0.6447 same
  L[back/B/min]_D[all/B/min]_T[act/B/max]              t=30  det=0.7107 pw_p=0.005  other=0.5823 same
  L[all/B/q75]_D[all/A/min]_T[all11/B/min]             t=25  det=0.7102 pw_p=0.005  other=0.6270 same
```

## 5. Permutation family-wise p (200 draws, labels shuffled within group)

Family statistic = max detection AUC over the whole grid and all three t. One permutation per draw at branch level, reused across t.

| corpus | grid | observed max | null mean | null p95 | null max | family p |
|---|---|---|---|---|---|---|
| A | raw | 0.8483 | 0.6957 | 0.7345 | 0.7589 | 0.0050 |
| A | res | 0.7682 | 0.7033 | 0.7327 | 0.7562 | 0.0050 |
| B | raw | 0.8167 | 0.6452 | 0.6654 | 0.6788 | 0.0050 |
| B | res | 0.7201 | 0.6492 | 0.6718 | 0.6897 | 0.0050 |

Raw grid with the baseline cell removed (the previous round's lesson: a family that contains the baseline only certifies the baseline):

* corpus A: observed 0.8483, family p = 0.0050 -- unchanged, because the raw family is saturated with near-copies of the baseline (median |within-group rank rho| with the baseline = 0.796 at t=30).
* corpus B: observed 0.8167, family p = 0.0050 -- unchanged, because the raw family is saturated with near-copies of the baseline (median |within-group rank rho| with the baseline = 0.575 at t=30).

## 6. Cross-corpus agreement

**Sign agreement of the RESIDUALS**: 20238/35100 = **57.7%** of cells (chance = 50%).

| slice | cells | sign agreement |
|---|---|---|
| t=20 | 11700 | 57.9% |
| t=25 | 11700 | 59.0% |
| t=30 | 11700 | 56.1% |
| layer_place=B | 17550 | 56.9% |
| layer_place=A | 17550 | 58.4% |
| den_place=B | 16200 | 58.5% |
| den_place=A | 16200 | 56.2% |
| tok_place=B | 16848 | 63.3% |
| tok_place=A | 16848 | 50.3% |
| layer_scope=all | 11700 | 57.7% |
| layer_scope=back | 11700 | 55.9% |
| layer_scope=front | 11700 | 59.4% |
| tok_scope=all11 | 16848 | 61.2% |
| tok_scope=act | 16848 | 52.4% |
| tok_scope=state | 1404 | 77.9% |
| den_scope=all | 32400 | 57.3% |
| den_scope=d9 | 2700 | 61.9% |

(Raw grid, for reference: 67.5% sign agreement.)

**Joint criterion** -- a cell counts only if it clears in both corpora. Statistic = min(det_A, det_B) with matching direction; null pairs draw i of corpus A with draw i of corpus B (independent corpora).

| grid | observed max min(det_A,det_B) | cell | null p95 | joint family p |
|---|---|---|---|---|
| raw | 0.8033 | `L[back/B/min]_D[all/B/max]_T[all11/A/mean]` t=30 | 0.6321 | 0.0050 |
| res | 0.6626 | `L[all/B/mean]_D[all/A/max]_T[all11/B/min]` t=25 | 0.6326 | 0.0050 |

* raw grid: **8548** of 35100 cells clear the joint FWER-95 threshold (0.6321) in both corpora.
```
  L[back/B/min]_D[all/B/max]_T[all11/A/mean]           t=30  A=0.8251 B=0.8033
  L[all/A/max]_D[all/A/mean]_T[act/B/std]              t=30  A=0.8159 B=0.7983
  L[all/A/mean]_D[d9/-/id]_T[act/B/std]                t=30  A=0.8023 B=0.7968
  L[back/B/min]_D[all/B/mean]_T[all11/A/mean]          t=30  A=0.8230 B=0.7963
  L[all/A/max]_D[all/A/q75]_T[act/B/std]               t=30  A=0.8118 B=0.7963
  L[all/A/max]_D[d9/-/id]_T[act/B/std]                 t=30  A=0.7954 B=0.8013
  L[back/B/min]_D[all/B/max]_T[act/A/mean]             t=30  A=0.8159 B=0.7953
  L[back/A/q75]_D[all/A/max]_T[act/B/std]              t=30  A=0.8145 B=0.7948
  L[back/A/min]_D[d9/-/id]_T[all11/B/std]              t=30  A=0.8203 B=0.7943
  L[back/B/std]_D[all/A/q75]_T[all11/A/min]            t=30  A=0.8023 B=0.7938
  L[back/A/max]_D[all/A/min]_T[act/B/mean]             t=30  A=0.7937 B=0.7983
  L[back/B/min]_D[all/A/median]_T[all11/A/mean]        t=30  A=0.8155 B=0.7923
  L[back/B/min]_D[all/B/median]_T[all11/A/mean]        t=30  A=0.8162 B=0.7923
  L[back/B/min]_D[all/B/q75]_T[all11/A/mean]           t=30  A=0.8241 B=0.7923
  L[back/B/min]_D[all/B/q75]_T[act/A/median]           t=30  A=0.7968 B=0.7918
```
* res grid: **118** of 35100 cells clear the joint FWER-95 threshold (0.6326) in both corpora.
```
  L[all/B/mean]_D[all/A/max]_T[all11/B/min]            t=25  A=0.6628 B=0.6626
  L[all/B/mean]_D[all/B/min]_T[all11/B/min]            t=25  A=0.6611 B=0.6624
  L[all/B/mean]_D[all/B/q75]_T[all11/B/min]            t=25  A=0.6611 B=0.6927
  L[back/A/min]_D[d9/-/id]_T[all11/B/std]              t=30  A=0.6642 B=0.6609
  L[front/B/median]_D[all/A/median]_T[all11/B/min]     t=25  A=0.7382 B=0.6604
  L[all/B/mean]_D[all/B/max]_T[all11/B/min]            t=25  A=0.6597 B=0.6863
  L[front/B/median]_D[all/A/mean]_T[all11/B/min]       t=25  A=0.7354 B=0.6589
  L[front/B/median]_D[d9/-/id]_T[all11/B/min]          t=25  A=0.7208 B=0.6579
  L[all/B/mean]_D[all/B/mean]_T[all11/B/min]           t=25  A=0.6573 B=0.6858
  L[front/B/median]_D[all/A/min]_T[all11/B/min]        t=25  A=0.7364 B=0.6569
  L[all/B/mean]_D[all/A/min]_T[all11/B/min]            t=25  A=0.6560 B=0.6897
  L[all/B/mean]_D[all/A/mean]_T[all11/B/min]           t=25  A=0.6556 B=0.6813
  L[front/B/median]_D[all/A/max]_T[all11/B/min]        t=25  A=0.7368 B=0.6554
  L[front/A/std]_D[all/B/min]_T[all11/B/min]           t=25  A=0.6550 B=0.6574
  L[front/B/median]_D[all/A/q75]_T[all11/B/min]        t=25  A=0.7361 B=0.6549
```

## 7. Degenerate / pruned cells

* corpus A: 0 of 11700 cells have a *constant* per-branch score at some t (zero variance -> AUC pinned at 0.5).
* corpus B: 0 of 11700 cells have a *constant* per-branch score at some t (zero variance -> AUC pinned at 0.5).


## 8. Order sensitivity AMONG the before-reductions (a fourth undeclared parameter)

The grid above fixes the order layer -> denoise -> token inside the Before group. Because before-ops act on probability vectors and are re-normalised after each step, they do not commute with each other either. `agg_order_probe.py` runs all 6 orders for placement BBB, 9 operator triples, 2 scope sets, both corpora (`/tmp/moe_agg_order_probe.csv`).

| operator triple (layer/denoise/token) | A all-scope span | A back/act span | B all-scope span | B back/act span |
|---|---|---|---|---|
| mean/mean/mean | 0.0003 | 0.0003 | 0.0005 | 0.0005 |
| median/median/median | 0.0232 | 0.0061 | 0.0239 | 0.0244 |
| min/min/min | 0.0140 | 0.0143 | 0.0378 | 0.0279 |
| q75/q75/q75 | 0.0740 | 0.0106 | 0.0523 | 0.0224 |
| max/mean/mean | 0.0068 | 0.0085 | 0.0264 | 0.0797 |
| mean/max/mean | 0.0147 | 0.0099 | 0.0204 | 0.0234 |
| mean/mean/max | 0.0068 | 0.0232 | 0.0204 | 0.0299 |
| std/mean/mean | 0.0259 | 0.0201 | 0.0349 | 0.0224 |
| max/min/median | 0.1057 | 0.0552 | 0.0677 | 0.0438 |

`mean` is the only operator that is (numerically) order-invariant: span 0.0003-0.0005, i.e. exactly the float32 renormalisation residue. Every other operator makes the order matter, up to **0.106 AUC** (corpus A, all-scope, max/min/median). So a fully specified aggregation needs FOUR declarations, not one: placement per axis, operator per axis, scope per axis, and the order among the before-reductions.

## 9. What is degenerate, and what was pruned

Nothing was pruned from the reported grid; `min` over probability vectors is not degenerate (it is the token/layer-wise 'consensus core' of the router distribution and renormalises to a well-conditioned simplex point). Zero of 11700 cells produce a constant per-branch score in either corpus (relative sd < 1e-3: 0 cells).

**One numerical degeneracy was found and audited.** Corpus B's state token is bitwise constant across the 10 denoise iterations (`max|p(d)-p(0)| = 0.0`, reproducing the previous round's finding). Therefore `denoise place=Before, op=std` on the state token reduces a constant axis: the resulting 'distribution' is float32 round-off (median std 9.3e-10 for token 0 vs 8.8e-4 for tokens 1-10, a factor of 9.4e5) renormalised to full scale. That is the same class of artefact as the forbidden bf16 top-4 tie jitter.

* 900 of 11700 cells use `D[all/B/std]`; 36 of those also restrict to the state token and are therefore *entirely* round-off in corpus B.
* **None of them is among the surviving cells.** Dropping all 900 from the family leaves the joint result bit-identical: joint max 0.6626, threshold 0.6326, joint family p = 0.0050, 118 cells clearing.
* Corpus A's state token is *not* constant across denoise (median std 4.7e-6, ~200x below the action tokens but ~5000x above corpus B's round-off floor), consistent with the weak real suffix leak reported on 2026-08-29.

## 10. The number that matters most for reproducibility

**144 cells of this grid all answer to the exact one-line description used in every prior report of this signal** -- "back block 12-15, action tokens 1-10, denoise 9, Hellinger to the previous chunk, 8-step window mean". They differ only in placement and operator, both of which no prior report declared.

| corpus | t=30 det AUC over those 144 cells | baseline |
|---|---|---|
| A | 0.503 .. 0.825 (**span 0.322**) | 0.799 |
| B | 0.523 .. 0.809 (**span 0.286**) | 0.751 |

From chance to 0.825 without changing a single word of the description. The whole grid spans 0.500-0.848 (A) and 0.500-0.817 (B) at t=30. For comparison, the entire 646-slicing sweep of 2026-08-29 moved the number by less than this.

The legacy V1-V4 variants reproduce exactly: corpus B t=30 detection AUC 0.642 (V1) / 0.751 (V2) / 0.744 (V3) / 0.721 (V4), **range 0.109**, matching the figure quoted in the brief.

## 11. Scope note: the length / phase confound (applies to the baseline identically)

Within group, the branch's eventual episode length T separates success from failure at detection AUC **0.989 (A) / 0.999 (B)** -- successes terminate, failures run to the cap. This is the length confound flagged as section 3.3 of the 2026-08-29 sweep, and it is a property of the corpora, not of any aggregation. It is reported here because it bounds what the absolute AUCs mean; it is not a competing signal and it is not the object of study.

Residualising the whole grid on T within group (`agg_lenctrl.py`):

| grid residualised on | corpus A max det | A family p | corpus B max det | B family p | joint cells clearing | joint p | sign agreement |
|---|---|---|---|---|---|---|---|
| baseline (section 5) | 0.7682 | 0.0050 | 0.7201 | 0.0050 | **118** | 0.0050 | 57.7% |
| episode length T | 0.8005 | 0.0050 | 0.5732 | **1.0000** | **0** | 1.0000 | 55.4% |
| baseline AND T | 0.7947 | 0.0050 | 0.6233 | **1.0000** | **0** | 1.0000 | 52.0% |

The baseline itself goes to 0.622 (A) / **0.506** (B) at t=30 under the T control. So in corpus B, *every* aggregation in this grid -- the baseline included -- is within-group rank-collinear with how long the rollout will run.

**How to read this.** T is measured after the fact and is nearly the label, so conditioning on it is over-adjustment on a near-outcome, not a confounder correction; 'nothing survives' is close to tautological and should NOT be read as 'the routing signal is fake'. What it does establish is the weaker, honest statement: at a fixed absolute query index t, no aggregation in this grid carries outcome information that is separable from relative task phase in both corpora. That was already true of the baseline, so it does not change the ranking among aggregations, which is what this sweep was asked to search.

## 12. Verdict and recommendation

### (2) Does any aggregation beat the baseline in both corpora?

**Raw: yes, and not rarely.** At t=30, 345 of 11700 cells (2.9%) exceed the baseline's detection AUC in *both* corpora with the same direction. The best is `L[back/B/min]_D[all/B/max]_T[all11/A/mean]` at A 0.825 / B 0.803 (baseline 0.799 / 0.751). The `T[act/B/std]` family is close behind and more coherent: A 0.802-0.816 / B 0.796-0.798.

**Residualised on the baseline: yes, but it is two effects, not 118.** 118 of 35100 cells clear the joint FWER-95 threshold (0.6326) in both corpora; under the null, 95% of draws produce *zero* such cells (max 3). The 118 collapse into two families:

1. **t=25, token reduced BEFORE the divergence with `min` over all 11 tokens** (`T[all11/B/min]`, 44 cells, plus its `all11/A/max`, `all11/A/std`, `state` relatives). Residual det AUC A 0.656-0.663 / B 0.656-0.693; raw A 0.694-0.700 / B 0.690-0.740 against a baseline of A 0.635 / **B 0.525 (chance)**. Rank correlation with the baseline is only 0.82 (A) / **0.19 (B)** -- for corpus B this is very nearly an independent channel. Leave-one-group-out: A 0.611-0.722, B 0.667-0.717.
2. **t=30, token reduced BEFORE the divergence with `std` over the tokens** (`T[act/B/std]` 24 cells, `T[all11/B/std]` 7). Residual A 0.605-0.664 / B 0.661-0.674.

Both families are token-**Before** with a **non-mean** operator -- exactly the corner of the space that the 646-slicing sweep never entered, because it only ever averaged.

**Caveats that keep this suggestive rather than adopted.** (a) Neither family survives the phase control of section 11 in corpus B. (b) Family 1 depends on *including the state token*: dropping it (`T[act/B/min]`) collapses corpus B from 0.690 to 0.515 raw and to 0.500 residual, and the survivor correlates 0.91 (A) / 0.66 (B) with the state token alone -- so this is the same state-token channel the layer-axis agent downgraded to suggestive on 2026-08-29, re-found through a different door. (c) It is not lead time: the t=25 survivor correlates only +0.10 (A) / +0.15 (B) with the baseline at t=30, and at t=20 it is 0.519 in A versus 0.680 in B -- inconsistent. The standing zero-lead-time result is untouched.

### (5) Which choice matters most, and does before or after dominate?

**No single axis's operator matters most in a transferable way.** One-way eta^2 on the crossed subgrid at t=30 (raw): corpus A is led by layer *scope* (0.192), then token *placement* (0.104), then denoise operator (0.064); corpus B is led by the denoise operator (0.114), then layer scope (0.068), then token operator (0.054). The two corpora do not agree on the leader. All eight main effects together explain only 0.479 (A) / 0.335 (B) of the raw grid's variance and **0.105 / 0.100 of the residualised grid's** -- so 89-90% of the incremental signal lives in *interactions* between the axes. The aggregation behaves as an inseparable whole; there is no 'the operator on axis X' to tune.

**Before vs after: after for layers, before for tokens -- in the raw grid only.** Mean detection AUC by placement code at t=30 ranks ABB=0.749, AAB=0.747 top-2 in corpus A and AAB=0.710, ABB=0.696 top-2 in corpus B: the two corpora agree that the best placement is **layer After + token Before**, and the worst are BBB/BBA (B) and BAA/AAA (A). Marginally, layer-After beats layer-Before by 0.013 (A) / 0.024 (B); token-Before beats token-After by 0.063 (A) but only 0.003 (B); the denoise placement is worth <0.012 either way and flips sign between corpora. **Under residualisation the placement means go flat** (A 0.558-0.566, B 0.549-0.572) and the ordering no longer transfers. The baseline's own placement (layer After + token After) is 5th of 8 in A and 3rd of 8 in B.

### (6) Recommended canonical aggregation

**Keep the current baseline -- all axes After, `mean` everywhere -- and promote it from a default to a declared convention.** The reason is not that it wins; it is 5th of 8 placements and 1907/11700 cells beat it in corpus A. The reasons are:

1. `mean` is the only operator under which the aggregation is **order-invariant** (span 0.0003-0.0005 over the 6 before-orders, versus up to 0.106 for every other operator). With any other operator the result depends on a fourth parameter nobody declares.
2. All-After with `mean` is the only choice for which **block value = mean of per-element values**, so per-layer / per-token decompositions are consistent with the aggregate. (This is the same reason V2 was recommended on 2026-08-29, now verified across the whole 11700-cell space rather than 4 variants.)
3. It is near the middle of the grid, not the max, so quoting it does not import selection bias. The grid maximum (0.848 A / 0.817 B) is a selected number and should never be quoted as the signal's strength.
4. Nothing beats it in both corpora once both required controls are applied.

**Mandatory reporting rule (extends rule 3.2 of the 2026-08-29 sweep).** Any AUC of this signal must declare all four aggregation parameters: placement per axis, operator per axis, scope per axis, and the before-reduction order. Without them the same one-line description spans 0.29-0.32 AUC (section 10), which is larger than every effect the 646-slicing sweep measured.

**One channel worth a follow-up, flagged as suggestive only:** `T[all11/B/min]` at t=25 -- take the element-wise minimum of the 11 token probability vectors *before* the divergence (the routing 'core' every suffix position agrees on), which is nearly independent of the baseline in corpus B (rho 0.19) and reaches 0.69-0.74 raw where the baseline is at chance. It is the state-token channel arriving by a different route, it does not survive the phase control, and it must not be adopted on this evidence.

### Clean negatives

* No **placement** rule transfers between corpora after residualisation: the placement main effects go from 0.104 (A, token) raw to 0.001-0.002 residualised.
* No **operator** choice on any single axis is the dominant factor; the two corpora disagree about which axis leads, and 89-90% of the residual variance is interaction.
* **Cross-corpus sign agreement of the residuals is 57.7%** overall -- above the 48% coin flip the denoise axis produced, below anything one would act on. Its structure is informative and one-sided: token place=Before 63.3% vs token place=After **50.3% (exactly chance)**; token scope all11 61.2% vs action-only 52.4%; token scope state **77.9%**, reproducing the layer-axis agent's 77.3% for the state token almost exactly. All of the reproducible structure in this grid is on the token axis, and specifically in whether the state token is included and whether the tokens are combined before the divergence.
* The raw grid's family-wise p = 0.0050 in both corpora is **uninformative** and stays 0.0050 with the baseline cell deleted -- the family is saturated with near-copies of the baseline (median |within-group rank rho| = 0.796 A / 0.575 B). Only the residualised and joint prices carry information. This is the 2026-08-29 lesson, confirmed again.

## 13. Artefacts

```
/tmp/moe_agg_placement.csv        70200 rows: corpus,placement,layer_op,denoise_op,token_op,
                                  layer_scope,denoise_scope,token_scope,t,auc,auc_resid,n_pos,n_neg
/tmp/moe_agg_placement.md         this report
/tmp/moe_agg_placement_t30.csv    t=30 slice with baseline rank-correlations
/tmp/moe_agg_order_probe.csv      648 rows: before-reduction order sensitivity
/tmp/agg_cube_{A,B}.npz           per-branch x 11700 cfg x 3 t window scores (cache)
/tmp/agg_stats_{A,B}.npz          AUC grids, pointwise p, baseline rho, family nulls
/tmp/agg_draws_{A,B}.npz          200 x 11700 x 3 permutation draws (for the joint null)
/tmp/agg_deepdive.txt /tmp/agg_final.txt /tmp/agg_lenctrl.txt
code: /home/jovyan/work/himoe-vla/agg_placement/{agg_core,agg_analyze,agg_report,
      agg_deepdive,agg_final,agg_lenctrl,agg_order_probe}.py
```

NOTE: /tmp is volatile. The code is in the repo; the cubes rebuild in ~2 min (A) / ~3 min (B).
