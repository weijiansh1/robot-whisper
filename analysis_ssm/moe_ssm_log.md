[A/real] BASELINE t=20: auc_succ=0.4057 det=0.5943 npairs=2933
[A/real] BASELINE t=25: auc_succ=0.6345 det=0.6345 npairs=2933
[A/real] BASELINE t=30: auc_succ=0.7988 det=0.7988 npairs=2933
[A/shuffle] BASELINE t=20: auc_succ=0.4916 det=0.5084 npairs=2933
[A/shuffle] BASELINE t=25: auc_succ=0.5462 det=0.5462 npairs=2933
[A/shuffle] BASELINE t=30: auc_succ=0.6062 det=0.6062 npairs=2933
[B/real] BASELINE t=20: auc_succ=0.6444 det=0.6444 npairs=2008
[B/real] BASELINE t=25: auc_succ=0.5249 det=0.5249 npairs=2008
[B/real] BASELINE t=30: auc_succ=0.7510 det=0.7510 npairs=2008
[B/shuffle] BASELINE t=20: auc_succ=0.6150 det=0.6150 npairs=2008
[B/shuffle] BASELINE t=25: auc_succ=0.5976 det=0.5976 npairs=2008
[B/shuffle] BASELINE t=30: auc_succ=0.5757 det=0.5757 npairs=2008

### Family-corrected increment over the baseline (residualised, max over 117 SSM cells, 200-draw within-group label permutation)
  A t=20: baseline det 0.594 | best residual cell r64d8:zfnorm_w8 det 0.697 (family null 95% = 0.669) p_family=0.010
  A t=25: baseline det 0.635 | best residual cell r64d32:innov_w3 det 0.715 (family null 95% = 0.670) p_family=0.005
  A t=30: baseline det 0.799 | best residual cell r16d4:zfnorm_w8 det 0.675 (family null 95% = 0.673) p_family=0.050
  B t=20: baseline det 0.644 | best residual cell r64d8:nis_cum det 0.620 (family null 95% = 0.615) p_family=0.035
  B t=25: baseline det 0.525 | best residual cell r16d8:innov_w12 det 0.704 (family null 95% = 0.626) p_family=0.005
  B t=30: baseline det 0.751 | best residual cell r32d4:zfnorm_w8 det 0.754 (family null 95% = 0.630) p_family=0.005

### Cross-corpus sign agreement of residualised cells (auc_succ vs 0.5)
  t=20: sign agreement 0.486 over 144 cells; 0.500 over the 72 cells with |AUC-0.5|>0.05 in A
  t=25: sign agreement 0.292 over 144 cells; 0.274 over the 106 cells with |AUC-0.5|>0.05 in A
  t=30: sign agreement 0.625 over 144 cells; 0.716 over the 81 cells with |AUC-0.5|>0.05 in A

### Sentinel 2 (clock) — A, latent r=32 d=8, chunks k<=30
H(t)=3.986 H(T)=2.739 H(t/T)=3.994 H(latent32)=4.867 bits
per-dim MI bits  vs t: max 0.793 mean 0.462 | vs T: max 0.163 | vs t/T: max 0.800
32-state latent MI bits  t 1.696 (0.425 of H(t)) | T 0.460 (0.168) | t/T 1.810 (0.453)
within-group Spearman(nis_w8 @t30, episode length T) = -0.068
within-group Spearman(base_w8 @t30, episode length T) = -0.031

### Sentinel 3 (redundancy vs baseline) — A @ t=30
  innov_w8  rho=+0.690  range [+0.484,+0.798]
  nis_w8    rho=+0.706  range [+0.409,+0.862]
  nis_w12   rho=+0.662  range [+0.407,+0.874]
  pcaz_w8   rho=+0.881  range [+0.792,+0.960]
  dzf_w8    rho=+0.864  range [+0.733,+0.950]

### Prediction 1 (innovation ~ large ||A_k||, small routing change) — A
  innov: b(||A||)=-0.2882 b(route change)=+0.3656 b(interaction)=-0.1747 | rho(.,||A||)=-0.252 partial rho given route change = -0.324  (n=10560)
    quadrant means (z): hiA_loD=-0.318, hiA_hiD=-0.151, loA_loD=-0.237, loA_hiD=+0.878
  nis: b(||A||)=-0.3131 b(route change)=+0.4671 b(interaction)=-0.0979 | rho(.,||A||)=-0.268 partial rho given route change = -0.370  (n=10560)
    quadrant means (z): hiA_loD=-0.472, hiA_hiD=-0.044, loA_loD=-0.248, loA_hiD=+0.942

### Prediction 2 (gain concentrates where best window != 8) — A @ t=30
  g 0 n=96 succ=25 W*=12 base=0.851 nisW8=0.854 gain=+0.003
  g 2 n=80 succ=5 W*= 8 base=0.699 nisW8=0.765 gain=+0.067
  g 3 n=96 succ=87 W*= 8 base=0.920 nisW8=0.556 gain=-0.364
  gain8: W*=8 groups mean -0.1487 (n=2) | W*!=8 groups mean +0.0028 (n=1)
  gain12: W*=8 groups mean -0.1427 (n=2) | W*!=8 groups mean +0.0304 (n=1)

### Sentinel 2 (clock) — B, latent r=32 d=8, chunks k<=30
H(t)=3.986 H(T)=1.786 H(t/T)=3.991 H(latent32)=4.872 bits
per-dim MI bits  vs t: max 1.305 mean 0.905 | vs T: max 0.043 | vs t/T: max 1.023
32-state latent MI bits  t 2.433 (0.610 of H(t)) | T 0.155 (0.087) | t/T 1.975 (0.495)
within-group Spearman(nis_w8 @t30, episode length T) = -0.242
within-group Spearman(base_w8 @t30, episode length T) = -0.475

### Sentinel 3 (redundancy vs baseline) — B @ t=30
  innov_w8  rho=+0.534  range [-0.213,+0.875]
  nis_w8    rho=+0.437  range [-0.288,+0.861]
  nis_w12   rho=+0.369  range [-0.284,+0.842]
  pcaz_w8   rho=+0.784  range [+0.270,+0.956]
  dzf_w8    rho=+0.696  range [+0.315,+0.951]

### Prediction 1 (innovation ~ large ||A_k||, small routing change) — B
  innov: b(||A||)=-0.2153 b(route change)=+0.5934 b(interaction)=-0.1570 | rho(.,||A||)=-0.152 partial rho given route change = -0.253  (n=15360)
    quadrant means (z): hiA_loD=-0.429, hiA_hiD=+0.206, loA_loD=-0.501, loA_hiD=+0.989
  nis: b(||A||)=-0.2426 b(route change)=+0.6065 b(interaction)=-0.1132 | rho(.,||A||)=-0.177 partial rho given route change = -0.289  (n=15360)
    quadrant means (z): hiA_loD=-0.487, hiA_hiD=+0.211, loA_loD=-0.480, loA_hiD=+1.007

### Prediction 2 (gain concentrates where best window != 8) — B @ t=30
  g 0 n=32 succ=14 W*= 5 base=0.917 nisW8=0.627 gain=-0.290
  g 3 n=32 succ=5 W*= 3 base=0.822 nisW8=0.800 gain=-0.022
  g 7 n=32 succ=29 W*=12 base=0.793 nisW8=0.805 gain=+0.011
  g10 n=32 succ=9 W*=20 base=0.850 nisW8=0.633 gain=-0.217
  g13 n=32 succ=9 W*= 8 base=0.783 nisW8=0.855 gain=+0.072
  g16 n=32 succ=28 W*= 3 base=1.000 nisW8=1.000 gain=+0.000
  g20 n=32 succ=23 W*=20 base=0.551 nisW8=0.575 gain=+0.024
  g23 n=32 succ=29 W*=12 base=0.575 nisW8=0.920 gain=+0.345
  g26 n=32 succ=9 W*=20 base=0.739 nisW8=0.754 gain=+0.014
  g29 n=32 succ=28 W*= 8 base=0.866 nisW8=0.634 gain=-0.232
  g39 n=32 succ=4 W*=20 base=0.625 nisW8=0.634 gain=+0.009
  g42 n=32 succ=14 W*= 3 base=0.643 nisW8=0.528 gain=-0.115
  g46 n=32 succ=31 W*= 3 base=0.968 nisW8=0.742 gain=-0.226
  gain8: W*=8 groups mean -0.0798 (n=2) | W*!=8 groups mean -0.0424 (n=11)
  gain12: W*=8 groups mean -0.0900 (n=2) | W*!=8 groups mean -0.0549 (n=11)

## 3. Innovation detection vs the scalar baseline (pre-declared main latent config r=32, d=8)
`auc_succ` = P(score | success > score | failure); `det` = max(auc, 1-auc) (the sign is chosen with the label, as in the previous rounds).
| corpus | t | baseline det | stat | auc_succ | raw det | p_perm raw | resid det | p_perm resid | rho vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 0.594 (p=0.060) | *baseline* | | | | | | |
| A | 20 | 0.594 | innov_w8 | 0.340 | 0.660 | 0.005 | 0.646 | 0.005 | -0.008 |
| A | 20 | 0.594 | nis_w8 | 0.483 | 0.517 | 0.706 | 0.547 | 0.308 | -0.239 |
| A | 20 | 0.594 | nis_w12 | 0.558 | 0.558 | 0.264 | 0.537 | 0.478 | +0.170 |
| A | 20 | 0.594 | zfnorm_w8 | 0.306 | 0.694 | 0.005 | 0.643 | 0.010 | +0.065 |
| A | 25 | 0.635 (p=0.015) | *baseline* | | | | | | |
| A | 25 | 0.635 | innov_w8 | 0.502 | 0.502 | 0.985 | 0.567 | 0.249 | +0.363 |
| A | 25 | 0.635 | nis_w8 | 0.679 | 0.679 | 0.005 | 0.646 | 0.010 | +0.279 |
| A | 25 | 0.635 | nis_w12 | 0.666 | 0.666 | 0.005 | 0.675 | 0.005 | -0.175 |
| A | 25 | 0.635 | zfnorm_w8 | 0.384 | 0.616 | 0.015 | 0.662 | 0.005 | +0.012 |
| A | 30 | 0.799 (p=0.005) | *baseline* | | | | | | |
| A | 30 | 0.799 | innov_w8 | 0.693 | 0.693 | 0.005 | 0.507 | 0.896 | +0.690 |
| A | 30 | 0.799 | nis_w8 | 0.695 | 0.695 | 0.005 | 0.529 | 0.587 | +0.706 |
| A | 30 | 0.799 | nis_w12 | 0.712 | 0.712 | 0.005 | 0.562 | 0.249 | +0.662 |
| A | 30 | 0.799 | zfnorm_w8 | 0.453 | 0.547 | 0.378 | 0.636 | 0.025 | +0.139 |
| B | 20 | 0.644 (p=0.005) | *baseline* | | | | | | |
| B | 20 | 0.644 | innov_w8 | 0.498 | 0.502 | 0.960 | 0.521 | 0.532 | +0.368 |
| B | 20 | 0.644 | nis_w8 | 0.413 | 0.587 | 0.015 | 0.587 | 0.020 | +0.241 |
| B | 20 | 0.644 | nis_w12 | 0.401 | 0.599 | 0.010 | 0.575 | 0.035 | +0.187 |
| B | 20 | 0.644 | zfnorm_w8 | 0.427 | 0.573 | 0.045 | 0.554 | 0.139 | -0.097 |
| B | 25 | 0.525 (p=0.488) | *baseline* | | | | | | |
| B | 25 | 0.525 | innov_w8 | 0.390 | 0.610 | 0.020 | 0.651 | 0.005 | +0.538 |
| B | 25 | 0.525 | nis_w8 | 0.382 | 0.618 | 0.010 | 0.633 | 0.005 | +0.488 |
| B | 25 | 0.525 | nis_w12 | 0.335 | 0.665 | 0.005 | 0.689 | 0.005 | +0.428 |
| B | 25 | 0.525 | zfnorm_w8 | 0.648 | 0.648 | 0.005 | 0.628 | 0.005 | -0.140 |
| B | 30 | 0.751 (p=0.005) | *baseline* | | | | | | |
| B | 30 | 0.751 | innov_w8 | 0.699 | 0.699 | 0.005 | 0.559 | 0.129 | +0.534 |
| B | 30 | 0.751 | nis_w8 | 0.599 | 0.599 | 0.010 | 0.520 | 0.557 | +0.437 |
| B | 30 | 0.751 | nis_w12 | 0.535 | 0.535 | 0.393 | 0.562 | 0.095 | +0.369 |
| B | 30 | 0.751 | zfnorm_w8 | 0.466 | 0.534 | 0.353 | 0.560 | 0.134 | +0.171 |

## 5. Sensitivity to latent rank (raw det AUC, real data)

**A, t=25** (baseline det 0.635)
```
stat   innov_w8  nis_w12  nis_w8
r  d                            
8  4      0.612    0.667   0.720
16 4      0.512    0.597   0.588
   8      0.505    0.646   0.658
32 4      0.505    0.596   0.601
   8      0.502    0.666   0.679
   16     0.583    0.670   0.685
64 8      0.647    0.652   0.708
   16     0.686    0.680   0.710
   32     0.690    0.682   0.711
```

**A, t=30** (baseline det 0.799)
```
stat   innov_w8  nis_w12  nis_w8
r  d                            
8  4      0.595    0.660   0.637
16 4      0.660    0.668   0.670
   8      0.635    0.697   0.698
32 4      0.716    0.702   0.689
   8      0.693    0.712   0.695
   16     0.681    0.711   0.696
64 8      0.717    0.718   0.722
   16     0.704    0.703   0.692
   32     0.670    0.710   0.693
```

**B, t=25** (baseline det 0.525)
```
stat   innov_w8  nis_w12  nis_w8
r  d                            
8  4      0.520    0.539   0.518
16 4      0.585    0.599   0.553
   8      0.664    0.663   0.619
32 4      0.587    0.621   0.586
   8      0.610    0.665   0.618
   16     0.515    0.609   0.562
64 8      0.565    0.625   0.583
   16     0.528    0.609   0.546
   32     0.504    0.612   0.554
```

**B, t=30** (baseline det 0.751)
```
stat   innov_w8  nis_w12  nis_w8
r  d                            
8  4      0.686    0.528   0.560
16 4      0.714    0.644   0.653
   8      0.670    0.502   0.556
32 4      0.723    0.617   0.670
   8      0.699    0.535   0.599
   16     0.727    0.527   0.588
64 8      0.635    0.548   0.585
   16     0.662    0.553   0.596
   32     0.703    0.545   0.606
```

## Sentinel 1 detail: same statistic, real vs order-shuffled
| corpus | t | family max REAL | family max SHUFFLE | cells where real>shuffle | median(real - shuffle) |
|---|---|---|---|---|---|
| A | 20 | 0.730 | 0.716 | 0.44 (144 cells) | -0.009 |
| A | 25 | 0.720 | 0.728 | 0.58 (144 cells) | +0.023 |
| A | 30 | 0.807 | 0.714 | 0.77 (144 cells) | +0.050 |
| B | 20 | 0.681 | 0.905 | 0.32 (144 cells) | -0.031 |
| B | 25 | 0.698 | 0.911 | 0.40 (144 cells) | -0.022 |
| B | 30 | 0.804 | 0.891 | 0.78 (144 cells) | +0.080 |

## Action-input ablation (main config r=32, d=8)
| corpus | held-out mean predictive nll/chunk (with action) | (action zeroed) | delta | stat | det with action | det without |
|---|---|---|---|---|---|---|
| A | 33.194 | 34.658 | -1.4637 | t=20 nis_w8 | 0.517 | 0.570 |
|  |  |  |  | t=20 innov_w8 | 0.660 | 0.676 |
|  |  |  |  | t=20 nis_w12 | 0.558 | 0.523 |
|  |  |  |  | t=25 nis_w8 | 0.679 | 0.638 |
|  |  |  |  | t=25 innov_w8 | 0.502 | 0.504 |
|  |  |  |  | t=25 nis_w12 | 0.666 | 0.603 |
|  |  |  |  | t=30 nis_w8 | 0.695 | 0.765 |
|  |  |  |  | t=30 innov_w8 | 0.693 | 0.706 |
|  |  |  |  | t=30 nis_w12 | 0.712 | 0.769 |
| B | 31.696 | 33.521 | -1.8250 | t=20 nis_w8 | 0.587 | 0.553 |
|  |  |  |  | t=20 innov_w8 | 0.502 | 0.502 |
|  |  |  |  | t=20 nis_w12 | 0.599 | 0.579 |
|  |  |  |  | t=25 nis_w8 | 0.618 | 0.631 |
|  |  |  |  | t=25 innov_w8 | 0.610 | 0.611 |
|  |  |  |  | t=25 nis_w12 | 0.665 | 0.651 |
|  |  |  |  | t=30 nis_w8 | 0.599 | 0.599 |
|  |  |  |  | t=30 innov_w8 | 0.699 | 0.698 |
|  |  |  |  | t=30 nis_w12 | 0.535 | 0.556 |
