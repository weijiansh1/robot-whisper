# Divergence function x temporal aggregation sweep (MoE routing tensor)

**Scope.** Only two reduction stages are varied here: (1) the divergence functional on the
32-dim expert axis and what each chunk is compared against, (2) the temporal aggregation
over chunks and its window width. **Everything the sibling agent owns is held fixed:**
placement = **divergence-first** (divergence evaluated per (layer, token) cell, then averaged
over the 40 cells), back-block HB layers **12-15**, action tokens **1-10**, denoise step **9**.
The two sweeps therefore compose: this grid is one column of the sibling's placement axis.

**Grid.** 26 divergences x 69 temporal specs = **1794 cells** per corpus per t; t in {20,25,30};
2 corpora; raw and baseline-residualised -> **10764 (cell, t, corpus) evaluations per grid**.

**Protocol.** Target = BDDL `success`. Risk set at query index t = branches with T > t; at
t<=30 that is *every* branch in both corpora (A min T=33, B min T=35), so there is no survival
conditioning anywhere in this report. AUC is computed **within group** (`worker` in A,
`init_state_id` in B) and pooled by pair count; groups that are single-class contribute
nothing (A worker 1 = 0/80; B init states 33 and 36 = 32/32, 49 = 0/32).
Contributing branches: **A 117S/155F (2933 pairs)**,
**B 232S/184F (2008 pairs)**.
No hard top-4 expert IDs are used; probabilities only.

**Sign convention.** `auc` in the CSV is the raw **directional** AUC = P(score | success >
score | failure), so cross-corpus sign flips stay visible. The **detection** AUC that compares
to the baseline is `max(auc, 1-auc)`.

**Baseline reproduction.** `hell|prev` + window mean + W=8 on the fixed slice:
**A 0.7988 /
B 0.7510** at t=30
against the quoted 0.795 / 0.759 -> within 0.004 / 0.008. Reproduced.
Its residual-on-itself is exactly 0.5000 in both corpora (arithmetic check on the residualiser).


## 1. The baseline itself across t (directional)

```
t           20      25      30
corpus                        
A       0.4057  0.6345  0.7988
B       0.6444  0.5249  0.7510
```

Direction at t=30 is `auc > 0.5` in both corpora: chunk-to-chunk route change is **higher on successes / lower on failures** (a failing branch's routing goes static). At t=20 corpus A flips sign (0.406) and corpus B is 0.644 -- the two corpora do not even agree on the direction before t=25, consistent with the known absence of lead time.


## 2. Raw grid marginals

### t=20: detection AUC by divergence (over all 69 temporal specs)

```
                   mean           max       
corpus                A      B      A      B
divergence                                  
bc|prev           0.539  0.593  0.607  0.676
bc|runmean        0.540  0.553  0.652  0.638
bhatdist|prev     0.539  0.593  0.606  0.676
bhatdist|runmean  0.540  0.553  0.652  0.638
chi2|prev         0.539  0.593  0.608  0.676
chi2|runmean      0.540  0.553  0.649  0.639
cos|prev          0.539  0.593  0.615  0.678
cos|runmean       0.542  0.558  0.653  0.640
dent_abs          0.542  0.554  0.616  0.624
dent_sgn          0.542  0.541  0.672  0.608
dtop1_abs         0.543  0.566  0.620  0.623
dtop1_sgn         0.533  0.545  0.629  0.628
emd|prev          0.542  0.550  0.659  0.656
emd|runmean       0.555  0.554  0.631  0.677
hell|prev         0.539  0.593  0.611  0.672
hell|runmean      0.543  0.551  0.635  0.627
js|prev           0.539  0.592  0.608  0.675
js|runmean        0.540  0.553  0.650  0.643
l1|prev           0.536  0.591  0.617  0.678
l1|runmean        0.554  0.554  0.684  0.633
l2|prev           0.538  0.593  0.606  0.672
l2|runmean        0.543  0.556  0.632  0.627
symkl|prev        0.539  0.593  0.606  0.676
symkl|runmean     0.540  0.553  0.651  0.637
tv|prev           0.536  0.591  0.617  0.678
tv|runmean        0.554  0.554  0.684  0.633
```

### t=25: detection AUC by divergence (over all 69 temporal specs)

```
                   mean           max       
corpus                A      B      A      B
divergence                                  
bc|prev           0.607  0.541  0.689  0.646
bc|runmean        0.572  0.539  0.651  0.638
bhatdist|prev     0.608  0.541  0.689  0.646
bhatdist|runmean  0.572  0.539  0.651  0.638
chi2|prev         0.607  0.541  0.689  0.645
chi2|runmean      0.572  0.539  0.651  0.638
cos|prev          0.607  0.541  0.685  0.643
cos|runmean       0.572  0.539  0.651  0.639
dent_abs          0.581  0.557  0.675  0.607
dent_sgn          0.546  0.546  0.693  0.621
dtop1_abs         0.568  0.578  0.643  0.622
dtop1_sgn         0.546  0.572  0.653  0.654
emd|prev          0.565  0.547  0.670  0.596
emd|runmean       0.548  0.542  0.614  0.664
hell|prev         0.613  0.538  0.689  0.631
hell|runmean      0.574  0.537  0.657  0.631
js|prev           0.607  0.541  0.689  0.645
js|runmean        0.572  0.539  0.650  0.638
l1|prev           0.607  0.536  0.683  0.633
l1|runmean        0.571  0.537  0.659  0.630
l2|prev           0.610  0.538  0.687  0.629
l2|runmean        0.572  0.539  0.657  0.631
symkl|prev        0.608  0.541  0.689  0.646
symkl|runmean     0.572  0.538  0.650  0.638
tv|prev           0.607  0.536  0.683  0.633
tv|runmean        0.571  0.537  0.659  0.630
```

### t=30: detection AUC by divergence (over all 69 temporal specs)

```
                   mean           max       
corpus                A      B      A      B
divergence                                  
bc|prev           0.727  0.712  0.806  0.791
bc|runmean        0.682  0.633  0.757  0.721
bhatdist|prev     0.727  0.712  0.805  0.791
bhatdist|runmean  0.682  0.633  0.757  0.721
chi2|prev         0.727  0.713  0.805  0.789
chi2|runmean      0.681  0.634  0.756  0.719
cos|prev          0.727  0.713  0.810  0.785
cos|runmean       0.680  0.634  0.754  0.717
dent_abs          0.602  0.604  0.685  0.679
dent_sgn          0.561  0.587  0.704  0.678
dtop1_abs         0.608  0.586  0.689  0.648
dtop1_sgn         0.555  0.600  0.660  0.694
emd|prev          0.690  0.701  0.775  0.780
emd|runmean       0.537  0.631  0.613  0.704
hell|prev         0.730  0.716  0.807  0.787
hell|runmean      0.678  0.632  0.750  0.719
js|prev           0.727  0.712  0.805  0.791
js|runmean        0.681  0.633  0.757  0.721
l1|prev           0.731  0.714  0.808  0.788
l1|runmean        0.671  0.626  0.744  0.702
l2|prev           0.729  0.715  0.806  0.787
l2|runmean        0.675  0.633  0.754  0.717
symkl|prev        0.727  0.712  0.805  0.791
symkl|runmean     0.682  0.633  0.758  0.720
tv|prev           0.731  0.714  0.808  0.788
tv|runmean        0.671  0.626  0.744  0.702
```


### t=20: detection AUC by temporal spec (over all 26 divergences)

```
                       mean           max       
corpus                    A      B      A      B
temporal      window                            
cum_over_t    0       0.595  0.604  0.635  0.678
ewm0.15       3       0.524  0.602  0.570  0.641
              5       0.520  0.587  0.551  0.634
              8       0.524  0.588  0.566  0.655
              12      0.531  0.604  0.573  0.678
              20      0.539  0.602  0.577  0.675
ewm0.30       3       0.529  0.602  0.575  0.644
              5       0.519  0.595  0.559  0.646
              8       0.517  0.596  0.550  0.648
              12      0.516  0.600  0.548  0.656
              20      0.516  0.600  0.547  0.655
ewm0.50       3       0.538  0.592  0.588  0.635
              5       0.531  0.592  0.581  0.640
              8       0.530  0.591  0.579  0.638
              12      0.530  0.592  0.579  0.639
              20      0.530  0.592  0.579  0.639
ewmall0.15    0       0.539  0.602  0.577  0.675
ewmall0.30    0       0.516  0.600  0.547  0.655
ewmall0.50    0       0.530  0.592  0.579  0.639
fracabove_q75 3       0.519  0.554  0.544  0.574
              5       0.524  0.532  0.581  0.558
              8       0.568  0.547  0.645  0.594
              12      0.549  0.551  0.630  0.598
              20      0.599  0.553  0.684  0.610
fracabove_q90 3       0.529  0.546  0.560  0.583
              5       0.537  0.543  0.595  0.613
              8       0.558  0.582  0.588  0.624
              12      0.539  0.587  0.577  0.643
              20      0.552  0.581  0.605  0.633
fracbelow_q10 3       0.521  0.517  0.598  0.562
              5       0.516  0.514  0.577  0.545
              8       0.521  0.531  0.583  0.609
              12      0.548  0.542  0.585  0.638
              20      0.552  0.560  0.593  0.628
fracbelow_q25 3       0.531  0.525  0.603  0.573
              5       0.537  0.522  0.604  0.552
              8       0.524  0.540  0.592  0.591
              12      0.540  0.536  0.590  0.600
              20      0.580  0.549  0.652  0.606
max           3       0.516  0.569  0.547  0.596
              5       0.550  0.540  0.613  0.628
              8       0.567  0.550  0.672  0.640
              12      0.552  0.584  0.646  0.613
              20      0.569  0.544  0.647  0.577
mean          3       0.521  0.599  0.574  0.637
              5       0.523  0.570  0.564  0.599
              8       0.576  0.589  0.615  0.656
              12      0.546  0.610  0.582  0.674
              20      0.595  0.604  0.635  0.678
median        3       0.528  0.583  0.613  0.607
              5       0.518  0.523  0.640  0.557
              8       0.586  0.572  0.631  0.630
              12      0.567  0.555  0.618  0.614
              20      0.592  0.554  0.629  0.612
min           3       0.524  0.569  0.596  0.601
              5       0.529  0.542  0.598  0.566
              8       0.538  0.562  0.607  0.605
              12      0.560  0.531  0.601  0.670
              20      0.560  0.554  0.594  0.576
slope         3       0.535  0.559  0.594  0.585
              5       0.544  0.562  0.592  0.588
              8       0.563  0.530  0.620  0.656
              12      0.514  0.544  0.563  0.594
              20      0.522  0.599  0.563  0.665
std           3       0.528  0.540  0.576  0.582
              5       0.565  0.542  0.659  0.623
              8       0.557  0.572  0.631  0.628
              12      0.551  0.621  0.613  0.677
              20      0.543  0.599  0.594  0.673
```

### t=25: detection AUC by temporal spec (over all 26 divergences)

```
                       mean           max       
corpus                    A      B      A      B
temporal      window                            
cum_over_t    0       0.518  0.559  0.562  0.611
ewm0.15       3       0.602  0.537  0.656  0.622
              5       0.620  0.568  0.672  0.590
              8       0.620  0.536  0.676  0.614
              12      0.619  0.537  0.689  0.608
              20      0.615  0.538  0.686  0.622
ewm0.30       3       0.601  0.525  0.652  0.603
              5       0.612  0.554  0.666  0.596
              8       0.618  0.541  0.679  0.619
              12      0.617  0.540  0.681  0.614
              20      0.617  0.540  0.681  0.614
ewm0.50       3       0.602  0.522  0.649  0.611
              5       0.609  0.525  0.660  0.608
              8       0.609  0.522  0.660  0.612
              12      0.609  0.522  0.661  0.612
              20      0.609  0.522  0.660  0.612
ewmall0.15    0       0.616  0.538  0.686  0.621
ewmall0.30    0       0.617  0.540  0.681  0.614
ewmall0.50    0       0.609  0.522  0.660  0.612
fracabove_q75 3       0.577  0.531  0.629  0.603
              5       0.621  0.539  0.689  0.588
              8       0.591  0.529  0.640  0.614
              12      0.569  0.524  0.629  0.612
              20      0.566  0.537  0.659  0.598
fracabove_q90 3       0.519  0.527  0.554  0.591
              5       0.539  0.527  0.583  0.558
              8       0.535  0.532  0.587  0.625
              12      0.547  0.542  0.623  0.624
              20      0.547  0.574  0.628  0.654
fracbelow_q10 3       0.528  0.531  0.574  0.570
              5       0.548  0.524  0.588  0.580
              8       0.551  0.525  0.589  0.589
              12      0.549  0.535  0.589  0.618
              20      0.542  0.548  0.594  0.647
fracbelow_q25 3       0.571  0.525  0.629  0.608
              5       0.601  0.521  0.689  0.589
              8       0.601  0.526  0.651  0.621
              12      0.595  0.533  0.638  0.573
              20      0.526  0.540  0.591  0.587
max           3       0.618  0.557  0.673  0.643
              5       0.630  0.583  0.658  0.613
              8       0.576  0.541  0.646  0.600
              12      0.589  0.547  0.693  0.609
              20      0.566  0.580  0.685  0.636
mean          3       0.600  0.547  0.653  0.641
              5       0.627  0.575  0.668  0.594
              8       0.596  0.534  0.635  0.618
              12      0.583  0.551  0.655  0.598
              20      0.535  0.553  0.592  0.599
median        3       0.587  0.525  0.646  0.584
              5       0.602  0.536  0.649  0.568
              8       0.596  0.526  0.668  0.602
              12      0.594  0.543  0.686  0.579
              20      0.555  0.560  0.578  0.599
min           3       0.570  0.532  0.668  0.612
              5       0.580  0.534  0.651  0.611
              8       0.568  0.527  0.625  0.596
              12      0.557  0.554  0.641  0.614
              20      0.544  0.583  0.613  0.640
slope         3       0.532  0.546  0.575  0.602
              5       0.530  0.620  0.614  0.646
              8       0.567  0.537  0.631  0.563
              12      0.625  0.542  0.659  0.612
              20      0.633  0.534  0.666  0.602
std           3       0.609  0.591  0.659  0.630
              5       0.633  0.557  0.672  0.613
              8       0.549  0.547  0.633  0.615
              12      0.581  0.540  0.664  0.615
              20      0.532  0.607  0.656  0.664
```

### t=30: detection AUC by temporal spec (over all 26 divergences)

```
                       mean           max       
corpus                    A      B      A      B
temporal      window                            
cum_over_t    0       0.662  0.675  0.743  0.769
ewm0.15       3       0.718  0.708  0.773  0.767
              5       0.723  0.708  0.789  0.769
              8       0.727  0.715  0.801  0.769
              12      0.733  0.712  0.808  0.777
              20      0.733  0.713  0.803  0.778
ewm0.30       3       0.713  0.711  0.767  0.772
              5       0.722  0.711  0.783  0.775
              8       0.725  0.716  0.786  0.780
              12      0.728  0.716  0.790  0.776
              20      0.727  0.716  0.790  0.776
ewm0.50       3       0.705  0.713  0.755  0.773
              5       0.712  0.714  0.767  0.777
              8       0.713  0.714  0.767  0.776
              12      0.713  0.714  0.767  0.776
              20      0.713  0.714  0.767  0.776
ewmall0.15    0       0.732  0.712  0.801  0.778
ewmall0.30    0       0.727  0.716  0.790  0.776
ewmall0.50    0       0.713  0.714  0.767  0.776
fracabove_q75 3       0.677  0.674  0.773  0.762
              5       0.689  0.690  0.756  0.781
              8       0.693  0.683  0.753  0.762
              12      0.712  0.671  0.810  0.742
              20      0.652  0.657  0.759  0.756
fracabove_q90 3       0.601  0.570  0.717  0.696
              5       0.629  0.582  0.728  0.703
              8       0.634  0.585  0.699  0.695
              12      0.636  0.592  0.714  0.709
              20      0.580  0.630  0.696  0.705
fracbelow_q10 3       0.637  0.593  0.736  0.698
              5       0.638  0.610  0.745  0.703
              8       0.628  0.615  0.738  0.695
              12      0.632  0.616  0.754  0.710
              20      0.615  0.618  0.732  0.707
fracbelow_q25 3       0.691  0.638  0.774  0.752
              5       0.672  0.616  0.752  0.773
              8       0.666  0.608  0.747  0.746
              12      0.687  0.600  0.798  0.736
              20      0.672  0.602  0.744  0.747
max           3       0.725  0.682  0.777  0.766
              5       0.721  0.683  0.776  0.748
              8       0.701  0.668  0.746  0.759
              12      0.655  0.610  0.734  0.755
              20      0.623  0.587  0.685  0.744
mean          3       0.719  0.707  0.773  0.764
              5       0.718  0.698  0.789  0.762
              8       0.716  0.695  0.799  0.753
              12      0.730  0.682  0.807  0.769
              20      0.688  0.685  0.758  0.784
median        3       0.703  0.726  0.759  0.791
              5       0.705  0.671  0.775  0.762
              8       0.712  0.615  0.798  0.693
              12      0.711  0.618  0.787  0.710
              20      0.673  0.660  0.766  0.748
min           3       0.697  0.685  0.765  0.768
              5       0.666  0.643  0.769  0.752
              8       0.639  0.628  0.746  0.765
              12      0.637  0.620  0.743  0.755
              20      0.609  0.614  0.685  0.749
slope         3       0.535  0.661  0.579  0.704
              5       0.572  0.668  0.611  0.729
              8       0.660  0.671  0.735  0.746
              12      0.642  0.660  0.704  0.741
              20      0.702  0.650  0.782  0.701
std           3       0.661  0.593  0.705  0.632
              5       0.681  0.661  0.732  0.703
              8       0.677  0.633  0.751  0.678
              12      0.613  0.602  0.676  0.675
              20      0.594  0.609  0.704  0.675
```


## 3. Which stage matters more? (two-way variance decomposition of detection AUC over the 26 x 69 table)

```
 t corpus      grid  frac_divergence  frac_temporal  frac_interaction  sd_total  range_div  range_tmp
20      A       det           0.0262         0.4493            0.5245    0.0330     0.0224     0.0857
20      A det_resid           0.0727         0.2952            0.6321    0.0342     0.0379     0.0713
20      B       det           0.1930         0.3900            0.4169    0.0446     0.0525     0.1064
20      B det_resid           0.1020         0.1594            0.7387    0.0267     0.0224     0.0564
25      A       det           0.1751         0.4105            0.4144    0.0515     0.0670     0.1152
25      A det_resid           0.1502         0.2293            0.6205    0.0400     0.0362     0.0778
25      B       det           0.0871         0.3510            0.5619    0.0345     0.0417     0.0996
25      B det_resid           0.0473         0.3438            0.6088    0.0366     0.0312     0.0932
30      A       det           0.4646         0.2967            0.2387    0.0839     0.1940     0.1982
30      A det_resid           0.1630         0.1826            0.6544    0.0350     0.0359     0.0725
30      B       det           0.3567         0.3384            0.3049    0.0783     0.1304     0.1558
30      B det_resid           0.2247         0.1654            0.6098    0.0400     0.0477     0.0794
```


## 4. Does anything beat the baseline in BOTH corpora?

* **t=20** (baseline A 0.594 / B 0.644): 6/1794 cells beat it in both corpora raw; 0 of those also agree on direction between corpora.
* **t=25** (baseline A 0.635 / B 0.525): 201/1794 cells beat it in both corpora raw; 29 of those also agree on direction between corpora.
* **t=30** (baseline A 0.799 / B 0.751): 28/1794 cells beat it in both corpora raw; 28 of those also agree on direction between corpora.

### Top 15 by min(det_A, det_B), RAW, direction required to match, t=30

```
  l2|prev|ewmall0.15|W0              min=0.7784  A=0.7978 B=0.7784  (raw dir A=0.798 B=0.778)
  hell|prev|ewm0.15|W20              min=0.7779  A=0.7992 B=0.7779  (raw dir A=0.799 B=0.778)
  l2|prev|ewm0.15|W20                min=0.7779  A=0.7995 B=0.7779  (raw dir A=0.800 B=0.778)
  hell|prev|ewm0.15|W12              min=0.7774  A=0.8046 B=0.7774  (raw dir A=0.805 B=0.777)
  l2|prev|ewm0.30|W12                min=0.7764  A=0.7876 B=0.7764  (raw dir A=0.788 B=0.776)
  hell|prev|ewmall0.15|W0            min=0.7759  A=0.7995 B=0.7759  (raw dir A=0.800 B=0.776)
  tv|prev|ewm0.15|W20                min=0.7759  A=0.8029 B=0.7759  (raw dir A=0.803 B=0.776)
  l2|prev|ewmall0.30|W0              min=0.7759  A=0.7869 B=0.7759  (raw dir A=0.787 B=0.776)
  l1|prev|ewm0.15|W20                min=0.7759  A=0.8029 B=0.7759  (raw dir A=0.803 B=0.776)
  l2|prev|ewm0.30|W20                min=0.7759  A=0.7869 B=0.7759  (raw dir A=0.787 B=0.776)
  l2|prev|ewm0.15|W12                min=0.7759  A=0.8033 B=0.7759  (raw dir A=0.803 B=0.776)
  tv|prev|ewm0.15|W12                min=0.7754  A=0.8084 B=0.7754  (raw dir A=0.808 B=0.775)
  l1|prev|ewm0.15|W12                min=0.7754  A=0.8084 B=0.7754  (raw dir A=0.808 B=0.775)
  l1|prev|ewmall0.15|W0              min=0.7749  A=0.8012 B=0.7749  (raw dir A=0.801 B=0.775)
  tv|prev|ewmall0.15|W0              min=0.7749  A=0.8012 B=0.7749  (raw dir A=0.801 B=0.775)
```

### Top 15 by min(det_resid_A, det_resid_B), RESIDUALISED, direction required to match, t=30

```
  hell|prev|fracabove_q75|W12        min=0.6345  A=0.6464 B=0.6345  (raw dir A=0.646 B=0.634)
  chi2|prev|fracabove_q75|W12        min=0.6320  A=0.6444 B=0.6320  (raw dir A=0.644 B=0.632)
  l2|prev|fracabove_q75|W12          min=0.6315  A=0.6458 B=0.6315  (raw dir A=0.646 B=0.631)
  cos|prev|fracabove_q75|W12         min=0.6275  A=0.6604 B=0.6275  (raw dir A=0.660 B=0.627)
  cos|prev|fracabove_q75|W20         min=0.6245  A=0.6311 B=0.6245  (raw dir A=0.631 B=0.625)
  l2|prev|fracabove_q75|W20          min=0.6245  A=0.6273 B=0.6245  (raw dir A=0.627 B=0.625)
  js|prev|fracabove_q75|W12          min=0.6240  A=0.6458 B=0.6240  (raw dir A=0.646 B=0.624)
  bhatdist|prev|fracabove_q75|W12    min=0.6220  A=0.6485 B=0.6220  (raw dir A=0.648 B=0.622)
  bc|prev|fracbelow_q25|W12          min=0.6220  A=0.6485 B=0.6220  (raw dir A=0.648 B=0.622)
  hell|runmean|fracbelow_q10|W3      min=0.6193  A=0.6236 B=0.6193  (raw dir A=0.624 B=0.619)
  symkl|prev|fracabove_q75|W12       min=0.6180  A=0.6485 B=0.6180  (raw dir A=0.648 B=0.618)
  hell|prev|ewm0.15|W12              min=0.6164  A=0.6164 B=0.6335  (raw dir A=0.616 B=0.633)
  js|prev|fracabove_q75|W20          min=0.6147  A=0.6147 B=0.6360  (raw dir A=0.615 B=0.636)
  chi2|prev|fracabove_q75|W20        min=0.6140  A=0.6140 B=0.6340  (raw dir A=0.614 B=0.634)
  l1|prev|ewm0.15|W12                min=0.6135  A=0.6362 B=0.6135  (raw dir A=0.636 B=0.614)
```


## 5. Permutation family-wise p (200 draws, labels shuffled WITHIN group at branch level, one draw reused across all t)

* corpus **A**, **raw** grid, full_family_incl_baseline: argmax `cos|prev|fracabove_q75|W12` at t=30, observed max det-AUC **0.8096**; null max mean 0.6640, p95 0.6980, largest of 200 0.7395 -> **family-wise p = 0.0050**
* corpus **A**, **raw** grid, family_excl_baseline: argmax `cos|prev|fracabove_q75|W12` at t=30, observed max det-AUC **0.8096**; null max mean 0.6640, p95 0.6980, largest of 200 0.7395 -> **family-wise p = 0.0050**
* corpus **A**, **resid** grid, full_family_incl_baseline: argmax `emd|runmean|fracbelow_q10|W8` at t=30, observed max det-AUC **0.6891**; null max mean 0.6727, p95 0.7116, largest of 200 0.7303 -> **family-wise p = 0.1891**
* corpus **A**, **resid** grid, family_excl_baseline: argmax `emd|runmean|fracbelow_q10|W8` at t=30, observed max det-AUC **0.6891**; null max mean 0.6727, p95 0.7116, largest of 200 0.7303 -> **family-wise p = 0.1891**
* corpus **B**, **raw** grid, full_family_incl_baseline: argmax `symkl|prev|median|W3` at t=30, observed max det-AUC **0.7913**; null max mean 0.6198, p95 0.6415, largest of 200 0.6668 -> **family-wise p = 0.0050**
* corpus **B**, **raw** grid, family_excl_baseline: argmax `symkl|prev|median|W3` at t=30, observed max det-AUC **0.7913**; null max mean 0.6198, p95 0.6415, largest of 200 0.6668 -> **family-wise p = 0.0050**
* corpus **B**, **resid** grid, full_family_incl_baseline: argmax `cos|prev|fracabove_q75|W5` at t=30, observed max det-AUC **0.7206**; null max mean 0.6253, p95 0.6470, largest of 200 0.6633 -> **family-wise p = 0.0050**
* corpus **B**, **resid** grid, family_excl_baseline: argmax `cos|prev|fracabove_q75|W5` at t=30, observed max det-AUC **0.7206**; null max mean 0.6253, p95 0.6470, largest of 200 0.6633 -> **family-wise p = 0.0050**


## 6. Cross-corpus sign agreement

* **raw**: overall 60.1% of the 5382 (cell, t) pairs agree on sign; by t: t=20 47.8%, t=25 42.0%, t=30 90.4%
* **residual**: overall 59.8% of the 5382 (cell, t) pairs agree on sign; by t: t=20 63.1%, t=25 43.9%, t=30 72.3%

### Residual sign agreement broken out (t=30)

```
by divergence:
divergence
bc|prev             0.812
bc|runmean          0.696
bhatdist|prev       0.812
bhatdist|runmean    0.696
chi2|prev           0.812
chi2|runmean        0.681
cos|prev            0.841
cos|runmean         0.725
dent_abs            0.754
dent_sgn            0.667
dtop1_abs           0.710
dtop1_sgn           0.710
emd|prev            0.696
emd|runmean         0.362
hell|prev           0.884
hell|runmean        0.623
js|prev             0.812
js|runmean          0.696
l1|prev             0.899
l1|runmean          0.507
l2|prev             0.870
l2|runmean          0.638
symkl|prev          0.812
symkl|runmean       0.681
tv|prev             0.899
tv|runmean          0.507

by temporal operator:
temporal
cum_over_t       0.846
ewm0.15          0.792
ewm0.30          0.623
ewm0.50          0.900
ewmall0.15       0.962
ewmall0.30       0.577
ewmall0.50       0.885
fracabove_q75    0.762
fracabove_q90    0.692
fracbelow_q10    0.846
fracbelow_q25    0.846
max              0.577
mean             0.600
median           0.538
min              0.800
slope            0.608
std              0.738

by window:
window
0     0.817
3     0.793
5     0.713
8     0.698
12    0.689
20    0.692
```


## 7. Pointwise (per-cell) permutation p for the residual leaders

```
  hell|prev|fracabove_q75|W12        A 0.6464 p=0.0050 | B 0.6345 p=0.0050
  chi2|prev|fracabove_q75|W12        A 0.6444 p=0.0050 | B 0.6320 p=0.0050
  l2|prev|fracabove_q75|W12          A 0.6458 p=0.0050 | B 0.6315 p=0.0050
  cos|prev|fracabove_q75|W12         A 0.6604 p=0.0050 | B 0.6275 p=0.0050
  cos|prev|fracabove_q75|W20         A 0.6311 p=0.0149 | B 0.6245 p=0.0050
  l2|prev|fracabove_q75|W20          A 0.6273 p=0.0100 | B 0.6245 p=0.0050
  js|prev|fracabove_q75|W12          A 0.6458 p=0.0050 | B 0.6240 p=0.0050
  bhatdist|prev|fracabove_q75|W12    A 0.6485 p=0.0050 | B 0.6220 p=0.0050
  bc|prev|fracbelow_q25|W12          A 0.6485 p=0.0050 | B 0.6220 p=0.0050
  hell|runmean|fracbelow_q10|W3      A 0.6236 p=0.0299 | B 0.6193 p=0.0050
```


## 8. The prior claim under test: does longer memory beat W=8?

```
  t=20 A: W3=0.524/0.596  W5=0.516/0.624  W8=0.594/0.500  W12=0.529/0.537  W20=0.582/0.538  ewmall0.15=0.527/0.586  ewmall0.30=0.513/0.603  ewmall0.50=0.539/0.615  cum/t=0.582/0.538
  t=20 B: W3=0.634/0.546  W5=0.594/0.514  W8=0.644/0.500  W12=0.666/0.602  W20=0.670/0.592  ewmall0.15=0.671/0.595  ewmall0.30=0.649/0.553  ewmall0.50=0.639/0.549  cum/t=0.670/0.592
  t=25 A: W3=0.653/0.609  W5=0.667/0.606  W8=0.635/0.500  W12=0.655/0.544  W20=0.509/0.521  ewmall0.15=0.686/0.630  ewmall0.30=0.681/0.641  ewmall0.50=0.660/0.618  cum/t=0.530/0.514
  t=25 B: W3=0.549/0.558  W5=0.581/0.639  W8=0.525/0.500  W12=0.541/0.505  W20=0.570/0.584  ewmall0.15=0.513/0.516  ewmall0.30=0.530/0.547  ewmall0.50=0.507/0.508  cum/t=0.577/0.578
  t=30 A: W3=0.771/0.576  W5=0.786/0.526  W8=0.799/0.500  W12=0.807/0.652  W20=0.758/0.588  ewmall0.15=0.800/0.604  ewmall0.30=0.790/0.584  ewmall0.50=0.765/0.579  cum/t=0.743/0.525
  t=30 B: W3=0.763/0.574  W5=0.760/0.617  W8=0.751/0.500  W12=0.769/0.597  W20=0.781/0.618  ewmall0.15=0.776/0.634  ewmall0.30=0.774/0.593  ewmall0.50=0.776/0.583  cum/t=0.764/0.596
```
(each entry is raw / residualised detection AUC)


## 9. Joint cross-corpus test: how many cells clear in BOTH corpora, against a paired null?

A cell/t 'clears' in a corpus if its detection AUC exceeds that same cell's own per-cell 95th-percentile permutation threshold (so the per-cell difficulty is absorbed). It **counts** only if it clears in both corpora *and* the two corpora agree on the sign of the effect. Under the null, permutation draw i of corpus A is paired with draw i of corpus B (the two corpora are independent experiments, so any pairing is valid), giving the null distribution of the joint count.

* **raw** grid (1794 cells x 3 t = 5382 tests): observed joint count **1356**; null mean 5.5, p95 29, max of 200 93; **p = 0.0050**
* **resid** grid (1794 cells x 3 t = 5382 tests): observed joint count **128**; null mean 5.8, p95 24, max of 200 76; **p = 0.0050**

  Cells clearing in both (residual grid):

```
    t=30  hell|prev|fracabove_q75|W12        min=0.6345  A=0.6464 B=0.6345
    t=30  chi2|prev|fracabove_q75|W12        min=0.6320  A=0.6444 B=0.6320
    t=30  l2|prev|fracabove_q75|W12          min=0.6315  A=0.6458 B=0.6315
    t=30  cos|prev|fracabove_q75|W12         min=0.6275  A=0.6604 B=0.6275
    t=30  l2|prev|fracabove_q75|W20          min=0.6245  A=0.6273 B=0.6245
    t=30  cos|prev|fracabove_q75|W20         min=0.6245  A=0.6311 B=0.6245
    t=30  js|prev|fracabove_q75|W12          min=0.6240  A=0.6458 B=0.6240
    t=30  bhatdist|prev|fracabove_q75|W12    min=0.6220  A=0.6485 B=0.6220
    t=30  bc|prev|fracbelow_q25|W12          min=0.6220  A=0.6485 B=0.6220
    t=30  hell|runmean|fracbelow_q10|W3      min=0.6193  A=0.6236 B=0.6193
    t=30  symkl|prev|fracabove_q75|W12       min=0.6180  A=0.6485 B=0.6180
    t=25  dent_abs|fracabove_q90|W20         min=0.6165  A=0.6185 B=0.6165
    t=30  hell|prev|ewm0.15|W12              min=0.6164  A=0.6164 B=0.6335
    t=25  dtop1_sgn|fracabove_q90|W3         min=0.6153  A=0.6158 B=0.6153
    t=25  dtop1_sgn|slope|W12                min=0.6150  A=0.6171 B=0.6150
    t=30  js|prev|fracabove_q75|W20          min=0.6147  A=0.6147 B=0.6360
    t=30  chi2|prev|fracabove_q75|W20        min=0.6140  A=0.6140 B=0.6340
    t=30  tv|runmean|fracbelow_q10|W5        min=0.6135  A=0.6140 B=0.6135
    t=30  tv|prev|ewm0.15|W12                min=0.6135  A=0.6362 B=0.6135
    t=30  l1|runmean|fracbelow_q10|W5        min=0.6135  A=0.6140 B=0.6135
    t=30  l1|prev|ewm0.15|W12                min=0.6135  A=0.6362 B=0.6135
    t=30  cos|prev|fracabove_q75|W3          min=0.6134  A=0.6134 B=0.6838
    t=30  cos|runmean|fracbelow_q10|W3       min=0.6130  A=0.6130 B=0.6277
    t=30  bhatdist|prev|fracabove_q75|W20    min=0.6130  A=0.6130 B=0.6330
    t=30  symkl|prev|fracabove_q75|W20       min=0.6123  A=0.6123 B=0.6300
    t=30  bc|prev|fracbelow_q25|W20          min=0.6120  A=0.6120 B=0.6350
    t=20  tv|prev|slope|W20                  min=0.6096  A=0.6117 B=0.6096
    t=20  l1|prev|slope|W20                  min=0.6096  A=0.6117 B=0.6096
    t=30  cos|prev|mean|W12                  min=0.6096  A=0.6331 B=0.6096
    t=30  tv|runmean|fracbelow_q10|W3        min=0.6081  A=0.6168 B=0.6081
    t=30  tv|prev|ewm0.15|W20                min=0.6081  A=0.6110 B=0.6081
    t=30  l1|runmean|fracbelow_q10|W3        min=0.6081  A=0.6168 B=0.6081
    t=30  l1|prev|ewm0.15|W20                min=0.6081  A=0.6110 B=0.6081
    t=30  tv|runmean|fracbelow_q10|W8        min=0.6079  A=0.6079 B=0.6116
    t=30  l1|runmean|fracbelow_q10|W8        min=0.6079  A=0.6079 B=0.6116
    t=25  emd|prev|mean|W8                   min=0.6076  A=0.6263 B=0.6076
    t=30  hell|prev|fracabove_q75|W20        min=0.6076  A=0.6076 B=0.6469
    t=30  chi2|prev|mean|W12                 min=0.6066  A=0.6468 B=0.6066
    t=30  l2|prev|ewm0.15|W12                min=0.6065  A=0.6065 B=0.6185
    t=30  symkl|prev|mean|W12                min=0.6061  A=0.6454 B=0.6061
```


## 10. Structured breakdown of the divergence axis

The 26 divergences are not 26 independent choices. They fall into three families:
**A** the 11 pairwise metrics against the previous chunk, **B** the same 11 against the branch's own running mean, **C** the 4 scalar-change measures (entropy / top-1 mass, absolute and signed).

```

  t=20 corpus A
                        det        det_resid       
                       mean    max      mean    max
  fam                                              
  A:pairwise|prev     0.539  0.659     0.565  0.625
  B:pairwise|runmean  0.545  0.684     0.556  0.671
  C:scalar-change     0.540  0.672     0.541  0.689

  t=20 corpus B
                        det        det_resid       
                       mean    max      mean    max
  fam                                              
  A:pairwise|prev     0.589  0.678     0.545  0.620
  B:pairwise|runmean  0.554  0.677     0.531  0.667
  C:scalar-change     0.552  0.628     0.530  0.586

  t=25 corpus A
                        det        det_resid       
                       mean    max      mean    max
  fam                                              
  A:pairwise|prev     0.604  0.689     0.582  0.674
  B:pairwise|runmean  0.570  0.659     0.552  0.677
  C:scalar-change     0.560  0.693     0.557  0.685

  t=25 corpus B
                        det        det_resid       
                       mean    max      mean    max
  fam                                              
  A:pairwise|prev     0.540  0.646     0.545  0.653
  B:pairwise|runmean  0.539  0.664     0.552  0.678
  C:scalar-change     0.563  0.654     0.559  0.657

  t=30 corpus A
                        det        det_resid       
                       mean    max      mean    max
  fam                                              
  A:pairwise|prev     0.725  0.810     0.563  0.660
  B:pairwise|runmean  0.665  0.758     0.538  0.689
  C:scalar-change     0.582  0.704     0.539  0.669

  t=30 corpus B
                        det        det_resid       
                       mean    max      mean    max
  fam                                              
  A:pairwise|prev     0.712  0.791     0.576  0.721
  B:pairwise|runmean  0.632  0.721     0.540  0.655
  C:scalar-change     0.594  0.694     0.540  0.617
```

### Spread *within* family A (the 11 f-divergences against the previous chunk), t=30, best temporal spec per metric

```
                  det         det_resid        
corpus              A       B         A       B
divergence                                     
bc|prev        0.8057  0.7908    0.6485  0.7126
bhatdist|prev  0.8053  0.7908    0.6485  0.7126
chi2|prev      0.8050  0.7893    0.6481  0.7156
cos|prev       0.8096  0.7854    0.6604  0.7206
emd|prev       0.7746  0.7799    0.6151  0.6559
hell|prev      0.8074  0.7874    0.6604  0.7181
js|prev        0.8050  0.7908    0.6485  0.7126
l1|prev        0.8084  0.7878    0.6516  0.7077
l2|prev        0.8057  0.7869    0.6567  0.7136
symkl|prev     0.8050  0.7913    0.6485  0.7052
tv|prev        0.8084  0.7878    0.6516  0.7077
```


## 11. Divergence vs temporal: variance decomposition, with and without the near-degenerate metric family

```
 t corpus      grid                 scope  frac_div  frac_temporal  frac_inter  range_div  range_temporal
20      A       det    all_26_divergences    0.0262         0.4493      0.5245     0.0224          0.0857
20      A       det only_pairwise_prev_11    0.0032         0.8229      0.1739     0.0059          0.0947
20      A det_resid    all_26_divergences    0.0727         0.2952      0.6321     0.0379          0.0713
20      A det_resid only_pairwise_prev_11    0.0018         0.8342      0.1640     0.0058          0.1023
20      B       det    all_26_divergences    0.1930         0.3900      0.4169     0.0525          0.1064
20      B       det only_pairwise_prev_11    0.0597         0.8428      0.0974     0.0431          0.1621
20      B det_resid    all_26_divergences    0.1020         0.1594      0.7387     0.0224          0.0564
20      B det_resid only_pairwise_prev_11    0.0369         0.6869      0.2762     0.0196          0.0915
25      A       det    all_26_divergences    0.1751         0.4105      0.4144     0.0670          0.1152
25      A       det only_pairwise_prev_11    0.0473         0.8619      0.0908     0.0479          0.1736
25      A det_resid    all_26_divergences    0.1502         0.2293      0.6205     0.0362          0.0778
25      A det_resid only_pairwise_prev_11    0.0402         0.6458      0.3140     0.0302          0.1412
25      B       det    all_26_divergences    0.0871         0.3510      0.5619     0.0417          0.0996
25      B       det only_pairwise_prev_11    0.0080         0.7790      0.2130     0.0108          0.1283
25      B det_resid    all_26_divergences    0.0473         0.3438      0.6088     0.0312          0.0932
25      B det_resid only_pairwise_prev_11    0.0238         0.7487      0.2276     0.0202          0.1363
30      A       det    all_26_divergences    0.4646         0.2967      0.2387     0.1940          0.1982
30      A       det only_pairwise_prev_11    0.0280         0.8897      0.0824     0.0417          0.2780
30      A det_resid    all_26_divergences    0.1630         0.1826      0.6544     0.0359          0.0725
30      A det_resid only_pairwise_prev_11    0.0638         0.7203      0.2159     0.0304          0.1345
30      B       det    all_26_divergences    0.3567         0.3384      0.3049     0.1304          0.1558
30      B       det only_pairwise_prev_11    0.0029         0.8756      0.1215     0.0148          0.2433
30      B det_resid    all_26_divergences    0.2247         0.1654      0.6098     0.0477          0.0794
30      B det_resid only_pairwise_prev_11    0.0053         0.7748      0.2198     0.0117          0.1867
```


## 12. Why the raw grid is not 1794 different quantities: rank correlation with the baseline (t=30, within group, pooled)

```
                            mean    50%    max
corpus fam                                    
A      A:pairwise|prev     0.640  0.678  1.000
       B:pairwise|runmean  0.514  0.564  0.801
       C:scalar-change     0.235  0.239  0.476
B      A:pairwise|prev     0.613  0.640  1.000
       B:pairwise|runmean  0.420  0.419  0.749
       C:scalar-change     0.159  0.157  0.335

  |rho| >= 0.9 with the baseline: A 38/1794, B 50/1794
  |rho| >= 0.7 with the baseline: A 427/1794, B 356/1794
```


## 13. Do the residual leaders survive at other t? (the previous round's failure mode)

```
  hell|prev|fracabove_q75|W12        t20: A=0.570 B=0.531  t25: A=0.592 B=0.510  t30: A=0.646 B=0.634
  chi2|prev|fracabove_q75|W12        t20: A=0.557 B=0.541  t25: A=0.565 B=0.513  t30: A=0.644 B=0.632
  l2|prev|fracabove_q75|W12          t20: A=0.560 B=0.532  t25: A=0.588 B=0.500  t30: A=0.646 B=0.631
  cos|prev|fracabove_q75|W12         t20: A=0.578 B=0.532  t25: A=0.584 B=0.515  t30: A=0.660 B=0.627
  cos|prev|fracabove_q75|W20         t20: A=0.525 B=0.527  t25: A=0.526 B=0.534  t30: A=0.631 B=0.625
  l2|prev|fracabove_q75|W20          t20: A=0.535 B=0.527  t25: A=0.522 B=0.542  t30: A=0.627 B=0.625
  js|prev|fracabove_q75|W12          t20: A=0.557 B=0.540  t25: A=0.565 B=0.514  t30: A=0.646 B=0.624
  bhatdist|prev|fracabove_q75|W12    t20: A=0.557 B=0.538  t25: A=0.565 B=0.517  t30: A=0.648 B=0.622
  bc|prev|fracbelow_q25|W12          t20: A=0.558 B=0.542  t25: A=0.566 B=0.517  t30: A=0.648 B=0.622
  hell|runmean|fracbelow_q10|W3      t20: A=0.631 B=0.514  t25: A=0.504 B=0.600  t30: A=0.624 B=0.619
  symkl|prev|fracabove_q75|W12       t20: A=0.555 B=0.538  t25: A=0.566 B=0.525  t30: A=0.648 B=0.618
  hell|prev|ewm0.15|W12              t20: A=0.602 B=0.602  t25: A=0.630 B=0.524  t30: A=0.616 B=0.633
```


## 14. The one survivor in close-up: burst-count aggregation (`fracabove_q75` / its mirror `fracbelow_q25` on a similarity)

`fracabove_q75|W` = fraction of the W chunk transitions in the window whose divergence exceeds a **global, label-free** 75th-percentile threshold (one number per corpus per divergence, estimated over all branches x steps 1..30). It counts *bursts* of route change rather than averaging the amount of it.

```
                         det         det_resid        
corpus                     A       B         A       B
temporal      window                                  
fracabove_q75 3       0.7138  0.7622    0.5919  0.6858
              5       0.7451  0.7814    0.5499  0.7181
              8       0.7433  0.7617    0.5128  0.6728
              12      0.8011  0.7420    0.6464  0.6345
              20      0.7518  0.7562    0.6076  0.6469
fracabove_q90 3       0.5992  0.5847    0.5765  0.5139
              5       0.6388  0.5904    0.5363  0.5276
              8       0.6343  0.5725    0.5442  0.5481
              12      0.6602  0.5964    0.5943  0.5050
              20      0.5426  0.6001    0.5650  0.5299
mean          3       0.7709  0.7634    0.5759  0.5737
              5       0.7859  0.7600    0.5264  0.6170
              8       0.7988  0.7510    0.5000  0.5000
              12      0.8074  0.7689    0.6522  0.5966
              20      0.7576  0.7814    0.5878  0.6175
```


## 15. DECISIVE TEST: does the burst operator survive a *richer* baseline?

B1 = the stated baseline only (`hell|prev` window mean W=8).
B2 = every window mean (W=3,5,8,12,20) -- kills anything that is just 'a longer memory'.
B3 = B2 + the window std / max / min / median / full-history EWM at W=8 -- kills anything that is just 'the mean plus its spread or its extremes'.
All residualisation is label-free rank OLS within group. Entries are detection AUC of the residual.

```

  t=20
  baseline_set                B1_stated        B2_all_window_means        B3_means+moments       
  corpus                              A      B                   A      B                A      B
  cell                                                                                           
  cos|prev|fracabove_q75|W12      0.578  0.532               0.515  0.520            0.560  0.507
  dent_abs|mean|W8                0.539  0.520               0.537  0.527            0.511  0.501
  dtop1_abs|mean|W8               0.534  0.520               0.527  0.525            0.520  0.550
  hell|prev|ewm0.15|W12           0.602  0.602               0.523  0.527            0.526  0.513
  hell|prev|fracabove_q75|W12     0.570  0.531               0.527  0.511            0.576  0.501
  hell|prev|fracabove_q75|W20     0.506  0.535               0.522  0.506            0.565  0.502
  hell|prev|fracabove_q90|W12     0.501  0.502               0.544  0.523            0.537  0.514
  hell|prev|fracbelow_q10|W12     0.523  0.538               0.582  0.501            0.587  0.524
  hell|prev|max|W3                0.587  0.515               0.561  0.542            0.543  0.503
  hell|prev|median|W3             0.592  0.533               0.556  0.509            0.562  0.502
  hell|prev|slope|W20             0.620  0.589               0.507  0.548            0.509  0.527
  hell|runmean|mean|W8            0.541  0.535               0.530  0.582            0.520  0.568

  t=25
  baseline_set                B1_stated        B2_all_window_means        B3_means+moments       
  corpus                              A      B                   A      B                A      B
  cell                                                                                           
  cos|prev|fracabove_q75|W12      0.584  0.515               0.562  0.510            0.531  0.516
  dent_abs|mean|W8                0.522  0.558               0.512  0.511            0.501  0.513
  dtop1_abs|mean|W8               0.509  0.611               0.504  0.571            0.512  0.502
  hell|prev|ewm0.15|W12           0.630  0.524               0.632  0.569            0.521  0.509
  hell|prev|fracabove_q75|W12     0.592  0.510               0.566  0.528            0.526  0.506
  hell|prev|fracabove_q75|W20     0.523  0.545               0.571  0.544            0.549  0.533
  hell|prev|fracabove_q90|W12     0.581  0.556               0.607  0.508            0.599  0.530
  hell|prev|fracbelow_q10|W12     0.502  0.535               0.554  0.527            0.517  0.527
  hell|prev|max|W3                0.635  0.580               0.598  0.502            0.502  0.512
  hell|prev|median|W3             0.597  0.528               0.529  0.530            0.512  0.533
  hell|prev|slope|W20             0.588  0.514               0.537  0.607            0.520  0.622
  hell|runmean|mean|W8            0.557  0.553               0.552  0.547            0.545  0.516

  t=30
  baseline_set                B1_stated        B2_all_window_means        B3_means+moments       
  corpus                              A      B                   A      B                A      B
  cell                                                                                           
  cos|prev|fracabove_q75|W12      0.660  0.627               0.565  0.555            0.549  0.500
  dent_abs|mean|W8                0.522  0.534               0.509  0.520            0.505  0.519
  dtop1_abs|mean|W8               0.551  0.575               0.523  0.583            0.512  0.590
  hell|prev|ewm0.15|W12           0.616  0.633               0.549  0.512            0.528  0.554
  hell|prev|fracabove_q75|W12     0.646  0.634               0.547  0.558            0.521  0.507
  hell|prev|fracabove_q75|W20     0.608  0.647               0.555  0.612            0.544  0.567
  hell|prev|fracabove_q90|W12     0.594  0.505               0.537  0.510            0.514  0.507
  hell|prev|fracbelow_q10|W12     0.618  0.570               0.552  0.568            0.516  0.558
  hell|prev|max|W3                0.610  0.546               0.543  0.503            0.528  0.510
  hell|prev|median|W3             0.544  0.630               0.591  0.572            0.573  0.566
  hell|prev|slope|W20             0.569  0.501               0.548  0.537            0.517  0.571
  hell|runmean|mean|W8            0.551  0.510               0.562  0.506            0.582  0.522

  pointwise permutation p under B3 at t=30:
  corpus                            A       B
  cell                                       
  cos|prev|fracabove_q75|W12   0.3532  1.0000
  dent_abs|mean|W8             0.9453  0.5721
  dtop1_abs|mean|W8            0.8010  0.0149
  hell|prev|ewm0.15|W12        0.5970  0.1592
  hell|prev|fracabove_q75|W12  0.6716  0.8557
  hell|prev|fracabove_q75|W20  0.4229  0.0896
  hell|prev|fracabove_q90|W12  0.7761  0.8607
  hell|prev|fracbelow_q10|W12  0.7711  0.1493
  hell|prev|max|W3             0.6219  0.7861
  hell|prev|median|W3          0.2040  0.0348
  hell|prev|slope|W20          0.7512  0.0547
  hell|runmean|mean|W8         0.0796  0.5423
```


## 16. Is the burst operator an artefact of the particular threshold?

```
                   det         det_resid        
corpus               A       B         A       B
thresh                                          
branch_own_q75  0.7639  0.6265    0.6338  0.5055
global_q0.50    0.7755  0.6803    0.6406  0.5378
global_q0.60    0.7924  0.7490    0.6502  0.6355
global_q0.70    0.7765  0.7535    0.5987  0.6290
global_q0.75    0.8011  0.7420    0.6464  0.6345
global_q0.80    0.7913  0.7286    0.6744  0.6270
global_q0.90    0.6602  0.5964    0.5943  0.5050
global_q0.95    0.5586  0.5483    0.5871  0.5204
```


## 17. The 11 f-divergences are one quantity, not eleven

```
  corpus A, Spearman vs hell|prev|mean|W8 (t=30): hell=+1.0000  bc=-0.9924  bhatdist=+0.9923  symkl=+0.9917  js=+0.9929  tv=+0.9990  l1=+0.9990  l2=+0.9995  cos=+0.9889  chi2=+0.9938  emd=+0.9522
  corpus B, Spearman vs hell|prev|mean|W8 (t=30): hell=+1.0000  bc=-0.9969  bhatdist=+0.9968  symkl=+0.9967  js=+0.9970  tv=+0.9988  l1=+0.9988  l2=+0.9997  cos=+0.9955  chi2=+0.9973  emd=+0.8945
```


## 18. Family-wise p on the DE-DUPLICATED family (414 cells: one representative metric `hell` x both references + the 4 scalar-change measures x 69 temporal specs)

Justified by section 17: the 11 f-divergences are rank-correlated 0.97-1.00 with each other, so counting them as 11 members inflates the family-max null. This family is the one that would have been pre-specified knowing that.

* corpus **A**, **raw**: argmax `hell|prev|mean|W12` at t=30, observed **0.8074**, null p95 0.6980, null max 0.7395 -> **p = 0.0050**
* corpus **A**, **resid**: argmax `dent_sgn|max|W8` at t=20, observed **0.6891**, null p95 0.7028, null max 0.7303 -> **p = 0.1443**
* corpus **B**, **raw**: argmax `hell|prev|median|W3` at t=30, observed **0.7874**, null p95 0.6390, null max 0.6604 -> **p = 0.0050**
* corpus **B**, **resid**: argmax `hell|prev|fracabove_q75|W5` at t=30, observed **0.7181**, null p95 0.6409, null max 0.6574 -> **p = 0.0050**


## 19. (superseded -- see section 21)

The first pass of this diagnostic divided by a zero-variance rank vector in corpus B and returned NaN. Section 21 is the corrected version; read that one.


## 20. The whole grid re-residualised on the richer baselines

Sections 5/9 residualised on the single stated baseline. If the surviving cells are really only reading the *window width* of the same shared factor, they must collapse once every window mean is in the baseline. Full 1794-cell grid, same 200-draw within-group permutation null.

### 20a. Family-max test

* corpus **A**, baseline set **B1_stated**: argmax `emd|runmean|fracbelow_q10|W8` at t=30, observed max det-AUC **0.6891**, null p95 0.7116, null max 0.7303 -> **family-wise p = 0.1891**
* corpus **A**, baseline set **B2_all_window_means**: argmax `l2|prev|mean|W20` at t=25, observed max det-AUC **0.7180**, null p95 0.7112, null max 0.7460 -> **family-wise p = 0.0398**
* corpus **A**, baseline set **B3_means+moments**: argmax `emd|prev|fracabove_q90|W5` at t=25, observed max det-AUC **0.6925**, null p95 0.7095, null max 0.7293 -> **family-wise p = 0.2090**
* corpus **B**, baseline set **B1_stated**: argmax `cos|prev|fracabove_q75|W5` at t=30, observed max det-AUC **0.7206**, null p95 0.6470, null max 0.6633 -> **family-wise p = 0.0050**
* corpus **B**, baseline set **B2_all_window_means**: argmax `tv|prev|fracabove_q75|W5` at t=30, observed max det-AUC **0.6693**, null p95 0.6474, null max 0.6803 -> **family-wise p = 0.0100**
* corpus **B**, baseline set **B3_means+moments**: argmax `tv|prev|fracabove_q75|W5` at t=30, observed max det-AUC **0.6604**, null p95 0.6554, null max 0.6691 -> **family-wise p = 0.0299**

### 20b. Joint cross-corpus clearing count (the test that survived in section 9)

* baseline set **B1_stated**: observed joint count **128** / 5382; null mean 5.8, p95 24, max of 200 76 -> **p = 0.0050**
  best: hell|prev|fracabove_q75|W12@t30 A=0.646 B=0.634; chi2|prev|fracabove_q75|W12@t30 A=0.644 B=0.632; l2|prev|fracabove_q75|W12@t30 A=0.646 B=0.631; cos|prev|fracabove_q75|W12@t30 A=0.660 B=0.627; l2|prev|fracabove_q75|W20@t30 A=0.627 B=0.625; cos|prev|fracabove_q75|W20@t30 A=0.631 B=0.625
* baseline set **B2_all_window_means**: observed joint count **42** / 5382; null mean 6.3, p95 22, max of 200 82 -> **p = 0.0149**
  best: cos|prev|fracabove_q75|W5@t30 A=0.629 B=0.660; l1|prev|slope|W8@t30 A=0.624 B=0.624; tv|prev|slope|W8@t30 A=0.624 B=0.624; cos|prev|fracabove_q75|W3@t30 A=0.621 B=0.640; l2|prev|slope|W8@t30 A=0.620 B=0.624; cos|prev|slope|W8@t30 A=0.613 B=0.616
* baseline set **B3_means+moments**: observed joint count **16** / 5382; null mean 6.4, p95 19, max of 200 151 -> **p = 0.0846**
  best: tv|prev|slope|W8@t30 A=0.634 B=0.622; l1|prev|slope|W8@t30 A=0.634 B=0.622; cos|prev|slope|W8@t30 A=0.620 B=0.614; hell|prev|slope|W8@t30 A=0.628 B=0.614; l2|prev|slope|W8@t30 A=0.632 B=0.613; cos|prev|fracabove_q75|W5@t30 A=0.613 B=0.630

## 21. Phase-confound diagnostic, corrected

```
  corpus A (n=352 branches in groups where T actually varies): baseline=-0.079  burst=+0.078  burst_resid=+0.140
  corpus B (n=480 branches in groups where T actually varies): baseline=-0.475  burst=-0.392  burst_resid=-0.146
```

Read this honestly, and note the two corpora differ. In **corpus A** the baseline's within-group correlation with episode length is small (-0.079), so at t=30 it is not mainly a duration readout. In **corpus B** it is **-0.475**: within an initial state, branches with more chunk-to-chunk route change end sooner, and since successes end sooner this is a substantial phase/duration channel. That is a property of **the baseline itself**, inherited by every cell in this grid -- it is the same length confound flagged in the 08-28 audit and in section 3.3 of the previous round, and nothing in this sweep addresses it. The one thing the diagnostic does say is that the burst operator's *incremental* part is less phase-loaded than the baseline in B (-0.146 vs -0.475), which is the opposite of the direction that would have explained it away -- but section 15 already shows that increment is not real.


## 22. Recommendation table (raw detection AUC; the residual column is w.r.t. the stated baseline)

```
                                                    note  t20_A  t20_B  t25_A  t25_B  t30_A  res30_A  t30_B  res30_B
spec                                                                                                                
hell|prev|mean|W8                      BASELINE (stated)  0.594  0.644  0.635  0.525  0.799    0.500  0.751    0.500
hell|prev|mean|W12                         longer window  0.529  0.666  0.655  0.541  0.807    0.652  0.769    0.597
hell|prev|ewmall0.15|W0          full-history EWM a=0.15  0.527  0.671  0.686  0.513  0.800    0.604  0.776    0.634
hell|prev|ewm0.15|W12          windowed EWM a=0.15, W=12  0.513  0.672  0.689  0.507  0.805    0.616  0.777    0.633
hell|prev|fracabove_q75|W12  burst count (the near-miss)  0.504  0.596  0.629  0.503  0.801    0.646  0.742    0.634
hell|prev|median|W3                         short median  0.531  0.602  0.645  0.505  0.756    0.544  0.787    0.630
hell|prev|max|W3                               short max  0.519  0.555  0.673  0.582  0.777    0.610  0.724    0.546
hell|prev|std|W8                              window std  0.568  0.561  0.539  0.518  0.613    0.514  0.548    0.513
hell|prev|slope|W8                          window slope  0.553  0.508  0.563  0.552  0.729    0.573  0.744    0.611
hell|prev|cum_over_t|W0                   cumulative / t  0.582  0.670  0.530  0.577  0.743    0.525  0.764    0.596
emd|prev|mean|W8                EMD instead of Hellinger  0.523  0.601  0.583  0.541  0.766    0.552  0.742    0.549
hell|runmean|mean|W8                 vs own running mean  0.575  0.540  0.596  0.529  0.700    0.551  0.652    0.510
dent_abs|mean|W8                        |entropy change|  0.505  0.584  0.614  0.580  0.644    0.522  0.643    0.534
dtop1_abs|mean|W8                    |top-1 mass change|  0.541  0.581  0.546  0.618  0.675    0.551  0.642    0.575
```


## 23. Verdict

**Clean negative on both stages, with one instructive near-miss.**

1. **The divergence functional is a free parameter with no content.** The ten f-divergences
   (Hellinger, Bhattacharyya coefficient and distance, symmetric KL, Jensen-Shannon, total
   variation / L1, L2, cosine, chi-squared) are rank-correlated **0.989-1.000** with each other
   on the same temporal spec in both corpora (section 17). Their best detection AUCs at t=30
   span **0.805-0.810 (A)** and **0.785-0.791 (B)** -- a spread of 0.005, i.e. nothing. Only the
   earth-mover distance separates, and it separates *downwards* (rho 0.89 in B, det 0.775/0.780),
   which is expected: it is the only one of the eleven that depends on the arbitrary numbering of
   the 32 experts. Restricted to this family, the divergence axis explains **0.3-6%** of the
   variance in the grid while the temporal axis explains **65-89%** (section 11).

2. **What each chunk is compared *against* does matter, and the current choice is already the
   right one.** Previous-chunk beats the branch's own running mean and beats the scalar-change
   measures (|entropy change|, |top-1 mass change|) in both corpora at every t
   (t=30 best det: 0.810/0.791 vs 0.758/0.721 vs 0.704/0.694), and its cross-corpus residual sign
   agreement is far better (0.70-0.90 for `|prev` vs 0.36-0.73 for `|runmean`). The apparent
   importance of "the divergence axis" in the unrestricted variance decomposition (36-46% at
   t=30) is entirely this reference contrast, not the choice of metric.

3. **The temporal aggregation has the largest spread of anything measured, but the spread is
   almost all downside.** Marginal detection AUC across temporal specs ranges over 0.24-0.28 AUC
   at t=30 -- larger than any layer/token/denoise effect in the previous sweep, which is why it
   had to be searched. But the window mean is already near the top of that range: the best
   temporal spec in the whole grid beats it by only **+0.011 (A, `cos|prev|fracabove_q75|W12`
   0.8096)** and **+0.040 (B, `symkl|prev|median|W3` 0.7913)** raw. Note the two corpora do not
   even agree on *which direction* the improvement lies in: corpus A prefers longer memory
   (W=12, EWM alpha=0.15), corpus B's raw winner is a **3-step median** while its runner-up is a
   20-step mean -- the classic signature of picking noise. Averaged over divergences, windowed
   max, min, std, slope and the quantile-crossing counts are all worse than the mean.

4. **Nothing clears in both corpora after residualisation.** Family-max on the residual grid:
   **A p = 0.189 (fails), B p = 0.005 (passes)**, and the two corpora peak at different
   statistics. De-duplicating the metric family (section 18) does not change the verdict:
   **A p = 0.144, B p = 0.005**, again with different argmaxes (A `dent_sgn|max|W8` at t=20,
   B `hell|prev|fracabove_q75|W5` at t=30). By the stated rule -- a cell counts only if it
   clears in both corpora -- **zero cells count.**

5. **The near-miss, and why it is not real.** A *burst count* aggregation --
   `fracabove_q75|W12`, the fraction of chunk transitions in the window whose divergence exceeds
   a global 75th-percentile threshold -- was the single most reproducible residual leader:
   residual det-AUC **A 0.646 / B 0.634** at t=30, pointwise p = 0.005 in both corpora, stable
   across all ten f-divergences and across thresholds q0.60-q0.80, and it drove a joint
   cross-corpus clearing count of 128 cells against a null max of 76 (p = 0.005). **It dies under
   a richer baseline.** Residualised on all five window means instead of just W=8, it falls to
   0.547/0.558; residualised additionally on the window std/max/min/median and the full-history
   EWM, it falls to **0.521/0.507 with pointwise p = 0.67 / 0.86** (section 15). The joint
   clearing count falls 128 -> 42 -> **16 (p = 0.085)** along the same ladder (section 20b).
   The burst count was reading nothing but the *memory length* of the same shared factor.

6. **The prior "longer memory" result is reproduced, not overturned.** W=12 and EWM alpha=0.15
   again beat W=8 raw in both corpora at t=30 (A 0.807/0.800 vs 0.799; B 0.769/0.776 vs 0.751),
   again fail residualisation in corpus A (family-max p = 0.189 vs the previous round's 0.110),
   and again the two corpora peak at different statistics. The evidence is *exactly* as strong
   and exactly as insufficient as last time. **Do not promote W=12 over W=8.**

### Recommendation

**Keep the baseline aggregation unchanged: Hellinger to the previous chunk, window mean, W=8.**

The reason is not that it won -- it is that **the grid contains no alternative that wins.** More
precisely:

* Changing the *metric* is free and pointless; use whichever is cheapest. Hellinger is bounded
  in [0,1], needs no epsilon guard (unlike symmetric KL / chi-squared, which need clipping and
  are unbounded), and is already the convention. **Never use EMD** on this axis: expert index has
  no metric meaning, and it measurably costs 0.03 AUC.
* Keep `|prev` as the reference. This is the one choice in the whole grid that is doing real work
  and it is already correct.
* Report W=8 as the headline, but **quote the memory-length sensitivity band explicitly**
  (W in 3..20 and EWM alpha in 0.15..0.5 give A 0.758-0.807 / B 0.751-0.781 at t=30). That band
  is honest uncertainty about a free parameter, not a menu to pick the maximum from.
* Do **not** adopt any thresholded / burst-count / order-statistic aggregation. Section 15 is
  the reason: they look like new information against a single-baseline residual and stop looking
  like it against a two-moment one.

### What this means for the sweep as a whole

The previous round found that varying *which slice* of the tensor to use buys nothing because
chunk-to-chunk route change is a single shared factor across layers, tokens and denoise steps.
This round finds the same thing along the two remaining axes, and sharpens it: the factor is
also single across **how you measure the change** (ten divergences, rho >= 0.99) and across
**how you summarise it in time** (every alternative aggregation collapses onto the family of
window means). The aggregation choice was worth 0.109 AUC in corpus B and therefore had to be
searched -- but that value turns out to be the *cost of choosing badly*, not headroom above the
current choice. **The tensor has one degree of freedom under this reading, and the baseline is
already sitting on it.**

