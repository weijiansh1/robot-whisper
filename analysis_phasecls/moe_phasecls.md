# MoE trap-type phase-classification validation  (200 perm draws, seed 20260829)

Groups: corpus A = worker (4), corpus B = init_state_id (16).
Classes: stagnation = label_stagnation; loop = loop_or_cycling & ~stagnation; other = neither.  Headline setting = HB back block (layers 12-15), denoise 9, 5 bins.

## 0. Corpus composition

- **corpus A**: 235 failure branches, groups=4 -> {'loop': 150, 'stagnation': 49, 'other': 36}
```
cls    loop  other  stagnation
row_0                         
0        33     19          19
1        76      2           2
2        35     12          28
3         6      3           0
```
- **corpus B**: 216 failure branches, groups=14 -> {'stagnation': 128, 'other': 47, 'loop': 41}
```
cls    loop  other  stagnation
row_0                         
0         2      6          10
3         7      3          17
7         1      1           1
10       16      5           2
13        0      1          22
16        0      1           3
20        0      9           0
23        0      3           0
26        4      4          15
29        0      2           2
39        3      9          16
42        8      2           8
46        0      1           0
49        0      0          32
```

## 1. Sentinels (reported before the headline)

### 1a. Length sentinel

| corpus | contrast | n_pos/n_neg | LOGO length AUC | median len pos | median len neg | MWU p |
|---|---|---|---|---|---|---|
| A | stagnation vs loop | 49/150 | 0.443 | 50 | 50 | 0.7 |
| A | stagnation vs other | 49/36 | 0.449 | 50 | 50 | 0.34 |
| A | loop vs other | 150/36 | 0.452 | 50 | 50 | 0.38 |
| B | stagnation vs loop | 128/41 | 0.500 | 52 | 52 | 1 (length CONSTANT)|
| B | stagnation vs other | 128/47 | 0.500 | 52 | 52 | 1 (length CONSTANT)|
| B | loop vs other | 41/47 | 0.500 | 52 | 52 | 1 (length CONSTANT)|

### 1b. Phase-shuffle sentinel (5 bins permuted within each branch, back/d9, 20 reshuffles)

| corpus | contrast | channel | true AUC | shuffled mean [min,max] | delta |
|---|---|---|---|---|---|
| A | stagnation vs loop | state | 0.994 | 0.996 [0.989,0.999] | +0.002 |
| A | stagnation vs loop | action | 0.993 | 0.994 [0.990,0.997] | +0.001 |
| A | stagnation vs loop | both | 0.993 | 0.994 [0.986,0.999] | +0.002 |
| A | stagnation vs other | state | 0.896 | 0.873 [0.785,0.930] | -0.023 |
| A | stagnation vs other | action | 0.879 | 0.876 [0.832,0.934] | -0.003 |
| A | stagnation vs other | both | 0.889 | 0.858 [0.765,0.927] | -0.030 |
| A | loop vs other | state | 0.589 | 0.688 [0.591,0.749] | +0.099 |
| A | loop vs other | action | 0.664 | 0.748 [0.674,0.814] | +0.084 |
| A | loop vs other | both | 0.641 | 0.705 [0.631,0.783] | +0.064 |
| B | stagnation vs loop | state | 0.994 | 0.994 [0.988,1.000] | -0.000 |
| B | stagnation vs loop | action | 0.991 | 0.995 [0.991,1.000] | +0.004 |
| B | stagnation vs loop | both | 0.994 | 0.993 [0.985,1.000] | -0.001 |
| B | stagnation vs other | state | 0.997 | 0.996 [0.989,1.000] | -0.001 |
| B | stagnation vs other | action | 0.995 | 0.995 [0.992,1.000] | +0.001 |
| B | stagnation vs other | both | 0.997 | 0.994 [0.987,0.997] | -0.004 |
| B | loop vs other | state | 0.734 | 0.764 [0.630,0.861] | +0.030 |
| B | loop vs other | action | 0.792 | 0.753 [0.601,0.855] | -0.039 |
| B | loop vs other | both | 0.717 | 0.732 [0.503,0.879] | +0.015 |

## 2. Observed sensitivity grid (LOGO within-group paired AUC, pooled by pair count)


**corpus A / stagnation vs loop** (n=49/150)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 0.993 | 0.994 | 0.991 | 0.994 |
| back | 9 | action | 0.993 | 0.993 | 0.993 | 0.993 |
| back | 9 | both | 0.991 | 0.993 | 0.992 | 0.993 |
| back | 0 | state | 0.993 | 0.994 | 0.991 | 0.994 |
| back | 0 | action | 0.995 | 0.996 | 0.997 | 0.995 |
| back | 0 | both | 0.992 | 0.996 | 0.993 | 0.994 |
| front | 9 | state | 0.991 | 0.992 | 0.990 | 0.991 |
| front | 9 | action | 0.982 | 0.982 | 0.980 | 0.978 |
| front | 9 | both | 0.991 | 0.991 | 0.990 | 0.989 |
| front | 0 | state | 0.991 | 0.992 | 0.990 | 0.991 |
| front | 0 | action | 0.686 | 0.721 | 0.688 | 0.661 |
| front | 0 | both | 0.991 | 0.992 | 0.989 | 0.986 |

**corpus A / stagnation vs other** (n=49/36)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 0.899 | 0.896 | 0.820 | 0.807 |
| back | 9 | action | 0.889 | 0.879 | 0.807 | 0.799 |
| back | 9 | both | 0.887 | 0.889 | 0.796 | 0.796 |
| back | 0 | state | 0.899 | 0.896 | 0.817 | 0.810 |
| back | 0 | action | 0.859 | 0.783 | 0.800 | 0.780 |
| back | 0 | both | 0.900 | 0.836 | 0.787 | 0.779 |
| front | 9 | state | 0.922 | 0.879 | 0.852 | 0.904 |
| front | 9 | action | 0.810 | 0.745 | 0.739 | 0.713 |
| front | 9 | both | 0.909 | 0.832 | 0.833 | 0.812 |
| front | 0 | state | 0.922 | 0.879 | 0.852 | 0.903 |
| front | 0 | action | 0.482 | 0.559 | 0.572 | 0.492 |
| front | 0 | both | 0.854 | 0.819 | 0.755 | 0.779 |

**corpus A / loop vs other** (n=150/36)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 0.661 | 0.589 | 0.644 | 0.604 |
| back | 9 | action | 0.675 | 0.664 | 0.593 | 0.661 |
| back | 9 | both | 0.675 | 0.641 | 0.593 | 0.656 |
| back | 0 | state | 0.660 | 0.589 | 0.645 | 0.601 |
| back | 0 | action | 0.683 | 0.700 | 0.682 | 0.735 |
| back | 0 | both | 0.638 | 0.684 | 0.680 | 0.661 |
| front | 9 | state | 0.560 | 0.582 | 0.546 | 0.596 |
| front | 9 | action | 0.763 | 0.767 | 0.666 | 0.649 |
| front | 9 | both | 0.730 | 0.653 | 0.620 | 0.602 |
| front | 0 | state | 0.560 | 0.582 | 0.545 | 0.596 |
| front | 0 | action | 0.587 | 0.686 | 0.593 | 0.624 |
| front | 0 | both | 0.573 | 0.611 | 0.549 | 0.592 |

**corpus B / stagnation vs loop** (n=128/41)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 1.000 | 0.994 | 0.994 | 0.994 |
| back | 9 | action | 0.994 | 0.991 | 0.994 | 0.997 |
| back | 9 | both | 0.994 | 0.994 | 0.994 | 0.994 |
| back | 0 | state | 1.000 | 0.994 | 0.994 | 0.994 |
| back | 0 | action | 0.994 | 0.994 | 0.991 | 0.994 |
| back | 0 | both | 1.000 | 0.994 | 0.994 | 0.991 |
| front | 9 | state | 0.997 | 0.994 | 0.994 | 0.994 |
| front | 9 | action | 0.997 | 0.997 | 0.997 | 0.994 |
| front | 9 | both | 0.997 | 0.997 | 0.997 | 0.991 |
| front | 0 | state | 0.997 | 0.994 | 0.994 | 0.994 |
| front | 0 | action | 0.808 | 0.817 | 0.840 | 0.828 |
| front | 0 | both | 0.997 | 0.997 | 0.997 | 0.994 |

**corpus B / stagnation vs other** (n=128/47)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 1.000 | 0.997 | 1.000 | 1.000 |
| back | 9 | action | 1.000 | 0.995 | 1.000 | 0.995 |
| back | 9 | both | 1.000 | 0.997 | 1.000 | 1.000 |
| back | 0 | state | 1.000 | 0.997 | 1.000 | 1.000 |
| back | 0 | action | 0.995 | 0.992 | 0.992 | 0.981 |
| back | 0 | both | 0.997 | 1.000 | 1.000 | 0.997 |
| front | 9 | state | 1.000 | 0.995 | 0.995 | 0.995 |
| front | 9 | action | 0.976 | 0.989 | 0.989 | 0.987 |
| front | 9 | both | 0.995 | 0.995 | 0.995 | 0.989 |
| front | 0 | state | 1.000 | 0.995 | 0.995 | 0.995 |
| front | 0 | action | 0.852 | 0.725 | 0.722 | 0.830 |
| front | 0 | both | 0.995 | 0.989 | 0.989 | 0.992 |

**corpus B / loop vs other** (n=41/47)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 0.792 | 0.734 | 0.809 | 0.780 |
| back | 9 | action | 0.850 | 0.792 | 0.786 | 0.723 |
| back | 9 | both | 0.821 | 0.717 | 0.798 | 0.740 |
| back | 0 | state | 0.792 | 0.734 | 0.809 | 0.780 |
| back | 0 | action | 0.769 | 0.711 | 0.688 | 0.717 |
| back | 0 | both | 0.769 | 0.734 | 0.740 | 0.746 |
| front | 9 | state | 0.763 | 0.682 | 0.705 | 0.682 |
| front | 9 | action | 0.526 | 0.601 | 0.503 | 0.590 |
| front | 9 | both | 0.665 | 0.671 | 0.647 | 0.705 |
| front | 0 | state | 0.763 | 0.682 | 0.705 | 0.682 |
| front | 0 | action | 0.595 | 0.590 | 0.497 | 0.572 |
| front | 0 | both | 0.746 | 0.717 | 0.682 | 0.694 |

**corpus Bstrict / stagnation vs loop** (n=101/43)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 1.000 | 1.000 | 1.000 | 1.000 |
| back | 9 | action | 1.000 | 1.000 | 1.000 | 1.000 |
| back | 9 | both | 1.000 | 1.000 | 1.000 | 1.000 |
| back | 0 | state | 1.000 | 1.000 | 1.000 | 1.000 |
| back | 0 | action | 1.000 | 1.000 | 1.000 | 1.000 |
| back | 0 | both | 1.000 | 1.000 | 1.000 | 1.000 |
| front | 9 | state | 1.000 | 1.000 | 1.000 | 1.000 |
| front | 9 | action | 1.000 | 1.000 | 1.000 | 1.000 |
| front | 9 | both | 1.000 | 1.000 | 1.000 | 1.000 |
| front | 0 | state | 1.000 | 1.000 | 1.000 | 1.000 |
| front | 0 | action | 0.832 | 0.852 | 0.869 | 0.832 |
| front | 0 | both | 1.000 | 1.000 | 1.000 | 1.000 |

**corpus Bstrict / stagnation vs other** (n=101/72)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 0.984 | 0.996 | 0.984 | 0.998 |
| back | 9 | action | 0.985 | 0.982 | 0.985 | 0.995 |
| back | 9 | both | 0.985 | 1.000 | 0.984 | 0.996 |
| back | 0 | state | 0.984 | 0.996 | 0.984 | 0.998 |
| back | 0 | action | 0.985 | 0.993 | 0.989 | 0.995 |
| back | 0 | both | 0.991 | 0.996 | 0.985 | 1.000 |
| front | 9 | state | 0.989 | 1.000 | 0.991 | 0.998 |
| front | 9 | action | 0.976 | 0.975 | 0.982 | 0.982 |
| front | 9 | both | 0.982 | 0.982 | 0.982 | 0.984 |
| front | 0 | state | 0.989 | 1.000 | 0.991 | 0.998 |
| front | 0 | action | 0.784 | 0.766 | 0.751 | 0.808 |
| front | 0 | both | 0.991 | 0.993 | 0.993 | 0.991 |

**corpus Bstrict / loop vs other** (n=43/72)

| block | denoise | channel | 3 bins | 5 bins | 8 bins | 10 bins |
|---|---|---|---|---|---|---|
| back | 9 | state | 0.867 | 0.825 | 0.860 | 0.856 |
| back | 9 | action | 0.888 | 0.846 | 0.860 | 0.842 |
| back | 9 | both | 0.870 | 0.807 | 0.863 | 0.846 |
| back | 0 | state | 0.867 | 0.825 | 0.860 | 0.856 |
| back | 0 | action | 0.842 | 0.818 | 0.796 | 0.779 |
| back | 0 | both | 0.839 | 0.828 | 0.842 | 0.814 |
| front | 9 | state | 0.825 | 0.789 | 0.807 | 0.807 |
| front | 9 | action | 0.737 | 0.730 | 0.740 | 0.786 |
| front | 9 | both | 0.786 | 0.793 | 0.807 | 0.818 |
| front | 0 | state | 0.825 | 0.789 | 0.807 | 0.807 |
| front | 0 | action | 0.632 | 0.600 | 0.558 | 0.561 |
| front | 0 | both | 0.828 | 0.814 | 0.782 | 0.825 |

## 3. Leave-one-group-out breakdown (back / denoise 9 / 5 bins)


- **A / stagnation vs loop / state**: pooled = 0.994 (3 contributing groups, 1759 pairs)
    - g0: AUC=0.984  pairs=627 (36% of weight)   -> pooled without g0 = 0.999
    - g1: AUC=0.993  pairs=152 (9% of weight)   -> pooled without g1 = 0.994
    - g2: AUC=1.000  pairs=980 (56% of weight)   -> pooled without g2 = 0.986

- **A / stagnation vs loop / action**: pooled = 0.993 (3 contributing groups, 1759 pairs)
    - g0: AUC=0.981  pairs=627 (36% of weight)   -> pooled without g0 = 0.999
    - g1: AUC=0.993  pairs=152 (9% of weight)   -> pooled without g1 = 0.993
    - g2: AUC=1.000  pairs=980 (56% of weight)   -> pooled without g2 = 0.983

- **A / stagnation vs loop / both**: pooled = 0.993 (3 contributing groups, 1759 pairs)
    - g0: AUC=0.981  pairs=627 (36% of weight)   -> pooled without g0 = 0.999
    - g1: AUC=0.993  pairs=152 (9% of weight)   -> pooled without g1 = 0.993
    - g2: AUC=1.000  pairs=980 (56% of weight)   -> pooled without g2 = 0.983

- **A / stagnation vs other / state**: pooled = 0.896 (3 contributing groups, 701 pairs)
    - g0: AUC=0.895  pairs=361 (51% of weight)   -> pooled without g0 = 0.897
    - g1: AUC=1.000  pairs=4 (1% of weight)   -> pooled without g1 = 0.895
    - g2: AUC=0.896  pairs=336 (48% of weight)   -> pooled without g2 = 0.896

- **A / stagnation vs other / action**: pooled = 0.879 (3 contributing groups, 701 pairs)
    - g0: AUC=0.898  pairs=361 (51% of weight)   -> pooled without g0 = 0.859
    - g1: AUC=1.000  pairs=4 (1% of weight)   -> pooled without g1 = 0.878
    - g2: AUC=0.857  pairs=336 (48% of weight)   -> pooled without g2 = 0.899

- **A / stagnation vs other / both**: pooled = 0.889 (3 contributing groups, 701 pairs)
    - g0: AUC=0.914  pairs=361 (51% of weight)   -> pooled without g0 = 0.862
    - g1: AUC=1.000  pairs=4 (1% of weight)   -> pooled without g1 = 0.888
    - g2: AUC=0.860  pairs=336 (48% of weight)   -> pooled without g2 = 0.915

- **A / loop vs other / state**: pooled = 0.589 (4 contributing groups, 1217 pairs)
    - g0: AUC=0.553  pairs=627 (52% of weight)   -> pooled without g0 = 0.627
    - g1: AUC=0.836  pairs=152 (12% of weight)   -> pooled without g1 = 0.554
    - g2: AUC=0.550  pairs=420 (35% of weight)   -> pooled without g2 = 0.610
    - g3: AUC=0.667  pairs=18 (1% of weight)   -> pooled without g3 = 0.588

- **A / loop vs other / action**: pooled = 0.664 (4 contributing groups, 1217 pairs)
    - g0: AUC=0.686  pairs=627 (52% of weight)   -> pooled without g0 = 0.641
    - g1: AUC=0.842  pairs=152 (12% of weight)   -> pooled without g1 = 0.638
    - g2: AUC=0.567  pairs=420 (35% of weight)   -> pooled without g2 = 0.715
    - g3: AUC=0.667  pairs=18 (1% of weight)   -> pooled without g3 = 0.664

- **A / loop vs other / both**: pooled = 0.641 (4 contributing groups, 1217 pairs)
    - g0: AUC=0.619  pairs=627 (52% of weight)   -> pooled without g0 = 0.664
    - g1: AUC=0.882  pairs=152 (12% of weight)   -> pooled without g1 = 0.607
    - g2: AUC=0.583  pairs=420 (35% of weight)   -> pooled without g2 = 0.671
    - g3: AUC=0.722  pairs=18 (1% of weight)   -> pooled without g3 = 0.640

- **B / stagnation vs loop / state**: pooled = 0.994 (7 contributing groups, 344 pairs)
    - g0: AUC=1.000  pairs=20 (6% of weight)   -> pooled without g0 = 0.994
    - g3: AUC=0.983  pairs=119 (35% of weight)   -> pooled without g3 = 1.000
    - g7: AUC=1.000  pairs=1 (0% of weight)   -> pooled without g7 = 0.994
    - g10: AUC=1.000  pairs=32 (9% of weight)   -> pooled without g10 = 0.994
    - g26: AUC=1.000  pairs=60 (17% of weight)   -> pooled without g26 = 0.993
    - g39: AUC=1.000  pairs=48 (14% of weight)   -> pooled without g39 = 0.993
    - g42: AUC=1.000  pairs=64 (19% of weight)   -> pooled without g42 = 0.993

- **B / stagnation vs loop / action**: pooled = 0.991 (7 contributing groups, 344 pairs)
    - g0: AUC=1.000  pairs=20 (6% of weight)   -> pooled without g0 = 0.991
    - g3: AUC=0.983  pairs=119 (35% of weight)   -> pooled without g3 = 0.996
    - g7: AUC=1.000  pairs=1 (0% of weight)   -> pooled without g7 = 0.991
    - g10: AUC=1.000  pairs=32 (9% of weight)   -> pooled without g10 = 0.990
    - g26: AUC=0.983  pairs=60 (17% of weight)   -> pooled without g26 = 0.993
    - g39: AUC=1.000  pairs=48 (14% of weight)   -> pooled without g39 = 0.990
    - g42: AUC=1.000  pairs=64 (19% of weight)   -> pooled without g42 = 0.989

- **B / stagnation vs loop / both**: pooled = 0.994 (7 contributing groups, 344 pairs)
    - g0: AUC=1.000  pairs=20 (6% of weight)   -> pooled without g0 = 0.994
    - g3: AUC=0.983  pairs=119 (35% of weight)   -> pooled without g3 = 1.000
    - g7: AUC=1.000  pairs=1 (0% of weight)   -> pooled without g7 = 0.994
    - g10: AUC=1.000  pairs=32 (9% of weight)   -> pooled without g10 = 0.994
    - g26: AUC=1.000  pairs=60 (17% of weight)   -> pooled without g26 = 0.993
    - g39: AUC=1.000  pairs=48 (14% of weight)   -> pooled without g39 = 0.993
    - g42: AUC=1.000  pairs=64 (19% of weight)   -> pooled without g42 = 0.993

- **B / stagnation vs other / state**: pooled = 0.997 (10 contributing groups, 371 pairs)
    - g0: AUC=1.000  pairs=60 (16% of weight)   -> pooled without g0 = 0.997
    - g3: AUC=0.980  pairs=51 (14% of weight)   -> pooled without g3 = 1.000
    - g7: AUC=1.000  pairs=1 (0% of weight)   -> pooled without g7 = 0.997
    - g10: AUC=1.000  pairs=10 (3% of weight)   -> pooled without g10 = 0.997
    - g13: AUC=1.000  pairs=22 (6% of weight)   -> pooled without g13 = 0.997
    - g16: AUC=1.000  pairs=3 (1% of weight)   -> pooled without g16 = 0.997
    - g26: AUC=1.000  pairs=60 (16% of weight)   -> pooled without g26 = 0.997
    - g29: AUC=1.000  pairs=4 (1% of weight)   -> pooled without g29 = 0.997
    - g39: AUC=1.000  pairs=144 (39% of weight)   -> pooled without g39 = 0.996
    - g42: AUC=1.000  pairs=16 (4% of weight)   -> pooled without g42 = 0.997

- **B / stagnation vs other / action**: pooled = 0.995 (10 contributing groups, 371 pairs)
    - g0: AUC=1.000  pairs=60 (16% of weight)   -> pooled without g0 = 0.994
    - g3: AUC=0.961  pairs=51 (14% of weight)   -> pooled without g3 = 1.000
    - g7: AUC=1.000  pairs=1 (0% of weight)   -> pooled without g7 = 0.995
    - g10: AUC=1.000  pairs=10 (3% of weight)   -> pooled without g10 = 0.994
    - g13: AUC=1.000  pairs=22 (6% of weight)   -> pooled without g13 = 0.994
    - g16: AUC=1.000  pairs=3 (1% of weight)   -> pooled without g16 = 0.995
    - g26: AUC=1.000  pairs=60 (16% of weight)   -> pooled without g26 = 0.994
    - g29: AUC=1.000  pairs=4 (1% of weight)   -> pooled without g29 = 0.995
    - g39: AUC=1.000  pairs=144 (39% of weight)   -> pooled without g39 = 0.991
    - g42: AUC=1.000  pairs=16 (4% of weight)   -> pooled without g42 = 0.994

- **B / stagnation vs other / both**: pooled = 0.997 (10 contributing groups, 371 pairs)
    - g0: AUC=1.000  pairs=60 (16% of weight)   -> pooled without g0 = 0.997
    - g3: AUC=0.980  pairs=51 (14% of weight)   -> pooled without g3 = 1.000
    - g7: AUC=1.000  pairs=1 (0% of weight)   -> pooled without g7 = 0.997
    - g10: AUC=1.000  pairs=10 (3% of weight)   -> pooled without g10 = 0.997
    - g13: AUC=1.000  pairs=22 (6% of weight)   -> pooled without g13 = 0.997
    - g16: AUC=1.000  pairs=3 (1% of weight)   -> pooled without g16 = 0.997
    - g26: AUC=1.000  pairs=60 (16% of weight)   -> pooled without g26 = 0.997
    - g29: AUC=1.000  pairs=4 (1% of weight)   -> pooled without g29 = 0.997
    - g39: AUC=1.000  pairs=144 (39% of weight)   -> pooled without g39 = 0.996
    - g42: AUC=1.000  pairs=16 (4% of weight)   -> pooled without g42 = 0.997

- **B / loop vs other / state**: pooled = 0.734 (7 contributing groups, 173 pairs)
    - g0: AUC=1.000  pairs=12 (7% of weight)   -> pooled without g0 = 0.714
    - g3: AUC=0.905  pairs=21 (12% of weight)   -> pooled without g3 = 0.711
    - g7: AUC=1.000  pairs=1 (1% of weight)   -> pooled without g7 = 0.733
    - g10: AUC=0.775  pairs=80 (46% of weight)   -> pooled without g10 = 0.699
    - g26: AUC=0.438  pairs=16 (9% of weight)   -> pooled without g26 = 0.764
    - g39: AUC=0.370  pairs=27 (16% of weight)   -> pooled without g39 = 0.801
    - g42: AUC=1.000  pairs=16 (9% of weight)   -> pooled without g42 = 0.707

- **B / loop vs other / action**: pooled = 0.792 (7 contributing groups, 173 pairs)
    - g0: AUC=1.000  pairs=12 (7% of weight)   -> pooled without g0 = 0.776
    - g3: AUC=0.857  pairs=21 (12% of weight)   -> pooled without g3 = 0.783
    - g7: AUC=1.000  pairs=1 (1% of weight)   -> pooled without g7 = 0.791
    - g10: AUC=0.925  pairs=80 (46% of weight)   -> pooled without g10 = 0.677
    - g26: AUC=0.625  pairs=16 (9% of weight)   -> pooled without g26 = 0.809
    - g39: AUC=0.222  pairs=27 (16% of weight)   -> pooled without g39 = 0.897
    - g42: AUC=1.000  pairs=16 (9% of weight)   -> pooled without g42 = 0.771

- **B / loop vs other / both**: pooled = 0.717 (7 contributing groups, 173 pairs)
    - g0: AUC=1.000  pairs=12 (7% of weight)   -> pooled without g0 = 0.696
    - g3: AUC=0.857  pairs=21 (12% of weight)   -> pooled without g3 = 0.697
    - g7: AUC=1.000  pairs=1 (1% of weight)   -> pooled without g7 = 0.715
    - g10: AUC=0.787  pairs=80 (46% of weight)   -> pooled without g10 = 0.656
    - g26: AUC=0.562  pairs=16 (9% of weight)   -> pooled without g26 = 0.732
    - g39: AUC=0.185  pairs=27 (16% of weight)   -> pooled without g39 = 0.815
    - g42: AUC=1.000  pairs=16 (9% of weight)   -> pooled without g42 = 0.688

## 5. Decomposition: level vs shape, and the matched physical control


**corpus A -- phase profile, state token, back/d9 (class mean)**

| class | n | bin1 | bin2 | bin3 | bin4 | bin5 | branch mean |
|---|---|---|---|---|---|---|---|
| stagnation | 49 | 0.0643 | 0.0700 | 0.0550 | 0.0180 | 0.0160 | 0.0446 |
| loop | 150 | 0.0606 | 0.0673 | 0.0639 | 0.0585 | 0.0610 | 0.0623 |
| other | 36 | 0.0636 | 0.0669 | 0.0610 | 0.0422 | 0.0460 | 0.0559 |

_matched physical control (eef displacement per query, same bins):_

| class | n | bin1 | bin2 | bin3 | bin4 | bin5 | branch mean |
|---|---|---|---|---|---|---|---|
| stagnation | 49 | 0.0324 | 0.0519 | 0.0478 | 0.0052 | 0.0031 | 0.0281 |
| loop | 150 | 0.0298 | 0.0440 | 0.0462 | 0.0348 | 0.0444 | 0.0398 |
| other | 36 | 0.0300 | 0.0518 | 0.0568 | 0.0263 | 0.0299 | 0.0389 |

Spearman(route change mean, eef step mean) = 0.861; late bin only = 0.862


| corpus A / state | 5-bin | mean only (1 feat) | shape only (mean removed) | late-minus-early (1 feat) | PHYS 5-bin | PHYS mean | route resid on phys |
|---|---|---|---|---|---|---|---|
| stagnation vs loop | 0.994 | 0.997 | 0.990 | 0.985 | 0.996 | 0.990 | 0.758 |
| stagnation vs other | 0.896 | 0.909 | 0.892 | 0.897 | 0.852 | 0.870 | 0.264 |
| loop vs other | 0.589 | 0.714 | 0.623 | 0.687 | 0.748 | 0.201 | 0.843 |

| corpus A / action | 5-bin | mean only (1 feat) | shape only (mean removed) | late-minus-early (1 feat) | PHYS 5-bin | PHYS mean | route resid on phys |
|---|---|---|---|---|---|---|---|
| stagnation vs loop | 0.993 | 0.994 | 0.991 | 0.982 | 0.996 | 0.990 | 0.874 |
| stagnation vs other | 0.879 | 0.897 | 0.890 | 0.906 | 0.852 | 0.870 | 0.320 |
| loop vs other | 0.664 | 0.765 | 0.656 | 0.638 | 0.748 | 0.201 | 0.883 |

**corpus B -- phase profile, state token, back/d9 (class mean)**

| class | n | bin1 | bin2 | bin3 | bin4 | bin5 | branch mean |
|---|---|---|---|---|---|---|---|
| stagnation | 128 | 0.0671 | 0.0815 | 0.0579 | 0.0103 | 0.0143 | 0.0462 |
| loop | 41 | 0.0661 | 0.0822 | 0.0665 | 0.0545 | 0.0698 | 0.0678 |
| other | 47 | 0.0654 | 0.0781 | 0.0669 | 0.0523 | 0.0511 | 0.0627 |

_matched physical control (eef displacement per query, same bins):_

| class | n | bin1 | bin2 | bin3 | bin4 | bin5 | branch mean |
|---|---|---|---|---|---|---|---|
| stagnation | 128 | 0.0389 | 0.0542 | 0.0571 | 0.0030 | 0.0048 | 0.0316 |
| loop | 41 | 0.0381 | 0.0560 | 0.0575 | 0.0435 | 0.0555 | 0.0501 |
| other | 47 | 0.0376 | 0.0519 | 0.0615 | 0.0350 | 0.0326 | 0.0437 |

Spearman(route change mean, eef step mean) = 0.954; late bin only = 0.958


| corpus B / state | 5-bin | mean only (1 feat) | shape only (mean removed) | late-minus-early (1 feat) | PHYS 5-bin | PHYS mean | route resid on phys |
|---|---|---|---|---|---|---|---|
| stagnation vs loop | 0.994 | 0.994 | 0.994 | 0.977 | 1.000 | 0.994 | 0.517 |
| stagnation vs other | 0.997 | 1.000 | 0.997 | 0.935 | 1.000 | 0.995 | 0.801 |
| loop vs other | 0.734 | 0.815 | 0.786 | 0.821 | 0.821 | 0.803 | 0.636 |

| corpus B / action | 5-bin | mean only (1 feat) | shape only (mean removed) | late-minus-early (1 feat) | PHYS 5-bin | PHYS mean | route resid on phys |
|---|---|---|---|---|---|---|---|
| stagnation vs loop | 0.991 | 0.997 | 0.994 | 0.948 | 1.000 | 0.994 | 0.270 |
| stagnation vs other | 0.995 | 0.995 | 0.992 | 0.922 | 1.000 | 0.995 | 0.806 |
| loop vs other | 0.792 | 0.815 | 0.803 | 0.792 | 0.821 | 0.803 | 0.671 |

## 4. Permutation null (200 draws, class labels shuffled WITHIN group at branch level, whole LOGO pipeline recomputed each draw)


_corpus A: 200 draws x 144 cells in 316s_

null max-statistic: family of 36 (back/d9) mean=0.618 p95=0.687 max=0.740; family of 144 mean=0.653 p95=0.715 max=0.752

| contrast | block | dn | channel | bins | obs AUC | null mean | null p95 | p_marg | p_FWER(36) | p_FWER(144) |
|---|---|---|---|---|---|---|---|---|---|---|
| stagnation vs loop | back | 9 | state | 3 | 0.993 | 0.500 | 0.589 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | state | 5 | 0.994 | 0.501 | 0.598 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | state | 8 | 0.991 | 0.498 | 0.603 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | state | 10 | 0.994 | 0.495 | 0.615 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 3 | 0.993 | 0.500 | 0.599 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 5 | 0.993 | 0.505 | 0.604 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 8 | 0.993 | 0.500 | 0.606 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 10 | 0.993 | 0.504 | 0.597 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 3 | 0.991 | 0.505 | 0.596 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 5 | 0.993 | 0.505 | 0.609 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 8 | 0.992 | 0.504 | 0.607 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 10 | 0.993 | 0.505 | 0.617 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 3 | 0.899 | 0.493 | 0.631 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 5 | 0.896 | 0.492 | 0.624 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 8 | 0.820 | 0.489 | 0.618 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 10 | 0.807 | 0.486 | 0.604 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 3 | 0.889 | 0.496 | 0.628 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 5 | 0.879 | 0.496 | 0.618 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 8 | 0.807 | 0.491 | 0.615 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 10 | 0.799 | 0.495 | 0.616 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 3 | 0.887 | 0.494 | 0.613 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 5 | 0.889 | 0.494 | 0.622 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 8 | 0.796 | 0.485 | 0.618 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 10 | 0.796 | 0.490 | 0.618 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | state | 3 | 0.661 | 0.498 | 0.609 | 0.0149 | 0.1791 | 0.4229 |
| loop vs other | back | 9 | state | 5 | 0.589 | 0.504 | 0.623 | 0.1443 | 0.7512 | 0.9204 |
| loop vs other | back | 9 | state | 8 | 0.644 | 0.501 | 0.632 | 0.0398 | 0.2886 | 0.6219 |
| loop vs other | back | 9 | state | 10 | 0.604 | 0.505 | 0.614 | 0.0697 | 0.6219 | 0.8806 |
| loop vs other | back | 9 | action | 3 | 0.675 | 0.500 | 0.606 | 0.0050 | 0.1194 | 0.2886 |
| loop vs other | back | 9 | action | 5 | 0.664 | 0.507 | 0.621 | 0.0149 | 0.1493 | 0.3781 |
| loop vs other | back | 9 | action | 8 | 0.593 | 0.507 | 0.606 | 0.0945 | 0.7065 | 0.9154 |
| loop vs other | back | 9 | action | 10 | 0.661 | 0.513 | 0.622 | 0.0199 | 0.1791 | 0.4229 |
| loop vs other | back | 9 | both | 3 | 0.675 | 0.499 | 0.607 | 0.0100 | 0.1194 | 0.2886 |
| loop vs other | back | 9 | both | 5 | 0.641 | 0.504 | 0.624 | 0.0249 | 0.3234 | 0.6667 |
| loop vs other | back | 9 | both | 8 | 0.593 | 0.504 | 0.614 | 0.0945 | 0.7065 | 0.9154 |
| loop vs other | back | 9 | both | 10 | 0.656 | 0.509 | 0.620 | 0.0199 | 0.2090 | 0.4677 |

_corpus B: 200 draws x 144 cells in 810s_

null max-statistic: family of 36 (back/d9) mean=0.630 p95=0.725 max=0.775; family of 144 mean=0.671 p95=0.746 max=0.786

| contrast | block | dn | channel | bins | obs AUC | null mean | null p95 | p_marg | p_FWER(36) | p_FWER(144) |
|---|---|---|---|---|---|---|---|---|---|---|
| stagnation vs loop | back | 9 | state | 3 | 1.000 | 0.501 | 0.613 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | state | 5 | 0.994 | 0.505 | 0.622 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | state | 8 | 0.994 | 0.507 | 0.622 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | state | 10 | 0.994 | 0.506 | 0.625 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 3 | 0.994 | 0.501 | 0.622 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 5 | 0.991 | 0.498 | 0.616 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 8 | 0.994 | 0.500 | 0.608 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | action | 10 | 0.997 | 0.504 | 0.628 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 3 | 0.994 | 0.501 | 0.616 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 5 | 0.994 | 0.503 | 0.616 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 8 | 0.994 | 0.504 | 0.617 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs loop | back | 9 | both | 10 | 0.994 | 0.502 | 0.608 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 3 | 1.000 | 0.502 | 0.612 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 5 | 0.997 | 0.499 | 0.617 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 8 | 1.000 | 0.501 | 0.618 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | state | 10 | 1.000 | 0.502 | 0.612 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 3 | 1.000 | 0.501 | 0.612 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 5 | 0.995 | 0.500 | 0.596 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 8 | 1.000 | 0.501 | 0.623 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | action | 10 | 0.995 | 0.498 | 0.615 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 3 | 1.000 | 0.502 | 0.612 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 5 | 0.997 | 0.501 | 0.601 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 8 | 1.000 | 0.502 | 0.634 | 0.0050 | 0.0050 | 0.0050 |
| stagnation vs other | back | 9 | both | 10 | 1.000 | 0.503 | 0.618 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | state | 3 | 0.792 | 0.485 | 0.647 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | state | 5 | 0.734 | 0.494 | 0.647 | 0.0149 | 0.0498 | 0.1294 |
| loop vs other | back | 9 | state | 8 | 0.809 | 0.499 | 0.676 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | state | 10 | 0.780 | 0.495 | 0.653 | 0.0050 | 0.0050 | 0.0199 |
| loop vs other | back | 9 | action | 3 | 0.850 | 0.500 | 0.659 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | action | 5 | 0.792 | 0.496 | 0.653 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | action | 8 | 0.786 | 0.505 | 0.677 | 0.0050 | 0.0050 | 0.0149 |
| loop vs other | back | 9 | action | 10 | 0.723 | 0.496 | 0.659 | 0.0249 | 0.0647 | 0.1791 |
| loop vs other | back | 9 | both | 3 | 0.821 | 0.487 | 0.671 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | both | 5 | 0.717 | 0.494 | 0.648 | 0.0100 | 0.0746 | 0.2189 |
| loop vs other | back | 9 | both | 8 | 0.798 | 0.499 | 0.671 | 0.0050 | 0.0050 | 0.0050 |
| loop vs other | back | 9 | both | 10 | 0.740 | 0.495 | 0.653 | 0.0100 | 0.0348 | 0.1045 |

DONE

## 6. Corpus-B label-transfer caveat (quantified on corpus A)

Corpus A ships DENSE physics (`control_sim_state`, 521 rows); corpus B ships only
QUERY-resolution physics (`sim_state`, 52 rows).  The loop predicate is already a
query-level predicate and ports exactly.  The stagnation predicate is a 20-DENSE-step
rolling window and cannot be evaluated literally on B.

Port validation on corpus A (both resolutions available):

| predicate | n fired | vs shipped label | TP | FP | FN |
|---|---|---|---|---|---|
| loop (verbatim port) | 163 | exact match | 163 | 0 | 0 |
| stagnation, dense (verbatim port) | 49 | exact match | 49 | 0 | 0 |
| stagnation, query surrogate (what B must use) | 60 | 0.953 agreement | 49 | 11 | 0 |

The surrogate is an empirical superset (recall 1.00, precision 0.82).  Swapping the
surrogate label in for the dense label **in corpus A** inflates the AUC:

| corpus A, back/d9/5 bins | dense label | query-surrogate label |
|---|---|---|
| stagnation vs loop (state) | 0.994 | 1.000 |
| stagnation vs loop (action) | 0.993 | 1.000 |
| stagnation vs other (state) | 0.896 | 0.963 |
| stagnation vs other (action) | 0.879 | 0.933 |

So corpus B's 0.994 should be read as ~0.99 and corpus B's 0.997 on stagnation-vs-other
should be de-inflated to ~0.93.  A strict (AND-of-both-criteria) variant is reported as
`corpus Bstrict` in section 2.

## 7. Circularity probe

The stagnation label is *defined* by a static-motion predicate.  The routing feature is a
motion readout.  Spearman between the routing late-phase change rate and the label's own
defining statistic (`late_static_window_fraction`):

| corpus | vs late_static_window_fraction | vs terminal_static_window_steps |
|---|---|---|
| A | -0.792 | -0.531 |
| B | -0.931 | -0.850 |

Stratifying stagnation-vs-loop on the physical late-phase motion (5 quantile strata) and
recomputing the routing paired AUC inside strata: **0.894 (A, 339 pairs)** and
**0.910 (B, 189 pairs)** -- a residual survives, but on a narrow overlap region holding
only 5% (A) / 55% (B) of the original pair mass.

## 8. Verdict

* **0.994 survives the permutation null.** p_marg = p_FWER(36) = p_FWER(144) = 0.005,
  the 1/201 floor.  Permuted labels never exceeded 0.752 (A) / 0.786 (B) anywhere in the
  144-cell grid.  Overfitting on 5-10 features / few effective groups is ruled out.
* **0.994 survives leave-one-group-out.** Corpus A: 3 contributing workers at
  0.984 / 0.993 / 1.000; dropping w0 -> 0.999, dropping w2 -> 0.986.  Corpus B: 7
  contributing init states, all 0.983-1.000.  It does not rest on one group.
* **0.994 survives the whole sensitivity grid** (0.978-0.997 at 44 of 48 settings in A, 0.991-1.000 at 44 of 48 in B;
  the only 4 low cells in each corpus are front block + denoise 0 + action tokens, at all bin counts).
* **0.994 does NOT survive the phase-shuffle sentinel.**  Shuffled-bin AUC = 0.996 (A) /
  0.994 (B).  A single scalar -- the whole-branch mean change rate, no bins at all --
  gives 0.997 (A) / 0.994 (B).  The 5-bin relative-phase representation buys nothing.
* **0.994 does not survive attribution to the MoE.**  The matched physical control
  (eef displacement per query, same bins) gives 0.996 (A) / 1.000 (B); Spearman with the
  routing feature is 0.861 (A) / 0.954 (B); residualising routing on it leaves
  0.758/0.517 (state) and 0.874/0.270 (action) -- no replicable increment.

**Causality limitation.**  The phase normalisation divides by the branch's total length,
so every feature is computed with knowledge of when the branch ended.  This is offline
classification of failure type *given that the branch failed*; it cannot be converted to
a runtime detector without being redone causally.
