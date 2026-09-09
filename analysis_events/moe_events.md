# Does MoE routing signal discrete physical events?

Corpus A = rolling-star branch capture (352 branches, 16158 chunks, 4 initial
states). Corpus B = SCENE8 replication (512 episodes, 22883 chunks, 16 initial
states). Task in both: KITCHEN_SCENE8 put both moka pots on the stove.

## 0. Data inventory (sim_layout.json + control_sim_state, verified numerically)

`sim_state` is 47-dim = [time(1)] + qpos(24) + qvel(22). Field map:

| index | contents | is_robot |
|---|---|---|
| 0 | sim time | - |
| 1:8 | **robot0_joint1..7 - the 7 arm joint angles** | yes |
| 8:10 | gripper0_finger_joint1,2 | yes |
| 10:13 / 13:17 | moka_pot_1 free-joint position / quaternion | no |
| 17:20 / 20:24 | moka_pot_2 free-joint position / quaternion | no |
| 24 | flat_stove_1_button (1 dof) | no |
| 25:32 | arm qvel (7) | yes |
| 32:34 | finger qvel (2) | yes |
| 34:40 / 40:46 | moka_pot_1 / moka_pot_2 qvel (6 each) | no |
| 46 | stove button qvel | no |

The 7 arm joint angles ARE present, so the joint-configuration ('twisted joint')
target is possible and was run. They are absent from the 8-dim proprio the
policy actually receives ([eef pos 3, axis-angle 3, finger joints 2]) - that
asymmetry is used below as the one provable blindness case.
The CALVIN cache under VLA_MUI_HUB/cache/HiMoE-VLA/calvin_d_d/ is empty
(.gitkeep only), so no CALVIN fallback was needed or possible.

## 1. The structural fact that kills the top-priority target

Contactless object motion does not occur in this task.

| corpus | chunk transitions | pot moved >1cm | of those, eef never within 15 cm | median eef-object dist during motion |
|---|---|---|---|---|
| A pot_1 | 15806 | 1044 | 4 (0.38%) | 0.068 m |
| A pot_2 | 15806 | 1866 | 19 (1.02%) | 0.065 m |
| B pot_1 | 22371 | 2537 | 1 (0.04%) | 0.069 m |
| B pot_2 | 22371 | 3566 | 0 (0.00%) | 0.066 m |

9013 object-motion events across both corpora; 24 (0.27%) without the
end-effector ever coming within 15 cm. A moka pot only moves when it is held.
Consequently every 'gripper open and far away' event has <=5 positives and is
dropped. This is a property of the task, not of the sample size.

## 2. Sentinels (run before any headline)

**Shuffle sentinel** - routing rows permuted within each branch, labels fixed.
Across all 74 event cells in both corpora the acausal control lands at
0.410-0.620 (median 0.516). Chunk-level events are NOT recoverable from
branch identity, unlike the branch-level tasks where an earlier agent found
0.93-0.96. The chunk-level design is clean.

**Clock sentinel** - t, T, t/T, T-t alone. This is the dominant already-known
baseline: phase reaches 0.998-1.000 on 'which pot moves' in corpus B, and
0.63-0.99 on most other events. Any raw AUC must be read against it.

## 3.1 Corpus A: per-event grouped AUC (sorted by tier-1 blindness gap)

```
feature_block             phase  tier1  routing  routing_shufchunk  tier1+routing  tier1+routing_perm  tier2  tier2+routing  gap_t2_minus_t1  routing_increment
ev           n_pos n_neg                                                                                                                                       
jext_m4      300   14802  0.397  0.510    0.512              0.492          0.487               0.507  0.695            NaN            0.185             -0.023
jext_m1      324   15834  0.348  0.662    0.599              0.481          0.654               0.629  0.812            NaN            0.149             -0.008
jext_m2      316   15490  0.385  0.628    0.557              0.494          0.597               0.588  0.753            NaN            0.125             -0.031
tilt1_m1     33    16125  0.703  0.817    0.811              0.580          0.838               0.769  0.939            NaN            0.122              0.021
goalreg2_m4  36    5750   0.476  0.738    0.674              0.508          0.673               0.698  0.853            NaN            0.114             -0.066
tilt2_m2     262   15544  0.585  0.794    0.752              0.512          0.797               0.766  0.907            NaN            0.112              0.003
tilt2_m4     497   14605  0.527  0.783    0.745              0.491          0.765               0.781  0.881            NaN            0.098             -0.019
twist_m2     43    15763  0.601  0.763    0.677              0.529          0.770               0.726  0.832            NaN            0.069              0.007
dropfail1_m1 37    898    0.691  0.752    0.660              0.469          0.720               0.710  0.820            NaN            0.067             -0.032
tilt1_m2     129   15677  0.721  0.808    0.788              0.614          0.836               0.831  0.869            NaN            0.061              0.028
drop1_m1     203   732    0.901  0.907    0.923              0.529          0.923               0.910  0.961            NaN            0.053              0.015
tilt2_m1     122   16036  0.564  0.819    0.777              0.528          0.837               0.808  0.866            NaN            0.047              0.019
dropfail1_m2 99    719    0.856  0.808    0.875              0.515          0.881               0.787  0.855            NaN            0.047              0.073
dispnc2_m4   40    6811   0.496  0.894    0.872              0.620          0.856               0.898  0.938            NaN            0.044             -0.039
dropfail2_m1 64    1286   0.413  0.925    0.912              0.497          0.920               0.889  0.951            NaN            0.026             -0.005
disp2_m2     2329  13477  0.875  0.945    0.952              0.529          0.951               0.944  0.971            NaN            0.026              0.006
dispnc1_m4   122   8991   0.866  0.950    0.951              0.561          0.968               0.948  0.973          0.966            0.023              0.018
disp2_m4     3235  11867  0.828  0.925    0.931              0.524          0.933               0.922  0.947            NaN            0.023              0.009
tilt1_m4     394   14708  0.757  0.878    0.853              0.532          0.875               0.872  0.896          0.893            0.019             -0.002
drop1_m2     373   445    0.948  0.966    0.971              0.474          0.971               0.965  0.980            NaN            0.014              0.006
disp2_m1     1871  14287  0.902  0.977    0.979              0.531          0.982               0.976  0.991            NaN            0.013              0.005
drop1_m4     324   258    0.896  0.961    0.960              0.527          0.964               0.963  0.971            NaN            0.011              0.004
drop2_m1     315   1035   0.610  0.961    0.939              0.471          0.945               0.959  0.970            NaN            0.010             -0.015
disp1_m4     1526  13576  0.916  0.976    0.969              0.572          0.978               0.976  0.981            NaN            0.005              0.002
drop2_m2     548   795    0.614  0.966    0.958              0.500          0.960               0.960  0.969          0.971            0.003             -0.006
disp1_m1     1142  15016  0.960  0.991    0.992              0.585          0.993               0.991  0.994            NaN            0.003              0.002
whichpot_m4  3192  1483   0.977  0.998    0.998              0.534          0.999               0.998  1.000            NaN            0.002              0.001
disp1_m2     1359  14447  0.938  0.986    0.978              0.577          0.987               0.985  0.987          0.987            0.001              0.001
whichpot_m2  2295  1325   0.988  0.999    0.998              0.516          0.999               0.999  1.000          1.000            0.001              0.000
whichpot_m1  1852  1123   0.993  0.999    0.998              0.514          0.999               0.999  0.999            NaN            0.000             -0.000
twist_m1     133   16025  0.955  0.950    0.892              0.500          0.943               0.939  0.949            NaN           -0.001             -0.007
dropfail2_m2 67    1276   0.549  0.934    0.914              0.564          0.936               0.927  0.931            NaN           -0.004              0.002
grip_m1      1042  15116  0.713  0.986    0.977              0.501          0.985               0.985  0.979            NaN           -0.006             -0.000
drop2_m4     800   530    0.653  0.970    0.954              0.484          0.967               0.968  0.963            NaN           -0.007             -0.003
grip_m2      2070  13736  0.733  0.977    0.964              0.514          0.976               0.975  0.966            NaN           -0.010             -0.000
grip_m4      4093  11009  0.756  0.942    0.918              0.517          0.941               0.941  0.929            NaN           -0.013             -0.001
twist_m4     78    15024  0.587  0.752    0.764              0.518          0.800               0.691  0.693            NaN           -0.059              0.048
dropfail1_m4 48    534    0.678  0.655    0.461              0.410          0.480               0.583  0.397            NaN           -0.258             -0.175
```

## 3.2 Corpus B: per-event grouped AUC (sorted by tier-1 blindness gap)

```
feature_block             phase  tier1  routing  routing_shufchunk  tier1+routing  tier1+routing_perm  tier2  tier2+routing  gap_t2_minus_t1  routing_increment
ev           n_pos n_neg                                                                                                                                       
jlim_m2      89    21770  0.628  0.821    0.826              0.482          0.838               0.801  0.940            NaN            0.119              0.017
jext_m1      448   21923  0.804  0.881    0.897              0.549          0.891               0.871  0.920            NaN            0.039              0.010
tilt2_m4     280   20555  0.707  0.900    0.898              0.569          0.906               0.887  0.936            NaN            0.036              0.006
jext_m2      438   21421  0.818  0.901    0.895              0.532          0.908               0.893  0.932            NaN            0.031              0.007
tilt2_m1     45    22326  0.721  0.940    0.894              0.539          0.920               0.924  0.959            NaN            0.019             -0.020
tilt1_m1     59    22312  0.794  0.935    0.879              0.541          0.937               0.896  0.943            NaN            0.008              0.002
tilt2_m2     140   21719  0.725  0.926    0.915              0.544          0.923               0.919  0.933            NaN            0.007             -0.003
tilt1_m2     219   21640  0.819  0.940    0.936              0.477          0.944               0.930  0.946            NaN            0.006              0.004
jext_m4      414   20421  0.789  0.901    0.868              0.538          0.898               0.900  0.904            NaN            0.003             -0.003
tilt1_m4     734   20101  0.863  0.945    0.938              0.518          0.947               0.942  0.947          0.947            0.002              0.002
dispnc1_m4   423   11224  0.977  0.985    0.983              0.495          0.984               0.985  0.988          0.988            0.002             -0.001
disp1_m4     3421  17414  0.951  0.984    0.982              0.516          0.985               0.983  0.986            NaN            0.002              0.001
disp1_m2     2903  18956  0.958  0.990    0.989              0.542          0.991               0.989  0.992          0.992            0.001              0.001
drop2_m2     1162  1504   0.767  0.994    0.994              0.500          0.995               0.993  0.995          0.995            0.001              0.001
drop2_m4     1769  885    0.707  0.992    0.989              0.501          0.991               0.990  0.993            NaN            0.001             -0.000
disp1_m1     2537  19834  0.969  0.995    0.994              0.556          0.995               0.994  0.996            NaN            0.001             -0.000
grip_m4      4855  15980  0.857  0.956    0.942              0.516          0.956               0.952  0.956            NaN            0.000              0.000
dropfail1_m2 167   1389   0.880  0.938    0.906              0.499          0.942               0.934  0.939            NaN            0.000              0.004
disp2_m2     4169  17690  0.983  0.997    0.997              0.501          0.998               0.997  0.998            NaN            0.000              0.000
disp2_m1     3566  18805  0.986  0.999    0.998              0.506          0.999               0.998  0.999            NaN            0.000              0.000
whichpot_m4  5236  3413   0.998  1.000    0.997              0.507          1.000               1.000  1.000            NaN            0.000             -0.000
drop1_m1     322   1539   0.945  0.983    0.979              0.491          0.982               0.976  0.983            NaN            0.000             -0.001
disp2_m4     5244  15591  0.975  0.994    0.986              0.503          0.994               0.993  0.994            NaN            0.000              0.000
whichpot_m1  3563  2534   1.000  1.000    1.000              0.531          1.000               1.000  1.000            NaN            0.000              0.000
whichpot_m2  4165  2899   0.999  1.000    1.000              0.516          1.000               1.000  1.000          1.000            0.000             -0.000
jlim_m1      59    22312  0.643  0.858    0.809              0.576          0.794               0.823  0.856            NaN           -0.002             -0.064
grip_m2      2464  19395  0.858  0.983    0.974              0.526          0.983               0.982  0.981            NaN           -0.002              0.000
dropfail2_m1 65    2608   0.698  0.984    0.976              0.521          0.986               0.983  0.982            NaN           -0.003              0.002
grip_m1      1240  21131  0.841  0.991    0.982              0.528          0.991               0.991  0.988            NaN           -0.004              0.000
drop1_m2     511   1045   0.953  0.991    0.989              0.507          0.991               0.990  0.987            NaN           -0.004              0.000
dropfail2_m2 73    2593   0.713  0.977    0.972              0.550          0.978               0.977  0.973            NaN           -0.004              0.000
drop1_m4     373   575    0.924  0.988    0.980              0.488          0.987               0.987  0.981            NaN           -0.008             -0.001
jlim_m4      149   20686  0.645  0.810    0.810              0.504          0.823               0.790  0.801            NaN           -0.010              0.013
twist_m4     74    20761  0.875  0.919    0.906              0.528          0.928               0.909  0.901            NaN           -0.017              0.010
dropfail1_m1 60    1801   0.862  0.939    0.903              0.500          0.919               0.891  0.914            NaN           -0.025             -0.019
```

## 4. Corpus A: routing increment over the deployable control (grouped bootstrap 95% CI)

```
       event  n_pos   delta      lo      hi   sig
dropfail1_m2     99  0.0734  0.0425  0.0886  True
    twist_m4     78  0.0478 -0.0398  0.1066 False
    tilt1_m2    129  0.0277 -0.0036  0.0656 False
    tilt1_m1     33  0.0211 -0.0503  0.0595 False
    tilt2_m1    122  0.0186 -0.0006  0.0558 False
  dispnc1_m4    122  0.0177  0.0044  0.0483  True
    drop1_m1    203  0.0152 -0.0038  0.0178 False
    disp2_m4   3235  0.0089  0.0020  0.0134  True
    twist_m2     43  0.0073 -0.0217  0.0477 False
    disp2_m2   2329  0.0062 -0.0022  0.0152 False
    drop1_m2    373  0.0056 -0.0064  0.0077 False
    disp2_m1   1871  0.0049  0.0007  0.0088  True
    drop1_m4    324  0.0038 -0.0061  0.0060 False
    tilt2_m2    262  0.0029 -0.0127  0.0212 False
    disp1_m1   1142  0.0023  0.0012  0.0046  True
    disp1_m4   1526  0.0023 -0.0012  0.0228 False
dropfail2_m2     67  0.0015 -0.0219  0.0064 False
 whichpot_m4   3192  0.0012  0.0002  0.0016  True
    disp1_m2   1359  0.0009 -0.0007  0.0075 False
 whichpot_m2   2295  0.0002 -0.0016  0.0008 False
     grip_m1   1042 -0.0003 -0.0010  0.0005 False
 whichpot_m1   1852 -0.0003 -0.0032  0.0004 False
     grip_m2   2070 -0.0003 -0.0019  0.0016 False
     grip_m4   4093 -0.0012 -0.0036  0.0017 False
    tilt1_m4    394 -0.0021 -0.0118  0.0157 False
    drop2_m4    800 -0.0028 -0.0115  0.0207 False
dropfail2_m1     64 -0.0046 -0.0626  0.0028 False
    drop2_m2    548 -0.0057 -0.0188  0.0145 False
    twist_m1    133 -0.0071 -0.0253  0.0074 False
     jext_m1    324 -0.0082 -0.0357  0.0175 False
    drop2_m1    315 -0.0152 -0.0294  0.0210 False
    tilt2_m4    497 -0.0186 -0.0695  0.0246 False
     jext_m4    300 -0.0229 -0.0724  0.0229 False
     jext_m2    316 -0.0315 -0.0798  0.0026 False
dropfail1_m1     37 -0.0320 -0.0351  0.0000 False
  dispnc2_m4     40 -0.0385 -0.2170 -0.0024  True
 goalreg2_m4     36 -0.0656 -0.0839  0.0315 False
dropfail1_m4     48 -0.1749 -0.2530 -0.0071  True
```

## 4. Corpus A: vs capacity-matched permuted-routing null (grouped bootstrap 95% CI)

```
       event  n_pos   delta      lo     hi   sig
    twist_m4     78  0.1081  0.0530 0.1391  True
dropfail1_m2     99  0.0944  0.0380 0.0953  True
    tilt1_m1     33  0.0689  0.0229 0.1589  True
    twist_m2     43  0.0440 -0.0052 0.1162 False
    tilt2_m2    262  0.0317 -0.0041 0.0605 False
dropfail2_m1     64  0.0314 -0.0068 0.1583 False
    tilt2_m1    122  0.0294  0.0145 0.0822  True
     jext_m1    324  0.0253 -0.0601 0.1199 False
  dispnc1_m4    122  0.0196  0.0076 0.0474  True
    drop1_m1    203  0.0125  0.0080 0.0299  True
    disp2_m4   3235  0.0119  0.0043 0.0180  True
dropfail1_m1     37  0.0104 -0.0958 0.1892 False
dropfail2_m2     67  0.0088 -0.0454 0.0279 False
     jext_m2    316  0.0087 -0.0414 0.0600 False
    disp2_m2   2329  0.0070 -0.0033 0.0227 False
    disp2_m1   1871  0.0061  0.0018 0.0105  True
    drop1_m2    373  0.0060 -0.0093 0.0079 False
    tilt1_m2    129  0.0048 -0.0244 0.0306 False
    twist_m1    133  0.0039 -0.0042 0.0116 False
    tilt1_m4    394  0.0033 -0.0097 0.0285 False
    disp1_m1   1142  0.0023  0.0011 0.0098  True
    disp1_m4   1526  0.0022 -0.0002 0.0170 False
    drop1_m4    324  0.0016 -0.0246 0.0029 False
    disp1_m2   1359  0.0015 -0.0000 0.0100 False
 whichpot_m4   3192  0.0010  0.0002 0.0013  True
     grip_m2   2070  0.0009 -0.0004 0.0024 False
     grip_m1   1042  0.0006 -0.0003 0.0020 False
    drop2_m2    548  0.0005 -0.0170 0.0233 False
 whichpot_m2   2295  0.0001 -0.0019 0.0007 False
 whichpot_m1   1852 -0.0004 -0.0033 0.0003 False
    drop2_m4    800 -0.0006 -0.0121 0.0247 False
     grip_m4   4093 -0.0007 -0.0024 0.0014 False
    drop2_m1    315 -0.0134 -0.0235 0.0297 False
    tilt2_m4    497 -0.0166 -0.0740 0.0299 False
     jext_m4    300 -0.0195 -0.0737 0.0486 False
 goalreg2_m4     36 -0.0251 -0.0310 0.0619 False
  dispnc2_m4     40 -0.0429 -0.2230 0.0082 False
dropfail1_m4     48 -0.1028 -0.1569 0.0526 False
```

## 4. Corpus B: routing increment over the deployable control (grouped bootstrap 95% CI)

```
       event  n_pos   delta      lo      hi   sig
     jlim_m2     89  0.0167 -0.0395  0.0358 False
     jlim_m4    149  0.0129 -0.0912  0.0512 False
     jext_m1    448  0.0104  0.0038  0.0278  True
    twist_m4     74  0.0097 -0.0456  0.0323 False
     jext_m2    438  0.0068 -0.0101  0.0502 False
    tilt2_m4    280  0.0059 -0.0053  0.0121 False
    tilt1_m2    219  0.0039 -0.0056  0.0170 False
dropfail1_m2    167  0.0037 -0.0063  0.0126 False
    tilt1_m1     59  0.0025 -0.0254  0.0263 False
    tilt1_m4    734  0.0024 -0.0011  0.0062 False
dropfail2_m1     65  0.0021 -0.0051  0.0124 False
    drop2_m2   1162  0.0014 -0.0000  0.0031 False
    disp1_m4   3421  0.0014  0.0003  0.0027  True
    disp1_m2   2903  0.0006 -0.0004  0.0018 False
     grip_m2   2464  0.0004 -0.0013  0.0018 False
    disp2_m2   4169  0.0003 -0.0000  0.0007 False
     grip_m4   4855  0.0003 -0.0031  0.0035 False
dropfail2_m2     73  0.0002 -0.0118  0.0117 False
    disp2_m1   3566  0.0002 -0.0000  0.0004 False
     grip_m1   1240  0.0002 -0.0007  0.0011 False
    drop1_m2    511  0.0001 -0.0032  0.0032 False
    disp2_m4   5244  0.0000 -0.0012  0.0012 False
 whichpot_m1   3563  0.0000  0.0000  0.0001 False
 whichpot_m2   4165 -0.0000 -0.0001  0.0000 False
 whichpot_m4   5236 -0.0000 -0.0001  0.0000 False
    drop2_m4   1769 -0.0001 -0.0034  0.0034 False
    disp1_m1   2537 -0.0001 -0.0013  0.0009 False
    drop1_m1    322 -0.0008 -0.0033  0.0013 False
  dispnc1_m4    423 -0.0009 -0.0036  0.0013 False
    drop1_m4    373 -0.0011 -0.0056  0.0030 False
    tilt2_m2    140 -0.0028 -0.0196  0.0092 False
     jext_m4    414 -0.0031 -0.0712  0.0080 False
dropfail1_m1     60 -0.0194 -0.0407 -0.0027  True
    tilt2_m1     45 -0.0199 -0.0346  0.0054 False
     jlim_m1     59 -0.0639 -0.0813 -0.0158  True
```

## 4. Corpus B: vs capacity-matched permuted-routing null (grouped bootstrap 95% CI)

```
       event  n_pos   delta      lo     hi   sig
    tilt1_m1     59  0.0417  0.0166 0.0777  True
     jlim_m2     89  0.0372 -0.0044 0.0493 False
     jlim_m4    149  0.0338 -0.0725 0.0726 False
dropfail1_m1     60  0.0283  0.0075 0.0655  True
     jext_m1    448  0.0205  0.0124 0.0657  True
    twist_m4     74  0.0198 -0.0247 0.0449 False
    tilt2_m4    280  0.0185 -0.0054 0.0320 False
     jext_m2    438  0.0146 -0.0033 0.0675 False
    tilt1_m2    219  0.0144 -0.0003 0.0370 False
dropfail1_m2    167  0.0082 -0.0043 0.0172 False
    drop1_m1    322  0.0061  0.0013 0.0117  True
    tilt1_m4    734  0.0048  0.0008 0.0095  True
     grip_m4   4855  0.0045  0.0009 0.0081  True
    tilt2_m2    140  0.0041 -0.0198 0.0169 False
dropfail2_m1     65  0.0032 -0.0037 0.0143 False
    disp1_m4   3421  0.0023  0.0010 0.0040  True
    drop2_m2   1162  0.0022  0.0005 0.0041  True
     grip_m2   2464  0.0017 -0.0001 0.0031 False
    disp1_m2   2903  0.0013  0.0003 0.0027  True
    drop2_m4   1769  0.0011 -0.0028 0.0051 False
    drop1_m2    511  0.0010 -0.0025 0.0043 False
    disp2_m4   5244  0.0009 -0.0005 0.0024 False
     grip_m1   1240  0.0004 -0.0003 0.0013 False
dropfail2_m2     73  0.0004 -0.0072 0.0049 False
    disp2_m1   3566  0.0004  0.0001 0.0008  True
    disp2_m2   4169  0.0004  0.0001 0.0006  True
    drop1_m4    373  0.0003 -0.0017 0.0021 False
    disp1_m1   2537  0.0003 -0.0009 0.0013 False
 whichpot_m1   3563  0.0000  0.0000 0.0001 False
 whichpot_m4   5236 -0.0000 -0.0001 0.0001 False
 whichpot_m2   4165 -0.0000 -0.0001 0.0000 False
  dispnc1_m4    423 -0.0002 -0.0030 0.0018 False
     jext_m4    414 -0.0025 -0.0813 0.0089 False
    tilt2_m1     45 -0.0037 -0.0268 0.0112 False
     jlim_m1     59 -0.0287 -0.0539 0.0250 False
```

## 5. Mirror-selection symmetry check

Restricting to rows where tier-1's out-of-fold probability sits near the base
rate makes routing look far better than tier-1. The mirror selection (restrict
on routing's own score, read tier-1) produces an equal or larger advantage for
tier-1. The effect is symmetric, i.e. a range-restriction artifact of
conditioning on a model's own score, not evidence of routing-specific
information. Both directions are in stratified_A2.csv / stratified_B2.csv.

## 6. Cells dropped for <30 positives (named, per the protocol)

| event family | corpus A positives (m=1/2/4) | corpus B | verdict |
|---|---|---|---|
| leavegoal1 (pot_1 leaves goal) | 0 / 0 / 0 (pop 214/90/44) | 4 / 9 / 14 | dropped both |
| leavegoal2 (pot_2 leaves goal) | 4 / 7 / 15 (pop 6058) | 0 / 0 / 0 (pop 13126) | dropped both |
| dispfar1/2 (motion, no contact anywhere in window) | 1 / 1 / 1 and 5 / 5 / 5 | 1 / 1 / 0 and 0 / 2 / 4 | dropped both |
| dispnc1/2 m=1,2 (motion, gripper open, eef >15 cm) | 1 / 2 and 5 / 5 | 1 / 15 and 0 / 2 | dropped; only m=4 survives |
| goalregnc1/2 (goal regression, no contact) | 0-2 | 0 | dropped both |
| whichpotnc (which pot, no contact) | 3 / 4 / 38 | 0 / 0 / 18 | dropped both |
| jlim (within 10% of a joint limit) | 0 / 0 / 0 | 59 / 89 / 149 | dropped in A, kept in B |
| twist (large dq, small d-eef) | 133 / 43 / 78 | 24 / 5 / 74 | kept in A; only m=4 in B |
| twistq (percentile version) | 30 / 5 / 4 | 3 / 1 / 0 | dropped both |
| goalreg1_m4 | 8 | 23 | dropped both |
| dropfail2_m4 | 28 | 3 | dropped both |
| tilt1_m1 (A) | 33 | 59 | kept, flagged as marginal |
