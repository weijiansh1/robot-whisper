# 任务内无监督 loop motif

每个任务独立在等长前缀的 positive-topology 事件上拟合 motif。
聚类完成后才读取 outcome；phase、period、task identity 和 outcome 均未进入拟合。

## 拟合概览

| task | episodes / failures | fit / full positive events | selected K | silhouette | restart ARI | status |
|---|---:|---:|---:|---:|---:|---|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 512 / 0 | 198 / 199 | 2 | 0.211 | 0.584 | low_stability |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 512 / 42 | 3049 / 3767 | 3 | 0.273 | 0.993 | ok |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 512 / 216 | 8542 / 12116 | 2 | 0.253 | 0.993 | ok |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 512 / 12 | 475 / 644 | 2 | 0.208 | 0.588 | low_stability |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 512 / 37 | 171 / 327 | 2 | 0.305 | 1.000 | ok |

Cluster rate 的检验只使用等长前缀，并在该任务的 initial-state 内比较。
`task FWER p` 校正该任务的全部 cluster rate；`4-task p` 再对四个含失败任务作 Bonferroni 校正。

## libero_goal/open_the_middle_drawer_of_the_cabinet

| cluster | fit / full events | median loop score | median min XRA recovery | >=5% recovery | median lag / phase | prefix rate failure / success | failure AUC | task FWER p | 4-task p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 105 / 105 | 0.388 | 1.46% | 1.0% | 8.000 / 0.750 | - / 0.023 | - | - | - |
| 2 | 93 / 94 | 0.465 | 1.53% | 2.1% | 6.000 / 0.909 | - / 0.021 | - | - | - |

## libero_goal/open_the_top_drawer_and_put_the_bowl_inside

| cluster | fit / full events | median loop score | median min XRA recovery | >=5% recovery | median lag / phase | prefix rate failure / success | failure AUC | task FWER p | 4-task p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 868 / 887 | 0.482 | 1.78% | 6.1% | 6.000 / 0.389 | 0.087 / 0.125 | 0.505 | 0.8770 | 1.0000 |
| 2 | 896 / 1269 | 0.402 | 3.58% | 29.9% | 14.000 / 0.882 | 0.061 / 0.131 | 0.356 | 1.0000 | 1.0000 |
| 3 | 1285 / 1611 | 0.746 | 4.47% | 42.1% | 9.000 / 0.526 | 0.177 / 0.180 | 0.448 | 0.9960 | 1.0000 |

## libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

| cluster | fit / full events | median loop score | median min XRA recovery | >=5% recovery | median lag / phase | prefix rate failure / success | failure AUC | task FWER p | 4-task p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3544 / 4662 | 0.387 | 2.47% | 10.2% | 15.000 / 0.550 | 0.220 / 0.213 | 0.467 | 0.9984 | 1.0000 |
| 2 | 4998 / 7454 | 0.651 | 6.64% | 73.6% | 19.000 / 0.686 | 0.328 / 0.288 | 0.640 | 0.0002 | 0.0008 |

## libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate

| cluster | fit / full events | median loop score | median min XRA recovery | >=5% recovery | median lag / phase | prefix rate failure / success | failure AUC | task FWER p | 4-task p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 172 / 224 | 0.548 | 2.74% | 17.0% | 5.000 / 0.667 | 0.000 / 0.060 | 0.307 | 1.0000 | 1.0000 |
| 2 | 303 / 420 | 0.680 | 4.55% | 42.4% | 6.000 / 0.750 | 0.083 / 0.105 | 0.406 | 0.9966 | 1.0000 |

## libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate

| cluster | fit / full events | median loop score | median min XRA recovery | >=5% recovery | median lag / phase | prefix rate failure / success | failure AUC | task FWER p | 4-task p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 103 / 168 | 0.544 | 1.36% | 6.5% | 8.000 / 0.583 | 0.128 / 0.017 | 0.810 | 0.0002 | 0.0008 |
| 2 | 68 / 159 | 0.334 | 2.43% | 11.9% | 8.000 / 0.667 | 0.017 / 0.017 | 0.509 | 0.6151 | 1.0000 |

## 事后组成效应检查

这一节不是预注册主检验。它检查显著 cluster rate 是否只是在复述 positive X/R/A-return topology 的总体频率。

| task | positive-topology rate failure / success | failure AUC | task p | 4-task p |
|---|---:|---:|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | - / 0.044 | - | - | - |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 0.325 / 0.436 | 0.386 | 0.9538 | 1.0000 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.549 / 0.501 | 0.650 | 0.0002 | 0.0008 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.083 / 0.165 | 0.205 | 1.0000 | 1.0000 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.145 / 0.034 | 0.823 | 0.0002 | 0.0008 |

## 结论

任务内无监督发现了以下 failure-associated motif（cluster-rate AUC > 0.5 且 task FWER p < 0.05）：
- libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove / cluster 2: AUC=0.640, task FWER p=0.0002, median min recovery=6.64%.
- libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate / cluster 1: AUC=0.810, task FWER p=0.0002, median min recovery=1.36%.
2 个 motif 在进一步的四任务 Bonferroni 校正后仍保留。

对上述显著 motif，再只看至少含一个 positive-topology 事件的 episode，并比较该 motif 在所有 positive-topology 事件中的占比：
- libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove / cluster 2: 512 episodes (216 failures), share AUC=0.577, task FWER p=0.0404, 4-task p=0.1616.
- libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate / cluster 1: 131 episodes (30 failures), share AUC=0.535, task FWER p=0.2889, 4-task p=1.0000.
若 composition effect 不保留，cluster-rate 关联主要应解释为回返拓扑总体更频繁，而不是某个无监督几何类型特异地增多。
簇是任务内几何类型，不是滑落、空抓等语义失败标签。
恢复幅度仍需单独检查；高 AUC 的 micro-return 不能称为强 basin loop。

## 限制

- 聚类是 transductive：使用所有 episode 的无标签等长前缀。
- X 仍是 qpos，而不是 RGB/vision embedding。
- 事件使用下一 query，只能用于发现而不是提前预警。
- Outcome 关联不能证明 routing recurrence 导致失败。
