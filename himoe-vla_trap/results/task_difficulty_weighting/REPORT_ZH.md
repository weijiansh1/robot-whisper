# MoE 路由、任务难度与报警加权

## 核心结论

本实验固定使用 37 个任务、14800 条轨迹。任务难度定义为当前 checkpoint、初始状态分布、noise 和 horizon 下的经验失败率，而不是任务固有难度。

难度是可重复测量的：互斥 seed halves 的失败率 Spearman `rho=0.837`。但第一个 query 的 MoE 路由仍没有形成可靠难度轴：BH 校正后显著标量为 0 个，路由距离与难度差的相关的 cross-fit 平均值为 `rho=0.120`，置换 `p=0.118`。

同一份第一个 query 路由却能在 held-out init 上以 99.0% 识别具体任务、100.0% 识别 suite。这再次说明 routing 主要编码任务/场景，不等于编码难度。

## 难度定义与稳定性

最难任务为 `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`，失败率 34.5%。所有任务失败率范围为 0.0%--34.5%。平均 episode 长度与失败率的相关仅为 `rho=0.145`，所以长任务不能直接当难任务。

| task | failures | failure_rate | failure_rate_wilson_low | failure_rate_wilson_high |
| --- | --- | --- | --- | --- |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 138 | 0.3450 | 0.3001 | 0.3929 |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 52 | 0.1300 | 0.1005 | 0.1665 |
| libero_long/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 43 | 0.1075 | 0.0808 | 0.1417 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 35 | 0.0875 | 0.0636 | 0.1193 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 25 | 0.0625 | 0.0427 | 0.0906 |
| libero_long/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 17 | 0.0425 | 0.0267 | 0.0670 |
| libero_goal/put_the_bowl_on_top_of_the_cabinet | 15 | 0.0375 | 0.0229 | 0.0609 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 14 | 0.0350 | 0.0210 | 0.0579 |
| libero_goal/put_the_bowl_on_the_plate | 12 | 0.0300 | 0.0172 | 0.0517 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 12 | 0.0300 | 0.0172 | 0.0517 |

## 第一个 query 的 MoE 能否预测难度

物理 outcome 在第一个 query 路由特征和路由 embedding 全部写盘后才加载。每个任务用 8 层 x 32 experts 的 final-flow action-route 均值作为路由签名；路由签名始终只预测另一组 noise seeds 的失败率，并交换两半复算。

最强的早期标量是 `back_front_volatility_ratio`，两方向平均 Spearman `rho=-0.233`，置换检验 `p=0.1512`，BH 后 `q=0.899`。没有早期标量通过 `q<=0.05`。

| signal | fold_1000_1003_routes_rho | fold_1004_1007_routes_rho | mean_crossfit_spearman_rho | permutation_p_two_sided | bh_q_value | mean_rho_without_hardest_task |
| --- | --- | --- | --- | --- | --- | --- |
| back_front_volatility_ratio | -0.2047 | -0.2610 | -0.2328 | 0.1512 | 0.8990 | -0.2085 |
| back_token_disagreement | -0.1410 | -0.2077 | -0.1743 | 0.2799 | 0.8990 | -0.1099 |
| front_token_disagreement | -0.1353 | -0.1128 | -0.1241 | 0.4379 | 0.8990 | -0.0610 |
| back_top12_margin | -0.0619 | -0.1302 | -0.0961 | 0.5485 | 0.8990 | -0.0714 |
| back_front_acceleration_ratio | -0.0794 | -0.1151 | -0.0972 | 0.5647 | 0.8990 | -0.0609 |
| front_route_acceleration | 0.1142 | 0.0655 | 0.0899 | 0.5763 | 0.8990 | 0.1273 |
| layer5_state_action_gap | 0.1534 | 0.0250 | 0.0892 | 0.5859 | 0.8990 | 0.1161 |
| front_late_flow_volatility | 0.0745 | 0.0644 | 0.0694 | 0.6627 | 0.8990 | 0.1051 |

路由 KNN 的正 `mae_improvement_over_median` 才表示优于不知道路由时的 leave-one-task median。结果如下：

| k | mean_fold_spearman_rho | mae | median_baseline_mae | mae_improvement_over_median | permutation_p_route_geometry |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1050 | 0.0317 | 0.0281 | -0.0035 | 0.0612 |
| 3 | 0.1288 | 0.0346 | 0.0281 | -0.0064 | 0.2028 |
| 5 | 0.1910 | 0.0334 | 0.0281 | -0.0053 | 0.1404 |
| 10 | 0.1139 | 0.0316 | 0.0281 | -0.0035 | 0.0754 |

## 固定报警预算下的难度加权

这是 50%--90% episode window 的离线 allocation stress test，不是当前 v2 的逐 query 在线性能。两个 seed halves 互换 calibration/evaluation；difficulty prior、健康分位数和阈值均只来自另一半。固定 detector 是 instability 与 lock-in 的等权均值，没有拟合 feature weight。所有方法在 calibration 上的目标 success alarm budget 均为 5%。

| variant | true_alarms | false_alarms | failure_recall | success_false_alarm_rate | precision | delta_recall_vs_uniform | delta_recall_ci_low | delta_recall_ci_high | failure_recall_excluding_hardest_task | macro_task_failure_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 318 | 667 | 0.6530 | 0.0466 | 0.3228 | 0.0000 | 0.0000 | 0.0000 | 0.7393 | 0.6517 |
| empirical_sqrt | 364 | 700 | 0.7474 | 0.0489 | 0.3421 | 0.0945 | -0.0193 | 0.1878 | 0.7507 | 0.6278 |
| empirical_linear | 371 | 689 | 0.7618 | 0.0481 | 0.3500 | 0.1088 | -0.0144 | 0.1940 | 0.7708 | 0.6133 |
| route_knn5_sqrt | 317 | 686 | 0.6509 | 0.0479 | 0.3161 | -0.0021 | -0.0216 | 0.0162 | 0.7364 | 0.6343 |

均匀预算命中 318/487；经验难度 sqrt 加权命中 364，线性加权命中 371。route-KNN 难度加权命中 317。是否有收益必须同时看实际 FPR 和 task-cluster CI，不能只比较命中数。

线性经验先验的净增量为 53 个失败命中，其中最难任务 `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` 单独贡献 42 个。剔除它后，召回变化为 +3.2%；跨任务 bootstrap 的总召回差 95% CI 为 [-1.4%, +19.4%]。

它优化的是按失败样本计的 micro recall；若每个有失败的任务等权，macro recall 反而从 65.2% 降至 61.3%。也就是说，难度加权明确牺牲低失败率任务，把报警预算集中到高失败率任务；是否值得取决于部署效用函数。

两个交换方向的线性加权命中分别为 184/250 和 187/237；对应均匀预算为 153/250 和 165/237。方向一致，但不代表能泛化到新任务。

## 解释边界

- difficulty prior 使用 disjoint rollout outcomes，属于 label-based calibration，不是纯 MoE-only。
- first-query route difficulty 部分不读取 outcome，但评价时需要 task failure rate。
- 报警加权实验使用 episode 中后段汇总路由，是 allocation 原理检查，不是 early alarm。
- 经验难度只对当前 checkpoint、数据分布和 horizon 有效。
- 若难度权重有效，它改变的是报警预算分配，不证明 MoE 本身理解了任务难度。
