# 前期路由熵信号

## 设计

- 数据：scene8，16 个初始状态 × 32 个 flow-noise seed，共 512 条 rollout。
- 熵：从完整 32-way soft router probabilities 以 float32 重算 Shannon entropy，并除以 log(32)；1 表示均匀路由。
- 主指标：同一初始状态内的 pair-weighted AUC。AUC > 0.5 表示失败分支熵更高，AUC < 0.5 表示失败分支熵更低。
- 标签 A：任意 timeout failure（216）vs success（296）；使用共同 seed-column 置换，保留跨初态共享 seed 结构。
- 标签 B：行为定义的停滞失败（197）vs 其余全部（315）；使用共同 seed-column 置换。
- 标签 C：停滞失败（197）vs success（296），排除 19 条其他失败；使用初态内置换。
- 搜索校正：5000 次置换；粗粒度前期统计与 q7/q12 层/去噪定位扫描分别做 maxT。

## 早期结果

| target | feature | AUC [hierarchical 95% CI] | discrimination | entropy delta | raw p | coarse maxT p |
|---|---|---:|---:|---:|---:|---:|
| any_timeout_failure | t0/current/action_all | 0.524 [0.427, 0.612] | 0.524 | +3.46e-06 | 0.4917 | 1.0000 |
| any_timeout_failure | t7/current/action_all | 0.489 [0.380, 0.581] | 0.511 | -3.76e-06 | 0.7538 | 1.0000 |
| any_timeout_failure | t7/recent_mean/action_all | 0.462 [0.355, 0.566] | 0.538 | -4.72e-06 | 0.2507 | 1.0000 |
| any_timeout_failure | t12/current/action_all | 0.582 [0.443, 0.722] | 0.582 | +4.00e-05 | 0.0264 | 0.5339 |
| any_timeout_failure | t12/recent_mean/action_all | 0.569 [0.467, 0.661] | 0.569 | +7.09e-06 | 0.0652 | 0.8478 |
| any_timeout_failure | t12/recent_slope/action_all | 0.609 [0.476, 0.743] | 0.609 | +1.82e-05 | 0.0010 | 0.0796 |
| any_timeout_failure | t12/current/state | 0.545 [0.435, 0.661] | 0.545 | +5.67e-05 | 0.2354 | 0.9992 |
| any_timeout_failure | t12/current/all_tokens | 0.541 [0.429, 0.655] | 0.541 | +4.16e-05 | 0.2745 | 1.0000 |
| stasis_vs_rest | t0/current/action_all | 0.548 [0.451, 0.633] | 0.548 | +7.44e-06 | 0.1536 | 0.9998 |
| stasis_vs_rest | t7/current/action_all | 0.495 [0.389, 0.595] | 0.505 | -9.93e-07 | 0.8884 | 1.0000 |
| stasis_vs_rest | t7/recent_mean/action_all | 0.489 [0.374, 0.605] | 0.511 | -2.72e-06 | 0.7656 | 1.0000 |
| stasis_vs_rest | t12/current/action_all | 0.570 [0.412, 0.716] | 0.570 | +3.47e-05 | 0.0794 | 0.8464 |
| stasis_vs_rest | t12/recent_mean/action_all | 0.575 [0.467, 0.679] | 0.575 | +7.22e-06 | 0.0424 | 0.7590 |
| stasis_vs_rest | t12/recent_slope/action_all | 0.609 [0.472, 0.740] | 0.609 | +1.85e-05 | 0.0014 | 0.0984 |
| stasis_vs_rest | t12/current/state | 0.518 [0.413, 0.650] | 0.518 | -8.37e-04 | 0.6609 | 1.0000 |
| stasis_vs_rest | t12/current/all_tokens | 0.502 [0.394, 0.632] | 0.502 | -4.45e-05 | 0.9648 | 1.0000 |
| clean_stasis_failure | t0/current/action_all | 0.540 [0.443, 0.629] | 0.540 | +6.17e-06 | 0.3211 | 1.0000 |
| clean_stasis_failure | t7/current/action_all | 0.490 [0.380, 0.590] | 0.510 | -3.10e-06 | 0.8016 | 1.0000 |
| clean_stasis_failure | t7/recent_mean/action_all | 0.480 [0.354, 0.594] | 0.520 | -3.66e-06 | 0.6141 | 1.0000 |
| clean_stasis_failure | t12/current/action_all | 0.595 [0.442, 0.744] | 0.595 | +4.45e-05 | 0.0186 | 0.4793 |
| clean_stasis_failure | t12/recent_mean/action_all | 0.578 [0.468, 0.685] | 0.578 | +7.90e-06 | 0.0478 | 0.8316 |
| clean_stasis_failure | t12/recent_slope/action_all | 0.623 [0.479, 0.756] | 0.623 | +2.12e-05 | 0.0018 | 0.0794 |
| clean_stasis_failure | t12/current/state | 0.522 [0.410, 0.654] | 0.522 | -1.25e-03 | 0.5791 | 1.0000 |
| clean_stasis_failure | t12/current/all_tokens | 0.514 [0.404, 0.642] | 0.514 | -7.35e-05 | 0.7229 | 1.0000 |

## q12 token 位置

| target | token group | AUC | entropy delta | states higher / informative |
|---|---|---:|---:|---:|
| any_timeout_failure | state | 0.545 | +5.67e-05 | 8/13 |
| any_timeout_failure | action_1_3 | 0.588 | +6.15e-05 | 9/13 |
| any_timeout_failure | action_4_7 | 0.564 | +1.87e-05 | 6/13 |
| any_timeout_failure | action_8_10 | 0.581 | +4.70e-05 | 6/13 |
| any_timeout_failure | action_all | 0.582 | +4.00e-05 | 9/13 |
| stasis_vs_rest | state | 0.518 | -8.37e-04 | 7/14 |
| stasis_vs_rest | action_1_3 | 0.586 | +6.01e-05 | 6/14 |
| stasis_vs_rest | action_4_7 | 0.555 | +1.66e-06 | 7/14 |
| stasis_vs_rest | action_8_10 | 0.588 | +5.33e-05 | 7/14 |
| stasis_vs_rest | action_all | 0.570 | +3.47e-05 | 8/14 |
| clean_stasis_failure | state | 0.522 | -1.25e-03 | 7/13 |
| clean_stasis_failure | action_1_3 | 0.598 | +6.79e-05 | 8/13 |
| clean_stasis_failure | action_4_7 | 0.573 | +1.60e-05 | 6/13 |
| clean_stasis_failure | action_8_10 | 0.598 | +5.92e-05 | 6/13 |
| clean_stasis_failure | action_all | 0.595 | +4.45e-05 | 8/13 |

## 搜索审计

- `any_timeout_failure` 粗粒度最强：`t12/recent_slope/action_all`，AUC=0.609，coarse maxT p=0.0796；定位扫描最强：`t12/current/action/L5_d5`，AUC=0.643，localization maxT p=0.0048。
- `stasis_vs_rest` 粗粒度最强：`t12/recent_slope/action_all`，AUC=0.609，coarse maxT p=0.0984；定位扫描最强：`t12/current/action/L4_d0`，AUC=0.650，localization maxT p=0.0056。
- `clean_stasis_failure` 粗粒度最强：`t12/recent_slope/action_all`，AUC=0.623，coarse maxT p=0.0794；定位扫描最强：`t12/current/action/L5_d4`，AUC=0.654，localization maxT p=0.0114。

## 整初态留出

- `any_timeout_failure` 4-fold：重选粗粒度特征后 AUC=0.485，reselection p=0.4645；重选层/去噪位置后 AUC=0.453，reselection p=0.7165。
- `any_timeout_failure` LOSO：重选粗粒度特征后 AUC=0.524，reselection p=0.3067；重选层/去噪位置后 AUC=0.512，reselection p=0.3597。
- `stasis_vs_rest` 4-fold：重选粗粒度特征后 AUC=0.401，reselection p=0.9324；重选层/去噪位置后 AUC=0.436，reselection p=0.7297。
- `stasis_vs_rest` LOSO：重选粗粒度特征后 AUC=0.464，reselection p=0.6247；重选层/去噪位置后 AUC=0.603，reselection p=0.0578。
- `clean_stasis_failure` 4-fold：重选粗粒度特征后 AUC=0.408，reselection p=0.9796；重选层/去噪位置后 AUC=0.441，reselection p=0.8844。
- `clean_stasis_failure` LOSO：重选粗粒度特征后 AUC=0.514，reselection p=0.4013；重选层/去噪位置后 AUC=0.476，reselection p=0.6113。

## 后期对照

| q | state-token current | action current | all-token current |
|---:|---:|---:|---:|
| 20 | 0.588 | 0.517 | 0.587 |
| 27 | 0.491 | 0.574 | 0.503 |
| 34 | 0.715 | 0.316 | 0.711 |

## 结论边界

- q0/q7 没有稳定的全局路由熵信号。
- q12 action entropy 是弱、局部且初态异质的相关信号；它不等同于已经识别到停滞。
- 整初态留出后，任意超时和 clean 停滞目标都回到机会线；当前没有可迁移的单独熵预警量。
- `stasis_vs_rest` 的 q12 定位在 LOSO 下仅为边缘趋势，且 4-fold 结果不一致，必须在新任务/新初态上固定位置后再验证。
- normalized entropy 整体接近 1，组间差值很小，因此它更适合作为组合特征，而不是单独告警量。
- q20 以后仅作为阶段读数；多数停滞失败的行为 onset 中位为 t19，不能称为前兆。
