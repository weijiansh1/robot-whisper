# 逐 chunk trap onset 的 MoE 实验

## 结论

**临近 1–4 query 时，MoE 没有超过物理/动作前缀控制，不能作为独立 trap 报警器。**

9–12 个 query 提前量出现了两种拆分同方向的 token-mean 候选信号，但未通过 9 重校正，只能列为探索性发现。

本实验不再预测最终 success/failure，而是在尚未发生 trap 的风险集中，预测未来固定 query 区间的首次物理事件。

## 事件覆盖

| event | branches | successful |
| --- | --- | --- |
| loop onset | 172 | 2 |
| 80-action static onset | 54 | 0 |
| either trap onset | 210 | 2 |
| post-trap recovery >=3.5cm | 6 | 2 |

恢复阈值是 onset 后目标距离至少再改善 3.5 cm。恢复样本过少，不训练恢复分类器。

## 主结果：同 snapshot、同 q 的未来 trap

| lead | model | snapshots | AUC | CI low | CI high | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| next_1_4 | physical_action | 17 | 0.907 | 0.855 | 0.952 | 0.023 |
| next_1_4 | physical | 17 | 0.897 | 0.850 | 0.940 | 0.030 |
| next_1_4 | physical_action+aggregate_current | 17 | 0.897 | 0.842 | 0.943 | 0.023 |
| next_1_4 | physical+aggregate_current | 17 | 0.901 | 0.861 | 0.938 | 0.029 |
| next_1_4 | physical_action+token_current | 17 | 0.896 | 0.833 | 0.949 | 0.025 |
| next_1_4 | physical_action+token_history | 17 | 0.884 | 0.821 | 0.939 | 0.025 |
| next_1_4 | physical+token_history | 17 | 0.902 | 0.861 | 0.942 | 0.028 |
| next_5_8 | physical_action | 18 | 0.898 | 0.855 | 0.938 | 0.033 |
| next_5_8 | physical | 18 | 0.805 | 0.731 | 0.872 | 0.056 |
| next_5_8 | physical_action+aggregate_current | 18 | 0.888 | 0.833 | 0.930 | 0.034 |
| next_5_8 | physical+aggregate_current | 18 | 0.767 | 0.709 | 0.822 | 0.055 |
| next_5_8 | physical_action+token_current | 18 | 0.878 | 0.829 | 0.922 | 0.034 |
| next_5_8 | physical_action+token_history | 18 | 0.886 | 0.845 | 0.922 | 0.035 |
| next_5_8 | physical+token_history | 18 | 0.852 | 0.797 | 0.904 | 0.047 |
| next_9_12 | physical_action | 20 | 0.729 | 0.672 | 0.786 | 0.069 |
| next_9_12 | physical | 20 | 0.669 | 0.580 | 0.752 | 0.082 |
| next_9_12 | physical_action+aggregate_current | 20 | 0.776 | 0.731 | 0.820 | 0.069 |
| next_9_12 | physical+aggregate_current | 20 | 0.723 | 0.650 | 0.792 | 0.081 |
| next_9_12 | physical_action+token_current | 20 | 0.752 | 0.707 | 0.799 | 0.071 |
| next_9_12 | physical_action+token_history | 20 | 0.758 | 0.704 | 0.811 | 0.069 |
| next_9_12 | physical+token_history | 20 | 0.712 | 0.642 | 0.784 | 0.078 |

AUC 先在每个 `snapshot x q` 内计算，再按 snapshot 平均；因此初态、主干阶段、q 编号和剩余预算都不能直接产生排序。
`physical` 只读状态轨迹；`physical_action` 再加入当前 action chunk。MoE 模型每个外层训练折内每个特征族只选一个坐标。

## MoE 增量与 sibling 错配

| split | comparison | snapshots | gain | CI low | CI high |
| --- | --- | --- | --- | --- | --- |
| heldout_snapshot | total_moe_increment | 17 | -0.022 | -0.047 | 0.003 |
| heldout_snapshot | aligned_vs_shifted_token_history | 17 | 0.010 | -0.034 | 0.051 |
| heldout_worker | total_moe_increment | 17 | -0.058 | -0.096 | -0.024 |
| heldout_worker | aligned_vs_shifted_token_history | 17 | 0.000 | -0.047 | 0.044 |
| heldout_snapshot | total_moe_brier_gain | 22 | -0.002 | -0.004 | -0.001 |
| heldout_worker | total_moe_brier_gain | 22 | -0.002 | -0.005 | 0.000 |

`total_moe_increment` 是完整逐 token+历史 MoE 相对物理+动作控制的 AUC 增量；`aligned_vs_shifted_token_history` 把测试分支的 MoE 换成同 snapshot、同 q 的另一个候选。
Brier gain 与 AUC gain 单位不同，表中保留原始数值。

## 提前 9–12 query 的探索性信号

| split | comparison | snapshots | gain | CI low | CI high | 9-test low | 9-test high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heldout_snapshot | action_increment | 20 | 0.060 | -0.046 | 0.181 | -0.085 | 0.237 |
| heldout_snapshot | aggregate_moe_over_physical | 20 | 0.054 | 0.009 | 0.102 | -0.008 | 0.123 |
| heldout_snapshot | aggregate_moe_increment | 20 | 0.047 | 0.006 | 0.091 | -0.011 | 0.110 |
| heldout_snapshot | aligned_vs_shifted_aggregate_current | 20 | 0.106 | 0.066 | 0.155 | NA | NA |
| heldout_snapshot | aligned_vs_shifted_physical_aggregate_current | 20 | 0.108 | 0.057 | 0.160 | NA | NA |
| heldout_worker | action_increment | 20 | 0.020 | -0.069 | 0.119 | -0.102 | 0.158 |
| heldout_worker | aggregate_moe_over_physical | 20 | 0.068 | 0.015 | 0.125 | -0.004 | 0.153 |
| heldout_worker | aggregate_moe_increment | 20 | 0.049 | 0.006 | 0.105 | -0.005 | 0.134 |
| heldout_worker | aligned_vs_shifted_aggregate_current | 20 | 0.106 | 0.051 | 0.168 | NA | NA |
| heldout_worker | aligned_vs_shifted_physical_aggregate_current | 20 | 0.126 | 0.077 | 0.173 | NA | NA |

这里使用 action-token 均值后的当前路由；`9-test` 区间同时校正 3 个提前量 x 3 种 MoE 表示。`action_increment` 可判断路由信息是否只是当前动作计划的替代。该窗口不是主终点，必须按探索性结果解释。

按决策 q 拆开：

| q | mixed strata | mean gain | median gain |
| --- | --- | --- | --- |
| 8 | 10 | 0.102 | 0.104 |
| 12 | 12 | -0.020 | 0.000 |
| 16 | 4 | -0.083 | 0.000 |
| 20 | 4 | 0.012 | 0.019 |
| 24 | 6 | -0.004 | 0.000 |
| 28 | 8 | 0.008 | 0.018 |
| 32 | 7 | 0.034 | 0.000 |

该窗口反复选择的 token-mean 坐标：

| family | feature | folds | coef mean | positive folds |
| --- | --- | --- | --- | --- |
| entropy_centered | aggregate/entropy\|back_12_15\|d4\|token_mean | 5 | -0.382 | 0 |
| top1_switch_rate | aggregate/top1_switch_rate\|back_12_15\|d8\|token_mean | 4 | -0.184 | 0 |
| top1_centered | aggregate/top1\|back_12_15\|d8\|token_mean | 3 | 0.140 | 3 |
| top4_turnover | aggregate/top4_turnover\|front_2_5\|d9\|token_mean | 3 | -0.046 | 1 |
| adjacent_hellinger | aggregate/adjacent_hellinger\|back_12_15\|d2\|token_mean | 2 | 0.184 | 2 |
| adjacent_hellinger | aggregate/adjacent_hellinger\|front_2_5\|d9\|token_mean | 2 | -0.052 | 1 |
| top1_centered | aggregate/top1\|back_12_15\|d7\|token_mean | 2 | 0.186 | 2 |
| top4_turnover | aggregate/top4_turnover\|front_2_5\|d6\|token_mean | 2 | 0.255 | 2 |
| adjacent_hellinger | aggregate/adjacent_hellinger\|front_2_5\|d8\|token_mean | 1 | -0.114 | 0 |
| top1_switch_rate | aggregate/top1_switch_rate\|front_2_5\|d8\|token_mean | 1 | 0.168 | 1 |

四个 worker 的严格留出增量：

| worker | snapshots | physical_action_auc | joint_auc | auc_gain |
| --- | --- | --- | --- | --- |
| 0 | 6 | 0.702 | 0.712 | 0.010 |
| 1 | 5 | 0.697 | 0.710 | 0.013 |
| 2 | 5 | 0.628 | 0.659 | 0.031 |
| 3 | 4 | 0.652 | 0.829 | 0.176 |

4-worker 精确单侧 sign-flip p=0.0625；只有四个独立初态时，4/4 同方向的最小 p 就是 0.0625。
增量幅度主要由 worker3 提供，其余三个 worker 只有小幅正值；这也是候选信号不能升级为结论的原因。

远期候选按事件类型拆开：

| target | split | snapshots | gain | CI low | CI high |
| --- | --- | --- | --- | --- | --- |
| trap | heldout_snapshot | 20 | 0.047 | 0.006 | 0.091 |
| trap | heldout_worker | 20 | 0.049 | 0.006 | 0.105 |
| loop | heldout_snapshot | 20 | 0.056 | -0.005 | 0.127 |
| loop | heldout_worker | 20 | 0.028 | -0.010 | 0.064 |
| static | heldout_snapshot | 12 | -0.111 | -0.201 | -0.036 |
| static | heldout_worker | 12 | -0.028 | -0.089 | 0.025 |

正增量主要指向 loop 风险；连续静止的 snapshot 留出增量为负。因此不能把该候选解释成通用的停滞编码。

## 严格 worker 留出

| split | model | snapshots | AUC | CI low | CI high |
| --- | --- | --- | --- | --- | --- |
| heldout_snapshot | physical_action | 17 | 0.907 | 0.855 | 0.952 |
| heldout_snapshot | physical | 17 | 0.897 | 0.850 | 0.940 |
| heldout_snapshot | physical_action+token_history | 17 | 0.884 | 0.821 | 0.939 |
| heldout_snapshot | physical+token_history | 17 | 0.902 | 0.861 | 0.942 |
| heldout_worker | physical_action | 17 | 0.894 | 0.832 | 0.946 |
| heldout_worker | physical | 17 | 0.884 | 0.823 | 0.938 |
| heldout_worker | physical_action+token_history | 17 | 0.836 | 0.777 | 0.891 |
| heldout_worker | physical+token_history | 17 | 0.878 | 0.832 | 0.921 |

worker 对应四个独立初态。worker1 几乎都 trap、worker3 几乎都不 trap，所以该拆分是高域偏移敏感性，不应只看 pooled AUC。

## Loop 与静止分开

| target | model | snapshots | AUC | CI low | CI high |
| --- | --- | --- | --- | --- | --- |
| trap | physical_action | 17 | 0.907 | 0.855 | 0.952 |
| trap | physical_action+token_history | 17 | 0.884 | 0.821 | 0.939 |
| loop | physical_action | 17 | 0.916 | 0.850 | 0.970 |
| loop | physical_action+token_history | 17 | 0.910 | 0.846 | 0.962 |
| static | physical_action | 7 | 0.901 | 0.822 | 0.965 |
| static | physical_action+token_history | 7 | 0.889 | 0.832 | 0.950 |

Loop onset 使用冻结的非局部物理回返规则；静止 onset 是第一次能够因果确认连续 80 个 action step 低运动的 query。两者标签均不读取 MoE。

## 主模型反复选择的坐标

| family | feature | folds | coef mean | positive folds |
| --- | --- | --- | --- | --- |
| top1_centered | position/top1_centered\|front_2_5\|d9\|t8 | 5 | -0.039 | 1 |
| entropy_centered | position/entropy_centered\|front_2_5\|d9\|t8 | 4 | 0.035 | 2 |
| adjacent_hellinger | position/adjacent_hellinger\|back_12_15\|d9\|t7_to_t8 | 3 | 0.117 | 3 |
| nonlocal_return_hellinger | position/nonlocal_return_hellinger\|front_2_5\|d9\|t8 | 3 | 0.266 | 3 |
| query_hellinger | position/query_hellinger\|back_12_15\|d9\|t8 | 2 | 0.373 | 2 |
| query_hellinger | position/query_hellinger\|front_2_5\|d9\|t8 | 2 | 0.374 | 2 |
| top4_turnover | position/top4_turnover\|front_2_5\|d6\|t1_to_t2 | 2 | -0.304 | 0 |
| adjacent_hellinger | position/adjacent_hellinger\|back_12_15\|d7\|t7_to_t8 | 1 | 0.134 | 1 |
| adjacent_hellinger | position/adjacent_hellinger\|back_12_15\|d8\|t7_to_t8 | 1 | 0.092 | 1 |
| entropy_centered | position/entropy_centered\|back_12_15\|d8\|t8 | 1 | -0.106 | 0 |
| nonlocal_return_hellinger | position/nonlocal_return_hellinger\|front_2_5\|d0\|t5 | 1 | 0.363 | 1 |
| nonlocal_return_hellinger | position/nonlocal_return_hellinger\|front_2_5\|d6\|t10 | 1 | 0.453 | 1 |

特征若只在少数折出现，不能解释为稳定机制；系数正值表示该路由量更高时预测临近 trap。

## 设计边界

- 这是观察性预测，不证明 MoE 路由导致或因果识别了 trap。
- 物理标签来自 MuJoCo 状态与成功终态参考，不是人工语义标注。
- 只有一个 Long 任务、四个 worker；跨任务泛化仍需同协议的新采集。
- 当前 route cache 没有隐藏层；结论只涉及保存的 HB-MoE 概率。
