# Train-free 风险与可恢复性判断器实验报告

## 结论

这个两级判断器有用，但两个头解决的问题不同：

- K2/K4 MoE 路由头适合在 rollout 中持续检测“不能按时完成”的风险；
- 预算边界的活动性头可以优先选择值得多给控制步的轨迹，但单独使用时不是通用的可恢复分类器；
- 若目标是尽量不牺牲晚成功，推荐 `K4 hard alarm OR recovery_score >= q75` 的保守策略。

保守策略在两套数据上结果一致：

| 数据集 | 风险轨迹 | 给予延长 | 救回晚成功 | 晚成功召回 | 直接干预组中持续失败占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| development_main | 487 | 282 | 28/30 | 93.33% | 203/205 = 99.02% |
| external_8b | 564 | 329 | 38/41 | 92.68% | 232/235 = 98.72% |

这意味着：两个头都不支持继续等待时，轨迹几乎一定是持续失败；只要路由曾经硬报警，或者到预算边界仍保留明显运动/路由可塑性，就先给最多 10 个额外 VLA queries。

## 在线结构

### 第一头：deadline risk

沿用共享 q95 阈值的 route-mobility 状态机：

```text
连续 2 个 query 越界 -> warning
连续 4 个 query 越界 -> hard alarm
```

它只读取当前及历史 MoE routing。晚成功和持续失败都属于风险正类，只有按时成功属于负类。

| 数据集 | 输出 | 风险召回 | 按时成功 FPR | 风险 precision | 晚成功召回 |
| --- | --- | ---: | ---: | ---: | ---: |
| main | K2 warning | 48.05% | 0.615% | 72.67% | 46.67% |
| main | K4 hard | 36.34% | 0.147% | 89.39% | 36.67% |
| external | K2 warning | 44.68% | 0.692% | 70.79% | 46.34% |
| external | K4 hard | 33.51% | 0.193% | 86.70% | 46.34% |

因此 K4 仍然是高精度风险报警，而不是只针对不可恢复失败的报警。

### 第二头：recoverability evidence

在最后一个原始 query 发出动作前，使用单 rollout 当时已经可得的六个量：

1. 最近四次 EEF 平移均值；
2. 最近一次 EEF 平移；
3. 实际 EEF 运动与历史动作命令幅度之比；
4. 当前 route mobility；
5. 当前 state/action feedback split；
6. 负的 lag recurrence。

每个量都只与同任务历史按时完成轨迹在终态附近的经验分布比较。六个经验百分位等权平均：

```text
recovery_score = mean(six empirical midranks)
```

没有分类器、损失函数或拟合权重。main 使用 leave-one-initial-state cross-fitting；external 的 profile 和 q75/q90 阈值完全来自旧批次。

## 恢复分数结果

| 数据集 | ROC-AUC | Average precision | 晚成功基率 |
| --- | ---: | ---: | ---: |
| main | 0.726 | 0.207 | 6.16% |
| external | 0.704 | 0.152 | 7.27% |

用历史超时轨迹 recovery score 的 q75 作为无标签预算阈值：

| 数据集 | 延长数量 | 救回 | 延长命中率 | 相对随机 lift | 晚成功召回 | 每救回一条的新增 query 成本 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 121/487 | 20/30 | 16.53% | 2.68x | 66.67% | 55.60 |
| external | 156/564 | 26/41 | 16.67% | 2.29x | 63.41% | 55.85 |

若所有风险轨迹都延长，成本分别为 157.70 和 133.29 queries/救回一条。因此 q75 经济策略把单位救回成本降低约 65% 和 58%。

## 两种部署策略

### 经济策略

```text
at H-1:
    recovery_score >= q75 -> extend
    otherwise             -> intervene
```

它只延长约四分之一的风险轨迹，救回约三分之二晚成功。直接干预组中的持续失败占 main 97.27%、external 96.32%。适合额外 query 昂贵的情况。

### 保守策略

```text
during rollout:
    latch K4 hard alarm

at H-1:
    hard_alarm OR recovery_score >= q75 -> extend first
    otherwise                            -> intervene first
```

它救回 main 28/30、external 38/41 条晚成功。与全部延长相比，额外 query 总成本从 main 4731 降到 2686，从 external 5465 降到 3127，同时只漏掉 2 和 3 条晚成功。

这个 OR 有机制含义：

- q75 活跃头捕获仍在推进的 slow-active rollout，主要对 long 有效；
- K4 捕获曾进入稳定路由锁死、但可能通过后续重新 query 解锁的 recoverable Trap，补回 spatial/object 晚成功。

## 不能忽略的负面结果

恢复分数不是跨 suite 的统一成功概率：

| Suite | main AUC | external AUC |
| --- | ---: | ---: |
| long | 0.785 | 0.785 |
| object | 0.538 | 0.401 |
| spatial | 0.418 | 0.371 |

总体 AUC 主要来自 long。spatial 中的晚成功往往已经路由锁死，单看“仍在活动”反而会漏掉，必须由 K4 头补充。

q90 也没有比 q75 更可靠：main 的延长命中率从 16.53% 降到 12.24%，external 基本不变（16.67% 到 16.07%）。因此 `recovery_score` 可用于排序预算优先级，但不能解释成校准后的恢复概率，“strong_extend”这个名称没有得到数据支持。

外部 `K4 AND q75` 小组达到 7/16 = 43.75% 晚成功，但 main 只有 3/16 = 18.75%，不具备跨批次稳定性，不应单独作为正式规则。

## 有效性边界

- 本轮提出方法前已经看过两套 outcome，因此属于 post-hoc 探索；external 的数值是参考批次外评估，但不是完全未见标签的独立方法验证。
- 恢复决策发生在原预算最后一个 query，而第一头才是提前风险报警。
- 没有真正施加纠正干预；策略成本根据已完成的无干预 `+10 queries` continuation 计算。
- 晚成功仍受十 query 右截断限制。
- 判断器没有 RGB 目标进度、接触或物体状态，所以目前不能可靠区分所有 spatial/object 可恢复失败。

## 产物

- 协议：`RISK_RECOVERY_JUDGE_PROTOCOL.md`
- 在线恢复特征与经验 profile：`risk_recovery_monitor.py`
- 复现实验：`evaluate_risk_recovery_judge.py`
- 30,400 条逐轨迹决策：`results/online_risk_recovery_judge/episode_decisions.csv`
- 风险头结果：`results/online_risk_recovery_judge/risk_metrics.csv`
- 恢复头结果：`results/online_risk_recovery_judge/recovery_metrics.csv`
- 两头组合策略：`results/online_risk_recovery_judge/joint_policy_metrics.csv`
- suite/task 分解：`results/online_risk_recovery_judge/recovery_by_suite.csv`、`recovery_by_task.csv`
- 校验哈希与机器可读总结：`results/online_risk_recovery_judge/summary.json`
