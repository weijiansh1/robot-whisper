# Train-free 单 rollout 高精度在线报警实验

## 结论

这次改进对“成功尽量不报、失败尽量多报”的目标是明显正面的，但它解决的是
**Trap 在线确认**，还不是 Trap 发生前的预警。

推荐把同一个单次可疑阈值用于三级状态机：

```text
1 个 query 越阈值：suspect，仅记录
连续 2 个 query：warning
连续 4 个 query：hard alarm，并锁存
```

在新的 `right-50x8b-20260903` 批次上，共评估 39 个任务、15,600 条
rollout，其中 564 条失败、15,036 条成功：

| 在线输出 | TP / 564 | FP / 15,036 | 失败召回 | 成功误报率 | 报警中失败占比 |
|---|---:|---:|---:|---:|---:|
| `warning`, K2 q95 | 252 | 104 | 44.68% | 0.692% | 70.79% |
| **`hard alarm`, K4 q95** | **189** | **29** | **33.51%** | **0.193%** | **86.70%** |
| `hard alarm`, K4 q97.5 | 125 | 10 | 22.16% | 0.067% | 92.59% |

因此，每 100 次 K4 q95 报警约有 87 次最终是失败；代价是约三分之二的失败不会被
这个高精度头捕获。K2 可以作为较灵敏的 warning，K4 作为真正触发恢复的硬报警。

## 在线信号

每次 VLA forward 只读取当前和历史的 `router_prob` 与 `expert_ids`。在 HB-MoE
第 12--15 层、最终去噪步、action token 1--10 上计算相邻 query 路由分布的
Hellinger 距离：

```text
m(q) = mean Hellinger(route(q), route(q-1))
u4(q) = mean(m(q-3:q))
score(q) = -u4(q)
```

所以这里确实结合了“瞬时”和“变化”：`m(q)` 是路由随 query 的一阶变化量，
`u4(q)` 是它的短窗趋势，K2/K4 再要求异常持续。当前最有判别力的方向不是一次
很高的导数峰，而是**路由变化率连续偏低**，即模型进入稳定计算锁死。

阈值按任务从旧批次的 400 条轨迹中无标签校准：先取每条参考轨迹的最大 causal
score，再取这些 episode maxima 的经验 q95 或 q97.5。这样约束的是“整条成功
rollout 是否曾误报”，而不是单 query 的越界概率。测试 rollout 从不参与阈值校准。

## 为什么共享阈值有效

预先封存的版本分别给 K1、K2、K4 校准阈值。连续条件越严格，历史 score 分布也越
低，重新取分位数会同时把阈值放宽，抵消了持续确认的收益。新规则只校准一次 K1
阈值，并让 K2/K4 复用它：

```text
K2 = 同一阈值连续越过 2 次
K4 = 同一阈值连续越过 4 次
```

因此 K4 必然是 K2 的子集，成功轨迹中的偶发低变化会被过滤，而 Trap 中持续的
路由锁死会保留。这不是训练出的双头；没有分类器、权重拟合或失败标签参与运行时
计算。

## 两批数据的一致性

共享阈值规则在旧批次与新批次上的数值很接近：

| cohort | 输出 | TP | FP | 召回 | 成功 FPR | Precision |
|---|---|---:|---:|---:|---:|---:|
| 旧批次，14,800 条 | K2 q95 | 234 | 88 | 48.05% | 0.615% | 72.67% |
| 新批次，15,600 条 | K2 q95 | 252 | 104 | 44.68% | 0.692% | 70.79% |
| 旧批次，14,800 条 | K4 q95 | 177 | 21 | 36.34% | 0.147% | 89.39% |
| 新批次，15,600 条 | K4 q95 | 189 | 29 | 33.51% | 0.193% | 86.70% |

新批次按任务 bootstrap 的 95% 区间为：

| 输出 | 失败召回 | 成功 FPR | Precision |
|---|---:|---:|---:|
| K2 q95 | 31.52%--67.19% | 0.403%--1.035% | 56.75%--82.92% |
| K4 q95 | 21.17%--55.67% | 0.079%--0.345% | 75.83%--94.38% |

不过任务差异仍然明显。新批次 K4 的 29 个成功误报只分布在 12 个任务中，其中
`KITCHEN_SCENE6...microwave...` 单个任务贡献 9 个 FP、1 个 TP。总体 Precision
不能替代逐任务检查。

## 与预先封存实验的区别

在查看新批次 outcome 之前冻结的主结果是 `mobility_w4_k4@q99`：

```text
TP = 108 / 564       recall = 19.15%
FP = 78 / 15,036     success FPR = 0.519%
precision = 58.06%
```

它满足预定的 FPR 不高于 1%，但因 K4 被独立重新校准，反而不如简单 K1 q99 的
67.11% precision。共享阈值修正是在查看该结果后提出的，所以即使它在旧、新两批上
都稳定，**仍属于 post-hoc 探索结果**，不能把新批次称为它的独立验证集。要形成
正式结论，需要在第三批未查看标签的 flow seeds 上原样冻结复验。

## 是否是提前预警

不是。旧批次有 325 条失败具备可用的物理 onset proxy。K4 q95 在其中：

```text
任意时刻报警                 29.85%
onset 之前报警                4.00%
严格 onset 前 [-4,-1] 命中     1.54%
首次报警相对 onset 的中位数      +8 query
```

K2 q95 的首次报警中位数也是 `+7 query`。因此该信号主要确认“已经锁进 Trap”，
可以为后续恢复策略提供触发点，但目前不能支持“提前避免 Trap”的表述。

## 单 rollout 使用

```python
from precision_cascade_monitor import TaskProfile
from shared_threshold_cascade_monitor import SharedThresholdCascadeMonitor

profile = TaskProfile.load(profile_path, task_name)
monitor = SharedThresholdCascadeMonitor(profile, quantile=0.95)

for router_prob, expert_ids in live_rollout:
    event = monitor.update(router_prob, expert_ids)
    # event["state"]: normal / suspect / warning / alarm
    # event["phenotype"]: loop / static / feedback / untyped，仅作诊断
```

状态机不知道轨迹最终成功还是失败，也不等待 episode 结束。只有实验评估阶段，才在
rollout 结束后把锁存报警与最终 outcome 交叉统计 TP、FP、FN 和 TN。

实现位于 `shared_threshold_cascade_monitor.py`，复现实验位于
`evaluate_shared_threshold_cascade.py`，逐任务和总体结果位于
`results/online_precision_cascade_shared/`。

## 验证

- 真实 `routes.zarr` 单轨迹逐 query 回放与离线首次 K2/K4 报警完全一致；
- 新批次 15,600 条 rollout 全部进入最终计数；
- 封存 scorer、profile、协议及结果的哈希验证通过；
- 新增与原封存实验相关测试共 9 项通过，`trainfree` 全部 39 项测试通过；
- 本次新增监控器、评估器和测试的静态检查及 Python 编译检查通过。
