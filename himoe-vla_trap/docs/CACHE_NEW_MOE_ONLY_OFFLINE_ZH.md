# cache_new：固定 v2 规则的严格 MoE-only 离线回放

## 核心结论

我把已经冻结的 `back_front_route_acceleration_v2` 原样应用到 `cache_new` 中同一任务的一次完整采集：400 条轨迹、17,460 次重规划。每条轨迹的第一个 query 用于建立跨 chunk 基线，因此产生 17,060 个可评分决策。预测阶段每个时刻的动态输入只有完整 HB MoE routing：

```text
[8 HB layers, 10 flow steps, 11 suffix tokens, 32 experts]
```

选择器不读取 action 数值、夹爪事件、EEF/物体位置、机器人或仿真状态、reward、success。episode id 只用于在轨迹边界清空 persistence 状态。所有逐 query 和逐 episode 预测先写盘，之后才加载物理状态和 outcome 做评价。

结果表明：固定的 persistence-2 规则在这批数据上是一个**低误报、低召回的通用失败风险告警**，但不是可靠的“未抓住”类型识别器。

| cache_new 子集 | failure 报警 | success 误报 |
|---|---:|---:|
| 全部 400 条 | 29/138 = 21.0% | 13/262 = 5.0% |
| 未出现在旧 B 的 272 个条件 | 20/89 = 22.5% | 5/183 = 2.7% |
| 与旧 B 名义同条件的 128 条 | 9/49 = 18.4% | 8/79 = 10.1% |

在全部 400 条上，正式报警与最终失败显著相关：odds ratio `5.10`，双侧 Fisher `p=1.61e-6`。在 272 个旧 B 未见条件上，odds ratio 为 `10.32`，`p=4.66e-7`。但是召回只有约 21%--22%，不能把“无报警”解释为“正常”。

## 是否只用了 MoE

需要区分运行时输入与整个方法开发过程：

- **运行时动态输入是 MoE-only**：当前 query 的 `hb_router_probs`。
- 固定参考是 5 条已知成功轨迹的 HB routing；选择这些参考时使用了 success metadata。
- 阈值由 5 条成功参考 leave-one-out 得到，没有失败标签、梯度优化或参数训练。
- v2 feature 的选择受到更早一批失败数据的 posthoc 分析启发。
- 本次物理量只用于预测完成后的失败分型和时序评价。

所以准确表述是：**train-free、runtime MoE-only、healthy-calibrated**；不能表述成“从方法设计到评价全程 label-free”。

## A/B 与在线序列一致性

同一个两阶段实现还完整回放了旧 A、旧 B 和 24 条 prospective 路由。A/B 是 retrospective stress test，不是新规则的独立验证；prospective 则用于检查离线实现是否忠实复现当时的在线决策。

| 语料 | episode | failure 报警 | success 误报 |
|---|---:|---:|---:|
| A | 352 | 183/235 = 77.9% | 18/117 = 15.4% |
| B | 512 | 34/216 = 15.7% | 8/296 = 2.7% |
| prospective | 24 | 8/14 = 57.1% | 1/10 = 10.0% |
| cache_new | 400 | 29/138 = 21.0% | 13/262 = 5.0% |

prospective 的 24 条轨迹中，离线重放的 raw 和 formal alarm query 序列与已保存在线记录 **24/24 完全相同**。这证明离线 evaluator 与在线选择器语义一致。A/B/C 的召回和误报差异很大，也说明单一健康阈值对不同 capture 分布并不稳定，不能把任一批结果当作普适性能。

## 未抓住代理分型

`cache_new` 的物理状态只在每次重规划边界保存一次，而每个 chunk 执行 10 个 action，因此这里只能建立 query-sampled kinematic proxy，不能声称是接触传感器真值。该代理找到：

- 53 条目标未抓住型失败；
- 85 条其他失败；
- 262 条成功；成功轨迹中没有出现该代理事件。

| 类型 | 正式报警 anywhere | 漏抓前 5 query 或当下正式报警 | 任意单点 raw | 漏抓前 5 query 或当下 raw |
|---|---:|---:|---:|---:|
| 目标未抓住代理，n=53 | 13/53 = 24.5% | 9/53 = 17.0% | 52/53 = 98.1% | 40/53 = 75.5% |
| 其他失败，n=85 | 16/85 = 18.8% | - | 56/85 = 65.9% | - |
| 成功，n=262 | 13/262 = 5.0% | - | 197/262 = 75.2% | - |

这里最重要的是：单点 raw 在成功轨迹中也有 `75.2%`，因此 raw 不能直接作为报警；persistence-2 虽将 success false alarm 压到 `5.0%`，也同时丢掉了大部分目标事件。

固定 formal alarm 对目标型和其他失败的区分不显著：odds ratio `1.40`，Fisher `p=0.520`。连续 `score_max` 对目标型与其他失败的 AUC 为 `0.741`，说明分数强度包含一定类型排序信息，但当前冻结阈值和 persistence 没有把它转化成可靠的类型判定。不能在本批上重新调阈值后再把同一批称为验证集。

## 与 24 条 prospective 结果的关系

此前 24 条 prospective 数据上，v2 命中严格目标 `5/7`、全部失败 `8/14`，成功误报 `1/10`。本次更大离线集的目标正式报警只有 `13/53`，及时报警只有 `9/53`。因此 `5/7` 的高召回没有在 `cache_new` 上维持，论文或汇报必须同时呈现这项修正。

这并不说明 MoE 完全无效：在 400 条上正式报警明显富集 failure，且旧 B 未见条件上 success false alarm 只有 `2.7%`。它说明目前规则更适合作为高置信风险支路，而不是单独承担漏抓检测。

## 数据独立性与代理检查

本批与旧 B 有 128 个名义相同的 `(init_state, flow_noise_seed)` 条件，另有 272 个条件未出现于旧 B。逐条件比较得到：

- 完整 route 精确重复：0；
- 首 query route 精确重复：0；
- 同长度：76/128；
- 同 outcome：107/128。

所以它们是新 capture，不是旧数组拷贝；但为保守起见，主表仍单独报告 272 个未见条件。

query-sampled 漏抓代理还在已有 24 条稠密 prospective 轨迹上做了检查：相对严格稠密标签为 TP 7、FN 0、TN 15、额外 2 条；两条额外样本恰好都是稠密定义中 15 mm 阈值的 borderline failure，没有成功样本被加入。它可用于大规模事后分型，但仍不是逐 action 接触真值。

## 可复现文件

- [严格两阶段 MoE-only 回放代码](../code/evaluate_moe_dynamics_large_offline.py)
- [A/B/prospective 回放摘要](../results/moe_dynamics_online_alarm/large_offline_replay/summary.json)
- [cache_new 事后漏抓审计代码](../code/audit_cache_new_moe_dynamics.py)
- [回放摘要](../results/moe_dynamics_online_alarm/cache_new_task8_replay/summary.json)
- [事后分型摘要](../results/moe_dynamics_online_alarm/cache_new_task8_replay/posthoc_summary.json)
- [17,060 条可评分逐 query 预测](../results/moe_dynamics_online_alarm/cache_new_task8_replay/query_predictions_moe_only.csv)
- [400 条逐 episode MoE-only 预测](../results/moe_dynamics_online_alarm/cache_new_task8_replay/episode_predictions_moe_only.csv)
- [400 条事后标签与评价](../results/moe_dynamics_online_alarm/cache_new_task8_replay/posthoc_missed_grasp_audit.csv)

本轮只纳入状态为 `complete` 的 `right-50x8-20260903`。审计时仍标记为 `running` 的 `right-50x8b-20260903` 和 `right-50x16c-20260904` 没有混入正式数字；它们应留作下一版冻结规则的真正新测试。
