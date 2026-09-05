# cache_new MoE 健康不变量报警

## 目的

这项实验不使用旧的跨语料概率表，也不把异常分数解释成失败概率。它只回答：

> 对一条逐步到来的随机轨迹，只看当前及历史 MoE 路由，固定的 train-free 判断器能否在 1% 成功轨迹误报预算下识别最终失败，并超过固定时间钟？

## 固定设计

数据只来自 `VLA_MUI_HUB/cache_new/HiMoE-VLA/*/*/right-50x8-20260903`，共 40 个任务、16,000 条轨迹和 253,722 次推理。

每条轨迹使用 q0--q3 建立自身基线，从 q4 开始计算三个机制头：

- `flow convergence failure`：完整去噪路由路径中 late-flow Hellinger 路径占比相对早期基线上升；
- `state-action response split`：相邻重规划间 front-HB state-token 路由跳变大于 action-token 跳变；
- `cross-chunk recurrence`：final-flow back-HB action routing 对 lag 1--4 历史的最大 weighted Jaccard 相对早期基线上升。

收敛和响应必须在同一 query 同时极端；复返必须连续两个 query 极端。两条分支取 OR，不学习权重。

任务在不读取结果时按 suite 内 SHA256 排名固定划分。32 个任务只用成功轨迹确认阈值，使整条成功轨迹的报警率不超过 1%；8 个任务完全留出。留出逐 query 预测先写盘并记录 SHA256，之后才读取最终 success/failure。运行时不读取任务 ID、动作、物理状态、reward、episode 长度或未来路由。

## 严格结果

**未通过。** 留出集有 3,200 条轨迹、114 个失败：

| 方法 | TP | FP | Precision | Failure recall | Success FPR |
|---|---:|---:|---:|---:|---:|
| MoE 健康不变量 | 51 | 32 | 61.45% | 44.74% | 1.04% |
| 预固定 q39 时间钟 | 47 | 3 | 94.00% | 41.23% | 0.10% |
| 事后等误报 q29 时间钟 | 68 | 27 | 71.58% | 59.65% | 0.87% |

等误报时间钟只在揭盲后选择，用作诊断压力测试，不属于预固定 comparator。它在更低误报下同时超过 MoE 的 precision 和 recall，因此当前判断器没有证明时长之外的独立检测价值。

83 条留出报警全部来自持续复返分支；收敛+响应分支为 0。32 个成功误报全部集中在一个任务，说明 recurrence 阈值仍对新任务的正常路由节奏敏感。

留出集中已有 71 个 query-boundary 物理 onset 代理。MoE 只在 5 个事件中于 `[-2,0]` query 首次报警，另有 27 个报警发生在 onset 之后。因此它不是 onset-localized early detector。

## 全量描述结果

留出预测冻结后，再揭开其余任务中从未用于阈值的失败标签。全部 532 个失败中检出 152 个：failure recall 28.57%、precision 49.51%、成功误报率 1.00%。固定 q39 时间钟检出 267 个，recall 50.19%、precision 71.77%、成功误报率 0.68%。

全量数字覆盖全部失败，但其中成功轨迹参与过阈值确认，所以严格主结论仍以 8 个留出任务为准。

## 产物

- 配置：[moe_invariant_alarm_cache_new.json](../configs/moe_invariant_alarm_cache_new.json)
- 实现：[evaluate_moe_invariant_alarm_cache_new.py](../code/evaluate_moe_invariant_alarm_cache_new.py)
- 专用校验：[validate_moe_invariant_alarm_cache_new.py](../code/validate_moe_invariant_alarm_cache_new.py)
- 自动报告：[REPORT_ZH.md](../results/moe_invariant_alarm_cache_new/REPORT_ZH.md)
- 机器摘要：[summary.json](../results/moe_invariant_alarm_cache_new/summary.json)
- 揭盲前预测：[heldout_predictions_label_free.csv.gz](../results/moe_invariant_alarm_cache_new/heldout_predictions_label_free.csv.gz)
- 预测清单：[prediction_manifest.json](../results/moe_invariant_alarm_cache_new/prediction_manifest.json)
- 留出翻牌：[heldout_episode_flip.csv](../results/moe_invariant_alarm_cache_new/tables/heldout_episode_flip.csv)
- 全量描述翻牌：[all_episode_flip_descriptive.csv](../results/moe_invariant_alarm_cache_new/tables/all_episode_flip_descriptive.csv)

复现：

```bash
python himoe-vla_trap/code/evaluate_moe_invariant_alarm_cache_new.py --workers 4
python himoe-vla_trap/code/validate_moe_invariant_alarm_cache_new.py
```
