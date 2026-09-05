# 全 VLA_MUI_HUB 的 train-free 二元 Trap 报警复验

## 直接结论

**对于“这条轨迹是否陷入 Trap”的二元判断，双头 `max` 不如信息匹配的等权
mean 单标量。** 这不是排序结果，而是每个 held-out init 都用同任务其他 init 的
成功轨迹校准 q95 后，直接统计 `ALARM / NO ALARM`。

| detector | failure recall | success FPR | precision |
|---|---:|---:|---:|
| `lock_in` 单分量头 | 53.2% | 5.3% | 25.5% |
| `mean(instability, lock_in)` 单标量 | **66.5%** | **5.2%** | **30.4%** |
| `max(instability, lock_in)` 双头 | 57.9% | 5.5% | 26.4% |
| 四表型 `max` | 56.7% | 5.6% | 25.5% |

双头相对 `lock_in` 增加 `+4.7 pp` recall，按 37 个任务成簇 bootstrap 的
95% CI 为 `[+1.5, +11.4] pp`。但这只说明 instability 提供了新增信息。
与读取完全相同两个分数的 mean 单标量相比，双头反而降低 `-8.6 pp`，95% CI
为 `[-17.3, -0.7] pp`。因此不能把增益归因于双头结构。

若让两个头各自使用 q95 再直接 OR，recall 为 59.8%，但 FPR 同时升到 10.7%；
这属于增加报警预算，不是检测能力的公平提升。

## 对后续修正的意义

两个输出仍值得保留，因为它们指向不同的恢复分支，但不应直接用 `max` 作为总体
Trap alarm：

- stagnation 的主要表型是低 route mobility、高 lag recurrence 的稳定锁死；
  `lock_in` 在 q95 下检出 67.6%，成功误报 5.3%。
- gripper cycling 的主要表型是 route acceleration、late-flow volatility 和
  state jump 上升；`instability` 在 q95 下检出 37.0%，属于中等而非高召回信号。
- goal regression、subtask undo 和 regrasp/drop 有局部 state-action gap，
  但固定 feedback 头在成功校准 q95 下几乎不触发，尚不能用于恢复触发。
- 四头没有超过双头；当前不应继续靠堆头提高总体报警。

建议接口保留连续的 `(instability_score, lock_in_score)` 供恢复策略分流，同时用
等权 mean 作为当前较好的总体离线报警基线。恢复收益本轮没有实验，不能把“分到
某一表型”表述成“已经会修正”。

## 数据与边界

分析只纳入 `right-50x8-20260903` 中元数据和采样状态均为 complete 的 37 个任务：
14,800 条轨迹、487 条失败、14,313 条成功、221,781 个 query。3 个仍不完整任务和
正在写入的 `right-50x8b-20260903` 均排除。

物理类别是在读取 routing 前冻结的 query-boundary 运动学代理。当前缓存没有 RGB、
contact、force 或 chunk 内重新 forward，因此 regrasp/drop、contact jam、false grasp
和 stale chunk 不能被当作语义真值。所有数字是 episode 50%--90% 相位的离线结果，
还不是 onset 前在线报警性能。

## 产物

- 完整中文报告：[`results/hub_binary_audit/REPORT_ZH.md`](results/hub_binary_audit/REPORT_ZH.md)
- 二元指标：[`results/hub_binary_audit/binary_detector_summary.csv`](results/hub_binary_audit/binary_detector_summary.csv)
- 配对差值：[`results/hub_binary_audit/paired_detector_deltas.csv`](results/hub_binary_audit/paired_detector_deltas.csv)
- 互斥类型召回：[`results/hub_binary_audit/primary_head_alarm_rates.csv`](results/hub_binary_audit/primary_head_alarm_rates.csv)
- 代表性轨迹：[`results/hub_binary_audit/representative_candidates.csv`](results/hub_binary_audit/representative_candidates.csv)
- 可复现脚本：[`../../himoe-vla_trap/code/analyze_hub_phenotype_atlas.py`](../../himoe-vla_trap/code/analyze_hub_phenotype_atlas.py)

复现已有特征上的统计：

```bash
python himoe-vla_trap/code/analyze_hub_phenotype_atlas.py \
  --out-dir double-selete/trainfree/results/hub_binary_audit \
  --reuse
```

