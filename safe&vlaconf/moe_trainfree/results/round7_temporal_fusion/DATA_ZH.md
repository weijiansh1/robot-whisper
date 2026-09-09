# 数据索引

## 人群与编号

- 原始语料 32,000 条，A/B 各 16,000；沿用之前 12 个划分。
- `episode_decisions.csv` 每行是一个方法、一个折、一次测试轨迹评估。
- 每方法 21,120 次测试评估；未见任务 14,400 次，其中失败 491、成功 13,909。
- 跨折重复轨迹存在，`global_row` 是原始语料的唯一行身份，
  `(fold, global_row, method)` 才是某次具体预测的键。
- `first_alarm_query=-1` 表示没有报警，其他值从零开始。
- query q 对应第 q+1 次推理、已完成 10q 个任务动作步，不含初始化静置步。
- 12 折平均指标与合并评估次数的指标不同，不能混用分母。

## 第一轮

- `episode_decisions.csv`: 十个方法在组校准 5% 下的逐轨迹首次报警。
- `predictions/*.npz`: 全 alpha 网格、两种校准、逐 query 测试和校准分数。
- `profiles/*.npz`: 完整参考点、缩放参数、阈值和 checkpoint，本地保留。
- `alarm_bins.csv`: 区间分布，含 `no_alarm` 类别。
- `query_distribution.csv`: 每个 query 的新增报警、累计报警、仍在运行且
  未报警的风险集、风险集内新增报警率、已结束未报警数。
- `relative_alarm_bins.csv`: 相对推理进度的分布，仅用于评价。
- `fold_macro_summary.csv`, `suite_summary.csv`: 折平均、套件结果。
- `paired_alarm_changes.csv`: 新增、减少、提前、推迟的成对变化。
- `paired_task_bootstrap.csv`: 任务聚类的区间，重复轨迹随任务一起重采样。
- `physical_timing.csv`, `physical_summary.csv`: 物理事件前、同 query、之后、漏报。
- `historical_*`: 原来冻结的 v7/v8/v8.2/v8.3 对照，与重新校准的 guard 区分。
- `v8_padding_audit.csv`: 旧 v8 对真实 query 与补齐区的阈值统计审计。
- `verification.json`: 校准秩、直接距离、原始输入和在线回放核验。

## 跨任务校准补充

`cross_task_calibration/` 下：

- `thresholds.csv`, `bank_exclusions.csv`: 旧/新阈值、每个 A 校准任务排除的参考点。
- `episode_decisions.csv`: 原 kNN、原 12D、新校准 10D/12D 的全部评估。
- `matched_episode_decisions.csv`: 与历史 v8 覆盖范围对齐的 kNN、v8 和 OR 组合。
- `pooled_matched_counts.csv`: 相同的 14,000 次未见任务评估，491 失败、13,509 成功。
- `paired_task_bootstrap.csv`: 该相同人群中的成对差值区间。
- `profiles/*.npz`: 可供在线接口读取的新阈值 profile，本地保留。
- `verification.json`: 独立校准检查和真实路由输入上的组合监控回放。

第一轮候选在评分前固定。跨任务校准是观察第一轮结果后提出的第二轮探索，
两组协议与封存记录分开保存。数据包不包含大型原始路由、输入缓存和参考
profile；这些都保留在本地，复现说明见代码目录的 README。
