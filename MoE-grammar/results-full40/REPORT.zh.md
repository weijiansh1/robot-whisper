# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success bits/query |
|---|---:|
| phase | 35.7666 |
| history1 | 35.5699 |
| bag | 35.5664 |
| history | 35.5626 |

二阶有序历史相对 task+position 的差为 -0.2040 bits/query (state-blocked 95% CI [-0.2110801081132363, -0.19630399485177827])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0038 bits/query (95% CI [-0.004813503939519924, -0.0028110811480019167])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.609 | [0.556, 0.674] | 0.073 | 0.137 |
| phase.page0.5 | 0.646 | [0.576, 0.725] | 0.080 | 0.147 |
| history.point | 0.632 | [0.557, 0.718] | 0.049 | 0.096 |
| history.window4 | 0.587 | [0.524, 0.662] | 0.060 | 0.086 |
| history.page0.5 | 0.652 | [0.592, 0.732] | 0.080 | 0.147 |
| history.glr | 0.642 | [0.578, 0.720] | 0.076 | 0.142 |
| residual.page0.5 | 0.569 | [0.520, 0.631] | 0.062 | 0.102 |
| calibration_selected | 0.645 | [0.578, 0.716] | 0.080 | 0.142 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.579 | [0.506, 0.637] | 0.091 | 0.340 |
| phase.page0.5 | 0.599 | [0.484, 0.691] | 0.101 | 0.437 |
| history.point | 0.574 | [0.511, 0.651] | 0.028 | 0.173 |
| history.window4 | 0.632 | [0.544, 0.705] | 0.094 | 0.315 |
| history.page0.5 | 0.611 | [0.496, 0.701] | 0.101 | 0.442 |
| history.glr | 0.617 | [0.507, 0.708] | 0.088 | 0.401 |
| residual.page0.5 | 0.599 | [0.548, 0.645] | 0.077 | 0.208 |
| calibration_selected | 0.608 | [0.507, 0.689] | 0.109 | 0.421 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
