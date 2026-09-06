# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 39.3483 |
| history1 | 39.1535 |
| bag | 39.1580 |
| history | 39.1529 |

二阶有序历史相对 task+position 的差为 -0.1955 bits/query (state-blocked 95% CI [-0.20723283812203416, -0.18364953650085855])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0052 bits/query (95% CI [-0.006848627586129371, -0.0037389385314497195])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.562 | [0.502, 0.624] | 0.060 | 0.138 |
| phase.page0.5 | 0.566 | [0.515, 0.639] | 0.062 | 0.196 |
| history.point | 0.548 | [0.491, 0.615] | 0.031 | 0.050 |
| history.window4 | 0.582 | [0.555, 0.618] | 0.056 | 0.117 |
| history.page0.5 | 0.568 | [0.517, 0.629] | 0.062 | 0.188 |
| history.glr | 0.573 | [0.518, 0.625] | 0.057 | 0.146 |
| residual.page0.5 | 0.560 | [0.499, 0.618] | 0.073 | 0.100 |
| calibration_selected | 0.579 | [0.532, 0.628] | 0.063 | 0.183 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.636 | [0.566, 0.714] | 0.054 | 0.317 |
| phase.page0.5 | 0.682 | [0.597, 0.759] | 0.059 | 0.408 |
| history.point | 0.650 | [0.586, 0.705] | 0.022 | 0.142 |
| history.window4 | 0.680 | [0.645, 0.725] | 0.051 | 0.279 |
| history.page0.5 | 0.681 | [0.607, 0.753] | 0.058 | 0.400 |
| history.glr | 0.695 | [0.609, 0.772] | 0.058 | 0.375 |
| residual.page0.5 | 0.610 | [0.547, 0.669] | 0.064 | 0.113 |
| calibration_selected | 0.691 | [0.589, 0.773] | 0.062 | 0.400 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
