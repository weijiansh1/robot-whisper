# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 39.3397 |
| history1 | 39.1472 |
| bag | 39.1522 |
| history | 39.1480 |

二阶有序历史相对 task+position 的差为 -0.1918 bits/query (state-blocked 95% CI [-0.20012968255390942, -0.1829696785336067])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0042 bits/query (95% CI [-0.0060075705660049975, -0.002724231065919953])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.581 | [0.514, 0.648] | 0.073 | 0.156 |
| phase.page0.5 | 0.625 | [0.566, 0.696] | 0.077 | 0.141 |
| history.point | 0.611 | [0.547, 0.694] | 0.059 | 0.156 |
| history.window4 | 0.582 | [0.547, 0.630] | 0.083 | 0.099 |
| history.page0.5 | 0.633 | [0.576, 0.693] | 0.076 | 0.145 |
| history.glr | 0.620 | [0.566, 0.684] | 0.080 | 0.160 |
| residual.page0.5 | 0.546 | [0.504, 0.588] | 0.073 | 0.095 |
| calibration_selected | 0.607 | [0.548, 0.677] | 0.058 | 0.164 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.614 | [0.461, 0.732] | 0.084 | 0.439 |
| phase.page0.5 | 0.623 | [0.497, 0.734] | 0.088 | 0.374 |
| history.point | 0.553 | [0.419, 0.675] | 0.046 | 0.244 |
| history.window4 | 0.597 | [0.484, 0.714] | 0.098 | 0.290 |
| history.page0.5 | 0.630 | [0.502, 0.740] | 0.086 | 0.378 |
| history.glr | 0.621 | [0.491, 0.727] | 0.082 | 0.385 |
| residual.page0.5 | 0.590 | [0.537, 0.644] | 0.080 | 0.134 |
| calibration_selected | 0.554 | [0.435, 0.673] | 0.045 | 0.229 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
