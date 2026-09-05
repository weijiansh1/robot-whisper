# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success bits/query |
|---|---:|
| phase | 35.6075 |
| history1 | 35.4021 |
| bag | 35.4006 |
| history | 35.3973 |

二阶有序历史相对 task+position 的差为 -0.2102 bits/query (state-blocked 95% CI [-0.21701448846389576, -0.20330057521064862])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0033 bits/query (95% CI [-0.00398246275330244, -0.002569212554846622])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.570 | [0.525, 0.607] | 0.078 | 0.229 |
| phase.page0.5 | 0.583 | [0.540, 0.624] | 0.078 | 0.217 |
| history.point | 0.612 | [0.555, 0.675] | 0.056 | 0.211 |
| history.window4 | 0.551 | [0.500, 0.612] | 0.044 | 0.114 |
| history.page0.5 | 0.598 | [0.570, 0.635] | 0.080 | 0.229 |
| history.glr | 0.595 | [0.543, 0.647] | 0.075 | 0.171 |
| residual.page0.5 | 0.568 | [0.533, 0.599] | 0.079 | 0.211 |
| calibration_selected | 0.599 | [0.562, 0.657] | 0.083 | 0.194 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.581 | [0.529, 0.667] | 0.067 | 0.343 |
| phase.page0.5 | 0.608 | [0.564, 0.645] | 0.076 | 0.360 |
| history.point | 0.621 | [0.566, 0.684] | 0.054 | 0.234 |
| history.window4 | 0.626 | [0.554, 0.693] | 0.040 | 0.366 |
| history.page0.5 | 0.621 | [0.589, 0.659] | 0.077 | 0.366 |
| history.glr | 0.631 | [0.593, 0.674] | 0.074 | 0.371 |
| residual.page0.5 | 0.581 | [0.497, 0.653] | 0.083 | 0.269 |
| calibration_selected | 0.624 | [0.580, 0.663] | 0.079 | 0.371 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
