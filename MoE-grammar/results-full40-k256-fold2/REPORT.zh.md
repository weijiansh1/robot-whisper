# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 36.8606 |
| history1 | 36.6169 |
| bag | 36.6352 |
| history | 36.6337 |

二阶有序历史相对 task+position 的差为 -0.2268 bits/query (state-blocked 95% CI [-0.23427324994704027, -0.22043124340649406])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0015 bits/query (95% CI [-0.0019522786253794466, -0.0009482614125315847])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.592 | [0.543, 0.646] | 0.077 | 0.229 |
| phase.page0.5 | 0.605 | [0.549, 0.681] | 0.087 | 0.240 |
| history.point | 0.603 | [0.545, 0.683] | 0.058 | 0.080 |
| history.window4 | 0.579 | [0.548, 0.614] | 0.079 | 0.206 |
| history.page0.5 | 0.608 | [0.548, 0.685] | 0.087 | 0.246 |
| history.glr | 0.598 | [0.538, 0.668] | 0.083 | 0.234 |
| residual.page0.5 | 0.555 | [0.500, 0.625] | 0.075 | 0.063 |
| calibration_selected | 0.572 | [0.512, 0.642] | 0.072 | 0.074 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.600 | [0.536, 0.689] | 0.072 | 0.314 |
| phase.page0.5 | 0.602 | [0.528, 0.685] | 0.067 | 0.394 |
| history.point | 0.596 | [0.550, 0.664] | 0.040 | 0.234 |
| history.window4 | 0.650 | [0.586, 0.712] | 0.068 | 0.343 |
| history.page0.5 | 0.608 | [0.532, 0.691] | 0.069 | 0.383 |
| history.glr | 0.608 | [0.536, 0.681] | 0.061 | 0.389 |
| residual.page0.5 | 0.609 | [0.545, 0.676] | 0.071 | 0.166 |
| calibration_selected | 0.637 | [0.550, 0.707] | 0.070 | 0.200 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
