# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 37.2737 |
| history1 | 37.0341 |
| bag | 37.0536 |
| history | 37.0519 |

二阶有序历史相对 task+position 的差为 -0.2217 bits/query (state-blocked 95% CI [-0.23290001360109783, -0.21153667439336843])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0017 bits/query (95% CI [-0.002072591054776768, -0.0012224799542679616])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.643 | [0.579, 0.719] | 0.077 | 0.107 |
| phase.page0.5 | 0.643 | [0.580, 0.711] | 0.081 | 0.137 |
| history.point | 0.617 | [0.577, 0.672] | 0.040 | 0.086 |
| history.window4 | 0.574 | [0.533, 0.620] | 0.074 | 0.107 |
| history.page0.5 | 0.646 | [0.584, 0.717] | 0.081 | 0.137 |
| history.glr | 0.639 | [0.585, 0.717] | 0.077 | 0.142 |
| residual.page0.5 | 0.507 | [0.460, 0.557] | 0.079 | 0.086 |
| calibration_selected | 0.662 | [0.598, 0.733] | 0.077 | 0.178 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.613 | [0.522, 0.705] | 0.079 | 0.371 |
| phase.page0.5 | 0.591 | [0.512, 0.687] | 0.084 | 0.437 |
| history.point | 0.578 | [0.515, 0.661] | 0.022 | 0.259 |
| history.window4 | 0.577 | [0.504, 0.659] | 0.088 | 0.355 |
| history.page0.5 | 0.598 | [0.517, 0.685] | 0.083 | 0.426 |
| history.glr | 0.589 | [0.517, 0.686] | 0.077 | 0.416 |
| residual.page0.5 | 0.527 | [0.470, 0.591] | 0.082 | 0.183 |
| calibration_selected | 0.629 | [0.543, 0.720] | 0.094 | 0.426 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
