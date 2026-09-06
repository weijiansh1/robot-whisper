# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 39.6123 |
| history1 | 39.4109 |
| bag | 39.4161 |
| history | 39.4098 |

二阶有序历史相对 task+position 的差为 -0.2023 bits/query (state-blocked 95% CI [-0.2145989683290093, -0.1900528701328748])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0062 bits/query (95% CI [-0.008066763483110832, -0.004627705395692838])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.605 | [0.541, 0.661] | 0.080 | 0.137 |
| phase.page0.5 | 0.606 | [0.541, 0.676] | 0.075 | 0.127 |
| history.point | 0.593 | [0.541, 0.654] | 0.051 | 0.117 |
| history.window4 | 0.549 | [0.501, 0.606] | 0.081 | 0.122 |
| history.page0.5 | 0.604 | [0.542, 0.670] | 0.077 | 0.127 |
| history.glr | 0.608 | [0.549, 0.679] | 0.067 | 0.142 |
| residual.page0.5 | 0.536 | [0.496, 0.578] | 0.071 | 0.137 |
| calibration_selected | 0.602 | [0.536, 0.669] | 0.078 | 0.127 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.574 | [0.459, 0.671] | 0.072 | 0.340 |
| phase.page0.5 | 0.570 | [0.473, 0.671] | 0.075 | 0.411 |
| history.point | 0.575 | [0.488, 0.663] | 0.036 | 0.198 |
| history.window4 | 0.567 | [0.474, 0.644] | 0.091 | 0.305 |
| history.page0.5 | 0.569 | [0.467, 0.667] | 0.072 | 0.411 |
| history.glr | 0.568 | [0.464, 0.672] | 0.073 | 0.381 |
| residual.page0.5 | 0.563 | [0.512, 0.635] | 0.066 | 0.173 |
| calibration_selected | 0.570 | [0.462, 0.674] | 0.072 | 0.411 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
