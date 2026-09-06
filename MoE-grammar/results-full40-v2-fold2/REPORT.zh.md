# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 39.1187 |
| history1 | 38.9117 |
| bag | 38.9158 |
| history | 38.9106 |

二阶有序历史相对 task+position 的差为 -0.2081 bits/query (state-blocked 95% CI [-0.21789866743049474, -0.1994782436480927])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0051 bits/query (95% CI [-0.006217028214908919, -0.004265134969948186])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.596 | [0.545, 0.662] | 0.079 | 0.223 |
| phase.page0.5 | 0.591 | [0.535, 0.663] | 0.085 | 0.217 |
| history.point | 0.607 | [0.543, 0.681] | 0.063 | 0.160 |
| history.window4 | 0.554 | [0.480, 0.626] | 0.070 | 0.189 |
| history.page0.5 | 0.598 | [0.546, 0.677] | 0.082 | 0.211 |
| history.glr | 0.606 | [0.552, 0.675] | 0.084 | 0.183 |
| residual.page0.5 | 0.539 | [0.483, 0.598] | 0.067 | 0.131 |
| calibration_selected | 0.597 | [0.557, 0.657] | 0.086 | 0.223 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.627 | [0.567, 0.686] | 0.072 | 0.280 |
| phase.page0.5 | 0.609 | [0.536, 0.680] | 0.080 | 0.354 |
| history.point | 0.599 | [0.534, 0.663] | 0.039 | 0.206 |
| history.window4 | 0.621 | [0.552, 0.719] | 0.057 | 0.343 |
| history.page0.5 | 0.615 | [0.546, 0.678] | 0.077 | 0.354 |
| history.glr | 0.621 | [0.547, 0.688] | 0.065 | 0.366 |
| residual.page0.5 | 0.586 | [0.549, 0.616] | 0.067 | 0.257 |
| calibration_selected | 0.618 | [0.560, 0.687] | 0.086 | 0.377 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
