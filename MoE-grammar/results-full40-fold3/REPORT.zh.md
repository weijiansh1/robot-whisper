# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success bits/query |
|---|---:|
| phase | 35.5175 |
| history1 | 35.3144 |
| bag | 35.3132 |
| history | 35.3094 |

二阶有序历史相对 task+position 的差为 -0.2081 bits/query (state-blocked 95% CI [-0.2175476600358922, -0.19911665909767273])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0038 bits/query (95% CI [-0.005684311073400708, -0.002322220343479144])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.531 | [0.495, 0.577] | 0.046 | 0.068 |
| phase.page0.5 | 0.561 | [0.511, 0.627] | 0.051 | 0.090 |
| history.point | 0.581 | [0.513, 0.636] | 0.043 | 0.068 |
| history.window4 | 0.512 | [0.467, 0.572] | 0.066 | 0.090 |
| history.page0.5 | 0.567 | [0.505, 0.632] | 0.050 | 0.090 |
| history.glr | 0.570 | [0.520, 0.632] | 0.053 | 0.095 |
| residual.page0.5 | 0.524 | [0.480, 0.565] | 0.080 | 0.036 |
| calibration_selected | 0.567 | [0.504, 0.605] | 0.052 | 0.099 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.549 | [0.461, 0.606] | 0.059 | 0.212 |
| phase.page0.5 | 0.543 | [0.490, 0.583] | 0.053 | 0.185 |
| history.point | 0.588 | [0.534, 0.632] | 0.025 | 0.104 |
| history.window4 | 0.540 | [0.498, 0.593] | 0.060 | 0.140 |
| history.page0.5 | 0.543 | [0.493, 0.590] | 0.049 | 0.171 |
| history.glr | 0.576 | [0.523, 0.624] | 0.048 | 0.185 |
| residual.page0.5 | 0.571 | [0.534, 0.614] | 0.063 | 0.072 |
| calibration_selected | 0.569 | [0.527, 0.618] | 0.045 | 0.176 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
