# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success bits/query |
|---|---:|
| phase | 35.5527 |
| history1 | 35.3392 |
| bag | 35.3394 |
| history | 35.3358 |

二阶有序历史相对 task+position 的差为 -0.2169 bits/query (state-blocked 95% CI [-0.22293835472295925, -0.21148045949583674])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0036 bits/query (95% CI [-0.004745473597467417, -0.0025552852893989727])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.554 | [0.502, 0.607] | 0.074 | 0.111 |
| phase.page0.5 | 0.573 | [0.506, 0.637] | 0.082 | 0.095 |
| history.point | 0.555 | [0.505, 0.608] | 0.056 | 0.042 |
| history.window4 | 0.523 | [0.467, 0.579] | 0.079 | 0.069 |
| history.page0.5 | 0.578 | [0.531, 0.635] | 0.081 | 0.080 |
| history.glr | 0.569 | [0.511, 0.624] | 0.080 | 0.095 |
| residual.page0.5 | 0.534 | [0.487, 0.583] | 0.061 | 0.057 |
| calibration_selected | 0.556 | [0.511, 0.609] | 0.078 | 0.103 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.629 | [0.489, 0.743] | 0.077 | 0.401 |
| phase.page0.5 | 0.653 | [0.546, 0.773] | 0.085 | 0.382 |
| history.point | 0.584 | [0.481, 0.676] | 0.056 | 0.088 |
| history.window4 | 0.606 | [0.488, 0.738] | 0.101 | 0.309 |
| history.page0.5 | 0.650 | [0.529, 0.755] | 0.086 | 0.378 |
| history.glr | 0.616 | [0.491, 0.725] | 0.081 | 0.363 |
| residual.page0.5 | 0.527 | [0.387, 0.647] | 0.072 | 0.153 |
| calibration_selected | 0.624 | [0.521, 0.712] | 0.089 | 0.370 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
