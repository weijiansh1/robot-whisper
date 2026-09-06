# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 39.0592 |
| history1 | 38.8648 |
| bag | 38.8696 |
| history | 38.8651 |

二阶有序历史相对 task+position 的差为 -0.1941 bits/query (state-blocked 95% CI [-0.20427538216400318, -0.18442130005100635])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0045 bits/query (95% CI [-0.00532143884385204, -0.0035755478899134794])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.551 | [0.490, 0.612] | 0.042 | 0.099 |
| phase.page0.5 | 0.561 | [0.496, 0.610] | 0.041 | 0.081 |
| history.point | 0.560 | [0.499, 0.607] | 0.030 | 0.090 |
| history.window4 | 0.544 | [0.489, 0.595] | 0.049 | 0.081 |
| history.page0.5 | 0.565 | [0.505, 0.615] | 0.040 | 0.090 |
| history.glr | 0.561 | [0.486, 0.616] | 0.039 | 0.099 |
| residual.page0.5 | 0.577 | [0.530, 0.626] | 0.074 | 0.068 |
| calibration_selected | 0.546 | [0.490, 0.599] | 0.048 | 0.081 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.560 | [0.500, 0.621] | 0.049 | 0.257 |
| phase.page0.5 | 0.540 | [0.462, 0.612] | 0.043 | 0.239 |
| history.point | 0.556 | [0.487, 0.604] | 0.029 | 0.126 |
| history.window4 | 0.563 | [0.469, 0.620] | 0.046 | 0.234 |
| history.page0.5 | 0.557 | [0.489, 0.625] | 0.043 | 0.243 |
| history.glr | 0.573 | [0.497, 0.641] | 0.049 | 0.243 |
| residual.page0.5 | 0.573 | [0.503, 0.640] | 0.062 | 0.077 |
| calibration_selected | 0.561 | [0.464, 0.618] | 0.046 | 0.234 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
