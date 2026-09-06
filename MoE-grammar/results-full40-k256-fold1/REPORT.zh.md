# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 36.9220 |
| history1 | 36.6797 |
| bag | 36.7002 |
| history | 36.6978 |

二阶有序历史相对 task+position 的差为 -0.2242 bits/query (state-blocked 95% CI [-0.23240743590801552, -0.2161068601464227])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0024 bits/query (95% CI [-0.003286958351964943, -0.0016013157395328842])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.564 | [0.507, 0.616] | 0.080 | 0.164 |
| phase.page0.5 | 0.604 | [0.540, 0.679] | 0.070 | 0.137 |
| history.point | 0.589 | [0.534, 0.649] | 0.037 | 0.076 |
| history.window4 | 0.580 | [0.543, 0.620] | 0.084 | 0.103 |
| history.page0.5 | 0.606 | [0.537, 0.672] | 0.070 | 0.137 |
| history.glr | 0.596 | [0.530, 0.658] | 0.064 | 0.122 |
| residual.page0.5 | 0.554 | [0.502, 0.611] | 0.056 | 0.073 |
| calibration_selected | 0.562 | [0.501, 0.621] | 0.072 | 0.103 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.642 | [0.487, 0.778] | 0.089 | 0.397 |
| phase.page0.5 | 0.655 | [0.520, 0.780] | 0.087 | 0.397 |
| history.point | 0.577 | [0.459, 0.689] | 0.025 | 0.145 |
| history.window4 | 0.594 | [0.480, 0.710] | 0.095 | 0.294 |
| history.page0.5 | 0.647 | [0.508, 0.769] | 0.085 | 0.393 |
| history.glr | 0.642 | [0.510, 0.759] | 0.074 | 0.370 |
| residual.page0.5 | 0.532 | [0.445, 0.606] | 0.059 | 0.103 |
| calibration_selected | 0.582 | [0.444, 0.683] | 0.061 | 0.176 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
