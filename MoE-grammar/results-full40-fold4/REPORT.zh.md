# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success bits/query |
|---|---:|
| phase | 35.5400 |
| history1 | 35.3425 |
| bag | 35.3377 |
| history | 35.3337 |

二阶有序历史相对 task+position 的差为 -0.2062 bits/query (state-blocked 95% CI [-0.21622931510736482, -0.1973758688202715])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0040 bits/query (95% CI [-0.005385915367607308, -0.0024360668297322076])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.549 | [0.481, 0.626] | 0.055 | 0.121 |
| phase.page0.5 | 0.600 | [0.550, 0.655] | 0.065 | 0.171 |
| history.point | 0.586 | [0.537, 0.639] | 0.033 | 0.046 |
| history.window4 | 0.547 | [0.496, 0.578] | 0.058 | 0.046 |
| history.page0.5 | 0.601 | [0.542, 0.649] | 0.066 | 0.171 |
| history.glr | 0.597 | [0.535, 0.648] | 0.061 | 0.133 |
| residual.page0.5 | 0.606 | [0.557, 0.651] | 0.063 | 0.067 |
| calibration_selected | 0.600 | [0.546, 0.666] | 0.060 | 0.117 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.595 | [0.518, 0.678] | 0.062 | 0.300 |
| phase.page0.5 | 0.666 | [0.580, 0.755] | 0.063 | 0.358 |
| history.point | 0.666 | [0.579, 0.746] | 0.014 | 0.133 |
| history.window4 | 0.640 | [0.589, 0.682] | 0.059 | 0.242 |
| history.page0.5 | 0.676 | [0.587, 0.761] | 0.064 | 0.342 |
| history.glr | 0.673 | [0.578, 0.760] | 0.060 | 0.333 |
| residual.page0.5 | 0.601 | [0.518, 0.657] | 0.059 | 0.117 |
| calibration_selected | 0.683 | [0.541, 0.771] | 0.055 | 0.354 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
