# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 36.9889 |
| history1 | 36.7410 |
| bag | 36.7614 |
| history | 36.7600 |

二阶有序历史相对 task+position 的差为 -0.2290 bits/query (state-blocked 95% CI [-0.23803267244718804, -0.2204118919985398])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0014 bits/query (95% CI [-0.0017518960088575186, -0.0009743078155527204])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.537 | [0.481, 0.597] | 0.053 | 0.142 |
| phase.page0.5 | 0.566 | [0.506, 0.631] | 0.055 | 0.167 |
| history.point | 0.564 | [0.511, 0.633] | 0.022 | 0.029 |
| history.window4 | 0.574 | [0.540, 0.611] | 0.057 | 0.075 |
| history.page0.5 | 0.566 | [0.520, 0.624] | 0.053 | 0.158 |
| history.glr | 0.559 | [0.504, 0.611] | 0.053 | 0.113 |
| residual.page0.5 | 0.558 | [0.510, 0.614] | 0.054 | 0.075 |
| calibration_selected | 0.588 | [0.548, 0.640] | 0.052 | 0.138 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.632 | [0.538, 0.719] | 0.059 | 0.321 |
| phase.page0.5 | 0.654 | [0.571, 0.728] | 0.050 | 0.367 |
| history.point | 0.637 | [0.572, 0.701] | 0.020 | 0.158 |
| history.window4 | 0.674 | [0.626, 0.722] | 0.044 | 0.242 |
| history.page0.5 | 0.658 | [0.574, 0.729] | 0.052 | 0.371 |
| history.glr | 0.656 | [0.568, 0.736] | 0.048 | 0.354 |
| residual.page0.5 | 0.591 | [0.533, 0.645] | 0.046 | 0.092 |
| calibration_selected | 0.670 | [0.570, 0.743] | 0.056 | 0.379 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
