# Full-40 健康语法与在线检测审计

## 数据与隔离

使用 40 个 LIBERO 任务，共 32,000 条轨迹、508,023 个 query；成功 30,904，失败 1,096。按 init-state 分成 30/10/10 个 train/cal/test state，同一 state 的 16 个 noise seed 不跨集合。

Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。

## 健康语法

| 模型 | held-out success continuous mixture NLL / ln2 |
|---|---:|
| phase | 36.6154 |
| history1 | 36.3679 |
| bag | 36.3887 |
| history | 36.3870 |

二阶有序历史相对 task+position 的差为 -0.2282 bits/query (state-blocked 95% CI [-0.23474411851157898, -0.22054079838305332])；负值表示历史改善预测。
有序二阶相对同样上下文的 bag 差为 -0.0017 bits/query (95% CI [-0.002331584522711447, -0.0009374110810963433])。

## q7/q12 在线前缀检测

所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success 得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。

### q7

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.549 | [0.512, 0.582] | 0.045 | 0.122 |
| phase.page0.5 | 0.542 | [0.485, 0.597] | 0.042 | 0.113 |
| history.point | 0.542 | [0.492, 0.589] | 0.028 | 0.041 |
| history.window4 | 0.531 | [0.486, 0.573] | 0.045 | 0.063 |
| history.page0.5 | 0.540 | [0.488, 0.587] | 0.043 | 0.108 |
| history.glr | 0.543 | [0.486, 0.592] | 0.041 | 0.081 |
| residual.page0.5 | 0.514 | [0.480, 0.554] | 0.072 | 0.090 |
| calibration_selected | 0.531 | [0.487, 0.577] | 0.045 | 0.063 |

### q12

| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|
| old_pooled_cusum | 0.524 | [0.420, 0.611] | 0.042 | 0.248 |
| phase.page0.5 | 0.521 | [0.431, 0.605] | 0.050 | 0.293 |
| history.point | 0.479 | [0.391, 0.549] | 0.028 | 0.122 |
| history.window4 | 0.534 | [0.447, 0.599] | 0.048 | 0.180 |
| history.page0.5 | 0.515 | [0.434, 0.591] | 0.049 | 0.293 |
| history.glr | 0.512 | [0.426, 0.585] | 0.048 | 0.239 |
| residual.page0.5 | 0.498 | [0.415, 0.578] | 0.072 | 0.108 |
| calibration_selected | 0.538 | [0.451, 0.613] | 0.048 | 0.180 |

## 解释边界

- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。
- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。
- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；它不能被称为只由健康数据学习的异常检测器。
- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。
