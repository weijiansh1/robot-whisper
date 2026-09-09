# 物理失败轨迹的 HB-MoE 全程变化

## 覆盖与比较口径

共对齐 `1442` 条物理失败、`4206` 条同 run/同初始状态候选成功对照，覆盖 `78` 个源 run。
其中 `1306` 条失败存在同 run、同初始状态成功对照；`136` 条只进入原始描述，不进入 success-normalized 主比较。
主观察性描述使用 `1403` 条失败，其中 `1267` 条进入 success-normalized 比较；`pin-base` 的逐字节重复失败和 `pin-on/off` 前层干预失败单列，不作为独立自然样本。

每个百分位都先在同 source run、同 init-state 的成功轨迹内计算，再以 episode 为单位汇总。`0.5` 表示与匹配成功无差异，越大表示该指标在失败中越高。专家编号只在相同 checkpoint 内比较。

## 全程主结果

最稳定的结构不是简单的全局塌缩，而是两个时间尺度方向相反：同一 chunk 的 d0→d9 路由变化更大，但相邻 chunk 的同位置路由、尤其后层 Top-4，反而更粘滞。后层的 state/action gap 同时升高。

| 指标（末段或全程） | 失败原始均值 | 匹配成功原始均值 | episode 百分位 | task 宏平均 | task-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| `late_front_action_entropy` | 0.9982 | 0.9983 | 0.350 | 0.283 | [0.226, 0.345] |
| `late_back_action_entropy` | 0.9979 | 0.9983 | 0.344 | 0.248 | [0.183, 0.319] |
| `late_front_token_dispersion` | 0.0305 | 0.0297 | 0.596 | 0.657 | [0.598, 0.710] |
| `late_back_token_dispersion` | 0.0246 | 0.0247 | 0.469 | 0.549 | [0.471, 0.627] |
| `late_front_denoise_soft_change` | 0.0103 | 0.0097 | 0.731 | 0.801 | [0.756, 0.841] |
| `late_back_denoise_soft_change` | 0.0084 | 0.0075 | 0.678 | 0.800 | [0.741, 0.855] |
| `late_front_query_soft_change` | 0.0366 | 0.0382 | 0.386 | 0.415 | [0.346, 0.482] |
| `late_back_query_soft_change` | 0.0308 | 0.0408 | 0.220 | 0.279 | [0.206, 0.364] |
| `late_front_query_hard_churn` | 0.7836 | 0.8313 | 0.197 | 0.165 | [0.120, 0.215] |
| `late_back_query_hard_churn` | 0.6724 | 0.8534 | 0.086 | 0.087 | [0.052, 0.135] |
| `back_boundary_T10_T1_soft_change` | 0.0597 | 0.0568 | 0.599 | 0.637 | [0.594, 0.676] |

完整 0%–100% 相位曲线在 `trajectory_phase_curves.csv`；末端点仍会包含成功提前终止、失败运行到 horizon 的语义差异，因此不能把末端分离直接解释为早期预警。

## Chunk 内：前层与后层

下面是失败相对匹配成功的 late-phase、动作 token T1–T10 平均百分位，列顺序为 d0（最噪）到 d9（最后一次前向）：

- `entropy` front: 0.451, 0.445, 0.430, 0.413, 0.378, 0.353, 0.345, 0.344, 0.363, 0.286
- `entropy` back:  0.380, 0.374, 0.369, 0.368, 0.363, 0.356, 0.351, 0.349, 0.356, 0.406
- `top1_mass` front: 0.583, 0.590, 0.597, 0.621, 0.642, 0.670, 0.679, 0.680, 0.669, 0.661
- `top1_mass` back:  0.627, 0.635, 0.642, 0.643, 0.644, 0.643, 0.641, 0.637, 0.626, 0.603
- `top4_mass` front: 0.583, 0.594, 0.604, 0.632, 0.667, 0.700, 0.714, 0.719, 0.701, 0.695
- `top4_mass` back:  0.625, 0.634, 0.641, 0.643, 0.646, 0.648, 0.650, 0.647, 0.642, 0.625

state token 按因果位置应基本不随 d 改变。实测每 episode 最大差的中位数为 `0.000673`，p95 为 `0.00387`，全局最大为 `0.013`；因此逐 d 的 state 值不作为十次独立证据。

## Chunk 间：后层逐 token

| token | 相邻 soft-route 变化百分位 | Top-4 churn 百分位 |
|---:|---:|---:|
| state | 0.299 | 0.300 |
| T1 | 0.366 | 0.285 |
| T2 | 0.371 | 0.277 |
| T3 | 0.376 | 0.269 |
| T4 | 0.385 | 0.277 |
| T5 | 0.398 | 0.272 |
| T6 | 0.399 | 0.257 |
| T7 | 0.402 | 0.260 |
| T8 | 0.406 | 0.271 |
| T9 | 0.392 | 0.268 |
| T10 | 0.379 | 0.268 |

这里比较相邻 query 的同位置 token；`T10(q) -> T1(q+1)` 的真实 chunk 边界另见 `boundary` 指标。soft-route 使用完整 32-way 概率，hard churn 使用权威保存的 Top-4 集合。

## 不同物理失败原因

下表列出预设 headline 指标中偏离 0.5 最大的一项。百分位只使用有同 run、同初始状态成功对照的失败；它是探索性摘要，不是从多重扫描中得到的确认性因果结论，样本少或只覆盖少数任务的原因尤其要谨慎。

| 物理失败原因 | 总数 | 已匹配 | 最强 headline 指标 | task 宏平均百分位 | 支持任务 |
|---|---:|---:|---|---:|---:|
| 目标前释放或掉落 | 548 | 531 | `late_back_query_hard_churn` | 0.081 | 31 |
| 未观测到稳定抓取 | 442 | 365 | `late_back_query_hard_churn` | 0.057 | 13 |
| 物体移动但目标未满足 | 200 | 190 | `late_back_query_hard_churn` | 0.131 | 26 |
| 目标谓词回退 | 65 | 57 | `late_back_query_hard_churn` | 0.048 | 4 |
| 持物超时 | 57 | 50 | `late_back_query_hard_churn` | 0.132 | 9 |
| 目标区外释放 | 48 | 48 | `late_back_query_hard_churn` | 0.090 | 18 |
| 接近但未观测到接触 | 32 | 17 | `late_back_query_hard_churn` | 0.079 | 6 |
| 机构阈值未达到 | 7 | 7 | `late_back_query_hard_churn` | 0.000 | 2 |
| 无显著目标进展 | 4 | 2 | `late_d9_back_action_top1` | 0.114 | 2 |

这里的九类是物理 outcome 标签，不等同于停滞、来回摆或周期运动标签。原始 1442 条物理标签没有 `stagnation`、`oscillation` 或 `periodicity` 字段，因此本表不把这些运动模式强行映射到物理原因。独立的 Scene8 运动学对照只覆盖 512 条轨迹（216 条失败，其中 197 条为预定义停滞型陷入）：它支持末期低变化/粘滞，不支持校正后的 period-2 到 period-5 MoE 周期；该结论只作为子集证据，不能外推成 1442 条的周期计数。详见 `../../moe-token-dynamics/results/CONCLUSION.zh.md`。

## Pin 数据单列

这些数只描述各 arm 内失败相对该 arm 成功的路由位置，不估计 pin 干预的因果效应。`pin-base` 又与对应 `right-16x32` 逐字节重复。

| run | 失败 | late 前层 entropy 百分位 | late 前层 hard churn 百分位 | late 后层 hard churn 百分位 |
|---|---:|---:|---:|---:|
| `pin-base` | 12 | 0.123 | 0.040 | 0.002 |
| `pin-off` | 19 | 0.156 | 0.028 | 0.001 |
| `pin-on` | 8 | 0.123 | 0.000 | 0.000 |

## AS 与完整性检查

- AS 概率在单 episode 内的最大跨度为 `0`；平均相邻 hard route 变化率为 `0`。
- 所有 Zarr 行均通过 episode id、summary inference_calls 和 offset 三重对齐。
- `pin-on/off` 的 soft router 概率是干预前 gate 分布，而保存的 hard Top-4 是干预后实际执行选择；两者只在干预附表中解释。
- 这些结果说明路由状态与失败模式相关，不证明 MoE 路由是物理失败的原因。视觉 token、attention、expert hidden/output 和实际动作几何不在本分析中。

## 产物

- `cohort.csv`: episode 对齐、失败标签、run regime 和匹配对照数。
- `trajectory_phase_curves.csv`: 0%–100% 全程前/后层曲线。
- `within_chunk_profiles.csv`: early/middle/late × d0–d9 × state/T1–T10 × 前/后层。
- `back_token_cross_chunk.csv`: 后层各 token 的 lag-1/lag-2 soft 与 Top-4 变化。
- `checkpoint_expert_load_shifts.csv`: checkpoint 内逐 expert 的失败-成功 load 差。
- `special_pin_run_effects.csv`: 三个 pin run 的失败子集描述。
- `episode_features.npz` / `matched_failure_features.npz`: 可复核的 episode 级数组。
