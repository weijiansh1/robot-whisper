# HiMoE-VLA 多轨道路由语法实验报告

## 结论

- **跨 query 有序性：支持。** 真实顺序相对 query-shuffle 的 NLL 差为 `-5.4415` bits/token，state-blocked 单侧 `p=3.052e-05`。
- **顺序相对词袋：支持。** PST 相对 bag-context 的 NLL 差为 `-0.0747` bits/token，`p=0.0003204`。
- **相对绝对 query 时钟：支持。** PST 相对 position-only baseline 的 NLL 差为 `-1.3492` bits/token，`p=3.052e-05`。
- **任务与阶段之后的历史增益：支持。** task-position+history 相对 task-position 的 NLL 差为 `-0.1082` bits/token，95% CI `[-0.1397, -0.0764]`，单侧 `p=3.052e-05`。
- **阶段条件的早期失败监控：不支持。** q7/q12 的 task-phase+history 相对 task-phase AUC 增量分别为 `-0.0008`/`-0.0002`；健康预测增益没有转化成早期失败区分。
- **query 内 flow 顺序：支持。** 真实 flow 相对重算后的 flow-shuffle lexical NLL 差为 `-138.2808` nats/query，`p=3.052e-05`；平均 word 改变率 `92.6%`。
- **duration 增益：不支持。** PST+duration 相对 PST 的差为 `0.0378` bits/token。

固定四阶 Markov 相对 PST 的 NLL 差为 `-0.0348` bits/token；因此有序结构是否存在与 PST 是否是最佳模型是两个问题。

因此最终判断是：支持多尺度 routing temporal structure。word history 在已知 task 与绝对 query 位置后仍有 held-out 增量，支持任务内的条件序列结构；由于该模型使用 task 标签，这不等于 task-invariant grammar。 flow-shuffle 只检验构词层。

## 数据与防泄漏

共 `2560` episodes、`51308` queries，其中成功 `2253`、失败 `307`。
按全局 init-state 做 `4` 折交叉拟合；每折 train、calibration、test state 互斥。同一 task/init-state 下 32 条 noise siblings 不会跨 split。
Tokenizer、PCA、GMM、grammar 只看训练 state 的成功 episode；词表大小只按 calibration 成功轨迹选择；所有报告数来自未见 test state。

## 健康序列预测

| 模型 | bits/token |
|---|---:|
| unigram | 4.8547 |
| position | 2.7543 |
| position_context | 1.3981 |
| bigram | 1.8407 |
| markov4 | 1.3784 |
| bag6 | 1.4883 |
| pst6 | 1.4126 |
| pst6_duration | 1.4500 |
| task_unigram | 3.0096 |
| task_position | 1.1783 |
| task_position_context | 1.0707 |
| task_pst6 | 1.2191 |

task-conditioned 项使用任务标签，只是检查任务异质性的 oracle control，不属于严格 MoE-only 在线模型。
`position_context` 是严格嵌套对照：先给定绝对位置，再只用同一位置内的最近历史更新；`task_position_context` 进一步给定任务标签。后者是本轮判断 history 是否超出任务阶段时钟的主检验。

## 固定前缀失败区分

这是次级 outcome 评价，不等同于 loop/static 因果识别。AUC 在 task × init-state 内配对后汇总。

| horizon | N (S/F) | single | bag | ordered | +duration | behavior |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 2560 (2253/307) | 0.609 | 0.614 | 0.615 | 0.626 | 0.500 |
| 12 | 1625 (1318/307) | 0.545 | 0.541 | 0.543 | 0.536 | 0.526 |
| 20 | 607 (300/307) | 0.501 | 0.503 | 0.500 | 0.544 | 0.526 |
| 27 | 554 (296/258) | 0.505 | 0.501 | 0.500 | 0.556 | 0.443 |
| 34 | 512 (296/216) | 0.616 | 0.610 | 0.611 | 0.725 | 0.792 |

有序模型相对基线的 AUC 增量（state-blocked 95% CI）：

| horizon | ordered-single | ordered-bag | duration-ordered |
|---:|---:|---:|---:|
| 7 | +0.006 [-0.005, +0.023] | +0.001 [-0.004, +0.005] | +0.011 [-0.005, +0.025] |
| 12 | -0.002 [-0.009, +0.006] | +0.002 [-0.003, +0.010] | -0.007 [-0.038, +0.023] |
| 20 | -0.001 [-0.023, +0.024] | -0.003 [-0.018, +0.013] | +0.044 [-0.024, +0.131] |
| 27 | -0.005 [-0.020, +0.010] | -0.001 [-0.007, +0.006] | +0.056 [-0.016, +0.123] |
| 34 | -0.005 [-0.019, +0.009] | +0.001 [+0.000, +0.004] | +0.114 [+0.011, +0.241] |

### 阶段条件失败审计

下表仍是 within task × state AUC。`residual` 为 history NLL 减 position NLL，直接读取相对阶段基线的额外异常；task 项是使用任务标签的 oracle control。

| horizon | phase | phase+history | residual | task+phase | task+phase+history | task residual |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.613 | 0.618 | 0.515 | 0.618 | 0.618 | 0.496 |
| 12 | 0.550 | 0.549 | 0.562 | 0.544 | 0.543 | 0.548 |
| 20 | 0.504 | 0.501 | 0.563 | 0.502 | 0.493 | 0.565 |
| 27 | 0.505 | 0.502 | 0.627 | 0.503 | 0.502 | 0.491 |
| 34 | 0.618 | 0.632 | 0.774 | 0.624 | 0.626 | 0.733 |

history 模型相对 phase baseline 的 AUC 增量（state-blocked 95% CI）：

| horizon | phase+history - phase | task phase+history - task phase | task residual - task phase |
|---:|---:|---:|---:|
| 7 | +0.005 [+0.000, +0.010] | -0.001 [-0.006, +0.003] | -0.123 [-0.195, -0.053] |
| 12 | -0.001 [-0.010, +0.006] | -0.000 [-0.004, +0.004] | +0.005 [-0.049, +0.059] |
| 20 | -0.003 [-0.027, +0.024] | -0.009 [-0.022, +0.002] | +0.063 [-0.038, +0.185] |
| 27 | -0.003 [-0.015, +0.007] | -0.001 [-0.005, +0.000] | -0.012 [-0.094, +0.059] |
| 34 | +0.015 [+0.002, +0.035] | +0.002 [-0.009, +0.013] | +0.109 [+0.004, +0.215] |

## Episode 误报与 scene8 停滞

阈值目标为 calibration 健康 episode 级 `5.0%` FPR。
scene8 事件比较使用 scene8 成功 calibration 重新标定 CDF/阈值；这是带任务标签的公平 operating-point audit，不属于严格全局 MoE-only 部署值。

| 方法 | 全局 test 健康 FPR | scene8 健康 FPR | onset 前召回 | onset+3 召回 |
|---|---:|---:|---:|---:|
| single | 0.070 | 0.081 | 0.107 | 0.112 |
| bag | 0.058 | 0.088 | 0.107 | 0.112 |
| ordered | 0.066 | 0.095 | 0.107 | 0.117 |
| ordered_duration | 0.061 | 0.084 | 0.071 | 0.081 |
| behavior | 0.058 | 0.179 | 0.218 | 0.234 |

阶段条件方法的 scene8 task-calibrated operating point：

| 方法 | scene8 健康 FPR | onset 前召回 | onset+3 召回 |
|---|---:|---:|---:|
| phase | 0.084 | 0.127 | 0.127 |
| phase_history | 0.084 | 0.102 | 0.117 |
| phase_residual | 0.074 | 0.066 | 0.112 |
| task_phase | 0.068 | 0.091 | 0.096 |
| task_phase_history | 0.078 | 0.096 | 0.102 |
| task_phase_residual | 0.061 | 0.091 | 0.162 |

全局 calibration 下的 scene8 onset 前召回分别为：single=0.010, bag=0.020, ordered=0.020, ordered_duration=0.066, behavior=0.467。

## 多尺度 Fl 与 Sy

预先分开计算：`Fl_query` 为最近 7 个 query 的平均 entropy slope；`Fl_flow` 为同一 query 内 late-minus-early flow entropy；`Sy` 为 L15/f9 的 token Top-4 support consensus。下表为 scene8 停滞失败相对成功的 within-state AUC。

| q | Fl_query | Fl_flow | Sy_L15/f9 | LowRank_L15/f9 |
|---:|---:|---:|---:|---:|
| 12 | 0.478 | 0.534 | 0.487 | 0.554 |
| 20 | 0.364 | 0.535 | 0.521 | 0.512 |
| 34 | 0.768 | 0.268 | 0.717 | 0.569 |

两轴在全部 scene8 queries 上的 Spearman `rho=0.022`，在 q34 为 `rho=0.350`。这支持将二者作为并行轨道读取，而不是硬编码成先后字母；但 q34 是晚期关联。

## 解释边界

- scene8 onset 来自仿真中两只 moka pot 的物理进展定义，不使用 MoE；19 条含糊失败被排除。
- `Fl` 是跨 query 或 flow 的 entropy 趋势，`Sy` 是 L15/f9 的 Top-4 支持同步；报告分别计算，不会把它们硬编码成互斥的 `Fl Fl Sy Sy`。
- 较晚 horizon 会有 survivor/episode-length 选择，尤其 t27/t34 只能解释为晚期读数。
- 主实验 split 阻断了 task/init-state root siblings；独立的 state+seed 双轴留出结果由 `run_dual_axis_audit` 生成，不能用主实验的全样本覆盖数字替代。
- scaler/PCA/GMM 按健康 query 拟合，较长的 scene8 成功轨迹会贡献更多 tokenizer 权重；held-out NLL 汇总则以 episode 为单位。
- behavior baseline 和相关性用于检查信号是否只是动作/物理停滞的读出；本实验不能给出路由因果结论。
- 当前数据没有可靠的 loop/static/mixed 分类和延长 timeout rollout，因而不评价 `<END>` 延长策略、候选选择或训练时正则化。

机器可读的完整效应、置信区间、fold 选择和 phenotype 结果见 `summary.json`。
