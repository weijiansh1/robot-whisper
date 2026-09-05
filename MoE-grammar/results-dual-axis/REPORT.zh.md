# State + noise-seed 双轴留出复验

## 结论

- **主检验：支持。历史在 task 与绝对 query 阶段之外仍有稳定预测增益。**
- task-position+history 相对 task-position 为 `-0.1341` bits/token，crossed bootstrap 95% CI `[-0.1623, -0.1094]`。
- 16 个 crossed cells 中 `16/16` 个方向为改善；cell effect 范围为 `[-0.1911, -0.0764]`。

这是对共享 noise-seed ID 解释的泄漏审计，不是新数据集复现。词表 K=32、PCA=24 和语法超参数在审计前由主实验固定，因此本轮不重新调参。

## 双轴协议

将 16 个 states 分成 `4` 组、32 个 noise seeds 分成 `4` 组，取全部 16 个 state-group × seed-group 测试 cell。
对每个 cell，训练只使用成功 episode，并排除所有共享测试 state **或**测试 seed 的样本。每条成功 episode 恰好作为测试样本一次。

测试覆盖 `2253` 条成功 episode；总语料为 `2560` episodes / `51308` queries。

## 健康序列预测

| 模型 | bits/token |
|---|---:|
| bag6 | 1.4291 |
| bigram | 1.7789 |
| markov4 | 1.3209 |
| position | 2.7469 |
| position_context | 1.2720 |
| pst6 | 1.3516 |
| pst6_duration | 1.4085 |
| task_position | 1.1135 |
| task_position_context | 0.9793 |
| task_pst6 | 1.1665 |
| task_unigram | 3.0090 |
| unigram | 4.8944 |

## 配对效应

负数表示左侧模型 NLL 更低。区间按 state 与 seed 两个聚类轴独立重采样。

| 对比 | 差值 | crossed 95% CI | state p< | seed p< | 改善 cells |
|---|---:|---:|---:|---:|---:|
| position_context_vs_position | -1.4749 | [-1.5358, -1.4110] | 3.052e-05 | 5e-06 | 16/16 |
| pst_vs_bag | -0.0775 | [-0.0973, -0.0575] | 4.578e-05 | 5e-06 | 16/16 |
| pst_vs_markov4 | +0.0307 | [+0.0204, +0.0425] | 1 | 1 | 2/16 |
| pst_vs_position | -1.3953 | [-1.4696, -1.3254] | 3.052e-05 | 5e-06 | 16/16 |
| task_position_context_vs_task_position | -0.1341 | [-0.1623, -0.1094] | 3.052e-05 | 5e-06 | 16/16 |
| task_pst_vs_task_position | +0.0530 | [-0.0099, +0.1157] | 0.9517 | 1 | 5/16 |

## 边界

- 双轴 bootstrap 用于稳健性区间；state-only 与 seed-only sign test 仅作敏感性检查。
- task-conditioned 模型使用任务标签，是解释性 oracle control，不是严格 MoE-only 部署模型。
- 复验仍来自同一批任务与 rollouts，不能替代新任务、新 checkpoint 或干预实验。
- 本报告只复验健康序列预测，不利用失败标签，也不选择控制策略。

完整 split、cell 和机器可读统计见 `summary.json`。
