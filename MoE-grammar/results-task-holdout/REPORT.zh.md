# Leave-one-task-out 语法迁移审计

## 结论

- **主检验：不支持零样本跨任务历史迁移。** 在完全未见任务上，position+ordered-history 相对 position 的 task-macro 差为 `+0.1787` bits/token，task-bootstrap 95% CI `[+0.0216, +0.3886]`，`1/5` 个任务同方向。
- **顺序增量方向一致但证据不足。** ordered-2 相对 bag-2 为 `-0.0214` bits/token，95% CI `[-0.0476, +0.0004]`，`4/5` 个任务同方向。
- task+position 条件下的历史增益只在任务内成立；本审计不支持将它升级为零样本 task-invariant grammar。任务数只有 5，结论仍需新 benchmark family 验证。

## 协议

每折完整留出一个任务。RobustScaler、PCA、K=32 GMM tokenizer 和全部 grammar counts 只使用其余四个任务的成功 episode；测试只使用被留出任务的成功 episode。失败标签未参与。
五折合计恰好覆盖 `2253` 条成功测试 episode。

## 健康序列预测

| 模型 | task-macro bits/token | episode-weighted |
|---|---:|---:|
| unigram | 4.7864 | 4.7950 |
| position | 6.2643 | 6.3029 |
| position_context1 | 6.4441 | 6.4743 |
| position_bag_context | 6.4643 | 6.4967 |
| position_context | 6.4430 | 6.4729 |
| bigram | 5.1084 | 5.0791 |
| markov4 | 5.3920 | 5.3344 |
| bag6 | 5.5612 | 5.5144 |
| pst6 | 5.3864 | 5.3283 |
| pst6_duration | 5.6054 | 5.5668 |

## 配对效应

负数表示左侧模型 NLL 更低。区间以 5 个 held-out tasks 为重采样单位。

| 对比 | task-macro 差值 | 95% CI | 同方向任务 |
|---|---:|---:|---:|
| position_context1_vs_position | +0.1798 | [+0.0317, +0.3878] | 0/5 |
| position_context_vs_position | +0.1787 | [+0.0216, +0.3886] | 1/5 |
| position_context_vs_position_bag_context | -0.0214 | [-0.0476, +0.0004] | 4/5 |
| position_context_vs_position_context1 | -0.0011 | [-0.0193, +0.0180] | 3/5 |
| pst_vs_bag | -0.1749 | [-0.3521, +0.0478] | 4/5 |
| pst_vs_markov4 | -0.0056 | [-0.0140, -0.0001] | 4/5 |

## 边界

- 5 个任务不足以精确估计新任务分布，bootstrap 区间只表达当前任务集合的不确定性。
- 同一 checkpoint、benchmark family 与数据生成流程仍可能产生共享结构。
- 本实验验证预测迁移，不验证异常检测、控制收益或因果机制。
- 结果应在新任务 family 或新 checkpoint 上确认后再使用 task-invariant 表述。

完整逐任务效应和 split 诊断见 `summary.json`。
