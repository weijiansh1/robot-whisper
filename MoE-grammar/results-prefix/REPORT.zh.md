# 因果 routing 前缀监督上界

单向 GRU 在 train states 的成功与失败轨迹上学习最终 outcome；calibration states 用于 early stopping 和 5% task-specific 阈值，所有数字来自未见 test states。这不是健康语法模型，而是 routing 前缀是否含有可学习早期信号的监督上界。

| 输入 | q | within task/state AUC | 95% CI | healthy FPR | failure recall |
|---|---:|---:|---:|---:|---:|
| behavior (3 seeds) | 3 | 0.522 | [0.475, 0.563] | 0.056 | 0.046 |
| behavior (3 seeds) | 7 | 0.633 | [0.586, 0.677] | 0.077 | 0.188 |
| behavior (3 seeds) | 12 | 0.672 | [0.597, 0.738] | 0.083 | 0.416 |
| combined (3 seeds) | 3 | 0.527 | [0.465, 0.586] | 0.065 | 0.086 |
| combined (3 seeds) | 7 | 0.573 | [0.520, 0.639] | 0.070 | 0.142 |
| combined (3 seeds) | 12 | 0.631 | [0.554, 0.690] | 0.080 | 0.371 |
| routing (3 seeds) | 3 | 0.502 | [0.454, 0.540] | 0.071 | 0.071 |
| routing (3 seeds) | 7 | 0.594 | [0.544, 0.646] | 0.078 | 0.117 |
| routing (3 seeds) | 12 | 0.646 | [0.594, 0.699] | 0.080 | 0.376 |

## 配对增量

- q3 `routing_minus_behavior`: AUC 差 -0.020, state-blocked 95% CI [-0.05627795780883018, 0.03171798642648869]。
- q3 `combined_minus_behavior`: AUC 差 0.006, state-blocked 95% CI [-0.06319722874079675, 0.08120477887603365]。
- q7 `routing_minus_behavior`: AUC 差 -0.040, state-blocked 95% CI [-0.07299707858402903, 0.0012366018978761393]。
- q7 `combined_minus_behavior`: AUC 差 -0.060, state-blocked 95% CI [-0.13094545057430393, 0.0005503144654087616]。
- q12 `routing_minus_behavior`: AUC 差 -0.026, state-blocked 95% CI [-0.09395219908550383, 0.05852539623534184]。
- q12 `combined_minus_behavior`: AUC 差 -0.041, state-blocked 95% CI [-0.09670794004463032, 0.02955097891899056]。

注意：标签仍然只是 episode 最终 success/failure，没有物理错误 onset。q7 的结果表示第 8 次规划结束时能够预测最终失败，不足以证明报警发生在错误动作之前。
