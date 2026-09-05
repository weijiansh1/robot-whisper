# Full-40 五折交叉拟合结论

## 是否是健康数据太少

40 个任务、32,000 条轨迹、30,904 条成功轨迹已全部纳入五折审计。每折使用 30 个 states 的约 18,500 条成功轨迹训练，10 个 states 校准，10 个 states 测试；50 个 states 均且仅测试一次。

task+position+二阶有序历史相对 task+position 的连续 NLL 差为 -0.2091 bits/query，fold 范围 [-0.21691362445925377, -0.20397734521571062]。有序二阶相对二词 bag 的差为 -0.00371 bits/query，fold 范围 [-0.004000821205465672, -0.003260094283350319]；五折区间均不含零。

固定 tokenizer 的样本量消融：

| train states | 平均成功轨迹 | history-phase | ordered-bag | 有序增益 folds |
|---:|---:|---:|---:|---:|
| 5 | 3081 | -0.2017 | -0.00021 | 3/5 |
| 10 | 6150 | -0.1990 | -0.00129 | 5/5 |
| 20 | 12343 | -0.2014 | -0.00270 | 5/5 |
| 30 | 18542 | -0.2091 | -0.00371 | 5/5 |

结论：较小数据已经足以看到前一词依赖；更多数据主要让很小的二阶顺序效应稳定下来。因此旧实验并非完全没学到，但确实不足以高精度估计稀疏的有序上下文。

## CUSUM 与早期检测

下面按 task×state 内正负 pair 聚合 AUC；阈值目标为 calibration-success 5% FPR。

| q | 方法 | AUC | fold 范围 | test FPR | failure recall |
|---:|---|---:|---:|---:|---:|
| 3 | old_pooled_cusum | 0.510 | [0.490, 0.545] | 0.060 | 0.089 |
| 3 | phase.page0.5 | 0.520 | [0.474, 0.552] | 0.073 | 0.090 |
| 3 | history.page0.5 | 0.522 | [0.481, 0.546] | 0.072 | 0.085 |
| 3 | history.glr | 0.519 | [0.487, 0.549] | 0.072 | 0.076 |
| 3 | residual.page0.5 | 0.516 | [0.497, 0.561] | 0.063 | 0.069 |
| 7 | old_pooled_cusum | 0.563 | [0.531, 0.609] | 0.065 | 0.128 |
| 7 | phase.page0.5 | 0.594 | [0.561, 0.646] | 0.071 | 0.140 |
| 7 | history.page0.5 | 0.600 | [0.567, 0.652] | 0.071 | 0.138 |
| 7 | history.glr | 0.595 | [0.569, 0.642] | 0.069 | 0.124 |
| 7 | residual.page0.5 | 0.560 | [0.524, 0.606] | 0.069 | 0.088 |
| 12 | old_pooled_cusum | 0.587 | [0.549, 0.629] | 0.071 | 0.320 |
| 12 | phase.page0.5 | 0.616 | [0.543, 0.666] | 0.076 | 0.343 |
| 12 | history.page0.5 | 0.622 | [0.543, 0.676] | 0.076 | 0.338 |
| 12 | history.glr | 0.624 | [0.576, 0.673] | 0.070 | 0.328 |
| 12 | residual.page0.5 | 0.577 | [0.527, 0.601] | 0.071 | 0.157 |

条件校准 history Page-CUSUM 相对旧 pooled CUSUM 的 AUC 增量为 q7 0.037、q12 0.035；但相对 phase-only 仅为 q7 0.006、q12 0.006。

结论：旧 CUSUM 的 pooled calibration 和 uniform-score kappa 确实不理想；改成 task+position 条件 CDF、normal-score Page-CUSUM 后有小幅提升。但提升几乎都能由 phase-only 得到，不是健康句法残差贡献。q3 接近随机，q7 只能弱预测，q12 才有中等信号。

五折 calibration 最优候选分别为：['lexical.page0.5', 'history1.page1', 'bag.page1', 'phase.page1', 'phase.page1']。选择器跨 fold 不稳定，不建议在线部署时搜索后取最好。

## 边界

- 这里的阳性是最终 episode failure；新数据没有独立物理 failure-onset 标签。所以只能说 q7/q12 预测最终失败，不能证明一定早于错误动作。
- q12 只包含长度大于 12 的 17,992 条成功轨迹，但包含全部 1,096 条失败轨迹；它是 survivor-conditioned 晚期读数，不能与 q7 的总体覆盖直接比较。
- nominal 5% 阈值在 unseen states 上通常漂移到更高 FPR，说明上线前还需要更保守的 state-blocked/conformal 阈值校准。
- 连续 NLL bits 与旧报告的硬 token bits 不是同一绝对标尺；这里只在同一 full-40 protocol 内比较差值和学习曲线。
