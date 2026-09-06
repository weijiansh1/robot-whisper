# Full-40 五折交叉拟合结论

## 是否是健康数据太少

40 个任务、32,000 条轨迹、30,904 条成功轨迹已全部纳入五折审计。每折使用 30 个 states 的约 18,500 条成功轨迹训练，10 个 states 校准，10 个 states 测试；50 个 states 均且仅测试一次。

task+position+二阶有序历史相对 task+position 的连续 NLL 差为 -0.1984 bits/query，fold 范围 [-0.20809303551442032, -0.19178040953643477]。有序二阶相对二词 bag 的差为 -0.00505 bits/query，fold 范围 [-0.006230592756451035, -0.004208419117775095]；五折区间均不含零。

固定 tokenizer 的样本量消融：

| train states | 平均成功轨迹 | history-phase | ordered-bag | 有序增益 folds |
|---:|---:|---:|---:|---:|
| 5 | 3081 | -0.1482 | -0.00021 | 4/5 |
| 10 | 6150 | -0.1613 | -0.00160 | 5/5 |
| 20 | 12343 | -0.1824 | -0.00320 | 5/5 |
| 30 | 18542 | -0.1983 | -0.00505 | 5/5 |

结论：较小数据已经足以看到前一词依赖；更多数据主要让很小的二阶顺序效应稳定下来。因此旧实验并非完全没学到，但确实不足以高精度估计稀疏的有序上下文。

## CUSUM 与早期检测

下面按 task×state 内正负 pair 聚合 AUC；阈值目标为 calibration-success 5% FPR。

| q | 方法 | AUC | fold 范围 | test FPR | failure recall |
|---:|---|---:|---:|---:|---:|
| 3 | old_pooled_cusum | 0.515 | [0.479, 0.531] | 0.060 | 0.089 |
| 3 | phase.page0.5 | 0.509 | [0.475, 0.520] | 0.070 | 0.099 |
| 3 | history.page0.5 | 0.514 | [0.486, 0.527] | 0.070 | 0.097 |
| 3 | history.glr | 0.512 | [0.488, 0.538] | 0.066 | 0.086 |
| 3 | residual.page0.5 | 0.512 | [0.457, 0.558] | 0.070 | 0.077 |
| 7 | old_pooled_cusum | 0.579 | [0.551, 0.605] | 0.067 | 0.148 |
| 7 | phase.page0.5 | 0.591 | [0.561, 0.625] | 0.068 | 0.151 |
| 7 | history.page0.5 | 0.595 | [0.565, 0.633] | 0.067 | 0.151 |
| 7 | history.glr | 0.594 | [0.561, 0.620] | 0.065 | 0.145 |
| 7 | residual.page0.5 | 0.551 | [0.536, 0.577] | 0.071 | 0.104 |
| 12 | old_pooled_cusum | 0.602 | [0.560, 0.636] | 0.066 | 0.332 |
| 12 | phase.page0.5 | 0.607 | [0.540, 0.682] | 0.069 | 0.358 |
| 12 | history.page0.5 | 0.612 | [0.557, 0.681] | 0.067 | 0.358 |
| 12 | history.glr | 0.617 | [0.568, 0.695] | 0.065 | 0.350 |
| 12 | residual.page0.5 | 0.585 | [0.563, 0.610] | 0.068 | 0.144 |

条件校准 history Page-CUSUM 相对旧 pooled CUSUM 的 AUC 增量为 q7 0.016、q12 0.010；但相对 phase-only 仅为 q7 0.004、q12 0.005。

结论：旧 CUSUM 的 pooled calibration 和 uniform-score kappa 确实不理想；改成 task+position 条件 CDF、normal-score Page-CUSUM 后有小幅提升。但提升几乎都能由 phase-only 得到，不是健康句法残差贡献。q3 接近随机，q7 只能弱预测，q12 才有中等信号。

五折 calibration 最优候选分别为：['bag.page0.5', 'history1.point', 'phase.page0', 'bag.window4', 'phase.page0']。选择器跨 fold 不稳定，不建议在线部署时搜索后取最好。

## 边界

- 这里的阳性是最终 episode failure；新数据没有独立物理 failure-onset 标签。所以只能说 q7/q12 预测最终失败，不能证明一定早于错误动作。
- q12 只包含长度大于 12 的 17,992 条成功轨迹，但包含全部 1,096 条失败轨迹；它是 survivor-conditioned 晚期读数，不能与 q7 的总体覆盖直接比较。
- nominal 5% 阈值在 unseen states 上通常漂移到更高 FPR，说明上线前还需要更保守的 state-blocked/conformal 阈值校准。
- 连续 NLL bits 与旧报告的硬 token bits 不是同一绝对标尺；这里只在同一 full-40 protocol 内比较差值和学习曲线。
