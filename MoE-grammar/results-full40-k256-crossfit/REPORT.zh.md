# Full-40 五折交叉拟合结论

## 是否是健康数据太少

40 个任务、32,000 条轨迹、30,904 条成功轨迹已全部纳入五折审计。每折使用 30 个 states 的约 18,500 条成功轨迹训练，10 个 states 校准，10 个 states 测试；50 个 states 均且仅测试一次。

task+position+二阶有序历史相对 task+position 的连续 NLL 差为 -0.2260 bits/query，fold 范围 [-0.22895792990881275, -0.22170669984064822]。有序二阶相对二词 bag 的差为 -0.00171 bits/query，fold 范围 [-0.002402712345892684, -0.0013543138507083872]；五折区间均不含零。

固定 tokenizer 的样本量消融：

| train states | 平均成功轨迹 | history-phase | ordered-bag | 有序增益 folds |
|---:|---:|---:|---:|---:|
| 5 | 3081 | -0.1920 | 0.00013 | 1/5 |
| 10 | 6150 | -0.2070 | -0.00034 | 5/5 |
| 20 | 12343 | -0.2185 | -0.00105 | 5/5 |
| 30 | 18542 | -0.2260 | -0.00171 | 5/5 |

结论：较小数据已经足以看到前一词依赖；更多数据主要让很小的二阶顺序效应稳定下来。因此旧实验并非完全没学到，但确实不足以高精度估计稀疏的有序上下文。

## CUSUM 与早期检测

下面按 task×state 内正负 pair 聚合 AUC；阈值目标为 calibration-success 5% FPR。

| q | 方法 | AUC | fold 范围 | test FPR | failure recall |
|---:|---|---:|---:|---:|---:|
| 3 | old_pooled_cusum | 0.503 | [0.452, 0.535] | 0.066 | 0.092 |
| 3 | phase.page0.5 | 0.507 | [0.454, 0.530] | 0.069 | 0.098 |
| 3 | history.page0.5 | 0.508 | [0.451, 0.529] | 0.069 | 0.099 |
| 3 | history.glr | 0.511 | [0.452, 0.533] | 0.065 | 0.075 |
| 3 | residual.page0.5 | 0.516 | [0.473, 0.540] | 0.065 | 0.053 |
| 7 | old_pooled_cusum | 0.578 | [0.537, 0.643] | 0.066 | 0.151 |
| 7 | phase.page0.5 | 0.593 | [0.542, 0.643] | 0.067 | 0.155 |
| 7 | history.page0.5 | 0.594 | [0.540, 0.646] | 0.067 | 0.153 |
| 7 | history.glr | 0.588 | [0.543, 0.639] | 0.064 | 0.133 |
| 7 | residual.page0.5 | 0.537 | [0.507, 0.558] | 0.067 | 0.078 |
| 12 | old_pooled_cusum | 0.605 | [0.524, 0.642] | 0.068 | 0.332 |
| 12 | phase.page0.5 | 0.607 | [0.521, 0.655] | 0.068 | 0.376 |
| 12 | history.page0.5 | 0.608 | [0.515, 0.658] | 0.068 | 0.372 |
| 12 | history.glr | 0.604 | [0.512, 0.656] | 0.062 | 0.351 |
| 12 | residual.page0.5 | 0.551 | [0.498, 0.609] | 0.066 | 0.126 |

条件校准 history Page-CUSUM 相对旧 pooled CUSUM 的 AUC 增量为 q7 0.017、q12 0.003；但相对 phase-only 仅为 q7 0.001、q12 0.001。

结论：旧 CUSUM 的 pooled calibration 和 uniform-score kappa 确实不理想；改成 task+position 条件 CDF、normal-score Page-CUSUM 后有小幅提升。但提升几乎都能由 phase-only 得到，不是健康句法残差贡献。q3 接近随机，q7 只能弱预测，q12 才有中等信号。

五折 calibration 最优候选分别为：['phase.page0', 'order_residual.page1', 'residual.page0', 'bag.window4', 'history.page0']。选择器跨 fold 不稳定，不建议在线部署时搜索后取最好。

## 边界

- 这里的阳性是最终 episode failure；新数据没有独立物理 failure-onset 标签。所以只能说 q7/q12 预测最终失败，不能证明一定早于错误动作。
- q12 只包含长度大于 12 的 17,992 条成功轨迹，但包含全部 1,096 条失败轨迹；它是 survivor-conditioned 晚期读数，不能与 q7 的总体覆盖直接比较。
- nominal 5% 阈值在 unseen states 上通常漂移到更高 FPR，说明上线前还需要更保守的 state-blocked/conformal 阈值校准。
- 连续 NLL bits 与旧报告的硬 token bits 不是同一绝对标尺；这里只在同一 full-40 protocol 内比较差值和学习曲线。
