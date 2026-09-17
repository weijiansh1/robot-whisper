# P1a 实验结果

本轮为固定观测推理诊断，未执行环境动作或 gate 干预，不能报告救回率。

- 父轨迹：拟合 14 条，预留验证 6 条；验证基础任务：[2, 5, 8]。
- 状态 39 个；模型调用 383 次；同状态重复 39 对；capture 对照 16 对。
- 训练使用动作差异标签，不使用成功标签；基座未训练、共享服务未修改。

| 诊断表示 | 预留 RMSE | 预留 MAE |
|---|---:|---:|
| noise | 0.018299 | 0.014054 |
| full_moe | 0.017703 | 0.013733 |
| old_center | 0.017394 | 0.013796 |
| structured_moe | 0.018501 | 0.013844 |
| noise_plus_moe | 0.018501 | 0.013844 |

结构化纯 MoE 相对纯噪声 RMSE 改善 -1.10%；2/3 预留任务 MAE 改善。
状态块置换路由的 RMSE 中位数 0.024663。候选动作差异门槛覆盖 100.0% 预留状态。

## 预先冻结的继续门槛

- integrity_and_coverage：通过
- rmse_improvement_at_least_10pct：未通过
- at_least_two_holdout_tasks_improve：通过
- better_than_median_shuffled_routes：通过
- at_least_half_states_have_action_delta_over_002：通过

阶段判定：停止本协议的扩大实验，不在该验证集重新挑参数。

## 解释限制

候选对、同父轨迹的查询及两个 benchmark 的同基础任务均有关联。这里只有 3 个预留基础任务，不声明统计显著性或广泛泛化。
q20 仅包含尚未结束的源轨迹。观测被冻结，未执行候选后的新观测，因此路由与动作的相关响应不证明路由对动作的因果作用，也不证明物理可恢复性。
原路由中心和噪声中心的比较仅是选中索引诊断，没有候选续跑结局。

配置、数据来源和哈希见 `config.json`；采集完整性见 `route-audit.json`；完整逐状态结果见 `analysis.json`。
