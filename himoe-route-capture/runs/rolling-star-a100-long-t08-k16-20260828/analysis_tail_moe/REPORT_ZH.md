# Rolling-star MoE 尾部信号复用实验

## 直接结论

一般成败上，MoE 没有可复用的折外增量。但在 q24-q31 预测最终停滞/循环时，加入 MoE 后相对同构运动/动作对照降低 Brier 0.0087（2.3%），4/4 worker 同方向，精确 p=0.0625。
这个增量只在最晚固定窗口出现：fixed_q0_q7=-0.0002 / fixed_q8_q15=-0.0011 / fixed_q16_q23=-0.0029 / fixed_q24_q31=+0.0087。因此它更像晚期 trap 条件特征，而不是从开局就存在的难度编码。
它还不是独立报警器：合并模型仍比常数发生率基线多 0.0299 Brier；在 K=16 选枝中，它把避开 trap 从 76.5% 提到 82.4%，实际只多纠正 1/17 个混合 snapshot。
固定窗口的判断优先于相对尾段，因为前者在所有分支仍存活时读取，不知道终止时间。相对尾段只说明失败尾部存在可描述的 MoE 状态，不能单独证明可在线使用。

## 数据与口径

- 352 条 K=16 分支，235 条最终失败，199 条最终带停滞或循环标签。
- 四个固定窗口依次为 q0-7、q8-15、q16-23、q24-31；最短轨迹有 33 个 query，因此都没有生存者筛选。
- `relative_tail_50_90`：各轨迹自身 50% 到 90% 的 10 个锚点，复现旧尾段方法；它使用了最终长度，只作离线描述。
- `relative_tail_minus_mid`：尾段特征减去 10% 到 50% 中段特征，检查变化而非绝对阶段。
- MoE 特征覆盖前层 2-5、后层 12-15、状态与动作路由 d0-d9、置信度、状态/动作差异、去噪变化、query 持久与非局部复现。
- 物理对照排除了 MuJoCo time；模型还控制 flow noise 和完整动作 chunk。没有 AUC、剩余时间、终止标记或隐藏层。
- 状态 token 跨去噪步最大概率差为 0.0234；它在新数据上不是严格常数，因此状态 d0-d9 已分别扫描。
- 主要预测器照旧方法在每个训练折内先从每个特征族选最强坐标，再限制为最多 4 个；测试 worker 不参与选择。

## 固定 query 的 worker 外推

以下是 K=16 snapshot 内中心化后，留一整个 worker/初态外推的结果。Brier 越低越好；`gain` 为对照误差减模型误差，正数才是改进。

| target | model | control_reference | brier | brier_gain_vs_rate | brier_gain_vs_controls | brier_gain_vs_controls_ci95_low | brier_gain_vs_controls_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eventual_failure | rate_only | controls | 0.3313 | 0.0000 | 0.0079 | -0.0335 | 0.0458 |
| eventual_failure | controls_sparse_base | controls | 0.3377 | -0.0064 | 0.0014 | 0.0006 | 0.0020 |
| eventual_failure | moe_sparse_trap | controls | 0.3433 | -0.0120 | -0.0041 | -0.0211 | 0.0129 |
| eventual_failure | moe_sparse_full | controls | 0.3534 | -0.0221 | -0.0142 | -0.0367 | 0.0083 |
| eventual_failure | controls+moe_sparse | controls_sparse_base | 0.3448 | -0.0136 | -0.0071 | -0.0085 | -0.0052 |
| eventual_stagnation_or_loop | rate_only | controls | 0.3406 | 0.0000 | 0.0375 | -0.0368 | 0.1220 |
| eventual_stagnation_or_loop | controls_sparse_base | controls | 0.3792 | -0.0386 | -0.0011 | -0.0051 | 0.0018 |
| eventual_stagnation_or_loop | moe_sparse_trap | controls | 0.3520 | -0.0114 | 0.0261 | -0.0294 | 0.1046 |
| eventual_stagnation_or_loop | moe_sparse_full | controls | 0.3511 | -0.0105 | 0.0270 | -0.0272 | 0.1053 |
| eventual_stagnation_or_loop | controls+moe_sparse | controls_sparse_base | 0.3705 | -0.0299 | 0.0087 | 0.0048 | 0.0144 |

训练折内四特征 MoE 加到同构对照模型后的严格判断：

| target | controls_brier | controls_plus_moe_brier | incremental_moe_gain | worker_gain_min | worker_gain_max | workers_positive | exact_worker_sign_flip_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eventual_failure | 0.3377 | 0.3448 | -0.0071 | -0.0088 | -0.0043 | 0 | 1.0000 |
| eventual_stagnation_or_loop | 0.3792 | 0.3705 | 0.0087 | 0.0045 | 0.0170 | 4 | 0.0625 |

只有 4 个 worker，单侧精确符号翻转检验的最小可能 p 值是 0.0625；因此即使 4/4 同方向也只能作为下一轮候选，不能写成确认性结论。

## 固定前缀走势

四个窗口都早于最短轨迹终止。若信号只在后段形成，MoE 相对同构对照的增量应当随窗口推进而增强；若来回跳动，则更像阶段/初态依赖。

| window | target | control_reference | brier | brier_gain_vs_rate | brier_gain_vs_controls | brier_gain_vs_controls_ci95_low | brier_gain_vs_controls_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_q0_q7 | eventual_failure | controls_sparse_base | 0.3605 | -0.0293 | 0.0010 | -0.0040 | 0.0070 |
| fixed_q0_q7 | eventual_stagnation_or_loop | controls_sparse_base | 0.3758 | -0.0352 | -0.0002 | -0.0021 | 0.0024 |
| fixed_q8_q15 | eventual_failure | controls_sparse_base | 0.3823 | -0.0511 | 0.0037 | -0.0054 | 0.0142 |
| fixed_q8_q15 | eventual_stagnation_or_loop | controls_sparse_base | 0.3978 | -0.0572 | -0.0011 | -0.0074 | 0.0042 |
| fixed_q16_q23 | eventual_failure | controls_sparse_base | 0.3707 | -0.0395 | 0.0015 | -0.0023 | 0.0054 |
| fixed_q16_q23 | eventual_stagnation_or_loop | controls_sparse_base | 0.3943 | -0.0537 | -0.0029 | -0.0053 | -0.0006 |
| fixed_q24_q31 | eventual_failure | controls_sparse_base | 0.3448 | -0.0136 | -0.0071 | -0.0085 | -0.0052 |
| fixed_q24_q31 | eventual_stagnation_or_loop | controls_sparse_base | 0.3705 | -0.0299 | 0.0087 | 0.0048 | 0.0144 |

## K=16 选枝检查

在每个同时含正负分支的 snapshot 内，选择预测风险最低的一支。它检验实际选枝价值，不是 AUC。

| target | model | mixed_snapshots | selected_avoidance_rate | random_avoidance_rate | selection_gain_vs_random | selection_gain_vs_random_ci95_low | selection_gain_vs_random_ci95_high | snapshots_improved_vs_controls | snapshots_worse_vs_controls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eventual_failure | controls_sparse_base | 15 | 0.6667 | 0.4875 | 0.1792 | 0.0208 | 0.3500 | 0 | 0 |
| eventual_failure | moe_sparse_full | 15 | 0.6000 | 0.4875 | 0.1125 | -0.0375 | 0.2792 | 1 | 2 |
| eventual_failure | controls+moe_sparse | 15 | 0.7333 | 0.4875 | 0.2458 | 0.0875 | 0.4208 | 1 | 0 |
| eventual_stagnation_or_loop | controls_sparse_base | 17 | 0.7647 | 0.4449 | 0.3199 | 0.1176 | 0.5221 | 0 | 0 |
| eventual_stagnation_or_loop | moe_sparse_full | 17 | 0.4118 | 0.4449 | -0.0331 | -0.2059 | 0.1581 | 0 | 6 |
| eventual_stagnation_or_loop | controls+moe_sparse | 17 | 0.8235 | 0.4449 | 0.3787 | 0.1948 | 0.5588 | 1 | 0 |

## 离线尾段复现

| window | model | control_reference | brier | brier_gain_vs_rate | brier_gain_vs_controls |
| --- | --- | --- | --- | --- | --- |
| relative_tail_50_90 | controls_sparse_base | controls | 0.3409 | -0.0096 | 0.0007 |
| relative_tail_50_90 | moe_sparse_full | controls | 0.3151 | 0.0162 | 0.0266 |
| relative_tail_50_90 | controls+moe_sparse | controls_sparse_base | 0.3403 | -0.0090 | 0.0006 |
| relative_tail_minus_mid | controls_sparse_base | controls | 0.3432 | -0.0119 | -0.0001 |
| relative_tail_minus_mid | moe_sparse_full | controls | 0.3394 | -0.0081 | 0.0037 |
| relative_tail_minus_mid | controls+moe_sparse | controls_sparse_base | 0.3453 | -0.0140 | -0.0021 |

## 最强单特征

效应是同一 snapshot 内的标准化正负差；正值表示目标标签更高。`max_global` 已校正单个窗口内的整套路由扫描；窗口间比较仍是探索性的。

| window | target | family | feature | effect_sigma | effect_ci95_low | effect_ci95_high | permutation_p_max_global | workers_same_direction | workers_estimable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_q0_q7 | eventual_failure | action_confidence | action_top1\|front_2_5\|d2\|phase_late | 0.8171 | 0.4886 | 1.1560 | 0.0286 | 3 | 3 |
| fixed_q0_q7 | eventual_failure | action_confidence | action_entropy\|back_12_15\|d6\|phase_mean | -0.7764 | -1.1848 | -0.2919 | 0.0622 | 2 | 3 |
| fixed_q0_q7 | eventual_failure | action_confidence | action_top1\|front_2_5\|d1\|phase_late | 0.7603 | 0.3815 | 1.1613 | 0.0844 | 3 | 3 |
| fixed_q0_q7 | eventual_stagnation_or_loop | action_confidence | action_entropy\|back_12_15\|d6\|phase_mean | -0.9100 | -1.3052 | -0.4738 | 0.0002 | 4 | 4 |
| fixed_q0_q7 | eventual_stagnation_or_loop | action_confidence | action_entropy\|back_12_15\|d5\|phase_mean | -0.9041 | -1.2660 | -0.4724 | 0.0002 | 4 | 4 |
| fixed_q0_q7 | eventual_stagnation_or_loop | action_confidence | action_entropy\|back_12_15\|d4\|phase_mean | -0.8799 | -1.2364 | -0.4662 | 0.0004 | 3 | 4 |
| fixed_q16_q23 | eventual_failure | denoise_soft_motion | action_denoise_motion\|front_2_5\|d8_to_d9\|phase_mean | 1.1018 | 0.6637 | 1.5199 | 0.0002 | 3 | 3 |
| fixed_q16_q23 | eventual_failure | denoise_soft_motion | action_denoise_motion\|back_12_15\|d1_to_d2\|phase_mean | 0.7366 | 0.2060 | 1.3827 | 0.0788 | 3 | 3 |
| fixed_q16_q23 | eventual_failure | denoise_soft_motion | action_denoise_motion\|front_2_5\|d5_to_d6\|phase_mean | 0.7306 | 0.1877 | 1.3438 | 0.0842 | 2 | 3 |
| fixed_q16_q23 | eventual_stagnation_or_loop | denoise_soft_motion | action_denoise_motion\|back_12_15\|d1_to_d2\|phase_mean | 1.1750 | 0.8574 | 1.5680 | 0.0002 | 4 | 4 |
| fixed_q16_q23 | eventual_stagnation_or_loop | denoise_soft_motion | action_denoise_motion\|front_2_5\|d8_to_d9\|phase_mean | 1.1717 | 0.7174 | 1.5499 | 0.0002 | 4 | 4 |
| fixed_q16_q23 | eventual_stagnation_or_loop | denoise_soft_motion | action_denoise_motion\|back_12_15\|d4_to_d5\|phase_mean | 1.0711 | 0.6603 | 1.4879 | 0.0002 | 3 | 4 |
| fixed_q24_q31 | eventual_failure | query_persistence | hard_route\|action\|back_12_15\|d2\|adjacent_mean | -1.4780 | -1.9639 | -0.8966 | 0.0002 | 2 | 3 |
| fixed_q24_q31 | eventual_failure | query_persistence | hard_route\|action\|back_12_15\|d1\|adjacent_mean | -1.4357 | -1.9715 | -0.8162 | 0.0002 | 2 | 3 |
| fixed_q24_q31 | eventual_failure | query_persistence | hard_route\|action\|back_12_15\|d0\|adjacent_mean | -1.4261 | -1.9157 | -0.8448 | 0.0002 | 2 | 3 |
| fixed_q24_q31 | eventual_stagnation_or_loop | action_confidence | action_entropy\|front_2_5\|d9\|phase_mean | -1.0878 | -1.3489 | -0.8448 | 0.0002 | 4 | 4 |
| fixed_q24_q31 | eventual_stagnation_or_loop | action_confidence | action_top4\|front_2_5\|d7\|phase_mean | 1.0558 | 0.7930 | 1.3743 | 0.0002 | 4 | 4 |
| fixed_q24_q31 | eventual_stagnation_or_loop | action_confidence | action_top4\|front_2_5\|d8\|phase_mean | 1.0363 | 0.7608 | 1.3408 | 0.0002 | 4 | 4 |
| fixed_q8_q15 | eventual_failure | denoise_soft_motion | action_denoise_motion\|front_2_5\|d8_to_d9\|phase_std | -0.7061 | -1.2277 | -0.2914 | 0.1564 | 3 | 3 |
| fixed_q8_q15 | eventual_failure | token_disagreement | action_token_dispersion\|back_12_15\|d4\|phase_mean | 0.6681 | 0.2543 | 1.0077 | 0.2899 | 2 | 3 |
| fixed_q8_q15 | eventual_failure | token_disagreement | action_token_dispersion\|back_12_15\|d8\|phase_mean | 0.6667 | 0.2611 | 1.0310 | 0.2943 | 2 | 3 |
| fixed_q8_q15 | eventual_stagnation_or_loop | token_disagreement | action_token_dispersion\|back_12_15\|d4\|phase_mean | 0.8778 | 0.3265 | 1.3425 | 0.0004 | 3 | 4 |
| fixed_q8_q15 | eventual_stagnation_or_loop | token_disagreement | action_token_dispersion\|back_12_15\|d3\|phase_mean | 0.8657 | 0.2871 | 1.3444 | 0.0004 | 3 | 4 |
| fixed_q8_q15 | eventual_stagnation_or_loop | token_disagreement | action_token_dispersion\|back_12_15\|d2\|phase_mean | 0.8544 | 0.2497 | 1.3487 | 0.0008 | 2 | 4 |
| relative_tail_50_90 | eventual_failure | action_confidence | action_entropy\|front_2_5\|d9\|phase_mean | -2.7195 | -3.7867 | -2.0167 | 0.0002 | 3 | 3 |
| relative_tail_50_90 | eventual_failure | action_confidence | action_top4\|front_2_5\|d9\|phase_mean | 2.5064 | 1.7800 | 3.5470 | 0.0002 | 3 | 3 |
| relative_tail_50_90 | eventual_failure | action_confidence | action_top4\|front_2_5\|d8\|phase_mean | 2.4409 | 1.7120 | 3.5847 | 0.0002 | 3 | 3 |
| relative_tail_50_90 | eventual_stagnation_or_loop | state_confidence | state_entropy\|back_12_15\|d9\|late_minus_early | -1.4674 | -1.9114 | -1.1187 | 0.0002 | 3 | 4 |
| relative_tail_50_90 | eventual_stagnation_or_loop | state_confidence | state_entropy\|back_12_15\|d7\|late_minus_early | -1.4660 | -1.9117 | -1.1161 | 0.0002 | 3 | 4 |
| relative_tail_50_90 | eventual_stagnation_or_loop | state_confidence | state_entropy\|back_12_15\|d6\|late_minus_early | -1.4659 | -1.9101 | -1.1171 | 0.0002 | 3 | 4 |
| relative_tail_minus_mid | eventual_failure | action_confidence | action_top4\|front_2_5\|d8\|phase_late | 2.4094 | 1.7690 | 3.3273 | 0.0002 | 3 | 3 |
| relative_tail_minus_mid | eventual_failure | action_confidence | action_entropy\|front_2_5\|d9\|phase_late | -2.3470 | -3.3064 | -1.6680 | 0.0002 | 3 | 3 |
| relative_tail_minus_mid | eventual_failure | action_confidence | action_top1\|front_2_5\|d8\|phase_late | 2.3414 | 1.7293 | 3.2192 | 0.0002 | 3 | 3 |
| relative_tail_minus_mid | eventual_stagnation_or_loop | state_action_mismatch | state_action_gap\|back_12_15\|d9\|phase_mean | -1.0994 | -1.5969 | -0.6857 | 0.0002 | 4 | 4 |
| relative_tail_minus_mid | eventual_stagnation_or_loop | state_action_mismatch | state_action_gap\|back_12_15\|d8\|phase_mean | -1.0817 | -1.5977 | -0.6651 | 0.0002 | 4 | 4 |
| relative_tail_minus_mid | eventual_stagnation_or_loop | state_action_mismatch | state_action_gap\|back_12_15\|d7\|phase_mean | -1.0279 | -1.5420 | -0.6218 | 0.0002 | 4 | 4 |

## 解释边界

- 这是现有轨迹的观察性复用实验，没有真的触发重采样或恢复动作。
- 固定 q24-q31 可以支持前缀预测判断，但目标仍是整条轨迹事后得到的物理标签，不等同于已定位 trap 的首次发生时刻。
- 真正的部署证据仍需在固定 query 上冻结检测器，再随机比较继续原策略与恢复策略的成功率。

完整数值见 `model_metrics.csv`、`worker_metrics.csv`、`selection_metrics.csv`、`selection_choices.csv`、`sparse_feature_selections.csv`、`route_feature_effects.csv` 和 `summary.json`。
