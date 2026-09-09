# Chunk 逐 token 路由是否含有成败信息

## 结论

**有一个局部 q0 候选，但广义 chunk 结构还不能稳定预测单条分支成败。**
上一轮冻结的 U 形、前后层差、去噪稳定化等 17 个结构分数没有一个通过全局 0.05；最强 `q0 soft-tail` 只有 `p=0.0970`。
全位置扫描找到 `q0|back_12_15|entropy_centered|d9|t7`，效应 -0.566 SD、3,760 格 max-T `p=0.0242`。训练折内重新选点后，q0 cell 相对初态先验的 Brier gain 为 +0.00234 [-0.00114,+0.00589]；相对错配 sibling 为 +0.00236 [-0.00103,+0.00596]。
预定义 q0 总结构相对初态先验只有 +0.00164 [-0.00302,+0.00619]；在未见初态上趋势为 +0.03079 [-0.00907,+0.07463]。

本报告区分统计差异、冻结复验和折外预测。所有主检验只使用 q0-q3；没有使用 rollout
长度、remaining time、terminal window、动作、hidden state 或 simulator state。

## 主数据与正负对照

- Long 共 512 条 rollout、216 失败、296 成功；16 个初态中 13 个同时有成败。
- 同初态主检验使用 416 条、13 个 mixed 初态。统计单位是初态，标签只在初态内置换。
- heldout-seed 预测把 32 个 noise seeds 分成 8 折；测试 seed 在训练中从未出现，但训练折可估计初态先验。
- heldout-init 完整留出两个初态，检验结构能否迁移到未见场景。
- 正对照是 heldout-seed 的 initial-state prior；负对照是初态内标签置换的 max-T 零分布。
- 第二个负对照把每条分支的结构替换成同初态下下一个 noise seed 的 sibling 路由；它保留初态，破坏 branch 对齐。

## 无监督发现的结构分数

下面这些分数由上一轮不看标签的 token 曲线定义。正 effect 表示失败分支的该结构更强。

| score | effect (SD) | 95% CI | raw p | global max-T p |
|---|---:|---:|---:|---:|
| `q0_q3_mean|entropy_u_shape` | +0.059 | [-0.183,+0.347] | 0.6456 | 0.9998 |
| `q0_q3_mean|soft_change_mean` | +0.035 | [-0.171,+0.279] | 0.7825 | 1.0000 |
| `q0_q3_mean|soft_early_minus_late` | +0.020 | [-0.223,+0.284] | 0.8741 | 1.0000 |
| `q0_q3_mean|soft_edge_minus_middle` | -0.069 | [-0.294,+0.134] | 0.5902 | 0.9998 |
| `q0_q3_mean|soft_front_minus_back` | -0.042 | [-0.240,+0.190] | 0.7386 | 1.0000 |
| `q0_q3_mean|top1_switch_mean` | -0.068 | [-0.380,+0.234] | 0.5995 | 0.9998 |
| `q0_q3_mean|top1_u_shape` | +0.049 | [-0.149,+0.255] | 0.6920 | 1.0000 |
| `q0_q3_mean|top4_turnover_mean` | -0.045 | [-0.263,+0.204] | 0.7215 | 1.0000 |
| `q0|entropy_u_shape` | +0.167 | [-0.131,+0.404] | 0.1882 | 0.8490 |
| `q0|soft_change_mean` | -0.168 | [-0.400,+0.062] | 0.1835 | 0.8441 |
| `q0|soft_early_minus_late` | -0.076 | [-0.248,+0.138] | 0.5382 | 0.9996 |
| `q0|soft_edge_minus_middle` | +0.076 | [-0.248,+0.348] | 0.5420 | 0.9995 |
| `q0|soft_front_minus_back` | -0.190 | [-0.406,+0.001] | 0.1358 | 0.7304 |
| `q0|soft_tail_d5_back` | +0.332 | [+0.069,+0.633] | 0.0096 | 0.0970 |
| `q0|top1_switch_mean` | -0.172 | [-0.543,+0.161] | 0.1657 | 0.8256 |
| `q0|top1_u_shape` | +0.128 | [-0.132,+0.320] | 0.3091 | 0.9687 |
| `q0|top4_turnover_mean` | -0.232 | [-0.497,-0.001] | 0.0651 | 0.4839 |

绝对效应最大的是 `q0|soft_tail_d5_back`：+0.332 SD，17 分数全局 max-T `p=0.0970`。

## 逐位置全扫描与冻结复验

全 416 条 mixed-state 分支扫描 3,760 个 cell，最大为 `q0|back_12_15|entropy_centered|d9|t7`：-0.566 SD，全局 max-T `p=0.0242`。
前 16 seeds 的 discovery 最大 cell 是 `q0|back_12_15|entropy_centered|d2|t6`：+0.687 SD，全局 `p=0.1420`。
冻结结果：

| dataset | cell | effect (SD) | raw p | global p |
|---|---|---:|---:|---:|
| long_seed_A_discovery | `q0|back_12_15|entropy_centered|d2|t6` | +0.687 | 0.0002 | 0.1420 |
| long_seed_B_frozen | `q0|back_12_15|entropy_centered|d2|t6` | +0.090 | 0.6493 | 0.6493 |
| gos_frozen_discovery_cell | `q0|back_12_15|entropy_centered|d2|t6` | +0.113 | 0.5083 | 0.5083 |
| gos_frozen_long_full_winner | `q0|back_12_15|entropy_centered|d9|t7` | +0.123 | 0.3906 | 0.3906 |

全样本赢家 `q0|back_12_15|entropy_centered|d9|t7` 在 seed A/B 的描述性效应分别为 -0.364/-0.831 SD，13 个 mixed 初态中 12/13 与总体同方向。这个拆半发生在全样本选点之后，所以是稳定性诊断，不是独立冻结检验。G/O/S 对该 Long 赢家另作独立 task 内方向检查。
heldout-seed 每折训练内的 q0 选点频次为：`q0|back_12_15|entropy_centered|d9|t7` 4/8; `q0|back_12_15|entropy_centered|d8|t7` 3/8; `q0|back_12_15|entropy_centered|d2|t6` 1/8。
但 3,760 个 cell 的 seed A/B 效应排名相关只有 `rho=+0.067`，说明除这个局部簇外，整张效应图并不稳定。

## 折外预测：结构有没有实际增量

Brier gain 为正才有用。`state_vs_structure_q0` 是第一次推理的逐 token 结构相对初态先验；
`aggregate_*_structure_increment` 检验逐 token 位置是否超过旧的 token-平均路由量。
`shifted_sibling_vs_aligned_*` 为正才表示真实 branch 路由优于同初态错配路由。
`nested_best_*_cell` 的位置选择完全发生在训练折内，因此是对 3,760 格探索峰的无泄漏预测检查。

| split | comparison | Brier gain | 95% state-bootstrap CI | positive states |
|---|---|---:|---:|---:|
| heldout_init | `state_vs_structure_q0` | +0.03079 | [-0.00907,+0.07463] | 10/16 |
| heldout_init | `state_vs_structure_q0_q3` | -0.02698 | [-0.09269,+0.03684] | 7/16 |
| heldout_init | `aggregate_q0_structure_increment` | -0.00907 | [-0.02301,+0.00410] | 7/16 |
| heldout_init | `aggregate_q0_q3_structure_increment` | -0.05501 | [-0.11009,-0.00995] | 5/16 |
| heldout_init | `q1_q3_increment_over_q0_structure` | -0.05777 | [-0.10716,-0.01827] | 3/16 |
| heldout_init | `initial_state_prior_vs_global` | +0.00000 | [+0.00000,+0.00000] | 0/16 |
| heldout_init | `shifted_sibling_vs_aligned_q0` | +0.00675 | [-0.00276,+0.01761] | 9/16 |
| heldout_init | `shifted_sibling_vs_aligned_q0_q3` | -0.00425 | [-0.02396,+0.01399] | 8/16 |
| heldout_init | `state_vs_nested_best_q0_cell` | +0.00779 | [-0.03105,+0.03590] | 12/16 |
| heldout_init | `state_vs_nested_best_q0_q3_cell` | +0.01662 | [-0.00028,+0.03283] | 12/16 |
| heldout_init | `nested_q1_q3_increment_over_q0` | +0.00883 | [-0.01169,+0.03818] | 1/16 |
| heldout_init | `shifted_sibling_vs_aligned_nested_q0` | +0.01183 | [-0.02750,+0.04027] | 11/16 |
| heldout_init | `shifted_sibling_vs_aligned_nested_q0_q3` | +0.02094 | [+0.00428,+0.03716] | 10/16 |
| heldout_seed | `state_vs_structure_q0` | +0.00164 | [-0.00302,+0.00619] | 9/16 |
| heldout_seed | `state_vs_structure_q0_q3` | -0.00161 | [-0.00860,+0.00476] | 10/16 |
| heldout_seed | `aggregate_q0_structure_increment` | +0.00054 | [-0.00418,+0.00544] | 7/16 |
| heldout_seed | `aggregate_q0_q3_structure_increment` | -0.00551 | [-0.01013,-0.00049] | 2/16 |
| heldout_seed | `q1_q3_increment_over_q0_structure` | -0.00325 | [-0.00717,+0.00046] | 4/16 |
| heldout_seed | `initial_state_prior_vs_global` | +0.10935 | [+0.07345,+0.14963] | 16/16 |
| heldout_seed | `shifted_sibling_vs_aligned_q0` | +0.00146 | [-0.00288,+0.00606] | 9/16 |
| heldout_seed | `shifted_sibling_vs_aligned_q0_q3` | +0.00276 | [-0.00151,+0.00698] | 9/16 |
| heldout_seed | `state_vs_nested_best_q0_cell` | +0.00234 | [-0.00114,+0.00589] | 11/16 |
| heldout_seed | `state_vs_nested_best_q0_q3_cell` | +0.00234 | [-0.00126,+0.00588] | 11/16 |
| heldout_seed | `nested_q1_q3_increment_over_q0` | +0.00000 | [+0.00000,+0.00000] | 0/16 |
| heldout_seed | `shifted_sibling_vs_aligned_nested_q0` | +0.00236 | [-0.00103,+0.00596] | 9/16 |
| heldout_seed | `shifted_sibling_vs_aligned_nested_q0_q3` | +0.00602 | [+0.00149,+0.01091] | 12/16 |

模型的绝对折外 Brier：

| split | model | Brier |
|---|---|---:|
| heldout_init | `structure_q0` | 0.22872 |
| heldout_init | `aggregate_q0` | 0.23502 |
| heldout_init | `structure_q0_shifted_sibling` | 0.23547 |
| heldout_init | `nested_best_q0_q3_cell` | 0.24290 |
| heldout_init | `aggregate_q0_plus_structure` | 0.24409 |
| heldout_init | `nested_best_q0_cell` | 0.25173 |
| heldout_init | `global_prior` | 0.25952 |
| heldout_init | `state_prior` | 0.25952 |
| heldout_init | `nested_best_q0_cell_shifted_sibling` | 0.26356 |
| heldout_init | `nested_best_q0_q3_cell_shifted_sibling` | 0.26383 |
| heldout_init | `structure_q0_q3_shifted_sibling` | 0.28224 |
| heldout_init | `structure_q0_q3` | 0.28650 |
| heldout_init | `aggregate_q0_q3` | 0.29078 |
| heldout_init | `aggregate_q0_q3_plus_structure` | 0.34580 |
| heldout_seed | `nested_best_q0_cell` | 0.13260 |
| heldout_seed | `nested_best_q0_q3_cell` | 0.13260 |
| heldout_seed | `structure_q0` | 0.13330 |
| heldout_seed | `structure_q0_shifted_sibling` | 0.13476 |
| heldout_seed | `state_prior` | 0.13494 |
| heldout_seed | `nested_best_q0_cell_shifted_sibling` | 0.13496 |
| heldout_seed | `structure_q0_q3` | 0.13655 |
| heldout_seed | `aggregate_q0_q3` | 0.13681 |
| heldout_seed | `aggregate_q0_plus_structure` | 0.13859 |
| heldout_seed | `nested_best_q0_q3_cell_shifted_sibling` | 0.13862 |
| heldout_seed | `aggregate_q0` | 0.13913 |
| heldout_seed | `structure_q0_q3_shifted_sibling` | 0.13932 |
| heldout_seed | `aggregate_q0_q3_plus_structure` | 0.14232 |
| heldout_seed | `global_prior` | 0.24429 |

## Goal/Object/Spatial 外部方向检查

G/O/S 有 850 条来自 mixed tasks 的 rollout、55 次失败。这里只有 task 固定效应，不能控制初态。
17 个结构效应与 Long 的 Spearman `rho=+0.505`。不同 checkpoint、低失败率和单次初态意味着这只能检查方向，不能替代 Long 的 sibling 正负对照。
更细的 3,760-cell Long/GOS 效应相关只有 `rho=+0.031`，且 Long 赢家在 G/O/S 中方向相反、`p=0.3906`，所以目前应把它视为 Long 特定候选。

## 解释边界

即使某个 q1-q3 结构能预测，它也已经包含前几个 action chunk 执行后的状态分叉；只有 q0
可以称为执行前信号。binary failure 在这个 Long 数据里等同最终未成功并跑满时限，但本分析
没有把时长输入模型。高 hard-expert switch 仍可能来自近似并列专家换序，因此主解释优先看
soft-route Hellinger 和折外 Brier，而不是专家 ID。
