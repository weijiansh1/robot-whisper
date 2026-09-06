# 三通道 MoE routing observer（v2）

## 核心结论

在 held-out 成功序列上，soft HMM 的完整前缀相对仅按内部时钟推进的 next-query 改善为 0.0130 bits/phenotype；连续 VAR(4) 相对无历史 Gaussian 的改善为 0.2174 bits/phenotype。

这只证明 routing history 可预测下一 query，不自动证明高 surprisal 等于 Trap。

健康 excursion 的局部 return 任务含 83235 个 held-out 异常前缀；三步不返回的 AUC 为 0.562，return-probability Brier 为 0.103。

Scene8 的物理 stasis 是当前唯一可靠 onset 锚点；没有可靠 loop onset，因此本报告不伪造 loop 结论。

## 物理 onset 结果（固定 5% success-episode FPR）

| channel | success episode FPR | recall q-3 | recall by onset | recall onset+3 | matched AUC (95% CI over events) |
|---|---:|---:|---:|---:|---:|
| current_innovation | 4.7% | 2.5% | 3.0% | 3.0% | 0.557 [0.511, 0.603] |
| var_innovation | 4.7% | 2.0% | 2.5% | 2.5% | 0.560 [0.515, 0.604] |
| hmm_innovation | 4.7% | 1.5% | 2.0% | 2.0% | 0.544 [0.497, 0.589] |
| innovation_persistent | 4.7% | 5.1% | 5.6% | 7.6% | 0.503 [0.478, 0.530] |
| overregularity | 4.7% | 4.1% | 4.1% | 4.6% | 0.561 [0.510, 0.615] |
| overregularity_persistent | 4.7% | 4.1% | 4.1% | 5.1% | 0.480 [0.457, 0.505] |
| return_failure | 4.7% | 2.5% | 2.5% | 3.6% | 0.533 [0.486, 0.581] |
| three_channel | 4.7% | 3.6% | 3.6% | 4.6% | 0.527 [0.481, 0.576] |

### Over-regularity components

| component | success episode FPR | recall q-3 | recall by onset | matched AUC (95% CI over events) |
|---|---:|---:|---:|---:|
| freeze | 4.4% | 5.1% | 6.1% | 0.501 [0.454, 0.550] |
| recurrence | 4.7% | 6.1% | 6.1% | 0.518 [0.465, 0.571] |
| low_belief_entropy | 0.0% | 0.0% | 0.0% | 0.598 [0.548, 0.647] |

所有分数均来自 init-state held-out 模型。这里仅用 pooled out-of-fold Scene8 success controls 读取 5% FPR operating point，再评估 stasis episodes；它用于公平比较通道，不冒充可部署阈值。

## Scene8 跨状态阈值部署

| channel | Scene8 success FPR | recall by onset | matched AUC (95% CI over events) |
|---|---:|---:|---:|
| current_innovation | 9.5% | 4.6% | 0.557 [0.511, 0.603] |
| var_innovation | 8.8% | 2.5% | 0.560 [0.515, 0.604] |
| hmm_innovation | 7.8% | 1.5% | 0.544 [0.497, 0.589] |
| innovation_persistent | 7.8% | 6.6% | 0.503 [0.478, 0.530] |
| overregularity | 5.1% | 4.6% | 0.561 [0.510, 0.615] |
| overregularity_persistent | 6.8% | 3.0% | 0.480 [0.457, 0.505] |
| return_failure | 6.4% | 2.5% | 0.533 [0.486, 0.581] |
| three_channel | 6.8% | 3.6% | 0.527 [0.481, 0.576] |

每折阈值由其他 Scene8 calibration states 的 success episodes 决定，再原样部署到 test states。实际 FPR 偏离 5% 表示 init-state threshold shift。

full40 阈值直接迁移到 Scene8 的完整诊断保存在 `summary.json` 的 `physical_stasis_global_threshold_transfer`；它用于度量更强的跨任务阈值漂移。

## 表征修正

- 删除跨层同编号 expert distribution 的直接比较。新量只比较每层置换不变的标量 profile。
- 新主模型使用 22 维 query phenotype，不经过 2187→PCA→hard GMM 链路。
- VAR/HMM 都是连续 emission；HMM 用 forward marginalization，不使用硬 word context。
- density episode 按 task 等量抽取，每 episode 均匀抽相同数量 phase positions。
- 在线输入不含 task ID、绝对 query index、物理状态或最终 outcome。

## 三个通道

- `hmm_innovation`：完整 routing prefix 下的上尾 surprisal。
- `overregularity`：lag-1 freeze、lag-2..4 recurrence、异常低 surprisal 中至少两项共同升高。
- `return_failure`：健康数据中相似 excursion 在未来 3 query 不返回的经验风险。
- `three_channel`：三个可解释风险通道的并集，不使用固定 drift 的单边 CUSUM。

## 边界

return 标签由健康 routing excursion 自身定义，用于验证局部序列目标；它不是新的物理 Trap 标签。物理有效性仍以独立 stasis onset 为准。若三通道不能在相同 episode FPR 下超过 current-query，就应把 grammar 降级为机制分析，而不是主 detector。
