# 三通道 MoE routing observer（v2）

## 核心结论

在 held-out 成功序列上，soft HMM 的完整前缀相对仅按内部时钟推进的 next-query 改善为 0.0129 bits/phenotype；连续 VAR(4) 相对无历史 Gaussian 的改善为 0.2174 bits/phenotype。

这只证明 routing history 可预测下一 query，不自动证明高 surprisal 等于 Trap。

健康 excursion 的局部 return 任务含 85148 个 held-out 异常前缀；三步不返回的 AUC 为 0.573，return-probability Brier 为 0.101。

Scene8 的物理 stasis 是当前唯一可靠 onset 锚点；没有可靠 loop onset，因此本报告不伪造 loop 结论。

## 物理 onset 结果（固定 5% success-episode FPR）

| channel | success episode FPR | recall q-3 | recall by onset | recall onset+3 | matched AUC (95% CI over events) |
|---|---:|---:|---:|---:|---:|
| current_innovation | 4.7% | 4.1% | 4.6% | 4.6% | 0.566 [0.517, 0.614] |
| var_innovation | 4.7% | 4.6% | 5.1% | 5.1% | 0.572 [0.526, 0.618] |
| hmm_innovation | 4.7% | 2.0% | 2.5% | 2.5% | 0.560 [0.514, 0.605] |
| innovation_persistent | 4.7% | 6.1% | 7.1% | 8.1% | 0.504 [0.473, 0.536] |
| overregularity | 4.7% | 4.1% | 4.1% | 4.6% | 0.501 [0.450, 0.552] |
| overregularity_persistent | 4.7% | 6.1% | 6.1% | 6.6% | 0.480 [0.458, 0.504] |
| return_failure | 4.7% | 6.6% | 6.6% | 7.6% | 0.512 [0.465, 0.561] |
| three_channel | 4.7% | 6.1% | 6.1% | 6.1% | 0.514 [0.467, 0.564] |

### Over-regularity components

| component | success episode FPR | recall q-3 | recall by onset | matched AUC (95% CI over events) |
|---|---:|---:|---:|---:|
| freeze | 4.7% | 4.6% | 5.1% | 0.502 [0.453, 0.552] |
| recurrence | 4.7% | 5.6% | 5.6% | 0.513 [0.461, 0.565] |
| low_belief_entropy | 0.0% | 0.0% | 0.0% | 0.522 [0.471, 0.573] |

所有分数均来自 init-state held-out 模型。这里仅用 pooled out-of-fold Scene8 success controls 读取 5% FPR operating point，再评估 stasis episodes；它用于公平比较通道，不冒充可部署阈值。

## Scene8 跨状态阈值部署

| channel | Scene8 success FPR | recall by onset | matched AUC (95% CI over events) |
|---|---:|---:|---:|
| current_innovation | 7.4% | 4.6% | 0.566 [0.517, 0.614] |
| var_innovation | 8.4% | 2.5% | 0.572 [0.526, 0.618] |
| hmm_innovation | 8.1% | 2.0% | 0.560 [0.514, 0.605] |
| innovation_persistent | 10.1% | 9.6% | 0.504 [0.473, 0.536] |
| overregularity | 5.1% | 4.1% | 0.501 [0.450, 0.552] |
| overregularity_persistent | 7.4% | 5.1% | 0.480 [0.458, 0.504] |
| return_failure | 5.7% | 4.6% | 0.512 [0.465, 0.561] |
| three_channel | 7.8% | 6.6% | 0.514 [0.467, 0.564] |

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
