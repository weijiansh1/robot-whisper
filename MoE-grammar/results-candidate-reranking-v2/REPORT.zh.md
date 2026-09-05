# 第三阶段：健康语法候选重排审计

## 结论

主分析包含 8 条独立失败 trunk、28 个同状态 fork snapshot、112 个候选；其中只有 4 个 snapshot 同时含成功和失败候选。

full-prefix grammar 选择成功率为 10.7%，随机选择期望为 5.4%，action medoid 为 7.1%，best-of-K oracle 为 14.3%。

grammar 相对随机的绝对变化为 5.4%，trunk-cluster bootstrap 95% CI [-1.9%, 13.3%]；在真正有选择机会的 snapshot 上命中成功候选的比例为 75.0%。

候选内（只比较同一 snapshot 的成功/失败对）grammar AUC 为 0.786，95% CI [0.667, 1.000]。

但 full-prefix / clock-only / last-query-only / no-prefix AUC 分别为 0.786 / 0.786 / 0.786 / 0.786。

真正提供成功/失败配对的独立 trunk 只有 3 条；以 trunk 为单位的单边精确 sign-flip p=0.125。

因此当前正向信号属于候选当前 query 的健康 emission compatibility；没有证据表明完整前缀比无前缀或内部时钟提供了额外的候选级排序信息。

## 实现与防泄漏

- 每个候选只使用共享 trunk 的全部 MoE 前缀和候选当前 query 的 22 维多轨道表型。
- HMM forward belief 压缩截止当前的完整前缀；候选由同一个冻结 belief 并行打分，互不更新。
- 每个 init state 使用将该状态置于测试集的健康语法 fold；训练只含成功 episode。
- success 只在选择完成后用于评估；oracle 是唯一读取 outcome 的上界。
- 所有候选的 `sim_state[0]` 与 `policy_state[0]` 均逐位一致；route IDs 逐 query 回查服务端 Zarr，再读取完整 32-way gate 概率。
- 置信区间按 trunk 重抽，而非把同一 trunk 的多个 fork 点伪装成独立样本。

## 选择器

| selector | success | uplift vs random | opportunity hit | oracle gap |
|---|---:|---:|---:|---:|
| random | 5.4% | 0.0% | 37.5% | 0.0% |
| action_medoid | 7.1% | 1.8% | 50.0% | 20.0% |
| grammar | 10.7% | 5.4% | 75.0% | 60.0% |
| grammar_action_diverse | 10.7% | 5.4% | 75.0% | 60.0% |
| grammar_route_diverse | 10.7% | 5.4% | 75.0% | 60.0% |
| anti_grammar | 3.6% | -1.8% | 25.0% | -20.0% |
| grammar_clock | 10.7% | 5.4% | 75.0% | 60.0% |
| grammar_local1 | 7.1% | 1.8% | 50.0% | 20.0% |
| grammar_no_prefix | 7.1% | 1.8% | 50.0% | 20.0% |
| oracle | 14.3% | 8.9% | 100.0% | 100.0% |

## 解释边界

这是候选级、同状态、严格因果输入的离线反事实审计，但样本量仍小。绝对成功率会受 fork 时机与 16-query 预算影响，因此核心证据是同 snapshot 的配对排序，不是跨 snapshot 的原始分数相关。

`k8_routed` 只有一条 trunk，只作为复现实例，不并入主置信区间。没有成功候选的 snapshot 可检验分数稳定性，但不能检验选择能力。

完整逐候选分数见 `candidate_scores.csv`，逐 snapshot 选择见 `snapshot_selections.csv`，机器可读汇总见 `summary.json`。
