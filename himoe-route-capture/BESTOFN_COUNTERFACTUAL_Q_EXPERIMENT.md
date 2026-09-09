# MoE-aware Best-of-N counterfactual-Q experiment

更新日期：2026-08-25

## 状态

实验协议、300 个 snapshot 的分层计划、候选/终局采集器、cross-fitted oracle 分析、四路 critic、排除实验和可恢复总控均已实现并冻结。源数据、checkpoint 和计划哈希预检通过。

正式采集尚未产生候选或 outcome。两张 H20 当前各只有约 7.1 GiB 空闲显存；冻结 HiMoE-VLA 的安全装载阈值为 24 GiB。总控已实际启动并以 `RESOURCE_UNAVAILABLE` 退出，run manifest 标记为 `waiting_for_gpu`，没有用旧候选或代理标签代替新实验。

Gate 1 的保守计算上界为 2,880 次 candidate query、243,552 次 R=4 continuation query 和约 243 万个 MuJoCo action step。按现有 rollout 约 1.5 秒/控制 query 的实测量级，单卡顺序运行的上界约 4 天；terminal early success 会降低实际时间。正式长跑前先完成一个 state 的可恢复实测，以校准吞吐和成功提前终止比例。

## 科学命题

主命题不是“routing 就是物理 Q”，而是：

> 在冻结 proposal policy、有限数据和部分可观测条件下，MoE 内部轨迹能否在 state 和最终 action chunk 之外，提高同 snapshot 候选的 terminal-Q 排序？

主比较为：

$$
\Delta_{\mathrm{MoE}}
=
\operatorname{Perf}(Q(s,A,R))
-
\operatorname{Perf}(Q(s,A)).
$$

## 新数据设计

- 任务：LIBERO-Goal task 0、1、3，各 100 个 snapshot。
- 分层：free-space、pre-contact、contact/manipulation、late/recovery，各任务每层 25 个。
- split：每层 15 train、5 validation、5 test；同一 parent episode 不跨 split。
- proposal：冻结 HiMoE-VLA，标准 flow noise，基础 $K=8$；每任务 20 个预先选定 snapshot 扩展至 $K=16$。
- 总候选数：2,880。
- 每个候选保存：完整 HB router probability、router input hidden、10-step flow trajectory、最终 $H=10$ action chunk、冻结 PaliGemma prefix feature、原始图像和 proprioception。
- 旧 rollout 只用于恢复 simulator snapshot 和 outcome-blind phase 分层。candidate noise、action、routing、continuation noise 和 terminal outcome 全部使用新的 seed domain。

## 反事实 terminal-Q 标签

对同一 snapshot 的每个候选执行其完整 action chunk，然后使用冻结 base HiMoE-VLA 延续至成功或 episode 总计 300 个环境步。不同候选共享相同的 continuation noise-seed 列，且 seed 推导明确排除 candidate ID。

初始 panel 为 $R=4$。Gate 1 判定后，只有预先规定的 selector disagreement、非退化 Bernoulli outcome 或候选分化状态补到 $R=8$；这些补采标签不回写 Gate 1 判定。

## Gate 1：是否存在可兑现的候选 headroom

每个 snapshot 将四个 continuation repeats 分成独立 A/B 两半：A 选择候选、B 评价，随后交换并平均。这样避免对 noisy Monte Carlo $\hat q$ 取最大值产生 winner's curse。

主统计量是 $N=8$ 的 task-macro cross-fitted headroom。bootstrap 先采 task、再在 task 内采 snapshot，共 10,000 次。通过条件：点估计大于 0，且 95% CI 下界大于 0。candidate、pair 或 subset 均不作为独立推断单位。

## Gate 2：routing 是否有增量价值

Gate 1 通过后训练容量匹配的四个 dueling chunk critic：

1. `state_action`
2. `state_route`
3. `state_action_route`
4. `state_action_router_hidden`

训练损失为按 snapshot 归一化的 binomial value loss，加同 snapshot、非 tie 候选的加权 pairwise rank loss。advantage 只在完整候选集合内中心化。每个主模型使用 5-member ensemble。

主评估使用 parent-episode-grouped test split；另对三个任务逐一做 leave-one-task-out。报告 pairwise accuracy、top-1 regret、selected Q、oracle recovery、Brier 和 binomial proportion NLL。

Gate 2 通过条件：held-out test 上 joint 相对 action-only 的 selected-Q paired hierarchical 95% CI 下界大于 0，同时 pairwise delta 大于 0、top-1 regret delta 小于 0。

## Routing 排除实验

- 同 snapshot candidate routing shuffle
- 跨 snapshot routing shuffle
- 全局 expert-ID permutation 后重新训练
- flow order reverse
- state-token routing 移除
- action-token routing 候选差异移除
- early/middle/late/final flow only
- AS-MoE negative control

若跨 snapshot shuffle 下降而同 snapshot shuffle 不下降，则 routing 主要是 state/task-stage telemetry，而不是 candidate advantage。

## Gate 3：闭环 Best-of-N

只有 Gate 2 通过才授权闭环实验。闭环比较 $N\in\{1,2,4,8\}$，使用 ensemble mean、disagreement penalty 和 uncertainty fallback；最终报告 success-cost 曲线，而不是仅报告离线 AUC。

## 冻结文件与命令

- 配置：`bestofn_experiment_config.json`
- snapshot 计划：`runs/bestofn-critic-20260825/plan.json`
- run manifest：`runs/bestofn-critic-20260825/run_manifest.json`

```bash
python3 run_bestofn_experiment.py preflight
python3 run_bestofn_experiment.py status
python3 run_bestofn_experiment.py gate1
python3 run_bestofn_experiment.py topup
python3 run_bestofn_experiment.py critics --device cuda:0
```

`gate1`、`topup` 和 `critics` 都是可恢复阶段；任何中断只认可带完整 artifact descriptor 和哈希校验的已发布前缀。
