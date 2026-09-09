# 全语料 lift 基准更正：suite 匹配 → task 匹配

`moe-prior-correction-0906` · 2026-09-06 · 纯 CPU

## 0. 结论

**7 个 bundle 已发表的 1,018 个检测器，lift 全部用的是被高估的 suite 匹配基准。改成 task 匹配后，中位 lift 从 1.4493 掉到 1.1249——超出基线的部分收缩 72.2%。其中 34 个（29 个严格低于 1）由"跑赢生存基线"变成"跑不赢"。**

原因是共享代码：这 7 个 bundle 全都 `from select_early_lock import prior_of`，而
`moe-hb-front-back-0905/experiments/select_early_lock.py:97` 的 `survival_prior`
按 **suite** 分组。之前的审计已经量化过这个基准的问题：264 个无阈值 cell 里，
suite 分层置换有 52 个存活，task 分层只剩 **1** 个。suite 分层不是"偏弱"，
是**反保守**的。

本 bundle 不重选任何东西、不重拟任何阈值——报警向量按发表原样读入，只换基准。

## 1. 度量本身是对的（两个必须通过的检查）

| 检查 | 结果 |
|---|---|
| `all_running_at_chunk{0,4,8,12}`：对"在第 q 步还在跑的全部 episode"报警 | 两种基准下 lift **精确等于 1.0**（代码中断言，1e-9） |
| 两种基准用**同一个** `survival_prior` 实现，只换 `group` 参数 | 差异只可能来自分层，不可能来自实现 |

第一项是定义性的：对所有存活者报警，其精度就是生存先验本身，lift 必须是 1。
两个 cohort、四个 chunk 全部精确通过。

## 2. 负对照：长度

| cohort | TP | FP | 精度 | **task lift** |
|---|---:|---:|---:|---:|
| development_main | 487 | 0 | 1.0000 | **7.1212** |
| external_8b | 564 | 4 | 0.9930 | **6.7357** |

长度检测器的 task lift 比任何 MoE 检测器都高（最好的存活者是 2.16），召回 1.000。
**这正是它不能当基线的原因**：风险的定义就是"没在上限前结束"，所以任何低于上限的
长度阈值按构造捕获全部风险。它保留为负对照——**任何声称跑赢它的东西都是在复述标签定义**。

## 3. 分 bundle 的影响

| bundle | 检测器（≥20 次报警） | 中位 suite lift | 中位 task lift | 跌破 1 | task lift > 1.3 |
|---|---:|---:|---:|---:|---:|
| moe-combination-rules-0906 | 640 | 1.4239 | 1.1196 | **9** | 84 |
| moe-circuit-analogy-0906 | 73 | 1.4952 | 1.1452 | **1** | 5 |
| moe-flow-semantics-0906 | 55 | 1.4920 | **1.2164** | 0 | 16 |
| moe-failure-modes-0906 | 56 | 1.4296 | 1.1183 | 0 | 9 |
| moe-two-tier-0906 | 48 | 1.4489 | 1.1183 | 0 | 8 |
| moe-state-channel-0906 | 35 | 1.5051 | **1.1769** | 0 | 11 |
| moe-token-geometry-0906 | 25 | 1.4778 | 1.1501 | 0 | 3 |

收缩幅度在各 bundle 之间高度一致（中位 task lift 全部落在 1.118–1.216），
说明这是**基准的系统性偏差，不是某个 bundle 的方法问题**。

## 4. 需要撤回的结论（≥20 次报警，suite lift > 1 但 task lift ≤ 1）

| 检测器 | 报警 | 精度 | suite lift | **task lift** |
|---|---:|---:|---:|---:|
| `flow_settling_log_ratio\|global\|0.85` (ext) | 36 | 0.167 | 1.6886 | **0.8932** |
| `conditional_effective_rank\|global\|0.7` (ext) | 2,030 | 0.158 | 1.5268 | **0.9901** |
| `partial_query_d1\|global\|0.7` (ext) | 2,053 | 0.157 | 1.4694 | **0.9957** |
| `conditional_effective_rank\|per_task\|0.96` (ext) | 71 | 0.930 | 1.4378 | **0.9906** |
| `partial_edge_std\|per_task\|0.96` (ext) | 76 | 0.908 | 1.3933 | **0.9980** |
| `partial_query_d1\|global\|0.6` (ext) | 2,890 | 0.127 | 1.3293 | **0.9907** |
| `flow_settling_log_ratio\|global\|0.85` (dev) | 55 | 0.109 | 1.2763 | **0.8446** |
| `flow_settling_log_ratio\|global\|0.8` (ext) | 80 | 0.088 | 1.1480 | **0.7971** |
| `expert_load_effective_rank\|global\|0.99` (ext) | 108 | 0.963 | 1.0816 | **0.9967** |
| `circuit\|th_slope\|per_task` (ext) | 24 | 0.417 | 1.0231 | **0.9925** |

**`flow_settling_log_ratio` 是最严重的一个**：它是"flow 轴"这条线的 headline，
被 4 个 bundle 引用（circuit-analogy、combination-rules、failure-modes、two-tier），
task lift 三个操作点全部**低于 1**——比直接押生存先验还差。它是一个任务选择器。

`action_consensus|global` 被同样这 4 个 bundle 引用：suite lift 3.34 →
**task lift 1.08**，基本等于什么也没多说。这是全表中收缩最大的一类。

## 5. 挺过 task 匹配的（≥20 次报警，按 task lift 取前 10，未做任何挑选）

| 检测器 | 报警 | 精度 | suite lift | **task lift** |
|---|---:|---:|---:|---:|
| `load_entropy_s0\|global` (ext) | 87 | 0.759 | 2.4439 | **2.1614** |
| `pc1corr_slope\|global` (ext) | 31 | 1.000 | 1.4924 | **1.6936** |
| `conditional_query_d1\|global\|0.99` (dev) | 90 | 0.978 | 1.6407 | **1.6835** |
| `conditional_query_d1\|global\|0.99` (ext) | 104 | 0.990 | 1.5468 | **1.6831** |
| `partial_query_d1\|global\|0.95` (dev) | 27 | 0.815 | 3.5467 | **1.6692** |
| `stretch_slope\|global` (ext) | 85 | 0.788 | 1.7289 | **1.6584** |
| `action_consensus\|global\|0.9` (ext) | 189 | **0.328** | 1.7590 | **1.6556** |
| `load_entropy_s6\|global` (ext) | 63 | 0.921 | 1.8052 | **1.6426** |
| `conditional_query_d1\|global\|0.98` (ext) | 149 | 0.940 | 1.5788 | **1.6352** |
| `load_entropy_s3\|global` (ext) | 140 | 0.779 | 1.8711 | **1.6240** |

注意第 7 行：`action_consensus|global|0.9` 的 task lift 排进前 10，但精度只有 **0.328**。
高 lift 不等于可用——它是在低先验区大量报警换来的，实际 127 个假警。**lift 与精度必须一起看。**

三类东西活下来了，而且都是 `global`（任务无关）阈值：

1. **逐去噪步的专家负载熵** `load_entropy_s{0,3,6}`——最强的存活者，
   且 s0 是**去噪第一步**。这条轴之前基本没用过。
2. **Schur 补条件几何** `conditional_query_d1`——两个 cohort 一致（1.6835 / 1.6831）。
3. **state 通道的斜率** `pc1corr_slope` / `stretch_slope`（`state_mobility_mean`
   task lift 1.6005 紧随其后）。

前 10 里有 **4 个**的 suite lift **低于** task lift（1.49→1.69、1.64→1.68、
1.55→1.68、1.58→1.64）——即 suite 基准**低估**了它们。这是可信度的正面信号：
任务混杂在压低而不是抬高它们的表现。

## 6. 低先验（早期）子集：基准也改变了子集本身

"早期"定义为报警时先验 < 0.25，而先验取决于基准，所以换基准会同时改变**哪些报警算早期**。
一致地按 task 匹配重算：

| 检测器 | suite 基准 TP/FP | suite lift | **task 基准 TP/FP** | **task lift** |
|---|---:|---:|---:|---:|
| `load_entropy_s0\|global` | 35/20 | 6.25 | **36/15** | **7.26** |
| `load_entropy_s3\|global` | 49/29 | 6.05 | **43/20** | **7.33** |
| `load_entropy_s6\|global` | 16/5 | 6.24 | **17/2** | **10.02** |
| `conditional_query_d1\|global\|0.975` | 14/1 | 11.40 | **9/0** | **11.72** |
| `conditional_query_d1\|global\|0.98` | 11/0 | 10.36 | **9/0** | **10.30** |
| `partial_edge_std\|global\|0.925` (dev) | 38/16 | 6.32 | **0/1** | **0.00** |
| `partial_query_d1\|global\|0.95` (dev) | 17/4 | 6.66 | **0/0** | — |

前五行在两种基准下都成立。**最后两行是警告**：`partial_edge_std|global|0.925` 在
suite 基准下有 38 个早期 TP、lift 6.32，换成 task 基准后**一个早期 TP 都不剩**——
那 38 次报警之所以看起来"早"，是因为它们打在 suite 内部的高风险任务上，而这些任务
在同 suite 平均先验下显得先验低。这类彻底塌陷占 ≥20 报警检测器的一小部分，但一旦
出现就是完全的假象。

## 7. 两条现在起适用于所有报告的硬性规则

**规则一：长度类方法不是基线，是负对照。** 因为失败=没在上限前结束，任何长度
阈值按构造召回 100%。它必须出现在每张表里（防止有人以为跑赢了它），但绝不能
作为比较基准，也不能进任何排名。

**规则二：任何量都要报 episode 内信息量，lift 必须用 task 匹配的先验。**
episode 内逐位恒定的通道要排除（它能靠 trailing mean 的浮点漂移通过冻结的选择规则——
`as_entropy` 就在 11 表示 × 2 方向 × 14 分位的搜索下拿到过 lift 1.56，
而它的 task 匹配 lift 精确等于 1.0000）。

## 8. 限制

1. 本次只更正**基准**。这 7 个 bundle 的**选择**过程仍然是在 suite 匹配 lift 上做的，
   所以选出来的操作点对 task 匹配基准而言是次优的——第 5 节的存活者是"在错误目标下
   选出、但在正确目标下仍然成立"，不是"在正确目标下的最优"。重选会给出不同（很可能更好）
   的操作点。
2. 先验在被打分的样本内估计。这对生存基线有利，因此对 lift 是**保守**方向。
3. `moe-audit-0906` 与 `moe-replay-ledger-0906` 没有存报警向量，未纳入重算；
   它们引用的 lift 数字来自上述 bundle，按同样比例失真。
4. 报警数 < 20 的检测器（1,018 中的 86 个）的 lift 两种基准下都无意义，
   已在主表中单列。最大的几个"收缩"发生在只报警 1–5 次的检测器上，不应引用。

## 9. 复现

```bash
cd /home/jovyan/work/himoe-vla
python moe-prior-correction-0906/experiments/recompute_task_matched_lift.py
```

约 1 秒，纯 CPU。输出 `results/task_matched_lift.csv`（1,028 行 = 1,018 个已发表
检测器 + 8 个控制项 + 2 个负对照）与 `results/summary.json`。控制项断言若不通过则脚本中止。
