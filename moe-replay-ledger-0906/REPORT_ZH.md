# 离线重放运行台账：96 个检测器，逐 chunk 记账，然后揭盲

本 bundle 不提出新检测器。它把这个 corpus 里已有的检测器全部拉到同一张表上，按运行现场的顺序重放一遍，
记录每一次报警**在报警那一刻能知道什么**，再揭盲结果与物理失效模式。

产物：

| 文件 | 内容 |
|---|---|
| `results/alarm_ledger_external.csv` | 逐报警台账，32 841 行（96 个方法 × 各自报警的 episode） |
| `results/alarm_ledger_development.csv` | 同上，development 侧 19 306 行（52 个方法） |
| `results/chunk_view_external.csv` | 逐 chunk 运行视图，分 suite 与 pooled |
| `results/method_detail_external.csv` | 每方法明细表（计数 / 结果拆分 / 模式拆分 / 时序四分位 / lift / 早期带 / 预算视图 / 漏检） |
| `results/method_by_suite_external.csv`、`method_by_task_external.csv` | suite 与 **task** 内分层 |
| `results/task_stratified_comparison.csv`、`task_consistency.csv` | 任务分层的成对置换检验 |
| `results/worked_traces.json` | 13 条逐 chunk 案卷 |
| `results/causality_check.json` | 因果性验证 |
| `results/bundle_disagreement.json` | 跨 bundle 复现失败的定位 |
| `results/mode_cross_check.csv`、`development_to_external_transfer.csv`、`headline_table.csv` | 交叉核对 |
| `results/manifest.json` | 纳入 / 排除清单、口径、全部注意事项 |

---

## 0. 读这份台账之前必须先接受的三件事

**（一）全部日志都是"无监视器在环"录下来的。** 策略自己跑到成功或跑到 horizon 上限，从来没有被任何报警打断过。
所以这份台账**只测量检测，不测量干预**。运行视图是"一个只看不动手的观察者会看到什么"，
它本身**不构成部署结论**。要回答"报警之后动手是否有用"，需要从同一个仿真快照分叉出多条续跑，
而缓存的路由重放做不到这件事。`manifest.json` 里记的是 `intervention_effect_measured: false`。

**（二）"episode"是一个事后构造物。** 一个 episode = 一条完整的日志 rollout，由 (task, 初始状态, flow noise seed) 唯一确定，
50 个初始状态 × 8 个 seed = 每个 task 400 条。它的路由数据确实来自真实策略推理，每个 chunk 一次前向。
但"这条 episode 长度是 30、结果是失败"这个单元是**跑完之后才存在的**：在线时只有一条正在跑、剩余长度未知的 rollout。
这就是 `length` 永远不能进入任何 score 的原因。台账的行单位（每个 (method, episode) 一行）不是运行时可得的东西。

**（三）length-only 基线不是竞争对手，是标签的另一种写法。** `original_failure` 的定义就是"没能在 horizon 上限之前结束"。
external 上 **568 条 episode 触到上限，其中 564 条是 risk，精度 0.9930**。所以一条只读已耗时长的规则
recall 必然 1.000。它量的是**定义**，不是策略，也不是检测能力。它作为参照行放进表里，**不是要被超越的目标**。
长度只能作为条件化 / 加权变量进入，绝不能作为特征——而"条件化"这件事，生存先验已经在做了。

---

## 1. 因果性：验证过，不是假设

台账用了各 bundle 已存的 first-alarm 数组。这些数组声称是逐 chunk 因果重放的产物。我们没有相信这个声称，而是重建并做了截断实验
（`experiments/verify_replay.py`，结果在 `results/causality_check.json`）：

对锚点 `mobility | L12 | low | q0.975 | global`，从原始逐 chunk mobility 数组
（`moe-v4-0904/results/layerwise_mobility/external_8b.npz`）出发，用 pooled development 峰值分位重建阈值 −0.0232071877：

| 检查 | 结果 |
|---|---|
| 重建数组与已封存数组 | **逐位相同**（15 600 行 0 处不同），195 TP / 17 FP，与发布值一致 |
| 把每条报警 episode 在**报警 chunk 处**截断后重放 | 212 / 212 条报警 chunk **完全不变** |
| 在**报警 chunk 之后 1 个 chunk** 处截断 | 212 / 212 不变 |
| 在**报警 chunk 之前 1 个 chunk** 处截断（证伪对照） | 212 / 212 报警**正确消失** |

同样的三项检查在 `mobility | L2 | low | q0.700 | per_task`（329 条，329/329 通过；重建与封存数组逐位相同）
和 v4 的 `L5 | high | w4 | k8 | q0.80`（39 条，39/39 通过）上重复。后者是本 corpus 里回看窗口最长的规则（8 次确认）。

> 正向检查（截断后报警不动）证明没有偷看未来；反向检查（提前一个 chunk 截断后报警消失）证明报警 chunk 确实是
> **证据首次成立的那一个 chunk**，不是更早就成立却被延迟记录的。两项都过。

另外，台账里每一个数组都被结构性断言过：**没有任何一次报警落在 episode 从未到达的 chunk 上**（96 / 96 通过）。

---

## 2. 复现情况：96 / 96 与发布值一致

| 锚点 | 要求 | 实测 |
|---|---|---|
| `mobility \| L12 \| low \| q0.975 \| global` | 195 / 17 | **195 / 17** ✓ |
| `mobility \| L2 \| low \| q0.700 \| per_task` | 272 / 57 | **272 / 57** ✓ |
| `expert_load_effective_rank \| L3 \| low \| q0.85 \| per_task` | 370 / 93 | **370 / 93** ✓ |
| v7 intrinsic guard | 439 / 80 | **439 / 80** ✓ |

external 上 96 个方法里，凡是 bundle 发布过计数的（79 个），**全部与我们组装出来的计数完全相同，零处不符**
（`manifest.json::discrepancies` 为空）。development 侧 52 个方法同样零处不符。

行对齐是断言出来的，不是假设的：标签文件的 task / episode 顺序与 mobility cache、layer-graph cache 逐行比对；
`moe-state-channel-0906` 与 `moe-two-tier-0906` 自带的 `risk / suite / length / task / episode` 元数据全部逐行核对通过。

---

## 3. 头部方法明细表（external_8b，15 600 条，564 risk = 523 persistent + 41 late success）

所有比率都带 n。物理模式的 n 是对全部 564 条 risk 计的。

### 3.1 计数与信息密度

| 方法 | 阈值模式 | 报警 | TP | FP | precision | recall | timely-FPR | 匹配先验均值 | **lift** |
|---|---|---|---|---|---|---|---|---|---|
| `baseline:length_only_task_devcal@0.005`（**标签复述，参照行**） | per_task | 609 | 564 | 45 | 0.926 | **1.000** | 0.0030 | 0.811 | 1.14 |
| **v7 intrinsic guard** | task_agnostic | 519 | **439** | 80 | 0.846 | 0.778 | 0.0053 | 0.606 | 1.40 |
| **v4 dual regime** | per_task | 491 | 410 | 81 | 0.835 | 0.727 | 0.0054 | 0.575 | 1.45 |
| **combo P-C** | per_task | 448 | 391 | 57 | 0.873 | 0.693 | 0.0038 | 0.602 | 1.45 |
| **跨帧三头 OR** | global | 435 | 381 | 54 | 0.876 | 0.676 | 0.0036 | 0.588 | 1.49 |
| `expert_load_effective_rank\|per_task` | per_task | 463 | 370 | 93 | 0.799 | 0.656 | 0.0062 | 0.516 | 1.55 |
| `twotier:WATCH_a3` | global | 607 | 410 | **197** | 0.675 | 0.727 | **0.0131** | 0.421 | 1.60 |
| **combo G-A** | global | 301 | 288 | **13** | **0.957** | 0.511 | 0.0009 | 0.715 | 1.34 |
| `mobility\|global` | global | 212 | 195 | 17 | 0.920 | 0.346 | 0.0011 | 0.521 | **1.76** |
| `twotier:ACT_b` | global | 155 | 146 | 9 | 0.942 | 0.259 | 0.0006 | 0.513 | **1.84** |
| `prior:HB_MoE_top4churn@1%`（先前 bundle） | task_agnostic | 517 | 131 | 386 | 0.253 | 0.232 | 0.0257 | **0.069** | **3.70** |

> lift = precision / 匹配到的生存先验均值。`prior:HB_MoE_top4churn` 的 lift 3.70 在 **TP ≥ 20 的方法里是全表最高**，
> **远超**本 corpus 长期挂在嘴边的 1.764 记录。（TP 不设下限时最高的是 `flow_settling_log_ratio|global` 的 4.54，
> 但它只有 3 TP / 2 FP，没有意义。）原因不神秘：它在 chunk 中位数 9 就开火，那时先验只有 0.069。它是唯一一个**主要工作在低先验带**的方法。
> 代价是 precision 0.253。这条比较说明"lift 记录"和"precision 记录"是两个不同的量，不能混着报。

### 3.2 结果拆分：报警落在什么结果上

| 方法 | 落在 timely success（15 036） | 落在 late success（41） | 落在 persistent failure（523） | late 召回 | persistent 召回 |
|---|---|---|---|---|---|
| v7 guard | 80 | 30 | 409 | 30/41 = 0.732 | 409/523 = 0.782 |
| v4 dual regime | 81 | 27 | 383 | 27/41 = 0.659 | 383/523 = 0.732 |
| combo P-C | 57 | 22 | 369 | 22/41 = 0.537 | 369/523 = 0.706 |
| 跨帧三头 OR | 54 | 25 | 356 | 25/41 = 0.610 | 356/523 = 0.681 |
| `expert_load\|per_task` | 93 | 18 | 352 | 18/41 = 0.439 | 352/523 = 0.673 |
| `twotier:WATCH_a3` | 197 | 28 | 382 | 28/41 = 0.683 | 382/523 = 0.730 |
| combo G-A | 13 | 19 | 269 | 19/41 = 0.463 | 269/523 = 0.514 |
| `mobility\|global` | 17 | 13 | 182 | 13/41 = 0.317 | 182/523 = 0.348 |
| `twotier:ACT_b` | 9 | 11 | 135 | 11/41 = 0.268 | 135/523 = 0.258 |
| `prior:HB_MoE_top4churn@1%` | 386 | 9 | 122 | 9/41 = 0.220 | 122/523 = 0.233 |
| length_only（参照） | 45 | 41 | 523 | 41/41 = 1.000 | 523/523 = 1.000 |

> 值得单独记一笔：**41 条 late success（本来只差 10 次 query 就能成功）里，v7 guard 报了 30 条、
> `twotier:WATCH_a3` 报了 28 条、v4 报了 27 条。** 按标签口径它们都算 TP。若部署时的正确处置是"再给一点时间"，
> 这些报警的净效应是正是负，取决于干预策略——而干预在本 corpus 里没有被测过。

### 3.3 物理失效模式拆分，每个模式都带 n

模式 n 对全部 564 条 risk：dropped 241、no_grasp 143、moved_unmet 89、regressed 26、released_outside 25、
timeout_holding 23、no_contact 14、no_progress 2、mechanism 1。
仅对 523 条 persistent 时：dropped 231、no_grasp 136、moved_unmet 83、regressed 26、released_outside 24、
no_contact 14、timeout_holding 8、mechanism 1。两组数字都在 `manifest.json` 里，与我们对物理标签表的独立 join 完全一致。

| 方法 | dropped (241) | no_grasp (143) | moved_unmet (89) | regressed (26) | released_outside (25) | timeout_holding (23) | **no_contact (14)** | 最差模式 |
|---|---|---|---|---|---|---|---|---|
| v7 guard | 188 (0.780) | 119 (0.832) | 60 (0.674) | 22 (0.846) | 16 (0.640) | 18 (0.783) | 13 (0.929) | **0.640** released_outside |
| v4 dual regime | 186 (0.772) | 106 (0.741) | 61 (0.685) | 16 (0.615) | 16 (0.640) | 18 (0.783) | 4 (0.286) | 0.286 no_contact |
| combo P-C | 217 (0.900) | 56 (0.392) | 54 (0.607) | 18 (0.692) | 24 (0.960) | 14 (0.609) | 5 (0.357) | 0.357 no_contact |
| 跨帧三头 OR | 165 (0.685) | 114 (0.797) | 50 (0.562) | 17 (0.654) | 15 (0.600) | 12 (0.522) | 5 (0.357) | 0.357 no_contact |
| `twotier:WATCH_a3` | 170 (0.705) | 123 (0.860) | 54 (0.607) | 21 (0.808) | 15 (0.600) | 13 (0.565) | 11 (0.786) | 0.565 timeout_holding |
| combo G-A | 125 (0.519) | 98 (0.685) | 31 (0.348) | 10 (0.385) | 12 (0.480) | 8 (0.348) | 3 (0.214) | 0.214 no_contact |
| `mobility\|global` | 61 (0.253) | 78 (0.545) | 27 (0.303) | 12 (0.462) | 3 (0.120) | 9 (0.391) | 2 (0.143) | 0.120 released_outside |
| `twotier:ACT_b` | 39 (0.162) | 70 (0.490) | 20 (0.225) | 4 (0.154) | 3 (0.120) | 7 (0.304) | **0 (0.000)** | **0.000** no_contact |

**关键读法**：`no_contact` n = 14、`timeout_holding` n = 23、`released_outside` n = 25、`regressed` n = 26。
这四个模式上的任何"覆盖率"差别都建立在两位数样本上，单条 episode 的翻转就能改变 0.07 以上的覆盖率。
**`twotier:ACT_b` 的 0/14 是确定的空白，但"0/14 vs 5/14"这种差别不构成证据。**

**置信度警告**：523 条 persistent failure 里 **497 条的失效原因置信度是 medium，只有 26 条是 high**
（`manifest.json::mode_confidence_over_persistent`）。全部 564 条通过了 physics_validation。
上面每一个按模式拆的数字都要在这个前提下读。

### 3.4 报警时机：中位数与四分位

| 方法 | 报警 chunk (q25/中位/q75) | 剩余预算 H−1−q (q25/中位/q75) | 匹配先验 (q25/中位/q75) | TP 的 lead 中位数 |
|---|---|---|---|---|
| v7 guard | 19 / 25 / 32 | 4 / 15 / 20 | 0.451 / 0.496 / 0.966 | 12 |
| v4 dual regime | 17 / 23 / 33 | 4 / 14 / 19 | 0.212 / 0.496 / 0.960 | 12 |
| combo P-C | 15 / 19 / 26 | 4 / 9 / 17 | 0.182 / 0.714 / 0.966 | 7 |
| 跨帧三头 OR | 16 / 20 / 34 | 5 / 12 / 17 | 0.426 / 0.504 / 0.959 | 11 |
| `expert_load\|per_task` | 13 / 18 / 24 | 6 / 11 / 21 | 0.077 / 0.482 / 0.957 | 9 |
| `twotier:WATCH_a3` | 15 / 18 / 27 | 7 / 17 / 32 | 0.071 / 0.414 / 0.854 | 12 |
| combo G-A | 17 / 21 / 36 | 4 / 8 / 15 | 0.496 / 0.881 / 0.966 | 8 |
| `mobility\|global` | 26 / 34 / 36 | 12 / 15 / 18 | 0.467 / 0.496 / 0.597 | 15 |
| `twotier:ACT_b` | 31 / 35 / 36 | 13 / 15 / 18 | 0.488 / 0.496 / 0.528 | 15 |
| `prior:HB_MoE_top4churn@1%` | 6 / 9 / 13 | 14 / 22 / 25 | 0.026 / 0.035 / 0.068 | 13 |
| length_only（参照） | 15 / 26 / 48 | 3 / 6 / 7 | 0.830 / 0.952 / 0.993 | 6 |

> `mobility|global` 与 `twotier:ACT_b` 的匹配先验四分位区间极窄（0.467–0.597、0.488–0.528）：
> 它们几乎只在一个很短的时间窗里开火。反过来 `expert_load|per_task` 与 `twotier:WATCH_a3` 的
> q25 先验只有 0.077 / 0.071，说明它们有相当一部分报警确实落在低先验带。

### 3.5 早期带：先验跨过 0.25 之前的 TP / FP

先验跨 0.25 的 chunk：goal 18、long 26、object 17、spatial 13。

| 方法 | 早期 TP | 早期 FP | 早期 precision | 早期 TP 占全部 TP |
|---|---|---|---|---|
| `expert_load_effective_rank\|per_task` | 97 | 69 | 0.584 | 0.262 |
| **combo P-C** | **96** | **31** | **0.756** | 0.246 |
| `twotier:WATCH_a3` | 113 | 171 | 0.398 | 0.276 |
| v4 dual regime | 76 | 54 | 0.585 | 0.185 |
| `flow:mobility_s0\|per_task` | 75 | 39 | 0.658 | 0.236 |
| **length_only_task_devcal@0.005（参照）** | **63** | **9** | **0.875** | 0.112 |
| 跨帧三头 OR | 59 | 29 | 0.670 | 0.155 |
| v7 guard | 57 | 44 | 0.564 | 0.130 |
| `mobility\|global` | 36 | 4 | 0.900 | 0.185 |
| `twotier:ACT_b` | 23 | **0** | 1.000 | 0.158 |
| `prior:HB_MoE_top4churn@1%` | **116** | 384 | 0.232 | 0.885 |

### 3.6 预算视图：要求至少 B 个 chunk 剩余预算后还剩多少（TP / FP）

**这一栏是全份报告里最重要的一栏。**

| 方法 | B=0 | B=2 | B=4 | B=8 | **B=12** |
|---|---|---|---|---|---|
| **length_only_task_devcal（参照）** | 564 / 45 | 564 / 45 | 315 / 42 | 115 / 26 | **73 / 14** |
| **v7 guard** | 439 / 80 | 379 / 80 | 331 / 79 | 251 / 74 | **221 / 71** |
| **v4 dual regime** | 410 / 81 | 364 / 81 | 330 / 81 | 229 / 77 | **210 / 72** |
| `twotier:WATCH_a3` | 410 / 197 | 402 / 197 | 360 / 197 | 255 / 193 | 219 / 190 |
| 跨帧三头 OR | 381 / 54 | 374 / 54 | 326 / 54 | 222 / 50 | **186 / 47** |
| combo P-C | 391 / 57 | 365 / 57 | 293 / 56 | 192 / 52 | 155 / 43 |
| `mobility\|global` | 195 / 17 | 188 / 17 | 171 / 17 | 161 / 17 | **151 / 16** |
| `twotier:ACT_b` | 146 / 9 | 144 / 9 | 142 / 9 | 136 / 9 | **125 / 8** |
| combo G-A | 288 / 13 | 249 / 13 | 214 / 12 | 152 / 11 | 111 / 9 |
| `prior:HB_MoE_top4churn@1%` | 131 / 386 | 131 / 386 | 131 / 386 | 131 / 386 | 88 / 376 |

> length_only 在 B=0 上以 564 / 45 碾压一切；**到 B=12 就只剩 73 TP**，被 v7 guard（221）压掉三倍。
> 原因很直白：它的报警按构造集中在 horizon 上限前的最后几个 chunk。**"计数上无敌"和"能提前示警"是两件事，
> 剩余预算这一列就是把它们分开的那把尺。**

### 3.7 回收干预预算（一个更好的排序工具，不是一个价值度量）

对一条 cap 为 H、在 chunk q 报警的真阳性，报警买到 H−1−q 个 chunk 的**潜在**干预窗。
上限取"每条 risk 都在 chunk 6 报警"（6 是本项目任何头结构上最早可能开火的 chunk：宽度 4 的 trailing mean 加 4 次确认），
**ceiling = 17 792 chunk**。

| 方法 | budget_won | 占上限 | 平均每 TP | budget_wasted_FP | gain / loss |
|---|---|---|---|---|---|
| `twotier:WATCH_a3` | 6 033 | 33.9% | 14.7 | 5 912 | 1.02 |
| **v7 guard** | **5 534** | **31.1%** | 12.6 | 1 894 | 2.92 |
| v4 dual regime | 5 092 | 28.6% | 12.4 | 2 138 | 2.38 |
| `twotier:WATCH_a1` | 5 022 | 28.2% | 14.4 | 1 340 | 3.75 |
| **跨帧三头 OR** | **4 732** | **26.6%** | 12.4 | 1 052 | **4.50** |
| `expert_load_effective_rank\|per_task` | 4 514 | 25.4% | 12.2 | 2 214 | 2.04 |
| combo P-C | 4 227 | 23.8% | 10.8 | 1 141 | 3.70 |
| length_only_task_devcal（参照） | 3 623 | 20.4% | 6.4 | 446 | 8.12 |
| `mobility\|per_task` | 3 162 | 17.8% | 11.6 | 1 456 | 2.17 |
| **`mobility\|global`** | 3 016 | 17.0% | **15.5** | 374 | **8.06** |
| `flow_settling_log_ratio\|per_task` | 2 908 | 16.3% | 9.6 | 1 785 | 1.63 |
| combo G-A | 2 667 | 15.0% | 9.3 | 244 | 10.93 |
| `twotier:ACT_b` | 2 242 | 12.6% | 15.4 | 171 | **13.11** |

这个排序和按 TP 排出来的不一样。最清楚的例子：`flow_settling_log_ratio|per_task` 的 TP（304）**多于**
`mobility|per_task`（272），但 budget_won 更少（2 908 vs 3 162），因为它平均每个 TP 只买到 9.6 个 chunk——
**抓得更多但抓得更晚**。反过来，`mobility|global` 平均每 TP 15.5 chunk 是全表最高之一，gain/loss 8.06，只是总量小。

> **限制，必须明说**：这个加权假设"剩余预算的价值随 chunk 数线性增长"。这几乎肯定是错的——
> 很可能存在一个下限，低于它报警毫无用处；而真正的价值函数由干预机制决定，而干预机制在本 corpus 里**完全没有被测过**。
> 请把它当作**比原始计数更好的排序工具**，不要当作绝对价值。

### 3.8 漏检，按模式拆

| 方法 | 漏检 risk | dropped | no_grasp | moved_unmet | regressed | released_outside | no_contact | timeout_holding |
|---|---|---|---|---|---|---|---|---|
| v7 guard | 125 | 53 | 24 | 29 | 4 | 9 | 1 | 5 |
| v4 dual regime | 154 | 55 | 37 | 28 | 10 | 9 | 10 | 5 |
| combo P-C | 173 | 24 | 87 | 35 | 8 | 1 | 9 | 9 |
| 跨帧三头 OR | 183 | 76 | 29 | 39 | 9 | 10 | 9 | 11 |
| `mobility\|global` | 369 | 180 | 65 | 62 | 14 | 22 | 12 | 14 |
| `twotier:ACT_b` | 418 | 202 | 73 | 69 | 22 | 22 | **14** | 16 |
| length_only（参照） | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

`combo:P-C` 与 `v7 guard` 的漏检结构**几乎正交**：P-C 漏掉 87/143 条 no_grasp 但只漏 24/241 条 dropped；
v7 guard 漏掉 53 条 dropped 但只漏 24 条 no_grasp。**没有任何一条 risk 被全部 85 个 routing 检测器同时漏掉**
（并集覆盖 564/564），但有 5 条 risk 只被 85 个里的 1 个抓到。

---

## 4. 运行视图：只看不动手的观察者当时看到的东西

`results/chunk_view_external.csv` 给出每个方法 × 每个 chunk × 每个 suite 的
（仍在跑的 episode 数、该 chunk 新增报警数、其中真的是 risk 的数、累计）。horizon 不同，所以分 suite 报，同时给 pooled。

**v7 guard，libero_goal（H=30，先验在 chunk 18 跨过 0.25）节选：**

| chunk | 仍在跑 | 生存先验 | 该 chunk 报警 | 其中 risk | 累计报警 | 累计 TP |
|---|---|---|---|---|---|---|
| 12 | 1 383 | 0.077 | 0 | 0 | 0 | 0 |
| 13 | 835 | 0.127 | 2 | 2 | 2 | 2 |
| 14 | 601 | 0.176 | 8 | 8 | 10 | 10 |
| 15 | 519 | 0.204 | 6 | 6 | 16 | 16 |
| 17 | 475 | 0.223 | 1 | 1 | 18 | 18 |
| 19 | 189 | **0.561** | 1 | 1 | 19 | 19 |
| 22 | 115 | 0.922 | 4 | 4 | 26 | 26 |
| 24 | 111 | 0.955 | 15 | 15 | 49 | 49 |
| 25 | 108 | 0.981 | 13 | 12 | 62 | 61 |
| 29 | 107 | 0.991 | 7 | 7 | 74 | 73 |

这张表把问题摆得很清楚：**v7 guard 在 libero_goal 上 74 次报警里有 48 次发生在 chunk 22 之后，
那时"还在跑"本身已经意味着 92% 以上的失败概率。** 早期（chunk 13–17）只有 18 次，但那 18 次**全部正确**（18/18），
先验只有 0.13–0.22。真正有信息的是前 18 次。

各方法最繁忙的三个 chunk（pooled）：

| 方法 | 最忙的 chunk | 仍在跑 | 报警 | 其中 risk / timely |
|---|---|---|---|---|
| `prior:HB_MoE_top4churn@1%` | 5 / 6 / 10 | 15 600 / 15 600 / 11 342 | 73 / 67 / 44 | 1/72、4/63、14/30 |
| `expert_load_effective_rank\|per_task` | 12 / 17 / 19 | 9 146 / 4 584 / 4 093 | 41 / 38 / 44 | 37/4、33/5、40/4 |
| `twotier:WATCH_a3` | 15 / 16 / 17 | 5 074 / 4 793 / 4 584 | 40 / 78 / 59 | 19/21、29/49、39/20 |
| `v4_dual_regime` | 15 / 20 / 25 | 5 074 / 3 995 / 1 542 | 35 / 39 / 31 | 34/1、30/9、30/1 |
| `v7_guard` | 19 / 31 / 32 | 4 093 / 587 / 569 | 41 / 35 / 36 | 33/8、28/7、33/3 |
| `mobility\|global` | 33 / 34 / 36 | 561 / 558 / 544 | 20 / 20 / 30 | 17/3、20/0、28/2 |
| `twotier:ACT_b` | 34 / 35 / 36 | 558 / 552 / 544 | 17 / 16 / 25 | 17/0、16/0、23/2 |

> `mobility|global` 和 `twotier:ACT_b` 的全部忙碌区间都在 chunk 33+，那时全池只剩约 550 条 episode 在跑，
> 其中大半注定失败。它们 0.92 / 0.94 的 precision **主要是生存基线，不是检测**——匹配先验均值 0.51–0.52，
> lift 1.76 / 1.84，即"知道它还在跑"已经解释了 precision 的一半以上。
>
> `prior:HB_MoE_top4churn@1%` 在 chunk 5–6 就开火，那时 15 600 条全部还在跑、先验 0.026–0.036。
> 它在 chunk 5 的 73 次报警里只有 1 次正确，chunk 6 的 67 次里 4 次正确。
> **这是唯一一个真正在"没人知道会不会失败"的时刻说话的检测器，也正因为如此它的 precision 只有 0.25。**

---

## 5. 任务内分层：大多数方法差异经不起检验

corpus 的审计结论是：264 个效应里只有 1 个在 **task 分层**置换下存活，52 个在 suite 分层下存活，且 suite 分层是**反保守**的。
所以本报告里每一个方法对比都用 **39 个 task 内的配对差 + 符号翻转置换（20 000 次）**（`results/task_stratified_comparison.csv`）。

| 对比（TP） | pooled | 任务内：左高 / 右高 / 平 | 置换 p | 存活？ |
|---|---|---|---|---|
| length_only_devcal vs **v7 guard** | 564 vs 439 | 26 / 0 / 13 | < 0.0001 | ✔ |
| length_only_devcal vs **combo P-C** | 564 vs 391 | 11 / 0 / 28 | 0.0012 | ✔ |
| **v7 guard vs v4 dual regime** | 439 vs 410 | **6 / 15 / 18** | **0.597** | ✘ |
| **combo P-C vs expert_load\|per_task** | 391 vs 370 | 14 / 3 / 22 | 0.301 | ✘ |
| **combo G-A vs `mobility\|global`** | 288 vs 195 | 12 / 10 / 17 | 0.109 | ✘ |
| **`mobility\|global` vs `mobility\|per_task`** | 195 vs 272 | 3 / 23 / 13 | **0.609** | ✘ |
| `twotier:WATCH_a3` vs `WATCH_a1` | 410 vs 349 | 19 / 0 / 20 | < 0.0001 | ✔ |
| `HB_MoE_top4churn@1%` vs `mobility\|global` | 131 vs 195 | 16 / 14 / 9 | 0.582 | ✘ |

**必须直说的三件事：**

1. **v7 guard 的 439 TP 与 v4 的 410 TP，在任务内是反向的。** 39 个 task 里 v7 更高的只有 6 个，v4 更高的有 15 个，18 个持平。
   pooled 的 +29 完全来自 task 组成（v7 在 libero_long 上 recall 0.861 vs v4 的 0.715；libero_long 有 274 条 risk，占全部 risk 的 48.6%）。
   **"v7 优于 v4"这句话在池化口径下成立，在任务内不成立。**
2. **`mobility|global` 195 TP vs `mobility|per_task` 272 TP，pooled 差 77，任务内 p = 0.61。**
   同一个量、同一套机器，只是阈值口径不同，池化差距完全被任务组成吞掉。
3. **早期带上，length_only 与所有头部 MoE 方法都没有显著差异**（对 v7 guard p = 0.90，对 v4 p = 0.76，
   对 combo P-C p = 0.23，对 expert_load p = 0.23）。在本报告检验的 15 组配对里，
   **早期带上任务内唯一显著优于 `mobility|global` 的是先前 bundle 的 top-4 churn 报警**
   （p = 0.024，22 个 task 更高 / 4 个更低）；另一组显著的是 `mobility|per_task` 优于 `mobility|global`（p = 0.0037）。

按 suite 拆同样重要（H 不同）：

| 方法 | goal recall（106 risk） | long（274） | object（44） | spatial（140） | FP：goal / long / object / spatial |
|---|---|---|---|---|---|
| v7 guard | 0.689 | 0.861 | 0.500 | 0.771 | 1 / 69 / 7 / 3 |
| v4 dual regime | 0.830 | 0.715 | 0.705 | 0.679 | 5 / 71 / 2 / 3 |
| combo P-C | 0.906 | 0.445 | 0.773 | 0.993 | 10 / 32 / 10 / 5 |
| 跨帧三头 OR | 0.368 | 0.708 | 0.182 | 1.000 | 1 / 33 / 0 / 20 |
| `mobility\|global` | 0.349 | 0.569 | 0.045 | **0.000** | 1 / 16 / 0 / 0 |
| `twotier:ACT_b` | 0.179 | 0.460 | 0.023 | **0.000** | 0 / 9 / 0 / 0 |
| length_only（参照） | 1.000 | 1.000 | 1.000 | 1.000 | 5 / 18 / 9 / 13 |

`mobility|global` 与 `twotier:ACT_b` 在 **libero_spatial 上 recall 恰好为 0**（0/140）。这两个是全表 lift 最高的两个 MoE 方法。
**它们的高 lift 是靠在一个 suite 上完全失明换来的。**

---

## 6. development → external 的阈值迁移

52 个方法两边都有。timely-FPR 膨胀最严重的：

| 方法 | dev TP/FP | ext TP/FP | FPR 膨胀 |
|---|---|---|---|
| `partial_query_d1\|global`（L3/high） | 40 / 21 | 65 / 173 | **7.84×** |
| `build:AND_mobility_flowsettling\|per_task` | 183 / 1 | 187 / 8 | **7.62×** |
| `conditional_effective_rank\|global`（L3/high） | 42 / 20 | 88 / 158 | **7.52×** |
| `flow_path\|global` | 49 / 1 | 50 / 4 | 3.81× |
| `partial_edge_std\|global`（L3/high） | 69 / 40 | 99 / 147 | **3.50×** |
| `expert_load_effective_rank\|per_task` | 322 / 45 | 370 / 93 | 1.97× |

迁移最稳的：`conditional_effective_rank|per_task` 0.48×、`v4:L5_switching` 0.68×、`mobility|global` 1.08×、
`v7 guard` 1.14×、`v4 dual regime` 1.15×。

> 三个 `L3 | high` 全局头（`partial_query_d1`、`conditional_effective_rank`、`partial_edge_std`）的池化全局阈值**都不外推**。
> 这与 two-tier bundle 的结论一致，而且是本 bundle 独立复现的。**任何依赖这三个头的组合规则都不该按其 development 数字定预算。**
>
> 新增观察：`mobility ∧ flow_settling`（per_task AND）在 development 上是 183 TP / **1** FP，看起来完美；
> external 上 187 / 8。8 倍膨胀但绝对量仍小，precision 0.959。这是"分母太小导致 dev 数字没有意义"的典型：
> 1 个 FP 的置信区间宽到什么都说明不了。

`moe-flow-semantics-0906`、`moe-circuit-analogy-0906`、`moe-token-geometry-0906`、`moe-state-channel-0906`、
`moe-two-tier-0906` 只发布了 external 数组，它们的迁移**无法评估**，已在 `manifest.json::skipped` 里逐条记录。

---

## 7. 13 条案卷（逐 chunk）

完整版在 `results/worked_traces.json`（含每条 episode 的完整生存先验曲线与全部 96 个方法的报警时刻）。这里挑要点。

### 案卷 A — 一条几乎被全体漏掉的失败
`libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` ep69，长度 52/52，persistent failure，模式 **dropped**（置信度 medium）。

85 个 routing 检测器里**只有 1 个开火**（`v7:acceleration`，chunk 37，先验 0.528）。头部方法全部沉默。
唯一在报警的是 length_only 参照行（chunk 48，先验 0.993，剩余预算 3）。
**这条 episode 的路由信号从头到尾没有异常。它不是"检测得晚"，是"根本没有可检测的东西"。**

### 案卷 B — 每一个头部方法都漏掉的失败
`libero_goal/open_the_top_drawer_and_put_the_bowl_inside` ep47，30/30，persistent failure，dropped。
9 个头部 routing 方法**全部沉默**，85 个里只有 3 个开火。最早的信号来自 length_only@5% 的 chunk 19（先验 0.561）——
也就是说，那个时刻能说的只有"还在跑"。

### 案卷 C — 一条 late success 上的集体误警
`libero_long/KITCHEN_SCENE8` ep8，52/52，**late_success_plus10**（再给 10 次 query 它就成功了），模式 timeout_holding。

| chunk | 先验 | 剩余预算 | 谁开火 |
|---|---|---|---|
| 8 | 0.069 | 43 | `conditional_query_d1\|global`（全场最早） |
| 21 | 0.079 | 30 | `twotier:WATCH_a3` |
| 33 | 0.488 | 18 | 跨帧三头 OR、`expert_load\|per_task` |
| 34 | 0.491 | 17 | combo G-A |
| 36 | 0.504 | 15 | v7 guard |
| 38 | 0.597 | 13 | `mobility\|global`、`twotier:ACT_b`、v4 dual regime |
| 48 | 0.993 | 3 | length_only 参照 |

33 个 routing 检测器都判它有危险。按本 corpus 的标签口径它们**全部算 TP**（risk = 没在 H 前结束）。
但物理上这条 rollout 只是**慢**，不是坏。41 条 late success 里 33 条被 v7 guard 抓到。
**如果 late success 的正确处置是"再给一点时间"而不是"中止"，那么这 33 次报警的价值取决于干预策略——
而干预在本 corpus 里没有被测过。这正是"检测 ≠ 部署"的具体形状。**

### 案卷 D — 一条几乎无人报警的 late success
同 task ep322，52/52，late_success_plus10，模式 no_grasp。85 个 routing 检测器只有 1 个开火
（`state:td_slope|global`，chunk 33，先验 0.488）。头部方法全部沉默。这条是"漏检"，但漏掉的其实是一次成功。

### 案卷 E — 早期正确报警长什么样
`libero_goal/put_the_bowl_on_the_plate` ep83，30/30，persistent failure，timeout_holding。

| chunk | 先验 | 剩余预算 | 谁开火 |
|---|---|---|---|
| **6** | **0.026** | 23 | `prior:HB_MoE_top4churn@1%` |
| 9 | 0.052 | 20 | length_only_task_devcal |
| 12 | 0.077 | 17 | combo P-C、`expert_load\|per_task` |
| 19 | 0.561 | 10 | v4 dual regime |

chunk 6，全池 15 600 条都还在跑，先验 0.026 —— 这才是"提前示警"的样子。
注意 length_only 在 chunk 9 就开火了：**这个 task 的成功 rollout 普遍很短，所以"跑到第 9 个 chunk"本身就异常**。

### 案卷 F — 一条 dropped，报警全部堆在最后
`libero_goal/open_the_top_drawer_and_put_the_bowl_inside` ep41，30/30，dropped。
37 个 routing 检测器开火，但 **v4 与 v7 在 chunk 25（先验 0.982，剩余预算 4）、combo P-C 在 26、
`mobility|global` 与 `twotier:WATCH_a3` 在 27（剩余预算 2）、`expert_load` 在 28（剩余预算 1）**。
先验此时已经 0.98–0.99。**这 37 次报警加起来提供的信息接近于零。**

### 案卷 G — 全场最贵的误警
`libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` ep174，
长度 22 = H，但**成功**（这就是那 4 条"触到上限仍然成功"的 episode 之一）。
42 个 routing 检测器在它上面开火，chunk 15 一次性来了 6 个头部方法（先验 0.952）。
**它是 length-only 规则的 4 条结构性误警之一，也是所有 MoE 方法的共同误警。**

### 案卷 H — regressed 模式，几乎完全不可见
`libero_long/KITCHEN_SCENE8` ep3，52/52，persistent failure，**regressed（置信度 high，26 条里仅有的 high 之一）**。
85 个 routing 检测器只有 3 个开火，最早 `v7:acceleration` chunk 37。头部方法全部沉默。

### 案卷 I — 一条被早早抓住的 dropped
`libero_goal/open_the_top_drawer_and_put_the_bowl_inside` ep147，30/30，dropped。
`prior:HB_MoE_top4churn@1%` 在 chunk 12（先验 0.077，剩余预算 17）开火；`expert_load|per_task` 在 19（0.561）；
combo P-C 在 20（0.763）；v4 与 v7 在 25（0.982）。**同一条 episode，最早与最晚的报警相差 13 个 chunk，
先验从 0.077 涨到 0.982。台账里"谁先说话"的差别就是这么大。**

其余 4 条（no_grasp / moved_unmet / released_outside / no_contact 各一条）见 JSON。

---

## 8. 与既有 bundle 的对账

### 8.1 与 `VLA_MUI_HUB/moe-failure-alarm` 的行级对账（成功）

它们的 36 098 条 trajectory 里，`run_id = right-50x8b-20260903` 的 16 000 条与我们的 15 600 条按
(task_name, episode_index) join：**15 600 / 15 600 全部命中，episode 长度 15 600 / 15 600 完全一致，
它们标的 564 条 failure 与我们的 564 条 risk 逐条相同（0 处不符）**。development 侧同样 14 800 / 14 800、487 / 487 全对。

因此它们的 alarm 被重新按**我们的**标签打分并放进台账（`prior:` 前缀，6 行）。

**口径差异必须说清，两边绝不合并：**

| | 他们 | 我们 |
|---|---|---|
| 范围 | 全部 89 个 source run，含 `pin-*` 与 `right-16x32` | 仅 `external_8b`（39 task × 400） |
| 正例 | 1 403 条物理 failure（observational 制度） | 564 条 original-horizon risk |
| 误报口径 | **逐 query** 校准 FPR（1% / 5% / 10%） | **逐 episode** timely-FPR（预算 0.005） |
| 阈值参照 | 同 run 同初始状态的成功对照，5 折交叉 | development corpus，冻结后外推 |

他们的运行点在他们的口径下：MoE alarm 5% 目标 → 4.7% FPR / 27.3% recall / precision 0.197 / episode AUC 0.821；
`length_only_same_run` → 2.8% FPR / **100.0% recall** / precision 0.600 / AUC 0.950。
我们独立复现了这个结论的方向：在**我们的**口径下 length_only 同样 100% recall（564/564）。

### 8.2 按失效模式的交叉核对

| 模式 | 他们 n（全 run） | 他们 recall | 我们 n（external_8b） | v7 guard | combo P-C | `mobility\|global` |
|---|---|---|---|---|---|---|
| dropped | 548 | 0.248 | 241 | 0.780 | 0.900 | 0.253 |
| no_grasp | 442 | 0.294 | 143 | 0.832 | 0.392 | 0.545 |
| moved_unmet | 200 | 0.265 | 89 | 0.674 | 0.607 | 0.303 |
| regressed | 65 | 0.123 | 26 | 0.846 | 0.692 | 0.462 |
| timeout_holding | 57 | 0.368 | 23 | 0.783 | 0.609 | 0.391 |
| released_outside | 48 | 0.438 | 25 | 0.640 | 0.960 | 0.120 |
| no_contact | 32 | 0.312 | 14 | 0.929 | 0.357 | 0.143 |

**分母不同，不能直接比大小。** 但方向上有一个共同点值得记：两边的 `regressed` 都是相对困难的模式
（他们 0.123 是全表最低；我们这边 `mobility|global` 在 26 条里只抓到 12 条），
而 `released_outside` 在两边都相对容易。**不共同的是 dropped**：他们 0.248，我们的 combo P-C 0.900。
这个差距的绝大部分来自口径（他们的报警集中在轨迹前半段、逐 query 校准；我们允许任意时刻报警），不构成"谁更好"。

### 8.3 `VLA_MUI_HUB/moe-physical-failure-dynamics`
本 bundle 只把它作为物理失效模式词表的来源使用（1 306 / 1 442 条 failure 有同 run 同初始状态的成功对照），
**没有重新分析它的匹配对照结果**。

---

## 9. 复现失败：我们定位到了根因

**问题**：`partial_edge_std | global` 被 `moe-hb-front-back-0905` 发布为 L3/high/**q0.90 → 99 TP / 147 FP**，
被 `moe-circuit-analogy-0906` 发布为 L3/high/**q0.95 → 42 TP / 35 FP**，两个 bundle 都声称用的是同一条冻结选择规则。

**我们复现了两边**（`results/bundle_disagreement.json`）：

1. 两个 bundle 的 external 数组我们都能逐位读出，计数分别精确等于 99/147 与 42/35。
2. 把 **frame-survey 的流水线跑在 circuit 选的那个分位数上**，得到的 external 计数**恰好是 42/35**。
   对 `conditional_effective_rank` 是 42/62、对 `partial_query_d1` 是 20/63，也都精确命中。
   **⇒ external 侧的特征值与阈值化机器在两个 bundle 之间是完全相同的。分歧全部在上游。**
3. 上游定位：重建 development 候选表，只换参照池（以 `partial_edge_std | L3 | high | global` 为例，TP/FP）：

| 参照池 | q0.90 | q0.925 | q0.95 | q0.96 | q0.975 | q0.98 | q0.99 |
|---|---|---|---|---|---|---|---|
| `development_main + development_extra`（16 000） | 69/40 | 47/20 | 20/10 | 12/6 | 6/2 | 3/2 | 0/0 |
| `development_main` only（14 800） | 128/141 | 83/67 | 51/22 | 34/18 | 18/7 | 12/6 | 6/2 |

第一行**逐格等于** `moe-hb-front-back-0905` 发布的 development 表；第二行**逐格等于** `moe-circuit-analogy-0906` 的。
三个头（`partial_edge_std`、`conditional_effective_rank`、`partial_query_d1`）**全部如此**。

> **根因**：两个 bundle 用了**不同的 development 参照池**做全局分位校准——frame survey 池化
> `development_main + development_extra`（16 000 条），circuit-analogy 只用 `development_main`（14 800 条）。
> 被留出的 1 200 条 `development_extra` episode 携带了这三个 `L3 | high` 量的极端上尾
> （extra-only 池在 q0.90 就已经 0 报警），去掉它们会把阈值压低，选择规则于是落在更严的分位数上。
>
> **两个数字都不是"错的"**：选择规则在给定参照池后是确定的，**参照池不同**。我们把两个变体都以不同名字放进台账
> （`partial_edge_std|global` 与 `circuitcopy:partial_edge_std|global`），**不挑边**。
>
> 12 个全局头里 11 个可比，**8 个逐位一致，3 个不一致，且这 3 个恰好就是全部的 `L3 | high` 头**。

其余复现结果：

- **v7 intrinsic guard 不是三个分支的 OR，是两个。** `external_freeze ∨ external_turbulence` **逐位等于**
  `external_guard`（439/80）。另外两个被打分的分支 `flow_acceleration`（289/205）与 `recurrence_loss`（341/1852）
  **从不进入 guard**。任务书里"v7 及其三个分支"应更正为"v7 及其四个被打分的分支，其中两个进入 guard"。
- `state_mobility_s9` 与 `state_mobility_mean` 在 `moe-state-channel-0906` 里是**逐位相同的两个数组**（两种模式都是），
  发布表也给出相同的 193/39 与 124/8。台账只收录前者，避免重复计数。
- `moe-v4-0904` 的 `cache_new_v4/sealed_first_alarms.npz` 与我们按
  `evaluate_layer_survival_baseline.py` 的重建**逐位相同**（external 与 development 两侧）。
- development 侧 v4 dual regime 的正确计数是 **372 / 67**（= risk_n 487 − fn 115）。

**没有任何一个方法因为无法复现而被排除。** 被跳过的只有 5 个新 bundle 缺少 development 孪生数组这一项，逐条记在 manifest。

---

## 10. 哪些能部署，哪些不能

**先划清 claim 的范围**：下面说的"可部署"**只指检测质量**——报警时机、正确率、失效模式覆盖。
**报警之后动手是否有帮助，这份台账完全没有测。** 日志录制时没有监视器在环，任何 rollout 都没有被打断过。
要回答那个问题必须从同一个仿真快照分叉多条续跑，缓存路由重放做不到。

### 看起来可以部署（就检测而言）

**1. `v7 intrinsic guard`（439/80，timely-FPR 0.00532）—— 唯一一个覆盖率没有硬空白的头部方法。**
最差模式覆盖 **0.640**（released_outside，n=25），是**全部 85 个 routing 方法里最高的**（第二是 WATCH_a3 的 0.565，
但它 FPR 超预算 2.6 倍）；`no_contact`（n=14）覆盖 13/14；漏检 125 条里没有任何模式被系统性放弃。
B=12 时仍保有 **221 TP / 71 FP**，是 timely-FPR ≤ 0.006 的方法里最高的。
budget_won 5 534（占上限 31.1%）。迁移最稳之一（FPR 膨胀 1.14×）。运行时**不需要任务标识**。

三条必须一起说的保留意见：

- **它轻微超预算。** timely-FPR 0.00532 是 0.005 预算的 106%；v4 dual regime 的 0.00539 是 108%。
  两者都不是"预算内"方法，只是超得不多。
- **它的报警偏晚。** libero_goal 上 74 次报警里 48 次发生在 chunk 22 之后，那时先验已 > 0.92；
  真正有信息的是早期那 18 次（18/18 全对，先验 0.13–0.22）。
- **它相对 v4 的优势在任务内不成立。** 39 个 task 里 6 个 v7 更高、15 个 v4 更高、18 个持平，p = 0.60。
  pooled 的 +29 TP 来自 task 组成，不是方法差异。

**2. 跨帧三头 OR（`mobility|global + flow_path|global + expert_load_effective_rank|global`，381/54，timely-FPR 0.0036）。**
预算内、precision 0.876、gain/loss 4.50、B=12 保有 186/47。全局阈值、无任务标识。
最差模式 0.357（no_contact，n=14）。它是"便宜且不太挑"的默认选择。

**3. `combo:P-C`（391/57，timely-FPR 0.00379）—— 早期带最好的预算内方法。**
早期 96 TP / 31 FP（precision 0.756）。在全部 70 个 timely-FPR ≤ 0.005 的 routing 方法里，
**它的早期 TP 是最高的**（第二名 `flow:mobility_s0|per_task` 75 TP / 39 FP），且早期 precision 仍在 0.75 以上。
但它有一个明确的结构性偏科：**no_grasp 只抓 56/143（0.392）**，而 dropped 抓 217/241（0.900）。
如果部署场景里 no_grasp 是主要风险，这个头不合适。

### 看起来不能部署

**4. `twotier:WATCH_a3`（410/197）—— 超预算 2.6 倍。** timely-FPR 0.0131 对 0.005 的预算。
它的覆盖率优势（最差模式 0.565）是靠 `partial_edge_std|global` 买来的，而那正是外推最差的三个头之一（3.50×），
并且**那个头本身就是跨 bundle 复现失败的对象**。不该按其 development 数字定预算。

**5. `twotier:ACT_b`（146/9）与 `mobility|global`（195/17）—— lift 最高，但有整块盲区。**
两者在 **libero_spatial 上 recall 恰好 0.000（0/140 risk）**；ACT_b 的 `no_contact` 覆盖 **0/14**。
两者报警中位 chunk 34–35，那时全池只剩约 550 条在跑。它们 0.92–0.94 的 precision 里有一半以上是生存基线直接给的
（匹配先验 0.51）。**作为"高置信升级层"可用，作为独立监视器会在一整个 suite 上失明。**

**6. `prior:HB_MoE_top4churn`（131/386 @1%）—— 唯一真正早的检测器，但 precision 0.25。**
它在 TP ≥ 20 的方法里 lift 最高（3.70），中位 chunk 9，早期 TP 116
（更高的只有它自己的 5% / 10% 变体与 `v7:periodicity`，那三个的早期 FP 分别是 690 / 1 313 / 1 798），
**并且它是本报告检验的配对里，早期带上任务内唯一显著优于 `mobility|global` 的方法**（p = 0.024）。
但 timely-FPR 0.0257 是预算的 5 倍。**它证明了早期信号确实存在，但当前形态不可部署。**

**7. 三个 `L3 | high` 全局头**（`partial_query_d1|global`、`conditional_effective_rank|global`、`partial_edge_std|global`）
—— 阈值不外推（FPR 膨胀 3.5×–7.8×），且是唯三跨 bundle 复现不一致的头。**任何情况下都不要单独部署。**

### 关于 length-only 参照行

它不在"可部署 / 不可部署"这条轴上，因为它不是检测器。它是 `original_failure` 定义的复述
（568 条触顶 / 564 条是 risk，precision 0.9930）。它的正确用法是作为**每一行 MoE 数字旁边的刻度**：
一个方法如果在 B=0 上赢不过它，说明它没有比"你已经跑太久了"多说任何东西；
一个方法如果在 B=12 上赢过它，说明它确实提前说了话。按这把尺，
**v7 guard（221 vs 73）、v4 dual regime（210 vs 73）、跨帧三头 OR（186 vs 73）在 B=12 上是 length-only 的 2.5–3 倍。**

---

## 11. 全部注意事项

1. **无干预效应。** 日志无监视器在环，`intervention_effect_measured: false`。台账只测检测。
2. **episode 是事后单元。** `length` 从不进入任何 score；台账的行单位不是运行时可得的。
3. **生存先验来自结果标签。** 它是"可知"块里唯一的结果派生量，是 per-suite、per-chunk 的常数，
   从不是 episode 级的量。这是本 corpus 的既定口径，此处照用并声明。
4. **物理失效原因置信度**：523 条 persistent failure 里 **497 条 medium / 26 条 high**；全部 564 条 physics_validation 通过。
5. **小样本模式**：no_contact n=14、timeout_holding n=23、released_outside n=25、regressed n=26、
   no_progress n=2、mechanism n=1。这些模式上的覆盖率差异不构成证据。
6. **`length_only` 的两种校准**：in-cohort 版本在它自己被打分的 episode 上拟合阈值（564/26），
   `devcal` 版本在 development 上拟合后原样迁移（564/45，其中 1 200 行因 task 不在 development 里而回退到 suite 级阈值），
   后者才是与 MoE 头公平的对照。两者都用结果标签定义校准集（只用成功样本），
   **这一点与所有 MoE 头不同**，已在方法 config 里注明。
7. **4 条 episode 触到 horizon 上限却成功**（goal 1、object 1、spatial 2），它们是 length-only 规则结构性误警的全部来源。
8. **`build:AND_mobility_flowsettling|per_task` 的 development 数字（183/1）不可用于定预算**：1 个 FP 的分母太小。
9. 5 个 2026-09-06 的新 bundle 只发布 external 数组，其阈值迁移**无法评估**。
10. 与 `moe-failure-alarm` 的对账是逐行的，但两边的**分母、正例定义、误报口径都不同，从不合并**。
11. **组合规则的重建口径**：G-A / G-B / P-C 通过 `moe-combination-rules-0906/experiments/combination_core.py`
    原样重建（latching / windowed quorum 与 cascade），WATCH / ACT 用 two-tier 的头数组按 OR 与 latching k-of-n 重建，
    三条 external 计数全部与发布值一致。

---

*本 bundle 的每一个数字都由 `experiments/` 下的脚本在本次运行中算出，没有任何数字是从别处抄来的。
凡是与既有 bundle 发布值对比过的，对比结果（相同或不同）都记录在 `results/manifest.json` 与
`results/bundle_disagreement.json` 中。*
