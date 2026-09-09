# 方法综合盘点：已建成的检测器"看得见什么"与"看不见什么"

`moe-method-synthesis-0906` · 2026-09-06 · 纯 CPU

本包不构建任何新检测器，不重新推导任何特征，不重新拟合任何阈值。所有 first-alarm 向量按各自 bundle 发布时的原样读入，本包只做对齐、归类与核算。选择一律在 `development_main`（14,800 集，487 个 risk）上完成，`external_8b`（15,600 集，564 个 risk）只评分一次；任何使用外部标签做选择的结果一律标注为 `ORACLE` / `IN_SAMPLE`，只作为上界。

约定：risk = `original_failure`；cap 为 goal 30 / long 52 / object 28 / spatial 22；in-window = 首次报警 chunk < 0.65 × cap；FP = 非 risk 集上的 in-window 报警；FPR = FP / 非 risk 集数。lift 一律用 **per-task** 生存先验（suite-matched 基准仅在 `detector_census.csv` 中并列保留，不用于任何结论）。本包不重拟合特征阈值，因此 `>` / `>=` 并列组问题在此不生效；所有计数型阈值（投票 k、预算 B、家族合并阈值）均取闭区间。

**长度是负对照，不是 baseline。** risk 的定义就是"没在 cap 前结束"，因此"长度达到 cap"按构造召回 100%（外部 564/564，仅 4 个 FP）。它在 `detector_census.csv` 中以 `is_baseline = False` 记录，排除在一切排序之外。

## 摘要：四个直接答案

**1. 真正独立的方法有多少个？** 两队列共有的 419 个检测器（414 个来自 7 个 bundle + 5 个 v7 封存流），按"在哪些 episode 上报警"聚类后坍缩为 **9 个家族**（development 9 个，external 独立导出 10 个，ARI **0.955**，置换零臂 ≈ 0.000）。速率匹配随机零臂给出 **363 个**家族（几乎全单例）。这 419 个检测器名义上号称 **17 种不同的量**——结构上只值 9 个。并入只在 external 上发布过向量的 4 个 bundle（共 622 个检测器、**105 种命名的量**）后为 **35 个**家族，其中 16 个是发射率 < 0.008 的单例。有效秩同样低：参与比 **12.47**、90% 方差需 **52 个**主成分，零臂为 24.14 / 144。**名字严重高估了多样性。**

**2. 抓不到的 risk 有什么共同点？** "在任何阈值下都没有任何方法抓到"的 risk 数量是 **0**——无约束并集在两队列召回率都是 1.000，代价是 FPR 0.83–0.85。未捕获集合完全是误报预算的函数。在与现有诚实臂同一工作点上（每检测器 dev FPR ≤ 0.01，9 家族并集 external recall 0.785 / 870 FP / FPR 0.0579，对照现有诚实臂 0.798 / 817 / 0.0543），**121 个 external risk（21.5%）无人捕获**：

- **suite**：libero_spatial 58/140 = **41.4%**（goal 21.7%，object 25.0%，long 仅 10.6%），未捕获集合的 47.9% 来自 spatial。
- **物理失效模式**：`object_released_or_dropped_before_goal` 贡献 **56.2%**（68/241）。
- **任务**：spatial `pick_up_the_black_bowl_on_the_wooden_cabinet…` 21 个 risk 中 17 个（81%）未捕获，`…in_the_top_drawer_of_the_wooden_cabinet…` 11/14（79%），`…on_the_stove…` 23/41（56%），goal `open_the_top_drawer_and_put_the_bowl_inside` 23/51（45%）。
- **长度：没有区别，且这一点是同义反复。** 每个 suite 内未捕获与被捕获 risk 平均长度完全相同（30.0/52.0/28.0/22.0），因为 risk 的定义就是跑满 cap。这正是长度只能当负对照的原因。
- **路由：它们看起来不异常。** 用已发布的 layerwise mobility 缓存做描述性对比（按任务分层置换，2,000 次）：被捕获 risk 相对成功 episode 在全部 8 层显著更低（z = −8.3…−27.4）；**未捕获 risk 相对成功 episode 在 L2/L12/L13/L14 无法区分**（z = −0.66/+1.24/−0.38/−1.63，p = 0.53/0.21/0.71/0.11），L3–L5 甚至偏向相反方向。它们唯一确实异常之处是初始状态：未捕获 risk 间共享 init state 概率 0.102，同任务被捕获 risk 匹配零臂 0.066（p < 0.0002）。

**3. 不可约的误报有什么共同点？** **libero_long 上跑得很长但最终成功的"险胜"，且集中在特定初始状态。** 870 个 external FP 中 **714 个（82.1%）只有一个家族报警**；被 ≥3 个家族同时报警的只有 **58 个**，其中 **91.4% 属于 libero_long**，长度为 cap 的 **0.707**（同任务重采样零臂 0.464，z = 38.4，p < 0.0002），同任务共享 init state 概率 **0.259**（零臂 0.018，14.6 倍，p < 0.0002）。≥5 票的 9 个更极端：88.9% long，44.4% 同一任务，长度 0.783 cap。只有 1 票的 714 个长度比 0.528（零臂 0.496），散布 38 个任务，init 集中度仅 0.069。

**4. 还有没有未被利用的互补性？没有。** 现有诚实臂 450 TP / 817 FP（0.798）；本包 dev 贪心 8 检测器诚实臂 463 TP / 852 FP（0.821）；**同一 419 个可诚实选择检测器上用外部答案全局搜索的 in-sample 上界只有 466–484 TP**，即最优组合相对现有诚实臂只多 **3–21 个 episode**，且是 in-sample。饱和曲线同向：前 5 个检测器 433 TP / 562 FP（1.30 FP/TP），第 6 个及以后共 45 TP / 725 FP（**16.1 FP/TP**，劣化 12.4 倍）；9 个家族代表中第 7–9 个 external 边际 TP 为 **0**。唯一余量不在"组合"里，而在**从未在 development 上评过分的 4 个 bundle**（circuit-analogy / flow-semantics / state-channel / token-geometry，203 个只有 external 向量的检测器）：并入后 in-sample 上界升至 504–516 TP，最多再多 41–53 个 episode，但**当前无法诚实取得**。

## 1. 覆盖账本

### 1.1 无约束并集：0 个 risk 对所有方法都不可见

| 队列 | 检测器数 | 并集 recall | 未捕获 | 并集 FP | FPR |
|---|---|---|---|---|---|
| external_8b | 622 | 1.000 | **0** | 12,817 | 0.852 |
| development_main | 419 | 1.000 | **0** | 11,845 | 0.828 |

达成它的代价是对 83–85% 安全 episode 报警。未捕获集合完全由误报预算决定。

### 1.2 预算阶梯（external，选择全在 development 上）

`budget` 是**单个检测器**在 development 上的 in-window FPR 上限；9 个家族代表 = 每族中该预算内 dev TP 最高的成员（并列按 FP、再按名字定序）；`ceiling` 为同预算下**全部**可用检测器的并集。

| budget | 9 家族 recall | 未捕获 | FP | FPR | precision | 仅 1 家族捕获 | 可用检测器 | ceiling recall | ceiling FP | ceiling 未捕获 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 | 0.512 | 275 | 195 | 0.0130 | 0.597 | 138 | 210 | 0.617 | 528 | 216 |
| 0.005 | 0.729 | 153 | 434 | 0.0289 | 0.486 | 144 | 270 | 0.807 | 1,122 | 109 |
| **0.010** | **0.785** | **121** | **870** | **0.0579** | **0.337** | **134** | **295** | **0.876** | **1,789** | **70** |
| 0.020 | 0.844 | 88 | 1,676 | 0.1115 | 0.221 | 116 | 326 | 0.947 | 3,577 | 30 |
| 0.0543 | 0.941 | 33 | 4,117 | 0.2738 | 0.114 | 95 | 365 | 0.972 | 6,641 | 16 |
| 0.100 | 0.966 | 19 | 6,618 | 0.4401 | 0.076 | 77 | 384 | 0.995 | 8,758 | 3 |
| 0.200 | 0.997 | 2 | 10,778 | 0.7168 | 0.050 | 24 | 403 | 1.000 | 11,976 | 0 |

粗体为参考工作点，与现有诚实臂几乎重合。development 同预算 0.799 / 734 FP / FPR 0.0513，与 external 0.785 的差即迁移损失。

### 1.3 只被一个方法抓到的 risk

443 个被捕获 external risk 中 **134 个（30.2%）只有一个家族抓到**，且**不是**同一家族反复贡献：

| 家族 | 代表检测器 | 独家捕获 | 占比 |
|---|---|---|---|
| 5 | `expert_load_effective_rank\|per_task\|0.7` | 39 | 29.1% |
| 2 | `conditional_query_d1\|global\|0.9` | 28 | 20.9% |
| 3 | `flow_path\|global\|0.925` | 23 | 17.2% |
| 4 | `flow_settling_log_ratio\|per_task\|0.7` | 20 | 14.9% |
| 9 | `v7_guard\|task_agnostic` | 18 | 13.4% |
| 1 | `conditional_energy\|global\|0.9` | 6 | 4.5% |

家族 6、7、8 一次独家捕获都没有。

### 1.4 饱和曲线（顺序在 development 上定死，external 只评分；预算 0.01，295 个可用检测器）

| 步 | 加入 | ext recall | ext FP | 边际 TP | 边际 FP |
|---|---|---|---|---|---|
| 1 | `expert_load_effective_rank\|per_task\|0.7` | 0.394 | 192 | 222 | 192 |
| 2 | `conditional_query_d1\|global\|0.9` | 0.585 | 311 | 108 | 119 |
| 3 | `moe-v7-0905::guard` | 0.663 | 342 | 44 | 31 |
| 4 | `flow_path\|global\|0.925` | 0.736 | 473 | 41 | 131 |
| 5 | `flow_settling_log_ratio\|per_task\|0.7` | 0.805 | 666 | 21 | 104 |
| 6 | `expert_load_effective_rank\|global\|0.85` | 0.818 | 793 | 7 | 127 |
| 7–16 | （10 个检测器） | 0.848 | 1,287 | 合计 38 | 合计 598 |

**前 5 步 433 TP / 562 FP = 1.30 FP per TP；第 6 步及以后 45 TP / 725 FP = 16.1 FP per TP。** 曲线在第 5–6 个方法处变平。9 个家族代表版本更干脆：第 7、8、9 个 external 边际 TP 全为 0，只带来 238 个 FP。

## 2. 误报账本

### 2.1 重叠结构：原始现象成立，但机制不是"误报彼此无关"

`concentration` = 各方法报警集大小之和 / 并集大小（1 = 完全不重合，9 = 完全一致）。`null_within` 保持每个方法在该分区内的报警条数不变、只打乱落点，隔离出"超出基础率之外的一致性"。external，参考工作点，9 个家族：

| suite | 分区 | concentration | null_within | 归一化超出 | 仅 1 个方法报警 |
|---|---|---|---|---|---|
| pooled | risk | 2.391 | 2.109 | 0.041 | 30.2% |
| pooled | safe | 1.291 | 1.032 | 0.032 | **82.1%** |
| goal | risk | 2.554 | 2.114 | 0.064 | 34.9% |
| goal | safe | 1.080 | 1.014 | 0.008 | 94.4% |
| long | risk | 2.584 | 2.430 | 0.023 | 22.5% |
| long | safe | 1.489 | 1.074 | **0.052** | 72.1% |
| object | risk | 2.061 | 1.772 | 0.040 | 39.4% |
| object | safe | 1.108 | 1.020 | 0.011 | 91.8% |
| spatial | risk | 1.780 | 1.445 | 0.044 | 45.1% |
| spatial | safe | 1.116 | 1.014 | 0.013 | 89.0% |

**原始观察成立**（真报警重叠 2.39 远高于假报警 1.29，82.1% 误报为单方法独有），**但机制需更正**：按分区内报警条数配平后，risk 侧超额一致性（0.041）与 safe 侧（0.032）几乎相同；在 libero_long 上 **safe 侧（0.052）反而高于 risk 侧（0.023）**。投票有用的原因主要是**基础率不同**——每个方法抓住约一半 risk（必然重叠），却只误报 2–6% 安全 episode（很难重叠）。

### 2.2 投票阈值 k 的分解（external，参考工作点，9 个家族）

| suite | k | TP | FP | recall | precision | 消除 FP | 丢失 TP | FP/TP | 召回保留率 |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 1 | 443 | 870 | 0.785 | 0.337 | – | – | – | – |
| pooled | 2 | 309 | 156 | 0.548 | 0.665 | 714 | 134 | **5.33** | 0.698 |
| pooled | 3 | 178 | 58 | 0.316 | 0.754 | 812 | 265 | 3.06 | 0.402 |
| goal | 2 | 54 | 7 | 0.509 | 0.885 | 118 | 29 | 4.07 | 0.651 |
| long | 2 | 190 | 118 | 0.693 | 0.617 | 305 | 55 | 5.55 | **0.776** |
| object | 2 | 20 | 13 | 0.455 | 0.606 | 145 | 13 | **11.15** | 0.606 |
| spatial | 2 | 45 | 18 | 0.321 | 0.714 | 146 | 37 | 3.95 | **0.549** |

k=1→2 被消除的 714 个 FP **100% 是"只有一票"的特异性误报**（按定义如此）；代价是同时丢掉 134 个真报警。**libero_long 召回保留率最高（0.776）、libero_spatial 最低（0.549）**；但按"每丢 1 个 TP 换掉多少 FP"衡量 libero_object 最高（11.15）。两个指标方向不同，如实并列。

用 5 方法臂（同预算 dev 贪心前 5）复核同向：long 的 risk 侧独家比例 0.343（k≥2 保留 65.7%），goal 0.750、object 0.773（k≥2 仅保留 25%、23%）。**方法越少，投票对"每方法召回率低"的 suite 破坏越大**——这正是 object/goal 上投票毁掉召回的直接原因。

### 2.3 不可约误报的刻画（external，参考工作点）

| 组 | n | 占安全集 | 长度/cap | 同任务零臂 | z | 共享 init state | 零臂 | 主 suite | 任务数 |
|---|---|---|---|---|---|---|---|---|---|
| 仅 1 票 | 714 | 4.75% | 0.528 | 0.496 | 19.5 | 0.069 | 0.019 | long 42.7% | 38 |
| 恰 2 票 | 98 | 0.65% | 0.628 | 0.508 | 24.6 | 0.143 | 0.020 | long 66.3% | 22 |
| ≥3 票 | 58 | 0.39% | **0.707** | 0.464 | **38.4** | **0.259** | 0.018 | long **91.4%** | 13 |
| ≥5 票 | 9 | 0.06% | 0.783 | – | – | – | – | long 88.9% | 5 |
| 从未报警的安全集 | 14,166 | 94.2% | 0.450 | – | – | – | – | goal 26.6% | 39 |

两条判断均通过同任务重采样零臂，非组成假象。任务层面，`LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` 的 ≥3 票误报率最高（8/396 = 2.0%，该任务 risk 率仅 1.0%；安全 episode 平均长度 25.2，被多方法报警者 41.9）。**投票能免费拿掉的是那 82.1% 特异性误报；剩下 58–156 个不可约误报本质是"表现得像失败但侥幸成功"的 episode，任何只读路由的方法都无法把它们与真失败分开。**

## 3. 方法家族

### 3.1 聚类协议

相似度**不看名字**，只看 in-window 报警集合，报告三种：`jaccard`（衡量**工作点**是否相同；用它在 419 个上按 0.5 合并得 **128 个"家族"**，但那测的是阈值粒度）；`excess_overlap`（**headline**）= |A∩B|/min(|A|,|B|) − max(|A|,|B|)/n，嵌套集合得 1，阈值阶梯坍缩为一族，且"几乎对所有 episode 报警"的检测器不会白得分数；`excess_jaccard` 作稳健性臂。零臂保持每检测器**每 suite 内**发射率不变随机重排。

### 3.2 419 个共有检测器 → 9 个家族

| 合并阈值 | 观测 | 零臂 |
|---|---|---|
| 0.2 | 3 | 277 |
| 0.3 | 6 | 321 |
| 0.4 | 9 | 350 |
| **0.5** | **9** | **363** |
| 0.6 | 15 | 383 |
| 0.7 | 20 | 395 |
| 0.8 | 30 | 399 |

家族数在 0.4–0.5 稳定于 9；零臂全区间接近全单例。dev 与 ext 独立导出的划分 ARI = **0.955**（置换零臂 −0.0001）。

### 3.3 9 个家族（development 上导出）

| 家族 | 成员 | 覆盖的命名量（root） | 归一化范围 | 最好成员 dev TP / FP |
|---|---|---|---|---|
| 9 | 124 | action_consensus, conditional_effective_rank, conditional_query_d1, mobility, partial_edge_std, partial_query_d1, v7_guard, v7_freeze | 主要 per_task（85/124） | 289 / 2171 |
| 4 | 68 | action_consensus, flow_endpoint, flow_path, flow_settling_log_ratio | 全 per_task | 231 / 902 |
| 2 | 64 | conditional_effective_rank, conditional_query_d1, partial_edge_std, partial_query_d1 | 全 global | 312 / 2247 |
| 1 | 34 | conditional_energy, state_action_alignment | 全 global | 228 / 2421 |
| 3 | 34 | flow_endpoint, flow_path | 全 global | 258 / 4630 |
| 5 | 34 | expert_load_effective_rank | global + per_task 各 17 | 338 / 4558 |
| 7 | 34 | conditional_energy, state_action_alignment | 全 per_task | 82 / 992 |
| 8 | 13 | flow_settling_log_ratio | 全 global | 81 / 2671 |
| 6 | 6 | v7_acceleration, v7_periodicity, v7_turbulence | task_agnostic | 166 / 1859 |

两个结构性事实：

1. **17 个命名的量只对应 9 个家族**，9 个家族中有 7 个跨越多个命名 root，364/411 个检测器落在跨命名家族里。`conditional_energy` 与 `state_action_alignment` 是同一个东西（家族 1、7）；`flow_endpoint` 与 `flow_path` 是同一个东西（家族 3）；`conditional_effective_rank` / `conditional_query_d1` / `partial_edge_std` / `partial_query_d1` 四个名字是同一个东西（家族 2）。**唯一自成一族的命名量是 `expert_load_effective_rank`（家族 5）**，也是单检测器 TP 最高者。
2. **归一化范围（global vs per_task）比"测的是哪个路由量"更能划分家族。** 家族 1/7 与 3/4/8 的分裂完全沿这条线。已发布的"多种路由量"在很大程度上是**同一份底层信号 × 两种归一化 × 15 档阈值**。

有效秩独立佐证：参与比 **12.47**、90% 方差 52 个主成分、首主成分占 19.6%；零臂 24.14 / 144 / 17.3%。观测矩阵维度**低于**同发射率随机矩阵，只可能来自真实共线性。

**逐 suite（池化会掩盖）**：libero_goal 上 419 个检测器只剩 **5 个**家族（最大一族 133 个成员，零臂 288）；long 17 个、object 14 个、spatial **20 个**。**spatial 是方法最"分歧"的 suite，也正是未捕获 risk 最集中的 suite——分歧多并没有换来覆盖。**

### 3.4 622 个检测器 → 35 个家族

并入 circuit-analogy（77）、flow-semantics（60）、state-channel（38）、token-geometry（28）后，**105 个命名的量**坍缩为 **35 个**家族（零臂 558），16 个是发射率 < 0.008 的单例。最大一族 220 个成员、跨 46 个命名 root、跨 8 个 bundle。**跨 bundle 的重名比 bundle 内部更严重**：各自发明的电路量、token 几何量、状态通道量绝大多数落进与 `mobility` / `partial_edge_std` 相同的家族。

**最直白的证据**：622 个检测器只有 **485 个互不相同的报警向量**，187 个与另一个**逐位相同**；development 上 419 个只有 322 个互不相同，137 个逐位重复。

## 4. 盲区矩阵：家族 × 物理失效模式

物理失效模式来自独立的 36,098 条轨迹重放，按 (run_id, suite, task, episode) 精确连接，覆盖两队列**全部** risk 且无任何安全 episode 带标签（`tests/test_synthesis.py` 有断言）。模式与 suite 强烈混杂，故每格同时给出 suite 内版本。

### 4.1 各模式的覆盖（external）

| 物理失效模式 | n | 9 家族并集 | 同预算全检测器 ceiling | 无约束并集 | suite 分布 |
|---|---|---|---|---|---|
| object_released_or_dropped_before_goal | 241 | **0.718** | **0.801** | 1.000 | goal 81 / long 50 / object 19 / spatial 91 |
| stable_grasp_not_observed | 143 | 0.804 | 0.944 | 1.000 | long 112 / spatial 26 / goal 5 |
| object_moved_but_goal_unmet | 89 | 0.809 | 0.910 | 1.000 | long 49 / object 15 / spatial 14 / goal 11 |
| goal_predicate_regressed | 26 | 0.923 | 0.962 | 1.000 | long 26 |
| object_released_outside_goal | 25 | 0.880 | 0.920 | 1.000 | 四个 suite 都有 |
| timeout_while_holding_target | 23 | 0.957 | 0.957 | 1.000 | long 16 / object 6 / goal 1 |
| approached_target_without_observed_contact | 14 | 0.857 | 0.857 | 1.000 | long 11 / spatial 3 |
| no_meaningful_target_progress | 2 | 1.000 | 1.000 | 1.000 | long 2 |
| mechanism_threshold_not_reached | 1 | 1.000 | 1.000 | 1.000 | long 1 |

**没有任何一个模式完全无人覆盖**。参考工作点上最差的是 `object_released_or_dropped_before_goal`（0.718）——"抓到了但在到达目标前掉了"，它同时是最大模式（占 external risk 的 42.7%）和未捕获集合的主体（56.2%）。第二差的 `stable_grasp_not_observed`（0.804）ceiling 是 0.944：**它不是本质盲区，而是被 9 家族这层精简砍掉的**。

### 4.2 家族层面的系统性遗漏

以"该家族在同 suite 的整体召回"为基准、在该 suite 的 risk 内置换模式标签（4,000 次），显著低于自身基准（p ≤ 0.05，格内 ≥10 个 risk）：

| 家族 | suite | 模式 | n | 该格 recall | 该 suite recall | p |
|---|---|---|---|---|---|---|
| 3 (flow_*, global) | long | stable_grasp_not_observed | 112 | 0.045 | 0.124 | 0.001 |
| 3 | spatial | stable_grasp_not_observed | 26 | 0.154 | 0.450 | 0.001 |
| 4 (flow_*, per_task) | long | stable_grasp_not_observed | 112 | 0.107 | 0.215 | <0.001 |
| 4 | spatial | stable_grasp_not_observed | 26 | 0.038 | 0.243 | 0.004 |
| 4 | goal | object_released_or_dropped_before_goal | 81 | 0.444 | 0.519 | 0.004 |
| 5 (expert_load_effective_rank) | long | stable_grasp_not_observed | 112 | 0.214 | 0.325 | 0.001 |
| 5 | spatial | object_moved_but_goal_unmet | 14 | 0.071 | 0.307 | 0.033 |
| 6 (v7 intrinsic) | long | stable_grasp_not_observed | 112 | 0.205 | 0.296 | 0.003 |
| 2 (conditional/partial, global) | long | object_released_or_dropped_before_goal | 50 | 0.400 | 0.529 | 0.034 |
| 9 (混合 per_task) | long | approached_target_without_observed_contact | 11 | 0.182 | 0.580 | 0.007 |
| **并集** | goal | object_released_or_dropped_before_goal | 81 | 0.716 | 0.783 | <0.001 |
| **并集** | long | object_released_or_dropped_before_goal | 50 | 0.760 | 0.894 | 0.001 |
| **并集** | spatial | object_moved_but_goal_unmet | 14 | 0.214 | 0.586 | 0.004 |
| **并集** | spatial | stable_grasp_not_observed | 26 | 0.346 | 0.586 | 0.008 |

**`stable_grasp_not_observed`（从未建立稳定抓握）是几乎所有家族的共同盲区**——家族 3、4、5、6 在 long 与 spatial 上都显著低于自身基准；只有家族 2 与 9（conditional/partial 系列，尤其 `v7_guard`）在该模式上不掉。并集层面剩下的系统性盲区是两条：**goal / long 上的"掉落"** 与 **spatial 上的"动了但没达成 / 没建立抓握"**——正是第 1 节那 121 个未捕获 risk 的主体。

物理解释自洽：这些方法读的是**路由的动力学异常**。当策略在"抓"这一步就从未稳定过，或物体半途脱手，路由本身并不需要变得异常——策略只是在继续执行一个从一开始就注定不成的计划。这与第 1.2 节的路由对比是同一件事。

## 5. 还剩多少互补性

### 5.1 家族两两互补（external，参考工作点）

家族间真报警 Jaccard 中位数：pooled 0.122、goal 0.091、long 0.106、object 0.087、**spatial 0.000**；假报警 Jaccard 中位数 pooled 0.025、long 0.044、其余三个 suite **0.000**。**"两个家族真报警重叠低、假报警重叠也低"这种配置到处都是**——尤其 spatial 上是普遍情况。这恰恰说明它**不是**免费午餐：假报警重叠低意味着并集 FP 近似相加。最好一对（家族 5 + 9）并集 326 TP / 235 FP，相对更强的单族只多 104 TP、多 43 FP（2.42 TP/FP）；第二好的一对已跌到 1.04 TP/FP，第五对 0.70 TP/FP。

### 5.2 与外部误报数配平的上界

左半为 in-sample 上界（用外部答案选择，不可诚实取得）：

| external FP | 最好单检测器 | 最好 OR 二元组 | 419 上界 | 622 上界 | 速率匹配零臂 |
|---|---|---|---|---|---|
| 100 | 185 | 256 | 279 | 321 | 19.0 |
| 200 | 222 | 317 | 348 | 401 | 30.8 |
| 400 | 246 | 343 | 420 | 473 | 52.2 |
| 600 | 267 | 368 | 448 | 494 | 66.0 |
| **800** | **267** | **384** | **466** | **504** | **80.8** |
| 1,000 | 267 | 398 | 484 | 516 | 99.0 |
| 2,000 | 295 | 460 | 529 | 541 | 165.2 |

诚实臂（选择全在 development 上）：

| 臂 | 方法数 | ext TP | ext FP | recall | FPR | precision | 相对 419 上界 | 相对 622 上界 |
|---|---|---|---|---|---|---|---|---|
| 现有诚实臂（外部给定） | – | 450 | 817 | 0.798 | 0.0543 | 0.355 | – | – |
| 9 家族并集 @0.01 | 9 | 443 | 870 | 0.785 | 0.0579 | 0.337 | 23–41 | 61–73 |
| **dev 贪心 8 @0.01** | **8** | **463** | **852** | **0.821** | **0.0567** | **0.352** | **3–21** | **41–53** |
| dev 贪心 8 @0.02 | 8 | 511 | 1,674 | 0.906 | 0.111 | 0.234 | 7–18 | 20–30 |
| dev 贪心 8 @0.0543 | 8 | 540 | 4,457 | 0.957 | 0.296 | 0.108 | 2–20 | 11–21 |

（差距上下界来自 FP 刻度粒度：诚实臂落在 852 FP，上界只在 800 与 1,000 两刻度计算。零臂在 800 FP 只到 80.8 TP，故这些上界远高于随机，互补性真实存在、**只是已经被用掉了**。）

### 5.3 结论：组合层面没有剩余空间

三条互相独立的证据指向同一答案：

1. **上界差距**：同一批可诚实选择的 419 个检测器上，把"如何组合"做到 in-sample 最优只比现有诚实臂多 3–21 个 episode。
2. **饱和**：第 6 个方法之后边际交换率从 1.30 FP/TP 劣化到 16.1 FP/TP；9 个家族代表中后 3 个 external 边际 TP 为 0。
3. **家族结构**：419 个检测器只有 9 个家族、有效秩 12.5、90% 方差 52 个主成分——可组合的独立方向本来就不多，贪心已在前 5–6 步用完。

**唯一还剩的余量在别处**：circuit-analogy / flow-semantics / state-channel / token-geometry 这 4 个 bundle 的 203 个检测器**从未在 development 上产出向量**，无法进入任何诚实选择流程。并入后 in-sample 上界从 466–484 升到 504–516，最多再多 41–53 个 episode（约 7–9 个召回点）。该数字是 in-sample 的、用 27–31 个检测器在 15,600 集上搜索出来的，过拟合成分不小，应读作**上界的上界**。把它变成真实收益的唯一途径是**在 development 队列上重跑这 4 个 bundle 的特征、发布配套 first-alarm 向量**——这是数据生产工作，不是算法工作。

超出该上界之外——即使拿到全部 622 个检测器、用外部答案搜索，800 FP 预算下仍有 **60 个 risk（10.6%）** 抓不到——是路由表示本身的边界。原因见第 1.2 节：这些 episode 的路由与成功 episode 无法区分。

## 6. 已知局限

- **未捕获 risk 的"路由无异常"结论只在已发布的 layerwise mobility 这一个表示上成立。** 本包按约束不重新推导特征，不能排除"另一种从未被计算过的路由统计量能分开它们"；能说的是现有 622 个检测器所张成的表示看不见它们。
- **622 检测器臂的一切数字都是 in-sample 的**，其贡献只能作为上界。
- **家族划分依赖 in-window 报警集合，因此依赖 0.65 × cap 这个窗口**；0.4–0.5 合并阈区间上家族数稳定在 9，但换窗口未做敏感性分析。
- **不可约误报的 init-state 检验按同任务重采样，未按 flow_noise_seed 分层。**
- **未捕获 vs 被捕获 risk 的 init-state 对比中重采样池被被捕获 risk 数量所限**（121 个未捕获只能对 93 个配平），该 p 值偏乐观。
- **投票分解在 k=2 处"被消除的 FP 100% 是特异性 FP"是定义使然**，不是发现；有信息量的是同时丢失的 TP 数与逐 suite 召回保留率。
- 本包未读取、未写入、未依赖 `moe-unified-detector-0906/`。

## 7. 产物

`results/`（36 个文件，`manifest.json` 含 sha256 与 18 个输入清单）：

| 文件 | 内容 |
|---|---|
| `key_numbers.json` | 本报告引用的**全部**数字，由 `summarise.py` 从下列表格回读生成 |
| `detector_census.csv` | 622 + 419 个检测器逐个的 TP/FP/recall/FPR/precision、task-matched 与 suite-matched lift、逐 suite 拆分、负对照行 |
| `duplicate_groups.csv` | 逐位相同的报警向量分组 |
| `families.csv` / `families.json` / `family_cut_profile.csv` / `family_labels.npz` | 家族划分、阈值敏感性、逐 suite 版本、零臂、有效秩、dev→ext ARI |
| `coverage_ledger.csv` / `uncaught_profile.csv` / `sole_catcher.csv` | 第 1 节 |
| `episode_ledger_{cohort}.csv.gz` | 逐 episode 的家族命中位图（参考工作点） |
| `uncaught_routing_contrast.csv` | 三组分层置换对比 |
| `saturation_curve.csv` | 贪心饱和曲线，2 臂 × 8 预算，dev 与 ext 并列 |
| `alarm_overlap.csv` / `vote_decomposition.csv` / `fp_profile.csv` / `heavy_fp_by_task.csv` | 第 2 节 |
| `irreducible_fp_tests.csv` | 同任务重采样零臂下的长度与 init-state 检验 |
| `blindspot_matrix.csv` / `blindspot.json` | 第 4 节 |
| `family_pair_complementarity.csv` / `combination_frontier.csv` / `combination_null.csv` | 第 5.1 节 |
| `headroom_ceiling.csv` / `headroom_null.csv` / `honest_operating_points.csv` / `honesty_gap.csv` | 第 5.2 节 |
| `synthesis_overview.png` | 饱和 / 上界 / 投票 / 家族数四联图 |

`experiments/` 共 11 个脚本，依赖顺序：`census.py` → `families.py` → `coverage_ledger.py` → `fp_ledger.py` → `blindspot.py` → `complementarity.py` → `irreducible_fp.py` → `headroom.py` → `summarise.py` → `make_figure.py` → `write_manifest.py`。

`tests/test_synthesis.py`：13 项不变量断言全部通过。其中三项直接守护本报告的诚实性——(a) 长度负对照按构造召回 100% 且 `is_baseline=False`；(b) 家族代表选择在打乱 external 标签后完全不变（证明选择未使用外部结果）；(c) in-sample 上界不得被任何诚实臂超过。
