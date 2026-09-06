# 组合规则：把弱 MoE 失败检测器的并集覆盖真正取出来

外部队列 15,600 条、564 条 risk。所有数字来自 `results/` 下的机器可读产物，脚本在 `experiments/`。
预注册文档 `results/PREREGISTRATION.json` 在打开 external 之前写好；开发集选择由
`select_on_development.py` 冻结成 `results/shortlist.json`，external 只被 `evaluate_external.py` 评了一次。

## 1. 结论先行

**有效，而且有效的原因不是组合的逻辑结构，是标定层级。**

之前的 OR 之所以假警报近似相加，是因为 12 个检测器各自被调到各自最优的分位点（0.85~0.99 混杂）。
把它们**全部换到同一个严格尾部**再组合，OR 立刻不再是垃圾：

| global 模式（不使用任务身份，这一栏才证明信号在 MoE 侧） | TP | FP | 精确率 | 召回 | 早期 TP | 早期 FP | 平均报警先验 | lift |
|---|---|---|---|---|---|---|---|---|
| 现有最好单检测器 `mobility` | 195 | 17 | 0.920 | 0.346 | 36 | 4 | 0.521 | 1.764 |
| 现有最好两两 AND `conditional_query_d1+mobility` | 140 | 9 | 0.940 | 0.248 | 10 | 0 | 0.560 | 1.679 |
| **冻结规则 G-A**：0.99 尾部至少 1 票 + 0.85 尾部至少 2 票（跨 ≥2 参考系，latching） | **288** | **13** | 0.957 | 0.511 | 24 | 6 | 0.715 | 1.338 |
| **冻结规则 G-B**：12 个量统一取 0.99 全局分位后做纯 OR（k=1） | **308** | **23** | 0.931 | 0.546 | 34 | 13 | 0.678 | 1.372 |
| 冻结规则 G-C：0.98 尾部至少 1 票 + 0.80 尾部至少 2 票（跨 ≥2 参考系） | 379 | 65 | 0.854 | 0.672 | 51 | 47 | 0.592 | 1.441 |

- **G-A 对现有最好单检测器构成严格支配**：TP 288 > 195，FP 13 < 17。召回从 0.346 提到 0.511。
- G-B 用 1.35 倍的假警报（23 vs 17）换 1.58 倍的真阳（308 vs 195）。
- 对现有最好 AND（140/9），G-A 是 **+148 TP / +4 FP**。

per_task 模式（使用任务身份，只作参考，不构成 MoE 侧证据）：

| per_task 模式 | TP | FP | 精确率 | 召回 | 早期 TP | 早期 FP | lift |
|---|---|---|---|---|---|---|---|
| 现有最好单检测器 `expert_load_effective_rank` | 370 | 93 | 0.799 | 0.656 | 97 | 69 | 1.548 |
| 现有最好 AND `expert_load+mobility` | 245 | 14 | 0.946 | 0.434 | 53 | 6 | 1.343 |
| 冻结规则 P-A：11 量同任务 0.80 分位，任意 5 票（latching） | 352 | 31 | 0.919 | 0.624 | 68 | 11 | 1.309 |
| **冻结规则 P-C**：11 量同任务 0.80 分位，任意 4 票（latching） | **391** | **57** | 0.873 | 0.693 | **96** | **31** | 1.450 |
| 冻结规则 P-D：0.96 尾部 1 票 + 0.925 尾部 2 票，窗口 W=2 | 287 | 56 | 0.837 | 0.509 | 96 | 30 | 1.554 |

- **P-C 同样严格支配现有最好单检测器**：TP 391 > 370，FP 57 < 93；早期带 96 TP / 31 FP 对 97 TP / 69 FP，
  早期真阳基本持平而早期假警报只有一半多一点。

开发集→外部集的迁移非常稳：以 recall 计，G-A 0.511→0.511，G-B 0.548→0.546，P-C 0.696→0.693
（`results/development_to_external_transfer.csv`）。假警报率放大 1.1~2.2 倍。

## 2. 预注册的目标与规则族

在打开 external 之前声明（`results/PREREGISTRATION.json`）：

- **检测器池**：帧调查（`moe-hb-front-back-0905/results/frame_survey`）选出的 12 个路由量，
  representation / direction 原样照抄，不做新的表示搜索；只扫描一个**共享的敏感度层级 s**。
  共享层级是关键：在层级 s 上每个成员都被同一个无标注参考峰值分布的同一分位数阈值化，
  于是「k 票」才是「有多少个独立路由视角把这条 episode 放进同一个 (1−s) 尾部」。
- **规则族**（都是显式计数/顺序/窗口，零学习参数）：
  - `quorum`：在长度 W+1 的滑动窗口内至少 k 个成员投票，且这些成员跨越至少 f 个参考系。
  - `cascade`：另加「至少 1 票来自更严格的层级 s_hi」，其余 k−1 票只需宽松层级 s_lo。
  - k∈{1..5}，W∈{0,1,2,4,8,∞}，f∈{1,2}，池∈{all12, dedup11}，s 取 15 个层级。
  - k=1 就是纯 OR；两成员池上 k=2、W=∞ 就是现有的两两 AND。**现有操作点是这个族的特例**，比较是同尺度的。
- **目标（两个都预注册）**：O1 最大化开发集 low_prior_tp；O2 最大化开发集 tp。
  约束：开发集 low_prior_precision ≥ 0.60，开发集 timely_fpr ≤ {0.0005, 0.001, 0.0015, 0.003, 0.005} 五档。
- **选择**：11,676 条组合规则 + 360 个单检测器 + 132 个两两 AND 全部在 development_main 上打分；
  每个 (模式, 池, 目标, FPR 上限) 取一条开发集冠军 → 40 条（32 条不重复）冻结进 shortlist；external 只跑这一次。

**正确性验证**：`build_detector_alarms.py` 重建的 24 个「selected」操作点在 external 上与帧调查的
封存数组 `external_first_alarms.npz` **逐条完全一致**，否则脚本报错退出。
`test_combination_core.py` 用一份独立的循环实现对 quorum 定义做了逐 episode 复核，并验证
k=1 等于 OR、两成员上 k=2/W=∞ 等于两两 AND、12 票 latching 等于全体 AND。
生存先验的 0.25 交叉点在 external 上算出来是 goal 18 / long 26 / object 17 / spatial 13，与任务说明一致。

## 3. 为什么会赢：不是逻辑结构，是跨 suite 覆盖

global 模式下每个量的报警高度按 suite 分化（`results/external_alarm_timing.csv`，
risk 数 goal 106 / long 274 / object 44 / spatial 140）：

| 检测器 | goal TP | long TP | object TP | spatial TP |
|---|---|---|---|---|
| `mobility` \| global | 37 | 156 | 2 | **0** |
| `expert_load_effective_rank` \| global | **0** | 101 | **0** | 136 |
| G-B（0.99 统一尾部 OR） | 30 | 145 | 6 | 127 |
| G-A | 23 | 139 | 1 | 125 |

现有最好的 global 单检测器在 libero_spatial 上是 **0/140**。组合规则拿回来的「并集覆盖」绝大部分就是
这块 suite 覆盖，而且是在**一个池化的全局阈值**下拿回来的，没有用任务身份。

**quorum/cascade 结构本身贡献多少？** 把每条冻结规则和同池纯 OR 敏感度曲线在同一 FP 数上做线性插值比较
（`results/headline_comparison.csv` 的 `tp_above_or_curve`）：

- global：G-A 在 13 FP 处比 OR 曲线高 **+62.7 TP**；G-B 就是 OR 曲线本身（+0）；G-C 只高 +6.7。
- per_task：P-A 在 31 FP 处高 **+172.8 TP**，P-B 高 +163.3，P-C 高 +128.1。

即：**per_task 模式下 k-of-n 结构做了大量真实工作；global 模式下结构只在极低假警报角落有用，
其余部分几乎全是「统一严格标定 + OR」的功劳。**

## 4. 窗口没有买到提前量（预期落空的一条）

任务书的假设是：「AND 取两者中较晚的时刻，把报警推进高先验区；一个能保留较早时刻又要求佐证的规则会严格更好。」
因果性不允许在佐证到达前报警，所以我实现的是**有界确认窗口**：k 票必须挤在 W+1 个 chunk 内，
于是报警最多落在这 k 票中第一票之后 W 个 chunk。

开发集上做匹配消融（只差 W、其余全同的规则配对，`results/development_window_ablation.csv`）：

| W | 匹配对数 | 窗口版平均先验中位数 | latching 版 | 先验下降中位数 | 先验更低的比例 | TP 比中位数 | 早期 TP 差中位数 | **早期 TP 变多的比例** |
|---|---|---|---|---|---|---|---|---|
| 0 | 1812 | 0.496 | 0.602 | 0.058 | 74.9% | 0.158 | −36 | **0.0%** |
| 1 | 1882 | 0.530 | 0.607 | 0.031 | 77.5% | 0.470 | −13 | **0.0%** |
| 2 | 1894 | 0.556 | 0.608 | 0.023 | 79.5% | 0.631 | −7 | **0.0%** |
| 4 | 1906 | 0.578 | 0.610 | 0.017 | 90.1% | 0.789 | −3 | **0.0%** |
| 8 | 1918 | 0.594 | 0.610 | 0.011 | 92.8% | 0.905 | 0 | **0.0%** |

**结论：有界窗口确实降低了平均报警先验（75~93% 的配对下降），但降低的方式是丢掉佐证来得晚的 episode，
不是把同一批检测挪早。没有任何一个匹配配对因为加窗口而增加了早期真阳（比例恒为 0.0%）。**
在匹配的开发集 FPR 上限下，latching 在 tp 上一律不输窗口版（`results/development_window_pareto.csv`）。
所以「保留较早报警时刻同时要求佐证」这条路，在这个数据上是**做不到**的：佐证时刻就是最早可判定时刻，
窗口只能当成一个 episode 筛子用。

唯一一个窗口有实质价值的位置是 per_task 的 P-D（cascade 0.96→0.925, k=2, W=2）：
外部 287 TP / 56 FP，早期 96/30，平均报警先验 0.539、lift 1.554；对照 latching 的
`expert_load+mobility` AND 平均先验 0.704、lift 1.343。窗口把报警拉回了低先验区，代价是精确率。

## 5. 负面结果

1. **global 模式下没有任何组合改进「早期带」。** 早期 Pareto 前沿上，`mobility|global` 的
   36 早期 TP / 4 早期 FP 仍然压过所有冻结规则（最接近的 G-A 是 24/6）。
   lift 从 mobility 的 1.764 掉到 1.338~1.464。**global 拿回来的覆盖全部在高先验（较晚）区间。**
   按项目的评分口径必须明说：它没有把预警提前。

2. **early 目标（O1）在 global 模式下选出的规则迁移失败。**
   `global|*|O1_early|fpr<=0.0015` 开发集 152 TP / 18 FP，外部 **183 TP / 130 FP**，FPR 放大 6.9 倍。
   原因可查：这条规则用的是 `selected` 层级池，其中三个 global 检测器本身就不迁移——
   `partial_query_d1` 65/173、`conditional_effective_rank` 88/158、`partial_edge_std` 99/147
   （开发集上分别只有 40/21、42/20、69/40）。**组合规则修不好一个不迁移的成员检测器。**
   这三个恰好都是 L3-high 家族，是一个可以单独追的线索。

3. **在 global 模式，赢的那条规则是纯 OR。** k-of-n、跨参考系、窗口这些机制在 global 只在 FP≤15 的
   角落里有额外收益。准确的说法是「发现了标定层级是主变量」，而不是「发明了一个组合规则」。

4. **并集前提我没能精确复现。** 在这 24 个检测器里：全部 24 个的并集是 534 TP / 530 FP（漏 30，94.7%）；
   按 external 覆盖率事后挑最好的 4 个是 519 TP / 306 FP（漏 45，92.0%，**这是事后选的，只当上界引用**）；
   任务书给的「四检测器并集 525/564、只漏 39」我没有在本池内复现出来。数量级一致，具体组合不同。

   > 控制方补注：该并集为 `L2 mobility(per_task) OR all_median mobility(per_task) OR
   > flow_settling(per_task) OR v7 guard`，v7 不在本 agent 的检测器池内，且其中三项是 per_task 模式，
   > 因此本 agent 复现不出属于预期。控制方复算该并集为 **525 TP / 198 FP，召回 93.1%，精度 0.726，漏 39**。
   > 原始表述只报了覆盖率而未报 198 个假警报的代价，是误导性的。

5. **`state_action_alignment` 与 `conditional_energy` 在选定操作点上几乎是同一个检测器**
   （external 上 global 差 1 条、per_task 差 2 条 episode）。因此预注册就准备了 `dedup11` 池。

## 6. 事后探针：和 v7 内在守卫放在一起看（明确标注为事后，未参与任何选择）

`moe-v7-0905` 的 task-agnostic 守卫在 external 上是 **439 TP / 80 FP**（召回 0.778，精确率 0.846）。
它用的是另一组路由量（freeze / acceleration / periodicity / turbulence），不在我的预注册池里，
也没有任何规则是对着它选的。为了不把结论说过头，必须写明：

**本轮 global 组合（最好 379/65，或 OR 曲线插值到 80 FP 约 395 TP）并没有超过 v7 守卫。**

但两族的假警报几乎不相交（`results/posthoc_v7_probe.json`）：

| 事后探索组合 | TP | FP | 精确率 | 召回 | 早期 TP | 早期 FP |
|---|---|---|---|---|---|---|
| v7 守卫（现有） | 439 | 80 | 0.846 | 0.778 | 57 | 44 |
| **v7 守卫 AND（0.98→0.90 cascade，379/75）** | **335** | **18** | **0.949** | 0.594 | 32 | 3 |
| v7 守卫 AND G-B | 271 | 8 | 0.971 | 0.481 | 26 | 2 |
| v7 守卫 OR（0.98→0.90 cascade） | 483 | 137 | 0.779 | 0.856 | 83 | 92 |

两者同时误报的 timely episode 只有 18 条，v7 单独误报 62 条、该 cascade 单独误报 57 条。
**跨 bundle 的 AND 把 v7 的 80 个假警报砍到 18 个，同时保住 439 个真阳里的 335 个。**
这是本轮最强的一个数字，但它是事后的，必须在下一轮重新预注册后才能当成结论。

## 7. 局限

- **global 阈值在开发集上是 in-sample。** 池化分位数用的是 development_main + development_extra，
  包含 development_main 自身，沿用帧调查的既有约定；开发集 global 数字因此略乐观，外部数字不受影响。
- **早期/晚期的划分依赖标签导出的生存先验表。** 这个表只用于**评分**，从未进入任何规则的决策；
  任务书提到的「用生存先验给报警加权」我**没有采用**，因为那会把标签导出的查找表放进运行时决策，
  和「MoE only / 训练无关」的约束冲突。这是一个我主动放弃的方向。
- **shortlist 有 40 条**（32 条不重复）在 external 上被评了。虽然全部由开发集冻结，
  但「最好的一条」仍然享受了 40 选 1 的选择优势；迁移比（recall transfer 0.94~1.05）是抵抗这一点的最好证据。
- `union_coverage_context.csv` 在 external 上扫了整条纯 OR 敏感度曲线。它只作为分母和背景报告，
  没有从中挑规则；但它确实构成了对 external 的额外一次观察，manifest 里已登记。
- 本轮所有检测器仍然继承帧调查的 (representation, direction) 选择，那一层是别人在开发集上选的，
  我没有重选，也没有重新验证它的选择偏差。

## 8. 下一步

1. 把跨 bundle 的 AND（v7 守卫 × 统一尾部 global quorum）正式预注册，在开发集上重选一次。
2. 追查 L3-high 那三个 global 检测器为什么假警报率跨队列放大 5~8 倍——这是检测器迁移问题，
   修好它才能解锁 early 目标。
3. 放弃有界确认窗口这条线；消融已经证明这个机制做不到它被假设能做的事。

## 9. 复现

```
python experiments/build_detector_alarms.py     # 重建 360 个报警数组，对封存 external 数组做一致性断言
python experiments/test_combination_core.py     # quorum 定义的暴力法复核 + 特例等价性
python experiments/select_on_development.py     # 11,676 条规则打分，冻结 shortlist.json（不打开 external）
python experiments/evaluate_external.py         # external 只跑这一次
python experiments/analyze_results.py           # 前提核查、前沿、suite 拆分、报警时刻
python experiments/make_headline_table.py       # 头对头表 + OR 曲线插值对照
python experiments/ablate_window.py             # 窗口消融（纯开发集）
python experiments/posthoc_v7_probe.py          # 事后探针
python experiments/write_manifest.py            # manifest + 迁移表
```

零学习参数，零梯度更新，全部规则是计数、顺序、窗口和分位层级。
