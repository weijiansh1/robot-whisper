# 双层 MoE 路由监控器（WATCH / ACT）：可行吗，各自代价是什么

结论先说。

**双层设计在 `global` 阈值模式下是成立的，但两层各自付出的代价完全不同，而且其中一层的代价大到无法在既定误报预算内交付。**

- **ACT 层（高置信触发）成立且便宜。** `action_consensus|global + conditional_query_d1|global + mobility|global` 的 2-of-3 规则在 external 上得到 **146 TP / 9 FP，precision 0.942，lift 1.838，早期 23 TP / 0 FP**。它的 155 次报警 **100% 落在 WATCH 报警集合内部**，且全部发生在 WATCH 报警时刻或之后。它吃掉了 WATCH 的 **35.6% 真阳性，却只吃掉 4.6% 的假阳性**——真假阳性的浓缩比 7.8×。这正是"ACT 是 WATCH 的有用子集"这句话要的证据，而且是测出来的，不是设计出来的（两层只共用 `mobility` 一个头）。
- **WATCH 层（覆盖保证）在开发集上成立，在 external 上超预算。** 覆盖最优的四头跨帧 OR 把最差物理失效模式覆盖率从 0.357 抬到 **0.565**、模式 CV 从 0.215 压到 **0.153**，但假阳性从 54 涨到 **197**，是 0.005 timely-FPR 预算（75.2）的 **2.6 倍**。抬升覆盖率的那个头（`partial_edge_std|global`）恰恰是阈值外推最差的头：开发集 40 FP → external 147 FP，×3.7。
- **"无盲区是无监督准则的免费副产品"这个强命题——部分成立，但前提被限定得很死。** 用二元开发结果做**贪心边际增益**选头，在 `global` 下选出的四头集合与用物理模式标签做 minimax 选出的集合**逐字相同**；而同样只用二元结果、只是把"每帧取单头最优"的两个变体，最差模式覆盖率分别差 **0.130 / 0.208**，判定为不近似。所以真正免费的不是"帧多样性"，而是"帧多样性 + 边际增益"。
- **组合仍然没有买到信息密度，只买到覆盖率。** WATCH 的 lift 是 1.603 / 1.593，明确低于单头锚点 1.764。ACT 的 lift 1.838 名义上高于锚点，但配对 bootstrap 的差值置信区间 **[-0.038, +0.184] 跨 0**（p(diff>0)=0.897），**不成立**。

---

## 1. 复现与自检（全部逐位通过）

在打开 external 之前，24 个 frame-survey 头在两个 split 上按原脚本的表示 / 方向 / 分位数重建；external 侧重建结果与已发布的 `external_first_alarms.npz` **逐位相同**（`results/head_inventory.csv`，24/24）。这是信任开发集孪生数组的前提。

三个锚点作为硬断言，全部通过（`evaluate_external.py::assert_anchors`）：

| 锚点 | TP | FP | precision | lift |
|---|---|---|---|---|
| `mobility \| L12 \| low \| q0.975 \| global` | 195 | 17 | 0.9198 | 1.7641 |
| `mobility \| L2 \| low \| q0.700 \| per_task` | 272 | 57 | — | — |
| `expert_load_effective_rank \| L3 \| low \| q0.85 \| per_task` | 370 | 93 | — | 1.5479 |

控制器给出的六项 external 测量，本 bundle 的 k-of-n 组合器全部复现（`results/analysis.json::controller_reproduction`）：

| 规则 | 期望 | 实测 |
|---|---|---|
| 跨帧三头 OR | 381 / 54，worst 0.357，CV 0.215 | 381 / 54，0.357，0.215 |
| 同帧三头 OR（adjacent_query） | 277 / 207，worst 0.240，CV 0.354 | 277 / 207，0.240，0.354 |
| 跨帧三头 AND（k=3） | 0 TP | 0 TP |
| `mobility ∧ flow_path` | 0 TP | 0 TP |
| `mobility ∧ expert_load` | 69 / 3，worst 0.000 | 69 / 3，precision 0.9583，worst 0.000 |
| `mobility ∧ flow_settling`（per_task） | 187 / 8 | 187 / 8 |

生存基线脚手架也复现：horizon 上限 goal 30 / long 52 / object 28 / spatial 22，先验穿越 0.25 的 chunk 分别是 **18 / 26 / 17 / 13**（`results/manifest.json::survival_scaffolding`）。

---

## 2. WATCH：变体 (a) 逼近变体 (b) 吗

七个被计分的物理失效模式（n ≥ 10 的门槛在开发前声明；两个 split 恰好都是同一组七个）：
`object_released_or_dropped_before_goal` n=241、`stable_grasp_not_observed` n=143、`object_moved_but_goal_unmet` n=89、`goal_predicate_regressed` n=26、`object_released_outside_goal` n=25、`timeout_while_holding_target` n=23、`approached_target_without_observed_contact` n=14。被排除的两个：`no_meaningful_target_progress` n=2、`mechanism_threshold_not_reached` n=1。

**变体 (b) 把物理模式标签放进了规则选择环节。这超出了本项目"只用二元开发结果选规则"的惯例，在此明确声明；它从不作为可部署的选择流程。** (a) 系列全部只读二元开发结果。

### external 结果（`primary` 池，`global` 模式）

| 规则 | 头数/帧数 | TP | FP | precision | lift | 早期 TP/FP | worst-mode | CV | 符合 75 FP 预算 |
|---|---|---|---|---|---|---|---|---|---|
| 单头 `mobility\|global` | 1 / 1 | 195 | 17 | 0.920 | 1.764 | 36 / 4 | 0.120 | 0.465 | ✓ |
| 控制器跨帧三头 OR（参照） | 3 / 3 | 381 | 54 | 0.876 | 1.488 | 59 / 29 | 0.357 | 0.215 | ✓ |
| **(a1)** 每帧单头最优后回退 | 2 / 2 | 349 | 60 | 0.853 | 1.498 | 62 / 32 | 0.435 | 0.218 | ✓ |
| **(a2)** 每帧单头最优（等额预算） | 4 / 4 | 378 | 60 | 0.863 | 1.491 | 60 / 33 | 0.357 | 0.229 | ✓ |
| **(a3)** 帧多样贪心边际增益 | 4 / 4 | 410 | 197 | 0.675 | 1.603 | 113 / 171 | **0.565** | 0.153 | ✗（2.6×） |
| **(b4)** 有监督 minimax（限 4 头） | 4 / 4 | 410 | 197 | 0.675 | 1.603 | 113 / 171 | **0.565** | 0.153 | ✗ |
| **(b)** 有监督 minimax（2–5 头） | 5 / 4 | 419 | 199 | 0.678 | 1.593 | 113 / 172 | **0.565** | 0.144 | ✗ |
| 同帧对照（4 头，token_graph） | 4 / 1 | 146 | 206 | 0.415 | 1.985 | 77 / 197 | 0.160 | 0.402 | ✗ |

**(a3) 与 (b4) 是同一个集合**：`expert_load_effective_rank|global + mobility|global + partial_edge_std|global + flow_path|global`，Jaccard = 1.0。(a3) 与 5 头的 (b) 相比，external worst-mode 差 **0.000**，CV 差 +0.009 —— 在预先声明的 0.05 门槛内，判定为**近似**。在 `secondary_with_v7` 池里，无帧限制的贪心 (a3_free) 与 (b) 也是**同一个集合** `{v7_guard, flow_endpoint, flow_path}`（worst 0.697）。`per_task` 下 (a3) 与 (b) 差 +0.0075，也判定近似，但两者绝对水平都只有 0.35 上下。

**但 (a1) 和 (a2) 明确不近似**：worst-mode 差 -0.130 和 -0.208。二者都是"每帧取单头最优"，只是排序口径不同。

> **诚实的结论**：把"参考系独立性"变成无盲区，**不是**帧多样性本身的免费副产品。用二元结果做"每帧挑最强单头"会挑到帧内最高召回的头，而不是帧间互补的头。只有把准则换成**运行中 OR 的边际真阳性增益**（同样只需二元结果），无监督选择才会精确落到有监督 minimax 的解上。这个更强的说法在 `global`（两个池）与 `per_task` 上都成立，但它是"某个特定无监督准则"的性质，不是"帧多样性"的性质。

### 跨帧 vs 同帧：尺寸与预算双匹配的对照

同帧 4 头对照（token_graph 帧内）用了 **206 FP**，与跨帧 (a3) 的 197 FP 基本相同，结果：TP 146 vs 410，worst-mode 0.160 vs 0.565，CV 0.402 vs 0.153。**在头数相同、误报预算相同的条件下，跨帧在每一个轴上都赢。** 这与控制器的方向性结论一致，并且这里是尺寸与预算双匹配的版本。

### WATCH 的真实代价：阈值外推

覆盖率是靠 `partial_edge_std|global` 买来的，而它是全池外推最差的头之一。开发集 → external 的假阳性膨胀比（`results/external_rules.csv` 中 `role=single`）：

| 头（global） | 表示/方向 | dev FP | ext FP | 膨胀 | ext precision |
|---|---|---|---|---|---|
| `partial_query_d1` | L3 / high | 21 | 173 | **8.2×** | 0.273 |
| `conditional_effective_rank` | L3 / high | 20 | 158 | **7.9×** | 0.358 |
| `partial_edge_std` | L3 / high | 40 | 147 | **3.7×** | 0.402 |
| `flow_path` | L15 / high | 1 | 4 | 4.0× | 0.926 |
| `expert_load_effective_rank` | L3 / low | 15 | 36 | 2.4× | 0.868 |
| `conditional_query_d1` | L5 / low | 30 | 31 | 1.0× | 0.862 |
| `mobility` | L12 / low | 15 | 17 | 1.1× | 0.920 |

三个 `L3|high` 头的池化全局阈值都不外推。消除盲区所需要的 token_graph 帧，其最佳覆盖头恰好在这三个里。**"无盲区"与"误报预算"在当前头池里是冲突的，不是可以同时拿到的两件事。**

预算内（ext FP ≤ 75.2）能拿到的最好 worst-mode 覆盖率，在被冻结的规则里是 **(a1) 的 0.435**（349 TP / 60 FP，lift 1.498）。这条排序是在 external 上做的，**属于事后（post-hoc）描述，不构成规则选择**（`results/external_watch_frontier.csv`）。

---

## 3. ACT：高置信触发

ACT 的头是**独立于 WATCH 搜索**的（全部 pair 与 triple，k=2 与 k=3，要求跨 ≥2 帧），最终两层只共用 `mobility` 一个头。

### 目标函数本身就是一个结果

- **ACT-A（最初声明：TP ≥ 50 下最大化 precision）**：`expert_load + flow_endpoint + partial_query_d1`，k=2 → external **80 TP / 5 FP，precision 0.941，lift 1.060，早期 1 TP / 0 FP**。开发集上有几十条规则达到 precision 1.000，它们的 lift 全在 1.0–1.2。**这类"精度"几乎全是生存基线**：规则在 chunk 19–40 才触发，那时还在跑的 episode 本来就大半要失败。只会事后开火的 ACT 不是 ACT。
- **ACT-B（附录中声明，external 打开前）：precision ≥ 0.95 且 TP ≥ 50 下最大化早期 TP**：`action_consensus|global + conditional_query_d1|global + mobility|global`，k=2 → external **146 TP / 9 FP，precision 0.942，lift 1.838，早期 23 TP / 0 FP**。

`per_task` 与含 v7 的池：ACT-B 分别是 304/41（precision 0.881，lift 1.401，早期 78/21）与 325/23（precision 0.934，lift 1.418，早期 41/5）。

### ACT-B 并不支配单头，必须说清楚

对比 `mobility|global` 单头（195/17，precision 0.920，lift 1.764，早期 36/4）：

| | ACT-B | 单头 | 差 |
|---|---|---|---|
| TP | 146 | 195 | **-49** |
| precision | 0.942 | 0.920 | +0.022 |
| lift | 1.838 | 1.764 | +0.073 |
| 早期 TP | 23 | 36 | **-13** |
| 早期 FP | **0** | 4 | -4 |

lift 的 bootstrap（按 suite 分层，4000 次重采样）：ACT-B 1.838，95% CI **[1.704, 1.977]**；与单头的**配对**差值 +0.073，95% CI **[-0.038, +0.184]**，p(diff>0)=0.897。

> 这是本项目里第一条点估计 lift 高于 1.764 锚点的组合规则，但**配对 bootstrap 没有把它和单头分开**。按项目规则如实陈述：**ACT-B 买到的是覆盖以外的一点点精度与零早期误报，没有证明买到了信息密度。**
>
> 另需说明：1.764 是 `mobility|global` 这一条具体锚点的 lift，不是单头 lift 的上界。同池里 `partial_edge_std|global` 单头 lift 2.009、`conditional_effective_rank|global` 1.963（代价是 precision 0.40 / 0.36）。所以"组合不涨信息密度"这句话只对被点名的 1.764 锚点成立。

### ACT-B 的盲区是结构性的

worst-mode 覆盖率 **0.000**（`approached_target_without_observed_contact`，n=14），模式 CV 0.692。按 suite 拆：libero_long 126 TP、libero_goal 19、libero_object 1、libero_spatial **0**。**ACT 层几乎只在 long 与 goal 上工作。** 这正是把 ACT 与 WATCH 分开选的理由：ACT 不负责覆盖，WATCH 负责。但它也意味着单靠 ACT 部署会在 spatial 上完全失明。

---

## 4. 两层合起来（`primary|global`，headline 配对 WATCH=(a3) / ACT=ACT-B）

15 600 条 external episode：

| 分层 | episode 数 | TP | FP |
|---|---|---|---|
| 到达 ACT | 155 | 146 | 9 |
| 只到 WATCH | 452 | 264 | 188 |
| 无报警 | 14 993 | 154（漏检） | — |
| WATCH 合计 | 607 | 410 | 197 |

- **包含关系：ACT ⊆ WATCH，155/155 = 1.000。** 两层只共用一个头，这是测出来的，不是构造保证的。
- **时序：155/155 的 ACT 报警时刻 ≥ WATCH 报警时刻**，中位延迟 0 chunk。ACT 是合法的升级，不是另一条并行通道。
- **浓缩比：ACT 升级了 WATCH 35.6% 的真阳性，只升级了 4.6% 的假阳性 → 7.8×。**

另外两个配置的浓缩比明显更差：`per_task` 为 76.6% / 30.2% = 2.5×；含 v7 的池为 71.0% / 26.4% = 2.7×。**只有 `global` 模式给出了干净的两层分离**，而 `global` 恰好是唯一不含任务侧信息、能证明信号在 MoE 侧的模式。

误报预算：ACT 层 9 FP（0.0006 timely-FPR，预算的 12%）；WATCH 层 197 FP（0.0131，预算的 262%）。**双层结构的账是：ACT 极便宜，WATCH 超支。** 若把 WATCH 换成预算内的 (a1)（349/60，worst 0.435），两层总误报 60，但最差模式覆盖率从 0.565 掉到 0.435。

---

## 5. OR 增量的模式集中性（说明通过了哪一种置换）

对 headline WATCH，取"OR 抓到但其最强单头没抓到"的增量集合，统计量为七个模式的增量率 CV，零假设在**层内**置换增量指示（保留每层增量数），10 000 次：

| 配置 | 增量 | 观测 CV | suite 分层 p | task 分层 p | 结论 |
|---|---|---|---|---|---|
| `primary\|global`（最强单头 `expert_load`，237 TP → OR 410 TP） | +173 TP / +161 FP | 0.373 | 0.081 | **0.171** | 两种分层都不通过 |
| `primary\|per_task`（`expert_load` → +`mobility`） | +27 TP / +43 FP | 1.489 | **0.0005** | **0.0039** | **两种分层都通过** |
| `secondary_with_v7\|global`（`v7_guard` → +`flow_endpoint`） | +19 TP / +7 FP | 1.338 | 0.208 | 0.154 | 都不通过 |

**如实说明**：控制器报告的"OR 增量是模式集中的"（p=0.015 suite / 0.047 task）是针对另一条 OR 的。**本 bundle 的 `global` 模式 WATCH 增量不显示模式集中性**（task 分层 p=0.171），这是一个负结果。唯一通过两种分层的是 `per_task` 的 `mobility` 增量，它集中在 `timeout_while_holding_target`（7/23 = 0.304），而在 `goal_predicate_regressed`（0/26）与 `object_released_outside_goal`（0/25）上为零。

按照伴随 agent 的告诫，本报告**不对任何单头的模式选择性做任何主张**。所有与模式相关的陈述都是关于组合的覆盖率或增量，且已标注通过了哪种分层。

---

## 6. 失败了什么 / 没做什么

1. **WATCH 的覆盖最优解交付不了。** 197 FP 是预算的 2.6 倍，原因是单头阈值外推失败（`L3|high` 系列 ×3.7–8.2）。这不是组合的问题，是头的问题。
2. **ACT-A 目标函数是错的**，它选出的规则 lift 1.060——全是基线。这在开发集上就能看出来（precision 1.000 与 lift 1.0–1.2 同时出现），但只有先写下"报告每条规则的 lift 与早期带"才会暴露。
3. **ACT-B 的 lift 优势没有统计上站住。** 点估计 1.838 > 1.764，配对 CI 跨 0。
4. **变体 (a1)/(a2) 的失败杀掉了命题的强形式。** "帧多样性 ⇒ 无盲区"不成立；成立的是"帧多样性 + 边际增益 ⇒ 无盲区"。
5. **`global` 的 WATCH 增量没有模式集中性**（task 分层 p=0.171）。
6. **未纳入更新 bundle 的量**（`moe-circuit-analogy-0906` 的 norm_fiedler 等、`moe-token-geometry-0906` 的 procrustes/bandedness、`moe-flow-semantics-0906` 的逐步 mobility）。这些 bundle 只发布了 external 侧报警数组，没有开发侧孪生；在不重建开发侧报警的前提下把它们放进选择环节会破坏"开发集选、external 评一次"的纪律。`procrustes_layer` 可能构成第五个参考系（跨层，而非跨查询/跨去噪步/token 图/专家负载）——这是最值得下一步做的事。
7. **`per_task` 全线更差**：ACT 浓缩比 2.5× vs 7.8×，(a3) 的 worst-mode 0.357 vs 0.565。这与项目结论一致——`global` 才是证明信号在 MoE 侧的模式。

---

## 7. 下一步

1. **修 `L3|high` 头的外推**，而不是换组合。若 `partial_edge_std|global` 的 external FP 能从 147 压到 40 量级，(a3) 就落进预算，两层结构立刻可交付：worst-mode 0.565、总误报 ~90。这是唯一一个能同时解决覆盖与预算的动作。
2. **给 `moe-token-geometry-0906` 的 `procrustes_layer` 重建开发侧报警**，并检验它是否是第五个参考系（用已有的假阳性依赖矩阵口径：共现 / 边际乘积）。若成立，贪心边际增益准则可以多吃一帧。
3. **把 ACT-B 的 spatial 盲区当成一个独立问题**：ACT 在 spatial 上 0 TP，需要一条 spatial 专用的 k=2 规则，或接受 spatial 只有 WATCH 层。
4. **把 (a3) 的贪心准则当作一个可复用发现去验证**：它在两个池、两个阈值模式下都复原了有监督 minimax 的解。在别的头池上再验一次，才能把它称为规律而不是巧合。

---

## 文件

- `experiments/twotier_lib.py` — 共享机制，全部从冻结的 v4 / hb-front-back 代码 import
- `experiments/build_alarms.py` — 重建两个 split 的逐头首次报警；external 侧逐位复现检查
- `experiments/select_on_development.py` — 只读开发集的 WATCH/ACT 选择
- `experiments/evaluate_external.py` — external 一次性评估 + 锚点断言 + 双层联合核算 + 增量置换检验
- `experiments/analyse_two_tier.py` — 控制器结论复现、bootstrap、预算前沿（事后描述）
- `experiments/make_manifest.py` — 输入/输出哈希、生存先验脚手架
- `results/PREREG.md` — external 打开前写下的目标函数、约束、禁止的主张；§10 为"看过开发集后、打开 external 前"的附录
- `results/{head_inventory,development_rules,external_rules,external_watch_frontier}.csv`
- `results/{alarm_build,development_selection,external_evaluation,analysis,manifest}.json`
