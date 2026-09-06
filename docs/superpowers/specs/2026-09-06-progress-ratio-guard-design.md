# 进度率保护器 v12：设计

日期：2026-09-06
状态：设计已确认，待实施计划
前置：`moe-v7-0905/REPORT_ZH.md`、`moe-v7-0905/docs/SURVIVAL_BASELINE_REPORT_ZH.md`、
`moe-v7-0905/docs/LOSO_VALIDATION_REPORT_ZH.md`、`moe-v7-0905/docs/CALIBRATION_VARIANTS_REPORT_ZH.md`、
`moe-hb-front-back-0905/docs/FRAME_SURVEY_REPORT_ZH.md`、`moe-hb-front-back-0905/docs/FLOW_AXIS_REPORT_ZH.md`

## 0. 目标与约束

目标是同时提高召回与精确度。三条约束在设计前已确定：

| 约束 | 取值 | 依据 |
|---|---|---|
| 目标口径 | 逐 chunk 超额 hazard（相对存活先验） | 推论 1：episode 级二值目标奖励"报得更晚" |
| 模型类 | 严格 train-free（无拟合权重、无梯度，阈值为无标签 order statistic） | 保持 v7 纪律与可审计性 |
| 任务信息 | 严格任务无关，运行时不读 task ID | frame survey：只有 global 模式能声称判据来自 MoE |

## 1. 第一性原理推导

### 1.1 五条已确立的事实

- **A1** `risk` ≡ 未在 horizon 上限前完成 ⇒ "还在跑"本身是证据，先验从 0.07 涨到 0.95+。
- **A2** routing 是物理/动作状态的读出，不是原因（committor 审计：q0 routing 不能选择命运，
  可读性到 q6–q7 状态分离后才出现）。
- **A3** 单标量 + 单阈值 ⇒ recall 与 precision 只能沿同一条 ROC 交换。
- **A4** mobility 与 flow_settling 的误报近乎正交（43 + 62 只共享 2 个），真检出重叠 35%。
- **A5** 误报的物理身份已知：libero_long 的 freeze 误报是慢速成功（episode 长度中位 32–44，
  及时成功中位 24），即"暂停后恢复了"。

### 1.2 推论

**推论 1（目标不适定）。** 由 A1，episode 级二值目标上的高精确可以靠等待免费获得。存活基线
报告已量化：v7 的 84.59% precision 中 60.6 pp 来自存活先验，净提升仅 1.40×。直接优化字面目标
会产出更会等待的检测器。**必须先换目标口径。**

**推论 2（存在物理下界，且已撞上）。** 由 A2，routing 可读性不能早于物理状态分离。
`libero_spatial` 在 12 个量 × 11 表征 × 2 方向 × 14 分位 × 5 条层规则下提升恒等于 1.0。这种
跨方法一致性不是选择效应，是信息论上界。**纯 routing 输入下 spatial 类任务不会被突破**，
新方法必须显式声明放弃哪一类。

**推论 3（精确来自合取，召回来自析取）。** 由 A4，误报独立而真检出相关的两个检测器，AND 对
FP 乘性衰减、对 TP 次乘性衰减。但单个 AND 召回必然低（0.332）。同时拿到两者的结构是若干个
内部合取项的析取。

**推论 4（tradeoff 的来源是不可撤销性）。** 由 A5，误报主体不是"看错了"，是"看对了但它后来
恢复了"。而暂时停顿与彻底卡死的区别，在定义上就是之后有没有释放。当前规则全部锁存，一次暂停
即一次永久误报。

### 1.3 空缺的格子

固定 q 与"还活着"之后，剩余判别问题是"这条 rollout 还在推进吗"。冻结只是不推进的一种；慢速
成功也会短暂看起来冻结。第一性原理下，推进与否是轨迹在商空间中的性质：**卡住 ⟺ routing 轨迹
被困在复返集；推进 ⟺ 轨迹有持续漂移分量。**

核查现有十二个量：`flow_path` / `flow_settling_log_ratio` / `flow_endpoint` 是 query 内沿 10 个
去噪步的量；`action_consensus` / `state_action_alignment` / `conditional_energy` /
`conditional_effective_rank` / `partial_edge_std` / `expert_load_effective_rank` 是瞬时图结构；
`conditional_query_d1` / `partial_query_d1` 与 mobility 是相邻 query 一阶差分。
`analysis_flow_straightness/` 做的是单次推理内动作潜变量的直线性，不是跨 query。

**没有任何一个量测"一个窗口内净走了多远"。** 这是本设计填补的格子。

## 2. 量的定义

在每层 `l`、`FINAL_FLOW = 9`、`ACTION = slice(1, 11)` 的 10 个 action token 上，沿用 v7 已有的
`hellinger()`：

```
d_l(q, q') = mean_{t=1..10} H( P_{l,t}(q), P_{l,t}(q') )
```

Hellinger 是度量，token 上的正系数平均仍是度量，故三角不等式成立。窗口 `W` 内：

```
L_l(q, W) = sum_{k=1..W} d_l(q-k+1, q-k)      # 路径长
D_l(q, W) = d_l(q, q-W)                        # 净位移，且 D_l <= L_l
R_l(q, W) = D_l / L_l   in [0, 1]              # 进度率
```

`R(q, W)` 取所选层组**组内**各层 `R_l` 的中位数。层组（front L2–L5 / back L12–L15 / all）与
`W` 在开发集上按固定规则选出后冻结。

**数值守卫。** 当 `L_l < eps_L` 时，由 `D_l <= L_l < eps_L` 可知确实没有净运动，令 `R_l := 0`。
这是极限的正确取值，不是分支——规则实现中没有 if/else。`eps_L` 取 pooled reference 语料全部
`L_l` 值的无标签 **1% 分位数**（不是第一四分位数——取 25% 会把四分之一的条目直接判为
零净运动，从而把 `r*` 逼到 0），与 profile 一并冻结。

**任务无关性的来源。** `R` 是同单位之比，无量纲。任务间路由尺度差异在分子分母同时约掉，
**不需要 episode 基线，也不需要语料常数定尺度**。这比 v7 的相对冻结强一层：v7 靠 q1–q4 基线
消尺度，`R` 靠构造消。

**边界（必须在报告中写明）。** `R` 消掉的是**尺度依赖**，不是**语料依赖**。最终切点 `r*` 仍是
语料分位数，因此 `LOSO_VALIDATION` §4 的 horizon 配方漂移依然适用。它只是把
`flow_settling` 那种"连量本身都需要 task ID"降级为"只有切点需要语料"。

## 3. 检测规则

```
stuck(q)  <=>  R(q, W) < r*  连续确认 K 次
```

`r*` 的定义无二义：对每条 reference trajectory 先取其全程 `min_q R(q, W)`，再对这些逐轨迹最小值
取 alpha 分位数；`R(q, W) < r*` 即触发。这是 v7"逐轨迹峰值分位数"的严格对偶——v7 控制"至少
向上穿越一次"的概率，此处控制"至少向下穿越一次"的概率。

这是刻意的：`CALIBRATION_VARIANTS` §2.1 证明 pooled trajectory-peak 隐式承担了逐轨迹多重检验
校正，换成逐 query 分位数会把 FPR 抬高约 100 倍。取逐轨迹最小值分位数保留同一件事。

规则只有一行，涵盖两个物理模态：

| | 低 mobility（L 小） | 高 mobility（L 大） |
|---|---|---|
| **高 R** | 缓慢定向推进 → 不报警（**否决当前误报主体**） | 快速推进 → 不报警 |
| **低 R** | 冻结 → 报警（≈ v7 relative_freeze） | 兜圈 / 反复重抓 → 报警（**新增召回**） |

左列低 R 即 v7 的 relative_freeze，**作为 R 规则的特例被自动回收**。相对 v7 的增量恰好是右下格，
消融边界天然干净，不需人为构造对照。

**推论 3 在此处如何落地。** 推论 3 主张"精确来自合取、召回来自析取"，但 §3 的规则是单一阈值，
表面上没有用到它。原因是 `R` 已经在**一个标量内部**完成了这件事：2×2 表是同一条轨迹的
`(L, D/L)` 正交分解，两个物理模态被同一条判据覆盖，无需外部合取。推论 3 因此不进入检测规则，
而是决定了 **K4 的对照对象**——方案 B（十二个量自归一化后按误报依赖矩阵组成的最佳 DNF）正是
推论 3 的直接实现，`R` 必须超过它才算有增量。

`W` 是唯一关键超参：必须长到能跨过一次正常的"停顿—恢复"，否则慢速成功仍会误报。现有全部脚本
固定 W4/K4 只扫分位数，三份报告均将此列为限制；本设计必须扫 `W`，同时补上该缺口。

## 4. 评估协议

- **存活先验** `pi_s(q) = P(risk | alive at q)`，按 suite 估计。**在 development cohort 上估计，
  应用到 external。** 现有 `evaluate_survival_baseline.py` 是同 cohort 估计（报告自标"对基线略有
  偏袒"），本设计修正这一点。
- 每次报警按其自身 chunk 匹配先验；**提升倍数 = precision / 平均匹配先验**。
- **主指标**：低先验区间（`pi_s < 0.25`）内的 TP 数、precision、提升倍数，以及低先验召回
  （有低先验报警的 risk episode / 全部 risk episode）。
- **先验切点做敏感性扫描** `{0.10, 0.25, 0.40}`。frame survey 标注该扫描从未做过，且排序结论
  对它有依赖。
- episode 二值口径（TP/FP/recall/precision/timely FPR）仍记录，**仅用于与 v4/v6/v7 对表，
  不参与选择**。

## 5. 防泄漏与封存

沿用 v7 三段纪律：

1. **机制与规则选择**使用 development outcomes，必须在报告中声明，不得隐去。
2. **阈值数值校准**不读 outcome，为 pooled reference trajectory 的经验 order statistic，
   成功与失败样本全部保留。
3. **external 评分**先写入并哈希 profile 与逐 episode first-alarm，再打开 outcome 文件。

本设计特有的四条：

4. **`W`、层组、`K` 在开发集选定后冻结。** `W` 是关键超参，事后调 `W` 等于事后调结论。
5. **禁止事后构造合取项。** `FLOW_AXIS` §6.3 记录了 AND 组合是在 external 上观察到误报不重叠
   之后才构造的。本设计预注册规则形态为单一 `R` 阈值加持续确认 `K`；任何合取必须先在开发集声明。
6. **LOSO 必须重跑。** `r*` 仍是语料分位数，按 published / loso_l1 / loso_l2 三级对照报告。
7. **external_8b 不是 pristine holdout**，v3–v7 与 HB bundle 已反复查看。最终确认应在新 seed
   cohort 上进行；若本轮无新 cohort，报告中必须写明结论强度受限。

**在线因果性。** `R(q, W)` 只读 `q-W .. q`，严格因果。用 v7 同款 harness 验证改写未来 query
不改变已有报警，并核对 GPU 重算与缓存一致。

## 6. 预注册的证否条件

按可触发的先后排序。前三条在接触 external 之前即可判死。

全部判据在开始前量化，不留事后解释空间。mobility 的高/低以同一开发集的中位数切分。

| | 量化条件 | 触发后果 |
|---|---|---|
| **K1** | 开发集上，至少 **20 个 risk episode** 的首次低-`R` 报警落在（高 mobility, 低 R）格 | 少于 20 则"兜圈"模态在本语料不存在，新增部分是空的，**立即停止并写否定结果** |
| **K2** | `R` 在 global 阈值下的 external TP 不低于其 per_task 阈值下 TP 的 **50%**（复现 frame survey 双跑协议） | 低于则"无量纲 ⇒ 任务无关"这一构造性主张被证否；对照点为 `flow_settling` 的 83 → 3 |
| **K3** | `R` 检出的 risk episode 集合中，**至少 10%** 是 v7 relative_freeze 未检出的 | 否则不是新轴，只是 freeze 的重写 |
| **K4** | external 低先验 TP **严格超过**方案 B 同协议基线 | 不超过即无增量，写否定结果 |
| **K5** | `libero_spatial` 提升倍数保持在 **[0.9, 1.15]** | 若显著超出，更可能是泄漏，**触发泄漏审计而非庆祝** |
| **K6** | libero_long 那 51 个 freeze 分支误报中，**至少 1/3（17 个）**在同一 operating point 下被 `R` 规则否决 | 若不足，§3 的 2×2 机制解释错误，即使总指标好看也须重新解释机制 |

K5 是反向哨兵：推论 2 指出 spatial 撞的是信息论上界，因此"结果太好"在此处是坏消息。

**执行顺序是强制的。** K1 → K2 → K3 全部在开发集上完成并通过，才允许接触 external；K4 → K6
在 external 上评估，且必须在 §5 的封存哈希写入之后。任一前置条件失败即停止，不得跳过继续跑
后续指标。

不可行或被证否的分支**不放宽约束重试**，完整候选表与 `feasible: false` 记录一并保留——这是
结论本身，与 `CALIBRATION_VARIANTS` 的处理一致。

## 7. 实现结构

新建 `moe-progress-ratio-v12-0906/`，与现有 bundle 同构。模块按单一职责切开：

| 文件 | 职责 | 依赖 |
|---|---|---|
| `method/progress_ratio.py` | 纯函数：route 张量窗口 → `L, D, R`。无 I/O | numpy |
| `method/progress_monitor.py` | 在线 monitor，`W+1` query 环形缓冲，逐 query 吐 `R` 与报警状态 | `progress_ratio` |
| `experiments/build_progress_cache.py` | 批量算 `R` 并缓存，可 resume | `progress_ratio` |
| `experiments/select_operating_point_v12.py` | 开发集扫 `W` × 层组 × `K` × 分位数，输出完整候选表 | 缓存 |
| `experiments/evaluate_baseline_dnf.py` | 方案 B 基线：十二个量做 episode 内自归一化，global 模式重扫，按误报依赖矩阵选近正交配对组成 DNF | 缓存、outcome |
| `experiments/evaluate_hazard.py` | 逐 chunk 超额 hazard 评估，三个先验切点 | 缓存、outcome |
| `experiments/evaluate_phenotype_2x2.py` | K1 / K3 / K6 诊断表 | 缓存、outcome |
| `experiments/evaluate_loso_v12.py` | 复用 v7 LOSO 协议，三级对照 | 缓存、outcome |
| `experiments/verify_raw_causal_gpu.py` | 原始 Zarr 因果回放 | `progress_monitor` |

`method/progress_monitor.py` 的接口对齐 `moe-v7-0905/method/intrinsic_guard_monitor.py`，
以便共用回放 harness。在线缓冲为 `W+1` 个 `[8, 10, 11, 32]` fp32 张量，约 `(W+1) x 112 KB`。

**测试**（`tests/`）：

- 度量性质：`D <= L`、`R in [0, 1]`、`L -> 0` 时 `R = 0`。
- 因果性：改写未来 query 不改变已有报警。
- 特例一致性锚点：低 mobility 区间上 `R` 规则与 v7 relative_freeze 的报警逐位对齐。

**关键产物**：`global_profile.npz`（五个常量 `W`、层组、`K`、`r*`、`eps_L`）、
`sealed_manifest.json`、`hazard_metrics.csv`、`phenotype_2x2.csv`、`loso_metrics.csv`、
`sealed_first_alarms.npz`。

## 8. 明确不做的事

- 不训练分类器，不拟合权重（约束二）。
- 不使用 task ID、动作、视觉、物理状态、outcome 或未来 query 作为运行时输入。
- 不做事后撤销状态机。HB 报告实测释放信号过慢（alarm+1 的 AUC 0.380，到 horizon 才 0.657），
  撤销回来时提前量已经没有了。本设计把撤销前移为**报警时否决**，绕开该问题。
- 不声称解决 `libero_spatial`。推论 2 指出那是信息论上界；K5 将其设为反向哨兵。
- 不回答"报警之后干预是否有用"。该问题需要同 snapshot 多分支 fork 反事实，且当前语料的独立
  trunk 数（候选重排 3 条、fork 语义 10 条、horizon recovery 23 对）不足以支撑，属于独立的
  数据采集项目。

---

# 修订一（2026-09-06，第二轮）

## R0. 这份修订的地位

**这是事后修订，不是盲测。** 假设来自第一轮开发集诊断，按 `FLOW_AXIS_REPORT_ZH.md` §6.3
对 AND 组合的处理惯例，必须如此标注。第一轮的闸门结果原样保留，不得改写。

## R1. 第一轮实测

| 闸门 | 实测 | 判定 |
|---|---|---|
| K2（global vs per_task） | 17 vs 18（FPR 上限内）；150 vs 151（无约束） | **通过** |
| K1（兜圈格 ≥ 20 risk） | FPR 上限内五点中四点为 1/2/2/4 | 不通过 |
| K3（新增检出 ≥ 10%） | 同四点为 0.033–0.048 | 不通过 |
| 预注册选点 | 0/840 可行；FPR ≤ 0.5% 内 `low_prior_tp` 上限 17 | 不可行 |

K2 通过是实质结论：`R` 的无量纲构造兑现了任务无关性，没有出现 `flow_settling`
从 83 塌到 3 的情形。

## R2. 失败归因：原设计的规则错了，量没错

原 §3 断言"`R` 已在单个标量内部完成合取，推论 3 因此不进入检测规则"。**该断言错误。**

按存活先验分区、并在 suite × chunk 内配对后（`experiments/diagnose_phenotype_timing.py`）：

| W=6/all | 低先验 < 0.25 | 高先验 ≥ 0.25 |
|---|---:|---:|
| freeze 格（低 mobility ∧ 低 R） | 1.55x | 5.35x |
| 兜圈格（高 mobility ∧ 低 R） | **5.60x** | 1.93x |

两格的信息窗口不重叠：freeze 是晚期信号，兜圈是早期信号。单一 `R` 阈值把两者相加，
晚期格以数量压倒早期格，因此低先验产量封顶在 17。

mobility 轴携带的不是冗余信息，而是**时间信息**。

未做配对时兜圈格 lift 为 0.30–0.61x，看起来是"成功特征"。那是时长混淆：risk episode
按定义跑满上限，贡献不成比例的晚期 chunk。配对后符号翻转为 1.04–4.01x。

## R3. 修订后的判据

$$\boxed{\ \text{stuck}(q)\iff L(q,W)\ge \ell^*\ \wedge\ R(q,W)<r^*\ \text{连续确认}\ K\ \text{次}\ }$$

即推论 3 的原始形式。`ℓ*` 与 `r*` 均为无标签 order statistic，`ℓ*` 取 pooled reference
`L` 的中位数（预注册，不扫描）。freeze 分支**不再由本判据覆盖**，它归 v7。

## R4. 目标改为 loop，并公开覆盖上限

`analysis_trap_taxonomy/REPORT.zh.md` 确认三类失败。按
`(task, init_state_id, flow_noise_seed)` 对齐后：

| | development（487 失败） | external（564 失败） |
|---|---:|---:|
| loop | 256（52.6%） | 314（55.7%） |
| static | 135（27.7%） | 164（29.1%） |
| phantom grasp | 93（19.1%） | 115（20.4%） |
| ┗ 无 loop 无 static | 22（4.5%） | 32（5.7%） |
| **loop ∪ static 覆盖上限** | **73.9%** | **76.4%** |

因此 **26.1% / 23.6% 的失败在本 2×2 上结构性不可达**，其中 phantom grasp
（平稳、有方向地朝错误目标前进）落在原设计标注为"健康"的高 mobility ∧ 高 R 格。

低先验区间内兜圈格对三类的富集倍数：

| 目标 | W4/all | W6/all | W4/front |
|---|---:|---:|---:|
| loop | 5.12x | 10.86x | 6.06x |
| static | 1.01x | 0.27x | 0.86x |
| phantom（纯） | 0.33x | 0.07x | 0.34x |

分工与机制预测一致。因此本方法**应命名为 loop 早期检测器**，不得称为通用 trap 检测器。

## R5. 修订后的闸门

第一轮六条闸门作废，替换为：

| | 量化条件 | 触发后果 |
|---|---|---|
| **A1** | 合取判据在 development 上低先验 loop TP ≥ 20，且 timely FPR ≤ 0.005 | 不满足即停止，写否定结果 |
| **A2** | 低先验区间内对 loop 的提升倍数 > 对 static 与 phantom 的提升倍数 | 否则不是 loop 特异，只是又一个通用晚期量 |
| **A3** | external 低先验 loop TP 严格超过方案 B 基线 | 不超过即无增量 |
| **A4** | `libero_spatial` 提升倍数保持 [0.9, 1.15] | 显著超出触发泄漏审计 |
| **A5** | 报告必须同时给出 §R4 的覆盖上限与 phantom 盲区 | 缺失即报告不完整 |

A4 保留第一轮的反向哨兵。**`ℓ*` 固定为中位数，不进入网格**——它是机制声明，不是可调参数；
让它可调会把修订变成二次搜索。

## R6. 不变的部分

§2 量的定义、§4 评估协议（逐 chunk 超额 hazard、三个先验切点）、§5 防泄漏与封存、
§7 实现结构、§8 明确不做的事，全部不变。external 在本修订完成封存前仍不得打开。
