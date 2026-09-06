# 状态通道与 chunk 内圆弧形成过程

`moe-state-channel-0906` · 2026-09-06 · 纯 CPU
预注册：`results/PREREG.md`（在打开 external 之前写死）· 锚点：三条已发表检测器全部逐位复现

## 结论先行

### A. chunk 内协同（主问题）

**A1. 圆弧是在 10 步去噪中被"造"出来的，前层从零造起。** 以 token 序号与中心化 Gram 主轴 PC1 的 |Pearson| 衡量（置换零假设 p95 = 0.632）：

| 层 | step 0 | step 9 | 首次越过零假设 |
|---|---|---|---|
| L2 | 0.399 | 0.913 | **step 4** |
| L3 | 0.598 | 0.924 | step 1 |
| L4 / L5 | 0.688 / 0.711 | 0.952 / 0.949 | step 0 |
| L12–L15 | 0.939–0.953 | 0.973–0.978 | step 0 |

external_8b 逐层复现（L2 同为 step 4）。Fiedler 版本更晚：L2 step 5、L3 step 4、L4 step 2、L5 step 1、后层 step 0。**后层一开始就是有序弧；前层 step 0 近乎各向同性团（`arc_stretch` 1.30，团=1.0），到 step 9 被拉成整条弧（3.04）。**

形状轨迹（dev 中位数，external 一致）：

| 量 | L2 s0→s9 | L15 s0→s9 | 团 | 均匀弧 |
|---|---|---|---|---|
| `arc_stretch` | 1.296 → **3.037** | 2.210 → 2.687 | 1.0 | ≈4.9 |
| `neighbour_ratio` | 0.876 → **0.456** | 0.462 → 0.259 | 1.0 | ≈0.06 |
| `pc1_share` | 0.374 → 0.526 | 0.548 → 0.626 | 0.11 | 1.0 |
| `erank_centred` | 0.601 → 0.464 | 0.454 → 0.353 | 1.0 | ≈0.1 |

**A2. 形成斜率可预测，但在后层几乎就是终点电平的重参数化。** external，`global` 模式，冻结协议：

| 头 | 层/方向/分位 | TP | FP | prec | recall | 早 TP/FP | lift [95%CI] | 中位 chunk |
|---|---|---|---|---|---|---|---|---|
| **`td_slope`** | L15 low q0.960 | 161 | 24 | 0.870 | 0.285 | 27 / 10 | **1.631 [1.526,1.708]** | 30 |
| `centred_slope` | L15 low q0.960 | 164 | 23 | 0.877 | 0.291 | 27 / 9 | 1.622 [1.521,1.695] | 30 |
| `td_late_early` | L15 low q0.960 | 161 | 25 | 0.866 | 0.285 | 26 / 11 | 1.630 [1.523,1.708] | 30 |
| `stretch_slope` | L14 high q0.975 | 67 | 18 | 0.788 | 0.119 | 9 / 6 | 1.729 [1.513,1.890] | 30 |
| `pc1corr_slope` | L14 high q0.990 | **31** | **0** | **1.000** | 0.055 | 0 / 0 | 1.492 [1.328,1.492] | 38 |
| `pc1share_slope` | L14 high q0.980 | 24 | 10 | 0.706 | 0.043 | 5 / 8 | 1.410 [1.075,1.661] | 28.5 |
| （对照）`mobility` s9 | L12 low q0.975 | 195 | 17 | 0.920 | 0.346 | 36 / 4 | **1.764 [1.679,1.821]** | 34 |
| （**阴性对照**）`saa_slope` | L14 high q0.980 | 41 | 21 | 0.661 | 0.073 | 15 / 18 | 1.559 [1.266,1.807] | 20.5 |

按预注册判据（lift CI 不含 1 且早 TP ≥ 20），`td_slope` / `centred_slope` / `td_late_early`（三者 ρ≈0.999，是同一个量）**通过**。但必须同时说三件事：

1. **没有打败 mobility。** lift 1.631 [1.526,1.708] vs 1.764 [1.679,1.821]，区间几乎不重叠且方向不利；recall 0.285 vs 0.346。
2. **阴性对照也"通过"了一半。** 被判定为"跨 step 完全平坦"的 `state_action_alignment`，其斜率在 per_task 下拿到 88 TP / 13 FP、lift 1.560 [1.418,1.653]。**在 11 representation × 2 方向 × 14 分位的搜索下，lift ≈ 1.5 本身不构成强证据。**
3. **后层斜率与终点电平高度冗余。** Spearman(`td_slope`, `token_differentiation_s9`) external L12 0.969、L14 0.978、L15 0.882；与 `mobility_s9` 0.67–0.75。把电平跑同一协议：`token_differentiation_s9` global 选到**同样的 L15 low**、相邻分位 q0.975，external 只有 **15 TP / 5 FP**。同层同向、ρ=0.88 的两个量给出 161 vs 15。这不是信息增益，是**单一 global 分位阈值脆弱**：差分量越阈率 dev→ext 是 3.99%→4.26%（比 1.07），电平量 2.49%→1.85%（比 0.74）。

**A3. 状态锚就是"共同分量"本身，圆弧长在它的正交补里。**

- state token 与 action 质心方向夹角（s9 中位数）：后层 **8.9°–11.0°**，前层 22.7°–28.4°。
- 圆弧主轴 PC1 与 state 方向 |cos|：后层 0.023–0.025（≈88.7°），前层 0.059–0.062（≈86.5°）。**近乎正交。**
- 沿 state 的构型能量占比：前层 0.53%–0.74%，后层 0.076%–0.104%。**Schur 补丢掉的正是这不到 1%。**
- 但这 1% 不是噪声：`state_index_corr` 中位数 −0.07 至 −0.42（L3 从 s0 的 −0.227 单调走到 s9 的 −0.423）。**越靠后的 action token 越不像当前观测——被丢掉的分量编码"距当前观测的时域距离"。**
- 单独作检测器：`along_state_energy_s9` per_task back_median low q0.950 → **34 TP / 4 FP，lift 1.747 [1.481,1.871]，早 16/4，中位 chunk 16**（比 mobility 早）。global 仅 21/4。

**A4. 那个未决矛盾：被代数解决了。** 对 raw Zarr 逐位验证（最大误差 2.5e-8）：

```
conditional_energy       = 1 − mean_t(c_t²)                  ← 纯状态对齐量
中心化构型尺度 tr(HGH)/10 = 0.9 · (1 − action_consensus)       ← 纯 action 共识量
沿 state 的中心化能量      = var_t(c_t) = 1 − CE − SAA²
tr(HGH)/10 = var_t(c_t) + tr(H·cond·H)/10                    ← 精确正交分解
```

**已发表的 `conditional_energy` 根本不测量 token 构型**，它只是 c 向量的二阶矩。两者算的是不同的东西，方向相反不构成矛盾。external 生存条件 AUC（s9，>0.5 = 越大越危险）：

| 量 | L3 | L12 | L14 | L15 |
|---|---|---|---|---|
| `conditional_energy` | 0.4516 | 0.5300 | 0.4980 | 0.4621 |
| `centred_energy`=0.9(1−AC) | 0.4806 | **0.4175** | **0.4152** | **0.4127** |
| `orthogonal_energy` | 0.4808 | 0.4174 | 0.4152 | 0.4127 |
| `along_state_energy` | 0.4641 | 0.4670 | 0.4647 | 0.4601 |
| `action_consensus` | 0.5194 | 0.5825 | 0.5848 | 0.5873 |

**缩小的是正交（Schur）分量。** `orthogonal_energy` 与 `centred_energy` 四位小数完全相同，沿 state 的 0.08% 对信息毫无贡献。**我复现不了"更大的中心化构型尺度（AUC 0.538–0.626）"**：按写死的定义 external 上是 0.4127–0.5137，方向是**更小**。若那条指的是 `action_consensus`（0.4863–0.5873）则符号一致，只是名字反了。

### B. 状态通道本身（次问题）

**B1. 是检测器，但严格弱于 action mobility，且不是独立参考系。**

| 头 | 模式 | 层/方向/分位 | TP | FP | prec | recall | 早 TP/FP | lift [CI] | 虚警依赖 |
|---|---|---|---|---|---|---|---|---|---|
| `state_mobility_s9` | global | L3 low q0.990 | 124 | 8 | **0.939** | 0.220 | 15 / 1 | 1.494 [1.407,1.541] | **663×** |
| `state_mobility_s9` | per_task | L2 low q0.960 | 193 | 39 | 0.832 | 0.342 | 68 / 25 | 1.505 [1.408,1.582] | 115× |
| `mobility_s9`（基线） | global | L12 low q0.975 | 195 | 17 | 0.920 | 0.346 | 36 / 4 | 1.764 [1.679,1.821] | — |

precision 0.939 是本 bundle global 模式最高（超过 mobility 的 0.920），但 lift CI 与 mobility 完全不重叠，recall 只有 0.220。**按预注册判据 B 判负，卡在独立性一项**：8 个虚警里 6 个与 mobility 重合（独立假设期望 0.009 个）。这是同一个参考系的低噪声版本，不是新参考系。

无阈值信息量同结论：external 生存条件 AUC，`mobility_s9` L14 = 0.3853（|gap| 0.1147）vs `state_mobility_s9` L13 = 0.3918（0.1082）。**状态通道信息量与 action 通道基本相当、略低。1.6–4.9 倍是尺度上的，不是信息量上的。**

**B2. 唯一超基线的是"状态减 action"，且在早期带。** `state_minus_action_mobility` per_task back_median low q0.950 → **212 TP / 48 FP，早 74/32**，对比 mobility per_task 272/57、早 72/43。**早期真阳更多（74 vs 72）且早期虚警更少（32 vs 43）**，lift 1.552 [1.454,1.633] vs 1.389 [1.314,1.451]。

**B3. `libero_object` global 模式全线失效**（44 风险，mobility global 抓 2 个，本 bundle 所有 global 头抓 0）。控制器报的耦合 AUC 0.202 我方向上复现（L12 0.230、L2 0.275、L15 0.244），但**不是 object 特有**：同 chunk 位置 goal 0.286、spatial 0.313。是 chunk 位置效应，不是 suite 效应。

## 0. 锚点（全部通过）

```
mobility_s9 global  : L12 low q0.975 -> 195 TP / 17 FP, precision 0.9198, lift 1.7641   ✓
mobility_s9 per_task: L2  low q0.700 -> 272 TP / 57 FP                                   ✓
load_entropy_s9 per_task: L3 low q0.850 -> 370/93, lift 1.5479 (= expert_load_eff_rank)  ✓
代数恒等式对 raw Zarr 最大误差 2.47e-08                                                    ✓
arc 几何与 v4 缓存逐行对齐（221,781 / 31,941 / 248,255 query）                             ✓
生存先验 horizon 30/52/28/22，先验越过 0.25 于 chunk 18/26/17/13                           ✓
生存 AUC 实现复现已发表值：mobility L12 ext 0.385871、conditional_energy L3 ext 0.451608   ✓
```

协议不是重写的：`sys.dont_write_bytecode = True` 后直接 import `moe-flow-semantics-0906/experiments/{protocol,detect_step_alarm}.py`。未修改任何既有目录。

## 1. 控制器四条探针的独立复核

| 探针 | 结论 |
|---|---|
| **1a 状态路由 chunk 内不变** | **复现且更强。** state 路由 s0→s9 Hellinger 中位数：L2–L13 **恰好 0.000000**（float16 下逐位相同），L14/L15 0.000244；45 个 step 对最大值中位数 0–0.00072。同期 action 走了 **0.0358–0.0426**。声称的 2e-5–4e-4 只对后层成立，前层是精确 0。 |
| **1b 跨 query 迁移率** | **精确复现。** external 全部 232,655 有效 query：state 中位 0.0986–0.1736（声称 0.099–0.174 ✓），action 0.0328–0.0606（✓），比值 **1.63–4.91**（✓），最大在 **L2**（4.91 ✓）。 |
| **2 对齐平、分化涨** | **精确复现（均值口径）。** `token_differentiation` s0→s9：L15 **+151%**（声称 +150%）、L12 **+154%**（+156%）、L2 **−18.7%**（−18%）。`state_action_alignment` L2 +0.036%、L15 −0.18%。external 重复。 |
| **3 终点一维有序弧** | **部分复现，一条不成立。** PC1-序号 \|Pearson\| 中位 0.912–0.978（声称 0.958–0.998，前层达不到下界）；\|Spearman\| 0.964–0.988。**"归一化拉普拉斯 Fiedler \|Spearman\| 中位数 8 层全 = 1.0000"不成立**：归一化+raw Gram 给 0.9636–0.9879，从未到 1；Schur+归一化给 0.8909–0.9879。只有**未归一化组合拉普拉斯 + raw Gram** 在 **L12–L15** 达到 1.0000（恰好 1 的比例 51.5%–58.8%），前层 0.9636–0.9879。 |
| **4 耦合前缀因果** | 见 §4，比声称的更糟。 |

## 2. 形成过程的完整刻画

**前后层造弧方式完全不同。** 后层（L12–L15）step 0 已是有序弧（0.94–0.95），去噪做的是**把弧撑大**：`centred_energy` L15 +184%、`token_differentiation` +151%、`erank_centred` 0.454→0.353；有序性只从 0.939 涨到 0.973。前层（L2）step 0 是各向同性团，去噪做的是**把团排成序**：有序性 0.399→0.913、`arc_stretch` 1.296→3.037，而**总分化量反而下降** 0.00367→0.00291（−21%）。

**这是本工作最干净的新观察：前层在去噪中"分化量下降但有序度上升"——token 之间变得更像，但排列变得更整齐。** 只看 `token_differentiation` 会把这解读为"前层什么也没做"。

**状态对齐全程是常数**（L2 0.91011→0.91032、L15 0.98862→0.98694）：10 个 action token 既不靠近也不远离锚点，只是彼此分开并沿一条与锚点正交的轴排序。

## 3. 检测器补充

- **`pc1corr_slope` global：31 TP / 0 FP，15,036 timely episode 零虚警。** 但中位 chunk 38、早 TP 0，且 31 个风险 **mobility 全部已抓**（`tp_missed_by_mobility = 0`），OR 组合与 mobility 完全相同（195/17）。**零虚警但零增量。**
- **`td_slope` 有真实互补但代价是 lift。** 抓到 45 个 mobility 漏掉的风险；OR(mobility, td_slope) = **240 TP / 35 FP，recall 0.426（vs 0.346），早 44/12（vs 36/4），lift 掉到 1.623**。
- **suite 结构**：global 模式基本是 libero_long 检测器。mobility global 195 TP 中 156 来自 long、37 goal、2 object、0 spatial。`td_slope` global 161 TP 中 111 long、**35 spatial（mobility global 在 spatial 上是 0）**、15 goal、0 object。这 35 个是 `td_slope` 唯一无可替代的贡献。
- **物理失效模式**（564 个 external 风险 100% 有标签，仅分析用）：`td_slope|global` 对 `stable_grasp_not_observed` 召回 0.531（143 例），对 `object_released_or_dropped_before_goal` 只有 0.195（241 例）；mobility per_task 相反（0.252 / 0.631）。**"没有建立稳定抓取"在 chunk 内形成过程里可见，"抓住后掉落"不可见。**

## 4. 状态-action 耦合：判负

| 窗口 | 层 | goal | long | object | spatial |
|---|---|---|---|---|---|
| **整段 episode** | L2 | 0.809 | 0.627 | 0.505 | **0.900** |
| | L12 | 0.723 | 0.795 | 0.473 | 0.845 |
| **前缀 K=12（幸存条件）** | L2 | 0.518 | 0.407 | **0.275** | 0.420 |
| | L12 | 0.286 | 0.484 | **0.230** | 0.313 |
| | L15 | 0.244 | 0.552 | 0.244 | 0.306 |

整段 AUC 高到 0.90，前缀因果版本**符号相反**（0.23–0.31，耦合越低越危险）。控制器报的 0.489 大概是跨 suite 池化抵消。

把耦合做成完全因果的逐 query 分数（`running_coupling`：queries 1..q 上 action 与 state mobility 的展开窗 Pearson）跑同一协议：

- **无阈值信息量在 external 上是全项目最强单量**：L13 生存条件 AUC **0.6196**（|gap| 0.1196 > mobility L14 的 0.1147 > norm_fiedler L12 的 0.0969）。
- **但按 chunk 分解符号会翻。** L12 libero_goal：chunk 3 的 0.504 → chunk 10 的 **0.235** → chunk 20 的 **0.623**；libero_object 同形（0.551→0.173→0.491）。池化的 0.62 是被全程正号的 libero_long 撑起来的。
- **dev→ext 落差大**（0.5431 → 0.6196）。
- **跑冻结协议后什么也不剩**：global 1 TP / 1 FP，per_task 2 TP / 1 FP。`fixed4_coupling` per_task 37/13 lift 1.924 [1.571,2.187] 早 19/10，global 51/18 lift 1.460，都没打过 mobility。

**结论：状态-action 耦合不是检测器。** 整段效应是生存时长伪影，前缀效应符号相反，逐 chunk 符号不稳定，train-free 协议下选不出可用工作点。

**顺带的方法学告警：`mobility` 自己的逐 chunk AUC 也强烈翻号**（L12 libero_object：chunk 9 是 **0.885**，chunk 13 是 0.308）。池化生存条件 AUC 在这个数据集上掩盖了巨大的 chunk × suite 异质性，不该单独作为"信息量"结论的依据。

## 5. 对"非平衡结构"框架的判定

**支持的一半**：风险确实伴随中心化构型**变小**（external AUC 0.4127–0.4175 在 L12–L15），且缩小完全落在与状态方向正交的子空间；chunk 内**建起来的分化量更少**（`td_slope` low → 风险）是同一件事的动力学版本，external global 161/24、lift 1.631。

**被证伪的一半**：形状的一维性并不向对称退化，反而**更极端**——`stretch_slope` high → 风险（AUC L14 0.5693）、`pc1corr_slope` high → 风险（L15 0.5523）、`pc1share_slope` high → 风险（L14 0.5593）。风险签名是"造出来的结构总量更少，但那点结构被排得更一维、更拉伸"，不是"塌回各向同性团"。

**削弱**：被判定为平坦的 `state_action_alignment` 其斜率作为阴性对照也拿到 lift ≈ 1.56 的两个头。

**净判定：框架的"量"这一半得到支持，"形状"这一半被证伪；而且支持的那一半在检测指标上没有超过 mobility。**

## 6. 失败与限制

1. **B 判负**：lift 1.494 严格低于 mobility 1.764（CI 不重叠），虚警依赖 663×。
2. **A 是弱通过**：阴性对照几乎同样通过；后层斜率与终点电平 ρ = 0.97–0.98。
3. **`pc1corr_slope` 的 0 虚警漂亮但无用**：增量 0，报警 chunk 38。
4. **单一 global 分位阈值脆弱**：ρ=0.88 的两个同层同向量给出 15 vs 161。
5. **`libero_object` global 模式全线失效**（44 风险，所有 global 头 0 TP）。
6. **`running_coupling` 的 0.62 不可用**：逐 chunk 翻号、dev/ext 落差、协议下 1 TP。
7. **探针 3 的"8 层全 = 1.0000"不成立**，只在 L12–L15 且用未归一化组合拉普拉斯时成立。
8. **未做**：逐 episode 的"形成时刻"分布；`arc_stretch` 与真实位姿的对应。

## 7. 下一步

1. **把"形成时刻"本身做成量**：现在的斜率是全 10 步 OLS。L2 在 step 4 越过零假设——**每个 query 各自的越过步数**（0–9 的整数）是完全 chunk 内、完全因果、且与电平正交的量，还没算过。
2. **抛弃单一 global 分位**：用秩变换或两点分位插值做阈值迁移，直接检验"15 vs 161"是不是纯阈值伪影。
3. **追 `state_minus_action_mobility` 的早期带优势**（早 74/32 vs mobility 的 72/43），唯一在早期带同时改善 TP 和 FP 的头。
4. **逐 chunk 分层的检测器**：信号强烈依赖 chunk 位置，冻结协议的单阈值设计正在丢掉这一维。

## 文件

```
experiments/  arc_lib.py  extract_arc_geometry.py  sweep_lib.py  analyze_formation.py
              run_detectors.py  verify_controller_probes.py  analyze_increment.py
              analyze_coupling_causal.py  build_manifest.py
results/      PREREG.md  manifest.json  headline_table.csv
              arc/ formation/ detectors/ controller_probes/ increment/ coupling/
tests/        test_anchors_and_identities.py
```

计算：CPU only，`hb_router_probs` 全量读取一次（501,977 query 行，约 22 GB），外加 8 个 external 任务 × 400 行用于探针复核。未占用 GPU。
