# 跨 bundle 审计：分层、一致性、冗余、矛盾

日期 2026-09-06。只读审计，未修改任何既有目录，未跑新的检测器实验。
所有数字要么直接读自产物文件（标注 **[读]**），要么在本目录重算（标注 **[算]**），
要么引自报告未独立核实（标注 **[引]**）。机器可读表在 `results/`。

---

## 0. 结论先行

**各个 bundle 看不见的那件事：任务混淆不是失败模式那一层的局部问题，它是整个语料的普遍性质。**

`moe-failure-modes-0906` 发现 261 个（检测器 × 模式）格在 suite 分层下 20 个显著、
task 分层下只剩 2 个。我把**同一个检验换到完全不同的对象上**——33 个路由图量 × 8 层的
无阈值生存条件 AUC，不涉及阈值、不涉及模式标签——得到**结构完全相同的结果**：

| | suite 分层 | task 分层 |
|---|---|---|
| \|AUC−0.5\| >= 0.04 的（量, 层）格 | **52 / 264** | **1 / 264** |
| 中位收缩 | — | **80.8%** |
| 符号翻转 | — | **24 / 52** |

**[算]** `results/audit_stratification_survival_auc.csv`。实现先对着两个 bundle 校验过：
复现 `moe-token-geometry-0906` 的 288 个格全部 < 2e-3；复现 `moe-circuit-analogy-0906`
的 `norm_fiedler|L12` = 0.5969、`gap_ratio|L12` = 0.5780、`spectral_erank|L12` = 0.5760、
`kirchhoff_efficiency|L12` = 0.5764、`vol|L3` = 0.4521，四位小数全中。**唯一改动的是分层变量。**

**suite 分层不是弱一点的 task 分层，它基本什么都没控制住。** 很多格的 suite 分层效应比
不分层还大（`mobility|L14`：不分层 0.116、suite 0.203、task 0.049）。项目里几乎所有
"分层过了"的说法指的都是 suite。

---

## 1. 分层审计（`results/stratification_audit.csv`，23 条实质主张）

### 1.1 类别 (iii) 里会改变结论的

| 主张 | 出处 | 后果 |
|---|---|---|
| **`norm_fiedler` 是全项目第二强的单一 MoE 量（AUC 0.5969）** | circuit §4 | **[算]** 换 task 分层 -> **0.5338**，效应掉 **65%**。仍是 33 个电路量里 task 分层下最大的，但离"第二强"很远。 |
| **风险 = 向均匀团坍缩，7/7 尺度无关不变量同向** | circuit §4.2 | **[算]** 七个量的 suite 分层值全部在 L12 逐位复现（0.5969 / 0.5780 / 0.5764 / 0.5760 / 0.4267 / 0.4311 / 0.4371）；换 task 分层后 **7 个里 6 个符号翻转**（gap_ratio ->0.4877、kirchhoff_efficiency ->0.4856、spectral_erank ->0.4853、res_cv ->0.5158、th_cv ->0.5268、thalt_cv ->0.5207），只有 `norm_fiedler` 保号（->0.5338）。**"7/7 同向"是 suite 层面的假象。** |
| **报警后单调恢复曲线 0.380 / 0.415 / 0.403 / 0.514 / 0.657** | hb-front-back §Stage 2 | **[读]** 池化 409 个 episode，用的是 **row bootstrap 不是 task bootstrap**，CI 下界 0.523、+8 已经是 0.514——**两个数都已落在控制器报的 0.49–0.53 塌陷带内**。只有 27 个阳性，task 分层大概率无法判定。这是最脆弱的一条。 |
| **mobility 是最强单一 MoE 量（\|AUC−0.5\| = 0.1147）** | circuit §4 | **[读]** `moe-token-geometry-0906` 自己的 within-task 表给出 mobility 平均 \|AUC−0.5\| = **0.0380**（对不分层 0.1067）。**两个 bundle 各自算出了这两个数，没有人把它们并排放过。** |
| **组合规则赢在跨 suite 并集覆盖** | combination §3 | **[读]** bundle 自己点名机制是 suite 覆盖（`mobility|global` 在 libero_spatial 上 0/140）。task 层面是否还有增益，未测。 |
| **非终点去噪步带来终点步没有的检出（step 0 多抓 96 个）** | flow-semantics §6 | **[读]** 池化计数，可能就是同一个 suite 覆盖机制。未测。 |
| **v7 439 TP / 80 FP、precision 84.59%** | v7 README | **[读]** LOTO §3.2 的 39 任务等权版本是 **72.30% / 78.14%**，即 −5.54pp / −6.45pp，**与该 bundle 当作主发现的整个 LOSO 退化同量级**，却被降格为"仅作参考"。 |

**LOSO / LOTO 分的是校准语料，不是评测比较。** 它们证明阈值能迁移到没见过的任务；
它们**不**证明分数在任务内部有判别力。这是 v7 系列七份文档的共同盲区。

### 1.2 类别 (iv)：结构上免疫，可以信

前后层比值 6.17（40/40、39/39、5/5 任务同向，task bootstrap CI [5.76, 7.06]）、
十个 token 的一维弧、十步去噪的噪声/观测梯度、`state_action_alignment` 与 `conditional_energy`
的代数恒等（`moe-state-channel-0906` 数值核验到 **2.54e-08**）、
v7 的 `mob_ratio`（相对 rollout 自身 q1–q4 基线）。
这些都是"同一次前向内部比较"或"同一 episode 内部比较"，不暴露于任务混淆。

### 1.3 类别 (i)：真做了 task 分层的只有两处

- `moe-failure-modes-0906` §2–§3 的置换零假设。
- `moe-token-geometry-0906` §6 的 within-task 生存条件 AUC。**[算]** 我从缓存核重算，288 个格全对上。

---

## 2. 交叉模式核对

**"能用的量都是相对量" —— 基本成立，但要修正。**
`conditional_energy` / `vol` / Kirchhoff / 生成树数这一整族**绝对标度**量，
task 分层下 |AUC−0.5| 全部 <= 0.007（L3：0.5062、0.5065、0.5043）**[算]**。
相对量里活下来的是 `norm_fiedler`（度归一化）和 `size`（中心化后）。
但"相对"不是充分条件：`gap_ratio` / `spectral_erank` / `kirchhoff_efficiency` / `res_cv`
都是尺度无关的相对量，task 分层下照样归零并翻转符号。

**"组合只买覆盖不买信息密度" —— 成立。**
**[算]** 扫 6 个 bundle 的全部 external 工作点：TP>=150 时最大 lift = **1.7641**（mobility 195/17）；
TP>=100 时 **1.8711**（`load_entropy_s3|global`，109/31）；TP>=20 时 **2.4614**
（`partial_edge_std|global`，42/35，circuit bundle）。lift 随 TP 单调下降，无一例外。

**但"1.764 是纪录、无人打破"这句话本身是错的。**
**[算]** TP>=20 的口径下有 **46 个** external 工作点 lift > 1.7641。
正确说法是：**mobility|L12|global 在 TP>=150 的条件下保持纪录**；1.764@195TP 是帕累托点，不是最大值。

**"单 episode 内比较活得下来，跨 episode 比较会溶解" —— 成立，且是本审计最干净的一条。**
1.2 节全部活下来，1.1 节全部有问题。

**"horizon cap 让每条风险轨迹都跑满上限，所以'还在跑'本身就是强证据" —— 成立，且比报的更强。**
**[算]** external 15,600 条：564 条 risk **全部**长度恰等于其 suite 上限；反过来跑满上限的有 568 条，
多出的 4 条是在最后一个 chunk 才成功的。所以**"跑到上限还在跑" => risk 的精确率 = 564/568 = 0.9930**。
`moe-v7-0905/docs/SURVIVAL_BASELINE` 报的"存活先验贡献 60.6% 准确率"是**下界**——它的先验按 suite 估，
按 task 估只会吞掉更多。

---

## 3. 数字一致性（`results/consistency_check.csv`）

四个锚点全部对上，**[算]** 逐位核验：

- `mobility|L12|low|q0.975|global` = 195 TP / 17 FP、lift **1.7640674271437853** —— **7 个 bundle 完全一致**。
- `mobility|L2|low|q0.700|per_task` = 272 / 57、lift **1.3886047537089383** —— 7 个 bundle 一致。
- v7 guard = 439 TP / 80 FP —— 至少 8 处一致，LOSO §2.2 声明逐位复现。

**不对上的：**

1. **`expert_load_effective_rank|L3|low|q0.85|per_task` 的 lift 有两个值。**
   `1.5479067335088585`（frame_survey / circuit / combination / failure-modes，4 个 bundle）
   vs `1.5478620878761244`（flow-semantics 的 `load_entropy_s9`、state-channel）。
   TP/FP/precision/早期计数全部相同，差异只在匹配存活先验的第 5 位小数
   （0.5162688757758398 vs 0.5162837667346665）——两条代码路径各自重算了先验。
   **该引 1.5479067**（4 个 bundle 的共识）。量级无关紧要，但意味着 state-channel 的 PREREG
   把 "lift 1.5479" 写成必须逐位复现的锚点时，它自己的实现其实差了 4.5e-5。

2. **`partial_edge_std|global` 在两个 bundle 里是两个不同的检测器。**
   frame_survey：L3/high/q0.90 -> **99 TP / 147 FP**，lift 2.0094。
   circuit：L3/high/q0.95 -> **42 TP / 35 FP**，lift 2.4614。
   两者都声称用逐字相同的冻结选择规则，且都精确复现了 mobility 与 expert_load。
   同一个量、同一个模式却选到不同分位、TP 差 2.4 倍。**没有哪个是"对的"——选择规则在跨 bundle
   之间不是确定性的。** `conditional_effective_rank|global` 同样：88/158 vs 42/62。

3. **"四检测器并集 525/564、只漏 39"在 `moe-hb-front-back-0905` 里不存在**（已 grep）。
   `moe-combination-rules-0906` §5.4 复现不出，控制方复算为 **525 TP / 198 FP、precision 0.726**。
   原始表述只报覆盖率不报 198 个假警报，是误导性的。**应引 525/198。**

4. **`moe-v7-0905/docs/SURVIVAL_BASELINE_REPORT_ZH.md` §4 算术错误**：正文"264 次报警（86%）落在
   chunk 20 之后"，但同一张表自己加起来是 115+119+25 = **259**（84.9%）。列合计 305 与 §1 吻合，
   配套的"46 次（15%）"也自洽，所以错的是正文那个 264。

5. **控制方关于 Fiedler 向量的前提太强。** "归一化拉普拉斯 Fiedler 向量在全部 8 层 median
   |Spearman| = 1.0000" —— **[读]** `moe-state-channel-0906/results/controller_probes/probe_summary.json`
   实测取决于用哪个矩阵：`gram_combinatorial` 只有后四层是 1.0（前层 0.9636–0.9879）；
   circuit 的 `norm_fiedler` 对应的是 `schur_normalised` = **0.8909–0.9879**；
   未归一化的 `schur_combinatorial` 在前层只有 **0.3939–0.5515**。
   所以 `norm_fiedler` 与几何 bundle 的弧序度量**在后层是同一个统计量，在前层不是**。

---

## 4. 冗余地图（`results/redundancy_map.csv`）

**6 个真正不同的量，49 个名字。**

| 等价类 | 名字数 | 证据 |
|---|---|---|
| **E1 残差路由总能量（标度）** | **19** | `conditional_energy` = `state_action_alignment`(负) = `ground_leak_mean`(ρ=1.0000) = `kernel_trace`(×10) = `lam_max`(.9991) = `res_chain_series`/`res_adjacent_mean`(.9987) = `vol`(.9986) = `log_spanning_trees`(.9985) = `log_kirchhoff`/`res_mean`(.9984) = `res_end2end`(.9962) = `fiedler`(.9945) = `th_{mean,min,max,first,last}`(.9845–.9886) = `commute_time`(定义冗余) |
| **E2 相对边形状（尺度无关）** | **12** | `partial_edge_std` ≈ `norm_fiedler`(.9740) ≈ `dd_deficit` ≈ `gap_ratio`(.8672) ≈ `spectral_erank`(.8638) ≈ `kirchhoff_efficiency`(.8637) ≈ `res_cv`(.8407)；**且后四者内部两两 >= 0.992，是同一个量** |
| **E3 token 弧序** | 5 | `bandedness` / Robinson / `pc1_index_corr` / `fiedler_index_rho` / `arc_stretch` |
| **E4 中心化构型尺度** | 3 | `size` = `centred_energy` = 0.9(1−`action_consensus`)，代数恒等核验 2.29e-08 |
| **E5 跨 query 路由变化** | 4 | `mobility` / `mobility_step` / `conditional_query_d1` / v7 `mob_ratio` |
| **E6 专家负载集中度** | 2 | `expert_load_effective_rank` = `load_entropy_s9` |

**各 bundle 自己没看见的新增两条：**

- **`kernel_trace` 就是 `conditional_energy`。** **[算]** 288 个生存 AUC 格逐格相同。
  `moe-token-geometry-0906` 把它当自家新量评了一遍，`moe-circuit-analogy-0906` 同时把
  `conditional_energy` 当已发表对照评了一遍，**两边都不知道对方在算同一个东西**。
  这直接产生了第 5 节那条"矛盾"。
- **`gap_ratio` / `spectral_erank` / `kirchhoff_efficiency` / `res_cv` 是一个量**，不是四个。
  circuit §4.2 的"7/7 一致"里有 4 个是同一条证据。

---

## 5. 矛盾（`results/contradictions.csv`）

### X01 —— 已解决：那从来不是矛盾，是分层口径错配

"风险伴随**更低** `conditional_energy`（AUC 0.4516）但**更高**中心化构型尺度（0.538–0.626）"。

**[算]** 从同一份缓存核、用同一套实现，把两个量放到同一个分层下：

| 分层 | `conditional_energy` L3 | `centred_config_size` L3 |
|---|---|---|
| suite | **0.4516** | **0.4808** |
| task | **0.5062** | **0.5091** |

**同一分层下方向一致，没有矛盾。** 0.4516 是 suite 分层数（circuit bundle），
0.538–0.626 是 task 分层数（geometry bundle）——两边根本没在比同一件事。
而且 `conditional_energy` 是**符号翻转格**：suite 下 < 0.5，task 下 > 0.5。

"总核能量 vs 去掉共同成分后的方差"这个框架也一并溶解：**[算]** 共同成分
（C 全部元素的均值）与 `conditional_energy` 在每一层都吻合到小数点后三位——
因为 circuit bundle 自己 §2 已经证明这张图是近似均匀完全图（中位 gap_ratio 0.9035、
谱有效秩 0.99953），均匀团只有"总电导"一个自由度。

### 其余六条

- **X03 参照系是否独立**：**已判定为否**。FRAME_SURVEY §4 推翻同 bundle 内 FLOW_AXIS §2
  的"只有 2 个共同误报"；circuit §5.3 独立测到虚警依赖度 46.55。
  **永远报依赖度比值，不要报原始重叠数。** FLOW_AXIS 那句"其余结论不受影响"没有依据。
- **X04 `flow_settling` 是不是 MoE 侧信号**：**已判定为否**。per_task 阈值 83 个早期 TP，
  global 阈值只剩 **3 TP / 2 FP**。**这是本语料里最大的一次任务标定塌陷**，
  和 20/261 -> 2/261 是同一个现象的另一面。
- **X02 最好的 mobility 层是 L2 还是 L12**：未解决但可解释——两种阈值模式各自独立扫层。
  **[算]** task 分层下 L2 = 0.0377、L12 = 0.0466，后层略高，都很小。
- **X05 组合是否买到信息**：不是矛盾，两边都对，bundle 自己也说了。
- **X06 抓取头是否有名字**：不是矛盾——效应是真的但**只在晚期**（no_grasp 早期召回 global 与
  per_task 都是 0.035）。应表述为"晚期确认头"，绝不能说成早期预警。
- **X07 v7 是否任务无关**：**未解决**。LOTO 证明阈值能迁移，不证明分数在任务内可判别。

---

## 6. 真正站得住的东西，按稳健性排序（`results/established_ranked.csv`）

1. **所有风险 episode 跑满 horizon cap；"跑到上限还在跑" => risk 精确率 0.9930。** **[算]** 结构性事实。
   威胁：建立在它之上的一切——报告的精确率里有很大一块就是这个基率。
2. **失败模式分工是任务混淆；唯一稳健的是"参考系 -> stable_grasp_not_observed"。** task 分层通过。**[引]**
3. **后层更刚性、更一维有序、更低维、相对分化更强。** 两个 bundle 两份独立缓存，
   40/40 + 37/37 任务同向。**结构免疫。** 威胁：是两个小残差之比，绝对量从未报过。
4. **十个去噪步是单调梯度；早步听噪声、晚步听观测。** 8/8 层、40/40 + 39/39 任务，
   两个 run_id 穿零点都在 step 3->4。**结构免疫。**
5. **几乎所有路由图量只是两个量。** 无监督，与标签无关，不受分层影响。**[算]**
6. **存活先验独占大部分表观精确率**（>=60.6%，spatial lift 0.998）。suite 分层，
   但未做的修正只会让它更强。**[引]**
7. **失败前几何形变：弧的一维秩序退化 + 构型变大。** **[算]** 我独立重算确认。
   **这是整个语料里唯一一个尺寸可观、且构造上就是 task 分层的效应**
   （bandedness 平均 \|AUC−0.5\| = 0.086、size = 0.076；对照 mobility 只有 0.038）。
   威胁：效应小，且转不成有竞争力的报警器。
8. **组合只买覆盖不买信息密度。** **[算]** 6 个 bundle 全扫，无例外。
9. **v7 = 439/80。** 数字没问题，**条件严重欠报**：任务等权是 72.30%/78.14%；
   未见 suite + 8 chunk 干预余量是 47.34%/70.63%。
10. **`norm_fiedler` 是第二强单一量 —— 作为标题已被推翻。** **[算]** 0.5969 是 suite 分层；
    task 分层 0.5338。52 个 suite 分层显著格里只有 1 个在 task 分层下存活，就是
    `norm_fiedler|L15`（0.0479 -> 0.0442）——本身就是最弱的那批之一。

---

## 7. 建议（三条，都是可预注册的）

1. **把 task 分层设为默认口径。** 生存先验按 task 估而不是按 suite 估；lift 用 task 先验；
   凡是报 suite 分层的地方并排报 task 分层。现有的 suite 分层数字应全部加注"未控制任务"。
2. **重跑 §Stage 2 的报警后恢复曲线（0.380->0.657）**，用 task bootstrap 而不是 row bootstrap。
   只有 27 个阳性，很可能得到"无法判定"——那也是结论，而且比现在这条叙事诚实。
3. **把 `size` / `bandedness` 做深，而不是 `norm_fiedler`。** 它是唯一在 task 分层下
   还有 0.07–0.11 效应的量，且 circuit bundle 追的那条线在 task 分层下 65%–100% 归零。
   注意 `size` 与 `conditional_effective_rank` 重叠 0.755，先做残差检验。

---

## 8. 产物

```
moe-audit-0906/
├── REPORT_ZH.md
├── experiments/                              # 全部只读上游，只写本目录
│   ├── recheck_stratification.py             # 从缓存核重算 pooled/suite/task 生存 AUC（校验 288/288）
│   ├── recheck_circuit.py / recheck_circuit_exact.py
│   ├── audit_stratification_table.py         # 逐位复现 circuit 口径，只换分层变量
│   └── write_{audit_tables,redundancy_map,contradictions}.py
└── results/
    ├── stratification_audit.csv              # 23 条实质主张 × 分层类别 × 是否会改变结论
    ├── consistency_check.csv                 # 14 项跨 bundle 数字核对，5 项不一致
    ├── redundancy_map.csv                    # 6 个等价类，49 个名字
    ├── contradictions.csv                    # 7 条，X01 已用代数+重算解决
    ├── established_ranked.csv                # 10 条，按稳健性排序
    ├── audit_stratification_survival_auc.csv # 264 格 suite vs task，含收缩率与符号翻转
    ├── recheck_survival_auc.csv              # conditional_energy / size / common_component
    ├── recheck_circuit_{auc,metric_suite_vs_task}.csv
    ├── recheck_meta.json                     # horizon cap 核验：risk 属于 at-cap，564/564
    └── manifest.json                         # sha256 + 实现校验记录
```

**未完成 / 未覆盖**：`moe-two-tier-0906` 不存在；`moe-state-channel-0906` 无 `REPORT_ZH.md`，
本审计只用了它已产出的 `results/`（PREREG、formation、controller_probes、detectors），
其结论待其自身报告确认。`moe-v4-0904` / `moe-v6-0905` 只作为工作点来源核对，未逐条审其主张。
第 1.1 节除 `norm_fiedler`、团不变量、`conditional_energy`、mobility 之外的条目**只做了分类，未重算**。
