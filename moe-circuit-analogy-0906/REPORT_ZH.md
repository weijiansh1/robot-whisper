# 把 MoE 路由图当电路看：能不能得到有用的量？

**目录**：`/home/jovyan/work/himoe-vla/moe-circuit-analogy-0906/`
**日期**：2026-09-06　**队列**：development_main（14,800 / 487 risk）、development_extra（1,200，仅作阈值参考语料）、external_8b（15,600 / 564 risk）

---

## 0. 结论先行

**是，但只有一个量真正赢了，而且赢得不多。**

1. **电路类比在这个对象上是成立的，不是装饰。** `conditional` 在全部 **4,015,816** 个 (层, query) 单元上都严格正定（最小特征值 ≥ 4.52e-6），负的非对角元只占非对角绝对质量的 **4.7e-7**，因此把它直接读成电导矩阵几乎不丢东西。Foster 定理在 4,015,800 个连通单元上以 **1.6e-14（dev）/ 1.4e-12（ext）** 的精度成立；在剩下 **16** 个单元上它以 **恰好 k−1** 的偏差失效，k 是连通分量数——也就是说 Foster 残差就是分量计数器，这是一个真正的诊断量而不是自证。

2. **但绝大多数电路量只是已有量的重参数化。** Kirchhoff 指数、生成树数、Fiedler 值、全部 Thévenin 电阻，与项目里早已发表的 `conditional_energy` 的 Spearman 相关都在 **0.988–1.000** 之间（`ground_leak_mean` 与它是 **精确单调变换，ρ = 1.0000**）。原因是这张图是一张 **近似均匀完全图**：中位 `gap_ratio` 0.9035、中位谱有效秩 0.99953、中位 Kirchhoff 效率 0.99906、成对有效电阻的变异系数中位数仅 0.021。一张均匀团只有"总电导"这一个自由度，而总电导就是 `conditional_energy`。

3. **唯一真正加分的是归一化拉普拉斯的代数连通度 `norm_fiedler`。** 在无阈值的"生存条件 AUC"上，它是**全项目第二强的单一 MoE 量**（external L12：0.5969，|AUC−0.5| = 0.0969），仅次于 mobility（0.1147），并且**显著强于它最接近的已发表亲戚 `partial_edge_std`**（0.0847）：配对 cluster bootstrap 差值 dev **+0.0063 [+0.0044, +0.0083]**、external **+0.0122 [+0.0102, +0.0143]**，两侧都不含 0。未归一化的 `gap_ratio` 则没有这个增量（external −0.0066 [−0.0130, −0.0001]，反而更差）——**增量来自度归一化本身**。

4. **但在报警指标上它没有打败基线。** external / global 模式：`norm_fiedler` 42 TP / 2 FP，**precision 0.9545（全部 global 检测器中最高，高于 mobility 的 0.9198）**，lift **1.470 [1.307, 1.520]**，recall 0.0745；mobility 基线 195 TP / 17 FP，lift **1.764 [1.679, 1.821]**，recall 0.3457。置信区间不重叠，**mobility 严格更好**。预先声明的合取（`spectral_erank` ∧ mobility）得到 36 TP / 1 FP、precision 0.9730、lift 1.680，**仍低于 mobility 单独的 1.764**，且两者虚警的共现率是独立假设的 **46.6 倍**，说明没有互补性可挖。

一句话：**电路理论在这里提供了一个真实但边际的收益（一个更好的图形状指标），而不是一个新的失败检测器。**

---

## 1. 构造与它的合法性审计

### 1.1 对象

最后一步去噪时，第 t 个 token 的路由分布 p_t ∈ Δ^31，取 R_t = √p_t（单位范数、非负），Gram G = R Rᵀ 即 Bhattacharyya/fidelity 核。对状态 token（第 0 个）取 Schur 补：

```
C = G[1:,1:] − G[0,1:] ⊗ G[0,1:]        (10 × 10)
```

C 是残差方向 r_i = R_i − ⟨R_i, R_0⟩ R_0 的 Gram 矩阵，因此 PSD。

### 1.2 电路读法（在看 external 之前固定）

```
电导   w_ij = max(C_ij, 0),  i ≠ j
接地   状态 token 为参考节点，漏电导 g_i = 1 − C_ii = G_{0i}²
```

`g_i` 的取法是一个**建模选择**，理由是单位范数预算：每个动作 token 的 √p 向量范数为 1，被状态 token 吸收掉的（平方）份额 G_{0i}² 漏到地，剩下 C_ii 留给 token 间耦合。这样 L_g = L + diag(g) 正定，每个节点到地的 Thévenin 电阻无须伪逆即可定义。

### 1.3 三条必须讲清楚的话

**(a) Kron 归约只是形式上的同一个运算，不是字面意义的。** 电路里消去一个内部节点＝对拉普拉斯取 Schur 补；这里对 Gram 取 Schur 补。若 G 本身是拉普拉斯，二者就是同一件事。但 G 不是。要把 C 直接读成"接地拉普拉斯"，需要 C 对角占优（leak_i = C_ii − Σ_j C_ij ≥ 0）。实测**只有 7.3e-6（dev）/ 9.3e-6（ext）的行是对角占优的**，平均行外和超出对角约 **7.5 倍**。所以那条捷径是无效的，必须走 1.2 的预算式接地。

**(b) 负电导：丢弃的质量可以忽略。** 全部 4,015,816 个 (层, query) 单元中：

| 队列 | 单元数 | 含负元的单元 | 负元占比 | **负绝对质量占比** | 裁剪后不连通 |
|---|---|---|---|---|---|
| development_main | 1,774,248 | 217 (0.0122%) | 9.54e-6 | **4.72e-7** | 6 |
| development_extra | 255,528 | 38 (0.0149%) | 1.41e-5 | **1.25e-6** | 3 |
| external_8b | 1,986,040 | 278 (0.0140%) | 1.07e-5 | **4.70e-7** | 7 |

即**取正部丢掉的是总非对角绝对质量的约 5e-7**。因此本工作用 PSD 的 `conditional` 形式并取正部，不做绝对值、不做阈值。
注意：`partial_ij = C_ij / √(C_ii C_jj)` 且 C_ii > 0，所以 **sign(partial) ≡ sign(conditional)**——上表同时覆盖了 partial 形式的负值问题，两者负号出现的位置完全一样。

**(c) 罕见但真实的解耦事件。** 16 个单元（1.6e-6）在取正部后图断开，全部是"某个动作 token 的残差路由方向与其余每一个 token 都反向"。典型例子：`libero_goal/open_the_top_drawer_and_put_the_bowl_inside` episode 367 的 query 16，L15 层，token 10 的路由分布明显更尖（max prob 0.242，其余 token 约 0.04），18 个非对角元为负，节点 10 被孤立。

### 1.4 Foster 定理：当作数值恒等式的校验

对任意连通的非负权网络，Σ_{i<j} w_ij R_ij = n − 1 = 9。实测：

* 连通单元（4,015,800 个）：**最大 |残差| = 1.60e-14（dev）、7.37e-13（extra）、1.44e-12（ext）**，dev 上平均 2.58e-15。
* 不连通单元（16 个）：残差 **精确等于 k − 1**（`np.allclose` 校验通过）。

必须讲清楚：Foster 恒等式对**任何**非负权都成立，所以通过它验证的是**代码路径**（拉普拉斯、伪逆、有效电阻的一致性），**不是建模选择的物理性**。它唯一能"抓到"的失效是不连通——而它确实抓到了，且给出的偏差正好是分量亏损。这是本工作里 Foster 的全部用处。

同时校验了几个理论极值（只在连通单元上）：`kirchhoff_efficiency` = n(n−1)²/(2·Kf·vol) ≤ 1（由 AM–HM 不等式，等号当且仅当均匀团；实测最大 0.9999991）；`spectral_erank` ≤ 1；`gap_ratio` ≤ 1（最大 0.99748）；`norm_fiedler` ≤ n/(n−1) = 1.1111（最大 1.11084）。
**更正一处我原本以为的界**：`res_shortcut` = Σ_i R(i,i+1) / R(1,10) 在均匀团上等于 9，但 **9 不是上界**——实测 17% 的单元超过 9（最大 2565）。它只是"团参考值"，不是不等式。

---

## 2. 这张图长什么样：近似均匀完全图

development_main，1,774,248 个单元：

| 量 | q0.1% | q1% | 中位 | q99% | 最大 | 均匀团取值 |
|---|---|---|---|---|---|---|
| `gap_ratio` λ₂/λₙ | 0.4756 | 0.6190 | **0.9035** | 0.9839 | 0.9974 | 1 |
| `spectral_erank` | 0.9771 | 0.9895 | **0.99953** | 0.99999 | 1.0000 | 1 |
| `kirchhoff_efficiency` | 0.9471 | 0.9775 | **0.99906** | 0.99998 | 1.0000 | 1 |
| `norm_fiedler` | 0.8866 | 0.9747 | **1.0947** | 1.1097 | 1.1108 | 10/9 = 1.1111 |
| `res_cv` | 0.0020 | 0.0035 | **0.0211** | 0.1101 | 1.928 | 0 |
| `th_cv` | 0.00024 | 0.00056 | **0.0061** | 0.0231 | 0.156 | 0 |
| `n_near_zero` | 1 | 1 | **1** | 1 | 2 | 1 |
| `vol` 总电导 | 0.408 | 0.551 | **2.579** | 29.24 | 42.52 | — |

**这就是全部电路量高度冗余的原因**：这张 10 节点图几乎总是一张所有边权相近的完全图，唯一有大动态范围的自由度是总电导 `vol`（跨两个数量级），而 `vol` 与 `conditional_energy` 的 ρ = 0.9986。

前后层对比（标准化差，仅作**佐证**，不是新发现——已知事实 #2 已建立）：后层离均匀团更远。`norm_fiedler` −1.538、`gap_ratio` −1.526、`spectral_erank` −1.154、`kirchhoff_efficiency` −1.013、`res_cv` +1.411。方向与"后层携带更分化的 token 结构"一致。

---

## 3. 冗余：哪些电路量其实不是新的

development_main，200,000 个 query（随机子采样，seed 20260906）× 8 层 = 1,600,000 个单元，与 11 个已发表 HB 层图指标 + layerwise mobility 做 Spearman：

| 电路量 | 最近的已发表量 | \|ρ\| |
|---|---|---|
| `ground_leak_mean` | conditional_energy | **1.0000**（精确单调变换） |
| `lam_max` | conditional_energy | 0.9991 |
| `dd_deficit` | conditional_effective_rank | 0.9989 |
| `res_chain_series` / `res_adjacent_mean` | conditional_energy | 0.9987 |
| `vol` | conditional_energy | 0.9986 |
| `log_spanning_trees`（矩阵树定理） | conditional_energy | 0.9985 |
| `log_kirchhoff` / `res_mean`（Kirchhoff 指数） | conditional_energy | 0.9984 |
| `res_end2end` | conditional_energy | 0.9962 |
| `fiedler`（代数连通度） | conditional_energy | 0.9945 |
| `th_mean` / `th_min` / `th_max` / `th_first` / `th_last` | conditional_energy | 0.9845–0.9886 |
| **`norm_fiedler`** | partial_edge_std | 0.9740 |
| `gap_ratio` | partial_edge_std | 0.8672 |
| `spectral_erank` | partial_edge_std | 0.8638 |
| `kirchhoff_efficiency` | partial_edge_std | 0.8637 |
| `res_cv` | partial_edge_std | 0.8407 |
| `res_shortcut` | partial_edge_std | 0.5621 |
| `th_cv` | action_consensus | 0.5465 |
| `th_slope` | partial_edge_std | 0.0895 |
| `n_near_zero` / `foster_residual` / `neg_count` / `neg_mass` | — | ≤ 0.020（近乎常数） |

**读法**：Kirchhoff 指数、矩阵树、Fiedler 值、Thévenin 电阻这四个"教科书主角"全部落在 ρ ≥ 0.984 一档——它们计算得出的是同一个数：残差路由能量的总标度。**通勤时间**（commute time = 2·vol·R）我在实现阶段就剔除了：它按定义是 `vol` 与 `R` 的乘积，不含任何新信息，属于 **(c) 不成立** 的量。

---

## 4. 无阈值的信息量：生存条件 AUC

为了绕开阈值/校准选择，定义：在每个 (suite, chunk q) 分层里，只取**在 q 时仍在运行**的 episode，算该量在 q 时刻对 risk 的 AUC；各层用 Mann–Whitney 权重 n₊·n₋ 汇总（要求 n₊ ≥ 20 且 n₋ ≥ 20）。0.5 表示"除了还在跑，什么都没多说"。

external_8b，每个量取跨 8 层的最佳：

| 排名 | 量 | 家族 | 层 | AUC | \|AUC−0.5\| | dev 上的 \|AUC−0.5\| |
|---|---|---|---|---|---|---|
| 1 | mobility | published | L14 | 0.3853 | **0.1147** | 0.0760 |
| **2** | **norm_fiedler** | **circuit** | **L12** | **0.5969** | **0.0969** | 0.0653 |
| 3 | partial_query_d1 | published | L12 | 0.4093 | 0.0907 | 0.0563 |
| 4 | action_consensus | published | L15 | 0.5873 | 0.0873 | 0.0563 |
| 5 | partial_edge_std | published | L12 | 0.4153 | 0.0847 | 0.0590 |
| 6 | dd_deficit | circuit | L12 | 0.5805 | 0.0805 | 0.0574 |
| 7 | flow_settling_log_ratio | published | L3 | 0.5803 | 0.0803 | 0.0750 |
| 8 | gap_ratio | circuit | L12 | 0.5780 | 0.0780 | 0.0586 |
| 9 | kirchhoff_efficiency | circuit | L12 | 0.5764 | 0.0764 | 0.0564 |
| 10 | spectral_erank | circuit | L12 | 0.5760 | 0.0760 | 0.0562 |
| … | （标度族：vol / fiedler / log_kirchhoff / log_spanning_trees / th_* / res_mean …） | circuit | L3 | ≈0.548 | **≈0.048** | ≈0.039 |
| … | conditional_energy（对照） | published | L3 | 0.4516 | **0.0484** | 0.0390 |

dev → external 的排序复现：**Spearman 0.9857、Pearson 0.9545**（43 个量）。信息排序是稳定的。

标度族全部聚在 0.048，与 `conditional_energy` 的 0.0484 完全一致——**Kirchhoff 指数和生成树数携带的信息与 `conditional_energy` 一模一样，一点不多**。

### 4.1 `norm_fiedler` 的增量是真的（事后检验，已标注）

配对 cluster bootstrap（按 suite 内 episode 有放回重采样，500 次，seed 20260906），统计量 = |汇总生存条件 AUC − 0.5| 之差，L12 层：

| 对比 | development_main | external_8b |
|---|---|---|
| norm_fiedler − **partial_edge_std** | **+0.0063 [+0.0044, +0.0083]** | **+0.0122 [+0.0102, +0.0143]** |
| norm_fiedler − mobility | −0.0055 [−0.0141, +0.0030] | **−0.0173 [−0.0261, −0.0095]** |
| gap_ratio − partial_edge_std | −0.0004 [−0.0064, +0.0057] | **−0.0066 [−0.0130, −0.0001]** |

**这是本工作最实的一条正面结果**：归一化拉普拉斯 L_norm = I − D^{-1/2} W D^{-1/2} 的 λ₂ 严格优于它最接近的已发表亲戚（ρ = 0.974），两个队列都成立；而**未**归一化的 `gap_ratio` 没有增量。机制解释：`partial_edge_std` 是逐条边做相关归一化，`norm_fiedler` 是在**整张图层面**除掉每个 token 的总电导，因此抽掉的是路由能量标度而保留了耦合的形状；实测证明整图层面的归一化抽得更干净。

**选择时点声明**：这个配对由**无标签**的冗余表（第 3 节，只用 development）确定，但"跑这个 bootstrap"的决定是在看过 external AUC 表之后做的，因此整条结果标注为**事后（post-hoc）**。

### 4.2 机制方向：风险 = 向均匀团坍缩

所有以"均匀团"为极值参考的**尺度无关**不变量，方向完全一致（external）：

| 量 | 均匀团处的取值 | 风险方向 | AUC |
|---|---|---|---|
| `norm_fiedler` | 最大 (10/9) | 更高 | 0.5969 |
| `gap_ratio` | 最大 (1) | 更高 | 0.5780 |
| `kirchhoff_efficiency` | 最大 (1) | 更高 | 0.5764 |
| `spectral_erank` | 最大 (1) | 更高 | 0.5760 |
| `res_cv` | 最小 (0) | 更低 | 0.4267 |
| `th_cv` | 最小 (0) | 更低 | 0.4311 |
| `thalt_cv` | 最小 (0) | 更低 | 0.4371 |

**7/7 一致**：当动作 token 的路由图变得**更像一张完美均匀的团**时，风险更高。即 chunk 内 10 个未来动作在专家空间上失去彼此的差异化，整段计划"路由得一模一样"。这与"后层比前层离团更远、且后层结构更分化"的已有事实在方向上自洽。
（`res_shortcut` 也偏向团参考值 9，但因 9 不是上界，这一条不纳入上述 7/7。）

---

## 5. 检测器：dev 选择，external 只跑一次

协议完全复用已发表代码（`moe-v4-0904/experiments/evaluate_layerwise_alarm_development.py` 的 trailing_mean / persistent_score / row_max / quantile_higher / first_query / crossfit_thresholds / representations / QUANTILES；`moe-hb-front-back-0905/experiments/select_early_lock.py` 的 survival_prior / prior_of / score_candidate；WIDTH=4、CONFIRMATIONS=4、MIN_LOW_PRIOR_PRECISION=0.60、MAX_TIMELY_FPR=0.005）。阈值**永远在 development 上拟合**：global = development_main + development_extra 的合并峰值分位；per_task = 同任务的 development 峰值分位。候选格 = 43 个量（31 电路 + 12 已发表对照）× 11 层视图 × 2 方向 × 14 分位 × 2 模式 = **26,488** 行 dev 候选。

**复现校验（重要）**：把已发表的量放进同一套代码，external 结果**逐位重合**——
`mobility | L12 | low | 0.975 | global` → **195 TP / 17 FP，precision 0.9198，lift 1.7641**；
`expert_load_effective_rank | L3 | low | 0.85 | per_task` → **370 TP / 93 FP，lift 1.5479**。
说明本 bundle 的数字与基线可直接比较。

external_8b 的 horizon cap（30/52/28/22）与先验跨越 0.25 的 chunk（goal 18 / long 26 / object 17 / spatial 13）由代码重算，与已有文档一致。

### 5.1 global 模式（唯一能证明信号在 MoE 侧的模式）

| 家族 | 量 | 视图 | 方向 | q | TP | FP | precision | 匹配先验 | **lift [95% CI]** | 早期 TP | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| published | **mobility**（基线） | L12 | low | .975 | **195** | 17 | 0.9198 | 0.5214 | **1.764 [1.679, 1.821]** | 36 | 0.346 |
| published | conditional_query_d1 | L5 | low | .95 | 194 | 31 | 0.8622 | 0.4994 | 1.726 [1.624, 1.805] | 38 | 0.344 |
| circuit | spectral_erank | back_median | high | .90 | 52 | 19 | 0.7324 | 0.4979 | 1.471 [1.244, 1.650] | 19 | 0.092 |
| circuit | kirchhoff_efficiency | back_median | high | .90 | 50 | 21 | 0.7042 | 0.4822 | 1.461 [1.223, 1.654] | 19 | 0.089 |
| circuit | res_cv | back_median | low | .90 | 42 | 18 | 0.7000 | 0.4603 | 1.521 [1.249, 1.740] | 18 | 0.074 |
| circuit | **norm_fiedler** | L15 | high | .95 | 42 | **2** | **0.9545** | 0.6495 | 1.470 [1.307, 1.520] | 15 | 0.074 |
| circuit | gap_ratio | back_median | high | .90 | 39 | 15 | 0.7222 | 0.4850 | 1.489 [1.219, 1.699] | 16 | 0.069 |
| circuit | res_shortcut | L12 | high | .96 | 39 | 6 | 0.8667 | 0.6716 | 1.290 [1.099, 1.396] | 9 | 0.069 |
| circuit | th_cv | back_median | low | .99 | 8 | 1 | 0.8889 | 0.6757 | **1.315 [0.836, 1.450]** ✗ | 2 | 0.014 |

`norm_fiedler` 的 **precision 0.9545 是全部 43 个 global 检测器中最高的**（mobility 0.9198），但只有 42 个 TP，lift 置信区间与 mobility 完全不重叠，**基线更好**。

### 5.2 per_task 模式

| 家族 | 量 | 视图 | 方向 | q | TP | FP | precision | lift [95% CI] | 早期 TP |
|---|---|---|---|---|---|---|---|---|---|
| published | **expert_load_effective_rank**（基线） | L3 | low | .85 | **370** | 93 | 0.7991 | **1.548 [1.473, 1.614]** | 97 |
| published | flow_settling_log_ratio | front_median | high | .85 | 304 | 87 | 0.7775 | 1.413 | 83 |
| circuit | **norm_fiedler** | L15 | high | .85 | 161 | 27 | 0.8564 | 1.325 [1.236, 1.392] | 41 |
| circuit | kirchhoff_efficiency | back_median | high | .90 | 114 | 8 | 0.9344 | 1.299 [1.218, 1.344] | 26 |
| circuit | res_cv | back_median | low | .85 | 126 | 13 | 0.9065 | 1.270 [1.186, 1.323] | 27 |
| circuit | log_kirchhoff | L12 | low | .95 | 69 | 15 | 0.8214 | 1.620 [1.432, 1.752] | 26 |
| circuit | **th_slope** | L3 | low | .98 | 10 | 14 | 0.4167 | **1.023 [0.601, 1.502]** ✗ | 0 |

43 个 external 检测器里 lift 置信下界 > 1 的：global **32/43**、per_task **38/43**。

### 5.3 预先声明的合取（在 external 打开之前写死）

`spectral_erank | back_median | high | 0.90 | global`（dev 上最好的 primary 电路检测器）**AND** `mobility | L12 | low | 0.975 | global`，两者都触发时在 max(first) 处报警：

* development：33 TP / **0** FP，precision 1.000，lift 1.744，早期 13 TP / 0 FP
* **external：36 TP / 1 FP，precision 0.9730，lift 1.680，timely_fpr 6.65e-5，早期 15 TP / 1 FP（早期 precision 0.9375）**
* 虚警依赖度（共现 / 独立期望）= **46.55**；`spectral_erank` 独有虚警 18 个，mobility 独有 16 个，共有 1 个。

**判定：合取没有超过 mobility 单独的 1.764。** 高达 46.6 的虚警依赖度说明这两个视角在"什么时候会误报"上高度同构，没有可挖的互补性。

---

## 6. 明确失败的候选量

按题面要求，逐一点名说明是 **(b) 机制上不可解释** 还是 **(c) 经验上无信息**：

| 候选 | 结论 | 数字 |
|---|---|---|
| **近零特征值个数 / 连通分量数**（`n_near_zero`） | **(c) 完全无信息** | 生存条件 AUC = **0.5000（精确）**；在 4,015,816 个单元里只有 16 个不等于 1；dev 上两种模式都无可行配置 |
| **Foster 残差**作为特征 | **(c) 无信息** | AUC 0.4951；它只在那 16 个单元上非零。作为**代码校验**它是有用的，作为**检测特征**不是 |
| **负电导计数 / 负质量**（`neg_count` / `neg_mass`） | **(c) 无信息** | AUC 0.5002；dev 上两种模式都不可行 |
| **通勤时间 / 随机游走** | **(c) 定义上冗余，实现阶段即剔除** | commute = 2·vol·R，是已有两量的乘积，无新自由度 |
| **Thévenin 电阻沿 chunk 的时间斜率**（`th_slope`） | **(c) 不复现** | dev AUC 0.5289 → external 0.4605（**符号翻转**）；global 模式不可行；per_task 上 10 TP / 14 FP，lift **1.023 [0.601, 1.502]**，含 1 → 空结果 |
| **Thévenin 离散度**（`th_cv`） | **(c) 弱且不显著** | global：8 TP / 1 FP，lift 1.315 **[0.836, 1.450]**，含 1 |
| **Kirchhoff 指数 / 矩阵树（生成树数）/ Fiedler 值 / Thévenin 电阻均值** | **(b) 可解释但 (c) 非新** | 与 `conditional_energy` 的 \|ρ\| = 0.984–1.000；AUC 与它一致到 0.048 vs 0.0484 |
| **Rayleigh 单调性** | **未做成可检验命题** | 它是关于"加边使电阻不增"的定理，在固定的每-query 图上不产生可比较的量；我没能把它变成一个不平凡的经验检验，因此不报告 |
| **未归一化谱隙比**（`gap_ratio`） | **(c) 无增量** | 相对 `partial_edge_std`：external −0.0066 [−0.0130, −0.0001] |
| **把 C 直接读成接地拉普拉斯（Kron 归约的字面版）** | **构造非法** | 仅 7.3e-6 / 9.3e-6 的行对角占优；行外和平均超出对角 7.5 倍 |

---

## 7. 局限（直说）

1. **多重比较。** 电路臂 31 个量、对照臂 12 个量，各自 11 视图 × 2 方向 × 14 分位 × 2 模式，共 26,488 个 dev 候选。选择规则与已发表 survey 逐字一致、且对每个量一视同仁，external 只跑一次；但"31 个量里最好的那个"本身带选择偏差。第 4 节的无阈值 AUC 与 4.1 的 bootstrap 不受阈值选择影响，是更可信的那部分证据。
2. **primary / secondary 划分**在打分前声明，依据是机制（尺度无关不变量）与**无标签**的冗余表，但它是我自己定的，不是外部预注册。
3. **先验在被打分的队列上估计**（`prior_estimated_in_sample = true`），与已有基线的做法相同，因此 lift 是可比的，但绝对值偏乐观。
4. **0.25 的"早期"切点没有做敏感性扫描**，沿用已有约定。
5. **lift 的置信区间**只对 precision 用 Wilson 区间，把匹配先验当作固定值处理；没有对先验的估计误差做传播。
6. **子采样**：只有第 3 节的秩相关用了 200,000 / 221,781 个 dev query（seed 20260906）和 4.1 的 500 次 bootstrap（seed 20260906）。**特征抽取与全部检测器评估都在完整队列上完成，无子采样。**
7. **只用了最后一步去噪**。前 9 步的图演化没有进入任何电路量——这是刻意的（与已有 `flow_*` 族划清界限），但也意味着"电路量随去噪演化"这条线索没被探索。
8. **`th_slope` 的符号翻转**说明存在至少一个电路量在两个队列间不稳定；我没有排查原因。

---

## 8. 下一步会做什么

1. **只保留 `norm_fiedler`，把它做深。** 它是唯一在两个队列上都严格超过最近已发表亲戚的量。值得试：(i) 把度归一化推广到 `res_cv` / `kirchhoff_efficiency`（即在 L_norm 而不是 L 上定义它们），看增量是否是"归一化"这一操作的普遍性质；(ii) `norm_fiedler` 的 Fiedler **向量**——它把 10 个动作 token 二分成哪两组？如果分界点系统性地落在 chunk 的某个时间位置，那是一个真正的机制发现。
2. **把电路量搬到去噪轨迹上。** 现在只用了 step 9。λ₂(L_norm) 沿 10 步的轨迹（"网络在去噪中如何连通起来"）是一个每-query 可算、机制上清晰的新对象，且与现有 `flow_*` 族不同（后者只看 √p 的位移，不看图结构）。
3. **放弃合取，改测"mobility 的残差"。** 46.6 的虚警依赖度说明 AND 无用；正确的问题是：把 mobility 回归掉之后，`norm_fiedler` 还剩多少生存条件 AUC？这是一个干净的增量检验。
4. **不再投入**：Kirchhoff 指数、矩阵树、Fiedler 值、Thévenin 电阻、通勤时间、Foster 残差、分量计数。它们要么等价于 `conditional_energy`，要么在 4e6 个单元上近乎常数。

---

## 附：文件

```
moe-circuit-analogy-0906/
├── REPORT_ZH.md
├── experiments/
│   ├── circuit_lib.py                 电路量的定义与实现（构造、界、Foster）
│   ├── extract_circuit_features.py    步骤 1：从 routes.zarr 抽 31 个量 × 8 层 × 全部 query
│   ├── diagnose_structure.py          步骤 2：团结构、Foster、冗余、前后层（无标签）
│   ├── protocol.py                    复用已发表协议的薄封装 + 预先声明的 primary 集合
│   ├── select_on_development.py       步骤 3：dev 上 26,488 个候选，冻结选择（不打开 external）
│   ├── evaluate_on_external.py        步骤 4：external 单次回放
│   ├── summarise_results.py           步骤 5：生存条件 AUC + Wilson / lift 置信区间
│   ├── bootstrap_increment.py         步骤 6：配对 cluster bootstrap（事后）
│   └── make_manifest.py
└── results/
    ├── manifest.json                  全部产物的 sha256 与重建命令
    ├── circuit_features/{development_main,development_extra,external_8b}.npz + extract_summary.json
    ├── diagnostics/{diagnostics.json,redundancy_vs_published.csv,
    │                circuit_internal_spearman.csv,front_back_contrast.csv}
    ├── detectors/{development_candidates.csv,development_selection.json,
    │              external_detectors.csv,external_detectors_with_ci.csv,
    │              external_evaluation.json,external_first_alarms.npz,
    │              survival_conditioned_auc.csv,bootstrap_increment.json,summary.json}
    └── logs/
```

**MoE-only 声明**：全部 31 个电路量只从 `hb_router_probs` 的最后一步去噪计算，未使用 episode 长度、结果、墙钟时间或任务身份。任务身份只出现在 `per_task` 阈值校准中，这正是它被单列为一种模式、并且以 `global` 模式作为"信号确在 MoE 侧"之证明的原因。
