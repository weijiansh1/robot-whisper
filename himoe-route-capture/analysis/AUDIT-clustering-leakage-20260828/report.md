# 无监督 MoE 路由聚类：泄漏与哨兵审计

日期 2026-08-28 ｜ 目录 `analysis/AUDIT-clustering-leakage-20260828/`
审计对象：`analysis/routing-organization-synthesis/report.md` 的「三方法共识失败核心」

---

## 判词

**共识核心里 routing 特异的成分是 0。** 它不是路由结构，是 `episode_length == 该任务 step cap`
这一个比特的有损重建。本语料里 outcome 与「跑满 cap」是同一件事，任何能恢复 length 的表示
自动是 0.99 精度的失败分类器。

三条硬证据，任一独立成立：

1. **哨兵超越**：`length ≥ step cap`（1 比特，零路由）precision **0.9935** / recall **1.0000**，
   全面优于共识核心 0.9816 / 0.6938。纯物理标量「末 11% 相位 EEF 平均步长」在同样 217 条预算下
   给出 **0.9816 / 0.6938 —— 与共识核心逐位相同**（同为 213/217）。
2. **条件独立**：给定 (task × length)，共识核心内失败数条件期望 213.49，实测 213
   （z = −0.87，p = 1.0）。raw C1 / event C0 / lag C7 **全部低于**条件期望。
3. **共同绝对索引下核心消失**：钉在固定整数 query 索引重跑同一管线，最佳块 precision 0.387–0.529，
   与 C1 的 Jaccard 仅 0.18–0.29。

必须同时记录：**拟合环节确实是 outcome-blind 的**（逐行核实通过）。问题在报告环节，
以及「outcome-blind 在被泄漏的 nuisance 本身就是标签时不提供任何保护」。

---

## 1 复现校验

`audit_pipeline.py` 是 aligned-kernel 下游管线的逐行重写，在 `feature_primary_geometry` 上重跑得到
**raw K6 与已发布标签逐元素相同**（ARI = 1.000）；event 端 Ward K4 重跑 ARI 也 = 1.000。

被审计的全部数字为真：raw C1 252/246（0.9762/0.8013）、event C0 214/209（0.9766/0.6808）、
lag C7 219/215（0.9817/0.7003）、共识 217 = 213+4（0.9816/0.6938）、交集 185、
并集 283 = 275+8（0.9717/0.8958）、两两 Jaccard 0.707/0.725/0.827、并集漏 32（30 moka-pot + 2 stove）。
truncate90 也为真：ARI 0.950、matched C1 249/219、0.851/0.841/0.734。

**复现不是问题。问题是这些数字的含义。**

---

## 2 语料结构性缺陷：outcome ≡ 跑满 cap

`meta.json:max_steps/10` 给出 cap：

| 任务 | cap | 失败 | 失败长度 | 成功 min/中位/max | cap 上的成功 |
|---|---:|---:|---|---|---:|
| goal/middle_drawer | 30 | 0 | — | 12/13/14 | 0 |
| goal/top_drawer | 30 | 42 | 全 = 30 | 17/19/21 | **0** |
| long/SCENE8 | 52 | 216 | 全 = 52 | 35/39/52 | **1** |
| spatial/ramekin | 22 | 12 | 全 = 22 | 9/10/22 | **1** |
| spatial/stove | 22 | 37 | 全 = 22 | 11/12/20 | **0** |
| **合计** | | **307** | **全部 = cap** | | **2 / 2253** |

（主 agent 独立从 `client/summaries.json:success` + `inference_calls` 复算，与上表一致。）

**五个任务全部成立** —— 这是 LIBERO 协议（成功即终止、失败即跑满）的直接推论，不是 SCENE8 的局部性质。
**42 个 (task×length) 分层里只有 2 个含两类样本，合计 228 F + 2 S。**

---

## 3 哨兵对照表

全部走同一管线、同一 episode 集合、同一 precision/recall 定义。

| # | 信息源 | 维度 | 方法 | n | fail/succ | precision | recall | F1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| — | **共识核心** | 320×10 路由 | ≥2 票 | 217 | 213/4 | **0.9816** | **0.6938** | 0.813 |
| — | aligned raw C1 | 同上 | Ward K6 | 252 | 246/6 | 0.9762 | 0.8013 | 0.880 |
| **S1** | **episode 长度** | **1** | `length ≥ cap` | 309 | 307/2 | **0.9935** | **1.0000** | **0.997** |
| S1b | 同上 | 1 | `length == 任务内 max` | 321 | 307/14 | 0.9564 | 1.0000 | 0.978 |
| **S2a** | **EEF 末 11% 平均步长** | **1** | 取最小 217 条 | 217 | 213/4 | **0.9816** | **0.6938** | 0.813 |
| S2a′ | 同，末 25% 相位 | 1 | 取最小 217 条 | 217 | 214/3 | 0.9862 | 0.6971 | 0.817 |
| **S2b** | **EEF 运动学（纯物理）** | 34→32 | 同一管线 K3/K6 | 108 | 108/0 | **1.0000** | 0.3518 | 0.521 |
| **S3** | **action chunk（无路由）** | 49→32 | 同一管线 K8 | 149 | 147/2 | **0.9866** | 0.4788 | 0.645 |
| **S4** | **路由 (task×相位) 内置换** | 320×10 | 同一管线 ×20 | — | — | 0.4216 中位 / 0.4586 最大 | — | 0.511 中位 |
| **S5** | **同维高斯噪声** | 320×10 | 同一管线 | 393 | 55/338 | 0.1399 | 0.1792 | 0.157 |
| S6 | (task,length) 代理张量 | 320×10 | 同一管线 K6 | 554 | 258/296 | 0.4657 | 0.8404 | 0.599 |

S2a 全局 AUC 0.9591（末 11%）/ 0.9529（末 25%）/ 0.9655（相位 0.5–1.0）。

**在 event 自己的管线上（57 维→PCA→Ward K4）：**

| 信息源 | n | fail/succ | precision | recall | vs event C0 Jaccard |
|---|---:|---|---:|---:|---:|
| 真实 57 维（复现 ARI = 1.000） | 214 | 209/5 | 0.9766 | 0.6808 | 1.000 |
| **仅由 length 合成的 57 维哨兵** | 232 | 202/30 | **0.8707** | 0.6580 | 0.372 |
| **单个二值 `late_stasis_indicator`** | 188 | 185/3 | **0.9840** | 0.6026 | **0.827** |

`late_stasis_indicator` 诊断：P(at cap│flag) = **0.9894**，P(failure│flag) = 0.9840。
它就是 step-cap 探测器；它与 event C0 的 Jaccard（0.827）**等于** event C0 与 lag C7 的 Jaccard（0.827）。

**判定标准触发**：S1 在两个指标上严格超过；S2a 与共识核心逐位相同。
S4/S5 确认这不是管线平凡产物（高斯 0.14、置换 0.42–0.46），管线确实提取了依赖 episode 内时间
一致性的真实结构 —— 只是那个结构是长度/停滞。

### 3.1 一个比特能复现「独立收敛」

| 配对 | Jaccard |
|---|---:|
| at_cap ~ raw C1 | **0.792** |
| at_cap ~ lag C7 | **0.698** |
| at_cap ~ event C0 | **0.676** |
| at_cap ~ 共识核心 | 0.691 |
| at_cap ~ 并集 | 0.879 |
| （已发布）三块彼此 | 0.707 / 0.725 / 0.827 |

纯 (task, at_cap) 代理划分（零路由数据）与已发布 raw K6 的 **ARI = 0.9475**（仅 task 标签 0.8778）。
被聚类的 17 维嵌入被 task one-hot 解释 R² = 0.703，加 L、L² 后 0.754（前 4 个 PC 各 0.84–0.89）。

---

## 4 分层内检验：功效为零，实测也为零

**条件置换（给定 task × length，20000 次）：**

| 集合 | 实测 | 条件期望 | sd | z | p(≥) |
|---|---:|---:|---:|---:|---:|
| 共识核心 | 213 | **213.49** | 0.559 | **−0.871** | 1.000 |
| raw C1 | 246 | 246.34 | 0.517 | −0.666 | 1.000 |
| event C0 | 209 | 209.49 | 0.559 | −0.871 | 1.000 |
| lag C7 | 215 | 215.41 | 0.491 | −0.826 | 1.000 |

**四个块全部低于条件期望。** 换分层：

| 分层 | 分层数 | 混合分层 | 混合内 episode | 共识核心 z | p |
|---|---:|---:|---:|---:|---:|
| task × length | 42 | **2** | 230 (228F+2S) | −0.871 | 1.000 |
| task × init | 80 | 40 | 1280 | **+20.757** | 0.0002 |
| task × init × length | 250 | **2** | **6** | −1.732 | 1.000 |

只控 init 时 z = +20.8 极显著；把 length 加进分层后 z = −1.73。
**这正是要防的第五次「漂亮的相关」。**

**分层内重聚类（同一管线从头重跑）：**

| 分层 | n | F | S | 最小可达单侧 p | 最佳块 | Fisher p |
|---|---:|---:|---:|---:|---|---:|
| **SCENE8 & len==52** | 217 | 216 | **1** | 0.00461 | K3 98/98 (prec 1.000) | **0.548** |
| | | | | | K2 119/118 | **1.000** |
| **全任务 at cap** | 309 | 307 | **2** | 0.000021 | K2 222/221 | **0.484** |
| | | | | | K4/K6 100/100 | **0.457** |
| SCENE8 & len≥47 | 227 | 216 | 11 | ~0 | K3 105/105 | 0.00087 |
| SCENE8 & len≥45 | 229 | 216 | 13 | ~0 | K3 103/102 | 0.00403 |

长度同质的两层内 p = 0.457–1.000，**完全没有判别力**（precision 1.000 是因为基准率本就 0.9954）。
宽带（≥47/≥45）显著是因为带内 `len==52` 仍近乎完美分类，把长度放回来了 —— 不能作为分层内证据。
**同 init state 检验在本语料无法与长度解耦。**

---

## 5 特征溯源

### 5.1 三方法共享的时间归一化缺陷

`PRIMARY_PHASE = PhaseConfig("primary", 0.5, 1.0, 10)`（`analyze_failure_routing_clusters.py:48-50,83`）：

1. **整个主特征只覆盖每条 episode 的后 50%。**「末段」不是特征的一部分 —— **特征全体都在末段**。
2. **锚点 10 精确落在最后一个 control step**（`alpha[9]=1.0`，无插值）：失败 = timeout 态，成功 = 完成态。
3. **重采样算子本身是 T 的闭式函数**：锚点间距 = `0.0556·(T−1)` 个真实 query
   （T=13 → 0.67 步，T=52 → 2.83 步）。`"episode_length_used_as_feature": False`
   对特征取值为真，**对采样算子为假**。
4. **`MAX_SHIFT=2` 的对齐容差同样正比于 T**（T=13 → 1.3 步，T=52 → 5.8 步）。

各锚点 320 维原始块被 `task one-hot × [1,T,T²]` 解释的 R²：
0.425 / 0.504 / 0.489 / 0.564 / 0.663 / 0.644 / 0.474 / 0.583 / 0.682 / 0.518；整体 3200 维 **0.569**。

### 5.2 aligned kernel geometry 320 维

| 组 | 维数 | 长度单独可算？ | 只由末段定？ | 活跃度单调？ |
|---|---:|---|---|---|
| S1–S8 序列量（entropy/top1/top4/dispersion/gap），各取 (mean, std, slope) | **192** | 值不是，**采样算子是** | 否（但全在后 50%） | **是**：冻结时 std→0、slope→0 |
| T1–T4 转移量（denoise Hellinger speed、top-4 保持率），各取 (mean, std, **last**, **max**) | **128** | 同上 | `last` 是 denoise 末值 | **是，按定义**。`speed` 就是活跃度；`max`/`last` 正是 `SESSION_NOTES §C.3` 判定为构造缺陷的极值统计 |

**64/320 = 20% 是纯极值/末值统计**，另 64 是 std、64 是 slope。§C.3 结论原样成立。
唯一缓解是 `normalize_trajectory_shape`（沿锚点去均值 + L2），去掉的是跨相位均匀水平，
去不掉「后半段活跃度形状」；且 `build_anchor_signatures` 的 PCA 在归一化**之前**、
在池化的 25600 行上拟合。

`canonicalize` 按 `diagnostic_mean_soft_speed` 升序排块 → **「C1」按构造就是活跃度第二低的块**
（C0 0.0085 / **C1 0.0128** / C2 0.0134 / C3 0.0139 / C4 0.0145 / C5 0.0178），正是冻结路由所在区间。

**锚点窗口消融**：只用锚点 1–7 或只用 8–10，与 C1 的最大 Jaccard 只有 0.210 / 0.206。
**⚠ 不可强解释** —— 子窗口改变了归一化本身。只能说「核心需要完整 10 锚点形状」。

### 5.3 route change events 57 维

| 类别 | 例 | 长度单独可算？ | 只由末段定？ | 活跃度单调？ |
|---|---|---|---|---|
| "fraction" 类 | `peak_count_fraction`(/9)、`longest_stasis_run_fraction`(/9)、`return_count_fraction`(/8)、`change_count_fraction`(/7) | **否 —— 分母是固定锚点数，不是 T** | 否 | 曲线全平时 →0 |
| 明确末段量（13 个） | `soft_terminal_speed_ratio`、`terminal_return_advantage/indicator`、`terminal_low_speed_indicator`、`soft_terminal_drift_normalized`、`layer_low_terminal_fraction`（末 5.6%）；`late_low_speed_fraction`（末 16.7%）；`soft/hard_late_early_ratio`、**`late_stasis_indicator`**（末 22.2% 比 0.5–0.722） | 否 | **是** | 比值型 |
| argmax 退化陷阱（5 个） | `global_peak_phase`、`layer_peak_exact_fraction`、**`layer_peak_within1_fraction`**、`soft_hard_peak_phase_gap`、`change_layer_within1_fraction` | 否 | 否 | **反向陷阱**：曲线全平时 argmax 并列取 0，同步性读数 **→1.0** |
| 尺度不变族 | 一律除 `soft_scale = median(speed)` | 否 | 否 | 已除掉绝对活跃度 |

scene8 内（task 固定）ρ(·,T) 与 ρ(·,failure) 在噪声范围内是同一个数：
hard_speed_cv +0.715/+0.691、hard_change_max_contrast +0.648/+0.648、denoise_std_cv +0.559/+0.621、
soft_speed_cv +0.557/+0.620、change_max_contrast +0.567/+0.604、late_stasis_indicator +0.528/+0.591、
layer_speed_curve_correlation +0.473/+0.523、layer_peak_within1_fraction +0.434/+0.514。

消融（Ward K4）：

| 变体 | vs 已发布 ARI | 最佳块 precision | vs C0 Jaccard |
|---|---:|---:|---:|
| 真实 57 维 | 1.000 | 0.9766 | 1.000 |
| 去 `late_stasis_indicator` 一列 | 0.919 | 0.9802 | 0.900 |
| 去全部 13 个末段特征 | 0.649 | 0.4667 | 0.361 |
| **对 (task×[1,T,T²]) 残差化** | 0.619 | **0.2560** | **0.151** |

**对长度残差化后 event C0 溶解**（0.977 → 0.256，AMI(outcome) 0.062）。

### 5.4 grammar / lag / peer_rank

- **grammar 187 维**：`endpoints` 块 16 维里 **8 维是最后一个锚点状态的 one-hot**，
  直接编码 timeout/completion 终态。脚本自报 length **R² = 0.960 / MAE 1.46 query**、
  task NMI 0.887、自标 `task_shadowed`。
- **lag_spectrum 648 维**：变分图横轴 lag k = `k·0.0556·(T−1)` 真实步，被 T 拉伸。
  Louvain 的 `length_nmi = 0.768` 是全部 10 个 (表示×方法) 拟合中最高，
  **且高于自身 task_nmi 0.689**。残差化后 vs C7 Jaccard 0.511 → 0.301。
- **peer_rank 54 维**：五种替代表示里**唯一**真正商掉 nuisance 的（32 条 sibling 内秩变换），
  于是 task_nmi 0.074 / length_nmi 0.121 / outcome_excess 0.011。
  **它是这批分析里的正确对照组** —— 真正商掉 task/length 后 outcome 信号也没了。

---

## 6 truncate90 修了什么 + 共同绝对索引截断

**truncate90 语义**（核实）：`PhaseConfig("truncate90", 0.5, 0.9, 10)` 不是截短 tape，
而是把 10 锚点重铺到 [0.5,0.9]，间距从 `0.0556·(T−1)` 变 `0.0444·(T−1)`。
它**不是** primary 锚点的子集（10 个只有 2 个重合），是「更短窗口上的更密采样」，
不是干净的「同特征去尾」消融。丢掉的绝对 query 数依 T 而变（T=13 丢 1、T=22/30 丢 2、T=52 丢 5）。

**关键**：`episode_length` NMI 从 full 的 **0.655** 只降到 **0.648**。
删末 10% 相位没有削弱长度耦合，因为它根本没改变 T，也没改变重采样算子对 T 的依赖。
truncate90 只排除了「核心由最后一个观测产生」这个最窄假设。

**更强版本（测试 D）**：从 `routes.zarr` 重算 `router_features`，把 10 锚点钉在每任务
「全体 rollout 都还在跑」的固定整数 query 索引（`t ≤ Tmin−1`），无插值、无 T 依赖、无幸存者偏差。
Tmin = 12/17/35/9/11。

| 窗口 | 最佳 F1 块 | precision | recall | vs C1 Jaccard | vs 共识 Jaccard | AMI(outcome) |
|---|---|---:|---:|---:|---:|---:|
| full common prefix | 313 / 121 F | **0.387** | 0.394 | **0.182** | 0.167 | 0.095 |
| common prefix 后半 | 359 / 190 F | **0.529** | 0.619 | **0.294** | 0.274 | 0.105 |
| （对照）已发布全窗口 | 252 / 246 | 0.976 | 0.801 | 1.000 | — | 0.273 |

**共识核心不在了。**

### 6.1 残留（新发现，需谨慎）

之前没被处理的混杂：**仅凭 init_state 就能在任务内区分 outcome**
（in-sample AUC：top_drawer 0.951、scene8 0.899、stove 0.849、ramekin 0.825）。
`audit_07` 做 20000 次 (task,init) 条件置换：

| 任务 | 窗口 | 最佳块 | 边际 p | 给定 init 的 z | 给定 init 的 p |
|---|---|---|---:|---:|---:|
| spatial/stove | full prefix (q≤10) | 36/34 F（基准 0.072） | 3.4e-47 | **+12.57** | **≤5e-5** |
| long/SCENE8 | full prefix (q≤34) | 117/91 F（基准 0.422） | 4.8e-19 | **+4.34** | **1.5e-4** |
| goal/top_drawer | full prefix (q≤16) | 186/27 | 1.2e-4 | −1.03 | 0.94 |
| spatial/ramekin | full prefix (q≤8) | 63/5 | 9.6e-3 | +0.38 | 0.88 |
| **四个任务全部** | **early half（前 ~1/3）** | — | 6e-8~0.06 | **−1.04 ~ +0.50** | **0.66 ~ 1.00** |

三条限定缺一不可：

(a) **严格早期窗口内四个任务全部为零** —— 先前看到的 4.4× 富集（stove q0–2）在控制 init 后
完全消失，那是初态编码不是路由前兆；
(b) 残留显著的两个任务用的是 `t ≤ Tmin−1`，而 Tmin 按定义是最短 episode 的长度，
对最短成功而言窗口几乎覆盖整条 episode（stove Tmin=11、成功中位 12），
所以仍是**并发检测**不是预警，与 `SESSION_NOTES §A.5` 一致；
(c) **绝对时间匹配与任务相位匹配不可兼得** —— 39 步成功与 52 步 timeout，
固定绝对索引 t=34 对前者是 87% 相位、对后者是 65%。
这不是可修的分析选择，是语料设计的性质。

---

## 7 outcome-blind 是真的吗

### 7.1 拟合环节：是真的（逐行核实）

特征只加载 `feature_primary_geometry`；`residualize_sequences("raw")` 直接返回拷贝；
`cluster_landmark_view(matrix, score, subsamples, seed)` 签名里没有 outcome；
稳定性门限与选择键全无监督；HDBSCAN/Louvain 参数是先验常量。

**`canonicalize(labels, score)` 的 score：**

| 脚本 | 调用处 | score | outcome 派生？ |
|---|---|---|---|
| `analyze_aligned_route_kernel.py` | :351（源 :844） | `diagnostic_mean_soft_speed` | **否**，纯路由活跃度 |
| `..._truncate90.py` | :149（:620） | 同上 | 否 |
| `analyze_route_change_events.py` | :557（:1017） | `soft_speed_mean_abs`（:368） | **否**；ρ(score,failure) = −0.016 |
| `analyze_alternative_routing_organizations.py` | :662/664/699/715（:967） | `diagnostic_mean_soft_speed` | 否 |
| `analyze_routing_state_grammar.py` | :570 | `switch_rate` | 否 |

**canonicalize 没有泄漏 outcome。** 但它按活跃度升序重排块索引，所以 "C0/C1" 编号
**本身就是活跃度秩** —— 这解释了核心为何总落在低编号块。

### 7.2 报告 / 选择环节：不是

1. `OUTCOME_EXCESS_MIN=0.05` + `fdr_bh_q<0.05` 是显式 outcome 门；
2. `all_eligible_deltas_positive` 按失败率挑块；
3. **报告的是 sensitivity 变体而非声明的 primary** —— `PRIMARY_VARIANT="task_residual"`
   实际是 `no_stable_partition`、`cross_task_failure_blocks=[]`、outcome excess **0.004**，
   传播出去的是 `raw`；
4. `..._truncate90.py:55-56,130-138` 把目标块**硬编码**为 `REFERENCE_RAW_BLOCK=1`，
   并对 `episodes!=252 or failures!=246` 直接 `raise` —— **对失败计数的硬断言**；
5. `raw C1 / event C0 / lag C7` 是手选常量，规则透明地是「失败率最高的块」，
   候选空间 **100+**（5 表示 × 2 方法 × full/truncate90 + 其它脚本），无多重性校正；
6. event 报告只展示 57 个特征中硬编码的 7–8 个，其中唯一真正区分 C0 的是
   `late_stasis_indicator`（0.850 vs 0.004/0.000/0.000）。

**`"outcome_used_before_label_freeze": False` 字面为真但对整条流程有误导性。
当被泄漏的 nuisance 本身等价于标签时，outcome-blind 拟合不提供任何保护。**

### 7.3 被审计脚本自己的反证

`route-change-events/summary.json` 原文（**已发布产物，非本次重跑**）：

| 模型 | ROC AUC |
|---|---:|
| task | 0.7654 |
| **task + length** | **0.9995**（AP 0.9884） |
| task + events | 0.9910 |
| task + length + events | 0.9975 |

| 增量 | ΔAUC | 95% CI | ΔAP |
|---|---:|---|---:|
| events over task | +0.2255 | [+0.1284, +0.3564] | +0.6788 |
| **events over task_length** | **−0.0020** | **[−0.0055, +0.00004]** | −0.0020 |

grammar 端同理（task+length 0.9977 > task+length+grammar 0.9949）。

**这个数字在综合报告表格里出现过，但与开头「三方法一致 = 强证据」的叙事直接冲突，冲突未被解决。**

---

## 8 成分分解与撤回清单

| 成分 | 占比 | 依据 |
|---|---|---|
| **task 身份** | 主体 | ARI(raw K6, task) = 0.878；嵌入被 task one-hot 解释 R² = 0.703；K6 六块里四块几乎就是「某任务的成功集合」（512/470/471/500） |
| **length / timeout 分层** | 其余全部 | ARI(raw K6, task⊗at_cap) = 0.9475（零路由）；at_cap~C1 Jaccard 0.792 |
| **routing 特异（长度之外）** | **0** | 给定 (task,length) 后 z = −0.87/−0.67/−0.87/−0.83，四块全 ≤ 条件期望 |

更准确地说：本语料在 episode 级别**没有能力**给出非零答案（§4 功效上限）；
在它有能力回答的方向上（§6 共同绝对索引）答案是**否**。

### 应撤回或重写

| 原断言 | 判定 |
|---|---|
| 「三种完全不同、outcome-blind 的组织独立抽出同一组 rollout」（作为独立收敛证据） | **撤回作为证据**。一个比特与三块 Jaccard 0.68–0.79，与三块彼此 0.71–0.83 同量级；三者是同一 nuisance 坐标的三个平滑函数 |
| 「这不是一次算法偶然性」（子采样 ARI 0.991/0.977） | 稳定性为真但不相关 —— 稳定地恢复 nuisance 仍是稳定的 |
| 「不是最后一帧伪影」（truncate90） | **过度解读**。length NMI 0.655→0.648 几乎不变；真正的检验里核心消失 |
| 「route speed 持续下降 / 晚期低速平台 / backward return 更频繁 / 层间更同步」 | 前两条是 `canonicalize` 的构造后果（C1 按定义是活跃度第二低）；「层间更同步」是 `layer_peak_within1_fraction` 的 argmax 退化陷阱；「backward return 更频繁」与已发布 profile 表矛盾（`joint_return_count_fraction` C2 0.116 / C3 0.104 > C0 0.078） |
| 「跨任务的晚期未完成核心」 | 应重述为「**跨任务的 timeout 分层**」—— 跨任务成立只因每任务的 timeout 都在其 cap 上 |
| 单项报告的限定条款 | `aligned-route-kernel-truncate90/report.md` 写了 "remains subject to task, **episode-length**, checkpoint, and **timeout proxy** explanations"；`route-change-events/report.md` 写了 "Successful phase 1.0 is completion, while failed phase 1.0 is timeout" 与 "Task and length ... **can remain recoverable**"。**这些在综合报告中被丢弃了，应恢复** |

### 保留为有效

- 拟合环节的 outcome-blind 纪律（实现正确）
- `peer_rank` 的结论（唯一真正商掉 nuisance 的表示，且诚实地报告信号随之消失）
- HDBSCAN 视角（307/307 失败全 noise，不依赖块选择）
- event 脚本自己的嵌套模型检验（本批分析里最强的结果，只是被埋在表格里）

---

## 9 下一步建议

1. **换语料不换方法** —— 需要包含在 cap 之前就失败终止、或跑满 cap 仍成功的 rollout。
   当前 2253 条成功里只有 2 条跑满 cap。
2. 若必须用现有语料，唯一可做的是 §6 那一类（固定绝对窗口 + (task,init) 条件置换）。
3. 报块时给**全部候选块的表**加多重性校正，`audit_pipeline.block_table()` 可直接产出。
4. **每条管线配哨兵** —— `SESSION_NOTES §D` 已把「给分析管线配阳性对照」写成对策，
   这轮若照做，S1 第一天就能拦下这个核心。

---

## 10 产物与未做部分

`analysis/AUDIT-clustering-leakage-20260828/`（40 MB）：
`audit_pipeline.py`（bit-exact 管线重写）、
`audit_00_reproduce_core.py` → `reproduce_core.json` / `core_membership.npz`、
`audit_01_extract_sentinels.py` → `sentinel_tapes.npz`、
`audit_02_sentinels.py` → `sentinels.json`、
`audit_03_stratified.py` → `stratified.json`、
`audit_04_common_prefix_extract.py` → `common_prefix_tapes.npz`、
`audit_05_common_prefix_cluster.py` → `common_prefix.json`、
`audit_06_event_and_lag.py` → `event_and_lag.json`、
`audit_07_prefix_init_control.py` → `prefix_init_control.json`，
另有 `anchor_ablation.json`、`early_prefix.json`、`eef_1d.json`、`labels_*.npy`。

重跑：按 00→07 顺序执行，`OMP_NUM_THREADS=8`，约 20 分钟。

**明确声明未做的部分**：

- (a) 未重跑 `analyze_alternative_routing_organizations.py` 的原生 Louvain（用 Ward K6 代理），
  因此「lag 残差化后仍留 86 条纯失败小块（recall 0.280、vs C7 Jaccard 0.301）」是**代理管线**结果，
  不足以支撑核心级结论，值得后续用原生 Louvain 复核；
- (b) 未做 grammar 端残差化消融（grammar 已被其自身脚本标记 `task_shadowed`、length R² 0.960，
  方向明确，不改变判词）；
- (c) 无 MuJoCo/LIBERO，无法生成「cap 之前失败终止」的对照 rollout。
