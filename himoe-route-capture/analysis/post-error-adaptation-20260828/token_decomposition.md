# R^state vs R^action：后缀 token 轴上的 HB 路由分解（描述性）

本文件**只做描述，不做因果宣称**。二阶失败实验的可行性判定见 `feasibility.md`（Gate 0 FAIL）。
生成命令：`python3 analyze_post_error_adaptation.py --stage tokens`。

`server/routes.zarr` 的 token 轴 `11` = 后缀 token：token 0 = state，token 1–10 = action。
因此 `R^state = [:, :, :, 0, :]`，`R^action = [:, :, :, 1:11, :]`。
任务固定为 `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`（22883 个 query）。

---

## 1 结构结论（最强的一条）

### 1.1 state token 在 denoise 轴上完全退化

| 量 | 值 |
|---|---:|
| `max_k |p^state(k) − p^state(0)|`（全 22883 query × 8 层 × 32 expert） | **0.0（逐位相同）** |
| state top-4 在 10 个 denoise 迭代间完全一致的比例 | **1.000** |

**state token 的路由不随 flow denoise 迭代变化，一位都不变。**
这在结构上是必然的：state token 的输入表示不依赖噪声水平，其 router 输入被缓存。
因此**「denoise 轴」对 state token 不存在**；只有 action token 有 denoise 轴。

这直接支持用户划分的方向：同一次推理里，只有 action token 的路由在 flow 迭代中
被重新计算。但见 §1.3——重新计算的幅度极小。

### 1.2 两种 token 路由的浓度差一个量级

| 量 | state token | action token | 均匀基线 |
|---|---:|---:|---:|
| top-1 概率 | **0.1425** | **0.0371** | 0.03125 |
| 归一化熵 `H / ln32` | **0.9232** | **0.9987** | 1.0 |
| top-4 边界间隙 `p₄ − p₅` | **3.13e-3** | **3.88e-4** | 0 |
| 相邻 query 的 top-4 更替率 | 0.311 | **0.695** | 0.875 |
| 相邻 query 的 `1 − Hellinger` | **1.05e-2** | **2.7e-4** | — |

**action token 的 HB router 基本是均匀分布**（top-1 只有均匀值的 1.19 倍，
归一化熵 0.9987）。它的 top-4 身份在相邻 query 之间更替 **69.5%**，
但概率分布本身只移动 `1−H = 2.7e-4`——比 state token 小 **39 倍**。

分层看（top-1 概率）：

| HB 层 | 2 | 3 | 4 | 5 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| state | 0.175 | 0.142 | **0.311** | **0.284** | 0.062 | 0.057 | 0.057 | 0.052 |
| action | 0.037 | 0.037 | 0.037 | 0.038 | 0.036 | 0.037 | 0.036 | 0.038 |

state token 在浅层（4、5）有明显的专家选择；action token 在**全部 8 层都是均匀的**。

### 1.3 denoise 轴上的 action 变化几乎全部是近似平局翻转

| denoise 迭代 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top-4 与迭代 0 的重叠 | 0.816 | 0.740 | 0.677 | 0.611 | 0.547 | 0.485 | 0.427 | 0.366 | **0.302** |
| prob L1 与迭代 0 的距离 | 0.013 | 0.020 | 0.026 | 0.033 | 0.041 | 0.049 | 0.057 | 0.067 | **0.082** |
| 平均熵 | 3.4625 | 3.4623 | 3.4620 | 3.4618 | 3.4615 | 3.4612 | 3.4608 | 3.4600 | 3.4585 |

到最后一个迭代，action token 的 top-4 集合**换掉了 70%**，
但概率分布只移动 L1 = 0.082（上界 2.0），熵只掉 0.004（上界 3.4657）。
相邻迭代的 `1 − Hellinger` 仅 **4.3e-5**。

与独立审计 `analysis/near-tie/summary.json` 一致：
`boundary_tie_rate = 0.391`、`gap_exactly_zero_rate = 0.238`、`gap_median = 4.88e-4`（bf16）。

> **必须写进任何后续解读**：action token 的 top-4 变化**主要是 bf16 近似平局的抖动**，
> 不是「计划改变」。任何建立在 action-token top-4 重叠上的量
> （包括 `analysis/replanning-reset-trap` 的 action-token route return AUC 0.733，
> 以及本设计的 `S_A^route`）都必须先排除平局伪影，
> 建议改用软 Hellinger 或 selected-prob 加权量。

### 1.4 state 与 action 路由到近乎不相交的专家

同一 query 内 state token 与 action token 的 top-4 重叠 = **0.0923**，
低于独立随机的期望 **0.125**（4×4/32/4）。
模型系统性地把「看到了什么」和「准备怎么做」分配给**不同**的专家。
这是一条实质结构事实，与 posterior / prior 的划分方向一致，
但它只说明二者在路由空间上被分开，**不说明谁在时间上领先**。

---

## 2 事件对齐动力学（39 条 EEF return-to-pot2-side / 123 条 stasis proxy）

事件来自独立物理账本 `analysis/failure-moe-signatures/episode_results.csv`，
不是本设计新造的 landmark。曲线先减本 episode `[−10,−7]` 基线，再除 pooled SD。
`lead = [−6,−2]`、`sync = [−1,+1]`、`after = [+2,+6]`。
需要完整 `[−10,+6]` 窗口，可用：`active_return` **29/39**、
`stagnation_core` **122/123**、`success` 296/296。

### 2.1 与 success 参照（matched-init）比较

| signal | class | lead | sync | after | 最早分开 |
|---|---|---:|---:|---:|---|
| `state_route_speed` | active_return | +1.98 | +1.00 | +0.97 | lead |
| `action_route_speed` | active_return | +1.49 | (ns) | (ns) | lead |
| `state_route_soft_speed` | active_return | +2.27 | +1.04 | +0.82 | lead |
| `action_route_soft_speed` | active_return | +2.37 | +1.71 | +1.36 | lead |
| `state_entropy` | active_return | −1.07 | −4.06 | −2.02 | lead |
| `action_entropy` | active_return | −2.15 | −1.82 | −2.45 | lead |
| `state_top1_mass` | active_return | +0.96 | +3.90 | +1.52 | lead |
| `action_top1_mass` | active_return | +2.53 | +2.36 | +1.46 | lead |
| `state_route_speed` | stagnation_core | +0.27 | (ns) | −1.29 | lead |
| `action_route_speed` | stagnation_core | (ns) | −0.47 | −1.62 | sync |
| `state_entropy` | stagnation_core | +0.20 | (ns) | +1.07 | lead |
| `action_entropy` | stagnation_core | (ns) | (ns) | (ns) | — |

分开的信号计数：`active_return` state 17/18、action 17/21；
`stagnation_core` state 12/18、action 6/21。

**state token 分开的次数不少于 action token，`stagnation_core` 上明显更多。
这不支持「action routing 是前验、先动；state routing 是后验、后动」的方向。**

⚠ 这一比较是**相位混杂**的：`active_return` onset 中位 42，
`stagnation_core` 中位 28，success 参照按首次进入 pot1 邻域（中位 28）对齐。
三条线的 lead 窗落在不同任务阶段上，只能描述各自 onset 附近的形态。

### 2.2 相位无关的比较：同一批 query 上，state 与 action 谁先偏离自身基线

state 和 action 是在**完全相同的 episode、完全相同的 query** 上测出来的，
比较它们各自偏离自身 `[−10,−7]` 基线的最早 query，**没有相位混杂**。
判据：`|effect| ≥ 0.2 SD` 且 init-bootstrap CI 不跨 0。

| class | signal | state 首次偏离 | action 首次偏离 | 谁更早 |
|---|---|---:|---:|---|
| active_return | entropy | −6 | **−7** | action 早 1 |
| active_return | top1_mass | −6 | **−7** | action 早 1 |
| active_return | route_speed | **−5** | −4 | state 早 1 |
| active_return | route_soft_speed | −5 | −5 | 并列 |
| active_return | route_recurrence | −5 | −5 | 并列 |
| stagnation_core | 全部 5 项 | −9 / −8 | −9 / −7 | 基线窗本身不稳，不可解读 |
| success | 全部 5 项 | −9 / −8 | −9 | 同上 |

**没有一致的领先关系**：action 早 1 query 的 2 项、state 早 1 query 的 1 项、并列 2 项。
`stagnation_core` 与 `success` 的曲线在 `[−9,−8]` 就已经偏离自身基线，
说明 `[−10,−7]` 参考窗对它们不稳定，这两类的「首次偏离」不可解读。

**结论：「新 v 是后验、由它产生的 action routing 是前验」这一说法，
在这份数据的时间分辨率（1 个 query = 10 个动作子步）上无法被证实也无法被证伪。**

---

## 3 denoise 轴判别力：整张表被哨兵作废

### 3.1 口径更正（重要）

`analysis/moe-rollout-trend/report.md` 第 18 行原文：
*"Queries k0..k8 are evaluated separately; no predictor reads another chunk."*

**那里的 `k` 是 query / control-step 索引，不是 flow denoise 迭代。**
`0.463 → 0.713` 描述的是「rollout 越往后，成功与失败的单 query 内部计算越可分」，
与本节的 denoise 轴是两条不同的轴，不能互相引用。

### 3.2 实测（EEF-return vs stasis，在各自 onset query 上）

| token | denoise | AUC(熵) | AUC(top-1 质量) |
|---|---:|---:|---:|
| **哨兵：onset query 索引** | — | **0.975** | **0.975** |
| state（denoise 不变） | — | 0.000 | 1.000 |
| action | 0 | 0.056 | 0.947 |
| action | 1–8 | 0.026–0.036 | 0.955–0.983 |
| action | 9 | 0.033 | 0.984 |

**1 维哨兵（onset query 索引）单独拿到 AUC 0.975**，所有真实信息源与它并列或仅高 0.025。

**判定：这张表整体作废。** `active_return` 与 `stagnation_core` 的 onset 分布
（42.4 ± 4.1 vs 28.4 ± 3.8）几乎不重叠，它测的是**任务相位**，不是路由。
与 `rerun-2026-08-27/SESSION_NOTES.md` §C.2 的哨兵事故同型。

可保留的只有一句**形状**描述：action token 在 denoise 轴上没有单调判别力趋势
（0.947 → 0.984，波动 < 0.04），**没有观察到「denoise 越往后越可分」的现象**。
这与 §1.3 一致：action 路由分布在 denoise 轴上几乎不动。

---

## 4 图

`token_decomposition.png`（2×3 面板）：
1–4 面板为事件对齐的 hard top-4 route speed / soft Hellinger route speed /
router entropy / route recurrence，虚线 = state token，实线 = action token，
红 = EEF return-to-pot2-side，蓝 = stasis proxy，灰 = success 参照。
第 2 面板中 action 三条线几乎压在 0 上，即 §1.3 的「action 概率分布几乎不动」。
第 5 面板为 denoise 轴 top-4 重叠（action 衰减到 0.30，state 恒为 1.0）。
第 6 面板为 denoise 轴判别力，含 onset 哨兵水平线（黑色点划线），
所有真实曲线都被哨兵压住。

## 5 产物

| 文件 | 内容 |
|---|---|
| `token_axis_summary.json` | 结构量、near-tie 诊断、逐层浓度、事件对齐平均曲线 |
| `token_event_effects.csv` | matched-init 的 lead / sync / after 效应量与 CI |
| `token_paired_onsets.csv` | 相位无关的逐 query 偏离检验（state / action / 两者之差） |
| `token_first_deviation.csv` | 各信号首次偏离自身基线的相对 query |
| `denoise_axis_auc.csv` | denoise 轴判别力 + onset 哨兵 |
| `token_decomposition.png` | 上述 6 面板图 |

## 6 限制

1. 只做了 long/SCENE8 一个任务，跨任务未验证。
2. 事件 onset 来自运动学代理，带 ±1 chunk 定位不确定性。
3. `active_return` 只有 29/39 条有完整 `[−10,+6]` 窗口，是**幸存者子集**
   （onset 太晚的被排除），本身带轻微选择偏倚。
4. 全部为关联性描述。机制结论需要对 route / expert 做受控干预，而本机没有模拟器。
