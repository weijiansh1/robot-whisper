# HiMoE-VLA 路由实验：审计与结论修订

日期 2026-08-28
范围：2026-08-25 ～ 08-28 的 MoE routing 实验线（无监督聚类 + 失败类型 + fixed-prefix 预测）
方法：5 个并行 agent 独立复现 / 审计，主控另做独立核验

**本文档自足**，不依赖任何先前对话。所有数字标注了来源与核验状态：

- 🔬 = 主控从原始数据独立核验（不采信 agent）
- 📄 = 从已发布产物文件直接读出
- 🤖 = agent 计算，未独立复核

---

## 0 背景与数据

**模型**：HiMoE-VLA（π0 型 VLA，PaliGemma + gemma action expert）。
MoE 门控分两类：AS-MoE（layers 0–1，3 experts，top-1）与 HB-MoE（layers 2–5, 12–15 共 8 层，
32 experts，top-4）。

**语料**：离线 route corpus，`VLA_MUI_HUB/cache/HiMoE-VLA/{suite}/{task}/right-16x32/`

- `client/episode_XX.npz`：`state (T,8)` 本体感受、`actions (T,10,7)` 动作 chunk、
  `sim_state (T,47)` 特权物理状态
- `server/routes.zarr`：`hb_router_probs (N,8,10,11,32)` fp16、`hb_expert_ids (N,8,10,11,4)`、
  `hb_selected_prob`、`hb_entropy (N,8,10,11)`
  - 轴含义：`N` = 拼接的全部 control step；`8` = HB 层；`10` = flow denoise 迭代；
    **`11` = 后缀 token，其中 token 0 = state，token 1–10 = action**；`32` = experts
- `server/hidden.zarr` 同存
- 设计：每任务 16 initial state × 32 flow-noise seed = 512 episodes
- 每 control step 含 10 个动作子步，**只记录步末位姿**；无 RGB / contact / force / chunk 内稠密状态
- 时序：`sim_state[k]` 记录于 chunk k **执行之前**（$A_k$ 的指令位移与 $s[k+1]-s[k]$ 相关 0.977），
  故"造成事件的 chunk"= 离开目标域的索引 − 1，事件定位天然带 ±1 chunk 不确定性

**任务与结局**（🔬 主控从 `client/summaries.json` 的 `success` + `inference_calls` 直接复算）：

| 任务 | step cap | 失败 | 成功 | 失败长度 | 成功长度 min/中位/max | 跑满 cap 的成功 |
|---|---:|---:|---:|---|---|---:|
| goal/open_the_middle_drawer | 30 | 0 | 512 | — | 12/13/14 | 0 |
| goal/open_the_top_drawer_and_put_the_bowl_inside | 30 | 42 | 470 | 全 = 30 | 17/19/21 | 0 |
| long/KITCHEN_SCENE8_put_both_moka_pots | 52 | 216 | 296 | 全 = 52 | 35/39/52 | 1 |
| spatial/ramekin | 22 | 12 | 500 | 全 = 22 | 9/10/22 | 1 |
| spatial/stove | 22 | 37 | 475 | 全 = 22 | 11/12/20 | 0 |
| **合计** | | **307** | **2253** | **全部 = cap** | | **2** |

**本机不可用**：`mujoco` / `robosuite` / `libero` / `robomimic` 全部 ModuleNotFoundError。
任何需要仿真回放或干预的实验（snapshot replay、expert swap、route clamp、可恢复性 $Q_k(h)$）**做不了**。

---

## 1 总判词

最近两周的 MoE 路线上，**三条主要结论被推翻了两条半，并暴露了一个影响整个度量体系的技术问题。**

被推翻的不是"脚本写错了"——工程可复现性极好：20 项声称数字逐位精确复现、
同种子重跑逐字节相同、12 个报告里 11 个字节级一致、`summary.json` 展平 120 个叶子值 0 差异。

被推翻的是**这些数字的含义**。

---

## 2 结论一：无监督"共识失败核心"等价于 `length == step cap`，routing 特异成分为 0

### 2.1 被审计的原结论

三种"outcome-blind"的表示各自抽出一个高失败率块，声称独立收敛：

| 方法 | 块 | n | success/failure | precision | failure recall |
|---|---|---:|---:|---:|---:|
| shift-aligned landmark kernel | raw C1 | 252 | 6/246 | 0.976 | 0.801 |
| route change-event statistics | C0 | 214 | 5/209 | 0.977 | 0.681 |
| lag-spectrum kNN + Louvain | C7 | 219 | 4/215 | 0.982 | 0.700 |
| **≥2 票共识** | | **217** | **4/213** | **0.982** | **0.694** |

三方法两两 Jaccard 0.707 / 0.725 / 0.827；子采样稳定性 ARI 0.991/0.977（raw K6）、0.862/0.828（event K4）。

### 2.2 语料结构性缺陷

🔬 **307 条失败全部正好跑满 step cap；2253 条成功里只有 2 条跑满。**
这是 LIBERO 协议（成功即终止、失败即跑满）在五个任务上的普遍推论，不是某任务的局部性质。
**42 个 (task × length) 分层里只有 2 个含两类样本**（合计 228 F + 2 S）。

### 2.3 哨兵对照（全部走同一管线、同一 episode 集合、同一指标定义）

| 信息源 | 维度 | precision | recall |
|---|---:|---:|---:|
| **`length ≥ step cap`**（零路由） | **1** | **0.9935** 🔬 | **1.0000** 🔬 |
| 末 11% 相位 EEF 平均步长（纯物理） | 1 | 0.9816 🤖 | 0.6938 🤖 |
| action chunk（无路由） | 49→32 | 0.9866 🤖 | 0.4788 🤖 |
| **三方法 ≥2 票共识核心** | **320×10** | **0.9816** | **0.6938** |
| 路由 (task×相位) 内置换（管线零基线） | 320×10 | 0.42–0.46 🤖 | — |
| 同维高斯噪声 | 320×10 | 0.1399 🤖 | — |

**一个比特在两个指标上都严格赢过整套路由聚类。** 纯物理标量与共识核心**逐位相同**（同为 213/217）。

置换与高斯对照确认这不是管线平凡产物——管线确实提取了依赖 episode 内时间一致性的真实结构，
只是那个结构是长度 / 停滞。

### 2.4 条件独立检验

给定 (task × length) 的 20000 次条件置换 🤖：

| 集合 | 实测失败数 | 条件期望 | z | p |
|---|---:|---:|---:|---:|
| ≥2 票共识核心 | 213 | 213.49 | −0.871 | 1.000 |
| raw C1 | 246 | 246.34 | −0.666 | 1.000 |
| event C0 | 209 | 209.49 | −0.871 | 1.000 |
| lag C7 | 215 | 215.41 | −0.826 | 1.000 |

**四个块全部低于条件期望。** 对比：只按 (task × init) 分层时 z = **+20.76**（p = 0.0002）；
把 length 加进分层后 z = **−1.73**。

钉在共同绝对 query 索引（`t ≤ Tmin−1`，无插值、无 T 依赖、无幸存者偏差）重跑同一管线 🤖：
最佳块 precision 0.387–0.529，与 C1 的 Jaccard 只有 0.18–0.29。**核心消失。**

### 2.5 "三方法独立收敛"用一个比特就能复现

| 配对 | Jaccard 🤖 |
|---|---:|
| `at_cap` ~ raw C1 | 0.792 |
| `at_cap` ~ lag C7 | 0.698 |
| `at_cap` ~ event C0 | 0.676 |
| `at_cap` ~ 共识核心 | 0.691 |
| （已发布）三块彼此 | 0.707 / 0.725 / 0.827 |

纯 (task, at_cap) 划分（零路由数据）与已发布 raw K6 的 **ARI = 0.9475**。

### 2.6 反驳早已在已发布文件里

📄 `analysis/route-change-events/summary.json` 原文（非重跑）：

| 模型 | ROC AUC |
|---|---:|
| task | 0.7654 |
| **task + length** | **0.9995**（AP 0.9884） |
| task + length + events | 0.9975 |

| 增量 | ΔAUC | 95% CI |
|---|---:|---|
| events over task | +0.2255 | [+0.1284, +0.3564] |
| **events over task + length** | **−0.0020** | **[−0.0055, +0.00004]** |

路由 change-event 特征在 (task, length) 之上的增量是 −0.002。
grammar 端同理（task+length 0.9977 > task+length+grammar 0.9949）。
**这个数字当时写进了表格，但与"三方法一致 = 强证据"的叙事直接冲突，冲突未被解决。**

### 2.7 第三根支柱不可复现

🤖 换 3 个新种子完整重跑 + 39 个扰动 Louvain 种子：

| 方法 | 自身划分 ARI | 核心块漂移 |
|---|---|---|
| event C0 | **1.0000** | 214 恒定，Jaccard 1.000 |
| aligned raw C1 | 0.992–0.993 | 252 → 255/260/264，Jaccard 0.954–0.965 |
| **lag C7 (Louvain)** | 0.853–0.940 | **219 → 109 / 109 / 109**，每个新种子都被劈成两半 |

固定 embedding 只扰动 Louvain 种子：**0/39 能用单一社区复出 C7**（Jaccard ≥ 0.9），
单社区最佳 Jaccard 中位数 **0.498**。

连带：≥2 票共识 n = **204–217**，precision 0.9756–0.9816，recall **0.6515–0.6938**；
三方法交集 185 → **97–183**。去掉 lag 的 `raw ∩ event` = 193 → 193/196/197（Jaccard 0.960–0.985），
稳健得多。

原综合报告自设了免责"Lag C7 不单独作为确认性证据"，但随后把它计入头条 217 / 0.982。
**头条数字的第三票来自一个不可重复的对象**，不自洽。

### 2.8 合并

"三种完全不同的方法独立收敛"在**两个互相独立的层面**都不成立：
(a) 它们不独立——一个比特就能复现收敛；(b) 第三个方法不可复现。
而 agent 找到的稳健内核 `raw ∩ event ≈ 193–197 条 / precision ≈ 0.98`，
正是等价于 `length == cap` 的那个集合。**稳健地恢复 nuisance 仍然是恢复 nuisance。**

### 2.9 泄漏定位（拟合是干净的，报告不是）

🤖 逐行核实：特征只加载路由块；`residualize_sequences("raw")` 直接返回拷贝；
聚类函数签名不含 outcome；稳定性门限与选择键全无监督；
`canonicalize(labels, score)` 的 score 是 `diagnostic_mean_soft_speed` / `soft_speed_mean_abs` /
`switch_rate`，**均非 outcome 派生**（ρ(score, failure) = −0.016）。

问题在报告 / 选择环节：
1. `OUTCOME_EXCESS_MIN=0.05` + FDR 是显式 outcome 门
2. 按失败率挑块，候选空间 100+（5 表示 × 2 方法 × full/truncate90 + 其它脚本），无多重性校正
3. `analyze_aligned_route_kernel_truncate90.py:55-56,130-138` 把目标块**硬编码**为
   `REFERENCE_RAW_BLOCK=1`，并对 `episodes!=252 or failures!=246` 直接 `raise`
4. 声明的 primary 变体（`task_residual`）实际是 `no_stable_partition` / outcome excess 0.004，
   传播出去的是 `raw`

**核心教训**：`"outcome_used_before_label_freeze": False` 字面为真，但
**当被泄漏的 nuisance 本身等价于标签时，outcome-blind 拟合不提供任何保护。**

### 2.10 时间归一化的具体缺陷

`PRIMARY_PHASE = PhaseConfig("primary", 0.5, 1.0, 10)`：

1. 主特征只覆盖每条 episode 的**后 50%**——"末段"不是特征之一，**特征全体都在末段**
2. 锚点 10 精确落在最后一个 control step（失败 = timeout 态，成功 = 完成态）
3. **重采样算子本身是 T 的闭式函数**：锚点间距 = `0.0556·(T−1)` 个真实 query。
   `"episode_length_used_as_feature": False` 对特征取值为真，**对采样算子为假**
4. `MAX_SHIFT=2` 的对齐容差同样正比于 T

各锚点原始块被 `task one-hot × [1,T,T²]` 解释的 R² = 0.425–0.682，整体 3200 维 **0.569**。

`canonicalize` 按活跃度升序排块 ⇒ **"C1" 按构造就是活跃度第二低的块**
（C0 0.0085 / C1 0.0128 / C2 0.0134 / …），正是冻结路由所在区间。
所以"C1 表现为 route speed 晚期低速平台"是循环论证。

**truncate90 不是去尾**：它把 10 锚点重铺到 [0.5,0.9]（与 primary 只有 2 个锚点重合），
是"更短窗口上的更密采样"。`episode_length` NMI 只从 **0.655** 降到 **0.648**。

---

## 3 结论二：fixed-prefix 负结果无效——指标坏了，两类被 lead time 完美分开

### 3.1 被审计的原结论

在 long moka-pot 任务里，输入固定截在 pot2 首次到位后第 8 个 query，
纳入截点早于物理 onset 的 158 条（stasis proxy 119 + EEF return-to-pot2-side 39，13 init），
按 init 留一：

| block | pooled LOGO AUC |
|---|---:|
| physical | 0.439 |
| physical + action | 0.457 |
| routing | 0.392 |
| physical + action + routing | 0.436 |

routing 增量 **−0.020 [−0.114, +0.049]** ⇒ 原结论"该协议下未见 routing 增量"。

### 3.2 pooled LOGO AUC 在此 cohort 上无意义

🤖 4641 个正负对里只有 391 个（**8.42%**）落在同一 init 内，其余 91.6% 在比较 13 个不同模型的
不可比分数；6/13 个 init 是单类别。**证据**：一个边际 AUC = 0.545（方向为正）的一维特征，
同管线下 pooled LOGO 拿到 **0.258**。折间系数会翻号。

换 within-init（折内可比）后：

| block | pooled（原报告） | **within-init（修正）** |
|---|---:|---:|
| physical | 0.439 | 0.430 |
| physical + action | 0.457 | 0.509 |
| routing | 0.392 | 0.437 |
| joint | 0.436 | 0.481 |

**全部随机。** 阳性对照（用 routing 预测窗口内 EEF 步长中位数）within-init **0.762** ⇒ 管线没坏。

### 3.3 两类本来就被 lead time 完美分开

| 哨兵（1 维） | 边际 AUC | within-init |
|---|---:|---:|
| **`lead_time`** | **0.978** | **1.000** |
| cut 索引 `q0+8` | 0.545 | 0.417 |
| `init_state_id` | 0.338 | 0.500 |

stasis 有 106/119 条 lead ≤ 2，EEF-return 全部 ≥ 9，重叠区仅 [9,19]。
三种匹配方案（`lead∈[9,19]` n=33、`[9,26]` n=45、±3 caliper 1:1 n=14）
**同时含两类的 init 全部为 0**。
原实验实际比的是「**2 个 query 后停住**」vs「**15 个 query 后回返**」。

### 3.4 功效

🤖 合成注入（沿"同一 rollout 更早相位"的真实路由插值，soft+hard 同时，不用高斯噪声；
λ=0 空注入检出率 0.00 标定正确）：

**MDE80（joint 增量）= +0.252**；routing 单独 vs physical+action = **+0.352**。
观测值 −0.028，**比最小可检出量小一个数量级**。

⇒ **"routing 增量 ≈ 0"这句话不携带信息。**

### 3.5 附带：transductive 泄漏是无罪的

方法复核里排第一的缺陷（pot1/pot2 goal reference 由全部 296 条成功构造，含 held-out init）
实际不影响任何东西：折内重算后参考点位移中位 0.6–0.7 mm、最大 **1.94 mm**，
16 折中 14 折 cohort 不变，within-init 结果与基线一致，增量 −0.0218 → −0.0281。

BDDL 真值坐标不可用：`cook_region` 是两个 pot 共用的 15×15 cm box site，stove 为焊死 fixture。

---

## 4 结论三：二阶失败（适应失败）实验在现有数据上做不了

### 4.1 被检验的命题

因果链把"失败后视觉回返 → 路由回返"钉成后验信号：

$$o_k \to R_k, A_k \to \text{执行 chunk } k \to E_k \to o_{k+1} \to R_{k+1}$$

若一阶事件 $E_k$ 发生在 chunk k 执行中，则 $R_{k+1}$ 已看到结果，只能做事后检测。
提出的转折：它虽不是**第一次**事件的前兆，可能是**第二次失败与最终 trap** 的前兆。
即区分：

- **一阶失败 $E^{(1)}_k$**：物理事件（滑落 / 碰撞 / 空抓 / 提前释放 / 移向错误目标）
- **二阶失败 $E^{(2)}_{k+1}$**：**适应失败**——已收到反馈 $o_{k+1}$，但 $R_{k+1} \approx R_j$ 且
  $A_{k+1} \approx A_j$（$j$ = 先前导致失败的相似状态）

干净设计：**所有样本都已发生一阶事件**，再比较之后恢复成功（$Y=0$）与继续循环（$Y=1$）。

### 4.2 判定：Gate 0 FAIL，219 个组合 0 个通过

🤖 遍历 (task × 一阶事件族 × 阈值 × 结局) 共 219 个组合，**0 个满足双结局风险集要求**。
**不是功效边缘问题，是结构性的。**

新判据 `setback_fraction`（事件当刻 $gd(k) - best(k{-}1) > 5\text{mm}$ 的比例，
即"这事件真的丢掉了已取得的进展吗"）与双结局支持量**严格反相关**：

| landmark | task | n | setback | 双结局聚类内 min(Y0,Y1) |
|---|---|---:|---:|---:|
| `push_ungrasped` | long | 512 | **0.002** | 184 |
| `heightloss@2cm` | long | 267 | **0.000** | 98 |
| `heightloss@3cm` | long | 143 | **0.000** | 54 |
| `grasp_fail@10cm` | long | 49 | 0.531 | 14 |
| `goal_regression@1cm` | long | 36 | **1.000** | 9 |
| `liftloss` / `release_offgoal` / `goal_nbr_loss` | 各任务 | 7–17 | 0.67–1.0 | 0–2 |

**样本够的 landmark 根本不是错误。** 决定性证据：`heightloss@2cm` 的 260 个事件里
**260/260 在 4 个 query 内目标距离就回到事件前最好水平，0/260 重新抬起**——
那是正常的下降放置动作，被当成了"掉落代理"。

早先的 Gate 0 盘点（transport-loss proxy）核实属实 📄：33 个事件，
**31 之后终局失败 / 2 之后终局成功**。阈值放松版 `joint_strict`：31 事件 / 6 成功 / 25 失败。

### 4.3 三条结构性原因

1. **真实一阶失败在这个策略下几乎不可逆**。任务内 `goal_regression ≥ 2cm` 的双结局是
   0/40、1/13、39/0、1/1。跨任务合并无效：off-goal lift-loss 的 33 条成功里 32 条来自 ramekin。
2. **策略基本不重试**。"抓取失败 → 下次再试"的风险集在 2560 条 rollout 里只有 **≤21 例**。
   **这本身是关于该策略的实质结论，值得单独记录。**
3. **终局与 episode 长度结构性混杂**（同 §2）。lead time 由结局决定，不可匹配。

### 4.4 降级演示（两个都失败）

- `grasp_fail` landmark（n=34，4 init）：routing 在 action chunk 之外增量
  **+0.007 [−0.074, +0.188]**——无可检测增量
- `push_ungrasped` landmark（n=416，13 init）：**2 维哨兵拿到 |AUC−0.5| = 0.305，
  而所有实质特征块 ≤ 0.061**——哨兵携带的秩信息多五倍，该 landmark 结论作废

---

## 5 结论四（技术性，影响面最大）：action token 的 HB 门控基本没有在路由

🔬 主控从 `routes.zarr` 独立核验（21 个控制步 × 8 层 × 10 denoise × 11 token）：

| | top-1 | top-4 质量 | 归一化熵 | gap(4th−5th) |
|---|---:|---:|---:|---:|
| **state token** | 0.1285 | 0.2799 | 0.9387 | 2.98e-03 |
| **action token** | **0.0369** | **0.1418** | **0.9988** | **3.78e-04** |
| 均匀分布 | 0.0312 | 0.1250 | 1.0000 | 0 |

逐层 top-1（层 = [2,3,4,5,12,13,14,15]）：

```
state : 0.179  0.112  0.299  0.209  0.063  0.061  0.054  0.052   ← 浅层有真实选择
action: 0.037  0.037  0.037  0.038  0.036  0.037  0.036  0.037   ← 八层全平
```

另外两条 🔬：

- **state token 在 denoise 轴上逐位相同**：`max|p(d) − p(0)| = 0.000e+00`（精确为 0；action = 0.131）。
  **denoise 轴对 state token 不存在。**
- **state / action 路由到近乎不相交的专家**：同 query top-4 重叠 **0.0908** < 随机期望 0.125

### 5.1 这是 bf16 量化平局

bf16 在 0.03 附近的分辨率约 1.2e-4，而第 4 与第 5 名的概率差是 3.8e-4（约 3 倍）。
项目里独立做过的 `analysis/near-tie` 早就量到 📄：
`boundary_tie_rate = 0.391`、`gap_exactly_zero_rate = 0.238`、`gap_median = 4.88e-4`。

🤖 相邻 query 的 top-4 更替率 **69.5%**，而概率分布只移动 `1−H = 2.7e-4`
（state token 是 1.05e-2，**大 39 倍**）；denoise 轴上 top-4 换掉 70%，`1−H` 仅 4.3e-5。

**⇒ 这些"路由变化"主要是量化平局抖动，不是路由决策。**

### 5.2 对二阶命题的直接回应

原假设：$R^{\text{state}}_{k+1}$ 是"模型看到了什么"（后验），
$R^{\text{action}}_{k+1}$ 是"模型准备怎么做"（对下一步的前验）。

**数据给出的答案是反的**：

- $R^{\text{state}}$ 是**唯一真正在路由的东西**，且集中在浅层 L4/L5
- $R^{\text{action}}$——被寄予前验希望的那个——**几乎不是路由**

🤖 用相位无关的比较测"action routing 先动"：5 个信号中 action 早 1 个 query 的 2 个、
state 早 1 个、并列 2 个。与 success 参照比，`stagnation_core` 上 state 分开 12/18、
action 只有 6/21。**"新 $v$ 是后验、由它产生的 action routing 是前验"这个转折不成立**——
不是因为它晚，是因为 action token 那一层没有可读的路由变化。

这同时顺带解释了几个孤立的旧负结果：routing 与 hidden 可互换、routing 是输入稳定度的影子
（回归掉输入持续性后残差 AUC 0.836 → 0.419，r = 0.934）。
如果唯一真实的路由变化就在 state token，而 state token 是观测的函数，这两条都是必然的。

### 5.3 受影响的度量（需要全部换软度量重算）

所有基于 **action token 硬专家身份**的量：
`hard overlap`、`top-4 Jaccard`、`hard lag-1 重叠`、`route recurrence`、`hard_jolt`、
route change events 的一大半。

至少直接波及：

- `replanning-reset-trap` 的 action-token route return AUC **0.733**
- `failure-rollout-periodicity` 的 hard top-4 lag-1 重叠 **0.802 / 0.979**
- `failure-moe-signatures` 描述表里的 hard overlap 列（0.601–0.798）

替代方案：Hellinger / KL / 归一化熵 / top-1 质量等**软**度量。
注意软统计量即使在近均匀分布上也能系统性变化，因此并非全部作废——但必须重算。

---

## 6 一条正结果（暂定，有保留）

🤖 修 fixed-prefix 的 cohort 缺陷（把 success 与 other_long_failure 放回来，
landmark 风险集，同一 cut），444 条（286 成功 + 158 已定义失败，16 init，1191 个折内对）：

| block | within-init AUC |
|---|---:|
| physical + action | 0.468 |
| **routing** | **0.704** |
| joint | 0.625 |

routing 对 physical+action 增量 **+0.237 [+0.074, +0.420]**（CI 不跨 0）；
joint **+0.157 [+0.050, +0.272]**。哨兵检查干净（cut 索引 within-init 0.496）。
全 498 条版本：routing within 0.693，增量 +0.149 [+0.026, +0.302]。

token 拆分无差异：$R^{\text{state}}$ within 0.688（+0.221 [+0.034, +0.398]）、
$R^{\text{action}}$ 0.704（+0.237 [+0.074, +0.420]），CI 大幅重叠。

### 6.1 为什么这条只能算暂定

🔬 主控读了 `PHYSICAL_SIGNALS` 源码定义——物理对照只有 **8 个信号 × 4 统计量 = 32 维**：

```
eef_step_m, eef_pot1_distance_m, eef_pot2_distance_m,
pot1_step_m, pot2_step_m, pot1_goal_distance_m,
pot2_goal_distance_m, eef_pot1_distance_change_m
```

**没有夹爪开度、没有末端姿态 / 轴角、没有抓握代理、没有物体倾角、没有 eef-物体相对位移向量。**

这恰好是 `geometry_control` 当初补进去的那 17 组相对几何量，而补进去之后
MoE 相对物理的增量就塌了（t27 **+0.114 → +0.025**，t34 **+0.039 → +0.005**）。

routing block 含 `route_entropy` / `route_top1_mass` / `expert_entropy`，
完全可以非线性编码"夹爪闭合了没有 / 手里有没有东西"，而这个 32 维对照读不出来。
physical 自身 within-init 只有 0.416。

**⇒ 在用几何增强的物理对照重跑之前，+0.237 不应写进任何结论。这是同一个坑的第二次。**

---

## 7 撤回 / 改口径清单

| 原断言 | 处置 |
|---|---|
| 「三种 outcome-blind 方法独立抽出同一组 rollout」作为独立收敛证据 | **撤回** |
| 「跨任务的晚期未完成核心」 | 改为「**跨任务的 timeout 分层**」 |
| 「不是最后一帧伪影」（truncate90 ARI 0.950） | 过度解读。truncate90 不是去尾；length NMI 0.655 → 0.648 |
| 「route speed 晚期低速平台」 | 循环论证：`canonicalize` 使 C1 按构造就是活跃度第二低的块 |
| 「层间更同步」 | argmax 在平坦曲线上并列取 0 的退化陷阱 |
| 「backward return 更频繁」 | 与已发布 profile 表矛盾（C2 0.116 / C3 0.104 > C0 0.078） |
| 共识核心 `n = 217`、`recall = 0.694` | 改报区间：n **204–217**、recall **0.652–0.694**。`precision ≈ 0.98` 才是稳定量 |
| 「残差失败可行视图两两 ARI 中位数 0.018 ⇒ 无亚型」 | **主统计量无效**（真值下也只有 0.236，观测子采样区间 [0.010, 0.072] 与阳性条件重叠）。改用 `max_excluding_aligned` = 0.164（q90 0.256，真值下 0.576），负结果成立但只覆盖质心分离 ≳0.9–1.1 组内 SD 的亚型 |
| 「peer_rank 全 noise 是无亚型的直接证据」 | **删除**。peer_rank 对语料中最强真实对比也只有 0.030，是零功效视图 |
| 「3 种表示全为 noise」 | 超参产物（`min_cluster_size` 5→15 时该计数在 0–7 变） |
| 「fixed-prefix routing 增量 −0.020 ⇒ 无增量」 | 无信息（MDE80 = +0.252，观测小一个数量级）；且 pooled 指标本身无效 |
| 「上游 goal reference 的 transductive 泄漏是主要缺陷」 | **无罪**（参考点最大位移 1.94 mm，结论不变） |
| 所有 action-token 硬身份度量 | 需换软度量重算（见 §5.3） |

### 保留为有效

- 拟合环节的 outcome-blind 纪律（逐行核实实现正确，`canonicalize` 的 score 不含 outcome）
- **`peer_rank` 表示**：五种替代表示里唯一真正商掉 task/length 的
  （task_nmi 0.074 / length_nmi 0.121 / outcome_excess 0.011），且诚实报告信号随之消失。
  **它是这批分析里的正确对照组**
- HDBSCAN 视角：307/307 失败全为 noise，不依赖块选择
- event 脚本自己的嵌套模型检验（本批最强结果，只是被埋在表格里）
- `raw ∩ event` 两方法核心的**内部**稳定性（193–197，Jaccard 0.96–0.99）——
  但它仍等价于 `at_cap`

---

## 8 一个新混杂（此前未被处理）

🤖 **仅凭 `init_state` 就能在任务内区分 outcome**：
in-sample AUC = top_drawer 0.951 / scene8 0.899 / stove 0.849 / ramekin 0.825。

做 (task, init) 条件置换后：

- **严格早期窗口内四个任务全部归零**（z ∈ [−1.04, +0.50]，p ∈ [0.66, 1.00]）。
  先前看到的 4.4× 富集（stove q0–2）在控制 init 后完全消失——**那是初态编码，不是路由前兆**
- 残留显著的两个任务（stove z = +12.57、scene8 z = +4.34）用的窗口是 `t ≤ Tmin−1`，
  而 Tmin 是最短 episode 长度，对最短成功而言几乎覆盖整条 episode
  （stove Tmin=11、成功中位 12）⇒ 仍是**并发检测**不是预警

**并且**：绝对时间匹配与任务相位匹配**不可兼得**——39 步成功与 52 步 timeout，
固定绝对索引 t=34 对前者是 87% 相位、对后者是 65%。
这不是可修的分析选择，**是语料设计的性质**。

---

## 9 尚未验证的一处怀疑（留给下一轮）

同一个相位混杂可能也波及**检测线**（`rerun-2026-08-27` 的主结论）。

该结论为：相对非特权对照（proprio + 自身 action chunk），MoE 特征在 t34 有增量
**+0.071 [+0.008, +0.147]**；合并窗 t20–t34 +0.043 [+0.006, +0.086]，p=0.012；
三任务 Fisher 合并 **p = 7.58e-5**；留一初始状态 13/13 全为正。

**怀疑**：增量集中在 t34，而 t34 时中位长度 39 的成功已处在 **87% 相位**（马上要成功），
52 步失败在 **65%**。那个 +0.071 可能测的是"这条马上要结束 vs 那条不会结束"。
方向一致的旁证：去掉 t34 后合并 p 掉到 **0.186**。

**这条尚无证据，只是同一混杂的另一种形态。** 建议的检验：
在 t34 上只保留长度相同的成功与失败（scene8 长度 52 的成功只有 1 条，需放宽到长度 ≥ 45 的 13 条），
或改用相位匹配而非绝对索引匹配，看 +0.071 还剩多少。

---

## 10 要让这条研究线重新可做，需要什么

按重要性排序：

1. **成功后不终止，所有 rollout 跑满固定步数。**
   单个最关键的改动——同时解掉 length 混杂（§2）与 lead-time 不可匹配（§3.3）。
   当前 2253 条成功里只有 2 条跑满 cap。
2. **主动注入一阶事件**（受控扰动物体位姿 / 强制开爪），同初态重复多 seed。
   这是唯一能保证 `setback_fraction = 1` 且两类都够的路线（§4）。
3. **恢复模拟器**取 contact 真值 + snapshot replay。当前 contact 真值总量只有 5 条 dense replay。
   这也是一切干预式实验（expert swap、route clamp、可恢复性 $Q_k(h)$）的前提。
4. **扩大 init 覆盖**：现在双结局 init 常只有 1–4 个。
5. **记录 chunk 内稠密状态**，消除 ±1 chunk 的事件定位模糊。
6. **检查路由数值精度**：若 HB 门控在 action token 上确实近均匀，
   应确认这是模型的真实性质还是 bf16 存储导致的信息损失——
   若是后者，重采时应以 fp32 存 router logits。

### 分析侧的方法学要求（无需重采即可执行）

- **每条管线配哨兵**：本轮五个 agent 里有三个是靠哨兵推翻结论的
  （length 哨兵、lead_time 哨兵、2 维时间哨兵）
- **每个负结果配阳性对照 + MDE**：本轮两个负结果都因缺 MDE 而不携带信息
- **逐特征筛查的零分布必须覆盖特征搜索本身**（family-wise max-T，非单特征分位）
- **报块时给全部候选块的表并做多重性校正**（当前候选空间 100+，无校正）
- **优先看逐个体曲线而非群体均值**

---

## 11 产物清单

| 路径 | 内容 |
|---|---|
| `analysis/REPRO-consensus-core-20260828.md` | 20 项逐位核对 + 3 种子完整重跑 + 39 种子 Louvain 扫描 |
| `analysis/AUDIT-clustering-leakage-20260828/` | 哨兵审计；`audit_pipeline.py`（bit-exact 管线重写）+ `audit_00`～`audit_07` |
| `analysis/REPRO-residual-and-alternatives-20260828.md` | 残差线复现 + 阳性对照 |
| `analysis/repro_residual_subtype_power.py` | 种子扫描 / 稳健性 / 阳性对照（带 `--self-test`） |
| `analysis/post-error-adaptation-20260828/` | 二阶失败 PREREG + `feasibility.md` + `token_decomposition.md` |
| `analyze_post_error_adaptation.py` | `--stage {all,gate,ladder,tokens}`，带 `--self-test` |
| `analysis/failure-moe-signatures-fixed/` | fixed-prefix 修复版报告 + 功效曲线 + 哨兵表 |
| `analyze_failure_moe_signatures_fixed.py` | 带 `--self-test`；连续两次全量运行 405 个叶子值 0 差异 |
| `analysis/failure-moe-signatures-repro/` | 原样复跑产物（用于 diff） |

原目录未被覆盖（一处例外：`analyze_residual_failure_dynamics.py` 不解析 `sys.argv`，
被 `--self-test` 扫描触发整跑并写回硬编码目录；已核验内容零变化（脚本确定性，
独立 out-dir 重跑 md5 全同）并恢复原时间戳）。

**建议**：给所有硬编码输出路径的脚本补 argparse。

---

## 12 一句话

> **可复现性没有问题，含义有问题。**
> 无监督"共识失败核心"是 `episode_length == step cap` 的有损重建；
> fixed-prefix 负结果的指标失效且无功效；
> 二阶失败实验因策略几乎不重试而结构性不可做；
> 而 action token 上的 HB 门控本身近似均匀，其"路由变化"主要是 bf16 平局抖动——
> 真正在路由的是 state token 的浅层，那是观测的读出，不是策略的前验。
