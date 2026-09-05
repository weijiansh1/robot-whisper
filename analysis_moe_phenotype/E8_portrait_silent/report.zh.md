# E8 MoE routing 相图与宏状态 + E9 MoE-silent 边界

- 日期 2026-09-04；协议 `../PROTOCOL.md` §4 / §5-E8·E9 / §7 / §8（Amendment 1）；审计 `../audit/AUDIT.md` §7 / §8
- 脚本 `run_e8e9.py`（44 s）、`make_figs.py`；seed 20260903，NPERM 2000；语料只读
- 产物：`report.zh.md, summary.json, macrostate_occupancy.csv, transitions.csv,
  silent_inventory.csv, family_pself.csv, silent_by_task.csv, survival_audit.csv,
  e9_failure_inventory.csv, e9_failure_inventory_taskpooled.csv, e9_success_fpbase.csv,
  fig1_macrostate_occupancy.png, fig2_event_bundles.png, fig3_phase_portrait.png`
- 输入：`features/{grid50x8,main16x32}/**/rows.npz`（305,030 行 / 18,560 集 / 45 任务）
  + `events/**/events.csv`（v2：SCENE8 full，其余 objaware）

**一句话结论**：五轴宏状态里失败集确实系统性地"离开低波动/高共识（flatten/shared）
盆地、进入高波动且高粘滞的态"，并且 loop 与 static 事件在宏状态带上呈现干净的**双分离**；
但这些差异**在预声明的组级驻留检验族里几乎全部不显著**（19 格中唯一 p<0.05 的格落在功效
最低的 object 套件、且不是占用增驻的态）。E9 的 silent 比例只有 2.4%（可评集 4.1%），
但**同判据下 88% 的成功集也被判 detectable** —— 这个判据的伪阳性底座太高，只能说
"失败比成功更少 silent（4.1% vs 12.3%）"，不能说"96% 的失败可被 MoE 检出"。

---

## 1. 口径与冻结（先于看结果声明）

**五轴 Γ_q**（组内无标签 z：`(x − 组中位) / 组MAD`，原始 MAD 不乘 1.4826）：

| bit | 轴 | 定义 | 高位语义 |
|---|---|---|---|
| 0 | `inst` | z(V + A) | high-vol（路由在 flow 上抖） |
| 1 | `comm` | z(top12_margin) | high-commit（门控尖锐） |
| 2 | `cons` | z(token_consensus − token_dispersion) | high-consensus（10 个 action token 同意） |
| 3 | `stick` | z(−mob1_w8) | sticky（d9 表示 8 步内复现） |
| 4 | `flow` | z(flow_com) | late-flow（去噪变化质心靠后） |

- 组 = (corpus, suite, task, scene/init_state)，与 PROTOCOL §1 的 (checkpoint, task, scene)
  一致（grid50x8 的 4 个套件各自一个 checkpoint）。
- **分析窗 q ≥ 8**（stickiness 用 mob1_w8，需 8 步拖尾）；集内 q 由 `control_step` 减该集
  首行 `control_step` 得到（rows.npz 的 `control_step` 是任务级累计计数，已断言归零后
  逐集连续 0..n-1）。
- 有效行 **156,478 / 305,030 = 51.3%**。组丢弃 **162 / 2,080 (7.8%)**，原因**全部**是
  "窗内行 < 8"（短任务的短集组）；**MAD = 0 的组 0 个**，五轴在所有存活组上都可用。
- 宏状态 = 五轴按组中位二分 → 32 态；转移 = 集内相邻 (q, q+1) 且两端有效；
  按 (corpus, suite, outcome) 池化。

### 1.1 32 态 vs 16 态：为什么留 32 态（冻结判据 + 两版占用都落盘）

冻结判据："任一语料的池化占用中 <1% 的态数 **> 8**（即多于 1/4）则去掉 flow_shape 并到 16 态"。

| 版本 | grid50x8 <1% 态数 | grid 最小占用 | main16x32 <1% 态数 | main 最小占用 | 触发合并？ |
|---|---|---|---|---|---|
| 32 态 | 3 | 0.80% | **8** | 0.51% | 否（8 不 > 8，卡在阈值上） |
| 16 态 | 0 | 1.98% | 0 | 1.28% | — |

判据**未触发**，主版 = 32 态。但 main16x32 正好卡在阈值（8 个态 <1%，最小 0.51%），
这是个边缘决定，因此 **16 态版的占用/转移/驻留全部同样落盘**
（`macrostate_occupancy.csv` 与 `transitions.csv` 的 `n_states` 列 = 32 / 16），
并另跑一遍同结构的次级检验族（§2.3）。两版的定性结论一致。

### 1.2 检验族（预声明，唯一主族，小族）

- 主族 = 7 个 corpus×suite 格 × 每格池化 Δocc = occ_fail − occ_succ 最大的**前 3 个可用态**
  （"可用" = 成/败两侧池化访问数各 ≥ 50）→ 名义 21 格，实际 **19 格**
  （grid object 只有 2 个态满足访问阈）。
- 每格效应 = **组内** P_fail(s→s) − P_succ(s→s)（每侧 from-s 转移 ≥ 3 才计入该组）。
- 置换 = 组级 sign-flip（整组翻号，同组跨格同号）2000 次，`groupwise_signflip_maxt`，
  maxT 覆盖全部可估计格。**选择统计（占用差）与检验统计（自转移差）解耦。**
- 16 态版同法另跑，**声明为次级族**（21 格），不与主族合并校正。
- 其余（占用图、事件轨迹束、相图、与既有工作对照）**全部描述性**，不做显著性主张。

---

## 2. E8-1 成功 vs 失败的占用与驻留

### 2.1 单轴边缘占用差（fail − succ，百分点；32 态汇总）

| corpus/suite | inst=1 | comm=1 | cons=1 | stick=1 | flow=1 |
|---|---:|---:|---:|---:|---:|
| grid50x8/goal | +4.58 | +1.77 | +6.01 | +7.33 | +1.19 |
| grid50x8/long | +5.76 | +3.35 | +1.94 | +9.73 | +0.76 |
| grid50x8/object | **+34.12** | +22.84 | **−23.74** | −3.94 | +23.80 |
| grid50x8/spatial | +19.90 | +8.28 | +7.51 | +7.57 | −9.79 |
| main16x32/goal | +5.78 | +1.03 | +8.83 | +6.75 | +1.27 |
| main16x32/long (SCENE8) | +0.71 | −1.20 | +4.41 | **+10.95** | −3.49 |
| main16x32/spatial | +32.37 | +14.93 | +12.40 | +14.54 | −12.38 |

**7/7 格 inst=1 增驻，6/7 格 stick=1 增驻**（grid object 例外，−3.9）。
comm=1 增驻 6/7。cons 与 flow 方向不一致（object 是唯一 cons 大幅下降的格）。
即：**失败行更常同时处在"高路由波动"和"高表示粘滞"上**——路由在 flow 内部抖，
但 d9 表示 8 步内回到老地方。这是本实验最稳的描述性事实（图 `fig1` 面板 A）。

### 2.2 失败增驻的态是哪几个（每格 top-3 Δocc，配象限语义）

| corpus/suite | rank | state | 语义标注 | occ_fail | occ_succ | Δocc | 组内 ΔP(s→s) | p_maxT |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| grid/goal | 1 | 31 | high-vol / high-commit / high-consensus / sticky / late-flow | .0481 | .0079 | **+.0402** | — (0 组) | — |
| grid/goal | 2 | 29 | high-vol / low-commit / high-consensus / sticky / late-flow | .0370 | .0084 | +.0286 | — (0 组) | — |
| grid/goal | 3 | 6 | low-vol / high-commit / high-consensus / mobile / early-flow | .0437 | .0276 | +.0161 | — (0 组) | — |
| grid/long | 1 | 27 | high-vol / high-commit / low-consensus / sticky / late-flow | .1184 | .0736 | **+.0448** | +0.168 (56 组) | 0.909 |
| grid/long | 2 | 31 | high-vol / high-commit / high-consensus / sticky / late-flow | .0320 | .0106 | +.0214 | +0.079 (4) | 1.000 |
| grid/long | 3 | 15 | high-vol / high-commit / high-consensus / sticky / early-flow | .0267 | .0090 | +.0178 | −0.067 (2) | 1.000 |
| grid/object | 1 | 19 | high-vol / high-commit / low-consensus / mobile / late-flow | .2486 | .1028 | **+.1459** | +0.108 (9) | 0.989 |
| grid/object | 2 | 27 | high-vol / high-commit / low-consensus / sticky / late-flow | .2338 | .2401 | −.0063 | **+0.472 (8)** | **0.018** |
| grid/spatial | 1 | 15 | high-vol / high-commit / high-consensus / sticky / early-flow | .1207 | .0117 | **+.1091** | — (0 组) | — |
| grid/spatial | 2 | 13 | high-vol / low-commit / high-consensus / sticky / early-flow | .0969 | .0159 | +.0810 | — (0 组) | — |
| grid/spatial | 3 | 31 | high-vol / high-commit / high-consensus / sticky / late-flow | .0306 | .0137 | +.0169 | — (0 组) | — |
| main/goal | 1 | 3 | high-vol / high-commit / low-consensus / mobile / early-flow | .0552 | .0364 | +.0188 | −0.066 (3) | 1.000 |
| main/goal | 2 | 4 | low-vol / low-commit / high-consensus / mobile / early-flow | .1061 | .1224 | −.0163 | +0.211 (6) | 0.802 |
| main/goal | 3 | 12 | low-vol / low-commit / high-consensus / sticky / early-flow | .1374 | .1734 | −.0359 | −0.061 (5) | 1.000 |
| main/long | 1 | 12 | low-vol / low-commit / high-consensus / sticky / early-flow | .1224 | .1007 | **+.0216** | +0.046 (13) | 1.000 |
| main/long | 2 | 14 | low-vol / high-commit / high-consensus / sticky / early-flow | .0479 | .0299 | +.0179 | +0.081 (12) | 0.997 |
| main/long | 3 | 29 | high-vol / low-commit / high-consensus / sticky / late-flow | .0203 | .0073 | +.0130 | −0.019 (4) | 1.000 |
| main/spatial | 1 | 19 | high-vol / high-commit / low-consensus / mobile / late-flow | .1239 | .0780 | **+.0459** | +0.251 (5) | 0.673 |
| main/spatial | 2 | 27 | high-vol / high-commit / low-consensus / sticky / late-flow | .0743 | .0786 | −.0043 | −0.228 (5) | 0.780 |

（完整 19 格 + 16 态次级族见 `family_pself.csv`；`macrostate_occupancy.csv` 有全部
7×32 与 7×16 格的占用与 P_self。）

**语义归纳（描述性）**：
1. **"原地空转"型**（grid goal / grid long / grid spatial 的 top 态：31、27、15、13、29）
   ——`high-vol + sticky` 共同出现。路由在 10 步 flow 内剧烈重排，但每个 query 的 d9
   终点几乎回到 8 步前，即"内部忙、外部不动"。这是四个 grid 套件里最一致的失败增驻象限。
2. **"高波动 + 低共识 + 仍在动"型**（grid object 19、main spatial 19、main goal 3）
   ——`high-vol + low-consensus + mobile`。10 个 action token 各说各话，表示还在漂。
   object 套件（全套仅 37 个失败集）尤其极端：Δocc 高达 +14.6pp。
3. **SCENE8（main long）是唯一的例外**：top 态 12 = `low-vol / low-commit /
   high-consensus / sticky / early-flow`——"安静、平坦、共识、冻住"，
   属于 static/flatten 家族而不是 switching 家族。失败富集主语料的失败表型
   与 grid 的短任务失败表型**不是同一个象限**。

### 2.3 检验结果（预声明族）：几乎全零结果

- 主族 19 格中可估计 13 格（其余 6 格**没有任何一个组同时在成/败两侧有 ≥3 条 from-s 转移**
  ——grid goal 与 grid spatial 全部 3 格都是这种情况：失败集在这些套件里平均每组不到 1 条，
  一个组根本凑不出成/败配对的驻留估计）。
- 13 个可估计格中，**只有 1 格 maxT 后 p < 0.05**：
  `grid50x8|libero_object`，state 27（high-vol/high-commit/low-consensus/sticky/late-flow），
  组内 ΔP(s→s) = **+0.472**（8 组），**p_maxT = 0.0175**。
  **必须打折看**：(a) 该态 Δocc = −0.006，**不是**占用增驻的态，它进族只是因为 object
  套件只有 2 个态过访问阈（"top-3"退化为"全部"）；(b) object 是全语料功效最低的套件
  （37 个失败集，AUDIT §7.2）；(c) 8 个组的 sign-flip 最小可达 p 本身就在 10⁻³ 量级。
  按 PROTOCOL §4"复现"要求（≥2 个独立单位同向），该格**没有第二个单位复现**
  （main spatial 的同一 state 27 是 −0.228），因此**不作为头条主张**。
- 其余 12 格 p ∈ [0.67, 1.00]，包括所有占用增驻最强的态。
- **16 态次级族**：21 格中 15 格可估计，**最小 p = 0.520**（main goal state 15），**全零结果**。

**零结果的诚实读法**：占用差是行级的、样本量大（15 万行）因而看得见；驻留差是组级的，
置换单位是组，而"每组既有失败又有成功、且两侧在同一个 32 分之一的态里各积累 ≥3 条转移"
这个条件在本语料里只有 2–69 个组能满足。**这不是"失败不粘"，而是"组级驻留差在本设计下
没有功效"**。下面这条描述性事实可以佐证方向：

> **全局驻留**：144 个两侧都有定义的 (cell, state) 组合里，**85.4% 满足 P_ii(fail) > P_ii(succ)**，
> 平均 +0.083、中位 +0.065（16 态版：93.3%，平均 +0.130）。即失败集的宏状态动力学
> **整体变慢**，而不是在某个特定盆地里变粘（图 `fig1` 面板 B 几乎整片偏红）。
> 这是描述性观察，**不进检验族**。

---

## 3. E8-2 事件轨迹束：loop 走哪条带、static 走哪条带

事件集：loop 用 **full + objaware 有效层**（按 AUDIT §7.1 / §8.4 剔除 grid long
KITCHEN_SCENE3 与两语料 goal middle_drawer 的 loop/trap 通道），static 全层有效。
主体 = 失败集事件；成功集事件另报。事件数与 AUDIT §8.2 对账一致：

| 束 | 事件集数 | onset−6 处落在 q≥8 窗内的行数 | onset 处 |
|---|---:|---:|---:|
| loop_fail | **389** | 292 | 348 |
| loop_succ | 257 | 36 | 179 |
| static_fail | **317** | 312 | 317 |
| static_succ | 21 | 21 | 21 |

（loop_fail 389 = AUDIT 全语料 loop 失败事件 395 − SCENE3 的 6；static_fail 317 与
AUDIT §8.2 逐套件相加完全一致。）

**带定义（冻结）**：switching/desync 带 = `inst=1 & cons=0`（8 个态）；
flatten/shared 带 = `inst=0 & cons=1`（8 个态）。

### 3.1 带占用：干净的双分离

| rel | −6 | −4 | −2 | 0 | +2 | +4 | +6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| loop_fail · switch/desync | .507 | .518 | .547 | .440 | .418 | .488 | .541 |
| loop_fail · flatten/shared | .267 | .283 | .254 | .270 | .248 | .265 | .216 |
| static_fail · switch/desync | .192 | .208 | .237 | .287 | .259 | .319 | .335 |
| static_fail · flatten/shared | .513 | .521 | .483 | .461 | .469 | .394 | .404 |

带极性（switch − flatten）：loop 全程 **+0.17 ~ +0.33**，static 全程 **−0.33 ~ −0.07**，
**13/13 个 lead 上两者符号相反、无交叉**（图 `fig2` 面板 C）。
预测"loop 应走 desync/switching 带、static 应入 flatten/shared 带"**成立**。

### 3.2 与 PROTOCOL §5 五条表型预测逐条对照

轴均值 = 事件集在该 lead 上的组内 z（0 = 组中位），非事件对照隐含在 z 的定义里。

| # | 预测（出处） | 本实验对应量 | 观测 | 判定 |
|---|---|---|---|---|
| 1 | **Disperse**：D_tok@−3 升，且**早于** V/A（§5-E2 新腿 a） | `cons = z(C − D_tok)` 的**谷** vs `inst = z(V+A)` 的**峰** | cons 谷在 rel −4（−0.89）/−3（−0.77）；inst 峰在 rel −3（+1.19） | **方向成立，"先于"只领先约 1 个 query**（弱）。cons 是复合量，无法把 C 与 D_tok 拆开，因此这是**降级核对**，不能替代 E2 的 D_tok 单信号腿 |
| 2 | **Switch**：V@−2、A@−2 升（§5-E2 复现腿） | `inst` | rel −6..0 全程 +0.85 ~ +1.19（−2 处 +1.06） | **成立**，但**在 −6 就已抬起**，不是 onset 前 2 步才出现的尖锐前兆 |
| 3 | **Unstable commitment**：S_margin@−2 升（§5-E2 新腿 b） | `comm = z(margin)` | rel −6..−1 全程 +0.75 ~ +1.11（−2 处 +0.99，−1 处 +1.11） | **成立**：门控更尖锐**与**路由更抖同时出现，与"unstable commitment"的语义一致 |
| 4 | **Collapse**：C@+1,+2 升而 C@−2 不升（§5-E2 新腿 c） | `cons` | −2 处 −0.51（低于组中位，"不升"✔）；+1 −0.11、+2 −0.09，从 −4 的 −0.89 **单调回升** | **成立（方向）**，但 +1/+2 仍**略低于**组中位，是"回到中位"而不是"越过中位" |
| 5a | **static flattening**：S_ent@0 升（=门控变平；§5-E3） | `comm = z(margin)`，变平 ⇒ comm 应**降** | comm 在 −6..+6 全程 ≈ −0.10 ~ +0.11，onset 处 +0.11 | **不成立**（零结果）。static 事件在 commitment 轴上与组中位无区别 |
| 5b | **static shared-support**：C@0 升（§5-E3） | `cons` | 从 −6 的 +0.34 **单调下降**到 onset 的 +0.15，之后继续降到 +6 的 −0.00 | **不成立**（方向相反）。cons 在 static 前略高，但**随 onset 逼近而下降**，不是 onset 锁定的上升 |

**static 的真实签名是第 4 轴**：`stick` 从 rel −6 的 **+1.94** 单调升到 onset 的 **+4.60** MAD
（图 `fig2` 面板 E）。这在两个意义上要打折：
(a) `stick = z(−mob1_w8)`，而 mob1_w8 就是 PROTOCOL §2 点名的"既有最强基线
route_mobility_w8"，**这不是新信号**；
(b) `static_succ`（n=21）显示**同样量级**的 stick 抬升（onset 处 +5.52），
说明这条抬升**锁事件不锁结局**——它反映"机器人不动⇒观测几乎不变⇒路由复现"，
不能当成失败预警。

### 3.3 一条计划外但清晰的观察：分叉在 onset 之后

loop 成功集（n=257）的 **onset 前**与失败集几乎同型：inst 在 −3 达 +1.36、comm 在 −3 达 +1.85
（比失败集更高）。分离发生在 **onset 之后**（图 `fig2` 面板 F）：

| rel | 0 | +2 | +4 | +6 |
|---|---:|---:|---:|---:|
| loop_fail · stick | +1.04 | +1.14 | +1.47 | **+2.03** |
| loop_succ · stick | −0.48 | −0.88 | **−1.06** | +0.18 |
| loop_succ · cons | −0.30 | −0.34 | **+0.57** | +0.48 |
| loop_succ · switch/desync 带占用 | .592 | .526 | **.239** | .202 |
| loop_succ · flatten/shared 带占用 | .229 | .265 | **.607** | .681 |

即：**"切换爆发"本身不是失败特征，失败特征是爆发之后没有重新流动起来**——
失败集继续冻结（stick 单调升到 +2.0），成功集重新移动并恢复共识（stick 转负、cons 转正、
带占用整体翻到 flatten/shared）。
**重要限制**：loop_succ 在 rel<0 只有 36–99 行落在 q≥8 窗内（事件总数 257），
所以它的 onset 前曲线只代表"onset 较晚的成功 loop"子集，**存在选择偏倚，不可当作对照组**。

---

## 4. E8-3 相图：instability × consensus 平面

图 `fig3_phase_portrait.png`（其余三轴边缘化；两轴均为组内 z，MAD 单位）。

- 平面上存在一条极强的**反对角脊**：inst 与 cons 近似线性负相关。也就是说，
  五轴里的"波动"和"共识"在很大程度上是**同一个自由度的两端**，
  这与 PROTOCOL §7 已知上限"组内跨 rollout 有效自由度 ~2"吻合。
- **成功束**（124,080 行在视野内）众数在 (inst ≈ −1.1, cons ≈ +1.1)。
- **失败束**（25,904 行）众数右移到 (≈ 0, ≈ +0.6)，并沿脊向 (inst 2–4, cons −2 ~ −3) 拖尾。
  差分面板：成功侧富集在 (−1, +1)，失败侧富集在 (0, +0.7) 及整条高波动/低共识臂。
- **loop-pre 束**（onset −6..−1，1,710 行）明显落在脊的**高波动/低共识那一半**。
- **static-pre 束**（1,881 行）反而**紧贴原点**（≈ −0.3, +0.6），比成功众数还靠右一点点：
  **static 在这张平面上几乎无特征**——它的信息在被边缘化掉的 stickiness 轴上。
  这解释了为什么 §3.2 里 static 的两条预测（flattening / shared-support）都不成立。

---

## 5. 与既有 signal-matrix 宏状态工作的对照

既有工作（`analysis_signal_matrix_trainfree/`，16 态）用的四轴是
**route_change / gate_entropy / late_flow_volatility / state_action_gap**，
阈值 = **语料全局中位**（在 t∈[8,13] 上取），窗 t∈[8,30)，并用同批 physical progress
标注 descriptive basin。结论：A 最高 `P_ii=0.645`@state10；B 最高 `P_ii=0.540`@state13；
高粘滞低进展态 = A:{10,11}、B:{11}。

**语料关系**：其 **B = `cache/.../KITCHEN_SCENE8/right-16x32`，就是本实验的
`main16x32/libero_long`（512 集）**，可直接对照；其 A 是
`himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828`（352 条分支），
**不在本实验的两个 paper-right 语料里**，只能做同型比较、不能同表。

**轴的对应关系（3/4 对得上，各有独有轴）**：

| 既有轴 | 本版轴 | 关系 |
|---|---|---|
| route_change（高=1） | `stick`（高=1 表示**低**移动） | 对应但**符号相反**；本版还多了 8 步拖尾平均 |
| gate_entropy（高=1） | `comm`（高=1 表示 margin 大 = 尖） | 对应但**符号相反** |
| late_flow_volatility（高=1） | `inst` = z(V+A) | 同向（本版并入了 route_acceleration） |
| state_action_gap | — | 本版无此轴（PROTOCOL §2 把它留作延续信号，不入 Γ_q） |
| — | `cons` = z(C − D_tok) | 本版新增（token 间共识/离散） |
| — | `flow` = z(flow_com) | 本版新增（32 态版专有） |
| 阈值 | 语料全局中位 | **本版改为组内中位**：占用差因此是**组内**对比，天然扣掉了任务/场景水平差 |

**是否复现同型结构：是。**
把既有 state13（route_change=1, entropy=0, late_flow=1, gap=1）按上表翻译，
对应本版 16 态的 `inst=1, comm=1, stick=0` = state **3**（cons=0）或 7（cons=1）。
在同一个语料（main16x32/libero_long = 既有 B）上：

| 本版 16 态 | 语义 | occ_fail | occ_succ | P_ii(fail) | P_ii(succ) | 池化 P_ii |
|---:|---|---:|---:|---:|---:|---:|
| **3** | high-vol / high-commit / low-consensus / **mobile** | .1397 | .1657 | .524 | .473 | **≈.498** |
| 11 | high-vol / high-commit / low-consensus / sticky | .1429 | .1523 | .525 | .452 | ≈.488 |
| 4 | low-vol / low-commit / high-consensus / mobile | .1268 | .1646 | .483 | .434 | ≈.456 |
| **12** | low-vol / low-commit / high-consensus / **sticky** | .1669 | .1389 | .502 | .412 | ≈.461 |

- 本版在该语料上的**最粘态就是 state 3**（池化 P_ii ≈ 0.498），即**既有 B 的 state13 的直接
  翻译**，且量级相当（0.498 vs 0.540）。**同型结构复现。**
- 既有 A 的 state10（entropy=1, gap=1, route_change=0, late_flow=0）翻译为本版
  `inst=0, comm=0, stick=1` = state **12**（cons=1）或 8（cons=0）。state 12 正是
  **main16x32/libero_long 失败增驻第 1 名**（Δocc +2.8pp，§2.2），
  语义"低波动/低承诺/高共识/粘滞" ≈ 既有的"低进展高粘滞"。**同型结构复现，且在本版里
  它带上了失败标签方向。**
- **两版的主要差异**：(a) 阈值从全局中位改成组内中位，本版的占用差全部是组内量；
  (b) 本版用 token 共识替换 state-action gap，用 P_ii 差替代 physical-progress 标注
  （PROTOCOL §7 禁止用物理特征给 MoE 结论补救）；(c) 本版 P_ii 绝对值普遍低于既有
  （0.41–0.53 vs 0.54–0.65），因为组内中位二分把每组切成 50/50，压掉了"整段轨迹都在
  同一侧"造成的长驻留。**结论上两版一致：最粘的宏状态是"低共识+高波动"或
  "低波动+粘滞"这两类，且失败侧更粘。**

---

## 6. E9 MoE-silent 边界（描述性 inventory）

### 6.1 判据（冻结）

- 9 信号 = 8 核心 {V, A, S_ent, S_margin, C, D_tok, D_layer, flow_com} + mob1_w8。
- 固定时点：long 套件 {20, 25, 30}，其余 {8, 12, 16}（PROTOCOL §1）。
- 百分位 = 该集该信号该**绝对 q** 的值，相对**同组成功集在同一 q** 的分布（midrank，
  参考样本 ≥5 否则该格 NA）。集级偏离 `dev_fixed = max_信号 max_时点 |pct − 50|`。
- 事件腿：onset = `trap_onset_q`（loop 通道无效的任务改用 `static_onset_q`），
  lead ∈ {−4..0}，`dev_event` 同法。
- **silent ⇔ dev_fixed < 40 且（无 onset 或 dev_event < 40）**；detectable ⇔ 任一已定义
  dev ≥ 40；`dev_fixed` 无可评格 ⇒ **not_evaluable**（不计入分母）。
- **反向审计**：成功集逐集**组内 leave-one-out**（把自己从参考分布里扣掉）跑同一判据。
- 次级变体（描述性）：参考池改为**同 task 全 scene** 的成功集同 q（缓解组内 n<5）。

### 6.2 主口径结果 × 套件

| corpus/suite | 失败 n | detectable | **silent** | not_evaluable | silent/可评 |
|---|---:|---:|---:|---:|---:|
| grid50x8/goal | 102 | 43 | **0** | 59 | 0.000 |
| grid50x8/long | 267 | 82 | **6** | 179 | 0.068 |
| grid50x8/object | 37 | 31 | **2** | 4 | 0.061 |
| grid50x8/spatial | 126 | 71 | **8** | 47 | 0.101 |
| main16x32/goal | 42 | 41 | **1** | 0 | 0.024 |
| main16x32/long (SCENE8) | 216 | 155 | **1** | 60 | 0.006 |
| main16x32/spatial | 49 | 47 | **2** | 0 | 0.041 |
| **合计** | **839** | **470** | **20** | **349** | **0.041** |

逐任务表见 `silent_by_task.csv`（39 个有失败集的任务中 14 个至少有 1 个 silent 集）；
silent 集清单见 `silent_inventory.csv`（含主口径 20 集 + 次级口径 12 集，`ref_variant` 列区分）。

### 6.3 反向审计：伪阳性底座（**这才是本节的重点**）

同判据跑 17,721 个成功集（组内留一）：

| corpus/suite | 成功 n | detectable | silent | not_evaluable | **成功集被判 detectable 的比例（可评）** |
|---|---:|---:|---:|---:|---:|
| grid50x8/goal | 3,898 | 2,658 | 372 | 868 | **0.877** |
| grid50x8/long | 3,733 | 2,771 | 501 | 461 | **0.847** |
| grid50x8/object | 3,963 | 3,589 | 370 | 4 | **0.907** |
| grid50x8/spatial | 3,874 | 2,959 | 484 | 431 | **0.859** |
| main16x32/goal | 982 | 913 | 64 | 5 | **0.934** |
| main16x32/long | 296 | 282 | 5 | 9 | **0.983** |
| main16x32/spatial | 975 | 808 | 167 | 0 | **0.829** |
| **合计** | **17,721** | **13,980** | **1,963** | **1,778** | **0.877** |

**结论：判据的伪阳性底座是 87.7%。**"96% 的失败集 detectable"这句话**不能说**——
因为 88% 的成功集也 detectable。唯一站得住的对比是：

> **可评集中 silent 比例：失败 4.08%（20/490） vs 成功 12.31%（1963/15943）。
> 失败集变 silent 的概率约为成功集的 1/3。**

即 routing 确实带结局信息（失败更容易在某处越出健康带），但"越带"这个二值判据本身
**不是一个可用的检测器**。

原因是可算的：判据要求 9 信号 × 3 时点（≤27 格）**全部**落在 10–90 分位带内。
在组 = (task, scene)、每组仅 8 draw 的设计下，同一 q 上的存活成功参考常常只有 5–7 条，
百分位被量化得很粗（n=5 时 midrank 只能取 {0,20,40,60,80,100}，其中 2/6 的取值
天然 |pct−50| ≥ 40）。事实上 detectable 失败集的 `dev_fixed` **中位数正好是 50.0**，
即"最响的那一格恰好落在参考样本之外"——这是参考集粒度，不是模型内部的强偏离。

### 6.4 次级变体（task 池化参考）：可评率修好了，底座还在

把参考池换成同 task 全 scene 的成功集（n 从 5–7 提升到 50–400）：

| | 失败 | 成功 |
|---|---|---|
| 可评 / 总数 | 827 / 839（98.6%） | 16,249 / 17,721（91.7%） |
| silent（占可评） | **12（1.45%）** | **3,837（23.61%）** |
| detectable（占可评） | 98.55% | **76.39%** |

方向与主口径一致且**分离更大**（1.45% vs 23.61%，约 1/16）：说明主口径的高伪阳性
主要来自小参考集的量化噪声，把参考做大以后成功集的"越带率"从 88% 降到 76%，
失败集反而更容易被判 detectable。但 **76% 的伪阳性底座依然让"detectable"这个标签
接近无信息**。两个口径的 silent 集只有 1 集重合
（grid spatial `bowl_on_cookie_box` scene40 repeat5），说明 silent 名单本身对参考池
选择很敏感——**不要把 silent 清单当成稳定的"这些失败 MoE 看不见"名录**。

### 6.5 silent vs detectable 对照（主口径，仅可评集）

| | n | n_queries 中位 | 有事件比例 | 可评格数中位 | dev_fixed 中位 |
|---|---:|---:|---:|---:|---:|
| detectable | 470 | 52 | 0.762 | 18 | **50.0** |
| silent | 20 | 25 | 0.550 | **9** | 35.7 |

- **n_queries 的差异是套件混淆**：套件内两者的 n_queries 中位完全相同
  （grid long 都是 52、spatial 都是 22、object 都是 28、main goal 都是 30）。
  失败集几乎都是跑满 query 上限的超时集。
- **真正的差异是可评格数**：silent 集中位只有 9 格（= 3 个固定时点里只有 1 个有
  ≥5 条同组成功参考），detectable 是 18 格。**"silent"和"参考不足"高度混淆。**
- **事件腿从未被真正核验**：20 个 silent 集**全部** `n_event_cells = 0`
  （其中 11 集是有 onset 的），即"事件对齐 lead∈{−4..0} 同样无越带"这条**一次都没能验证**
  ——onset 附近同组存活成功集不足 5 条。按判据它们按 `dev_fixed` 归类并打了
  `event_unverified` 标记。**这 20 个 silent 集应被读作"在可评的那 1–3 个固定时点上没越带"，
  而不是"全轨迹 MoE 无痕"。**
- detectable 集里"最响信号"的分布（`argmax_fixed`）：
  V 228 / S_margin 64 / A 42 / S_ent 39 / flow_com 33 / C 28 / mob1_w8 25 / D_layer 9 / **D_tok 2**。
  V（late-flow volatility）近半数。D_layer 只有 9 次、D_tok 只有 2 次，
  与 PROTOCOL §7 "D^layer 预期近退化、当负对照"一致。

### 6.6 存活审计（PROTOCOL §1 要求，固定时点必附）

`survival_audit.csv`；`frac_alive = P(n_queries > t)`：

| corpus/suite | t | 失败存活 | 成功存活 |
|---|---:|---:|---:|
| grid/goal | 8 / 12 / 16 | 1.00 / 1.00 / 1.00 | 0.793 / 0.326 / **0.098** |
| grid/long | 20 / 25 / 30 | 1.00 / 1.00 / 1.00 | 0.911 / 0.305 / **0.090** |
| grid/object | 8 / 12 / 16 | 1.00 / 1.00 / 1.00 | 1.000 / 0.851 / **0.047** |
| grid/spatial | 8 / 12 / 16 | 1.00 / 1.00 / 1.00 | 0.915 / 0.164 / **0.001** |
| main/goal | 8 / 12 / 16 | 1.00 / 1.00 / 1.00 | 1.000 / 0.780 / 0.479 |
| main/long (SCENE8) | 20 / 25 / 30 | 1.00 / 1.00 / 1.00 | 1.000 / 1.000 / **1.000** |
| main/spatial | 8 / 12 / 16 | 1.00 / 1.00 / 1.00 | 1.000 / 0.263 / **0.004** |

这是 E9 全部 not_evaluable（349/839 = 41.6%）的根源，而且是**结构性**的：
**失败集 100% 活到所有固定时点（它们是超时集），成功集在最后一个固定时点上普遍只剩
0.1%–9.8%**。也就是说，在"失败还在跑"的时刻，**同组根本没有健康轨迹可比**。
唯一不受此影响的单位是 **main16x32/SCENE8**（成功集 100% 活到 t=30），
而它恰好是 silent 比例最低的格（1/216）。

---

## 7. 边界段（PROTOCOL §7 诚实条款）

1. **这是可观测性边界，不是"MoE 看不见 Trap"的证据。** 本节所有 silent 判定都受限于
   同组健康参考的存在性。20 个 silent 集里 0 个通过了事件腿核验，11 个有事件却无法核验；
   silent 名单在换参考池后只有 1 集稳定。**任何"某类 Trap 是 MoE-silent"的主张在本数据上
   都不成立**——不是被证伪，是**没有做出这个判断的条件**。
2. **不做外部特征补救。** 未用任何动作/物理特征去"救"silent 集，也未用 physical progress
   给宏状态贴盆地标签（既有 signal-matrix 工作用过，本版按 §7 去掉了）。
3. **判据本身的信息量极低。** 87.7%（主）/ 76.4%（次级）的成功集也被判 detectable。
   本节唯一可引用的量化结论是**相对**的：失败集 silent 概率是成功集的 1/3（主）到 1/16（次级）。
4. **E8 的正向发现全部是描述性的。** 预声明的 19 格组级驻留族只有 1 格显著，且该格
   (a) 不是占用增驻态、(b) 落在功效最低套件、(c) 无第二单位复现，
   按 PROTOCOL §4 不构成头条主张。占用差与带分离虽然幅度大（最高 +34pp），
   但**没有配套的置换检验支撑**——它们是相图的形状描述，不是显著性结论。
5. **已知上限如实引用**：相图显示 inst 与 cons 近乎共线（图 `fig3`），与"组内跨 rollout
   有效自由度 ~2"一致；`D_layer` 在 E9 里只当过 9 次最响信号（近退化，符合负对照预期）；
   static 事件的唯一强签名 `stick` 就是既有最强基线 `mob1_w8` 的单调变换，
   **不是新信号**，且它锁事件不锁结局（成功 static 事件抬升同样大）。
6. **窗口代价**：q ≥ 8 的分析窗丢掉 48.7% 的行与 7.8% 的组，短任务（goal/object/spatial，
   queries 中位 8–13）在 E8 里只贡献轨迹尾部。E8 的结论对"早期"没有发言权。
7. **loop 事件通道**：已按 AUDIT §7.1 剔除 KITCHEN_SCENE3 与 middle_drawer；
   AUDIT §8.4 提到的 bbq_sauce/milk 等高重试 pick 任务的成功集 loop 触发偏高（14.7%/5.9%）
   仍留在 loop_succ 束里，是 §3.3 那条观察的已知污染源。

---

## 8. 复现

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd analysis_moe_phenotype
python3 E8_portrait_silent/run_e8e9.py      # 44 s -> 全部 CSV/JSON/中间量
python3 E8_portrait_silent/make_figs.py     # -> fig1/fig2/fig3
```
