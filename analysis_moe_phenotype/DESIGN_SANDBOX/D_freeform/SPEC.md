# 方案 D：极性匹配滤波检测器（Polarity Matched Filter, PMF）——冻结规格

> 校准/开发集：仅 SCENE8 两单位（`main16x32` 512 集 / `grid50x8` 400 集，`proxy_grade=full`）。
> 留出集从未读取。本文件 + `detector.py` 足以在留出集上一次性执行。
> 全部数字见 `calib_report.zh.md` 与 `summary.json`。

---

## 0. 为什么是这个结构，它比阈值状态机和宏状态带驻留强在哪、弱在哪

**结构一句话**：把 E2/E3 给出的 **lead 剖面**当成模板，在线做 **(信号 × 时间) 二维匹配滤波**；
类型不是另外判出来的，而是**模板极性的符号**直接给出的。

### 0.1 为什么不是阈值状态机

阈值状态机问的是"**某个信号有没有越界**"。这批数据有三条事实让这个问题变得低效：

1. **E8 §4 / PROTOCOL §7**：`inst` 与 `cons` 在相图上近乎共线，"组内跨 rollout 有效自由度 ~2"。
   八个信号里真正独立的方向只有两三个，逐信号设阈等于把同一个自由度数了三遍。
   我在校准集的**成功集行**上直接测了冗余（label-free）：`C↔D_tok` ρ=−0.996、
   `S_ent↔D_layer` ρ=−0.97、`V↔A` ρ=+0.94。所以我先**按 ρ>0.90 合并成轴**再匹配，
   而不是对每个信号各设一个阈。
2. **E2 §4 的次序统计是零结果**：D_tok/V/A/C 的首越时刻统计上不可区分，而且首越普遍
   发生在 onset 前 20–37 个 query。"哪个信号先越界"这件事没有信息 ⇒ 以"越界次序"为骨架的
   状态机在这批数据上没有可用的转移条件。E2 给出的其实是 **onset 前 ~3 步的同时相变**，
   而"同时"正好是匹配滤波（内积）而不是级联阈值的场景。
3. **误报**：单信号越阈的报警率由该信号的边缘分布决定；把 K 个轴投影到一个**方向**上，
   同样的误报预算下可以换来更大的有效偏移。本设计把误报预算显式做成一个旋钮 α，
   阈值 = 参照成功集"每集最大分"的 (1−α/2) 分位 —— **时间维的多重比较被这个"每集最大"
   的口径自动吸收**，不需要额外的"连续 N 拍"规则。

### 0.2 为什么不是宏状态带驻留

驻留法把连续量二值化成宏状态再数停留时间。E8 已经把这条路走到底并给出了负结果：
**19 格预声明的组级驻留检验只有 1 格显著，且那格不是占用增驻的态、落在功效最低的套件、
无第二单位复现**（E8 §2.3）。E8 自己的诊断是"这不是'失败不粘'，而是**组级驻留差在本设计下
没有功效**"。二值化把 ±MAD 量级的连续偏移全部丢掉，而 static 事件在 `min_mobk` 上的偏移是
**−7 个 MAD**（我在校准集上实测），把它压成一个 bit 是巨大的信息浪费。
本设计保留连续量，只在**时间**维做窗口平均（这就是驻留想抓的东西的连续版本）。

### 0.3 这个结构真正新的地方

- **类型是免费的**：E2×E3 的头条是"loop 与 static 在门控维度方向**相反**"。
  于是共享轴上的**符号**本身就是分型器：`pol(t) = ⟨axis, T_shared⟩`，`pol>0` 侧是 loop、
  `pol<0` 侧是 static。极性门的原点 0 = 参照成功集的中位，**不是一个需要调的阈**。
  这是消融臂 C（`mob1_w8`）**结构上做不到**的事：loop 与 static 事件的 `mob1_w8` **同号**
  （两者 onset 前都发粘，我在校准集上实测 loop@−6 的 z≈−1.1/−0.6、static@−4 的 z≈−7.7），
  单信号只能靠幅度分型，不能靠方向。
- **两个通道用不同的时间模板**：E2 的 loop 剖面在 lead −10..−5 是 0、−4 起阶跃；
  E3 的 static 剖面"onset 之前就已建立、贯穿始终"是一条平台。所以 loop 用 4 拍窗、
  static 用 5 拍窗，窗长直接等于各自在 E2/E3 里的**显著 lead 跨度**，不是调出来的。
- **"能否离开"做成次级输出而不是报警条件**：E4/E8 的正面发现（成功 loop 爆发后回落、
  致败 loop 平台不落）在 E4 的检验里**没过复现红线**，而且要等 +4..+6 拍才能判，
  会吃掉提前量。所以我把它做成报警之后附带的 `persist` 标志（升级/降级用），
  **不进报警时刻**。

### 0.34 评测口径（**先读这条，否则下面的数字会被误读**）

本任务的失败集全部是跑满 52 拍的超时集，干净成功集几乎全部在 q≈40 前结束。因此
**"全集任意时刻是否报过警"这个口径衡量的是"谁是更好的钟"，不是谁更懂 MoE**：
我实现的平凡集长臂 **T**（一级信号 = 集内 query 序号 `q`，完全不读 MoE 列）在该口径下拿到
**检出 1.000 / 1.000 / 1.000（loop / static / 无事件失败），实测 FA 只有 0.106（main）/ 0.171（grid）**，
压过所有 MoE 臂。我原以为 `q_hi=40` 弃权规则能挡住钟——**实测挡不住**（T 恰好在 q=39/40 响）。

所以载荷口径**必须**是下面两个之一（两者都把纯 q 报警器构造性地归零，并已实测确认）：

| 口径 | 定义 | T 的实测值 |
|---|---|---|
| **匹配视界 margin** | 在同一绝对 q 上，`P(已报警 | 失败-有事件, 存活到 q) − P(已报警 | 干净成功, 存活到 q)` | **0.000**（两单位 × 六个 q 全部） |
| **前缀口径** | 只认 `step ≤ Q0`（Q0=36）的报警 | 检出 **0.000**，FA **0.000**（两单位） |

匹配视界 margin 峰值：主臂 0.747 / 0.552，**臂 P 0.811 / 0.738**，C 0.757 / 0.538，T 0.000 / 0.000。
分型一律用**宏平均（平衡准确率）**：T 与"恒判 static"的常数分类器都恰为 0.500；
微平均会让常数分类器拿到 0.826 / 0.805，具误导性，不得单独引用。

### 0.35 校准集上的结论摘要（细节见 `calib_report.zh.md`）

按**实测误报率对齐**的配对差（同一次运行、同一集；格式 = anytime / prefix36）：

| 比较 | main16x32 | grid50x8 |
|---|---|---|
| 臂 **P** vs C，loop-only 检出（目标 FA 0.10） | **+0.333 / +0.333**（p=.0041） | **+0.386 / +0.404**（p=3e-5 / 2e-5） |
| 臂 **P** vs C，static-only 检出 | −0.008 / 0.000（p=1.0） | −0.016 / −0.031（p=1.0 / .5） |
| **主臂** vs C，loop-only 检出 | −0.133 / −0.133（p=.24） | −0.053 / −0.018（p=.63 / 1.0） |
| **主臂** vs C，static-only 检出 | **−0.226 / −0.218**（p≈0） | −0.016 / −0.031（p=1.0 / .5） |
| 臂 **P** vs **T**，loop-only 检出 | −0.200 / **+0.800**（p=.004 / ≈0） | −0.298 / **+0.684**（p=2e-5 / ≈0） |
| 分型宏平均（主臂 / P / C / T） | .765 / .715 / .561 / **.500** | .733 / **.892** / .679 / **.500** |

⇒ **只用 `rows.npz` 的主臂打不过消融臂 C；加上 E5 的流形轴（臂 P）之后，
loop 通道在两个单位、两个口径下都显著优于 C，static 通道打平；分型（宏平均）主臂与 P 明显优于 C，
而平凡钟臂 T 恰为 0.500。P−C 的结论在剥钟与未剥钟口径下一致，因为 P 与 C 都不是钟。**

### 0.4 它弱在哪（先说清楚，细节见 §7 与 calib_report）

1. **static 通道基本打不过 `mob1_w8`。** static 事件的物理定义就是"机器人不动"，
   而 `mob1_w8`/`min_mobk` 是"表示不动"，二者近乎同义反复（校准集上 static onset 附近
   `mob1_w8` 的 z 到 −10）。我的多信号模板在这条腿上**没有增量**，配对差在两个单位上
   都不显著或为负。这与 E8 §7.5 的"static 的唯一强签名 `stick` 就是既有基线的单调变换，
   不是新信号"完全一致。
2. **主臂（只用 rows.npz）在 loop 通道上对 C 的优势不稳**：main16x32 上配对差为正
   （低 α 处显著），grid50x8 上为零或略负。**只有补上 E5 的流形轴（臂 P）之后，
   loop 通道才在两个单位上同时明显优于 C。**
3. **分型准确率天花板低**：loop-only 失败集被 static 通道先报出来时，报警窗内 static 模板的
   五条轴平均有 **4.91 / 4.75 条同时为正**（`_axis_autopsy.py`，main / grid），
   即路由在那一刻**真的**是冻结型的（只是 `period` 轴的幅度 3.0 / 2.8 明显低于真 static 事件的
   3.9 / 3.8）。报"static"不是规则坏了；但按物理事件标签算，它就是分型错误。
4. **可评区间被健康参照卡死在 q ≤ 40 左右**：校准集里干净成功集在 q=42 之后只剩 ~10 集，
   而 loop onset 中位是 41–44。检测器在 q>q_hi 上**弃权**。这是结构性限制，
   任何"同 q 对照健康集"的方法都撞同一堵墙（E9 §6.6 已量化过）。

---

## 1. 输入契约

`detect(rows_npz_path, group_col) -> {episode_id: record}`

必需的 `rows.npz` 列：

| 列 | 用途 |
|---|---|
| `episode_id`, `control_step` | 集内 query 序号 q（按每集首行归零；断言集内连续） |
| `success` | **只用于构造参照池**（永远排除被评组），见 §2.1 |
| `<group_col>`（`scene` 或 `init_state`） | 组 = scene(=init_state) |
| `late_flow_volatility` (V), `route_acceleration` (A) | 轴 `inst` |
| `token_dispersion` (D_tok), `token_consensus` (C) | 轴 `desync` |
| `gate_entropy` (S_ent) | 轴 `flat` |
| `top12_margin` (S_margin) | 轴 `sharp` |
| `mob_k` (N,8) | 轴 `period` = min_{k≥2} mob_k |
| `mob1_w8` | **仅消融臂 C** 使用 |

**不使用**：`layer_disagreement`（协议指定的负对照，且与 `S_ent` ρ=−0.97 冗余）、
`flow_com`（E3 只有单单位显著、E2 的 U3 方向相反）、`state_action_gap`、`flow_profile`、
`flow_total`、`repeat`、`flow_noise_seed`。

**臂 P 追加**：同目录下的 `reps.npy`（(N,1280) f16，PROTOCOL §3 的标准 layout）。
若文件缺失，臂 P 直接报错（不静默降级）。

**检测器从不读 `events.csv`**：它不知道 onset，也不知道被评集的结局。

---

## 2. 定义与记号

设组 g、集 e、集内 query t（=q）。

### 2.1 参照池（唯一用到 `success` 的地方）

> R_g := { 所有 `success==1` 且 **组 ≠ g** 的集 }

评估组 g 的任何一集时，用到的一切（标准化的中位/MAD、可评上界 q_hi、报警阈值）
都只来自 R_g 与本组的**无标签**行。⇒ 组 g 内部把 `success` 随机置换，组 g 的报警逐位不变（探针 P1）。

### 2.2 标准化（L1）

对每个原始信号 x_i：

```
mu_i^(g)(q)    = median{ x_i(row) : row ∈ R_g, |q(row) − q| ≤ B_ref }
sigma_i^(g)(q) = 1.4826 · MAD(同上)
z_i^raw(e,t)   = ( x_i(e,t) − mu_i^(g)(q=t) ) / sigma_i^(g)(t)
delta_i(e)     = median{ z_i^raw(e', t') : e' ∈ 组(e) \ {e}, q_lo ≤ t' ≤ q_cen_hi }   # 无标签、留一集外
z_i(e,t)       = z_i^raw(e,t) − delta_i(e)
```

- 参照行数 < `n_ref_rows` 或 MAD ≤ 0 的 (q, 信号) 记为**不可评**（z=NaN，该拍不报警）。
- `delta` 用的是本组**全部**集（含失败集）的中位 —— 故意不看标签；在失败富集的组里
  这会把参照拉向失败侧，是**保守**偏差（少报，不是多报）。
- 组心化可用行 < `n_cen_rows` 时 `delta=0`。

### 2.3 轴（冗余合并后；合并规则在成功集行上执行，label-free）

```
inst   = ( z_V + z_A ) / 2                 # ρ(V,A) = +0.938
desync = ( z_D_tok − z_C ) / 2             # ρ(C,D_tok) = −0.996
flat   = z_S_ent                           # D_layer 与 S_ent ρ=−0.97，且是协议负对照 → 弃用
sharp  = z_S_margin
period = z_{min_{k≥2} mob_k}
manifold = z_d_healthy                     # 仅臂 P，见 §5.2
```

### 2.4 模板（**只用符号**，权重一律 1）

```
T_shared = { inst:+1, desync:+1, flat:−1 }                                  # loop/static 反号的共享轴
T_loop   = { inst:+1, desync:+1, flat:−1 }                                  # E2 §7
T_static = { inst:−1, desync:−1, flat:+1, sharp:−1, period:−1 }             # E3 §6
T_loop^P = T_loop ∪ { manifold:+1 }                                         # 臂 P
```

**不用效应量当权重**：E2/E3 的效应量是在含失败集的事件对齐上算出来的，
拿它当权重等于把事件标签写进参数。符号是方向主张，方向已经在两个单位上复现过。

### 2.5 通道分数（L2）与时间匹配滤波（L3）

```
g_c(t)   = mean_{i ∈ T_c} ( s_i^c · axis_i(t) )                 # 模板内积
pol(t)   = mean_{i ∈ T_shared} ( s_i · axis_i(t) )              # 极性
G_loop(t)   = mean( g_loop[t−W_loop+1 .. t] )      W_loop = 4
G_static(t) = mean( g_static[t−W_stat+1 .. t] )    W_stat = 5
极性门： G_loop(t)   仅在 mean(pol[t−3..t]) > 0 时有效
         G_static(t) 仅在 mean(pol[t−4..t]) < 0 时有效
```

窗内任一拍 z 为 NaN ⇒ 该拍 G 为 NaN（不报警）。

### 2.6 可评区间与阈值

```
q_hi(g) = max{ q : #{ e ∈ R_g : n_q(e) > q } ≥ n_ref_eps }        # 健康参照存活到哪
op(e,t) ⇔ q_lo ≤ t ≤ q_hi(g)
theta_c^(g) = Quantile_{1−α/2} { max_{t ∈ op} G_c(e,t) : e ∈ R_g }
```

- `max` 取不到（整集无可评拍）时该集的 max 记为 −∞，仍参与分位数 ⇒ α 是**每集触发预算**。
- 两个通道各分 α/2（Bonferroni），使**任一通道触发**的每集预算 ≈ α。

### 2.7 报警与输出

```
alarm(e) = [ {step:t, channel:c, score:G_c(t), threshold:theta_c, margin:G_c(t)−theta_c}
             for all t ∈ op, c ∈ {loop, static}, G_c(t) ≥ theta_c ]     # 完整序列，按 (t, −margin) 排序
first_alarm = alarm[0]（同拍双触发取 margin 大者）
first_by_channel = 每个通道的首次越阈步
persist = ( G_{c*}(t* + exit_delta) ≥ theta_{c*} )，c*/t* 为 first_alarm；
          若 t*+exit_delta 超出集长或超出 q_hi 则为 None（未决）
```

**下游语义**：`first_alarm.channel == "loop"` → 重采 flow 噪声；`== "static"` → 重规划。
`persist == True` 是升级信号（"没能离开"），`False` 是降级信号。`persist` 不改报警时刻。

---

## 3. 伪代码（可直接对照 `detector.py`）

```python
for g in groups:
    R_g   = [e for e in episodes if e.success == 1 and e.group != g]
    q_hi  = max q such that #{e in R_g : n_q(e) > q} >= n_ref_eps
    # L1：用 R_g 标准化 *全部* 行（因为阈值要用 R_g 集的分数）
    for q in q_lo..qmax:
        ref = rows(R_g) with |q(row) - q| <= B_ref
        if |ref| < n_ref_rows: continue
        for each signal i:  mu, sigma = median(ref_i), 1.4826*MAD(ref_i)
        z_i[rows at q] = (x_i - mu) / sigma
    for each episode e:                      # 组心化（无标签、留一集外）
        donors = rows of (group(e) \ {e}) with q in [q_lo, q_cen_hi]
        if |donors| >= n_cen_rows: z_i[e] -= median(z_i[donors])
    # L2 + L3
    axes  = merge(z)                          # §2.3
    g_L, g_S, pol = template_inner_products(axes)      # §2.4/2.5
    G_L, G_S = trailing_mean(g_L, 4), trailing_mean(g_S, 5)
    G_L[mean(pol,4) <= 0] = NaN ;  G_S[mean(pol,5) >= 0] = NaN      # 极性门
    # 阈值（只用 R_g 的成功集）
    theta_L = quantile_{1-alpha/2}( max_{op} G_L over e in R_g )
    theta_S = quantile_{1-alpha/2}( max_{op} G_S over e in R_g )
    # 报警（只对本组的集）
    for e in group g:
        emit every (t, c) with t in op and G_c(t) >= theta_c
```

---

## 4. 冻结参数表

| 旋钮 | 值 | 语义 | 选定依据（**不得溯源到任何留出集数字**） |
|---|---|---|---|
| `alpha` | **0.10** | 每集**总**误报预算（在参照成功集上），两通道各分 α/2 | 纯工作点参数，取整数值 0.10 = "每 10 个健康集最多 1 个报警"。**没有**用任何实测底座反推。全 α 扫描（0.02/0.05/0.10/0.20/0.35）在 `summary.json`；载荷主张是同一 α 下 arm-vs-C 的配对差，对 α 的选择不敏感（§7.3） |
| `B_ref` | 2 | 参照 q 窗半宽 | 使每个 q 的参照行数在 SCENE8 两单位上都 ≥ 700（q≤38），同时窗宽 ≤ 5 拍不会跨越任务相位；灵敏度见 §7.3 |
| `n_ref_rows` | 60 | 一个 (q, 信号) 可评所需最少参照行 | 中位/MAD 在 n=60 上的相对误差 ≈ 12%，足够；低于此宁可弃权 |
| `n_ref_eps` | 25 | 一个 q 可评所需最少**存活参照集** | 决定 q_hi。25 集 ⇒ 分位/尺度估计不被 3–5 条慢成功轨迹绑架。SCENE8 两单位下得到 q_hi = 40 |
| `q_lo` | 8 | 起始 query | `mob_k`(k=8) 与 `mob1_w8` 都要 8 步拖尾，q<8 无定义（PROTOCOL §2） |
| `q_cen_hi` | 30 | 组心化估计窗上界 | 该窗内成功集 100% 存活（E5 §1 存活审计：main/SCENE8 t=30 存活 1.000），组心化不受存活偏倚污染 |
| `center` | True | 是否组心化 | 组=scene 的水平差是本工作区所有统计的口径（PROTOCOL §1"AUC 只在组内配对"） |
| `n_cen_rows` | 40 | 组心化最少行数 | grid 每组仅 8 集，40 行 ≈ 2 集 × 20 拍，是能给出稳定中位的下限 |
| `W_loop_sig` | 4 | loop 时间窗 | = E2 中 V/A/D_tok/C 的显著 lead 跨度 **−4..−1**（E2 §3） |
| `W_static` | 5 | static 时间窗 | = E3 显著 lead 窗的因果一半 **−4..0**（E3 §1；E3 报全 7 lead 显著，因果侧只能取到 0） |
| `exit_delta` | 6 | "能否离开"观察步数 | E4 对齐曲线与 E8 §3.3 的分叉都在 **+4..+6** |
| `filt_loop` / `filt_static` | `DC` / `DC` | 时间滤波形状 | 见 §4.1 —— 用 **label-free 规则**在校准集上择优 |
| `agg` | `mean` | 信号空间聚合 | 见 §4.2 |
| `polarity_gate` | True | 极性门 | 直接实现 E2×E3 的头条（两型在门控维度反号）；门限 0 = 参照中位，非旋钮 |
| `rho_merge` | 0.90 | 轴合并阈 | 在**成功集行**上算 Spearman，label-free；0.90 之上的三对合并后剩 5 个轴，与 PROTOCOL §7"有效自由度 ~2"的量级相符 |

旋钮总数 **14**，全部有语义、全部在此冻结。留出集上**一个都不许改**。

### 4.1 时间滤波形状的择优（label-free，已执行）

候选：`DC`（平台 = trailing mean）与 `CS`（零均值中心-环绕 = 短窗均值 − 前置基线均值）。
判据先于看检出率声明：**在 E2/E3 声明的剖面形状下算出两者的信号幅度；幅度相同的候选之间，
取"参照成功集每集 max 统计的 (1−α/2) 分位（=阈值）"更小者**（等信号、低阈 ⇒ 严格更灵敏）。

- static 剖面是全程平台（E3：onset 前已建立、贯穿始终）⇒ `CS` 幅度 = 1 − 1 = **0**，直接淘汰
  （实测阈值也更高：main 1.675 vs 1.102、grid 1.619 vs 1.080）。
- loop 剖面是 0（lead ≤ −5）→ 1（lead −4..−1）⇒ 两者幅度都 = 1。实测 LOGO 阈值
  （`_filtsel.py`，冻结实现下）：main `DC` **1.961** vs `CS` 2.057；grid `DC` **2.084** vs `CS` 2.284
  ⇒ **两单位一致选 `DC`**。

诚实记录：**"loop 是瞬变、应该用零均值滤波"这个我最初的设计直觉被自己的 label-free 判据否掉了**
（差分把噪声也翻倍）。冻结为 `DC`。

### 4.2 信号空间聚合：`mean` 还是 `median`（披露）

我先实现了 `median`（"巧合门"：过半轴同时同向才可能给大值，不需要额外旋钮）。
最终冻结为 `mean`，理由是**先验的**：匹配滤波的定义就是与模板取内积，`median` 是它的
稳健化变体、要付效率代价。我在校准集上把两者都算了（`main_median` 诊断臂全部数字在
`summary.json`），`mean` 在两个单位、两个通道上都不劣于 `median`。
**披露**：我是在看到两套数字之后才落笔冻结的；理由本身不依赖检出率，但读者应知道这个次序。

---

## 5. 交付的臂

| 臂 | 一级信号 | 角色 |
|---|---|---|
| **`main`** | 5 轴（inst/desync/flat/sharp/period），只需 `rows.npz` | **主交付**。完全按 §0–§4 冻结 |
| **`P`** | main + `manifold`=z(d_healthy) 进 loop 模板，额外需要 `reps.npy` | **第二交付**，见 §5.2。构造依据来自校准集观察 ⇒ 留出集上应作为**待验假设**读 |
| `C` | `mob1_w8`（loop:+1 / static:−1） | **强制消融臂**（CONSTRAINTS §3）。同结构、同 α、同 LOGO 阈值 |
| `C_sticky` | `mob1_w8`（两通道都 −1） | 描述性上界：单信号在**放弃分型**后的检出天花板。其"分型"是并列时的排序假象，不可引用 |
| **`T`** | 集内 query 序号 `q`（**不读任何 MoE 列**），单腿 + 全额 α | **平凡集长/超时对照臂（必跑）**。用来证明报告的口径没有在量"谁是更好的钟"。它在匹配视界与前缀口径下必须恒为 0 |
| `main_nomob` | main 去掉 `period` 轴 | 诊断：static 通道里有多少不是 mobility |
| `main_median` | main 改用 median 聚合 | 诊断：§4.2 |

### 5.1 消融臂 C 的构造（务求公平）

C 与 main 共享**全部**结构：同一 L1 标准化、同一组心化、同一 q_hi、同一时间窗、
同一极性门、同一 LOGO 阈值口径、同一 α。唯一差别是轴集合 = {`mob1_w8`}。
C 的 loop 方向取 `+1`（高 mobility = churn）、static 取 `−1`（低 mobility = sticky），
这是该信号唯一能给出两个反向通道的取法。
额外提供 `C_sticky`（两通道都取 sticky），因为校准集实测显示 **loop 与 static 事件的
`mob1_w8` 同号**——这条正是 C 在分型上的结构性上限，必须让它以最有利的形态出场。

### 5.2 臂 P：E5 的相位/流形轴

```
d_healthy(e,t) = median of the 5 smallest V2-Hellinger distances from rep(e,t)
                 to { rep(e',t') : e' ∈ R_g, |t' − t| ≤ B_ref, episode(e') ≠ e }
manifold(e,t)  = z(d_healthy) 用与其它信号完全相同的 L1 口径标准化
T_loop^P       = { inst:+1, desync:+1, flat:−1, manifold:+1 }
T_static^P     = T_static （不变：E5 与我的实测都显示 static 失败**没有**离开流形）
```

- rep = `reps.npy` 的 d9 表示（sqrt(P)，(4 深层, 10 action token, 32 专家) 展平 1280 维）；
  距离口径 = PROTOCOL §1 的 V2（逐 (layer,token) 格先算 Hellinger 再对 40 格取均值）。
- 候选集剔除**本集自身的 episode**（避免自匹配把尺度压塌），并整组剔除（LOGO）⇒ 探针 P1 通过。
- 只用 `d_healthy`，**不用** φ̂ / v̂：φ̂ 需要参照集的**总长** `n_q`，在线不可得且会引入长度泄漏。
- **方向来自 E5**（失败 = 偏离健康流形，E5 §3 唯一在 `mob1_w8` 残差后三时点全存活的腿），
  是预先存在的主张；**"只加进 loop 模板、不加进 static 模板"这一条是我在校准集上看到
  loop/static 分离之后决定的**，属校准集内的结构选择，留出集上必须当作待验假设，不是预注册。

---

## 6. 执行契约

```python
import detector
recs = detector.detect(rows_npz_path, group_col)          # 主臂
recs = detector.detect(rows_npz_path, group_col, arm="C") # 消融臂
recs = detector.detect(rows_npz_path, group_col, arm="P") # 需要同目录 reps.npy
recs = detector.detect(rows_npz_path, group_col, arm="T") # 平凡集长对照臂（不读 MoE）
# 对齐工作点：
for cfg in detector.PARAM_SWEEP:                          # [{'alpha':0.02},...,{'alpha':0.35}]
    recs = detector.detect(path, group_col, config=cfg, arm=...)
# 复用同一次评分做多 alpha（快）：
S = detector.score(path, group_col, arm="main")
recs = detector.alarms_from_score(S, alpha=0.05)
```

返回 `{episode_id: record}`，`record` 字段：

| 字段 | 含义 |
|---|---|
| `alarms` | **完整**报警序列 `[{step, channel, score, threshold, margin}, ...]`，按 (step, −margin) 排序 |
| `first_by_channel` | `{"loop": step or None, "static": step or None}` |
| `first_alarm` | `{"step", "channel"}` 或 `None` |
| `alarm`, `alarm_q`, `alarm_type` | 首次报警的便捷字段（等价于 `first_alarm`） |
| `persist` | `True/False/None`（未决） |
| `evaluable_range` | `[q_lo, q_hi(g)]`，`q_hi` 之外检测器**弃权** |
| `maxGL`, `maxGS`, `thr_loop`, `thr_static`, `n_ref_episodes`, `q_hi`, `group`, `n_q` | 审计用 |

**α → 实测 FA 的映射是逐臂的**（同一 α 下不同臂的实测误报率可差 0.02–0.05）。
比较两臂时请用 `PARAM_SWEEP` 对齐到同一**实测** FA；若只在同一 α 下比较，
必须声明"Δ 同时包含召回差与工作点差"。

**机械自检**（`selftest.py`，两语料 × 两臂已全部通过）：
- P1 组内 `success` 随机置换 ⇒ 组内报警序列逐位不变。
- P2 截断到首次报警步 s 仍报 s；截断到 s−1 完全不报。

---

## 7. 已知失效模式与适用边界

0. **不要引用"全集任意时刻"的绝对检出率**（§0.34）。平凡钟臂 T 在该口径下检出 1.000。
   任何在留出集上的执行都必须同时报**匹配视界 margin** 与**前缀口径**，并把 T 一起跑。
   留出集的短任务（queries 中位 8–13）里成功/失败的集长分布可能更接近，届时 T 会更弱，
   但**不能假定**——必须实测。
1. **q > q_hi 弃权**。SCENE8 两单位 q_hi = 40（全部 16/50 个留一组一致）。
   loop onset 中位 41/44、static onset 中位 37/38 ⇒ **相当一部分 loop 事件只能靠"提前很多"
   被抓到，或者根本抓不到**。留出集上短任务（goal/object/spatial，queries 中位 8–13）
   的 q_hi 会低到与 `q_lo=8` 接近，届时**可评窗可能为空**，检测器会对整个任务弃权。
   这是必须如实报的覆盖率数字，不是可以调掉的。
2. **失败富集组的组心化是保守的**：`delta` 用无标签中位，组内失败越多、参照被拉得越高、
   越难报警。SCENE8 两单位里成功集 < 2 集的组（main 1/16、grid 约 10/50）直接 `delta=0`。
3. **`period`（min_{k≥2} mob_k）与 `mob1_w8` 在健康行上只有 ρ≈0.28–0.31，但在 static 事件上
   两者都塌到 −7..−10**。所以 static 通道**看起来**独立、**实际上**与消融臂高度重叠。
   `main_nomob` 诊断臂量化了这一点。
4. **分型的两个口径要分开读**：`first_alarm.channel` 是运营口径（先来先服务）；
   "正确通道是否曾报警"是另一个更宽松的口径。二者在 loop 通道上差得很大。
5. **`persist` 与"集还在跑"混杂**：成功集常在报警后 6 拍内就结束，`persist` 因此
   不可判（`None`）。可判子集是"慢成功 vs 失败"，功效有限，只作描述。
6. **恢复性 loop 的样本量是 2（main）/4（grid）**，任何"能区分暂时错误与真 Trap"的
   结论在本校准集上**都没有功效**。E4 已在更大人群上给出同方向但未过复现红线的结果。
7. **红旗任务**：AUDIT §7.1/§8.4 指出 `KITCHEN_SCENE3`、`open_the_middle_drawer` 的
   loop/trap 通道无效（铰接机构盲区）。留出集执行时，这两个任务的 **loop 通道判定应作废，
   static 通道仍有效**。检测器本身不知道这件事，需要执行器在评分侧剔除。
8. **本方案没有走 E7 的任务流形通道**：E7 的判别结果覆盖留出集任务，不能作为调参依据；
   而且它需要**跨任务**的成功集作在线参照（要同时持有 9 个别的任务的成功流形），
   在线可行性远差于只需同任务成功集的相位通道。故弃用，改用 E5 路线（臂 P）。
