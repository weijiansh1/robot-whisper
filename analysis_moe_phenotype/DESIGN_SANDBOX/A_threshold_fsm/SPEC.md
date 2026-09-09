# 方案 A：标量阈值状态机 —— 冻结规格

> 冻结日 2026-09-04。校准集 = SCENE8 两个单位（`main16x32` 512 集 / `grid50x8` 400 集，
> `proxy_grade=full`）。留出集（39 个 grid 任务 + 4 个 main 任务）从未读取。
> 本文件是**可直接在留出集上一次性执行**的规格；参数已全部冻死，不留"待调"项。

---

## 0. 一句话

两个互不合并的在线通道，各是一个跑在**组内自校准 robust z** 上的标量阈值状态机：
**loop 通道**判"进了切换带能不能出来"，**static 通道**判"低速复现 + 高共识是否早已建立
且持续"。每次报警带类型（`loop` / `static`），**输出完整报警序列**而不是只报第一次。

---

## 1. 输入契约

### 1.1 必需（`rows.npz`，逐行 = 一个 control step / query）

| 列 | 类型 | 用途 |
|---|---|---|
| `episode_id` | int | 行按 episode 连续分块（断言） |
| `control_step` | int | 任务级累计计数；集内 q = `control_step − 该集首行 control_step`，断言逐集连续 `0..n-1` |
| `success` | int8 | **只用于阈值分位（成功集 + 留一组外）**，不进入归一化、不进入任何逐行判定 |
| `<group_col>`（默认 `scene`） | int | 组 = scene（grid50x8）/ init_state（main16x32，同为 `scene` 列） |
| `late_flow_volatility` (V) | f32 | 臂 A loop 一级信号 `V+A` |
| `route_acceleration` (A) | f32 | 同上 |
| `mob_k` | f32 (R,8) | 臂 A static 一级信号 `min_{k=2..8} mob_k`（= `mob_k[:,1:]` 的行最小） |
| `token_consensus` (C) | f32 | 两臂共用的 static 二级信号 |
| `mob1_w8` | f32 | **消融臂 C** 一级信号 |

### 1.2 检测器**不接受**事件表

`detect()` 的签名里没有 `events_csv_path`。事件表（含 `success` 与三个 onset）由执行器
独占并独自评分；检测器只从 `rows.npz` 拿 `success` 用于阈值分位。

### 1.3 信号定义域

`mob_k(k≥2)` 与 `mob1_w8` 在 `q < 8` 无定义（NaN）。因此**两臂的所有通道统一从 q ≥ 8 起
判定**（结构常量 `q_min = 8`），以保证两臂在完全相同的支撑上比较。

---

## 2. 组内自校准 z（**定位/尺度不看标签**）

```
对每个信号 x、每个组 g：
    med_g = median( x 在组 g 的**全部**有限行 )            # 不看 success
    mad_g = median(|x − med_g|) × 1.4826
    z(t)  = (x(t) − med_g) / mad_g
组内有限行 < 60 或 mad_g ≈ 0 → 退到任务级全部行；仍不行 → z = NaN（弃权）
```

这是 E4 §(b) / E8 §1 的"组内**无标签** robust z"口径。**标签只出现在 §3 的阈值分位里。**
两个直接后果：

- **P1（组内 `success` 随机置换 → 组内报警逐位不变）按构造成立**：
  组内 z 与标签无关；组 g 的阈值只由**其它组**的成功行决定（LOGO）。
- **没有"参照贫乏"弃权**：SCENE8 两单位 912/912 集全部拿到组级尺度
  （曾经的成功集 z 版本在 grid 上有 27% 退到任务级）。

分位阈值对 MAD 常数不变（1.4826 只是同比缩放），故不存在 E4 §补充腿4 记录的
"结论对 z 尺度敏感"问题。

---

## 3. 参数表（6 个旋钮，全部冻结）

**统一约定（先于看任何检出/误报数字确定）：每条腿都放在"成功集 10% 尾"上。**
幅度腿取分位（p90 / p10）；持续腿取"使健康驻留段长度 ≥ 该值的概率 ≤ 0.10 的最小整数"
（离散量的同一约定）。`θ_exit` 是唯一的非尾旋钮，取 z 的中心 p50。

| # | 旋钮 | 冻结值 | 语义 | 选定依据（**只来自校准集的成功集统计**） |
|---|---|---|---|---|
| 1 | `θ_enter` | **成功集 z(V+A) 的 p90** | 进入"切换带"的门槛 | 10% 健康尾约定。进入必须**宽松** —— 恢复性与致败 loop **都会**进带，进入只负责起表，判别全靠"出不出得来" |
| 2 | `θ_exit` | **成功集 z(V+A) 的 p50** | "回落到健康中心"的门槛 | z 的中心。分位形式使它与 MAD 常数无关 |
| 3 | `K` | **6** 步 | 进入后允许的最长"未回落"步数；第 K 个有效步仍未回落 → 报警 | 校准集成功集上"从越 θ_enter 起、连续处于 z ≥ θ_exit 的驻留段长度"的 10% 尾：两单位合并 `P(run≥5)=0.107`、`P(run≥6)=0.090` → K=6。**两个单位独立各自也都给 6** |
| 4 | `θ_s` | **成功集 z(min_{k≥2}mob_k) 的 p10** | "低速复现"腿 | 10% 健康尾约定 |
| 5 | `θ_c` | **成功集 z(token_consensus) 的 p90** | "高共识"腿 | 10% 健康尾约定 |
| 6 | `M` | **3** 步 | 两条件同时成立所需的连续有效步数 | 校准集成功集上"两条件同时成立的连续段长度"的 10% 尾：合并 `P(run≥2)=0.194`、`P(run≥3)=0.010` → M=3。**两个单位独立各自也都给 3** |

### 3.1 参数溯源审计（守门条款）

| 旋钮 | 唯一数据来源 | 是否触及留出集派生量 |
|---|---|---|
| θ_enter / θ_exit / θ_s / θ_c | 校准集成功行的 z 经验分位（LOGO） | **否** |
| K | 校准集成功集的 loop 驻留段长度经验分布 | **否** |
| M | 校准集成功集的 static 联合条件段长度经验分布 | **否** |
| q_min = 8 | `mob_k(k≥2)`/`mob1_w8` 的定义域 | **否** |
| min_scale_rows = 60 / min_thr_rows = 30 | MAD 与分位的估计稳定性下限（结构常量） | **否** |

**没有任何参数溯源到 E4 报告的 24–81% 误报底座**（该数字来自含留出集任务的分析）。
本设计对该数字既不拟合、也不作为选参目标；校准报告只在"口径说明"里提到它一次。

### 3.2 结构常量（非旋钮）

| 常量 | 值 | 依据 |
|---|---|---|
| `q_min` | 8 | 信号定义域；两臂统一，保证同支撑对比 |
| `min_scale_rows` | 60 | 组内定位/尺度所需最少有限行（MAD 相对标准误 ~10% 量级） |
| `min_thr_rows` | 30 | LOGO 分位所需最少成功行；不足则该组弃权（`abstain=True`） |
| MAD 常数 | 1.4826 | 与 E4 主口径一致；对分位阈值无影响 |

### 3.3 阈值的数值形态与执行方式

**冻结的是分位规则，不是数字。** 执行时按下式取数（**留一组外**，LOGO）：

```
θ(g) = percentile( { z：所有 success==1 且 q>=8 且 组 != g 的行 }, 冻结分位 )
      # 样本 < 30 行 -> NaN -> 该组 abstain=True，不报警且不计入任何分母
```

校准集上实现值（供审计对表，**不是**留出集应写死的数字）：

| 单位 | θ_enter | θ_exit | θ_s | θ_c | LOGO 极差（θ_enter） |
|---|---|---|---|---|---|
| main16x32 SCENE8 | +1.671 | +0.073 | −1.291 | +1.080 | [1.621, 1.714] |
| grid50x8 SCENE8 | +1.641 | +0.126 | −1.285 | +1.041 | [1.620, 1.665] |

`K=6` 与 `M=3` 是**冻结整数**，留出集上不重估。

---

## 4. 判定规则（伪代码）

两个通道并行。**每次驻留/每个满足段最多报一次**，回落/重置后重新武装 —— 因此输出是
一条完整的报警序列，而不是只有首次报警。**无效行**（z 为 NaN）一律"跳过"：
既不进入、不回落，也不推进任何计数器。

```
# ---------- loop 通道 ----------
inband, clock, armed <- False, 0, True
for t = q_min .. n-1:
    z <- z_VA[t];  if not finite(z): continue
    if not inband:
        if z > theta_enter:  inband, clock, armed <- True, 0, True   # 进带起表
    else:
        if z < theta_exit:   inband, clock, armed <- False, 0, True  # 回落=能离开
                             continue
        clock <- clock + 1
        if armed and clock >= K:
            EMIT(step=t, channel="loop");  armed <- False            # 本次驻留只报一次

# ---------- static 通道 ----------
run, armed <- 0, True
for t = q_min .. n-1:
    a, b <- z_minmobk[t], z_C[t]
    if not (finite(a) and finite(b)): continue
    if a < theta_s and b > theta_c:
        run <- run + 1
        if armed and run >= M:  EMIT(step=t, channel="static");  armed <- False
    else:
        run, armed <- 0, True
```

- loop 报警时刻 = **进入后第 K 个有效行**（进入行本身不计入 clock），无缺失时 `t0 + K`。
- 允许重复进出：回落后回到 IDLE，可再次进入并重新起表、重新报警。

## 5. 输出契约

```
detect(rows_npz_path, group_col="scene") -> {episode_id: record}

record = {
  "group": int, "n_q": int, "scale_level": "group"|"task"|"none",
  "alarms": [{"step": int, "channel": "loop"|"static"}, ...],   # 时间升序，完整序列
  "first_loop_alarm_q", "first_static_alarm_q",                 # -1 = 未报
  "first_alarm_q", "first_alarm_channel",
  "loop_n_entries", "static_max_run",
  "theta_enter", "theta_exit", "theta_s", "theta_c",            # 该集实际用的 LOGO 阈值
  "abstain": bool                                               # True -> 不计入任何分母
}
```

## 6. 工作点对齐（`PARAM_SWEEP`）

`detector.PARAM_SWEEP` 给出四条曲线，供执行器把两臂拉到**同一个误报工作点**再比较：

| key | 内容 |
|---|---|
| `"A"` / `"C"` | loop 通道：只动进入分位 ∈ {60,70,75,80,85,90,93,95,97}，其余冻结 |
| `"A_static"` / `"C_static"` | static 通道：只动 θ_c 分位 ∈ {50,60,70,75,80,85,90,95}，其余冻结 |

每个元素是 `(label, Params)`，直接传给 `detect(..., params=P)`。
**若执行器不做对齐**：两臂 FA 差 > 0.01 时，报告必须逐字带上
"**Δ 同时包含召回差与工作点差**"的警示。

## 7. 在线因果性与机械自检

第 t 行的判定只用到该 episode 的第 `0..t` 行。三个探针（`detector.py` 内，
`run_calib.py` 每次运行都跑）：

| 探针 | 定义 | 校准集结果 |
|---|---|---|
| **P1a** | 组内随机置换 `success` → **该组内**报警序列逐位不变 | **0 / 640 与 0 / 160 不符**（20 次置换） |
| **P1b** | 同时置换**所有组** → 全体报警序列 | 1.88% / 2.63% 变化 —— 这是 CONSTRAINTS §1 强制的 LOGO 成功集校准本身的抖动（其它组的分位随之移动），**不是泄漏**；如实披露 |
| **P2** | 对每个报警步 s：截断到 s 仍报 s；截断到 s−1 **不**报 s | **0 违规**（臂 A 206/130 个报警、臂 C 151/96 个） |
| 破坏性因果 | 把 t0 之后所有行换成 `NaN`/`±1e6` 重跑，要求 ≤t0 的报警序列逐位不变 | **0 违规**（每单位每臂抽 120 集 × 6 个 t0 × 3 种填充） |

推论（评估用）：状态机 latch 且不回退 ⇒ "在 q ≤ T 的前缀上运行会报警" ⇔
"完整运行的首次报警 ≤ T"。所有前缀/匹配视界指标据此计算。

**离线成分声明**：组内 med/MAD 与 LOGO 分位来自离线参照池（其它 episode 的行）。
这是部署假设（上线前需要该组若干条 rollout + 其它组的成功标签），不含当前 episode
的未来行。

---

## 8. 消融臂 C（强制对照）

同一套状态机结构，一级信号换成 `mob1_w8`（= 既有最强 train-free 基线 d9+Hellinger+W8）。
**方向翻转**：失败时 mobility **降低**，故 loop 通道进入条件是 `z(mob1_w8) < θ_enter^C`、
回落条件是 `z(mob1_w8) > θ_exit^C`；static 一级腿本来就取低侧，直接同分位。

| 臂 A | 臂 C（低侧镜像） |
|---|---|
| θ_enter = p90 of z(V+A) | θ_enter^C = p(100−90) = p10 of z(mob1_w8) |
| θ_exit = p50 of z(V+A) | θ_exit^C = p(100−50) = p50 of z(mob1_w8) |
| θ_s = p10 of z(min mob_k) | θ_s^C = p10 of z(mob1_w8) |
| θ_c = p90 of z(C) | 同（二级腿不变） |

持续腿有两种同口径读法，**两个都报**：

- **C-rulematched（主）**：同一条"10% 健康尾"规则作用在 `mob1_w8` 自己的驻留段分布上
  → `K_C = 12`（合并 `P(run≥11)=0.107`, `P(run≥12)=0.072`）、`M_C = 2`
  （`P(run≥2)=0.065`）。**这才是真正的同口径** —— 两臂每条腿都拿到同样的 10% 健康预算。
- **C-sameknobs（次）**：旋钮数值与臂 A 完全相同（K=6, M=3）。列出以免被指"改了旋钮"。

## 8b. 平凡集长臂 T（更强的对照，**必读**）

**臂 T = 同结构状态机，一级信号换成集内 query 序号 `q`，完全不读 MoE。**

| 臂 A | 臂 T |
|---|---|
| θ_enter = p90 of z(V+A)，进入 = z 高 | θ_enter^T = p90 of z(q)，进入 = z(q) 高（"集跑得久"） |
| θ_exit = p50 of z(V+A) | θ_exit^T = p50 of z(q) |
| θ_s = p10 of z(min mob_k) | θ_s^T = p10 of z(−q)（低侧同分位） |
| θ_c = p90 of z(C) | **无**（读不到 MoE）→ static 通道只剩一条腿（比合取更容易响，不削弱 T） |
| K=6, M=3 | **K_T = 5、M_T = 6**（同一"10% 健康尾"规则；两单位独立各自也都给 5 与 6） |

`DT.PARAMS_T` / `DT.PARAMS_T_SAMEKNOBS`；扫描曲线 `PARAM_SWEEP["T"]` / `["T_static"]`。

**臂 T 作废了"全集任意时刻报警"这个口径**：它在该口径上以 FA 0.038/0.019 拿到
det_loop 1.000/0.645、det_static 1.000/0.855，碾压 A 与 C。任何在该口径上的横向比较
**都不构成 MoE 信息的证据**。

**唯一能把"钟"与"路由信息"分开的口径 = 匹配视界（同绝对 q 配对）**：配对的两集在同一个
绝对 q 上被评判，故任何纯 q 报警器必然 `mh_det ≡ mh_fa`（对角线）。实测臂 T 在 U1 上
margin 恰为 0.000（9/9 工作点），构造性验证。**执行者在留出集上必须同时跑臂 T，
并以 `margin = mh_det − mh_fa` 作为主指标；只报全集 det/FA 的结果一律无效。**

---

## 9. 在留出集上执行

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - <<'EOF'
import sys; sys.path.insert(0, "DESIGN_SANDBOX/A_threshold_fsm")
import detector as DT
res = DT.detect("features/<corpus>/<suite>/<task>/rows.npz", group_col="scene")
# 消融臂：DT.detect(..., params=DT.PARAMS_C)   /  DT.PARAMS_C_SAMEKNOBS
# 工作点对齐：遍历 DT.PARAM_SWEEP["A"] 与 ["C"]
EOF
```

或 `python3 detector.py <rows.npz>`。校准集全套复现：`python3 run_calib.py`（约 15 s）。

**不得**在留出集上重估 `K`、`M`、任何分位级别，或改动 `q_min` / 回退规则。

## 10. 已知限制（执行者必须一并读 `calib_report.zh.md`）

1. **static 通道在冻结点是死的**（检出 0.000 / 0.014）：`min_mobk` 低与 `C` 高这两条腿
   在健康行上负相关（r = −0.65 / −0.68），p10 ∧ p90 近乎互斥。
   单留 `min_mobk` 腿（M=3 不变）给 0.769 / 0.913 检出、提前量 −4 / −5、
   ~100% 早于 onset，代价 FA 0.210 / 0.288 —— **这不是冻结规格**，记为下一轮候选。
2. **loop 提前量不足**：中位 −2（U1）/ **+1**（U2），只有 52% / 47% 早于 onset。
3. **z 池化 over q 有强漂移**：干净成功集 z(V+A) 中位从 q24–31 的 −0.46/−0.49 升到
   q≥39 的 +1.68/+1.94（已越过 θ_enter）。检测器**部分是一只钟**。
4. **集长混杂（最严重）**：`n_queries` 的组内配对 AUC（失败 vs 干净成功）= 0.9992 / 1.0000。
   **平凡集长臂 T 在"全集任意时刻"口径上打败臂 A 与臂 C**（§8b）。
   ⇒ 该口径的绝对数字与横向比较**全部无效**；**载荷主张必须锚在匹配视界的
   `margin = mh_det − mh_fa`（臂 T 在此构造性为零）与前缀截断 q≤38 上**。
   校准集结果：loop 通道 margin 在 9/9 工作点、两个单位上全为正（中位 +0.108/+0.139），
   臂 T 为 0/9，臂 C 只有一个单位为正。
5. **"能否离开"未获支持**：臂 T 在同一 onset 窗上同向且 p 更小（0.159/0.102 vs A 的
   0.273/0.354）——该窗不是时点中性的。本方案对该命题无独立贡献。
6. **分型只在冻结点之外兑现**：冻结点上臂 A 与臂 T 的**平衡分型都是 0.500**（常数分类器）；
   θ_c 松到 ≤p70 后臂 A 达 0.674–0.799（双单位），臂 T 恒为 0.500、臂 C ≤0.52。
