# 方案 B 冻结规格：宏状态带驻留时间检测器

> 版本 2026-09-04（v2，已按守门 agent 的三条接口修正 + 方法学更正重做）。
> **参数一次冻死。** 校准集 = SCENE8 两单位（`main16x32` / `grid50x8` 各一个
> `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`）。留出集从未读取。
> 实现 = `detector.py`；本文件是它的语义定义，二者不一致以本文件为准并视为 bug。

---

## 0. 一句话

把 E8 的五轴宏状态原样搬过来，只问一件事：**「连续待在某条带里多久」**。
loop 通道盯 switching/desync 带，static 通道盯 flatten/shared 带，连续驻留达到 K / M 步
就报一条警并给出分型。每通道只有一个旋钮。

---

## 1. 输入契约

```python
from detector import detect, PARAMS, PARAM_SWEEP
alarms = detect(rows_npz_path, group_col)        # group_col 默认 "scene"
# -> {episode_id: {"alarms": [{"step": int, "channel": "loop"|"static"}, ...],
#                  "ref_level": "scene"|"task"|"none", "group": int, "n_q": int}}
detect(rows_npz_path, group_col, arm="C")        # 消融臂 C，channel = "sticky"
```

`rows.npz` 必须含（全部来自现有 `features/<corpus>/<suite>/<task>/rows.npz`，只读）：

| 列 | 用途 |
|---|---|
| `episode_id` (int) | 集划分；行必须按 (episode_id, control_step) 升序（已断言） |
| `control_step` (int) | 任务级累计计数；集内 q = control_step − 该集首行 control_step，断言 = 0..n−1 |
| `scene` (int)（= `group_col`） | 组标识 = init_state |
| `late_flow_volatility`, `route_acceleration` | inst 轴 = V + A |
| `top12_margin` | comm 轴 |
| `token_consensus`, `token_dispersion` | cons 轴 = C − D_tok |
| `mob1_w8` | stick 轴 = −mob1_w8（也是消融臂 C 的唯一输入） |
| `flow_com` | flow 轴 |

**不需要也不读取**：`events.csv`、`success` 列、`n_queries`、任何物理量。
`success` 在检测路径上一次都没有被访问 —— 这是探针 P1 的直接保证（`load_unit()` 连读都不读）。

`alarms` 是**完整序列**，按 `step` 升序；两个通道各自独立、互不抑制；同一通道的带游程
每次（重新）达到阈值都报一条（游程断掉后重新武装）。因此「先误发 loop、后正确发 static」
的集在 `alarms` 里两条都在，评分方不会因为首条报错就判它未检出。

---

## 2. 五轴 Γ_q（继承 E8 §1，未改）

| bit | 轴 | 原始量 | 高位语义 |
|---|---|---|---|
| 0 | `inst` | `late_flow_volatility + route_acceleration` | high-vol |
| 1 | `comm` | `top12_margin` | high-commit |
| 2 | `cons` | `token_consensus − token_dispersion` | high-consensus |
| 3 | `stick` | `− mob1_w8` | sticky |
| 4 | `flow` | `flow_com` | late-flow |

二分点 = **组内无标签中位**（leave-one-episode-out），即 E8 §1 的「组内无标签 z 化」
去掉 MAD 缩放（MAD 是正的，不影响 `x > med` 的符号，只影响 z 的幅度）。
**带的定义（哪几个 bit 组成哪条带）完全继承 E8，未做任何重划。**

> **偏离声明**：任务书原文写「各轴按组内**成功集**中位二分」。成功集中位在被评分的
> 那个文件里是结局标签的函数，会让守门探针 P1（组内 `success` 置换后报警逐位不变）必然失败。
> 因此二分点改为**组内无标签中位**（= E8 原口径，也与任务书对 Γ_q 的「组内无标签 z 化」
> 一致）；**成功集只用来选 K / M / K_C 三个旋钮**（留一组外）。
> 成功集中位口径的数字作为敏感性分析记在 `calib_report.zh.md` §附录，**不交付**。
> 代价是实打实的：无标签中位被失败行拉动，灵敏度下降（详见 calib_report §9.1）。

---

## 3. 参考池（label-free，t-independent）

对每一集 e（所属组 g）：

```
level 'scene': g 组内、q>=QMIN、五轴有限、**去掉 e 自己**的全部行
               准入：集数 >= MIN_REF_EPS 且 行数 >= MIN_REF_ROWS
level 'task' : 退化 —— 全 task 同条件、去掉 e 自己的全部行（行数 >= MIN_REF_ROWS）
level 'none' : 两级都不满足 -> 该集不可评（必须单独上报，不进分子也不进分母）
med_a = median(参考池的轴 a)     a ∈ {inst, comm, cons, stick, flow}
```

- **不使用任何结局标签** ⇒ 探针 P1 通过。
- **去掉本集自身** ⇒ 参考池与 t 无关 ⇒ 探针 P2 通过。
- 校准集上 `ref_level` 100% 是 `scene`（main 512/512、grid 400/400），无退化组。

在线部署时参考池应换成预先采集的同场景 rollout 池；本实现从被评分文件的**同组其它集**
取，是离线评测的等价物。

---

## 4. 带（继承 E8 §3，冻结，未重划）

```
switching/desync 带 = (inst==1) & (cons==0)     # 32 态里的 8 个 —— loop 通道
flatten/shared  带 = (inst==0) & (cons==1)     # 另外 8 个 —— static 通道
sticky 带（消融臂 C）= (stick==1)                # 单轴 mob1_w8，16 个态
state5（仅诊断）     = 五轴宏状态与上一步完全相同   # E8 P_self 的集级对应量
```

一行「有效」当且仅当 `q >= QMIN` 且五轴原始量全部有限。**无效行把驻留计数清零。**

---

## 5. 判定规则（伪代码）

```python
med = reference_median(episode)        # §3，离线、不含本集、不含任何标签
run_sw = run_fl = run_st = 0
for q = 0, 1, 2, ...:                  # 在线，逐 query
    x  = axes_raw(row_q)                                  # 只用第 q 行
    ok = all_finite(x) and q >= QMIN
    bit = {a: x[a] > med[a] for a in axes}

    run_sw = run_sw + 1 if (ok and  bit.inst and not bit.cons) else 0
    run_fl = run_fl + 1 if (ok and not bit.inst and     bit.cons) else 0
    run_st = run_st + 1 if (ok and bit.stick)                     else 0

    if run_sw == K_LOOP:   emit {"step": q, "channel": "loop"}      # 臂 B
    if run_fl == M_STATIC: emit {"step": q, "channel": "static"}    # 臂 B
    if run_st == K_C:      emit {"step": q, "channel": "sticky"}    # 臂 C
```

下游按 channel 选干预（loop → 重规划；static → 重采 flow 噪声）。

---

## 6. 参数表（**已冻结，唯一一组**）

| 参数 | 值 | 语义 | 选定依据 |
|---|---:|---|---|
| `QMIN` | 8 | 分析窗下界 | 继承 E8：`stick` 轴的 `mob1_w8` 需 8 步拖尾。**不是为本方案调的** |
| `MIN_REF_EPS` | 3 | scene 级参考池最少集数 | 中位数在 n=3 才有意义 |
| `MIN_REF_ROWS` | 30 | 参考池最少行数 | ≈ 一条轨迹在窗内的行数（31），保证中位不被单集主导 |
| `FA_TARGET` | 0.10 | **先验运维容忍度**：每 10 条成功 rollout 最多容忍 1 次无谓干预 | **先验成本声明，不从任何数据推导**（既不来自校准集，也不来自留出集）。改这个数只需重跑 `PARAM_SWEEP`，不改架构 |
| **`K_LOOP`** | **11** | switching 带连续驻留步数 → loop 报警 | 两单位**干净成功集**池化 FA 曲线上满足 FA≤`FA_TARGET` 的最小 K。LOGO（组=`corpus\|scene`，66 折）**66/66 折全部 = 11** |
| **`M_STATIC`** | **9** | flatten 带连续驻留步数 → static 报警 | 同规则。LOGO **66/66 折全部 = 9** |
| **`K_C`** | **14** | 消融臂 C：sticky 带连续驻留步数 | 同规则。LOGO **66/66 折全部 = 14** |
| `ARM_T_Q0` | 41 | 平凡对照臂：活过 q=41 即报警 | 同规则（干净成功集 `P(n_queries>41)=0.069`）。**不是交付件**，是必须被打败的对象 |

旋钮规则（冻结）：`K* = min{ K≥1 : FA_干净成功集(K) ≤ FA_TARGET }`，只在**干净成功集**
（无 loop 也无 static 事件的成功集）上算 FA。**失败集标签在参数选取中一次都没出现。**

**旋钮规则的已知缺陷**：FA 曲线是阶梯函数且在选中点前有断崖
（switch：K=10 → 0.119，K=11 → 0.033；flatten：M=8 → 0.135，M=9 → 0.016）。
「≤ 目标」的规则因此**打不中目标**，实际工作点比声明的容忍度严 3–6 倍，
代价是检出率被牺牲（static 通道 M=8→9 使 pre-onset 检出从 0.42/0.61 掉到 0.27/0.48）。
横向对比时**不要用冻结点比，用 `PARAM_SWEEP` 对齐工作点**。

### 6.1 `PARAM_SWEEP`（供执行器把两臂对齐到同一工作点）

`detector.PARAM_SWEEP` 是四条一维单调族，每个元素是可直接传给 `detect(..., params=...)`
的覆盖字典；冻结点是各族中的一个元素：

| key | 族 | 元素数 |
|---|---|---:|
| `"B"` | `{"K_LOOP": i+2, "M_STATIC": i}`，i = 4..25（保持冻结点的 K−M=2 偏移） | 22 |
| `"B_loop"` | `{"K_LOOP": k}`，k = 4..27 | 24 |
| `"B_static"` | `{"M_STATIC": m}`，m = 4..27 | 24 |
| `"C"` | `{"K_C": k}`，k = 4..33 | 30 |

推荐用法：先在**干净成功集**上算每个元素的每集 FA，再把 B 的某通道与 C 对到同一 FA，
最后比 pre-onset 检出 / 提前量。校准集上这样对齐后的配对差见 `paired_BvsC_*.csv`。

---

## 7. 输出契约

见 §1。附便利函数 `alarms_to_frame(alarms) -> DataFrame`（逐集：首次报警步、首个通道、
各通道首次报警步、报警条数、`ref_level`）。**评分方应使用完整 `alarms` 序列，
而不是 `first_step`。**

---

## 8. 已知失效条件（执行留出集前必读）

1. **q≥8 的窗 + K=11 / M=9 ⇒ 最早报警在 q=18 / q=16。** 短套件（goal/object/spatial，
   queries 中位 8–13）上本方案预期近乎全零检出。**这不是 bug，是设计域外**——
   本方案只在 long 类长任务上有意义。
2. **`ref_level='task'` 退化组是污染源**：scene 水平差不再被扣掉。校准集上没有出现
   （100% scene），留出集上若出现必须分层报，不能只报合计。
3. **static 通道锁事件不锁结局**：校准集上 4 个「成功集里的瞬态 static」有 3 个被报警
   （main 3/3、grid 0/1）。它检出的是「停住了」，不是「停住且出不来」。
4. **物理 static onset 自带 +8 query 的确认滞后**（规则 = 宽 2 滑窗 × 连续 7 窗，
   返回第 7 窗 +2）。static 通道 −7 的「提前量」中几乎全部是标签自身的滞后，
   **不得宣称「MoE 比身体先知道」**。loop onset 无此滞后。
5. **loop 通道打不过只看集长的平凡臂 T**（`calib_report.zh.md` §4）。它唯一独有的价值是
   不报恢复性 loop（校准集 0/6）。
6. **绝对 FA / 检出数字只能当上界估计**：校准集里失败集几乎全是超时集
   （`n_queries` 组内配对 AUC = 0.999 / 1.000），任何驻留类量都被集长混杂。
   **载荷主张是同一次运行内 B vs C 的配对差**，不是绝对值。
7. **无标签中位随组内失败比例漂移**：SCENE8 的失败率是 42% / 35%，中位被失败行拉动，
   灵敏度被压低。留出集里失败率更低的组会让同一套参数**更敏感**（FA 与检出同时上升）。
   SCENE8 的数字对低失败率组是**保守**估计，不是保证。

---

## 9. 机械自检

| 探针 | 定义 | main | grid |
|---|---|---|---|
| P1 | 组内 `success` 随机置换（3 次）后 `detect()` 输出逐位不变 | ✔ | ✔ |
| P2a | 截断到首个报警步 s 后仍在 s 报警 | ✔（15 集） | ✔（15 集） |
| P2b | 截断到 s−1 后**完全不报警** | ✔ | ✔ |

装载期硬断言：行按 (episode, control_step) 有序；集内 q 严格 = 0..n−1；缺列即抛。
（P1 在本实现里是结构性成立的：`load_unit()` 根本不读 `success` 列。）

---

## 10. 复现

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd analysis_moe_phenotype/DESIGN_SANDBOX/B_macrostate_dwell
python3 calibrate.py    #  61 s -> summary.json, per_episode.csv, knob_sweep.csv,
                        #          paired_BvsC_*.csv（含 P1/P2 探针）
python3 diagnose.py     #  97 s -> diagnostics.json, matched_fa_*.csv
python3 finalize.py     #   5 s -> summary.json(headline), fig_calibration.png
```

留出集执行：`from detector import detect; detect(<rows.npz>, "scene")`，不改任何参数。
