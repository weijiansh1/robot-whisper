# 留出集执行器：接口契约与运行手册

> 配套：`GATE/HOLDOUT_PROTOCOL.md`（判据来源）、`GATE/run_holdout.py`（实现）。
> 本文件定义**三个原型 agent 的产物必须满足的接口**，使同一个执行器能消费 A / B / D 三套 `detector.py`。
> 接口在 SPEC 冻结前有效；冻结后接口本身也进入 `HOLDOUT_PROTOCOL.md` §7.1 的禁改清单。

---

## 1. 目录与模块契约

每个原型目录必须是可 import 的模块目录：

```
DESIGN_SANDBOX/<arm_dir>/
├── SPEC.md          # 冻结规格（front-matter 见 §1.2）
├── detector.py      # 唯一入口，执行器只 import 这一个文件
├── calib_report.zh.md
└── summary.json
```

`<arm_dir>` ∈ `{A_threshold_fsm, B_macrostate_dwell, D_freeform}`。

### 1.1 `detector.py` 必须导出的符号

```python
SPEC_ID: str            # 与 SPEC.md front-matter 的 spec_id 逐字相等，例 "A_threshold_fsm@v1"
ARM: str                # "A" | "B" | "D"
WARMUP: int             # 最早允许发射报警的 control_step（含）；执行器用 P6 校验
REQUIRED_COLUMNS: list[str]   # 会从 rows.npz 读取的列名（含 "success" 若用于建参照库）
GROUP_COL: str          # 固定为 "scene"

PARAMS: dict            # 主臂冻结参数表（一组，不留待定）
PARAMS_C: dict          # 消融臂 C 冻结参数表：同一状态机，一级信号换成 mob1_w8
ABLATION_KEYS: list[str]  # 声明 PARAMS 与 PARAMS_C 允许不同的键；执行器用 P5 强制

PARAM_SWEEP: list[dict] | None    # 可选，≤5 个「完全写死」的参数字典，必须包含 PARAMS 本身
PARAM_SWEEP_C: list[dict] | None  # 同上，对应 C

def detect(rows_npz_path: str, group_col: str = "scene",
           params: dict | None = None) -> dict[int, dict]:
    """params=None 等价于 PARAMS。返回 {episode_id: {...}}，见 §2。"""
```

### 1.2 `SPEC.md` front-matter（YAML，执行器解析）

```yaml
---
spec_id: A_threshold_fsm@v1
arm: A
frozen_at: 2026-09-04T12:00:00Z
warmup: 8
ablation_keys: [primary_signal]
param_sweep: false
---
```

`spec_id` 与 `detector.py::SPEC_ID` 不一致 ⇒ 执行器拒跑该臂。

---

## 2. `detect()` 的输入 / 输出

### 输入

| 参数 | 值 | 说明 |
|---|---|---|
| `rows_npz_path` | `features/<corpus>/<suite>/<task>/rows.npz` 的绝对路径 | **单任务全部集的全部行**，含 `episode_id, control_step, scene, repeat, success, flow_noise_seed` 与全部 z_q 标量（`mob_k` 是 (N,8)、`flow_profile` 是 (N,9)，其余 (N,)） |
| `group_col` | `"scene"` | 组 = init_state；`HOLDOUT_PROTOCOL.md` §4.4 |
| `params` | `PARAMS` 或 `PARAMS_C` 或 sweep 中的一项 | 执行器显式传入；`detect` 不得读全局默认以外的任何配置文件 |

**`events_csv_path` 不在接口里，这是有意的。**
`CONSTRAINTS.md` §交付 给出的签名就是 `detect(rows_npz_path, group_col)`，本契约与之一致。
事件表含 `success`、`loop_onset_q`、`static_onset_q`、`trap_onset_q` —— 把它交给检测器等于
把结局标签和物理 onset 一起递过去。**执行器持有事件表并独自完成评分，检测器永远看不到它。**
（若某原型已按 `detect(rows_npz, events_csv, group_col)` 实现，执行器会按签名反射调用并传 `events_csv_path=None`，
同时在 `probes.json` 记 `WARN_SIGNATURE`；`detect` 必须能在该参数为 `None` 时正常工作。）

### 输出

```python
{
  episode_id (int): {
     "alarm_step": int | None,       # 首次报警的 control_step；None = 全程无报警
     "channel":    "loop" | "static" | None,   # alarm_step 非 None 时必须非 None
     "score":      float | None,     # 可选，报警时刻的分数，仅供描述性排序
     "alarms":     [ {"step": int, "channel": str, "score": float}, ... ]  # 可选，全部报警
  },
  ...
}
```

- **必须为 rows.npz 中出现的每一个 `episode_id` 返回一项**（无报警也要给 `alarm_step=None`）。执行器用 P7 校验。
- `alarm_step` 必须满足 `WARMUP ≤ alarm_step ≤ n_queries − 1`，**且必须是集内 query 序号**（见 §2.1）。
  执行器用 P6 + P8 校验。

### 2.1 时间坐标陷阱（2026-09-04 发现，**必读**）

> **`rows.npz` 的 `control_step` 是整任务的全局行号（0..N−1），不是集内 query 序号。**
> 只有 `episode_id == 0` 两者巧合相等。

实测（全部 45 任务）：`grid50x8/libero_long/LIVING_ROOM_SCENE1` 的 `control_step` 最大值是
**10,782**，而该任务集长中位只有 25；`episode 5` 的 `control_step` 是 `125..149`，其集内 q 是 `0..24`。

而 `events.csv` 的 `loop_onset_q` / `static_onset_q` / `n_queries` **全部是集内坐标**。
所以：**把 `control_step` 当时间轴 = 全部时间比较、预热判定、q 相关逻辑在 episode 0 之外统统错位。**

**规范换算**（已在 45/45 任务验证：集内 `control_step` 严格 +1 连续、逐集行数 == `n_queries`）：

```python
q = control_step - control_step[episode_id == e].min()      # 集内 query 序号 ∈ [0, n_queries-1]
# 等价写法：q = rank of the row within its episode, ordered by control_step
```

**三条要求**：
1. `detect()` 返回的 `alarm_step`（以及 `alarms[].step`）**必须是 q**。
2. `WARMUP` 的语义是"集内已走了几步"，比较对象也是 q。
3. 任何"随 q 漂移"的归一化/参照库（例如按绝对 query 位置取分位）必须用 q，不是 `control_step`。

执行器的 **P8** 会用 `n_queries`（= 逐集行数）逐集核对，命中即硬失败并直接指出这个陷阱。
**P2（前缀截断）也会在这种实现上失败**，因为截断阈值是按 q 换算的。
- 双通道检测器若在同一集先后发出 loop 与 static 报警：`alarm_step`/`channel` 取**最早那次**，
  完整序列放 `alarms`。执行器在通道匹配时会优先用 `alarms`（若提供），否则只用首次报警。
  **强烈建议提供 `alarms`**：只给首次报警会让"先误发一个 loop、后正确发 static"的集被判为未检出。

---

## 3. `detect()` 必须遵守的行为约束

| 编号 | 约束 | 执行器如何强制 |
|---|---|---|
| C-1 | **只读 `rows_npz_path` 一个文件**。不得打开 events/、不得读其他任务、不得读网络 | 探针 P4（I/O 白名单）+ P4b（源码静态扫描） |
| C-2 | **在线因果**：某集在 `alarm_step = s` 的报警，必须能由该集 `control_step ≤ s` 的行 + 参照库重现 | 探针 P2（前缀截断） |
| C-3 | **参照库留一组外**：给组 `g` 的集打分时，统计量只能来自 `group != g` 的**成功**集 | 探针 P1（组内标签置换） |
| C-4 | **确定性**：同输入两次调用逐位相同；不得依赖未固定的 RNG、不得依赖字典遍历序 | 探针 P3 |
| C-5 | **不训练**：无梯度、无 PCA/k-means/probe/分类器拟合。只允许分位点、中位/MAD、计数、状态机 | SPEC 审查（人工）+ P4b 源码扫描 `sklearn|torch|fit(|lstsq|eig` |
| C-6 | **臂 C 同口径**：C 与主臂走同一个 `detect` 函数、同一套参数键，只有 `ABLATION_KEYS` 中的键不同 | 探针 P5 |

### 3.1 为什么 P1 / P2 是这样设计的

- **P1**：把某组 `g` 内各集的 episode 级 `success` 标签随机置换后重跑。
  若检测器守 LOGO，则 `g` 的参照来自 `g` 之外，**`g` 内各集的报警必须逐位不变**（组外可以变，那是合法的）。
  变了 ⇒ 用了组内（乃至本集）结局标签 ⇒ 违反 `CONSTRAINTS.md` §硬约束 1。
  若某检测器是 LOEO（留一集外）而非 LOGO，P1 同样会抓到——这也是违规。
- **P2**：对一个已报警集 `e`（`alarm_step = s`）：
  - 截断 `e` 到 `control_step ≤ s` 重跑 ⇒ `e` 的 `alarm_step` 必须仍是 `s`，`channel` 不变；
  - 截断到 `≤ s−1` 重跑 ⇒ `e` 必须**完全无报警**（若还有报警，说明它本可以更早报，与"首次报警在 s"矛盾）。
  - 截断 `e` 会改变**其他组**的参照（若 `e` 是成功集），但按 LOGO 不会改变 `e` **自己**的参照，
    所以只断言 `e` 自己的结果，其他集允许变化。

---

## 4. 探针清单（`--preflight` 阶段执行，任一失败 ⇒ 该臂不得计分）

| 探针 | 检查 | 抽样规模 | 失败后果 |
|---|---|---|---|
| **P0** | `SPEC_ID` == SPEC.md `spec_id`；必需符号齐全；`WARMUP ≥ 0` | 全部 | 硬失败 |
| **P1** | 组内 `success` 置换 ⇒ 组内报警不变 | 3 任务 × 2 个混合标签组 | 硬失败 |
| **P2** | 前缀截断 ⇒ `s` 处仍报 / `s−1` 处不报 | 3 任务 × 5 集 | 硬失败 |
| **P3** | 同输入两次调用逐位相同 | 2 任务 | 硬失败 |
| **P4** | I/O 白名单（monkeypatch `open`/`np.load`/`Path.open`/`pd.read_csv`） | 1 小任务 | 硬失败 |
| **P4b** | 源码静态扫描：`events`、`onset`、`trap_`、`summaries`、`sklearn`、`torch`、`\.fit\(` 字面量 | 全部 | 命中 ⇒ WARN + 转人工（SPEC 审查） |
| **P5** | `PARAMS.keys() == PARAMS_C.keys()`；差异键 ⊆ `ABLATION_KEYS` ≠ ∅；同一 `detect` 函数对象 | 全部 | 硬失败 |
| **P6** | 全部 `alarm_step ≥ WARMUP` 且 `≤ n_queries−1` | 全量（评分时） | 硬失败 |
| **P7** | 返回键集 == rows.npz 的 `episode_id` 集；`channel` 与 `alarm_step` 同为 None 或同非 None | 全量 | 硬失败 |

**预检在校准集单位（SCENE8）上跑**，不是在留出集上跑——预检本身不产生任何留出集指标。
预检不通过就地中止，避免污染唯一一次留出集运行。

---

## 5. 运行流程（三阶段，No-peek）

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/jovyan/work/himoe-vla/analysis_moe_phenotype

# 阶段 0：预检（校准集，可反复跑）
python3 DESIGN_SANDBOX/GATE/run_holdout.py --preflight \
    --arms A_threshold_fsm B_macrostate_dwell D_freeform

# 阶段 1：留出集执行 —— 只产出 alarm 明细，不打印任何指标
#   要求三套 SPEC 全部冻结；四臂（A/B/D + 各自的 C）一次跑完
python3 DESIGN_SANDBOX/GATE/run_holdout.py --run \
    --arms A_threshold_fsm B_macrostate_dwell D_freeform \
    --confirm-all-specs-frozen

# 阶段 2：评分 —— 一次性写出全部指标表与判定书
python3 DESIGN_SANDBOX/GATE/run_holdout.py --score --all-arms-present
```

**阶段 1 的执行器在任何情况下都不计算、不打印率值**（源码里 `--run` 分支不 import 评分函数）。
这不是形式主义：留出集只允许跑一次，如果 `--run` 顺手打印了 Det，
"再改一版 SPEC 重跑"的诱惑就有了实际抓手。

**消融臂 C 不是独立目录**：执行器对每个 `<arm_dir>` 调 `detect(..., params=PARAMS)` 得到主臂，
调 `detect(..., params=PARAMS_C)` 得到该臂的 C。因此结果目录是
`results/A_threshold_fsm/`、`results/A_threshold_fsm__C/`，依此类推。

> 注意：三个原型各有一个 C。若三个 C 的实现口径不同（不同状态机结构包同一个 `mob1_w8`），
> **它们不是同一条基线**。`HOLDOUT_PROTOCOL.md` §4.1 的对比是**臂内配对**（A vs C_A、B vs C_B、D vs C_D），
> 这正是"同一套状态机结构只换一级信号"的原意，不需要三个 C 相等。
> 但三个 C 的 Det/FA 会一并披露；若差异巨大，说明"打赢 C"这件事在很大程度上取决于状态机外壳，
> 报告须点出这一点。

---

## 6. 输出格式

### `results/<arm>/alarms/<corpus>__<suite>__<task>.csv`

```
episode_id,alarm_step,channel,score,n_alarms
0,-1,,,0
1,17,loop,2.31,3
```
`alarm_step = -1` 表示无报警（CSV 里不写 None）。

### `results/metrics_primary.csv` — 每臂（含各自的 C 与 T）一行

**AMD-2 主端点列**：`MH_loop, MH_loop_sign, MH_loop_strata, MH_static, MH_static_sign,
MH_macro, prefix_n, prefix_det_rate, Type_TypeAcc_macro, MH_admissible`
**附录 A 列**（受集长混杂支配，不可作载荷主张）：`det_n, det_k, det_rate, fa_n, fa_k, fa_rate,
fa_ci_lo, fa_ci_hi, fa_worst_cell, fa_worst_cell_rate, lead50, lead_q25, lead_q75,
mistype_rate, S7_n, S7_det_rate, gate_G1, gate_G2, gate_G3_appendixA`

评分时会打印**分母自检**，三个数必须完全吻合，否则中止：
`有锚致败集 = 332`、`干净成功集 = 15591`、`含有锚致败集的组 = 180`。
匹配视界端点的自检量（应当吻合）：`MH_loop_strata = 95`、`prefix_n = 97`，
逐视界（p10..p90）的 (层数, 事件, 干净成功) =
`(100,172,556) (83,140,451) (80,136,437) (79,135,413) (70,97,317) (51,69,202) (45,62,179) (36,48,134) (24,36,67)`。

### `results/mh_validity.json` — **V 闸（run-voiding）**

臂 T 的逐视界 `margin_T(q)`。理论值**恒等 0**（`HOLDOUT_PROTOCOL` §3A.2 构造性论证）。
`|MH(T)| > 0.02` 或 >2 个视界超容差 ⇒ **匹配视界实现有误，整次运行作废重跑**。
自测参考：合规实现下 9/9 个视界 `margin_T = 0.0`（逐位精确）。

### `results/metrics_secondary.csv` — 每臂一行

`S1_fatal_loop_rate/n, S1_recovery_loop_rate/n, S1_delta`（主口径，排除 grid/object：243 vs 97）、
`S1_delta_all_cells`（全集口径 272 vs 251）、`S2_unanchored_fail_rate/n`（153）、
`S3_loop_recall`（272）、`S3_static_recall/n`（105）、
`S4_uncertain_loop_alarm_rate`（3 个 loop 无效任务的成功集，分母 1,306）。

### `results/arbitration.json` — 仲裁结果

`contrasts[]`（每个原型一项：`delta_pooled, delta_group, p_maxT_F1, disc_rate,
R1_grid, R1_main, R2_cells_positive, W1..W5, beats_C`）、`winner`、
`verdict`（§4.3 六条 case 逐条布尔 + `headline` + `mandatory_notes`）、`seed`、`nperm`、`family_F1`。

### `results/RESULTS.md`

**AMD-2 追加必含项**（列在最前）：
0. `mh_validity.json` 的 V 闸结论；不过则整份报告只写"运行作废"；
0b. 主端点表：三臂的 `MH_loop`（+ 符号 k/9）、`MH_loop − MH_loop(C)`、`p(F1_M)`、`p(F1_MC)`；
0c. 第二证据线：`Det_prefix`（n=97）三臂对比；
0d. 分型端点：`TypeAcc_macro` 与常数基线 0.500（**禁止出现微平均数字**）；
0e. 附录 A 全表，每处引用附"受集长混杂支配、不可作载荷主张"。

以下为原有必含项（AMD-1 及之前）：
1. §4.3 六条负结果出口的**逐条勾选表**（每条写"命中/未命中"）；
2. 四臂（含各自 C）的完整指标行（`HOLDOUT_PROTOCOL.md` §6 R-D 全员披露）；
3. 第二名的数值 + "argmax over 3 存在效应量向上偏倚"这句话；
4. 若 `|FA(arm)−FA(C)| > 0.01` ⇒ §4.1 那段逐字警示；
5. 若 `Lead50 ∈ [0,+2]` ⇒ 标"同步型"并声明全文禁用"提前/预警"；
6. 若未过 S1 ⇒ §5.1 那句逐字声明；
7. 观测到的 `disc_rate` 与事后功效说明。

---

## 7. 给三个原型 agent 的 checklist（照此写就能被消费）

- [ ] `detector.py` 导出 §1.1 的全部符号，`SPEC_ID` 与 SPEC.md front-matter 一致
- [ ] `detect(rows_npz_path, group_col="scene", params=None)` 签名正确，`params` 生效
- [ ] 返回值覆盖**每一个** episode_id，`alarm_step=None` 表无报警，`channel` 同步为 None
- [ ] 提供 `alarms` 全序列（强烈建议）
- [ ] `PARAMS` / `PARAMS_C` 键集完全相同，差异键在 `ABLATION_KEYS` 里，且 `ABLATION_KEYS` 非空
- [ ] 只打开 `rows_npz_path`；源码里没有 `events` / `onset` / `trap_` 字样
- [ ] `success` 只用于建参照库，且严格 LOGO（组 = `scene`）
- [ ] 因果：任何报警都能由该集前缀重现；有明确的 `WARMUP`
- [ ] 确定性：没有未固定 seed 的随机、没有集合/字典遍历序依赖
- [ ] `SPEC.md` 里没有"待定 / TBD / 留出集上再定 / 视情况"这类字样（任务 3 会逐条查）
