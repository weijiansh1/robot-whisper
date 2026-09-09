# 二阶失败（适应失败）预注册设计

本文件在看到任何路由数据之前固定：一阶事件定义、landmark、风险集、匹配变量、
模型阶梯、哨兵、零分布和判定阈值。可行性核查结果写在同目录 `feasibility.md`，
描述性结果写在 `token_decomposition.md`。

执行入口（脚本按仓库惯例放在仓库根目录，本目录有同名符号链接）：

```bash
python3 analyze_post_error_adaptation.py --self-test
python3 analyze_post_error_adaptation.py --stage gate       # S4 Gate 0
python3 analyze_post_error_adaptation.py --stage ladder     # S6 模型阶梯
python3 analyze_post_error_adaptation.py --stage tokens     # S8 描述性分解
```

---

## S1 命题与因果边界

现有 MoE 结论的限定：`o_k → R_k, A_k → 执行 chunk k → E_k → o_{k+1} → R_{k+1}`。
若一阶物理事件 `E^(1)_k` 发生在 chunk k 执行过程中，则 `R_{k+1}` 已经看到事件结果，
**任何在 `k+1` 上的路由信号都只能是事后检测，不能预警第一次失败**。本设计不试图
推翻这一点。

被检验的命题只有一条：

> 在**所有样本都已经发生过一阶事件**的风险集内，`k⁺ = k+1` 上的路由计算是否携带
> 关于**二阶失败**（模型收到失败反馈后没有真正更新策略）的增量信息，
> 且该增量在**已经控制了当前物理状态与已发出的 action chunk 之后**仍然存在。

一阶事件 `E^(1)`：物理事件（滑落 / 碰撞 / 空抓 / 提前释放 / 移向错误目标）。
二阶事件 `E^(2)_{k+1}`：适应失败，即 `R_{k+1} ≈ R_j` 且 `A_{k+1} ≈ A_j`
（`j` 是本 rollout 内更早、导致相似失败的 query）。

**不做因果宣称。** 本机没有 mujoco / robosuite / libero / robomimic，
任何从状态 k 重放或干预的实验（expert swap、route clamp、snapshot replay、
可恢复性 `Q_k(h)`）都不可执行，本设计不包含它们。

### 时序口径（已核实）

- `sim_state[k]` 记录于 chunk k **执行之前**（`A_k` 指令位移与 `s[k+1]−s[k]` 相关 0.977）。
- 因此 landmark `onset = k` 表示"事件在缓存中第一次可观测的 query"；
  **造成事件的 chunk 是 `onset−1`，事件时刻本身带 ±1 chunk 的定位不确定性**。
- 每个 control step 含 10 个动作子步，只记录步末位姿；chunk 内没有稠密状态，
  也没有 RGB / contact / force。因此所有一阶事件都是**运动学代理**，
  不能写成 collision / true grasp / true drop / visual confusion。

---

## S2 一阶事件（landmark）定义

全部使用**绝对阈值**，不使用"超过成功 q95"这类 success-normalized 阈值。
理由：success-normalized 阈值在构造上就把成功轨迹排除在事件之外，
必然让 `Y=0` 类为空，从而使二阶设计不可证伪。

对每个任务目标物 `o`（成功轨迹中平均位移 > 3 cm 的自由关节），令

- `z_rel(t)` = 物体高度相对自身初值
- `gd(t)` = 到"留一 init、留一 seed"的成功终点参考集的最近距离
- `best(t) = min_{s≤t} gd(s)`
- `de(t)` = EEF 与物体中心距离
- `g_cmd(t)` = chunk 内夹爪指令均值（>0 为闭合）

| 族 | 定义（onset = 满足条件的 query k） | 阈值网格 |
|---|---|---|
| `liftloss` | `z_rel` 从 >1 cm 落回 ≤1 cm，且 `gd(k) > 5 cm`，落差 ≥ θ | θ ∈ {0, 0.01} |
| `heightloss` | `g_cmd(k−1)>0`、`z_rel(k−1)>1 cm`、`z_rel(k−1)−z_rel(k) ≥ θ`、`gd(k)>5 cm` | θ ∈ {0.02, 0.03, 0.05} |
| `drop_separation` | 同上，且 `de(k) > de(k−1) + 5 mm`（夹爪与物体分离） | θ ∈ {0.01, 0.02, 0.03} |
| `goal_regression` | `gd(k) − best(k) ≥ θ` 的首次上穿 | θ ∈ {0.01, 0.02, 0.03, 0.05} |
| `goal_nbr_loss` | 物体曾进入 5 cm 成功邻域，之后离开到 `5 cm + θ` 外 | θ ∈ {0.025, 0.05} |
| `release_offgoal` | `g_cmd` 由闭合翻为张开，且物体仍被抬起且 `gd>5 cm` | — |
| `grasp_fail` | 「`g_cmd>0` 且 `de ≤ θ`」的连续段结束，且该段内 `z_rel` 抬升 < 5 mm | θ ∈ {0.05, 0.08, 0.10} |
| `eef_approach_leave` | EEF 曾进入目标 5 cm，之后离开到 θ 外 | θ ∈ {0.12, 0.15} |
| `push_ungrasped` | 夹爪张开时物体单步位移 ≥ θ | θ ∈ {0.01, 0.02} |

共 22 个 (族, 阈值) 组合 × 5 任务 × 3 种结局定义。

**事件真实性判据 `setback_fraction`**：事件当刻 `gd(k) − best(k−1) > 5 mm` 的比例。
一个真正的一阶失败必须**丢掉已经取得的进展**。`setback_fraction` 低的 landmark
不是错误，只是正常操作动作（例如"下降放置"）。

---

## S3 风险集与结局

### 风险集

`R(landmark) = {每条 rollout 中该 landmark 的第一次 onset}`，
一条 rollout 至多贡献一个样本（统计单位 = rollout；聚类单位 = init state）。

**关键约束（用户明确要求）**：不比较"发生事件 vs 未发生事件"。
所有进入分析的样本都已经发生过一阶事件。

### 结局

| 名称 | 定义 | 说明 |
|---|---|---|
| `Y_terminal` | rollout 最终失败 = 1 | 与 episode 长度结构性混杂：**失败恒定跑满步数上限，成功在成功时终止**。因此 lead time 不可匹配，只能匹配 `onset`（= 已用 query 数 = 剩余预算，因为上限固定）。 |
| `Y_local_h4` / `Y_local_h8` | 事件后 H 个 query 内，`gd` 未回到 `best(k−1) + 0.5·(gd(k)−best(k−1))` = 1 | 固定视界，要求 `onset + H < T`，因此两类样本的后续观测长度**完全相同**，彻底消除 lead-time 混杂。 |

`Y_local` 是本设计首选结局：它直接对应"收到失败反馈后有没有真正更新策略"，
并且不继承 episode 长度。

---

## S4 Gate 0：可行性硬门槛（先于任何路由计算）

对每一个 (task, family, threshold, label) 组合，同时满足以下全部条件才允许进入
S6 的模型阶梯：

| 条件 | 阈值 |
|---|---|
| 事件数 | `n_events ≥ 60` |
| 两类各自样本数 | `min(n_Y0, n_Y1) ≥ 20` |
| 双结局 init 聚类数 | `n_mixed_clusters ≥ 6` |
| 落在双结局聚类内的样本数 | `n_in_mixed ≥ 60` |
| 双结局聚类内两类各自样本数 | `min ≥ 20` |
| 事件率 | `n_events / 512 ≤ 0.60`（否则不是"事件"） |
| 事件真实性 | `setback_fraction ≥ 0.50` |

任一 landmark 通过 → 执行 S6 主分析。
全部不通过 → **Gate 0 判定 FAIL**，主分析不得执行，
只按 S7 跑"降级演示"并明确标注 `DEMONSTRATION_ONLY_GATE_FAILED`。

---

## S5 匹配与控制变量

模型阶梯的 `M_phys` 块**必须**包含以下控制量（共 20 维）：

- **事件位置 / 任务阶段**：`onset`、`onset / cap`（cap = 该任务的 query 上限，先验已知常数）
- **当前物理状态**：EEF xyz、物体 xyz、夹爪开度、夹爪指令
- **进展状态**：`gd(k)`、`best(k−1)`、最近 2 query 的进展差
- **事件严重程度**：`gd(k) − best(k−1)`、`severity_m`
- **是否仍在可恢复区域**：物体高度 `z_rel(k)`、`de(k)`、`gd(k)`
- **近期运动**：EEF 与物体过去 3 query 的位移

`init_state_id` 用作留一折的分组，**不进入特征**；`episode_length`、
`task id`、`seed`、终局标签都**不进入特征**。

lead time 不进入特征，也不可作为匹配变量（见 S3：它由结局决定）。
时间维度的匹配靠 `onset` 完成；`Y_local` 结局额外保证后续观测长度相同。

---

## S6 模型阶梯

统计单位 = rollout，聚类单位 = `init_state_id`，交叉验证 = leave-one-init-out。
`SimpleImputer(median) → RobustScaler → [PCA(8)，仅当维数 > 4√n_train] → LogisticRegression(C=0.5)`，
**全部在训练折内拟合**。只保留双结局 init 聚类内的样本。

| 模型 | 特征块 |
|---|---|
| `sentinel_onset_only` | 仅 `[onset, onset/cap]`（2 维哨兵） |
| `M_phys` | S5 全部物理控制量 |
| `M_phys+route_state` | + state token（后缀 token 0）路由：速度、软速度、熵、top-1 质量、recurrence、32 维层平均 prob |
| `M_phys+route_action` | + action token（后缀 token 1–10）路由：同上 + denoise drift/span + state-action overlap |
| `M_phys+route` | + 两者 |
| `M_phys+action` | + 已发出的 chunk / 机器人 qpos（9 维） |
| `M_joint` | phys + route(state+action) + action |
| `M_joint+hidden` | + `hb_hidden` 在 landmark query 的 state / action token 表示 |

### 主判定

**只有当 `M_phys+route_*` 相对 `M_phys+action` 的 AUC 增量满足：
点估计 > 0 且 init-cluster bootstrap 95% CI 下界 > 0**，
才可以说 routing 读的是"策略响应"而不是重复读物理事件。

保持怀疑：`M_phys+action`（action chunk 本身）很可能更直接更强。
若 `M_phys+action ≥ M_phys+route`，必须如实报告，不得为 MoE 叙事强行解释。

### 哨兵（强制）

`sentinel_onset_only` 与所有模型跑同一管线。
**若哨兵的 AUC 与真实特征块相当（差距 < 0.05 或 CI 重叠且哨兵点估计不低于其一半增量），
则该 landmark 上的全部结论作废。**
本项目已经被这个坑过一次：`rerun-2026-08-27/SESSION_NOTES.md` §C.2 中
1 维 chunk-index 哨兵拿到理论最大值 0.500 / p=0.0005，五个"显著"信息源全部落在其下。

### 零分布与置信区间

- 增量 CI：按 `init_state_id` 聚类的 bootstrap（重抽 init，2000 次），报 2.5%/97.5% 分位。
- 置换零分布：在 init 聚类内置换 `Y`（保持每个 init 的类别构成），2000 次。
- 报告点估计**和** CI，不得只报点估计。

---

## S7 Gate 0 失败时的降级方案

Gate 0 FAIL 时，仍执行两个明确标注为 `DEMONSTRATION_ONLY_GATE_FAILED` 的演示：

1. `setback_landmark`：在 `setback_fraction ≥ 0.5` 且 `event_rate ≤ 0.6` 的候选中，
   取双结局聚类内两类最小样本数最大者。这是"最接近真实一阶失败"的可用 landmark。
2. `max_support_landmark`：取双结局支持最大者（不论是否真实事件）。
   它的作用是展示"支持量大的 landmark 为什么其实不是错误"。

两者的结论都不得写成对命题的检验，只能用于展示管线行为与哨兵表现。

---

## S8 描述性分解：R^state vs R^action（无论 Gate 结果都做）

`server/routes.zarr` 的 token 轴：`11 = 后缀 token`，token 0 = state，token 1–10 = action。
因此 `R^state = [:, :, :, 0, :]`，`R^action = [:, :, :, 1:11, :]`。

用户假设：同一次推理里，state token 的 routing 是"模型看到了什么"（后验），
action token 的 routing 是"模型准备怎么做"（对下一步的前验）。

**S8.1 结构核查（先做）**：state token 的路由是否随 flow denoise 迭代变化。
若不变，则"denoise 轴"对 state token 是退化的，这本身就是对上述假设的直接结构证据。

**S8.2 事件对齐动力学**：在**已确认存在**的 long 任务事件上做（不是本设计新造的 landmark）：
- `active_return`（EEF return-to-pot2-side）39 条，onset 中位 42
- `stagnation_core`（suffix-low-motion stasis proxy）123 条，onset 中位 28
- `success` 296 条，按首次进入 pot1 邻域对齐，仅作行为参照

对每条曲线先减去本 episode `[−10,−7]` 的参考，再除以 pooled SD。
窗口：`lead = [−6,−2]`、`sync = [−1,+1]`、`after = [+2,+6]`。
只有 `|effect| ≥ 0.2` 且 init-bootstrap CI 不跨 0 才记为 separation。
比较 state 与 action 两条线**哪一条更早分开**。

**S8.3 denoise 轴**：报告 action token 各 denoise 迭代相对迭代 0 的 top-4 overlap、
prob L1 距离、熵，以及各迭代上的判别力。

⚠ **口径更正（必须写进报告）**：`analysis/moe-rollout-trend/report.md` 中
"routed-full AUC 从 k0=0.463 涨到 k8=0.713"的 `k` 是 **query / control step 索引**
（原文："Queries k0..k8 are evaluated separately"），**不是 flow denoise 迭代**。
本节的 denoise 轴是另一条轴，两者不能互相引用。

**S8 只做描述，不做因果宣称。**

---

## S9 已知限制（写进任何报告）

1. 全部一阶事件都是 query 边界上的运动学代理，缓存没有 RGB / contact / force /
   chunk 内稠密状态。
2. 成功终点参考由成功 rollout 构造。已做"留一 init、留一 seed"，但仍是
   transductive 的弱形式；真正干净的做法是使用环境真值 goal。
3. 终局标签与 episode 长度结构性混杂（失败恒定跑满 cap）。
4. `hb_hidden` 是 router 输入的上下文 hidden state，不是 expert 贡献。
5. 任何"策略回返 / MoE trap / 视觉误认"的措辞都不被数据支持，必须保留 proxy 字样。
