# 可行性报告：二阶失败（适应失败）实验在现有缓存上做不了

## 判定

**Gate 0 FAIL。219 个 (task × 一阶事件族 × 阈值 × 结局定义) 组合中，0 个通过。**

不是功效不足的边缘问题，而是**结构性的**：在这份缓存里，
「真正的一阶失败」与「后来恢复成功」这两个条件**几乎互斥**。
所有能给出双结局风险集的 landmark，都不是错误。

生成命令：`python3 analyze_post_error_adaptation.py --stage gate`
产物：`landmark_inventory.csv`（219 行全表）、`landmark_events.csv`（逐事件）、`gate_decision.json`。

---

## 1 先核实已有盘点

### 1.1 `analysis/post-error-recovery-hub`

`event_inventory.csv` 33 行，逐条核对：

| task | 事件 | 终局成功 | 终局失败 |
|---|---:|---:|---:|
| `open_the_top_drawer_and_put_the_bowl_inside` | 22 | 0 | 22 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 2 | 0 | 2 |
| 两个 spatial 任务合计 | 9 | 2 | 7 |
| **合计** | **33** | **2** | **31** |

`report.md` 的表（ramekin 2 → 1/1、stove 7 → 1/6）与之一致。**该盘点属实。**
`future_label` 只有 `terminal_failure_after_proxy` 31 / `terminal_success_after_proxy` 2。

### 1.2 `audit_post_error_dense_replays.py` 的稠密回放

`dense_replay_audit.md`：7 条 purposive 审计，5 条 contact-confirmed loss，
其中 **1 条终局成功（stove ep 3，确认 regrasp）、4 条终局失败**；2 条因为
「drop 前没有稳定双指接触」被否决。磁盘上只剩 4 个 task 目录、共 5 个 episode 的
`events.npz`。

**本机 `mujoco` / `robosuite` / `libero` / `robomimic` 全部 ModuleNotFoundError**
（已实测）。那 5 条 dense replay 是上一轮会话留下的缓存产物，**无法扩充**。
因此 contact 级真值事件的总量就是 5 条，其中恢复成功 1 条。任何检验都不可能。

---

## 2 穷举搜索：22 个一阶事件族 × 阈值

全部使用**绝对阈值**（不用 success-normalized 阈值，否则成功轨迹在构造上就不会
被标成事件，`Y=0` 必然为空）。定义见 `PREREG.md` S2。

新增的判据是 **`setback_fraction`**：事件当刻 `gd(k) − best(k−1) > 5 mm` 的比例，
即「这个事件真的丢掉了已经取得的进展吗」。这一列是本次可行性核查最重要的发现。

### 2.1 主表（每个事件族取双结局支持最好的一行）

`bal = min(双结局 init 聚类内 Y0 数, Y1 数)`，是模型阶梯真正可用的样本量。

| 事件族 | task | 阈值 | 结局 | n | 事件率 | **setback** | Y0 | Y1 | mixed init | bal | onset Y0/Y1 | lead Y0/Y1 | 哨兵 AUC |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| `push_ungrasped` | long | 0.01 | terminal | 512 | 1.000 | **0.002** | 296 | 216 | 13 | **184** | 19/19 | 19/32 | 0.461 |
| `heightloss` | long | 0.02 | terminal | 267 | 0.522 | **0.000** | 169 | 98 | 11 | **98** | 18/18 | 20/33 | 0.599 |
| `heightloss` | long | 0.03 | terminal | 143 | 0.279 | **0.000** | 79 | 64 | 8 | **54** | 18/18 | 20/33 | 0.591 |
| `grasp_fail` | long | 0.10 | local h4 | 49 | 0.096 | 0.531 | 15 | 34 | 4 | 14 | 35/36 | 16/15 | 0.481 |
| `goal_regression` | long | 0.01 | local h4 | 36 | 0.070 | **1.000** | 14 | 22 | 6 | 9 | 41/40 | 5/11 | 0.378 |
| `goal_regression` | long | 0.01 | terminal | 47 | 0.092 | **1.000** | 9 | 38 | 4 | 7 | 35/46 | 7/5 | 0.889 |
| `drop_separation` | long | 0.02 | terminal | 14 | 0.027 | 0.000 | 9 | 5 | 3 | 4 | 18/34 | 21/17 | 0.589 |
| `liftloss` | long | 0.01 | local h8 | 9 | 0.018 | 0.667 | 3 | 6 | 2 | 2 | 34/40 | 17/10 | 0.833 |
| `release_offgoal` | top_drawer | — | local h4 | 17 | 0.033 | 0.176 | 1 | 16 | 1 | 1 | 18/18 | 11/11 | 0.375 |
| `eef_approach_leave` | stove | 0.12 | terminal | 16 | 0.031 | 0.000 | 1 | 15 | 1 | 1 | 7/7 | 8/14 | 0.633 |
| `goal_nbr_loss` | long | 0.025 | terminal | 7 | 0.014 | 1.000 | 0 | 7 | 0 | 0 | —/41 | —/10 | — |

### 2.2 表里唯一的规律

**`setback_fraction` 与 `bal` 严格反相关。**

- `setback_fraction ≥ 0.5`（真的丢进展）的组合共 18 个，`bal` 最大值 **14**（`grasp_fail@0.10`），
  第二 9，其余 ≤ 7。
- `bal ≥ 50` 的组合共 4 个，`setback_fraction` 全部 ≤ **0.002**。

也就是说：**样本够的 landmark 不是错误，是错误的 landmark 样本不够。**

### 2.3 「大样本 landmark 其实不是错误」的直接证据

用固定视界的局部恢复结局做检验（`Y_local`，见 `PREREG.md` S3）：

- long/SCENE8 的 `heightloss@0.02`：260 个满足 `onset+4 < T` 的事件中，
  **260/260 在 4 个 query 内目标物的 goal 距离就回到事件前最好水平**，
  且 **0/260 在此后重新抬起**。
- 这不是「摔了之后恢复」，这是**正常的下降放置动作**：物体在被放下，
  距离目标还 > 5 cm 只是因为它正从上方接近。
- `push_ungrasped` 事件率 1.000（512/512），`setback_fraction` 0.002。它等价于
  「物体动过」，不是事件。

因此 `analysis/failure-behavior-taxonomy` 里 `regrasp_or_drop` 在成功轨迹上的
9 条、`heightloss` 在成功轨迹上的上百条，都不能当作「成功 rollout 里的挫折事件」。
**用户要求专门去成功轨迹里找的那一类事件，在这份缓存里检索不到足够数量。**

---

## 3 为什么会这样：三条结构性原因

### 3.1 真实的一阶失败在这个策略下几乎不可逆

在任务内（不跨任务），真正的 setback landmark 与终局几乎一一对应：

| landmark | goal/top_drawer | long/SCENE8 | spatial/stove | spatial/ramekin |
|---|---|---|---|---|
| `goal_regression ≥ 2 cm` | 0 成功 / 40 失败 | 1 / 13 | 39 / 0 | 1 / 1 |
| `goal_regression ≥ 3 cm` | 0 / 39 | 1 / 10 | 16 / 0 | 0 / 0 |
| `goal_regression ≥ 5 cm` | 0 / 39 | 0 / 9 | 0 / 0 | 0 / 0 |
| `liftloss` | 0 / 40 | 0 / 10 | 1 / 6 | 2 / 1 |
| `release_offgoal` | 0 / 17 | 0 / 0 | 0 / 0 | 1 / 0 |
| `goal_nbr_loss` | 0 / 0 | 0 / 7 | 0 / 0 | 0 / 0 |

`Y=0` 类要么为空，要么是个位数。跨任务合并没用：off-goal lift-loss 的 33 条成功
里 **32 条来自 ramekin、1 条来自 stove**，而 59 条失败集中在 top_drawer / long。
合并后分类器学到的是 task，不是适应。30 个 (task, init) 聚类里只有 3 个是双结局。

### 3.2 重抓风险集太小

「上一次抓取尝试失败 → 下一次尝试成功吗」是最贴合用户 E^(2) 定义的设计。
把「夹爪指令闭合且 EEF 在目标 R 内」的连续段当作一次尝试：

| R | 全部尝试 | 风险集（本次没抬起且后面还有一次尝试） | 恢复 | 未恢复 | 双结局聚类 |
|---:|---:|---:|---:|---:|---:|
| 0.05 m | 980 | 21 | 19 | 2 | 1 |
| 0.08 m | 2413 | 3 | 2 | 1 | 1 |
| 0.10 m | 2509 | 20 | 13 | 7 | 0 |

一次抓取失败后**再来一次**的行为在这 2560 条 rollout 里几乎不存在（≤ 21 例）。
这本身是关于这个策略的一条实质结论：**它基本不重试**。

### 3.3 终局标签与 episode 长度结构性混杂

| task | 成功长度 中位/范围 | 失败长度 |
|---|---|---|
| goal/top_drawer | 19 (17–21) | 30 恒定 |
| long/SCENE8 | 39 (35–52) | 52 恒定 |
| spatial/ramekin | 10 (9–22) | 22 恒定 |
| spatial/stove | 12 (11–20) | 22 恒定 |

**失败恒定跑满步数上限，成功在成功时终止。** 因此 lead time 由结局决定，
不可能匹配（这正是上一轮 stasis 中位 2 query vs return 中位 15 query 的病根）。
本设计的应对是：(a) `onset` 进 `M_phys` 作任务阶段控制且同时作 1 维哨兵；
(b) 首选固定视界的 `Y_local` 结局，两类样本后续观测长度完全相同。
但 `Y_local` 与终局高度重合（例如 `grasp_fail@0.05` pooled：
`P(终局成功 | Y_local=0) = 0.93`，`P(终局成功 | Y_local=1) = 0.00`），
所以它并没有把二阶从一阶里真正分离出来。

---

## 4 降级演示（`--stage ladder`，明确标注 `DEMONSTRATION_ONLY_GATE_FAILED`）

按 `PREREG.md` S7 跑了两个 landmark。**两者的结论都不构成对命题的检验。**

### 4.1 `setback_landmark`：long/SCENE8 `grasp_fail@0.10`，`Y_local_h4`，n=34（4 个 init）

| 模型 | 维数 | OOF AUC | 95% CI |
|---|---:|---:|---|
| `sentinel_onset_only` | 2 | 0.438 | [0.186, 0.629] |
| `M_phys` | 20 | 0.657 | [0.420, 0.897] |
| `M_phys+route_state` | 57 | 0.561 | [0.488, 0.750] |
| `M_phys+route_action` | 60 | 0.696 | [0.589, 0.825] |
| `M_phys+route` | 97 | 0.532 | [0.434, 0.725] |
| `M_phys+action` | 29 | 0.689 | [0.600, 0.778] |
| `M_joint` | 106 | 0.521 | [0.429, 0.722] |
| `M_joint+hidden` | 16490 | 0.625 | [0.562, 0.756] |

增量（相对 `M_phys+action`）：

| 对比 | Δ | 95% CI |
|---|---:|---|
| `route_state` | −0.129 | [−0.179, +0.008] |
| `route_action` | **+0.007** | [−0.074, +0.188] |
| `route(state+action)` | −0.157 | [−0.237, −0.021] |
| `joint` | −0.168 | [−0.229, −0.042] |
| `joint+hidden − joint` | +0.104 | [−0.108, +0.267] |

**读法**：34 个样本、4 个 init 聚类，AUC 的 CI 宽度 ≈ 0.24–0.48。
`route_action` 相对 `action chunk` 的增量点估计 +0.007，CI 跨 0。
**routing 在 action chunk 之外没有可检测的增量**；这与
`analysis/failure-moe-signatures` 的 −0.020 [−0.114, +0.049] 方向一致。
但样本量决定了它连「无增量」都算不上稳健证据，只能说该协议下未见增量。

### 4.2 `max_support_landmark`：long/SCENE8 `push_ungrasped@0.01`，`Y_terminal`，n=416（13 个 init）

| 模型 | OOF AUC | 95% CI |
|---|---:|---|
| `sentinel_onset_only` | **0.195** | [0.142, 0.286] |
| `M_phys` | 0.560 | [0.409, 0.712] |
| `M_phys+route_state` | 0.503 | [0.331, 0.651] |
| `M_phys+route_action` | 0.481 | [0.319, 0.624] |
| `M_phys+action` | 0.479 | [0.320, 0.664] |
| `M_joint` | 0.483 | [0.283, 0.693] |
| `M_joint+hidden` | 0.561 | [0.396, 0.724] |

**哨兵结论**：2 维哨兵（`onset`、`onset/cap`）拿到 AUC 0.195，
即 `|AUC − 0.5| = 0.305`；所有实质特征块的 `|AUC − 0.5| ≤ 0.061`。
哨兵携带的秩信息比任何真实信息源都多**五倍**，而且在留一 init 上系统性反号
（说明 onset→结局的关系在 init 之间翻转）。这是时间混杂的教科书签名，
与 `rerun-2026-08-27/SESSION_NOTES.md` §C.2 中 1 维 chunk-index 哨兵拿到理论
最大值 0.500 是同一个坑。**该 landmark 上的一切结论作废。**

---

## 5 明确判定：哪些 landmark 支持 / 不支持二阶实验

| landmark | 支持？ | 理由 |
|---|---|---|
| transport-loss / `liftloss` / `drop_separation` | **不支持** | 真实事件，但成功侧 ≤ 3 条；跨任务合并后 Y=0 等于 task 标签 |
| `goal_regression`（≥1 cm，任一阈值） | **不支持** | `setback_fraction = 1.0`，是唯一无争议的一阶失败，但任务内 Y=0 为 0–9 条 |
| `goal_nbr_loss` / `subtask_undo` | **不支持** | long 7 条，全部失败，双结局聚类 0 |
| `release_offgoal` | **不支持** | 17 条，16 条失败，落在 1 个 init |
| `grasp_fail`（重抓风险集） | **不支持** | 最贴近命题，但风险集 ≤ 21 例；`bal = 14`，4 个 init |
| EEF return-to-pot2-side（39 条） | **不支持**（作为 landmark） | 39/39 全是失败，Y=0 为空；只能用作 §S8 的描述性事件 |
| 严格 pot1-side stasis（109 条） | **不支持**（作为 landmark） | 同上，且 onset 与 return 类相差 14 个 query |
| `heightloss` / `push_ungrasped` / `tilt` / `eef_backtrack` | **不支持** | 双结局支持充足（bal 54–184、8–13 个双结局 init），但 `setback_fraction ≤ 0.002`，固定视界检验显示 260/260 在 4 query 内恢复——**它们是正常放置动作，不是错误** |

---

## 6 要把这个实验做成，需要什么

按重要性排序：

1. **恢复模拟器 + 事件真值。** 需要 `mujoco` / `robosuite` / `libero`，
   以便 (a) 用 contact / force 标注真实滑落、空抓、碰撞，而不是运动学代理；
   (b) 从事件后状态做 snapshot replay，把「适应」变成可干预量。
   现在 contact 真值只有 5 条。
2. **让一阶事件后有恢复的可能。** 当前策略的行为是「失败后基本不重试」
   （重抓风险集 ≤ 21 例），因此 `Y=0` 类天然稀缺。需要
   (a) 提高步数上限（尤其 long 任务的 52，成功中位 39，余量太小），
   (b) 或者**主动注入一阶事件**（在受控 query 上扰动物体位姿 / 强制开爪），
   在同一初态下重复多个 seed，人为构造双结局风险集。这是唯一能保证
   `setback_fraction = 1` 且两类都够的路线。
3. **解耦终局与 episode 长度。** 成功后不要立刻终止，让所有 rollout 跑满
   固定步数，`Y` 才能与剩余预算正交。
4. **扩大 init 覆盖。** 现在每任务 16 init，双结局 init 常常只有 1–4 个；
   留一 init 的折数不足以给出可用的 CI。
5. **记录 chunk 内稠密状态。** 现在只存步末位姿，事件时刻带 ±1 chunk 不确定性，
   「事件前 / 事件后」的划分本身就有一个 chunk 的模糊带。
