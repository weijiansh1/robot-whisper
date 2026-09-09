# fixed-prefix MoE 负结果：复跑 + 修三个方法缺陷

脚本：`analyze_failure_moe_signatures_fixed.py`（仓库根目录，带 `--self-test`，可重跑）
产物：本目录 `summary.json` / `baseline_folds.csv` / `sentinels.csv` / `fold_goal_reference.csv` /
`fold_goal_predictions.csv` / `power_curve.csv` / `power_curve.png` / `baseline_cohort_features.csv.gz`
原样复跑产物：`../failure-moe-signatures-repro/`（未覆盖原目录）

---

## 判词

**负结果没有幸存 —— 但不是因为“方向反了”，而是因为这个实验从一开始就没有功效去支持任何结论，而且它的主指标本身是失效的。**

三条独立的、可量化的结论：

1. **主指标失效。** 报告用的 pooled leave-one-init-out OOF AUC，在这个 cohort 里 **91.6% 的正负配对是跨 init 的**，由 13 个不同模型打出的不可比分数拼在一起排序。把一维哨兵 `cut 处的 query 索引 q0+8`（边际 AUC **0.545**，方向为正）放进同一管线，pooled LOGO AUC 变成 **0.258**。所以主表里 0.439 / 0.412 / 0.457 / 0.392 / 0.436 这五个“低于随机”的数字，是池化伪影，不是效应反向。换成折内可比的 within-init AUC，五个 block 是 0.430 / 0.496 / **0.509** / **0.437** / 0.481 —— 全部 ≈ 随机。

2. **没有功效。** 用合成注入（把真实路由沿“同一 rollout 更早相位”这条真实轴插值，不用高斯噪声）做功效曲线：这套 n=158 / 13 init / joint 100 维的设计，**MDE80 = +0.252 within-init AUC 增量**（routing 单独 vs physical+action 则是 **+0.352**）。观测到的增量是 −0.028。也就是说，观测值比这个设计的最小可检出量小一个数量级，**“routing 增量 ≈ 0”这句话不携带信息**。

3. **一旦修掉 cohort 选择，routing 反而有明确增量。** 把 success 和 other_long_failure 放回来，在同一个固定 cut、同一个 landmark 风险集（所有 rollout 在 cut 时都还没发生物理事件）上做“会不会失败”：
   - routing 单独 within-init AUC **0.704**，physical+action **0.468**；
   - routing 对 physical+action 的增量 **+0.237，init-cluster bootstrap 95% CI [+0.074, +0.420]**（CI 不跨 0）；
   - joint 对 physical+action 的增量 **+0.157 [+0.050, +0.272]**。
   这个真实效应量（+0.16 ~ +0.24）**正好落在原设计 MDE80（+0.25）之下** —— 原协议在结构上就不可能看到它。

三个被点名的缺陷里，**只有 lead-time 不匹配和 cohort 选择是致命的**；transductive goal 泄漏是真的，但数值上可以忽略（折内重算后参考点最多移动 1.94 mm，测试集完全不变，AUC 变化 ≤ 0.003）。

---

## 1. 第 1 步：原样复跑（逐数字对照）

`python3 analyze_failure_moe_signatures.py --self-test` → `self-test passed`。
全量跑到 `analysis/failure-moe-signatures-repro/`：**与原目录逐位相同**
（`report.md` 与 `episode_results.csv` 的 md5 一致；`summary.json` 展平后 120 个叶子值 0 处差异）。

### 1.1 主表

| feature block | 声称 | 实测（原脚本复跑） | 独立重实现 | dims 声称/实测 |
|---|---:|---:|---:|---:|
| physical | 0.439 | 0.43934496875673346 | 0.43934496875673346 | 32 / 32 |
| action | 0.412 | 0.41241111829347130 | 0.41241111829347130 | 28 / 28 |
| physical + action | 0.457 | 0.45658263305322133 | 0.45658263305322133 | 60 / 60 |
| routing | 0.392 | 0.39237233354880413 | 0.39237233354880413 | 40 / 40 |
| physical + action + routing | 0.436 | 0.43632837750484810 | 0.43632837750484810 | 100 / 100 |

AP：0.262 / 0.260 / 0.301 / 0.204 / 0.251 → 实测 0.26232 / 0.26007 / 0.30125 / 0.20362 / 0.25102，一致。
balanced acc：0.443 / 0.444 / 0.435 / 0.427 / 0.469 → 实测 0.44333 / 0.44398 / 0.43536 / 0.42696 / 0.46919，一致。
routing 增量：声称 −0.020，CI [−0.114, +0.049] → 实测 **−0.020254**，CI **[−0.114391, +0.049237]**，2000 次有效抽样，一致。

### 1.2 稳定性审计表

| feature block | LOIO 声称/实测 | LOSO 声称/实测 | LOIO 仅双类 init 声称/实测 |
|---|---:|---:|---:|
| physical + action | 0.457 / 0.45658 | 0.822 / 0.82181 | 0.387 / 0.38718 |
| routing | 0.392 / 0.39237 | 0.651 / 0.65094 | 0.327 / 0.32670 |
| joint | 0.436 / 0.43633 | 0.798 / 0.79832 | 0.349 / 0.34903 |

两个增量：声称 −0.020 / −0.023 → 实测 −0.020254 / −0.023486，一致。

### 1.3 cohort 与 lead time

| 项 | 声称 | 实测 |
|---|---|---|
| cohort n | 158 | 158 |
| stasis(`stagnation_core`) / EEF return(`active_return`) | 119 / 39 | 119 / 39 |
| init 数 | 13 | 13 |
| 单类别 init | 6 | 6 |
| 全部 cut < onset | 是 | 是（最小间隔 1 query） |
| stasis lead 均值/中位/最小 | 2.40 / 2 / — | 2.4034 / 2 / 1 |
| active_return lead 均值/中位/最小 | 16.15 / 15 / 9 | 16.1538 / 15 / 9 |
| 物理类型计数 | 296 / 123 / 54 / 39 | 296 / 123 / 54 / 39 |

**唯一的不一致（已标出）：** 我的独立重实现把 pot 位置读成 float64（原脚本是 float32），
LOSO 的 physical+action pooled AUC 得到 **0.821590** 而不是 **0.821806**（Δ = 2.2e-4，一个 rank 交换）。
主表 5 个 AUC 因为是秩统计量，完全不受影响（17 位一致）。
上游 `episode_audit.csv` 的 `pot2_first_goal_query` / `long_physical_stage` / `eef_terminal_basin`
被我重新推导后 **512/512 全部吻合**，说明整条 goal → q0 → 标签链条被完整复现。

**基线可复现，确认无误。**

---

## 2. 第 2 步：哨兵与「为什么 AUC 全部 < 0.5」的诊断

### 2.1 哨兵表（单特征，走同一条 LOGO 管线）

| 哨兵（1 维） | 边际 AUC | pooled LOGO AUC | within-init AUC |
|---|---:|---:|---:|
| `lead_time`（onset − cut） | **0.978** | **0.974** | **1.000** |
| `episode_length` | — | — | — （cohort 内恒等于 52，退化） |
| `cut 处 query 索引 q0+8` | 0.545 | **0.258** | 0.417 |
| `pot2_anchor_query`（= q0） | 0.545 | 0.258 | 0.417 |
| `init_state_id` 原始编码（极端对照） | 0.338 | 0.537 | 0.500 |

三个结论：

- **lead_time 单独就能完美分开两类**（within-init AUC = 1.000）。两类在“距事件多远”这个轴上根本不重叠，而 lead_time 是标签本身的函数。任何与“事件还有多久发生”相关的东西都能赢，而 100 维真实特征做不到。
- **`cut 索引` 是决定性证据：一个边际方向为正（0.545）的特征，在同一管线下 pooled LOGO AUC = 0.258。** 一个指标能把正向关联翻成“远差于随机”，它就不是判别力度量。
- `init_state_id` 的边际 AUC 是 0.338（低于随机），但 pooled LOGO 给出 0.537 —— 同样说明 pooled 数字与真实关联脱钩。

### 2.2 为什么 AUC 会低于 0.5：机制

**(a) 91.6% 的配对是跨 init 的。** cohort 有 39 正 × 119 负 = 4641 个正负对，其中落在同一 init 内的只有 **391 个（8.42%）**。
剩下 4250 对，比较的是**两个不同模型**（held-out init A 的模型 vs held-out init B 的模型）打出的概率，没有共同标定。

把 pooled AUC 按配对来源拆开：

| block | pooled AUC | 跨 init 部分 | 折内（within-init）部分 |
|---|---:|---:|---:|
| physical | 0.439 | 0.440 | 0.430 |
| action | 0.412 | 0.405 | 0.496 |
| physical + action | 0.457 | 0.452 | **0.509** |
| routing | 0.392 | 0.388 | **0.437** |
| joint | 0.436 | 0.432 | 0.481 |

报告的数字几乎就等于“跨 init 部分”。折内部分全部回到 0.43–0.51，即**随机**。

**(b) 6/13 个 init 只有一个类别，占 49/158 条。** 这些 init 在折内不贡献任何对比，只能整体和别的折比高低。
`baseline_folds.csv` 里最典型的一组：

| held-out init | n | 正类数 | physical+action OOF 均分 |
|---:|---:|---:|---:|
| 10 | 6 | 6（全正） | 0.291 |
| 20 | 6 | 0（全负） | 0.676 |
| 49 | 31 | 0（全负） | 0.467 |
| 13 | 19 | 1 | 0.639 |

全正的 init 拿到最低分、全负的 init 拿到最高分，pooled AUC 自然被压到 0.5 以下。

**(c) 折间系数会翻号、分数尺度不可比。** 用 `cut 索引` 这个 1 维特征把机制看透：

| held-out init | n | 正类 | 折内 logistic 系数 | 该折 OOF 均分 | 该折 cut 取值范围 |
|---:|---:|---:|---:|---:|---|
| 10 | 6 | 6 | **+0.419** | 0.355 | 23–24 |
| 49 | 31 | 0 | **−0.183** | 0.608 | 24–25 |
| 0 | 16 | 7 | +0.294 | 0.457 | 24–26 |
| 39 | 26 | 11 | −0.010 | 0.499 | 26–28 |

init 10（6 条全正）和 init 49（31 条全负）的 cut 值几乎一样（23–24 vs 24–25），
但两折模型学到的系数一个 +0.42、一个 −0.18，于是全正的那 6 条得 0.355、全负的那 31 条得 0.608。
仅这两折就制造了 186 个不一致配对。**折间效应方向不一致 + OOF 分数跨折不可比，两条都成立。**

### 2.3 阳性对照：管线本身没坏

用同一 cohort、同一 LOGO、同一 routing block，去预测一个 routing 一定跟随的量
（窗口内平均 EEF 步长的中位数二分）：

| block | pooled AUC | within-init AUC |
|---|---:|---:|
| routing | 0.793 | **0.762** |
| physical + action | 0.958 | 0.985 |

管线能从 routing 里读出物理状态。**读不出的是那个标签对比，不是 routing。**

### 2.4 顺带修正：leave-one-seed-out 的 0.822 不是「另一种划分」

把 LOSO 的同一批 OOF 预测按不同分组重新打分：

| block | pooled（= 报告值） | within-seed | **within-init** |
|---|---:|---:|---:|
| physical + action | 0.822 | 0.850 | **0.691** |
| routing | 0.651 | 0.685 | **0.573** |
| joint | 0.798 | 0.827 | **0.665** |

LOSO 训练时见过同一 init 的其它 seed，所以模型能学到 init 专属的判别面，折内 AUC 提到 0.69。
这说明 **前缀里确实存在 init 内可判别的信息（physical+action 0.691），但它不跨 init 迁移**（LOGO 下降到 0.509）。
LOSO 里 routing 的增量仍是 −0.026 —— 即使在有 init 记忆的条件下，routing 也没超过 physical+action。

---

## 3. 第 3 步：修 transductive 泄漏（折内重算 goal reference）

### 3.1 环境真值 goal 不可得（已核查）

- BDDL（`LIBERO/libero/libero/bddl_files/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`）的 goal 是
  `(And (On moka_pot_1 flat_stove_1_cook_region) (On moka_pot_2 flat_stove_1_cook_region) (Turnon flat_stove_1))`。
- `cook_region` 在 `assets/articulated_objects/flat_stove.xml` 里是一个 `size="0.075 0.075 0.0025"` 的 box site，
  **两个 pot 共用同一个区域**，不是两个点；`On` 在 LIBERO 里是 contact + 区域包含判定，不是“到某点的距离”。
- `flat_stove_1` 是焊死的 fixture（`sim_layout.json` 里只有 `flat_stove_1_button` 进 qpos），
  世界位姿只存在于运行时生成的 XML 里；本机 **没有 mujoco / robosuite / libero**，离线无法还原。
- `analyze_post_error_recovery.py` 里的 “Transport targets are frozen from the BDDL goals” 指的是
  `TASK_TRANSPORT_OBJECTS`，即**只冻结了目标物体的身份**（`moka_pot_1` / `moka_pot_2`），**没有冻结坐标**；
  它的几何仍然来自 sim_state。

所以只能走“每折重算成功参考”这条路。

### 3.2 折内重算的结果

每个 held-out init 用**只含训练折 init 的成功轨迹**重算 pot1/pot2 终点均值，
再用该参考重推 `q0`、`long_physical_stage`、approach、onset、标签、cut 和 goal-distance 特征，
最后只用该折的训练 cohort 拟合、只对该折的测试 init 预测。

| 项 | 结果 |
|---|---|
| goal 参考位移（pot1）| 中位 0.68 mm，最大 **1.94 mm** |
| goal 参考位移（pot2）| 中位 0.60 mm，最大 **1.58 mm** |
| 每折 cohort 规模 | 16 折中 14 折为 158；held-out init 20 与 init 46 时为 157（掉的那 1 条 `active_return` 落在**训练**侧） |
| 测试集 | **与基线 158 条完全相同**（`test_set_identical_to_baseline_cohort = true`） |

**“cohort 随折变化”的处理：** 每折的 cohort 由该折自己的（无泄漏）参考定义，训练与测试都用同一套定义；
实测下来只有 2/16 折的**训练集**少 1 条，测试集集合完全不变，因此可以直接和基线逐 block 对比。

### 3.3 修完之后的主表

| feature block | pooled AUC（基线 → 折内 goal） | within-init AUC（基线 → 折内 goal） |
|---|---:|---:|
| physical | 0.4393 → **0.4409** | 0.4297 → **0.4348** |
| action | 0.4124 → **0.4137** | 0.4962 → 0.4962 |
| physical + action | 0.4566 → **0.4572** | 0.5090 → 0.5090 |
| routing | 0.3924 → **0.3900** | 0.4373 → 0.4373 |
| joint | 0.4363 → **0.4355** | 0.4808 → 0.4808 |

routing 增量（joint − physical+action）：

| 指标 | 基线 | 折内 goal |
|---|---|---|
| pooled | −0.0203 [−0.1144, +0.0492] | **−0.0218 [−0.1263, +0.0541]** |
| within-init | −0.0281 [−0.2709, +0.1220] | **−0.0281 [−0.2709, +0.1220]** |

**结论：transductive 泄漏真实存在，但幅度是毫米级，对 q0、cohort 和全部结论没有可测影响。这条缺陷不是负结果的成因。**

---

## 4. 第 4 步：修 lead time 不匹配

### 4.1 两类的 lead time 分布几乎不相交

| lead (query) | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 10 | 11–17 | 18 | 19 | 20–26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stasis (`stagnation_core`, n=119) | 47 | 59 | 3 | 2 | 1 | 1 | 1 | 1 | 0 | 3 | 1 | 0 |
| EEF return (`active_return`, n=39) | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 24 | 0 | 2 | 11 |

**106/119 条 stasis 的 lead ≤ 2 query；39 条 EEF return 全部 ≥ 9。** 重叠区只有 [9, 19]。

### 4.2 三种匹配方案，全部不可行

| 方案 | n | active_return | stasis | init 数 | **同时含两类的 init** | 折内正负对 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| lead ∈ [9,19]（完全重叠区） | 33 | 27 | 6 | 9 | **0** | **0** | 不可行 |
| lead ∈ [9,26] | 45 | 39 | 6 | 10 | **0** | **0** | 不可行 |
| 1:1 caliper 匹配（±3 query，优先同 init） | 14 | 7 | 7 | 6 | **0** | **0** | 不可行 |

三种方案里**没有任何一个 init 同时包含两类**，折内正负对全部为 0。
在这种情况下任何 AUC 都 100% 由跨 init 比较构成，也就是第 2 节已经证明失效的那种指标。
（作为参考：如果硬算，caliper 匹配集的 pooled AUC 是 physical+action 0.000 / routing 0.020 / joint 0.000 ——
这些数字唯一说明的是指标本身没有意义。）

**结论：lead-time 匹配版本在这份数据里不存在。原实验实际比较的是“2 个 query 后停住”和“15 个 query 后回返”，
这不是同一个预测问题，负结果无法被解释为“routing 在固定预测距离上没有增量”。**

---

## 5. 第 5 步：修 cohort 由未来筛选（把 success 与 other_long_failure 放回来）

### 5.1 协议

同一固定 cut（q0+8），landmark 风险集：**只要 rollout 在 cut 时还没发生已定义的物理事件就纳入**
（失败要求 cut < onset；成功要求 cut < 首次进入 pot1 的 13 cm 邻域）。
纳入 498 条：success 286、stagnation_core 119、active_return 39、other_long_failure 54，16 个 init。
（`other_long_failure` 没有可验证的 onset，因此另外报告一个把它剔除的干净版本。）

### 5.2 「会不会失败」的在线预警（这是原实验想回答但没有回答的问题）

**A. 全部 498 条（含 other_long_failure）**

| block | dims | pooled AUC | **within-init AUC** |
|---|---:|---:|---:|
| physical | 32 | 0.671 | 0.474 |
| action | 28 | 0.513 | 0.565 |
| physical + action | 60 | 0.592 | 0.544 |
| **routing** | 40 | **0.720** | **0.693** |
| joint | 100 | 0.678 | 0.620 |

routing 对 physical+action 增量：within-init **+0.149 [+0.026, +0.302]**；pooled +0.128 [−0.021, +0.298]。
joint 对 physical+action 增量：within-init +0.076 [−0.035, +0.202]。

**B. 干净版本 444 条（success 286 + 两类已定义失败 158，全部 cut < 物理事件）**

| block | dims | pooled AUC | **within-init AUC** |
|---|---:|---:|---:|
| physical | 32 | 0.682 | 0.416 |
| action | 28 | 0.543 | 0.508 |
| physical + action | 60 | 0.627 | 0.468 |
| **routing** | 40 | **0.763** | **0.704** |
| joint | 100 | 0.754 | 0.625 |

routing 对 physical+action 增量：within-init **+0.237 [+0.074, +0.420]**（CI 不跨 0）；pooled +0.136 [−0.004, +0.315]。
joint 对 physical+action 增量：within-init **+0.157 [+0.050, +0.272]**（CI 不跨 0）；pooled +0.127 [+0.036, +0.244]。
折内正负对 1191 个（vs 原 cohort 的 391），16 个 init 中只有 4 个单类别。

**C. 4 类多分类（macro one-vs-rest）**

| block | pooled macro OVR | within-init macro OVR |
|---|---:|---:|
| physical | 0.522 | 0.487 |
| action | 0.521 | 0.559 |
| physical + action | 0.563 | 0.539 |
| routing | 0.582 | **0.573** |
| joint | 0.579 | 0.570 |

**D. 同样这一池里的原两类对比**：完全不变（within-init 0.509 / 0.437 / 0.481，增量 −0.028）。
说明差别不是“多加了样本让模型变好”，而是**问题本身换了**。

### 5.3 这是不是时间/元数据混淆？（哨兵检查）

| 哨兵 | 边际 AUC | pooled LOGO | within-init |
|---|---:|---:|---:|
| `cut 索引 q0+8` | 0.459 | 0.346 | **0.496** |
| `pot2_anchor_query` | 0.459 | 0.346 | 0.496 |
| `init_state_id` 原始编码 | 0.476 | 0.280 | 0.500 |
| `episode_length` | 0.998 | 0.997 | **0.999** |

- cut 位置在折内 AUC 0.496 —— **routing 的 0.704 不是“任务进度早晚”的伪影**。
- `episode_length` 折内 AUC 0.999：失败一律跑满 52 query、成功提前结束，所以标签在事后被时长完全决定。
  但 `episode_length` **在 cut 时不可观测、也没有进入任何模型**，前缀窗口 `[q0, q0+8]` 全部落在 query 31 之前。
  这条哨兵说明的是“这个任务的失败=超时”，不是泄漏。

### 5.4 必须写明的限定

- physical block 在这个问题上折内只有 0.416–0.474（goal-distance 特征在 pot2 放好之后接近常数），
  所以 “routing > physical+action” 部分反映的是 **physical 基线弱**，不能读成“routing 比物理状态含更多信息”的普适命题。
- 只有一个 task、一个 cut、一个模型；不做因果宣称；`stagnation_core` / `active_return` 仍是运动学代理。
- 这是“不同的问题”，如方法复核所要求的那样单独报告：**多类 / 全体在线预警 ≠ 两类条件判别**。

---

## 6. 第 6 步：token 拆分（R^state vs R^action）

现有 40 维 routing block 用的是 token 1–10（action tokens）。拆成 token 0（state）与 token 1–10（action）各 40 维。

**在原来的选择性两类协议上（n=158）**

| routing 变体 | routing within-init | joint within-init | joint 增量 (within) | routing 增量 (within) |
|---|---:|---:|---|---|
| R^action（原） | 0.437 | 0.481 | −0.028 [−0.271, +0.122] | −0.072 [−0.323, +0.036] |
| R^state | 0.381 | 0.448 | −0.061 [−0.127, −0.019] | −0.128 [−0.265, −0.005] |

R^state 的两个增量 CI 落在 0 以下 —— 但这**不是“state token 的路由含反信息”**，
而是在 158 条样本上再加 40 个无信息维度带来的方差；在同一张表里 pooled 尺度的增量是 +0.001 / +0.031（跨 0）。
两个尺度符号相反，本身就说明这个 cohort 上任何符号都不可信。

**在干净 landmark 协议上（n=444，才有分辨力）**

| routing 变体 | routing pooled | routing within-init | routing 增量 (within) | joint 增量 (within) |
|---|---:|---:|---|---|
| R^action | 0.763 | **0.704** | +0.237 [+0.074, +0.420] | +0.157 [+0.050, +0.272] |
| R^state | 0.817 | **0.688** | +0.221 [+0.034, +0.398] | +0.142 [+0.028, +0.241] |

**结论：token 轴上没有可检出的差异。** state token 与 action token 的路由携带基本等量的前瞻信息
（折内 0.688 vs 0.704，CI 大幅重叠）。原 block 只用 action token 不构成一个方法缺陷。

---

## 7. 第 7 步：功效（合成注入，MDE80）

**设计**（对齐 `rerun-2026-08-27/gate1_power_audit.py`）：
对 39 条正类（`active_return`）episode，把前缀窗口内某个随机起点到 cut 的路由，
按强度 λ **朝该 rollout 自己在更早某个随机 query 的路由插值**
（soft router 概率与 hard expert occupancy 同时插值，全部保持在单纯形上）—— **不使用高斯噪声**。
然后重算全部 10 个 routing 信号 × 4 个统计量，重跑同一条 LOGO 管线与同一条推断规则。
检出规则 = “joint 对 physical+action 的 within-init AUC 增量的 init-cluster bootstrap 95% CI 下界 > 0”，
每格 20 次重复，每次 400 抽样。

| λ | 诱发的 joint 增量（中位） | routing 单独 within-init AUC | 检出率（joint 规则） | 检出率（routing 单独规则） |
|---:|---:|---:|---:|---:|
| 0.000（空注入对照） | −0.028 | 0.437 | 0.00 | 0.00 |
| 0.002 | −0.001 | 0.494 | 0.00 | 0.00 |
| 0.004 | +0.013 | 0.541 | 0.00 | 0.00 |
| 0.006 | +0.038 | 0.586 | 0.00 | 0.00 |
| 0.008 | +0.073 | 0.633 | 0.00 | 0.00 |
| 0.011 | +0.129 | 0.679 | 0.10 | 0.00 |
| 0.015 | +0.191 | 0.738 | 0.60 | 0.00 |
| 0.020 | +0.267 | 0.785 | **0.85** | 0.30 |
| 0.030 | +0.372 | 0.891 | 1.00 | 1.00 |

- **MDE80（joint 对 physical+action）= +0.252 within-init AUC 增量**（λ≈0.019）
- **MDE80（routing 单独对 physical+action）= +0.352**（λ≈0.027）
- 观测值 = **−0.028**
- λ=0 的空注入给出 −0.028 与 0.00 检出率，说明检出规则本身标定正确（无假阳性）。

**解释：** 这个 n=158 / 13 init / 7 个双类 init / 391 个折内正负对 / joint 100 维的设计，
只能检出 **+0.25 及以上**的 routing 增量。观测到的 −0.028 与 0、与 +0.1、与 +0.2 都无法区分。
而第 5 节测到的真实效应量是 **+0.157 ~ +0.237** —— **正好在这个设计的检出阈值之下**。
所以原实验不是“发现 routing 没有增量”，而是“用一个看不见该量级效应的设计得到了一个空结果”。

---

## 8. 六条缺陷的逐条处置

| # | 缺陷 | 处置 | 是否影响原结论 |
|---:|---|---|---|
| 1 | 上游 goal 参考 transductive | 每折用训练折成功轨迹重算，连带重推 q0/标签/cohort；另核查 BDDL 真值不可离线获取（第 3 节） | **否**（位移 ≤1.94 mm，测试集不变，AUC 变化 ≤0.003） |
| 2 | cohort 由未来筛选 | 加入 success 与 other_long_failure，做 landmark 二分类 + 4 类多分类（第 5 节） | **是，决定性**（routing 增量由 −0.028 变为 +0.157 ~ +0.237） |
| 3 | lead time 严重不匹配 | (a) 加 `lead_time` 一维哨兵（within-init AUC = 1.000）；(b) 三种匹配方案全部证明不可行（第 2、4 节） | **是，决定性**（两类不在同一预测距离上；匹配版本不存在） |
| 4 | init 分布偏 | 折内/跨折配对拆解：391/4641 = 8.4% 折内；6/13 单类 init 占 49 条；改用 within-init AUC 作为主指标（第 2 节） | **是，决定性**（主指标失效） |
| 5 | 1-query 间隔 ≠ 物理事件前 | 未修：缓存只有 query 边界，chunk 内状态不可得。但已被 (3) 覆盖 —— 106/119 条 stasis 的 lead ≤2 query，本来就贴着 cut | 保留为限制 |
| 6 | 功效有限 | 合成注入功效曲线，MDE80 = +0.252（第 7 节） | **是，决定性**（观测值远小于 MDE） |

---

## 9. 残余局限（本报告也没有解决的）

- 只有一个 task（KITCHEN_SCENE8 long）、一个 cut（q0+8）、一个模型族（L2 logistic）。
- 第 5 节的正向结果依赖 landmark 规则对 success 的过滤（296 → 286，去掉了 10 条在 cut 前就已接近 pot1 的快速成功）；
  规则本身仍用到 goal 代理，虽然第 3 节证明该代理的折内/全局差异可忽略。
- 缓存没有 RGB / contact / force / chunk 内稠密状态；所有物理标签仍是运动学代理。
- 全部为预测性关联；没有做 route clamp / expert swap / snapshot replay 之类的干预，不作因果宣称。
- 功效曲线的 λ 与“真实机制强度”没有直接单位换算；只有“诱发的 AUC 增量”这一列可以和观测值直接比较，
  MDE 应按该列读，不要按 λ 读。

---

## 10. 复现

```bash
cd /home/jovyan/work/himoe-vla/himoe-route-capture

# 原样复跑（不覆盖原目录）
python3 analyze_failure_moe_signatures.py --self-test
python3 analyze_failure_moe_signatures.py --out-dir analysis/failure-moe-signatures-repro

# 修正版（本报告的全部数字）
python3 analyze_failure_moe_signatures_fixed.py --self-test
OMP_NUM_THREADS=8 python3 analyze_failure_moe_signatures_fixed.py \
    --out-dir analysis/failure-moe-signatures-fixed \
    --bootstrap 2000 --power-bootstrap 400 --power-repeats 20 \
    --power-strengths 0.0 0.002 0.004 0.006 0.008 0.011 0.015 0.020 0.030
```

第一次运行会写 `route_reduced.npz`（94 MB，token 拆分后的 4 组 `[query, layer, expert]` 分布缓存），
之后自动复用；总运行时间约 13 分钟（其中功效扫描约 10 分钟）。
连续两次全量运行的 `summary.json` 除 `runtime_seconds` 外 405 个叶子值 **0 处差异**。
