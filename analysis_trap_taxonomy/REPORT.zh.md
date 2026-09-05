# HiMoE-VLA Trap 数量与 MoE 路由规则：无训练离线审计

> 结论日期：2026-09-04。主分析使用两个已经完整冻结的 50 scene x 8 flow-noise
> 批次，共 40 个 LIBERO 任务、32,000 条轨迹、508,023 次推理。
> 全程没有训练分类器，没有用失败标签选择阈值、特征方向或权重。

## 直接答案

上一版把“失败机制”和“轨迹动力学表型”混成了同一层，因此漏掉了一个真实类型。修正后的
最小结论是：**至少 3 个已确认的操作性 Trap，但它们不是互斥的平级聚类。**

其中两个是宏观动力学表型：

1. `loop / switching / retry`：身体或物体绕回旧状态，但中间走过显著路径；MoE 路由在
   flow 内加速、切换增多，跨 query 复现度下降。
2. `static / freeze`：EEF、任务物体、夹爪持续低运动；MoE 路由冻结，跨 query 近乎复现，
   flow 内变化和加速度同时下降。

第三个是语义/物理失配机制：

3. `belief--physics mismatch / phantom grasp`：夹爪在尚未完成的目标附近闭合并保持闭合，目标
   没有随末端运动，策略却继续让机械臂离开，行为上已经进入“搬运”阶段。严格物理代理在 A/B
   分别找到 114/115 条 timeout，占两批 timeout 的 21.4%/20.4%，成功轨迹中为 0；该现象跨
   15/14 个任务出现。

第三类可以随后诱发 loop 或 static，所以不能把三类计数相加。在 229 条严格 phantom-grasp
轨迹中，154 条同时以 loop 为首个宏事件，20 条以 static 为首个宏事件，55 条没有任何
loop/static。后 55 条直接证明它不是前两类换一个名字。

这仍不是“总共恰好 3 类”的证明。严格说法是：**3 类已确认 + 一个仍未完成的语义候选集合 +
右删失 unknown 桶**。没有触发任何已知事件的 timeout 不能被宣布为正常。

按表面物理行为还能稳定复现 6 个重叠候选：stagnation、gripper cycling、goal regression、
approach-leave、subtask undo、regrasp/drop。但留出审计没有发现它们形成 6 个独立 MoE 态：
cycling 回到 switching 轴，stagnation 回到 freeze 轴，其余没有跨批次稳定的新方向。这只能说明
它们尚未获得独立路由表型，不能再被表述成“没有其他物理失败类型”。

## Timeout 的正确语义

| 冻结批次 | 轨迹 | success | `success=false` | 失败恰好结束于 `max_steps` |
| --- | ---: | ---: | ---: | ---: |
| seeds 1000--1007 | 16,000 | 15,468 | 532 | 532/532 |
| seeds 1008--1015 | 16,000 | 15,436 | 564 | 564/564 |
| 合计 | 32,000 | 30,904 | 1,096 | 1,096/1,096 |

没有成功轨迹走到或超过 `max_steps`。因此这批数据里的 `success=false` 只表示：任务在预算耗尽时
仍未完成。它没有记录一个独立的“失败原因”。这带来两个必须同时保留的区分：

- **timeout 不是机制性的 Trap 类型**；它是评测的删失边界。
- **进入 Trap 也不等于最终失败**；一些轨迹之后会恢复并成功。

在 1,096 个 timeout 中，832 个在截止前出现 loop/static，264 个没有**宏观动力学事件**。
新审计表明后 264 个中有 55 个严格 phantom grasp，所以原先把 264 个全部写成 unknown 是错误的。
扣除 phantom grasp 及已有的重叠物理候选后，A/B 各剩 36 条、合计 72 条完全未命中；这 72 条是
当前观察到的 `right-censored / unknown` 残余桶，不代表存在 72 条新机制。其余候选尚未都有独立
MoE 表型，已知检测器本身也会漏检，因此不能据此宣布 taxonomy 已闭合。

## 两类宏观 Trap 的物理规则

事件定义在查看 B 批路由结果前冻结。每个 query 使用 EEF、所有可识别 free-joint 任务物体和
夹爪孔径；抽屉和旋钮两个铰接任务的 loop 通道因代理无效而排除，static 通道仍保留。

### 1. Loop / Switching

在当前 query `r` 之前存在 `l < r-2`，同时满足：

- EEF 回到 4.5 cm 内；
- 所有任务物体回到 3.0 cm 内；
- 夹爪孔径回到 1.2 cm 内；
- 但 `l -> r` 的 EEF 累积路径至少 12 cm。

这区分“真正绕了一圈”与普通微小停留。它仍是 query-boundary 运动学代理，不是接触级真值。

### 2. Static / Freeze

宽度为 2 个 query 的滑窗同时满足：EEF 累积位移不超过 2.0 cm、任务物体不超过 0.5 cm、
夹爪不超过 0.1 cm；连续命中 7 个窗后记为 static onset。

## MoE 的实际门控规则

### 模型结构

- AS-MoE 位于层 0、1、16、17，是 `top-1-of-3`。门控输入是 action-space `data_mask`，
  不是当前视觉/动作 hidden state。在每个任务内部，A/B 的 AS 概率逐 query 完全不变，top-1
  固定为 `[2, 0, 0, 1]`；它只能反映 action-space/checkpoint 常量，不能分 Trap。
- HB-MoE 位于层 2--5、12--15。每个 token 计算
  `p = softmax(W h_norm)`，选择未排序的 top-4，再把四个权重归一化；路由专家输出之外始终加一个
  shared expert。
- 本审计主看深层 HB 12--15、10 个 action token 和完整 32 路概率。client NPZ 里的
  `expert_ids=0` 是占位；真实记录在 server `routes.zarr`。

因此 Trap 路由规则首先是**分布随时间怎样运动**，其次才是哪些 layer-expert 概率增减。
不能把 `topk(sorted=False)` 的 slot 0 当作 top-1，也不应从 fp16 概率重建实际 top-4。

### 宏观路由规则

记：

- `V`：深层 action-token 在最后几个 flow step 的路由变化；
- `A`：32 路概率平方根轨迹沿 flow 的二阶变化；
- `Rec`：当前 query 与前 1--4 个 query 的最大 weighted-Jaccard 相似度。

每个事件与**同任务、同初始 scene、同绝对 query、但没有该类事件**的轨迹比较。下表是 onset
处定向 AUC；0.5 为无差别，越大表示越符合箭头方向。

| Trap | 无训练路由规则 | A 批定向 AUC | B 批定向 AUC |
| --- | --- | ---: | ---: |
| loop | `V ↑` | 0.737 | 0.733 |
| loop | `A ↑` | 0.766 | 0.750 |
| loop | `Rec ↓` | 0.688 | 0.695 |
| static | `V ↓` | 0.723 | 0.726 |
| static | `A ↓` | 0.677 | 0.709 |
| static | `Rec ↑` | **0.904** | **0.925** |

方向在 onset 前 3--4 个 query 已经出现，并在 onset 附近最强。已有更丰富特征还显示：

- loop：action token 共识下降、token dispersion 上升；gate 往往变尖，但 entropy 这条腿只算辅助证据。
- static：action token 共识上升、dispersion 下降；gate 往往变平、margin 下降，但该门控方向在另一种
  宏状态分析中有冲突，且对 mobility 残差化后不独立。最稳的规则是 `Rec ↑ + V/A ↓`。

这两类不是同一条“严重度轴”的两端：loop 是高切换、低复现，static 是低切换、高复现，合并统计
会互相抵消。

### Belief--Physics Mismatch / Phantom Grasp

这类事件使用已经在旧抓取审计中固定的运动学阈值，并增加“目标尚未完成”和“离开时仍保持闭合”
两条必要约束：

- 夹爪 aperture 从 `>=0.05` 跨到 `<0.05`，闭合点距最近的未完成目标小于 16 cm；
- 目标在闭合时距成功目标位置大于 5 cm，排除双物体任务中已经放好的邻近物体；
- 夹爪从闭合到离开一直保持 `<0.05`；
- EEF 离开至少 10 cm，目标此后总位移仍小于 1 cm，EEF--目标距离达到至少 15 cm。

这是保守的 query-boundary 代理，没有 contact/force 真值；“目标此后一直不动”会漏掉先抓空、后来
成功重抓的可恢复事件。严格下界如下：

| 批次 | phantom-grasp timeout | 占本批 timeout | 涉及任务 | success 命中 |
| --- | ---: | ---: | ---: | ---: |
| A, seeds 1000--1007 | 114/532 | 21.4% | 15 | 0/15,468 |
| B, seeds 1008--1015 | 115/564 | 20.4% | 14 | 0/15,436 |

若只要求曾在闭合状态下出现物理失配、不要求此后永远未恢复，A/B 还各有 5/10 条成功轨迹命中；
它们可能是恢复，也可能是 query 采样代理的误差，因此没有计入上面的确认数。严格事件中，从闭合到
物理失配可见的中位延迟在两批都是 3 个 query。

严格标签使用了“此后一直不动”这一未来条件，因此 success 中 0 命中是物理一致性检查，不是一个
可在线使用的 0% false-positive 结果。在线检测只能看有限历史窗口，并需要把“短暂失配后恢复”与
真正持续陷阱分开。

与同 task、同 scene、同目标的成功耦合抓取比较，路由在闭合后的 `+2/+3` query 形成稳定结构：

| 路由量 | 相对 query | A 定向 AUC | B 定向 AUC | 方向 |
| --- | ---: | ---: | ---: | --- |
| route acceleration | +2 | 0.923 | 0.941 | `↑` |
| route acceleration | +3 | 0.924 | 0.915 | `↑` |
| late-flow volatility | +2 | 0.891 | 0.847 | `↑` |
| late-flow volatility | +3 | 0.863 | 0.889 | `↑` |
| layer-5 state/action gap | +2 | 0.686 | 0.675 | `↑` |
| front state/action gap | +3 | 0.827 | 0.815 | `↑` |
| back chunk jump | +3 | 0.738 | 0.817 | `↑` |

因此可复现的群体规则是：**闭合后的新观测引起 back route 跳变，同时 HB 在 flow 内更弯曲，
front state/action routing 逐渐分裂；但这种变化没有把行为及时拉回接近/重抓阶段。**

一个重要反证也必须保留：先前单案例里“action routing 仍落在健康搬运模板内”没有跨任务复现。
群体上 front action-route 到健康模板的距离反而更大，`+2` 的 A/B AUC 为 0.824/0.780，`+3`
为 0.819/0.827。更准确的解释不是“MoE 完全没看到抓空”，而是“MoE 对新反馈发生了明显响应，
却没有形成正确的恢复动作”。`belief` 在这里只是由闭合空搬运行为推断的操作性名称，不是被直接
读出的模型内部变量。

这套量也不是 MoE-only 的充分判据。和其他失败轨迹中的近物闭合比较，`+2` route acceleration
只有 0.681/0.623，late-flow volatility 只有 0.675/0.560，而且 95% CI 均跨 0.5。最可靠的
检测仍需把物体--EEF 耦合检查与路由异常合取；单看 route acceleration 会混入 loop 和一般失败。

### 具体 layer-expert 签名

进一步直接检查 4 x 32 个深层 layer-expert。每个事件仍做同 task/scene/query 匹配；分别检验完整
soft probability 和模型实际执行的 top-4 占用；loop/static 每个事件/指标/批次内对 128 项做 BH
校正，phantom grasp 则对 `2 query x 4 layer x 32 expert = 256` 项一起校正，再要求 A/B 同方向。

| Trap | 路由量 | A/B 128 维效应相关 | A/B 都通过 BH 的单元 | 最佳固定单元 | A/B 定向 AUC |
| --- | --- | ---: | ---: | --- | --- |
| loop | soft probability | 0.846 | 9 | `L13:E6 ↓` | 0.646 / 0.654 |
| loop | actual top-4 occupancy | 0.798 | 17 | `L13:E10 ↑` | 0.606 / 0.611 |
| static | soft probability | 0.880 | 27 | `L15:E1 ↓` | 0.748 / 0.788 |
| static | actual top-4 occupancy | 0.875 | 36 | `L14:E10 ↑` | 0.747 / 0.760 |
| phantom +2 | soft probability | 0.858 | 12 | `L15:E22 ↓` | 0.791 / 0.787 |
| phantom +2 | actual top-4 occupancy | 0.848 | 11 | `L14:E0 ↑` | 0.726 / 0.765 |
| phantom +3 | soft probability | 0.875 | 16 | `L13:E20 ↑` | 0.877 / 0.774 |
| phantom +3 | actual top-4 occupancy | 0.820 | 17 | `L13:E20 ↑` | 0.740 / 0.696 |

较强且复现的 soft 签名包括：

- loop：`L13:E6 ↓`、`L15:E5 ↑`、`L14:E12 ↓`；
- static：`L15:E1 ↓`、`L13:E15 ↑`、`L14:E10 ↑`、`L15:E6 ↑`。
- phantom grasp：`+2` 的 `L15:E22 ↓`、`L13:E6 ↓`、`L15:E0 ↑`，以及 `+3` 的
  `L13:E20 ↑`、`L14:E2 ↑`、`L15:E28 ↓`。

有些单元直接区分两种 Trap：`L14:E12` 和 `L14:E8` 在 loop 中下降、static 中上升；
`L14:E29` 在 loop 中上升、static 中下降。实际 top-4 中，`L15:E17` 和 `L15:E13` 也呈反向占用。

所以原先的“没有 expert ID 规律”需要改得更精确：

**没有单一 expert 是确定性 Trap 开关，但存在跨种子复现的分布式 expert 重权重签名。**

最佳单一单元远未达到 AUC=1，而且不同层的同编号专家是不同网络，不能跨层合并编号。实际应用应
以 V/A/Rec 和整条 32 路分布为主，具体 ID 只做辅助解释。

phantom-grasp 的 BH family 更严格，包含 2 个相对 query x 4 层 x 32 experts；但这些签名仍是
相对成功耦合抓取得到的，尚未证明能与所有其他失败机制稳定分开。

## Trap 深度：事件不等于失败

按首次事件做互斥分类：

| 首达状态 | 总轨迹 | 最终成功 | timeout | 事件后成功率 |
| --- | ---: | ---: | ---: | ---: |
| loop first | 1,089 | 514 | 575 | **47.2%** |
| static first | 281 | 24 | 257 | **8.5%** |
| no event | 30,630 | 30,366 | 264 | 99.1% |

static 是明显更深的吸引域；loop 经常只是一次可恢复的重试。已有等预算 committor 与驻留分析还显示：
loop 的逃逸率强依赖已驻留时间，更适合半 Markov 描述；static 更接近低而稳定的逃逸 hazard。
而 onset 当下的路由不能可靠预测“这次能否逃逸”，说明 MoE 更像状态传感器，不是结局预言机。

## 其余候选的留出验证

phantom grasp 已由上节单独确认；其中仍有 A/B 各 23/32 条严格事件完全没有 loop/static。下面继续
只看其余没有 loop/static 的 timeout。任务物体、目标位置和所有成功分位阈值只从 A 批成功轨迹
确定，然后冻结并应用到 B 批；没有使用失败标签调规则。

| 重叠物理候选 | A 批 no-event timeout | B 批 no-event timeout | 留出路由结论 |
| --- | ---: | ---: | --- |
| stagnation | 20 | 20 | 倾向 `A ↓, Rec ↑`，属于 freeze 轴；B 精确匹配 cell 功效不足 |
| gripper cycling | 15 | 15 | B 的可比 cell 全部 `V ↑, A ↑`，属于 switching 轴 |
| goal regression | 36 | 28 | 没有跨批次稳定的新 V/A/Rec 方向 |
| approach-leave | 5 | 6 | 倾向 `V/A ↑`，但只有 1--3 个可比 cell |
| subtask undo | 18 | 18 | 没有跨批次稳定的新方向 |
| regrasp/drop | 54 | 52 | A 偏 switching，B 不复现 |
| 任一候选 | 81/131 | 78/133 | 标签互相重叠，不能相加为类型数 |
| 无上述 8 个候选 | 50/131 | 55/133 | 其中仍含一部分 phantom grasp |
| 再扣除 phantom grasp | 36/131 | 36/133 | 当前完全未命中的 right-censored 残余桶 |

`active_retry` 和 `eef_oscillation` 在 A/B 失败中均为 0；它们不能作为额外失败类型。

另外两个看似可能的新类型也没有通过：

- wrong-manifold drift：A 批 108 个失败候选中 91 个同时有 loop/static，而且漂移只在中晚期出现，
  更像 Trap 后的任务流形退化或修饰量，不是独立早期 Trap。
- self-confident lock-in：既有 22 个候选中绝大多数其实伴随 static，真正 no-event 只剩极少数；
  失败也没有相对成功富集，不能立为第三类。

## 可落地的无训练判据

原始路由值携带极强的 task identity，不能跨任务用一个绝对阈值。在线规则应先用同任务/同阶段的
成功参照，或当前轨迹早期的 causal self-baseline，转成方向一致的秩或稳健 z-score：

```text
switching_score = high(V) + high(A) + low(Rec)
freeze_score    = low(V)  + low(A)  + high(Rec)

switching_score 持续超过成功校准阈值 -> loop-like Trap
freeze_score    持续超过成功校准阈值 -> static-like Trap
两者都不满足                         -> unknown，不强判失败
```

phantom grasp 不能仅凭 MoE 可靠确认。若在线可获得目标跟踪，应使用：

```text
gripper_closed
AND eef_departed
AND target_not_coupled
AND (route_acceleration_high OR front_state_action_gap_high)
-> belief--physics mismatch alarm
```

其中前三项给出语义物理真值，MoE 只作为“策略仍在异常演化”的第二证据。没有目标跟踪时，
`A/V ↑ + gap ↑ + back jump ↑` 只能报风险，不能声称模型一定“以为已经抓住”。

阈值只应由成功轨迹按目标误报率校准，并加入 persistence/hysteresis；不要从当前失败集优化。具体
expert 签名可作为第二证据通道，但不应单独报警。尤其不能用 episode length 或绝对 query 当特征，
否则会退化成“离时限越近越像失败”的时钟。

## 证据边界

- 所有结论都是观察性关联，不是干预因果；要证明某种 routing 修正能救回轨迹，仍需在线分叉实验。
- 物理标签来自 query-boundary state/sim_state，没有 RGB、contact、force 真值。
- phantom-grasp 的 229 条是“始终未恢复”的严格下界；短暂抓空后恢复的事件没有完整计入。
- 严格 phantom 标签读取事件后的整段轨迹，只能用于事后分型，不能直接作为 causal online alarm。
- belief-mismatch 路由与成功抓取区分很强，但相对其他失败闭合的样本只有 10--11 个可比 cell，
  置信区间跨 0.5，尚不是独立 MoE-only detector。
- phantom-grasp 的核心距离阈值沿用旧审计，但“目标未完成”和“持续闭合”是本轮加入的语义约束；
  A/B 是两批离线复现，不是完整规则冻结后的第三批前瞻验证。
- A/B 共享同一模型，只改变 flow-noise seeds；它们是独立 rollout 批次，不是独立 checkpoint。
- B 留出确认了 loop/static 方向与 expert 签名，也复现了 phantom-grasp 数量和群体路由结构；
  但相对其他失败的同 scene 可比 cell 仍少。
- 因而“至少 3”是当前可证实的下界，不是不可推翻的总数上界。

## 产物

| 文件 | 内容 |
| --- | --- |
| `run_holdout_audit.py` | A/B 事件、终止语义、onset 路由效应主审计 |
| `validate_candidate_modes.py` | A 成功校准、B 冻结验证的其余候选审计 |
| `audit_expert_identity.py` | 128 个 layer-expert soft/top-4 的跨批次审计 |
| `validate_belief_mismatch.py` | 持闭合 phantom-grasp 物理标签、A/B 路由及失败负对照 |
| `audit_belief_mismatch_experts.py` | phantom-grasp 的 soft/top-4 专家身份跨批次审计 |
| `results/summary.json` | 主库存、删失、架构摘要 |
| `results/onset_route_effects.csv` | V/A/Rec 的 lead -4..+2 完整数值 |
| `results/candidate_summary.json` | 候选行为 A/B 库存 |
| `results/candidate_route_effects.csv` | no-event timeout 中的候选路由效应 |
| `results/expert_identity_summary.json` | expert 签名跨批次摘要 |
| `results/expert_identity_effects.csv` | 1,024 个 expert 单元检验结果 |
| `results/belief_mismatch_summary.json` | 第三类数量、重叠关系和路由规则摘要 |
| `results/belief_mismatch_route_effects.csv` | 相对成功耦合抓取的事件时刻 A/B 效应 |
| `results/belief_mismatch_structural_route_effects.csv` | layer gap、chunk jump、action-template 距离 |
| `results/belief_mismatch_other_failure_route_effects.csv` | 相对其他失败闭合的特异性负对照 |
| `results/belief_mismatch_expert_summary.json` | 第三类的专家效应相关、BH 复现与固定最佳单元 |
| `results/belief_mismatch_expert_effects.csv` | 1,024 个相对 query/layer/expert 检验结果 |
