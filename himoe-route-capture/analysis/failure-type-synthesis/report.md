# 失败类型与 MoE 特征综合结论

## 一句话结论

当前数据只支持一个相对明确的 MoE 现象：**rollout 进入低运动停滞后，尾部 routing 也变得持续、低变化，因而可以作为停滞状态的内部读出。** 对其余失败，物理过程已经能分出若干可重复现象，但没有发现稳定、独立、可提前预测的 MoE 机制类别。

## 跨分析一致的结构

下表是保守口径。各行是可重叠现象，数量不能相加。

| 现象 | 保守数量 | 物理特点 | 当前 MoE 结论 | 证据等级 |
|---|---:|---|---|---|
| 严格 pot1 侧停滞核心 | 109 | long task 中 pot2 已完成，EEF 停在未完成 pot1 一侧，末段持续低运动 | 尾部 route/action 均平台化；是目前最清楚的 routing trap | 强关联，非因果 |
| EEF 回到 pot2 一侧 | 39 | 已接近 pot1，随后 EEF 又回到已完成的 pot2 一侧 | 路由继续活跃并随物理状态分叉；不是稳定 recurrence 簇 | 强运动学，MoE 状态影子 |
| 目标距离回退 | 45 | 两套独立规则都发现相对历史最佳目标距离明显恶化 | 未形成独立路由类型，常与 lift/height-loss 代理重叠 | 中等 |
| 运输/高度丢失代理 | 31 | 闭爪附近物体与 EEF 同动，随后在 goal 外下降并分离 | 各任务尾部多表现为减速/持续，但没有专属 MoE 模式 | 强代理，非真实 drop 标签 |
| 未检测到运输代理 | 29 | stove 取碗失败中没有保守 EEF-object 共动证据；其中 27 条为 primary pattern | 多数尾部低动作、路由持续；不能由 route 判定“没抓住” | 中等代理 |
| ramekin 位移异常 | 10/12 | 非目标 ramekin 位移超过同任务成功 q99 | 样本少，只有描述性尾部持续，不能称碰撞机制 | 强运动学，原因未知 |
| 未命中离散规则的低进展尾部 | 88 | 87 条中后段速度低于 sibling-success q25，79 条进展低于 q05 | 位于成功到停滞的连续轴中间，没有新路由簇 | 强连续结构 |

缓存没有 RGB、contact、force 或 chunk 内稠密状态。因此 `drop`、`collision`、`wrong target`、`visual confusion` 和真实抓取状态都不能作为真值结论。

## “活跃重试”没有形成新类别

弱事件规则发现 20 条“末段仍动，并对未完成目标重复靠近、闭爪”的代理。但 20/20 的速度都没有达到同 task/init 成功轨迹的 q25。更严格的 success-normalized `active_retry` 和 EEF 异常振荡标签在失败中均为 0。

因此更准确的描述是：机器人仍有动作，但整体已经减速且进展很低；当前证据不支持一个独立的“高速异常重试/周期循环”失败类。

## MoE 能说明什么

完整 episode 的描述结果显示：

- 严格 pot1 停滞核心的末段 hard-route overlap 中位数为 0.798，action speed 为 0.21；表现为路由和动作共同持续化。
- EEF 回到 pot2 一侧的末段 route speed 仍为归一化 1.08，hard overlap 为 0.601，action speed 为 0.77；它仍在闭环运动，而不是停滞 trap。
- drawer regression、无运输代理、运输丢失代理和 ramekin 位移异常也常伴随尾部减速/路由持续，但彼此数值接近，不能仅凭 routing 判断具体物理原因。

所以尾部 MoE 信号目前能较可靠地区分“已经持续停住”与“仍在运动”，但不能可靠回答为什么停住、是否掉落、是否碰撞或是否选错目标。

## 真正前瞻的检验是负结果

在 long task 中，把输入固定截在 pot2 首次到位后第 8 个 query，并只纳入截点仍早于物理 onset 的 158 条轨迹：

| 特征 | leave-one-init-out AUC |
|---|---:|
| physical + action | 0.457 |
| routing | 0.392 |
| physical + action + routing | 0.436 |

加入 routing 的 AUC 增量为 -0.020，init-cluster bootstrap 95% CI 为 [-0.114, +0.049]。因此只能说：**这套选择性协议下没有检测到 routing 的提前增量。**

该实验仍是回顾性 case-control：cohort 由未来终局筛选，goal reference 使用全部成功轨迹，且两类 lead time 不匹配（停滞中位 2 query，EEF 回返中位 15 query）。它不能证明 routing 普遍没有前瞻信息，但足以否定“当前尾部 trap 已经被证明是通用 pre-failure signal”的说法。

## 当前研究图景

1. 已确认：尾部 routing trap 是持续停滞的状态读出。
2. 已确认：39 条 EEF 回返、45 条目标距离回退、31 条运输/高度丢失代理等物理现象真实存在。
3. 未确认：这些非停滞现象各自具有独立的 MoE 路由盆地。
4. 未确认：MoE routing 在物理/action 之外提前预测这些事件。
5. 未确认：routing trap 或某个 expert 是失败的因果来源。

## 下一轮最严格的实验

- 为每种物理事件定义 query-boundary onset，并使用相同 lead time 的风险集。
- 每个训练折单独构造 goal reference，同时匹配 task、init、当前物理状态和动作计划。
- 依次比较 physical、physical+action、physical+action+routing、hidden state。
- 只有 routing 出现稳定增量后，再做 expert swap、route clamp 或 snapshot replay 干预。

## 来源

- [物理行为谱](../failure-behavior-taxonomy/report.md)
- [全部失败事件账本](../failure-event-audit/report.md)
- [MoE 时序与固定前缀检验](../failure-moe-signatures/report.md)
- [独立方法复核](../failure-type-method-review/report.md)
