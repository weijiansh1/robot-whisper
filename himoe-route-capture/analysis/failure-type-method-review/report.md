# 失败类型方法复核

## 审稿结论

三套结果的数据主键和总失败数一致，但标签回答的问题不同。可以合并的是两类长任务运动学结构：109 条严格的 pot1 侧静止核心，以及 39 条接近 pot1 后回到 pot2 侧的 EEF 回返。不能合并的是 `stasis=141/166/123/109`、`retry=0/20`、`drop=31/58` 这些同名或近义数字。

固定前缀 MoE 实验没有发现直接的尾部特征泄漏：模型列只取 `q0:q0+8`，递归距离只查询当前及过去，缩放和分类器都在 LOGO 训练折内拟合，`task/init/episode_length/onset` 元数据也没有进入模型。但是，上游 goal 参考使用了全部成功轨迹，入选 cohort 由未来终局决定，且两类事件距 cut 的时间非常不对称。因此它是一个前缀上的回顾性 case-control 实验，还不是可部署的全体 rollout 在线预警实验。

## 四个 stasis 数字

| 数字 | 范围 | 实际定义 | 与其他口径的关系 |
|---:|---|---|---|
| 141 | 全部 307 failures | 最后 25% query 的 EEF 平均步长 `<5 mm`，且 `>1 cm` 的比例不超过 25% | 与 taxonomy 166 相交 136；其中 long task 109 |
| 166 | 全部 307 failures | 前 90% 内出现一段 EEF、任务物体、夹爪同时低变化的最长连续段，并超过同 task/init 成功 q95 | 30 条后来又动起来；另漏掉 5 条末段 EEF 静止但物体/夹爪或连续段规则不满足的轨迹 |
| 123 | 仅 long task | pot2-only、终点在 pot1 侧、曾接近 pot1，且从某个 onset 到结尾至少 80% EEF 步长 `<1 cm`、p80 `<1.5 cm` | 包含严格 109，另含 14 条按末段绝对规则属于 active/intermittent 的轨迹 |
| 109 | 仅 long task | event audit 中 `late_motion_class=stasis` 且停在未完成 pot1 的 primary 摘要 | 109/109 同时属于 taxonomy stagnation 和 MoE `stagnation_core`，是可复用的严格交集 |

所以 166 是“早中期曾长期全系统静止”，141 是“末段 EEF 静止”，123 是“特定任务阶段下的宽松 suffix 低运动”，109 才是三套定义共同支持的严格 long-task 核心。四个数字不能平均、相减后解释成新失败类，也不能都简称为 timeout stasis。

## fixed-prefix 风险

### 没有发现的硬泄漏

- 前缀统计只使用 pot2 首次到位 `q0` 至 `q0+8`，包含 9 个 query；所有纳入样本满足 `cut < physical_onset`。
- routing recurrence 在每个 query 只与更早 query 比较。虽然先算了完整 pairwise 距离矩阵，但前缀行本身不依赖未来行。
- `SimpleImputer`、`RobustScaler` 和 logistic regression 位于每个 `LeaveOneGroupOut(init_state_id)` 训练 pipeline 内。
- 主实验固定为同一个 long task，task id、init id、seed、episode length、onset 和终局标签均未作为特征。

### 仍需修正的方法风险

1. **上游参考是 transductive 的。** pot1/pot2 goal 坐标由全部 296 条成功轨迹的终点均值构造，包含 held-out init 的成功 siblings；`q0` 和物理标签也继承这一参考。因此“测试 init 未参与”只对 scaler/model fitting 成立。应改用环境真值 goal，或在每个 LOGO fold 内重算成功参考。
2. **cohort 由未来筛选。** 只保留最终成为 `stagnation_core` 或 `active_return` 且 onset 晚于 cut 的失败。线上在 cut 时并不知道样本属于该 cohort；成功和 54 条 `other_long_failure` 被排除。当前结果只能称两种未来终局的条件判别。
3. **lead time 不匹配。** `stagnation_core` 的 onset-cut 间隔均值 2.40 query、中位数为 2；`active_return` 均值 16.15 query、中位数为 15，最小也为 9。两类标签的事件定义天然产生不同预测距离。应固定 lead time，或做 landmark/survival 风险集比较。
4. **init 分布仍很偏。** 158 条样本仅覆盖 13 个 init；其中 6 个 init 只有一个类别，只有 7 个同时包含两类。LOGO 阻止精确 init 记忆，但不能消除小组数、类别组成和几何状态差异带来的分布偏移。应报告双类别 init 的组内结果，并做按 init 匹配或条件置换。
5. **最短 1-query 间隔不等于物理事件前。** 缓存只有 query 边界；cut 处已生成的 action chunk 可能正是下一边界低运动 onset 的原因。它仍是“观测 onset 前”，但不能称 chunk 内物理错误发生前。
6. **统计功效有限。** `joint` 有 100 维而 cohort 只有 158 条；路由增量 AUC 为 -0.020，95% CI `[-0.114,+0.049]`。可合并结论只有“该协议下未见 routing 增量”，不能推出 routing 普遍无信息或产生负作用。

## retry 为什么是 0 和 20

这不是数据冲突，而是命名碰撞。

| 来源 | 定义 | failures |
|---|---|---:|
| behavior taxonomy `active_retry` | 后半程 EEF 速度至少达到同 task/init 成功 q25，同时进展差、存在反向/夹爪翻转/重抬等重复控制，并且不是 stagnation | 0 |
| event audit `active_retry_proxy` | 最后 25% 满足绝对 active 阈值，且对某个未完成目标至少出现两段“靠近 + 闭爪命令” | 20 |

逐条交叉后，20 条 event retry 全部通过“进展差”，也全部不属于 taxonomy stagnation，但 20/20 都没有达到成功 q25 的后半程速度。因此 taxonomy 的第一个门槛把它们全部排除。建议把前者改称 `success-speed low-progress repeated-control`，后者改称 `active repeated target-near-close proxy`；两者都不应直接写成“机器人确认重试”。另外，39 条 `active_return` 中 event retry 为 0，说明“回到 pot2 一侧”和“重复靠近未完成目标闭爪”也是两种不同现象。

## 可合并的证据

- **严格静止核心：109 条。** 三套口径一致支持它们在完成 pot2 后停在未完成 pot1 一侧。
- **EEF 回到 pot2 侧：39 条。** MoE `active_return` 与 event `controller_subgoal_regression_proxy` 39/39 完全一致。它确认的是 EEF 空间回返，不确认内部 controller、routing reset 或重新抓取。
- **运输/高度丢失代理：31 条保守子集。** event 的 31 条 `drop_proxy` 全部落在 taxonomy 的 58 条 `regrasp_or_drop` 内；后者另含 27 条。可报告 31 条强代理，不能把 58 条都叫真实 drop。
- **目标距离回退：45 条共同支持。** taxonomy `goal_regression=49` 与 event `progress_regression_outlier=46` 相交 45。可合并成存在稳定的 goal-distance regression 现象，但两个总数仍应保留各自定义。
- **MoE 增量为负结果。** 在 158 条选择性 cohort、固定 cut 和 LOGO 下，routing 没有超过 physical+action；这是当前最稳妥的路由结论。

## 不能直接合并的代理

| 当前名字 | 风险 | 建议表述 |
|---|---|---|
| `regrasp_or_drop` | z 方向重复 lift/height loss 不证明抓取或掉落 | `repeat-lift_or_offgoal-height-loss proxy` |
| `controller_subgoal_regression_proxy` | 只观察到 EEF 从 pot1 邻域回到更靠近 pot2 的位置 | `eef-return-to-pot2-side proxy` |
| `active_return` | 不证明“策略回返”或 MoE recurrence | `eef-return-to-pot2-side` |
| `ramekin_displacement_interference_proxy` | 位移是真实运动学，interference/contact 是原因推断 | `ramekin-displacement outlier` |
| `bowl_never_transported` | 只是不满足保守 EEF-object 共动规则 | `no-transport-proxy-detected` |
| `empty_goal_side_close_proxy` | 闭爪命令在 plate 侧，不证明夹爪为空 | `goal-side-close-with-target-far proxy` |
| `subtask_undo` / `placed_object_goal_loss` | goal 是成功终点邻域代理，不是环境 predicate | `success-goal-neighborhood loss` |
| `wrong_object_close_proxy` | 规则只能检测“目标远时靠近非目标物闭爪”；本次计数还是 0 | 保留为负结果，不作 wrong-target 结论 |

缓存没有 RGB、contact、force 或 chunk 内稠密状态。任何 `collision/contact/visual confusion/true grasp/true drop/wrong target` 都不能从这三套结果升级为真值语义。

## 建议的统一报告口径

1. 先报告 109 条严格 pot1-side stasis 和 39 条 EEF return-to-pot2-side，二者是当前最稳定的 long-task结构。
2. 其余现象保持多标签账本，不再强行合成互斥失败类型；每个数字同时给出时间窗、任务范围和阈值。
3. MoE 结论限定为：“在未来筛出的两类 long-task failure 中，固定前缀 routing 对 physical+action 没有可检测增量。”
4. 下一轮实验用 fold 内 goal reference、相同 lead time、同 init/相近物理状态风险集，并加入 success 与 other failure，才检验在线 recovery prognosis。
