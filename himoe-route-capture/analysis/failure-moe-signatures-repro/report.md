# 失败类型的 MoE 时序信号

## 主结论（选择性 fixed-prefix case-control）

**在这套由未来终局筛出的选择性协议下，没有看到 routing 超过 physical+action 的增量；这不能泛化成“routing 普遍没有信息”。**

主问题只比较同一 long moka-pot 任务中的两种物理终局：宽松的 suffix-low-motion stasis proxy 是完成 pot 2、转向 pot 1 后进入持续低 EEF 运动；EEF return-to-pot2-side 是靠近 pot 1 后，EEF 又回到 pot 2 一侧。标签和 onset 只由机器人/物体轨迹定义，不使用路由投票。内部 CSV 为兼容已有代码仍保留 `stagnation_core` / `active_return` 键，显示名不把 EEF 回返解释成策略或 controller 回返。

所有输入统一截到 pot 2 首次到位后的第 8 个 query。最终纳入 158 条（stasis proxy 119，EEF return-to-pot2-side 39，13 个 init）；每条 cut 都早于 query-boundary onset，最近相隔 1 个 query。模型按 init 留一，held-out init 未参与该折的 scaler 或 classifier 拟合。

| feature block | ROC AUC | AP | balanced acc @ 0.5 | dims |
|---|---:|---:|---:|---:|
| physical | 0.439 | 0.262 | 0.443 | 32 |
| action | 0.412 | 0.260 | 0.444 | 28 |
| physical + action | 0.457 | 0.301 | 0.435 | 60 |
| routing | 0.392 | 0.204 | 0.427 | 40 |
| physical + action + routing | 0.436 | 0.251 | 0.469 | 100 |

在 physical+action 上加入 routing 的 OOF AUC 增量是 -0.020，按 init 聚类 bootstrap 95% CI 为 [-0.114, +0.049]。因此只能说“此选择性协议下未见增量”，不能说 routing 产生负作用，也不能说它在其他任务、风险集或时间点普遍无信息。

## Fixed-prefix 方法风险

没有发现直接把尾部路由放进特征的硬泄漏：统计只取 `q0:q0+8`，recurrence 只查当前和过去，imputer/scaler/model 都在训练折内拟合，元数据和 onset 没有进模型。但独立方法复核指出以下限制：

1. pot1/pot2 success-goal reference 由全部 296 条成功 rollout 的终点构造，包含 held-out init；因此上游 `q0`、goal-distance 特征和物理标签是 transductive 的。真正 LOGO 需要环境真值 goal 或每折重算 reference。
2. cohort 由未来筛选：成功与 54 条 other-long failure 被排除，线上在 cut 时并不知道 rollout 会落入这两类。这里是前缀上的回顾性 case-control，不是全体 rollout 在线预警。
3. lead time 严重不匹配：stasis proxy 的 onset-cut 均值/中位数为 2.40/2 queries；EEF return-to-pot2-side 为 16.15/15，最小仍为 9。两类不是同一预测距离上的风险集。
4. 13 个 init 中 6 个只有一个类别；joint 有 100 维但只有 158 条。即使 LOGO 阻止精确 init 记忆，小组数、类别组成和几何偏移仍限制功效。
5. 最短 1-query 间隔只表示在缓存的 query 边界尚未观察到 onset；cut 处生成的 action chunk 可能正造成下一边界的低运动，不能声称 chunk 内物理事件前预警。

## 稳定性审计

| feature block | leave-one-init-out | leave-one-seed-out | leave-one-init-out, 仅双方都有样本的 init |
|---|---:|---:|---:|
| physical + action | 0.457 | 0.822 | 0.387 |
| routing | 0.392 | 0.651 | 0.327 |
| joint | 0.436 | 0.798 | 0.349 |

按 seed 留一仍会在训练中看到同一 init 的其他 seed；它的较高分数不能替代跨 init 验证。双方都有样本的 7 个 init 上共 109 条，跨 init 结果仍低，说明主表的负结果不是只由单标签 init 造成。两种划分中 routing 对 physical+action 的 AUC 增量分别为 -0.020 和 -0.023。

## 相对物理 onset 的时序

每条曲线先减去本 episode 的 `[-10,-7]` reference，再按 pooled reference SD 标准化；表内是 matched-init 的 EEF return-to-pot2-side 减 stasis-proxy 动态差。`lead`=`[-6,-2]`，`sync`=`[-1,+1]`，`after`=`[+2,+6]`。只有 |effect|≥0.2 且 init-bootstrap CI 不跨 0 才标为 separation。

| routing signal | earliest separation | lead | sync | after |
|---|---|---:|---:|---:|
| route_speed | lead_-6_-2 | +3.66 | +3.83 | +5.69 |
| route_recurrence_advantage | lead_-6_-2 | +2.27 | -0.88 | -1.06 |
| route_recurrence_lag_fraction | after_+2_+6 | +0.44 | +0.33 | +2.79 |
| route_anchor_advantage | lead_-6_-2 | +2.85 | +2.83 | +4.79 |
| route_entropy | lead_-6_-2 | -3.76 | -5.83 | -4.74 |
| route_top1_mass | lead_-6_-2 | +3.04 | +3.68 | +3.26 |
| expert_speed | lead_-6_-2 | +2.08 | +2.73 | +5.04 |
| expert_entropy | sync_-1_+1 | -0.64 | -2.58 | -0.73 |
| expert_top1_mass | sync_-1_+1 | -0.32 | +1.06 | -0.33 |
| route_layer_synchrony | lead_-6_-2 | -1.19 | -3.32 | -4.41 |

停滞 onset 定义在低运动 transition 开始前的 query，因此 onset query 的 route/action 是对下一 chunk 的计划；EEF return onset 则是第一次已经观察到 EEF 重新更靠 pot 2 的 query，因此同 query routing 同时含有物理状态读出。成功曲线按首次进入 pot 1 的 13 cm 邻域对齐，只是行为参照，不把它称作失败 onset。

两种事件的 lead time 和物理定义并不相同，因此这张对齐表只能描述各自 onset 附近的形态，不能把大量 `lead` 自动解释成共同早期预警。实际上 physical/action 也分别有 5/8 和 4/7 个信号在 lead 窗分开，说明 routing 很可能同时跟随已不同的物理阶段和待执行动作。局部 lead 最多是下一动作前验候选；sync/after 更像状态读出。结合统一 cut 上不跨 init 泛化，不能升级为通用 MoE trap 机制。

## 物理类型覆盖

- suffix-low-motion stasis proxy（内部键 `stagnation_core`）: 123。完成 pot 2、接近 pot 1，随后剩余 EEF transitions 中至少 80% 小于 1 cm/query，且 p80 小于 1.5 cm/query。独立 event audit 的严格交集是 109 条 `stalled_at_unfinished_pot1`；123 不能和 109 混称同一个 core。
- EEF return-to-pot2-side（内部键 `active_return`）: 39。完成 pot 2、进入 pot 1 的 13 cm 邻域，之后首次重新变成离 pot 2 更近；它只确认 EEF 空间回返，不确认内部策略、controller 或 routing reset。
- `success`: 296。这里只作为下一阶段的事件对照，不参与上面的二分类主结果。
- `other_long_failure`: 54。不满足上述干净定义，主结果不强行归类。

## 全 307 失败的描述表

下面使用完整 episode 的 10-bin 相对相位，只回答“终局类型伴随什么动态”，不能用于 pre-onset 预测。route/action speed 除以各 episode 中位数；hard overlap 是相邻相位 hard-expert occupancy 的 `1-Hellinger`；entropy 是归一化 hard occupancy entropy。

| task (short) | physical pattern | n | route speed E/M/L | hard overlap L | hard entropy L | action speed L |
|---|---|---:|---|---:|---:|---:|
| open_the_top_drawer_an | bowl_transport_loss_proxy | 22 | 1.47/0.99/0.79 | 0.751 | 0.849 | 0.26 |
| open_the_top_drawer_an | drawer_subgoal_regression | 17 | 1.64/1.00/0.69 | 0.785 | 0.847 | 0.13 |
| KITCHEN_SCENE8_put_bot | stalled_at_unfinished_pot1 | 109 | 1.76/0.93/0.94 | 0.798 | 0.865 | 0.21 |
| KITCHEN_SCENE8_put_bot | EEF return-to-pot2-side proxy | 39 | 0.81/1.01/1.08 | 0.601 | 0.861 | 0.77 |
| pick_up_the_black_bowl | ramekin-displacement outlier | 10 | 1.53/1.02/0.79 | 0.794 | 0.812 | 0.22 |
| pick_up_the_black_bowl | bowl_transport_loss_proxy | 1 | 0.99/1.05/0.88 | 0.721 | 0.811 | 0.78 |
| pick_up_the_black_bowl | no-transport-proxy-detected | 27 | 1.69/1.16/0.71 | 0.791 | 0.794 | 0.10 |
| pick_up_the_black_bowl | bowl_transport_loss_proxy | 6 | 1.49/1.08/0.78 | 0.775 | 0.806 | 0.15 |

这一层的类型来自独立物理账本：top-drawer 分 transport-loss proxy / drawer regression，long 分 stall / EEF return-to-pot2-side，ramekin 只称 displacement outlier，stove 分 no-transport-proxy-detected / transport-loss proxy。缓存没有 RGB、contact 或 force，不能把这些代理升级为真实碰撞、抓取、掉落或视觉混淆。小组只报中位数，不做 AUC 或机制宣称。

## 5 条跨任务孤立失败

每条只和同 task、同 init 的成功 sibling 比整段路由百分位；1 表示高于所有 sibling。

| task (short) | episode | success siblings | late route-speed pct | late hard-overlap pct | late hard-entropy pct |
|---|---:|---:|---:|---:|---:|
| open_the_top_drawer_and_ | 179 | 5 | 0.40 | 1.00 | 0.80 |
| pick_up_the_black_bowl_o | 449 | 28 | 0.00 | 0.00 | 0.00 |
| pick_up_the_black_bowl_o | 76 | 31 | 0.00 | 0.00 | 0.00 |
| pick_up_the_black_bowl_o | 106 | 30 | 0.43 | 0.00 | 0.00 |
| pick_up_the_black_bowl_o | 122 | 30 | 0.93 | 0.00 | 0.00 |

## 如何解释

- `physical` 是当前状态读出；`action` 是已经生成但尚未执行完的下一 chunk 计划。
- `routing` 若只在 onset 同步或之后分开，最多说明内部状态读出了已发生的物理变化；lead 分开也必须先排除 lead-time、物理状态和 action 差异。
- 固定 cut 上 routing 超过 physical+action，才是“对未来恢复动作有额外预后信息”的候选证据；即使如此仍只能叫预测关联。
- 机制结论还需要对 route/expert 或 action 做受控干预，本报告不作因果宣称。

## 产物

- `episode_results.csv`: 逐 episode 物理类型、onset、cut 和 OOF 分数。
- `prefix_features.csv.gz`: cut 前压缩特征，便于复核；不含原始路由张量。
- `query_descriptors.npz`: 每 query 的标量 physical/action/routing 描述符。
- `event_window_effects.csv` / `event_aligned.png`: onset 前、同步、后的 matched-init 动态差。
- `failure_pattern_descriptors.csv` / `failure_pattern_summary.csv`: 全 307 失败的物理 taxonomy 与整段动态。
- `isolated_other_task_failures.csv`: 5 条孤立失败相对成功 sibling 的百分位。
- `summary.json`: 计数、协议和指标。
- `prospective_auc.png`: 主结果图。
