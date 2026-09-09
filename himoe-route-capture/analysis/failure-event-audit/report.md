# 全部失败 rollout 的物理事件审计

## 直接结论

本审计覆盖全部 307 条失败。按最后 25% query 的 EEF 运动，121 条仍持续运动，45 条间歇运动，141 条进入物理静滞。因此，失败并不等同于停滞。

缓存能确认的是物体/EEF/抽屉的运动学状态：223 条终点仍保留部分子目标，28 条曾达到目标代理后又失去，18 条出现超出成功 99% 基线的非目标物位移。另有 31 条满足保守运输丢失代理，20 条满足“仍活跃且对未完成目标重复靠近闭爪”的 retry 代理。

不能确认的是接触、碰撞、真实抓取力和视觉误认。下文的 `drop/retry/interference/wrong-object` 都明确保留 `proxy`，不能改写成真值事件。

## 事件账本

| task | failures | active/intermittent/stasis | partial final | subgoal loss | drop proxy | active retry | controller regression | distractor displacement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `open_the_top_drawer_and_put_the_bowl_inside` | 42 | 27/3/12 | 23 | 19 | 22 | 1 | 0 | 0 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 216 | 79/28/109 | 200 | 9 | 2 | 19 | 39 | 0 |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 12 | 7/5/0 | 0 | 0 | 1 | 0 | 0 | 10 |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 37 | 8/9/20 | 0 | 0 | 6 | 0 | 0 | 8 |

这些列是多标签，可以重叠，不能横向相加。

## 物理模式摘要

`primary_physical_pattern` 只是为了浏览账本而设置的任务内优先级摘要；正式结论应使用上面的多标签事件列。

| task | primary pattern | n |
|---|---|---:|
| `open_the_top_drawer_and_put_the_bowl_inside` | `bowl_transport_loss_proxy` | 22 |
| `open_the_top_drawer_and_put_the_bowl_inside` | `drawer_subgoal_regression` | 17 |
| `open_the_top_drawer_and_put_the_bowl_inside` | `ongoing_bowl_placement_failure` | 3 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | `active_at_unfinished_pot1` | 45 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | `other_long_incomplete` | 14 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | `placed_object_goal_loss` | 9 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | `return_to_completed_pot2_proxy` | 39 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | `stalled_at_unfinished_pot1` | 109 |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | `bowl_transport_loss_proxy` | 1 |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | `ramekin_displacement_interference_proxy` | 10 |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | `transported_bowl_not_placed` | 1 |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | `bowl_never_transported` | 27 |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | `bowl_transport_loss_proxy` | 6 |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | `empty_goal_side_close_proxy` | 2 |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | `transported_bowl_not_placed` | 2 |

## 四个失败任务在做什么

**Top drawer + bowl（42）**：所有失败都曾打开抽屉并抬起碗。其中 19 条终点失去 drawer-open 子目标，22 条有保守运输丢失代理；终端仍 active 的有 27 条。

**Two moka pots（216）**：主体是完成 pot 2 后未完成 pot 1。200 条终点保留部分完成，39 条在接近 pot 1 后回到 pot 2 侧，9 条把物体带到 goal 邻域后又失去；只有 19 条满足 active retry 代理。

**Bowl on ramekin（12）**：目标碗均出现运输证据，但 10 条同时让非目标 ramekin 产生远超成功基线的位移。这确认了非目标物被明显扰动，但没有 contact 字段，原因只能称 interference proxy。

**Bowl on stove（37）**：29 条没有保守运输证据；5 条在目标碗仍远时于 plate 一侧闭爪，6 条有运输丢失代理。

## 判据与证据等级

- 可确认运动学：目标是否进入任一成功终点 5 cm 邻域、是否在终点保留、抽屉开度、EEF 路径、非目标物位移。
- 强代理：闭合夹爪附近的物体与 EEF 同动、相对运动残差受限；随后下降分离记为 drop proxy。
- 弱代理：重复靠近并闭爪记为 retry proxy；靠近 goal receptacle 但目标仍远时闭爪记为 empty-goal-side proxy。
- 无法判断：真实 contact/collision、夹爪受力、遮挡、视觉误识别和动作 chunk 内事件时刻。缓存没有 RGB、视频、contact 或 force。

持续运动定义为最后 25% query transition 中，EEF 平均位移至少 1 cm/query，且至少一半 transition 超过 1 cm。静滞定义为平均不足 5 mm/query 且超过 1 cm 的比例不高于 25%；其余为 intermittent。

## 产物

- `episode_events.csv`: 307 条失败的完整多轴事件账本。
- `representative_episodes.csv`: 每个任务内摘要模式的中心代表 episode。
- `summary.json`: 阈值、任务计数和证据限制。
