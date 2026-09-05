# MoE-only 在线报警：失败类型事后审计

## 核心结论

三批共 40 条在线轨迹（18 成功、22 失败）。失败不能视为同一种 Trap：9 条满足保守的 `missed-grasp-then-departure` 运动学代理，另外 13 条表现为其他失败。

正式规则是连续两次 raw reject（persistence=2）。合并三批后：目标漏抓型 1/9、其他失败 2/13、成功误报 0/18。GPU5 扩展批次首次出现正式报警，但目标型召回仍低，不能声称已经可用。

单次 raw reject 在目标漏抓型上是 7/9，成功轨迹也有 7/18 出现；所以它只是诊断信号，不能直接当成报警结果。

## 分轨迹结果

| run | ep/init | outcome | family | generic | miss q | raw q | alarm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preliminary | 0/3 | failure | target_missed_grasp_proxy | loop_or_cycling | [37] | [46] | no |
| preliminary | 1/0 | success | success | success | [] | [34] | no |
| preliminary | 2/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [14,33] | [36] | no |
| preliminary | 3/7 | success | success | success | [] | [] | no |
| preliminary | 4/7 | success | success | success | [] | [14,22] | no |
| preliminary | 5/20 | success | success | success | [] | [] | no |
| preliminary | 6/0 | success | success | success | [] | [10] | no |
| preliminary | 7/3 | failure | other_failure | single_subtask_omission | [] | [6] | no |
| heldout | 0/7 | success | success | success | [] | [] | no |
| heldout | 1/3 | failure | target_missed_grasp_proxy | loop_or_cycling | [34] | [7,10,12,35,43] | no |
| heldout | 2/20 | failure | other_failure | subtask_undo | [] | [10] | no |
| heldout | 3/0 | failure | other_failure | single_subtask_omission | [] | [] | no |
| heldout | 4/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [14,34] | [10] | no |
| heldout | 5/7 | success | success | success | [] | [36] | no |
| heldout | 6/3 | failure | other_failure | single_subtask_omission | [] | [10,12] | no |
| heldout | 7/0 | success | success | success | [] | [36] | no |
| gpu5_extended | 0/7 | success | success | success | [] | [36] | no |
| gpu5_extended | 1/20 | failure | other_failure | drop_or_regrasp | [] | [] | no |
| gpu5_extended | 2/0 | success | success | success | [] | [] | no |
| gpu5_extended | 3/3 | failure | other_failure | loop_or_cycling | [] | [2,7,10,15] | no |
| gpu5_extended | 4/3 | failure | target_missed_grasp_proxy | loop_or_cycling | [48] | [10,31] | no |
| gpu5_extended | 5/7 | success | success | success | [] | [] | no |
| gpu5_extended | 6/0 | success | success | success | [] | [33,37] | no |
| gpu5_extended | 7/20 | failure | other_failure | drop_or_regrasp | [] | [] | no |
| gpu5_extended | 8/7 | success | success | success | [] | [] | no |
| gpu5_extended | 9/0 | failure | other_failure | single_subtask_omission | [] | [] | no |
| gpu5_extended | 10/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [15,35] | [37] | no |
| gpu5_extended | 11/3 | failure | other_failure | loop_or_cycling | [] | [2,5,10,11,27] | yes |
| gpu5_extended | 12/7 | success | success | success | [] | [] | no |
| gpu5_extended | 13/0 | success | success | success | [] | [] | no |
| gpu5_extended | 14/20 | failure | target_missed_grasp_proxy | subtask_undo | [37] | [38,43,44] | yes |
| gpu5_extended | 15/3 | failure | other_failure | loop_or_cycling | [] | [2,7,10,30] | no |
| gpu5_extended | 16/7 | success | success | success | [] | [] | no |
| gpu5_extended | 17/20 | failure | other_failure | subtask_undo | [] | [9,19,41] | no |
| gpu5_extended | 18/0 | success | success | success | [] | [] | no |
| gpu5_extended | 19/3 | failure | other_failure | active_retry | [] | [7,10,12] | no |
| gpu5_extended | 20/7 | success | success | success | [] | [] | no |
| gpu5_extended | 21/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [14,34] | [] | no |
| gpu5_extended | 22/3 | failure | other_failure | loop_or_cycling | [] | [2,7,34,35] | yes |
| gpu5_extended | 23/0 | failure | target_missed_grasp_proxy | loop_or_cycling | [45] | [] | no |

## 边界

- 在线 selector 只读 HB MoE routing 和成功 routing reference；不读物理距离、机器人状态、动作值、reward 或 success。
- 本审计中的物理位置只在 rollout 完成后用于区分失败类型，不参与阈值或在线决策。
- 三批客户端与服务端保存的完整 HB 路由均逐元素一致；hook failure 均为 0。
- `missed-grasp-then-departure` 是运动学代理：闭爪时靠近目标，随后 EEF 离开而目标近似静止。数据没有接触力，不能把它升级成真实 contact 或模型 belief 的直接观测。
- 通用 taxonomy 描述最终行为表型；例如漏抓造成后续 loop 时，`loop_or_cycling` 是结果，不一定是根因。
