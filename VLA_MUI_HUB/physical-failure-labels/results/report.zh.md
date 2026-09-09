# LIBERO 物理失败标注报告

## 覆盖范围

本次标注覆盖 `89` 个完整 run、`36098` 个 episode；其中成功 `34656`，失败 `1442`。
失败样本共恢复并检查 `56920` 个原始 MuJoCo 控制状态。原 rollout 目录未被修改。

| suite | episodes | success | failure |
|---|---|---|---|
| libero_goal | 9024 | 8774 | 250 |
| libero_long | 8512 | 7755 | 757 |
| libero_object | 8000 | 7919 | 81 |
| libero_spatial | 10562 | 10208 | 354 |

## 主失败模式

| primary_failure_reason | episodes |
|---|---|
| object_released_or_dropped_before_goal | 587 |
| stable_grasp_not_observed | 442 |
| object_moved_but_goal_unmet | 200 |
| goal_predicate_regressed | 65 |
| timeout_while_holding_target | 57 |
| object_released_outside_goal | 48 |
| approached_target_without_observed_contact | 32 |
| mechanism_threshold_not_reached | 7 |
| no_meaningful_target_progress | 4 |

这些标签描述可观测的物理失败模式，不声称已经证明模型内部的因果根因。每条失败记录同时保存原始 BDDL 谓词、对象位移、末端距离、接触、双指稳定抓取、关节运动、证据依据和置信度。

## 物理一致性

- 恢复状态中完整目标意外为真的次数：`0`。
- BDDL 原子谓词与环境 `check_success` 不一致次数：`0`。
- dense replay：按动作步数、outcome、qpos 与记录的机器人状态判定，`31/37` 通过；其中 `28` 条完整状态也在容差内，outcome 分叉 `1` 条。
- 最大完整状态 / qpos / qvel / 机器人状态误差为 `3.300404082644` / `0.966478058749` / `3.300404082644` / `0.737630605698`（qpos 与机器人状态容差 `0.005`）。

## 使用边界

- `sim_state` 是每 10 个动作记录一次的原始控制检查点，失败 episode 的最后一个检查点通常距离终态 10 步；确切数值保存在每条记录的 `unobserved_terminal_tail_actions`。
- 小于一个控制区间的瞬时接触或抓取可能漏检，因此基于“从未接触/抓取”的归因最多给中等置信度。
- float32 动作的独立重放在接触敏感轨迹上可能分叉；`dense_replay_audit.jsonl` 显式保留分量误差与 outcome 是否一致。失败原因使用的是原始轨迹保存状态，而不是分叉后的重放状态。
- 成功标签来自采集时逐动作运行的 LIBERO `check_success`；失败标签额外经过离线状态恢复物理检查。
- CALVIN 未纳入：现有截断探针不具备兼容的 per-episode LIBERO `sim_state` 契约。

## 文件

- `episodes.csv`：所有 episode 的轻量索引。
- `episodes.jsonl`：所有 episode；成功记录保留在线物理 outcome，失败记录包含完整证据。
- `failures.jsonl`：仅失败 episode 的完整标注。
- `dense_replay_audit.jsonl`：逐动作重放审计。
- `summary.json`：机器可读汇总与阈值。
