# MoE dynamics v2 prospective 在线审计

固定 24 条：10 成功、14 失败。v2 严格目标漏抓报警 5/7，其他失败 3/7，成功误报 1/10。全部失败召回 8/14。旧 v1 规则在完全相同轨迹上的结果保存在 `v1_paired_same_trajectories`。

阈值仅由旧健康 reference 的 leave-one-out 序列上界确定。物理量只在 rollout 完成后用于本审计；特征设计受上一批 GPU5 posthoc 结果启发，所以本批才是 v2 的第一次 prospective test。

| ep/init | outcome | family | generic | miss q | alarm q |
| --- | --- | --- | --- | --- | --- |
| 0/3 | failure | other_failure | subtask_undo | [] | [] |
| 1/7 | success | success | success | [] | [] |
| 2/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [13,33] | [13] |
| 3/0 | failure | other_failure | single_subtask_omission | [] | [] |
| 4/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [15,34] | [14,15,24,34] |
| 5/0 | failure | other_failure | single_subtask_omission | [] | [] |
| 6/3 | failure | other_failure | loop_or_cycling | [] | [30] |
| 7/7 | success | success | success | [] | [] |
| 8/3 | failure | target_missed_grasp_proxy | single_subtask_omission | [41] | [31,32,33,34,38,41,42,43] |
| 9/20 | success | success | success | [] | [35] |
| 10/7 | success | success | success | [] | [] |
| 11/0 | success | success | success | [] | [] |
| 12/0 | success | success | success | [] | [] |
| 13/3 | failure | other_failure | loop_or_cycling | [] | [50,51] |
| 14/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [13,33] | [] |
| 15/7 | success | success | success | [] | [] |
| 16/0 | success | success | success | [] | [] |
| 17/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [36] | [] |
| 18/7 | success | success | success | [] | [] |
| 19/3 | failure | target_missed_grasp_proxy | loop_or_cycling | [35] | [12,33,36,37,38,39] |
| 20/7 | failure | other_failure | subtask_undo | [] | [] |
| 21/0 | success | success | success | [] | [] |
| 22/20 | failure | target_missed_grasp_proxy | loop_or_cycling | [13,33] | [13] |
| 23/3 | failure | other_failure | drop_or_regrasp | [] | [12] |
