# VLA_MUI_HUB 跨任务 MoE Trap 表型审计

## 直接结论

本轮只使用 `right-50x8-20260903` 中元数据标记 complete 的任务：37 个任务、14800 条轨迹、487 条失败。正在写入的续批和不完整任务均未纳入。

物理标签先于 routing 冻结，并由同 task/init 的成功 siblings 校准；路由头则对 held-out init 只使用同任务其他 init 的成功轨迹校准。这些仍是 query-boundary 运动学代理，不是视频/contact 真值。

**好结果：** stagnation 明确对应稳定计算锁死。固定 `lock_in` 头在其他失败对照上的方向 AUC 为 0.898；q95 下检出 67.6%，成功误报 5.3%。单独的 flat+narrow 现象存在，但远弱于跨 query 的低 mobility + 高 recurrence。

**有限结果：** gripper cycling 符合 routing instability，固定头方向 AUC 0.711；q95 检出 37.0%，成功误报 5.4%。它能提供候选，但还不是高召回 detector。

**坏结果：** goal regression、subtask undo 和 regrasp/drop 虽在失败内部可见部分 state-action gap，但固定 feedback 头在成功校准的 q95 下几乎不触发；lag periodicity 也没有支持预期的阶段往返方向。当前数据不支持把它们宣称为可用的 train-free 报警头。

**总体二元 Trap 判断：双头 `max` 不好。** 在完全相同的 routing 输入和同一成功误报预算下，等权 mean 单标量检出 66.5%（FPR 5.2%，precision 30.4%），双头 `max` 只检出 57.9%（FPR 5.5%，precision 26.4%）。这里评估的是固定阈值后的 ACCEPT/ALARM，不是轨迹排序。

按 37 个任务成簇 bootstrap，双头相对信息匹配 mean 的召回差为 -8.6% [-17.3%, -0.7%]，因此下降不能用抽样波动解释。双头相对只看 `lock_in` 仍增加 +4.7% [+1.5%, +11.4%]，说明新增 instability 信号有用，但不能证明双头结构更好。四头相对双头为 -1.2% [-2.6%, +1.4%]。

若每个头各自用 q95 后直接 OR，双头检出率可到 59.8%，但 FPR 同时升至 10.7%；这不是同预算提升。两个分数仍可保留给后续分型修正，但当前数据只验证了表型分流，没有验证任何修正动作能提高成功率。

## 物理表型库存

| label | failure_n | failure_rate | success_n | success_rate | failure_tasks |
| --- | --- | --- | --- | --- | --- |
| stagnation | 136 | 0.2793 | 36 | 0.0025 | 15 |
| active_retry | 0 | 0.0000 | 136 | 0.0095 | 0 |
| eef_oscillation | 0 | 0.0000 | 277 | 0.0194 | 0 |
| gripper_cycling | 81 | 0.1663 | 174 | 0.0122 | 23 |
| goal_regression | 104 | 0.2136 | 4 | 0.0003 | 9 |
| goal_approach_leave | 10 | 0.0205 | 0 | 0.0000 | 5 |
| subtask_undo | 37 | 0.0760 | 2 | 0.0001 | 4 |
| regrasp_or_drop | 164 | 0.3368 | 89 | 0.0062 | 22 |

## Train-free 多头的 episode-level 报警

每个头的分量和方向固定，不用失败标签拟合权重；对每个 held-out init，只用同任务其他 init 的成功轨迹做经验分位数和 q95 阈值。这是 50%--90% 相位的离线表型检查，不是在线 query 报警性能。
联合双头/四头直接取分位数头分数的最大值，并用 held-out init 之外的成功轨迹重新校准联合 q95；没有用失败标签学习组合权重。

### 二元 Trap 判断

| detector | tp | fp | tpr | fpr | precision | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| instability | 32 | 780 | 0.0657 | 0.0545 | 0.0394 | 0.5056 |
| lock_in | 259 | 758 | 0.5318 | 0.0530 | 0.2547 | 0.7394 |
| flat_narrow_support | 40 | 750 | 0.0821 | 0.0524 | 0.0506 | 0.5149 |
| feedback_decoupling | 30 | 785 | 0.0616 | 0.0548 | 0.0368 | 0.5034 |
| dual_extremes | 282 | 785 | 0.5791 | 0.0548 | 0.2643 | 0.7621 |
| dual_mean_control | 324 | 742 | 0.6653 | 0.0518 | 0.3039 | 0.8067 |
| multi_phenotype | 276 | 808 | 0.5667 | 0.0565 | 0.2546 | 0.7551 |
| dual_per_head_q95_or | 291 | 1538 | 0.5975 | 0.1075 | 0.1591 | 0.7450 |
| multi_per_head_q95_or | 307 | 2583 | 0.6304 | 0.1805 | 0.1062 | 0.7250 |

### 公平预算配对差值

正的 `delta_tpr` 表示 candidate 在同一成功误报校准下提高失败召回；区间按 37 个任务成簇 bootstrap。

| baseline | candidate | delta_tpr | delta_tpr_ci_low | delta_tpr_ci_high | delta_fpr | delta_fpr_ci_low | delta_fpr_ci_high | failure_baseline_only | failure_candidate_only | success_baseline_only | success_candidate_only | bootstrap_unit | bootstrap_draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lock_in | dual_extremes | 0.0472 | 0.0153 | 0.1143 | 0.0019 | 0.0008 | 0.0030 | 8 | 31 | 409 | 436 | task | 2000 |
| dual_mean_control | dual_extremes | -0.0862 | -0.1728 | -0.0071 | 0.0030 | 0.0017 | 0.0042 | 67 | 25 | 596 | 639 | task | 2000 |
| dual_extremes | multi_phenotype | -0.0123 | -0.0257 | 0.0135 | 0.0016 | 0.0003 | 0.0029 | 13 | 7 | 225 | 248 | task | 2000 |

| head | target | eligible | triggered | rate |
| --- | --- | --- | --- | --- |
| instability | success_false_alarm | 14313 | 780 | 0.0545 |
| instability | all_failures | 487 | 32 | 0.0657 |
| instability | stagnation | 136 | 0 | 0.0000 |
| instability | gripper_cycling | 81 | 30 | 0.3704 |
| instability | goal_regression | 104 | 3 | 0.0288 |
| instability | goal_approach_leave | 10 | 0 | 0.0000 |
| instability | subtask_undo | 37 | 0 | 0.0000 |
| instability | regrasp_or_drop | 164 | 15 | 0.0915 |
| lock_in | success_false_alarm | 14313 | 758 | 0.0530 |
| lock_in | all_failures | 487 | 259 | 0.5318 |
| lock_in | stagnation | 136 | 92 | 0.6765 |
| lock_in | gripper_cycling | 81 | 21 | 0.2593 |
| lock_in | goal_regression | 104 | 74 | 0.7115 |
| lock_in | goal_approach_leave | 10 | 4 | 0.4000 |
| lock_in | subtask_undo | 37 | 14 | 0.3784 |
| lock_in | regrasp_or_drop | 164 | 93 | 0.5671 |
| flat_narrow_support | success_false_alarm | 14313 | 750 | 0.0524 |
| flat_narrow_support | all_failures | 487 | 40 | 0.0821 |
| flat_narrow_support | stagnation | 136 | 34 | 0.2500 |
| flat_narrow_support | gripper_cycling | 81 | 0 | 0.0000 |
| flat_narrow_support | goal_regression | 104 | 6 | 0.0577 |
| flat_narrow_support | goal_approach_leave | 10 | 0 | 0.0000 |
| flat_narrow_support | subtask_undo | 37 | 1 | 0.0270 |
| flat_narrow_support | regrasp_or_drop | 164 | 5 | 0.0305 |
| feedback_decoupling | success_false_alarm | 14313 | 785 | 0.0548 |
| feedback_decoupling | all_failures | 487 | 30 | 0.0616 |
| feedback_decoupling | stagnation | 136 | 0 | 0.0000 |
| feedback_decoupling | gripper_cycling | 81 | 18 | 0.2222 |
| feedback_decoupling | goal_regression | 104 | 0 | 0.0000 |
| feedback_decoupling | goal_approach_leave | 10 | 0 | 0.0000 |
| feedback_decoupling | subtask_undo | 37 | 0 | 0.0000 |
| feedback_decoupling | regrasp_or_drop | 164 | 2 | 0.0122 |
| dual_extremes | success_false_alarm | 14313 | 785 | 0.0548 |
| dual_extremes | all_failures | 487 | 282 | 0.5791 |
| dual_extremes | stagnation | 136 | 89 | 0.6544 |
| dual_extremes | gripper_cycling | 81 | 49 | 0.6049 |
| dual_extremes | goal_regression | 104 | 75 | 0.7212 |
| dual_extremes | goal_approach_leave | 10 | 3 | 0.3000 |
| dual_extremes | subtask_undo | 37 | 14 | 0.3784 |
| dual_extremes | regrasp_or_drop | 164 | 102 | 0.6220 |
| dual_mean_control | success_false_alarm | 14313 | 742 | 0.0518 |
| dual_mean_control | all_failures | 487 | 324 | 0.6653 |
| dual_mean_control | stagnation | 136 | 111 | 0.8162 |
| dual_mean_control | gripper_cycling | 81 | 45 | 0.5556 |
| dual_mean_control | goal_regression | 104 | 87 | 0.8365 |
| dual_mean_control | goal_approach_leave | 10 | 7 | 0.7000 |
| dual_mean_control | subtask_undo | 37 | 27 | 0.7297 |
| dual_mean_control | regrasp_or_drop | 164 | 120 | 0.7317 |
| multi_phenotype | success_false_alarm | 14313 | 808 | 0.0565 |
| multi_phenotype | all_failures | 487 | 276 | 0.5667 |
| multi_phenotype | stagnation | 136 | 84 | 0.6176 |
| multi_phenotype | gripper_cycling | 81 | 50 | 0.6173 |
| multi_phenotype | goal_regression | 104 | 72 | 0.6923 |
| multi_phenotype | goal_approach_leave | 10 | 3 | 0.3000 |
| multi_phenotype | subtask_undo | 37 | 13 | 0.3514 |
| multi_phenotype | regrasp_or_drop | 164 | 98 | 0.5976 |

### 互斥主标签检查

上表中的物理标签允许重叠。下表将每条失败只归到一个 primary behavior，用于检查某个头是否只是被重叠样本抬高；数值仍是 q95 触发率。

| target | n | instability | lock_in | flat_narrow_support | feedback_decoupling | dual_extremes | dual_mean_control | multi_phenotype |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success_false_alarm | 14313 | 0.0545 | 0.0530 | 0.0524 | 0.0548 | 0.0548 | 0.0518 | 0.0565 |
| stagnation | 108 | 0.0000 | 0.6759 | 0.2870 | 0.0000 | 0.6667 | 0.8148 | 0.6204 |
| gripper_cycling | 38 | 0.3947 | 0.2105 | 0.0000 | 0.4211 | 0.5789 | 0.5263 | 0.6316 |
| goal_regression | 5 | 0.0000 | 0.8000 | 0.0000 | 0.0000 | 0.8000 | 1.0000 | 0.8000 |
| goal_approach_leave | 4 | 0.0000 | 0.7500 | 0.0000 | 0.0000 | 0.5000 | 0.7500 | 0.5000 |
| subtask_undo | 37 | 0.0000 | 0.3784 | 0.0270 | 0.0000 | 0.3784 | 0.7297 | 0.3514 |
| regrasp_or_drop | 133 | 0.1128 | 0.6015 | 0.0376 | 0.0150 | 0.6767 | 0.7143 | 0.6541 |
| other | 162 | 0.0123 | 0.4753 | 0.0185 | 0.0741 | 0.4815 | 0.5309 | 0.4877 |

## 路由表型分离

AUC 在 task/init 内计算后对 mixed groups 等权平均。`other_failure` 比较回答某一物理表型能否与其他失败区分；`success` 比较更容易受到阶段和轨迹长度混杂。置信区间为 task/init group bootstrap，当前属于探索性结果，未做全族多重校正。
`direction` 和 `det_auc` 是看过数据后选方向的表型诊断，不能当作预先规定的detector 性能；固定头必须读取原始 `auc`，低于 0.5 就代表方向相反。

### stagnation vs other failures

| label | comparison | signal | direction | auc | auc_ci_low | auc_ci_high | det_auc | ci_low | ci_high | mixed_groups | positive_n | negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stagnation | other_failure | lag_recurrence_mean | high | 0.9588 | 0.9190 | 0.9912 | 0.9588 | 0.9190 | 0.9912 | 36 | 88 | 82 |
| stagnation | other_failure | route_mobility_mean | low | 0.0492 | 0.0106 | 0.0988 | 0.9508 | 0.9012 | 0.9894 | 36 | 88 | 82 |
| stagnation | other_failure | front_state_jump_mean | low | 0.0843 | 0.0162 | 0.1745 | 0.9157 | 0.8255 | 0.9838 | 36 | 88 | 82 |
| stagnation | other_failure | front_action_jump_mean | low | 0.0863 | 0.0269 | 0.1679 | 0.9137 | 0.8321 | 0.9731 | 36 | 88 | 82 |
| stagnation | other_failure | front_feedback_split_p90 | low | 0.0968 | 0.0296 | 0.1750 | 0.9032 | 0.8250 | 0.9704 | 36 | 88 | 82 |
| stagnation | other_failure | front_feedback_split_mean | low | 0.0981 | 0.0208 | 0.1931 | 0.9019 | 0.8069 | 0.9792 | 36 | 88 | 82 |

### gripper_cycling vs other failures

| label | comparison | signal | direction | auc | auc_ci_low | auc_ci_high | det_auc | ci_low | ci_high | mixed_groups | positive_n | negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gripper_cycling | other_failure | front_state_jump_mean | high | 0.9048 | 0.8373 | 0.9643 | 0.9048 | 0.8373 | 0.9643 | 21 | 34 | 49 |
| gripper_cycling | other_failure | front_feedback_split_mean | high | 0.9008 | 0.8293 | 0.9683 | 0.9008 | 0.8293 | 0.9683 | 21 | 34 | 49 |
| gripper_cycling | other_failure | front_action_jump_mean | high | 0.8750 | 0.7460 | 0.9702 | 0.8750 | 0.7460 | 0.9702 | 21 | 34 | 49 |
| gripper_cycling | other_failure | route_acceleration_mean | high | 0.8700 | 0.7460 | 0.9613 | 0.8700 | 0.7460 | 0.9613 | 21 | 34 | 49 |
| gripper_cycling | other_failure | front_feedback_split_p90 | high | 0.8671 | 0.7658 | 0.9563 | 0.8671 | 0.7658 | 0.9563 | 21 | 34 | 49 |
| gripper_cycling | other_failure | late_flow_volatility_mean | high | 0.8532 | 0.7222 | 0.9563 | 0.8532 | 0.7222 | 0.9563 | 21 | 34 | 49 |

### goal_regression vs other failures

| label | comparison | signal | direction | auc | auc_ci_low | auc_ci_high | det_auc | ci_low | ci_high | mixed_groups | positive_n | negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_regression | other_failure | lag_periodicity_mean | low | 0.2565 | 0.1281 | 0.4158 | 0.7435 | 0.5842 | 0.8719 | 20 | 34 | 58 |
| goal_regression | other_failure | back_state_action_gap_mean | high | 0.7309 | 0.5751 | 0.8609 | 0.7309 | 0.5751 | 0.8609 | 20 | 34 | 58 |
| goal_regression | other_failure | gate_entropy_slope | low | 0.2698 | 0.1181 | 0.4338 | 0.7302 | 0.5662 | 0.8819 | 20 | 34 | 58 |
| goal_regression | other_failure | front_state_action_gap_mean | high | 0.7207 | 0.5316 | 0.8833 | 0.7207 | 0.5316 | 0.8833 | 20 | 34 | 58 |
| goal_regression | other_failure | front_feedback_split_p90 | high | 0.6868 | 0.5450 | 0.8175 | 0.6868 | 0.5450 | 0.8175 | 20 | 34 | 58 |
| goal_regression | other_failure | front_state_jump_mean | high | 0.6773 | 0.5239 | 0.8148 | 0.6773 | 0.5239 | 0.8148 | 20 | 34 | 58 |

### goal_approach_leave vs other failures

| label | comparison | signal | direction | auc | auc_ci_low | auc_ci_high | det_auc | ci_low | ci_high | mixed_groups | positive_n | negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_approach_leave | other_failure | front_state_action_gap_mean | high | 0.9762 | 0.9286 | 1.0000 | 0.9762 | 0.9286 | 1.0000 | 6 | 6 | 15 |
| goal_approach_leave | other_failure | lag_periodicity_mean | low | 0.1071 | 0.0000 | 0.2738 | 0.8929 | 0.7262 | 1.0000 | 6 | 6 | 15 |
| goal_approach_leave | other_failure | back_state_action_gap_mean | high | 0.8810 | 0.6429 | 1.0000 | 0.8810 | 0.6429 | 1.0000 | 6 | 6 | 15 |
| goal_approach_leave | other_failure | support_entropy_mean | low | 0.1905 | 0.0000 | 0.5238 | 0.8095 | 0.4762 | 1.0000 | 6 | 6 | 15 |
| goal_approach_leave | other_failure | token_disagreement_mean | high | 0.7976 | 0.5595 | 1.0000 | 0.7976 | 0.5595 | 1.0000 | 6 | 6 | 15 |
| goal_approach_leave | other_failure | front_feedback_split_p90 | high | 0.7738 | 0.4881 | 1.0000 | 0.7738 | 0.4881 | 1.0000 | 6 | 6 | 15 |

### subtask_undo vs other failures

| label | comparison | signal | direction | auc | auc_ci_low | auc_ci_high | det_auc | ci_low | ci_high | mixed_groups | positive_n | negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subtask_undo | other_failure | back_state_action_gap_mean | high | 0.8012 | 0.6371 | 0.9336 | 0.8012 | 0.6371 | 0.9336 | 16 | 22 | 52 |
| subtask_undo | other_failure | front_feedback_split_p90 | high | 0.7992 | 0.6911 | 0.9047 | 0.7992 | 0.6911 | 0.9047 | 16 | 22 | 52 |
| subtask_undo | other_failure | front_feedback_split_mean | high | 0.7747 | 0.6139 | 0.9062 | 0.7747 | 0.6139 | 0.9062 | 16 | 22 | 52 |
| subtask_undo | other_failure | front_state_jump_mean | high | 0.7747 | 0.6167 | 0.9062 | 0.7747 | 0.6167 | 0.9062 | 16 | 22 | 52 |
| subtask_undo | other_failure | lag_periodicity_mean | low | 0.2331 | 0.1029 | 0.3776 | 0.7669 | 0.6224 | 0.8971 | 16 | 22 | 52 |
| subtask_undo | other_failure | front_state_action_gap_mean | high | 0.7603 | 0.5707 | 0.9134 | 0.7603 | 0.5707 | 0.9134 | 16 | 22 | 52 |

### regrasp_or_drop vs other failures

| label | comparison | signal | direction | auc | auc_ci_low | auc_ci_high | det_auc | ci_low | ci_high | mixed_groups | positive_n | negative_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regrasp_or_drop | other_failure | lag_recurrence_mean | low | 0.2964 | 0.1808 | 0.4189 | 0.7036 | 0.5811 | 0.8192 | 31 | 53 | 79 |
| regrasp_or_drop | other_failure | back_state_action_gap_mean | high | 0.6772 | 0.5204 | 0.8144 | 0.6772 | 0.5204 | 0.8144 | 31 | 53 | 79 |
| regrasp_or_drop | other_failure | token_disagreement_mean | high | 0.6767 | 0.5361 | 0.8145 | 0.6767 | 0.5361 | 0.8145 | 31 | 53 | 79 |
| regrasp_or_drop | other_failure | route_acceleration_mean | high | 0.6675 | 0.5174 | 0.8024 | 0.6675 | 0.5174 | 0.8024 | 31 | 53 | 79 |
| regrasp_or_drop | other_failure | front_action_jump_mean | high | 0.6636 | 0.5269 | 0.7950 | 0.6636 | 0.5269 | 0.7950 | 31 | 53 | 79 |
| regrasp_or_drop | other_failure | top12_margin_mean | high | 0.6594 | 0.5191 | 0.7925 | 0.6594 | 0.5191 | 0.7925 | 31 | 53 | 79 |

## 解释边界

- `identify_targets` 通过成功轨迹中平均移动超过 3 cm 的 free joint 识别任务物体；drawer/stove 这类非 free-joint 任务没有物体进度标签。
- 路由窗口按 episode 相对时间 50%--90% 对齐，并不能保证语义阶段严格一致。
- `state_action_gap` 是 token-routing 差异，不是校准后的认识不确定性或语义信念。
- 当前记录没有 RGB、contact、force，也没有 chunk 内重新 forward；因此不能把代理升级为 true grasp/drop/contact，也不能在 stale chunk 中途给出 MoE 响应。
- 本轮用于建立表型关联，不把任何 AUC 或 q95 触发率解释为因果性或恢复收益。

## 代表性轨迹

共输出 48 个高严重度候选到 `representative_candidates.csv`。

## 可观测性

自动识别不到 free-joint 任务目标的任务数：2。
