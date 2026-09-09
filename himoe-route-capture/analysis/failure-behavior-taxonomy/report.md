# 失败 rollout 的物理/控制行为谱

## 直接结论

这里不把 307 条失败强行切成互斥簇，而是给每条 rollout 标注可重叠的物理/控制行为。所有连续阈值都只由同 task、同初态的成功 siblings 校准；初态成功样本不足时向 task 成功阈值收缩。episode length、task id 和终点坐标都不作为分类特征。

最强的结构是停滞：166/307 条失败满足，而且 primary label 中有 153 条。repeated-lift/drop proxy 覆盖 58 条；它比旧 metadata 中保守的 transport-loss 更宽，只能解释为运动学代理。

另外有 88 条（28.7%）没有满足任何离散行为规则。它们并非像成功轨迹一样正常：其中 87 条在 50%--90% 相位的 EEF 速度低于 sibling-success q25，79 条至少一个进展效率低于 success q05。更准确的描述是“分散式减速和低进展”，而不是新的一种异常重试簇。

`active_retry` 和 `eef_oscillation` 在失败中都为 0。这是需要保留的负结果：此前路由残余组相对停滞核心更活跃，不等于它们相对成功轨迹出现了异常重试。

停滞规则只看 0%--90% 相位中的连续低运动段，最后 10% 完全不参与定义；下面另有专门审计。

## 标签数量

| 行为 | failure n/rate | success n/rate | truncate90 Jaccard | bootstrap Jaccard median [p10,p90] |
|---|---:|---:|---:|---:|
| stagnation | 166 / 0.541 | 5 / 0.002 | 0.970 | 1.000 [1.000,1.000] |
| active_retry | 0 / 0.000 | 24 / 0.011 | 1.000 | vacuous (0 positive) |
| eef_oscillation | 0 / 0.000 | 37 / 0.016 | 1.000 | vacuous (0 positive) |
| gripper_cycling | 9 / 0.029 | 22 / 0.010 | 0.462 | 0.900 [0.529,1.000] |
| goal_regression | 49 / 0.160 | 0 / 0.000 | 0.900 | 1.000 [1.000,1.000] |
| goal_approach_leave | 5 / 0.016 | 0 / 0.000 | 1.000 | 1.000 [1.000,1.000] |
| subtask_undo | 8 / 0.026 | 0 / 0.000 | 0.500 | 1.000 [1.000,1.000] |
| regrasp_or_drop | 58 / 0.189 | 9 / 0.004 | 0.966 | 1.000 [0.983,1.000] |

## Primary behavior（仅为摘要，不替代多标签）

| primary | n | fraction |
|---|---:|---:|
| stagnation | 153 | 0.498 |
| other | 88 | 0.287 |
| regrasp_or_drop | 50 | 0.163 |
| subtask_undo | 8 | 0.026 |
| gripper_cycling | 5 | 0.016 |
| goal_approach_leave | 3 | 0.010 |

## Multi-label overlap

| label A | label B | intersection n | Jaccard |
|---|---|---:|---:|
| goal_regression | regrasp_or_drop | 46 | 0.754 |
| stagnation | regrasp_or_drop | 13 | 0.062 |
| stagnation | goal_regression | 12 | 0.059 |
| goal_regression | subtask_undo | 8 | 0.163 |
| subtask_undo | regrasp_or_drop | 5 | 0.082 |
| goal_regression | goal_approach_leave | 5 | 0.102 |
| goal_approach_leave | regrasp_or_drop | 5 | 0.086 |
| gripper_cycling | regrasp_or_drop | 4 | 0.063 |
| goal_approach_leave | subtask_undo | 2 | 0.182 |
| gripper_cycling | goal_regression | 1 | 0.018 |
| gripper_cycling | goal_approach_leave | 1 | 0.077 |

## 未标注 residual 的特点

| 检查项 | n / 88 |
|---|---:|
| 50%--90% EEF speed < sibling-success q25 | 87 |
| late path efficiency < success q05 | 38 |
| late goal progress < success q05 | 79 |
| 至少 3 次 gripper 符号切换 | 61 |
| 归一化 flip rate > success q95 | 0 |
| 有任意 pre-90% 静止 transition，但没有长停滞段 | 55 |
| 末端 10% 多数静止 | 13 |
| 去掉末尾 10% 后仍未标注 | 83 |

按各自 sibling-success q25 归一化后，50%--90% EEF 活动比的中位数依次是：成功 1.03、未标注 residual 0.66、停滞 0.24。所以 residual 位于连续的 activity-to-stasis 轴中间，并没有支持把它们再硬切成几种稳定循环。

而且 residual 有 62/88 条来自双 moka-pot 任务，旧结局标签中 53/88 条是 `partial`。其余还横跨 never_grasped、misplaced、lifted_not_placed 和 reached_then_lost。这个任务集中性和结局异质性进一步反对把 residual 命名成一个普适机制。

夹爪结果尤其说明长度归一化为什么重要：许多 residual 有多次符号切换，但按相位/transition 归一化后没有一条超过成功 q95，不能称为异常 cycling。

## 停滞不是末端低位移的自动标签

- 默认 pre-90% 规则标出 166 条；直接看整条轨迹的 naive 规则标出 165 条，其中只有 4 条是整条规则独有。
- 有 33 条失败虽然末端 10% 多数静止，却没有被标成 stagnation；反过来有 45 条 stagnation 在末端并非多数静止。
- 停滞最长段开始相位中位数为 0.598（IQR 0.578--0.652），157 条在 75% 相位前已经开始。
- 砍掉最后 10% 后 failure assignment Jaccard=0.970。
- 独立末 25% event audit 的 stasis 是 141 条，与 taxonomy 166 的交集为 136；taxonomy 的 166 中有 30 条后来恢复过运动。因此 166 的准确含义是“前 90% 曾出现长连续全系统低变化段”，不是“166 条都静止到 timeout”。
- 双 moka-pot 中严格的 `stalled_at_unfinished_pot1` 有 109 条，109/109 同时属于 taxonomy stagnation；独立方法复核还确认这 109 条与 MoE stagnation core 一致。来源见 `analysis/failure-type-method-review/report.md`。

## 与旧 heuristic failure_mode 的交叉表

| primary | lifted_not_placed | misplaced | never_grasped | partial | reached_then_lost | total |
|---|---:|---:|---:|---:|---:|---:|
| goal_approach_leave | 0 | 0 | 0 | 3 | 0 | 3 |
| gripper_cycling | 0 | 1 | 2 | 2 | 0 | 5 |
| other | 2 | 14 | 14 | 53 | 5 | 88 |
| regrasp_or_drop | 40 | 7 | 0 | 3 | 0 | 50 |
| stagnation | 0 | 3 | 13 | 137 | 0 | 153 |
| subtask_undo | 0 | 0 | 0 | 0 | 8 | 8 |

这里的 `failure_mode` 只用于外部一致性检查，不参与行为规则；因此交叉表不是标签定义的同义反复。

## 可解释规则

- `stagnation`: 前 90% 相位内，EEF 每 query 位移不超过 5 mm、任务物体不超过 3 mm、夹爪孔径变化不超过 1 mm 的最长连续段，超过成功 q95（且至少占 15%）。
- `active_retry`: 后半段仍达到成功 q25 的运动速度，但目标进展效率处于成功 q05 以下，并同时出现 EEF 反向、夹爪反转、目标接近后离开或重新抓取信号。
- `eef_oscillation`: 有效 EEF 位移之间夹角余弦小于 -0.25 的反向率超过成功 q95。
- `gripper_cycling`: chunk 平均夹爪命令的符号反转率超过成功 q95，且至少出现 3 次反转。
- `goal_regression`: 相对历史最佳目标距离回退至少 5 cm，并超过成功 q95。
- `goal_approach_leave`: 进入目标 10 cm 后又退出到 15 cm 外，事件率超过成功 q95。
- `subtask_undo`: 任务物体曾到达 5 cm 目标区，之后又离开到 7.5 cm 外。
- `regrasp_or_drop`: 单个目标重复 lift，或 lifted 后在目标区外掉落，并超过成功 q95。

Bootstrap 共 200 次，每次在 task x init 内对成功 siblings 有放回重采样；primary agreement 中位数为 0.997，p10=0.974。q90/q97.5 和 truncate90 的完整敏感性结果在 `summary.json`。

## 限制

- 这些是 query 边界上的运动学代理，不是人工视频语义标注。
- 多标签重叠是有意保留的；例如 active retry 可以同时伴随夹爪 cycling。
- success-normalized 阈值减少 task/init 混杂，但不能证明任何内部路由信号导致了这些行为。

## 产物

- `episode_features_labels.csv`: 逐 episode 特征、阈值、多标签和 primary label。
- `behavior_curves.npz`: 20 相位压缩曲线；不含原始 sim state 或路由矩阵。
- `bootstrap_assignment_stability.csv`: 每条失败在成功基线重采样下的标签概率。
- `thresholds_by_task_init.csv`: success-normalized 阈值审计表。
- `multilabel_overlap_counts.csv` / `multilabel_overlap_jaccard.csv`: 失败标签重叠。
- `summary.json`: 标签、failure_mode 交叉表和敏感性结果。
- `behavior_overview.png`: prevalence、标签重叠和敏感性；`normalized_behavior_profiles.png`: 相对相位行为曲线。
