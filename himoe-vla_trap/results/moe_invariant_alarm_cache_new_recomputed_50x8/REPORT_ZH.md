# cache_new MoE 健康不变量报警实验

## 核心结果

**严格判定：未通过。** 固定三头 MoE 判断器在等误报比较中被时间钟支配，且首次报警没有稳定贴近物理 onset。

本实验完全不使用旧的跨语料概率表。检测器只读取当前轨迹截至当前 query 的 HB full-softmax MoE 路由；任务 ID、动作、物理状态、reward、结果和未来长度均不进入检测。

固定任务级哈希划分得到 32 个阈值确认任务和 8 个留出任务。阈值确认只使用成功轨迹，整条成功轨迹误报预算为 1.0%，实际为 0.99%。失败标签没有参与特征、组合或阈值。

留出集共 3200 条轨迹，其中失败 114 条。MoE 判断器报警 83 条：TP=51、FP=32、FN=63、TN=3054；precision=61.45%、failure recall=44.74%、成功误报率=1.04%。

同一 1.0% 阈值确认预算下，固定时间钟在 q=39 报警；留出 precision=94.00%、failure recall=41.23%、成功误报率=0.10%。

由于任务迁移后 q=39 的实际误报率远低于 MoE，另给出一个只在揭盲后用于公平诊断的等误报时钟 q=29：precision=71.58%、failure recall=59.65%、成功误报率=0.87%。它不参与阈值选择，只是对 MoE 结果的保守压力测试。

留出报警涉及 5/8 个任务；误报只涉及 1 个任务。收敛+响应联合分支命中 0 条，持续复返分支命中 83 条。这说明最终结果实际退化成单一 recurrence 检测器，而不是预期的三种机制互补。

因此是否证明 MoE 有独立价值，只看它在相同误报率下能否超过时间钟，而不根据绝对 precision 单独下结论。

在留出预测固定后，才额外揭开阈值确认任务中此前未使用的失败标签。全 40 任务、16000 条轨迹的描述性翻牌中，检测器检出 152/532 个失败（28.57%），precision=49.51%、成功误报率=1.00%。这张表覆盖全部失败，但成功误报率包含阈值确认样本，不能替代严格留出结果。

## 固定检测器

每条轨迹使用 q0--q3 建立自身基线，从 q4 开始评分。三个不可学习的机制头为：

- 去噪收敛失败：完整 flow 路由路径中发生在 late flow 的 Hellinger 路径占比，相对自身早期基线上升；
- state-action 响应脱节：相邻重规划间 front-HB state-token 路由跳变大于 action-token 跳变；
- 跨 chunk 复返：final-flow back-HB action routing 对 lag 1--4 历史的最大 weighted Jaccard。

前两个头必须在同一 query 同时极端；复返头必须连续两个 query 极端。两条分支取 OR，但不学习任何权重。
每个头的数值只换算成相对阈值确认成功轨迹整段最大值的健康尾部置信度。这是异常显著性，不是失败概率。

## 报警原因翻牌

| alarm_cause | alarms | failures | successes | empirical_failure_fraction | first_alarm_query_median | first_alarm_phase_median |
| --- | --- | --- | --- | --- | --- | --- |
| persistent_recurrence | 83.0000 | 51.0000 | 32.0000 | 0.6145 | 17.0000 | 0.7692 |

## 留出任务

| suite | task |
| --- | --- |
| libero_goal | libero_goal/put_the_bowl_on_the_stove |
| libero_goal | libero_goal/put_the_bowl_on_top_of_the_cabinet |
| libero_long | libero_long/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | libero_long/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_object | libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket |
| libero_spatial | libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate |
| libero_spatial | libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate |

## 逐任务结果

| task | suite | episodes | failures | successes | detector_tp | detector_fp | detector_fn | detector_tn | detector_precision | detector_failure_recall | detector_success_false_alarm_rate | clock_tp | clock_fp | clock_fn | clock_tn | clock_precision | clock_failure_recall | clock_success_false_alarm_rate | matched_clock_tp | matched_clock_fp | matched_clock_fn | matched_clock_tn | matched_clock_precision | matched_clock_failure_recall | matched_clock_success_false_alarm_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libero_goal/put_the_bowl_on_the_stove | libero_goal | 400.0000 | 6.0000 | 394.0000 | 0.0000 | 0.0000 | 6.0000 | 394.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.0000 | 394.0000 | 0.0000 | 0.0000 | 0.0000 | 6.0000 | 0.0000 | 0.0000 | 394.0000 | 1.0000 | 1.0000 | 0.0000 |
| libero_goal/put_the_bowl_on_top_of_the_cabinet | libero_goal | 400.0000 | 15.0000 | 385.0000 | 13.0000 | 0.0000 | 2.0000 | 385.0000 | 1.0000 | 0.8667 | 0.0000 | 0.0000 | 0.0000 | 15.0000 | 385.0000 | 0.0000 | 0.0000 | 0.0000 | 15.0000 | 0.0000 | 0.0000 | 385.0000 | 1.0000 | 1.0000 | 0.0000 |
| libero_long/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | libero_long | 400.0000 | 4.0000 | 396.0000 | 2.0000 | 0.0000 | 2.0000 | 396.0000 | 1.0000 | 0.5000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 | 396.0000 | 1.0000 | 1.0000 | 0.0000 | 4.0000 | 10.0000 | 0.0000 | 386.0000 | 0.2857 | 1.0000 | 0.0253 |
| libero_long/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | libero_long | 400.0000 | 43.0000 | 357.0000 | 0.0000 | 0.0000 | 43.0000 | 357.0000 | 0.0000 | 0.0000 | 0.0000 | 43.0000 | 3.0000 | 0.0000 | 354.0000 | 0.9348 | 1.0000 | 0.0084 | 43.0000 | 17.0000 | 0.0000 | 340.0000 | 0.7167 | 1.0000 | 0.0476 |
| libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket | libero_object | 400.0000 | 11.0000 | 389.0000 | 6.0000 | 0.0000 | 5.0000 | 389.0000 | 1.0000 | 0.5455 | 0.0000 | 0.0000 | 0.0000 | 11.0000 | 389.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.0000 | 389.0000 | 0.0000 | 0.0000 | 0.0000 |
| libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket | libero_object | 400.0000 | 0.0000 | 400.0000 | 0.0000 | 32.0000 | 0.0000 | 368.0000 | 0.0000 | 0.0000 | 0.0800 | 0.0000 | 0.0000 | 0.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | libero_spatial | 400.0000 | 0.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 400.0000 | 0.0000 | 0.0000 | 0.0000 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | libero_spatial | 400.0000 | 35.0000 | 365.0000 | 30.0000 | 0.0000 | 5.0000 | 365.0000 | 1.0000 | 0.8571 | 0.0000 | 0.0000 | 0.0000 | 35.0000 | 365.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 35.0000 | 365.0000 | 0.0000 | 0.0000 | 0.0000 |

## onset 代理的二次审计

留出任务中有 71 个既有的 query-boundary 物理 onset 代理。MoE 首次报警不晚于 onset 为 5，位于 [-2,0] 为 5；预固定时间钟对应为 1 和 1；等误报时钟对应为 8 和 2。

这些标签仅用于预测文件哈希后的事后审计，属于运动学代理，不是 contact/video 真值。

## 解释边界

- 最终 success/failure 翻牌评价的是任意 endpoint failure 关联，不等价于特定 Trap 类型。
- `confidence` 表示健康路由下的经验极端程度，不是单轨迹失败概率。
- 数据集此前已被用于其他探索性研究；任务划分和本检测器未使用留出失败标签，但不能称为全新 prospective benchmark。
- 只有当 MoE 在同成功误报预算下稳定超过时间钟，才能声称它提供了时长之外的失败信息。

## 可复现产物

- `configs/moe_invariant_alarm_cache_new.json`：预固定设计；
- `code/evaluate_moe_invariant_alarm_cache_new.py`：路由提取、阈值确认、盲回放、揭盲和审计；
- `heldout_predictions_label_free.csv.gz`：揭盲前逐 query 预测；
- `prediction_manifest.json`：预测文件 SHA256 和标签隔离声明；
- `tables/heldout_episode_flip.csv`：逐轨迹翻牌；
- `tables/all_episode_flip_descriptive.csv`：预测冻结后生成的全 16,000 轨迹描述性翻牌；
- `summary.json`：机器可读主结果。
