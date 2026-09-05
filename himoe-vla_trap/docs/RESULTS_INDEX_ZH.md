# 结果文件索引

## Expert 编号与身份连续性消融

| 路径 | 内容 |
|---|---|
| `results/moe_expert_identity_ablation/summary.json` | 两个 seed 组、8 种路由表示和全局重编号不变性的机器摘要 |
| `results/moe_expert_identity_ablation/REPORT_ZH.md` | 冻结主结果、22-FP 同预算诊断、top-4 tie 审计和 onset 结果 |
| `results/moe_expert_identity_ablation/route_only_tasks/` | 80 个 task-run、508,023 queries 的无标签消融缓存 |
| `results/moe_expert_identity_ablation/predictions_label_free/` | 两方向共 508,023 行、读取目标 outcome 前冻结并哈希的逐 query 预测 |
| `results/moe_expert_identity_ablation/tables/combined_metrics.csv` | 每种表示的严格双向冻结 TP/FP/FN/TN |
| `results/moe_expert_identity_ablation/tables/posthoc_fp_budget_sweep.csv` | 相同成功误报数量下的表示比较与固定时间钟负控 |
| `results/moe_expert_identity_ablation/tables/heldout_onset_timing.csv` | 120 个物理 onset 的首次报警时机 |

统一全局重编号不改变路由距离，因此 expert 的数字标签没有语义。在揭盲后的相同 22 个成功误报预算下，固定坐标 soft recurrence 检出 164 个失败，删除身份的概率排序只检出 84 个，每 chunk 独立重编号的 soft/support 只检出 0/24 个。实际、重建、tie-filter top-4 分别检出 116/122/115 个，说明保存的 top-4 tie-break ID 并非必要；真正有用的是固定 checkpoint 中 expert 参数分支的时间连续性。soft-coordinate 的 88 个 onset 报警中 73 个仍然太晚，因此该结果没有验证可靠 early detector。

## 结构化 MoE 三机制报警

| 路径 | 内容 |
|---|---|
| `results/moe_structured_alarm_two_runs/summary.json` | 两个 seed 组双向冻结、逐任务鲁棒阈值和严格留出主结果 |
| `results/moe_structured_alarm_two_runs/REPORT_ZH.md` | 24-cell、三种路由视角和三机制头的主报告 |
| `results/moe_structured_alarm_two_runs/route_only_tasks/` | 80 个 task-run、508,023 queries 的无标签结构化 MoE 缓存 |
| `results/moe_structured_alarm_two_runs/predictions_label_free/` | 两方向共 348,023 行、目标 outcome 加载前冻结并哈希的逐 query 预测 |
| `results/moe_structured_alarm_two_runs/tables/direction_and_head_metrics.csv` | 双方向组合头、锁死、抖动、脱节和时间钟的精确 TP/FP/FN/TN |
| `results/moe_structured_alarm_two_runs/tables/heldout_onset_timing.csv` | 两组共 120 个留出物理 onset 的首次报警和窗口激活时序 |
| `results/moe_structured_alarm_two_runs/audit/REPORT_ZH.md` | 同协议旧三标量公平消融、固定误报数量 sweep 和结论边界 |
| `results/moe_structured_alarm_two_runs/redundant_recurrence_views_v0/` | 将相关 recurrence 视角重复计票的第一版负结果 |
| `results/moe_structured_alarm_two_runs/pooled_threshold_v1/` | 修正重复计票但采用总体平均阈值的中间结果 |

严格测试合计 TP=121、FP=17、FN=118、TN=6,144；完全相同协议下旧三标量为 TP=79、FP=27、FN=160、TN=6,134。因此 actual expert support/weight 与位置保持结构带来了增量信息。120 个 onset 中只有 10 个首次报警位于 `[-2,0]`，38 个是 onset 后报警；138 次报警中 lock-in 占 134 次。结论是 endpoint 风险筛查得到改善，但 early Trap detection 尚未成立。

## 当前 MoE 信息使用审计

| 路径 | 内容 |
|---|---|
| `results/moe_information_usage_audit/REPORT_ZH.md` | 原始字段、实际访问坐标、压缩损失、未用信息和下一版三机制头的完整审计 |
| `results/moe_information_usage_audit/summary.json` | 每 query 精确计数、报警来源、AS 全量审计和 24-cell 重组方案 |
| `results/moe_invariant_alarm_two_run_audit/REPORT_ZH.md` | 两个独立 seed 组的结构完整性、公式复算、冻结规则迁移和合并混淆矩阵 |
| `results/moe_invariant_alarm_two_run_audit/tables/transfer_metrics.csv` | 两个方向的冻结阈值迁移及固定时间钟对照 |
| `results/moe_invariant_alarm_two_run_audit/tables/independent_formula_recompute.csv` | 3,592 个标量的独立公式复算误差 |

每次重规划保存 `8x10x11x32 = 28,160` 个 HB 路由概率。当前判断器只访问 14,208 个，完全不读 13,952 个；访问到的内容又被压成三个标量。两个 seed 组的严格留出合计 TP=106、FP=58、FN=133、TN=6,103，而 164 次报警中 163 次由 recurrence 产生。因此当前实现实际上接近单一复返检测器，并未充分利用 layer、flow、token 和 top-4 支持结构。AS 已在全部 508,023 次推理上确认任务内恒定，应从动态报警中排除。下一轮冻结现规则作为负结果，按锁死、抖动、state-action 脱节三个机制头重提取。

## cache_new MoE 健康不变量报警

| 路径 | 内容 |
|---|---|
| `results/moe_invariant_alarm_cache_new/summary.json` | 40 tasks、16,000 episodes、严格留出和全量描述结果 |
| `results/moe_invariant_alarm_cache_new/REPORT_ZH.md` | 固定三头方法、时间钟对照、报警原因和结论边界 |
| `results/moe_invariant_alarm_cache_new/heldout_predictions_label_free.csv.gz` | 揭盲前冻结的 34,935 行逐 query 留出预测 |
| `results/moe_invariant_alarm_cache_new/prediction_manifest.json` | 预测 SHA256、任务划分哈希和标签隔离声明 |
| `results/moe_invariant_alarm_cache_new/tables/heldout_episode_flip.csv` | 3,200 条留出轨迹的逐 episode 翻牌 |
| `results/moe_invariant_alarm_cache_new/tables/all_episode_flip_descriptive.csv` | 全 16,000 条轨迹、532 个失败的描述性翻牌 |
| `results/moe_invariant_alarm_cache_new/tables/heldout_clock_sweep.csv` | 揭盲后等误报时间钟压力测试 |
| `results/moe_invariant_alarm_cache_new/tables/heldout_onset_proxy_audit.csv` | 71 个留出物理 onset 代理的首次报警时序 |

该实验完全删除旧概率表依赖，只保留三种固定 MoE 健康不变量和成功轨迹 1% 整段误报阈值。严格留出中 MoE 为 TP=51、FP=32、failure recall 44.74%；等误报 q29 时间钟为 TP=68、FP=27、recall 59.65%。全部留出报警均来自持续 recurrence，71 个 onset 中只有 5 个在 `[-2,0]` 首次报警。按预定标准结果未通过。完整解释见 [MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md](MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md)。

## Train-free Trap 概率与 75% 报警审计

| 路径 | 内容 |
|---|---|
| `results/trainfree_trap_probability/summary.json` | A→B/B→A、H=0/2/4、loop/static/trap 的机器可读概率评价与输入隔离声明 |
| `results/trainfree_trap_probability/REPORT_ZH.md` | 自动生成的主结果、阈值审计和时钟混淆说明 |
| `results/trainfree_trap_probability/intermediate/corpus_*_route_only.npz` | 加载 onset/outcome 之前物化的逐 query MoE 特征 |
| `results/trainfree_trap_probability/intermediate/states_*.npz` | 只用 source 无标签 CDF 构造的跨语料离散 route states |
| `results/trainfree_trap_probability/predictions_label_free/` | target label 加载前写出的 18 份 onset-hazard 概率 CSV |
| `results/trainfree_trap_probability/predictions_label_free_absorbing/` | target label 加载前写出的 6 份宽松 absorbing-proxy 概率 CSV |
| `results/trainfree_trap_probability/tables/cross_corpus_probability_evaluation.csv` | 18 个正式 onset-hazard 概率任务的 AUROC/AP/Brier/ECE 与常数/时钟对照 |
| `results/trainfree_trap_probability/tables/primary_probability_threshold_sweep.csv` | 主 H2 Trap 目标在预先声明阈值上的 precision、recall 和 episode FPR |
| `results/trainfree_trap_probability/tables/primary_episode_alarm_audit.csv` | 863 条 target episode 的 75% 及时报警审计 |
| `results/trainfree_trap_probability/tables/absorbing_detection_evaluation.csv` | onset 后整段标正的上限压力测试及 query-clock 对照 |
| `results/trainfree_trap_probability/figures/trainfree_trap_probability.png` | 跨语料 reliability、最大概率和 AP/基率三联图 |

主 onset-hazard 目标上，A→B/B→A 的最大预测概率为 `35.8%/51.3%`，75% 均为零报警。AUROC 为 `0.819/0.717`，说明存在排序信息；但最高状态在 target 实测仅 `21.4%/15.8%`，Brier 也没有稳定优于常数基率。25% 阈值仍不足以部署。完整解释见 [TRAINFREE_TRAP_PROBABILITY_ZH.md](TRAINFREE_TRAP_PROBABILITY_ZH.md)。

### cache_new 40-task 外推

| 路径 | 内容 |
|---|---|
| `results/trainfree_trap_probability/cache_new_40task/summary.json` | 40 tasks、16,000 episodes、253,722 queries 的成功路径概率与 endpoint 混淆摘要 |
| `results/trainfree_trap_probability/cache_new_40task/route_only_manifest.json` | outcome 加载前的 40 个 task cache、75,159 行预测及哈希 |
| `results/trainfree_trap_probability/cache_new_40task/cache_new_probabilities_label_free.csv.gz` | 不含 success/failure/onset 的逐 query 双校准表概率 |
| `results/trainfree_trap_probability/cache_new_40task/tables/episode_outcome_audit.csv` | 解盲后的 16,000 条 episode first/mean/max 概率 |
| `results/trainfree_trap_probability/cache_new_40task/tables/endpoint_fixed_query_ranking.csv` | 每个绝对 query 的 pooled 与同任务配对 outcome AUC |
| `results/trainfree_trap_probability/cache_new_40task/tables/endpoint_threshold_audit.csv` | 阈值、未见任务、报警时机及等误报时钟对照 |
| `results/trainfree_trap_probability/cache_new_40task/tables/endpoint_phase_probability.csv` | 成功/失败轨迹分 phase 的概率分布 |
| `results/trainfree_trap_probability/cache_new_40task/figures/cache_new_probability_audit.png` | 成功背景、沿途概率和混淆控制总览 |

该外推中 q12 同任务 AUC 仅为 `0.578/0.561`，而 duration control 接近完美；B-table 50% 的 `133/532` endpoint 命中也被更早、更低误报的固定时钟支配。因为没有统一 Trap onset，这些结果只用于否定不可靠的跨任务概率解释。

### 时序约束阈值审计

| 路径 | 内容 |
|---|---|
| `results/trainfree_trap_probability/timing_constrained_alarm/summary.json` | 37-task 开发、3-task 留出、误报预算和 onset 时序的机器摘要 |
| `results/trainfree_trap_probability/timing_constrained_alarm/tables/physical_onset_labels.csv` | 16,000 条 episode 的独立物理标签及 260 个 onset proxy |
| `results/trainfree_trap_probability/timing_constrained_alarm/tables/operating_points.csv` | A/B 概率表在 0.5%--10% 开发误报预算下的冻结阈值、留出时序和固定时钟对照 |
| `results/trainfree_trap_probability/timing_constrained_alarm/tables/fixed_threshold_timing.csv` | 10%--75% 固定阈值的首次报警与 `[-2,0]` 命中率 |
| `results/trainfree_trap_probability/timing_constrained_alarm/tables/heldout_onset_aligned_probability.csv` | 3 个留出任务在 onset `-6..+6` 的概率曲线 |
| `results/trainfree_trap_probability/timing_constrained_alarm/figures/timing_constrained_probability_alarm.png` | 误报/及时召回、近 onset 召回和留出对齐曲线 |

开发误报预算不超过 5% 时，近 onset 首次报警最高只有开发 `5/198`、留出 `2/24`。B-table 的 1% 预算阈值为 37.12%，留出及时召回虽为 7/24，但近 onset 只有 1/24，且固定 q39 时钟同样及时命中 7/24。没有找到满足用户时序要求的替代阈值。

## 同噪声输入版本反事实

| 路径 | 内容 |
|---|---|
| `results/input_version_counterfactual/formal_capture/paired_counterfactual_routes.npz` | 96 个相邻输入版本对、4 个配对 noise、旧/新输入各一次，共 768 次推理的 HB full-softmax 与动作 |
| `results/input_version_counterfactual/formal_capture/manifest.json` | 输入、checkpoint、配对设计、哈希与标签隔离声明 |
| `results/input_version_counterfactual/analysis/summary.json` | 无量纲 `input-effect / noise-effect` 分数、自然边界报警和 replay fidelity |
| `results/input_version_counterfactual/analysis/tables/primary_scores.csv` | 8 条轨迹 x 12 个相对 query 的逐记录主分数 |
| `results/input_version_counterfactual/analysis/tables/comparison_to_self_reference_v3.csv` | 同一 1+7 轨迹与现有 v3 报警时序对照 |
| `results/input_version_counterfactual/modality_capture/modality_counterfactual_routes.npz` | `V00/V01/V10/V11` 的 2x2 图像/状态反事实，40 对、320 次新增推理 |
| `results/input_version_counterfactual/modality_analysis/summary.json` | 视觉、本体状态、noise 对前后层 state/action routing 的分解 |
| `results/input_version_counterfactual/modality_analysis/input_novelty_summary.json` | 原始像素和 8 维状态变化量的事后混杂审计，不进入报警 |
| `results/input_version_counterfactual/modality_analysis/tables/action_output_counterfactual.csv` | action-only 同噪声反事实基线 |
| `results/input_version_counterfactual/modality_analysis/figures/failed_input_version_pairs.png` | 失败轨迹 `-2..+2` 的相邻输入版本可视化 |

失败轨迹仅在抓取前一轮满足自然边界 `C=0.539<1`，7 条成功对照为 `1.593--2.132`；但该轮图像和状态变化本身也低于全部成功对照。抓空后 `+1/+2` 的视觉与状态效应均进入后层 action routing，并高于全部成功对照，因而否定“反馈被 MoE 截断”的强解释。该实验是 1 条失败对 7 条同 snapshot 成功 sibling 的机制探针，不是总体 detector 验证。完整报告见 [INPUT_VERSION_COUNTERFACTUAL_ZH.md](INPUT_VERSION_COUNTERFACTUAL_ZH.md)。

## 无任务先验的自参照 MoE 报警

| 路径 | 内容 |
|---|---|
| `results/task_free_self_reference_selector/summary.json` | 40-task 描述统计、排除开发任务后的 39-task 严格结果及固定时钟负控 |
| `results/task_free_self_reference_selector/episode_predictions.csv` | 16,000 条 episode 的冻结 MoE 预测与第二阶段 outcome |
| `results/task_free_self_reference_selector/query_decisions.csv` | 逐 query 的三个自参照 MoE 比率与报警决策 |
| `results/task_free_self_reference_selector/task_metrics.csv` | 逐任务 TPR/FPR 与异质性 |
| `results/task_free_self_reference_selector/clock_baselines.csv` | phase 0.10--1.00 的 endpoint 时钟负控 |
| `results/task_free_self_reference_selector/gpu6_audit/summary.json` | GPU6 16 条 prospective、物理 plateau 和跨 GPU same-seed 分叉汇总 |
| `results/task_free_self_reference_selector/gpu6_audit/online_episodes.csv` | GPU6 两个 init x 八个 seeds 的逐 episode 结果 |
| `results/task_free_self_reference_selector/gpu6_audit/physical_timing.csv` | 唯一失败的盘子 plateau、MoE alarm 与固定时钟对齐 |
| `results/task_free_self_reference_selector/gpu6_audit/same_seed_cross_gpu.csv` | 旧 GPU3 失败条件与 GPU6 replay 的 q0 数值差异 |
| `results/task_free_self_reference_selector/gpu6_audit/gpu6_self_reference_audit.png` | 16 条结果、路由 gate 和物理时序三联图 |
| `results/task_free_self_reference_selector/gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/` | 唯一在线报警失败的 clean、annotated full 和 clip 视频 |

严格 held-out 为 failure `78/530`、success false alarm `35/15070`；GPU6 小样本为 1 个失败报警、15 个成功不报警。该失败先在 `q16` 进入物理 plateau，后在 `q20` 报警；所以当前是 post-event phenotype signal。endpoint 上 phase-0.81 时钟占优，不能把 timeout 分类当作 early-warning 证据。完整解释见 [TASK_FREE_SELF_REFERENCE_ALARM_ZH.md](TASK_FREE_SELF_REFERENCE_ALARM_ZH.md)。

## 任务难度与报警预算加权

| 路径 | 内容 |
|---|---|
| `results/task_difficulty_weighting/summary.json` | 37 tasks、14,800 episodes、标签隔离、cross-fit 和四种预算分配的机器摘要 |
| `results/task_difficulty_weighting/REPORT_ZH.md` | 自动生成的完整数值报告与解释边界 |
| `results/task_difficulty_weighting/first_query_moe_features.csv` | outcome 加载前写出的 14,800 行首 query HB MoE-only 标量 |
| `results/task_difficulty_weighting/first_query_action_route_embeddings.npz` | outcome 加载前写出的 `14,800 x 8 x 32` action-route embedding |
| `results/task_difficulty_weighting/task_difficulty.csv` | 37 个任务的全量、seed-half、init-parity 失败率与 Wilson 区间 |
| `results/task_difficulty_weighting/early_route_difficulty_associations.csv` | 15 个首 query route 标量的双向 cross-fit Spearman、置换检验与 BH 校正 |
| `results/task_difficulty_weighting/early_route_knn_predictions.csv` | `k=1/3/5/10` 的两方向 leave-one-task route-KNN 难度预测 |
| `results/task_difficulty_weighting/weighting_summary.csv` | uniform、经验 sqrt/linear、route-KNN sqrt 的 micro/macro/FPR/cluster-CI 汇总 |
| `results/task_difficulty_weighting/weighting_fold_evaluation.csv` | 两个 seed-half 交换方向分别评价 |
| `results/task_difficulty_weighting/weighting_task_evaluation.csv` | 逐任务的命中与误报增量，用于审计收益集中性 |
| `results/task_difficulty_weighting/weighting_task_allocations.csv` | 每 fold、每任务、每方法的先验、预算与阈值 |
| `results/task_difficulty_weighting/weighting_episode_predictions.csv` | 14,800 条 episode 的四种冻结决策 |
| `results/task_difficulty_weighting/difficulty_weighting_audit.png` | 难度稳定性、micro/macro 取舍和逐任务贡献四联图 |

核心结果：任务失败率跨 seed halves 稳定（`rho=0.837`）；首 query route 可识别 task `98.96%`，但 route-KNN 难度 MAE 均差于 median baseline。经验线性 prior 把 micro recall 从 `65.3%` 提至 `76.2%`，macro task recall 却从 `65.2%` 降至 `61.3%`，且 53 个新增命中中 42 个来自最难任务。完整解释见 [TASK_DIFFICULTY_WEIGHTING_ZH.md](TASK_DIFFICULTY_WEIGHTING_ZH.md)。

## cache_new 严格 MoE-only 离线回放

| 路径 | 内容 |
|---|---|
| `results/moe_dynamics_online_alarm/cache_new_task8_replay/summary.json` | 固定 v2 对 400 条、17,460 routing query 的 MoE-only 回放摘要和输入隔离声明 |
| `results/moe_dynamics_online_alarm/cache_new_task8_replay/query_predictions_moe_only.csv` | 在加载物理标签前写出的 17,060 条可评分 query；每条轨迹首 query 只建立基线 |
| `results/moe_dynamics_online_alarm/cache_new_task8_replay/episode_predictions_moe_only.csv` | 在加载物理标签前写出的 400 条逐 episode 预测 |
| `results/moe_dynamics_online_alarm/cache_new_task8_replay/episode_evaluation_posthoc.csv` | 第二阶段加载 outcome 后的通用失败评价 |
| `results/moe_dynamics_online_alarm/cache_new_task8_replay/posthoc_summary.json` | query-sampled 漏抓代理、Fisher/AUC、旧 B 重合和代理有效性审计 |
| `results/moe_dynamics_online_alarm/cache_new_task8_replay/posthoc_missed_grasp_audit.csv` | 400 条逐 episode 事后分型及时序 |

固定 v2 在全部 400 条上为 failure `29/138`、success false alarm `13/262`；旧 B 未见的 272 个条件上为 `20/89` 和 `5/183`。目标未抓住代理只有 `13/53` 正式报警、`9/53` 在事件前 5 query 或当下报警。详细边界见 [CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md](CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md)。

## A/B/prospective 严格 MoE-only 回放

| 路径 | 内容 |
|---|---|
| `results/moe_dynamics_online_alarm/large_offline_replay/summary.json` | A 352、B 512、prospective 24 条的固定 v2 汇总与 24/24 在线序列一致性 |
| `results/moe_dynamics_online_alarm/large_offline_replay/query_predictions_moe_only.csv` | 39,282 条可评分 query 预测，标签加载前写出 |
| `results/moe_dynamics_online_alarm/large_offline_replay/episode_predictions_moe_only.csv` | 888 条纯预测结果，标签加载前写出 |
| `results/moe_dynamics_online_alarm/large_offline_replay/episode_evaluation_posthoc.csv` | 标签加载后的 outcome/failure-family 评价 |

A 的 failure alarm/success false alarm 为 `183/235`、`18/117`；B 为 `34/216`、`8/296`；prospective 为 `8/14`、`1/10`。A/B 是 retrospective stress test，不能当作独立验证。

## 修正后的 MoE dynamics v2

| 路径 | 内容 |
|---|---|
| `results/moe_dynamics_online_alarm/calibration_v2/calibration.json` | 5 条健康 leave-one-out 得到的冻结阈值 `1.0457019658` |
| `results/moe_dynamics_online_alarm/calibration_v2/healthy_loo_query_scores.csv` | 健康校准逐 query acceleration excess |
| `results/moe_dynamics_online_alarm/offline_replay/summary.json` | v2 在旧数据上的开发回放，明确标记非 prospective |
| `results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/` | 全新 seed 固定 24 条在线轨迹与 27 个报警视频 |
| `results/moe_dynamics_online_alarm/gpu5_server_seed20260908/routes.zarr` | 1129 行服务端完整 HB routing |
| `results/moe_dynamics_online_alarm/prospective_audit/summary.json` | 严格漏抓/其他失败/成功分层、Wilson CI、v1/v2 配对和 capture 审计 |
| `results/moe_dynamics_online_alarm/prospective_audit/episode_audit.csv` | 每条轨迹的 v2 在线报警、v1 配对重放和漏抓时序 |

同一批 prospective 轨迹上，旧 v1 到新 v2：严格漏抓 `1/7 -> 5/7`，全部失败 `1/14 -> 8/14`，成功误报 `0/10 -> 1/10`；失败配对 McNemar `p=0.015625`。但后续 400 条 `cache_new` 回放没有维持该目标召回，不能只引用这 24 条。完整解释和视频见 [MOE_DYNAMICS_ALARM_V2_ZH.md](MOE_DYNAMICS_ALARM_V2_ZH.md)。

## 不使用物理距离的 MoE-only 在线报警

| 路径 | 内容 |
|---|---|
| `results/moe_only_online_alarm/failure_type_audit/summary.json` | 三批 40 条在线轨迹按目标漏抓、其他失败、成功分层后的正式结果 |
| `results/moe_only_online_alarm/failure_type_audit/episode_audit.csv` | 逐轨迹 raw/alarm、两锅漏抓事件和通用物理 taxonomy |
| `results/moe_only_online_alarm/method_diagnosis/` | persistence 排除、事件窗口特征、AUC、5000 次置换和 maxT 结果 |
| `results/moe_only_online_alarm/gpu4_fresh_random_seed20260905/` | preliminary 8 条在线轨迹和逐 query 完整 HB routing |
| `results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz` | 从 preliminary 的 5 条成功轨迹建立的 H20 calibration reference |
| `results/moe_only_online_alarm/gpu4_heldout_random_seed20260906/` | 规则冻结后的 held-out 8 条在线轨迹 |
| `results/moe_only_online_alarm/gpu5_extended_random_seed20260907/` | 冻结同一规则后的 GPU5 24 条在线轨迹与 3 组报警视频 |
| `results/moe_only_online_alarm/gpu4_server_seed20260905/routes.zarr` | preliminary 服务端 349 行完整 HB routing |
| `results/moe_only_online_alarm/gpu4_server_seed20260906/routes.zarr` | held-out 服务端 381 行完整 HB routing |
| `results/moe_only_online_alarm/gpu5_server_seed20260907/routes.zarr` | GPU5 extended 服务端 1114 行完整 HB routing |
| `results/moe_only_online_alarm/review_videos/` | 目标漏抓、其他失败、成功 raw reject 三类确定性事后回放 |

三批合并后，正式 persistence-2 规则为目标漏抓 1/9、其他失败 2/13、成功误报 0/18。GPU5 冻结测试中目标漏抓为 1/5，放宽 persistence 仍未提高；但事后发现后层 route acceleration 在漏抓后窗口有信号。详细时序、统计边界和报警视频见 [MOE_ONLY_ONLINE_ALARM_ZH.md](MOE_ONLY_ONLINE_ALARM_ZH.md)。

## GPU 4 在线 belief-state 报警

| 路径 | 内容 |
|---|---|
| `results/online_belief_alarm/summary.json` | 4 条 fresh random 与 1 条 H20 replay 的总结果和 233-row 一致性审计 |
| `tables/episode_summary.csv` | 每条轨迹的 outcome、闭合、报警、继续执行和物理位移 |
| `tables/alarm_events.csv` | 5 个报警事件的三项路由比值与事后 failed-grasp 几何 |
| `fresh_random_gpu4_20260904/` | 4 条全新随机在线轨迹；2 条报警 episode 含 clean/full/clip 视频 |
| `replay_candidate06_gpu4_20260903/` | 旧随机噪声在 H20 上的在线重放和假阳性视频 |
| `gpu4_server_20260903/routes.zarr` | 服务端保存的 233 行完整 HB 路由 |
| `gpu4_server_20260903/capture_summary.json` | hook、自检、耗时和完整性摘要 |

详细解释和可点击视频见 [ONLINE_BELIEF_ALARM_GPU4_ZH.md](ONLINE_BELIEF_ALARM_GPU4_ZH.md)。fresh random 中有 1 个真实 failed-grasp 报警和 1 个成功抓取假阳性；当前规则不具备独立部署可靠性。

## Train-free belief selector

| 路径 | 内容 |
|---|---|
| `results/trainfree_belief_selector/summary.json` | `transition_split_v1` trigger 与通用 K8 ranker 负控结论 |
| `tables/trigger_decisions.csv` | 失败和 7 条健康 LOO 对照的 64 个逐 query ACCEPT/REJECT 决策 |
| `tables/candidate_pool_choices.csv` | 5 tasks、80 states、320 个 K8 池的 min-gap 选择 |
| `tables/candidate_task_summary.csv` | 逐任务 selected success、随机期望和差值 |
| `figures/trainfree_belief_selector_audit.png` | trigger 三条件与 K8 负控结果 |

详细解释见 [TRAINFREE_BELIEF_SELECTOR_ZH.md](TRAINFREE_BELIEF_SELECTOR_ZH.md)。当前支持的是 train-free stale-chunk rejector；通用 min-gap candidate ranker 比随机低 1.13 个百分点，明确不作为已验证 recovery controller。

## Belief-state mismatch

| 路径 | 内容 |
|---|---|
| `results/belief_state_mismatch/summary.json` | 抓空后物理/动作/MoE 三层证据、核心关系和外部 gate 审计 |
| `tables/belief_physics_alignment.csv` | 每个相对 query 的末端/物体位移、分离距离和夹爪命令 |
| `tables/action_phase_consistency.csv` | 失败 action 到同阶段和最近成功阶段的距离 |
| `tables/hb_token_match_to_success.csv` | final-flow 前/后层 state/action soft-route 距离 |
| `tables/hb_flow_token_match_to_success.csv` | 逐 query × flow × layer-group × token-type 距离 |
| `tables/hb_flow_state_action_gap.csv` | 逐 flow 的 HB state/action gap |
| `tables/hb_action_position_match_to_success.csv` | 10 个 action positions 的完整 flow 路由差异 |
| `tables/hb_state_action_gap_layer.csv` | 模型层 2--5、12--15 的持续 gap 定位 |
| `tables/hb_cross_chunk_response.csv` | 新观测后的前/后层跨 chunk 跳变 |
| `figures/belief_state_mismatch_signature.png` | 物理脱耦、动作 phase 和路由分裂总览 |
| `figures/belief_state_mismatch_flow_position.png` | `q+2` 逐 flow 与逐 action-position 定位 |

详细解释见 [BELIEF_STATE_MISMATCH_ZH.md](BELIEF_STATE_MISMATCH_ZH.md)。核心是 `q+2` 的 front state/action 分裂和 layer-5 持续 gap；这是 train-free 单案例机制证据，不是已泛化分类器。

## 失败抓取 AS/HB 动力学

| 路径 | 内容 |
|---|---|
| `results/failed_grasp_moe_dynamics/summary.json` | 失败事件、同 snapshot 成功对照、AS/HB 核心数值与限制 |
| `tables/physical_event_alignment.csv` | 闭合 query、chunk 内动作位置、末端距离和后续抬升 |
| `tables/action_token_event.csv` | 失败与成功抓取的逐 action-token 命令 |
| `tables/as_route_state.csv` | 16,180 chunks 上四个 AS 层的恒定性审计 |
| `tables/hb_event_layer_metrics.csv` | 逐 HB 层的瞬时 flow/chunk 指标 |
| `tables/hb_event_group_summary.csv` | 前后 HB 层相对 7 条成功对照的范围与秩 |
| `tables/hb_instantaneous_expert_paths.csv` | 逐层逐 flow 的 top-1/top-4 与 soft distance |
| `tables/hb_flow_token_distance.csv` | 逐 flow × action-token 的失败到成功中心距离 |
| `tables/aligned_chunk_dynamics.csv` | 闭合事件前后 -8 到 +8 chunks 的连续状态 |
| `figures/failed_grasp_moe_dynamics.png` | 物理、瞬时 HB、连续 HB 和 flow 演化总览 |
| `figures/hb_continuous_flow_dynamics.png` | 前后层 late-flow volatility/acceleration 连续曲线 |
| `figures/hb_flow_token_distance_heatmap.png` | 前后层 flow × token 距离热图 |

详细解释见 [FAILED_GRASP_MOE_DYNAMICS_ZH.md](FAILED_GRASP_MOE_DYNAMICS_ZH.md)。该实验是 1 个失败抓取对 7 个同 snapshot 成功抓取的 train-free 个案分析，不作为跨任务 detector 性能证据。

## Train-free signal matrix

| 路径 | 内容 |
|---|---|
| `results/trainfree_signal_matrix/summary.json` | seed、2000 permutations、无训练声明、feature audit |
| `tables/fixed_time_auc.csv` | 48 行固定 t=20/25/30 的 routing 评估 |
| `tables/comparison_auc.csv` | d9 与 action-only 固定时点对照 |
| `tables/onset_alignment.csv` | 1092 行聚合 Trap onset 对齐结果 |
| `tables/type_onset_alignment.csv` | 1092 行 loop/static 分层结果与 maxT |
| `tables/*_group_effects.csv` | 组级统计输入，可重做置换审计 |
| `tables/aligned_event_values.csv` | onset 对齐的事件级数值 |
| `tables/macro_states.csv` / `macro_transition.csv` | 确定性宏状态与转移矩阵 |
| `intermediate/corpus_*_route_features.npz` | 从两套 full routing Zarr 提取的逐行标量缓存 |
| `figures/loop_precursor_vs_action_baselines.png` | 最关键的 loop lead=-2 对照图 |
| `figures/onset_aligned_signals.png` | onset 对齐均值曲线 |
| `figures/trainfree_signal_matrix.png` | 全信号矩阵 |

## Snapshot-Fork recovery

| 路径 | 内容 |
|---|---|
| `tables/recovery_window_summary.csv` | init-7 成功 trunk 与 init-3 三窗口恢复率 |
| `tables/recovery_signal_audit.csv` | init-3 信号相对 22 条历史 no-loop controls 的百分位 |
| `runs/init07_seed20260830_trunk_success/` | 预定首个 trunk；当前 H20 上成功，未分叉 |
| `runs/init03_seed20260903_loop/trunk.*` | 52-query 失败 trunk、完整路由与视频 |
| `runs/init03_seed20260903_loop/fork_-4/` | query 32 快照与 8 条 continuation |
| `runs/init03_seed20260903_loop/fork_-2/` | query 34 快照与 8 条 continuation |
| `runs/init03_seed20260903_loop/fork_+0/` | query 36 快照与 8 条 continuation |
| `capture/routes_484_inferences.zarr` | trunk 与所有 fork 的 484 行 server-side full routing |
| `capture/capture_summary.json` | capture shape、hook、耗时和完整性摘要 |

每个 fork 保留 8 份 `candidate_XX.json/.npz`。前两个 candidate 另有代表性 MP4；没有根据 outcome 删除候选。测试 smoke 与首次被外部 SIGTERM 中断的 server capture 不属于正式结果，未复制进本目录。

## 历史 d9 基线

`results/legacy_d9_baselines/` 保存最终报告实际比较的早期实验：final denoise step `d=9` 的 full-softmax route change、Top-4 churn onset 和物理 loop onset。对应代码位于 `code/legacy_baselines/`。

这些结果是历史探索与基线来源，不计入本轮 `2 x 8 x 5` maxT 复现集合。特别是新 `weighted_recurrence` 与 d9 Hellinger mobility 近乎完全负相关，因此不能当作独立新信号。
