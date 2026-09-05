# HiMoE-VLA Trap Experiments

本目录集中保存 HiMoE-VLA 的 train-free Trap 与失败抓取 MoE 动力学实验，包括可运行代码、冻结配置、正式结果、原始 Snapshot-Fork 分支和中文报告。原始目录仍保留；这里是命名清楚、可独立审计的整理版本。

## 一句话结论

在物理 loop onset 前两次重规划，action-token routing 的 `late-flow volatility` 和 `route acceleration` 在 A/B 两套语料中同方向复现；但新的 Snapshot-Fork 在 -4/-2/0 三个触发窗口均为 0/8 恢复。因此当前证据支持 **routing 是 loop precursor sensor**，不支持它已经是可靠在线报警器或 recovery controller。

另一个独立失败抓取个案显示：AS routing 在全 store 严格恒定；HB 在闭合 chunk 内的 late-flow volatility/acceleration 高于 7/7 个同 snapshot 成功对照，且抓空后后层 HB 在随后两个 chunk 才整体大跳变。该结果是机制个案，不是泛化检测性能。

对该个案进一步按 belief/physics mismatch 对齐后，`q+2` 起出现稳定的 HB state/action 分裂：前层 state route 在完整 10/10 flow 上偏离成功范围，而前层 action route 仍保持成功搬运模式；模型第 5 层的 state/action gap 持续到 `q+8`。这支持“状态侧出现纠错信息、动作侧仍沿用 stale 抓取信念”，但仍只是 1 个失败对 7 个同 snapshot 成功对照。

新的同噪声输入版本反事实进一步收窄并修正了上述解释。固定 flow noise 后，抓空后的 `q+1/q+2` 中，新图像和新 8 维状态对后层 action routing 的效应均高于 7/7 成功对照；新反馈并没有在 MoE 中被截断。失败轨迹在抓取前一轮出现自然的 `input-effect / noise-effect = 0.539 < 1`，而成功对照为 `1.593--2.132`，但该时刻原始输入本身也异常静止，且 action-only 反事实同样明显。因此目前最准确的机制表述是：**MoE 可测量动作计算对反馈相对于采样噪声的敏感度，但不能单独读出 belief 的语义；本个案是一个候选报警机制，不是已泛化的失败分类器。**

基于该分裂实现了 train-free `ACCEPT / REJECT_STALE_CHUNK` 选择器：开发案例只在 `q+2` 拒绝，7 条健康对照的 56 个 leave-one-out 决策为 0 次触发。独立 5-task K8 负控同时表明，直接选择 layer-5 gap 最小的候选比随机低 1.13 个百分点，因此当前可用的是 rejector，不是恢复 action chooser。

GPU 4 在线测试进一步给出了更严格的边界：4 条全新随机轨迹中，规则报警了 1 条真实 failed-grasp，但也误报了 1 条最终成功的抓取；另 2 条失败没有进入“近锅闭合”窗口。报警后轨迹均继续执行并保存视频。故当前规则只能算候选在线 signal，**还不是可靠报警器**。

进一步移除物理距离和夹爪事件后，又在 GPU5 用冻结规则完成 24 条在线轨迹。GPU5 上 10 成功、14 失败，正式报警 3 条且均为失败，但严格漏抓型只命中 1/5，并且晚于漏抓 7 query；放宽 persistence 也没有提高目标召回。事后特征诊断发现后层 route acceleration 在漏抓后 5 query 内有分离（GPU5 AUC 1.00，48 项 maxT `p=0.0118`），但漏抓前没有校正后成立的信号。故当前证据否定的是旧三条件规则，不是否定 MoE telemetry；它尚未成为经验证的 early detector。

基于该诊断构造了 train-free `back_front_route_acceleration_v2`：后层/前层完整去噪 acceleration 相对同阶段健康上界连续越界即报警，阈值只由 5 条健康 leave-one-out 轨迹确定。在全新 GPU5 seed `20260908` 的 24 条 prospective 轨迹上，同轨迹旧 v1 对严格漏抓为 1/7、全部失败 1/14，新 v2 提高到 5/7 和 8/14；成功误报由 0/10 增为 1/10。总体失败的配对 McNemar `p=0.0156`。因此规则已获得实质修正，但仍是风险报警器而非可部署的确定性失败判定器。

随后对 `cache_new` 同任务的 400 条新 capture 做了严格两阶段离线回放：先仅凭 MoE 写出全部预测，再加载物理标签评价。固定 v2 对全部失败报警 `29/138`，成功误报 `13/262`；对 query-sampled 未抓住代理仅命中 `13/53`，及时命中 `9/53`。因此先前 prospective 的 `5/7` 目标召回没有在大样本上保持；当前最准确结论是**低误报、低召回的通用失败风险告警，不是可靠的漏抓类型识别器**。

同一离线实现还回放了 A/B 与已有 prospective，共 888 条；prospective 的 raw/formal alarm 时序与在线记录 24/24 完全一致。A 的 failure/success-alarm 为 `183/235`、`18/117`，B 为 `34/216`、`8/296`，显示同一固定阈值存在明显 capture-distribution 敏感性。

最后对 `cache_new` 中 37 个完整任务、14,800 条轨迹做了任务难度审计。任务失败率在互斥 noise-seed halves 上稳定（Spearman `rho=0.837`）；第一个 query 的 MoE route 在 held-out init 上可识别具体任务 `98.96%`，却不能可靠预测难度，route-KNN 也差于不看 route 的 median baseline。在另一半 rollout outcome 提供经验 task prior 时，线性难度加权把固定 MoE detector 的 micro failure recall 从 `65.3%` 提至 `76.2%`，但 macro task recall 从 `65.2%` 降至 `61.3%`，且跨任务 bootstrap CI 跨 0。故 task prior 可以重分配报警预算，但不能称为 MoE 学出的难度。

为绕开新任务身份与难度先验，又实现了只使用当前 episode 自身路由前缀的 `self_reference_coupling_collapse_v3`。严格排除开发任务后，它在 39 tasks、15,600 episodes 上为 failure `78/530`、success false alarm `35/15070`，即低误报但低召回。GPU6 的 16 条前瞻轨迹中，15 条成功均未报警，唯一失败于 `q20` 报警并继续执行至 horizon；事后物理 plateau 在 `q16`，因此这是晚 4 query 的 post-event response-collapse signal，不是 precursor。phase-0.81 固定时钟在 endpoint 评价上反而占优，明确暴露 timeout 混杂。

最后把 A/B onset 信号改写成显式的跨语料 Trap 概率。离散 Beta-Binomial 校准不训练特征权重，但明确使用 source onset 标签；运行时仍只读取 MoE。主目标为当前或未来 2 query 内 onset，A→B/B→A 的 AUROC 为 `0.819/0.717`、AP 为 `0.136/0.179`，但最大预测概率仅 `35.8%/51.3%`，75% 阈值均为零报警；最高状态在 target 的实测发生率只有 `21.4%/15.8%`。降低到 25% 也出现低 precision 或高 episode 误报。故当前 routing 能提供有限风险排序，**还不能输出跨 capture 可信的 75% Trap 概率**。

将这两张概率表冻结后，又回放了 `cache_new` 全部 40 个完整任务、16,000 条轨迹和 253,722 次推理。最终成功轨迹中只有 `9010/15468` 能活到 q12 获得分数；成功 query 概率中位数为 A-table `6.86%`、B-table `0.27%`。q12 pooled endpoint AUC 为 `0.670/0.685`，但同任务配对后只有 `0.578/0.561`；episode-max 的 `0.868/0.890` 又低于 duration control `0.910`。B-table 50% 虽命中 `133/532` 失败且只报警 `61/15468` 成功，但中位已到 horizon 82.4%，更早的固定时钟在更低误报下命中全部失败。因此该外推揭示的是任务/时长混淆，不是通用 Trap 概率成立。

进一步从 `cache_new` 的逐 query 物理状态重建了 260 个可解释的 stagnation/regression/undo onset，其中 222 个晚于 q12、可用于当前概率规则的时序评价。37 个开发任务按成功轨迹误报预算选阈值，3 个后完成任务保持留出。B-table 在 1% 开发误报预算下选出 `37.12%`：留出集虽然有 `7/24` 次首次报警早于 onset，但只有 `1/24` 落在 `[-2,0]` query，等预算固定时钟同样命中 `7/24` 且近 onset 为 `2/24`。在开发误报预算不超过 5% 的全部 operating point 中，近 onset 首次报警最多仅 `5/198`（开发）和 `2/24`（留出）。所以 **75% 可以放弃，但当前标量不存在满足 onset 时序要求的替代阈值**。

按照后续简化要求，又完全移除旧概率表，只在 `cache_new` 上固定了三个 MoE 健康不变量：去噪收敛失败、state-action 响应脱节和跨 chunk 复返。每条轨迹 q0--q3 自参照，32 个任务的成功轨迹只确认 1% 整段误报阈值，8 个任务严格留出并在预测哈希后翻牌。留出结果为 TP=51、FP=32、failure recall `44.7%`；同实际误报率下 q29 时间钟为 TP=68、FP=27、recall `59.6%`。83 条报警全部来自复返分支，71 个 onset 代理中只有 5 个在 `[-2,0]` 报警。全量描述也只检出 `152/532`，低于 q39 时间钟的 `267/532`。因此该固定判断器**严格未通过**，但这次失败已经与跨语料概率校准无关。

第二个独立 seed 组的完整 40-task/16,000-episode 回放确认了这个负结果。两组严格留出合计 6,400 条轨迹、239 个失败：MoE 判断器为 TP=106、FP=58、FN=133、TN=6,103；164 次报警中 163 次来自 recurrence，只有 1 次来自其余两个头的联合触发。信息审计进一步发现，每次重规划保存 28,160 个 HB 概率，当前规则只访问 14,208 个，并将其压成三个标量；另有 13,952 个概率位置完全未读。AS 在 508,023 次推理中均为任务内静态量，不应加入动态 Trap 报警。下一版因此按锁死、抖动和 state-action 脱节重组，而不再给现有三个标量调阈值。

结构化重组已完成双向冻结回放。新方法保留 `2 layer stages x 3 flow stages x 4 token groups` 的 24-cell 布局，并同时使用 full-softmax、实际 top-4 expert ID 和 selected weight；24 个健康参考任务、8 个逐任务阈值确认任务和 8 个测试任务严格分离。两方向共 6,400 条测试轨迹、239 个失败：结构化规则为 TP=121、FP=17、FN=118、TN=6,144；同协议旧三标量为 TP=79、FP=27、FN=160、TN=6,134。说明未用的 MoE 结构有增量信息。但 120 个可评价物理 onset 中，只有 10 个首次报警落在 `[-2,0]`，38 个在 onset 后才报警，且 138 次报警中 134 次仍由 lock-in 主导。因此它是更好的低误报 endpoint 风险筛查器，仍不是可靠 early Trap detector。

随后对 expert ID 的含义做了专门消融。统一全局重编号在实际数据上不改变 support/selected 距离，说明数字标签没有语义；但将 32 维概率按大小排序、彻底删除 expert 身份后，在相同 22 个成功误报预算下只能检出 `84/239` 个失败，保留固定 expert 坐标时为 `164/239`。每个 chunk 独立重编号后，soft/support 分别只检出 `0/239` 和 `24/239`。同时，实际 top-4、float16 重建 top-4、过滤 p4/p5 tie 的 top-4 分别检出 `116/239`、`122/239`、`115/239`，表明保存的 tie-break ID 并非必要。准确结论是：**编号本身无意义，但同一 checkpoint 中 expert 参数分支的跨 chunk 身份连续性有信息**。该信号仍以晚报为主：120 个 onset 中 soft-coordinate 只有 11 个首次报警落在 `[-2,0]`，73 个发生在 onset 后。

## 目录

```text
himoe-vla_trap/
├── code/
│   ├── analyze_trainfree_signal_matrix.py
│   ├── analyze_trainfree_trap_probability.py
│   ├── evaluate_trainfree_trap_probability_cache_new.py
│   ├── analyze_timing_constrained_probability_alarm.py
│   ├── evaluate_moe_invariant_alarm_cache_new.py
│   ├── validate_moe_invariant_alarm_cache_new.py
│   ├── evaluate_moe_structured_alarm_two_runs.py
│   ├── audit_moe_structured_alarm.py
│   ├── ablate_moe_expert_identity.py
│   ├── analyze_failed_grasp_moe_dynamics.py
│   ├── analyze_belief_state_mismatch.py
│   ├── analyze_task_difficulty_weighting.py
│   ├── collect_input_version_counterfactual.py
│   ├── analyze_input_version_counterfactual.py
│   ├── collect_input_modality_counterfactual.py
│   ├── analyze_input_modality_counterfactual.py
│   ├── audit_input_version_novelty.py
│   ├── moe_self_reference_selector.py
│   ├── evaluate_moe_self_reference_selector.py
│   ├── audit_self_reference_gpu6.py
│   ├── trainfree_belief_selector.py
│   ├── evaluate_trainfree_belief_selector.py
│   ├── collect_online_belief_alarms.py
│   ├── summarize_online_belief_alarm.py
│   ├── build_moe_only_healthy_reference.py
│   ├── build_moe_only_h20_reference.py
│   ├── moe_only_online_selector.py
│   ├── moe_dynamics_online_selector.py
│   ├── build_moe_dynamics_calibration.py
│   ├── evaluate_moe_dynamics_selector.py
│   ├── evaluate_moe_dynamics_large_offline.py
│   ├── audit_moe_dynamics_online_results.py
│   ├── audit_cache_new_moe_dynamics.py
│   ├── collect_online_moe_only_alarms.py
│   ├── audit_moe_only_online_failure_types.py
│   ├── render_moe_only_review_videos.py
│   ├── collect_snapshot_fork_recovery.py
│   ├── serve_with_full_route_capture.py
│   ├── validate_artifacts.py
│   └── legacy_baselines/
├── configs/
│   ├── data_sources.json
│   ├── failed_grasp_moe_dynamics.json
│   ├── belief_state_mismatch.json
│   ├── trainfree_belief_selector.json
│   ├── trainfree_signal_matrix.json
│   ├── trainfree_trap_probability.json
│   ├── trainfree_trap_probability_cache_new.json
│   ├── timing_constrained_probability_alarm.json
│   ├── moe_invariant_alarm_cache_new.json
│   ├── moe_structured_alarm_two_runs.json
│   ├── moe_expert_identity_ablation.json
│   ├── snapshot_fork_recovery.json
│   ├── online_belief_alarm_gpu4.json
│   ├── task_difficulty_weighting.json
│   ├── self_reference_coupling_collapse_v3.json
│   ├── input_version_counterfactual.json
│   ├── counterfactual_open_loop_alarm_v1.json
│   └── task08_physical_trap_definitions.json
├── docs/
│   ├── EXPERIMENT_REPORT_ZH.md
│   ├── FAILED_GRASP_MOE_DYNAMICS_ZH.md
│   ├── BELIEF_STATE_MISMATCH_ZH.md
│   ├── TRAINFREE_BELIEF_SELECTOR_ZH.md
│   ├── ONLINE_BELIEF_ALARM_GPU4_ZH.md
│   ├── MOE_ONLY_ONLINE_ALARM_ZH.md
│   ├── MOE_DYNAMICS_ALARM_V2_ZH.md
│   ├── CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md
│   ├── TASK_DIFFICULTY_WEIGHTING_ZH.md
│   ├── TASK_FREE_SELF_REFERENCE_ALARM_ZH.md
│   ├── INPUT_VERSION_COUNTERFACTUAL_ZH.md
│   ├── TRAINFREE_TRAP_PROBABILITY_ZH.md
│   ├── MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md
│   ├── METHODS_AND_REPRODUCIBILITY_ZH.md
│   └── RESULTS_INDEX_ZH.md
├── results/
│   ├── trainfree_signal_matrix/
│   ├── trainfree_trap_probability/
│   ├── moe_invariant_alarm_cache_new/
│   ├── moe_invariant_alarm_two_run_audit/
│   ├── moe_information_usage_audit/
│   ├── moe_structured_alarm_two_runs/
│   ├── moe_expert_identity_ablation/
│   ├── failed_grasp_moe_dynamics/
│   ├── belief_state_mismatch/
│   ├── trainfree_belief_selector/
│   ├── online_belief_alarm/
│   ├── moe_only_online_alarm/
│   ├── moe_dynamics_online_alarm/
│   ├── task_difficulty_weighting/
│   ├── task_free_self_reference_selector/
│   ├── input_version_counterfactual/
│   ├── snapshot_fork_recovery/
│   └── legacy_d9_baselines/
└── SHA256SUMS
```

## 从这里开始

- 详细结果：[docs/EXPERIMENT_REPORT_ZH.md](docs/EXPERIMENT_REPORT_ZH.md)
- 失败抓取 AS/HB 分析：[docs/FAILED_GRASP_MOE_DYNAMICS_ZH.md](docs/FAILED_GRASP_MOE_DYNAMICS_ZH.md)
- belief-state mismatch 定位：[docs/BELIEF_STATE_MISMATCH_ZH.md](docs/BELIEF_STATE_MISMATCH_ZH.md)
- train-free belief selector：[docs/TRAINFREE_BELIEF_SELECTOR_ZH.md](docs/TRAINFREE_BELIEF_SELECTOR_ZH.md)
- GPU 4 在线报警与视频：[docs/ONLINE_BELIEF_ALARM_GPU4_ZH.md](docs/ONLINE_BELIEF_ALARM_GPU4_ZH.md)
- 不用物理距离的 MoE-only 在线实验：[docs/MOE_ONLY_ONLINE_ALARM_ZH.md](docs/MOE_ONLY_ONLINE_ALARM_ZH.md)
- 修正后的 MoE dynamics v2：[docs/MOE_DYNAMICS_ALARM_V2_ZH.md](docs/MOE_DYNAMICS_ALARM_V2_ZH.md)
- `cache_new` 严格 MoE-only 离线回放：[docs/CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md](docs/CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md)
- 任务难度与报警预算加权：[docs/TASK_DIFFICULTY_WEIGHTING_ZH.md](docs/TASK_DIFFICULTY_WEIGHTING_ZH.md)
- 无任务先验的自参照 MoE 报警与 GPU6 视频：[docs/TASK_FREE_SELF_REFERENCE_ALARM_ZH.md](docs/TASK_FREE_SELF_REFERENCE_ALARM_ZH.md)
- 同噪声输入版本与 2x2 模态反事实：[docs/INPUT_VERSION_COUNTERFACTUAL_ZH.md](docs/INPUT_VERSION_COUNTERFACTUAL_ZH.md)
- 跨语料 train-free Trap 概率与 75% 报警审计：[docs/TRAINFREE_TRAP_PROBABILITY_ZH.md](docs/TRAINFREE_TRAP_PROBABILITY_ZH.md)
- 不使用旧概率表的 `cache_new` MoE 健康不变量盲回放：[docs/MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md](docs/MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md)
- 两个 seed 组的公式、数据结构与迁移审计：[results/moe_invariant_alarm_two_run_audit/REPORT_ZH.md](results/moe_invariant_alarm_two_run_audit/REPORT_ZH.md)
- 当前判断器用了/没用哪些 MoE 信息及重组方案：[results/moe_information_usage_audit/REPORT_ZH.md](results/moe_information_usage_audit/REPORT_ZH.md)
- 结构化 MoE 双向冻结主结果：[results/moe_structured_alarm_two_runs/REPORT_ZH.md](results/moe_structured_alarm_two_runs/REPORT_ZH.md)
- 与旧三标量和固定时间钟的公平审计：[results/moe_structured_alarm_two_runs/audit/REPORT_ZH.md](results/moe_structured_alarm_two_runs/audit/REPORT_ZH.md)
- Expert 编号、身份连续性与 top-4 tie-break 消融：[results/moe_expert_identity_ablation/REPORT_ZH.md](results/moe_expert_identity_ablation/REPORT_ZH.md)
- 方法与复现：[docs/METHODS_AND_REPRODUCIBILITY_ZH.md](docs/METHODS_AND_REPRODUCIBILITY_ZH.md)
- 结果文件索引：[docs/RESULTS_INDEX_ZH.md](docs/RESULTS_INDEX_ZH.md)
- 机器可读配置：[configs/](configs/)

## 快速审计

从工作区根目录运行：

```bash
python himoe-vla_trap/code/validate_artifacts.py
```

验证器检查正式表格规模、无训练声明、三个严格跨语料复现单元、恢复 manifest、24 条候选 continuation、跨 offset 配对噪声、484 行恢复路由 capture、失败抓取的 1+7 匹配、belief-state mismatch 的逐 flow/token/layer 核心关系、train-free selector 的正负两项审计、GPU 4 在线实验的 233 行路由一致性、MoE-only 三批 349/381/1114 行路由一致性、v2 的健康校准与 24 条 prospective 结果、A/B/prospective 的 888 条严格回放、`cache_new` 的 400 条严格后验评价、37-task 难度/预算实验的 14,800 条标签隔离输入、自参照规则的 40-task 离线回放与 GPU6 16 条前瞻在线审计、同噪声 96 对/768 次推理和 2x2 模态 40 对/320 次推理的反事实完整性、A↔B 概率校准与 260 个物理 onset 的时序审计、不依赖旧概率表的新 40-task/16,000-episode MoE 健康不变量盲回放，以及 508,023-query expert 身份/tie-break 消融的冻结预测哈希和等误报负控。

## 范围说明

本目录包含派生结果、recovery 原始数据以及 GPU4/GPU5/GPU6 在线路由与视频。A/B 原始语料分别约 913 MB 和 34 GB，不重复复制；它们的路径、shape、ground-truth 等级和 checkpoint 身份记录在 `configs/data_sources.json`。
