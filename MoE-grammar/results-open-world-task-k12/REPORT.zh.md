# MoE-only 开放集状态观察器审计

## 协议

通用模型只读取成功轨迹，不训练 Loop/Static/Failure 分类器。每折 30 个 train states 内部再拆为 20 个 density states 和 10 个 return states；原 calibration states 只定健康误报阈值，test states 只评价。

每个 query 使用 22 个可解释 MoE phenotype；任务内 robust scaling、task-specific routing-only phase 和 task×phase 健康密度给出到健康动力学的距离。另一个共享 phase-HMM 用完整已观测前缀递推 belief，直接计算当前 routing chord 对健康下一词分布的预测密度。归一化进度只用作离线 phase 回归目标，在线 phase 估计的输入仍只有 MoE。健康 return table 估计异常后三个 query 内返回健康区的概率。Transformer 和失败标签都不参与通用分数。

Scene8 的物理 stasis onset 只是 known-static anchor。其模板只用每折 train states 的 stasis，其他失败仍保持未命名；当前没有可靠 Loop anchor。

## Routing Phase

held-out success 上 routing phase 与离线归一化进度的 Spearman 相关为 0.736；相邻 query 向前推进比例为 0.576。

## 健康全前缀语法

下表只比较 held-out success 上的下一 chord 密度。clock 对照仅根据健康转移矩阵推进；full-prefix 每步用已观测 routing 更新 belief。负差值表示历史改善预测。

| 范围 | clock bits/phenotype | full-prefix bits/phenotype | full-clock |
|---|---:|---:|---:|
| q1+ | 0.5359 | 0.5239 | -0.0120 |
| q4+ | 0.5639 | 0.5488 | -0.0151 |

## Full-40 最终失败读数

这里的阳性仍只是 eventual failure，不等同于真实 Trap。

| q | 方法 | task/state AUC | 95% CI | healthy FPR | failure recall |
|---:|---|---:|---:|---:|---:|
| 3 | instant | 0.510 | [0.487, 0.531] | 0.063 | 0.102 |
| 3 | learned_persistent | 0.509 | [0.491, 0.530] | 0.062 | 0.089 |
| 3 | dwell_persistent | 0.503 | [0.486, 0.522] | 0.065 | 0.097 |
| 3 | ewma_persistent | 0.514 | [0.497, 0.531] | 0.067 | 0.106 |
| 3 | grammar_instant | 0.507 | [0.481, 0.530] | 0.059 | 0.081 |
| 3 | grammar_learned_persistent | 0.503 | [0.480, 0.524] | 0.064 | 0.087 |
| 3 | grammar_dwell_persistent | 0.512 | [0.494, 0.529] | 0.064 | 0.106 |
| 3 | grammar_ewma_persistent | 0.510 | [0.487, 0.530] | 0.066 | 0.119 |
| 3 | joint_instant | 0.512 | [0.489, 0.533] | 0.062 | 0.071 |
| 3 | joint_learned_persistent | 0.519 | [0.498, 0.540] | 0.061 | 0.103 |
| 3 | joint_dwell_persistent | 0.518 | [0.502, 0.535] | 0.065 | 0.084 |
| 3 | joint_ewma_persistent | 0.516 | [0.495, 0.537] | 0.064 | 0.099 |
| 7 | instant | 0.556 | [0.529, 0.582] | 0.056 | 0.135 |
| 7 | learned_persistent | 0.558 | [0.528, 0.584] | 0.058 | 0.137 |
| 7 | dwell_persistent | 0.545 | [0.521, 0.568] | 0.062 | 0.127 |
| 7 | ewma_persistent | 0.556 | [0.529, 0.582] | 0.063 | 0.147 |
| 7 | grammar_instant | 0.554 | [0.527, 0.581] | 0.052 | 0.107 |
| 7 | grammar_learned_persistent | 0.553 | [0.527, 0.577] | 0.065 | 0.141 |
| 7 | grammar_dwell_persistent | 0.551 | [0.524, 0.576] | 0.063 | 0.153 |
| 7 | grammar_ewma_persistent | 0.563 | [0.538, 0.589] | 0.066 | 0.160 |
| 7 | joint_instant | 0.543 | [0.517, 0.568] | 0.056 | 0.096 |
| 7 | joint_learned_persistent | 0.557 | [0.530, 0.581] | 0.059 | 0.143 |
| 7 | joint_dwell_persistent | 0.554 | [0.527, 0.581] | 0.065 | 0.140 |
| 7 | joint_ewma_persistent | 0.556 | [0.529, 0.581] | 0.066 | 0.155 |
| 12 | instant | 0.594 | [0.553, 0.633] | 0.055 | 0.295 |
| 12 | learned_persistent | 0.598 | [0.552, 0.640] | 0.059 | 0.350 |
| 12 | dwell_persistent | 0.594 | [0.554, 0.632] | 0.062 | 0.381 |
| 12 | ewma_persistent | 0.620 | [0.575, 0.662] | 0.064 | 0.409 |
| 12 | grammar_instant | 0.592 | [0.546, 0.631] | 0.053 | 0.211 |
| 12 | grammar_learned_persistent | 0.584 | [0.541, 0.622] | 0.066 | 0.317 |
| 12 | grammar_dwell_persistent | 0.596 | [0.558, 0.632] | 0.067 | 0.341 |
| 12 | grammar_ewma_persistent | 0.610 | [0.565, 0.651] | 0.070 | 0.367 |
| 12 | joint_instant | 0.590 | [0.542, 0.637] | 0.053 | 0.263 |
| 12 | joint_learned_persistent | 0.601 | [0.552, 0.645] | 0.063 | 0.373 |
| 12 | joint_dwell_persistent | 0.616 | [0.572, 0.659] | 0.065 | 0.380 |
| 12 | joint_ewma_persistent | 0.616 | [0.568, 0.661] | 0.065 | 0.402 |

## Scene8 物理 Stasis 锚点

阈值只由 calibration-success 的整条 episode 最大值给出，因此下表是 anytime 5% FPR 目标。

| 方法 | success FPR | onset 前 recall | onset+3 recall | ever recall |
|---|---:|---:|---:|---:|
| instant | 0.081 | 0.081 | 0.086 | 0.173 |
| learned_persistent | 0.044 | 0.096 | 0.107 | 0.157 |
| dwell_persistent | 0.111 | 0.234 | 0.264 | 0.467 |
| ewma_persistent | 0.111 | 0.249 | 0.289 | 0.553 |
| grammar_instant | 0.071 | 0.142 | 0.142 | 0.178 |
| grammar_learned_persistent | 0.024 | 0.188 | 0.193 | 0.289 |
| grammar_dwell_persistent | 0.051 | 0.244 | 0.259 | 0.381 |
| grammar_ewma_persistent | 0.068 | 0.213 | 0.228 | 0.365 |
| joint_instant | 0.071 | 0.096 | 0.096 | 0.173 |
| joint_learned_persistent | 0.125 | 0.157 | 0.162 | 0.239 |
| joint_dwell_persistent | 0.091 | 0.269 | 0.284 | 0.406 |
| joint_ewma_persistent | 0.095 | 0.289 | 0.305 | 0.452 |
| known_stasis | 0.061 | 0.086 | 0.112 | 0.188 |

在 197 条 cross-fit 物理 stasis 中，通用 persistent 分数到 onset+3 的召回为 0.107；known-static 模板为 0.112。漏报只能称为 observer-unseen stasis；它们是 MoE-silent 候选，但尚不能排除特征或模型能力不足。

## 未知表型 Atlas

跨折共收集 2757 个持久偏离事件，HDBSCAN 得到 3 个 cluster；2005 个事件保留为 cluster noise。

这些 cluster 是待审计 routing phenotypes，不是自动获得物理语义的 Trap 类别。只有同时偏离健康、具有持续性、且经过物理审计后，才能升级为新的 known anchor。

## 结论边界

- 通用 open-set detector 的训练和方法固定不读取 failure/stasis 标签；known-static 模板是单独的半监督锚点。
- success 中的短暂高分只可称 transient-like correction；没有环境干预标签时不能断言它完成了物理纠错。
- full-40 未报警的最终失败只能称 unobserved failure；即使 Scene8 有物理 onset，仍需穷尽合理 MoE 读出后才能称 MoE-silent。
- cluster 的 post-hoc failure fraction 只用于安排人工审计，不能用于选择 detector 或报告无偏分类性能。
