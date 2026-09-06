# MoE-only 开放集状态观察器审计

## 协议

通用模型只读取成功轨迹，不训练 Loop/Static/Failure 分类器。每折 30 个 train states 内部再拆为 20 个 density states 和 10 个 return states；原 calibration states 只定健康误报阈值，test states 只评价。

每个 query 使用 22 个可解释 MoE phenotype；全局 robust scaling、全局 routing-only phase 和 global×phase 健康密度给出到健康动力学的距离。另一个共享 phase-HMM 用完整已观测前缀递推 belief，直接计算当前 routing chord 对健康下一词分布的预测密度。归一化进度只用作离线 phase 回归目标，在线 phase 估计的输入仍只有 MoE。健康 return table 估计异常后三个 query 内返回健康区的概率。Transformer 和失败标签都不参与通用分数。

Scene8 的物理 stasis onset 只是 known-static anchor。其模板只用每折 train states 的 stasis，其他失败仍保持未命名；当前没有可靠 Loop anchor。

## Routing Phase

held-out success 上 routing phase 与离线归一化进度的 Spearman 相关为 0.319；相邻 query 向前推进比例为 0.541。

## 健康全前缀语法

下表只比较 held-out success 上的下一 chord 密度。clock 对照仅根据健康转移矩阵推进；full-prefix 每步用已观测 routing 更新 belief。负差值表示历史改善预测。

| 范围 | clock bits/phenotype | full-prefix bits/phenotype | full-clock |
|---|---:|---:|---:|
| q1+ | 0.0907 | 0.0809 | -0.0098 |
| q4+ | 0.1189 | 0.1060 | -0.0128 |

## Full-40 最终失败读数

这里的阳性仍只是 eventual failure，不等同于真实 Trap。

| q | 方法 | task/state AUC | 95% CI | healthy FPR | failure recall |
|---:|---|---:|---:|---:|---:|
| 3 | instant | 0.506 | [0.480, 0.530] | 0.051 | 0.061 |
| 3 | learned_persistent | 0.506 | [0.488, 0.521] | 0.052 | 0.056 |
| 3 | dwell_persistent | 0.502 | [0.489, 0.515] | 0.050 | 0.059 |
| 3 | ewma_persistent | 0.505 | [0.491, 0.521] | 0.050 | 0.060 |
| 3 | grammar_instant | 0.512 | [0.487, 0.537] | 0.050 | 0.052 |
| 3 | grammar_learned_persistent | 0.506 | [0.485, 0.525] | 0.050 | 0.052 |
| 3 | grammar_dwell_persistent | 0.510 | [0.495, 0.523] | 0.051 | 0.067 |
| 3 | grammar_ewma_persistent | 0.504 | [0.486, 0.521] | 0.050 | 0.069 |
| 3 | joint_instant | 0.509 | [0.486, 0.534] | 0.050 | 0.051 |
| 3 | joint_learned_persistent | 0.512 | [0.494, 0.531] | 0.047 | 0.054 |
| 3 | joint_dwell_persistent | 0.499 | [0.486, 0.512] | 0.049 | 0.057 |
| 3 | joint_ewma_persistent | 0.505 | [0.488, 0.522] | 0.050 | 0.063 |
| 7 | instant | 0.508 | [0.479, 0.538] | 0.050 | 0.066 |
| 7 | learned_persistent | 0.520 | [0.498, 0.540] | 0.053 | 0.057 |
| 7 | dwell_persistent | 0.517 | [0.494, 0.538] | 0.049 | 0.056 |
| 7 | ewma_persistent | 0.522 | [0.498, 0.546] | 0.048 | 0.052 |
| 7 | grammar_instant | 0.532 | [0.504, 0.560] | 0.049 | 0.075 |
| 7 | grammar_learned_persistent | 0.530 | [0.508, 0.551] | 0.051 | 0.077 |
| 7 | grammar_dwell_persistent | 0.529 | [0.509, 0.548] | 0.051 | 0.082 |
| 7 | grammar_ewma_persistent | 0.529 | [0.506, 0.553] | 0.050 | 0.072 |
| 7 | joint_instant | 0.529 | [0.501, 0.553] | 0.050 | 0.057 |
| 7 | joint_learned_persistent | 0.533 | [0.510, 0.554] | 0.048 | 0.064 |
| 7 | joint_dwell_persistent | 0.523 | [0.501, 0.544] | 0.049 | 0.053 |
| 7 | joint_ewma_persistent | 0.521 | [0.496, 0.542] | 0.050 | 0.055 |
| 12 | instant | 0.555 | [0.512, 0.594] | 0.049 | 0.203 |
| 12 | learned_persistent | 0.565 | [0.528, 0.597] | 0.056 | 0.134 |
| 12 | dwell_persistent | 0.565 | [0.536, 0.592] | 0.050 | 0.199 |
| 12 | ewma_persistent | 0.566 | [0.530, 0.598] | 0.049 | 0.199 |
| 12 | grammar_instant | 0.585 | [0.543, 0.619] | 0.050 | 0.237 |
| 12 | grammar_learned_persistent | 0.590 | [0.551, 0.630] | 0.049 | 0.278 |
| 12 | grammar_dwell_persistent | 0.590 | [0.555, 0.626] | 0.049 | 0.293 |
| 12 | grammar_ewma_persistent | 0.587 | [0.546, 0.625] | 0.049 | 0.298 |
| 12 | joint_instant | 0.576 | [0.531, 0.617] | 0.048 | 0.199 |
| 12 | joint_learned_persistent | 0.584 | [0.548, 0.619] | 0.049 | 0.158 |
| 12 | joint_dwell_persistent | 0.577 | [0.542, 0.609] | 0.049 | 0.203 |
| 12 | joint_ewma_persistent | 0.577 | [0.541, 0.613] | 0.049 | 0.227 |

## Scene8 物理 Stasis 锚点

阈值只由 calibration-success 的整条 episode 最大值给出，因此下表是 anytime 5% FPR 目标。

| 方法 | success FPR | onset 前 recall | onset+3 recall | ever recall |
|---|---:|---:|---:|---:|
| instant | 0.162 | 0.076 | 0.081 | 0.239 |
| learned_persistent | 0.122 | 0.107 | 0.122 | 0.249 |
| dwell_persistent | 0.095 | 0.107 | 0.137 | 0.411 |
| ewma_persistent | 0.078 | 0.081 | 0.112 | 0.426 |
| grammar_instant | 0.091 | 0.030 | 0.030 | 0.168 |
| grammar_learned_persistent | 0.057 | 0.046 | 0.051 | 0.168 |
| grammar_dwell_persistent | 0.108 | 0.086 | 0.132 | 0.503 |
| grammar_ewma_persistent | 0.101 | 0.091 | 0.127 | 0.513 |
| joint_instant | 0.098 | 0.030 | 0.030 | 0.218 |
| joint_learned_persistent | 0.084 | 0.015 | 0.015 | 0.117 |
| joint_dwell_persistent | 0.061 | 0.061 | 0.086 | 0.350 |
| joint_ewma_persistent | 0.105 | 0.076 | 0.102 | 0.406 |
| known_stasis | 0.074 | 0.005 | 0.030 | 0.056 |

在 197 条 cross-fit 物理 stasis 中，通用 persistent 分数到 onset+3 的召回为 0.122；known-static 模板为 0.030。漏报只能称为 observer-unseen stasis；它们是 MoE-silent 候选，但尚不能排除特征或模型能力不足。

## 未知表型 Atlas

跨折共收集 2100 个持久偏离事件，HDBSCAN 得到 2 个 cluster；1355 个事件保留为 cluster noise。

这些 cluster 是待审计 routing phenotypes，不是自动获得物理语义的 Trap 类别。只有同时偏离健康、具有持续性、且经过物理审计后，才能升级为新的 known anchor。

## 结论边界

- 通用 open-set detector 的训练和方法固定不读取 failure/stasis 标签；known-static 模板是单独的半监督锚点。
- success 中的短暂高分只可称 transient-like correction；没有环境干预标签时不能断言它完成了物理纠错。
- full-40 未报警的最终失败只能称 unobserved failure；即使 Scene8 有物理 onset，仍需穷尽合理 MoE 读出后才能称 MoE-silent。
- cluster 的 post-hoc failure fraction 只用于安排人工审计，不能用于选择 detector 或报告无偏分类性能。
