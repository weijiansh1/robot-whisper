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
| q1+ | 0.0785 | 0.0569 | -0.0216 |
| q4+ | 0.1154 | 0.0886 | -0.0268 |

## Full-40 最终失败读数

这里的阳性仍只是 eventual failure，不等同于真实 Trap。

| q | 方法 | task/state AUC | 95% CI | healthy FPR | failure recall |
|---:|---|---:|---:|---:|---:|
| 3 | instant | 0.506 | [0.480, 0.530] | 0.051 | 0.061 |
| 3 | learned_persistent | 0.506 | [0.488, 0.521] | 0.052 | 0.056 |
| 3 | dwell_persistent | 0.502 | [0.489, 0.515] | 0.050 | 0.059 |
| 3 | ewma_persistent | 0.505 | [0.491, 0.521] | 0.050 | 0.060 |
| 3 | grammar_instant | 0.517 | [0.492, 0.542] | 0.051 | 0.056 |
| 3 | grammar_learned_persistent | 0.509 | [0.491, 0.525] | 0.049 | 0.055 |
| 3 | grammar_dwell_persistent | 0.503 | [0.487, 0.519] | 0.050 | 0.061 |
| 3 | grammar_ewma_persistent | 0.516 | [0.497, 0.535] | 0.049 | 0.073 |
| 3 | joint_instant | 0.508 | [0.485, 0.533] | 0.050 | 0.055 |
| 3 | joint_learned_persistent | 0.512 | [0.494, 0.530] | 0.050 | 0.057 |
| 3 | joint_dwell_persistent | 0.506 | [0.491, 0.521] | 0.048 | 0.057 |
| 3 | joint_ewma_persistent | 0.508 | [0.490, 0.526] | 0.049 | 0.060 |
| 7 | instant | 0.508 | [0.479, 0.538] | 0.050 | 0.066 |
| 7 | learned_persistent | 0.520 | [0.498, 0.540] | 0.053 | 0.057 |
| 7 | dwell_persistent | 0.517 | [0.494, 0.538] | 0.049 | 0.056 |
| 7 | ewma_persistent | 0.522 | [0.498, 0.546] | 0.048 | 0.052 |
| 7 | grammar_instant | 0.540 | [0.509, 0.568] | 0.051 | 0.087 |
| 7 | grammar_learned_persistent | 0.531 | [0.509, 0.552] | 0.056 | 0.078 |
| 7 | grammar_dwell_persistent | 0.528 | [0.509, 0.547] | 0.052 | 0.078 |
| 7 | grammar_ewma_persistent | 0.537 | [0.512, 0.561] | 0.049 | 0.077 |
| 7 | joint_instant | 0.521 | [0.492, 0.550] | 0.049 | 0.066 |
| 7 | joint_learned_persistent | 0.531 | [0.508, 0.553] | 0.050 | 0.065 |
| 7 | joint_dwell_persistent | 0.525 | [0.506, 0.545] | 0.049 | 0.060 |
| 7 | joint_ewma_persistent | 0.523 | [0.500, 0.545] | 0.051 | 0.058 |
| 12 | instant | 0.555 | [0.512, 0.594] | 0.049 | 0.203 |
| 12 | learned_persistent | 0.565 | [0.528, 0.597] | 0.056 | 0.134 |
| 12 | dwell_persistent | 0.565 | [0.536, 0.592] | 0.050 | 0.199 |
| 12 | ewma_persistent | 0.566 | [0.530, 0.598] | 0.049 | 0.199 |
| 12 | grammar_instant | 0.583 | [0.533, 0.620] | 0.050 | 0.239 |
| 12 | grammar_learned_persistent | 0.584 | [0.542, 0.624] | 0.055 | 0.269 |
| 12 | grammar_dwell_persistent | 0.593 | [0.555, 0.632] | 0.049 | 0.278 |
| 12 | grammar_ewma_persistent | 0.593 | [0.547, 0.637] | 0.048 | 0.292 |
| 12 | joint_instant | 0.568 | [0.526, 0.608] | 0.050 | 0.207 |
| 12 | joint_learned_persistent | 0.565 | [0.526, 0.602] | 0.050 | 0.176 |
| 12 | joint_dwell_persistent | 0.574 | [0.537, 0.607] | 0.050 | 0.200 |
| 12 | joint_ewma_persistent | 0.562 | [0.523, 0.600] | 0.051 | 0.222 |

## Scene8 物理 Stasis 锚点

阈值只由 calibration-success 的整条 episode 最大值给出，因此下表是 anytime 5% FPR 目标。

| 方法 | success FPR | onset 前 recall | onset+3 recall | ever recall |
|---|---:|---:|---:|---:|
| instant | 0.162 | 0.076 | 0.081 | 0.239 |
| learned_persistent | 0.122 | 0.107 | 0.122 | 0.249 |
| dwell_persistent | 0.095 | 0.107 | 0.137 | 0.411 |
| ewma_persistent | 0.078 | 0.081 | 0.112 | 0.426 |
| grammar_instant | 0.095 | 0.036 | 0.036 | 0.157 |
| grammar_learned_persistent | 0.078 | 0.046 | 0.056 | 0.168 |
| grammar_dwell_persistent | 0.088 | 0.112 | 0.147 | 0.497 |
| grammar_ewma_persistent | 0.088 | 0.107 | 0.137 | 0.543 |
| joint_instant | 0.091 | 0.036 | 0.036 | 0.203 |
| joint_learned_persistent | 0.057 | 0.010 | 0.010 | 0.071 |
| joint_dwell_persistent | 0.068 | 0.071 | 0.096 | 0.396 |
| joint_ewma_persistent | 0.101 | 0.076 | 0.102 | 0.421 |
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
