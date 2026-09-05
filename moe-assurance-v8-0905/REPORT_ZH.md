# HiMoE-VLA MoE Assurance v8：九条路线的完整复现与机制结论

日期：2026-09-05

## 结论先行

九条路线已经全部尝试。最重要的结果不是得到一个新的“通用 MoE confidence”，而是证明当前数据支持三个不同层级，并且不能互相冒充：

```text
MoE 内部证据          模式后验                 真实结果把握
route/graph signals -> P(H,C,PL,L,PS,S) -> P(outcome | same snapshot forks)
```

当前最稳的内部本质有两条：

1. **loop 是 flow 内部转向失稳**。`route_acceleration` 在 onset-2 的 matched AUC 为 main 0.628、外部 grid 0.678；`flow_total` 外部为 0.680。瞬时 gate 更尖只是一小部分，不能替代路径曲率。
2. **static 是跨 query 的路由锁定**。相邻 query 的 Hellinger route distance 下降，在 onset-2 的 matched AUC 为 main 0.972、外部 grid 0.970；lag-4 distance 为 0.980/0.985，L15 token consensus 上升为 0.900/0.815。

但这两条仍是**内部状态读出**，不是成功概率。q0 的 2,080 个真实 committor cells 表明：这些 MoE 统计量与终局 failure/loop/static rate 在外部 35/29/18 个可评任务上的 task-macro Spearman 都接近 0，置信区间跨 0。main 拟合后迁移到 grid 的结果概率模型多数比常数基率更差。

因此推荐保留 train-free 的分型证据，不推荐部署 HSMM 或 MoE-to-committor 拟合器。真正的 `MoE Assurance Tensor` 已建立，但目前只有 q0 `continue x terminal outcome` 和少量 recovery 切片有真实 fork 支撑，不能伪造尚未采集的 `h=1/2/4` 概率。

## 与 v7 的差别：不是“少用 MoE”，而是用途不同

用户指出 v7 最终规则只使用少数 MoE 量，这个判断是对的。v7 虽从 `hb_router_probs[8,10,11,32]` 重算，但很快压缩为：

```text
relative freeze OR (persistent flow acceleration AND recurrence loss)
```

v8 则先尽量展开现有可复现缓存，再决定哪些轴能跨语料站住：

| 项目 | v7 intrinsic guard | v8 assurance audit |
|---|---|---|
| 目标 | 单一在线风险报警 | 解释内部结构，并区分证据/模式/结果概率 |
| 最终输入维度 | 3 个动态原语 | 109 个候选轴（含 4 x 13 个 per-layer graph 轴） |
| token-expert 结构 | 主要压成 mobility/curvature/recurrence | 每层保留 13 个图量：entropy、top mass、margin、consensus、rank、MI、occupancy、support 等 |
| flow 结构 | curvature 标量 | 9 个相邻 denoise transitions + 聚合量 |
| query 结构 | lag recurrence loss | lag 1..8 的 route distance 全谱 |
| 风险类型 | freeze 与 turbulence 做 OR | loop 与 static 独立通道，禁止方向抵消 |
| 任务身份 | 运行时不用 | 运行时不用；只在评估时分层 |
| 结果概率 | 不提供 | 只在真实 fork cells 提供 |
| 主要价值 | 高召回保护器 | 机制审计和概率边界 |

这里还要诚实说明数据边界：v8 的 109 轴来自已有缓存中的 40 个聚合量、L12-L15 final-flow 的 `4 x 13` token-expert 图量、9 个 flow transitions 和 8 个 recurrence lags。它比 v7 的**决策表示**丰富很多，但不是模型全部八个 MoE 层、全部 hidden/expert outputs 的无损副本。现有全语料缓存只为后四层保存了可恢复的完整 soft token-expert distribution；报告没有把“109 轴”包装成“全部内部信息”。

最终推荐仍然稀疏，是因为 109 轴扫描发现大量量不稳定或重复，而不是因为读取时少用了 MoE。正确策略是“用全结构发现本质，再锁定跨语料稳定轴”，而不是强迫每一维进入报警分数。

## 数据、计算和评估口径

### 数据范围

| corpus | tasks | episodes | query rows | 用途 |
|---|---:|---:|---:|---|
| main16x32 | 5 | 2,560 | 51,308 | 机制发现/参考拟合 |
| grid50x8 | 40 | 16,000 | 253,722 | 外部确认/任务切分校准 |
| 合计 | 45 | 18,560 | 305,030 | profile |

q0 outcome assurance 另使用 80 个 main K=32 cells 和 2,000 个 grid K=8 cells。route 5 使用 20 个 snapshot、每个 32 条独立 Gaussian flow-noise continuation。route 6 使用 5 个任务、80 pools、2,560 candidate 的离线 top-4 expert reconstruction。

### Profile 是否偷看结果

双卡 profile 构建不加载 outcome，不使用 task-conditioned feature。每个 query 保存：

- 40 个聚合轴；
- L12-L15 各 13 个 token-expert 二部图量；
- 9 个 flow transition distances；
- lag 1..8 recurrence distances。

main 上只确定轴的方向或构造 discovery composite，grid 保持为外部确认。matched AUC 的正例是 event episode 在固定 lead 的 query，负例是同 task、scene、绝对 query 且 loop/static 都未发生的 episode。

### GPU

`CUDA_VISIBLE_DEVICES=6,7` 下两个 worker 各处理一半任务。两张 NVIDIA H20-3e 都实际执行，峰值各 129.47 MiB。这是特征计算而非训练；数据量不足以合理占满 140 GB 显存，所以没有人为放大 batch 或复制张量。

## 109 轴全扫描得到的本质

### Loop：不稳定承诺，而非单纯高熵/低熵

main 冻结方向后，最稳定的 q-2 轴如下：

| 轴（event 方向） | main AUC | grid AUC |
|---|---:|---:|
| route acceleration 上升 | 0.628 | 0.678 |
| flow transition 4 上升 | 0.623 | 0.656 |
| flow transition 5 上升 | 0.623 | 0.661 |
| flow total 上升 | 0.619 | 0.680 |
| L15 top-1 mass 上升 | 0.637 | 0.609 |

main 选出的五轴等权 percentile composite 为 0.673/0.654，没有超过外部单轴 route acceleration。这说明加入更多 MoE 维度不自动提升泛化；loop 的稳健核心是 denoising 路径的二阶变化，而不是复杂加权。

机制表述是：

```text
瞬时选择略更集中 + flow 路径快速转向 = unstable commitment
```

它不是“gate 越尖越有把握”。route 1 的 commitment q-2 AUC 只有 main 0.543、grid 0.616。

### Static：路由停止更新并同质化

| 轴（event 方向） | main AUC | grid AUC |
|---|---:|---:|
| lag-4 route distance 下降 | 0.980 | 0.985 |
| lag-1 route distance 下降 | 0.972 | 0.970 |
| `mob1_w8` 下降 | 0.969 | 0.959 |
| L15 token consensus 上升 | 0.900 | 0.815 |
| L15 token dispersion 下降 | 0.871 | 0.809 |

static composite 在 main/grid 的 AUC 随 lead 演化为：

| lead | main | grid |
|---:|---:|---:|
| -8 | 0.738 | 0.740 |
| -6 | 0.887 | 0.878 |
| -4 | 0.918 | 0.934 |
| -2 | 0.973 | 0.981 |
| 0 | 0.997 | 0.955 |

但不能把 q-2 写成“两步提前预测”。物理 static onset 的定义需要 7 个连续、相互重叠的 width-2 静止窗口；到 q-2 时静止过程通常已形成约 5 个窗口。这里测到的是**正在形成的 static 被内部 route lock-in 确认**，不是在运动仍正常时预测遥远命运。其首报中位 lead 约 -7，也与静止开始形成的时间相符。

## 九条路线逐项结果

### 路线 1：Gate commitment

实现了 normalized entropy、top-1 mass、top-4 mass、top1-top2 margin 和 `1-H/log(E)`。完全 train-free、零额外前向。

- loop q-2 commitment AUC：0.543/0.616（main/grid）；
- static 的 event-high AUC：0.416/0.438，方向相反；
- 单独使用 margin 接近随机。

结论：只能叫“路由承诺度”。它既不能统一 loop/static，也不能叫成功把握。

### 路线 2：Flow coherence

实现 late-flow volatility、route acceleration 及其经验 percentile 等权组合。完全 train-free、零额外前向。

- route acceleration q-2：0.628/0.678；
- flow-instability composite q-2：0.599/0.673；
- static 对同一量反向：flow-instability detection AUC 0.750/0.700。

结论：这是当前最稳的 loop 内部证据，但外部效果仍只是中等，不足以单独做低误报报警。

### 路线 3：Token-expert graph

从每层 `W[10,32]` 计算 13 个图量，并测试：

- loop graph evidence：commitment + token dispersion + token-expert MI；
- static authority loss：entropy 高、consensus 高、effective rank 低、occupancy 集中、support 变窄；
- static lock-in：lag-1 Hellinger route distance 的低分位。

q-2 外部 AUC 分别为 loop graph 0.581、static authority 0.644、static lock-in 0.970。说明 static 的关键不是单 query 图本身，而是图在 query 间停止更新；后层 token 共识是第二证据。

### 路线 4：Healthy-manifold typicality

对 10 个 MoE 特征做经验分位数、Gaussian rank transform 和 Ledoit-Wolf shrinkage Mahalanobis energy。参考集为 main 中成功且无 loop/static 的 2,148 episodes，每集最多 8 rows，共 17,184 rows。

这不是 failure classifier，但健康样本选择确实使用了 outcome/event 标签。外部 q-2 AUC 为 loop 0.523、static 0.682；loop 不复现，static 中等。结论是不值得为这一条引入 reference fitting。

### 路线 5：Perturbation robustness / basin stability

现有资产是相同物理 snapshot 上 32 个**独立 Gaussian noise** continuation，不是无穷小 `delta noise`，所以没有偷换成 local Lipschitz robustness。

- candidate route distance 与 outcome distance：组均 rho=0.049，permutation p=0.046；效应极弱；
- candidate local route sensitivity 与 progress：rho=0.024，p=0.625；为零结果；
- snapshot-level route dispersion 与 outcome spread：rho=0.536，p=0.017，n=20；
- 强制二盆地的 dominant mass 0.881，但 silhouette 只有 0.148；仅 1 个 snapshot 同时有成功/失败。

结论：群体层面的 route geometry 可能读出状态多模态，但不能给单 candidate 成功把握。真正的 infinitesimal perturbation 仍未被测量。

### 路线 6：Expert disagreement

对 top-4 expert outputs 计算加权方差、cancellation 和 conflict。由于没有 runtime hook，本次是离线重构；top-4 set match 0.885，选中概率 MAE `5.98e-5`，不能称 runtime-exact。

disagreement 与三种归一化 sensitivity 的 task-mean Spearman 分别为：

- post-MoE block sensitivity：0.081，95% bootstrap [0.012, 0.161]；
- routed-output normalized：0.463，[0.416, 0.515]；
- input normalized：-0.280，[-0.448, -0.107]。

符号随分母改变，说明它更像局部几何量。独立 closed-loop failure controls 中，routed-full failure AUC 在 denoise 0/8 为 0.484/0.506，接近随机。结论：重要负对照，不是 outcome confidence。

### 路线 7：HSMM mode belief

实现了 age-expanded 六状态 causal filter：`H,C,PL,L,PS,S`。main 拟合 robust scaling、对角 Gaussian emissions、允许的状态转移和经验 dwell survival；运行时不用 task ID。`C` 训练标签用未来两 query 判断 pulse 是否返回，这只参与训练 target，在线 filter 保持 prefix-causal。

- balanced accuracy：main 0.433、grid 0.400；
- q-2 mode AUC：loop 0.622 -> 0.519，static 0.912 -> 0.794；
- grid horizon-2 Brier：loop 0.0655，task prevalence baseline 0.0298；static 0.0897，baseline 0.0108。

模式分型尚有 static 信息，但概率严重失准；不应部署，也绝不能把 posterior 当 committor。

### 路线 8：Snapshot-fork committor / Assurance Tensor

这是唯一产生真实 outcome assurance 的路线。对每个 q0 physical-state cell，把 K 个 noise rollout 分成 mutually exclusive：

```text
success / loop / static / other_failure
```

使用 `Dirichlet(count + 0.5)`，保存 posterior mean、95% marginal interval 和 support。`assurance_tensor.csv` 中 q0 有 2,080 cells x 4 outcomes；每个 cell 的四个 posterior means 严格和为 1。

但是 MoE -> committor 映射没有跨语料成立。grid 上五个候选特征与 failure/loop/static rate 的 task-macro rho 绝对值均不超过约 0.041，所有 bootstrap CI 跨 0。main 训练五特征 logistic 后迁移到 grid：

| outcome | model Brier | constant Brier | model log loss | constant log loss |
|---|---:|---:|---:|---:|
| failure | 0.0611 | 0.0397 | 0.2172 | 0.1940 |
| loop | 0.01954 | 0.01948 | 0.1021 | 0.1068 |
| static | 0.0412 | 0.0126 | 0.1307 | 0.0902 |

loop 只有 log loss 极小改善，Brier 仍略差；整体不支持可部署映射。

Recovery tensor 也按真实计数输出，但采集未完成：计划 200 trunks，preregistered budget-20 仅完成 1 trunk，另有 8 个不应混合的 legacy trunks。唯一 prereg trunk 中：control 0/8、triggered 7/8、delay+4 4/8、delay+8 0/8；triggered 的 Jeffreys posterior mean 0.833，95% interval [0.546,0.986]。这是单 trunk 机制线索，不是总体恢复率。

当前明确缺失：多 snapshot 的两-query no-loop physical committor，以及足够多完整 recovery trunks。

### 路线 9：Conformal / risk-controlled alarm

使用 healthy episode maximum 和 scene-cluster maximum 做后处理，并做两种审计：

1. grid 的 20 calibration tasks / 20 unseen test tasks；
2. 同 task 内 repeat 0-3 calibration / repeat 4-7 test。

这输出的是风险报警阈值，不是个体概率。unseen-task 结果：

| 通道/单位 | threshold | test episode FPR | q-2 recall | scene-cluster FPR |
|---|---:|---:|---:|---:|
| loop flow-instability / episode max | 0.99361 | 6.31% | 9.94% | 15.84% |
| static lock-in / episode max | 0.92850 | 2.64% | 98.31% | 9.18% |
| static lock-in / scene max | 0.93474 | 0.84% | 95.76% | 3.83% |

Learn-then-Test 风格的有限候选族选择，loop 仍只有 9.63% q-2 recall；static 选择 lock-in q98，test episode FPR 0.957%、scene-cluster FPR 4.34%、q-2 recall 95.76%。同任务 disjoint repeats 的 Bonferroni familywise OR 得到 loop FPR 3.92%/recall 12.06%，static FPR 4.78%/recall 94.19%。

不能把所有 static 通道直接 OR：unseen-task FPR 会升到 13.98%。此外，任务分布迁移和同 scene siblings 不满足字面 iid/exchangeability；这里是 stress test，不宣称无条件有限样本保证。

## 哪些方案保留，哪些放弃

| 路线 | 是否拟合/训练 | 最终判断 |
|---|---|---|
| 1 commitment | 否 | 保留为解释轴，不作 confidence |
| 2 coherence | 否 | 保留；loop 首选内部证据 |
| 3 token-expert graph | 否 | 保留；static lock-in 最强 |
| 4 healthy manifold | 只拟合健康参考 | 不推荐；loop 外部失效 |
| 5 perturbation | 否，多 fork | 保留 snapshot-level 机制线索；个体量为零结果 |
| 6 expert disagreement | 否，离线重构 | 负对照；不作结果把握 |
| 7 HSMM | 是，小型统计拟合 | 不推荐；外部校准差 |
| 8 fork committor | posterior 本身不训练；映射器有拟合 | posterior 保留；MoE 映射器拒绝 |
| 9 conformal/LTT | 仅 calibration | static 可作审计阈值；loop 不够敏感 |

这满足“最好不要训练”的要求：最终机制建议来自 routes 1-3 的 train-free signals 和 routes 5-6 的无训练反事实/负对照。routes 4、7、8 的拟合器是为了验证提案是否成立；结果不好就明确拒绝，没有因为已经实现而强行推荐。

## 建议的实际结构

在线对象应继续采用严格类型边界：

```text
MoEAssuranceState
  evidence:
    commitment
    flow_coherence
    layer_token_expert_graph[4,13]
    recurrence_distance[1..8]
    loop_evidence
    static_lockin_evidence

  mode_belief:
    H/C/PL/L/PS/S              # experimental, not outcome probability

  outcome_assurance:
    list[intervention,horizon,outcome,mean,interval,support]
                                  # only when real fork counts exist
```

推荐决策如下：

- 若目标是广义在线风险保护，继续使用 v7，而不是让 v8 的弱 loop assurance 取代它；
- 若目标是解释 static，本版的低 recurrence distance + L15 consensus 是更干净的内部表型；
- 若目标是“未来两个 chunk 不进入 loop 的概率”，当前应返回 unavailable，而不是把 AUC、HSMM posterior 或 conformal score 填进去；
- 新采集优先做多个 pre-loop snapshot x K paired continuations，直接补齐 `A_q[continue,2,{loop,not-loop}]`；
- recovery 采集必须先完成 preregistered 200 trunks，再比较 triggered/short-chunk/rewind/retract，不能从单 trunk 推总体效果。

## 可复现性与文件

完整命令见 `README.md`。关键机器可读产物：

- `results/final_summary.json`：九路线结论；
- `results/full_moe_axes/summary.json`：109 轴与 external lock；
- `results/route_8/assurance_tensor.csv`：真实概率边界；
- `results/example_assurance_q0.json`：分层 state 实例；
- `results/sealed_manifest.json`：所有代码、报告和结果哈希。

测试覆盖 profile 数值恒等式、HSMM posterior 归一化与 prefix causality、q0 四结果 simplex、recovery 不完整边界、conformal split、109 轴 inventory 和 state 类型分离。

## 文献边界

CARE 和 EMoE 支持将 router concentration、expert disagreement 等用作 token/OOD/generation uncertainty signals，但它们并没有建立 HiMoE-VLA 闭环成功概率。Conformal 和 Learn-then-Test 可以校准 risk rule，但依赖相应的交换性/独立校准条件。EDL 可以输出 Dirichlet evidence，但鉴于现有理论争议和本项目已经拥有真实 fork counts，本版没有增加一个自称 evidence 的神经头。

- CARE: https://arxiv.org/html/2607.26052v1
- EMoE: https://arxiv.org/abs/2505.13273
- Conformal tutorial: https://arxiv.org/html/2107.07511v6
- Learn-then-Test: https://arxiv.org/abs/2110.01052
- Evidential Deep Learning: https://arxiv.org/abs/1806.01768
- EDL critique: https://arxiv.org/html/2402.06160v3
