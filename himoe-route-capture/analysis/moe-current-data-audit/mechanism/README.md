# 当前 MoE 机制证据审计

## 审计结论

现有证据支持的最强表述是：**HB-MoE route 是当前状态和 noisy action latent
条件下的局部去噪计算路径，同时可以作为低成本 diagnostic readout。** 它不是每轮
独立重置的白噪声，expert outputs 也不是完全相同或完全不起作用。

现有证据不支持更强的四个命题：

1. route 比当前 action latent 更早知道最终 action basin；
2. expert-output/functional route 在控制 hidden、shared、action/noise 后仍有稳定增量；
3. 跨 control 的 route dynamics 构成独立于 robot state/proprioception 的技能状态；
4. 某个 expert ID 对动作语义、物理后果或成功率具有稳定因果职责。

因此，本审计把当前机制证据定为 **Level 1 / exploratory**。可以写“参与局部计算、
包含可读诊断信号”，不能写“提前形成高层意图”“专家技能状态机”或“因果控制开关”。

## 本次做了什么

本次没有启动模型、simulator 或新统计搜索，也没有修改既有分析脚本。工作内容是：

- 逐项复核既有报告与 `summary.json`；
- 对照样本量、效应值、CI/p 值、实验状态和明确限制；
- 识别共享数据，防止把同一 capture 的多个分析当成独立复现；
- 区分实际运行、离线重建、工程 smoke、撤回结果和仅预注册协议。

机器可读审计见 [summary.json](summary.json)。

## 证据账本

| 证据组 | 实际状态 | 样本 | 可保留结论 | 不能外推 |
|---|---|---:|---|---|
| action commitment / unsupervised route / weighted noise | 既有真实 capture 已运行；本次仅复核 | 1 task、1 scene、4 rollouts、44 queries、K16 | route 跟随 current latent 和局部更新；action geometry 末段才定型 | 跨任务 action commitment、route 领先 action |
| multi-task pruning shadow | 既有 retrospective 分析已运行 | 10 tasks、701 K8 pools | route 优于随机保留 action consensus | route 不优于 initial-noise baseline；无闭环收益 |
| K32 route trajectories | 既有完整 rollout 已运行 | 1 state、32 rollouts | route 后段重组增强；无简单成败分界 | 跨任务 outcome signature |
| control-route speed | 既有跨任务分析已运行 | 30 tasks、1,500 rollouts、18,384 queries | route 跨 control 有连续性和 lag 结构 | 物理速度、任务阶段语义、value |
| state-token Markov | 既有分析已运行 | 5 tasks、2,560 rollouts | 下一 route state 可预测 | 超过 proprio 的独立 latent state |
| continuous denoise Markov transfer | 既有分析已运行 | 3 tasks x 64 candidates x 2 cohorts | denoise path 本身可预测 | held-task outcome selection；宏平均为负 |
| expert activation future | 既有 offline reconstruction 已运行 | 1 task、16 scenes x 32 seeds | expert/shared direction 含未来 action 关联 | expert-specific 优势或成功预测器 |
| hidden-matched expert activation | 既有 retrospective proxy 已运行 | 5 tasks、80 pools、2,560 candidates | expert geometry 可解码 immediate sensitivity | 超过 hidden/shared；strict twins 太少 |
| offline vector screen | 既有 nested screen 已运行 | 同上 | functional routed vector 与 action geometry 相关 | residual 增量；固定 gate 未通过 |
| raw internal activation correction | 既有修正 screen 已运行 | 5 tasks、8 states x 16 candidates | pre-down raw magnitude 已被正确区分和测试 | 对 baseline 的增量；五任务 0/5 改善 |
| legacy synthetic expert probes | 模型 probe 实际运行，但 provenance 不完整 | contribution=2 random queries；swap n 未写入 artifact | routed branch 非零；experts 非完全同函数 | 真实状态、任务行为或稳定因果语义 |
| HB5/d0 intervention v4 | runtime 工程 smoke 实际运行 | 1 pair、3 arms | 配对/公式正确；扰动可传播到 `x10` | 科学 policy 对比、行为/Q/success |
| raw pruning v5 | 仅预注册，未运行 | 无 | 协议存在 | 不贡献任何经验结果 |
| MoE action decision formation | 既有综合报告；无独立实验 | 不适用 | 汇总上述材料 | 不能作为额外一次复现 |

## 1. 去噪时序

### Action 本身何时定型

在共享的 flow-lead pilot 中，候选 action geometry 相对最终 geometry 的 Spearman
从 `tau0=0.231` 只升到 `tau9=0.428`；`tau9` 的 exact final medoid 仍只有
`20.5%`。44/44 个 query 都到最终轮才首次达到预设 `rho>=0.90`。

remaining-correction readout 的 OOF R2 为：

| 输入 | OOF R2 |
|---|---:|
| round | 0.9232 |
| round + route | 0.9675 |
| round + current latent | 0.9974 |
| round + current latent + route | 0.9969 |

所以 route 确实读出了“还剩多少计算”，但加入 current latent 后的 route 增量为
`-0.00049`。这不是 route 提前承诺 final action 的证据。

### Route 跟随什么

同一批 44 个 query 的完整 32-way route 距离显示：

- `tau0 route-current latent rho = 0.880`；
- `tau0 route-immediate update | current rho = 0.147`；
- `tau0 route-final action rho = 0.168`；
- `tau0 route-final action | current rho = -0.079`；
- 到 `tau9`，最后一项也只有 `0.018`，置乱 `p=0.374`。

最稳妥的机制解释是 route 选择“这一轮怎样修正当前 latent”，而不是保存一个尚未
出现在 latent 中的最终动作编码。相邻 denoise 的 top-1 expert 保持率还从约
`0.744` 降至末段约 `0.591`，不符合 experts 在后段逐步锁定的简单故事。

这里的 action commitment、weighted-noise evolution、unsupervised MoE dynamics 和
cumulative-prefix 共用 **1 task / 1 scene / 4 rollouts**。它们是同一机制 capture 的
互补分析，不是四次独立复现。

## 2. Functional / Expert-Output 证据

### 有结构，但没有稳定 hidden 之外增量

Long t08 的 16 scenes x 32 seeds 中，step-0 routed direction 在有效初始噪声上的
action-basin AUC 增量为 `+0.060`，但 shared direction 的最佳增量为 `+0.092`。
这说明向量方向包含未来 action 信息，却不能归因于 expert specialization。

五任务 hidden-matched proxy 每任务 16 个 K32 pool。balanced proxy 中 task-mean AUC
为 hidden `0.623`、shared `0.662`、conflict `0.680`、DCQ `0.660`。关键 contrast
均跨零：

- conflict - hidden: `+0.057 [-0.053, +0.162]`；
- conflict - shared: `+0.017 [-0.079, +0.111]`；
- DCQ - hidden: `+0.037 [-0.040, +0.118]`；
- DCQ - shared: `-0.003 [-0.090, +0.069]`。

strict global-q10 hidden match 与 global-q90 action divergence 的交集，五任务合计仅
`45` 个 divergent pairs；其中 goal-top 为 `0`。balanced tail proxy 可探索，但不能
替代 strict twin 证据。

### Functional routed vector 的 gate 未通过

80 个 K32 pool 的 offline-vector screen 中，fixed base 的 within-held8 Spearman 为
`0.5124`，加入 residualized routed vector 后为 `0.4982`，宏平均变化
`-0.0142 [-0.0215, -0.0071]`。五任务的变化为 `+0.015, -0.052, +0.024,
-0.016, -0.043`，固定 screen 为 `false`。

这不是说 routed vector 没有信息，而是它没有显示出超越匹配 baseline 的稳定提取
优势。standalone routed 与 hidden/action geometry 相关，不能自动转写成增量机制证据。

### Expert 幅值结果进一步收缩

固定 expert、控制 exact-seed template 后，raw d6/d7/d8 对总 flow 位移的 pairwise
ranking advantage 只有 `0.93/0.95/0.73 pp`。修正后的 pre-down activation screen
更直接：加入 exact40 raw `||m_e||2` 后，equal-task macro relative RMSE gain 为
`-0.547% [-0.735%, -0.388%]`，五任务 `0/5` 改善。

此前 runtime-vector v3 虽然实际运行，但后来发现 target 几乎由 exact d0 velocity
RMS 决定，而 baseline 无法表达这个 norm，因此其推断结果已正式撤回。它只能作为
方法审计历史，不能计入正证据或零结果。

### Offline replacement 只回答即时敏感性

rank 5-8 替换 top-4 后，D/C/DCQ 与 `delta/routed` 的 task-mean Spearman 约为
`0.463/0.484/0.485`。这证明 expert geometry 能预测其自身 immediate block-output
变化，但 predictor 与 target 由同一组 expert vectors 构造，并且换 denominator 后
方向会翻转。没有继续跑剩余 denoise 或 simulator，因此不能称为最终 action fragility。

## 3. 跨 Control Route Dynamics

30-task 审计覆盖 1,500 rollout、18,384 control query 和 16,884 个相邻转移。
task-equal 相邻距离为 `0.02423`，rollout 内随机顺序期望为 `0.02798`，连续性比值
`0.8659`；差值 CI 为 `[0.00325, 0.00427]`，task sign-flip `p<0.0001`。

这足以说明 route 有真实时间连续性。它不说明 route 自身拥有高层语义，因为该分析
没有使用 action、robot state、reward、contact 或 outcome。

更直接的 matched control 来自五任务 state-token Markov 分析：K16 route 相对零阶的
预测增益为 `2.638 bit/step`，proprio 为 `2.709 bit/step`；route 只在 `1/5` task
胜过 proprio。有限状态还是分析者选择的 K-means code，不是发现的自然技能阶段。

单任务 sibling-noise 分解也一致：route-cloud phase ICC 为 `0.861`，但 robot-state
distance 与阶段差的 rho 为 `0.908`；控制 robot state 后 route phase rho 只剩
`0.126`。因此“状态进度骨架 + 局部噪声响应”合理，“独立内部时钟”不合理。

最后，连续 denoise Markov 在 development / objstate replication 的 selector delta 为
`-0.042 / -0.120`，replication AUC `0.413`。虽然真实路径 likelihood 比零阶高约
`1.935 bit/coordinate/transition`，可迁移 outcome 关联并不存在。

## 4. Intervention 证据

### Legacy probes

早期 expert-contribution 与 expert-swap 是实际模型 probe，但输入是随机生成的图像、
wrist、state 和 noise，summary 没有 checkpoint digest、命令、run manifest 或 CI。
contribution summary 对应 2 个随机 query、每层 80 个采样 token-site；swap summary
没有编码 n，脚本默认是 4 个 observation。

它们仍能约束一个极弱命题：routed branch 占总 MLP branch norm 的中位比例约
`0.118-0.366`，all-expert pairwise cosine 均值约 `0.028-0.044`；随机替换全部 top-4
使 MoE output、velocity、最终 action 的相对变化分别约 `31.3%/9.6%/6.0%`。
因此“experts 完全相同”不成立，但这些 synthetic/global interventions 不能说明真实
任务中的稳定 expert role。

### Runtime v4 smoke

HB5/d0 v4 是当前唯一配对完整、实际继续到 `x10` 的 intervention artifact，但只有
1 pair / 3 arms：

- baseline no-op error `0`；
- pair 内 original difference `0`；
- drop formula error `1.53e-8`；
- drop final live-7 RMS delta `0.002062`；
- matched-random final delta `0.002305`；
- 两者 amplification 约 `6.02x/7.25x`。

这通过的是 instrumentation gate：扰动被正确实施并传播。它没有样本量去比较 drop
与 random，更没有环境执行、dense trajectory、event、continuation Q 或 success。

runtime-exact raw-pruning v5 只有 preregistration，workspace 中没有 capture/result。
不能把“已冻结协议”写成“已完成实验”。

## 最终边界

当前数据可以支撑：

> MoE routing 和 expert decomposition 是局部去噪计算的可观测结构；route 对当前
> latent、局部更新和跨 control 状态连续性有诊断价值。

当前数据不能支撑：

> route 在 action 尚未显现时已经编码高层控制承诺，或者 expert IDs 构成具有稳定
> 行为语义和因果作用的技能状态机。

要升级证据等级，至少需要 runtime-exact matched intervention 覆盖预注册 denoise
轮次和 held-out tasks/states，并保存 downstream action/route path。要讨论 behavior
abstraction，还必须另外采集 dense physical trajectory、event tape、post-chunk state
和 paired-CRN continuation Q；现有机制数据不能替代这一步。

## 主要来源

- [Action commitment](../../action-commitment-pilot/REPORT.md)
- [MoE action decision formation](../../moe-action-decision-formation/REPORT.md)
- [Unsupervised MoE dynamics](../../unsupervised-moe-dynamics/flow-lead-real/report.md)
- [Expert activation hidden-matched](../../expert-activation-hidden-matched/REPORT.md)
- [Offline vector screen](../../offline-vector-screen/REPORT.md)
- [Raw internal activation correction](../../raw-internal-activation-correction/observational-hb5-d0/REPORT.md)
- [Control-route speed](../../control-route-speed/report.md)
- [State-token Markov](../../markov-routing-all/REPORT.md)
- [Continuous denoise Markov transfer](../../intraquery-continuous-markov-transfer/REPORT.md)
- [v4 intervention validation](../../../runs/hb5-intervention-v4-smoke-goal-k1/validation.json)
- [v5 preregistration](../../raw-pruning-v5/PREREGISTRATION.md)
