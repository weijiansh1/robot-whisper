# HiMoE-VLA 历史 MoE 实验总审计

日期：2026-08-29

## 最终结论

把所有旧结果放到同一套统计标准下，结论可以压缩成四句：

1. **MoE 确实在参与计算。** state routing 可读出当前运动状态，action routing 随采样动作变化，专家交换、分支消融和路由钉扎都会改变内部输出或轨迹。它不是无效装饰。
2. **但“参与计算”不等于“知道最终成败”。** 现有实验没有得到一个跨初态、跨 worker、在完整物理/动作对照之外仍稳定的早期失败评分器，也没有得到可靠的 Best-of-N 候选选择器。
3. **旧的全轨迹失败聚类必须撤回为 MoE 失败机制证据。** 它主要重建了任务、episode 长度和跑满 step cap 的超时终态；一个 `length == cap` 哨兵比所谓三方法共识核心更强。
4. **仍有一个值得继续的窄方向。** 路由在失败已经发展起来后，能读出停滞、持续回返、重抓/掉落等不同轨迹状态；最新滚动星形树里 q24-q31 对最终停滞/循环有小幅同方向增量，但只有 4 个 worker、窗口又是扫描后选出的，当前只能叫候选信号。

所以，最准确的总表述是：

> HiMoE 的 routing 是当前状态、采样动作和去噪进程的低成本内部读出；它包含失败轨迹发展后的机制痕迹，但尚未证明能在物理现象出现前独立预警最终失败，也没有证据支持删除某个 expert。

## 审计覆盖

本次不是只看最近几份报告，而是递归盘点了 `analysis/` 下的全部实验目录，并把上级目录中的历史汇总一起纳入语义审计：

| 项目 | 数量 |
| --- | ---: |
| `analysis/` 一级实验目录 | 149 |
| 去掉显式复现/seed/PCA 副本后的规范目录名 | 119 |
| 复现或敏感性目录 | 31 |
| 含 report 的目录 | 115 |
| 含 summary 的目录 | 113 |
| 上级历史报告 | 15 |
| 未分类目录 | 0 |

完整逐目录索引在 [directory_inventory.csv](directory_inventory.csv)，统计摘要在 [inventory_summary.json](inventory_summary.json)。149 个目录不等于 149 次独立实验：大量目录是同一批数据上的不同特征、聚类、seed 复现或审计。

主要数据群如下：

| 数据群 | 规模 | 独立性边界 | 主要用途 |
| --- | --- | --- | --- |
| legacy 同快照 fork | 20 个快照，每个 K=32 | continuation 每候选仅一次 | medoid、候选价值、动作响应 |
| 五任务 `right-16x32` | 5 任务、80 个初态池、2560 rollout | 同一 flow seed 贯穿整条 rollout | 绝大多数失败/路由/初态分析 |
| flow-lead | 1 任务、1 scene、4 条相关 rollout、44 query、K=16 | 样本极小且相关 | 去噪内 routing 与 latent |
| 30-task route corpus | 30 任务、1500 rollout、18384 query | 观察性 | 跨 control 连续性 |
| 干预小样本 | 多个 smoke/pilot | 多数功效不足 | 专家交换、分支关闭、路由钉扎 |
| 最新 rolling-star | 4 worker、22 snapshot、K=16、352 分支 | 同 snapshot 的 16 条是 sibling；独立 worker 仅 4 | 前缀预测、失败类型、视频对照 |

## 证据分级

- **A：可保留事实。** 数据链路完整，结果复现，命题主要是机制或工程事实。
- **B：可保留关联。** 有合理对照和留出，但仍是观察性状态读出。
- **C：候选。** 方向有趣，但样本、迁移、多重比较或独立验证不足。
- **D：关闭或撤回。** 关键对照失败、存在标签泄漏，或后续更完整实验推翻。
- **E：尚不可回答。** 数据里没有双结局风险集或没有真正执行所需干预。

## 全部实验族裁决

| 实验族/命题 | 等级 | 最终裁决 |
| --- | --- | --- |
| capture、fork、route/action ID 保存 | A | 原始 route、实际执行 Top-K、动作和多数 simulator state 的保存链路可信；但历史 fp32 gate 未保存，不能从 fp16 概率反推历史 fp32 Top-K。 |
| AS/state gate 的去噪变化 | A | LIBERO 缓存里 AS mask 恒定，因此 AS routing 恒定是架构/输入掩码结果，不是成功信号。 |
| action hard expert ID | D | action gate 大量近似平局，当前缓存 `p4=p5` 约 28.6%；hard ID 更替不能直接解释成策略切换或技能切换。实际记录的 executed ID 本身仍可信。 |
| action soft routing 随候选变化 | B | 同初态 K=32 内，route 可解码中心化 action，留出初态 `R2=0.605`；它是 sampled action 的有损表征。 |
| routing 对短时物理响应的增量 | D | route 单独可预测 0.5 s EEF 响应，但完整 action 已达 `R2=0.9985`；action+route 相对 action 为 `-0.000369`。没有额外价值。 |
| state routing 编码当前状态 | B | 35/35 个概念在探索性线性读出中可读，只有 8/35 对线性 geometry 有正增量，最强集中在速度/进度速度；更像动态 state shadow。 |
| 跨 control routing 连续性 | B | 相邻 query 路由距离 `0.02423`，随机配对 `0.02798`，连续性比 `0.8659`；但匹配 proprio 同样强或更强，不能叫独立技能 FSM。 |
| 去噪内 routing 领先最终 action | D | flow-lead 中 d0 route 与当前 noisy latent 相关 `rho=0.880`；控制当前 latent 后，对最终 action 的增量接近 0。routing 跟随当前 latent，而非提前宣布最终动作。 |
| expert 输出是否功能不同 | A/B | expert swap、routed/shared/block 操作会改变 MoE 输出和动作；说明专家有功能差异。尚未证明某个专家提升期望成功率。 |
| q0 early commitment | D | 16x8 commitment grid 的 16/16 行都同时出现成功和失败，`MI` 置换 `p=0.921`；第一步 route 不是命运。 |
| 初始状态风险 | A（非 MoE） | 已知初态时，任意失败 Brier `0.1025 -> 0.0632`，下降 38.3%；这是 geometry/rollout 难度先验，不是 MoE 识别失败。 |
| q0/q0-q2 routing 超越初态先验 | D | 任意失败 q0 增量 `+0.0003`，CI `[-0.0007,0.0014]`；6 个目标无一正下界，PCA32 也未救回。 |
| clean-444 q0+8 routing 失败预测 | D | 补齐全部缓存 geometry+action 后，state route `+0.008 [-0.017,0.025]`，action route `-0.014 [-0.037,0.013]`；旧 `+0.237` 是不完整物理基线造成。 |
| clean-444 router-input hidden | C | hidden 增量 `+0.053 [+0.014,+0.085]`，但 hidden 是 gate 输入，不是专家贡献；区间未证明真实增量达到预设 `+0.05` 实用阈值。 |
| t34 remaining-time early warning | D | 精确风险重叠仅 `1 success / 216 failures`。phase/remaining-time 哨兵几乎完美只是协议同义反复；该数字不再作为 MoE 结果引用。 |
| 全轨迹无监督失败聚类 | D/撤回 | 307 个失败全部跑满 cap，成功只有 2 条在 cap；1-bit cap 哨兵 precision 0.9935/recall 1.0，固定绝对 query 后“核心”消失。 |
| residual failure atlas | B（后验诊断） | 在失败内部、task×init 分层后，不同物理代理对应不同 routing 家族；说明信息不止停滞特征。但特征来自相对 phase 0.5-0.9，只能叫轨迹状态/失败类型读出。 |
| literal periodic loop | D | 只有 3/307 失败过 route-cycle，1 条同时过物理/动作联合规则，独立物理振荡 0/307；普遍现象是 persistence，不是周期 loop。 |
| 跨任务早期失败签名 | D | query0/pre-anchor 聚合特征没有跨任务通过联合校正；强差异多在接触之后出现。 |
| Best-of-N route medoid/consensus | D | legacy K32、五任务 K32、task-held-out K8 均无稳健收益；多数概率 medoid 只是固定 seed 模板。 |
| action majority selector | C/D | 只有弱、任务依赖或功效不足的点估计，没有可复用成功率收益。 |
| route dispersion 分配 adaptive K | D | task-macro 相关接近 0，task-held-out MAE 比均值基线差；不能据此决定哪里多 rollout。 |
| 去噪动态早停 | D（当前方法） | MoE 激活 86.8% 方差由 round 解释，固定 round/动作外推基线更强；没有可迁移的 MoE 动态停止器。 |
| controlled OOD sentinel | B/C | 人为 wrist-mask flip 的同 slice 对照中 route 能明显检出，而 proprio 静默；但只测了受控 corruption，没测真实光照、遮挡、干扰物或语言改写。 |
| setback 后 recovery | E | 219 个候选风险集组合中 0 个同时含足够恢复/失败双结局；自然 rollout 不能训练可解释 recovery predictor。 |
| route/expert 因果干预 | C | 路由钉扎和 swap 可改变动作/轨迹，但成功率总体效应尚不稳定；现有样本主要证明“会影响实现”，不是“提高 value”。 |
| expert pruning | E/NO-GO | K=1/prune smoke 和小样本 ablation 不足以决定删哪个 expert；最新单 expert discovery 也未在 prospective validation 复现。 |

机器可读版本见 [VERDICTS.csv](VERDICTS.csv)。

## 三项必须纠正的旧结论

### 1. “三种聚类独立发现失败核心”应撤回

旧管线 outcome-blind 这一点本身是真的，但它从每条 episode 的后半段按相对 phase 取样。LIBERO 又是成功即终止、失败跑满 cap，于是采样网格、终态和长度共同泄露结果。

关键反证来自 [泄漏审计](../AUDIT-clustering-leakage-20260828/report.md)：

- 307/307 失败都达到任务 step cap；2253 个成功中只有 2 个达到 cap。
- `length >= cap` 只用 1 bit，precision 0.9935、recall 1.0，强于 routing 共识核心的 0.9816/0.6938。
- 纯物理的末段 EEF 平均步长按同样 217 条预算，选出的 213 个失败与共识核心计数完全相同。
- 给定 `task x length` 后，共识核心预期失败 213.49，实测 213，`p=1.0`，routing-specific 成分没有出现。
- 把所有 episode 钉在共同绝对 query 索引后，最佳块 precision 只有 0.387-0.529，旧核心 Jaccard 只有 0.18-0.29。

因此可以说“MoE/动作/物理轨迹在失败尾段一起冻结”，不能说“聚类发现了隐藏失败机制”。

### 2. “初始 routing 识别 rollout 难度”应改写

[初态先验审计](../initial-state-early-moe-signal-20260828/report.md)证明初始状态确实显著影响失败率，但 routing 没有提供稳定额外信息：

- 任意失败：task prior Brier 0.1025，initial-state prior 0.0632，下降 38.3%。
- 停滞、重抓/掉落、目标回退分别下降约 38.5%、25.3%、30.9%。
- 同一初态留出 flow seed 后，q0 route 相对 init prior 只改善 0.0003，区间跨 0。
- 留出完整初态、扩展到 q0-q2、增大到 PCA32，均没有得到可靠正增量。

说人话：有些摆放本来就难，这能很早知道；但目前没有证据表明 MoE 比“认出这是哪个初态”多知道一步。

### 3. “后半段签名是提前预警”应降级为并发诊断

[residual atlas](../residual-failure-moe-atlas-20260828/report.md)有一个真实且有意思的结果：拿掉 persistence/recurrence 这类 trap 特征后，其他 routing 家族仍覆盖重抓/掉落、active return、残余错误等；在其 dual-control 口径下，trap-only 覆盖 108/212，完整特征覆盖 168/212，success 触发率约 13%。这说明 routing 信息不只一维，也不只等于停滞。

但它使用每条轨迹相对 phase 0.5-0.9，观察点随最终长度变化，而且很多现象已经发生。因此它支持：

> 不同失败轨迹在 MoE 中留下不同的分布式状态痕迹。

它不支持：

> MoE 在错误发生之前已经知道最后会失败。

`nontrap_only` 在固定四特征预算下甚至覆盖 174/212，高于 full 的 168/212，这是特征选择器互相挤占，不是“删掉 trap experts 会更好”。这里剔除的是离线特征族，不是模型里的专家。

## 目前最可信的正结果

### Action routing 是动作条件表征

[candidate-level soft routing](../candidate-level-soft-routing/report.md)在同一 cached state 的 32 个 seeds 内比较，排除了初态差异这一大混杂：

- route/action 距离池内 Spearman 0.175；
- route 解码中心化 action 的 held-init `R2=0.605`；
- route 单独解码短时 EEF 响应 `R2=0.626`；
- 完整 action 为 `R2=0.9985`，再加 route 反而为 `R2=0.9981`。

所以 route 确实“看起来像这次采样出的动作”，但没有比动作本身多提供已观测短时物理信息。

### State routing 是动态状态影子

[state concept matrix](../state-routing-concept-matrix/report.md)和跨任务迁移结果共同支持：state route 能读出 EEF speed、target speed、进度速度、姿态范数等动态量，静态 geometry 的增量很少；强差异又主要发生在物理交互之后。因此更稳妥的解释是运动阶段/当前状态编码，而不是抽象成功/失败编码。

### 专家确实有功能影响

expert swap、分支关闭和 route pin 会改变 MoE 输出、动作或具体成功的是哪几条 rollout。现有 50 对分支 ablation 是 baseline 14/50、shared-off 15/50、routed-off 12/50、整块 off 7/50。这个规模只提示整块关闭方向较差，不能证明 routed 分支无用，也不能定位可删专家。

## 目前最可信的负结果

1. **没有稳定的 route-based candidate selector。** medoid、consensus、noise+route head、action majority 都没有跨数据集稳定胜过随机/动作基线。
2. **没有 route-based adaptive K。** dispersion 与实际 headroom 几乎无关，跨任务预测更差。
3. **没有通用 MoE 动态早停。** routed activation 大部分是去噪时钟；固定 round 加数值外推更简单且更强。旧 Long “28.2% 不安全”还混入了夹爪连续幅值，修正后 fixed round 8 仅 3.9% 超参考带，round 9 为 0/181。
4. **没有普遍的周期 loop。** 失败更持久、更少变化，但不按固定周期回到同一路由/物理/动作状态。
5. **没有可训练的自然 recovery 数据。** setback 后几乎没有同风险状态下的恢复/失败双结局。
6. **没有 expert pruning 许可。** “某几个 expert 与失败相关”无法推出“删除会提高成功率”；这需要同状态、同噪声、随机化 expert intervention。

## 最新 rolling-star 实验的独立审计

数据位于 [rolling-star run](../../runs/rolling-star-a100-long-t08-k16-20260828/RESULTS_ZH.md)：

- 22 个 trunk snapshot，每个 K=16，共 352 条终止分支；117 成功、235 失败；15 个 snapshot 内同时有成功和失败。
- 4 个 worker 的成功率为 26.0%、0%、6.25%、90.6%，说明 worker/初态难度是巨大混杂。
- 保存 16180 个 query 的完整 HB routing，未保存 hidden；352/352 终止谓词复算一致。
- 原始文件、派生文件和 3 个成功/失败对照 MP4 的校验和均通过。
- 所有 352 条分支都至少活到 q31，因此 q0-q7、q8-q15、q16-q23、q24-q31 是共同绝对前缀，不存在按最终长度取尾部的问题。

### q0 结论

早期单 expert 和单坐标 discovery 都没在预先冻结的 validation 快照复现；全 2560 个 expert 概率格的 max-T 也未通过。按 snapshot 分折看起来有效的模型，leave-one-worker-out 后变差，说明它主要认出了 worker/初态几何。

因此 q0 不能用于删 expert，也不能称为跨初态失败预警。

### q24-q31 候选

[tail analysis](../../runs/rolling-star-a100-long-t08-k16-20260828/analysis_tail_moe/REPORT_ZH.md)在固定窗口上得到：

| 固定窗口 | MoE 相对同构物理/动作对照的 trap Brier 增量 |
| --- | ---: |
| q0-q7 | -0.0002 |
| q8-q15 | -0.0011 |
| q16-q23 | -0.0029 |
| q24-q31 | +0.0087 |

q24-q31 的 4/4 worker 同方向，精确单侧 sign-flip `p=0.0625`。这是四个独立组能达到的最小 p 值，不能再把基于 4 cluster 的 bootstrap CI 当成额外独立证据。并且这里扫描了 4 个窗口和多个目标，报告内只校正了单窗口特征搜索，没有校正“最后选中 q24”这一层选择。

实际收益也很小：K=16 避开 trap 从 13/17 提高到 14/17，只多修正一个 mixed snapshot；合并模型自身仍比常数发生率基线差 0.0299 Brier。它最适合被冻结成下一批 worker 的唯一候选，而不是现在上线。

## 为什么现在不能剔除“尾部 trap experts”

这里有三个不同操作，不能混在一起：

1. **离线剔除 trap 特征族：已做。** residual atlas 的 `nontrap_only` 删除 persistence/recurrence 特征后仍能读出其他失败类型，说明信息确实没有只集中在 trap 特征。
2. **选择分支时忽略 trap score：可以作为下一轮 A/B 臂。** 但最新数据只多选对 1/17 个 snapshot，尚无稳定收益。
3. **从模型里删除 expert：没有依据。** trap 信号由 entropy、top-k mass、跨 query persistence、denoise motion 等分布式量构成，不对应稳定的某个 expert；单 expert 候选又未复现。删除 expert 还会同时破坏正常状态和其他失败类型所需计算。

真正的 pruning 实验必须先冻结 expert/group，然后在相同 simulator snapshot、相同 flow/environment seed 下随机执行 baseline 与 prune/pin/swap，比较成功率和失败类型。相关性筛 expert 后直接删，会把选择偏差当成因果。

## 统计和解释护栏

后续所有 MoE 失败实验应继续执行这些规则：

1. 特征窗口使用共同绝对 query，或预测未来固定 horizon 内的首次事件；禁止按最终 episode 长度取相对尾部后称为早期预警。
2. 同时放入初始 geometry、当前完整 simulator state、完整 action、noise/seed、query index 和 task/worker 对照。
3. train/test 按完整 worker、初态、task 或 snapshot 分组，不能把 sibling branch 或同 rollout query 当独立样本。
4. 先冻结一个目标、一个窗口、一个模型，再去新 worker 上验证；不把扫描后的最好窗口的普通 CI 当确认性证据。
5. 区分三类命题：`可读出`、`对照外有增量`、`干预能提高成功率`。前一个成立不能自动推出后两个。
6. 物理失败标签仍是基于 kinematics 的代理。没有 RGB/contact/force 时，应写“掉落代理”“回退代理”“停滞代理”，不能当真实接触因果标签。
7. 不再报告 remaining-time/phase 哨兵为 MoE 性能；它只用于发现协议泄漏。

## 可以公开说什么

- HiMoE routing 对当前状态、动作候选和去噪进程有可重复的结构。
- action routing 是 candidate-conditioned action representation；state routing 更像动态状态/运动阶段影子。
- 失败轨迹后半程存在不同的分布式 routing 痕迹，不只停滞/持续性一类。
- 当前没有跨初态验证的早期最终失败预测器、Best-of-N selector、adaptive-K 规则或 expert pruning 结论。
- 最新 q24-q31 trap 增量是待独立复现的候选，不是已验证报警器。

## 不应再说什么

- “三种无监督聚类独立发现了 MoE 失败核心。”
- “0.998 说明 MoE 几乎完美识别失败。”
- “初始状态 Brier 提升说明 MoE 会判断任务难度。”
- “某个 expert 在失败里更活跃，所以可以删掉它。”
- “路由 persistence 就是机器人在 loop。”
- “相对尾段能区分失败，所以模型提前知道会失败。”

## 可复现性检查

本轮完成的检查：

- 全部分析 JSON 均可解析。
- 29 个提供 `--self-test` 的分析脚本全部通过。
- `pytest -q test_*.py`：491 passed。
- 全量 `pytest -q` 在收集 `branch_replay_test.py` 时缺少外部模块 `himoe_libero_bridge`，属于本环境依赖缺失，测试尚未进入执行；不是统计实验失败。
- 初态主分析和 PCA32 敏感性校验和全部通过。
- rolling-star 原始清单、派生清单、tail-analysis 清单及视频清单全部通过 SHA-256。
- 关键分析脚本通过语法编译；本审计清单脚本通过静态检查。

这里的“通过”表示工件未损坏、可由现有代码重读或复算；不表示 119 个规范目录都从模型 forward 重新采集了一遍。

## 工件

- [总目录 CSV](directory_inventory.csv)
- [目录统计 JSON](inventory_summary.json)
- [裁决表 CSV](VERDICTS.csv)
- [审计工件 SHA-256](SHA256SUMS.txt)
- 清单生成脚本：`../../audit_moe_experiment_inventory.py`
