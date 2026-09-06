# 审计意见的逐条修复与重测

本文件记录针对 `idea.md` 设计审计的实现修复，以及修复后在同一语料上的重测结果。
所有数字来自五折 state-blocked 交叉拟合（30 states 训练 / 10 校准 / 10 测试，50 个
init state 各被测试一次），特征为重新抽取的 `artifacts/features-full40-v2`。

## 一、表征层修复

### 1. 跨层 expert 编号比较（审计 #2）

旧 `layer_disagreement` 直接比较不同 MoE layer 上编号相同的 expert 概率。每层可以独立
重排自己的 32 个 expert 而保持模型函数等价，因此该量并非置换不变，与提取器
`"Extract ... permutation-invariant routing phenotypes"` 的契约冲突。

替换为 `layer_profile_disagreement`：只比较每层置换不变的标量 profile（entropy、margin、
top1/top4 mass、soft/top4 consensus、effective rank）的跨层离散度。
`test_features_are_invariant_to_independent_expert_permutations_by_layer` 守护该性质。

**影响范围经实测限定**：新旧特征在第 0–79 列上逐位相同（max abs diff = 0），只有第 80 列
改变（均值 0.0350 → 0.0186）。因此该缺陷**只影响读取第 80 列的 GMM/PST 流水线**；
22 维 open-world phenotype 现场重算该量，从未受影响。

### 2. 对原生动力学重复求导（审计 #3）

`build_query_descriptors` 对全部 81 条轨道一律施加 level / Δ / Δ²。对 `flow_velocity` 与
`flow_acceleration` 而言，这等于在 10 个 flow 点上再取三阶、四阶有限差分，主要表达数值
抖动。`build_clean_query_descriptors` 改为：每条 flow 曲线取 early/middle/late 三段均值，
仅对 level 类指标附加斜率，共 300 维。

此前该函数只被单元测试引用，是死代码。现已接入全部四条离散审计
（`run_experiments`、`run_full40_audit`、`run_dual_axis_audit`、`run_task_holdout_audit`）。

同时修复 `run_experiments.py` 中 `names.index("layer_disagreement")` 的硬编码——它使旧
流水线被钉死在带缺陷的特征存档上，一旦用修正代码重抽特征即会 ValueError。

### 3. 表征修复的效果

| 指标 | 旧（2187 维，带缺陷特征） | 新（300 维，修正特征） |
|---|---:|---:|
| PCA-24 保留方差（fold 0） | 0.462 | **0.831** |
| `history − phase` 五折 | −0.2091 [−0.2169, −0.2040] 5/5 | −0.1984 [−0.2081, −0.1918] 5/5 |
| `ordered − bag` 五折 | −0.00371 [−0.0040, −0.0033] 5/5 | **−0.00505** [−0.0062, −0.0042] 5/5 |
| `history − history1` | −0.00727 (fold 0, p=0.002) | **−0.00034，0/5 折显著** |

主效应稳定，有序效应增强 36%，但**第二个词的增量消失**。旧 README 中
“adding a second word contributes 0.044 bits/token” 一条不再成立。

绝对 NLL 不可跨表征比较（维度从 2187 降至 300 改变了 emission density 的量纲），只有差值可比。

看绝对值可知这两条结论并不矛盾：`history` (39.4098) < `history1` (39.4109) < `bag` (39.4161)。
**无序地池化第二个词是有害的，有序版本只是把损失补回到单词水平。** 顺序确实不可交换，
但机制不是“第二个词带来了有序信息”。

## 二、语法通道补全（审计 #5–#8）

四个此前缺失的通道已实现、并入 `RAW_METRICS`，与既有通道在同一 detector 选择框架下比较。

- `unknown`：GMM 增加平坦背景分量，输出 `P(out-of-vocabulary | query)`，低密度 query 不再
  被强行指派到最近的健康词。
- `order_residual`：`context NLL − 无时钟 unigram NLL`（`PositionGrammar.unigram_continuous_nll`），
  分离“这个词本身罕见”与“这个词在此处罕见”。既有的 `residual = history − phase` 度量的是
  另一件事（history 相对时钟）。
- `recurrence`：`LagRecurrenceModel`，比较 lag 1–4 的词复现模式。exact-run duration 把
  `A B A B A B` 看成六段长度为 1 的 run，完全无法反应。
- `end_hazard`：`1 − P(END within 4 queries)`，由一阶可达性表 `end_hitting_table` 计算
  （逐位置蒙特卡洛对 508k query 不可行）。

### 结果：四个通道都没有超过既有通道

五折、每折 81 个 detector 变体、按 calibration q7/q12 平均 AUC 选择：

| 通道 | 五折最佳排名 | 平均 AUC |
|---|---|---:|
| `unknown` | 7, 7, 11, 16, 17 | 0.6079 |
| `order_residual` | 3, 9, 16, 43, 49 | 0.5943 |
| `recurrence` | 47, 63, 64, 66, 68 | 0.5191 |
| `end_hazard` | 61, 65, 65, 74, 74 | 0.5069 |
| 各折最佳（既有通道） | 1 | **0.6211** |

五折中没有任何一折选中新通道（实际选中：`bag.page0.5`、`history1.point`、`phase.page0`、
`bag.window4`、`phase.page0`）。`recurrence` 与 `end_hazard` 稳定在随机水平。

这是对审计核心假设的直接回答：这些通道的缺失**不是**早期检测失败的原因。

## 三、soft history（审计 #4）

`continuous_nll` 对当前 query 边缘化所有 GMM 分量，但把每个历史 query 硬指派为 argmax 词。
新增 `PositionContextGrammar.beam_continuous_nll` 保留加权的历史 beam。

在 fold 0 的 1,500 条 held-out 成功轨迹上：

| | bits/query |
|---|---:|
| hard argmax history | 39.3096 |
| beam marginalized history | 39.2916 |
| **差值** | **−0.01799（状态分块配对 p=0.0020）** |

14.85% 的 query 与次优词相差不到 1 bit，处于边界抖动区。

**这个测量误差是 `ordered − bag` 效应（0.00505）的 3.6 倍。** 也就是说，硬 argmax 近似
本身的代价大于被测效应，审计对这一点的担心是成立且量级重要的。

## 四、three-channel observer 的三个缺陷

诊断见 `moe_grammar/run_stasis_diagnostics.py`，结果在 `results-three-channel-v3/`。

1. **复合通道含自身反相**：`overregularity` 的第三个分量原为 `percentile(-hmm)`，而
   `hmm_innovation` 为 `percentile(+hmm)`，二者过同一个 phase-conditional CDF，实测相关
   系数 **−1.000**。审计公式要求的是 `−H(M_q | M_<q)`，该量（`belief_entropy`）早已算出
   但从未使用。改用后相关系数变为 **+0.2298**。
2. **哨兵值污染校准**：`causal_recurrence_features` 用 `100.0` 填充每个 episode 开头未定义的
   位置，使 8.3% 的 `periodic` 校准行落在极值上、压缩可用 percentile 量程。改为显式有效性
   掩码，校准器只在有定义处拟合。修复后 `speed` 97.76%、`periodic` 95.53% 的行有定义。
3. **统计单元错误**：matched AUC 原本池化 1,594 个 pair，但独立单元是 159 个有匹配对照的
   stasis event。池化既高估点估计（`current_innovation` 池化 0.592 vs per-event 0.557），
   也低估不确定性。改为 per-event win rate + event-level bootstrap。

### 修复前后

| 通道 | 修复前 matched AUC | 修复后（95% CI over events） |
|---|---:|---|
| `overregularity` | 0.446 | 0.509 [0.457, 0.560] |
| `low_surprise` → `low_belief_entropy` | 0.420 | 0.529 [0.475, 0.583] |
| `recurrence` | 0.479 | 0.517 [0.465, 0.570] |
| `freeze` | 0.497 | 0.508 [0.459, 0.557] |
| `three_channel` | 0.533 | 0.526 [0.479, 0.576] |
| `current_innovation` | 0.592（池化） | 0.557 [0.509, 0.605] |

修复消除了低于随机的病态，但没有救回过度规则性假设：修复后**除 `current_innovation`
和 `var_innovation` 外，所有通道的 CI 都包含 0.5**，而这两者也仅勉强排除。

### 为什么救不回来

在 159 个物理 stasis event 上，routing speed 与同 state、同 query 位置的对照相比，
onset 处配对差为 **−0.019，95% CI [−0.055, +0.018]**，且在 offset −10…+10 全程一致。
不是“来得太晚”，是不存在：机械臂物理卡住时，MoE routing 仍以正常速率继续变化。

同一批行上 `hmm_innovation` 能到 0.546，说明 onset 标签是对齐且有信息量的，
因此该零结果不是标注错位造成的伪影。

## 五、用修正特征重建两条 GPU 链路

归档的 `results-global-prefix/` 与 `results-open-world-global-k12/` 都是用修复前的特征训练/
选择的。两条链路已在修正特征上重建。

### 1. 全局前缀因果 Transformer（`results-global-prefix-v2/`）

五折全部重训（每折一张 GPU 并行，batch 4096）。

| 指标 | 旧（带缺陷特征） | 新（修正特征） |
|---|---|---|
| `global − local2` (q1+) | −0.3384 [−0.3510, −0.3261] | −0.1951 [−0.2051, −0.1850] |
| `global − local2` (q4+) | −0.3756 [−0.3900, −0.3616] | −0.2173 [−0.2287, −0.2056] |
| `global − phase` (q1+) | −0.5568 | −0.4021 |
| remote-order 控制 (q4..q12) | 0.1394 | 0.1152 |
| q12 `global_current` within-task/state AUC | 0.5999 | 0.5840 |

全前缀相对局部二词的优势**缩水约 42%**，但仍高度显著（50 states，p≈5e-6，CI 不含零）。
保持最近两词不变、只反转更早前缀仍使 NLL 恶化 0.1152 bits/query —— 这是"局部上下文之外还
存在有序信息"的最直接证据，**它在修正表征下存活**。检测增量依旧不存在。

### 2. open-world 观测器的 K 选择网格被截断

README 记录 K 由 held-out-success next-chord NLL 单独选择、`K in {4, 8, 12}` 且 K=12 胜出。
在修正数据集上把网格扩展后：

| K | held-out prefix NLL (q4+) | prefix − clock |
|---:|---:|---:|
| 4 | 0.124833 | −0.005673 |
| 6 | 0.113321 | −0.010034 |
| 8 | 0.106049 | −0.012836 |
| 16 | 0.099010 | −0.018183 |
| 20 | 0.096586 | −0.020786 |
| 24 | 0.095533 | −0.022474 |
| **32** | **0.088631** | **−0.026786** |

held-out NLL 在测到的最大 K 处仍单调改善。**K=12 之所以胜出只是因为它是当时网格里最大的
候选**，不是因为存在极小值。prefix 相对 clock 的增益在 K=32 处为 0.0268 bits/phenotype，
是 README 所报 K=12 数值（0.0165）的 1.6 倍。

因此"存在可测的健康前缀结构"这一结论不但成立，而且此前被低估；但它同样没有转化为检测增益。

## 六、词表规模与词稳定性（审计 #1.6、#1.7）

审计要求「GMM 测 K=8,16,32,64，至少 10 次初始化」和「检查不同 fold/seed 下 word matching
的 AMI/stability」。两项此前都没做过：实现固定 K=64、`n_init=2`，没有任何稳定性检查。

在 CPU 上这是数百次 sklearn 拟合，不可行。`moe_grammar/gpu_tokenizer.py` 实现了批量对角
GMM，把一个 K 的全部初始化作为前导维一起做 EM，E-step 变成单次大批量 matmul。
`tests/test_gpu_tokenizer.py` 校验它与 sklearn 的等价性：给定相同参数时 `score_samples`
与 `predict` 一致，从同一起点做一步 EM 得到相同的 means/covariances/weights。

扫描协议：K ∈ {8…1024}，每个 K 用 30 次初始化批量 EM、5 个独立 seed 重复、5 折全跑；
选择只用 held-out 成功 query 的 NLL；AMI 对标签重排不敏感，只惩罚真正的划分变化。

| K | held-out bits/query | 初始化 bound 极差 | 同折跨 seed AMI | 跨折 AMI |
|---:|---:|---:|---:|---:|
| 8 | 45.5737 | 0.1230 | 0.867 | **0.796** |
| 16 | 44.6754 | 0.1441 | 0.721 | 0.682 |
| 32 | 43.7126 | 0.1756 | 0.800 | 0.755 |
| **64（当前实现）** | 42.8154 | 0.1304 | 0.782 | 0.726 |
| 128 | 41.9896 | 0.1089 | 0.758 | 0.687 |
| 256 | 41.3358 | 0.1073 | 0.727 | 0.609 |
| 512 | 40.8348 | 0.0832 | 0.675 | 0.525 |
| **1024** | **40.6517** | 0.0742 | 0.600 | 0.432 |

三个结论：

1. **按 held-out NLL 选 K 没有内部极小值**，一路单调改善到网格边界。这与第五节的
   open-world K 扫描是同一个病：现有协议宣称「K 由 held-out NLL 单独选择」，实际选出的
   永远是网格里最大的候选。
2. **NLL 与词稳定性指向相反方向**。跨折 AMI 从 K=8 的 0.796 崩到 K=1024 的 0.432——
   在最大 K 处，换一组训练 state 就有超过一半的划分改变。当前的 K=64 处在 0.726。
3. **即使固定折，词也不完全可复现**：同折换 seed 的 AMI 为 0.60–0.87，即 13–40% 的划分
   取决于随机种子；K=64 处约 22%。初始化 bound 极差 0.07–0.18 nat，说明 `n_init=2`
   有实质概率停在明显更差的解上。

因此「query word」这个对象本身的可复现性，比语法层测到的效应（`ordered − bag` = 0.005
bits/query）要脆弱得多。这为第三节的 soft-history 结果提供了机制解释：硬指派之所以代价
大，是因为词边界本身就不稳定。

## 七、第四层：语义（审计 #9，五折）

`idea.md` 第 28–29 行把语法与语义分开：语法是健康句子的条件转移概率，语义是句子未来进入
loop/static/恢复/完成的概率。此前所有失败监测都是拿语法 surprisal 直接回归最终 outcome，
既不是语法该做的事，也不是语义的定义。第四层此前零代码。

`moe_grammar/semantics.py` 按原文实现 Semantic Suffix Trie：给 context 挂 $n_{c,h}$ 计数、
取 Dirichlet 后验、支持不足时沿 suffix 回退。判定标准是 context-conditioned $\mu(c,h)$
能否胜过把 context 去掉的同一估计器。五折 state-blocked，共 118,193 个 held-out 偏离 query，
log-loss（越低越好）：

| 任务 | context − root | 95% CI（折间 bootstrap） | folds |
|---|---:|---|---:|
| 全字母表 | **−0.1877** | [−0.1997, −0.1773] | 5/5 |
| 仅 return vs persist | **−0.0572** | [−0.0689, −0.0488] | 5/5 |

第二行是混淆控制：剔除 `end_*` 后重归一化，去掉「这个 context 预示 episode 快结束了」的
信号。约 70% 的全字母表增益来自 episode 长度，但**剩下的纯 return/persist 语义五折全部
显著**。作为量级参照，语法层最强的有序证据 `ordered − bag` 是 0.005 bits/query，而
return/persist 语义是 0.082 bits/query，约为其 **16 倍**。

干预语义只做 arm 层（fork 语料仅 10 个独立 trunk，按 trunk 分块 bootstrap）：
`control` 0.020、`triggered` 0.320（减 control **+0.3125**，CI [+0.025, +0.583]）、
`delayed_+4` 0.220（+0.2083，CI 触零）、`delayed_+8` 0.118（+0.0938，CI 触零）。
只有 `triggered` 排除零且区间极宽，与原文第 1092 行的预判一致。

**边界**：`return`/`persist` 由语法自身 surprisal 的百分位定义，不是物理标签。本层证明的是
「context 能预测这条 routing 轨迹会不会回到健康流形」，**不是**「能预测机器人会不会脱困」。

## 八、K 截断对既有结论的影响面

第五、六节发现两处 K 选择都没有内部极小值。逐一检查它们是否藏住了结论。

### 1. open-world phase states：没有藏住检测器

把 three-channel 观测器在 K ∈ {12, 16, 20, 24, 32} 上各跑一遍：

| channel | K=12 | K=16 | K=20 | K=24 | K=32 |
|---|---:|---:|---:|---:|---:|
| `current_innovation` | 0.557* | 0.566* | 0.557* | 0.559* | 0.557* |
| `var_innovation` | 0.558* | 0.572* | 0.554* | 0.563* | 0.560* |
| `low_belief_entropy` | 0.529 | 0.522 | 0.495 | 0.527 | **0.598\*** |
| `overregularity` | 0.509 | 0.501 | 0.495 | 0.516 | **0.561\*** |

（`*` = event-level 95% CI 排除 0.5）

`low_belief_entropy` 与 `overregularity` 只在 K=32 单点跳到显著，前四个 K 全在 0.49–0.53
之间，**不是单调趋势**。8 通道 × 5 个 K = 40 次检验，出现一两个显著正是预期；且
`overregularity` 本就包含 `low_belief_entropy`，两者的跳变是同一件事。稳定显著的只有
`current_innovation` 与 `var_innovation`，在所有 K 上都是 0.554–0.572。

结论：K 截断**低估了「健康前缀结构存在」的量级**（0.0165 → 0.0268 bits/phenotype），
但**没有藏住检测器**。

### 2. GMM 词表：藏住了一个反向结论

按 held-out NLL，K=256 明显优于当前的 K=64。在 K=256 上重跑五折：

| 效应 | K=64 | K=256 |
|---|---:|---:|
| `history1 − phase`（前一词） | −0.19800，5/5 | **−0.24417，5/5** |
| `history − phase` | −0.19834，5/5 | −0.22599，5/5 |
| `ordered − bag` | −0.00505，5/5 | **−0.00171，5/5** |
| `history − history1`（第二个词的增量） | −0.00034，0/5 | **+0.01818，0/5** |
| 有效词数 | 55.6 | 217.4 |

**单前词依赖在更好的词表下变强（−0.198 → −0.244），而有序二词效应缩水 66%，且
`history − history1` 由负转正**——在 K=256 下，有序二词上下文比单词上下文还差。

也就是说：跨词表稳健的是**「前一个词有信息」**；而 README 的 headline「有序二词语法」
是**词表规模的产物**，换一个 NLL 更优的词表就基本消失。

检测侧无变化：q3/q7/q12 的跨折 AUC 差为 −0.012 / −0.003 / +0.013，CI 全部含零，
折数 2/5、2/5、3/5；failure recall 在三个 horizon 上反而下降
（0.096→0.075、0.156→0.111、0.330→0.272）。fold 0 上看似的大幅提升（q7 0.602→0.662）
是折间噪声。

## 九、遗留的功率与统计约束

- **该评测的分辨率不足以裁决它要问的问题。** 159 个 event 的 event-level CI 宽约 ±0.05，
  而通道之间的差异在 0.01–0.05 量级。在做更多建模之前，瓶颈是更多带标注的 onset。
- **GPU 功率取决于负载类型，不是设备本身的限制。** 抽特征是 zarr 读盘密集、离散审计是
  sklearn CPU 密集、HMM forward 与前缀 Transformer 只是数秒级突发，这些阶段功率停在
  120–140 W。而第六节的批量 GMM 扫描（K≤1024、30 次初始化同时 EM）在 6 张卡上持续跑到
  **495–502 W / 500 W、利用率 85–87%**——同一批硬件，密集矩阵负载就能压满。
  仍受约束的是与其他作业共享的 **cgroup 8 GiB**：
  每个 CUDA 进程仅上下文与数据暂存就要 1–2 GB，因此最多同时跑 3–5 个 GPU 作业。
  实测代价是累计 17 次 OOM kill——7 卡抽特征在 batch=16384 时被杀 4 个分片（降到 4096
  才稳定）、full-40 五折首轮被杀 3 折、K 扫描被杀 2 个、前缀训练被杀 2 折。
  `run_remaining_folds.sh` 与 `run_global_prefix_v2.sh` 的等待-重试逻辑是必需的，
  不是保险。

## 十、尚未实现

- 审计 #1 的“字母→单词”可组合语言由并行工作实现，见 README 的
  `Strict Ten-Flow Query-Word Audit` 与 `Interpretable Query-Level Discrete Audit`；
  其结论与本文收敛（recurrence / low-surprisal / return 通道同样救不回早期检测）。
- 语义层的 `return`/`persist` 目标是 routing 自定义的，缺一个物理定义的恢复标签。
- 干预语义要条件化到 routing context，需要显著多于 10 个独立 trunk。
- 审计 #1 的「字母→单词」可组合语言由并行工作实现（见 README 的
  `Strict Ten-Flow Query-Word Audit` 与 `Interpretable Query-Level Discrete Audit`），
  其结论与本文收敛。
