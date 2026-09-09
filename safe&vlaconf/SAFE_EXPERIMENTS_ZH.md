# SAFE v2 实验核查

核查日期：2026-09-07。对象：*SAFE: Multitask Failure Detection for Vision-Language-Action Models*，arXiv:2506.09937v2。

依据是本地[论文 LaTeX 源码包](arXiv-2506.09937v2.tar.gz)中的正文、附录、表格和图注，并核对[官方论文网页](https://arxiv.org/html/2506.09937v2)。下文的章节、图表编号均指 v2。本文是文献实验核查，没有重新运行作者实验，也没有声称完成官方代码复现。

SAFE 的策略权重保持冻结，但 SAFE-MLP/LSTM 要用成功、失败轨迹训练检测网络。因此原方法并非我们要求的 train-free；其特征分析、距离基线和评价设计可以借鉴。我们的实际实现另见[方法总览](moe_trainfree/METHOD_SUMMARY_ZH.md)。

## 1. 实际做过的实验总览

| 实验 | 实际内容与目的 | 原文位置 |
| --- | --- | --- |
| 仿真失败检测与未见任务泛化 | 在 LIBERO-10 和 SimplerEnv 比较检测器，分别报告 seen/unseen ROC-AUC | Sec. 5.1、6.1；Table 1 |
| 真机失败检测与未见任务泛化 | Franka + pi0-FAST-DROID、WidowX + OpenVLA 两套真机实验 | Sec. 5.1、6.1；Figure 6；App. B.5-B.6 |
| 二维特征结构 | 同一 t-SNE 投影按成败/时间和任务着色，叠加轨迹与观测；补充其他模型和真机 | Sec. 4.1；Figure 1；App. C.1、Figure 8 |
| 检测头对照 | 累积逐步输出的 MLP 与读取历史的 LSTM | Sec. 4.2；Table 1、Figure 6；App. B.2 |
| 基线比较 | token 不确定性、特征距离、学习式 OOD、动作采样一致性、相邻 chunk 一致性 | Sec. 5.2-5.3；Table 1、Figure 6 |
| 报警阈值与检出速度权衡 | 扫描 functional CP 的 alpha，比较误报、检出、balanced accuracy 和报警时间 | Sec. 5.4、6.2；Figure 4；App. C.2、Figure 9 |
| 单条轨迹定性检查 | 画分数、时变阈值和视频观测，检查漏抓、抖动、冻结、物体滑落等 | Sec. 6.3；Figure 5、Figure 6 |
| 报警与人工失败时刻对齐 | 累计检出时间曲线及每条轨迹的检测时刻/人工时刻散点 | App. C.3；Figure 10 |
| 特征聚合与超参数选择 | token、action horizon、flow step 维度的不同聚合；归一化前后特征；距离参数等 | App. B.1、B.8；Tables 9-11 |
| 参考训练任务数量消融 | OpenVLA + LIBERO 中使用 1、3、5、7 个训练任务，固定未见测试任务 | App. D.1；Table 6 |
| 特征来源消融 | 真机 Franka 上比较 DINOv2、CLIP、两者拼接、VLA 末层特征 | App. D.2；Table 7 |
| 重复实验与开销 | 不同随机任务划分的均值/标准差，以及检测器参数量和推理时延 | App. C.4、Table 8；Sec. 6.4 |

## 2. 数据、任务留出和模型

| 场景 | 策略 | 数据量 | 检测器任务划分 | train / eval-seen / eval-unseen |
| --- | --- | --- | --- | --- |
| LIBERO-10 | OpenVLA、pi0-FAST、pi0，分别评估 | 每个策略 10 任务 x 50 条 = 500 条 | 7 seen / 3 unseen | 每个策略 210 / 140 / 150 |
| SimplerEnv Google Robot | 复现版 pi0，文中记 pi0* | 4 任务 x 100 条 = 400 条 | 3 seen / 1 unseen，见下方原文冲突说明 | 198 / 102 / 100 |
| SimplerEnv WidowX | 独立 checkpoint 的 pi0* | 4 任务 x 100 条 = 400 条 | 3 seen / 1 unseen，见下方原文冲突说明 | 198 / 102 / 100 |
| 真机 Franka | pi0-FAST-DROID | 13 任务，每任务 30 成功 + 30 失败，共 780 条 | 10 seen / 3 unseen | 450 / 150 / 180 |
| 真机 WidowX | OpenVLA，Open-X Magic Soup++ checkpoint | 8 任务，共 532 条；244 成功、288 失败 | 6 seen / 2 unseen | Table 5 列示 250 / 133 / 149 |

这里的 unseen 指失败检测器未见过该任务，不等于 VLA 策略在预训练或微调时从未见过该任务。LIBERO 使用各作者发布的 LIBERO 微调 checkpoint，SAFE 不再微调策略。

训练数据来自 seen 任务；eval-seen 用于检测性能验证、超参数选择，其中成功轨迹还用于 CP 校准；unseen 任务用于跨任务测试。不能把 eval-seen 同时承担的这些用途改写成三个完全独立的数据集合。

SimplerEnv 的两个 embodiment 分别训练、分别评价检测器，再平均指标。它不是把 Google Robot 的检测器直接迁移到 WidowX 的实验。成功率达 98% 的 pickup coke 任务被排除。

仿真主表平均 3 个随机种子及对应的不同 seen/unseen 任务划分；真机 Figure 6 平均 5 个种子及不同任务划分。WidowX 每任务条数并不完全相等，其具体 rollout 分割数量可能随任务划分变化。

## 3. 那张二维图到底检查了什么

Figure 1 使用 pi0-FAST 在 LIBERO-10 上产生的内部特征，经 t-SNE 降到二维。一个特征对应一次策略推理时提取并聚合的 embedding；策略预测 chunk，执行其中前 H' 个动作后才进行下次推理，因此不能默认每个环境控制步都有一个新的 embedding。

作者对同一批点、同一投影做了三种展示：

1. 成功轨迹的点始终为蓝色；失败轨迹的点随时间从蓝变红。
2. 保持坐标不变，按 task ID 着色，检查失败区域是否跨任务共享。
3. 标出一条成功和一条失败轨迹的运动路径，并放上对应机器人观测。

**红色首先表示失败轨迹的后期，不是每一帧经过人工标注的物理失败状态。** t-SNE 不使用成败标签降维，标签用于着色；图本身没有给出可部署的二维失败分类边界。

附录 Figure 8 检查这个现象能否推广：

| 模型与场景 | 观察 |
| --- | --- |
| pi0-FAST / pi0，LIBERO | 失败后期特征有聚到相同区域的现象 |
| OpenVLA，LIBERO | 失败特征表现为散布的小簇，与冻结、局部抖动相关，没有单一统一失败区 |
| pi0-FAST，真机 Franka | 成功与失败在二维图中不容易分开；作者推测任务和失败模式更复杂 |

因此不能从该论文推出“失败必在外围”“所有模型都有一个失败区”或“二维点到边界的距离就是失败概率”。我们的 MoE 二维图需要同样检查任务、时间和失败类型混淆，并在原特征空间评估检测效果。

## 4. 比较了哪些方法

### 4.1 与我们最相关的距离基线

Sec. 5.3 明确写出：从训练轨迹保存成功特征库 E_success 和失败特征库 E_failure，使用

```text
s_t = d(e_t, E_success) - d(e_t, E_failure)
```

即越远离成功特征、越接近失败特征，失败分数越高。Mahalanobis、欧氏 kNN 和余弦 kNN 都在距离对照中；kNN 对 k 个近邻距离取平均，超参数候选为 k = 1、5、10。

失败库按整条 rollout 的最终标签构建，包含失败轨迹早期可能仍正常的特征，并非只收集物理失败之后的点。该区别在复用时也要保留并检查。

论文还比较 PCA-KMeans，搜索 PCA 维数 32、64、128 和簇数 16、32、64。正文将它列在特征距离基线中，但没有像上式一样展开其具体评分实现；不能仅凭名称就认为它等于我们的 10D 成功参考 C=4 加局部半径归一化。

我们最新全量 kNN/K-means 主线使用成功参考距离，和上述双库分数不同。较早 Round 3 已有成功/失败距离，Round 5 也探索过混合参考；需要对齐的是最新特征、数据划分和校准协议，而不是把双库概念当作从未试过的新方法。

### 4.2 其他基线

| 类别 | 实际比较内容 | 适用性与开销 |
| --- | --- | --- |
| Token uncertainty | token 的最大/平均负对数概率、最大/平均熵 | 不适用于 pi0/pi0* 的连续 flow 输出 |
| Learned OOD | RND、LogpZO，分别学习成功和失败特征分布，再作分数差 | 要训练额外网络，不满足我们的 train-free 约束 |
| Sample consistency | 同一观测采样 10 次动作，计算整体、平移、旋转、夹爪方差和聚类熵 | 主表灰色项；增加策略推理开销 |
| Action consistency | STAC 比较前后两个预测 chunk 重叠部分的 MMD；另比较 STAC-Single | STAC 在仿真用 10 个样本；Single 每次只用一个；OpenVLA 无 chunk 重叠，不适用 |
| SAFE | 两层 MLP，或一层 LSTM，隐藏维数 256 | 使用整条轨迹成败标签训练，不需要失败起始时刻标签 |

动作采样一致性是在**同一个当前观测下多次预测动作**，不是从该状态执行 10 条未来环境 rollout，不会自行形成逐 chunk 的环境分支树。聚类熵使用 Ward 层次聚类，也不是 PCA-KMeans 那一行基线。

除 RND 和 LogpZO 外，基线还选择是否对逐步分数做累计和。SAFE-MLP 将每步 sigmoid 输出累加；SAFE-LSTM 顺序读取历史再输出分数。累计和与后面评价使用的历史最大值是两个不同操作。

## 5. 怎么评价误报、报警时间和时长混淆

### 5.1 不固定阈值的排名评价

Sec. 5.4 用整条轨迹的历史最大分数计算 ROC-AUC，因为成功轨迹任意一次越阈都会成为误报。结果分别报告 seen 和 unseen。

**LIBERO 主表有关键的时长控制。** 成功轨迹提前结束，失败轨迹跑到上限，单纯计时就可能得到完美区分。因此 App. B.5 明确规定：Table 1 对每个任务取该任务最短 rollout 长度作为共同 T，将成功和失败都截到相同窗口再评价。不能把论文主表理解为不受限制地使用每条轨迹各自的完整长度。

这里也有一个需要复现时确认的符号差异：Sec. 5.4 写历史最大分数，App. B.5 写共同 T 处的 s_T。共同长度处理是明确的，但对非单调分数究竟取末端值还是窗口内最大值，单靠这两处文字不能完全消歧，应进一步核对所用代码版本。也不能把 Table 1 的截断规则自动推广到所有 CP 和时序图。

SimplerEnv 无论成败都执行到最大长度；真机 Franka 同任务也固定长度，任务间为 200-500 步。已核查文本没有对真机 WidowX 给出同样明确的固定长度说明。

### 5.2 固定报警规则的评价

采用成功校准轨迹构造 functional CP 的时变上界：

```text
threshold_t = mu_t + h_t
alarm = first time s_t > threshold_t
```

这不同于我们当前“轨迹峰值 -> task/init 组峰值 -> 一个固定阈值”的校准。论文扫描 alpha，观察 TPR、TNR/FPR、balanced accuracy 和平均相对检测时间；balanced accuracy = (TPR + TNR) / 2。T-det 对所有失败轨迹平均，漏检的检测时间设为 1，不能只对已检出的失败取均值。

Figure 4 给出检出表现与检测速度的权衡，Figure 9 进一步按 alpha 展示 TNR、TPR 和 balanced accuracy。正文与附录也指出跨任务分布变化可能破坏 CP 所需假设，实际 TNR 可以低于 1-alpha；名义 alpha 不是任意新任务的无条件误报保证，更不是报警精确率。

### 5.3 是否真的在物理失败前报警

App. C.3 在轨迹录完后由人工看视频标记失败时刻，执行时没有干预。冻结或反复尝试的情况按“人认为需要接管的时刻”标注；看起来合理但超时未成功的轨迹可能标在结束时。只标注失败轨迹，成功轨迹中间的暂时问题没有同等标注。

Figure 10 包括 pi0-LIBERO、pi0-FAST-LIBERO、pi0-FAST-Franka，使用 seed 0、alpha=0.15：左列是随时间累计检出的失败数量，右列逐条比较检测时刻和人工时刻，漏检另作标记。这是报警时间分布分析，超过只报告一个中位数的信息量，但不是成功轨迹误报时间分布的完整审计。

## 6. 消融实验及其实际结论

### 6.1 特征聚合和超参数

App. B.1/B.8 比较适用于对应策略的 First、Last、Mean、First&Last 聚合。OpenVLA/pi0-FAST 沿输出 token 聚合；pi0/pi0* 分别沿动作 horizon 和 flow step 聚合。pi0-FAST 还比较最终 RMSNorm 之前的 encoded 和之后的 pre-logits。真机实验固定使用 pre-logits + Mean。

Tables 9-11 主要列搜索候选与选中的配置，不等于为每一种配置都公开了完整性能曲线。参数按 eval-seen ROC-AUC 选择。作者没有在这里完成逐 Transformer 层的全面消融，不能把 token/flow 聚合实验误称为层选择实验。

### 6.2 训练任务数量

Table 6 在 OpenVLA + LIBERO 改变训练任务数，未见测试任务保持一致，平均 3 个种子。以下为 unseen ROC-AUC，沿用论文的百分制显示：

| 训练任务数 | 1 | 3 | 5 | 7 |
| --- | --- | --- | --- | --- |
| SAFE-MLP | 63.76 | 67.03 | 68.17 | 73.47 |
| SAFE-LSTM | 52.25 | 63.31 | 66.31 | 72.47 |

该结果支持任务覆盖增加有助于检测器泛化，但没有独立隔离“任务多样性”和“总参考数据量”两个因素。对 train-free 方法，可以改做参考任务数量消融，并增加固定总参考点数的对照。

### 6.3 特征来源

Table 7 在相同 Franka 轨迹上更换特征来源。以下为 unseen ROC-AUC：

| 特征 | SAFE-MLP | SAFE-LSTM |
| --- | --- | --- |
| DINOv2 | 59.46 | 56.96 |
| CLIP | 59.77 | 52.71 |
| DINOv2 + CLIP | 58.43 | 59.65 |
| VLA 末层 | 64.16 | 58.70 |

最好的组合是 VLA 特征 + MLP，但 VLA 特征并非在每一种检测头下都最好。我们要论证 MoE 信息的价值，需要同策略、同轨迹、同评分和校准下与末层 hidden 特征对照，不能只比较来自不同论文的结果数字。

### 6.4 稳定性和运行效率

Table 8 补充结果标准差，重复实验改变随机种子和任务划分。Sec. 6.4 给出一个部署开销实例：SAFE-LSTM 约 2.3M 参数，增加 0.73 ms；对照 pi0 约 3.3B 参数、149 ms 推理，小于 1% 开销。这是检测头开销，不包含另行执行未来环境 rollout 的成本。

## 7. 主结果能支持多强的结论

Table 1 的仿真 unseen 平均 ROC-AUC 为 SAFE-MLP 78.00、SAFE-LSTM 77.04、余弦 kNN 73.93、欧氏 kNN 69.62、PCA-KMeans 58.44。SAFE 在平均结果上较好，但不是每个模型/场景都第一，例如 pi0-FAST-LIBERO 的 STAC 为 85.31，高于 SAFE-LSTM 的 84.48 和 MLP 的 80.44。

真机 Figure 6 中，Franka unseen 的 SAFE-MLP 为 64.16，欧氏 kNN 为 60.27，泛化仍有明显困难；WidowX unseen 的 SAFE-MLP 为 88.42，表现更强。这与二维特征在 Franka 上难以分开的附录观察一致，但不能据此证明二维可分性就是唯一原因。

这些数字是 ROC-AUC，不是报警 precision。尤其 Franka 人为收集每任务各 30 条成功和失败，而我们全量数据失败率约 3.425%，不能直接拿论文 ROC-AUC 与我们的 precision 比谁更强。论文中 PCA-KMeans 的结果也不能直接判断我们的 C=4 半径方法好坏，因为表征和评分不同。

## 8. 没有做的实验，以及我们最值得借鉴的部分

App. F 明确将以下内容留给未来：中间层或多层特征融合、在线自适应 CP、报警后的恢复策略、通过 activation steering 改善执行。论文没有给出 MoE 路由检测、改噪声后成功率、同一状态分支搜索或闭环救回率的实验结果。

结合我们已有的实现，优先值得补齐或统一的对照是：

1. **控制执行时长。** 在最新 32,000 条全量协议中补充共同窗口或预先固定检查点的比较，并保留真实完整执行时的误报分布，检验方法是否只在临近超时报警。
2. **对齐双库距离。** 在最新 MoE 特征和参考预算下比较成功距离、成功减失败距离、现有 K-means 半径分数；复用早期代码思想，重新做完整任务留出与校准。
3. **做匹配的特征来源对照。** 固定 train-free 评分器，比较末层 hidden、MoE 路由和动态组合，才能识别提升来自哪里。
4. **完整画时序分布。** 仿照 Figure 10 对齐物理事件，同时增加成功轨迹首次误报分布、每任务分布和有效轨迹数量；不能只保留成功检出的案例。
5. **测新任务参考覆盖。** 扫描历史参考任务数，固定未见测试任务，并单独控制参考点总量，判断误报是否来自覆盖不足。

这些是由文献和当前诊断得到的后续实验建议，不是本次核查已经完成的新实验。执行恢复实验时还需要另外评价干预后的实际成功率，检测提前不能替代救回证据。

## 9. 原文内部需要核对的地方

| 项目 | 冲突或不完整处 | 本文采用的处理 |
| --- | --- | --- |
| SimplerEnv seen/unseen 数量 | Sec. 5.1 和 App. B.5 写 3/1；Table 5 写 2/2，但 198+102 训练/seen 与 100 unseen 的条数又对应 3/1 | 以正文、详细协议和条数一致的 3/1 描述，同时保留冲突说明 |
| Table 5 的 Octo | 表中有 Octo SimplerEnv 一行，但主模型说明及主结果没有对应完整对照 | 不将其算作已完整报告的主要模型实验 |
| LIBERO 主表取分数方式 | Sec. 5.4 写历史最大分数，App. B.5 写共同 T 的 s_T | 确认有共同窗口处理；末端值与窗口最大值的实现需代码消歧 |
| PCA-KMeans 精确评分 | 正文提供距离类总公式，但没有单独展开该方法实现 | 确认比较过 PCA-KMeans 及其参数，避免擅自补成我们当前的评分公式 |

对应源码入口：`sections/3-method.tex`、`sections/4-exp.tex`、`sections/5-results.tex`、`sections/a-appendix.tex`，以及 `figure-table/sim-results-v3.tex`、`real-results.tex`、`fail-detect-timing.tex`、`hparam-*.tex`。这些文件均位于上述本地源码包内。
