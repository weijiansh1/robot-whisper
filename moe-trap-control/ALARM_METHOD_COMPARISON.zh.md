# v7、v8、KNN、KMeans 比较与 MoE 提取方案

更新：按“全部只用一套参数”的要求完成了 [固定参数全量比较](FROZEN_ALARM_COMPARISON.zh.md)，后续查阅固定工作点请以该报告为准。本文的分折重校准结果保留为此前的独立实验，其参数、参考范围和指标不能与固定版本混用；下文的候选排序也仅对应本文实验。

日期：2026-09-08。本轮使用已有 HUB 数据进行了统一离线比较，未生成新 rollout，未调用 GPU 模型，未采集 hidden，未使用物理 GPU 6。Pro/Plus 的干预收益尚未测量。

当前证据支持把 v8.2 与 KMeans C4 作为后续 MoE-Control 的两条候选，保留 v7 和 KNN-20 对照。v8.2 在本轮统一校准下误报较少；C4 能多检出一部分失败；KNN 的高召回伴随更多误报。它们均不能仅凭完整轨迹指标被认定为有效的早期救回触发器。

## 1. 本次怎样保证可比

使用 A/B 两批共 **32,000 条唯一轨迹、1,096 条失败、30,904 条成功、508,023 次真实 query**。沿用 Round 9 的 16 个任务留出折，每条轨迹仅测试一次。每折使用同一 7 个参考任务，A 中 30 个初态建参考，另外 10 个初态校准，测试任务整体排除；不混用不同 checkpoint 的参考点。

所有主表方法独立按成功校准轨迹的全程峰值、再按 task/init 组内最大值校准，名义预算为 2% / 5%。校准和测试均屏蔽无效 padding；测试成功轨迹在真实结束前的所有报警均算 FP，不因提前量不足而删除。

v7/v8/v8.2 主表是复用 [已有 guard 适配器](/home/jovyan/work/himoe-vla/safe&vlaconf/moe_trainfree/v82_validation/monitor.py:108)、在本次相同参考任务上重新计算尺度与成功阈值的版本，名称为 `*_calibrated`。它们保留对应头、平滑、确认与时间斜率，但不是原项目的冻结多阈值工作点。原版结果见第 4 节。

KNN/KMeans 复用相同成员的既有分数与簇模型，本轮重新计算阈值和首次报警，并逐项确认与旧全量结果一致。没有重新挑选 k 或 C。**C4 已由此前扫描选出，属于探索候选；C32 是原 KMeans 主配置。** 数据已被历史研究检查过，本次不是新的盲测。相同名义预算不保证相同测试 FPR，也不构成 Pro/Plus 上的误报率保证。

## 2. 完整轨迹工作点

统一 task/init 校准，名义预算 5%；正例为最终失败的轨迹：

| 方法 | 检出失败 TP | 成功误报 FP | 召回率 | 实际 FPR | 报警精确率 |
|---|---:|---:|---:|---:|---:|
| v7，重新校准 | 797 | 1,221 | 72.72% | 3.95% | 39.49% |
| v8，重新校准 | 878 | 997 | 80.11% | 3.23% | 46.83% |
| v8.2，重新校准 | 893 | 984 | 81.48% | 3.18% | 47.58% |
| 欧氏 KNN-20 | 977 | 2,754 | 89.14% | 8.91% | 26.19% |
| KMeans C4，最近簇半径归一化，探索 | 942 | 1,220 | 85.95% | 3.95% | 43.57% |
| KMeans C32，最近簇半径归一化 | 982 | 2,906 | 89.60% | 9.40% | 25.26% |
| KMeans C32，簇并集半径 | 921 | 1,799 | 84.03% | 5.82% | 33.86% |
| 仅 10D 向量范数 | 824 | 1,467 | 75.18% | 4.75% | 35.97% |
| 纯余弦 KNN-20 | 38 | 672 | 3.47% | 2.17% | 5.35% |
| 仅 query 时钟 | 1,096 | 750 | 100.00% | 2.43% | 59.37% |

v8.2 相对 v7 净增 96 TP、减少 237 FP；相对 KNN-20 少 84 TP、减少 1,770 FP。C4 相对 KNN-20 少 35 TP、减少 1,534 FP。这里比较的是所列工作点，没有用测试标签强行对齐实际 FPR。

v8.2 与 C4 有互补：849 条失败二者都检出，v8.2 独有 44 条，C4 独有 93 条。但布尔 OR 会得到 986 TP / 1,568 FP，精确率降至约 38.61%，不能将“两者都有用”直接改写为“默认 OR 更好”。这是已有独立阈值的事后交并诊断，尚无新的整体误报预算校准。

更严格的名义 2% 预算：

| 方法 | TP | FP | 召回率 | 实际 FPR | 精确率 |
|---|---:|---:|---:|---:|---:|
| v7，重新校准 | 664 | 749 | 60.58% | 2.42% | 46.99% |
| v8，重新校准 | 749 | 755 | 68.34% | 2.44% | 49.80% |
| v8.2，重新校准 | 759 | 677 | 69.25% | 2.19% | 52.86% |
| KNN-20 | 773 | 1,318 | 70.53% | 4.26% | 36.97% |
| KMeans C4，最近簇半径，探索 | 721 | 401 | 65.78% | 1.30% | 64.26% |

任务间差异很大。例如 5% 工作点，v8.2 在 Goal 的 FPR 为 6.93%，在 Long 为 1.06%；C4 在 Goal 为 9.89%，在 Long 为 1.10%。整体均值不能代替逐套件、逐任务检查。所有明细见 [统一指标](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/metrics.csv) 和 [套件指标](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/suite_metrics.csv)。

## 3. 用于干预时，必须同时看报警时机

时钟基线的高完整轨迹成绩提示明显的运行时长效应：失败轨迹通常持续到上限，成功轨迹可能已经结束。时钟没有读取最终长度作为运行输入，但完整轨迹评价仍会奖励晚期检出。它在 q13 之前没有任何检出，不能据全程 100% 召回认定它适合早期干预。

以下为同一个 5% 阈值下，**截至 q13 的累计报警**。q 从 0 开始，本批记录中 q13 推理前已执行约 130 个动作，不含初始化静置；成功轨迹的早期误报始终保留在分母中：

| 方法 | q13 前 TP | q13 前 FP | 对全部失败的召回 | 此时报警精确率 |
|---|---:|---:|---:|---:|
| v7，重新校准 | 57 | 1,144 | 5.20% | 4.75% |
| v8，重新校准 | 122 | 903 | 11.13% | 11.90% |
| v8.2，重新校准 | 128 | 874 | 11.68% | 12.77% |
| KNN-20 | 282 | 2,572 | 25.73% | 9.88% |
| KMeans C4，探索 | 198 | 1,066 | 18.07% | 15.66% |
| 时钟 | 0 | 0 | 0% | 无报警 |

q13 是统一诊断窗口，不是对每个任务都相同的物理阶段。没有逐状态可恢复标签，因此此表仍不能表示救回率。

进一步只在同 task、同 q、成功和失败都仍在运行的样本中比较 q7..q13 连续分数，先对 q、再对 task 平均。B 有 36 个可比较任务：

| 连续分数 | B 的 task/query 条件 AUROC |
|---|---:|
| v7，重新校准 | 0.476 |
| v8，重新校准 | 0.626 |
| v8.2，重新校准 | 0.634 |
| KNN-20 | 0.596 |
| KMeans C4 | 0.594 |
| 末端低运动量对照 | 0.696 |
| 时钟 | 0.500 |

v8 新头有早期排序信息，但并未超过该运动量对照。此处是连续分数的条件排序，不是冻结二值报警的 AUROC，也未建立机制因果性。应以 Pro/Plus 中从相同状态展开的 C0/C1/T1 救回、伤害和净增益作为后续判据。分数相关，单个 query 和同初态的多个 seed 不能当作独立样本；本轮未新增置信区间或显著性检验。

## 4. 原版冻结 v7/v8 的历史结果

按轨迹 ID 对齐完整 B 的 16,000 条，564 失败、15,436 成功，计入所有真实执行期间的成功误报：

| 历史工作点 | TP | FP | 召回率 | FPR | 精确率 |
|---|---:|---:|---:|---:|---:|
| v7 frozen | 439 | 81 | 77.84% | 0.52% | 84.42% |
| v8 frozen | 460 | 91 | 81.56% | 0.59% | 83.48% |
| v8.2 frozen | 475 | 99 | 84.22% | 0.64% | 82.75% |
| v8，修正 padding 校准 | 501 | 152 | 88.83% | 0.98% | 76.72% |
| v8.2，修正 padding 校准 | 521 | 228 | 92.38% | 1.48% | 69.56% |

原版使用全局历史参考，参考任务范围和主表任务留出不同，不能将这里的约 83% 精确率与主表 KNN 的 26% 直接比较。原 v8 校准存在全 NaN padding 经 `nansum` 变为零的历史问题；修正版排除无效区后，阈值和工作点都改变。本轮主表采用有效区掩码与同折校准。

历史锚点来自已完成的 [v8.2 补充验证](/home/jovyan/work/himoe-vla/safe&vlaconf/moe_trainfree/results/v82_validation_20260908/REPORT_ZH.md)，本轮重新按相同轨迹和指标汇总，保存输入哈希；没有重选历史冻结阈值。完整表见 [历史工作点](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/historical_frozen_metrics.csv)。

## 5. MoE 需要提取什么

四类方法都可以读取同一次正常推理产生的路由。这里提取的是 **专家分配、路由概率和实际合并权重**，不是另存 hidden 或专家内部激活。

| 方法 | 使用的 MoE 信息 | 在线参考需求 |
|---|---|---|
| v7 | 相邻 query 的逐层路由变化、flow 内加速度、历史复现性 | 冻结尺度/阈值和本轨迹历史 |
| v8/v8.2 | v7 信息，加前后层 flow 路径比和三点 flow-speed 曲率 | 新头尺度/阈值；v8.2 另使用当前 q |
| 10D KNN | 8 个逐层相对 mobility + acceleration + periodicity | 同 checkpoint 的成功参考库、尺度和阈值 |
| KMeans | 同一 10D | 成功参考簇中心、半径、尺度和阈值 |
| JA 路由对照 | 最后 denoise 的实际 top-4 专家集合 | 对齐层/token 的成功参考集合 |

KNN 的 k 是近邻数量，KMeans 的 C 是参考簇数量；它们都不是 MoE 专家数。本模型每个 HB gate 有 32 个路由专家，实际 top-4 分配。

基础张量保持 `HB [8,10,11,32]`：层号 `[2,3,4,5,12,13,14,15]`，10 次 denoise，1 个 state token 加 10 个 action token。当前这些报警特征使用 action token，但保留完整 suffix 轴以兼容现有接口。**只保存最后一个 denoise 或只有 top-4 权重，均不能复现当前全部方法。**

推荐的提取顺序：记录当前 gate 输入下的 native 概率与真实 native ID/weight，执行预定干预，再记录实际返回的 effective ID/weight。在没有干预的主轨迹上，两组相等。干预分支中，下游 gate 的 native 概率已经受上游干预影响，应按实际分支输入解释。不能通过 float16 概率重新 top-k 冒充实际 dispatch，也不能在修改 ID 后重新归一化概率冒充执行权重。

同一 query 的 hook 按层/denoise 顺序收集，在请求结束时统一转 CPU、落盘；begin/infer/end 要在该模型的同一个请求上下文中，避免并发串行号。不为每个报警器各挂一套重复 recorder，也不在每个 gate 上单独进行 GPU 同步。AS 路由可作为补充记录，以上主方法不依赖 AS 或 hidden。

### 字段和容量

| 每 query 数据 | dtype | 未压缩大小 |
|---|---|---:|
| 完整 HB native probability `[8,10,11,32]` | float16 | 55 KiB |
| native 与 effective ID，两份 `[8,10,11,4]` | uint8 | 6.875 KiB |
| native 与 effective 实际权重，两份同形状 | float32 | 27.5 KiB |
| mobility 8、acceleration 1、periodicity 1、flow speed 72 | float32 | 328 B |
| 小计 | | **91,848 B，约 89.7 KiB** |

这是两份 ID/weight 都显式保存的布局，不含 state、action、noise、元数据和报警快照。原计划的 80 KiB/query 不能作为这个布局的原始大小上限，应重新测量压缩率并更新预算。未干预 gate 可用明确的 equality/alias 标志避免重复保存，但读取器必须保留 native/effective 语义。

8-query block 的上述数组约 0.70 MiB，双缓冲约 1.40 MiB，另加轨迹数值字段。10D 特征中的周期性尺度和参考归一化依赖 profile，因此不能只存某个 profile 下的 10D 向量后丢掉原始概率。保存 full HB 后，同一批主轨迹可离线比较更多方法，无需重新执行模型。

机器可读方案：[moe_capture_contract.json](/home/jovyan/work/himoe-vla/moe-trap-control/design/moe_capture_contract.json)。它明确标为 `design_only_not_deployed`。

### 实测提取正确性与 CPU 开销

从 HUB 四个套件各取一条真实轨迹，共重放 **76 query**。直接由原始 HB 概率提取，mobility 最大绝对差 `7.16e-7`，acceleration 差小于 `4.53e-10`，periodicity 和 v8 两个原始特征与缓存逐项一致。

每个套件在一个真实 q12 上热身后重复 200 次，单线程 CPU 中位耗时：

| 部分 | 中位耗时范围 | 主要数值参考数组 |
|---|---:|---:|
| v7 基础统计 + v8 flow 特征 | 0.413..0.419 ms/query | 本轨迹路由历史 |
| KNN-20 搜索 | 0.183..0.207 ms/query | 4,096 x 10 float64，320 KiB |
| KMeans C4，三种分数一起计算 | 0.0181..0.0182 ms/query | 中心和半径 352 B |
| KMeans C32，三种分数一起计算 | 0.0199..0.0200 ms/query | 中心和半径 2,816 B |

这些只测已在 CPU 的路由特征计算与距离读取，不包含 gate hook、GPU 到 CPU 传输、时间平滑、完整 monitor 更新、RPC、VLA 推理或模拟器。KNN 内部索引、profile 元数据和尺度也不包含在表中参考数组字节里，不能将其称为总进程内存。

当前模型服务仍仅返回动作 token 的 top-4，完整提取与新 collector 尚未上线；旧 recorder 的实际权重、hidden 强制开关和快照缺项仍需按 [hook 审计](/home/jovyan/work/himoe-vla/moe-trap-control/HOOK_COLLECTION_AUDIT.zh.md) 修复。本次离线验证不等于已部署在线采集。

## 6. 对 Pro/Plus 采集策略的影响

建议先用同一次 full-HB 提取同时运行 v7、v8.2、KNN-20、KMeans C4，分别记录版本、profile 哈希、分数、阈值、首次报警和锁存状态。v8 固定阈值版保留为时间斜率消融。普通 LIBERO 的现有统计不能直接当作 Pro/Plus 的报警率、精确率或救回率；在 Pro/Plus 上先固定参考与配置，再做 P0。

主轨迹继续不干预，报警时保存执行前状态，主轨迹结束并提交后才展开分支。比较不同报警器时，不能只保存 v7 的报警点：应对每种方法的首次报警按 `(main_id, query_id)` 去重保存，并记录全部触发方法。恢复快照必须包括前次审计确认的 RNG、observable 计时/缓存和真实策略输入。chunk 边界不重置任何报警器历史。

多方法的首次报警可能发生在不同位置，因此分支预算会增加。本轮 5% 校准的四方法组合在 32,000 条主轨迹上：

- 4,389 条至少有一种报警；按每方法首次报警、同一 query 去重后，有 **7,552 个状态**。
- 若每状态都展开 C0 x1、C1 x4、T1 x4，就需 **67,968 条后缀**；若只取每主轨迹最早一个点则是 39,501 条，但无法完整评价其他方法后来独有的报警点。
- 单主轨迹最多有 4 个不同首次报警点。这是现有 LIBERO 数据的预算示例，不是 Pro/Plus 数量预测。

因此先保留去重快照，再按冻结规则从“共同报警/某方法独有报警/较早或较晚报警”分层抽取展开状态并记录纳入概率。比较路由干预效果时，C1 与 T1 必须从同一个状态、同一组 flow/环境噪声出发；比较报警器时则还要报告各自触发覆盖和计算成本，不能将不同起点直接当作同状态的配对干预。

## 7. 验证与复现产物

本轮完成 16 折、11 种方法、2 档预算，独立核验 **704,000 个轨迹级首次报警判断**；所有轨迹只测试一次，校准与测试任务无重叠，既有 KNN/KMeans 预测复现。验证了 v7 基础量与 v8 特征的 q0..q13 前缀不受未来内容改变影响。

相关现有 CPU 测试 **41 passed**；旧 v8 测试有一个已知空切片均值 warning。新比较显式屏蔽无效 query，不使用旧 padding 分数。没有因此修改历史结果文件。

代码：[compare_alarm_methods.py](/home/jovyan/work/himoe-vla/moe-trap-control/compare_alarm_methods.py)、[probe_moe_readouts.py](/home/jovyan/work/himoe-vla/moe-trap-control/probe_moe_readouts.py)。

结果：[校验记录](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/verification.json)、[逐折分数与报警](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/folds)、[成对报警差异](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/paired_alarm_counts.csv)、[早期条件 AUROC](/home/jovyan/work/himoe-vla/moe-trap-control/design/alarm_comparison_20260908/early_conditional_auc.csv)、[MoE 回放与耗时](/home/jovyan/work/himoe-vla/moe-trap-control/design/moe_readout_probe.json)。

```bash
python compare_alarm_methods.py --output design/alarm_comparison_reproduction
python probe_moe_readouts.py --output design/moe_readout_reproduction.json
```

比较输出目录需尚不存在，以免覆盖已有结果。两条命令都只读取已有数据，自动禁用本进程 CUDA。
