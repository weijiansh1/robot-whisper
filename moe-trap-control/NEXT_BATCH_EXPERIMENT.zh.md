# 追加采集：Long、Pro、Plus

后续已完成 [Long 报警后一步干预实验](LONG_CONTINUATION_EXPERIMENT.zh.md)：复用 120 条 Long 主轨迹，补齐 v8、欧氏及欧氏 OR 余弦 kNN 各自报警后的干预状态，比较随机短 chunk、路由边缘噪声与专家替换。新增 1,568 条后缀全部审计通过，不增加唯一主轨迹数量。下文保留原 1,200 条追加采集及其 v7 报警点干预结果，两种干预时序分开统计。

2026-09-08。状态：**按 Long → Pro → Plus 的顺序完成本轮 1,200 条主轨迹、1,200 条完整 C0 和 2,096 条 C1/T1 后缀，全部通过独立审计及冻结报警复算**。原 35 个服务的实际推理健康检查通过，全部新增采集进程已经退出。GPU 6 未用于推理、渲染或 MPS，不保存 hidden。

## 采集结果

| 项目 | Long 新种子 | Pro-Goal | Plus-Goal | 本轮合计 |
|---|---:|---:|---:|---:|
| 唯一主轨迹 | 250 | 250 | 700 | **1,200** |
| 原生成功 / 失败 | 90 / 160 | 80 / 170 | 539 / 161 | 709 / 491 |
| 首次 v7 报警状态 | 121 | 62 | 79 | 262 |
| 完整 C0，通过 / 总数 | 250 / 250 | 250 / 250 | 700 / 700 | **1,200 / 1,200** |
| C1/T1 后缀，不含 C0 | 968 | 496 | 632 | 2,096 |
| 主轨迹 query | 10,621 | 5,918 | 11,107 | 27,646 |
| C0 query | 6,801 | 3,753 | 6,493 | 17,047 |
| C1/T1 query | 24,928 | 4,880 | 6,478 | 36,286 |
| 全部 query | 42,350 | 14,551 | 24,078 | **80,979** |
| 轨迹逻辑字节，GiB | 2.740 | 1.111 | 2.117 | **5.968** |
| 采集分钟，不含加载和独立审计 | 20.03 | 8.52 | 14.62 | **43.17** |
| 完整吞吐，query/s | 35.25 | 28.46 | 27.45 | 31.27 |

本轮轨迹文件共 **6,408,053,762 bytes**。没有运行失败、C0 不一致、缺失分支或配额截停。独立审计逐块核对校验和、种子、主轨迹提交、完整恢复、分支起点/前缀身份、实际输入、配对噪声和剩余步数。1,048 对 C1/T1 首 query 动作全部有数值变化；每条 T1 首 query 都精确验证了规定的 400 个路由槽位替换。

各段 [Long 审计](design/experiment_next_long_seed_audit_20260908.json)、[Pro 审计](design/experiment_next_pro_goal_audit_20260908.json)、[Plus 审计](design/experiment_next_plus_goal_audit_20260908.json) 均通过。[Goal 合并验证](design/experiment_goal_screen_summary_20260908/verification.json) 确认 Pro/Plus 两段恰好覆盖原 Goal 初筛 950 个 ID 一次。与此前 Long 初筛 950 条合并后，累计 **2,150 个唯一正式 main ID**、526 个报警状态、4,208 条 C1/T1 后缀、188,408 次推理。

[统一完成索引](design/collection_status.json) 中，**初筛为 1,900 / 3,800，Long 扩展为 250 条，分开计数**。Long、Goal 两个 suite 的初筛均已完成；Spatial/Object 各 950 条尚待正式采集，预检样本不计入正式进度。原 14,030 条覆盖清单还剩 11,880 条。

## 报警比较

下表为本轮新增 **Goal 全部 950 条主轨迹**，共 331 条原生失败、619 条原生成功。精确率为 `TP/(TP+FP)`，召回率为 `TP/331`，误报率为 `FP/619`。全程沿用同一冻结参数，不按 suite 重新校准。

| 方法 | TP | FP | 精确率 | 召回率 | 误报率 |
|---|---:|---:|---:|---:|---:|
| v7 | 135 | 6 | 95.74% | 40.79% | 0.97% |
| v8 | 259 | 29 | 89.93% | 78.25% | 4.68% |
| v8.2 | 271 | 30 | 90.03% | 81.87% | 4.85% |
| 欧氏 kNN，k=20 | 252 | 7 | 97.30% | 76.13% | 1.13% |
| 余弦 kNN，k=20 | 4 | 3 | 57.14% | 1.21% | 0.48% |
| 欧氏 OR 余弦 | 253 | 10 | 96.20% | 76.44% | 1.62% |
| k-means 32 簇联合半径 | 208 | 8 | 96.30% | 62.84% | 1.29% |

Pro-Goal 的 v7/v8/v8.2 召回率分别为 36.47%/74.12%/78.82%，本批 80 条成功轨迹上均无误报。Plus-Goal 对应召回率为 45.34%/82.61%/85.09%，误报率为 1.11%/5.38%/5.57%。因此在同一参数下，v8/v8.2 的召回增加伴随 Plus-Goal 误报增加；观测到 `0/80` 也不代表真实误报概率为零。

欧氏与余弦 OR 在 Pro-Goal 多检出 1 条失败且未增加误报，在 Plus-Goal 没有新增失败召回、增加 3 个误报。合并 Goal 后是多 1 个 TP、多 3 个 FP，不能只报告召回增加。

Long 新种子 250 条单列：v7 召回 74.38%、误报 2.22%；v8.2 召回 87.50%、误报 3.33%；欧氏 kNN 召回 99.38%、误报 5.56%。该组是 Pro Long 的额外种子，样本构成与之前 Long 初筛的 Pro/Plus 混合不同。

全部 11 种方法及 TP/FP/FN/TN 已保存：[Goal 合并指标](design/experiment_goal_screen_summary_20260908/metrics.csv)、[Long 新种子指标](design/experiment_next_long_seed_alarm_comparison_20260908/metrics.csv)、[Pro 指标](design/experiment_next_pro_goal_alarm_comparison_20260908/metrics.csv)、[Plus 指标](design/experiment_next_plus_goal_alarm_comparison_20260908/metrics.csv)。本轮 27,646 次主轨迹在线 v7 分数和报警均由离线复算精确复现，v8/v8.2 确认逻辑、几何距离和前缀/顺序不变性也通过检查。以上均为当前抽样的描述性结果，不是完整 benchmark 分数。

## 干预结果

**本轮原生失败报警状态中，C1 和 T1 均未观察到救回。** Goal 的 135 个失败报警状态全部未恢复，6 个成功报警状态两组均保持成功，`T1-C1=0`。

Long 扩展主要扰动分析中有 120 个报警状态，包括 118 个原生失败和 2 个原生成功。状态内先平均四次重复后，C1 报警状态成功率 0.625%，T1 为 0.417%，差值 **-0.208 个百分点**；差异来自成功状态的破坏，两组破坏率分别为 62.5% 和 75.0%，分母仅 2 个状态。该结果没有按原始任务聚类估计置信区间，不作显著性结论，也不使用 best-of-four。结论限于当前冻结 `swap1_near`、噪声流和已采状态。

[Long 配对分析](design/experiment_next_long_seed_analysis_20260908.json)、[Pro 配对分析](design/experiment_next_pro_goal_analysis_20260908.json)、[Plus 配对分析](design/experiment_next_plus_goal_analysis_20260908.json) 保留逐状态结果。只有原冻结在线 v7 报警位置具有配对干预数据，离线 v8/kNN 新报警点尚无本轮干预收益证据。

## 实测资源

| 指标 | Long 新种子 | Pro-Goal | Plus-Goal |
|---|---:|---:|---:|
| 全程平均功耗，W/卡 | 234.41 | 211.91 | 209.12 |
| 七卡任务位均供满时平均功耗，W/卡 | 254.39 | 225.69 | 219.60 |
| 任务位均供满时平均 GPU-Util | 94.53% | 89.48% | 88.42% |
| 单卡采样峰值，W | 307.59 | 289.33 | 294.23 |

**峰值 307.59 W，不是持续 300 W。** 三段均包含真实模拟器、采集和分支工作；队列收尾时负载下降，GPU-Util 也不等于 FLOPs 利用率。宿主内存最大峰值约 319.56 GiB。本轮在保留原服务和 batch=1 的条件下执行，没有调功率上限、时钟或制造额外负载。

[Long 功耗曲线](design/experiment_next_long_seed_resources_20260908.png)、[Pro 功耗曲线](design/experiment_next_pro_goal_resources_20260908.png)、[Plus 功耗曲线](design/experiment_next_plus_goal_resources_20260908.png) 均有同名 PDF 和来源校验 JSON，并已检查图像。最终工作盘可用约 **25.69 GiB**，高于 8 GiB 底线。

[最终健康检查](design/experiment_next_final_health_20260908.json) 验证原 PID 28531 下 **35 个端点实际推理、checkpoint 和归一化全部通过**；本轮及此前记录中的临时模型、环境和 MPS 进程均已退出。GPU 6 排除。

## 冻结范围

本轮新增 **1,200 条唯一主轨迹**，按 Long 新策略种子、Pro-Goal、Plus-Goal 的顺序执行。Long 是任务套件，Pro/Plus 是扰动基准；本轮在 Long 之后扩展 Goal，Spatial/Object 留待后续批次。选样不读取成功、失败或报警结局。

| 顺序 | 样本 | 新主轨迹 | 冻结计划 |
|---|---|---:|---|
| 1 | Pro Long：全部 50 个变体，init 0..4，原清单 policy seed index 1 | 250 | [Long 新种子](design/experiment_next_long_seed_plan_20260908.json) |
| 2 | Pro Goal：全部 50 个变体，init 0..4，原清单 policy seed index 0 | 250 | [Pro Goal](design/experiment_next_pro_goal_plan_20260908.json) |
| 3 | Plus Goal：原冻结初筛，七类各 100 个变体 | 700 | [Plus Goal](design/experiment_next_plus_goal_plan_20260908.json) |

第一段属于原 14,030 条清单中的覆盖扩展，**不计入 3,800 条初筛的新增进度**。它重复同一任务/初始状态但使用原清单中另一条固定策略噪声种子；分析时保留这种配对和原始任务聚类关系。后两段恰好完成 Goal 的 950 条初筛。全轮含 1,175 条扰动样本和 25 条 Long 原场景对照，后者单列。

## 身份与规范

- 原清单 SHA-256：`4ebe63a918a48202d628fb428f0fe5998fdfef1a832faa40db6597796ca57767`。
- 报警参数 SHA-256：`f4e65d359411309756b008423d14d8f3e5d6bf604ee71b63ae8a7e510c5f8216`，全程不重新拟合。
- 第一段计划 SHA-256：`301c2541c830f46a2b14572a8ac5bf5ad41eb7fa4ac9da2c98e4a7e8f5ebf56c`。
- 第二段计划 SHA-256：`49f81e0256bb002ca885f9fd90b2d61d3ff4d5d00dad0d9314f2643d71fe5c9e`。
- 第三段计划 SHA-256：`ef8d798a5051490e7d3a729ab60ab326619e81910c16fbbf2cb9bf03464da371`。

沿用 `moe_control.paired_swap1.v1`：原生主轨迹不干预，保存首次 v7 报警的推理前状态，主轨迹提交后完整 C0 精确恢复，通过后再执行 C1 四次、T1 四次配对后缀。所有首次报警都展开，包括原主轨迹最终成功的样本；无报警做 q5/q0 完整 C0 质量检查。Long/Goal 时限分别为 520/300 环境步，分支只使用起点剩余步数。

记录完整 HB/AS 路由概率、原生/执行专家及实际权重、动作、proprio、噪声、状态和恢复快照。8 query/chunk，最多 2 个待写块。不保存 hidden，不改 batch，不挑最优重复。离线按同一冻结配置比较 v7/v8/v8.2、欧氏/余弦 kNN、k-means 和原有组合；只有在线 v7 报警状态具有本轮配对干预数据。

## 资源与执行

GPU 0/1/2/3/4/5/7，每卡私有 MPS、8 个独立 batch=1 推理进程和 8 个环境任务位，共 56 路。渲染使用 0/3/4/5/7。**GPU 6 排除**，原 35 个预加载服务保留；每段只清理本段新增进程。

冻结时工作盘空闲约 31.8 GiB。启动前按首批 Long 实测均值估算三段合计 9.90 GiB，加 15% 余量约 11.39 GiB；实际完成量为 5.968 GiB，差异来自各段报警比例、主轨迹与后缀长度。每段数据配额分别为 8/6/12 GiB，全部遵守 8 GiB 可用空间底线，派发时预留在途主轨迹及承诺分支的最坏情况空间。单任务时限 1,800 秒。

启动时允许卡空闲显存约 48 GiB；[前一段结束健康检查](design/experiment_long_scale_final_health_20260908.json) 验证原 35 个服务的实际推理、checkpoint、归一化和进程身份全部通过。三段新服务启动均逐卡逐进程与原服务核对固定输入的动作与路由，全部通过。

各段运行目录：

- [Long 新种子](design/experiment_next_long_seed_20260908/summary.json)
- [Pro Goal](design/experiment_next_pro_goal_20260908/summary.json)
- [Plus Goal](design/experiment_next_plus_goal_20260908/summary.json)

三段全部完成主轨迹/C0/C1/T1 独立审计、冻结报警复算及资源分析，通过验收的 ID 已进入统一完成索引。Goal 两段恰好覆盖原初筛集合一次，Long 扩展单列。

本轮增加了显式的 `coverage_extension` 计划分区，默认计划仍只接受初筛任务。原 CSV、主轨迹 ID、种子及干预协议保持冻结。5 项针对性测试验证：扩展必须声明、不可重分类初筛、种子不可变、已完成 ID 拒绝重复、扩展不会减少初筛剩余数。
