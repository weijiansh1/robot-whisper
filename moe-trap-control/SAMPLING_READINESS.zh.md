# 采样就绪状态与实测利用率

2026-09-08 后续干预更新：已复用 Pro 50、Plus 70 条 Long 主轨迹，在 v7、v8、欧氏 kNN、欧氏 OR 余弦 kNN 各自报警后的下一条 query 上比较随机噪声、随机短 chunk、路由边缘噪声和专家替换。新增 **120 次完整 C0、98 个状态、1,568 条后缀、40,105 次实际推理**，全部通过独立审计；唯一主轨迹覆盖仍为 2,150 条。短 chunk 未优于随机，边缘在 kNN 下多一次救回也多一次破坏，净收益为零。定义、分母、Pro/Plus 分组及功耗见 [Long 报警后一步实验](LONG_CONTINUATION_EXPERIMENT.zh.md)。原 35 个端点健康检查通过，临时进程已退出，GPU 6 排除，无 hidden。

2026-09-08 正式实验更新：四个 suite 的 120 个唯一预检变体已通过完整主轨迹/C0 审计。七卡 8 路 MPS 的正式采集现已累计 **2,150 条唯一主轨迹、2,150 条完整 C0、4,208 条 C1/T1 后缀**，全部通过独立审计；包含 Long/Goal 各 950 条初筛，以及 250 条 Long 新种子扩展。GPU 6 排除、无 hidden，原 35 个服务实际推理健康检查通过，新增进程均已退出。最新 1,200 条的任务、种子、冻结报警比较、干预结果与真实功耗见 [追加批次记录](NEXT_BATCH_EXPERIMENT.zh.md)。以下压测和就绪描述保留为此前阶段记录。

后续 [MPS 功耗与吞吐实测](MPS_POWER_RESULT.zh.md)：单卡 8 路约 7.95 query/s、290 W，12 路平均约 296 W。真实观测回放全部一致；MPS 下完整 C0 和七卡采集仍需验证，本页的真实采集数据仍来自非 MPS 配置。

更新：2026-09-08。原生采集入口已验证 Pro/Plus Goal 24 个变体和 Long 48 个变体，全部主轨迹与完整 C0 通过审计，无 hidden，沿用冻结 v7。Long 已使用七卡、每卡两个共享只读权重的独立推理进程及四个常驻环境任务位完成真实采集，13.07 query/s，全程平均 GPU-Util 70.85%、功耗 163.32 W。C1/T1 干预尚未接入，Spatial/Object 尚未做该入口的完整恢复验证。详见 [Long 试采样与功率报告](LONG_HIGHLOAD_TRIAL.zh.md) 和此前的 [Goal 采集报告](RESOURCE_UTILIZATION_RUN.zh.md)。

[合并审计](design/collection_goal_long_audit_20260908.json) 已通过全部 72 个唯一变体。Long 的额外样本不能替代 Spatial/Object 的 suite 覆盖，因此不把 72 直接解释成 P0 的完成比例。

后续已完成 [独立进程对照](GPU_UTILIZATION_IMPROVEMENT.zh.md)：相同 Goal 负载下，七卡吞吐从 1.96 提升到 13.20 query/s，平均利用率从 6.18% 提升到 40.50%。这是临时服务的实测，原 35 模型服务尚未切换；不代表完整采集器已就绪。

此前的 [同卡双进程对照](GPU_EFFICIENCY_FOLLOWUP.zh.md) 是 Goal 推理短测。最新 Long 实测已验证七卡双进程的完整 HB 与 C0；补充的单卡四进程回放把吞吐从 2.67 提升到 3.27 query/s，功耗从 174 W 升到 185 W。四进程尚未完成七卡完整 rollout 验证，GPU-Util 接近 99% 不代表算力饱和。

## 当前状态

| 项目 | 证据与状态 |
|---|---|
| 预加载模型 | 物理 GPU 0/1/2/3/4/5/7 上 35/35 个端点本次通过身份、checkpoint 加载记录、归一化及真实推理检查 |
| Pro / Plus | 已安装；200 / 10,030 个注册条目核验通过；此前 20 + 28 个代表组合完成短程模拟器检查 |
| 报警参数 | 已冻结并完成 32,000 条离线计算与复现，接通采集时可直接加载 |
| 正式清单 | 14,030 个唯一 main_id，仍为 planned_pending_preflight；24 个 Goal 与 48 个 Long 预检变体单独保存，不重复记入正式样本 |
| 原服务在线 hook | 原 28 个 LIBERO 端点仍只返回 top-4 ID/weight；完整 HB 由新增独立采集实例提供，原服务未升级 |
| 新采集入口 | run_collection_preflight.py / collect_preflight_worker.py 返回完整 HB 概率及实际 native/effective ID、weight，不存 hidden；原 baseline 入口保持原样 |
| 报警快照恢复 | 新入口补齐 RNG、observable、实际观测、控制器数值状态、gripper.current_action 和 actuator 状态；24 个 Goal 与 48 个 Long 的完整 C0 后缀逐项一致 |
| 分支与分块 | 首次报警快照持久化，main_complete 提交后才执行 C0；8 query/chunk、最多 2 个待写块。C1/T1 和正式前缀引用仍待实现 |
| 渲染与 CPU 性能 | 推理可用七卡，EGL 默认绕开回放有差异的 GPU 1/2；Plus glass noise 的字节与 RNG 等价加速已通过 20 组函数对照及一条完整轨迹对照 |

Pro/Plus 的安装 smoke 不等于正式 rollout；当前 design/experiment_config.json 也仍明确标为 design_only_not_a_runnable_collector_config。旧 recorder 的实际执行权重记录和强制 hidden 开关问题仍须在新采集接口中处理，详见 [hook 审计](HOOK_COLLECTION_AUDIT.zh.md)。

## 之前的短压测

负载为真实预加载模型、合成合法观测、显式固定 flow noise，四个 LIBERO 模型轮询，**开启现有 top-4 路由返回**。每 GPU 两个请求流，每档持续 30 秒后排空队列。没有模拟器、完整 HB recorder、报警快照、干预分支或正式数据写入，因此不是完整采集吞吐。

| 模式 | 允许 GPU 数 | 完成 query | 总吞吐 query/s | 每卡平均 GPU-Util | 平均功耗/卡 |
|---|---:|---:|---:|---:|---:|
| 当前七卡矩阵 | 7 | 56 | 1.478 | 4.13% | 127.71 W |
| 同负载仅调用物理 GPU 0 | 1 | 52 | 1.650 | 35.41% | 145.74 W |

七卡各卡的平均利用率范围为 3.86%..4.32%，峰值采样为 20%；各卡平均功耗 119.41..140.65 W，功率限制均为 500 W。单卡峰值利用率为 49%。利用率来自 nvidia-smi GPU-Util，不是 FLOPs 利用率或模型显存占比。

七卡吞吐包含 37.89 秒的发起和排空阶段；单卡也按排空后的实际耗时计算。GPU 利用率只汇总发起窗口内的定时采样。短测存在采样波动，不能解释为长时稳定上限。

七卡总吞吐只有单卡对照的约 0.90 倍。此前关闭路由返回的七卡短测约为 6.1% / 2 query/s；本次开启 top-4 的数字不能外推为完整路由采集、Pro/Plus 仿真和恢复落盘后的性能。

## 原矩阵的瓶颈

当前仍是 PID 28531 的 serve_model_matrix.py：一个 Python 进程放置全部 35 个模型，每张 GPU 共享一个单线程执行队列，并经过进程共享的 heap-trim 锁。模型参数已经常驻，但宿主进程的多卡推理扩展效率很差。

单卡/七卡对照支持优先改成每张允许卡一个独立推理进程、每卡用有界环境请求队列；GIL、算子发射、内存回收各自耗时尚未分别剖析。进程拆分后应重新测真实采集吞吐，不能提前承诺 80%..100% GPU 利用率。单卡本次约 35% 只是可参考的实测点，不是七卡或完整 collector 的保证。

当前每张允许卡约占 95,008..95,068 MiB / 143,771 MiB，显存占比约 66.1%；这一占比不代表计算利用率。宿主 cgroup 有 512 GiB 内存与 96 核 CPU 配额，工作盘剩约 41 GiB。当前首要问题是采集接口完整性和多卡扩展效率，正式存储预算还需用完整 schema 的实测压缩量更新。

## 尚需完成的工作

1. Goal/Long 完整链路已经验证；Spatial/Object 已有模型映射，但仍需逐 suite 验证快照与完整 C0。
2. 接通 C1/T1 路由干预、事件和分支身份、正式前缀引用；修正 Pro 中未改变场景条目的分析角色。
3. 继续提高专家执行效率，并验证四进程方案的完整 rollout。七卡 Long 双进程本批全程 GPU-Util 70.85%，功耗 163.32 W；收尾阶段仍有空闲，且尚未证明计算单元饱和。
4. Goal 平均约 2.79 MiB/主轨迹+C0，Long 平均约 4.81 MiB；保留 8 GiB 后按 Long 分布约还能存 6,734 对。C1/T1 接入后需要重新预算。

本次压测客户端已退出，原 35 个预加载服务保留。物理 GPU 6 未参与本次模型加载、推理、渲染或压测；未采集 hidden，未修改报警参数。

数据：[七卡当前压测](design/preload_readiness_current_20260908.json)、[单卡当前对照](design/preload_readiness_single_current_20260908.json)、[既有安装验证](benchmarks/READINESS.json)、[正式清单](design/collection_manifest.csv)。
