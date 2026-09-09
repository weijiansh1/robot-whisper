# GPU 利用率提升：独立进程实测

后续已完成 [同卡双进程与批量推理对照](GPU_EFFICIENCY_FOLLOWUP.zh.md)：同卡两个独立进程各保持 batch=1，吞吐从 1.83 提高到 2.75 query/s，GPU-Util 约 98%，本次动作及 top-4 路由核验完全一致。batch=8 可达到 8.49 query/s，但路由集合约 12.5% 发生变化，暂不适合直接接入严格配对采集。

2026-09-08 03:09..03:12 UTC。最先值得实施的改动是 **每张允许 GPU 一个独立推理进程**。在同一 Goal checkpoint、batch=1、开启 top-4 hook 的对照中，七卡吞吐从 **1.96 提升到 13.20 query/s，约 6.74 倍**；平均 GPU-Util 从 **6.18% 提升到 40.50%**。

本次已经实现可运行的独立进程服务包装和自动对照脚本，但没有替换原来的 35 个预加载服务。临时进程在测试结束后全部退出；正式采集器仍有 [就绪报告](SAMPLING_READINESS.zh.md) 中的接口待完成项。

## 1. 控制条件与结果

使用物理 GPU 0/1/2/3/4/5/7，GPU 6 不参与加载、推理或渲染。利用各卡空闲显存临时加载一份相同的 Goal 模型，原五模型 bundle 始终保留，因此各阶段的常驻显存条件相同。

两种架构调用相同的 HiMoEPolicy、checkpoint、归一化、checkpoint-right 腕部布局、DeviceBoundPolicy、每卡单执行线程与推理后的 heap trim。主要变化是七卡共享宿主进程，或每卡独立宿主进程。独立进程按 GPU UUID 绑定并核验只可见一个正确设备。

每卡两个客户端请求流，使用相同规则生成的合法观测与显式 flow noise，仅请求 Goal，开启真实 top-4 专家 ID/weight 返回。每个阶段发起请求 30 秒，再等待在途请求排空；不含模拟器、完整 HB 概率、报警快照或分支落盘。

| 架构 | 活跃 GPU 数 | 总 query/s | 平均 GPU-Util/卡 | 平均功耗/卡 |
|---|---:|---:|---:|---:|
| 原共享进程 | 7 | 1.959 | 6.18% | 129.67 W |
| 独立进程 | 1 | 1.675 | 34.55% | 144.67 W |
| 独立进程 | 2 | 3.615 | 38.62% | 151.00 W |
| 独立进程 | 7 | 13.198 | 40.50% | 153.19 W |

独立七卡共完成 418 次请求，含排空耗时 31.67 秒，各卡平均利用率 35.32%..42.64%。共享七卡完成 70 次，含排空耗时 35.73 秒。吞吐包含排空时间，利用率/功耗来自发起窗口内定时 nvidia-smi 采样，两者统计窗口略有不同。

该表是同一个 Goal 模型的短测，不是之前四套件轮询压测，也不是 Pro/Plus 完整 rollout 吞吐。单卡、双卡、七卡依次运行，时钟和温度波动可能影响短测值；不能由略高于线性的局部比例推断额外收益，也不能将 13.20 query/s 当作完整采集的保证。

## 2. 结果一致性

每张卡选两个固定输入，共 14 个不同 GPU/输入组合，压测前后各检查一次，共 **28 次配对核验**。原服务与独立服务的以下数组均逐项完全一致，最大绝对差为 0：

- `(10,7)` 动作。
- `(10,8,10,4)` HB top-4 专家 ID 和实际合并权重。
- HB 层号及显式 flow noise 哈希。

同时核对 checkpoint 哈希、严格加载记录、归一化哈希、腕部预处理布局和 upstream 工作树身份。测试不采 hidden，不改权重、报警参数、denoise 次数或 top-k。

这是固定输入的推理等价性检查，不替代 Pro/Plus 全轨迹恢复和 C0/C1/T1 的验证。记录见 [results.json](design/process_scaling_20260908/results.json)。

## 3. 具体怎样提高

**第一步：将正式模型服务切换为每卡独立进程。** 保留每卡五个模型和现有物理卡端口映射，每个进程内的模型共享一个有界执行队列。这个方案的进程隔离收益已经在单 Goal 对照中实测；完整五模型 bundle 的混合负载仍需验证。当前旧矩阵仍在运行，未在本轮进行服务迁移。

各卡目前 nvidia-smi 显示约 95,000 MiB（92.8 GiB）显存占用，剩余显存足以容纳本次临时单模型对照，但不足以在旧矩阵旁再加载完整五模型副本。正式迁移需要排空请求、切换服务实例并重新核验全部端点。

**第二步：用环境并发和异步写入持续供给模型。** 每卡先用两个独立环境 worker，再比较 1/2/4 个 worker 的真实吞吐。模型推理、环境 step/渲染、特征计算和写入尽量流水执行，队列保持有界。主轨迹仍先完整结束，其分支才可执行；按 main/event 标识调度，chunk 不重置报警历史。这里的环境 worker 数是下一轮测试起点，不是已测出的最优配置。

**第三步：做同 checkpoint 的小批量推理。** 测试 batch=2/4，收集不同环境同一时刻可执行的请求，分别保存 flow noise、query 编号和路由轴。当前 [hook 明确限制 batch=1](/home/jovyan/work/himoe-vla/himoe-libero-wrist-fix/src/himoe_libero_bridge/policies.py:330)，策略入口也会自动增加批次维度并只返回第一个样本；因此不能仅增加客户端或改一个 batch 参数。批量化需要同步改输入拆装与 hook，并验证动作、路由和逐轨迹报警一致性。

**第四步：减少 MoE 中的 CPU 同步和小算子开销。** 目前 HB 与 AS 的 [专家分派代码](/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/src/moevla/models/modeling_moe.py:233) 都执行 `bincount().cpu().numpy().cumsum()`，再用 Python 循环逐个专家计算；[denoise 循环](/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/src/moevla/models/moevla.py:432) 也用 GPU 标量作 Python 条件。应先采集 CPU/CUDA profiler 数据，量化同步、expert MLP 和 launch 耗时，再验证 GPU 端分派、分组矩阵乘以及适合固定形状部分的图捕获。不要直接对含 CPU NumPy 和数据相关控制流的整段推理承诺图捕获收益。

本实验没有分别剖析 GIL、算子发射与进程共享 heap-trim 锁的贡献，因此只确认当前共享进程结构限制了多卡扩展。独立进程达到约 40% 后，batch=1 的小算子和同步路径仍是值得继续测量的部分；**80%..100% 尚无实测证据**。

## 4. 代码与复现

- [serve_isolated_model.py](serve_isolated_model.py)：按允许的物理 GPU/UUID 绑定独立进程，可选择加载的模型；复用已有严格加载器和执行队列。
- [probe_process_scaling.py](probe_process_scaling.py)：检查空闲显存与备用端口，临时启动单 Goal 副本，检查输出一致性，测量共享/独立进程，最后回收自己创建的进程。
- [结果和逐卡日志](design/process_scaling_20260908)：包含请求延迟、GPU 采样、模型进程身份和配对输出误差。

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  probe_process_scaling.py --gpus 0,1,2,3,4,5,7 --seconds 30 --clients 2 \
  --output design/process_scaling_reproduction
```

输出目录必须尚不存在，原矩阵端口必须可用。脚本检查每张测试卡至少剩余 26,000 MiB，并检查备用端口空闲。显式请求 GPU 6 会被拒绝。本次结束时七个临时 PID 均已退出，显存回到原来的约 95,000 MiB/卡，原预加载矩阵继续运行。
