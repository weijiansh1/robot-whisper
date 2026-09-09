# 真实采集与资源利用

后续已完成 [48 条 Long 七卡高并发试采样及功率检查](LONG_HIGHLOAD_TRIAL.zh.md)：全部主轨迹+C0 通过，13.07 query/s。共享只读权重解决了双推理进程的额外权重显存问题，原 35 模型保留。下文为此前 Goal 批次的历史结果与当时限制。

2026-09-08。本轮把独立推理进程接到了真实 Pro/Plus Goal 主轨迹、冻结报警、持久快照与 C0 后缀，并优化了导致 GPU 等待的 CPU 图像扰动。全部推理保持 batch=1；物理 GPU 6 不参与加载、推理或渲染；不保存 hidden，不修改报警参数。原 35 个预加载模型服务保留。

## 已完成的数据

| 项目 | 有效去重结果 |
|---|---:|
| Pro / Plus Goal 变体 | 10 / 14 |
| 通过审计的主轨迹与 C0 配对 | 24 / 24 |
| 主轨迹 query / C0 query | 433 / 265 |
| 主轨迹实际 action step | 4,276 |
| 主轨迹成功数 | 15 |
| 有首次 v7 报警且持久化快照的主轨迹 | 4 |
| 有效轨迹目录总大小 | 70,155,620 bytes，约 66.91 MiB |

这是每个 benchmark/category 两个 Goal 变体、每个变体 init=0 与固定 seed 的预检。24 个样本不足以估计总体成功率、误报率或召回率，不用于重校准报警。正式 14,030 条清单保持原状态。另一次噪声优化对照重跑同一个变体，不重复计为新轨迹。

[逐块审计](design/collection_preflight_audit_20260908.json) 验证了校验和、query/step 连续性、完整 HB 形状、实际 top-4 ID/weight、无 hidden、冻结 v7 离线复算、报警点 RNG 和状态，以及主轨迹提交后的完整 C0。每个有效 C0 后缀的输入哈希、显式噪声、动作、HB 概率、native/effective ID 与实际权重、物理状态和终止结果完全一致。

## 有效的性能改进

Plus 的 `glass_blur` 在 CPU 上逐像素调用随机数并更新图像，成为一条轨迹的主要耗时。新路径批量生成相同的 NumPy 随机整数，用现有 Numba 执行原顺序的整数循环；保留原始 Gaussian 算法及 NumPy view 的赋值语义，没有改变扰动规则。

| 检查 | 原实现 | 加速后 |
|---|---:|---:|
| 20 个 224x224 RGB 输入，覆盖 10 个强度 | 12.673 秒 | 0.127 秒 |
| 同一真实噪声轨迹的主段仿真累计耗时 | 100.570 秒 | 3.797 秒 |
| 同一轨迹的主段 + C0 worker 耗时 | 201.001 秒 | 26.820 秒 |

20 组函数对照逐像素、逐项 RNG 状态完全相同。真实对照包含 17 次主段 query、12 次 C0 query、169 个主段 action step；除耗时之外，原有全部落盘字段一致，最终都成功。主段仿真约快 26.5 倍；该轨迹整体约快 7.49 倍。整体对照的原运行与优化运行并发负载不同，因此不能把全部 7.49 倍都归因于该函数；函数对照和主段仿真耗时提供了更直接的证据。JIT 热身不计入函数对照的计时，启动时间包含在调度器的整批耗时中。

证据：[函数对照](design/collection_noise_exactness_20260908.json)、[整条轨迹对照](design/collection_noise_rollout_comparison_20260908.json)、[优化后完整采集](design/collection_noise_rollout_fast_20260908/summary.json)。快路径仅在安装的 Plus 源码 SHA-256 与已验证版本相同时启用，源码变化会拒绝运行。

## 实际利用率与限制

| 运行 | 工作内容 | query/s | 平均每卡 GPU-Util |
|---|---|---:|---:|
| 首次七卡预检 | 24 个任务，16 个通过，8 个相机失败 | 2.17 | 12.79% |
| 七卡定点补跑 | 8 条主轨迹完成，6 个 C0 通过、2 个不一致 | 4.88 | 20.69% |
| 分离渲染的两卡补跑 | 补齐上述 2 个有效配对 | 1.44 | 17.52% |

这些吞吐包含环境启动、完整主轨迹和已经执行的 C0 请求，分母为模型加载后开始采集至最后任务结束；失败重试和未通过 C0 的工作不能作为有效配对产量。各行任务、并发量和长度不同，不能相除当作方案加速比。最初一批在其他任务完成后，长时间等待一个 CPU 噪声任务；优化后尚未重测七卡长队列稳态吞吐。

此前约 98% 是单卡双推理进程的短测结果。当前保留原 35 个模型，新采集器每卡另起一个 Goal 实例，只有新增实例支持完整 HB；不能直接把旧端点作为第二个完整采集实例。完整 HB 推理实测峰值 allocated 为 24,029.95 MiB；把新增进程限制到 21,000 MiB 时无法完成推理。原有每卡约 95,000 MiB 加上两个额外完整实例，再加 CUDA/context/render 开销，当前布局没有足够余量。没有调整功率上限，也没有用空负载维持利用率数字。

[内存限额检查](design/collection_memory_limit_20260908.json) 只说明该限额不足，不证明所有内存优化方案都不可行。后续提高 GPU 吞吐应评估把原矩阵迁为支持完整 hook 的独立进程组，再测试两个活跃推理进程的七卡扩展，并继续保持 batch=1。

## 已修复和隔离的问题

1. 快照补齐 Python/NumPy/policy RNG、实际观测、observable 缓存与计时、控制器数值状态、gripper.current_action 和额外 actuator 状态。仅补旧快照的 RNG 仍会在夹爪后续控制中分叉。
2. 原始七卡运行中，GPU 1/2 的 EGL 出现空图；单环境检查可以出图，但后续真实 C0 仍有输入不一致。设备 UUID 与物理卡号匹配，没有误用 GPU 6。模型继续在 GPU 1/2 推理，将两条任务的渲染移到 GPU 0/3 后，C0 全后缀通过。底层 EGL 差异原因尚未定位。
3. 默认渲染池从所选卡中排除 GPU 1/2，同卡环境初始化加锁。仍允许显式指定渲染卡作诊断。推理 GPU 与渲染 GPU 分别记入结果；EGL UUID 在创建环境前验证。UUID 查询接口采用 [Khronos EGL 扩展定义](https://raw.githubusercontent.com/KhronosGroup/EGL-Registry/main/api/EGL/eglext.h)。
4. 采集记录为 8 query/chunk，最多 2 个待写块；临时文件 fsync 后原子重命名，manifest 带 SHA-256。失败任务保留，重试写入新目录。主轨迹成功或达到 300 步后提交 main_complete，之后才从首个报警点或无报警时的预检点执行 C0。
5. 第一版运行 summary 的 completed 只表示任务进程完成，可能包含 C0 不一致，不能当作通过。新调度器将这种情况标为 fidelity_failed，并支持只重试未通过任务。有效样本以独立 audit 为准。

## 容量估算

本批主段压缩后平均约 55.21 KiB/query；主段+C0、快照、初末相机图和元数据平均约 2.79 MiB/变体。审计时工作盘剩余约 40.06 GiB，预留 8 GiB 后，按这一批 Goal 分布约可容纳 11,777 对。

同样的均值下，14,030 对约需 38.20 GiB，超过预留空间后的预算。该估计不是全套实验容量：其他 suite 的长度、C1/T1 分支、每条轨迹多个报警事件、视频或逐帧图像都会改变体积。当前只留初始双相机和最终主相机 PNG，没有逐帧视频；正式存储预算应在完整分支 schema 验证后更新。

## 运行入口

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  run_collection_preflight.py --gpus 0,1,2,3,4,5,7 \
  --render-gpus 0,3,4,5,7 --workers-per-gpu 2 --per-category 2 \
  --port-base 9700 --output design/collection_next_preflight
```

默认启用已验证的噪声快路径，可用 `--no-exact-noise-fastpath` 做原实现对照；`--retry-from <目录>` 只补未完成或 C0 未通过的任务，`--variants <ID...>` 用于指定变体。输出目录必须是新目录，端口需空闲。单任务默认 900 秒超时，输出预算 4 GiB、磁盘保留 8 GiB，达到限制停止并清理本轮进程；不会终止原模型矩阵。

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  audit_collection_preflight.py \
  --runs design/collection_preflight_20260908 \
         design/collection_preflight_retry2_20260908 \
         design/collection_preflight_render_split_20260908 \
  --output design/collection_preflight_audit_repeat.json
```

当前实现仅为原生 Goal 采集和 C0 验证；C1/T1 干预、其他 suite 和正式清单调度仍需实现。结束状态见 [资源与原模型检查](design/collection_final_health_20260908.json)。
