# 预加载模型与七卡运行检查

检查日期：2026-09-08，主要推理压测在 01:24-01:28 UTC 完成。

**模型已经全部加载且能推理，但当前服务结构没有把七张 GPU 跑满。** 允许卡为物理 GPU `0,1,2,3,4,5,7`；物理 GPU 6 未用于本次推理、加载或渲染，检查后仍为 1 MiB / 0% 利用率。

## 1. 模型是否齐全

预加载进程为 `online-servers/serve_model_matrix.py`，PID 28531。它从 01:13 起顺序加载，01:23 开放全部 **35 个端口**。最初部分卡显存很低对应“尚未轮到加载”，不是 checkpoint 缺失。

| 物理 GPU | Goal | Spatial | Object | Long | CALVIN | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 8800 | 8801 | 8802 | 8803 | 8804 | 5/5 通过 |
| 1 | 8810 | 8811 | 8812 | 8813 | 8814 | 5/5 通过 |
| 2 | 8820 | 8821 | 8822 | 8823 | 8824 | 5/5 通过 |
| 3 | 8830 | 8831 | 8832 | 8833 | 8834 | 5/5 通过 |
| 4 | 8840 | 8841 | 8842 | 8843 | 8844 | 5/5 通过 |
| 5 | 8850 | 8851 | 8852 | 8853 | 8854 | 5/5 通过 |
| 6 | 排除 | 排除 | 排除 | 排除 | 排除 | 不调度 |
| 7 | 8870 | 8871 | 8872 | 8873 | 8874 | 5/5 通过 |

每个端口都核验了物理/逻辑 GPU、模型名、checkpoint 身份、严格加载记录、归一化文件哈希和真实推理输出。LIBERO 输出 `(10,7)`，CALVIN 输出 `(10,8)`，均为有限数值。28 个 LIBERO 端口还实际返回并验证了 `(10,8,10,4)` 的 top-4 专家 ID / combine weights。

五份独立 checkpoint 每份均为 **8,138,322,389 bytes**；每个实例严格加载 **1,581 个 state_dict 键**。完整权重 SHA-256 由服务构造器在加载时逐字节计算，并与固定 suite 规格比较；本次检查再次核对该加载元数据、文件大小、stats 哈希和实际推理，没有另行重读五份大文件做重复哈希。

| 模型 | SHA-256 | 归一化资产 |
|---|---|---|
| Goal | `98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953` | `libero_goal_no_noops` |
| Spatial | `1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137` | `libero_spatial_no_noops` |
| Object | `f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa` | `libero_object_no_noops` |
| Long | `cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256` | `libero_10_no_noops` |
| CALVIN | `1ea466b4f8aa3561f94f8a120d06adb26f9438548e0513f6c12b3ab3e3c9ebfa` | `calvin_d_joint` |

Pro/Plus 的四个基础 suite 所需 HiMoE checkpoint 已齐全，可以复用对应 Goal/Spatial/Object/Long 权重；不需要为了开始这些 OOD 评估另外准备专属微调权重。后续加载工作已完成两个 benchmark 的资源安装、200 / 10,030 个注册条目核验及 48 个组合的真实模拟器短程验证，见 [加载报告](benchmarks/README.zh.md)。报警状态恢复协议和控制采集接口仍须按实验方案实现与预检。

当前五模型常驻约 85.75 GiB/卡，压测后 CUDA 缓存使 `nvidia-smi` 读数约 94,908-95,028 MiB/卡。硬件总显存 143,771 MiB/卡，功率上限 500 W。cgroup 当前限制为 **512 GiB 内存、96 个 CPU 核配额**；启动脚本注释中的旧 8 GiB 限制已经不对应当前环境。

## 2. 满载压测结果

使用真实预加载模型、合成的合法形状观测、固定 flow noise，四个 LIBERO 模型轮询。每档持续约 45 秒并等待队列排空；吞吐包含排空时间。没有运行模拟器、没有采 hidden、没有路由干预；健康检查开启 top-4 返回，吞吐阶段关闭路由返回。因此这些数值是模型服务吞吐，不是 Pro/Plus rollout 吞吐。

| 请求流/卡 | 七卡总请求流 | 七卡总 query/s | 平均 GPU 利用率 | 平均功耗/卡 | 平均单请求耗时 |
|---|---:|---:|---:|---:|---:|
| 1 | 7 | 1.978 | 6.15% | 128.95 W | 3.52 s |
| 2 | 14 | 1.999 | 6.15% | 128.94 W | 6.74 s |
| 4 | 28 | 1.976 | 6.09% | 129.14 W | 12.82 s |

额外仅对 GPU 0 做相同模型轮询负载：**1.848 query/s、39.72% 平均利用率、148.72 W、0.541 秒/请求**。七卡合计只达到单卡约 1.07 倍吞吐。增加请求流并未提高当前矩阵的有效吞吐。

七卡压测中各卡平均功耗约 121-142 W，最高单次 GPU 利用率 25%，最高观测温度 43°C，没有 OOM 事件。本次不能把“模型已预加载”写成“可以高功率跑满七卡”，也没有修改功率限制来制造高功耗。

完整数据：[七卡压测](design/preload_probe.json)、[单卡对照](design/preload_probe_single_gpu.json)。观察量仅代表这次短时压测，持续运行和真实环境吞吐仍需单独验证。

## 3. 性能瓶颈与具体下一步

当前 [serve_model_matrix.py](../online-servers/serve_model_matrix.py) 将 35 个模型放在一个 Python 进程中，每 GPU 创建 `ThreadPoolExecutor(max_workers=1)`，该 GPU 的五个端口共享一个 CUDA 工作线程。每次推理后还经过进程共享的 heap-trim 锁。GPU 之间同时争用宿主进程的 Python/算子发射路径。

单卡与七卡对照、增加客户端只增加延迟，共同指向共享进程的扩展性瓶颈。尚未做能够区分 GIL、算子发射和 heap-trim 各自占比的完整剖析，因此不把其中某一个因素当作已确认的唯一根因。

建议下一步在相同 checkpoint、动作数值和端口映射下，对以下部署进行 A/B 验证：

1. **每张允许卡一个独立推理进程**，复用已有 [serve_model_bundle.py](../online-servers/serve_model_bundle.py) 的五模型 bundle。七卡共七个进程，模型参数互相独立。当前 512 GiB 的宿主内存配额允许重新评估这个方案，不必继续受旧 8 GiB 限制下的单进程设计约束。
2. 每卡先接 2 个环境 worker，必要时增至 4 个，采用有界请求队列。35 个模型端口是模型选择入口，不能按 35 个独立 GPU 执行队列调度。
3. 先验证单卡/七卡吞吐是否能随进程数扩展，再评估同 suite 微批处理或 CUDA graph 等进一步优化；这些改动必须验证动作、路由、随机数和报警时序的一致性。即便进程拆分有效，也不能预先承诺 batch=1 的模型会达到 500 W 或 100% 利用率。
4. GPU 6 的限制覆盖推理、模型加载和 EGL 渲染。当前矩阵中 `cuda:6` 是物理 GPU 7；改成每卡进程后各进程通常只有 `cuda:0`，必须依 metadata/UUID 确认，不能混用逻辑与物理编号。

本次没有重启、替换或卸载用户预加载的模型服务；压测客户端已正常结束，35 个服务端口保留。按卡拆进程属于下一步部署改造，目前没有将其标记为已实现或已验证。

## 4. MoE-Control 接口缺口

现有预加载 LIBERO 服务支持 `routing/capture`，返回动作 token 的 top-4 IDs 和实际 combine weights，shape 为 `(denoise=10, layer=8, action_token=10, k=4)`。它没有返回：

- v7 需要的完整 HB 原生概率 `(8,10,11,32)`。
- state token、AS 路由与完整的请求级归档字段。
- T1 所需的按层/token/query 控制的路由干预接口。

因此模型已经就绪，但正式 v7 + MoE-Control 采集还不能直接使用当前普通推理端口完成。需要在复用模型参数的服务包装层接入完整 recorder 和干预协议，记录真实生效的合并权重，并保持 hidden 关闭。已有独立 `serve_with_recorder.py` 的能力可以作为改造参考，不应再为每个环境 worker 重复加载一份模型。

## 5. 对采集预算的影响

原 [实验方案](MOE_CONTROL_EXPERIMENT_PLAN.zh.md) 中 4 query/s 是假设。当前矩阵连纯推理压测都只有约 2 query/s，不能继续把 4 当作实测依据。

以整集群 2 query/s 重算，30% 报警、主轨迹 25 query、分支 15 query 的条件下，原 3,800 条首轮主轨迹需要约 **40.57 小时**；14,030 条主轨迹、最多 600 个展开状态的覆盖方案约 **69.39 小时**。存储仍分别约 13.30 / 24.28 GiB。真实模拟器、完整 recorder 和队列开销会影响吞吐，这些仍是条件预算。

复算文件：[当前矩阵预算](design/budget_current_matrix.json)。部署改造后，应按新的真实 rollout 吞吐再次更新。

## 6. 可重复检查

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python probe_preloaded_models.py
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python probe_preloaded_models.py --gpus 0 --load-seconds 30 --clients-per-gpu 1 --out /tmp/preload-single.json
```

[检查脚本](probe_preloaded_models.py) 在客户端将 `CUDA_VISIBLE_DEVICES` 置空，使用 WebSocket 调用已有服务，并硬性拒绝物理 GPU 6。已测试拒绝 `--gpus 6`。依赖路径优先使用 `himoe-libero-wrist-fix` 下含 Long 的 bridge，避免被 CALVIN 兼容副本覆盖。
