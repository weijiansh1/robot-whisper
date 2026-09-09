# Long 小批次采集与功率检查

后续 [MPS 单卡对照](MPS_POWER_RESULT.zh.md) 已把推理功耗提高到平均约 290–296 W，8 路达到约 7.95 query/s。下文保留本次非 MPS 七卡真实采集的结果；MPS 尚未验证七卡完整采集。

2026-09-08。已完成 **48 条 Long 主轨迹与 48 条完整 C0 后缀**，全部通过独立审计。七卡双推理进程的真实采集吞吐为 **13.07 query/s**；全程平均每卡 GPU-Util 为 **70.85%**，平均功耗为 **163.32 W**。这属于高忙碌率采集，尚不能称为算力满载。

## 采集结果

| 项目 | Pro | Plus | 合计 |
|---|---:|---:|---:|
| Long 唯一变体 | 20 | 28 | 48 |
| 主轨迹 query | 853 | 954 | 1,807 |
| C0 query | 440 | 627 | 1,067 |
| 主轨迹成功数 | 7 | 19 | 26 |
| 首次 v7 报警并保存快照 | 12 | 7 | 19 |
| 完整 C0 通过 | 20 | 28 | 48 |

覆盖 Pro 的 5 类、Plus 的 7 类扰动，每类 4 个变体，包含 9 个 Long 基础任务。每个变体使用 init=0 与由 variant_id 确定的固定 seed，成功即结束，否则运行完整 520 action steps。实际主轨迹共 17,959 steps。样本按类别选取，不能当作总体成功率、误报率或召回率评估，也未用于重校准。

本轮是原生主轨迹与 C0 恢复预检：首次报警前的完整状态持久化，主轨迹继续正常运行；main_complete 提交后，才从首次报警快照恢复并执行 C0。没有报警时，从预检 q=5 或 q=0 检查恢复。**C1/T1 干预尚未执行。** 正式 14,030 条清单保持 planned 状态，本批数据独立保存。

主轨迹与 C0 每 8 query 提交一个 chunk，最多 2 个待写块。记录完整 HB 概率、实际 native/effective 专家 ID 和合并权重、动作、显式噪声、输入哈希、仿真状态、报警与耗时；不保存 hidden。HB 概率形状为 `(8,10,11,32)`，专家 ID/权重为 `(8,10,11,4)`。AS 路由与正式分支前缀引用尚未接入。

冻结参数 SHA-256 为 `f4e65d359411309756b008423d14d8f3e5d6bf604ee71b63ae8a7e510c5f8216`，v7 profile 为 `a92c9d1103487ddf62e4b369c7f23b6637127730dffa8780f54423cf580e3054`。48 条的离线报警复算均一致。C0 全后缀的观测输入哈希、噪声、动作、全部 HB 数组、报警、物理状态与终止结果逐项完全一致。

证据：[采集汇总](design/collection_long_highload_20260908/summary.json)、[独立审计](design/collection_long_highload_audit_20260908.json)、[性能及跨负载一致性](design/collection_long_highload_analysis_20260908.json)。先行的 2 条试跑是这 48 个变体的重复验证，不另计新增样本；它们在小负载与七卡负载下的全部非耗时字段也完全一致。

## 七卡执行方式

- 推理卡为物理 GPU 0/1/2/3/4/5/7，每卡 2 个独立进程，每进程 batch=1、独立 hook 与单执行线程。
- 每卡只新增一份只读 Long CUDA 权重，再用 PyTorch spawn/CUDA IPC 共享给同卡进程；原 35 个预加载模型继续保留。可变 buffer 单独复制，退出时先回收共享权重的使用进程，再退出权重所有者。采用 [PyTorch 2.6 的 CUDA 多进程约束](https://docs.pytorch.org/docs/2.6/notes/multiprocessing.html)。
- 每卡 4 个环境任务位，最多 28 条轨迹同时运行，共用持续任务队列。Pro/Plus 分属独立常驻 Python 环境进程，复用连接和 suite 缓存；12 条任务复用了已经执行过任务的进程。
- 渲染池为 GPU 0/3/4/5/7，继续绕开此前 GPU 1/2 的 EGL 恢复问题。GPU 6 未参与模型加载、推理、渲染或压测。
- Plus 噪声使用已验证的逐像素及 RNG 等价加速。所有在线报警仍使用同一冻结 v7 profile。

| 实测指标 | 数值 |
|---|---:|
| 采集耗时，含环境启动、主轨迹、C0、收尾，不含模型加载 | 219.92 秒 |
| 主轨迹+C0 吞吐 | 13.07 query/s |
| 全程七卡平均 GPU-Util / 功耗 | 70.85% / 163.32 W |
| 每卡至少两条任务运行期间的平均 GPU-Util / 功耗 | 88.15% / 171.36 W |
| 单卡功耗采样最大值 | 194.49 W |
| 单卡总显存采样峰值，含原有模型 | 118,900 MiB |
| GPU 温度采样最大值 | 48 C |

“至少两条任务”区间包括环境初始化、仿真和恢复，不能作为纯推理或 SM 利用率。任务队列耗尽后，各卡结束时间不同，收尾阶段拉低全程均值。此前 Goal 批次的任务长度、失败重试和噪声路径不同，不能直接与本表相除作为优化倍数。

## 为什么功率仍低

采集期间检查到七卡均为 P0，SM 时钟为标称最高的 1980 MHz，显存时钟为 3199 MHz；没有功率、温度或外部电源限频。功率上限已经是 500 W。本次没有修改功率或频率设置。

GPU-Util 表示采样时间内至少一个 kernel 正在执行的时间比例，不是计算单元或 Tensor Core 的饱和程度，见 [NVIDIA 指标定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。功耗除以 500 W 也不能换算成 FLOPs 利用率。

实际抽查本批前 8 条主轨迹中的 314 query、25,120 次 HB gate 调用：每次平均激活 19.70 个专家，每个活跃专家平均只收到 **2.23 个 token**，中位数 2，最大 11。上游 `modeling_moe.py:230` 的 HB 推理会把专家计数搬回 CPU，然后用 Python 逐专家执行 MLP、索引和 scatter。已有同架构 Goal profiler 还记录到每 query 约 3.9 万次 CUDA kernel 发射。这些证据支持小矩阵和算子发射开销是重点优化方向；当前尚未取得 Long 的 SM/Tensor Core 性能计数，不能据此量化各瓶颈占比。

## 四路并发补测

在物理 GPU 0 上让 4 个共享权重的进程始终常驻，按 `1,2,4,4,2,1` 的顺序改变活跃进程数，每段发起请求 20 秒后排空。使用本批 4 条 Plus Long 轨迹中的 8 个真实观测与原始 flow noise，开启完整 HB capture，所有请求仍为 batch=1。此项没有运行模拟器，也没有新增主轨迹。

| 活跃进程数 | query/s | 平均功耗 | 平均 GPU-Util |
|---|---:|---:|---:|
| 1 | 1.680 | 152.64 W | 41.66% |
| 2 | 2.668 | 174.34 W | 98.51% |
| 4 | 3.265 | 184.77 W | 99.46% |

四路比两路吞吐提高 **22.37%**，功耗增加约 10.44 W，仍远低于 500 W。318 次计时响应的动作与全部 HB 数组均和原采集记录完全一致，前后还与原 Long 服务进行了并发动作/top-4 核验。这里是 8 个观测的重复对照，不是 318 个独立场景。

这证明更高并发有额外收益，也说明单靠增加进程难以保证高功率。下一步优先验证专家分组执行、减少 CPU 往返和 kernel 发射；每种改动都要检查动作、路由、冻结报警与 C0 完整后缀。当前正式采集入口仍限制为每卡 1/2 路，四路只完成了单卡回放验证，未声称七卡四路完整采集已通过。

证据：[并发与功耗对照](design/long_concurrency_power_20260908/results.json)、[功耗汇总与专家 token 统计](design/long_concurrency_power_analysis_20260908.json)、[可重复探针](probe_long_concurrency.py)。四路服务占用 20 个端口的跨度，不能按当前七卡双路的 10 端口间隔直接部署。

## 存储与结束状态

48 个有效主轨迹+C0 目录共 **242,109,446 bytes，约 230.89 MiB**，平均 **4.81 MiB/对**。当前剩余约 39.64 GiB，保留 8 GiB 后，按本批 Long 长度分布约还能容纳 **6,734 对**。这只是当前 HB+主轨迹+C0 schema 的估计，不包括 C1/T1、视频、多报警事件等额外数据，不能作为正式全量实验容量。

采集和并发探针的临时模型已退出，显存回到原有约 95,000 MiB/卡。原 35 个端点身份、checkpoint 加载记录和归一化检查通过，仍属于 PID 28531；这次结束检查没有重新对全部 35 模型发起推理。详见 [结束检查](design/long_highload_final_health_20260908.json)。

复现七卡小批次，输出目录和端口须空闲：

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  run_collection_preflight.py --model long --gpus 0,1,2,3,4,5,7 \
  --render-gpus 0,3,4,5,7 --replicas 2 --workers-per-gpu 4 \
  --per-category 4 --port-base 9900 --output design/collection_long_repeat
```

输出预算 4 GiB、磁盘保留 8 GiB、单任务超时 900 秒。复现四路并发对照：

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  probe_long_concurrency.py --gpu 0 --seconds 20 \
  --collection design/collection_long_highload_20260908 \
  --output design/long_concurrency_repeat
```
