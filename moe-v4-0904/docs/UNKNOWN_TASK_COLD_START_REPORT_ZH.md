# 未知任务的 MoE-only 冷启动报警

## 结论

可以用单条 rollout 前四次 query 的 MoE 路由为未知任务设阈值，但不能
无损替代已知任务的 400 条无标签参考轨迹。严格留一任务实验的结果是：

| 模式 | 数据 | Recall | Precision | 正常轨迹 FPR |
|---|---|---:|---:|---:|
| 高精度冷启动（lock） | development 14,800 | 47.23% | 90.91% | 0.161% |
| 高精度冷启动（lock） | external 8B 15,600 | 44.50% | 85.67% | 0.279% |
| 平衡冷启动（dual） | development 14,800 | 58.52% | 86.10% | 0.321% |
| 平衡冷启动（dual） | external 8B 15,600 | 57.45% | 78.83% | 0.579% |
| 已知任务 v4（对照） | external 8B 15,600 | 72.70% | 83.50% | 0.539% |

这里的 risk 沿用主实验定义，即原 horizon 内的失败，包括再多执行十次 query
后变成成功的慢成功。没有用任务 ID、图像、动作、机器人状态或成败标签生成
报警；成败标签只在报警数组封存后用于计算上表。

## 单条 rollout 如何定阈值

### Lock 头：用前期变化率估计本轨迹尺度

前四次 query 提供三次相邻路由变化。令

```text
b = mean_{q=1..3}(median_layer Hellinger(route_q, route_{q-1}))
```

每个历史参考任务只用无标签路由得到正常尺度与 q75 lock cutoff 的比值
`r_j`。未知 rollout 使用参考库中 `r_j` 的保守 q10：

```text
lock_cutoff = b * higher_quantile({r_j}, 0.10)
```

这不是识别任务名称，而是把当前轨迹归一化到自己的早期路由速度。

### Instability 头：用绝对专家图样找相似计算原型

将前四次 query、最终去噪步、十个 action token 的专家概率取均值，保留八层
和 32 个专家，并作平方根 Hellinger 嵌入。它与 40 个无标签参考任务的路由
质心比较，取最近的 12 个参考原型；这些原型的 L5 q80 cutoff 的中位数就是
该 rollout 的 instability cutoff。

开发集的事后诊断显示，预测 cutoff 与被隐藏的真实任务 cutoff 在任务层面的
Pearson 相关约为 0.903，中位相对误差约为 6.0%。这说明早期绝对路由确实
包含任务/场景计算结构，但极端任务的阈值仍会被近邻中位数拉向总体中心。

## 严格未知任务协议

对每个被测任务，评估代码会从参考库中删除该任务的全部 400 条历史轨迹和
路由质心，再为它的每条 rollout 定阈值。任务字符串只用于实施这次离线
leave-one-task-out 审计；在线类 `UnknownTaskMonitor` 的接口不接收 task ID。

前四次 query 仅用于校准。lock 头最早仍需四个连续低流动窗口，instability
头需八个连续高流动窗口，因此校准不会推迟两头原有的最早合法报警时刻。

## 如何使用结果

未知任务第一次运行且更重视不打扰成功轨迹时，应先用 lock-only 高精度模式。
双头模式额外检出约 13 个百分点的失败，但外部精度降至 78.83%；新增误报
主要来自 instability 近邻阈值。因此当前不能声称未知任务已经复现已知任务
v4 的性能。

复现实验：

```bash
python moe-v4-0904/experiments/extract_early_route_fingerprints.py \
  --cohorts main extra
python moe-v4-0904/experiments/extract_early_route_fingerprints.py \
  --cohorts external \
  --output moe-v4-0904/results/unknown_task/early_route_fingerprints_external.npz
python moe-v4-0904/experiments/evaluate_unknown_task_cold_start.py
```

在线入口位于 `method/unknown_task_monitor.py`。加载
`results/unknown_task/cold_start_v1/cold_start_reference.npz` 后，每次 query
只需调用一次 `update(hb_router_probs)`。
