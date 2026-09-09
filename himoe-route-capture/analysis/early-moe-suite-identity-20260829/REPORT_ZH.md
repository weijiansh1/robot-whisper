# q0 MoE 的 LIBERO suite 类别比较

## 这次比较的是什么

主比较不是各 suite 内成功对失败，而是 Goal、Object、Spatial 三类任务的 q0 路由几何。
每个任务固定 `flow_noise_seed=1000`，并只取 Long 数据中同样的 16 个初态编号；
先把每个任务平均成一个点，再按任务留一，避免把 16 个初态伪装成 16 个独立任务。
Long 只有 `two_moka_pots` 一个任务，因此只投影到三类坐标系中，不参与四类统计检验。

## 结果一：类别从第一次去噪就能读出来

| q0 去噪步 | 留一任务分类正确率 | Long 最靠近 |
|---:|---:|---|
| d0 | 30/30 = 100.0% | Goal |
| d1 | 30/30 = 100.0% | Goal |
| d2 | 30/30 = 100.0% | Goal |
| d3 | 29/30 = 96.7% | Goal |
| d4 | 29/30 = 96.7% | Goal |
| d5 | 29/30 = 96.7% | Goal |
| d6 | 29/30 = 96.7% | Goal |
| d7 | 29/30 = 96.7% | Goal |
| d8 | 29/30 = 96.7% | Goal |
| d9 | 27/30 = 90.0% | Goal |

最强的单个可对齐坐标是 `q0|action|front_2_5|top4|d1`：suite 标签解释了任务间 99.7% 的方差，10,000 次任务标签置换后的 118 格 max-T `p=0.0001`。
所以 q0 MoE 确实带有非常强的 suite/模型类别信息，而且 d0 已经存在，不需要等到 d5。

## 结果二：这个类别信息不是一把难度尺

| suite | 匹配样本超时率 | q0 back/top1/d5 |
|---|---:|---:|
| Goal | 3/160 = 1.88% | 0.036091 |
| Object | 3/160 = 1.88% | 0.039222 |
| Spatial | 12/160 = 7.50% | 0.040609 |
| Long | 6/16 = 37.50% | 0.037116 |

难度顺序是 Goal/Object < Spatial << Long；但 d5 集中度顺序是 Goal < Long < Object < Spatial。
Long 在 10 个去噪步的多指标距离里也始终最靠近 Goal，而不是形成一个‘最难’端点。
因此跨 suite 的绝对路由值主要是类别/模型基线，不能直接解释为难度。

## 为什么不能把四类做成干净的难度实验

- 四个 suite 使用四个不同 checkpoint；expert 编号没有可靠的跨 checkpoint 对齐，所以这里只比较熵、top-k mass、token dispersion 等置换不变量。
- Goal/Object/Spatial 是 `checkpoint-right`，唯一 Long 路由是 `paper-right`。
- Long 只有 1/10 个任务有路由，无法估计 Long suite 的任务间变化。

checkpoint SHA-256 前 12 位：Goal `98ee29d09d18`，Object `f9c5661533d2`，Spatial `1029d0827030`，Long `cdc2b21f9ef6`。

## 结论

第一次 chunk 的 MoE 能非常清楚地区分 Goal/Object/Spatial，并把当前 Long 任务放在更接近 Goal 的路由几何区域。
但这说明的是任务族/独立 checkpoint 的编码，不是模型已经算出了任务难度。
此前 Long 内部 q0 d5 与初态失败率相关，仍然是同一 checkpoint 内的相对难度信号；
它不能用跨 suite 的绝对数值来解释。
