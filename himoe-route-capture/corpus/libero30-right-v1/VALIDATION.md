# libero30-right-v1 完整性校验（P5）

校验日期：2026-08-14（采集完成于 2026-08-13）
工具：`validate_corpus.py --root corpus/libero30-right-v1 --episodes 50`
机器可读结果：[`REPORT.json`](REPORT.json)

## 结论

**通过。** 30/30 计划任务全部存在、0 失败、0 缺失，1500 集齐全。

## 自带的完整性校验：成功率对表 Section 4.1

这批语料用的是 `checkpoint-right` 操作点、每任务全部 50 个 init state，
正是论文 Section 4.1 的官方评测协议，所以成功率本身就是一道校验——
对不上就说明采集配置错了。

| suite | 语料 | 已发表的正式 500 局 | 差 |
|---|---:|---:|---:|
| goal | 490/500 = **98.0%** | 97.8% | +0.2 pp |
| spatial | 471/500 = **94.2%** | 94.2% | **±0.0 pp** |
| object | 484/500 = **96.8%** | 96.6% | +0.2 pp |

三套各差 ≤1 集。spatial 逐集相同。差异与
[`capture-not-bit-reproducible`] 一致：flow 噪声流不同（本批固定
`flow_noise_seed=1000`，正式复评用的是服务端内部噪声流），推理本身是确定性的，
但噪声流一换，边界集的成败就可能翻转。

**这些数字不能与正式复评并表做配对比较**——两者操作点相同但噪声流不同，
只能作为同一操作点下的独立复现。

## 结构性校验

- 逐任务 `control_steps == sum(inference_calls)`：全部通过
- 每集 `action_steps <= max_steps`（goal 300 / spatial 220 / object 280）：全部通过
- `INDEX.csv`：1500 行，每集一行
- 总量：18,384 控制步，781,509,910 字节（0.78 GB）

## 一处需要说明的时序

P4 采集在 08-13 完成（30 个任务的 `meta.json` 均为 `status: complete`），
但 P5 当时没有收尾：`INDEX.csv` 只写了 100 行（6.7%），`MANIFEST.totals` 的
`bytes` / `control_steps` 仍是 `null`。本次补跑校验并重建了两者。

**采集数据本身自始至终是完整的**，缺的只是索引与统计。任何在
08-13 到 08-14 之间读过 `INDEX.csv` 的分析都只覆盖了 goal 的前两个任务，
需要重跑。

[`capture-not-bit-reproducible`]: ../../../himoe-vla-moe-routing-64draw-2026-08-09.md
