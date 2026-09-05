# MoE 在线风险预警 v5 迭代报告

## 结论

本轮没有用单一候选覆盖 v4，而是得到三个有明确用途的运行档：

| 运行档 | 外部 TP / FP | 风险召回 | early-4 召回 | early-8 召回 | 精度 | 非风险 FPR |
|---|---:|---:|---:|---:|---:|---:|
| v4 原始档 | 410 / 81 | 72.70% | 58.51% | 40.60% | 83.50% | 0.539% |
| long guard 高精度档 | 391 / 34 | 69.33% | 55.67% | 37.41% | 92.00% | 0.226% |
| lead guard 早报档 | 402 / 73 | 71.28% | 61.35% | 43.79% | 84.63% | 0.486% |

`long_guard_v5` 适合误触发代价高的部署；`lead_guard_v5` 适合至少需要
四个 query 干预余量的部署。v4 仍保留为原始总召回基线。

这些外部结果不是新的盲测。external 8B 在 v3/v4 及本轮探索中都已经被
查看，下一步必须用新 seed 或新任务作一次不可回看的确认实验。

## 1. 完整 sweep 恢复

重新运行开发集 layerwise sweep 后，`candidate_metrics.csv` 恢复为完整的
13,440 行候选，而不是先前残留的截断结果。候选生成不读取 external 8B
outcome；同任务阈值按 initial state 留出全部 8 个 seed，用其余 392 条轨迹
校准。

完整 sweep 暴露出两个重要事实：低 mobility 仍是主要信号，但更早报警并
不要求固定使用全层中位数；单层 L12/L14/L15 在不同误报预算下能形成更好的
前沿。因此后续选择都按最终 OR 报警重新计分，没有拿单头指标代替最终系统。

## 2. 精确持续统计量校准：否决

尝试把阈值直接校准到实际部署的统计量：

```text
trajectory score = max_query persistent_score(query)
```

225 组双头 quantile 组合中，没有一组同时满足开发集整体 FPR、suite FPR、
task FPR 和 75% precision 的约束；precision 约束通过数为 0。最保守的
max-reference 双头控制在开发集仅有 13.14% 召回、64.65% 精度，在外部集仅有
6.56% 召回、46.84% 精度。

原因不是实现错误，而是无标签 reference 本身包含风险轨迹。风险占比约 3%，
它们的持续极值进入 reference 上尾后，直接极值校准会把需要检测的信号当作
正常尾部。这个方案应保留为否定实验，不应通过放松既定约束把它包装成升级。

## 3. long guard 高精度档

v4 的外部 81 个误报中有 71 个来自 `libero_long`。开发集只搜索 long-suite
lock head，非 long 保持 v4，L5 instability head 全部保持 v4。开发选择为：

```text
libero_long lock = L12 low mobility, W4, K8, q70
other lock       = v4 all_median, W4, K4, q75
instability      = v4 L5, W4, K8, q80
```

外部 `libero_long` 误报从 71 降到 24，suite precision 从 73.41% 升到
88.06%，但 TP 从 196 降到 177。全局结果因此是明显的 precision/recall 交换，
而不是无代价提升。外部 long-suite FPR 为 0.644%，最差任务 FPR 为 1.269%，
分别守住 1% 和 2.5% 的选择约束，但任务间差异仍需在新 cohort 中确认。

## 4. lead guard 早报档

在 long guard 固定后，只对非 long 的最终 OR 系统搜索全层中位数及八个单层
low-mobility lock。共评估 1,890 组；1,310 组满足开发集整体 FPR <= 0.5%、
suite FPR <= 1%、task FPR <= 2.5% 和 precision >= 75%。按 early-4、
early-8、总召回依次排序后选中：

```text
non-long lock    = L15 low mobility, W1, K2, q85
libero_long lock = L12 low mobility, W4, K8, q70
instability      = v4 L5, W4, K8, q80
```

外部相对 v4 多得到 16 个至少提前四个 query 的风险报警，同时少 8 个总 TP、
少 8 个 FP。收益主要来自 `libero_spatial`，其 early-4 召回从 28.57% 升到
52.86%；`libero_object` precision 则降到 53.70%。外部最差单任务 FPR 为
2.506%，以一个离散报警的幅度略超开发 2.5% 约束。因此该档可以继续盲测，
但当前证据不足以设为唯一默认档。

## 5. 可部署性与复现

两个新运行档都导出逐任务 `deployment_profiles.npz`。在线 monitor 只接收当前
query 的 `hb_router_probs`，使用 float32 因果均值、连续越阈计数和 latch；
测试会从原始 Zarr 路由逐 query 回放，核对 lock、instability 和最终 OR 的
首次报警位置。

```bash
python moe-v4-0904/experiments/evaluate_persistent_calibration_v5.py
python moe-v4-0904/experiments/evaluate_long_guard_v5.py
python moe-v4-0904/experiments/evaluate_lead_guard_v5.py
pytest -q moe-v4-0904/tests
```

结果分别位于 `results/persistent_calibration_v5/`、`results/long_guard_v5/`
和 `results/lead_guard_v5/`。每个目录保留开发候选、selection、无 outcome
阈值、封存首报、逐 episode、逐 suite/任务表和带 SHA-256 的 summary。

## 下一步门槛

最优先事项不再是继续查看 external 8B 调参，而是生成一批新的盲测 seed，
在运行前固定 profile hash、选择规则和验收条件。建议至少同时报告三档结果，
并以 task-cluster bootstrap 给出 early-4、FPR 和 precision 差值区间。若
`libero_object` 的 precision 下降在新数据复现，应把 lead guard 再细分为
spatial 专用档，而不是继续降低全局阈值。
