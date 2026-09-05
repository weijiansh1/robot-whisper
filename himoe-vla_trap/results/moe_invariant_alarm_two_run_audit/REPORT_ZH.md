# 两套 50x8 MoE 报警实现与复现审计

## 结论

低召回不是路由轴读反、episode 行错位或向量化公式算错造成的。修复后的程序从原始 Zarr 重算第一组后，逐 query 预测文件与旧结果 SHA256 完全一致；第二套独立 flow-noise 数据又复现了几乎相同的失败模式。

但是原实现确实有三个工程问题：只配置了第一套 50x8；缓存身份没有绑定 run/config；相对 `--output` 会在写清单时崩溃。后二者已经修复，第一项由本实验补齐。它们没有改变第一组的预测结果。

## 数据覆盖

共检查 2 个 run、80 个 task-run、32000 条轨迹、508023 次重规划；失败共 1096 条。两组 flow seed 分别为 1000--1007 与 1008--1015。所有 Zarr 行数、episode_id、control_step、client inference_calls 和 `[8,10,11,32]` 路由几何均精确一致。

## 同组固定协议

| reference_seed_group | evaluation_seed_group | failures | detector_tp | detector_fp | detector_precision | detector_failure_recall | detector_success_false_alarm_rate | matched_clock_failure_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed1000_1007 | seed1000_1007 | 114.0000 | 51.0000 | 32.0000 | 0.6145 | 0.4474 | 0.0104 | 0.5965 |
| seed1008_1015 | seed1008_1015 | 125.0000 | 55.0000 | 26.0000 | 0.6790 | 0.4400 | 0.0085 | 0.5840 |

两组留出集合并后，MoE 检出 106/239 个失败，failure recall=44.35%，成功误报率=0.94%。这不是部署结果：同误报时间钟仍然更强。

## 跨 run 冻结迁移

下面不在目标 run 重估参考分布或阈值：直接把一组的完整规则应用到另一组。

| reference_seed_group | evaluation_seed_group | failures | detector_tp | detector_fp | detector_precision | detector_failure_recall | detector_success_false_alarm_rate | matched_clock_failure_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed1000_1007 | seed1008_1015 | 125.0000 | 55.0000 | 30.0000 | 0.6471 | 0.4400 | 0.0098 | 0.5840 |
| seed1008_1015 | seed1000_1007 | 114.0000 | 51.0000 | 28.0000 | 0.6456 | 0.4474 | 0.0091 | 0.5965 |

## 公式独立复算

慢速逐 cell 标量实现与主程序共比较 3592 个值。最大绝对差：convergence=0.000167334，state jump=3.28855e-06，action jump=2.71869e-05，response=2.69942e-05，recurrence=1.37641e-07。差异来自主程序中间缓存的 float16 量化，远低于信号尺度，没有报警逻辑分歧。

## 为什么判定为规则问题

- 第一组修复前后预测哈希完全相同，排除了缓存修复或输出路径修复改变数值。
- 第二组使用不同的 8 个 flow-noise seeds，阈值、召回、报警位置和 recurrence 主导现象仍近似相同。
- 两组都被相同成功误报率下的固定时间钟支配，说明 detector 主要利用了晚期复返/轨迹变长，而不是提前出现的 Trap 特异信号。
- convergence+response 分支几乎从不触发；所谓三头规则实际上退化为单一 recurrence 规则。

因此应保留这份结果作为负结果，不再围绕当前阈值微调。下一版需要先重新定义能与时长解耦、并在 onset 前局部出现的 MoE evidence。
