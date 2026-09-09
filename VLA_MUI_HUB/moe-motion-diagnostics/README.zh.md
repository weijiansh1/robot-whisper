# 振荡、抖动与 MoE 告警的联合观察

入口：[本轮报告](results/report.zh.md)，[总体图](results/overview.png)，[固定分析协议](PROTOCOL.md)。

使用 `VLA_MUI_HUB` 中与 v7/v8/v8.2 对齐的全部 `right-16x32` 数据：5 个任务、2,560 条轨迹。
逐条读取末端位置、预测动作和真实 `server/routes.zarr`，区分停滞、方向反复、周期性回返及预测动作抖动。
比较冻结方法的有效告警与上述现象的先后关系，不将最终失败标签当作振荡标签。

```bash
python VLA_MUI_HUB/moe-motion-diagnostics/analyze.py
python -m pytest VLA_MUI_HUB/moe-motion-diagnostics -q
```

`results/episode_metrics.csv` 是逐条索引，`results/examples.csv` 是确定性选取的案例及同初态对照。
`results/event_alignment.csv` 区分早于确认与早于整个观察窗；`results/fixed_query_metrics.csv` 控制任务、初态和观察时刻。

真实末端位置的采样间隔是 10 个动作；10 步预测指令的高频变化不等于真实机械振动。
本轮是离线观察，没有训练新模型、重新选择 v8 阈值或执行机器人控制干预。
