# 无任务先验的 MoE 自参照报警实验

## 核心结果

规则没有训练参数，运行时只读取当前 episode 已产生的 HB router probability。它不用任务 ID、正常轨迹库、动作数值、reward、success 或物理距离。阈值曾查看一个开发任务，因此本实验是 **train-free、runtime reference-free，但不是开发阶段 label-free**。

严格冻结验证排除开发任务，覆盖 39 个未参与规则设定的任务：

- failure recall：78/530 = 14.72%
- success false-alarm：35/15070 = 0.232%
- precision：69.03%
- 检出失败的中位提前量：8.5 个 query
- 至少提前 5 个 query 的检出：60/530

包含开发任务的 40-task 描述值为 recall 15.04%、FPR 0.226%。开发任务 `push_the_plate_to_the_front_of_the_stove` 的两条失败均报警，398 条成功均未报警；它不能再算独立验证。

## 方法

每条轨迹用 q1--q6 的路由作为自身基线。每次只对最近 3 个 query 取中位数，同时要求：

1. 前四层 state-token 的跨 query 路由跳变降到自身基线的 0.25 倍以下；
2. 后四层 action-token 的完整去噪路由加速度升到 1.10 倍以上；
3. 前四层 state/action 路由间隙升到 1.20 倍以上。

三项相交描述的是：输入侧路由响应近乎冻结，但 action-token 的去噪计算仍在更剧烈修正，并与 state-token 路由进一步分离。它比单独使用高 recurrence 更不容易把任务固有的稳定计算误判为 Trap。

## 时钟混杂审计

只要 episode 最终失败，它通常会跑满 horizon，因此“很晚还没结束”本身就是强信号。与 MoE 相同或更低 FPR 的固定归一化时钟位于 horizon phase 0.81，recall 为 100.0%，但中位提前量只有 9.0 个 query。匹配 MoE 中位提前量的时钟 FPR 为 0.21%。因此不能只报 episode recall；提前量和成功误报必须一起报告。

## 异质性与边界

最差 held-out 任务是 `libero_long/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`：FPR 6.35%（25/394）。这说明自参照消除了大部分静态任务差异，但没有消除所有合法阶段转换。

总体 recall 只有约 15%，所以它不是通用失败检测器。更准确的定位是一个高精度的 **internal response-collapse phenotype head**：报警时证据较强，不报警不能说明机器人正常。它也不能仅凭 routing 证明“belief 错了”或识别具体物理原因；这些需要视频、接触或对象状态作事后解释。

## 可复现文件

- `episode_predictions.csv`：逐 episode 冻结预测和解盲结果
- `query_decisions.csv`：逐 query 三个 MoE 比率及报警
- `task_metrics.csv`：逐任务 TPR/FPR
- `clock_baselines.csv`：固定 horizon 时钟审计
- `summary.json`：机器可读汇总与 task-bootstrap 区间
- `task_free_self_reference_audit.png`：任务异质性及 lead/FPR 图
