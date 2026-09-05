# 单 rollout 的 MoE 路由导数报警实验

## 结论

可以把“瞬时路由表型”和“随 query 的变化”写成零阶、一阶和二阶信号，
但本次严格 train-free 实验的结果是负面的：**把变化率直接并入硬报警，既没有
提高失败检出率，也没有减少成功误报。** 当前应继续使用两帧持续性
`persistent_level` 作为硬报警；导数最多作为 `suspect` 状态或后续时序确认的输入。

这不是 AUC 排序实验。每条 rollout 从头开始逐 query 更新，一旦分数严格超过按任务
冻结的阈值就锁存报警，最后才用结果标签统计 TP、FP、FN 和 TN。

## 方法

每个 query 先得到四维 MoE 路由表型：

```text
h(q) = [instability, lock_in, flat_narrow_support, feedback_decoupling]
level(q) = max(h(q))
```

然后只利用当前和历史 query 计算：

```text
velocity_raw(q)     = ||h(q) - h(q-1)||_2 / 2
acceleration_raw(q) = ||h(q) - 2h(q-1) + h(q-2)||_2 / 4
```

`velocity_raw` 和 `acceleration_raw` 再按同任务、同 query 的历史参考库转换成经验
百分位。因为 loop 与 static 没有统一的有符号方向，这里测量的是四维表型变化的
模长，不把某一个方向强行规定为异常。

冻结的组合包括：

```text
persistent_level   = min(level(q-1), level(q))
level_velocity     = sqrt(level * velocity)
level_acceleration = sqrt(level * acceleration)
derivative_mean    = mean(level, velocity, acceleration)
derivative_fusion  = max(persistent_level,
                         level_velocity,
                         level_acceleration)
```

所有检测器统一从 query 7 开始。每个初始状态及其 8 个 flow-noise draw 一起留出，
其余 392 条历史轨迹构成 outcome-blind 参考库。成功/失败、最终长度、超时、
`sim_state`、state 和 action 都不进入在线分数。

## q95 事件结果

数据包含 37 个任务、14,800 条 rollout，其中 487 条失败、14,313 条成功。

| 检测器 | 失败报警 TP / 487 | 成功误报 FP / 14,313 | Precision | 提前 8 query 检出 |
|---|---:|---:|---:|---:|
| `instant_level` | 97 (19.92%) | 762 (5.32%) | 11.29% | 18.28% |
| **`persistent_level`** | **111 (22.79%)** | **725 (5.07%)** | **13.28%** | **20.53%** |
| `level_velocity` | 62 (12.73%) | 806 (5.63%) | 7.14% | 11.29% |
| `level_acceleration` | 88 (18.07%) | 790 (5.52%) | 10.02% | 14.99% |
| `derivative_mean` | 67 (13.76%) | 837 (5.85%) | 7.41% | 11.70% |
| **`derivative_fusion`** | **81 (16.63%)** | **801 (5.60%)** | **9.18%** | **14.58%** |

`derivative_fusion` 相对 `persistent_level`：

- 失败召回下降 6.16 个百分点，task-bootstrap 95% CI
  `[-11.33, -2.86]`；
- 成功误报增加 0.53 个百分点，95% CI `[+0.28, +0.82]`；
- 提前 8 query 的召回下降 5.95 个百分点，95% CI
  `[-10.41, -3.17]`。

因此它没有通过预先冻结的决策条件。

## 为什么导数变差

第一，正常执行本来就包含抓取、闭合、抬升、运输和放置等阶段切换。这些切换会让
路由表型快速变化，所以“大导数”不是 Trap 专属信号。融合报警中，成功轨迹有
541/801（67.5%）由速度分支主导；失败轨迹只有 37/81（45.7%）由该分支主导。

第二，在固定总体报警预算下，加入更多候选分支会抬高融合阈值。q95 下融合相对
持续性规则新增了 13 个失败，却新增了 437 个成功报警；同时丢掉了持续性规则能
抓到的 43 个失败。新增导数报警的失败占比只有 `13 / (13 + 437) = 2.89%`。

第三，物理 onset 对齐也没有显示更好的前兆能力。在 325 个有可用 onset 的失败中，
`persistent_level` 的 onset 前报警率为 11.38%，`derivative_fusion` 为 10.15%；
严格 `[-4,-1]` query 前兆率分别为 6.46% 和 5.23%。

所以当前数据支持的解释是：

```text
瞬时变化大 = 任务阶段切换 或 异常修正
持续保持异常 = 更像 Trap
```

导数能说明“模型正在改变计算方式”，但单靠它不能说明这种改变是健康重规划还是
失败前兆。

## 在线使用建议

硬报警保留：

```text
alarm(q) = persistent_level(q) > task_threshold_q95
```

导数只输出不触发干预的候选状态：

```text
suspect(q) = high(level(q)) and high(velocity(q) or acceleration(q))
```

后续若仍坚持纯 MoE，可以研究“导数脉冲是否在短窗口内复发”，而不是把一次导数
峰直接当作 Trap。该规则需要重新冻结并在新 cohort 验证，不能在当前标签上挑参数
后宣称有效。

运行接口为：

```python
profile = DerivativeTaskProfile.load(profile_path, task_name)
monitor = RouteDerivativeMonitor(
    profile,
    quantile=0.95,
    alarm_detector="persistent_level",
)

result = monitor.update(current_router_prob, current_expert_ids)
```

每次 `update` 只接收当前 forward 的 `[8,10,11,32]` router probability 和
`[8,10,11,4]` expert ID。纯 CPU 基准为中位数 1.04 ms/query、p95 1.13 ms/query。

## 验证

- 一个完整任务的 400 条原始 rollout、5,064 次在线更新，与向量化重放比较；
- 2,400 个“rollout × detector”q95 报警结果零不一致；
- 最大逐 query 分数误差 `5.96e-8`；
- 全部 30 个测试通过；
- `persistent_level` 与上一轮封存的 `persistent_route` 在分数、阈值和报警上逐元素一致。

本实验是在同一 cohort 上查看前一轮结果后设计的探索性实验，不是独立验证。
