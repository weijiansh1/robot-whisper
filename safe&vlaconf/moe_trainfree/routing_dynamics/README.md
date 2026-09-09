# 通用 MoE 路由动力学编码

本模块将“计算状态如何变化”编码为连续特征。输入只有当前 HB 路由概率，输出 22 维汇总向量及 8 层各 3 维相对量。它不输出失败概率或不可恢复 trap 标签。

## 主要抽象

先区分两个时间尺度：

- 查询间变化：相邻两次策略调用最后一个去噪步的路由差异，使用 Hellinger 距离。
- 查询内变化：同一次策略调用中 10 个去噪步的路由路径长度和二阶变化，后者不是机械臂加速度。

每条轨迹用 q1..q4 的读数中位数作为固定个体参考；当前量取最近 3 次查询均值。对数比表达相对变化，保留绝对量供下游区分绝对低活动与相对变慢。这个个体参考不保证是健康状态。

设 `m` 是后四层查询间变化的相对对数比中位数，`a` 是查询内二阶变化的相对对数比：

| 状态方向 | 编码 |
|---|---|
| 查询间减弱、查询内增强 | `decoupling = min(max(-m, 0), max(a, 0))` |
| 查询间增强、查询内增强 | `both_increasing = min(max(m, 0), max(a, 0))` |
| 查询间减弱、查询内减弱 | `both_decreasing = min(max(-m, 0), max(-a, 0))` |
| 查询间增强、查询内减弱 | `outer_up_inner_down = min(max(m, 0), max(-a, 0))` |

四种方向均保留，不将所有失败强行归入第一种。例如 m=-0.30、a=+0.20，则合取强度为 0.20；只有一项变化时，该合取为 0。

另外编码逐层共同变化、token 多样性变化、最近 4 次的持续性和占据率，以及两个相邻近期窗口之间的信号下降和路由重新变化。`routing_recovery` 仅表示计算信号缓解，不能据此撤销成功轨迹上的历史误报或宣布物理恢复。

## 在线调用

从仓库根目录导入，以下 `routing_queries` 代表现有服务中逐次收到的路由张量：

```python
from moe_trainfree.routing_dynamics.encoder import RoutingDynamicsEncoder

encoder = RoutingDynamicsEncoder()
for hb_router_probs in routing_queries:
    encoded = encoder.update(hb_router_probs)  # shape [8, 10, 11, 32]
    if encoded.ready:
        current = encoded.as_dict()
        decoupling = current["decoupling"]
        layer_changes = encoded.per_layer  # shape [8, 3]

encoder.reset()  # 下一条轨迹开始时清空参考和近期状态
```

不需要传入 task、query、初态、成败、最终长度或剩余时长。内部计数仅用于收集参考和保证窗口可用，不作为分数。在线内存有界：固定参考、3 次基础读数、8 次即时特征，以及上一条最终路由。

`encoded.ready` 表示前 16 个即时维度可用；后续维度仍可能为 NaN。应按字段检查有限性。不能把 NaN 当作正常、低风险或恢复。

| 特征维度 | 字段 | 默认首次可用 |
|---|---|---:|
| 0..3 | `back_mobility`, `back_flow_path`, `flow_acceleration`, `back_token_diversity` | q7 |
| 4..8 | `mobility_log_ratio`, `acceleration_log_ratio`, `path_log_ratio`, `diversity_log_ratio`, `front_back_path_log_ratio` | q7 |
| 9..12 | 四种查询间/查询内方向组合 | q7 |
| 13..15 | `path_decoupling`, `layer_agreement`, `token_diversity_contraction` | q7 |
| 16..18 | `recent_decoupling`, `persistent_decoupling`, `decoupling_occupancy` | q10 |
| 19..21 | `decoupling_drop`, `mobility_rebound`, `routing_recovery` | q14 |

`path_decoupling` 先在同一层内合取变化，再取后四层中位数。`layer_agreement` 统计全部 8 层参与该方向的比例。`persistent_decoupling` 是最近 4 次强度的最小值；一项消失后不会无限锁存。

## 批量编码

`encode_readouts(values, valid)` 接收 `[episode, query, 25]` 的基础读数。字段依次为 8 层 mobility、8 层 flow path、1 个后四层 acceleration、8 层 token diversity。返回 `[episode, query, 22]` 汇总特征及 `[episode, query, 8, 3]` 逐层特征。无效补齐不会进入参考或特征。

完整记录的同一专家编号置换不改变定义；路由先按专家维归一化。底层缓存使用不同有限精度计算时允许小数值差，实际回放误差见结果中的 verification.json。

## 复现

```bash
python -m unittest discover -s moe_trainfree/routing_dynamics -p 'test_*.py' -v
python moe_trainfree/routing_dynamics/extract_readouts.py --output /path/to/new_result_directory
python moe_trainfree/routing_dynamics/evaluate.py --input /path/to/new_result_directory
```

`extract_readouts.py --reuse-readouts /path/to/previous_result_directory` 可复用通过哈希校验的基础读数。现有输出不会被静默覆盖。

上述编码实验固定了窗口和特征，没有拟合检测器或搜索阈值。A/B 都已在先前分析中使用，编码的全任务检查属于回顾性关联检查。主统计按同任务、同 query 比较，并记录仍有观测的两类样本数；没有时长基线或全程峰值主指标。完整结果见 [编码报告](../results/routing_dynamics_20260908/REPORT_ZH.md)。

## 已整合的无训练报警器

`guard.py` 将原 v8.2 分数、连续确认后的内部变化增强、连续确认后的近期脱耦三条分支合并。只以 A 成功参考的峰值秩标定阈值，不训练分类器或学习权重。固定协议见 [整合协议](INTEGRATION_PROTOCOL_ZH.md)，实测结果见 [整合报告](../results/routing_guard_20260908/REPORT_ZH.md)。

```python
from pathlib import Path
import json
from moe_trainfree.routing_dynamics.guard import RoutingGuardMonitor

profile_path = Path("moe_trainfree/results/routing_guard_20260908/fold_0_profile.json")
profile = json.loads(profile_path.read_text())
# checkpoint 应使用当前执行策略的实际标识，需在 profile["checkpoints"] 中。
guard = RoutingGuardMonitor(profile_path, checkpoint=current_checkpoint,
                            method="v82_integrated", kind="task_init", alpha=0.01)
for hb_router_probs in routing_queries:
    result = guard.update(hb_router_probs)
    first_alarm = result["first_alarm_query"]
    current_alarm = result["alarm_active"]
    branch_evidence = result["branch_scores"]
guard.reset()
```

每折 profile 是独立实验配置，示例 fold 0 不代表五折总体性能，也不是根据 B 选择出的部署配置。可用方法为 `v82_reference`、`dynamics_only`、`v82_integrated`，标定单位为 `episode`、`task_init`。原始冻结阈值与新增分支并集只作为离线诊断对照，没有替换原 v8.2 生产入口。

`tail` 是合并后的成功参考尾部秩，`score=-log(tail)` 是其截断分数，不是失败概率。`branch_tails` 保留各分支的秩；AUROC 补充分析使用选中分支的 `-log(min(branch_tails))`，避免合并秩截断至 1 造成信息丢失。该排序分数不改变报警门槛或首次报警。

`NORMAL/WATCH/ALARM` 是当前计算信号状态。连续两次有效低风险观测才能解除当前状态，但 `ever_alarm` 和 `first_alarm_query` 永不撤销。缺失不作为恢复。编码器仍使用有界内存；整合器为复现旧 v8.2 分数保留本条轨迹的低维读数。原 v8.2 的 `0.0015*q` 阈值松弛被原样保留，因此整合方法并非完全无查询序号依赖；没有输入最终时长或计算时长基线。

```bash
python -m unittest discover -s moe_trainfree/routing_dynamics -p 'test_*.py' -v
python moe_trainfree/routing_dynamics/run_guard_experiment.py --output /path/to/new_guard_result
python moe_trainfree/routing_dynamics/verify_guard.py --output /path/to/new_guard_result
```

实验先保存 A 标定 profile、分支阈值和哈希，再读取 B 分数和汇总指标。输入复用已有经过核验的全量特征与 v8.2 缓存。`diagnose_guard.py` 对默认结果目录输出事后分支归因和未截断排序诊断，保留固定协议下的原始 AUROC 表。
