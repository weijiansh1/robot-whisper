# Train-free 多头在线 Trap 报警实验

## 直接结论

**总体结果不好。** 在 14,800 条主 cohort 上，纯 MoE、train-free 的四头在线
报警器能富集失败，但不能作为可靠的统一 Trap 触发器；它没有超过更简单的双头均值，
也明显弱于“轨迹已经运行多久”的时钟基线。

主 operating point 是完全无标签定标的 q95。逐条轨迹的结果是：

| detector | TP / 487 failures | FP / 14,313 successes | failure recall | success FPR | precision | detected-failure lead median |
|---|---:|---:|---:|---:|---:|---:|
| elapsed-time baseline | 194 | 301 | **39.84%** | **2.10%** | **39.19%** | 12 q |
| instant four-head max | 102 | 807 | 20.94% | 5.64% | 11.22% | 18 q |
| dual mean | 88 | 665 | 18.07% | 4.65% | 11.69% | 17 q |
| **causal four-head max** | **83** | **773** | **17.04%** | **5.40%** | **9.70%** | **18 q** |
| instability | 80 | 729 | 16.43% | 5.09% | 9.89% | 18 q |
| lock-in | 62 | 737 | 12.73% | 5.15% | 7.76% | 18 q |

也就是四头一共报警 856 条，只有 83 条最终失败，773 条最终成功。报警后的失败率
从先验的 3.29% 上升到 9.70%，约富集 2.95 倍，但 **90.3% 的报警仍是成功轨迹**。

按任务成簇 bootstrap，四头相对最佳 typed 单头 `instability` 的 recall 只增加
0.62 pp，95% CI `[-4.49, +4.08]` pp，不显著；相对 `dual_mean` 反而减少
1.03 pp，同时 FPR 增加 0.75 pp。相对时钟基线，recall 低 22.79 pp，95% CI
`[-59.58, -5.22]` pp。它不满足预先冻结的“总体改善”判据。

这不是 AUC 结论。所有数字来自实际 `ALARM / NO ALARM` 事件。

## 整个过程

1. 从旧事后分析中只拿表型假设，固定四个头：`instability`、`lock_in`、
   `flat_narrow_support`、`feedback_decoupling`。
2. 顺序回放 37 个任务、14,800 条轨迹、221,781 个 query。每个 q 只使用当前
   MoE forward 和过去 q；没有 centered window，也不读取未来 q。
3. 对 held-out 初始状态的 8 条轨迹，只用同任务其他 49 个初始状态的 392 条无标签
   轨迹建立同-q经验分布。成功与失败都保留，评分端不读取 outcome。
4. 每个 typed head 等权合并固定方向的 routing percentile，再取过去 3 个 q 的均值。
   `multi_max` 是四头最大值。q0--q3 为 warm-up，首次可报警 query 是 q4。
5. 阈值不是逐 query q95，而是 392 条参考轨迹各自“全程最大因果分数”的 q95；因此
   已经支付了多次查看带来的报警预算。超过阈值后报警锁存。
6. 先写出分数、阈值、布尔报警和 SHA-256 封条；manifest 中
   `labels_used=[]`。之后独立评价脚本才读取成功/失败及物理代理 onset。
7. 报警后再看最终 outcome：成功计 FP，失败计 TP。失败内部再检查物理代理类型和
   onset，但这些标签从不回流到报警阈值。

所以用户此前描述的“14,800 条，一旦报警，最后看失败还是成功”正是本次主评价，
但这个 outcome 只用于最后记账，不用于学习或定阈值。

## 报警预算

四头在三个冻结 operating point 下表现为：

| unlabeled threshold | TP | FP | recall | FPR | precision | >=4 q early recall |
|---|---:|---:|---:|---:|---:|---:|
| q90 | 132 | 1,500 | 27.10% | 10.48% | 8.09% | 26.49% |
| **q95** | **83** | **773** | **17.04%** | **5.40%** | **9.70%** | **16.02%** |
| q97.5 | 52 | 416 | 10.68% | 2.91% | 11.11% | 9.86% |

降低阈值主要是在换取更多误报，没有出现一个兼具高 recall 和低 FPR 的工作点。

## Onset 附近

487 条失败中有 325 条能定义协议中的物理代理 onset。四头 q95 的首次报警：

| primary behavior | n | strict precursor `[-4,-1]` | timely `[-2,+2]` | reaction `[0,+4]` | any alarm |
|---|---:|---:|---:|---:|---:|
| all proxy onsets | 325 | 5.54% | 10.77% | 8.62% | 22.15% |
| gripper cycling | 38 | **13.16%** | **23.68%** | 13.16% | 36.84% |
| regrasp/drop | 133 | 6.77% | 15.79% | 11.28% | 23.31% |
| stagnation | 108 | **0.93%** | **2.78%** | 5.56% | 16.67% |

这说明在线信号最像“cycling/retry 附近的 instability”，不是可靠的 static 前兆。
在及时命中的 9 条 gripper-cycling 中，7 条由 instability 头胜出；及时命中的 3 条
stagnation 全部由 lock-in 头胜出。对 regrasp/drop 的 21 次及时命中，16 次也是
instability，而不是 feedback 头。

因此多头对**命中后的恢复分流**有一点机制价值，但覆盖不足以单独决定是否干预。

## 为什么事后强、在线弱

旧事后审计在 episode 50%--90% 上聚合整段统计，并用其他成功轨迹定标；它知道哪里是
“健康参考”，也能看到 Trap 形成后的长期锁死。新实验同时移除了三种优势：

- 当前轨迹只能看过去，不能利用后续长期均值；
- 阈值由所有无标签轨迹定标，不能把成功轨迹挑出来当健康基线；
- 阈值针对全程最大值，必须控制连续查看造成的多重报警。

在高失败任务里，无标签参考库会被常见失败表型污染。例如 moka-pots 任务失败率很高，
其 Trap 路由会被经验分布视为“常见”，不再是 tail anomaly。另一方面，成功轨迹也会在
抓取、阶段切换和健康纠错时短暂出现 volatility、recurrence 或窄支持集；只看路由而
不看任务进度，无法判断这种瞬态最后是恢复还是恶化。这就是 normal 混入的主要原因。

另一个实测问题是启动瞬态。严格按协议让平滑头从 q4 开始后，四头 recall 从最初实现
有效 q6 warm-up 的 19.30% 降到 17.04%，FPR 从 5.23% 升到 5.40%。q4--q5 的正常
启动变化抬高了 episode-max 阈值，反而压掉后续信号。修正前产物被保留为
`results/online_multihead_hub_pre_q4_fix`，最终表格使用协议一致的 q4 版本。

## 外部三任务复验

旧审计完成后又新增了 3 个任务、1,200 条轨迹。报警器代码在读取它们的 outcome 前已
冻结。该 cohort 有 45 条失败：

| detector | TP | FP | recall | FPR | precision |
|---|---:|---:|---:|---:|---:|
| dual mean | **30** | 34 | **66.67%** | 2.94% | 46.88% |
| elapsed-time | 27 | **17** | 60.00% | **1.47%** | **61.36%** |
| four-head max | 27 | 43 | 60.00% | 3.72% | 38.57% |

协议一致的 q4 sensitivity 保持 27/45 recall，只新增 2 个 FP，FPR 为 3.90%。这是
有价值的外部信号，但只有 3 个任务，四头 recall 的任务-bootstrap 95% CI 为
`[16.67%, 76.92%]`，且增益集中在其中两个任务。更重要的是，四头仍没有超过
`dual_mean` 或时钟基线，所以它不推翻主结论。

## 对后续修正的实际意义

当前可保留的接口是连续 typed scores，而不是四头 OR/max 的统一触发：

- 高 instability：候选恢复为重采 flow seed、重新观测或撤回后重规划；
- 高 lock-in / flat support：候选恢复为打破稳定动作模式或显式检查任务进度；
- feedback 头当前证据最弱，不能据此宣称能识别 false grasp。

一个未预注册的诊断是把 `dual_mean@q95` 与时钟报警 OR：recall 47.84%、FPR
6.46%、precision 20.14%，其中 MoE 额外抓到 39 条时钟漏掉的失败。因为 OR 增加了
报警预算，这不是公平的主结果，但它指出下一版应该是：

```text
progress / survival hazard head  -> 决定是否需要报警
MoE phenotype heads              -> 提供额外证据并选择恢复分支
```

也就是说，MoE 多头更适合做“发生了哪种内部失配”的条件信息，不适合脱离任务进度单独
承担“是否已经是 Trap”的判断。

## 产物

- 冻结协议：`ONLINE_MULTIHEAD_PROTOCOL.md`
- 在线实现：`online_multihead_alarm.py`
- outcome/onset 评价：`evaluate_online_multihead.py`
- 主结果：`results/online_multihead_hub/`
- 外部复验：`results/online_multihead_hub_external/`
- 协议一致外部敏感性：`results/online_multihead_hub_external_q4_sensitivity/`
- 图：`figures/online_multihead_alarm.png`
- 测试：`test_online_multihead.py`

本实验使用预计算路由，CPU 比 GPU 更合适。GPU 6 上原有 5 个进程和约 80 GB 显存占用
未被终止、清理或修改。
