# MoE 信息使用审计与重组方案

## 核心判断

当前判断器并不是“充分使用了 MoE 但效果差”，而是读取了一个很丰富的 HB 路由张量后，过早压成三个标量。两套留出数据的 164 次报警中，163 次来自跨 chunk recurrence，只有 1 次来自另外两个信号。因此最终实现近似一个晚期复返判断器。

每次重规划原始 HB 路由包含：

```text
8 HB layers x 10 flow steps x 11 suffix tokens x 32 experts
= 28,160 probabilities
```

当前三个信号实际接触到其中 14,208 个位置对应的概率，完全不读取另外 13,952 个位置；更严重的是，14,208 个数最终被压成三个标量，layer、flow、token 和 expert 局部结构基本消失。

## 存储字段库存

| 字段 | 每次重规划形状 | 当前判断器 | 判断 |
| --- | ---: | --- | --- |
| `hb_router_probs` | `8x10x11x32` | 部分使用 | 唯一进入判断器的原始字段 |
| `hb_expert_ids` | `8x10x11x4` | 未显式使用 | 必须直接读取；保存的 full softmax 为 float16，第 4/5 名并列时无法恢复原始 top-4 tie-break |
| `hb_selected_prob` | `8x10x11x4` | 未使用 | 配合实际 expert ID 构造 selected-weight 路由；当前 full-soft 距离可能被 28 个非选中 expert 稀释 |
| `hb_entropy` | `8x10x11` | 未使用 | 可由 full softmax 重算；不适合单独报警，只适合作为上下文 |
| `as_expert_ids` | `4` | 未使用 | 两个 run 的所有 508,023 次重规划中，每个任务内恒定 |
| `as_probs` | `4x3` | 未使用 | 每个任务内完全恒定，跨任务只有 4 个静态模式；没有动态 Trap 信息 |

`hb_entropy` 可以从 `hb_router_probs` 重算，给定实际 ID 后 `hb_selected_prob` 也等于对应位置的保存概率；但 `hb_expert_ids` 不能可靠地从 float16 full softmax 重建。在一个任务前 32 次推理的 28,160 个 layer-flow-token 位置中，7,841 个位置因 top-4 边界量化并列而无法精确恢复实际 ID 集合。因此旧判断器确实丢掉了实际 expert support tie-break；同时也没有构造 selected-weight、entropy/margin 等不同视角。

## 当前到底用了什么

### 去噪末段占比

- 使用后四个 HB 层，即模型实际层 12--15；
- 使用 action token 1--10；
- 使用全部 10 个 flow step；
- 计算相邻 flow 的 Hellinger 距离；
- 最终只保留“最后三个 transition 占完整路径的比例”。

它丢掉了绝对波动大小。完整路径和末段同时变得更不稳定时，占比可能不变。程序已经计算并缓存 `late_flow_volatility` 和 `route_acceleration`，但二者没有进入报警。

### state/action 响应差

- 使用前四个 HB 层，即模型实际层 2--5；
- 只看 final flow；
- 先把 10 个 action token 求均值，再与上一 chunk 比较；
- 最终只保留 `state jump - action jump` 的正部分。

先平均 action token 会抵消不同位置的相反变化；取正部分又删除了 action 比 state 变化更大的另一类异常。后四层的 state/action gap 完全没有进入判断。

### 跨 chunk recurrence

- 使用后四层、final flow、action token 1--10；
- 分别和前 1--4 个 chunk 比较 full-softmax weighted Jaccard；
- 对 40 个 layer-token cell 求均值；
- 对四个 lag 取最大值；
- 连续两个 query 超阈值后报警。

这里丢掉了主 lag、每个 lag 的完整曲线、top-4 支持是否真的相同、哪个 token/layer 在复返。更重要的是，q0--q3 的自基线分别只有 0--3 个可用 lag，而 q4 以后固定有 4 个 lag；用早期不完整的 max-lag 分布给后期 max-lag 打基线，结构并不对称。

## 没有利用的坐标

| 轴 | 当前保留 | 被丢掉的信息 |
| --- | --- | --- |
| layer | 只区分前四层/后四层并求均值 | 单层异常、前后层传播次序、layer 间一致性 |
| flow | 后层 action 的末段占比；其他头只看 final flow | 绝对 volatility、acceleration、早中晚形状、收敛方向 |
| token | state 与 action 均值 | action 1--3、4--7、8--10 的位置差异及 token disagreement |
| expert | full-softmax 距离 | top-4 ID 切换、实际 combine weight、margin、support size、expert transition |
| chunk time | max lag 1--4 | 每个 lag 的独立基线、mobility 趋势、change point、条件熵和持续时间 |
| MoE family | 只用 HB | AS 未用；但实测 AS 在任务内恒定，应排除而不是强行加入 |

原始 HB 中完全不被当前三个头访问的 13,952 个概率位置为：

- 前四层的 flow 0--8、全部 11 个 token：12,672 个；
- 后四层的 state token、全部 10 个 flow：1,280 个。

## 历史结果对重组的约束

以下数字只能指导组织，不能直接当成新的在线结果：

- 中后段表型实验中，`lock_in` 检出 487 个失败中的 259 个，同时在 14,313 个成功中报警 758 个；其中检出 136 个 stagnation 中的 92 个。
- 将 instability 与 lock-in 做等权平均，检出 487 个失败中的 324 个，同时在 14,313 个成功中报警 742 个。但该实验使用轨迹 50%--90% 的窗口和同任务成功参考，不是 early online detector。
- 完全自参照的 response-collapse 规则只检出 530 个失败中的 78 个，在 15,070 个成功中报警 35 个。它说明窄表型可以高选择性报警，但不能覆盖全部 Trap。
- 单独 gate entropy 的早期信号没有跨初始状态稳定复现，因此不能因为“尚未使用”就直接加入 OR。
- Scene8 的晚期实验中，top-4/selected-route 距离比 full-softmax weighted Jaccard 更稳定；这支持将支持集合与软概率分开，但仍需在 40 个任务上重新验证。

## 下一版的组织方式

先保留一个小型位置网格，不直接对所有 token 求均值：

```text
2 layer stages: front / back
x 3 flow stages: early / middle / late
x 4 token groups: state / action1-3 / action4-7 / action8-10
= 24 cells
```

每个 cell 同时保留两种路由视角：

1. soft view：32 维概率的 Hellinger/JS；
2. support view：top-4 ID 与归一化 combine weight。

然后只构造三个机制头。

### 1. 锁死头

组成：

- lag-1 route mobility 降低；
- 同一个 lag 的 top-4 recurrence 升高；
- top-4 union 收缩；
- token disagreement 降低。

要求 mobility 与 recurrence 同时异常，support/token 至少一个确认。lag 1--4 分开维护基线，禁止先取最大值再与不完整的早期 max-lag 基线比较。

### 2. 抖动头

组成：

- late-flow 绝对 volatility 升高；
- route acceleration 升高；
- 相邻 flow 的 top-4 switch count 升高；
- 跨 chunk mobility 升高。

该头对应去噪无法稳定和阶段抖动。当前缓存中前两项已经存在，只是没有被评分。

### 3. 脱节头

组成：

- front state/action gap；
- back state/action gap；
- state jump 与 action jump 分开保留，不截断符号；
- action8--10 相对 action1--3 的 route disagreement。

要求 state/action gap 与 token/layer 传播异常共同出现，避免仅凭一次 state jump 报警。

### 上下文量，不单独报警

- gate entropy；
- top1--top2 margin；
- effective expert count；
- flow 总路径长度。

这些量用于确认“锁死是高置信锁死还是平坦犹豫”，不能各自直接 OR 进总报警。

## Train-free 合并

- 每个机制头内部使用固定方向和中位数/分位数，不拟合权重；
- 每个头要求最近 3 个 query 中至少 2 次成立；
- 最终三头取 OR，但阈值对整个 OR 后的 episode 最大值统一确认，不能给每个头单独放宽后直接相加；
- 成功参考只用于控制报警数量，不使用失败标签选 feature 或方向；
- 同时保留纯 episode 自基线版本，检查跨新任务时是否更稳；
- 评价必须报告具体 TP、FP、FN、TN，并与相同 FP 数量的固定时间钟比较。

## 当前不能利用的信息

`cache_new` 设置了 `store_hidden=false`，因此没有：

- router 输入 hidden state；
- 每个 expert 的输出或加权贡献；
- attention/VLM hidden；
- 接触、力觉或真实 grasp belief。

所以当前最多能证明“路由动力学出现某种模式”，不能从这些缓存恢复专家内部语义，也不能仅凭路由确认机器人是否真正夹住物体。

## 决策

冻结当前三标量判断器作为负结果。下一轮不调它的阈值，而是按“锁死、抖动、脱节”三头重新提取位置保持的 MoE 证据；AS 和单独 entropy 不进入主报警。首先在两套共 32,000 条轨迹上做逐头消融，再冻结整套 OR 规则检查另一 seed 组。
