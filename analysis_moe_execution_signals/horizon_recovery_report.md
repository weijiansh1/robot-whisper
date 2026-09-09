# MoE 触发的短时域 Trap 恢复 pilot

## 主要结果

- Trap 前 event 状态：h10 成功 0/23，h2-burst10 成功 0/23；配对净变化 0.0 个百分点（95% bootstrap CI 0.0 至 0.0；maxT p=1）。
- 健康匹配状态：h10 成功 21/23，h2-burst10 成功 19/23。event-minus-control 的差分差分为 8.7 个百分点（95% CI 0.0 至 21.7；maxT p=0.5）。
- 这里正的差分差分完全来自 control 的成功率下降，不是 event rescue：event effect=0/23，control 中有 0 次 rescue、2 次 harm。

## 检测到恢复

- 固定 cosine 报警命中 event 19/23，健康误报 3/23。
- 报警后才启用短 burst 的 event 成功率增量为 0.0 个百分点；等数量随机触发的均值 0.0 个百分点，单侧精确子集 p=1。
- 报警门控策略的绝对结果：event 0/23（always-h10 为 0/23），control 20/23（always-h10 为 21/23）。
- 平均额外查询成本：event 4.00，control 4.70。

## 时间与 recurrence

| 状态 | arm | success@50 | @100 | @150 | @200 | recurrence | mean queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| event | h10 | 0/23 | 0/23 | 0/23 | 0/23 | 2/23 | 20.00 |
| event | h2_burst10 | 0/23 | 0/23 | 0/23 | 0/23 | 2/23 | 24.00 |
| control | h10 | 15/23 | 19/23 | 21/23 | 21/23 | 0/23 | 6.87 |
| control | h2_burst10 | 13/23 | 18/23 | 19/23 | 19/23 | 1/23 | 11.57 |

- 短 burst 相对 h10 防止/诱发 recurrence：event 1/1，control 0/1。

## 解释边界

这是同一重建状态上的因果 A/B，但仍是单任务、单个未来噪声流的 pilot。它否定的是前十个物理步采用 h=2 的局部 burst，不是否定更长 cooldown、持续 receding horizon 或能改变物理状态的恢复动作。检测阈值来自同一批健康控制，必须在新 rollout 上做前瞻验证。
