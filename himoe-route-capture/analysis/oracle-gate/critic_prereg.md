# 状态条件价值 critic 协议

> 状态：2026-08-22 冻结草案，尚未采集新数据，本文没有 critic 结果。

## 问题与准入

目标是检验 `Q(观测, proprio, phase, 候选动作块)` 能否在新任务中从 K 个候选中选出更有价值的动作。纯路由、纯噪声和候选云中心度不再单独立项；route/noise 只作为 critic 的消融特征。

采集前先在筛选 seed `11000..11999` 上找非饱和任务。任务入选规则固定为基线成功率在 `25%-75%`，并尽量覆盖不同 suite、场景和接触类型；筛选 rollout 不进入训练或评价。每个真实价值标签先跑 oracle：若真实标签直接选候选的增益不超过 `+3pp`，对应标签立即淘汰，不训练 critic。

## 探索轮数据

- 4 个开发任务，每任务 20 个 fork 快照，共 80 个；按起始、接近、接触、完成/恢复四个 phase 分层，每层 5 个。episode 中段快照必须占多数。
- 每快照 K=16 个候选，候选 seed 使用 `20000..29999`；每候选执行一个 10-action 动作块，再接 C=4 个共同未来流，future seed 使用 `40000..49999`，向前运行 50 个物理 action step。
- 同一快照、同一 future 列必须复用完全相同的后续 flow-noise tensor 序列。保存原始观测图像、冻结的视觉/语言 embedding、proprio、phase/contact、sim state、候选初始噪声、完整动作块、route 和专家标量。
- 连续进展函数必须在看到候选结果前写入 `task_value_specs.json`。原始值和同快照 rank 都保存；success-in-window 作为第二标签。

快照级 restore 是硬门：初始观测 hash 必须相同；同候选重复执行的状态漂移不得超过候选间 IQR 的 5%；future tensor mismatch 必须为 0。失败快照整组删除并报告 phase 分布，不能删单个候选。

标签可靠性使用 C=4 的全部 2-vs-2 split，报告 split-half 中位相关和 Spearman-Brown。中位校正可靠性低于 0.60 时，在任何模型拟合前把 C 扩到 8；仍低于 0.60 则关闭该标签。

## 模型与分割

critic 使用两塔结构：状态塔读取冻结的观测/语言 embedding、proprio 和 phase/contact，动作塔读取归一化 10x7 动作块，最终层显式包含状态与动作的交互项。只做加性线性拼接不合法，因为状态在同一候选池内恒定，会在排序时抵消。

探索轮只比较四臂：action-only、state-action critic、critic+route、critic+noise。超参数只能在训练任务内按完整快照分组选择；候选永不跨 train/test。开发任务采用 leave-one-task-out，within-pool AUC 仅为诊断。探索轮用 max-statistic FWER，最终只留下一个冻结模型。

## 对照与评价

- 免费下界：initial-noise-only。
- 上界：连续进展 oracle、success oracle，各自在同一池上选候选。
- 零对照：state-only 和 AS-only 在候选内必须得到 `AUC=0.500`、score range `0.000`。
- 所有方法在同一快照、候选和 CRN 列上配对；只报方法间配对差。
- 主端点：新任务上 best-of-K 的 success 增益，计算量与随机 K 候选匹配。连续短时程价值增益、AUC 和校准只作次端点。

## 功效与确认轮

80 个探索快照只用于选择模型和估计快照级配对方差，不出头条。确认样本量在看确认数据前按

```text
n_per_task = ceil(((1.96 + 0.84) * paired_snapshot_sd / 0.02)^2)
```

计算，并按标签可靠性及同 episode 聚类设计效应向上膨胀。候选分池不能伪装成独立快照。现有二值 K8 数据的 256 个有效池给出约 `3-4pp` 的 CI 半宽，因此若确认仍主要依赖二值 success，应按至少约 1000 个有效选择池预算，而不是把 50–100 个快照当成充分确认。

确认轮冻结探索胜出的唯一模型，在 4 个从未用于模型/方法选择的新任务上评价；candidate seed 使用 `60000..69999`，future seed 使用 `80000..89999`。确认轮不再选方法、不做 FWER。预设裁决为：跨新任务宏平均 success 增益至少 `+2pp`，且 4/4 任务同号；否则 best-of-K critic 线终审关闭。头条只允许引用确认轮。
