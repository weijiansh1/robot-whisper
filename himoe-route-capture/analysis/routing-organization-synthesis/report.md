# MoE 路由组织方式探索：综合结论

## 直接结论

原先把 10 个相对相位的 recurrence 向量直接展平，再用 Ward 强制分成 K 类，确实不是合适的主组织方式。它主要恢复 task、checkpoint、episode length 和终止阶段，而且把本应拒绝分配的稀疏失败点硬塞进成功簇边缘。

八种替代表示给出的共同结构不是“六种失败”，而是下面四层：

1. **任务特定的成功流形**：成功 rollout 形成紧密、可重复、但强 task-specific 的路由轨迹族。
2. **跨任务的晚期未完成核心**：大部分失败 rollout 共享一种路由减速、晚期平台化、非局部回返增加、层间变化同步的轨迹形状。
3. **稀疏失败尾部**：没有进入共同核心的失败大多来自 long moka-pot task；它们不形成一个新的稳定密度簇。
4. **持续高 sibling-deviation 表型**：相对同 task、同初态的 31 条 noise siblings，有一小群 rollout 从中段到末段一直处于 route position/speed/bend 的高偏离端。它与失败相关，但不是失败专属类。

因此，更合适的组织不是 flat taxonomy，而是：

```text
task-conditioned success manifolds
            |
            +-- shared noncompletion core
            |
            +-- heterogeneous off-manifold tail
            |
            +-- persistent peer-deviation phenotype
```

## 共同失败核心

三种完全不同、且 outcome-blind 拟合的组织独立抽出了几乎同一组 rollout：

| 方法 | 块 | n | success/failure | precision | failure recall |
|---|---|---:|---:|---:|---:|
| shift-aligned landmark kernel | raw C1 | 252 | 6/246 | 0.976 | 0.801 |
| route change-event statistics | C0 | 214 | 5/209 | 0.977 | 0.681 |
| lag-spectrum kNN graph | Louvain C7 | 219 | 4/215 | 0.982 | 0.700 |

- 三方法至少两票：`217 = 213 failure + 4 success`，precision `0.982`，failure recall `0.694`。
- 三方法共同交集：`185 = 182 failure + 3 success`。
- 三方法并集：`283 = 275 failure + 8 success`，precision `0.972`，failure recall `0.896`。
- 并集遗漏的 32 个失败中，30 个来自 long moka-pot，2 个来自 stove spatial。
- Raw C1、Event C0、Lag C7 两两 Jaccard 为 `0.707 / 0.725 / 0.827`。

这不是一次算法偶然性。Aligned raw K6 的 80% 子采样稳定性为 ARI median/P10 `0.991/0.977`；event K4 为 `0.862/0.828`。Lag C7 是探索性图社区复现，不单独作为确认性证据。

该核心的路由动力学特点是：

- route speed 从中段较高值持续下降，晚期进入低速平台；
- backward/nonlocal return 比其余 rollout 更频繁；
- route change 在不同 MoE layers 上更同步；
- 它更像“未完成演化/停滞终局”，不是某一种已标注物理错误。

## 不是最后一帧伪影

Aligned kernel 在删除每条 rollout 最后 10% 后重新完整拟合：

| 指标 | 结果 |
|---|---:|
| full raw K6 vs truncate90 raw K6 ARI | 0.950 |
| full C1 | 252 / 246 failure |
| truncate90 matched C1 | 249 / 219 failure |
| episode precision / recall / Jaccard | 0.851 / 0.841 / 0.734 |
| failure-only precision / recall / Jaccard | 0.945 / 0.841 / 0.802 |
| truncate90 raw K6 stability ARI median/P10 | 0.985 / 0.934 |

四个有失败的任务中，truncate90 C1 的 failure-rate delta 均为正。Lag-spectrum 的 truncate90 Louvain C9 同样为 `237 = 224 failure + 13 success`；它与 full C7 的 Jaccard 为 `0.900`。因此共同核心不是仅由最后一个终止观察产生。

但这仍然只是 **post-onset, pre-terminal noncompletion signature**。相对相位依赖最终 episode horizon，0.9T 可能已经晚于第一次物理错误；它不能被解释为第一次滑落或碰撞的前兆。

## Density 视角

HDBSCAN 不强迫所有点归类，得到一个看似不同、实际兼容的答案：

- full lag-spectrum：`307/307` failures 都是 noise；9 个密集块全部是 success，且几乎都是单任务块。
- truncate90 lag-spectrum：`306/307` failures 是 noise。
- full route topology：`304/307` failures 是 noise。
- truncate90 route topology：`288/307` failures 是 noise。
- path signature 和 layer wave 在冻结密度尺度下没有任何高密度簇。

Raw C1 的 252 条、Event C0 的 214 条、Lag C7 的 219 条全部属于 lag-HDBSCAN 的 noise 超集。因此两种说法不矛盾：

> HDBSCAN 先识别“没有进入紧密成功岛”的宽泛异质超集；shift alignment 和图社区再从该稀疏区域中抽出一个高纯失败核心。

这也解释了旧 Ward 分析里“失败在边界/卫星团”的现象：它们不是多个等密度球形类别，而是成功流形之外的一条稀疏分支。

## Peer-deviation 表型

`peer_rank` 先对每个 `task x initial-state` 的 32 条 seed rollout 做 sibling-median residual，再只保留 route position、speed、bend 在 sibling 内的分位数。它几乎消除了 task/length partition：NMI 分别为 `0.017/0.043`。

HDBSCAN C1 是一个真实且参数稳定的高偏离岛：

| window | C1 n/failure | C1 failure rate | task-init null expected | count ratio | conditional p |
|---|---:|---:|---:|---:|---:|
| full | 245/70 | 0.286 | 34.19 | 2.048 | 0.00002 |
| truncate90 | 255/69 | 0.271 | 31.66 | 2.180 | 0.00002 |

- C1 的 point/speed/bend 平均 sibling rank 约为 `0.84/0.84/0.84`，non-C1 约为 `0.46`。
- 每个相位都保持高偏离；不是 terminal spike。
- full/truncate90 的 failure-member Jaccard 为 `0.805`。
- 42 组 HDBSCAN 参数中，可行高偏离岛对默认 C1 的 Jaccard median 为 full `0.938`、truncate90 `0.957`。

但是加入 exact episode length 后，full 的预期 failure 数为 `69.5`、实际为 `70`，`p=0.498`；truncate90 为 `68.25` 对 `69`，`p=0.251`。证据主要来自两个 spatial tasks，top-drawer 和 long moka-pot 的任务内效应较弱。

所以它应被描述为：

> 当前语料中跨五任务、持续高 peer-deviation 的路由密度岛；其 membership 与失败相关，但主要经由 longer/timeout trajectories，不是独立失败机制。

## 其他组织方式

| 表示 | 得到的结构 | 结论 |
|---|---|---|
| route-state grammar | 稳定 K6，state vocabulary seed ARI `0.997` | task excess NMI `0.884`，length R2 `0.960`；是任务语法，不是失败语法 |
| route change events | 稳定 K4，Event C0 高纯失败 | 控制 task+exact length 后 AUC 增量 `-0.002`，没有独立 outcome 增量 |
| aligned route kernel | raw 稳定 K6，C1 跨四任务 | 最强共同失败核心；task residual 后无稳定 partition，outcome excess 降到 `0.004` |
| lag variogram | 成功密度岛 + failure noise；Louvain 抽出 C7 | 支持 off-manifold failure core，但 partition 强 task/length shadow |
| persistent topology | 成功密度岛 + failure noise | 没有比 lag variogram 更干净的失败类；Louvain 41/49 communities 明显碎片化 |
| path signature | HDBSCAN 全 noise，Louvain silhouette 为负或很低 | 没有离散的 reparameterization-invariant path-shape 类 |
| layer wave | HDBSCAN 全 noise，Louvain 弱分区 | 层间传播不是一个独立、紧密的失败 taxonomy |
| peer rank | 两个小密度岛 + 大量 noise | 得到持续高偏离表型，但不是高纯失败类 |

## 最严谨解释

当前数据支持：

> MoE routing 对成功行为形成紧密、任务特定的流形；一部分已经进入 noncompletion、停滞或 timeout 演化的 rollout 会共同离开这些流形，并形成一个跨四任务、跨三种表示复现的晚期失败核心。其余失败主要是 long-task 的异质稀疏尾部。

当前数据不支持：

- 六种或其他固定数量的任务无关物理失败类别；
- 第一次物理错误发生前的 MoE 预测信号；
- 与 task、timeout length、已发生失败状态独立的 recovery-trap 机制；
- unseen-task 泛化。

下一阶段若研究在线预警，应停止按最终 T 归一化整条 episode，改为按第一次物理错误或第一次恢复 query 对齐，在固定可用历史上计算 aligned-kernel、lag-noise 和 peer-deviation 三类分数。

## 产物

- [Aligned kernel full report](../aligned-route-kernel/report.md)
- [Aligned kernel truncate90 report](../aligned-route-kernel-truncate90/report.md)
- [Change-event report](../route-change-events/report.md)
- [State-grammar report](../routing-state-grammar/report.md)
- [Alternative quotient/topology report](../alternative-routing-organizations/report.md)
- [Alternative terminal-truncation audit](../alternative-routing-organizations/truncate90_report.md)
- [Peer-rank C1 audit](../alternative-routing-organizations/peer-rank-c1-audit/report.md)
