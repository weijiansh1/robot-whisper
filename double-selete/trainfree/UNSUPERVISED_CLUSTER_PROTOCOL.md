# MoE 峰谷无监督聚类协议

状态：在新聚类 assignment 生成、且在打开逐轨迹 loop/static 标签之前冻结。

## 问题

只使用 MoE routing，轨迹是否会自然形成可解释的多个密度簇？在聚类结果封存后，
这些簇是否对应 `loop`、`static` 和 `normal`？簇数不预设为 3。

## 主语料

- `right-16x32` moka-pot corpus B，共 512 条轨迹、16 个 init pool、每池 32 条。
- 所有轨迹共同可见的固定前缀为 `q4..q34`，因此主聚类不使用 episode 长度，
  也不把成功提前结束与失败跑满 horizon 当成结构。
- 聚类阶段禁止读取 success、动作、物理状态、loop/static 标签和 onset。

这是 label-blind 无监督计算，但不是 discovery-independent：十个 soft routing
component 来自此前的机制分析。聚类不得读取已经合成的 `loop_score` 或
`static_score`，也不得使用 hard Top-4 expert ID。

## 无标签表示

每个 component 已在同一 `(init pool, query)` 内转换成 percentile rank；单维取反
不改变欧氏距离，因此 `_high/_low` 后缀不向聚类提供类别方向。

使用十个 soft component：gate level/change、late-flow volatility、route
acceleration、route mobility、deep soft consensus/mixture change，以及 final-layer
soft mixture、consensus 和 final-denoise jump。

对每条轨迹，在 `q7..q31` 中选择

```text
argmax_q RMS(component_rank(q) - 0.5)
```

作为纯 MoE 极值锚点。取锚点前后各 3 个 query，拼接：

- 7 x 10 原始 rank 曲线；
- 6 x 10 一阶差分；
- 每个 component 的 mean/std/min/max。

共 170 维。逐维使用全体轨迹的 median/IQR 做 robust scaling，截断到 `[-10,10]`，
随后 PCA 保留至少 90% 方差。整个过程不读取标签。

## 自适应聚类

主算法为 HDBSCAN：

```text
min_cluster_size = ceil(sqrt(512)) = 23
min_samples      = ceil(log2(512)) = 9
cluster_selection_method = eom
metric = euclidean
```

HDBSCAN 自行决定簇数，并允许样本为 noise。若只得到 0/1 个簇或绝大多数为 noise，
这就是“没有可用离散簇”的有效负结果，不得改用标签挑参数。

无标签敏感性包括：

- `min_cluster_size in {16,23,32}`、`min_samples in {8,9,10}`；
- full fixed-prefix curve 和 temporal-novelty anchor 两种表示；
- OPTICS-Xi；
- diagonal Gaussian mixture 的 BIC，在 `K=1..12` 中自行选 K；
- 1% jitter、80% feature subsampling、逐 `(pool,query,component)` 时间身份置乱；
- 同规模三簇 Gaussian 阳性对照。

## 标签解封后的检验

`loop/static` 同时为真的轨迹保留为 `both`，不强行塞入三类。主检验把 HDBSCAN
noise 当作一个 assignment，报告四类 AMI、contingency、cluster coverage 和每簇
组成；显著性用在 16 个 init pool 内置乱标签的 20,000 次条件置换。

同时报告 cluster 与 success、episode length、init pool 的关联作为混杂哨兵，
以及 MoE 极值锚点相对物理 onset 的时差。标签只解释已经冻结的簇，不参与选 K、
选表示、选参数或重排 cluster ID。

