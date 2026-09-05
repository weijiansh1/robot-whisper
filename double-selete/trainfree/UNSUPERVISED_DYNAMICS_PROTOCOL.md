# MoE 路由动态语法无监督聚类协议

状态：2026-09-04 在标签揭盲前冻结。第一次无标签诊断中，状态层 HDBSCAN 将
13,824 个窗口全部判为 noise；因此在未读取标签的前提下，将状态词汇主算法修订为
下述 GMM-BIC。HDBSCAN 全 noise 结果保留为模型诊断，不用于选择标签关联更强的版本。

本协议是在上一轮单极值聚类及其标签审计之后提出，因此属于**探索性后续分析**，
不是 discovery-independent 预注册。实现仍须保持计算隔离：聚类程序只能读取无标签
MoE 曲线；`success`、loop/static 标签和 onset 只能在 assignment 与哈希封存后由另一
程序读取。

## 问题

单个 MoE 峰同时出现在健康修正与 Trap 中。这里检验的对象改为动态语法：

```text
局部路由窗口 -> 无标签路由状态 -> 轨迹状态序列
               -> occupancy / dwell / transition / recurrence
               -> 自适应轨迹聚类
```

检验这些轨迹簇是否在揭盲后对应 `normal`、`loop_only`、`static_only`、`both`。
簇数不预设为 3 或 4。

## 数据与信息边界

- 使用 `right-16x32` corpus B 的 512 条轨迹、16 个 init pool。
- 使用与上一轮相同的十个 soft-probability MoE component；不使用 hard Top-4 expert
  ID，不读取已经合成的 loop/static head score。
- 每个 component 已在同一 `(init pool, query)` 内转换为 percentile rank。
- 主分析为所有轨迹共同存在的公共前缀 `q4..q34`；五-query 状态窗的中心为
  `q6..q32`，每条轨迹固定 27 个状态，因此不使用 episode length。
- 次分析使用每条轨迹完整可见范围，状态窗中心为 `q6..q(T-3)`。语法特征使用比例
  和归一化时长，不直接输入 `T`，但可见后缀仍由成功早停决定，所以只能解释为
  retrospective，不得称为在线报警结果。

## 无标签状态词汇

只在公共前缀的 512 x 27 = 13,824 个局部窗口上学习状态词汇。每个中心 query 的
描述为：

- 前后各 2 query 的 5 x 10 component level；
- 相邻 query 的 4 x 10 一阶差分。

共 90 维。逐维 median/IQR robust scaling 并截断到 `[-10, 10]`，PCA 保留至少
90% 方差。

状态算法为 diagonal Gaussian mixture，在 `K=1..12` 中以最低 BIC 自适应选择 K；
每个 K 使用 5 次确定性初始化。HDBSCAN 仍按下列无标签参数作为密度结构诊断：

```text
min_cluster_size = ceil(sqrt(13824)) = 118
min_samples      = ceil(log2(13824)) = 14
cluster_selection_method = eom
metric = euclidean
```

状态数由 GMM-BIC 决定。状态 ID 按 component mean 的第一主成分坐标排序，仅用于
确定性输出；每个状态另存最接近 component mean 的真实训练成员作为 prototype。
所有轨迹序列窗口由冻结的 GMM 后验分配；计算相对所分 component 对角协方差的
Mahalanobis 距离，若超过该状态训练成员距离的 95% 分位数，则记为 noise state。
90% 和 99% 接纳分位数作为无标签敏感性。

## 轨迹动态语法

noise 被保留为一个明确状态，不改名为 normal。对每条状态序列提取：

- 每个状态的 occupancy、visit rate、longest dwell fraction；
- 完整有向 transition matrix（按全部转移数归一化）；
- global switch rate、run-count rate、run-length mean/max/std；
- occupancy entropy；
- lag 1..8 的 exact-state recurrence 与 non-noise recurrence；
- lag 2..8 的 return-after-change rate（例如 A-B-A）；
- 每个状态在序列前半段与后半段的 occupancy 差。

所有量均为比例或由固定状态数确定；episode length 本身不进入特征。常量列被移除，
robust scaling 后 PCA 保留至少 90% 方差。

## 自适应轨迹聚类

公共前缀为冻结的主结果；完整轨迹为冻结的 retrospective 次结果。两者分别使用：

```text
HDBSCAN min_cluster_size = ceil(sqrt(512)) = 23
HDBSCAN min_samples      = ceil(log2(512)) = 9
```

并报告：

- HDBSCAN `min_cluster_size in {16,23,32}`、`min_samples in {8,9,10}`；
- diagonal GMM 在 `K=1..12` 上由 BIC 选择 K；
- OPTICS-Xi；
- 1% feature jitter、80% feature subsampling；
- 公共前缀内逐 `(pool, query, component)` 打乱轨迹身份的语法哨兵；
- 三个确定性合成动态语法簇的阳性对照（稳定驻留、二状态交替、三段式驻留）。

任何 0/1 簇或高 noise 覆盖都是有效负结果，不得使用揭盲标签改参数、选半径或选择
公共/完整窗口。

## 揭盲检验

在无标签 assignment、状态序列、模型选择表和 manifest 哈希封存后，独立程序才可
读取逐 episode 标签。主检验：

- `normal / loop_only / static_only / both` 四类 AMI；
- 排除 `both` 的三分类；
- Trap vs normal、success vs failure 仅作次要结果；
- AMI within-init-pool 条件置换 20,000 次；
- 报告 cluster coverage、contingency、AMI、NMI、ARI，不用标签重排状态或轨迹簇。
