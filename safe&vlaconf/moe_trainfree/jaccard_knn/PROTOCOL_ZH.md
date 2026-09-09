# JA / WJ kNN：运行前固定方案

2026-09-06。用户要求用 JA 或 WJ 做 kNN。沿用已探索数据，本轮是回顾性距离对照。

## 输入与固定项

沿用 round5_knn 的 12 个 suite/split、全部 reference/calibration/test 行、成功参考点的
episode/query 身份及顺序。每库 4,096 点，k=20；所有方法从零基 q7 起评分。
每个 query 使用最后一个 flow 步 d9，8 个 HB MoE 层、10 个动作 token，对齐 80 个位置。
JA 使用实际记录的 Top-4 专家 ID，避免从 float16 概率重新选 Top-4 时出现并列歧义。
WJ 与 Hellinger 使用全部 32 个专家的路由概率，各位置独立归一化；不对概率做中心化或 MAD 缩放。

这不是把 JA/WJ 直接套在含负数的 10 维动态上。层和 token 的对应关系保留，未使用状态 token，
未平均掉 token 身份，也未搜索层、flow、token、k 或窗口。最后 flow 和全部 8 层与原路由对照一致。
新路由表示保留 token，所以需要同表示 Hellinger 对照，不能把相对旧 10 维方法的差异全部归因于距离。

## 距离定义

设位置 s=1..80，P_s、Q_s 为归一化的 32 专家概率，S_s、T_s 为记录的 Top-4 集合。

- `route_ja_k20`：`d = mean_s(1 - |S_s intersect T_s| / |S_s union T_s|)`。
- `route_wj_site_k20`：`d = mean_s(1 - sum_e min(P_se,Q_se) / sum_e max(P_se,Q_se))`。
- `route_wj_aligned_k20`：保留位置并展平为 2,560 维，`d = 1 - sum_se min / sum_se max`。
- `route_hellinger_aligned_k20`：80 个位置 Hellinger 距离的 RMS，
  `d = sqrt(sum_se (sqrt(P_se)-sqrt(Q_se))^2 / (2*80))`。

两种 WJ 分别对应仓库中的逐位置平均和对齐后整体 min/max 口径，不混为同一个指标。
全部精确检索，各方法独立找最近 20 个成功参考点，异常分数为 20 个距离的均值。
Numba 仅加速穷举距离；不使用近似邻居，不训练嵌入或检测器。

## 校准、评估与核验

独立 A 成功校准轨迹，按原规则同时保存 episode 与 task/init 组峰值校准。
alpha=1/3/5/10/15/20%，主工作点仍为 task/init 组校准 5%；严格超阈值后报警并锁存。
不因路由特征无需历史而提前到 q0，保持与上一轮同样的观察起点。
评分代码只读 A 的成功标签；所有新预测封存后，评估器才读取 B 的最终标签。

复用原 evaluator 的 seen/unseen、完整/同任务统一观察长度/q7/q14/q21 前缀指标，
比较召回、实际 FPR、t_det、同任务和同初态 AUC，以及相对目标 release 的时序。
读取 round5 原指标作为基线，不按 B 结果重新选工作点。标称 alpha 不保证 unseen FPR。
B 数据已经被多轮探索，跨折重复也不算新的独立轨迹。

核验原始路由哈希、episode/query 对齐、参考身份与校准秩。
用直接集合交并、min/max、平方根欧氏距离复算独立样本；验证 NumPy 与加速检索一致，
对选定完整轨迹逐 query 重放并核对首次报警。保存全部预测、阈值、逐点示意 CSV、图与报告。
