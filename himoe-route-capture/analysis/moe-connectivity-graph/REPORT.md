# MoE 候选连通图结果

## 结论

- 动作端点：路由邻接与最终完整动作块的局部几何存在经校正的正联系。`T_action=0.24993`，未校正
  `p=0.0001`，两端点 maxT 校正后
  `p=0.0001`；相对共同置换零均值的超额为
  `0.25025`。
- 成败端点：没有检出成功标签在路由图上的稳定局部聚集。`T_success=-0.05241`，未校正
  `p=0.2637`，maxT 校正后
  `p=0.4672`；相对固定类别数零分布的超额为
  `0.00997`。
- 完整 hidden 基线至少同样强，因此路由最多是结构化、较便宜的监测接口，不是 hidden 之外的新信息。

这两个问题必须分开：动作图有效，只能支持“相似计算路由对应相似动作候选”；只有成败图也有效，才可能进一步讨论好/坏盆地。即使二者都有效，这仍是关联证据，不是换专家的因果证据。

`T_success` 的原始值为负不等于显著“反盆地”：固定一个池里的成功数、并用不含自己的邻居投票时，随机标签零均值本来就是 `-0.06238`，不是 0。因此判断依据是共同置换及其超额，而不是单看 `AUC-0.5` 的符号。

## 数据与检验

五个任务各有 16 个严格相同 observation 的状态池，每池共用 32 个 flow-noise seed，共 80 张候选图、2,560 个候选。动作端点使用全部 80 池；成败端点仅使用同时含成功和失败的 40 池。主图由 HB5/d0 十个 action token 的完整 32-way router probability 的 Hellinger 距离构成，未使用 `expert_ids[...,0]`。

统计量固定为 `k=1..8` 的 AUK8。零分布使用 9,999 次共同 seed-column 置换：每次在所有任务和状态上使用同一列置换。动作与成败两个主端点再做 studentized maxT 家族校正。

## 每任务效应

| task | action AUK8 | success AUK8 | mixed pools |
|---|---:|---:|---:|
| goal-middle | 0.30080 | NA | 0 |
| goal-top | 0.25130 | -0.06983 | 7 |
| long-t08 | 0.20602 | -0.04565 | 13 |
| spatial-ramekin | 0.26293 | -0.05264 | 8 |
| spatial-stove | 0.22858 | -0.04151 | 12 |


## 基线边界

下面所有数值都使用完全相同的 AUK8 定义。它们是效应量比较，不是额外的显著性家族。

| source graph | action AUK8 | route minus source |
|---|---:|---:|
| hidden | 0.42414 | -0.17421 |
| noise7 | 0.38751 | -0.13758 |
| noise24 | 0.22043 | 0.02950 |
| top4_weighted | 0.13870 | 0.11123 |
| top4_jaccard | 0.13100 | 0.11893 |


## 连通图与 expert-output 探索

`per_pool.csv` 包含每张图的 k=4 union/mutual component、最大连通分量、归一化 Laplacian `lambda2`、MST 和 balanced-cut 描述量。连通分量本身只描述图的形态，不是成败检验。

MST 是另一个不需要选择阈值的次要口径：动作 `p=0.0001`，mixed40 成败 `p=0.7744`。它不能替换失败的 AUK8 主端点。balanced16 的 success AUK8 为 `-0.00435`，仅作固定敏感性描述。

`disagreement/cancellation/conflict` 来自离线重建的真实 expert-output proxy，而不是 router ID。池内相关显示 `DCQ -> routed sensitivity` 为 `0.485`，但固定的 `route + DCQ` 组合只有 `0.355`，增量 `-0.129`；连通图没有改善这个张力信号。与局部 action distortion 的最大绝对相关也只有 `-0.120`。

直接把 expert 标量当作图上的 target 后，route graph 的 smoothness AUK8 为：D `0.036`、C `0.028`、Q `0.003`；对应 hidden graph 是 `0.069`、`0.067`、`0.070`。这说明 full router 能组织动作候选，但并没有同样清楚地组织 expert-output 张力。

这些都是探索性效应量，没有 p 值，而且 D/C/Q 与 sensitivity 共用重建 expert 输出的范数，不能视作独立验证。下一轮若继续，应在独立 observation 上预注册一个固定张力分数，并直接记录 runtime expert vectors 与 intervention sensitivity。

## 限制

结果只覆盖 HB5/d0、每个 episode 的首次 inference、五个任务和三个 checkpoint suite。图与动作/结果的关系是观察性几何，不是 expert 替换的因果效应；当前数据也没有完整 runtime expert vector，不能构造真正的 signed expert-conflict graph。

## 旧结果撤回

所有把 `hb_expert_ids[...,0]` 当作 top-1 的旧 adjacency-MI 或候选图数字均无效，因为存储的 top-k 是未排序集合。这里的主结果只使用完整 router probability；top-4 控制也是顺序不变的。

## 文件

- `summary.json`：所有汇总效应、p 值、验证和解释边界；
- `per_pool.csv`：80 个状态池的逐图结果；
- `permutation_nulls.npz`：共同置换索引与两个主零分布；
- `overview.png`：固定示例图、完整 k 曲线、零分布与探索性相关矩阵。
