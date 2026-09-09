# Query 内 MoE 微步马尔科夫模型

## 模型边界

这里不再把环境 control step 当成 Markov step。一次 policy query 展开为 `10 denoise × 8 HB layer = 80` 个有序事件；状态 `z[tau, layer]` 是该层 10 个 action token 的完整 32 路概率指纹，经 Hellinger 嵌入和逐层 K-means 离散化。

主链按 `layer 2 -> 3 -> 4 -> 5 -> 12 -> 13 -> 14 -> 15 -> 下一 denoise round 的 layer 2` 展开。转移矩阵以源层为条件，所以不同层的 expert 编号从未被假设为同一种专家。

`occupancy` 是不使用前态的零阶对照；`periodic Markov` 在 10 轮间共享 8 类层边；`position Markov` 为 79 条绝对位置边分别建模，用于检查 denoise 非齐次性。模型只使用路由概率和成败标签，不读取专家职责或 expert output。

## 选择协议

同一初始观测的 32 个 seed 被拆为 4 个互斥 K8 池。每个测试池的完整初始场景，以及该池对应的 8 个 seed，都从成败转移矩阵训练中删除。候选分数为 `log p(path | success) - log p(path | failure)`；并列候选按等概率分摊，避免用 seed ID 人为破局。

主规格固定为 `K=16`、看完前 `3` 个 denoise round（24 个微事件）后选 top-2。

## 主任务的前缀结果

任务：`libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`，成功率 `0.578`。

| 已观察 denoise 轮数 | 零阶 top-2 相对随机 | Markov top-1 | Markov top-2 | Markov top-4 | Markov 池内 AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | -0.008 | +0.023 | +0.051 | +0.039 | 0.557 |
| 2 | -0.008 | +0.039 | +0.078 | +0.043 | 0.584 |
| 3 | -0.016 | +0.039 | +0.066 | +0.031 | 0.575 |
| 5 | -0.008 | +0.031 | +0.074 | +0.027 | 0.571 |
| 10 | +0.023 | -0.031 | +0.039 | +0.039 | 0.539 |

## 主检验

主规格选中候选的平均成功率为 `0.645`，K8 内精确随机期望为 `0.578`，差值 `+0.066`，scene-bootstrap 95% CI `[+0.020, +0.115]`，一致 seed-column 置换单侧 `p=0.0121`。

同前缀、同 K 的零阶占用 top-2 成功率为 `0.562`；Markov 比它高 `+0.082`，配对一致置换单侧 `p=0.0078`。Markov 的期望选择权重覆盖 `28/32` 个 seed，有效 seed 数 `23.4`，最大单 seed 权重占比 `0.094`。

## 跨任务与离散粒度

| 任务 | rollout 成功率 | K16 前 3 轮 top-2 相对随机 | 95% CI | 转移相对位置占用增益 (bit/transition) |
|---|---:|---:|---:|---:|
| `libero_goal/open_the_middle_drawer_of_the_cabinet` | 1.000 | 不可评估（单一 outcome） | - | - |
| `libero_goal/open_the_top_drawer_and_put_the_bowl_inside` | 0.918 | +0.012 | [-0.008, +0.031] | +0.531 |
| `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 0.578 | +0.066 | [+0.020, +0.115] | -0.363 |
| `libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 0.977 | +0.000 | [-0.023, +0.020] | +0.290 |
| `libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 0.928 | +0.010 | [-0.055, +0.059] | -0.170 |

主任务 K 敏感性（前 3 轮、periodic Markov、top-2）：

| K | 选中成功率 | 相对随机 | 池内 AUC |
|---:|---:|---:|---:|
| 8 | 0.633 | +0.055 | 0.574 |
| 16 | 0.645 | +0.066 | 0.575 |
| 32 | 0.596 | +0.018 | 0.531 |

## 结论

时间轴必须放在 query 内部，才能让候选拥有不同的路径；但这并不自动保证一阶 Markov 假设成立。位置条件链相对位置条件零阶占用的留出预测增益在 4 个可评估任务中为正 `2/4` 次，Long 主任务反而为 `-0.363 bit/transition`，说明当前离散链的生成拟合并不稳定。

候选排序在 Long 的固定主规格上出现正关联（相对随机 `+0.066`，`p=0.0121`），也优于零阶占用 `+0.082`；但另外 3 个可评估任务均未形成明确收益，跨任务宏平均只有 `+0.022`。这是可进入独立闭环复验的探索性信号，不是‘已经能稳定选出好 rollout’的结论。

限制：标签属于整个 episode，记录 seed 也影响后续 replanning 噪声流；本分析只使用每条 episode 的第一次 policy query。真正的‘每次选择’仍需要在同一 snapshot 上分叉候选、执行所选动作并使用共同未来噪声做闭环因果验证。

离散 codebook 在每个任务的全部路由协变量上无标签拟合，因此当前检验是 transductive 的；部署或独立复验时必须冻结在历史训练数据上，不能用待评估 query 重拟合。
