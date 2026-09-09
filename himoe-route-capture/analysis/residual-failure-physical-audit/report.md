# 剩余失败的物理阶段审计

## 直接结论

三方法 2/3 共识覆盖 213/307 个失败。剩余 94 条由 1 票 62 条和 0 票 32 条组成。

它们并不是一组新的、干净的物理失败类：89/94 来自双摩卡壶长任务，而且与核心失败大多处于同一个子任务阶段。长任务剩余失败中，只有 pot 2 到位而 pot 1 未到位的有 81/89；核心中对应为 123/127。

真正清楚的差别是失败后的运动形态：0 票长任务失败多数在接近 pot 1 后又回到已经完成的 pot 2/炉灶一侧；三票核心则停在未完成的 pot 1 一侧。前者晚期路由仍活跃并出现长跨度回返，后者更像减速后静滞。因此，没有聚进共同核心不等于没有结构，而是共同核心偏向抓住一种特定的静滞终局。

## 组成

| task | 0 vote | 1 vote | total |
|---|---:|---:|---:|
| `open_the_top_drawer_and_put_the_bowl_inside` | 0 | 1 | 1 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 30 | 59 | 89 |
| `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 0 | 1 | 1 |
| `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 2 | 1 | 3 |

所有失败均跑到各任务固定的失败 horizon：long=52、top-drawer=30、ramekin/stove=22 queries。剩余组不是由更短轨迹造成。

| physical failure proxy | 0 vote | 1 vote | total |
|---|---:|---:|---:|
| lifted_not_placed | 0 | 3 | 3 |
| misplaced | 0 | 3 | 3 |
| never_grasped | 2 | 0 | 2 |
| partial | 27 | 51 | 78 |
| reached_then_lost | 3 | 5 | 8 |

`failure_mode` 来自既有 MuJoCo 轨迹启发式，不是真值人工标签。

## 长任务阶段

成功轨迹通常先让 pot 2 到达成功终点代理（median query 18），再让 pot 1 到达（median query 38）。在同时穿过两个 5 cm 代理球、因而可判定顺序的 289/296 条成功轨迹中，289/289 都是 pot 2 先于 pot 1；其余成功轨迹不被这个代理判定。

| votes | pot2 only | pot1 only | both at goal | neither |
|---:|---:|---:|---:|---:|
| 0 | 28 | 0 | 0 | 2 |
| 1 | 53 | 0 | 3 | 3 |
| 2 | 23 | 1 | 3 | 0 |
| 3 | 100 | 0 | 0 | 0 |

到位判据与 `replanning-reset-trap` 一致：物体 XYZ 距成功轨迹终点均值不超过 0.05 m。它只是可复现的阶段代理。

## 终端回退

终端盆地只比较最后一个 query 的末端执行器 XYZ 到两个物体当前 XYZ 哪个更近，不使用 outcome，也不要求发生接触。

| votes | terminal closer to pot1 | terminal closer to pot2 |
|---:|---:|---:|
| 0 | 3 | 27 |
| 1 | 47 | 12 |
| 2 | 26 | 1 |
| 3 | 100 | 0 |

在 30 条长任务 0 票失败中，27 条终止时更靠 pot 2；在 100 条三票失败中，100 条都更靠 pot 1。0 票组后半程到 pot 1 的最小距离 median 约 0.094 m，说明它们通常不是从未切换，而是靠近 pot 1 后又退回。

一票本身不是一个类别：

| vote pattern (aligned/event/lag) | n, long task | pot1 basin | pot2 basin |
|---|---:|---:|---:|
| 100 | 42 | 41 | 1 |
| 010 | 9 | 2 | 7 |
| 001 | 8 | 4 | 4 |

Aligned-only 更接近共同核心的第二物体终局；event-only 与 lag-only 更混合。把 62 条一票失败平均成一类会掩盖这个差别。

## 路由动态

以下只比较同一个 long task、同一个 52-query 失败 horizon。

| votes | n | late/early route speed | mean return lag | peak phase | late stasis | terminal return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 1.337 | 0.685 | 0.625 | 6.7% | 53.3% |
| 1 | 59 | 0.951 | 0.667 | 0.375 | 8.5% | 42.4% |
| 2 | 27 | 0.826 | 0.333 | 0.500 | 33.3% | 37.0% |
| 3 | 100 | 0.560 | 0.222 | 0.000 | 88.0% | 45.0% |

整体上，票数越高，路由越明显减速并进入晚期静滞；三票组的变化峰值也最早。0 票组则相反：晚期变化重新增强，回返跨度更长。这是连续梯度，不支持再硬切出若干等价密度的失败类。

## 初态与随机种子

| init | failures | 0 vote | 1 vote | core |
|---:|---:|---:|---:|---:|
| 10 | 23 | 4 | 19 | 0 |
| 39 | 28 | 10 | 3 | 15 |
| 42 | 18 | 1 | 10 | 7 |
| 3 | 27 | 3 | 7 | 17 |
| 0 | 18 | 4 | 5 | 9 |
| 26 | 23 | 4 | 4 | 15 |
| 20 | 9 | 2 | 4 | 3 |
| 7 | 3 | 0 | 3 | 0 |
| 23 | 3 | 1 | 1 | 1 |
| 13 | 23 | 1 | 0 | 22 |
| 29 | 4 | 0 | 1 | 3 |
| 46 | 1 | 0 | 1 | 0 |
| 49 | 32 | 0 | 1 | 31 |

在完整 16x32 long 交叉网格上，固定 seed 后 residual membership 与 init 有关（permutation p=0.0002）；固定 init 后，整体 residual 与 flow seed 没有清楚关系（p=0.3663）。

0 票子集单独看时，init 与 seed 都有探索性关联（p=0.0002 / 0.0024）。这是事后挑出的 30 条小样本，不能当成确认性 seed 机制。

## 与成功轨迹的邻近性

距离在三个已保存分析 embedding 中分别计算；每条 rollout 只找同 task、同 init 的成功 sibling。表中再除以该 cell 内成功对成功的典型最近邻距离；1 表示约等于成功岛内部尺度。

| failure votes | available | aligned ratio | event ratio | lag ratio |
|---:|---:|---:|---:|---:|
| 0 | 32 | 2.25 | 1.69 | 1.40 |
| 1 | 61 | 2.91 | 1.59 | 1.67 |
| 2 | 23 | 3.38 | 4.40 | 2.19 |
| 3 | 159 | 5.64 | 5.22 | 3.39 |

0/1 票失败确实比三票核心更接近相同初态的成功轨迹，但距离通常仍大于成功岛内部尺度。结合 lag-HDBSCAN 把这些失败判为 noise，准确说法是“更靠成功流形的稀疏边缘”，不是“与成功无法区分”。

## 下一步实验

最值得验证的二级结构不是重新对完整 episode 强制聚类，而是按 pot 2 首次到位对齐，只看之后的固定历史：

1. `stay-at-pot1`：切到第二物体后停在那里；
2. `return-to-pot2`：靠近第二物体后又回到已经完成的第一子目标；
3. 对两组分别匹配 init、pot 1/pot 2/eef 位置和剩余预算，再测试 routing 是否仍有增量。

这能直接检验“路由暴露策略没有真正更新”，同时避免把物体位置或最终 timeout 当作 MoE 机制。

## 限制

- 三个投票块来自先前的探索性分析；这里冻结它们，不构成独立确认。
- 相对相位特征仍依赖 episode 的最终 horizon，只能作 post-onset、pre-terminal 描述。
- 长任务 goal 是成功终点均值代理，不是从环境内部重新构造的成功判据。
- 终端盆地只是距离哪个物体更近，不代表发生抓取或接触。
- embedding 欧氏距离只作比较诊断，不是校准后的物理距离或 kernel 距离。
- init/seed 置换检验是探索性的，未校正多重比较。

## 产物

- `episode_audit.csv`: 2,560 条逐 episode 投票、失败代理、物理阶段和成功邻距。
- `long_initial_state_counts.csv`: long task 的 init 组成。
- `long_flow_seed_counts.csv`: long task 的 flow seed 组成。
- `summary.json`: 报告中全部计数和检验。
