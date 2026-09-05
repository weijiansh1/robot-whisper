# Train-free 的 T10 → 下一 chunk T1 检查

## 核心结论

- 重规划边界确实有动作跳变：六维机械臂命令 RMS 为 **0.0524**，是 chunk 内边缘相邻动作的 **2.18×**。按 checkpoint 已有动作尺度换算后为 0.223 个标准差。
- MoE 在边界也明显重排：Top-4 Jaccard 均值 **0.052**，完整 80-site Top-4 一致为 **0/48748**。
- 但两种跳变没有稳定耦合。控制任务、初始状态和 query 后，实际 selected-route 与 arm jump 的秩相关为 **r=0.011**，六种固定分数 maxT `p=0.8554`。
- 完整 32-way soft router 的原始距离略强（**r=0.029**），但六分数 maxT `p=0.0694`；绝对相关最大的固定标量就是 `soft_raw`，仍未达到显著。
- 逐 denoise × 逐动作维的 70 格扫描中，最大绝对相关为 **r=0.022**（D2 / dz），maxT `p=0.4009`。
- 失败侧出现一个候选：前 8 个边界的 `soft_raw_h8` 直接 AUC **0.606**，14 分数 maxT `p=0.0003`；4 个有失败的任务均为正方向。
- 但它还不是通用告警：去掉 `spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` 后 AUC 降到 **0.519**，说明主要增益由单个任务贡献。

## 方法（无训练）

每个边界的主向量为 10 维：`[d0, ..., d9]`。第 d 维是旧 T10 与新 T1 路由在该 denoise step 的 Hellinger 距离，并对 8 个 HB 层平均。`selected_*` 使用实际 Top-4 combine weights；`soft_*` 使用完整 32-way router probabilities。没有 PCA、回归、分类器、拟合阈值或学习权重。
两种表示各自固定两个无参数对照：`*_minus_same = boundary - mean(same T1, same T10)`，去掉同位置跨 chunk 的变化；`*_minus_edge = boundary - mean(old T9→T10, new T1→T2)`，去掉 chunk 两端局部 token 变化。它们没有从成败标签中估计系数。
动作侧直接使用 `new T1 - old T10` 的 7 维命令差；主统计使用前六维 raw RMS，checkpoint 元数据中的既有 action std 只作为辅助量纲。
相关性固定使用所有任务共有的 q0-q7，并在 task × init-state × query 内做秩变换。置换时对所有任务和初始状态使用同一个 flow-noise seed 列置换，保留采样网格依赖。

## 10 维路由边界向量的总体均值

```text
[0.9453, 0.9500, 0.9526, 0.9557, 0.9559, 0.9573, 0.9565, 0.9557, 0.9529, 0.9417]
```

完整 soft-router 的对应均值：

```text
[0.0482, 0.0497, 0.0522, 0.0551, 0.0580, 0.0612, 0.0645, 0.0684, 0.0729, 0.0773]
```

## 固定路由标量与动作跳变

| route score | rank correlation | six-score maxT p |
|---|---:|---:|
| `selected_raw` | +0.011 | 0.8554 |
| `selected_minus_same` | +0.002 | 1.0000 |
| `selected_minus_edge` | +0.006 | 0.9821 |
| `soft_raw` | +0.029 | 0.0694 |
| `soft_minus_same` | -0.000 | 1.0000 |
| `soft_minus_edge` | +0.028 | 0.0841 |

## 各任务

| task | selected H | soft H | same-position selected H | Top-4 Jaccard | arm RMS | boundary/within | selected r | soft r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `goal/open_the_middle_drawer_of_the_cabinet` | 0.957 | 0.045 | 0.807 | 0.047 | 0.0491 | 2.56× | -0.007 | +0.015 |
| `goal/open_the_top_drawer_and_put_the_bowl_inside` | 0.955 | 0.047 | 0.825 | 0.049 | 0.0568 | 1.63× | -0.022 | +0.016 |
| `long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 0.950 | 0.055 | 0.753 | 0.054 | 0.0399 | 3.04× | +0.032 | -0.000 |
| `spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 0.950 | 0.081 | 0.849 | 0.055 | 0.0621 | 1.52× | -0.009 | -0.000 |
| `spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 0.950 | 0.076 | 0.808 | 0.054 | 0.0541 | 2.16× | +0.060 | +0.113 |

## 固定标量的失败 AUC

`route_*_h4/h8` 是前 4/8 个边界的相应固定 route score 均值；`action_h4/h8` 是同一前缀的 arm RMS 均值。AUC > 0.5 表示值越大越偏失败。

| fixed score | AUC | 14-test maxT p |
|---|---:|---:|
| `selected_raw_h4` | 0.515 | 1.0000 |
| `selected_raw_h8` | 0.457 | 0.6985 |
| `selected_minus_same_h4` | 0.465 | 0.8852 |
| `selected_minus_same_h8` | 0.520 | 0.9978 |
| `selected_minus_edge_h4` | 0.487 | 1.0000 |
| `selected_minus_edge_h8` | 0.499 | 1.0000 |
| `soft_raw_h4` | 0.503 | 1.0000 |
| `soft_raw_h8` | 0.606 | 0.0003 |
| `soft_minus_same_h4` | 0.481 | 0.9990 |
| `soft_minus_same_h8` | 0.567 | 0.1263 |
| `soft_minus_edge_h4` | 0.489 | 1.0000 |
| `soft_minus_edge_h8` | 0.574 | 0.0609 |
| `action_h4` | 0.497 | 1.0000 |
| `action_h8` | 0.555 | 0.3499 |

`soft_raw_h8` 在 4/4 个可评估任务中 AUC > 0.5，逐任务中位数为 0.524；leave-one-task-out 范围为 0.519–0.701。这说明方向一致，但效应强度尚未跨任务稳定。

| task | failures | soft_raw_h8 AUC | state directions | state median | leave-one-state range |
|---|---:|---:|---:|---:|---:|
| `goal/open_the_top_drawer_and_put_the_bowl_inside` | 42 | 0.509 | 3/7 | 0.484 | 0.480–0.548 |
| `long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 216 | 0.521 | 8/13 | 0.541 | 0.499–0.535 |
| `spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 12 | 0.528 | 4/8 | 0.580 | 0.483–0.589 |
| `spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 37 | 0.900 | 10/12 | 0.958 | 0.867–0.934 |

该 AUC 是 task/初始状态内的 success-failure pair 加权结果，没有训练预测器；全成功的 middle-drawer 任务自动不参与 AUC。它仍是同一批离线数据上的探索性检验，不是独立验证集性能。

## 解释

重规划会同时重置 action-token 的相对位置（T10 回到 T1）、更新图像/状态并注入下一次 flow 初值，所以路由重排很大并不必然造成动作跳变。当前结果表明二者更多是并行发生，MoE 路由不能稳定替代动作连续性指标。
失败预测是另一件事：`soft_raw_h8` 可能在累积早期状态偏离，而不是测量 T10→T1 的动作跳变。它可以作为无需训练的候选排序分数继续做 held-out task/checkpoint 验证；若要变成二值报警器，仍需要一个不从当前评估集调出来的阈值。

## 产物

- `transitions.npz`: 48,748 个边界的 selected/soft route 10D、原始 7D action delta 和固定对照。
- `episode_features.npz`: 固定前 4/8 个边界的 train-free episode 标量。
- `summary.json`: 完整相关矩阵、maxT p、逐任务结果。
- `overview.png`: 动作/路由边界、相关性和失败 AUC 总览。
