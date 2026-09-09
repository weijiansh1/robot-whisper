# 无监督 replanning loop 发现

本实验先完成物理回返选择、无标签分位数标定、候选筛选和聚类，
之后才读取 episode outcome。failure/success 没有参与任何发现决策。

## 数据覆盖

| task | episodes | failures | queries | shared prefix |
|---|---:|---:|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 512 | 0 | 6452 | 12 |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 512 | 42 | 10018 | 17 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 512 | 216 | 22883 | 35 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 512 | 12 | 5139 | 9 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 512 | 37 | 6816 | 11 |

## 标签盲发现

每个当前 query 先仅凭 qpos 选择一个“离开后返回”的历史起点。
主候选要求无标签联合分数超过各任务等长前缀的 99% 分位，
并且 routing 与 action 的返回增益都为正。

| task | seed pairs | positive topology | q99 threshold | raw candidates | NMS candidates / episodes |
|---|---:|---:|---:|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 4404 | 199 | 0.739 | 1 | 1 / 1 |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 7970 | 3767 | 0.885 | 84 | 77 / 75 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 20835 | 12116 | 0.874 | 320 | 238 / 132 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 3091 | 644 | 0.840 | 30 | 26 / 26 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 4768 | 327 | 0.724 | 35 | 28 / 22 |

## 无监督 motif 聚类

Ward 聚类在看不到 outcome 的情况下选择了 2 个簇，silhouette=0.278，状态=ok。
Cluster-task purity=0.832；越接近 1 表示聚类越像在区分任务而非共享 motif。

| cluster | events / episodes | median score | median lag | median phase | X closure | route gain | action gain | failure episodes after unblinding | matched baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 247 / 139 | 0.917 | 19.000 | 0.763 | 0.495 | +0.089 | +1.102 | 70/139 | 0.414 |
| 2 | 123 / 118 | 0.892 | 9.000 | 0.500 | 0.312 | +0.061 | +0.650 | 27/118 | 0.108 |

## Outcome 解盲验证

这里只使用各任务等长前缀，并在 task x initial-state 内比较。
AUC > 0.5 表示标签盲发现分数在最终失败 episode 中更高。

| endpoint | failure AUC | cluster 95% CI | p one-sided | family-wise p |
|---|---:|---|---:|---:|
| maximum loop score | 0.542 | [0.444, 0.644] | 0.0606 | 0.0960 |
| 90th-percentile loop score | 0.564 | [0.458, 0.670] | 0.0110 | 0.0160 |
| q99 candidate present | 0.556 | [0.507, 0.608] | 0.0004 | 0.0332 |

### 逐任务异质性

| task | failure candidates | success candidates | q99 candidate AUC | p one-sided |
|---|---:|---:|---:|---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | 0/0 | 1/512 | all success | - |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 6/42 | 64/470 | 0.501 | 0.6147 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 37/216 | 68/296 | 0.515 | 0.2937 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0/12 | 25/500 | 0.473 | 1.0000 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 18/37 | 1/475 | 0.722 | 0.0002 |

Leave-one-task-out 是解盲后的稳健性检查。

| omitted task | remaining q99 candidate AUC | p one-sided |
|---|---:|---:|
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | 0.566 | 0.0004 |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.602 | 0.0002 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.565 | 0.0004 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.508 | 0.3917 |

### 阈值和最小周期敏感性

| endpoint | failure AUC | 95% CI | p one-sided |
|---|---:|---|---:|
| top 0.5% candidate | 0.540 | [0.512, 0.573] | 0.0008 |
| top 2% candidate | 0.535 | [0.473, 0.607] | 0.0224 |
| top 5% candidate | 0.534 | [0.458, 0.620] | 0.0314 |
| min lag 4: maximum loop score | 0.525 | [0.420, 0.627] | 0.1788 |
| min lag 4: 90th-percentile loop score | 0.560 | [0.452, 0.665] | 0.0134 |
| min lag 4: q99 candidate present | 0.554 | [0.503, 0.609] | 0.0004 |

### Post-hoc 绝对回返幅度审计

该检查在首次 outcome 解盲后新增，不属于预注册主检验。
它要求 X、routing、action 各自至少恢复指定比例的离开幅度，
用于区分微小回摆和有实际幅度的闭环。

| minimum recovered departure | failure AUC | 95% CI | p one-sided | family-wise p |
|---:|---:|---|---:|---:|
| 1.0% | 0.543 | [0.503, 0.584] | 0.0030 | 0.0032 |
| 2.5% | 0.523 | [0.491, 0.555] | 0.0616 | 0.0776 |
| 5.0% | 0.512 | [0.485, 0.539] | 0.2158 | 0.3027 |
| 10.0% | 0.501 | [0.490, 0.511] | 0.5585 | 0.7301 |
| 25.0% | 0.500 | [0.500, 0.500] | 1.0000 | 1.0000 |

## 结论

标签盲 q99 回环候选在最终失败中富集；这支持 recurrent geometry 与 outcome 的关联。
该方向只在一个任务中成立：libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate；移除它后总体关联消失。
要求三种表示都至少恢复 5% 离开幅度后不再显著，因此当前证据更像任务特定的微回返，而非强 routing basin 闭环。
这项分析不需要知道失败原因，但也不能把任何几何簇命名为滑落、空抓或目标错误。
cluster outcome 比例来自全轨迹，受 episode 长度和终止阶段影响，只作描述。

## 限制

- X 是 MuJoCo qpos，不是 RGB 或 vision embedding。
- 检测使用当前及下一 query，因此是事件发现，不是提前预警。
- MoE 信号是 action-token top-4 routing identity，不是 expert output。
- Outcome 富集是关联证据，不是 routing 导致失败的因果证据。
- 没有使用或生成语义 failure-cause 标签。
