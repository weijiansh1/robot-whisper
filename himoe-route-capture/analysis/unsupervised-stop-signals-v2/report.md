# 无监督逐轮早停信号补充验证

## 实验口径

- Goal：`704` 条轨迹、`44` 个 query、`4` 条 rollout；Long：`181` 条轨迹、`4` 条 rollout。
- 信号只读取当前 query 已完成轮次的 flow trajectory 和路由，不读取 reward、success 或未来轮次。每个 query 独立重置。
- 第 10 轮只在离线阶段充当自监督 endpoint：先作 oracle 可分性门，再在 Goal 的训练 rollout 上调阈值。Long 的信号、误差和结果均未参与选择或调阈值。
- controller-aware endpoint：前 6 维先乘 action std 回到物理动作尺度再算 RMS；第 7 维夹爪只比较符号。参考带 `0.009267` 是 Goal 固定 stage-8 常加速度的 P95，只表示‘不劣于这个开发集基准’，不是权威控制安全阈值。
- 所有新信号预定义为‘数值越小越稳定’，不扫描反方向；阈值按完整 rollout 留一。

## Oracle gate：信号里到底有没有剩余误差信息

共构造 `47` 个前缀信号；`42` 个 signal×endpoint 组合达到预设门槛：stage 4–8 平均 Spearman `>= 0.10` 且至少 3/5 个 stage 同方向。这里是信息存在性检查，不是可部署成绩。

| 家族 | 信号 | 输出器 | Goal 平均 rho | Long 同方向 rho | Goal 超带 AUC |
|---|---|---|---:|---:|---:|
| candidate_consistency | `candidate_cv_endpoint_spread` | constant_velocity | 0.587 | nan | 0.825 |
| physical_scale | `physical_translation_acceleration_rms` | constant_velocity | 0.511 | 0.463 | 0.726 |
| physical_scale | `translation_endpoint_ensemble_rms` | constant_velocity | 0.511 | 0.463 | 0.726 |
| physical_scale | `cv_temporal_translation_rms` | constant_velocity | 0.511 | 0.463 | 0.726 |
| candidate_consistency | `candidate_ca_endpoint_spread` | constant_velocity | 0.375 | nan | 0.763 |
| candidate_consistency | `candidate_cv_endpoint_spread` | constant_acceleration | 0.375 | nan | 0.754 |
| layer_route | `router_weight_amplitude_layer_cv` | constant_velocity | 0.339 | -0.174 | 0.700 |
| physical_scale | `ca_temporal_translation_rms` | constant_velocity | 0.325 | 0.234 | 0.614 |
| physical_scale | `physical_translation_acceleration_rms` | constant_acceleration | 0.321 | 0.318 | 0.707 |
| physical_scale | `translation_endpoint_ensemble_rms` | constant_acceleration | 0.321 | 0.318 | 0.707 |
| physical_scale | `cv_temporal_translation_rms` | constant_acceleration | 0.321 | 0.318 | 0.707 |
| physical_scale | `physical_acceleration_rms` | constant_velocity | 0.310 | 0.180 | 0.660 |

## Goal rollout 留一

阈值只看其余 3 条 rollout 的 x10 endpoint 误差，并要求训练折零超带；下表测试整条未见 rollout。由于同时探索多个信号，这一列只用于选出一个冻结候选，不能当确认性结果。

| 家族 | 信号 | 输出器 | 最早轮 | 节省 | 超参考带 | score P95 |
|---|---|---|---:|---:|---:|---:|
| candidate_consistency | `candidate_cv_endpoint_spread` | constant_acceleration | 8 | 5.5% | 0.6% (4) | 0.00728 |
| candidate_consistency | `candidate_cv_endpoint_spread` | constant_velocity | 8 | 5.5% | 1.0% (7) | 0.00739 |
| endpoint_consistency | `cv_vs_ca_endpoint_rms` | constant_acceleration | 6 | 4.7% | 0.1% (1) | 0.00635 |
| endpoint_consistency | `cv_vs_ca_endpoint_rms` | constant_acceleration | 8 | 4.7% | 0.1% (1) | 0.00635 |
| endpoint_consistency | `cv_vs_ca_endpoint_rms` | constant_velocity | 6 | 4.5% | 0.1% (1) | 0.00660 |
| endpoint_consistency | `cv_vs_ca_endpoint_rms` | constant_velocity | 8 | 4.5% | 0.1% (1) | 0.00660 |
| candidate_consistency | `candidate_ca_endpoint_spread` | constant_acceleration | 6 | 3.9% | 0.6% (4) | 0.00694 |
| candidate_consistency | `candidate_ca_endpoint_spread` | constant_acceleration | 8 | 3.9% | 0.6% (4) | 0.00694 |
| candidate_consistency | `candidate_ca_endpoint_spread` | constant_velocity | 6 | 3.9% | 1.0% (7) | 0.00671 |
| candidate_consistency | `candidate_ca_endpoint_spread` | constant_velocity | 8 | 3.9% | 1.0% (7) | 0.00671 |
| endpoint_consistency | `endpoint_ensemble_rms__relative_to_early` | constant_acceleration | 6 | 3.1% | 0.3% (2) | 0.00645 |
| endpoint_consistency | `endpoint_ensemble_rms__relative_to_early` | constant_acceleration | 8 | 3.1% | 0.3% (2) | 0.00645 |

### 匹配 20% 计算量的诊断

这一阈值只用训练信号分布对齐节省率，不看训练或测试误差；用于判断动态规则是否真的优于固定第 8 轮。

| 信号 | 输出器 | 实际节省 | 超参考带 |
|---|---|---:|---:|
| `router_weight_amplitude_layer_cv` | constant_acceleration | 19.6% | 4.8% (34) |
| `candidate_cv_endpoint_spread` | constant_acceleration | 19.8% | 4.8% (34) |
| `candidate_ca_endpoint_spread` | constant_acceleration | 19.8% | 4.8% (34) |
| `velocity_change_relative__relative_to_early` | constant_acceleration | 20.0% | 5.0% (35) |
| `physical_acceleration_rms__relative_to_early` | constant_acceleration | 20.0% | 5.0% (35) |
| `endpoint_ensemble_rms__relative_to_early` | constant_acceleration | 20.0% | 5.0% (35) |
| `physical_translation_acceleration_rms` | constant_acceleration | 19.9% | 5.1% (36) |
| `cv_temporal_translation_rms` | constant_acceleration | 19.9% | 5.1% (36) |

## Goal 选择后冻结到 Long

| 选择范围 | 家族/信号 | 输出器 | Goal LOEO | Long 节省 | Long 超参考带 | 6D 物理 RMS（提前样本） | 6D max P95 |
|---|---|---|---:|---:|---:|---:|---:|
| global Goal best | endpoint_consistency / `cv_vs_ca_endpoint_rms` | constant_acceleration | 4.7%, 0.1% 超带 | 6.1% | 0.0% (0) | 0.00438 | 0.02287 |
| Goal best in flow_consistency | flow_consistency / `velocity_change_relative__relative_to_early` | constant_acceleration | 0.8%, 0.6% 超带 | 1.4% | 0.0% (0) | 0.00630 | 0.02552 |
| Goal best in layer_route | layer_route / `route_layer_hellinger_cv` | constant_acceleration | 2.8%, 0.1% 超带 | 2.3% | 0.0% (0) | 0.00568 | 0.02858 |
| Goal best in physical_scale | physical_scale / `ca_temporal_translation_rms` | constant_acceleration | 3.0%, 0.4% 超带 | 6.5% | 0.6% (1) | 0.00496 | 0.02433 |
| 固定对照 | stage 9 | constant_velocity | - | 10.0% | 0.0% (0) | 0.00467 | 0.02330 |
| 固定对照 | stage 8 | constant_acceleration | - | 20.0% | 3.9% (7) | 0.00664 | 0.03469 |

### Controller-aware 固定轮数拆分

| 数据 | 固定输出 | 6D RMS mean | translation RMS P95 | rotation RMS P95 | gripper sign flip | 超参考带 |
|---|---|---:|---:|---:|---:|---:|
| Goal | stage 9 + constant_velocity | 0.00566 | 0.00993 | 0.00216 | 0.0% | 0.0% (0) |
| Goal | stage 8 + constant_acceleration | 0.00746 | 0.01290 | 0.00330 | 0.0% | 5.1% (36) |
| Long | stage 9 + constant_velocity | 0.00467 | 0.00890 | 0.00178 | 0.0% | 0.0% (0) |
| Long | stage 8 + constant_acceleration | 0.00664 | 0.01259 | 0.00255 | 0.0% | 3.9% (7) |

## 裁决

- Goal 选出的全局候选是 `cv_vs_ca_endpoint_rms` + `constant_acceleration`；冻结到 Long 后节省 `6.1%`，超参考带 `0.0%`。这只是 endpoint 等价性筛选，不等价于在线任务成功率。
- 它在 Long 实际只是在 stage 9 停 111/181、其余跑满，节省 6.1%；固定 stage 9 对全部 181 条都停，节省 10.0%，同样 0 条超带。因此动态规则被固定 clock 严格支配，不进入在线 A/B。
- 匹配约 20% 节省时，新信号最多只少 `2` 个 Goal 超带样本（34 对固定 stage 8 的 36），同时节省略低（19.6% 对 20.0%）；这不是有说服力的动态选择增益。
- 原 7D 相对 L2 的 Long stage-8 `28.2%` 不能继续当主结论：controller-aware 指标下同一输出只有 7/181（3.9%）超过 Goal 经验参考带，且夹爪符号翻转为 0。
- 固定对照同时报告 translation、rotation 与 gripper sign。由于仓库里没有可作为权威依据的平移/旋转安全阈值，本报告不把经验参考带包装成控制安全保证。
- endpoint 外推的一致性、flow 离散曲率、translation/rotation/gripper 物理分解、层间路由一致性都已纳入；候选云信号只在 Goal 的 K=16 上诊断，Long 是 K=1，因此不把它伪装成可迁移方法。
- `router_weight_amplitude_*` 只是 top-k 路由权重幅度，不是 `w_e E_e(h)` 的真实专家输出幅度；现有 trace 没保存 expert output，不能用本报告回答真实激活贡献是否能早停。
- Long oracle rho 仅用于事后解释迁移失败或成功，绝未参与候选、方向、轮数和阈值选择。

## 复现

```bash
cd /home/jovyan/work/himoe-vla/himoe-route-capture
python analyze_unsupervised_stop_signals.py \
  --run runs/flow-lead-cpu-t0s24 \
  --holdout-run runs/flow-stop-sweep-long-t08s0-k1-seeds6100-6103 \
  --out-dir analysis/unsupervised-stop-signals-v2
```
