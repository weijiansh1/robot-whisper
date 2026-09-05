# 失败轨迹延时续跑与标签清洗报告

## 结论

对当前在线报警研究使用的两套数据中的全部 1,051 条原始失败轨迹，精确重放原轨迹后继续闭环执行最多 10 个 VLA control queries：

- 71 条在延长窗口内成功，占原失败的 6.76%，应视为 horizon-censored late success；
- 980 条在延长窗口内仍未成功，作为清洗后的失败保留；
- 如果“多 10 个控制步”特指 10 个低层 LIBERO actions，即只多 1 个 VLA query，则只有 7 条成功，占 0.67%。

项目数据中的一个 `control_step` 是一次 VLA query 和一个包含 10 个低层动作的 action chunk。本报告以 `+10 queries`（最多 100 个低层动作）为主口径，同时保留 `+1 query`（最多 10 个低层动作）口径。

| 数据集 | 原失败 | +1 query 成功 | +10 queries 成功 | 清洗后失败 |
| --- | ---: | ---: | ---: | ---: |
| development_main（14,800 条） | 487 | 5（1.03%） | 30（6.16%） | 457 |
| external_8b（15,600 条） | 564 | 2（0.35%） | 41（7.27%） | 523 |
| 合计 | 1,051 | 7（0.67%） | 71（6.76%） | 980 |

## Suite 分布

| Suite | 原失败 | +1 query 成功 | +10 queries 成功 | 清洗后失败 |
| --- | ---: | ---: | ---: | ---: |
| goal | 208 | 1（0.48%） | 1（0.48%） | 207 |
| long | 496 | 3（0.60%） | 44（8.87%） | 452 |
| object | 81 | 2（2.47%） | 10（12.35%） | 71 |
| spatial | 266 | 1（0.38%） | 16（6.02%） | 250 |

慢成功具有明显任务依赖。数量最多的是 `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`（33/289）；比例最高的主要任务是 `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`（13/22，59.09%）。goal 中只有一条 `put_the_bowl_on_the_plate` 在原 horizon 后第一个低层动作成功。

首次成功分布在新增 query 1 到 10：`7, 14, 4, 3, 4, 8, 8, 7, 3, 13`。有 13 条直到第 10 个新增 query 才成功，因此对更长延时窗口而言，71 条是一个右截断的下界；但对本次冻结的 `+10 queries` 定义而言，标签是确定的。

## 方法

没有从头重新采样失败轨迹。每条轨迹执行以下过程：

1. 使用相同 task、initial state、environment seed、settle steps 和 `paper-right` wrist layout 重建环境。
2. 逐动作重放保存的原始 action prefix；在每个原 query 边界比较 simulator state 和 policy state。
3. 仅当两个 float32 状态与源 NPZ 逐位相同、且原 horizon 前始终未成功时才允许续跑。
4. 从原 `flow_noise_seed` 恢复 NumPy RNG，消费原 query 数量的随机数后，使用后续随机数调用相同 checkpoint。
5. 最多新增 10 个 action chunks，并在每个低层动作后检查 LIBERO success condition，首次成功即停止。

这样测量的是同一条已保存失败轨迹的真实 continuation，而不是跨 GPU 从头重跑导致的另一条随机 rollout。

## 完整性审计

- 正式续跑：1,051/1,051 完成，运行错误 0；
- exact prefix replay：1,051/1,051；
- simulator state 最大误差：0.0；
- policy state 最大误差：0.0；
- 源 NPZ 哈希不匹配：0；
- continuation array 哈希不匹配：0；
- checkpoint 与 wrist-layout identity 不匹配：0；
- 原始数据未覆盖，原始正常样本被改标签的数量：0。

GPU5 上原有进程未停止或清除；本实验启动的 server/client 已在流水线结束后正常退出。

## 对在线报警结果的影响

清洗后，部分原来的“正确报警”变成了对慢成功的误报，因此 precision 下降。这说明原失败标签确实让此前报警结果偏乐观，也说明路由变化报警器会把一部分慢但最终成功的轨迹识别为异常。

| 数据集与报警器 | 原 precision | 清洗后 precision | 原 recall | 清洗后 recall | 被报警的慢成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| main, mobility_w4_k4 @ 0.95 | 89.39% | 83.84% | 36.34% | 36.32% | 11/30 |
| main, mobility_w4_k4 @ 0.975 | 95.38% | 87.69% | 25.46% | 24.95% | 10/30 |
| external, shared mobility_w4_k4 @ 0.95 | 86.70% | 77.98% | 33.51% | 32.50% | 19/41 |
| external, independently sealed @ 0.99 | 58.06% | 52.69% | 19.15% | 18.74% | 10/41 |

后续若目标是“成功尽量不报警、失败尽量报警”，应以这套清洗标签重新选择阈值或构造 progress-aware veto；不能继续把 late success 当作正类来优化报警器。

## 产物

- 冻结协议：`TIMEOUT_EXTENSION_PROTOCOL.md`
- 全部 continuation：`results/timeout_extension_plus10/continuation_results.csv`
- 应从失败中剔除的 71 条：`results/timeout_extension_plus10/late_successes.csv`
- 清洗后保留的 980 条失败：`results/timeout_extension_plus10/remaining_failures.csv`
- 主集清洗标签：`results/timeout_extension_plus10/development_main_clean_labels.csv`
- 外部集清洗标签：`results/timeout_extension_plus10/external_8b_clean_labels.csv`
- 按 cohort、suite、task 和延长步数汇总：`results/timeout_extension_plus10/summary_by_*.csv`、`late_success_curve.csv`
- 清洗后的报警指标：`results/timeout_extension_plus10/cleaned_alarm_metrics.csv`
- 机器可读总览与校验哈希：`results/timeout_extension_plus10/summary.json`

“清洗后失败”只表示在额外 10 个 VLA queries 内仍未成功，不等价于已经证明它是某一种语义 Trap。
