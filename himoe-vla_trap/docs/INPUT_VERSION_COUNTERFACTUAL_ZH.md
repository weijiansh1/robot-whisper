# 同噪声输入版本反事实：MoE 到底有没有接收抓空后的反馈？

## 核心结论

这轮实验推翻了一个过强的解释：**该失败抓取不是因为新观测进入 state token 后，在前后层之间被 MoE 路由“截断”了。**

更准确的时间顺序是：

1. 抓取 chunk 前一轮（`relative=-1`），失败轨迹的相邻图像和 8 维状态本来就变化很小，因而 state/action routing 与动作输出都近似不变。
2. 抓取 chunk（`0`）的动作对 flow seed 异常敏感，但不存在清晰的逐层 state-to-action 路由断裂。
3. 抓空后的第一个新输入（`+1`）到来时，图像与 8 维状态对后层 action routing 的影响都高于 7/7 成功对照；动作输出也发生了强变化。
4. 因此模型不是“没看到物理变化”。它接收并放大了变化，但产生的动作语义/闭环纠错仍然错误。MoE 路由能显示计算路径变了，不能证明内部 belief 的语义是对还是错。

## 实验设计

目标案例是 snapshot 0、candidate 06：query 31 的第 8 个控制动作闭合夹爪，末端距目标锅约 `0.0977 m`，此后锅的最大抬升为 `0`。对照是同一 simulator snapshot 的 7 条成功 sibling，分别按各自真实抓取闭合时刻对齐。

每个相邻输入版本对 (V_{t-1},V_t) 使用完全相同的 flow noise：

\[
R^-_m=R(V_{t-1},\xi_m),\qquad R^+_m=R(V_t,\xi_m).
\]

正式采集覆盖 8 条轨迹、`relative=-3..+8`、每点 4 个 noise，共 96 个版本对、768 次 GPU 6 推理。报警分数只读取后层 12--15、完整 10 个 flow、10 个 action token 的 HB 概率：

\[
d_V=\operatorname{median}_m H(R^-_m,R^+_m),
\]

\[
d_\xi=\operatorname{median}_{V,i<j}H(R(V,\xi_i),R(V,\xi_j)),
\]

\[
C_t=\frac{d_V}{d_\xi}.
\]

自然的 train-free 边界为 (C_t<1)：改变随机 seed 对 action routing 的影响，大于输入从旧版本更新到新版本的影响。没有拟合参数、正常轨迹库、任务 ID、动作值或物理距离。

随后又做了 2×2 因子反事实，共 40 个版本对、320 次推理：

| 输入 | 图像 | 8 维状态 |
|---|---|---|
| `V00` | 旧 | 旧 |
| `V10` | 新 | 旧 |
| `V01` | 旧 | 新 |
| `V11` | 新 | 新 |

四种输入使用相同 noise，从而把视觉与 proprioception 对 MoE 的因果影响分开。

## 主结果

| 相对抓取 query | 失败 (C_t) | 7 条成功范围 | 自然阈值报警 |
|---:|---:|---:|---:|
| -3 | 1.208 | 0.945--1.412 | 否 |
| -2 | 1.168 | 1.115--1.382 | 否 |
| **-1** | **0.539** | **1.593--2.132** | **是，成功 0/7** |
| 0 | 1.230 | 1.581--1.835 | 否 |
| +1 | 1.967 | 1.787--2.237 | 否 |
| +2 | 2.170 | 1.341--1.875 | 否 |

所以这条失败轨迹能在实际夹爪闭合前一个 query 被自然边界命中。但这还不是可靠 detector：在完整 `-3..+8` 窗口中，成功记录有 `4/84` 次报警，并涉及 `3/7` 条成功轨迹。只用 2 个 noise、4 次推理的低成本版本更差，成功记录报警率为 `8/84`，涉及 `5/7` 条成功轨迹。

现有 `self_reference_coupling_collapse_v3` 在同一失败上直到 `+12` 才报警，7 条成功均不报。故新分数的真实取舍是：**更早，但每次需 8 次 shadow inference，且有限对照上的 episode specificity 更差。**

## 为什么 `-1` 不是“内部反馈断裂”

在 `-1`：

| 量 | 失败 | 7 条成功范围 | 关系 |
|---|---:|---:|---|
| base 图像 MAE（uint8） | 1.757 | 4.307--7.424 | 低于全部 |
| wrist 图像 MAE | 3.653 | 5.181--10.122 | 低于全部 |
| 8 维状态 L2 变化 | 0.0235 | 0.0333--0.0628 | 低于全部 |
| 后层 action route 的视觉效应 | 0.0113 | 0.0284--0.0373 | 低于全部 |
| 后层 action route 的状态效应 | 0.00310 | 0.00213--0.00370 | 正常范围 |

但按输入变化幅度归一化后：

- 视觉 route / pixel MAE 为 `0.00417`，落在成功的 `0.00338--0.00722` 内；
- proprio route / state L2 为 `0.1321`，反而高于成功的 `0.0415--0.0687`。

因此 MoE 没有压制一份强反馈；输入本身就近乎静止。该报警更像 **pre-grasp quiescence / stale action phase**，不是 belief semantic mismatch 的直接证据。

## 抓空后模型是否“看见了”？

看见了。`+1` 时：

- 后层 action routing 的纯视觉效应 `0.03724`，高于成功范围 `0.02342--0.03371`；
- 纯 8 维状态效应 `0.03944`，高于成功范围 `0.02437--0.03120`；
- 动作输出版本效应 RMS `0.56475`，高于成功范围 `0.17957--0.25037`。

`+2` 时两种 routing 模态效应仍都高于全部成功对照。原先“state route 变了、action route 仍像成功模板”的观察是真实的相对模板差异，但不能推出“新输入没有传到动作计算”。同噪声反事实说明新输入已经明显改变 action routing 和 action output。

## action-only 对照

MoE 不是唯一信号：

- `-1` 时失败动作 chunk 的版本效应 RMS 仅 `0.0229`，低于成功的 `0.4760--0.6758`；
- `0` 时失败动作的跨 noise RMS 为 `0.3340`，远高于成功的 `0.0153--0.0191`；
- `+1` 时动作版本效应高于全部成功。

所以在这个案例里，动作输出本身已经暴露出“抓取前计划没有切换”和“抓取 chunk 高度 seed-sensitive”。当前证据不能支持 MoE 比 action-only 更本质，只能说 MoE 的无量纲 (C_t<1) 在该案例上给出一个有物理含义的自然边界。

## 架构发现

前层 state token 的跨 noise 路由差异中位数严格为 `0`，全数据最大仅 `3.46e-4`。原因是 state token 位于 noisy action tokens 之前，在当前 causal mask 下基本不受后续 action noise 影响。因此最初设想的“state 输入效应 / state 噪声效应”会除以近零并退化，不能作为报警分数。

## 能证明与不能证明

目前可以证明：

- 同噪声条件下，失败抓取前一个 query 的后层 action routing 对输入更新的响应小于对 flow noise 的响应；
- 这个 `-1` 异常覆盖全部 8 个 HB 层，并在 7 条阶段对齐成功对照之外；
- 抓空后的新视觉和本体状态都已经进入并强烈改变 action routing；
- 全流程没有训练，也没有用正常库、任务身份或物理距离形成分数。

目前不能证明：

- 这是可泛化的 belief-state failure classifier；这里只分析了 1 条失败；
- MoE 路由能读出“模型相信已经抓住锅”这一语义；路由只有计算路径，没有 expert 输出内容；
- 该规则可直接部署；8 次推理昂贵，且 3/7 成功轨迹在窗口其他位置至少误报一次；
- MoE 优于 action-only；本案例的 action counterfactual 也很强。

最稳妥的本质表述是：

> **MoE routing 能测量动作计算相对于外部反馈和内部采样噪声的敏感度，但不能单独解码 belief 的语义。这个失败在抓取前出现短暂的 feedback-under-noise 状态；抓空后反馈已经进入网络，失败发生在如何解释并利用反馈，而不是反馈是否进入。**

## 产物

- 主结果：[`results/input_version_counterfactual/analysis/summary.json`](../results/input_version_counterfactual/analysis/summary.json)
- 主图：[`counterfactual_feedback_transmission.png`](../results/input_version_counterfactual/analysis/figures/counterfactual_feedback_transmission.png)
- 模态分解：[`modality_analysis/summary.json`](../results/input_version_counterfactual/modality_analysis/summary.json)
- 原始输入审计：[`input_novelty_summary.json`](../results/input_version_counterfactual/modality_analysis/input_novelty_summary.json)
- 输入版本图：[`failed_input_version_pairs.png`](../results/input_version_counterfactual/modality_analysis/figures/failed_input_version_pairs.png)
- 动作对照：[`action_output_counterfactual.csv`](../results/input_version_counterfactual/modality_analysis/tables/action_output_counterfactual.csv)
- 正式同噪声原始路由：[`paired_counterfactual_routes.npz`](../results/input_version_counterfactual/formal_capture/paired_counterfactual_routes.npz)
- 2×2 原始路由：[`modality_counterfactual_routes.npz`](../results/input_version_counterfactual/modality_capture/modality_counterfactual_routes.npz)
- 冻结规则：[`counterfactual_open_loop_alarm_v1.json`](../configs/counterfactual_open_loop_alarm_v1.json)

