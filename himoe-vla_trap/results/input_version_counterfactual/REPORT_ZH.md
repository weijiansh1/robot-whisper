# 输入版本反事实实验

完整方法、结果和边界见 [`../../docs/INPUT_VERSION_COUNTERFACTUAL_ZH.md`](../../docs/INPUT_VERSION_COUNTERFACTUAL_ZH.md)。

核心结果：同噪声自然比值在失败抓取前一个 query 为 `0.539 < 1`，7 条成功对照为 `1.593--2.132`；但原始图像和 8 维状态变化也都低于全部成功对照。抓空后的 `+1/+2`，视觉与 proprioception 对后层 action routing 的效应反而高于全部成功对照。因此它是一个 pre-grasp feedback-under-noise 信号，不是“抓空后反馈被 MoE 截断”的证据。

正式主采集：96 个版本对、768 次推理。2×2 模态采集：40 个版本对、320 次推理。全部 `training=false`。
