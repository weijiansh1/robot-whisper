# 文献核对

核对日期：2026-09-15。下面仅区分外部已有依据与本轮仍须验证的假设，不把其他模型的结果当作 HiMoE 实验结果。

| 论点 | 一手来源 | 核对结果 |
| --- | --- | --- |
| 生成长度与实际执行长度可以分开，丢弃后缀后重规划 | [AutoHorizon](https://arxiv.org/html/2602.21445v1)，3.1、3.2、8.4 节 | 支持接口。但其 p=10 的结果常在完整执行端达到最佳；不能据此认定 HiMoE 的 10→5 必有收益。 |
| 极短执行长度可能犹豫或停滞 | [AutoHorizon](https://arxiv.org/html/2602.21445v1)，真实实验段落 | 有对应观察，属于其模型和任务条件。5 步、20 步窗口仍只是本轮待测设置。 |
| 内部特征可用于失败检测 | [SAFE，NeurIPS 官方摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html) | 支持失败预测任务，不提供本控制器的干预收益估计。 |
| 终局成功置信度与执行方案改变后的成功差不同 | [VLAConf](https://arxiv.org/abs/2605.29605)、[作者项目页](https://sites.google.com/view/vlaconf/home) | 其目标是从已观测前缀估计最终成功；不能直接解释为改变执行长度的收益。 |
| 部分错误需要恢复技能，不只是重复调用原策略 | [FLARE，CVPR 官方页面](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html) | Retry/Reset 框架包含恢复数据与技能训练，支持划分恢复能力边界，不证明当前冻结策略具备这些技能。 |
| 冻结策略不等于免费物理回退 | [CoRe](https://arxiv.org/html/2608.14822v1)，Counterfactual / Realignment 方法章节 | 涉及合成观测、反事实延续和机器人/场景重对齐；不能把本轮模拟器重建当作部署时已有的回退能力。 |

本地已由代码和模拟器预检确认的只是控制接口可实现且不会改变同一动作序列的执行含义。终局收益、适用状态和 v8.2 时机价值必须分别由配对实验回答。

另一个仍未被两臂实验单独识别的因素是新观测与额外策略调用的贡献。物理时间噪声配对可以消除随机流错位，但不能替代旧观测消融；本轮不将任何成功直接归因为某个 MoE 专家或单一反馈机制。
