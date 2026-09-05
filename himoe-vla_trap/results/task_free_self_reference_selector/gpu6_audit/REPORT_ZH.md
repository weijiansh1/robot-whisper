# 无任务先验 MoE 自参照报警：GPU6 审计

## 核心结论

这是一条 **train-free、运行时只读当前 episode 自身 HB MoE 路由前缀** 的窄型报警规则。它不读取任务 ID、正常轨迹库、动作、物理距离、reward、success 或 timeout。阈值查看过一个开发任务，因此不是开发阶段 label-free。

GPU6 前瞻性固定扫描共 16 条：15 条成功全部不报警，1 条失败中 1 条报警；唯一失败在 q20 首次报警并继续执行 100 个 control steps，未自行恢复。样本只有 1 个失败，不能把 100% recall/precision 当成总体性能。

## 报警到底发生在什么时候

失败轨迹中的盘子并非完全没动。按事后物理轨迹定义：盘子已移动至少 5 cm，且此后总 excursion 不超过 1 mm，则 plateau 从 q16 开始。MoE 在 q20 报警，晚 4 个 query；同 FPR 档的 phase-0.81 固定时钟在 q24 才响，因此 MoE 比时钟早 4 个 query。

所以当前正确表述是：**错误物理状态稳定后，MoE 出现持续的 internal response collapse**。它不是这条轨迹的事前 precursor，也不能仅凭 routing 判定具体失败原因。

报警瞬间三个无量纲量为：state response `0.194`（阈值 <=0.25）、action planning churn `1.211`（阈值 >=1.10）、state/action gap `1.596`（阈值 >=1.20）。

## 大规模离线边界

严格排除开发任务后，39 个未参与阈值设定的任务共有 15600 条：failure recall `78/530 = 14.72%`，success FPR `35/15070 = 0.232%`，precision `69.03%`。这是一个高精度、低召回的 phenotype head，不是通用失败检测器。

更重要的是，endpoint success/timeout 标签被 horizon 严重混杂：phase-0.81 固定时钟在 held-out 数据上达到 100% failure recall，FPR 约 0.21%。因此后续主评估必须改成物理或语义事件 onset，而不能继续只比较最终 success。

## 同 seed 跨 GPU 复现边界

旧 GPU3 的两条失败条件在 GPU6 上都成功。两端 checkpoint、代码 commit、diff hash 和 wrist layout 完全一致，q0 policy/sim state 也逐值相同；但 q0 action MAE 为 `0.000690`--`0.000813`，q0 route MAE 为 `0.000106`--`0.000109`。这些小数值差异在接触闭环中累积，改变了 outcome。因此这里应称为 **same-seed cross-GPU divergence audit**，不能称为精确失败复现。

## 文件

- `online_episodes.csv`：GPU6 16 条逐 episode 在线结果
- `physical_timing.csv`：物体 plateau、MoE alarm 与时钟的事后对齐
- `same_seed_cross_gpu.csv`：GPU3/GPU6 两个同 seed 条件的 q0 差异
- `gpu6_self_reference_audit.png`：在线结果、三项 MoE 比率和物理时序
- `summary.json`：机器可读汇总

报警视频位于：

- `../gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/alarm_clean_full.mp4`
- `../gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/alarm_annotated_full.mp4`
- `../gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/alarm_annotated_clip.mp4`
