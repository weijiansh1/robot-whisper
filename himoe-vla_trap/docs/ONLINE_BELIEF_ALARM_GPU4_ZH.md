# GPU 4 在线 belief-state 报警实验

## 核心结论

在线链路已经跑通，但当前 `transition_split_v1` **不能作为可靠的独立报警器部署**。

在 4 条全新随机轨迹中：

- 1 条真实 failed-grasp 失败轨迹被报警；
- 1 条成功抓取轨迹也被报警，是假阳性；
- 另外 2 条失败轨迹从未发生“靠近目标锅的夹爪闭合”，不在这个抓取后选择器的适用窗口内，也没有报警。

因此，回答“给一条随机轨迹，能否只靠当前 MoE 规则可靠识别这种失败并报警”：**目前不能保证**。它能在线捕获一个真实 failed-grasp，但也会误报正常抓取，而且不覆盖抓取尝试之前的失败。

## 实验约束

- 推理设备：GPU 4，NVIDIA H20-3e。
- 任务：LIBERO-10 task 8，`put both moka pots on the stove`。
- 路由输入：每次重规划实时返回 `8 layers x 10 flow steps x 11 suffix tokens x 32 experts` 的完整 HB gate probability。
- 规则：冻结的 `transition_split_v1`；阈值来自此前 7 条 A100 同阶段健康轨迹。
- 训练：无训练；看到在线结果后没有改阈值。
- 报警时机：模型已经生成当前 action chunk、机器人尚未执行该 chunk。
- 报警之后：当前 chunk 与所有后续 chunk 原样执行，直到 success 或 520-step horizon。
- 物理位移、任务 success 和未来观测只做事后判断，未输入报警器。

## 全新随机结果

| episode | init | outcome | 近锅闭合 | 在线报警 | 解释 |
|---|---:|---|---:|---|---|
| 0 | 7 | success, 382 steps | q30 | q31 (+1) | 假阳性；锅已被夹住并随末端移动 |
| 1 | 3 | failure, 520 steps | 无 | 无 | 未进入抓取后判定窗口 |
| 2 | 20 | failure, 520 steps | 无 | 无 | 未进入抓取后判定窗口 |
| 3 | 0 | failure, 520 steps | q42 | q43 (+1), q45 (+3), q46 (+4) | failed-grasp；q45/q46 物理脱耦明确 |

成功 episode 0 在闭合后：

$$
\max \Delta p_{pot}=41.02\text{ cm},\qquad
\max \Delta z_{pot}=14.56\text{ cm},\qquad
\max d(EEF,pot)=7.84\text{ cm}.
$$

报警时锅仍随末端运动，最终完成任务，因此这是明确假阳性。

失败 episode 3 在闭合后：

$$
\max \Delta p_{pot}=0.00082\text{ cm},\qquad
\max \Delta p_{EEF}=43.64\text{ cm},\qquad
\max d(EEF,pot)=49.67\text{ cm}.
$$

在 q43 首报时机械臂刚闭合，还没有足够物理运动证明抓空；到 q45 和 q46，末端分别已经离开闭合点 10.65 cm 和 18.30 cm，而锅几乎完全没动。这两个持续报警与 belief-state failed-grasp 一致。该 episode 最终失败，并在首报后继续执行了 90 个动作。

## 先看这两个视频

红框是在线报警的 chunk；黄色文字是近锅夹爪闭合；`AFTER ALARM` 表示报警后仍原样继续推理和执行。

- 真 failed-grasp，报警短片：[episode 3 annotated clip](../results/online_belief_alarm/fresh_random_gpu4_20260904/episode_003_init_00_random/videos/alarm_annotated_clip.mp4)
- 真 failed-grasp，完整轨迹：[episode 3 annotated full](../results/online_belief_alarm/fresh_random_gpu4_20260904/episode_003_init_00_random/videos/alarm_annotated_full.mp4)
- 假阳性成功抓取，报警短片：[episode 0 annotated clip](../results/online_belief_alarm/fresh_random_gpu4_20260904/episode_000_init_07_random/videos/alarm_annotated_clip.mp4)
- 假阳性成功抓取，完整轨迹：[episode 0 annotated full](../results/online_belief_alarm/fresh_random_gpu4_20260904/episode_000_init_07_random/videos/alarm_annotated_full.mp4)

每个目录同时保存无文字的 `alarm_clean_full.mp4`，可用于排除标注遮挡。

## H20 重放开发噪声

旧 candidate 06 的 52 个 flow-noise tensor 也在 GPU 4 上重新在线推断。H20 轨迹没有复现旧 A100 结果：闭合从旧数据的 q31 提前到 q29，最终在第 377 步成功；规则在 q34 (+5) 报警，属于假阳性。这是硬件数值分叉和跨设备阈值漂移的直接证据，不计作新的独立测试。

- 重放报警短片：[candidate 06 H20 replay clip](../results/online_belief_alarm/replay_candidate06_gpu4_20260903/episode_000_init_00_replay/videos/alarm_annotated_clip.mp4)
- 重放完整轨迹：[candidate 06 H20 replay full](../results/online_belief_alarm/replay_candidate06_gpu4_20260903/episode_000_init_00_replay/videos/alarm_annotated_full.mp4)

## 完整性审计

本次服务共处理 233 次重规划：

$$
38\text{ (replay)} + 39 + 52 + 52 + 52\text{ (fresh)} = 233.
$$

客户端每次在线收到并用于判定的 float16 full-route，与服务端 `routes.zarr` 中 233 行数据逐元素完全一致。服务端 hook 完成 480 次自检，失败为 0。所有报警 query 的当前 chunk 都有实际执行记录。

机器可读结果：

- [summary.json](../results/online_belief_alarm/summary.json)
- [episode_summary.csv](../results/online_belief_alarm/tables/episode_summary.csv)
- [alarm_events.csv](../results/online_belief_alarm/tables/alarm_events.csv)
- [fresh random manifest](../results/online_belief_alarm/fresh_random_gpu4_20260904/manifest.json)
- [server capture summary](../results/online_belief_alarm/gpu4_server_20260903/capture_summary.json)

## 代码

- 在线采集与视频标注：[collect_online_belief_alarms.py](../code/collect_online_belief_alarms.py)
- 健康参考与阈值生成：[evaluate_trainfree_belief_selector.py](../code/evaluate_trainfree_belief_selector.py)
- 汇总和一致性审计：[summarize_online_belief_alarm.py](../code/summarize_online_belief_alarm.py)
- 冻结配置：[online_belief_alarm_gpu4.json](../configs/online_belief_alarm_gpu4.json)

## 当前判断

这个结果支持：MoE 路由可在真实 failed-grasp 轨迹中产生在线可见的异常，并且报警可以在物理脱耦完全显现之前出现。

这个结果不支持：当前健康最大值阈值能泛化为可靠检测器。最直接的问题是跨 init-state 和跨 GPU 的健康分布漂移。下一版仍可保持 train-free，但至少需要当前硬件、多个 init-state 的健康 calibration，并要求连续两次或带 margin 的持久报警；这些新规则必须在另一批冻结随机轨迹上评估，不能用本次 4 条轨迹调完再报本次性能。
