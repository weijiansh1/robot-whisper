# 不使用物理距离的 MoE-only 在线报警实验

> 本文记录旧 v1 规则的失败定位。根据该诊断构造的 route-acceleration v2 已完成独立新 seed 在线验证；结果见 [MoE dynamics v2 报告](MOE_DYNAMICS_ALARM_V2_ZH.md)。

## 核心结论

卡 5 新跑了 24 条固定长度在线轨迹，选择器沿用卡 4 后冻结的 H20 健康参考和 `transition_split_v1 + persistence=2`，运行中没有调规则，也没有用物理距离、动作、reward 或 success。

结论不是“MoE 完全没用”，而是：

1. **当前三条件在线规则不适合识别 belief-state missed-grasp trap。** 卡 5 的 5 条严格目标失败只报出 1 条，且晚于漏抓 7 个 query。
2. **漏报不只是 persistence 太严格。** 改成任意一次 raw reject 虽把目标覆盖提高到 3/5，同时也触发 6/9 个其他失败和 2/10 个成功；`2-in-5`、`2-in-10` 等放宽规则仍只有 1/5 目标覆盖。
3. **MoE telemetry 本身仍含失败后的信息。** 事后对齐漏抓事件后，后层完整去噪 route acceleration 在事件后 5 个 query 内区分 5 条目标失败与 10 条成功抓取，AUC=1.00，48 项 maxT 校正后 `p=0.0118`。
4. **目前没有经验证的提前信号。** 冻结三条件在漏抓前 5 个 query 的 AUC=0.42；最好的事后候选 phase-match distance 为 AUC=0.86，但 maxT 校正后 `p=0.351`，不能据此发布新报警器。

因此最准确的判断是：**现有 feature conjunction / phase-matching 规则错了；“MoE 可作为 belief-trap sensor”没有被否定，但现在只验证到事后报警信号，还没有验证出提前报警器。**

## 卡 5 在线结果

- GPU：物理卡 5，NVIDIA H20-3e
- seed：`20260907`
- 24 条轨迹，每个 init `0/3/7/20` 各 6 条
- 10 成功、14 失败
- 3 条正式报警，全部发生在最终失败轨迹；10 条成功中 0 次正式报警
- 所有报警后继续执行原 action chunk，没有干预策略
- 服务端保存 1114 次推理的完整 HB routing，shape 为 `[1114,8,10,11,32]`
- 客户端与服务端逐元素完全相同，episode ID 完全相同，hook failure 为 0

| 事后类型 | 轨迹数 | 至少一次 raw reject | 正式 alarm |
|---|---:|---:|---:|
| 严格漏抓后继续执行代理 | 5 | 3/5 | **1/5** |
| 其他失败/Trap | 9 | 6/9 | **2/9** |
| 成功 | 10 | 2/10 | **0/10** |

把所有 14 条失败合在一起，正式 failure recall 是 3/14（21.4%，Wilson 95% CI 7.6%--47.6%）。观察到的成功 false-positive rate 是 0/10，但其 Wilson 上界仍为 27.8%；样本很小，不能把 0 次观察误报解释为高可靠性。

## 三次报警到底报到了什么

| episode/init | alarm q | 事后归因 | 与相关事件的时序 | 结论 |
|---|---:|---|---|---|
| 11/3 | 11 | 其他失败，最终进入 loop | q11 当下第二口锅确实随手移动；随后 q29 附近重新操作 | 是失败轨迹报警，但不是漏抓报警，机制归因不确定 |
| 14/20 | 44 | 严格 missed-grasp proxy | 第一口锅在 q37 漏抓，报警晚 7 query | 命中目标类型，但太晚，不能算 early warning |
| 22/3 | 35 | 15 mm 敏感性边界 missed-grasp | q33 闭爪后锅仅移动 11.79 mm、无抬升，报警晚 2 query | 主阈值下归为其他失败；视频显示锅已倒伏，属于边界案例 |

主标签要求目标总位移 `<10 mm`。若仅在事后审计中把该阈值放宽到 `<15 mm`，GPU5 目标组变为 8 条，其中 2/8 正式报警；这不改变在线输入，也不改变主结果。

## 报警视频

红框是正式 MoE-only alarm；黄色框只是单次 raw reject。每次报警后 chunk 仍原样执行，因此视频同时显示报警时刻和后续结果。

- [ep11：其他失败，q11 报警](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/episode_011_init_03_random/videos/alarm_annotated_full.mp4)
- [ep11：报警附近短片](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/episode_011_init_03_random/videos/alarm_annotated_clip.mp4)
- [ep14：严格漏抓型，q44 报警](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/episode_014_init_20_random/videos/alarm_annotated_full.mp4)
- [ep14：报警附近短片](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/episode_014_init_20_random/videos/alarm_annotated_clip.mp4)
- [ep22：15 mm 边界型，q35 报警](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/episode_022_init_03_random/videos/alarm_annotated_full.mp4)
- [ep22：报警附近短片](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/episode_022_init_03_random/videos/alarm_annotated_clip.mp4)

clean 版本和文件哈希记录在 [GPU5 manifest](../results/moe_only_online_alarm/gpu5_extended_random_seed20260907/manifest.json)。

## persistence 排除实验

下面只重算相同 raw reject 序列的时序组合，不训练模型、不改变特征阈值：

| GPU5 规则 | 目标漏抓 | 其他失败 | 成功触发 |
|---|---:|---:|---:|
| 任意一次 raw | 3/5 | 6/9 | 2/10 |
| 连续 2 次（在线正式规则） | 1/5 | 2/9 | 0/10 |
| 5 query 内 2 次 | 1/5 | 5/9 | 1/10 |
| 10 query 内 2 次 | 1/5 | 5/9 | 1/10 |
| 10 query 内 3 次 | 1/5 | 4/9 | 0/10 |

如果问题只在 persistence，那么 `2-in-5` 或 `2-in-10` 应该明显恢复目标召回；实际没有。它们主要增加了其他 Trap 的触发。这说明目标漏抓没有稳定地产生当前三条件 conjunction，而不只是两个异常点没有紧挨在一起。

## MoE 特征诊断

严格目标组以漏抓 query 为 0；成功组以真正搬起目标的闭爪 query 为 0。每条 episode 在指定窗口内取该特征最大值，再计算 episode-level AUC。GPU5 上共比较 12 个特征 x 4 个窗口，使用 5000 次整条 episode 标签置换和 48 项 maxT 校正。

| 窗口 | 特征 | AUC | 单边 p | maxT p | 可以怎么说 |
|---|---|---:|---:|---:|---|
| 漏抓前 q-5..q-1 | 冻结三条件 margin | 0.42 | 0.666 | 1.000 | 旧规则没有前兆分离 |
| 漏抓前 q-5..q-1 | phase-match distance | 0.86 | 0.0116 | 0.351 | 候选现象，校正后不显著 |
| 漏抓当下 q0 | back late-flow WJ | 0.92 | 0.0044 | 0.124 | 当下候选，仍未通过校正 |
| 漏抓后 q+1..q+5 | back route acceleration | **1.00** | 0.00080 | **0.0118** | 明确的 post-event MoE signal |

这里最重要的方向信息是“后层 + 完整去噪动力学”，不是旧规则依赖的 layer-5 gap、跨 chunk jump 和前层 action nominality 三者同时越界。后层 acceleration 的数值间隔很窄，而且是看完标签后发现的，必须先冻结新的健康校准规则，再用全新 seed 做 prospective 在线测试。

## 三批合并审计

卡 4 preliminary、卡 4 held-out 和卡 5 extended 共 40 条轨迹：18 成功、22 失败。

| 事后类型 | 轨迹数 | 至少一次 raw reject | 正式 alarm |
|---|---:|---:|---:|
| 严格漏抓代理 | 9 | 7/9 | **1/9** |
| 其他失败/Trap | 13 | 9/13 | **2/13** |
| 成功 | 18 | 7/18 | **0/18** |

三批一共保存 349 + 381 + 1114 = 1844 行完整路由，三批客户端/服务端检查均完全一致。合并统计只用于描述稳定性；卡 4 preliminary 曾用于建立健康参考，因此不能把 40 条全部称为独立 held-out test。

## 在线规则与标签边界

在线每个 query 只接收：

```text
HB routing: [8 layers, 10 flow steps, 11 suffix tokens, 32 experts]
```

它只使用当前/上一 query 的 HB routing、已知成功轨迹参考库，以及前层 action-token routing 的 closed-begin monotone DTW。健康参考和经验阈值属于 calibration；没有梯度优化或拟合分类器，因此是 train-free，但不是 zero-reference。

严格漏抓代理只在 rollout 结束后计算：闭爪时距任一 moka pot `<0.16 m`，此后 EEF 位移 `>=0.10 m`、目标总位移 `<0.01 m`、两者最大分离 `>=0.15 m`。数据没有 contact/force 或显式 belief variable，所以它只能叫 `kinematic missed-grasp-then-departure proxy`。

## 可复算文件

- [失败类型审计摘要](../results/moe_only_online_alarm/failure_type_audit/summary.json)
- [逐 episode 审计](../results/moe_only_online_alarm/failure_type_audit/episode_audit.csv)
- [逐闭合事件审计](../results/moe_only_online_alarm/failure_type_audit/grasp_events.jsonl)
- [方法诊断摘要](../results/moe_only_online_alarm/method_diagnosis/summary.json)
- [时序规则诊断](../results/moe_only_online_alarm/method_diagnosis/episode_rule_diagnostics.csv)
- [事件窗口逐 episode 分数](../results/moe_only_online_alarm/method_diagnosis/event_window_scores.csv)
- [全部 AUC 与置换检验](../results/moe_only_online_alarm/method_diagnosis/event_auc.csv)
- [GPU5 服务端 capture 摘要](../results/moe_only_online_alarm/gpu5_server_seed20260907/capture_summary.json)

## 后续状态

下面的严格步骤已由 v2 实验执行：健康 reference 冻结阈值后，在 seed `20260908` 固定运行 24 条。v2 将同轨迹严格目标召回从 1/7 提高到 5/7、全部失败召回从 1/14 提高到 8/14，但引入 1/10 成功误报。因此它是有效修正，还不是最终 detector。

## v2 采用的严格判定标准

v2 按以下预先约束执行，用来把“MoE 里有 post-event signal”升级为可检验的在线报警候选：

1. 只用健康 calibration 冻结后层 route acceleration 的尺度、阈值和 persistence。
2. 在新 seed 上连续采集固定数量 episode，不能因结果不好而替换。
3. 预先声明严格 missed-grasp recall、全部失败 recall、成功 FPR 和报警相对事件时延。
4. 只有在漏抓之前或当下稳定报警，才计作 early warning；`q+1..q+5` 只计作 post-event detection。

v2 满足这些数据隔离条件，并取得目标 5/7、全部失败 8/14、成功误报 1/10；具体事件时序和限制见 [v2 报告](MOE_DYNAMICS_ALARM_V2_ZH.md)。
