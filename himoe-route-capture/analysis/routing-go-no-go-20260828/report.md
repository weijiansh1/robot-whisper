# HiMoE-VLA routing 关门实验总审计

日期：2026-08-28

## 总结

现有缓存包含多组不同 cohort；其中主 16×32 corpus 有 2,560 条 rollout。它们已足以关闭若干特定旧叙事，但不足以关闭所有 MoE 研究：

1. **关闭**“action hard expert-ID 的回返/更替等价于策略回返/改变”。实际执行 ID 的日志可信，但 action gate 的边界大量平局，hard-ID 不携带软分布移动幅度，也不能单独证明功能输出变化。
2. **在 clean-444 q0+8 单任务协议内关闭**“routing 在完整 cached physical state 和 action 之外提供实用失败预测增量”。state soft routing 增量为 `+0.008`，action soft routing 为 `-0.014`；预设实用阈值是 `+0.05`。这不外推到其他任务、时点或模型族。
3. **保留**“action soft routing 是 candidate-conditioned action representation”。同一 cached simulator/proprio state 的 32 个 seeds 中，它可解码 sampled action chunk，held-init `R2=0.605`；RGB 没有存档，不能独立核对 pixel equality。
4. **当前不支持**“action soft routing 是 Best-of-N quality selector”。完整 70 维 action 对第一 chunk 的 EEF 响应已达 `R2=0.9985`，加 route 的增量为 `-0.00037`；真正的 lift/progress 标签在 0.5 秒内退化。final-success additive 增量为 `+0.030 [-0.010,+0.070]`，但来自事后筛选的 mixed pools。
5. **state-routing 结果与 dynamic state/action-phase shadow 一致**。探索性矩阵中主要增量来自速度、进度速度和姿态范数，而不是完整 geometry 之外的静态几何或通用风险；这只是观察性读出，不是 expert 机制或监测效用验证。
6. **关闭当前 t34 early-warning 叙事**。t34 的 future remaining-time 风险集只有 `1 success / 216 failures` 的精确重叠，phase 和 remaining-time 哨兵 AUC 均为 `0.998`。
7. **阻塞 post-error adaptation 命题**。现有自然 rollout 的 219 个候选风险集组合中，0 个通过双结局 Gate；需要受控 setback 和 checkpoint continuation 新采集。

因此，最准确的当前结论是：

> HiMoE-VLA 的 soft action routing 会随 sampled candidate 改变，state routing 能读出动态状态；但在已完成的特定协议中，没有证据表明 routing 在完整 cached state 与 action 之外提供了可用的失败预警或候选质量增量。router-input hidden 在 clean-444 q0+8 上留下 `+0.053` AUC 的暂定非零信号，但其区间没有证明真实增量达到 `+0.05`。

## 决策表

| 研究命题 | 关键结果 | 判定 | 判掉/保留的范围 |
|---|---|---|---|
| 实际执行 expert ID 是否被正确保存 | canonical cache 中，actual ID 与 saved-fp16 probability 的确定性 Top-4 集合不一致率 `0%` | GO（事实日志） | 只说明记录到的 executed IDs 可信 |
| saved probability 是否唯一识别 action Top-4 | action `p4=p5` 为 `28.6%`；hypothetical BF16 output-round stable fraction `61.3%` | NO-GO | 不能从概率唯一重建大量边界，也不能把 ID change 当策略 change |
| 历史 fp32 gate 是否会给相同 Top-K | 历史 CUDA logits/softmax dtype 与 fp32 shadow 未保存 | UNRESOLVED | 必须 fresh capture；现有 fp16 上采样无效 |
| action soft routing 是否随 candidate 变化 | 80 个 task×init pools、每池 32 seeds；action decoder `R2=0.605 [0.598,0.610]`，route/action 距离 Spearman `0.175 [0.167,0.182]`；state-route control 为 0 | GO（关联） | route 与 action 都可能由 flow seed/action hidden 共同驱动；没有 router intervention 因果证据 |
| routing 是否预测第一 chunk 的物理响应 | EEF 0.5 s：route `R2=0.626`，完整 action `R2=0.9985`，action+route `R2=0.9981` | GO（route-only 可读）/ NO-GO（增量） | action 之外增量 `-0.00037 [-0.00051,-0.00028]`；不等于候选质量 |
| routing 是否在 action 之外提高短时质量预测 | lift/progress 在 0.5 s 内为 0；可识别的 EEF/approach 响应上增量约为 0 | BLOCKED（质量） | additive test 已有效，但当前 horizon 没有可识别的任务质量标签 |
| 当前 decoder 是否可选最终成功 candidate | mixed-pool route AUC `0.400 [0.327,0.468]`；action+route 相对 action `+0.030 [-0.010,+0.070]`，MDE `0.057` | NO-GO（当前 decoder） | 事后 outcome-selected 的 40 pools/1,280 candidates；不能扩大成“无任何 outcome 信息” |
| state routing 编码什么 | 探索性 35/35 概念可读；8/35 对 geometry 有正 CI；最大增量为 EEF speed `+0.132`、progress speed `+0.036`、axis-angle norm `+0.035`、target speed `+0.032` | CONSISTENT WITH STATE SHADOW | 5 tasks、线性 baseline、未做多重性校正；未验证监测效用或因果机制 |
| q0+8 routing 是否超越完整 cached state+action | long/SCENE8 clean-444、单一 q0+8 cut、L2-logistic：state `+0.008 [-0.017,+0.025]`；action `-0.014 [-0.037,+0.013]` | NO-GO（该协议） | 旧 routing 正结果在该协议补全 cached state 后塌缩，不外推到其他任务/时点 |
| hidden 是否超越完整 cached state+action | 同一 clean-444 协议：router-input hidden `+0.053 [+0.014,+0.085]`，MDE `0.053` | TENTATIVE NONZERO SIGNAL | 点估计超过 `+0.05`，但 CI 下界仅 `+0.014`；未证明真实效应达到实用阈值 |
| t34 是否是 phase-matched early warning | exact remaining-time overlap `1 S / 216 F`；哨兵 AUC `0.998`；route 增量 `+0.029` 至 `+0.032`，MDE `0.063` 至 `0.069` | NO-GO（当前语料） | 只能解释并发状态，不支持 phase-matched 预警；且对 +0.05 仍欠功效 |
| 自然 setback 后 routing 是否预测恢复 | 219 个 task×event×threshold×outcome 组合，0 个通过 Gate；真正 setback 的 recovery success 极少 | BLOCKED | 不是一个可解释的负结果；风险集结构不存在 |
| 旧全轨迹 routing 聚类是否发现失败机制 | failure 与跑满 cap 几乎等价；1-bit length sentinel 超过 routing 核心 | NO-GO | 旧六块主要是 task/length/timeout 组织，不是失败机制 taxonomy |

## 三个主问题的答案

### 1. Action gate 是否形成 candidate-dependent routing？

**是，但只在 soft representation 层面成立。**

query 0 的 cached simulator/proprio state 在每个 task×init 的 32 seeds 内逐位相同，state-token route 也完全不变；RGB 没有存档，pixel equality 未被独立验证。action-token soft route 能在留出 init 上解释 `60.5%` 的 centered action-chunk variance，并显著超过 seed sentinel。这排除了“只是不同初态”这一主要混杂，但 route 和 action 仍可能共同受 flow seed/action hidden 驱动。

它还不是 selector 证据。route 对下一边界 EEF 位移的 `R2=0.626`，说明它是 sampled action 的有损表征；完整 70 维 action 达到 `0.9985`。blockwise joint 保留相同的完整 action，再加入 16 维 route，结果反而为 `0.9981`，增量 `-0.00037`。因此，对这个短时物理响应，route 在完整 action 之外没有实用增量。

### 2. State routing 是否提供完整物理状态之外的信息？

**观察结果与可线性读出的动态 state/phase shadow 一致；对静态 geometry 和失败风险没有已证实的独立增量。**

最明确的增量集中在 EEF speed、target speed、task-progress speed 和姿态范数。位置、相对距离、夹爪 aperture、target quaternion 等静态量已经被线性 geometry baseline 几乎完全解释，加入 routing 通常不改善，部分还下降。矩阵是 5-task 探索性读出，35 个概念未做多重性校正，也没有验证在线监测收益。

在 long/SCENE8 clean-444 的单一 q0+8 failure cohort 中，补入所有缓存 `sim_state` 坐标和 action 后，state/action routing 的失败 AUC 增量分别只有 `+0.008` 和 `-0.014`。这只支持在该 task/cut/model family 内将结果解释为 state/action-phase shadow，不能推广到全部失败时点。

### 3. 收到失败反馈后，内部表征是否预测可恢复性？

**当前缓存无法回答。**

真正 setback 在任务内几乎总与终局失败绑定；样本多的 `heightloss` 等事件又被验证为正常放置阶段。最贴近重抓的问题只有不超过 21 条风险事件。对这样的数据训练分类器，只会重新学习 task、onset 或剩余 budget。

## 下一步只做两类新实验

### A. FP32 数值链路 capture

同一次 forward 同时保存：

- deployed router logits/scores 及每一步 runtime dtype；
- actual pre-cast `topk_idx`；
- autocast-disabled、相同 hidden/weight 的 fp32 shadow logits/scores/Top-K；
- cast 前和 cast 后的 Top-K；
- 每个 expert 的 weighted contribution `w_e f_e(h)`。

主统计量是 `P(TopK_fp32 != TopK_deployed)`，其次才是 margin、entropy 和 hard turnover。若 fp32 action gate 仍近均匀，hard-ID 语义线正式结束；若 fp32 明确而 deployed/cached 不稳定，问题属于数值执行或日志链路。

### B. 受控 candidate-quality / recovery capture

- 固定记录 horizon，同时保留 first-success time；post-success hold 不进入 active-task control。
- 用 privileged physical predicates 保存 grasp 前、形成抓握、抬升后、transport、目标附近和 release 前 checkpoint。
- 从同一 checkpoint 分叉 no-perturbation、轻/中位姿偏移、短暂强制开爪等 setback。
- 每个 post-event state 使用共同 continuation seed bank，形成 recovery success / repeated failure 双结局。
- 对 `h in {0,1,3,5,10}` 执行当前 chunk 前 h 步后再 replanning，估计 `Q_k(h)`。
- 保存 RGB/vision embedding、chunk 内 dense sim state、contact/grasp truth、完整相对 geometry、hidden、完整 action、flow/environment seed。

在预注册中继续使用：within-checkpoint 或 within-init 比较、cluster bootstrap、phase/remaining-time sentinel、双结局 Gate，以及 `delta_min=0.05` 的实用阈值。没有通过风险集 Gate 时不训练 recovery predictor。

## 产物索引

- 路由精度与 hard-ID 可识别性：`analysis/router-quantization-identifiability/report.md`
- 同初态 32-seed candidate 实验：`analysis/candidate-level-soft-routing/report.md`
- state-token 概念矩阵：`analysis/state-routing-concept-matrix/report.md`
- clean-444 完整 cached-state ladder：`analysis/full-geometry-routing-ladder/report.md`
- t34 关门审计：`analysis/t34-closure/report.md`
- post-error 风险集 Gate：`analysis/post-error-adaptation-20260828/feasibility.md`
- post-error 新采集预注册：`analysis/post-error-adaptation-20260828/PREREG.md`
- 旧全轨迹聚类泄漏审计：`analysis/AUDIT-clustering-leakage-20260828/report.md`

## 解释限制

- “full cached state” 指缓存里所有 `sim_state` 坐标，不包含未记录的 RGB、contact、force 或完整环境真值。
- q0+8 的成功终点 goal reference 是 transductive global-success mean；因此 hidden 正结果仍需严格 unseen-init 复现。
- 多数 CI/MDE 对固定 OOF prediction table 做 cluster resampling，没有在每个 bootstrap 内重训整个 pipeline。
- candidate 实验的 pool centering 使用完整 K=32 pool，估计的是 transductive within-pool ranking，不是单 candidate 的 inductive deployment。
- 概念矩阵只有 5 个 task，且 geometry baseline 是线性模型；routing 增量可能包含非线性重编码，而不是新的物理信息。
