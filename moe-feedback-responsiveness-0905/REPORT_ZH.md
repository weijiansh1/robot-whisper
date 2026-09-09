# HiMoE-VLA：MoE feedback responsiveness 与 deadline risk

日期：2026-09-05

## 结论

“健康 MoE 的关键不是持续有序，而是持续响应”这个方向是对的，但现有数据把它修正成了
一个更具体的两阶段结论：

```text
v4 首次报警：deadline risk / feedback watch
        ↓ 继续因果更新，不立即把报警等同于不可恢复 Trap
路由结构重新响应：更可能值得 Extend
路由结构没有重新响应：更可能需要 Intervene
```

最关键的结果不是某个 alarm 时刻的绝对 mobility，而是从 alarm 到当前时刻的状态转移。
在 external 8B 的 v4 已报警风险轨迹中：

| MoE-only evidence | development AUC | external AUC | external AP |
|---|---:|---:|---:|
| alarm 时刻 quiet route history | 0.669 | 0.664 | 0.187 |
| alarm 时刻 quiet route + quiet graph | 0.716 | 0.688 | 0.196 |
| alarm→H route activity release | 0.676 | 0.663 | 0.158 |
| alarm→H layer back-handoff | 0.740 | 0.709 | 0.222 |
| alarm→H state-conditioned front graph release | 0.739 | **0.734** | 0.169 |
| alarm→H 三视角等权融合 | 0.726 | **0.732** | **0.257** |

external 的 late-success 基率只有 27/410 = 6.59%，所以 AP 0.257 相当于明显的排序
增益；但这些数都不是恢复概率。三视角融合的 row-stratified 95% bootstrap AUC 区间为
`[0.619, 0.839]`，task-clustered 区间更宽，为 `[0.563, 0.889]`。

本实验不使用 event onset，也不使用 `q-2`。`q-2` 在这里既不是定义，也不是必要条件。

## 1. v4 究竟在预测什么

v4 的 outcome target 本来就是：

```text
original_failure = late success within +10 OR persistent after +10
```

因此把它解释成 deadline-risk monitor 比解释成 irrecoverable-failure detector 更准确。

| external 8B outcome | 总数 | v4 报警 | recall |
|---|---:|---:|---:|
| late success +10 | 41 | 27 | 65.85% |
| persistent after +10 | 523 | 383 | 73.23% |
| timely success | 15,036 | 81 | FPR 0.539% |

late 和 persistent 的首次报警 phase 中位数分别约 71.43% 和 70.59%，也没有明显的
“late 一定更晚才报警”捷径。

但还不能把当前 v4 描述成数值上对称的 Freeze/Switching monitor。external 已检测风险中，
first branch 为：

| outcome | lock first | instability first |
|---|---:|---:|
| late | 26 | 1 |
| persistent | 354 | 29 |

所以当前 v4 的实际性能主要由 lock head 提供。`feedback responsiveness monitor` 是一个
正确的上层解释，但现有实现仍偏向 freeze detector。

## 2. 为什么 alarm 时刻不能直接判断 Extend

我们先用已有的 12 个 causal MoE query 特征和 8 层 mobility，按 task 和绝对 query
在 outcome-blind reference 中做经验 rank。没有 classifier、损失函数或 outcome 权重。

直觉上，“仍有 activity”似乎应该意味着更可恢复。数据给出了相反结果：

| alarm 时刻原方向分数 | development AUC | external AUC |
|---|---:|---:|
| route activity 高 → late | 0.320 | 0.390 |
| response history 高 → late | 0.331 | 0.336 |

换言之，v4 已报警条件下的 late-success 在 alarm 时反而更安静、更 recurrent、burst 更少。
external 中 route activity 的中位数为 late `0.176`、persistent `0.288`；lag recurrence
prefix mean 为 late `0.608`、persistent `0.532`。

这不是“越锁死越健康”。至少有两个原因不能越过：

1. 这是条件在 v4 alarm 上的选择后比较，存在 collider/selection effect；
2. late-success 多数来自少数 long/spatial task，只有 27 个 external 正例。

alarm 时将 quiet route 与 quiet graph 等权组合，late-vs-persistent external AUC 为 0.688，
但同任务不确定性仍很大。它可以做 watch-list 排序，不能支持即时硬干预。

更严格的在线口径还要包含 81 个 timely-success false alarms。此时正类是“不要干预”
（timely + late，共 108/491）。quiet route history 的 external AUC 为 0.677、AP 0.385；
加入 graph 后反而降到 AUC 0.618。因此在线不能只报告条件于未来 `original_failure` 的漂亮数字。

## 3. 完整图建模怎样利用前层吸噪、后层成结构

本次不是只看 layer mean。external 8B 的全部原始路由被重新计算：

```text
P[q, layer=8, flow=10, token=11, expert=32]
```

令 Hellinger embedding 为 `r_u = sqrt(p_u)`，token Gram 为：

\[
G_{uv}=\langle r_u,r_v\rangle.
\]

去掉 action token 共同继承的 state direction：

\[
K_{A\mid S}=G_{AA}-G_{AS}G_{SA}.
\]

将前四层的 terminal-flow conditional action graph 展平为 \(\Phi_q^F\)，定义跨 query
响应：

\[
d_q^F=\|\Phi_q^F-\Phi_{q-1}^F\|_2,
\qquad
r_q^F=\widehat F_{task,q}(d_q^F).
\]

其中 \(\widehat F_{task,q}\) 只来自同 task、同 query 的旧路由经验分布，不读取 outcome。
在 v4 首次报警 \(a\) 后，真正有用的动态量是：

\[
\boxed{R_{a\rightarrow q}^{F}=r_q^F-r_a^F}.
\]

它问的是：前层 state-conditioned action structure 在报警后有没有重新开始响应，而不是
“当前 gate 尖不尖”。

GPU profile 构建审计：

| 项目 | 数值 |
|---|---:|
| external tasks / episodes | 39 / 15,600 |
| query rows | 248,255 |
| graph/transfer axes per query | 584 |
| feature build loaded outcomes | false |
| fitted weights | false |
| output bytes | 469,222,657 |
| GPU 6 / 7 peak allocated | 369.77 / 369.77 MiB |

584 个轴的 development 与 external 原始效应具有 Spearman `rho=0.557`。这些轴高度相关，
所以 p-value 不能当作 584 个独立复现；它只说明结构效应整体具有跨批次一致性。

## 4. 最重要的动态结果：Lock 后的 Restart

在 v4 报警点，late-success 的 graph responsiveness 也更低：固定三种 action graph view
平均后，原方向 AUC 为 development 0.327、external 0.319。方向在 rollout 后段反转。

对 `state-conditioned front graph release`，只在仍有对应 query 的轨迹上计算：

| 相对首次报警 | development AUC | external AUC | external n / late |
|---:|---:|---:|---:|
| +1 query | 0.445 | 0.530 | 390 / 27 |
| +2 queries | 0.439 | 0.577 | 364 / 26 |
| +4 queries | 0.571 | 0.563 | 330 / 25 |
| +8 queries | 0.620 | 0.612 | 229 / 15 |
| original horizon | **0.739** | **0.734** | 409 / 27 |

不同 offset 的样本数不同，因为部分报警离 horizon 不足 8 queries；不能把这张表当成同一
population 上严格单调的 learning curve。它仍清楚说明：+1 时没有可靠信息，restart 证据
是在继续接收 feedback 后逐渐形成的。

在 horizon 边界，external 分 suite 的 graph release AUC 为：

| suite | n / late | AUC |
|---|---:|---:|
| long | 196 / 15 | 0.726 |
| object | 30 / 3 | 0.753 |
| spatial | 95 / 9 | 0.735 |

同任务 pair-weighted AUC 为 0.840，说明它不像许多 terminal activity 分数那样完全靠 long
task 构成。但只有 9 个 task 同时有 late 和 persistent，置信度仍有限。

一个重要消融是：归一化的 `action_partial` relative topology 没有复现这个 release；有效的
是保留 residual magnitude 的 `action_conditional` graph。这说明可恢复信号不仅是 token
关系图换形，还包含 state-orthogonal action routing energy 的重新展开。

因此，“前层吸收噪声、后层形成结构”对 trap 识别的实际用途是：

```text
不要把正常 contraction 当风险
    → 保存 alarm 时的 conditional graph 基线
    → 持续监测前层重新响应和 front/back transfer gap
    → 将 lock → restart 作为 extendability evidence
```

## 5. 三视角融合与资源分配

探索性等权分数为：

\[
S_{restart}=\frac13\left(
\Delta r^{front,conditional}
+\Delta r^{route\ activity}
+\Delta r^{layer\ handoff}
\right).
\]

三个分量分别保留 conditional token graph、全局 query activity 和前后层传递。公式不训练
权重，但它是在看过 development 结果后形成的，因此是 post-hoc mechanism fusion。

external 已报警风险上的结果：AUC 0.732、AP 0.257，late 基率 0.066；同任务 weighted
AUC 0.811。分 suite AUC 为 long 0.796、object 0.815、spatial 0.610。spatial 上单独的
conditional graph release 更好，说明统一融合仍会稀释 phenotype-specific 信息。

用 development 的 score 分布做 outcome-blind 预算分位阈值，external 回放为：

| allocation | Extend 数 | late 找回 | late recall | persistent 被延长 | late precision | 每找回一条的 extra-query cost |
|---|---:|---:|---:|---:|---:|---:|
| q50 | 199 | 22/27 | 81.5% | 177/383 | 11.1% | 84.7 |
| q75 | 114 | 17/27 | 63.0% | 97/383 | 14.9% | 61.2 |
| q90 | 41 | 12/27 | 44.4% | 29/383 | 29.3% | 28.2 |
| 全部延长 | 410 | 27/27 | 100% | 383/383 | 6.6% | 146.7 |

q75 将单位找回成本相对全部延长降低约 58%，但会漏掉 10/27 个已报警 late-success。这个
表只评估 Extend 预算，不代表真正施加 Intervene 后的因果收益。

## 6. 哪些值能“理论计算”

以下量可以从当前和历史路由确定性计算：

- v4 mobility 与持续越界状态；
- conditional token graph、front/back transfer gap；
- alarm-to-current restart evidence；
- 相对于 outcome-blind reference 的经验 rank。

但不能仅由模型结构理论推出：

\[
P(\text{late success}\mid R_{\le q})
\quad\text{或}\quad
P(\text{intervention succeeds}\mid do(u),R_{\le q}).
\]

当前的 0.732 是 AUC，不是 73.2% 恢复概率；q75 是资源分位阈值，也不是 75% 把握。
真实概率仍需要同 snapshot continuation/fork count 和单独校准。

## 7. +20 / +40 为什么仍然必要

当前 `persistent` 的准确含义只是“+10 内没有成功”，它是右截断标签，不等于不可恢复。
因此现在能支持的 Stage 2 名称是：

```text
recoverable-within-10 ranking
```

不能写成 irrecoverable-trap classifier。

下一轮应只对现有 980 条 `+10` remaining failures 延长同一 noise stream，并记录：

```text
success by +20
success by +40
right-censored after +40
```

不建议先训练分类器。更干净的检验是按 alarm-to-current graph restart score 的冻结分位组，
画 Kaplan-Meier / cumulative recovery curve，并检验：

\[
P(T_{success}\le h\mid S_{restart}\text{ stratum}),
\qquad h\in\{10,20,40\}.
\]

为了保持严格可比，+20/+40 必须从原始失败 prefix 精确 replay，并使用与 +10 相同的前十个
continuation noises；只是在同一 RNG stream 上继续取后续 noise。此次没有启动这项数小时级
simulator continuation，GPU 6/7 用于完成了全量 external graph profile。

## 8. 有效性边界

- external outcome 以前已被项目查看，不是 pristine holdout；所有新分数都应视为 post-hoc
  跨批次确认。
- external 已报警 late-success 只有 27 条，且只有 9 个 task 同时含两类 outcome。
- alarm 后 offset 曲线存在可用剩余 horizon 的选择差异。
- `original horizon` 决策适合 Extend/Intervene 预算分配，但太晚，不能替代需要提前动作的
  safety intervention。
- v4 未报警的 14/41 external late-success 不进入这个 Stage 2；端到端系统还需要处理 Stage 1
  miss。
- 所有结果来自 routing evidence；没有验证真实 rescue action 的效果。

## 9. 最终建议

保留 v4 作为低误报 deadline-risk latch，但不要在第一次报警时立刻把 trajectory 判成
persistent Trap。Stage 2 应是一个持续更新的有限状态过程：

```text
WATCH(alarm baseline)
  ├─ conditional graph restart rises -> EXTEND_CANDIDATE
  ├─ remains frozen near deadline      -> INTERVENE_CANDIDATE
  └─ sustained switching               -> separate instability policy
```

当前最可靠的新信息是 `lock → conditional-graph restart`，不是绝对有序、绝对无序，也不是
固定 lead。下一次决定性实验是 +20/+40 survival replay，而不是继续增加一个静态 entropy
分数。

## 复现产物

- `results/summary.json`：基础 12-feature/8-layer 分析；
- `results/graph_profiles/external_8b/build_summary.json`：GPU 全图 profile 审计；
- `results/graph_stage2/summary.json`：584 轴、固定组合和 post-alarm curve；
- `results/fusion/summary.json`：多视角融合；
- `results/fusion/allocation_metrics.csv`：q50/q75/q90 分配结果；
- `results/fusion/transition_scores.csv`：逐轨迹 causal transition evidence。
