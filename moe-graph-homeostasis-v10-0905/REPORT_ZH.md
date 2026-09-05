# HiMoE-VLA：MoE 图结构自稳实验报告

日期：2026-09-05

## 结论先行

这次实验支持的不是“Gate 越尖越可靠”，而是一个两时间尺度的结构模型：

```text
                     跨 query 仍有响应
                            ↑
       loop：高曲率、回返、难沉降  |  healthy：flow 内消噪，同时响应新观测
                            |
       ---------------------+--------------------→ flow 内自稳
                            |
       static：响应冻结、token 共识升高
```

更准确地说，MoE 的健康状态需要同时满足：

1. 对同一 snapshot 的 flow-noise 扰动具有适度吸引性；
2. 对下一 query 的新视觉和状态输入仍然敏感。

只满足第一点可能是“稳定地错”；只满足第二点而 flow 内不收敛，则可能处在 loop 式反复修正。

最重要的数值结果：

| 检验 | main16x32 | grid50x8 | 解释 |
|---|---:|---:|---|
| 固定 loop flow-instability，q-2 AUC | 0.606 | 0.607 | 弱到中等，不能作硬报警 |
| main 选 top-4，loop q-2 AUC | 0.685 | **0.660** | 确认集下降，未超过 v9 |
| 固定 static rigid-consensus，q-2 AUC | 0.866 | 0.865 | 多维结构表型可复现 |
| main 选 top-4，static q-2 AUC | 0.984 | **0.989** | 很强，但主要来自 query 冻结 |
| 最强 loop 图轴，q-2 AUC | 0.659 | **0.663** | action-edge flow curvature |
| 最强 static 图轴，q-2 AUC | 0.985 | **0.989** | back state-edge lag-4 位移低 |

与 v9 相比，graph top-4 的外部确认 AUC 没有变好：loop 为 0.660，对比 v9
full-dynamic composite 的 0.673 和 v8 最强 route acceleration 的 0.678；static 为
0.989，与 v9 最强 lag-4 轴相同。收益是结构解释和紧凑性，不是更高的数字。

## 1. 图是怎样定义的

每个 layer 和 flow step 都有一个加权 token-expert 二部图：

\[
G_{q,l,f}=(\mathcal U,\mathcal E,W),\qquad W_{u,e}=p_{q,l,f,u,e}.
\]

没有训练 GNN。我们直接从同一张图构造五个互补视角：

| 视角 | 表征 | 保留的信息 |
|---|---|---|
| `edge_action` | 10 个 action token 的 `sqrt(p)` 边 | 完整专家身份和软权重 |
| `edge_state` | state token 的 `sqrt(p)` 边 | state 路由 |
| `token_gram` | `sqrt(P) sqrt(P)^T` | token 关系；对专家重编号不变 |
| `expert_load` | action token 平均后的专家负载 | occupancy 和负载迁移 |
| `spectrum` | token Gram 的归一化谱 | 有效自由度；对专家重编号不变 |

平方根概率使 edge 距离对应归一化 Hellinger 几何。`token_gram` 很重要：如果只看
expert ID，无法区分“所有 token 一起换专家”和“token 间组织关系真的改变”。

## 2. 什么叫结构自稳

### 2.1 单次 rollout 的内部动力学

对每种图嵌入 \(\Phi_f\)，计算：

\[
v_f=\|\Phi_{f+1}-\Phi_f\|,
\quad
a_f=\|\Phi_{f+1}-2\Phi_f+\Phi_{f-1}\|,
\]

以及 early/late speed、path、endpoint、directness、curvature 和
`log(late_speed / early_speed)`。这描述一次 forward 是否沉降，但不能单独证明对扰动稳定。

跨 query 再计算 d1、acceleration、jerk、lag-4 displacement、return advantage 和
turn instability。所有量只依赖当前及过去 query，是 prefix-causal 的。

### 2.2 same-snapshot fork 的 ensemble 收缩

对同一 simulator/controller snapshot 的候选 \(i,j\)，定义：

\[
D_f=\operatorname{median}_{i<j}\|\Phi_f^{(i)}-\Phi_f^{(j)}\|,
\qquad
\rho=D_9/D_0.
\]

`rho < 1` 表示不同独立 flow-noise 路径在该图视角上收缩。这里使用的是全局独立噪声，
不是成对微小扰动，因此不能称为 infinitesimal Lyapunov stability。

## 3. GPU 和数据审计

完整 profile 使用 GPU 6、7 构建：

| 项目 | 数值 |
|---|---:|
| tasks | 45 |
| episodes | 18,560 |
| query rows | 305,030 |
| 每行原始 router shape | 8×10×11×32 |
| within-flow axes | 123 |
| cross-query axes | 95 |
| 总轴数 | 218 |
| 输出大小 | 138,976,830 bytes（132.5 MiB） |
| 每卡 peak allocated memory | 506.1 MiB |
| 最大 float16 量化绝对误差 | 0.00390625 |

profile 构建阶段 `outcomes_loaded=false`、`fitted_weights=false`、
`task_conditioned_features=false`。v9 的 2,460 轴 profile 是 1,525,119,430 bytes；
本版约小 11 倍，但原始 MoE tensor 的每个概率都参与了图计算。

## 4. fork 实验：先收缩，后分化

数据有两套：

- `fork_pilot_n32`：20 个物理 snapshot，每个 32 个独立 Gaussian flow-noise 候选；
- `rolling_star_k16`：22 个物理 snapshot，每个 16 条完整 rollout，本实验只比较共同起点 q0。

下表给出 snapshot-level `rho` 的中位数和 bootstrap 95% 区间：

| 图视角 | pilot rho [95% CI] | rolling rho [95% CI] |
|---|---:|---:|
| front action edge | **0.733 [0.658, 0.771]** | **0.550 [0.521, 0.591]** |
| front token Gram | **0.833 [0.721, 0.909]** | **0.542 [0.471, 0.580]** |
| back action edge | 1.136 [0.772, 1.651] | **0.886 [0.868, 0.905]** |
| back token Gram | 2.867 [0.950, 8.416] | **1.232 [1.200, 1.279]** |

绝对距离也必须同时看：

- rolling front action edge：0.03767 → 0.02034；
- rolling back action edge：0.01931 → 0.01698；
- rolling back token Gram：0.000454 → 0.000542。

所以后层 token Gram 的相对扩张是真实、一致的 22/22，但绝对量很小。合理解释是：

> 前层首先把不同 noise proposal 拉回共同的专家路由区域；深层的整体边向量仍可略微收缩，
> 同时 token-token 的细粒度相对关系发生分化。

这比“entropy 降了，所以模型更有把握”更接近内部机制。它也说明自稳是分层、分视角的，
不存在一个对所有层都成立的单调收缩标量。

收缩曲线见 `results/fork_contraction/fork_contraction.png`。

## 5. 自稳不是成功把握

rolling-star 有 15 个 snapshot 同时包含成功和失败分支。以 snapshot 为统计单位，q0 的
六个预定义稳态轴对最终 success 的平均 within-snapshot AUC 为：

| 轴 | AUC [bootstrap 95% CI] | Wilcoxon p vs 0.5 |
|---|---:|---:|
| flow nonsettling | 0.430 [0.300, 0.554] | 0.379 |
| flow curvature | 0.406 [0.308, 0.499] | 0.069 |
| flow directness | 0.540 [0.414, 0.666] | 0.513 |
| flow late motion | 0.394 [0.274, 0.518] | 0.105 |
| effective-rank diversity | 0.546 [0.398, 0.690] | 0.589 |
| token coherence | 0.460 [0.329, 0.588] | 0.551 |

没有一个达到 `p < 0.05`；这里还没有做多重检验校正。pilot 只有 1 个 mixed-success
snapshot，不能提供 outcome 推断。结论是：

\[
\text{graph contraction}\not\Rightarrow\text{rollout success}.
\]

它是结构证据，不应命名为 `p_success`。

## 6. q-2 相图：loop 与 static 是两种失稳

事件 query 与 healthy episode 按同 task、scene、绝对 query 配对。相图坐标是相对于
main16x32 全部 query 的经验 percentile，参考分布不使用 outcome。

### Loop

q-2 时，跨-query responsiveness 相对 matched control 的变化：

- main16x32：`+0.124`，95% CI `[+0.069, +0.180]`；
- grid50x8：`+0.077`，95% CI `[+0.050, +0.102]`。

flow settling 在 main 降 `-0.056 [-0.099, -0.011]`，但 grid 为
`-0.008 [-0.056, +0.041]`，没有跨语料稳定。因此最可靠的说法不是“loop 一定不收敛”，
而是：

> loop 前内部结构对连续 query 过度响应，同时 action-edge 轨迹曲率升高。

最强可复现图轴是 `flow|all|edge_action|curvature`，AUC 0.659/0.663。

### Static

static 的跨-query responsiveness 明显坍缩：

- main：event 0.034 vs control 0.286，差 `-0.252 [-0.270, -0.235]`；
- grid：event 0.047 vs control 0.305，差 `-0.258 [-0.304, -0.214]`。

同时 token coherence 上升：

- main：`+0.191 [+0.157, +0.226]`；
- grid：`+0.157 [+0.105, +0.210]`。

但 within-flow settling 没有升高，反而在 main 略降。这修正了最初的“static 是所有意义上
都更稳定”假设：

> static 的本质是跨 query 的结构冻结和 token 同质化，不是单次 flow 轨迹必然更平滑。

相图见 `results/phase_portrait/phase_portrait.png`。

## 7. 优化结果怎样看

这里报告三层结果，不能混写：

1. `prior_fixed`：根据机制讨论事先固定的等权组合；
2. `main16x32_rank_locked`：使用 main 的 outcome 排名前四轴，再固定方向去看 grid；
3. `post_hoc_axis_audit`：两套结果都看过后的描述性机制搜索。

没有分类器、GNN、logistic regression 或 outcome 权重拟合。尽管如此，第 2 类仍然用了标签做
feature selection，不能冒充完全 train-free 的先验检验；而且 grid 是以前分析过的语料，不是
全新的 holdout。

主语料 top-4 确认结果表明，多用 MoE 信息没有自动提升精度：

- loop：0.685（main）→ 0.660（grid）；
- static：0.984（main）→ 0.989（grid）。

static 多视角几乎完全冗余，核心就是 route graph 在四个 query 尺度上冻结。loop 的多个
curvature/settling 轴能互补一点，但外部结果仍低于 v9 0.673。

## 8. 当前最可信的内部图景

1. **前层是 noise absorber。** action-edge、token Gram、expert load 和谱在两套 fork 中都收缩。
2. **后层不是简单继续压缩。** labeled action edges 可收缩，而 permutation-invariant token
   relation 可扩张，说明后层在形成相对结构。
3. **loop 是动态过调。** route path 对 query 变化过强，flow 曲率升高；但其信号只有中等强度。
4. **static 是闭环失去可塑性。** route 在 query 间冻结，token 关系趋同；这是很强的确认信号。
5. **稳定性和正确性正交。** q0 fork 的结构稳态轴不能可靠区分最终成功与失败。

因此，最合适的数据结构不是单一 confidence，而是至少保留：

```text
GraphHomeostasisState:
    within_flow_contraction
    within_flow_curvature
    cross_query_responsiveness
    cross_query_return
    token_coherence
    effective_rank
    layer_differentiation
```

只有将这个状态与同 snapshot 的真实 continuation counts 校准，才可以再映射成 outcome assurance。

## 9. 限制与下一步

- fork 使用独立 Gaussian noise，不是受控小扰动对，因此目前测的是 ensemble contraction。
- rolling-star 只有一个任务、22 个 snapshot；success 关系的置信区间仍宽。
- static onset 标签依赖连续重叠静止窗口，q-2 更接近正在形成的 static 确认，而非纯预测。
- 218 轴全扫描存在多重比较，只能用于提出机制假设。
- 真正检验局部结构自稳，需要从多个任务 snapshot 采集 `xi` 与 `xi + epsilon` 成对 CRN fork，
  并同时记录 action/output Jacobian 和物理 outcome。

在补采集之前，不建议训练 confidence head；当前更可靠的用途是 static guard、机制诊断和
snapshot 采样位置选择。
