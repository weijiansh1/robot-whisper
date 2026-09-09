# 现有数据上的 MoE 全实验审计

日期：2026-08-25

## 一句话结论

现有数据支持把 HiMoE routing 描述为**当前状态与 noisy action latent 条件下的局部
计算路径和低成本 diagnostic readout**；不支持把 route consensus 用作候选选择器，
不支持用 route dispersion 分配采样预算，也不支持 route 先于 action 表达高层控制
承诺或 expert ID 构成稳定技能状态机。

本轮使用现有数据在 CPU 上覆盖了所有当前可执行的 MoE 实验族。没有启动模型推理或
simulator，也没有把 observational full-rollout success 冒充 candidate-chunk Q。

## 总裁决

| 问题 | 当前结果 | 裁决 |
|---|---|---|
| MoE route 的“多数中心”是否是更好的候选？ | legacy endpoint、五任务 K32、K8 task-held-out 三组结果均无稳定收益 | 不支持 |
| route 是否比 action/noise/hidden 多提供 outcome 信息？ | 严格 task+seed-fold 双留出增量为负，区间跨 0 | 不支持 |
| route dispersion 能否判断哪里值得多采样？ | task-macro rho 接近 0，task-held-out prediction 差于均值基线 | 不支持 |
| route geometry 是否跟随后果而非动作？ | 两个点估计方向正确，但有效 snapshot 仅 3 和 1 | gate 不通过 |
| route 是否先于 action 显示最终分叉？ | route 主要跟随 current latent 和局部更新；控制 latent 后无最终 action 增量 | 不支持 |
| functional expert output 是否比 raw router 更有信息？ | 单任务有边界性弱信号；跨任务 hidden/shared-matched 增量不稳定 | 仅 exploratory |
| route 是否具有跨 control 动力学？ | 相邻 control route 显著连续，有限状态转移可预测 | 支持描述性动力学 |
| 该动力学是否是独立技能状态？ | matched proprio 同样强或更强，route K16 仅 1/5 task 胜出 | 不支持 |
| expert intervention 是否具有 outcome 因果效应？ | 仅 1 pair / 3 arms instrumentation smoke；扰动能传播到 x10 | 尚不可检验 |

## 数据与推断边界

### Legacy 同状态 fork

- 20 个 exact snapshot，每个 K=32，共 640 个 candidate chunk。
- 保存完整 HB route、最终 action chunk 和 drawer/object endpoint proxy。
- 每个 candidate 只有一次 continuation realization，没有 repeated-CRN Q。
- 没有 candidate 执行过程的 dense trajectory/contact/event tape。
- success 只在 1/20 snapshot 内 mixed，chunk drawer delta 只在 4/20 mixed。

### 五任务 16x32 grid

- 5 个任务、3 个 suite、80 个 initial-state pool、2,560 条 rollout。
- 每个 state 使用相同的 32 个 flow-noise seed，并保存 route、hidden、activation 和
  最终 episode success。
- 40/80 个 state pool mixed，但一个任务 512/512 全成功，多个任务接近饱和。
- 一个 seed 同时决定首 query 和之后的 replanning noise stream，因此 success 是
  episode-start shadow label，不是首个 chunk 的因果 value。

### 时序与机制数据

- Flow-lead：1 task、1 scene、4 correlated rollouts、44 query、每 query K16，保存
  `x0...x10`。
- Route corpus：30 tasks、1,500 rollouts、18,384 control query。
- Functional cache：五任务 80 个 K32 pool 的 hidden/expert-output 离线重建。
- Runtime intervention：仅 HB5/d0 的 1 个 matched pair、3 arms 工程 smoke。

统计推断始终先在 snapshot/state 内聚合，再以 snapshot/state/task 为单位；不会把成千
上万个 pair 当成独立样本。

## 1. MoE route consensus selector

“最接近其他候选共识”被实现为 route-space medoid：

\[
i_R^*=\arg\min_i\frac{1}{K-1}\sum_{j\ne i}d_R(\Gamma_i,\Gamma_j).
\]

概率距离使用完整 32-way router distribution 的 site-normalized RMS Hellinger；另测
order-invariant top-4 expert-ID Jaccard。所有候选只使用 action tokens，排除候选不变的
state token。

### 1.1 Legacy K32 endpoint

| selector | continuation drawer progress vs random | 95% snapshot CI |
|---|---:|---:|
| action medoid | +0.003171 | [-0.001879, +0.010248] |
| full HB probability medoid | +0.002457 | [-0.001274, +0.006533] |
| expert-ID medoid | +0.001117 | [-0.002645, +0.005115] |

Full route medoid 相对 action medoid为 `-0.000714`，95% CI
`[-0.004846,+0.002004]`。两者只在 10% snapshot 选中同一候选。success 的表面正值
不能解释，因为只有一个 mixed-success snapshot。

结论：legacy proxy 上没有建立 route consensus 优势。

来源：[legacy K32 report](../moe-consensus-audit/legacy-k32/REPORT.md)。

### 1.2 五任务 K32 success shadow

| selector | selected - exact random | 95% state bootstrap CI | seed concentration |
|---|---:|---:|---:|
| route probability d0 | +0.74 pp | [-3.75, +5.08] pp | 91.3% 选同一 seed |
| route probability d0-2 | +0.74 pp | [-3.71, +5.16] pp | 97.5% 选 seed 1013 |
| route probability full | +0.74 pp | [-3.79, +5.12] pp | 97.5% 选 seed 1013 |
| expert-ID d0 | -0.51 pp | [-4.84, +3.48] pp | 11 个 seed |
| expert-ID d0-2 | +0.74 pp | [-3.05, +4.26] pp | 9 个 seed |
| expert-ID full | -1.76 pp | [-6.80, +3.09] pp | 7 个 seed |

Probability medoid 在 d0、d0-2 和 full 得到同一 selected success，并非逐步形成稳定
控制共识：full 规则在 78/80 state 都恢复同一个全局 seed template。

### 1.3 K8 task-held-out selector

主要留出任务上，early route center 相对随机为 `+1.95 pp`，95% CI
`[-1.37,+5.27] pp`；route+noise self-supervised head 为 `-1.17 pp`。五个任务轮流
留出后，route center 与 route+noise head 的 task-macro 分别为 `-0.82 pp` 和
`-1.13 pp`。

该 head 可以预测 future-route centrality，平均 Spearman 为 `0.817`，但这不转化为
成功率。early route centrality 与 initial-noise centrality 的池内秩相关为 `0.902`。

来源：[cross-task audit](cross-task/REPORT.md) 与
[K8 selector report](route-noise-selector/report.md)。

## 2. MoE 的条件增量

本轮新增固定线性 probe，同时留出完整 task 和测试 K8 的八个 seed：

| feature | K8 AUC | selected success - random |
|---|---:|---:|
| noise | 0.476 | -0.20 pp |
| action | 0.561 | +0.74 pp |
| hidden | 0.479 | -0.20 pp |
| route | 0.478 | -0.82 pp |
| action + noise | 0.533 | +0.43 pp |
| action + noise + route | 0.510 | +0.12 pp |
| action + noise + hidden | 0.449 | -0.20 pp |
| action + noise + hidden + route | 0.440 | -0.82 pp |

Route 加到 action+noise 后为 `-0.31 pp`，95% CI `[-2.81,+1.88] pp`；再控制
hidden 后为 `-0.63 pp`，CI `[-3.13,+1.88] pp`。这否定当前固定 readout 的可复用
增量，不等价于证明所有非线性信息都不存在。

Early-action geometry 结果与此一致：state+seed-disjoint 的 hidden、layer-resolved
route、noise Spearman 分别为 `0.542 / 0.446 / 0.013`。Route 可以解码一部分 action
geometry，但 hidden 更强，而且该 target 不是 value。

来源：[conditional increment](cross-task/task_seed_heldout_increment.md) 与
[early action head](early-action-head/REPORT.md)。

## 3. MoE opportunity / adaptive-K

五任务 K32 的经验 headroom 定义为：

\[
\widetilde O_{32}(s)=\max_i y_i-\frac1{32}\sum_i y_i,
\]

其中 `y_i` 是一次完整 rollout success，而不是真实 chunk Q。

| route observable | task-macro rho with headroom | 95% hierarchical CI | p |
|---|---:|---:|---:|
| action-route dispersion d0 | +0.052 | [-0.282, +0.371] | 0.6856 |
| action-route dispersion d0-2 | +0.089 | [-0.243, +0.391] | 0.4866 |
| action-route dispersion full | +0.025 | [-0.302, +0.347] | 0.8535 |

Full-route dispersion 的 task-held-out MAE 为 `0.188`，差于训练任务均值基线
`0.161`。固定 ridge 的 route feature OOF R2 为 `-0.173`；加入 state-token route 后
更差。state token 在同 state 候选间的最大 Hellinger variation 仅 `3e-8`，所以它只能
做 query-level 信号，不能做 candidate ranking。

另一份 44-query K16 数据直接预测 action-cloud dispersion：state routing 的 LOO R2
为 `-0.055`，proprio baseline 为 `+0.067`，proprio+route 为 `-0.027`。

结论：当前数据不支持 route-based adaptive K。

来源：[opportunity report](opportunity/REPORT.md)。

## 4. Route geometry 与 outcome geometry

主指标是 8 layers x 10 denoise x 10 action tokens x 32 experts 的完整 HB RMS
Hellinger。结果按 snapshot 等权：

| action stratum | eligible snapshots | different - same route distance | 95% CI | p |
|---|---:|---:|---:|---:|
| far action | 3 | +0.000281 | [-0.000947, +0.001838] | 0.50 |
| near action | 1 | +0.000968 | unavailable | 0.50 |

两个点估计方向都符合预期，但 `both CI > 0` 的预设 gate 为 false。瓶颈是 mixed
outcome snapshot，而不是 pair 数；标签还是 drawer endpoint proxy，不是 repeated Q。

来源：[route-outcome report](route-outcome/REPORT.md)。

## 5. 去噪过程中 route 是否领先 action

### 5.1 Action commitment

1 task / 1 scene / 4 rollout / 44 query / K16 的真实 `x0...x10` 数据显示：

- provisional/final action geometry rho 从 d0 的 `0.231` 到 d9 仅 `0.428`；
- d9 exact final medoid 仍只有 `20.5%`；
- 44/44 query 都到最终轮才首次达到 `rho >= 0.90`；
- round-only 对 remaining correction 的 OOF R2 为 `0.923`；
- round+route 为 `0.968`；
- round+current latent 为 `0.997`；
- round+latent+route 为 `0.997`，route 增量 `-0.00049`。

Route 确实能读出“还剩多少计算”，但 current latent 更强，route 没有条件增量。

### 5.2 Route 跟随什么

| relation | rho |
|---|---:|
| d0 route vs current latent | 0.880 |
| d0 route vs immediate update, controlling current latent | 0.147 |
| d0 route vs final action | 0.168 |
| d0 route vs final action, controlling current latent | -0.079 |
| d9 route vs final action, controlling current latent | 0.018 (`p=0.374`) |

因此更准确的解释是：route 表示当前 noisy latent 以及下一步如何修正它，而不是先于
latent 保存最终 action basin。前三轮 route cloud 还与 live7 initial-noise geometry
高度相关，rho `0.879`。

这些时序分析复用同一小型 capture，不是多次独立复现。

来源：[action commitment](action-commitment/REPORT.md)、
[route dynamics](unsupervised-moe-dynamics/report.md)、
[noise evolution](weighted-noise-evolution/report.md) 与
[route prefix](cumulative-moe-prefix/report.md)。

## 6. Functional MoE / expert-output

### 6.1 单任务 outcome 弱信号

Libero-Long 16x32 上，step-0 完整专家标量的严格双留出 AUC 为 `0.587`，选择增量
`+7.8 pp`；重训练置换分别为 `p=0.059` 和 `p=0.057`。无监督“高 expert-size
dispersion”增量为 `+6.2 pp`，raw `p=0.074`，FWER `p=0.615`。

Raw router center、低 entropy 和 route stability 均无收益。可保留的是一个边界性
线索：专家间抵消/离散结构可能比 expert ID 或 router probability 更有信息，但目前
既未过门槛，也没有跨任务确认。

### 6.2 Hidden/shared controls

五任务 hidden-matched functional proxy 的 task-mean AUC：full hidden `0.623`、shared
`0.662`、expert conflict `0.680`、DCQ `0.660`。关键增量均跨零：

- conflict-hidden：`+0.057 [-0.053,+0.162]`；
- conflict-shared：`+0.017 [-0.079,+0.111]`；
- DCQ-hidden：`+0.037 [-0.040,+0.118]`；
- DCQ-shared：`-0.003 [-0.090,+0.069]`。

严格 hidden-q10 / final-action-q90 twin 在五任务总共只有 45 个 divergent pair。

新增 raw-size hidden-matched 复核也为零结果：raw/mass level 的 macro AUC 约 `0.480`，
raw/mass pair difference 约 `0.520`，四项 maxT p 均大于 `0.92`。

### 6.3 Exact-seed template 控制

只用其他 state 的同 seed target template，预测总 flow displacement 的描述性 rho 已达
`0.971`。去掉该 template 后，固定 expert raw d6/d7/d8 只留下约
`0.93/0.95/0.73 pp` 的 pairwise ranking advantage，量级不足以支持 pruning 或
commitment。

来源：[MoE value signal](moe-value-signal/report.md)、
[raw-size hidden match](raw-size-hidden-matched/REPORT.md)、
[fixed-expert displacement](fixed-expert-flow-displacement/REPORT.md) 与
[mechanism ledger](mechanism/README.md)。

## 7. 跨 control 的 route dynamics

30-task corpus 覆盖 1,500 rollout、18,384 control query 和 16,884 个相邻转移：

- 相邻 route distance：`0.02423`；
- rollout 内随机顺序期望：`0.02798`；
- continuity ratio：`0.8659`；
- distance difference CI：`[0.00325,0.00427]`；
- task sign-flip：`p<0.0001`。

Route 不是每个 query 独立重置的白噪声，这一点是当前最稳健的正结果之一。

但五任务 matched finite-state Markov 控制显示：K16 route 的 first-over-zero gain 为
`2.638 bit/step`，proprio 为 `2.709 bit/step`，route 只在 1/5 task 胜出。不同 K
的符号也不一致。有限状态由 K-means 分辨率定义，不是自然发现的技能节点。

单任务 route-cloud phase ICC 为 `0.861`，但 robot-state distance 与 phase gap 的 rho
为 `0.908`；控制 robot state 后 route phase rho 只剩 `0.126`。因此“状态进度骨架 +
局部噪声响应”合理，“独立技能时钟”不成立。

来源：[control-route speed](control-route-speed/report.md) 与
[state-token Markov](markov-routing-all/REPORT.md)。

## 8. Intervention 能回答到哪一步

HB5/d0 v4 是唯一实际继续到 `x10` 的 matched runtime artifact，但只有 1 pair / 3 arms。
本轮重新验证：

- baseline no-op error：`0`；
- original pair difference：`0`；
- drop formula error：`1.53e-8`；
- drop final live7 RMS delta：`0.002062`；
- matched-random final delta：`0.002305`。

它只通过 instrumentation gate：专家扰动被正确实施并能传播到最终 latent。样本量无法
比较 drop 与 random，也没有环境执行、trajectory、event、continuation Q 或 success。

来源：[intervention validation](hb5-intervention-smoke.json) 与
[mechanism ledger](mechanism/README.md)。

## 最终解释

当前可以写：

> HiMoE routing 和 expert decomposition 对当前状态、当前 noisy action latent、局部
> 去噪更新以及跨 control 的连续变化提供了可读的内部诊断结构。

当前不能写：

> MoE route consensus 能选出更好的 action；route 在 action 尚未显现时已经形成高层
> 控制承诺；或 expert IDs 构成具有稳定行为语义和因果职责的技能状态机。

对“多数投票”思路的直接裁决是：**不要把 MoE route medoid 作为当前 selector。**
它在两类数据上都没有稳定 outcome gain，并且跨任务 probability medoid 几乎退化为
固定 seed 选择。

## 现数据无法完成的实验

- `A -> dense X -> contact/event -> s' -> repeated-CRN Q` 的完整行为链；
- 对 outcome equivalence interval 的正式检验；
- candidate-level repeated continuation value 和 pairwise critic；
- held-out task 的 runtime-exact expert patch/drop/swap 并继续到 simulator outcome；
- 基于真实 `O_K(s)` 的在线 adaptive-K；
- confirmatory 的 route consensus selector 或跨任务行为状态图。

这些问题需要新采集，不能通过增加现有 pair 数或重新切分同一 seed grid解决。

## 本轮实际执行

本轮在 CPU 上实际运行或重新运行：

```bash
python analyze_moe_consensus_audit.py
python analysis/moe-current-data-audit/cross-task/k32_route_medoid_audit.py
python analysis/moe-current-data-audit/cross-task/task_seed_heldout_increment_audit.py
python analyze_moe_opportunity.py --out-dir analysis/moe-current-data-audit/opportunity
python analyze_adaptive_k.py --run runs/flow-lead-cpu-t0s24
python analyze_route_outcome_geometry.py --mode legacy ...
python analyze_route_noise_selector.py --out-dir analysis/moe-current-data-audit/route-noise-selector
python analyze_moe_value_signal.py --out-dir analysis/moe-current-data-audit/moe-value-signal
python analyze_early_action_head.py --out-dir analysis/moe-current-data-audit/early-action-head
python analyze_action_commitment_pilot.py --out-dir analysis/moe-current-data-audit/action-commitment
python analyze_unsupervised_moe_dynamics.py --run runs/flow-lead-cpu-t0s24 ...
python analyze_weighted_noise_evolution.py --run runs/flow-lead-cpu-t0s24 ...
python analyze_cumulative_moe_prefix.py --run runs/flow-lead-cpu-t0s24 ...
python analyze_raw_size_hidden_matched.py --out-dir analysis/moe-current-data-audit/raw-size-hidden-matched
python analyze_fixed_expert_flow_displacement.py --out-dir analysis/moe-current-data-audit/fixed-expert-flow-displacement
python analyze_control_route_speed.py --out-dir analysis/moe-current-data-audit/control-route-speed
python analyze_markov_routing.py --all-tasks --out-dir analysis/moe-current-data-audit/markov-routing-all
python validate_hb5_intervention.py ...
```

所有详细 JSON、NPZ、图表和子报告位于当前目录及
`analysis/moe-consensus-audit/legacy-k32/`。机制账本还复核了不能安全重跑或不应被当成
新独立证据的历史 artifacts，并明确标记 withdrawn、engineering-smoke 和
preregistered-only 状态。
