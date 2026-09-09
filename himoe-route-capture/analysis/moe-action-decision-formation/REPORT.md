# 从随机噪声到最终 Action Chunk：MoE Experts 如何参与“决策形成”

## 摘要

本文考察的命题是：

> 随着动作方案从初始 flow noise 逐渐变成最终 action chunk，MoE experts
> 是否以可识别的方式参与了动作决策的形成？

现有实验支持一个有限但一致的回答：**HB-MoE 主要参与每轮局部去噪修正的
计算路径选择，而不是在早期选定一个语义化动作，再由后续轮次把它展开。**
router 会随当前 noisy action latent 改变；被选 experts 的加权输出参与本轮 flow
velocity；latent 更新后，下一轮重新路由。路由因此能读出“当前计算进行到哪里、
还剩多少修正”，但在控制当前 latent 或初始噪声后，对最终 action 的独立信息很弱。

动作几何本身直到最后一两轮才明显接近最终几何。与此同时，路由分布仅轻微收紧，
后段 expert identity 的重组反而更强。这不符合“某个 expert 逐渐取得控制权”的简单
故事，更符合一个分布式、反复重算的局部修正过程。

目前仍缺少正式的逐轮 expert intervention。已有 HB5/denoise-0 drop-one 仅完成
K=1 smoke，证明配对与干预公式正确，不能回答哪个 expert 对最终动作具有稳定因果
贡献。因此本文的结论是机制约束，不是 expert specialization 或因果归因结论。

## 1. 命题边界

这里必须区分三个不同问题：

1. **动作形成**：一次 policy query 内，模型如何把 `x0` 变成 `x10`。
2. **动作承诺**：在第 `tau` 轮时，最终 action chunk 是否已经基本确定。
3. **任务结果**：执行该 chunk 并继续 replan 后，整条 rollout 是否成功。

本文的主问题是前两项。最终 success 距离某个 MoE block 很远，混合了环境动力学、
后续观测、后续 flow noise 和多次 replan，不能直接作为单次 query 内动作形成的标签。

“expert 参与”也有不同强度：

- **结构参与**：expert output 出现在模型前向计算中。这由架构直接保证。
- **可读参与**：route 或 expert feature 能预测当前修正、剩余修正或最终 action。
- **独立参与**：在控制当前 latent、hidden、shared branch 和 noise 后仍有增量。
- **因果参与**：对 expert 做配对干预会稳定改变剩余去噪或最终 action。
- **语义参与**：特定 expert 在不同状态和任务中重复承担某种动作语义。

现有证据主要到达前两级；第三层只有弱信号；第四层只有仪器 smoke；第五层没有
得到支持。

## 2. 一次 Action Chunk 是怎样生成的

一次 policy query 先采样模型归一化的动作噪声 `x0`，随后执行 10 轮 flow denoise：

```text
observation, x_tau
        |
        v
transformer hidden h_tau
        |
        v
HB router: 每个 action token 选择 top-4 experts
        |
        v
sum_e w_e E_e(h_tau) + shared branch
        |
        v
flow velocity v_tau
        |
        v
x_(tau+1) = update(x_tau, v_tau)
```

每轮都会重新执行 action suffix 的 18 层 transformer，其中 8 层是 HB-MoE、
4 层是 AS-MoE、6 层是 dense。HB-MoE 同时处理 1 个 state token 和 10 个
action tokens。因此每次 query 有 `10 denoise x 8 HB layers = 80` 次 HB 层级
路由；每个站点选择 top-4/32 experts。环境只执行最终 action chunk，并不会执行
十个中间方案。

AS routing 由恒定 data mask 决定，在十轮内不变。与动作形成共同变化的主要信号是
action-token HB routing。

## 3. 证据地图

| 实验 | 数据范围 | 直接回答的问题 | 主要结果 | 证据等级 |
|---|---|---|---|---|
| action commitment pilot | 1 task、1 scene、4 rollouts、44 queries、K=16 | `x_tau` 何时接近 `x10`；route 能否读出剩余修正 | action geometry 到最后一轮才定型；route 是进度信号，但弱于 current latent | 机制 pilot |
| weighted noise evolution | 同上，保存 `x0...x10` | 初始噪声如何被改写 | 到 step 8 仍保留很强噪声几何，最后两步快速转成 action geometry | 机制 pilot |
| unsupervised MoE dynamics | 同上，完整 32-way HB route | route 对应当前 latent、下一步更新还是最终动作 | route 强对应当前 latent 和局部更新；控制 current latent 后几乎不对应最终 action | 置乱支持的关联 |
| K32 route trajectories | Long/Moka 同一状态，32 完整 rollouts | route 在十轮及多次 replan 中如何变化 | 分布轻微收紧，后段重组增强；成功/失败无简单 route 分界 | 描述性完整轨迹 |
| expert activation future | Long t08，16 scenes x 32 seeds | expert output 是否预测最终 action/outcome | 早期方向有弱增量，但 shared 更强；幅值不稳定 | held-out scene 关联 |
| fixed-expert flow displacement | 5 tasks x 16 states x 32 seeds | 同一 expert 的幅值是否解释 `x0 -> x10` 位移 | 严格 seed-template 控制后仅约 0.7-0.95 pp 排序增益 | 多任务观察性关联 |
| raw internal activation correction | 5 tasks x 8 states x 16 candidates | pre-down `m_e` 是否增加剩余修正预测 | 相对完整 baseline 变差，五任务 0/5 改善 | 双轴 held-out screen |
| HB5 drop-one v4 smoke | 1 pair、3 arms | expert 干预能否被正确执行和追踪 | 数值与配对审计通过；样本不足以作科学判断 | 工程 smoke |

## 4. 动作不是平滑地提前定型，而是在末段快速落定

真实 latent pilot 直接保存了 `x0...x10`。在每个 K=16 pool 内，将当前
`x_tau` 的候选间 pairwise geometry 与最终 `x10` geometry 比较：

| 已完成 denoise 轮数 | 当前/最终 geometry Spearman | exact final medoid |
|---:|---:|---:|
| 0 | 0.231 | 15.9% |
| 3 | 0.237 | 15.9% |
| 5 | 0.247 | 15.9% |
| 7 | 0.276 | 20.5% |
| 9 | 0.428 | 20.5% |
| 10 | 1.000 | 100.0% |

逐轮剩余修正比例在 step 8 仍为 `0.207`，step 9 为 `0.104`，到 step 10 才为
零。raw live-7 geometry 与初始噪声的相关到 step 8 仍为 `0.971`，step 9 为
`0.899`，随后才转为最终 action geometry。

这说明“决策形成”不能描述成一个候选动作在早期已经完成，只是在后面做小幅抛光。
在当前 sampler 中，初始噪声结构长期保留，最终 action 的候选几何主要在最后一两步
显现。

不过，初始噪声也不是与最终动作完全无关。对完整 24 维 `x0` 学习跨维线性映射，
held-out rollout 的 final-action basin AUC 可达到 `0.797`。更准确的表述是：
初始噪声携带可读出的未来方向，但 raw latent geometry 直到末段才真正成为最终动作
geometry。

来源：

- [Action commitment pilot](../action-commitment-pilot/REPORT.md)
- [Weighted noise evolution](../weighted-noise-evolution/flow-lead-real/report.md)

## 5. Router 在十轮中做了什么

### 5.1 分布只轻微收紧

K32 完整 rollout 中，十轮平均路由统计为：

- top-4 probability mass：`0.1396 -> 0.1473`；
- normalized entropy：`0.999066 -> 0.997909`；
- 相邻内部轮 Hellinger change：约 `0.0042 -> 0.0127`。

32-way 分布一直接近均匀。后段虽然 top-4 mass 增加、熵下降，但量级很小，不能称为
winner-take-all 或某个 expert 的逐步垄断。

### 5.2 后段不是“锁定”，而是更强重组

相邻 denoise step 的 top-1 expert 保持率从 `tau0->1` 的 `0.744` 降到
`tau8->9` 的 `0.591`。route geometry 相对 tau0 的稳定性也从接近 1 降到
tau9 的 `0.639`。这表示动作接近最终形态时，路由并没有冻结，反而发生更明显的
expert identity 重排。

### 5.3 Route 首先是当前 noisy latent 的投影

同观测、不同噪声候选内的距离分析如下：

| 指标 | tau0 | tau9 |
|---|---:|---:|
| route vs current latent | 0.880 | 0.585 |
| route vs immediate update，控制 current latent | 0.147 | 0.324 |
| route vs final action | 0.168 | 0.272 |
| route vs final action，控制 current latent | -0.079 | 0.018 |

tau9 的最后一项 `p=0.374`。因此 route 与最终 action 的表面相关主要通过当前
latent 传递；route 额外描述的是模型接下来如何修正当前 latent，而不是独立保存一个
尚未显现在 latent 中的最终动作编码。

### 5.4 Route 是可用的计算进度传感器，但不是独立决策头

预测 pool RMS remaining correction 时：

| 可观察量 | leave-one-rollout-out R2 |
|---|---:|
| flow round | 0.923 |
| flow round + route | 0.968 |
| flow round + current latent | 0.997 |
| flow round + current latent + route | 0.997 |

route 明确包含“这次计算还有多不完整”的信息；但 current latent 已经包含更多，加入
route 的变化约为 `-0.0005`。所以 route 可以是诊断传感器，却尚未显示为 current
latent 之外的决策变量。

来源：

- [Unsupervised MoE dynamics](../unsupervised-moe-dynamics/flow-lead-real/report.md)
- [K32 route trajectories](../k32-route-trajectories/report.md)
- [Action commitment pilot](../action-commitment-pilot/REPORT.md)

## 6. 跨 Control 的路由结构：状态骨架加噪声响应

在 4 条 rollout x 11 个连续 controls 的 sibling-noise 数据中，早期 route 可分成
两个描述性成分：

```text
route(control, noise) ~= phase/state backbone(control)
                      + local noise response(control, noise)
```

16-noise route cloud 的中心在不同 rollout 的同一 control stage 高度复现，完整前缀
phase ICC 为 `0.861`；单次执行 route 的 ICC 为 `0.513`。前三个 route evaluation
内，live-7 noise-route rho 为 `0.879`。浅层 HB2-5 更像平滑状态进度轴，深层
HB12-15 的同阶段 motif 更稳定但不随时间差单调变化。

这个结果不应解释为独立“内部时钟”：robot-state distance 与阶段差的 rho 为
`0.908`；控制 robot state 后，route center 的剩余阶段 rho 只有 `0.126`。因此
backbone 很可能主要编码当前状态进度，noise response 则决定当前局部计算路径。

在更大的 30-task、1,500-rollout 审计中，相邻 control 的 route distance 为
`0.0242`，明显小于 rollout 内随机 control 配对的 `0.0280`，说明 route 也不是每次
query 完全重置的白噪声。

来源：

- [Cumulative MoE prefix](../cumulative-moe-prefix/flow-lead-real/report.md)
- [Control-route speed](../control-route-speed/report.md)

## 7. Expert Outputs 是否提供了 Router 之外的决策信息

Router probability 只说明选择和权重，不能替代真实 expert output。相关实验依次检查了
加权 expert output、未加权 post-down output，以及 pre-down 内部激活。

### 7.1 早期 expert direction 有信息，但不显示为 expert-specific 优势

Long t08 的 16 scenes x 32 seeds 实验比较了有效初始噪声、router、expert size、
routed direction 和 shared direction：

- step 0 的 live-noise action-basin AUC：`0.674`；
- expert size 在 noise 上的最大增量：`+0.015`；
- routed expert direction 的最大增量：`+0.060`；
- shared direction 的最大增量：`+0.092`；
- router 在有效噪声上的增量约为 `0.000-0.002`。

完整向量方向确实包含未来 action 信息，但 shared branch 比 routed expert branch 更强。
因此不能把方向信号归因于 expert specialization。expert size 的增量很小，并在后半程
降到约零。

对最终 success，step 0 expert size 的探索性条件 AUC 为 `0.585`，比 scene+noise
基线高 `0.117`，但固定正则下 log loss 反而更差，且跨 denoise step 不稳定。这不是
可靠 outcome predictor，更不是因果效应。

来源：[Expert activation future](../expert-activation-future/long-t08/report.md)

### 7.2 控制 seed template 后，幅值增量极小

五任务 fixed-expert flow-displacement 审计发现，`x0 -> x10` 总位移高度受初始噪声
幅值和 exact-seed template 支配。只用其他 states 的同 seed target template，
描述性 rho 已达 `0.971`。

在 discovery states 构造 template、对 validation states 的 residual 做严格检验后，
raw expert output 在 d6-d8 仍有五任务同向关系，但效果仅相当于：

- raw d6：约 `+0.93 pp` pairwise ranking advantage；
- raw d7：约 `+0.95 pp`；
- raw d8：约 `+0.73 pp`。

这个量级不能支持 candidate pruning、early stop 或动作承诺判断，而且仍然是
selected-only 的观察性关联。

来源：[Fixed-expert flow displacement](../fixed-expert-flow-displacement/REPORT.md)

### 7.3 Pre-down 内部激活没有增加 held-out 预测

最新修正实验严格区分：

- router weight `w_e`；
- pre-down activation `m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`；
- pre-gate output `v_e = down_proj_e(m_e)`；
- routed contribution `w_e v_e`。

在 HB5/d0 上，完整 baseline 已含 `x0`、`x1`、hidden、shared、full router、selected
IDs/weights 和实际 routed vector。加入每 token 排序后的四个 raw `||m_e||_2` 后，
equal-task macro relative RMSE gain 为 `-0.547%`，95% interval
`[-0.735%, -0.388%]`，五任务 `0/5` 改善。

也就是说，pre-down 幅值没有显示出 baseline 之外的剩余修正信息。它不能被解释为
expert confidence，且 raw pre-down magnitude 本身还受参数化尺度影响。

来源：[Raw internal activation correction](../raw-internal-activation-correction/observational-hb5-d0/REPORT.md)

## 8. Early Route 是否已经决定最终结果

一个 16 first-routes x 8 common-future-streams 的实验固定完整 control-step-0 route 和
first action，只改变后续 flow noise：

- 16/16 rows 同时出现 success 和 failure；
- `H(outcome | first route) = 0.9357 bits`；
- plug-in mutual information 为 `0.0466 bits`；
- mutual-information permutation `p=0.921`。

因此相同 early route 可以通向不同最终结果。早期 route 既不是 success 的充分条件，
也没有在这个 grid 中表现为稳定 outcome selector。

K32 完整轨迹中，成功与失败的平均 top-4 mass 只差 `-0.000540`，平均 route speed
只差 `-0.000214`，同样没有显示简单的成功专家模式。

来源：

- [Early route commitment grid](../commitment-grid-s24/REPORT.md)
- [K32 route trajectories](../k32-route-trajectories/report.md)

## 9. 当前最合理的机制解释

综合这些结果，现阶段最受支持的解释是：

1. `x0` 提供候选级随机差异，同时包含可经跨维映射读出的未来 action 偏置。
2. observation、robot state 和任务上下文在 HB routes 中形成共享的状态/阶段骨架。
3. 每轮 router 根据当前 `x_tau` 和 hidden，为每个 action token 分配 top-4 experts。
4. experts 与 shared branch 共同产生本轮局部 flow correction。
5. `x_tau` 更新后，下一轮重新路由；后段 latent 变化与 route 重组都加速。
6. 最终 action geometry 在最后一两轮落定，而不是由 early route 单独承诺。

可以把 HB-MoE 的作用概括为：

> **状态条件下、噪声条件下的局部修正计算图。**

它更像动态选择“当前这一轮怎样算”，而不是静态选择“最终要做什么”。

### 已支持

- action-token HB route 对当前 noisy latent 高度敏感；
- route 与下一步局部修正存在控制 current latent 后的关系；
- route 能读出计算完成度；
- route 同时含状态/阶段 backbone 与 candidate-noise response；
- expert/shared 向量方向含有部分未来 action 信息；
- K=1 smoke 中，一个 expert block 的扰动可以沿剩余 flow 传播。

### 尚未支持

- 某个 expert 对应固定动作语义；
- route 在 early denoise 中已经选定最终 action basin；
- expert magnitude 是 confidence 或 commitment；
- route 比 current latent 或 known noise 更早知道最终 action；
- success/failure 存在简单稳定的 expert signature；
- 基于 expert size 的 pruning 或 early stop 已经可用。

## 10. 证据的有效性边界

上述结果不能被看成同一批独立、确认性实验：

- action commitment、weighted noise evolution 和 unsupervised MoE dynamics
  共享一个 task、一个 scene 和 4 条相关 rollouts；它们从不同角度分析同一机制数据，
  不能算三次独立复现；
- K32 route trajectory 只有 Long/Moka 的一个初始状态，适合展示完整动力学，不足以
  证明跨任务普遍性；
- 30-task control-route speed 支持 route 的时间连续性，但没有使用 action 或 outcome，
  因而不能单独支持动作语义；
- 五任务 expert-output 分析扩大了任务范围，但主要是 first-query、selected-only、
  offline reconstruction 的观察性结果；conditioning on selected experts 可能诱导关联；
- 同一 K=16/K=32 pool 内的大量 candidate pairs 高度依赖。有效推断单位应是
  task/state/rollout/candidate，而不是把成千上万的 pairs 当作独立样本；
- 部分分析属于 retrospective exploration。本文采用修正后的报告和严格 control，
  但这些结果不能替代新的 preregistered holdout；
- success 是远端 endpoint，不能用来决定单次 query 内何时完成 action commitment；
- 尚无 all-denoise runtime intervention，因此目前不能识别稳定的 expert-level
  因果分工。

所以本文给出的是当前数据约束下的最佳机制解释，而不是完成的因果机制鉴定。

## 11. 因果证据目前到哪里

HB5/d0 v4 drop-one smoke 使用完全相同的 observation 和显式 `x0`，比较 baseline、
drop 和 matched-random 三个 arms。审计结果包括：

- baseline no-op error：`0`；
- pair 内原始量最大差异：`0`；
- drop formula 最大误差：`1.53e-8`；
- drop arm 的 final live-7 RMS delta：`0.002062`；
- matched-random arm：`0.002305`；
- 从初始 perturbation 到 final delta 的 amplification：约 `6.02x` 和 `7.25x`。

这证明干预可以正确执行，且一个 HB5/d0 block perturbation 能传播到 `x10`。但它只有
一个 pair，不能判断哪种 expert removal 更安全，也不能估计跨状态或跨任务效应。

runtime-exact v5 已冻结以下 drop policies：

- minimum pre-down internal L2；
- minimum router weight；
- minimum pre-gate output L2；
- uniform-random selected slot；
- 对应 maximum policies 作为 controls。

截至本文，workspace 中只有 v5 preregistration 和实现，没有正式 v5 capture/result。

来源：

- [v4 K=1 validation](../../runs/hb5-intervention-v4-smoke-goal-k1/validation.json)
- [Runtime-exact v5 preregistration](../raw-pruning-v5/PREREGISTRATION.md)

## 12. 能真正回答命题的下一步实验

现有 v5 只在 HB5/d0 干预，能够回答 pruning damage，但不足以回答“决策怎样逐轮形成”。
建议把正式实验定义为 **all-denoise matched expert intervention**。

### 12.1 配对单位

固定：

- task、source state、observation；
- candidate `x0`；
- 当前真实 baseline latent `x_tau`；
- 除 intervention 外的全部计算和随机量。

统计独立单位必须是 task/state/candidate，不能把 token、expert slot 或 candidate pair
当作独立样本。

### 12.2 干预时间

至少预注册 `tau in {0, 3, 6, 9}`，覆盖 early、middle、late 和 final-preparation。
若成本允许，应记录全部十轮，但不能看结果后再选择“最有故事”的 round。

### 12.3 干预 arms

每个时间点至少包括：

- baseline no-op；
- drop minimum internal-L2 expert；
- drop minimum router-weight expert；
- drop minimum raw-output expert；
- uniform-random selected expert；
- matched-norm random direction control。

drop 后继续完成剩余 flow rounds，并记录 downstream routes，不能只计算当前 block 的
离线 output delta。

### 12.4 主 endpoints

1. **即时作用**：`RMS(delta v_tau)` 或 `RMS(delta x_(tau+1))`。
2. **最终作用**：`RMS_live7(x10_arm - x10_baseline)`。
3. **剩余路径作用**：两臂 `x_(tau+1)...x10` 的 trajectory distance。
4. **放大/修复**：final delta / initial delta，以及 downstream route 是否补偿。
5. **动作几何**：干预是否改变 candidate basin rank 或 medoid identity。

success 和 environment outcome 只能作为后续 secondary endpoint；先确认 action-level
因果效应，再决定是否值得支付 continuation rollout 成本。

### 12.5 判定规则

只有满足以下条件，才能把结果写成 expert 参与“决策形成”的因果证据：

- intervention 对 `x10` 的影响超过数值与 matched-random control；
- 影响在 held-out states/tasks 上同向复现；
- 效应随 `tau` 呈预注册的时间结构，而不是只在事后挑选的 round 出现；
- 控制当前 latent、velocity、hidden、shared output 和 router weight 后仍有 expert-policy
  差异；
- bootstrap/permutation 的单位是 task/state/candidate。

若某 expert ID 在跨任务、跨状态和 matched-hidden 条件下仍产生可重复方向效应，才有
资格继续讨论语义 specialization。仅凭 route ID 频率或 output norm 不足以命名专家。

## 13. 最终裁决

对原命题，当前最稳妥的回答是：

> MoE experts 确实参与了从噪声到 action chunk 的每轮计算，但现有证据显示这种参与
> 主要是对当前 noisy latent 的动态、分布式局部修正。router 跟踪当前计算状态并选择
> 本轮处理路径；experts 与 shared branch 一起产生 flow update；latent 改变后再重新
> 路由。最终 action 主要在最后一两轮形成，而不是由早期某组 experts 提前决定。

现有观察性实验不支持“早期 route 就是最终意图”或“expert 幅值就是置信度”这两个
简单版本，但还不足以证明单个 experts 在决策形成中的稳定因果分工。完成逐轮、配对、
common-state 的 runtime intervention 后，才能把“局部计算路径”进一步分解为可归因
的 expert contribution。
