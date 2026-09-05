# HiMoE-VLA 前后层路由结构传递实验

## 结论先行

这次实验真正利用了“前层吸收噪声、后层形成相对结构”，但结论需要分成两层：

1. **结构传递机制成立，而且跨两套同快照扰动数据非常稳定。** 前层的候选间路由差异随 flow 收缩，后层的 action 相对结构重新展开。
2. **结构传递强弱本身不是 trap 分数。** 在远离 trap 的 q0，同快照 fixed composite 对 loop/static 基本是随机水平。真正对 loop 有用的是结构传递在闭环 query 轴上的失稳，尤其是 q-2 的前层条件图加速度。

因此可用的识别原则不是“后层越展开越危险”，而是：

> 先把正常的前层消噪到后层结构化看成一个传递算子，再监测该算子是否在 query 时间上突然过校正、错相或停滞。

整个流程没有训练神经网络、分类器或 outcome 权重。原始特征构建阶段完全不加载结果标签；标签只在之后的 matched-control 评估中使用。需要说明的是，最终 loop 轴从 main16x32 的 584 个轴中选择，因此它是**无训练但有发现集选择**，不能把 main16x32 数字当作无偏泛化结果。

## 1. 为什么上一版没有充分利用这个机制

只分别计算前四层和后四层的 entropy、rank 或图密度，回答的是“前层是什么样、后层是什么样”，没有回答：

- 同一份噪声扰动在前层被压掉了多少；
- 被压掉的公共成分如何在后层转化为 action-token 间的相对关系；
- 这个前到后的映射在连续 query 上是否稳定。

本版保留完整的 `[8 layers, 10 flow steps, 11 tokens, 32 experts]` 概率张量，以 token-expert 二部图为基础，同时建模 layer、flow 和 query 三个时间/深度方向。

## 2. 状态条件化 action graph

对每个 layer 和 flow step，将 token `u` 的路由概率写成 Hellinger 嵌入：

```text
r_u = sqrt(p_u),       ||r_u||_2 = 1
G_uv = <r_u, r_v>
```

`G` 是 11 个 token 的 Gram 图。令 token 0 为 state token，token 1..10 为 action tokens。定义状态条件化 action Gram：

```text
K_A|S = G_AA - G_AS G_SA
```

它等价于先把每个 action routing vector 在 state routing 方向上的投影去掉，再计算 action-action Gram。进一步归一化得到 partial affinity：

```text
P_uv = K_uv / sqrt(K_uu K_vv)
```

这一分解有三个直接作用：

- 去掉所有 action token 共同继承的 state-aligned 成分；
- 保留 10 个 action token 在 32 个 expert 概率空间中的相对几何；
- 避免只看 top-1 expert ID 或单 token entropy 所造成的信息损失。

前层取 layers 0..3，后层取 layers 4..7。除了 `action_conditional` 和 `action_partial`，还保留原始 action relation、中心化 shape、state relation，以及前层 expert-edge 到后层 action graph 的动态通道。

## 3. 数据和计算审计

| 项目 | 数字 |
|---|---:|
| 任务数 | 45 |
| episodes | 18,560 |
| queries | 305,030 |
| 单 query 原始路由张量 | 8 x 10 x 11 x 32 |
| query 内传递特征 | 280 |
| 跨 query 动态特征 | 304 |
| 总动态轴 | 584 |
| 特征构建时加载 outcome | 否 |
| 拟合权重 | 否 |
| task-conditioned 特征 | 否 |

GPU 使用 `CUDA_VISIBLE_DEVICES=6,7`。两张 H20-3e 并行生成全量条件图 profile，每张峰值显存约 369.77 MiB；同快照 fork 分析峰值分别为 303.71 MiB 和 523.21 MiB。该工作主要受 Zarr I/O 限制，因此没有为了占满显存而复制无用批次。

对原始 state-token routing 做了逐 query 全量审计。统计量是所有 layer/expert 上，10 个 flow step 的最大概率跨度：

| corpus | queries | flow 上严格不变 | 中位最大跨度 | p95 | p99 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| main16x32 | 51,308 | 100.0% | 0 | 0 | 0 | 0 |
| grid50x8 | 253,722 | 36.46% | 3.36e-4 | 8.24e-4 | 2.93e-3 | 2.20e-2 |

所以 state token 在 main 中是严格 flow-invariant，在 grid 中通常近似稳定但并非严格不变。`K_A|S` 每个 `(layer, flow)` 独立消除当下 state direction，本身不要求 state 跨 flow 恒定；state 的残余漂移也作为单独的 `state_relation/state_shape` 动态保留下来。

## 4. 机制结果：确实存在前收缩、后结构化

对同一 simulator snapshot 的 K 个独立 noise candidates，在每个 flow step 计算候选间结构向量的 pairwise median distance `D_f`，并定义：

```text
rho_front = D_front,9 / D_front,0
rho_back  = D_back,9  / D_back,0
handoff   = rho_back / rho_front
```

`rho < 1` 表示 flow 过程中吸收候选差异，`rho > 1` 表示候选间相对结构展开。

| 数据 | 快照 x K | 图视图 | front rho | back rho | paired handoff | 正 handoff |
|---|---:|---|---:|---:|---:|---:|
| fork pilot | 20 x 32 | action conditional | 0.763 | 1.595 | 1.695 | 19/20 |
| fork pilot | 20 x 32 | action partial | 0.612 | 2.807 | 4.295 | 20/20 |
| rolling fork | 22 x 16 | action conditional | 0.562 | 1.006 | 1.695 | 22/22 |
| rolling fork | 22 x 16 | action partial | 0.399 | 1.357 | 3.479 | 22/22 |

这里的 `paired handoff` 是逐快照 `log(rho_back)-log(rho_front)` 的中位数再取指数，不要求等于两列总体中位数之比。rolling 数据上 action-partial 的 paired log-handoff 95% bootstrap CI 为 `[1.133, 1.315]`，20-snapshot pilot 为 `[0.391, 2.074]`；符号检验均显著。

这比“前层 entropy 低、后层 entropy 高”更强：它是在**相同物理状态、不同噪声候选**之间直接观察到同一扰动先被压缩、再被转换成 token-relative geometry。

## 5. 对 trap 识别究竟有什么用

### 5.1 先排除一个错误用法

把 `rho_front`、`rho_back` 或 handoff ratio 直接当成风险，在 q0 fork 上失败：

| fixed q0 score | mixed snapshots | direction-locked AUC | 95% bootstrap CI |
|---|---:|---:|---:|
| loop composite | 18 | 0.484 | [0.400, 0.572] |
| static composite | 12 | 0.511 | [0.381, 0.648] |

也就是说，健康候选和未来 trap 候选都可以经历前收缩、后展开。handoff 是模型的正常计算方式，不是异常本身。

### 5.2 正确用法：监测 handoff 的时间控制是否失稳

发现集上最强且在外部 corpus 保持同方向的 loop 轴是：

```text
query|action_partial|front_acceleration
```

它计算前层 state-orthogonal action partial graph 在连续三个 query 上的二阶差分范数。高值表示前层“噪声吸收器”不是平滑适应，而是在 q-2 突然改变其相对结构投影，形成 over-correction precursor。

锁定轴和方向后，使用 main16x32 的经验分位数做无参数尺度统一，并与旧的 within-flow curvature baseline 等权融合：

```text
S_loop(q) = 0.5 * percentile_main(old_curvature(q))
          + 0.5 * percentile_main(front_partial_acceleration(q))
```

这不是概率，也没有拟合 outcome 权重。

### 5.3 q-2 数字

| corpus | score | event-mean AUC | 95% CI | pair-weighted AUC |
|---|---|---:|---:|---:|
| main16x32, discovery | old baseline | 0.622 | [0.554, 0.688] | 0.659 |
| main16x32, discovery | conditional axis | 0.726 | [0.667, 0.783] | 0.674 |
| main16x32, discovery | equal fusion | 0.743 | [0.683, 0.800] | 0.722 |
| grid50x8, held-out direction | old baseline | 0.674 | [0.630, 0.717] | 0.663 |
| grid50x8, held-out direction | conditional axis | 0.658 | [0.614, 0.702] | 0.624 |
| grid50x8, held-out direction | equal fusion | 0.709 | [0.668, 0.750] | 0.677 |

paired event-mean 增量：

- main16x32：`+0.122`，95% CI `[+0.053, +0.191]`；这是发现集数字，偏乐观。
- grid50x8：`+0.035`，95% CI `[-0.013, +0.084]`；方向上复现，但增量区间跨 0。

因此最诚实的表述是：**条件图动态提供了与旧 curvature 不同的信息，并把 held-out event-mean AUC 从 0.674 提到 0.709；但外部增量尚未达到确定性证据。** held-out pair-weighted 增量更小，为 `0.677-0.663=+0.014`。

### 5.4 这个信号是 q-2 precursor，不是任意早期异常

| offset | main fusion AUC | grid fusion AUC |
|---:|---:|---:|
| -8 | 0.491 | 0.550 |
| -6 | 0.548 | 0.528 |
| -4 | 0.548 | 0.532 |
| -2 | 0.743 | 0.709 |
| 0 | 0.620 | 0.781 |

它在 q-2 才明显抬升，符合“前层结构投影在进入 loop 前突然过校正”的机制，而不是一个静态 episode 难度代理。

## 6. Loop 与 static 不能共用同一个 handoff 分数

static 的最强结构是 back graph 在 query 轴上的 lag-4 motion 下降；例如 `query|action_conditional|back_lag4` 取低风险方向，在 main/grid 上分别达到 0.967/0.958。但旧 static baseline 已经是 0.984/0.983，新的 fixed fusion 为 0.969/0.990，没有跨 corpus 一致改善。

所以当前建议是：

- **loop**：旧 within-flow curvature + 前层 conditional/partial graph 的 query acceleration；
- **static**：继续使用旧 detector，back lag-4 arrest 只保留为结构解释和冗余证据；
- 不把二者压成统一 `MoE confidence`。

q0 的 324-axis post-hoc 扫描出现了一些可研究信号，例如 loop 的 action-shape alignment change 两侧 AUC 0.620、static 的 action-partial lag-alignment-p2 两侧 AUC 0.695；但这些都没有做 324 重比较校正，不能作为确认性结论。

## 7. 本质解释

这项机制对路由检测的价值可以压缩为三个判断：

1. **前层是 nuisance filter。** 独立噪声造成的 raw route spread 大量会被正常吸收，所以单次 gate 波动不应直接报警。
2. **后层是 relational constructor。** 后层重新展开并不等于不稳定；应观察去掉 state 共性后的 action 相对图，而不是只数 expert occupancy。
3. **trap 是 transfer control failure。** loop 更像前层投影在 query 轴上的突然过校正/错相，static 更像后层相对图停止更新。异常发生在“结构如何被传递”上，而非某一层绝对尖或平。

这也解释了为什么使用更多 MoE 信息并不意味着堆更多 entropy 标量。有效的信息增量来自：

```text
all expert probabilities
  -> token Gram geometry
  -> state-conditioned action geometry
  -> front/back flow transfer
  -> cross-query acceleration / lag / recurrence
```

## 8. 限制与下一次决定性实验

- 外部 corpus 只锁定了方向，轴本身来自 584-axis main 扫描；需要一套完全新 corpus 做最终无偏复现。
- q0 fork 离 trap 较远，只能证明 population handoff，不能直接检验 q-2 precursor。
- 当前 fork 是独立完整 noise streams，不是严格的 infinitesimal `xi` 与 `xi + epsilon` common-random-number 扰动。
- 条件 Gram 是 Hellinger 空间中的线性投影，只移除 state routing direction；它不等价于因果干预或非线性条件独立。
- grid50x8 的 state token 不是严格 flow-invariant；极端 query 的最大概率跨度达到 0.022，不能把它当作完全固定的外生变量。
- 当前只使用 router probabilities，没有 expert output disagreement。

最有信息量的下一次采集不是训练，而是在 loop onset 的 `q-4`、`q-2` 和 matched healthy query 保存相同 simulator snapshot，然后对每个 snapshot 做 K=8 或 K=16 的 paired `xi, xi+epsilon` continuation。预先锁定三个检验：

```text
H1: front action-partial acceleration 在 q-2 升高
H2: front contraction -> back differentiation 的 handoff delay 在 q-2 缩短或错相
H3: 上述变化与两步 committor P(loop within 2) 单调相关
```

这一步仍不需要训练；它会直接回答当前 `+0.035` held-out 增量是真机制，还是发现集选择后的弱相关。

## 9. 复现

在本目录执行：

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=6,7 python experiments/build_transfer_profiles_gpu.py \
  --output results/conditional_profiles
python experiments/evaluate_transfer.py \
  --profile-root results/conditional_profiles \
  --output results/conditional_evaluation
python experiments/evaluate_locked_loop_detector.py --bootstrap 20000
CUDA_VISIBLE_DEVICES=6,7 python experiments/analyze_conditional_forks_gpu.py \
  --bootstrap 10000
python experiments/audit_state_anchor.py
pytest -q
```

主要机器可读结果：

- `results/conditional_profiles/build_summary.json`
- `results/conditional_evaluation/summary.json`
- `results/locked_loop_detector/summary.json`
- `results/conditional_forks/summary.json`
- `results/state_anchor_audit.json`
