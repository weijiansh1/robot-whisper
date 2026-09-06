# MoE Routing Grammar v2：实验判决

## 结论

这轮修正后，可以保留的结论是：

$$
\boxed{\text{健康 MoE routing 前缀具有连续时序可预测性。}}
$$

目前不能保留的结论是：

$$
\boxed{\text{这种前缀结构已经能够可靠地提前发现物理 Trap。}}
$$

在现有 Scene8 stasis 锚点上，当前 query、完整前缀、持续性、复返和 return
通道都没有形成可部署的早检器。候选实验出现了正向的健康 emission compatibility，
但没有观察到完整前缀相对 no-prefix 的增量。

## 修正内容

1. 删除未对齐跨层 expert ID 的直接比较，改为每层置换不变标量 profile 的离散度。
2. 主模型使用 22 维连续 query phenotype，不再经过 2187D、whitened PCA、hard GMM word。
3. 原生 velocity 和 acceleration 不再被统一重复差分。
4. 同时实现连续 VAR(4) 和 soft Gaussian HMM；HMM 用 forward belief 边缘化完整前缀。
5. 检测拆成 innovation、over-regularity、return-failure，不再使用单边上尾 CUSUM。
6. 阈值按 success episode 最大分数校准，并额外给出固定 5% FPR operating point。
7. 候选从同一 sim/policy snapshot 出发；本地 Top-4 route IDs 与服务端 32-way gate
   Zarr 逐 query 精确回查，允许并发 episode 行交错，但不允许近似匹配。

## 数据与隔离

- full40：32,000 episodes，508,023 queries，30,904 successes。
- Scene8：512 episodes，其中 296 successes，197 个可靠物理 stasis onset。
- 五折按 init state 隔离；测试状态不参与健康模型拟合。
- 严格在线输入只有 MoE routing，不使用 task ID、绝对 query、动作、物理状态或 outcome。
- 当前语料没有可靠 loop onset，因此没有报告 loop 早检结果。

## 健康前缀学习

连续 VAR 学习：

$$
z_q = \sum_{k=1}^{4} A_k z_{q-k} + \epsilon_q.
$$

soft HMM 学习：

$$
b_q(m) \propto p(z_q\mid m)\sum_{m'}b_{q-1}(m')P(m\mid m').
$$

held-out success 结果：

| 比较 | 历史增益 |
|---|---:|
| VAR(4) vs current-query Gaussian | 0.2174 bits/phenotype |
| full-prefix HMM vs clock-only HMM | 0.0118 bits/phenotype |

因此，历史确实含下一 query 信息；但 HMM 中绝大部分健康 phase 可预测性可由内部时钟解释。

## 可解释离散句法复验

这轮额外直接实现了五条可同时激活的三值轨道：

$$
[Fl/\varnothing/Sh,\ Ds/\varnothing/Sy,\ LR/\varnothing/HR,\
Lk/\varnothing/Sw,\ Ff/\varnothing/Fs].
$$

主 tokenizer 逐 query 独立编码，不使用滞回，因此不会先人为制造 `Fl Fl` 再把它解释为
时序结构。所有阈值只由 density-state 成功 episode 拟合，并按 episode 等权。

| 比较 | held-out 增益 | init-state cluster 95% CI | 正方向 states |
|---|---:|---:|---:|
| ordered last-4 vs 相同四词 bag | 0.0301 bits/track | [0.0285, 0.0317] | 50/50 |
| full-prefix categorical HMM vs clock | 0.0179 bits/track | [0.0173, 0.0185] | 50/50 |

因此，现在可以更强地确认：

$$
\boxed{\text{健康 routing 和弦具有小但稳定、跨 init-state 的局部离散顺序结构。}}
$$

历史长度消融进一步限定了这个结论。order-4 相对 order-1 的增益为 0.01893
bits/track，其中 q-2、q-3、q-4 的逐级贡献分别为 0.01593、0.00262、0.00038。
full-prefix HMM 相对“clock + 仅上一个和弦”反而差 0.00235 bits/track，49/50 states
方向为负。物理 stasis 上两者 q-3/onset 召回完全相同，matched AUC 为 0.481/0.479。
所以当前不能把局部 2-3 query 依赖升级成“整个前缀的全局语法”。

但预定义 `Fl Fl Sy Sy` 在 85.5% 的成功 Scene8 episode 中也出现，物理 onset 的
state/query-matched AUC 为 0.508。固定 4.7% success-episode FPR 时，ordered-4 的 q-3
召回为 0%，full-prefix HMM 为 0.5%，离散三通道并集为 5.1%。所以该实验同时否定了：

$$
\boxed{\text{`Fl Fl Sy Sy` 本身是 stasis 特异病句。}}
$$

完整结果见 `results-discrete-grammar-v1/`。

## 严格十流单词复验

上面的离散审计仍把一个 query 汇总成一个五轨和弦。为直接回答 `Fl Fl Sy Sy` 为什么
不作为最终句法，又做了一版不压缩 flow 轴的实验：每个 query 保留 10 个 flow step，
每步保留五个三值轨道，因此一个 query word 是完整的 $10\times5=50$ 个符号。

Query 内 held-out 结果为：

| 比较 | 增益 bits/flow-track | init-state cluster 95% CI | 正方向 states |
|---|---:|---:|---:|
| order-2 vs order-1 | 0.01009 | [0.00994, 0.01026] | 50/50 |
| order-3 vs order-2 | 0.00035 | [0.00028, 0.00043] | 46/50 |
| order-4 vs order-3 | -0.00067 | [-0.00072, -0.00061] | 0/50 |
| ordered-4 vs bag-4 | 0.00176 | [0.00165, 0.00187] | 50/50 |

所以 flow 内确有顺序结构，但主体是相邻两三步的生成 morphology；继续加长 context
已经过拟合。跨 query 时，完整 50-symbol word-prefix 相对相同模型的
`clock + last full word` 仍改善 0.00093 bits/track，95% CI [0.00058, 0.00127]，41/50
states 为正。这说明更早的完整前缀有一个统计稳定但极小的 residual，修正了粗粒度审计中
“完全没有全局前缀增量”的结论，但不能称为强全局语法。

精确字符串本身不可建词表：75,722 个健康 query 中有 72,022 个不同的 50-symbol
words，95.1% 唯一，最高频 word 只出现 27 次。因此实现采用 factorized emission，比较
每个 flow/track 的相似性，而不是把整串硬编码成 one-hot ID。

字面 motif 的物理检验给出更直接的否定证据：

| motif | 成功 episode 出现率 | stasis q-3 recall | matched AUC |
|---|---:|---:|---:|
| 任意 flow 窗口 `Fl Fl Sy Sy` | 100.0% | 100.0% | 0.439 |
| 固定末端 `f6,f7=Fl; f8,f9=Sy` | 96.3% | 99.0% | 0.499 |
| 末端同 flow `Fl & Sy` | 5.4% | 1.0% | 0.500 |

它们要么是成功轨迹中普遍存在的正常去噪形态，要么没有区分度。完整前缀 innovation 在
4.7% success-episode FPR 下只有 4.1% q-3 recall，matched AUC 0.450。专门检测
“过度规律”的低-surprisal 通道最好，但也只有 8.1% q-3 recall（95% CI
[5.1%, 12.8%]）和 0.550 AUC。因此严格十流实验的判决是：

$$
\boxed{\text{弱局部构词法通过；微弱全局前缀 residual 通过；Trap 在线早检 No-Go。}}
$$

完整结果见 `results-flow-word-v1/`。

## 物理早检

以下阈值统一读取 pooled out-of-fold Scene8 success controls 的 4.7% episode FPR。
它是公平 operating point，不是用来声称可部署的阈值。

| 分数 | q-3 recall | onset recall | onset+3 recall | state/query matched AUC |
|---|---:|---:|---:|---:|
| current innovation | 4.1% | 5.1% | 5.1% | 0.592 |
| VAR innovation | 3.0% | 3.6% | 3.6% | 0.578 |
| HMM prefix innovation | 2.0% | 3.0% | 3.0% | 0.580 |
| persistent innovation | 5.6% | 7.6% | 8.6% | 0.537 |
| over-regularity | 4.1% | 4.1% | 4.1% | 0.446 |
| return failure | 4.1% | 4.1% | 4.6% | 0.522 |
| three-channel union | 5.1% | 5.1% | 5.6% | 0.533 |

over-regularity 的三个组成量也没有隐藏的强信号：

| 组成量 | q-3 recall | onset recall | matched AUC |
|---|---:|---:|---:|
| freeze | 4.1% | 4.6% | 0.497 |
| lag-2..4 recurrence | 5.1% | 5.1% | 0.479 |
| low surprisal | 7.1% | 7.1% | 0.420 |

健康 excursion 的三步 return 任务有 104,399 个 held-out 前缀。return-failure AUC 为
0.544，而只看 deviation 的 AUC 为 0.535，增量约 0.009。它不足以支持 persistence
observer。

Scene8 calibration states 上设定的 5% 阈值部署到新 states 后，实际 FPR 为
5.1% 到 12.8%；full40 阈值迁移到 Scene8 时 over-regularity 类通道漂移更大。
阈值跨状态和跨任务都不稳定。

## 候选重排

主分析包含 8 条失败 trunks、28 个同状态 K=4 snapshots 和 112 个候选。只有 4 个
snapshots、3 条独立 trunks 同时含成功与失败候选。

| 选择器 | 成功率 | 相对随机 |
|---|---:|---:|
| random expectation | 5.4% | 0.0% |
| action medoid | 7.1% | +1.8% |
| full-prefix healthy score | 10.7% | +5.4% |
| oracle | 14.3% | +8.9% |

full-prefix 的 trunk-cluster bootstrap uplift 区间为 [-1.9%, 13.3%]。同 snapshot
成功/失败配对 AUC 为 0.786，但 full-prefix、clock-only、last-query-only 和 no-prefix
四个版本的 AUC 全部是 0.786。只有 3 条 informative trunks，单边精确 sign-flip
$p=0.125$。

所以这里可提出的假设是：

$$
\boxed{\text{健康 current-query emission 可能帮助候选重排。}}
$$

不能提出的结论是：

$$
\boxed{\text{全局 routing 句法已经帮助候选重排。}}
$$

## Go / No-Go

| 条件 | 判定 | 证据 |
|---|---|---|
| 历史对下一 query 有新增信息 | 弱通过 | 十流 word-prefix 比 last full word 高 0.00093 bits/track，但效应极小 |
| 历史能区分 transient 与 persistent | 不通过 | return AUC 0.544，仅比 deviation 高约 0.009 |
| innovation 与 over-regularity 能覆盖物理异常 | 不通过 | 固定 FPR 下 early recall 接近误报率，组合不优于 current query |
| prefix 对候选 outcome 有增量 | 不通过 | 四种 prefix ablation 的配对 AUC 完全相同 |

当前总判定：

$$
\boxed{\text{MoE grammar 可作为机制分析；作为在线 Trap detector 暂时 No-Go。}}
$$

## 下一轮最有信息量的实验

1. 不继续扩大 Transformer 或词表。当前瓶颈是目标可观测性，不是模型容量。
2. 收集带精确 onset 的 loop、freeze、recovery，并以独立 init-state/trunk 为统计单位。
3. 候选采集应提高 opportunity snapshot 比例；至少获得数十条同时含成功/失败候选的
   独立 trunks，再预注册比较 full-prefix 与 no-prefix 的配对差值。
4. 若 full-prefix 仍无增量，应将方法准确命名为 routing-emission candidate selector，
   不再称为 grammar regularization。
5. 若物理 Trap 在 MoE-only 下继续不可分，应加入 dense hidden state 或行为反馈做传感器，
   而不是将 observer miss 直接解释成未知 Trap。

完整连续结果见 `results-three-channel-v2/`，候选逐项结果见
`results-candidate-reranking-v2/`。
