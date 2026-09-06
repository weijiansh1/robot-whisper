# v7 内在保护器在 legacy 16x32 上的实测

`moe-v7-legacy16x32-0906` · 2026-09-06 · 纯 CPU

## 0. 结论

**16x32 的数字从外推变成了实测，而且 external 锚点在计算任何 16x32 数字之前就已复现。**

冻结 profile 原样应用，未重新标定任何东西：

| | TP | FP | 精度 | 召回 | timely FPR |
|---|---:|---:|---:|---:|---:|
| **guard（freeze OR turbulence）** | **257** | **25** | **0.9113** | **0.8371** | 0.01110 |
| freeze 分支 | 253 | 17 | 0.9370 | 0.8241 | 0.00755 |
| turbulence 分支 | 18 | 9 | 0.6667 | 0.0586 | 0.00400 |

turbulence 相对 freeze 单独的边际贡献是 **+4 TP / +8 FP**——在这个 cohort 上净负。

**修正后的全语料合计：32,960 episodes、1,358 risks → 1,078 TP / 172 FP，精度 0.8624，召回 0.7938，timely FPR 0.005443。**

## 1. 锚点（在算 16x32 之前全绿）

- 从原始 Zarr 抽取的 `route_acceleration` 与 `lag_periodicity` 对已发布缓存**逐位相同**（最大绝对误差 0.0），覆盖 external_8b 全部 15,600 episodes / 248,255 queries 与 development_main 全部 14,800 / 221,781。
- 由这些特征重放的 guard **逐元素复现** `sealed_first_alarms.npz` 的全部五个数组，两个 cohort 都是：**439/80** 与 **382/67**。
- 48 条 episode 逐 query 过 `IntrinsicGuardMonitor`：首次报警元组与封存一致，也与批处理路径逐位一致。
- 从 16,000 条参考语料重算的冻结操作点与已发布 `global_profile.npz` 逐位相同。
- **关于那个精度陷阱**：`np.quantile` 对 float32 输入返回 float32，而 `intrinsic_score_arrays` 又用 `np.float32(scale)` 重铸，因此 float32 与 float64 路径给出相同值——**这次没有咬到**。float64 重算与断言仍保留在代码里。
- 敏感性：换成完全原始的 mobility（monitor 内部双重归一化的 `hellinger`，与缓存约定相差约 2e-6）会翻转 **1 条 development_main episode**（67→68 FP），external 与 16x32 **零翻转**。主结果用封存约定；16x32 不在浮点刀刃上。

## 2. 分 suite

| suite | episodes | risks | 风险率 | TP | FP | 精度 | 召回 | timely FPR | 中位提前量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| libero_goal | 1,024 | 42 | 4.10% | 41 | 0 | 1.0000 | 0.9762 | 0.00000 | 5 |
| libero_spatial | 1,024 | 49 | 4.79% | 41 | 3 | 0.9318 | 0.8367 | 0.00308 | 2 |
| libero_long | 512 | 216 | **42.19%** | 175 | 22 | 0.8883 | 0.8102 | 0.07432 | 19 |

suite 内部各任务的风险率差异显著：goal 0% vs 8.2%；spatial 2.34% vs 7.23%。

## 3. 生存条件校正：guard 没有被任务身份灌水

| 分组 | 报警数 | 精度 | suite 先验 | suite lift | task 先验 | **task lift** |
|---|---:|---:|---:|---:|---:|---:|
| 全体 | 282 | 0.9113 | 0.6269 | **1.4537** | 0.6267 | **1.4543** |
| libero_goal | 41 | 1.0000 | 0.9984 | 1.0016 | 0.9984 | 1.0016 |
| libero_long | 197 | 0.8883 | 0.4750 | **1.8703** | 0.4750 | **1.8703** |
| libero_spatial | 44 | 0.9318 | 0.9610 | 0.9696 | 0.9594 | 0.9713 |

**suite 匹配与 task 匹配的 lift 几乎完全相等**——guard 的报警不是任务选择器。

为证明这个 cohort **确实有灌水空间**，另跑了一个纯任务身份对照（在每个 suite 的高风险任务上于 chunk 0 直接报警，完全不读路由）：**suite 匹配 lift 1.7363（goal 2.0000、spatial 1.5102），task 匹配 lift 精确等于 1.0000。** 另有 `control_all_running_at_chunk{0,4,8,12}` 在两种基准下 lift 都精确等于 1.000000（代码中断言），验证先验机制本身正确。

**早期带（先验 < 0.25）：0 次报警。** 原因可算：libero_long 的先验从 chunk 0 起就 ≥0.25（基础风险率 42.19%），所以那里结构上不可能有早期报警；goal 在 chunk 19、spatial 在 chunk 13 越过 0.25，而 guard 最早的报警分别在 chunk 20 与 16。**这个 0 里有多少是构成、多少是报得晚，无法分离。**

## 4. 剩余干预余量

`budget_left = cap − 1 − chunk`；上限已验证：goal 30、long 52、spatial 22，且每条失败的长度精确等于上限。

| B | 0 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| 全体 TP/FP | 257/25 | 244/25 | 210/23 | 160/21 | 158/21 |
| goal | 41/0 | 40/0 | 35/0 | 1/0 | **0/0** |
| long | 175/22 | 175/22 | 170/22 | 159/21 | 158/21 |
| spatial | 41/3 | 29/3 | 5/1 | **0/0** | 0/0 |

**B=12 时只有 libero_long 存活**；短 horizon 的 suite 没有可用的干预窗口。

## 5. Episode 内信息量（新增硬性要求）

| 通道 | episode 内恒定的条数 | 每 episode 最少不同值 | 不同值/query 中位 | 组内方差占比 |
|---|---:|---:|---:|---:|
| route_acceleration | **0** / 2560 | 9 | 1.000 | 0.103 |
| lag_periodicity | **0** / 2560 | 7 | 1.000 | 0.823 |
| layer_mobility L2…L15 | **0** / 2560 | 8 | 1.000 | 0.18–0.37 |
| score: freeze | 0 / 2560 | 3 | 1.000 | 0.492 |
| score: acceleration_persistent | 690 / 2339 | 1 | 0.395 | 0.135 |
| score: periodicity_persistent | 475 / 2098 | 1 | 0.625 | 0.142 |
| **对照: episode_length** | **2560 / 2560** | 1 | 0.077 | **0.000** |

三个原始输入在每条 episode 的**每一个 query 上都取不同值**——没有逐位恒定的通道。persistent 分数在 22–30% 的 episode 上是平的，这来自 `min over k confirmations` 算子（k=8、k=4），不是通道退化。

## 6. 长度检测器：标签定义的复述，不是基线

307 TP / 2 FP，召回 **1.000（结构性的，不是挣来的）**，精度 0.9935。在 `outcome_metrics.csv` 中标记 `is_baseline = False`，从所有排名中排除；其匹配生存 lift 精确等于 1.000。那 2 个 FP 是恰好跑到上限的成功（1 条 long、1 条 spatial）。

## 7. 构成差异 vs 性能退化

16x32 的 5 个任务在另外两个 cohort 中都存在，因此可以做同任务比较：

| cohort（共享 5 任务） | episodes | risks | TP | FP | 召回 | timely FPR | 精度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| development_main | 2,000 | 250 | 208 | 17 | 0.8320 | 0.00971 | 0.9244 |
| external_8b | 2,000 | 255 | 207 | 15 | 0.8118 | 0.00860 | 0.9324 |
| legacy_16x32 | 2,560 | 307 | 257 | 25 | 0.8371 | 0.01110 | 0.9113 |
| *按 dev 逐任务率的期望* | 2,560 | 307 | 252.2 | 19.2 | 0.8215 | 0.00853 | — |
| *按 ext 逐任务率的期望* | 2,560 | 307 | 249.6 | 18.0 | 0.8129 | 0.00799 | — |

**可分辨的**：表面上 2.1 倍的 timely-FPR 差距主要是**构成**——16x32 有 20% 是 libero_long（FPR 0.065 dev / 0.056 ext / 0.074 此处），且没有 libero_object。按任务对齐后差距收缩到 1.30–1.39 倍（观测 25 vs 期望 18.0–19.2，约 1.3–1.7σ 泊松）。召回略**高于**期望。

**不可分辨的**：残余的 6–7 个 FP 来自采样设计（16 初态 × 32 噪声 vs 50 × 8，初态多样性更低、组内相关更强）还是 run 级漂移。5 个任务 3 个 suite 下，任务聚类 bootstrap 太宽（guard 召回 CI [0.810, 0.969]）。

**迁移是好的、不是退化**——精度 0.911 与召回 0.837 都超过已发布的两个 cohort。但信息集中在 libero_long（lift 1.87）；goal 是 1.0016、spatial 是 0.97（**低于生存基线**）。这与 external 自身的模式一致（goal 1.29 / spatial 1.00 / long 1.80），所以是 **v7 的性质，不是 16x32 的缺陷**。

## 8. 必须带进任何引用的限制

1. dev/ext 的 `original_failure` = 晚成功 ∪ 持久失败；16x32 只有原始 horizon 结局。**风险定义相同**，但 16x32 无法拆分晚成功/持久失败。
2. 先验在被评分的样本内估计，这对生存基线有利（对 lift 而言是保守的）。
3. 16x32 的逐任务数字（尤其 spatial/ramekin 只有 12 个风险）不应单独引用。
4. **没有任何东西在 16x32 上标定过**：四个 profile 标量、三个分位水平、布尔机制、基线窗口与确认长度全部来自冻结的 v7 bundle。

## 9. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-legacy16x32-0906
python experiments/extract_cohort_features.py
python experiments/verify_external_anchor.py      # 不通过则中止
python experiments/evaluate_legacy16x32_guard.py
pytest -q tests
```

全部三个 cohort（32,960 episodes / 521,344 queries）从原始 Zarr 抽取约 20 秒（20 进程，纯 CPU）。`results/manifest.json` 记录 9 个输入、7 个代码文件、20 个输出的 sha256。`tests/test_legacy16x32.py` 10 个测试全部通过。
