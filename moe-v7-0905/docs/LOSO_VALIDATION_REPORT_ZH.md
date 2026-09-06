# v7 内在保护器的 Leave-One-Suite-Out 任务无关性验证

日期：2026-09-05
协议：`docs/superpowers/specs/2026-09-05-loso-task-agnostic-validation-design.md`

## 0. 为什么要做这个

v7 的 sealed manifest 里 `runtime_task_identity: false` 是成立的——运行时确实只吃当前
`hb_router_probs`。但核对缓存后发现：

```
reference corpus (main + extra) : 40 tasks / 16,000 traj
external_8b                     : 39 tasks / 15,600 traj
external \ reference            : 空集
```

39 个外部任务**全部**落在 40 个校准任务里。因此 v7 报告的 77.84% recall 是**在见过的任务上**
取得的，从未在未见任务、更未在未见 suite 上测过。本次实验补上这一格。

## 1. 结论

留出整个 suite 后，v7 的 macro precision/recall 为 **88.04% / 62.93%**。相对全语料校准，
recall 下降 **7.60 个百分点**，precision 基本持平（+0.75 pp）。其中 operating-point 泄漏
（L2 − L1）只占 recall **+0.64 pp**、precision +0.88 pp——**退化的主体是阈值数值随语料配方
漂移，不是 operating point 过拟合了 suite 组合**。

预注册判读（spec §3.4）：门限为 macro precision < 70% 则不得宣称任务无关。实测 88.04%，
**通过**。v7 可以在 suite 层面声称任务无关，但必须同时报告 recall 的 7.60 pp 损失，以及下面
第 4 节的长度偏置机制。

## 2. 三级对照

`published` = v7 已发布的全语料 profile（含全部泄漏）；`loso_l1` = 分位数水平锁死、只重算四个
常量的数值；`loso_l2` = 额外在校准 suite 的 development outcomes 上重跑 640 点选择。

### 2.1 macro（4 折等权）

| 级别 | risk recall | early-4 recall | precision | timely FPR |
|---|---:|---:|---:|---:|
| published | 70.54% | 49.05% | 87.30% | 0.538% |
| loso_l1 | 62.30% | 43.00% | 87.16% | 0.839% |
| **loso_l2** | **62.93%** | **45.25%** | **88.04%** | **0.772%** |

### 2.2 micro（episode 汇总）

| 级别 | TP / FP | risk recall | precision | timely FPR |
|---|---:|---:|---:|---:|
| published | 439 / 80 | 77.84% | 84.59% | 0.532% |
| loso_l1 | 413 / 125 | 73.23% | 76.77% | 0.831% |
| **loso_l2** | **414 / 115** | **73.40%** | **78.26%** | **0.765%** |

`published` 的 micro 行正是 v7 报告的 439/80、77.84%、84.59%、0.532%，逐位一致，说明本实验的
计分路径与 v7 evaluator 等价。

**macro 与 micro 分歧很大，必须同时看。** macro precision 几乎不掉（+0.75 pp），micro precision
掉 6.33 pp。原因是四折中三折的 FP 绝对数只有 1/3/2 个，等权平均把它们和 `libero_long` 的
109 个 FP 拉平了。只报 macro 会掩盖真实的误报集中。

### 2.3 逐折

| 留出 suite | 级别 | TP / FP | recall | early-4 | precision | FPR | 中位提前量 |
|---|---|---:|---:|---:|---:|---:|---:|
| libero_goal | published | 73 / 1 | 68.87% | 57.55% | 98.65% | 0.026% | 5 |
| libero_goal | loso_l1 | 61 / 1 | 57.55% | 40.57% | 98.39% | 0.026% | 5 |
| libero_goal | loso_l2 | 63 / 1 | 59.43% | 43.40% | 98.44% | 0.026% | 5 |
| libero_long | published | 236 / 69 | 86.13% | 84.31% | 77.38% | 1.852% | 19 |
| libero_long | loso_l1 | 247 / 119 | 90.15% | 89.05% | 67.49% | 3.194% | 20 |
| libero_long | loso_l2 | 252 / 109 | 91.97% | 90.51% | 69.81% | 2.925% | 19 |
| libero_object | published | 22 / 7 | 50.00% | 38.64% | 75.86% | 0.197% | 6 |
| libero_object | loso_l1 | 17 / 3 | 38.64% | 29.55% | 85.00% | 0.084% | 5 |
| libero_object | loso_l2 | 19 / 3 | 43.18% | 36.36% | 86.36% | 0.084% | 6 |
| libero_spatial | published | 108 / 3 | 77.14% | 15.71% | 97.30% | 0.078% | 2 |
| libero_spatial | loso_l1 | 88 / 2 | 62.86% | 12.86% | 97.78% | 0.052% | 2 |
| libero_spatial | loso_l2 | 80 / 2 | 57.14% | 10.71% | 97.56% | 0.052% | 1 |

四折全部 feasible，无需记录 infeasible。

### 2.4 eligible 分母

freeze 分支最早可能在 q6 报警、confirmed turbulence 最早 q10，短 horizon 的 episode 在物理上
无法产生（或只能产生 freeze 类）报警。分母必须显式给出，否则 recall 无法解释：

| 留出 suite | episodes | risk n | timely n | 可 freeze 报警 | 可 turbulence 报警 |
|---|---:|---:|---:|---:|---:|
| libero_goal | 4,000 | 106 | 3,894 | 4,000 | 1,672 |
| libero_long | 4,000 | 274 | 3,726 | 4,000 | 4,000 |
| libero_object | 3,600 | 44 | 3,556 | 3,600 | 3,600 |
| libero_spatial | 4,000 | 140 | 3,860 | 4,000 | 2,070 |

`libero_goal` 只有 41.8% 的 episode 长到能触发湍流分支，`libero_spatial` 只有 51.8%。

**但这个分母只约束误报机会，不约束风险检出。** 每个 suite 有固定的 horizon 上限
（goal 30、long 52、object 28、spatial 22 个 chunk），而**所有风险 episode 都跑满上限**——
风险的定义就是"没能在上限前结束"。提前结束的全是及时成功（中位 9-24 个 chunk）。因此风险
episode 一律长到足以触发两个分支，上表 41.8% / 51.8% 的缺口全部来自早早成功的 episode。

**`libero_spatial` 的 early-4 低是真的报得晚，不是窗口窄。** 它的风险 episode 有 22 个
chunk，最早可报在 chunk 6，只要首报 ≤ chunk 17 就能拿到 lead ≥ 4，空间充裕。实测首报中位数
是 chunk 19，报警相位中位 90.5%：

| suite | horizon | 风险 episode 首报 chunk 中位 | 报警相位中位 | 中位提前量 | lead ≥ 4 占比 |
|---|---:|---:|---:|---:|---:|
| libero_long | 52 | 32 | 62.7% | 19 | 98% |
| libero_object | 28 | 21 | 77.8% | 6 | 77% |
| libero_goal | 30 | 24 | 82.8% | 5 | 84% |
| libero_spatial | 22 | 19 | 90.5% | 2 | 20% |

`libero_long` 的高 early-4 同样有两重来源：它的报警相位最早（62.7%），horizon 又最长（52），
所以同样的相位换算成绝对 chunk 数就是 19。early-4 是绝对 chunk 门槛，跨 suite 比较时必须同时
看相位和 horizon，不能只看比例。

> **更正记录**：本节初版称 spatial 的 early-4 低是"median length 11、窗口极窄，与检测能力
> 无关"。那是用全体 episode 的长度中位数（10.5，被及时成功占满）代替了风险 episode 的长度
> （恒为 22），结论错误。v7 主报告 `REPORT_ZH.md` 中"短 horizon 的 libero_spatial 受 q6 最早
> 报警限制"一句有同样的问题。

## 3. 阈值漂移

`loso_l1` 级（分位数水平不变，只换语料切片）的四个常量：

| 留出 suite | freeze | acceleration | periodicity | periodicity_scale |
|---|---:|---:|---:|---:|
| libero_goal | +12.22% | +10.65% | +12.27% | +3.71% |
| libero_long | **−19.29%** | **−27.21%** | −0.23% | +1.83% |
| libero_object | **+25.17%** | +14.27% | −8.60% | −5.18% |
| libero_spatial | +6.99% | +17.30% | −4.58% | +0.60% |

摆动幅度：freeze 阈值 44.5 pp（−19.29% 到 +25.17%），acceleration 阈值 44.5 pp，
periodicity 阈值 20.9 pp，periodicity_scale 只有 8.9 pp。

**这里修正设计阶段的一个判断。** 设计时我把 `periodicity_scale` 单独点名，因为它是唯一被焊进
分数内部（而非只做阈值）的语料常数。实测它是四个常量里**最稳的**（±5%）。真正不稳的是 freeze
与 acceleration 两个阈值切点（±20-27%），而 freeze 是主分支——v7 自己的消融里 freeze 单独就有
343 TP。实验 B 应当优先重构这两个切点，而不是优先换掉 `periodicity_scale`。

## 4. 长度偏置被直接证实

设计文档 §1.2 预测：pooled trajectory-peak 分位数有长度偏置，轨迹越长 peak 越有机会冲高，
所以阈值主要由语料的 suite 配方决定。各 suite 的 median rollout length 为
goal 10、spatial 11、object 14、long 24-25。

实测方向完全吻合：

| 留出 suite | 校准语料构成 | freeze 阈值 | 留出 suite recall | 留出 suite FPR |
|---|---|---:|---|---|
| libero_long | 纯短 horizon | **−19.29%** | 86.13% → 91.97%（↑） | 1.852% → 2.925%（↑） |
| libero_goal | 含 1/3 long | +12.22% | 68.87% → 59.43%（↓） | 0.026% → 0.026% |
| libero_spatial | 含 1/3 long | +6.99% | 77.14% → 57.14%（↓） | 0.078% → 0.052%（↓） |
| libero_object | 含 1/3 long | +25.17% | 50.00% → 43.18%（↓） | 0.197% → 0.084%（↓） |

把 long 拿出校准集，语料变成纯短 horizon，peak 分布整体下移，阈值跟着下降 19-27%，于是报警更
容易触发——recall 升到 91.97%，但 FP 从 69 涨到 109，FPR 从 1.852% 涨到 2.925%。反过来，留出任一
短 suite 时语料里 long 占比从 1/4 升到 1/3，peak 分布上移，阈值抬高 7-25%，报警更难触发，
recall 全线下降。

**这不是 MoE 路由的性质，是语料配方的性质。** 一个号称"只依赖内在路由"的判据，它的判定线却由
"校准集里有多少条长轨迹"决定——这正是设计文档要检验的 L1 层缺陷，现在有了直接证据。

## 5. F-long：最极端的一折

按 spec §7 的预注册，这一折不得因难看而剔除。它是唯一 recall 上升而 precision 下降的折：
90.15%/91.97% 的 recall 配 67.49%/69.81% 的 precision，FPR 2.9-3.2%，远超其他三折。
109 个 FP 占 loso_l2 全部 115 个 FP 的 94.8%。

它同时也是**唯一校准语料与评估对象 horizon 量级完全不同**的一折，因此是长度偏置最纯粹的读数。
如果只报另外三折的平均，会得到一个漂亮但无意义的 precision。

## 6. 与 v7 主报告的关系

v7 报告的 439 TP / 80 FP / 77.84% / 84.59% 没有算错，但它回答的是
**"在见过的 40 个任务上，运行时不查任务身份能做到多少"**，不是
**"遇到未见任务能做到多少"**。本报告回答后者：留出整个 suite 后，micro 是
414 TP / 115 FP / 73.40% / 78.26%。

两个数字都应保留，且必须标注各自的条件。

## 7. 限制

- **external 不会因 LOSO 变成 pristine holdout。** 它在 v3-v6 中被反复查看过。LOSO 消除的是
  "留出 suite 的任务身份进入了校准"这一条泄漏，消除不了"研究者已经看过这批数据"。
- 只有 4 折且 suite 数就是 4，macro 平均的方差很大，不是稳定的泛化估计。
- `libero_object` 只有 44 个 risk，其 recall 的 95% CI 为 23.3%-66.0%，点估计不应单独引用。
  `libero_goal` 的 106 个 risk 也只给出 34.4%-91.8% 的区间。区间宽是真实的，未做窄化。
- 40 个 task 全部来自同一个 7B checkpoint，跨模型/跨本体漂移不在本实验范围。
- 本实验只回答"未见 suite"。"未见单任务、同 suite"由配套的 40 折 LOTO 实验覆盖，见
  `LOTO_VALIDATION_REPORT_ZH.md`：那里 pooled recall 78.72%、precision 84.57%，与全语料实质
  无差别，阈值漂移 std 只有本实验的 1/6 到 1/10。**伤害 v7 的不是"没见过这个任务"，而是
  "校准语料的构成变了"。**
- `loso_l2` 的 operating point 由校准 suite 的 development outcomes 选出，这仍是"规则选择用了
  标签"，只是标签不再来自留出 suite。阈值**数值**在三个级别下都是无标签 order statistic。

## 8. 对实验 B 的结论

> **本节的建议已被后续实验证伪，以 `CALIBRATION_VARIANTS_REPORT_ZH.md` 为准。**
> 前缀内秩统计量这条路会把 FPR 抬高约两个数量级：pooled trajectory-peak 分位数同时承担了
> 逐轨迹多重检验校正、绝对尺度参照、非存活条件比较基准三件工作，每条去语料路径都恰好打断
> 其中一件。下面保留原始判断以存档。

**需要做，且方向应该调整。**

判据：freeze 与 acceleration 两个阈值切点在四折间摆动 44.5 pp，且摆动方向由校准语料的 horizon
配方单调决定（§4）。只要判定线还是"pooled trajectory-peak 的某个分位数"，换一批任务就会换一条线，
"任务无关"就只能做到 L0（运行时不读任务身份），做不到 L1（判定线不依赖语料配方）。

B 的目标是把这两个切点换成**前缀内秩统计量**：可交换性下，当前分数是前缀 running max 的概率为
`1/q`，连续 K 次即给出分布无关、无语料、无任务的误报上界。这条路径同时消掉长度偏置——秩统计量
对轨迹长度是自适应的，不像 pooled peak 分位数那样被长轨迹拉高。

优先级修正（见 §3）：先重构 freeze 与 acceleration 的切点，`periodicity_scale` 可以留到后面，
它的漂移只有 ±5%，不是当前瓶颈。

## 9. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python experiments/evaluate_loso_suite.py
pytest -q tests
```

评估耗时约 3 秒，纯 CPU，无 GPU，不重跑 rollout，不重算原始特征。

关键产物：

- `results/loso_validation/loso_metrics.csv`：三级 × 四折全部指标与 cluster bootstrap CI
- `results/loso_validation/threshold_drift.csv`：四常量 × 四折漂移，含指标后果列
- `results/loso_validation/fold_selection.json`：每折 640 点选择审计与隔离计数
- `results/loso_validation/fold_profiles.npz`：四折 × 两级的常量
- `results/loso_validation/loso_manifest.json`：封存顺序声明与 SHA-256
- `results/loso_validation/loso_summary.json`：macro/micro 汇总

回归锚点（`tests/test_loso_suite.py`）：全语料校准复现 `global_profile.npz` 的四个常量；已发布
常量复现 `sealed_first_alarms.npz` 的五个外部报警数组。两条都过，因此本报告的退化确实来自校准
切片，不是重实现差异。
