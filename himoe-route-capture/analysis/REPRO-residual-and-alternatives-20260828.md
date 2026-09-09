# 复现审计：「共识核心之外的残差失败」与「替代路由组织方式」两条实验线

复现日期：2026-08-28（原产出 2026-08-28 00:00–02:00）
执行者：独立复现 agent。所有复现输出写入 `<原目录>-repro`。
**一处例外已披露：`--self-test` 扫描意外重写了 `analysis/residual-failure-dynamics/`
（内容零变化，已恢复时间戳），详见 §1.1。** 其余原产物全部未动。

---

## 0. 一句话结论

**逐数字全部对上（12 条声称值 12 条复现，其中 11 个脚本的 `report.md` 与原文件逐字节相同）；
但「残差失败切不出稳定 MoE 子类型」这个负结果的*统计功效很低*，原报告用来支撑它的头号统计量
（可行视图两两 ARI 中位数 = 0.018）在人工注入真实、强分离的两簇结构时也只有 0.06–0.12，
因此该数字本身几乎不能区分「没有结构」与「有结构但看不见」。**
换用 `max_excluding_aligned` 统计量后负结果仍成立，但只对
「路由质心分离 ≳ 0.9–1.1 个组内合并标准差」（≈ 成功-vs-停滞核心整体路由差距的 40–45%）
以上的亚型有效；比这更弱的亚型在 n=89 下**不可检出**，不能被排除。

---

## 1. 执行的命令与耗时

环境：`OMP_NUM_THREADS=8`，与另外 4 个 agent 共享 240 核（观测 load average 24–33）。
磁盘全程保持 34 GB 可用；复现产物合计 < 80 MB。

### 1.1 自检

```
python3 <script>.py --self-test
```

| 脚本 | `--self-test` | 备注 |
|---|---|---|
| `analyze_residual_failure_subtypes.py` | passed | |
| `analyze_residual_failure_dynamics.py` | **不存在** | **该脚本完全没有 argparse**，既无 `--self-test` 也无 `--out-dir`；`OUT_DIR` 硬编码 |
| `analyze_residual_failure_physical_audit.py` | passed | |
| `audit_peer_rank_hdbscan_c1.py` | passed | |
| `analyze_routing_state_grammar.py` | passed | |
| `analyze_aligned_route_kernel_truncate90.py` | passed | |
| `audit_alternative_route_truncation.py` | **不存在** | 无 `--self-test` 参数（有 `--out-dir`） |
| `analyze_failure_routing_clusters.py` | passed | |
| `analyze_within_task_unsupervised_loops.py` | passed | |
| `analyze_unsupervised_replanning_loops.py` | passed | |

> 任务书称「都有 `--self-test`」，实测 10 个脚本里有 2 个没有。
> `analyze_residual_failure_dynamics.py` 因为没有 `--out-dir`，直接运行会**覆盖原产物**；
> 后续正式复现时用 `python3 -c "import analyze_residual_failure_dynamics as m; m.OUT_DIR=...; m.main()"` 绕开。

> ### ⚠️ 事故披露：`analysis/residual-failure-dynamics/` 被本次自检意外重写（内容未变）
>
> `analyze_residual_failure_dynamics.py` **完全不解析 `sys.argv`**，所以上面那轮
> `--self-test` 扫描（06:07）实际把整个分析跑了一遍，并写回硬编码的
> `analysis/residual-failure-dynamics/`。随后为排查 `--help` 无输出又跑了一次（06:08）。
> 这违反了「不要覆盖已有输出」的纪律，特此披露。
>
> **影响评估：内容零变化，只有 mtime 从 `01:14` 变成 `06:07`。** 证据：
> 1. 事故发生**之前**（06:06）已经 `ls -la` 记录过五个文件的字节数，
>    事故后逐一比对完全一致：`assignments.csv` 586186、`long_task_dynamics.png` 346702、
>    `report.md` 8031、`summary.json` 59418、`vote_axis_and_physics.png` 236355。
> 2. 事故发生**之前**已完整读过 `report.md`；其全部数字（213/307、94=62+32、
>    exact-vote axis 1.437/1.054/0.969/0.617、Spearman −0.729/−0.699/+0.735、
>    物理表 2.197/2.467/1.628/1.576、投票模式 42/9/8、zero-vote C1/C2/C3 = 6/15/9）
>    与现文件逐条相同。
> 3. 06:28 用打过 OUT_DIR 补丁的独立目录再跑一次，五个文件与现产物 **md5 全部相同**
>    （`c5f8724da3` / `7833fe7e9e` / `a75334ed44` / `c1974b545f` / `82323a0ed3`），
>    说明该脚本在固定输入下完全确定，重写不可能改变内容。
>
> 已用 `touch -d '2026-08-28 01:14:00' analysis/residual-failure-dynamics/*` 恢复原时间戳；
> 内容无需恢复。其余 9 个脚本的 `--self-test` 均正常（`audit_alternative_route_truncation.py`
> 因 argparse 报「unrecognized arguments」而**提前退出，未执行任何分析**，未产生写入）。

### 1.2 正式复现命令

| # | 命令 | 耗时 | 结果 |
|---|---|---:|---|
| 1 | `analyze_residual_failure_subtypes.py --subsamples 50 --permutations 5000 --seed 20260829 --out-dir analysis/residual-failure-subtypes-repro` | 30 s | `report.md` / `summary.json` / `assignments.csv` / `post_pot2_features.npz` **逐字节相同** |
| 1b | 同上，`--seed 11 / 20250101 / 987654321` | 71 / 61 / 48 s | 见 §3 |
| 6a | `analyze_aligned_route_kernel_truncate90.py --out-dir analysis/aligned-route-kernel-truncate90-repro` | 981 s | `report.md` 仅 `Run class` 行不同（`formal_sensitivity`→`nonformal`，因 out-dir 非默认），**所有数字相同** |
| 6b | `audit_alternative_route_truncation.py --primary-results analysis/alternative-routing-organizations --out-dir analysis/alternative-routing-organizations-repro --permutations 2000 --seed 20260829` | 584 s | **1 行漂移**（peer_rank Louvain，见 §2.6） |
| 5 | `analyze_routing_state_grammar.py --out-dir analysis/routing-state-grammar-repro --subsamples 50 --permutations 5000 --seed 20260828` | 491 s | `report.md` **逐字节相同** |
| 4 | `audit_peer_rank_hdbscan_c1.py --out-dir .../peer-rank-c1-audit-repro --permutations 50000 --seed 20260828` | 71 s | `report.md` **逐字节相同** |
| 7 | `analyze_failure_routing_clusters.py --subsamples 200 --association-permutations 5000 --seed 20260827 --out-dir analysis/failure-routing-clusters-repro` | 770 s | `report.md` **逐字节相同** |
| 2 | `analyze_residual_failure_dynamics.py`（OUT_DIR 打补丁 → `-repro`） | 32 s | `report.md` 与 `summary.json` **逐字节相同** |
| 3 | `analyze_residual_failure_physical_audit.py --out-dir analysis/residual-failure-physical-audit-repro --permutations 5000 --seed 20260828` | 7 s | `report.md` **逐字节相同** |
| 8a | `analyze_within_task_unsupervised_loops.py --out-dir analysis/within-task-unsupervised-loops-repro --permutations 5000 --bootstraps 5000 --seed 20260827` | 18 s | `report.md` **逐字节相同** |
| 8b | `analyze_unsupervised_replanning_loops.py --out-dir analysis/unsupervised-replanning-loops-repro --permutations 5000 --bootstraps 5000 --seed 20260827` | 36 s | `report.md` **逐字节相同** |
| 8c | 8a 改用 8b 的 `seed_pairs.csv` / `episode_scores.csv` 作输入（链式复现） | 7 s | `report.md` **逐字节相同** |
| 额外 | `repro_residual_subtype_power.py`（本次新写，见 §3） | 75 s + 109 s | 种子扫描 + 阳性对照 |

**全部 8 项任务均已执行，无遗漏。** 优先级顺序 1 > 6 > 5 > 4 > 7 > 2,3 > 8 全部跑完。

### 1.3 输入完整性

复现期间有其他 agent 在并行重跑 `analyze_aligned_route_kernel.py` / `analyze_route_change_events.py` /
`analyze_all_outcome_routing_clusters.py` 等上游脚本。复现开始与结束时对 9 个上游输入做了 md5 快照，
**全部未变**（`aligned-route-kernel/assignments.csv` `96151e78e59f`、
`route-change-events/event_features.npz` `9b31a0f3a431`、
`alternative-routing-organizations/representation_features.npz` `2c280130628d`、
`feature_cache/manifest.json` `28768575ca1c` 等，时间戳仍为 2026-08-27 22:49 ~ 2026-08-28 00:30）。

> **重要范围限定**：脚本 1（`analyze_residual_failure_subtypes.py`）**不重新推导共识核心**。
> 它直接读 `aligned-route-kernel/`、`route-change-events/`、`alternative-routing-organizations/`
> 三个已冻结的 `assignments.csv` 与特征 `npz`。因此本次是「给定上游冻结产物的条件复现」，
> 不是从 raw zarr 到结论的端到端复现。

---

## 2. 逐条数字对照

### 2.1 residual-failure-subtypes

| 项 | 声称 | 实测 | 差异 | 判定 |
|---|---|---|---|---|
| 共识核心覆盖 | 213 / 307 | 213 / 307 | 0 | ✅ |
| 残差组成 | 94 = 62（1 票）+ 32（0 票） | 94 = 62 + 32 | 0 | ✅ |
| 残差集中度 | 89/94 来自 long moka-pot，全部正好 52 query | 89/94；脚本硬断言 `set(episode_length)=={52}` 通过 | 0 | ✅ |
| 89 条内票数 | 1 票 59 / 0 票 30 | 59 / 30 | 0 | ✅ |
| 其余 5 条 | 三个任务 1 / 1 / 3 | top_drawer 1、ramekin 1、stove 3（0 票 2 + 1 票 1） | 0 | ✅ |
| 九种表示 HDBSCAN | 3 种全为 noise | `peer_rank` / `lag_spectrum` / `path_signature` clusters=0 | 0 | ✅（但见 §3.3，这 3 个表示对*真实*强对比也全 noise） |
| 可行视图两两 ARI 中位数 | 0.018 | 0.018366 | +0.0004 | ✅ |
| 除去同源 aligned 变体后最高 | 0.163 | 0.163466 | +0.0005 | ✅ |
| 删末 10% ARI（event） | 0.073 | 0.073019 | 0 | ✅ |
| 删末 10% ARI（aligned-raw） | 0.057 | 0.056854 | 0 | ✅ |
| 删末 10% ARI（task-residual） | −0.004 | −0.003960 | 0 | ✅ |

补充核对（原报告正文数字，非任务书列表）：

| 项 | 声称 | 实测 | 判定 |
|---|---|---|---|
| aligned_raw 与终端侧别 adjusted MI | 0.600 | 0.600 | ✅ |
| aligned_task_init adjusted MI | 0.515 | 0.515 | ✅ |
| event C1 = 12 条无 nonlocal return，删末 10% 后变 22 条、仅留 4 条 | Jaccard 0.133 | 0.13333 | ✅ |
| event 小簇 80% 子采样 ARI median/P10 | 1.000 / 0.289 | 1.000 / 0.2889 | ✅（**P10 对种子敏感，见 §3.1**） |
| 0/1 票与 init-state NMI / 置换 p | 0.103 / p=0.0148 | 0.1028 / 0.014797 | ✅ |
| post-pot2 recurrence full/tr90 ARI | 0.404 | 0.404 | ✅（**对种子敏感，见 §3.1**） |
| post-pot2 geometry 终端侧别 AUC full | 0.923 | 0.923 | ✅ |

### 2.2 residual-failure-dynamics / physical-audit

| 项 | 声称 | 实测 | 判定 |
|---|---|---|---|
| 共识核心 | 213 / 307 | 213 / 307 | ✅ |
| 残差 94，89 来自 long | 94 / 89 | 94 / 89 | ✅ |
| 长任务残差中 pot2 到位而 pot1 未到位 | 81 / 89 | 28（0 票）+ 53（1 票）= **81 / 89** | ✅ |
| 核心中对应 | 123 / 127 | 23（2 票）+ 100（3 票）= **123 / 127** | ✅ |
| 0 票路由仍活跃 / 晚期重启 | 定性 | late/early speed **1.437**、peak phase **0.827**、late stasis **0.067**、terminal return **0.533** | ✅ |
| 3 票最大变化在早期随后强烈减速 | 定性 | late/early **0.617**、peak phase **0.547**、late stasis **0.880** | ✅ |
| 单调趋势 | — | Spearman ρ：late/early **−0.729**、peak phase **−0.699**、late stasis **+0.735**（truncate90：−0.805 / −0.764 / +0.740） | ✅ |

`analysis/residual-failure-dynamics-repro/{report.md,summary.json}` 与原产物**逐字节相同**；
`analysis/residual-failure-physical-audit-repro/report.md` 同样**逐字节相同**。

### 2.3 routing-state-grammar

| 项 | 声称 | 实测 | 判定 |
|---|---|---|---|
| primary states | 8 | 8 | ✅ |
| state counts | 1722/7708/4538/2578/1346/4341/2658/709 | 完全一致 | ✅ |
| seed-refit ARI median/min | 0.997 / 0.996 | 0.996765 / 0.996332 | ✅ |
| grammar 维度 | 187 | 187（`completion.json` 亦为 187） | ✅ |

### 2.4 aligned-route-kernel-truncate90

| 项 | 声称 | 实测 | 判定 |
|---|---|---|---|
| full raw K6 vs truncate-90 raw K6 ARI | 0.950 | 0.949923 | ✅ |
| reference episodes/failures | 252 / 246 | 252 / 246 | ✅ |
| candidate（truncate90 matched C1） | 249 / 219 | 249 / 219 | ✅ |
| episode precision / recall / Jaccard | 0.851 / 0.841 / 0.734 | 0.851406 / 0.841270 / 0.733564 | ✅ |
| failure-only precision / recall / Jaccard | 0.945 / 0.841 / 0.802 | 0.945205 / 0.841463 / 0.802326 | ✅ |

### 2.5 within-task-unsupervised-loops

| task | 声称 K / silhouette / restart ARI / status | 实测 | 判定 |
|---|---|---|---|
| goal/middle_drawer | 2 / 0.211 / 0.584 / low_stability | 完全一致 | ✅ |
| goal/top_drawer | 3 / 0.273 / 0.993 / ok | 完全一致 | ✅ |
| long/SCENE8 | 2 / 0.253 / 0.993 / ok | 完全一致 | ✅ |
| spatial/ramekin | 2 / 0.208 / 0.588 / low_stability | 完全一致 | ✅ |
| spatial/stove | 2 / 0.305 / 1.000 / ok | 完全一致 | ✅ |

上游 `analyze_unsupervised_replanning_loops.py` 与链式重跑（8c）同样逐字节相同。

### 2.6 唯一未完全复现的一行（peer_rank Louvain）

`audit_alternative_route_truncation.py` 的 `truncate90_report.md`：

| | 声称（原） | 实测（repro） |
|---|---|---|
| peer_rank / louvain K | 9 | **8** |
| outcome excess | 0.013 | 0.014 |
| task NMI | 0.079 | 0.070 |
| length NMI | 0.136 | 0.124 |
| ARI to full | 0.266 | **0.249** |

其余 9 行（含全部 HDBSCAN 行、lag_spectrum、path_signature、route_topology、layer_wave）完全一致。
命令使用了默认 `--seed 20260829`，因此这是 **networkx Louvain 在同种子下的非确定性**（社区数 9↔8）。
该行不在任务书的声称清单内，且原报告的定性解读（peer-rank 保留小密度岛、效应量 < 0.05）不受影响，
但**说明 Louvain 相关的数字不应按 3 位小数引用**。

---

## 3. 额外必做项

### 3.1 种子稳健性：换 3 个种子重跑脚本 1

`--seed 20260829 / 11 / 20250101 / 987654321`（out-dir 各自独立）：

| run | 两两 ARI 中位数 | 最大 | 去 aligned 同源后最大 | 全 noise 视图数 | 删末10% event/aligned/task ARI | init NMI | event 子采样 P10 | post-pot2 recurrence ARI |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| orig 20260829 | 0.0184 | 0.5756 | 0.1635 | 3 | 0.073 / 0.057 / −0.004 | 0.1028 | 0.289 | 0.404 |
| repro 20260829 | 0.0184 | 0.5756 | 0.1635 | 3 | 0.073 / 0.057 / −0.004 | 0.1028 | 0.289 | 0.404 |
| seed 11 | 0.0184 | 0.5756 | 0.1635 | 3 | 0.073 / 0.057 / −0.004 | 0.1028 | 0.269 | 0.503 |
| seed 20250101 | 0.0184 | 0.5756 | 0.1635 | 3 | 0.073 / 0.057 / −0.004 | 0.1028 | **0.000** | 0.423 |
| seed 987654321 | 0.0184 | 0.5756 | 0.1635 | 3 | 0.073 / 0.057 / −0.004 | 0.1028 | 0.285 | 0.343 |

**结论 A：`0.018` 对种子完全不敏感——但这是因为整条聚类管线本身对种子近乎确定性**
（`RobustScaler` 无随机性；`PCA` 在 n=89 下走 full SVD；`HDBSCAN` 确定性）。
所以「换种子仍是 0.018」**不构成**对该负结果的稳健性证据，它只是复述了同一次计算。

**结论 B：真正带随机性的两个附属数字确实会动**：
- event 小簇的 80% 子采样 ARI **P10 = 0.289 → 0.000**（seed 20250101）。原报告写「full window 内看似稳定」时引用的 P10 不可按面值使用。
- post-pot2 recurrence 的 full/truncate90 ARI **0.343–0.503**（原文引用 0.404）。定性结论（都很低）不变，数值不可按 3 位小数引用。

**为补足「种子」的空洞，另做了两类真正的扰动**（`repro_residual_subtype_power.py`）：

*(a) 队列扰动（80% 子采样，30 次）*

| 统计量 | 观测值(n=89) | 子采样中位数 | q10 | q90 |
|---|---:|---:|---:|---:|
| 两两 ARI 中位数 | 0.0184 | 0.0356 | 0.0103 | 0.0719 |
| 去 aligned 后最大 | 0.1635 | 0.1469 | — | 0.2564 |
| 全 noise 视图数 | 3 | 4 | — | — |

*(b) HDBSCAN 超参网格（`min_cluster_size` × `min_samples`）*

| mcs | ms | 全 noise 视图数 | 两两 ARI 中位数 | 去 aligned 后最大 |
|---:|---:|---:|---:|---:|
| 5 | 3 | **0** | 0.0536 | 0.2840 |
| 5 | 4 | 2 | 0.0274 | 0.2328 |
| 5 | 6 | 4 | 0.0086 | 0.1875 |
| 8 | 3 | 2 | 0.0414 | 0.2045 |
| **8** | **4** | **3** | **0.0184** | **0.1635**（论文设置） |
| 8 | 6 | 5 | −0.0040 | 0.0596 |
| 12 | 3 | 5 | 0.0068 | 0.0245 |
| 12 | 4 | 5 | 0.0062 | 0.0287 |
| 12 | 6 | 5 | −0.0040 | 0.0584 |
| 15 | 3 | 6 | −0.0074 | −0.0074 |
| 15 | 4 | 6 | −0.0009 | −0.0009 |
| 15 | 6 | 7 | 0.3669 | n/a（可行视图不足） |

→ 「九种表示中 **3 种**全为 noise」这个数在合理超参范围内是 **0 到 5**。该计数不是数据性质，是超参选择。

### 3.2 合成阳性对照（不用高斯噪声）

设计：对 89 条残差失败的**真实**路由特征做锚点插值
`x'_i = (1−α)·x_i + α·anchor_{g(i)}`，其中
`anchor_0` = 同任务 296 条**成功** rollout 的特征中位数，
`anchor_1` = 同任务 127 条**共识核心失败**的特征中位数。
两个锚点都是实测路由特征，插值结果仍在观测流形的凸包内；不加任何噪声。
随机划分 89 条为两组（平衡 45/44 与不平衡 20/69 两种），再跑**完全相同**的
`RobustScaler → PCA(90% 方差) → HDBSCAN(8,4)` 管线，看能否找回植入标签。
`sep` = 植入两簇质心距 ÷ 组内合并 RMS 半径（在聚类器自己的嵌入里测）。

平衡 45/44：

| α | sep(aligned_raw) | sep(event) | ARI aligned_raw | ARI event | ARI lag_spec | ARI route_topo | ARI state_gram | ARI peer_rank | ARI path_sig | ARI layer_wave | 跨视图 ARI 中位数 | 跨视图去-aligned 最大 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.21 | 0.19 | −0.005 | −0.005 | 0.000 | −0.004 | −0.006 | 0.000 | 0.000 | −0.002 | 0.0184 | 0.1635 |
| 0.20 | 0.37 | 0.52 | −0.008 | −0.002 | 0.000 | 0.029 | −0.002 | 0.000 | 0.000 | −0.002 | 0.0345 | 0.1248 |
| 0.30 | 0.55 | 0.68 | −0.006 | −0.004 | 0.066 | 0.213 | 0.018 | 0.000 | 0.000 | −0.001 | 0.0301 | 0.1587 |
| 0.35 | 0.65 | 0.83 | −0.010 | 0.003 | 0.252 | 0.242 | 0.040 | 0.000 | 0.000 | 0.001 | 0.0275 | 0.1768 |
| 0.40 | 0.75 | 0.94 | 0.022 | −0.002 | **0.527** | 0.349 | 0.025 | 0.000 | 0.000 | 0.000 | 0.0256 | 0.2129 |
| 0.45 | 0.89 | 1.03 | 0.204 | −0.001 | **0.796** | 0.381 | 0.073 | 0.000 | 0.000 | 0.000 | 0.0276 | **0.3223** |
| 0.50 | 1.04 | 1.15 | 0.271 | −0.001 | **0.923** | 0.490 | 0.134 | 0.000 | 0.000 | 0.010 | 0.0295 | **0.5367** |
| 0.60 | 1.31 | 1.38 | **0.637** | **0.742** | **1.000** | **0.616** | 0.243 | 0.000 | 0.000 | 0.025 | 0.0634 | **0.7455** |

不平衡 20/69：

| α | ARI aligned_raw | ARI event | ARI lag_spec | ARI route_topo | ARI state_gram | 跨视图 ARI 中位数 | 跨视图去-aligned 最大 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | −0.002 | −0.043 | 0.000 | 0.076 | −0.031 | 0.0200 | 0.1092 |
| 0.35 | 0.014 | −0.007 | 0.000 | 0.341 | −0.030 | 0.0218 | 0.0969 |
| 0.40 | 0.078 | 0.027 | **0.710** | **0.672** | −0.028 | 0.0278 | **0.5429** |
| 0.45 | 0.142 | 0.016 | **0.933** | **0.771** | −0.034 | 0.0251 | **0.7669** |
| 0.60 | **0.976** | 0.351 | **1.000** | **0.933** | **0.559** | 0.1172 | **0.9843** |

### 3.3 真实阳性对照（不含任何合成成分）

用语料中**最大的真实路由对比**做同 n 对照：随机取同任务 45 条真实成功 + 44 条真实共识核心失败
（n=89，同任务、同管线、10 次重复）：

| 表示 | ARI 中位数 | min | max | 簇数中位数 |
|---|---:|---:|---:|---:|
| lag_spectrum | **0.730** | 0.646 | 0.955 | 2 |
| aligned_raw | **0.553** | 0.000 | 0.654 | 2 |
| state_grammar | **0.529** | 0.390 | 0.806 | 3 |
| event | **0.435** | 0.166 | 0.709 | 2 |
| route_topology | 0.336 | 0.000 | 0.654 | 2 |
| peer_rank | **0.030** | 0.000 | 0.102 | 2 |
| aligned_task_init | **0.000** | 0.000 | 0.020 | 0 |
| path_signature | **0.000** | 0.000 | 0.000 | 0 |
| layer_wave | **0.000** | 0.000 | 0.125 | 0 |

**跨视图两两 ARI（在真实两组真值下）：中位数 0.236，最大 0.576。**

这三条结论直接改变对负结果的解读：

1. **头号统计量几乎没有动态范围。** 观测 0.018 → 真实强两组真值下也只有 **0.236**；
   人工植入 α=0.60（4/9 视图 ARI 0.62–1.00）时仍只有 **0.063–0.117**，
   而观测值的 80% 子采样区间是 **[0.010, 0.072]**——**与阳性条件下的取值重叠**。
   ⇒ 「两两 ARI 中位数 = 0.018」**不能**作为「无亚型」的证据。中位数被那些结构上看不见任何东西的
   视图对拖死了：论文设置下可行视图 6 个 = 15 对，而 α=0.60 时只有 4 个视图找回植入结构，
   即最多 6 / 15 对能同时看见它——中位数按定义落在看不见的那一半。
2. **`max_excluding_aligned` 才是有效统计量。** 观测 0.1635（子采样 q90 = 0.256）；
   真实两组真值下 0.576；植入 α≥0.45（平衡）或 α≥0.40（不平衡）时 0.32–0.98。
   ⇒ 用这个量，负结果**成立**，但只覆盖 `sep ≳ 0.9–1.1`（≈ 成功-vs-核心整体路由差距的 40–45%）以上的亚型。
3. **「3 种表示全为 noise」不是关于残差队列的证据。**
   `path_signature`、`layer_wave`、`aligned_task_init` 在**真实成功-vs-核心**对比上也是 0 簇 / ARI 0.000；
   `peer_rank` 是 ARI 0.030。它们在 n=89 下**根本没有检出能力**。
   原报告「`peer_rank` … 全部被 HDBSCAN 判为 noise。这是没有稳定、去初态离散亚型的直接证据」
   这句话**不成立**，应删除或改写为「peer_rank 在 n=89 下无检出功效，不提供信息」。

### 3.4 `StratifiedGroupKFold` 用了什么 group / 什么 stratify

`analyze_residual_failure_subtypes.py:312-315`

```python
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in splitter.split(values, side, groups=initial_state):
```

- `groups = initial_state` = `aligned["init_state_id"]`（**就是 init state**）
- `stratify (y) = side` = `endpoint_side`（终端 EEF 更近 pot1 → 1，更近 pot2 → 0），即预测目标本身
- 唯一使用处：`post_pot2_relative_phase.*.endpoint_side_separability`（报告里的 AUC 0.923/0.936/0.812 等）

**→ group 就是 init state，任务书列出的风险条件（group 不是 init state 或 episode）不成立。**
但仍有 4 个应当报告的问题：

1. **只有 13 个 group、极不均衡**（init 10 有 23 条，init 13/29/46/49 各 1 条）。
   实际折叠：fold0 的测试集 = **单个 init state（init 10，n=23，占 26%）**；
   fold1 = {29,39,46}；fold2 = {0,42}；fold3 = {3,7,49}；fold4 = {13,20,23,26}。
   5 折 group CV 在 13 个组上极其粗糙，AUC 的有效自由度远小于 n=89 暗示的量级。
2. **AUC 是把 5 折的 out-of-fold 概率拼起来一次性算的**（`roc_auc_score(side, probability)`），
   而每折的 `RobustScaler`+`PCA`+`LogisticRegression` 是独立拟合的，概率尺度不可比。
   折内类别也不平衡（fold1 = [11,4]）。这个 pooled OOF AUC 会有难以定向的偏差。
3. **标签本身带 init 信息**：NMI(endpoint_side, init_state) = 0.127，
   side 在 init 内的纯度 = 0.775。按 init 分组会**低估**（保守）而不是高估，方向上安全。
4. **最实质的问题不在 CV 而在窗口**：`full` / `truncate90` 窗口的特征取自 pot2 里程碑到**最终终止**
   的整段路由，而标签是**最终终止时**的 EEF 侧别。AUC 0.923 有相当部分是「用终点读终点」。
   原报告已在正文标注 `prefix60` 不是严格在线窗口，也说明 AUC 只能当线索——这个 caveat 是必要且正确的。

### 3.5 `post_pot2_features.npz` 能否复用

**可以，而且不需要担心磁盘。**

- 任务书说该文件 5.3 GB，**实测 5,315,943 字节 ≈ 5.3 MB**（差 1000 倍）。
- 内容：`episode(89)`、`post_pot2_start_query(89)`、`endpoint_side(89)`，
  以及 `feature_{geometry,change,recurrence}_{full,truncate90,prefix60}`
  （形状 89×3200 / 89×2880 / 89×1080，float32）与对应 `label_*`（int16）。
  即 §「从 pot-2 完成后重新对齐」整节所需的全部特征都在里面，可直接跳过 zarr 重算。
- 但**没有必要**：整脚本端到端只要 **30 秒**（zarr 读取约 520 MB，89 episode × 52 query）。
- 复现产出的 npz 与原文件**所有数组逐元素相同**（`np.array_equal` 全 True）。

---

## 4. 判定汇总

| 类别 | 条数 | 结果 |
|---|---:|---|
| 任务书列出的声称数字 | 12 | **12 条全部对上**（最大偏差 < 0.001，均为四舍五入） |
| 补充核对的报告正文数字 | 7 | 7 条全部对上 |
| 逐字节相同的 `report.md` | 11 / 12 | 例外：truncate90 只差 `Run class` 一行（out-dir 非默认所致） |
| 未完全复现 | 1 行 | `audit_alternative_route_truncation.py` 的 peer_rank Louvain（K 9→8，ARI-to-full 0.266→0.249），networkx Louvain 非确定性 |
| 脚本接口与任务书描述不符 | 2 | `analyze_residual_failure_dynamics.py` 无 argparse（无 `--self-test` / `--out-dir`，直接跑会覆盖原产物）；`audit_alternative_route_truncation.py` 无 `--self-test` |
| 任务书事实性错误 | 1 | `post_pot2_features.npz` 是 5.3 **MB** 不是 5.3 GB |
| 纪律事故 | 1 | `--self-test` 扫描意外重写 `analysis/residual-failure-dynamics/`（5 个文件，字节数与内容均未变，仅 mtime 01:14→06:07）；详见 §1.1 披露框 |

---

## 5. 关于负结果的最终判断

**「残差失败切不出稳定 MoE 子类型」这个负结果：数值上完全复现，方向上仍然成立，但原文用来支撑它的主要
统计量功效不足，结论的适用范围必须收窄。**

具体地：

- ✅ **复现**：所有数字 bit-level 一致，与种子无关。
- ⚠️ **主统计量无效**：「可行视图两两 ARI 中位数 0.018」在真实两组真值下也只有 0.236，
  在强植入结构下只有 0.06–0.12，而观测值的子采样 q90 已经到 0.072。
  该统计量的信噪比不足以支持「没有结构」的推断，应当从结论句中撤下。
- ✅ **替代统计量支持负结果**：`max_excluding_aligned` = 0.1635（子采样 q90 0.256）
  显著低于真实两组真值的 0.576 与植入 α≥0.45 的 0.32–0.98。
- 📏 **功效边界**：本管线在 n=89、这些维度（54–648 维）下，
  只能检出**质心分离 ≳ 0.9–1.1 个组内合并 SD**（≈ 成功-vs-停滞核心整体路由差距的 40–45%）的两簇结构；
  更弱的亚型不可检出，**不能被排除**。
- ❌ **应删除的一句**：「`peer_rank` 全部被判为 noise 是没有去初态离散亚型的直接证据」——
  peer_rank（以及 path_signature、layer_wave、aligned_task_init）对语料中最强的真实路由对比也检不出，
  这是表示/管线的零功效，不是数据的性质。
- ⚠️ **「3 种表示全为 noise」是超参产物**：`min_cluster_size` 从 5 到 15 时该计数在 0–7 之间变化。
- ✅ **正结果侧全部稳固**：共识核心 213/307、truncate90 持久性（ARI 0.950、249/219、0.851/0.841/0.734）、
  state grammar（8 状态、187 维、seed ARI 0.997/0.996）、peer-rank C1 审计、within-task loops 五任务表
  ——全部逐字节复现。

### 建议的最小改写

原报告第 10 行：

> 可行视图两两 ARI 中位数为 `0.018`。除去同源的两个 aligned 变体，最高也只有 `0.163`。

建议改为：

> 除去同源的两个 aligned 变体后，跨视图最高 ARI 只有 `0.163`（80% 子采样 q90 = `0.256`）；
> 作为对照，在同任务、同 n=89、同管线下用真实的「成功 vs 共识核心」两组真值，该量为 `0.576`。
> 两两 ARI 的**中位数**（`0.018`）不作为证据引用：在真实两组真值下它也仅为 `0.236`。
> 本管线的检出下限约为质心分离 `0.9–1.1` 个组内 SD；更弱的亚型无法排除。

---

## 6. 产物清单

复现输出（全部为新建 `-repro` 目录，未覆盖任何原文件）：

- `analysis/residual-failure-subtypes-repro/`（含 `seed_and_power_audit.json` = §3 全部原始数字）
- `analysis/residual-failure-subtypes-repro-seed{11,20250101,987654321}/`
- `analysis/residual-failure-dynamics-repro/`
- `analysis/residual-failure-physical-audit-repro/`
- `analysis/alternative-routing-organizations/peer-rank-c1-audit-repro/`
- `analysis/routing-state-grammar-repro/`
- `analysis/aligned-route-kernel-truncate90-repro/`
- `analysis/alternative-routing-organizations-repro/`（仅 `truncate90_*` 四个文件）
- `analysis/failure-routing-clusters-repro/`
- `analysis/within-task-unsupervised-loops-repro/`、`...-repro-chained/`
- `analysis/unsupervised-replanning-loops-repro/`
- `repro_logs/*.log`（每条命令的 stdout + 退出码 + 耗时）
- `repro_residual_subtype_power.py`（本次新写的种子扫描 / 稳健性 / 阳性对照脚本，带 `--self-test`）
