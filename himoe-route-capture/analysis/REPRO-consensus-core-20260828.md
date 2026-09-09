# 复现审计：无监督 MoE 路由聚类「三方法共识核心」结论

审计日期 2026-08-28。原实验产出时间 2026-08-27 夜 ~ 2026-08-28 02:00。

审计范围：`analysis/routing-organization-synthesis/report.md` 的「共同失败核心」章节，
以及其四个上游脚本的全部被引用数字。所有复现输出写入 `<原目录>-repro[-seed<S>]`，
**未覆盖任何已有目录**。

**一句话结论：13 项声称数字 100% 逐位复现，零不一致；但「≥2 票 = 217、precision 0.982」
中的 `217` 是一个种子实例（换种子落到 204–217），三支柱之一的 lag-spectrum Louvain C7
（n=219）根本不是种子稳定的对象（39 个扰动种子中 0 个能复出单一社区 Jaccard ≥ 0.9）。
共识核心作为一个「集合」是稳健的（跨种子 Jaccard 0.927–0.991，precision 恒在 0.975–0.982），
作为一个「精确计数」不是。**

---

## 1. 环境与数据

- `python3` numpy 2.5.1 / scipy 1.17.1 / sklearn 1.8.0 / pandas 3.0.1 / zarr / networkx；无 `hdbscan` 包（脚本用 `sklearn.cluster.HDBSCAN`），符合预期。
- 实际 cache root 为 `/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA`（**不是**任务书里写的 `/home/jovyan/work/VLA_MUI_HUB/...`，后者不存在）。5 个 formal run 的 `routes.zarr` 合计约 2.0 GB（压缩后），非 76 GB；`_input_fingerprints()` 会对其做全量 sha256，耗时可接受。
- 复用了 `analysis/all-outcome-routing-clusters/feature_cache`（592 MB）与 `matched_prefix_cache`（12 MB，手动预拷贝到 repro 目录以避免从 raw zarr 重抽）。未重读原始数据做特征。
- 全部 repro 产物合计约 150 MB，磁盘余量未受影响（34 GB 全程不变）。
- 并行度：每进程 `OMP_NUM_THREADS=4` + 脚本内 `threadpool_limits(limits=4)`，同时最多 6 个进程（≈24 线程），符合 ~32 上限。

### 自检

四个脚本 `--self-test` 全部通过：

| 脚本 | self-test 耗时 |
|---|---:|
| `analyze_all_outcome_routing_clusters.py` | 57 s |
| `analyze_alternative_routing_organizations.py` | 52 s |
| `analyze_aligned_route_kernel.py` | 5 s |
| `analyze_route_change_events.py` | 1 s |

---

## 2. 实际执行的命令与耗时

工作目录 `/home/jovyan/work/himoe-vla/himoe-route-capture`；下文 `$A = analysis`。

### 2.1 原参数复现（formal）

```bash
python3 analyze_all_outcome_routing_clusters.py \
  --out-dir $A/all-outcome-routing-clusters-repro \
  --feature-cache $A/all-outcome-routing-clusters/feature_cache \
  --subsamples 50 --seed 20260828                                   # 1973 s, run_class=formal

python3 analyze_alternative_routing_organizations.py \
  --feature-cache $A/all-outcome-routing-clusters/feature_cache \
  --out-dir $A/alternative-routing-organizations-repro \
  --permutations 2000 --seed 20260828                               #  583 s

python3 analyze_aligned_route_kernel.py \
  --source-dir $A/all-outcome-routing-clusters \
  --out-dir $A/aligned-route-kernel-repro \
  --subsamples 50 --permutations 5000 --seed 20260829               # 1435 s, run_class=formal_exploratory

python3 analyze_route_change_events.py \
  --cache-dir $A/all-outcome-routing-clusters/feature_cache \
  --prior-output $A/all-outcome-routing-clusters/embeddings_and_labels.npz \
  --out-dir $A/route-change-events-repro --seed 20260827            #  137 s
```

四个 formal 运行并发执行，墙钟约 33 min。

> 说明：`analyze_aligned_route_kernel.py` 的 `formal` 门槛硬编码
> `args.source_dir.resolve() == SOURCE_DIR.resolve()`，因此 `--source-dir` 必须指向**原始**
> `all-outcome-routing-clusters`（只读，不写入）。这样做同时保证了输入与原运行逐字节一致。

### 2.2 换种子（每个方法 × 3 个新种子 20260830 / 12345 / 999）

同样命令，仅改 `--seed` 与 `--out-dir <...>-repro-seed<S>`。耗时：

| 方法 | seed 20260830 | seed 12345 | seed 999 |
|---|---:|---:|---:|
| `route_change_events` | 153 s | 152 s | 154 s |
| `alternative_routing_organizations` | 570 s | 570 s | 572 s |
| `aligned_route_kernel` | 1117 s | 1116 s | 1113 s |
| `all_outcome_routing_clusters` | 1448 s | 1448 s | 1446 s |

所有换种子运行 `run_class` 自动降级为 `nonformal`（`aligned_route_kernel` / `all_outcome_routing_clusters`
的 formal 门槛把原始 seed 写死了）。这是脚本设计使然，不是参数降级：`--subsamples 50`、
`--permutations 5000` / `2000`、`--association-permutations` 默认值全部保持原样，**没有任何降级**。

### 2.3 纯 Louvain 种子扫描（额外增量）

为了把「Louvain 种子」与「PCA 种子」两个随机源拆开，我固定原运行保存的
`embedding_lag_spectrum`（`alternative-routing-organizations/embeddings_and_labels.npz`），
按脚本原样重建 kNN 图（`LOUVAIN_NEIGHBORS=25`，高斯权 + max 去重，2560 节点 / 42582 边），
只扫 `nx.community.louvain_communities(seed=...)`。

**校验**：在脚本真实使用的种子 `args.seed + 1000*index = 20260828 + 1000 = 20261828` 下，
replay 精确复出原始 C7（n=219，fail=215，Jaccard 1.000），确认图重建无误。
（首次 replay 我误用了 `args.seed` 本身而对不上——脚本对第 `index` 个表示用 `args.seed + 1000*index`。）

结果落在 `analysis/_repro_logs/louvain_seed_sweep.csv`。

---

## 3. 逐条数字对照表

失败标签总数 F = 307（2560 episodes，2253 成功 / 307 失败），与原报告一致。

「块」的指认方式：本审计不按标签下标取块，而按**内容**取块（该划分中失败计数最大的块），
以保证跨种子可比。原种子下该规则唯一地取到 raw C1 / event C0 / lag C7。

| # | 项 | 声称值 | 实测值 | 差异 | 判定 |
|---:|---|---|---|---|:---:|
| 1 | shift-aligned landmark kernel, raw C1 | n=252, 6 succ / 246 fail | n=252, 6 / 246 | 0 | ✅ |
| 2 | raw C1 precision | 0.976 | 0.97619 | 0 | ✅ |
| 3 | raw C1 failure recall | 0.801 | 0.80130 | 0 | ✅ |
| 4 | route change-event, C0 | n=214, 5 / 209 | n=214, 5 / 209 | 0 | ✅ |
| 5 | C0 precision / recall | 0.977 / 0.681 | 0.97664 / 0.68078 | 0 | ✅ |
| 6 | lag-spectrum kNN graph, Louvain C7 | n=219, 4 / 215 | n=219, 4 / 215 | 0 | ✅ |
| 7 | C7 precision / recall | 0.982 / 0.700 | 0.98174 / 0.70033 | 0 | ✅ |
| 8 | ≥2 票共识 | 217 = 213 fail + 4 succ | 217 = 213 + 4 | 0 | ✅ |
| 9 | ≥2 票 precision / failure recall | 0.982 / 0.694 | 0.98157 / 0.69381 | 0 | ✅ |
| 10 | 三方法交集 | 185 = 182 fail + 3 succ | 185 = 182 + 3（precision 0.98378） | 0 | ✅ |
| 11 | 三方法并集 | 283 = 275 fail + 8 succ | 283 = 275 + 8 | 0 | ✅ |
| 12 | 并集 precision / recall | 0.972 / 0.896 | 0.97173 / 0.89577 | 0 | ✅ |
| 13 | 并集遗漏 32 失败的来源 | 30 long moka-pot + 2 stove spatial | 30 `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` + 2 `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`（合计 32） | 0 | ✅ |
| 14 | 两两 Jaccard **配对归属** | Raw C1/Event C0 = 0.707；Raw C1/Lag C7 = 0.725；Event C0/Lag C7 = 0.827 | 0.70701 / 0.72532 / 0.82700 | 0 | ✅ **配对正确** |
| 15 | Aligned raw K6 80% 子采样稳定性 | ARI median / P10 = 0.991 / 0.977 | 0.99150 / 0.97749（`subsample_fraction=0.80`） | 0 | ✅ |
| 16 | Event K4 稳定性 | 0.862 / 0.828 | 0.862 / 0.828 | 0 | ✅ |
| 17 | route-change-events 分区 | status `stable_partition`, K=4, sizes [214, 1643, 205, 498] | 完全一致 | 0 | ✅ |
| 18 | route-change-events K=2 | silhouette 0.317, ARI 0.819/0.782, PAC 0.104 | 0.317 / 0.819 / 0.782 / 0.104 | 0 | ✅ |
| 19 | all-outcome 主视图 | 无任何 primary view 同时通过 frozen stability + conditional outcome gates | report.md 第 5 行逐字一致 | 0 | ✅ |
| 20 | all-outcome same-K 比较 | geometry ARI 1.000 (preserved)、recurrence 0.723 (moderately_changed)、expert_occupancy 0.000 (reorganized) | 1.000 / 0.723 / 0.000，class 标签一致 | 0 | ✅ |

**不一致项：0 项。** 全部 20 条（对应任务书 13 行表格）逐位复现。

### 3.1 逐位一致性（同种子重跑）

四个脚本在**新进程、不同日期**下重跑，产物 sha256：

| 脚本 | `assignments.csv` sha256 | `report.md` | 结论 |
|---|:---:|:---:|---|
| `analyze_all_outcome_routing_clusters.py` | 一致 | 一致 | 逐位复现（`summary.json` 仅 `/method` 路径字段不同：1 个叶子） |
| `analyze_alternative_routing_organizations.py` | 一致 | 一致 | 逐位复现 |
| `analyze_aligned_route_kernel.py` | 一致 | 一致 | 逐位复现 |
| `analyze_route_change_events.py` | 一致 | 一致 | 逐位复现 |

`all-outcome-routing-clusters-repro/completion.json` 的 11 个 `output_sha256` 中 10 个 MATCH，
唯一 DIFFER 的 `summary.json` 差异经逐叶子比对只有 `/method` 记录的绝对路径不同。
`run_class` 复现为 `formal`。

**→ 同种子重跑完全一致，未发现任何未固定的随机源。**

### 3.2 随机源清点

我逐一 grep 了四个脚本的随机来源，**全部被 `--seed` 固定**：

| 随机源 | 位置 | 是否固定 |
|---|---|:---:|
| `PCA(svd_solver="randomized", random_state=seed)` | `alt:294,502`；`aligned:148`；`events:416` | ✅ |
| `nx.community.louvain_communities(..., seed=seed)` | `alt:552` | ✅ |
| `kmeans_plusplus(..., random_state=seed)`（256 landmark 基） | `aligned:193` | ✅ |
| `silhouette_score(..., random_state=seed)` | `all_outcome:417` | ✅ |
| `np.random.default_rng(seed + K)` 子采样/置换/bootstrap | 各脚本多处 | ✅ |
| `np.random.default_rng(1)/(3)/(4)` 硬编码 | **仅在 `self_test()` 内**，不影响正式路径 | ✅ 无影响 |
| `KMeans(n_init=...)` | **未使用**。聚类为确定性 `scipy.linkage(method="ward")` + `fcluster`，以及确定性 `sklearn.cluster.HDBSCAN` | — |

**结论：不存在泄漏的随机源。下面观察到的种子敏感性是估计量本身的真实不稳定，不是 bug。**

---

## 4. 种子稳健性

### 4.1 各方法自身的划分 ARI（新种子 vs 原种子）

| 方法 | 划分 | seed 20260830 | seed 12345 | seed 999 |
|---|---|---:|---:|---:|
| aligned route kernel | raw K6 | 0.9921 | 0.9915 | 0.9931 |
| route change events | event K4 | **1.0000** | **1.0000** | **1.0000** |
| alternative | lag_spectrum **louvain** | **0.8530** | 0.9370 | 0.9398 |
| alternative | lag_spectrum hdbscan | 0.9291 | 0.9092 | **0.8193** |
| all-outcome | geometry | 1.0000 | 1.0000 | 1.0000 |
| all-outcome | change | 0.9765 | 0.9564 | 0.9486 |
| all-outcome | recurrence | 1.0000 | 1.0000 | 1.0000 |
| all-outcome | expert_occupancy | 0.9690 | 0.9943 | 1.0000 |

`route_change_events` 的 K=4 划分是**逐点完全相同**（exact label match = 1.000000），
K=2/3/4/5 的 sizes 与 `stable_partition` 判定在四个种子下完全一致；只有基于子采样的
ARI/PAC/silhouette 诊断值在小数第 3 位漂动。

### 4.2 被选中的核心块的漂移

「核心块」= 该划分中失败计数最大的块（规则 A）。对 lag 另给规则 B =「所有 failure-rate ≥ 0.9
的社区之并」（更贴近原报告实际挑 C7 的方式：一个大的、高纯度、跨任务的失败块）。

| 块 | 原种子 | 20260830 | 12345 | 999 | vs 原种子 Jaccard |
|---|---|---|---|---|---|
| **raw core (C1)** | 252 / p=0.976 / r=0.801 | 260 / 0.973 / 0.824 | 264 / **0.947** / 0.814 | 255 / 0.973 / 0.808 | 0.954 – 0.965 |
| **event core (C0)** | 214 / 0.977 / 0.681 | 214 / 0.977 / 0.681 | 同 | 同 | **1.000** |
| **lag core（规则 A）** | 219 / 0.982 / 0.700 | **109 / 1.000 / 0.355** | **109 / 1.000 / 0.355** | **109 / 1.000 / 0.355** | **0.498** |
| **lag core（规则 B）** | 219 / 0.982 / 0.700 | **109 / 1.000 / 0.355** | 214 / 0.981 / 0.684 | **109 / 1.000 / 0.355** | 0.498 / 0.977 / 0.498 |

**关键现象：原始 C7（n=219）在三个新种子下都被劈成两半。** 每个新种子都出现一个 n=109、
100% 失败、完全被原 C7 包含的社区；另一半（约 105–110 条）落进第二个社区，其纯度随种子波动：

| 新种子 | 第二半所在社区 | n | failure | failure rate | ∩ 原 C7 |
|---|---|---:|---:|---:|---:|
| 20260830 | C12 | 125 | 86 | 0.688 | 88 |
| 12345 | C11 | 105 | 101 | **0.962** | 105 |
| 999 | C10 | 140 | 101 | 0.721 | 105 |

只有 seed 12345 的第二半足够纯，能通过 ≥0.9 过滤而拼回 n=214（Jaccard 0.977 vs 原 C7）。

### 4.3 纯 Louvain 种子扫描（固定 embedding、固定图，39 个扰动种子）

| 指标 | 结果 |
|---|---|
| 社区数 | 14 – 17（原 16） |
| 单一社区对原 C7 的最佳 Jaccard | median **0.498**，range [0.410, 0.758] |
| **有多少种子能用单一社区复出 C7（Jaccard ≥ 0.9）** | **0 / 39（0%）** |
| failure-rate ≥ 0.9 社区之并：n | median 214，range **[0, 309]** |
| 该并集对原 C7 的 Jaccard | median 0.709，range [0.000, 1.000]，**仅 23% ≥ 0.9** |
| 该并集的 precision | 多数在 0.94–1.00；2 个种子（10、26）**完全没有** ≥0.9 纯度的社区，n=0 |

即：`Lag C7 = 219 / precision 0.982 / recall 0.700` 是「原种子恰好把失败核心切成一整块」
的结果。在 39 个扰动种子里，两个（seed 4、11）以两社区之并精确复出 n=219 / 0.982 / 0.700，
其余把它切成 107–109 + 剩余，或并成 284–322 的更大块。

同时注意：两个随机源都在起作用——固定 embedding 只扫 Louvain 种子时，
seed 20261830 / 13345 / 1999 给出的规则 B 并集为 214 / 214 / 309；而完整重跑（PCA 也换种子）
给出 109 / 214 / 109。所以 `PCA(svd_solver="randomized")` 的种子也会改变 lag 社区结构。

### 4.4 ≥2 票共识核心的种子间变动

| lag 块规则 | 原种子 | 20260830 | 12345 | 999 | 变动范围 |
|---|---|---|---|---|---|
| 规则 A（max-fail） | **217** / p **0.9816** / r 0.6938 | 205 / 0.9805 / 0.6547 | 205 / 0.9756 / 0.6515 | 204 / 0.9804 / 0.6515 | n ∈ [204, 217]，p ∈ [0.976, 0.982]，r ∈ [0.652, 0.694] |
| 规则 B（purity ≥ 0.9） | **217** / 0.9816 / 0.6938 | 205 / 0.9805 / 0.6547 | **217** / 0.9770 / 0.6906 | 204 / 0.9804 / 0.6515 | n ∈ [204, 217]，p ∈ [0.977, 0.982]，r ∈ [0.652, 0.694] |
| ≥2 票集合 vs 原集合 Jaccard | 1.000 | 0.9269 | 0.9358 / **0.9908**(规则B) | 0.9312 | 0.927 – 0.991 |

三方法交集 / 并集：

| 集合 | 原种子 | 20260830 | 12345 | 999 |
|---|---|---|---|---|
| 交集（规则 B） | 185 (182F+3S) | 101 | 183 | 97 |
| 并集（规则 B） | 283 (275F+8S), p 0.972, r 0.896 | 277, 0.971, 0.876 | 292, 0.949, 0.902 | 277, 0.971, 0.876 |

**交集是三个量里最脆的**：它必须同时包含不稳定的 lag 块，故 185 → 97–183。

### 4.5 去掉 lag 方法的两方法核心（额外做的稳健性对照）

`raw core ∩ event core`（都不依赖 Louvain）：

| | 原种子 | 20260830 | 12345 | 999 |
|---|---|---|---|---|
| n / precision / recall | 193 / 0.9793 / 0.6156 | 197 / 0.9797 / 0.6287 | 196 / 0.9745 / 0.6221 | 193 / 0.9793 / 0.6156 |
| Jaccard vs 原 | 1.000 | 0.960 | 0.985 | 0.969 |
| `raw ∪ event` n / p / r | 273 / 0.9744 / 0.8664 | 277 / 0.9711 / 0.8762 | 282 / 0.9504 / 0.8730 | 276 / 0.9710 / 0.8730 |

**两方法核心比三方法交集稳健得多**，且 precision 与三方法 ≥2 票几乎相同。

### 4.6 all-outcome 主结论的种子稳健性

| 声称 | 原种子 | 20260830 | 12345 | 999 |
|---|---|---|---|---|
| 「无 primary view 同时通过两道 gate」 | 是 | 是 | 是 | 是 |
| same-K geometry ARI / class | 1.000 / preserved | 1.000 / preserved | 1.000 / preserved | 1.000 / preserved |
| same-K recurrence ARI / class | 0.723 / moderately_changed | 0.723 | 0.723 | 0.723 |
| same-K expert_occupancy ARI / class | 0.000 / reorganized | 0.000 | 0.000 | 0.000 |
| geometry sizes | 1536/1024 | 1536/1024 | 1536/1024 | 1536/1024 |
| recurrence sizes | 101/874/231/525/268/561 | 同 | 同 | 同 |
| change sizes | 2295/265 | 2304/256 | 2294/266 | 2297/263 |
| expert_occupancy sizes | 125/512/470/472/478/503 | 512/147/470/472/496/463 | 512/134/470/472/478/494 | 125/512/470/472/478/503 |

**任务书列出的 all-outcome 三个 same-K 数字（1.000 / 0.723 / 0.000）在四个种子下完全不变。**
只有 `change` 与 `expert_occupancy` 的块大小有小幅漂动（ARI 0.949–0.994）。

---

## 5. 结论

### 5.1 「≥2 票 = 217、precision 0.982」是稳定量还是种子实例？

**`precision ≈ 0.98` 是稳定量；`n = 217` 是种子实例。**

- precision 在四个种子下为 0.9756 / 0.9770 / 0.9804 / 0.9805 / 0.9816——始终在 0.975–0.982，
  可以写成「precision ≈ 0.98」。
- `n = 217` 只在原种子（及 seed 12345 + 规则 B）出现；另两个种子给 204 / 205。
  应报告为 **n ≈ 205–217**。
- failure recall 0.694 同样是上端：跨种子为 0.652–0.694，应报告 **≈ 0.65–0.69**。
- ≥2 票集合本身跨种子 Jaccard 0.927–0.991，**作为集合是稳健的**。

### 5.2 「三种完全不同、outcome-blind 的方法独立抽出几乎同一组 rollout」是否成立？

**成立，但支柱强度不均，需要重新表述：**

| 支柱 | 种子稳健性 | 可否作为独立证据 |
|---|---|---|
| aligned route kernel raw C1 | 强（划分 ARI 0.992–0.993，核心 Jaccard 0.954–0.965，K=6 恒定） | ✅ |
| route change events C0 | **完美**（逐点相同，ARI 1.000） | ✅ |
| lag-spectrum Louvain C7 | **弱**（0/39 扰动种子能复出单一社区 Jaccard ≥ 0.9；n ∈ {0, 107…322}） | ❌ 只能作为定性支持 |

原综合报告已自设免责声明「Lag C7 是探索性图社区复现，不单独作为确认性证据」，
本审计结果**与该免责声明一致并进一步量化了它**。但同一份报告随后又把 C7 计入
「三方法至少两票 = 217 / 0.982」的头条数字，这一步是不自洽的：头条数字的第三票
来自一个不可重复的对象。

更严谨的表述应为：**两个稳健、独立、outcome-blind 的表示（shift-aligned landmark kernel 与
route change-event statistics）各自抽出约 250 / 214 条 rollout，其交集为 193–197 条
（precision ≈ 0.975–0.980，failure recall ≈ 0.62），并集为 273–282 条
（precision ≈ 0.95–0.97，failure recall ≈ 0.87）；第三种 lag-spectrum 图社区在定性上
落在同一区域，但其块边界随图划分种子剧烈变化，不构成第三张独立选票。**

### 5.3 一句话结论

> **共识核心结论以「逐位精确」的强度复现了原报告的全部 20 项声称数字（零不一致），
> 同种子重跑逐字节一致、无泄漏随机源；但种子稳健性检验表明，「三方法 ≥2 票 = 217」
> 中的精确计数与三支柱之一（lag-spectrum Louvain C7, n=219）都是种子实例——
> 前者跨种子为 204–217，后者在 39 个扰动种子中 0 次被单一社区复出。
> 结论的稳健内核是「aligned-kernel raw C1 ∩ change-event C0 ≈ 195 条、precision ≈ 0.98」，
> 应以区间而非点值报告。**

---

## 6. 产物清单

复现输出（均为新目录，未覆盖原有）：

```
analysis/all-outcome-routing-clusters-repro{,-seed20260830,-seed12345,-seed999}
analysis/alternative-routing-organizations-repro{,-seed20260830,-seed12345,-seed999}
analysis/aligned-route-kernel-repro{,-seed20260830,-seed12345,-seed999}
analysis/route-change-events-repro{,-seed20260830,-seed12345,-seed999}
analysis/_repro_logs/                        # 运行日志、耗时戳
analysis/_repro_logs/consensus.py            # 三方法共识统计脚本
analysis/_repro_logs/consensus_across_seeds.csv
analysis/_repro_logs/louvain_seed_sweep.csv  # 39 个扰动 Louvain 种子的扫描结果
```

## 7. 未做 / 限制

- 只扫了 3 个新种子做**完整**重跑（每方法），加 39 个种子只扫 Louvain 一步。
  raw C1 的种子分布只有 3 个样本，区间 [252, 264] 是下界估计。
- 未复现 `aligned-route-kernel-truncate90`、`peer-rank-c1-audit`、
  `routing-state-grammar` 等综合报告的其他章节（不在本任务的核对清单内）；
  运行期间有其它 agent 正在跑这些目录的 `-repro`。
- 未做 raw feature 重抽（复用了 `feature_cache`）。若特征缓存本身有误，本审计无法发现；
  但缓存 manifest 的 `extractor_sha256` / `input_fingerprints` 校验在每次运行时都通过了。
