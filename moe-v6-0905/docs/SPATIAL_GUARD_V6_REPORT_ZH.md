# MoE Spatial Guard v6 优化报告

## 结论

v6 修复了 lead guard v5 将 spatial 的早报配置同时应用到 goal/object 所造成
的误报扩散。最终策略按 suite 分三档：

```text
libero_long    : v5 long guard, L12 low, W4, K8, q70
libero_spatial : v6 spatial guard, back_median low, W1, K2, q80
goal / object  : v4 lock, all_median low, W4, K4, q75
instability    : v4 L5 high, W4, K8, q80（全部 suite）
```

External 8B 结果：

| 档位 | TP / FP | 风险召回 | Early-4 | 精度 | 非风险 FPR |
|---|---:|---:|---:|---:|---:|
| v4 baseline | 410 / 81 | 72.70% | 58.51% | 83.50% | 0.539% |
| v5 long guard | 391 / 34 | 69.33% | 55.67% | 92.00% | 0.226% |
| v5 lead guard | 402 / 73 | 71.28% | 61.35% | 84.63% | 0.486% |
| **v6 spatial guard** | **408 / 45** | **72.34%** | **62.77%** | **90.07%** | **0.299%** |

相对 v4，v6 少 2 个总 TP，但多 24 个至少提前 4 query 的 TP，并减少 36 个
FP；Early-4 增加 4.26 个百分点、精度增加 6.56 个百分点、FPR 相对下降
44.4%。相对 lead v5，v6 同时增加 6 个 TP、8 个 Early-4 TP，并减少 28 个
FP，因此在本 cohort 上严格支配 lead v5。

## 1. 选择过程

搜索只读取 development cohort 的 14,800 条轨迹。51,200 个候选由 64 个
方向化表示、5 个窗口宽度、8 个持续长度和 20 个分位数组合而成。W8 在搜索
前被排除，因为短轨迹任务无法提供最低 32 个有限 cross-fit reference peak；
没有插补缺失值。

候选必须同时满足：整体 FPR <= 0.25%、精度 >= 90%、召回 >= 75%、最差
suite FPR <= 1%、最差 task FPR <= 2.5%。GPU sweep 找到 31 个合格候选，
这 31 个候选全部经 NumPy float32 路径复核，排序第一仍为：

```text
back_median low mobility, W1, K2, q80
development: 366 TP / 34 FP
recall 75.15%, Early-4 67.15%, precision 91.50%, FPR 0.238%
```

阈值对每个 task 按 initial state 做 leave-one-group-out 校准：被评估的 8 条
seed 不进入其阈值 reference，其余 392 条轨迹参与 higher quantile。阈值校准
不读取 outcome；method selection 使用 development outcome。

## 2. Suite 结果

v6 的 External 8B 报警由下列四部分构成：

| Suite | TP / FP | Recall | Early-4 | Precision | FPR |
|---|---:|---:|---:|---:|---:|
| goal（v4） | 88 / 5 | 83.02% | 74.53% | 94.62% | 0.128% |
| long（v5 long） | 177 / 24 | 64.60% | 62.77% | 88.06% | 0.644% |
| object（v4） | 31 / 2 | 70.45% | 52.27% | 93.94% | 0.056% |
| spatial（v6） | 112 / 14 | 80.00% | 57.14% | 88.89% | 0.363% |

外部最差 task FPR 为 1.393%，最差 suite FPR 为 0.644%，均低于开发选择
约束。spatial 的主要收益来自 stove、top-drawer 和 wooden-cabinet 三个任务；
新增 FP 也集中在少数 spatial 任务，未再污染 object/goal。

## 3. GPU 使用

运行时设置 `CUDA_VISIBLE_DEVICES=6,7`，两张 H20-3e 各处理 25,600 个候选。
summary 记录的 worker 时间约 2.5 秒/卡，峰值 PyTorch allocation 约 143 MiB；
外部 `nvidia-smi dmon` 采样到两卡同时约 60%-67% SM 利用率。GPU 负责表示、
滑窗、cross-fit 分位数、持续判断及按 task/suite 的批量计数；最终 shortlist
在 CPU 上复核。

## 4. 部署与验证

`deployment_profiles.npz` 为 39 个外部任务保存 representation、direction、
width、confirmations 和 outcome-blind threshold。`SpatialGuardMonitor` 每次只
接收当前 `hb_router_probs`，以 float32 因果累计均值、连续越阈计数和 latch
产生报警。

测试从原始 Zarr 路由分别回放 long、spatial、goal profile，逐 query 对齐
lock、instability 和最终首次报警位置。固定指标、profile 模式、provenance
SHA-256 也在测试中检查。

```bash
cd /home/jovyan/work/himoe-vla/moe-v6-0905
CUDA_VISIBLE_DEVICES=6,7 \
  python experiments/evaluate_spatial_guard_v6_gpu.py
pytest -q tests
```

## 5. 有效性边界

External 8B 在 v3/v4/v5 中已经被查看，本轮 suite 限制也受到 v5 外部误差分析
启发，因此不是新盲测，不能用来声称泛化提升已经确认。v6 还依赖运行时 task
ID；unknown-task cold start 不在本轮优化范围内。

下一次确认应在运行前固定当前 profile 与哈希，并生成新 seed 或新任务 cohort。
至少报告 v4、long v5、spatial v6 三档，按 task cluster bootstrap 比较 Early-4、
precision 和 FPR；在此之前不应把 v6 标为唯一默认部署档。
