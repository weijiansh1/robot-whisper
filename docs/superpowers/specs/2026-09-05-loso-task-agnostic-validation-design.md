# Leave-One-Suite-Out 任务无关性验证（v7 前置实验 A）

日期：2026-09-05
状态：设计已确认，待实现
范围：只验证，不改机制

## 1. 问题

`moe-v7-0905` 声称 task-agnostic，其 sealed manifest 中 `runtime_task_identity: false`、
`task_or_suite_parameters: false` 都成立——运行时确实只吃当前 `hb_router_probs`。这一层
（下称 **L0**）不需要重新验证。

问题在下一层。核对三个缓存的任务名：

```
reference corpus (main + extra) : 40 tasks / 16,000 traj
external_8b                     : 39 tasks / 15,600 traj
external \ reference            : 空集
```

39 个外部任务全部落在 40 个 reference 任务里。**v7 的 77.84% recall 从未在任何未见任务上
测过**，更不用说未见 suite。

### 1.1 四个语料常量

`experiments/evaluate_intrinsic_guard_v7.py:158-167` 产生的全局 profile 含四个标量：

| 常量 | 定义 | 作用位置 |
|---|---|---|
| `freeze_threshold` | pooled trajectory-peak `z_F` 的 q97.5 | 阈值 |
| `acceleration_threshold` | pooled trajectory-peak `z_A` 的 q70 | 阈值 |
| `periodicity_threshold` | pooled trajectory-peak `z_R` 的 q65 | 阈值 |
| `periodicity_scale` | pooled `\|lag_periodicity\|` 的 q75 | **分数内部**（`intrinsic_guard_monitor.py:182-184` 的除数） |

`z_F` 和 `z_A` 是对自身 baseline 的 log 比值，尺度自动消去，本身与语料无关；只有 `z_R`
把一个语料常数焊进了分数。四个常量都是这 40 个任务的经验分位数，因此都是
**corpus-conditional**。

### 1.2 长度偏置

trajectory-peak 分位数有长度偏置：轨迹越长，peak 越有机会冲高。各 suite 的 rollout 长度
量级完全不同：

| suite | reference traj | median length | p10 | p90 |
|---|---:|---:|---:|---:|
| libero_goal | 4,000 | 10 | 8 | 18 |
| libero_spatial | 4,000 | 11 | 9 | 13 |
| libero_object | 4,000 | 14 | 12 | 16 |
| libero_long | 4,000 | 24 | 19 | 40 |

`libero_long` 比 `goal`/`spatial` 长约 2.4 倍。pooled peak 分位数因此主要由语料的 suite 配方
决定，而不是由 MoE 路由决定。v7 报告中 `libero_long` 独占 80 个外部 FP 中的 69 个，怀疑相当
部分来自这里，而非 long 本身更难。

### 1.3 边界解

`select_operating_point.py:46-48` 的网格中，被选中的 `acceleration_quantile = 0.70` 恰好是
`ACCELERATION_QUANTILES` 的**下界**，`periodicity_quantile = 0.65` 也靠近 `PERIODICITY_QUANTILES`
的低端。这说明选择器想往更低走但被网格挡住，当前 operating point 不是内点最优。换语料后
它很可能要移到网格之外。

## 2. 目标与非目标

**目标**：给出 v7 在"整个 suite 从未进入校准"条件下的诚实指标，并把退化分解成两个来源
（阈值数值 vs. operating point）。据此决定实验 B（秩统计量重构）是否必要、要做到多狠。

**非目标**：
- 不改机制、不改 `intrinsic_guard_monitor.py`
- 不碰 GPU、不重跑 rollout、不重算原始 Zarr 特征
- 不修改 v4 / v6 / v7 任何已封存产物
- 不把 external 洗成 pristine holdout（见 §7）

## 3. 设计

### 3.1 折的划分

4 折，每折留出一个 suite。校准只用另外 3 个 suite，评估只用留出 suite 的 external 轨迹。

| fold | 留出 suite | 校准 reference | 校准 development 标签 | 评估 external | 评估 tasks | 评估 risk n |
|---|---|---:|---:|---:|---:|---:|
| F-goal | libero_goal | 12,000 | 10,800 | 4,000 | 10 | 106 |
| F-long | libero_long | 12,000 | 12,000 | 4,000 | 10 | 274 |
| F-object | libero_object | 12,000 | 10,800 | 3,600 | 9 | 44 |
| F-spatial | libero_spatial | 12,000 | 10,800 | 4,000 | 10 | 140 |

校准 reference 恒为 12,000 条，因为 reference 四个 suite 各 4,000 条（`libero_long` 由 main
2,800 + extra 1,200 组成）。校准 development 标签数不同，因为 `extra` 的 1,200 条 long 轨迹
只有路由、没有标签，只能进 reference、不能进 selection。

`quantile_higher` 要求至少 32 个有限参考值（`intrinsic_guard_monitor.py:101`），12,000 条远超，
无需放宽。

### 3.2 两级泄漏分解

这是本实验的核心设计。单跑一个 LOSO 数字说明不了问题，因为 v7 有两条独立的泄漏路径。

**LOSO-L1（threshold-only）**
分位数水平**固定**为已发布的 `(freeze 0.975, acceleration 0.70, periodicity 0.65, scale 0.75)`，
只把四个常量的**数值**用留出后的 12,000 条 reference 重算。
→ 隔离测量：语料配方敏感性。

**LOSO-L2（full）**
在 LOSO-L1 基础上，额外用校准 suite 的 development outcomes 重跑 640 点选择
（`8 × 8 × 10`，约束与 ranking 完全沿用 `select_operating_point.py:87-105` 的 primary 档：
`precision >= 0.85`、`timely_fpr <= 0.005`、`risk_recall >= 0.70`，按 `f1 → early4 → recall` 排序），
得到该折自己的分位数水平，再计算常量。
→ 隔离测量：operating point 是否也过拟合了 suite 组合。

**报告三条并列曲线**：`published`（现状，含全部泄漏）、`LOSO-L1`、`LOSO-L2`。
`L1 − L2` 的差即 operating-point 泄漏项。

若某折的 640 点网格中无候选满足 primary 约束，记录该折为 `infeasible` 并落盘全部候选，
**不得**放宽约束重试。这本身就是结论。

### 3.3 指标口径

与 v7 完全一致，便于逐行对比：

- 正类 = `original_failure`（persistent failure ∪ `+10` late success）；timely success 为负类
- `risk_recall`、`timely_fpr`、`precision`、`early{2,4,8,12}_risk_recall`、`detected_risk_lead_median`
- CI：task cluster bootstrap，5,000 draws，seed 20260905（与 v7 同参）

两点折内特有的口径：

1. **按 task 聚类，折内只有 9-10 个 task**，CI 必然远宽于 v7 报告里的全语料 CI。这是真实的
   不确定性，不做任何窄化处理。`F-object` 只有 44 个 risk，其 CI 会宽到近乎无信息量——照实报。
2. **必须显式给出 eligible 分母**：freeze 最早可能在 q6 报警、confirmed turbulence 最早 q10。
   `libero_goal` 的 p10 length 只有 8，`libero_spatial` median 11，相当比例的 episode 在物理上
   无法产生（或只能产生 freeze 类）报警。每折需报 `eligible_freeze_n` / `eligible_turbulence_n`，
   否则 recall 无法解释。

macro 平均按 4 折等权，同时给出 micro（按 episode 汇总），两者都报。

### 3.4 阈值漂移报告

`threshold_drift.csv` 是决定实验 B 的关键产物：4 折 × 4 常量 × {L1, L2}，加全语料基线值与相对
漂移百分比，再加"把该折常量换成基线常量后留出 suite 的 FPR/recall 变化"这一敏感度列。

本实验**不预设通过/不通过门限**，它是描述性的，不是假设检验。只预注册一条判读规则：

> 若 LOSO-L2 的 macro precision 低于 70%，v7 不得在论文或报告中宣称"任务无关"，
> 必须改述为"运行时不读任务身份，但校准依赖任务语料"。

### 3.5 封存纪律

沿用 v7 的顺序，每折独立执行：

1. 用校准 suite 的 reference（不读 outcome）算出四常量
2. L2 折：读**校准 suite 的 development outcomes** 跑 640 点选择——绝不触碰留出 suite 的任何
   outcome，development 与 external 都不行
3. 写入该折 profile + 留出 suite 的 first-alarm 数组 + SHA-256
4. 只有在 3 完成后，才打开留出 suite 的 external outcome 文件

### 3.6 产物

放在 `moe-v7-0905/` 下，因为这是对 v7 的验证而非新方法；只新增，不改动已封存文件。

```
moe-v7-0905/experiments/evaluate_loso_suite.py
moe-v7-0905/results/loso_validation/
    fold_profiles.npz              4 折 x 2 级的四常量
    threshold_drift.csv            §3.4
    fold_selection.json            L2 折的 640 点审计（含 infeasible 记录）
    loso_metrics.csv               overall / by-suite / by-task
    episode_alarms.csv             逐 episode first-alarm
    loso_manifest.json             哈希 + outcome 打开时点声明
moe-v7-0905/docs/LOSO_VALIDATION_REPORT_ZH.md
moe-v7-0905/tests/test_loso_suite.py
```

### 3.7 测试

- **隔离断言**：每折校准集的 task 名集合与留出 suite 的 task 名集合交集为空
- **selection 隔离**：L2 折 candidate 表的 episode 计数等于该折 development 标签数（§3.1 表），
  证明留出 suite 未参与选择
- **回归锚点**：用四个已发布常量走本脚本的评分路径，应逐 episode 复现
  `results/intrinsic_guard_v7/sealed_first_alarms.npz` 中的 `external_guard`。这条保证 LOSO 脚本
  与 v7 evaluator 计算等价，退化确实来自校准而非实现差异。
- **固定指标**：跑通后钉住数值
- **provenance**：SHA-256 校验

## 4. 数据来源

全部复用已有缓存，无新特征计算：

```
moe-v4-0904/results/layerwise_mobility/{main_reference,extra_reference,external_8b}.npz
double-selete/trainfree/results/online_multihead_hub{,_external}/unlabeled_query_features.npz
double-selete/trainfree/results/online_precision_cascade_external/unlabeled_query_features.npz
double-selete/trainfree/results/timeout_extension_plus10/{development_main,external_8b}_clean_labels.csv
```

计算量：4 折 × (1 + 640) 次评分，每次是 12,000-16,000 × T 的 numpy 数组运算。纯 CPU，分钟级。

## 5. 实现要点

复用 `method/intrinsic_guard_monitor.py` 的 `intrinsic_score_arrays`、`quantile_higher`、
`row_max`、`first_from_score`、`first_and`、`first_or`，不复制实现。

`periodicity_scale` 必须**随折重算**（它进 `intrinsic_score_arrays` 的入参），因此每折的分数
数组要重算一遍，不能复用 v7 的分数缓存。这是本实验唯一容易写错的地方。

## 6. 验收标准

1. 三条曲线（published / LOSO-L1 / LOSO-L2）× 4 折的完整表格
2. 阈值漂移表
3. 报告能明确回答一句话：
   > v7 在未见 suite 上的 macro precision/recall 为 X/Y，相对全语料下降 Z 个百分点，
   > 其中 operating-point 泄漏占 W 个百分点。
4. 据此给出实验 B 的必要性判断

## 7. 已知限制

- **external 不会因 LOSO 变成 pristine holdout**。它在 v3-v6 中被反复查看过。LOSO 消除的是
  "留出 suite 的任务身份进入了校准"这一条泄漏，消除不了"研究者已经看过这批数据"。报告结论
  必须保留这句限制。
- 只有 4 折且 suite 数就是 4，macro 平均的方差很大，不能当作稳定的泛化估计。
- `F-long` 是最极端的一折：校准集变成纯短 horizon 语料，按 §1.2 的长度偏置分析，预期最差。
  **不得因为它难看而事后剔除或改成 3 折。**
- `F-object` 只有 44 个 risk，统计功效很低，其点估计不应单独引用。
- 40 个 task 全部来自同一个 7B checkpoint。跨模型 / 跨本体漂移不在本实验范围。
- 本实验只回答"未见 suite"。"未见单任务、同 suite"（leave-one-task-out）没有覆盖，若需要是
  另一个 40 折实验。

## 8. 后续

本实验只决定实验 B 的必要性与强度。B 的方向（不在本 spec 范围）：把"分数 > 常量"换成前缀内
秩统计量——可交换性下当前分数为前缀 running max 的概率为 `1/q`，连续 K 次即给出分布无关、
无语料、无任务的误报上界；同时把 `periodicity_scale` 换成 rollout 自身前缀的 MAD。改完之后
判据里不含任何外部数字。
