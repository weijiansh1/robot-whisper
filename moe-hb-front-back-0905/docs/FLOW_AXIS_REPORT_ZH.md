# 去噪步方向：一个此前未被用于检测的信号轴

日期：2026-09-05
前置：`SURVIVAL_BASELINE_ADDENDUM_ZH.md`、`EARLY_LOCK_REPORT_ZH.md`

## 1. 结论

本项目所有 lock 头都只读**最后一个去噪步**（`FINAL_FLOW = 9`）的路由。`layer_graphs` 缓存里
其实已经有三个描述**去噪轨迹本身**的量，从未被当作检测器测过。

测下来其中一个是强信号，而且和 mobility 机制不同、高度互补：

```
flow_settling_log_ratio = log( 后三个去噪步的平均速度 / 前三个去噪步的平均速度 )
```

取 **high 方向**——即"denoising 到最后还在快速移动，这个 action chunk 没有收敛"。

| 检测器 | 总 TP/FP | 低先验 TP/FP | 低先验精度 | **低先验提升** | 召回 |
|---|---|---|---:|---:|---:|
| v4（参考） | 410 / 81 | 76 / 54 | 0.585 | 5.08× | 0.727 |
| mobility L2 q0.70 | 272 / 57 | 72 / 43 | 0.626 | 5.68× | 0.482 |
| **flow_settling front_median q0.85** | 304 / 87 | **83** / 62 | 0.572 | **6.77×** | 0.539 |
| **两者 AND** | **187 / 8** | 40 / **2** | **0.952** | 5.89× | 0.332 |

`flow_settling` 单独就是目前测过**信息增量最高**的检测器（6.77×）。

> **后续修正（`FRAME_SURVEY_REPORT_ZH.md`）**：上表全部使用**同任务分位阈值**。换成不含任何
> 任务侧信息的全局池化阈值后，`flow_settling` 只剩 3 TP / 2 FP，即在统一选择规则下没有任何
> 全局配置能保留有意义的检出量。**这个量本身只读 `hb_router_probs`，但把它变成判据需要知道
> 任务**，因此不能宣称是纯 MoE 判据。经得起零任务信息的只有 `mobility` 与
> `conditional_query_d1`。本报告其余结论（与 mobility 的互补性、合取的干净程度）不受影响，
> 但都应理解为"在已知任务的前提下"。

## 2. 两个轴测的是不同的东西

- **mobility**：相邻 query 之间、最终去噪步上的路由变化——**chunk 与 chunk 之间**变了没有。
- **flow_settling**：单个 query 内部、沿十个去噪步的速度衰减——**chunk 内部**收敛了没有。

低先验区间的检出重叠证实它们不是同一件事：

```
mobility 命中 72   flow 命中 83   两者都命中 40   并集 115
仅 mobility 32     仅 flow 43
```

只有 35% 重叠，flow 有 43 个提前检出是 mobility 完全漏掉的。

**更关键的是误报几乎不重叠**：mobility 43 个、flow 62 个，共同的只有 **2 个**。

## 3. 合取极其干净，且跨 cohort 复现

误报近乎正交意味着 OR 会把误报累加（并集 103 个，精度掉到 0.528），而 **AND 会把误报几乎清零**：

| | 开发集 14,800 | external 15,600 |
|---|---|---|
| AND 总计 | 183 TP / **1 FP**（精度 0.995） | 187 TP / **8 FP**（精度 0.959，timely FPR 0.053%） |
| AND 低先验 | 26 TP / **0 FP**（精度 1.000） | 40 TP / **2 FP**（精度 0.952） |

**15,600 条 external 里只有 8 次误报。** 对比 v4 的 81 次。

分 suite（external，AND）：

| suite | 总 TP/FP | 低先验 TP/FP |
|---|---|---|
| libero_goal | 49 / 2 | **31 / 1** |
| libero_long | 45 / 2 | 9 / 1 |
| libero_object | 13 / 1 | 0 / 0 |
| libero_spatial | 80 / 3 | 0 / 0 |

AND 的提前检出集中在 `libero_goal`（40 个里 31 个）。`libero_spatial` 依然是 **0 个提前检出**——
和前两份报告一致，spatial 在任何已测表征下都没有可提前读出的信号。

## 4. 代价与定位

AND 的召回只有 0.332（v4 是 0.727），而且它的**平均报警先验是 0.750**（v4 0.575），提升倍数
5.89× 略高于 v4 但低于 flow_settling 单独。原因是 `first_and` 取两者报警时刻的较大值，合取
天然会推迟报警。

所以三者定位不同，不是替代关系：

- **要高置信度触发干预**：用 AND。15,600 条里只错 8 次，提前区间 40 次里只错 2 次。
- **要最大信息增量**：用 flow_settling 单独，6.77×。
- **要覆盖率**：v4 的 0.727 召回仍最高，但其中大部分是高先验区间的晚期检出。

## 5. 其余两个 flow 量较弱

同一协议下：

| 量 | 最佳配置 | 低先验 TP/FP | 低先验精度 |
|---|---|---|---:|
| flow_path | L3 high q0.96 | 43 / 21 | 0.672 |
| flow_endpoint | L12 high q0.96 | 42 / 23 | 0.646 |
| flow_settling_log_ratio | front_median high q0.85 | **83 / 62** | 0.572 |

`flow_path`（路径总长）和 `flow_endpoint`（首末直线距离）都只是"这个 chunk 的路由走了多远"，
而 `flow_settling` 是"走的过程有没有慢下来"。**是速度的时间剖面在带信息，不是位移量本身。**

## 6. 选择纪律

必须分清三件事的来源：

1. **`flow_settling` 的配置**（front_median / high / W4 / K4 / q0.85）由
   `compare_flow_metrics_early.py` 在**开发集**上按一条对三个 flow 量统一适用的固定规则选出，
   合规。
2. **mobility L2 q0.70** 来自 `EARLY_LOCK_REPORT_ZH` 的开发集入围名单，合规。
3. **AND 这个组合本身没有预注册。** 我是在 external 上观察到"两者误报几乎不重叠"之后才构造
   它的。事后在开发集上验证，得到 183 TP / 1 FP、提前区间 26 TP / 0 FP，与 external 同方向且
   更强——这是对**组合**的独立确认，但**构造这个组合的想法来自 external**，不能声称是盲测。

要真正确认，需要一批新 seed 或新任务的 cohort，把 AND 规则连同两个阈值一起冻结后再跑。

## 7. 限制

- external 8B 在 v3–v6 及本 bundle 前序工作中已被反复查看，不是全新盲测。
- 存活先验在被评分的同一 cohort 上估计，与前序报告同一限制。
- 阈值仍是**同任务经验分位数**，需要运行时 task ID；未在未见任务/未见 suite 下验证。
- AND 的提前检出只有 40 个，其中 31 个来自 `libero_goal`，跨 suite 泛化性未知。
- `libero_object` 在 AND 下 0 个提前检出，`libero_spatial` 同样，两个 suite 无有效结论。
- 三个 flow 量都只在 W4/K4 下扫了分位数和方向，没有扫窗口与持续长度。
- 本实验没有触及"报警之后干预是否有用"，那仍需同 snapshot fork 的反事实实验。

## 8. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-hb-front-back-0905
python experiments/compare_flow_metrics_early.py
```

约 14 秒，纯 CPU。产物在 `results/flow_early/`：`development_flow_candidates.csv`（三个 flow 量
× 11 表征 × 2 方向 × 14 分位的完整开发集评分）、`external_flow.csv`、`comparison.json`。

AND 组合的逐 cohort 数字由报告正文给出，脚本化封存留待冻结确认时一并完成。
