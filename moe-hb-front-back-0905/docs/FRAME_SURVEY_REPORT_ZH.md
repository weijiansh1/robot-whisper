# 十二个 MoE 路由量的检测能力普查，以及"哪些信号真的只来自 MoE"

日期：2026-09-06
前置：`FLOW_AXIS_REPORT_ZH.md`、`SURVIVAL_BASELINE_ADDENDUM_ZH.md`

## 0. 动机与约束

AND 实验发现两个误报几乎不重叠的检测器合取后极干净。本次普查问两件事：

1. 这种不重叠是否来自**参照系**——每个量把"正常"定义在什么坐标系里；
2. 有没有更多互相独立的参照系。

同时加一条硬约束：**保证信号来自 MoE**。为此每个量都用两种阈值模式各跑一遍：

| 模式 | 阈值 | 是否含任务侧信息 |
|---|---|---|
| `per_task` | 同任务经验分位数（v4 做法） | **是**，运行时需要 task ID |
| `global` | 全语料池化的单一分位数（v7 做法） | **否** |

十二个量全部只从 `hb_router_probs` 计算，构造上就是 MoE-only。但只有在 `global` 模式下仍然
成立、且提升倍数明显大于 1 的量，才能说**判据本身**来自 MoE，而不是靠任务先验补偿。

## 0.5 "提前"的定义

本报告及自 `SURVIVAL_BASELINE_ADDENDUM_ZH.md` 起的所有报告中，"提前 TP/FP"指的是**报警时刻
的存活先验** `P(risk | 仍在运行于第 q 个 chunk) < 0.25`，即那一刻"还在跑"本身尚不足以判定失败。

这与 v7 原有的 `early-4`（`lead = length − 1 − first ≥ 4`，剩余 chunk 数）**不是同一口径**。
换用先验口径的原因见附录：原始精度大部分是 base rate，剩余时间多不等于信息上还早。

换算成具体位置，external 上先验跨过 0.25 的 chunk 为：

| suite | horizon | 先验 ≥ 0.25 起于 | 提前窗口 |
|---|---:|---:|---|
| libero_long | 52 | chunk 26 | 0–25 |
| libero_goal | 30 | chunk 18 | 0–17 |
| libero_object | 28 | chunk 17 | 0–16 |
| libero_spatial | 22 | chunk 13 | 0–12 |

`libero_long` 的先验到 q22 仍只有 0.09，窗口宽 26 个 chunk，这是它提前检出最多的结构性原因。
`libero_spatial` 的先验在 q10 还是 0.07、q14 已跳到 0.85，而它实测首报中位在 chunk 19，
**全部落在窗口关闭之后**——这就是它在全部十二个量下提前检出恒为 0 的原因。

**0.25 是人为切点，未做敏感性扫描。** 换成 0.1 或 0.4 时各检测器的相对排序可能改变。

## 1. 主结论

**能经得起"零任务信息"的只有两个量：`mobility` 和 `conditional_query_d1`。**

`global` 模式，external 15,600 条：

| 量 | 配置 | TP / FP | 精度 | 提前 TP/FP | **提升** | 召回 |
|---|---|---|---:|---|---:|---:|
| **mobility** | L12 low q0.975 | **195 / 17** | **0.920** | 36 / 4 | **1.764** | 0.346 |
| **conditional_query_d1** | L5 low q0.95 | 194 / 31 | 0.862 | 38 / 10 | 1.726 | 0.344 |
| expert_load_effective_rank | L3 low q0.925 | 237 / 36 | 0.868 | 30 / 22 | 1.294 | 0.420 |
| partial_query_d1 | L3 high q0.925 | 65 / 173 | 0.273 | 37 / 168 | 1.773 | 0.115 |
| **flow_settling_log_ratio** | L12 low q0.95 | **3 / 2** | — | 3 / 2 | — | **0.005** |
| flow_path | L15 high q0.99 | 50 / 4 | 0.926 | 3 / 4 | 1.235 | 0.089 |
| conditional_effective_rank | L3 high q0.925 | 88 / 158 | 0.358 | 46 / 155 | 1.963 | 0.156 |
| partial_edge_std | L3 high q0.90 | 99 / 147 | 0.402 | 55 / 143 | 2.009 | 0.176 |

**单层 L12 的 mobility 配一个全局阈值就拿到 195 TP / 17 FP、提升 1.764**——这个提升倍数高于
v7 那套三机制任务无关保护器（1.396），代价是召回低得多（0.346 vs 0.778）。

## 2. flow_settling 不是纯 MoE 判据

上一份报告里 `flow_settling_log_ratio` 是提升倍数最高的检测器（6.77×，83 个提前检出）。
**那是在 per-task 阈值下取得的。** 换成全局阈值后它只剩 3 个 TP：在统一的选择规则下，没有任何
一个全局配置能同时满足 FPR 与提前精度约束并保留有意义的检出量。

也就是说：这个量本身只读 `hb_router_probs`，但**把它变成判据需要知道任务**。它的判别力有实质
一部分来自任务侧标定，不能宣称是纯 MoE 信号。`expert_load_effective_rank` 同样退化（提升
1.548 → 1.294，提前精度 0.584 → 0.577）。

这条修正对上一份报告的结论有直接影响，已在该报告中交叉标注。

## 3. per_task 模式：最优组合换人了

| 组合 | TP / FP | 提前 TP/FP | 精度 | 提前精度 | 召回 |
|---|---|---|---:|---:|---:|
| expert_load_effective_rank（单） | 370 / 93 | **97** / 69 | 0.799 | 0.584 | 0.656 |
| flow_settling（单） | 304 / 87 | 83 / 62 | 0.777 | 0.572 | 0.539 |
| mobility（单） | 272 / 57 | 72 / 43 | 0.827 | 0.626 | 0.482 |
| **mobility ∧ expert_load** | **245 / 14** | **53 / 6** | 0.946 | 0.898 | 0.434 |
| mobility ∧ flow_settling（前报的 AND） | 187 / 8 | 40 / 2 | 0.959 | 0.952 | 0.332 |
| mobility ∧ flow_settling ∧ expert_load | 167 / 8 | 31 / 2 | 0.954 | 0.939 | 0.296 |
| flow_settling ∧ partial_edge_std | 85 / 2 | 27 / **0** | 0.977 | **1.000** | 0.151 |

两点：

1. **`expert_load_effective_rank` 是本次发现的最强单检测器**——97 个提前检出，超过
   flow_settling（83）和 mobility（72）。它测的是专家负载分布的熵有效秩，即"路由是否塌缩到
   少数专家"，与前面所有量都不同。
2. **三路合取不如最好的两路。** 加第三个只损失 TP，误报已经压到 8 个再降不动。
   `mobility ∧ expert_load` 相对上一份报告的 AND 多 58 个 TP、多 13 个提前 TP，只多 6 个误报。

## 4. 参照系假设：方向成立，但检验功率不足

误报依赖比定义为"观测共现 / 边际率乘积"（等于 1 表示独立）。用比值而非原始重叠率，是为了
排除"两个检测器都报得晚所以自然重叠"这个时机假象。

按事先声明的参照系分组：

```
组内（同参照系）  均值 177.6   中位 131.9   n=16
组间（跨参照系）  均值  75.6   中位  61.9   n=50
```

方向支持假设（组内约为组间的 2.35 倍），最低的实质配对是 `mobility × flow_settling` 的 24.3。

**但结论要打折**，有两个问题：

- 所有配对的比值都远大于 1。各检测器本来就在抓同一批"看起来怪"的 episode，参照系只是让它们
  **不那么**相关，不是让它们独立。
- 误报绝对数太少（10–93 个 / 15,036）。例如 `conditional_effective_rank`(10 FP) ×
  `partial_edge_std`(14 FP) 的独立预期共现只有 0.009，实测 0.00 完全在噪声内。
  **矩阵里所有 `0.00` 都不是独立性证据。**

## 5. 一个副产品：发现了代数冗余

`state_action_alignment` 与 `conditional_energy` 的依赖比是 **1253**，两者所有指标逐位相同。
原因在构造里：

```
conditional = action_gram − state_action ⊗ state_action
conditional_energy = mean(diag(conditional)) ≈ 1 − state_action²
```

它们是同一个量的单调变换，不是两个独立的参照系。`token_graph` 这一族的有效维度比声称的少一个。
这一点应在 `extract_layer_graphs_gpu.py` 的 METRIC_NAMES 注释里标明，避免后续再被当作两个证据。

## 6. 限制

- external 8B 在前序工作中已被反复查看，不是全新盲测。
- 存活先验在被评分的同一 cohort 上估计。
- 参照系分组是**运行前声明的假设**，依赖矩阵是对它的检验；检验功率受误报稀少限制（§4）。
- 所有配置在 W4/K4 下只扫了表征、方向、分位数，未扫窗口与持续长度。两种阈值模式各自独立
  选择，因此同一个量在两种模式下可能落在不同的层。
- 合取的报警时刻取两者较晚者，因此合取天然推迟报警、平均先验升高，提升倍数普遍低于单检测器。
- 未在未见任务/未见 suite 下验证；`global` 模式虽然不用 task ID，但阈值仍标定自同一批任务的
  语料，与 `moe-v7-0905` 的 LOSO 结论共享同一限制。
- `libero_spatial` 在全部十二个量下依然没有任何低先验检出，与前三份报告一致。

## 7. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-hb-front-back-0905
python experiments/survey_reference_frames.py
```

约 54 秒，纯 CPU。产物在 `results/frame_survey/`：`development_candidates.csv`（12 量 × 11 表征
× 2 方向 × 14 分位 × 2 阈值模式的完整开发集评分）、`external_detectors.csv`、
`external_first_alarms.npz`（24 组逐 episode 首报，可直接重做任意合取）、
`false_alarm_dependence_global.csv`、`survey.json`。
