# 十步去噪的功能分化：HiMoE-VLA 路由的 flow 轴

目录：`moe-flow-semantics-0906/`。所有量均只来自 `hb_router_probs`，不使用 episode 长度、结果标签、墙钟时间或任务身份。

---

## 结论先行

**十个去噪步确实功能不同，但它们不是"前段/后段"那样的两块划分，而是一条单调梯度。**

一句话的机制：**早期步在听采样噪声，晚期步在听观测**。这个方向在 8 个 HB 层里全部成立，在 40 个开发任务和 39 个外部任务里全部同向复现。

两个可量化的核心数字：

1. **query 0 的噪声/观测识别**（同一初始状态下观测完全相同，只有 flow 噪声种子不同 → 纯噪声；同一种子不同初始状态 → 纯观测）。观测/噪声比值从 step 0 到 step 9 单调上升，8/8 层成立：L2 从 0.190 升到 0.740，L12 从 1.090 升到 2.799。
2. **既有"前/后层"结论只是终点步的说法**。跨 query mobility 的 back−front 差值在 step 0 是负的（前层动得更多），在 step 9 是正的（后层动得更多）；**在两个 run_id 里都恰好在 step 3→4 之间穿零**。开发集 37 个任务中 back>front 的任务数从 step 0 的 4 个升到 step 9 的 37 个；外部集从 4/39 升到 39/39。

实用结论（本次最重要的子问题）：**是的，非终点步携带终点步没有的检测信号，而且量很大。**
在 per_task 阈值下，step 4 的 mobility 头给出 344 TP / 61 FP（recall 0.610），而 step 9 只有 272 / 57（recall 0.482）；step 0 单独抓到 **96 个 step 9 漏掉的风险 episode**，代价是多 29 个误报。在 global（无任务身份）阈值下，一个读整条 flow 的量 `mobility_step_range` 给出 **226 TP / 20 FP，precision 0.919，recall 0.401，lift 1.626**，对比基线 L12-step9-mobility 的 195 / 17、precision 0.920、recall 0.346、lift 1.764 —— **精度持平、召回相对提升 16%，但报警时刻更晚，所以 lift 更低**。没有任何一个头在有意义的报警量下超过基线的 lift 1.764。

---

## 一、做了什么

### 数据与提取

从 `hb_router_probs`（`[query, 8 层, 10 步, 11 token, 32 专家]`，float16）完整保留步轴，抽取三个 cohort：

| cohort | run_id | 任务 | episode | 有效 query |
|---|---|---|---|---|
| development_main | right-50x8-20260903 | 37 | 14 800 | 221 781 |
| development_extra | right-50x8-20260903 | 3 | 1 200 | 31 941 |
| external_8b | right-50x8b-20260903 | 39 | 15 600 | 248 255 |

**无子采样，无任务丢弃，全部 501 977 个 query 全部计算。** 纯 CPU、24 个 fork worker，全量 81 秒（GPU 当时 8 卡全部 139/143 GB 占用、100% 利用率，因此完全不碰 GPU）。

每个 (query, 层, 步) 计算八个"步内"量 + 两个"跨 query"量：

| 量 | 定义 | 为什么选它 |
|---|---|---|
| `token_entropy` | 10 个 action token 的 H(p) 均值 | 该步路由器的犹豫程度 |
| `load_entropy` | H(action token 平均后的 p) | 该步产生的**专家负载**集中度 |
| `token_differentiation` | `load_entropy − token_entropy` | 在均匀 token 先验下这**恰好是互信息 I(token; expert)**：所有 token 路由相同时为 0，token 之间分家时增大。这是"token 结构分化"在步轴上的直接测量 |
| `action_consensus` | 45 个 action token 对的 Bhattacharyya 系数均值 | 同一问题的二阶（几何）视角 |
| `state_action_alignment` | state token 与各 action token 的 BC 均值 | 该步路由还有多少是"条件路由" |
| `conditional_energy` | 去掉 state 方向后 action Gram 的对角均值 | 剩余的 token 自有路由强度 |
| `conditional_effective_rank` | 上述条件 Gram 特征值的熵有效秩 / 10 | 该步 token 构型的维数 |
| `flow_speed` | 与上一步 action 路由的 Hellinger 距离 | 缓存里 `flow_path` 的逐步被加项（step 0 为 NaN） |
| `mobility_step` | 与**上一 query** 同一步 action 路由的 Hellinger 距离 | step 9 处与冻结的 v4 `mobility` 定义逐位一致；step 0–8 从未被算过 |
| `state_mobility_step` | 同上但只看 state token | 观测变化探针 |

**校验**：新算的 step 9 mobility 与冻结的 v4 缓存在三个 cohort 上分别相差 2.19e-6 / 1.37e-6 / 1.89e-6（纯 float32 舍入；v4 算 `sqrt(p_a·p_b)`，本文算 `sqrt(p_a)·sqrt(p_b)`）。见 `tests/test_anchors.py`。

### query 0 的因果识别

每个任务是 50 个初始状态 × 8 个 flow 噪声种子（1000–1007），且**每个初始状态复用同一组 8 个种子**。在 query 0 时：

- 同初始状态、不同种子 → 观测完全相同，差异**只来自 flow 噪声**；
- 同种子、不同初始状态 → 差异**只来自观测**（前提：种子决定的噪声张量与状态无关）。

这给出了"每一步在听什么"的干净识别，是层轴上"前层响应观测变化"结论在步轴上的对应实验。

---

## 二、每个量沿步轴怎么演化

以下是 `development_main` 的全体有效 query 均值；`external_8b` 的对应表在 `results/step_structure/step_profiles.csv`，两者逐格差异极小（例如 `token_differentiation` 最大差 1.7e-4，`conditional_effective_rank` 最大差 7.7e-4）。

**路由熵**（自然对数，上界 ln 32 = 3.4657）——后层随步单调变锐，前层几乎不动：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 3.45631 | 3.45606 | 3.45636 | 3.45633 |
| L5 | 3.45629 | 3.45573 | 3.45529 | 3.45262 |
| L12 | 3.45769 | 3.45745 | 3.45681 | 3.45413 |
| L15 | 3.45736 | 3.45666 | 3.45502 | 3.45002 |

**专家负载有效秩** exp(H_load)/32 —— L15 从 0.99241 降到 0.98913，L2 基本不动（0.99215 → 0.99206）。负载集中化几乎全部发生在后层的后半段 flow。

**token 分化 I(token; expert)**（nats）—— 后层翻 2.5 倍，前层反向：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 0.00402 | 0.00402 | 0.00390 | 0.00327 |
| L5 | 0.00398 | 0.00450 | 0.00509 | 0.00631 |
| L12 | 0.00184 | 0.00222 | 0.00292 | 0.00467 |
| L15 | 0.00221 | 0.00272 | 0.00360 | 0.00555 |

**token 图结构** `conditional_effective_rank` —— L15 从 0.12593 升到 0.15316（+22%），L2 从 0.10853 降到 0.10699（−1.4%）。

**步间移动** `flow_speed` —— 中间平坦、末步暴涨。L12：s1 0.00458 → s9 0.01623（3.5 倍）。**8/8 层最大的归一化增量都出现在 s8→s9**（0.446–0.556），这也正是既有 `flow_settling_log_ratio` 能工作的来源。

**跨 query mobility** —— 前后层方向完全相反：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 0.0424 | 0.0406 | 0.0375 | 0.0345 |
| L5 | 0.0410 | 0.0413 | 0.0423 | 0.0474 |
| L12 | 0.0327 | 0.0360 | 0.0418 | 0.0556 |
| L15 | 0.0355 | 0.0393 | 0.0460 | 0.0619 |

**state token 的路由在步轴上是不变的。** 整条 flow 上 state token 路由的 Hellinger 位移只有 2e-5 到 4e-4，而 action token 是 0.027–0.033，相差 100–1500 倍。因此 `state_mobility_step` 在十个步上完全相同（步间相关阵恒等于 1.000，late−early 差值在 40/40 个开发任务上精确为 0），它作为"步轴上的量"携带零信息。**所有步依赖性都住在 action token 里。**

---

## 三、有没有"前段/后段"式的分块？——没有，是梯度

用同一个打分函数比较两条轴：把每个量的 10×10（步）与 8×8（层）**任务内**相关矩阵做 Fisher 平均，再对所有两分块切法取"块内平均相关 − 块间平均相关"的最大值。

`development_main`（外部集数字见 `results/step_structure/axis_block_split.csv`，结论一致）：

| 量 | 步轴最佳切点 / 分数 | 层轴 4/4 切点分数 | 步轴 s0↔s9 相关 | 层轴 L2↔L15 相关 |
|---|---|---|---|---|
| mobility | 7 / **0.0397** | **0.1377** | 0.880 | 0.596 |
| conditional_effective_rank | 8 / **0.0357** | **0.2446** | 0.903 | 0.230 |
| token_differentiation | 6 / **0.0692** | **0.1464** | 0.798 | 0.459 |
| load_entropy | 9 / **0.1000** | **0.1111** | 0.765 | 0.138 |
| flow_speed | 4 / **0.0870** | **0.0887** | 0.680 | 0.745 |

步轴的非对角平均相关是 0.93–0.97（层轴 0.09–0.76）。**步轴的分块强度比层轴弱 2–7 倍**：十个步是一条高度共变的单调梯度，不是两个功能块。

唯一被单独挑出来的步是 **step 9**（最后一个 Euler 步）：`flow_speed` 在 8/8 层、`action_consensus` 与 `conditional_effective_rank` 在 4/4 后层，最大归一化增量都落在 s8→s9。

**但是"梯度"不等于"冗余"。** 分层看跨 query mobility 的 step 0 与 step 9 相关：

| | L2 | L3 | L4 | L5 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|
| development_main | 0.236 | 0.340 | 0.375 | 0.572 | 0.876 | 0.878 | 0.883 | 0.808 |
| external_8b | 0.347 | 0.395 | 0.462 | 0.571 | 0.877 | 0.873 | 0.882 | 0.801 |

**前层的 step 0 mobility 与 step 9 mobility 几乎是两个不同的量**（L2 只有 0.24 / 0.35）。既有的所有检测器都只读 step 9，因此把前层这一半信息整个丢掉了。

---

## 四、每一步在听什么：噪声 vs 观测

query 0 的识别结果（`development_main`，action token，成对 Hellinger 距离均值）：

**纯噪声敏感度（同观测、异种子）随步单调下降**：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 0.0401 | 0.0372 | 0.0323 | 0.0205 |
| L12 | 0.0215 | 0.0203 | 0.0191 | 0.0168 |

**纯观测敏感度（同种子、异初始状态）随步单调上升**：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 0.0078 | 0.0093 | 0.0118 | 0.0155 |
| L12 | 0.0238 | 0.0290 | 0.0357 | 0.0466 |

**观测/噪声比，8/8 层单调上升**：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 0.190 | 0.246 | 0.356 | 0.740 |
| L5 | 0.301 | 0.386 | 0.524 | 1.016 |
| L12 | 1.090 | 1.406 | 1.839 | 2.799 |
| L15 | 1.163 | 1.456 | 1.847 | 2.739 |

前四层直到最后一步才勉强跨过 1（L5 在 s9 才到 1.016），后四层从 s0 就已经在 1.1 以上。这就是"前层/后层"和"早步/晚步"两条轴的关系：**它们不是同一件事，而是相乘的**——后层 + 晚步 = 最听观测；前层 + 早步 = 最听采样器。

state token 作为对照：噪声敏感度 1e-4–4e-4（基本为零），观测敏感度 0.076–0.111，比值 173–1274，且它的观测敏感度在十个步上到小数点后四位完全相同。

### 把 mobility 拆成噪声地板 + 余量

相邻 query 之间**噪声是重抽的**，所以上面的"纯噪声"距离就是同一步上 mobility 的地板。余量占比 =(mobility − 噪声地板)/mobility：

| 层 | s0 | s3 | s6 | s9 |
|---|---|---|---|---|
| L2 | 5.6% | 8.4% | 13.8% | 40.5% |
| L5 | 9.8% | 13.3% | 21.7% | 51.1% |
| L12 | 34.4% | 43.6% | 54.2% | 69.9% |
| L15 | 43.6% | 51.1% | 58.9% | 70.7% |

也就是说：**前层 step 0 的 mobility 有 94% 是对重抽噪声的响应，只有 6% 是对场景的响应。** 这解释了第三节里 L2 的 s0↔s9 低相关，也提出一个尚未验证的机制假设：step 0 检测器之所以能工作，读的可能是**路由器对噪声响应的增益**（增益变小 = 路由变"僵"），而不是对场景变化的响应本身。见第七节的诚实说明。

---

## 五、复现性

对每个量取"晚（步 7–9 均值）减早（步 0–2 均值）"的对比量，逐任务看符号是否与总体方向一致。开发集 = 40 个任务（37 main + 3 extra），外部集 = 39 个任务（另一个 run_id）。

| 量 | 层组 | 方向 | 开发一致 | 外部一致 |
|---|---|---|---|---|
| token_entropy | back−front | 晚 < 早 | **40/40** | **39/39** |
| load_entropy | back−front | 晚 < 早 | **40/40** | **39/39** |
| token_differentiation | back−front | 晚 > 早 | **40/40** | **39/39** |
| action_consensus | back−front | 晚 < 早 | **40/40** | **39/39** |
| conditional_effective_rank | back−front | 晚 > 早 | **40/40** | **39/39** |
| flow_speed | back−front | 晚 > 早 | **40/40** | **39/39** |
| mobility | back−front | 晚 > 早 | **40/40** | **39/39** |
| state_action_alignment | back−front | 晚 < 早 | 27/40 | 26/39 |
| conditional_energy | back−front | 晚 > 早 | 30/40 | 29/39 |
| state_mobility | back−front | — | 22/40 | 20/39（差值为 1e-7 级浮点噪声） |

单层看：L12、L15 在全部七个"有方向"的量上都是 40/40 与 39/39；L2 只在 mobility 和 flow_speed 上是 40/40，其余多在 21–37/40 之间——**前层的步依赖性本来就弱**，与第二、四节一致。

**front/back 交叉点在两个 run_id 上完全一致**（back>front 的任务数）：

| 量 | cohort | s0 | s2 | s3 | s4 | s6 | s9 |
|---|---|---|---|---|---|---|---|
| mobility | dev_main (37) | 4 | 10 | 17 | 27 | 33 | **37** |
| mobility | external (39) | 4 | 10 | 14 | 27 | 34 | **39** |
| token_differentiation | dev_main (37) | 0 | 0 | 0 | 0 | 1 | **32** |
| token_differentiation | external (39) | 0 | 0 | 0 | 0 | 1 | **32** |

mobility 的中位数 back−front 在 dev 是 −0.00018(s3) → +0.00125(s4)，在 external 是 −0.00047(s3) → +0.00097(s4)：**两个 run_id 的穿零点都在 step 3 与 step 4 之间。** `token_differentiation` 的穿零点都在 step 7 与 step 8 之间。

`conditional_effective_rank` 则在十个步上都是 back>front（37/37、39/39），没有交叉——说明"后层 token 构型维数更高"不是终点步的特例。

3 个 libero_long 的 extra 任务穿零更晚（step 5→6），是唯一系统性的偏离。

---

## 六、检测器：非终点步是否带来终点步没有的信号

### 协议

`experiments/protocol.py` 是既有协议的**逐字副本**（W=4 因果滑动均值、K=4 连续确认、同一 14 档分位网格、11 个层/组表示、per_task 交叉拟合阈值与 global 池化阈值、同一固定挑选规则：`timely_fpr ≤ 0.005` 且 `low_prior_precision ≥ 0.60`，按 `low_prior_tp` 再 `low_prior_precision` 排序）。挑选只读开发集标签，外部集只打开一次。

**两个锚点都精确复现**（`tests/test_anchors.py`）：
- `mobility_s9` per_task → L2 low q0.70 → **272 TP / 57 FP**（与已发表值一致）；
- `mobility_s9` global → L12 low q0.975 → **195 TP / 17 FP，precision 0.920，lift 1.764，recall 0.346**（与已发表值一致）；
- 额外：`load_entropy_s9` per_task → L3 low q0.850 → **370 TP / 93 FP，lift 1.548**，与已发表的 "expert-load effective rank" 基线逐位相同（有效秩是熵的严格单调变换）。

### 逐步 mobility 的外部结果

| 步 | mode | 表示 | TP | FP | precision | recall | early TP/FP | 匹配先验 | lift |
|---|---|---|---|---|---|---|---|---|---|
| s0 | per_task | L13 low | 318 | 50 | 0.864 | 0.564 | 75 / 39 | 0.596 | 1.449 |
| s2 | per_task | L13 low | 333 | 51 | 0.867 | 0.590 | 72 / 40 | 0.581 | 1.492 |
| **s4** | per_task | L13 low | **344** | 61 | 0.849 | **0.610** | 67 / 47 | 0.573 | 1.483 |
| s8 | per_task | all_median low | 309 | 51 | 0.858 | 0.548 | 61 / 34 | 0.645 | 1.331 |
| s9 | per_task | L2 low | 272 | 57 | 0.827 | 0.482 | 72 / 43 | 0.595 | 1.389 |
| s0 | global | back_median low | 109 | 11 | 0.908 | 0.193 | 22 / 11 | 0.660 | 1.377 |
| s3 | global | back_median low | 142 | 5 | **0.966** | 0.252 | 26 / 4 | 0.590 | 1.638 |
| s5 | global | L14 low | 206 | 19 | 0.916 | 0.365 | 38 / 11 | 0.530 | 1.727 |
| **s8** | global | L14 low | **239** | 39 | 0.860 | **0.424** | 43 / 21 | 0.518 | 1.660 |
| s9 | global | L12 low | 195 | 17 | 0.920 | 0.346 | 36 / 4 | 0.521 | **1.764** |

（完整十步 × 两种阈值见 `results/step_alarm/external_detectors.csv`。）

**开发集在打开外部集之前就给出了同样的排序**：per_task 下开发集可行头的最大 TP 在 step 4（314）、最小在 step 9（265）；global 下最大 TP 在 step 8（211）、step 9 是 190，而 global 的最佳 lift 在 step 9（开发 1.743 / 外部 1.764）。**"早/中步召回更高、终点步单报警更干净"这个模式在两个 run_id 上都成立。**

### 非终点步捞到了什么

| head | mode | TP | FP | step 9 漏掉但它抓到的风险 | 相对 step 9 新增的误报 |
|---|---|---|---|---|---|
| mobility_s0 | per_task | 318 | 50 | **96** | 29 |
| mobility_s4 | per_task | 344 | 61 | **106** | 38 |
| mobility_s8 | global | 239 | 39 | **44** | 22 |
| mobility_step_range | global | 226 | 20 | **41** | 10 |
| mobility_s0 | global | 109 | 11 | 5 | 10 |

**答案是明确的是。** 在 per_task 下 step 0 抓到 96 个 step 9 完全漏掉的风险 episode；在 global 下读整条 flow 的 `mobility_step_range` 抓到 41 个，只多付 10 个误报。

### 读整条 flow 的两个派生量（global，无任务身份）

| head | 表示 | TP | FP | precision | recall | early TP/FP | lift |
|---|---|---|---|---|---|---|---|
| **mobility_step_range** | all_median **low** q0.975 | **226** | **20** | **0.919** | **0.401** | 42 / 7 | 1.626 |
| 基线 mobility_s9 | L12 low q0.975 | 195 | 17 | 0.920 | 0.346 | 36 / 4 | **1.764** |
| mobility_settling_log_ratio | all_median high q0.960 | 35 | 14 | 0.714 | 0.062 | 12 / 11 | 1.673 |

`mobility_step_range` = 十个步的 mobility 最大值减最小值，方向是"低"：**当整条 flow 的跨 query 响应在十个步上变得一致（不再有前后分化）时报警。** 它以完全相同的精度（0.919 vs 0.920）多抓 31 个风险（相对召回 +16%），早期带也从 36/4 变成 42/7。代价是它的报警平均落在更高的存活先验上（0.565 vs 0.521），所以 lift 从 1.764 掉到 1.626。

作为对照，把 `flow_settling_log_ratio` 的思路搬到跨 query 侧（`mobility_settling_log_ratio`）在 global 下几乎失效：35 TP / 14 FP。这与"该量在全局阈值下坍塌"的既有结论一致。

### 组合头

预先声明的两个组合（由已挑选的 s0 头与 s9 头机械组合，无额外搜索）：

| head | mode | TP | FP | precision | recall | lift |
|---|---|---|---|---|---|---|
| s0 **OR** s9 | per_task | 368 | 86 | 0.811 | 0.652 | 1.408 |
| s0 **AND** s9 | per_task | 222 | 21 | 0.914 | 0.394 | 1.443 |
| s0 **OR** s9 | global | 200 | 27 | 0.881 | 0.355 | 1.730 |
| s0 **AND** s9 | global | 104 | **1** | **0.990** | 0.184 | 1.403 |

global 下 s0 AND s9 给出 104 TP / **1 FP**（precision 0.990），但它的报警很晚（平均先验 0.706），lift 只有 1.403，所以它买到的主要是"确认"而不是"提前量"。

### 步内量（探索性）

`token_differentiation`、`conditional_effective_rank`、`load_entropy`、`flow_speed` 在 s0/s3/s6/s9 四个探针步上的结果见 `results/step_alarm/external_detectors.csv`。要点：

- `load_entropy`（= 专家负载集中度）**只在 s9 上强**：per_task 370/93，而 s0 只有 9/1、s3 100/16、s6 107/23。专家负载这个信号几乎完全住在最后一步。
- `conditional_effective_rank` 也随步单调变强（per_task：s0 60/10 → s9 103/10）。
- `token_differentiation` 在 per_task 下反而是中间步更好（s3 74/5 precision 0.937、s6 43/1 precision 0.977），但报警量都很小。
- `flow_speed` 只有 s9 有量（per_task 285/62）。

**没有任何一个头在有实际报警量的前提下超过 global 基线的 lift 1.764。** 表里几个 lift 3–5 的条目（`flow_speed_s9` global 5/0 lift 4.823、`token_differentiation_s3` global 1/4 lift 5.509）报警数是个位数，属于噪声，不作为结论。

---

## 七、失败与负面结果（照实报告）

1. **没有找到步轴上的两块划分。** 步轴最佳两分块分数 0.036–0.100，层轴 4/4 划分是 0.089–0.245，弱 2–7 倍；步轴非对角平均相关 0.93–0.97。步轴是梯度。
2. **`mobility_settling_log_ratio` 在 global 下坍塌**：35 TP / 14 FP、recall 0.062。把 flow 内的 settling 思路搬到跨 query 侧没有救活它。
3. **episode 级 ICC 是个失败的诊断。** 我试图用"步 mobility 的方差有多少落在 episode 之间"来检验"噪声增益是否与状态相关"。结果是所有层/步的 ICC 中位数都 ≤0.12，大部分是 0 或负值，而且**不跨 run_id 复现**（L2 step 0：开发 0.122、外部 0.023）。因此第四节末尾的"增益假设"**没有被证实**，只是一个与数据不矛盾的解释。step-mobility 的方差几乎全部在 episode 内部，检测器读的是 episode 内的时间动态而不是 episode 级的偏移。
4. **`state_mobility_step` 完全没用。** state token 的路由在十个步上不变，所以它在步轴上的 late−early 差值在 40/40 个开发任务上精确为 0，方向一致率 22/40、20/39（即随机）。这本身是个干净的结论（步依赖性全在 action token），但作为一个"量"它是空的。
5. **`state_action_alignment` 与 `conditional_energy` 的 back−front 步对比不复现**（27/40、30/40），它们的步轴任务内相关高达 0.9996——这两个量在 query 层面几乎完全没有步结构（虽然均值随步在动）。
6. **没有超过 lift 基线。** 本工作在召回上明显赢，在"每个报警的信息量"上没有赢。

---

## 八、局限

- **多重比较。** 外部集只打开了一次，但那一次里打了 60 个头（28 个声明量 × 2 种阈值 + 4 个组合）。预先指定的**主**比较是十步 mobility 的**剖面**（不是选冠军）；60 个里的最优者是乐观偏置的，不作为调好的检测器主张。`mobility_step_range` 与 `mobility_settling_log_ratio` 也在看外部集之前声明。
- **query-0 识别的前提未被独立验证。** "同一种子在不同初始状态下产生相同噪声张量"这一点我只能从路由侧间接支持（state token 的噪声敏感度是 1e-4 量级，即基本不受噪声影响），不能从采样器代码验证。若该前提不成立，"纯观测"那一列会混入噪声，比值会被低估——方向不会反转，但幅度会变。
- **噪声地板的可比性。** 相邻 query 与 query-0 组内对比都是"两次独立噪声抽样"，所以尺度可比；但相邻 query 还多了一点观测变化，所以余量占比是一个偏保守的估计（地板可能略偏高，因为组内 episode 的观测完全相同而相邻 query 不是）。
- **两条轴的分块分数没有做长度归一化**（步轴 10 长、层轴 8 长），也分别是在层平均 / 步平均后的值上算的。数量级差 2–7 倍应当稳健，但精确比值不要当作可比统计量。
- **float16 源数据。** 用 fp16 量级的乘性扰动做过敏感性检查：`token_differentiation` 的扰动 RMS 是 6.8e-7，而它的跨 query 标准差是 1.0e-3，相差三个数量级，可忽略。
- **单一 checkpoint、两个 run_id、只有 LIBERO 四个 suite。** 步轴的方向性结论在 79 个任务上全部同向，但"十步"这个数是配置的一部分，换步数后的行为未测。
- **`hb_expert_ids`、`hb_selected_prob`、`as_probs`、`as_expert_ids` 仍未使用**（`hb_entropy` 只做过一次一致性抽查，未进入任何结果）。

---

## 九、产物

```
experiments/
  extract_flow_steps.py         逐步抽取（CPU，24 worker，81 s 全量）
  analyze_step_structure.py     步剖面 / 前后交叉 / 复现计数 / 两轴分块
  analyze_step_decomposition.py 噪声地板拆解 / 步增量 / episode ICC
  protocol.py                   既有检测协议的逐字副本
  detect_step_alarm.py          逐步检测器（开发挑选 + 外部一次）
  build_manifest.py
tests/
  test_anchors.py               step9 与 v4 缓存一致；协议复现两个已发表头
results/
  step_profiles/                稠密逐 query 剖面（4.9 GiB，.npy，按仓库惯例不入 git，81 s 可重建）
  step_structure/               步剖面、复现表、分块表、query-0 识别、噪声地板拆解
  step_alarm/                   开发网格、挑选、外部检测器、增量价值、报警索引
  manifest.json                 输入哈希、脚本链路、输出清单
```

复现：

```bash
python experiments/extract_flow_steps.py --workers 24
python experiments/analyze_step_structure.py
python experiments/analyze_step_decomposition.py
python experiments/detect_step_alarm.py
python tests/test_anchors.py
python experiments/build_manifest.py
```

---

## 十、下一步建议

1. **把 step_range 与 step-9 mobility 做联合头。** 二者在 global 下的误报只有 10 个是 step_range 独有的，而它多抓 41 个风险；一个在开发集上挑选的 OR/AND 组合值得单独做一次预注册评估（本次没做，因为我已经看过外部集，现在再挑就是事后选择）。
2. **验证"增益假设"。** 需要能控制 flow 噪声的在线实验：在同一 query 上重跑两次不同噪声，直接测 step-0 路由的响应幅度，再看这个幅度是否随 episode 状态变化。离线缓存做不到。
3. **把步轴放进层轴的联合表示。** 目前所有头都是"选一步 + 选一层"。第三节显示 L2-s0 与 L12-s9 相关只有 0.24，说明 (层, 步) 是一个二维的、远未被利用的表示；一个在 8×10 网格上的低秩读出可能比任何单格都好。
4. **检验 step 9 的特殊性是不是采样器假象。** s8→s9 的 flow_speed 暴涨可能来自最后一步的积分/裁剪细节而非功能分工。改变步数或求解器可以区分。
