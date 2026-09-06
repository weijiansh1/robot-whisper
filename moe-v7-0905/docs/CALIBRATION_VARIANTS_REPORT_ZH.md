# 判定线重构（实验 B）：三条去语料路径全部失败，及其原因

日期：2026-09-05
前置：`LOSO_VALIDATION_REPORT_ZH.md`、`LOTO_VALIDATION_REPORT_ZH.md`

## 1. 结论

**实验 B 的原定方向是错的，这里给出证伪与原因。**

LOSO 报告 §8 曾建议把 freeze 与 acceleration 的切点换成"前缀内秩统计量"，理由是它们随语料
horizon 配方摆动 44.5 pp。本实验实现了三条互不相同的去语料路径，在同一套 LOSO 协议、同一批
分数流、同一个 640 点选择规程下比较。**三条全部在四折中不可行**——没有任何候选同时满足
预注册的 precision ≥ 0.85、timely FPR ≤ 0.005、recall ≥ 0.70。

失败原因三者不同，且互斥：

| 变体 | FPR 下界 | recall 上限 | 合格候选数（goal/long/object/spatial） | 失败于 |
|---|---|---|---|---|
| `pooled_peak`（v7） | 0.01–0.02% | 0.82–0.90 | 399 / 240 / 472 / 327 | — |
| `query_indexed` | 1.18–2.10% | 0.82–0.93 | 0 / 0 / 0 / 0 | FPR |
| `self_normalized` | 1.11–1.46%（long 0.11%） | 0.96–0.98 | 0 / 80 / 0 / 0 | FPR |
| `query_normalized_peak` | 0.00–0.21% | 0.37–0.54 | 160 / 0 / 160 / 76 | recall |

可以拿到 FPR 控制（前两行），也可以拿到语料无关性（中间两行），但拿不到两者兼备。

**v7 的 pooled trajectory-peak 分位数不是一个可以随手替换的实现细节。它同时在做三件事**，
每一条替代路径都恰好打断其中一件。

## 2. 三件被同时承担的工作

### 2.1 多重检验校正（`query_indexed` 打断的）

`query_indexed` 把单个标量换成按 query 索引的向量：在第 q 步，把当前分数与"语料中所有活到
第 q 步的轨迹的前缀最大值分布"的 97.5 分位比。看起来更公平，实际 FPR 下界从 0.01% 涨到
1.18–2.10%，**约 100 倍**，网格最保守的一端也压不下来。

原因是：一条轨迹有 T 次穿越机会。每个 query 各自留 2.5% 的越界概率，累积到 20-25 个 query
就是几乎必然报警。v7 的"整条轨迹峰值的 97.5 分位"隐含地把"至少穿越一次"的概率控制在 2.5%，
**它一直在做逐轨迹的多重检验校正**，只是没写在协议里。按 query 拆开就把校正拆没了。

### 2.2 绝对尺度参照（`self_normalized` 打断的）

`self_normalized` 完全不用语料：判定线是该 rollout 自身前缀的中位数加 c 倍稳健离差。它的
recall 上限最高（0.96–0.98），但 FPR 下界同样在 1.1–1.5%。

问题是前缀太短。本 cohort 的 median rollout length 是 goal 10、spatial 11、object 14、
long 25。任何纯前缀内的秩检验，在第 q 步能给出的最小 p 值就是 `1/q`，即 0.04–0.10 量级；
而要在 ~25 个 query 上守住 0.5% 的逐轨迹 FPR，需要每个 query 约 2×10⁻⁴ 的水平。**差 2 到 3
个数量级。** 用 median+MAD 形式不受这个硬下界约束，但换成了等价的困难：MAD 由 ≤ q 个值估出，
分布尾部无法外推。

这条是信息量层面的限制，不是实现问题：**在这个 horizon 量级下，rollout 自身的前缀不足以支撑
所需强度的证据，外部参照语料不是可选项。**

### 2.3 非存活条件的比较基准（`query_normalized_peak` 打断的）

`query_normalized_peak` 是三条里设计上最讲究的：先用语料的逐 query 中位数与 IQR 把分数化成
"每个 query 的语料单位"（这一步去掉配方倾斜），再对归一化后的流套用 v7 的整条轨迹峰值分位数
（这一步保留多重检验校正）。FPR 确实守住了（0.00–0.21%），四折中三折有合格候选。

但 recall 上限塌到 0.37–0.54，long 折是 0.370，无论如何达不到 0.70。

原因看语料的逐 query 存活数：

```
q=6: 16,000   q=12: 9,542   q=20: 3,976   q=25: 1,549   q=30: 602
```

到 q=20 只剩 25% 的轨迹，到 q=30 只剩 3.8%。**而"在一个 median length 11 的 suite 里还没结束"
本身就是最强的风险指示。** 按 query 归一化等于对"活到第 q 步"取条件，把要检测的信号一并减掉了。
v7 的 pooled 形式之所以有效，恰恰因为它把当前轨迹和**全部**轨迹比——包括那些早就成功结束的
短轨迹。

## 3. 实验设置

四个变体走完全相同的管线：同样的分数流（`freeze` / `acceleration_persistent` /
`periodicity_persistent`）、同样的 4 折 LOSO、同样的 640 点网格、同样的预注册约束与排序
（`select_operating_point.choose` 的 primary 档），只有判定线的形状不同。v7"在平滑流上标定、
在 persistent 流上触发"的不对称性在四个变体中一致保留，因此比较隔离的是切点本身。

网格：`pooled_peak` / `query_indexed` / `query_normalized_peak` 用 v7 原有的
8 × 8 × 10 分位数网格；`self_normalized` 用同样规模的 c 值网格
（freeze/acceleration 各 8 个 1.0–5.0，periodicity 10 个 0.5–5.0）。

`self_normalized` 需要至少 8 个前缀点才开始判定，因此最早报警在 q8。
`query_indexed` 与 `query_normalized_peak` 在存活参考少于 32 条的 query 上不产生判定
（52 个 query 中有 46 个可用），对极长轨迹是覆盖损失而非外推。

一处值得记录的性质：`self_normalized` 的判定线对分数的仿射变换不变，因此 z_R 分母里那个语料
常数 `periodicity_scale` 会在比较中约掉。这个变体确实是**完全无语料**的，它的失败不能归因于
残留的语料依赖。

## 4. 对 v7 与后续工作的结论

1. **LOSO 报告 §8 的建议作废。** 那里说"B 应把 freeze 与 acceleration 的切点换成前缀内秩
   统计量"，本实验证明这条路会把 FPR 抬高约两个数量级。以本报告为准。

2. **v7 的语料配方依赖是真实的，但不可通过重塑判定线消除。** 它是"用整条轨迹峰值做逐轨迹
   FPR 控制"这一正确选择的附带代价：峰值分布必然受语料 horizon 构成影响。

3. **正确的应对是协议层面而非方法层面**：把校准语料的 horizon 构成当作一个需要声明和控制的
   量，而不是当作可以忽略的实现细节。具体建议——
   - 在 profile 里显式记录校准语料的 suite/长度构成；
   - 部署到新任务分布时，报告 LOSO 意义下的敏感度区间（本轮为 recall −7.60 pp、
     micro precision −6.33 pp），而不是只报全语料点估计；
   - 若新分布的 horizon 量级与校准语料差异大，应重新校准而非直接迁移。

4. **仍然开放的方向**（本轮未做）：在保留整条轨迹峰值形式的前提下，对校准语料做**长度分层
   重加权**——按目标分布的 horizon 构成给参考轨迹加权，再算峰值分位数。这不改变判定线的
   形状，只改变语料的有效构成，因此不会打断 §2 的三件工作中的任何一件。它需要知道目标分布的
   长度构成，属于"部署前已知"而非"运行时已知"，与在线因果性不冲突。

## 5. 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python experiments/evaluate_calibration_variants.py
```

约 13 秒，纯 CPU。产物在 `results/calibration_variants/`：
`variant_summary.csv`、`variant_fold_metrics.csv`、`variant_selection.json`、
`candidates_<variant>_<fold>.csv`（16 张 640 行的候选表）、`sealed_first_alarms.npz`、
`variants_manifest.json`。

不可行的折不会放宽约束重试——`variant_selection.json` 中 `feasible: false` 的记录连同完整
候选表一并保留，这是结论本身。
