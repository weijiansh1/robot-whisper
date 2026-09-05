# HiMoE-VLA 全动态 MoE 审计 v9

日期：2026-09-05

## 结论先行

这版已经按“使用更多、而且是动态的 MoE 信息”重做。v8 的 109 个候选轴扩展为
2,460 个动态轴，直接覆盖原始 router 的全部 8 层、10 个 flow step、11 个 token
和 32 个 expert，并加入 query 间 lag 1..8、二阶/三阶变化和 expert identity 的流量轨迹。

最重要的结果有三点：

1. **Static 的内部本质更清楚了**：不是简单 gate 变平，而是 late-layer state-token
   route 在多个 query 间锁死。`late/state/recurrence_lag4` 的 q-2 matched detection AUC
   为 main 0.985、grid 0.989。
2. **Loop 是多尺度的 unstable commitment**：flow 内部加速度、图统计总变差、expert
   load 迁移和 early-flow 的 L15 top-1 concentration 同时升高。但具体 layer/step/expert
   坐标在语料间并不稳定，完整样本八轴组合只有 main 0.774、grid 0.673。
3. **更多信息不等于更强的最终标量**：需要 8-query 历史的 loop 组合在 grid 达到
   0.700，但只覆盖 177/306 个事件；不能拿它直接击败覆盖 306 个事件的 v8 0.678。

因此 v9 的真正优化是把“读取的信息量”和“最终规则维度”分开：全量动态信息用于发现、
审计和机制定位，最终报警只保留跨语料稳定的证据。没有训练分类器，也没有制造新的
“成功把握”。

## 与 v8 的实质差别

| 信息维度 | v8 | v9 |
|---|---:|---:|
| 候选轴 | 109 | 2,460 |
| MoE 层 | 主要是 L12-L15 final-flow 图量 | 全部 8 层 |
| flow | 9 个全局 transition | 每层、state/action、每个 transition 的速度/加速度/转向/flux |
| token | final-flow 图摘要 | 11 个 token 各自的 path、accel、endpoint、efficiency |
| graph | 4 层 final-flow 的 13 类摘要 | 8 层 x 10 flow x 6 图统计及其轨迹导数 |
| expert identity | 主要是聚合 occupancy/support | 8 x 32 个 expert 的 endpoint 与 total variation，共 512 轴 |
| query history | 全局 recurrence lag 1..8 | 分层、state/action 的 recurrence、accel、jerk、turn 和图导数 |
| profile 构建 | 无训练 | 无训练、且不读取 outcome |

v9 并不是把 2,460 维全部平均成一个分数。那会让相关轴重复计票，也会把 loop/static
相反方向抵消。所有轴都被保存和扫描，但组合按机制分通道，并对 feature family 去重。

## 动态画像

原始单 query router 为：

```text
P[q, layer=8, flow=10, token=11, expert=32]
```

先使用平方根概率嵌入。相邻分布距离为 Hellinger distance：

```text
d(P_t, P_{t-1}) = ||sqrt(P_t) - sqrt(P_{t-1})||_2 / sqrt(2)
```

2,065 个 query 内轴包括：

| 家族 | 轴数 | 含义 |
|---|---:|---|
| flow speed/accel/turn/flux | 544 | denoising 路径的一阶、二阶、方向和概率质量迁移 |
| per-token path summaries | 352 | 每个 state/action token 的路程、曲率、端点和效率 |
| graph level | 480 | 每层每 flow 的 commitment、top mass、token affinity、MI、state-action affinity |
| graph endpoint/TV/curvature | 144 | token-expert 图在 flow 内如何变化 |
| expert endpoint/TV | 512 | 每个 expert 的负载方向和总迁移量 |
| robust aggregates | 33 | all/early/late layer group 摘要 |

395 个跨 query 轴包括：

| 家族 | 轴数 | 含义 |
|---|---:|---|
| layer/token-group recurrence | 128 | lag 1..8 的 route distance |
| query accel/jerk/turn | 48 | route trajectory 的二阶、三阶和转向 |
| graph d1/d2/d3 | 144 | 六类图统计跨 query 的有符号导数 |
| robust aggregates | 75 | all/early/late 的分组摘要 |

跨 query 特征严格只使用当前及历史前缀。测试已验证：修改未来 query 会改变未来行，
但当前和过去的特征逐元素不变。

## 数据与评估

| corpus | tasks | episodes | query rows | 用途 |
|---|---:|---:|---:|---|
| main16x32 | 5 | 2,560 | 51,308 | 方向和轴选择 |
| grid50x8 | 40 | 16,000 | 253,722 | 跨语料压力测试 |
| 合计 | 45 | 18,560 | 305,030 | 无标签动态 profile |

正例为事件 episode 的固定 lead query；负例为相同 task、scene、绝对 query 且没有
loop/static 的 episode。profile 构建完全不加载 outcome；main 标签只在评估阶段确定方向
和 discovery 轴。grid 在此前 v8 已被查看过，因此是跨语料确认，不再宣称 pristine holdout。

双 GPU 构建使用 `CUDA_VISIBLE_DEVICES=6,7`。两张 NVIDIA H20-3e 均实际计算，峰值显存
各 733.94 MiB。这是流式特征计算，不是训练；为了“占满卡”而复制 140 GB 张量没有意义。
输出 profile 共 1,525,119,430 bytes，float16 最大抽样量化误差为 `9.50e-4`。

## Loop 结果

### 组合结果

| score | main q-2 AUC | grid q-2 AUC | grid events/pairs |
|---|---:|---:|---:|
| 预声明四机制组合 | 0.620 | 0.668 | 306 / 1,113 |
| discovery 全覆盖八家族 | 0.774 | 0.673 | 306 / 1,113 |
| discovery 长历史八家族 | 0.774 | 0.700 | 177 / 546 |

长历史组合覆盖下降，是因为其中的 lag-7 recurrence 在早发 loop 上尚不可计算。最终结果
将它作为附加分析，而不是主结果。

grid 上全覆盖 discovery score 随 lead 的 AUC 为：

| lead | -8 | -6 | -4 | -2 | 0 |
|---:|---:|---:|---:|---:|---:|
| AUC | 0.566 | 0.553 | 0.609 | 0.673 | 0.763 |

它越接近 onset 越强，但 q-4 以前仍很弱。

### 内部本质

最稳定的单轴之一是 `graph_level|L15|f1|top1_mass` 上升：main 0.663、grid 0.674。
这说明 loop 前并非 router 一直犹豫；后层在很早的 flow step 反而更快集中。

与此同时：

- action route 的 flow acceleration 上升；
- top-4 mass 的 graph total variation 上升；
- soft probability flux 和 expert-load TV 上升；
- route 路程变长、转向增多。

联合解释仍是：

```text
局部选择更尖 + 完整 denoising 路径更不稳定 = unstable commitment
```

但精确坐标并不稳。例如 main 最强的 `flow_accel|L4|action|t6` 为 0.696，grid 只有
0.551；grid 最强的全局 expert-load TV 为 0.692，main 只有 0.592。扫描 2,460 轴使这种
选择不稳定性暴露出来，而不是被一个漂亮的 best-axis 数字掩盖。

与 v8 比较：v8 discovery composite 的 grid q-2 AUC 为 0.654，v9 全覆盖组合为 0.673；
但 v8 单轴 route acceleration 为 0.678，仍不低于 v9 全覆盖组合。结论只能是分辨率提高、
机制更完整，不能声称 dense score 普遍胜出。

## Static 结果

| score/axis | main q-2 AUC | grid q-2 AUC | grid events/pairs |
|---|---:|---:|---:|
| 预声明动态组合 | 0.973 | 0.955 | 100 / 267 |
| discovery 八个去重家族 | 0.978 | 0.951 | 100 / 267 |
| late state recurrence lag-4 下降 | 0.985 | 0.989 | 100 / 267 |

加入许多“不同”轴反而不如最干净的 state recurrence。原因不是其他 MoE 信息没读，
而是 recurrence、acceleration、jerk 和 graph derivative 都是同一个 route-freeze 过程的
不同投影。强迫它们等权只会稀释最稳定方向。

grid 预声明 score 随 lead 的 AUC 为 0.781、0.938、0.941、0.955、0.910
（lead -8/-6/-4/-2/0）。`late/state/lag4` 比 v8 全局 lag-4 的 0.985 略高，说明将 state
token 与 action token、late layer 与 early layer 分开确实提供了额外机制分辨率。

必须保留时间定义边界：static onset 需要 7 个连续、相互重叠的 width-2 静止窗口。
q-2 时静止通常已经持续，不是“机器人仍正常运动时提前两步预测未来命运”。这里是
route lock-in 对正在形成的 static 的确认。

## 无训练风险规则

使用 healthy episode maximum 做 5% conformal-style calibration。grid 中每个 suite
按任务名奇偶拆成 calibration/test，阈值只在 calibration tasks 上确定。

| 通道 | threshold | test healthy N | test episode FPR | 95% CI | q-2 recall |
|---|---:|---:|---:|---:|---:|
| predeclared loop | 0.76076 | 7,525 | 6.55% | [6.00%, 7.13%] | 8.39% |
| predeclared static | 0.90530 | 7,525 | 3.22% | [2.83%, 3.64%] | 98.31% |

同任务、repeat 0-3 校准 / 4-7 测试时，全覆盖 loop discovery 为 4.82% FPR、10.51%
q-2 recall；static 预声明规则为 4.82% FPR、96.51% recall。

与 v8 的 unseen-task LTT 结果相比：

- loop：v8 为 5.48% FPR / 9.63% recall，v9 预声明为 6.55% / 8.39%，没有改善；
- static：v8 为 0.96% FPR / 95.76% recall，v9 为 3.22% / 98.31%，是灵敏度与误报的
  可解释交换，而不是无条件胜出。

因此只推荐 static 通道作为审计型 guard。loop 分数适合排序、触发日志或低成本复查，
不适合直接硬停机器人。

## 在线接口

`dynamic.features.compute_dynamic_history` 接收最近的 router history，计算严格因果的
完整画像；保留当前加过去 8 个 query 即可覆盖所有定义轴。

`dynamic.scoring.DynamicScoreReference` 加载
`results/dynamic_evaluation/score_reference.npz`，使用封存的 main 经验分位参考、轴定义
和方向重算六个 score。结果级测试已验证它与离线保存分数逐元素一致。

这两个接口都不需要 task ID、失败标签或训练权重。部署到新 domain 时，应重新校准阈值，
但不必训练 feature extractor。

## 不能越过的 Assurance 边界

v9 只建立：

```text
Routing evidence -> episode-level risk alarm
```

它没有建立：

```text
P(success / loop / static | current physical snapshot, intervention)
```

AUC 0.989 也不是“98.9% 成功把握”，阈值 0.905 也不是 90.5% 概率。真正的 outcome
assurance 必须来自同一 simulator/controller snapshot 的重复 continuation counts。现有真实
概率仍只应引用 v8 的 `results/route_8/assurance_tensor.csv`；v8 已证明将 MoE 特征拟合到
committor 后不能跨语料可靠泛化，v9 没有再训练一个新概率头掩盖这个负结果。

## 最终建议

1. 研究和日志层保留 2,460 维完整动态 profile，尤其是 layer/flow/token/expert identity
   分解；这是本版相对 v8 的主要增量。
2. Static 在线通道先用预声明的 lag-1 recurrence、query acceleration、jerk 和 graph d1
   低分位组合，并在目标域重做 episode-level threshold calibration。
3. Loop 保留两个输出：全覆盖 evidence 与 history-available evidence。绝不能在 lag-7
   不可用时静默丢弃早发事件，也不能把较高但低覆盖的 0.700 当主结果。
4. 若目标仍是“两 chunk 内不进入 loop 的真实把握”，下一步不是增加第 2,461 个轴，
   而是在多个 q-2 snapshot 上做 paired repeated forks，直接估计 committor 和区间。

## 可复现产物

- `results/final_summary.json`：关键结论和 v8 同口径对比；
- `results/dynamic_profiles/build_summary.json`：45 个 profile 的来源、大小与 SHA-256；
- `results/dynamic_evaluation/all_axis_matched_auc.csv`：2,460 轴 x 2 语料 x 2 事件 x 5 leads；
- `results/dynamic_evaluation/composite_matched_auc.csv`：预声明、全覆盖、长历史组合；
- `results/dynamic_evaluation/score_reference.npz`：运行时分位参考；
- `results/dynamic_risk/summary.json`：任务拆分和风险规则；
- `results/sealed_manifest.json`：全部代码、报告、profiles 和结果哈希。
