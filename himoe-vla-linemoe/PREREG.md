# 预注册：失败轨迹上的 chunk，其 flow 去噪还直不直

> 2026-09-03 冻结，采集启动前写定。特征定义**先于任何结局标签**存在 —— 它们就是
> `analysis_flow_straightness/run.py` 在两批无标签语料上已经算过的那套，一字未改。
> 标签只在评估阶段进入。

## 问题

已确立（`analysis_flow_straightness/report.zh.md`）：健康 rollout 上，一次推理内的 flow 轨迹
路径/弦长 = 1.002、相邻速度 cos = 1.000、动作在第 1 轮外推就落在最终值 7.3%/10.1% 内、
夹爪符号全程不翻。

本实验问：**卡住 / 失败的 chunk 上，这条线还直吗？** 三个互斥的可能：

- **H_straight**：一样直。失败与"模型内部犹豫"无关，它只是把一个自信的错动作直着算出来。
- **H_bend**：变弯（路径/弦长上升、cos 下降、猜测漂移增大）。则存在一个**动作空间**的
  train-free 前兆，且它独立于已有的路由信号。
- **H_flip**：直线性不变，但终点本身在候选间/相邻 chunk 间乱跳（决定不稳定，而非过程不稳定）。

这同时补上 trap 报告点名的缺口："当前 capture 没有每个 flow step 的中间 action trajectory，
不能在同一 A/B 上计算真正的 flow-action acceleration 基线。"本轮采完即可计算。

## 采集（冻结）

- 任务：`libero_10 / t08 KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` ——
  本项目唯一有稳定结局方差的捕获任务（历史成功率 ~57.8%）。
- 网格：**16 初态 {0,3,6,…,45} × 8 flow-noise seed {7000..7007} = 128 集**。
  与 2026-08-22 截断先导同网格，初态即组内配对的分组变量。
- `K=1`（on-policy，不采兄弟候选）、`replan=10`、`settle=10`、`max_steps=520`。
- 服务端 `serve_flow_trace.py`，**`--libero-wrist-layout checkpoint-right`** ——
  与既有两批 flow 语料一致，故可与本文 §1 基线同表；
  **绝不可与 right-50x8（paper-right）同表**。
- GPU 6（与既有 4 个空转 online-server 共卡，不动它们）。
- 每个 query 落盘：11 点完整 flow 轨迹、逐 Euler 步 32 路 full softmax、执行的动作 chunk、
  8 维本体状态。每集落盘：success / action_steps / control_steps / query 区间
  （`rollout_flow_lead.py` 本轮新增 `episodes.json`，纯增量）。

## 度量（冻结，逐 chunk）

全部只用前 7 个活维；速度由相邻两点精确还原 `v_m=(x_m−x_{m+1})/dt`。

| 名称 | 定义 | 方向读法 |
|---|---|---|
| `path_over_chord` | Σ‖Δx‖ / ‖x_0−x_10‖ | 弯曲度 |
| `cos_min` | min_m cos(v_m, v_{m+1}) | 最大转折 |
| `cos_chord_mean` | mean_m cos(v_m, 首末弦) | 整体对齐 |
| `xhat_travel` | ‖x̂(0)−x_10‖ / ‖x_10‖ | 猜测总位移 |
| `xhat_drift_mean` | mean_m ‖x̂(m+1)−x̂(m)‖ / ‖x_10‖ | 逐轮改主意幅度 |
| `xhat_drift_late` | m=5..9 的同上 | 末段是否还在改 |
| `speed_ratio` | ‖v_9‖/‖v_0‖ | 加/减速 |
| `grip_flip` | 去噪过程中夹爪符号翻转次数 | 离散决定动摇 |

对照（同样 train-free，用于证明新量不是旧量的改写）：既有 `route_mobility`（相邻 chunk
Hellinger W8）、`action_change`、`action_magnitude`。

## 统计（冻结）

- 主检验：固定控制步 `t=20/25/30`，**组内（初态）成功–失败配对 AUC**，跨组池化一律不用。
- 置换：2000 次组内标签置换，maxT 同时校正 `8 signals × 3 时点 = 24` 个单元，seed 20260903。
- 残差：对同组同 t 的 `route_mobility` 做组内秩残差，回答"是否只是旧变化率的改写"。
- Onset 对齐：用 `query_records.json` 的 8 维状态做 query 分辨率 loop/static 代理
  （trap 工作已在稠密真值上验证过的同一代理），lead ∈ {−4,−2,0} 与同初态同绝对 query 的
  no-event 轨迹比较。
- 判据：**跨"固定时点"与"onset 对齐"两种读法同向、且 d9/route_mobility 残差后仍存活**，
  才算 H_bend 成立；只在原始量显著记为候选，不入结论。
- n=128 集、单任务单 checkpoint，只能排除大效应。这是先导，不是确认。

## 不做什么

不训练分类器、不拟合 PCA/k-means/probe、不从评估集调阈值。不与 paper-right 语料合表。
不因为结果好看就加时点或加信号 —— 上表 8 个信号 × 3 时点是全部检验族。
