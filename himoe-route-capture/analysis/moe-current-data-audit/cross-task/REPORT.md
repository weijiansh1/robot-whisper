# 五任务 16x32 MoE 跨任务审计

## 结论

当前数据不支持 route consensus selector、route 相对 action/noise/hidden 的条件增量，
也不支持用 route dispersion 预测哪里值得多采样。可保留的弱信号是单个
Libero-Long 任务上 step-0 实际专家输出的抵消/离散结构；它不是 raw router，
且没有通过 0.05 的重训练置换门槛。

## 数据规模

- 5 个任务、3 个 suite，每任务 16 个 init state、每 state 32 个共同 seed：
  共 80 个独立 state pool、2,560 条 rollout。
- 同一 K32 被固定拆成 4 个 K8，因此 320 个 K8 不是 320 个独立状态。
- 五任务成功率依次为 `1.000 / 0.918 / 0.578 / 0.977 / 0.928`。
  一个任务完全饱和，真正具有广泛成败变化的主要是 Libero-Long。

## Route consensus

legacy K8 严格 task + seed 留出中，route central、route head、route+noise head
相对精确随机的 task-macro 分别为 `-0.82pp / -1.13pp / -1.13pp`。主要留出任务
上 early route central 是 `+1.95pp`，但 95% CI 为 `[-1.37,+5.27]pp`；
route+noise head 为 `-1.17pp`。

本轮实际运行的 K32 medoid 结果：

| selector | task-macro delta | state-bootstrap 95% CI | seed concentration |
|---|---:|---:|---:|
| route probability d0 | +0.74pp | [-3.75,+5.08]pp | 91.3% 选同一 seed |
| route probability d0-2 | +0.74pp | [-3.71,+5.16]pp | 97.5% 选 seed 1013 |
| route probability full | +0.74pp | [-3.79,+5.12]pp | 97.5% 选 seed 1013 |
| expert-ID d0 | -0.51pp | [-4.84,+3.48]pp | 11 个不同 seed |
| expert-ID d0-2 | +0.74pp | [-3.05,+4.26]pp | 9 个不同 seed |
| expert-ID full | -1.76pp | [-6.80,+3.09]pp | 7 个不同 seed |

d0、d0-2、full 得到同样的成功率不是“跨去噪稳定”的证据，因为 probability
medoid 几乎总在恢复一个全局 seed 模板。

## 条件增量

early-action-head 的 seed-disjoint action-geometry 结果中，hidden、layer-resolved
route、noise 的 Spearman 分别为 `0.542 / 0.446 / 0.013`。这说明 route 能读出一部分
动作几何，但 hidden 更强；目标不是 value，且该分析不是 task-heldout。

本轮新增固定线性 probe 同时留出完整 task 和测试 K8 的八个 seed：

| feature | K8 AUC | success delta vs random |
|---|---:|---:|
| noise | 0.476 | -0.20pp |
| action | 0.561 | +0.74pp |
| hidden | 0.479 | -0.20pp |
| route | 0.478 | -0.82pp |
| action + noise | 0.533 | +0.43pp |
| action + noise + route | 0.510 | +0.12pp |
| action + noise + hidden | 0.449 | -0.20pp |
| action + noise + hidden + route | 0.440 | -0.82pp |

route 加到 action+noise 后的配对增量为 `-0.31pp`，95% CI
`[-2.81,+1.88]pp`；再控制 hidden 后为 `-0.63pp`，CI `[-3.13,+1.88]pp`。
这只否定当前固定 PCA+ridge readout 的可复用增量，不能证明所有非线性信息均不存在。

## Dispersion 与 opportunity

K32 route-probability dispersion 对 opportunity 的有限任务内 Spearman macro：
d0 `0.052`、d0-2 `0.089`、full `0.025`。task-heldout 一维校准的 MAE 均约
`0.188`，差于只用训练任务 opportunity 均值的 `0.161`。当前 route dispersion
不能作为 adaptive-K 预算信号。

## 其他反证

- oracle gate 的 action centrality、future route change、late route centrality
  增量分别为 `-0.24pp / +0.93pp / -1.42pp`，均未过 `+3pp` 门槛。
- 单任务局部进展 route 相关从同 snapshot 拟合 `rho=0.112` 降到留出 snapshot
  `rho=0.010, p=0.434`。
- 固定首 route、改变共同未来噪声的 route-outcome MI 为 `0.0466 bits, p=0.921`。
- 单任务 expert functional scalar 的 AUC `0.587`、选择增量 `+7.81pp`，但重训练
  置换 `p=0.059/0.057`；这是边界性线索，不是跨任务 route 结果。

## 解释边界

每个 candidate 只有一次完整 episode outcome；seed 同时决定首 query 和之后全部
重规划噪声。因而这里是 episode-start shadow association，不是第一个 action chunk
的因果价值。所有 selector 还需要先算完 K8/K32，并未产生在线 compute gain。

下一步需要同 snapshot 的真实 action chunk 执行、物理轨迹和重复 CRN continuation，
并在非饱和任务上按 snapshot/task bootstrap；在这些数据到位前，不应继续把 route
中心度、expert ID 或 route dispersion 写成控制抽象或 selector。
