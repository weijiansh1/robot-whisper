# MoE 概率读出

从 `../VLA_MUI_HUB` 的真实 rollout 估计每个执行前 chunk 的预算内自然成功概率。
中文实验结果见 [results/REPORT.zh.md](results/REPORT.zh.md)，图见
[results/probability_audit.png](results/probability_audit.png)。

**预算与任务先验修正：**原版的“仅预算”实际是 suite 分模加执行时钟；原 MoE 模型也包含这些信息。
它不能直接代表纯 MoE 的能力。新审计见
[results_no_budget/REPORT.zh.md](results_no_budget/REPORT.zh.md)，使用固定局部窗口、共享模型，
并增加同任务同 q 及未见任务评估。

**固定窗口物理进展实验：**新实验预测未来 5 个 chunk（50 个动作）内是否出现物理里程碑，
结果见 [results_progress/REPORT.zh.md](results_progress/REPORT.zh.md)，
标签与评估规则见 [PROGRESS_PROTOCOL.md](PROGRESS_PROTOCOL.md)。
它使用成功和失败轨迹的模拟器状态构造离线标签，在线仍只输入最近 8 个 chunk 的 MoE。
物理里程碑不是独立的 trap/脱困真值。

```bash
bash extract_progress.sh --workers 4
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u -m probability.run_progress --threads 4
```

提取脚本自动使用数据采集时的 LIBERO/robosuite Python 3.8 环境，仅恢复检查点，不执行新动作。
完整提取覆盖两批自然采集的 32,000 集。后续拟合使用本目录的常规 Python 环境。
重绘报告可运行 `python -m probability.run_progress --render-only`。

```python
import joblib
from probability.progress_model import MoEProgressMonitor

monitor = MoEProgressMonitor(joblib.load("results_progress/models/moe_progress_shared.joblib"))
result = monitor.update(hb_router_probs)
# result["progress_probability"]: 未来 5 个 chunk 内出现物理里程碑的概率。
```

每集新建一个 monitor；前 7 次返回未就绪。接口没有预算或物理状态参数，但模型的验证范围
要求有完整的未来观察窗口，原执行上限附近的使用尚未验证。当前已采样的 action chunk
必须保留。`natural_escape_probability` 仍为 `None`。
`results_progress/` 保存物理测量、标签分量、逐点预测、分任务与分阶段评估、聚类区间和审计。

## VLAConf 式分数校准

该实验固定 MoE 特征和成功支持度分数，只用终局标签学习两个 sigmoid 校准参数。
实现和边界见 [VLACONF_PROTOCOL.md](VLACONF_PROTOCOL.md)，结果见
[results_vlaconf/REPORT.zh.md](results_vlaconf/REPORT.zh.md)。
它用成功训练轨迹的近邻距离替代论文的 CFN 网络，是借鉴其分数与概率分离的思路。

局部主方法在主测试上的 AUROC 为 0.7435、Brier 为 0.07740；原无预算监督模型
在相同点上为 0.9092 / 0.05833。再匹配任务、query 和物理阶段后，局部方法 AUROC
为 0.4955（95% CI 0.3968–0.5952），尚不支持稳定的局部区分能力。
历史最大值的总体 AUROC 更高，但它的成功概率不能回升。

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m probability.run_scalar_confidence --threads 4
```

```python
import joblib
from probability.scalar_confidence import MoEScalarMonitor

monitor = MoEScalarMonitor(joblib.load("results_vlaconf/models/shared.joblib"))
result = monitor.update(hb_router_probs)
# result["success_probability"]: 最近 4 个支持度分数均值的校准概率。
```

主接口没有任务、suite、步数或预算参数。单个分数使用最近 8 个 query 的 18 维 MoE
特征；4 分数窗口最多覆盖 11 个 query。每集新建 monitor，前 7 次返回未就绪。
`probabilities` 同时返回当前值、局部均值、历史均值和历史最大值的对照结果。
历史最大风险的校准成功概率不能回升，其他聚合的概率回升也不等于已验证物理脱困。

成功参考库的建立仍使用训练集成功标签；校准仍使用独立集合的成败标签。
它不训练新神经网络，但不是完全不使用标签的方法，近邻查询也有存储和推理开销。
目标是原始截止条件下的终局成功，`natural_escape_probability` 仍为 `None`。
所有新结果独立保存；重绘可运行 `python -m probability.run_scalar_confidence --render-only`。

## 复现

在本目录执行，使用已安装 NumPy、pandas、scikit-learn、Zarr 的 Python 环境：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest -q
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u -m probability.run --threads 4
```

仅用既有测量值重新生成图表与报告：`python -m probability.run --render-only`。

复现无显式预算/时钟的追加审计（读取已有、经过哈希校验的实验缓存）：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u -m probability.audit_budget --threads 4
```

依赖列在 `requirements.txt`；实际运行版本保存在 `results/summary.json`。
脚本会读取源数据并校验其哈希，重建本目录结果。原始 Hub 数据始终只读。
完整复现需要 Hub 的两个 50x8 自然采集批次、原始 `server/routes.zarr`、
`client/summaries.json`、策略元数据，以及已有的 `moe-history-only/results/features`
缓存和 manifest。报警器直接复用 `../VLA_MUI_HUB/moe-history-only/monitor.py`。

## 无预算在线读出

共享模型只接收最近 8 个 query 的路由，不传任务 ID、suite、预算或已执行步数：

```python
import joblib
from probability.pure_moe import MoEWindowMonitor

bundle = joblib.load("results_no_budget/models/moe_window_shared.joblib")
monitor = MoEWindowMonitor(bundle)
result = monitor.update(hb_router_probs)
```

每条 rollout 新建一个 monitor。前 7 次 `update` 返回 `ready=False`、概率 `None`，
以后只用固定长度的路由窗口。移除了初始基线、报警持续时间和全历史计数。
该模型估计的是原语料任务、策略与截止条件混合分布下的成功概率；不能对任意指定的
剩余预算给出已验证的 `V(x,b)`。MoE 本身仍可能携带任务或阶段信息，输入契约不保证
内部表征与时间统计独立。

新产物保存在 `results_no_budget/`，包含 `metrics.csv`、`within_task_query.csv`、
`cluster_intervals.csv`、`task_folds.json`、逐 query 预测、原始路由抽查和共享在线模型。
未见任务预测来自 5 折模型，每折训练和校准均排除测试任务；共享在线模型则使用全部训练任务。

## 原预算模型

每个 suite 对应一个固定策略和预算的独立模型。每条 rollout 新建一次 monitor：

```python
import joblib
from probability.model import SuccessMonitor

bundle = joblib.load("results/models/libero_long.joblib")
monitor = SuccessMonitor(bundle, checkpoint_sha256=policy_checkpoint_sha256)

# 在当前推理完成后、执行其 action chunk 前调用。
# hb_router_probs 为 [8, 10, 11, 32]，或最后去噪步的 action 路由 [8, 10, 32]。
result = monitor.update(hb_router_probs)
probability = result["success_probability"]
```

`policy_checkpoint_sha256` 必须来自实际运行策略。调用方也须保持 bundle 中记录的
归一化、采样方式、10 步重规划、环境配置和最大预算；API 只强制核对 checkpoint。
概率条件包含已采样的当前 chunk，不能在读出后重抽当前动作再把它当成同一估计对象。
流式 API 接受原始概率，不能将 `expert_ids` 或 client NPZ 中的零占位路由传入。

`alarm_now` 是当前冻结规则是否成立，`ever_alarm` 是历史上是否报过警；
二者都不是 `Z_t` 真值。`trap_probability` 和 `natural_escape_probability`
返回 `None`，因为现有语料没有可用的独立 trap/脱困标签。
本版实现用户方案中“先做成功概率读出头”的路径。

终止后使用 `SuccessMonitor.terminal_probability(...)`：成功为 1，已知不可逆失败
或预算耗尽且未成功为 0。非终止状态由 MoE 前缀读出。
当前模型只在原始采集预算上接受验证，不支持任意延长/缩短预算的反事实推断。

## 结果文件

| 文件 | 内容 |
|---|---|
| `results/REPORT.zh.md` | 实验结论、方法、证据范围与限制 |
| `results/episodes.csv` | 32,000 集来源、完整终局标签、初始状态分组与首次报警 |
| `results/alarm_success.csv` | 每集一次的报警后成功率、Wilson 与聚类区间 |
| `results/alarm_rule_sensitivity.csv` | 全部现有冻结规则的报警后成功率，不按结果筛选规则 |
| `results/alarm_success_half_k4.csv` | 既有 half-k4 探索性规则的聚类区间，单独报告 |
| `results/metrics.csv` | 主测试、新噪声次测试、校准集；整体、首次报警与固定 q 指标 |
| `results/calibration.csv` | 可靠性曲线的概率 bin、样本数、预测均值与实际比例 |
| `results/cluster_intervals.csv` | MoE 与预算基线的聚类置信区间和配对损失差 |
| `results/predictions.csv.gz` | 每个 query 的原始/校准概率；与 episodes 按 episode_row 关联 |
| `results/first_alarm_predictions.csv` | 每条报警轨迹首次报警点的概率 |
| `results/matched_controls.csv` | 同任务、同批次、同 q 的未报警风险集对照 |
| `results/branch_candidates.csv` | 不依赖终局选择的报警与匹配对照检查点 |
| `results/branch_candidates_half_k4.csv` | half-k4 规则的补充采样；同 checkpoint_id 必须归为同组 |
| `results/models/*.joblib` | 4 个 suite 的成功概率模型与预算基线 |
| `results/data_audit.json` | 来源 SHA-256、策略配置、分组与排除记录 |
| `results/raw_verification.json` | 每个源 run 一集原始路由的流式/离线一致性检查 |
| `results/artifact_manifest.json` | 实验代码、数据产物与模型 SHA-256 |

训练采用实际 query 分布，不平衡重采样、不做 class_weight，不使用终局长度加权。
长轨迹的多个 chunk 在统计上有依赖，因此误差区间在任务内以初始状态为聚类单元。
零成功或全成功时经验 bootstrap 退化，聚类区间记为未定义；Wilson 区间仅作独立样本参考。
每集等权损失仅是另一种评估分布的诊断，不与实际 query 分布下的条件概率混称。

## 分支续跑契约

`branch_candidates.csv` 是待重建的检查点清单，不是已采集的分支。
同 q 的不同 rollout 不能直接组成中途检查点的 K 次未来。
独立核验局部受困状态后，需从同一完整闭环快照生成每点 64 次原策略续跑，
保留当前 chunk，只改变之后的随机性，截止时间保持不变。
对照已匹配任务和预算，但未验证同任务阶段。

新分支结果可交给严格校验接口生成成功概率软标签：

```bash
python -m probability.run --branch-results /path/to/verified_branches.csv --output results/branches
```

必需 CSV 列：

```text
checkpoint_id,branch_id,future_seed,snapshot_sha256,policy_sha256,current_chunk_sha256,restore_audit_sha256,remaining_action_steps,success,complete,policy_unchanged,current_chunk_preserved,full_restore_verified,independent_future_rng
```

每个验证标志必须为真实验证后的 `True`，哈希必须指向实际采集及恢复审计产物。
接口验证记录一致性，不能从 CSV 自行证明恢复正确；不得为了通过校验填入虚构标志。
同父检查点的所有分支在后续训练/测试中必须归为一组。
输出包含 `soft_label_trials` 与 Wilson 区间，0/64 不解释为真实概率严格为零。

自然脱困头仍需要独立标注的 trap 状态与首次物理脱困时间。
报警消失、最终成功、旧恢复实验中的干预结局都不能直接替代这些标签。
