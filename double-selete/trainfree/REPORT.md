# Train-free 双头 Trap 选择器结果

> **全 HUB 二元报警复验（2026-09-03）：** 37-task、14,800 轨迹的公平 q95
> 检验表明，双头 `max` 的总体 Trap recall 为 57.9%，低于信息匹配 mean 单标量的
> 66.5%。该结果不涉及排序，详见 [`HUB_BINARY_AUDIT.md`](HUB_BINARY_AUDIT.md)。

> **无监督聚类复验（2026-09-03）：** 自适应 HDBSCAN 只找到两个稀疏 MoE
> 尾部簇（覆盖 14.8%，85.2% 为 noise），且 `normal/loop/static/both` 四类 AMI
> 仅 0.013，条件置换 `p=0.146`。峰谷形状存在，但不能自然恢复 Trap 类型，详见
> [`UNSUPERVISED_CLUSTER_REPORT.md`](UNSUPERVISED_CLUSTER_REPORT.md)。

> **动态语法聚类（2026-09-04）：** 公共前缀内没有早期分离（GMM-BIC 四类
> AMI=0.0005，`p=0.421`）；完整轨迹出现偏弱但显著的 retrospective 类型结构
> （AMI=0.081，等长 failure 内 AMI=0.084），但不存在稳定自然密度簇，不能解释为
> 在线报警提升。详见
> [`UNSUPERVISED_DYNAMICS_REPORT.md`](UNSUPERVISED_DYNAMICS_REPORT.md)。

## 结论

**可以提高“分类型判断 Trap”的能力，并且适合给后续分类型修正提供入口；但不能把两头再压成一个 `max` 标量，并宣称总体 Trap 排序也提高了。**

在冻结的主设置（query 34、概率型 routing 特征、每头独立 Top-8）下：

| 输出 | 对应标签 AUC / recall | 信息匹配单头 | 差值（95% grouped bootstrap CI） |
|---|---:|---:|---:|
| Loop head，group-macro AUC | 0.673 | 0.632 | - |
| Static head，group-macro AUC | 0.936 | 0.628 | - |
| 两类 macro AUC | **0.805** | 0.630 | **+0.174 [0.114, 0.235]** |
| Loop head，Top-8 loop recall | **49.1%** | 29.8% | **+19.3 pp [9.1, 36.0]** |
| Static head，Top-8 static recall | **45.2%** | 28.1% | **+17.1 pp [8.3, 35.4]** |

两类 macro AUC 的配对 bootstrap `p < 1e-4`；Loop Top-8 recall
`p = 0.0001`，Static Top-8 recall `p < 1e-4`。这里的“信息匹配单头”是
`(loop_score + static_score) / 2`，使用完全相同的输入和分量，差别仅在于是否保留两个输出。

## 为什么不能合成一个分数

query 34 的 any-Trap group-macro AUC：

| 单标量排序 | AUC |
|---|---:|
| `max(loop_head, static_head)` | **0.657** |
| 信息匹配单头 mean | 0.705 |
| 通用低 route-mobility 基线 | 0.765 |

`max` 对任意一头的孤立高值都敏感。Loop head 对 static 常给低分，而 Static
head 对 loop 也不是同一方向；取最大值会把各自的假阳性一起带进 any-Trap
排名。因此，多头的收益存在于二维输出 `(loop_score, static_score)`，不是一个
更高的 union AUC。

每头独立 Top-8 后，16 个 pool 一共选中 237 条不同轨迹（平均 14.81/32；
19 条同时被两头选中）：

| 选择器（同为 14.81/pool） | Trap precision | Trap recall | Loop recall | Static recall | subtype macro recall |
|---|---:|---:|---:|---:|---:|
| 双头自然并集 | 46.8% | 57.5% | 61.4% | 56.2% | 58.8% |
| 单头 mean，同大小 | 46.8% | 57.5% | 68.4% | 52.7% | 60.6% |
| route mobility，同大小 | 46.8% | 57.5% | 29.8% | 70.5% | 50.2% |

所以并集没有提高同选择预算下的总体 Trap recall。它相对 route mobility
明显补回 loop（`+31.6 pp`），但牺牲 static（`-14.4 pp`）；macro 差值
`+8.6 pp` 的 CI 为 `[-0.1, 16.9] pp`，尚不能视为稳定提升。相对信息匹配
单头，macro 差值是 `-1.8 pp`，CI `[-9.3, 8.1] pp`。

## Onset 对齐

这组结果更直接支持后续修正时机：

| 检测点 | 类型头 AUC | 单头 mean | route mobility |
|---|---:|---:|---:|
| Loop onset 前 2 query | **0.846** | 0.524 | 0.262 |
| Static onset 当下 | **0.941** | 0.678 | **0.996** |

Loop head 相对单头的配对差值为 `+0.322 [0.218, 0.410]`，说明它提供了
通用 mobility 没有的早期 loop 信号。Static head 相对单头为
`+0.263 [0.164, 0.354]`，但仍低于 route mobility `0.055`；static 的低
mobility collapse 已经是非常强的无训练检测器。由于部分 pool 没有同类型
阴性对照，onset 分析覆盖 57 个 loop 事件/10 个 pool 和 114 个 static
事件/11 个 pool。

## 推荐接口

后续修正模块应直接读取两个布尔选择和两个连续分数：

```text
loop_selected, static_selected, loop_score, static_score
```

- `loop_only`：进入 loop 专用纠偏分支。
- `static_only`：进入 static 专用纠偏分支。
- `both`：进入冲突/组合分支，不要用 `max` 丢掉类型信息。
- `none`：不触发 routing-based 修正。

无标签分流清单位于
[`results/unlabeled_assignments.csv`](results/unlabeled_assignments.csv)，包含
q30/q34 和每头 Top-4/8/16 的操作点。带物理标签的审计版本位于
[`results/assignment_audit_labeled.csv`](results/assignment_audit_labeled.csv)。

## Train-free 边界

打分阶段没有读取 success、物理状态、action、Trap 标签或 onset，也没有
训练、标准化拟合、阈值校准和参数搜索。所有分量只在同一初始状态的候选池内
转成 percentile rank，再按协议中的固定等权公式合成。标签只在未标注分数和
协议哈希封存后，由独立评估脚本读取。

这仍然不是 discovery-independent 证据：方向来自此前对同一任务的机制分析。
因此当前数字可以用于这个任务上的方案选择和修正接口设计，不能替代新任务上
的冻结复验。

## 文件

- 冻结协议：[`PROTOCOL.md`](PROTOCOL.md)
- 无标签打分：[`score_heads.py`](score_heads.py)
- 标签评估：[`evaluate_heads.py`](evaluate_heads.py)
- 自动验证：[`validate_results.py`](validate_results.py)
- 主图：[`figures/trainfree_double_selector.png`](figures/trainfree_double_selector.png)
- 全部表格：[`results/`](results/)

硬 Top-4 support 结果仅作诊断。它的 q34 subtype-macro AUC 为 0.707，低于
概率型主结果 0.805；onset 对齐也分别低 0.073（loop）和 0.077（static）。
这与 fp16 第四/第五专家边界不稳定的风险一致。
