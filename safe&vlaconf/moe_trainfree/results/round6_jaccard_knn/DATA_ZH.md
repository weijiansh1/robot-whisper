# JA / WJ 实验数据

本目录包含 12 个 suite/split 设置。跨折轨迹可能重复，不应当作独立新增样本。
图表用相同 B 测试轨迹评价四个新距离方法，并保留第五轮基线。

## 直接读 CSV

| 文件 | 内容 |
| --- | --- |
| `unseen_group5_summary.csv` | 未见任务、组校准 5% 下的平均 recall、FPR、t_det |
| `suite_group5_summary.csv` | 同一工作点按套件分开 |
| `unseen_operating_points.csv` | 预先固定的全部 alpha 工作点 |
| `common_horizon_summary.csv` | 同任务统一观察长度 AUC、同初态 AUC、可评分比例 |
| `ranking_metrics.csv` | 每折、方法、seen/unseen、观察窗口的完整排名指标 |
| `alarm_metrics.csv` | 每折、方法、校准类型、alpha 的完整报警指标 |
| `episode_decisions.csv` | 四个新方法在 5% 校准下的逐轨迹首次报警 |
| `illustration_points.csv` | 上一张投影的 1,162 个示意点，追加四种新方法的分数、阈值和判断 |
| `outcome_alignment.csv` | 全部 32,000 条轨迹的身份、最终结果，行序与预测的 global row 对应 |

`recall` 和 `fpr` 是 0 到 1 的比例，转为百分比需乘 100。
`t_det` 是失败轨迹的 `(首次报警query+1)/轨迹长度` 均值，漏检取 1。
`first_alarm=-1` 表示未报警；`failure` 是整条轨迹的最终失败标签，不是逐 query 故障标签。

## 逐 query 预测

`predictions/<suite>_<seed>.npz` 可用 `numpy.load(..., allow_pickle=False)` 读取。

```python
import numpy as np
import pandas as pd

z = np.load('predictions/libero_goal_20260907.npz', allow_pickle=False)
frame = pd.read_csv('outcome_alignment.csv')
method = list(z['methods']).index('route_wj_site_k20')
scores = z['scores'][method]       # [1760 test episodes, 52 queries]
test_metadata = frame.iloc[z['test_rows']]
unseen = z['test_unseen']
alpha_index = list(z['alphas']).index(0.05)
threshold = z['thresholds'][1, alpha_index, method]
first_alarm = z['first'][1, alpha_index, method]
```

`query` 从 0 开始，表示一次策略推理。每次推理生成一个 10 步动作 chunk。
q7 前及轨迹结束后的分数为 NaN。`thresholds` 和 `first` 的首轴依次为 episode、task/init 组。
`calibration_scores` 是 560 条 A 校准轨迹的分数，`calibration_labels=0` 表示历史成功。

| 方法 ID | 定义 |
| --- | --- |
| `route_ja_k20` | 80 个对齐位置的 Top-4 集合 Jaccard 距离平均 |
| `route_wj_site_k20` | 各位置 WJ 距离再平均 |
| `route_wj_aligned_k20` | 保留对齐并展平后的整体 WJ 距离 |
| `route_hellinger_aligned_k20` | 各位置 Hellinger 距离的 RMS |

每种方法都独立选择最近 20 个成功参考点，以距离均值评分，严格大于自己的阈值才报警。

## 投影点

`illustration_points.csv` 保留之前的 `pc1/pc2`、10 维和二维基线分数；
新增列为 `<方法ID>_score`、`_threshold`、`_ratio`、`_exceeds`。
比值大于 1 表示当前分数超阈值，不是失败概率，也不是已经锁存的报警状态。

`pc1/pc2` 仍来自原 10 维动态的冻结 PCA，不是对 JA/WJ 路由距离重新拟合的坐标。
这些示意点提高了失败抽样比例，每条轨迹最多 8 个点，不应拿来替代完整测试集的检出率评估。

## 数据包范围

`jaccard_knn_results.zip` 包含本说明、报告、CSV、12 折逐 query 预测、校准表、图及核验记录。
它提供结果分析所需数据，不包含较大的 `profiles/` 参考概率和 `route_cache/` 原始路由缓存；
后两者已保存在本地同目录。复算原始距离的代码与依赖见原仓库的 `jaccard_knn/README.md`。
