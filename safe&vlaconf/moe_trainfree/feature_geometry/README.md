# MoE 特征结构与 SAFE 式可视化

结果：[第四轮报告](../results/round4_geometry/REPORT_ZH.md)。

针对“失败点是否更多位于外围”的后续检查：[外围统计](../results/round4_geometry/edge_probe/REPORT_ZH.md)。

输入来自第三轮的完整轨迹缓存。本分析比较专家使用分布与 v7 相对动态，
不训练失败检测器。t-SNE/PCA 本身是无监督数值拟合，只用于解释性图表。

从仓库根目录运行，复现使用新的输出目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/feature_geometry/analyze.py' --output 'safe&vlaconf/moe_trainfree/results/geometry_reproduction'
bash 'safe&vlaconf/moe_trainfree/feature_geometry/run_render.sh' --output 'safe&vlaconf/moe_trainfree/results/geometry_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/feature_geometry/random_init.py' --input 'safe&vlaconf/moe_trainfree/results/geometry_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/feature_geometry/plot.py' --input 'safe&vlaconf/moe_trainfree/results/geometry_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/feature_geometry/verify.py' --input 'safe&vlaconf/moe_trainfree/results/geometry_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/feature_geometry/edge_probe.py' --input 'safe&vlaconf/moe_trainfree/results/geometry_reproduction'
```

可视化依赖当前 NumPy、pandas、SciPy、scikit-learn、matplotlib、Pillow。
状态重绘 wrapper 使用仓库已有 Python 3.8 / LIBERO / robosuite 1.4.1 / OSMesa 环境。
全流程使用 CPU，不加载策略模型，不重新执行 rollout。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q 'safe&vlaconf/moe_trainfree/feature_geometry/test_geometry.py'
```

## 产物

- `*_atlas.png/pdf`：一行一套坐标，四种着色；配对两行的 episode/query 完全相同。
- `*_sensitivity.png/pdf`：主 PCA 初始化、两个随机初始化、直接 PCA。
- `*_case.png/pdf`：同 task/init 成败分支的投影路径、v7 信号、历史运动和状态画面。
- `*_recorded_path.gif`：失败案例的记录状态与特征路径同步动画，每帧一个 query。
- `projections/`：全部点元数据、归一化后向量、原始 t-SNE 坐标和 PCA 坐标。
- `random_init/`：初始化核查后追加的结果，保留原有重复投影。
- `neighbor_points.csv`：全部 B 轨迹在固定 query 的跨任务高维近邻诊断。
- `neighborhood_summary.csv`：按任务宏平均的邻居失败比例与候选基准比例。
- `selected_episodes.csv`、`task_key.csv`、`cases.json`：取样与图例索引。
- `manifest.json`、`render_verification.json`、`final_verification.json`：哈希、重绘与复算核验。

图中增加了失败样本的取样比例；颜色、最终失败类型与归一化进度是事后标注。
投影邻近和高维标签富集均不能单独证明因果机制、故障前预测或跨任务部署性能。
