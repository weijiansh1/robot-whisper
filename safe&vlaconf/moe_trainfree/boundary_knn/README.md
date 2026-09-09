# MoE 外围分数与 kNN

当前完整方法、最新结果与各版本关系见[方法总览](../METHOD_SUMMARY_ZH.md)。

结果：[第五轮报告](../results/round5_knn/REPORT_ZH.md)。方案：[PROTOCOL_ZH.md](PROTOCOL_ZH.md)。

复用 v7 的 10 维相对动态，对照参考半径、成功 kNN 距离、混合库失败投票、
参考 PCA 二维 kNN、256 维路由和行为信号。无新增网络训练，历史标签用于参考筛选和成功校准。

## 复现

从仓库根目录运行。新输出目录必须不存在，避免覆盖已经保存的预测。

```bash
knn_out='safe&vlaconf/moe_trainfree/results/knn_reproduction'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python 'safe&vlaconf/moe_trainfree/boundary_knn/run_knn.py' --output "$knn_out"
python 'safe&vlaconf/moe_trainfree/boundary_knn/evaluate_knn.py' --input "$knn_out"
python 'safe&vlaconf/moe_trainfree/boundary_knn/verify_knn.py' --input "$knn_out"
python 'safe&vlaconf/moe_trainfree/boundary_knn/plot_knn.py' --input "$knn_out"
python 'safe&vlaconf/moe_trainfree/boundary_knn/project_knn10.py' --input "$knn_out"
python -m pytest -q 'safe&vlaconf/moe_trainfree/boundary_knn/test_knn.py'
```

使用当前 NumPy、pandas、SciPy、scikit-learn、matplotlib、zarr，以及第三轮缓存。
全流程在 CPU 上运行，不需要加载 VLA 或启动环境。

## 在线接口

[knn_monitor.py](knn_monitor.py) 接收每次决策的完整 flow 路由，逐 query 计算特征、分数与锁存报警。
`checkpoint` 必须传入当前策略的真实标识，与 profile 严格匹配；新轨迹调用 `reset()`。

```python
import sys
sys.path.insert(0, 'safe&vlaconf/moe_trainfree/boundary_knn')
from knn_monitor import BoundaryKNNMonitor

monitor = BoundaryKNNMonitor(
    'safe&vlaconf/moe_trainfree/results/round5_knn/profiles/libero_long_20260907.npz',
    checkpoint=policy_checkpoint_id,
    method='dyn_success_knn_k20',
    alpha=0.05,
    grouped=True,
)
result = monitor.update(hb_router_probs)  # [8, 10, 11, 32]
```

q7 前等待；持续确认版本 q9 前等待。接口限定本轮核验过的 52-query 时域。
行为对照 `eef_motion_low` 还需逐步传入 `eef_position`，MoE 分数不读取 EEF。
这是可重放的监控接口，本轮没有执行策略恢复或真实机器人干预。

## 产物

- `profiles/`：参考向量、原始 episode/query 身份、归一化、PCA 参数与阈值。
- `predictions/`：全部测试分数、校准分数、首次报警、划分及方法名称。
- `calibration/`：独立成功轨迹及 task/init 组的校准秩和超阈值计数。
- `ranking_metrics.csv`、`alarm_metrics.csv`：新方法与同折历史基线的 seen/unseen 结果。
- `task_*_metrics.csv`、`physical_timing.csv`：任务明细及释放事件时序。
- `figures/pca2_knn_regions.png/pdf`：冻结参考 PCA 的二维对照及其实际校准边界。
- `projection10/figures/knn10_projected_scores.png/pdf`：真实 10 维 kNN 距离/阈值投影；各套件另有三栏对照图。
- `projection10/projected_points.csv`、`calibration_diagnostics.csv`：逐点坐标、两种分数与超阈值判断，以及圈大小的校准诊断。
- `sealed_manifest.json`、`evaluation_summary.json`、`verification.json`：输入、代码、预测与复算记录。

主方法事先固定为 10 维动态的成功 kNN-20，主工作点为 task/init 组校准 5%。
不同任务的真实误报率可能显著偏离标称值；不可用完整轨迹 AUC 代替早期预警能力。

## 一条轨迹的逐步视图

[52 格大图、动画与逐步数据](../results/round5_knn/trajectory_dynamics/long_episode_223/README_ZH.md)
沿用已固定的 Long 失败案例，逐 query 展示真实 10D 的 20 个近邻、固定二维投影、
记录状态重绘和完整距离曲线。脚本为 `plot_trajectory_knn.py`，仅复用已有数据。
当前距离回落与已触发的锁存报警分别标注；本次没有改变噪声或执行救回干预。

## 余弦距离对照

[固定方案](COSINE_PROTOCOL_ZH.md) / [结果、逐步图与数据](../results/round8_cosine_knn/REPORT_ZH.md)。
固定 Round 5 的 10 维特征、参考 chunk、k=20 和全部 12 折，只替换为余弦距离并重新校准。
主工作点下，未见任务误报率从 6.09% 降至 1.59%，召回率也从 89.82% 降至 3.05%；
纯余弦不能直接替换现有分数。原欧氏实现、参考库和预测保持原样。

[低召回原因诊断](../results/round8_cosine_knn/diagnosis/REPORT_ZH.md)进一步固定近邻身份与打分分别对照：
保留余弦近邻、恢复欧氏打分，未见任务检出回升到 356/491；固定欧氏近邻、仅用余弦打分仍为 13/491。
另检查了中心化、校准峰值来源和幅度分解，全部 248,406 个有效 chunk 的原分数由独立 SciPy 距离矩阵复现。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python 'safe&vlaconf/moe_trainfree/boundary_knn/cosine_knn.py' score --output /tmp/himoe-cosine-reproduction
python 'safe&vlaconf/moe_trainfree/boundary_knn/cosine_knn.py' verify --output /tmp/himoe-cosine-reproduction
python 'safe&vlaconf/moe_trainfree/boundary_knn/compare_cosine.py' --output /tmp/himoe-cosine-reproduction
```

## 全部 32,000 条轨迹

[全量划分方案](FULL_CORPUS_PROTOCOL_ZH.md) / [结果、分布图和逐轨迹数据](../results/round9_full_corpus/REPORT_ZH.md)。
四个套件的 40 个任务按 16 轮完整任务留出覆盖，每条 A/B 轨迹恰好测试一次。
本轮重新建立七任务的 A 参考库和独立初态校准集；同轮测试任务不参与任何统计量拟合。
沿用六种几何分数、10D、k=20 和 task/init 组校准 5%，不训练网络或新增 rollout。

1,096 条失败中，欧氏检出 977 条、纯余弦检出 38 条、余弦近邻加欧氏打分检出 910 条；
30,904 条成功中分别误报 2,754、672、1,967 条。Goal 套件的欧氏误报率为 24.73%。
这是新划分的完整交叉验证，不能直接与旧 12 折的重复出现次数相加。

`trajectory_results.csv` 恰好 32,000 行；`all_test_scores.npz` 保存全部 `[6,32000,52]` 测试分数，
另有 A/B、suite、task、逐 q 和实际控制进度分布。复现命令见结果报告。

## k=1 至 10 扫描

[固定方案](K_SWEEP_PROTOCOL_ZH.md) / [全部结果和图](../results/round10_k_sweep/REPORT_ZH.md)。
沿用 Round 9 的同一参考库、归一化和完整任务留出划分，对五种 kNN 设置扫描 k=1 至 10，
另以 k=20 核对基线；每个 k 独立校准。欧氏 k=1、10、20 的误报率分别为 10.80%、9.57%、8.91%。
原 k=20 的 2,754 条误报中有 2,480 条在 k=1 至 10 全部报警，减少邻居数未解决任务集中误报。
所有逐轨迹数据、首次报警分布、固定 k20 阈值诊断对照和复现命令均在报告内。

## K-means 成功模式

[固定方案](KMEANS_PROTOCOL_ZH.md) / [精确率、召回和完整数据](../results/round11_kmeans/REPORT_ZH.md)。
策略参数冻结，在同一成功参考库上离线拟合 C=1、2、4、8、16、32、64 的聚类中心和半径。
比较最近中心距离、最近簇半径归一化、多个尺度球的并集；每种方法使用同样的四档校准预算。
事先主配置为 C=32；扫描中的探索候选 C=4 最近簇尺度版本在标称 5% 下检出 942/1,096，
成功误报 1,220/30,904，报警精确率 43.57%，原 kNN 为 26.19%。严格 2% 时精确率为 64.26%，
召回率降至 65.78%。C=4 来自本轮扫描，尚不是新盲测结论；所有剩余误报和漏检已导出。

## 成功终止前的共同窗口

[固定方案](COMMON_HORIZON_PROTOCOL_ZH.md) / [LIBERO-10 结果和原 82 条误报明细](../results/round12_common_horizon/REPORT_ZH.md)。
复用冻结分数，将同任务轨迹截到该任务首次成功前，另统一全体 8,000 条轨迹到 q13 / 130 动作。
C4 加半径保留原阈值时，共同任务窗口检出 181 条、误报 39 条；统一 130 步时为 33 条、11 条。
历史校准前缀也截到 130 步再定阈值后，检出为 246 条、误报 373 条，精确率 39.74%。
原完整轨迹 AUROC 98.44 不能代替早期预测能力。脚本 `evaluate_common_horizon.py` 拒绝覆盖已有结果。
