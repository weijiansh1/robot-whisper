# 无训练 MoE：SAFE / VLAConf 实验适配

当前方法的统一说明：[方法总览](METHOD_SUMMARY_ZH.md)。包含数据与控制步定义、10D 特征公式、参考与校准、kNN/K-means、v7/v8 融合、完整实验结果、在线接口限制、数据格式与复现命令。更新至 Round 11 及 Round 12 的 LIBERO-10 共同窗口诊断。

文献实验核查：[SAFE v2 实验清单与评价细节](../SAFE_EXPERIMENTS_ZH.md)。区分论文已做实验、原文不一致处及可借鉴的 train-free 对照。

本目录使用现有冻结 HiMoE-VLA 的路由缓存，实际执行失败检测、执行前/在线排序、参考量与任务留出等实验。不训练额外预测网络。前两轮限制为无标签参考；第三轮按用户明确的 train-free 要求，允许标签参考和成功轨迹校准，并直接复用 v7。

首轮结果与解释：[REPORT_ZH.md](results/round1/REPORT_ZH.md)。

第二轮专家输出/hidden/动作配对实验：[REPORT_ZH.md](results/round2/REPORT_ZH.md)，[独立运行说明](round2/README.md)。

第三轮完整轨迹、SAFE 式成功校准与 v7 复用：[REPORT_ZH.md](results/round3_safe/REPORT_ZH.md)，[运行说明](safe_protocol/README.md)。下方命令和数据说明对应首轮。

第四轮 SAFE 式特征结构、同坐标多种着色与状态回放：[REPORT_ZH.md](results/round4_geometry/REPORT_ZH.md)，[运行说明](feature_geometry/README.md)。

第五轮外围分数、动态 kNN 与在线校准：[REPORT_ZH.md](results/round5_knn/REPORT_ZH.md)，[运行说明](boundary_knn/README.md)。

第六轮 JA/WJ 路由 kNN 距离对照：[REPORT_ZH.md](results/round6_jaccard_knn/REPORT_ZH.md)，[运行说明](jaccard_knn/README.md)。

第七轮完整报警分布、v7/v8 融合与跨任务校准：[REPORT_ZH.md](results/round7_temporal_fusion/REPORT_ZH.md)，[运行说明](temporal_fusion/README.md)。

全部 32,000 条轨迹的完整任务留出评估，含欧氏、余弦及幅度诊断对照：
[REPORT_ZH.md](results/round9_full_corpus/REPORT_ZH.md)、[逐轨迹 CSV](results/round9_full_corpus/trajectory_results.csv)。

固定全量参考库的 k=1 至 10 扫描及误报原因分析：
[REPORT_ZH.md](results/round10_k_sweep/REPORT_ZH.md)。

K-means 成功参考模式、局部半径和报警精确率实验：
[REPORT_ZH.md](results/round11_kmeans/REPORT_ZH.md)、[任务、阈值和时长诊断](results/round11_kmeans/diagnosis/DIAGNOSIS_ZH.md)。

LIBERO-10 成功终止前的共同窗口、原阈值与前缀重校准：
[REPORT_ZH.md](results/round12_common_horizon/REPORT_ZH.md)、[原 82 条成功误报明细](results/round12_common_horizon/original_82_false_alarms.csv)。

## 运行

从仓库根目录执行，需现有 NumPy、pandas、matplotlib、pytest、scikit-learn 环境及本仓库大规模特征缓存。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q 'safe&vlaconf/moe_trainfree'
```

复现完整过程使用一个尚不存在的输出目录；封存脚本拒绝覆盖已有结果：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/seal.py' --output 'safe&vlaconf/moe_trainfree/results/reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/evaluate.py' --input 'safe&vlaconf/moe_trainfree/results/reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/diagnostics.py' --input 'safe&vlaconf/moe_trainfree/results/reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/report.py' --input 'safe&vlaconf/moe_trainfree/results/reproduction'
```

代码和缓存路径由脚本位置定位。`seal.py` 不读取成功/失败标签；`evaluate.py` 验证所有预测和源文件哈希后才连接结果标签。`diagnostics.py` 是结果出来后增加的解释性核查，沿用已封存分数，不改变检测器。

## 输入

- MoE 分层 mobility：`moe-v4-0904/results/layerwise_mobility/`。
- 路由 acceleration / periodicity / margin：`double-selete/trainfree/results/` 下三个 `unlabeled_query_features.npz`。
- layer × flow 统计：`moe-flow-semantics-0906/results/step_profiles/`。
- 仅评价使用的标签：`VLA_MUI_HUB/physical-failure-labels/results/episodes.csv`、`failures.jsonl`。
- 复用方法：`moe-v7-0905/method/intrinsic_guard_monitor.py`、`unlabeled_budget_calibration.py`。

精确输入清单及 SHA-256 记录在 `results/round1/sealed_manifest.json`。首轮采用 16,000 条无标签参考和 15,600 条测试轨迹；历史研究接触过这些数据，不视为全新盲测。

## 输出

- `ranking_metrics.csv`：固定 query、事后半程、任务最短长度和完整轨迹的 AUC / AP / 覆盖率。
- `alarm_metrics.csv`：所有预算和方法的整段误报、检出率、提前量及主设置聚类区间。
- `suite_metrics.csv`、`scope_*_metrics.csv`：每套件及标准三套件 / Long 分组。
- `subject_release_metrics.csv`：失败目标物体匹配的脱手事件时刻，以此表解释物理提前量。
- `physical_timing.csv`：较粗的所有物体最早 release 评价，保留供审计。
- `selective_risk.csv`：不同因果检查点上未报警轨迹的被动失败率，不代表干预成功率。
- `paired_comparisons.csv`：主方法相对时钟和 freeze 的配对聚类差值区间。
- `reference_budget_audit.csv`、`diagnostic_summary.json`：预算、真实前缀回放和物理时间轴核验。
- `figures/`：独立 PNG / PDF 研究图。

3% 是无标签参考轨迹总报警预算，不是成功 FPR 保证。末层特征配对基线、概率校准指标、受控扰动和真实专家接管尚未完成；见报告中的逐项状态。
