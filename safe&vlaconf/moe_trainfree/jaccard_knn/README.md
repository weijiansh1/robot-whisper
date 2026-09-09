# JA / WJ 成功参考 kNN

采用最后 flow 的 8 层 x 10 动作 token 路由，对齐位置计算 Top-4 Jaccard、逐位置平均 WJ、
对齐展平 WJ 和 Hellinger RMS 距离。每个 query 精确检索 20 个成功参考点。

[第六轮报告](../results/round6_jaccard_knn/REPORT_ZH.md)、[数据说明](../results/round6_jaccard_knn/DATA_ZH.md)、[固定方案](PROTOCOL_ZH.md)。
输出目录：`../results/round6_jaccard_knn/`。
沿用第五轮 12 个划分和参考点身份，q7 起评分，保留原 alpha 与两种成功校准规则。
10 维动态及原路由 kNN 的预测作为已有基线，不重新训练或调整。

## 运行

从仓库根目录运行。需要 NumPy、Numba、pandas、SciPy、scikit-learn、matplotlib、zarr 和 pytest。
当前系统 NumPy 2.5 与已有 Numba 不兼容，本轮在隔离环境中使用 NumPy 2.4.6。

```bash
python -m venv --system-site-packages /tmp/himoe-jaccard-knn-env
/tmp/himoe-jaccard-knn-env/bin/python -m pip install 'numpy==2.4.6'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 BLOSC_NTHREADS=2
knn_python=/tmp/himoe-jaccard-knn-env/bin/python
knn_out='safe&vlaconf/moe_trainfree/results/jaccard_reproduction'
"$knn_python" -m pytest -q 'safe&vlaconf/moe_trainfree/jaccard_knn/test_metrics.py'
"$knn_python" 'safe&vlaconf/moe_trainfree/jaccard_knn/extract_routes.py' --output "$knn_out"
"$knn_python" 'safe&vlaconf/moe_trainfree/jaccard_knn/run_jaccard.py' --output "$knn_out"
"$knn_python" 'safe&vlaconf/moe_trainfree/jaccard_knn/evaluate_jaccard.py' --input "$knn_out"
"$knn_python" 'safe&vlaconf/moe_trainfree/jaccard_knn/verify_jaccard.py' --input "$knn_out"
"$knn_python" 'safe&vlaconf/moe_trainfree/jaccard_knn/summarize_jaccard.py' --input "$knn_out"
```

提取器可复用哈希一致的原始路由缓存；评分器拒绝覆盖已保存的预测。
原始数据和第五轮结果是运行依赖，包含成功参考身份、标签校准和原基线。

## 在线接口

`metrics.JaccardMonitor` 接收每次推理记录的完整概率 `[8,10,11,32]` 和实际 Top-4 ID `[8,10,11,4]`，
通过 `update(probabilities, expert_ids)` 返回分数、阈值、当前触发和锁存报警。
需传入匹配 checkpoint 的 profile；每条新轨迹调用 `reset()`。本轮只做原始数据回放，没有执行策略干预。

## 数据字段

- `predictions/*.npz`：`scores[method,episode,query]`、`calibration_scores`、`thresholds[kind,alpha,method]`、`first`；
  `methods` 给出方法顺序，`test_rows` 对应 `outcome_alignment.csv` 的零基行号，`test_unseen` 标出未见任务。
- `profiles/*.npz`：相同的成功参考 episode/query、float16 路由概率、实际 Top-4 ID、checkpoint 与阈值。
- `unseen_group5_summary.csv`、`suite_group5_summary.csv`：相同组校准 5% 下的 recall、FPR 和 t_det。
- `common_horizon_summary.csv`：同任务统一观察窗口 AUC 和同初态 AUC。
- `episode_decisions.csv`：全部新方法在 5% 下的逐轨迹首次报警；`-1` 表示未报警。
- `illustration_points.csv`：沿用上一张图的 1,162 个示意点，追加四种新方法的分数、阈值、比值和超阈值判断。
  其中 `pc1/pc2` 仍是原 10 维动态 PCA 坐标，`failure` 是最终轨迹标签，示意点提高了失败抽样比例。
- `verification.json`：原始路由、直接距离排序、校准和在线回放核验。

`query` 从 0 开始，表示一次策略推理；分数严格大于阈值才触发。
无新增网络训练，但成功参考筛选和阈值校准需要历史成功标签。
