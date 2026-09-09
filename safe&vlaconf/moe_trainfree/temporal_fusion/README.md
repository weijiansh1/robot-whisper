# 报警分布与 kNN / v7 / v8 的无训练结合

[结果报告](../results/round7_temporal_fusion/REPORT_ZH.md)记录完整分布和两轮修改。
[第一轮协议](PROTOCOL_ZH.md)固定十组规则；
[第二轮协议](CROSS_TASK_CALIBRATION_ZH.md)记录随后提出的跨任务校准。

全部使用既有 MoE 路由缓存，无网络训练、无新 rollout。A 成功标签用于
参考库和校准，B 标签只用于评价。这些 B 任务已被探索，不能当成新盲测。

## 在线接口

`TemporalFusionMonitor` 使用每折 profile，支持第一轮的全部十组候选。
`KnNV8Monitor` 将 kNN 分支与现有冻结全局 v7/v8 或 v8.2 做布尔 OR。
两个接口都逐次接收 `[8,10,11,32]` 路由张量，跨轨迹必须调用 `reset()`。
每个 query 是一次模型推理，产生十个动作；q 从零开始。
接口不执行环境动作，不读取最终轨迹长度或任务身份。

例如，在仓库根目录启动独立 Python 进程：

```python
import sys
import numpy as np
sys.path.insert(0, 'safe&vlaconf/moe_trainfree/temporal_fusion')
from hybrid_monitor import KnNV8Monitor

path = 'safe&vlaconf/moe_trainfree/results/round7_temporal_fusion/cross_task_calibration/profiles/libero_long_20260907.npz'
with np.load(path) as profile:
    checkpoint = str(profile['checkpoint'])

monitor = KnNV8Monitor(path, checkpoint, knn_method='knn12_v8', version='v8.2')
result = monitor.update(hb_router_probs)
# result includes alarm, first_alarm_query, knn_alarm, v7_alarm, and v8_alarm.
```

实际接入时 `checkpoint` 必须来自当前运行策略，与 profile 身份核对；
上例从文件读取只是展示该离线 profile 的调用方式。
跨任务校准 profile 只替换 `knn10`、`knn12_v8` 两个方法的阈值，
其他候选仍是原 Round 7 阈值。OR 保留分支各自阈值，没有联合 5% 保证。
第一轮事先指定的 `knn_and_v8` 在事件前预警方面退步，不建议据此替换原方法。

## 复现

在当前仓库环境中使用 NumPy、pandas、scikit-learn、matplotlib、zarr、pytest。
选择不存在的输出目录，保留既有封存结果。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLOSC_NTHREADS=2
result_dir='safe&vlaconf/moe_trainfree/results/temporal_fusion_reproduction'
python 'safe&vlaconf/moe_trainfree/temporal_fusion/extract_v8.py' --output "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/run_fusion.py' --output "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/analyze_fusion.py' --input "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/verify_fusion.py' --input "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/plot_fusion.py' --input "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/cross_task_calibration.py' --input "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/verify_cross_task.py' --input "$result_dir"
python 'safe&vlaconf/moe_trainfree/temporal_fusion/summarize_cross_task.py' --input "$result_dir"
python -m pytest -q 'safe&vlaconf/moe_trainfree/temporal_fusion/test_fusion.py' 'safe&vlaconf/moe_trainfree/boundary_knn/test_knn.py'
```

`v8_padding_audit.csv` 记录旧 v8 的补齐区如何进入阈值统计。
新代码用真实有效 query 掩码；历史冻结报警超出实际轨迹的记录也被排除。
历史全局 v8 与同折重新校准的 `v8_guard` 是两组不同对照，结果不能混用。

## 误报原因诊断

[诊断报告](../results/round7_temporal_fusion/false_alarm_diagnosis/REPORT_ZH.md)
分解任务、初始状态、实际控制步比例与十维距离贡献。固定阈值和特征尺度的
等量补库对照支持任务参考覆盖不足这一解释。补入未见任务 A 成功样本仅用于
事后归因，不能作为未见任务泛化结果。复现脚本为 `diagnose_false_alarms.py`。

所有图为可独立导出的 PNG/PDF；所有首次报警、分布、风险集和校准阈值
均导出为 CSV/NPZ。模型评价不以剩余提前量为理由移除成功轨迹的误报。
