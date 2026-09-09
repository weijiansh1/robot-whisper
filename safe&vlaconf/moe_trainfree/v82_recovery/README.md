# v8.2-R 路由状态监测实验

本目录实现绝对冻结支持、近期扰动组合、持续确认消融和可清除路由状态。
完整结果见 [REPORT_ZH.md](../results/v82_recovery_20260908/REPORT_ZH.md)，
预先固定的候选、划分与选择规则见 [PROTOCOL_ZH.md](PROTOCOL_ZH.md)。

主工作点没有提升：同预算对照 274 TP / 27 FP，开发选中的五折流程 255 TP / 29 FP。
不能用后续信号清除从误报中扣除曾经发生的报警，也不将这些状态等同于物理 trap 或恢复。
默认导出的 A-only 在线 profile 按开发规则选择了原分数回退项。

## 复现

在 `safe&vlaconf` 目录中运行，使用尚不存在的实验输出目录。
依赖现有 NumPy、SciPy、pandas、matplotlib、pytest、zarr，以及同工作区的 v7 和 HUB 缓存。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q moe_trainfree/v82_recovery/test_recovery.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python moe_trainfree/v82_recovery/run_experiment.py --output /tmp/v82-recovery-replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python moe_trainfree/v82_recovery/verify_recovery.py --output /tmp/v82-recovery-replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python moe_trainfree/v82_recovery/report_recovery.py --output /tmp/v82-recovery-replay
```

自定义目录报告里的代码目录相对链接需按位置解析；标准结果位于 `moe_trainfree/results/`。

## 在线使用

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("moe_trainfree/v82_recovery").resolve()))
from recovery import RecoveryGuardMonitor

profile = "moe_trainfree/results/v82_recovery_20260908/profile_deploy.json"
guard = RecoveryGuardMonitor(profile, checkpoint=checkpoint_sha256)
result = guard.update(hb_router_probs)  # 当前 [8,10,11,32] 非负、有限概率
```

开始新轨迹时调用 `guard.reset()`。profile 验证 checkpoint 指纹，不按任务分支。
`profile_0.json` 至 `profile_4.json` 是不同初态留出折的实验 profile；
`profile_deploy.json` 是单独使用 A 数据导出的单一在线配置。

- `state`：当前 NORMAL / WATCH / ALARM，低阈值连续两次未越界后可清除。
- `ready`：是否已有可用的因果分数；warm-up 的 NORMAL 不能理解成安全认证。
- `trigger_now`：本 query 的分数是否超过高阈值。
- `ever_alarm` / `first_alarm_query`：历史记录，状态清除后保持不变。
- `signal_cleared`：当前路由告警状态清除的事件，不表示目标达成或物理恢复。

接口不加载策略，不执行动作，不触发恢复控制；当前实现沿用仓库的逐前缀重放方式。
统计只在保存语料的 query 范围内验证，不能据此声称任意长执行的误报保证。
