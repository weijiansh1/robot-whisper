# ST/AC 确认支路：回溯实验结果

2026-09-09。已实现、已完成全量缓存评价和抽样原始路由回放。

结论：ST/AC 确认能过滤部分 AC 新增误报，但同时损失新增检出。
本轮没有证明它改善整体判别能力，也没有解决 S05 成功轨迹中的异常问题。
作为可选实验支路保留，不替换既有默认监测器。

## 1. 数据和方法

- 现有 40 任务、A/B 各 16000 条轨迹，共 508023 次查询。
- B：564 失败、15436 成功；每条 B 只评价一次。
- 五折按 `(task, init_state_id)` 隔离 A 参考、A 校准、B 测试角色。
- A/B 此前均被研究过，所以是回溯验证，不是独立盲测。
- 不训练分类器、融合权重或策略；仅用 A 成功参考的经验分位校准。
- 新 ST 支路只用当前及历史路由，不使用轨迹总时长、进度或物理信号。
  原 v8.2 的 `0.0015*q` 阈值放宽项原样保留；本轮未计算时长/时钟基线。

增加 ST 最后去噪步的跨查询路由变化。AC 沿用当前的逐 token Hellinger
再求平均定义，未混入另一缓存中“先平均 token 再求距离”的 action 指标。

各层分别以前四个有效变化的中位数为参考，取最近三次查询均值：

```
r_ac = log((current_ac + 1e-6) / (reference_ac + 1e-6))
r_st = log((current_st + 1e-6) / (reference_st + 1e-6))
gap  = median_L12..15(r_ac - r_st)
```

gap 连续两次平滑分数取最小值确认，最早 q8（q 从 0 开始）。
正值表示 ST 的跨查询变化相对 AC 减弱，不等于物理状态不变或任务陷入 trap。
该自参照归一化特征不是旧实验原始 ST-AC 差值的等价改写。

```
AC 触发 = 原有 acceleration/decoupling 两支路合并尾部 <= alpha
ST 确认 = 当前 confirmed_gap > 0 且 gap 的成功轨迹峰值尾部 <= 0.10
新增触发 = 同一查询 AC 触发且 ST 确认
最终首次报警 = 原冻结 v8.2 与新增支路首次报警的较早者
```

AC 的校准库和阈值全部复用旧结果。ST 确认阈值在本轮结果计算前固定为 0.10。
没有重新分摊原 v8.2 的误报预算。组合的整体名义误报率不能叫做 alpha；
下表报告实际 FPR。ST/AC 不假定独立，未相乘其尾部概率。

## 2. 主设置：按初始状态分组校准，新增 AC 预算 1%

| 方法 | 检出失败 / 564 | 误报成功 / 15436 | 召回率 | FPR |
|---|---:|---:|---:|---:|
| 原冻结 v8.2 | 475 | 99 | 84.22% | 0.641% |
| 原 v8.2 + AC | 479 | 115 | 84.93% | 0.745% |
| 原 v8.2 + AC/ST 确认 | 477 | 108 | 84.57% | 0.700% |

相对 AC 组合：少 7 个误报，少检出 2 个失败，另有 3 个共同检出发生延迟。
过滤的 7 个成功和损失的 2 个失败全部来自 libero_long。

相对原冻结 v8.2：新增 2 个检出、9 个误报；原 475 个检出中 7 个提前、468 个同刻，
没有丢失或延后任何原检出。原来的 99 个误报也全部保留。

从增量看，AC 原本增加 4 个检出、16 个误报；ST 保留其中 2 个检出和 9 个误报。
因此不能把误报下降单独描述为方法进步：增加确认条件本来就会减少触发，
本轮需要同时承担新增检出减少一半的代价。

## 3. 补充设置：按轨迹校准，新增 AC 预算 1%

| 方法 | 检出失败 | 误报成功 | 召回率 | FPR |
|---|---:|---:|---:|---:|
| 原冻结 v8.2 | 475 | 99 | 84.22% | 0.641% |
| 原 v8.2 + AC | 493 | 174 | 87.41% | 1.127% |
| 原 v8.2 + AC/ST 确认 | 490 | 153 | 86.88% | 0.991% |

相对 AC 组合，少 21 个误报、少检出 3 个失败，19 个共同检出延迟。
相对原方法仍多出 15 个检出和 54 个误报。
两种校准的统计单位不同，不能将它们当作同一工作点。

所有固定预算的增量：

| 校准 | alpha | 相对原 v8.2 新增 TP | 新增 FP | 相对 AC 组合损失 TP | 减少 FP |
|---|---:|---:|---:|---:|---:|
| episode | 0.005 | 4 | 20 | 1 | 8 |
| episode | 0.010 | 15 | 54 | 3 | 21 |
| episode | 0.020 | 24 | 103 | 12 | 68 |
| episode | 0.050 | 35 | 217 | 28 | 301 |
| task_init | 0.005 | 0 | 3 | 0 | 0 |
| task_init | 0.010 | 2 | 9 | 2 | 7 |
| task_init | 0.020 | 3 | 13 | 2 | 7 |
| task_init | 0.050 | 10 | 45 | 11 | 49 |

所有设置都保留原 v8.2 的报警；没有出现“增加检出且零新增误报”的设置。
本轮没有用这些 B 结果反向选择阈值或改动规则。

## 4. 同任务、同查询位置的 AUROC

共同有限观测支持上先逐任务/查询计算，再在任务内平均查询，最后平均任务。
可比任务 36 个；这不是把不同任务、不同长度的全轨迹峰值直接池化得到的 AUROC。
原 v8.2 的参考值用已有 A 校准路由分数计算，不是二值冻结报警的 AUROC。

| 主设置下的分数 | q8..13 | 全程可比查询 |
|---|---:|---:|
| v8.2 参考路由分数 | 0.600587 | 0.645009 |
| AC 动力学排序 | 0.720877 | 0.704806 |
| AC/ST 确认裕量 | 0.720871 | 0.704780 |
| 归一化 ST/AC 关系 gap | 0.588247 | 0.578754 |
| 低原始 ST-AC 差值 | 0.555261 | 0.548997 |
| ST 相对变化偏低 | 0.539063 | 0.529144 |
| ST 绝对变化偏低 | 0.513461 | 0.497512 |

AC/ST 确认相对 AC 的全程差值为 -0.000026，按 suite 内任务 bootstrap
95% 区间 [-0.000629, 0.000475]。补充 episode 设置的差值为 +0.000128，
区间 [-0.002307, 0.002860]。均没有显示稳定提升。

AC 排序使用未截断尾部，避免大量正常查询被截成同分；确认裕量为
`min(log(.01/(2*min(p_acc,p_decoupling))), log(.10/p_gap_effective))`。
gap<=0 时 p_gap_effective=1。这个连续分数依赖固定主阈值 .01；
它描述新增支路的排序，不应被称为包含冻结 v8.2 OR 组合的完整 AUROC。

gap 单独触发的固定诊断也没有提供很强的原方法增量：主设置下为
160 TP / 25 FP，但相对原 v8.2 没有新增 TP，新增 4 FP。
其 160 个检出不能误认为是补回了 160 个原方法漏检。

## 5. S05 和自行恢复问题

任务：盘子旁的黑碗移到盘子。B 共 24 失败、376 成功。

| 方法 | 主设置 TP / FP | episode 补充设置 TP / FP |
|---|---:|---:|
| 原 v8.2 | 18 / 2 | 18 / 2 |
| 原 v8.2 + AC | 18 / 2 | 19 / 2 |
| 原 v8.2 + AC/ST 确认 | 18 / 2 | 19 / 2 |

- 成功 episode 174：原 v8.2 q17 报警；episode 设置下 AC 与 AC/ST 都在 q16 报警。
  ST 确认未过滤该成功轨迹中的 AC 异常。
- 成功 episode 190：三种组合都未报警。
- 失败 episode 286：episode 设置下 AC 与 AC/ST 都在 q16 检出，原方法漏检；
  主 task_init 设置下仍漏检。
- 失败 episode 96：两种组合、两种校准均未补回。

原方法报警保留规则本身不允许撤回成功 episode 115/174 的旧误报。
但 174 的新增 AC 触发也通过了 ST 门，说明本次新增条件本身没有解释或排除
该恢复案例。最终成功不等于过程中没有异常；路由异常也不足以判定不可恢复失败。

## 6. 验证

- 20 项单元测试通过，包括 5 项新增 ST/AC 测试。
- 80 个来源、32000 条轨迹、508023 个查询完成缓存轴核对。
- 冻结 v8.2 在 32000 条轨迹上的首次报警与历史结果完全一致。
- 原 AC 支路全部 8 个设置、128000 个首次报警比较完全一致。
- 196 条原始轨迹、5668 个查询回放，其中包括 51 条阈值附近轨迹，
  以及主/补充 1% 设置相对原方法的全部新增报警轨迹。
- ST 原始读数与缓存逐值一致；gap 最大绝对差 3.2425e-5，来自 AC 原始/缓存的微小误差。
- 41 个经验尾部秩有差异，但 7840 个首次报警比较均一致。
  此结论覆盖所抽查的原始轨迹，并非全部 16000 条 B 的原始读取认证。
- 独立重算 1845 行报警指标、720 行配对增量，以及 210 个 Mann-Whitney AUROC。
- 历史报警不随当前异常减弱而消失；所有成功报警仍计入 FP。

## 7. 复现与入口

从 `safe&vlaconf` 运行，输出目录须不存在，以免覆盖结果：

```bash
python -m unittest discover -s moe_trainfree/routing_dynamics -p 'test_*.py' -v
python moe_trainfree/routing_dynamics/run_state_action_experiment.py --output /tmp/state_action_replay
python moe_trainfree/routing_dynamics/verify_state_action.py --output /tmp/state_action_replay
```

在线入口 `moe_trainfree.routing_dynamics.state_action.StateActionGuardMonitor`：

```python
from moe_trainfree.routing_dynamics.state_action import StateActionGuardMonitor

guard = StateActionGuardMonitor(profile_path, checkpoint, kind="task_init", alpha=.01)
result = guard.update(hb_router_probs)  # Only the current [8,10,11,32] routing.
first_alarm = result["first_alarm_query"]
guard.reset()  # Call at each new episode.
```

回放 B 时 profile 选 `fold_(init_state_id % 5)_profile.json`，由外层评测器选择，
init 不输入在线监测器。部署时应先确定单个参考 profile，不能根据在线表现挑选。

协议：[STATE_ACTION_PROTOCOL_ZH.md](../../routing_dynamics/STATE_ACTION_PROTOCOL_ZH.md)。
代码：[state_action.py](../../routing_dynamics/state_action.py)。
细表：[alarm_metrics.csv](alarm_metrics.csv)、[paired_changes.csv](paired_changes.csv)、
[auroc_summary.csv](auroc_summary.csv)、[s05_alarms.csv](s05_alarms.csv)。
验证：[independent_verification.json](independent_verification.json)、[raw_replay.csv](raw_replay.csv)。
