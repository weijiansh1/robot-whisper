# 复现顺序

CPU only。本机 8 张卡被无关进程占满，全部脚本不使用 GPU；缓存构建是 I/O 受限的，
GPU 也无收益。

## 0. 前置数据

不需要重跑任何 rollout。以下是只读依赖：

| 路径 | 内容 |
|---|---|
| `../VLA_MUI_HUB/cache_new/HiMoE-VLA/<task>/<run_id>/server/routes.zarr` | 原始 `hb_router_probs [rows,8,10,11,32]` float16 |
| `../moe-v4-0904/results/layerwise_mobility/*.npz` | 队列索引与逐层 mobility |
| `../moe-v7-0905/results/intrinsic_guard_v7/` | v7 profile 与封存首报 |
| `../moe-hb-front-back-0905/results/layer_graphs/` | token 图指标 |
| `../analysis_trap_taxonomy/results/` | 失败模态与闭合事件标注 |
| `../double-selete/trainfree/results/` | v7 输入特征与 outcome 标签 |

`development_main` ↔ run `seed1000_1007` ↔ run_id `right-50x8-20260903`；
`external_8b` ↔ `seed1008_1015` ↔ `right-50x8b-20260903`。
episode id 跨 run 重复，涉及 taxonomy 的连接一律用
`(task, init_state_id, flow_noise_seed)`。

## 1. 缓存（先跑，其余都依赖它）

```bash
python experiments/build_progress_cache.py --output results/progress_cache --batch 512 --resume
```
约 1 分 50 秒，三个队列各 ~35 MB（压缩后）。lag-1 距离必须逐位复现 v4 的
`mobility`，容差 2e-5；不一致会抛错退出，**不要放宽**。

## 2. 主结果

| 命令 | 耗时 | 对应报告 |
|---|---|---|
| `python experiments/verify_clock_ceiling.py` | 1 秒 | §二 时钟天花板 |
| `python experiments/combine_clock_and_routing.py` | 20 秒 | §三 组合交付 |
| `python experiments/fuse_v7_branches.py` | 3 分 | §七 v7 三支融合 |

`fuse_v7_branches.py` 与 `combine_clock_and_routing.py` 会先重建 v7 guard 并与
`sealed_first_alarms.npz` 逐位比对，不一致即退出。

## 3. 审计

| 命令 | 耗时 | 对应报告 |
|---|---|---|
| `python experiments/audit_reading_reliability.py` | 28 秒 | §五 读数可靠性 |
| `python experiments/build_complementarity_ledger.py` | 7 秒 | §二 互补性账本 |
| `python experiments/ablate_information_sources.py` | 2 分 | §六 信息来源消融 |

## 4. 闭环诊断

| 命令 | 耗时 | 对应报告 |
|---|---|---|
| `python experiments/diagnose_object_readability.py` | 16 秒 | §四 物体可读性 |
| `python experiments/diagnose_object_readability_failed.py` | 13 秒 | §四 剥离结局混淆 |
| `python experiments/verify_phase_readout.py` | 38 秒 | §四 相位可读 |
| `python experiments/verify_remaining_duration.py` | 13 秒 | §四 剩余时长（否定） |
| `python experiments/discover_regimes.py` | 1 分 15 秒 | §四 自监督分段（否定） |
| `python experiments/verify_graph_transfer.py` | 26 秒 | 限制 图迁移 |

## 5. 已作废的两轮检测器（保留可复现）

```bash
python experiments/select_operating_point_v12.py --output results/operating_point
python experiments/evaluate_phenotype_2x2.py --output results/phenotype
python experiments/select_loop_operating_point.py
python experiments/diagnose_phenotype_timing.py     # 含撤回声明
python experiments/replicate_loop_specificity.py
python experiments/characterise_v7_missed_loops.py
```

两轮的 `selection.json` 均记 `feasible: false`，不得改写。
`diagnose_phenotype_timing.py` 同时输出被撤回的 `*_mean_of_ratios` 列与正确的
`*_pooled` 列，后者为准。

## 6. 测试

```bash
pytest -q
```

303 个测试。若 `tests/test_event_sensor.py` 存在，它不属于本 bundle 的工作，
用 `--ignore=tests/test_event_sensor.py` 排除。

## 产物大小

`results/progress_cache/` 约 71 MB，已在仓库 `.gitignore` 中排除，两分钟可重建。
其余产物合计约 19 MB，随代码提交。
