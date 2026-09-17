# 2026-09-17 快照：MoE 报警驱动的控制实验（09-16 ~ 09-17）

来源：本地 8 卡 H20 机器 `/home/swj/data`（DSW 容器），同一份内容已 rsync 到 funhpc A100 机器 `/data`。
分支 `backup/2026-09-17` 从 `backup/2026-09-09` 分出，只新增本目录，不改动其它路径。

## 本目录里有什么（路径与 `/home/swj/data` 下一致）

| 路径 | 内容 |
| --- | --- |
| `libero-runtime/*.py` | 控制实验客户端与分析脚本：`control_branch.py`（分支重放 + 全部策略/触发器）、`run_control_episode_v1–v4.py`、`run_control_experiment.py`、`run_control_scale.py`、`run_parallel_batch.py`、`analyze_*`、`plot_*`、`rqa_topology.py`、`probe_flow_path_straightness.py` 等 |
| `libero-runtime/controls/` | `REPORT.md`、`SUMMARY.md`、`ANALYSIS.md`（v8.2→kick 的置信区间/确认种子/触发器统计）、13 条 `exp-*-chain.sh`、`parents-*.json`、`reachability_sweep.py`、`failure_anatomy.py`、`summarize_controls.py` |
| `libero-runtime/controls/exp-*/` | 54 个实验：`report.json`、`analysis.json`、图，以及每条分支的 `result.json`（逐 query 决策） |
| `libero-runtime/controls/anatomy/`、`pro-base/` | 失败分型与 Pro 基线的 JSON |
| `libero-runtime/samples/*-2026091{6,7}/` | 四份中文报告：去噪路径直线性、弯曲 vs 陷阱、复返定量分析（RQA）、复返率触发的并行 control 迭代、hold16 推广实验 |
| `libero-runtime/simulations/*/` | `control-r1..r3`、`control-scale`、父轨迹集 `topo-all-i`/`topo-libero10`/`topo-plus` 的顶层 `report.json`/`summary.csv` |
| `moe-capture/topo-20260916/_grid/` | 拓扑定义网格的 `REPORT.md`、`VALIDATION.md`、`auroc-table.csv` |
| `srv/` | 三个策略服务端（`serve_with_recorder.py`、`serve_moe_capture.py`、`serve_moe_batch.py`）、`bench_batch_infer.py`、`src/himoe_libero_bridge/`（含 `intervene/as_expert`、`flow/path_actions` 协议扩展） |
| `bin/` | `himoe-server`、`himoe-batch` 服务管理脚本（路径写死为 `/home/swj/data`） |
| `coding/moe-control-experiments/` | P3h/P3i/P3j 阶段的 README/STATUS/PLAN 与代码、运行汇总 JSON |

## 主要结论（详见 `libero-runtime/controls/REPORT.md` 与 `ANALYSIS.md`）

- 候选 chunk 选择规则（路由、去噪路径、聚类、边界、v8.2 最小化）全部不优于随机重采样：flow 策略对噪声近乎确定。
- 有效的干预是物理恢复原语 `kick`（松爪 + 抬升，3 个 chunk 后交还策略）。在线 v8.2 报警触发在 LIBERO-Plus 上 +6.7 个百分点
  （确认种子 700 配对，聚类 bootstrap CI [+0.7, +13.0]，按父轨迹 t 检验 p=0.034）；LIBERO-10 原版无效（−0.8）；Pro swap 0/74。
- LIBERO-10 的失败在事后分支上可救（报警前 16 query 处 kick 16/62），在线失效的原因是 v8.2 报警偏晚（中位 q34）。
- 温和干预（增益、部分去噪、观测扰动、AS 专家切换）与随机无差别；kNN 触发提前到 q17 但误触发翻倍，净值等于 v8.2。
- 路由可达性扫描：209 种强制路由组合没有一种能把背向目标的动作方向扭转；专家是同一行为的精修，不是备选行为。

## 没有放进 GitHub 的内容

- 每个 episode 的 `episode-trace.npz`、`control.npz`（逐 query 的 RR/直径/候选/路由点）、`routes.zarr`、录像 `mp4`、
  `moe-capture` 的逐步激活 npz（136 GB）、`_grid/features.pkl`（0.6 GB）、kNN 参考库 `knn-bank-success.npz`（78 MB，npz 规则排除）。
  这些全部在 funhpc 机器 `/data` 下的同名路径（moe-capture npz 除外，仅本地）。
- 排除规则同 09-09 备份：扩展名 npz/npy/zarr/pkl/mp4/log/gz/tgz/tar/zip 及大于 95 MiB 的文件。
