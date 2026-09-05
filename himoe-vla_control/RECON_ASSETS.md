# 资产地图（2026-09-04 侦察，E 系实验的数据依据）

## 正典实现（必须复用，不自造）
- **onset 规则**：`himoe-vla_trap/code/analyze_trainfree_signal_matrix.py::physical_onsets`
  (L347-393)。loop=非局部回返最早 right；static=宽2滑窗连续7窗(=80 action)后+2。
  最新生成器 `analysis_moe_phenotype/events/build_events.py`（阈值同源，含 objaware 降级）。
- **V^late/A^route/sharpness**：同文件 `extract_row_features`(L151-235)。⚠ 它的 A^route
  未做 Δs 归一化、V 用 weighted-Jaccard——与 EXPERIMENT_PROGRAM 的归一化定义**不是同一个量**；
  E0 复现用旧版原样，E1/E2/H8 用 control_metrics 归一化版，两版并报。
- **官方 baseline**：`route_mobility_w4/w8`（新信号须对它残差化后存活）。
- **matched control 口径**：`all_no_event`（同组同绝对 query）; "22 条 controls" 实为
  语料 B scene3 的条件重复，不作独立单位。

## 事件库存（独立单位=trunk/episode）
| 语料 | loop | static | 备注 |
|---|---|---|---|
| A rolling-star t08 | 172 事件/仅 22 独立 trunk | 54 | dense 真值；trunk 只到 q0-5 有快照 |
| B right-16x32 (SCENE8) | 57 | 146 | query-proxy（A 上验证 sens1.0/spec0.94）|
| grid50x8（新语料） | 835（293 失败） | 180 | ⚠ 17,648/18,560 为 objaware 降级：**loop
  specificity 仅 0.077-0.081，loop 通道在降级格上无效**；static 代理可用(0.97+) |
| main16x32 | 204 | 158 | 同上分级 |
事件表：`analysis_moe_phenotype/events/{grid50x8,main16x32}/<suite>/<task>/events.csv`。

## q−2 前兆的现有效应（E0 基准 + 诚实上下文）
- 严格跨语料复现单元仅 loop 的 V@−2 与 A@−2（A 语料 AUC≈0.92, B≈0.85-0.89）。
- **最新 8 信号×7 lead 族 + mob1_w8 残差化后只有 A^route@−2 存活**
  （V@−2 maxT 0.071/0.225）——E0 预注册要正视。
- static: late_flow_volatility 方向相反（下降），lag_periodicity@0 AUC≈0.94。

## Snapshot-Fork 0/8（E2 的对照原点）
- 代码 `himoe-vla_trap/code/collect_snapshot_fork_recovery.py` + 专用服务器
  `serve_with_full_route_capture.py --store-full-probs --return-full-probs`。
- 唯一 loop trunk：`results/snapshot_fork_recovery/runs/init03_seed20260903_loop/`
  （onset q36，fork@32/34/36 各 0/8，配对流 key=9731，Wilson [0,0.324]）。
  fork_* 目录自带 full_state+policy_input（即 3 个现成可重放快照）。
- 结论措辞冻结："sensor positive, actuator negative/inconclusive"；0/24≠24 独立干预。

## E1/E2 缺口与来源
- 需新采 24-32 个 loop trunk × {q−4,−2,0,+2} 快照。**重放机制现成**：
  `delayed_fork_collect.py` 的 rewind[query]=(save_full_state,policy_obs,steps)
  (L324) / `collect_snapshot_fork_recovery.py` (L287-330,447-448)。
- 靶点先验：noise_sensitivity_map 的半开格（SCENE8 init 0/1/8/10/11/15/16/27/40/47 等
  37 个 3-5/8 格）；采集时自录 dense control_* 流自产 full-grade onset 标签。
- ⚠ wrist layout：rolling-star/trap 线=checkpoint-right；right-50x8 语料=paper-right。
  E1/E2 采集全程 checkpoint-right（与 0/8 与 A 语料可比）；50x8 的 map 仅作选格先验。

## 并行线协调（不冲突守则）
- `trap-recovery-depth-20260904/`（卡5在跑，5 臂等预算 20-query，200 trunk 计划）
  与 E5 高度重叠——**E5 设计必须等预算**（不等预算时 delayed 臂算术上不可能成功，
  最快救回 14 步）。开 E5 前先看它的结果，避免重复采。
- 我方一切写入仅在 himoe-vla_control/ 与其 runs/；种子命名空间 "e1/e2/..." 独立；
  不触碰 himoe-vla_trap / analysis_moe_phenotype / trap-recovery-depth 的任何文件。
