# Snapshot Scope

## 时间边界

- 主时间窗：2026-09-01 00:00:00 至 2026-09-05 当前快照时刻（UTC）。
- 新实验目录按整体纳入；为保持相对导入和复现说明可用，也保留了少量早于窗口但被新实验依赖的源码。
- 原工作目录起初不是 Git 仓库，因此时间窗依据文件修改时间、目录命名、README 和实验报告交叉确认，
  不是依据此前不存在的 commit history。

## 已纳入

- Python/Shell/JavaScript/LaTeX 源码和测试。
- 预注册协议、实验计划、README、中文报告和审计说明。
- JSON/CSV 汇总、阈值、指标、选择结果及小型派生数据。
- 报警记录、matched AUC、fork/outcome assurance 等非隐藏层实验数据。
- PNG/PDF 图表，以及 Robot Whisper 的 PDF/PPTX 演示成品。
- 为解释或运行结果所需、且体积合理的 NPZ 数据。
- v7-v11、actual-execution signals 和扩展 routing-grammar 的代码、报告及聚合证据。

## 未纳入

- `VLA_MUI_HUB/cache*` 下的原始路由和轨迹语料。
- Zarr、模型权重、检查点、完整 rollout、视频与大型特征矩阵。
- 逐 query 的隐藏层激活和可重建稠密 profile，包括 v8 `results/profiles`、v9
  `results/dynamic_profiles`、v10 `results/structural_profiles`、v11 的 profile 目录和
  `MoE-grammar/artifacts`；对应构建清单、哈希和聚合结果仍保留。
- `analysis_moe_execution_signals` 的三份行级 NPZ 缓存；CSV/JSON 统计、图和报告仍保留。
- `node_modules`、`.next`、Python 缓存、测试缓存、PID 和运行日志。
- 真实 `.env`、本机 Claude 设置、嵌套 Git 元数据和临时构建文件。
- 重复的 `double-selete/trainfree/results/*/` 历史中间目录；其冻结后的核心版本位于
  `moe-v4-0904`、`moe-v6-0905` 和 `himoe-vla_trap`。
- `analysis_signal_matrix_trainfree/recovery_window/` 在线采集原件；仓库保留汇总 CSV、图和报告。

## 注意事项

- 文档中的 `/home/jovyan/work/himoe-vla/...` 是原实验机路径，用于记录数据来源，不是仓库内路径。
- 精简快照保留结论证据和运行代码，但不能替代完整本地归档。
- 所有排除项仍保存在原工作区；整理过程未修改原实验文件。
