# Robot Whisper

这是 `himoe-vla` 在 **2026-09-01 至 2026-09-05（含首尾）** 的实验与内容快照。
仓库保留可读、可审计和可复现实验所需的代码、协议、测试、报告、汇总表、图以及演示材料；
体积很大的原始路由语料、轨迹、模型权重和中间特征没有复制到 GitHub。

## 主要内容

| 日期 | 主题 | 入口 |
| --- | --- | --- |
| 09-01 | MoE 机制、token/层路由统计与汇报材料 | [`PPT_MOE_MECHANISM.md`](PPT_MOE_MECHANISM.md)、[`himoe-vla-moe-value-ppt-2026-09-01.md`](himoe-vla-moe-value-ppt-2026-09-01.md)、[`ppt-assets/`](ppt-assets/) |
| 09-01 ~ 09-03 | token dynamics、位置 PCA、replan boundary 与无训练信号矩阵 | [`VLA_MUI_HUB/`](VLA_MUI_HUB/)、[`analysis_signal_matrix_trainfree/`](analysis_signal_matrix_trainfree/)、[`analysis_kolmogorov_trainfree/`](analysis_kolmogorov_trainfree/) |
| 09-03 ~ 09-04 | double selector、MoE routing phase portrait 与 Trap taxonomy | [`double-selete/`](double-selete/)、[`analysis_moe_phenotype/`](analysis_moe_phenotype/)、[`analysis_trap_taxonomy/`](analysis_trap_taxonomy/) |
| 09-04 | committor/驻留逃逸、控制实验、恢复深度与 v4/v5 在线报警 | [`analysis_committor/`](analysis_committor/)、[`himoe-vla_control/`](himoe-vla_control/)、[`trap-recovery-depth-20260904/`](trap-recovery-depth-20260904/)、[`moe-v4-0904/`](moe-v4-0904/) |
| 09-05 | spatial guard v6、MoE routing grammar 与多 GPU 在线服务 | [`moe-v6-0905/`](moe-v6-0905/)、[`MoE-grammar/`](MoE-grammar/)、[`online-servers/`](online-servers/) |
| 09-01 ~ 09-02 | Robot Whisper 演示、LaTeX/PDF/PPTX 与浏览器查看器 | [`demo/`](demo/) |

## 快速阅读

- 总体 Trap 实验索引：[`himoe-vla_trap/README.md`](himoe-vla_trap/README.md)
- MoE 表型报告：[`analysis_moe_phenotype/REPORT.zh.md`](analysis_moe_phenotype/REPORT.zh.md)
- committor 与逃逸报告：[`analysis_committor/REPORT.zh.md`](analysis_committor/REPORT.zh.md)
- v4/v5 在线报警：[`moe-v4-0904/README.md`](moe-v4-0904/README.md)
- v6 spatial guard：[`moe-v6-0905/README.md`](moe-v6-0905/README.md)
- routing grammar：[`MoE-grammar/README.md`](MoE-grammar/README.md)

## 数据与复现

多数分析默认读取原工作区中的 `VLA_MUI_HUB` 路由语料。GitHub 快照只包含分析程序和压缩后的
结果证据，不包含 `cache/`、`cache_new/`、Zarr 路由、在线 rollout、模型权重或大型特征矩阵。
因此，报告和表格可直接审阅；完整重跑需要按各子目录 README 配置对应数据源。

[`himoe-vla_trap/MANIFEST.json`](himoe-vla_trap/MANIFEST.json) 和
[`himoe-vla_trap/SHA256SUMS`](himoe-vla_trap/SHA256SUMS) 描述的是原始完整本地 bundle。
本仓库是精简公开快照，未承诺包含清单里的每一个大体积产物。具体边界见
[`SNAPSHOT_SCOPE.md`](SNAPSHOT_SCOPE.md)。
