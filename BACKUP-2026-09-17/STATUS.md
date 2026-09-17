# 备份实际完成情况（2026-09-17）

分支 `backup/2026-09-17`，从 `backup/2026-09-09`（HEAD 5da830a3）分出，已推送到 `origin`。

| 提交 | 内容 | 文件数 |
| --- | --- | ---: |
| `61eb0fa8` (1) | 控制实验代码、报告、汇总 JSON/CSV、图、本目录 README | 1,108 |
| `f55a4861` (2) | 54 个 `controls/exp-*` 实验的逐分支 `result.json` | 22,558 |
| (3) | 本文件 | 1 |

- 来源机器：DSW 容器 `/home/swj/data`（8 卡 H20）。推送方式：`gh auth login` 设备码登录 + HTTPS。
- 排除：npz/npy/zarr/pkl/mp4/log/gz/tgz/tar/zip 及 >95 MiB；运行日志（`driver.log`、`client.log`）也未纳入。
- 同一天 rsync 到 funhpc A100 机器 `/data` 的内容比本分支多：`topo-all-i` 等父轨迹 `episode-trace.npz`、
  3,279 个 `control.npz`、`_grid/features.pkl`、flow-path X/V npz、`knn-bank-success.npz`。
- 仅留在本地 H20 机器：`routes.zarr`（7.8 GB）、录像、`moe-capture` 逐步激活 npz（136 GB）。
