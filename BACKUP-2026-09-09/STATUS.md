# 备份实际完成情况（2026-09-09，换服务器前约 30 分钟内完成）

这份说明记录 `backup/2026-09-09` 分支和 `backup-2026-09-09` 预发布里**实际**有什么，
以 README.md 描述的计划为准的部分请以本文件为准。

## 分支 `backup/2026-09-09` 已包含

1. 工作分支 `publish-moe-routing-0906` 的全部提交（HEAD 8e37b8d，含此前未推送的 45 个提交，已同步推到同名远端分支）。
2. 工作区 4 个已修改文件的当前版本。
3. 未跟踪且未被忽略的源码、报告、配置和小结果，按批次：
   - 批次 1：除 moe-trap-control、safe&vlaconf、himoe-route-capture 外的全部目录，以及 `BACKUP-2026-09-09/` 清单。
   - 批次 2：himoe-route-capture。
   - 批次 3：moe-trap-control（约 12.8 万个 JSON 及源码、报告）。
   - 批次 4b：safe&vlaconf 中小于 20 MiB 的文件。
   - 批次 5b：safe&vlaconf 中 20 到 95 MiB 的文件（若本文件之后的提交记录里有它，说明推送成功）。
   - 提交 "(4)" 和 "(5)" 是空提交，因脚本 bug 未加入任何文件，可忽略。
4. 排除规则同 README.md：npz/npy/gz/tar/zip 等数据扩展名和大于 95 MiB 的文件不在分支里，见 `git-exclude.tsv`。

## 预发布 `backup-2026-09-09` 实际包含

- 仅 `himoe-vla-cache_himoe-libero-bridge.tar.zst.part000`（1 GiB，SHA-256 见 parts-manifest.tsv）。
  这是 formal-artifacts / paper-alignment / metadata 打包流的前 1 GiB，后续分卷因时间不足未上传。
  用 `cat part000 | zstd -d | tar -x` 可以解出流内完整的文件，最后一个被截断的文件会报错，属预期。
- 其余约 93 GiB 原始数据**没有上传**。逐文件 SHA-256 在分支的 `BACKUP-2026-09-09/manifest-raw-data.tsv.zst`，
  迁移后如原目录仍在，用它核对；如原目录丢失，这些数据不可从本备份恢复。

## 未备份

- 五份 HiMoE 权重、Python/uv/conda 环境、`himoe-vla-cache` 中的上游代码和数据集。
- 十个嵌套仓库和 worktree 的内容（各自已推送到 weijiansh1/himoe-libero-bridge，见 nested-repos.tsv）。
- `/home/jovyan/work` 中 himoe-vla 以外的目录。
