# 2026-09-09 换服务器前的应急备份

来源：`/home/jovyan/work/himoe-vla`，分支 `publish-moe-routing-0906`，HEAD `8e37b8db1a35115a22c0787083b11b2485f47c15`。
这不是整理后的发布版本，只是为了在换物理服务器前把不可重建的内容存到 GitHub。

## 分支 `backup/2026-09-09` 里有什么

- 上述 HEAD 的全部已提交内容。
- 工作区 4 个已修改文件的当前版本：`.gitignore`、`MoE-grammar/moe_grammar/run_channel_comparison_audit.py`、
  `MoE-grammar/results-channel-comparison/summary.json`、`analysis_moe_execution_signals/README.md`。
- 所有未跟踪且未被 `.gitignore` 排除的源码、报告、配置和小型结果，
  排除扩展名 npz/npy/gz/tgz/tar/zip/7z/h5/hdf5/mp4/avi/mkv/bin/parquet/arrow 以及大于 95 MiB 的文件。
  排除清单见 `git-exclude.tsv`。
- 本目录的清单文件。

## Release `backup-2026-09-09` 里有什么

原始数据按目录打包成 `<组名>.tar.zst.partNNN`，每卷 1 GiB。组名中的 `/` 和 `&` 替换成了 `_`。
还原某一组：

    cat <组名>.tar.zst.part* | zstd -d | tar -x -C himoe-vla

`parts-manifest.tsv` 记录每个分卷的字节数和 SHA-256。`manifest-raw-data.tsv.zst` 记录
每个原始数据文件的 SHA-256、字节数和 mtime，无论它是否已上传，用于迁移后核对。
`pack-order.lst` 是上传顺序；`groups-done.lst` 是已完整上传的组。

## 没有备份的内容

- 五份 HiMoE 权重（各 8,138,322,389 字节）和所有 Python/uv/conda 环境缓存。
  权重的 HF revision 与 LFS SHA 见 `himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-lineage-audit-20260803.json`。
- `himoe-vla-cache/` 中除 `himoe-libero-bridge/{formal-artifacts,paper-alignment,metadata,moevla-data}`
  和 `vla-adapter-repro/results` 以外的内容（上游代码、数据集、环境），见 `skipped-ignored.lst`。
- 十个嵌套 Git 仓库和 worktree 的内容。它们各自已推送到远端且无未提交修改，见 `nested-repos.tsv`。
- `/home/jovyan/work` 中 himoe-vla 之外的目录。
