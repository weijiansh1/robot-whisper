# LIBERO-Pro / LIBERO-Plus 加载记录

2026-09-08 UTC：两个 benchmark 已安装并接入现有 HiMoE 七卡模型服务，完成资源核验和实际模拟器验证。机器可读结果见 [READINESS.json](READINESS.json)。

| 项目 | Pro | Plus |
| --- | --- | --- |
| 注册任务读取检查 | 200 / 200 | 10,030 / 10,030 |
| 每任务可用初始状态下限 | 50 | 1 |
| 代表组合运行检查 | 20 / 20 | 28 / 28 |
| 真实模型推理次数 | 40 | 56 |
| 策略执行环境步数 | 400 | 560 |

每个代表组合执行 10 步 settle 和 20 步策略动作，检查主相机、腕部相机、非空像素、模型身份、显式 flow noise 回执、有限动作及模拟器状态。覆盖四个基础 suite 与所有标准扰动类型，两组检查均覆盖物理 GPU `0,1,2,3,4,5,7`。这是安装验证，不是完整成功率评测或正式 MoE-Control 数据采集。

GPU 6 被启动参数白名单排除；`--gpus 6` 会在创建任何环境之前报错。最终检查 GPU 6 为 1 MiB、0% 利用率。模型仍使用原先的 35 个常驻实例，本次没有重启、卸载或再次加载模型。验证用模拟器进程已退出；后续任务按需创建模拟器并调用常驻模型服务。

## 安装位置与来源

资源根目录：`/home/jovyan/work/himoe-vla/himoe-vla-cache/libero-extensions/`。

- `LIBERO-PRO/`：代码与 Pro 任务资源。
- `LIBERO-plus/`：代码与 Plus 完整资产。
- `config/pro/`、`config/plus/`：分别指定对应的 BDDL、init 和 assets 路径。
- `python-extras/`、`system-libs/`：Plus 的图像扰动依赖。
- `provenance/`：固定提交的 GitHub 文件清单。

两个 benchmark 共用现有 Python 3.8 基础依赖，通过独立进程、`PYTHONPATH` 和 `LIBERO_CONFIG_PATH` 选择各自代码。原有标准 LIBERO 安装保持原路径。额外安装 Wand 0.6.13、scikit-image 0.21.0 及其缺少的依赖，并在上述私有目录部署 ImageMagick 动态库。

固定 [Pro 代码提交](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec) 与 [Plus 代码提交](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182)，分别核验 140 和 141 个源码、配置文件的 Git blob 哈希。

[Pro 数据](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/tree/c86fc3b8293185a6f373677018ff3e37f8391602)固定版本 `c86fc3b8293185a6f373677018ff3e37f8391602`，676 个文件全部核验。下载也包含新七类扩展的额外文件，本次 200 条标准注册项验证不代表已接入该额外扩展。

[Plus 资产](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/dd2bd61b7d9a6fef1abc52d606e983b41886a149)固定版本 `dd2bd61b7d9a6fef1abc52d606e983b41886a149`。`assets.zip` 为 6,395,849,578 bytes，SHA-256 为 `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`，已与发布的 LFS 对象哈希一致性核对。解压读取全部 ZIP 条目并通过 CRC 检查。发布包带作者机器目录前缀，安装器已将其规范放到 `LIBERO-plus/libero/libero/assets/`。

安装后总磁盘占用约 **10.31 GiB**，其中 Plus 资产逻辑大小 8.34 GiB；大量小文件会增加实际分配空间。检查时工作盘仍剩约 40 GiB，原方案 25 GiB 采集配额可保留，开始采集前仍需复查其他任务的空间增长。

## Pro 环境条目的限制

官方 GitHub 代码和下载数据未包含 40 个标准 `*_env` 条目的预生成 BDDL/init。本次调用固定提交中的 `BDDLCombinedPerturbator` 生成 BDDL，并为每个任务以 seed 7 执行 50 次无渲染 `ControlEnv.reset()`，保存 float64 初始状态。重用环境时设置 `hard_reset=False`；每个文件均验证有 50 个不同、有限的状态。它们是**本地生成的初始化样本**，不是作者发布的预生成文件。

该提交的环境替换实现固定使用 `living_room_table`。Libero-10 中已有 5 个 living-room 任务的 BDDL 因而与原任务相同，已在 `READINESS.json` 的 `pro_environment_generation.bddl_unchanged_from_base` 中明确列出。200 条注册项可用于任务表对齐；统计实际 OOD 干预收益时，这 5 条应单列为原场景对照，不能当作发生了环境变化。按其余 195 条采样时，Pro 初筛为 975 条，覆盖方案为 3,900 条；原先 1,000 / 4,000 条预算仍可作为上限。

40 个生成条目的文件哈希、种子及生成方式见 [生成记录目录](runs/pro-env-generation/pro)。环境变化后的长期物理稳定性、报警状态完整恢复及干预有效性仍属于正式实验预检。

## 使用

在 `moe-trap-control` 目录执行。默认允许 GPU 列表已经排除物理卡 6。

```bash
# 只检查全部注册任务和初始状态，约束不涉及模型推理。
python benchmarks/run_benchmarks.py inventory

# 重跑四个 suite 与所有标准扰动类型的短程安装验证。
python benchmarks/run_benchmarks.py smoke --gpus 0,1,2,3,4,5,7

# Pro：Goal 的第 0 个语义扰动任务，完整基准时限，调用 GPU 0。
python benchmarks/run_benchmarks.py run --benchmark pro --suite libero_goal --category Semantic --task-index 0 --gpus 0

# Plus：Goal 的第 0 个注册变体，调用物理 GPU 7。
python benchmarks/run_benchmarks.py run --benchmark plus --suite libero_goal --task-index 0 --gpus 7
```

每次输出写到 `benchmarks/runs/` 下的新目录；可用 `--output` 指定位置。`--variant` 可直接使用 `design/benchmark_inventory.json` 中的变体 ID。原始注册表中的任务名称和索引保持固定，模型输入指令从实际 BDDL 读取，避免把文件名里的视角、纹理参数当成指令。

模型端口为 `8800 + 10 * 物理GPU编号 + suite偏移`，Goal/Spatial/Object/Long 偏移为 0/1/2/3。模型检查同时验证 suite、物理 GPU 和 checkpoint 哈希。模拟器进程不加载 CUDA 模型，EGL 渲染明确选用允许的物理 GPU。

启动脚本目前提供基线 rollout 和安装验证，未实现报警快照后的分叉调度或路由干预。没有采集 hidden，也没有把安装验证结果写入正式采样清单。原模型服务的共享进程吞吐限制仍见 [GPU 报告](../GPU_PRELOAD_READINESS.zh.md)。

验证结果：[Pro 全量资源](runs/inventory-pro/pro/result.json)、[Plus 全量资源](runs/inventory-plus/plus/result.json)、[Pro 模拟器](runs/smoke-pro/summary.json)、[Plus 模拟器](runs/smoke-plus/summary.json)。各模拟器结果目录包含双相机截图、动作请求日志及运行结果。
