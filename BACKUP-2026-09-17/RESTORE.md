# /data 复原说明（HiMoE-VLA + LIBERO 策略服务与控制实验）

写于 2026-09-17。目标：把这份 `/data` 搬到任何一台新的 GPU 服务器后，能重新跑起策略服务、评测客户端和 MoE 控制实验。
更细的历史记录见 `README.replica.md`（09-16 从本机复刻到 H20 机器的全过程），本文只讲"怎么复原"。

## 0. /data 里有什么

| 路径 | 作用 | 体积 |
| --- | --- | ---: |
| `srv/` | 策略服务端：`serve_with_recorder.py`（评测+路由记录）、`serve_moe_capture.py`（抓 MoE 激活）、`serve_moe_batch.py`（批推理，8 倍吞吐）；桥接层 `src/himoe_libero_bridge/`；`packages/openpi-client`；`third_party/` | 91 MB |
| `himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10/` | 模型权重 `pytorch_model.pth`（8,138,322,389 字节）+ `checkpoint.env` + `stats.json` | 7.6 GB |
| `venv311/` | 服务端 Python 3.11 环境（torch、模型代码依赖） | 7.7 GB |
| `libero-runtime/envs/libero/` | 客户端 Python 环境（MuJoCo、robosuite、LIBERO） | 1.9 GB |
| `miniconda/envs/torch/` | 分析用 conda 环境（numpy/scipy/matplotlib/zarr…） | 6.1 GB |
| `libero-runtime/upstream/{LIBERO-PRO,LIBERO-plus}`、`dependencies/`、`configs/` | 三个基准的任务定义、初态、扰动 | 12 GB |
| `libero-runtime/*.py` | 客户端与实验驱动：`run_benchmark.py`、`run_parallel_batch.py`、`control_branch.py`、`run_control_*.py`、分析/画图脚本 | — |
| `libero-runtime/controls/` | 控制实验：`REPORT.md`、`SUMMARY.md`、`ANALYSIS.md`、`exp-*/`（54 个实验结果）、`parents-*.json`、`knn-bank-success.npz` | 0.6 GB |
| `libero-runtime/simulations/topo-all-i/` | 控制实验的 170 条父轨迹（`episode-trace.npz`），`parents-topo-all-i.json` 指向它 | 0.97 GB |
| `libero-runtime/simulations/control-*/` | 三轮在线控制迭代与推广实验，逐 query 数据 `control.npz` | 0.9 GB |
| `libero-runtime/samples/*-2026091{6,7}/` | 去噪路径、复返分析、控制迭代、推广实验的中文报告与数据 | — |
| `moe-capture/topo-20260916/_grid/` | 拓扑定义网格：`REPORT.md`、`VALIDATION.md`、`features.pkl` | 0.76 GB |
| `coding/robot-whisper-0909/` | GitHub 仓库 `weijiansh1/robot-whisper` 的克隆（含 `moe-trap-control/v82_closed_loop.py`，控制实验依赖） | — |
| `coding/moe-control-experiments/` | P3h/P3i/P3j 阶段的协议、代码与汇总 | — |
| `env.sh`、`serve.sh`、`serve.local.sh`、`_local-bin/`、`_manifests/` | 环境入口、服务启动脚本、管理脚本、包清单 | <1 MB |

## 1. 新机器要求

- NVIDIA GPU，单实例约 22 GB 显存（batch-1）或 24–26 GB（批推理 batch 32）；驱动支持 CUDA 12。
- Ubuntu 22.04/24.04，x86_64。CPU 核数决定评测吞吐（MuJoCo 渲染在 CPU 上，OSMesa 约 44 ms/帧）。
- 磁盘 ≥ 60 GB（上表合计约 40 GB，留出输出空间）。

## 2. 拷贝

    rsync -aH --info=progress2 -e "ssh -p <port>" root@<本机>:/data/ /data/

放到新机器的 `/data` 下可以省掉第 4 步的路径重定位。不要复制 `/data/run0`、`moe-capture/topo-*/server-p*`（都是旧输出）。

## 3. 系统库（需要 root；无 root 见 `README.replica.md` "无 root 的补救"）

    apt-get install -y libosmesa6 libgl1 libglvnd0 libegl1 libglx0 libglu1-mesa libopengl0 \
        libimage-exiftool-perl ffmpeg imagemagick-6.q16 libmagickwand-6.q16-6 llvm-15-runtime

完整清单在 `_manifests/apt.manual.txt`（原机所有手工安装包）和 `_manifests/dpkg.txt`。
渲染后端：`MUJOCO_GL=osmesa` 在任何机器都能跑；`MUJOCO_GL=egl` 快 30 倍但需要 NVIDIA EGL vendor 文件
（`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`），且像素与 OSMesa 不逐位相同——同一批实验内不要混用。

## 4. Python 环境

三个环境都是原样同步的目录，路径写死在 `pyvenv.cfg` 和 `bin/*` 的 shebang 里。

- 放在 `/data` 下：不用动。
- 放在别处（例如 `$NEW`）：对 `venv311` 和 `libero-runtime/envs/libero` 各跑一次

      _manifests/relocate-venv.sh $NEW/venv311 /data $NEW <基础 python 所在 bin 目录>
      _manifests/relocate-venv.sh $NEW/libero-runtime/envs/libero /data $NEW <基础 python 所在 bin 目录>

  conda 环境用 `conda-unpack` 或按 `_manifests/torch.conda.yml` 重建。
- 从零重建（不搬目录）：`srv/pyproject.toml` + `srv/uv.lock`（服务端），`_manifests/libero_libero.pip.txt`（客户端），
  `_manifests/torch.conda.yml`（分析）。注意 HF 权重下载在国内很慢，权重仍应从本机 rsync。

## 5. 环境入口

`env.sh` 是本机版本（`DATA=/home/swj/data`，并把无 root 补装的库加进 `LD_LIBRARY_PATH`）。新机器：

    cp env.sh env.local.sh && sed -i 's#^export DATA=.*#export DATA=/data#' env.local.sh
    # 若系统库是 apt 正常安装的，删掉 env.local.sh 里 ~/.local/syslib 和 MAGICK_* 相关行
    source /data/env.local.sh

`_local-bin/himoe-server`、`himoe-batch`、`himoe-capture` 顶部的 `DATA=/home/swj/data` 同样改成 `/data`，然后放进 PATH。

## 6. 启动策略服务

三种服务端协议相同（WebSocket，`PolicyServer`/`PolicyClient`），同一权重，按端口区分，GPU 号与端口末位一致。

| 用途 | 启动 | 端口 |
| --- | --- | --- |
| 单实例、评测 + 路由记录 | `bash serve.sh`（原机脚本，9500）或 `himoe-server start [gpu] [port]` | 9500 / 9510+ |
| 8 卡各一个 | `himoe-server start-all 8` | 9510–9517 |
| 批推理（大规模评测、控制实验） | `himoe-batch start-all none 8 9570 32` | 9570–9577 |
| 抓 MoE 激活（每请求约 10 MB npz） | `himoe-capture ...` 或直接 `serve_moe_capture.py --port 952x --gpu x --out <dir>` | 9520–9527 |

要点：

- GPU 选择必须走服务端的 `--gpu`，不能只设 `CUDA_VISIBLE_DEVICES`。
- 单卡 40 GB 只能放一个实例；批推理 `max_batch` 按显存调（A100-40G 建议 8）。
- **批推理模式永远用 `none`**（不落盘）。带 `--capture-out` 时每个请求写 8 MB，控制实验曾在一夜之间写满 4.7 TB。
- 批推理与 batch-1 的动作有约 1e-2 差异，结果只能在同一种服务内比较。
- 就绪标志：日志出现 `serving on ws://...`，模型加载约 1–2 分钟。

## 7. 冒烟测试（客户端）

    source /data/env.local.sh; cd /data/libero-runtime
    MUJOCO_GL=osmesa envs/libero/bin/python run_benchmark.py --benchmark pro --perturbation none \
        --task-id 0 --init-state-id 0 --port 9510 --output-root /tmp/smoke

成功标志：`/tmp/smoke/episode-*/summary.json` 存在且 `status: completed`。首次运行 LIBERO 会写 `~/.libero/config.yaml`，
多进程并发首次运行有写入竞争，驱动脚本已内置重试。

参考结果（本机 09-16，flow seed 42）：LIBERO-10 原版约 60%，Plus 约 36%，Pro swap 0%。

## 8. 复现控制实验

    himoe-batch start-all none 8 9570 32
    cd /data/libero-runtime
    python3 run_control_experiment.py --parents controls/parents-topo-all-i.json --ports 9570,...,9577 \
        --branch online --trigger v82 --include-successes --control-queries 3 --seeds 0,1 \
        --strategies native,kick --out controls/<name>
    python3 controls/summarize_controls.py      # 汇总到 controls/SUMMARY.md

`parents-topo-all-i.json` 里的路径是 `/home/swj/data/...`，新机器先 `sed -i 's#/home/swj/data#/data#g'`。
`control_branch.py` 依赖 `coding/robot-whisper-0909/moe-trap-control/v82_closed_loop.py`（已含）和
`controls/knn-bank-success.npz`（kNN 触发器）。全部策略与触发器的定义在 `control_branch.py` 开头。

## 9. 没有放在 /data 里的内容

- `moe-capture` 的逐步激活 npz（136 GB）、`routes.zarr`（7.8 GB）、episode 录像：只在 H20 机器 `/home/swj/data`。
  它们是分析产物，不是复现前提。
- 其它父轨迹批次 `topo-all-c…h`：同上，只有 `topo-all-i`（控制实验实际用的）在这里。
- 代码的公开副本：GitHub `weijiansh1/robot-whisper` 分支 `backup/2026-09-17`，目录 `BACKUP-2026-09-17/`（无权重、无 npz）。

## 10. 已知的坑

1. `import libero` 需要把 `libero-runtime/upstream/LIBERO-PRO`（或 `LIBERO-plus`）加进 `sys.path`，客户端脚本已处理。
2. 镜像自带的 `LD_LIBRARY_PATH` 若含系统 torch 库目录，会让 venv 里的 torch 崩溃；`env.sh` 已去掉，新机器留意。
3. 容器有 `HTTP_PROXY` 时，探测本机端口要 `curl --noproxy '*'`，Python 用 `urllib.request.ProxyHandler({})`。
4. 在 shell 里 `pkill -f <模式>` 时模式不要与当前命令行本身匹配，否则会杀掉自己的 shell。
5. LIBERO-Plus 依赖 ImageMagick 6（Wand）；apt 装 `imagemagick-6.q16` 即可，无 root 时见 `env.sh` 的 `MAGICK_*`。
