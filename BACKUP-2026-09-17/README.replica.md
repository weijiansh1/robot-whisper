# 远程 A100 机器环境复刻说明

来源: `<funhpc-A100-host>（主机名见本地 ~/data/README.replica.md）`（Ubuntu 22.04, A100-40G）
目标: 本机 `/home/swj/data`（Ubuntu 24.04, 2x H20, 无 root 权限，因此无法创建 `/data`）

## 路径映射
| 远程 | 本机 |
|---|---|
| `/data/...` | `/home/swj/data/...`（`$DATA`） |
| `/root/`（dotfiles，不含 .cache） | `/home/swj/data/_root/` |
| `/usr/lib/code-server` | `/home/swj/.local/lib/code-server` |
| `/root/.codex` | `/home/swj/.codex` |
| `/root/.config/code-server/config.yaml` | `/home/swj/.config/code-server/config.yaml` |

## 环境
- `source ~/data/env.sh`（已挂到 ~/.bashrc）: 激活 conda `torch`、nvm/node 20、npm-global(codex)、OPENAI_* 变量
- `$DATA/miniconda` : 本机全新安装 Miniconda3-py312_24.5.0-0，`envs/torch` 用 conda-pack 从远程搬运
- `$DATA/venv311`   : 远程原样同步 + 路径迁移（Python 3.11.16, torch 2.6.0+cu124, jax 0.5.0）
- `$DATA/libero-runtime/envs/libero` : 同上（Python 3.8.20，LIBERO 可编辑安装）
- `$DATA/envs/model`: conda-pack 搬运（Python 3.11.15，几乎空）
- 远程构建记录见 `$DATA/inst.sh reuse.sh go.sh go2.sh`，本机版启动脚本 `$DATA/serve.local.sh`

## 包清单（用于核对/重建）
`$DATA/_manifests/`: conda env export、pip freeze、dpkg/apt 列表等

## 验证结果（2026-09-16）
- venv311: Python 3.11.16, torch 2.6.0+cu124, `torch.cuda.is_available()=True`（H20），transformers 4.48.1, jax 0.5.0；
  `moevla / himoe_libero_bridge / openpi_client` 按 serve.local.sh 的 PYTHONPATH 可导入
- conda torch: Python 3.10.16, torch 2.6.0+cu124, CUDA 可用；`conda activate torch` 正常
- libero env: Python 3.8.20, mujoco 3.2.3, robosuite 1.4.0；`MUJOCO_GL=osmesa` 离屏渲染通过
  （EGL 在远程同样报 EGLError，远程实际也只有 OSMesa 可用；`import libero` 在远程同样失败，需脚本自行加路径）
- node v20.20.2 / npm 10.8.2 / codex-cli 0.154.0（来自同步的 nvm + npm-global）
- code-server 4.96.2（`code-server` 命令，配置沿用远程 0.0.0.0:8080 + 密码）
- uv 0.12.7 在 ~/.local/bin

## 无 root 的补救
- rsync: `~/bin/rsync`（apt 包解包到 ~/.local/rsync）
- OpenGL/EGL/OSMesa/LLVM 运行库: 解包到 `~/.local/syslib`，由 env.sh 设置 LD_LIBRARY_PATH / __EGL_VENDOR_LIBRARY_DIRS
- ~/.npmrc 去掉了 prefix=（nvm 与 prefix 冲突会每次告警），codex 仍通过 PATH 里的 npm-global/bin 使用

## 再次同步
`~/bin/sync-remote`：增量同步远程 /data 与 /root（已排除本机重建的环境目录），日志在 `_sync-logs/`
`_installers/` 保留了 Miniconda 安装器、conda-pack 的 torch.tar.gz / model.tar.gz、uv 压缩包，可重复使用

## 与远程的差异
- 系统级 apt 包（ffmpeg、libegl1、libosmesa6、libgl1-mesa-glx、cudnn9 等）本机无 root 无法安装，
  LIBERO 渲染（EGL/OSMesa）可能受影响，见 `_manifests/apt.manual.txt` 与本机对比
- 远程 GPU 驱动 560 / A100，本机驱动 570 / H20，cu124 wheel 均兼容
- 远程 `.pth` 里指向 `/home/jovyan/...` 的可编辑安装本来就已失效，服务靠 PYTHONPATH 工作，保持一致

## 跑实验（2026-09-16 已验证）
1. 起策略服务（GPU 0，端口 9510；本机 9500 被系统 root 服务占用）：
   `himoe-server start [gpu]` / `himoe-server status` / `himoe-server log` / `himoe-server stop`
   日志与路由记录在 `libero-runtime/model-local-<时间戳>.log` 与同名目录
2. 跑一集（LIBERO 环境，必须用 OSMesa 渲染）：
   ```
   source ~/data/env.sh
   cd ~/data/libero-runtime
   MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa envs/libero/bin/python run_benchmark.py --benchmark pro --port 9510 --output-root simulations/pro-local
   ```
3. 与远程同参数（pro/swap, task00, seed 7, init 0, flow seed 42）对比：
   status completed / success False / 520 步 / 52 次推理，本机 93 s（远程 138 s）；
   第 0 步观测图像逐像素一致，前 10 步预测动作最大差 ≤ 0.011（H20 与 A100 浮点差异），之后轨迹自然分叉。

## 本机改动记录（与远程文件不同之处）
- `libero-runtime/configs/*/config.yaml`: /data -> /home/swj/data（sync 已排除该目录）
- `srv/patches` 软链接改指向本机 wristfix/patches（`~/bin/fix-data-symlinks`，sync 结束自动执行）
- `env.sh` 会剔除镜像自带 LD_LIBRARY_PATH 里系统 python3.12 的 torch 库目录（否则 libero 环境 import torch 崩溃）
- `libero-runtime/*.py` 里还有约 30 个分析脚本写死了 /data/...，用到时需改成 $DATA

## 大规模并行评测（2026-09-16 已验证）
- `himoe-server start-all 8`：在 8 张 H20 上各起一个服务，端口 9510..9517（`status` / `stop all`）
- `libero-runtime/run_parallel_batch.py --ports 9510,...,9517 --clients-per-server 4 --tasks 0-9 --init-states 0-7 --out simulations/<name>`
  每集一个 run_benchmark.py 进程、独立输出目录；产出 summary.csv 与 report.json
- 实测：80 集（pro/swap，10 任务 × 8 初始状态），32 路并发，341 s 跑完，14 集/分钟，全部完成，成功 0（与远程该扰动结果一致）
- 服务端选卡必须走 `--gpu`（serve.local.sh 已通过 GPU 环境变量传入）；只设 CUDA_VISIBLE_DEVICES 会被服务端默认值 0 覆盖

## 三个基准同时评测（2026-09-16 已验证）
```
himoe-server start-all 8            # 8 卡各一个服务 9510..9517
cd ~/data/libero-runtime
python3 run_parallel_batch.py --ports 9510,9511,9512 --clients-per-server 4 --benchmark pro --perturbation swap --tasks 0-9 --init-states 0-3 --out simulations/eval-pro
python3 run_parallel_batch.py --ports 9516,9517     --clients-per-server 4 --benchmark pro --perturbation none --tasks 0-9 --init-states 0-4 --out simulations/eval-libero10
python3 run_parallel_batch.py --ports 9513,9514,9515 --clients-per-server 4 --benchmark plus --plus-per-category 1 --init-states 0 --out simulations/eval-plus
```
- `--perturbation none`（run_benchmark.py 新增）= 用 Pro 源里的原版 libero_10 套件，即无扰动 LIBERO-10
- `--plus-per-category K` = Plus 的 7 类扰动 × 10 个基础任务各抽 K 个变体（seed 固定）
- 驱动脚本支持断点续跑（已有 summary.json 的集跳过）和失败自动重试一次；
  LIBERO 在 import 时读写自己的 config.yaml，多进程同时启动偶发读到空文件，重试即可
- LIBERO-Plus 依赖 ImageMagick（Wand），运行库已解包到 ~/.local/syslib，env.sh 设置 MAGICK_HOME 等变量

## 拓扑定义网格实验（2026-09-16）
- 捕获服务：`himoe-capture start-all <run> 8 9520`（`srv/serve_moe_capture.py`，每次请求落盘完整 HB 路由概率 + MoE input/shared/total 张量，约 8 MB/次）
- 客户端：`run_benchmark_capture.py --episode-id N --capture-tag TAG ...`（给请求打 episode 标记），批量用 `run_parallel_batch.py --runner run_benchmark_capture.py --episode-id-base N`
- 数据：`~/data/moe-capture/topo-20260916/server-p95xx/<tag>/qNNN.npz`，170 集（LIBERO-10 100 集 61 成功，Plus 70 集 25 成功），50 GB
- 分析：`topology_grid.py distances` → `grid`（60 个距离矩阵 × 复返区域 162 组参数 × 3 种确认规则 × 6 锚点 + 持久同调 + 简单距离特征，向量化 AUROC）；`topology_grid_validate.py` 做置换检验、跨基准迁移、留一任务验证
- 结果：`~/data/moe-capture/topo-20260916/_grid/{REPORT.md,VALIDATION.md,auroc-table.csv,headline.json,validation.json}`
- 结论（2026-09-16）：复返区域族 25,740 种参数无一在六尺度下一致，置换检验 P=0.63，跨基准迁移 0.57–0.66，判定无结局信息；
  MoE 运动幅度类简单特征（窗口直径 / 步长 / H1 寿命，q16–q20）跨基准迁移 0.74–0.80，优于 v8.2（0.60–0.63）。
  注意：锚点后的特征必须要求完整后续窗口，否则"剩余观测点数"会泄漏结局（第一版验证曾因此出现 0.97 的假迁移分数）。

## 并行度优化（2026-09-16 晚）
实测瓶颈：单进程/卡的服务 batch=1 推理约 0.47 s，nvidia-smi 只有 26–40%；同卡 1/2/3/4 个进程吞吐 2.08/3.14/3.51/3.66 请求/秒（拐点 3 个）。
- `himoe-server start-all 8 3` / `himoe-capture start-all <run> 8 9540 3`：每卡 3 个实例，端口 base + gpu*3 + j
- `run_parallel_batch.py`：动态选负载最少的服务、长集优先（Plus 的 Sensor Noise / Light Conditions 最慢，先跑）、
  `--extra 'benchmark=plus,plus_per_category=1,init_states=0'` 把多个基准并入同一批以重叠尾部、
  客户端进程 OMP/ImageMagick/llvmpipe 线程数钉为 1（否则 120 个客户端把 160 核压到 load 267）
- `libero_runtime._configure_libero` 改为原子写且内容不变不写：根治多客户端同时启动时 LIBERO import 读到空 config.yaml
- `HIMOE_RENDER=egl` 可选：NVIDIA EGL 渲染 1.2 ms/帧（OSMesa 44 ms），需 `__EGL_VENDOR_LIBRARY_FILENAMES` 指向 nvidia json；
  像素与 OSMesa 差约 1/255、动作差 ~0.01，不逐位一致，因此默认仍是 OSMesa（与远程可比）
- `bench_server_load.py`：服务端合成压测（注意 venv311 的 websockets 会读 HTTP_PROXY，脚本内已清除）
- `topology_grid_validate.py` 置换检验多进程化：15 分钟 → 3 分钟
结果：LIBERO-10 100 集 211 s（此前稳态约 480 s）；稳态吞吐 13.7 → 19–23.5 请求/秒；5 集抽样动作/图像逐位一致

## 批推理服务（2026-09-16 晚，用户批准的模型侧改动）
- `srv/serve_moe_batch.py` + `himoe-batch start-all <run|none> 8 9570 32`：请求进队列，凑满 32 或等 15 ms 后一次 `sample_actions`，
  捕获张量按样本拆分落盘（布局与 serve_moe_capture 相同，多一个 batch_size 字段）。每卡 1 个进程即可。
- 单卡吞吐：batch 1/2/4/8/16/32 = 2.2/3.7/6.1/9.5/13.2/16.8 请求/秒（`srv/bench_batch_infer.py`），显存只多 2 GB。
- 数值：batch>1 的 bf16 推理与 batch=1 动作差约 3e-3（最大 8e-3），170 集里 10 集最终成败翻转（86 → 88 成功）。
  批量服务的结果彼此可比，但与 serve_moe_capture 的逐位一致运行不可比；需要逐位复现远程结果时仍用 serve_moe_capture。
- 端到端：170 集 905 s → 378 s（多进程+调度+噪声加速）→ 303 s（批推理，稳态 30.8 请求/秒），此时瓶颈转到客户端 CPU（OSMesa 渲染）。
- `fast_perturbations.py`：LIBERO-plus 的 glass_blur（逐像素 Python 循环，0.84 s/帧 → 0.012 s）和 zoom_blur（线程化 scipy zoom，3 倍）
  的逐位一致加速，由 run_benchmark.py 自动挂载（HIMOE_FAST_PERTURB=0 关闭）。
- 加 `HIMOE_RENDER=egl`（客户端按端口序号分配渲染 GPU）：170 集 186 s，稳态 48 请求/秒，平均 batch 6.6；此时墙钟由最长单集（约 165 s）决定。
  三种口径的成功数：OSMesa+batch1 86 / OSMesa+批推理 88 / EGL+批推理 84，差异来自数值和渲染器，不是环境错误。
- 服务清单：9510–9517 普通服务（逐位可比）、9540–9563 捕获服务 3/卡（逐位可比）、9570–9577 批推理捕获服务（最快）；
  `himoe-server` / `himoe-capture` / `himoe-batch` 各自 `stop all` 释放。

## MoE 控制实验（2026-09-16 深夜起，进行中）
- 工具：`control_branch.py`（从父轨迹重放到分支点后按策略继续；重放状态与记录逐位一致）、`run_control_experiment.py`（并行分支驱动，动态端口）、
  批量服务端新增 `candidates/noises`（一次请求 K 个噪声候选）、`capture/return_probs`、`capture/return_flow`（去噪路径）。
- 父轨迹：`controls/parents-topo-all-i.json`（EGL+批推理的 170 集，86 失败，68 有 v8.2 报警）、`controls/parents-proswap.json`（Pro swap 40 集全失败，37 报警）。
- 结果目录 `controls/exp-*/report.json`：
  - exp-01（报警点、单位噪声、控 1 次）：12 种策略救回 1–4/136，本底 1/136，全部无效。
  - 诊断：单位噪声下 8 个候选 xyz 动作只差 5–13%，flow 策略近乎确定性；2.5 倍噪声差 25%，4 倍夹爪开始乱翻。
  - exp-02a（报警前 8 次查询、噪声 2.5、控 3 次）：native 4、random 9、MoE 选择规则 7–10、**kick（松爪抬升原语）19/136**。
  - exp-02c 上限：随机重采样 8 个种子任一成功仅 9/68 父状态（13%）。
- 结论方向：MoE 报警用于决定"何时"干预，"做什么"用物理恢复原语；exp-03 系列在测原语变体、时机、持续次数与在线触发。
- 其它套件（Goal/Spatial/Object）的 checkpoint 无法下载（HF 直连/镜像均 15 KB/s），暂只能在 LIBERO-10、Plus、Pro 上验证。
- exp-03/04/05/06/07（原语变体、时机、在线触发）：
  - 原语：kick_wiggle 20、kick 后交还策略 20、kick 三次 19、只松爪 11、闭爪抬升 5（/136，报警前 8 次查询）；一次 kick 即可。
  - 时机（事后）：报警前 16/8/4/0 次查询 kick 救回 24/19/15/6（/136），越早越好；报警前 8 次时 23/68 父状态可被某种 kick 救回。
  - 固定时刻 kick 全体（含成功集）：q12/q16/q24 分别打断 80/38/18 个成功集，净负；必须有选择性触发。
  - 在线（从 q0 起跑、含成功集、每臂 340 条）：native 52.1%；v8.2 触发 + kick×3 55.6%；直径触发(1.10) 53.8%；直径或 v8.2 54.4%；
    夹爪感知 kick 52–53%；重触发无益。Pro swap 40 集任何策略 0 救回（换物体位置属任务语义改变，原语救不了）。
  - 夹爪状态有信息：被 kick 打断的成功集 53% 处于闭爪（持物），被救回的失败集 79% 开爪。
- 完整控制实验报告：`libero-runtime/controls/REPORT.md`（在线 v8.2→kick 在 Plus 上 +4.5～+5.5 个百分点，p≤0.03；LIBERO-10 无显著变化；Pro swap 无效）
