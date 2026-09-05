# HiMoE-VLA MoE 路由状态捕获与消融

分析报告：
- [`../himoe-vla-moe-routing-analysis-2026-08-07.md`](../himoe-vla-moe-routing-analysis-2026-08-07.md) — 消融与路由统计（438 集）
- [`../himoe-vla-moe-routing-64draw-2026-08-09.md`](../himoe-vla-moe-routing-64draw-2026-08-09.md) — **同一场景 64 次 flow 噪声**：步 0 无信号、第 10–14 步才分离；交互可视化在 `viz/`
- [`EARLY_ROUTE_COMMITMENT.md`](EARLY_ROUTE_COMMITMENT.md) — **固定首步完整路由、替换未来噪声**：16×8 配对网格中 16/16 行均翻转，否定严格“早期注定”
- [`RUNTIME_VECTOR_V3_ANALYSIS.md`](RUNTIME_VECTOR_V3_ANALYSIS.md) — HB5/d0 后 future-path energy 主端点与 `x10-x1` 次端点；task-local state+candidate 双组留出，K1 smoke 明确拒绝推断

本目录在 `/home/jovyan/work` 下（`/dev/loop0`，独立块设备），**容器重启不丢**。

## 重启后需要重装的东西

`/opt/conda` 在容器 overlay 层上，重启会回到镜像状态。本目录的脚本依赖一个不在镜像里的包：

```bash
ALL_PROXY=socks5://net-relay:1080 HTTPS_PROXY=socks5://net-relay:1080 pip install zarr
```

没装 zarr 时，`himoe_route_store.py` 和读 `runs/fullprob-right/routes.zarr` 的分析会
`ModuleNotFoundError`；按集的 `.npz` 数据不受影响。

中文字体已随目录固化在 `assets/NotoSansSC.ttf`，绘图脚本直接引用绝对路径，无需系统安装。

## 数据

`runs/`，1018 个 LIBERO episode，25 个带 `summaries.json` 的客户端批次。几种存储结构并存：

| 格式 | 批次 | 内容 |
|---|---|---|
| 按集 `episode_NN.npz` | 24 个（890 集；启用 `--no-routing-capture` 的批次含全零路由占位） | `expert_ids / expert_weights / state / actions`，ZIP deflate |
| commitment 按集 `episode_NNNN.npz` | `commitment-grid-s24-client`（128 集） | 每个控制步的 `flow_noises / states / action_chunks`，用于精确配对审计 |
| `routes.zarr`（Zarr v3 + Zstd） | `fullprob-right`、3 个 `within64-*`、`commitment-grid-s24-server` | 含完整 32 维 router softmax |
| 单个 `state.npz` | `state-tier` | 四层完整 MoE 状态，27 个数组 |

每个批次都带 `summaries.json`（成败/步数/init_state/flow noise 种子）和
`server_metadata.json`（wrist layout / checkpoint sha256）。**路由 trace 单独没有意义，
必须与这两个文件配对使用。**

### 批次对照

| 目录 | 条件 | 集数 |
|---|---|---|
| `right50` / `left50` | 两种 wrist layout 基线 | 50 / 50 |
| `within` | 8 个 init state × 10 次 flow noise 重复 | 80 |
| `abl-{random,worst,shared-off,block-off}` | **天花板上的消融，结论不可引用**（见报告 5.1） | 25 ×4 |
| `abl-left-{random,blockoff}` | **敏感点上的消融，结论所在**（见报告 5.2） | 50 ×2 |
| `fullprob-right` | 完整 32 维概率 | 312 控制步 |
| **`within64-s24`** + `within64-s24-client` | **init state 24 固定 × 64 次 flow noise，完整 32 维概率**（见 64-draw 报告） | 64 集 / 1682 控制步 / 68 MB |
| `within64-{t1s19,t3s0}` + 对应 client | 两个额外 task/init state 的同场景 flow-noise 重复 | 64 ×2 |
| **`commitment-grid-s24-server`** + `commitment-grid-s24-client` | **16 条固定首步路由 × 8 条共同未来噪声流**（见 early commitment 报告） | 128 集 / 3292 控制步 / 16 行全部成败混合 |
| **`abl2-{none,shared_off,routed_off,block_off}`** | **分支消融四臂配对**（2026-08-10，同一 MIG 分片、同 50 init state 与种子）。结论：两条分支冗余 | 50 ×4，**无路由 trace** |
| `state-tier` | 四层完整状态 | 36 控制步 |
| `left` / `right` / `rr-client2` / `cf-client` / `state-client` / `fp-right-client` | 探路与工具验证 | 共 54 |
| `logs/` | 各次服务器与客户端日志 | 21 个 |

## 跑一次采集

模型服务器（py3.11）持有 hook，LIBERO 客户端（py3.8）驱动环境，两者必须分进程。

```bash
MIG=MIG-60ef5cbe-ca36-5f3f-9931-35f2ce1878d3   # nvidia-smi -L 确认 UUID
export PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src
export MOEVLA_DATA_HOME=/home/jovyan/.cache/himoe-libero-bridge/moevla-data
CKPT=/home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal
UP=/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA

# 终端 A：服务器（加载约 90-120 秒）
setsid nohup .../envs/model/bin/python -u serve_with_recorder.py \
  --port 8101 --gpu "$MIG" --suite goal --checkpoint-dir $CKPT --upstream-root $UP \
  --libero-wrist-layout checkpoint-right --out /path/to/out > srv.log 2>&1 < /dev/null &

# 终端 B：客户端
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=/home/jovyan/.cache/himoe-libero-bridge/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=$PYTHONPATH:/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:\
/home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO:\
/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src \
  .../envs/libero/bin/python -u rollout_with_routes.py \
  --port 8101 --task-id 0 --episodes 50 --no-routing-capture \
  --label myrun --out /path/to/client --libero-root .../upstream/LIBERO
```

服务器用 **SIGTERM** 停（`kill <pid>`），它在信号处理器里 flush；
**不要靠进程自然退出**，`atexit` 时 zarr 的 async executor 已拆，最后一个分片会静默丢失。

### 复现 `within64-s24`（同一场景 64 次 flow 噪声）

服务器加 `--store-full-probs --libero-wrist-layout released-left`，客户端换成：

```bash
  --task-id 0 --benchmark libero_goal \
  --init-state-ids 24 --repeats 64 --noise-seed-base 1000 --no-routing-capture
```

约 30 分钟 / 64 集。跑完先等客户端自己退出，再 SIGTERM 服务器，然后：

```bash
python3 within64_analyze.py  --server-dir runs/within64-s24 --client-dir runs/within64-s24-client --out analysis/within64-s24
python3 within64_figures.py  --server-dir runs/within64-s24 --client-dir runs/within64-s24-client --out figures/within64-s24
python3 within64_animate.py  --server-dir runs/within64-s24 --client-dir runs/within64-s24-client --out figures/within64-s24
python3 within64_export.py   --server-dir runs/within64-s24 --client-dir runs/within64-s24-client --out viz/data.js
```

分析脚本吃纯 numpy，无需 zarr 以外的依赖；`--allow-partial` 可以在 run 还没跑完时先看中间结果。

### 踩过的坑

- `pkill -f <pattern>` 会匹配到发起命令的 shell 自身，连带杀掉正在跑的服务器。用 PID。
- `MIG=... python ... "$MIG"` 里 `$MIG` 在赋值生效前展开成空 → `--gpu ""` → "CUDA is required"。分行写。
- 残留服务器占着 32G MIG 分片时，新进程报 `NVML_SUCCESS == r INTERNAL ASSERT FAILED`，
  实际是显存不够，不是 NVML 坏了。**其它租户占着也一样**——先探空闲显存再起服务器：
  `CUDA_VISIBLE_DEVICES=<uuid> python -c "import torch;print(torch.cuda.mem_get_info()[0]/2**30)"`。
- **换 MIG profile 会破坏逐位复现。** `1g.35gb`(12 SM) 与 `2g.35gb`(16 SM) 上跑同一个
  flow noise seed，成败一致但步数可能不同（`within64-s24` 的 10 个重叠 seed：成败 10/10、
  步数 7/10）。要做配对比较就把 profile 钉死并记录在案。只用两个 32G 分片，别碰 `4g.71gb`。
- `n_suffix` 在 LIBERO 是 11（`n_action_steps=10` + 1 个 state token），
  上游默认配置是 51。`ZarrRouteWriter` 的默认值 51 对 LIBERO 是错的。
- **canvas 的 HiDPI 缩放要同时改宽和高。** `viz/index.html` 一度只把 `canvas.width` 乘了
  `devicePixelRatio`，却对上下文做了 `setTransform(r,0,0,r,0,0)`，于是在 2× 屏上每个面板下方
  1−1/r 的部分被静默裁掉——恰好是 x 轴和图例。DPR=1 的截图完全看不出来，
  必须用 `--force-device-scale-factor=2` 复核。
- **热图的行序要走坐标尺，不要从顶端往下数。** 面板 ④ 曾经把去噪步 0 画在最上面，
  而纵轴刻度是 0 在底部，导致"当前去噪步"的黑框标在错误的行上。

## 第一阶段：action--physics--outcome 行为几何

`capture_behavior_forks.py` 是 `fork_pilot.py` 的正式后继。它在同一完整模拟器
snapshot 下采样 `K` 个 chunk，记录 `H+1` 个时刻的物理轨迹和接触，并从每个
chunk 终点用候选间共享的 continuation noise 重复 `R` 次。第一阶段默认不采路由，
因此服务端不要求独占；只有显式传 `--capture-routes` 时才写 route trace 对齐信息。
采集器会核对源 rollout 与当前策略的 checkpoint、wrist layout 和 action normalization；
semantic event tape 保留逐时刻 raw contact identity，并区分目标物体和左右夹指。

下面假设兼容的 Goal policy server 已在 `8101` 端口运行：

```bash
BR=/home/jovyan/.cache/himoe-libero-bridge
ROOT=/home/jovyan/work/himoe-vla/himoe-route-capture

CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:\
/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:\
$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
  $BR/envs/libero/bin/python -u "$ROOT/capture_behavior_forks.py" \
  --port 8101 \
  --client-dir "$ROOT/runs/objstate-t0s24-client" \
  --libero-root "$BR/upstream/LIBERO" \
  --episodes 0,1,2,3,4 --fork-steps 6,10,11,12 \
  --n-candidates 32 --n-continuations 48 --continuation-steps 5 \
  --out "$ROOT/runs/behavior-forks-calibration"
```

`R=48` 不是功效保证，但已超过当前 95% paired interval 在零 discordance 时完整落入
`[-0.1, 0.1]` 所需的最小值 42。正式数据应把 snapshot 分成 calibration 和 evaluation；
先在不相交的 calibration snapshot 上拟合分量尺度和 near/far 阈值，再冻结后分析
evaluation capture：

```bash
cd /home/jovyan/work/himoe-vla/himoe-route-capture

python3 analyze_behavior_geometry.py \
  --capture runs/behavior-forks-calibration \
  --out-dir analysis/behavior-geometry/calibration --mode formal

python3 analyze_behavior_geometry.py \
  --capture runs/behavior-forks-evaluation \
  --out-dir analysis/behavior-geometry/evaluation --mode formal \
  --physics-scales-in analysis/behavior-geometry/calibration/physical_scales.json \
  --thresholds-in analysis/behavior-geometry/calibration/thresholds.json
```

第二条命令只有在 calibration/evaluation 的完整 snapshot state hash 确实不相交、
两个外部文件都保留 calibration provenance、每个 pool 的 `R` 足以支持等价区间，且
semantic event mapping 完整时，才会标记 `confirmatory: true`。bootstrap 单位是
snapshot；单 snapshot 不输出伪精确置信区间，pair 也不会被当作独立样本。pairwise CI
用于 mining，未经多重性校正，选出的 pair 仍需独立 continuation 复验。

route-off run 可以 `--resume`，完整 journal 会自动采用，未写入 journal 的原子文件会
保留到 `orphaned/`。route-on 一旦中断则拒绝 resume，因为当前 recorder 尚不回传实际
持久化 row。现有 `runs/fork-pilot-n32-client` 只能做 endpoint proxy 复算；它没有 dense
trajectory/contact 且每候选只有一个 continuation，分析输出必须保持
`strict_pair_mining_supported: false`。

旧 fork pilot 已保存的真实 HiMoE chunks 可以先做无模型的 dense 物理补采。该路径不调用
policy server，只验证 `A -> X,E,s'`，不能补出 continuation 或 Q label：

```bash
CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src:\
$BR/upstream/LIBERO:$BR/upstream/HiMoE-VLA/packages/openpi-client/src \
  $BR/envs/libero/bin/python -u "$ROOT/replay_dense_forks.py" \
  --source-rollout-dir "$ROOT/runs/objstate-t0s24-client" \
  --legacy-fork-dir "$ROOT/runs/fork-pilot-n32-client" \
  --episode 0 --fork-step 11 --candidate-count 32 \
  --benchmark libero_goal --libero-root "$BR/upstream/LIBERO" \
  --settle-steps 10 --replan-steps 10 --seed 1701 \
  --out "$ROOT/runs/dense-replay-legacy-n32-ep0-t11"
```

当 cgroup 内存不足以常规载入 checkpoint 时，GPU server 可加 `--low-memory-load`：模型在
`meta` 上构造，checkpoint 经 CPU mmap 逐张量转为模型原始 dtype 并直接送入目标 GPU，避免
15.2 GiB 的 CPU 模型副本。没有空闲 GPU 时仍可加
`--gpu cpu --low-memory-load --no-route-capture` 做小规模端到端 smoke；CPU bf16 的
kernel/tie-breaking 与部署 GPU 不同，不能把它用于 GPU 候选的数值复现实验。两种模式都会
记录严格的 checkpoint load audit。

## 18-state behavior-geometry micro-pilot

`run_behavior_micro_pilot.py` 实现当前冻结的非确认性小验证。设计在
`behavior_micro_pilot_config.json`：task 0 校准，task 1/3 评估；每任务 6 个独立 source
episode snapshot；screen/formal 各 K=32；screen R8，formal R48；只有 label coverage 或
main eligible snapshot 数不足才把所有 main/enriched formal pool 一次性补到 R96。

先在 CPU 上重放旧 rollout，补出 dense physical/contact/event tape 并冻结 18 个 snapshot：

```bash
python3 run_behavior_micro_pilot.py prepare \
  --out runs/behavior-micro-pilot-20260824

python3 run_behavior_micro_pilot.py status \
  --out runs/behavior-micro-pilot-20260824 --gpu 0 --json
```

有至少 24 GiB 空闲显存和 1.5 GiB cgroup host-memory headroom 后，运行完整小验证：

```bash
python3 run_behavior_micro_pilot.py run \
  --out runs/behavior-micro-pilot-20260824 --gpu 0
```

runner 会先做同一输入/噪声下 A（无 hook）、B（flow only）、C（完整 route+hidden+flow）
逐位一致性审计，随后自行管理 request-gated server。只有 36 个 candidate pool 会进入共享
`routes.zarr`、`hidden.zarr`、`flow_trajectory.zarr`，总行数必须精确为 `18*32*2=1152`；
所有 continuation acknowledgement 都必须保持 `(row_count,durable)=(1152,1151)`。

恢复以已发布且 checksum 通过的 candidate artifact 为唯一依据。任一 K=32 pool 失败后，
runner 立即 SIGTERM server；重启时三类 store 一起截到 artifact-confirmed `[0,N)` 前缀。
因此客户端在半池断开，或完成第 32 条但尚未发布 artifact 时，都不会把 orphan rows 混入
下一 pool。screen/main assembly、screen-positive enriched queue、formal R48/R96 analysis 和
最终 `confirmatory:false` decision 都是不可变、可审计的分阶段 artifact。route effect 的方向
和 CI 会报告，但永不参与 gate；micro-pilot 不会启动跨控制步状态图或 selector 训练。

## 脚本

| 文件 | 用途 |
|---|---|
| `himoe_router_recorder.py` | router 档 hook：AS+HB、raw 概率、entropy、可选完整分布 |
| `himoe_state_recorder.py` | 四层完整状态：router / block / per-expert / raw 分档 |
| `himoe_route_store.py` | Zarr v3 + Zstd 读写（codec 按 dtype 分配，**不要开 bitshuffle**） |
| `serve_with_recorder.py` | 挂 router recorder 的服务器 |
| `serve_state.py` | 挂四层 recorder 的服务器（开销 4.3×，见报告） |
| `serve_ablated_router.py` | 路由消融：`none / random / worst / uniform` |
| `serve_ablated_branch.py` | 分支消融：`none / shared_off / routed_off / shared_rescaled / block_off` |
| `probe_counterfactual.py` | 离线反事实路由探针 |
| `rollout_with_routes.py` | LIBERO rollout 驱动，支持 `--init-state-ids` × `--repeats` |
| `capture_behavior_forks.py` | 同 snapshot 的 `A -> X -> E,s' -> Q` 正式采集；dense physics、raw contact、共享 CRN continuation |
| `replay_dense_forks.py` | 对旧 fork pilot 的已保存 action chunk 做无模型 dense 物理/接触补采；明确不产生 continuation 或 Q claim |
| `himoe_low_memory_load.py` | 8 GiB cgroup 下的 opt-in CPU/GPU mmap/meta/assign checkpoint loader |
| `behavior_geometry.py` | action/physics/event 距离、paired 等价区间、四象限分类和 snapshot bootstrap 数值原语 |
| `analyze_behavior_geometry.py` | 正式行为几何与旧 fork pilot 的受限 proxy 分析；支持冻结尺度和阈值 |
| `analyze_route_outcome_geometry.py` | 阶段二小验证：复用阶段一阈值，以完整 HB route 的 RMS Hellinger 距离直接检验两条 action-conditioned outcome 不等式 |
| `run_behavior_micro_pilot.py` | 18-state candidate-only 小验证总控；资源门、A/B/C no-op gate、artifact-prefix 恢复、R48→全局 R96、最终决策 |
| `capture_behavior_study.py`, `assemble_behavior_study.py` | v2 source-event/candidate/continuation 分阶段采集与 checksum assembly；screen/main/enriched 严格分池 |
| `audit_candidate_capture_noop.py` | 部署 GPU 上验证 route/hidden/flow hooks 对 action 与 flow trajectory 逐位无扰动 |
| `analyze_routes.py` | 基础度量与 layout 对比 |
| `analyze_controls.py` | 对照检验（随机基线、长度混淆） |
| `analyze_decode.py` | 线性探针解码 + 置换零分布 |
| `analyze_within_state.py` | 组内（同 init state）成败对比 |
| `analyze_router_sharpness.py` | router 锐度（熵 / top1 / margin） |
| `analyze_expert_weights.py` | 专家权重差异（在 model env 跑） |
| `bench_real.py` | 真实 trace 上的压缩基准（**只测了 bitshuffle，见"已知未完成项"**） |
| `bench_formats.py` | 格式基准：zarr / HDF5 / blosc2 / npy / npz，体积 + 写 + 全读/抽步/抽层 |
| `run_branch_ablation.sh` | 四臂分支消融驱动（串行起停服务器、断点续跑） |
| `analyze_branch_ablation.py` | 配对 McNemar 精确检验 + Wilson CI + 分支分解 |
| `analyze_markov_routing.py` | control-step 尺度的 state-token 轨迹链；用于环境轨迹动力学，不用于同一 query 的候选选择 |
| `analyze_intraquery_markov.py` | query 内 `10 denoise × 8 HB layer` 的 action-token 微步链；双留出 success/failure 路径似然比与 top-k shadow 选择 |
| `analyze_intraquery_markov_transfer.py` | 三个 Goal snapshot 的 task+seed 双留出迁移；同一冻结 split 模型在 within64 与 objstate 重采集上复验 |
| `continuous_route_markov.py`, `analyze_intraquery_continuous_markov_transfer.py` | 去掉 K-means，以每轮完整 Hellinger router 张量为连续状态的一阶 Markov；同样执行 task+seed 双留出、objstate 复验和正则敏感性检查 |
| `make_figures.py`, `fig_*.py` | 图 1–8 |
| `within64_lib.py` | 把 `serve_with_recorder` 的扁平 zarr 按 `summaries.json` 的 `inference_calls` 切回逐集（服务器不记录 episode 边界） |
| `within64_analyze.py` | 64 集分析：步 0 / 前缀解码（ridge 探针 + 闭式 LOO + 置换）、锐度、churn、动作集合对照、跨 MIG 复现性 |
| `within64_figures.py` | 图 f1–f5 |
| `within64_animate.py` | 路由演化 MP4 / GIF |
| `within64_export.py` | 打包 `viz/data.js`（分两档：全部集的聚合 + 8 个精细样本） |
| `rollout_commitment_grid.py` | 首步噪声 × 未来噪声的配对 rollout 网格；精确注入/确认 flow noise，支持 pilot 与断点续跑 |
| `analyze_commitment_grid.py` | 早期承诺完整性审计、成败矩阵、列保持置换检验和逐去噪轮探索性路由信号 |
| `analyze_runtime_vector_v3.py` | v3 HB5/d0 runtime vector：冻结 B1 + exact S_LOO/Gram block，五任务等权 macro、state bootstrap、跨 sketch/capacity sensitivity；不接受单 pool/K1 |
| `rollout_flow_lead.py` | sibling-candidate flow trace 驱动；`--query-base` 可隔离共享 server 的多次 client invocation |
| `test_commitment_grid.py` | commitment 网格规划、精确配对与统计检验单测 |
| `viz/himoe-routing-viewer.html` | **单文件交互页面**，双击即开；分体版是 `viz/index.html` + `viz/data.js` |

## 已知的未完成项

- 采集格式分裂：25 个客户端批次用 `np.savez_compressed`（deflate），只有 5 个服务端批次走 Zarr+Zstd。
  npz 的随机读比 zarr **慢 500 倍**（抽 32 个随机控制步 4.88s vs 0.009s），必须整体解压。
- **`_codecs()` 的 `shuffle` 默认是 `None`** —— 所有已存批次都是纯 Zstd(3)，
  Blosc 一次也没实际用过。`bench_real.py` 当初只测了 `bitshuffle`（确实有害），
  漏掉了普通 byte-shuffle。

  2026-08-10 报告 §4.4 据此建议改用 `Blosc(zstd, clevel=1, shuffle)`，称对现状是
  **帕累托改进**（体积 −7%、编码 −40%）。**2026-08-13 在真实 `within64-s24` 上复测，
  该结论不复现**：`bench_corpus_codec.py` 用同一批数组逐个重编码，clevel=1 在
  **每一个数组上都更大**，总体 **+5.6%**，编码时间也只快 0.8%。前沿实际上是
  `zstd9`（−3.5% / +6.4% 时间）与 `blosc5+shuffle`（−3.5% / +4.5%），两者等价；
  `blosc7` 再多 1 个百分点但慢 37%。

  差异的可能来源：`within64-s24` 的体积 86% 来自 `hb_router_probs`（全 32 维概率），
  而 §4.4 的基准可能跑在不含 full-probs 的批次上。**未定论，不要再引用 −7%/−40%。**

  结论：**保持 Zstd(3) 不变**。收益上限 3.5%（约 35 MB / 1 GB 语料），
  而编码耗时本就只占推理的 0.06%（0.45 ms vs 717 ms 每控制步），不值得让新语料
  与已有 5 个 zarr 批次的格式分叉。
- ~~**通用客户端不记录 episode 边界**~~ **已修（2026-08-13）**：`rollout_with_routes.py`
  现在会发 `episode_id`，但**仅当服务端在 metadata 里广告 `episode_id_key`**
  （`serve_with_recorder.py` 会广告；其它服务器把 observation 原样交给策略，
  多一个键会打到策略上）。回归测试见 `test_episode_id.py`。
  **旧批次不受影响**，仍只能靠 `summaries.json` 的 `inference_calls` 累加切分。
  `control_step` 语义**故意没动**——`within64_lib.py:134` 依赖它全局连续来检测丢 chunk。
- 四层档 4.3× 开销来自 268 个 per-expert hook 的逐次 `.norm()`（kernel launch 风暴），可优化。
- `himoe_state_recorder.py` 的 `raw` 档从未在真实 rollout 上跑过。
- `serve_ablated_router.py --mode uniform` 从未运行。
