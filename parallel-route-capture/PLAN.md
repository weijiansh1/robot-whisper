# 并行 MoE 路由采集计划（parcap-20260903）

商量结论（2026-09-03）：四个方向都采，只存路由+低维状态；**输出按 VLA_MUI_HUB
约定逐任务落盘到 `VLA_MUI_HUB/cache_new/`**（用户指定，2026-09-03）。
几何沿用 rolling-star K=16 同胞分支设计（environment_seed=7、replan 10、settle 10、
max-steps 520、checkpoint-right wrist 布局、store-full-probs）。campaign seed=20260903。

## 拓扑：一任务一服务器，全部 40 个 LIBERO 任务，两波接力

（2026-09-03 用户确认：所有任务全量；整机 8 卡独占。）每个任务独占一个
`serve_with_recorder` 实例（约 22.6GB 显存）→ 每任务一个 `routes.zarr`。
每卡 5 个服务器（≈113GB/141GB；GPU 7 留一槽给 CALVIN）：

- **wave 0**：39 个任务服务器几乎一波全上（long 的 6-snap 长杆在最前）
- **wave 1**：仅 1 个收尾任务（object 任务 9），大概率与 CALVIN 尾段重叠
- **CALVIN**：GPU 7 常驻，单客户端 200 条序列，跨波运行

注意：此机器上瓶颈是 GPU 推理（环境步进在 240 核上近乎免费，冒烟实测
0.55s/推理、分支 15s≈26 次推理），故并发已饱和 GPU；单波的意义是消灭
波间栅栏空转并让先完成任务的 GPU 时间自动流向同卡未完成任务。

覆盖与预算（2026-09-03 用户要求加量后）：
- long：任务 0–7,9 × init {0,20,42} × 6 snap + t08 × init {5,12,30,42,17,25} × 6 snap
- goal / spatial / object：**各 10/10 任务** × init {0,20,42} × 4 snap
- 合计 123 个 worker、**558 snapshot = 8928 个 K=16 分支** + CALVIN 300 条
- 加量原则：优先加初始状态（统计分组数 2→3，跨组泛化检验更硬），
  t08 加到 6 个新 init（与旧 rolling-star 的 {0,20,3,7} 不重叠）
- 端口：LIBERO 任务 8600–8639，CALVIN 8699

## 产物布局（对齐 hub `cache/` 约定）

```
VLA_MUI_HUB/cache_new/HiMoE-VLA/
├── libero_long|libero_goal|libero_spatial|libero_object/
│   └── <task_name>/rolling-star-k16-20260903/
│       ├── meta.json                 # route-corpus/1，rolling-star 采样设计说明
│       ├── logs/{server.log, client-iNN.log}
│       ├── server/routes.zarr        # 该任务全部路由（fp16 full probs + entropy + top4）
│       └── client/init-NN/snapshot_XXX/{trunk.npz, candidate_XX.json/npz, manifest.json}
└── calvin_d/task_D_D/routes-v2-20260903/{meta.json, logs/, server/, client/}
```

任务 id → 目录名用 LIBERO benchmark 官方顺序（已固化在 plan.json `task_names`，
与 hub `cache/` 现有任务目录名逐一核对一致）。hub 根 `manifest.json` 不动
（`corpus_layout.load_hub_manifest` 在读它）；战役完成后另行为 cache_new 写清单。

## 冒烟结论（2026-09-03，已通过）

704 行路由 / 16/16 分支 / snapshot 7.2 分钟 / 22MB；推理 550ms/步、捕获开销 2ms。
坑与修复：K=16 硬冻结；容器缺 glvnd `libEGL.so.1`（worker 注入 system-libs 的
LD_LIBRARY_PATH）；服务器加载有 ~1.5 分钟静默期；残留 server 要用 ps -ww 找。
冒烟 zarr 因两次运行写同库已污染，仅作机制验证，隔离在 runs/…/smoke/ 不并入语料。

## 预算与保护

- 磁盘 ≈ 11–12GB（~20MB/snapshot × 558 + CALVIN ~0.7GB），可用 17GB；
  watchdog 阈值降到 <3GB 全停（预计完成时剩 ~5GB）。
- 墙钟（按推理量算）：≈36 万次推理 × 0.55s ≈ 55 GPU·h ÷ 8 卡 ≈ **6–8h**（过夜档）；
  CALVIN 300 条串行 ~3.8h 并行其中；全 campaign 截止 = 启动 + 12h。
- 显存 gate：每卡空闲 ≥ 120GB 才起。并发流 15/卡（5 服务器 × 3 worker），
  GPU 推理排队变深但吞吐不变（推理瓶颈已饱和）。
- 断点：LIBERO 按 snapshot manifest 续跑；CALVIN 按 sequences.jsonl 追加；
  波内单 worker 挂掉不阻塞波推进（monitor 标 DEAD，事后补跑该 worker+server）。
- 显存：每卡最多 3 服务器 ≈ 66GB / 141GB；gate 要求每卡空闲 ≥ 70GB。
- 共享租户环境：外部容器任务会来去（nvidia-smi 显示 [Not Found] PID），launch gate 会等。

## 操作

```bash
cd /home/jovyan/work/himoe-vla/parallel-route-capture
bash launch.sh full    # 冒烟已通过；auto 会先重跑冒烟
bash monitor.sh        # 进度表 + 磁盘 + GPU
bash launch.sh stop    # 全停（先客户端后服务器 flush）
```

单 worker 重启：`STOP_BEFORE_UNIX=<unix> bash cmds/worker-p<port>-i<init>.sh`（自动续跑）。
编排痕迹（pids、集中日志、冒烟）在 `runs/parcap-20260903/`，语料本体全部在 cache_new。
