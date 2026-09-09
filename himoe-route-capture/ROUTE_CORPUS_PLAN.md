# LIBERO 30 任务路由语料采集计划（v1）

日期：2026-08-13
目标：在**正确 wrist layout**（`checkpoint-right`）下，采集 LIBERO `goal` / `spatial` / `object`
三套共 30 个任务、每任务 50 个 init state 的完整 MoE 路由状态，形成一个可长期复用、
按任务分目录、带完整溯源的语料库。

配套：[`README.md`](README.md)（采集工具与踩过的坑）、[`VLA_MOE_FLOW.md`](VLA_MOE_FLOW.md)（路由张量结构）、
[08-12 工作进展](../himoe-vla-session-log-2026-08-12.md)

---

## 执行状态（2026-08-13 实时）

| 阶段 | 状态 | 实测结果 |
|---|---|---|
| P0 前检 | ✅ | 三套 checkpoint sha256 **全部与 `suites.py` 逐位吻合**（spatial/object 首次校验）；zarr 3.0.6；两个 32G 分片各 32.4 GB 空闲；`ffree*512` = 62.3 GB。详见 `_preflight/P0_PREFLIGHT.md` |
| P1 存储层 | ✅ | `corpus_layout.py` + `test_corpus_layout.py`，30 passed；两个 env（py3.8 / py3.11）都能导入；任务表已冻结进 `_tasks.json` |
| P2 还债 | ✅ 2 修 1 撤 | ① `episode_id` 已补（含广告位协商 + 5 条回归测试）② **codec 建议被实测推翻，保持 Zstd(3)** ③ `serve_forever` 已加 try/finally |
| P3 冒烟 | ✅ | 3 suite × 3 集 = 9/9 成功、0 failure；`episode_id` 在 zarr 里逐集正确；`control_step` 连续；**spatial/object 首次运行即通过** |
| P4 正式采集 | 🔄 进行中 | 两道并行：A(2g.35gb) 任务 0–4、B(1g.35gb) 任务 5–9，各 15 个任务 |
| P5 校验 | ⏸ | 工具 `validate_corpus.py` 已就绪并在冒烟上验证过 |
| P6 分析 | ⏸ | — |

### P2 的一处更正：codec 建议不复现

08-10 报告 §4.4 建议改用 `Blosc(zstd, clevel=1, shuffle)`，称体积 −7%、编码 −40%。
`bench_corpus_codec.py` 在真实 `within64-s24` 上逐数组重编码复测，**结论不成立**：

| 配置 | 体积 | 编码时间 |
|---|---|---|
| **Zstd(3)（现状）** | **65.49 MB** | **753 ms** |
| Blosc(clevel=1, shuffle)（被建议的） | 69.15 MB (**+5.6%**) | 747 ms (−0.8%) |
| zstd9 | 63.18 MB (−3.5%) | 801 ms (+6.4%) |
| Blosc(clevel=5, shuffle) | 63.20 MB (−3.5%) | 787 ms (+4.5%) |
| Blosc(clevel=7, shuffle) | 62.50 MB (−4.6%) | 1030 ms (+36.8%) |

clevel=1 在**每一个数组上都更大**。可能的差异来源：`within64-s24` 有 86% 的体积来自
`hb_router_probs`（全 32 维概率），而 §4.4 可能测的是不含 full-probs 的批次。
**未定论，但 −7%/−40% 这个数字不能再引用。**

决定：**保持 Zstd(3)**。前沿收益上限 3.5%（≈35 MB / 0.7 GB 语料），
而编码耗时只占推理的 0.06%，不值得让新语料与已有 5 个 zarr 批次格式分叉。

### P3 实测的容量与速度（比预估好）

| 量 | 预估 | 实测 |
|---|---|---|
| 单集体积 | 1.06 MB | **0.51 MB**（正确配置下 episode 更短） |
| 语料总量 | ≈1.2 GB | **≈0.7 GB** |
| 单集墙钟 | 16 s | A 道 16 s / B 道 20 s（1g 分片慢约 25%） |
| 单任务（50 集 + 加载） | 13.5 min | A ≈15 min / B ≈18.5 min |
| **总墙钟** | 4 h | **≈4.6 h**（B 道是长尾） |

---

## 零、与既有数据的关系

现存所有路由 trace 都集中在 **`libero_goal` 的 3 个任务（t0/t1/t3）**，且全部跑在
`released-left`（故意喂错的 wrist 槽位，成功率 28%–70%）。外部评审明确指出过：
在故意喂错输入的策略上测出的性质未必外推。

本次是**正交的一批**：正确配置、全 30 任务、每任务全部 50 个 init state。它不替代旧数据，
两者操作点不同，**不可混在同一张表里做配对比较**。

副产品：`checkpoint-right` + 50 init state × 10 任务正是论文 Section 4.1 的官方评测协议，
所以这批采集**自带一个完整性校验**——成功率必须落在已复现的
Goal 97.8% / Spatial 94.2% / Object 96.6% 附近。对不上就是采集配置错了，不用等分析。

---

## 一、存储规范（先定死，代码按它写）

### 1.1 根位置

```
/home/jovyan/work/himoe-vla/himoe-route-capture/corpus/libero30-right-v1/
```

- 放在 `corpus/` 而**不是** `runs/`：`runs/` 已有 78 个扁平的实验批次目录，是"一次性实验"的语义；
  语料库是"长期资产"，要能整体复制、整体校验、整体引用。
- 目录名编码三件事：范围 `libero30`、操作点 `right`、版本 `v1`。换操作点或改协议就开 `v2`，
  **不在原目录里就地覆盖**。
- 在 `/home/jovyan/work` 下（`/dev/loop0` 独立块设备），容器重启不丢。

### 1.2 目录树

```
corpus/libero30-right-v1/
├── MANIFEST.json              # 语料级溯源，采集开始时写，结束时补统计
├── INDEX.csv                  # 每集一行，全语料唯一入口表
├── README.md                  # 怎么读这批数据（生成，不手写）
├── _logs/
│   ├── srv-libero_goal-t00.log
│   └── cli-libero_goal-t00.log
├── libero_goal/
│   ├── _suite.json            # suite 级：checkpoint sha256、norm asset、max_steps
│   ├── t00__open_the_middle_drawer_of_the_cabinet/
│   │   ├── meta.json          # 任务级：task_id、prompt、MIG UUID+profile、耗时、STATUS
│   │   ├── server/
│   │   │   ├── routes.zarr/           # 完整 32 维 router softmax
│   │   │   └── capture_summary.json   # gates 清单、control_steps、开销
│   │   └── client/
│   │       ├── summaries.json         # 逐集成败/步数/init_state/flow seed
│   │       ├── server_metadata.json   # 客户端握手拿到的服务端指纹
│   │       └── episode_0000.npz ...   # state/actions（--no-routing-capture）
│   ├── t01__put_the_bowl_on_the_stove/
│   └── ...
├── libero_spatial/
└── libero_object/
```

### 1.3 命名规则（硬约束）

| 项 | 规则 | 理由 |
|---|---|---|
| suite 目录 | `libero_goal` / `libero_spatial` / `libero_object` | 与 `--benchmark` 参数逐字相同，不做美化 |
| 任务目录 | `t{id:02d}__{task_name}` | 两位补零保证字典序 = 数字序；`__` 双下划线分隔，因为任务名本身含单下划线 |
| 任务名来源 | **`benchmark.get_task(i).name`** | `--task-id` 的顺序**不是** bddl 字母序（goal t1=`put_the_bowl_on_the_stove`，字母序第 2 个其实是 t3）。用 `ls` 生成目录名必错 |
| episode 文件 | `episode_{idx:04d}.npz` | 四位，为将来 >1000 集留位 |
| 目录名生成 | 由 `corpus_layout.py::task_dir()` 单一函数产出，驱动脚本与分析脚本都调它 | 两边各写一份字符串拼接迟早对不上 |
| 采集中/失败 | `meta.json` 里的 `status: running\|complete\|failed`；未完成的**不写** `INDEX.csv` | 避免半截数据被当成完整数据统计 |

**任务目录一旦写入不得改名**——`INDEX.csv` 与所有分析产物都以它为键。

### 1.4 MANIFEST.json 必须记的溯源字段

一条都不能省，缺任何一条这批数据就没法引用：

```jsonc
{
  "schema_version": "route-corpus/1",
  "created": "2026-08-13T...",
  "operating_point": {"wrist_layout": "checkpoint-right"},
  "protocol": {"init_states": "all-50", "repeats": 1, "replan_steps": 10,
               "settle_steps": 10, "seed": 7,
               "max_steps": {"goal": 300, "spatial": 220, "object": 280}},
  "checkpoints": {"goal": {"sha256": "98ee29d0...", "bytes": 8138322389}, ...},
  "code": {"route_capture_rev": "<git rev 或目录 sha>",
           "bridge_rev": "himoe-libero-wrist-fix @ <rev>"},
  "storage": {"zarr": "v3", "codec": "Blosc(zstd, clevel=1, shuffle=shuffle)"},
  "hardware": {"per_task_mig": "见各任务 meta.json"},
  "totals": {"tasks": 30, "episodes": 1500, "control_steps": null, "bytes": null}
}
```

`checkpoints[*].sha256` 从 `himoe_libero_bridge/suites.py` 的 `SuiteSpec.weights_sha256` 取，
**并在 P0 实际校验一遍磁盘上的文件**（spatial/object 从未跑过，不能假定完好）。

### 1.5 INDEX.csv 列

```
suite, task_id, task_dir, task_name, prompt, episode_index, init_state_id,
flow_noise_seed, success, action_steps, inference_calls, control_step_offset,
mig_uuid, mig_profile, wall_s
```

`control_step_offset` 是这一集在本任务 `routes.zarr` 里的起始下标——
服务端 zarr 是扁平的、不记 episode 边界，靠 `summaries.json` 的 `inference_calls`
累加切分（`within64_lib.load_run` 的做法）。把它固化进 INDEX，分析侧就不必每次重算。

---

## 二、计划表

图例：**P0–P6 顺序执行**；「产出」列是这一阶段结束时必须存在的东西，没有就不算完成。

| # | 阶段 | 具体动作 | 产出 | 预计耗时 | 风险/依赖 |
|---|---|---|---|---|---|
| **P0** | 环境前检 | ① `zarr` 在 model env 存在（3.0.6 ✓，容器重启后需重装）② 两个 32G MIG 空闲探测 ③ 磁盘 ④ **实测校验 spatial/object 两个 checkpoint 的 sha256** ⑤ 确认 4g.71gb 上他人/自己的 routing-cloud 作业不受影响 | `P0_PREFLIGHT.md` 一页核对表 | 20 min（sha256 校 2×7.6 GB 约 2 min） | spatial/object 的 sha256 从未验证过；对不上就得重下（走 `socks5://net-relay:1080`） |
| **P1** | 存储层落地 | 写 `corpus_layout.py`：`task_dir()` / `write_manifest()` / `write_task_meta()` / `build_index()`；配单测（命名幂等、字典序=数字序、任务名取自 benchmark 而非 ls） | `corpus_layout.py` + `test_corpus_layout.py` 全绿 | 1 h | 无 |
| **P2** | 先还债，再采集 | 三条**会污染新语料**的已知债：① `rollout_with_routes.py` 不发 `episode_id`（边界只能靠累加推断）→ 补发并由服务端记录 ② zarr codec 换 `Blosc(zstd, clevel=1, shuffle)`（实测体积 −7%、编码 −40%，帕累托改进，但从未启用）③ `serve_with_recorder.py::serve_forever` 缺 try/finally（异常退出丢最后一个 chunk） | 三处改动 + 回归测试 | 2 h | 只在新语料上启用；**不回改旧批次**，否则旧分析不可复现 |
| **P3** | 冒烟 | 每个 suite 取 t00，各跑 **3 集**：校验 prompt 正确、suite/checkpoint 握手通过、zarr 切分总数 == `sum(inference_calls)`、目录树与 INDEX 生成正确 | `corpus/libero30-right-v1-smoke/`（**独立目录，验完即删**） | 40 min（含 3 次 server 加载） | **spatial/object 首次运行**，归一化资产、prompt、max_steps 都可能翻车。此处翻车比第 900 集翻车便宜 |
| **P4** | 正式采集 | 30 任务 × 50 init state × 1 draw = **1500 集**。两条 32G MIG 并行，一任务一次 server（`--out` 只在启动时取一次，要按任务分目录就必须重启）。断点续跑：已有 `status: complete` 的任务跳过 | 30 个任务目录 + `_logs/` | **约 4 h**（rollout 3.4 h + 30 次 server 加载 ≈ 30 min，已按双卡并行折算） | 见 §三 |
| **P5** | 完整性校验 | ① 逐任务 `control_steps == sum(inference_calls)` ② 每集 `action_steps ≤ max_steps` ③ **成功率对表 §4.1**（Goal 97.8 / Spatial 94.2 / Object 96.6）④ 抽样反查 zarr 切片与 npz 对齐 ⑤ 写 `MANIFEST.totals` 与 `INDEX.csv` | `VALIDATION.md`（含逐 suite 成功率与论文值对比） | 40 min | 成功率明显偏低 = 配置错了，**在这里拦住**，不要带进分析 |
| **P6** | 首轮聚合分析 | 30 任务 × 32 专家使用矩阵、AS/HB 分工、任务间路由距离（同 suite 内 vs 跨 suite）、层深梯度。先出数不下结论 | `analysis/libero30-right-v1/` + 图 | 1.5 h | 跨任务比较受 MIG profile 差异影响，见 §三 |

**总计：约 9–10 小时**，其中 GPU 占用约 4.5 小时。P0–P3 可当天做完，P4 挂后台。

---

## 三、已知风险与对策

| 风险 | 依据 | 对策 |
|---|---|---|
| **两个 MIG profile 不互相逐位复现** | `2g.35gb` (16 SM) 与 `1g.35gb` (12 SM) 跑同一 flow seed：成败 10/10 一致，**步数只有 7/10 一致** | 整任务绑定单张卡，`meta.json` 记 UUID+profile；跨任务比较只用**路由分布类**统计量，不做步数级配对 |
| spatial/object 首次运行 | 这两个 checkpoint 在本地但从无运行记录 | P0 校 sha256 + P3 冒烟；两处都过再进 P4 |
| `pgrep -f <pattern>` 自匹配杀掉服务器 | README 记过，08-12 又踩了一次，导致 5 个任务全废 | 驱动脚本一律用 PID 文件，禁止 `pkill -f` |
| 服务器自然退出丢最后一个 chunk | `atexit` 时 zarr async executor 已拆 | 一律 `kill -TERM`，等 flush；P2 补 try/finally |
| 显存被别的租户占住 → 报 `NVML_SUCCESS == r INTERNAL ASSERT FAILED` | 实为显存不足，不是 NVML 坏 | 起服务器前先探空闲显存；**只用两个 32G 分片，不碰 4g.71gb** |
| `df` 在 `/home/jovyan/work` 上给的是 xfs project quota | 本卷已知问题 | 用 statvfs 核实；本批预估仅约 1.2 GB，余量充裕 |
| 采集开销 | 实测 `mean_capture_ms = 2.57 ms` vs `mean_infer_ms = 717 ms` | 约 0.4%，可忽略；四层完整状态档（4.3× 开销）**本次不启用** |

---

## 四、容量与耗时的推算依据

| 量 | 实测锚点 | 本批推算 |
|---|---|---|
| 单集大小（全 32 维概率） | `within64-s24`：68 MB / 64 集 / 1682 控制步 → **41 KB/控制步** | 正确配置下 episode 更短（约 13 步/集）→ 1500 × 13 × 41 KB ≈ **0.8 GB** |
| 客户端 npz | 720 KB / 64 集 | 1500 集 ≈ **17 MB** |
| **合计** | — | **≈ 1.2 GB**（含 1.5× 余量） |
| 单集墙钟 | `right50`（正确配置）14.2 s/集；`within64`（错误配置，失败集跑满 horizon）28.2 s/集 | 取 16 s/集 → 1500 × 16 s = 6.7 h 单流 → **双卡 3.4 h** |
| server 加载 | 90–120 s | 30 次 → 双卡并行 **≈ 30 min** |

---

## 五、需要拍板的两个可选项

1. **是否顺带存四层完整状态？** 开销 4.3×（每层 268 个 per-expert hook 的 `.norm()` kernel 风暴），
   会把 P4 从 4 h 拉到 15 h+。建议**否**，只在选中的少数任务上补采。
2. **是否让 server 支持按任务轮换 `--out`？** 能省掉 30 次 × 2 min 的加载，
   但要动服务端协议。省的 30 分钟不值当前的改动风险，建议 v1 用「一任务一 server」，
   把轮换留给 v2。
