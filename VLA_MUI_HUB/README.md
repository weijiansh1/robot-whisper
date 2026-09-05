# VLA_MUI_HUB

Rollout 收集目录，对应 HiMoE-VLA 论文（arXiv 2512.05693v2）的**仿真评测任务**。
命名与布局对齐 `MUI_HUB`（`/home/jovyan/public-ro/MUI_HUB`），便于复用同一套下游分析工具。

**当前状态（2026-08-22）：76 GB 已入库。`right-16x32` 语料（5 任务 × 512 集，51,308 control steps，含 hidden）+ 4 个 `pin-*` 消融臂 + 1 个截断的 CALVIN 探针；其余 69/74 骨架任务目录仍为空。**
本文件与盘上内容的逐项核对见仓库根 `himoe-vla-mui-hub-audit-2026-08-22.md`。

## 布局

```
VLA_MUI_HUB/
├── manifest.json                 # 74 任务清单 + 出处 + 论文分数 + routes.zarr 的 record_schema
│                                 #（手工维护；corpus_layout.load_hub_manifest 会读它，改字段要小心）
├── _pipeline/<run_id>/           # pipeline 级证据：plan.json + REPORT.md + lane 日志
│                                 #（right-16x32 的 plan.json 是事后重建的，以 meta/summaries 为准）
├── activation-viz/               # hidden.zarr 的固化消费者（生成 data.json / 单文件 viewer）
├── moe-trajectory-viz/           # control step × action token 的 MoE 纵向轨迹播放器
├── moe-token-position-pca/       # 逐 action-token 的 PCA-10 路由指纹与跨 chunk 变化分析
└── cache/
    └── HiMoE-VLA/               # 后续可加 pi0 / flower-himoe / w-o-moe 等
        ├── calvin_d_d/           # 34 tasks，全空（本工具链无正式 CALVIN 采集路径）
        ├── calvin_d/task_D_D/    # ⚠ 骨架外：routes-v1 CALVIN 探针（截断，见下）
        ├── libero_spatial/       # 10 tasks，已采 2；ramekin 任务另有 pin-* 消融
        ├── libero_object/       # 10 tasks，全空
        ├── libero_goal/          # 10 tasks，已采 2
        └── libero_long/          # 10 tasks，已采 1  (LIBERO 自己的 id 是 libero_10)
            └── <task>/
                └── <run_id>/
                    ├── meta.json
                    ├── logs/{server,client}.log
                    ├── server/{routes.zarr, capture_summary.json[, hidden.zarr]}
                    └── client/{summaries.json, server_metadata.json, sim_layout.json, episode_NN.npz}
```

对照 `MUI_HUB/cache/<model>/<benchmark>/<run_dir>/`，这里多了一层 `<task>/`：
LLM benchmark 的样本是无状态的题目，而 rollout 是按任务 episode 组织的，
每个任务有独立的初始状态分布和成功判据，放在同层会混。

## 盘上现有数据（2026-08-22）

`right-16x32` = 16 个初始状态（{0,3,7,10,13,16,20,23,26,29,33,36,39,42,46,49}）× 32 个
flow-noise 种子（1000–1031）= 512 集/任务；paper-right wrist layout，replan 10 / settle 10 /
resize 224 / seed 7。pipeline 证据在 `_pipeline/right-16x32/`。

| 任务 | run_id | 集数 | 成功 | control steps | hidden |
|---|---|---:|---:|---:|:---:|
| libero_goal/open_the_middle_drawer_of_the_cabinet | right-16x32 | 512 | 512 | 6,452 | ✓ |
| libero_goal/open_the_top_drawer_and_put_the_bowl_inside | right-16x32 | 512 | 470 | 10,018 | ✓ |
| libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | right-16x32 | 512 | 296 | 22,883 | ✓ |
| libero_spatial/…on_the_ramekin… | right-16x32 | 512 | 500 | 5,139 | ✓ |
| libero_spatial/…on_the_stove… | right-16x32 | 512 | 475 | 6,816 | ✓ |
| libero_spatial/…on_the_ramekin… | pin-base | 512 | 500 | 5,139 | — |
| libero_spatial/…on_the_ramekin… | pin-on | 512 | 504 | 5,052 | — |
| libero_spatial/…on_the_ramekin… | pin-off | 512 | 493 | 5,176 | — |
| libero_spatial/…on_the_ramekin… | pin-smoke | 2 | 2 | 19 | — |
| calvin_d/task_D_D | routes-v1 | 80/1000 链 | Sum 3.8625 | 3,621 | — |

- 5 个 `right-16x32` 的 control steps 合计 **51,308**，与 REPORT.md 及双 lane 日志逐条相符。
- REPORT.md 里 libero_10 的 "−34.6pp" 是**单任务**（moka pots，已知难）对**全套件** 500-ep
  参考的错位比较，不是采集协议出错。
- `pin-base` 与 `right-16x32` 逐字节相同（512 个 npz sha256 全同、routes 数组全同）——
  这条 pipeline 是 bit 确定性的，pin 的对照臂不含独立信息；`pin-on/off` 是真实干预
  （层 2–5 路由被替换，95.8% 站点 top-4 改变）。pin-* 各臂 `hook_verified_calls` 为 0。
- **CALVIN 探针 `routes-v1` 是截断 run**：80/1000 链（`coverage_complete:false`），且
  `episode_id` 全零（server 未按链递增），切分只能靠 `client/sequences.jsonl` 的
  `inference_calls` 前缀和重建。别当正式语料。

## 记录里有什么 + 读取契约

`server/routes.zarr`（Zarr v3 + Zstd(3)，`himoe_router_trace_v2`；N = control steps，扁平）：

| 数组 | shape | dtype | 说明 |
|---|---|---|---|
| `hb_expert_ids` | [N,8,10,11,4] | uint8 | HB 层（2–5,12–15）× denoise × token（0=state，1–10=action）× top-4 |
| `hb_selected_prob` | [N,8,10,11,4] | f16 | **归一化前**的 softmax gather，top-4 和 ≈0.13–0.17，不是 1 |
| `hb_router_probs` | [N,8,10,11,32] | f16 | 完整 32 路 softmax，行和 ≈1 |
| `hb_entropy` | [N,8,10,11] | f16 | fp32 自然对数熵，落盘转 f16 |
| `as_expert_ids` / `as_probs` | [N,4] / [N,4,3] | uint8 / f16 | AS 层（0,1,16,17），top-1；probs 是完整 3 路 softmax |
| `episode_id` / `control_step` | [N] | int32 | 切分依据 / 全局连续 |

`server/hidden.zarr`（`himoe_router_hidden_v1`，仅 `--store-hidden` 时存在；**占 hub 96% 字节**）：
`hb_hidden [N,8,10,11,1024] f16` 是各 HB gate 的输入（RMSNorm 后）；`as_hidden [N,4,10,11,24]`
恒为 `[1]*7+[0]*17` —— 它就是 AS 门控输入 data_mask（24 = max_action_dim，7 = LIBERO 动作维），
恒定是设计不是损坏。两个 store 的 `episode_id`/`control_step` 逐元素对齐。

读取时必须知道的四件事：

1. **npz 里的路由是全零占位。** client 始终以 `--no-routing-capture` 运行，所有
   `episode_*.npz` 的 `expert_ids (T,1)`/`expert_weights (T,1)` 恒为 0、`layer_indices=None`；
   真路由只在 `server/routes.zarr`。2026-08-22 起：9 个既有 run 的 `client/server_metadata.json`
   已追加 `client_routing_capture: false` 标注（server 广告键原样保留），且
   `rollout_with_routes.py` 在 no-capture 模式下不再写这三个键——缺键会 KeyError，
   不会再静默读到 0。npz 只用来读 `state (T,8)` / `actions (T,10,7)` /
   `sim_state`（宽度按 suite：goal 79 / long 47 / spatial 92，见 `sim_layout.json`）。
   文件名实际是 `episode_%02d.npz`（≥100 后变 3 位，**字典序 ≠ 数值序**，用 `summaries.json`
   的 `episode_index` 对齐；`corpus_layout.episode_filename` 目前返回 `%04d`，别用它拼名）。
2. `hb_expert_ids` 槽位是 `topk(sorted=False)`：slot 0 只有 ~45% 是 argmax；
   真 top-1 从 `hb_router_probs` 重算。
3. **fp16 平票**：门控在 bf16 下大量 4th/5th 平票，用存储的 f16 probs 重推 top-4 会在
   4–18% 站点与 `hb_expert_ids` 集合不一致（抽查 10,030/10,030 全是 p[4]==p[5] 平票）。
   以 `hb_expert_ids` 为准——它是模型实际选的专家（recorder 前 4 步有 hook 自校验）。
4. 打开必须 `zarr.open(..., mode='r')` ——默认 `mode='a'` 会动 store。读大数组用
   `/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python`（zarr 3.0.6；
   `/opt/conda` 的 zarr 容器重启会丢）。combine weight = `raw / raw.sum(-1, keepdims=True)`
   （现成实现：`himoe_route_store.ZarrRouteReader.combine_weight`）。

## 怎么往里写

```bash
cd /home/jovyan/work/himoe-vla/himoe-route-capture
PYTHONPATH=/home/jovyan/work/himoe-libero-wrist-fix/src \
python3 run_corpus_capture.py --run-id right-50x1 \
  --gpu <MIG-UUID> --port 8410 --episodes 50 [--tasks 0,1,2]
```

驱动**直接写进本目录**，没有中间格式、没有迁移步骤。两张 32G MIG 分片可以并行跑
（`--tasks` 分组、端口错开）；已完成的任务会跳过，所以中断后原命令重跑即可续。

跨 suite 的批量采集用 `capture_routes_pipeline.py`（`right-16x32` 即由它产出，双 lane 分派 +
预检 checkpoint sha256）。**记得 `python -u` 启动**：right-16x32 当时没加，stdout 块缓冲全丢，
`_pipeline/right-16x32/plan.json` 是事后按命令行重建的。

校验：

```bash
python3 validate_corpus.py --run-id right-50x1 --episodes 50
```

会逐任务核对 `episode_id` 切片、`control_step` 连续性、prompt、checkpoint sha256、
`sampling.complete`，并把每个 suite 的成功率与论文 Section 4.1 复现值对表。

## `<run_id>` 的规则

- 正则 `^[a-z0-9][a-z0-9._-]{0,63}$`，在同一个任务目录内唯一即可。
- **不要求时间戳。** 唯一性是唯一的硬约束；重复跑同一配置（例如换 MIG 分片重采）
  才需要靠它区分。
- **绝不把设计规模或完成度编进名字。** 采集仓库里有个目录叫 `preaction64x8`，
  名字承诺 64×8 = 512 集，实际只有 **28 集、覆盖 4 个首步种子**。
  所有数字放 `meta.json` 的 `sampling` 块，并由脚本从数据里数出来。
- 建议格式 `<操作点>-<协议>`，例如 `right-50x1`、`left-1x64`。

## `sampling`：让采样设计可被机器读

```json
"sampling": {
  "design": "init-states x flow-noise draws",
  "designed_episodes": 50, "actual_episodes": 50,
  "coverage": 1.0, "complete": true,
  "unique_init_states": 50, "unique_flow_noise_seeds": 1,
  "held_fixed": ["task", "wrist_layout", "flow_noise_seed"],
  "varies": ["init_state"]
}
```

`actual_*` 由 `corpus_layout.sampling_block()` 从 `client/summaries.json` 数出来，
不是手写的。消费者应当只信 `complete`。

`held_fixed` / `varies` 摆出来之后，一批数据的**设计轴**就是自明的：
`n_scenes=1, n_draws=64` 和 `n_scenes=50, n_draws=1` 是完全不同的东西，
不必翻脚本才知道。

## 一处目录名与参数名不一致（唯一的一处）

| | 值 |
|---|---|
| hub 目录 | `libero_long` |
| 客户端 `--benchmark` | `libero_10` |
| bridge `--suite` | `long` |

LIBERO 把这套按**任务数**编号：LIBERO-100 拆成 LIBERO-90 与 LIBERO-10，所以 `10` 不是版本号。
但另外三套也各有 10 个任务，`libero_10` 读起来像版本、容易归错档，因此 hub 目录改用说明性的名字
`libero_long`——与另外三套的 `libero_` 前缀对齐。**`libero_10` 这个 id 在所有作为参数的地方保持不变**——
它是 `benchmark.get_benchmark_dict()` 的键，也是 bddl 目录名，改不得。
映射由 `corpus_layout.HUB_DIR` 单点维护，manifest 里记在 `benchmarks.libero_10.hub_dir`。

## 不变量

- `episode_id` 是切分扁平 zarr 的唯一依据（**已知例外**：calvin `routes-v1` 从未写入，全零）；
- `control_step` 全局连续，断了就说明丢了 chunk；
- zarr 默认 `write_empty_chunks=False`：**全零 chunk 不落盘**，读回 fill_value=0 无损——
  所以"目录里缺 chunk 文件"单独不构成损坏证据，副作用是"全零数组"与"没写过的数组"
  在文件层面不可分（calvin 的 `episode_id` 就是这样被发现的）；
- `sum(inference_calls) == capture_summary.control_steps`；
- 任务目录名来自 `benchmark.get_task(i).name`，**永远不要用 `ls` 重新生成**。

## 覆盖范围与当前能力

74 个任务 = CALVIN 34 + LIBERO 40。**只含仿真**，不含真机（xArm7 3 个 + ALOHA 3 个）。

| benchmark | 目录 | 能否用本工具链采集 |
|---|---|---|
| `libero_goal` / `libero_spatial` / `libero_object` | ✅ 各 10 | ✅ 权重齐全、sha256 已校验 |
| `libero_long`（LIBERO id = `libero_10`） | ✅ 10 | ✅ 权重 2026-08-14 到位并校验（sha256 `cdc2b21f…`） |
| `calvin_d_d` | ✅ 34 | ❌ 无正式采集路径；仅有 `serve_calvin_with_recorder` 的截断探针 `calvin_d/task_D_D/routes-v1`（80/1000 链，`episode_id` 全零，见上） |

各 suite 的 rollout 步数上限（来自上游 `examples/libero/main.py:60-69`，
bridge 的 `SUITE_MAX_STEPS` 只有前三个）：

| suite | spatial | object | goal | **long** |
|---|---|---|---|---|
| max_steps | 220 | 280 | 300 | **520** |

`long` 的 520 意味着它单集最长可达其它三套的 1.7–2.4 倍，采集耗时要按这个折算。

**每个 LIBERO 任务恰好 50 个初始状态**（四套实测确认）。要超过 50 集就必须在同一场景上
重复采样，这属于另一种设计，必须写进 `sampling.design`——驱动会拒绝 `--episodes > 50`
而不是去请求一个不存在的初始状态。

任务名不是从论文抄的（论文没列），是从上游 benchmark API 取的，
**不是 bddl 文件的字母序**：`libero_goal` 的 task 1 是 `put_the_bowl_on_the_stove`，
而字母序第 2 个其实是 task 3。四套 LIBERO 的顺序都不是字母序。

## CALVIN 的一个结构性注意事项

CALVIN 的 34 个任务目录是**按任务归档 rollout 用的**，不是论文的评测单元。
论文 D→D 报的是 5 子任务链的 1–5 步完成率与 Sum.（3.98），链是跨任务采样的，
没有 per-task 成绩。所以：

- 一条链的 rollout 会横跨多个任务目录，需要用 `run_id` + 链内序号把它重新拼起来
- 直接对 34 个目录做成功率平均**得不到** Sum.，两者不是同一个量

LIBERO 没有这个问题，是 per-task 独立评测，40 个目录与论文的 suite 平均直接对得上。
