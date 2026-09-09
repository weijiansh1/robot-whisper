# VLA_MUI_HUB MoE 路由缓存正确性审计

审计日期：2026-08-22  
审计对象：`/home/jovyan/work/himoe-vla/VLA_MUI_HUB`  
审计方式：全量库存与元数据检查，加关键结论的独立数值复核。Zarr 均以 `mode="r"` 打开，使用 `/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python`（Python 环境内 `zarr==3.0.6`、`numpy==1.26.4`）。  
写入边界：本次没有修改 `VLA_MUI_HUB` 内任何文件；本报告位于 hub 外。

并发说明：审计期间检测到 `VLA_MUI_HUB/README.md` 于 09:13 UTC 被外部进程更新（本审计的命令均为只读）。下文已重新读取该版本；不回退该并发修改。

## 1. 结论先行

**LIBERO 的 `routes.zarr` 和 `hidden.zarr` 可以作为可信的 MoE 路由缓存使用。** 独立复核没有发现专家 id、概率、熵、router 输入、层轴、denoise/token 轴或跨 store 对齐发生损坏。指定目标 run 的 5,677,760 个 HB 路由站点全部满足：专家 id 在 `[0,31]`、每个 top-4 内互异、`hb_selected_prob` 与 `hb_router_probs[ids]` 逐位相等、没有严格 top-4 违例。

**但不能把整个 hub 无条件视为“所有入口都正确”。** 有两个高严重度的数据契约缺陷：

1. CALVIN `routes-v1` 的 3,621 个 `episode_id` 全为 0，Zarr 自身无法切分 80 条序列；只能通过 `sequences.jsonl` 重建边界。
2. 全部 4,098 个客户端 NPZ 的 `expert_ids`、`expert_weights` 都是 `(T,1)` 全零占位，`layer_indices=None`；真实路由只在服务端 Zarr 中。按 NPZ 字段名直接读取会静默得到错误结果。

存储容器本身完好：15 个 Zarr v3 store、100 个数组全部可读，100/100 数组均使用 `bytes(little-endian) + Zstd(level=3)`。47,435 个逻辑 chunk 中有 47,320 个物理 chunk；少的 115 个全是 fill-value 为 0 时不落盘的 chunk，其中 114 个正对应 CALVIN 的语义缺陷，另 1 个是 LIBERO-Long episode 0 的正常稀疏省略。

建议的消费规则是：LIBERO 路由读 `server/routes.zarr`，router 输入读 `server/hidden.zarr`，episode 描述读 `client/summaries.json`；不要从现有 NPZ 读取路由；CALVIN 在生成显式边界表之前不要按 `episode_id` 分组。

## 2. 范围与库存

文件系统实测：hub 内 51,692 个文件，其中 `cache/` 内 51,678 个；在 09:13 UTC 的 README 更新后，文件 payload 合计 80,783,600,041 bytes（75.236 GiB，`du -sh` 显示 76G）。共有 10 个 populated run、10 个 `routes.zarr`、5 个 `hidden.zarr`、4,098 个 NPZ。

| run（相对 `cache/HiMoE-VLA`） | Zarr N | 客户端记录数 | `sum(inference_calls)` | routes MB | hidden GB |
|---|---:|---:|---:|---:|---:|
| `calvin_d/task_D_D/routes-v1` | 3,621 | 80 sequences | 3,621 | 161.675 | - |
| `libero_goal/open_the_middle_drawer_of_the_cabinet/right-16x32` | 6,452 | 512 | 6,452 | 254.933 | 9.785 |
| `libero_goal/open_the_top_drawer_and_put_the_bowl_inside/right-16x32` | 10,018 | 512 | 10,018 | 397.396 | 15.196 |
| `libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32` | 22,883 | 512 | 22,883 | 927.797 | 34.718 |
| `libero_spatial/...ramekin.../right-16x32` | 5,139 | 512 | 5,139 | 224.077 | 7.783 |
| `libero_spatial/...stove.../right-16x32` | 6,816 | 512 | 6,816 | 292.723 | 10.327 |
| `libero_spatial/...ramekin.../pin-base` | 5,139 | 512 | 5,139 | 224.077 | - |
| `libero_spatial/...ramekin.../pin-off` | 5,176 | 512 | 5,176 | 225.908 | - |
| `libero_spatial/...ramekin.../pin-on` | 5,052 | 512 | 5,052 | 220.233 | - |
| `libero_spatial/...ramekin.../pin-smoke` | 19 | 2 | 19 | 0.833 | - |

复核结果：10/10 run 都满足 `sum(inference_calls) == Zarr N`。5 个 `right-16x32` 的 N 合计为 `6,452 + 10,018 + 22,883 + 5,139 + 6,816 = 51,308`，与 `_pipeline/right-16x32/REPORT.md` 和两条 lane 日志一致。

5 个 `right-16x32` run 均为 16 个 init state 乘 32 个 flow-noise seed：每个 run 有 512 个不同 `(init_state_id, flow_noise_seed)` 对，每对恰好出现 1 次；与 `plan.json` 的 `scenes=16`、`draws=32`、`episodes_per_task=512` 一致。

## 3. 模型结构与缓存 schema

### 3.1 源码真值

源码给出的 action expert 是 18 层、hidden size 1024：

- `paligemma_with_expert.py:143-150`：action expert `hidden_size=1024`、`num_hidden_layers=18`。
- `models/himoe.py:379-384`：层 0、1、16、17 使用 AS-MoE；层 2、3、4、5、12、13、14、15 使用 HB-MoE；层 6 至 11 为 dense MLP。
- `models/himoe.py:85-115`：AS 为 3 专家 top-1、门控输入 24 维；HB 为 32 专家 top-4、门控输入 1024 维。
- `modeling_moe.py:66-82`：HB 先做 32 路 softmax，再执行 `torch.topk(..., sorted=False)`，用于专家混合的 top-4 权重随后归一化。
- `modeling_moe.py:259-263`：AS 不读 block hidden state 作为 gate 输入，而是把 `data_mask` 扩展到所有 suffix token 后送入 gate。
- `training/config.py:2280-2289`：LIBERO Goal 的 `data_mask = [1] * 7 + [0] * 9 + [0] * 8`，即 7 个有效动作维加 17 个 padding 维。

目标 run 的 `capture_summary.json` 实际列出 12 个 gate，顺序和配置为：AS `[(0,3,1),(1,3,1),(16,3,1),(17,3,1)]`，HB `[(2,32,4),(3,32,4),(4,32,4),(5,32,4),(12,32,4),(13,32,4),(14,32,4),(15,32,4)]`。`hook_verified_calls=480`，`hook_verify_failures=[]`。

### 3.2 全 store attrs

10/10 route store 的关键 attrs 精确一致：

```text
format=himoe_router_trace_v2
n_hb_layers=8  n_as_layers=4
n_denoise=10  n_suffix=11
top_k=4  n_hb_experts=32  n_as_experts=3
```

5/5 hidden store 的关键 attrs 精确一致：

```text
format=himoe_router_hidden_v1
n_hb_layers=8  n_as_layers=4
n_denoise=10  n_suffix=11
hb_hidden_dim=1024  as_hidden_dim=24
```

指定目标 `libero_goal/open_the_middle_drawer_of_the_cabinet/right-16x32` 的数组为：

| 数组 | shape | dtype | chunks |
|---|---|---|---|
| `hb_router_probs` | `(6452,8,10,11,32)` | float16 | `(4,8,10,11,32)` |
| `hb_expert_ids` | `(6452,8,10,11,4)` | uint8 | `(32,8,10,11,4)` |
| `hb_selected_prob` | `(6452,8,10,11,4)` | float16 | `(32,8,10,11,4)` |
| `hb_entropy` | `(6452,8,10,11)` | float16 | `(32,8,10,11)` |
| `as_expert_ids` | `(6452,4)` | uint8 | `(32,4)` |
| `as_probs` | `(6452,4,3)` | float16 | `(32,4,3)` |
| `episode_id`, `control_step` | `(6452,)` | int32 | `(32,)` |
| `hb_hidden` | `(6452,8,10,11,1024)` | float16 | `(8,8,10,11,1024)` |
| `as_hidden` | `(6452,4,10,11,24)` | float16 | `(8,4,10,11,24)` |

## 4. 内容正确性复核

### 4.1 HB 专家选择与概率

对目标 run 的全部 6,452 行分块扫描，共 5,677,760 个 HB 站点、22,711,040 个已选概率元素：

| 检查 | 实际输出 | 判定 |
|---|---:|---|
| expert id 范围 | min=0，max=31 | 通过 |
| top-4 行内重复 | 0 / 5,677,760 | 通过 |
| `selected == gather(probs, ids)` | mismatch=0 / 22,711,040，max abs diff=0 | 通过 |
| 32 路概率行和 | max abs error=`2.86102295e-4`，mean abs error=`4.16085592e-5` | fp16 正常 |
| 严格 top-4 违例 | 0 / 5,677,760 | 通过 |
| top-4 原始概率质量 | mean=`0.150806138`，min=`0.128417969`，max=`0.683776855` | 合理，且不是 combine weight |

逐层 top-4 原始质量均值为 `[0.157644, 0.156480, 0.164461, 0.153705, 0.143880, 0.144104, 0.143060, 0.143117]`。这解释了常见的“约 0.13 至 0.17”：它描述的是层均值，不是已经归一到 1 的专家混合权重。

`hb_entropy` 是 recorder 用 fp32 和自然对数计算后再存成 fp16。直接用未重归一的已存 fp16 概率重算，max/mean abs diff 为 `1.568317e-3 / 5.054129e-4`；先把 fp16 行重新归一化再重算，max/mean 为 `1.267433e-3 / 4.972073e-4`。两种结果都符合“原始计算值与概率均各自经过 fp16 舍入”的预期。

### 4.2 从 hidden 与 checkpoint 独立重建 gate

运行：

```bash
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  verify_capture.py --run-id right-16x32 --suites libero_goal --steps 32
```

实际输出：

```text
open_the_middle_drawer_of_the_cabinet:
  max|dp|=0.00230, top4 differs 5313/28160, max gap=0.000496
open_the_top_drawer_and_put_the_bowl_inside:
  max|dp|=0.00363, top4 differs 5102/28160, max gap=0.000535
0 task(s) failed verification
```

两个 run 的概率最大差都低于 bfloat16 round-to-nearest 的 unit roundoff `2^-8 = 0.00390625`；所有重建 top-4 分歧的最大 4th/5th 概率间隔也远低于该量级。这个检查使用 checkpoint 的 8 个 `[32,1024]` gate 权重和 `hidden.zarr`，独立于 recorder 已存的概率，支持“hook 捕获的是正确 router 输入”这一关键结论。

### 4.3 hidden 激活

在目标 run 上均匀抽取 32 个控制行，读取 28,835,840 个 `hb_hidden` 元素，得到 RMS=`0.985000507`、min=`-8.3671875`、max=`8.1328125`。它与进入 HB gate 前的 1024 维 RMSNorm 输出相符，没有全零、常量或量级异常。

目标 run 的全部 68,133,120 个 `as_hidden` 元素与 `[1,1,1,1,1,1,1,0,...,0]` 精确比较，mismatch=0。四个 AS 层的全程唯一专家分别为：layer 0 -> `{2}`、layer 1 -> `{0}`、layer 16 -> `{0}`、layer 17 -> `{1}`；每层 `as_probs` 也只有 1 个唯一行。该“退化”由恒定 `data_mask` 输入直接导致，是当前设计的可预期行为，不是缓存损坏。

### 4.4 episode、control 与双 store 对齐

目标 run：

- `episode_id` 有 512 个唯一值，精确为 `0..511`，每个 episode 有 12 至 14 行，均值 `12.6015625`；512 段在 Zarr 内连续排列。
- `control_step` 精确为 `0..6451`。
- routes 与 hidden 的 `episode_id`、`control_step` 均逐元素相同。

全量复核进一步得到：9/9 LIBERO run 的所有 route 数组首维都等于 N，`control_step` 都精确为 `0..N-1`，`episode_id` 都与 `summaries.json` 中 `episode_index` 按 `inference_calls` 展开后的数组逐元素相同。5/5 hidden store 的所有数组首维均为对应 N，且两个索引数组与 routes 逐元素相同。

## 5. 存储正确性

### 5.1 格式、codec 与 chunk

全量解析 115 个 `zarr.json`：15 个 group metadata 和 100 个 array metadata 的 `zarr_format` 都为 3；100/100 数组的 codec pipeline 都是：

```text
bytes(endian=little) -> zstd(level=3, checksum=false)
```

物理 chunk 计数：

```text
logical nchunks total = 47,435
physical chunk files  = 47,320
difference            = 115
```

差异只来自两个 `episode_id` 数组：

- CALVIN：逻辑 114 chunks、物理 0；3,621 行读回均为 fill value 0。这不是压缩损坏，但其全零语义是缺陷，见 §6.1。
- LIBERO-Long：逻辑 716 chunks、物理 715；只缺 `c/0`。前 32 行确实全为 episode 0；前 64 行的计数为 `{0:52, 1:12}`，因此该全零 chunk 被正常省略。

除这两个 fill-value 情况外，所有数组的物理 chunk 数都等于 `nchunks`。

### 5.2 空间占用

5 个 hidden store 合计 77,809,144,644 bytes（72.465 GiB），占 hub 全部文件 bytes 的 `96.3180%`。09:13 UTC 更新后的 README 已补入这项；08-14 的 manifest `record_schema.contents` 仍没有 `hidden.zarr`。

4,098 个 NPZ 合计只有 38,733,969 bytes（0.0361 GiB）。因此“hub 很大”的主要原因是可反事实重算 router 所需的 `hb_hidden`，不是客户端 episode 文件。

## 6. 缺陷清单

### 6.1 高：CALVIN `episode_id` 全零，且只完成 80/1000

路径：`cache/HiMoE-VLA/calvin_d/task_D_D/routes-v1/server/routes.zarr`

实际输出：

```text
N=3621, episode_id unique=[0], nonzero=0
episode_id logical chunks=114, physical chunks=0
control_step=0..3620, exact arange=True
sequences.jsonl lines=80, sequence_index=0..79
sum(inference_calls)=3621, min=25, max=76, mean=45.2625
selected_sequence_count=1000, evaluated_sequences=80
coverage_complete=false
```

根因可从 `serve_calvin_with_recorder.py:86-95` 直接看到：服务端注释称不知道客户端五子任务链 id，并固定调用 `begin_control_step(episode_id=0, ...)`。

影响：Zarr 不满足 hub 文档中“`episode_id` 是切分扁平 Zarr 的唯一依据”的不变量。80 条边界仍可无歧义恢复：按 `sequences.jsonl` 顺序，把每条 `sequence_index` 重复其 `inference_calls` 次，所得数组长度正好为 3,621、唯一 id 数为 80；但这依赖外部 sidecar，不能从 Zarr 单独恢复。

附带元数据问题：真实目录是 `calvin_d/`，manifest 只有空骨架 `calvin_d_d/`；该 run 也没有按 LIBERO layout 提供 `meta.json` 和 logs。它应明确标记为 coverage 8% 的部分 run，不能作为 1000-sequence 正式结果。

### 6.2 高：4,098/4,098 个 NPZ 的路由字段都是占位值

对全部 NPZ 做只读扫描，结果为：

```text
NPZ_TOTAL=4098
ids_all_zero=4098
weights_all_zero=4098
shape_is_(T,1)=4098
layer_indices_is_None=4098
unreadable_or_missing_required_keys=0
```

固定种子 `20260822` 随机抽出的三例：

| 文件 | `expert_ids` / `expert_weights` | `layer_indices` | metadata 广告 shape |
|---|---|---|---|
| goal top drawer `episode_422.npz` | `(18,1)`，全零 | `None` | `[10,8,10,4]` |
| long moka `episode_111.npz` | `(37,1)`，全零 | `None` | `[10,8,10,4]` |
| long moka `episode_232.npz` | `(40,1)`，全零 | `None` | `[10,8,10,4]` |

根因是 `rollout_with_routes.py:122-131` 的 `no_capture` 分支主动 append `np.zeros((1,))`，而本批次由服务端直接写 Zarr。`server_metadata.json` 同时仍声明 `routing_capture_supported=true` 并广告真实响应 shape，NPZ 本身没有“placeholder/unavailable”标志。

影响：读取 `actions`、`state`、`sim_state` 没问题；读取 NPZ 的路由字段会得到语义错误但形状合法的零值。路由消费者必须改读 `server/routes.zarr`。

### 6.3 中：manifest 与文件名 resolver 过时；README 已更新但仍有残余错误

审计开始时 README 仍称“空骨架”；09:13 UTC 的并发更新已补入 76 GB 库存、hidden、`pin-*`、CALVIN 探针、NPZ 占位和真实文件名，因此这些不再算当前 README 缺陷。仍存在的实际问题是：

- manifest 的 mtime 仍为 `2026-08-14 17:33:08 UTC`；没有 5 个 `hidden.zarr`（占 96.3180% bytes）、4 个 `pin-*` run 和 `calvin_d/`。
- manifest benchmark keys 是 `calvin_d_d, libero_10, libero_goal, libero_object, libero_spatial`，没有实际有数据的 `calvin_d`。
- manifest 声明 `episode_NNNN.npz`；`corpus_layout.episode_filename()` 返回 `episode_%04d.npz`，但写入端 `rollout_with_routes.py:268` 使用 `%02d`（最小宽度 2）。全 hub 实际有 4,098 个 2/3 位 episode 文件、0 个固定 4 位文件。
- 新 README 写“`pin-*` 各臂 `hook_verified_calls` 为 0”，但实际 `pin-base=480`，只有 `pin-off=0`、`pin-on=0`、`pin-smoke=0`。这是文档计数错误，不影响 Zarr payload。

影响：用 `corpus_layout.episode_filename(0)` 得到 `episode_0000.npz`，实际文件为 `episode_00.npz`，会 miss；直接字典序枚举还会把 `episode_100.npz` 排在 `episode_11.npz` 前后不符合数值顺序。应始终以 `summaries.json` 的 `episode_index` 为当前权威顺序。

### 6.4 低：`pin-base` 与 `right-16x32` 没有独立 payload 信息

同一 ramekin 任务上：

```text
routes.zarr: 2421/2421 files byte-identical
               224,076,746 identical bytes
NPZ:          512/512 SHA-256 identical
                 3,517,463 identical bytes
identical payload total = 227,594,209 bytes
pin-base directory size = 228,033,276 bytes
right-16x32 directory size = 8,011,491,413 bytes
```

两边 512 条 summary 的 `episode_index`、`init_state_id`、`flow_noise_seed`、`success`、`action_steps`、`inference_calls`、`first_action_chunk_sha256` 都逐条相同；成功数均为 500，结局 phi=`1.0`。meta、server instance 与 wall time 不同，较符合 bit-deterministic 重跑而非简单复制，但该对照臂没有提供独立统计信息。

### 6.5 低：Zarr 内混入 Jupyter checkpoint 文件，但正式 metadata 未受损

路径：`.../open_the_middle_drawer.../hidden.zarr/hb_hidden/.ipynb_checkpoints/zarr-checkpoint.json`

```text
live zarr.json:       702 bytes, shape[0]=6452
checkpoint metadata: 701 bytes, shape[0]=184
SHA-256: f51790e8... vs 1250ec67...
after normalizing only shape[0]: JSON objects equal=True
```

checkpoint mtime 比 live metadata 早约 2 小时，说明它是采集进行到 N=184 时留下的旧快照。codec、chunk grid、dtype、fill value 等其他字段完全相同；当前 live `zarr.json` 与 6,452 行数据和 807 个 `hb_hidden` chunks 一致。因此这是 store 卫生问题，不是 metadata 被错误编辑的证据。

## 7. 使用注意

1. **`hb_selected_prob` 不是和为 1 的 combine weight。** 它是 32 路 softmax 中所选 top-4 的原始 gather；目标 run 的总质量均值为 0.150806。模型实际混合时使用 `raw / raw.sum(-1)`。
2. **`hb_expert_ids[...,0]` 不是 top-1。** 上游调用 `topk(sorted=False)`；目标 run 有 3,120,829 / 5,677,760 (`54.965849%`) 的站点 slot 0 不是 argmax。真 top-1 应由 `hb_router_probs.argmax(-1)` 得到。
3. **fp16 平票必须用 tie-aware 规则校验。** 对目标 run 用 CPU PyTorch 从已存 fp16 概率重新 top-4：1,002,955 / 5,677,760 (`17.664625%`) 的站点得到不同集合；按四个成员槽位计是 1,027,606 / 22,711,040 (`4.524698%`) 被替换。所有不一致站点都满足 fp16 下 4th 概率等于 5th 概率，`mismatch_without_p4_eq_p5=0`，且严格 top-4 违例为 0。此前“约 4.3% 站点重排”的说法混用了分母：4.x% 接近成员槽位率，不是站点率。
4. **AS 恒定是输入契约，不是坏数据。** 当前 LIBERO checkpoint 在固定 24 维 data mask 上每层只选一个专家。
5. **`REPORT.md` 的 LIBERO-Long `-34.6 pp` 是错位比较。** 57.8% 是单个 t08 moka-pots 任务的 `296/512`；92.4% 是全套件参考，不能据此判断缓存协议错误。
6. **只读打开 Zarr 时显式使用 `mode="r"`。** CALVIN 例外地不能信任 `episode_id`；先由 `sequences.jsonl` 构造边界。

## 8. 可选修复建议（本次均未执行）

1. **CALVIN：新建版本，不要原地改。** 从 80 条 `sequences.jsonl` 重建 episode/sequence id，写成 `routes-v2` 或独立的校验过的 boundary sidecar；保留 `routes-v1` 作为原始证据。后续采集应把 `sequence_index` 传给 server，并让 validator 同时核对 Zarr、sidecar 和 `sum(inference_calls)`。
2. **NPZ 契约：消除静默占位。** 两种可接受方案是实际写入 `[T,10,8,10,4]` 路由，或彻底删除 NPZ 的三个路由 key 并增加机器可读的 `routing_location=server/routes.zarr`、`routing_in_npz=false`。不要继续用合法 dtype/shape 的零数组表示 unavailable。
3. **统一 episode 文件名。** 让写入端与 `corpus_layout.episode_filename` 使用同一格式；若选择 `%04d`，为现有文件提供显式迁移表或兼容 resolver，不要靠字典序猜 episode。
4. **自动生成 manifest/README 的 inventory 部分。** 将 09:13 更新后的 README 作为当前人类可读入口，并从盘上元数据生成 manifest 的 populated run、coverage、hidden store、实际 bytes、pin arm、CALVIN 真实目录；同时修正 README 的 pin-base hook 计数。
5. **校验器采用集合和边界平票规则。** 检查 id 范围、行内唯一、selected gather、`min(selected) >= max(unselected)`；不要要求 `topk(sorted=False)` 的 slot 顺序，也不要在 fp16 4th/5th 平票时指定唯一集合。
6. **存储卫生。** 用户确认后再决定是否去重 `pin-base` 的 227.6 MB payload，以及是否移除 `.ipynb_checkpoints`；在此之前不要删除原始证据。

## 9. 关键复核输出索引

后续工作可直接复用以下已验证事实：

```text
target HB sites                         5,677,760
id range / duplicate sites             0..31 / 0
selected gather mismatches              0 / 22,711,040
probability row-sum max abs error       2.86102295e-4
entropy max abs error (renormalized)    1.26743317e-3
AS hidden mismatches                    0 / 68,133,120
AS expert ids by layer 0,1,16,17        [2], [0], [0], [1]
target episodes / rows per episode      512 / 12..14 (mean 12.6015625)
LIBERO episode/control exact checks      9 / 9
routes-hidden exact alignment            5 / 5
right-16x32 N total                     51,308
all run call-count equality              10 / 10
Zarr stores / arrays                    15 / 100
logical / physical chunks              47,435 / 47,320
CALVIN episode ids                     3,621 zeros; 80/1000 sequences complete
NPZ placeholder audit                  4,098 / 4,098
hidden bytes share                     77,809,144,644 / 80,783,600,041 = 96.3180%
```
