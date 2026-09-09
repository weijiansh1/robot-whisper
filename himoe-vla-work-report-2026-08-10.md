# HiMoE-VLA 工作报告

日期：2026-08-10
范围：论文复现现状 + MoE 路由分析三阶段 + 存储与工件清单
论文：arXiv:2512.05693v2，官方仓库 [ZhiyingDu/HiMoE-VLA](https://github.com/ZhiyingDu/HiMoE-VLA)

---

## 一句话现状

**复现侧**：CALVIN D→D 完全复现（Sum 4.008 vs 论文 3.98）；**FLOWER 基线行也已复现**
（Sum 4.301，五个论文值全部落在 Wilson 95% 内）；LIBERO 三个 suite 在修复 wrist 槽位错配后
差距收到 0.8–4.0 个百分点；只有 Table 1(a) 的 `FLOWER + HiMoE` 一行经审计
**不可由公开工件严格复现**——缺的是模型集成代码与权重，不是评测能力。

**分析侧**：三轮实验把"HiMoE 的 MoE 到底在做什么"收敛成一句话——
**MoE 的计算有用，但路由选择、以及两条分支中的任何一条，单独都不重要**；
路由状态携带信息，但那是一个**晚的读出**，不是早期预警。

---

## 一、论文复现

### 1.1 CALVIN D→D（完全复现）

1000 条官方序列、4008 个子任务、零基础设施失败。论文五个等级的值**全部落在本次复跑的
Wilson 95% 区间内**。

| 指标 | 本次 (1000 条) | Wilson 95% | 论文 Table 1(a) | 差值 |
|---|---|---|---|---|
| ≥1 任务 | 0.939 | [0.922, 0.952] | 0.938 | +0.001 |
| ≥2 任务 | 0.869 | [0.847, 0.889] | 0.866 | +0.003 |
| ≥3 任务 | 0.799 | [0.773, 0.823] | 0.794 | +0.005 |
| ≥4 任务 | 0.734 | [0.706, 0.761] | 0.723 | +0.011 |
| ≥5 任务 | 0.667 | [0.637, 0.696] | 0.659 | +0.008 |
| **Sum** | **4.008** | — | **3.98** | **+0.028** |

协议逐项对照官方 `examples/calvin/main.py`：序列宇宙 SHA-256 锁定、初始状态映射重实现
（pyhash fnv1_32 UTF-16LE 语义，测试向量锁定）、EP_LEN 360、`resize_with_pad` 224。

### 1.2 LIBERO Section 4.1（主差距已定位并修复，未完全复现）

根因是 `LiberoInputs` 的 **wrist 相机槽位错配**，不是 checkpoint 本身。

| Suite | 论文 | 修复前 500-run | 修复后 500-run | 残差 |
|---|---|---|---|---|
| Goal | 98.6% | 82.0% | **97.8%** | −0.8 pp |
| Spatial | 98.2% | 81.2% | **94.2%** | −4.0 pp |
| Object | 99.4% | 79.2% | **96.6%** | −2.8 pp |
| Long | 95.8% | 未运行 | 未运行 | 本地缺完整权重 |

已排除的残差解释：robosuite/MuJoCo 小版本（paper-pin 只把 Goal 83% 改到 82%）、
horizon 不足（Spatial task 4 延长到 1000 步仍 8/8 失败）、replan 步长
（Spatial replan 5 只到 95.6%）、服务器随机漂移（fresh-server 动作/状态/帧/MP4 精确重放）。

### 1.3 Table 1(a) 的 FLOWER 基线行（**已复现**，2026-08-10）

判据与 CALVIN 主行一致：论文值是否落在复跑的 Wilson 95% 区间内。

| 等级 | 本次 (1000) | Wilson 95% | 论文 FLOWER 行 | model card |
|---|---|---|---|---|
| ≥1 | 0.981 | [0.971, 0.988] | 0.974 ✓ | 0.984 ✓ |
| ≥2 | 0.935 | [0.918, 0.949] | 0.924 ✓ | 0.940 ✓ |
| ≥3 | 0.869 | [0.847, 0.889] | 0.869 ✓ 精确相等 | 0.879 ✓ |
| ≥4 | 0.790 | [0.764, 0.814] | 0.813 ✓ | 0.817 **✗** |
| ≥5 | 0.726 | [0.698, 0.753] | 0.749 ✓ | 0.741 ✓ |
| **Sum** | **4.301** | — | 4.329（分级相加） | 4.361 |

**五个论文值全部在区间内，这一行复现成立。** 42,970 次推理 / 410,646 个环境步。

**4.301 与三个参考值的差距，已由上游日志完全解释。** FLOWER 作者把 `training.log`
一并传到了 `mbreuss/flower_calvin_d`，里面有 6 次完整的 1000 序列评测：

| Epoch | 10 | 15 | 20 | 25 | **30** | **35（末轮）** |
|---|---|---|---|---|---|---|
| 分级相加 | 3.906 | 4.214 | 3.995 | 4.324 | **4.361** | **4.286** |

**Epoch 30 逐位等于 model card 印的那组数（4.36）；而发布的 `model.safetensors`
（仓库里只有这一个模型文件）对应的是末轮水平。** 也就是说 **card 报的是最优轮次，
发布的权重是末轮**。我们的 4.301 落在末轮 4.286 旁边（+0.015，而该评测跨轮次波动
本身有 ±0.2 量级），说明复现是准的，不存在"跑低了"。

差距还有方向性，且上游日志上同样存在——逐级对比论文那一行，本次是
+0.007 / +0.011 / 0.000 / −0.023 / −0.023（Sum −0.028），上游末轮是
+0.004 / +0.001 / −0.001 / −0.020 / −0.027（Sum −0.043）：
两边都是浅层持平或偏高、深层偏低，而**本次复跑比作者自己的末轮日志更接近论文**。
所以准确表述是：**论文引用的那一行高于发布权重的实际水平，高出部分集中在深层。**

同一种"card 用最优轮次、HF 放另一个 checkpoint"的错配，在 FLOWER 的 ABC→D split 上
已被第三方报告（[flower_vla_calvin#20](https://github.com/intuitive-robots/flower_vla_calvin/issues/20)，
复现低 0.2 Sum），**该 issue 至今无维护者回复**。

论文那一行（分级相加 4.329、印刷 4.35）与日志里任何一轮都对不上，多半转引自 FLOWER 原论文，
这也解释了它表内为何不自洽。

执行方式：1000 条序列按 `--start-index/--end-index` 分 6 片并行（约 3 小时，
单进程需约 7 小时），`scripts/run_flower_shards.sh` 起、`scripts/merge_flower_shards.py` 合并。
合并强制校验模型身份逐位一致与序列覆盖完整无重复。
**需披露**：shard d（序列 838–999）跑在 `1g.35gb` 分片、其余在 `2g.35gb`，
逐序列轨迹在两种分片间不逐位可复现（见 §6.1），对 1000 条独立序列的聚合率无影响，
已记入 `summary.json` 的 `shard_gpu_partitions`。

产物：`himoe-flower-calvin/formal-artifacts/flower-baseline-1000/`。

### 1.4 Table 1(a) 的 `FLOWER + HiMoE`（仍不可严格复现，但缺件清单已更正）

> **2026-08-12 更正。** 本节原先写"缺模型集成代码与层布局配置"，**这是错的**。集成代码公开在
> [`ZhiyingDu/flower_himoe`](https://github.com/ZhiyingDu/flower_himoe)（同一作者的另一个仓库，
> 2026-01-27，MIT）。08-10 的检索只覆盖了 HiMoE-VLA 仓库本身、作者 HF 主页和论文里的唯一
> URL，**没做跨仓库的 GitHub 搜索**。详见
> `himoe-flower-calvin/FLOWER_HIMOE_REPRODUCTION.md` 的"2026-08-12 更正"节。

集成方式：`FlowBlock` 增加 `use_moe` 开关，`if use_moe: self.mlp = DeepseekMoE2(...)`，
新增 `modeling_deepseek.py`（含 `MoEGate_load_bal`）。

**但公开实现与论文描述的 HiMoE 不一致**：只有 HB 门控、**没有 AS 门控**
（`mutual_info`/`data_mask` 零命中）、**没有层级**（18 层全部 `use_moe=True`）、
专家数 **8 / top-2**（HiMoE 发布版是 HB 32/4 + AS 3/1）。论文正文称这一行是
"Replacing FLOWER's dense action expert with **HiMoE**"，而 HiMoE 的定义就是那个三层级结构。

**更正后仍然缺**：训练权重（`flower_himoe` 无 release、无 checkpoint）、训练运行清单。

**本地不能自训**：`HulcDataModule` 要 CALVIN 原始布局 + `rel_actions`，即 task_D_D **177.4 GB**
（本机可用 67 GB；HF 上的 `calvin_d_joint` 是关节角 Parquet，不能替代）；
算力上 `devices: 8` / 4 GPU 12 小时的配置，在本机两个 MIG 分片（合计 28 SM，约 1/15）
折算约 **30 天**。

**未能查证的来源**：这篇投过 ICLR 2026（被拒），OpenReview forum `TX3oGD99CJ` 有补充材料，
但**从本机不可达**（浏览器验证，curl http=000）。

论文目标值 0.979 / 0.943 / 0.904 / 0.859 / 0.801，Sum 4.49。

论文只说"把 FLOWER 的稠密动作专家换成 HiMoE"。**公开工件缺四样东西**：模型集成定义、
`FLOWER + HiMoE` 训练权重、FLOWER 专用层布局与权重迁移配置、训练运行清单。
直接评测公开 FLOWER 或主 HiMoE checkpoint 得到的是另一个模型的结果，不能标记为这一行的复现。

已完成的部分：评测器扩展为同时支持 HiMoE 与 FLOWER（锁定官方 1000 条序列、校验
checkpoint SHA-256、FLOWER 用原始双相机图像 + `10×7` 动作块 + `cartesian_rel`）。
发现并绕过一个发布配置不一致：FLOWER 配置写 `query_seq_len=100`，但发布权重全部 18 层的
RoPE 缓冲区长度是 120；适配器以权重张量形状为准，`strict=True` 完整加载。

机器可读证据：`himoe-flower-calvin/manifests/flower-himoe-public-artifact-audit-v1.json`。

---

## 二、MoE 路由分析：三阶段

全部在 LIBERO-Goal task 0、`HiMoE-VLA-Libero-Goal` checkpoint
（`98ee29d0…`）上。带 `summaries.json` 的批次共 **698 个 episode**
（阶段一+二 498，今日四臂消融 200）。

### 阶段一（2026-08-07，438 集）：计算重要，路由不重要

在有动态范围的操作点 `released-left`（基线 28%）上做行为学消融：

| 条件 | 成功 | McNemar p |
|---|---|---|
| 未改动 | 14/50 = 28% | — |
| HB router → 均匀随机 top-4 | 16/50 = 32% | 0.83（10↓/12↑，对称） |
| 整个 HB-MoE 子层置零 | 5/50 = 10% | 0.049（13↓/4↑，单向） |

结论分两句，不能合并：**那 8 个 HB MLP 子层在做有用的计算**（置零掉 18 点）；
**用哪 4 个专家去做无关紧要**（随机替换不降反升）。

同时发现 AS-MoE 在 LIBERO 上是**常量路由器**（`data_mask` 恒定），其中 layer 1 的专家
选择由 bf16 舍入决定（三个 logit 精确塌成 0.25，最大两行只差 1.22e-4）。

路由状态可解码：wrist layout **100%**，扣除场景后预测成败 **0.710**（置换 p = 0.006）。
但第五节证明改变路由不改变行为——**路由是镜子，不是方向盘**。

### 阶段二（2026-08-09，64 集单场景）：步 0 没有信号

固定 init state 24、64 次不同 flow 噪声，成功 **23/64 = 35.9%**。
**实测确认**控制步 0 的机器人状态在 64 集之间逐位相同（`max|Δ| = 0.0`），
动作块不同（每维 std 0.002–0.017）——同一输入，只有噪声不同。

| 步 0 特征 | 平衡准确率 | 置换 p |
|---|---|---|
| 动作 token，8 层全部 | **0.423** | 0.92 |
| state token，8 层全部 | 0.500 | 1.00 |
| 只用 router 熵 | 0.452 | 0.78 |
| 逐层（2/3/4/5/12/13/14/15） | 0.389 – 0.510 | — |

**对照：同一批数据上的 flow 集合。** 步 0 的 64 个动作块就是 K=64 的免费集合，
用相对集合均值的偏离量预测成败 **AUROC = 0.577**（成功组 0.0864 vs 失败组 0.0845）。
**两条路线在步 0 都是空的。**

分离要到第 10–14 个控制步才出现（组间/组内分离度 0.285 → 1.052，前 17 步解码 0.898，
p = 0.001），而那时已走了 140 个环境步，抽屉开没开基本定局。
**路由是一个晚的读出。**

### 阶段三（2026-08-10，200 集四臂配对）：两条分支是冗余的

阶段一留下的问题：那 18 点由 shared 分支还是 routed 分支承担？
四个 arm 在同一 MIG 分片、同样 50 个 init state 与噪声种子上配对：

| 条件 | 成功 | Wilson 95% | McNemar（降/升） |
|---|---|---|---|
| 未改动（基线） | 14/50 = 28% | [17.5%, 41.7%] | — |
| **shared 置零**（只剩 routed） | **15/50 = 30%** | [19.1%, 43.8%] | p = 1.000（7↓/8↑） |
| **routed 置零**（只剩 shared） | **12/50 = 24%** | [14.3%, 37.4%] | p = 0.815（10↓/8↑） |
| 整个 block 置零 | 7/50 = 14% | [7.0%, 26.2%] | p = 0.143（12↓/5↑） |

**单独去掉任何一条都测不出影响，两条一起去掉才掉点。** routed 分支在 action token 上
只占 block 输出范数的 5.6%，**单靠它就能维持基线性能**；shared 分支占 67–93% 范数，
去掉也不掉点。所以这个 block 的作用与幅度无关——它只需要"有某个非线性变换在那里"。

消融确实生效：三个 arm 的不一致配对数是 15/18/17（共 50 对），轨迹被明显改变；
只是 shared_off / routed_off 的翻转对称（7↓/8↑、10↓/8↑），block_off 才单向（12↓/5↑）。

**一个必须记录的负面结果**：已发表的 block_off 效应**没能复现到 p < 0.05**
（本次 7/50 = 14%，掉 14 点，p = 0.143；已发表 5/50 = 10%，掉 18 点，p = 0.049）。
方向和幅度接近，显著性落在线的另一边。原报告自己就标注了"勉强越过 0.05"这个限制。
**结论：block 消融的效应方向可信，但 50 集不足以定量，p = 0.049 不应被当作稳固结论引用。**

### 2.4 新发现（尚未写入任何报告）：路由的尖锐性全在 state token 上

在做存储基准时发现 router 概率最大值达 0.956，与"路由几乎均匀"矛盾。查证后有清晰结构：

| | 熵/ln32 | 有效专家 | top-1 |
|---|---|---|---|
| **state token（token 0），早层 2–5** | **0.80–0.89** | **16.1–21.9 / 32** | 0.20–0.32（最大 0.956） |
| state token，晚层 12–15 | 0.984–0.989 | 30.3–30.8 | 0.057–0.061 |
| **动作 token（1–10），全部 8 层** | **0.998–0.999** | **31.8–31.9** | 0.037 |

**动作 token 上没有任何一个路由站点的 top-1 超过 0.5**；state token 上有 4.4%。
两个批次、两种 wrist 布局（`released-left` / `checkpoint-right`）都成立。

阶段一报告的表格（熵/ln32 = 0.981–0.998）是把 1 个尖锐 token 和 10 个平坦 token
**池化**的结果，读起来像"处处均匀"。拆开后结论其实**更强**：
在真正产生动作的 token 上，router 均匀到离满熵不足 0.2%。
这也和阶段一 §4 已有的发现自洽——routed 分支的范数占比在 state token 上高得多
（0.55 vs 0.26 @ layer 2；0.38 vs 0.056 @ layer 15）。

**必要限定**：state token 的末层输出确实被丢弃（`moevla.py:483` 只取后 10 个），
但它的中间层表示通过 self-attention 影响动作 token，所以不能说"唯一做决策的地方被扔掉了"。

**待办**：阶段一报告第三节的表格与措辞需要按此修正。

---

## 三、可视化产物

| 文件 | 说明 |
|---|---|
| `himoe-route-capture/viz/himoe-routing-viewer.html` | **单文件交互页面（2.85 MB，双击即开）**，6 个联动面板 |
| `himoe-route-capture/viz/index.html` + `data.js` | 分体版，供本地 HTTP 服务 |
| `figures/within64-s24/routing_evolution.mp4` / `.gif` | 路由状态随控制步演化的动画 |
| `figures/within64-s24/f1…f5*.png` | 静态图：router 有多平 / 专家×控制步 / 群体散开 / top-4 翻搅 / 去噪轴 |

页面最直接的一屏：64 次抽样投影到共同主成分平面，带从第 0 步到当前步的轨迹尾迹——
64 条轨迹从同一点出发，沿几乎相同的路径走过前 10 步，在第 10–14 步分成两簇。

所有面板画的都是**减去各专家自身均值后的残差**（原始概率贴着 1/32，不放大看不见），
页面里有"原始概率"开关可以切回去看它有多平。

---

## 四、存储与工件清单

### 4.1 持久化 cache（`work/himoe-vla/himoe-vla-cache/`，共约 67 GB）

| 目录 | 大小 | 内容 |
|---|---|---|
| `himoe-libero-bridge/` | 36 GB | Goal/Spatial/Object checkpoint 各 8.1 GB、LIBERO 与 HiMoE 上游、py3.8+py3.11 环境、正式评测工件 |
| `openvla-repro-cache/` | 17 GB | OpenVLA 阳性对照环境与模型 |
| `himoe-calvin-alignment/` | 8.5 GB | CALVIN-D checkpoint、calvin 仓库+官方补丁、py3.8 客户端环境 |
| `flower-calvin-d/` | 3.8 GB | FLOWER 基线权重 |
| `florence2-large-code/` | 1.5 GB | FLOWER 依赖 |
| `uv-python/` | 240 MB | 全部 venv 依赖的解释器仓库（缺它所有虚拟环境失效） |

⚠️ 重启后 overlay 上的软链接会消失，先跑
`bash /home/jovyan/work/himoe-vla-cache/restore-cache-symlinks.sh`。

### 4.2 路由 trace（`himoe-route-capture/`，135 MB，容器重启不丢）

**698 个 episode / 20 个带 summaries 的批次**（另有 `fullprob-right`、`state-tier`、
`within64-s24` 三个纯服务器端采集目录，episode 计数在配对的 client 目录里），三种格式并存：

| 格式 | 批次 | 内容 |
|---|---|---|
| `routes.zarr`（Zarr v3 + Zstd） | `within64-s24`（1682 控制步 / 68 MB）、`fullprob-right`（312 步 / 13 MB） | **完整 32 维 router softmax** `[T, 8层, 10去噪步, 11token, 32专家]` fp16，加 top-4 序号 / 选中概率 / 熵 / AS 层 |
| 逐集 `episode_NN.npz` | `left50`、`right50`、`within`(80)、`left`、`right` 共 196 集 | top-4 序号 + 权重 `[T,10,8,10,4]` + state + actions |
| 单个 `state.npz` | `state-tier` | 四层完整 MoE 状态，27 个数组（含 block/routed/shared 范数分解） |

**两个必须知道的坑**：
1. `within64-s24-client/` 里的 `expert_ids` 是**全零占位**——那次是服务器端采集，路由在
   `within64-s24/routes.zarr`。用 `within64_lib.load_run()` 把两边配起来。
2. **全部 10 个消融批次（`abl-*` / `abl2-*`）没有路由数据**，只有成败与轨迹。

路由 trace 单独没有意义，必须与同目录的 `summaries.json`（成败/步数/init_state/噪声种子）
和 `server_metadata.json`（wrist layout / checkpoint SHA）配对使用。

### 4.3 "全部路由状态"的量级（供后续决策）

HB 每控制步有 8 层 × 10 去噪步 × 11 token = **880 个路由站点**（hidden 1024 维）：

| 档 | 内容 | 每控制步 | 一次 64 集 run |
|---|---|---|---|
| ① router（现状） | 32 维完整 softmax | 56 KB | 95 MB 原始 / 68 MB 落盘 |
| ② + gate 输入 | 每站点 1024 维隐状态 | 1.80 MB | +3.0 GB |
| ③ + block 内部 | shared/routed/block 三个输出 | 5.4 MB | +9.1 GB |
| ④ + per-expert 输出 | 被选中的 4 个专家各自输出 | 7.2 MB | +12.1 GB |

**就路由而言 ① 档已经完整**：router 是 `logits = W_gate @ x`，`W_gate` 固定在 checkpoint 里，
有 32 维 softmax 就能反推 top-4、权重、熵、margin。1024 维 gate 输入大 32 倍却不提供额外的
*路由* 信息——它提供的是隐状态本身。②–④ 只在需要**反事实**时才必要。

现存的 `hb_selected_prob` 和 `hb_entropy` 是 `hb_router_probs` 的纯导出量（约 13% 冗余）；
`hb_expert_ids` 要留，因为 bf16 舍入会决定 top-k，argsort 反推未必等于模型的实际选择。

### 4.4 存储格式实测结论

在真实 trace 上重做了基准（`bench_formats.py`）：

- **npz 的随机读慢 500 倍**（4.88 s vs 0.009 s，抽 32 个随机控制步），因为必须整体解压。
  它只适合"一集一个文件"的写入端——而且 LIBERO 客户端是 py3.8 环境，**根本没装 zarr**。
- **byte-shuffle 在四类数组上全赢，bitshuffle 全输**（权重 2.21× vs 1.94×、
  32 维分布 1.93× vs 1.67×、熵 7.67× vs 6.60×、序号 2.31× vs 1.69×）。
  老报告"bitshuffle 有害"的结论对且可推广，但当初**只测了 bitshuffle**，
  漏掉的 byte-shuffle 每类还能再拿 8–34%。
- **`Blosc(zstd, clevel=1, shuffle)` 对现状是严格的帕累托改进**：
  体积 −7%、编码时间 −40%。clevel 6/7/8 被 5 严格支配（更慢且更大），
  clevel 9 是断崖（354.6 ms/chunk，生产模式下占推理 27.8%）。
  **Blosc 的 clevel 不是 zstd 的 level**，它非线性映射到后端范围。
- 更激进：`Transpose(去噪轴到末位) + FixedScaleOffset(→u2) + PCodec` → **−29%**，
  误差 1.4e-5（残差中位数的 0.7%，比 fp16 自身在 0.031 附近的 ulp 1.5e-5 还小）。
  注意 `Delta` 必须先 `Transpose`——它按内存最后一维差分，而最后一维是不相关的专家轴。
- ZFP 不适用（最多 4 维，且数据沿专家轴是噪声状而非光滑场）；
  BitRound keep=8 的误差 9.8e-4 已达残差中位数的一半。

**当前建议**：router 档保持不动。采集总开销实测 2.57 ms/控制步（占推理 717 ms 的 0.36%），
这个量级下不值得动。等决定要不要存 ②–④ 档再一起改——那时写入会变成真瓶颈，
需要异步 flush（现在 flush 同步挂在推理路径上，每 32 步一次停顿）。

---

## 五、本轮产出的工具

| 文件 | 用途 |
|---|---|
| `within64_lib.py` | 把服务器端扁平 zarr 按客户端 `inference_calls` 切回逐集（服务器不记 episode 边界），带总数校验与 `--allow-partial` |
| `within64_analyze.py` | 步 0 / 前缀解码（ridge 探针 + 闭式 LOO + 1000 次置换）、锐度、churn、动作集合对照、跨 MIG 复现性 |
| `within64_figures.py` / `within64_animate.py` / `within64_export.py` | 静态图 / 动画 / 网页数据打包 |
| `run_branch_ablation.sh` | 四臂分支消融驱动，串行起停服务器、断点续跑 |
| `analyze_branch_ablation.py` | 配对 McNemar 精确检验 + Wilson CI + 分解读出 |
| `bench_formats.py` | 格式基准：体积 + 写 + 全读 / 抽步 / 抽层三种读模式 |

---

## 六、方法学教训（都是这轮踩出来的）

1. **跨 MIG profile 不逐位复现。** `1g.35gb`(12 SM) 与 `2g.35gb`(16 SM) 上跑同一个 seed，
   聚合完全稳定（基线两次都是 14/50 = 28%），但**逐集有 6/50 成败翻转、11/50 步数不同**。
   SM 数变了 cuBLAS kernel 选择就变。做配对检验必须把 profile 钉死，
   拿新 arm 配对旧基线会有 12% 的翻转来自硬件。

2. **长度即标签。** 成功的集一开完抽屉就结束（17–21 控制步），失败集跑满 30 个。
   任何跨整集的特征都会先读到长度。所有解码都限制在等长前缀内，
   群体曲线在第 17 步之后标灰并说明"那里在比较不同的集合"。

3. **指标的粒度决定结论。** top-4 churn 必须**按单个路由站点**算（得 0.25–0.31，
   与已发表的 0.16–0.22 同量级）；把一整个控制步的 440 个 slot 并起来看会得到 0.91 的假象——
   近乎均匀的 router 抽 440 次几乎覆盖全部 32 个专家。

4. **池化会掩盖结构。** 熵按 11 个 token 池化 → "处处均匀"；拆开才发现尖锐性全在 token 0
   （见 §2.4）。同理，残差热图必须**按层**中心化，混合 8 层中心化会把各层自身的偏好
   变成恒定横条，看起来像结构其实不随时间变。

5. **量化的误差必须对着信号量。** 残差 uint8 看起来省 92%，但量化步长 7.2e-3 是残差中位数
   1.9e-3 的 4 倍——它压得好正是因为把大部分数据量化成了 0。

6. **HiDPI canvas**：只把 `canvas.width` 乘 devicePixelRatio 却对上下文做 `setTransform(r,r)`，
   会在 2× 屏上静默裁掉每个面板下方 1−1/r，恰好是 x 轴和图例。DPR=1 的截图完全看不出来。

7. **codec 的直觉不可靠。** "等级越高越好"在 Blosc 上错得离谱——必须实测整条曲线。

---

## 七、未决与下一步

**修正类（应尽快做）**
- 阶段一报告第三节的锐度表格与"路由几乎均匀"的措辞，按 §2.4 拆分 token 后修正。

**实验类（按价值排序）**
1. **把 64 次协议搬到 OOD 输入上。** 步 0 的零结论是在"输入完全相同"下得到的，
   这恰好是路由探针最没有优势的设定——它测的是 aleatoric。真正该测的是
   **epistemic**：换光照 / 干扰物 / 错误语言指令，看路由在步 0 能否分辨见过/没见过。
   已有强正对照（路由 100% 解码 wrist layout）。
   关键是**步 0 不需要 rollout**，一次前向就有——检测部分 15–30 分钟即可，
   只有它为真才值得花 ~2 小时跑失败标签。
2. **把 flow 集合判据做全。** 目前只测了"偏离集合均值"一个标量，AUROC 0.577
   只否证了最朴素的用法。应补：按维度归一化的方差、只看夹爪维、chunk 内时序方差。
   纯重算，0 GPU。
3. **加集数收紧 block_off。** 唯一有真实效应的 arm 却欠功效。每任务只有 50 个 init state，
   只能靠噪声重复（50 × 3 = 150 集/arm）。只跑 `none` 和 `block_off` 约 2.5 小时。
   `shared_off` / `routed_off` 不必加——两个都是近乎完美对称的零效应。
4. **换初始状态复现步 0 零结论**，确认不是 init state 24 特有。
5. 阶段一遗留：消融扩到 Spatial / Object / 其它 task。

**已知未完成项**
- `himoe_state_recorder.py` 的 `raw` 档从未在真实 rollout 上跑过。
- `serve_ablated_router.py --mode uniform` 从未运行。
- 采集格式分裂（16/18 批次 npz、2 个 zarr），分析代码要伺候两套。
- **服务器不记录 episode 边界**（`serve_with_recorder.py:78` 的 `episode_id` 永远回退成常量）。
  让客户端把 episode 序号发上来会更稳。
