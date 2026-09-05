# MoE routing phase portrait：协议（启动前冻结，2026-09-04）

> 主线：**物理世界负责定义 Trap，MoE 独立揭示 Trap 在模型内部的表型。**
> 物理轨迹只在离线定义事件（onset、类型）与验证；检测/分型/预测只读 MoE routing。
> 所有特征定义先于标签存在；标签只在评估阶段进入。不训练分类器/PCA/probe，
> 不从评估集调阈值。seed 20260903。

## 1. 数据（只用 cache 与 cache_new，全部 paper-right，可同表）

| 代号 | 位置 | 规模 | 用途 |
|---|---|---|---|
| `grid50x8` | `VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_{goal,long,object,spatial}/<task>/right-50x8-20260903` | 4 checkpoint × 10 任务 × 50 scene × 8 draw = 16,000 集，532 失败 | 跨 checkpoint 复现、任务流形、表型几何 |
| `main16x32` | `VLA_MUI_HUB/cache/HiMoE-VLA/*/<task>/right-16x32` | 5 任务 × ~512 集（goal×2, spatial×2, long/SCENE8×1） | 失败富集（SCENE8 216 失败）、事件对齐主语料 |
| CALVIN | `cache{,_new}/HiMoE-VLA/calvin_d/` | 只做 inventory，不进 v1 分析 | — |

**口径红线**（违反即作废）：路由只读 server `routes.zarr`（client NPZ 的 expert_ids 是占位）；
主用 32 路 soft probability，禁用 top-4 硬 ID 当主信号（bf16 平票 23.8%）；聚合一律 V2
（逐 (layer,token) 格先算距离再平均）；AUC 只在组内配对，跨组不池化；state token 与
action tokens 永不混平均；paper-right 语料绝不与 checkpoint-right（flow-lead、linemoe 等）同表。

组定义：`grid50x8` 组 = (checkpoint, task, scene)；`main16x32` 组 = (task, init_state)。
结局标签来自 client `summaries.json`。行↔集对账：按 `episode_id` 分组的 zarr 行数必须等于
该集 `inference_calls`，不符的集整集剔除并记录，不得猜。

固定时点（存活审计必附）：goal/object/spatial 用 t∈{8,12,16}；long 与 SCENE8 用 t∈{20,25,30}。

## 2. 表型向量 z_q（冻结定义，见 `phenotype/features.py`）

路由张量 P[l,f,u,e]：l=存储 HB 层 slice(4,8)（全局 12–15），f=10 flow 步，u=0 state /1–10 action，
e=32 专家。d9 = 最后去噪步。

| 符号(文档) | 名称 | 定义 |
|---|---|---|
| V | `late_flow_volatility` | f∈6–9 相邻 flow 的 weighted-Jaccard 距离，(l,u∈act) V2 均值（=trap 版） |
| A | `route_acceleration` | sqrt(P) 沿 flow 的二阶差分范数均值 /√2（=trap 版） |
| S | `gate_entropy`（低=尖） / `top12_margin`（高=尖） | d9、action tokens、深层均值 |
| C | `token_consensus` | d9：10 个 action token 两两 weighted-Jaccard **相似度**，45 对 × 层均值 |
| D^tok | `token_dispersion` | 同上但 Hellinger **距离** |
| D^layer | `layer_disagreement` | d9：4 深层两两 Hellinger，6 对 × token 均值 |
| Rec^(k) | `mob_k`, k=1..8 | d9 表示与 q−k 的 V2 Hellinger 距离（低=复现）；`mob_1` 即逐步 mobility |
| — | `mob1_w8` | mob_1 的 8 步拖尾均值 = 既有最强基线 route_mobility_w8（残差化用它） |
| F* | `flow_com` | flow 变化 profile d_f (f=0..8) 的质心 Σ(f+.5)d_f/Σd_f；另存 `flow_total`=Σd_f 与 (n,9) profile |
| — | `state_action_gap` | d9：state token 与各 action token 的 Hellinger 均值（延续既有信号） |

d9 表示 rep = sqrt(P[深层, d9, action]) ∈ (4,10,32)，存 `reps.npy` (N,1280) f16，供相位/流形/画像。

## 3. 提取物 layout

```
analysis_moe_phenotype/
├── phenotype/{features.py, stats.py, extract_task.py}
├── features/<corpus>/<suite>/<task>/{rows.npz, reps.npy, meta.json}
│     corpus ∈ {grid50x8, main16x32}
├── events/<corpus>/<suite>/<task>/events.csv   # 每集: success, loop_onset_q, static_onset_q, trap_onset_q, n_queries
├── audit/{AUDIT.md, data_map.json}
└── E*/…（各实验，见 §5）
```

rows.npz 含全部 z_q 标量 + meta 列（episode_id, control_step, scene, repeat, success, flow_noise_seed）。

## 4. 统计规则（全体实验共用，`phenotype/stats.py`）

- 检测类：组内成功–失败配对 AUC；置换单位 = **episode**（组内打乱集级标签，行广播）；
  2000 次；maxT 同时校正该实验**预先声明的整个检验族**；det AUC 只表可分性。
- 事件对齐类：事件集在 onset+lead 的值 vs 同组、同绝对 query 的全部 no-event 集（含其他失败），
  组级效应 + **组级 sign-flip** 2000 次 + maxT（=trap 口径）。
- 新颖性：凡主张"新信号"，必须对 `mob1_w8` 做组内秩残差后仍存活。
- 复现：头条主张需在 ≥2 个独立单位（不同 checkpoint 或不同语料）同向。
- 事件真值分级：SCENE8 用 `analysis_ssm/cache` 既有已验证代理表；其他任务用 state(8) 的
  降级代理（eef+gripper，无物体项），**必须先在 SCENE8 上对照完整代理报告 sens/spec**，
  且所有产物标注 `proxy_grade: degraded`。
- 内部相位类：模板只用成功集、leave-one-episode-out；必须做"内部计时器"证伪
  （同组同绝对 q 比较成败的 φ̂ 滞后；若 φ̂ 只是计时器则无滞后）。

## 5. 实验规格

### E2 loop 时序表型（Disperse–Switch–Collapse）→ `E2_loop_sequence/`
数据：SCENE8（主）+ grid50x8 long 各失败富集任务（事件代理 degraded）。
族：8 信号 {D_tok, C, S_ent, S_margin, V, A, D_layer, flow_com} × lead {−4..+2} = 56 格/语料。
预named 单元：复现腿 V@−2, A@−2；新腿 (a) D_tok@−3 升（先于 V/A），(b) S_margin@−2 升
（=unstable commitment：尖锐+切换并存），(c) C@+1,+2 升而 C@−2 不升（collapse 是 onset 后）。
另做次序统计：逐事件集的 D_tok/V/C 首次越过组对照 90 分位的时刻中位序 + 符号检验。
残差腿：全部对 mob1_w8 残差重跑。

### E3 static 表型（flattening + shared-support）→ `E3_static/`
同 E2 框架，事件=static。预named：S_ent@0 升（变平）、C@0 升（共享支持集）、
lag_periodicity 类由 mob_k 谱代替：min_k>1 mob_k @0 降。方向冲突时如实报。

### E4 复发脉冲列 + lock-in 亚型 → `E4_recurrence_lockin/`
(a) 逐集 mob_k 谱：late 窗（长语料 t 25–35 / 短语料 12–18）best-cycle 增益
g = mob_1 − min_{k≥2} mob_k；失败 vs 成功组内 AUC。
(b) 脉冲：V+A 的组内无标签 z（对组中位/MAD），阈 2；孤立脉冲（3 步内回落）在**成功集**中
的频率 = 误报审计；≥3 个间隔 ≤8 的脉冲 = pulse train，统计成败富集。
(c) lock-in 象限：late 窗均值，象限 = {V,A < 组成功中位 且 margin, (1−mob1_w8) > 组成功中位}；
失败中占比 vs 成功（组内置换）；与事件类型交叉表（loop 应属 switching 象限）。
族 = {cycle_gain, pulse_count, train_flag, lockin_flag, switch_flag} × 时窗(1) = 5 格 + 交叉表描述。

### E5 内部相位 φ̂ 与 slow-normal → `E5_phase/`
主语料 SCENE8（组=init_state，每组 32 draw）；复现：grid50x8 long 失败富集 2 任务（组=scene，
模板跨 scene 池化时把 scene 作配对层）。
φ̂_q：在同任务成功集（LOEO）reps 上取 top-5 NN 的归一化进度（NN 所在集的 control_step / 该集长）
中位；d_healthy = NN 距离中位。
检验族：{d_healthy, v̂=Δφ̂, stall(|v̂|<组成功 p10), regress(v̂<0), osc(v̂ 变号数)} × 固定时点 3 = 15 格
+ 1 格计时器证伪（同绝对 q 成败 φ̂ 差，组级 sign-flip）。
Slow-normal：成功集按长度三分位，慢成功 vs 快成功的 d_healthy（应不升）与 v̂（应 >0）。

### E7 任务条件流形 → `E7_task_manifold/`
grid50x8 每 checkpoint 内 10 任务：q∈{2..6} 的 reps，LOEO 最近邻任务判别准确率（chance 10%），
4 checkpoint 独立报告 + Wilson CI。若判别成立：逐失败集 mismatch 分数
= d(own-task 成功流形) − min d(other-task)；组内 AUC × 3 时点族；"wrong-manifold" 集清单（描述）。

### E8 相图 + 宏状态 + E9 MoE-silent 边界 → `E8_portrait_silent/`
Γ_q = 组内无标签 z 化的 (V+A, margin, C−D_tok, −mob1_w8, flow_com)。各轴组中位二分 → ≤32 宏状态
（稀疏则并到 16）。成功/失败转移矩阵、驻留 P(s→s)；loop/static 集 onset 前后的 basin 占用。
E9：逐失败集 max 信号偏离（8 核心信号在固定时点相对组内成功分布的百分位）；
silent = 全部 |pct−50|<40 且事件对齐无偏离；报 silent 比例 × 任务/套件 + 物理特征描述表。
E8/E9 均以描述+置换驻留差为主，不做多重新主张。

## 6. 环境与工程

- `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`（240 核机，不设会慢 100 倍）。
- zarr v3 API：`zarr.open_group(path, mode="r")`；float16 一律先转 float32 再算。
- 语料目录**只读**；产物只写 `analysis_moe_phenotype/` 自己的子目录。磁盘仅 ~12G 空闲，
  reps 用 f16，不落中间大文件。
- 图内文字用英文（容器 CJK 字体不保证），报告用中文。
- 每个实验产出 `report.zh.md + summary.json`，零结果照实写；缺数据就缩范围并声明，不造。

## 7. 诚实条款

某类 Trap 若在全部表型上无稳定偏离 → 记为 **MoE-silent**，不得用动作/物理特征补救后仍称
"MoE signal"。已知上限如实引用：组内跨 rollout 有效自由度 ~2；hidden 随机 32 维可匹敌 routing
（信息不独占，卖点是表型结构）；D^layer 预期近退化（cascade 延迟全 0），当负对照。

## 8. 修订记录

**Amendment 1（2026-09-04，先于一切事件对齐统计）**：审计证实降级 loop 代理不可信
（SCENE8 上 spec 0.077/0.081，93% 集误标——去掉物体项后正常取放-返回即触发），
static 降级代理可信（sens/spec 0.97+）。据此解冻"物体项来源"一条：允许用每任务
`sim_layout.json` 具名物体槽位构建 **obj-aware 代理**，阈值与 full 完全一致
（loop obj≤0.030、static obj≤0.005；goal 项经 1-Lipschitz 论证 + SCENE8 512 集
实证为冗余而省略）。准入条件：obj-aware 在 SCENE8 上与 full **零不一致**（已满足，
161/161 onset 一致）。事件表升级为 proxy_grade=objaware 后，E2/E4/E9 方可使用
非 SCENE8 任务的 loop/trap 事件；修订前的 degraded loop 列作废。
