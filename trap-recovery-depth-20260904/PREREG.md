# 报警之后多久干预还救得回来 — 预注册（2026-09-04，启动前冻结）

> 修订 3（2026-09-04，见下「等预算」一节）：各臂改为**等预算**，Δ 收紧为 {0,+4,+8}，
> trunk 改用 seed 区分。修订 2：触发点由**物理 loop onset** 改为 **MoE 报警点**（用户指定）。
> 修订 1（相对 onset 的 Δ → 绝对 fork query）连同修订 1 的动机一并作废，
> 记录保留在 git 之外的本文件历史段落末尾。两次修订都在看到任何救回结果之前做出。

## 问题

失败的 rollout，从**报警点**重抽噪声，与从**报警之后一段时间**重抽，哪个更容易救回？

这与 `himoe-vla_trap` 那批不同：那批从物理 loop onset 的 −4/−2/0 分叉（三窗口各 0/8），
用的是需要特权物体位姿才能算的物理判据。本实验用**部署时真能拿到的** MoE 报警点。

它同时回答一个悬着的问题：organization-sweep §7 判定**路由空间**里没有 trap 吸收集，
但**物理行为**上"陷得越久越难救"是否成立，从未测过。

## 报警器（冻结，取自 `triggered_fork_collect.py`，四个常数早于本实验冻结）

    d(t)  = 1 − |top4(t−1) ∩ top4(t)| / |top4(t−1) ∪ top4(t)|
            在 HB 12–15 层 × 10 个 action token、去噪步 9 上平均
    r(t)  = mean(最近 W=4 个 d) / mean(自身最初 W0=8 个 d)
    报警  = 连续 K=3 步满足 r(t) < θ=0.95 且已有 ≥ 2·W0 = 16 步历史

最后那个历史下限不可省：没有它，比值会在自己的基线还没成型时就被读取。
（`demo/rescue.json` 那条 trunk 的 `ratios` 在 index 12–14 已连续三步 < 0.95，
正是被该下限挡掉，真实报警落在 q32。）

报警器只读 `routing/expert_ids`，在客户端计算，**不使用任何特权仿真状态**。
top-4 身份只用于把一个站点与它自己上一步比对，重编号 32 个专家不改变任何 d(t)。

## 干预

在 fork 点重抽 flow noise（fresh Gaussian），其余一切不变——唯一已验证 on-manifold 的干预。
σ 固定为 1，不扫幅度网格。candidate k 在所有臂上共用同一相对噪声流（配对）。

## 设计（冻结）

- 任务 `libero_10` / task 8，**操作点 `paper-right`**，checkpoint sha `cdc2b21f…`，
  environment_seed 7，settle 10，replan 10，max_steps 520，max_trunk_queries 52。
- **只对失败的 trunk 分叉**（`--failures-only`）。trunk 成功即跳过，它回答不了这个问题。
- 五个臂，每臂 K=8，**每臂从各自 fork 点起统一给 20 个 query**（`--fork-budget-queries 20`）：

  | 臂 | fork 点 | 含义 |
  |---|---|---|
  | `triggered` | 报警点 q_a | 刚报警 |
  | `delayed_+4` | q_a + 4 | 报警后 4 步 |
  | `delayed_+8` | q_a + 8 | 报警后 8 步 |
  | `control` | 报警前均匀随机取一点（≥ 2·W0） | 时机对照 |

- **不干预对照**：原 trunk 自己跑完（已记录，免费）。
- **主端点截止线：fork 后 20 个 query**（等预算）。
- **次端点：原 cap 第 52 query**，逐 candidate 记 `within_original_cap`，同一批数据直接读出。
- 环境 horizon 抬到 `--max-steps 900`（= robosuite horizon 911），否则分支跑过 q53 会被
  `executing action in terminated episode` 拒绝。trunk 仍由 `--max-trunk-queries 52` 限制，
  行为逐位不变。

## 为什么必须等预算（2026-09-04 实测，启动前）

首轮不等预算的一条 trunk（i0，alarm q34）给出 `triggered 5/8 → +4 0/8 → +8 0/8 → +16 0/8`，
形状完全符合假设。但逐条查成功者步数后该结果作废：

    triggered  预算 18  成功 5/8  成功者用了 14, 15, 15, 15, 17 步
    +4         预算 14  成功 0/8  8 条全部跑满 14
    +8         预算 10  成功 0/8  8 条全部跑满 10
    +16        预算  2  成功 0/8  8 条全部跑满 2

**救回最快需 14 步**，而 +8/+16 在算术上不可能成功——纵使 trap 不存在也会是 0/8。
这些格子对"陷入深度"零信息量。等预算把该解释排除后（K=3 冒烟）：
`control 1/3 → triggered 2/3 → +4 0/3 → +8 0/3`，各臂均为 20 步机会。

`control` 一格是干净的：它在报警**前**、预算更大，却更差，故预算无法解释报警点的优势。

## 剩余预算

报警点显著早于物理 onset——冒烟那条 trunk 报警 q29、物理 onset q42。
按 blog 口径报警落点 q18–q36、中位 q28。各臂的**机会窗口一律 20 query**，故预算不再随 Δ 变化；
`residual_budget = 52 − fork_query`（原 cap 口径）仍逐 candidate 落盘，仅供次端点使用。

## 端点与判据（冻结）

- 主端点：等预算下救回率对延迟 Δ ∈ {0,+4,+8} 的趋势（logistic，trunk 为聚类单元），
  bootstrap 95% CI 以 trunk 为重抽单元（10k 次）。
- 次端点：逐臂绝对救回率与 Wilson CI；control 臂同口径值；不干预 trunk 的结局。
- 判定：
  - Δ 斜率 CI 完全在零以下 → 支持"越晚越难救"；
  - 斜率 CI 覆盖零且 `triggered` 与 `control` 的差也覆盖零 → 报警时机无价值，
    与旧结果（q32 报警 3/8 而随机 q19 也 3/8）一致；
  - `triggered` 显著高于 `control` 但对 Δ 无斜率 → 报警点特殊、但特殊性不随时间衰减。
- **不中途窥视**；分析在采集全部完成后一次进行。

## 样本量

- 扫 `init_state_id` 0–49 × **4 个 seed**（20260904 + w·1000）= **200 条 trunk**，四 shard 并行。
  `worker_id` 不进 `trunk_noise` / `branch_noise`，只进 `episode_id`——首轮误用它区分 trunk，
  导致 w0–w3 逐位重复，已改。
- 先验：paper-right 下 t08 成功 62%；失败中约 80% 会报警 → 期望 ~61 条可用 trunk。
- n 只够排除大效应。**这是先导，不是确认。**

## 资源

单卡 GPU 5，4 个服务实例（用户限定只用卡 5）。预计约 4 小时。

## 产物

`runs/i<init>_s<seed_slot>/`：`trunk.json`（含逐步 `ratios`/`distances`）、
`triggered/`、`delayed_+N/`、`control/`、`manifest.json`。
代码 `code/delayed_fork_collect.py`，自 `himoe-route-capture/triggered_fork_collect.py` 复制，
改动四处：新增 `--delay-offsets`、`--failures-only`、`--fork-budget-queries`，报警臂后增加 delayed 臂。
报警器、噪声流、control 臂逻辑逐字未动。
