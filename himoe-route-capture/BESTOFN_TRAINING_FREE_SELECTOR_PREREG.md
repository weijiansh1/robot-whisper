# Best-of-N 免训练 selector 臂 — 预注册（2026-08-25，采集启动前冻结）

## 冻结时点声明

冻结此文档时 `runs/bestofn-critic-20260825/run_manifest.json` 的 `progress` 为
`candidate_artifacts: 0`、`initial_continuation_artifacts: 0`、`confirmed_rows: 0`，
`state: waiting_for_gpu`（两张 H20 各仅 3.88 GiB 空闲，安全装载阈值 24 GiB）。
**尚无任何候选或 outcome 数据存在**，因此本次增臂是采集前修改，不是事后加 selector。

变更前的 config sha256 为 `1e5e0ff3bbf5a59557922d6f4184fb1a897c66ddd772f420ddba80d3e702f6df`，
留作审计链；实施后 config 与 `run_manifest.json` 的 `implementation_sha256` 一并重新冻结。

## 动机

`bestofn_experiment_config.json` 的 `critic.baselines` 四个臂
（`state_action` / `state_route` / `state_action_route` / `state_action_router_hidden`）
**全部需要训练**。设计中没有任何免训练 selector，因此 Gate 2 只能回答
"routing 是否给学出来的 critic 加分"，无法回答"这一切是否打得过文献里已有的
动作空间多数投票"。

文献里的多数投票是**动作空间**共识；已被 2026-08-25 审计否掉的
`route medoid` 是 HiMoE 专属的 route 空间共识，两者不是同一个方法。审计
（`analysis/moe-current-data-audit/REPORT.md`）只在 legacy K32（20 snapshot、
单任务、endpoint proxy）上跑过 action medoid，在五任务 K32 上没有该臂。

免训练 selector 是候选动作的确定性函数，**不需要任何额外采集、额外 rollout 或额外
显存**，`arrays["action"]` 已存展平的 `10x7` chunk。边际成本为零。

## 加入的臂（冻结定义）

分数一律取"到同 snapshot 其他候选的平均距离的相反数"，即 medoid 中心度：

$$s_i = -\frac{1}{K-1}\sum_{j\ne i} d(c_i, c_j)$$

平局按候选 id 升序取最小（`stable_argmin` 语义），不引入随机数。

| 臂 | 距离 | 地位 |
|---|---|---|
| `action_medoid` | `behavior_geometry.action_distance_matrix`：checkpoint 归一化逐步 L2，gripper 权重 `0.25` | **主臂**（文献多数投票） |
| `action_rms_medoid` | `behavior_geometry.action_rms_distance_matrix`：legacy RAD bridge 展平 RMS | 度量敏感性 |
| `route_prob_medoid` | 完整 HB router 分布的 RMS Hellinger（8 HB × 10 denoise × 10 action token，仅 action token） | HiMoE 对照 |
| `exact_random` | 无 | 下界，取 `snapshot_metrics` 已有的 `random_q`，解析期望而非蒙特卡洛 |

`action_std` 取采集服务写入的 `normalization_action_std`（`capture_bestofn.py` 已记录），
形状 `(7,)`、有限且为正，否则直接报错终止，不做回退。
`route_prob_medoid` 读候选 artifact 里的 `full_hb_router_probabilities` 原始分布，
不用 critic dataset 里的摘要特征。

## 评估位置与指标

- 与 critic 同受 `authorized_only_after_oracle_gate` 约束：**Gate 1 通过之前不评估**，
  避免用 outcome 窥视。
- 同一 parent-episode-grouped test split，同一 `bestofn_critic.snapshot_metrics`
  与 `aggregate_metrics`，同一 `hierarchical_delta_ci`（task→snapshot，10,000 draws）。
- **只报排序类指标**：`pairwise_accuracy`、`decisive_pairwise_accuracy`、
  `selected_q`、`top1_regret`、`oracle_recovery_ratio`。
- **禁止报 `brier` 与 `binomial_proportion_nll`**：medoid 中心度不是标定概率，
  这两项对免训练臂无意义。

## 假设与判据（冻结）

- H_A：`action_medoid` 的 `selected_q` 高于 `exact_random`。
- H_B：`state_action` critic 的 `selected_q` 高于 `action_medoid`
  —— 即"训练"相对"免费投票"是否买到了东西。
- H_C：`state_action_route`（joint）的 `selected_q` 高于 `action_medoid`。

**Gate 2 保持原样不动**（主判据仍是 joint 减 `state_action`）。另设预先声明的
**Gate 2b（次级门，强制报告）**：joint 减 `action_medoid` 的 `selected_q` 配对
分层 bootstrap 95% CI 下界大于 0。

Gate 2 通过而 Gate 2b 不通过时，结论必须写成"MoE routing 对学出来的 critic 有增量，
但整套 pipeline 未证明优于免训练动作多数投票"，不得只报 Gate 2。

## 退化披露（强制）

免训练 selector 在本仓库的历史数据上出现过退化（五任务 K32 上 route probability
medoid 有 97.5% 的 state 选中同一个 seed 1013）。因此无论结果方向，必须同时报告：

- 选中候选 id 的分布与最大占比（`max_candidate_share`）、不同候选 id 的个数；
- 每个 snapshot 的候选内动作离散度（`action_distance_matrix` 的上三角中位数）；
- 该离散度与 `top1_regret` 的 task-macro Spearman。

若 `max_candidate_share` 超过 0.5，主结论须记为"selector 退化为近似固定候选选择"，
其 `selected_q` 对比不作为方法有效性证据。

## 不做什么

- 不改采集内容、不改 `proposal.capture`、不改 Gate 1 的任何定义与判定。
- 不用免训练 selector 的结果回改 Gate 1，也不用它训练或选择任何 critic 超参。
- 不改 `critic.baselines` 这四个已冻结的训练臂；新臂放在独立键下。
- 不在本轮引入 verifier / reward model / VLM 打分类方法；那是另一份预注册。

## 实施改动清单

1. `bestofn_experiment_config.json`：新增 `critic.training_free_selectors` 块
   （臂名单、`gripper_weight: 0.25`、允许上报的指标白名单、`gate_2b` 定义）。
2. `bestofn_protocol.py`：`validate_config` 增加对该块的校验；
   原有四臂 `expected_baselines` 检查**原样保留**。
3. 新增 `bestofn_training_free.py` + `test_bestofn_training_free.py`。
4. `train_bestofn_critics.py`：`run()` 中计算免训练臂并接入 `render_report`。
5. `BESTOFN_COUNTERFACTUAL_Q_EXPERIMENT.md`：补写该臂与 Gate 2b。
6. 重新冻结：`config_sha256` 与 `implementation_sha256` 写回
   `runs/bestofn-critic-20260825/run_manifest.json`，并保留变更前 sha 作为审计链。
7. `python3 run_bestofn_experiment.py preflight` 必须重新通过。

## 产物

`analysis/bestofn-critic-20260825/training_free_selectors.json` 与同名 `.md`，
以及 `train_bestofn_critics.py` 主报告中的对照表。
