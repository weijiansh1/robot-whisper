# 语料审计与事件表构建报告（AUDIT）

- 日期：2026-09-04；协议：`PROTOCOL.md`（冻结阈值未做任何改动；事件代理等级按
  §8 Amendment 1 升级为 v2，见本文 §8）
- 产物：`audit/data_map.json`（机器可读清点）、`audit/proxy_validation.json`（验证明细，
  v1+v2）、`events/<corpus>/<suite>/<task>/events.csv`（45 任务，**现行 v2**）、
  `features/main16x32/`（5 任务）
- 快照说明：仍在运行的采集进程写的是 `~/cache_new_spill_20260904/right-50x16c-20260904`
  （新 run，不在本次审计范围）；本次审计的 `right-50x8-20260903` 与 `right-16x32` 均为
  `meta.json status=complete` 的稳定快照。

## 1. 语料清点（任务 1）

两套语料均 paper-right；路由只读 server `routes.zarr`；全部 45 任务 `hb_router_probs`
(N,8,10,11,32) float16 齐全，`sim_layout.json` 齐全。

**行↔集对账：45/45 任务全部通过**——每任务 zarr 总行数 = Σclient `inference_calls`，
且逐集 zarr 行数 = 该集 `inference_calls`，**零失配、零剔除**。

### 1.1 失败预算（每套件失败总数 = 检测类实验的功效上限）

| 语料 | 套件 | 任务 | 集数 | 成功 | 失败 | 备注 |
|---|---|---|---|---|---|---|
| grid50x8 | libero_goal | 10 | 4,000 | 3,898 | **102** | 52 集中在 open_top_drawer+bowl |
| grid50x8 | libero_long | 10 | 4,000 | 3,733 | **267** | 138 集中在 SCENE8 |
| grid50x8 | libero_object | 10 | 4,000 | 3,963 | **37** | 功效极低 |
| grid50x8 | libero_spatial | 10 | 4,000 | 3,874 | **126** | 最多单任务 35 |
| grid50x8 | 合计 | 40 | **16,000** | 15,468 | **532** | 与协议预期 532 完全一致 |
| main16x32 | libero_goal | 2 | 1,024 | 982 | **42** | |
| main16x32 | libero_long (SCENE8) | 1 | 512 | 296 | **216** | 失败富集主语料，与协议一致 |
| main16x32 | libero_spatial | 2 | 1,024 | 975 | **49** | |
| main16x32 | 合计 | 5 | **2,560** | 2,253 | **307** | |

### 1.2 每任务明细（集数/失败/queries 中位/zarr 行/对账/事件数，事件括号内为失败集中数）

（**历史表**：loop/static/trap 列为 v1 构建值——非 SCENE8 任务 degraded、SCENE8 full。
v1 degraded 层已被 §8 的 v2 objaware 层**覆写**，此表事件列仅保留作 v1↔v2 对比；
集数/失败/对账/queries 列不受影响，仍为现行值。v2 每任务事件计数见
`audit/proxy_validation.json: task_event_summaries_v2`，套件级汇总见 §8.2。）

### grid50x8
| suite | task | eps | fail | q中位 | zarr行 | 对账 | loop | static | trap |
|---|---|---|---|---|---|---|---|---|---|
| goal | open_the_middle_drawer_of_the_cabinet | 400 | 0 | 13 | 5064 | OK | 47(0f) | 0(0f) | 47(0f) |
| goal | open_the_top_drawer_and_put_the_bowl_inside | 400 | 52 | 19 | 8041 | OK | 51(45f) | 5(5f) | 54(48f) |
| goal | push_the_plate_to_the_front_of_the_stove | 400 | 2 | 13 | 5582 | OK | 28(0f) | 1(1f) | 29(1f) |
| goal | put_the_bowl_on_the_plate | 400 | 12 | 8 | 3476 | OK | 13(12f) | 0(0f) | 13(12f) |
| goal | put_the_bowl_on_the_stove | 400 | 6 | 9 | 3765 | OK | 0(0f) | 0(0f) | 0(0f) |
| goal | put_the_bowl_on_top_of_the_cabinet | 400 | 15 | 9 | 3900 | OK | 7(7f) | 13(13f) | 14(14f) |
| goal | put_the_cream_cheese_in_the_bowl | 400 | 10 | 9 | 3981 | OK | 14(10f) | 0(0f) | 14(10f) |
| goal | put_the_wine_bottle_on_the_rack | 400 | 3 | 13 | 5431 | OK | 2(0f) | 0(0f) | 2(0f) |
| goal | put_the_wine_bottle_on_top_of_the_cabinet | 400 | 2 | 9 | 3820 | OK | 0(0f) | 1(1f) | 1(1f) |
| goal | turn_on_the_stove | 400 | 0 | 8 | 3129 | OK | 0(0f) | 0(0f) | 0(0f) |
| long | KITCHEN_SCENE3_turn_on_stove+moka_pot | 400 | 6 | 25 | 10165 | OK | 254(5f) | 4(4f) | 254(5f) |
| long | KITCHEN_SCENE4_bowl_in_drawer+close | 400 | 4 | 22 | 9091 | OK | 27(0f) | 5(4f) | 32(4f) |
| long | KITCHEN_SCENE6_mug_in_microwave+close | 400 | 6 | 25 | 10472 | OK | 7(0f) | 3(2f) | 9(2f) |
| long | KITCHEN_SCENE8_both_moka_pots（full） | 400 | 138 | 40 | 17460 | OK | 66(62f) | 70(69f) | 131(126f) |
| long | LIVING_ROOM_SCENE1_soup+cream_cheese | 400 | 26 | 25 | 10783 | OK | 386(21f) | 14(13f) | 390(25f) |
| long | LIVING_ROOM_SCENE2_soup+tomato_sauce | 400 | 13 | 26 | 10993 | OK | 400(13f) | 9(7f) | 400(13f) |
| long | LIVING_ROOM_SCENE2_cream_cheese+butter | 400 | 6 | 25 | 10162 | OK | 392(5f) | 3(3f) | 393(6f) |
| long | LIVING_ROOM_SCENE5_two_mugs_two_plates | 400 | 17 | 23 | 9699 | OK | 73(11f) | 5(3f) | 75(13f) |
| long | LIVING_ROOM_SCENE6_mug_on_plate+pudding | 400 | 43 | 23 | 10660 | OK | 48(17f) | 23(15f) | 70(31f) |
| long | STUDY_SCENE1_book_in_caddy | 400 | 8 | 18 | 7689 | OK | 0(0f) | 7(7f) | 7(7f) |
| object | pick_alphabet_soup | 400 | 3 | 14 | 5763 | OK | 189(2f) | 1(0f) | 190(2f) |
| object | pick_bbq_sauce | 400 | 11 | 13 | 5641 | OK | 64(6f) | 2(2f) | 66(8f) |
| object | pick_butter | 400 | 2 | 15 | 6143 | OK | 2(2f) | 0(0f) | 2(2f) |
| object | pick_chocolate_pudding | 400 | 0 | 15 | 6052 | OK | 0(0f) | 0(0f) | 0(0f) |
| object | pick_cream_cheese | 400 | 5 | 13 | 5345 | OK | 10(5f) | 0(0f) | 10(5f) |
| object | pick_ketchup | 400 | 0 | 14 | 5739 | OK | 0(0f) | 0(0f) | 0(0f) |
| object | pick_milk | 400 | 8 | 13 | 5566 | OK | 70(3f) | 2(2f) | 71(4f) |
| object | pick_orange_juice | 400 | 3 | 13 | 5169 | OK | 55(3f) | 0(0f) | 55(3f) |
| object | pick_salad_dressing | 400 | 1 | 12 | 5050 | OK | 20(1f) | 0(0f) | 20(1f) |
| object | pick_tomato_sauce | 400 | 4 | 13 | 5609 | OK | 192(4f) | 1(1f) | 192(4f) |
| spatial | bowl_between_plate_ramekin | 400 | 0 | 8 | 3310 | OK | 13(0f) | 0(0f) | 13(0f) |
| spatial | bowl_from_table_center | 400 | 8 | 10 | 4140 | OK | 0(0f) | 1(1f) | 1(1f) |
| spatial | bowl_in_top_drawer | 400 | 12 | 13 | 5402 | OK | 0(0f) | 0(0f) | 0(0f) |
| spatial | bowl_next_to_cookie_box | 400 | 3 | 11 | 4401 | OK | 1(0f) | 0(0f) | 1(0f) |
| spatial | bowl_next_to_plate | 400 | 14 | 10 | 4170 | OK | 16(14f) | 0(0f) | 16(14f) |
| spatial | bowl_next_to_ramekin | 400 | 9 | 11 | 4648 | OK | 3(3f) | 0(0f) | 3(3f) |
| spatial | bowl_on_cookie_box | 400 | 9 | 9 | 3700 | OK | 6(6f) | 3(3f) | 6(6f) |
| spatial | bowl_on_ramekin | 400 | 25 | 10 | 4206 | OK | 20(6f) | 0(0f) | 20(6f) |
| spatial | bowl_on_stove | 400 | 35 | 13 | 5316 | OK | 1(1f) | 0(0f) | 1(1f) |
| spatial | bowl_on_wooden_cabinet | 400 | 11 | 12 | 4989 | OK | 5(5f) | 0(0f) | 5(5f) |

### main16x32
| suite | task | eps | fail | q中位 | zarr行 | 对账 | loop | static | trap |
|---|---|---|---|---|---|---|---|---|---|
| goal | open_the_middle_drawer_of_the_cabinet | 512 | 0 | 13 | 6452 | OK | 78(0f) | 0(0f) | 78(0f) |
| goal | open_the_top_drawer_and_put_the_bowl_inside | 512 | 42 | 19 | 10018 | OK | 37(31f) | 12(12f) | 45(39f) |
| long | KITCHEN_SCENE8_both_moka_pots（full） | 512 | 216 | 40 | 22883 | OK | 57(55f) | 146(143f) | 193(188f) |
| spatial | bowl_on_ramekin | 512 | 12 | 10 | 5139 | OK | 24(7f) | 0(0f) | 24(7f) |
| spatial | bowl_on_stove | 512 | 37 | 13 | 6816 | OK | 0(0f) | 0(0f) | 0(0f) |

### 1.3 CALVIN inventory（只记规模，不分析）

| cache | 路径 | run | 大小 | zarr 行 | hb_router_probs |
|---|---|---|---|---|---|
| cache | calvin_d/task_D_D | routes-v1 | 159 MB | 3,621 | 有 |
| cache_new | calvin_d/task_D_D | routes-v2-20260903 | 400 MB | 9,065 | 有 |
| cache | calvin_d_d/*（34 任务目录） | — | 空目录 | — | — |

两个 calvin run 均无 `client/summaries.json`（无集级结局），不可与 LIBERO 同表。

## 2. main16x32 特征提取（任务 2）

`phenotype/extract_task.py`（V2 聚合、d9 reps f16），5/5 任务成功，**零剔除**，
state token 跨 flow 不变性偏差全部为 0（f32 存储路径）：

| 任务 | rows | eps | succ | dropped |
|---|---|---|---|---|
| goal/open_the_middle_drawer_of_the_cabinet | 6,452 | 512 | 512 | 0 |
| goal/open_the_top_drawer_and_put_the_bowl_inside | 10,018 | 512 | 470 | 0 |
| long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 22,883 | 512 | 296 | 0 |
| spatial/bowl_on_ramekin | 5,139 | 512 | 500 | 0 |
| spatial/bowl_on_stove | 6,816 | 512 | 475 | 0 |

产物 `features/main16x32/<suite>/<task>/{rows.npz, reps.npy, meta.json}`，共 130 MB。

## 3. 事件表（任务 3）——v1 构建记录（degraded 层已被 §8 v2 覆写）

`events/<corpus>/<suite>/<task>/events.csv`，45 个文件，列：
`episode_id, scene(init_state_id), repeat, success, n_queries, loop_onset_q, static_onset_q,
trap_onset_q(-1=无), proxy_grade`。构建脚本 `events/build_events.py`，规则逐字复刻
`himoe-vla_trap/code/analyze_trainfree_signal_matrix.py::physical_onsets`（阈值未动）。

- **full**（仅 KITCHEN_SCENE8，两套语料各 1 任务）：eef=state[:,:3]、
  objects=sim_state[10:13]/[17:20]（sim_layout 断言 moka_pot_1@10:17、moka_pot_2@17:24、
  state_dim=47）、grip=state[:,6:8].mean、goal 参考=rolling-star dryrun_v10 冻结 JSON。
- **degraded**（其余 43 任务）：eef=state[:,:3]、grip=state[:,6]（第 7 维）；去掉 obj 与
  goal 项，其余阈值不变。

### 3.1 事件汇总（按 proxy_grade 分层；括号 = 失败集/成功集拆分）

| 层 | eps | 失败 | loop | static | trap |
|---|---|---|---|---|---|
| grid50x8 · SCENE8 full | 400 | 138 | 66 (62f/4s) | 70 (69f/1s) | 131 (126f/5s) |
| main16x32 · SCENE8 full | 512 | 216 | 57 (55f/2s) | 146 (143f/3s) | 193 (188f/5s) |
| grid50x8 · degraded ×39 | 15,600 | 394 | 2,416 (207f/**2,209s**) | 103 (87f/16s) | 2,476 (257f/2,219s) |
| main16x32 · degraded ×4 | 2,048 | 91 | 139 (38f/**101s**) | 12 (12f/0s) | 147 (46f/101s) |

degraded 的 loop 事件大多落在成功集——这不是发现，是假阳性（见 §4）。

## 4. 代理验证（SCENE8 上 degraded vs full，决定 degraded 可信度）

在 SCENE8 两套语料上同时计算两种代理，以 full 为真值：

| 事件 | 语料 | full n | deg n | sensitivity | specificity | Δonset 中位(deg−full) | |Δ|≤2 占比 |
|---|---|---|---|---|---|---|---|
| loop | main16x32 | 57 | 476 | 0.982 | **0.077** | **−17** | 0.018 |
| loop | grid50x8 | 66 | 372 | 0.985 | **0.081** | −11 | 0.031 |
| static | main16x32 | 146 | 154 | **0.986** | **0.973** | **−1** | 0.972 |
| static | grid50x8 | 70 | 70 | **0.971** | **0.994** | −1 | 0.971 |
| trap | main16x32 | 193 | 500 | 0.995 | 0.034 | −13 | 0.156 |
| trap | grid50x8 | 131 | 384 | 1.000 | 0.060 | −13 | 0.115 |

**结论（如实）：**
- **degraded static 可信**：sens/spec ≥ 0.97，onset 中位只早 1 个 query（obj 条件去除使滑窗
  略早合格），97% 的事件 |Δ|≤2。可用于 E3。
- **degraded loop 不可信**：specificity 0.077/0.081——去掉 obj+goal 项后，正常"取放-返回"
  行为（eef 回到旧位置、路径≥0.120、指宽相近）就会触发；SCENE8 有 93% 的集被标 loop，
  且 onset 中位偏早 17 个 query。两物体任务最严重（LIVING_ROOM_SCENE1/2、KITCHEN_SCENE3
  成功集触发率 96–100%：放下第一个物体回来取第二个 = 天然"回访"）。
  **43 个 degraded 任务的 loop_onset_q / trap_onset_q 不得当作 loop 真值或事件对齐锚点使用；
  degraded 层只有 static_onset_q 可用。**

（本节验证直接促成 PROTOCOL §8 Amendment 1：degraded 层废弃，全量升级为 §8 的
objaware 层；本节数字保留作依据。）

## 5. SCENE8 回归测试（实现正确性）

我方 full 代理 vs 既有已验证产物（`trainfree_signal_matrix/tables`、`analysis_ssm/cache`）：

| 对照项 | 结果 |
|---|---|
| 事件计数 vs `onset_inventory.csv`(B) loop/static/trap 及成败拆分 | **3/3 完全一致**（57=55f+2s / 146=143f+3s / 193=188f+5s） |
| 逐集 trap onset vs `aligned_event_values.csv`(B, relative=0) | **161/161 完全一致**（match_rate 1.0） |
| n_queries、success vs `B_meta.csv` | **512/512、512/512 完全一致** |

实现与既有验证管线逐集等价；left-扫描按 right 向量化不改变返回值（onset=right，与命中
哪个 left 无关）。

## 6. 附注：goal 项冗余性与升级路径（已被 Amendment 1 采纳 → §8）

- 数学事实：goal_distance 对物体位置 1-Lipschitz ⇒ obj 项（max_obj_disp≤0.030）成立时
  |goal[l]−goal[r]|≤0.030<0.035，goal 项自动成立。**实证：在 SCENE8 512 集上，
  "obj-aware（去 goal 项、保 obj 项）"与 full 逐集完全一致（0 处不一致）。**
- 所有 45 任务的 run 都带 `client/sim_layout.json`（具名 free-joint qpos 槽位）⇒ 物体位置
  可对全语料解析。obj-aware 代理可把 loop 事件升级到 full 等价，且**不需要 goal 参考**。
  该路径已由 PROTOCOL §8 Amendment 1 采纳（物体集合冻结为"该场景全部非机器人
  free joint"），v2 重建见 §8。

## 7. 红旗清单

1. **degraded loop 不可信（v1 最重要的负结果；已处置）**：spec≈0.08、Δonset −17/−11。
   已按 Amendment 1 用 obj-aware 代理重建全部 43 任务（§8）；**残余红旗**：铰接机构任务
   （KITCHEN_SCENE3 旋钮、open_middle_drawer 抽屉）的 loop 通道在 v2 下仍无效（§8.4），
   E2 不得使用这两类任务的 loop 事件。
2. **功效预算严重不均**：grid object 套件全套只有 37 个失败集、goal 102、spatial 126；
   失败富集几乎只在 long（267，其中 SCENE8 占 138）与 main16x32 SCENE8（216）。
   组内成功–失败配对 AUC 在 object/spatial 大多数任务上功效可忽略。
3. **零事件/零失败任务**：grid 有 5 个任务 0 失败（middle_drawer、turn_on_stove、
   chocolate_pudding、ketchup、bowl_between）；main16x32 goal/middle_drawer 512/512 全成功。
   （v1 曾记 main spatial/bowl_on_stove 37 失败零事件；v2 objaware 下该任务恢复出
   loop 20(14f)——v1 degraded 的严格 gripper 定义掩盖了它们。）v2 下 grid long 的
   KITCHEN_SCENE4/STUDY_SCENE1 仍 0 loop（只有 static 4f/6f），KITCHEN_SCENE6 loop
   无失败集事件。
4. **短任务 vs 固定时点**：goal/spatial 多任务 queries 中位仅 8–13（最短 8），协议固定时点
   t∈{8,12,16} 中 t=16 在这些任务上只有长尾集存活（往往偏失败）——存活审计必须附
   （协议已要求），否则固定时点比较有选择偏倚。
5. **full 代理的 grip 项近退化**（复刻既有规则，不是本次引入）：state[:,6:8].mean 两指
   对称相消，幅度 ~3e-3，恒满足 0.012 阈；full 的辨别力实际来自 eef/obj/goal/path 项。
   degraded 用第 7 维（0–0.04）反而更严格。不影响回归一致性，但解释了两代理的部分差异。
6. **文档口径差异**：任务简报写 sim_state 为 79 维；实际 45–123 维随场景变化（79 仅
   libero_goal 场景）。冻结解析只用于 SCENE8（47 维，已断言布局），不受影响。
7. calvin：cache/calvin_d_d 34 个任务目录全空；两个 task_D_D run 无 summaries.json
   （无结局标签）。按协议只 inventory。
8. 磁盘：本次新增 ~132 MB（features 130 + events/audit ~2）；work 卷剩 ~11 GB，
   另有采集进程在写 spill 目录，需留意后续空间。

## 8. v2 事件层（PROTOCOL §8 Amendment 1：objaware 全量重建）

Amendment 1 采纳 §6 升级路径：非 SCENE8 的 43 任务改用 **objaware** 代理并覆写
events.csv（proxy_grade="objaware"）；SCENE8 两个保持 full **未动**（文件未重写）。
规则 = full 去 goal 项，阈值逐项一致（loop obj≤0.030、static obj≤0.005、grip 用
state[:,6:8].mean、其余同 §3）；物体集合冻结为该场景 `sim_layout.json` 中全部非机器人
free joint（7 维槽）的位置，各任务 2–8 个（分布 2:6 任务 / 4:13 / 5:14 / 7:10 / 8:2，
逐任务清单在 `proxy_validation.json: object_slots_v2`）。

### 8.1 准入回归（先于覆写执行；不通过即中止）

| 对照 | 结果 |
|---|---|
| main16x32 SCENE8：objaware(layout 槽位) vs full(冻结槽位+goal) 逐集 (loop,static,trap) 三元组 | **512/512 全等（0 失配）** |
| main16x32 SCENE8：objaware trap onset vs `aligned_event_values.csv`(B) | **161/161** |
| grid50x8 SCENE8：同上三元组 | **400/400 全等** |
| layout 槽位自检 | 恰为 moka_pot_1/2_joint0（铰接 stove button 自动排除，与冻结 full 一致） |

### 8.2 v2 事件统计（现行口径）

分层聚合（括号=失败/成功拆分）：

| 层 | eps | 失败 | loop | static | trap |
|---|---|---|---|---|---|
| grid50x8 · SCENE8 full | 400 | 138 | 66 (62f/4s) | 70 (69f/1s) | 131 (126f/5s) |
| main16x32 · SCENE8 full | 512 | 216 | 57 (55f/2s) | 146 (143f/3s) | 193 (188f/5s) |
| grid50x8 · objaware ×39 | 15,600 | 394 | 769 (231f/538s) | 110 (93f/17s) | 828 (275f/553s) |
| main16x32 · objaware ×4 | 2,048 | 91 | 147 (47f/100s) | 12 (12f/0s) | 157 (57f/100s) |

每套件（SCENE8 计入 long 行）：

| corpus/suite | eps | 失败 | loop | static | trap |
|---|---|---|---|---|---|
| grid50x8/goal | 4,000 | 102 | 137 (72f/65s) | 21 (21f/0s) | 143 (78f/65s) |
| grid50x8/long | 4,000 | 267 | 412 (124f/288s) | 151 (133f/18s) | 527 (223f/304s) |
| grid50x8/object | 4,000 | 37 | 183 (29f/154s) | 4 (4f/0s) | 185 (31f/154s) |
| grid50x8/spatial | 4,000 | 126 | 103 (68f/35s) | 4 (4f/0s) | 104 (69f/35s) |
| main16x32/goal | 1,024 | 42 | 117 (26f/91s) | 12 (12f/0s) | 127 (36f/91s) |
| main16x32/long(SCENE8) | 512 | 216 | 57 (55f/2s) | 146 (143f/3s) | 193 (188f/5s) |
| main16x32/spatial | 1,024 | 49 | 30 (21f/9s) | 0 | 30 (21f/9s) |

### 8.3 grid long 各任务 loop 事件（E2 复现腿功效预算）

| 任务 | loop 总 | 失败集 | 成功集 | static(失败) | E2 可用性 |
|---|---|---|---|---|---|
| KITCHEN_SCENE8（full） | 66 | **62** | 4 | 70(69f) | 主复现腿 |
| LIVING_ROOM_SCENE1 | 22 | **19** | 3 | 15(14f) | 可用 |
| LIVING_ROOM_SCENE2 soup+tomato | 24 | **12** | 12 | 9(7f) | 可用（成功集触发 3.1%） |
| LIVING_ROOM_SCENE6 | 14 | **11** | 3 | 27(17f) | 可用 |
| LIVING_ROOM_SCENE5 | 13 | **10** | 3 | 4(4f) | 可用 |
| KITCHEN_SCENE3 | 256 | 6 | **250** | 6(6f) | **loop 通道无效**（§8.4） |
| LIVING_ROOM_SCENE2 cheese+butter | 10 | 4 | 6 | 3(3f) | 弱 |
| KITCHEN_SCENE6 | 7 | 0 | 7 | 4(3f) | loop 无失败事件 |
| KITCHEN_SCENE4 / STUDY_SCENE1 | 0 | 0 | 0 | 5(4f)/8(6f) | 仅 static |

### 8.4 成功集 loop 触发率：是否回落到 full 量级（诚实评估）

基准（full/SCENE8）：main 2/296=**0.68%**、grid 4/262=**1.53%**。

| 口径 | v1 degraded | v2 objaware |
|---|---|---|
| grid50x8 成功集 loop 率（39 任务） | 2,209/15,206 = 14.5% | 538/15,206 = **3.54%** |
| main16x32 成功集 loop 率（4 任务） | 101/1,957 = 5.2% | 100/1,957 = **5.11%** |
| grid 排除铰接机构任务（SCENE3、middle_drawer） | — | 235/14,412 = **1.63%** |
| main 排除 middle_drawer | — | 16/1,445 = **1.11%** |

- **两物体回访假阳性已修复**：LIVING_ROOM_SCENE2 soup+tomato 100%→3.1%、
  cheese+butter 98.2%→1.5%、SCENE1 97.6%→0.8%、tomato_sauce 47.5%→8.1%。
- **残余集中在铰接机构任务**（loop 率几乎不降）：KITCHEN_SCENE3 63.5%（250/394，转旋钮
  = 合法的"原地回访+路径累积"，free 物体不动）、open_middle_drawer 13.2%/16.4%（拉抽屉
  同理）。这**不是 objaware 实现缺陷**：goal 项冗余 ⇒ 即使有 goal 参考的真·full 规则在这些
  任务上也会同样触发——是冻结 loop 规则对"被操纵对象非 free joint"任务的语义盲区。
  这两类任务的 loop/trap 通道判为无效（红旗 §7.1），static 通道不受影响。
- 次级观察：个别高重试 pick 任务成功集触发偏高（bbq_sauce 14.7%、milk 5.9%），无真值
  无法区分"良性重试真 loop"与假阳性，用作事件对齐时建议组内配对消化。
- 结论：**排除铰接机构任务后，objaware 成功集触发率（1.63%/1.11%）回落到 full 量级
  （1.53%/0.68%），升级可用**；SCENE3 与 middle_drawer 的 loop 通道除外。

## 9. 复现

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd analysis_moe_phenotype
python3 audit/inventory.py          # → audit/data_map.json
python3 events/build_events.py      # v2：准入回归 → 覆写 events/**/events.csv
                                    #     + audit/proxy_validation.json（保留 v1 记录）
# 特征（每任务）：
python3 phenotype/extract_task.py --task-dir <run> --out features/main16x32/<suite>/<task>
```
