# E7 任务条件路由流形（grid50x8，4 checkpoint 独立）

> 问题 1：routing（d9 表示）是否携带任务身份？——**是，4/4 suite 1-NN 判别 100%（chance 10%），排除同 (task,scene) 邻居后仍 ≥99.88%。**
> 问题 2：失败是否走进别的任务的流形？——**早窗没有任何集（成败皆然）离开自任务流形；到 mid/late，失败相对成功显著向他任务流形漂移（object/spatial mid、long late，p_maxT=0.0005，残差腿存活），其中一部分失败真正越界（d_other<d_own）：object 19/37、long 55/267（late）、goal 27/102、spatial 6/126，而成功集越界率 ≤0.7%（long late 2.6%）。goal 的 mid 效应（AUC 0.606, p=0.021）对 mob1_w8 残差后不存活，不计为独立信号。**

复现单位：4 个 suite = 4 个独立训练 checkpoint，全程未跨 suite 池化。seed=20260903，置换 2000 次。

## 1. 数据与口径

- 语料：`features/grid50x8/<suite>/<task>/{rows.npz, reps.npy}`；每 suite 10 任务 × 400 集（50 scene × 8 repeat），共 16,000 集；失败：goal 102 / long 267 / object 37 / spatial 126（合计 532）。无剔除集（各 meta.json `dropped_episodes=[]`）。
- 表示：rep = sqrt(P) 展平 (4 深层, 10 action token, 32 专家)，d9 切片，f16 → f32。已验证每 (layer,token) 格 ‖r‖²=1±4e-4（f16 舍入）。
- 距离（唯一口径）：reshape (40,32)，逐格 bc=Σ_e r1·r2，hell=sqrt(clip(1−bc,0,1))，40 格取均值（V2）。集间距离 = **匹配 within-episode query 序号**的逐 q 距离均值（rows.npz 的 `control_step` 是任务级累计计数，已按每集首行归零为集内 q；每集内已断言连续递增）。
- 窗口：early q∈{2..6}（全 16,000 集覆盖）、mid q∈{8..12}、late q∈{20..24}（仅 long，其余 suite 覆盖率 ≈0 不设）。窗内配对取双方共有的 q。
- **mismatch 符号口径（明示）**：mismatch = d_own − d_other。d_own = 到同任务成功集 5-NN 距离中位（LOEO，自身排除；同 scene 成功集算候选）；d_other = 9 个他任务各自 5-NN 中位的最小值。**"更像别的任务"（wrong-manifold）⇔ d_other < d_own ⇔ mismatch > 0**。任务说明中"mismatch<0（更像别的任务）"与其自身公式矛盾，本报告按语义执行并在 CSV 保留 d_own/d_other 原值供复核。
- 统计：`phenotype/stats.py` 的 `paired_auc`（组=(task,scene)，跨组不池化）+ `joint_maxt`（episode 级组内置换 2000 次；族 = 该 suite 的窗口数：short 2 格、long 3 格）。残差腿（§4 新颖性）：mismatch 对 mob1_w8 窗均值做组内秩残差后同族重跑。

## 2. 问题 1：任务判别（early q∈{2..6}，1-NN LOEO）

| suite | 准确率 | Wilson 95% CI | 严格变体（排除同 task,scene） | Wilson 95% CI | n | 1-NN 距离中位 |
|---|---|---|---|---|---|---|
| libero_goal | **1.0000** | [0.9990, 1.0000] | 1.0000 | [0.9990, 1.0000] | 4000 | 0.0119 |
| libero_long | **1.0000** | [0.9990, 1.0000] | 0.9988（5 错） | [0.9971, 0.9995] | 4000 | 0.0165 |
| libero_object | **1.0000** | [0.9990, 1.0000] | 0.9998（1 错） | [0.9986, 1.0000] | 4000 | 0.0225 |
| libero_spatial | **1.0000** | [0.9990, 1.0000] | 1.0000 | [0.9990, 1.0000] | 4000 | 0.0297 |

chance = 10%。四张混淆矩阵（`confusion_<suite>.csv`）全部为纯对角（400/400）。严格变体排除全部同 (task,scene) 邻居（同场景 8 draw 近重复），准确率几乎不降：**任务身份不是靠场景布局近重复撑起来的**——即便在 goal（10 个任务共用同一厨房、只有指令/目标不同）里也 100%。前 5 个控制步的深层路由就把任务身份写死了。

## 3. 问题 2：失败与他任务流形

### 3.1 预声明检验族（mismatch 组内成功–失败配对 AUC；y=1=失败，AUC>0.5=失败 mismatch 更高=更偏离自流形）

| suite | 窗口 | det AUC | p_maxT | 残差 AUC (对 mob1_w8) | 残差 p_maxT | 配对数 | 失败集数 | 覆盖集数 |
|---|---|---|---|---|---|---|---|---|
| goal | early | 0.411 | 0.0695 | —（见 §4.4） | — | 448 | 102 | 4000 |
| goal | mid | 0.606 | **0.0210** | 0.569 | 0.1139（不存活） | 371 | 102 | 3195 |
| long | early | 0.378 | 0.0005（反向，见 §4.1） | — | — | 833 | 267 | 4000 |
| long | mid | 0.514 | 0.9590 | 0.493 | 0.9705 | 833 | 267 | 4000 |
| long | late | **0.711** | **0.0005** | 0.666 | **0.0005**（存活） | 827 | 267 | 3666 |
| object | early | 0.435 | 0.4533 | — | — | 237 | 37 | 4000 |
| object | mid | **0.857** | **0.0005** | 0.823 | **0.0005**（存活） | 237 | 37 | 4000 |
| spatial | early | 0.462 | 0.4508 | — | — | 660 | 126 | 4000 |
| spatial | mid | **0.838** | **0.0005** | 0.747 | **0.0005**（存活） | 655 | 126 | 3671 |

复现条款（§4 头条主张 ≥2 独立单位同向）：mid/late 失败漂移在 **object、spatial、long 三个独立 checkpoint** 同向显著且残差存活，成立。goal 的 mid 主检验显著（p=0.021）但残差腿不存活，按新颖性条款**不计为独立信号**（其 AUC 相当部分可由 mob1_w8（路由不稳定拖尾）解释）。long 的 mid 为零结果——长时程任务漂移出现得晚（late 才显著），与集长（中位 25 q）一致。

### 3.2 wrong-manifold（真越界：d_other < d_own）比例，失败 vs 成功

| suite | 窗口 | 失败越界 | 成功越界 | 富集 |
|---|---|---|---|---|
| 全部 4 suite | early | **0 / 532** | **0 / 15,468** | — |
| goal | mid | 27/102 (26.5%) | 10/3093 (0.32%) | ×82 |
| long | mid | 40/267 (15.0%) | 14/3733 (0.38%) | ×40 |
| long | late | 55/267 (20.6%) | 88/3399 (2.59%) | ×8 |
| object | mid | 19/37 (51.4%) | 29/3963 (0.73%) | ×70 |
| spatial | mid | 6/126 (4.8%) | 4/3545 (0.11%) | ×42 |

- **early 零越界是双向的干净结果**：失败集在头 5 个控制步全部牢牢待在自任务流形里（与 Q1 100% 自洽）——"从一开始就走错任务"的假说在这批数据里不成立；越界是失败**过程中发展出来**的。
- long 有 56 个越界失败集（mid∪late）：39 个 mid→late 持续越界、16 个 late 新增、仅 1 个 mid 后回归。
- 效应量参考（mismatch 中位，失败 vs 成功）：object mid +0.0006 vs −0.0161；spatial mid −0.0097 vs −0.0261；goal mid −0.0143 vs −0.0220；long late −0.0156 vs −0.0170；量级与 1-NN 距离中位（0.012–0.030）同尺度。

### 3.3 分解与稳健性（描述性，不进检验族）

逐窗把 mismatch 拆成 d_own、d_other 的单独组内 AUC，并做"跨场景"变体（d_own 候选剔除同 scene 成功集，使成功/失败候选池完全一致，消除 LOEO 候选数不对称）：

| suite | 窗口 | AUC(d_own) | AUC(d_other) | AUC(mismatch, 跨场景变体) |
|---|---|---|---|---|
| goal | mid | 0.620 | 0.526 | 0.685 |
| long | late | 0.742 | 0.626 | 0.724 |
| object | mid | 0.861 | 0.873 | 0.810 |
| spatial | mid | 0.863 | 0.794 | 0.788 |

mid/late 的失败漂移是**双侧的**：既离自流形更远（d_own 升），最近他流形距离也升（d_other 升）但升得更少，mismatch 仍向"他任务相对更近"移动；跨场景变体全部同向（0.69–0.81）——不是同场景近重复候选的伪影。

### 3.4 wrong-manifold 吸引结构（描述）

越界不是随机散射，吸引目标高度集中（完整清单见附录）：

- **long：52/95 条越界行指向 KITCHEN_SCENE8_put_both_moka_pots_on_the_stove**——两个 LIVING_ROOM 篮子任务（SCENE1/SCENE2 alphabet_soup 系）的失败，路由跑到了另一间厨房任务的流形上。SCENE8 恰是本工作区已知的失败富集任务；其自身失败则漂向 LIVING_ROOM_SCENE6。跨房间（视觉完全不同）的吸引说明这不是视觉相似性，更像退化路由汇入一个共同的"宽流形"任务。
- **goal：27 条越界里 16 条指向 put_the_cream_cheese_in_the_bowl**（12 条来自 put_the_bowl_on_the_plate，其中 scene32 一个场景贡献 5/8 个 repeat）；cream_cheese 自己的失败（7 条，scene0 为主）则漂向 put_the_bowl_on_the_stove。同一物体家族（bowl 系）内互吸。
- **object：9/19 指向 pick_up_the_chocolate_pudding**（bbq_sauce 5、orange_juice 2、tomato_sauce 2）——同为"抓取放篮"结构的近邻任务。
- **spatial：6 条里 4 条是 ramekin-bowl → table_center-bowl**——同物体同目标、仅起始位置不同的语义最近邻对。
- 越界高度按 scene 聚簇（如 long SCENE1 任务的 scene3：8 个 repeat 全灭、16 条越界行），与"难场景整组失败"结构一致。

## 4. 边界与诚实条款

1. **early 的反向显著（long AUC 0.378, p=0.0005）判为口径伪影嫌疑，不作为发现**。三条证据：(a) 分解显示效应全在 d_own（0.375），d_other 无差（0.497）；(b) 跨场景变体直接翻向（0.546>0.5）；(c) LOEO 造成候选数不对称——同 (task,scene) 组里成功集给自己当候选时被剔除（同场景近重复候选少 1 个），失败集不被剔除，5-NN 中位被系统性压低，方向恰与观测一致。goal early（0.411, p=0.0695）同构。early 的诚实结论是**零结果**：无越界、无可靠的成败可分性。
2. **object suite 只有 37 个失败**：mid AUC 0.857（237 对）显著且残差存活，但 51.4% 的越界率承 n=37，95% CI 约 [35%, 67%]，富集倍数点估计不稳；按单 suite 证据看待，好在方向与 spatial/long 一致。
3. **mid 窗覆盖有选择**：goal 3195/4000、spatial 3671/4000（被排除的是 <9 q 的快成功；失败集全部保留）。mid 的"成功参照"因此偏向慢成功；快任务尤甚（goal put_the_bowl_on_the_plate 在 q=8 仅 16 个成功集覆盖，其 own-流形参照很薄，该任务贡献了 12/27 的 goal 越界）。组内配对使成败面对同一参照，AUC 方向不受此偏置直接驱动，但越界的逐集归因在快任务上要打折扣。
4. **残差腿在 early 结构性缺失**：mob1_w8 需要 8 步拖尾，q<8 全 NaN，故残差族只含 mid/late（long 2 格、short 1 格）。early 本为零结果，不受影响。
5. **Q1 的 100% 有近重复成分**（同 scene 8 draw 高度相似，1-NN 距离中位 0.012–0.030），但严格变体已控制：排除同 (task,scene) 后仍 ≥99.88%，任务身份主张不依赖近重复。
6. late 窗成功集越界率升到 2.6%（88/3399）：晚期慢成功的路由本身也不再那么"任务典型"，wrong-manifold 在 late 不是失败专属，只是失败富集 ×8。
7. 全部结论 4 suite 独立计算，未跨 checkpoint 池化；AUC 只在 (task,scene) 组内配对。未画图。

## 5. 产物

| 文件 | 内容 |
|---|---|
| `summary.json` | 全部数字（Q1 准确率/CI、逐窗 AUC/p/配对数/越界计数/分解 AUC/残差腿） |
| `confusion_<suite>.csv` | 4 张 10×10 混淆矩阵（行=真任务，列=预测；全对角） |
| `mismatch_scores.csv` | 532 个失败集 × 可用窗口（1331 行）：d_own、d_other、mismatch、最近他任务、wrong_manifold（=mismatch>0）、跨场景变体 |
| `mismatch_ep_<suite>.npz` | 逐集审计值（全部 4000 集 × 窗口：mismatch、跨场景 mismatch、最近他任务、mob1_w8 窗均值） |
| `run_e7.py` / `run_e7.log` | 复算脚本与日志（OPENBLAS=8 仅用于 matmul 段） |

---

# 附录：wrong-manifold 清单

### 聚合：失败集被吸向哪个任务（wrong-manifold 行计数）

| suite | 窗口 | 失败任务 | 最近他任务 | 集数 |
|---|---|---|---|---|
| libero_goal | mid | put_the_bowl_on_the_plate | put_the_cream_cheese_in_the_bowl | 12 |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | put_the_bowl_on_the_stove | 6 |
| libero_goal | mid | put_the_bowl_on_top_of_the_cabinet | put_the_cream_cheese_in_the_bowl | 4 |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | push_the_plate_to_the_front_of_the_stove | 2 |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | turn_on_the_stove | 1 |
| libero_goal | mid | open_the_top_drawer_and_put_the_bowl_inside | push_the_plate_to_the_front_of_the_stove | 1 |
| libero_goal | mid | push_the_plate_to_the_front_of_the_stove | turn_on_the_stove | 1 |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 9 |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 9 |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 8 |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 4 |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 3 |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 3 |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 2 |
| libero_long | late | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 2 |
| libero_long | late | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 2 |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 2 |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 1 |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 1 |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 1 |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 1 |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 1 |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 1 |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 1 |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 1 |
| libero_long | late | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 1 |
| libero_long | late | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 1 |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 1 |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 21 |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 10 |
| libero_long | mid | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 2 |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 1 |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 1 |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 1 |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 1 |
| libero_long | mid | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 1 |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 1 |
| libero_long | mid | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 1 |
| libero_object | mid | pick_up_the_bbq_sauce_and_place_it_in_the_basket | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 5 |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | pick_up_the_ketchup_and_place_it_in_the_basket | 3 |
| libero_object | mid | pick_up_the_tomato_sauce_and_place_it_in_the_basket | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 2 |
| libero_object | mid | pick_up_the_orange_juice_and_place_it_in_the_basket | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 2 |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | pick_up_the_alphabet_soup_and_place_it_in_the_basket | 1 |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 1 |
| libero_object | mid | pick_up_the_orange_juice_and_place_it_in_the_basket | pick_up_the_ketchup_and_place_it_in_the_basket | 1 |
| libero_object | mid | pick_up_the_alphabet_soup_and_place_it_in_the_basket | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 1 |
| libero_object | mid | pick_up_the_salad_dressing_and_place_it_in_the_basket | pick_up_the_alphabet_soup_and_place_it_in_the_basket | 1 |
| libero_object | mid | pick_up_the_butter_and_place_it_in_the_basket | pick_up_the_ketchup_and_place_it_in_the_basket | 1 |
| libero_object | mid | pick_up_the_milk_and_place_it_in_the_basket | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 1 |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 4 |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 1 |
| libero_spatial | mid | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 1 |

### 逐集清单（wrong-manifold，即 d_other < d_own；按 suite、窗口、mismatch 降序）

| suite | 窗口 | task | scene | repeat | d_own | d_other | mismatch | 最近他任务 |
|---|---|---|---|---|---|---|---|---|
| libero_goal | mid | put_the_bowl_on_the_plate | 32 | 7 | 0.0625 | 0.0422 | +0.0203 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 32 | 3 | 0.0668 | 0.0465 | +0.0203 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 32 | 4 | 0.0650 | 0.0462 | +0.0188 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 32 | 1 | 0.0639 | 0.0452 | +0.0187 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 32 | 6 | 0.0644 | 0.0458 | +0.0186 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 7 | 1 | 0.0651 | 0.0466 | +0.0185 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 46 | 1 | 0.0644 | 0.0461 | +0.0183 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 7 | 3 | 0.0652 | 0.0473 | +0.0180 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 36 | 7 | 0.0647 | 0.0473 | +0.0174 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 36 | 6 | 0.0650 | 0.0479 | +0.0171 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 34 | 7 | 0.0659 | 0.0502 | +0.0157 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_top_of_the_cabinet | 27 | 3 | 0.0515 | 0.0456 | +0.0059 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_the_plate | 7 | 7 | 0.0496 | 0.0449 | +0.0047 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_bowl_on_top_of_the_cabinet | 5 | 0 | 0.0490 | 0.0454 | +0.0036 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 14 | 2 | 0.0538 | 0.0506 | +0.0032 | turn_on_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 2 | 0.0552 | 0.0521 | +0.0031 | push_the_plate_to_the_front_of_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 4 | 0.0549 | 0.0520 | +0.0029 | put_the_bowl_on_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 7 | 0.0532 | 0.0510 | +0.0022 | put_the_bowl_on_the_stove |
| libero_goal | mid | put_the_bowl_on_top_of_the_cabinet | 5 | 5 | 0.0426 | 0.0405 | +0.0021 | put_the_cream_cheese_in_the_bowl |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 14 | 4 | 0.0554 | 0.0533 | +0.0021 | push_the_plate_to_the_front_of_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 0 | 0.0549 | 0.0528 | +0.0020 | put_the_bowl_on_the_stove |
| libero_goal | mid | open_the_top_drawer_and_put_the_bowl_inside | 8 | 6 | 0.0462 | 0.0446 | +0.0016 | push_the_plate_to_the_front_of_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 5 | 0.0540 | 0.0526 | +0.0014 | put_the_bowl_on_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 3 | 0.0538 | 0.0524 | +0.0014 | put_the_bowl_on_the_stove |
| libero_goal | mid | put_the_cream_cheese_in_the_bowl | 0 | 6 | 0.0541 | 0.0533 | +0.0009 | put_the_bowl_on_the_stove |
| libero_goal | mid | push_the_plate_to_the_front_of_the_stove | 30 | 3 | 0.0375 | 0.0369 | +0.0006 | turn_on_the_stove |
| libero_goal | mid | put_the_bowl_on_top_of_the_cabinet | 22 | 4 | 0.0446 | 0.0445 | +0.0001 | put_the_cream_cheese_in_the_bowl |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 33 | 7 | 0.0642 | 0.0517 | +0.0125 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 2 | 0.0645 | 0.0521 | +0.0123 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 25 | 6 | 0.0646 | 0.0541 | +0.0105 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 36 | 0 | 0.0634 | 0.0535 | +0.0099 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 19 | 3 | 0.0644 | 0.0554 | +0.0090 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 33 | 4 | 0.0656 | 0.0567 | +0.0089 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 19 | 1 | 0.0628 | 0.0540 | +0.0089 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 47 | 0 | 0.0631 | 0.0544 | +0.0087 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 26 | 4 | 0.0536 | 0.0449 | +0.0087 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 0 | 0.0635 | 0.0548 | +0.0087 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 7 | 0.0615 | 0.0530 | +0.0085 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 1 | 0.0624 | 0.0540 | +0.0084 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 23 | 0 | 0.0659 | 0.0575 | +0.0084 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 4 | 7 | 0.0666 | 0.0583 | +0.0083 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 23 | 2 | 0.0649 | 0.0567 | +0.0082 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 1 | 6 | 0.0626 | 0.0545 | +0.0081 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 26 | 7 | 0.0654 | 0.0576 | +0.0078 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 19 | 2 | 0.0652 | 0.0574 | +0.0078 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 1 | 3 | 0.0642 | 0.0566 | +0.0077 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 0 | 0.0652 | 0.0575 | +0.0076 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 6 | 0.0657 | 0.0581 | +0.0076 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 7 | 0.0672 | 0.0597 | +0.0075 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 4 | 2 | 0.0651 | 0.0577 | +0.0074 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 31 | 3 | 0.0685 | 0.0611 | +0.0074 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 5 | 0.0614 | 0.0543 | +0.0071 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 1 | 4 | 0.0583 | 0.0517 | +0.0066 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 2 | 0.0632 | 0.0569 | +0.0064 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 4 | 0.0648 | 0.0586 | +0.0062 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 22 | 3 | 0.0510 | 0.0450 | +0.0059 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 1 | 0.0589 | 0.0537 | +0.0052 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 34 | 1 | 0.0626 | 0.0575 | +0.0051 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 3 | 0.0589 | 0.0540 | +0.0049 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 41 | 5 | 0.0591 | 0.0545 | +0.0046 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 41 | 0 | 0.0596 | 0.0551 | +0.0045 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 3 | 0.0534 | 0.0493 | +0.0042 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 23 | 6 | 0.0613 | 0.0573 | +0.0040 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 13 | 6 | 0.0597 | 0.0559 | +0.0038 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 0 | 0.0544 | 0.0508 | +0.0036 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 4 | 0.0540 | 0.0504 | +0.0036 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 6 | 0.0606 | 0.0571 | +0.0034 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 1 | 0.0537 | 0.0507 | +0.0030 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 40 | 3 | 0.0472 | 0.0445 | +0.0027 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 35 | 3 | 0.0517 | 0.0490 | +0.0026 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 7 | 0.0529 | 0.0503 | +0.0026 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 5 | 0.0524 | 0.0499 | +0.0024 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 2 | 0.0531 | 0.0507 | +0.0024 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 28 | 0 | 0.0562 | 0.0543 | +0.0019 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | late | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 12 | 3 | 0.0606 | 0.0589 | +0.0017 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | late | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 40 | 7 | 0.0561 | 0.0546 | +0.0014 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 15 | 1 | 0.0566 | 0.0553 | +0.0013 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 14 | 1 | 0.0558 | 0.0545 | +0.0013 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 33 | 5 | 0.0594 | 0.0584 | +0.0009 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | late | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 17 | 4 | 0.0526 | 0.0521 | +0.0005 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 40 | 1 | 0.0549 | 0.0546 | +0.0003 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | late | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 6 | 1 | 0.0560 | 0.0558 | +0.0003 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 2 | 0.0570 | 0.0453 | +0.0116 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 7 | 0.0609 | 0.0504 | +0.0105 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 4 | 0.0582 | 0.0484 | +0.0098 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 6 | 0.0562 | 0.0469 | +0.0093 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 5 | 0.0598 | 0.0506 | +0.0093 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 3 | 0.0561 | 0.0469 | +0.0091 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 1 | 0.0579 | 0.0502 | +0.0077 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 19 | 2 | 0.0595 | 0.0518 | +0.0077 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 0 | 0.0566 | 0.0492 | +0.0073 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 1 | 6 | 0.0582 | 0.0521 | +0.0061 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 36 | 0 | 0.0587 | 0.0528 | +0.0059 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 1 | 4 | 0.0580 | 0.0526 | +0.0054 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 3 | 0 | 0.0606 | 0.0554 | +0.0052 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 23 | 2 | 0.0602 | 0.0556 | +0.0046 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 23 | 0 | 0.0612 | 0.0566 | +0.0046 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 1 | 0.0543 | 0.0499 | +0.0044 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 7 | 0.0545 | 0.0502 | +0.0042 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 47 | 0 | 0.0630 | 0.0588 | +0.0042 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 0 | 0.0545 | 0.0503 | +0.0042 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 2 | 0.0540 | 0.0500 | +0.0040 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 19 | 3 | 0.0588 | 0.0550 | +0.0039 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 4 | 0.0539 | 0.0501 | +0.0037 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 5 | 0.0534 | 0.0497 | +0.0037 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 19 | 1 | 0.0581 | 0.0545 | +0.0036 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 4 | 2 | 0.0626 | 0.0590 | +0.0036 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 1 | 0.0572 | 0.0538 | +0.0034 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 49 | 3 | 0.0534 | 0.0501 | +0.0033 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 1 | 3 | 0.0556 | 0.0523 | +0.0033 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 2 | 0.0600 | 0.0569 | +0.0031 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 38 | 7 | 0.0589 | 0.0563 | +0.0026 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 14 | 1 | 0.0551 | 0.0525 | +0.0025 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 4 | 7 | 0.0643 | 0.0618 | +0.0025 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy |
| libero_long | mid | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 23 | 6 | 0.0633 | 0.0612 | +0.0021 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it |
| libero_long | mid | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 17 | 4 | 0.0583 | 0.0567 | +0.0015 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 41 | 5 | 0.0569 | 0.0555 | +0.0014 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 41 | 0 | 0.0567 | 0.0557 | +0.0010 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove |
| libero_long | mid | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 22 | 3 | 0.0570 | 0.0561 | +0.0009 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_long | mid | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 33 | 4 | 0.0614 | 0.0611 | +0.0002 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_long | mid | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 40 | 7 | 0.0542 | 0.0540 | +0.0002 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket |
| libero_long | mid | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 33 | 3 | 0.0575 | 0.0575 | +0.0000 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | 30 | 6 | 0.0750 | 0.0696 | +0.0054 | pick_up_the_alphabet_soup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 33 | 1 | 0.0683 | 0.0633 | +0.0050 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | 47 | 4 | 0.0716 | 0.0667 | +0.0049 | pick_up_the_ketchup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | 47 | 5 | 0.0739 | 0.0693 | +0.0046 | pick_up_the_ketchup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_tomato_sauce_and_place_it_in_the_basket | 2 | 3 | 0.0675 | 0.0640 | +0.0036 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | 34 | 2 | 0.0677 | 0.0643 | +0.0033 | pick_up_the_ketchup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_cream_cheese_and_place_it_in_the_basket | 8 | 4 | 0.0688 | 0.0656 | +0.0033 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 28 | 5 | 0.0643 | 0.0613 | +0.0030 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 30 | 7 | 0.0674 | 0.0646 | +0.0028 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_orange_juice_and_place_it_in_the_basket | 34 | 7 | 0.0635 | 0.0610 | +0.0025 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 38 | 3 | 0.0638 | 0.0614 | +0.0024 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_orange_juice_and_place_it_in_the_basket | 25 | 4 | 0.0654 | 0.0630 | +0.0023 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 27 | 2 | 0.0659 | 0.0637 | +0.0023 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_orange_juice_and_place_it_in_the_basket | 34 | 0 | 0.0655 | 0.0634 | +0.0021 | pick_up_the_ketchup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_alphabet_soup_and_place_it_in_the_basket | 41 | 5 | 0.0662 | 0.0646 | +0.0015 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_salad_dressing_and_place_it_in_the_basket | 34 | 4 | 0.0593 | 0.0579 | +0.0015 | pick_up_the_alphabet_soup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_butter_and_place_it_in_the_basket | 11 | 7 | 0.0681 | 0.0672 | +0.0009 | pick_up_the_ketchup_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_milk_and_place_it_in_the_basket | 4 | 2 | 0.0638 | 0.0630 | +0.0007 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_object | mid | pick_up_the_tomato_sauce_and_place_it_in_the_basket | 28 | 2 | 0.0640 | 0.0635 | +0.0006 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 40 | 0 | 0.0919 | 0.0860 | +0.0060 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 46 | 1 | 0.0907 | 0.0880 | +0.0027 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 40 | 6 | 0.0792 | 0.0770 | +0.0022 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 40 | 7 | 0.0893 | 0.0874 | +0.0019 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate |
| libero_spatial | mid | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 38 | 4 | 0.0822 | 0.0810 | +0.0012 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate |
| libero_spatial | mid | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 20 | 4 | 0.0833 | 0.0821 | +0.0012 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate |
