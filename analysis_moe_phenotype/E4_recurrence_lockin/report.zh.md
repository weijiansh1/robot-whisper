# E4 复发脉冲列 + lock-in 亚型 + 恢复性 loop 负对照

- 日期 2026-09-04；协议 PROTOCOL §4/§5-E4/§8（Amendment 1 objaware 事件层）；AUDIT §7/§8。
- seed 20260903，置换 2000 次，episode 级组内置换 + joint maxT（族内校正）。
- 组：grid50x8 = (task, scene)，main16x32 = (task, init_state)。语料单位：grid 四套件分开 +
  main16x32 整体；family2 另设 grid 四套件合并补充腿（组不变，只合并配对池）。
- late 窗：libero_long（含 SCENE8）q∈[25,35]；goal/object/spatial q∈[12,18]。
- 脉冲分数 = (V+A) 的组内无标签 robust z（组全行中位 / 1.4826·MAD，含成败全 query）；
  脉冲 = z>2 连续 ≥1 行；孤立 = 结束后 3 行内出现 z<1；train = ≥3 脉冲且相邻
  (下一起点−上一终点) ≤ 8；post_pulse_return = 各脉冲结束后 ≤5 行 z 均值的集级均值。
  零个组出现 MAD 退化（bad_mad_groups=[]）。
- 红旗排除（loop/trap 通道无效，AUDIT §8.4）：grid long KITCHEN_SCENE3、两语料 goal
  open_the_middle_drawer —— 从 (b) 三人群与 (c) 事件交叉表剔除；(a) 与 (c) 的
  fail-vs-succ 检验不使用事件，故保留这些任务。
- 检验族（预先声明）：
  - family1（fail=1 vs succ，每单位 3 格 joint maxT）：{cycle_gain, lockin_flag, switch_flag}。
    置换下 lock-in/switching 的组内成功中位阈值随置换标签**重算**（全管线置换，无泄漏）。
  - family2（致败 loop=1 vs 恢复性 loop，每单位 5 格 joint maxT）：{cycle_gain, pulse_count,
    train_flag, isolated_pulse_flag, post_pulse_return}。
  - 其余全部为描述（误报底座、best-k 分布、交叉表、对齐曲线、vs 干净成功 AUC）。

## 存活审计（窗覆盖；AUC 只在覆盖集间配对）

| 单位 | 集数 | 失败 | 覆盖成功 | 覆盖失败 | 双方覆盖组 | 阈值可定义组(≥3 成功覆盖) |
|---|---|---|---|---|---|---|
| grid/goal | 4,000 | 102 | 1,271 | **102（100%）** | 31 | 170/500 |
| grid/long | 4,000 | 267 | 1,138 | **267（100%）** | 65 | 171/500 |
| grid/object | 4,000 | 37 | 3,372 | **37（100%）** | 29 | 448/500 |
| grid/spatial | 4,000 | 126 | 634 | **126（100%）** | 27 | 89/500 |
| main16x32 | 2,560 | 307 | 1,318 | **307（100%）** | 35 | 56/80 |

失败集全部活到 late 窗（超时机制），成功侧只有长尾（spatial 仅 16% 成功集覆盖）。
因此 (a)/(c) 的成功对照系统性偏向"慢成功"，见边界段。

## (a) 周期复发谱：复发是 lag-1 冻结，不是 lag-k 循环

cycle_gain = mob_1 − min_{k=2..8} mob_k 窗内中位（>0 = 存在优于 lag-1 的 lag-k 循环）。

| 单位 | AUC(fail 高) | p_maxT | pairs | fail: %k=1 / 中位 g / %g>0 | succ: %k=1 / 中位 g / %g>0 |
|---|---|---|---|---|---|
| grid/goal | 0.605 | 0.108 | 248 | 42.2% / +0.0029 / 68.6% | 68.5% / −0.0008 / 43.0% |
| grid/long | 0.513 | 0.975 | 491 | **95.1%** / −0.0058 / 7.9% | 63.4% / −0.0034 / 39.4% |
| grid/object | 0.534 | 0.812 | 223 | 56.8% / +0.0012 / 51.4% | **9.3%** / +0.0082 / 92.9% |
| grid/spatial | **0.121** | **0.0005** | 132 | **93.7%** / −0.0125 / 9.5% | 79.8% / −0.0066 / 20.7% |
| main16x32 | 0.526 | 0.631 | 2,941 | **92.8%** / −0.0062 / 10.1% | 82.1% / −0.0064 / 31.2% |

best-k 分布（覆盖集，k=1..8 计数见 summary.json）回答标题问题：

- **失败侧复发压倒性是 lag-1 冻结**：long 失败 95.1%、spatial 93.7%、main 92.8% 的
  best-k=1，中位 cycle_gain 为负。预 named 方向（失败有更高 lag-k 循环增益）**不成立**。
- 唯一过 maxT 的格方向相反：**spatial AUC 0.121（p=0.0005）= 失败的 cycle_gain 显著更低**
  ——失败集比成功集更"冻结"、更缺周期结构。
- object 是镜像证据：成功集 90.7% best-k≥2（峰在 k=3–4，即取放节律的合法 lag-k 循环），
  失败集反而回落到 lag-1（56.8% k=1）。周期性 mob-k 结构在此语料是**健康节律**的标志。
- goal 失败偏混合（58% k≥2，其中 25/102 在 k=8 端点），AUC 0.605 未过 maxT。

## (b) 脉冲检测与恢复性 loop 负对照（核心）

人群（objaware/full 事件层，红旗任务已剔）：

| 单位 | 恢复性 loop | 致败 loop | 干净成功 | family2 双方覆盖组 |
|---|---|---|---|---|
| grid/goal | 12 | 72 | 3,486 | 4 |
| grid/long | 38 | 118 | 3,285 | 6 |
| grid/object | 154 | 29 | 3,809 | 16 |
| grid/spatial | 35 | 68 | 3,839 | 4 |
| grid/ALL 合并（补充腿） | **239** | **287** | — | 30 |
| main16x32 | 18 | 102 | 1,720 | 7 |

### family2：致败(1) vs 恢复性(0)，AUC>0.5 = 致败侧更高

| 格 | goal | long | object | spatial | **grid/ALL** | **main** |
|---|---|---|---|---|---|---|
| cycle_gain | 0.765 (0.51) | 0.677 (0.51) | 0.768 (0.33) | 0.500 (1.0) | **0.732 (0.047)** | 0.492 (1.0) |
| pulse_count | 0.824 (0.36) | 0.548 (1.0) | 0.705 (0.48) | 1.000 (0.32) | 0.696 (0.084) | 0.761 (0.14) |
| train_flag | 0.500 (1.0) | 0.661 (0.55) | 0.589 (0.90) | 0.667 (1.0) | 0.600 (0.51) | 0.529 (1.0) |
| isolated_pulse_flag | 0.735 (0.61) | 0.581 (0.92) | 0.652 (0.70) | 0.750 (0.89) | 0.650 (0.22) | 0.783 (0.094) |
| post_pulse_return | 0.250 (0.59) | 0.577 (0.97) | 0.833 (0.24) | 0.667 (1.0) | 0.590 (0.58) | **0.828 (0.035)** |

（括号 = 族内 maxT p；pairs 见 summary.json，goal/spatial 单套件仅 4–6 组同时含两类，
post_pulse_return 格只覆盖有脉冲的集，pairs 更少——goal 该格 4 对，无解释力。）

**判别性判定（"MoE 能否区分暂时错误与真 Trap"）：有中等强度、方向一致的信号，但
未达复现红线。** 过 maxT 的格各语料一个：grid 合并腿 cycle_gain 0.732（p=0.047，
致败 loop 携带真 lag-k 复发结构而恢复性 loop 没有）；main post_pulse_return 0.828
（p=0.035，致败 loop 脉冲后不回落）。方向复现最好的格是 pulse_count（0.696/0.761，
p=0.084/0.14）与 isolated_pulse_flag（0.650/0.783——注意方向为**致败更高**，与预 named
相反，因该 flag 与脉冲总数正相关：脉冲越多越可能凑出一个孤立脉冲）。cycle_gain 在
main 无方向（0.49）。**没有任何一格在两个语料同时过 maxT** ⇒ 按 §4 不能立"头条主张"，
只能记为初步证据。预 named 主张逐条：

- "恢复性 loop：onset 附近有脉冲 + 快速回落"——**回落成立，"有脉冲"两人群共有**。
  对齐曲线（图）：grid 恢复性 loop 在 lead −3..−1 有明确脉冲峰（z≈0.86–1.06），onset 时
  已在回落（0.56），+4..+5 落到 −0.7..−0.8（低于参照带）；near-onset 脉冲率恢复性 42%/61%
  （goal/long）但 19–37%（object/spatial/main），非普适。
- "致败 loop：脉冲后不回落或成串"——**"不回落"在曲线与 main 检验成立**：grid 致败
  全程 z≈0.80–1.08 平台不落；main 致败 +6..+7 反升到 1.7–2.2；post_pulse_return main
  p=0.035。"成串"（train_flag）方向对但弱（0.53–0.67，全不显著）。

### 描述：三人群拉开的程度（vs 干净成功，组内配对 AUC，无 p）

pulse_count：致败 vs 干净 0.72–0.91（五单位全部 >0.7）；恢复性 vs 干净仅 0.48–0.72。
即**致败 loop 在脉冲维度明确偏离干净成功，恢复性 loop 接近干净成功**——负对照按预期工作。
脉冲级孤立占比（n_iso/n_pulses）：恢复性 0.56–0.91 vs 致败 0.50–0.82（object/spatial 方向符合
预 named，goal/main 相反），集级 flag 受脉冲数混杂，见边界段。

### 误报底座（任何在线报警的假阳性下限，干净成功集）

| 单位 | n | ≥1 脉冲集率 | ≥1 孤立脉冲集率 | train 集率 | 脉冲/100 query |
|---|---|---|---|---|---|
| grid/goal | 3,486 | **55.8%** | 15.6% | 1.35% | 6.02 |
| grid/long | 3,285 | **81.5%** | 76.2% | **10.6%** | 5.70 |
| grid/object | 3,809 | 23.9% | 16.2% | **0.11%** | 1.97 |
| grid/spatial | 3,839 | 45.0% | 27.1% | 0.47% | 5.18 |
| main16x32 | 1,720 | **58.5%** | 31.3% | 2.97% | 5.13 |

**结论：z>2 单脉冲作为在线报警不可用**（干净成功 24–81% 的集至少触发一次）；
train_flag 是唯一低底座的脉冲特征（短套件 0.1–1.4%、main 3.0%），但 long 套件仍高达
10.6%，且其判别力弱（上表）。

### onset 对齐曲线

见 `fig_onset_aligned.png`（lead −4..+8，均值 ±1.96·SEM，n<8 的 lead 不画；干净成功参照带
= 每个 loop 集在同组内随机匹配一个 n_queries>onset 的干净成功集、取相同绝对 query）。
参照带本身在 −3..−2 有小峰、+1..+2 有谷——loop onset 在组内聚集于特定任务相位，
带内结构即任务相位结构；判别信息在**致败 loop 相对参照带/恢复性 loop 的持续抬升**，
而非"有无峰"。

## (c) lock-in 象限：switching 强富集失败，lock-in 基本不存在

阈值 = 组内成功覆盖集中位（≥3 集才定义；无标签泄漏——置换检验中阈值随置换标签重算）。
lock-in = V<∧A<∧margin>∧mob1_w8<；switching = V>∧A>。

| 单位 | lock-in：fail% / succ% / AUC / p | switching：fail% / succ% / AUC / p |
|---|---|---|
| grid/goal | 0.0% / 7.0% / 0.474 / 0.93 | **87.9% / 39.0% / 0.774 / 0.0005** |
| grid/long | 6.9% / 7.0% / 0.512 / 0.98 | 42.5% / 40.3% / 0.531 / 0.74 |
| grid/object | 0.0% / 7.3% / 0.487 / 0.98 | **88.9% / 41.6% / 0.761 / 0.0005** |
| grid/spatial | 4.8% / 7.2% / 0.489 / 1.0 | **76.2% / 41.2% / 0.742 / 0.001** |
| main16x32 | 6.1% / 7.9% / 0.497 / 1.0 | 41.5% / 40.3% / 0.537 / 0.36 |

- **switching 双高象限在全部三个短套件强显著富集失败**（覆盖失败 76–89% 落入，成功基率
  ~40%）；long/main（失败以 SCENE8 static 为主）不富集——与 E3 的 static=flattening 表型
  相容：长任务失败更多是"停"而不是"切"。
- **lock-in 象限不富集失败**：五单位全 null，失败落入率（0–6.9%）≤ 成功基率（7.0–7.9%）。
  作为"表型亚型"的 lock-in **不成立**（零结果照实报）。

### 事件类型 × 象限交叉表（覆盖失败集，红旗任务剔除；全表见 lockin_crosstab.csv）

loop 类失败（loop_only+loop_and_static）落 switching 的比例：goal 23/24、object 25/28、
spatial 8/11、long 29/48、main 54/88；static_only 失败落 switching 仅 long 3/28、main 19/111
（主要落 other/lockin）。**"loop 失败应富集 switching"成立**，且与 static 失败清楚分离。

### lock-in 失败存在性判定与清单

存在但稀少且**不是**"无事件自信做错"：共 **22 例** lock-in 失败，按事件类精确拆分为
**static_only 14 / loop_and_static 4 / loop_only 2 / 无事件 2**（= **18 例带 static 事件**、
6 例带 loop 事件；含 main SCENE8 init_state 13 的 8 例聚簇，static onset 35–39——lock-in
在路由端呈现的多是"静止冻结"集），**无事件仅 2 例**（下表加粗）。
供人工看录像的完整清单（corpus/task/scene/repeat；n_queries 均为超时上限）：

| corpus | suite/task | scene | repeat | loop/static onset |
|---|---|---|---|---|
| grid50x8 | long/KITCHEN_SCENE8 | 15 | 4 | −1 / 34 |
| grid50x8 | long/KITCHEN_SCENE8 | 16 | 0 | 46 / 37 |
| grid50x8 | long/KITCHEN_SCENE8 | 19 | 3 | −1 / 39 |
| grid50x8 | long/LIVING_ROOM_SCENE1 | 4 | 7 | 11 / 46 |
| grid50x8 | long/LIVING_ROOM_SCENE2_cheese+butter | 33 | 5 | 10 / −1 |
| grid50x8 | long/LIVING_ROOM_SCENE6 | 30 | 6 | −1 / 32 |
| grid50x8 | spatial/bowl_in_top_drawer | 32 | 1 | 20 / −1 |
| main16x32 | long/KITCHEN_SCENE8 | 0 | 7 | −1 / 37 |
| main16x32 | long/KITCHEN_SCENE8 | 3 | 14 | 42 / 50 |
| main16x32 | long/KITCHEN_SCENE8 | 10 | 1 | −1 / 35 |
| main16x32 | long/KITCHEN_SCENE8 | 13 | 2,9,10,14,16,25,27,30 | −1 / 35–39（8 例聚簇） |
| main16x32 | long/KITCHEN_SCENE8 | 23 | 16 | 48 / 43 |
| main16x32 | long/KITCHEN_SCENE8 | 42 | 12 | −1 / 39 |
| **main16x32** | **long/KITCHEN_SCENE8** | **42** | **27** | **−1 / −1（无事件）** |
| **main16x32** | **spatial/bowl_on_stove** | **29** | **16** | **−1 / −1（无事件）** |

「自信持续做错」候选就是这最后 2 例；量级上该表型在两套语料 1.86 万集中几乎不存在。

**反向验证（无事件失败落在哪个象限）**：上表是"lock-in 失败里有几个无事件"；反过来问
"无事件失败是不是 lock-in"更能证伪。实测方向相反——无事件失败的 **switching 率**
goal 83.3%(n=6)、object 83.3%(6)、spatial 80.0%(10)、main 61.7%(47)、long 40.0%(10)，
而 **lock-in 率 0–4.3%**（同单位成功基率 6.8–7.9%，即无事件失败的 lock-in 率还**低于**
成功基率）。**无事件失败在路由端与 loop 失败同型（切换不稳），不是"冻结且自信"。**
两个方向合起来关闭 lock-in 亚型主张：它既不富集失败，也不对应无事件失败。

## 补充腿（并行 `run_e4_supp.py`，结果在 summary.json:supplementary）

同目录并行补充 run 在主族之外加了 4 条稳健性腿（口径复用主管线 load_task）：

1. **mob1_w8 残差腿（§4 新颖性条款）**：switch_flag 富集在三个短套件残差后**存活**
   （goal 0.687/p=0.0055、object 0.662/p=0.010、spatial 0.725/p=0.025）——switching
   信号不是 mob1_w8 的改写。spatial cycle_gain 反向结果残差后仍显著（0.227/p=0.006）。
   grid 合并 family2 cycle_gain 残差后存活（0.777/p=0.0295）；main post_pulse_return
   残差后降为边缘（0.852/p=0.063）。
2. **集长混杂（重要，足以推翻 pulse_count 格）**：组内配对的 n_queries AUC 本身是
   **1.000 / 0.997 / 1.000 / 1.000 / 0.9995**（失败 vs 成功，goal/long/object/spatial/main）
   与 **1.000 / 0.936 / 1.000 / 1.000 / 1.000**（致败 vs 恢复性 loop，grid 合并 0.982）；
   集长中位致败 30/52/28/22/52 vs 恢复性 20.5/36/17/13/18。
   即"这一集很长"这个零成本、非路由的特征在两个对比上都给 AUC 0.94–1.00，
   而 family2 最好的路由格只有 0.70–0.83。**pulse_count（全集累计脉冲数）作为判别子
   没有任何增量——它是集长的单调弱化版**，"致败 loop 脉冲更多"必须改读为
   "致败 loop 更长、因而脉冲更多"。(b) 中仅 `post_pulse_return`（每脉冲后 z 均值）与
   `cycle_gain`（窗内中位）是长度归一化量，是唯二可能携带非平凡信息的格。
3. **task 级配对补充腿**（配对更粗、功效更高）：pulse_count 在 grid 合并
   （0.662/p=0.005）与 main（0.720/p=0.037）**同向显著**，spatial 出现
   cycle_gain 反向 p=0.001 + post_pulse_return 0.812/p=0.001；但该腿把 scene 差异
   放进对比且承接上条长度混杂，仍不升级为头条。
4. **MAD 尺度稳健性**（z 不乘 1.4826 = 任务书字面口径，阈值实际松 1.35×）：判别全面增强，
   且 **`post_pulse_return` 在两语料同时显著同向**——grid 合并 0.792（p=0.0005）、
   main 0.803（p=0.0265）；grid 合并 cycle_gain 0.732（p=0.0035）、pulse_count 0.832
   （p=0.0005）。这是全实验里唯一逼近 §4 复现红线（≥2 独立单位同向 + 显著）的结果。
   **但本报告不据此改判**：z 尺度是事后可选项，主口径已冻结为 1.4826·MAD，主口径下
   post_pulse_return 只在 main 单语料显著；且松阈值的代价是误报底座同步恶化
   （干净成功孤立脉冲率 grid 59.0%、main 68.7%，train 率 11.9%/15.0%）。
   如实记录为"**结论对 z 尺度敏感**"，并把 raw-MAD 结果留给预注册的后续实验去确认。

## 边界段（诚实条款）

1. **窗存活选择偏倚**：失败集 100% 活到 late 窗（超时），成功侧只有长尾覆盖
   （spatial 16%、long 30%、goal 33%、main 59%、object 85%）。(a)/(c) 的 AUC 语义是
   "失败 vs **慢成功**"；spatial 的 cycle_gain 反向显著也应在此语义下读。
2. **(b) 单套件功效不足**：family2 每套件同时含两类 loop 的组仅 4–16 个（pairs 17–56），
   单套件 p 无解释力；合并腿（30 组/108–110 对）与 main（7 组/63–69 对）才有初步功效。
   main 恢复性 loop 仅 18 集。**两条显著格（grid 合并 cycle_gain、main post_pulse_return）
   均未在另一语料复现方向+显著性，按 §4 不构成头条主张。**
3. **恢复性 loop 人群纯度**：objaware 代理在高重试 pick 任务成功集触发率偏高
   （bbq_sauce 14.7%、milk 5.9%，AUDIT §8.4），grid/object 的 154 个"恢复性 loop"中可能
   混入代理假阳性；无真值不可分辨其对 family2 的稀释方向。
4. **isolated_pulse_flag 与 pulse_count 混杂**：集级"有无孤立脉冲"随脉冲总数单调上升，
   致败侧反而更高；按预 named 语义应看脉冲级孤立占比（已给描述），该口径下方向在
   object/spatial 符合、goal/main 相反——此格的预 named 方向定义不佳，如实记录。
5. **post_pulse_return 覆盖偏窄**：仅有脉冲的集有值（pairs 3–39/单位）；集尾脉冲无后续行
   不计入；孤立判定在集尾 <3 行时保守记"非孤立"。
6. **组内 MAD 的组很小**（grid 每组 8 集），z 尺度估计噪声大但无退化组；main 每组 32 集
   更稳。曲线的参照带含任务相位结构（onset 聚集），只能作带内对照不能当零线。
7. **红旗任务**：SCENE3 与 middle_drawer 只从事件相关分析剔除；其 fail-vs-succ 格保留
   （不使用事件）。SCENE3 的 6 个失败仍在 grid/long family1 内。
8. **集长混杂**：致败 loop ≈ 超时长集（n_queries AUC 0.94–1.0，见补充腿 2），
   pulse_count/train 等累计量的 family2 差异含机械成分；窗口量（cycle_gain、象限）与
   每脉冲量（post_pulse_return）不受此机械混杂，但仍与"失败=长"共变。
9. **多重性口径**：maxT 只覆盖各单位声明族（3 格 / 5 格）；跨 6 个单位未再校正（与 E5/E7
   跨语料口径一致）；描述性 AUC 一律无 p；补充腿为事后稳健性检查，不计入声明族。

## 结论（逐子实验判定）

**(a)** 失败端的"复发"是 **lag-1 冻结而非 lag-k 循环**（失败集 best-k=1 占比 long 95.1%、
spatial 93.7%、main 92.8%；唯一过 maxT 的格 spatial AUC 0.121/p=0.0005 方向为"失败更冻结"，
残差后仍存活 0.227/p=0.006）。预 named 的"失败=更强 lag-k 循环"**不成立**。

**(c)** 短套件失败在路由上压倒性落 **switching 双高象限**（AUC 0.742–0.774，p≤0.001，
三套件独立复现且对 mob1_w8 残差后全部存活）；long/main 的 static 型失败反而**去**富集
switching（loop 失败 vs static 失败的 switching 判别 AUC task 级 0.706–0.74）。
**lock-in「自信持续做错」亚型不成立**：落入率 0–6.9% ≤ 成功基率 7.0–7.9%（≈4 轴独立
中位分割的 6.25% 机会水平），22 例 lock-in 失败里 18 例带 static 事件、仅 2 例无事件；
反向看，无事件失败有 40–83% 落 switching、仅 0–4.3% 落 lock-in。

**(b)** **恢复性 loop 与致败 loop 的判别性未建立。** 主口径（scene 级配对、1.4826·MAD）
下只有 grid 合并 cycle_gain 0.732/p=0.047 与 main post_pulse_return 0.828/p=0.035 过 maxT，
互不复现；换成 task 级配对两者都塌（0.44 / 0.47）。唯一两语料同向显著的 pulse_count
**被集长完全解释**——n_queries 自身的组内配对 AUC 就是 0.94–1.00，高于任何路由格
（0.70–0.83），故 pulse_count 无增量。定性上负对照按预期工作（对齐曲线：致败 loop 在
onset 后维持 z≈0.8–1.1 平台且 main 于 +6..+7 反升到 1.7–2.2，恢复性 loop 于 +4..+5
跌到 −0.7..−0.8 低于参照带），但**定量证据不足以宣称"MoE 能区分暂时错误与真 Trap"**。
最接近复现红线的是 raw-MAD 口径下的 post_pulse_return（0.792/p=0.0005 与 0.803/p=0.0265
两语料同向显著）——因 z 尺度属事后自由度，只作为下一轮预注册的候选主张。

**误报底座**：z>2 单脉冲作在线报警不可用（干净成功 24–81% 的集至少触发一次孤立脉冲
15.6–76.2%）；train_flag 底座低得多（0.11–10.6%）但判别力弱（0.53–0.67，全不显著）。

## 产物

- `summary.json`（全部数字：族 AUC/p/pairs、人群、误报、best-k、lock-in 清单、审计）
- `pulse_stats.csv`（18,560 集 × 31 列逐集统计：脉冲/窗均值/象限/人群）
- `lockin_crosstab.csv`（单位 × 成败 × 事件类 × 象限计数）
- `fig_onset_aligned.png`（三人群 onset 对齐 V+A z，grid/main 两 panel，英文标签）
- `curves_onset_aligned.npz`（曲线原始矩阵）、`run_e4.py`/`make_fig.py`/`run_e4.log`（复现）
- `run_e4_supp.py`（补充腿脚本，复用 `run_e4.load_task`；结果并入 summary.json:supplementary）

数据只读：全程只读 `features/`、`events/`，写入仅限本目录。
