# 检测器原型共享约束（所有 sandbox agent 必读并遵守）

## 目标

设计一个**只读 MoE routing** 的在线检测器，下游是**触发干预**（重采 flow 噪声 / 重规划）。
双通道：**loop** 与 **static**（二者在门控维度方向相反，不可合并成单一分数）。
报警时必须连带输出类型（分型）。

## 数据切分（**违反即作废**）

- **校准/开发集 = SCENE8 两个单位，仅此**：
  - `features/main16x32/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/`（512 集，216 失败）
  - `features/grid50x8/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/`（400 集，138 失败）
  - 对应 `events/<corpus>/libero_long/KITCHEN_SCENE8.../events.csv`（`proxy_grade=full`，唯一完整代理层）
- **留出集 = 其余 39 个 grid 任务 + 4 个 main 任务：绝对不许读取、不许统计、不许"看一眼"。**
  表型的**结构**（信号/lead/方向）本身是在 SCENE8 上发现的，所以留出的是结构不只是数据。
  你的产物必须能被别人在留出集上一次性执行，而你自己从未见过留出集的任何数字。

## 硬约束

1. **阈值只能从「成功集」分布、留一组外（leave-one-group-out）校准**；组 = scene(=init_state)。
   不许用失败集定阈值，不许用结局标签拟合任何参数。
2. **不训练分类器、不拟合 PCA/k-means/probe、不做梯度优化。** 允许：分位点、中位/MAD、
   计数、状态机。
3. **必配消融臂 C（强制）**：同一套状态机结构，把一级信号换成既有最强 train-free 基线
   `mob1_w8`（d9+Hellinger+W8）。如果你的检测器打不过 C，如实写出来——这是本轮最大的
   诚实弱点（多数表型腿扛不住 mob1_w8 残差），必须直面。
4. **旋钮必须一次冻死**：你可以在校准集上探索，但最终交付的参数表必须是**一组**，
   不是"留待调"。每个旋钮写清语义与选定依据。禁止用留出集选参数（你也读不到）。
5. 在线因果性：t 时刻的判定只能用 ≤t 的行。不许用整段轨迹、不许用未来。
6. `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`（240 核机）。
   数据目录只读，只写自己的 `DESIGN_SANDBOX/<你的目录>/`。

## 必报指标（校准集上，留一组外）

- **每集误报率**：在**干净成功集**（无任何事件）上触发的比例。既有基线参考：
  单脉冲 z>2 的误报底座是 **24–81%/集**，这是要打败的对象。
- **检出率**：致败 loop / static 事件集中被触发的比例。
- **提前量**：报警时刻相对物理 onset 的分布（中位与四分位）。**负数=早于 onset**。
  下游是干预，所以提前量比低 FA 更值钱，但 FA 必须报。
- **恢复性 vs 致败的判别**：成功集里的瞬态 loop（SCENE8 full 层：main 2 个、grid 4 个——
  数量少，如实说明功效不足；必要时用 objaware 层同任务补充并标注等级）不应被报警，
  致败 loop 应被报警。这是"能否离开"的直接兑现。
- **两语料一致性**：main 与 grid 两个 SCENE8 单位各自独立报，不合并。

## 交付

`DESIGN_SANDBOX/<你的目录>/`：
- `SPEC.md`：冻结的检测器规格——伪代码级别的判定规则、参数表（每个值 + 语义 + 选定依据）、
  输入契约（需要哪些 rows.npz 列）。别人拿这个就能在留出集上执行。
- `detector.py`：可执行实现，接口 `detect(rows_npz_path, group_col) -> per-episode alarms`
- `calib_report.zh.md`：校准集上的指标表（含消融臂 C）、失败模式、你自己发现的弱点
- `summary.json`

诚实优先：零结果、打不过基线、旋钮敏感——全部照实写。**不要为了让方案好看而调参。**
