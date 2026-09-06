# MoE 路由失败检测器是否按物理失败模式分工？

**结论先行。**

1. **原始探针完全复现，但「按模式分工」大部分是任务混淆。** 控制 suite 后，261 个
   （检测器 × 模式）单元里 20 个显著；控制到 **task** 后只剩 2 个，而 task 分层的
   最小可检测效应**更小**（0.020–0.033 vs 0.029–0.046），所以这是效应真的缩水，
   不是检验没功效。最大的模式 `dropped`（n=241）的效应有 **89%** 来自任务层面。
2. **但有一条轴是真的，而且它不在「哪个路由量」上，在「阈值校准方式」上。**
   对 `stable_grasp_not_observed`（未观测到稳定抓取），**global（池化分位数、
   不用任务身份）比 per_task（同任务分位数）好，12 个路由量 12/12 全部同号**，
   external 精确符号翻转检验 p = 0.00049（去掉一对重复量后 11/11，p = 0.00098），
   development 上 7/12、p = 0.033。这是本研究唯一一条在 task 分层下复现的、
   非小样本的模式轴。
3. **可解释的多头是可能的，但只有一个头能被命名。** 名字是
   `stable_grasp_not_observed`，头的机制定义是「绝对路由异常头 = global 池化阈值」。
   证据来自模拟器标签，不来自检测器互相聚类。用标签**事先预测**出的两个 global 头
   （`mobility|global`、`conditional_query_d1|global`）确实、也**只有**它们能把
   per_task 头的抓取盲区补上（no_grasp 召回 0.420 → 0.839 / 0.881，其余十个
   global 头最好只有 0.643）。
4. **多头在 global 家族和混合两头上确实赢过单头；在 per_task 家族基本不赢。**
   global-only：同样 ≤57 external FP 下，4 头 **392 TP / 57 FP**（precision 0.873，
   lift 1.485）对最好单头 237 TP / 36 FP。混合两头 **460 TP / 106 FP**
   （recall 0.816，lift 1.616，早期 105 TP / 72 FP）对同量级 per_task 单头 370–384 TP。
   但 v7 guard（本身就是 4 头 bundle）439 TP / 80 FP、precision 0.846 仍未被两头超越。

**一句话：模式分工在「路由量 → 物理模式」这一层基本是任务混淆的假象；真正稳健的
分工在「参考系（是否使用任务身份）→ 物理模式」这一层。多头因此买到了一点点可解释性
（一个头可以被诚实地命名为「抓取失败头」），但远不是「每个头一个物理模式」。**

---

## 0. 数据与连接（已独立复核）

`experiments/verify_join.py` 从原始文件重建连接，未采信控制器的说法：

| | development (`right-50x8-20260903`) | external (`right-50x8b-20260903`) |
|---|---|---|
| 路由缓存行数 | 14,800 | 15,600 |
| 物理标签匹配 | 14,800 / 14,800 | 15,600 / 15,600 |
| risk 数 | 487 | 564 |
| 带 `primary_failure_reason` 的 risk | 487 | 564 |
| `physics_validation_status` | 全部 `passed` | 全部 `passed` |
| 未用于连接的 `init_state_id` / `flow_noise_seed` 一致 | 14,800 / 14,800 | 15,600 / 15,600 |
| `recorded_success` 与结局标签一致 | 14,800 | 15,600 |

**置信度（须报告）**：external 523 个持续失败中 **medium 497、high 26**；把 41 个
late success 也算进来的 564 个 risk 里是 medium 538 / high 26。**没有一条高于 high，
绝大多数是 medium。** 标签作者明确说明这是**失败模式标签，不是策略内部因果机制的证明**，
本报告全程遵守这一框定。

external 持续失败模式计数完全吻合：dropped 231、no_grasp 136、moved_unmet 83、
regressed 26、released_outside 24、no_contact 14、timeout_holding 8、mechanism 1。

**一个结构性事实（后面所有时序控制的基础）**：**每一条 risk 轨迹的长度都恰好等于其
suite 的 horizon cap**（487/487 与 564/564）。所以在同一 suite 内，不同物理模式的
episode 暴露给检测器的 chunk 数完全相同，不存在「机会不等」这种时序混淆。生存先验
跨过 0.25 的 chunk 也独立复核为 **goal 18 / long 26 / object 17 / spatial 13**。

24 个冻结检测器在 external 上被逐比特重建成功（`results/alarm_rebuild_audit.json`，
24/24 完全一致），development 侧用同样机制新建（per_task 用留一 init-state 交叉拟合
阈值，global 用 development_main + development_extra 池化分位数）。

---

## 1. 探针复现：逐格一致

`results/mode_recall.csv`。external、564 个 risk：

| 检测器 | dropped (241) | no_grasp (143) | moved_unmet (89) | regressed (26) | released_outside (25) | timeout_holding (23) | 总召回 |
|---|---|---|---|---|---|---|---|
| mobility\|global | 0.253 | 0.545 | 0.303 | 0.462 | 0.120 | 0.391 | 0.346 |
| mobility\|per_task | 0.631 | 0.252 | 0.483 | 0.462 | 0.400 | 0.522 | 0.482 |
| expert_load\|per_task | 0.838 | 0.420 | 0.629 | 0.654 | 0.880 | 0.391 | 0.656 |
| flow_settling\|per_task | 0.672 | 0.350 | 0.438 | 0.346 | 0.800 | 0.609 | 0.539 |
| v7 guard | 0.780 | 0.832 | 0.674 | 0.846 | 0.640 | 0.783 | 0.778 |

**30 格全部与控制器给出的数字一致。** 这是复现，不是独立检验（external 的这张表在
本研究开始前已被看到，见 `results/PREREG.md` 的「先验暴露」声明）。

---

## 2. Q1：混淆控制后还剩什么

统计量 = 检测器 d 在模式 m 上的召回；零假设 = 在**层内**打乱物理模式；效应
`SEL(d,m) = log2( 召回 / 层匹配期望 )`，其中期望 = 检测器**自己**的各层召回按模式 m
的层分布重新加权 —— 因此效应天然相对于检测器自身的总召回。20,000 次置换，种子 20260906，
BH q=0.05。

### 2.1 结果

| 分层 | external 显著格 | development 显著格 | dev→ext 复现格（同号且 ext 原始 p<0.05） |
|---|---|---|---|
| suite | 20 / 261 | 36 / 261 | **20**（零假设下期望 1.8） |
| task | 2 / 261 | 6 / 261 | **2**（均为 n=14 小格） |

### 2.2 task 分层不是「没功效」，是效应真的消失

| 模式 | n (ext) | MDE(suite) | MDE(task) |
|---|---|---|---|
| dropped | 241 | 0.029 | **0.020** |
| no_grasp | 143 | 0.046 | **0.033** |
| moved_unmet | 89 | 0.065 | **0.050** |
| regressed | 26 | 0.124 | **0.106** |

task 分层的最小可检测效应**处处更小**。混淆结构：Cramér's V(task, mode) = 0.553、
V(suite, mode) = 0.360；564 个 risk 中 **515 个**落在「至少两种模式」的任务里，
`no_grasp` 的 143 个**全部**落在混合模式任务里 —— 置换自由度是存在的。

**效应分解**（`|suite 控制后效应| > 0.05` 且 n≥25 的 59 格）：within-task 份额
中位数 **0.491**；但按模式看差别极大 —— `dropped` 仅 **0.113**（即 89% 是任务效应），
`regressed` 0.649、`moved_unmet` 0.610、`no_grasp` 0.345。

典型例子（external）：

- `expert_load|per_task` × dropped：召回 0.838，suite 期望 0.794（SEL +0.078, p=0.007），
  **task 期望 0.840（SEL −0.004, p=0.842）** —— 完全是任务效应。
- `mobility|per_task` × no_grasp：0.252 vs suite 期望 0.405（SEL −0.687, p<1e-4），
  **task 期望 0.269（SEL −0.098, p=0.288）**。
- `mobility|global` × no_grasp：0.545 vs suite 期望 0.458（SEL +0.251, p=0.004），
  **task 期望 0.468（SEL +0.222, p=0.002）** —— 这一格 task 分层下依然存活。

### 2.3 时序控制

因为所有 risk 都跑满 cap，"生存先验匹配召回" 与 "早期（先验<0.25）召回" 在数值上
**完全等价**（两张表逐格相同），所以只报一张。结论对分工叙事很不利：

| 检测器 | dropped 早期召回 | no_grasp 早期召回 | no_grasp 平均报警 chunk |
|---|---|---|---|
| mobility\|global | 0.062 | **0.035** | 34.6 |
| mobility\|per_task | 0.129 | **0.035** | 26.6 |
| expert_load\|per_task | 0.207 | **0.084** | 22.2 |
| v7 guard | 0.104 | **0.049** | 29.9 |

`mobility|global` 在 no_grasp 上的优势**全部发生在晚期**（平均报警时先验已 0.516）。
早期带内它与 `mobility|per_task` 完全打平（0.035 = 5/143）。**所以：这条模式轴是
真的，但它不产出早期预警。** `no_grasp` 对所有检测器都是早期召回最低的模式。

### 2.4 v7 是不是通才

**external 上是。** v7 guard 的 omnibus 置换 p = 0.179（suite）/ 0.561（task），
没有任何一个模式格显著。但 **development 上它不是通才**（p = 0.003 / 0.028），
唯一 dev→ext 复现的缺口是 `object_moved_but_goal_unmet`：
dev 0.625 vs 期望 0.777（SEL −0.314），ext 0.674 vs 0.764（SEL −0.181, p=0.021）。
即 v7 guard 略微系统性地漏掉「物体动了但目标没达成」。

---

## 3. Q2：真正稳健的映射是「参考系 → 模式」，不是「路由量 → 模式」

### 3.1 配对设计

12 个路由量每个都有 `global` 与 `per_task` 两个版本，构成 12 个配对。对每个模式 m，
取 `delta_q = [召回−task期望](q|global) − [召回−task期望](q|per_task)`，做**精确**
符号翻转置换（2^12 = 4096 枚举）。

| 模式 | n(ext) | ext 平均 delta | ext 同号 | ext 精确 p | dev 同号 | dev 精确 p |
|---|---|---|---|---|---|---|
| **no_grasp** | **143** | **+0.0361** | **12/12** | **0.00049** | 7/12 | **0.033** |
| regressed | 26 | −0.1018 | 1/12 | 0.0015 | 3/12 | 0.012 |
| dropped | 241 | −0.0098 | 4/12 | 0.153 | 5/12 | 0.379 |
| moved_unmet | 89 | +0.0041 | 5/12 | 0.724 | 5/12 | 0.499 |
| released_outside | 25 | +0.0107 | 6/12 | 0.687 | 8/12 | 0.087 |
| timeout_holding | 23 | −0.0183 | 6/12 | 0.510 | 6/12 | 0.510 |

`state_action_alignment` 与 `conditional_energy` 被选到同一层、方向镜像，产出**逐比特
相同**的报警，因此 12 对中有一对是重复。去重后 `no_grasp` 仍是 **11/11、p = 0.00098**。

**方向**：`global` 头**多抓** no_grasp，`per_task` 头**少抓**；`regressed` 相反
（但 n=26，小格，只作提示）。

### 3.2 机制核对（用标签只做分析）

提出的机制：per_task 阈值用**该任务自己的** episode 标定；如果一个任务主要以「从未
形成稳定抓取」失败，它的参考分布里装满了本该被标记的行为，同任务分位数被抬高，
检测器就瞎了。

| 队列 | 任务 no_grasp 占比 vs 任务 risk 率 Pearson | risk 率（no_grasp 占比最高 1/3 任务） | risk 率（最低 1/3） |
|---|---|---|---|
| development | 0.447 | 0.084 | 0.027 |
| external | **0.549** | **0.101** | **0.036** |

no_grasp 占比高的任务失败率约为低占比任务的 **2.8–3.1 倍**，参考分布确实被污染。
对照：`dropped` 占比与任务 risk 率相关为 **负**（−0.23），不存在同样污染。

**注意**：这是关于**检测器校准**的机制，不是关于策略内部机制的断言；后者标签作者
明确拒绝支持。

### 3.3 dev → external 稳定性

83 个双队列 n≥30 的格：Spearman(SEL_dev, SEL_ext) = **0.632**（suite）/ **0.735**（task）；
符号一致率 **0.771**；每检测器 argmax 模式一致率 0.429（suite）/ **0.786**（task）。
即：**排序稳定，但 suite 分层下的「最强模式」有一半以上换人**，只有 task 分层下才稳。

### 3.4 20 个 suite 分层复现格的全貌

- **11 格是 `no_grasp`**：其中 10 格为负（几乎所有检测器系统性漏抓），
  唯一为正的是 `partial_edge_std|global`（dev SEL +0.654 / ext +0.420, p=0.001）。
- **5 格是 `dropped`**：全为正（`conditional_query_d1|per_task` +0.352、
  `partial_query_d1|per_task` +0.315、`v7_periodicity` +0.177、`mobility|per_task` +0.135、
  `expert_load|per_task` +0.078）—— 但这 5 格在 task 分层下全部塌陷（|SEL| ≤ 0.17）。
- task 分层下唯一复现的 2 格都是 `no_contact`（n=14，dev 与 ext 各 14）：
  `v7_periodicity` 0.786 vs 期望 0.361（p=1e-4）、`v7_turbulence` 0.643 vs 0.188（p<1e-4）。
  **两队列召回一模一样（11/14 与 9/14），复现极强，但 n=14 不足以支撑结论。**

---

## 4. Q3：可解释的多头

### 4.1 预注册的「最大化 TP」多头产出的头**没有一个**能被命名

`results/multihead.json`。按预注册规则（development outcomes 贪心 OR、FP 预算），
5 个 bundle 的**全部 15 个头**在严格命名规则（dev argmax 模式 + SEL>0 + suite 与
task 两种分层下 p<0.05）下都得到 `unselective`。放宽到只看 suite 分层时，它们几乎
全部被命名为 `dropped` —— 也就是**最大的那个模式**，这不是区分度，是基率。

**这条本身就是结论：为 TP 挑选的 bundle 是集成多样性装置，不是可解释性装置。**

### 4.2 唯一可命名的头，和它的非循环证据

- **机制定义（路由语言）**：`global` 头 = 在整个未标注参考语料上取**一个**池化分位数，
  运行时不使用任务身份，因此它对**绝对**路由异常报警，而不是对「相对于本任务而言的」
  异常报警。
- **物理模式名**：`stable_grasp_not_observed`。
- **不来自该头自身的证据**：§3.1 的 12/12 配对检验（证据来自模拟器标签在**全部 12 个
  路由量**上的一致方向，与任何单个头无关）、§3.2 的参考污染机制、以及下面的预测检验。

**预测检验（`results/partner_control.json`）**：仅凭 task 分层选择性（物理标签），
预言只有 `mobility|global` 与 `conditional_query_d1|global` 是正选择性的抓取头。
把它们分别与最好的 per_task 头 `expert_load_effective_rank|per_task` 配对：

| 搭档 global 头 | no_grasp 召回 | dev TP | ext TP / FP |
|---|---|---|---|
| （单独 base） | 0.420 | 322 | 370 / 93 |
| **mobility\|global**（预言） | **0.839** | 395 | 460 / 106 |
| **conditional_query_d1\|global**（预言） | **0.881** | 395 | 470 / 118 |
| 其余 10 个 global 头中最好者 | 0.643 | — | — |
| 其余 10 个的中位数 | 0.455 | — | — |

**预言完全成立**：两个被预言的头的 no_grasp 召回都高于其余十个中的**任何**一个；
而且它们同时也是按 **development TP** 排名的前二（395、395）—— 即用纯 outcome 准则
也会选到它们。**可解释性在这里是免费的。**

### 4.3 多头到底多抓了什么（标签判定，不可能循环）

`results/bundle_mode_gain.csv`。零假设：把单头**漏掉**的 risk 的物理模式在层内打乱。

**两头（`expert_load|per_task` OR `mobility|global`）相对 `expert_load|per_task`**：
额外 **+90 TP / +13 FP**。这 90 个额外命中的模式构成：

| 模式 | n | 单头抓到 | 两头抓到 | 额外 | suite 期望 | task 期望 | p(suite) | p(task) |
|---|---|---|---|---|---|---|---|---|
| **no_grasp** | 143 | 60 | **120** | **60** | 35.3 | 43.9 | **0.0000** | **0.0000** |
| dropped | 241 | 202 | 210 | 8 | 18.8 | 13.1 | 0.000 | 0.011 |
| moved_unmet | 89 | 56 | 65 | 9 | 15.8 | 12.1 | 0.029 | 0.213 |
| regressed | 26 | 17 | 19 | 2 | 8.2 | 6.2 | 0.011 | 0.019 |
| timeout_holding | 23 | 9 | 16 | 7 | 5.1 | 6.3 | 0.404 | 0.760 |

集中度 omnibus **p = 0.0000（suite）与 0.0000（task）**。第二个头把 `no_grasp` 召回
从 0.420 抬到 0.839，而对 `dropped` 几乎没贡献（8 个，显著**低于**期望）。
**这就是「第二个头的名字是抓取失败头」的可验证含义。**

对照，其它 bundle 的额外收益集中度弱得多：`global_fp57` p=0.015/0.047，
`task_agnostic_fp100` p=0.044/0.112（且其唯一显著格是 n=25 的 `released_outside`）。

### 4.4 无法命名的头

按 task 分层命名规则（`results/named_heads.csv`）能命名的头只有 8 个，其中 6 个的
命名模式是 `no_contact`（n=14）、1 个 `regressed`（n=26）、1 个 `released_outside`
（n=18/25）—— **全部是小格**。因此「每个头一个物理模式」的多头是**做不出来的**：
只有一个非小格的名字（`stable_grasp_not_observed`），而且它对应的是参考系而不是路由量。

---

## 5. Q4：等假警下多头是否赢过单头

`results/headline_operating_points.csv`、`results/frontier.json`。

### 5.1 预注册（匹配 **development** FP，选择从不看 external）

**预注册协议部分失效，必须明说：development 的 FP 计数不迁移到 external。**
dev FP ≤ 16 的 global bundle 在 external 上是 40 FP；dev FP ≤ 88 的 per_task bundle
在 external 上是 155 FP。所以「匹配 dev FP」并不等于「匹配 external FP」。

| 预算 | bundle (ext) | 匹配后的单头 (ext) |
|---|---|---|
| global, dev FP≤16 | 255 TP / 40 FP | `mobility\|global` q=0.975：195 TP / 17 FP |
| global, dev FP≤12 | 114 TP / 21 FP | `mobility\|global` q=0.98：178 TP / 13 FP |
| per_task, dev FP≤88 | 409 TP / 155 FP | `expert_load\|per_task` q=0.80：384 TP / 119 FP |
| per_task, dev FP≤54 | 372 TP / 101 FP | `expert_load\|per_task` q=0.85：370 TP / 93 FP |
| v7+global, dev FP≤76 | 473 TP / 94 FP | v7 guard：439 TP / 80 FP |

在这个口径下**没有一个 bundle 是干净的胜利** —— TP 涨，FP 也涨。

### 5.2 描述性前沿（在 external TP–FP 平面上比较；这是**事后**口径）

前沿的每个点都由一个 development 侧旋钮（预声明分位数网格或贪心预算）生成，
external 只用于**读出**结果，不用于选择。

**global 家族（唯一能证明信号在 MoE 侧的家族）**

| external FP 上限 | 最好单头 | 最好 bundle |
|---|---|---|
| ≤17 | `expert_load\|global` q=0.96 **200 TP / 16 FP** | 82 TP / 13 FP（**大输**） |
| ≤40 | `expert_load\|global` 237 / 36 | 255 / 40（+18） |
| ≤57 | `expert_load\|global` 237 / 36 | **392 / 57**（**+155**，precision 0.873，lift 1.485，早期 59 TP / 30 FP） |
| ≤71 | `expert_load\|global` 269 / 62 | **407 / 71**（+138，lift 1.564，早期 86 TP / 39 FP） |

**但要诚实**：项目此前冻结的 global 组合是 **288 TP / 13 FP**，本研究的贪心在
13 FP 附近只有 82 TP，**远远不如**。本研究的贪心目标（最大化 dev TP）在极低 FP 端
是错的目标；在 40–71 FP 区间才优。

**per_task 家族**：≤57 FP 打平（272 vs 272）、≤93 FP 打平（370 vs 370）、
≤120 FP 仅 +10、≤155 FP +45。**多头在 per_task 家族基本不买账。**

**混合两头（事后，且使用任务身份，因此本身不证明 MoE 侧信号）**：
`expert_load|per_task` OR `mobility|global` = **460 TP / 106 FP**，precision 0.813，
recall 0.816，lift 1.616，早期 105 TP / 72 FP。另一版
（OR `conditional_query_d1|global`）= 470 TP / 118 FP，lift 1.670，早期 126 TP / 79 FP。
同量级 FP 下最好的 per_task 单头是 370–384 TP，两头 **+76 ~ +90 TP**。

**但两头没有赢过 v7 guard**：439 TP / 80 FP、precision 0.846。两头多 21 TP，
多 26 FP，精度更低。而且 v7 guard **本身就是一个 4 头 bundle**，
`v7 guard OR mobility|global` = 439 TP / 81 FP，与 v7 guard **完全一样** ——
抓取头被 v7 guard 完整吞掉（v7 guard 的 no_grasp 召回已经是 0.832）。

---

## 6. 失败的东西 / 局限

1. **预注册的等假警协议失效**：dev FP 不迁移到 external（§5.1）。前沿比较是事后替代品，
   已标注。
2. **预注册的命名规则过严**：它要求 suite 与 task 两种分层同时显著，但一个纯粹
   within-task 的真实效应本来就可能在 suite 分层下不显著（`mobility|global` × no_grasp
   正是如此：dev 上 task p=0.028 而 suite p=0.148）。§4.2 使用的 task-优先规则是**事后**的，
   理由是 task 分层严格强于 suite 分层。
3. **贪心 bundle 在极低 FP 端明显不如既有冻结组合**（82 vs 288 TP @ ~13 FP）。
4. **小格**：`no_contact` n=14、`released_outside` n=25、`regressed` n=26、
   `timeout_holding` n=23、`mechanism` n=1、`no_meaningful_target_progress` n=2。
   task 分层下唯一复现的两格都是 n=14，**不据此下结论**。
5. **标签置信度**：external 持续失败 497/523 为 medium。所有模式级结论都继承这一
   不确定性。
6. **唯一稳健的模式轴不产出早期预警**（§2.3）。
7. **两头使用任务身份**，只有它的 global 分量满足「无任务侧信息」。

---

## 7. 下一步

1. **把「参考系」当成头的设计维度，而不是评测口径。** 既然分工在 global/per_task 之间
   而不在路由量之间，值得系统扫一个**中间**参考系网格：suite 级分位数、
   任务聚类级分位数、按任务失败率重加权的分位数。预测：随着参考系变粗，
   no_grasp 召回单调上升、dropped 召回单调下降。这是一个可预注册的强检验。
2. **直接检验污染机制**：用「只由该任务的 **成功** episode 标定」的 per_task 阈值
   （train-free，只用 outcome，属于允许的选择方式）重建 12 个 per_task 头。
   如果 §3.2 的机制正确，抓取盲区应当消失，且 12/12 的符号模式应当被抵消。
3. **修正低 FP 端的 bundle 搜索目标**：把贪心目标从 dev TP 换成 dev 低先验 TP，
   并加 external-FP 稳健性约束（例如按 dev 上 FP 的任务分散度惩罚），
   去正面挑战 288 TP / 13 FP。
4. **扩大小格**：`no_contact` 与 `regressed` 是 task 分层下信号最强的两个模式
   （SEL 1.1–1.8），但各只有 14 / 26 例。再跑一个 run 或把物理标注扩到 `cache`
   下的其它 run，可以把这两格推过 30 例门槛，才能判断 v7 periodicity / turbulence
   是否真是「接触缺失头」。
5. **重新检查 v7 guard 的 `moved_unmet` 缺口**（唯一 dev→ext 复现的 v7 缺口，
   SEL −0.31 / −0.18），看是否存在一个能补它的 global 头 —— 这将是第二个可命名的头。

---

## 附：产物

- `results/PREREG.md` — 目标、约束、先验暴露声明（在任何头搜索之前写下）
- `results/join_audit.json`、`results/joined_*.csv` — 连接复核
- `results/mode_recall.csv` — 全部 (队列 × 群体 × 检测器 × 模式) 单元，含分层期望、
  置换 p、BH、MDE、within-task 份额
- `results/replication.{csv,json}` — dev→ext 两阶段复现
- `results/confound_structure.json`、`results/alarm_timing_by_mode.csv` — 混淆结构与时序
- `results/interpretable_heads.json` — 12/12 配对检验、污染机制、可命名头
- `results/multihead.json`、`results/frontier_points.csv`、`results/frontier.json` — Q4
- `results/bundle_mode_gain.{csv,json}` — 多头额外收益的模式分解
- `results/partner_control.{csv,json}` — 标签预测 / outcome 检验
- `results/headline_operating_points.csv` — 统一口径的全部工作点
- `results/manifest.json` — 输入 / 脚本 / 输出的 sha256
- `run.sh` — 端到端确定性重跑（纯 CPU）
