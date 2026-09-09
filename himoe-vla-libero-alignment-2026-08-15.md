# HiMoE-VLA Section 4.1（LIBERO）对齐报告

日期：2026-08-15
论文：[HiMoE-VLA, arXiv:2512.05693v2](https://arxiv.org/abs/2512.05693)（2026-07-08）
本报告取代 [`himoe-vla-section4.1-reproduction-guide-2026-08-04.md`](himoe-vla-section4.1-reproduction-guide-2026-08-04.md)
中的结果表；该文的操作步骤仍然有效，但 `--libero-wrist-layout` 的推荐值已变。

---

## 一、结论

在**符合论文输入契约**的配置下重跑四个 suite，**Goal 与 Object 与论文统计相容，Spatial 与 Long 不相容**。

| suite | 论文 | 本次 `paper-right` | Wilson 95% | 判定 | 旧配置 | Δ |
|---|---:|---:|---|:--:|---:|---:|
| Goal | 98.6% | **98.0%** (490/500) | [96.36, 98.91] | ✅ 相容 | 97.8% | +1 |
| Object | 99.4% | **99.0%** (495/500) | [97.68, 99.57] | ✅ 相容 | 96.6% | **+12** |
| Spatial | 98.2% | **95.4%** (477/500) | [93.19, 96.92] | ❌ | 94.2% | +6 |
| Long | 95.8% | **92.4%** (462/500) | [89.74, 94.41] | ❌ | 87.0% | **+27** |

判据与 CALVIN 主行一致：**论文点估计是否落在复跑的 Wilson 95% 区间内**。

复现层次仍是**公开 checkpoint 的评测复现**，不是训练复现。

---

## 二、关键修复：wrist 槽位——权重与代码库不自洽

### 代码库的约定是统一的：单腕相机放 left

查遍 ICLR 投稿补充材料（2025-09-25）的四条数据管线，**全部把主 wrist 相机放进
`left_wrist_0_rgb`**，单相机场景把 `right_wrist_0_rgb` 置零：

| 管线 | 主 wrist → | 文件 |
|---|---|---|
| LIBERO | **left** | `policies/libero_policy.py` |
| CALVIN | **left** | `policies/calvin_policy.py` |
| OXE（预训练） | **left**（第二腕 → right） | `policies/oxe_policy.py` |
| ALOHA | **left**（第二腕 → right） | `policies/aloha_policy.py` |

发布版与投稿版在这一点上**逐字相同**（`libero_policy.py` 的完整 diff 只有模块改名
`openpi`→`moevla`、以及 state/action padding 从 `LiberoInputs` 移到模型 transforms）。
所以这不是发布时的回归，是贯穿投稿与发布、预训练与微调的统一约定。

### 论文正文说的是相反的

> "In single-arm demonstrations, the available arm is mapped to the right-arm
> channel, while the left-arm channel is zero-padded with masks."
> —— example.tex:591（v1、v2 两版一致）

**论文正文才是孤例。**

### 实测：CALVIN 权重与代码自洽，LIBERO 四个不自洽

| 基准 | wrist 槽位 | 我们 | 论文 | 判定 |
|---|---|---:|---:|:--:|
| CALVIN | **left**（代码约定） | **4.008** | 3.98 | ✅ 相容 |
| LIBERO | **left**（代码约定） | ~80% | ~98% | ❌ 差约 18 pp |
| LIBERO | **right**（违反代码） | 95–99% | ~98% | ✅ 两套相容 |

同一套代码、同一个约定：CALVIN 的 checkpoint 按代码原样跑就复现论文；LIBERO 的四个
必须**反着代码来**才复现。

**因此准确的表述不是「我们按论文修正了代码的错误」，而是：发布的四个 LIBERO
checkpoint 的行为与产生它们的代码库不一致——它们表现得像是用 right 槽位训练的，
而代码库处处用 left。**

承重的是两条已确立的事实，都不依赖任何进一步实验：

1. **LIBERO 四个权重与代码约定冲突**：4 套件 × 500 局，left ~80% vs right 95–99%，
   差 15–19 pp，配对 McNemar `p=0.0386`。
2. **代码的 left 约定本身有效**：CALVIN checkpoint 按该约定原样评测即复现论文
   （4.008 vs 3.98）。若 left 普遍失效，这一条不可能成立。

因此异常**局限于 LIBERO 的四个权重**。

> **附注（2026-08-15，CALVIN 槽位反向验证）**：另跑了 CALVIN + `paper-right` 与
> 既有 1000 序列跑做配对比较，用以检验「CALVIN 权重*偏好* left」这一更强的推断。
> 83 条配对时 Δ = −0.048（前 20 条曾为 −0.450，34 条 −0.088，53 条 +0.113，
> 在零附近震荡）。**该推断不成立：CALVIN 对 wrist 槽位不敏感。**
> 报告中一度出现的「两族权重的约定相反」这一措辞据此撤回；上面两条结论不受影响。

### 三档配置的定义

| 配置 | left 内容 | left mask | 结果 |
|---|---|---|---|
| `released-left` | 真实 wrist 图像 | true | ~80%（代码原样） |
| `checkpoint-right` | 全零 | true | 94.2–97.8% |
| **`paper-right`** | 全零 | **false** | **95.4–99.0%（本报告主表）** |

08-03 选 `checkpoint-right` 时，`paper-right` 只在 20 局上比过（18/20 vs 20/20）。
n=20 在饱和任务上分辨不出任何东西，那次比较不足以支撑该决定。

### 改用 `paper-right` 的效应

| suite | Δ | 主要来源 |
|---|---:|---|
| Long | **+27** | t08「把**两个** moka pot 都放上炉子」14 → 31 |
| Object | **+12** | t06 45 → 50；后五个任务全部 50/50 |
| Spatial | **+6** | t04「两个一模一样的黑碗，靠 in/on 区分」38 → 48 |
| Goal | +1 | 天花板效应，基线十个任务里六个已达 49–50/50 |

**增益集中在需要区分同类物体的任务上**，跨四个 suite 一致。同场景的单物体任务
（Long t02「放**一个** moka pot」）在两种配置下都是 49–50/50。

机理：checkpoint 训练时 left 槽位是 masked-out 的；推理时把 mask 置 true，
等于让模型去 attend 一张训练中从未见过的全黑图，污染的正是这类精细视觉判断。

**这是本轮唯一有论文明文依据的改动。**

---

## 三、逐任务数据

### Goal 490/500

`[50, 49, 50, 45, 50, 50, 48, 50, 48, 50]`（旧配置 `[49, 48, 50, 46, 49, 50, 50, 50, 50, 47]`）

### Object 495/500

`[50, 49, 50, 46, 50, 50, 50, 50, 50, 50]`（旧配置 `[48, 50, 49, 47, 49, 48, 45, 48, 49, 50]`）

### Spatial 477/500

| 任务 | 成绩 | 失败 |
|---|---:|---:|
| t07 `...on the stove...` | 45/50 | −5 |
| t05 `...on the ramekin...` | 45/50 | −5 |
| t08 `...next to the plate...` | 47/50 | −3 |
| t09 `...on the wooden cabinet...` | 47/50 | −3 |
| t03 / t04 / t06 | 48/50 | −2 |
| t02 | 49/50 | −1 |
| t00 / t01 | 50/50 | 0 |

### Long 462/500

`[49, 50, 50, 44, 48, 50, 43, 47, 31, 50]`（旧配置 `[46, 48, 49, 48, 45, 50, 39, 47, 14, 49]`）

---

## 四、残差的形态：弥散，不是聚集

| | wrist 契约错配（已修） | 现在的残差 |
|---|---|---|
| 形态 | **聚集**：Spatial t04 单任务错 12、Long t08 错 36 | **弥散**：Spatial 23 局失败散在 8 个任务，最多 5 局 |
| 成因类型 | 配置错误 | —— |

配置错误产生聚集型失败：某个条件被喂错，触发该条件的任务集体崩溃。
现在剩下的是每任务 2–10% 的均匀失误率，**没有哪个任务是坏的，只是每个都差一点**。
这种形态的典型成因是模型本身不同，不是配置不同。

Spatial 尤其敏感，因为它是最纯粹的语言接地测试：10 个任务动作完全相同
（「拿起黑碗放到盘子上」），只有空间指代不同，且每个场景有两个一模一样的黑碗。

---

## 五、训练步数假说（证据不足，仅存疑）

曾观察到一个相关：论文写 45k 的两套（Goal、Object）在本次均相容，写 35k/40k 的两套
（Spatial、Long）均不相容，且偏离越大差得越多。

**但支撑它的证据在 2026-08-15 读完 ICLR 补充材料后垮了。** `num_train_steps` 这个字段
在三份制品之间取值完全不同：

| 来源 | LIBERO 微调步数 |
|---|---|
| 论文 v1 & v2 | 35k / 40k / 45k / 45k |
| **ICLR 投稿代码**（2025-09） | **80k**（另有 `_bs64` 变体 200k） |
| 发布代码（GitHub） | 50k |

而投稿版 `scripts/serve_policy.py` 的 checkpoint 注册表里，LIBERO 实际只填了
**10k / 20k / 25k** 三个步数（`EnvMode` 声明到 80k 但未填充），**论文声称的
35k/40k/45k 一个都没出现**。

**这个字段显然没有被维护，不能作为「发布权重训了多少步」的证据。**
此前把发布版的 50k 当作过训依据是错的。假说本身未被证伪，但目前**没有任何证据支持它**。

无论如何都无法直接验证：`pytorch_model.pth` 的 zip 只有 8 个条目
（`data.pkl` + 张量存储 + 格式版本），**没有 `global_step`、没有优化器状态、
没有训练配置**。HF 上四个仓库各只有 4 个提交、无分支无 tag，权重自 2025-12-08 起从未更新。

注意 batch size 并**不**矛盾：发布配置的 `batch_size` 是每卡值
（`train.py:200` 的 `total = batch_size × num_processes × grad_accum`，日志打印为
"Instantaneous batch size per device"），8×8 卡 = 论文的 64，spatial 4×8 = 32。

**当前对残差最好的解释仍是 §二**：LIBERO 权重与代码库的输入约定不自洽，说明发布的
权重与论文表格所用的训练状态之间存在未公开的差异。

---

## 六、已排除的解释

| 假设 | 排除依据 |
|---|---|
| 随机种子 | 同 checkpoint 同协议、只换噪声流跑两轮 500 局：goal 490 vs 489、spatial 471 vs 471、object 484 vs 483。**≤1 局 = ±0.2 pp**，比缺口小一个数量级 |
| 按任务分左右手 | 三 suite 30 个任务逐个比 left vs right：left 只在 2 个任务上好 1–2 局（噪声）；right 赢时达 +14/+20/+24/+30/+38 |
| `both` 布局 | 三档单调序已排出：左槽塞真实图像 ~80% < 全零+mask true 94.2% < 全零+mask false 95.4%。左通道越不被使用越好，`both` 方向相反 |
| state 补齐/validity mask | `libero_policy.py:40` 那行注释是冗余的；`PadStatesAndActions` 在模型 transforms 里，推理同样生效，`is_libero_or_oxe=True` 走 `pad_libero_oxe_state_to_dim`，mask 正确置 1 |
| 图像预处理 | 两路均 180° 旋转 + `resize_with_pad`，与训练一致 |
| batch size | 见上，每卡值 × 卡数 = 论文值 |
| 评测协议参数 | 与发布仓库默认逐项一致（trials 50 / wait 10 / resize 224 / seed 7 / horizon 220-280-300-520）|

---

## 七、论文未规定的部分

全文关键词计数：`replan` **0** 次、`seed` **0** 次、`max step` / `step limit` **0** 次、
`evaluation protocol` **0** 次；`trial` 13 次**全部在真机章节**。

论文关于 LIBERO 评测的全部操作性描述只有一句：

> "we perform supervised fine-tuning within each task suite using the successful
> demonstrations and evaluate policies on held-out task episodes."

对比之下，真机评测在附录里把「4 settings × 4 trials/setting = 16 trials」逐条列出。
**LIBERO 的评测协议在论文中完全缺失**，所有操作参数都只能取自发布仓库。

### `replan` 消融（不进对齐表）

发布的 `examples/libero/main.py` 是 openpi 同名文件的副本，**只有一处语义改动：
`replan_steps` 5 → 10**（连注释里 `stabilize i n sim` 的拼写错都相同）。

**2026-08-15 补充材料定案：作者用的就是 10。** ICLR 投稿版 `examples/libero/main.py`
与发布版逐项相同——`replan_steps=10`、`num_trials_per_task=50`、`num_steps_wait=10`、
`resize_size=224`、`seed=7`、horizon 220/280/300/520；两版 diff 只有默认 suite 名与
视频输出路径。所以 `replan=5` **偏离作者实际协议**，这一点现在有硬证据，
不再只是「论文没写所以不敢用」。主表的 replan=10 是正确协议。

| Spatial 配置 | 结果 |
|---|---:|
| checkpoint-right + replan10 | 471/500 = 94.2% |
| paper-right + replan10 | 477/500 = 95.4% |
| paper-right + replan5 | **482/500 = 96.4%**（Wilson [94.38, 97.71]，仍不相容）|

两个改动各值约 +6 局且可叠加，但 `paper-right` 修的是同物消歧（t04 +10），
`replan5` 修的是需要更频繁闭环纠正的任务（t05/t06/t07 各 +2）。

**论文对 replan 只字未提，因此 replan5 只作协议敏感性消融，不能标为论文复现。**
它量化的是：论文未公布的参数可带来约 ±1 pp 的摆动。

Long 的同类消融在跑（见 §九）。

---

## 七之二、ICLR 补充材料带来的三处修正（2026-08-15）

来源：OpenReview forum `TX3oGD99CJ` 的补充材料
`4904_HiMoE_VLA_Hierarchical_Mi_Supplementary Material.zip`（2025-09-25，含完整代码
243 个文件）。本机无法直连 OpenReview（四条路径均被 CAPTCHA 拦截），由用户下载提供。

| # | 此前的说法 | 修正后 |
|---|---|---|
| 1 | 「发布代码把 wrist 放进 left，与论文矛盾」——暗示是发布时的回归 | **投稿版也是 left，且 OXE/CALVIN/ALOHA 四条管线一律 left。是贯穿全代码库的统一约定，论文正文才是孤例**（见 §二） |
| 2 | 「发布权重是 50k 训的，Spatial/Long 过训」 | **`num_train_steps` 三份制品三个值（80k/50k/35-45k），字段未维护，不能作为证据**（见 §五） |
| 3 | 「论文没写 replan，所以 5 只能算消融」 | **投稿代码确认作者用 10，`replan=5` 明确偏离作者协议**（见 §七） |

另外两项经补充材料核实**无需修正**：

- `libero_policy.py` 中 state/action padding 的注释掉是**等价重构**——投稿版在
  `LiberoInputs` 里做 padding 且模型 transforms 无 `PadStatesAndActions`；发布版反之。
  两条路径功能一致，与 08-15 早先的代码级核实吻合。
- 评测协议参数（trials/horizon/wait/resize/seed）投稿版与我们所用完全一致。

补充材料中 `examples/libero/README.md` 是 openpi 原样残留（结果表列的是 π₀ 与 π₀-FAST
的数字，命令是 `serve_policy.py --env LIBERO`）。其中 π₀ 行
（96.8 / 98.8 / 95.8 / 85.2 / 94.15）与论文 Table 2 的 π₀ 基线**逐位相同**，
说明论文的 π₀ 基线取自 openpi README，非自行复现。

## 八、给作者的问题

1. **单臂 wrist 槽位**（最重要）：代码库四条管线（LIBERO/CALVIN/OXE/ALOHA）在投稿版与
   发布版中一律把主 wrist 相机放 `left_wrist_0_rgb`，而论文 example.tex:591 写的是
   right。实测发布的 LIBERO 权重在 right 下高出 15–19 pp，CALVIN 权重则在 left 下
   正常复现。**这四个 LIBERO checkpoint 是用什么输入约定训练的？**
2. LIBERO 四个 suite 各自对应 Table 2 的 **checkpoint step**。论文（v1/v2）写
   35k/40k/45k/45k，投稿代码写 80k，发布代码写 50k，投稿版 `serve_policy.py` 的注册表
   只填到 25k——四者互不相同，且发布权重不含 `global_step`。
3. 能否提供 Spatial / Long 对应论文表格的中间 checkpoint，用同一批固定 episode 做配对评测。
4. LIBERO **Long** 的数字为何在 v1→v2 之间从 94.8 改为 95.8（其余三套逐位未变），
   依据是什么。
5. 生成这些 checkpoint 的 git commit 与全部 submodule commit。

（第 3 项原为「LIBERO 评测协议」，已由补充材料回答：replan=10 / trials=50 / wait=10 /
resize=224 / seed=7 / horizon 220-280-300-520，与我们所用一致，故删除。）

---

## 九、复现方法

```bash
cd /home/jovyan/work/himoe-vla/himoe-libero-wrist-fix
# 用法: run_libero_eval.sh <suite> <layout> <trials> <tag> [mig-uuid] [port] [replan]
bash scripts/run_libero_eval.sh object paper-right 50 my-run \
     MIG-ed0ef408-41bd-59d1-8006-30afa5a1d96a 8071 10
```

驱动脚本已固化四个坑：代理绕过（websockets ≥14 会把 127.0.0.1 也塞进
`HTTP_PROXY`）、握手 stderr 不吞、PID 文件关停（禁用 `pkill -f`，它会自匹配）、
metadata 强校验（benchmark / layout / checkpoint sha256）。

**必须显式传 `paper-right`。** bridge 默认仍是 `released-left`（冻结的旧基线），
漏传不会报错，但会静默回到约 80%。

### 运行环境

模型服务器只能用两个 32 GB MIG 分片（`2g.35gb` / `1g.35gb`）；MIG 模式会关闭整卡
图形引擎，因此 EGL 渲染必须借 GPU 0（客户端 `CUDA_VISIBLE_DEVICES=` 为空，
只用图形管线，不占其 CUDA）。详见项目记忆 `gpu-only-two-32g-mig`。

### 产物

```
himoe-vla-cache/himoe-libero-bridge/paper-alignment/
├── paperright-goal-20260815T095349Z/     490/500
├── paperright-object-20260815T082509Z/   495/500
├── paperright-spatial-20260815T040911Z/  477/500
├── paperright-long-20260815T040911Z/     462/500
└── pr-replan5-spatial-20260815T060251Z/  482/500（消融）
```

每个目录含 `eval.log`（附 SHA-256）、`server.log`、`server_metadata.json`、`videos/`。

### 已终止的运行

`pr-replan5-long-20260815T110329Z`（Long + replan5）在 152/500 处终止。终止原因：
ICLR 补充材料证明作者用的是 replan=10，该消融已无信息价值（见 §七之二）。
终止时前两个任务为 43/50 与 48/50，相对 replan=10 的 49 与 50 为 −3 与 −2，
方向与 Spatial 上的 +5 相反；此为不完整数据，仅作记录，不得引用。

---

## 十、与既往文档的关系

- [`himoe-repro-report.md`](himoe-repro-report.md)（08-02）：主结论已被 wrist 修复证伪，
  文件顶部已加更正横幅。
- [`himoe-vla-reproduction-status-2026-08-03.md`](himoe-vla-reproduction-status-2026-08-03.md)：
  wrist 修复前的冻结基线与取证过程，顶部已有更正横幅。
- [`himoe-vla-section4.1-alignment-2026-08-03.md`](himoe-vla-section4.1-alignment-2026-08-03.md)：
  wrist 错配的发现过程与统计证据，仍然有效；但其推荐的 `checkpoint-right` 已被本报告取代。
- [`himoe-vla-section4.1-reproduction-guide-2026-08-04.md`](himoe-vla-section4.1-reproduction-guide-2026-08-04.md)：
  操作步骤有效，**结果表与 `--libero-wrist-layout` 推荐值以本报告为准**；
  其中「Long 因完整 checkpoint 不可用而未复现」一句已过时——权重已于 2026-08-14 下载并校验
  （sha256 `cdc2b21f…`，与发布的 LFS 指针逐字节吻合）。
