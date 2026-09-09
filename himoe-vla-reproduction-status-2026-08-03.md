# HiMoE-VLA 当前复现状态、权重取证与完整复现路线

> **2026-08-03 对齐更新：** 本报告记录的是 wrist 契约修复前的冻结基线和取证过程。
> 后续已经确认公开 `LiberoInputs` 与论文的单臂 right-wrist 契约冲突，并完成修复与三套正式复评。
> 当前结论、结果和运行命令请以
> [`himoe-vla-section4.1-alignment-2026-08-03.md`](himoe-vla-section4.1-alignment-2026-08-03.md)
> 为准；本文件保留作为前序证据，不能再单独引用其“原因尚未确定”的阶段性结论。

报告日期：2026-08-03  
报告范围：公开代码与 checkpoint、HF lineage、运行时源码、LIBERO 评测、精确重放、对照实验、RAD 与训练复现路线  
论文版本：[HiMoE-VLA, arXiv:2512.05693v2](https://arxiv.org/html/2512.05693v2)，2026-07-08 修订  
本地上游代码：`27a2c46932d8b6373ca0074eb997f299bcd4f6f5`  
本地桥接代码：`f5dc26bbf4d878ad6621c7be85335eb61f494dc0`

---

## 1. 执行摘要

当前已经完成的是公开制品评测审计，即使用作者公开的代码与 Goal、Spatial、Object 三个
checkpoint，在对齐论文依赖和 LIBERO 协议后重新执行评测。当前还不是训练复现，因为训练数据、
论文表格所对应的 checkpoint step 和完整训练制品 lineage 尚不充分公开。

**结论先说清楚：权重文件发生下载损坏的可能性很低；当前优先级最高的工作假说是，公开权重虽然是
有效模型权重，但现有公开元数据无法证明它们就是论文 Table 1(b) 使用的最终 checkpoint。** 紧随其后的
两项风险是权重与训练时推理代码版本未绑定，以及 LIBERO 数据转换/normalization 契约未完全公开。

核心结论如下：

1. Goal、Spatial、Object 三个公开 checkpoint 的论文依赖 pin 版 `10 tasks x 50 episodes`
   评测均已完成，分别得到 `82.0%`、`81.2%`、`79.2%`。
2. 论文 v2 Table 1(b) 对应数字为 `98.6%`、`98.2%`、`99.4%`，本地结果低
   `16.6`、`17.0`、`20.2` 个百分点。
3. OpenVLA Goal 阳性对照得到 `39/50 = 78.0%`，与论文引用的 `79.2%` 只差
   `1.2` 个百分点。它不能单独证明环境绝对无误，但显著降低了“通用 LIBERO 环境整体错误”的可能性。
4. 默认环境的完整 Goal 评测为 `415/500 = 83.0%`；论文依赖 pin 版为
   `410/500 = 82.0%`。依赖与环境调整只改变 1 个百分点，无法解释 16.6 个百分点的主差距。
5. Goal 精确重放验收已经完成：换用全新模型 server 后，模型动作、MuJoCo 状态、原始渲染帧和
   MP4 字节完全一致。
6. Spatial task 4 的 8 个失败初始状态把 horizon 从 220 提升到 1000 后仍全部失败，
   每个 episode 都执行 1000 步和 100 次模型推理。失败并非单纯预算不足。
7. Goal RAD 50 个配对案例已经完成。预注册主比较 `routing_medoid - random` 为
   `+2` 个百分点，task-cluster 95% 区间 `[-8, +12]`，McNemar `p=1.0`；
   当前没有统计证据支持 routing medoid 提升闭环成功率。
8. 冻结基准的 `master` 仍 clean，`make test` 为 `138 passed`；新建的独立审计分支
   `audit/data-contract-v1` 已提交为 `9e04e85`，其测试为 `148 passed in 2.70s`。
9. HF 历史已核验：四个 LIBERO 模型文件在 `main` 上仍由 2025-12-08 的最初制品提交提供，
   normalization 目录也各只有一个历史提交；三个已下载权重的远端 SHA 与本地完全一致。因此
   “后续 revision 偷换权重或 stats”已降为低优先级。
10. 三个本地 checkpoint 均是只含 1581 个 tensor 的 `OrderedDict`，没有 `global_step`、`epoch`、
    optimizer 或 scheduler 元数据，无法从文件本身证明对应论文的 35k/45k step。
11. 已计算实际运行时导入的 Gemma、LeRobot、HiMoE model/policy/transform 文件 SHA；当前安装的
    `modeling_gemma.py` 与 vendored 文件一致，LeRobot 运行时树与 vendored 树一致。
12. 发现一个尚未解释的论文—代码契约差异：论文写单臂输入映射到 right-arm channel；公开
    `LiberoInputs` 把真实 wrist 图像放入 `left_wrist_0_rgb`，并把 `right_wrist_0_rgb` 置零和 mask false。
    这可能只是命名差异，也可能是 checkpoint 训练/推理版本错配，必须用固定输入 paired A/B 验证。
13. 仅 1.9 GB 的不可替代研究产物已迁入持久化 `work`。约 97 GB 的 checkpoint、环境和包 cache
   仍在系统盘，重启后仍可能丢失。
14. `released-checkpoint baseline v1` 已用 annotated Git tag 固定到 bridge `f5dc26b`；Goal 82.0%、
    Spatial 81.2%、Object 79.2% 的协议、checkpoint/stats SHA 和证据日志 SHA 已写入持久化 manifest。
15. 新增的 Goal 合成数据契约审计已经实际运行：train/eval 公共输入、resize 图像、token IDs、state/action
    padding 全部 exact；7D action 的 normalize → 24D pad → unpad → unnormalize 最大误差为
    `2.78e-17`，17 个无效 action 维度全部为零。
16. 阶段 1 尚未通过。公开 transform 把 8 个 state 值写入 model state，却只设置 7 个 validity bit；真实
    wrist 仍位于 left slot，而论文描述 single-arm 映射到 right slot。这两点当前只能标记为
    `needs_review`，不能直接宣布为 bug。
17. robosuite 1.4.1 + MuJoCo 3.2.3 的控制器方向探针已通过：平移和旋转轴/符号均为 `6/6`；在测试姿态
    下响应与 world/base 轴对齐；夹爪语义为 `-1=open`、`+1=close`。该检查使用 OSMesa 且不比较图像，
    不会替代或改写冻结的 EGL 基准。
18. 当前机器没有 LIBERO demonstrations，`LIBERO/datasets` 目录也不存在；真实 100-sample train/eval
    对比、demonstration replay、teacher-forcing、一步状态对比和 normalization 重算均被同一数据缺口阻塞。

当前最稳妥的判断是：已观察到的性能缺口被强烈收窄到 HiMoE 发布 checkpoint、其精确训练 step、
代码版本绑定或未公开数据处理契约一侧。现有证据还不足以断言“论文错误”或“checkpoint 一定错误”。
执行策略已经正式切换为：**冻结基线 → 数据契约审计 → demonstration/offline 定位 → 80-case paired
协议筛选 → Goal 微调重建**。Long、Spatial/Object exact、OpenVLA 500、RAD 扩展和新的环境扫描暂缓。

---

## 2. 复现范围和状态矩阵

| 工作项 | 当前状态 | 完成标准 / 结果 |
|---|---|---|
| 上游代码固定 | 已完成 | HiMoE `27a2c46`，LIBERO `8f1084e` |
| 模型与 LIBERO 双环境 | 已完成 | 模型 Python 3.11；LIBERO Python 3.8.20 |
| Goal checkpoint | 已完成 | 文件存在，8,138,322,389 bytes，SHA-256 已校验 |
| Spatial checkpoint | 已完成 | 文件存在，8,138,322,389 bytes，SHA-256 已校验 |
| Object checkpoint | 已完成 | 文件存在，8,138,322,389 bytes，SHA-256 已校验 |
| HF revision / LFS lineage | 已完成 | 四套原始 revision 已固定，main 未改模型与 normalization 历史 |
| Checkpoint 顶层元数据 | 已完成 | 三套均为 1581 tensor `OrderedDict`，无 global step |
| 实际运行时源码哈希 | 已完成 | Gemma、LeRobot、HiMoE model/policy/transform 已落盘 |
| released-checkpoint baseline v1 | **已冻结** | tag → `f5dc26b`；协议、三套结果与证据 SHA 已写入 manifest |
| 独立数据契约分支 | **已完成首版** | `audit/data-contract-v1` @ `9e04e85`，148 tests passed |
| Long checkpoint | 已暂缓 | 远端 SHA 已知；按新主线不下载、不运行 Long 10x50 |
| Goal 正式评测 | 已完成 | 410/500，82.0%，论文 pin 环境 |
| Spatial 正式评测 | 已完成 | 406/500，81.2%，论文 pin 环境 |
| Object 正式评测 | 已完成 | 396/500，79.2%，论文 pin 环境 |
| OpenVLA Goal 阳性对照 | 部分完成 | 39/50，78.0%；尚未扩展到 500 episodes |
| Goal fresh-server exact replay | 已完成 | 动作、状态、帧、MP4 全部 exact |
| Spatial fresh-server exact replay | 已暂缓 | 完整 suite 已评测；不再作为当前定位主线 |
| Object fresh-server exact replay | 已暂缓 | 完整 suite 已评测；不再作为当前定位主线 |
| 超时假说验证 | 已完成 | Spatial task 4 的 8/8 失败在 1000 步仍失败 |
| top-k=8 对照 | 已完成 | 41/50，82%；相对 K=4 的 40/50 仅多 1 局 |
| RAD Goal 10x5 四臂实验 | 已完成 | 50/50 paired cases，有完整 aggregate statistics |
| 合成数据契约黄金指纹 | **已完成** | 图像、token、state/action/mask exact；NPZ 与 JSON 已持久化 |
| 真实 demonstration 黄金指纹 | 阻塞 | 本机无 LIBERO demonstrations；不能用 rollout trace 替代 |
| wrist left/right channel paired A/B | 未完成 | 已发现论文—代码差异，待固定 20–40 episode panel |
| state validity 8-vs-7 审计 | **needs_review** | state 有 8 个值，公开 mask 只有前 7 位 true |
| action round-trip | **已通过** | 最大绝对误差 2.78e-17；padding 维度全部为零 |
| 控制器方向与夹爪语义 | **已通过** | xyz/rxyz 12/12；`-1=open`、`+1=close` |
| demonstration replay / offline action | 阻塞 | 缺少 demonstrations |
| 80-episode 诊断集 | 已冻结 / 未运行 | 60 hard + 20 easy；M1/M2 未闭合前禁止运行 |
| normalization 重算 | 阻塞 | HF 只发布 meta/stats；当前没有源 demonstrations |
| 训练复现 | 严格复现受阻 / 独立实现可行 | 论文数据 lineage 不完整，且完整训练计算需求远超当前环境 |

复现层次说明：

- **A. 评测复现**：使用作者公开 fine-tuned checkpoint 复现论文结果。运行链路已完成，Goal、Spatial、
  Object 已完成 500-episode 评测，但论文数字未复现；Long 尚缺。
- **B. 微调复现**：从作者 Base checkpoint 和 LIBERO demonstrations 重新训练四套模型。这是 A 的取证
  到极限后的下一阶段；必须先闭合数据转换、normalization 和训练/推理 transform 契约。
- **C. 从零完整训练**：按论文在 OXE 与公开 ALOHA 混合数据上预训练 100k steps，再做 LIBERO 微调。
  论文报告约 4B 参数、16 张 A100 40GB、全局 batch 256、约 4 天。当前 dataset、multi-dataset sampler
  和预训练代码未完整发布，因此可以独立重实现，但不能声称与论文训练逐制品一致。

当前不再为了“补齐 A 的形式完整性”优先运行 Long、Spatial/Object exact 或 OpenVLA 500。先闭合 A/B
共同依赖的数据契约与 demonstration replay；若公开 checkpoint 的离线动作仍差，直接进入 Goal 微调重建。
只有 Goal 能稳定恢复后才扩展 Spatial、Object、Long；只有训练数据契约充分闭合后才考虑 C。

---

## 3. 论文参考值

论文 v2 的 LIBERO 主表为 Table 1(b)：

| 模型 | Spatial | Object | Goal | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| HiMoE-VLA | 98.2 | 99.4 | 98.6 | 95.8 | 98.0 |

论文同时说明部署配置为 `N=32, top-K=4`。附录给出的 LIBERO fine-tuning schedule 是：

| Suite | 论文 batch size | 论文训练步数 |
|---|---:|---:|
| Long | 64 | 40k |
| Goal | 64 | 45k |
| Object | 64 | 45k |
| Spatial | 32 | 35k |

本地固定的官方代码在四个 LIBERO `TrainConfig` 中统一写为 `num_train_steps=50_000`。
公开 checkpoint README 和文件元数据没有标明实际 `global_step`，因此当前无法证明下载到的文件就是
论文 Table 1(b) 使用的 35k/40k/45k 权重。

---

## 4. 主评测结果

### 4.1 论文 pin 环境主结果

| 模型 / suite | 成功数 | Episodes | 成功率 | Wilson 95% CI | 论文值 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| OpenVLA Goal 阳性对照 | 39 | 50 | 78.0% | [64.8%, 87.2%] | 79.2% | -1.2 pp |
| HiMoE-VLA Goal | 410 | 500 | 82.0% | [78.4%, 85.1%] | 98.6% | -16.6 pp |
| HiMoE-VLA Spatial | 406 | 500 | 81.2% | [77.5%, 84.4%] | 98.2% | -17.0 pp |
| HiMoE-VLA Object | 396 | 500 | 79.2% | [75.4%, 82.5%] | 99.4% | -20.2 pp |

三套已评测 HiMoE suite 的宏平均为 `80.8%`；论文对应三项宏平均约为 `98.7%`，
平均差约 `17.9` 个百分点。

主证据：

- [Goal 论文 pin eval.log](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-rs141-egl-paperpins-20260801T1737Z/eval.log)
- [Spatial 论文 pin eval.log](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-spatial-rs141-egl-paperpins-20260802T0908Z/eval.log)
- [Object 论文 pin eval.log](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-object-rs141-egl-paperpins-20260802T1134Z/eval.log)
- [OpenVLA Goal 50-episode 输出](himoe-env-ab/openvla-control-goal-20260802T1232Z/EVAL-libero_goal-openvla-2026_08_02-13_06_30.txt)

### 4.2 Goal 两套完整评测的区别

| Goal 运行 | 成功数 | Episodes | 成功率 | 用途 |
|---|---:|---:|---:|---|
| 早期默认环境完整运行 | 415 | 500 | 83.0% | 环境敏感性参考 |
| Python 3.8.20 + robosuite 1.4.1 + 论文依赖 pin | 410 | 500 | 82.0% | 当前主审计数字 |

早期目录 `official-main-libero-goal-...T1137Z` 是未完成启动，不含最终 `Total` 行，不纳入任何统计。
完整的早期默认运行是：
[Goal 83.0% eval.log](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-goal-10x50-seed7-20260801T1143Z/eval.log)。

这两次运行相差 5/500，仅 1 个百分点。因为它们不是只改变单一变量的严格 paired A/B，
不应把 1 个百分点解释成某一个依赖或渲染后端的精确因果效应；但它足以说明论文 pin 不会把结果
从约 82% 推升到约 99%。

### 4.3 各任务成功率

以下均为论文 pin 环境，每个单元格是 50 episodes 成功率：

| Task ID | Goal | Spatial | Object |
|---:|---:|---:|---:|
| 0 | 22% | 86% | 88% |
| 1 | 96% | 96% | 72% |
| 2 | 96% | 94% | 92% |
| 3 | 80% | 92% | 72% |
| 4 | 84% | 28% | 92% |
| 5 | 98% | 86% | 36% |
| 6 | 100% | 90% | 64% |
| 7 | 100% | 86% | 98% |
| 8 | 98% | 94% | 90% |
| 9 | 46% | 60% | 88% |

最显著的弱项：

- Goal task 0，`open the middle drawer of the cabinet`：22%。
- Goal task 9，`put the wine bottle on the rack`：46%。
- Spatial task 4，`pick up the black bowl in the top drawer ...`：28%。
- Spatial task 9，`pick up the black bowl on the wooden cabinet ...`：60%。
- Object task 5，`pick up the tomato sauce and place it in the basket`：36%。
- Object task 6，`pick up the butter and place it in the basket`：64%。

这种分布不是所有任务均匀下降，而是少数任务严重失败、其余任务接近饱和。

---

## 5. 评测协议与运行环境

### 5.1 主协议

| 项目 | 设置 |
|---|---|
| suite 数 | 已完成 Goal / Spatial / Object，Long 未完成 |
| 每个 suite | 10 tasks x 50 episodes = 500 episodes |
| 环境 seed | 7 |
| policy seed | 42，官方 `Policy.__init__` 内设置 |
| settling wait | `num_steps_wait=10` |
| action chunk | 输出 `10 x 7`，每次最多执行 10 步 |
| 内部 action | `10 x 24` flow/action 表示 |
| image resize | 224 x 224 |
| robot state | 8 维 |
| Goal horizon | 300 |
| Spatial horizon | 220 |
| Object horizon | 280 |
| 正式评测脚本 | 官方 `examples/libero/main.py`，主逻辑未修改 |
| 模型服务 | GPU 1，Python 3.11 |
| LIBERO 客户端 | CPU，Python 3.8.20；论文 pin 正式运行使用 EGL |

模型 server 与仿真 client 分离。GPU 0 未用于本项目；模型服务使用物理 GPU 1，记录到的设备为
`NVIDIA H20-3e MIG 4g.71gb`，模型加载后分配显存约 16.3 GB。

正式日志中会出现 `datasets path ... does not exist` 警告。该路径是训练 demonstrations 目录；
评测实际使用的 BDDL、init state 和 assets 已正常加载，500-episode 运行均完成，因此该警告未被计为
基础设施失败。日志中还存在 EGL context 析构阶段的 `EGL_NOT_INITIALIZED` / `libGLU.so.0` 警告，
但它们出现在环境销毁路径，运行继续并产生完整 500-episode 总计。

### 5.2 论文 pin 依赖

| 组件 | 版本 |
|---|---|
| Python | 3.8.20 |
| robosuite | 1.4.1 |
| MuJoCo | 3.2.3 |
| NumPy | 1.22.4 |
| PyOpenGL | 3.1.7 |
| glfw | 1.12.0 |
| numba | 0.53.1 |
| llvmlite | 0.36.0 |
| renderer | EGL，`MUJOCO_EGL_DEVICE_ID=0` |

环境证据：

- [Goal environment.json](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-rs141-egl-paperpins-20260801T1737Z/environment.json)
- [Spatial environment.json](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-spatial-rs141-egl-paperpins-20260802T0908Z/environment.json)
- [Object environment.json](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-object-rs141-egl-paperpins-20260802T1134Z/environment.json)

---

## 6. 代码、checkpoint 与制品身份

### 6.1 Git 状态

截至本次数据契约审计提交后，桥接 `master` 与四个独立 worktree 均为 clean：

| 目录 / 分支 | Commit | 状态 | 作用 |
|---|---|---|---|
| `himoe-libero-bridge` / `master` | `f5dc26b` | clean | 冻结基准；tag `released-checkpoint-baseline-v1` |
| `himoe-libero-data-contract` / `audit/data-contract-v1` | `9e04e85` | clean | 数据契约、控制器探针、80-case manifest |
| `himoe-libero-rad-v2` / `rad-v2-protocol` | `a89e877` | clean | RAD metric / paired protocol |
| `himoe-libero-source-id-fix` | `a5b2884` | clean | batch source identity |
| `himoe-libero-identity-toctou` | `60f2fd6` | clean | identity timing gap 修复 |

上游 HiMoE 工作树固定在 `27a2c46`，因为运行时应用了三份审计 patch，所以预期为 dirty。
LIBERO 固定在 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。

当前单元测试：

```text
PYTHONPATH=src pytest -q
138 passed in 2.66s
```

### 6.2 Checkpoint manifest

三个 `pytorch_model.pth` 文件大小均为 `8,138,322,389` bytes：

| Suite | Checkpoint SHA-256 | Normalization stats SHA-256 |
|---|---|---|
| Goal | `98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953` | `0a900ab12eb7cf3a25a0afb406dfc95dbe04a215b4feed7cd4c1fa5be44acfcf` |
| Spatial | `1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137` | `4b06853e1320af2a05614f14bdc79e441b4f175c9c18970657c80c7ed779d266` |
| Object | `f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa` | `50719783647bee5adb7d936863eff02e0500da6e13867ddbcbfbc551b3a06487` |

此前离线 checkpoint 审计还确认：1581 个张量均可 `strict=True` 加载；没有 NaN/Inf、全零张量或
重复专家；Goal 与 Spatial 并非同源复制。该重型检查本次未重新执行，详细记录保留在
[历史审计报告](himoe-repro-report.md)。

### 6.3 Hugging Face 历史 revision 取证

2026-08-03 已通过 Hugging Face 官方文件页、提交页和目录历史完成无需重复下载的 lineage 审计：

| Suite | 最初制品 commit | HF `main` 模型 SHA-256 | norm 目录历史 | 本地状态 |
|---|---|---|---:|---|
| Goal | [`f957f85`](https://huggingface.co/ZhiyingDu/HiMoE-VLA-Libero-Goal/commit/f957f85f224a14db902a2f813f64bb90f06cbc2c) | `98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953` | 1 commit | 与 HF 一致 |
| Spatial | [`15ddfdd`](https://huggingface.co/ZhiyingDu/HiMoE-VLA-Libero-Spatial/commit/15ddfddd7d48eb1bb503b0a13f67e9b665c199e6) | `1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137` | 1 commit | 与 HF 一致 |
| Object | [`aca4891`](https://huggingface.co/ZhiyingDu/HiMoE-VLA-Libero-Object/commit/aca489193fa990b0fa447b88fa4cd038c8a02da0) | `f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa` | 1 commit | 与 HF 一致 |
| Long / Libero-10 | [`c466da7`](https://huggingface.co/ZhiyingDu/HiMoE-VLA-Libero-10/commit/c466da7e2b33da74370c44946c0273bd5917fd74) | `cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256` | 1 commit | 尚未下载 |

HF 当前 `main` 的 `pytorch_model.pth` 页面仍将模型文件归因到上述最初制品 commit；后续提交是
README/metadata 变更。对应 normalization 目录的 History 均显示只有最初一次提交。因此目前已排除：

- 2025-12-08 之后在 `main` 静默替换模型文件；
- 后续提交替换 normalization 目录；
- 本地三套文件与 HF 当前 LFS/Xet 对象不一致。

这项结果**不能**证明最初上传的制品就是论文 Table 1(b) 使用的 checkpoint，只能证明公开历史中
没有第二个可供评测的权重版本。

机器可读证据：
[checkpoint-lineage-audit-20260803.json](himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-lineage-audit-20260803.json)。

### 6.4 Checkpoint 顶层元数据正式审计

三个本地模型均以 `torch.load(..., mmap=True)` 重新检查：

| Suite | 容器 | 顶层键数 | 顶层值 | step/epoch/optimizer/scheduler |
|---|---|---:|---|---|
| Goal | `OrderedDict` | 1581 | 全部是 tensor | 全部不存在 |
| Spatial | `OrderedDict` | 1581 | 全部是 tensor | 全部不存在 |
| Object | `OrderedDict` | 1581 | 全部是 tensor | 全部不存在 |

所以这些文件是纯 state dict，不是完整 trainer checkpoint。`strict=True` 只能证明张量名称和形状与
当前模型结构兼容，不能证明训练 step、optimizer/scheduler 状态、训练代码语义或数据版本一致。

### 6.5 审计 patch

| Patch | SHA-256 | 作用 |
|---|---|---|
| `himoe-fixed-noise.patch` | `d4275c5908353c6a59c178eb25dd53861455892effa41c63202c4e09239dba56` | 允许显式 flow noise，支持 exact replay |
| `himoe-runtime.patch` | `6a37331c4a09402ee4f1f323055e90c51290584148732cb272a584e1dec05428` | 运行时适配 |
| `himoe-transformers-cache.patch` | `4972e8ae67cb5ee696fb28811b8f70f234966e013a8c5ec664d70e20bb607463` | 修复 transformers Cache 导入 |

三份 patch 的 hash 被写入每个 server/episode 元数据。它们没有改变正式评测的任务定义、
checkpoint 张量或归一化资产。

### 6.6 实际运行时源文件指纹

仅记录 Git commit 不足以覆盖手工复制到 site-packages 的代码。以下哈希来自模型环境中实际导入的文件：

| Import | 实际文件 SHA-256 |
|---|---|
| `transformers.models.gemma.modeling_gemma` | `c46f52377f247cd21433bed754a9c4441f36005242c099ea4bd93522e010596b` |
| `lerobot` (`__init__.py`) | `76783a1a95526920dbbbb350076f6e36e67bbc2d04a51b08969ddacfb4a50140` |
| `moevla.models.model` | `8b7bc2915aa98956b741cace75970fa5bf63bc8d7d2fba4323cc2cf9131d42d6` |
| `moevla.models.moevla` | `2603a33766837c77171004fd8c329b7e7c33d12490ac84cbe8769eaf83a88a31` |
| `moevla.policies.libero_policy` | `b9b5998891e2f2e3b50f93d3221a12987d392987fb7c808cff6f83ff46c09ecc` |
| `moevla.policies.policy` | `2939685561c8a64d94a6cc706b485a3b5c58d8b73a82de985e29ca7fa305ee71` |
| `moevla.policies.policy_config` | `186773ddd4bc8e96a9532d9e47a3930ffbd63166691edff221abaed89c630d6c` |
| `moevla.training.config` | `db8a52eed79428921dca8bfa4709a7670634a36ef61edf93efcd5c790c54c276` |
| `moevla.transforms` | `6ded6c9f261da6c24b2f7eccafb8dfd8aa277af3781b31fe60dec7f3f9ab65ab` |

运行环境为 Python `3.11.15`、PyTorch `2.6.0`、CUDA runtime `12.4`、Transformers `4.48.1`、
Accelerate `1.5.2`、LeRobot distribution `0.1.0`。逐文件比较确认：

- site-packages 中的 `modeling_gemma.py` 与上游 `third_party/modeling_gemma.py` 字节一致；
- site-packages 中的 LeRobot 树与上游 vendored 树一致（忽略 `__pycache__` / `.pyc`）；
- 本地上游 clone 目前只有一个可见 Git commit，正式做版本二分前需要获取完整远端历史。

机器可读证据：
[runtime-source-audit-20260803.json](himoe-vla-cache/himoe-libero-bridge/metadata/runtime-source-audit-20260803.json)。

### 6.7 新发现：论文与公开 LIBERO wrist channel 的契约差异

论文 v2 Appendix B 表述：单臂数据映射到 right-arm channel，left-arm channel zero-pad 并 mask。
公开 commit `27a2c46` 的 `src/moevla/policies/libero_policy.py` 实际执行：

```python
"left_wrist_0_rgb": wrist_image,
"right_wrist_0_rgb": np.zeros_like(base_image),
```

对应 mask 是 left `True`、right `False`。评测与公开代码的训练 transform 都走该类，因此这不能直接证明
当前推理错误；但如果论文 checkpoint 来自另一版 transform，左右 wrist token 位置可能发生静默错配。
它比继续调整 MuJoCo 小版本或 horizon 更值得优先做 paired A/B。A/B 必须保持 checkpoint、输入帧、
prompt、flow noise、init state 和所有其余代码不变，只交换 wrist slot 与 mask。

---

## 7. 精确重放与推理链路验收

Goal fresh-server exact acceptance 已完成：

| 检查项 | 结果 |
|---|---|
| source episode | 成功，74 action steps，8 inference calls |
| replay episode | 成功，74 action steps，8 inference calls |
| server 是否重启 | 是，source/replay server instance ID 不同 |
| model action chunks | bitwise exact |
| post-action MuJoCo states | exact |
| raw render frames | exact |
| MP4 bytes / SHA-256 | exact |

证据：
[exact acceptance summary.json](himoe-libero-bridge/artifacts/exact-acceptance-goal-20260731T174918397053967Z/summary.json)。

这项验收排除了以下类别的错误：

- server 重启后随机数状态漂移；
- flow noise 未真正被模型接收；
- action chunk 执行前缀不一致；
- replay 只比较动作、不比较仿真状态或渲染帧；
- 同一 server 内缓存状态导致的伪确定性。

它不能排除 checkpoint 本身不对应论文 step，也不能验证未公开训练数据处理。

Spatial 和 Object 尚未做同级 fresh-server exact gate。它们能提高形式审计完整性，但不能解释主差距，
因此已从当前 P1 暂缓；这不会把它们错误写成“已通过”。

---

## 8. 失败模式与超时实验

正式轨迹审计显示，任务失败主要表现为跑满 suite horizon，而不是程序异常或中途基础设施失败。
成功 episode 通常只使用预算的一部分；失败 episode 往往进入无法恢复的闭环状态。

最强的超时反证来自 Spatial task 4。选择原始 220-step 评测中失败的 8 个 init state：

```text
init_state_id = 0, 2, 5, 6, 7, 9, 10, 11
```

将每个 episode 的 horizon 提到 1000，结果全部为：

```text
status=completed
success=false
action_steps=1000
inference_calls=100
```

证据目录：
[timeout-probe-spatial-task04](himoe-env-ab/timeout-probe-spatial-task04/)。

因此当前可排除“论文用了更长 horizon，所以成功率从 81% 上升到 98%”这一简单解释。
失败更符合策略对语言-视觉指代、铰接物体或容纳关系进入错误吸引状态的表现。

Spatial 是一个较强的内部对照：十个任务的主动作都是把黑碗放到盘子上，区别主要来自碗的初始位置
和语言指代。task 4 的 28% 对应 `in the top drawer`，而较容易的 `between`、`next to`、
`on the stove` 等任务达到 86%-96%。这支持指代/场景绑定问题，但仍属于行为证据，不是模型内部
因果证明。

---

## 9. 已完成的替代解释检查

| 假说 | 实验 / 证据 | 当前判断 |
|---|---|---|
| 通用 LIBERO 环境整体错误 | OpenVLA Goal 39/50=78.0%，论文 79.2% | 不支持该假说 |
| robosuite 1.4.0 vs 1.4.1 是主因 | task 0/9 小样本 3/10 vs 4/10；完整 Goal 默认 83% vs pin 82% | 不足以解释 16.6 pp |
| top-K 应为 8 | K=8 为 41/50，K=4 对应 40/50 | 仅 1 episode 差异 |
| horizon 太短 | 8 个失败 episode 在 1000 步仍 8/8 失败 | 已直接反驳 |
| 推理随机或重启不一致 | Goal fresh-server exact replay 全部 exact | 已反驳 |
| checkpoint 损坏 / NaN / 全零 | 离线张量审计、strict load、三套独立 hash | 未发现损坏 |
| 三套 checkpoint 实际同一个文件 | 三套 SHA 不同；逐张量审计显示大量差异 | 已反驳 |
| LIBERO commit 改了任务资产 | 固定 commit 对比显示差异不涉及 BDDL/init/env 逻辑 | 不支持该假说 |
| EGL 析构警告导致任务失败 | 运行在警告后继续，均产生 500-episode Total | 不支持该假说 |

对于渲染后端，需要保持措辞准确：现有 OSMesa exact replay 和 EGL 完整评测都能稳定工作，
没有看到可解释 17-20 pp 的信号；但目前没有同一批 500 episodes、只切换 EGL/OSMesa 的严格 paired
实验，因此不应声称得到了渲染后端的精确因果效应。

---

## 10. RAD 四臂闭环实验

### 10.1 实验设计

最新完整批次为
[rad-goal-10x5-tv-v4](himoe-vla-cache/himoe-libero-bridge/formal-artifacts/rad-goal-10x5-tv-v4/)，
覆盖 10 个 Goal task、每个 task 前 5 个官方 init state，共 50 个 paired cases。

四个预注册 arm：

| Arm | 候选数 | 选择方式 |
|---|---:|---|
| `k1` | 1 | 直接执行候选 0 |
| `random` | 8 | 独立确定性 RNG 随机选择 |
| `action_medoid` | 8 | checkpoint action-std 归一化 RMS medoid |
| `routing_medoid` | 8 | 对齐 flow/layer/action 的专家分布 Total Variation medoid |

四个 arm 共享 suite、task、init state、环境 seed、checkpoint、归一化和 master noise schedule。
50 个案例均 `fairness_exact=true`，无 pending、incomplete 或 infra failed case，source identity 为 clean
`f5dc26b`。

### 10.2 结果

| Arm | 成功数 | 成功率 | Wilson 95% CI | 平均模型调用 | 平均时长 |
|---|---:|---:|---:|---:|---:|
| `k1` | 39/50 | 78% | [64.8%, 87.2%] | 14.62 | 44.7 s |
| `random` | 37/50 | 74% | [60.4%, 84.1%] | 123.68 | 125.1 s |
| `action_medoid` | 41/50 | 82% | [69.2%, 90.2%] | 112.16 | 114.2 s |
| `routing_medoid` | 38/50 | 76% | [62.6%, 85.7%] | 121.92 | 124.7 s |

预注册主比较：

```text
routing_medoid - random = +0.02
task-cluster bootstrap 95% CI = [-0.08, +0.12]
discordant pairs = 5 vs 4
exact McNemar two-sided p = 1.0
```

次要比较 `action_medoid - random = +0.08`，task-cluster 95% 区间 `[-0.02, +0.18]`，
McNemar `p=0.2891`。它方向上更好，但区间仍跨 0，且是次要比较，不能据此宣称显著收益。

当前 RAD 结论：

- routing medoid 没有显示稳定的闭环成功率优势；
- action-space medoid 值得继续研究，但当前 50-case 证据不足；
- K=8 arm 的计算成本显著高于 K=1，部署收益必须同时覆盖时延和模型调用成本；
- 如要形成论文级结论，应增加 init states、环境 seeds 或 suite，并保持预注册 primary metric 不变。

汇总证据：
[rad-batch-summary.json](himoe-vla-cache/himoe-libero-bridge/formal-artifacts/rad-goal-10x5-tv-v4/rad-batch-summary.json)。

历史批次说明：`rad-goal-10x5-tv-v3` 在 1/50 case 后暂停，不用于最终结论；
`rad-goal-10x1-tv-v2` 是 10-case 早期验证。当前应以 v4 为准。

---

## 11. 当前最可能的缺口来源

按现有证据排序：

| 风险 | 当前判断 | 已有证据 |
|---|---|---|
| 文件下载损坏 | 很低 | HF LFS SHA 与本地一致；strict load、张量健康检查均通过 |
| Goal/Object/Spatial 下载错套件 | 较低 | 三套 SHA、stats 和失败分布不同 |
| 后续 HF revision 偷换模型或 stats | 已基本排除 | 模型与 norm 目录仍追溯到最初制品 commit |
| 公开 checkpoint 不是论文最终 step | **最高优先级** | 论文 35k/40k/45k，公开代码 50k，state dict 无 step |
| 权重对应旧版/未公开推理实现 | **高优先级** | 官方手工覆盖第三方包；无 checkpoint-code 绑定；存在 wrist channel 契约疑点 |
| normalization 与 checkpoint 训练 run 不匹配 | 中高优先级 | stats 文件存在且历史稳定，但没有 run ID/数据 manifest 证明归属 |
| 数据转换和动作编码不同 | **高优先级** | 数据、完整 sampler 与预训练代码未发布，无法逐制品重建 |
| LIBERO 仿真整体错误 | 低优先级 | OpenVLA Goal 阳性对照接近公开值 |
| horizon、top-K、普通推理随机性 | 已基本排除 | 对照和 exact replay 均不能解释 17–20 pp 差距 |

### 11.1 公开 checkpoint 与论文表格 checkpoint 不一致

这是当前优先级最高的解释。论文明确给出 Goal/Object 45k、Spatial 35k、Long 40k；公开代码四套
配置统一为 50k，模型卡又不提供实际 global step。HF 历史审计证明公开历史只有这一套模型对象，
但三个 state dict 自身不含 step。因此“后续换过权重”已排除，“最初公开权重是否是论文权重”仍未解决。
需要作者给出 Table 1(b) 的精确制品 ID、step 和 SHA 才能关闭。

### 11.2 权重与代码 commit / 依赖快照绑定不足

官方仓库没有为每个 checkpoint 提供不可变的代码、submodule、transformers/lerobot 快照绑定。
手工覆盖的模型实现若与权重训练时实现不同，可能造成静默语义偏差，即使 `strict=True` 可以加载。
论文的 right-arm channel 表述与公开 `LiberoInputs` 的 left-wrist 实际映射是当前最具体的待验证差异。

### 11.3 未公开的数据处理契约

固定上游 commit 的 README 仍将“Release the dataset”和“Release the multi-dataset sampler and
pre-training code”标为未完成。当前无法完全核验：

- failure trajectory 的过滤规则；
- gripper 编码与离散/连续转换；
- action padding 与 mask；
- normalization stats 的计算时点；
- suite 数据转换后样本数与 hash；
- checkpoint selection / early stopping / seed selection。

### 11.4 Normalization 文件存在，但训练归属仍未证明

HF 历史已证明每套 normalization 目录自最初上传后未变，本地运行也确实读取对应 suite 的 stats。
这排除了“最近 metadata 更新破坏 stats”，但没有证明这些统计量来自生成该 checkpoint 的同一份转换后数据。
最快的验证不是任意交换 stats 后看偶然成功率，而是先从正式 demonstrations 重算每维 mean/std；若重算无法
闭合，再在固定失败 panel 上做同权重、不同 stats 的 paired 诊断。

### 11.5 论文报告包含未公开的 checkpoint 或 seed 选择

当前无法排除。要验证需拿到 500 个 episode 的明确 ID、所有 policy seeds、checkpoint step 和完整命令。

---

## 12. 不能从当前实验推出的结论

以下说法目前证据不足，不应写入论文结论：

- “论文数字一定错误”；
- “作者发布了错误权重”；
- “MoE 架构本身无效”；
- “环境完全没有任何差异”；
- “RAD routing selector 永远无效”；
- “已经完成 HiMoE-VLA 训练复现”；
- “已经覆盖论文全部 LIBERO 结果”。

当前可以稳健陈述的是：

> 在固定的公开代码 `27a2c46`、已校验的公开 Goal/Spatial/Object checkpoint、论文依赖 pin 和
> 10x50 LIBERO 协议下，实测成功率为 82.0%、81.2%、79.2%，显著低于论文 v2 Table 1(b)；
> OpenVLA Goal 小样本阳性对照与公开值接近。已测试的依赖、top-k、horizon 和推理确定性解释均不能
> 解释主要差距，因此需要作者提供论文表格对应的 checkpoint 与数据 lineage。

---

## 13. 存储与重启风险

### 13.1 已持久化

以下内容已经迁入 `/home/jovyan/work/himoe-vla-cache`，当前约 1.9 GB、至少 4585 个文件：

| 目录 | 内容 |
|---|---|
| `himoe-libero-bridge/formal-artifacts` | batch、RAD traces、视频、合成数据契约指纹和控制器方向报告 |
| `himoe-libero-bridge/paper-alignment` | 三套正式 eval.log、环境与 server 元数据 |
| `himoe-libero-bridge/metadata` | 运行元数据、baseline v1 和阶段 gate 状态 |
| `himoe-libero-bridge/moevla-data` | 辅助数据 |
| `vla-adapter-repro/results` | smoke 结果 |

原 `/home/jovyan/.cache/...` 入口目前是指向这些目录的绝对软链接。迁移前后已用
`rsync -aHcni --delete` 做内容校验。

### 13.2 尚未持久化

| 系统盘 cache | 当前占用 | 重启风险 |
|---|---:|---|
| `himoe-libero-bridge` | 34 GB | checkpoint、env、upstream 可能丢失 |
| `vla-adapter-repro` | 23 GB | env/checkpoint/runtime 可能丢失 |
| `openvla-repro` | 17 GB | 模型与环境可能丢失 |
| `uv` | 23 GB | 包 cache 可能丢失，可重建 |

`work` 挂载总容量只有 8.0 GB，当前已用 6.7 GB，可用 1.4 GB。即使清理全部无关目录，
该卷的 8 GB 上限仍放不下任意一个完整大模型 cache。完整持久化需要至少约 110 GB 可用空间。

重启后必须区分两件事：

1. `work/himoe-vla-cache` 中的数据应继续存在；
2. 系统盘 checkpoint/env 以及位于 `~/.cache` 的软链接入口本身可能消失，需要重建或重新下载。

持久化说明：
[himoe-vla-cache/README.md](himoe-vla-cache/README.md)。

---

## 14. 当前工程使用方式

### 14.1 基础检查

```bash
cd /home/jovyan/work/himoe-libero-bridge

make test
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/himoe-libero doctor
df -h /home/jovyan/work
```

重启后先确认：

```bash
test -x /home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/himoe-libero
test -d /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal
readlink -f /home/jovyan/.cache/himoe-libero-bridge/formal-artifacts
readlink -f /home/jovyan/.cache/himoe-libero-bridge/paper-alignment
```

### 14.2 Goal 模型服务

```bash
cd /home/jovyan/work/himoe-libero-bridge

CUDA_VISIBLE_DEVICES=1 \
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/himoe-libero serve \
  --backend himoe \
  --suite goal \
  --gpu 1 \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoint-dir /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal
```

不要同时启动 Goal、Spatial、Object 三个大模型 server。

### 14.3 单 episode 与 exact acceptance

另一个终端运行单 episode：

```bash
cd /home/jovyan/work/himoe-libero-bridge

CUDA_VISIBLE_DEVICES="" \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
bash scripts/libero.sh run \
  --suite goal \
  --host 127.0.0.1 \
  --port 8000 \
  --task-id 8 \
  --init-state-id 6 \
  --seed 7 \
  --max-steps 300 \
  --replan-steps 10 \
  --flow-noise-seed 42
```

严格 fresh-server 验收：

```bash
HIMOE_GPU=1 \
HIMOE_SUITE=goal \
HIMOE_PORT=8040 \
bash scripts/acceptance_exact.sh 8 6 7 42
```

当前 `work` 只剩 1.4 GB。不要在未检查空间的情况下启动新的大批量全视频实验。

### 14.4 数据契约与控制器审计

审计代码不在冻结基准分支，而在独立 worktree：

```bash
cd /home/jovyan/work/himoe-libero-data-contract

make test
make contract-audit SUITE=goal
make controller-audit SUITE=goal
```

`contract-audit` 默认退出码为 `0=全部通过`、`1=明确失败`、`2=仍有 needs_review/blocked`。当前因为
两项 layout 待判定和 demonstrations 缺失，预期返回 2；只生成证据而不作为 gate 时，按
[DATA_CONTRACT_AUDIT.md](himoe-libero-data-contract/DATA_CONTRACT_AUDIT.md) 中的完整命令添加
`--report-only`。每次执行都会新建 artifact 目录，不覆盖旧报告。

### 14.5 当前允许的操作边界

- 可以运行不加载模型的 tensor/controller 单元测试；
- 可以准备并校验 demonstrations manifest，但在扩容前不要下载整套数据；
- M1/M2 闭合前不运行 80-case panel；
- 当前不要启动 Long 10x50、Spatial/Object exact、OpenVLA 500、RAD 扩展或新的环境扫描；
- 任何新闭环实验都必须从 `audit/data-contract-v1` 或后续独立分支产生新目录。

---

## 15. 新主线：数据契约 → 小规模协议 → Goal 重建

原则：不再把完整 500 episodes 当成排查工具，也不再围绕 MuJoCo 小版本、随机种子、top-K 或 horizon
做无目标搜索。排查顺序固定为 tensor contract、demonstration replay、offline action、一步状态、80-case
paired protocol；只有通过这些关卡的设置才进入正式评测或训练。

| 阶段 | 当前状态 | 退出条件 |
|---|---|---|
| 0. 冻结 released checkpoint baseline | **已完成** | tag、manifest、证据 SHA 全部固定 |
| 1. 数据契约单元测试 | **部分通过** | 真实 demo train/eval 对齐、8-vs-7 mask 与 wrist slot 得到解释 |
| 2. replay / offline 定位 | **控制器通过，其余阻塞** | demo replay >=95%，teacher-forcing 与一步状态测试完成 |
| 3. 80-case 协议筛选 | **manifest 已冻结，未运行** | 找到达到晋级阈值的稳定协议，或证明最佳协议仍约 80% |
| 4. Goal 训练重建 | **未开始** | Goal dev >=95%，冻结 checkpoint 后正式 500 episodes |

### 15.1 阶段 0：released-checkpoint baseline v1 已冻结

annotated tag `released-checkpoint-baseline-v1` 已固定到 bridge `f5dc26b`，不是指向新的审计实现。
独立审计 worktree 为 `/home/jovyan/work/himoe-libero-data-contract`，分支
`audit/data-contract-v1`，commit `9e04e85`。baseline manifest 在分支与持久化 metadata 中字节一致，
记录三套结果、协议和三个正式 eval.log 的 SHA。仍需在大容量持久盘可用后保存 checkpoint、Python env、
Git bundle/submodule，并制作容器或可重建 lockfile。

每个新实验至少保存以下 manifest：

```yaml
run:
  run_id: <UTC-unique-id>
  timestamp_utc: <ISO-8601>
  suite: libero_goal
  task_ids: [0, 9]
  init_state_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  environment_seed: 7
  policy_seed: 42

source:
  himoe_commit: 27a2c46932d8b6373ca0074eb997f299bcd4f6f5
  libero_commit: 8f1084e3132a39270c3a13ebe37270a43ece2a01
  bridge_commit: f5dc26bbf4d878ad6621c7be85335eb61f494dc0
  patch_sha256: [d4275c..., 6a3733..., 4972e8...]
  runtime_source_audit: runtime-source-audit-20260803.json

checkpoint:
  hf_repo: ZhiyingDu/HiMoE-VLA-Libero-Goal
  hf_artifact_revision: f957f85f224a14db902a2f813f64bb90f06cbc2c
  model_sha256: 98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953
  norm_stats_sha256: 0a900ab12eb7cf3a25a0afb406dfc95dbe04a215b4feed7cd4c1fa5be44acfcf
  claimed_global_step: unknown

runtime:
  model_python: 3.11.15
  torch: 2.6.0
  transformers: 4.48.1
  cuda_runtime: 12.4
  libero_python: 3.8.20
  mujoco: 3.2.3
  robosuite: 1.4.1
  renderer: EGL

protocol:
  image_size: [224, 224]
  action_chunk: 10
  action_dim_environment: 7
  internal_action_dim: 24
  settling_steps: 10
  horizon: 300
```

阶段 0 的结果 manifest 已闭合；大体积 runtime 的重启持久化仍受容量阻塞，但不会改变 baseline 身份。

### 15.2 已完成前置取证：Hugging Face 不可变 revision

**状态：已完成，不需要重复下载 Goal/Spatial/Object。** 已证明当前 `main` 的模型和 normalization 仍来自
最初制品 commit，且本地三套模型 SHA 与远端一致。这个结果排除了后续替换，却没有建立 training step。

Long 的远端身份已经固定为：

```text
repo: ZhiyingDu/HiMoE-VLA-Libero-10
revision: c466da7e2b33da74370c44946c0273bd5917fd74
sha256: cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256
bytes: 8138322389
```

Long 当前明确暂缓。未来若在 Goal 主线闭合后下载，必须先校验上述 SHA，不再以可变 `main` 作为身份记录。

### 15.3 已完成前置取证：checkpoint 内部训练身份

**状态：Goal/Spatial/Object 已完成。** 三套都是 1581 tensor 的纯 `OrderedDict`，不存在 `global_step`、
`epoch`、optimizer 或 scheduler。其正式结论是：

> 文件可以被当前结构严格加载，但不能从自身元数据证明它对应 Goal/Object 45k 或 Spatial 35k。

不要把模型卡中的“相应 suite checkpoint”解释成 Table 1 制品的强绑定。

### 15.4 阶段 1：数据契约单元测试与 normalization

已运行的 Goal 合成审计使用当前公开 upstream transform、当前 Goal stats 与持久化 PaliGemma tokenizer，
不加载模型权重。结果如下：

| 检查 | 结果 |
|---|---|
| 合成 train/eval base、left/right wrist | bitwise exact |
| state、token IDs、token mask | bitwise exact |
| 公开 transform 与独立 reference padding | bitwise exact |
| 7D action round-trip | 最大绝对误差 `2.78e-17` |
| 24D action 无效槽位 | 17 维全部为零 |
| model state | 实际是 `48D = 24D values + 24D validity` |
| state validity | 原始 state 为 8D，但 validity true count 为 7，`needs_review` |
| wrist slot | left=真实、right=零且 false，与论文 right-arm 表述不一致，`needs_review` |
| 真实 demonstration 对齐 | blocked：本机没有数据 |

公开实现的真实顺序不是“先 pad 再 normalize”，而是：

```text
raw 8D state / raw 7D action
→ LiberoInputs
→ 在原始维度上 Normalize
→ resize / tokenize
→ state pad 为 24D values + 24D validity，action pad 为 24D
```

输出顺序为 `24D unpad → 7D unnormalize → LiberoOutputs`。这条顺序已经由实际 upstream 类执行并与
reference exact 对齐，后续数据重建必须遵循，不能按自然语言自行交换。

阶段 1 下一步仍然是重算 normalization，而不是跨 suite 盲换：

优先级高于大规模 stats 盲换。顺序如下：

1. 获得或严格重建公开 LIBERO demonstrations 的逐帧 LeRobot 数据。
2. 按训练 transform 重算 state/action 的 mean、std 和分位数。
3. 与 HF `meta/stats.json` 逐维比较 absolute/relative error。
4. 只有无法闭合或出现明确异常时，再进入小规模跨 suite stats A/B。

结果解释：

| 现象 | 优先检查 |
|---|---|
| gripper 维差异最大 | `{-1,+1}` / `{0,1}`、开闭符号、no-op 过滤 |
| 平移/旋转维固定比例差 | 单位、控制频率、delta/absolute 定义 |
| mean 接近但 std 明显不同 | 失败轨迹、no-op、尾帧和 chunk padding 过滤 |
| 全维均明显不同 | 数据源、字段顺序或转换流程不一致 |

若进入交叉诊断，先使用 Goal task 0 和 task 9，每任务固定 10 个 init states，共 20 episodes；固定 env seed 7、
policy seed 42 和相同 flow noise。比较 Goal weight + Goal stats、Goal weight + Spatial stats、Goal weight + Object
stats、Goal weight + 重算 Goal stats。跨 suite stats 应视为负对照，不应因单个偶然成功就升级到 500 局。

### 15.5 阶段 1.1：黄金输入—输出指纹

合成代码级黄金指纹已经完成并保存为 JSON + NPZ，包含源图像、180 度翻转、eval resize、raw/normalized
state、48D model state、raw/normalized/padded/round-trip action、token IDs、token mask 与 data mask。该制品
证明共同 transform 代码可 exact 对齐，但不证明原始训练数据字段语义正确。

第二层黄金指纹要等 demonstrations 到位后，固定一个 demonstration observation、一个闭环成功 episode 和
一个稳定失败 episode，在第一帧继续保存：

- 原始 agentview / wrist RGB 与 resize 后 uint8；
- 输入 VLM 的最终 image tensor、image masks 和 channel keys；
- 原始 8D robot state、Normalize 后 state、24D value slots 与 24D validity mask；
- prompt 的 UTF-8 bytes、最终模板、token IDs 和 token mask；
- 固定 Gaussian flow noise；
- 每个 flow integration step 的 24D action；
- 每层 MoE top-K indices、router probabilities 和最终 7D environment action；
- unnormalization 前后 action 与首个完整 action chunk。

统一使用连续内存的原始 bytes 计算 SHA：

```python
import hashlib
import numpy as np

def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
```

黄金指纹不是只保存最终动作。任何候选代码只要在图像、token、state layout、flow step 或 24D→7D 提取层
首次出现差异，就应在该层停止并解释，不必先跑 500 episodes。

### 15.6 阶段 3 条件分支：代码—权重 paired 二分

当前 site-packages 与公开 commit 的 vendored Gemma/LeRobot 一致，但这只说明安装正确，不说明它与训练权重
版本一致。该工作不再先于 demonstration replay；只有阶段 1/2 表明 checkpoint 在训练分布 observation 上动作
明显错误、且 layout 无法解释时才进入。正式二分前还要先获取官方仓库完整历史；当前本地 clone 只有一个可见 commit。

候选顺序：

1. **wrist slot/mask A/B**：当前 left=真实、right=零，与论文 right-arm 表述做严格交换对照；
2. `third_party/modeling_gemma.py` 历史版本；
3. `third_party/lerobot` 历史版本；
4. `libero_policy.py`、`transforms.py`、`training/config.py`；
5. `model.py` / `moevla.py` 的 flow integration、KV cache 与 action extraction。

每个候选先过黄金指纹，再跑 15.9 中冻结的 80-case paired panel。升级到正式 500 episodes 的门槛：

- 困难集绝对提升至少 15 个百分点；
- 至少 3 个困难任务改善，而非 1–2 个偶然 episode；
- 容易控制集下降不超过 5 个百分点；
- action 尺度和 gripper 行为合理；
- fresh-server exact replay 仍通过。

不满足门槛立即停止，不扩大样本。每次只替换一个文件或一个 channel 契约。

### 15.7 阶段 1.2：闭合真实 LIBERO 数据转换契约

开始微调前必须生成数据 manifest，而不是只记录数据集名字：

```yaml
dataset_manifest:
  suite: libero_goal
  libero_commit: <full-sha>
  source_files: [{path: <path>, sha256: <sha256>}]
  filtering:
    failed_trajectory_rule: <exact-rule>
    noop_rule: <exact-rule>
    noop_threshold: <number>
    frame_counts_before_after: <counts>
  images:
    camera_names: [agentview_image, robot0_eye_in_hand_image]
    source_color_order: RGB
    rotate_180: true
    crop: null
    resize: [224, 224]
    wrist_slot: <left-or-right-model-key>
    missing_view_mask: <exact-mask>
  state:
    source_fields: <ordered-fields>
    raw_dim: 8
    model_dim: 24
    validity_mask: <24-bools>
  action:
    source_fields: <ordered-fields>
    coordinate_frame: <frame>
    delta_or_absolute: <mode>
    gripper_source_target_range: <mapping>
    raw_dim: 7
    model_dim: 24
    chunk_size: 10
    padding_and_loss_mask: <rules>
  timing:
    source_frequency_hz: 10
    action_repeat: <number>
    frame_skip: <number>
```

当前 HF 模型仓库只包含 meta 文件与 stats，没有可用于逐帧重算的 parquet/image 数据；官方 README 仍把
dataset 和 multi-dataset sampler/pretraining code 列为未发布。因此这一阶段目前只能独立重建，不能声称
与论文数据逐字节一致。应向作者请求转换后样本数、源文件 manifest、过滤日志和 stats 生成命令。

### 15.8 阶段 2：demonstration replay 与离线动作定位

控制器方向子 gate 已在 Goal task 0 / init 0 上实际完成。每个 probe 都回到同一官方初始状态，settle 10 步，
随后重复 5 次幅值 0.2 的单轴命令：

| 项目 | 结果 |
|---|---|
| `+/- dx,dy,dz` | 轴与符号 6/6，通过 |
| `+/- rx,ry,rz` | 相对 axis-angle 主轴与符号 6/6，通过 |
| 坐标系 | 测试姿态下与 world/base 轴对齐；单姿态结论，不外推为全局不变量 |
| gripper | `-1` 开度 0.07810，`+1` 开度 0.05059；`-1=open`、`+1=close` |
| runtime | Python 3.8.20、robosuite 1.4.1、MuJoCo 3.2.3、OSMesa |

OSMesa 只用于状态方向探针，不比较图像；冻结的 500-episode EGL baseline 不变。M2 仍未通过，因为同一数据
缺口阻塞了下面三项：

1. 六个严重失败任务各选择 20 条成功 demonstration，从其保存的初始状态直接 replay raw 7D action；
2. 在 demonstration observation 上用固定 flow noise 做 teacher-forcing 离线预测；
3. 从同一时刻状态分别执行真实 action 与模型首个 action，比较下一时刻 EEF、物体和 gripper 状态。

demonstration replay 的判定规则：低于 90% 说明 action/controller/init contract 明显有问题；90%–95% 仍需
检查 gripper、频率与版本；高于 95% 才把重点移到模型/推理。离线至少报告 xyz MSE/方向余弦、rotation error、
gripper sign accuracy、完整 chunk MSE、前 1/2/4/10 步误差、幅值偏差和 saturation 比例。

结果矩阵的解释固定为：

| 离线 action | 闭环 | 主方向 |
|---|---|---|
| 差 | 差 | checkpoint、预处理或训练状态不匹配 |
| 好 | 差 | execute_len、action execution、flow/noise/KV 或闭环控制 |
| 易任务好、难任务差 | 差 | 公开 checkpoint 的任务能力或训练选择问题 |
| 全部好 | 差 | frame、频率、chunk execution 或环境交互 |

### 15.9 阶段 3：冻结的 80-case paired 诊断集

诊断集已经写入
[diagnostic-panel-v1.json](himoe-libero-data-contract/manifests/diagnostic-panel-v1.json)，SHA-256 为
`f9ae811156e207d965bffd1709cca327ad07bd72fb2589668ba681adbcde8bef`。测试确认 80 个 case 唯一：

- 困难集 60：Goal 0/9、Spatial 4/9、Object 5/6，每任务 init 0–9；
- 容易控制集 20：Goal 6/7、Spatial 1、Object 7，每任务 init 0–4；
- env seed 固定 7；flow-noise seed 按 case 顺序从 42000 连续生成；
- 所有候选共享同一 task、init state、env seed、flow-noise schedule、checkpoint 与 stats。

第一矩阵只改变真正执行的 action 数，模型始终预测 10 步：

```text
execute_len = 1, 2, 4, 5, 10
```

晋级条件同时满足：困难集绝对提升 >=15 pp、容易集下降 <=5 pp、至少 3 个困难任务改善，并提供 case-level
paired 结果。达到门槛后才逐变量检查 integration steps、noise lifecycle、KV cache on/off 和 FP32 solver；
camera A/B 只有在阶段 1 给出明确 layout 假说后才运行。

每个失败 episode 需要记录：未接近、接近未对准、抓取失败、抓后掉落、铰接失败、运输失败、放置错误、
predicate 未触发，以及最小 EEF 距离、夹爪闭合、物体最大离桌高度、最终位置、action saturation、gripper
翻转和 replan 次数。只改善“接近”而不改善“抓取/放置”的候选不晋级。

**当前禁止运行该 panel。** 原因不是算力，而是 M1 的真实 demonstration 对齐与 M2 replay 尚未闭合；现在运行
会把尚未解释的数据契约变量重新混入闭环成功率。

### 15.10 阶段 4：仅重建 Goal 微调

若阶段 1/2 闭合、阶段 3 的最佳协议仍约 80%，停止抢救公开 suite checkpoint，只训练 Goal。训练必须逐级
通过，不能直接投入 45k steps：

1. **单 batch 过拟合**：固定 batch 训练 500–2000 次；flow loss 应显著下降，mask 外维度不贡献 loss，
   router 无 NaN，保存/重载后同 noise 输出一致。
2. **1–10 条轨迹过拟合**：离线 action error 接近低值，并在相同训练 init state 闭环成功；若离线成功、
   闭环失败，优先检查 unnormalization、gripper、control frequency、chunk 前缀、camera key 和 prompt。
3. **单任务训练**：同时选择一个公开权重强任务和弱任务，例如 Goal task 6（100%）与 task 0（22%）。
4. **Goal suite 微调**：仅在前三层通过后执行，并按论文 schedule 保存多个中间 checkpoint。

Goal 的论文 schedule：

| Suite | Global batch | 论文 steps | 必须保留 checkpoint |
|---|---:|---:|---|
| Goal | 64 | 45k | 5k、10k、15k、20k、25k、30k、32.5k、35k、37.5k、40k、42.5k、45k、47.5k、50k |

论文还给出 AdamW、初始 LR `2.5e-5`、weight decay `1e-4`、warmup 1k、最低 LR `2.5e-6`、
`N=32`、`top-K=4`、`lambda_AS=0.002`、`lambda_HB=0.001`。但论文同一段同时写 cosine schedule 和
“之后 exponential decay 到 30k”，必须以冻结后的 scheduler 代码行为为准，并把每 step LR 写入训练日志。

论文 v2 Appendix C.6 明确描述两阶段 MoE warm-up：第一阶段冻结 VLM 和 action module 中所有非 MoE 参数，
只更新 routers、experts 与 shared expert；第二阶段解冻全模型。但论文只写“short warm-up phase”，没有给出
LIBERO 的精确长度。因此先从同一 base、data order 和 seed 运行三个总长 5k 的 pilot：

| Pilot | MoE-only warm-up |
|---|---:|
| A | 0 steps |
| B | 500 steps |
| C | 1000 steps |

选择指标为 offline action error、gripper accuracy、AS/HB routing collapse、训练 loss，以及困难/容易任务各
20 个 dev rollout；不得用最终 500 episodes 选择 pilot。训练期间每 task 固定 10 个 init states 为 dev，另
40 个为 final；checkpoint 选择同时看宏平均、最差 task、困难任务、离线误差和 gripper accuracy。

中间 checkpoint 先跑预注册的 100-episode validation panel；正式 500 episodes 只在配置和选择规则冻结后
运行一次，避免用测试集反复挑 checkpoint。

### 15.11 正式评测输出要求

每个 episode 至少记录：

```json
{
  "suite": "goal",
  "task_id": 0,
  "init_state_id": 7,
  "environment_seed": 7,
  "policy_seed": 42,
  "flow_noise_seed": 42,
  "success": false,
  "action_steps": 300,
  "inference_calls": 30,
  "checkpoint_sha256": "...",
  "norm_stats_sha256": "...",
  "runtime_source_audit_sha256": "...",
  "failure_stage": "grasp_failed"
}
```

最终报告同时提供每 task 成功率、suite 成功率、Wilson 95% CI、失败阶段分解、checkpoint 选择规则和
所有试验但未采用的配置。

### 15.12 当前执行顺序

| 优先级 | 工作 | 当前状态 / 下一动作 |
|---|---|---|
| P0 | 扩大持久盘并固化 97 GB runtime/checkpoint | 仍受 8 GB work 容量阻塞 |
| P0 | baseline v1、HF lineage、checkpoint metadata | **已完成并冻结**，不重复评测/下载 |
| P1 | 合成 train/eval contract + action round-trip | **已完成**，两项 layout needs_review |
| P1 | 获取 demonstrations 并生成源/转换 manifest | **当前唯一外部阻塞项**；扩容前不下载 |
| P1 | 真实 100-sample 对齐、stats 重算 | demonstrations 到位后立即执行 |
| P1 | demonstration replay、teacher-forcing、一步状态 | 同一数据到位后执行 |
| P2 | 80-case execute_len / flow / KV paired panel | manifest 已冻结；M1/M2 通过前不运行 |
| P2 | Goal 单 batch、少轨迹、单任务训练 | 数据与训练/评测闭环一致后执行 |
| P2 | Goal 0/500/1000 warm-up 5k pilots → 45k | 前述训练 gate 通过后执行 |
| 暂缓 | Long、Spatial/Object exact、OpenVLA 500、RAD、环境扫描 | 不定位当前主差距，停止投入 |

### 15.13 向作者请求的最小可复核包

1. Table 1(b) 四套 checkpoint 的精确 global step、文件 SHA-256 和训练 run ID。
2. 生成这些 checkpoint 的 Git commit、全部 submodule commit、容器或 lockfile。
3. 每套 normalization 文件的 SHA 及其对应数据 manifest。
4. 正式 500 个 episode ID、environment/policy/noise seeds 和完整命令。
5. 一个固定 observation/state/prompt/noise 的逐层 golden fingerprint。
6. LIBERO 转换后数据集的样本数、源文件 hash、过滤日志和转换命令。

只有在现有环境对照、HF lineage、运行时源码、真实 preprocessing/layout、stats 重算、demonstration replay
和 Goal 独立微调均闭合后，公开权重仍稳定在约 80% 且独立微调明显更高，才有充分依据把主因归到公开
checkpoint lineage 或未公开训练/推理绑定；不再把“四套 exact gate 全部完成”设成启动 Goal 重建的前置条件。
在此之前不写“作者传错权重”。

---

## 16. 证据目录索引

| 证据 | 路径 |
|---|---|
| 历史 checkpoint / 失败模式审计 | [himoe-repro-report.md](himoe-repro-report.md) |
| 主运行手册 | [himoe-libero-bridge/README.md](himoe-libero-bridge/README.md) |
| HF / checkpoint lineage 审计 | [checkpoint-lineage-audit-20260803.json](himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-lineage-audit-20260803.json) |
| 实际运行时源文件审计 | [runtime-source-audit-20260803.json](himoe-vla-cache/himoe-libero-bridge/metadata/runtime-source-audit-20260803.json) |
| released-checkpoint baseline v1 | [released-checkpoint-baseline-v1.json](himoe-vla-cache/himoe-libero-bridge/metadata/released-checkpoint-baseline-v1.json) |
| 新主线 machine status | [data-contract-stage-status-20260803.json](himoe-vla-cache/himoe-libero-bridge/metadata/data-contract-stage-status-20260803.json) |
| 数据契约审计运行手册 | [DATA_CONTRACT_AUDIT.md](himoe-libero-data-contract/DATA_CONTRACT_AUDIT.md) |
| 80-case 固定诊断集 | [diagnostic-panel-v1.json](himoe-libero-data-contract/manifests/diagnostic-panel-v1.json) |
| Goal 合成数据契约报告 | [data-contract report](himoe-vla-cache/himoe-libero-bridge/formal-artifacts/data-contract-goal-20260803T125234441831Z/report.json) |
| Goal 合成 golden tensors | [synthetic-golden-contract.npz](himoe-vla-cache/himoe-libero-bridge/formal-artifacts/data-contract-goal-20260803T125234441831Z/synthetic-golden-contract.npz) |
| robosuite 1.4.1 控制器方向报告 | [controller report](himoe-vla-cache/himoe-libero-bridge/formal-artifacts/controller-direction-goal-task00-20260803T125938010895Z/report.json) |
| Goal exact acceptance | [summary.json](himoe-libero-bridge/artifacts/exact-acceptance-goal-20260731T174918397053967Z/summary.json) |
| Goal 正式评测 | [paper-alignment Goal](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-rs141-egl-paperpins-20260801T1737Z/) |
| Spatial 正式评测 | [paper-alignment Spatial](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-spatial-rs141-egl-paperpins-20260802T0908Z/) |
| Object 正式评测 | [paper-alignment Object](himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-libero-object-rs141-egl-paperpins-20260802T1134Z/) |
| 超时反证 | [timeout-probe-spatial-task04](himoe-env-ab/timeout-probe-spatial-task04/) |
| top-K=8 对照 | [topk8-goal-20260802T1155Z](himoe-env-ab/topk8-goal-20260802T1155Z/) |
| OpenVLA 阳性对照 | [openvla-control-goal-20260802T1232Z](himoe-env-ab/openvla-control-goal-20260802T1232Z/) |
| 环境 A/B | [himoe-paper-ab](himoe-paper-ab/) |
| RAD v4 完整结果 | [rad-goal-10x5-tv-v4](himoe-vla-cache/himoe-libero-bridge/formal-artifacts/rad-goal-10x5-tv-v4/) |
| 持久化说明 | [himoe-vla-cache/README.md](himoe-vla-cache/README.md) |

注意：主运行手册中仍有少量“pending”文字，它描述的是实验启动前的计划状态；Goal 10x50、RAD smoke、
RAD 50-case 等工作实际上已经完成。当前完成状态以本报告和对应 JSON/log 为准，运行命令仍以 README 为准。

---

## 17. 机器可读 manifest

```yaml
report:
  date: 2026-08-03
  scope: released-artifact evaluation plus data-contract audit

paper:
  arxiv: 2512.05693v2
  revision_date: 2026-07-08
  libero_claims:
    spatial: 0.982
    object: 0.994
    goal: 0.986
    long: 0.958

source:
  himoe_commit: 27a2c46932d8b6373ca0074eb997f299bcd4f6f5
  libero_commit: 8f1084e3132a39270c3a13ebe37270a43ece2a01
  bridge_commit: f5dc26bbf4d878ad6621c7be85335eb61f494dc0
  baseline_tag: released-checkpoint-baseline-v1
  bridge_tests: 138_passed
  audit_branch: audit/data-contract-v1
  audit_commit: 9e04e85f96c7bbd92ec72baa8f82f65e52726e87
  audit_tests: 148_passed
  baseline_manifest: himoe-vla-cache/himoe-libero-bridge/metadata/released-checkpoint-baseline-v1.json
  baseline_manifest_sha256: 9ec4194d0237eaffef9ce9a4b1b41fb69252d1a8e2bf7b425c0c348a439fb67a
  stage_status: himoe-vla-cache/himoe-libero-bridge/metadata/data-contract-stage-status-20260803.json
  stage_status_sha256: 09754f6a3123f13c02e1e638880df85810dffdbd4661a7e4e0908746b2bfe9b1
  runtime_source_audit: himoe-vla-cache/himoe-libero-bridge/metadata/runtime-source-audit-20260803.json
  runtime_source_audit_sha256: 40184470dfa95a9d0e9c051f7a94013c622a8536bc200f708f6f72af71da449d

checkpoints:
  lineage_audit: himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-lineage-audit-20260803.json
  lineage_audit_sha256: d195772c8368508898da0ca5622c6ec2b8b28b6c66f1b3d38dfe5c00810b43d8
  goal:
    hf_artifact_revision: f957f85f224a14db902a2f813f64bb90f06cbc2c
    bytes: 8138322389
    sha256: 98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953
    normalization_sha256: 0a900ab12eb7cf3a25a0afb406dfc95dbe04a215b4feed7cd4c1fa5be44acfcf
    container: {type: OrderedDict, tensors: 1581, global_step: null}
  spatial:
    hf_artifact_revision: 15ddfddd7d48eb1bb503b0a13f67e9b665c199e6
    bytes: 8138322389
    sha256: 1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137
    normalization_sha256: 4b06853e1320af2a05614f14bdc79e441b4f175c9c18970657c80c7ed779d266
    container: {type: OrderedDict, tensors: 1581, global_step: null}
  object:
    hf_artifact_revision: aca489193fa990b0fa447b88fa4cd038c8a02da0
    bytes: 8138322389
    sha256: f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa
    normalization_sha256: 50719783647bee5adb7d936863eff02e0500da6e13867ddbcbfbc551b3a06487
    container: {type: OrderedDict, tensors: 1581, global_step: null}
  long:
    status: deferred_remote_identity_verified_local_missing
    hf_artifact_revision: c466da7e2b33da74370c44946c0273bd5917fd74
    expected_sha256: cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256

evaluation:
  environment_seed: 7
  policy_seed: 42
  trials_per_task: 50
  tasks_per_suite: 10
  replan_steps: 10
  resize: 224
  results:
    goal: {successes: 410, episodes: 500, rate: 0.820}
    spatial: {successes: 406, episodes: 500, rate: 0.812}
    object: {successes: 396, episodes: 500, rate: 0.792}
    long: {status: not_run}
    openvla_goal_control: {successes: 39, episodes: 50, rate: 0.780}

exact_acceptance:
  goal: completed_exact
  spatial: deferred
  object: deferred

data_contract_goal:
  status: partial_needs_review_and_demonstrations
  synthetic_train_eval_common_inputs: bitwise_exact
  action_round_trip_max_abs_error: 2.7755575615628914e-17
  invalid_action_dimension_nonzero_count: 0
  raw_state_dim: 8
  model_state_dim: 48
  validity_true_count: 7
  wrist_slot: released_left_paper_describes_right_needs_review
  report: himoe-vla-cache/himoe-libero-bridge/formal-artifacts/data-contract-goal-20260803T125234441831Z/report.json
  report_sha256: e53f9e14a940f627da7b41ec4a3051eb8eca698d7161d56fae20e6cd4d79b3d1
  tensor_archive_sha256: f4067f7596a5a66876340dad82ff4dc7b9a0a1e9da25fdea0a53cfce23ac8c11

controller_direction_goal:
  status: passed
  runtime: {python: 3.8.20, robosuite: 1.4.1, mujoco: 3.2.3, renderer: OSMesa}
  translation_axis_sign: 6_of_6
  rotation_axis_sign: 6_of_6
  gripper: {open: -1, close: 1}
  report_sha256: 4566bbb1b7ccf2a09b52f8157adb0e59b21d09aef92f906c4b0933701488247a

next_stages:
  demonstrations_present: false
  demonstration_replay: blocked
  teacher_forcing: blocked
  one_step_state_test: blocked
  diagnostic_panel: {status: frozen_not_run, cases: 80, execute_len: [1, 2, 4, 5, 10]}
  goal_training: not_started
  deferred: [long_10x50, spatial_object_exact, openvla_500, rad_expansion, environment_sweeps]

rad_goal_10x5_v4:
  status: completed
  paired_cases: 50
  random: 0.74
  action_medoid: 0.82
  routing_medoid: 0.76
  k1: 0.78
  primary_delta: 0.02
  primary_task_cluster_ci95: [-0.08, 0.12]
  primary_mcnemar_p: 1.0

storage:
  persistent_research_artifacts_gib: 1.9
  work_total_gib: 8.0
  work_available_gib: 1.4
  large_runtime_caches_persistent: false

forensics:
  hf_original_vs_main: completed_no_weight_or_norm_replacement_found
  checkpoint_training_step: unprovable_from_state_dict
  golden_input_output_fingerprint: synthetic_completed_real_demonstration_blocked
  wrist_channel_paired_ab: pending
  reconstructed_norm_stats: blocked_no_demonstrations
```

---

## 18. 最终结论

截至 2026-08-03，HiMoE-VLA 的公开 checkpoint 评测链路已被建立并经过较强审计：三套正式
500-episode 评测、OpenVLA 阳性对照、Goal fresh-server exact replay、超时反证、top-k 对照、
依赖 A/B、50-case RAD、HF revision lineage、checkpoint 顶层元数据和实际运行时源码哈希均已有落盘证据。
上述结果现已冻结为 `released-checkpoint baseline v1`；后续审计在独立 branch/worktree 进行。

这些证据一致表明：当前公开 Goal/Spatial/Object checkpoint 是稳定、可加载且未被后续 revision 替换的
有效模型，但在对齐环境下稳定落在约 79%-82%，无法复现论文的约 98%-99%；其纯 state dict 元数据又
不能证明训练 step。环境和协议仍不能被数学意义上完全排除，但已测试的替代解释都不足以解释主差距。

新完成的张量级审计进一步证明：在合成共同输入上，公开 train/eval transform、resize、token、padding 与
action round-trip 可以 exact 闭合；robosuite 1.4.1 的控制器方向和 gripper 语义也通过。它同时把两个具体
问题留在台面上：8D state 对应 7 个 validity bit，以及论文 right-arm 表述与公开 left-wrist slot 不一致。

当前最高信息增益动作不是再跑闭环大样本，而是取得 LIBERO demonstrations，完成真实 100-sample 对齐、
normalization 重算、demonstration replay、teacher-forcing 和一步状态测试。通过后才运行已冻结的 80-case
execute_len/flow/KV panel；若最佳协议仍约 80%，立即进入 Goal 的单 batch → 少轨迹 → 单任务 → 5k warm-up
pilot → 45k 微调重建。Long、Spatial/Object exact、OpenVLA 500 和 RAD 扩展暂缓。

在真实数据契约与 Goal 独立微调完成前，最严谨的措辞仍是“公开权重身份无法与论文 Table 1(b)
checkpoint 建立充分绑定”，而不是“权重损坏”或“作者传错权重”。
