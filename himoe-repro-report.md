# HiMoE-VLA 仿真结果复现审计报告

> **2026-08-14 更正：本报告的核心结论已被证伪，不能再单独引用。**
> 下文断言「问题不在我们的环境或协议，而特定于 HiMoE 发布的权重制品」。
> 实际原因在**我们这一侧的桥接输入实现**：公开 `LiberoInputs` 把真实 wrist 图像
> 放进了 left 槽位，而论文与发布 checkpoint 都要求单臂映射到 right 槽位。
> 修复后 Goal 82.0% → **97.8%**、Spatial 81.2% → **94.2%**、Object 79.2% → **96.6%**，
> 即本报告表格中 16.6–20.2 点的差距绝大部分来自该错配，而非权重制品。
>
> 当前结论、结果与运行命令请以
> [`himoe-vla-section4.1-reproduction-guide-2026-08-04.md`](himoe-vla-section4.1-reproduction-guide-2026-08-04.md)
> 为准，原因与统计证据见
> [`himoe-vla-section4.1-alignment-2026-08-03.md`](himoe-vla-section4.1-alignment-2026-08-03.md)。
>
> 本文件保留作为前序证据：第三节的审计项（依赖 pin、协议、checkpoint 完整性与
> 独立性、K=8 对照等）与第二节的失败模式分析仍然有效，**第一节的主结果表和
> 第四节的归因不再成立**。

日期：2026-08-02
论文：HiMoE-VLA (arXiv:2512.05693, v2 2026-07-08)
代码：https://github.com/ZhiyingDu/HiMoE-VLA @ `27a2c46`

---

## 一句话结论

在同一台机器、同一套 LIBERO 环境与评测协议下，**公开基线 OpenVLA 精确复现了它的公开数字（78.0% vs 79.2%，差 1.2 点）**，
而 **HiMoE-VLA 发布的三个 checkpoint 一致比论文低 16.6–20.2 点**。
问题不在我们的环境或协议，而特定于 HiMoE 发布的权重制品。

---

## 一、主结果

| 模型 / suite | 成功/总数 | 成功率 | Wilson 95% CI | 论文值 | 差值 |
|---|---:|---:|---|---:|---:|
| **OpenVLA-7B Goal（阳性对照）** | 39/50 | **78.0%** | [64.8%, 87.2%] | 79.2% | **−1.2** ✓ |
| HiMoE-VLA Goal | 410/500 | **82.0%** | [78.4%, 85.1%] | 98.6% | **−16.6** |
| HiMoE-VLA Spatial | 406/500 | **81.2%** | [77.5%, 84.4%] | 98.2% | **−17.0** |
| HiMoE-VLA Object | 396/500 | **79.2%** | [75.4%, 82.5%] | 99.4% | **−20.2** |

论文声称 HiMoE 在 Goal 上比 OpenVLA 高 19.4 点（98.6 vs 79.2）；
实测两者为 82.0 vs 78.0，**仅差 2.0 点**。OpenVLA 一侧复现准确，收窄几乎全部来自 HiMoE 一侧。

### 分任务成功率

**LIBERO-Goal（HiMoE，500 局）**

| # | 任务 | 成功率 |
|---|---|---:|
| 0 | open the middle drawer of the cabinet | **22%** |
| 1 | put the bowl on the stove | 96% |
| 2 | put the wine bottle on top of the cabinet | 96% |
| 3 | open the top drawer and put the bowl inside | 80% |
| 4 | put the bowl on top of the cabinet | 84% |
| 5 | push the plate to the front of the stove | 98% |
| 6 | put the cream cheese in the bowl | 100% |
| 7 | turn on the stove | 100% |
| 8 | put the bowl on the plate | 98% |
| 9 | put the wine bottle on the rack | **46%** |

**LIBERO-Spatial（500 局）**：86 / 96 / 94 / 92 / **28** / 86 / 90 / 86 / 94 / **60**
**LIBERO-Object（500 局）**：88 / 72 / 92 / 72 / 92 / **36** / 64 / 98 / 90 / 88
**OpenVLA Goal（50 局，对照）**：40 / 80 / 100 / 60 / 100 / 80 / 60 / 100 / 100 / 60

---

## 二、失败模式分析

1. **100% 的失败都是耗尽步数上限**，无一例中途失败。
   成功回合仅用预算的 24–59%（Goal 71–173 步 / 上限 300；Spatial 74–129 / 上限 220）。
2. **超时假说已实验证伪**：将 Spatial task4 的 8 个失败 episode 上限从 220 提到 1000（4.5 倍），
   **8/8 仍然跑满且全部失败**。失败不是"差一口气"，而是策略进入无法恢复的状态。
3. **失败集中在三类**（两个 suite 一致）：铰接物体（抽屉/柜子）、容纳关系 `In`（而非表面 `On`）、
   多候选物的细粒度空间指代。
4. LIBERO-Spatial 是天然对照：10 个任务动作完全相同（拿黑碗放盘子），
   每个场景有**两个一模一样的碗**，只靠语言指代区分。最差的 28% 恰是**唯一一个两碗同处一件家具**
   （一个在抽屉里、一个在柜顶）、只能靠 in/on 区分的场景。指向**语言→视觉指代绑定**失败，而非运动控制失败。

---

## 三、审计：已验证一致的项（未发现可解释差距的问题）

| 类别 | 结论 |
|---|---|
| 依赖 pin | python 3.8.20 / robosuite 1.4.1 / mujoco 3.2.3 / numpy 1.22.4 / PyOpenGL 3.1.7 / glfw 1.12.0 / numba 0.53.1 / llvmlite 0.36.0 —— 与官方 `requirements.txt` 逐条吻合 |
| 评测协议 | seed 7、10 任务×50 局、`num_steps_wait=10`、各 suite 步数上限、resize 224、replan 10、图像 180° 旋转、8 维 state —— 全部一致 |
| LIBERO 版本 | 论文 pin `f78abd6` vs 我们 `8f1084e`，diff 仅涉及 README/下载脚本；bddl_files、init_files、envs、benchmark **一行未改** |
| 模型结构 | 4.596B 参数（论文 ~4B）；AS-MoE 4层×3专家；HB-MoE 8层×32专家（论文 N=32 ✓） |
| 正则系数 | λ_AS=0.002、λ_HB=0.001 —— 与论文完全一致 |
| 归一化 | Normalize/Unnormalize 链路完整，三 suite 统计量各自独立且数值合理 |
| checkpoint 完整性 | 全 1581 张量 bf16；**0 NaN/Inf、0 全零张量、0 重复专家**；`strict=True` 加载通过 |
| checkpoint 独立性 | Goal vs Spatial 逐张量对比：1581 个中仅 23 个逐字节相同；HB-MoE 专家差异 15.2%、router 16.0%、VLM 6.6% → **微调确实训练了 MoE，三个 checkpoint 非同源导出** |
| 训练 prompt | OpenVLA 数据再生脚本用 `task.language`（文件名派生），与评测端一致 → 无 train/eval prompt 失配 |
| 推理确定性 | `Policy.__init__` 中有 `torch.manual_seed(42)` + `set_seed(42)`，同 server 内确定性；exact-replay 验收通过 |
| 上游 patch | 三个 patch 均无副作用（可选噪声入参 / config 初始化后被 strict 加载覆盖 / 修正 `from pytest import Cache` 笔误） |

### 已实验排除的假说

- 环境依赖差异
- 渲染后端（OSMesa vs EGL：500 局 83.0% vs 82.0%，噪声内）
- robosuite 1.4.0 vs 1.4.1
- LIBERO commit 差异
- 步数上限 / 超时（8/8 在 4.5 倍预算下仍失败）
- 评测协议差异
- 推理链路 bug
- **top-k 路由**（K=8 对照：41/50=82% vs K=4 基线 40/50=80%，+1 局纯噪声）
- 专家未被训练 / 三 checkpoint 同源 / DeepSpeed 合并漏参数 / 全零或 NaN 污染

> 注：arXiv v1 正文写 "top-K routing of 8"，与其自身 Table 9（只测 K∈{2,4}，最优 K=4/N=32）矛盾。
> **v2 已更正为 K=4**，与发布代码一致。我们的 K=8 对照独立验证了这一修订。

---

## 四、仍无法排除的原因（需作者信息）

1. **发布权重不是论文表格所用权重（概率最高）**
   论文（v2）明确：Goal 45k、Object 45k、Spatial 35k、Long 40k steps；
   而发布的 `TrainConfig` 一律写死 `num_train_steps=50_000`。
   Spatial 相差 15k、Goal/Object 相差 5k。发布页面未标注 `global_step`。
   `serve_policy.py` 中仅注册了 `LIBERO_10_20k`（且路径为占位符），暗示评测可能用的是别的 step。
2. **权重与当前代码版本不匹配** —— 官方安装要求手动覆盖修改版 `lerobot` 与 `modeling_gemma.py`，
   仓库无 release/tag，缺少"论文权重 ↔ 代码 commit"的绑定点。
3. **隐藏的数据处理契约** —— 数据集未发布（README To-Do 未勾选），
   无法核验失败轨迹判定、gripper 编码、action padding、统计量计算时点等。
4. **论文数字本身可能含 checkpoint / seed 选择**。

---

## 五、建议的下一步

1. 向作者索取五项：
   ① 四个 suite 各自对应 Table 2 的 checkpoint step；
   ② 生成这些 checkpoint 的 git commit 与全部 submodule commit；
   ③ 每个 checkpoint 对应的归一化文件及 hash；
   ④ 论文评测的 500 个 episode ID、seed 与完整命令；
   ⑤ 一个固定 observation/state/prompt/seed 的 golden action 输出。
2. 若能拿到 35k/40k/45k/50k 中间 checkpoint，用同一批固定 episode 做 paired evaluation。
3. 补跑 LIBERO-Long（libero_10）以补齐 Table 2 第四个 suite（checkpoint 尚未下载）。

---

## 六、复现清单（manifest）

```yaml
paper:
  arxiv: 2512.05693 (v2, 2026-07-08)

code:
  himoe_commit: 27a2c46932d8b6373ca0074eb997f299bcd4f6f5
  libero_commit: 8f1084e3132a39270c3a13ebe37270a43ece2a01   # 论文 submodule pin: f78abd6 (差异仅 README/下载脚本)
  bridge_commit: f5dc26bbf4d878ad6621c7be85335eb61f494dc0

checkpoints:                                # 均已校验 SHA-256，各 8,138,322,389 bytes
  goal:    98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953
  spatial: 1029d0827030a7521361d1904eeb3e7e7f2792be5c99abdb5701abb7ee87c137
  object:  f9c5661533d271dec15d54d56fcd8c6c8811fc2b96095ac87638f7b3b2bdaafa
  claimed_global_step: unknown              # 发布页面未标注

runtime:
  python: 3.8.20        robosuite: 1.4.1     mujoco: 3.2.3
  numpy: 1.22.4         PyOpenGL: 3.1.7      glfw: 1.12.0
  numba: 0.53.1         llvmlite: 0.36.0
  render: EGL (HiMoE 三次评测) / OSMesa (OpenVLA 对照)
  gpu: NVIDIA H20-3e MIG 4g.71gb

evaluation:
  script: HiMoE-VLA/examples/libero/main.py (官方，未修改)
  env_seed: 7           policy_seed: 42 (Policy.__init__ 内置)
  trials_per_task: 50   num_steps_wait: 10  replan_steps: 10   resize: 224
  max_steps: spatial 220 / object 280 / goal 300
```

### 复现层次说明

本次完成的是 **artifact reproduction audit**（评测作者发布的权重）。
由于微调数据、checkpoint step 与完整制品 lineage 未充分公开，
严格意义的 **training reproduction** 与 **result reproduction** 目前无法完成。
