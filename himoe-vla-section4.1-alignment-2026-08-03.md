# HiMoE-VLA Section 4.1 LIBERO 复现差距定位与对齐报告

报告日期：2026-08-03  
论文制品：`arXiv-2512.05693v2.tar.gz`  
论文制品 SHA-256：`9804dc969092b5788ca681544b3777047781c1cc2a883211eaed7628ea2d351a`  
官方 HiMoE-VLA commit：`27a2c46932d8b6373ca0074eb997f299bcd4f6f5`  
官方 LIBERO commit：`8f1084e3132a39270c3a13ebe37270a43ece2a01`  
冻结旧基线 bridge：`f5dc26bbf4d878ad6621c7be85335eb61f494dc0`  
right-slot 正式结果 bridge：`927640c55e096a81f04c4effcd649fa435984c97`  
最终可审计实现 bridge：`d7797415d0fce6b4fd25393dbc0d1923ac28ed3b`

---

## 1. 结论

本轮已经找到导致原复现结果约 80%、显著低于论文约 98% 的主要原因：

> **公开 LIBERO transform 把唯一真实 wrist 图像放入 `left_wrist_0_rgb`，同时把
> `right_wrist_0_rgb` 置零。这与论文的单臂 right-arm 对齐约定形成跨模态槽位不一致；
> 更关键的是，固定噪声 paired rollout 和三套完整评测均表明，公开 checkpoint 的实际行为强烈
> 支持它期望真实 wrist 位于 right slot。**

这不是 MuJoCo、robosuite、EGL、top-K、horizon、checkpoint 下载损坏或普通随机种子问题。
它是一个不会触发 shape/type 错误的**语义槽位错配**：模型仍可 `strict=True` 加载和正常输出动作，
但腕部视觉 token 进入了错误的位置，尤其破坏依赖接触、抓取和精细对准的任务。

只把 wrist 图像从 left slot 移到 right slot，不改两个最终 wrist mask、权重、normalization、prompt、
state、action、flow solver、环境、控制器或 rollout 协议后：

- 固定 50-case、固定 flow-noise 的 Goal paired panel 从 `40/50 = 80%` 提升到 `48/50 = 96%`；
- 10 个失败转成功，2 个成功转失败，净增 8 局；双侧 exact McNemar `p = 0.038574219`；
- Goal 官方 `10 tasks x 50 episodes` 从 `410/500 = 82.0%` 提升到
  `489/500 = 97.8%`；
- Goal 与论文 `98.6%` 只差 `0.8 pp`，Wilson 95% 区间为
  `[96.10%, 98.77%]`，包含论文点估计。
- Spatial 在公开默认 `replan=10` 下从 `406/500 = 81.2%` 提升到
  `471/500 = 94.2%`，提升 `13.0 pp`；但仍比论文 `98.2%` 低 `4.0 pp`，
  Wilson 95% 区间 `[91.79%, 95.93%]` 不包含论文点估计。
- Object 官方 `10 tasks x 50 episodes` 从 `396/500 = 79.2%` 提升到
  `483/500 = 96.6%`，提升 `17.4 pp`；但仍比论文 `99.4%` 低 `2.8 pp`，
  Wilson 95% 区间 `[94.62%, 97.87%]` 不包含论文点估计。
- right-slot 下关闭 left 空图像 mask 的严格消融从 `20/20` 降到 `18/20`，没有提供额外对齐收益；
- Spatial 将 replan 从公开默认 10 改成 5 后达到 `478/500 = 95.6%`，但 95% 区间
  `[93.43%, 97.08%]` 仍不包含论文 `98.2%`，不能视为完全复现。

因此已找到并修复三套已下载 checkpoint 共同的主差距。Goal 从“明确无法复现”推进到“点估计接近
且统计相容”；Spatial 和 Object 获得大幅恢复，但其论文点估计仍落在本地 95% 区间之外，不能声明
完全复现。也不能写成逐 episode 或逐随机流的精确复现，因为论文没有发布正式 episode manifest、
policy-noise 流、checkpoint global step 或 Table 1(b) 的完整制品绑定。

---

## 2. Section 4.1 结果状态

论文 TeX 的 LIBERO Table 1(b) 位于 `example.tex:253-263`：

| Suite | 论文 | 公开代码旧基线 | `checkpoint-right` 对齐结果 | 与论文差值 | 当前判断 |
|---|---:|---:|---:|---:|---|
| Spatial | 98.2% | 406/500 = 81.2% | **471/500 = 94.2%** | **-4.0 pp** | 主差距修复，未完全复现 |
| Object | 99.4% | 396/500 = 79.2% | **483/500 = 96.6%** | **-2.8 pp** | 主差距修复，未完全复现 |
| Goal | 98.6% | 410/500 = 82.0% | **489/500 = 97.8%** | **-0.8 pp** | 主差距已对齐 |
| Long | 95.8% | 未运行 | 未运行 | 不可计算 | 本地缺完整权重 |

三套可运行 checkpoint 合计从 `1212/1500 = 80.8%` 提升到 `1443/1500 = 96.2%`。论文三套
点估计对应 `1481/1500 = 98.73%`；因此按成功数作描述性比较，wrist 修复恢复了原缺口中的
`231/269 = 85.9%`。该汇总来自独立的完整 run，不能当作逐 episode 配对统计；严格因果对照是
第 4.2 节固定 flow-noise 的 Goal paired A/B。

这里的 `checkpoint-right` 指“left 图像置零、真实 wrist 放在 right、两个 wrist mask 都有效”。旧日志
在 bridge `927640c` 中把该模式命名为 `paper-right`；第 3.3 节证明上游后续 mask rewrite 使其实际
语义是 `checkpoint-right`。最终 bridge `d779741` 已把两种模式拆开，避免再次混淆。

表中保留公开脚本默认 `replan=10` 作为主协议。诊断性 `replan=5` 的 Spatial 最佳观察值为
`478/500 = 95.6%`（与论文差 `-2.6 pp`），但论文没有公布该执行长度，且结果仍未统计对齐。

Goal 对齐后每任务结果：

| Task ID | 旧版 `released-left` | 修复后 `checkpoint-right` | 变化 |
|---:|---:|---:|---:|
| 0 | 22% | 98% | +76 pp |
| 1 | 96% | 96% | 0 pp |
| 2 | 96% | 100% | +4 pp |
| 3 | 80% | 92% | +12 pp |
| 4 | 84% | 98% | +14 pp |
| 5 | 98% | 100% | +2 pp |
| 6 | 100% | 100% | 0 pp |
| 7 | 100% | 100% | 0 pp |
| 8 | 98% | 100% | +2 pp |
| 9 | 46% | 94% | +48 pp |

Goal 的提升不是均匀的小幅波动，而是原先两个最严重弱项 task 0 和 task 9 大幅恢复，这与 wrist
视觉槽位错误对精细操作任务的预期影响一致。

Spatial 对齐后每任务结果：

| Task ID | 旧版 `released-left` | 修复后 `checkpoint-right` | 变化 |
|---:|---:|---:|---:|
| 0 | 86% | 100% | +14 pp |
| 1 | 96% | 92% | -4 pp |
| 2 | 94% | 96% | +2 pp |
| 3 | 92% | 100% | +8 pp |
| 4 | 28% | 76% | +48 pp |
| 5 | 86% | 94% | +8 pp |
| 6 | 90% | 98% | +8 pp |
| 7 | 86% | 94% | +8 pp |
| 8 | 94% | 96% | +2 pp |
| 9 | 60% | 96% | +36 pp |

Spatial 的主差距同样被 wrist 修复消除，但残余失败集中在 task 4。失败视频显示模型能够打开抽屉并
继续执行抓取/运输阶段，却在取碗时漏抓；这不是 horizon 耗尽、场景未加载或 success predicate 漏报。
因此 Spatial 不能声明达到论文结果，残差需由未发布 checkpoint step/训练 lineage 或未唯一化的
评测随机流解释，而不能继续归因于已排除的通用环境因素。

Object 对齐后每任务结果：

| Task ID | 旧版 `released-left` | 修复后 `checkpoint-right` | 变化 |
|---:|---:|---:|---:|
| 0 | 88% | 96% | +8 pp |
| 1 | 72% | 100% | +28 pp |
| 2 | 92% | 98% | +6 pp |
| 3 | 72% | 94% | +22 pp |
| 4 | 92% | 98% | +6 pp |
| 5 | 36% | 96% | +60 pp |
| 6 | 64% | 90% | +26 pp |
| 7 | 98% | 96% | -2 pp |
| 8 | 90% | 98% | +8 pp |
| 9 | 88% | 100% | +12 pp |

Object 原最弱的 task 5 从 36% 恢复到 96%，task 6 从 64% 恢复到 90%。这提供了第二个跨 suite
行为证据：同一个 right-slot 修复恢复了原先集中退化的精细抓取任务；但 task 6 仍有稳定残差，所以
Object 不能被表述为达到论文 99.4%。

---

## 3. 论文约束、公开代码与 checkpoint 行为不一致

### 3.1 论文 TeX

本地 TeX `example.tex:590-592` 明确描述：

- 视觉输入是一张第三人称视角加两张 wrist 视角；
- 缺失视角需要 zero-pad 并通过 attention mask 标为无效；
- 单臂 demonstration 的可用 arm 映射到 right-arm channel；
- left-arm channel zero-pad 并 mask。

这不是从网页或二手报告推断，而是来自用户提供的原始 arXiv TeX 制品。需要严格区分：TeX 没有
逐字段写出 `right_wrist_0_rgb`，且 right-arm 句子紧接 state/action 表述；因此，仅凭这句话不能把
wrist 字段名当成明文规范。这里的直接事实是论文采用 right-arm 单臂约定，而公开 transform 却把
单个 wrist 视觉信号命名为 left slot；checkpoint 究竟期望哪个视觉槽位，由第 4 节的受控实验判定。

### 3.2 公开 LIBERO transform

官方 commit `27a2c46` 的 `src/moevla/policies/libero_policy.py:47-59` 实际是：

```python
"image": {
    "base_0_rgb": base_image,
    "left_wrist_0_rgb": wrist_image,
    "right_wrist_0_rgb": np.zeros_like(base_image),
},
"image_mask": {
    "base_0_rgb": np.True_,
    "left_wrist_0_rgb": np.True_,
    "right_wrist_0_rgb": np.False_,
},
```

这是 `LiberoInputs` 的直接输出：

```text
left wrist  = 真实 wrist，valid
right wrist = 全零，invalid
```

然而公开 `LeRobotLiberoDataConfig` 随后无条件执行 `DropStateAndImage`。即使 dropout 概率为默认的
`0.0`，该 transform 仍把两个 wrist mask 都重新写成 `True`。因此模型实际收到的是：

```text
left wrist  = 真实 wrist，valid
right wrist = 全零，valid
```

本地最小链路探针的输出为：

```text
after_libero_inputs     left=True, right=False
after_drop_state_image left=True, right=True
```

公开 `libero_policy.py` SHA-256：

```text
b9b5998891e2f2e3b50f93d3221a12987d392987fb7c808cff6f83ff46c09ecc
```

包含 `DropStateAndImage` 的公开 `transforms.py` SHA-256：

```text
6ded6c9f261da6c24b2f7eccafb8dfd8aa277af3781b31fe60dec7f3f9ab65ab
```

公开的 `LeRobotLiberoDataConfig` 在训练侧也复用了同一串 transform，所以当前公开源码按字面看是
训练/推理一致的 left-slot、双 valid-mask 实现；但发布 checkpoint 对 right-slot 显著更好。这说明公开 checkpoint
很可能由另一版未发布 transform、不同数据槽位约定或未公开训练分支产生。官方 Git 历史中该文件只有
一个可见提交，checkpoint 又没有记录训练 commit，无法从公开制品反推出训练时究竟使用了哪一版。

### 3.3 旧结果的真实 mask 语义

bridge `927640c` 把 right-slot remap 插在 `LiberoInputs` 后、`DropStateAndImage` 前。remap 当时先设置
`left=False/right=True`，但下一步立即把两者覆盖为 `True/True`。所以三套完整结果和 Goal 固定噪声
A/B 的真实唯一变量都是**wrist 图像位置**；两个条件最终 mask 完全相同。这不是结果失效，反而排除了
“提升只来自 attention mask”这一混杂因素。

最终提交 `d779741` 将 remap 移到 `DropStateAndImage` 之后，并把模式明确拆成：

```text
released-left:    left=真实/valid, right=零/valid
checkpoint-right: left=零/valid, right=真实/valid
paper-right:      left=零/invalid, right=真实/valid
both:             left=真实/valid, right=真实/valid
```

### 3.4 为什么旧结果看起来“能运行但不正确”

两个 wrist 槽位的 tensor shape 都是 `224 x 224 x 3`，mask 也都是标量布尔值。因此交换语义后：

- checkpoint 参数名、shape 和 dtype 完全不变；
- `load_state_dict(strict=True)` 仍通过；
- 推理不会抛异常；
- action 数值仍在合理范围；
- 简单或主要依赖第三人称视角的任务仍可成功；
- 依赖 wrist 近场视觉的任务会系统性退化。

这解释了为什么旧版不是 0%，而是稳定在 79%-82%，且失败集中在少数任务。

---

## 4. 因果证据链

### 4.1 离线通道敏感性

对真实 Goal rollout 保存的 observation 使用固定 flow noise，只改变图像通道，模型动作的 normalized
RMS 差值为：

| 干预 | mean normalized action RMS delta |
|---|---:|
| wrist zero | 0.2511 |
| wrist shuffle | 0.2827 |
| base zero | 0.1028 |
| base shuffle | 0.1006 |
| wrist 移到 right slot | 0.2049 |
| wrist 同时放两槽 | 0.2198 |

说明 checkpoint 对 wrist 内容和 wrist token 位置都高度敏感。该探针使用旧 trace 与当前 runtime，
不能作为 exact replay 证据；它只作为“通道不是无关输入”的支持证据。正式因果结论来自下一节的
固定 episode / 固定 flow-noise paired rollout。

证据：
`himoe-vla-cache/himoe-libero-bridge/formal-artifacts/channel-sensitivity-goal-v1/report.json`。

### 4.2 Goal 10x5 固定噪声 paired A/B

严格固定：

- 相同 checkpoint 及 SHA；
- 相同 normalization 及 SHA；
- 相同 task、init-state、environment seed；
- 每 case 相同显式 `10 x 24` flow noise；
- 相同 prompt、图像像素、state、action unpack；
- 相同 robosuite 1.4.1、MuJoCo 3.2.3、EGL；
- 相同 replan 10、settle 10 和 horizon；
- 唯一变化是 wrist 图像 slot；两个条件的最终 wrist mask 都是 `True/True`。

结果：

| 最终布局 | 成功 | 成功率 |
|---|---:|---:|
| `released-left` | 40/50 | 80% |
| `checkpoint-right` | 48/50 | 96% |

配对转移：

| 转移 | 数量 |
|---|---:|
| failure -> success | 10 |
| success -> failure | 2 |
| success -> success | 38 |
| failure -> failure | 0 |

净提升 `+8/50 = +16 pp`，双侧 exact McNemar `p = 0.038574219`。

旧版 summary SHA-256：
`ba195dc5484d2b351d2e4c10b91463f391c8f0f315d9cd995258e8475a435986`。  
修复版 summary SHA-256：
`803b9cb6129e47a293c483a754439d374a0e454f240a0341279fb5099f2e4646`。

#### 4.2.1 right-slot mask paired A/B

发现 `DropStateAndImage` 的 mask rewrite 后，在最终实现 `d779741` 上选 Goal task 0、1、6、9，
每任务固定 init state 0-4，并对每个 case 使用完全相同的显式 flow-noise seed。唯一变量是 left 空图像
对应的 attention mask：

| 最终布局 | Task 0 | Task 1 | Task 6 | Task 9 | 总计 |
|---|---:|---:|---:|---:|---:|
| `checkpoint-right`：left/right mask=True/True | 5/5 | 5/5 | 5/5 | 5/5 | **20/20** |
| `paper-right`：left/right mask=False/True | 5/5 | 4/5 | 4/5 | 5/5 | **18/20** |

配对转移是 18 个 success -> success、2 个 success -> failure，没有 failure -> success；双侧 exact
McNemar `p=0.5`。样本量不足以声称 mask 差异显著，但结果明确没有支持关闭 left mask，且出现了两个
反向退化。因此三套完整 500-run 所验证的 `checkpoint-right` 是发布 checkpoint 的推荐模式；严格
`paper-right` 保留为论文 mask 语义消融，不冒充已经完成的正式对齐结果。

`checkpoint-right` summary SHA-256：
`184c23ed8ecad4ef147bd0acf1dde1824196caf5957abd6ad3fd98793177d831`。  
`paper-right` summary SHA-256：
`a4ad5e6ad6d557cd28870b74b8a0a91242f94c4a53ade8532d9f31c2583e6ba5`。

此外，将最终提交 `d779741` 的 `checkpoint-right` 与旧提交 `927640c` 的有效 right-slot 布局逐 case
比较。对 task 0、1、6、9 的 20 个固定 case，保存的 `arrays`、`result`、`config` 全部一致，
`20/20` 无 mismatch；其中 predicted actions、图像、wrist 图像、MuJoCo states、reward/done 和 flow
noise 的 dtype、shape、SHA-256 均逐项相同。这证明最终命名修复没有改变已经完成 500-run 所使用的
模型输入/闭环行为。

compatibility audit SHA-256：
`2490e77dff6a1f62627b8b04965ce5dbdd6b8d5b4d19c00dbbb9bdbd207a7a97`。

### 4.3 Goal 官方 500 回合

使用官方 `examples/libero/main.py`，新启模型 server 后不做 inference probe，保留官方内部
`torch.manual_seed(42)` flow-noise 生命周期：

| 运行 | 成功数 | 成功率 | Wilson 95% CI | 论文值 |
|---|---:|---:|---:|---:|
| 公开 transform | 410/500 | 82.0% | [78.4%, 85.1%] | 98.6% |
| `checkpoint-right` | **489/500** | **97.8%** | **[96.10%, 98.77%]** | 98.6% |

修复版日志 SHA-256：
`f6546708b944e90e5df9c241c1ccc37e2d7f72f587e0fbde2bb1a64693ed2152`。

### 4.4 Spatial 官方 500 回合

同样使用官方 `examples/libero/main.py`，保持 robosuite 1.4.1、MuJoCo 3.2.3、EGL、seed 7、
replan 10 和全新 seed-42 模型服务：

| 运行 | 成功数 | 成功率 | Wilson 95% CI | 论文值 |
|---|---:|---:|---:|---:|
| 公开 transform | 406/500 | 81.2% | [77.54%, 84.38%] | 98.2% |
| `checkpoint-right` | **471/500** | **94.2%** | **[91.79%, 95.93%]** | 98.2% |

修复版日志 SHA-256：
`7e4ead3ebcf1000c6adf02cd5064f06812c25ad58f02e205bf8906034d9bb1c6`。

运行在 500 episodes 和最终统计全部写入后出现 robosuite EGL context 析构告警，但主进程退出码为 0；
该告警不发生在任何 episode 内，不改变成功计数。task 4 的失败样本把旧版 horizon 从 220 延长到
1000 步仍 8/8 失败，因此没有通过增大 horizon 人为抬高本次数字。

### 4.5 Object 官方 500 回合

Object 继续使用同一冻结协议和全新的 seed-42 模型服务：

| 运行 | 成功数 | 成功率 | Wilson 95% CI | 论文值 |
|---|---:|---:|---:|---:|
| 公开 transform | 396/500 | 79.2% | [75.43%, 82.53%] | 99.4% |
| `checkpoint-right` | **483/500** | **96.6%** | **[94.62%, 97.87%]** | 99.4% |

修复版日志 SHA-256：
`4356a2a945c80e3fe71164f6bca892bcd275803d8eb32160088d3fb7bbd7b9bc`。

500 个 episode 全部正常计数，主进程退出码为 0。其逐任务提升覆盖 9/10 个任务；唯一小幅回退的
task 7 为 `98% -> 96%`，而原最弱 task 5/task 6 分别净增 30 和 13 次成功。整体净增
`87/500 = 17.4 pp`，但仍不足以支持论文 99.4% 的点估计。

### 4.6 Spatial `replan=5` 协议筛选

论文 TeX 没有给出 action chunk 实际执行长度；公开 `examples/libero/main.py:29` 的默认值是 10。
因此先在固定 flow-noise 的 Spatial `10 tasks x 5 init states` panel 上只改变执行长度：

| replan | 成功 | 困难 task 4+9 | 其余 8 tasks |
|---:|---:|---:|---:|
| 10 | 46/50 | 6/10 | 40/40 |
| 5 | 49/50 | 9/10 | 40/40 |

配对转移为 4 个 failure -> success、1 个 success -> failure，exact McNemar `p=0.375`；小样本只满足
预设筛选门槛，不构成显著结论。随后用新的 seed-42 server 跑完整 500 局：

| Task | replan 10 | replan 5 | 变化 |
|---:|---:|---:|---:|
| 0 | 100% | 96% | -4 pp |
| 1 | 92% | 94% | +2 pp |
| 2 | 96% | 100% | +4 pp |
| 3 | 100% | 98% | -2 pp |
| 4 | 76% | 88% | +12 pp |
| 5 | 94% | 98% | +4 pp |
| 6 | 98% | 98% | 0 pp |
| 7 | 94% | 88% | -6 pp |
| 8 | 96% | 100% | +4 pp |
| 9 | 96% | 96% | 0 pp |
| **总计** | **471/500 = 94.2%** | **478/500 = 95.6%** | **+1.4 pp** |

`replan=5` 的 Wilson 95% 区间是 `[93.43%, 97.08%]`，仍不包含论文 `98.2%`。两次完整 run 使用
相同环境 init states，但 server 的全局 flow-noise 流会在行为分叉后发生偏移，所以 500 局数字只能作
aggregate 比较，不能伪装成逐 episode paired 结果。它改善 task 4，却让 task 7 回退；完整结果也远低于
50-case panel 暗示的 98%。因此公开默认 replan 10 仍是主复现协议，replan 5 只作为当前观察到的较优
诊断协议，不能解释剩余差距，更不能声称是论文未公开协议。

完整日志 SHA-256：
`04bd6ddd87976a2809a3941d7ebe4b0c3cc73b0e6f6c0eabb89267a3df7316a5`。

### 4.7 state validity 反证

公开 transform 把 8 个 LIBERO state 值写入 state value 部分，但沿用 7D action mask，只设置 7 个
state-validity bit。为避免把另一个可疑点和主修复混在一起，独立实现了默认关闭的 `full-8` A/B。

同一 Goal 50-case 固定噪声 panel：

| wrist 布局 | state validity | 成功 |
|---|---|---:|
| `checkpoint-right` | 公开 `released-7` | **48/50** |
| `checkpoint-right` | 实验 `full-8` | 46/50 |

`full-8` 没有补齐残差，反而净下降 2 局。因此它不进入正式修复；这也表明 checkpoint 很可能就是按
公开的 7-bit state validity 行为训练，而 wrist token 则来自另一版/论文所述的 right-slot transform。

---

## 5. 对齐实现

### 5.1 代码位置

干净 worktree：

```text
/home/jovyan/work/himoe-libero-wrist-fix
```

Git 分支与 commit：

```text
fix/libero-paper-wrist-layout
d7797415d0fce6b4fd25393dbc0d1923ac28ed3b
```

修改文件：

```text
src/himoe_libero_bridge/policies.py
src/himoe_libero_bridge/server.py
src/himoe_libero_bridge/cli.py
tests/test_policies.py
tests/test_cli.py
```

测试：`155 passed`。

### 5.2 实现行为

新增 `--libero-wrist-layout`：

| 值 | 行为 | 用途 |
|---|---|---|
| `released-left` | 真实 wrist 在 left；最终两 wrist mask 均 valid | 冻结旧基线，默认值 |
| `checkpoint-right` | 真实 wrist 在 right；最终两 wrist mask 均 valid | 已完成三套 500-run 的 checkpoint 对齐模式 |
| `paper-right` | 真实 wrist 在 right；left invalid、right valid | 严格 mask 契约消融 |
| `both` | 两个槽位均使用真实 wrist/valid | 消融，不作为正式结果 |

transform 被插入到官方唯一一个 `DropStateAndImage` 之后；若它存在但不紧跟唯一的 `LiberoInputs`，
代码会立即失败，避免再次被后续 transform 静默覆盖。remap 创建新 image/mask 字典，不修改输入对象，
不改变真实 wrist 像素内容。

保留 `released-left` 为默认值是为了不破坏已经冻结的审计基线。复现本报告已完成的对齐结果必须显式传：

```text
--libero-wrist-layout checkpoint-right
```

---

## 6. 如何运行

### 6.1 启动 Goal 模型服务

```bash
cd /home/jovyan/work/himoe-libero-wrist-fix

PYTHONPATH=src \
MOEVLA_DATA_HOME=/home/jovyan/.cache/himoe-libero-bridge/moevla-data \
/home/jovyan/.cache/himoe-libero-bridge/envs/model/bin/python \
  -m himoe_libero_bridge.cli serve \
  --backend himoe \
  --suite goal \
  --gpu 0 \
  --host 127.0.0.1 \
  --port 8063 \
  --checkpoint-dir /home/jovyan/.cache/himoe-libero-bridge/checkpoints/HiMoE-VLA-Libero-Goal \
  --upstream-root /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA \
  --libero-wrist-layout checkpoint-right
```

服务 metadata 必须包含：

```json
{
  "suite": "goal",
  "libero_wrist_layout": "checkpoint-right",
  "libero_wrist_layout_transform_anchor": "DropStateAndImage",
  "checkpoint_sha256": "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953",
  "normalization_stats_sha256": "0a900ab12eb7cf3a25a0afb406dfc95dbe04a215b4feed7cd4c1fa5be44acfcf"
}
```

### 6.2 运行官方 Goal 10x50

另开终端：

```bash
OUT=/home/jovyan/work/himoe-vla-cache/himoe-libero-bridge/paper-alignment/manual-checkpoint-right-goal
mkdir -p "$OUT/videos"

CUDA_VISIBLE_DEVICES= \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
LD_LIBRARY_PATH=/home/jovyan/.cache/himoe-libero-bridge/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
PYTHONPATH=/home/jovyan/work/.rs141-audit:/home/jovyan/work/.paper-eval-overlay:/home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO:/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src \
/home/jovyan/.cache/himoe-libero-bridge/envs/libero/bin/python \
  /home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/examples/libero/main.py \
  --host 127.0.0.1 \
  --port 8063 \
  --resize-size 224 \
  --replan-steps 10 \
  --task-suite-name libero_goal \
  --num-steps-wait 10 \
  --num-trials-per-task 50 \
  --video-out-path "$OUT/videos" \
  --seed 7 2>&1 | tee "$OUT/eval.log"
```

注意：正式运行前必须重启模型 server，且不能先发送 inference probe；否则会消费内部 flow-noise RNG，
使结果不能与本报告的正式运行直接比较。只做 WebSocket metadata handshake 不会消费 noise。

### 6.3 切换 suite

Spatial/Object 需要同时修改三处：

```text
serve --suite spatial/object
对应 checkpoint-dir
main.py --task-suite-name libero_spatial/libero_object
```

不要把 Goal normalization 与 Spatial/Object checkpoint 混用。

---

## 7. 已排除的解释

| 假说 | 结果 | 依据 |
|---|---|---|
| checkpoint 下载损坏 | 排除为主因 | HF LFS SHA 与本地 SHA 一致；1581 tensors 健康；strict load |
| 后续 HF revision 偷换权重 | 已基本排除 | 原始 revision 与 main 指向同一模型对象和 stats 历史 |
| MuJoCo/robosuite 小版本 | 排除为主因 | paper-pin 只把 Goal 83% 改到 82% |
| EGL/OSMesa | 排除为主因 | 渲染/环境 A/B 无法解释 16.6 pp |
| top-K | 排除为主因 | K=8 仅 41/50，对 K=4 的 40/50 只多 1 局 |
| horizon 不足 | 排除为主因 | Spatial task 4 的旧失败延长到 1000 步仍 8/8 失败 |
| 控制器轴/符号/夹爪 | 已通过 | xyz/rxyz 12/12；`-1=open`, `+1=close` |
| 普通 server 随机漂移 | 已通过 | Goal fresh-server 动作、状态、帧、MP4 exact replay |
| 8th state validity | 不采用 | 固定 paired panel 从 48/50 降到 46/50 |
| strict left wrist mask | 不采用 | right-slot 固定 paired panel 从 20/20 降到 18/20 |
| replan 10 是全部残差 | 排除 | Spatial replan 5 仅到 95.6%，论文值仍在 95% 区间之外 |

---

## 8. 仍无法完全复现的原因

### 8.1 论文没有唯一化正式评测随机流

TeX 只说在 held-out task episodes 上评测，没有发布：

- 500 个正式 episode 的不可变 manifest；
- 每 episode 的 policy/flow-noise seed；
- noise generator 是否按 episode 重置；
- 论文表格是否使用单 run、多个 seed 均值或 checkpoint selection run。

公开官方代码只固定环境 NumPy seed 7；模型端在 Policy 初始化时固定 torch seed 42，然后整个 server
连续消费 noise。episode 提前成功会减少 inference 次数，从而改变后续 episode 的 noise 流。因此不同
transform 的官方 500-run 只能比较 aggregate，不能在第一个行为分叉后继续声称逐 episode 配对。

### 8.2 checkpoint 自身不含训练身份

论文 `example.tex:625-627` 给出：Goal/Object 45k、Spatial 35k、Long 40k。公开代码四个 LIBERO
TrainConfig 却统一为 50k。三个已下载 checkpoint 都只是含 1581 个 tensor 的 `OrderedDict`，没有：

```text
global_step
epoch
optimizer
scheduler
training run ID
code commit
dataset manifest
```

因此不能证明公开权重与 Table 1(b) 使用的精确 step 完全相同。wrist 修复消除了主差距，但剩余
`0.8-4.0 pp` 差异仍可能来自 checkpoint step、训练 seed、checkpoint selection 或未公开评测噪声。

### 8.3 训练数据契约仍不完整

当前没有本地 LIBERO demonstrations，官方仓库也未完整发布训练数据 lineage 和 multi-dataset sampler。
因此暂时无法完成：

- 真实 train/eval 100-sample tensor 对齐；
- demonstration replay；
- teacher-forcing 离线动作误差；
- normalization stats 重算；
- 从 Base checkpoint 的严格 Goal 45k 微调重建。

### 8.4 Long 当前无法运行

本地 `HiMoE-VLA-Libero-10/pytorch_model.pth` 只有 135-byte Git-LFS pointer：

```text
oid sha256:cdc2b21f9ef657ab31049cfd2b1e2086ceb193a1b8c16af0cd6688925491a256
size 8138322389
```

完整 8.14 GB 权重不在本机。2026-08-03 的 GitHub、Hugging Face 和 HF mirror 本地连接分别出现
443 连接失败、超时和 connection reset；因此本轮不能诚实地给出 Long 结果。不能用其他 suite 权重
替代，也不能把 LFS pointer 当 checkpoint。

---

## 9. 证据索引

| 证据 | 路径 |
|---|---|
| 旧版完整状态报告 | `himoe-vla-reproduction-status-2026-08-03.md` |
| Goal 旧正式 500-run | `himoe-vla-cache/himoe-libero-bridge/paper-alignment/official-main-rs141-egl-paperpins-20260801T1737Z/` |
| Goal 修复正式 500-run | `himoe-vla-cache/himoe-libero-bridge/paper-alignment/paper-right-libero-goal-rs141-egl-20260803T1437Z/` |
| Goal 旧固定 10x5 | `himoe-paper-ab/rs141-egl-10x5/batch-libero-goal-20260801T172202979655Z/` |
| Goal 修复固定 10x5 | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/wrist-layout-paper-right-goal-10x5-v1/` |
| Goal task0/task9 修复 pilot | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/wrist-layout-paper-right-goal-task00-task09-v1/` |
| wrist 离线敏感性 | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/channel-sensitivity-goal-v1/report.json` |
| state validity A/B | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/state-validity-full8-paper-right-goal-10x5-v1/` |
| Spatial 修复 10x5 | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/wrist-layout-paper-right-spatial-10x5-v1/` |
| Spatial 修复正式 500-run | `himoe-vla-cache/himoe-libero-bridge/paper-alignment/paper-right-libero-spatial-rs141-egl-20260803T1624Z/` |
| Spatial replan 5 固定 10x5 | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/wrist-layout-paper-right-spatial-replan5-10x5-v1/` |
| Spatial replan 5 正式 500-run | `himoe-vla-cache/himoe-libero-bridge/paper-alignment/paper-right-replan5-libero-spatial-rs141-egl-20260803T2006Z/` |
| Object 修复正式 500-run | `himoe-vla-cache/himoe-libero-bridge/paper-alignment/paper-right-libero-object-rs141-egl-20260803T1752Z/` |
| right-slot 双 valid-mask 4x5 | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/wrist-mask-checkpoint-right-goal-4x5-v2/` |
| right-slot strict mask 4x5 | `himoe-vla-cache/himoe-libero-bridge/formal-artifacts/wrist-mask-paper-right-goal-4x5-v2/` |
| 最终/旧版 right-slot exact compatibility | `himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-right-compatibility-audit-20260803.json` |
| 本报告机器清单 | `himoe-vla-cache/himoe-libero-bridge/metadata/section4.1-alignment-20260803.json` |
| checkpoint lineage | `himoe-vla-cache/himoe-libero-bridge/metadata/checkpoint-lineage-audit-20260803.json` |
| runtime source hashes | `himoe-vla-cache/himoe-libero-bridge/metadata/runtime-source-audit-20260803.json` |

---

## 10. 最终可声明范围

当前可以严谨声明：

> 公开 transform 的 wrist 槽位与论文单臂 right-arm 约定及发布 checkpoint 的实际输入偏好不一致，
> 是此前 LIBERO 评测约 80% 的主要原因。只把真实 wrist 图像移到 right slot、保持两 mask 有效后，
> Goal 从 82.0% 提升到 97.8%，与论文
> 98.6% 的差距缩小到 0.8 个百分点，且论文点估计落在本地 95% Wilson 区间内；Spatial 从
> 81.2% 提升到 94.2%，Object 从 79.2% 提升到 96.6%，但后二者仍未统计对齐论文点估计。

Spatial 的诊断性最佳观察值是 replan 5 下的 95.6%，仍未对齐。上述历史 artifact 目录名中的
`paper-right` 是 commit `927640c` 当时的 CLI 名称，其有效语义均为本报告定义的 `checkpoint-right`；
最终代码已用两个独立选项消除该命名歧义。

当前不能声明：

- 四个 suite 已逐项精确复现；
- 公开 checkpoint 一定就是论文指定 step；
- 剩余差距一定来自随机种子；
- 作者上传了错误权重；
- Long 已评测。

Long 只有取得并校验完整权重后才能进入评测；当前保持“无法复现，原因是必需制品不可得”。
