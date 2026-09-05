# MoE 内在路由保护器 v7：无训练、任务无关迭代

日期：2026-09-05

## 结论

v7 在独立目录中实现了一套只读 MoE 内部路由、无需训练、运行时完全不知道任务身份的在线保护器。最终 external_8b 结果为：

| TP | FP | 风险召回 | early-4 召回 | 精度 | timely FPR | 检出风险中位提前量 |
|---:|---:|---:|---:|---:|---:|---:|
| 439 | 80 | 77.84% | 58.69% | 84.59% | 0.532% | 12 queries |

它相对旧 cold-start 方法多检出 115 个风险 episode，同时少 7 个误报；旧方法仍然依赖任务原型，v7 不依赖。

这里的核心不是训练一个更复杂的分类器，而是找出两个可解释、互补的内部路由相态：

1. **相对冻结**：后层路由相对该 rollout 自己的早期活动水平持续收缩。
2. **确认湍流**：denoising route curvature 持续升高，并同时失去长 lag recurrence；单独出现其中一项不报警。

最终规则只有：

```text
relative_freeze OR (persistent_flow_acceleration AND persistent_recurrence_loss)
```

## 严格的任务无关约束

运行时输入只有当前 query 的 `hb_router_probs[8,10,11,32]`。全局 profile 只有三个标量阈值和一个标量尺度，明确不含：

- task/suite ID；
- 任务原型、最近任务匹配或 task-indexed 数组；
- 每任务、每 suite 阈值；
- outcome、horizon、模拟器状态或未来 query；
- 分类器、拟合权重或梯度更新。

每条 rollout 只用自身 q1..q7 的早期路由建立相对基线。不同任务的尺度差异因此在 episode 内消去，而不是靠任务查表补偿。

## 内部量

### 1. 相对冻结

在 HB 后四层 L12..L15、final flow、10 个 action token 上，先计算相邻 query soft route 的 Hellinger mobility：

```text
m_l(q) = mean_token Hellinger(P_l(q), P_l(q-1))
b_l    = mean_q=1..4 m_l(q)
z_F(q) = W6-mean median_l[-log((m_l(q)+eps)/(b_l+eps))]
```

`z_F` 高表示后层路由相对自身初始状态进入低变化/锁定区。阈值是 pooled 16,000 条无标签 reference trajectory peak 的 q97.5。

### 2. 确认湍流

对后四层的十步 denoising route，计算平方根概率曲线的二阶差分范数：

```text
a(q) = mean ||sqrt(P_f+1) - 2 sqrt(P_f) + sqrt(P_f-1)|| / sqrt(2)
z_A  = W3-mean log((a(q)+eps)/(median_q=2..7 a(q)+eps))
```

高 `z_A` 必须连续确认 8 次。其阈值来自无标签 trajectory peak q70。

第二个头比较当前 route 与 lag 1..4 route 的 weighted Jaccard similarity：

```text
p(q)   = max(sim_lag2..4) - sim_lag1
z_R(q) = W6-mean -(p(q) - median_q=2..6 p(q)) / robust_scale
```

高 `z_R` 表示较长 lag 的复现优势消失，必须连续确认 4 次；阈值来自无标签 trajectory peak q65。只有 `z_A` 和 `z_R` 都曾确认，湍流分支才锁存报警。

最早可能的冻结报警是 q6，最早可能的确认湍流报警是 q10。所有平滑窗都只向过去看。

## 选择、校准与封存

必须区分三件事：

1. **机制发现/规则选择**使用过 development outcomes。固定机制后，对 640 个全局分位数组合做审计；主点约束为 precision >= 85%、FPR <= 0.5%、recall >= 70%，再最大化 F1。
2. **阈值数值校准**不读 outcome。三个阈值都只是完整 16,000 条 pooled reference routing trajectory 的经验 order statistic，成功和失败样本全部保留。
3. **external 评分**先写入并哈希 profile 与 15,600 条 first-alarm，再打开 outcome 文件。

因此这是 train-free 方法，但不是 outcome-blind 的方法发现。没有模型训练、参数拟合或任务监督；开发标签确实影响了 operating point 的选择，不能隐去。

external_8b 在 v3-v6 中已被查看过，所以也不宣称它是 pristine holdout。它仍可检验 sealed replay、任务身份移除后的迁移，以及缓存到原始张量的一致性。

## 结果

### 总体

| cohort | TP | FP | recall | early-4 | precision | FPR | median lead |
|---|---:|---:|---:|---:|---:|---:|---:|
| development_main | 382 | 67 | 78.44% | 55.85% | 85.08% | 0.468% | 9 |
| external_8b | 439 | 80 | 77.84% | 58.69% | 84.59% | 0.532% | 12 |

按 task cluster bootstrap，external recall 95% CI 为 68.90%-83.63%，precision 为 72.85%-91.04%，FPR 为 0.237%-0.911%。区间较宽，说明任务间异质性不能被总体 episode 数掩盖。

### 机制消融

| external detector | TP | FP | recall | precision | FPR |
|---|---:|---:|---:|---:|---:|
| relative freeze | 343 | 59 | 60.82% | 85.32% | 0.392% |
| flow acceleration alone | 289 | 205 | 51.24% | 58.50% | 1.363% |
| recurrence loss alone | 341 | 1852 | 60.46% | 15.55% | 12.317% |
| confirmed turbulence | 167 | 31 | 29.61% | 84.34% | 0.206% |
| final union | 439 | 80 | 77.84% | 84.59% | 0.532% |

两个湍流原语单独都不够特异；持久交集把误报压到 31。相对冻结与确认湍流重叠 71 TP / 10 FP，湍流分支在冻结之外净增 96 TP / 21 FP。这是最终提升的主要来源。

### 与已有方法比较

| external method | task-agnostic | TP | FP | recall | precision | FPR |
|---|:---:|---:|---:|---:|---:|---:|
| unknown-task dual + task prototypes | no | 324 | 87 | 57.45% | 78.83% | 0.579% |
| v4 task profile | no | 410 | 81 | 72.70% | 83.50% | 0.539% |
| v6 suite/task profile | no | 408 | 45 | 72.34% | 90.07% | 0.299% |
| **v7 intrinsic guard** | **yes** | **439** | **80** | **77.84%** | **84.59%** | **0.532%** |

v7 严格支配旧 cold-start，也以 +29 TP / -1 FP 超过 task-aware v4。相对 v6，它换取 +31 TP，但增加 35 FP；若目标首先是 precision/FPR，v6 仍更优。开发集审计另保留 q98/q70/q70 的保守点：340 TP / 37 FP，precision 90.19%，但它不是本报告的 external 主点。

### 跨 suite 行为

| external suite | risk n | TP | FP | recall | precision | FPR | early-4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_goal | 106 | 73 | 1 | 68.87% | 98.65% | 0.026% | 57.55% |
| libero_long | 274 | 236 | 69 | 86.13% | 77.38% | 1.852% | 84.31% |
| libero_object | 44 | 22 | 7 | 50.00% | 75.86% | 0.197% | 38.64% |
| libero_spatial | 140 | 108 | 3 | 77.14% | 97.30% | 0.078% | 15.71% |

统一 profile 在四个 suite 都能检出风险，但并非均匀校准。`libero_long` 占 80 个 external FP 中的 69 个；`libero_object` recall 较低；短 horizon 的 `libero_spatial` 受 q6 最早报警限制，early-4 较低。为保持任务无关，本版本没有针对这些差异追加 suite 修正。

## GPU 原始在线前缀回放

命令：

```bash
CUDA_VISIBLE_DEVICES=6,7 python experiments/verify_raw_causal_gpu.py
```

验证直接读取原始 Zarr 中的 `float16 hb_router_probs`，在两张 NVIDIA H20-3e 上以 float32 重算内部量，再逐 query 调用同一个运行时 monitor：

| 检查 | 结果 |
|---|---:|
| episode / query | 24 / 735 |
| 卡 6、7 都实际执行 | yes |
| first alarm 与 sealed cache 完全一致 | 24/24 |
| 改写未来 query 后既有前缀不变 | 24/24 |
| GPU vs cache 最大原始特征误差 | 1.67e-6 |
| stream vs batch 最大 score 误差 | 2.46e-5 |
| 最大单卡显存 | 12.21 MiB |

样本无 outcome 参与，均匀覆盖 freeze-only、turbulence-only、no-alarm 三类。显存占用很小，因为在线实现只需一个 episode；没有为了“用满卡”而复制全量缓存。

## “因果”的准确边界

本次验证建立的是**计算图与时间方向上的因果性**：query q 的分数只由 `P(0..q)` 决定；改变 `P(q+1..)` 不会改变已有报警。它排除了未来泄漏和离线 centered-window 伪影。

它没有证明“MoE routing 导致失败”。已有 committor 审计表明，q0 routing 和动作都无法选择最终命运；可读性要到机器人状态已经分离约 q6-q7 后才出现。旧 token-dynamics 结果也表明晚期低变化主要是已停滞物理/动作轨迹的读出。因此 v7 应被解释为**内部状态传感器**，不是先验 fate oracle，更不是 MoE-specific 因果终止机制。

要回答干预因果问题，需要在报警前的同一 simulator snapshot 上 fork 多个未来噪声续跑，比较报警/非报警或 route intervention 的 committor；当前 raw route replay 不具备同 snapshot 多分支反事实，不能替代该实验。

## 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-v7-0905
python experiments/select_operating_point.py
python experiments/evaluate_intrinsic_guard_v7.py
CUDA_VISIBLE_DEVICES=6,7 python experiments/verify_raw_causal_gpu.py
pytest -q
```

关键产物：

- `results/intrinsic_guard_v7/development_threshold_selection.json`：640 点开发集选择审计；
- `results/intrinsic_guard_v7/global_profile.npz`：唯一全局 profile；
- `results/intrinsic_guard_v7/sealed_manifest.json`：代码、协议、输入缓存和报警哈希；
- `results/intrinsic_guard_v7/evaluation_summary.json`：总体指标与 cluster bootstrap CI；
- `results/intrinsic_guard_v7/outcome_metrics_by_suite.csv`：suite 异质性；
- `results/intrinsic_guard_v7/raw_causal_gpu_verification.json`：双卡原始回放摘要；
- `results/intrinsic_guard_v7/raw_causal_gpu_samples.csv`：24 个逐 episode 审计记录。
