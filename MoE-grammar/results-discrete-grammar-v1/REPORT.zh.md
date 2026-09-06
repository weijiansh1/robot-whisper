# 多轨离散 MoE 语法审计

## 结论

在完全不带记忆的五轨三值 tokenizer 上，ordered-4 相对同一历史词袋的 held-out next-symbol 增益为 0.0301 bits/track；全前缀 HMM 相对仅按内部时钟推进的增益为 0.0179 bits/track。

滞回 tokenizer 的对应前缀增益为 0.0222 bits/track。它只作为敏感性分析，因为滞回会把连续性写进字母本身。

这两个量回答是否存在可泛化的离散顺序结构；物理 onset 表回答这种结构是否真的能提前发现错误。

物理结论是 no-go：固定 4.7% success-episode FPR 时，ordered-4 在 q-3 的召回为 0.0%，全前缀 HMM 为 0.5%。健康顺序可预测，不等于 stasis 是违反该顺序的句子。

预定义 `Fl Fl Sy Sy` 在 85.5% 的成功 episode 中也出现，state/query-matched AUC 仅 0.508；它是常见 routing motif，不是 Trap 特异规则。

## 字母与和弦

每个 query 同时发出五个三值轨道：`Fl/normal/Sh`、`Ds/normal/Sy`、`LR/normal/HR`、`Lk/normal/Sw`、`Ff/normal/Fs`。阈值只由 density-state 成功 episode 的 20/80 分位数拟合，且每 episode 等权。主 tokenizer 无滞回。

## Held-out 健康 next chord

| model | nats / track |
|---|---:|
| unigram / no history | 0.9465 |
| bag of last 4 chords | 0.7557 |
| ordered last 4 chords | 0.7348 |
| latent clock only | 0.9200 |
| full-prefix categorical HMM | 0.9076 |

所有 headline NLL 均不含 `<END>`，模型在线不读取 task ID、绝对 query index、物理状态或 outcome。HMM 的训练相位由成功轨迹归一化进度初始化，因此 HMM 是全前缀滤波检验，不被解释为无监督发现的任务语义。

## 折间与 init-state 稳定性

| contrast | gain bits/track | state-cluster 95% CI | positive states | sign-flip p |
|---|---:|---:|---:|---:|
| ordered-4 vs bag-4 | 0.0301 | [0.0285, 0.0317] | 100.0% | 5e-05 |
| ordered-4 vs ordered-1 | 0.0189 | [0.0175, 0.0204] | 100.0% | 5e-05 |
| full prefix vs clock | 0.0179 | [0.0173, 0.0185] | 100.0% | 5e-05 |
| full prefix vs clock + last chord | -0.0023 | [-0.0028, -0.0019] | 2.0% | 5e-05 |
| ordered-4 vs unigram | 0.3054 | [0.3010, 0.3100] | 100.0% | 5e-05 |

置信区间以 held-out init state 为 cluster 重采样，而不是把同一 episode 的 query 误当独立样本。

### 历史长度分解

| maximum ordered context | nats/track | incremental gain bits/track |
|---:|---:|---:|
| 1 | 0.7479 | - |
| 2 | 0.7369 | 0.01593 |
| 3 | 0.7351 | 0.00262 |
| 4 | 0.7348 | 0.00038 |

绝大多数多步增益来自 q-2；q-3 较小，q-4 已接近零。更严格地，full-prefix HMM 相对 clock+last-chord 的增益为 -0.00235 bits/track，即完整前缀在这个模型中反而更差。因此当前证据支持 2-3 query 的局部离散句法，不支持长程全局句法。

## Scene8 物理 stasis（固定 pooled OOF success-episode FPR）

| score | success FPR | recall q-3 | recall onset | recall onset+3 | matched AUC |
|---|---:|---:|---:|---:|---:|
| unigram_innovation | 0.0% | 0.0% | 0.0% | 0.0% | 0.520 |
| clock_innovation | 4.7% | 1.5% | 1.5% | 2.0% | 0.476 |
| prefix_innovation | 4.7% | 0.5% | 0.5% | 0.5% | 0.481 |
| ordered4_innovation | 4.7% | 0.0% | 1.0% | 3.0% | 0.594 |
| bag4_innovation | 4.7% | 0.5% | 1.0% | 3.0% | 0.570 |
| prefix_persistent | 4.7% | 3.0% | 4.6% | 5.1% | 0.503 |
| overregularity | 4.7% | 6.1% | 6.1% | 6.6% | 0.516 |
| overregularity_persistent | 4.7% | 5.1% | 5.1% | 5.6% | 0.536 |
| return_failure | 4.4% | 4.6% | 5.1% | 6.1% | 0.481 |
| discrete_three_channel | 4.7% | 5.1% | 5.1% | 5.6% | 0.491 |

### 与连续 v2 的同 operating-point 对照

| representation / score | recall q-3 | recall onset | matched AUC |
|---|---:|---:|---:|
| discrete ordered-4 | 0.0% | 1.0% | 0.594 |
| discrete prefix HMM | 0.5% | 0.5% | 0.481 |
| discrete overregularity | 6.1% | 6.1% | 0.516 |
| continuous current query | 4.1% | 5.1% | 0.592 |
| continuous VAR(4) | 3.0% | 3.6% | 0.578 |
| continuous prefix HMM | 2.0% | 3.0% | 0.580 |

### 全前缀的物理增量

| history | success FPR | recall q-3 | recall onset | matched AUC |
|---|---:|---:|---:|---:|
| clock + last chord | 4.7% | 0.5% | 0.5% | 0.479 |
| complete prefix | 4.7% | 0.5% | 0.5% | 0.481 |

该阈值只用于同 FPR 比较。可部署的跨 init-state calibration 结果保存在 `summary.json` 的 `physical_stasis_anchor_calibrated`。

## 直接检验 `Fl Fl Sy Sy`

| predefined motif | success episode rate | recall q-3 | recall onset | matched AUC |
|---|---:|---:|---:|---:|
| fl_fl_sy_sy | 85.5% | 66.0% | 67.0% | 0.508 |
| fl_parallel_sy | 86.1% | 77.2% | 77.7% | 0.492 |
| switching_pair | 40.5% | 9.6% | 10.2% | 0.484 |
| lock_streak | 0.0% | 0.0% | 0.0% | 0.500 |
| abab | 29.1% | 16.8% | 17.3% | 0.517 |

`fl_fl_sy_sy` 要求四个连续 query 中前两次为 Fl、后两次为 Sy，允许并发；`fl_parallel_sy` 更接近 static 假设，要求四次中至少三次 Fl 且末两次 Sy。这些规则在看结果前固定，没有从 stasis 标签调参。

## 语言样本

fold-0 健康训练子集出现 139 个和弦。最高频和弦为：

- `Neutral`: 19.8%
- `Sy`: 5.2%
- `Lk`: 5.1%
- `Sh+Ds+HR+Sw+Fs`: 4.0%
- `Fs`: 3.2%
- `Ff`: 3.0%
- `Fl+Sy+LR+Ff`: 2.7%
- `Sw`: 2.7%

最高权重的 episode-balanced 转移为：

- `Neutral -> Neutral`: weight 424.62
- `Neutral -> Sy`: weight 105.99
- `Sy -> Neutral`: weight 87.57
- `Lk -> Lk`: weight 74.37
- `Fs -> Neutral`: weight 70.27
- `Neutral -> Fs`: weight 59.51
- `Ds -> Neutral`: weight 53.82
- `Neutral -> Ff`: weight 52.85

## 判定边界

离散顺序增益若为正，只能证明健康和弦的排列不是词袋；是否可称为 Trap grammar，仍取决于 prefix/ordered、duration、periodicity 和 return 通道能否在相同 episode FPR 下稳定超过无历史分数及连续 VAR/HMM。Scene8 目前仍只有可靠 stasis onset，没有可靠 loop onset。
