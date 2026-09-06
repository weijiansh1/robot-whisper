# 十流和弦 Query-Word 审计

## 结论

每个 flow step 被编码为五轨三值和弦，十个和弦原样组成 50-symbol query word。在 query 内，ordered-4 相对相同四和弦 bag 的增益为 0.00176 bits/flow-track。

但依赖几乎都是局部的：order-2 相对 order-1 增益为 0.01009 bits/flow-track，order-3 只再增加 0.00035，order-4 相对 order-3 反而降低 0.00067。

跨 query 时，完整 word-prefix 相对 clock+last-word 的增益为 0.00093 bits/flow-track。其 state-cluster 95% CI 排除零，但效应很小。这直接检验完整的十流单词，而不是 query 汇总近似。

物理 early-warning 不成立：最佳低-surprisal 通道在 4.7% success-episode FPR 下的 q-3 recall 为 8.1% （95% CI 5.1%--12.8%），matched AUC 仅 0.550。

字面 `Fl Fl Sy Sy` 也不是病句：任意 flow 窗口版本出现在 100% 的成功 episode，固定末端版本出现在 96.3%。因此本审计确认弱健康时序结构，同时否定其当前的 Trap-specific 在线检测能力。

## 表示

每个 flow chord 为 `[Fl/normal/Sh, Ds/normal/Sy, LR/normal/HR, Lk/normal/Sw, Ff/normal/Fs]`。每个 primitive 先按训练成功数据 robust scaling，字母阈值也只在 density states 上以 episode-uniform query 抽样拟合。

## Query 内构词法

| context | held-out NLL nats/flow-track |
|---|---:|
| flow position only | 0.76374 |
| ordered-1 | 0.39236 |
| ordered-2 | 0.38537 |
| ordered-3 | 0.38512 |
| ordered-4 | 0.38558 |
| bag-4 | 0.38680 |

| contrast | bits/flow-track | state-cluster 95% CI | positive states |
|---|---:|---:|---:|
| ordered-4 vs bag-4 | 0.00176 | [0.00165, 0.00187] | 100.0% |
| ordered-4 vs ordered-1 | 0.00978 | [0.00959, 0.00998] | 100.0% |
| ordered-2 vs ordered-1 | 0.01009 | [0.00994, 0.01026] | 100.0% |
| ordered-3 vs ordered-2 | 0.00035 | [0.00028, 0.00043] | 92.0% |
| ordered-4 vs ordered-3 | -0.00067 | [-0.00072, -0.00061] | 0.0% |

## 跨 Query 的完整单词预测

| context | held-out NLL nats/flow-track |
|---|---:|
| latent clock | 0.68660 |
| clock + last 50-symbol word | 0.67929 |
| complete word prefix | 0.67865 |

- full prefix vs clock: 0.01146 bits/track, 95% CI [0.01086, 0.01208], positive states 100.0%.
- full prefix vs clock + last word: 0.00093 bits/track, 95% CI [0.00058, 0.00127], positive states 82.0%.

## Scene8 物理 stasis（固定 pooled OOF success-episode FPR）

| score | success FPR | recall q-3 | recall onset | matched AUC |
|---|---:|---:|---:|---:|
| word_clock_innovation | 4.7% | 2.5% | 4.1% | 0.434 |
| word_last1_innovation | 4.7% | 2.5% | 3.0% | 0.435 |
| word_prefix_innovation | 4.7% | 4.1% | 5.1% | 0.450 |
| word_freeze | 4.4% | 0.5% | 0.5% | 0.436 |
| word_periodic | 2.7% | 1.5% | 2.0% | 0.543 |
| word_low_surprise | 4.7% | 8.1% | 8.1% | 0.550 |
| word_overregularity | 4.7% | 4.6% | 4.6% | 0.503 |
| word_overregularity_persistent | 4.7% | 6.6% | 7.1% | 0.501 |

## Flow 内直接 `Fl Fl Sy Sy` 检验

| fixed motif | success episode rate | recall q-3 | recall onset | matched AUC |
|---|---:|---:|---:|---:|
| flow_fl_fl_sy_sy_any | 100.0% | 100.0% | 100.0% | 0.439 |
| flow_fl_fl_sy_sy_terminal | 96.3% | 99.0% | 99.5% | 0.499 |
| flow_fl_parallel_sy_terminal | 54.7% | 58.9% | 59.4% | 0.499 |
| flow_terminal_fl_sy | 5.4% | 1.0% | 1.0% | 0.500 |

## 词汇稀疏性

fold-0 的 75,722 个 episode-balanced 健康 query 产生 72,022 个不同的精确 50-symbol words （95.1% 唯一）。

最高频精确单词：

- `Neutral -> Ff -> Neutral^6 -> Sw+Fs -> Sh+Fs`: 27 (0.036%)
- `Neutral -> Ff -> Neutral^6 -> Sw+Fs -> Fs`: 26 (0.034%)
- `Neutral -> Ff -> Neutral^6 -> Sw -> Fs`: 24 (0.032%)
- `Neutral -> Ff -> Neutral^5 -> Sw -> Sw+Fs -> Sh+HR+Sw+Fs`: 18 (0.024%)
- `Neutral -> Ff -> Neutral^6 -> Sw+Fs -> Sh+HR+Fs`: 17 (0.022%)
- `Sh+Ds+HR^7 -> Sh+Ds+HR+Sw -> Sh+Ds+HR+Sw+Fs^2`: 16 (0.021%)
- `Neutral -> Ff^2 -> Neutral^5 -> Sw -> Fs`: 16 (0.021%)
- `Neutral^8 -> Sw+Fs -> Sh+Fs`: 15 (0.020%)

精确单词不被强行压成 one-hot ID；HMM emission 对 50 个符号做因子化概率比较，因此相差一个 flow/track 的单词仍然相近。

## 实验判决

1. 健康 routing 有稳定的 flow-position morphology 和以相邻两步为主的局部构词规律。
2. 保留完整十流单词后，较早 query 的全局前缀仍有非零但极小的预测增量。
3. 95.1% 的精确单词只出现一次，说明不能把 50-symbol word 直接当成稠密词表 ID。
4. 所有基于该语法的 stasis 提前预警都很弱；当前结论是机制分析通过、在线检测 No-Go。
