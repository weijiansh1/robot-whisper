# q0 同初态成败信息的跨任务复现

## 先说结论

**不支持把 Long 的 q0 AUC=0.602 当成跨任务普遍成立的早期成败信号。**

三个可估计外部任务各自重新训练后的 AUC 为：goal/drawer + bowl 0.467 [0.299, 0.673]；spatial/ramekin -> plate 0.367 [0.083, 0.642]；spatial/stove -> plate 0.604 [0.472, 0.736]。其中 0/3 的 95% 区间下界高于 0.5，方向也不一致。

更严格地冻结 Long 判别器后，三个目标任务 AUC 为：goal/drawer + bowl 0.581；spatial/ramekin -> plate 0.580；spatial/stove -> plate 0.536；经三任务 Holm 校正后 0/3 显著。共享 checkpoint 的两个 Spatial 任务互迁移也接近 0.5。

## 样本

| task | fail | success | mixed states | checkpoint |
| --- | --- | --- | --- | --- |
| goal/middle drawer | 0 | 512 | 0 | 98ee29d09d18 |
| goal/drawer + bowl | 42 | 470 | 7 | 98ee29d09d18 |
| long/two moka pots | 216 | 296 | 13 | cdc2b21f9ef6 |
| spatial/ramekin -> plate | 12 | 500 | 8 | 1029d0827030 |
| spatial/stove -> plate | 37 | 475 | 12 | 1029d0827030 |

`middle drawer` 为 512/512 成功，没有正负对照，因此不能计算成败 AUC。
其余任务均为同一任务内 16 初态 x 32 noise seeds；AUC 只比较同一初态的成功和失败 sibling。

## 每个任务内部重新训练：嵌套 held-out-seed

| task | q0 AUC | shifted AUC | mixed states | CI low | CI high |
| --- | --- | --- | --- | --- | --- |
| goal/drawer + bowl | 0.467 | 0.511 | 7 | 0.299 | 0.673 |
| long/two moka pots | 0.602 | 0.466 | 13 | 0.474 | 0.711 |
| spatial/ramekin -> plate | 0.367 | 0.544 | 8 | 0.083 | 0.642 |
| spatial/stove -> plate | 0.604 | 0.465 | 12 | 0.472 | 0.736 |

每折完整留出 4 个 noise seeds；特征坐标只在其余 28 seeds 中选择。
`shifted AUC` 保留标签和初态，但换成另一个测试 sibling 的 q0 路由。
q0 没有跨 chunk 历史，所以这里的 `token_current` 与此前的 `token_history` 数值完全相同。

## Long 判别器完全冻结后迁移

| task | frozen AUC | shifted AUC | mixed states | CI low | CI high | perm p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| goal/drawer + bowl | 0.581 | 0.489 | 7 | 0.373 | 0.749 | 0.180 | 0.539 |
| spatial/ramekin -> plate | 0.580 | 0.542 | 8 | 0.385 | 0.769 | 0.209 | 0.539 |
| spatial/stove -> plate | 0.536 | 0.540 | 12 | 0.389 | 0.685 | 0.316 | 0.539 |

这里连特征坐标、标准化和回归权重都只在 Long 上拟合；目标任务标签只用于最后算 AUC。
置换检验在目标任务每个初态内部打乱成败，Holm p 同时校正三个可估计目标任务。

Long 全量训练选择的 5 个 q0 坐标：

- `position/top1_centered|front_2_5|d0|t10`
- `position/entropy_centered|back_12_15|d9|t7`
- `position/adjacent_hellinger|back_12_15|d4|t9_to_t10`
- `position/top1_switch_rate|back_12_15|d7|t7_to_t8`
- `position/top4_turnover|front_2_5|d8|t4_to_t5`

## 同 checkpoint 的 Spatial 直接迁移

| source | target | AUC | CI low | CI high | perm p |
| --- | --- | --- | --- | --- | --- |
| spatial/ramekin -> plate | spatial/stove -> plate | 0.511 | 0.348 | 0.682 | 0.442 |
| spatial/stove -> plate | spatial/ramekin -> plate | 0.493 | 0.363 | 0.653 | 0.531 |

这项对照排除了 Long 与 Spatial 使用不同 checkpoint 这一混杂，但样本中的失败仍然较少。

## 复算校验

Long 旧报告 q0 AUC=0.601540；本脚本从原始 zarr 独立抽 q0 后得到 0.601540；绝对差 0。

## 解释边界

这是 q0 MoE 路由与最终成败的观察性可分性，不是因果证据。独立任务内重训若有效，只说明目标任务也存在某种 q0 信息；只有完全冻结迁移也有效，才支持 Long 中学到的是可复用的同一种失败结构。
不同 suite 使用不同 checkpoint，因此跨 suite 冻结迁移是严格但带域偏移的测试；同 checkpoint 的 Spatial 互迁移用于补这个对照。
