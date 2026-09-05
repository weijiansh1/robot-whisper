# 按 token 位置检查 chunk 间 MoE 路由变化

## 结论

- 相邻 chunk 的同位置 token 路由重排很大：完整 80 个路由站点的 Top-4 集合完全一致为 **0 / 487,480**（0.00%）。
- 同时存在可测的位置骨架：在下一个 chunk 的 T1–T10 中做最近邻，同位置命中率为 **28.42%**（随机基线 10%），但这不等于位置固定。
- 同位置距离比错位位置平均低 **16.37%**；lag 1→4 的距离逐步增大，之后基本饱和，表现为位置骨架上的连续变化叠加显著路由重排。
- PCA-10 只保留相邻变化约 **9.79%** 的平方距离能量，距离相关为 **r=0.577**；它适合可视化和粗比较，不适合精确同路由判定。

## 数据与表示

共分析 51,308 个 control chunks、2,560 个 episodes、48,748 对 episode 内相邻 chunks。
每个 action token 先构造成 2560 维稀疏路由向量：`8 HB layers × 10 denoise steps × 32 experts`。每个站点使用实际保存的 Top-4 expert id，并将 `hb_selected_prob` 在 Top-4 内归一化为真实 combine weight。
T1–T10 共用同一个 PCA 基底；PCA 分量是特征轴，token 位置是样本轴。三个 checkpoint（goal / spatial / long）分别拟合。用于跨 checkpoint 汇总的距离采用白化 PC 坐标，并对三个 checkpoint 等权平均。

## Checkpoint 结果

| checkpoint | chunks | adjacent pairs | PCA-10 variance | raw/PCA distance r | subspace overlap | aligned RMS | off-position RMS | nearest-position |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal | 16,470 | 15,446 | 12.94% | 0.604 | 98.58% | 1.168 | 1.373 | 26.94% |
| long | 22,883 | 22,371 | 14.70% | 0.561 | 97.09% | 1.093 | 1.334 | 30.74% |
| spatial | 11,955 | 10,931 | 12.95% | 0.565 | 95.83% | 1.146 | 1.367 | 27.57% |

`PCA-10 variance` 衡量对所有静态路由差异的保留量；`raw/PCA distance r` 衡量对相邻 chunk 变化大小排序的保真度。`subspace overlap` 是换一套 episode 抽样与随机种子后，十个主方向的平均 cos² 重合度。
三套 PCA-10 子空间复拟合重合度为 95.83%–98.58%，说明主子空间稳定；但累计方差仍低，10 维不应替代原始 Top-4 做“完全相同”判定。

## 逐 token 的相邻变化

下表是三个 checkpoint 的等权宏平均。白化距离单位是每个 PC 的标准差；Top-4 Jaccard 和 cosine 均在原始 2560 维表示上计算。

| token | PCA-10 RMS | raw cosine | Top-4 Jaccard | site Top-4 exact | whole-token exact |
|---:|---:|---:|---:|---:|---:|
| T1 | 0.969 | 0.338 | 0.226 | 1.06% | 0.00% |
| T2 | 1.096 | 0.259 | 0.167 | 0.35% | 0.00% |
| T3 | 1.167 | 0.227 | 0.145 | 0.24% | 0.00% |
| T4 | 1.181 | 0.215 | 0.136 | 0.21% | 0.00% |
| T5 | 1.210 | 0.208 | 0.132 | 0.19% | 0.00% |
| T6 | 1.206 | 0.211 | 0.134 | 0.20% | 0.00% |
| T7 | 1.199 | 0.218 | 0.139 | 0.22% | 0.00% |
| T8 | 1.189 | 0.226 | 0.144 | 0.23% | 0.00% |
| T9 | 1.144 | 0.246 | 0.158 | 0.30% | 0.00% |
| T10 | 0.996 | 0.335 | 0.223 | 0.92% | 0.00% |

## 随 chunk 间隔的变化

| lag | same-position PCA-10 RMS |
|---:|---:|
| 1 | 1.136 |
| 2 | 1.199 |
| 3 | 1.232 |
| 4 | 1.246 |
| 5 | 1.246 |

## 一个实际 10 维向量

示例来自 `goal/open_the_middle_drawer_of_the_cabinet`，episode 0，q0 → q1 的 T1：

- q0: `[-1.685, +1.544, +0.150, -0.984, -0.243, +0.993, -1.041, +2.994, -1.902, +1.020]`
- q1: `[-1.467, +0.687, -0.860, +1.377, +0.466, +1.779, +0.342, -0.646, -1.578, +1.109]`
- delta: `[+0.218, -0.857, -1.011, +2.361, +0.708, +0.786, +1.383, -3.640, +0.324, +0.089]`

这些正负数只表示 checkpoint-local PCA 坐标，不分别对应前移、旋转或夹爪等动作语义。

## 产物

- `token_pca10.npz`: `pca_scores` / `pca_whitened` 的 shape 均为 `[chunk, 10 token positions, 10 PCs]`。
- `adjacent_chunk_deltas.npz`: episode 内每对相邻 chunk 的白化 10 维 delta 及原始路由指标。
- `pca_models.npz`: 各 checkpoint 的 mean、components、explained variance，可复算新样本。
- `summary.json`: 全部汇总矩阵、逐位置指标和方法元数据。
- `overview.png` / `component_change.png`: 位置距离矩阵、lag 曲线、累计方差和逐 PC 变化。

读取示例：

```python
import numpy as np

data = np.load('token_pca10.npz')
z = data['pca_whitened']       # [51308, 10, 10]
one_token = z[0, 0]            # 第 0 个 chunk 的 T1，10 维
```

## 边界

这里只比较同一 episode 内的 control chunks；episode 边界没有连成相邻对。PCA 拟合样本对每个 episode 等量抽取，所有 token 位置等权。不同 checkpoint 的专家槽位没有共同身份，因此没有把三套原始路由拼成一个 PCA。
