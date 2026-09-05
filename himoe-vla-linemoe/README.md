# himoe-vla-linemoe

**问题**：HiMoE-VLA 一次推理内的 flow 去噪是一条直线（已确立）；**那失败 / 卡住的
chunk 上，这条线还直吗？**

## 背景（已确立的部分）

`analysis_flow_straightness/`（2026-09-03，纯离线）在两批无标签 flow 轨迹语料上量到：

| 量 | goal/t00 | long/t08 | 完美直线 |
|---|---:|---:|---:|
| 路径长 / 弦长 | 1.0019 | 1.0025 | 1.0 |
| `cos(v_m, v_{m+1})` m=0..6 | 1.000 | 1.000 | 1.0 |
| 第 1 轮外推距最终动作 | 7.3% | 10.1% | — |
| 夹爪符号第 1 轮即与最终一致 | 100% | 99.6% | — |

上游训练是线性 flow matching（`x_t = t·noise + (1−t)·actions`，`u_t = noise − actions`），
目标速度沿条件路径与 t 无关，所以"直"正是目标要求的形状；外推式
`x̂ = x_m − t_m·v_m` 是它的精确代数反解。那两批语料**没有结局标签**，
所以问不了成败差异 —— 本目录补的就是这一块。

## 本目录做了什么

新采集一批**带结局标签**的完整 flow 轨迹语料，然后按预先冻结的 8 个信号做
组内配对检验。设计与判据见 [PREREG.md](PREREG.md)，在采集启动前写定；
信号定义直接沿用上面那套无标签度量，一字未改。

- 任务 `libero_10/t08 put both moka pots on the stove`（唯一有稳定结局方差的捕获任务）
- 16 初态 × 8 flow-noise seed = 128 集，K=1 on-policy
- `--libero-wrist-layout checkpoint-right`，与既有两批 flow 语料同口径
  —— **绝不可与 right-50x8（paper-right）同表**
- 每 query 落盘 11 点 flow 轨迹 + 逐 Euler 步 32 路 full softmax + 动作 chunk + 8 维本体状态

顺带补了上游一个洞：`rollout_flow_lead.py` 此前把每集的 success **只打印不落盘**，
现在增量写 `episodes.json`（这正是既有两批语料没有标签的原因）。

## 目录

```text
himoe-vla-linemoe/
├── PREREG.md            冻结的设计、度量与判据
├── code/
│   ├── capture.sh       采集驱动（smoke / lane A / lane B / status / stop）
│   └── analyze.py       冻结的 8 信号 × 3 时点、组内配对 AUC、episode 级置换 maxT
├── capture/             原始 run，每个初态一个目录
└── results/             summary.json 与报告
```

## 复现

```bash
bash code/capture.sh smoke          # 1 初态 x 2 集，先量速度
bash code/capture.sh lane A         # 初态 0,6,...,42
bash code/capture.sh lane B         # 初态 3,9,...,45
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 code/analyze.py --nperm 2000
```

本机 240 核，OpenBLAS 默认按核数开线程；numpy 重的脚本前面那两个环境变量不是可选项。
