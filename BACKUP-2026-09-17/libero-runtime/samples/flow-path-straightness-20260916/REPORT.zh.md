# 一次推理内的去噪路径是否为直线

日期：2026-09-16。用 GPU 7 上的独立模型实例重放两条已保存 episode 的全部 query（观测 + 固定 flow noise），
包一层 `MoEVLA.denoise_step` 记录 10 个 Euler 步的 `x_t`、`v_t`，在归一化动作空间里度量直线性。
不改模型、不启动环境、不触碰共享服务；新增环境动作 0。

## 采样过程（`srv/src/moevla/models/moevla.py: sample_actions`）

- `x_0 = noise`（t = 1），`dt = -0.1`，循环 10 次：`v_t = denoise_step(x_t, t)`，`x_t += dt * v_t`，`t += dt`，最终 `x_10` 为动作（t = 0）。
- autocast bf16 下 `v_t` 为 bf16，`x_t` 保持 float32 累加；`dt * v_t` 的 bf16 舍入约为每步位移的 0.2%，远小于下面测到的弯曲。
- 内部动作 24 维，其中 7 维真实（data_mask），17 维为填充维，输出时丢弃。主结果只看 7 维 × 10 个 token 展平后的 70 维路径。

## 核验

- 两条 episode 共 92 个 query，重放返回的动作块与 trace 中 `predicted_actions` 逐位一致（max |diff| = 0）。
- 用与模型相同的 dtype 复现 `x_{k+1} = x_k + dt·v_k`，与下一步实际输入逐位一致；`x_0` 与提供的噪声逐位一致；钩子无泄漏。

## 结果（真实 7 维，每个 query 一条路径）

| 指标 | Pro swap task00，失败，52 query | LIBERO-10 task08，成功，40 query |
| --- | ---: | ---: |
| 路径长度 / 弦长，中位（最小–最大） | 1.0114（1.0012–1.1115） | 1.0037（1.0009–1.0093） |
| 首末步方向夹角 ∠(v0, v9)，中位（最大） | 24.3°（65.0°） | 14.5°（23.6°） |
| 偏离弦线最大垂距 / 弦长，中位（最大） | 0.056（0.210） | 0.028（0.052） |
| 相邻步余弦 cos(v_k, v_k+1)，前 6 个转折 | ≥ 0.997 | ≥ 0.999 |
| 相邻步余弦，最后三个转折 6-7 / 7-8 / 8-9 | 0.994 / 0.990 / 0.976 | 0.999 / 0.997 / 0.986 |
| 速度大小 max/min，中位 | 1.08 | 1.09 |
| 用第一步速度直线外推的终点误差 / 弦长，中位 | 0.139 | 0.065 |
| 路径方差在 PC1 / PC1+PC2 上的比例，中位 | 99.6% / 99.97% | 99.9% / 99.99% |

- 填充维一起看（24 维）更弯：L/D 1.03–1.05、首末夹角 43–51°，但这些维不进入动作。
- 按 token：chunk 里靠后的动作 token（8、9）在失败 episode 里更弯（L/D 中位 1.02，前几个 token 约 1.005）。

## 判断

路径不是严格直线，但非常接近直线：前 7 步方向几乎不变，弯曲集中在最后 2–3 步（t → 0 时），整条路径接近平面弧。
最后几步并非可省略：用第一步速度一步外推，终点与十步结果相差弦长的 6–14%。
失败 episode 的路径系统性地比成功 episode 更弯，但这只是两条 episode 的观察，不构成信号结论。

## 文件与复现

- `libero-runtime/probe_flow_path_straightness.py`：重放、记录、核验与指标；`plot_flow_path_straightness.py`：作图。
- `report.json`：全部汇总；`<episode>.npz`：X (Q,11,10,24)、V (Q,10,10,24)、时间步、data_mask；`straightness.png/pdf`。

```bash
source ~/data/env.sh; cd ~/data/srv
PYTHONPATH=$DATA/srv/src:$DATA/srv/packages/openpi-client/src:$DATA/srv CUDA_VISIBLE_DEVICES=7 \
HIMOE_UPSTREAM_COMMIT=27a2c46932d8b6373ca0074eb997f299bcd4f6f5 \
$DATA/venv311/bin/python $DATA/libero-runtime/probe_flow_path_straightness.py --episode <episode_dir> ... --out <out_dir>
$DATA/miniconda/envs/torch/bin/python $DATA/libero-runtime/plot_flow_path_straightness.py
```
