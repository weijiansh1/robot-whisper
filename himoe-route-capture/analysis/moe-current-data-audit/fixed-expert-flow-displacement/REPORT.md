# 固定专家的总 Flow 位移审计

本分析固定 `task / state / denoise / action token / expert ID`，只在同一
expert 被选中的候选之间比较。`raw` 是乘 gate 前的 `RMS(E_e(h))`，
`weighted` 是 `w_e * RMS(E_e(h))`，`gate` 是 `w_e`。

## 目标与边界

目标是标准化 live-7 action 空间中的 `RMS(x0-x10)`。这是从初始 flow
noise 到最终 chunk 的总位移，不是 `RMS(x_d-x10)` 剩余修正，也不是
`RMS(x_(d+1)-x_d)` 单步更新。源 NPZ 没有中间 `x_d` 或专家输出向量。

## 对齐校正

源 NPZ 的 `final_actions_standardized` 实际是 `action/std`。本分析从每个
source run 读取 normalization mean，修复为 `(action-mean)/std`，并逐
episode 复核 task/state/seed/action 对齐。

| task | seed grid | action/std max error | corrected x10 max error |
|---|---:|---:|---:|
| goal-middle | yes | 1.19e-07 | 1.19e-07 |
| goal-top | yes | 1.19e-07 | 1.19e-07 |
| long-t08 | yes | 1.19e-07 | 1.19e-07 |
| spatial-ramekin | yes | 1.19e-07 | 1.19e-07 |
| spatial-stove | yes | 5.95e-08 | 5.95e-08 |

x0 按 `default_rng(seed).standard_normal((10,24))[:, :7]` 重建；轴为
`seed, action_token, live_dim`，shape `[32, 10, 7]`。

## 免费 baseline

总位移本身可被不读取 MoE 的量强预测。下表在每个 task/state 的 32 个 exact-seed
候选内算 Spearman，再对 state、task 等权平均；这里只报告描述性 rho。

| baseline | macro rho | per-task rho |
|---|---:|---|
| RMS(x0) | 0.666 | +0.6532 / +0.6583 / +0.7688 / +0.6703 / +0.5796 |
| exact-seed task-local state-LOO target template | 0.971 | +0.9612 / +0.9893 / +0.9563 / +0.9727 / +0.9778 |

第二个 baseline 对每个 task/state/seed 使用同任务、同 seed 的其他 15 states
目标均值。它使用其他 state 的 target 标签，是严格的 nuisance control，不是部署时
可免费获得的剪枝信号；约 0.971 的 rho 说明必须先去掉 seed-template 才能谈 MoE 增量。

## 全搜索族

机制对齐的搜索族为 39 个 feature × 3 endpoints = 117 项：raw / weighted /
gate 在 10 轮的值，以及三者在十轮持续选中同一 expert 时的 slope / rebound /
curvature。5000 次共同 seed-column 置换对 117 项做全局双侧 maxT。

| endpoint | maxT-significant |
|---|---:|
| action_eccentricity | 0 |
| noise_to_final_correction | 4 |
| success | 0 |

通过全局 maxT 且五任务同向的总位移关系：

| feature | macro concordance | per-task concordance | global p |
|---|---:|---|---:|
| weighted.rebound | +0.1090 | +0.1241 / +0.1137 / +0.1414 / +0.1159 / +0.0497 | 0.0056 |
| raw.d8 | -0.0394 | -0.0396 / -0.0458 / -0.0225 / -0.0410 / -0.0482 | 0.0056 |
| raw.d6 | -0.0524 | -0.0346 / -0.0438 / -0.0266 / -0.0776 / -0.0792 | 0.0182 |
| raw.d7 | -0.0429 | -0.0185 / -0.0470 / -0.0159 / -0.0614 / -0.0718 | 0.0366 |

同轮 gate-only 对照不复现 raw d6-d8 关系：

| feature | macro concordance | five-task direction | global p |
|---|---:|---:|---:|
| gate.d6 | -0.0026 | no | 1.0000 |
| gate.d7 | +0.0021 | no | 1.0000 |
| gate.d8 | +0.0027 | no | 1.0000 |
| gate.rebound | +0.0383 | no | 0.9836 |

action eccentricity 与 success 均无全局校正后信号。`goal-middle` 为
512/512 全成功，因此 success 的五任务共性在定义上不可检验。

## 冻结 State 切分

四个 feature 是在全 state 事后探索后冻结的。以下切分只检验 state 集中性与
稳定性，不是未见数据上的前瞻确认。每个 split 用 20,000 次共同 seed-column
置换，并在四个冻结方向内做单侧 maxT。

### contiguous_discovery_0_7

| feature | macro | per-task | five-task direction | maxT4 p |
|---|---:|---|---:|---:|
| raw.d6 | -0.0538 | -0.0375 / -0.0465 / -0.0279 / -0.0795 / -0.0775 | yes | 0.00035 |
| raw.d7 | -0.0424 | -0.0238 / -0.0458 / -0.0129 / -0.0657 / -0.0638 | yes | 0.00080 |
| raw.d8 | -0.0401 | -0.0406 / -0.0559 / -0.0211 / -0.0412 / -0.0416 | yes | 0.00020 |
| weighted.rebound | +0.1157 | +0.1331 / +0.1225 / +0.1497 / +0.1132 / +0.0598 | yes | 0.00015 |

### contiguous_validation_8_15

| feature | macro | per-task | five-task direction | maxT4 p |
|---|---:|---|---:|---:|
| raw.d6 | -0.0510 | -0.0316 / -0.0412 / -0.0253 / -0.0757 / -0.0809 | yes | 0.00045 |
| raw.d7 | -0.0434 | -0.0133 / -0.0481 / -0.0188 / -0.0571 / -0.0798 | yes | 0.00070 |
| raw.d8 | -0.0388 | -0.0386 / -0.0356 / -0.0240 / -0.0408 / -0.0547 | yes | 0.00010 |
| weighted.rebound | +0.1023 | +0.1151 / +0.1050 / +0.1331 / +0.1186 / +0.0397 | yes | 0.00040 |

### even_scene_axis_sensitivity

| feature | macro | per-task | five-task direction | maxT4 p |
|---|---:|---|---:|---:|
| raw.d6 | -0.0512 | -0.0303 / -0.0476 / -0.0267 / -0.0786 / -0.0726 | yes | 0.00020 |
| raw.d7 | -0.0417 | -0.0065 / -0.0434 / -0.0252 / -0.0625 / -0.0709 | yes | 0.00065 |
| raw.d8 | -0.0431 | -0.0414 / -0.0462 / -0.0282 / -0.0495 / -0.0504 | yes | 0.00005 |
| weighted.rebound | +0.1035 | +0.0778 / +0.1107 / +0.1537 / +0.1257 / +0.0497 | yes | 0.00005 |

### odd_scene_axis_sensitivity

| feature | macro | per-task | five-task direction | maxT4 p |
|---|---:|---|---:|---:|
| raw.d6 | -0.0536 | -0.0389 / -0.0401 / -0.0265 / -0.0766 / -0.0858 | yes | 0.00015 |
| raw.d7 | -0.0441 | -0.0305 / -0.0505 / -0.0065 / -0.0603 / -0.0727 | yes | 0.00045 |
| raw.d8 | -0.0357 | -0.0378 / -0.0453 / -0.0169 / -0.0324 / -0.0460 | yes | 0.00020 |
| weighted.rebound | +0.1144 | +0.1704 / +0.1167 / +0.1292 / +0.1061 / +0.0498 | yes | 0.00010 |

## Discovery-template → Validation-residual 严格控制

只用 discovery scene axes `0..7` 构造每个 `task + exact seed` 的 target 均值模板；
在 validation axes `8..15` 上令 `residual = target - template`。随后只检验四个
冻结 feature，用 20,000 次共同 seed-column permutation 做单侧 maxT4。

| feature | residual macro | per-task | five-task direction | ranking advantage | maxT4 p |
|---|---:|---|---:|---:|---:|
| raw.d6 | -0.01857 | -0.0199 / -0.0116 / -0.0308 / -0.0269 / -0.0038 | yes | +0.93 pp | 0.00275 |
| raw.d7 | -0.01896 | -0.0158 / -0.0328 / -0.0187 / -0.0270 / -0.0005 | yes | +0.95 pp | 0.00195 |
| raw.d8 | -0.01468 | -0.0151 / -0.0210 / -0.0229 / -0.0085 / -0.0059 | yes | +0.73 pp | 0.01130 |
| weighted.rebound | +0.02546 | +0.0325 / +0.0040 / +0.0566 / +0.0354 / -0.0012 | no | +1.27 pp | 0.16764 |

raw d6/d7/d8 在 validation residual 上仍为五任务负向，但 concordance 只有
约 -0.015 到 -0.019：换成直观排序准确率，只比 50% 高约 0.73–0.95 个百分点。
weighted rebound 不再五任务同向，也未通过 maxT4。这个量级不能支持候选剪枝。

该 residual 控制的 template 构造与 validation state 严格分离；但四个 feature
来自更早的全 state 探索，因此仍不是完全未触碰数据上的前瞻确认。

## 裁决

1. 原始总位移高度受 x0/seed template 支配；不控制它时的 MoE 关联会被夸大。
2. 严格 discovery-template → validation-residual 后，raw d6-d8 仍保持五任务负向；
   weighted rebound 不通过，因此撤回它作为稳定冻结信号。
3. raw 的增量只有约 0.7–0.95 pp pairwise ranking advantage，不能用于剪枝、
   early stop 或动作承诺判断。
4. 结果仍是 selected-only、离线 fp16-hidden 重建后的关联；固定选中集合会带来
   selection conditioning，也不能解释为 expert 幅度的因果作用。
