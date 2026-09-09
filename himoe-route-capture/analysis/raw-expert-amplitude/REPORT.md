# Gate 加权分支幅度审计（已被更正）

> **方法范围更正：** 本报告测量的是 gate 加权后的 branch 幅度，不是每个
> selected expert 在乘 gate 前的原始输出范数。它不能回答未加权 raw expert
> activation 或动作承诺问题；相关外推已撤回。

固定单元为 HB5 / denoise 0；五个任务均为 16 个同观测池 × 32 个共同 seed。
这里的 expert_mass、routed_rms 均保留绝对 RMS 单位，不除以 hidden、shared、
routed 或 post-MoE 幅度。1024 维下 L2 恰为 RMS×32，因此不会改变候选排序。
回归器只在训练折内做 StandardScaler，以保证数值条件，不改变实验变量定义。

## 原始量级

| task | expert mass | routed RMS | shared RMS | absolute swap delta | mass pool-CV |
|---|---:|---:|---:|---:|---:|
| goal-middle | 0.1703 | 0.0929 | 0.1961 | 0.1179 | 0.024 |
| goal-top | 0.1675 | 0.0913 | 0.1950 | 0.1163 | 0.023 |
| long-t08 | 0.1784 | 0.0973 | 0.1915 | 0.1222 | 0.024 |
| spatial-ramekin | 0.1835 | 0.0978 | 0.1837 | 0.1263 | 0.017 |
| spatial-stove | 0.1823 | 0.0973 | 0.1820 | 0.1252 | 0.015 |

## 绝对专家替换敏感度

目标是 rank-5–8 替换 top-4 后的绝对 routed-vector RMS 变化；不使用任何分母。
相关系数先在每个同观测 K=32 池内计算，再对任务和池等权平均。
expert 两项的 p 值以共同 seed 置换做 maxT 校正；control 的 p 值未纳入该发现族。

| predictor | Pearson | Spearman | hierarchical 95% CI | permutation p |
|---|---:|---:|---:|---:|
| expert_mass | +0.624 | +0.587 | [+0.534, +0.708] | 0.0002 |
| routed_rms | +0.637 | +0.602 | [+0.572, +0.710] | 0.0002 |
| input_rms | +0.131 | +0.099 | [-0.102, +0.348] | 0.0960 |
| shared_rms | +0.136 | +0.108 | [+0.029, +0.246] | 0.0786 |

按 seed 留出的线性读出：input/shared/router 控制的池内 Pearson 为 0.320，加入原始 expert mass/routed RMS 后为 0.587，增量 +0.266，95% CI [+0.153, +0.377]。
该结果与目标共享同一 block 的专家输出，只能解释为即时机械敏感度。

## 最终动作盆地

| task | noise baseline | + expert raw | + control raw | expert Δ |
|---|---:|---:|---:|---:|
| goal-middle | 0.719 | 0.710 | 0.738 | -0.009 |
| goal-top | 0.744 | 0.700 | 0.603 | -0.044 |
| long-t08 | 0.584 | 0.554 | 0.620 | -0.030 |
| spatial-ramekin | 0.669 | 0.667 | 0.522 | -0.002 |
| spatial-stove | 0.743 | 0.738 | 0.485 | -0.005 |

五任务宏平均：baseline 0.692，+expert 0.674，+control 0.594。
expert 增量 -0.018，95% CI [-0.035, -0.003]；expert 相对 control +0.080，95% CI [-0.020, +0.181]。

## 最终成功（探索性）

| task | baseline | + expert raw | + control raw | expert Δ |
|---|---:|---:|---:|---:|
| goal-middle | NA | NA | NA | NA |
| goal-top | 0.617 | 0.619 | 0.701 | +0.002 |
| long-t08 | 0.426 | 0.545 | 0.517 | +0.118 |
| spatial-ramekin | 0.399 | 0.560 | 0.395 | +0.161 |
| spatial-stove | 0.541 | 0.612 | 0.472 | +0.070 |

四个可估计任务宏平均：baseline 0.489，+expert 0.581，+control 0.511；expert 增量 +0.092，95% CI [-0.013, +0.206]。
未纳入任务：goal-middle（success label has only one class）。
可计算池数：goal-top=7, long-t08=13, spatial-ramekin=8, spatial-stove=12。
但四个任务的 expert 模型 log loss 都比 baseline 更差；宏平均 0.2231 → 0.2333。

## 去噪时钟检查

long-t08 的十轮原始幅度中，单独由 round 解释的变异比例：

- expert_mass: 55.0%
- routed_rms: 59.2%
- input_rms: 91.4%
- shared_rms: 88.6%

## 已撤回的外推

此前把替换相关称为 Level-1 信号、把 median near-pair 称为动作盆地，以及据此
否定动作承诺的表述均已撤回。最窄的有效结果只是：固定线性 readout 下，
HB5/d0 的 gate 加权聚合量没有改善“最终动作 pair 距离是否低于池中位数”的
分类；这不等价于 per-expert raw norm 或时间承诺。

## 边界

- 这是从 fp16 hidden 与 checkpoint 离线重算的 HB block 代理，不是 runtime-exact 输出。
- 数据没有逐轮 flow latent，因此盆地目标是最终 action chunk 相似性，不是剩余动作修正量。
- rank-5–8 替换只计算即时 block 输出，没有继续完成后续 flow 或环境 rollout。
- success 距离 MoE block 很远，只能作探索性关联，不能解释为动作已经承诺成败。

完整数值见 `summary.json`，候选级原始数组见 `candidate_raw_amplitudes.npz`。
