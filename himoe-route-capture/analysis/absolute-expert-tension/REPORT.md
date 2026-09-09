# 保留 gate 的未归一化专家张力

本报告只回答一个问题：去掉 `D/C` 的尺度归一化、但保留真实 top-4 gate 权重后，
selected experts 的输出几何如何变化。它不把输出范数称为专家内部状态或置信度。

## 定义

对同一个 action token，令 `v_e=E_e(h)`、`alpha_e` 为归一化后的 top-4 gate 权重、
`r=sum_e alpha_e v_e`、`rho=RMS(r)`：

- `s1=sum_e alpha_e RMS(v_e)`
- `s2=sum_e alpha_e RMS(v_e)^2`
- `D_abs=sqrt(max(s2-rho^2, 0))`
- `C_abs=s1-rho`

这才是“去掉归一化但保留 gate”的定义。与比例量的精确关系是
`D_abs=sqrt(s2*D)`、`C_abs=s1*C`。下面先对 10 个 action token 取均值。

## 绝对量轨迹

| task | D_abs d0 | D_abs d9 | delta | down | C_abs d0 | C_abs d9 | delta | down |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal-middle | 0.14664 | 0.13115 | -0.01542 | 100.0% | 0.07782 | 0.07193 | -0.00609 | 100.0% |
| goal-top | 0.14399 | 0.13076 | -0.01344 | 100.0% | 0.07707 | 0.07220 | -0.00485 | 99.0% |
| long-t08 | 0.15152 | 0.14408 | -0.00738 | 81.8% | 0.08082 | 0.08052 | -0.00020 | 52.1% |
| spatial-ramekin | 0.15555 | 0.16992 | +0.01440 | 1.0% | 0.08521 | 0.09336 | +0.00823 | 0.6% |
| spatial-stove | 0.15454 | 0.16822 | +0.01412 | 5.1% | 0.08469 | 0.09407 | +0.00946 | 3.3% |

Goal 的绝对张力下降，Spatial 的绝对张力上升，Long 接近持平；因此 absolute
量没有五任务同向的“越来越肯定”结论。normalized `D/C` 的中位数则五任务均上升，
它表达的是张力占总 expert scale 的比例增加，并不与 absolute 量同义。

## 同状态 K=32 离散度

数值为每个状态内 32 个候选的 population SD，再取 16 个状态的中位数。ratio<1 表示候选间收缩。

| task | D_abs SD d0 | d9 | ratio | C_abs SD d0 | d9 | ratio | D ratio | C ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| goal-middle | 0.003602 | 0.001969 | 0.547 | 0.001907 | 0.001085 | 0.569 | 0.554 | 0.612 |
| goal-top | 0.003490 | 0.002166 | 0.621 | 0.001811 | 0.001223 | 0.675 | 0.542 | 0.604 |
| long-t08 | 0.003442 | 0.003213 | 0.933 | 0.001591 | 0.001599 | 1.005 | 0.562 | 0.547 |
| spatial-ramekin | 0.002520 | 0.002253 | 0.894 | 0.001443 | 0.001328 | 0.920 | 0.550 | 0.625 |
| spatial-stove | 0.002202 | 0.002951 | 1.340 | 0.001180 | 0.001750 | 1.484 | 0.735 | 0.721 |

normalized `D/C` 的 K32 离散度五任务都收缩；absolute 量在 spatial-stove 反而扩张。
这更像 hidden/denoise 驱动的候选几何收缩，不能单独归因于某个专家。

## 尺度混杂

下表是在每个同状态 K32 池内算 Spearman，再对 16 个池等权平均。

| task | rho(D_abs,s1) d0 | d9 | rho(C_abs,s1) d0 | d9 |
|---|---:|---:|---:|---:|
| goal-middle | +0.970 | +0.975 | +0.907 | +0.907 |
| goal-top | +0.969 | +0.969 | +0.873 | +0.899 |
| long-t08 | +0.975 | +0.982 | +0.834 | +0.937 |
| spatial-ramekin | +0.970 | +0.972 | +0.821 | +0.873 |
| spatial-stove | +0.971 | +0.979 | +0.797 | +0.924 |

`D_abs` 与总 expert scale `s1` 的同池相关约 0.97，`C_abs` 也很高。
这不是计算错误，而是 absolute 定义必然保留尺度；所以它适合诊断实际张力幅度，
不适合作为独立的 expert-specific 或 confidence 证据。

## Success 探索

下表 AUC>0.5 表示分数越高越容易成功。goal-middle 没有成败混合池，因此不进入 macro。
这些是事后 effect size，没有 permutation、multiple-testing 校正或确认集。

| score | feature | macro AUC | goal-top | long | ramekin | stove |
|---|---|---:|---:|---:|---:|---:|
| d_abs | d0 | 0.586 | 0.619 | 0.557 | 0.557 | 0.611 |
| d_abs | d9 | 0.455 | 0.364 | 0.513 | 0.449 | 0.495 |
| d_abs | linear_slope_d0_d9 | 0.437 | 0.337 | 0.460 | 0.486 | 0.462 |
| c_abs | d0 | 0.587 | 0.647 | 0.547 | 0.541 | 0.614 |
| c_abs | d9 | 0.423 | 0.312 | 0.490 | 0.412 | 0.476 |
| c_abs | linear_slope_d0_d9 | 0.410 | 0.320 | 0.459 | 0.427 | 0.436 |

d0 的约 0.59 macro AUC 与 raw amplitude 的既有结果几乎相同，且随后轮次方向改变。
结合上面的尺度相关，它不是新的成功路由结论。

## 边界

- expert IDs 和 gate 权重来自捕获；expert outputs 来自 fp16 hidden 的离线重建。
- 这里只分析 HB5、selected top-4 和第一个控制状态；没有 all-32 expert 对照。
- `D_abs/C_abs` 是 selected-expert 输出几何，不是持久的专家内部状态。
- 本报告是探索性审计，不提供显著性声明。

候选级数组见 `arrays.npz`，完整数值见 `summary.json`。
