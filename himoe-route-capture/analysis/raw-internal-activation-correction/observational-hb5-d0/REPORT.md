# HB5/d0 expert-MLP 内部激活审计

## 先区分三层

1. **Router weight**：选择和混合专家的权重，不是专家激活。
2. **`down_proj` 前内部激活**：`m_e = SiLU(gate_proj_e(h)) * up_proj_e(h)`；这是本实验新增测量的量。
3. **`down_proj` 后专家输出**：`E_e(h) = down_proj_e(m_e)`；v3 原先保存的是这个量，不能再把它简称为 `m`。

主幅值是 `||m_e||₂` 的绝对 checkpoint 单位，不乘也不除 router weight，也不除 hidden、shared、routed 或 post-MoE。RMS 仅是固定宽度恒等换算 `L2/sqrt(1024)`。训练折标准化只用于核岭回归的数值条件，不改变下表原始量级。

不同专家的 pre-down 神经元没有可识别的共同坐标轴，因此不能对 `m_e` 跨专家算向量夹角或 cancellation。主块只保留每个专家内部的 raw L2，并对四个值排序；weighted mean/dispersion 只属于 secondary 描述，不进入主判定。

## 原始量级

| task | raw L2 | raw L1 | RMS=L2/sqrt(1024) | active fraction | gate closed | gate open |
|---|---:|---:|---:|---:|---:|---:|
| libero_10:8 | 7.994204 | 143.596634 | 0.249819 | 0.981451 | 0.000000 | 0.000000 |
| libero_goal:0 | 7.662759 | 137.234711 | 0.239461 | 0.980555 | 0.000000 | 0.000000 |
| libero_goal:3 | 7.586528 | 136.128372 | 0.237079 | 0.980617 | 0.000000 | 0.000000 |
| libero_spatial:5 | 7.983345 | 144.945404 | 0.249480 | 0.981573 | 0.000000 | 0.000000 |
| libero_spatial:7 | 7.870539 | 142.671631 | 0.245954 | 0.981475 | 0.000000 | 0.000000 |

## 主结果

主目标是 d0 之后的完整 `flatten10x7(x10-x1)` remaining correction。每个任务都用 state 与 candidate 双重不相交的 nested CV。B1 的八个原始 family 用完整坐标构造等权 additive linear kernel，没有 CountSketch。

主块是每 token 四个排序后的 raw `||m_e||₂`，共 exact40；它完全不使用 router weight。matched control 对 post-down `||E_e(h)||₂` 做同样的 exact40 构造。

| task | B1 RMSE | + post-down exact40 | + pre-down exact40 | rel. gain vs B1 | correction geometry rho |
|---|---:|---:|---:|---:|---:|
| libero_10:8 | 0.894164 | 0.902942 | 0.904017 | -1.102% | 0.472873 |
| libero_goal:0 | 0.884514 | 0.888142 | 0.888286 | -0.426% | 0.391122 |
| libero_goal:3 | 0.885075 | 0.888517 | 0.889542 | -0.505% | 0.366382 |
| libero_spatial:5 | 0.890350 | 0.892610 | 0.893479 | -0.351% | 0.548998 |
| libero_spatial:7 | 0.862398 | 0.865595 | 0.865406 | -0.349% | 0.496033 |

## 判定

equal-task macro relative RMSE gain vs exact B1: -0.547% (fixed-crossfit conditional two-axis 95% interval [-0.735%, -0.388%]); task direction 0/5; better than matched post-down control in 1/5 tasks.

这是探索性 observational screen，不是 pruning gate。future path energy 没有进入主判定，因为它几乎由已观察到的 d0 velocity 决定。这里只测 HB5/d0，不能据此声称模型在去噪中‘越来越确定’。

重构使用 v3 保存的 fp16 HB input 和 bf16 checkpoint，在 float32 中计算。报告中的 post-down 重构误差验证专家 ID、权重文件和算子路径匹配；它不可能与原始 bf16 runtime input bit-exact。
