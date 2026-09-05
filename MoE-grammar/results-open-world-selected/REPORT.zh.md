# MoE-only 开放集观察器：模型选择与结论

## 健康-only 模型选择

K 只按 30,904 条 cross-fit held-out success 的 q4+ 下一 chord NLL 选择；选择过程不读取 failure 或 stasis 标签。

| phase states | prefix bits/phenotype | prefix-clock |
|---:|---:|---:|
| 4 | 0.0331 | -0.0052 |
| 8 | 0.0141 | -0.0126 |
| 12 | 0.0065 | -0.0165 |

选择 K=12。负的 prefix-clock 说明完整已观测前缀确实改善健康下一词预测，但这不自动等于 Trap 可检测。

## 严格 MoE-only 在线读数

固定汇总方法为 `grammar_ewma_persistent`；它不是按 failure outcome 选择的。

| q | task/state AUC | 95% CI | healthy FPR | failure recall |
|---:|---:|---:|---:|---:|
| 3 | 0.505 | [0.487, 0.523] | 0.048 | 0.062 |
| 7 | 0.529 | [0.504, 0.553] | 0.050 | 0.069 |
| 12 | 0.593 | [0.558, 0.631] | 0.051 | 0.300 |

q3 与随机水平不可区分；q7 只有很弱信号；q12 才出现中等但偏晚的关联。Full-40 阳性只是 eventual failure，因此这些数字仍不能证明早于第一个错误动作。

## 物理 Stasis Onset

| 方法 | success FPR | onset 前 recall | onset+3 recall |
|---|---:|---:|---:|
| grammar_instant | 0.074 | 0.030 | 0.030 |
| grammar_learned_persistent | 0.054 | 0.056 | 0.081 |
| grammar_dwell_persistent | 0.108 | 0.117 | 0.142 |
| grammar_ewma_persistent | 0.071 | 0.102 | 0.117 |
| known_stasis | 0.095 | 0.056 | 0.076 |

## Task-conditioned 诊断

加入已知 task ID 后的结果只用于定位 nuisance variation，不属于严格 MoE-only 部署值。

| q | task/state AUC | healthy FPR | failure recall |
|---:|---:|---:|---:|
| 3 | 0.510 | 0.066 | 0.119 |
| 7 | 0.563 | 0.066 | 0.160 |
| 12 | 0.610 | 0.070 | 0.367 |

## 判定

当前证据支持“成功 routing 存在可预测的完整前缀结构”，但不支持“该结构已经构成可靠的开放世界早检器”。Unknown clusters 只是待物理审计的 phenotype candidates；漏报是 observer-unseen，不等于已经证明 MoE-silent。
