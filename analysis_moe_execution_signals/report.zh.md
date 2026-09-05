# HiMoE actual-execution F0-F3 test

正式统计使用 2000 次组级 sign-flip / bootstrap；全部 feature 在读取 Trap 标签前固定。

## 核心结论

1. 新信息首次稳定出现在 F2 的跨 query 实际 dispatch，而不是 F1 的另一种 entropy/margin。`query_support_churn` 与 `query_exec_churn` 在 aggregate Trap onset 前 4 个 query 下降，对旧 F0 loop 信号做秩残差后仍在 A/B 同方向且两边 maxT 通过。
2. 这个增量由 static 主导：static 的 lead=-4/-2 强复现；loop 没有任何新信号在 F0-residual 后双语料通过。因此不能把它表述成通用的『Trap 前 support 更不稳定』；观察到的是 static 前执行 support 提前冻结。
3. F1 的 tail mass、Top-4 执行熵和 Top-4/5 margin 在 loop lead=-2 有大效应，但 B 的统一 maxT 未通过，且 F0-residual 后消失。它们目前是原 soft-routing precursor 的重表达。
4. 逐 token 单点存在显著差异，但三个预注册 token-slope 都未跨语料确认，所以现有数据不支持按 action slot 推导 safe prefix。

## Capture audit

| corpus | rows | fp16 Top-4 mismatch | negative boundary | zero boundary | selected mass range |
|---|---:|---:|---:|---:|---:|
| A | 16180 | 0.000% | 0.000% | 26.419% | 0.1295--0.9514 |
| B | 22883 | 0.000% | 0.000% | 28.386% | 0.1285--0.9504 |

`negative actual boundary` 表示 fp16 full probability 与运行时 stored Top-4 不再严格一致；所有 support/churn 因此只使用 stored IDs，绝不从 fp16 概率重算 Top-4。

## Information increment at aggregate Trap / static onset

| event | signal | lead | direction A/B | residual det AUC A/B | maxT p A/B |
|---|---|---:|---|---:|---:|
| static | `early_state_query_churn` | -4 | event_low/event_low | 0.806/0.771 | 0.0160/0.0010 |
| static | `early_state_query_churn` | -2 | event_low/event_low | 0.852/0.747 | 0.0030/0.0150 |
| trap | `query_exec_churn` | -4 | event_low/event_low | 0.732/0.726 | 0.0035/0.0210 |
| static | `query_exec_churn` | -4 | event_low/event_low | 0.767/0.762 | 0.0405/0.0030 |
| static | `query_exec_churn` | -2 | event_low/event_low | 0.865/0.772 | 0.0010/0.0010 |
| trap | `query_support_churn` | -4 | event_low/event_low | 0.738/0.733 | 0.0020/0.0160 |
| static | `query_support_churn` | -4 | event_low/event_low | 0.767/0.766 | 0.0405/0.0030 |
| static | `query_support_churn` | -2 | event_low/event_low | 0.863/0.768 | 0.0010/0.0030 |

所有通过单元的方向都是 `event_low`：事件轨迹的跨 query actual support/weight 变化更小。

## Representation redundancy

| corpus | support vs sparse-weight churn rho | tail mass vs full entropy rho | exec entropy vs full entropy rho | p45 vs p12 margin rho |
|---|---:|---:|---:|---:|
| A | 0.995 | 0.938 | 0.554 | 0.241 |
| B | 0.995 | 0.919 | 0.593 | 0.333 |

## Best cross-corpus loop precursors (raw)

| signal | lead | direction A/B | det AUC A/B | maxT p A/B |
|---|---:|---|---:|---:|
| `tail_mass` | -2 | event_low/event_low | 0.900/0.810 | 0.0090/0.1004 |
| `exec_entropy` | -2 | event_low/event_low | 0.866/0.717 | 0.0195/0.3758 |
| `top45_margin` | -2 | event_high/event_high | 0.814/0.699 | 0.0435/0.5047 |
| `tail_mass` | -4 | event_low/event_low | 0.769/0.659 | 0.1114/0.7546 |
| `early_state_query_churn` | -6 | event_high/event_high | 0.635/0.625 | 0.7931/0.9405 |
| `top45_margin` | -4 | event_high/event_high | 0.684/0.624 | 0.3848/0.9495 |
| `all_layer_exec_churn` | -4 | event_low/event_low | 0.599/0.654 | 0.9960/0.7811 |
| `exec_entropy` | -4 | event_low/event_low | 0.701/0.586 | 0.3063/1.0000 |

## Increment beyond frozen F0 loop signals

这里先在同 snapshot/init-state、同绝对 query 内，将每个新信号对冻结的 `late_flow_volatility` 与 `route_acceleration` 做无标签秩残差，再进行相同 matched AUC。

| signal | lead | direction A/B | residual det AUC A/B | maxT p A/B |
|---|---:|---|---:|---:|
| `query_exec_churn` | -8 | event_high/event_high | 0.630/0.658 | 0.6392/0.3548 |
| `query_support_churn` | -8 | event_high/event_high | 0.628/0.657 | 0.6462/0.3653 |
| `early_state_query_churn` | -6 | event_high/event_high | 0.632/0.617 | 0.5867/0.8356 |
| `all_layer_exec_churn` | -4 | event_low/event_low | 0.599/0.636 | 0.9595/0.6187 |
| `tail_mass` | -6 | event_low/event_low | 0.599/0.604 | 0.9605/0.9015 |
| `all_layer_support_churn` | -4 | event_low/event_low | 0.595/0.649 | 0.9765/0.4593 |
| `tail_mass` | -2 | event_low/event_low | 0.689/0.594 | 0.1939/0.9555 |
| `late_flow_support_churn` | -4 | event_low/event_low | 0.601/0.590 | 0.9480/0.9680 |

## Safe-prefix localization audit

| metric | mean abs(AUC-0.5) early/late A | early/late B | best token A/B | family p A/B |
|---|---:|---:|---:|---:|
| `top45_margin` | 0.145/0.157 | 0.109/0.134 | 10/9 | 0.0875/0.2854 |
| `tail_mass` | 0.321/0.309 | 0.248/0.303 | 10/9 | 0.0005/0.0065 |
| `exec_entropy` | 0.204/0.223 | 0.184/0.214 | 9/8 | 0.0035/0.0325 |
| `late_flow_support_churn` | 0.052/0.038 | 0.076/0.079 | 4/8 | 0.4068/0.5817 |
| `late_flow_exec_churn` | 0.064/0.040 | 0.068/0.081 | 4/8 | 0.3378/0.4843 |
| `query_support_churn` | 0.025/0.071 | 0.188/0.205 | 9/10 | 0.4353/0.0905 |
| `query_exec_churn` | 0.024/0.074 | 0.188/0.203 | 9/10 | 0.4578/0.0965 |

## State-to-action propagation (label-free)

| corpus | state leads action by queries | rho | episode-bootstrap 95% CI |
|---|---:|---:|---:|
| A | 0 | 0.494 | [0.468, 0.520] |
| A | 1 | 0.421 | [0.395, 0.445] |
| A | 2 | 0.291 | [0.261, 0.320] |
| A | 3 | 0.155 | [0.121, 0.191] |
| A | 4 | 0.082 | [0.043, 0.122] |
| B | 0 | 0.302 | [0.277, 0.325] |
| B | 1 | 0.246 | [0.222, 0.269] |
| B | 2 | 0.176 | [0.151, 0.201] |
| B | 3 | 0.109 | [0.082, 0.135] |
| B | 4 | 0.077 | [0.051, 0.104] |

## Decision gate

- loop 中跨语料、同方向、两边 maxT<=0.05 的 F0-residual 新 precursor：0 个。
- aggregate Trap/static 中通过相同 gate 的新单元：8 个。
- 描述上后五个 token 在 A/B 都强于前五个的指标：4 个；这不是显著性 gate。
- lead=-2 跨语料确认的预注册 token-slope 信号：0 个。
- 观察数据不能估计 safe-prefix 或 reroute 的干预 ATE；只有定位结果复现后，才值得启动 paired snapshot rollout。
- Authority/cancellation/functional geometry 不在现有 routes.zarr 中；它们需要 runtime-exact rich capture。

## Artifacts

- `onset_alignment.csv`: F0-F3 raw 与 F0-residual 的 Trap/loop/static matched tests。
- `onset_group_effects.csv`: maxT 所用的逐组效应。
- `token_position_auc.csv`: loop lead=-2 的逐 action-token 检验。
- `propagation.csv`: early-state 到 late-action 的跨 query 相关。
- `f0_f3_loop_matrix.png` / `token_position_loop_lead2.png`: 方向与位置图。
