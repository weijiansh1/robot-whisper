# 完整 rollout 的逐 chunk MoE 成败 AUC

## 结论

逐 chunk 分析覆盖 q0-q46。q0-q34 的 512 条分支全部存活，因此这里的 AUC 不可能靠 episode 长度或 remaining-time 得到。
完整样本上，保留 token 位置并加入跨 chunk 变化的探索性最高点是 q34：同初态折外 AUC=0.885 [0.813,0.951]。
更严格的 seed 拆半先在 1000-1015 中选择 q34，再冻结到 1016-1031：验证 AUC=0.870 [0.752,0.963]；错配 sibling 平均 AUC=0.538。
位置解析不是始终有用：q24 保留 token 位置相对 token 均值增加 AUC +0.218 [+0.040,+0.398]；q34 的增量则为 -0.016 [-0.054,+0.028]。

AUC 的主口径是在每个初态内部计算后再平均：0.5 表示同一初态的成功/失败 seed 无法排序。 pooled AUC 另列，只用于观察初态难度；不能拿它代替 branch 级成败信息。

## 数据与防泄漏口径

- 512 条 rollout：296 成功、216 失败；16 个初态，每个 32 seeds。
- 最短轨迹有 35 个 query，所以 q0-q34 是完整共同前缀。
- q35-q46 只评价当时仍在运行的分支，明确标成 risk set。
- 每个 q 单独训练和评价；模型输入没有 q 编号、轨迹长度、剩余时间、terminal 标记、动作、hidden state 或 simulator state。
- heldout-seed 为 8 折，每折完整留出 4 个 noise seeds；特征坐标只在训练折内选择。
- shifted sibling 把测试分支的路由换成同初态另一 seed 的路由，但保留原标签。
- 若错误地把 q0-q46 风险集混在一起，单用 query 编号就有 AUC=0.581；本报告没有使用这个泄漏口径。

## 逐 chunk AUC

| q | cohort | model | n | mixed states | within-state AUC | CI low | CI high | pooled AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | common | aggregate_history | 512 | 13 | 0.540 | 0.428 | 0.651 | 0.876 |
| 0 | common | token_current | 512 | 13 | 0.602 | 0.475 | 0.709 | 0.876 |
| 0 | common | token_history | 512 | 13 | 0.602 | 0.470 | 0.708 | 0.876 |
| 4 | common | aggregate_history | 512 | 13 | 0.432 | 0.315 | 0.545 | 0.874 |
| 4 | common | token_current | 512 | 13 | 0.538 | 0.445 | 0.625 | 0.871 |
| 4 | common | token_history | 512 | 13 | 0.494 | 0.426 | 0.568 | 0.867 |
| 8 | common | aggregate_history | 512 | 13 | 0.625 | 0.522 | 0.724 | 0.895 |
| 8 | common | token_current | 512 | 13 | 0.448 | 0.331 | 0.577 | 0.855 |
| 8 | common | token_history | 512 | 13 | 0.420 | 0.290 | 0.559 | 0.856 |
| 12 | common | aggregate_history | 512 | 13 | 0.559 | 0.422 | 0.679 | 0.888 |
| 12 | common | token_current | 512 | 13 | 0.538 | 0.459 | 0.612 | 0.874 |
| 12 | common | token_history | 512 | 13 | 0.474 | 0.349 | 0.592 | 0.875 |
| 16 | common | aggregate_history | 512 | 13 | 0.551 | 0.423 | 0.662 | 0.894 |
| 16 | common | token_current | 512 | 13 | 0.646 | 0.509 | 0.761 | 0.895 |
| 16 | common | token_history | 512 | 13 | 0.570 | 0.445 | 0.673 | 0.894 |
| 20 | common | aggregate_history | 512 | 13 | 0.654 | 0.538 | 0.773 | 0.881 |
| 20 | common | token_current | 512 | 13 | 0.479 | 0.386 | 0.583 | 0.863 |
| 20 | common | token_history | 512 | 13 | 0.610 | 0.478 | 0.741 | 0.876 |
| 24 | common | aggregate_history | 512 | 13 | 0.447 | 0.312 | 0.573 | 0.869 |
| 24 | common | token_current | 512 | 13 | 0.710 | 0.614 | 0.807 | 0.897 |
| 24 | common | token_history | 512 | 13 | 0.665 | 0.580 | 0.755 | 0.893 |
| 28 | common | aggregate_history | 512 | 13 | 0.614 | 0.500 | 0.732 | 0.898 |
| 28 | common | token_current | 512 | 13 | 0.588 | 0.459 | 0.701 | 0.887 |
| 28 | common | token_history | 512 | 13 | 0.639 | 0.541 | 0.741 | 0.897 |
| 32 | common | aggregate_history | 512 | 13 | 0.761 | 0.607 | 0.878 | 0.936 |
| 32 | common | token_current | 512 | 13 | 0.788 | 0.688 | 0.875 | 0.926 |
| 32 | common | token_history | 512 | 13 | 0.765 | 0.614 | 0.883 | 0.932 |
| 34 | common | aggregate_history | 512 | 13 | 0.902 | 0.801 | 0.973 | 0.964 |
| 34 | common | token_current | 512 | 13 | 0.869 | 0.788 | 0.937 | 0.956 |
| 34 | common | token_history | 512 | 13 | 0.885 | 0.813 | 0.951 | 0.961 |
| 36 | risk_set | aggregate_history | 504 | 13 | 0.859 | 0.790 | 0.925 | 0.954 |
| 36 | risk_set | token_current | 504 | 13 | 0.845 | 0.764 | 0.925 | 0.938 |
| 36 | risk_set | token_history | 504 | 13 | 0.872 | 0.809 | 0.936 | 0.951 |
| 38 | risk_set | aggregate_history | 421 | 12 | 0.744 | 0.621 | 0.862 | 0.951 |
| 38 | risk_set | token_current | 421 | 12 | 0.732 | 0.569 | 0.879 | 0.931 |
| 38 | risk_set | token_history | 421 | 12 | 0.790 | 0.670 | 0.903 | 0.951 |
| 40 | risk_set | aggregate_history | 252 | 10 | 0.782 | 0.648 | 0.908 | 0.894 |
| 40 | risk_set | token_current | 252 | 10 | 0.787 | 0.567 | 0.957 | 0.842 |
| 40 | risk_set | token_history | 252 | 10 | 0.738 | 0.575 | 0.893 | 0.868 |
| 42 | risk_set | aggregate_history | 232 | 7 | 0.786 | 0.500 | 1.000 | 0.894 |
| 42 | risk_set | token_current | 232 | 7 | 0.643 | 0.286 | 0.929 | 0.825 |
| 42 | risk_set | token_history | 232 | 7 | 0.714 | 0.429 | 0.929 | 0.870 |
| 44 | risk_set | aggregate_history | 229 | 6 | 0.667 | 0.333 | 1.000 | 0.877 |
| 44 | risk_set | token_current | 229 | 6 | 0.625 | 0.292 | 0.958 | 0.841 |
| 44 | risk_set | token_history | 229 | 6 | 0.583 | 0.250 | 0.917 | 0.855 |
| 46 | risk_set | aggregate_history | 227 | 6 | 0.792 | 0.458 | 1.000 | 0.807 |
| 46 | risk_set | token_current | 227 | 6 | 0.375 | 0.042 | 0.708 | 0.693 |
| 46 | risk_set | token_history | 227 | 6 | 0.354 | 0.042 | 0.688 | 0.695 |

模型含义：`aggregate_history` 把 10 个 action token 平均后使用当前路由与跨 query 变化；`token_current` 保留当前 chunk 的 token 位置；`token_history` 同时保留 token 位置和跨 chunk 路由变化。
在 q30，给逐 token 当前结构加入历史变化的 AUC 增量为 +0.094 [+0.028,+0.157]。

## 严格拆半验证

发现半部的折外曲线选择 q34，发现 AUC=0.908。
冻结特征族每族只选一个训练内坐标，共 8 个。下表效应完全来自未参与选择的后 16 seeds；正值表示失败更高：

| frozen feature | effect SD | CI low | CI high |
| --- | --- | --- | --- |
| aggregate/top1\|back_12_15\|d2\|token_mean | 0.848 | 0.392 | 1.405 |
| position/entropy_centered\|back_12_15\|d9\|t5 | -0.630 | -1.281 | -0.213 |
| aggregate/adjacent_hellinger\|front_2_5\|d8\|token_mean | 0.959 | 0.504 | 1.387 |
| aggregate/top1_switch_rate\|front_2_5\|d9\|token_mean | 0.808 | 0.398 | 1.218 |
| aggregate/top4_turnover\|front_2_5\|d9\|token_mean | 1.292 | 0.801 | 1.888 |
| position/query_hellinger\|back_12_15\|d9\|t1 | -0.904 | -1.478 | -0.374 |
| aggregate/query_top4_turnover\|back_12_15\|d6\|token_mean | -1.065 | -1.617 | -0.573 |
| position/nonlocal_return_hellinger\|back_12_15\|d9\|t1 | -1.409 | -2.206 | -0.799 |

组合方向是：失败分支在前层 d8/d9 的 chunk 内相邻-token Hellinger 和 top-4 turnover 更高，但后层 d9 的 t1 跨-query Hellinger 与非相邻 return Hellinger 更低。说人话就是 chunk 内更抖，跨 chunk 却更容易重复旧路由状态。

## 解释边界

这里预测的是最终 success/failure，而不是停滞标签。q 越晚，MoE 可以反映已经发生的状态分叉、循环或停滞，所以晚期 AUC 不是纯初态难度；但它仍是观察性相关，不等于 MoE 已经因果识别出失败原因。
q35 以后成功轨迹陆续退出，风险集越来越小，必须结合 n、mixed states 和置信区间读，不能只看最高 AUC。

完整数值见 `auc_metrics.csv`、`auc_deltas.csv`、`state_aucs.csv.gz`、`oof_predictions.csv.gz`、`feature_selections.csv.gz` 和 `discovery_validation.json`。
