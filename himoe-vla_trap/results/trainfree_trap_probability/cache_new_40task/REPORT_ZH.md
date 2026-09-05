# cache_new 40-task Trap 概率压力测试

## 核心结果

冻结的 A/B 概率表已直接应用于 `cache_new` 的 40 个完整任务、16,000 条轨迹和 253,722 次推理。预测阶段只读取 HB MoE routing；逐 query 概率写盘并记录 SHA256 后才加载最终 success/failure。

该语料没有统一的逐 query Trap onset，因此本实验只能回答成功轨迹沿途会得到什么数值，以及概率与最终 outcome 是否相关。它不能评价原定义的 `P(未来两 query 内 Trap onset)` 是否校准。

| source 概率表 | 可评分成功 / 全部成功 | 成功 query 中位 / p90 / max | 成功 episode-max 中位 / p90 | q12 pooled / within-task AUROC | episode-max AUROC | 75% 报警 |
|---|---:|---:|---:|---:|---:|---:|
| A dense | 9,010/15,468 | 6.86% / 15.33% / 35.83% | 6.86% / 20.38% | 0.670 / 0.578 | 0.868 | 0 |
| B proxy | 9,010/15,468 | 0.27% / 7.17% / 51.32% | 0.27% / 11.19% | 0.685 / 0.561 | 0.890 | 0 |

方法从 q12 才能输出，因为 q4--q11 用于 episode 自身基线。共有 6,458 条成功轨迹在 q12 前已经完成，不能被当前方法评分。这是部署延迟，不应把它们计作“低风险预测正确”。

## 25% 阈值

- A dense：scoreable failure recall 28.76%，scoreable success alarm rate 2.50%，endpoint precision 40.48%。
- B proxy：scoreable failure recall 45.86%，scoreable success alarm rate 3.03%，endpoint precision 47.20%。

B proxy 的 50% 状态看起来更保守：命中 133/532 个最终失败，并在全部成功中报警 61/15468，endpoint precision 为 68.56%。排除与 A/B source 相同的 moka-pot 任务后仍为 recall 23.60%、scoreable-success alarm 0.63%、precision 62.84%。首次命中的中位位置已经是 horizon 的 82.4%，只覆盖 20 个任务，且最大单任务贡献 30.1% 的命中。相同或更低成功误报率的固定 phase-0.79 时钟可覆盖 100.00% 的失败，而该 MoE 阈值只有 25.00%；时钟还略早于 MoE 的中位报警位置。因此不能把它当成已经成立的通用 detector。

这些是与最终失败的描述性关联，不是 Trap-onset precision。75% 仍然必然零报警，因为冻结 A/B onset-hazard 表本身的最大值只有 35.8% 和 51.3%。

## 解释

pooled q12 AUC 看似有 0.670/0.685，但在每个任务内部配对后只剩 0.578/0.561。这说明早期 pooled 分离相当一部分来自任务构成。episode-max AUC 又因失败轨迹更长、碰到高状态的机会更多而升高；最终 duration 自身的 pooled AUROC 为 0.910，同任务配对后接近 1.000。所以不能用 endpoint outcome 把这个概率重新命名为“失败概率”。

最准确的结论仍然是：A/B 单任务 onset 频率表不能直接作为 40-task 通用概率标尺；`cache_new` 更适合检查成功轨迹的概率背景和分布漂移。真正验证近两步报警仍需要为新任务构造独立的 query-level Trap onset。
