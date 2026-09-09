# Rolling-star K=16 实验结论

## 一句话结论

这批数据支持“MoE 路由包含运动阶段/失败机制的分布式痕迹”，但不支持“某个 expert 是失败开关”，也还不支持用 q0 路由直接在线报警。最强的单 expert 和单坐标 discovery 信号都没有在预先冻结的 validation 集复现，因此现在剔除 expert 没有统计依据。

## 数据与完整性

- 任务：`put both moka pots on the stove`。
- 22 个滚动主干快照，每个快照 K=16，共 352 条终止分支；117 成功、235 失败，15 个快照同时含成功和失败 sibling。
- 4 条独立 worker/initial-state 轨迹的成功率差异极大：26.0%、0%、6.25%、90.6%。这是必须控制的 rollout 难度混杂。
- 保存 16,180 个 query 的完整 HB 路由概率，形状为 `[16180, 8 layers, 10 denoise, 11 suffix, 32 experts]`；未保存隐藏层。
- 352/352 条分支重新执行原始 BDDL 终止谓词，记录结果与重放结果 0 个不一致。
- 8,450 个远端原始文件已在本地逐个通过 SHA-256；17 个 MP4 均由 ffmpeg 完整解码。
- 所有分析只使用 q0 或 q0-q2，不使用尾部、remaining-time、超时哨兵，也不报告 AUC。

## 失败到底是什么

主失败类型互斥计数：

| 类型 | 分支数 |
| --- | ---: |
| loop/cycling | 138 |
| 单子任务遗漏 | 55 |
| 掉落/重抓 | 22 |
| 子任务撤销 | 8 |
| 接触目标但未满足精确谓词 | 5 |
| 纯停滞 | 4 |
| 其他超时 | 2 |
| active retry | 1 |

物理标签允许重叠时，163 个失败有 loop，49 个有停滞，二者并集覆盖 199/235 = 84.7%。剩余 36 个未被停滞/loop 覆盖的失败，主要是单子任务遗漏 23、near miss 5、掉落/重抓 3、撤销 2、其他超时 2、active retry 1。也就是说，原来的停滞/loop 解释很有覆盖率，但确实漏掉了一批“持续在动，却把子任务做错或漏做”的错误。

## 单 expert 假设没有通过

采集中途冻结了两条有明确方向的 prospective 信号，只用后来尚未启动的 5 个快照验证：

1. `q0 action / front layers 2-5 / d0 / expert 0`：validation 的失败减成功概率差仅 `+0.000031`，95% CI `[-0.000283, 0.000212]`，两信号 Bonferroni 后 `p=0.844`。
2. `q0 state / front layers 2-5 / d1 / top1 mass`：validation 的高低四分位失败率差为 0，95% CI `[-25, +25]` 个百分点，Bonferroni 后 `p=1.0`。

两条 discovery 效应都没有复现。全 2,560 个 expert 概率格子的最好 maxT 校正值也只有 `p=0.771`；expert 0 的全量 maxT 为 `p=0.937`。所以不能据此删除、固定或替换任何 expert。若要谈因果，必须做同状态随机路由干预，而不是继续从相关性表里挑 expert。

## 前层、后层和去噪步

最强总体单坐标是 `q0 state / front 2-5 / d0 / top1 mass`：快照内高低四分位失败率差 23.3 个百分点，bootstrap CI `[10.0, 36.7]`，原始 `p=0.0008`；但在 280 格的 maxT 校正后为 `p=0.0572`，没有越过 5% 门槛。紧随其后的仍是前层 state token 的 d1-d3，而后层和较晚去噪步明显弱。

失败子类型的单坐标扫描也没有通过家族校正。最强线索集中在 q0-q2 的前层 action d0/d2 top1 mass，以及后层 d7 entropy；最低 maxT 仍为 `p=0.124`。因此更合理的解释是信息分散在多层、多去噪步和多个 expert 上，而不是存在一个可直接读取的“失败 expert”。

## MoE 能否识别 rollout 难度

能识别熟悉轨迹里的难度，但没有证明可泛化到新 initial state。

- 对 22 个快照的 K=16 失败率做真正 leave-one-snapshot-out：初始物体/EEF 几何的 MSE 为 `0.0207`，q0 平均 action-route 为 `0.0542`，常数均值基线为 `0.1518`。几何明显强于 MoE；几何加 MoE 反而是 `0.0544`，没有增量收益。
- leave-one-worker-out 时，几何 MSE 为 `0.2221`，均值基线 `0.2476`，CI 跨 0；平均 action-route 为 `0.3030`，更差。
- 候选级的绝对 MoE 模型在按快照分组的 5-fold 中看起来很强，例如 q0 action-route Brier `0.1411`，对照失败率基线 `0.2232`；但 leave-one-worker-out 变成 `0.3961`，比基线 `0.3313` 差。
- 去掉每个 K=16 快照的共同均值后，q0-q2 state-route 相对噪声/动作基线只改善 Brier `0.00369`，相对失败率基线 CI 跨 0，且 best-of-16 成功率没有提升；换 worker 后同样变差。

所以绝对路由主要编码“这是哪一种初始几何/运动阶段、这条 rollout 有多难”。它在见过相同 worker 的其他快照时有预测力，但不是一个已证实的、跨 initial-state 的 sibling 选择器。

## 其他失败类型有没有新信号

有线索，但没有一个已经成为稳健在线信号。

- 对未被停滞/loop 覆盖的 36 个失败，最好 q0-q2 快照内中心化 state-route 相对失败率基线改善 Brier `0.0091`，CI `[-0.0009, 0.0218]`；相对噪声/动作改善 `0.0058`，CI `[-0.0034, 0.0168]`。换 worker 后不成立。
- 单子任务遗漏在按快照分组的 5-fold 中，q0-q2 绝对 state-route 相对噪声/动作改善 `0.0492`，CI `[0.0151, 0.0818]`；但相对简单失败率基线 CI 跨 0，leave-one-worker-out 也不优于失败率基线。它更像 initial-state/rollout 类型信息。
- 唯一方向较一致的是“已知最终失败后，区分 loop 与其他失败”：q0 的完整 state-route 向量先做 K=16 快照内中心化，再 PCA+逻辑回归，leave-one-worker-out Brier 为 `0.2393`，优于失败率基线 `0.2596` 和噪声/动作 `0.2679`。4 个 worker 的 Brier 增益方向都为正。
- 但独立单位只有 4 个 worker，精确符号翻转检验的单侧 `p=0.0625`，这是 4 组数据能达到的最小值；而且该模型是 364 个子类型模型中事后选出的，只能作为下一轮冻结候选。

更关键的是，把同一个 q0 模型直接用于全部 352 条分支、在线预测“最终是否 loop”时，Brier 变成 `0.3424`，反而差于失败率基线 `0.3240` 和噪声/动作 `0.3285`。这说明它目前只是失败条件下的机制指纹，可能还受 conditioning/collider bias 影响，不能单独用作在线 loop 报警器。

## 当前可下的结论

1. 停滞/loop 的物理检测器解释了大部分失败，但没有解释遗漏、near miss、掉落和撤销。
2. MoE 的确含有比单 expert 更分布式的机制信息；q0 state-route 对“失败内部的 loop 类型”最值得下一轮继续验证。
3. 当前数据没有给出跨 worker、全分支可用的早期失败/loop 评分器。
4. 初始几何比 MoE 更直接地解释 rollout 难度；MoE 的强绝对效果不能被解读成独立失败认知。
5. 不剔除 expert。先做更多独立 worker/任务的 prospective 验证，再考虑路由干预。

## 下一轮最有价值的实验

使用已经保存的逐 query 路由，在固定因果时点 q5/q10/q20 上，只对仍在运行的分支预测“未来 H 个 query 内首次进入 loop/stagnation”，并同时放入纯运动学、动作、噪声基线；按 worker 和任务整组留出，禁止任何 remaining-time 或尾部归一化。这样才能回答模型是否在物理停滞发生前看见了风险，而不是事后识别一种失败轨迹。

## 文件入口

- 完整英文统计报告：[analysis/report.md](analysis/report.md)
- 正式审计：[analysis/audit.json](analysis/audit.json)
- 冻结验证：[analysis/frozen_signal_validation_family.csv](analysis/frozen_signal_validation_family.csv)
- 失败标签：[analysis/candidate_physical_labels.csv](analysis/candidate_physical_labels.csv)
- loop 逐 worker 审计：[analysis/exploratory_loop_q0_worker_audit.json](analysis/exploratory_loop_q0_worker_audit.json)
- 全分支线上口径反证：[analysis/exploratory_loop_q0_all_branch_audit.json](analysis/exploratory_loop_q0_all_branch_audit.json)
- 视频目录：[VIDEO_CATALOG.md](VIDEO_CATALOG.md)
- 环境与来源：[PROVENANCE.json](PROVENANCE.json)
- 正式派生文件校验：[DERIVED_SHA256SUMS.txt](DERIVED_SHA256SUMS.txt)
