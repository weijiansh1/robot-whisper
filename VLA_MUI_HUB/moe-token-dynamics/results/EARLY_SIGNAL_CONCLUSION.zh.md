# MoE 前期停滞信号：离线实验结论

## 结论

这批离线数据支持一个范围很窄的前期信号，但不支持“早期整个 MoE 已经清楚进入末端状态”。

1. **t7 没有可迁移信号。** 所有 route、hidden、token 位置和纵向动量组织均在机会线附近；预先指定的全 MoE history 为 AUC 0.462。
2. **t12 出现弱前兆。** 最强的是保留 `(layer, flow, token)` 位置的归一化 HB hidden direction：within-state AUC 0.624。在共同 seed-column 置换、init-state + seed 双留出、32 个早期检验 maxT 矫正后，`p=0.019`。
3. **这个信号不等于所有 MoE 量都有效。** 全部 route/hidden/动量直接合并的 `moe_history` 在 t12 只有 0.564，maxT `p=0.624`；无选择地堆特征会稀释窄信号。
4. **route 的前兆偏 action chunk 后段。** t12 的 action token 8–10 route AUC 0.617、32-test maxT `p=0.041`；state token、action 1–3、action 4–7 分别为 0.583、0.537、0.549，均未通过全族矫正。
5. **连续距离/动量不是早期主因。** t12 route temporal（含 cosine、JA、WJ 和速度方向动量）AUC 0.597、maxT `p=0.142`；hidden temporal 为 0.607、maxT `p=0.087`。当前 hidden orientation 比单独的纵向变化更强。
6. **t27/t34 是强末期读数，不是前期预警。** route history AUC 从 t20 的 0.638 升到 t27 的 0.807、t34 的 0.924；多数停滞型失败的行为陷入点中位为 t19，故这些高分主要是已发生的阶段分离。

## “陷入”如何定义

- 任务是 scene8，同一任务 16 个初始状态 × 32 个 flow-noise seed，共 512 条轨迹。
- 成功终点从其他初始状态、其他 seed 的成功轨迹构成参考集合；只使用两只 moka pot 的仿真位置生成标签，MoE 不参与标签定义。
- 若历史最好状态仍离成功集合超过 5 cm，且以后再也不能改善至少 1 cm，则最早满足点定义为停滞型陷入起点。
- 197/216 条失败属于这一类；另 19 条曾接近目标、后来回退或终点仍很近，不强行并入同一机制。
- 主阈值下陷入点最早 t15、中位 t19、10%–90% 为 t16–t30。因此 t7/t12 都严格位于主标签的陷入点之前。
- 目标半径改为 3–10 cm、进展阈值改为 0.5–2 cm 时，t12 hidden AUC 保持 0.616–0.630；类别成员基本稳定。进展阈值会改变陷入点记在 t18、t19 还是 t29，所以“精确 onset”不能过度解释。

## 最有效的组织方式

保留以下坐标，而不是先对 token 求均值：

```text
control history
  -> layer
    -> flow iteration
      -> token position (state, action 1 ... action 10)
        -> normalized hidden direction / route expert distribution
```

对 hidden，当前控制步内分别保留最后 flow、flow 均值和 flow 斜率；对 route，再计算同一 token 位置跨控制步的 cosine、JA、WJ 和动量。实验显示，**前期优先看 hidden identity，并单列 action token 8–10 route；不要先把六类量全部拼成一个总向量。**

## 对照和解释边界

- action-chunk history 在 t7/t12 的 AUC 为 0.438/0.556；加入 hidden 后为 0.430–0.457/0.555–0.563。动作本身没有解释掉 hidden 单独的 0.624，但当前线性组合也没有证明 hidden 对动作存在稳定增量。
- 同一保守目标下，sim-state history 在 t7/t12 为 0.523/0.552；早期物理状态对照同样较弱。
- 缓存只有 HB router 和进入 expert MLP 前的 contextualized `hb_hidden`，没有 expert output/contribution。因此结论是“MoE 输入状态含弱前兆”，不是“某个专家贡献导致陷入”。
- AUC 0.624 只够说明群体排序存在弱信息，不能据此对单条 rollout 下“已经陷入”的确定结论。要做在线告警，还需要独立任务/新采样确认、概率校准和固定误报率下的灵敏度评估。
- 这是离线预测证据，不是因果干预证据。

## 工件

- 主实验：`analyze_early_structure.py`
- 全族/seed-block 审计：`audit_early_familywise.py`
- 主图：`results/early_structure.png`
- 主结果：`results/early_structure_report.md`、`results/early_structure_summary.json`
- 32-test 全族审计：`results/early_familywise_report.md`
- 保守共同 seed-column 审计：`results/early_seedblock_report.md`、`results/early_seedblock_summary.json`
- 动作输出对照：`audit_early_action_control.py`、`results/early_action_control_report.md`
