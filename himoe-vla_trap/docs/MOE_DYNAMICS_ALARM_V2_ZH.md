# MoE dynamics v2：修正规则与 prospective 在线验证

> 后续证据更新：固定 v2 已在 `cache_new` 的 400 条新 capture 上严格离线回放。全部失败报警为 `29/138`，成功误报为 `13/262`；query-sampled 未抓住代理为 `13/53`，其中事件前 5 query 或当下为 `9/53`。因此本页 24 条 prospective 的 `5/7` 目标召回不能外推为稳定泛化性能。完整更新见 [cache_new 报告](CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md)。

## 结论

旧 `transition_split_v1` 的问题已经定位并修正了一版。新规则不再要求 layer-5 state/action gap、跨 chunk jump 和前层 action nominality 同时越界，而是直接检测**后层相对前层的完整去噪 route acceleration 异常**。

在预先固定的 GPU5 新 seed `20260908`、24 条在线轨迹上：

| 同一批轨迹 | 旧 v1 配对重放 | 新 v2 在线 |
|---|---:|---:|
| 严格漏抓型 | 1/7 | **5/7** |
| 其他失败/Trap | 0/7 | **3/7** |
| 全部失败 | 1/14 | **8/14** |
| 成功误报 | 0/10 | **1/10** |

14 条失败中，v2 独有命中 7 条、v1 独有 0 条、两者都命中 1 条；精确 McNemar 双侧 `p=0.015625`。因此可以说 **v2 在这批 prospective 数据上显著提高了总体失败报警**。

但它还不是可部署 detector：严格漏抓召回 `5/7=71.4%`（Wilson 95% CI 35.9%--91.8%），全部失败召回 `8/14=57.1%`（32.6%--78.6%），成功误报 `1/10=10%`（1.8%--40.4%）。样本仍小，且存在 2 条目标漏报。

## 新规则

对当前 query 的 HB routing：

```text
R: [8 HB layers, 10 flow steps, 11 suffix tokens, 32 experts]
```

只取 10 个 action tokens。对层组 `G` 定义完整去噪 route acceleration：

$$
A_G(R)=\operatorname{mean}_{l,f,u}
\frac{\left\|\sqrt{R_{l,f+1,u}}-2\sqrt{R_{l,f,u}}+\sqrt{R_{l,f-1,u}}\right\|_2}{\sqrt{2}}.
$$

前层为 HB 2--5，后层为 HB 12--15。用同一个 query 内的前层消除共同尺度：

$$
C(R)=\frac{A_{back}(R)}{A_{front}(R)}.
$$

旧健康参考仍通过 front action routing 的 closed-begin online monotone DTW 匹配阶段。设五条匹配健康 routing 为 $R_i^H$：

$$
s_t=\frac{C(R_t)}{\max_i C(R_{i,t}^H)}.
$$

在线 raw rule：

$$
s_t>1.0457019658.
$$

连续两次 raw 才产生正式 alarm。选择器仍然只读 MoE routing 与成功参考，不读取动作值、夹爪、EEF、目标位置、reward 或 success。

## 阈值怎样得到

阈值只使用 seed `20260905` 的 5 条成功 H20 routing reference：每次拿出一条作为健康 query sequence，其余 4 条做阶段参考；对每条轨迹计算能形成 persistence-2 的最大支持值，最后取五条的最大值。

| healthy episode id | 最大单点 score | persistence-2 最大支持 |
|---:|---:|---:|
| 1610000001 | 1.0448 | 1.0030 |
| 1610000003 | 1.0592 | 1.0012 |
| 1610000004 | 1.0913 | 1.0176 |
| 1610000005 | 1.0775 | **1.045702** |
| 1610000006 | 1.0478 | 1.0038 |

所以健康 leave-one-out 正式报警为 0/5。这里没有梯度、参数拟合或失败标签，属于 train-free healthy calibration；但 feature 设计受到 seed `20260907` 的 posthoc 失败分析启发，因此只有新 seed `20260908` 才是 v2 的 prospective test。

## 实验协议

- 物理 GPU：卡 5，NVIDIA H20-3e
- task：LIBERO-10 task 8，`put both moka pots on the stove`
- 固定 24 条，init `0/3/7/20` 各 6 条
- seed：`20260908`
- 阈值、persistence、init schedule 和 calibration SHA256 在运行前冻结
- 报警后继续执行原 chunk，不影响后续 action 或 outcome
- 10 成功、14 失败、9 个报警 episode
- 1129 次推理的完整路由均由客户端和服务端保存
- 客户端/服务端路由逐元素完全相同，episode ID 完全相同，hook failure 为 0

因为报警没有干预动作，旧 v1 可以在同一批保存路由上精确配对重放。它不是跨 seed 的间接比较。

## 目标漏抓时序

| ep/init | 严格漏抓 q | v2 alarm q | 判定 |
|---|---|---|---|
| 2/20 | 13,33 | 13 | 第一次漏抓当下报警 |
| 4/20 | 15,34 | 14,15,24,34 | 第一次提前 1 query，两次均在事件处持续报警 |
| 8/3 | 41 | 31--34,38,41--43 | 事件前 3 query 和事件当下均报警 |
| 14/20 | 13,33 | - | 漏报；q13 有单次 raw，被 persistence 滤掉 |
| 17/20 | 36 | - | 漏报；q35 有单次 raw，被 persistence 滤掉 |
| 19/3 | 35 | 12,33,36--39 | 事件前 2 query 报警，最近报警为事件后 1 query |
| 22/20 | 13,33 | 13 | 第一次漏抓当下报警 |

5 条被命中的目标轨迹都在漏抓前 5 query 或漏抓当下出现正式报警。旧 v1 在这 7 条中只命中 ep8，而且是漏抓后 q43 才报警。

需要注意因果边界：q13 routing 是在 q13 action chunk 执行前生成的。此时模型能表达“这次动作计算异常/风险高”，但还不能观测尚未发生的真实接触结果。因此成功的高风险抓取也可能触发。

## 唯一成功误报

ep9/init20 在 q34、q35 连续异常并于 q35 报警；真正闭爪发生在 q36，锅随后被成功抬起，episode 在 q43 完成。按最终 outcome 这是 false positive。

- [ep9 成功误报 annotated full](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_009_init_20_random/videos/alarm_annotated_full.mp4)
- [ep9 报警附近 clip](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_009_init_20_random/videos/alarm_annotated_clip.mp4)

这说明 v2 更接近“高风险动作计算报警”，还不能确定高风险动作最终一定失败。若把 persistence 降为 1，7 条目标会全部出现 raw signal，但成功 raw 也会从 1/10 formal 增至 3/10，因此不能直接这样改。

## 目标报警视频

- [ep2：q13 漏抓当下报警](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_002_init_20_random/videos/alarm_annotated_full.mp4)
- [ep4：q15/q34 两次漏抓](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_004_init_20_random/videos/alarm_annotated_full.mp4)
- [ep8：q41 漏抓前后持续异常](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_008_init_03_random/videos/alarm_annotated_full.mp4)
- [ep19：q35 漏抓附近报警](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_019_init_03_random/videos/alarm_annotated_full.mp4)
- [ep22：q13 漏抓当下报警](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/episode_022_init_20_random/videos/alarm_annotated_full.mp4)

其他失败报警视频位于 ep6、ep13、ep23。所有 9 个报警 episode 都保存 clean full、annotated full 和 annotated clip，共 27 个视频，哈希记录在 [run manifest](../results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908/manifest.json)。

## 结论边界

可以成立的结论：

- 旧三条件 conjunction 确实选错了信号。
- 后层相对前层 route acceleration 是更合适的 train-free 在线 signal。
- v2 在全新 seed 上把同轨迹总体失败覆盖从 1/14 提高到 8/14。
- 它对目标漏抓的时序明显早于 v1。

目前不能成立的结论：

- v2 已经可靠部署；
- MoE 能确定尚未发生的物理接触结果；
- 所有 failure/Trap 都具有相同 acceleration 模式；
- 1/10 的成功误报可以忽略。
- prospective 的 `5/7` 目标召回可推广到更大 `cache_new` 数据；后续结果为 `13/53`，且及时报警只有 `9/53`。

下一版如果继续优化，应该保留 v2 作为 risk branch，再增加**动作执行后的 belief-update branch**，专门判断新视觉状态是否支持“物体已随手移动”。该 branch 仍可只使用下一 query 的 MoE routing，但必须再用全新数据验证，不能在本批上调完阈值后回报。

## 代码与结果

- [v2 selector](../code/moe_dynamics_online_selector.py)
- [健康校准脚本](../code/build_moe_dynamics_calibration.py)
- [旧数据开发回放](../code/evaluate_moe_dynamics_selector.py)
- [在线 collector](../code/collect_online_moe_only_alarms.py)
- [prospective 审计脚本](../code/audit_moe_dynamics_online_results.py)
- [冻结配置](../configs/moe_dynamics_online_alarm_v2.json)
- [健康 calibration](../results/moe_dynamics_online_alarm/calibration_v2/calibration.json)
- [prospective 审计摘要](../results/moe_dynamics_online_alarm/prospective_audit/summary.json)
- [逐 episode 配对结果](../results/moe_dynamics_online_alarm/prospective_audit/episode_audit.csv)
- [cache_new 严格 MoE-only 回放](CACHE_NEW_MOE_ONLY_OFFLINE_ZH.md)
