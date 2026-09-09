# 复用 v7 的后续实验

根据用户进一步要求，直接复用 moe-v7-0905/method 中的实现。
本设计在第三轮及其距离 follow-up 之后制定，不将后续选择称为预注册或新盲测。

## 不改变的部分

复用 intrinsic_score_arrays、IntrinsicGuardMonitor._query_features、
calibrate_reference、calibrate_peaks、alarms_from_scores。
保留 v7 原有基准、窗口、持续确认以及锁存 AND/OR：
freeze OR (persistent acceleration AND persistent recurrence loss)。
复用已有 31,600 条特征缓存，补齐缺失的 400 条，并用原始概率抽样重放核对。
每个 checkpoint 内沿用第三轮相同的 7/3 任务划分和三个初态隔离集合。
不把使用过全部任务的旧全局阈值当成未见任务结果。

## 对照

1. v7 原始无标签预算 API，在 seen 参考轨迹上按 1/3/5/10/15/20% 总报警预算标定。
   该预算不是成功 FPR 承诺。
2. 同一 Boolean 规则，在独立成功校准轨迹上标定同样的经验报警预算。
   尺度来自参考集；调用原有 calibrate_peaks，不训练任何模型。
3. v7 的连续 guard 分数 + 成功轨迹的整条轨迹 conformal 峰值阈值。
4. 相同 guard 分数 + 第三轮的时间变化带，比较阈值形状。
5. freeze-only、turbulence-only，用相同的常数峰值校准。
6. guard 与纯成功 MoE 距离的固定 max 融合，用相同常数峰值校准。

连续分数的定义：对三个已有 head，以 seen 参考中的全部有限时刻的 median/MAD
固定标准化。令 F 为 freeze，A/P 为已经平滑、持续确认的两个 head。
G_t=max(F_t,min(max_{u<=t} A_u,max_{u<=t} P_u))，保留两 head 的锁存语义。
不存在足够历史的 head 不参与；全部不存在时不评分。结束后的数组位置始终无效。
该同尺度连续 guard 是明确的新读出，不冒充原版三个独立阈值的精确等价。
融合时再用参考集 median/MAD 将 G 与纯成功距离各自标准化，取固定 max。
无成功/失败标签调权，无窗口搜索，无测试集阈值调优。

单轨迹与 task/init 组校准均保留。原始 v7 预算点单独列出其预算含义。
完整轨迹、同任务统一观察长度、固定 query、物理时序、误报和开销沿用第三轮评价。
本轮目标是比较已实现的时间信号与非参数空间距离，而非要求先于每次物理故障。
