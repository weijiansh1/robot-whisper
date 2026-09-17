# 复返率触发的并行 control 迭代（API 服务式）

日期：2026-09-16 晚至 09-17 凌晨。目标：基于"弯曲 = 不确定而非跳出、失败 = 塌缩进紧致复返区、真循环不存在"这些判断，
用现有 8 个 HiMoE 策略服务（9510–9517）做并行请求，在线触发干预并在 170 集上迭代三轮。

## 设置

- 样本：`simulations/topo-libero10`（100 集，61 成功）+ `topo-plus`（70 集，25 成功），同 flow seed 42。客户端复刻 `run_episode` 的主循环，
  native 臂与记录的 trace 逐位一致（20/20 动作、逐 query RR 相同），因此每个臂都与 native 严格配对。
- 触发：服务端返回的 top-4 路由（`routing/expert_ids`、`routing/expert_weights`）→ 后四层动作 token 的稀疏 sqrt 概率向量 → 14 个 query 因果窗口的复返率 RR
  （Theiler 3，ε = 0.883 为 170 集早期窗口距离的 10% 分位）。RR ≥ 0.4 连续 2 次触发（q ≥ 13）。离线标定：84 个失败中 78 个触发（中位 q29.5），86 个成功中 6 个。
- 干预臂（触发后执行，冷却 6 个 query）：
  - 第 1 轮（170 集）：`withdraw` 抬 4 cm 后回撤 ≤ 4 cm；`escalate` 回撤→正侧绕→负侧绕；`resample_escape` 触发后 6 个 query 各发 8 路不同噪声的并行请求，选路由点离最近 14 个历史点最远的候选；`resample_random` 同请求随机选。最多 3 次。
  - 第 2 轮（84 失败集；hold16/persist 另跑 86 成功集）：`hold16` 16 步零位移（物理空操作对照）；`withdraw8` 抬 8 cm 回撤 8 cm；`persist` 回撤→侧绕→8 cm→夹爪开合→抬升，最多 6 次；`withdraw_early` θ=0.3。
  - 第 3 轮（84 失败集）：`reprompt` 触发后 6 个 query 用改写指令；`noise2` flow noise ×2；`retrace` 抬升后退回 3 个 query 前的位置。
- 规模：1,440 次完整运行，最多 84 个客户端并发；37 次因服务端握手超时失败，改为按需连接 + 重试后断点续跑全部补齐。

## 结果

| 臂 | 救回 / 84 失败 | 95% 区间 | 误伤 / 86 成功 | 干预后 6 query 的 RR 均值 | RR 降到阈值下的比例 |
| --- | ---: | --- | ---: | ---: | ---: |
| hold16（空操作） | 7 | 4–16% | 0 | 0.72 | 30% |
| withdraw | 5 | 3–13% | 0 | 0.69 | 32% |
| withdraw8 | 5 | 3–13% | 未测 | 0.68 | 34% |
| withdraw_early | 5 | 3–13% | 未测 | 0.65 | 37% |
| retrace | 5 | 3–13% | 未测 | 0.71 | 32% |
| escalate | 4 | 2–12% | 0 | 0.69 | 32% |
| persist | 4 | 2–12% | 0 | 0.69 | 32% |
| resample_escape | 2 | 1–8% | 0 | 0.83 | 17% |
| resample_random | 1 | 0–6% | 0 | 0.84 | 15% |
| reprompt | 1 | 0–6% | 未测 | 0.83 | 14% |
| noise2 | 1 | 0–6% | 未测 | 0.80 | 16% |

- 误触发的 6 个成功集在所有测过的臂里都仍成功，只是慢 8–43 步。
- 11 个臂的救回并集为 13/84（15.5%）；6 集被 ≥3 个臂救回，`task04-init08` 被全部 11 个臂救回。
- 配对比较（不一致对的精确二项检验）：hold16 对 noise2 / reprompt / resample_random 为 6/0（p = 0.03）；物理臂对 hold16 为 3/5（p = 0.73），物理臂之间 2/2、3/3。
- 救回案例从首次干预到成功要 121–324 步；触发前既有停滞型也有仍在移动的失败。

## 判断

1. 复返率触发器可用且安全：在线计算只需服务端已返回的 top-4 路由，误触发不造成损失（0/86）。
2. 有救回，但小且与干预内容无关：空操作 hold16 是最好的单臂（7/84）；回撤、侧绕、8 cm、退回、升级、持续都不比它好。救回的是约 15% 的"脆弱"失败集，任何扰动都能让它们分叉。
3. API 侧扰动（噪声重采样、逃逸选择、改写指令、噪声 ×2）明显不如物理停顿（p = 0.03），它们离不开复返区（RR 仍 0.80–0.84）。
4. 没有一种干预能溶解复返区：干预后 RR 平均回到 0.65–0.72，只有约三分之一在 6 个 query 内降到阈值下；升级到 6 次也没有增益。剩下约 85% 的失败是稳固的错误决策区，不是轻推就能出去的动力学状态。
5. 对当前方法的含义：把"复返率触发 + 短暂停顿"当作零成本、零损失的基线（本集合 86 → 93/170）；再往上需要改变策略输入的语义（子目标、感知），而不是继续调路由或动作候选。

## 局限

同一批 170 集、同一噪声种子，触发阈值在这批上标定；救回数小，各臂区间大幅重叠；withdraw8 / withdraw_early / retrace / reprompt / noise2 未测成功集误伤；
物理操作用 OSC 增量位置控制，实际位移约为名义的 15%/步；未做新初态或新噪声的前瞻验证。

## 文件

- 运行器：`run_control_episode.py`（v1）、`run_control_episode_v2.py`、`run_control_episode_v3.py`；驱动 `run_control_batch.py`（`CONTROL_RUNNER` 选运行器）。
- 分析：`analyze_control_batch.py`（每轮 `analysis.json`）、`analyze_control_overlap.py`（`control-r1/overlap.json`）、`plot_control_batch.py`（`control-results.png`）。
- 数据：`simulations/control-r1`、`control-r2`、`control-r3`，每次运行含 `summary.json`、`control.npz`（逐 query RR/直径/候选、末端位置、动作、路由点）、`client.log`。
- 标定：`samples/moe-recurrence-20260916/wire-route-calibration.json`、`wire-route-rr.npz`（`wire_recurrence_calibration.py`）。
