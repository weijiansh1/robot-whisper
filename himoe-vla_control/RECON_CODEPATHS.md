# 代码通路侦察结论（2026-09-04，实现依据）

## E1/E2 重放采集（零服务器改动）
- 噪声 = 整段 `float32[10,24]`（`flow/noise` 键，n_action_steps×max_action_dim，与 F 无关），
  按值传入、服务器回 SHA256（`flow/noise_sha256`, `sha256-f32-c-v1`）→ 任意数值扰动合法。
- 重放输入 = snapshot 目录的 `policy_input.npz`{image,wrist_image,state} + manifest`prompt`
  → 请求键 `observation/image|wrist_image|state` + `prompt` + `flow/noise` (+`episode_id`)。
  **不需要 LIBERO 环境**，model env(py3.11) 即可跑客户端。
- legacy 模式（`--store-full-probs`，非 request-gated）：1 次推理 = routes.zarr 1 行 =
  `[L=8, D=F, U=11, E=32]` 全 flow 路由 → 单行即得 Ψ 路径。按 `episode_id` 对齐。
- request-gated 模式额外给 `flow_trajectory.zarr` 的 `x_traj[N,F+1,10,24]`（A^flow 基线、
  E1-B 的 x_{f*} 源；v_f=(x_{f+1}−x_f)/dt 精确反解），但**强制存 hidden（1.46MB/行）**
  → E1+E2 约 3.1 万行 ≈ 45GB，磁盘只有 ~11G。解法：给 serve_with_recorder 加
  `--no-hidden-capture`（改 4 处约 20 行）后再切 request-gated。
- 客户端组件直接复用：`capture_behavior_study.GatedQueryLedger`（gated 模式勿重写）、
  `bestofn_protocol.seed_words/flow_noise`（uint32 域分隔种子命名空间）、
  `capture_bestofn.capture_candidates`（E2 模板）。gated 模式最后一条必须
  `flush_after=True` 否则静默丢数据；query_id 全 run 唯一。

## F 参数化（E5/H8）
- `--trunc-rounds` 是早停+外推算子，**不是** F 重离散化；F 扫描须在
  `serve_with_recorder.py:474-476` 前设 `policy._policy.model.config.num_steps=F`
  （加 `--flow-steps` 参数，2 行；recorder/writer/FlowTracer 全自动跟随；
  wire 协议不变）。F≠10 时禁用 in-band `routing/capture`（triggered_fork 依赖它）。

## E1-B latent 注入
- `x_t` 在去噪循环中是同一张量原地 `+=`：hook 内 `x_t.copy_(...)` 即注入生效。
- 推荐走 FlowTracer 模板（~15 行 + 残差门禁跳过注入轮 + 协议加 2 个可选键）；
  或 fork `serve_adaptive_denoise.py`（整循环已在 Python 手里）。

## E5/E6 闭环
- 恢复：`branch_snapshot.restore_full_state(env, snap)`（顺序敏感：set_init_state →
  控制器 → 时钟/warmstart；接触相 fork 必须含 qacc_warmstart）。保真度验证脚本已有
  （branch_replay_test.py 等）。
- h 纯客户端：`_run_branch` 的 `replan_steps` 入参改为 horizon 可调用（<10 行），
  E5 固定 h 与 E6 触发式 h 同一实现。E5 主指标可从现有 `control_*` 逐低层动作数组算。
- E6 骨架 = `triggered_fork_collect.py` 换触发函数（z_V/z_A percentile）。

## 风险与前置
1. **bf16 Top-4 近平局**：86% 位点第 4/5 名差距 ≤5 ulp（probe_near_tie.py 实测）→
   E1 "boundary-fragile" 判读与 E3 截断伪影红线是高概率事件；跑前先过
   probe_near_tie 三件套；CPU/GPU 打平规则不同，CPU 臂须配 CPU 对照。
2. **语料 C 的 episode 无图像**（只有 state/actions/sim_state[79]）→ 在 C 的 trunk 上
   做 E1/E2 前需 ~60 行工具：LIBERO env 按 sim_state 恢复 → build_policy_observation
   → 导出 policy_input.npz。rolling-star 快照（t08, 22 snapshot）自带 policy_input，
   可先做 pilot。
3. 落地顺序：①legacy 模式打通 e1e2 采集+分析冒烟 → ②--no-hidden-capture 补丁后切
   gated 拿 x_traj → ③--flow-steps 伪影检查 → ④horizon 泛化跑 E5 → ⑤E1-B。
