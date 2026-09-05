# 控制论实验程序（E0–E8）· 规范文档

因果证据链：表型复现 → 扰动响应 → 错误盆地形成 → 事件触发控制 → 因果定位。
**在线判断只读 MoE**；物理轨迹只用于离线定义 onset / 成败 / 干预评价，绝不进 detector。

## 统一坐标系（所有实验共用）

- gate 概率 p_{q,f,l,u} ∈ Δ^31（E=32）；平方根嵌入 ψ(p)=√p；
  Ψ_{q,f} = vec_{l,u}[√p]，Hellinger 即欧氏距离 d_H(p,p')=‖√p−√p'‖₂/√2。
- 归一化 flow 时间 s_f=f/(F−1)，Δs=1/(F−1)；
  速度 v_{q,f}=(Ψ_{q,f+1}−Ψ_{q,f})/Δs；加速度 a_{q,f}=(v_{q,f+1}−v_{q,f})/Δs。
- V_q^late = Σ_{f∈late}‖v‖Δs；A_q^route = Σ‖a‖Δs / (Σ‖v‖Δs+ε)。
  （时间归一化 ⇒ F∈{5,10,20} 变换下不机械缩放。）

## 实验一览与优先级

| # | 验证 | 核心操作 | 阶段 |
|---|---|---|---|
| E1 | loop 前内部扰动放大？ | 同 snapshot 微扰 ξ+δξ，测 impulse response | **P1 go/no-go** |
| E2 | fresh noise 中途分叉终局汇聚？ | 同 snapshot 64 seeds，D_seed(f) 曲线 | **P1 go/no-go** |
| E5 | 多想 vs 早看环境 | F∈{5,10,20} × h∈{2,5,10} 因子 Snapshot-Fork | **P2** |
| E6 | MoE 前兆能否当事件触发器 | MoE-triggered h:10→2，六对照 | P3（须 E5 成立） |
| E3 | static=低秩共享支持？ | gate 矩阵 r_eff/stable rank + J_{g→y} 谱 | P4 机制 |
| E7 | sensor 还是 causal actuator | healthy gate / expert-output patch 六条件 | P4 机制 |
| E4 | 不同谱模态？ | 低维 AR(2)/DMD（探索项，不做主结论） | P4 |
| E8 | slow-normal 判别 | 延长 timeout + routing phase/manifold | P5 |
| E0 | 独立语料 C 预注册复现 | 四个主检验（见下） | 与 P2 并行可采 |

## E1 细则

- E1-A（噪声敏感度，先做）：同 observation snapshot，ξ'=ξ+δξ，
  ‖δξ‖/‖ξ‖ ∈ {1e-4,1e-3,1e-2}，每 snapshot 32 个随机方向。
  D_f = d_H(P_f(ξ),P_f(ξ'))；G_f=D_f/‖δξ‖；G_peak=max_f G_f；
  G_terminal=G_{F−1}；Funnel=(G_peak−G_terminal)/(G_peak+ε)。
  ⚠ 只能称 input-to-routing sensitivity，不得直接称 contraction。
- E1-B（更强，若可在 flow 中段复制 latent x_{f*}，f*∈{2,5,8}）：
  x'=x+εv 续跑，λ_{f*:F}=log((D_F+η)/(D_{f*+1}+η))/(F−f*)。
- 位置：每个 loop trunk 的 q−4, q−2, q0, q+2 + matched no-loop controls
  （同 task/init/绝对 query）。
- 预期：healthy G_peak 低且 λ<0；pre-loop G_peak↑ 且 λ→0 或 G_terminal 回落
  （分叉后汇聚）；static 二选一：dead-zone（G_peak↓）vs boundary-fragile
  （soft 变化小但 Top-4 ID 大量翻转）。
- 否证：控制配对后 G_peak^preloop ≈ G_peak^healthy ⇒ 只可称
  "routing path geometrically non-smooth"。

## E2 细则

- 同 snapshot 采 ξ_1..ξ_64；D_seed(f) = 平均成对 d_H。
- FanOut=max_f D_seed(f)；TerminalDiversity=D_seed(F)；
  FunnelIndex=1−D_seed(F)/(FanOut+ε)。
- 机制三分：A 全程相似（低噪声-路由可控性）；B 中间散终局聚
  （最想看到：FanOut_{q−2}↑ 且 FunnelIndex↑ ⇒ 0/8 变机制结果）；
  C 终局多盆地（routing 多样 ≠ 恢复多样）。
- 规模：24–32 loop trunks × 4 时点 × 64 seeds + 等量 matched healthy。
  宁多 trunk 勿多 seed。

## E5 细则

- 因子：F∈{5,10,20}（内部计算）× h∈{2,5,10}（执行长度），9 cells。
- Fork 位置 q−4/q−2/q0 + matched healthy；common random numbers,
  8–16 paired seeds/cell；≥24–40 独立 loop trunks（先 12 trunk pilot）。
- 主指标（h 不同不能用 query 数）：P(未来 40 个低层动作内进 loop)、
  P(相同总 action budget 内完成)。副：precursor 消失率、funnel、
  query 数、wall-clock、误干预成本。
- 解释矩阵：F=20 有效→内部计算不足；短 h 有效→反馈延迟主因；
  都无效→深盆地需 rewind；仅 q−4 有效→可恢复窗早于前兆。

## E6 细则（E5 成立才做）

- trigger：z_V,z_A = healthy percentile 化；s_q=max(z_V,z_A)；
  τ=Q_0.99(healthy)；触发 h:10→2 持续 2–3 query；迟滞 τ_on>τ_off。
- 六对照：固定 h10 / 固定 h2 / **预算匹配随机触发** / 固定时刻触发 /
  MoE-triggered / oracle(q−2)。通过标准：MoE > random（同预算）。

## E3 细则

- G_{q,l,f} ∈ R^{10×32}；四指标同测：H_row（行熵）、C_soft（1−成对 d_H）、
  r_eff=exp(奇异值熵)、r_stable=‖G‖_F²/‖G‖₂²。
- static 预测（L15,d9,q+2 附近）：H_row↑ + C_soft↑ + r_eff↓。
- ⚠ 伪影排除：若仅离散 Top-4 occupancy 变化而 C_soft/r_eff 不动 ⇒
  截断伪影，不得称自由度下降。
- 强版本：J_{g→y} 用 JVP/VJP 随机估计 Frobenius/top-σ/r_eff。

## E7 细则

六条件：原始 pre-loop / healthy gate patch（L12–15,f7–9 pre-topK logits）/
shuffled healthy / equal-norm random / healthy expert-output patch /
reverse patch（healthy←pre-loop）。三层读出：内部表型、下游计算、闭环结果。
预期：gate 可读、expert-output 更可控（observer–actuator 分工）。

## E8 细则

延长 H1=1.5–2×H0 不重置获得 late-success 标签；只用成功轨迹建
M_g(φ)（时间归一化+DTW 模板）；d^M、v^R=Δφ̂；selective extension
对照（不延/全延/随机延/MoE-only 延），指标=新增成功/额外动作数。

## 数值伪影检查（穿插 E5）

F∈{5,10,20} 路径插值到同一 s∈[0,1] 网格重算 V/A；要求排序、lead、AUC
稳健且不依赖单一 d9 索引。附录须比 A^route vs A^flow（动作流加速度基线）。

## 统计红线

1. 独立单位 = trunk；hierarchical bootstrap over trunks；同 snapshot 的
   seeds 是条件重复。
2. 恢复类实验先加 trunk 再加 seed（30×8 优于 1×240）。
3. common random numbers 跨条件配对。
4. 确认性检验只保留预注册假设（E0 四个 H0），其余进探索附录。
5. 在线阈值需 200–300 独立 healthy episodes 校准（22 条只够 sanity）。

## E0 预注册四检验（语料 C 上，不得重新调参）

H01: ΔV^late_{loop,q−2}=0；H02: ΔA^route_{loop,q−2}=0；
H03: ΔSharpness_{static,q+2}=0；H04: ΔOccupancy_{static,L15,d9}=0。
固定 layers/token 范围/onset 定义/窗口/聚合/方向/校正。规模 ≥40 loop
+80 static 独立 trunks + matched controls。static 须在连续 soft 指标上
也成立（防 Top-4 边界伪影）。

## 运行次序（8 卡）

P1: E1+E2（24 loop trunks + 24 controls；32 扰动 + 64 seeds；q∈{−4,−2,0,+2}）
→ P2: E5（12 trunk pilot → 30–40）→ P3: E6 → P4: E3/E7/E4 → P5: E8。
每卡常驻一份模型、以 snapshot 为 job 批量化，不跨卡同步。
