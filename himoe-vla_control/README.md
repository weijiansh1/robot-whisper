# himoe-vla_control

控制论视角的 MoE-routing 研究工作目录（始于 2026-09-04）。

**主命题**：VLA Trap 是策略—环境闭环的错误吸引子（而非"不稳定"）；MoE routing
是控制器快速内部动力学的可观测坐标——loop 前兆为路由瞬态放大、后同步入重复盆地，
static 为路由有效自由度下降；目标是把它做成物理 Trap 成形前调节反馈频率的
内部事件触发器（MoE-only observer + event-triggered horizon）。

**假设栈**：H1 收缩裕度(成对ε扰动, λ/G_max/G_terminal) · H2 瞬态多路径终局单盆地
(D_seed(f) 形状) · H3 static 有效秩下降(gate 矩阵 SVD + Jacobian) · H4 谱结构(DMD,
探索项) · H5 MoE 触发短 chunk(h∈{1,2,5,10}) · H6 slow-normal 健康流形+相位速度 ·
H7 observer–actuator 四象限 · H8 flow 离散化否证(F∈{5,10,20}, 最高优先穿插)。

**实验优先级**：①H1 → ②H2 → ③H5 → ④H3 → ⑤H6 → ⑥H4（H8 尽早）。

**红线**：波动≠失收缩（须成对扰动）；未测 settling/λ 前不称 critical slowing down；
phase 需匹配任务/初态/绝对 query 防 elapsed-time 混淆；trigger 消融需
policy-query 与 wall-clock 双预算口径。

**可用资产**：
- 32,000 集 50×16 全网格路由语料 + CALVIN 200 链（`VLA_MUI_HUB/cache_new/`，审计见
  `parallel-route-capture/analysis/audit-20260904.md`）
- 噪声敏感度地图（同目录 `noise_sensitivity_map.csv`；5 个 8/16 对半格为首选靶点）
- rolling-star 冻结快照/恢复 + 显式噪声注入机械（`himoe-route-capture/`）
- 在线模型服务器（卡6，端口 8803/8804/8810-8812）
