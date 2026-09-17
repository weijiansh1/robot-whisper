# MoE Control Experiments

实验边界、阶段门槛和执行协议见 [PLAN.zh.md](PLAN.zh.md)，当前进度见 [STATUS.zh.md](STATUS.zh.md)。

P3j 正在执行：[仅 MoE 连续计算协议](P3J_MOE_COMPUTE_PLAN.zh.md)，目录 `runs/p3j-moe-compute-20260915-p_f_bx4y`。根据用户范围，特征与预测目标都限制在 MoE；动作只做重放核验。只读补采旧 48 条失败续跑的 MoE input/shared/total，预算 1,699 次前向，新增环境动作 0。六个分叉的采集开关和重复核验通过，连续采集、CPU 时序分析和原始数据审计正在进行；尚无完整预测比较或新恢复结论。

P3i 拓扑与复返分析已完成：[固定协议](P3I_TOPOLOGY_PLAN.zh.md)、[结果报告](runs/p3i-topology-20260915-ir7fgyvx/REPORT.zh.md)。仅使用已有 48 条失败续跑和 12 个失败定点反事实状态；新增模型前向与环境动作均为 0。主设置下，原策略和短窗口都各有 12/24 条在路由参考邻域中末次离开后未观察到返回，但全部仍失败；21/24 个配对分类相同。保持主表示、仅改变三档半径，两臂都只有 4/24 条分类稳定，不能把当前拓扑指标当作可靠脱困判据。

P3i 的 432 个设置判定、9,504 个打乱对照和 24 个等效计算匹配对已独立复核；九张图已生成。十二个条件对照均呈现“路由四个不同点、有效计算两个等效类”，因此拓扑处理不会自动解决路由与计算分离的问题。P3h 没有连续有效 MoE input/total，本轮没有验证其计算拓扑，也没有产生新的恢复成功率；[主分类图](runs/p3i-topology-20260915-ir7fgyvx/primary-regions.png)、[条件等效图](runs/p3i-topology-20260915-ir7fgyvx/conditional-equivalence.png)。

P3i 最终 102 项测试通过，75 个产物已[封存](runs/p3i-topology-20260915-ir7fgyvx/verification.json)；175 个冻结来源、7,116 个旧封存文件和原 9500 服务身份保持不变。拓扑计算使用独立目录中的 GUDHI 3.13.0，不修改共享 Python 环境。

P3h 失败子集已完成并通过独立审计：[短时重观测与重规划协议](P3H_REPLAN_WINDOW_PLAN.zh.md)、[失败样本结果](runs/p3h-replan-window-20260915-65syu8x3/FAILURE_ONLY_REPORT.zh.md)。按用户最新要求不补跑原成功状态。6 个原失败状态、4 组配对噪声、两组共 48 条完整续跑：原策略成功 0/24，短窗口成功 0/24，救回 0/24；仅新噪声也是 0/18。F=10、H=10 不变，只在 20 个物理步内改为 h=5。当前设置未显示恢复收益，未扩展 MoE 在线时机对照或调整阈值。

P3h 原计划为 64 条，未完成原成功父轨迹的终止步数校验错误完整保留并排除；不把原完整采集标记为完成，也不声称取得了结束时参数内容哈希。失败子集为 1,502 次调用，含 6 次重复校验；另有未完成分支 12 次调用，资源账共 1,514 次。24 对分叉状态、首个提案及前五步一致，6 条历史噪声原策略全程复现；原 9500 服务保持运行，实验模型已退出。结果边界和核验入口见上述报告。

P3h 48 段失败样本视频完整解码，91 项最终测试通过；1,927 个产物已[封存](runs/p3h-replan-window-20260915-65syu8x3/verification.json)，167 个冻结来源与 5,188 个旧封存文件保持不变。仅失败子集完成，不构成原成功轨迹安全性或 MoE 在线时机价值的验证。

最新 P3g 已完成：[交叉方法终局续跑协议](P3G_CROSSOVER_ROLLOUT_PLAN.zh.md)、[结果报告](runs/p3g-crossover-rollout-20260915-1gmt0y3o/REPORT.zh.md)。等 9521 验证自然结束后执行 5 场景四组共 20 条真实轨迹、790 次模型前向；正常干预、改路由回填原生、原门控回填候选均救回 0/4、误伤 0/1。实际介入为 58/58、57/58、58/58；输出回填有 56 次旧筛查不通过但本轮仍执行，没有新增成功，完整轨迹调用量增加 48.13%。五个场景的实际路由历史都让两种 O1 方法从第二次介入起产生不同动作，但没有终局收益。

P3g [总览图](runs/p3g-crossover-rollout-20260915-1gmt0y3o/crossover-rollout.png)、五场景终局及 MoE 曲线、20 段实际视频与 [全量审计](runs/p3g-crossover-rollout-20260915-1gmt0y3o/audit.json) 已完成。116 个交叉匹配对、40 次控制验证和 81 项单元测试通过，976 个产物已[封存](runs/p3g-crossover-rollout-20260915-1gmt0y3o/verification.json)。原参数内容、旧九轮 4,202 个产物、158 个冻结来源及原 9500 身份未变；首次资源失败完整保留。采集约 58.37 分钟、run 约 9.1 GiB，独立模型已退出。样本仍为既有开发场景，不构成泛化或安全保证；[执行状态](P3G_PREPARATION_STATUS.zh.md)。

上一轮 P3f 已完成：[MoE 交叉对照协议](P3F_CROSSOVER_PLAN.zh.md)、[完整输出边界修订](P3F_CROSSOVER_V2_PLAN.zh.md)、[结果报告](runs/p3f-crossover-v2-20260915-ihpn5n13/REPORT.zh.md)。15 个固定输入、135 次正式前向，另有初始失败批次 5 次和范围定位 2 次，共 142 次。只研究 MoE，不启动环境、不读取模拟器状态、不修改权重。改路由但回填原生输出，动作与原生完全一致，仍通过瞬时筛查 15/15；原门控回填候选输出，动作与正常干预完全一致，筛查却为 0/15，当步 FP32 复算为 1/15。这说明交叉干预下路由指标与有效计算可以分离，不等于路由无用或已经恢复任务。

P3f [对照图](runs/p3f-crossover-v2-20260915-ihpn5n13/crossover.png)、[原始张量独立审计](runs/p3f-crossover-v2-20260915-ihpn5n13/audit.json)、[初始数值耦合定位](runs/p3f-crossover-v2-20260915-ihpn5n13/scope-diagnostics/summary.json) 已保存。正式输出回填覆盖 L12-15 的全部 11 个 token，门控偏置仍只影响动作 token；初始失败批次完整保留已有文件。74 项测试通过，169 个产物、151 个冻结来源和 3 个后处理脚本已[封存](runs/p3f-crossover-v2-20260915-ihpn5n13/verification.json)。旧八轮 4,033 个产物、参数内容哈希与共享服务未变；修订版采集约 9.88 分钟、run 约 2.5 GiB。

上一轮 P3e 已完成：[传递链诊断协议](P3E_MECHANISM_PLAN.zh.md)，[结果报告](runs/p3e-mechanism-20260915-9u_9j9b8/REPORT.zh.md)。15 个同观测配对状态、105 次新模型前向，另重建 P3d 十条保存轨迹的 4,815 个物理状态；新增环境动作 0，不是新恢复率试验。路由合成输出确实变化，但整体表示的相对变化较小；冻结案例至迟在报警前 50 步已出现物理停滞；倒置案例组合提前 73 步关屉，却未把碗放进去。这支持继续研究任务阶段与控制目标的对齐，不证明全部失败根因已识别。

P3e [响应图](runs/p3e-mechanism-20260915-9u_9j9b8/mechanism-response.png)、逐场景帧序列和目标时间线、[独立复核](runs/p3e-mechanism-20260915-9u_9j9b8/audit.json) 已保存。64 项测试通过，182 个产物、138 个冻结来源和 4 个后处理脚本已[封存](runs/p3e-mechanism-20260915-9u_9j9b8/verification.json)。固定残差的绝对扰动基本完整保留，不能把不同分母下的百分比下降解释成信号被消除。采集约 6.72 分钟，run 约 1.6 GiB，旧七轮 3,851 个产物和共享服务未变。

上一轮 P3d 已完成：[多机制筛查控制器真实续跑协议](P3D_GATED_ROLLOUT_PLAN.zh.md)，[结果报告](runs/p3d-gated-rollout-20260915-t9k1v766/REPORT.zh.md)。五个既有报警后状态、四臂，共 20 条实际仿真轨迹和 649 次模型前向；三种控制器均救回 0/4、误伤 0/1。mobility / mobility_balance / 匹配随机实际介入 15/58、49/58、0/58；组合让冻结分数连续 15 个查询低于阈值，仍未救回。误报成功样本为原生 323 步、单独解冻 324 步、组合 322 步、随机 323 步；完整在线轨迹调用量增加约 24.07%，没有任务恢复收益。

P3d 的 [总览图](runs/p3d-gated-rollout-20260915-t9k1v766/rollout-overview.png)、五状态终局与分量曲线、20 段实际视频及 [独立审计](runs/p3d-gated-rollout-20260915-t9k1v766/audit.json) 已保存。56 项测试通过，845 个产物完成 [封存](runs/p3d-gated-rollout-20260915-t9k1v766/verification.json)；采集约 39.46 分钟，run 约 276 MiB。独立模型正常退出，原服务与旧六轮的 3,006 个产物及冻结源码未变。本轮仍是目的性开发评估，不将指标可控或 0/1 误伤解释为恢复有效性或安全保证。

上一轮 P3c 已完成：[多机制响应矩阵协议](P3C_RESPONSE_MATRIX_PLAN.zh.md)，[结果报告](runs/p3c-response-matrix-20260915-463ir6kj/REPORT.zh.md)。五个既有报警后状态、三种定向 gate 算子和两种组合，正反两档与等请求能量随机对照；175 次实际模型前向、150 个不同非零探针，新增环境动作 0。mobility 与 balance 的正向降目标、反向升目标均在 5/5 成立，但存在跨指标副作用；两个组合的即时多指标筛查均通过 4/5，inversion 状态仍有冻结冲突。当时将 mobility_balance 列为优先续跑候选，后续实际控制结果见 P3d；筛查通过数不是救回率。

P3c 的 [响应矩阵](runs/p3c-response-matrix-20260915-463ir6kj/response-matrix.png)、每状态图、真实 logits / 专家分配 / 动作与 [独立审计](runs/p3c-response-matrix-20260915-463ir6kj/audit.json) 已保存；44 项测试通过，207 个产物完成 [封存](runs/p3c-response-matrix-20260915-463ir6kj/verification.json) 并只读复核。采集含加载约 4.80 分钟，run 约 74 MiB。独立实验模型正常退出，共享服务身份、旧五轮的 2,799 个产物与冻结源码均未变。

上一轮 P3b 已完成：[边缘选择与 v8.2 报警头定向纠正协议](P3B_HEAD_CONTROL_PLAN.zh.md)，[结果报告](runs/p3b-head-control-20260915-s39ow9qi/REPORT.zh.md)。四类报警各一个失败样本，另加一个误报成功样本；只替换报警后的一个 10 步动作块，所有四候选均实际续跑。20 条仿真轨迹、457 次模型前向，中心、边缘、单头降分/升分与综合风险选择均救回 0/4、误伤 0/1；候选池 oracle 也为 0/4。curvature / inversion 的下降延续到后续查询，仍未救回；freeze 没有更低候选，turbulence 的当次分数被历史窗口遮蔽。这不是“所有真实原因均已消除”的证明，也不与 P3 的多次五步介入直接作单变量比较。

P3b 的 [指标曲线](runs/p3b-head-control-20260915-s39ow9qi/head-trajectories.png)、20 段视频及独立审计已保存；35 项测试通过，609 个产物完成 [封存](runs/p3b-head-control-20260915-s39ow9qi/verification.json)。run 约 97 MiB，采集约 35.31 分钟。独立实验模型已退出，原 9500 继续运行，旧四轮数据与源码保持不变。

上一轮 P3：[在线纠正开发试验协议](P3_ONLINE_PLAN.zh.md)，[结果报告](runs/p3-online-20260915-tc0mhlig/REPORT.zh.md)。5 个固定父轨迹、4 个实验臂已完成，17 次实际环境运行、881 次模型推理；三种纠正均救回 0/3，误报成功样本未变失败但慢了 48--49 步。各臂均为 2/5 成功，路由中心方案完整轨迹调用量增加约 80.5%。[独立审计](runs/p3-online-20260915-tc0mhlig/audit.json)通过，模型进程已退出；共享服务与原模型不变。这是目的性开发试验，不是独立留出评估，也不是原计划中学习型 P3 的完成声明。

已完成 P2a：[动作候选覆盖预检协议](P2A_COVERAGE_PLAN.zh.md)，run 为 `runs/p2a-coverage-20260914-4i18r353`。8 条新 init39 父轨迹已完成，正式 393 次模型前向，另有 1 次真实 logits 审计重放，共 394 次。这是物理续跑前的几何诊断，不是恢复率测试；以下旧阶段描述按当时状态保留。

P2a [结果报告](runs/p2a-coverage-20260914-4i18r353/REPORT.zh.md)、[覆盖图](runs/p2a-coverage-20260914-4i18r353/coverage.png) 和 [独立审计](runs/p2a-coverage-20260914-4i18r353/independent-audit.json) 已生成。状态 gate 的参考覆盖增益为 0.61%，噪声对照 17.83%；这不是恢复成功率。精度诊断和两次审计修正见 [执行说明](COVERAGE_EXECUTION_NOTES.zh.md)。

P2a 最终 21 项测试通过，426 个产物完成 [哈希封存](runs/p2a-coverage-20260914-4i18r353/verification.json)，run 约 90 MiB。不要在封存 run 上重新执行会写产物的分析、审计、诊断或 verify 命令。

首轮：[`runs/p1a-20260914-my_4lkxd`](runs/p1a-20260914-my_4lkxd)。配置在采集前冻结，原始候选和分析结果保存在该目录。

首轮 P0/P1a 已完成并通过数据审计；预定主方法未通过继续门槛，当时停止自动推进。详见 [P1a 结果报告](runs/p1a-20260914-my_4lkxd/REPORT.zh.md) 和 [阶段判定](STATUS.zh.md)。这不是物理恢复率测试。

用户随后要求执行 gate 因果探针，独立 P1b 已完成：8 条父轨迹、512 个不同非零探针、569 次正式模型前向，另有 5 次预检前向；13 项测试和独立审计通过。P1a 阴性判定保持不变，P2-P4 未启动。

- [P1b 冻结协议](P1B_GATE_PLAN.zh.md)
- [P1b 结果报告](runs/p1b-gate-20260914-1nvvkzw2/REPORT.zh.md) 与 [响应图](runs/p1b-gate-20260914-1nvvkzw2/gate-response.png)
- [P1b 独立审计](runs/p1b-gate-20260914-1nvvkzw2/independent-audit.json) 与 [最终校验](runs/p1b-gate-20260914-1nvvkzw2/verification.json)
- [运行方式、精度与结果边界](GATE_EXECUTION_NOTES.zh.md)

## 实现

- `moe_compute_capture.py`、`moe_compute_protocol.py`、`run_moe_compute_experiment.py`：P3j 只读 MoE 端口补采、因果缩放、固定交叉对照及任务分组预测协议。
- `analyze_moe_compute_experiment.py`、`audit_moe_compute_experiment.py`、`report_moe_compute_experiment.py`、`finalize_moe_compute_experiment.py`：P3j 连续计算分析、独立复算、统一报告和封存。
- `topology_protocol.py`、`test_topology_protocol.py`、`run_topology_experiment.py`：P3i 因果复返图、尺度敏感性、持久同调与时间打乱对照，仅分析旧失败数据。
- `audit_topology_experiment.py`、`report_topology_experiment.py`、`finalize_topology_experiment.py`：P3i 独立距离/图/时间判定复算、报告与图表、旧数据保护及封存。
- `crossover_rollout_protocol.py`、`run_crossover_rollout.py`、`test_crossover_rollout.py`：P3g 固定窗口、共同保护、真实历史反馈和四组终局续跑；20 条实际轨迹已完成。
- `audit_crossover_rollout.py`、`plot_crossover_rollout.py`、`finalize_crossover_rollout.py`：P3g 原始数据审计、反馈分歧定位、终局图表与保护封存，已完成端到端验证并冻结。
- `crossover_protocol.py`、`crossover_capture.py`、`run_crossover_experiment.py`：P3f 初始动作 token 输出隔离协议与失败批次实现，已冻结保留。
- `crossover_capture_v2.py`、`run_crossover_experiment_v2.py`：完整 MoE 输出回填、两次范围定位、固定 15 状态交叉和真实参数内容哈希。
- `test_crossover_protocol.py`、`test_crossover_v2.py`：范围、原样回填、hook 顺序与清理、参数哈希和固定预算测试。
- `analyze_crossover_experiment.py`、`plot_crossover_experiment.py`、`finalize_crossover_experiment.py`：路由与计算分离审计、精度敏感性、图表和包含初始失败批次的保护封存。
- `protocol.py`：固定随机流、结构化路由距离、父轨迹等权诊断回归。
- `run_experiment.py`：准备、采集、磁盘完整路由核对、分析和复核。
- `test_protocol.py`：协议单元测试。
- `audit_independently.py`：不导入协议实现，直接从原始候选重新验证距离、随机流、回归最优性和误差。
- `gate_runtime.py`、`gate_capture.py`：独立模型与请求级 gate 干预、实际专家 dispatch 精确核对。
- `gate_protocol.py`、`run_gate_experiment.py`：固定范围、随机方向、幅度、采集与分析。
- `test_gate_probe.py`、`audit_gate_independently.py`：gate 单元测试与独立 CPU 原始数据复算。
- `plot_gate_results.py`：从已保存的分析结果绘图。
- `coverage_protocol.py`、`run_coverage_experiment.py`：P2a 固定候选池、独立参考动作和覆盖分析。
- `test_coverage_protocol.py`、`audit_coverage_independently.py`、`plot_coverage_results.py`：P2a 测试、独立复算与绘图。
- `audit_coverage_independently_v2.py`：最终 P2a 独立审计，修正有符号零重建并基于真实 logits 验证加法和 softmax；旧审计实现保留。
- `diagnose_coverage_logits.py`：采集后一次原候选 logits 审计重放，不参与候选覆盖分析。
- `online_protocol.py`、`online_env_worker.py`、`run_online_experiment.py`：P3 冻结路由选择、隔离仿真工作进程和实际闭环。
- `test_online_protocol.py`、`audit_online_experiment.py`：P3 随机流与选择测试、候选决策和真实动作独立复算。
- `head_control_protocol.py`、`run_head_control_experiment.py`：P3b 单次干预与七种固定 MoE 候选选择，枚举四候选真实后缀。
- `test_head_control_protocol.py`、`audit_head_control_experiment.py`：对应头反向控制、前缀不变测试，以及实际分数、动作和候选池上限复核。
- `plot_head_control_results.py`、`finalize_head_control_experiment.py`：P3b 终局与指标轨迹图，完整产物及旧实验保护检查。
- `response_matrix_protocol.py`、`run_response_matrix_experiment.py`：P3c 等请求能量定向 gate 干预、匹配随机、正反两档与实际 logits 采集。
- `test_response_matrix_protocol.py`、`audit_response_matrix.py`：方向、能量、可复用历史与筛查测试，以及真实 gate / 五分量独立复核。
- `plot_response_matrix.py`、`summarize_response_matrix.py`、`finalize_response_matrix.py`：总体与逐状态矩阵、剂量和精度敏感性、产物及旧实验保护封存。
- `gated_rollout_protocol.py`、`run_gated_rollout_experiment.py`：P3d 固定预算的即时多指标筛查、实际 gate 候选与四臂仿真闭环。
- `test_gated_rollout_protocol.py`、`audit_gated_rollout_experiment.py`：窗口、因果历史和筛查测试；全部路由、实际选择、动作与终局独立复核，支持只读部分审计。
- `plot_gated_rollout_results.py`、`finalize_gated_rollout_experiment.py`：P3d 实际终局与五分量轨迹、最终测试及旧六轮保护封存。

运行环境为 `/data/venv311/bin/python`。本目录不改变已下载的上游方法、不训练 VLA、不启动或重启共享模型服务。

## 执行入口

协议测试不调用 VLA，也不重写已归档结果：

```bash
/data/venv311/bin/python -m unittest discover -s /data/coding/moe-control-experiments -p 'test_*.py' -v
```

P1b 本轮按 `gate_preflight.py`、`run_gate_experiment.py prepare`、`collect --run <run>`、`analyze --run <run>`、独立审计、绘图、`verify --run <run>` 的顺序执行。主脚本支持 `--help`；审计和绘图脚本的位置参数为 run 路径。绘图使用已有的 `/data/miniconda/envs/torch/bin/python`。

这是一份冻结实验实现，不是无限重复采集服务；源批次、预检引用和请求 ID 固定。当前请求 ID 已使用，准备阶段会拒绝重复采集。后续实验需在新协议和新实现中分配新请求 ID，不能修改已冻结文件或覆盖已有 run。分析、审计和最终校验均会写产物，不要在已封存的 run 上重复执行。

已启动的实验不自动覆盖或补采；只读推理诊断不能解释成物理恢复实验。
