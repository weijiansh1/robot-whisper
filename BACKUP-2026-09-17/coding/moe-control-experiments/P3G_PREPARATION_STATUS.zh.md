# P3g 准备与执行状态

2026-09-15: 用户授权等待后，9521 验证已自然结束并释放显存。正式目录 `runs/p3g-crossover-rollout-20260915-1gmt0y3o` 于 18:37:37 开始、19:35:59 完成采集 (北京时间)。20 条轨迹、790 次前向完成，三种干预均救回 0/4、误伤 0/1；完整审计、图表及 976 个产物的封存已完成。

- [x] 固定 5 个父场景、四组实际续跑、同一 12 查询介入窗口和统一预测动作保护。
- [x] 接入当前观测/噪声供体、完整 MoE 输出回填、实际路由历史提交和精确计算检查。
- [x] 81 项单元测试通过；另用 15 组 P3f 已保存动作核对控制器与独立审计的保护判定，未调用模型。
- [x] 独立审计、终局图表与封存脚本通过新完整轨迹的端到端验证。
- [x] 用户授权等待其他验证自然结束，再按原冻结协议执行；不终止其他服务。
- [x] 9521 验证自然结束；原 9500 保留，可用显存恢复至 17,764 MiB，磁盘余量约 34 GiB。
- [x] 模型加载与参数基线哈希完成；首场景四组及其首次控制检查完成。
- [x] 新 run 20 条轨迹、790 次前向完成；全量独立审计、116 个交叉匹配对和 40 次控制验证通过。
- [x] 参数内容 SHA256 前后一致，独立进程及仿真 worker 正常退出，显存恢复至原服务占用。
- [x] 20 段视频完整解码，11 张图已检查；结果报告、81 项最终测试与封存完成。
- [x] 旧九轮 4,202 个产物、158 个冻结来源和原 9500 服务身份不变，首次资源失败的 4 个文件保留。

结果: [报告](runs/p3g-crossover-rollout-20260915-1gmt0y3o/REPORT.zh.md)、[总览图](runs/p3g-crossover-rollout-20260915-1gmt0y3o/crossover-rollout.png)、[独立审计](runs/p3g-crossover-rollout-20260915-1gmt0y3o/audit.json)、[封存](runs/p3g-crossover-rollout-20260915-1gmt0y3o/verification.json)。这是 5 个既有开发父场景，不是独立留出评估。

首次准备目录为 `runs/p3g-crossover-rollout-20260915-lq1_j641`。其 `failure.json` 记录了加载前的显存保护拒绝: `Insufficient free memory for bounded isolated load`。实际模型前向 0 次，环境动作 0 步，未创建仿真 worker。

首次资源核查时原 9500 服务仍运行，另有独立任务 `run_alarm_model_experiments.py --port 9521` 及其验证任务占用剩余显存。GPU 当时可用约 1,324 MiB，低于隔离加载要求的 17,152 MiB。随后等待该验证生成自己的完成标记并自然退出，未停止或修改这些进程，也未创建或修改其完成标记。

现有服务未提供本轮所需的完整输出回填接口，因此不能通过向它发送普通推理请求替代隔离实验，也不能把其他实验的记录混入本轮。

## 已执行流程

资源释放后重新核对 GPU 余量和原服务身份，运行 `run_crossover_rollout.py prepare` 生成新目录，随后对新目录执行 `collect --run <new-run>`。首次启动失败目录保留，未删除 `failure.json`，未在其上补采。

完成采集后依次执行了 `audit_crossover_rollout.py`、`plot_crossover_rollout.py` 和 `finalize_crossover_rollout.py`，均传入正式新目录的 `--run`。绘图使用 `/data/miniconda/envs/torch/bin/python`，其他脚本使用 `/data/venv311/bin/python`。

协议: [P3G_CROSSOVER_ROLLOUT_PLAN.zh.md](P3G_CROSSOVER_ROLLOUT_PLAN.zh.md)。正式目录已封存，不可重新执行会写产物的采集、审计、绘图或封存命令；新的实验必须另建目录。
