# 方法与复现

## 1. 实验问题

本项目只回答三个受限问题：MoE routing 是否包含 Trap 信息；信号是否早于物理 loop onset；该信号触发 fresh-noise continuation 时能否提高恢复率。它不训练 Trap classifier，也不把阈值 calibration 算作训练。

## 2. 路由观测

每次重规划记录 `8 HB layers x 10 flow steps x 11 suffix tokens x 32 experts` 的 full softmax。分析使用模型 HB 12--15 层、state token 0、action tokens 1--10。state token 在 flow 轴上近似不变，因此 volatility 和 acceleration 只在 action tokens 上定义。

八个预先固定的信号是 weighted recurrence、lag periodicity、late-flow volatility、route acceleration、gate entropy、top1-top2 margin、state-action gap、deterministic macro stickiness。所有 feature 在读取 outcome/Trap label 前确定。

`late-flow volatility` 是 flow 6--9 中相邻路由分布的平均 weighted-Jaccard distance。`route acceleration` 是完整 10-step route square-root coordinates 的二阶差分范数。前者测末段是否仍切换，后者测整个路由轨迹是否持续弯折。

## 3. Trap 真值与统计

Corpus A 使用 520-step 稠密物理轨迹定义 loop/static onset。Corpus B 缺少同等级稠密轨迹，因此使用先在 A 上验证的 query proxy；报告始终把 B 标为 proxy。

事件样本与同 snapshot/init-state、同绝对 query 的所有 no-same-event 轨迹比较，包括其他失败。每个 routing signal 还对旧 d9 Hellinger mobility 做组内秩残差。正式 `p_maxT` 同时校正 `2 event types x 8 signals x 5 leads = 80` 个单元，置换次数 2000，seed 20260903。

严格复现要求 A/B 具有完全相同的 event、signal、lead、方向，并各自通过 maxT。最终只有：

1. loop / late-flow volatility / -2 / high；
2. loop / route acceleration / -2 / high；
3. static / lag periodicity / 0 / high。

## 4. Snapshot-Fork

先运行一条确定性 trunk，并在每个 query 边界保存 simulator、controller 和 policy input。loop onset 完全由物理轨迹确定，不读取 MoE。随后从 onset-4、onset-2、onset 三个完全相同的保存状态分别启动 K=8 fresh-noise continuations。candidate `k` 在三个 offset 使用相同的相对 noise stream，使窗口比较保持配对。

三个 offset 是三个不同物理状态，且共享噪声流，因此不能把三个 0/8 合并成 24 个独立 Bernoulli 样本。每个 0/8 的 Wilson 95% CI 为 `[0, 0.324]`。

## 5. 复现离线分析

默认工作区是 `/home/jovyan/work/himoe-vla`，也可设置 `HIMOE_VLA_WORKSPACE`。原始 A/B 数据不在本目录重复保存，路径见 `configs/data_sources.json`。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python himoe-vla_trap/code/analyze_trainfree_signal_matrix.py \
  --nperm 2000 \
  --output-dir /tmp/himoe_trap_signal_rerun
```

若目标 output 已包含匹配 schema 的 `intermediate/corpus_A_route_features.npz` 和 `corpus_B_route_features.npz`，会复用缓存；`--rebuild-cache` 强制从 full Zarr 重新提取。

## 6. 复现恢复实验

模型服务和 LIBERO 客户端使用两个既有环境。下面的 `<new-output>` 必须是新路径，collector 拒绝覆盖已有 manifest。

终端 A：

```bash
MOEVLA_DATA_HOME=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/moevla-data \
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/model/bin/python -u \
  himoe-vla_trap/code/serve_with_full_route_capture.py \
  --port 7777 --gpu 1 --suite long \
  --checkpoint-dir himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10 \
  --upstream-root himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA \
  --libero-wrist-layout checkpoint-right \
  --store-full-probs --return-full-probs \
  --out <new-output>/server
```

终端 B，在服务端打印 `serving on ws` 后运行：

```bash
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/libero/bin/python -u \
  himoe-vla_trap/code/collect_snapshot_fork_recovery.py \
  --port 7777 \
  --libero-root himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO \
  --out <new-output>/init03_seed20260903_loop
```

结束时向服务进程发送 `SIGTERM`，使最后的 Zarr group 与 capture summary 正常 flush。

## 7. 复现失败抓取 AS/HB 分析

该分析自动从 snapshot 0 定位 candidate 06 的近物闭合但未抬升事件，并选择同 snapshot 中闭合后成功抬升目标 pot 的 7 条 sibling rollout。它不拟合阈值或 predictor。

```bash
python himoe-vla_trap/code/analyze_failed_grasp_moe_dynamics.py
```

AS 审计覆盖 route store 全部 16,180 行；HB 事件比较使用模型前层 2--5、后层 12--15、完整 10 flow、10 个 action tokens。原 capture 未开启 `store_hidden`，所以该实验比较的是 routing probability，不是 hidden vector 或 expert output。

## 8. 复现 belief-state mismatch 分析

该分析继续使用失败抓取的 1 条 candidate 和 7 条同 snapshot 成功 sibling，把真实闭合 query 对齐为 0。它从三个相互独立的观测面定义 mismatch：末端运动但目标物体不随动；action chunk 仍接近成功搬运阶段；HB state/action token routes 相对成功阶段发生分裂。

```bash
python himoe-vla_trap/code/analyze_belief_state_mismatch.py \
  --config himoe-vla_trap/configs/belief_state_mismatch.json
```

比较覆盖 HB 层 2--5、12--15，完整 10 flow、1 个 state token 和 10 个 action positions。失败到成功中心的 Hellinger distance 与 7 个成功分支的 leave-one-out 距离范围比较；没有训练 predictor 或用失败标签拟合阈值。`belief state` 是物理/动作分歧的行为推断，因为 capture 没有显式 belief variable 或 contact sensor。后续第 15 节的同噪声反事实证明新反馈确实改变 action routing，所以该观察不能再解释为 causal feedback blockage。

## 9. 复现 train-free belief selector

`transition_split_v1` 使用 phase-matched 健康路由的经验最大值校准 layer-5 state/action gap、back-layer chunk jump 和 front action-route distance。规则只做三个逻辑比较，不拟合权重，也不读取失败标签。

同一脚本还在 5 个 `right-16x32` 网格上评估预先固定的 min-layer5-gap K8 candidate ranker。候选分数只读取第一个 policy query 的完整 soft routing；success 在选择结束后单独聚合。CI 按 task 内 init-state cluster bootstrap，置换在每个 task/state 的 32 个候选内进行。

```bash
python himoe-vla_trap/code/evaluate_trainfree_belief_selector.py \
  --config himoe-vla_trap/configs/trainfree_belief_selector.json
```

## 10. 复现 MoE-only 在线审计与视频

在线选择器代码为 `code/moe_only_online_selector.py`，collector 为 `code/collect_online_moe_only_alarms.py`。它们只使用完整 HB routing 和成功 routing reference；物理轨迹在 episode 结束后才落盘。

已有三批结果的失败分型和客户端/服务端逐元素一致性可直接复算：

```bash
python himoe-vla_trap/code/audit_moe_only_online_failure_types.py
```

事后审计同时检查两口 moka pot。目标代理的冻结条件是闭合距离 `<0.16 m`、此后 EEF 位移 `>=0.10 m`、目标总位移 `<0.01 m`、最大分离 `>=0.15 m`。这些物理量不进入 selector。

GPU5 扩展批次沿用 GPU4 preliminary 成功轨迹建立并在运行前冻结的 H20 reference；24 条轨迹全部按预定 init schedule 跑完，不按 alarm 或 outcome 停止。方法失败定位可复算：

```bash
python himoe-vla_trap/code/diagnose_moe_only_alarm_method.py
```

该脚本只在事后使用物理标签做事件对齐和评价，不重新拟合在线 selector。它比较 `any raw`、连续 2 次、`2-in-5`、`2-in-10`、`3-in-10`，并对 12 个路由特征 x 4 个事件窗口执行 5000 次 episode-label permutation 和 48 项 maxT 校正。GPU5 对冻结规则是 held-out；所有替代时序规则和最佳特征都是 posthoc diagnostics，不能回称 held-out detector。

三条代表视频通过重放已保存 action 生成；manifest 要求逐步 simulator state 与原轨迹误差为 0：

```bash
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=4 \
LD_LIBRARY_PATH=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/system-libs/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
himoe-vla-cache/himoe-libero-bridge/cache/envs/libero/bin/python \
  himoe-vla_trap/code/render_moe_only_review_videos.py \
  --output /tmp/himoe_moe_only_review
```

上面的重放视频是早期 GPU4 批次的 posthoc review，不是在线报警触发视频。GPU5 的 3 次正式报警由 collector 当场保存 clean/full/clip 视频，位于：

```text
results/moe_only_online_alarm/gpu5_extended_random_seed20260907/
  episode_011_init_03_random/videos/
  episode_014_init_20_random/videos/
  episode_022_init_03_random/videos/
```

三条均在报警后继续原样执行 action chunk；视频红框是正式 alarm，黄色框是单次 raw reject。

## 11. 复现 MoE dynamics v2

先从同一 5 条 H20 健康参考重建阈值，再做旧数据开发回放：

```bash
python himoe-vla_trap/code/build_moe_dynamics_calibration.py
python himoe-vla_trap/code/evaluate_moe_dynamics_selector.py
```

第一条命令只使用健康 routing，输出阈值 `1.0457019658379743`；第二条会读取失败标签做开发评价，因此不能作为 prospective 结果。

seed `20260908` 在线批次在运行前已冻结于 `configs/moe_dynamics_online_alarm_v2.json`。服务端与上一节相同，端口为 8855，输出到 `results/moe_dynamics_online_alarm/gpu5_server_seed20260908`。collector 的关键新增参数为：

```bash
--selector-version back_front_route_acceleration_v2 \
--dynamics-calibration himoe-vla_trap/results/moe_dynamics_online_alarm/calibration_v2/calibration.json \
--healthy-reference himoe-vla_trap/results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz \
--episodes 24 --seed 20260908 --episode-id-base 1640000000
```

完成并正常停止服务端后，复算物理分型、同轨迹 v1 重放、Wilson CI、McNemar 检验和客户端/服务端一致性：

```bash
python himoe-vla_trap/code/audit_moe_dynamics_online_results.py
```

该审计的物理轨迹仍只用于 rollout 后的标签，不进入 v2 selector。v2 的 feature 设计受 seed `20260907` 启发，但阈值只用健康数据；seed `20260908` 是第一次冻结规则后的 prospective test。

## 12. 复现 cache_new 的严格 MoE-only 回放

第一条命令按两阶段协议工作：预测阶段只打开 `hb_router_probs`、episode 边界和冻结健康参考，先写出逐 query/episode 预测文件；关闭预测阶段后才加载 outcome。设置 BLAS 单线程以免多 worker 各自创建大量线程。

先复算已归档的 A/B/prospective 结果：

```bash
PYTHONPATH=himoe-vla_trap/code \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/evaluate_moe_dynamics_large_offline.py \
  --corpora A B prospective \
  --output himoe-vla_trap/results/moe_dynamics_online_alarm/large_offline_replay \
  --workers 24
```

再单独复算 `cache_new`，避免覆盖上一个输出：

```bash
PYTHONPATH=himoe-vla_trap/code \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/evaluate_moe_dynamics_large_offline.py \
  --corpora C_cache_new \
  --output himoe-vla_trap/results/moe_dynamics_online_alarm/cache_new_task8_replay \
  --workers 24
```

第二条命令只对已经写出的预测做事后物理分型、事件时序、Fisher/AUC、旧 B 条件重合以及 query proxy 有效性审计；它不会重算或修改报警：

```bash
PYTHONPATH=himoe-vla_trap/code \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/audit_cache_new_moe_dynamics.py
```

正式数字只使用 `cache_new/.../right-50x8-20260903`，因为其元数据状态为 `complete`。审计当时另外两个 run 仍为 `running`，故排除。物理状态每 query 采样一次，漏抓标签是 kinematic proxy，不是逐 action contact ground truth。

## 13. 复现任务难度与报警加权实验

任务难度被限定为

```text
P(failure | checkpoint, task, init/noise distribution, controller, horizon)
```

不把它解释为任务固有属性。37 个完整 task run 各含 50 initial states x 8 flow-noise seeds。seeds 1000--1003 和 1004--1007 双向交换：一半建立 route/task reference、经验失败先验、健康 route 分位数与报警阈值，另一半只评价；随后反向复算。

脚本首先只从 atlas 读取 `task, episode` 身份，然后从每条轨迹的第一个 query 提取 `8 layers x 10 flow x 11 tokens x 32 experts` HB routing。15 个标量和 final-flow action-token `8 x 32` embedding 写盘后，脚本才加载含 outcome 的完整 atlas。首 query route 标量以另一半 seed 的任务失败率为 target，执行 5,000 次 task-label permutation 和 15 项 BH 校正。route-KNN 始终排除目标任务，用其他任务在 source half 的失败率预测 target half。

预算实验使用 atlas 已固定的 episode 50%--90% MoE 汇总量。detector 为 instability 与 lock-in 两个经验百分位 head 的等权均值，不拟合 feature weight。每任务成功报警预算为

```text
alpha_task = clip(scale * prior_task**gamma, 0.005, 0.20)
```

其中 `scale` 使 calibration successes 加权平均预算为 5%；`gamma=0, 0.5, 1` 分别对应 uniform、经验平方根和经验线性分配。第四个版本用首 query route 的 leave-one-task 5-NN 推出 prior。经验 prior 使用 Jeffreys smoothing `(failures + 0.5) / (episodes + 1)`。因此整个实验无梯度训练，但经验难度与 detector calibration 明确使用另一半 rollout labels，不是无标签或纯 MoE-only 方法。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/analyze_task_difficulty_weighting.py \
  --bootstrap 5000 --permutations 5000
```

若 `first_query_moe_features.csv` 和 embedding 已由同一脚本严格生成，可加 `--reuse-early`。结果写入 `results/task_difficulty_weighting/`，冻结设置见 `configs/task_difficulty_weighting.json`。该实验是 episode-level allocation stress test，不是 v2 在线逐 query 时序回放。

## 14. 复现无任务先验的自参照报警

该规则不用任务 ID 或跨轨迹 healthy bank。`q1--q6` 建立当前 episode 自身 MoE 路由基线，最近 3 个 query 同时满足 state response collapse、action planning churn 和 state/action gap 三个 effect-size gate 才报警。配置明确声明运行时禁用动作、物理状态、reward、success 与 timeout。

离线扫描严格先生成 route-only 预测，再加载 outcome：

```bash
PYTHONPATH=himoe-vla_trap/code \
python himoe-vla_trap/code/evaluate_moe_self_reference_selector.py
```

GPU6 在线 collector 使用同一 `self_reference_coupling_collapse_v3`，两个 init state 分别固定扫描 seeds `1000--1007`；报警后 policy 不变，继续执行到 success 或 300-step horizon。在线轨迹、旧 GPU3 同 seed 条件和事后物体 plateau 由下列命令统一审计：

```bash
python himoe-vla_trap/code/audit_self_reference_gpu6.py
```

物体状态只在审计阶段加载，不进入 selector。三次在模型推理前因 EGL 或 noise shape 失败的工程启动被排除；有效 prospective 恰为 16 条。完整输入隔离、视频和时钟负控见 [TASK_FREE_SELF_REFERENCE_ALARM_ZH.md](TASK_FREE_SELF_REFERENCE_ALARM_ZH.md)。

## 15. 复现同噪声输入版本反事实

该实验不是普通相邻 chunk 比较。对每个相邻输入版本，旧输入和新输入必须使用逐元素相同的 `10 x 24` flow noise；服务端还必须返回每次调用的 `8 x 10 x 11 x 32` HB full-softmax。主采集使用 4 个 noise，因此每个版本对需要 8 次推理。

终端 A 启动 GPU6 服务。`<server-output>` 必须是不存在的新目录：

```bash
MOEVLA_DATA_HOME=/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/moevla-data \
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/model/bin/python -u \
  himoe-vla_trap/code/serve_with_full_route_capture.py \
  --port 8866 --gpu 6 --suite long \
  --checkpoint-dir himoe-vla-cache/himoe-libero-bridge/cache/checkpoints/HiMoE-VLA-Libero-10 \
  --upstream-root himoe-vla-cache/himoe-libero-bridge/cache/upstream/HiMoE-VLA \
  --libero-wrist-layout checkpoint-right \
  --store-full-probs --return-full-probs \
  --out <server-output>
```

终端 B 采集 8 条轨迹的 96 个版本对，并生成主分析：

```bash
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/libero/bin/python -u \
  himoe-vla_trap/code/collect_input_version_counterfactual.py \
  --port 8866 --out <paired-output>

python himoe-vla_trap/code/analyze_input_version_counterfactual.py \
  --capture <paired-output> --output <analysis-output>
```

主分析只使用 routing 建分数；成功/失败字段只用于写完逐记录分数后的评价。正式归档服务端含先行 smoke 的 48 次调用，所以 `control_steps=816`；纯正式复跑应是 768 次。

2x2 模态实验需要停止第一个服务并用新的 `<factorial-server-output>` 重启，随后运行：

```bash
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/libero/bin/python -u \
  himoe-vla_trap/code/collect_input_modality_counterfactual.py \
  --port 8866 --paired <paired-output> --out <modality-output>

python himoe-vla_trap/code/analyze_input_modality_counterfactual.py \
  --paired <paired-output> --modality <modality-output> \
  --output <modality-analysis-output>

CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/jovyan/work/himoe-vla/himoe-vla-cache/himoe-libero-bridge/cache/envs/libero/bin/python \
  himoe-vla_trap/code/audit_input_version_novelty.py \
  --paired <paired-output> --modality <modality-output> \
  --output <modality-analysis-output>
```

第二个服务仅收集 `V10` 和 `V01`，`V00/V11` 复用主采集，所以新增 `40 x 4 x 2 = 320` 次调用。原始像素与物理状态只由最后一个脚本事后读取，用于检验“路由变化小是否只是输入变化小”，不进入报警。

## 16. 复现 Train-free Trap 概率校准

该实验固定使用 `late_flow_volatility_w4`、`route_acceleration_w4` 和 `lag_periodicity_w4`。每条 episode 的 `q4--q11` 只用于自归一化；source 的无标签经验 CDF 把值转换成 percentile。loop/static/trap score、score bins、persistence gate、最小 cell support 和 75% 报警阈值全部写在配置中。

程序先在完全不加载 onset、outcome、action 或物理状态的阶段写出 route-only 特征。随后 source onset 仅用于 Jeffreys Beta-Binomial 条件频率校准；target 概率 CSV 写盘并记录 SHA256 后，target onset 才用于评价。它无梯度、无拟合 feature weights，但不是无标签方法。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/analyze_trainfree_trap_probability.py \
  --config himoe-vla_trap/configs/trainfree_trap_probability.json \
  --output himoe-vla_trap/results/trainfree_trap_probability
```

主评价为 onset 前 risk set 上的 `P(onset within 2 queries)`。另存的 absorbing proxy 把 onset 后所有 query 继续标正，只用于暴露 detection upper bound 与 query-clock 混淆，不是 active Trap duration ground truth。完整结果见 [TRAINFREE_TRAP_PROBABILITY_ZH.md](TRAINFREE_TRAP_PROBABILITY_ZH.md)。

对 `cache_new` 40 个完整任务的冻结迁移压力测试使用相同 source CDF、概率表和特征规则：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/evaluate_trainfree_trap_probability_cache_new.py \
  --config himoe-vla_trap/configs/trainfree_trap_probability_cache_new.json \
  --output himoe-vla_trap/results/trainfree_trap_probability/cache_new_40task \
  --workers 8
```

脚本先由 route-only worker 逐任务生成 `.npz`，再写入不含 outcome/label/onset 的压缩概率表及其 SHA256；只有该阶段关闭后才读取 client summary 做 endpoint 审计。`cache_new` 没有统一的逐 query Trap onset，因此这里只能检验成功轨迹沿途概率和最终 outcome 的相关性，不能把最终失败当成 `H=2` Trap 标签。固定 query、同任务排序以及纯 query-clock 是强制负控。

时序约束实验进一步从 client `state/actions/sim_state` 事后重建物理 onset，并在 37 个开发任务上按成功 episode 误报预算选择阈值，3 个后完成任务保持留出。物理量从不进入概率或报警：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=himoe-vla_trap/code \
python himoe-vla_trap/code/analyze_timing_constrained_probability_alarm.py \
  --config himoe-vla_trap/configs/timing_constrained_probability_alarm.json \
  --output himoe-vla_trap/results/trainfree_trap_probability/timing_constrained_alarm \
  --rebuild-physical
```

首次运行使用 `--rebuild-physical`；物理标签完整写盘后可省略。主指标分别要求首次报警不晚于 onset，以及更严格地位于 `[-2,0]` query。每个阈值都比较开发集同等或更低成功误报率的绝对 query 时钟。该 onset 是 query-boundary kinematic proxy，不覆盖所有语义、接触或 belief-state Trap。

## 17. 复现 cache_new MoE 健康不变量报警

该实验不读取旧概率表。每条轨迹由 q0--q3 自身路由建立基线，q4 起进行严格因果判断；32 个任务的成功轨迹只用于确认整条 episode 误报不超过 1% 的阈值，8 个 suite-stratified SHA256 任务留出。留出预测压缩文件写盘并记录 SHA256 后，脚本才打开留出 `summaries.json` 翻 success/failure。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python himoe-vla_trap/code/evaluate_moe_invariant_alarm_cache_new.py \
  --config himoe-vla_trap/configs/moe_invariant_alarm_cache_new.json \
  --output himoe-vla_trap/results/moe_invariant_alarm_cache_new \
  --workers 4

python himoe-vla_trap/code/validate_moe_invariant_alarm_cache_new.py
```

逐 query detector 只接收 HB full-softmax MoE routing。任务路径只用于预先固定 task split 和输出行身份，不进入计算；动作、物理状态、reward、未来长度与结果均被排除。既有物理 onset 表只在预测揭盲后作二次时序审计。完整定义与负结果见 [MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md](MOE_INVARIANT_ALARM_CACHE_NEW_ZH.md)。

## 18. 可复现性边界

旧 init-7 seed 20260830 在 A100 上失败，但本次 H20 上于 375 actions 成功，说明跨硬件浮点差异会改变长期闭环结果。归档同时保留该成功 trunk，没有为了得到 Trap 而删除它。离线统计可确定性复算；闭环 rollout 的逐动作结果只承诺在相同软件、硬件和 checkpoint 条件下复现。
