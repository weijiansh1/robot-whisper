# Hook 与采集策略审计

日期：2026-09-08。范围：现有模型服务、旧 recorder / 分叉脚本、v7 报警接口，以及本目录 Pro/Plus 采样计划与运行器。

后续更新：以下问题保留为旧入口的审计记录。新采集路径已接通完整 HB/AS 概率、实际 native/effective ID 和合并权重、冻结 v7、持久快照、主轨迹提交后的 C0/C1/T1 与有界分块，不采 hidden。四个 suite 的 120 个唯一预检变体通过主轨迹及完整 C0 审计；Long 的独立 12 条 MPS 配对预检和 48 条 C1/T1 后缀也已通过。T1 的实际 gate 返回值、400 个替换槽位、原槽位权重保留和后续撤销均检查。七卡正式 Long 首批 **120 条主轨迹、120 条 C0、312 条 C1/T1 后缀全部通过审计**，原 35 个在线服务保留。见 [当前实验记录](EXPERIMENT_RUN.zh.md)。

结论：现有动作 token 的 top-4 hook 在本次固定输入测试中不改变动作；但 **完整路由 -> v7 报警 -> 可恢复快照 -> 主轨迹结束后配对分支 -> 分块提交** 尚未贯通，当前不能把 baseline 运行器当作正式 MoE-Control collector。以下问题按影响排列。本轮新增诊断和报告，未修改线上服务、正式快照模块或采样清单，未采集模型 hidden，未使用物理 GPU 6。

## 1. 需要先处理的问题

### P1：当前采集入口没有运行 v7，线上 top-4 数据也不足以作为其输入

[run_benchmarks.py](/home/jovyan/work/himoe-vla/moe-trap-control/benchmarks/run_benchmarks.py:277) 明确发送 `routing/capture=False`，query 日志只有动作、推理耗时和 noise 哈希；没有报警、报警快照和分支调度。[experiment_config.json](/home/jovyan/work/himoe-vla/moe-trap-control/design/experiment_config.json:3) 明确标记为设计配置。

[当前 hook](/home/jovyan/work/himoe-vla/himoe-libero-wrist-fix/src/himoe_libero_bridge/policies.py:328) 从 gate 的真实返回值取专家 ID 和 combine weights，只保留最后 10 个 action token，结果为 `[denoise=10, layer=8, action=10, topk=4]`。它不返回完整 HB 概率。

[v7 输入检查](/home/jovyan/work/himoe-vla/moe-v7-0905/method/intrinsic_guard_monitor.py:301) 要求 `[layer=8, denoise=10, suffix=11, expert=32]`。这里 11 包括一个 state token，但当前特征实际使用 action token 1..10；state token 路由并不是 hidden。根本缺失是完整 32 专家概率，top-4 归一化权重无法恢复其余 28 个专家的概率，也无法恢复原始概率质量。

接通时应保留现有 v7 输入契约，明确轴顺序和层号，返回完整 HB 概率并验证归一化、有限值和一次请求的完整 denoise 数。不能把 top-4 补零后称为原 v7。HB 全概率以 float16 保存仅约 55 KiB/query，不需要落盘 hidden；HB/AS recorder 中间使用 gate 输入计算概率，也不等于需要保存 hidden。

### P1：旧快照漏掉 RNG 和观测采样状态，不能直接保证配对恢复

[save_full_state](/home/jovyan/work/himoe-vla/himoe-route-capture/branch_snapshot.py:90) 已保存 float64 sim state、控制器、插值器、环境时钟/终止标志和 `qacc_warmstart`。缺少环境/NumPy RNG、真实策略输入、robosuite observable 的缓存与采样计时，以及重新创建环境所需的完整身份信息。

[restore_full_state](/home/jovyan/work/himoe-vla/himoe-route-capture/branch_snapshot.py:115) 调用 `set_init_state`，会强制更新 observable；这个更新本身推进 observable 的内部计时。[Plus 的恢复接口](/home/jovyan/work/himoe-vla/himoe-vla-cache/libero-extensions/LIBERO-plus/libero/libero/envs/env_wrapper.py:383) 还绕过了 [step 中的图像噪声处理](/home/jovyan/work/himoe-vla/himoe-vla-cache/libero-extensions/LIBERO-plus/libero/libero/envs/env_wrapper.py:285)，因此不能直接将其返回图像作为报警点的原始输入。

真实模拟器诊断：Pro Goal Semantic、Plus Goal Sensor Noise `_noise_40` 各一个早期快照，重放三个固定动作，使用 CPU OSMesa，无模型请求。以下为第三步 RGB 的最大通道差，范围 0..255：

| 恢复方式 | Pro | Plus |
|---|---:|---:|
| 原快照函数 | 41 | 156 |
| 额外恢复 NumPy RNG | 40 | 4 |
| 额外恢复 RNG、observable 缓存和计时 | 0 | 0 |
| 关闭原环境，新建环境，再恢复 RNG、observable 状态 | 0 | 0 |

原函数重放时，Pro 图像平均通道差仅 0.0029，约 0.074% 像素有差；Plus 平均差 40.31，约 99.68% 像素有差。对应 sim state 最大差约 `1e-16`，所以仅检查 sim state 会漏掉观测不一致。立即调用旧恢复函数所得图像与原输入的最大差分别为 13 / 163。

诊断补回的 observable 字段是 `_time_since_last_sample`、`_current_delay`、`_current_observed_value`、`_sampled`，另保存环境 `_obs_cache`。这是定位问题的对照，**尚未成为生产快照修复**，也不证明接触、终止状态或完整 rollout 都能复现。正式实现还应验证 active wrapper 的其他有状态组件；首次分支输入直接用保存的实际观测，后续观测从保存的 RNG 和采样状态继续。

证据：[快照实测](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_snapshot_audit.json)。

### P1：完整 recorder 没有保存干预后实际执行的 combine weights

[旧 recorder](/home/jovyan/work/himoe-vla/himoe-route-capture/himoe_router_recorder.py:210) 读取了 `topk_weight`，却只记录重新计算的原生概率与执行 ID；[读取器](/home/jovyan/work/himoe-vla/himoe-route-capture/himoe_route_store.py:286) 再将这些选中概率归一化为权重。这个推导对原生 gate 成立，但对修改 ID 或权重的 intervention 不成立。[服务中的 pin](/home/jovyan/work/himoe-vla/himoe-route-capture/serve_with_recorder.py:482) 正好在 recorder 前执行。

用真实 `HiMoERouteRecorder` 配合 CPU mock gate 复现了计划中的“替换最弱槽位专家、保留槽位权重”：未干预时误差为 0；替换后 ID 记录正确，但最后槽位实际权重为 `0.0320586`，读取器会给出 `0.0120376`，最大绝对误差 `0.020021`。

必须直接保存 gate 实际返回的 `effective_ids/effective_weights`，同时记录该 gate 输入下的 `native_probs/native_ids`，需要逐项审计时也保存原生返回权重。保持 ID 与权重的槽位对应关系，不能单独排序 ID。当前轻量 top-4 hook 已直接复制真实返回权重，这个问题针对完整 recorder，不应混淆两套实现。

证据：[权重复现](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_weight_audit.json)。

### P1：旧请求级 recorder 模式会强制启用 hidden

[serve_with_recorder.py](/home/jovyan/work/himoe-vla/himoe-route-capture/serve_with_recorder.py:357) 在 `--request-gated-capture` 下无条件执行 `args.store_hidden=True`。省略 `--store-hidden` 也不能关闭。它还 [禁止同时启用 `--return-full-probs`](/home/jovyan/work/himoe-vla/himoe-route-capture/serve_with_recorder.py:353)，而在线 v7 需要读取本次完整 HB 概率。

因此不能直接将该旧模式接到新采集器。应使请求级 capture 支持无 hidden 的 schema，并在请求内返回完整 HB 概率或计算报警。现有服务本次没有走这个旧模式；四个在线模型的诊断响应均无 hidden。

### P2：旧分叉脚本的时序可参考，但报警、恢复持久化和 chunk 策略均不符合新实验

[triggered_fork_collect.py](/home/jovyan/work/himoe-vla/himoe-route-capture/triggered_fork_collect.py:304) 确实在推理前保存状态，推理后计算报警，并在 [主轨迹结束之后](/home/jovyan/work/himoe-vla/himoe-route-capture/triggered_fork_collect.py:335) 才启动分支。这个时序符合用户要求。

但它使用自己的 [top-4 Jaccard detector](/home/jovyan/work/himoe-vla/himoe-route-capture/triggered_fork_collect.py:108)，不是冻结的 v7；每个 query 都在 `rewind` 中保留快照和观测，报警快照也只在内存中，进程退出后不能按报警点恢复。[fan_out](/home/jovyan/work/himoe-vla/himoe-route-capture/triggered_fork_collect.py:217) 展开的是 K 条新噪声后缀，没有本实验的 C0 原噪声重放与 C1/T1 配对路由干预，整条分支数组和视频帧累积后才落盘。

[旧 Zarr writer](/home/jovyan/work/himoe-vla/himoe-route-capture/himoe_route_store.py:235) 已有 durable rows、flush 和回滚机制，可复用；但按服务累计行数的 chunk 不等于按主轨迹组织的 prefix/suffix 和提交协议。新采集必须显式保存 `main_id/event_id/branch_id/query_id/attempt_id`，并让分支依赖已经提交的 `main_complete`。报警快照需要在主轨迹继续运行时就持久化，不能等结束后才尝试重建。

### P2：采样清单尚未机器标注 Pro 中未改变场景的对照项

[清单生成器](/home/jovyan/work/himoe-vla/moe-trap-control/moe_control_plan.py:196) 保留 200 个 Pro 注册条目，其中 5 个 Environment 条目的 BDDL 实际未变。[安装报告](/home/jovyan/work/himoe-vla/moe-trap-control/benchmarks/READINESS.json) 和实验文档已经说明，但 CSV 没有 `analysis_role/is_ood` 等字段，后续直接按 `category=Environment` 汇总会将其混入 OOD。

| 统计口径 | Pro 初筛 | Plus 初筛 | 初筛合计 | Pro 覆盖 | Plus 覆盖 | 覆盖合计 |
|---|---:|---:|---:|---:|---:|---:|
| 包含这 5 项对照 | 1,000 | 2,800 | 3,800 | 4,000 | 10,030 | 14,030 |
| OOD 主分析 | 975 | 2,800 | 3,775 | 3,900 | 10,030 | 13,930 |

建议保留对照记录并增加机器可读角色，不静默删除或将安装 smoke 算作正式 rollout。实查 14,030 个 `main_id` 全部唯一，40 个原始任务的跨 benchmark fold 无冲突；Plus 初筛 28 个 `suite/category` 格均为 100 条，纳入概率合法。全部记录仍为 `planned_pending_preflight`。

证据：[采样审计](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_sampling_audit.json)。

## 2. 应采用的采集流程

1. query `q` 推理前暂存状态、真实双相机输入、proprio、有效指令、RNG、observable 状态、控制器和报警器历史。未报警时只保留这一份临时快照。
2. 原策略完成推理，采集完整 HB 概率与执行路由，用冻结 v7 更新报警。首次 `false -> true` 时持久化推理前快照，并引用原 query 的路由、分数和机制。主轨迹继续执行原动作，不受分支采样或 RNG 操作影响。
3. 主轨迹到成功或原 horizon 后，完成所有 block 和终止记录提交，再把事件放入可执行分支队列。chunk 边界不能重置环境、报警历史或随机数序列。
4. 恢复报警起点，执行 C0 原噪声重放一次、C1 新噪声四次、T1 对应新噪声加路由干预四次。C1/T1 的 flow noise 与环境噪声均配对。T1 只在首个分支 query 的 HB 12..15、action token 1..10、全部 10 次 denoise 替换一个专家，保留原槽位权重，随后恢复原路由。
5. 所有分支只引用主轨迹前缀 `[0,q)`，从 `q` 保存自己的后缀；剩余时限统一为 `H-s`，其中 `s` 是报警 query 前已执行环境步数。恢复后不额外 settle。保存预测动作与 `executed_action_count`，不能把未执行的动作当作真实轨迹。
6. 存储采用 8 query/block、每 worker 最多 2 个待写 block；末块允许不足 8。通过校验和与原子提交登记完成状态，失败分支保留 `pending/error`。不采 hidden，不存全程视频；报警点图像属于恢复所需的实际输入，应保留。

这里“动作 chunk”通常是一次推理的 10 个环境动作，“存储 block”是 8 次推理。分块控制缓存峰值；磁盘节省主要来自前缀引用和不存 hidden/全程图像。分块不会减少常驻模型显存。

C0 必须先用于验证恢复；T1 对 C1 的差异才隔离路由干预的增量效果。所有首次报警均按既定规则入样，包括最终成功的主轨迹，才能同时估计 rescue、harm 和净收益。预算限制下抽报警状态需记录纳入概率；置信区间按原始任务/主轨迹聚类，不能把同一状态的 9 条后缀当作 9 个独立主样本。

## 3. 容量与运行约束

[现有预算](/home/jovyan/work/himoe-vla/moe-trap-control/design/budget_current_matrix.json) 使用 25 GiB 配额、集群 2 query/s、每主轨迹平均 25 query、每后缀平均 15 query、30% 报警、每展开状态 9 后缀，压缩后按 48 KiB/query 估计。按这些假设，24 小时可完成约 **2,247 条主轨迹及其全部计划后缀**；3,800 条初筛约 13.3 GiB / 40.6 小时；14,030 条覆盖加最多 600 个展开状态约 24.3 GiB / 69.4 小时。

这些是条件预算，2 query/s 来自此前模型请求压测，尚未包含已接通的 Pro/Plus 模拟器、完整 recorder、恢复和落盘全流程实测；48/80 KiB 也分别是压缩/原始规划值。因此不能将以上数量称为已确认的采集上限，更不能由模型已占显存推断 GPU 会满功率。

正式 rollout 前应完成原方案的 P0：覆盖 Pro/Plus 扰动类别、接触与终止状态、新环境恢复、C0 完整后缀；检查双相机输入、动作、路由、sim state、最终 success，以及 T1 的层/token/query 范围和撤销。随后用实际 query/s、压缩字节、快照大小、主/后缀长度和报警率重算预算。未通过恢复的配置先修复，不能混入配对干预效果估计。

本次线上探针只请求物理 GPU 0 的四个已加载模型，其余模拟器诊断使用 CPU OSMesa。检查结束时物理 GPU 6 仍为 1 MiB / 0% 利用率。现有 Pro/Plus 运行器硬性排除 GPU 6；[通用 matrix 启动器](/home/jovyan/work/himoe-vla/online-servers/serve_model_matrix.py:215) 的默认列表仍包含 6，未来启动须沿用显式 `--gpus 0,1,2,3,4,5,7`，不得使用该默认值。当前服务未重启。

## 4. 本次验证与产物

| 验证 | 结果与范围 |
|---|---|
| 四个真实 LIBERO 模型各四次固定输入请求 | 共 16 query；capture 开、重复请求、capture 关闭后，动作均逐项一致；重复 route ID/weight 一致 |
| 线上 route 数据 | `[10,8,10,4]`；ID 在 0..31；权重和最大误差 `1.19e-7`；无 full HB probs、无 hidden 响应 |
| 真实 recorder + CPU mock intervention | 复现实际 combine weights 与记录推导不一致 |
| Pro / Plus 真实模拟器 | 两个早期快照、各三个固定动作；补回 RNG/observable 后旧实例与新实例均恢复一致图像，sim 最大差约 `1e-16` |
| 现有 CPU 测试 | 36 passed；涵盖 v7 prefix 因果性、流式/批量一致性、mock 路由 patch、候选存储回滚/恢复、快照 codec 与提交描述符 |
| CSV 清单 | 14,030 唯一主轨迹，无原始任务 fold 冲突；确认 100 条覆盖 / 25 条初筛属于未改变场景的 Pro 对照 |

诊断代码：[audit_hook_collection.py](/home/jovyan/work/himoe-vla/moe-trap-control/audit_hook_collection.py)。其 `live` 模式只调用现有服务，`weights` 用 CPU mock，`snapshots` 用已安装 Pro/Plus 的 CPU 模拟器，`sampling` 只检查清单。模拟器对照没有运行策略，因此不能据此宣称 C0 全后缀验证已通过。

结果文件：[线上 hook](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_live_audit.json)、[权重](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_weight_audit.json)、[快照](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_snapshot_audit.json)、[采样](/home/jovyan/work/himoe-vla/moe-trap-control/design/hook_sampling_audit.json)。

本轮新增了这些诊断产物；生产 hook、recorder、快照和分支采集仍有上述待修项。测试通过的旧模块可以复用，但这些测试不覆盖完整新 collector，也不替代 P0。
