# SAFE 思路的无训练 MoE 实验

2026-09-06。第三轮，在计算本轮结果前固定此协议。

## 目标与约束

学习 SAFE 的完整轨迹失败检测、成功轨迹阈值校准和未见任务测试。
冻结策略，不训练 MLP、LSTM、表示投影或概率头。允许参考集使用最终成功/失败标签；
这叫没有新增参数训练，不叫 label-free。已有数据和部分候选曾被探索，本轮不是盲测。

## 数据

使用 VLA_MUI_HUB/cache_new 中完整的 right-50x8-20260903 (A) 与
right-50x8b-20260903 (B)，各 40 个 LIBERO 任务、每任务 400 集。
每个任务 50 个初态，每初态 8 个 flow-noise 种子。逐来源检查 checkpoint、
episode_id、control_step、summaries、client state/actions 对齐。
不同 checkpoint 分开建立参考、校准和预测，绝不直接比较不同模型的专家坐标。
完整 hidden 和真实专家输出当前不在这两批缓存中；不将路由冒充专家实际输出。

## 划分

对四个 suite 分别进行 3 次随机 7 seen / 3 unseen 任务划分，种子
20260907、20260908、20260909。这里 unseen 是相对检测器参考和校准。
每次以独立、无标签随机排列初态 ID：前 30 个用于 A 的参考，之后 10 个用于
A 的校准，最后 10 个用于 B 的 seen 测试。B 的 unseen 任务全部测试。
参考、校准、seen 测试的 task/init 均无重叠；A/B 的噪声种子无重叠。
同一测试样本在重复划分中可能出现多次，不能算成独立新增轨迹。

## 固定表征与评分

每次 query 取 HB 的 8 层、最后 flow、10 个 action token 的 32 路概率。
softmax 重新按行归一化，不从低精度概率重建 dispatch。

- load: 每层 action-token 平均概率的平方根，共 256 维。
- stats: 每层 token entropy / log(32)、top1-top2 margin、token Jensen-Shannon
  divergence / log(32)、state-action Hellinger distance，共 32 维。
- history: stats 拼接每层 mobility 相对 q1..q4 固定均值的对数比，共 40 维；
  q0..q4 不评分，基准无效则保留不评分。
- behavior: 当前预测 action chunk 的平均平移、转动范数、夹爪均值与标准差，
  共 4 维。动作使用模型既有 normalization_action_std。只读当前预测动作。
- 单标量：后四层低熵/高熵、低 token JS、相对自身初期的路由冻结、
  当前动作平移幅度低、过去最多四个 query 的末端位移低。
- 控制：clock、固定种子的独立 random。

load 用其固定几何，其余向量用参考集 median/MAD 归一化。
每条参考轨迹至多取 8 个等间隔观测，每类至多 4096 点，按固定种子下采样。
距离为最近 5 点的平均 RMS 欧氏距离。主评分是
`d(z, success) - d(z, failure)`。另比较 stats 的 success-only 距离，
检验标签参考的作用。不存在失败参考时显式报错，不隐式换方法。
每个基础读数分别使用 current 和 cumsum；clock/random 仅 current。
主方法提前固定为 stats 的 contrast kNN + cumsum；不根据测试结果更换主方法。
所有累计值只使用已观测前缀，报警发生在当前 query 动作执行前。

## 统计校准

参考成功轨迹给出每个绝对 query 的 median 和 MAD；少于 20 个有限观察时，
沿用最近已支持时刻的统计。scale 下界为全局 MAD 的 0.1 倍和 1e-6。
若早期没有足够观察，使用全局统计。此参考形状不使用校准和测试标签。
对独立校准成功轨迹计算整条轨迹最大标准化分数，使用
ceil((n+1)(1-alpha)) 的顺序统计量；秩大于 n 时阈值为正无穷。
严格大于阈值才报警。alpha = 0.01/0.03/0.05/0.10/0.15/0.20，主点 0.05。
该时间带借鉴 functional split conformal；不是 SAFE adaptive modulation 的逐项复现。
另报告以 task/init 为组、对组内成功轨迹取最大值的校准敏感性。
共同噪声、任务聚类与 unseen 任务分布偏移使名义 FPR 保证不能直接照搬。
组校准也不恢复未见任务的交换性，实际误报必须实测。

## 评价和图表

预测、阈值、源文件哈希封存后，再打开 B 的结果标签。
主 AUC 是同任务统一截断长度内的 max-so-far，截断为该任务 B 的最短轨迹长度，
只用于离线评价；另列完整轨迹 AUC、固定绝对 q 的前缀 AUC。
报告每任务与每 suite 的 AUC/AP、同 init AUC、任务宏平均，避免把跨任务难度当作信息。
报警报告所有成功轨迹的曾报警率、失败 recall、balanced accuracy、首次报警时间；
T-det=(first+1)/length，漏报记 1。所有误报计入，不用提前量过滤成功误报。
输出 alpha-accuracy-time 曲线、固定 query 风险演化、对应成功与失败的轨迹例子。
物理失败时刻只作事后分析，对应失败 goal 的 release 与正常放置分开。
数据量消融将参考点上限降至每类 512 和 1024，保持相同校准与测试。
所有重复、方法和工作点均保留，报告主方法和明确标记的描述性候选。

## 核验

核验来源和全部行对齐、初态/任务隔离、校准样本与阈值秩、前缀重放一致性、
在线监测和离线首次报警一致性、参考标签翻转的作用以及测试标签置换不改变预测。
记录提取和评分开销；不把包含磁盘读取的离线用时称为部署推理延迟。
