# 第二轮：功能 MoE 的无训练配对检验

2026-09-06。下列设计在计算本轮结果前固定。使用历史探索缓存，候选方向来自已有 pilot，不称为新的盲测。

## A. 首次动作执行前

数据为 `analysis/moe-state-impact/compact` 的四个任务，共 2,048 集。每个任务 16 个初始状态、32 个噪声分支，全部取 q0。三个不同 checkpoint 分别属于 Goal、Long、Spatial；不跨 checkpoint 混合统计参考或比较向量坐标。

使用已有 routed、shared、pre-MoE hidden 的 64 维每 token 固定随机投影；不读取 next-state delta。它们是同一首次 query 的配对表征，hidden 是 MoE 输入，不能称为 SAFE 原版的末层 hidden。专家输出由 fp16 历史 hidden、记录的 dispatch 和权重重建，不声称运行时数值完全相同。

四层为 2/5/12/15，保留 action tokens，比较 flow d0、d9、全部 flow 均值。共享与 routed 的投影种子和维度相同。路由用对应层、flow、token 的完整概率；行为对照为原始 q0 的机器人状态和按 checkpoint std 缩放的完整动作 chunk。

固定直接分数：低 routed authority、低归一化专家 disagreement、低 cancellation、高 routed/shared cosine、低 router entropy、低 top1/top2 margin、低 token/expert MI。每个分数同样比较 d0/d9/flow mean。方向继承已有研究，不根据本轮 AUC 翻转。

固定距离评分：每维使用参考 median 和 `max(1.4826*MAD, 1e-6)`；比较平均绝对标准偏差和到参考中最近 5 个样本的平均 RMS 距离。后者仅保存无标签样本、做距离查询，不拟合投影、分类器、概率头或组合权重。相同算法应用于 routed/shared/hidden、路由、四个功能标量，以及机器人状态/动作对照。

四折初始状态留出为主设置：按排序后 init 的序号 mod 4。测试折 f，校准折 (f+1) mod 4，另外两折为统计参考。三者完全分离，每个任务每折分别 128/128/256 集。新噪声留出为辅助设置，用排序后 noise seed 的序号分成四折，角色同上；此设置允许初始状态重叠。

阈值仅来自校准折全部无标签样本，预算 1/3/5/10%，3% 为主。校准样本不进入 median/MAD 或 kNN 参考。报告 pooled AUC、任务宏 AUC、同一 task/init 内的宏 AUC，以及 recall/FPR/覆盖率。初始姿态仅能区分 init，故同 init 的姿态分数应为平票。

主表征比较固定为 `routed_d9__knn5` 对 `hidden_d9__knn5` 与 `shared_d9__knn5`；主标量为已有方向的 `cosine_high__flowmean`。其余方法完整报告，最优候选只作描述。

## B. 固定在线 query

同一 Long 任务的 512 集全部取 q34，使用已有 296 维 pre-MoE hidden 描述、配对路由、当前机器人状态及动作。复用 A 的初始状态三方划分和两种距离评分。没有 q34 专家实际输出，不伪造这组对照。

q34 是预先固定的绝对时刻，所有 512 集仍有观测；这个任务与时刻已有历史研究，不作为新发现。最终失败与物理事件时序分开解释。

## C. 功能事件 pilot

沿用 24 个 init 匹配 pair、-4/-2 两个静止 proxy 前时刻和已有恢复容差，不改样本选择。恢复不合格时整对排除。主功能标量沿用最后四层、全部 flow、全部 action tokens；与 routing、当前动作幅度、过去四个 query 的末端位移作配对比较。

另用同一无标签距离评分比较功能四标量、routing、当前姿态和动作。每次排除整个 init/pair 的两条分支及两个 lead 作为参考，同 lead 内取其他 init 的全部事件/对照，不用标签选择健康参考。不将该刻意平衡的病例对照样本校准为实际部署 FPR。

按 task/run/episode 连接原始物理记录，区分静止 proxy、任意目标 release、尚失败 goal 的 release、脱手相关 failed goal 的 release。所有这些 release 均为首次观测到的抓持丢失，不当作已证明的不可逆故障时刻。

用同一 pair 内排名准确率和 pooled AUC 报告功能量/行为读数；bootstrap 以 init 为单位联合保留两个 lead。增加功能量相对动作的配对差值区间，不能仅靠单独的高 AUC 声称功能增量。

## 核验

预测、阈值与源文件哈希先写入独立封存文件，然后评价结果标签。数据容器中存在历史标签，但评分函数不接受标签。固定 query 只读取其当时状态/动作；未来 delta、剩余长度、最终成功均不作为输入。

所有本轮方法均无参数训练，不等同无需参考数据。此次不训练 SAFE/CFN，不拟合成功概率，不从被动检测推断干预收益。
