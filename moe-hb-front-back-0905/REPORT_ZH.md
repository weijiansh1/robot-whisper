# HB 路由前后层结构与 Trap 识别

## 结论

这次扩大数据以后，可以确认两件不同的事。

第一，HB 前层和后层确实都有稳定结构，但分工不同：

- 前层对 flow 和新 observation 的绝对变化更敏感；
- 后层整体变化更小、更接近 state token，但去掉共同 state 成分后，action token
  之间的相对关系反而更分化、更稳定；
- 该结构在两批 50x8 seed 和一批 16x32 数据中同方向复现。

第二，这种结构对 trap 判断有用，但不能简单改成“只看后层”或“八层全部平均”：

- 后层单独看 lock 会提高召回，同时约翻倍误报；
- 前层单独看 lock 精度高，但漏掉一半以上风险；
- v4 的合理之处是用全层一致低响应确认 lock，再用 L5 的持续高响应补充 switching；
- 报警后判断 Extend/Intervene 时，应同时观察“前层重新响应”和“后层相对结构重新展开”。

本实验没有训练分类器，没有使用 q-2。所有动态量只使用当前及历史 query；
outcome 只在分数构造完成后用于评估。

## 数据规模

| 数据 | episodes | tasks | valid queries | 用途 |
|---|---:|---:|---:|---|
| development 50x8 | 14,800 | 37 | 221,781 | 阈值参考与评估 |
| extra reference 50x8 | 1,200 | 3 | 31,941 | 仅补齐参考任务 |
| external 50x8b | 15,600 | 39 | 248,255 | 跨 seed 评估 |
| legacy 16x32 | 2,560 | 5 | 51,308 | 第三采样设计复现 |
| 合计 | 34,160 | 40 个不同任务 | 553,285 | 逐层完整 HB 路由 |

其中 32,960 条有 original-horizon success/failure 标签；30,400 条还有 +10
continuation 标签。每个 query 使用完整的 `hb_router_probs[8,10,11,32]`。

## 前后层结构是否真实存在

以下比值先在每个任务内计算，再对任务取中位数。大于 1 表示预期方向成立。

| 结构量 | reference 50x8 | external 50x8b | legacy 16x32 | 方向一致任务 |
|---|---:|---:|---:|---:|
| 前/后：跨 query 条件图响应 | 7.35 | 7.41 | 6.38 | 40/40, 39/39, 5/5 |
| 后/前：action 相对边分化 | 6.17 | 6.27 | 5.90 | 40/40, 39/39, 5/5 |
| 前/后：state-orthogonal 能量 | 5.37 | 5.40 | 5.36 | 40/40, 39/39, 5/5 |
| 后/前：条件图有效秩 | 1.31 | 1.31 | 1.25 | 40/40, 39/39, 5/5 |
| 前/后：flow 路由路径长度 | 1.34 | 1.33 | 1.34 | 40/40, 39/39, 5/5 |

external 上任务 bootstrap 95% 区间分别为：

```text
前/后 query response:       [6.88, 7.67]
后/前 relative edge spread: [5.76, 7.06]
前/后 conditional energy:   [4.77, 5.99]
后/前 effective rank:       [1.287, 1.317]
前/后 flow path:            [1.300, 1.365]
```

这不是仅由总体均值造成的。相同 45 条 action-token 边的平均模板在两批 50x8
之间，partial graph pooled Spearman 为 0.9995，任务内中位数为 0.9984；
50x8 与 16x32 的对应数字为 0.9990 和 0.9986。

原始 action routing 的 token 共识本来就很高，约为 0.997--0.998。后层又更贴近
state routing，state-action Hellinger affinity 约为 0.983--0.986。关键结构藏在
这个共同方向之外：减去 state-aligned 分量以后，后层剩余能量虽小，但相对
token topology 更丰富，而且跨 seed、跨采样设计稳定。因此它不像独立数值噪声。

同快照多噪声的独立实验也给出相同方向：前层 action-conditional candidate
spread 在 flow 中收缩到 0.562 倍，后层为 1.006 倍；action-partial 分别为
0.399 和 1.357。即同一扰动先在前层被压缩，再在后层转成相对 action 结构。

## 对整条轨迹的 Stage 1 判断

所有比较都固定使用 v4 的窗口、持续长度和 outcome-blind 同任务分位数，仅替换
参与统计的层。external 15,600 条结果如下：

| 规则 | TP | FP | 风险召回 | 精度 | timely FPR |
|---|---:|---:|---:|---:|---:|
| 前四层 median lock | 240 | 36 | 42.55% | 86.96% | 0.239% |
| 后四层 median lock | 409 | 146 | 72.52% | 73.69% | 0.971% |
| 全八层 median lock | 382 | 76 | 67.73% | 83.41% | 0.505% |
| L5 sustained switching | 34 | 5 | 6.03% | 87.18% | 0.033% |
| v4：全层 lock OR L5 switching | 410 | 81 | 72.70% | 83.50% | 0.539% |

所以后层确实更敏感于 lock，但不能直接替换全层：它多抓 27 个风险，同时多报
70 个及时成功。全层一致性相当于一个误报抑制条件。

16x32 第三语料也是同样取舍：

| 规则 | TP / failure | FP | 召回 | 精度 |
|---|---:|---:|---:|---:|
| 前四层 lock | 76 / 307 | 1 | 24.76% | 98.70% |
| 后四层 lock | 235 / 307 | 16 | 76.55% | 93.63% |
| 全层 lock / v4 | 229 / 307 | 13 | 74.59% | 94.63% |

三批有标签数据合计 32,960 条，冻结 v4 共检测到 1,011/1,358 个 original-horizon
风险，描述性合并召回 74.45%，精度 86.26%，及时成功误报率 0.509%。三个 cohort
的定义和任务构成不同，因此这个合并数不是新的独立泛化估计。

## 报警后 Extend 还是 Intervene

在首次 v4 报警处记录基线，再比较原 horizon 时的逐层变化。external 中有 410
条被 v4 抓到的真实风险，其中 27 条 +10 内成功、383 条仍失败。

以“恢复后数值升高”为正方向：

| 报警到 horizon 的变化 | AUC | AP |
|---|---:|---:|
| 前层条件图绝对响应 | 0.659 | 0.166 |
| L4 条件图绝对响应 | 0.691 | 0.165 |
| L14 条件图绝对响应 | 0.673 | 0.142 |
| 后层 partial topology 响应 | 0.656 | 0.175 |
| 后层条件图有效秩恢复 | 0.652 | 0.201 |
| 八层条件图响应平均 | 0.657 | 0.168 |

八层平均的 95% row bootstrap 区间为 `[0.523, 0.787]`，有效样本 409；阳性率
只有 6.59%，因此这仍是小阳性样本结果。

动态过程比终点更重要。八层平均条件图响应的 AUC 为：

```text
alarm +1: 0.380
alarm +2: 0.415
alarm +4: 0.403
alarm +8: 0.514
horizon:  0.657
```

不同 offset 的可用轨迹数不同，不能把它当作严格单调曲线。但方向很清楚：
late-success 在报警后最初仍比 persistent failure 更安静，不应在报警瞬间就决定
Extend；真正的重新响应接近 deadline 才出现。

之前把前四层作为一个联合向量计算的 state-conditioned graph release 仍更强：
development/external AUC 为 0.740/0.734。加入本次后层 topology 后变成
0.732/0.741；加入后层 rank 后变成 0.679/0.742。external 的小幅增长没有在
development 同方向出现，因此不能声称升级成功。逐层信息增强了解释，但当前
还没有可靠超越联合图基线。

## 实际含义

更合理的在线状态机是：

```text
正常：前层随 observation 适度变化，后层保持相对控制结构
  |
  +-- 全层持续低响应 --------------------------> Lock alarm
  |
  +-- L5 持续高响应 ----------------------------> Switching alarm

Alarm -> WATCH
  |
  +-- 前层绝对响应恢复 + 后层 topology/rank 恢复 -> Extend
  |
  +-- 全层继续冻结，或只有无结构乱切 ----------> Intervene
```

这里“前层吸收噪声、后层形成结构”的价值，不是额外堆两个 entropy，而是给
响应赋予位置和含义：前层变化是输入反馈通道，后层相对图是控制结构通道。
只有两者的传递关系失常并持续，才更像 trap。

## 限制

- external 8B 已在前序研究中被查看，不是全新盲测；本次没有用它拟合阈值，
  但探索性组合仍可能受重复观察影响。
- 16x32 只有 5 个任务，且没有 +10 continuation，不能验证 Extend/Intervene。
- 平均 token-edge 模板可能部分来自固定 action-token 位置先验；稳定结构不自动
  等于因果控制结构。
- +10 late-success 阳性只有 development 30、external 41；进入 v4 Stage 2 的
  分别只有 20 和 27。
- 当前只使用 router probabilities，没有 expert output 或环境变化量，因而只能
  测“内部是否响应”，不能直接判断这次响应是否物理上正确。
- AUC 是排序能力，不是成功概率。要输出概率仍需更多 continuation 或 fork
  outcome 做校准。

## 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-hb-front-back-0905
CUDA_VISIBLE_DEVICES=6,7 python experiments/extract_layer_graphs_gpu.py \
  --output results/layer_graphs --batch 256 --resume
python experiments/analyze_front_back.py \
  --output results/analysis --bootstrap 5000
pytest -q
```

机器可读结果见 `results/layer_graphs/build_summary.json` 和
`results/analysis/summary.json`。
