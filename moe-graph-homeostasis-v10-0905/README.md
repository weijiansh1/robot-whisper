# HiMoE-VLA MoE Graph Homeostasis v10

独立、无训练的 MoE 图动力学实验。它使用完整的
`[8 layers, 10 flow steps, 11 tokens, 32 experts]` router probability，回答两个问题：

1. 同一物理 snapshot 的不同 flow noise 会在 MoE 图中收缩还是分岔？
2. flow 内稳定性与 query 间响应性怎样区分 healthy、loop 和 static？

详细数字、解释和限制见 [`REPORT_ZH.md`](REPORT_ZH.md)。

## 目录

```text
graph/structure.py                         图视角与因果动力学特征
experiments/build_structural_profiles_gpu.py  45 任务双 GPU profile 构建
experiments/analyze_fork_contraction_gpu.py   两套 same-snapshot fork 实验
experiments/evaluate_phase_portrait.py        matched q-2 相图与跨语料评估
experiments/seal_results.py                   结果审计、摘要与 SHA-256 封存
tests/                                        单元和结果测试
results/fork_contraction/                     fork 数字与收缩图
results/phase_portrait/                       AUC、相图和 matched point
results/structural_profiles/                  45 个紧凑 profile
```

## 复现

在本目录执行：

```bash
python -m pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=6,7 \
  python experiments/build_structural_profiles_gpu.py --batch 512 --resume

CUDA_VISIBLE_DEVICES=6,7 \
  python experiments/analyze_fork_contraction_gpu.py

python experiments/evaluate_phase_portrait.py
python experiments/seal_results.py
pytest -q
```

profile 构建不读取结果标签，也不拟合权重。`main_ranked_*` 是另外报告的
label-based discovery：只用 `main16x32` 选择轴和方向，再在 `grid50x8` 固定确认；
它不是训练，但也不是先验固定结果。

## 语义边界

- `rho = D(flow=9) / D(flow=0)` 衡量独立噪声 ensemble 的全局收缩，不是局部
  Lyapunov 指数。
- 单 rollout 的 settling、curvature 和 query response 是 routing evidence，不是成功概率。
- static 的强结果是持续停滞的确认信号；由于 onset 定义本身需要连续窗口，不能把
  q-2 AUC 解释成提前两步预测随机命运。
- 没有 Snapshot-Fork outcome counts 的校准，本实验不输出 `P(success)`。
