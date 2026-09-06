# HB Front/Back Routing Dynamics

独立验证 HiMoE-VLA 的八个 HB 路由层是否具有稳定的前后层结构，以及这种
结构是否对 trajectory-level deadline risk 和 v4 报警后的 extend/intervene
判断提供增量。

本实验不训练分类器，不使用 q-2，也不在特征提取阶段加载 outcome。层名称为：

```text
front: L2, L3, L4, L5
back:  L12, L13, L14, L15
```

## 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-hb-front-back-0905
CUDA_VISIBLE_DEVICES=6,7 python experiments/extract_layer_graphs_gpu.py \
  --output results/layer_graphs --batch 256 --resume
python experiments/analyze_front_back.py \
  --output results/analysis --bootstrap 5000
pytest -q
```

`extract_layer_graphs_gpu.py` 读取完整的
`hb_router_probs[8,10,11,32]`，输出逐层 flow 路径、收敛速度、
state-conditioned action 图、有效秩、expert load 秩和跨 query 图响应。

`analyze_front_back.py` 固定比较单层、前四层、后四层和全层。阈值来自同任务
reference 的经验分位数，external outcome 不参与阈值或方向选择。

完整数字和限制见 `REPORT_ZH.md`，机器可读结果位于 `results/analysis/`。
