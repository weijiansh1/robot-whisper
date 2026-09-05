# HiMoE-VLA Dynamic MoE Assurance v9

这是一个独立、无训练的全路由动态实验。它直接流式读取原始
`hb_router_probs[N,8,10,11,32]`，不复制原始 Zarr，不训练 classifier，
也不把路由分数冒充成机器人成功概率。

## 本版读取了什么

- 8 个 MoE 层：`L2,L3,L4,L5,L12,L13,L14,L15`
- 每个 query 的 10 个 flow step
- 1 个 state token 和 10 个 action token
- 32 个 expert 的完整 soft routing distribution
- 45 个任务、18,560 个 episode、305,030 个 query

最终每个 query 保存 2,460 个动态轴：2,065 个 query 内 flow 动态轴和
395 个跨 query 动态轴。原始张量不被复制，压缩后的动态 profiles 约 1.5 GB。

## 复现

```bash
cd /home/jovyan/work/himoe-vla/moe-dynamic-assurance-v9-0905

CUDA_VISIBLE_DEVICES=6,7 \
  python experiments/build_dynamic_profiles_gpu.py --batch 1024

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/evaluate_dynamic_axes.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/evaluate_dynamic_risk.py

python experiments/seal_results.py
python experiments/seal_results.py --verify
pytest -q
```

若构建被中断，可加 `--resume`；已有 profile 会先验证 schema 和 shape。默认使用
ZIP store，减少 CPU 压缩瓶颈；若磁盘更重要，可加 `--compressed`。

## 在线计算

```python
from pathlib import Path
import sys

sys.path.insert(0, "dynamic")
from features import compute_dynamic_history
from scoring import DynamicScoreReference

# router_history: [Q, 8, 10, 11, 32]，Q 最多保留最近 9 个 query 即可
profile = compute_dynamic_history(router_history[-9:])
reference = DynamicScoreReference.load(
    Path("results/dynamic_evaluation/score_reference.npz")
)
scores = reference.score(profile["features"])
latest_loop_evidence = scores["predeclared_loop"][-1]
latest_static_evidence = scores["predeclared_static"][-1]
```

`scores` 是经验分位证据，不是 outcome probability。当前未见任务压力测试中的
static 阈值为 `0.9053006172`；部署前仍应按目标域重新做 episode-level calibration。

## 关键文件

- `REPORT_ZH.md`：结论、v8 对比和边界说明
- `dynamic/features.py`：完整 flow/query 动态画像
- `dynamic/scoring.py`：封存参考上的无训练运行时打分
- `results/dynamic_profiles/build_summary.json`：双 GPU 构建清单和逐文件哈希
- `results/dynamic_evaluation/summary.json`：2,460 轴扫描及跨语料结果
- `results/dynamic_evaluation/score_reference.npz`：运行时经验分位参考
- `results/dynamic_risk/summary.json`：episode-level 风险校准
- `results/final_summary.json`：机器可读最终结论
- `results/sealed_manifest.json`：完整产物哈希

## 语义边界

```text
dynamic routing axes -> evidence score -> calibrated alarm
                                           != outcome probability
```

真实 `P(outcome | snapshot, intervention)` 仍必须由同一物理 snapshot 的重复 rollout
计数给出；本目录没有用标签拟合一个概率头来填补这项缺失。
