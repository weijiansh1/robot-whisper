# SAFE 思路与 v7 的无训练实验

结果：[第三轮报告](../results/round3_safe/REPORT_ZH.md)。
使用完整轨迹、成功轨迹校准、未见任务评价；不训练新增预测模型。
原始设计、看到结果后的距离实验和用户要求的 v7 复用分别保留，不能混作预注册盲测。

从仓库根目录运行，重复实验使用新的输出目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/extract.py' --output 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/run.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/evaluate.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/verify.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/followup.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/evaluate.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction/followup'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/v7_adapter.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/evaluate.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction/v7'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/evaluate_v7_budget.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction/v7'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/verify_extensions.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/safe_protocol/report.py' --input 'safe&vlaconf/moe_trainfree/results/safe_reproduction'
```

依赖 NumPy、pandas、SciPy、scikit-learn、Zarr、matplotlib、pytest。无需加载 VLA 或使用 GPU。
`run.py --resume` 只恢复代码哈希未变化且已有预测哈希正确的未完成运行。
输出目录的 `features` 来自原始路由，后续子实验用相对符号链接复用该缓存。

测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q 'safe&vlaconf/moe_trainfree' moe-v7-0905/tests/test_intrinsic_guard.py moe-v7-0905/tests/test_unlabeled_budget.py
```

`monitor.py` 提供原主方法的逐 query 接口，接收原始 `hb_router_probs`，检查 checkpoint，
支持 reset，超出已校准的 52-query 范围会明确拒绝。参考 profile 在 `profiles/*.npz`。
v7 原始机制直接使用其已有 `IntrinsicGuardMonitor`，所需四个 profile 标量保存在
`v7/profiles/*.json` 的 `budget_profiles` 中。

success-only 距离不需要失败参考，但需要识别成功样本。v7 无标签预算版本连校准也不使用结果标签。
它们都不训练新增模型，标签需求和参考数据需求应分别报告。
