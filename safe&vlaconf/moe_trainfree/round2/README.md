# 第二轮：专家输出与配对对照

结果：[REPORT_ZH.md](../results/round2/REPORT_ZH.md)。设计：[PROTOCOL_ZH.md](PROTOCOL_ZH.md)。

在冻结 HiMoE-VLA 的已有缓存上比较 routed/shared/pre-MoE hidden、路由、动作及姿态的固定无训练评分。q0 使用四任务 2,048 集；q34 使用 Long 512 集；事件 pilot 使用每个 lead 23 对合格样本。结果标签不参与统计参考、分数方向和阈值拟合。

从仓库根目录运行测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q 'safe&vlaconf/moe_trainfree'
```

完整复现使用尚不存在的输出目录，依次运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/round2/seal.py' --output 'safe&vlaconf/moe_trainfree/results/round2_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/round2/evaluate.py' --input 'safe&vlaconf/moe_trainfree/results/round2_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/round2/inference_audit.py' --input 'safe&vlaconf/moe_trainfree/results/round2_reproduction'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 'safe&vlaconf/moe_trainfree/round2/report.py' --input 'safe&vlaconf/moe_trainfree/results/round2_reproduction'
```

依赖现有 NumPy、pandas、SciPy、scikit-learn、Zarr、matplotlib、pytest。无需加载策略或使用 GPU。

输入为 `himoe-route-capture/analysis/moe-state-impact/compact`、`analysis/t34-closure/hidden_t34.npz`、对应原始 route/client 缓存，以及 `analysis_moe_execution_signals/rich_event_*`。仅评价阶段使用 `VLA_MUI_HUB/physical-failure-labels/results` 的结果标签和物理事件。精确路径与哈希列在封存文件。

`inference_audit.py` 是首批指标出来后增加的解释性核查，保留共同噪声种子的依赖，并重复随机控制。它不改变已封存的任何方法或阈值。数据与候选方向已有历史探索，本轮不是全新确认集。
