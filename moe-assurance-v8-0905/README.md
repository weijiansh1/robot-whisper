# HiMoE-VLA MoE Assurance v8

日期：2026-09-05

这个独立目录逐一实现并审计了九条 MoE assurance 路线。核心约束是把三类量分开：

```text
routing evidence -> mode belief -> fork-calibrated outcome assurance
```

- `routing evidence` 是路由承诺度、flow 曲率、token-expert 图、复现距离等内部读数；
- `mode belief` 是拟合时序模型对 `H/C/PL/L/PS/S` 的后验；
- `outcome assurance` 只在相同物理 snapshot 存在重复 rollout/fork 计数时生成。

因此，任何 gate entropy、HSMM posterior 或 conformal alarm 都不会被命名为机器人成功概率。完整结论见 [REPORT_ZH.md](REPORT_ZH.md)。

## 数据与范围

原始资产只读并原地引用，没有复制：

- `../analysis_moe_phenotype/features`：45 个任务、18,560 episodes、305,030 query rows；
- `../analysis_moe_phenotype/events`：独立物理事件标签；
- `../analysis_committor/out/c1_cells.csv`：2,080 个 q0 committor cells；
- `../himoe-route-capture/runs/fork-pilot-n32-client`：20 snapshots x 32 noise continuations；
- `../himoe-route-capture/analysis/expert-activation-hidden-matched`：80 expert-output pools；
- `../trap-recovery-depth-20260904`：未完成的 recovery collection。

可复现缓存提供 L12-L15、10 action tokens、32 experts 的 final-flow soft routing，以及 9 个 flow transitions 和 8 个 query lags。v8 保留 40 个聚合轴、4 x 13 个 layer-token-expert 图轴、9 个 flow 轴和 8 个 recurrence 轴，共 109 个候选轴。它不是整个模型所有 MoE 激活的无损副本。

## 复现

环境版本记录在 `requirements.txt`。从本目录执行：

```bash
cd /home/jovyan/work/himoe-vla/moe-assurance-v8-0905

# outcome-free profile；物理 GPU 6、7 分摊 45 个任务
CUDA_VISIBLE_DEVICES=6,7 python experiments/build_graph_profiles_gpu.py

# 路线 1-4，以及对 109 个 MoE 轴的发现/外部确认
python experiments/evaluate_routes_1_4.py
python experiments/analyze_full_moe_axes.py

# 路线 5-6：同 snapshot noise forks 与 expert outputs
python experiments/evaluate_routes_5_6.py

# 路线 7-9：HSMM、真实 fork posterior、risk calibration
python experiments/evaluate_route_7.py
python experiments/evaluate_route_8.py
python experiments/evaluate_route_9.py

# 导出一个严格分层的 query state 示例
python experiments/export_query_assurance.py \
  --corpus main16x32 \
  --suite libero_goal \
  --task open_the_top_drawer_and_put_the_bowl_inside \
  --episode 0 --query 0 \
  --output results/example_assurance_q0.json

# 汇总、哈希封存和测试
python experiments/seal_results.py
pytest -q
```

除 profile 构建外，其余步骤读取已压缩特征，默认在 CPU 上运行。profile 构建的两张 H20 峰值各 129.47 MiB；没有为了提高 GPU 利用率而复制数据或启动无意义训练。

## 产物

- `results/profiles/build_summary.json`：双卡、输入范围和 45 个 profile 哈希；
- `results/full_moe_axes/summary.json`：109 轴扫描与 discovery-locked external confirmation；
- `results/routes_1_4/summary.json`：commitment/coherence/graph/manifold；
- `results/routes_5_6/summary.json`：perturbation 与 expert disagreement；
- `results/route_7/summary.json`：HSMM mode belief；
- `results/route_8/assurance_tensor.csv`：唯一允许解释为 outcome probability 的表；
- `results/route_9/summary.json`：episode/scene-cluster risk calibration；
- `results/example_assurance_q0.json`：完整 `MoEAssuranceState` 示例；
- `results/final_summary.json`：九路线机器可读结论；
- `results/sealed_manifest.json`：代码、报告和结果 SHA-256 清单。

`results/profiles`、逐任务 scores 和 posteriors 是派生缓存；磁盘紧张时可删除并按上面的命令重建。整个独立目录约 125 MiB，未复制原始 rollout。
